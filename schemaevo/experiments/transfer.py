from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from schemaevo.benchmarks.openai_fixed_pool import (
    OpenAIFixedPoolBenchmarkConfig,
    build_openai_benchmark_program,
    make_train_traces_from_examples,
    _load_examples,
    _scorer_for_dataset,
)
from schemaevo.eval.cost_ledger import make_cost_meter
from schemaevo.eval.scoring import CandidateEvalResult, evaluate_program
from schemaevo.optimizers.fixed_pool_schema import FixedPoolConfig, FixedPoolResult, run_fixed_pool_schema_mvp
from schemaevo.programs.compile_schema_program import compile_schema_program
from schemaevo.schemas.proposer import SchemaProposer
from schemaevo.schemas.serialization import write_json


@dataclass(frozen=True)
class CrossModelTransferReport:
    dataset: str
    source_model: str
    target_model: str
    transferred_schema_id: str
    source_baseline_mean: float
    source_schema_mean: float
    source_delta: float
    target_baseline_mean: float
    target_schema_mean: float
    target_delta: float
    schema_transfer_retention: float
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return asdict(self)


def run_openai_cross_model_schema_transfer(
    *,
    benchmark_config: OpenAIFixedPoolBenchmarkConfig,
    source_model: str,
    target_model: str,
    fixed_pool_config: FixedPoolConfig,
    artifact_dir: str | Path | None = None,
    proposer: SchemaProposer | None = None,
    client: Any | None = None,
) -> CrossModelTransferReport:
    source_config = _replace_model(benchmark_config, source_model)
    target_config = _replace_model(benchmark_config, target_model)
    artifact_root = Path(artifact_dir) if artifact_dir else None
    source_result = _run_source_schemaevo(
        benchmark_config=source_config,
        fixed_pool_config=fixed_pool_config,
        artifact_dir=artifact_root / "source_schemaevo" if artifact_root else None,
        proposer=proposer,
        client=client,
    )
    target_program = build_openai_benchmark_program(target_config, client=client)
    target_examples = _load_examples(
        dataset=target_config.dataset,
        path=target_config.heldout_path or target_config.confirmation_path,
        split="final_test" if target_config.heldout_path else "validation_confirmation",
        limit=target_config.heldout_limit or target_config.confirmation_limit,
    )
    scorer = _scorer_for_dataset(target_config.dataset)
    cost_meter = make_cost_meter(
        model_prices=fixed_pool_config.model_prices,
        use_tiktoken=fixed_pool_config.use_tiktoken_costing,
    )
    primary_schema = source_result.primary_confirmation_result.schema_id
    schema_by_id = {schema.schema_id: schema for schema in source_result.schema_pool}
    if primary_schema not in schema_by_id:
        raise RuntimeError(f"primary schema {primary_schema!r} not found in source schema pool")
    transferred_program = compile_schema_program(
        base_program=target_program,
        schema=schema_by_id[primary_schema],
        freeze_prompt_text=fixed_pool_config.freeze_prompt_text,
        allow_only_schema_contract_insert=fixed_pool_config.allow_only_schema_contract_insert,
    )
    target_artifact_root = artifact_root / "target_transfer_eval" if artifact_root else None
    target_baseline = evaluate_program(
        program=target_program,
        examples=target_examples,
        scorer=scorer,
        method="cross_model_target_baseline",
        candidate_id="cross_model_target_baseline",
        seed=fixed_pool_config.seed,
        baseline_program=target_program,
        strict_invalid_policy=fixed_pool_config.strict_invalid_policy,
        artifact_dir=target_artifact_root,
        cost_meter=cost_meter,
        run_id="cross_model_target_baseline",
    )
    target_schema = evaluate_program(
        program=transferred_program,
        examples=target_examples,
        scorer=scorer,
        method="cross_model_schema_transfer",
        candidate_id=f"cross_model_schema_transfer_{primary_schema}",
        seed=fixed_pool_config.seed,
        baseline_program=target_program,
        strict_invalid_policy=fixed_pool_config.strict_invalid_policy,
        artifact_dir=target_artifact_root,
        cost_meter=cost_meter,
        run_id=f"cross_model_schema_transfer_{primary_schema}",
    )
    report = _build_report(
        dataset=benchmark_config.dataset,
        source_model=source_model,
        target_model=target_model,
        source_result=source_result,
        target_baseline=target_baseline,
        target_schema=target_schema,
        artifact_root=artifact_root,
    )
    return report


def _run_source_schemaevo(
    *,
    benchmark_config: OpenAIFixedPoolBenchmarkConfig,
    fixed_pool_config: FixedPoolConfig,
    artifact_dir: Path | None,
    proposer: SchemaProposer | None,
    client: Any | None,
) -> FixedPoolResult:
    program = build_openai_benchmark_program(benchmark_config, client=client)
    scorer = _scorer_for_dataset(benchmark_config.dataset)
    train_examples = _load_examples(
        dataset=benchmark_config.dataset,
        path=benchmark_config.train_path,
        split="train",
        limit=benchmark_config.train_limit,
    )
    selection_examples = _load_examples(
        dataset=benchmark_config.dataset,
        path=benchmark_config.selection_path,
        split="validation_selection",
        limit=benchmark_config.selection_limit,
    )
    confirmation_examples = _load_examples(
        dataset=benchmark_config.dataset,
        path=benchmark_config.confirmation_path,
        split="validation_confirmation",
        limit=benchmark_config.confirmation_limit,
    )
    smoke_examples = (
        _load_examples(
            dataset=benchmark_config.dataset,
            path=benchmark_config.smoke_path,
            split="validation_smoke",
            limit=benchmark_config.smoke_limit,
        )
        if benchmark_config.smoke_path
        else ()
    )
    return run_fixed_pool_schema_mvp(
        base_program=program,
        train_traces=make_train_traces_from_examples(dataset=benchmark_config.dataset, examples=train_examples),
        smoke_examples=smoke_examples,
        selection_examples=selection_examples,
        confirmation_examples=confirmation_examples,
        scorer=scorer,
        config=fixed_pool_config,
        proposer=proposer,
        artifact_dir=artifact_dir,
    )


def _build_report(
    *,
    dataset: str,
    source_model: str,
    target_model: str,
    source_result: FixedPoolResult,
    target_baseline: CandidateEvalResult,
    target_schema: CandidateEvalResult,
    artifact_root: Path | None,
) -> CrossModelTransferReport:
    source_delta = (
        source_result.primary_confirmation_result.mean_score
        - source_result.baseline_confirmation_result.mean_score
    )
    target_delta = target_schema.mean_score - target_baseline.mean_score
    retention = target_delta / source_delta if source_delta > 0 else 0.0
    report = CrossModelTransferReport(
        dataset=dataset,
        source_model=source_model,
        target_model=target_model,
        transferred_schema_id=source_result.primary_confirmation_result.schema_id,
        source_baseline_mean=source_result.baseline_confirmation_result.mean_score,
        source_schema_mean=source_result.primary_confirmation_result.mean_score,
        source_delta=source_delta,
        target_baseline_mean=target_baseline.mean_score,
        target_schema_mean=target_schema.mean_score,
        target_delta=target_delta,
        schema_transfer_retention=retention,
        artifacts={},
    )
    if artifact_root:
        artifact_root.mkdir(parents=True, exist_ok=True)
        json_path = write_json(report.summary(), artifact_root / "cross_model_transfer_report.json")
        md_path = artifact_root / "cross_model_transfer_report.md"
        md_path.write_text(_markdown(report), encoding="utf-8")
        report = CrossModelTransferReport(
            dataset=report.dataset,
            source_model=report.source_model,
            target_model=report.target_model,
            transferred_schema_id=report.transferred_schema_id,
            source_baseline_mean=report.source_baseline_mean,
            source_schema_mean=report.source_schema_mean,
            source_delta=report.source_delta,
            target_baseline_mean=report.target_baseline_mean,
            target_schema_mean=report.target_schema_mean,
            target_delta=report.target_delta,
            schema_transfer_retention=report.schema_transfer_retention,
            artifacts={"summary": str(json_path), "markdown": str(md_path)},
        )
        write_json(report.summary(), json_path)
    return report


def _replace_model(config: OpenAIFixedPoolBenchmarkConfig, model: str) -> OpenAIFixedPoolBenchmarkConfig:
    return OpenAIFixedPoolBenchmarkConfig(
        dataset=config.dataset,
        train_path=config.train_path,
        smoke_path=config.smoke_path,
        selection_path=config.selection_path,
        confirmation_path=config.confirmation_path,
        heldout_path=config.heldout_path,
        train_limit=config.train_limit,
        smoke_limit=config.smoke_limit,
        selection_limit=config.selection_limit,
        confirmation_limit=config.confirmation_limit,
        heldout_limit=config.heldout_limit,
        model=model,
        temperature=config.temperature,
        retriever_top_k=config.retriever_top_k,
    )


def _markdown(report: CrossModelTransferReport) -> str:
    return (
        "# Cross-Model Schema Transfer Report\n\n"
        f"Dataset: `{report.dataset}`\n\n"
        f"Source model: `{report.source_model}`\n\n"
        f"Target model: `{report.target_model}`\n\n"
        f"Transferred schema: `{report.transferred_schema_id}`\n\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"| Source delta | {report.source_delta:.6f} |\n"
        f"| Target delta | {report.target_delta:.6f} |\n"
        f"| Schema transfer retention | {report.schema_transfer_retention:.6f} |\n"
    )
