from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from schemaevo.benchmarks.openai_fixed_pool import (
    DatasetName,
    OpenAIFixedPoolBenchmarkConfig,
    _load_examples,
    _scorer_for_dataset,
    build_openai_benchmark_program,
)
from schemaevo.eval.cost_ledger import make_cost_meter
from schemaevo.eval.scoring import CandidateEvalResult, evaluate_program
from schemaevo.optimizers.schema_evo import SchemaEvoConfig, SchemaEvoRunResult, schema_evo_optimize
from schemaevo.schemas.serialization import write_json


@dataclass(frozen=True)
class OpenAIClosedLoopBenchmarkConfig:
    dataset: DatasetName
    optimizer_path: str | Path
    confirmation_path: str | Path
    heldout_path: str | Path | None = None
    optimizer_limit: int | None = None
    confirmation_limit: int | None = None
    heldout_limit: int | None = None
    model: str = "gpt-4.1-mini"
    temperature: float | None = None
    retriever_top_k: int = 0


@dataclass(frozen=True)
class OpenAIClosedLoopBenchmarkResult:
    optimizer_result: SchemaEvoRunResult
    confirmation_baseline: CandidateEvalResult
    confirmation_results: tuple[CandidateEvalResult, ...]
    primary_confirmation: CandidateEvalResult | None
    best_confirmation: CandidateEvalResult | None
    heldout_baseline: CandidateEvalResult | None
    heldout_results: tuple[CandidateEvalResult, ...]
    primary_heldout: CandidateEvalResult | None
    best_heldout: CandidateEvalResult | None
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer_result.summary(),
            "confirmation_baseline_mean": self.confirmation_baseline.mean_score,
            "primary_confirmation_schema_id": (
                self.primary_confirmation.schema_id if self.primary_confirmation else None
            ),
            "primary_confirmation_mean": (
                self.primary_confirmation.mean_score if self.primary_confirmation else None
            ),
            "best_confirmation_mean": self.best_confirmation.mean_score if self.best_confirmation else None,
            "heldout_baseline_mean": self.heldout_baseline.mean_score if self.heldout_baseline else None,
            "primary_heldout_schema_id": self.primary_heldout.schema_id if self.primary_heldout else None,
            "primary_heldout_mean": self.primary_heldout.mean_score if self.primary_heldout else None,
            "best_heldout_mean": self.best_heldout.mean_score if self.best_heldout else None,
            "confirmation_schema_ids": [result.schema_id for result in self.confirmation_results],
            "heldout_schema_ids": [result.schema_id for result in self.heldout_results],
            "artifacts": self.artifacts,
        }


def run_openai_closed_loop_benchmark(
    *,
    benchmark_config: OpenAIClosedLoopBenchmarkConfig,
    schema_config: SchemaEvoConfig,
    artifact_dir: str | Path | None = None,
    client: Any | None = None,
) -> OpenAIClosedLoopBenchmarkResult:
    fixed_program_config = OpenAIFixedPoolBenchmarkConfig(
        dataset=benchmark_config.dataset,
        train_path=benchmark_config.optimizer_path,
        selection_path=benchmark_config.optimizer_path,
        confirmation_path=benchmark_config.confirmation_path,
        heldout_path=benchmark_config.heldout_path,
        model=benchmark_config.model,
        temperature=benchmark_config.temperature,
        retriever_top_k=benchmark_config.retriever_top_k,
    )
    program = build_openai_benchmark_program(fixed_program_config, client=client)
    scorer = _scorer_for_dataset(benchmark_config.dataset)
    optimizer_examples = _load_examples(
        dataset=benchmark_config.dataset,
        path=benchmark_config.optimizer_path,
        split="validation_selection",
        limit=benchmark_config.optimizer_limit,
    )
    confirmation_examples = _load_examples(
        dataset=benchmark_config.dataset,
        path=benchmark_config.confirmation_path,
        split="validation_confirmation",
        limit=benchmark_config.confirmation_limit,
    )
    heldout_examples = (
        _load_examples(
            dataset=benchmark_config.dataset,
            path=benchmark_config.heldout_path,
            split="final_test",
            limit=benchmark_config.heldout_limit,
        )
        if benchmark_config.heldout_path
        else ()
    )
    artifact_root = Path(artifact_dir) if artifact_dir else None
    optimizer_result = schema_evo_optimize(
        base_program=program,
        examples=optimizer_examples,
        scorer=scorer,
        config=schema_config,
        artifact_dir=artifact_root / "optimizer" if artifact_root else None,
    )
    cost_meter = make_cost_meter(
        model_prices=schema_config.model_prices,
        use_tiktoken=schema_config.use_tiktoken_costing,
    )
    confirmation_baseline = evaluate_program(
        program=program,
        examples=confirmation_examples,
        scorer=scorer,
        method="closed_loop_confirmation_baseline",
        candidate_id="closed_loop_confirmation_baseline",
        seed=schema_config.seed,
        baseline_program=program,
        strict_invalid_policy=schema_config.strict_invalid_policy,
        artifact_dir=artifact_root,
        cost_meter=cost_meter,
        run_id="closed_loop_confirmation_baseline",
    )
    confirmation_results = tuple(
        evaluate_program(
            program=record.program,
            examples=confirmation_examples,
            scorer=scorer,
            method="closed_loop_confirmation",
            candidate_id=f"closed_loop_confirmation_{index}",
            seed=schema_config.seed,
            baseline_program=program,
            strict_invalid_policy=schema_config.strict_invalid_policy,
            artifact_dir=artifact_root,
            cost_meter=cost_meter,
            run_id=f"closed_loop_confirmation_{index}_{record.schema.schema_id}",
        )
        for index, record in enumerate(optimizer_result.final_records)
    )
    primary_confirmation = _primary(confirmation_results)
    best_confirmation = primary_confirmation
    heldout_baseline = None
    heldout_results: tuple[CandidateEvalResult, ...] = ()
    primary_heldout = None
    if heldout_examples:
        heldout_baseline = evaluate_program(
            program=program,
            examples=heldout_examples,
            scorer=scorer,
            method="closed_loop_heldout_baseline",
            candidate_id="closed_loop_heldout_baseline",
            seed=schema_config.seed,
            baseline_program=program,
            strict_invalid_policy=schema_config.strict_invalid_policy,
            artifact_dir=artifact_root,
            cost_meter=cost_meter,
            run_id="closed_loop_heldout_baseline",
        )
        heldout_records = optimizer_result.final_records[:1]
        heldout_results = tuple(
            evaluate_program(
                program=record.program,
                examples=heldout_examples,
                scorer=scorer,
                method="closed_loop_heldout",
                candidate_id=f"closed_loop_heldout_{index}",
                seed=schema_config.seed,
                baseline_program=program,
                strict_invalid_policy=schema_config.strict_invalid_policy,
                artifact_dir=artifact_root,
                cost_meter=cost_meter,
                run_id=f"closed_loop_heldout_{index}_{record.schema.schema_id}",
            )
            for index, record in enumerate(heldout_records)
        )
        primary_heldout = _primary(heldout_results)
    result = OpenAIClosedLoopBenchmarkResult(
        optimizer_result=optimizer_result,
        confirmation_baseline=confirmation_baseline,
        confirmation_results=confirmation_results,
        primary_confirmation=primary_confirmation,
        best_confirmation=best_confirmation,
        heldout_baseline=heldout_baseline,
        heldout_results=heldout_results,
        primary_heldout=primary_heldout,
        best_heldout=_best(heldout_results),
        artifacts={},
    )
    artifacts: dict[str, str] = {}
    if artifact_root:
        summary_path = write_json(
            {
                "config": _jsonable_config(benchmark_config),
                "schema_config": asdict(schema_config),
                "summary": result.summary(),
                "confirmation_baseline": confirmation_baseline.to_dict(),
                "confirmation_results": [item.to_dict() for item in confirmation_results],
                "primary_confirmation": (
                    primary_confirmation.to_dict() if primary_confirmation else None
                ),
                "heldout_baseline": heldout_baseline.to_dict() if heldout_baseline else None,
                "heldout_results": [item.to_dict() for item in heldout_results],
                "primary_heldout": primary_heldout.to_dict() if primary_heldout else None,
            },
            artifact_root / "results" / "closed_loop_summary.json",
        )
        artifacts["summary"] = str(summary_path)
        result = OpenAIClosedLoopBenchmarkResult(
            optimizer_result=optimizer_result,
            confirmation_baseline=confirmation_baseline,
            confirmation_results=confirmation_results,
            primary_confirmation=primary_confirmation,
            best_confirmation=best_confirmation,
            heldout_baseline=heldout_baseline,
            heldout_results=heldout_results,
            primary_heldout=primary_heldout,
            best_heldout=_best(heldout_results),
            artifacts=artifacts,
        )
        write_json(result.summary(), artifact_root / "results" / "closed_loop_brief.json")
    return result


def _primary(results: tuple[CandidateEvalResult, ...]) -> CandidateEvalResult | None:
    return results[0] if results else None


def _best(results: tuple[CandidateEvalResult, ...]) -> CandidateEvalResult | None:
    return (
        max(results, key=lambda result: (result.mean_score, -result.invalid_output_rate, result.schema_id))
        if results
        else None
    )


def _jsonable_config(config: OpenAIClosedLoopBenchmarkConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data
