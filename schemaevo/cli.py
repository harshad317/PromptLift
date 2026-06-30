from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import yaml

from schemaevo.benchmarks.openai_closed_loop import (
    OpenAIClosedLoopBenchmarkConfig,
    run_openai_closed_loop_benchmark,
)
from schemaevo.benchmarks.openai_fixed_pool import (
    OpenAIFixedPoolBenchmarkConfig,
    build_openai_benchmark_program,
    make_proposer,
    run_openai_fixed_pool_benchmark,
)
from schemaevo.benchmarks.readiness import check_benchmark_readiness, check_fixed_pool_split_readiness
from schemaevo.datasets.scorers import hotpotqa_exact_match, hotpotqa_f1, hover_label_accuracy
from schemaevo.datasets.hotpotqa import load_hotpotqa_examples
from schemaevo.datasets.hover import load_hover_examples
from schemaevo.examples.toy_multihop import (
    build_toy_program,
    make_toy_examples,
    make_toy_traces,
    toy_scorer,
)
from schemaevo.experiments.budget_pareto import build_budget_pareto_report
from schemaevo.experiments.causal_pilot import build_causal_pilot_report
from schemaevo.experiments.deployment_invariance import build_fixed_pool_deployment_report
from schemaevo.experiments.external_prompt_optimizer import ExternalPromptOptimizer
from schemaevo.experiments.transfer import run_openai_cross_model_schema_transfer
from schemaevo.experiments.composability import run_prompt_optimizer_then_schemaevo
from schemaevo.optimizers.fixed_pool_schema import FixedPoolConfig, run_fixed_pool_schema_mvp
from schemaevo.optimizers.schema_evo import SchemaEvoConfig, schema_evo_optimize
from schemaevo.schemas.proposer import HeuristicTraceSchemaProposer, OpenAISchemaProposer, SchemaProposer


def _add_fixed_pool_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/toy_schemaevo.yaml")
    parser.add_argument("--dataset", choices=("hotpotqa", "hover"), required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--smoke", default=None)
    parser.add_argument("--heldout", default=None)
    parser.add_argument("--out", default="artifacts/openai_fixed_pool")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--progress", choices=("auto", "rich", "tqdm", "none"), default=None)
    parser.add_argument("--use-tiktoken-costing", action="store_true")
    parser.add_argument("--input-price-per-million", type=float, default=None)
    parser.add_argument("--output-price-per-million", type=float, default=None)
    parser.add_argument("--cached-input-price-per-million", type=float, default=0.0)
    parser.add_argument("--price-source-date", default="cli")
    parser.add_argument("--max-target-task-calls", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-dollar-cost", type=float, default=None)
    parser.add_argument("--allow-unready", action="store_true")
    parser.add_argument("--allow-contextless", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schemaevo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    toy = subparsers.add_parser("run-toy-mvp", help="Run local fixed-pool SchemaEvo MVP demo.")
    toy.add_argument("--config", default="configs/toy_schemaevo.yaml")
    toy.add_argument("--out", default="artifacts/toy_mvp")
    toy.add_argument("--workers", type=int, default=None, help="Process workers for independent candidate evaluation.")
    toy.add_argument(
        "--progress",
        choices=("auto", "rich", "tqdm", "none"),
        default=None,
        help="Progress renderer. Writes to stderr so stdout stays machine-readable.",
    )

    closed_loop = subparsers.add_parser(
        "run-toy-closed-loop",
        help="Run local closed-loop SchemaEvo demo without external APIs.",
    )
    closed_loop.add_argument("--config", default="configs/toy_schemaevo.yaml")
    closed_loop.add_argument("--out", default="artifacts/toy_closed_loop")
    closed_loop.add_argument(
        "--progress",
        choices=("auto", "rich", "tqdm", "none"),
        default=None,
        help="Progress renderer. Writes to stderr so stdout stays machine-readable.",
    )

    readiness = subparsers.add_parser(
        "check-benchmark-readiness",
        help="Check credentials, packages, and local data needed for real HotpotQA/HoVer runs.",
    )
    readiness.add_argument("--hotpotqa", default=None, help="Path to local HotpotQA JSON/JSONL data.")
    readiness.add_argument("--hover", default=None, help="Path to local HoVer JSON/JSONL data.")
    readiness.add_argument("--strict", action="store_true", help="Exit nonzero when readiness fails.")
    readiness.add_argument(
        "--allow-contextless",
        action="store_true",
        help="Permit question/answer-only smoke files. Do not use for benchmark claims.",
    )
    readiness.add_argument("--inspect-limit", type=int, default=200)

    real_fixed = subparsers.add_parser(
        "run-openai-fixed-pool",
        help="Run fixed-pool SchemaEvo on local HotpotQA/HoVer files with OpenAI module runners.",
    )
    real_fixed.add_argument("--config", default="configs/toy_schemaevo.yaml")
    real_fixed.add_argument("--dataset", choices=("hotpotqa", "hover"), required=True)
    real_fixed.add_argument("--train", required=True)
    real_fixed.add_argument("--selection", required=True)
    real_fixed.add_argument("--confirmation", required=True)
    real_fixed.add_argument("--smoke", default=None)
    real_fixed.add_argument("--heldout", default=None)
    real_fixed.add_argument("--out", default="artifacts/openai_fixed_pool")
    real_fixed.add_argument("--model", default=None)
    real_fixed.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process workers for independent candidate evaluation. Budget-capped runs stay serial.",
    )
    real_fixed.add_argument(
        "--progress",
        choices=("auto", "rich", "tqdm", "none"),
        default=None,
        help="Progress renderer. Writes to stderr so stdout stays machine-readable.",
    )
    real_fixed.add_argument(
        "--use-tiktoken-costing",
        action="store_true",
        help="Use tiktoken for token accounting when available.",
    )
    real_fixed.add_argument(
        "--input-price-per-million",
        type=float,
        default=None,
        help="Input token price in dollars per 1M tokens for the selected model.",
    )
    real_fixed.add_argument(
        "--output-price-per-million",
        type=float,
        default=None,
        help="Output token price in dollars per 1M tokens for the selected model.",
    )
    real_fixed.add_argument(
        "--cached-input-price-per-million",
        type=float,
        default=0.0,
        help="Cached input token price in dollars per 1M tokens for the selected model.",
    )
    real_fixed.add_argument(
        "--price-source-date",
        default="cli",
        help="Free-form date/source label stored with CLI-provided prices.",
    )
    real_fixed.add_argument(
        "--max-target-task-calls",
        type=int,
        default=None,
        help="Maximum target-program LLM calls for the fixed-pool run.",
    )
    real_fixed.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Maximum prompt tokens for the fixed-pool run.",
    )
    real_fixed.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens for the fixed-pool run.",
    )
    real_fixed.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Maximum prompt plus completion tokens for the fixed-pool run.",
    )
    real_fixed.add_argument(
        "--max-dollar-cost",
        type=float,
        default=None,
        help="Maximum estimated dollar cost for the fixed-pool run.",
    )
    real_fixed.add_argument(
        "--allow-unready",
        action="store_true",
        help="Skip readiness failure. Intended only for injected-client tests or dry development.",
    )
    real_fixed.add_argument(
        "--allow-contextless",
        action="store_true",
        help="Permit question/answer-only smoke files. Do not use for benchmark claims.",
    )

    causal_pilot = subparsers.add_parser(
        "run-openai-causal-pilot",
        help="Run the fixed-pool causal pilot and write mask/shuffle go/no-go reports.",
    )
    _add_fixed_pool_data_args(causal_pilot)
    causal_pilot.add_argument("--min-fraction-of-delta", type=float, default=0.5)
    causal_pilot.add_argument("--min-absolute-drop", type=float, default=0.015)

    closed_real = subparsers.add_parser(
        "run-openai-closed-loop",
        help="Run closed-loop SchemaEvo on local HotpotQA/HoVer files with OpenAI module runners.",
    )
    closed_real.add_argument("--config", default="configs/toy_schemaevo.yaml")
    closed_real.add_argument("--dataset", choices=("hotpotqa", "hover"), required=True)
    closed_real.add_argument("--optimizer", required=True)
    closed_real.add_argument("--confirmation", required=True)
    closed_real.add_argument("--heldout", default=None)
    closed_real.add_argument("--out", default="artifacts/openai_closed_loop")
    closed_real.add_argument("--model", default="gpt-4.1-mini")
    closed_real.add_argument("--use-tiktoken-costing", action="store_true")
    closed_real.add_argument("--input-price-per-million", type=float, default=None)
    closed_real.add_argument("--output-price-per-million", type=float, default=None)
    closed_real.add_argument("--cached-input-price-per-million", type=float, default=0.0)
    closed_real.add_argument("--price-source-date", default="cli")
    closed_real.add_argument("--progress", choices=("auto", "rich", "tqdm", "none"), default=None)

    composability = subparsers.add_parser(
        "run-openai-composability",
        help="Run external GEPA/MIPRO-style prompt optimizer, then SchemaEvo, then additive evaluation.",
    )
    composability.add_argument("--config", default="configs/toy_schemaevo.yaml")
    composability.add_argument("--dataset", choices=("hotpotqa", "hover"), required=True)
    composability.add_argument("--schema-optimizer", required=True)
    composability.add_argument("--eval", required=True)
    composability.add_argument("--out", default="artifacts/openai_composability")
    composability.add_argument("--model", default="gpt-4.1-mini")
    composability.add_argument("--prompt-optimizer-name", default="external_prompt_optimizer")
    composability.add_argument("--prompt-optimizer-command", required=True)
    composability.add_argument("--allow-demo-changes", action="store_true")
    composability.add_argument("--use-tiktoken-costing", action="store_true")
    composability.add_argument("--input-price-per-million", type=float, default=None)
    composability.add_argument("--output-price-per-million", type=float, default=None)
    composability.add_argument("--cached-input-price-per-million", type=float, default=0.0)
    composability.add_argument("--price-source-date", default="cli")
    composability.add_argument("--progress", choices=("auto", "rich", "tqdm", "none"), default=None)

    transfer = subparsers.add_parser(
        "run-openai-cross-model-transfer",
        help="Optimize schemas on one model and evaluate the transferred schema on another model.",
    )
    _add_fixed_pool_data_args(transfer)
    transfer.add_argument("--source-model", required=True)
    transfer.add_argument("--target-model", required=True)

    pareto = subparsers.add_parser(
        "write-budget-pareto-report",
        help="Aggregate run summaries into accuracy-vs-budget and Pareto report artifacts.",
    )
    pareto.add_argument("--run", action="append", default=[], help="Method/path pair: name=/path/to/summary.json")
    pareto.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "run-toy-mvp":
        return _run_toy_mvp(
            config_path=Path(args.config),
            out_dir=Path(args.out),
            workers=args.workers,
            progress=args.progress,
        )
    if args.command == "run-toy-closed-loop":
        return _run_toy_closed_loop(
            config_path=Path(args.config),
            out_dir=Path(args.out),
            progress=args.progress,
        )
    if args.command == "check-benchmark-readiness":
        return _check_benchmark_readiness(
            hotpotqa_path=args.hotpotqa,
            hover_path=args.hover,
            strict=bool(args.strict),
            require_context=not bool(args.allow_contextless),
            inspect_limit=int(args.inspect_limit),
        )
    if args.command == "run-openai-fixed-pool":
        return _run_openai_fixed_pool(
            config_path=Path(args.config),
            dataset=args.dataset,
            train_path=Path(args.train),
            selection_path=Path(args.selection),
            confirmation_path=Path(args.confirmation),
            smoke_path=Path(args.smoke) if args.smoke else None,
            heldout_path=Path(args.heldout) if args.heldout else None,
            out_dir=Path(args.out),
            model=args.model,
            workers=args.workers,
            progress=args.progress,
            use_tiktoken_costing=bool(args.use_tiktoken_costing),
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            cached_input_price_per_million=float(args.cached_input_price_per_million),
            price_source_date=str(args.price_source_date),
            max_target_task_calls=args.max_target_task_calls,
            max_prompt_tokens=args.max_prompt_tokens,
            max_completion_tokens=args.max_completion_tokens,
            max_total_tokens=args.max_total_tokens,
            max_dollar_cost=args.max_dollar_cost,
            strict_readiness=not bool(args.allow_unready),
            require_context=not bool(args.allow_contextless),
        )
    if args.command == "run-openai-causal-pilot":
        return _run_openai_causal_pilot(
            config_path=Path(args.config),
            dataset=args.dataset,
            train_path=Path(args.train),
            selection_path=Path(args.selection),
            confirmation_path=Path(args.confirmation),
            smoke_path=Path(args.smoke) if args.smoke else None,
            heldout_path=Path(args.heldout) if args.heldout else None,
            out_dir=Path(args.out),
            model=args.model,
            workers=args.workers,
            progress=args.progress,
            use_tiktoken_costing=bool(args.use_tiktoken_costing),
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            cached_input_price_per_million=float(args.cached_input_price_per_million),
            price_source_date=str(args.price_source_date),
            max_target_task_calls=args.max_target_task_calls,
            max_prompt_tokens=args.max_prompt_tokens,
            max_completion_tokens=args.max_completion_tokens,
            max_total_tokens=args.max_total_tokens,
            max_dollar_cost=args.max_dollar_cost,
            strict_readiness=not bool(args.allow_unready),
            require_context=not bool(args.allow_contextless),
            min_fraction_of_delta=float(args.min_fraction_of_delta),
            min_absolute_drop=float(args.min_absolute_drop),
        )
    if args.command == "run-openai-closed-loop":
        return _run_openai_closed_loop(
            config_path=Path(args.config),
            dataset=args.dataset,
            optimizer_path=Path(args.optimizer),
            confirmation_path=Path(args.confirmation),
            heldout_path=Path(args.heldout) if args.heldout else None,
            out_dir=Path(args.out),
            model=str(args.model),
            use_tiktoken_costing=bool(args.use_tiktoken_costing),
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            cached_input_price_per_million=float(args.cached_input_price_per_million),
            price_source_date=str(args.price_source_date),
            progress=args.progress,
        )
    if args.command == "run-openai-composability":
        return _run_openai_composability(
            config_path=Path(args.config),
            dataset=args.dataset,
            schema_optimizer_path=Path(args.schema_optimizer),
            eval_path=Path(args.eval),
            out_dir=Path(args.out),
            model=str(args.model),
            prompt_optimizer_name=str(args.prompt_optimizer_name),
            prompt_optimizer_command=str(args.prompt_optimizer_command),
            allow_demo_changes=bool(args.allow_demo_changes),
            use_tiktoken_costing=bool(args.use_tiktoken_costing),
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            cached_input_price_per_million=float(args.cached_input_price_per_million),
            price_source_date=str(args.price_source_date),
            progress=args.progress,
        )
    if args.command == "run-openai-cross-model-transfer":
        return _run_openai_cross_model_transfer(
            config_path=Path(args.config),
            dataset=args.dataset,
            train_path=Path(args.train),
            selection_path=Path(args.selection),
            confirmation_path=Path(args.confirmation),
            smoke_path=Path(args.smoke) if args.smoke else None,
            heldout_path=Path(args.heldout) if args.heldout else None,
            out_dir=Path(args.out),
            source_model=str(args.source_model),
            target_model=str(args.target_model),
            workers=args.workers,
            progress=args.progress,
            use_tiktoken_costing=bool(args.use_tiktoken_costing),
            input_price_per_million=args.input_price_per_million,
            output_price_per_million=args.output_price_per_million,
            cached_input_price_per_million=float(args.cached_input_price_per_million),
            price_source_date=str(args.price_source_date),
            max_target_task_calls=args.max_target_task_calls,
            max_prompt_tokens=args.max_prompt_tokens,
            max_completion_tokens=args.max_completion_tokens,
            max_total_tokens=args.max_total_tokens,
            max_dollar_cost=args.max_dollar_cost,
            strict_readiness=not bool(args.allow_unready),
            require_context=not bool(args.allow_contextless),
        )
    if args.command == "write-budget-pareto-report":
        return _write_budget_pareto_report(run_specs=args.run, out_dir=Path(args.out))
    raise ValueError(args.command)


def _run_toy_mvp(
    *,
    config_path: Path,
    out_dir: Path,
    workers: int | None = None,
    progress: str | None = None,
) -> int:
    raw = _load_yaml(config_path)
    fixed_pool_config = FixedPoolConfig(**raw.get("fixed_pool", {}))
    fixed_pool_config = _apply_runtime_overrides(
        fixed_pool_config,
        workers=workers,
        progress=progress,
    )
    smoke_n = int(raw.get("smoke_examples", 4))
    selection_n = int(raw.get("selection_examples", 12))
    confirmation_n = int(raw.get("confirmation_examples", 20))
    result = run_fixed_pool_schema_mvp(
        base_program=build_toy_program(),
        train_traces=make_toy_traces(),
        smoke_examples=make_toy_examples("validation_smoke", smoke_n),
        selection_examples=make_toy_examples("validation_selection", selection_n),
        confirmation_examples=make_toy_examples("validation_confirmation", confirmation_n),
        scorer=toy_scorer,
        config=fixed_pool_config,
        proposer=_make_proposer(raw),
        heldout_test_examples=make_toy_examples(
            "final_test",
            int(raw.get("heldout_test_examples", 0)),
        )
        if int(raw.get("heldout_test_examples", 0))
        else (),
        artifact_dir=out_dir,
    )
    print(json.dumps(result.summary(), sort_keys=True, indent=2))
    return 0


def _run_toy_closed_loop(
    *,
    config_path: Path,
    out_dir: Path,
    progress: str | None = None,
) -> int:
    raw = _load_yaml(config_path)
    config = SchemaEvoConfig(**raw.get("closed_loop", {}))
    if progress is not None:
        config = replace(config, progress=progress)
    examples_n = int(raw.get("closed_loop_examples", 24))
    result = schema_evo_optimize(
        base_program=build_toy_program(),
        examples=make_toy_examples("optimizer_validation", examples_n),
        scorer=toy_scorer,
        config=config,
        artifact_dir=out_dir,
    )
    print(json.dumps(result.summary(), sort_keys=True, indent=2))
    return 0


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return loaded


def _check_benchmark_readiness(
    *,
    hotpotqa_path: str | None,
    hover_path: str | None,
    strict: bool,
    require_context: bool = True,
    inspect_limit: int = 200,
) -> int:
    readiness = check_benchmark_readiness(
        hotpotqa_path=hotpotqa_path,
        hover_path=hover_path,
        require_context=require_context,
        inspect_limit=inspect_limit,
    )
    print(json.dumps(readiness.to_dict(), sort_keys=True, indent=2))
    return 1 if strict and not readiness.ready else 0


def _run_openai_fixed_pool(
    *,
    config_path: Path,
    dataset: str,
    train_path: Path,
    selection_path: Path,
    confirmation_path: Path,
    smoke_path: Path | None,
    heldout_path: Path | None,
    out_dir: Path,
    model: str | None,
    workers: int | None,
    progress: str | None,
    use_tiktoken_costing: bool,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    cached_input_price_per_million: float,
    price_source_date: str,
    max_target_task_calls: int | None,
    max_prompt_tokens: int | None,
    max_completion_tokens: int | None,
    max_total_tokens: int | None,
    max_dollar_cost: float | None,
    strict_readiness: bool,
    require_context: bool,
) -> int:
    prepared = _prepare_openai_fixed_pool_run(
        config_path=config_path,
        dataset=dataset,
        train_path=train_path,
        selection_path=selection_path,
        confirmation_path=confirmation_path,
        smoke_path=smoke_path,
        heldout_path=heldout_path,
        model=model,
        workers=workers,
        progress=progress,
        use_tiktoken_costing=use_tiktoken_costing,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        cached_input_price_per_million=cached_input_price_per_million,
        price_source_date=price_source_date,
        max_target_task_calls=max_target_task_calls,
        max_prompt_tokens=max_prompt_tokens,
        max_completion_tokens=max_completion_tokens,
        max_total_tokens=max_total_tokens,
        max_dollar_cost=max_dollar_cost,
        strict_readiness=strict_readiness,
        require_context=require_context,
    )
    if isinstance(prepared, dict):
        print(json.dumps(prepared, sort_keys=True, indent=2))
        return 1
    benchmark_config, fixed_pool_config, proposer = prepared
    result = run_openai_fixed_pool_benchmark(
        benchmark_config=benchmark_config,
        fixed_pool_config=fixed_pool_config,
        proposer=proposer,
        artifact_dir=out_dir,
    )
    print(json.dumps(result.summary(), sort_keys=True, indent=2))
    return 0


def _prepare_openai_fixed_pool_run(
    *,
    config_path: Path,
    dataset: str,
    train_path: Path,
    selection_path: Path,
    confirmation_path: Path,
    smoke_path: Path | None,
    heldout_path: Path | None,
    model: str | None,
    workers: int | None,
    progress: str | None,
    use_tiktoken_costing: bool,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    cached_input_price_per_million: float,
    price_source_date: str,
    max_target_task_calls: int | None,
    max_prompt_tokens: int | None,
    max_completion_tokens: int | None,
    max_total_tokens: int | None,
    max_dollar_cost: float | None,
    strict_readiness: bool,
    require_context: bool,
) -> tuple[OpenAIFixedPoolBenchmarkConfig, FixedPoolConfig, SchemaProposer] | dict[str, Any]:
    _require_existing_paths(train_path, selection_path, confirmation_path, smoke_path, heldout_path)
    readiness = check_fixed_pool_split_readiness(
        dataset=dataset,
        train_path=train_path,
        smoke_path=smoke_path,
        selection_path=selection_path,
        confirmation_path=confirmation_path,
        heldout_path=heldout_path,
        require_context=require_context,
    )
    raw = _load_yaml(config_path)
    fixed_pool_config = FixedPoolConfig(**raw.get("fixed_pool", {}))
    proposer_config = raw.get("proposer", {})
    if not isinstance(proposer_config, dict):
        raise ValueError("proposer config must be a mapping")
    selected_model = model or proposer_config.get("model") or fixed_pool_config.target_model
    fixed_pool_config = replace(
        fixed_pool_config,
        task="HotpotQA" if dataset == "hotpotqa" else "HoVer",
        target_model=str(selected_model),
        use_tiktoken_costing=bool(use_tiktoken_costing or fixed_pool_config.use_tiktoken_costing),
    )
    fixed_pool_config = _apply_runtime_overrides(
        fixed_pool_config,
        workers=workers,
        progress=progress,
    )
    fixed_pool_config = _apply_budget_overrides(
        fixed_pool_config,
        max_target_task_calls=max_target_task_calls,
        max_prompt_tokens=max_prompt_tokens,
        max_completion_tokens=max_completion_tokens,
        max_total_tokens=max_total_tokens,
        max_dollar_cost=max_dollar_cost,
    )
    if input_price_per_million is not None or output_price_per_million is not None:
        model_prices = dict(fixed_pool_config.model_prices)
        model_prices[str(selected_model)] = {
            "input_per_million": float(input_price_per_million or 0.0),
            "output_per_million": float(output_price_per_million or 0.0),
            "cached_input_per_million": float(cached_input_price_per_million),
            "source_date": price_source_date,
        }
        fixed_pool_config = replace(fixed_pool_config, model_prices=model_prices)
    accounting_reasons = _strict_accounting_reasons(
        config=fixed_pool_config,
        selected_model=str(selected_model),
    )
    if strict_readiness and (not readiness.ready or accounting_reasons):
        output = readiness.to_dict()
        output["ready"] = False
        output["accounting"] = {
            "ok": not accounting_reasons,
            "reasons": accounting_reasons,
            "selected_model": str(selected_model),
        }
        output["reasons"] = [*output.get("reasons", []), *accounting_reasons]
        return output
    benchmark_config = OpenAIFixedPoolBenchmarkConfig(
        dataset=dataset,  # type: ignore[arg-type]
        train_path=train_path,
        smoke_path=smoke_path,
        selection_path=selection_path,
        confirmation_path=confirmation_path,
        heldout_path=heldout_path,
        train_limit=_optional_int(raw.get("train_examples")),
        smoke_limit=_optional_int(raw.get("smoke_examples")),
        selection_limit=_optional_int(raw.get("selection_examples")),
        confirmation_limit=_optional_int(raw.get("confirmation_examples")),
        heldout_limit=_optional_int(raw.get("heldout_test_examples")),
        model=str(selected_model),
        temperature=(
            float(proposer_config["target_temperature"])
            if proposer_config.get("target_temperature") is not None
            else None
        ),
        retriever_top_k=int(raw.get("retriever_top_k", 0)),
    )
    proposer = make_proposer(
        str(proposer_config.get("kind", "heuristic")),
        model=str(proposer_config.get("model", "gpt-4.1-mini")),
        temperature=float(proposer_config.get("temperature", 0.7)),
        max_output_tokens=int(proposer_config.get("max_output_tokens", 4096)),
    )
    return benchmark_config, fixed_pool_config, proposer


def _run_openai_causal_pilot(
    *,
    config_path: Path,
    dataset: str,
    train_path: Path,
    selection_path: Path,
    confirmation_path: Path,
    smoke_path: Path | None,
    heldout_path: Path | None,
    out_dir: Path,
    model: str | None,
    workers: int | None,
    progress: str | None,
    use_tiktoken_costing: bool,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    cached_input_price_per_million: float,
    price_source_date: str,
    max_target_task_calls: int | None,
    max_prompt_tokens: int | None,
    max_completion_tokens: int | None,
    max_total_tokens: int | None,
    max_dollar_cost: float | None,
    strict_readiness: bool,
    require_context: bool,
    min_fraction_of_delta: float,
    min_absolute_drop: float,
) -> int:
    prepared = _prepare_openai_fixed_pool_run(
        config_path=config_path,
        dataset=dataset,
        train_path=train_path,
        selection_path=selection_path,
        confirmation_path=confirmation_path,
        smoke_path=smoke_path,
        heldout_path=heldout_path,
        model=model,
        workers=workers,
        progress=progress,
        use_tiktoken_costing=use_tiktoken_costing,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        cached_input_price_per_million=cached_input_price_per_million,
        price_source_date=price_source_date,
        max_target_task_calls=max_target_task_calls,
        max_prompt_tokens=max_prompt_tokens,
        max_completion_tokens=max_completion_tokens,
        max_total_tokens=max_total_tokens,
        max_dollar_cost=max_dollar_cost,
        strict_readiness=strict_readiness,
        require_context=require_context,
    )
    if isinstance(prepared, dict):
        print(json.dumps(prepared, sort_keys=True, indent=2))
        return 1
    benchmark_config, fixed_pool_config, proposer = prepared
    fixed_pool_result = run_openai_fixed_pool_benchmark(
        benchmark_config=benchmark_config,
        fixed_pool_config=fixed_pool_config,
        proposer=proposer,
        artifact_dir=out_dir / "fixed_pool",
    )
    pilot_report = build_causal_pilot_report(
        result=fixed_pool_result,
        dataset=dataset,
        model=benchmark_config.model,
        artifact_dir=out_dir,
        min_fraction_of_delta=min_fraction_of_delta,
        min_absolute_drop=min_absolute_drop,
    )
    deployment_report = build_fixed_pool_deployment_report(
        result=fixed_pool_result,
        artifact_dir=out_dir,
    )
    output = {
        "causal_pilot": pilot_report.summary(),
        "deployment_invariance": deployment_report.summary(),
        "fixed_pool": fixed_pool_result.summary(),
    }
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0 if pilot_report.proceed else 2


def _run_openai_closed_loop(
    *,
    config_path: Path,
    dataset: str,
    optimizer_path: Path,
    confirmation_path: Path,
    heldout_path: Path | None,
    out_dir: Path,
    model: str,
    use_tiktoken_costing: bool,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    cached_input_price_per_million: float,
    price_source_date: str,
    progress: str | None,
) -> int:
    _require_existing_paths(optimizer_path, confirmation_path, heldout_path)
    raw = _load_yaml(config_path)
    schema_config = SchemaEvoConfig(**raw.get("closed_loop", raw.get("schema_evo", {})))
    schema_config = replace(
        schema_config,
        task="HotpotQA" if dataset == "hotpotqa" else "HoVer",
        use_tiktoken_costing=bool(use_tiktoken_costing or schema_config.use_tiktoken_costing),
        progress=progress or schema_config.progress,
    )
    schema_config = _with_cli_price(
        schema_config,
        model=model,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        cached_input_price_per_million=cached_input_price_per_million,
        price_source_date=price_source_date,
    )
    result = run_openai_closed_loop_benchmark(
        benchmark_config=OpenAIClosedLoopBenchmarkConfig(
            dataset=dataset,  # type: ignore[arg-type]
            optimizer_path=optimizer_path,
            confirmation_path=confirmation_path,
            heldout_path=heldout_path,
            optimizer_limit=_optional_int(raw.get("selection_examples")),
            confirmation_limit=_optional_int(raw.get("confirmation_examples")),
            heldout_limit=_optional_int(raw.get("heldout_test_examples")),
            model=model,
            retriever_top_k=int(raw.get("retriever_top_k", 0)),
        ),
        schema_config=schema_config,
        artifact_dir=out_dir,
    )
    print(json.dumps(result.summary(), sort_keys=True, indent=2))
    return 0


def _run_openai_composability(
    *,
    config_path: Path,
    dataset: str,
    schema_optimizer_path: Path,
    eval_path: Path,
    out_dir: Path,
    model: str,
    prompt_optimizer_name: str,
    prompt_optimizer_command: str,
    allow_demo_changes: bool,
    use_tiktoken_costing: bool,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    cached_input_price_per_million: float,
    price_source_date: str,
    progress: str | None,
) -> int:
    _require_existing_paths(schema_optimizer_path, eval_path)
    raw = _load_yaml(config_path)
    schema_config = SchemaEvoConfig(**raw.get("closed_loop", raw.get("schema_evo", {})))
    schema_config = replace(
        schema_config,
        task="HotpotQA" if dataset == "hotpotqa" else "HoVer",
        use_tiktoken_costing=bool(use_tiktoken_costing or schema_config.use_tiktoken_costing),
        progress=progress or schema_config.progress,
    )
    schema_config = _with_cli_price(
        schema_config,
        model=model,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        cached_input_price_per_million=cached_input_price_per_million,
        price_source_date=price_source_date,
    )
    base_program = build_openai_benchmark_program(
        OpenAIFixedPoolBenchmarkConfig(
            dataset=dataset,  # type: ignore[arg-type]
            train_path=schema_optimizer_path,
            selection_path=schema_optimizer_path,
            confirmation_path=eval_path,
            model=model,
            retriever_top_k=int(raw.get("retriever_top_k", 0)),
        )
    )
    optimizer_examples = _load_dataset_examples(
        dataset=dataset,
        path=schema_optimizer_path,
        split="validation_selection",
        limit=_optional_int(raw.get("selection_examples")),
    )
    eval_examples = _load_dataset_examples(
        dataset=dataset,
        path=eval_path,
        split="validation_confirmation",
        limit=_optional_int(raw.get("confirmation_examples")),
    )
    prompt_optimizer = ExternalPromptOptimizer(
        name=prompt_optimizer_name,
        command=prompt_optimizer_command,
        artifact_dir=out_dir / "prompt_optimizer",
        allow_demo_changes=allow_demo_changes,
    )
    result = run_prompt_optimizer_then_schemaevo(
        base_program=base_program,
        prompt_optimizer=prompt_optimizer,
        prompt_eval_examples=eval_examples,
        schema_optimizer_examples=optimizer_examples,
        scorer=_scorer(dataset),
        schema_config=schema_config,
        artifact_dir=out_dir,
    )
    print(json.dumps(result.summary(), sort_keys=True, indent=2))
    return 0


def _run_openai_cross_model_transfer(
    *,
    config_path: Path,
    dataset: str,
    train_path: Path,
    selection_path: Path,
    confirmation_path: Path,
    smoke_path: Path | None,
    heldout_path: Path | None,
    out_dir: Path,
    source_model: str,
    target_model: str,
    workers: int | None,
    progress: str | None,
    use_tiktoken_costing: bool,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    cached_input_price_per_million: float,
    price_source_date: str,
    max_target_task_calls: int | None,
    max_prompt_tokens: int | None,
    max_completion_tokens: int | None,
    max_total_tokens: int | None,
    max_dollar_cost: float | None,
    strict_readiness: bool,
    require_context: bool,
) -> int:
    prepared = _prepare_openai_fixed_pool_run(
        config_path=config_path,
        dataset=dataset,
        train_path=train_path,
        selection_path=selection_path,
        confirmation_path=confirmation_path,
        smoke_path=smoke_path,
        heldout_path=heldout_path,
        model=source_model,
        workers=workers,
        progress=progress,
        use_tiktoken_costing=use_tiktoken_costing,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        cached_input_price_per_million=cached_input_price_per_million,
        price_source_date=price_source_date,
        max_target_task_calls=max_target_task_calls,
        max_prompt_tokens=max_prompt_tokens,
        max_completion_tokens=max_completion_tokens,
        max_total_tokens=max_total_tokens,
        max_dollar_cost=max_dollar_cost,
        strict_readiness=strict_readiness,
        require_context=require_context,
    )
    if isinstance(prepared, dict):
        print(json.dumps(prepared, sort_keys=True, indent=2))
        return 1
    benchmark_config, fixed_pool_config, proposer = prepared
    report = run_openai_cross_model_schema_transfer(
        benchmark_config=benchmark_config,
        source_model=source_model,
        target_model=target_model,
        fixed_pool_config=fixed_pool_config,
        proposer=proposer,
        artifact_dir=out_dir,
    )
    print(json.dumps(report.summary(), sort_keys=True, indent=2))
    return 0


def _write_budget_pareto_report(*, run_specs: list[str], out_dir: Path) -> int:
    runs: dict[str, Path] = {}
    for item in run_specs:
        if "=" not in item:
            raise ValueError("--run must be formatted as name=/path/to/summary.json")
        name, path = item.split("=", 1)
        runs[name] = Path(path)
    if not runs:
        raise ValueError("at least one --run is required")
    report = build_budget_pareto_report(run_paths=runs, artifact_dir=out_dir)
    print(json.dumps(report.summary(), sort_keys=True, indent=2))
    return 0


def _with_cli_price(
    config: SchemaEvoConfig,
    *,
    model: str,
    input_price_per_million: float | None,
    output_price_per_million: float | None,
    cached_input_price_per_million: float,
    price_source_date: str,
) -> SchemaEvoConfig:
    if input_price_per_million is None and output_price_per_million is None:
        return config
    prices = dict(config.model_prices)
    prices[model] = {
        "input_per_million": float(input_price_per_million or 0.0),
        "output_per_million": float(output_price_per_million or 0.0),
        "cached_input_per_million": float(cached_input_price_per_million),
        "source_date": price_source_date,
    }
    return replace(config, model_prices=prices)


def _load_dataset_examples(
    *,
    dataset: str,
    path: Path,
    split: str,
    limit: int | None,
):
    if dataset == "hotpotqa":
        return load_hotpotqa_examples(path, split=split, limit=limit)
    if dataset == "hover":
        return load_hover_examples(path, split=split, limit=limit)
    raise ValueError(f"unsupported dataset: {dataset}")


def _scorer(dataset: str):
    if dataset == "hotpotqa":
        return hotpotqa_f1
    if dataset == "hover":
        return hover_label_accuracy
    raise ValueError(f"unsupported dataset: {dataset}")


def _apply_budget_overrides(
    config: FixedPoolConfig,
    *,
    max_target_task_calls: int | None,
    max_prompt_tokens: int | None,
    max_completion_tokens: int | None,
    max_total_tokens: int | None,
    max_dollar_cost: float | None,
) -> FixedPoolConfig:
    overrides = {
        "max_target_task_calls": max_target_task_calls,
        "max_prompt_tokens": max_prompt_tokens,
        "max_completion_tokens": max_completion_tokens,
        "max_total_tokens": max_total_tokens,
        "max_dollar_cost": max_dollar_cost,
    }
    selected = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **selected) if selected else config


def _apply_runtime_overrides(
    config: FixedPoolConfig,
    *,
    workers: int | None,
    progress: str | None,
) -> FixedPoolConfig:
    overrides: dict[str, Any] = {}
    if workers is not None:
        overrides["workers"] = workers
    if progress is not None:
        overrides["progress"] = progress
    return replace(config, **overrides) if overrides else config


def _strict_accounting_reasons(*, config: FixedPoolConfig, selected_model: str) -> list[str]:
    reasons: list[str] = []
    if not config.use_tiktoken_costing:
        reasons.append("strict OpenAI fixed-pool runs require tiktoken costing")
    raw_price = config.model_prices.get(selected_model)
    if not raw_price:
        reasons.append(f"missing pricing table for selected model {selected_model!r}")
        return reasons
    input_price = float(raw_price.get("input_per_million", 0.0))
    output_price = float(raw_price.get("output_per_million", 0.0))
    source_date = str(raw_price.get("source_date", "")).strip()
    if input_price <= 0:
        reasons.append(f"input price for selected model {selected_model!r} must be > 0")
    if output_price <= 0:
        reasons.append(f"output price for selected model {selected_model!r} must be > 0")
    if not source_date or source_date == "unset":
        reasons.append(f"price source date for selected model {selected_model!r} is required")
    return reasons


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _require_existing_paths(*paths: Path | None) -> None:
    missing = [str(path) for path in paths if path is not None and not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing data paths: {missing}")


def _make_proposer(raw: dict[str, Any]) -> SchemaProposer:
    proposer_config = raw.get("proposer", {})
    if not isinstance(proposer_config, dict):
        raise ValueError("proposer config must be a mapping")
    kind = proposer_config.get("kind", "heuristic")
    if kind == "heuristic":
        return HeuristicTraceSchemaProposer()
    if kind == "openai":
        return OpenAISchemaProposer(
            model=proposer_config.get("model", "gpt-4.1-mini"),
            temperature=float(proposer_config.get("temperature", 0.7)),
            max_output_tokens=int(proposer_config.get("max_output_tokens", 4096)),
        )
    raise ValueError(f"unknown proposer kind: {kind}")


if __name__ == "__main__":
    raise SystemExit(main())
