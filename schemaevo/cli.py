from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import yaml

from schemaevo.benchmarks.openai_fixed_pool import (
    OpenAIFixedPoolBenchmarkConfig,
    make_proposer,
    run_openai_fixed_pool_benchmark,
)
from schemaevo.benchmarks.readiness import check_benchmark_readiness
from schemaevo.examples.toy_multihop import (
    build_toy_program,
    make_toy_examples,
    make_toy_traces,
    toy_scorer,
)
from schemaevo.optimizers.fixed_pool_schema import FixedPoolConfig, run_fixed_pool_schema_mvp
from schemaevo.optimizers.schema_evo import SchemaEvoConfig, schema_evo_optimize
from schemaevo.schemas.proposer import HeuristicTraceSchemaProposer, OpenAISchemaProposer, SchemaProposer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="schemaevo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    toy = subparsers.add_parser("run-toy-mvp", help="Run local fixed-pool SchemaEvo MVP demo.")
    toy.add_argument("--config", default="configs/toy_schemaevo.yaml")
    toy.add_argument("--out", default="artifacts/toy_mvp")

    closed_loop = subparsers.add_parser(
        "run-toy-closed-loop",
        help="Run local closed-loop SchemaEvo demo without external APIs.",
    )
    closed_loop.add_argument("--config", default="configs/toy_schemaevo.yaml")
    closed_loop.add_argument("--out", default="artifacts/toy_closed_loop")

    readiness = subparsers.add_parser(
        "check-benchmark-readiness",
        help="Check credentials, packages, and local data needed for real HotpotQA/HoVer runs.",
    )
    readiness.add_argument("--hotpotqa", default=None, help="Path to local HotpotQA JSON/JSONL data.")
    readiness.add_argument("--hover", default=None, help="Path to local HoVer JSON/JSONL data.")
    readiness.add_argument("--strict", action="store_true", help="Exit nonzero when readiness fails.")

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
        "--allow-unready",
        action="store_true",
        help="Skip readiness failure. Intended only for injected-client tests or dry development.",
    )

    args = parser.parse_args(argv)
    if args.command == "run-toy-mvp":
        return _run_toy_mvp(config_path=Path(args.config), out_dir=Path(args.out))
    if args.command == "run-toy-closed-loop":
        return _run_toy_closed_loop(config_path=Path(args.config), out_dir=Path(args.out))
    if args.command == "check-benchmark-readiness":
        return _check_benchmark_readiness(
            hotpotqa_path=args.hotpotqa,
            hover_path=args.hover,
            strict=bool(args.strict),
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
            strict_readiness=not bool(args.allow_unready),
        )
    raise ValueError(args.command)


def _run_toy_mvp(*, config_path: Path, out_dir: Path) -> int:
    raw = _load_yaml(config_path)
    fixed_pool_config = FixedPoolConfig(**raw.get("fixed_pool", {}))
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


def _run_toy_closed_loop(*, config_path: Path, out_dir: Path) -> int:
    raw = _load_yaml(config_path)
    config = SchemaEvoConfig(**raw.get("closed_loop", {}))
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
) -> int:
    readiness = check_benchmark_readiness(
        hotpotqa_path=hotpotqa_path,
        hover_path=hover_path,
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
    strict_readiness: bool,
) -> int:
    _require_existing_paths(train_path, selection_path, confirmation_path, smoke_path, heldout_path)
    data_path = selection_path
    readiness = check_benchmark_readiness(
        hotpotqa_path=data_path if dataset == "hotpotqa" else None,
        hover_path=data_path if dataset == "hover" else None,
    )
    if strict_readiness and not readiness.ready:
        print(json.dumps(readiness.to_dict(), sort_keys=True, indent=2))
        return 1
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
    )
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
    result = run_openai_fixed_pool_benchmark(
        benchmark_config=benchmark_config,
        fixed_pool_config=fixed_pool_config,
        proposer=make_proposer(
            str(proposer_config.get("kind", "heuristic")),
            model=str(proposer_config.get("model", "gpt-4.1-mini")),
            temperature=float(proposer_config.get("temperature", 0.7)),
            max_output_tokens=int(proposer_config.get("max_output_tokens", 4096)),
        ),
        artifact_dir=out_dir,
    )
    print(json.dumps(result.summary(), sort_keys=True, indent=2))
    return 0


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
