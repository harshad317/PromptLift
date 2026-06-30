from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

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

    args = parser.parse_args(argv)
    if args.command == "run-toy-mvp":
        return _run_toy_mvp(config_path=Path(args.config), out_dir=Path(args.out))
    if args.command == "run-toy-closed-loop":
        return _run_toy_closed_loop(config_path=Path(args.config), out_dir=Path(args.out))
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
