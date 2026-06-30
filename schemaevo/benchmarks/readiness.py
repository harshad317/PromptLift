from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import os
from pathlib import Path
from typing import Any

from schemaevo.datasets.hotpotqa import load_hotpotqa_examples
from schemaevo.datasets.hover import load_hover_examples
from schemaevo.programs.base import ProgramExample


@dataclass(frozen=True)
class BenchmarkReadiness:
    openai_api_key: bool
    openai_package: bool
    tiktoken_package: bool
    dspy_package: bool
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)
    ready: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_benchmark_readiness(
    *,
    hotpotqa_path: str | Path | None = None,
    hover_path: str | Path | None = None,
    require_context: bool = True,
    inspect_limit: int = 200,
) -> BenchmarkReadiness:
    datasets: dict[str, dict[str, Any]] = {}
    (
        openai_api_key,
        openai_package,
        tiktoken_package,
        dspy_package,
        reasons,
    ) = _environment_readiness()

    if hotpotqa_path is not None:
        datasets["hotpotqa"] = _check_dataset(
            path=hotpotqa_path,
            loader=lambda path: load_hotpotqa_examples(path, split="readiness", limit=inspect_limit),
            dataset="hotpotqa",
            require_context=require_context,
        )
        if not datasets["hotpotqa"]["ok"]:
            reasons.append(f"HotpotQA data unavailable: {datasets['hotpotqa']['reason']}")
    if hover_path is not None:
        datasets["hover"] = _check_dataset(
            path=hover_path,
            loader=lambda path: load_hover_examples(path, split="readiness", limit=inspect_limit),
            dataset="hover",
            require_context=require_context,
        )
        if not datasets["hover"]["ok"]:
            reasons.append(f"HoVer data unavailable: {datasets['hover']['reason']}")
    if hotpotqa_path is None and hover_path is None:
        reasons.append("no HotpotQA or HoVer data path was provided")

    ready = not reasons
    return BenchmarkReadiness(
        openai_api_key=openai_api_key,
        openai_package=openai_package,
        tiktoken_package=tiktoken_package,
        dspy_package=dspy_package,
        datasets=datasets,
        ready=ready,
        reasons=tuple(reasons),
    )


def check_fixed_pool_split_readiness(
    *,
    dataset: str,
    train_path: str | Path,
    selection_path: str | Path,
    confirmation_path: str | Path,
    smoke_path: str | Path | None = None,
    heldout_path: str | Path | None = None,
    require_context: bool = True,
    inspect_limit: int = 200,
) -> BenchmarkReadiness:
    datasets: dict[str, dict[str, Any]] = {}
    (
        openai_api_key,
        openai_package,
        tiktoken_package,
        dspy_package,
        reasons,
    ) = _environment_readiness()
    split_specs = (
        ("train", train_path, "train"),
        ("smoke", smoke_path, "validation_smoke"),
        ("selection", selection_path, "validation_selection"),
        ("confirmation", confirmation_path, "validation_confirmation"),
        ("heldout", heldout_path, "final_test"),
    )
    examples_by_label: dict[str, tuple[ProgramExample, ...]] = {}
    for label, path, runtime_split in split_specs:
        if path is None:
            continue
        key = f"{dataset}.{label}"
        loader = _loader_for_dataset(
            dataset=dataset,
            runtime_split=runtime_split,
            inspect_limit=inspect_limit,
        )
        dataset_check, examples = _check_dataset_with_examples(
            path=path,
            loader=loader,
            dataset=dataset,
            require_context=require_context,
        )
        datasets[key] = dataset_check
        if examples:
            examples_by_label[label] = examples
        if not datasets[key]["ok"]:
            reasons.append(f"{_dataset_label(dataset)} {label} data unavailable: {datasets[key]['reason']}")
    for error in _split_overlap_errors(dataset=dataset, examples_by_label=examples_by_label):
        reasons.append(error)
    return BenchmarkReadiness(
        openai_api_key=openai_api_key,
        openai_package=openai_package,
        tiktoken_package=tiktoken_package,
        dspy_package=dspy_package,
        datasets=datasets,
        ready=not reasons,
        reasons=tuple(reasons),
    )


def _environment_readiness() -> tuple[bool, bool, bool, bool, list[str]]:
    reasons: list[str] = []
    openai_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    openai_package = _package_available("openai")
    tiktoken_package = _package_available("tiktoken")
    dspy_package = _package_available("dspy")
    if not openai_api_key:
        reasons.append("OPENAI_API_KEY is not set")
    if not openai_package:
        reasons.append("openai package is not installed")
    if not tiktoken_package:
        reasons.append("tiktoken package is not installed")
    return openai_api_key, openai_package, tiktoken_package, dspy_package, reasons


def _loader_for_dataset(*, dataset: str, runtime_split: str, inspect_limit: int):
    if dataset == "hotpotqa":
        return lambda path: load_hotpotqa_examples(path, split=runtime_split, limit=inspect_limit)
    if dataset == "hover":
        return lambda path: load_hover_examples(path, split=runtime_split, limit=inspect_limit)
    raise ValueError(f"unsupported dataset: {dataset}")


def _dataset_label(dataset: str) -> str:
    if dataset == "hotpotqa":
        return "HotpotQA"
    if dataset == "hover":
        return "HoVer"
    return dataset


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_dataset(
    *,
    path: str | Path,
    loader,
    dataset: str,
    require_context: bool,
) -> dict[str, Any]:
    result, _examples = _check_dataset_with_examples(
        path=path,
        loader=loader,
        dataset=dataset,
        require_context=require_context,
    )
    return result


def _check_dataset_with_examples(
    *,
    path: str | Path,
    loader,
    dataset: str,
    require_context: bool,
) -> tuple[dict[str, Any], tuple[ProgramExample, ...]]:
    source = Path(path)
    if not source.exists():
        return (
            {"ok": False, "path": str(source), "count": 0, "reason": "path does not exist"},
            (),
        )
    try:
        examples = tuple(loader(source))
    except Exception as exc:
        return (
            {"ok": False, "path": str(source), "count": 0, "reason": str(exc)},
            (),
        )
    quality = _quality_metrics(dataset=dataset, examples=examples)
    quality_errors = _quality_errors(dataset=dataset, quality=quality, require_context=require_context)
    return (
        {
            "ok": bool(examples) and not quality_errors,
            "path": str(source),
            "count": len(examples),
            "reason": "; ".join(quality_errors) if quality_errors else ("" if examples else "loader returned no examples"),
            "quality": quality,
        },
        examples,
    )


def _split_overlap_errors(
    *,
    dataset: str,
    examples_by_label: dict[str, tuple[ProgramExample, ...]],
) -> list[str]:
    errors: list[str] = []
    labels = list(examples_by_label)
    for left_index, left_label in enumerate(labels):
        left_ids = {example.example_id for example in examples_by_label[left_label]}
        for right_label in labels[left_index + 1 :]:
            right_ids = {example.example_id for example in examples_by_label[right_label]}
            overlap = sorted(left_ids & right_ids)
            if overlap:
                preview = ", ".join(overlap[:5])
                suffix = "" if len(overlap) <= 5 else f", ... +{len(overlap) - 5} more"
                errors.append(
                    f"{_dataset_label(dataset)} example IDs overlap between "
                    f"{left_label} and {right_label}: {preview}{suffix}"
                )
    return errors


def _quality_metrics(
    *,
    dataset: str,
    examples: tuple[ProgramExample, ...],
) -> dict[str, Any]:
    ids = [example.example_id for example in examples]
    duplicate_ids = len(ids) - len(set(ids))
    context_nonempty = sum(1 for example in examples if _has_context(example.inputs.get("context")))
    question_nonempty = sum(1 for example in examples if example.inputs.get("question") or example.inputs.get("claim"))
    if dataset == "hotpotqa":
        target_nonempty = sum(1 for example in examples if example.expected.get("answer"))
    elif dataset == "hover":
        target_nonempty = sum(1 for example in examples if example.expected.get("label"))
    else:
        target_nonempty = 0
    count = len(examples)
    return {
        "duplicate_ids": duplicate_ids,
        "question_or_claim_coverage": _ratio(question_nonempty, count),
        "target_coverage": _ratio(target_nonempty, count),
        "context_coverage": _ratio(context_nonempty, count),
        "context_nonempty_count": context_nonempty,
    }


def _quality_errors(
    *,
    dataset: str,
    quality: dict[str, Any],
    require_context: bool,
) -> list[str]:
    errors: list[str] = []
    if quality.get("duplicate_ids", 0) > 0:
        errors.append(f"duplicate example IDs: {quality['duplicate_ids']}")
    if quality.get("question_or_claim_coverage", 0.0) < 1.0:
        errors.append("missing question/claim fields")
    if quality.get("target_coverage", 0.0) < 1.0:
        errors.append("missing answer/label targets")
    if require_context and dataset in {"hotpotqa", "hover"} and quality.get("context_coverage", 0.0) < 1.0:
        errors.append(
            f"context coverage {quality.get('context_coverage', 0.0):.3f} < 1.0; "
            "this is not a full context benchmark file"
        )
    return errors


def _has_context(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_context(item) for item in value)
    if isinstance(value, dict):
        if "sentences" in value:
            return _has_context(value.get("sentences"))
        return any(_has_context(item) for item in value.values())
    return value is not None


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
