from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import os
from pathlib import Path
from typing import Any

from schemaevo.datasets.hotpotqa import load_hotpotqa_examples
from schemaevo.datasets.hover import load_hover_examples


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
) -> BenchmarkReadiness:
    datasets: dict[str, dict[str, Any]] = {}
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

    if hotpotqa_path is not None:
        datasets["hotpotqa"] = _check_dataset(
            path=hotpotqa_path,
            loader=lambda path: load_hotpotqa_examples(path, split="readiness", limit=3),
        )
        if not datasets["hotpotqa"]["ok"]:
            reasons.append(f"HotpotQA data unavailable: {datasets['hotpotqa']['reason']}")
    if hover_path is not None:
        datasets["hover"] = _check_dataset(
            path=hover_path,
            loader=lambda path: load_hover_examples(path, split="readiness", limit=3),
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


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_dataset(
    *,
    path: str | Path,
    loader,
) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {"ok": False, "path": str(source), "count": 0, "reason": "path does not exist"}
    try:
        examples = loader(source)
    except Exception as exc:
        return {"ok": False, "path": str(source), "count": 0, "reason": str(exc)}
    return {
        "ok": bool(examples),
        "path": str(source),
        "count": len(examples),
        "reason": "" if examples else "loader returned no examples",
    }
