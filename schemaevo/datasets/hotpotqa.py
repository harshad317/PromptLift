from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemaevo.programs.base import ProgramExample


def load_hotpotqa_examples(
    path: str | Path,
    *,
    split: str,
    limit: int | None = None,
) -> tuple[ProgramExample, ...]:
    examples: list[ProgramExample] = []
    for index, item in enumerate(_read_records(path)):
        if limit is not None and len(examples) >= limit:
            break
        example_id = str(item.get("_id") or item.get("id") or f"hotpotqa_{split}_{index}")
        context = _normalize_context(item.get("context", []))
        examples.append(
            ProgramExample(
                example_id=example_id,
                split=split,
                inputs={
                    "question": item.get("question", ""),
                    "context": context,
                },
                expected={
                    "answer": item.get("answer"),
                    "supporting_facts": item.get("supporting_facts", []),
                },
                metadata={
                    "dataset": "hotpotqa",
                    "level": item.get("level"),
                    "type": item.get("type"),
                    "raw_index": index,
                },
            )
        )
    return tuple(examples)


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        loaded = json.load(handle)
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    if isinstance(loaded, dict):
        for key in ("data", "examples", "records"):
            value = loaded.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"unsupported HotpotQA file shape: {source}")


def _normalize_context(raw_context: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_context, list):
        return normalized
    for item in raw_context:
        if isinstance(item, list) and len(item) == 2:
            title, sentences = item
            normalized.append({"title": title, "sentences": list(sentences or [])})
        elif isinstance(item, dict):
            normalized.append(dict(item))
    return normalized
