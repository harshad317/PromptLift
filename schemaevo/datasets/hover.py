from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemaevo.programs.base import ProgramExample


def load_hover_examples(
    path: str | Path,
    *,
    split: str,
    limit: int | None = None,
) -> tuple[ProgramExample, ...]:
    examples: list[ProgramExample] = []
    for index, item in enumerate(_read_records(path)):
        if limit is not None and len(examples) >= limit:
            break
        example_id = str(item.get("uid") or item.get("id") or f"hover_{split}_{index}")
        claim = item.get("claim") or item.get("question") or ""
        evidence = item.get("evidence") or item.get("context") or item.get("supporting_facts") or []
        examples.append(
            ProgramExample(
                example_id=example_id,
                split=split,
                inputs={
                    "claim": claim,
                    "context": evidence,
                },
                expected={
                    "label": item.get("label"),
                    "supporting_facts": item.get("supporting_facts", []),
                },
                metadata={
                    "dataset": "hover",
                    "num_hops": item.get("num_hops"),
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
    raise ValueError(f"unsupported HoVer file shape: {source}")
