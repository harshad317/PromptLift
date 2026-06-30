from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from schemaevo.schemas.candidate import SchemaCandidate


def freeze_jsonl(candidates: Iterable[SchemaCandidate], path: str | Path) -> Path:
    """Write a frozen candidate pool atomically in deterministic JSONL order."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_dict() for candidate in candidates]
    rows.sort(key=lambda row: row["schema_id"])
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_path.parent) as tmp:
        tmp_path = Path(tmp.name)
        for row in rows:
            tmp.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    tmp_path.replace(output_path)
    return output_path


def load_jsonl(path: str | Path) -> list[SchemaCandidate]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [SchemaCandidate.from_dict(json.loads(line)) for line in handle if line.strip()]


def write_json(data: object, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_path.parent) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(data, tmp, sort_keys=True, indent=2, ensure_ascii=True)
        tmp.write("\n")
    tmp_path.replace(output_path)
    return output_path
