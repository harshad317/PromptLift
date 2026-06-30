#!/usr/bin/env python3
"""Slice a HotpotQA distractor JSON file into disjoint pilot splits.

Usage:
    python3 scripts/make_pilot_splits.py path/to/hotpot_dev_distractor_v1.json

HotpotQA's native distractor format already has the fields the SchemaEvo loader
needs (_id, question, answer, context), so this just takes disjoint slices and
writes data/hotpotqa/{train,selection,confirmation}.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Must match configs/pilot_hotpotqa_gpt41mini.yaml (train 20 / selection 20 / confirmation 40).
SIZES = {"train": 20, "selection": 20, "confirmation": 40}
KEEP = ("_id", "question", "answer", "context")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    source = Path(sys.argv[1])
    items = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit("expected a JSON list of HotpotQA items")

    needed = sum(SIZES.values())
    usable = [it for it in items if all(k in it and it.get("context") for k in KEEP)]
    if len(usable) < needed:
        raise SystemExit(f"need {needed} items with context, found {len(usable)}")

    out_dir = Path("data/hotpotqa")
    out_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    for split, size in SIZES.items():
        chunk = usable[cursor : cursor + size]
        cursor += size
        records = [{k: it[k] for k in KEEP} for it in chunk]
        path = out_dir / f"{split}.json"
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(records):>3} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
