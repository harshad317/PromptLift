#!/usr/bin/env python3
"""Fetch HotpotQA from Hugging Face and write disjoint pilot splits.

Avoids the flaky CMU mirror. Requires:  pip install datasets

Usage:
    python3 scripts/fetch_hotpotqa_hf.py

Writes data/hotpotqa/{train,selection,confirmation}.json in the
[_id, question, answer, context=[[title,[sentences...]], ...]] shape the
SchemaEvo loader expects.
"""
from __future__ import annotations

import json
from pathlib import Path

# Must match configs/pilot_hotpotqa_gpt41mini.yaml (train 20 / selection 20 / confirmation 40).
SIZES = {"train": 20, "selection": 20, "confirmation": 40}


def _load_split():
    from datasets import load_dataset

    # HF deprecated the bare id at various points; try the namespaced one first.
    for dataset_id in ("hotpotqa/hotpot_qa", "hotpot_qa"):
        try:
            return load_dataset(dataset_id, "distractor", split="validation", trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001 - report and try the fallback id
            last = exc
    raise SystemExit(f"could not load HotpotQA from Hugging Face: {last}")


def _to_context(ctx: dict) -> list:
    # HF stores context as parallel lists: {"title": [...], "sentences": [[...], ...]}.
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])
    return [[title, list(sents)] for title, sents in zip(titles, sentences)]


def main() -> int:
    rows = _load_split()
    needed = sum(SIZES.values())
    records = []
    for row in rows:
        context = _to_context(row["context"])
        if not context:
            continue
        records.append(
            {
                "_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "context": context,
            }
        )
        if len(records) >= needed:
            break
    if len(records) < needed:
        raise SystemExit(f"need {needed} usable items, got {len(records)}")

    out_dir = Path("data/hotpotqa")
    out_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    for split, size in SIZES.items():
        chunk = records[cursor : cursor + size]
        cursor += size
        path = out_dir / f"{split}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(chunk):>3} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
