#!/usr/bin/env python3
"""Convert a MuSiQue jsonl file into disjoint 3+ hop pilot splits.

MuSiQue is the right stress test for the interface hypothesis: 3-4 hop questions
over many distractor paragraphs, so a single prose plan_summary genuinely loses
information between hops. We reuse the existing HotpotQA plumbing by emitting the
same [_id, question, answer, context=[[title,[text]], ...]] shape, so you run it
with --dataset hotpotqa and no code changes.

Get the data (any one of these), then point this script at the .jsonl:
  - Official repo StonyBrookNLP/musique -> musique_ans_v1.0_dev.jsonl
  - or a Hugging Face mirror of MuSiQue-Answerable exported to jsonl

Usage:
    python3 scripts/make_musique_splits.py path/to/musique_ans_v1.0_dev.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SIZES = {"train": 20, "selection": 20, "confirmation": 40}
MIN_HOPS = 3


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _to_record(item: dict) -> dict | None:
    answer = (item.get("answer") or "").strip()
    if not answer or not item.get("answerable", True):
        return None
    if len(item.get("question_decomposition", [])) < MIN_HOPS:
        return None
    context = [
        [p.get("title", f"para_{p.get('idx', i)}"), [p.get("paragraph_text", "")]]
        for i, p in enumerate(item.get("paragraphs", []))
        if p.get("paragraph_text")
    ]
    if not context:
        return None
    return {"_id": item["id"], "question": item["question"], "answer": answer, "context": context}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    source = Path(sys.argv[1])
    needed = sum(SIZES.values())
    records: list[dict] = []
    for item in _iter_jsonl(source):
        rec = _to_record(item)
        if rec:
            records.append(rec)
        if len(records) >= needed:
            break
    if len(records) < needed:
        raise SystemExit(f"need {needed} answerable 3+ hop items, found {len(records)}")

    out_dir = Path("data/musique")
    out_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    for split, size in SIZES.items():
        chunk = records[cursor : cursor + size]
        cursor += size
        path = out_dir / f"{split}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(chunk):>3} -> {path}")
    print(f"(filtered to >= {MIN_HOPS} hops; avg paragraphs/item can be large -> watch --max-dollar-cost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
