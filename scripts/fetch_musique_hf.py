#!/usr/bin/env python3
"""Fetch MuSiQue from Hugging Face and write disjoint 3+ hop pilot splits.

Avoids the rate-limited Google Drive mirror. Requires:  pip install datasets

Usage:
    python3 scripts/fetch_musique_hf.py

Writes data/musique/{train,selection,confirmation}.json in the
[_id, question, answer, context=[[title,[text]], ...]] shape the HotpotQA
loader already reads, so you run it with --dataset hotpotqa and no code changes.
"""
from __future__ import annotations

import json
from pathlib import Path

SIZES = {"train": 20, "selection": 20, "confirmation": 40}
MIN_HOPS = 3


def _load_validation():
    from datasets import load_dataset

    last: Exception | None = None
    for dataset_id in ("dgslibisey/MuSiQue", "bdsaglam/musique"):
        for split in ("validation", "dev"):
            try:
                return load_dataset(dataset_id, split=split, trust_remote_code=True)
            except Exception as exc:  # noqa: BLE001 - keep trying other ids/splits
                last = exc
    raise SystemExit(f"could not load MuSiQue from Hugging Face: {last}")


def _to_context(paragraphs) -> list:
    # HF mirrors store paragraphs either as a list of dicts or parallel lists.
    if isinstance(paragraphs, dict):
        titles = paragraphs.get("title", [])
        texts = paragraphs.get("paragraph_text", [])
        return [[t, [txt]] for t, txt in zip(titles, texts) if txt]
    context = []
    for i, p in enumerate(paragraphs or []):
        txt = p.get("paragraph_text", "")
        if txt:
            context.append([p.get("title", f"para_{p.get('idx', i)}"), [txt]])
    return context


def _hops(row) -> int:
    decomp = row.get("question_decomposition")
    if isinstance(decomp, dict):
        return len(decomp.get("question", []))
    return len(decomp or [])


def main() -> int:
    rows = _load_validation()
    needed = sum(SIZES.values())
    records: list[dict] = []
    for row in rows:
        if not row.get("answerable", True):
            continue
        answer = (row.get("answer") or "").strip()
        if not answer or _hops(row) < MIN_HOPS:
            continue
        context = _to_context(row.get("paragraphs"))
        if not context:
            continue
        records.append({"_id": row["id"], "question": row["question"], "answer": answer, "context": context})
        if len(records) >= needed:
            break
    if len(records) < needed:
        raise SystemExit(f"need {needed} answerable {MIN_HOPS}+ hop items, found {len(records)}")

    out_dir = Path("data/musique")
    out_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    for split, size in SIZES.items():
        chunk = records[cursor : cursor + size]
        cursor += size
        path = out_dir / f"{split}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(chunk):>3} -> {path}")
    print(f"(filtered to >= {MIN_HOPS} hops; many paragraphs/item -> watch --max-dollar-cost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
