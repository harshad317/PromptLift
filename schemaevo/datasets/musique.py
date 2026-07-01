from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemaevo.programs.base import ProgramExample


def load_musique_examples(
    path: str | Path,
    *,
    split: str,
    limit: int | None = None,
    source_split: str | None = None,
) -> tuple[ProgramExample, ...]:
    examples: list[ProgramExample] = []
    for index, item in enumerate(_read_records(path, split=split, source_split=source_split)):
        if limit is not None and len(examples) >= limit:
            break
        example_id = str(item.get("id") or item.get("_id") or f"musique_{split}_{index}")
        context = _normalize_context(item.get("context"))
        if not context:
            context = _normalize_paragraphs(item.get("paragraphs"))
        examples.append(
            ProgramExample(
                example_id=example_id,
                split=split,
                inputs={
                    "question": item.get("question", ""),
                    "context": context,
                },
                expected={
                    "answer": item.get("answer", item.get("gold")),
                    "answer_aliases": item.get("answer_aliases", ()),
                    "question_decomposition": item.get("question_decomposition", ()),
                },
                metadata={
                    "dataset": "musique",
                    "answerable": item.get("answerable"),
                    "question_type": item.get("question_type") or item.get("type"),
                    "source_split": source_split or _source_split_for_runtime_split(split),
                    "raw_index": index,
                },
            )
        )
    return tuple(examples)


def _read_records(
    path: str | Path,
    *,
    split: str,
    source_split: str | None = None,
) -> list[dict[str, Any]]:
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
        records = _records_from_split_dict(loaded, split=split, source_split=source_split)
        if records is not None:
            return records
    raise ValueError(f"unsupported MuSiQue file shape: {source}")


def _records_from_split_dict(
    loaded: dict[str, Any],
    *,
    split: str,
    source_split: str | None = None,
) -> list[dict[str, Any]] | None:
    requested = source_split or _source_split_for_runtime_split(split)
    candidates: list[str] = []
    if requested:
        candidates.append(requested)
    candidates.extend(_split_fallbacks(split))
    for key in candidates:
        value = loaded.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return None


def _source_split_for_runtime_split(split: str) -> str | None:
    return {
        "train": "train",
        "validation_smoke": "smoke",
        "validation_selection": "selection",
        "validation_confirmation": "confirmation",
        "optimizer_validation": "optimizer_validation",
        "heldout_validation": "heldout_validation",
        "final_test": "test",
        "readiness": "selection",
    }.get(split)


def _split_fallbacks(split: str) -> list[str]:
    if split == "final_test":
        return ["heldout_validation", "test"]
    if split == "readiness":
        return ["selection", "validation", "dev", "train", "test", "heldout_validation"]
    return [split, "data", "examples", "records"]


def _normalize_context(raw_context: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_context, list):
        return normalized
    for item in raw_context:
        if isinstance(item, list) and len(item) == 2:
            title, sentences = item
            sentence_list = list(sentences or []) if isinstance(sentences, list) else [str(sentences)]
            normalized.append(
                {
                    "title": title,
                    "sentences": sentence_list,
                    "text": " ".join(str(sentence) for sentence in sentence_list),
                }
            )
        elif isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


def _normalize_paragraphs(raw_paragraphs: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(raw_paragraphs, list):
        return normalized
    for index, paragraph in enumerate(raw_paragraphs):
        if not isinstance(paragraph, dict):
            continue
        text = paragraph.get("paragraph_text") or paragraph.get("text") or ""
        sentences = paragraph.get("sentences")
        sentence_list = list(sentences) if isinstance(sentences, list) else ([str(text)] if text else [])
        normalized.append(
            {
                "title": paragraph.get("title") or paragraph.get("paragraph_title") or f"paragraph_{index}",
                "sentences": sentence_list,
                "text": " ".join(str(sentence) for sentence in sentence_list),
                "idx": paragraph.get("idx", index),
                "is_supporting": bool(paragraph.get("is_supporting", False)),
            }
        )
    return normalized
