from __future__ import annotations

import re
import string

from schemaevo.programs.base import ProgramExample, ProgramPrediction


def hotpotqa_exact_match(example: ProgramExample, prediction: ProgramPrediction) -> float:
    expected = _normalize_answer(str(example.expected.get("answer", "")))
    predicted = _normalize_answer(str(prediction.final_output.get("answer", "")))
    return 1.0 if expected and predicted == expected else 0.0


def hover_label_accuracy(example: ProgramExample, prediction: ProgramPrediction) -> float:
    expected = _normalize_label(example.expected.get("label"))
    predicted = _normalize_label(prediction.final_output.get("label"))
    return 1.0 if expected and predicted == expected else 0.0


def _normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    return " ".join(value.split())


def _normalize_label(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")
