from __future__ import annotations

import re
import string

from schemaevo.programs.base import ProgramExample, ProgramPrediction


def hotpotqa_exact_match(example: ProgramExample, prediction: ProgramPrediction) -> float:
    expected = _normalize_answer(str(example.expected.get("answer", "")))
    predicted = _normalize_answer(str(prediction.final_output.get("answer", "")))
    return 1.0 if expected and predicted == expected else 0.0


def musique_exact_match(example: ProgramExample, prediction: ProgramPrediction) -> float:
    expected_answers = [str(example.expected.get("answer", ""))]
    aliases = example.expected.get("answer_aliases", ())
    if isinstance(aliases, (list, tuple)):
        expected_answers.extend(str(alias) for alias in aliases)
    predicted = _normalize_answer(str(prediction.final_output.get("answer", "")))
    return 1.0 if predicted and predicted in {_normalize_answer(answer) for answer in expected_answers if answer} else 0.0


def hotpotqa_f1(example: ProgramExample, prediction: ProgramPrediction) -> float:
    """Token-overlap F1, the standard HotpotQA metric alongside exact match.

    Robust to verbose answers that contain the gold span (which EM scores 0).
    """
    expected = _normalize_answer(str(example.expected.get("answer", "")))
    predicted = _normalize_answer(str(prediction.final_output.get("answer", "")))
    if not expected or not predicted:
        return 0.0
    # yes/no/noanswer must match exactly, matching the official HotpotQA scorer.
    if expected in {"yes", "no", "noanswer"} or predicted in {"yes", "no", "noanswer"}:
        return 1.0 if predicted == expected else 0.0
    gold_tokens = expected.split()
    pred_tokens = predicted.split()
    gold_counts: dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    num_same = sum(min(count, gold_counts.get(token, 0)) for token, count in pred_counts.items())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


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
