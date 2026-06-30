from __future__ import annotations

from dataclasses import dataclass

from schemaevo.eval.scoring import CandidateEvalResult
from schemaevo.optimizers.selection import schema_selection_value
from schemaevo.programs.base import LMProgram
from schemaevo.schemas.candidate import SchemaCandidate
from schemaevo.schemas.mutations import Mutation


@dataclass(frozen=True)
class CandidateRecord:
    schema: SchemaCandidate
    program: LMProgram
    result: CandidateEvalResult
    mutation: Mutation | None = None


class ParetoFront:
    def __init__(self) -> None:
        self._records: list[CandidateRecord] = []

    @property
    def records(self) -> tuple[CandidateRecord, ...]:
        return tuple(self._records)

    def update(self, candidate: CandidateRecord) -> None:
        survivors: list[CandidateRecord] = []
        for existing in self._records:
            if _dominates(candidate, existing):
                continue
            if _dominates(existing, candidate):
                return
            survivors.append(existing)
        survivors.append(candidate)
        self._records = survivors

    def top(self, k: int) -> tuple[CandidateRecord, ...]:
        return tuple(
            sorted(
                self._records,
                key=lambda record: (
                    schema_selection_value(record.result, use_field_bonus=True),
                    record.result.mean_score,
                    -record.result.invalid_output_rate,
                    -record.result.dollar_cost_per_example,
                    record.schema.schema_id,
                ),
                reverse=True,
            )[:k]
        )


def _dominates(left: CandidateRecord, right: CandidateRecord) -> bool:
    score_separated = _score_lcb(left.result) >= _score_ucb(right.result)
    no_worse_cost = left.result.dollar_cost_per_example <= right.result.dollar_cost_per_example
    no_worse_tokens = _tokens_per_example(left.result) <= _tokens_per_example(right.result)
    no_worse_invalidity = left.result.invalid_output_rate <= right.result.invalid_output_rate
    no_worse_latency = left.result.p95_latency_ms <= right.result.p95_latency_ms
    strictly_better_non_score = any(
        (
            left.result.dollar_cost_per_example < right.result.dollar_cost_per_example,
            _tokens_per_example(left.result) < _tokens_per_example(right.result),
            left.result.invalid_output_rate < right.result.invalid_output_rate,
            left.result.p95_latency_ms < right.result.p95_latency_ms,
        )
    )
    return (
        score_separated
        and no_worse_cost
        and no_worse_tokens
        and no_worse_invalidity
        and no_worse_latency
        and (score_separated or strictly_better_non_score)
    )


def _score_lcb(result: CandidateEvalResult, z: float = 1.0) -> float:
    return result.score_ci_low


def _score_ucb(result: CandidateEvalResult, z: float = 1.0) -> float:
    return result.score_ci_high


def _tokens_per_example(result: CandidateEvalResult) -> float:
    return float(result.prompt_tokens + result.completion_tokens) / max(1, result.n_examples)
