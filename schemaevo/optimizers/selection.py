from __future__ import annotations

from schemaevo.eval.scoring import CandidateEvalResult


def schema_selection_value(result: CandidateEvalResult, *, use_field_bonus: bool = False) -> float:
    score_lcb = result.score_ci_low
    baseline_cost = result.baseline_dollar_cost_per_example or result.dollar_cost_per_example or 1.0
    relative_dollar_overhead = (result.dollar_cost_per_example / baseline_cost) - 1.0
    invalid_penalty = 0.0 if result.invalid_score_policy == "zero" else 2.0 * result.invalid_output_rate
    cost_penalty = 0.1 * max(0.0, relative_dollar_overhead)
    field_use_bonus = 0.05 * result.field_use_score if use_field_bonus else 0.0
    return score_lcb - cost_penalty - invalid_penalty + field_use_bonus


def select_top_k_by_lcb(
    results: list[CandidateEvalResult],
    *,
    k: int,
    use_field_bonus: bool = False,
) -> list[CandidateEvalResult]:
    return sorted(
        results,
        key=lambda result: (
            schema_selection_value(result, use_field_bonus=use_field_bonus),
            result.mean_score,
            -result.invalid_output_rate,
            result.schema_id,
        ),
        reverse=True,
    )[:k]
