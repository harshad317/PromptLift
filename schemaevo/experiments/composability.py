from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemaevo.eval.cost_ledger import make_cost_meter
from schemaevo.eval.scoring import CandidateEvalResult, Scorer, evaluate_program
from schemaevo.eval.stats import compare_paired
from schemaevo.optimizers.schema_evo import SchemaEvoConfig, SchemaEvoRunResult, schema_evo_optimize
from schemaevo.programs.base import LMProgram, ProgramExample
from schemaevo.programs.call_graph import assert_same_call_graph
from schemaevo.schemas.serialization import write_json

PromptOptimizer = Callable[[LMProgram], LMProgram]


@dataclass(frozen=True)
class ComposabilityRunResult:
    base_result: CandidateEvalResult
    prompt_result: CandidateEvalResult
    schemaevo_result: SchemaEvoRunResult
    schemaevo_eval_results: tuple[CandidateEvalResult, ...]
    best_schemaevo_eval_result: CandidateEvalResult | None
    prompt_delta: float
    schemaevo_additive_delta: float
    paired_stats: dict[str, Any]
    budget_summary: dict[str, Any]
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "base_mean": self.base_result.mean_score,
            "prompt_mean": self.prompt_result.mean_score,
            "schemaevo_best_mean": (
                self.best_schemaevo_eval_result.mean_score
                if self.best_schemaevo_eval_result
                else None
            ),
            "schemaevo_optimizer_best_mean": (
                max((record.result.mean_score for record in self.schemaevo_result.final_records), default=0.0)
            ),
            "prompt_delta": self.prompt_delta,
            "schemaevo_additive_delta": self.schemaevo_additive_delta,
            "paired_stats": self.paired_stats,
            "same_eval_examples": _same_eval_examples(
                self.base_result,
                self.prompt_result,
                self.schemaevo_eval_results,
            ),
            "budget": self.budget_summary,
            "schemaevo": self.schemaevo_result.summary(),
            "artifacts": self.artifacts,
        }


def run_prompt_optimizer_then_schemaevo(
    *,
    base_program: LMProgram,
    prompt_optimizer: PromptOptimizer,
    prompt_eval_examples: tuple[ProgramExample, ...],
    schema_optimizer_examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    schema_config: SchemaEvoConfig,
    artifact_dir: str | Path | None = None,
) -> ComposabilityRunResult:
    """Run an external prompt optimizer first, then SchemaEvo on the frozen interface.

    The harness deliberately accepts the prompt optimizer as a callable instead
    of implementing GEPA/MIPRO inside this package. That keeps SchemaEvo as the
    composable layer while still producing budget-matched additive deltas.
    """

    artifact_root = Path(artifact_dir) if artifact_dir else None
    if artifact_root:
        artifact_root.mkdir(parents=True, exist_ok=True)
    eval_cost_meter = make_cost_meter(
        model_prices=schema_config.model_prices,
        use_tiktoken=schema_config.use_tiktoken_costing,
    )

    base_result = evaluate_program(
        program=base_program,
        examples=prompt_eval_examples,
        scorer=scorer,
        method="composability_base",
        candidate_id="composability_base",
        seed=schema_config.seed,
        baseline_program=base_program,
        strict_invalid_policy=schema_config.strict_invalid_policy,
        artifact_dir=artifact_root,
        cost_meter=eval_cost_meter,
        run_id="composability_base",
    )
    prompt_program = prompt_optimizer(base_program.clone())
    assert_same_call_graph(prompt_program, base_program)
    prompt_result = evaluate_program(
        program=prompt_program,
        examples=prompt_eval_examples,
        scorer=scorer,
        method="composability_prompt_optimizer",
        candidate_id="composability_prompt_optimizer",
        seed=schema_config.seed,
        baseline_program=base_program,
        strict_invalid_policy=schema_config.strict_invalid_policy,
        artifact_dir=artifact_root,
        cost_meter=eval_cost_meter,
        run_id="composability_prompt_optimizer",
    )
    schema_result = schema_evo_optimize(
        base_program=prompt_program,
        examples=schema_optimizer_examples,
        scorer=scorer,
        config=schema_config,
        artifact_dir=artifact_root / "schemaevo" if artifact_root else None,
    )
    schema_eval_results = tuple(
        _evaluate_schemaevo_final_record(
            record=record,
            examples=prompt_eval_examples,
            scorer=scorer,
            schema_config=schema_config,
            baseline_program=base_program,
            artifact_root=artifact_root,
            cost_meter=eval_cost_meter,
            index=index,
        )
        for index, record in enumerate(schema_result.final_records)
    )
    best_schema_eval = (
        max(
            schema_eval_results,
            key=lambda result: (result.mean_score, -result.invalid_output_rate, result.schema_id),
        )
        if schema_eval_results
        else None
    )
    paired_stats = {
        "prompt_vs_base": _paired_summary(base_result, prompt_result, seed=schema_config.seed),
        "schemaevo_vs_prompt": (
            _paired_summary(prompt_result, best_schema_eval, seed=schema_config.seed + 1)
            if best_schema_eval
            else None
        ),
    }
    budget_summary = _budget_summary(
        base_result=base_result,
        prompt_result=prompt_result,
        schemaevo_eval_results=schema_eval_results,
        schemaevo_result=schema_result,
    )
    result = ComposabilityRunResult(
        base_result=base_result,
        prompt_result=prompt_result,
        schemaevo_result=schema_result,
        schemaevo_eval_results=schema_eval_results,
        best_schemaevo_eval_result=best_schema_eval,
        prompt_delta=prompt_result.mean_score - base_result.mean_score,
        schemaevo_additive_delta=(
            best_schema_eval.mean_score - prompt_result.mean_score if best_schema_eval else 0.0
        ),
        paired_stats=paired_stats,
        budget_summary=budget_summary,
        artifacts={},
    )
    artifacts: dict[str, str] = {}
    if artifact_root:
        summary_path = write_json(
            {
                "base_result": base_result.to_dict(),
                "prompt_result": prompt_result.to_dict(),
                "schemaevo_eval_results": [item.to_dict() for item in schema_eval_results],
                "schemaevo_summary": schema_result.summary(),
                "paired_stats": paired_stats,
                "summary": result.summary(),
            },
            artifact_root / "composability_summary.json",
        )
        artifacts["summary"] = str(summary_path)
        result = ComposabilityRunResult(
            base_result=base_result,
            prompt_result=prompt_result,
            schemaevo_result=schema_result,
            schemaevo_eval_results=schema_eval_results,
            best_schemaevo_eval_result=best_schema_eval,
            prompt_delta=result.prompt_delta,
            schemaevo_additive_delta=result.schemaevo_additive_delta,
            paired_stats=paired_stats,
            budget_summary=budget_summary,
            artifacts=artifacts,
        )
    return result


def _evaluate_schemaevo_final_record(
    *,
    record: Any,
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    schema_config: SchemaEvoConfig,
    baseline_program: LMProgram,
    artifact_root: Path | None,
    cost_meter: Any,
    index: int,
) -> CandidateEvalResult:
    result = evaluate_program(
        program=record.program,
        examples=examples,
        scorer=scorer,
        method="composability_schemaevo_final",
        candidate_id=f"composability_schemaevo_final_{index}",
        seed=schema_config.seed,
        baseline_program=baseline_program,
        strict_invalid_policy=schema_config.strict_invalid_policy,
        artifact_dir=artifact_root,
        cost_meter=cost_meter,
        schema_generation_calls=0,
        run_id=f"composability_schemaevo_final_{index}_{record.schema.schema_id}",
    )
    return result


def _budget_summary(
    *,
    base_result: CandidateEvalResult,
    prompt_result: CandidateEvalResult,
    schemaevo_eval_results: tuple[CandidateEvalResult, ...],
    schemaevo_result: SchemaEvoRunResult,
) -> dict[str, Any]:
    eval_results = (base_result, prompt_result, *schemaevo_eval_results)
    evaluation = {
        "target_task_calls": sum(result.target_task_calls for result in eval_results),
        "prompt_tokens": sum(result.prompt_tokens for result in eval_results),
        "completion_tokens": sum(result.completion_tokens for result in eval_results),
        "total_tokens": sum(result.prompt_tokens + result.completion_tokens for result in eval_results),
        "dollar_cost": sum(result.dollar_cost for result in eval_results),
    }
    optimizer = dict(schemaevo_result.budget_summary)
    return {
        "evaluation": evaluation,
        "schemaevo_optimizer": optimizer,
        "total_target_task_calls": evaluation["target_task_calls"]
        + int(optimizer.get("target_task_calls", 0) or 0),
        "total_tokens": evaluation["total_tokens"] + int(optimizer.get("total_tokens", 0) or 0),
        "total_dollar_cost": evaluation["dollar_cost"] + float(optimizer.get("dollar_cost", 0.0) or 0.0),
    }


def _paired_summary(
    baseline: CandidateEvalResult,
    candidate: CandidateEvalResult,
    *,
    seed: int,
) -> dict[str, Any]:
    stats = compare_paired(
        baseline.per_example_scores,
        candidate.per_example_scores,
        n_resamples=1000,
        n_swaps=1000,
        seed=seed,
    )
    return {
        "bootstrap": {
            "mean_diff": stats.bootstrap.mean_diff,
            "ci_low": stats.bootstrap.ci_low,
            "ci_high": stats.bootstrap.ci_high,
            "n_resamples": stats.bootstrap.n_resamples,
        },
        "approximate_randomization_p": stats.approximate_randomization_p,
        "adjusted_p": stats.adjusted_p,
        "correction": stats.correction,
    }
def _example_ids(result: CandidateEvalResult) -> tuple[str, ...]:
    return tuple(prediction.example_id for prediction in result.predictions)


def _same_eval_examples(
    base_result: CandidateEvalResult,
    prompt_result: CandidateEvalResult,
    schemaevo_eval_results: tuple[CandidateEvalResult, ...],
) -> bool:
    expected = _example_ids(base_result)
    if _example_ids(prompt_result) != expected:
        return False
    return all(_example_ids(result) == expected for result in schemaevo_eval_results)
