from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemaevo.eval.scoring import CandidateEvalResult, Scorer, evaluate_program
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
    prompt_delta: float
    schemaevo_additive_delta: float
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "base_mean": self.base_result.mean_score,
            "prompt_mean": self.prompt_result.mean_score,
            "schemaevo_best_mean": (
                max((record.result.mean_score for record in self.schemaevo_result.final_records), default=0.0)
            ),
            "prompt_delta": self.prompt_delta,
            "schemaevo_additive_delta": self.schemaevo_additive_delta,
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
        run_id="composability_prompt_optimizer",
    )
    schema_result = schema_evo_optimize(
        base_program=prompt_program,
        examples=schema_optimizer_examples,
        scorer=scorer,
        config=schema_config,
        artifact_dir=artifact_root / "schemaevo" if artifact_root else None,
    )
    schema_best = max((record.result.mean_score for record in schema_result.final_records), default=0.0)
    result = ComposabilityRunResult(
        base_result=base_result,
        prompt_result=prompt_result,
        schemaevo_result=schema_result,
        prompt_delta=prompt_result.mean_score - base_result.mean_score,
        schemaevo_additive_delta=schema_best - prompt_result.mean_score,
        artifacts={},
    )
    artifacts: dict[str, str] = {}
    if artifact_root:
        summary_path = write_json(
            {
                "base_result": base_result.to_dict(),
                "prompt_result": prompt_result.to_dict(),
                "schemaevo_summary": schema_result.summary(),
                "summary": result.summary(),
            },
            artifact_root / "composability_summary.json",
        )
        artifacts["summary"] = str(summary_path)
        result = ComposabilityRunResult(
            base_result=base_result,
            prompt_result=prompt_result,
            schemaevo_result=schema_result,
            prompt_delta=result.prompt_delta,
            schemaevo_additive_delta=result.schemaevo_additive_delta,
            artifacts=artifacts,
        )
    return result
