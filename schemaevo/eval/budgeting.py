from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from schemaevo.eval.cost_ledger import CostMeter
from schemaevo.programs.base import LMProgram, ProgramExample


@dataclass(frozen=True)
class EvaluationBudgetEstimate:
    target_task_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dollar_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def plus(self, other: "EvaluationBudgetEstimate") -> "EvaluationBudgetEstimate":
        return EvaluationBudgetEstimate(
            target_task_calls=self.target_task_calls + other.target_task_calls,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            dollar_cost=self.dollar_cost + other.dollar_cost,
        )


def estimate_evaluation_budget(
    *,
    program: LMProgram,
    examples: tuple[ProgramExample, ...],
    cost_meter: CostMeter,
) -> EvaluationBudgetEstimate:
    prompt_tokens = 0
    completion_tokens = 0
    dollar_cost = 0.0
    for example in examples:
        for module in program.modules:
            prompt_probe = _prompt_probe(program=program, module_name=module.name, example=example)
            module_prompt_tokens = cost_meter.count_tokens(model=module.model, text=prompt_probe)
            module_completion_tokens = max(0, module.max_output_tokens) * max(1, module.llm_calls)
            prompt_tokens += module_prompt_tokens
            completion_tokens += module_completion_tokens
            dollar_cost += cost_meter.compute(
                model=module.model,
                prompt_tokens=module_prompt_tokens,
                completion_tokens=module_completion_tokens,
            )
    return EvaluationBudgetEstimate(
        target_task_calls=len(examples) * program.calls_per_example,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        dollar_cost=dollar_cost,
    )


def _prompt_probe(*, program: LMProgram, module_name: str, example: ProgramExample) -> str:
    module = next(item for item in program.modules if item.name == module_name)
    probe: dict[str, Any] = {
        "inputs": example.inputs,
        "metadata": example.metadata,
        "schema_fields": {},
        "module_outputs": {},
    }
    if program.schema_candidate:
        probe["schema_id"] = program.schema_candidate.schema_id
        probe["schema_fields_declared"] = program.schema_candidate.evolved_field_names
    return module.prompt + str(probe)
