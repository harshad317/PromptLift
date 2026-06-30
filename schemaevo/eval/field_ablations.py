from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemaevo.eval.cost_ledger import CostMeter
from schemaevo.eval.scoring import CandidateEvalResult, Scorer, evaluate_program
from schemaevo.programs.base import FieldIntervention, LMProgram, ProgramExample


@dataclass(frozen=True)
class FieldAblationResult:
    ablation: str
    field_name: str
    mean_score: float
    drop_vs_unablated: float
    invalid_output_rate: float
    per_example_scores: tuple[float, ...]


class MaskFieldsIntervention:
    def __init__(self, fields: tuple[str, ...], consumer_modules: tuple[str, ...] | None = None) -> None:
        self.fields = set(fields)
        self.consumer_modules = set(consumer_modules or ())

    def before_module(
        self,
        *,
        module_name: str,
        state: dict[str, Any],
        example: ProgramExample,
    ) -> None:
        if self.consumer_modules and module_name not in self.consumer_modules:
            return
        for field_name in self.fields:
            if field_name in state["schema_fields"]:
                state["schema_fields"][field_name] = None


class BlankFieldsIntervention:
    def __init__(self, fields: tuple[str, ...], consumer_modules: tuple[str, ...] | None = None) -> None:
        self.fields = set(fields)
        self.consumer_modules = set(consumer_modules or ())

    def before_module(
        self,
        *,
        module_name: str,
        state: dict[str, Any],
        example: ProgramExample,
    ) -> None:
        if self.consumer_modules and module_name not in self.consumer_modules:
            return
        for field_name in self.fields:
            if field_name in state["schema_fields"]:
                state["schema_fields"][field_name] = ""


class ShuffleFieldsIntervention:
    def __init__(
        self,
        field_values_by_example: dict[str, dict[str, Any]],
        *,
        consumer_modules: tuple[str, ...] | None = None,
    ) -> None:
        self.field_values_by_example = field_values_by_example
        self.consumer_modules = set(consumer_modules or ())

    def before_module(
        self,
        *,
        module_name: str,
        state: dict[str, Any],
        example: ProgramExample,
    ) -> None:
        if self.consumer_modules and module_name not in self.consumer_modules:
            return
        for field_name, values_by_example in self.field_values_by_example.items():
            if field_name in state["schema_fields"] and example.example_id in values_by_example:
                state["schema_fields"][field_name] = values_by_example[example.example_id]


class DownstreamConsumptionDisabledIntervention:
    def __init__(self, consumer_modules: tuple[str, ...] | None = None) -> None:
        self.consumer_modules = set(consumer_modules or ())

    def before_module(
        self,
        *,
        module_name: str,
        state: dict[str, Any],
        example: ProgramExample,
    ) -> None:
        if self.consumer_modules and module_name not in self.consumer_modules:
            return
        state["schema_fields"] = {}


def run_field_use_ablations(
    *,
    program: LMProgram,
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    unablated_result: CandidateEvalResult,
    fields: tuple[str, ...],
    seed: int,
    artifact_dir: str | Path | None = None,
    cost_meter: CostMeter | None = None,
) -> list[FieldAblationResult]:
    results: list[FieldAblationResult] = []
    consumer_modules = _consumer_modules(program)
    shuffled_values = _rotated_field_values_by_example(unablated_result, fields)
    for field_name in fields:
        for ablation, intervention in (
            ("mask", MaskFieldsIntervention((field_name,), consumer_modules=consumer_modules)),
            ("blank", BlankFieldsIntervention((field_name,), consumer_modules=consumer_modules)),
            (
                "shuffle",
                ShuffleFieldsIntervention(
                    {field_name: shuffled_values.get(field_name, {})},
                    consumer_modules=consumer_modules,
                ),
            ),
        ):
            result = _eval_intervention(
                program=program,
                examples=examples,
                scorer=scorer,
                seed=seed,
                artifact_dir=artifact_dir,
                cost_meter=cost_meter,
                intervention=intervention,
                method=f"field_{ablation}",
                field_name=field_name,
            )
            results.append(
                FieldAblationResult(
                    ablation=ablation,
                    field_name=field_name,
                    mean_score=result.mean_score,
                    drop_vs_unablated=unablated_result.mean_score - result.mean_score,
                    invalid_output_rate=result.invalid_output_rate,
                    per_example_scores=result.per_example_scores,
                )
            )
    if fields:
        result = _eval_intervention(
            program=program,
            examples=examples,
            scorer=scorer,
            seed=seed,
            artifact_dir=artifact_dir,
            cost_meter=cost_meter,
            intervention=DownstreamConsumptionDisabledIntervention(consumer_modules=consumer_modules),
            method="field_downstream_disabled",
            field_name="__all__",
        )
        results.append(
            FieldAblationResult(
                ablation="downstream_disabled",
                field_name="__all__",
                mean_score=result.mean_score,
                drop_vs_unablated=unablated_result.mean_score - result.mean_score,
                invalid_output_rate=result.invalid_output_rate,
                per_example_scores=result.per_example_scores,
            )
        )
    return results


def _eval_intervention(
    *,
    program: LMProgram,
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    seed: int,
    artifact_dir: str | Path | None,
    cost_meter: CostMeter | None,
    intervention: FieldIntervention,
    method: str,
    field_name: str,
) -> CandidateEvalResult:
    return evaluate_program(
        program=program,
        examples=examples,
        scorer=scorer,
        method=method,
        candidate_id=f"{method}_{field_name}",
        seed=seed,
        baseline_program=None,
        field_intervention=intervention,
        artifact_dir=artifact_dir,
        cost_meter=cost_meter,
        run_id=f"{method}_{field_name}",
    )


def _consumer_modules(program: LMProgram) -> tuple[str, ...]:
    if not program.schema_candidate:
        return ()
    return tuple(sorted({rule.consumer_module for rule in program.schema_candidate.consumption_rules}))


def _rotated_field_values_by_example(
    unablated_result: CandidateEvalResult,
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    values_by_field: dict[str, list[tuple[str, Any]]] = {field: [] for field in fields}
    for prediction in unablated_result.predictions:
        for field_name in fields:
            value = _find_field_value(prediction.module_outputs, field_name)
            if value is not _MISSING:
                values_by_field[field_name].append((prediction.example_id, value))

    rotated: dict[str, dict[str, Any]] = {}
    for field_name, pairs in values_by_field.items():
        if len(pairs) < 2:
            continue
        rotated[field_name] = {
            example_id: pairs[(index + 1) % len(pairs)][1]
            for index, (example_id, _value) in enumerate(pairs)
        }
    return rotated


_MISSING = object()


def _find_field_value(module_outputs: dict[str, dict[str, Any]], field_name: str) -> Any:
    for output in module_outputs.values():
        if field_name in output:
            return output[field_name]
    return _MISSING
