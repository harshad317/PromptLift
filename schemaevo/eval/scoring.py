from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
from typing import Callable
from uuid import uuid4

from schemaevo.eval.cost_ledger import CostLedger, CostMeter
from schemaevo.eval.cache import RolloutCache
from schemaevo.eval.stats import bootstrap_mean_ci
from schemaevo.eval.logging import JSONLLogSink, MemoryLogSink, OutputPayloadStore
from schemaevo.programs.base import FieldIntervention, LMProgram, ProgramExample, ProgramPrediction
from schemaevo.programs.call_graph import assert_same_call_graph
from schemaevo.schemas.validators import ValidationPolicy

Scorer = Callable[[ProgramExample, ProgramPrediction], float]


@dataclass
class CandidateEvalResult:
    run_id: str
    method: str
    candidate_id: str
    schema_id: str
    task: str
    split: str
    n_examples: int
    mean_score: float
    standard_error: float
    score_ci_low: float
    score_ci_high: float
    per_example_scores_path: str
    target_task_calls: int
    optimizer_proposal_calls: int
    optimizer_reflection_calls: int
    schema_generation_calls: int
    schema_validation_repair_calls: int
    retriever_calls: int
    prompt_tokens: int
    completion_tokens: int
    dollar_cost: float
    wall_clock_seconds: float
    p50_latency_ms: float
    p95_latency_ms: float
    invalid_output_rate: float
    schema_validation_failure_count: int
    invalid_score_policy: str
    field_use_score: float = 0.0
    baseline_dollar_cost_per_example: float = 0.0
    per_example_scores: tuple[float, ...] = ()
    predictions: tuple[ProgramPrediction, ...] = ()

    @property
    def dollar_cost_per_example(self) -> float:
        return self.dollar_cost / self.n_examples if self.n_examples else 0.0

    def to_dict(self, include_predictions: bool = False) -> dict[str, object]:
        data = asdict(self)
        data["predictions"] = (
            [_prediction_summary(prediction) for prediction in self.predictions]
            if include_predictions
            else []
        )
        return data


def evaluate_program(
    *,
    program: LMProgram,
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    method: str,
    candidate_id: str,
    seed: int,
    baseline_program: LMProgram | None = None,
    validation_policy: ValidationPolicy = "deterministic_coercion",
    strict_invalid_policy: bool = True,
    run_id: str | None = None,
    artifact_dir: str | Path | None = None,
    cost_meter: CostMeter | None = None,
    field_intervention: FieldIntervention | None = None,
    rollout_cache: RolloutCache | None = None,
    intervention_id: str = "none",
    optimizer_proposal_calls: int = 0,
    optimizer_reflection_calls: int = 0,
    schema_generation_calls: int = 0,
) -> CandidateEvalResult:
    if baseline_program is not None:
        assert_same_call_graph(program, baseline_program)
    run_id = run_id or f"{method}_{uuid4().hex[:12]}"
    artifact_root = Path(artifact_dir) if artifact_dir else None
    log_sink = JSONLLogSink(artifact_root / "logs" / f"{run_id}.jsonl") if artifact_root else MemoryLogSink()
    payload_store = OutputPayloadStore(artifact_root / "payloads" if artifact_root else None)
    cost_ledger = CostLedger(artifact_root / "cost_ledgers" / f"{run_id}.jsonl") if artifact_root else None

    predictions: list[ProgramPrediction] = []
    scores: list[float] = []
    for example in examples:
        cache_key = (
            rollout_cache.key(
                program=program,
                example=example,
                seed=seed,
                intervention_id=intervention_id,
            )
            if rollout_cache
            else ""
        )
        cached = rollout_cache.get(cache_key) if rollout_cache else None
        if cached:
            prediction = cached
        else:
            prediction = program.run(
                example,
                run_id=run_id,
                method=method,
                candidate_id=candidate_id,
                seed=seed,
                validation_policy=validation_policy,
                log_sink=log_sink,
                cost_meter=cost_meter,
                cost_ledger=cost_ledger,
                payload_store=payload_store,
                field_intervention=field_intervention,
            )
            if rollout_cache:
                rollout_cache.set(cache_key, prediction)
        score = 0.0 if strict_invalid_policy and not prediction.valid else scorer(example, prediction)
        scores.append(float(score))
        predictions.append(prediction)

    score_path = ""
    if artifact_root:
        score_dir = artifact_root / "results"
        score_dir.mkdir(parents=True, exist_ok=True)
        score_file = score_dir / f"{run_id}_scores.jsonl"
        with score_file.open("w", encoding="utf-8") as handle:
            for example, score, prediction in zip(examples, scores, predictions):
                handle.write(
                    json.dumps(
                        _score_row(example=example, score=score, prediction=prediction),
                        sort_keys=True,
                        ensure_ascii=True,
                        default=str,
                    )
                    + "\n"
                )
        score_path = str(score_file)

    latencies = [prediction.latency_ms for prediction in predictions]
    score_ci_low, score_ci_high = bootstrap_mean_ci(tuple(scores), n_resamples=1000, seed=seed)
    invalid_count = sum(1 for prediction in predictions if not prediction.valid)
    repair_calls = sum(
        sum(log.llm_repair_call_count for log in prediction.module_logs) for prediction in predictions
    )
    field_uses = sum(len(prediction.field_use_events) for prediction in predictions)
    candidate_schema_fields = len(program.schema_candidate.all_fields) if program.schema_candidate else 0
    field_use_score = (
        field_uses / max(1, len(examples) * candidate_schema_fields) if candidate_schema_fields else 0.0
    )
    return CandidateEvalResult(
        run_id=run_id,
        method=method,
        candidate_id=candidate_id,
        schema_id=program.schema_candidate.schema_id if program.schema_candidate else "original_schema",
        task=program.task,
        split=_single_split(examples),
        n_examples=len(examples),
        mean_score=_mean(scores),
        standard_error=_standard_error(scores),
        score_ci_low=score_ci_low,
        score_ci_high=score_ci_high,
        per_example_scores_path=score_path,
        target_task_calls=sum(prediction.target_task_calls for prediction in predictions),
        optimizer_proposal_calls=optimizer_proposal_calls,
        optimizer_reflection_calls=optimizer_reflection_calls,
        schema_generation_calls=schema_generation_calls,
        schema_validation_repair_calls=repair_calls,
        retriever_calls=sum(prediction.retriever_calls for prediction in predictions),
        prompt_tokens=sum(prediction.prompt_tokens for prediction in predictions),
        completion_tokens=sum(prediction.completion_tokens for prediction in predictions),
        dollar_cost=sum(prediction.dollar_cost for prediction in predictions),
        wall_clock_seconds=sum(prediction.latency_ms for prediction in predictions) / 1000.0,
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        invalid_output_rate=invalid_count / len(predictions) if predictions else 0.0,
        schema_validation_failure_count=invalid_count,
        invalid_score_policy="zero" if strict_invalid_policy else "score",
        field_use_score=field_use_score,
        per_example_scores=tuple(scores),
        predictions=tuple(predictions),
    )


def evaluate_candidates(
    *,
    candidates: tuple[LMProgram, ...],
    baseline_program: LMProgram,
    examples: tuple[ProgramExample, ...],
    scorer: Scorer,
    method: str,
    seed: int,
    artifact_dir: str | Path | None = None,
    validation_policy: ValidationPolicy = "deterministic_coercion",
    strict_invalid_policy: bool = True,
    cost_meter: CostMeter | None = None,
) -> list[CandidateEvalResult]:
    results = []
    for index, candidate in enumerate(candidates):
        schema_id = candidate.schema_candidate.schema_id if candidate.schema_candidate else "original_schema"
        results.append(
            evaluate_program(
                program=candidate,
                examples=examples,
                scorer=scorer,
                method=method,
                candidate_id=f"{method}_{index}",
                seed=seed,
                baseline_program=baseline_program,
                validation_policy=validation_policy,
                strict_invalid_policy=strict_invalid_policy,
                artifact_dir=artifact_dir,
                cost_meter=cost_meter,
                schema_generation_calls=1 if candidate.schema_candidate else 0,
                run_id=f"{method}_{schema_id}_{index}",
            )
        )
    return results


def _prediction_summary(prediction: ProgramPrediction) -> dict[str, object]:
    return {
        "run_id": prediction.run_id,
        "example_id": prediction.example_id,
        "candidate_id": prediction.candidate_id,
        "schema_id": prediction.schema_id,
        "final_output": prediction.final_output,
        "valid": prediction.valid,
        "validation_errors": list(prediction.validation_errors),
        "target_task_calls": prediction.target_task_calls,
        "retriever_calls": prediction.retriever_calls,
        "prompt_tokens": prediction.prompt_tokens,
        "completion_tokens": prediction.completion_tokens,
        "dollar_cost": prediction.dollar_cost,
        "latency_ms": prediction.latency_ms,
    }


def _score_row(
    *,
    example: ProgramExample,
    score: float,
    prediction: ProgramPrediction,
) -> dict[str, object]:
    row = _prediction_summary(prediction)
    row.update(
        {
            "task_split": example.split,
            "score": score,
            "schema_validation_repair_calls": sum(
                getattr(log, "llm_repair_call_count", 0) for log in prediction.module_logs
            ),
            "field_use_events": [asdict(event) for event in prediction.field_use_events],
        }
    )
    return row


def _single_split(examples: tuple[ProgramExample, ...]) -> str:
    splits = sorted({example.split for example in examples})
    return splits[0] if len(splits) == 1 else "+".join(splits)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)
