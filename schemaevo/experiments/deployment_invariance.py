from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from schemaevo.eval.scoring import CandidateEvalResult
from schemaevo.optimizers.fixed_pool_schema import FixedPoolResult
from schemaevo.schemas.serialization import write_json


@dataclass(frozen=True)
class DeploymentInvarianceReport:
    method: str
    baseline_schema_id: str
    candidate_schema_id: str
    same_target_task_calls_per_example: bool
    same_retriever_calls_per_example: bool
    target_task_calls_per_example_delta: float
    retriever_calls_per_example_delta: float
    dollar_cost_per_example_delta: float
    p95_latency_ms_delta: float
    serving_invariant: bool
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return asdict(self)


def build_fixed_pool_deployment_report(
    *,
    result: FixedPoolResult,
    artifact_dir: str | Path | None = None,
) -> DeploymentInvarianceReport:
    return build_deployment_invariance_report(
        baseline=result.baseline_confirmation_result,
        candidate=result.primary_confirmation_result,
        method="schemaevo_fixed_pool",
        artifact_dir=artifact_dir,
    )


def build_deployment_invariance_report(
    *,
    baseline: CandidateEvalResult,
    candidate: CandidateEvalResult,
    method: str,
    artifact_dir: str | Path | None = None,
) -> DeploymentInvarianceReport:
    baseline_calls = _per_example(baseline.target_task_calls, baseline.n_examples)
    candidate_calls = _per_example(candidate.target_task_calls, candidate.n_examples)
    baseline_retriever = _per_example(baseline.retriever_calls, baseline.n_examples)
    candidate_retriever = _per_example(candidate.retriever_calls, candidate.n_examples)
    call_delta = candidate_calls - baseline_calls
    retriever_delta = candidate_retriever - baseline_retriever
    report = DeploymentInvarianceReport(
        method=method,
        baseline_schema_id=baseline.schema_id,
        candidate_schema_id=candidate.schema_id,
        same_target_task_calls_per_example=call_delta == 0.0,
        same_retriever_calls_per_example=retriever_delta == 0.0,
        target_task_calls_per_example_delta=call_delta,
        retriever_calls_per_example_delta=retriever_delta,
        dollar_cost_per_example_delta=candidate.dollar_cost_per_example - baseline.dollar_cost_per_example,
        p95_latency_ms_delta=candidate.p95_latency_ms - baseline.p95_latency_ms,
        serving_invariant=call_delta == 0.0 and retriever_delta == 0.0,
        artifacts={},
    )
    if artifact_dir:
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        json_path = write_json(report.summary(), root / "deployment_invariance_report.json")
        md_path = root / "deployment_invariance_report.md"
        md_path.write_text(_markdown(report), encoding="utf-8")
        report = DeploymentInvarianceReport(
            method=report.method,
            baseline_schema_id=report.baseline_schema_id,
            candidate_schema_id=report.candidate_schema_id,
            same_target_task_calls_per_example=report.same_target_task_calls_per_example,
            same_retriever_calls_per_example=report.same_retriever_calls_per_example,
            target_task_calls_per_example_delta=report.target_task_calls_per_example_delta,
            retriever_calls_per_example_delta=report.retriever_calls_per_example_delta,
            dollar_cost_per_example_delta=report.dollar_cost_per_example_delta,
            p95_latency_ms_delta=report.p95_latency_ms_delta,
            serving_invariant=report.serving_invariant,
            artifacts={"summary": str(json_path), "markdown": str(md_path)},
        )
        write_json(report.summary(), json_path)
    return report


def _per_example(value: int, n_examples: int) -> float:
    return value / n_examples if n_examples else 0.0


def _markdown(report: DeploymentInvarianceReport) -> str:
    decision = "PASS" if report.serving_invariant else "FAIL"
    return (
        "# Deployment-Cost Invariance Report\n\n"
        f"Decision: **{decision}**\n\n"
        f"Method: `{report.method}`\n\n"
        "| Metric | Delta |\n"
        "| --- | ---: |\n"
        f"| Target task calls / example | {report.target_task_calls_per_example_delta:.6f} |\n"
        f"| Retriever calls / example | {report.retriever_calls_per_example_delta:.6f} |\n"
        f"| Dollars / example | {report.dollar_cost_per_example_delta:.10f} |\n"
        f"| p95 latency ms | {report.p95_latency_ms_delta:.6f} |\n"
    )
