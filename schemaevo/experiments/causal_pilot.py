from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from schemaevo.optimizers.fixed_pool_schema import FixedPoolResult
from schemaevo.schemas.serialization import write_json


@dataclass(frozen=True)
class CausalPilotReport:
    dataset: str
    model: str
    primary_schema_id: str
    baseline_mean: float
    primary_mean: float
    score_delta: float
    max_mask_drop: float
    max_shuffle_drop: float
    max_blank_drop: float
    max_downstream_disabled_drop: float
    causal_drop_fraction: float
    control_in_top_k_warning: bool
    best_control_vs_primary_delta: float | None
    proceed: bool
    reasons: tuple[str, ...]
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return asdict(self)


def build_causal_pilot_report(
    *,
    result: FixedPoolResult,
    dataset: str,
    model: str,
    artifact_dir: str | Path | None = None,
    min_fraction_of_delta: float = 0.5,
    min_absolute_drop: float = 0.015,
) -> CausalPilotReport:
    score_delta = result.primary_confirmation_result.mean_score - result.baseline_confirmation_result.mean_score
    max_mask = _max_drop(result, "mask")
    max_shuffle = _max_drop(result, "shuffle")
    max_blank = _max_drop(result, "blank")
    max_downstream_disabled = _max_drop(result, "downstream_disabled")
    causal_drop = max(max_mask, max_shuffle)
    causal_fraction = causal_drop / score_delta if score_delta > 0 else 0.0
    reasons: list[str] = []
    if score_delta <= 0:
        reasons.append("primary schema did not improve over the fixed-schema baseline")
    if causal_drop < min_absolute_drop and causal_fraction < min_fraction_of_delta:
        reasons.append("mask/shuffle ablations did not remove enough of the observed gain")
    if not result.field_ablation_results:
        reasons.append("field ablation results are missing")
    control_delta = result.control_guardrail.best_control_vs_primary_delta
    if control_delta is not None and control_delta >= 0.0:
        reasons.append("a random or validator-only control schema matched or beat the primary schema")
    proceed = not reasons
    artifacts: dict[str, str] = {}
    report = CausalPilotReport(
        dataset=dataset,
        model=model,
        primary_schema_id=result.primary_confirmation_result.schema_id,
        baseline_mean=result.baseline_confirmation_result.mean_score,
        primary_mean=result.primary_confirmation_result.mean_score,
        score_delta=score_delta,
        max_mask_drop=max_mask,
        max_shuffle_drop=max_shuffle,
        max_blank_drop=max_blank,
        max_downstream_disabled_drop=max_downstream_disabled,
        causal_drop_fraction=causal_fraction,
        control_in_top_k_warning=result.control_guardrail.control_in_top_k_warning,
        best_control_vs_primary_delta=control_delta,
        proceed=proceed,
        reasons=tuple(reasons),
        artifacts=artifacts,
    )
    if artifact_dir:
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        json_path = write_json(report.summary(), root / "causal_pilot_report.json")
        md_path = root / "causal_pilot_report.md"
        md_path.write_text(_markdown(report), encoding="utf-8")
        artifacts = {"summary": str(json_path), "markdown": str(md_path)}
        report = CausalPilotReport(
            dataset=report.dataset,
            model=report.model,
            primary_schema_id=report.primary_schema_id,
            baseline_mean=report.baseline_mean,
            primary_mean=report.primary_mean,
            score_delta=report.score_delta,
            max_mask_drop=report.max_mask_drop,
            max_shuffle_drop=report.max_shuffle_drop,
            max_blank_drop=report.max_blank_drop,
            max_downstream_disabled_drop=report.max_downstream_disabled_drop,
            causal_drop_fraction=report.causal_drop_fraction,
            control_in_top_k_warning=report.control_in_top_k_warning,
            best_control_vs_primary_delta=report.best_control_vs_primary_delta,
            proceed=report.proceed,
            reasons=report.reasons,
            artifacts=artifacts,
        )
        write_json(report.summary(), json_path)
    return report


def _max_drop(result: FixedPoolResult, ablation: str) -> float:
    return max(
        (
            item.drop_vs_unablated
            for item in result.field_ablation_results
            if item.ablation == ablation
        ),
        default=0.0,
    )


def _markdown(report: CausalPilotReport) -> str:
    decision = "GO" if report.proceed else "NO-GO"
    reasons = "\n".join(f"- {reason}" for reason in report.reasons) or "- none"
    return (
        "# Causal Pilot Report\n\n"
        f"Decision: **{decision}**\n\n"
        f"Dataset: `{report.dataset}`\n\n"
        f"Model: `{report.model}`\n\n"
        f"Primary schema: `{report.primary_schema_id}`\n\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"| Baseline mean | {report.baseline_mean:.6f} |\n"
        f"| Primary mean | {report.primary_mean:.6f} |\n"
        f"| Score delta | {report.score_delta:.6f} |\n"
        f"| Max mask drop | {report.max_mask_drop:.6f} |\n"
        f"| Max shuffle drop | {report.max_shuffle_drop:.6f} |\n"
        f"| Max blank drop | {report.max_blank_drop:.6f} |\n"
        f"| Max downstream-disabled drop | {report.max_downstream_disabled_drop:.6f} |\n"
        f"| Causal drop fraction | {report.causal_drop_fraction:.6f} |\n"
        f"| Control in top-k warning | {str(report.control_in_top_k_warning)} |\n"
        f"| Best control vs primary delta | {_format_optional(report.best_control_vs_primary_delta)} |\n\n"
        "Reasons:\n"
        f"{reasons}\n\n"
        f"Statement: scrambling field content drops score by {report.max_shuffle_drop:.6f}.\n"
    )


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
