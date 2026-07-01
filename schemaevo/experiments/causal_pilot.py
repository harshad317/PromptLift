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
    ablation_signal_interpretable: bool
    ablation_supports_primary_gain: bool
    control_in_top_k_warning: bool
    primary_is_control: bool
    primary_control_type: str | None
    best_control_vs_primary_delta: float | None
    empirical_status: str
    null_signal_warning: bool
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
    primary_is_control = result.control_guardrail.primary_is_control
    ablation_signal_interpretable = score_delta > 0 and not primary_is_control
    causal_fraction = causal_drop / score_delta if ablation_signal_interpretable else 0.0
    ablation_supports_primary_gain = ablation_signal_interpretable and (
        causal_drop >= min_absolute_drop or causal_fraction >= min_fraction_of_delta
    )
    control_delta = result.control_guardrail.best_control_vs_primary_delta
    control_matched_or_beat_primary = control_delta is not None and control_delta >= 0.0
    empirical_status = _empirical_status(
        score_delta=score_delta,
        primary_is_control=primary_is_control,
        control_matched_or_beat_primary=control_matched_or_beat_primary,
        ablation_supports_primary_gain=ablation_supports_primary_gain,
    )
    reasons: list[str] = []
    if primary_is_control:
        control_type = result.control_guardrail.primary_control_type or "control"
        reasons.append(f"primary selected schema is a {control_type} control, not a designed schema")
    if score_delta <= 0:
        reasons.append("primary schema did not improve over the fixed-schema baseline")
        reasons.append("field ablations are not interpretable as causal support without a positive primary gain")
    elif not ablation_supports_primary_gain:
        reasons.append("mask/shuffle ablations did not remove enough of the observed gain")
    if primary_is_control:
        reasons.append("field ablations are not applicable when the primary schema is a control")
    elif not result.field_ablation_results:
        reasons.append("field ablation results are missing")
    if control_matched_or_beat_primary and not primary_is_control:
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
        ablation_signal_interpretable=ablation_signal_interpretable,
        ablation_supports_primary_gain=ablation_supports_primary_gain,
        control_in_top_k_warning=result.control_guardrail.control_in_top_k_warning,
        primary_is_control=primary_is_control,
        primary_control_type=result.control_guardrail.primary_control_type,
        best_control_vs_primary_delta=control_delta,
        empirical_status=empirical_status,
        null_signal_warning=empirical_status != "positive_schema_signal",
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
            ablation_signal_interpretable=report.ablation_signal_interpretable,
            ablation_supports_primary_gain=report.ablation_supports_primary_gain,
            control_in_top_k_warning=report.control_in_top_k_warning,
            primary_is_control=report.primary_is_control,
            primary_control_type=report.primary_control_type,
            best_control_vs_primary_delta=report.best_control_vs_primary_delta,
            empirical_status=report.empirical_status,
            null_signal_warning=report.null_signal_warning,
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


def _empirical_status(
    *,
    score_delta: float,
    primary_is_control: bool,
    control_matched_or_beat_primary: bool,
    ablation_supports_primary_gain: bool,
) -> str:
    if primary_is_control:
        return "control_selected_as_primary"
    if score_delta <= 0:
        return "negative_or_no_primary_gain"
    if control_matched_or_beat_primary:
        return "control_matches_or_beats_primary"
    if not ablation_supports_primary_gain:
        return "no_causal_ablation_support"
    return "positive_schema_signal"


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
        f"| Ablation signal interpretable | {str(report.ablation_signal_interpretable)} |\n"
        f"| Ablation supports primary gain | {str(report.ablation_supports_primary_gain)} |\n"
        f"| Control in top-k warning | {str(report.control_in_top_k_warning)} |\n"
        f"| Primary is control | {str(report.primary_is_control)} |\n"
        f"| Primary control type | {report.primary_control_type or 'n/a'} |\n"
        f"| Best control vs primary delta | {_format_optional(report.best_control_vs_primary_delta)} |\n"
        f"| Empirical status | {report.empirical_status} |\n"
        f"| Null-signal warning | {str(report.null_signal_warning)} |\n\n"
        "Reasons:\n"
        f"{reasons}\n\n"
        f"Statement: scrambling field content drops score by {report.max_shuffle_drop:.6f}.\n"
    )


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"
