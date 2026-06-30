from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

from schemaevo.schemas.serialization import write_json


@dataclass(frozen=True)
class BudgetParetoRow:
    method: str
    score: float
    target_task_calls: int
    retriever_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    dollar_cost: float
    p95_latency_ms: float
    score_per_dollar: float
    score_per_1k_tokens: float
    source_path: str


@dataclass(frozen=True)
class BudgetParetoReport:
    rows: tuple[BudgetParetoRow, ...]
    pareto_methods: tuple[str, ...]
    equal_spend_ready: bool
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "rows": [asdict(row) for row in self.rows],
            "pareto_methods": list(self.pareto_methods),
            "equal_spend_ready": self.equal_spend_ready,
            "artifacts": self.artifacts,
        }


def build_budget_pareto_report(
    *,
    run_paths: dict[str, str | Path],
    artifact_dir: str | Path | None = None,
) -> BudgetParetoReport:
    rows = tuple(
        _row_from_summary(method=method, path=Path(path))
        for method, path in run_paths.items()
    )
    pareto = tuple(row.method for row in rows if _is_pareto(row, rows))
    report = BudgetParetoReport(
        rows=rows,
        pareto_methods=pareto,
        equal_spend_ready=_equal_spend_ready(rows),
        artifacts={},
    )
    if artifact_dir:
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        json_path = write_json(report.summary(), root / "budget_pareto_report.json")
        csv_path = root / "budget_pareto_report.csv"
        _write_csv(rows, csv_path)
        md_path = root / "budget_pareto_report.md"
        md_path.write_text(_markdown(report), encoding="utf-8")
        report = BudgetParetoReport(
            rows=rows,
            pareto_methods=pareto,
            equal_spend_ready=report.equal_spend_ready,
            artifacts={
                "summary": str(json_path),
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
        )
        write_json(report.summary(), json_path)
    return report


def _row_from_summary(*, method: str, path: Path) -> BudgetParetoRow:
    data = json.loads(path.read_text(encoding="utf-8"))
    score = _score_from_summary(data)
    cost = _cost_from_summary(data)
    total_tokens = int(cost.get("total_tokens", int(cost.get("prompt_tokens", 0)) + int(cost.get("completion_tokens", 0))))
    dollar_cost = float(cost.get("dollar_cost", 0.0))
    return BudgetParetoRow(
        method=method,
        score=score,
        target_task_calls=int(cost.get("target_task_calls", 0)),
        retriever_calls=int(cost.get("retriever_calls", 0)),
        prompt_tokens=int(cost.get("prompt_tokens", 0)),
        completion_tokens=int(cost.get("completion_tokens", 0)),
        total_tokens=total_tokens,
        dollar_cost=dollar_cost,
        p95_latency_ms=float(cost.get("max_p95_latency_ms", cost.get("p95_latency_ms", 0.0))),
        score_per_dollar=score / dollar_cost if dollar_cost > 0 else 0.0,
        score_per_1k_tokens=score / (total_tokens / 1000.0) if total_tokens > 0 else 0.0,
        source_path=str(path),
    )


def _score_from_summary(data: dict[str, Any]) -> float:
    for key in (
        "heldout_test_mean",
        "primary_confirmation_mean",
        "schemaevo_best_mean",
        "prompt_mean",
        "baseline_mean",
    ):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    nested = data.get("summary")
    if isinstance(nested, dict):
        return _score_from_summary(nested)
    return 0.0


def _cost_from_summary(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("cost_summary", "budget"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    summary = data.get("summary")
    if isinstance(summary, dict):
        return _cost_from_summary(summary)
    budget = data.get("budget")
    return budget if isinstance(budget, dict) else {}


def _is_pareto(row: BudgetParetoRow, rows: tuple[BudgetParetoRow, ...]) -> bool:
    for other in rows:
        if other is row:
            continue
        no_worse = other.score >= row.score and other.dollar_cost <= row.dollar_cost
        strictly_better = other.score > row.score or other.dollar_cost < row.dollar_cost
        if no_worse and strictly_better:
            return False
    return True


def _equal_spend_ready(rows: tuple[BudgetParetoRow, ...]) -> bool:
    if len(rows) <= 1:
        return False
    positive_costs = [row.dollar_cost for row in rows if row.dollar_cost > 0]
    if not positive_costs:
        return False
    return max(positive_costs) / min(positive_costs) <= 1.05


def _write_csv(rows: tuple[BudgetParetoRow, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else ["method"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _markdown(report: BudgetParetoReport) -> str:
    lines = [
        "# Budget Pareto Report",
        "",
        f"Equal-spend ready: `{report.equal_spend_ready}`",
        "",
        "| Method | Score | Dollars | Tokens | Calls | p95 ms | Pareto |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    pareto = set(report.pareto_methods)
    for row in report.rows:
        lines.append(
            f"| {row.method} | {row.score:.6f} | {row.dollar_cost:.8f} | "
            f"{row.total_tokens} | {row.target_task_calls} | {row.p95_latency_ms:.3f} | "
            f"{row.method in pareto} |"
        )
    return "\n".join(lines) + "\n"
