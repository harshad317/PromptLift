from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

CallType = Literal[
    "target_task",
    "prompt_proposal",
    "reflection",
    "schema_generation",
    "schema_repair",
    "analysis_ablation",
]


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cached_input_per_million: float = 0.0
    source_date: str = "unset"


@dataclass(frozen=True)
class CostLedgerEntry:
    run_id: str
    method: str
    candidate_id: str
    model: str
    call_type: CallType
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    dollar_cost: float
    latency_ms: int


class CostMeter:
    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        self.prices = prices or {}

    def compute(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        price = self.prices.get(model, ModelPrice())
        uncached_prompt = max(0, prompt_tokens - cached_tokens)
        return (
            uncached_prompt * price.input_per_million
            + cached_tokens * price.cached_input_per_million
            + completion_tokens * price.output_per_million
        ) / 1_000_000.0


class CostLedger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.entries: list[CostLedgerEntry] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, entry: CostLedgerEntry) -> None:
        self.entries.append(entry)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(entry), sort_keys=True, ensure_ascii=True) + "\n")

    def totals(self) -> dict[str, float | int]:
        return {
            "calls": len(self.entries),
            "prompt_tokens": sum(entry.prompt_tokens for entry in self.entries),
            "completion_tokens": sum(entry.completion_tokens for entry in self.entries),
            "cached_tokens": sum(entry.cached_tokens for entry in self.entries),
            "dollar_cost": sum(entry.dollar_cost for entry in self.entries),
            "latency_ms": sum(entry.latency_ms for entry in self.entries),
        }

    def totals_by_call_type(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[CostLedgerEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.call_type, []).append(entry)
        return {
            call_type: {
                "calls": len(entries),
                "prompt_tokens": sum(entry.prompt_tokens for entry in entries),
                "completion_tokens": sum(entry.completion_tokens for entry in entries),
                "cached_tokens": sum(entry.cached_tokens for entry in entries),
                "dollar_cost": sum(entry.dollar_cost for entry in entries),
                "latency_ms": sum(entry.latency_ms for entry in entries),
            }
            for call_type, entries in grouped.items()
        }
