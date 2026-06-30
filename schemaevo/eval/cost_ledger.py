from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Literal

from schemaevo.schemas.candidate import approximate_token_count

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
    def __init__(
        self,
        prices: dict[str, ModelPrice] | None = None,
        *,
        token_counter: Callable[[str, str], int] | None = None,
        use_tiktoken: bool | None = None,
    ) -> None:
        self.prices = prices or {}
        self.token_counter = token_counter
        self.use_tiktoken = (
            os.environ.get("SCHEMAEVO_USE_TIKTOKEN") == "1" if use_tiktoken is None else use_tiktoken
        )
        self._encodings: dict[str, object] = {}

    def count_tokens(self, *, model: str, text: str) -> int:
        if not text:
            return 0
        if self.token_counter:
            return max(1, int(self.token_counter(model, text)))
        if self.use_tiktoken:
            counted = self._count_with_tiktoken(model, text)
            if counted is not None:
                return counted
        return approximate_token_count(text)

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

    def _count_with_tiktoken(self, model: str, text: str) -> int | None:
        try:
            import tiktoken
        except Exception:
            return None
        try:
            encoding = self._encodings.get(model)
            if encoding is None:
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")
                self._encodings[model] = encoding
            return max(1, len(encoding.encode(text)))  # type: ignore[attr-defined]
        except Exception:
            return None


def make_cost_meter(
    *,
    model_prices: dict[str, dict[str, float | str]] | None = None,
    use_tiktoken: bool = False,
) -> CostMeter:
    prices: dict[str, ModelPrice] = {}
    for model, raw in (model_prices or {}).items():
        prices[model] = ModelPrice(
            input_per_million=float(raw.get("input_per_million", 0.0)),
            output_per_million=float(raw.get("output_per_million", 0.0)),
            cached_input_per_million=float(raw.get("cached_input_per_million", 0.0)),
            source_date=str(raw.get("source_date", "config")),
        )
    return CostMeter(prices=prices, use_tiktoken=use_tiktoken)


@dataclass(frozen=True)
class BudgetLimits:
    max_target_task_calls: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    max_dollar_cost: float | None = None

    @property
    def enabled(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_target_task_calls,
                self.max_prompt_tokens,
                self.max_completion_tokens,
                self.max_total_tokens,
                self.max_dollar_cost,
            )
        )


class BudgetTracker:
    def __init__(self, limits: BudgetLimits | None = None) -> None:
        self.limits = limits or BudgetLimits()
        self.target_task_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.dollar_cost = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def exhausted(self) -> bool:
        limits = self.limits
        return any(
            (
                limits.max_target_task_calls is not None
                and self.target_task_calls >= limits.max_target_task_calls,
                limits.max_prompt_tokens is not None
                and self.prompt_tokens >= limits.max_prompt_tokens,
                limits.max_completion_tokens is not None
                and self.completion_tokens >= limits.max_completion_tokens,
                limits.max_total_tokens is not None
                and self.total_tokens >= limits.max_total_tokens,
                limits.max_dollar_cost is not None
                and self.dollar_cost >= limits.max_dollar_cost,
            )
        )

    def can_start(
        self,
        *,
        min_target_task_calls: int = 0,
        min_prompt_tokens: int = 0,
        min_completion_tokens: int = 0,
        min_dollar_cost: float = 0.0,
    ) -> bool:
        limits = self.limits
        if limits.max_target_task_calls is not None:
            if self.target_task_calls + min_target_task_calls > limits.max_target_task_calls:
                return False
        if limits.max_prompt_tokens is not None:
            if self.prompt_tokens + min_prompt_tokens > limits.max_prompt_tokens:
                return False
        if limits.max_completion_tokens is not None:
            if self.completion_tokens + min_completion_tokens > limits.max_completion_tokens:
                return False
        if limits.max_total_tokens is not None:
            if self.total_tokens + min_prompt_tokens + min_completion_tokens > limits.max_total_tokens:
                return False
        if limits.max_dollar_cost is not None:
            if self.dollar_cost + min_dollar_cost > limits.max_dollar_cost:
                return False
        return True

    def record_result(self, result: object) -> None:
        self.target_task_calls += int(getattr(result, "target_task_calls", 0))
        self.prompt_tokens += int(getattr(result, "prompt_tokens", 0))
        self.completion_tokens += int(getattr(result, "completion_tokens", 0))
        self.dollar_cost += float(getattr(result, "dollar_cost", 0.0))

    def summary(self) -> dict[str, float | int | bool | None]:
        return {
            "enabled": self.limits.enabled,
            "exhausted": self.exhausted,
            "target_task_calls": self.target_task_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "dollar_cost": self.dollar_cost,
            "max_target_task_calls": self.limits.max_target_task_calls,
            "max_prompt_tokens": self.limits.max_prompt_tokens,
            "max_completion_tokens": self.limits.max_completion_tokens,
            "max_total_tokens": self.limits.max_total_tokens,
            "max_dollar_cost": self.limits.max_dollar_cost,
        }


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
