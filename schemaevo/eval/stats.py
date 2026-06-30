from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class BootstrapDiff:
    mean_diff: float
    ci_low: float
    ci_high: float
    n_resamples: int


@dataclass(frozen=True)
class PairedComparison:
    bootstrap: BootstrapDiff
    approximate_randomization_p: float
    adjusted_p: float | None = None
    correction: str | None = None


def paired_bootstrap_diff(
    y_base: tuple[float, ...],
    y_method: tuple[float, ...],
    *,
    n_resamples: int = 10000,
    seed: int = 0,
) -> BootstrapDiff:
    _validate_paired(y_base, y_method)
    rng = random.Random(seed)
    n = len(y_base)
    diffs: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            index = rng.randrange(n)
            total += y_method[index] - y_base[index]
        diffs.append(total / n)
    diffs.sort()
    return BootstrapDiff(
        mean_diff=sum(y_method[i] - y_base[i] for i in range(n)) / n,
        ci_low=_quantile(diffs, 0.025),
        ci_high=_quantile(diffs, 0.975),
        n_resamples=n_resamples,
    )


def approximate_randomization(
    y_base: tuple[float, ...],
    y_method: tuple[float, ...],
    *,
    n_swaps: int = 10000,
    seed: int = 0,
) -> float:
    _validate_paired(y_base, y_method)
    rng = random.Random(seed)
    observed = abs(_mean_diff(y_base, y_method))
    count = 0
    n = len(y_base)
    for _ in range(n_swaps):
        swapped_base = list(y_base)
        swapped_method = list(y_method)
        for index in range(n):
            if rng.random() < 0.5:
                swapped_base[index], swapped_method[index] = swapped_method[index], swapped_base[index]
        stat = abs(_mean_diff(tuple(swapped_base), tuple(swapped_method)))
        if stat >= observed:
            count += 1
    return (count + 1) / (n_swaps + 1)


def compare_paired(
    y_base: tuple[float, ...],
    y_method: tuple[float, ...],
    *,
    n_resamples: int = 10000,
    n_swaps: int = 10000,
    seed: int = 0,
) -> PairedComparison:
    return PairedComparison(
        bootstrap=paired_bootstrap_diff(
            y_base,
            y_method,
            n_resamples=n_resamples,
            seed=seed,
        ),
        approximate_randomization_p=approximate_randomization(
            y_base,
            y_method,
            n_swaps=n_swaps,
            seed=seed + 1,
        ),
    )


def bootstrap_mean_ci(
    values: tuple[float, ...],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return _quantile(means, 0.025), _quantile(means, 0.975)


def bonferroni_adjust(p_values: tuple[float, ...]) -> tuple[float, ...]:
    if not p_values:
        return ()
    m = len(p_values)
    return tuple(min(1.0, p * m) for p in p_values)


def benjamini_hochberg_adjust(p_values: tuple[float, ...]) -> tuple[float, ...]:
    if not p_values:
        return ()
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(indexed)
    adjusted = [1.0] * m
    running_min = 1.0
    for rank_from_end, (original_index, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        value = min(1.0, p_value * m / rank)
        running_min = min(running_min, value)
        adjusted[original_index] = running_min
    return tuple(adjusted)


def _validate_paired(y_base: tuple[float, ...], y_method: tuple[float, ...]) -> None:
    if len(y_base) != len(y_method):
        raise ValueError("paired arrays must have equal length")
    if not y_base:
        raise ValueError("paired arrays must be non-empty")


def _mean_diff(y_base: tuple[float, ...], y_method: tuple[float, ...]) -> float:
    return sum(y_method[i] - y_base[i] for i in range(len(y_base))) / len(y_base)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight
