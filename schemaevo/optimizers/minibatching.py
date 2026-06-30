from __future__ import annotations

import random
from typing import Callable, Hashable

from schemaevo.programs.base import ProgramExample

StratifyKey = Callable[[ProgramExample], Hashable]


def sample_minibatch(
    *,
    examples: tuple[ProgramExample, ...],
    batch_size: int,
    seed: int,
    stratify_key: StratifyKey | None = None,
) -> tuple[ProgramExample, ...]:
    if not examples:
        raise ValueError("examples must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size >= len(examples):
        return examples
    rng = random.Random(seed)
    if stratify_key is None:
        return tuple(rng.sample(list(examples), batch_size))

    strata: dict[Hashable, list[ProgramExample]] = {}
    for example in examples:
        strata.setdefault(stratify_key(example), []).append(example)

    selected: list[ProgramExample] = []
    ordered_keys = sorted(strata, key=str)
    while len(selected) < batch_size and ordered_keys:
        for key in ordered_keys:
            bucket = strata[key]
            if bucket and len(selected) < batch_size:
                selected.append(bucket.pop(rng.randrange(len(bucket))))
        ordered_keys = [key for key in ordered_keys if strata[key]]
    return tuple(selected)
