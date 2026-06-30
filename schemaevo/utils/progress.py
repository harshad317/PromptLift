from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from typing import Literal, TypeVar

ProgressMode = Literal["auto", "rich", "tqdm", "none"]

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    description: str = "",
    mode: ProgressMode = "auto",
) -> Iterator[T]:
    selected = _select_progress_mode(mode)
    if selected == "rich":
        yield from _rich_progress(iterable, total=total, description=description)
        return
    if selected == "tqdm":
        yield from _tqdm_progress(iterable, total=total, description=description)
        return
    yield from iterable


def _select_progress_mode(mode: ProgressMode) -> ProgressMode:
    if mode == "none":
        return "none"
    if mode in {"rich", "tqdm"}:
        return mode
    if not sys.stderr.isatty():
        return "none"
    try:
        import rich  # noqa: F401
    except Exception:
        try:
            import tqdm  # noqa: F401
        except Exception:
            return "none"
        return "tqdm"
    return "rich"


def _rich_progress(
    iterable: Iterable[T],
    *,
    total: int | None,
    description: str,
) -> Iterator[T]:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    console = Console(stderr=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(description or "working", total=total)
        for item in iterable:
            yield item
            progress.advance(task_id)


def _tqdm_progress(
    iterable: Iterable[T],
    *,
    total: int | None,
    description: str,
) -> Iterator[T]:
    from tqdm.auto import tqdm

    yield from tqdm(
        iterable,
        total=total,
        desc=description or None,
        file=sys.stderr,
        leave=False,
        dynamic_ncols=True,
    )
