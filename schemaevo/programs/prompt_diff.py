from __future__ import annotations

import difflib


def unified_prompt_diff(before: str, after: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before_prompt",
            tofile="after_prompt",
            lineterm="",
        )
    )
