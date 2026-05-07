from __future__ import annotations

from typing import Any


def is_success_status(value: Any) -> bool:
    return str(value or "").strip().lower() == "success"


def is_question_eligible(
    *, run_status: Any, intervention_status: Any, skipped: bool
) -> bool:
    return bool(
        is_success_status(run_status)
        and is_success_status(intervention_status)
        and not bool(skipped)
    )


__all__ = ["is_success_status", "is_question_eligible"]
