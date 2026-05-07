"""Shared helpers for producing Terminal-Bench-compatible final answers."""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class FinalAnswerFormatter(Protocol):
    def __call__(
        self,
        pred: int,
        choices: Sequence[Any],
        *,
        benchmark_name: str | None = None,
        with_prefix: bool = False,
    ) -> str:  # pragma: no cover - interface
        ...


def format_final_answer(
    pred: int,
    choices: Sequence[Any],
    *,
    benchmark_name: str | None = None,
    with_prefix: bool = False,
) -> str:
    """
    Normalize predictions into the canonical `Final answer: <label>` payload.

    The UCR datasets expect the raw numeric label, and the enrichment pipeline
    extracts digits with a regex, so avoid decorating the answer with letters
    or extra text unless explicitly requested via with_prefix (useful for
    writing Terminal-Bench-style logs).
    """
    _ = benchmark_name
    text: str
    if choices:
        index = int(pred)
        if 0 <= index < len(choices):
            text = str(choices[index])
        else:
            text = str(index)
    else:
        text = str(int(pred))
    if with_prefix:
        return f"Final answer: {text}"
    return text
