from __future__ import annotations

from typing import Literal


TsenvEvalMode = Literal["direct", "code", "open-ended"]

_ALIASES = {
    "direct": "direct",
    "code": "code",
    "open-ended": "open-ended",
}


def normalize_tsenv_eval_mode(
    raw_mode: object,
    *,
    default: TsenvEvalMode = "direct",
) -> TsenvEvalMode:
    lowered = str(raw_mode or "").strip().lower()
    if not lowered:
        return default
    normalized = _ALIASES.get(lowered)
    if normalized is None:
        raise ValueError(
            f"Unsupported tsENV eval_mode={raw_mode!r}; expected one of "
            f"{sorted(_ALIASES)}."
        )
    return normalized


def is_tsenv_code_mode(raw_mode: object) -> bool:
    return normalize_tsenv_eval_mode(raw_mode) == "code"


def is_tsenv_direct_mode(raw_mode: object) -> bool:
    return normalize_tsenv_eval_mode(raw_mode) == "direct"


def is_tsenv_open_ended_mode(raw_mode: object) -> bool:
    return normalize_tsenv_eval_mode(raw_mode) == "open-ended"


__all__ = [
    "TsenvEvalMode",
    "is_tsenv_code_mode",
    "is_tsenv_direct_mode",
    "is_tsenv_open_ended_mode",
    "normalize_tsenv_eval_mode",
]
