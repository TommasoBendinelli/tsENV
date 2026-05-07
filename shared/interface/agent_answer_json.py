from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class UCRAgentAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    final_answer: str

    @field_validator("final_answer", mode="before")
    @classmethod
    def _require_non_empty_final_answer(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("final_answer must be a non-empty string")
        return value


class TsenvAgentAnswer(BaseModel):
    model_config = ConfigDict(extra="allow")

    change_time: Optional[float] = None
    final_answer: str

    @field_validator("final_answer", mode="before")
    @classmethod
    def _require_non_empty_final_answer(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("final_answer must be a non-empty string")
        return value

    @field_validator("change_time", mode="before")
    @classmethod
    def _coerce_change_time(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            raise ValueError("change_time must be a number, -1, or null")
        numeric = float(value)
        if numeric < 0 and numeric != -1:
            raise ValueError("change_time must be >= 0, -1, or null")
        return numeric


class TsenvDirectAgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: dict[str, str | list[str]]

    @field_validator("predictions", mode="after")
    @classmethod
    def _require_non_empty_unique_samples(cls, value: dict[str, str | list[str]]) -> dict[str, str | list[str]]:
        if not value:
            raise ValueError("predictions must contain at least one entry")
        return value


def validate_ucr_agent_answer(payload: Any, *, path: str = "results.json") -> UCRAgentAnswer:
    try:
        return UCRAgentAnswer.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid UCRAgentAnswer schema at {path}: {exc}") from exc


def validate_tsenv_agent_answer(
    payload: Any, *, path: str = "results.json"
) -> TsenvAgentAnswer:
    try:
        return TsenvAgentAnswer.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid TsenvAgentAnswer schema at {path}: {exc}") from exc


def validate_tsenv_direct_agent_answer(
    payload: Any,
    *,
    expected_samples: Sequence[str] | None = None,
    path: str = "results.json",
) -> TsenvDirectAgentAnswer:
    normalized_expected = [str(sample).strip() for sample in (expected_samples or []) if str(sample).strip()]
    expected_by_key: dict[str, str] = {}
    if normalized_expected:
        for expected in normalized_expected:
            expected_by_key[expected] = expected
            expected_path = Path(expected)
            if expected_path.suffix == ".parquet":
                expected_by_key[expected_path.with_suffix("").as_posix()] = expected
                expected_by_key[expected_path.stem] = expected
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid TsenvDirectAgentAnswer schema at {path}: results.json must be a JSON object")
    normalized_predictions: dict[str, str | list[str]] = {}
    for raw_sample, raw_label in payload.items():
        sample = str(raw_sample).strip()
        if not sample:
            raise ValueError(
                f"Invalid TsenvDirectAgentAnswer schema at {path}: prediction keys must be non-empty strings"
            )
        normalized_sample = expected_by_key.get(sample, sample)
        if normalized_sample in normalized_predictions:
            raise ValueError(
                f"Invalid TsenvDirectAgentAnswer schema at {path}: duplicate prediction keys "
                f"for {normalized_sample!r}; use either the .parquet filename or the stem, not both"
            )
        if isinstance(raw_label, str):
            if not raw_label.strip():
                raise ValueError(
                    f"Invalid TsenvDirectAgentAnswer schema at {path}: prediction for {sample!r} must be non-empty"
                )
            normalized_predictions[normalized_sample] = raw_label.strip()
            continue
        if isinstance(raw_label, list):
            labels = []
            for idx, item in enumerate(raw_label):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        f"Invalid TsenvDirectAgentAnswer schema at {path}: prediction for {sample!r}[{idx}] must be a non-empty string"
                    )
                labels.append(item.strip())
            if not labels:
                raise ValueError(
                    f"Invalid TsenvDirectAgentAnswer schema at {path}: prediction for {sample!r} must contain at least one label"
                )
            if len(set(labels)) != len(labels):
                raise ValueError(
                    f"Invalid TsenvDirectAgentAnswer schema at {path}: prediction for {sample!r} must not contain duplicate labels"
                )
            normalized_predictions[normalized_sample] = labels
            continue
        raise ValueError(
            f"Invalid TsenvDirectAgentAnswer schema at {path}: prediction for {sample!r} must be a non-empty string or list of strings"
        )
    try:
        validated = TsenvDirectAgentAnswer.model_validate(
            {"predictions": normalized_predictions}
        )
    except ValidationError as exc:
        raise ValueError(f"Invalid TsenvDirectAgentAnswer schema at {path}: {exc}") from exc
    if normalized_expected:
        observed_samples = sorted(validated.predictions)
        if set(observed_samples) != set(normalized_expected):
            raise ValueError(
                f"Invalid TsenvDirectAgentAnswer schema at {path}: prediction keys "
                f"{observed_samples!r} do not match expected samples {sorted(normalized_expected)!r}"
            )
    return validated
