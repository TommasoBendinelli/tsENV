from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError, field_validator


class FinalMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    average_top1_accuracy: float
    average_shortlist_score: float
    average_num_answers: float

    @field_validator("average_top1_accuracy", "average_shortlist_score", mode="before")
    @classmethod
    def _coerce_unit_metric_value(cls, value: Any) -> float:
        numeric = _coerce_float(value, "final metric values")
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError("final metric values must be between 0 and 1")
        return numeric

    @field_validator("average_num_answers", mode="before")
    @classmethod
    def _coerce_average_num_answers(cls, value: Any) -> float:
        numeric = _coerce_float(value, "average_num_answers")
        if numeric < 0.0:
            raise ValueError("average_num_answers must be non-negative")
        return numeric


class SampleResultEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: List[str]
    top1_correct: StrictBool
    shortlist_score: float
    num_answers: Optional[float]
    sample_type: Literal["test", "other"]
    error: Optional[str] = None

    @field_validator("predictions", mode="before")
    @classmethod
    def _coerce_predictions(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            raise TypeError("predictions must be a list of strings")
        labels: List[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise TypeError(
                    f"predictions[{idx}] must be a non-empty string"
                )
            labels.append(item)
        if len(set(labels)) != len(labels):
            raise TypeError("predictions list must not contain duplicate labels")
        return labels

    @field_validator("shortlist_score", mode="before")
    @classmethod
    def _coerce_shortlist_score(cls, value: Any) -> float:
        numeric = _coerce_float(value, "shortlist_score")
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError("shortlist_score must be between 0 and 1")
        return numeric

    @field_validator("num_answers", mode="before")
    @classmethod
    def _coerce_num_answers(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        numeric = _coerce_float(value, "num_answers")
        if numeric < 0.0:
            raise ValueError("num_answers must be non-negative")
        return numeric

    @field_validator("error", mode="before")
    @classmethod
    def _coerce_error(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("error must be a string")
        return value


class ScoresPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent_run_id: str
    is_correct_format: Optional[StrictBool] = None
    final_metric_test: FinalMetric
    sample_results: dict[str, SampleResultEntry]
    final_metric_other: Optional[FinalMetric] = None

    @field_validator("agent_run_id", mode="before")
    @classmethod
    def _require_agent_run_id(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise TypeError("agent_run_id must be a non-empty string")
        return value

    @field_validator("sample_results", mode="before")
    @classmethod
    def _require_sample_results(cls, value: Any) -> Any:
        if not isinstance(value, dict) or not value:
            raise TypeError("sample_results must be a non-empty dict")
        return value

    @property
    def metrics(self) -> Dict[str, float]:
        return {
            "average_top1_accuracy": float(
                self.final_metric_test.average_top1_accuracy
            ),
            "average_shortlist_score": float(
                self.final_metric_test.average_shortlist_score
            ),
            "average_num_answers": float(
                self.final_metric_test.average_num_answers
            ),
        }


def _coerce_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def validate_scores_payload(payload: Any, *, path: str = "scores.json") -> ScoresPayload:
    try:
        return ScoresPayload.model_validate(payload)
    except (ValidationError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scores.json schema at {path}: {exc}") from exc
