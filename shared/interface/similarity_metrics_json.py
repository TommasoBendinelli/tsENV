from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import parse_qs, urlparse

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "similarity_metrics.schema.json"
)

_DETECTABILITY_VALUES = {"yes", "no", "error"}


class SimilarityMetricsSchemaError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def _schema_validator() -> Draft7Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft7Validator(schema)


def validate_similarity_metrics_schema(payload: Any) -> None:
    try:
        _schema_validator().validate(payload)
    except JsonSchemaValidationError as exc:
        loc = list(exc.path)
        raise SimilarityMetricsSchemaError(
            f"eligibility_metrics.json schema validation failed at {loc}: {exc.message}"
        ) from exc


def _validate_webapp_run_url(value: object, *, context: str) -> None:
    text = str(value or "").strip()
    parsed = urlparse(text)
    if (
        parsed.scheme != "http"
        or parsed.netloc != "localhost:3001"
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(f"url must be a http://localhost:3001 deeplink ({context})")
    query = parse_qs(parsed.query, keep_blank_values=True)
    if query.get("compare") != ["none"]:
        raise ValueError(f"url must include compare=none ({context})")
    if set(query) != {"model", "run", "compare"}:
        raise ValueError(f"url must contain only model, run, and compare ({context})")
    if len(query.get("model", [])) != 1 or not str(query["model"][0]).strip():
        raise ValueError(f"url must include a non-empty model parameter ({context})")
    if len(query.get("run", [])) != 1 or not str(query["run"][0]).strip():
        raise ValueError(f"url must include a non-empty run parameter ({context})")


def validate_similarity_metrics_semantics(
    payload: Any,
    *,
    is_baseline_a_class: Optional[bool] = None,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError("similarity_metrics must be a dict")
    _ = is_baseline_a_class
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        return
    total_baselines = payload.get("total_baselines")
    if (
        isinstance(total_baselines, bool)
        or not isinstance(total_baselines, int)
        or total_baselines < 0
    ):
        raise ValueError("total_baselines must be a non-negative integer")
    if int(total_baselines) != len(baselines):
        raise ValueError("total_baselines must equal the number of baselines")
    eligible_baselines = payload.get("eligible_baselines")
    if (
        isinstance(eligible_baselines, bool)
        or not isinstance(eligible_baselines, int)
        or eligible_baselines < 0
    ):
        raise ValueError("eligible_baselines must be a non-negative integer")
    actual_eligible_baselines = sum(
        1
        for baseline in baselines.values()
        if isinstance(baseline, dict) and baseline.get("family_eligible") is True
    )
    if int(eligible_baselines) != actual_eligible_baselines:
        raise ValueError(
            "eligible_baselines must equal the number of baselines with family_eligible=true"
        )
    for baseline_id, baseline in baselines.items():
        if not isinstance(baseline, dict):
            continue
        _validate_webapp_run_url(
            baseline.get("url"),
            context=f"baseline={baseline_id}",
        )
        if "rule_eval" in baseline:
            raise ValueError(f"rule_eval is not allowed (baseline={baseline_id})")
        if "evaluate_rule" in baseline:
            raise ValueError(f"evaluate_rule is not allowed (baseline={baseline_id})")

        children = baseline.get("children")
        if not isinstance(children, dict):
            continue
        for child_id, child in children.items():
            if not isinstance(child, dict):
                continue
            if "rule_eval" in child:
                raise ValueError(
                    f"rule_eval is not allowed (baseline={baseline_id}, child={child_id})"
                )
            if "evaluate_rule" in child:
                raise ValueError(
                    f"evaluate_rule is not allowed (baseline={baseline_id}, child={child_id})"
                )
            _validate_webapp_run_url(
                child.get("url"),
                context=f"baseline={baseline_id}, child={child_id}",
            )


class DetectabilityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_euclidean_distance_clean_dirty: list[float] = Field(default_factory=list)
    mean_euclidean_distance_clean_baseline: list[float] = Field(default_factory=list)
    mean_SNR: list[Optional[float]] = Field(default_factory=list)
    first_diff: list[Optional[float]] = Field(default_factory=list)

    @field_validator(
        "mean_euclidean_distance_clean_dirty",
        "mean_euclidean_distance_clean_baseline",
        mode="before",
    )
    @classmethod
    def _coerce_float_list(cls, value: Any) -> list[float]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("detectability metrics must be lists")
        out: list[float] = []
        for item in value:
            parsed = float(item)
            if parsed != parsed or parsed in (float("inf"), float("-inf")):
                raise ValueError("detectability metrics must be finite")
            out.append(parsed)
        return out

    @field_validator("mean_SNR", "first_diff", mode="before")
    @classmethod
    def _coerce_nullable_float_list(cls, value: Any) -> list[Optional[float]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("detectability metrics must be lists")
        out: list[Optional[float]] = []
        for item in value:
            if item is None:
                out.append(None)
                continue
            parsed = float(item)
            if parsed != parsed or parsed in (float("inf"), float("-inf")):
                raise ValueError("detectability metrics must be finite or null")
            out.append(parsed)
        return out

    @model_validator(mode="after")
    def _validate_lengths(self) -> "DetectabilityOutput":
        if len(self.mean_euclidean_distance_clean_dirty) != len(self.first_diff):
            raise ValueError(
                "mean_euclidean_distance_clean_dirty and first_diff must have the same length"
            )
        if len(self.mean_euclidean_distance_clean_baseline) != len(self.first_diff):
            raise ValueError(
                "mean_euclidean_distance_clean_baseline and first_diff must have the same length"
            )
        if len(self.mean_SNR) != len(self.first_diff):
            raise ValueError("mean_SNR and first_diff must have the same length")
        return self


class BaselineDetectabilityStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment_specific_detectability: str
    detectable: str
    detectability_output: DetectabilityOutput

    @field_validator(
        "environment_specific_detectability",
        "detectable",
        mode="before",
    )
    @classmethod
    def _validate_detectability_status(cls, value: Any) -> str:
        detectable = str(value or "").strip().lower()
        if detectable not in _DETECTABILITY_VALUES:
            raise ValueError(
                f"detectability status must be one of {sorted(_DETECTABILITY_VALUES)!r}"
            )
        return detectable


class Time0DetectabilityStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detectable: str
    detectability_output: DetectabilityOutput

    @field_validator("detectable", mode="before")
    @classmethod
    def _validate_detectable(cls, value: Any) -> str:
        detectable = str(value or "").strip().lower()
        if detectable not in _DETECTABILITY_VALUES:
            raise ValueError(
                f"detectability status must be one of {sorted(_DETECTABILITY_VALUES)!r}"
            )
        return detectable


class DetectabilitySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vs_baseline: BaselineDetectabilityStatus
    vs_time0_baseline: Time0DetectabilityStatus


class ChildSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    detectability: DetectabilitySummary
    eligible: bool

    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("url must be non-empty")
        return text


class BaselineSimilarity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    family_eligible: bool
    eligible: bool
    children: Dict[str, ChildSimilarity] = Field(default_factory=dict)

    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("url must be non-empty")
        return text


class SimilarityMetricsJson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str
    noise_adder_md5: Optional[str] = None
    eligible_baselines: int = Field(ge=0)
    total_baselines: int = Field(ge=0)
    baselines: Dict[str, BaselineSimilarity] = Field(default_factory=dict)

    @field_validator("timestamp", mode="before")
    @classmethod
    def _validate_timestamp(cls, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("timestamp must be a non-empty UTC ISO 8601 string")
        return text

    @field_validator("noise_adder_md5", mode="before")
    @classmethod
    def _validate_optional_hash(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value or "").strip().lower()
        if not text:
            return None
        if len(text) != 32 or any(ch not in "0123456789abcdef" for ch in text):
            raise ValueError("md5 hash must be a 32-character lowercase hex string")
        return text

    @model_validator(mode="after")
    def _validate_eligible_baselines(self) -> "SimilarityMetricsJson":
        if self.total_baselines != len(self.baselines):
            raise ValueError("total_baselines must equal the number of baselines")
        if self.eligible_baselines != sum(
            1 for baseline in self.baselines.values() if baseline.family_eligible is True
        ):
            raise ValueError(
                "eligible_baselines must equal the number of family-eligible baselines"
            )
        return self


def load_similarity_metrics_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_similarity_metrics_schema(payload)
    validate_similarity_metrics_semantics(payload)
    parsed = SimilarityMetricsJson.model_validate(payload)
    return parsed.model_dump(mode="json", exclude_unset=True, by_alias=True)


def dump_similarity_metrics_json(
    path: Union[str, Path],
    payload: Dict[str, Any],
    *,
    indent: int = 2,
) -> None:
    path = Path(path)
    if not isinstance(payload, dict):
        raise TypeError("similarity_metrics must be a dict")
    validate_similarity_metrics_schema(payload)
    validate_similarity_metrics_semantics(payload)
    parsed = SimilarityMetricsJson.model_validate(payload)
    path.write_text(
        json.dumps(
            parsed.model_dump(mode="json", exclude_unset=True, by_alias=True),
            indent=indent,
        ),
        encoding="utf-8",
    )


__all__ = [
    "BaselineDetectabilityStatus",
    "BaselineSimilarity",
    "ChildSimilarity",
    "DetectabilityOutput",
    "DetectabilitySummary",
    "SimilarityMetricsJson",
    "SimilarityMetricsSchemaError",
    "Time0DetectabilityStatus",
    "ValidationError",
    "dump_similarity_metrics_json",
    "load_similarity_metrics_json",
    "validate_similarity_metrics_semantics",
    "validate_similarity_metrics_schema",
]
