from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Tuple, Union

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

SamplingType = Literal["uniform", "loguniform"]
ObservableSignalType = Literal["continuous", "impulse_like"]

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "experiment_config.schema.json"
)


class ExperimentConfigSchemaError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def _schema_validator() -> Draft7Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft7Validator(schema)


def validate_experiment_config_schema(payload: Any) -> None:
    if isinstance(payload, Mapping):
        detectability = payload.get("detectability")
        if isinstance(detectability, Mapping) and "signal_to_ratio" in detectability:
            raise ExperimentConfigSchemaError(
                "experiment_config.json uses legacy detectability.signal_to_ratio; "
                "use detectability.signal_to_noise_ratio_db_thresholds instead"
            )
    try:
        _schema_validator().validate(payload)
    except JsonSchemaValidationError as exc:
        loc = list(exc.path)
        raise ExperimentConfigSchemaError(
            f"experiment_config.json schema validation failed at {loc}: {exc.message}"
        ) from exc


class IntervalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_intervals: Tuple[float, float]
    sampling_strategy: SamplingType

    @field_validator("allowed_intervals", mode="before")
    @classmethod
    def _coerce_interval_tuple(cls, value: Any) -> Tuple[float, float]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            lo, hi = value
            return (float(lo), float(hi))
        raise TypeError("allowed_intervals must be a length-2 list/tuple of numbers")

    @model_validator(mode="after")
    def _validate_interval(self) -> "IntervalSpec":
        lo, hi = self.allowed_intervals
        if lo > hi:
            raise ValueError("allowed_intervals must satisfy low <= high")
        if self.sampling_strategy == "loguniform" and lo <= 0:
            raise ValueError("loguniform allowed_intervals must have low > 0")
        return self


class VariableIntervalSpec(IntervalSpec):
    min_srd_distance: float
    min_abs_dist: float

    @field_validator("min_srd_distance", "min_abs_dist", mode="before")
    @classmethod
    def _coerce_non_negative_threshold(cls, value: Any) -> float:
        parsed = float(value)
        if parsed < 0:
            raise ValueError("parameter threshold must be >= 0")
        return parsed


class ParameterIntervalSpec(VariableIntervalSpec):
    pass


class ExposedVariablesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_state: Dict[str, VariableIntervalSpec] = Field(default_factory=dict)
    parameters: Dict[str, ParameterIntervalSpec] = Field(default_factory=dict)

    @field_validator("initial_state", "parameters", mode="before")
    @classmethod
    def _coerce_named_interval_map(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("exposed_variables entries must be objects keyed by variable name")
        cleaned: Dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("exposed_variables keys must be non-empty")
            cleaned[key] = raw_value
        return cleaned


class ObservableSignalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ObservableSignalType

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_signal_type(cls, value: Any) -> ObservableSignalType:
        signal_type = str(value).strip().lower().replace("-", "_")
        if signal_type not in {"continuous", "impulse_like"}:
            raise ValueError(
                "observable_signals.signal_type values must define type as "
                "'continuous' or 'impulse_like'"
            )
        return signal_type  # type: ignore[return-value]


class ObservableSignalsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observable_signals: list[str]
    signal_type: Dict[str, ObservableSignalSpec]

    @field_validator("observable_signals", mode="before")
    @classmethod
    def _coerce_observable_signals(cls, value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("observable_signals.observable_signals must be an array of non-empty signal names")
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw_item in value:
            item = str(raw_item).strip()
            if not item:
                raise ValueError("observable_signals entries must be non-empty strings")
            if item == "time":
                raise ValueError("observable_signals must not include `time`")
            if item in seen:
                raise ValueError("observable_signals entries must be unique")
            seen.add(item)
            cleaned.append(item)
        return cleaned

    @field_validator("signal_type", mode="before")
    @classmethod
    def _coerce_signal_type(
        cls, value: Any
    ) -> Dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("observable_signals.signal_type must be an object")
        cleaned: Dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("observable_signals.signal_type keys must be non-empty")
            if not isinstance(raw_value, dict):
                raise TypeError(
                    "observable_signals.signal_type values must be objects with "
                    "type"
                )
            cleaned[key] = raw_value
        return cleaned

    @model_validator(mode="after")
    def _validate_signal_type_keys(self) -> "ObservableSignalsConfig":
        signal_set = set(self.observable_signals)
        type_keys = set(self.signal_type.keys())
        if signal_set != type_keys:
            missing = sorted(signal_set - type_keys)
            extra = sorted(type_keys - signal_set)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(
                "observable_signals.signal_type keys must match observable_signals exactly"
                + (f" ({', '.join(details)})" if details else "")
            )
        return self


class ContinuousDetectabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_srd_distance: float
    epsilon_SRD: float
    minimum_consecurive_below_SRD: int

    @field_validator(
        "min_srd_distance",
        "epsilon_SRD",
        mode="before",
    )
    @classmethod
    def _coerce_non_negative_float(cls, value: Any) -> float:
        parsed = float(value)
        if parsed < 0:
            raise ValueError("detectability value must be >= 0")
        return parsed

    @field_validator("minimum_consecurive_below_SRD", mode="before")
    @classmethod
    def _coerce_non_negative_int(cls, value: Any) -> int:
        parsed = int(value)
        if parsed < 0:
            raise ValueError("detectability value must be >= 0")
        return parsed


class ImpulseLikeDetectabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_srd_distance: float
    epsilon_SRD: float

    @field_validator(
        "min_srd_distance",
        "epsilon_SRD",
        mode="before",
    )
    @classmethod
    def _coerce_non_negative_float(cls, value: Any) -> float:
        parsed = float(value)
        if parsed < 0:
            raise ValueError("detectability value must be >= 0")
        return parsed


class DetectabilityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continuous: ContinuousDetectabilityConfig
    impulse_like: ImpulseLikeDetectabilityConfig
    signal_to_noise_ratio_db_thresholds: Dict[str, Tuple[float, float]] = Field(
        default_factory=dict
    )

    @field_validator("signal_to_noise_ratio_db_thresholds", mode="before")
    @classmethod
    def _coerce_signal_to_noise_ratio_db_thresholds(
        cls, value: Any
    ) -> Dict[str, Tuple[float, float]]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("signal_to_noise_ratio_db_thresholds must be an object")
        out: Dict[str, Tuple[float, float]] = {}
        for raw_signal, raw_thresholds in value.items():
            signal = str(raw_signal).strip()
            if not signal:
                raise ValueError(
                    "signal_to_noise_ratio_db_thresholds keys must be non-empty"
                )
            if (
                not isinstance(raw_thresholds, (list, tuple))
                or len(raw_thresholds) != 2
            ):
                raise ValueError(
                    "signal_to_noise_ratio_db_thresholds values must be [low_db, high_db]"
                )
            low = float(raw_thresholds[0])
            high = float(raw_thresholds[1])
            if not math.isfinite(low) or not math.isfinite(high):
                raise ValueError(
                    "signal_to_noise_ratio_db_thresholds thresholds must be finite"
                )
            out[signal] = (low, high)
        return out


class DistributionJson(BaseModel):
    """Schema for experiment configuration stored alongside models."""

    model_config = ConfigDict(extra="forbid")

    exposed_variables: ExposedVariablesConfig
    sampling_rate_hz: float
    end_time_input_s: float
    observable_signals: ObservableSignalsConfig
    detectability: DetectabilityConfig

    @field_validator("sampling_rate_hz", "end_time_input_s", mode="before")
    @classmethod
    def _coerce_positive_float(cls, value: Any) -> float:
        parsed = float(value)
        if parsed <= 0:
            raise ValueError("configuration value must be > 0")
        return parsed

    @property
    def observable_signal_names(self) -> list[str]:
        return list(self.observable_signals.observable_signals)

    @property
    def intervention_parameter_names(self) -> list[str]:
        return sorted(self.exposed_variables.parameters.keys(), key=str.casefold)

    @property
    def observable_signal_types(self) -> Dict[str, ObservableSignalType]:
        return {
            signal: spec.type
            for signal, spec in self.observable_signals.signal_type.items()
        }

    @property
    def min_srd_distance(self) -> float:
        return self.detectability.continuous.min_srd_distance

    @property
    def epsilon_SRD(self) -> float:
        return self.detectability.continuous.epsilon_SRD

    @property
    def minimum_consecurive_below_SRD(self) -> int:
        return self.detectability.continuous.minimum_consecurive_below_SRD


def load_distribution_json(path: Union[str, Path]) -> DistributionJson:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_experiment_config_schema(payload)
    return DistributionJson.model_validate(payload)


def load_experiment_config_json(path: Union[str, Path]) -> DistributionJson:
    return load_distribution_json(path)


__all__ = [
    "DistributionJson",
    "DetectabilityConfig",
    "ExposedVariablesConfig",
    "ContinuousDetectabilityConfig",
    "ImpulseLikeDetectabilityConfig",
    "IntervalSpec",
    "ParameterIntervalSpec",
    "VariableIntervalSpec",
    "ObservableSignalSpec",
    "ObservableSignalType",
    "ObservableSignalsConfig",
    "SamplingType",
    "ValidationError",
    "ExperimentConfigSchemaError",
    "load_distribution_json",
    "load_experiment_config_json",
    "validate_experiment_config_schema",
]
