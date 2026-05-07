from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Dict, List, Union

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "schemas"
    / "simulink_generated_metadata.schema.json"
)


class SimulinkGeneratedMetadataSchemaError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def _schema_validator() -> Draft7Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft7Validator(schema)


def validate_simulink_generated_metadata_schema(payload: Any) -> None:
    try:
        _schema_validator().validate(payload)
    except JsonSchemaValidationError as exc:
        loc = list(exc.path)
        raise SimulinkGeneratedMetadataSchemaError(
            f"generated/metadata.json schema validation failed at {loc}: {exc.message}"
        ) from exc
    _validate_metadata_payload_consistency(payload)


def _validate_metadata_payload_consistency(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    parameter_set = payload.get("parameter_set")
    intervention_block_map = payload.get("intervention_block_map")
    default_values = payload.get("default_values")
    if not (
        isinstance(parameter_set, list)
        and isinstance(intervention_block_map, dict)
        and isinstance(default_values, dict)
    ):
        return

    params = {str(name) for name in parameter_set}
    block_keys = set(intervention_block_map.keys())
    default_keys = set(default_values.keys())
    expected_default_keys = params | {"end_time_input_s"}

    if params != block_keys:
        raise SimulinkGeneratedMetadataSchemaError(
            "parameter_set must match intervention_block_map keys"
        )
    if expected_default_keys != default_keys:
        raise SimulinkGeneratedMetadataSchemaError(
            "default_values keys must match parameter_set plus end_time_input_s"
        )


class InterventionBlockParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    name: str
    expression: str
    runtime_type: bool


class InterventionBlockMapEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: List[InterventionBlockParameter]


class SimulinkGeneratedMetadataBase(BaseModel):
    """Base schema for `models/simulink/*/generated/metadata.json`."""

    model_config = ConfigDict(extra="forbid")

    parameter_set: List[str]
    intervention_block_map: Dict[str, InterventionBlockMapEntry]
    default_values: Dict[str, float]
    simulink_signals_available: List[str]
    simscape_signals_available: List[str]

    @field_validator("simulink_signals_available", mode="after")
    @classmethod
    def _validate_simulink_signals_available(cls, value: List[str]) -> List[str]:
        invalid = [signal for signal in value if " " in signal]
        if invalid:
            raise ValueError(
                "simulink_signals_available entries must contain no spaces: "
                + ", ".join(repr(signal) for signal in invalid)
            )
        return value

    @model_validator(mode="after")
    def _validate_parameter_key_consistency(self) -> "SimulinkGeneratedMetadataBase":
        params = set(self.parameter_set)
        block_keys = set(self.intervention_block_map.keys())
        default_keys = set(self.default_values.keys())
        expected_default_keys = params | {"end_time_input_s"}

        if params != block_keys:
            raise ValueError("parameter_set must match intervention_block_map keys")
        if expected_default_keys != default_keys:
            raise ValueError(
                "default_values keys must match parameter_set plus end_time_input_s"
            )

        return self


class SimulinkGeneratedMetadata(SimulinkGeneratedMetadataBase):
    """Ground-truth schema for `generated/metadata.json`."""

    model_config = ConfigDict(extra="forbid")


def load_simulink_generated_metadata(
    path: Union[str, Path],
) -> SimulinkGeneratedMetadataBase:
    """Load and validate a `generated/metadata.json` file."""

    metadata_path = Path(path)
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_simulink_generated_metadata_schema(raw)
    return SimulinkGeneratedMetadata.model_validate(raw)


__all__ = [
    "InterventionBlockMapEntry",
    "InterventionBlockParameter",
    "SimulinkGeneratedMetadata",
    "SimulinkGeneratedMetadataBase",
    "ValidationError",
    "SimulinkGeneratedMetadataSchemaError",
    "load_simulink_generated_metadata",
    "validate_simulink_generated_metadata_schema",
]
