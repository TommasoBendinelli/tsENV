from __future__ import annotations

import functools
import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

RunStatus = Literal["not_run", "success", "failed"]
RuntimeRunType = Literal["baseline", "intervention", "time0_baseline"]

_HASH_HEX_LEN = 32
_MODEL_RECORD_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "model_record.schema.json"
)


class ModelRecordSchemaError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def _load_model_record_json_schema() -> Dict[str, Any]:
    return json.loads(_MODEL_RECORD_SCHEMA_PATH.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _model_record_schema_validator() -> Draft7Validator:
    return Draft7Validator(_load_model_record_json_schema())


def validate_model_record_schema(payload: Any) -> None:
    try:
        _model_record_schema_validator().validate(payload)
    except JsonSchemaValidationError as exc:
        loc = list(exc.path)
        raise ModelRecordSchemaError(
            f"ModelRecord JSON schema validation failed at {loc}: {exc.message}"
        ) from exc


def _sha256_32_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_HEX_LEN]


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def _normalize_identity_numbers(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        if value == 0.0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, list):
        return [_normalize_identity_numbers(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_identity_numbers(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _normalize_identity_numbers(item) for key, item in value.items()
        }
    return value


def _normalize_uuid_hex(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    try:
        return uuid.UUID(hex=text).hex
    except Exception as exc:
        raise ValueError(f"{field_name} must be a valid UUID hex string") from exc


def documented_model_run_specs_baseline_parameter_names(
    experiment_config: Any,
) -> set[str]:
    exposed_variables = getattr(experiment_config, "exposed_variables", None)
    if exposed_variables is None:
        raise ValueError("experiment_config.json is missing exposed_variables")
    parameter_names = {
        str(name).strip()
        for name in (getattr(exposed_variables, "parameters", {}) or {}).keys()
        if str(name).strip()
    }
    initial_state_names = {
        str(name).strip()
        for name in (getattr(exposed_variables, "initial_state", {}) or {}).keys()
        if str(name).strip()
    }
    return set(parameter_names | initial_state_names)


def documented_model_run_specs_child_parameter_names(
    experiment_config: Any,
) -> set[str]:
    exposed_variables = getattr(experiment_config, "exposed_variables", None)
    if exposed_variables is None:
        raise ValueError("experiment_config.json is missing exposed_variables")
    return {
        str(name).strip()
        for name in (getattr(exposed_variables, "parameters", {}) or {}).keys()
        if str(name).strip()
    }


def compute_parameters_hash(*, parameters: Dict[str, Any]) -> str:
    return _sha256_32_hex(
        _canonical_json(
            _normalize_identity_numbers(
                parameters if isinstance(parameters, dict) else {}
            )
        )
    )


def compute_child_parameters_hash(
    *,
    parent_parameters_hash: str,
    child_parameters: Dict[str, Any],
    intervention_time: Any,
) -> str:
    payload = {
        "parent_parameters_hash": str(parent_parameters_hash or "").strip(),
        "parameters": _normalize_identity_numbers(
            child_parameters if isinstance(child_parameters, dict) else {}
        ),
        "intervention_time": _normalize_identity_numbers(intervention_time),
    }
    return _sha256_32_hex(_canonical_json(payload))


def compute_time0_baseline_hash(*, child_parameters_hash: str) -> str:
    payload = {
        "parameter_hash": str(child_parameters_hash or "").strip(),
        "kind": "time0",
    }
    return _sha256_32_hex(_canonical_json(payload))


def _normalize_run_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text not in {"not_run", "success", "failed"}:
        raise ValueError(
            "runtime status must be one of 'not_run', 'success', or 'failed'"
        )
    return text


def _normalize_runtime_hash(value: Any) -> str:
    text = str(value or "").strip().lower()
    if len(text) != _HASH_HEX_LEN or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError("hash fields must be 32-character lowercase hex strings")
    return text


def _normalize_runtime_run_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text not in {"baseline", "intervention", "time0_baseline"}:
        raise ValueError(
            "run_type must be one of 'baseline', 'intervention', or 'time0_baseline'"
        )
    return text


def _normalize_runtime_entry(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError("model_record.json entries must be objects")
    out: Dict[str, Any] = {}
    out["parameters_hash"] = _normalize_runtime_hash(item.get("parameters_hash"))
    out["run_type"] = _normalize_runtime_run_type(item.get("run_type"))
    out["class_internal"] = str(item.get("class_internal") or "").strip()
    out["class_agent_facing_name"] = str(
        item.get("class_agent_facing_name") or ""
    ).strip()
    out["status"] = _normalize_run_status(item.get("status") or "not_run")
    out["timestamp"] = str(item.get("timestamp") or "").strip()
    end_time = item.get("end_time_simulation")
    if end_time not in (None, ""):
        out["end_time_simulation"] = float(end_time)
    error = item.get("error")
    if error not in (None, ""):
        out["error"] = str(error)
    reserved = {
        "parameters_hash",
        "run_type",
        "class_internal",
        "class_agent_facing_name",
        "status",
        "timestamp",
        "end_time_simulation",
        "error",
    }
    for key, value in item.items():
        if key in reserved:
            continue
        out[str(key)] = value
    return out


def _validate_runtime_map(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("model_record.json must be a JSON object")
    out: Dict[str, Any] = {}
    for run_id, item in payload.items():
        run_uuid = _normalize_uuid_hex(run_id, field_name="run_id")
        if not run_uuid:
            raise ValueError("model_record.json contains an empty run_id key")
        out[run_uuid] = _normalize_runtime_entry(item)
    return out


def load_model_record_json(path: Union[str, Path]) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "baselines" in payload:
        raise TypeError("model_record.json must be a flat runtime map, not a baselines list")
    return _validate_runtime_map(payload)


def dump_model_record_json(
    path: Union[str, Path],
    model_record: Dict[str, Any],
    *,
    indent: int = 2,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    runtime_map = _validate_runtime_map(model_record)
    target.write_text(json.dumps(runtime_map, indent=indent), encoding="utf-8")


def load_model_run_specs_json(
    path: Union[str, Path],
    *,
    enforce_baseline_pair_diversity: bool = True,
) -> Dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raise TypeError("model_run_specs.json must be a UUID-keyed object, not a list")
    if not isinstance(payload, dict):
        raise TypeError("model_run_specs.json must be a JSON object")
    if not payload:
        raise ValueError("model_run_specs.json top-level object must be non-empty")
    try:
        import refresh_model_run_spec_hashes as refresh
    except Exception as exc:
        raise RuntimeError("Failed to import model_run_specs refresh validator") from exc

    prepared_payload = refresh._prepare_payload(
        path,
        enforce_baseline_pair_diversity=enforce_baseline_pair_diversity,
    )
    hash_uniqueness_errors = refresh._collect_hash_uniqueness_errors(
        [(path, prepared_payload)]
    )
    if hash_uniqueness_errors:
        raise ValueError(hash_uniqueness_errors[0])
    return prepared_payload


__all__ = [
    "ModelRecordSchemaError",
    "RunStatus",
    "compute_child_parameters_hash",
    "compute_parameters_hash",
    "compute_time0_baseline_hash",
    "documented_model_run_specs_baseline_parameter_names",
    "documented_model_run_specs_child_parameter_names",
    "dump_model_record_json",
    "load_model_record_json",
    "load_model_run_specs_json",
    "validate_model_record_schema",
]
