from __future__ import annotations

import inspect
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping

from shared.model_noise_adder import load_noise_adder_from_path


DESCRIPTION_PREFIXES = ("by simulating",)
REQUIRED_ENVIRONMENT_PROFILE_FILES = (
    "description_levels.json",
    "experiment_config.json",
    "noise_adder.py",
    "detectability_specific_environment.py",
    "features.py",
)


class EnvironmentProfileValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EnvironmentProfileValidationResult:
    observable_signals: List[str]
    description_parameter_mapping: Dict[str, str]
    all_signal_names: List[str]


def validate_environment_profile_required_files(model_dir: Path) -> None:
    missing = [
        file_name
        for file_name in REQUIRED_ENVIRONMENT_PROFILE_FILES
        if not (model_dir / file_name).exists()
    ]
    if missing:
        raise EnvironmentProfileValidationError(
            f"{model_dir.name} is missing required environment profile files: "
            + ", ".join(missing)
        )


def validate_environment_profile_config_keys(config_path: Path) -> None:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnvironmentProfileValidationError(
            f"Invalid experiment_config.json at {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EnvironmentProfileValidationError(
            f"experiment_config.json must be a JSON object: {config_path}"
        )
    paper_facing_name = str(payload.get("paper_facing_name") or "").strip()
    if not paper_facing_name:
        raise EnvironmentProfileValidationError(
            f"experiment_config.json must define non-empty paper_facing_name: {config_path}"
        )
    detectability = payload.get("detectability")
    if not isinstance(detectability, dict):
        raise EnvironmentProfileValidationError(
            f"experiment_config.json must define a detectability object: {config_path}"
        )
    if "signal_to_noise_ratio_db_thresholds" in detectability:
        raise EnvironmentProfileValidationError(
            "experiment_config.json uses legacy "
            "detectability.signal_to_noise_ratio_db_thresholds; "
            f"use detectability.RMS_thresholds instead: {config_path}"
        )
    required_by_profile = {
        "continuous": (
            "min_srd_distance",
            "epsilon_SRD",
            "minimum_consecurive_below_SRD",
        ),
        "impulse_like": (
            "min_srd_distance",
            "epsilon_SRD",
        ),
    }
    missing: list[str] = []
    for profile, keys in required_by_profile.items():
        profile_payload = detectability.get(profile)
        if not isinstance(profile_payload, dict):
            missing.append(profile)
            continue
        missing.extend(
            f"{profile}.{key}"
            for key in keys
            if key not in profile_payload
        )
    if missing:
        raise EnvironmentProfileValidationError(
            f"experiment_config.json must define canonical detectability keys "
            f"{missing}: {config_path}"
        )
    rms_thresholds = detectability.get("RMS_thresholds")
    if not isinstance(rms_thresholds, dict) or not rms_thresholds:
        raise EnvironmentProfileValidationError(
            "experiment_config.json detectability.RMS_thresholds must be a "
            f"non-empty object: {config_path}"
        )
    for signal, threshold in rms_thresholds.items():
        try:
            parsed_threshold = float(threshold)
        except (TypeError, ValueError):
            parsed_threshold = float("nan")
        if (
            not isinstance(signal, str)
            or not signal.strip()
            or not math.isfinite(parsed_threshold)
            or parsed_threshold < 0.0
        ):
            raise EnvironmentProfileValidationError(
                "experiment_config.json detectability.RMS_thresholds must map "
                f"signals to finite non-negative numbers: {config_path}"
            )


def load_description_levels_payload(levels_path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(levels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnvironmentProfileValidationError(
            f"Invalid description_levels.json at {levels_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise EnvironmentProfileValidationError(
            f"description_levels.json must be a JSON object: {levels_path}"
        )
    textual_description = payload.get("textual_description")
    if not isinstance(textual_description, dict):
        raise EnvironmentProfileValidationError(
            f"description_levels.json must define a textual_description object: {levels_path}"
        )
    for field_name in ("low", "high"):
        value = str(textual_description.get(field_name) or "").strip()
        if not value.startswith(DESCRIPTION_PREFIXES):
            raise EnvironmentProfileValidationError(
                f"description_levels.json textual_description.{field_name} must begin with "
                f"one of {', '.join(repr(prefix) for prefix in DESCRIPTION_PREFIXES)}: {levels_path}"
            )
    observed_signals_description = payload.get("observed_signals_description")
    if (
        not isinstance(observed_signals_description, str)
        or not observed_signals_description.strip()
    ):
        raise EnvironmentProfileValidationError(
            "description_levels.json must define a non-empty "
            f"'observed_signals_description' string: {levels_path}"
        )
    return payload


def _load_environment_profile_module(path: Path) -> ModuleType:
    try:
        resolved = path.expanduser().resolve()
        spec = importlib.util.spec_from_file_location(
            f"environment_profile_{resolved.stem}_{abs(hash(str(resolved)))}",
            resolved,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to create import spec for {resolved}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise EnvironmentProfileValidationError(
            f"Failed to import environment profile script {path}: {exc}"
        ) from exc


def validate_environment_profile_detectability_hook(model_dir: Path) -> None:
    path = model_dir / "detectability_specific_environment.py"
    module = _load_environment_profile_module(path)
    is_detectable = getattr(module, "is_detectable", None)
    if not callable(is_detectable):
        raise EnvironmentProfileValidationError(
            f"detectability_specific_environment.py must export callable is_detectable: {path}"
        )
    signature = inspect.signature(is_detectable)
    if list(signature.parameters) != [
        "intervention",
        "baseline",
        "run_parameters",
        "intervention_time",
        "min_first_diff",
    ]:
        raise EnvironmentProfileValidationError(
            "detectability_specific_environment.py is_detectable must have "
            "(intervention, baseline, run_parameters, intervention_time, "
            f"min_first_diff): {path}"
        )


def validate_environment_profile_features(model_dir: Path) -> None:
    path = model_dir / "features.py"
    module = _load_environment_profile_module(path)
    feature_names = getattr(module, "FEATURE_NAMES", None)
    if not isinstance(feature_names, (tuple, list)) or not feature_names:
        raise EnvironmentProfileValidationError(
            f"features.py must export a non-empty FEATURE_NAMES tuple/list: {path}"
        )
    invalid = [
        value
        for value in feature_names
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        raise EnvironmentProfileValidationError(
            f"features.py FEATURE_NAMES must contain only non-empty strings: {path}"
        )


def validate_environment_profile_noise_adder(model_dir: Path) -> None:
    path = model_dir / "noise_adder.py"
    try:
        add_noise = load_noise_adder_from_path(path)
    except Exception as exc:
        raise EnvironmentProfileValidationError(
            f"noise_adder.py must follow the documented environment profile interface: {exc}"
        ) from exc
    signature = inspect.signature(add_noise)
    if list(signature.parameters) != ["src", "seed", "noise_level", "ref"]:
        raise EnvironmentProfileValidationError(
            "noise_adder.py add_noise must have documented signature "
            "(src, seed, noise_level, ref): "
            f"{path}"
        )


def load_description_levels_observable_signals(levels_path: Path) -> List[str]:
    payload = load_description_levels_payload(levels_path)
    mapping = payload.get("internal_naming_to_agent_facing_signal")
    if not isinstance(mapping, dict) or not mapping:
        raise EnvironmentProfileValidationError(
            f"description_levels.json must define a non-empty "
            f"'internal_naming_to_agent_facing_signal' mapping: {levels_path}"
        )
    mapping_keys = [str(raw_key or "").strip() for raw_key in mapping.keys()]
    if "time" not in mapping_keys:
        raise EnvironmentProfileValidationError(
            "description_levels.json internal_naming_to_agent_facing_signal must "
            f"include a final 'time' entry: {levels_path}"
        )
    if mapping_keys[-1] != "time":
        raise EnvironmentProfileValidationError(
            "description_levels.json internal_naming_to_agent_facing_signal must "
            f"have 'time' as the final entry: {levels_path}"
        )
    observable_signals = [
        key for key in mapping_keys if key and key != "time"
    ]
    if not observable_signals:
        raise EnvironmentProfileValidationError(
            f"description_levels.json must define at least one non-time observable signal: "
            f"{levels_path}"
        )
    return observable_signals


def load_description_levels_parameter_mapping(levels_path: Path) -> Dict[str, str]:
    payload = load_description_levels_payload(levels_path)
    mapping = payload.get("internal_naming_to_agent_facing_parameter")
    if not isinstance(mapping, dict):
        raise EnvironmentProfileValidationError(
            "description_levels.json must define an "
            f"'internal_naming_to_agent_facing_parameter' mapping: {levels_path}"
        )
    out: Dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "").strip()
        if key and value:
            out[key] = value
    return out


def _metadata_signal_names(metadata: Mapping[str, Any]) -> List[str]:
    all_signal_names: List[str] = []
    for signal in list(metadata.get("simulink_signals_available") or []) + list(
        metadata.get("simscape_signals_available") or []
    ):
        signal_name = str(signal or "").strip()
        if signal_name and signal_name not in all_signal_names:
            all_signal_names.append(signal_name)
    return all_signal_names


def validate_environment_profile_consistency(
    *,
    model_dir: Path,
    experiment_config: Any,
    metadata: Mapping[str, Any],
) -> EnvironmentProfileValidationResult:
    validate_environment_profile_required_files(model_dir)
    validate_environment_profile_noise_adder(model_dir)
    validate_environment_profile_detectability_hook(model_dir)
    validate_environment_profile_features(model_dir)
    description_levels_path = model_dir / "description_levels.json"
    if not description_levels_path.exists():
        raise EnvironmentProfileValidationError(
            f"description_levels.json missing for {model_dir}"
        )

    observable_signals = list(experiment_config.observable_signal_names)
    if not observable_signals:
        raise EnvironmentProfileValidationError(
            f"experiment_config.json must define observable_signals for {model_dir.name}."
        )

    description_level_signals = load_description_levels_observable_signals(
        description_levels_path
    )
    if observable_signals != description_level_signals:
        raise EnvironmentProfileValidationError(
            "experiment_config.json observable_signals must match "
            "description_levels.json internal_naming_to_agent_facing_signal keys "
            f"(excluding 'time'): got {observable_signals} expected {description_level_signals}"
        )

    description_parameter_mapping = load_description_levels_parameter_mapping(
        description_levels_path
    )
    expected_parameter_keys = list(
        (getattr(experiment_config.exposed_variables, "parameters", {}) or {}).keys()
    )
    description_parameter_keys = list(description_parameter_mapping.keys())
    if description_parameter_keys != expected_parameter_keys:
        raise EnvironmentProfileValidationError(
            "description_levels.json internal_naming_to_agent_facing_parameter keys "
            "must match experiment_config.json exposed_variables.parameters exactly "
            f"and in order: got {description_parameter_keys} expected {expected_parameter_keys}"
        )

    available_signals = set(metadata.get("simulink_signals_available") or []) | set(
        metadata.get("simscape_signals_available") or []
    )
    missing_observable_signals = [
        signal for signal in observable_signals if signal not in available_signals
    ]
    if missing_observable_signals:
        raise EnvironmentProfileValidationError(
            "experiment_config.json observable_signals are missing from metadata.json: "
            + ", ".join(missing_observable_signals)
        )

    return EnvironmentProfileValidationResult(
        observable_signals=observable_signals,
        description_parameter_mapping=description_parameter_mapping,
        all_signal_names=_metadata_signal_names(metadata),
    )


__all__ = [
    "DESCRIPTION_PREFIXES",
    "REQUIRED_ENVIRONMENT_PROFILE_FILES",
    "EnvironmentProfileValidationError",
    "EnvironmentProfileValidationResult",
    "load_description_levels_observable_signals",
    "load_description_levels_parameter_mapping",
    "load_description_levels_payload",
    "validate_environment_profile_detectability_hook",
    "validate_environment_profile_features",
    "validate_environment_profile_noise_adder",
    "validate_environment_profile_config_keys",
    "validate_environment_profile_consistency",
    "validate_environment_profile_required_files",
]
