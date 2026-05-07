from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import click
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.interface.distribution_json import (
    ValidationError as DistributionValidationError,
    load_experiment_config_json,
)
from shared.interface.model_record_json import (
    dump_model_record_json,
    load_model_record_json,
    load_model_run_specs_json,
)
from shared.interface.simulink_metadata_json import (
    ValidationError as MetadataValidationError,
    load_simulink_generated_metadata,
)
from shared.benchmark_utils import ALLOWED_TSENV_MODELS
from shared.environment_profiles import (
    EnvironmentProfileValidationError,
    load_description_levels_observable_signals,
    load_description_levels_parameter_mapping,
    validate_environment_profile_config_keys,
    validate_environment_profile_consistency,
    validate_environment_profile_required_files,
)
from shared.model_lock import ModelLockError, model_lock
from shared.model_run_specs_runtime import (
    build_expected_runtime_model_record,
    expected_run_ids,
    reconcile_runtime_model_record,
)
from shared.run_artifacts import resolve_model_record_path, resolve_runs_root
from shared.time_series_metrics import assert_signal_prefix as _assert_simulink_signal_prefix
import shared.simulation as simulation_api
import workflows.simulate.build_metadata as bm_module

_MIN_SAMPLED_POINTS = 50
_RUNTIME_HASH_FIELDS = ("parameters_hash",)
_PARENT_RESERVED_PARAMETER_KEYS = (
    "intervention_time",
    "end_time_input_s",
    "sampling_rate_hz",
)
_INTERRUPT_EXCEPTION_NAMES = frozenset({"InterruptedError", "CancelledError"})
@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    kind: str
    parent_id: Optional[str]
    baseline_parameters: Dict[str, Any]
    intervention_parameters: Dict[str, Any]
    intervention_mode: str
    intervention_time: float
    internal_sampling_rate_hz: float
    sampling_rate_hz: float
    end_time_input_s: float


class _TimingRecorder:
    def __init__(self, *, enabled: bool, path: Optional[Path], model_name: str) -> None:
        self.enabled = bool(enabled)
        self.path = path
        self.model_name = model_name
        if self.enabled and self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    @contextmanager
    def span(self, phase: str, *, run_id: Optional[str] = None) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.record(
                phase,
                time.perf_counter() - started,
                run_id=run_id,
                status=status,
            )

    def record(
        self,
        phase: str,
        duration_s: float,
        *,
        run_id: Optional[str] = None,
        status: str = "success",
        **extra: Any,
    ) -> None:
        if not self.enabled or self.path is None:
            return
        payload = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "run_id": run_id,
            "phase": str(phase),
            "duration_s": float(duration_s),
            "status": str(status),
        }
        payload.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def callback(self, *, phase: str, duration_s: float, **extra: Any) -> None:
        run_id = extra.pop("run_id", None)
        status = str(extra.pop("status", "success"))
        self.record(phase, duration_s, run_id=run_id, status=status, **extra)


def _resolve_model_runs_root(
    model_dir: Path,
    *,
    runs_root_base: Optional[Path],
) -> Path:
    if runs_root_base is None:
        return resolve_runs_root(model_dir)
    return Path(runs_root_base).expanduser().resolve() / model_dir.name / "runs"


def _run_refresh_model_run_spec_hashes(*, target: Path, reason: str) -> None:
    command = [
        sys.executable,
        str(root_dir / "refresh_model_run_spec_hashes.py"),
        str(target),
    ]
    print(f"Refreshing model_run_specs hashes ({reason})...")
    try:
        subprocess.run(command, cwd=str(root_dir), check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"refresh_model_run_spec_hashes.py failed for {target}: {exc}"
        ) from exc


def _load_model_run_specs_with_refresh(
    *,
    specs_path: Path,
    model_record_path: Path,
) -> Dict[str, Any]:
    refreshed = False
    if model_record_path.exists() and specs_path.stat().st_mtime > model_record_path.stat().st_mtime:
        _run_refresh_model_run_spec_hashes(
            target=specs_path,
            reason="model_run_specs.json is newer than model_record.json",
        )
        refreshed = True
    try:
        return load_model_run_specs_json(specs_path)
    except ValueError as exc:
        if "mismatch" not in str(exc) or refreshed:
            raise
        _run_refresh_model_run_spec_hashes(
            target=specs_path,
            reason=f"spec hash validation failed: {exc}",
        )
        return load_model_run_specs_json(specs_path)


def _coerce_finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _is_interrupt_exception(exc: BaseException) -> bool:
    if isinstance(exc, KeyboardInterrupt):
        return True
    exc_type = type(exc)
    return (
        exc_type.__module__.startswith("matlab.engine")
        and exc_type.__name__ in _INTERRUPT_EXCEPTION_NAMES
    )


def _propagate_interrupt(
    exc: BaseException,
    *,
    runtime_entry: Dict[str, Any],
    run_dir: Optional[Path] = None,
) -> None:
    if run_dir is not None:
        _remove_run_dir(run_dir)
    _set_runtime_state(runtime_entry, status="not_run")
    if isinstance(exc, KeyboardInterrupt):
        raise exc
    raise KeyboardInterrupt(str(exc)) from exc


def _validate_environment_profile_required_files(model_dir: Path) -> None:
    try:
        validate_environment_profile_required_files(model_dir)
    except EnvironmentProfileValidationError as exc:
        raise SystemExit(str(exc)) from exc


def _validate_environment_profile_config_keys(config_path: Path) -> None:
    try:
        validate_environment_profile_config_keys(config_path)
    except EnvironmentProfileValidationError as exc:
        raise SystemExit(str(exc)) from exc


def _load_description_levels_observable_signals(levels_path: Path) -> List[str]:
    try:
        return load_description_levels_observable_signals(levels_path)
    except EnvironmentProfileValidationError as exc:
        raise SystemExit(str(exc)) from exc


def _load_description_levels_parameter_mapping(levels_path: Path) -> Dict[str, str]:
    try:
        return load_description_levels_parameter_mapping(levels_path)
    except EnvironmentProfileValidationError as exc:
        raise SystemExit(str(exc)) from exc


def _normalize_name_tokens(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    normalized = text.replace("_", " ").replace("-", " ").lower()
    parts = [part for part in normalized.split() if part]
    tokens: set[str] = set()
    for part in parts:
        tokens.add(part)
        if part.endswith("s") and len(part) > 3:
            tokens.add(part[:-1])
    synonym_pairs = {
        "coefficient": {"stiffness"},
        "stiffness": {"coefficient"},
    }
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(synonym_pairs.get(token, set()))
    return expanded


def _build_parameter_alias_map(
    *,
    metadata: Dict[str, Any],
    documented_parameters: Sequence[str],
    agent_facing_parameter_map: Dict[str, str],
) -> Dict[str, str]:
    internal_names = [str(name).strip() for name in metadata.get("parameter_set") or [] if str(name).strip()]
    internal_name_set = set(internal_names)
    if not internal_names:
        return {}

    def _candidate_tokens(internal_name: str) -> set[str]:
        tokens = set(_normalize_name_tokens(internal_name))
        tokens.update(_normalize_name_tokens(agent_facing_parameter_map.get(internal_name)))
        return tokens

    alias_map: Dict[str, str] = {}
    for documented_name in documented_parameters:
        documented = str(documented_name or "").strip()
        if not documented or documented in internal_name_set:
            continue
        documented_tokens = _normalize_name_tokens(documented)
        if not documented_tokens:
            continue
        best_internal = ""
        best_score = 0
        best_tiebreak = -1
        ambiguous = False
        for internal_name in internal_names:
            candidate_tokens = _candidate_tokens(internal_name)
            score = len(documented_tokens & candidate_tokens)
            tiebreak = len(candidate_tokens)
            if score > best_score or (score == best_score and score > 0 and tiebreak > best_tiebreak):
                best_internal = internal_name
                best_score = score
                best_tiebreak = tiebreak
                ambiguous = False
            elif score == best_score and score > 0:
                ambiguous = True
        if best_score <= 0 or ambiguous or not best_internal:
            raise SystemExit(
                f"Unable to map documented intervention parameter '{documented}' to "
                f"metadata.json::parameter_set for this model."
            )
        alias_map[documented] = best_internal
    return alias_map


def _normalize_parameter_dict(
    values: Dict[str, Any],
    *,
    alias_map: Dict[str, str],
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        normalized_key = alias_map.get(key, key)
        normalized[normalized_key] = raw_value
    return normalized


def _resolve_models_root() -> Path:
    cwd_models_root = Path(os.getcwd()).resolve() / "models" / "simulink"
    if cwd_models_root.exists():
        return cwd_models_root
    return root_dir / "models" / "simulink"


def _list_model_dirs(models_root: Path) -> List[Path]:
    if not models_root.exists():
        raise SystemExit(f"Models root {models_root} missing")
    return sorted(
        [path.resolve() for path in models_root.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )


def _allowed_model_dir(model_id: str, *, models_root: Path) -> Path:
    return (models_root / str(model_id).strip()).resolve()


def _resolve_target_model_dirs(*, model: Optional[str]) -> List[Path]:
    models_root = _resolve_models_root()
    if model is not None and str(model).strip():
        model_id = str(model).strip()
        if "/" in model_id or "\\" in model_id:
            raise SystemExit(
                f"Expected a model id (e.g. BallDrop), got a path-like value: {model!r}"
            )
        if model_id not in ALLOWED_TSENV_MODELS:
            raise SystemExit(
                f"Model '{model_id}' is not an allowed tsENV model. "
                "Update shared/benchmark_utils.py (ALLOWED_TSENV_MODELS) to add it."
            )
        model_dir = _allowed_model_dir(model_id, models_root=models_root)
        if not model_dir.exists():
            raise SystemExit(f"Model directory {model_dir} missing")
        return [model_dir]

    model_dirs = [
        model_dir
        for model_id in ALLOWED_TSENV_MODELS
        for model_dir in [_allowed_model_dir(model_id, models_root=models_root)]
        if model_dir.exists()
    ]
    if not model_dirs:
        raise SystemExit(
            f"No allowed tsENV model directories found under {models_root}"
        )
    return model_dirs


def _sampled_point_count(*, sampling_rate_hz: Any, end_time_input_s: Any) -> int:
    sr = _coerce_finite_float(sampling_rate_hz)
    end_time = _coerce_finite_float(end_time_input_s)
    if sr is None or end_time is None or sr <= 0.0 or end_time < 0.0:
        return 0
    return max(0, int(np.floor(end_time * sr)) - 1)


def _assert_run_time_and_sampling_sanity(
    *,
    run_id: str,
    intervention_time: Any,
    sampling_rate_hz: Any,
    configured_end_time_s: float,
) -> None:
    run_label = str(run_id or "").strip() or "<missing_run_id>"
    intervention_time = _coerce_finite_float(intervention_time)
    if intervention_time is None:
        raise ValueError(
            f"Invalid intervention_time for run '{run_label}': must be finite."
        )
    if not (0.0 < intervention_time < float(configured_end_time_s)):
        raise ValueError(
            f"Invalid intervention_time for run '{run_label}': expected "
            "0 < intervention_time < end_time_input_s"
        )
    n_points = _sampled_point_count(
        sampling_rate_hz=sampling_rate_hz,
        end_time_input_s=configured_end_time_s,
    )
    if n_points < _MIN_SAMPLED_POINTS:
        raise ValueError(
            f"Insufficient sampled points for run '{run_label}': requires >= {_MIN_SAMPLED_POINTS}"
        )


def _remove_run_dir(run_dir: Path) -> int:
    if not run_dir.exists():
        return 0
    shutil.rmtree(run_dir)
    return 1


def _load_end_time_from_parquet(run_dir: Path) -> Optional[float]:
    parquet_path = run_dir / "data.parquet"
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["time"])
    except Exception:
        return None
    if "time" not in df.columns or df.empty:
        return None
    return _coerce_finite_float(pd.to_numeric(df["time"], errors="coerce").max())


def _resolve_runtime_hash_entry(record: Dict[str, Any], run_id: str) -> tuple[str, str]:
    entry = record.get(run_id) or {}
    if not isinstance(entry, dict):
        return "", ""
    for field_name in _RUNTIME_HASH_FIELDS:
        value = str(entry.get(field_name) or "").strip()
        if value:
            return field_name, value
    return "", ""


def _documented_parameter_names_from_specs(
    *,
    specs: Mapping[str, Any],
    baseline_parameters_by_id: Dict[str, Dict[str, Any]],
) -> List[str]:
    names: set[str] = set()
    for baseline_uuid, baseline_parameters in baseline_parameters_by_id.items():
        _ = baseline_uuid
        for raw_key in baseline_parameters.keys():
            key = str(raw_key or "").strip()
            if key and key not in _PARENT_RESERVED_PARAMETER_KEYS:
                names.add(key)
    for raw_parent in specs.values():
        if not isinstance(raw_parent, dict):
            continue
        children = raw_parent.get("children")
        if not isinstance(children, dict):
            continue
        for raw_child in children.values():
            if not isinstance(raw_child, dict):
                continue
            parameters = raw_child.get("parameters")
            if not isinstance(parameters, dict):
                continue
            for raw_key in parameters.keys():
                key = str(raw_key or "").strip()
                if key:
                    names.add(key)
    return sorted(names, key=str.casefold)


def _baseline_parameters_from_raw_specs(
    *,
    specs_payload: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for baseline_uuid, raw_parent in specs_payload.items():
        if not isinstance(raw_parent, dict):
            continue
        baseline_parameters = raw_parent.get("baseline_parameters")
        if not isinstance(baseline_parameters, dict):
            continue
        by_id[str(baseline_uuid).strip().lower()] = dict(baseline_parameters)
    return by_id


def _required_execution_key(
    *,
    baseline_uuid: str,
    baseline_parameters: Dict[str, Any],
    key: str,
) -> float:
    value = _coerce_finite_float(baseline_parameters.get(key))
    if value is None:
        raise ValueError(
            f"baseline_parameters for '{baseline_uuid}' must define finite {key!r}"
        )
    return value


def _child_intervention_time(
    *,
    baseline_uuid: str,
    child_uuid: str,
    raw_child: Mapping[str, Any],
) -> float:
    value = _coerce_finite_float(raw_child.get("intervention_time"))
    if value is None:
        raise ValueError(
            f"child '{child_uuid}' under baseline '{baseline_uuid}' must define finite intervention_time"
        )
    return value


def _baseline_planning_intervention_time(
    *,
    baseline_uuid: str,
    children: Mapping[str, Any],
) -> float:
    values: List[float] = []
    for child_uuid, raw_child in children.items():
        if not isinstance(raw_child, Mapping):
            continue
        child_parameters = raw_child.get("parameters")
        if not isinstance(child_parameters, Mapping) or len(child_parameters) != 1:
            continue
        values.append(
            _child_intervention_time(
                baseline_uuid=baseline_uuid,
                child_uuid=str(child_uuid),
                raw_child=raw_child,
            )
        )
    if not values:
        raise ValueError(
            f"baseline '{baseline_uuid}' must have at least one child with finite intervention_time"
        )
    first = values[0]
    return first if all(value == first for value in values) else first


def _build_planned_runs(
    specs: Mapping[str, Any],
    *,
    model_dir: Path,
    baseline_parameters_by_id: Dict[str, Dict[str, Any]],
    experiment_config: Any,
) -> List[PlannedRun]:
    planned_runs: List[PlannedRun] = []
    sampling_rate_hz = float(experiment_config.sampling_rate_hz)
    internal_sampling_rate_hz = simulation_api.resolve_internal_sampling_rate_hz(
        model_dir=model_dir,
        sampling_rate_hz=sampling_rate_hz,
    )
    end_time_input_s = float(experiment_config.end_time_input_s)
    for baseline_uuid, raw_parent in specs.items():
        if not isinstance(raw_parent, dict):
            continue
        baseline_parameters_full = dict(
            baseline_parameters_by_id.get(str(baseline_uuid).strip().lower()) or {}
        )
        baseline_parameters = dict(baseline_parameters_full)
        children = raw_parent.get("children")
        if not isinstance(children, dict):
            children = {}
        baseline_intervention_time = _baseline_planning_intervention_time(
            baseline_uuid=str(baseline_uuid),
            children=children,
        )
        planned_runs.append(
            PlannedRun(
                run_id=str(baseline_uuid),
                kind="baseline",
                parent_id=None,
                baseline_parameters=baseline_parameters,
                intervention_parameters={},
                intervention_mode="none",
                intervention_time=float(baseline_intervention_time),
                internal_sampling_rate_hz=float(internal_sampling_rate_hz),
                sampling_rate_hz=float(sampling_rate_hz),
                end_time_input_s=end_time_input_s,
            )
        )
        for child_uuid, raw_child in children.items():
            if not isinstance(raw_child, dict):
                continue
            child_parameters = raw_child.get("parameters")
            if not isinstance(child_parameters, dict) or len(child_parameters) != 1:
                continue
            parameter, set_value = next(iter(child_parameters.items()))
            if set_value is None:
                continue
            intervention_time = _child_intervention_time(
                baseline_uuid=str(baseline_uuid),
                child_uuid=str(child_uuid),
                raw_child=raw_child,
            )
            intervention_parameters = {str(parameter): set_value}
            planned_runs.append(
                PlannedRun(
                    run_id=str(child_uuid),
                    kind="intervention",
                    parent_id=str(baseline_uuid),
                    baseline_parameters=baseline_parameters,
                    intervention_parameters=intervention_parameters,
                    intervention_mode="at_intervention_time",
                    intervention_time=float(intervention_time),
                    internal_sampling_rate_hz=float(internal_sampling_rate_hz),
                    sampling_rate_hz=float(sampling_rate_hz),
                    end_time_input_s=end_time_input_s,
                )
            )
            time0_baseline_uuid = str(raw_child.get("time0_baseline_uuid") or "").strip()
            if time0_baseline_uuid:
                planned_runs.append(
                    PlannedRun(
                        run_id=time0_baseline_uuid,
                        kind="time-zero baseline",
                        parent_id=str(baseline_uuid),
                        baseline_parameters=baseline_parameters,
                        intervention_parameters=intervention_parameters,
                        intervention_mode="from_time_zero",
                        intervention_time=float(intervention_time),
                        internal_sampling_rate_hz=float(internal_sampling_rate_hz),
                        sampling_rate_hz=float(sampling_rate_hz),
                        end_time_input_s=end_time_input_s,
                    )
                )
    return planned_runs


def _find_models_with_run_id(
    *,
    model_dirs: Sequence[Path],
    run_id: str,
) -> List[Path]:
    normalized_run_id = str(run_id or "").strip().lower()
    matches: List[Path] = []
    for model_dir in model_dirs:
        specs_path = model_dir / "model_run_specs.json"
        if not specs_path.exists():
            continue
        model_record_path = resolve_model_record_path(model_dir)
        specs_payload = _load_model_run_specs_with_refresh(
            specs_path=specs_path,
            model_record_path=model_record_path,
        )
        if normalized_run_id in expected_run_ids(specs_payload):
            matches.append(model_dir.resolve())
    return matches


def _set_runtime_state(
    entry: Dict[str, Any],
    *,
    status: str,
    timestamp: str = "",
    end_time_simulation: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    entry["status"] = status
    entry.pop("timestamp", None)
    entry.pop("end_time_simulation", None)
    entry.pop("error", None)
    if timestamp:
        entry["timestamp"] = timestamp
    if end_time_simulation is not None:
        entry["end_time_simulation"] = end_time_simulation
    if error not in (None, ""):
        entry["error"] = error


def _runtime_end_time(entry: Dict[str, Any]) -> Optional[float]:
    return _coerce_finite_float(entry.get("end_time_simulation"))


@click.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.option(
    "--model",
    default=None,
    help="Optional model name. If omitted, iterate over all allowed tsENV models.",
)
@click.option(
    "--raise-on-errors",
    is_flag=True,
    default=False,
    help="Raise exceptions immediately instead of marking registry statuses as failed.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Mark all runs as not_run and materialize them again.",
)
@click.option(
    "--rerun-failed",
    is_flag=True,
    default=False,
    help="Mark failed runs as not_run and materialize them again.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Persist per-run debug artifacts under runs/<UUID>/debug/.",
)
@click.option(
    "--run-id",
    default=None,
    help="Run and overwrite only the specified UUID entry.",
)
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help=(
        "Optional base directory for run artifacts. Each model uses "
        "<runs-root>/<MODEL>/runs/."
    ),
)
@click.option(
    "--matlab-cache-root",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help=(
        "Optional base directory for MATLAB/Simulink generated cache files. "
        "If omitted, MATLAB uses its normal cache/preference locations."
    ),
)
@click.option(
    "--profile-timing",
    is_flag=True,
    default=False,
    help="Write per-phase timing records to profile_timing.jsonl in the selected run root.",
)
@click.option(
    "--compute-features",
    is_flag=True,
    default=False,
    help=(
        "Compute problem-specific features and write features.json. "
        "Optional feature names may follow the flag."
    ),
)
@click.pass_context
def main(
    ctx: click.Context,
    model: Optional[str],
    raise_on_errors: bool,
    overwrite: bool,
    rerun_failed: bool,
    debug: bool,
    run_id: Optional[str],
    runs_root: Optional[Path],
    matlab_cache_root: Optional[Path],
    profile_timing: bool,
    compute_features: bool,
) -> None:
    feature_names = tuple(str(arg) for arg in ctx.args)
    if feature_names and not compute_features:
        raise click.UsageError(
            "Unexpected extra arguments: "
            + " ".join(feature_names)
            + ". Feature names are only accepted after --compute-features."
        )
    if any(name.startswith("-") for name in feature_names):
        raise click.UsageError(
            "Feature names after --compute-features must not start with '-'."
        )
    run_pipeline(
        model=model,
        raise_on_errors=raise_on_errors,
        overwrite=overwrite,
        rerun_failed=rerun_failed,
        debug=debug,
        run_id=run_id,
        runs_root=runs_root,
        matlab_cache_root=matlab_cache_root,
        profile_timing=profile_timing,
        compute_features=compute_features,
        compute_feature_names=feature_names if compute_features else None,
    )


def run_pipeline(
    *,
    model: Optional[str],
    raise_on_errors: bool,
    overwrite: bool,
    rerun_failed: bool = False,
    debug: bool = False,
    run_id: Optional[str] = None,
    runs_root: Optional[Path] = None,
    matlab_cache_root: Optional[Path] = None,
    profile_timing: bool = False,
    compute_features: bool = False,
    compute_feature_names: Optional[Sequence[str]] = None,
) -> None:
    model_dirs = _resolve_target_model_dirs(model=model)
    normalized_run_id = str(run_id or "").strip().lower() or None
    runs_root_base = Path(runs_root).expanduser().resolve() if runs_root is not None else None
    matlab_cache_root_base = (
        Path(matlab_cache_root).expanduser().resolve()
        if matlab_cache_root is not None
        else None
    )
    if normalized_run_id and model is None:
        matches = _find_models_with_run_id(
            model_dirs=model_dirs,
            run_id=normalized_run_id,
        )
        if not matches:
            raise SystemExit(
                f"run_id '{normalized_run_id}' is not present in any model under {_resolve_models_root()}."
            )
        if len(matches) > 1:
            model_names = ", ".join(path.name for path in matches)
            raise SystemExit(
                f"run_id '{normalized_run_id}' is present in multiple models: {model_names}"
            )
        model_dirs = matches

    try:
        for model_dir in model_dirs:
            try:
                with model_lock(
                    model_dir.name,
                    purpose="run_pending_sims",
                    lock_root=model_dir / ".locks",
                ):
                    run_model_kwargs: Dict[str, Any] = {
                        "raise_on_errors": raise_on_errors,
                        "overwrite": overwrite,
                        "rerun_failed": rerun_failed,
                        "debug": debug,
                        "run_id": normalized_run_id,
                        "project_root": model_dir.parent.parent.resolve(),
                    }
                    if runs_root_base is not None:
                        run_model_kwargs["runs_root_base"] = runs_root_base
                    if matlab_cache_root_base is not None:
                        run_model_kwargs["matlab_cache_root"] = matlab_cache_root_base
                    if profile_timing:
                        run_model_kwargs["profile_timing"] = True
                    if compute_features:
                        run_model_kwargs["compute_features"] = True
                        run_model_kwargs["compute_feature_names"] = tuple(
                            compute_feature_names or ()
                        )
                    _run_model_dir(model_dir, **run_model_kwargs)
            except ModelLockError as exc:
                raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt as exc:
        raise SystemExit(130) from exc


def _run_model_dir(
    model_dir: Path,
    *,
    raise_on_errors: bool,
    overwrite: bool,
    rerun_failed: bool = False,
    debug: bool = False,
    run_id: Optional[str] = None,
    project_root: Path,
    runs_root_base: Optional[Path] = None,
    matlab_cache_root: Optional[Path] = None,
    profile_timing: bool = False,
    compute_features: bool = False,
    compute_feature_names: Optional[Sequence[str]] = None,
) -> None:
    model_dir = model_dir.resolve()
    specs_path = model_dir / "model_run_specs.json"
    if not specs_path.exists():
        raise SystemExit(f"model_run_specs.json missing for {model_dir}.")
    _validate_environment_profile_required_files(model_dir)

    config_path = model_dir / "experiment_config.json"
    if not config_path.exists():
        raise SystemExit(f"experiment_config.json missing for {model_dir}")
    _validate_environment_profile_config_keys(config_path)
    try:
        experiment_config = load_experiment_config_json(config_path)
    except DistributionValidationError as exc:
        raise SystemExit(f"Invalid experiment config at {config_path}: {exc}") from exc

    metadata_path = model_dir / "generated" / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"generated/metadata.json missing for {model_dir}")
    try:
        metadata_model = load_simulink_generated_metadata(metadata_path)
    except MetadataValidationError as exc:
        raise SystemExit(f"Invalid metadata at {metadata_path}: {exc}") from exc
    metadata = metadata_model.model_dump(mode="python")

    runs_root = _resolve_model_runs_root(
        model_dir,
        runs_root_base=runs_root_base,
    )
    runs_root.mkdir(exist_ok=True, parents=True)
    model_record_path = resolve_model_record_path(model_dir, runs_dir=runs_root)
    timing = _TimingRecorder(
        enabled=profile_timing,
        path=runs_root / "profile_timing.jsonl" if profile_timing else None,
        model_name=model_dir.name,
    )

    specs_payload = _load_model_run_specs_with_refresh(
        specs_path=specs_path,
        model_record_path=model_record_path,
    )
    baseline_parameters_by_id = _baseline_parameters_from_raw_specs(
        specs_payload=specs_payload,
    )
    try:
        planned_runs = _build_planned_runs(
            specs_payload,
            model_dir=model_dir,
            baseline_parameters_by_id=baseline_parameters_by_id,
            experiment_config=experiment_config,
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid baseline_parameters execution keys in {specs_path}: {exc}") from exc
    planned_runs_by_id = {planned.run_id: planned for planned in planned_runs}
    expected_run_id_set = expected_run_ids(specs_payload)
    expected_runtime_model_record = build_expected_runtime_model_record(specs_payload)
    runtime_model_record: Dict[str, Any] = {}
    if model_record_path.exists():
        runtime_model_record = load_model_record_json(model_record_path)
    runtime_state = reconcile_runtime_model_record(
        specs=specs_payload,
        runtime_map=runtime_model_record,
    )
    normalized_run_id = str(run_id or "").strip().lower() or None
    if normalized_run_id and normalized_run_id not in planned_runs_by_id:
        raise SystemExit(
            f"run_id '{normalized_run_id}' is not present in model_run_specs.json for {model_dir.name}."
        )

    observable_signals = list(experiment_config.observable_signal_names)
    observable_signal_types = experiment_config.observable_signal_types
    if not observable_signals:
        raise SystemExit(
            f"experiment_config.json must define observable_signals for {model_dir.name}."
        )

    try:
        profile_validation = validate_environment_profile_consistency(
            model_dir=model_dir,
            experiment_config=experiment_config,
            metadata=metadata,
        )
    except EnvironmentProfileValidationError as exc:
        raise SystemExit(str(exc)) from exc
    description_level_parameters = profile_validation.description_parameter_mapping
    all_signal_names = profile_validation.all_signal_names

    documented_parameter_names = _documented_parameter_names_from_specs(
        specs=specs_payload,
        baseline_parameters_by_id=baseline_parameters_by_id,
    )
    parameter_alias_map: Dict[str, str] = {}
    if documented_parameter_names:
        parameter_alias_map = _build_parameter_alias_map(
            metadata=metadata,
            documented_parameters=documented_parameter_names,
            agent_facing_parameter_map=description_level_parameters,
        )

    def persist_runtime_registry() -> None:
        dump_model_record_json(model_record_path, runtime_state, indent=2)

    def has_run_artifacts(run_id: str) -> bool:
        run_dir = runs_root / str(run_id).strip()
        return (run_dir / "data.parquet").exists()

    def cleanup_orphan_run_artifacts() -> int:
        removed = 0
        if not runs_root.exists():
            return 0
        for candidate in runs_root.iterdir():
            if not candidate.is_dir():
                continue
            name = str(candidate.name or "").strip().lower()
            if name in expected_run_id_set:
                continue
            removed += _remove_run_dir(candidate)
        return removed

    def reconcile_registry_statuses() -> int:
        updated = cleanup_orphan_run_artifacts()
        for planned in planned_runs:
            entry = runtime_state[planned.run_id]
            expected_hash = _resolve_runtime_hash_entry(
                expected_runtime_model_record,
                planned.run_id,
            )
            runtime_hash = _resolve_runtime_hash_entry(
                runtime_model_record,
                planned.run_id,
            )
            hash_matches = (
                bool(expected_hash[0])
                and expected_hash == runtime_hash
            )
            if entry["status"] == "success" and (
                not has_run_artifacts(planned.run_id) or not hash_matches
            ):
                _set_runtime_state(entry, status="not_run")
                _remove_run_dir(runs_root / planned.run_id)
                updated += 1
            if entry["status"] == "not_run":
                if _remove_run_dir(runs_root / planned.run_id):
                    updated += 1
                if any(
                    key in entry for key in ("timestamp", "end_time_simulation", "error")
                ):
                    _set_runtime_state(entry, status="not_run")
                    updated += 1

            end_time = _load_end_time_from_parquet(runs_root / planned.run_id)
            if end_time is not None and _runtime_end_time(entry) != end_time:
                entry["end_time_simulation"] = end_time
                updated += 1
        return updated

    with timing.span("reconcile_registry_statuses"):
        reconciled = reconcile_registry_statuses()
    if reconciled:
        print(f"Reconciled {reconciled} registry status fields based on disk artifacts.")

    if normalized_run_id:
        _set_runtime_state(runtime_state[normalized_run_id], status="not_run")
        _remove_run_dir(runs_root / normalized_run_id)
        print(f"Targeted overwrite requested: marked run {normalized_run_id} as not_run.")
        pending_runs = [planned_runs_by_id[normalized_run_id]]
    elif overwrite:
        for planned in planned_runs:
            _set_runtime_state(runtime_state[planned.run_id], status="not_run")
            _remove_run_dir(runs_root / planned.run_id)
        print("Overwrite requested: marked all runs as not_run.")
        pending_runs = list(planned_runs)
    elif rerun_failed:
        rerun_count = 0
        for planned in planned_runs:
            if runtime_state[planned.run_id]["status"] != "failed":
                continue
            _set_runtime_state(runtime_state[planned.run_id], status="not_run")
            _remove_run_dir(runs_root / planned.run_id)
            rerun_count += 1
        print(f"Rerun-failed requested: marked {rerun_count} failed runs as not_run.")
        pending_runs = [
            planned for planned in planned_runs if runtime_state[planned.run_id]["status"] == "not_run"
        ]
    else:
        pending_runs = [
            planned for planned in planned_runs if runtime_state[planned.run_id]["status"] == "not_run"
        ]

    if not pending_runs:
        with timing.span("persist_runtime_registry"):
            persist_runtime_registry()
        print("No not-run simulations to execute.")
        return

    os.chdir(str(model_dir))
    original_cwd = project_root.resolve()
    matlab_session_kwargs: Dict[str, Any] = {}
    if matlab_cache_root is not None:
        matlab_session_kwargs["cache_root"] = Path(matlab_cache_root)
        matlab_session_kwargs["keep_cache"] = debug
    if matlab_session_kwargs:
        matlab_session_cm = bm_module.matlab_session(
            original_cwd,
            **matlab_session_kwargs,
        )
    else:
        matlab_session_cm = bm_module.matlab_session(original_cwd)
    mle = None
    try:
        with timing.span("matlab_startup"):
            mle = matlab_session_cm.__enter__()

        sim_script = model_dir.parent / "sim_the_model.m"
        expected_simulink_prefix = [
            signal
            for signal in observable_signals
            if signal in set(metadata.get("simulink_signals_available") or [])
        ]
        rel_model = (
            str(model_dir.relative_to(root_dir / "models"))
            if model_dir.is_relative_to(root_dir / "models")
            else str(model_dir)
        )

        temp_source = Path("simulink_model.mdl")
        created_temp = False
        if not temp_source.exists() and Path("simulink_model_original.mdl").exists():
            shutil.copy("simulink_model_original.mdl", temp_source)
            created_temp = True

        try:
            for planned in pending_runs:
                run_id = planned.run_id
                print(f"Simulating {planned.kind} {run_id}...")
                runtime_entry = runtime_state[run_id]
                with timing.span("run_total", run_id=run_id):
                    try:
                        configured_end_time_s = float(planned.end_time_input_s or 0.0)
                        with timing.span("sanity_checks", run_id=run_id):
                            _assert_run_time_and_sampling_sanity(
                                run_id=run_id,
                                intervention_time=planned.intervention_time,
                                sampling_rate_hz=planned.sampling_rate_hz,
                                configured_end_time_s=configured_end_time_s,
                            )
                    except KeyboardInterrupt as exc:
                        _propagate_interrupt(exc, runtime_entry=runtime_entry)
                    except Exception as exc:
                        if _is_interrupt_exception(exc):
                            _propagate_interrupt(exc, runtime_entry=runtime_entry)
                        if raise_on_errors:
                            raise
                        _set_runtime_state(
                            runtime_entry,
                            status="failed",
                            timestamp=datetime.now().isoformat(),
                            error=str(exc),
                        )
                        print(f"Failed sanity checks for {run_id}: {exc}")
                        continue

                    run_dir = runs_root / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    normalized_baseline_parameters = _normalize_parameter_dict(
                        dict(planned.baseline_parameters),
                        alias_map=parameter_alias_map,
                    )
                    recipe = {
                        "id": run_id,
                        "run_id": run_id,
                        "model": rel_model,
                        "parent_id": planned.parent_id,
                        "baseline_parameters": normalized_baseline_parameters,
                        "intervention_mode": planned.intervention_mode,
                        "intervention_time": float(planned.intervention_time),
                        "end_time_input_s": float(planned.end_time_input_s),
                    }
                    if planned.intervention_parameters:
                        recipe["intervention_parameters"] = _normalize_parameter_dict(
                            dict(planned.intervention_parameters),
                            alias_map=parameter_alias_map,
                        )
                    try:
                        simulate_kwargs = {
                            "matlab_engine": mle,
                            "metadata": metadata,
                            "run_dir": run_dir,
                            "internal_sampling_rate_hz": planned.internal_sampling_rate_hz,
                            "sampling_rate_hz": planned.sampling_rate_hz,
                            "sim_script": sim_script,
                            "all_signal_names": all_signal_names,
                            "feature_model_dir": model_dir,
                            "return_features": True,
                            "compute_features": bool(compute_features),
                            "feature_names": tuple(compute_feature_names or ())
                            if compute_features and compute_feature_names
                            else None,
                            "runtime_model_snapshot_path": run_dir / "simulink_model.mdl",
                            "debug": debug,
                        }
                        if profile_timing:
                            simulate_kwargs["timing_callback"] = timing.callback
                        with timing.span("simulate_recipe", run_id=run_id):
                            simulation_result = simulation_api.simulate_recipe(
                                recipe,
                                observable_signals,
                                **simulate_kwargs,
                            )
                        if (
                            isinstance(simulation_result, tuple)
                            and len(simulation_result) == 2
                        ):
                            signal_dict, feature_dict = simulation_result
                        else:
                            signal_dict = simulation_result
                            feature_dict = {}
                        with timing.span("resample_signal_dict", run_id=run_id):
                            df = simulation_api.resample_signal_dict(
                                signal_dict,
                                observable_signal_types,
                                internal_sampling_rate_hz=planned.internal_sampling_rate_hz,
                                sampling_rate_hz=planned.sampling_rate_hz,
                                end_time_input_s=configured_end_time_s,
                            )
                        with timing.span("assert_signal_prefix", run_id=run_id):
                            _assert_simulink_signal_prefix(
                                df,
                                expected_simulink_prefix,
                                context=f"run_id={run_id}",
                            )
                        with timing.span("serialize", run_id=run_id):
                            meta = simulation_api.serialize(
                                df,
                                run_id,
                                run_dir=run_dir,
                            )
                        with timing.span("validate", run_id=run_id):
                            simulation_api.validate(
                                run_id,
                                run_dir=run_dir,
                                observable_signals=observable_signals,
                                sampling_rate_hz=planned.sampling_rate_hz,
                                end_time_input_s=configured_end_time_s,
                            )
                        if compute_features:
                            with timing.span("save_features", run_id=run_id):
                                simulation_api.save_feature_dict(
                                    feature_dict,
                                    run_dir=run_dir,
                                )
                        measured_end_time_s = _coerce_finite_float(meta.get("time_end_s"))
                        _set_runtime_state(
                            runtime_entry,
                            status="success",
                            timestamp=datetime.now().isoformat(),
                            end_time_simulation=measured_end_time_s,
                        )
                    except KeyboardInterrupt as exc:
                        _propagate_interrupt(
                            exc,
                            runtime_entry=runtime_entry,
                            run_dir=run_dir,
                        )
                    except Exception as exc:
                        if _is_interrupt_exception(exc):
                            _propagate_interrupt(
                                exc,
                                runtime_entry=runtime_entry,
                                run_dir=run_dir,
                            )
                        if raise_on_errors:
                            raise
                        traceback.print_exc()
                        _remove_run_dir(run_dir)
                        _set_runtime_state(
                            runtime_entry,
                            status="failed",
                            timestamp=datetime.now().isoformat(),
                            error=f"{exc}\n{traceback.format_exc()}",
                        )
        finally:
            if created_temp and temp_source.exists():
                temp_source.unlink()
    finally:
        if matlab_session_cm is not None:
            try:
                matlab_session_cm.__exit__(None, None, None)
            except Exception:
                pass
        os.chdir(str(project_root.resolve()))
        with timing.span("persist_runtime_registry"):
            persist_runtime_registry()


if __name__ == "__main__":
    main()
