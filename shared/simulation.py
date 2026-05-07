"""Shared simulation helpers used by the simulation stage."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib.util
import json
import pickle
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from shared.intervention_sampling import (
    collect_intervention_targets,
    eval_numeric_expr,
)

BuildTVPFn = Any
SimulateCaseFn = Any
TimingCallback = Callable[..., None]

_MIN_SAMPLED_POINTS = 50
_FIRST_STAGE_VALUES = {"validate_only", "interpolation", "zero_filling"}
_SECOND_STAGE_VALUES = {"decimation", "abs_max_pooling"}
_SIGNAL_TYPE_VALUES = {"continuous", "impulse_like"}
_SIMULATION_STOP_TIME_INFLATION = 1.2
FEATURES_FILENAME = "features.json"


def _simulate_core_module():
    import workflows.simulate_core as sim_module

    return sim_module


def _emit_timing(
    timing_callback: Optional[TimingCallback],
    *,
    phase: str,
    duration_s: float,
    **extra: Any,
) -> None:
    if timing_callback is None:
        return
    try:
        timing_callback(phase=phase, duration_s=float(duration_s), **extra)
    except Exception:
        return


@contextmanager
def _timed_phase(
    timing_callback: Optional[TimingCallback],
    phase: str,
    **extra: Any,
) -> Iterator[None]:
    if timing_callback is None:
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
        _emit_timing(
            timing_callback,
            phase=phase,
            duration_s=time.perf_counter() - started,
            status=status,
            **extra,
        )


def _coerce_finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _resolve_solver_step(*, sampling_rate_hz: Any) -> float:
    sampling_rate = _coerce_finite_float(sampling_rate_hz)
    if sampling_rate is not None and sampling_rate > 0.0:
        return 1.0 / float(sampling_rate)
    raise ValueError("Unable to resolve solver_step from sampling_rate_hz")


def resolve_internal_sampling_rate_hz(
    *,
    model_dir: Optional[Path],
    sampling_rate_hz: Any,
) -> float:
    del model_dir
    sampling_rate = _coerce_finite_float(sampling_rate_hz)
    if sampling_rate is None or sampling_rate <= 0.0:
        raise ValueError("Unable to resolve internal_sampling_rate_hz from sampling_rate_hz")
    return float(sampling_rate)


def _normalize_resample_source(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df_work = df.copy()
    df_work.index = pd.to_numeric(df_work.index, errors="coerce").astype(float)
    df_work = df_work[~pd.isna(df_work.index)]
    return df_work.groupby(level=0).mean().sort_index()


def _target_time_index(*, sampling_rate_hz: float, end_time_input_s: float) -> pd.Index:
    dt = 1.0 / float(sampling_rate_hz)
    n_points = max(
        0,
        int(np.floor(float(end_time_input_s) * float(sampling_rate_hz))) - 1,
    )
    times = dt * np.arange(1, n_points + 1, dtype=np.float64)
    return pd.Index(times, name="time_s")


def _clean_source_series(source: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    source = source.dropna()
    if source.empty:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    source_times = pd.to_numeric(source.index, errors="coerce").to_numpy(dtype=float)
    source_values = pd.to_numeric(source, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(source_times) & np.isfinite(source_values)
    source_times = source_times[finite]
    source_values = source_values[finite]
    if source_times.size == 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    order = np.argsort(source_times)
    return source_times[order], source_values[order]


def _interpolate_series_to_target(
    source_times: np.ndarray,
    source_values: np.ndarray,
    *,
    target_times: np.ndarray,
) -> np.ndarray:
    if source_times.size == 0:
        return np.full(target_times.shape, np.nan, dtype=float)
    interpolated = np.interp(
        target_times,
        source_times,
        source_values,
        left=source_values[0],
        right=source_values[-1],
    )
    return np.asarray(interpolated, dtype=float)


def _normalize_sampling_strategy_pair(value: Any) -> Tuple[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "sampling_strategy entries must be length-2 arrays of [first_stage, second_stage]"
        )
    raw_first, raw_second = value
    first_stage = str(raw_first).strip().lower().replace("-", "_")
    second_stage = str(raw_second).strip().lower().replace("-", "_")
    if first_stage not in _FIRST_STAGE_VALUES:
        raise ValueError(
            f"Unsupported first-stage sampling strategy {raw_first!r}; "
            f"expected one of {sorted(_FIRST_STAGE_VALUES)}"
        )
    if second_stage not in _SECOND_STAGE_VALUES:
        raise ValueError(
            f"Unsupported second-stage sampling strategy {raw_second!r}; "
            f"expected one of {sorted(_SECOND_STAGE_VALUES)}"
        )
    return first_stage, second_stage


def _normalize_signal_type(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("type")
    signal_type = str(value).strip().lower().replace("-", "_")
    if signal_type not in _SIGNAL_TYPE_VALUES:
        raise ValueError(
            f"Unsupported observable signal_type {value!r}; "
            f"expected one of {sorted(_SIGNAL_TYPE_VALUES)}"
        )
    return signal_type


def sampling_strategy_from_signal_types(
    signal_types: Mapping[str, Any],
) -> Dict[str, Tuple[str, str]]:
    strategies: Dict[str, Tuple[str, str]] = {}
    for signal_name, raw_signal_type in signal_types.items():
        signal_type = _normalize_signal_type(raw_signal_type)
        if signal_type == "impulse_like":
            strategies[str(signal_name)] = ("validate_only", "abs_max_pooling")
        else:
            strategies[str(signal_name)] = ("validate_only", "decimation")
    return strategies


def _validate_regular_high_frequency_sampling(
    source_times: np.ndarray,
    *,
    internal_sampling_rate_hz: float,
) -> None:
    if source_times.size < 2:
        raise ValueError(
            "validate_only requires at least two raw samples to verify regular sampling"
        )
    diffs = np.diff(source_times)
    if np.any(diffs <= 0.0):
        raise ValueError(
            "validate_only requires strictly increasing raw signal timestamps"
        )
    raw_dt = float(np.median(diffs))
    raw_dt_atol = max(1e-9, abs(raw_dt) * 1e-6)
    if not np.allclose(
        diffs,
        np.full(diffs.shape, raw_dt, dtype=float),
        rtol=0.0,
        atol=raw_dt_atol,
    ):
        raise ValueError(
            "validate_only requires the raw signal to already be regularly sampled"
        )
    expected_dt = 1.0 / float(internal_sampling_rate_hz)
    expected_dt_atol = max(1e-9, abs(expected_dt) * 1e-6)
    if not np.isclose(raw_dt, expected_dt, rtol=0.0, atol=expected_dt_atol):
        raise ValueError(
            "validate_only requires the raw signal to be sampled at a frequency "
            "equal to internal_sampling_rate_hz"
        )


def _select_last_sample_in_target_bins(
    source_times: np.ndarray,
    source_values: np.ndarray,
    *,
    target_times: np.ndarray,
    sampling_rate_hz: float,
    fill_empty_with_zero: bool,
) -> np.ndarray:
    dt = 1.0 / float(sampling_rate_hz)
    default_value = 0.0 if fill_empty_with_zero else np.nan
    out = np.full(target_times.shape, default_value, dtype=float)
    for idx, sample_time in enumerate(target_times):
        if idx == 0:
            in_window = source_times <= sample_time + 1e-12
        else:
            lower = sample_time - dt
            in_window = (source_times > lower) & (source_times <= sample_time + 1e-12)
        if np.any(in_window):
            out[idx] = source_values[np.flatnonzero(in_window)[-1]]
    return out


def _pool_absmax_column(
    source_times: np.ndarray,
    source_values: np.ndarray,
    *,
    target_times: np.ndarray,
    first_stage: str,
    internal_sampling_rate_hz: float,
    sampling_rate_hz: float,
) -> np.ndarray:
    dt = 1.0 / float(sampling_rate_hz)

    if first_stage == "validate_only":
        _validate_regular_high_frequency_sampling(
            source_times,
            internal_sampling_rate_hz=internal_sampling_rate_hz,
        )
    fallback = None
    if first_stage == "interpolation":
        fallback = _interpolate_series_to_target(
            source_times,
            source_values,
            target_times=target_times,
        )
    out = np.full(target_times.shape, np.nan, dtype=float)
    for idx, sample_time in enumerate(target_times):
        if idx == 0:
            in_window = source_times <= sample_time + 1e-12
        else:
            lower = sample_time - dt
            in_window = (source_times > lower) & (source_times <= sample_time + 1e-12)
        if not np.any(in_window):
            if first_stage == "zero_filling":
                out[idx] = 0.0
            elif fallback is not None:
                out[idx] = fallback[idx]
            continue
        values = source_values[in_window]
        peak_idx = int(np.argmax(np.abs(values)))
        out[idx] = values[peak_idx]
    return out


def _resample_column(
    source: pd.Series,
    *,
    strategy_pair: Tuple[str, str],
    target_times: np.ndarray,
    internal_sampling_rate_hz: float,
    sampling_rate_hz: float,
) -> np.ndarray:
    first_stage, second_stage = strategy_pair
    source_times, source_values = _clean_source_series(source)
    if source_times.size == 0:
        if first_stage == "zero_filling":
            return np.zeros(target_times.shape, dtype=float)
        return np.full(target_times.shape, np.nan, dtype=float)
    if second_stage == "decimation":
        if first_stage == "validate_only":
            _validate_regular_high_frequency_sampling(
                source_times,
                internal_sampling_rate_hz=internal_sampling_rate_hz,
            )
            return _select_last_sample_in_target_bins(
                source_times,
                source_values,
                target_times=target_times,
                sampling_rate_hz=sampling_rate_hz,
                fill_empty_with_zero=False,
            )
        if first_stage == "zero_filling":
            return _select_last_sample_in_target_bins(
                source_times,
                source_values,
                target_times=target_times,
                sampling_rate_hz=sampling_rate_hz,
                fill_empty_with_zero=True,
            )
        return _interpolate_series_to_target(
            source_times,
            source_values,
            target_times=target_times,
        )
    return _pool_absmax_column(
        source_times,
        source_values,
        target_times=target_times,
        first_stage=first_stage,
        internal_sampling_rate_hz=internal_sampling_rate_hz,
        sampling_rate_hz=sampling_rate_hz,
    )


def resample_dataframe(
    df: pd.DataFrame,
    *,
    sampling_strategy: Mapping[str, Any],
    internal_sampling_rate_hz: Optional[float] = None,
    sampling_rate_hz: float,
    end_time_input_s: float,
) -> pd.DataFrame:
    if df.empty:
        return df
    df_work = _normalize_resample_source(df)
    target_index = _target_time_index(
        sampling_rate_hz=float(sampling_rate_hz),
        end_time_input_s=float(end_time_input_s),
    )
    target_times = pd.to_numeric(target_index, errors="coerce").to_numpy(dtype=float)
    resolved_internal_sampling_rate_hz = float(
        internal_sampling_rate_hz
        if internal_sampling_rate_hz is not None
        else sampling_rate_hz
    )
    out = pd.DataFrame(index=target_index, columns=list(df_work.columns), dtype=float)
    for column in df_work.columns:
        if column not in sampling_strategy:
            raise KeyError(f"Missing sampling_strategy for signal '{column}'")
        out[column] = _resample_column(
            df_work[column],
            strategy_pair=_normalize_sampling_strategy_pair(sampling_strategy[column]),
            target_times=target_times,
            internal_sampling_rate_hz=resolved_internal_sampling_rate_hz,
            sampling_rate_hz=float(sampling_rate_hz),
        )

    return out


def resample_dataframe_by_signal_type(
    df: pd.DataFrame,
    *,
    signal_type: Mapping[str, Any],
    internal_sampling_rate_hz: Optional[float] = None,
    sampling_rate_hz: float,
    end_time_input_s: float,
) -> pd.DataFrame:
    missing = [column for column in df.columns if column not in signal_type]
    if missing:
        raise KeyError(f"Missing signal_type for signal '{missing[0]}'")
    return resample_dataframe(
        df,
        sampling_strategy=sampling_strategy_from_signal_types(signal_type),
        internal_sampling_rate_hz=internal_sampling_rate_hz,
        sampling_rate_hz=sampling_rate_hz,
        end_time_input_s=end_time_input_s,
    )


def dataframe_to_signal_dict(
    df: pd.DataFrame,
    *,
    observable_signals: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    if df.empty:
        return {}
    ordered_columns = (
        [signal for signal in observable_signals if signal in df.columns]
        if observable_signals is not None
        else list(df.columns)
    )
    signal_dict: Dict[str, Dict[str, np.ndarray]] = {}
    for column in ordered_columns:
        signal_dict[str(column)] = {
            "timestamp": pd.to_numeric(df.index, errors="coerce").to_numpy(dtype=float),
            "data": pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float),
        }
    return signal_dict


def signal_dict_to_dataframe(signal_dict: Mapping[str, Any]) -> pd.DataFrame:
    series: List[pd.Series] = []
    for signal_name, payload in signal_dict.items():
        if not isinstance(payload, Mapping):
            continue
        timestamps = payload.get("timestamp")
        data = payload.get("data")
        if timestamps is None or data is None:
            continue
        time_values = np.asarray(timestamps, dtype=float).reshape(-1)
        signal_values = np.asarray(data, dtype=float).reshape(-1)
        if time_values.shape[0] != signal_values.shape[0]:
            raise ValueError(f"Signal '{signal_name}' has mismatched timestamp/data lengths")
        series.append(pd.Series(signal_values, index=time_values, name=str(signal_name)))
    if not series:
        return pd.DataFrame()
    df = pd.concat(series, axis=1).sort_index()
    df.index.name = "time_s"
    return _normalize_resample_source(df)


def _feature_module_path(model_dir: Optional[Path]) -> Optional[Path]:
    if model_dir is None:
        return None
    model_path = Path(model_dir)
    features_path = model_path / "features.py"
    return features_path if features_path.exists() else None


def _load_features_module(model_dir: Optional[Path]) -> Optional[Any]:
    features_path = _feature_module_path(model_dir)
    if features_path is None:
        return None
    module_name = f"_tsenv_features_{features_path.parent.name}_{abs(hash(features_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, features_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load feature module at {features_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise
    return module


def _load_features_function(model_dir: Optional[Path]) -> Optional[Any]:
    module = _load_features_module(model_dir)
    if module is None:
        return None
    fn = getattr(module, "compute_problem_specific_features", None)
    if not callable(fn):
        features_path = _feature_module_path(model_dir)
        raise AttributeError(
            f"{features_path} must define callable compute_problem_specific_features(all_signal_simulation)"
        )
    return fn


def _ensure_json_serializable_feature_dict(feature_dict: Any) -> Dict[str, Any]:
    if not isinstance(feature_dict, dict):
        raise TypeError("compute_problem_specific_features must return a dict")
    try:
        json.dumps(feature_dict, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "compute_problem_specific_features must return a JSON-serializable dict "
            "without NaN or Infinity values"
        ) from exc
    return dict(feature_dict)


def compute_problem_specific_features(
    all_signal_simulation: Mapping[str, Any],
    *,
    model_dir: Optional[Path] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    module = _load_features_module(model_dir)
    if module is None:
        return {}
    requested_names = (
        tuple(str(name) for name in feature_names)
        if feature_names is not None
        else None
    )
    if requested_names is not None:
        fn = getattr(module, "compute_features", None)
        if not callable(fn):
            features_path = _feature_module_path(model_dir)
            raise AttributeError(
                f"{features_path} must define callable compute_features("
                "all_signal_simulation, feature_names=...) when named features "
                "are requested"
            )
        return _ensure_json_serializable_feature_dict(
            fn(all_signal_simulation, feature_names=requested_names)
        )
    fn = getattr(module, "compute_problem_specific_features", None)
    if not callable(fn):
        features_path = _feature_module_path(model_dir)
        raise AttributeError(
            f"{features_path} must define callable compute_problem_specific_features(all_signal_simulation)"
        )
    return _ensure_json_serializable_feature_dict(fn(all_signal_simulation))


def _slice_signal_dict_before_end_time(
    signal_dict: Mapping[str, Any],
    *,
    end_time_input_s: float,
) -> Dict[str, Any]:
    end_time = float(end_time_input_s)
    sliced: Dict[str, Any] = {}
    for signal_name, payload in signal_dict.items():
        if not isinstance(payload, Mapping):
            sliced[str(signal_name)] = payload
            continue
        timestamps = payload.get("timestamp")
        data = payload.get("data")
        if timestamps is None or data is None:
            sliced[str(signal_name)] = dict(payload)
            continue

        time_values = np.asarray(timestamps).reshape(-1)
        data_values = np.asarray(data)
        if data_values.shape[0] != time_values.shape[0]:
            raise ValueError(
                f"Signal '{signal_name}' has mismatched timestamp/data lengths"
            )
        mask = time_values < end_time
        sliced_payload = dict(payload)
        sliced_payload["timestamp"] = time_values[mask]
        sliced_payload["data"] = data_values[mask]
        sliced[str(signal_name)] = sliced_payload
    return sliced


def save_feature_dict(
    feature_dict: Mapping[str, Any],
    *,
    run_dir: Path,
) -> Path:
    checked = _ensure_json_serializable_feature_dict(dict(feature_dict))
    run_dir.mkdir(parents=True, exist_ok=True)
    path = Path(run_dir) / FEATURES_FILENAME
    path.write_text(
        json.dumps(checked, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _intervention_bindings_by_variable(
    metadata: Mapping[str, Any],
) -> Dict[str, List[Any]]:
    return collect_intervention_targets(metadata)


def _build_initial_parameter_values(
    *,
    metadata: Mapping[str, Any],
    baseline_variables: Mapping[str, Any],
    parameter_addresses: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[Tuple[str, str], float]:
    parameter_values: Dict[Tuple[str, str], float] = {}
    requested_addresses = (
        set(parameter_addresses) if parameter_addresses is not None else None
    )
    for targets in _intervention_bindings_by_variable(metadata).values():
        for target in targets:
            parameter_address = (target.path, target.parameter)
            if (
                requested_addresses is not None
                and parameter_address not in requested_addresses
            ):
                continue
            expression = str(target.expression or "").strip() or target.parameter
            parameter_values[parameter_address] = float(
                eval_numeric_expr(expression, baseline_variables)
            )
    return parameter_values


def _baseline_variables_with_metadata_defaults(
    *,
    metadata: Mapping[str, Any],
    baseline_variables: Mapping[str, Any],
) -> Dict[str, Any]:
    defaults = metadata.get("default_values") or {}
    resolved: Dict[str, Any] = {}
    if isinstance(defaults, Mapping):
        resolved.update(
            {
                str(name): value
                for name, value in defaults.items()
                if str(name) != "end_time_input_s"
            }
        )
    resolved.update({str(name): value for name, value in baseline_variables.items()})
    return resolved


def _all_parameter_addresses_from_metadata(
    metadata: Mapping[str, Any],
) -> List[Tuple[str, str]]:
    addresses = {
        (str(target.path), str(target.parameter))
        for targets in _intervention_bindings_by_variable(metadata).values()
        for target in targets
    }
    return sorted(addresses)


def _build_modifications_from_child_parameters(
    *,
    metadata: Mapping[str, Any],
    baseline_variables: Mapping[str, Any],
    child_parameters: Mapping[str, Any],
    intervention_time: float,
) -> List[Dict[str, Any]]:
    modifications: List[Dict[str, Any]] = []
    if not child_parameters:
        return modifications
    if len(child_parameters) != 1:
        raise ValueError("children_parameter must contain exactly one intervened field")

    intervention_bindings = _intervention_bindings_by_variable(metadata)
    baseline_variables = dict(baseline_variables)
    for intervention_parameter, raw_value in child_parameters.items():
        variable_name = str(intervention_parameter or "").strip()
        if not variable_name:
            raise ValueError("children_parameter key must be non-empty")
        if variable_name not in intervention_bindings:
            raise ValueError(f"Variable {variable_name} not modifiable")

        new_variable_value = float(raw_value)
        if not np.isfinite(new_variable_value):
            raise ValueError(f"Non-finite value for intervention {variable_name}")

        for target in intervention_bindings[variable_name]:
            parameter_address = (target.path, target.parameter)
            expression = str(target.expression or "").strip() or variable_name
            old_param_value = float(
                eval_numeric_expr(
                    expression,
                    baseline_variables,
                )
            )
            modified_variables = dict(baseline_variables)
            modified_variables[variable_name] = new_variable_value
            new_param_value = float(
                eval_numeric_expr(
                    expression,
                    modified_variables,
                )
            )
            modifications.append(
                {
                    "identifier": parameter_address[0],
                    "key": parameter_address[1],
                    "parameter_id": f"{parameter_address[0]}::{parameter_address[1]}",
                    "transition_type": "step",
                    "start_time": float(intervention_time),
                    "end_time": None,
                    "old_value": old_param_value,
                    "new_value": new_param_value,
                    "intervention_parameter": variable_name,
                    "is_runtime_type_parameter": bool(
                        getattr(target, "is_runtype", False)
                    ),
                }
            )
        baseline_variables[variable_name] = new_variable_value
    return modifications


def simulate_recipe(
    recipe: Dict[str, Any],
    observable_signals: Sequence[str],
    *,
    matlab_engine: Any,
    metadata: Dict[str, Any],
    run_dir: Path,
    internal_sampling_rate_hz: Optional[float] = None,
    sampling_rate_hz: float,
    sim_script: Path,
    all_signal_names: Optional[Sequence[str]] = None,
    feature_model_dir: Optional[Path] = None,
    return_features: bool = False,
    runtime_model_snapshot_path: Optional[Path] = None,
    initial_state: Optional[Any] = None,
    modifications: Optional[List[Dict[str, Any]]] = None,
    debug: bool = False,
    timing_callback: Optional[TimingCallback] = None,
    compute_features: bool = True,
    feature_names: Optional[Sequence[str]] = None,
) -> Union[
    Dict[str, Dict[str, np.ndarray]],
    Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]],
]:
    sim_module = _simulate_core_module()
    baseline_parameters = recipe.get("baseline_parameters")
    if not isinstance(baseline_parameters, Mapping):
        raise TypeError("recipe.baseline_parameters must be an object")
    configured_end_time_input_s = _coerce_finite_float(recipe.get("end_time_input_s"))
    if configured_end_time_input_s is None or configured_end_time_input_s <= 0.0:
        raise ValueError(
            "recipe.end_time_input_s must be present and positive for simulation"
        )
    resolved_internal_sampling_rate_hz = float(
        internal_sampling_rate_hz
        if internal_sampling_rate_hz is not None
        else sampling_rate_hz
    )
    runtime_end_time_input_s = float(configured_end_time_input_s) * float(
        _SIMULATION_STOP_TIME_INFLATION
    )
    baseline_variable_values = _baseline_variables_with_metadata_defaults(
        metadata=metadata,
        baseline_variables=baseline_parameters,
    )
    if initial_state is not None:
        extra_variables = getattr(initial_state, "variable_values", {}) or {}
        if isinstance(extra_variables, Mapping):
            for var_name, raw_value in extra_variables.items():
                if str(var_name) not in baseline_variable_values:
                    baseline_variable_values[str(var_name)] = raw_value

    child_parameters_raw = recipe.get("intervention_parameters")
    if not isinstance(child_parameters_raw, Mapping):
        child_parameters_raw = recipe.get("children_parameter")
    child_parameters = (
        dict(child_parameters_raw) if isinstance(child_parameters_raw, Mapping) else {}
    )
    intervention_mode = str(recipe.get("intervention_mode") or "").strip().lower()
    if not intervention_mode:
        intervention_mode = "at_intervention_time" if child_parameters else "none"
    if intervention_mode not in {"none", "at_intervention_time", "from_time_zero"}:
        raise ValueError(
            "recipe.intervention_mode must be one of none, at_intervention_time, or from_time_zero"
        )
    intervention_time = float(recipe.get("intervention_time") or 0.0)
    if modifications is None:
        modifications_intervention_time = (
            0.0 if intervention_mode == "from_time_zero" else intervention_time
        )
        modifications = _build_modifications_from_child_parameters(
            metadata=metadata,
            baseline_variables=baseline_variable_values,
            child_parameters={} if intervention_mode == "none" else child_parameters,
            intervention_time=modifications_intervention_time,
        )

    modified_params = {
        (str(modification["identifier"]), str(modification["key"]))
        for modification in modifications
        if modification.get("identifier") is not None
        and modification.get("key") is not None
    }
    all_params = sorted(
        {
            *_all_parameter_addresses_from_metadata(metadata),
            *modified_params,
        }
    )
    initial_parameter_values = _build_initial_parameter_values(
        metadata=metadata,
        baseline_variables=baseline_variable_values,
        parameter_addresses=all_params,
    )
    requested_signal_names = list(all_signal_names or observable_signals)
    simscape_available = set(metadata.get("simscape_signals_available") or [])
    simscape_signals = [
        signal for signal in requested_signal_names if signal in simscape_available
    ]
    simulink_signals = [
        signal for signal in requested_signal_names if signal not in simscape_available
    ]
    internal_recipe = {
        "id": str(recipe.get("run_id") or recipe.get("id") or ""),
        "model": str(recipe.get("model") or ""),
        "parent_id": recipe.get("parent_id"),
        "baseline_parameters": baseline_variable_values,
        "initial_parameter_values": initial_parameter_values,
        "end_time_input_s": float(runtime_end_time_input_s),
        "solver_step": _resolve_solver_step(
            sampling_rate_hz=resolved_internal_sampling_rate_hz,
        ),
        "internal_sampling_rate_hz": resolved_internal_sampling_rate_hz,
        "sampling_rate_hz": float(sampling_rate_hz),
        "intervention_time": float(intervention_time),
        "correct": "healthy",
        "modifications": modifications,
    }

    tvp = sim_module._build_complete_tvp(
        internal_recipe,
        all_params,
        modifications_to_apply=modifications if modifications else None,
    )
    debug_dir_cm = nullcontext(run_dir / "debug") if debug else TemporaryDirectory(
        prefix="tsenv-sim-debug-"
    )
    with debug_dir_cm as debug_dir_root:
        debug_dir = Path(debug_dir_root)
        all_signal_dict = sim_module._simulate_case_to_signal_dict(
            matlab_engine,
            internal_recipe,
            tvp=tvp,
            debug_dir=debug_dir,
            expected_stop_time=float(runtime_end_time_input_s),
            sim_script=sim_script,
            simscape_signals=simscape_signals,
            simulink_signals=simulink_signals,
            runtime_model_snapshot_path=runtime_model_snapshot_path,
            save_simscape_mat=False,
            debug=debug,
            timing_callback=timing_callback,
        )
        returned_signal_dict = {
            signal: all_signal_dict[signal]
            for signal in observable_signals
            if signal in all_signal_dict
        }
        feature_dict: Dict[str, Any] = {}
        if compute_features:
            feature_signal_dict = _slice_signal_dict_before_end_time(
                returned_signal_dict,
                end_time_input_s=float(configured_end_time_input_s),
            )
            with _timed_phase(
                timing_callback,
                "compute_problem_specific_features",
                run_id=str(recipe.get("run_id") or recipe.get("id") or ""),
            ):
                feature_dict = compute_problem_specific_features(
                    feature_signal_dict,
                    model_dir=feature_model_dir,
                    feature_names=feature_names,
                )
        if debug:
            input_signals_path = debug_dir / "input_signals.mat"
            if not input_signals_path.exists():
                raise FileNotFoundError(
                    "Debug simulation is missing expected MATLAB artifact "
                    f"'{input_signals_path}'."
                )
            raw_simulation_output_path = debug_dir / "raw_simulation.pickle"
            raw_simulation_output_path.parent.mkdir(parents=True, exist_ok=True)
            with raw_simulation_output_path.open("wb") as handle:
                pickle.dump(
                    returned_signal_dict,
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
    if return_features:
        return returned_signal_dict, feature_dict
    return returned_signal_dict


def resample_signal_dict(
    signal_dict: Mapping[str, Any],
    signal_type: Mapping[str, Any],
    *,
    internal_sampling_rate_hz: Optional[float] = None,
    sampling_rate_hz: float,
    end_time_input_s: float,
) -> pd.DataFrame:
    df = signal_dict_to_dataframe(signal_dict)
    return resample_dataframe_by_signal_type(
        df,
        signal_type=signal_type,
        internal_sampling_rate_hz=internal_sampling_rate_hz,
        sampling_rate_hz=float(sampling_rate_hz),
        end_time_input_s=float(end_time_input_s),
    )


def serialize(
    df: pd.DataFrame,
    uuid: str,
    *,
    run_dir: Path,
) -> Dict[str, Any]:
    def _cast_float32_checked(values: Any, *, field_name: str) -> np.ndarray:
        numeric_values = pd.to_numeric(values, errors="coerce")
        source = np.asarray(numeric_values, dtype=float)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            try:
                cast = source.astype(np.float32)
            except RuntimeWarning as exc:
                raise OverflowError(
                    f"Run '{uuid}' encountered overflow while casting {field_name} to float32"
                ) from exc
        overflow_mask = np.isfinite(source) & ~np.isfinite(cast.astype(np.float64))
        if np.any(overflow_mask):
            raise OverflowError(
                f"Run '{uuid}' encountered overflow while casting {field_name} to float32"
            )
        return cast

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "data.parquet"
    out = df.copy()
    for column in out.columns:
        out[column] = _cast_float32_checked(out[column], field_name=f"column '{column}'")
    serialized = out.copy()
    serialized["time"] = _cast_float32_checked(serialized.index, field_name="time")
    serialized.reset_index(drop=True, inplace=True)
    serialized.to_parquet(path, compression="zstd", index=False)
    return {
        "run_id": str(uuid),
        "time_end_s": float(pd.to_numeric(df.index, errors="coerce").max()),
        "sampled_points": int(len(df.index)),
    }


def validate(
    run_ref: str,
    *,
    df: Optional[pd.DataFrame] = None,
    run_dir: Optional[Path] = None,
    min_sampled_points: int = _MIN_SAMPLED_POINTS,
    observable_signals: Optional[Sequence[str]] = None,
    sampling_rate_hz: Optional[float] = None,
    end_time_input_s: Optional[float] = None,
) -> None:
    loaded_df: Optional[pd.DataFrame] = None
    if run_dir is not None:
        parquet_path = Path(run_dir) / "data.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"No serialized data found for run {run_ref}")
        loaded_df = pd.read_parquet(parquet_path)
        if "time" not in loaded_df.columns:
            raise ValueError(f"Run '{run_ref}' is missing time column after serialization")
        if str(loaded_df["time"].dtype) != "float32":
            raise ValueError(
                f"Run '{run_ref}' time column must be float32, got {loaded_df['time'].dtype}"
            )
        for column in loaded_df.columns:
            if str(loaded_df[column].dtype) != "float32":
                raise ValueError(
                    f"Run '{run_ref}' column '{column}' must be float32, got {loaded_df[column].dtype}"
                )
        df = loaded_df.set_index("time")
    elif df is None:
        raise ValueError("validate requires either df or run_dir")

    if len(df.index) < int(min_sampled_points):
        raise ValueError(
            f"Run '{run_ref}' has too few sampled points: {len(df.index)} < {int(min_sampled_points)}"
        )

    signal_columns = [str(column) for column in df.columns]
    if observable_signals is not None and signal_columns != list(observable_signals):
        raise ValueError(
            f"Run '{run_ref}' signal columns must match observable_signals exactly; "
            f"got {signal_columns} expected {list(observable_signals)}"
        )

    time_values = pd.to_numeric(pd.Index(df.index), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(time_values).all():
        raise ValueError(f"Run '{run_ref}' contains non-finite time samples")
    signal_values = pd.DataFrame(df).apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(signal_values.to_numpy(dtype=float)).all():
        raise ValueError(f"Run '{run_ref}' contains non-finite signal samples")
    if sampling_rate_hz is not None:
        expected_dt = 1.0 / float(sampling_rate_hz)
        if time_values.size == 0:
            raise ValueError(f"Run '{run_ref}' contains no time samples")
        if end_time_input_s is not None:
            expected_count = max(
                0,
                int(np.floor(float(end_time_input_s) * float(sampling_rate_hz))) - 1,
            )
            if time_values.size != expected_count:
                raise ValueError(
                    f"Run '{run_ref}' must contain exactly {expected_count} samples; "
                    f"got {time_values.size}"
                )
            if float(time_values[-1]) >= float(end_time_input_s):
                raise ValueError(
                    "Run "
                    f"'{run_ref}' last sample must be strictly smaller than "
                    f"end_time_input_s={float(end_time_input_s)}"
                )
        expected_times = expected_dt * np.arange(1, time_values.size + 1, dtype=np.float64)
        grid_atol = max(
            1e-6,
            abs(expected_dt) * 1e-6,
            np.finfo(np.float32).eps * max(1.0, float(expected_times[-1])) * 8.0,
        )
        first_time = float(time_values[0])
        if not np.isclose(first_time, expected_dt, rtol=0.0, atol=grid_atol):
            raise ValueError(
                f"Run '{run_ref}' first sample must be at {expected_dt}s, got {first_time}s"
            )
        if not np.allclose(
            time_values,
            expected_times,
            rtol=0.0,
            atol=grid_atol,
        ):
            raise ValueError(
                f"Run '{run_ref}' samples are not regularly spaced at {expected_dt}s"
            )


__all__ = [
    "BuildTVPFn",
    "FEATURES_FILENAME",
    "SimulateCaseFn",
    "compute_problem_specific_features",
    "dataframe_to_signal_dict",
    "resample_dataframe",
    "resample_signal_dict",
    "save_feature_dict",
    "serialize",
    "signal_dict_to_dataframe",
    "simulate_recipe",
    "validate",
]
