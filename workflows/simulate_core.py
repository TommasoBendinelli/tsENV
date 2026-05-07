"""
Slim simulation helpers shared by workflows.

This module provides the subset of simulation helpers needed by
`workflows/simulate/run_pending_sims.py` and `workflows/simulate/build_metadata.py`.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matlab.engine as _matlab_engine  # type: ignore
except Exception:  # noqa: BLE001
    _matlab_engine = None


class _MatlabExecutionErrorFallback(Exception):
    pass


class _MatlabTimeoutErrorFallback(Exception):
    pass


class _MatlabInterruptedErrorFallback(Exception):
    pass


class _MatlabCancelledErrorFallback(Exception):
    pass


_MatlabExecutionError = (
    _matlab_engine.MatlabExecutionError
    if _matlab_engine is not None and hasattr(_matlab_engine, "MatlabExecutionError")
    else _MatlabExecutionErrorFallback
)
_MatlabTimeoutError = (
    _matlab_engine.TimeoutError
    if _matlab_engine is not None and hasattr(_matlab_engine, "TimeoutError")
    else _MatlabTimeoutErrorFallback
)
_MatlabInterruptedError = (
    _matlab_engine.InterruptedError
    if _matlab_engine is not None and hasattr(_matlab_engine, "InterruptedError")
    else _MatlabInterruptedErrorFallback
)
_MatlabCancelledError = (
    _matlab_engine.CancelledError
    if _matlab_engine is not None and hasattr(_matlab_engine, "CancelledError")
    else _MatlabCancelledErrorFallback
)

from shared.intervention_sampling import InitialSamplingState
from shared.matlab_runtime import (
    MatlabSegmentFailure,
    MatlabUserInterrupt,
    _MatlabStream,
    _reset_working_copy,
    force_stop_matlab_processes,
    is_matlab_user_interrupt_text,
)
from shared.metrics import step
from shared.simulink_utils import (
    LOCAL_SOLVER_STEP_SIZE_VARIABLE,
    MATLAB_IDENTIFIER_RE,
    PRIMARY_STOP_TIME_WORKSPACE_VARIABLE,
    _ensure_model_stopped,
    _ident_to_working,
    get_configured_stop_time,
    save_model_with_workspace_values,
)

logger = logging.getLogger(__name__)

_SIMULATION_TIMEOUT_SECONDS = float(os.environ.get("SIMULATE_RUN_TIMEOUT", "90"))
_SEGMENT_FAILURE_RE = re.compile(r"Segment:\\s*(\\d+)\\s+failed", re.IGNORECASE)
_FIXED_STEP_SOLVERS = {
    "fixedstepdiscrete",
    "ode1",
    "ode2",
    "ode3",
    "ode4",
    "ode5",
    "ode8",
    "ode14x",
}
TimingCallback = Callable[..., None]


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


def _matlab_make_valid_name(name: str) -> str:
    """Approximate MATLAB's makeValidName used in sim_the_model.m."""
    if not name:
        return "x"
    cleaned = re.sub(r"[^0-9A-Za-z_]", "", str(name))
    if not cleaned:
        return "x"
    if not re.match(r"[A-Za-z]", cleaned[0]):
        cleaned = f"x{cleaned}"
    return cleaned


def _coerce_positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed) or parsed <= 0.0:
        return None
    return float(parsed)


def _matlab_session_started_at(mle: Any) -> Optional[float]:
    try:
        value = getattr(mle, "_tsenv_session_started_at", None)
    except Exception:
        return None
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _matlab_interrupt_message(*parts: Any) -> str:
    for part in parts:
        text = str(part or "").strip()
        if text and is_matlab_user_interrupt_text(text):
            return text
    for part in parts:
        text = str(part or "").strip()
        if text:
            return text
    return "the MATLAB function has been cancelled"


def _raise_matlab_interrupt(
    exc: Optional[BaseException] = None,
    *parts: Any,
) -> None:
    message = _matlab_interrupt_message(*parts, exc)
    interrupt = MatlabUserInterrupt(message)
    if exc is not None:
        raise interrupt from exc
    raise interrupt


def _is_matlab_interrupt_exception(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, _MatlabInterruptedError, _MatlabCancelledError)) or (
        is_matlab_user_interrupt_text(str(exc))
    )


def _raise_if_matlab_interrupt(
    exc: BaseException,
    *,
    stdout_text: str = "",
    stderr_text: str = "",
) -> None:
    combined = "\n".join(
        text for text in (str(exc or ""), stdout_text, stderr_text) if str(text or "").strip()
    )
    if _is_matlab_interrupt_exception(exc) or is_matlab_user_interrupt_text(combined):
        _raise_matlab_interrupt(exc, combined)


def _assert_model_stop_time_matches_expected(
    mle: Any,
    *,
    model_name: str,
    expected_stop_time: float,
) -> float:
    actual_stop_time = get_configured_stop_time(mle, model_name)
    tolerance = max(1e-9, abs(float(expected_stop_time)) * 1e-9)
    if abs(actual_stop_time - float(expected_stop_time)) <= tolerance:
        return float(actual_stop_time)
    raise RuntimeError(
        f"Configured model StopTime mismatch for '{model_name}': "
        f"expected applied end_time_input_s={float(expected_stop_time):.15g}s but model "
        f"resolves to {actual_stop_time:.15g}s after rewriting the working model."
    )


def _build_model_workspace_from_recipe(recipe: Dict[str, Any]) -> Dict[str, float]:
    variable_values = recipe.get("baseline_parameters")
    if not isinstance(variable_values, Mapping):
        initial_state = recipe.get("initial_state")
        variable_values = getattr(initial_state, "variable_values", {}) or {}
    if not isinstance(variable_values, Mapping):
        raise TypeError(
            "recipe.baseline_parameters must be a mapping when building the model workspace"
        )

    model_workspace: Dict[str, float] = {}
    for var_name, raw_value in variable_values.items():
        if not isinstance(var_name, str) or not MATLAB_IDENTIFIER_RE.fullmatch(var_name):
            raise ValueError(
                f"recipe baseline/model workspace contains an invalid MATLAB identifier: {var_name!r}"
            )
        try:
            value = float(raw_value)
        except Exception as exc:
            raise ValueError(
                f"recipe baseline/model workspace value {var_name!r} must be numeric"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"recipe baseline/model workspace value {var_name!r} must be finite"
            )
        model_workspace[var_name] = float(value)

    end_time_input_s = _coerce_positive_float(recipe.get("end_time_input_s"))
    if end_time_input_s is None:
        raise ValueError("recipe.end_time_input_s must be present and positive")
    model_workspace[PRIMARY_STOP_TIME_WORKSPACE_VARIABLE] = float(end_time_input_s)
    solver_step = _coerce_positive_float(recipe.get("solver_step"))
    if solver_step is not None:
        model_workspace[LOCAL_SOLVER_STEP_SIZE_VARIABLE] = float(solver_step)
    return model_workspace


def _recipe_initial_parameter_values(recipe: Dict[str, Any]) -> Mapping[Tuple[str, str], Any]:
    values = recipe.get("initial_parameter_values")
    if isinstance(values, Mapping):
        return values
    initial_state = recipe.get("initial_state")
    values = getattr(initial_state, "parameter_values", {}) or {}
    if isinstance(values, Mapping):
        return values
    raise ValueError(
        "recipe.initial_parameter_values must be provided for runtime parameter initialization"
    )


def apply_ModelWorkspace(
    mle: Any,
    model_workspace: Mapping[str, Any],
    *,
    model_name: str = "simulink_model",
) -> None:
    if not isinstance(model_workspace, Mapping):
        raise TypeError("model_workspace must be a mapping")

    try:
        mle.eval(
            f"mw = get_param('{model_name}','ModelWorkspace'); clear mw;",
            nargout=0,
        )
    except Exception as exc:
        _raise_if_matlab_interrupt(exc)
        raise RuntimeError(
            f"Unable to access ModelWorkspace for model '{model_name}'"
        ) from exc

    try:
        for var_name, raw_value in model_workspace.items():
            if not isinstance(var_name, str) or not MATLAB_IDENTIFIER_RE.fullmatch(var_name):
                raise ValueError(
                    f"model_workspace contains an invalid MATLAB identifier: {var_name!r}"
                )
            try:
                value = float(raw_value)
            except Exception as exc:
                raise ValueError(
                    f"model_workspace[{var_name!r}] must be numeric"
                ) from exc
            if not np.isfinite(value):
                raise ValueError(
                    f"model_workspace[{var_name!r}] must be finite"
                )

            mle.workspace["model_workspace_value_tmp"] = float(value)
            try:
                mle.eval(
                    "mw = get_param("
                    f"'{model_name}'"
                    ",'ModelWorkspace'); "
                    f"assignin(mw,'{var_name}',model_workspace_value_tmp); clear mw;",
                    nargout=0,
                )
            except Exception as exc:
                _raise_if_matlab_interrupt(exc)
                raise RuntimeError(
                    f"Failed to assign ModelWorkspace variable {var_name!r} on '{model_name}'"
                ) from exc
    finally:
        with suppress(Exception):
            mle.eval("clear model_workspace_value_tmp", nargout=0)


def _resolve_fixed_step_solver_name(optimizer_info: Optional[Dict[str, Any]]) -> str:
    info = optimizer_info if isinstance(optimizer_info, dict) else {}
    solver_raw = str(info.get("Solver") or "").strip()
    solver_type = str(info.get("SolverType") or "").strip().lower()
    solver_key = solver_raw.lower()

    if solver_type == "fixed-step" and solver_raw:
        return solver_raw
    if solver_key in _FIXED_STEP_SOLVERS:
        return solver_raw
    if "discrete" in solver_key:
        return "FixedStepDiscrete"
    return "ode4"


def _series_to_signal_payload(series: pd.Series) -> Dict[str, np.ndarray]:
    return {
        "timestamp": pd.to_numeric(series.index, errors="coerce").to_numpy(dtype=float),
        "data": pd.to_numeric(series, errors="coerce").to_numpy(dtype=float),
    }


def _raw_signal_dict_sample_count(signal_dict: Mapping[str, Any]) -> int:
    finite_time_values: List[np.ndarray] = []
    for payload in signal_dict.values():
        if not isinstance(payload, Mapping):
            continue
        timestamps = payload.get("timestamp")
        if timestamps is None:
            continue
        time_values = np.asarray(timestamps, dtype=float).reshape(-1)
        finite = time_values[np.isfinite(time_values)]
        if finite.size:
            finite_time_values.append(finite)
    if not finite_time_values:
        return 0
    return int(np.unique(np.concatenate(finite_time_values)).shape[0])


def _extract_simulink_signal_dict(
    res: Mapping[str, Any],
    *,
    requested_signals: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    if res is None:
        _raise_matlab_interrupt(None, "the MATLAB function has been cancelled")
    if not isinstance(res, Mapping):
        raise TypeError(f"Expected MATLAB simulation result mapping, got {type(res).__name__}")
    skip_fields = {"OperatingPoint", "simlogSegments", "simlog", "simscapeMerged"}
    available_signals: Dict[str, pd.Series] = {}
    for name, block in res.items():
        if name in skip_fields or "Signal_" in str(name):
            continue
        if not isinstance(block, Mapping):
            continue
        if "Time" not in block or "Data" not in block:
            continue

        time_values = np.asarray(block["Time"]).reshape(-1).astype(float)
        data_values = np.asarray(block["Data"])
        if data_values.ndim != 1:
            data_values = data_values.reshape(-1)
        data_values = data_values.astype(float)
        if time_values.shape[0] != data_values.shape[0]:
            continue
        series = pd.Series(data_values, index=time_values, name=str(name))
        series.index.name = "time_s"
        series = series.groupby(level=0).mean().sort_index()
        available_signals[str(name)] = series

    if not available_signals:
        return {}

    requested = None if requested_signals is None else [str(signal) for signal in requested_signals]
    if requested is not None and not requested:
        return {}

    if requested is None:
        selected_targets = list(available_signals.keys())
        selected_sources = list(available_signals.keys())
    else:
        selected_targets = []
        selected_sources = []
        for target in requested:
            if target in available_signals:
                selected_targets.append(target)
                selected_sources.append(target)
                continue
            matlab_name = _matlab_make_valid_name(target)
            if matlab_name in available_signals:
                selected_targets.append(target)
                selected_sources.append(matlab_name)
        if not selected_sources:
            raise ValueError("None of simulink_signals are present in simulation outputs")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for target, source in zip(selected_targets, selected_sources):
        out[target] = _series_to_signal_payload(available_signals[source].rename(target))
    return out


def _extract_simscape_signal_dict(
    res: Mapping[str, Any],
    *,
    matlab_engine: Any,
    requested_signals: Optional[Sequence[str]] = None,
    stop_time: Optional[float] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    if res is None:
        _raise_matlab_interrupt(None, "the MATLAB function has been cancelled")
    requested = None if requested_signals is None else [str(signal) for signal in requested_signals]
    if requested is not None and not requested:
        return {}

    simscape_frames = get_time_series_simscapes(
        res,
        matlab_engine=matlab_engine,
        requested_signals=requested,
        stop_time=stop_time,
    )
    if not simscape_frames:
        return {}

    requested_set = set(requested) if requested is not None else None
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, pdf in simscape_frames.items():
        signal_name = str(name)
        if requested_set is not None and signal_name not in requested_set:
            continue
        if pdf.empty:
            continue
        col = "value_0" if "value_0" in pdf.columns else str(pdf.columns[0])
        series = pdf[col].rename(signal_name)
        series = series.groupby(level=0).mean().sort_index()
        out[signal_name] = _series_to_signal_payload(series)

    if requested is None:
        return out
    return {signal: out[signal] for signal in requested if signal in out}
def get_time_series_simscapes(
    res: Mapping[str, Any],
    *,
    matlab_engine: Any,
    requested_signals: Optional[Sequence[str]] = None,
    stop_time: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Extract Simscape log signals into per-path DataFrames.

    Requires a live MATLAB engine because Simscape log objects cannot be
    deserialized directly in Python.
    """
    if matlab_engine is None:
        raise ValueError("matlab_engine is required to extract Simscape time series")

    def _block_to_df(block: Mapping[str, Any]) -> pd.DataFrame:
        t = np.asarray(block["Time"]).reshape(-1).astype(float)
        data = np.asarray(block["Data"])
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.shape[0] != t.shape[0] and data.shape[1] == t.shape[0]:
            data = data.T
        cols = [f"value_{i}" for i in range(data.shape[1])]
        df_block = pd.DataFrame(data, index=t, columns=cols)
        df_block.index.name = "time_s"
        return df_block

    def _iter_struct_items(struct_obj: Any):
        if isinstance(struct_obj, Mapping):
            return struct_obj.items()
        fieldnames = getattr(struct_obj, "_fieldnames", None)
        if fieldnames:
            return [(fn, getattr(struct_obj, fn, None)) for fn in fieldnames]
        return []

    def _get_field(obj: Any, name: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(name)
        if hasattr(obj, name):
            return getattr(obj, name)
        try:
            return obj[name]
        except Exception:
            return None

    simlog_segments = res.get("simlogSegments") or []
    if not simlog_segments:
        return {}

    def _simlog_to_struct(simlog_obj: Any) -> Optional[Mapping[str, Any]]:
        if simlog_obj is None:
            return None

        matlab_engine.workspace["simscape_log_tmp"] = simlog_obj
        try:
            matlab_engine.eval(
                r"""
simscape_struct_tmp = struct();

stack = {simscape_log_tmp};
path_stack = {''};

while ~isempty(stack)
    node = stack{1};
    node_path = path_stack{1};
    stack(1) = [];
    path_stack(1) = [];

    % 1) If this node has a non-empty series, record it
    try
        s = node.series;  % simscape.logging.Series (may be empty)
        if s.points > 0
            t = time(s);                  % row vector
            v = values(s);                % steps x dims
            if isempty(node_path)
                nm = node.id;
            else
                nm = node_path;
            end
            % Turn path into a valid field name
            fn = matlab.lang.makeValidName(strrep(nm, '.', '_'));
            simscape_struct_tmp.(fn).Time = t(:);  % column
            simscape_struct_tmp.(fn).Data = v;
        end
    catch
        % accessing series on non-variable nodes is okay, it just returns empty
    end

    % 2) Traverse children that are Nodes
    f = fieldnames(node);
    for k = 1:numel(f)
        child = node.(f{k});
        if isa(child, 'simscape.logging.Node')
            if isempty(node_path)
                new_path = f{k};
            else
                new_path = [node_path '.' f{k}];
            end
            stack{end+1} = child;          %#ok<AGROW>
            path_stack{end+1} = new_path;  %#ok<AGROW>
        end
    end
end
""",
                nargout=0,
            )
            struct_data = matlab_engine.workspace["simscape_struct_tmp"]
        finally:
            with suppress(Exception):
                matlab_engine.eval(
                    "clear simscape_log_tmp simscape_struct_tmp stack path_stack node node_path s t v f k new_path nm fn",
                    nargout=0,
                )
        return struct_data

    merged: Dict[str, pd.DataFrame] = {}
    for simlog in simlog_segments:
        seg_struct = _simlog_to_struct(simlog)
        if not seg_struct:
            continue
        for path, block in _iter_struct_items(seg_struct):  # type: ignore[assignment]
            time_field = _get_field(block, "Time")
            data_field = _get_field(block, "Data")
            if time_field is None or data_field is None:
                continue
            df_block = _block_to_df({"Time": time_field, "Data": data_field})
            merged[path] = (
                pd.concat([merged[path], df_block]).sort_index()
                if path in merged
                else df_block
            )

    return merged


def _apply_initial_values_to_model(mle: Any, recipe: Dict[str, Any]) -> None:
    initial_state = recipe["initial_state"]
    inits = initial_state.parameter_values
    for address, kv in inits.items():
        ident_working = _ident_to_working(address[0])
        try:
            mle.set_param(ident_working, address[1], str(kv), nargout=0)
        except Exception as exc:  # noqa: BLE001
            _raise_if_matlab_interrupt(exc)
            logger.warning(
                "set_param failed for %s / %s=%s :: %s",
                ident_working,
                address[1],
                kv,
                exc,
            )

    # Also push workspace-only initial variables (not mapped to block parameters).
    # BallDrop's `initial_velocity` is one such variable and must be set in model
    # workspace before simulation starts.
    variable_values = getattr(initial_state, "variable_values", {}) or {}
    if not isinstance(variable_values, Mapping):
        return

    for var_name, raw_value in variable_values.items():
        if not isinstance(var_name, str) or not MATLAB_IDENTIFIER_RE.fullmatch(var_name):
            logger.warning("Skipping invalid MATLAB identifier in initial variables: %r", var_name)
            continue
        try:
            value = float(raw_value)
        except Exception:  # noqa: BLE001
            logger.warning("Skipping non-numeric initial variable %s=%r", var_name, raw_value)
            continue
        if not np.isfinite(value):
            logger.warning("Skipping non-finite initial variable %s=%r", var_name, raw_value)
            continue
        try:
            mle.workspace["initial_state_value_tmp"] = value
            mle.eval(
                "mw = get_param('simulink_model','ModelWorkspace'); "
                f"assignin(mw,'{var_name}',initial_state_value_tmp); clear mw;",
                nargout=0,
            )
        except Exception as exc:  # noqa: BLE001
            _raise_if_matlab_interrupt(exc)
            try:
                mle.eval(
                    f"assignin('base','{var_name}',initial_state_value_tmp);",
                    nargout=0,
                )
            except Exception as base_exc:  # noqa: BLE001
                _raise_if_matlab_interrupt(base_exc)
                logger.warning(
                    "Failed to assign workspace variable %s=%s (model/base): %s | %s",
                    var_name,
                    value,
                    exc,
                    base_exc,
                )
    with suppress(Exception):
        mle.eval("clear initial_state_value_tmp", nargout=0)


def _build_df_for_modification(recipe: Dict[str, Any], modification: Dict[str, Any]) -> pd.DataFrame:
    solver_step = float(recipe["solver_step"])
    end_time_input_s = float(recipe["end_time_input_s"])
    n_points = int(round(end_time_input_s / solver_step)) + 1
    times = np.arange(n_points, dtype=float) * solver_step

    ident = modification["identifier"]
    key = modification["key"]
    if ident is None or key is None:
        raise ValueError("Modification is missing identifier or key")

    init_val = _recipe_initial_parameter_values(recipe)[(ident, key)]
    return pd.DataFrame({"time": times, ident: float(init_val)})


def _apply_modification(df: pd.DataFrame, modification: Dict[str, Any], intervention_time: float) -> pd.DataFrame:
    ident = modification["identifier"]
    trans = modification["transition_type"]
    if trans != "step":
        raise ValueError(f"Unsupported transition_type: {trans!r}")
    new_value = float(modification["new_value"])
    return step(df, target_column=ident, time=intervention_time, end_value=new_value)


def _tvp_from_df(
    df: pd.DataFrame,
    modification: Dict[str, Any],
    *,
    container: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import matlab

    ident = modification["identifier"]
    key = modification["key"]

    def compress_df(df_inner: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        atol = 1e-12
        change_indices = [0]
        values = df_inner[target_column].values
        for i in range(1, len(df_inner)):
            if not np.isclose(values[i], values[i - 1], rtol=0.0, atol=atol):
                change_indices.append(i)
        time_delta = df_inner["time"].iloc[change_indices].values
        values_delta = values[change_indices]
        return time_delta, values_delta

    t, v = compress_df(df, target_column=ident)

    container = container or {"identifier": [], "key": [], "time": [], "values": [], "seen": []}
    container["identifier"].append(_ident_to_working(ident) if ident is not None else "None")
    container["key"].append(key if ident is not None else "None")
    container["time"].append(matlab.double([float(x) for x in t]))
    container["values"].append(matlab.double([float(y) for y in v]))
    container["seen"].append(matlab.double([0 for _ in range(len(v))]))
    return container


def _build_complete_tvp(
    recipe: Dict[str, Any],
    all_params: List[Tuple[str, str]],
    modifications_to_apply: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    import matlab

    end_time_input_s = float(recipe["end_time_input_s"])
    init_values = _recipe_initial_parameter_values(recipe)

    tvp: Dict[str, Any] = {"identifier": [], "key": [], "time": [], "values": [], "seen": []}
    for ident, key in all_params:
        init_val = init_values[(ident, key)]
        param_mods = [
            mod
            for mod in (modifications_to_apply or [])
            if mod["identifier"] == ident and mod["key"] == key
        ]
        if param_mods:
            df_mod = _build_df_for_modification(recipe, {"identifier": ident, "key": key})
            for mod in sorted(param_mods, key=lambda x: x["start_time"]):
                df_mod = _apply_modification(df_mod, mod, mod["start_time"])
            _tvp_from_df(df_mod, {"identifier": ident, "key": key}, container=tvp)
        else:
            tvp["identifier"].append(_ident_to_working(ident))
            tvp["key"].append(key)
            tvp["time"].append(matlab.double([[0.0, end_time_input_s]]))
            tvp["values"].append(matlab.double([[float(init_val), float(init_val)]]))
            tvp["seen"].append(matlab.double([[0.0, 0.0]]))

    return tvp


def _apply_optimizer_info_to_model(
    mle: Any,
    optimizer_info: Optional[Dict[str, Any]],
    *,
    solver_step: Optional[float] = None,
) -> None:
    if not optimizer_info:
        return

    def _set_params(target: str, params: Dict[str, Any]) -> None:
        for param_name, param_value in params.items():
            if isinstance(param_value, bool):
                param_value_str = "on" if param_value else "off"
            else:
                param_value_str = str(param_value)
            try:
                mle.set_param(target, param_name, param_value_str, nargout=0)
            except _MatlabExecutionError as exc:
                _raise_if_matlab_interrupt(exc)
                logger.warning(
                    "Failed to set parameter '%s'='%s' on %s: %s",
                    param_name,
                    param_value_str,
                    target,
                    exc,
                )

    def _find_solver_configuration_block(model_name: str = "simulink_model") -> Optional[str]:
        try:
            sc_blocks = mle.find_system(
                model_name,
                "LookUnderMasks",
                "all",
                "FollowLinks",
                "on",
                "MaskType",
                "Solver Configuration",
                nargout=1,
            )
        except Exception as exc:
            _raise_if_matlab_interrupt(exc)
            return None

        raw_blocks: List[Any]
        if sc_blocks is None:
            raw_blocks = []
        elif isinstance(sc_blocks, str):
            raw_blocks = [sc_blocks]
        elif isinstance(sc_blocks, (list, tuple)):
            raw_blocks = list(sc_blocks)
        else:
            try:
                raw_blocks = list(sc_blocks)
            except Exception:
                raw_blocks = [sc_blocks]

        block_paths: List[str] = []
        seen_paths = set()
        for raw_block in raw_blocks:
            block_path = str(raw_block or "").strip()
            if not block_path or block_path in seen_paths:
                continue
            seen_paths.add(block_path)
            block_paths.append(block_path)

        if not block_paths:
            return None
        if len(block_paths) > 1:
            logger.warning(
                "Found %d Solver Configuration blocks in %s; using %s",
                len(block_paths),
                model_name,
                block_paths[0],
            )
        return block_paths[0]

    simscape_solver = optimizer_info.get("simscape_solver")
    if simscape_solver:
        solver_block = _find_solver_configuration_block("simulink_model")
        if not solver_block:
            raise RuntimeError(
                "No Simscape Solver Configuration block found in model "
                "'simulink_model' while applying optimizer_info.simscape_solver."
            )
        _set_params(solver_block, dict(simscape_solver))
        return

    resolved_solver_step = _coerce_positive_float(solver_step)
    if resolved_solver_step is None:
        return

    _set_params(
        "simulink_model",
        {
            "SolverType": "Fixed-step",
            "Solver": _resolve_fixed_step_solver_name(optimizer_info),
            "FixedStep": resolved_solver_step,
        },
    )


def run_simulation(
    mle: Any,
    *,
    stop_time: float,
    debug_dir: Path,
    time_varying_parameters: Optional[dict] = None,
    debug: bool = False,
    sim_the_model_path: str = "",
    save_simscape_mat: bool = False,
    timing_callback: Optional[TimingCallback] = None,
    run_id: Optional[str] = None,
):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not sim_the_model_path or not Path(sim_the_model_path).is_file():
        raise ValueError(f"sim_the_model_path not found: {sim_the_model_path}")

    args: List[Any] = [
        "DebugDataPath",
        str(debug_dir),
        "debug",
        bool(debug),
        "SaveSimscapeMat",
        bool(save_simscape_mat),
    ]
    if time_varying_parameters is not None:
        args += ["TimeVaryingParameters", time_varying_parameters]

    tmp_script = Path("sim_the_model.m")
    with _timed_phase(timing_callback, "copy_sim_script", run_id=run_id):
        shutil.copy(sim_the_model_path, tmp_script)

    matlab_stdout = _MatlabStream(level=logging.INFO, prefix="MATLAB stdout")
    matlab_stderr = _MatlabStream(level=logging.ERROR, prefix="MATLAB stderr")
    future = None
    interrupted = False
    try:
        with _timed_phase(timing_callback, "matlab_sim_the_model", run_id=run_id):
            future = mle.sim_the_model(*args, stdout=matlab_stdout, stderr=matlab_stderr, background=True)
            result = future.result(timeout=_SIMULATION_TIMEOUT_SECONDS)
        stdout_text = matlab_stdout.captured_text()
        stderr_text = matlab_stderr.captured_text()
        combined_output = "\n".join(
            text for text in (stdout_text, stderr_text) if str(text or "").strip()
        )
        if is_matlab_user_interrupt_text(combined_output):
            interrupted = True
            _raise_matlab_interrupt(None, combined_output)
        if result is None:
            raise RuntimeError("MATLAB simulation returned no result")
        if stdout_text:
            print(f"MATLAB stdout:\n{stdout_text}")
        if stderr_text:
            print(f"MATLAB stderr:\n{stderr_text}", file=sys.stderr)
        return result
    except KeyboardInterrupt as exc:
        interrupted = True
        if future is not None:
            with suppress(Exception):
                future.cancel()
        _raise_matlab_interrupt(exc)
    except (_MatlabInterruptedError, _MatlabCancelledError) as exc:
        interrupted = True
        if future is not None:
            with suppress(Exception):
                future.cancel()
        _raise_matlab_interrupt(exc)
    except _MatlabTimeoutError as exc:
        if future is not None:
            with suppress(Exception):
                future.cancel()
        logger.error("Simulation timed out after %.1fs: %s", _SIMULATION_TIMEOUT_SECONDS, exc)
        _ensure_model_stopped(mle, model_name="simulink_model")
        raise TimeoutError(f"Simulation exceeded {_SIMULATION_TIMEOUT_SECONDS:.1f} seconds") from exc
    except _MatlabExecutionError as exc:
        stdout_text = matlab_stdout.captured_text()
        stderr_text = matlab_stderr.captured_text()
        combined = "\n".join(
            text for text in (str(exc or ""), stdout_text, stderr_text) if str(text or "").strip()
        )
        if _is_matlab_interrupt_exception(exc) or is_matlab_user_interrupt_text(combined):
            interrupted = True
            _raise_matlab_interrupt(exc, combined)
        sections = []
        if stdout_text:
            sections.append(f"MATLAB stdout:\n{stdout_text}")
        if stderr_text:
            sections.append(f"MATLAB stderr:\n{stderr_text}")
        details = "\n\n".join(sections) if sections else None
        segment_failure: Optional[int] = None
        failed_to_converge = False
        if stderr_text:
            match = _SEGMENT_FAILURE_RE.search(stderr_text)
            if match:
                with suppress(ValueError):
                    segment_failure = int(match.group(1))
            if "solver failed to converge" in stderr_text:
                failed_to_converge = True
        message = str(exc).strip() or "MATLAB simulation failed"
        raise MatlabSegmentFailure(
            message,
            segment_failure=segment_failure,
            details=details,
            original_exception=exc,
            failed_to_converge=failed_to_converge,
        ) from exc
    finally:
        matlab_stdout.flush()
        matlab_stderr.flush()
        if interrupted:
            with suppress(Exception):
                if future is not None:
                    future.cancel()
            force_stop_matlab_processes(
                started_at=_matlab_session_started_at(mle),
                reason="simulation interrupt",
            )
        else:
            with suppress(Exception):
                _ensure_model_stopped(mle, model_name="simulink_model")
            with suppress(Exception):
                mle.set_param("simulink_model", "FastRestart", "off", nargout=0)
            with suppress(Exception):
                mle.close_system("simulink_model", 0, nargout=0)
        with suppress(FileNotFoundError, PermissionError):
            tmp_script.unlink()


def _simulate_case_to_signal_dict(
    mle: Any,
    recipe: Dict[str, Any],
    *,
    tvp: Optional[Dict[str, Any]],
    debug_dir: Path,
    expected_stop_time: float,
    sim_script: Path,
    optimizer_info: Optional[Dict[str, Any]] = None,
    simscape_signals: Optional[List[str]] = None,
    simulink_signals: Optional[List[str]] = None,
    runtime_model_snapshot_path: Optional[Path] = None,
    save_simscape_mat: bool = False,
    debug: bool = False,
    timing_callback: Optional[TimingCallback] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    if mle is None:
        raise KeyboardInterrupt("MATLAB engine not available; aborting scenario")
    debug_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(recipe.get("run_id") or recipe.get("id") or "") or None
    try:
        with _timed_phase(timing_callback, "reset_working_copy", run_id=run_id):
            _reset_working_copy()
        working_model_mdl = Path("simulink_model.mdl")
        model_workspace = _build_model_workspace_from_recipe(recipe)
        if not working_model_mdl.exists():
            raise FileNotFoundError(
                "Working model copy missing after _reset_working_copy(): simulink_model.mdl"
            )
        with _timed_phase(timing_callback, "save_model_workspace", run_id=run_id):
            save_model_with_workspace_values(
                working_model_mdl,
                model_workspace,
                source_path=working_model_mdl,
            )
        # Avoid Simulink shadowing a same-name .mdl with a .slx in the folder.
        with _timed_phase(timing_callback, "matlab_load_system", run_id=run_id):
            mle.load_system(str(working_model_mdl.resolve()))
        with _timed_phase(timing_callback, "apply_model_workspace", run_id=run_id):
            apply_ModelWorkspace(
                mle,
                model_workspace,
                model_name="simulink_model",
            )
        with _timed_phase(timing_callback, "assert_model_stop_time", run_id=run_id):
            resolved_stop_time = _assert_model_stop_time_matches_expected(
                mle,
                model_name="simulink_model",
                expected_stop_time=float(expected_stop_time),
            )
        recipe["end_time_input_s"] = float(resolved_stop_time)
        resolved_model_workspace = _build_model_workspace_from_recipe(recipe)
        if resolved_model_workspace != model_workspace:
            with _timed_phase(timing_callback, "save_resolved_model_workspace", run_id=run_id):
                save_model_with_workspace_values(
                    working_model_mdl,
                    resolved_model_workspace,
                    source_path=working_model_mdl,
                )
        if runtime_model_snapshot_path is not None:
            runtime_model_snapshot_path = Path(runtime_model_snapshot_path)
            with _timed_phase(timing_callback, "runtime_model_snapshot", run_id=run_id):
                runtime_model_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(working_model_mdl, runtime_model_snapshot_path)
            logger.info(
                "Saved runtime-equivalent model snapshot: %s",
                runtime_model_snapshot_path.resolve(),
            )

        with _timed_phase(timing_callback, "run_simulation", run_id=run_id):
            res = run_simulation(
                mle=mle,
                stop_time=resolved_stop_time,
                debug_dir=debug_dir,
                time_varying_parameters=tvp,
                debug=debug,
                sim_the_model_path=str(sim_script),
                save_simscape_mat=save_simscape_mat,
                timing_callback=timing_callback,
                run_id=run_id,
            )
    except Exception as exc:
        _raise_if_matlab_interrupt(exc)
        raise

    raw_signal_dict: Dict[str, Dict[str, np.ndarray]] = {}
    with _timed_phase(timing_callback, "extract_simulink_signals", run_id=run_id):
        raw_signal_dict.update(
            _extract_simulink_signal_dict(
                res,
                requested_signals=simulink_signals,
            )
        )
    with _timed_phase(timing_callback, "extract_simscape_signals", run_id=run_id):
        raw_signal_dict.update(
            _extract_simscape_signal_dict(
                res,
                matlab_engine=mle,
                requested_signals=simscape_signals,
                stop_time=resolved_stop_time,
            )
        )

    if _raw_signal_dict_sample_count(raw_signal_dict) <= 10:
        raise ValueError("Simulation returned too few samples")
    return raw_signal_dict
