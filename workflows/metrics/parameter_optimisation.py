from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import textwrap
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.parse import urlencode

import click
import numpy as np
import pandas as pd

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.interface.distribution_json import load_experiment_config_json  # noqa: E402
from shared.interface.model_record_json import (  # noqa: E402
    load_model_record_json,
    load_model_run_specs_json,
)
from shared.interface.simulink_metadata_json import (  # noqa: E402
    load_simulink_generated_metadata,
)
from shared.intervention_sampling import eval_numeric_expr  # noqa: E402
from shared.model_noise_adder import (  # noqa: E402
    call_noise_adder,
    load_model_noise_adder,
    normalize_noise_profile,
)
from shared.question_eligibility import is_success_status  # noqa: E402
from shared.run_artifacts import resolve_model_record_path, resolve_runs_root  # noqa: E402
from shared.simulink_utils import MATLAB_IDENTIFIER_RE  # noqa: E402
from shared.time_series_metrics import load_run_df  # noqa: E402
import workflows.simulate.build_metadata as build_metadata  # noqa: E402


REPORT_FILENAME = "optimisation_results.json"
_FAILED_CANDIDATE_LOSS = 1.0e12
_EPS = 1.0e-12
_DEFAULT_NOISE_SEED = 0
_DEFAULT_PARAMETER_VALUE_SRD_MATCH_THRESHOLD = 0.02
_IMPULSE_EVENT_THRESHOLD_FRACTION = 0.05
_IMPULSE_EVENT_MATCH_WINDOW_S = 0.1
_IMPULSE_EVENT_UNMATCHED_PENALTY = 1.0
_LOSS_NORMALISATION_FLOOR = 1.0e-6
_COARSE_GRID_EARLY_STOP_LOSS = 1.0e-4
_BALL_DROP_MODEL_ID = "balldrop"
_BALL_DROP_IMPULSE_SIGNAL = "Hard_Stop_f"
_BALL_DROP_VELOCITY_SIGNAL = "Velocity"
_BALL_DROP_VELOCITY_CROSSOVER_WINDOW_S = 0.1
_WEBAPP_BASE_URL = "http://localhost:3001/"
_MATLAB_WORKERS_ENV = "TSENV_MATLAB_WORKERS"
_DEFAULT_MATLAB_WORKERS = min(16, os.cpu_count() or 1)


@dataclass(frozen=True)
class ChildRunContext:
    run_id: str
    baseline_run_id: str
    time0_baseline_run_id: Optional[str]
    ground_truth_parameter: str
    intervention_time: float
    baseline_parameters: Dict[str, Any]
    child_parameters: Dict[str, Any]
    child_set_value: Any
    child_spec: Dict[str, Any]
    runtime_entry: Dict[str, Any]


@dataclass(frozen=True)
class CandidateBinding:
    path: str
    name: str
    expression: str


@dataclass(frozen=True)
class CandidateSpec:
    parameter: str
    minimum: float
    maximum: float
    sampling_strategy: str
    bindings: tuple[CandidateBinding, ...]


@dataclass(frozen=True)
class BaselineBlockBinding:
    path: str
    name: str
    value: float


@dataclass(frozen=True)
class CandidateResult:
    parameter: str
    status: str
    loss: Optional[float] = None
    p_opt: Optional[float] = None
    optimisation_loss: Optional[float] = None
    continuous_loss: Optional[float] = None
    impulse_loss: Optional[float] = None
    final_profile_losses: Optional[Mapping[str, Mapping[str, Optional[float]]]] = None
    rms: Optional[Sequence[float]] = None
    error: Optional[str] = None
    iterations: Optional[int] = None
    evaluations: Optional[int] = None
    coarse_grid_points: Optional[int] = None
    coarse_best_p: Optional[float] = None
    coarse_best_loss: Optional[float] = None
    optimisation_coarse_best_loss: Optional[float] = None
    refinement_bracket: Optional[tuple[float, float]] = None

    def to_json(self) -> Dict[str, Any]:
        def finite_or_none(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None

        payload: Dict[str, Any] = {
            "parameter": self.parameter,
            "status": self.status,
            "loss": finite_or_none(self.loss),
            "p_opt": finite_or_none(self.p_opt),
            "param_opt": finite_or_none(self.p_opt),
            "RMS": [
                finite
                for value in (self.rms or [])
                if (finite := finite_or_none(value)) is not None
            ],
        }
        if self.optimisation_loss is not None:
            payload["optimisation_loss"] = finite_or_none(self.optimisation_loss)
        if self.continuous_loss is not None:
            payload["continuous_loss"] = finite_or_none(self.continuous_loss)
            payload["continuos_loss"] = finite_or_none(self.continuous_loss)
        if self.impulse_loss is not None:
            payload["impulse_loss"] = finite_or_none(self.impulse_loss)
        if self.final_profile_losses is not None:
            payload["results_final"] = _normalise_results_final(self.final_profile_losses)
        if self.error:
            payload["error"] = self.error
        if self.iterations is not None:
            payload["iterations"] = self.iterations
        if self.evaluations is not None:
            payload["evaluations"] = self.evaluations
        if self.coarse_grid_points is not None:
            payload["coarse_grid_points"] = int(self.coarse_grid_points)
        if self.coarse_best_p is not None:
            payload["coarse_best_p"] = finite_or_none(self.coarse_best_p)
        if self.coarse_best_loss is not None:
            payload["coarse_best_loss"] = finite_or_none(self.coarse_best_loss)
        if self.optimisation_coarse_best_loss is not None:
            payload["optimisation_coarse_best_loss"] = finite_or_none(
                self.optimisation_coarse_best_loss
            )
        if self.refinement_bracket is not None:
            left, right = self.refinement_bracket
            payload["refinement_bracket"] = [finite_or_none(left), finite_or_none(right)]
        return payload


OptimizerFn = Callable[..., Sequence[CandidateResult]]


def _resolve_models_root() -> Path:
    cwd_models_root = Path(os.getcwd()).resolve() / "models" / "simulink"
    if cwd_models_root.exists():
        return cwd_models_root
    return root_dir / "models" / "simulink"


def _now_iso8601_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _webapp_run_url(*, model_id: str, run_id: str) -> str:
    query = urlencode(
        {
            "model": str(model_id or "").strip(),
            "run": str(run_id or "").strip(),
            "compare": "none",
        }
    )
    return f"{_WEBAPP_BASE_URL}?{query}"


def _coerce_finite_float(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return float(parsed)


def _parameter_value_srd(left: Any, right: Any) -> Optional[float]:
    try:
        left_num = float(left)
        right_num = float(right)
    except Exception:
        return None
    if not math.isfinite(left_num) or not math.isfinite(right_num):
        return None
    denominator = abs(left_num) + abs(right_num)
    if denominator <= 0.0:
        return 0.0
    return float((2.0 * abs(left_num - right_num)) / denominator)


def _normalise_run_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _find_child_context(
    *,
    run_id: str,
    specs: Mapping[str, Any],
    runtime_record: Mapping[str, Any],
) -> ChildRunContext:
    normalized_run_id = _normalise_run_id(run_id)
    if not normalized_run_id:
        raise ValueError("--run-id must be non-empty")

    runtime_entry = runtime_record.get(normalized_run_id)
    if not isinstance(runtime_entry, dict):
        raise ValueError(f"run_id {normalized_run_id!r} is not present in model_record.json")
    run_type = str(runtime_entry.get("run_type") or "").strip().lower()
    if run_type != "intervention":
        raise ValueError(
            f"RUN_ID must be a child intervention run; {normalized_run_id!r} has run_type={run_type!r}"
        )
    if not is_success_status(runtime_entry.get("status")):
        raise ValueError(
            f"RUN_ID {normalized_run_id!r} is not successful; status={runtime_entry.get('status')!r}"
        )

    for baseline_run_id, raw_parent in specs.items():
        if not isinstance(raw_parent, dict):
            continue
        children = raw_parent.get("children")
        if not isinstance(children, dict):
            continue
        raw_child = children.get(normalized_run_id)
        if not isinstance(raw_child, dict):
            continue
        child_parameters = dict(raw_child.get("parameters") or {})
        non_null_parameters = [
            (str(key).strip(), value)
            for key, value in child_parameters.items()
            if str(key).strip() and value is not None
        ]
        if len(non_null_parameters) != 1:
            raise ValueError(
                f"Child run {normalized_run_id!r} must define exactly one changed parameter"
            )
        parameter, set_value = non_null_parameters[0]
        intervention_time = _coerce_finite_float(
            raw_child.get("intervention_time"),
            label=f"{normalized_run_id}.intervention_time",
        )
        time0_id = str(raw_child.get("time0_baseline_uuid") or "").strip().lower() or None
        return ChildRunContext(
            run_id=normalized_run_id,
            baseline_run_id=str(baseline_run_id).strip().lower(),
            time0_baseline_run_id=time0_id,
            ground_truth_parameter=parameter,
            intervention_time=intervention_time,
            baseline_parameters=dict(raw_parent.get("baseline_parameters") or {}),
            child_parameters={parameter: set_value},
            child_set_value=set_value,
            child_spec=dict(raw_child),
            runtime_entry=dict(runtime_entry),
        )

    raise ValueError(f"run_id {normalized_run_id!r} is not present as a child in model_run_specs.json")


def _candidate_specs_from_config_and_metadata(
    *,
    experiment_config: Any,
    metadata: Mapping[str, Any],
) -> list[CandidateSpec]:
    parameter_configs = getattr(experiment_config.exposed_variables, "parameters", {}) or {}
    intervention_map = metadata.get("intervention_block_map")
    if not isinstance(intervention_map, Mapping):
        raise ValueError("generated/metadata.json is missing intervention_block_map")

    candidates: list[CandidateSpec] = []
    for parameter in sorted(parameter_configs.keys(), key=str.casefold):
        spec = parameter_configs[parameter]
        interval = getattr(spec, "allowed_intervals")
        minimum = _coerce_finite_float(interval[0], label=f"{parameter}.allowed_intervals[0]")
        maximum = _coerce_finite_float(interval[1], label=f"{parameter}.allowed_intervals[1]")
        if minimum > maximum:
            raise ValueError(f"{parameter}.allowed_intervals must satisfy low <= high")
        raw_entry = intervention_map.get(parameter)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"generated/metadata.json has no block mapping for candidate {parameter!r}")
        raw_bindings = raw_entry.get("parameters")
        if not isinstance(raw_bindings, Sequence) or not raw_bindings:
            raise ValueError(f"candidate {parameter!r} has no block-parameter bindings")
        bindings: list[CandidateBinding] = []
        for raw_binding in raw_bindings:
            if not isinstance(raw_binding, Mapping):
                raise ValueError(f"candidate {parameter!r} contains an invalid binding")
            path = str(raw_binding.get("path") or "").strip()
            name = str(raw_binding.get("name") or "").strip()
            expression = str(raw_binding.get("expression") or "").strip()
            if not path or not name or not expression:
                raise ValueError(
                    f"candidate {parameter!r} binding must define path, name, and expression"
                )
            bindings.append(CandidateBinding(path=path, name=name, expression=expression))
        candidates.append(
            CandidateSpec(
                parameter=str(parameter),
                minimum=minimum,
                maximum=maximum,
                sampling_strategy=str(getattr(spec, "sampling_strategy", "uniform")),
                bindings=tuple(bindings),
            )
        )
    return candidates


def _simulink_model_path_for_snapshot(path: Any) -> str:
    return str(path or "").strip().replace("simulink_model_original", "simulink_model")


def _baseline_variables_with_metadata_defaults(
    *,
    metadata: Mapping[str, Any],
    baseline_parameters: Mapping[str, Any],
) -> dict[str, float]:
    variables: dict[str, float] = {}
    defaults = metadata.get("default_values") or {}
    if isinstance(defaults, Mapping):
        for name, value in defaults.items():
            variable_name = str(name).strip()
            if not variable_name or variable_name == "end_time_input_s":
                continue
            variables[variable_name] = _coerce_finite_float(
                value,
                label=f"metadata default {variable_name}",
            )
    for name, value in baseline_parameters.items():
        variable_name = str(name).strip()
        if not variable_name:
            continue
        variables[variable_name] = _coerce_finite_float(
            value,
            label=f"baseline parameter {variable_name}",
        )
    return variables


def _baseline_block_bindings_from_metadata(
    *,
    metadata: Mapping[str, Any],
    baseline_parameters: Mapping[str, Any],
) -> list[BaselineBlockBinding]:
    intervention_map = metadata.get("intervention_block_map")
    if not isinstance(intervention_map, Mapping):
        raise ValueError("generated/metadata.json is missing intervention_block_map")

    variables = _baseline_variables_with_metadata_defaults(
        metadata=metadata,
        baseline_parameters=baseline_parameters,
    )
    bindings_by_address: dict[tuple[str, str], BaselineBlockBinding] = {}
    for intervention_parameter in sorted(intervention_map.keys(), key=str.casefold):
        raw_entry = intervention_map[intervention_parameter]
        if not isinstance(raw_entry, Mapping):
            continue
        raw_bindings = raw_entry.get("parameters")
        if not isinstance(raw_bindings, Sequence):
            continue
        for raw_binding in raw_bindings:
            if not isinstance(raw_binding, Mapping):
                continue
            path = _simulink_model_path_for_snapshot(raw_binding.get("path"))
            name = str(raw_binding.get("name") or "").strip()
            expression = str(raw_binding.get("expression") or "").strip() or name
            if not path or not name:
                continue
            value = float(eval_numeric_expr(expression, variables))
            if not math.isfinite(value):
                raise ValueError(
                    "Baseline block binding expression did not evaluate to a finite "
                    f"scalar for {path!r}.{name!r}"
                )
            bindings_by_address[(path, name)] = BaselineBlockBinding(
                path=path,
                name=name,
                value=value,
            )
    return [
        bindings_by_address[address]
        for address in sorted(
            bindings_by_address,
            key=lambda item: (item[0].casefold(), item[1].casefold()),
        )
    ]


def _observable_signal_names(experiment_config: Any) -> list[str]:
    names = list(getattr(experiment_config, "observable_signal_names", []) or [])
    if not names:
        raise ValueError("experiment_config.json must define observable_signals")
    return [str(name) for name in names]


def _observable_signals_for_parameter(experiment_config: Any, parameter: str) -> list[str]:
    parameter_configs = getattr(experiment_config.exposed_variables, "parameters", {}) or {}
    if parameter_configs.get(parameter) is None:
        raise ValueError(
            f"ground-truth parameter {parameter!r} is not present in experiment_config.json"
        )
    return _observable_signal_names(experiment_config)


def _normalise_requested_signals(signals: Sequence[str]) -> list[str]:
    requested: list[str] = []
    seen: set[str] = set()
    for raw_signal in signals:
        signal = str(raw_signal or "").strip()
        if not signal:
            raise ValueError("--signals entries must be non-empty")
        if signal in seen:
            raise ValueError(f"--signals contains duplicate signal {signal!r}")
        seen.add(signal)
        requested.append(signal)
    if not requested:
        raise ValueError("--signals must contain at least one signal")
    return requested


def _observable_signal_selection(
    experiment_config: Any,
    *,
    selected_signals: Optional[Sequence[str]] = None,
    ignore_impulse_signals: bool,
) -> tuple[list[str], list[str]]:
    names = _observable_signal_names(experiment_config)
    selected = (
        _normalise_requested_signals(selected_signals)
        if selected_signals is not None
        else names
    )
    known = set(names)
    unknown = [signal for signal in selected if signal not in known]
    if unknown:
        raise ValueError(f"Unknown observable signals requested: {unknown}")
    if not ignore_impulse_signals:
        return selected, []

    signal_types = getattr(experiment_config, "observable_signal_types", {}) or {}
    used: list[str] = []
    ignored: list[str] = []
    for name in selected:
        signal_type = str(signal_types.get(name, "")).strip().lower()
        if signal_type == "impulse_like":
            ignored.append(name)
        else:
            used.append(name)
    if not used:
        raise ValueError("No observable signals remain after --ignore-impulse-signals")
    return used, ignored


def _require_run_df(runs_root: Path, run_id: str) -> pd.DataFrame:
    df = load_run_df(runs_root / run_id)
    if df is None:
        raise ValueError(f"Missing or invalid data.parquet/data.csv for run {run_id!r}")
    return df


def _resolve_noise_seed(context: ChildRunContext) -> int:
    for source in (context.runtime_entry, context.child_spec):
        for key in ("noise_seed", "seed"):
            value = source.get(key)
            if value in (None, ""):
                continue
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer when present") from exc
    return _DEFAULT_NOISE_SEED


def _apply_optional_noise(
    df: pd.DataFrame,
    *,
    baseline_df: pd.DataFrame,
    model_id: str,
    models_root: Path,
    noise_profile: str,
    noise_seed: int,
) -> tuple[pd.DataFrame, Optional[dict[str, Any]]]:
    resolved_profile = normalize_noise_profile(noise_profile)
    if resolved_profile == "none":
        return df.copy(), None
    add_noise = load_model_noise_adder(model_id=model_id, models_root=Path(models_root))
    noisy_df, _noise_analysis = call_noise_adder(
        add_noise,
        df,
        baseline_df=baseline_df,
        seed=int(noise_seed),
        noise_level=resolved_profile,
    )
    return noisy_df, _noise_analysis


def _target_dataframes_for_noise_profiles(
    df: pd.DataFrame,
    *,
    baseline_df: pd.DataFrame,
    model_id: str,
    models_root: Path,
    noise_seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Optional[dict[str, Any]]]]:
    targets: dict[str, pd.DataFrame] = {"none": df.copy()}
    analyses: dict[str, Optional[dict[str, Any]]] = {"none": None}
    add_noise = load_model_noise_adder(model_id=model_id, models_root=Path(models_root))
    for profile in ("low", "high"):
        noisy_df, noise_analysis = call_noise_adder(
            add_noise,
            df,
            baseline_df=baseline_df,
            seed=int(noise_seed),
            noise_level=profile,
        )
        targets[profile] = noisy_df
        analyses[profile] = noise_analysis
    return targets, analyses


def _observable_signal_types_for_signals(
    experiment_config: Any,
    observable_signals: Sequence[str],
) -> dict[str, str]:
    configured = getattr(experiment_config, "observable_signal_types", {}) or {}
    signal_types: dict[str, str] = {}
    for raw_signal in observable_signals:
        signal = str(raw_signal)
        signal_type = str(configured.get(signal, "continuous")).strip().lower().replace("-", "_")
        if signal_type not in {"continuous", "impulse_like"}:
            raise ValueError(f"Unsupported observable signal type for {signal!r}: {signal_type!r}")
        signal_types[signal] = signal_type
    return signal_types


def _continuous_signals_from_types(
    observable_signals: Sequence[str],
    signal_types: Mapping[str, str],
) -> list[str]:
    return [
        str(signal)
        for signal in observable_signals
        if str(signal_types.get(str(signal), "continuous")).strip().lower() == "continuous"
    ]


def _uses_ball_drop_impulse_rule(*, model_id: Optional[str], signal: str) -> bool:
    return (
        str(model_id or "").strip().lower() == _BALL_DROP_MODEL_ID
        and str(signal) == _BALL_DROP_IMPULSE_SIGNAL
    )


def _ball_drop_velocity_crossover_present(
    *,
    peak_time: float,
    velocity_times: Optional[np.ndarray],
    velocity_values: Optional[np.ndarray],
) -> bool:
    if velocity_times is None or velocity_values is None:
        return False
    times = np.asarray(velocity_times, dtype=float)
    values = np.asarray(velocity_values, dtype=float)
    finite = np.isfinite(times) & np.isfinite(values)
    if not np.any(finite):
        return False
    times = times[finite]
    values = values[finite]
    window = _BALL_DROP_VELOCITY_CROSSOVER_WINDOW_S
    pre_mask = (times >= float(peak_time) - window) & (times <= float(peak_time))
    post_mask = (times >= float(peak_time)) & (times <= float(peak_time) + window)
    if not np.any(pre_mask) or not np.any(post_mask):
        return False
    return bool(
        float(np.median(values[pre_mask])) < 0.0
        and float(np.median(values[post_mask])) > 0.0
    )


def _collapse_impulse_events_by_time(
    events: Sequence[tuple[float, float]],
    *,
    min_gap_time: float,
) -> list[tuple[float, float]]:
    if not events:
        return []
    sorted_events = sorted(events, key=lambda item: item[0])
    collapsed: list[tuple[float, float]] = []
    current: list[tuple[float, float]] = [sorted_events[0]]
    for event in sorted_events[1:]:
        if float(event[0]) - float(current[-1][0]) <= float(min_gap_time):
            current.append(event)
            continue
        collapsed.append(max(current, key=lambda item: abs(item[1])))
        current = [event]
    collapsed.append(max(current, key=lambda item: abs(item[1])))
    return collapsed


def _pre_intervention_abs_max(
    df: pd.DataFrame,
    *,
    signal: str,
    intervention_time: float,
) -> Optional[float]:
    if signal not in df.columns or "time" not in df.columns:
        return None
    times = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df[signal], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(values) & (times < float(intervention_time))
    if not np.any(finite):
        return None
    scale = float(np.max(np.abs(values[finite])))
    return scale if math.isfinite(scale) and scale > _EPS else None


def _signal_loss_scales(
    *,
    reference_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    observable_signals: Sequence[str],
    intervention_time: float,
) -> dict[str, float]:
    scales: dict[str, float] = {}
    for raw_signal in observable_signals:
        signal = str(raw_signal)
        scale = _pre_intervention_abs_max(
            reference_df,
            signal=signal,
            intervention_time=intervention_time,
        )
        if scale is None:
            scale = _pre_intervention_abs_max(
                baseline_df,
                signal=signal,
                intervention_time=intervention_time,
            )
        scales[signal] = float(scale) if scale is not None else 1.0
    return scales


def _post_intervention_rows(df: pd.DataFrame, *, intervention_time: float) -> pd.DataFrame:
    times = pd.to_numeric(df["time"], errors="coerce")
    out = df.copy()
    out["time"] = times
    out = out.dropna(subset=["time"]).sort_values("time")
    mask = out["time"].to_numpy(dtype=float) >= float(intervention_time)
    if not np.any(mask):
        mask = np.ones(len(out), dtype=bool)
    return out.loc[mask]


def _detect_impulse_events(
    times: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float,
    velocity_times: Optional[np.ndarray] = None,
    velocity_values: Optional[np.ndarray] = None,
) -> list[tuple[float, float]]:
    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]
    if times.size == 0:
        return []
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    event_indices = np.flatnonzero(np.abs(values) > float(threshold))
    if event_indices.size == 0:
        return []
    candidate_events: list[tuple[float, float]] = []
    for group in np.split(event_indices, np.flatnonzero(np.diff(event_indices) > 1) + 1):
        if group.size == 0:
            continue
        local = int(group[np.argmax(np.abs(values[group]))])
        candidate_events.append((float(times[local]), float(values[local])))
    if velocity_times is None and velocity_values is None:
        return candidate_events
    events = [
        event
        for event in candidate_events
        if _ball_drop_velocity_crossover_present(
            peak_time=event[0],
            velocity_times=velocity_times,
            velocity_values=velocity_values,
        )
    ]
    return _collapse_impulse_events_by_time(
        events,
        min_gap_time=2.0 * _BALL_DROP_VELOCITY_CROSSOVER_WINDOW_S,
    )


def _impulse_event_loss(
    *,
    target_times: np.ndarray,
    target_values: np.ndarray,
    candidate_times: np.ndarray,
    candidate_values: np.ndarray,
    scale: float,
    target_velocity_times: Optional[np.ndarray] = None,
    target_velocity_values: Optional[np.ndarray] = None,
    candidate_velocity_times: Optional[np.ndarray] = None,
    candidate_velocity_values: Optional[np.ndarray] = None,
) -> float:
    threshold = _IMPULSE_EVENT_THRESHOLD_FRACTION * max(float(scale), _EPS)
    target_events = _detect_impulse_events(
        target_times,
        target_values,
        threshold=threshold,
        velocity_times=target_velocity_times,
        velocity_values=target_velocity_values,
    )
    candidate_events = _detect_impulse_events(
        candidate_times,
        candidate_values,
        threshold=threshold,
        velocity_times=candidate_velocity_times,
        velocity_values=candidate_velocity_values,
    )
    if not target_events and not candidate_events:
        return 0.0
    safe_scale = max(float(scale), _EPS)
    used_candidates: set[int] = set()
    loss = 0.0
    for target_time, target_value in target_events:
        eligible = [
            idx
            for idx, (candidate_time, _candidate_value) in enumerate(candidate_events)
            if idx not in used_candidates
            and abs(candidate_time - target_time) <= _IMPULSE_EVENT_MATCH_WINDOW_S
        ]
        if not eligible:
            loss += _IMPULSE_EVENT_UNMATCHED_PENALTY
            continue
        best_idx = max(
            eligible,
            key=lambda idx: (
                abs(candidate_events[idx][1]),
                -abs(candidate_events[idx][0] - target_time),
            ),
        )
        used_candidates.add(best_idx)
        candidate_time, candidate_value = candidate_events[best_idx]
        loss += abs(candidate_time - target_time) / _IMPULSE_EVENT_MATCH_WINDOW_S
        loss += abs(abs(candidate_value) - abs(target_value)) / safe_scale
    loss += _IMPULSE_EVENT_UNMATCHED_PENALTY * (
        len(candidate_events) - len(used_candidates)
    )
    return float(loss)


def _continuous_signal_loss(
    *,
    target_times: np.ndarray,
    target_values: np.ndarray,
    candidate_times: np.ndarray,
    candidate_values: np.ndarray,
    scale: float,
) -> float:
    finite = np.isfinite(candidate_times) & np.isfinite(candidate_values)
    if not np.any(finite):
        raise ValueError("No finite candidate values")
    interpolated = np.interp(
        target_times,
        candidate_times[finite],
        candidate_values[finite],
        left=candidate_values[finite][0],
        right=candidate_values[finite][-1],
    )
    residual = (interpolated - target_values) / max(float(scale), _EPS)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        raise ValueError("No finite trajectory residuals available")
    return float(np.mean(np.square(residual, dtype=np.float64), dtype=np.float64))


def _trajectory_signal_losses(
    *,
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    observable_signals: Sequence[str],
    intervention_time: float,
    signal_types: Optional[Mapping[str, str]] = None,
    signal_scales: Optional[Mapping[str, float]] = None,
    model_id: Optional[str] = None,
) -> dict[str, float]:
    auxiliary_signals = [
        _BALL_DROP_VELOCITY_SIGNAL
        for signal in observable_signals
        if _uses_ball_drop_impulse_rule(model_id=model_id, signal=str(signal))
        and _BALL_DROP_VELOCITY_SIGNAL not in observable_signals
    ]
    candidate_columns = ["time", *observable_signals, *auxiliary_signals]
    missing = [
        signal
        for signal in [*observable_signals, *auxiliary_signals]
        if signal not in reference_df.columns or signal not in candidate_df.columns
    ]
    if missing:
        raise ValueError(f"Missing observable signals in run data: {missing}")
    ref = _post_intervention_rows(
        reference_df[["time", *observable_signals]].copy(),
        intervention_time=intervention_time,
    )
    cand = candidate_df[candidate_columns].copy()
    cand["time"] = pd.to_numeric(cand["time"], errors="coerce")
    cand = cand.dropna(subset=["time"]).sort_values("time")
    candidate_post = _post_intervention_rows(cand, intervention_time=intervention_time)
    target_times = ref["time"].to_numpy(dtype=float)
    losses: dict[str, float] = {}
    for signal in observable_signals:
        signal_type = str((signal_types or {}).get(signal, "continuous"))
        scale = float((signal_scales or {}).get(signal, 1.0))
        target_values = pd.to_numeric(ref[signal], errors="coerce").to_numpy(dtype=float)
        finite_target = np.isfinite(target_times) & np.isfinite(target_values)
        if not np.any(finite_target):
            continue
        source_times = cand["time"].to_numpy(dtype=float)
        source_values = pd.to_numeric(cand[signal], errors="coerce").to_numpy(dtype=float)
        if not np.any(np.isfinite(source_times) & np.isfinite(source_values)):
            raise ValueError(f"No finite values for observable signal {signal!r}")
        if signal_type == "impulse_like":
            target_velocity_times = None
            target_velocity_values = None
            candidate_velocity_times = None
            candidate_velocity_values = None
            if _uses_ball_drop_impulse_rule(model_id=model_id, signal=signal):
                if (
                    _BALL_DROP_VELOCITY_SIGNAL not in reference_df.columns
                    or _BALL_DROP_VELOCITY_SIGNAL not in candidate_df.columns
                ):
                    raise ValueError(
                        "BallDrop Hard_Stop_f impulse loss requires Velocity in target "
                        "and candidate trajectories"
                    )
                target_velocity_times = pd.to_numeric(
                    reference_df["time"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                target_velocity_values = pd.to_numeric(
                    reference_df[_BALL_DROP_VELOCITY_SIGNAL],
                    errors="coerce",
                ).to_numpy(dtype=float)
                candidate_velocity_times = source_times
                candidate_velocity_values = pd.to_numeric(
                    cand[_BALL_DROP_VELOCITY_SIGNAL],
                    errors="coerce",
                ).to_numpy(dtype=float)
            losses[str(signal)] = _impulse_event_loss(
                target_times=target_times[finite_target],
                target_values=target_values[finite_target],
                candidate_times=candidate_post["time"].to_numpy(dtype=float),
                candidate_values=pd.to_numeric(
                    candidate_post[signal],
                    errors="coerce",
                ).to_numpy(dtype=float),
                scale=scale,
                target_velocity_times=target_velocity_times,
                target_velocity_values=target_velocity_values,
                candidate_velocity_times=candidate_velocity_times,
                candidate_velocity_values=candidate_velocity_values,
            )
        else:
            losses[str(signal)] = _continuous_signal_loss(
                target_times=target_times[finite_target],
                target_values=target_values[finite_target],
                candidate_times=source_times,
                candidate_values=source_values,
                scale=scale,
            )
    if not losses:
        raise ValueError("No finite trajectory losses available")
    return losses


def _trajectory_mse(
    *,
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    observable_signals: Sequence[str],
    intervention_time: float,
    signal_types: Optional[Mapping[str, str]] = None,
    signal_scales: Optional[Mapping[str, float]] = None,
    model_id: Optional[str] = None,
    baseline_signal_losses: Optional[Mapping[str, float]] = None,
) -> float:
    losses = _trajectory_signal_losses(
        reference_df=reference_df,
        candidate_df=candidate_df,
        observable_signals=observable_signals,
        intervention_time=intervention_time,
        signal_types=signal_types,
        signal_scales=signal_scales,
        model_id=model_id,
    )
    ordered_signals = [str(signal) for signal in observable_signals if str(signal) in losses]
    ordered_losses = [losses[signal] for signal in ordered_signals]
    if baseline_signal_losses is None:
        return float(np.sum(ordered_losses, dtype=np.float64))
    normalised_losses = [
        float(loss)
        / max(
            float(baseline_signal_losses.get(signal, 0.0)),
            _LOSS_NORMALISATION_FLOOR,
        )
        for signal, loss in zip(ordered_signals, ordered_losses)
    ]
    if not normalised_losses:
        raise ValueError("No finite trajectory losses available")
    return float(np.mean(normalised_losses, dtype=np.float64))


def _final_loss_entry(
    *,
    loss: Optional[float],
    continuous_loss: Optional[float],
    impulse_loss: Optional[float],
) -> dict[str, Optional[float]]:
    def finite_or_none(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    return {
        "loss": finite_or_none(loss),
        "loss_continuos": finite_or_none(continuous_loss),
        "loss_continuous": finite_or_none(continuous_loss),
        "loss_impulse": finite_or_none(impulse_loss),
    }


def _empty_results_final() -> dict[str, dict[str, Optional[float]]]:
    return {
        profile: _final_loss_entry(
            loss=None,
            continuous_loss=None,
            impulse_loss=None,
        )
        for profile in ("none", "low", "high")
    }


def _normalise_results_final(
    value: Optional[Mapping[str, Mapping[str, Optional[float]]]],
) -> dict[str, dict[str, Optional[float]]]:
    results = _empty_results_final()
    if not isinstance(value, Mapping):
        return results
    for raw_profile, raw_entry in value.items():
        profile = normalize_noise_profile(raw_profile)
        if not isinstance(raw_entry, Mapping):
            continue
        results[profile] = _final_loss_entry(
            loss=raw_entry.get("loss"),
            continuous_loss=raw_entry.get(
                "loss_continuos",
                raw_entry.get("loss_continuous"),
            ),
            impulse_loss=raw_entry.get("loss_impulse"),
        )
    return results


def _finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _documented_candidate_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    rms = candidate.get("RMS")
    rms_values = [
        finite
        for value in (rms if isinstance(rms, Sequence) and not isinstance(rms, str) else [])
        if (finite := _finite_or_none(value)) is not None
    ]
    return {
        "RMS": rms_values,
        "param_opt": _finite_or_none(candidate.get("param_opt", candidate.get("p_opt"))),
        "optimisation_loss": _finite_or_none(candidate.get("optimisation_loss")),
        "coarse_grid_points": candidate.get("coarse_grid_points"),
        "evaluations": candidate.get("evaluations"),
        "iterations": candidate.get("iterations"),
        "parameter": str(candidate.get("parameter") or "").strip() or None,
        "refinement_bracket": candidate.get("refinement_bracket"),
        "status": str(candidate.get("status") or "failed"),
    }


def _documented_intervention_payload(*, parameter: Any, value: Any) -> dict[str, Any]:
    return {
        "parameter": str(parameter or "").strip() or None,
        "value": value,
    }


def _documented_oracle_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "url": report.get("url"),
        "uuid": report.get("uuid"),
        "ground_truth_intervention": _documented_intervention_payload(
            parameter=report.get("parameter_ground_truth", report.get("ground_truth_parameter")),
            value=report.get("parameter_ground_truth_value", report.get("ground_truth_value")),
        ),
        "best_found_intervention": _documented_intervention_payload(
            parameter=report.get("parameter_found", report.get("best_parameter")),
            value=report.get("parameter_found_value"),
        ),
        "hash": report.get("hash"),
        "candidates": [
            _documented_candidate_payload(candidate)
            for candidate in report.get("candidates", [])
            if isinstance(candidate, Mapping)
        ],
        "results_final": _normalise_results_final(report.get("results_final")),
    }


def _build_verdict(
    *,
    model_id: str,
    context: ChildRunContext,
    baseline_mse: float,
    candidate_results: Sequence[CandidateResult],
    min_baseline_improvement: float,
    min_next_best_separation: float,
    optimizer: str,
    ignore_impulse_signals: bool = False,
    observable_signals_used: Sequence[str] = (),
    observable_signals_ignored: Sequence[str] = (),
    optimisation_signals_used: Sequence[str] = (),
    optimisation_baseline_mse: Optional[float] = None,
    baseline_signal_losses: Optional[Mapping[str, float]] = None,
    noise_profile: str = "none",
    noise_seed: int = _DEFAULT_NOISE_SEED,
    noise_analysis: Optional[Mapping[str, Any]] = None,
    parameter_value_srd_match_threshold: float = _DEFAULT_PARAMETER_VALUE_SRD_MATCH_THRESHOLD,
) -> Dict[str, Any]:
    serialised_candidates = [result.to_json() for result in candidate_results]
    successful = [
        result
        for result in candidate_results
        if result.status == "success"
        and result.loss is not None
        and math.isfinite(float(result.loss))
    ]
    successful_sorted = sorted(successful, key=lambda item: (float(item.loss), item.parameter))

    best_parameter: Optional[str] = None
    best_parameter_value: Optional[float] = None
    best_loss: Optional[float] = None
    second_best_parameter: Optional[str] = None
    second_best_loss: Optional[float] = None
    rank: Optional[int] = None
    baseline_improvement: Optional[float] = None
    next_best_separation: Optional[float] = None
    parameter_value_srd: Optional[float] = None
    best_result: Optional[CandidateResult] = None
    detectable = False

    if successful_sorted:
        best = successful_sorted[0]
        best_result = best
        best_parameter = best.parameter
        if best.p_opt is not None and math.isfinite(float(best.p_opt)):
            best_parameter_value = float(best.p_opt)
        best_loss = float(best.loss) if best.loss is not None else None
        for idx, result in enumerate(successful_sorted, start=1):
            if result.parameter == context.ground_truth_parameter:
                rank = idx
                break
        if best_loss is not None:
            if baseline_mse > _EPS:
                baseline_improvement = (float(baseline_mse) - best_loss) / float(baseline_mse)
            else:
                baseline_improvement = 0.0
            if len(successful_sorted) >= 2:
                second_best_parameter = successful_sorted[1].parameter
                next_loss = float(successful_sorted[1].loss or 0.0)
                second_best_loss = next_loss
                next_best_separation = (next_loss - best_loss) / max(abs(next_loss), _EPS)
            else:
                next_best_separation = None
            parameter_value_srd = _parameter_value_srd(
                best_parameter_value,
                context.child_set_value,
            )
            detectable = bool(
                best_parameter == context.ground_truth_parameter
                and best_loss is not None
                and math.isfinite(float(best_loss))
                and baseline_improvement is not None
                and baseline_improvement >= float(min_baseline_improvement)
                and next_best_separation is not None
                and next_best_separation >= float(min_next_best_separation)
            )

    child_hash = str(context.child_spec.get("parameter_hash") or "").strip() or None
    results_final = _normalise_results_final(
        best_result.final_profile_losses if best_result is not None else None
    )
    active_profile = normalize_noise_profile(noise_profile)
    if (
        active_profile in results_final
        and best_result is not None
        and results_final[active_profile]["loss"] is None
    ):
        results_final[active_profile] = _final_loss_entry(
            loss=best_result.loss,
            continuous_loss=best_result.continuous_loss,
            impulse_loss=best_result.impulse_loss,
        )
    return {
        "schema_version": 1,
        "generated_at": _now_iso8601_utc(),
        "url": _webapp_run_url(model_id=model_id, run_id=context.run_id),
        "uuid": context.run_id,
        "hash": child_hash,
        "model": model_id,
        "run_id": context.run_id,
        "baseline_run_id": context.baseline_run_id,
        "time0_baseline_run_id": context.time0_baseline_run_id,
        "parameter_ground_truth": context.ground_truth_parameter,
        "parameter_ground_truth_value": context.child_set_value,
        "ground_truth_parameter": context.ground_truth_parameter,
        "ground_truth_value": context.child_set_value,
        "intervention_time": context.intervention_time,
        "optimizer": optimizer,
        "noise_profile": normalize_noise_profile(noise_profile),
        "noise_seed": int(noise_seed),
        "SNR": dict(noise_analysis) if isinstance(noise_analysis, Mapping) else None,
        "ignore_impulse_signals": bool(ignore_impulse_signals),
        "observable_signals_used": [str(signal) for signal in observable_signals_used],
        "observable_signals_ignored": [str(signal) for signal in observable_signals_ignored],
        "final_loss_signals_used": [str(signal) for signal in observable_signals_used],
        "optimisation_signals_used": [str(signal) for signal in optimisation_signals_used],
        "thresholds": {
            "min_baseline_improvement": float(min_baseline_improvement),
            "min_next_best_separation": float(min_next_best_separation),
            "parameter_value_srd_match_threshold": float(
                parameter_value_srd_match_threshold
            ),
        },
        "loss_normalisation": "per_signal_baseline",
        "loss_normalisation_floor": _LOSS_NORMALISATION_FLOOR,
        "baseline_signal_losses": {
            str(signal): float(loss)
            for signal, loss in (baseline_signal_losses or {}).items()
            if math.isfinite(float(loss))
        },
        "loss_baseline": float(baseline_mse),
        "baseline_mse": float(baseline_mse),
        "optimisation_loss_baseline": (
            float(optimisation_baseline_mse)
            if optimisation_baseline_mse is not None
            else None
        ),
        "parameter_found": best_parameter,
        "parameter_found_value": best_parameter_value,
        "best_parameter": best_parameter,
        "loss": best_loss,
        "best_loss": best_loss,
        "base_second_parameter": second_best_parameter,
        "best_second_loss": second_best_loss,
        "rank": rank,
        "baseline_improvement_fraction": baseline_improvement,
        "next_best_separation_fraction": next_best_separation,
        "parameter_value_srd": parameter_value_srd,
        "is_match": detectable,
        "optimisation_detectable": detectable,
        "results_final": results_final,
        "candidates": serialised_candidates,
    }


def _matlab_char_literal(value: Any) -> str:
    # char([...]) preserves block paths that contain literal newlines.
    codepoints = " ".join(str(ord(ch)) for ch in str(value))
    return f"char([{codepoints}])"


def _matlab_cell_array(values: Sequence[Any]) -> str:
    return "{" + ",".join(_matlab_char_literal(value) for value in values) + "}"


def _matlab_numeric_vector(values: Sequence[float]) -> str:
    return "[" + " ".join(f"{float(value):.17g}" for value in values) + "]"


def _coarse_search_enabled(coarse_grid_points: Optional[int]) -> bool:
    return coarse_grid_points is not None and int(coarse_grid_points) > 1


def _optimizer_name(coarse_grid_points: Optional[int]) -> str:
    if _coarse_search_enabled(coarse_grid_points):
        return "matlab_parsim_grid_fminbnd"
    return "matlab_fminbnd"


def _resolve_matlab_workers(matlab_workers: Optional[int]) -> int:
    if matlab_workers is not None:
        workers = int(matlab_workers)
    else:
        raw_env = str(os.environ.get(_MATLAB_WORKERS_ENV, "")).strip()
        workers = int(raw_env) if raw_env else _DEFAULT_MATLAB_WORKERS
    if workers < 0:
        raise ValueError("--matlab-workers must be >= 0")
    return workers


def _ensure_matlab_parallel_pool(mle: Any, matlab_workers: int) -> int:
    workers = int(matlab_workers)
    if workers <= 0:
        click.echo("MATLAB parallel pool startup disabled.", err=True)
        return 0
    try:
        mle.eval(
            "tsenv_pool = gcp(\"nocreate\"); "
            f"if ~isempty(tsenv_pool) && tsenv_pool.NumWorkers ~= {workers}, "
            "delete(tsenv_pool); tsenv_pool = []; end; "
            "if isempty(tsenv_pool), "
            f"try, tsenv_pool = parpool(\"Processes\", {workers}); "
            f"catch, tsenv_pool = parpool({workers}); end; "
            "end; "
            "tsenv_parallel_pool_workers = tsenv_pool.NumWorkers;",
            nargout=0,
        )
        active_workers = int(mle.workspace["tsenv_parallel_pool_workers"])
        click.echo(
            f"Using MATLAB parallel pool with {active_workers} workers.",
            err=True,
        )
        return active_workers
    except Exception as exc:  # noqa: BLE001 - optimisation can still run serially
        click.echo(
            f"MATLAB parallel pool startup failed; continuing without explicit pool: {exc}",
            err=True,
        )
        return 0


def _select_candidate_optimum(
    *,
    fminbnd_p: float,
    fminbnd_loss: float,
    coarse_best_p: Optional[float],
    coarse_best_loss: Optional[float],
) -> tuple[float, float]:
    if coarse_best_p is not None and coarse_best_loss is not None:
        coarse_p = float(coarse_best_p)
        coarse_loss = float(coarse_best_loss)
        fmin_loss = float(fminbnd_loss)
        if math.isfinite(coarse_p) and math.isfinite(coarse_loss) and (
            not math.isfinite(fmin_loss) or coarse_loss < fmin_loss
        ):
            return coarse_p, coarse_loss
    return float(fminbnd_p), float(fminbnd_loss)


def _coarse_grid_matches_baseline(
    losses: Optional[Sequence[Any]],
    *,
    baseline_loss: float = 1.0,
) -> bool:
    if losses is None:
        return False
    try:
        raw_losses = list(losses)
    except TypeError:
        return False
    if not raw_losses:
        return False
    parsed_losses: list[float] = []
    for loss in raw_losses:
        try:
            parsed = float(loss)
        except Exception:
            return False
        if not math.isfinite(parsed):
            return False
        parsed_losses.append(parsed)
    baseline = float(baseline_loss)
    tolerance = abs(baseline) * 0.01 + 1.0e-12
    return all(
        abs(loss - baseline) <= tolerance
        for loss in parsed_losses
    )


def _simulink_signal_names_for_loss(
    *,
    model_id: str,
    signals: Sequence[str],
    simulink_signals_available: set[str],
) -> list[str]:
    names = [signal for signal in signals if signal in simulink_signals_available]
    if (
        str(model_id).strip().lower() == _BALL_DROP_MODEL_ID
        and _BALL_DROP_IMPULSE_SIGNAL in signals
        and _BALL_DROP_VELOCITY_SIGNAL in simulink_signals_available
        and _BALL_DROP_VELOCITY_SIGNAL not in names
    ):
        names.append(_BALL_DROP_VELOCITY_SIGNAL)
    return names


def _write_matlab_prepare_helper(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            r"""
            function pre = tsenv_parameter_optimisation_prepare(target)
                modelName = "simulink_model";
                if ~bdIsLoaded(modelName)
                    load_system(modelName);
                end
                configureLogging(modelName);
                set_param(modelName, "FastRestart", "off");
                applyBaselineInitialBlockParameters(modelName, target);
                save_system(modelName);
                simIn = Simulink.SimulationInput(modelName);
                simIn = simIn.setModelParameter( ...
                    "StopTime", num2str(target.intervention_time), ...
                    "SaveFinalState", "on", ...
                    "SaveOperatingPoint", "on");
                simOut = sim(simIn);
                if ~isprop(simOut, "xFinal") || isempty(simOut.xFinal)
                    error("Pre-intervention simulation did not return xFinal operating point.");
                end
                pre = struct();
                pre.model = modelName;
                pre.operating_point = simOut.xFinal;
            end

            function applyBaselineInitialBlockParameters(modelName, target)
                if ~isfield(target, "initial_paths") || isempty(target.initial_paths)
                    return
                end
                for idx = 1:numel(target.initial_paths)
                    set_param( ...
                        target.initial_paths{idx}, ...
                        target.initial_names{idx}, ...
                        num2str(double(target.initial_values(idx))));
                end
                set_param(modelName, "SimulationCommand", "update");
            end

            function configureLogging(modelName)
                try
                    set_param(modelName, "SimscapeLogType", "all");
                    set_param(modelName, "SimscapeLogToSDI", "off");
                    set_param(modelName, "SimscapeLogName", "simlog");
                    set_param(modelName, "SimscapeLogLimitData", "off");
                catch
                end
                set_param(modelName, "SignalLogging", "on");
                set_param(modelName, "SignalLoggingName", "logsout");
                set_param(modelName, "SignalLoggingSaveFormat", "Dataset");
            end
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _write_matlab_loss_helper(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            r"""
            function varargout = tsenv_parameter_optimisation_loss(varargin)
                if nargin >= 1 && ischar(varargin{1}) && strcmp(varargin{1}, "coarse_grid_all")
                    [varargout{1:nargout}] = tsenv_parameter_optimisation_coarse_grid_all(varargin{2}, varargin{3}, varargin{4});
                    return
                end
                if nargin >= 1 && ischar(varargin{1}) && strcmp(varargin{1}, "refine_all")
                    [varargout{1:nargout}] = tsenv_parameter_optimisation_refine_all(varargin{2}, varargin{3}, varargin{4}, varargin{5});
                    return
                end
                if nargin >= 1 && ischar(varargin{1}) && strcmp(varargin{1}, "rms")
                    [varargout{1:nargout}] = tsenv_parameter_optimisation_rms_from_pre(varargin{2}, varargin{3}, varargin{4}, varargin{3}.pre);
                    return
                end
                p = varargin{1};
                target = varargin{2};
                candidate = varargin{3};
                try
                    J = tsenv_parameter_optimisation_loss_from_pre(p, target, candidate, target.pre);
                catch ME
                    warning("tsenv:parameterOptimisationLoss", "%s", ME.message);
                    J = 1e12;
                end
                varargout{1} = J;
            end

            function J = tsenv_parameter_optimisation_loss_from_pre(p, target, candidate, pre)
                try
                    simIn = Simulink.SimulationInput(pre.model);
                    simIn = simIn.setInitialState(pre.operating_point);
                    simIn = simIn.setModelParameter("StopTime", num2str(target.end_time));
                    simIn = applyCandidateToSimulationInput(simIn, p, candidate);
                    simOut = sim(simIn);
                    J = trajectoryLossFromOutput(simOut, target);
                catch ME
                    warning("tsenv:parameterOptimisationLossFromPre", "%s", ME.message);
                    J = 1e12;
                end
            end

            function rmsValues = tsenv_parameter_optimisation_rms_from_pre(p, target, candidate, pre)
                try
                    simIn = Simulink.SimulationInput(pre.model);
                    simIn = simIn.setInitialState(pre.operating_point);
                    simIn = simIn.setModelParameter("StopTime", num2str(target.end_time));
                    simIn = applyCandidateToSimulationInput(simIn, p, candidate);
                    simOut = sim(simIn);
                    rmsValues = trajectoryRmsFromOutput(simOut, target);
                catch ME
                    warning("tsenv:parameterOptimisationRmsFromPre", "%s", ME.message);
                    rmsValues = [];
                end
            end

            function [bestP, bestLoss, leftBounds, rightBounds, gridLossMatrix] = tsenv_parameter_optimisation_coarse_grid_all(gridMatrix, target, candidates)
                gridMatrix = double(gridMatrix);
                [nGrid, nCandidates] = size(gridMatrix);
                if nGrid < 2
                    error("Combined coarse grid requires at least two points.");
                end
                total = nGrid * nCandidates;
                simInputs(total, 1) = Simulink.SimulationInput(target.pre.model);
                candidateIndex = zeros(total, 1);
                gridIndex = zeros(total, 1);
                outIdx = 1;
                for candidateIdx = 1:nCandidates
                    candidate = candidates(candidateIdx);
                    for gridIdx = 1:nGrid
                        simIn = Simulink.SimulationInput(target.pre.model);
                        simIn = simIn.setInitialState(target.pre.operating_point);
                        simIn = simIn.setModelParameter("StopTime", num2str(target.end_time));
                        simIn = applyCandidateToSimulationInput(simIn, gridMatrix(gridIdx, candidateIdx), candidate);
                        simInputs(outIdx) = simIn;
                        candidateIndex(outIdx) = candidateIdx;
                        gridIndex(outIdx) = gridIdx;
                        outIdx = outIdx + 1;
                    end
                end

                simOut = parsim(simInputs, "ShowProgress", "off");
                gridLossMatrix = ones(nGrid, nCandidates) * 1e12;
                for outIdx = 1:total
                    try
                        loss = trajectoryLossFromOutput(simOut(outIdx), target);
                    catch ME
                        warning("tsenv:parameterOptimisationCombinedCoarseLoss", "%s", ME.message);
                        loss = 1e12;
                    end
                    if ~isfinite(loss)
                        loss = 1e12;
                    end
                    gridLossMatrix(gridIndex(outIdx), candidateIndex(outIdx)) = loss;
                end

                bestP = zeros(nCandidates, 1);
                bestLoss = zeros(nCandidates, 1);
                leftBounds = zeros(nCandidates, 1);
                rightBounds = zeros(nCandidates, 1);
                for candidateIdx = 1:nCandidates
                    losses = gridLossMatrix(:, candidateIdx);
                    [bestLoss(candidateIdx), bestIdx] = min(losses);
                    grid = gridMatrix(:, candidateIdx);
                    bestP(candidateIdx) = grid(bestIdx);
                    leftIdx = max(1, bestIdx - 1);
                    rightIdx = min(nGrid, bestIdx + 1);
                    if leftIdx == rightIdx
                        if bestIdx == 1
                            rightIdx = min(nGrid, 2);
                        else
                            leftIdx = max(1, nGrid - 1);
                        end
                    end
                    leftBounds(candidateIdx) = min(grid(leftIdx), grid(rightIdx));
                    rightBounds(candidateIdx) = max(grid(leftIdx), grid(rightIdx));
                end
            end

            function [pOpt, lossOpt, iterations, evaluations] = tsenv_parameter_optimisation_refine_all(candidates, bounds, target, options)
                nCandidates = numel(candidates);
                pOpt = NaN(nCandidates, 1);
                lossOpt = ones(nCandidates, 1) * 1e12;
                iterations = NaN(nCandidates, 1);
                evaluations = NaN(nCandidates, 1);
                parfor candidateIdx = 1:nCandidates
                    candidate = candidates(candidateIdx);
                    leftBound = double(bounds(candidateIdx, 1));
                    rightBound = double(bounds(candidateIdx, 2));
                    try
                        [p, J, ~, output] = fminbnd(@(p) tsenv_parameter_optimisation_loss(p, target, candidate), leftBound, rightBound, options);
                        if ~isfinite(J)
                            J = 1e12;
                        end
                        pOpt(candidateIdx) = p;
                        lossOpt(candidateIdx) = J;
                        try
                            iterations(candidateIdx) = output.iterations;
                        catch
                        end
                        try
                            evaluations(candidateIdx) = output.funcCount;
                        catch
                        end
                    catch ME
                        warning("tsenv:parameterOptimisationRefineAll", "%s", ME.message);
                    end
                end
            end

            function J = trajectoryLossFromOutput(simOut, target)
                channelLosses = zeros(numel(target.signal_names), 1);
                for idx = 1:numel(target.signal_names)
                    signalName = target.signal_names{idx};
                    [t, y] = extractSignal(simOut, target.pre.model, signalName, target.simulink_signal_names);
                    if isempty(t) || isempty(y)
                        error("Empty signal %s", signalName);
                    end
                    [t, y] = dedupeSignalSamples(t, y);
                    signalType = target.signal_types{idx};
                    scale = max(double(target.signal_scales(idx)), eps);
                    if strcmp(char(signalType), "impulse_like")
                        if isBallDropHardStopSignal(target, signalName)
                            if ~isfield(target, "aux_velocity_time") || ~isfield(target, "aux_velocity_values")
                                error("BallDrop Hard_Stop_f impulse loss requires target Velocity.");
                            end
                            [velocityT, velocityY] = extractSignal(simOut, target.pre.model, "Velocity", target.simulink_signal_names);
                            if isempty(velocityT) || isempty(velocityY)
                                error("BallDrop Hard_Stop_f impulse loss requires candidate Velocity.");
                            end
                            [velocityT, velocityY] = dedupeSignalSamples(velocityT, velocityY);
                            channelLosses(idx) = impulseEventLoss( ...
                                t, y, target.time(:), target.values(:, idx), scale, target.intervention_time, ...
                                target.aux_velocity_time(:), target.aux_velocity_values(:), velocityT, velocityY);
                        else
                            channelLosses(idx) = impulseEventLoss(t, y, target.time(:), target.values(:, idx), scale, target.intervention_time, [], [], [], []);
                        end
                    else
                        channelLosses(idx) = continuousSignalLoss(t, y, target.time(:), target.values(:, idx), scale);
                    end
                end
                if isfield(target, "baseline_signal_losses") && numel(target.baseline_signal_losses) == numel(channelLosses)
                    floorValue = 1e-6;
                    if isfield(target, "loss_normalisation_floor")
                        floorValue = double(target.loss_normalisation_floor);
                    end
                    denominators = max(double(target.baseline_signal_losses(:)), floorValue);
                    normalisedLosses = channelLosses(:) ./ denominators;
                    normalisedLosses = normalisedLosses(isfinite(normalisedLosses));
                    if isempty(normalisedLosses)
                        J = 1e12;
                    else
                        J = mean(normalisedLosses);
                    end
                else
                    J = sum(channelLosses);
                end
                if ~isfinite(J)
                    J = 1e12;
                end
            end

            function rmsValues = trajectoryRmsFromOutput(simOut, target)
                rmsValues = NaN(numel(target.signal_names), 1);
                targetTime = double(target.time(:));
                for idx = 1:numel(target.signal_names)
                    signalName = target.signal_names{idx};
                    [t, y] = extractSignal(simOut, target.pre.model, signalName, target.simulink_signal_names);
                    if isempty(t) || isempty(y)
                        error("Empty signal %s", signalName);
                    end
                    [t, y] = dedupeSignalSamples(t, y);
                    targetValue = double(target.values(:, idx));
                    finiteTarget = isfinite(targetTime) & isfinite(targetValue);
                    if ~any(finiteTarget)
                        continue
                    end
                    yq = interp1(t, y, targetTime(finiteTarget), "linear", "extrap");
                    residual = yq(:) - targetValue(finiteTarget);
                    residual = residual(isfinite(residual));
                    if ~isempty(residual)
                        rmsValues(idx) = sqrt(mean(residual .^ 2));
                    end
                end
                rmsValues = rmsValues(isfinite(rmsValues));
            end

            function J = continuousSignalLoss(t, y, targetTime, targetValue, scale)
                yq = interp1(t, y, targetTime, "linear", "extrap");
                residual = (yq(:) - targetValue(:)) ./ scale;
                residual = residual(isfinite(residual));
                if isempty(residual)
                    J = 1e12;
                else
                    J = mean(residual .^ 2);
                end
            end

            function tf = isBallDropHardStopSignal(target, signalName)
                tf = isfield(target, "model_id") ...
                    && strcmpi(char(target.model_id), "BallDrop") ...
                    && strcmp(char(signalName), "Hard_Stop_f");
            end

            function J = impulseEventLoss(t, y, targetTime, targetValue, scale, interventionTime, targetVelocityTime, targetVelocityValue, candidateVelocityTime, candidateVelocityValue)
                t = double(t(:));
                y = double(y(:));
                candidateMask = isfinite(t) & isfinite(y) & t >= interventionTime;
                if ~any(candidateMask)
                    candidateMask = isfinite(t) & isfinite(y);
                end
                targetTime = double(targetTime(:));
                targetValue = double(targetValue(:));
                finiteTarget = isfinite(targetTime) & isfinite(targetValue);
                targetTime = targetTime(finiteTarget);
                targetValue = targetValue(finiteTarget);
                if isempty(targetValue)
                    J = 1e12;
                    return
                end
                threshold = 0.05 * scale;
                [targetEventTime, targetEventValue] = detectImpulseEvents(targetTime, targetValue, threshold, targetVelocityTime, targetVelocityValue);
                [candidateEventTime, candidateEventValue] = detectImpulseEvents(t(candidateMask), y(candidateMask), threshold, candidateVelocityTime, candidateVelocityValue);
                if isempty(targetEventTime) && isempty(candidateEventTime)
                    J = 0;
                    return
                end
                matchWindow = 0.1;
                unmatchedPenalty = 1.0;
                usedCandidate = false(numel(candidateEventTime), 1);
                J = 0;
                for targetIdx = 1:numel(targetEventTime)
                    eligible = find(~usedCandidate & abs(candidateEventTime - targetEventTime(targetIdx)) <= matchWindow);
                    if isempty(eligible)
                        J = J + unmatchedPenalty;
                        continue
                    end
                    [~, localIdx] = max(abs(candidateEventValue(eligible)));
                    bestIdx = eligible(localIdx);
                    usedCandidate(bestIdx) = true;
                    J = J + abs(candidateEventTime(bestIdx) - targetEventTime(targetIdx)) / matchWindow;
                    J = J + abs(abs(candidateEventValue(bestIdx)) - abs(targetEventValue(targetIdx))) / scale;
                end
                J = J + unmatchedPenalty * sum(~usedCandidate);
            end

            function [eventTime, eventValue] = detectImpulseEvents(t, y, threshold, velocityT, velocityY)
                if nargin < 4
                    velocityT = [];
                    velocityY = [];
                end
                t = double(t(:));
                y = double(y(:));
                finite = isfinite(t) & isfinite(y);
                t = t(finite);
                y = y(finite);
                eventTime = [];
                eventValue = [];
                if isempty(t)
                    return
                end
                [t, order] = sort(t);
                y = y(order);
                eventIdx = find(abs(y) > threshold);
                if isempty(eventIdx)
                    return
                end
                candidateEventTime = [];
                candidateEventValue = [];
                breaks = [0; find(diff(eventIdx) > 1); numel(eventIdx)];
                for groupIdx = 1:numel(breaks)-1
                    group = eventIdx((breaks(groupIdx) + 1):breaks(groupIdx + 1));
                    [~, localIdx] = max(abs(y(group)));
                    peakIdx = group(localIdx);
                    candidateEventTime(end + 1, 1) = t(peakIdx); %#ok<AGROW>
                    candidateEventValue(end + 1, 1) = y(peakIdx); %#ok<AGROW>
                end
                if isempty(velocityT) && isempty(velocityY)
                    eventTime = candidateEventTime;
                    eventValue = candidateEventValue;
                    return
                end
                validEventTime = [];
                validEventValue = [];
                for eventIdx = 1:numel(candidateEventTime)
                    if hasBallDropVelocityCrossover(candidateEventTime(eventIdx), velocityT, velocityY)
                        validEventTime(end + 1, 1) = candidateEventTime(eventIdx); %#ok<AGROW>
                        validEventValue(end + 1, 1) = candidateEventValue(eventIdx); %#ok<AGROW>
                    end
                end
                if isempty(validEventTime)
                    return
                end
                minGap = 0.2;
                current = 1;
                for idx = 2:numel(validEventTime)
                    if validEventTime(idx) - validEventTime(idx - 1) <= minGap
                        current(end + 1, 1) = idx; %#ok<AGROW>
                    else
                        [~, localIdx] = max(abs(validEventValue(current)));
                        bestIdx = current(localIdx);
                        eventTime(end + 1, 1) = validEventTime(bestIdx); %#ok<AGROW>
                        eventValue(end + 1, 1) = validEventValue(bestIdx); %#ok<AGROW>
                        current = idx;
                    end
                end
                [~, localIdx] = max(abs(validEventValue(current)));
                bestIdx = current(localIdx);
                eventTime(end + 1, 1) = validEventTime(bestIdx); %#ok<AGROW>
                eventValue(end + 1, 1) = validEventValue(bestIdx); %#ok<AGROW>
            end

            function tf = hasBallDropVelocityCrossover(peakTime, velocityT, velocityY)
                velocityT = double(velocityT(:));
                velocityY = double(velocityY(:));
                finite = isfinite(velocityT) & isfinite(velocityY);
                if ~any(finite)
                    tf = false;
                    return
                end
                velocityT = velocityT(finite);
                velocityY = velocityY(finite);
                window = 0.1;
                preMask = velocityT >= peakTime - window & velocityT <= peakTime;
                postMask = velocityT >= peakTime & velocityT <= peakTime + window;
                if ~any(preMask) || ~any(postMask)
                    tf = false;
                    return
                end
                tf = median(velocityY(preMask)) < 0 && median(velocityY(postMask)) > 0;
            end

            function simIn = applyCandidateToSimulationInput(simIn, p, candidate)
                n = numel(candidate.paths);
                for idx = 1:n
                    candidateValues = candidate.baseline_values;
                    candidateValues(candidate.parameter_index) = p;
                    changedValue = evaluateExpression(candidate.expressions{idx}, candidate.variable_names, candidateValues);
                    simIn = simIn.setBlockParameter( ...
                        candidate.paths{idx}, ...
                        candidate.names{idx}, ...
                        num2str(changedValue));
                end
            end

            function value = evaluateExpression(expression, variableNames, variableValues)
                for nameIdx = 1:numel(variableNames)
                    eval(sprintf("%s = %.17g;", variableNames{nameIdx}, variableValues(nameIdx)));
                end
                value = eval(expression);
                value = double(value);
                if ~isscalar(value) || ~isfinite(value)
                    error("Expression did not evaluate to a finite scalar: %s", expression);
                end
            end

            function [t, y] = extractSignal(simOut, modelName, signalName, simulinkSignalNames)
                if isSimulinkSignal(signalName, simulinkSignalNames)
                    [t, y] = extractSimulinkSignal(simOut, modelName, signalName);
                    if ~isempty(t)
                        return
                    end
                end
                [t, y] = extractSimscapeSignal(simOut, signalName);
                if isempty(t)
                    error("Missing signal %s", signalName);
                end
            end

            function tf = isSimulinkSignal(signalName, simulinkSignalNames)
                if isempty(simulinkSignalNames)
                    tf = true;
                    return
                end
                tf = any(strcmp(char(signalName), simulinkSignalNames));
            end

            function [t, y] = extractSimulinkSignal(simOut, modelName, signalName)
                t = [];
                y = [];
                try
                    logName = char(get_param(modelName, "SignalLoggingName"));
                    if isprop(simOut, logName)
                        logs = simOut.(logName);
                    elseif isprop(simOut, 'logsout')
                        logs = simOut.logsout;
                    else
                        return
                    end
                    sig = logs.get(char(signalName));
                    if isempty(sig)
                        validName = matlab.lang.makeValidName(char(signalName));
                        sig = logs.get(validName);
                    end
                    if isempty(sig)
                        return
                    end
                    ts = sig.Values;
                    t = double(ts.Time(:));
                    y = double(ts.Data);
                    y = y(:);
                catch
                    t = [];
                    y = [];
                end
            end

            function [t, y] = extractSimscapeSignal(simOut, signalName)
                t = [];
                y = [];
                try
                    if isprop(simOut, 'simlog')
                        root = simOut.simlog;
                    else
                        try
                            root = simOut.get('simlog');
                        catch
                            return
                        end
                    end
                    [t, y] = findSimscapeSeries(root, char(signalName));
                catch
                    t = [];
                    y = [];
                end
            end

            function [t, y] = findSimscapeSeries(root, signalName)
                t = [];
                y = [];
                stack = {root};
                pathStack = {''};
                while ~isempty(stack)
                    node = stack{1};
                    nodePath = pathStack{1};
                    stack(1) = [];
                    pathStack(1) = [];
                    try
                        s = node.series;
                        if s.points > 0
                            if isempty(nodePath)
                                rawName = node.id;
                            else
                                rawName = nodePath;
                            end
                            validName = matlab.lang.makeValidName(strrep(char(rawName), '.', '_'));
                            if strcmp(validName, char(signalName))
                                t = double(time(s));
                                y = double(values(s));
                                t = t(:);
                                y = y(:);
                                return
                            end
                        end
                    catch
                    end
                    try
                        f = fieldnames(node);
                        for idx = 1:numel(f)
                            child = node.(f{idx});
                            if isa(child, 'simscape.logging.Node')
                                if isempty(nodePath)
                                    newPath = f{idx};
                                else
                                    newPath = [nodePath '.' f{idx}];
                                end
                                stack{end+1} = child; %#ok<AGROW>
                                pathStack{end+1} = newPath; %#ok<AGROW>
                            end
                        end
                    catch
                    end
                end
            end

            function [t, y] = dedupeSignalSamples(t, y)
                t = double(t(:));
                y = double(y(:));
                finite = isfinite(t) & isfinite(y);
                t = t(finite);
                y = y(finite);
                if isempty(t)
                    return
                end
                [t, order] = sort(t);
                y = y(order);
                [tUnique, ~, groups] = unique(t);
                yUnique = accumarray(groups, y, [], @mean);
                t = tUnique;
                y = yUnique;
            end
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _run_matlab_candidate_optimisations(
    *,
    model_dir: Path,
    run_model_path: Path,
    target_df: pd.DataFrame,
    context: ChildRunContext,
    observable_signals: Sequence[str],
    candidates: Sequence[CandidateSpec],
    baseline_block_bindings: Sequence[BaselineBlockBinding],
    max_iter: Optional[int],
    tol_x: Optional[float],
    coarse_grid_points: Optional[int],
    ignore_impulse_signals: bool = False,
    debug_temp: bool = False,
    simulink_signal_names: Optional[Sequence[str]] = None,
    time0_df: Optional[pd.DataFrame] = None,
    signal_types: Optional[Mapping[str, str]] = None,
    signal_scales: Optional[Mapping[str, float]] = None,
    baseline_signal_losses: Optional[Mapping[str, float]] = None,
    final_loss_signals: Optional[Sequence[str]] = None,
    final_signal_types: Optional[Mapping[str, str]] = None,
    final_signal_scales: Optional[Mapping[str, float]] = None,
    final_profile_targets: Optional[Mapping[str, pd.DataFrame]] = None,
    final_profile_signal_scales: Optional[Mapping[str, Mapping[str, float]]] = None,
    final_profile_baseline_signal_losses: Optional[Mapping[str, Mapping[str, float]]] = None,
    active_final_profile: str = "none",
    final_simulink_signal_names: Optional[Sequence[str]] = None,
    matlab_workers: int = _DEFAULT_MATLAB_WORKERS,
    matlab_engine: Any = None,
) -> Sequence[CandidateResult]:
    _ = time0_df
    try:
        import matlab  # type: ignore
    except Exception as exc:
        raise RuntimeError("MATLAB engine package is required for parameter optimisation") from exc

    active_final_profile = normalize_noise_profile(active_final_profile)
    final_loss_signal_list = [str(signal) for signal in (final_loss_signals or observable_signals)]
    final_signal_type_lookup = final_signal_types or signal_types or {}
    final_has_impulse_signals = any(
        str(final_signal_type_lookup.get(signal, "continuous")).strip().lower()
        == "impulse_like"
        for signal in final_loss_signal_list
    )
    final_continuous_signal_list = [
        signal
        for signal in final_loss_signal_list
        if str(final_signal_type_lookup.get(signal, "continuous")).strip().lower()
        == "continuous"
    ]
    final_impulse_signal_list = [
        signal
        for signal in final_loss_signal_list
        if str(final_signal_type_lookup.get(signal, "continuous")).strip().lower()
        == "impulse_like"
    ]

    def target_arrays(
        signals: Sequence[str],
        *,
        source_df: pd.DataFrame = target_df,
    ) -> tuple[np.ndarray, np.ndarray]:
        signal_list = [str(signal) for signal in signals]
        signal_df = source_df[["time", *signal_list]].copy()
        mask = pd.to_numeric(signal_df["time"], errors="coerce").to_numpy(dtype=float) >= float(
            context.intervention_time
        )
        if not np.any(mask):
            mask = np.ones(len(signal_df), dtype=bool)
        times = pd.to_numeric(signal_df.loc[mask, "time"], errors="coerce").to_numpy(dtype=float)
        values = signal_df.loc[mask, signal_list].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=float)
        finite_rows = np.isfinite(times) & np.all(np.isfinite(values), axis=1)
        times = times[finite_rows]
        values = values[finite_rows, :]
        if times.size == 0 or values.size == 0:
            raise ValueError("Target child trajectory has no finite post-intervention samples")
        return times, values

    def velocity_arrays(source_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if _BALL_DROP_VELOCITY_SIGNAL not in source_df.columns:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        velocity_times = pd.to_numeric(source_df["time"], errors="coerce").to_numpy(dtype=float)
        velocity_values = pd.to_numeric(
            source_df[_BALL_DROP_VELOCITY_SIGNAL],
            errors="coerce",
        ).to_numpy(dtype=float)
        finite_velocity = np.isfinite(velocity_times) & np.isfinite(velocity_values)
        return velocity_times[finite_velocity], velocity_values[finite_velocity]

    target_times, target_values = target_arrays(observable_signals)
    final_target_times, final_target_values = target_arrays(final_loss_signal_list)
    continuous_target_arrays = (
        target_arrays(final_continuous_signal_list)
        if final_continuous_signal_list
        else None
    )
    impulse_target_arrays = (
        target_arrays(final_impulse_signal_list)
        if final_impulse_signal_list
        else None
    )
    needs_ball_drop_velocity = (
        str(model_dir.name).strip().lower() == _BALL_DROP_MODEL_ID
        and _BALL_DROP_IMPULSE_SIGNAL in final_loss_signal_list
    )
    if needs_ball_drop_velocity and _BALL_DROP_VELOCITY_SIGNAL not in target_df.columns:
        raise ValueError("BallDrop Hard_Stop_f impulse loss requires target Velocity")
    aux_velocity_times, aux_velocity_values = velocity_arrays(target_df)
    profile_target_dfs: dict[str, pd.DataFrame] = {
        normalize_noise_profile(profile): profile_df
        for profile, profile_df in (final_profile_targets or {}).items()
    }
    if active_final_profile not in profile_target_dfs:
        profile_target_dfs[active_final_profile] = target_df
    profile_target_dfs = {
        profile: profile_target_dfs[profile]
        for profile in ("none", "low", "high")
        if profile in profile_target_dfs
    }

    baseline_variables = {
        str(key): _coerce_finite_float(value, label=f"baseline parameter {key}")
        for key, value in context.baseline_parameters.items()
        if str(key).strip()
    }
    variable_names = sorted(baseline_variables.keys(), key=str.casefold)
    for name in variable_names:
        if not MATLAB_IDENTIFIER_RE.fullmatch(name):
            raise ValueError(f"Baseline parameter {name!r} is not a valid MATLAB identifier")

    temp_root = Path(tempfile.mkdtemp(prefix="tsenv-parameter-optimisation-"))
    if debug_temp:
        click.echo(f"Retaining temporary parameter optimisation directory: {temp_root}")
    try:
        shutil.copy2(run_model_path, temp_root / "simulink_model.mdl")
        _write_matlab_prepare_helper(temp_root / "tsenv_parameter_optimisation_prepare.m")
        _write_matlab_loss_helper(temp_root / "tsenv_parameter_optimisation_loss.m")

        original_cwd = Path.cwd()
        os.chdir(temp_root)
        results: list[CandidateResult] = []
        mle: Any = None
        temp_path_added = False
        owns_matlab_session = matlab_engine is None
        matlab_session = (
            build_metadata.matlab_session(original_cwd)
            if owns_matlab_session
            else nullcontext(matlab_engine)
        )
        try:
            with matlab_session as mle:
                if mle is None:
                    raise RuntimeError("MATLAB session did not start")
                mle.cd(str(temp_root), nargout=0)
                mle.eval(
                    "if bdIsLoaded('simulink_model'), close_system('simulink_model', 0); end; "
                    "clear tsenv_parameter_optimisation_loss tsenv_parameter_optimisation_prepare; "
                    "clear functions;",
                    nargout=0,
                )
                mle.eval(f"addpath({_matlab_char_literal(temp_root)}, '-begin');", nargout=0)
                temp_path_added = True
                mle.addpath(str(model_dir.parent), nargout=0)
                if owns_matlab_session:
                    _ensure_matlab_parallel_pool(mle, matlab_workers)
                mle.load_system(str((temp_root / "simulink_model.mdl").resolve()), nargout=0)
                mle.workspace["tsenv_target_time"] = matlab.double(
                    [float(value) for value in target_times.reshape(-1)]
                )
                mle.workspace["tsenv_target_values"] = matlab.double(
                    target_values.astype(float).tolist()
                )
                mle.workspace["tsenv_final_target_time"] = matlab.double(
                    [float(value) for value in final_target_times.reshape(-1)]
                )
                mle.workspace["tsenv_final_target_values"] = matlab.double(
                    final_target_values.astype(float).tolist()
                )
                mle.workspace["tsenv_aux_velocity_time"] = matlab.double(
                    [float(value) for value in aux_velocity_times.reshape(-1)]
                )
                mle.workspace["tsenv_aux_velocity_values"] = matlab.double(
                    [float(value) for value in aux_velocity_values.reshape(-1)]
                )
                resolved_signal_types = {
                    str(signal): str((signal_types or {}).get(signal, "continuous"))
                    for signal in observable_signals
                }
                resolved_signal_scales = {
                    str(signal): float((signal_scales or {}).get(signal, 1.0))
                    for signal in observable_signals
                }
                resolved_final_signal_types = {
                    str(signal): str((final_signal_types or signal_types or {}).get(signal, "continuous"))
                    for signal in final_loss_signal_list
                }
                resolved_final_signal_scales = {
                    str(signal): float((final_signal_scales or signal_scales or {}).get(signal, 1.0))
                    for signal in final_loss_signal_list
                }

                def assign_target_struct(
                    *,
                    target_name: str,
                    signal_list: Sequence[str],
                    times: np.ndarray,
                    values: np.ndarray,
                    aux_times: np.ndarray,
                    aux_values: np.ndarray,
                    signal_types_lookup: Mapping[str, str],
                    signal_scales_lookup: Mapping[str, float],
                    signal_names_for_simulink: Sequence[str],
                    baseline_losses: Optional[Mapping[str, float]] = None,
                ) -> None:
                    time_var = f"{target_name}_time"
                    values_var = f"{target_name}_values"
                    aux_time_var = f"{target_name}_aux_velocity_time"
                    aux_values_var = f"{target_name}_aux_velocity_values"
                    mle.workspace[time_var] = matlab.double(
                        [float(value) for value in times.reshape(-1)]
                    )
                    mle.workspace[values_var] = matlab.double(
                        values.astype(float).tolist()
                    )
                    mle.workspace[aux_time_var] = matlab.double(
                        [float(value) for value in aux_times.reshape(-1)]
                    )
                    mle.workspace[aux_values_var] = matlab.double(
                        [float(value) for value in aux_values.reshape(-1)]
                    )
                    mle.eval(
                        f"{target_name} = struct(); "
                        f"{target_name}.model_id = {_matlab_char_literal(model_dir.name)}; "
                        f"{target_name}.time = {time_var}(:); "
                        f"{target_name}.signal_names = {_matlab_cell_array(signal_list)}; "
                        f"{target_name}.signal_types = {_matlab_cell_array([signal_types_lookup[str(signal)] for signal in signal_list])}; "
                        f"{target_name}.signal_scales = {_matlab_numeric_vector([signal_scales_lookup[str(signal)] for signal in signal_list])}; "
                        f"{target_name}.simulink_signal_names = {_matlab_cell_array(signal_names_for_simulink)}; "
                        f"{target_name}.values = {values_var}; "
                        f"{target_name}.aux_velocity_time = {aux_time_var}(:); "
                        f"{target_name}.aux_velocity_values = {aux_values_var}(:); "
                        f"{target_name}.intervention_time = {float(context.intervention_time):.17g}; "
                        f"{target_name}.end_time = {float(np.max(times)):.17g};",
                        nargout=0,
                    )
                    if baseline_losses is not None:
                        mle.eval(
                            f"{target_name}.baseline_signal_losses = "
                            f"{_matlab_numeric_vector([baseline_losses.get(str(signal), 0.0) for signal in signal_list])}; "
                            f"{target_name}.loss_normalisation_floor = {_LOSS_NORMALISATION_FLOOR:.17g};",
                            nargout=0,
                        )

                mle.eval(
                    "tsenv_target = struct(); "
                    f"tsenv_target.model_id = {_matlab_char_literal(model_dir.name)}; "
                    "tsenv_target.time = tsenv_target_time(:); "
                    f"tsenv_target.signal_names = {_matlab_cell_array(observable_signals)}; "
                    f"tsenv_target.signal_types = {_matlab_cell_array([resolved_signal_types[str(signal)] for signal in observable_signals])}; "
                    f"tsenv_target.signal_scales = {_matlab_numeric_vector([resolved_signal_scales[str(signal)] for signal in observable_signals])}; "
                    f"tsenv_target.simulink_signal_names = {_matlab_cell_array(simulink_signal_names or [])}; "
                    "tsenv_target.values = tsenv_target_values; "
                    "tsenv_target.aux_velocity_time = tsenv_aux_velocity_time(:); "
                    "tsenv_target.aux_velocity_values = tsenv_aux_velocity_values(:); "
                    f"tsenv_target.intervention_time = {float(context.intervention_time):.17g}; "
                    f"tsenv_target.end_time = {float(np.max(target_times)):.17g};",
                    nargout=0,
                )
                mle.eval(
                    "tsenv_target.initial_paths = "
                    f"{_matlab_cell_array([binding.path for binding in baseline_block_bindings])}; "
                    "tsenv_target.initial_names = "
                    f"{_matlab_cell_array([binding.name for binding in baseline_block_bindings])}; "
                    "tsenv_target.initial_values = "
                    f"{_matlab_numeric_vector([binding.value for binding in baseline_block_bindings])};",
                    nargout=0,
                )
                if baseline_signal_losses is not None:
                    mle.eval(
                        "tsenv_target.baseline_signal_losses = "
                        f"{_matlab_numeric_vector([baseline_signal_losses.get(str(signal), 0.0) for signal in observable_signals])}; "
                        f"tsenv_target.loss_normalisation_floor = {_LOSS_NORMALISATION_FLOOR:.17g};",
                        nargout=0,
                    )
                mle.eval(
                    "tsenv_final_target = struct(); "
                    f"tsenv_final_target.model_id = {_matlab_char_literal(model_dir.name)}; "
                    "tsenv_final_target.time = tsenv_final_target_time(:); "
                    f"tsenv_final_target.signal_names = {_matlab_cell_array(final_loss_signal_list)}; "
                    f"tsenv_final_target.signal_types = {_matlab_cell_array([resolved_final_signal_types[str(signal)] for signal in final_loss_signal_list])}; "
                    f"tsenv_final_target.signal_scales = {_matlab_numeric_vector([resolved_final_signal_scales[str(signal)] for signal in final_loss_signal_list])}; "
                    f"tsenv_final_target.simulink_signal_names = {_matlab_cell_array(final_simulink_signal_names or simulink_signal_names or [])}; "
                    "tsenv_final_target.values = tsenv_final_target_values; "
                    "tsenv_final_target.aux_velocity_time = tsenv_aux_velocity_time(:); "
                    "tsenv_final_target.aux_velocity_values = tsenv_aux_velocity_values(:); "
                    f"tsenv_final_target.intervention_time = {float(context.intervention_time):.17g}; "
                    f"tsenv_final_target.end_time = {float(np.max(final_target_times)):.17g};",
                    nargout=0,
                )
                active_final_baseline_losses = (
                    (final_profile_baseline_signal_losses or {}).get(active_final_profile)
                    if final_profile_baseline_signal_losses is not None
                    else None
                )
                if active_final_baseline_losses is not None:
                    mle.eval(
                        "tsenv_final_target.baseline_signal_losses = "
                        f"{_matlab_numeric_vector([active_final_baseline_losses.get(str(signal), 0.0) for signal in final_loss_signal_list])}; "
                        f"tsenv_final_target.loss_normalisation_floor = {_LOSS_NORMALISATION_FLOOR:.17g};",
                        nargout=0,
                    )
                if continuous_target_arrays is not None:
                    continuous_times, continuous_values = continuous_target_arrays
                    assign_target_struct(
                        target_name="tsenv_continuous_target",
                        signal_list=final_continuous_signal_list,
                        times=continuous_times,
                        values=continuous_values,
                        aux_times=aux_velocity_times,
                        aux_values=aux_velocity_values,
                        signal_types_lookup=resolved_final_signal_types,
                        signal_scales_lookup=resolved_final_signal_scales,
                        signal_names_for_simulink=final_simulink_signal_names or simulink_signal_names or [],
                    )
                if impulse_target_arrays is not None:
                    impulse_times, impulse_values = impulse_target_arrays
                    assign_target_struct(
                        target_name="tsenv_impulse_target",
                        signal_list=final_impulse_signal_list,
                        times=impulse_times,
                        values=impulse_values,
                        aux_times=aux_velocity_times,
                        aux_values=aux_velocity_values,
                        signal_types_lookup=resolved_final_signal_types,
                        signal_scales_lookup=resolved_final_signal_scales,
                        signal_names_for_simulink=final_simulink_signal_names or simulink_signal_names or [],
                    )
                profile_target_structs: dict[str, dict[str, Optional[str]]] = {}
                for profile, profile_df in profile_target_dfs.items():
                    profile_scales = {
                        str(signal): float(
                            (
                                (final_profile_signal_scales or {})
                                .get(profile, {})
                                .get(signal, resolved_final_signal_scales[str(signal)])
                            )
                        )
                        for signal in final_loss_signal_list
                    }
                    profile_aux_times, profile_aux_values = velocity_arrays(profile_df)
                    profile_full_times, profile_full_values = target_arrays(
                        final_loss_signal_list,
                        source_df=profile_df,
                    )
                    full_target_name = f"tsenv_final_profile_{profile}"
                    assign_target_struct(
                        target_name=full_target_name,
                        signal_list=final_loss_signal_list,
                        times=profile_full_times,
                        values=profile_full_values,
                        aux_times=profile_aux_times,
                        aux_values=profile_aux_values,
                        signal_types_lookup=resolved_final_signal_types,
                        signal_scales_lookup=profile_scales,
                        signal_names_for_simulink=final_simulink_signal_names or simulink_signal_names or [],
                        baseline_losses=(final_profile_baseline_signal_losses or {}).get(profile),
                    )
                    continuous_target_name: Optional[str] = None
                    if final_continuous_signal_list:
                        profile_continuous_times, profile_continuous_values = target_arrays(
                            final_continuous_signal_list,
                            source_df=profile_df,
                        )
                        continuous_target_name = f"tsenv_continuous_profile_{profile}"
                        assign_target_struct(
                            target_name=continuous_target_name,
                            signal_list=final_continuous_signal_list,
                            times=profile_continuous_times,
                            values=profile_continuous_values,
                            aux_times=profile_aux_times,
                            aux_values=profile_aux_values,
                            signal_types_lookup=resolved_final_signal_types,
                            signal_scales_lookup=profile_scales,
                            signal_names_for_simulink=final_simulink_signal_names or simulink_signal_names or [],
                        )
                    impulse_target_name: Optional[str] = None
                    if final_impulse_signal_list:
                        profile_impulse_times, profile_impulse_values = target_arrays(
                            final_impulse_signal_list,
                            source_df=profile_df,
                        )
                        impulse_target_name = f"tsenv_impulse_profile_{profile}"
                        assign_target_struct(
                            target_name=impulse_target_name,
                            signal_list=final_impulse_signal_list,
                            times=profile_impulse_times,
                            values=profile_impulse_values,
                            aux_times=profile_aux_times,
                            aux_values=profile_aux_values,
                            signal_types_lookup=resolved_final_signal_types,
                            signal_scales_lookup=profile_scales,
                            signal_names_for_simulink=final_simulink_signal_names or simulink_signal_names or [],
                        )
                    profile_target_structs[profile] = {
                        "full": full_target_name,
                        "continuous": continuous_target_name,
                        "impulse": impulse_target_name,
                    }
                click.echo(
                    f"Preparing baseline operating point at t={context.intervention_time:g}s...",
                    err=True,
                )
                mle.eval(
                    "tsenv_target.pre = tsenv_parameter_optimisation_prepare(tsenv_target);",
                    nargout=0,
                )
                mle.eval("tsenv_final_target.pre = tsenv_target.pre;", nargout=0)
                if continuous_target_arrays is not None:
                    mle.eval("tsenv_continuous_target.pre = tsenv_target.pre;", nargout=0)
                if impulse_target_arrays is not None:
                    mle.eval("tsenv_impulse_target.pre = tsenv_target.pre;", nargout=0)
                for profile_structs in profile_target_structs.values():
                    for target_name in profile_structs.values():
                        if target_name is not None:
                            mle.eval(f"{target_name}.pre = tsenv_target.pre;", nargout=0)

                def vector_from_workspace(name: str) -> list[float]:
                    raw = np.asarray(mle.workspace[name], dtype=float).reshape(-1)
                    return [float(value) for value in raw]

                def finite_scalar(value: Any) -> Optional[float]:
                    try:
                        parsed = float(value)
                    except Exception:
                        return None
                    return parsed if math.isfinite(parsed) else None

                def assign_candidate_struct(index: int, candidate: CandidateSpec) -> None:
                    if candidate.parameter not in baseline_variables:
                        raise ValueError(
                            f"candidate {candidate.parameter!r} is missing from baseline parameters"
                        )
                    candidate_values = [baseline_variables[name] for name in variable_names]
                    parameter_index = variable_names.index(candidate.parameter) + 1
                    mle.eval("tsenv_candidate = struct();", nargout=0)
                    mle.eval(
                        f"tsenv_candidate.parameter = {_matlab_char_literal(candidate.parameter)};",
                        nargout=0,
                    )
                    mle.eval(
                        f"tsenv_candidate.paths = {_matlab_cell_array([_simulink_model_path_for_snapshot(binding.path) for binding in candidate.bindings])};",
                        nargout=0,
                    )
                    mle.eval(
                        f"tsenv_candidate.names = {_matlab_cell_array([binding.name for binding in candidate.bindings])};",
                        nargout=0,
                    )
                    mle.eval(
                        f"tsenv_candidate.expressions = {_matlab_cell_array([binding.expression for binding in candidate.bindings])};",
                        nargout=0,
                    )
                    mle.eval(
                        f"tsenv_candidate.variable_names = {_matlab_cell_array(variable_names)};",
                        nargout=0,
                    )
                    mle.eval(
                        f"tsenv_candidate.baseline_values = {_matlab_numeric_vector(candidate_values)};",
                        nargout=0,
                    )
                    mle.eval(
                        f"tsenv_candidate.parameter_index = {int(parameter_index)};",
                        nargout=0,
                    )
                    mle.eval(
                        f"tsenv_candidate.intervention_time = {float(context.intervention_time):.17g};",
                        nargout=0,
                    )
                    mle.eval(f"tsenv_candidates({int(index)}, 1) = tsenv_candidate;", nargout=0)

                optimset_args = []
                if max_iter is not None:
                    optimset_args.extend(["'MaxIter'", str(int(max_iter))])
                if tol_x is not None:
                    optimset_args.extend(["'TolX'", f"{float(tol_x):.17g}"])
                options_expr = (
                    "optimset(" + ",".join(optimset_args) + ")"
                    if optimset_args
                    else "optimset()"
                )
                mle.eval(f"tsenv_refine_options = {options_expr};", nargout=0)

                active_candidates: list[CandidateSpec] = []
                active_matlab_indices: dict[str, int] = {}
                mle.eval("clear tsenv_candidates;", nargout=0)
                for candidate in candidates:
                    try:
                        active_candidates.append(candidate)
                        active_matlab_indices[candidate.parameter] = len(active_candidates)
                        assign_candidate_struct(len(active_candidates), candidate)
                    except Exception as exc:  # noqa: BLE001 - keep other candidates running
                        results.append(
                            CandidateResult(
                                parameter=candidate.parameter,
                                status="failed",
                                loss=None,
                                p_opt=None,
                                error=str(exc),
                            )
                        )
                if not active_candidates:
                    return results

                coarse_by_parameter: dict[str, dict[str, Any]] = {
                    candidate.parameter: {
                        "coarse_best_p": None,
                        "coarse_best_loss": None,
                        "coarse_grid_losses": None,
                        "refinement_bracket": None,
                        "coarse_grid_points": None,
                    }
                    for candidate in active_candidates
                }

                if _coarse_search_enabled(coarse_grid_points):
                    effective_coarse_points = int(coarse_grid_points)
                    click.echo(
                        "Running combined coarse grid for "
                        f"{len(active_candidates)} candidates and "
                        f"{effective_coarse_points} grid points each...",
                        err=True,
                    )
                    grids = [
                        np.linspace(
                            float(candidate.minimum),
                            float(candidate.maximum),
                            effective_coarse_points,
                            dtype=float,
                        )
                        for candidate in active_candidates
                    ]
                    grid_matrix = np.column_stack(grids)
                    try:
                        mle.workspace["tsenv_coarse_grid_matrix"] = matlab.double(
                            grid_matrix.astype(float).tolist()
                        )
                        mle.eval(
                            "[tsenv_all_coarse_best_p, tsenv_all_coarse_best_loss, "
                            "tsenv_all_refine_left, tsenv_all_refine_right, "
                            "tsenv_all_grid_loss] = "
                            "tsenv_parameter_optimisation_loss('coarse_grid_all', "
                            "tsenv_coarse_grid_matrix, tsenv_target, tsenv_candidates);",
                            nargout=0,
                        )
                        coarse_best_p_values = vector_from_workspace("tsenv_all_coarse_best_p")
                        coarse_best_loss_values = vector_from_workspace("tsenv_all_coarse_best_loss")
                        refine_left_values = vector_from_workspace("tsenv_all_refine_left")
                        refine_right_values = vector_from_workspace("tsenv_all_refine_right")
                        raw_grid_loss_matrix = np.asarray(
                            mle.workspace["tsenv_all_grid_loss"],
                            dtype=float,
                        )
                        if raw_grid_loss_matrix.ndim == 1:
                            raw_grid_loss_matrix = raw_grid_loss_matrix.reshape(
                                effective_coarse_points,
                                len(active_candidates),
                            )
                        for idx, candidate in enumerate(active_candidates):
                            coarse_by_parameter[candidate.parameter] = {
                                "coarse_best_p": coarse_best_p_values[idx],
                                "coarse_best_loss": coarse_best_loss_values[idx],
                                "coarse_grid_losses": [
                                    float(value)
                                    for value in raw_grid_loss_matrix[:, idx].reshape(-1)
                                ],
                                "refinement_bracket": (
                                    refine_left_values[idx],
                                    refine_right_values[idx],
                                ),
                                "coarse_grid_points": effective_coarse_points,
                            }
                    except Exception as exc:  # noqa: BLE001 - documented fallback keeps run usable
                        click.echo(
                            "Combined coarse parsim search failed; falling back to "
                            f"full-interval fminbnd refinement: {exc}",
                            err=True,
                        )

                candidate_records: list[dict[str, Any]] = []
                baseline_return_parameters: set[str] = set()
                for candidate in active_candidates:
                    coarse = coarse_by_parameter[candidate.parameter]
                    if not _coarse_grid_matches_baseline(coarse.get("coarse_grid_losses")):
                        continue
                    if candidate.parameter not in baseline_variables:
                        continue
                    click.echo(
                        f"Coarse grid for {candidate.parameter} matches baseline loss; "
                        "using baseline parameter value.",
                        err=True,
                    )
                    baseline_return_parameters.add(candidate.parameter)
                    candidate_records.append(
                        {
                            "candidate": candidate,
                            "matlab_index": active_matlab_indices[candidate.parameter],
                            "p_opt": float(baseline_variables[candidate.parameter]),
                            "optimisation_loss": 1.0,
                            "iterations": None,
                            "evaluations": None,
                            **coarse,
                        }
                    )

                early_stop_candidate: Optional[CandidateSpec] = None
                early_stop_loss: Optional[float] = None
                for candidate in active_candidates:
                    if candidate.parameter in baseline_return_parameters:
                        continue
                    coarse = coarse_by_parameter[candidate.parameter]
                    loss = finite_scalar(coarse.get("coarse_best_loss"))
                    if loss is None or loss >= _COARSE_GRID_EARLY_STOP_LOSS:
                        continue
                    if early_stop_loss is None or loss < early_stop_loss:
                        early_stop_loss = loss
                        early_stop_candidate = candidate

                if early_stop_candidate is not None:
                    coarse = coarse_by_parameter[early_stop_candidate.parameter]
                    click.echo(
                        f"Coarse grid loss below {_COARSE_GRID_EARLY_STOP_LOSS:g} "
                        f"for {early_stop_candidate.parameter}; stopping search early.",
                        err=True,
                    )
                    candidate_records.append(
                        {
                            "candidate": early_stop_candidate,
                            "matlab_index": active_matlab_indices[early_stop_candidate.parameter],
                            "p_opt": float(coarse["coarse_best_p"]),
                            "optimisation_loss": float(coarse["coarse_best_loss"]),
                            "iterations": None,
                            "evaluations": None,
                            **coarse,
                        }
                    )
                else:
                    refine_candidates = [
                        candidate
                        for candidate in active_candidates
                        if candidate.parameter not in baseline_return_parameters
                    ]
                    bounds: list[list[float]] = []
                    for candidate in refine_candidates:
                        coarse = coarse_by_parameter[candidate.parameter]
                        bracket = coarse.get("refinement_bracket")
                        if bracket is None:
                            left_bound = float(candidate.minimum)
                            right_bound = float(candidate.maximum)
                        else:
                            left_bound, right_bound = bracket
                        bounds.append([float(left_bound), float(right_bound)])
                    refine_p_values: list[Optional[float]] = [None] * len(refine_candidates)
                    refine_loss_values: list[Optional[float]] = [None] * len(refine_candidates)
                    refine_iterations: list[Optional[int]] = [None] * len(refine_candidates)
                    refine_evaluations: list[Optional[int]] = [None] * len(refine_candidates)
                    if refine_candidates:
                        try:
                            click.echo(
                                f"Refining {len(refine_candidates)} candidates in parallel...",
                                err=True,
                            )
                            mle.eval("clear tsenv_refine_candidates;", nargout=0)
                            for refine_idx, candidate in enumerate(refine_candidates, start=1):
                                mle.eval(
                                    "tsenv_refine_candidates"
                                    f"({refine_idx}, 1) = "
                                    "tsenv_candidates"
                                    f"({active_matlab_indices[candidate.parameter]});",
                                    nargout=0,
                                )
                            mle.workspace["tsenv_refine_bounds"] = matlab.double(bounds)
                            mle.eval(
                                "[tsenv_refine_p, tsenv_refine_loss, "
                                "tsenv_refine_iterations, tsenv_refine_evaluations] = "
                                "tsenv_parameter_optimisation_loss('refine_all', "
                                "tsenv_refine_candidates, tsenv_refine_bounds, tsenv_target, "
                                "tsenv_refine_options);",
                                nargout=0,
                            )
                            refine_p_values = [finite_scalar(value) for value in vector_from_workspace("tsenv_refine_p")]
                            refine_loss_values = [finite_scalar(value) for value in vector_from_workspace("tsenv_refine_loss")]
                            refine_iterations = [
                                int(value) if finite_scalar(value) is not None else None
                                for value in vector_from_workspace("tsenv_refine_iterations")
                            ]
                            refine_evaluations = [
                                int(value) if finite_scalar(value) is not None else None
                                for value in vector_from_workspace("tsenv_refine_evaluations")
                            ]
                        except Exception as exc:  # noqa: BLE001 - parallel fallback path
                            click.echo(
                                "Parallel fminbnd refinement failed; falling back to serial "
                                f"refinement: {exc}",
                                err=True,
                            )
                            for idx, candidate in enumerate(refine_candidates):
                                left_bound, right_bound = bounds[idx]
                                click.echo(
                                    f"Refining candidate {candidate.parameter} "
                                    f"({idx + 1}/{len(refine_candidates)})...",
                                    err=True,
                                )
                                try:
                                    mle.eval(
                                        f"tsenv_candidate = tsenv_candidates({active_matlab_indices[candidate.parameter]});",
                                        nargout=0,
                                    )
                                    mle.eval(
                                        "[tsenv_p_opt, tsenv_loss, tsenv_exitflag, tsenv_output] = "
                                        "fminbnd(@(p) tsenv_parameter_optimisation_loss(p, tsenv_target, tsenv_candidate), "
                                        f"{left_bound:.17g}, {right_bound:.17g}, tsenv_refine_options);",
                                        nargout=0,
                                    )
                                    refine_p_values[idx] = finite_scalar(mle.workspace["tsenv_p_opt"])
                                    refine_loss_values[idx] = finite_scalar(mle.workspace["tsenv_loss"])
                                    try:
                                        refine_iterations[idx] = int(mle.eval("tsenv_output.iterations", nargout=1))
                                    except Exception:
                                        refine_iterations[idx] = None
                                    try:
                                        refine_evaluations[idx] = int(mle.eval("tsenv_output.funcCount", nargout=1))
                                    except Exception:
                                        refine_evaluations[idx] = None
                                except Exception as candidate_exc:  # noqa: BLE001 - keep other candidates running
                                    click.echo(
                                        f"Serial refinement failed for {candidate.parameter}: {candidate_exc}",
                                        err=True,
                                    )
                        for idx, candidate in enumerate(refine_candidates):
                            coarse = coarse_by_parameter[candidate.parameter]
                            fmin_p = refine_p_values[idx]
                            fmin_loss = refine_loss_values[idx]
                            if fmin_p is None:
                                fmin_p = float(candidate.minimum)
                            if fmin_loss is None:
                                fmin_loss = _FAILED_CANDIDATE_LOSS
                            p_opt, optimisation_loss = _select_candidate_optimum(
                                fminbnd_p=float(fmin_p),
                                fminbnd_loss=float(fmin_loss),
                                coarse_best_p=coarse.get("coarse_best_p"),
                                coarse_best_loss=coarse.get("coarse_best_loss"),
                            )
                            candidate_records.append(
                                {
                                    "candidate": candidate,
                                    "matlab_index": active_matlab_indices[candidate.parameter],
                                    "p_opt": p_opt,
                                    "optimisation_loss": optimisation_loss,
                                    "iterations": refine_iterations[idx],
                                    "evaluations": refine_evaluations[idx],
                                    **coarse,
                                }
                            )

                def evaluate_target_loss(
                    *,
                    target_name: str,
                    workspace_name: str,
                    label: str,
                    candidate_parameter: str,
                ) -> Optional[float]:
                    try:
                        mle.eval(
                            f"{workspace_name} = tsenv_parameter_optimisation_loss("
                            f"tsenv_p_opt, {target_name}, tsenv_candidate);",
                            nargout=0,
                        )
                        return float(mle.workspace[workspace_name])
                    except Exception as exc:  # noqa: BLE001 - component diagnostics should not abort
                        click.echo(
                            f"{label} failed for {candidate_parameter}: {exc}",
                            err=True,
                        )
                        return None

                for record in candidate_records:
                    candidate = record["candidate"]
                    try:
                        mle.eval(
                            f"tsenv_candidate = tsenv_candidates({int(record['matlab_index'])});",
                            nargout=0,
                        )
                        mle.workspace["tsenv_p_opt"] = float(record["p_opt"])
                        continuous_loss = None
                        impulse_loss = None
                        final_profile_losses: dict[str, dict[str, Optional[float]]] = {}
                        for profile, profile_structs in profile_target_structs.items():
                            profile_loss = evaluate_target_loss(
                                target_name=str(profile_structs["full"]),
                                workspace_name=f"tsenv_{profile}_final_loss",
                                label=f"Final {profile} loss",
                                candidate_parameter=candidate.parameter,
                            )
                            profile_continuous_loss = None
                            continuous_target_name = profile_structs.get("continuous")
                            if continuous_target_name is not None:
                                profile_continuous_loss = evaluate_target_loss(
                                    target_name=continuous_target_name,
                                    workspace_name=f"tsenv_{profile}_continuous_loss",
                                    label=f"Continuous {profile} component loss",
                                    candidate_parameter=candidate.parameter,
                                )
                            profile_impulse_loss = None
                            impulse_target_name = profile_structs.get("impulse")
                            if final_has_impulse_signals and impulse_target_name is not None:
                                profile_impulse_loss = evaluate_target_loss(
                                    target_name=impulse_target_name,
                                    workspace_name=f"tsenv_{profile}_impulse_loss",
                                    label=f"Impulse {profile} component loss",
                                    candidate_parameter=candidate.parameter,
                                )
                            final_profile_losses[profile] = _final_loss_entry(
                                loss=profile_loss,
                                continuous_loss=profile_continuous_loss,
                                impulse_loss=profile_impulse_loss,
                            )

                        active_entry = final_profile_losses.get(active_final_profile)
                        if active_entry is not None and active_entry["loss"] is not None:
                            final_loss = float(active_entry["loss"])
                            continuous_loss = active_entry.get("loss_continuous")
                            impulse_loss = active_entry.get("loss_impulse")
                        else:
                            final_loss = _FAILED_CANDIDATE_LOSS
                        rms_values: list[float] = []
                        active_profile_structs = profile_target_structs.get(active_final_profile)
                        if active_profile_structs is not None:
                            active_target_name = active_profile_structs.get("full")
                            if active_target_name is not None:
                                try:
                                    mle.eval(
                                        "tsenv_active_rms = "
                                        "tsenv_parameter_optimisation_loss('rms', "
                                        f"tsenv_p_opt, {active_target_name}, tsenv_candidate);",
                                        nargout=0,
                                    )
                                    raw_rms = np.asarray(
                                        mle.workspace["tsenv_active_rms"],
                                        dtype=float,
                                    ).reshape(-1)
                                    rms_values = [
                                        float(value)
                                        for value in raw_rms
                                        if math.isfinite(float(value))
                                    ]
                                except Exception as exc:  # noqa: BLE001 - RMS diagnostics should not abort
                                    click.echo(
                                        f"RMS evaluation failed for {candidate.parameter}: {exc}",
                                        err=True,
                                    )
                        status = (
                            "success"
                            if math.isfinite(final_loss)
                            and final_loss < _FAILED_CANDIDATE_LOSS
                            else "failed"
                        )
                        results.append(
                            CandidateResult(
                                parameter=candidate.parameter,
                                status=status,
                                loss=final_loss,
                                p_opt=record["p_opt"],
                                optimisation_loss=record["optimisation_loss"],
                                continuous_loss=continuous_loss,
                                impulse_loss=impulse_loss,
                                final_profile_losses=final_profile_losses,
                                rms=rms_values,
                                iterations=record.get("iterations"),
                                evaluations=record.get("evaluations"),
                                error=None if status == "success" else "candidate optimisation returned penalty loss",
                                coarse_grid_points=record.get("coarse_grid_points"),
                                coarse_best_p=record.get("coarse_best_p"),
                                coarse_best_loss=record.get("coarse_best_loss"),
                                optimisation_coarse_best_loss=record.get("coarse_best_loss"),
                                refinement_bracket=record.get("refinement_bracket"),
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - keep other candidates running
                        results.append(
                            CandidateResult(
                                parameter=candidate.parameter,
                                status="failed",
                                loss=None,
                                p_opt=None,
                                error=str(exc),
                            )
                        )
                return results
        finally:
            if mle is not None:
                with suppress(Exception):
                    mle.eval(
                        "if bdIsLoaded('simulink_model'), close_system('simulink_model', 0); end;",
                        nargout=0,
                    )
                if temp_path_added:
                    with suppress(Exception):
                        mle.eval(f"rmpath({_matlab_char_literal(temp_root)});", nargout=0)
            os.chdir(original_cwd)
    finally:
        if not debug_temp:
            shutil.rmtree(temp_root, ignore_errors=True)


def run_for_model(
    *,
    model_id: str,
    run_id: str,
    output: Optional[Path] = None,
    min_baseline_improvement: float = 0.25,
    min_next_best_separation: float = 0.05,
    max_iter: Optional[int] = None,
    tol_x: Optional[float] = None,
    coarse_grid_points: Optional[int] = 9,
    signals: Optional[Sequence[str]] = None,
    countinous_only: bool = False,
    ignore_impulse_signals: bool = False,
    noise: Optional[str] = None,
    noise_seed: Optional[int] = None,
    matlab_workers: Optional[int] = None,
    matlab_engine: Any = None,
    debug_temp: bool = False,
    optimizer_fn: OptimizerFn = _run_matlab_candidate_optimisations,
) -> Dict[str, Any]:
    model_name = str(model_id or "").strip()
    if not model_name or "/" in model_name or "\\" in model_name:
        raise ValueError(f"Expected a model id under models/simulink, got {model_id!r}")
    model_dir = _resolve_models_root() / model_name
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")

    specs = load_model_run_specs_json(
        model_dir / "model_run_specs.json",
        enforce_baseline_pair_diversity=False,
    )
    runtime_record = load_model_record_json(resolve_model_record_path(model_dir))
    experiment_config = load_experiment_config_json(model_dir / "experiment_config.json")
    metadata_model = load_simulink_generated_metadata(model_dir / "generated" / "metadata.json")
    metadata = metadata_model.model_dump(mode="python")

    context = _find_child_context(
        run_id=run_id,
        specs=specs,
        runtime_record=runtime_record,
    )
    runs_root = resolve_runs_root(model_dir)
    run_model_path = runs_root / context.run_id / "simulink_model.mdl"
    if not run_model_path.exists():
        raise ValueError(f"Missing child run model snapshot: {run_model_path}")

    selected_signals = (
        _normalise_requested_signals(signals)
        if signals is not None
        else _observable_signals_for_parameter(
            experiment_config,
            context.ground_truth_parameter,
        )
    )
    observable_signals, ignored_observable_signals = _observable_signal_selection(
        experiment_config,
        selected_signals=selected_signals,
        ignore_impulse_signals=ignore_impulse_signals,
    )
    baseline_df = _require_run_df(runs_root, context.baseline_run_id)
    clean_child_df = _require_run_df(runs_root, context.run_id)
    time0_df: Optional[pd.DataFrame] = None
    if context.time0_baseline_run_id:
        time0_df = _require_run_df(runs_root, context.time0_baseline_run_id)

    noise_profile = normalize_noise_profile(noise)
    resolved_noise_seed = int(noise_seed) if noise_seed is not None else _resolve_noise_seed(context)
    target_dfs_by_profile, noise_analysis_by_profile = _target_dataframes_for_noise_profiles(
        clean_child_df,
        baseline_df=baseline_df,
        model_id=model_name,
        models_root=model_dir.parent,
        noise_seed=resolved_noise_seed,
    )
    child_df = target_dfs_by_profile[noise_profile]
    noise_analysis = noise_analysis_by_profile[noise_profile]
    resolved_matlab_workers = _resolve_matlab_workers(matlab_workers)

    signal_types = _observable_signal_types_for_signals(
        experiment_config,
        observable_signals,
    )
    optimisation_signals = (
        _continuous_signals_from_types(observable_signals, signal_types)
        if countinous_only
        else observable_signals
    )
    if not optimisation_signals:
        if countinous_only:
            raise ValueError("No continuous observable signals available for optimisation search")
        raise ValueError("No observable signals available for optimisation search")
    signal_scales = _signal_loss_scales(
        reference_df=child_df,
        baseline_df=baseline_df,
        observable_signals=observable_signals,
        intervention_time=context.intervention_time,
    )
    final_profile_signal_scales = {
        profile: _signal_loss_scales(
            reference_df=profile_df,
            baseline_df=baseline_df,
            observable_signals=observable_signals,
            intervention_time=context.intervention_time,
        )
        for profile, profile_df in target_dfs_by_profile.items()
    }
    optimisation_signal_types = {
        signal: signal_types[signal]
        for signal in optimisation_signals
    }
    optimisation_signal_scales = {
        signal: signal_scales[signal]
        for signal in optimisation_signals
    }
    optimisation_baseline_signal_losses = _trajectory_signal_losses(
        reference_df=child_df,
        candidate_df=baseline_df,
        observable_signals=optimisation_signals,
        intervention_time=context.intervention_time,
        signal_types=optimisation_signal_types,
        signal_scales=optimisation_signal_scales,
        model_id=model_name,
    )
    baseline_signal_losses = _trajectory_signal_losses(
        reference_df=child_df,
        candidate_df=baseline_df,
        observable_signals=observable_signals,
        intervention_time=context.intervention_time,
        signal_types=signal_types,
        signal_scales=signal_scales,
        model_id=model_name,
    )
    final_profile_baseline_signal_losses = {
        profile: _trajectory_signal_losses(
            reference_df=profile_df,
            candidate_df=baseline_df,
            observable_signals=observable_signals,
            intervention_time=context.intervention_time,
            signal_types=signal_types,
            signal_scales=final_profile_signal_scales[profile],
            model_id=model_name,
        )
        for profile, profile_df in target_dfs_by_profile.items()
    }
    optimisation_baseline_mse = _trajectory_mse(
        reference_df=child_df,
        candidate_df=baseline_df,
        observable_signals=optimisation_signals,
        intervention_time=context.intervention_time,
        signal_types=optimisation_signal_types,
        signal_scales=optimisation_signal_scales,
        model_id=model_name,
        baseline_signal_losses=optimisation_baseline_signal_losses,
    )
    baseline_mse = _trajectory_mse(
        reference_df=child_df,
        candidate_df=baseline_df,
        observable_signals=observable_signals,
        intervention_time=context.intervention_time,
        signal_types=signal_types,
        signal_scales=signal_scales,
        model_id=model_name,
        baseline_signal_losses=baseline_signal_losses,
    )
    candidates = _candidate_specs_from_config_and_metadata(
        experiment_config=experiment_config,
        metadata=metadata,
    )
    baseline_block_bindings = _baseline_block_bindings_from_metadata(
        metadata=metadata,
        baseline_parameters=context.baseline_parameters,
    )
    simulink_signals_available = set(metadata.get("simulink_signals_available") or [])
    candidate_results = optimizer_fn(
        model_dir=model_dir,
        run_model_path=run_model_path,
        target_df=child_df,
        time0_df=time0_df,
        context=context,
        observable_signals=optimisation_signals,
        candidates=candidates,
        baseline_block_bindings=baseline_block_bindings,
        max_iter=max_iter,
        tol_x=tol_x,
        coarse_grid_points=coarse_grid_points,
        ignore_impulse_signals=ignore_impulse_signals,
        debug_temp=debug_temp,
        signal_types=optimisation_signal_types,
        signal_scales=optimisation_signal_scales,
        baseline_signal_losses=optimisation_baseline_signal_losses,
        final_loss_signals=observable_signals,
        final_signal_types=signal_types,
        final_signal_scales=signal_scales,
        final_profile_targets=target_dfs_by_profile,
        final_profile_signal_scales=final_profile_signal_scales,
        final_profile_baseline_signal_losses=final_profile_baseline_signal_losses,
        active_final_profile=noise_profile,
        matlab_workers=resolved_matlab_workers,
        matlab_engine=matlab_engine,
        final_simulink_signal_names=_simulink_signal_names_for_loss(
            model_id=model_name,
            signals=observable_signals,
            simulink_signals_available=simulink_signals_available,
        ),
        simulink_signal_names=_simulink_signal_names_for_loss(
            model_id=model_name,
            signals=optimisation_signals,
            simulink_signals_available=simulink_signals_available,
        ),
    )
    if not any(result.status == "success" for result in candidate_results):
        raise RuntimeError("All parameter optimisation candidates failed")

    report = _build_verdict(
        model_id=model_name,
        context=context,
        baseline_mse=baseline_mse,
        candidate_results=candidate_results,
        min_baseline_improvement=min_baseline_improvement,
        min_next_best_separation=min_next_best_separation,
        optimizer=_optimizer_name(coarse_grid_points),
        ignore_impulse_signals=ignore_impulse_signals,
        observable_signals_used=observable_signals,
        observable_signals_ignored=ignored_observable_signals,
        optimisation_signals_used=optimisation_signals,
        optimisation_baseline_mse=optimisation_baseline_mse,
        baseline_signal_losses=baseline_signal_losses,
        noise_profile=noise_profile,
        noise_seed=resolved_noise_seed,
        noise_analysis=noise_analysis,
    )
    documented_report = _documented_oracle_report(report)
    out_path = Path(output) if output is not None else runs_root / context.run_id / REPORT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(documented_report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": True,
        "path": str(out_path),
        "report": report,
        "documented_report": documented_report,
    }


@click.command()
@click.option("--model", required=True, type=str, help="Model id under models/simulink/.")
@click.option("--run-id", required=True, type=str, help="Successful child intervention run id.")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Optional output JSON path.")
@click.option("--min-baseline-improvement", type=float, default=0.25, show_default=True)
@click.option("--min-next-best-separation", type=float, default=0.05, show_default=True)
@click.option("--max-iter", type=int, default=None, help="Optional MATLAB fminbnd MaxIter.")
@click.option(
    "--matlab-workers",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "MATLAB parallel pool workers. Defaults to TSENV_MATLAB_WORKERS or "
        f"{_DEFAULT_MATLAB_WORKERS}; use 0 to disable explicit pool startup."
    ),
)
@click.option(
    "--coarse-grid-points",
    type=int,
    default=9,
    show_default=True,
    help="Number of parsim coarse-grid points before fminbnd refinement; use 1 to disable.",
)
@click.option(
    "--signals",
    multiple=True,
    help="Observable signal to include in the loss; repeat for multiple signals.",
)
@click.option(
    "--countinous-only",
    "--continuous-only",
    "countinous_only",
    is_flag=True,
    default=False,
    help="Use only continuous observable signals for coarse grid and fminbnd.",
)
@click.option(
    "--noise",
    type=click.Choice(["LOW", "HIGH"], case_sensitive=False),
    default=None,
    help="Add the model-specific LOW or HIGH noise profile before optimisation.",
)
def cli(
    model: str,
    run_id: str,
    output: Optional[Path],
    min_baseline_improvement: float,
    min_next_best_separation: float,
    max_iter: Optional[int],
    matlab_workers: Optional[int],
    coarse_grid_points: int,
    signals: tuple[str, ...],
    countinous_only: bool,
    noise: Optional[str],
) -> None:
    """Compute the parameter optimisation diagnostic for one child run."""
    click.echo(f"starting parameter optimisation for run {run_id}")
    try:
        result = run_for_model(
            model_id=model,
            run_id=run_id,
            output=output,
            min_baseline_improvement=min_baseline_improvement,
            min_next_best_separation=min_next_best_separation,
            max_iter=max_iter,
            matlab_workers=matlab_workers,
            coarse_grid_points=coarse_grid_points,
            signals=signals or None,
            countinous_only=countinous_only,
            noise=noise,
        )
    except Exception as exc:  # noqa: BLE001 - Click entrypoint should report cleanly
        raise click.ClickException(str(exc)) from exc
    report = result["report"]
    click.echo(
        f"{model}/{run_id}: best={report['best_parameter']} "
        f"truth={report['ground_truth_parameter']} "
        f"rank={report['rank']} detectable={report['optimisation_detectable']}"
    )
    click.echo(f"wrote {result['path']}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
