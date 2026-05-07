"""Helpers for sampling interventions and initial values from metadata.

This module centralises the logic that used to live inside
``make_data_recipes.py`` so that both the data-generation pipeline and the
simulation runner can reuse the same routines.
"""

from __future__ import annotations

import ast
import logging
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .utils import short_block_name


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD_TUNING: Dict[str, float] = {
    "min_abs_change": 1e-4,
    "min_range_fraction": 0.25,
    "min_rel_change": 0.5,
    "min_ratio_change": 1.8,
    "max_sampling_attempts": 10000,
}


# ---------------------------------------------------------------------------
# Expression helpers
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS = {
    "Uniform": lambda a, b: float(np.random.uniform(float(a), float(b))),
    "LogUniform": lambda a, b: float(
        np.exp(np.random.uniform(np.log(float(a)), np.log(float(b))))
    ),
    "abs": abs,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "pow": pow,
    "cos": math.cos,
    "cosd": lambda x: math.cos(math.radians(float(x))),
    "sin": math.sin,
    "sind": lambda x: math.sin(math.radians(float(x))),
    "tan": math.tan,
    "tand": lambda x: math.tan(math.radians(float(x))),
}

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}

_UOPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def _normalize_numeric_expr(expr: str) -> str:
    """Normalize common MATLAB operators into Python-equivalent syntax."""
    normalized = str(expr)
    normalized = normalized.replace(".^", "**")
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace(".*", "*")
    normalized = normalized.replace("./", "/")
    normalized = normalized.replace("~=", "!=")
    normalized = normalized.replace("&&", " and ")
    normalized = normalized.replace("||", " or ")
    return normalized


def _safe_eval_expr(node: ast.AST, names: Mapping[str, float]) -> float:
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator {type(node.op).__name__}")
        return float(op(_safe_eval_expr(node.left, names), _safe_eval_expr(node.right, names)))

    if isinstance(node, ast.UnaryOp):
        op = _UOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator {type(node.op).__name__}")
        return float(op(_safe_eval_expr(node.operand, names)))

    if isinstance(node, ast.Name):
        if node.id in names:
            return float(names[node.id])
        if node.id == "pi":
            return float(math.pi)
        raise KeyError(f"Unknown variable '{node.id}' in expression")

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed in expressions")

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed in expressions")
        func = _ALLOWED_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError(f"Function '{node.func.id}' is not permitted in expressions")
        args = [_safe_eval_expr(arg, names) for arg in node.args]
        return float(func(*args))

    if isinstance(node, ast.Compare):
        left = _safe_eval_expr(node.left, names)
        result = True
        for op_node, comparator in zip(node.ops, node.comparators):
            right = _safe_eval_expr(comparator, names)
            if isinstance(op_node, ast.Gt):
                ok = left > right
            elif isinstance(op_node, ast.Ge):
                ok = left >= right
            elif isinstance(op_node, ast.Lt):
                ok = left < right
            elif isinstance(op_node, ast.Le):
                ok = left <= right
            elif isinstance(op_node, ast.Eq):
                ok = abs(left - right) <= 1e-12
            elif isinstance(op_node, ast.NotEq):
                ok = abs(left - right) > 1e-12
            else:
                raise ValueError(f"Unsupported comparison operator {type(op_node).__name__}")
            result = result and ok
            left = right
        return 1.0 if result else 0.0

    if isinstance(node, ast.BoolOp):
        values = [_safe_eval_expr(v, names) for v in node.values]
        if isinstance(node.op, ast.And):
            return 1.0 if all(v != 0.0 for v in values) else 0.0
        if isinstance(node.op, ast.Or):
            return 1.0 if any(v != 0.0 for v in values) else 0.0
        raise ValueError(f"Unsupported boolean operator {type(node.op).__name__}")

    raise ValueError(f"Unsupported AST node {type(node).__name__}")


def eval_numeric_expr(expr: str, perturbation_values: Mapping[str, float]) -> float:
    tree = ast.parse(_normalize_numeric_expr(expr), mode="eval")
    return float(_safe_eval_expr(tree.body, perturbation_values))


# ---------------------------------------------------------------------------
# Sampling spec helpers
# ---------------------------------------------------------------------------


def get_interval_sampling(spec: Mapping[str, Any]) -> Tuple[Any, Any]:
    interval = spec.get("interval")
    if not isinstance(interval, Sequence) or len(interval) != 2:
        raise KeyError("Sampling spec missing a two-element 'interval'")
    return interval[0], interval[1]


def _sample_uniform_with_gap(
    lo: float,
    hi: float,
    center: Optional[float],
    min_gap: float,
) -> float:
    """Uniform on [lo, hi] but at least min_gap away from center (if provided)."""
    if center is None or min_gap <= 0:
        return float(np.random.uniform(lo, hi))

    left: Tuple[float, float] = (lo, min(center - min_gap, hi))
    right: Tuple[float, float] = (max(center + min_gap, lo), hi)

    intervals: List[Tuple[float, float]] = []
    if left[1] > left[0]:
        intervals.append(left)
    if right[1] > right[0]:
        intervals.append(right)

    if not intervals:
        raise ValueError("Not possible to sample!")

    lengths = [b - a for (a, b) in intervals]
    total_len = sum(lengths)

    u = np.random.uniform(0.0, total_len)
    for (a, b), length in zip(intervals, lengths):
        if u <= length:
            return float(a + u)
        u -= length

    a, b = intervals[-1]
    return float(np.random.uniform(a, b))


def _sample_loguniform_with_gap(
    lo: float,
    hi: float,
    center: Optional[float],
    min_log_gap: float,
) -> float:
    """Log-uniform on (lo, hi) but at least min_log_gap away from log(center)."""
    if lo <= 0 or hi <= 0:
        raise ValueError(f"loguniform bounds must be positive (got {lo}, {hi})")

    log_lo = np.log(lo)
    log_hi = np.log(hi)

    if center is None or min_log_gap <= 0:
        return float(np.exp(np.random.uniform(log_lo, log_hi)))

    log_c = np.log(center)

    left = (log_lo, min(log_c - min_log_gap, log_hi))
    right = (max(log_c + min_log_gap, log_lo), log_hi)

    intervals: List[Tuple[float, float]] = []
    if left[1] > left[0]:
        intervals.append(left)
    if right[1] > right[0]:
        intervals.append(right)

    if not intervals:
        return float(np.exp(np.random.uniform(log_lo, log_hi)))

    lengths = [b - a for (a, b) in intervals]
    total_len = sum(lengths)

    u = np.random.uniform(0.0, total_len)
    for (a, b), length in zip(intervals, lengths):
        if u <= length:
            return float(np.exp(a + u))
        u -= length

    a, b = intervals[-1]
    return float(np.exp(np.random.uniform(a, b)))



def sample_from_spec(spec: Mapping[str, Any],  center=None, min_distance=0.3, min_log_distance=0.3 ) -> float:
    kind = (spec.get("type") or "uniform").lower()
    lo, hi = get_interval_sampling(spec)

    if kind == "uniform":
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            return float(lo)
        return _sample_uniform_with_gap(
            lo,
            hi,
            center=center,
            min_gap=min_distance,
        )

    elif kind == "loguniform":
        if hi < lo:
            lo, hi = hi, lo
        return _sample_loguniform_with_gap(
            lo,
            hi,
            center=center,
            min_log_gap=min_log_distance,
        )

    else:
        raise NotImplementedError()


    # if kind == "derived":
    #     expr = spec.get("expr", "")
    #     if not expr:
    #         raise ValueError("Derived sampling spec missing 'expr'")
    #     return eval_numeric_expr(expr, env)

    # value = spec.get("value")
    # if value is not None:
    #     return float(value)

    # raise ValueError(f"Unsupported sampling spec kind '{kind}'")


# def _looks_like_sampling_spec(obj: Any) -> bool:
#     if not isinstance(obj, Mapping):
#         return False
#     keys = {"interval", "value", "expr", "type", "distribution"}
#     return any(key in obj for key in keys)


# def _block_variants(block_ident: str) -> List[str]:
#     variants: List[str] = []
#     tail = block_ident.split("/")[-1].replace("\n", " ").strip()
#     tail_snake = re.sub(r"\s+", "_", tail)
#     for candidate in (
#         block_ident,
#         short_block_name(block_ident),
#         tail,
#         tail_snake,
#         tail.lower(),
#         tail_snake.lower(),
#     ):
#         if candidate and candidate not in variants:
#             variants.append(candidate)
#     return variants


# def _find_sampling_spec_in_entry(
#     entry: Any,
#     block_ident: str,
#     param: str,
#     block_variants: Sequence[str],
#     *,
#     seen: Optional[set] = None,
# ) -> Optional[Mapping[str, Any]]:
#     if not isinstance(entry, Mapping):
#         return None

#     if _looks_like_sampling_spec(entry):
#         return entry

#     entry_id = id(entry)
#     seen = seen or set()
#     if entry_id in seen:
#         return None
#     seen.add(entry_id)

#     for key, value in entry.items():
#         if not isinstance(value, Mapping):
#             continue

#         if key in block_variants and _looks_like_sampling_spec(value):
#             return value

#         spec = _find_sampling_spec_in_entry(
#             value, block_ident, param, block_variants, seen=seen
#         )
#         if spec is not None:
#             return spec

#     return None


# def resolve_sampling_spec(
#     metadata: Mapping[str, Any],
#     section_key: str,
#     intervention_parameter: str,
#     block_ident: str,
#     param: str,
# ) -> Optional[Mapping[str, Any]]:
#     section = metadata.get(section_key)
#     if not isinstance(section, Mapping):
#         return None

#     block_variants = _block_variants(block_ident)

#     preferred = section.get(intervention_parameter)
#     if isinstance(preferred, Mapping):
#         spec = _find_sampling_spec_in_entry(preferred, block_ident, param, block_variants)
#         if spec is not None:
#             return spec

#     for entry in section.values():
#         if not isinstance(entry, Mapping):
#             continue
#         spec = _find_sampling_spec_in_entry(entry, block_ident, param, block_variants)
#         if spec is not None:
#             return spec

#     return None


# ---------------------------------------------------------------------------
# Intervention metadata collection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterventionTarget:
    path: str
    parameter: str
    expression: Optional[str]
    is_perfect_match: bool = False
    is_runtype: bool = False




def collect_intervention_targets(metadata: Mapping[str, Any]) -> Dict[str, List[InterventionTarget]]:
    targets: Dict[str, List[InterventionTarget]] = {}
    intervention_block_map = metadata.get("intervention_block_map") or {}
    active_parameters = set(metadata.get("parameter_set") or [])
    for intervention_parameter, info in intervention_block_map.items():
        if active_parameters and intervention_parameter not in active_parameters:
            continue
        bindings: List[InterventionTarget] = []
        for entry in info["parameters"]:
            path = entry.get("path")
            param = entry.get("name")
            if not path or not param:
                continue
            bindings.append(
                InterventionTarget(
                    path=path,
                    parameter=param,
                    expression=entry.get("expression"),
                    is_perfect_match=bool(entry.get("is_perfect_match")),
                    is_runtype=bool(entry["runtime_type"])
                )
            )
        if bindings:
            targets[intervention_parameter] = bindings
    return targets


# ---------------------------------------------------------------------------
# Initial value sampling
# ---------------------------------------------------------------------------


def quantize_to_solver_step(time_value: float, solver_step: float) -> float:
    if solver_step <= 0:
        return float(time_value)
    return float(round(time_value / solver_step) * solver_step)


def quantize_to_sampling_rate(time_value: float, sampling_rate_hz: float) -> float:
    if sampling_rate_hz <= 0:
        return float(time_value)
    dt = 1.0 / float(sampling_rate_hz)
    if not np.isfinite(dt) or dt <= 0:
        return float(time_value)
    return float(round(float(time_value) / dt) * dt)


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(fallback)


def _resolve_interval_bounds(spec: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    #try:
    lo, hi = get_interval_sampling(spec)
    # except (KeyError, ValueError, TypeError):
    #     return None, None
    # try:
    #     lo = _resolve_bound(raw_lo, env)
    #     hi = _resolve_bound(raw_hi, env)
    # except Exception:
    #     return None, None
    if hi < lo:
        lo, hi = hi, lo
    return float(lo), float(hi)


def sample_with_change_threshold(
    spec: Mapping[str, Any],
    *,
    initial_value: float,
    thresholds: Mapping[str, float],
    direction: Optional[str] = None,
) -> Optional[float]:
    kind = (spec.get("type") or "").lower()
    direction_norm = (direction or "").lower()
    if direction_norm not in {"increase", "decrease"}:
        direction_norm = None

    interval_lo, interval_hi = _resolve_interval_bounds(spec)


    min_abs = float(thresholds["min_abs_change"])

    min_fraction = float(thresholds["min_range_fraction"])

    candidate = sample_from_spec(spec, center=initial_value, min_distance=initial_value*min_fraction, min_log_distance=math.log(1.4))
    return candidate

@dataclass
class InitialSamplingState:
    variable_values: Dict[str, float]
    parameter_values: Dict[Tuple[str, str], float]
    variables_to_parameters: Dict[str, List[Tuple[str, str]]]
    parameter_expressions: Dict[Tuple[str, str], Optional[str]]
    sampling_type: Dict[Tuple[str, str], Optional[str]]
    runtime_type: Dict[Tuple[str,str], bool]
    





def sample_initial_state(
    metadata: Mapping[str, Any],
    *,
    initial_sampling_interval: Optional[Mapping[str, Any]] = None,
    max_tries: int = 500,
    use_default: bool = False,
) -> InitialSamplingState:
    intervention_block_map = metadata["intervention_block_map"]
    initial_interval = initial_sampling_interval or metadata.get("initial_sampling_interval") or {}
    if not isinstance(initial_interval, Mapping):
        raise TypeError("metadata.initial_sampling_interval must be a mapping")

    perturbation_values: Dict[str, float] = {}
    # Start by filling perturbation values 
    for intervention_parameter in intervention_block_map.keys():
        if use_default or intervention_parameter not in initial_interval:
            # Use default value 
            value = metadata["default_values"][intervention_parameter]
        else:
            #raise ValueError("Check this out")
            spec = initial_interval[intervention_parameter]
            value = float(sample_from_spec(spec))
        
        perturbation_values[intervention_parameter] = value


    parameter_values: Dict[Tuple[str, str], float] = {}
    variables_to_parameters: Dict[str, List[Tuple[str, str]]] = {}
    parameter_expressions: Dict[Tuple[str, str], Optional[str]] = {}
    sampling_type = {
        k: (v.get("type") if isinstance(v, Mapping) else None)
        for k, v in initial_interval.items()
    }
    runtime_type = {}
    for intervention_parameter, var_info in intervention_block_map.items():
        entries: List[Dict[str, Any]] = []
        for entry in var_info["parameters"]:
            path = entry.get("path")
            param = entry.get("name")
            if not path or not param:
                raise ValueError("Something fishy")
                continue
            entries.append(entry)

        if not entries:
            raise ValueError("Something fishy")
            continue
        
        variables_to_parameters[intervention_parameter] = [(e["path"], e["name"]) for e in entries]
        for entry in entries:
            path = entry["path"]
            parameter = entry["name"]
            parameter_expressions[(path, parameter)] = entry.get("expression")

            expression = entry["expression"]
      
            param_value = float(eval_numeric_expr(expression, perturbation_values))
   

            parameter_values[(path, parameter)] = param_value
            runtime_type[(path,parameter)] = entry["runtime_type"]

    return InitialSamplingState(
        perturbation_values,
        parameter_values,
        variables_to_parameters,
        parameter_expressions,
        sampling_type,
        runtime_type,
    )


def build_initial_state_from_values(
    metadata: Mapping[str, Any],
    variable_values: Mapping[str, float],
    *,
    fill_defaults: bool = True,
) -> InitialSamplingState:
    """Build an InitialSamplingState from provided variable values.

    This is a lightweight alternative to `sample_initial_state` for cases where
    variable values are already defined (e.g. loaded from a registry) and no
    random sampling is desired.
    """
    intervention_block_map = metadata["intervention_block_map"]
    defaults = metadata.get("default_values") or {}

    resolved_values: Dict[str, float] = {}
    for intervention_parameter in intervention_block_map.keys():
        if intervention_parameter in variable_values:
            resolved_values[intervention_parameter] = float(variable_values[intervention_parameter])
        elif fill_defaults and intervention_parameter in defaults:
            resolved_values[intervention_parameter] = float(defaults[intervention_parameter])
        else:
            raise KeyError(f"Missing initial value for {intervention_parameter}")

    # Preserve extra initial variables that are not part of intervention_block_map
    # (for example model-workspace-only variables such as BallDrop initial_velocity).
    for var_name, raw_value in variable_values.items():
        if var_name in resolved_values:
            continue
        resolved_values[str(var_name)] = float(raw_value)

    parameter_values: Dict[Tuple[str, str], float] = {}
    variables_to_parameters: Dict[str, List[Tuple[str, str]]] = {}
    parameter_expressions: Dict[Tuple[str, str], Optional[str]] = {}
    runtime_type: Dict[Tuple[str, str], bool] = {}

    for intervention_parameter, var_info in intervention_block_map.items():
        entries: List[Dict[str, Any]] = []
        for entry in var_info["parameters"]:
            path = entry.get("path")
            param = entry.get("name")
            if not path or not param:
                raise ValueError(f"Invalid parameter binding for {intervention_parameter}")
            entries.append(entry)

        if not entries:
            raise ValueError(f"No parameters configured for {intervention_parameter}")

        variables_to_parameters[intervention_parameter] = [(e["path"], e["name"]) for e in entries]
        for entry in entries:
            path = entry["path"]
            parameter = entry["name"]
            expression = entry.get("expression")
            if not expression:
                raise ValueError(f"Missing expression for {intervention_parameter} -> {path}::{parameter}")
            parameter_expressions[(path, parameter)] = expression
            parameter_values[(path, parameter)] = float(eval_numeric_expr(expression, resolved_values))
            runtime_type[(path, parameter)] = bool(entry.get("runtime_type"))

    return InitialSamplingState(
        resolved_values,
        parameter_values,
        variables_to_parameters,
        parameter_expressions,
        {},
        runtime_type,
    )


# ---------------------------------------------------------------------------
# Intervention value sampling
# ---------------------------------------------------------------------------


def sample_intervention_value(
    metadata: Mapping[str, Any],
    intervention_parameter: str,
    # env: Mapping[str, float],
    initial_value: float,
    thresholds: Optional[Mapping[str, float]] = None,
    direction: Optional[str] = None,
    max_attempts: int = 50,
    *,
    perturbation_sampling_interval: Optional[Mapping[str, Any]] = None,
) -> float:
    thresholds = thresholds or DEFAULT_THRESHOLD_TUNING
    sampling_interval = perturbation_sampling_interval or metadata.get("perturbation_sampling_interval") or {}
    if intervention_parameter not in sampling_interval:
        raise KeyError(
            f"Missing perturbation sampling interval for {intervention_parameter}. "
            "Provide `perturbation_sampling_interval=` or include "
            "`perturbation_sampling_interval` in metadata."
        )
    spec = sampling_interval[intervention_parameter]
    try:
        return sample_with_change_threshold(
            spec,
            # env,
            initial_value=initial_value,
            thresholds=thresholds,
            direction=direction,
        )
    except ValueError as exc:
        if str(exc) == "Not possible to sample!":
            lo, hi = _resolve_interval_bounds(spec)
            raise ValueError(
                f"Not possible to sample {intervention_parameter} "
                f"(base_value={initial_value}, lo={lo}, hi={hi})"
            ) from exc
        raise


def build_intervention_modifications(
    *,
    intervention_parameter: str,
    new_value: float,
    initial_state: InitialSamplingState,
    intervention_time: float,
    end_time: Optional[float],
    transition_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    modifications: List[Dict[str, Any]] = [] # Multiple modifications, one for each parameter
    time_key = "step" if transition_type is None else transition_type
    transition = time_key or "step"

    parameter_values = initial_state.parameter_values
    modified_state = initial_state.variable_values.copy()
    assert intervention_parameter in modified_state
    modified_state[intervention_parameter] = new_value
    for parameter in initial_state.variables_to_parameters[intervention_parameter]:
        default_value = float(parameter_values[parameter])

        param_value = eval_numeric_expr(initial_state.parameter_expressions[parameter], modified_state)

        if intervention_time > 0 and not initial_state.runtime_type[parameter]:
            continue

        modifications.append(
            {
                "identifier": parameter[0],
                "key": parameter[1],
                "parameter_id": f"{parameter[0]}::{parameter[1]}",
                "transition_type": transition,
                "start_time": float(intervention_time),
                "end_time": None if transition == "step" else float(end_time) if end_time is not None else None,
                "old_value": float(default_value if default_value is not None else 0.0),
                "new_value": float(param_value),
                "intervention_parameter": intervention_parameter,
                "is_runtime_type_parameter": initial_state.runtime_type[parameter],
            }
        )
        # base_env[f"{parameter[0]}::{parameter[1]}"] = float(param_value)

    return modifications


# ---------------------------------------------------------------------------
# Choices / prompts helpers
# ---------------------------------------------------------------------------


def build_intervention_variable_candidates(
    metadata: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    intervention_block_map = metadata["intervention_block_map"]
    candidates: List[Dict[str, Any]] = []

    normalized_qtypes: Dict[str, List[str]] = {}
    for intervention_parameter, info in intervention_block_map.items():
        if not info or not info.get("is_multiple_choice_question", True):
            continue
        
        # qtypes = normalized_qtypes.get(intervention_parameter, [])
        # if "increase" in qtypes or "increase_decrease" in qtypes:
        #     candidates.append(
        #         {
        #             "intervention_parameter": intervention_parameter,
        #             "question_type": "increase",
        #             "direction": "increase",
        #             "prompt": f"The '{intervention_parameter}' parameter increases.",
        #         }
        #     )
        # if "decrease" in qtypes or "increase_decrease" in qtypes:
        #     candidates.append(
        #         {
        #             "intervention_parameter": intervention_parameter,
        #             "question_type": "decrease",
        #             "direction": "decrease",
        #             "prompt": f"The '{intervention_parameter}' parameter decreases.",
        #         }
        #     )

        # if not qtypes:
        candidates.append(
            {
                "intervention_parameter": intervention_parameter,
                "question_type": "changes",
                "direction": None,
                "prompt": f"The '{intervention_parameter}' parameter changes.",
            }
        )
    return candidates


__all__ = [
    "DEFAULT_THRESHOLD_TUNING",
    "InitialSamplingState",
    "InterventionTarget",
    "build_intervention_modifications",
    # "build_parameter_map_from_state",
    "build_intervention_variable_candidates",
    "collect_intervention_targets",
    "sample_initial_state",
    "quantize_to_solver_step",
    "sample_intervention_value",
    "sample_from_spec",
    #"resolve_sampling_spec",
]
