"""Shared helpers for JSON serialization across the project."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

try:  # Avoid circular import during type checking
    from .intervention_sampling import InitialSamplingState
except Exception:  # pragma: no cover - fallback when module unavailable
    InitialSamplingState = None  # type: ignore


def serialize_initial_state(state: "InitialSamplingState") -> Dict[str, Any]:
    """Convert an InitialSamplingState into primitives safe for JSON dumps."""

    return {
        "variable_values": dict(state.variable_values),
        "parameter_values": {
            f"{path}::{param}": val for (path, param), val in state.parameter_values.items()
        },
        "variables_to_parameters": {
            var: [list(pair) for pair in pairs]
            for var, pairs in state.variables_to_parameters.items()
        },
        "parameter_expressions": {
            f"{path}::{param}": expr
            for (path, param), expr in state.parameter_expressions.items()
        },
    }


def to_serializable(value: Any) -> Any:
    """Best-effort conversion of common project types into JSON primitives."""

    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.to_list()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, Path):
        return str(value)
    if InitialSamplingState is not None and isinstance(value, InitialSamplingState):
        return serialize_initial_state(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_safe_raw_questions(obj: Any) -> Any:
    """Recursively convert tuples/sets and numpy scalars to JSON-safe types."""

    if isinstance(obj, Mapping):
        return {k: json_safe_raw_questions(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [json_safe_raw_questions(v) for v in obj]

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()

    return obj


# def serialize_raw_question_results(
#     raw_question_results: Optional[Mapping[str, Any]],
# ) -> Optional[Dict[str, Any]]:
#     """Best-effort JSON-serializable copy of raw question results."""

#     if not raw_question_results or not isinstance(raw_question_results, Mapping):
#         return None

#     return json_safe_raw_questions(dict(raw_question_results))
