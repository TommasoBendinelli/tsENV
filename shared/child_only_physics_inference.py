from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from shared.child_only_inference_common import load_child_df

@dataclass(frozen=True)
class ChildOnlyInferenceResult:
    model_id: str
    predicted_parameter: Optional[str]
    intervention_time: float
    parameter_pre_value: Optional[Any] = None
    parameter_post_value: Optional[Any] = None
    pre_parameters: Dict[str, Any] = field(default_factory=dict)
    post_parameters: Dict[str, Any] = field(default_factory=dict)


# Global thresholds used by child-only inference models.
# They are overridable from CLI by mutating these module globals.
CHANGE_THRESHOLD_REL = 0.05
RAW_SIGNALS_CHANGE_THRESHOLD_REL = 0.03
MINIMUM_ABS_DELTA = 0.01


def _to_result(payload: dict[str, Any], *, default_model_id: str) -> ChildOnlyInferenceResult:
    return ChildOnlyInferenceResult(
        model_id=str(payload.get("model_id", default_model_id)),
        predicted_parameter=payload.get("predicted_parameter"),
        intervention_time=float(payload.get("intervention_time", np.nan)),
        parameter_pre_value=payload.get("parameter_pre_value"),
        parameter_post_value=payload.get("parameter_post_value"),
        pre_parameters=dict(payload.get("pre_parameters", {}) or {}),
        post_parameters=dict(payload.get("post_parameters", {}) or {}),
    )


def _prediction_to_result(
    predicted_parameter: object,
    *,
    default_model_id: str,
) -> ChildOnlyInferenceResult:
    predicted = str(predicted_parameter or "").strip() or None
    return ChildOnlyInferenceResult(
        model_id=default_model_id,
        predicted_parameter=predicted,
        intervention_time=float("nan"),
    )


def _infer_inclined_plane(
    *,
    child_df: pd.DataFrame,
    run_id: Optional[str] = None,
) -> ChildOnlyInferenceResult:
    _ = run_id
    from models.simulink.InclinedPlane import basic_rule as local

    predicted_parameter = local.predict(child_df)
    return _prediction_to_result(
        predicted_parameter,
        default_model_id="InclinedPlane",
    )


def _infer_ball_drop(
    *,
    child_df: pd.DataFrame,
    run_id: Optional[str] = None,
    ball_drop_inference_version: Optional[str] = None,
) -> ChildOnlyInferenceResult:
    _ = run_id, ball_drop_inference_version
    from models.simulink.BallDrop import basic_rule as local

    predicted_parameter = local.predict(child_df)
    return _prediction_to_result(
        predicted_parameter,
        default_model_id="BallDrop",
    )


def _infer_damped_mass_between_walls(
    *,
    child_df: pd.DataFrame,
    run_id: Optional[str] = None,
    v_min: float = 1.0,
) -> ChildOnlyInferenceResult:
    _ = run_id, v_min
    from models.simulink.DampedMassBetweenWalls import basic_rule as local

    predicted_parameter = local.predict(child_df)
    return _prediction_to_result(
        predicted_parameter,
        default_model_id="DampedMassBetweenWalls",
    )


def _infer_mass_spring_damper_with_pid(
    *,
    child_df: pd.DataFrame,
    run_id: Optional[str] = None,
) -> ChildOnlyInferenceResult:
    _ = run_id
    from models.simulink.MassSpringDamperWithPID import basic_rule as local

    predicted_parameter = local.predict(child_df)
    return _prediction_to_result(
        predicted_parameter,
        default_model_id="MassSpringDamperWithPID",
    )


def _infer_double_mass_spring_damper_same_coeffs(
    *,
    child_df: pd.DataFrame,
    run_id: Optional[str] = None,
) -> ChildOnlyInferenceResult:
    _ = run_id
    from models.simulink.DoubleMassSpringDamperSameCoeffs import basic_rule as local

    predicted_parameter = local.predict(child_df)
    return _prediction_to_result(
        predicted_parameter,
        default_model_id="DoubleMassSpringDamperSameCoeffs",
    )


def _infer_transmission_line(
    *,
    child_df: pd.DataFrame,
    run_id: Optional[str] = None,
) -> ChildOnlyInferenceResult:
    from models.simulink.TransmissionLine import inference as local

    payload = local.infer_changed_parameter_child_only(
        child_df=child_df,
        run_id=run_id,
        change_threshold_rel=float(CHANGE_THRESHOLD_REL),
    )
    return _to_result(payload, default_model_id="TransmissionLine")


_INFERERS = {
    "InclinedPlane": _infer_inclined_plane,
    "BallDrop": _infer_ball_drop,
    "MassSpringDamperWithPID": _infer_mass_spring_damper_with_pid,
    "DampedMassBetweenWalls": _infer_damped_mass_between_walls,
    "DoubleMassSpringDamperSameCoeffs": _infer_double_mass_spring_damper_same_coeffs,
}


def infer_changed_parameter_child_only(
    *,
    model_id: str,
    child_df: pd.DataFrame,
    run_id: Optional[str] = None,
    v_min: Optional[float] = None,
    ball_drop_inference_version: Optional[str] = None,
) -> ChildOnlyInferenceResult:
    model = str(model_id)
    inferer = _INFERERS.get(model)
    if inferer is None:
        known = ", ".join(sorted(_INFERERS.keys()))
        raise ValueError(f"Unsupported model_id '{model_id}'. Known models: {known}")

    if model == "DampedMassBetweenWalls":
        if v_min is None:
            raise ValueError("v_min is required for DampedMassBetweenWalls inference")
        result = _infer_damped_mass_between_walls(
            child_df=child_df,
            run_id=run_id,
            v_min=float(v_min),
        )
    elif model == "BallDrop":
        result = _infer_ball_drop(
            child_df=child_df,
            run_id=run_id,
            ball_drop_inference_version=ball_drop_inference_version,
        )
    else:
        result = inferer(
            child_df=child_df,
            run_id=run_id,
        )
    return result
