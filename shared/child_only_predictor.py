from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Callable, Optional

from shared.child_only_cli_utils import normalize_run_id
from shared.child_only_physics_inference import infer_changed_parameter_child_only
from shared.model_noise_adder import apply_model_noise_profile, normalize_noise_profile
from shared.model_intervention_interface import load_allowed_interventions
from shared.tsenv_combinations import TIME0_BASELINE_AGENT_FACING_LABEL

_NO_CHANGE_KEYS = {"no_parameter_change", "nothing_happened"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_models_root(models_root: Optional[Path]) -> Path:
    if models_root is not None:
        return Path(models_root).expanduser().resolve()
    return _repo_root() / "models" / "simulink"


def _to_finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


@functools.lru_cache(maxsize=None)
def _load_agent_facing_parameter_map(
    *,
    model_id: str,
    models_root: Optional[Path] = None,
) -> dict[str, str]:
    resolved_models_root = _resolve_models_root(models_root)
    path = resolved_models_root / str(model_id) / "description_levels.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    raw_mapping = payload.get("internal_naming_to_agent_facing_parameter")
    if not isinstance(raw_mapping, dict):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in raw_mapping.items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "").strip()
        if key and value:
            out[key] = value
    return out


def parameter_key_to_display_label(
    *,
    model_id: str,
    parameter_key: object,
    models_root: Optional[Path] = None,
) -> Optional[str]:
    raw_key = str(parameter_key or "").strip()
    if not raw_key:
        return None
    if raw_key in _NO_CHANGE_KEYS:
        return TIME0_BASELINE_AGENT_FACING_LABEL
    mapping = _load_agent_facing_parameter_map(
        model_id=model_id,
        models_root=models_root,
    )
    return str(mapping.get(raw_key) or raw_key)


def canonical_prediction_payload(
    *,
    model_id: str,
    predicted_parameter: object,
    change_time: object,
    models_root: Optional[Path] = None,
) -> dict[str, object]:
    parsed_time = _to_finite_float(change_time)
    final_answer = parameter_key_to_display_label(
        model_id=model_id,
        parameter_key=predicted_parameter,
        models_root=models_root,
    )
    return {
        "change_time": parsed_time,
        "final_answer": final_answer,
    }


def make_predictor(
    model_id: str,
    *,
    v_min: float = 0.01,
    ball_drop_inference_version: Optional[str] = None,
    noise_profile: str = "none",
    noise_seed: int = 0,
    run_id: Optional[str] = None,
    run_token: Optional[str] = None,
    models_root: Optional[Path] = None,
) -> Callable[[Any], dict[str, object]]:
    resolved_models_root = _resolve_models_root(models_root)
    allowed_params = set(
        load_allowed_interventions(
            model_id=model_id,
            models_root=resolved_models_root,
        )
    )
    resolved_noise_profile = normalize_noise_profile(noise_profile)
    normalized_run_id = normalize_run_id(run_id)
    _ = str(run_token or "").strip() or str(normalized_run_id or "predict")

    def predict(df: Any) -> dict[str, object]:
        child_df = apply_model_noise_profile(
            df,
            model_id=model_id,
            models_root=resolved_models_root,
            noise_profile=resolved_noise_profile,
            noise_seed=int(noise_seed),
        )

        kwargs: dict[str, object] = {
            "model_id": model_id,
            "child_df": child_df,
            "run_id": normalized_run_id,
        }
        if model_id == "DampedMassBetweenWalls":
            kwargs["v_min"] = float(v_min)
        if model_id == "BallDrop" and ball_drop_inference_version is not None:
            kwargs["ball_drop_inference_version"] = str(ball_drop_inference_version)

        result = infer_changed_parameter_child_only(**kwargs)
        predicted_parameter = result.predicted_parameter
        if (
            predicted_parameter is not None
            and str(predicted_parameter).strip()
            and str(predicted_parameter) not in allowed_params
            and str(predicted_parameter) not in _NO_CHANGE_KEYS
        ):
            raise ValueError(
                f"Predicted parameter '{predicted_parameter}' is outside allowed interface "
                f"for model '{model_id}'."
            )
        return canonical_prediction_payload(
            model_id=model_id,
            predicted_parameter=predicted_parameter,
            change_time=result.intervention_time,
            models_root=resolved_models_root,
        )

    return predict


__all__ = [
    "canonical_prediction_payload",
    "make_predictor",
    "parameter_key_to_display_label",
]
