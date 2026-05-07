from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import click

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.child_only_cli_utils import normalize_run_id
from shared.child_only_physics_inference import infer_changed_parameter_child_only
from shared.child_only_predictor import (
    canonical_prediction_payload,
    parameter_key_to_display_label,
)
from shared.interface.model_record_json import (
    load_model_record_json,
    load_model_run_specs_json,
)
from shared.interface.distribution_json import load_experiment_config_json
from shared.model_run_specs_runtime import (
    build_model_record_registry,
)
from shared.model_noise_adder import apply_model_noise_profile, normalize_noise_profile
from shared.model_intervention_interface import load_allowed_interventions
from shared.question_eligibility import is_question_eligible
from shared.run_artifacts import resolve_model_record_path, resolve_runs_root

_RULE_WITH_NO_NOISE_NOT_IMPLEMENTED = "not_implemented"
_NO_CHANGE_KEYS = {"no_parameter_change", "nothing_happened"}


def _normalize_display_label(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _to_finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _build_display_label_to_parameter_key_map(
    *,
    model_id: str,
    models_root: Path,
    allowed_params: set[str],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for parameter_key in sorted(str(p).strip() for p in allowed_params if str(p).strip()):
        display_label = parameter_key_to_display_label(
            model_id=model_id,
            parameter_key=parameter_key,
            models_root=models_root,
        )
        normalized = _normalize_display_label(display_label)
        if not normalized:
            continue
        existing = out.get(normalized)
        if existing is not None and existing != parameter_key:
            raise click.ClickException(
                f"Ambiguous display label '{display_label}' for model '{model_id}': "
                f"maps to both '{existing}' and '{parameter_key}'."
            )
        out[normalized] = str(parameter_key)
    return out


def _allowed_display_labels_for_model(
    *,
    model_id: str,
    models_root: Path,
    allowed_params: set[str],
) -> list[str]:
    labels: set[str] = set()
    for parameter_key in sorted(str(p).strip() for p in allowed_params if str(p).strip()):
        display_label = parameter_key_to_display_label(
            model_id=model_id,
            parameter_key=parameter_key,
            models_root=models_root,
        )
        normalized = str(display_label or "").strip()
        if normalized:
            labels.add(normalized)
    return sorted(labels)


def _load_rule_predict_fn(rule_path: Path) -> Callable[[Any], Any]:
    resolved = Path(rule_path).expanduser().resolve()
    if not resolved.exists():
        raise click.ClickException(f"--rule_path not found: {resolved}")
    spec = importlib.util.spec_from_file_location("infer_verify_rule_module", resolved)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Unable to import rule module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise click.ClickException(f"{resolved} must define a callable predict(input).")
    return predict


def _normalize_rule_prediction_labels(
    value: Any,
    *,
    context: str,
) -> list[str]:
    if isinstance(value, str):
        label = str(value).strip()
        if not label:
            raise ValueError(f"{context} returned an empty string; expected non-empty label.")
        return [label]
    if isinstance(value, list):
        labels: list[str] = []
        for idx, item in enumerate(value):
            label = str(item or "").strip()
            if not label:
                raise ValueError(
                    f"{context} returned an empty label at index {idx}; expected non-empty labels."
                )
            labels.append(label)
        if not labels:
            raise ValueError(f"{context} returned an empty list; expected at least one label.")
        if len(set(labels)) != len(labels):
            raise ValueError(f"{context} returned duplicate labels; expected unique labels.")
        return labels
    raise ValueError(
        f"{context} must return a non-empty string or list of non-empty strings."
    )


def _extract_rule_prediction(prediction: Any) -> tuple[str, Optional[float]]:
    if isinstance(prediction, str):
        label = str(prediction).strip()
        if not label:
            raise ValueError(
                "rule.py predict(input) returned an empty string; expected non-empty label."
            )
        return label, None
    if isinstance(prediction, dict):
        label_raw = prediction.get("final_answer")
        label = str(label_raw or "").strip()
        if not label:
            raise ValueError(
                "rule.py predict(input) dict output must include non-empty 'final_answer'."
            )
        return label, _to_finite_float(prediction.get("change_time"))
    raise ValueError(
        "rule.py predict(input) must return a non-empty string or "
        "{'final_answer': <str>, 'change_time': <optional number>}."
    )


def _extract_rule_prediction_labels(prediction: Any) -> tuple[list[str], Optional[float]]:
    if isinstance(prediction, (str, list)):
        return _normalize_rule_prediction_labels(
            prediction,
            context="rule.py predict(input)",
        ), None
    if isinstance(prediction, dict):
        return _normalize_rule_prediction_labels(
            prediction.get("final_answer"),
            context="rule.py predict(input) dict output 'final_answer'",
        ), _to_finite_float(prediction.get("change_time"))
    raise ValueError(
        "rule.py predict(input) must return a non-empty string, non-empty list of labels, "
        "or {'final_answer': <str | list[str]>, 'change_time': <optional number>}."
    )


def _predict_case(
    *,
    model_id: str,
    child_df: Any,
    run_id: str,
    v_min: float,
    ball_drop_inference_version: str,
    allowed_params: set[str],
    models_root: Path,
    rule_predict_fn: Optional[Callable[[Any], Any]],
    rule_display_to_param: Optional[dict[str, str]],
    rule_allowed_labels: Optional[list[str]],
    rule_input: Any = None,
) -> dict[str, object]:
    if rule_predict_fn is not None:
        if rule_display_to_param is None:
            raise ValueError("Internal error: rule label mapping is not initialized.")
        raw_prediction = rule_predict_fn(child_df if rule_input is None else rule_input)
        label_texts, rule_change_time = _extract_rule_prediction_labels(raw_prediction)
        pred_keys: list[str] = []
        final_answers: list[str] = []
        allowed_text = ", ".join(rule_allowed_labels or sorted(rule_display_to_param))
        for label_text in label_texts:
            normalized_label = _normalize_display_label(label_text)
            pred_key = rule_display_to_param.get(normalized_label)
            if pred_key is None:
                raise ValueError(
                    f"rule.py predict(input) returned label '{label_text}' not in allowed labels: "
                    f"[{allowed_text}]"
                )
            pred_keys.append(str(pred_key))
            final_answer = parameter_key_to_display_label(
                model_id=model_id,
                parameter_key=pred_key,
                models_root=models_root,
            )
            final_answer_text = str(final_answer or "").strip()
            if not final_answer_text and pred_key == "no_parameter_change":
                final_answer_text = "no_parameter_change"
            if not final_answer_text:
                raise ValueError(
                    f"Could not resolve canonical display label for parameter '{pred_key}'."
                )
            final_answers.append(final_answer_text)
        return {
            "pred_key": pred_keys[0] if len(pred_keys) == 1 else pred_keys,
            "final_answer": final_answers[0] if len(final_answers) == 1 else final_answers,
            "change_time": _to_finite_float(rule_change_time),
            "pre_params_json": "{}",
            "post_params_json": "{}",
            "pred_pre_value": None,
            "pred_post_value": None,
        }

    kwargs: dict[str, object] = {
        "model_id": model_id,
        "child_df": child_df,
        "run_id": normalize_run_id(run_id),
    }
    if model_id == "DampedMassBetweenWalls":
        kwargs["v_min"] = float(v_min)
    if model_id == "BallDrop":
        kwargs["ball_drop_inference_version"] = str(ball_drop_inference_version)
    result = infer_changed_parameter_child_only(**kwargs)
    pred_key = result.predicted_parameter
    if (
        pred_key is not None
        and str(pred_key) not in allowed_params
        and str(pred_key) not in _NO_CHANGE_KEYS
    ):
        raise click.ClickException(
            f"Predicted parameter '{pred_key}' for run '{run_id}' is outside "
            f"allowed interface for model '{model_id}'"
        )
    canonical = canonical_prediction_payload(
        model_id=model_id,
        predicted_parameter=pred_key,
        change_time=result.intervention_time,
        models_root=models_root,
    )
    return {
        "pred_key": pred_key,
        "final_answer": (
            str(canonical.get("final_answer")).strip()
            if canonical.get("final_answer") is not None
            else None
        ),
        "change_time": _to_finite_float(canonical.get("change_time")),
        "pre_params_json": json.dumps(result.pre_parameters, sort_keys=True),
        "post_params_json": json.dumps(result.post_parameters, sort_keys=True),
        "pred_pre_value": result.parameter_pre_value,
        "pred_post_value": result.parameter_post_value,
    }


def _iter_labeled_children(
    *,
    model_id: str,
    models_root: Path,
    excluded_run_ids: set[str],
    excluded_baseline_spec_ids: set[str],
    runs_dir_name: Optional[str],
) -> Iterable[tuple[str, str, float, str, Path, str, str]]:
    model_dir = models_root / model_id
    record_path = resolve_model_record_path(
        model_dir,
        runs_dir_name=runs_dir_name,
    )
    specs_path = model_dir / "model_run_specs.json"
    runs_root = resolve_runs_root(model_dir, runs_dir_name=runs_dir_name)
    if not record_path.exists():
        raise click.ClickException(f"Missing model_record.json for model '{model_id}': {record_path}")
    if not specs_path.exists():
        raise click.ClickException(f"Missing model_run_specs.json for model '{model_id}': {specs_path}")
    try:
        experiment_config = load_experiment_config_json(model_dir / "experiment_config.json")
        payload = build_model_record_registry(
            model_id=model_id,
            specs=load_model_run_specs_json(
                specs_path,
                enforce_baseline_pair_diversity=False,
            ),
            runtime_map=load_model_record_json(record_path),
            experiment_config=experiment_config,
        )
    except Exception as exc:
        raise click.ClickException(
            f"Failed to build runtime registry for model '{model_id}': {exc}"
        ) from exc

    for baseline in payload.get("baselines", []):
        baseline_spec_id = str(
            baseline.get("baseline_uuid") or baseline.get("run_id") or ""
        )
        if baseline_spec_id in excluded_baseline_spec_ids:
            continue
        for intervention in baseline.get("interventions", []):
            run_id = str(intervention.get("name") or "")
            truth = str(intervention.get("parameter") or "")
            t_true = float(intervention.get("intervention_time", float("nan")))
            intervention_id = str(intervention.get("intervention_uuid") or "")
            time0_baseline_run_id = str(
                intervention.get("time0_baseline_uuid") or ""
            ).strip()
            if not run_id or not truth:
                continue
            if not is_question_eligible(
                run_status=baseline.get("status"),
                intervention_status=intervention.get("status"),
                skipped=False,
            ):
                continue
            if run_id in excluded_run_ids:
                continue
            parquet_path = runs_root / run_id / "data.parquet"
            if parquet_path.exists():
                yield (
                    run_id,
                    truth,
                    t_true,
                    intervention_id,
                    parquet_path,
                    baseline_spec_id,
                    time0_baseline_run_id,
                )


def _load_allowed_parameter_keys(*, model_id: str, models_root: Path) -> set[str]:
    return set(load_allowed_interventions(model_id=model_id, models_root=models_root))


def evaluate_rule_with_noise_outcome(
    *,
    model_id: str,
    run_id: str,
    truth_parameter: str,
    child_df: Any,
    models_root: Path,
    noise_profile: str = "none",
    noise_seed: int = 0,
    v_min: float = 0.01,
    ball_drop_inference_version: str = "v1",
) -> str:
    allowed_params = set(
        load_allowed_interventions(model_id=model_id, models_root=models_root)
    )
    try:
        resolved_noise_profile = normalize_noise_profile(noise_profile)
        noisy_child_df = apply_model_noise_profile(
            child_df,
            model_id=model_id,
            models_root=models_root,
            noise_profile=resolved_noise_profile,
            noise_seed=int(noise_seed),
        )
        prediction_payload = _predict_case(
            model_id=model_id,
            child_df=noisy_child_df,
            run_id=run_id,
            v_min=float(v_min),
            ball_drop_inference_version=str(ball_drop_inference_version).lower(),
            allowed_params=allowed_params,
            models_root=models_root,
            rule_predict_fn=None,
            rule_display_to_param=None,
            rule_allowed_labels=None,
        )
    except NotImplementedError:
        return _RULE_WITH_NO_NOISE_NOT_IMPLEMENTED
    except Exception as exc:
        exc_text = str(exc).strip().lower()
        if "not implement" in exc_text or "unsupported model_id" in exc_text:
            return _RULE_WITH_NO_NOISE_NOT_IMPLEMENTED
        return "error"

    pred_key = prediction_payload.get("pred_key")
    if pred_key is None or not str(pred_key).strip():
        return "none"
    if str(pred_key).strip() == str(truth_parameter or "").strip():
        return "correct"
    return "wrong"
