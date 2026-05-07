#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import click

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.child_only_physics_inference import load_child_df
from shared.child_only_predictor import parameter_key_to_display_label
from shared.model_intervention_interface import (
    list_models_with_allowed_interventions,
    load_allowed_interventions,
)
from shared.interface.model_record_json import load_model_record_json
from shared.shortlist_score import sample_score
from shared.model_noise_adder import (
    call_noise_adder,
    load_noise_adder_from_path,
    normalize_noise_profile,
)
from shared.tsenv_combinations import TIME0_BASELINE_AGENT_FACING_LABEL
from shared.tsenv_task_materialization import materialize
from workflows.metrics._evaluate_rule_helpers import (
    _allowed_display_labels_for_model,
    _build_display_label_to_parameter_key_map,
    _extract_rule_prediction,
    _extract_rule_prediction_labels,
    _iter_labeled_children,
    _load_rule_predict_fn,
    _predict_case,
)

_NO_CHANGE_CLASS_INTERNAL = "no_parameter_change"
_NO_CHANGE_CLASS_AGENT_FACING = TIME0_BASELINE_AGENT_FACING_LABEL
_LEGACY_NO_CHANGE_CLASS_INTERNAL = "nothing_happened"
_LEGACY_NO_CHANGE_CLASS_AGENT_FACING = "Nothing happened"
_LEGACY_NO_CHANGE_ALIASES = {
    _NO_CHANGE_CLASS_INTERNAL,
    "No_parameter_change",
    "No parameter changed",
    "Not sure",
    _LEGACY_NO_CHANGE_CLASS_INTERNAL,
    _NO_CHANGE_CLASS_AGENT_FACING,
    _LEGACY_NO_CHANGE_CLASS_AGENT_FACING,
    TIME0_BASELINE_AGENT_FACING_LABEL,
}


def _resolve_models_root(models_root: Optional[Path]) -> Path:
    if models_root is not None:
        return Path(models_root).expanduser().resolve()
    return root_dir / "models" / "simulink"


def _candidate_runs_dir_names(model_dir: Path) -> list[str]:
    names = sorted(
        {
            p.name
            for p in model_dir.iterdir()
            if p.is_dir() and str(p.name).startswith("runs")
        }
    )
    return names or ["runs"]


def _normalize_display_label(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()


def _is_no_change_label(value: object) -> bool:
    normalized = _normalize_display_label(value)
    return normalized in {
        _normalize_display_label(alias)
        for alias in _LEGACY_NO_CHANGE_ALIASES
    }


def _score_label_key(value: object) -> str:
    if _is_no_change_label(value):
        return _normalize_display_label(_NO_CHANGE_CLASS_AGENT_FACING)
    return _normalize_display_label(value)


def _allowed_choices_include_no_change(
    allowed_choices: Optional[Sequence[str]],
) -> bool:
    return any(_is_no_change_label(choice) for choice in (allowed_choices or []))


def _add_no_change_prediction_support(
    *,
    prediction_allowed_params: set[str],
    rule_display_to_param: dict[str, str],
    rule_allowed_labels: list[str],
) -> list[str]:
    prediction_allowed_params.add(_NO_CHANGE_CLASS_INTERNAL)
    for label in _LEGACY_NO_CHANGE_ALIASES:
        rule_display_to_param[_normalize_display_label(label)] = _NO_CHANGE_CLASS_INTERNAL
    if _NO_CHANGE_CLASS_AGENT_FACING not in rule_allowed_labels:
        return sorted([*rule_allowed_labels, _NO_CHANGE_CLASS_AGENT_FACING])
    return rule_allowed_labels


def _coerce_optional_path(value: object) -> Optional[Path]:
    if value in (None, (), ""):
        return None
    text = str(value).strip()
    if not text or text == "()":
        return None
    return Path(text).expanduser().resolve()


def _resolve_uuid_data_path(uuid_path: object) -> Path:
    path = Path(str(uuid_path or "")).expanduser()
    if path.is_dir():
        path = path / "data.parquet"
    return path.resolve()


def _run_id_from_uuid_path(uuid_path: object) -> str:
    path = Path(str(uuid_path or "")).expanduser()
    if path.suffix:
        return str(path.parent.name).strip()
    return str(path.name).strip()


def _model_dir_from_record_path(model_record: Path) -> Path:
    resolved = Path(model_record).expanduser().resolve()
    if resolved.name != "model_record.json":
        return resolved.parent
    if resolved.parent.name == "runs":
        return resolved.parent.parent
    return resolved.parent


def _runs_root_from_uuid_path(uuid_path: object) -> Optional[Path]:
    path = Path(str(uuid_path or "")).expanduser()
    run_dir = path.parent if path.suffix else path
    if run_dir.name:
        runs_root = run_dir.parent
        if runs_root.name == "runs":
            return runs_root.resolve()
    return None


def _load_materialized_df(
    uuid_path: object,
    *,
    model_id: str,
    models_root: Path,
    noise_level: str,
    seed: int,
    noise_adder_path: Optional[Path],
    first_diff: float = -1.0,
) -> Any:
    run_id = _run_id_from_uuid_path(uuid_path)
    runs_root = _runs_root_from_uuid_path(uuid_path)
    dataframe, _ = materialize(
        str(model_id),
        run_id,
        noise_level,
        int(seed),
        float(first_diff),
        models_root=models_root,
        runs_root=runs_root,
        noise_adder_path=noise_adder_path,
    )
    return dataframe


def _context_truth_label(
    entry: dict[str, object],
    context_level: str,
    *,
    model_id: str,
    models_root: Path,
) -> str:
    normalized = str(context_level or "none").strip().lower()
    internal = str(entry.get("class_internal") or "").strip()
    run_type = str(entry.get("run_type") or "").strip()
    no_change_aliases = {
        _normalize_display_label(alias)
        for alias in _LEGACY_NO_CHANGE_ALIASES
    }
    if run_type in {"baseline", "time0_baseline"} or internal in {
        "baseline",
        _NO_CHANGE_CLASS_INTERNAL,
        _LEGACY_NO_CHANGE_CLASS_INTERNAL,
    } or _normalize_display_label(internal) in no_change_aliases:
        if normalized == "none":
            return _NO_CHANGE_CLASS_INTERNAL
        if normalized in {"low", "high"}:
            return _NO_CHANGE_CLASS_INTERNAL
        raise ValueError("context_level must be one of 'none', 'low', or 'high'.")

    if normalized == "none":
        label = internal
    elif normalized in {"low", "high"}:
        record_label = str(entry.get("class_agent_facing_name") or "").strip()
        label = str(
            parameter_key_to_display_label(
                model_id=str(model_id),
                parameter_key=internal or record_label,
                models_root=models_root,
            )
            or record_label
            or internal
        ).strip()
    else:
        raise ValueError("context_level must be one of 'none', 'low', or 'high'.")
    if not label:
        raise ValueError("Ground-truth label must be non-empty.")
    return label


def _canonical_prediction_for_context(
    *,
    prediction_label: object,
    context_level: str,
    model_id: str,
    models_root: Path,
    truth_label: str,
) -> str:
    label = str(prediction_label or "").strip()
    if not label:
        raise ValueError("rule.py returned an empty canonical prediction.")

    allowed_params = set(
        load_allowed_interventions(model_id=str(model_id), models_root=models_root)
    )
    display_to_param = _build_display_label_to_parameter_key_map(
        model_id=str(model_id),
        models_root=models_root,
        allowed_params=set(allowed_params),
    )
    param_to_display = {
        parameter: str(
            parameter_key_to_display_label(
                model_id=str(model_id),
                parameter_key=parameter,
                models_root=models_root,
            )
            or parameter
        ).strip()
        for parameter in allowed_params
    }
    special_labels = {_NO_CHANGE_CLASS_INTERNAL: _NO_CHANGE_CLASS_INTERNAL}
    param_to_display.update(special_labels)
    display_to_param.update(
        {
            _normalize_display_label(display): key
            for key, display in special_labels.items()
            if str(display).strip()
        }
    )
    for alias in _LEGACY_NO_CHANGE_ALIASES:
        display_to_param[_normalize_display_label(alias)] = _NO_CHANGE_CLASS_INTERNAL

    normalized_label = _normalize_display_label(label)
    normalized_context = str(context_level or "none").strip().lower()
    if normalized_context == "none":
        if label in param_to_display:
            return label
        if normalized_label in display_to_param:
            return str(display_to_param[normalized_label]).strip()
        allowed = sorted({*param_to_display.keys(), truth_label})
    elif normalized_context in {"low", "high"}:
        if normalized_label in display_to_param:
            key = display_to_param[normalized_label]
            return str(param_to_display.get(key) or label).strip()
        if label in param_to_display:
            return str(param_to_display[label]).strip()
        allowed = sorted({*param_to_display.values(), truth_label})
    else:
        raise ValueError("context_level must be one of 'none', 'low', or 'high'.")

    if _normalize_display_label(label) == _normalize_display_label(truth_label):
        return str(truth_label).strip()
    raise ValueError(
        f"Prediction '{label}' is not in allowed labels {allowed}"
    )


def _candidate_model_ids(
    *,
    models_root: Path,
    model_spec_path: Optional[Path],
) -> list[str]:
    known_models = list_models_with_allowed_interventions(models_root)
    if model_spec_path is None:
        return known_models
    candidate_dir = (
        model_spec_path
        if model_spec_path.is_dir()
        else model_spec_path.parent
    )
    candidate_id = str(candidate_dir.name or "").strip()
    if candidate_id and candidate_id in known_models:
        return [candidate_id]
    return known_models


def _resolve_uuid_entry(
    *,
    sample_uuid: str,
    models_root: Path,
    model_spec_path: Optional[Path] = None,
) -> dict[str, object]:
    target = str(sample_uuid or "").strip()
    if not target:
        raise click.ClickException("uuid must be a non-empty string.")

    excluded_run_ids: set[str] = set()
    excluded_baseline_spec_ids: set[str] = set()

    for model_id in _candidate_model_ids(
        models_root=models_root,
        model_spec_path=model_spec_path,
    ):
        model_dir = models_root / model_id
        for runs_dir_name in _candidate_runs_dir_names(model_dir):
            try:
                rows = list(
                    _iter_labeled_children(
                        model_id=model_id,
                        models_root=models_root,
                        excluded_run_ids=excluded_run_ids,
                        excluded_baseline_spec_ids=excluded_baseline_spec_ids,
                        runs_dir_name=runs_dir_name,
                    )
                )
            except Exception:
                continue

            for (
                run_id,
                truth_parameter,
                _t_true,
                intervention_uuid,
                parquet_path,
                baseline_spec_id,
                time0_baseline_run_id,
            ) in rows:
                if str(run_id) != target:
                    continue
                return {
                    "resolved_lookup_type": "run_id",
                    "model_id": str(model_id),
                    "run_id": str(run_id),
                    "truth_label": str(truth_parameter),
                    "intervention_uuid": str(intervention_uuid),
                    "data_path": Path(parquet_path).expanduser().resolve(),
                    "runs_dir_name": str(runs_dir_name),
                    "baseline_spec_id": str(baseline_spec_id),
                    "time0_baseline_run_id": str(time0_baseline_run_id),
                }

            for (
                run_id,
                truth_parameter,
                _t_true,
                intervention_uuid,
                parquet_path,
                baseline_spec_id,
                time0_baseline_run_id,
            ) in rows:
                if str(intervention_uuid) != target:
                    continue
                return {
                    "resolved_lookup_type": "intervention_uuid",
                    "model_id": str(model_id),
                    "run_id": str(run_id),
                    "truth_label": str(truth_parameter),
                    "intervention_uuid": str(intervention_uuid),
                    "data_path": Path(parquet_path).expanduser().resolve(),
                    "runs_dir_name": str(runs_dir_name),
                    "baseline_spec_id": str(baseline_spec_id),
                    "time0_baseline_run_id": str(time0_baseline_run_id),
                }

    raise click.ClickException(
        f"Could not resolve uuid '{target}' to a labeled child run under {models_root}."
    )


def _empty_result(
    *,
    model_id: object = None,
    run_id: object = None,
    ground_truth: object = None,
    predicted_label: object = None,
    noise_level: object = "none",
    seed: int = 0,
    rule_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    error: object = None,
) -> dict[str, object]:
    return {
        "is_correct": False,
        "outcome": "error",
        "model_id": str(model_id).strip() if model_id is not None else None,
        "run_id": str(run_id).strip() if run_id is not None else None,
        "ground_truth": str(ground_truth).strip() if ground_truth is not None else None,
        "predicted_label": (
            str(predicted_label).strip() if predicted_label is not None else None
        ),
        "noise_level": str(noise_level or "").strip(),
        "seed": int(seed),
        "rule_path": str(Path(rule_path).expanduser().resolve()) if rule_path else None,
        "data_path": str(Path(data_path).expanduser().resolve()) if data_path else None,
        "error": str(error).strip() if error is not None else None,
    }


def _public_rule_result(payload: dict[str, object]) -> dict[str, object]:
    return {
        "ground_truth": payload.get("ground_truth"),
        "predicted_label": payload.get("predicted_label"),
        "is_correct": bool(payload.get("is_correct") is True),
    }


def _public_shortlist_rule_result(payload: dict[str, object]) -> dict[str, object]:
    return {
        "ground_truth": payload.get("ground_truth"),
        "predicted_label": payload.get("predicted_label"),
        "top1_correct": bool(payload.get("top1_correct") is True),
        "shortlist_score": float(payload.get("shortlist_score") or 0.0),
        "num_answers": int(payload.get("num_answers") or 0),
    }


def _evaluation_type(value: object) -> str:
    normalized = str(value or "accuracy").strip().lower()
    if normalized not in {"accuracy", "shortlist_score"}:
        raise ValueError(
            "evaluation_type must be one of 'accuracy' or 'shortlist_score'."
        )
    return normalized


def _shortlist_score_from_predictions(
    *,
    predicted_labels: Sequence[str],
    truth_label: str,
) -> float:
    prediction_keys = list(
        dict.fromkeys(
            _score_label_key(label)
            for label in predicted_labels
            if str(label).strip()
        )
    )
    return sample_score(
        prediction_keys,
        _score_label_key(truth_label),
    )


def _apply_noise_profile(
    *,
    child_df: Any,
    baseline_df: Any = None,
    model_id: str,
    models_root: Path,
    noise_level: str,
    seed: int,
    noise_adder_path: Optional[Path],
    first_diff: float = -1.0,
) -> Any:
    noise_profile = normalize_noise_profile(noise_level)
    if noise_profile == "none":
        return child_df.copy() if hasattr(child_df, "copy") else child_df
    if noise_adder_path is not None:
        add_noise = load_noise_adder_from_path(noise_adder_path)
        noisy_df, _ = call_noise_adder(
            add_noise,
            child_df,
            baseline_df=baseline_df,
            first_diff=float(first_diff),
            seed=int(seed),
            noise_level=noise_profile,
        )
        return noisy_df
    raise ValueError(
        "noise_adder_path is required when noise_level is not 'none'."
    )


def _resolve_truth_labels(
    *,
    model_id: str,
    truth_label: str,
    models_root: Path,
    allowed_params: set[str],
) -> tuple[str, str]:
    truth_text = str(truth_label or "").strip()
    if not truth_text:
        raise ValueError("Ground-truth label must be non-empty.")
    if _normalize_display_label(truth_text) in {
        _normalize_display_label(alias)
        for alias in _LEGACY_NO_CHANGE_ALIASES
    }:
        return _NO_CHANGE_CLASS_INTERNAL, _NO_CHANGE_CLASS_INTERNAL
    if truth_text in allowed_params:
        return truth_text, str(
            parameter_key_to_display_label(
                model_id=model_id,
                parameter_key=truth_text,
                models_root=models_root,
            )
            or truth_text
        )

    for parameter_key in sorted(allowed_params):
        display_label = str(
            parameter_key_to_display_label(
                model_id=model_id,
                parameter_key=parameter_key,
                models_root=models_root,
            )
            or parameter_key
        ).strip()
        if _normalize_display_label(display_label) == _normalize_display_label(truth_text):
            return parameter_key, display_label
    if _normalize_display_label(truth_text) in {
        _normalize_display_label(_NO_CHANGE_CLASS_AGENT_FACING),
        _normalize_display_label(_LEGACY_NO_CHANGE_CLASS_AGENT_FACING),
        _normalize_display_label(_LEGACY_NO_CHANGE_CLASS_INTERNAL),
        _normalize_display_label(TIME0_BASELINE_AGENT_FACING_LABEL),
    }:
        return _NO_CHANGE_CLASS_INTERNAL, _NO_CHANGE_CLASS_INTERNAL
    raise ValueError(
        f"Ground-truth label '{truth_text}' is outside the allowed interface for model '{model_id}'."
    )


def _normalize_predicted_labels(value: object) -> list[str]:
    if isinstance(value, str):
        label = str(value).strip()
        if not label:
            raise ValueError("Predicted label must be non-empty.")
        return [label]
    if isinstance(value, list):
        labels: list[str] = []
        for idx, item in enumerate(value):
            label = str(item or "").strip()
            if not label:
                raise ValueError(f"Predicted label at index {idx} must be non-empty.")
            labels.append(label)
        if not labels:
            raise ValueError("Predicted labels list must contain at least one label.")
        if len(set(labels)) != len(labels):
            raise ValueError("Predicted labels list must not contain duplicates.")
        return labels
    raise ValueError("Predicted label must be a string or list of strings.")


def _shortlist_score(
    *,
    predicted_labels: Sequence[str],
    truth_label: str,
    allowed_choices: Optional[Sequence[str]],
) -> float:
    normalized_predictions = {
        _score_label_key(label)
        for label in predicted_labels
        if str(label).strip()
    }
    normalized_truth = _score_label_key(truth_label)
    if normalized_truth not in normalized_predictions:
        return 0.0
    return 1.0 / float(max(1, len(normalized_predictions)))


def _allowed_choice_label_map(
    allowed_choices: Optional[Sequence[str]],
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for choice in allowed_choices or []:
        label = str(choice or "").strip()
        if not label:
            continue
        labels.setdefault(_score_label_key(label), label)
    return labels


def evaluate_rule_on_dataframe(
    *,
    model_id: str,
    child_df: Any,
    truth_label: str,
    rule_path: Path,
    noise_adder_path: object = (),
    model_spec_path: object = (),
    models_root: Optional[Path] = None,
    run_id: str = "",
    noise_level: str = "none",
    seed: int = 0,
    allowed_choices: Optional[Sequence[str]] = None,
    data_path: Optional[Path] = None,
) -> dict[str, object]:
    resolved_models_root = _resolve_models_root(models_root)
    resolved_rule_path = Path(rule_path).expanduser().resolve()
    resolved_noise_adder_path = _coerce_optional_path(noise_adder_path)
    _ = _coerce_optional_path(model_spec_path)
    result = _empty_result(
        model_id=model_id,
        run_id=run_id,
        ground_truth=truth_label,
        noise_level=noise_level,
        seed=int(seed),
        rule_path=resolved_rule_path,
        data_path=data_path,
    )
    try:
        choice_by_score_key = _allowed_choice_label_map(allowed_choices)
        truth_text = str(truth_label or "").strip()
        truth_score_key = _score_label_key(truth_text)
        if truth_text and truth_score_key in choice_by_score_key:
            noisy_child_df = _apply_noise_profile(
                child_df=child_df,
                model_id=str(model_id),
                models_root=resolved_models_root,
                noise_level=str(noise_level),
                seed=int(seed),
                noise_adder_path=resolved_noise_adder_path,
            )
            rule_predict_fn = _load_rule_predict_fn(resolved_rule_path)
            raw_prediction = rule_predict_fn(noisy_child_df)
            raw_predicted_labels, _change_time = _extract_rule_prediction_labels(raw_prediction)
            predicted_labels: list[str] = []
            seen_prediction_keys: set[str] = set()
            invalid_predictions: list[str] = []
            for label in raw_predicted_labels:
                prediction_key = _score_label_key(label)
                canonical_label = choice_by_score_key.get(prediction_key)
                if canonical_label is None:
                    invalid_predictions.append(str(label).strip())
                    continue
                if prediction_key in seen_prediction_keys:
                    raise ValueError("Predicted labels list must not contain duplicates.")
                seen_prediction_keys.add(prediction_key)
                predicted_labels.append(canonical_label)
            if invalid_predictions:
                raise ValueError(
                    "Predictions "
                    f"{invalid_predictions!r} are not in multiple_choices {sorted(choice_by_score_key.values())}"
                )
            top1_correct = bool(predicted_labels) and (
                _score_label_key(predicted_labels[0]) == truth_score_key
            )
            predicted_label: object = (
                predicted_labels[0] if len(predicted_labels) == 1 else list(predicted_labels)
            )
            result.update(
                {
                    "is_correct": bool(top1_correct),
                    "top1_correct": bool(top1_correct),
                    "outcome": "correct" if top1_correct else "wrong",
                    "ground_truth": choice_by_score_key[truth_score_key],
                    "predicted_label": predicted_label,
                    "shortlist_score": _shortlist_score(
                        predicted_labels=predicted_labels,
                        truth_label=truth_text,
                        allowed_choices=allowed_choices,
                    ),
                    "answer_count": len(predicted_labels),
                    "error": None,
                }
            )
            return result

        allowed_params = set(
            load_allowed_interventions(
                model_id=str(model_id),
                models_root=resolved_models_root,
            )
        )
        truth_internal, truth_display = _resolve_truth_labels(
            model_id=str(model_id),
            truth_label=str(truth_label),
            models_root=resolved_models_root,
            allowed_params=allowed_params,
        )
        prediction_allowed_params = set(allowed_params)
        rule_display_to_param = _build_display_label_to_parameter_key_map(
            model_id=str(model_id),
            models_root=resolved_models_root,
            allowed_params=prediction_allowed_params,
        )
        rule_allowed_labels = _allowed_display_labels_for_model(
            model_id=str(model_id),
            models_root=resolved_models_root,
            allowed_params=prediction_allowed_params,
        )
        if truth_internal == _NO_CHANGE_CLASS_INTERNAL or _allowed_choices_include_no_change(allowed_choices):
            rule_allowed_labels = _add_no_change_prediction_support(
                prediction_allowed_params=prediction_allowed_params,
                rule_display_to_param=rule_display_to_param,
                rule_allowed_labels=rule_allowed_labels,
            )

        noisy_child_df = _apply_noise_profile(
            child_df=child_df,
            model_id=str(model_id),
            models_root=resolved_models_root,
            noise_level=str(noise_level),
            seed=int(seed),
            noise_adder_path=resolved_noise_adder_path,
        )
        rule_predict_fn = _load_rule_predict_fn(resolved_rule_path)
        prediction_payload = _predict_case(
            model_id=str(model_id),
            child_df=noisy_child_df,
            run_id=str(run_id).strip(),
            v_min=0.01,
            ball_drop_inference_version="v1",
            allowed_params=prediction_allowed_params,
            models_root=resolved_models_root,
            rule_predict_fn=rule_predict_fn,
            rule_display_to_param=rule_display_to_param,
            rule_allowed_labels=rule_allowed_labels,
        )
        predicted_label_raw = prediction_payload.get("final_answer")
        if predicted_label_raw is None:
            predicted_key = prediction_payload.get("pred_key")
            if predicted_key is not None:
                predicted_label_raw = parameter_key_to_display_label(
                    model_id=str(model_id),
                    parameter_key=predicted_key,
                    models_root=resolved_models_root,
                )
        if predicted_label_raw is None:
            raise ValueError("rule.py returned an empty canonical prediction.")
        predicted_labels = [
            _NO_CHANGE_CLASS_AGENT_FACING if _is_no_change_label(label) else label
            for label in _normalize_predicted_labels(predicted_label_raw)
        ]
        if allowed_choices is not None:
            valid_choices = {
                _score_label_key(choice)
                for choice in allowed_choices
                if str(choice).strip()
            }
            invalid_predictions = [
                label
                for label in predicted_labels
                if _score_label_key(label) not in valid_choices
            ]
            if valid_choices and invalid_predictions:
                raise ValueError(
                    "Predictions "
                    f"{invalid_predictions!r} are not in multiple_choices {sorted(valid_choices)}"
                )
        top1_correct = bool(predicted_labels) and (
            _score_label_key(predicted_labels[0]) == _score_label_key(truth_display)
        )
        predicted_label: object = (
            predicted_labels[0] if len(predicted_labels) == 1 else list(predicted_labels)
        )
        result.update(
            {
                "is_correct": bool(top1_correct),
                "top1_correct": bool(top1_correct),
                "outcome": "correct" if top1_correct else "wrong",
                "ground_truth": truth_display,
                "predicted_label": predicted_label,
                "shortlist_score": _shortlist_score(
                    predicted_labels=predicted_labels,
                    truth_label=truth_display,
                    allowed_choices=allowed_choices,
                ),
                "answer_count": len(predicted_labels),
                "error": None,
            }
        )
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["top1_correct"] = False
        result["shortlist_score"] = 0.0
        result["answer_count"] = 0
        return result


def evaluate_rule(
    uuid_path: object,
    rule_path: Path,
    model_record: Path,
    context_level: str = "none",
    noise_level: str = "none",
    seed: int = 0,
    noise_adder_path: object = None,
    evaluation_type: str = "accuracy",
    first_diff: float = -1.0,
) -> dict[str, object]:
    resolved_evaluation_type = _evaluation_type(evaluation_type)
    resolved_rule_path = Path(rule_path).expanduser().resolve()
    resolved_model_record = Path(model_record).expanduser().resolve()
    resolved_noise_adder_path = _coerce_optional_path(noise_adder_path)
    if (
        normalize_noise_profile(noise_level) != "none"
        and resolved_noise_adder_path is None
    ):
        raise ValueError(
            "noise_adder_path is required when noise_level is not 'none'."
        )
    model_dir = _model_dir_from_record_path(resolved_model_record)
    models_root = model_dir.parent
    model_id = model_dir.name
    run_id = _run_id_from_uuid_path(uuid_path)
    result = _empty_result(
        model_id=model_id,
        run_id=run_id,
        noise_level=noise_level,
        seed=int(seed),
        rule_path=resolved_rule_path,
        data_path=_resolve_uuid_data_path(uuid_path),
    )
    try:
        model_record_payload = load_model_record_json(resolved_model_record)
        entry = model_record_payload.get(run_id)
        if not isinstance(entry, dict):
            raise ValueError(f"uuid '{run_id}' is missing from model_record.json.")
        truth_label = _context_truth_label(
            entry,
            context_level,
            model_id=model_id,
            models_root=models_root,
        )
        result["ground_truth"] = truth_label
        child_df = _load_materialized_df(
            uuid_path,
            model_id=model_id,
            models_root=models_root,
            noise_level=str(noise_level),
            seed=int(seed),
            noise_adder_path=resolved_noise_adder_path,
            first_diff=float(first_diff),
        )
        predict = _load_rule_predict_fn(resolved_rule_path)
        raw_prediction = predict(child_df)
        if resolved_evaluation_type == "shortlist_score":
            predicted_label_values, _change_time = _extract_rule_prediction_labels(raw_prediction)
            predicted_labels = [
                _canonical_prediction_for_context(
                    prediction_label=label,
                    context_level=context_level,
                    model_id=model_id,
                    models_root=models_root,
                    truth_label=truth_label,
                )
                for label in predicted_label_values
            ]
            top1_correct = bool(predicted_labels) and (
                _score_label_key(predicted_labels[0]) == _score_label_key(truth_label)
            )
            result.update(
                {
                    "ground_truth": truth_label,
                    "predicted_label": predicted_labels,
                    "top1_correct": bool(top1_correct),
                    "shortlist_score": _shortlist_score_from_predictions(
                        predicted_labels=predicted_labels,
                        truth_label=truth_label,
                    ),
                    "num_answers": len(predicted_labels),
                    "error": None,
                }
            )
            return _public_shortlist_rule_result(result)
        predicted_label_raw, _change_time = _extract_rule_prediction(raw_prediction)
        predicted_label = _canonical_prediction_for_context(
            prediction_label=predicted_label_raw,
            context_level=context_level,
            model_id=model_id,
            models_root=models_root,
            truth_label=truth_label,
        )
        result.update(
            {
                "ground_truth": truth_label,
                "predicted_label": predicted_label,
                "is_correct": (
                    _normalize_display_label(predicted_label)
                    == _normalize_display_label(truth_label)
                ),
                "error": None,
            }
        )
        return _public_rule_result(result)
    except Exception as exc:
        result["ground_truth"] = result.get("ground_truth")
        result["error"] = f"{type(exc).__name__}: {exc}"
        if resolved_evaluation_type == "shortlist_score":
            return _public_shortlist_rule_result(result)
        return _public_rule_result(result)


@click.command(help="Evaluate a rule.py on one labeled tsENV run.")
@click.argument("uuid_path")
@click.option("--rule_path", required=True, type=click.Path(path_type=Path))
@click.option("--model_record", required=True, type=click.Path(path_type=Path))
@click.option(
    "--context_level",
    default="none",
    show_default=True,
    type=click.Choice(["none", "low", "high"]),
)
@click.option(
    "--noise_level",
    default="none",
    show_default=True,
    type=click.Choice(["none", "low", "high"]),
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--noise_adder_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--evaluation_type",
    default="accuracy",
    show_default=True,
    type=click.Choice(["accuracy", "shortlist_score"]),
)
@click.option("--first_diff", type=float, default=-1.0, show_default=True)
@click.option(
    "--json-indent",
    type=int,
    default=2,
    show_default=True,
    help="Indent level for JSON output.",
)
def main(
    uuid_path: str,
    rule_path: Path,
    model_record: Path,
    context_level: str,
    noise_level: str,
    seed: int,
    noise_adder_path: Optional[Path],
    evaluation_type: str,
    first_diff: float,
    json_indent: int,
) -> None:
    payload = evaluate_rule(
        uuid_path,
        rule_path,
        model_record,
        context_level=context_level,
        noise_level=noise_level,
        seed=int(seed),
        noise_adder_path=noise_adder_path,
        evaluation_type=evaluation_type,
        first_diff=float(first_diff),
    )
    click.echo(json.dumps(payload, indent=int(json_indent)))


if __name__ == "__main__":
    main()
