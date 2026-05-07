#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import re
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import click
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.benchmark_utils import (
    benchmark_root_from_label,
    is_classification_run,
    is_equation_run,
    is_simbench_cls_benchmark,
)
from shared.agentic_rollout_paths import resolve_final_response_path
from shared.interface.agent_answer_json import validate_tsenv_direct_agent_answer
from shared.scores_schema import validate_scores_payload
from shared.tsenv_eval_mode import normalize_tsenv_eval_mode
from shared.tsenv_metadata import (
    _question_manifest_item,
    label_choices_from_payload,
    label_for_question_sample,
    load_metadata_payload,
    metadata_questions_by_id,
    question_sample_paths,
    resolve_tsenv_payload_path,
)
from shared.tsenv_task_materialization import materialize_dataframe as materialize
from workflows.evaluate_on_run_params import resolve_run_base_seed, resolve_run_noise_level
from workflows.metrics.evaluate_rule import evaluate_rule_on_dataframe


RUNS_ROOT = REPO_ROOT / "terminal-bench" / "runs"
TSENV_ROOT = REPO_ROOT / "tsENV_questions"
MODELS_ROOT = REPO_ROOT / "models" / "simulink"
ACCURACY_SUMMARY_NAME = "accuracy_summary.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_run_dir(agentic_run_id: str | None, run_path: Path | None) -> Path:
    has_run_id = bool(str(agentic_run_id or "").strip())
    has_run_path = run_path is not None
    if has_run_id == has_run_path:
        raise click.ClickException("Pass exactly one of <agentic_run_id> or --path.")
    if has_run_id:
        run_dir = (RUNS_ROOT / str(agentic_run_id).strip()).resolve()
    else:
        run_dir = Path(run_path).expanduser().resolve()
    if not run_dir.exists():
        raise click.ClickException(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise click.ClickException(f"Run path is not a directory: {run_dir}")
    return run_dir


def _normalize_path_key(value: object) -> str:
    normalized = str(value).strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _question_key_from_entry(entry: Mapping[str, Any], scenario: Mapping[str, Any]) -> str:
    for raw_value in (
        scenario.get("question_id"),
        scenario.get("question_hash"),
        scenario.get("hash"),
        entry.get("question_id"),
        entry.get("task_hash"),
        entry.get("task_id"),
    ):
        value = str(raw_value or "").strip()
        if value:
            return value
    raise ValueError("Unable to resolve question key for trial")


def _question_hash_candidates(question: Mapping[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for field_name in ("question_id", "question_hash", "hash", "train_test_sample_hash"):
        value = str(question.get(field_name) or "").strip()
        if value:
            candidates.add(value)
    return candidates


def _load_tsenv_question_lookup(run_metadata: Mapping[str, Any], question_id: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_path_value = run_metadata.get("dataset_path")
    dataset_path = Path(str(dataset_path_value or "")).expanduser().resolve()
    model_name = dataset_path.parent.name
    dataset_slug = dataset_path.name
    payload_path = resolve_tsenv_payload_path(TSENV_ROOT / model_name)
    payload = load_metadata_payload(payload_path)
    questions = metadata_questions_by_id(payload)

    candidate_keys = [
        candidate
        for candidate in (str(question_id or "").strip(), str(dataset_slug or "").strip())
        if candidate
    ]
    question = next(
        (questions[candidate] for candidate in candidate_keys if candidate in questions),
        None,
    )
    if question is None:
        matching_questions = [
            candidate_question
            for candidate_question in questions.values()
            if str(question_id or "").strip() in _question_hash_candidates(candidate_question)
        ]
        if len(matching_questions) == 1:
            question = matching_questions[0]
        elif len(matching_questions) > 1:
            raise KeyError(
                f"question_id {question_id!r} matched multiple questions in {payload_path}"
            )
    if question is None:
        rendered_candidates = ", ".join(repr(candidate) for candidate in candidate_keys)
        raise KeyError(
            f"question_id {question_id!r} not found in {payload_path}; tried {rendered_candidates}"
        )
    return model_name, payload, question


def _scenario_variant(scenario: Mapping[str, Any]) -> str:
    context = str(
        scenario.get("desc_level") or scenario.get("context") or "unknown"
    ).strip().lower()
    shot = str(scenario.get("shot") or "unknown").strip().lower()
    return f"{context}_{shot}"


def _normalize_eval_mode(value: object) -> str:
    return normalize_tsenv_eval_mode(value)


def _ensure_non_empty_string(value: Any, label: str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
        return value
    if value is None:
        raise TypeError(f"{label} must be a non-empty string")
    rendered = json.dumps(value, ensure_ascii=True)
    if not rendered:
        raise ValueError(f"{label} must be a non-empty string")
    return rendered


def _sample_id_from_path(value: object) -> str:
    return Path(_normalize_path_key(value)).stem


def _test_samples(question_row: Mapping[str, Any]) -> list[str]:
    raw_value = question_row.get("test_samples")
    if not isinstance(raw_value, list):
        return []
    return [str(path) for path in raw_value if str(path).strip()]


def _test_source_samples(question_row: Mapping[str, Any]) -> list[str]:
    raw_value = question_row.get("test_samples_source_paths")
    if not isinstance(raw_value, list):
        return []
    return [str(path) for path in raw_value if str(path).strip()]


def _test_label_map(question_row: Mapping[str, Any]) -> dict[str, str]:
    label_map_raw = question_row.get("test_sample_labels")
    if not isinstance(label_map_raw, dict):
        raise ValueError("test_sample_labels is required for scoring.")
    return {
        str(path): str(label)
        for path, label in label_map_raw.items()
        if str(path).strip() and str(label).strip()
    }


def _build_scores_payload(
    *,
    agent_run_id: str,
    sample_results: Mapping[str, Mapping[str, Any]],
    include_final_metric_other: bool = False,
    is_correct_format: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "agent_run_id": str(agent_run_id),
        "final_metric_test": _metric_for_sample_type(sample_results, "test"),
        "sample_results": {
            str(key): dict(value) for key, value in sample_results.items()
        },
    }
    if is_correct_format is not None:
        payload["is_correct_format"] = bool(is_correct_format)
    if include_final_metric_other:
        payload["final_metric_other"] = _metric_for_sample_type(sample_results, "other")
    validated = validate_scores_payload(payload).model_dump()
    if validated.get("is_correct_format") is None:
        validated.pop("is_correct_format", None)
    if validated.get("final_metric_other") is None:
        validated.pop("final_metric_other", None)
    for sample_result in validated["sample_results"].values():
        if sample_result.get("error") is None:
            sample_result.pop("error", None)
    return validated


def _metric_for_sample_type(
    sample_results: Mapping[str, Mapping[str, Any]],
    sample_type: str,
) -> dict[str, float]:
    entries = [
        dict(value)
        for value in sample_results.values()
        if str(value.get("sample_type") or "").strip() == sample_type
    ]
    if not entries:
        return {
            "average_top1_accuracy": 0.0,
            "average_shortlist_score": 0.0,
            "average_num_answers": 0.0,
        }
    denominator = float(len(entries))
    return {
        "average_top1_accuracy": sum(
            1.0 if entry.get("top1_correct") is True else 0.0
            for entry in entries
        )
        / denominator,
        "average_shortlist_score": sum(
            float(entry.get("shortlist_score") or 0.0) for entry in entries
        )
        / denominator,
        "average_num_answers": sum(
            value
            for value in (
                _finite_or_none(entry.get("num_answers")) for entry in entries
            )
            if value is not None
        )
        / max(
            1.0,
            float(
                len(
                    [
                        entry
                        for entry in entries
                        if _finite_or_none(entry.get("num_answers")) is not None
                    ]
                )
            ),
        ),
    }


def _sample_result_entry(
    *,
    predictions: object,
    top1_correct: bool,
    shortlist_score: float,
    num_answers: float | None,
    sample_type: str,
    error: str | None = None,
) -> dict[str, Any]:
    entry = {
        "predictions": _prediction_list(predictions),
        "top1_correct": bool(top1_correct),
        "shortlist_score": float(shortlist_score),
        "num_answers": None if num_answers is None else float(num_answers),
        "sample_type": sample_type,
    }
    if error is not None:
        entry["error"] = str(error)
    return entry


def _prediction_list(predictions: object) -> list[str]:
    if isinstance(predictions, list):
        return [str(item).strip() for item in predictions if str(item).strip()]
    text = str(predictions or "").strip()
    return [text] if text else []


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _resolve_classification_choice(answer: str, choices: Sequence[str]) -> str:
    normalized_answer = str(answer).strip()
    if not normalized_answer:
        raise ValueError("Final answer must be non-empty.")
    for choice in choices:
        normalized_choice = str(choice).strip()
        if normalized_answer == normalized_choice:
            return normalized_choice
    lowered_matches = [
        str(choice).strip()
        for choice in choices
        if normalized_answer.lower() == str(choice).strip().lower()
    ]
    if len(lowered_matches) == 1:
        return lowered_matches[0]
    raise ValueError(
        f"Final answer must match one of the provided choices: {choices!r}."
    )


def _resolve_classification_choices(answer: Any, choices: Sequence[str]) -> list[str]:
    if isinstance(answer, list):
        resolved = [_resolve_classification_choice(str(item), choices) for item in answer]
        if len(set(resolved)) != len(resolved):
            raise ValueError(f"Final answer must not repeat choices: {answer!r}.")
        return resolved
    return [_resolve_classification_choice(str(answer), choices)]


def _score_direct_prediction(predicted: Sequence[str], truth: str, choices: Sequence[str]) -> float:
    if truth not in set(predicted):
        return 0.0
    return 1.0 / float(max(1, len(predicted)))


def _direct_results_path(trial_dir: Path) -> Path:
    return trial_dir / "results.json"


def _direct_results_candidate_paths(trial_dir: Path) -> list[Path]:
    primary = _direct_results_path(trial_dir)
    artifact = trial_dir / "artifacts" / "results.json"
    if artifact == primary:
        return [primary]
    return [primary, artifact]


def _validate_direct_predictions(
    payload: Any,
    *,
    expected_samples: Sequence[str],
    expected_keys: Sequence[str],
    path: Path,
) -> dict[str, str | list[str]]:
    validated = validate_tsenv_direct_agent_answer(
        payload,
        expected_samples=expected_keys,
        path=str(path),
    )
    expected_by_key = dict(zip(expected_keys, expected_samples))
    return {
        expected_by_key[key]: value
        for key, value in validated.predictions.items()
        if key in expected_by_key
    }


def _load_direct_prediction_map(
    *,
    trial_dir: Path,
    expected_samples: Sequence[str],
) -> tuple[dict[str, str | list[str]], bool]:
    normalized_expected = [
        _normalize_path_key(sample)
        for sample in expected_samples
        if str(sample).strip()
    ]
    if not normalized_expected:
        raise ValueError("No direct test_samples found for scoring.")

    first_error: ValueError | None = None
    basename_keys = [Path(sample).name for sample in normalized_expected]
    for results_path in _direct_results_candidate_paths(trial_dir):
        if not results_path.exists():
            continue
        payload = _read_json(results_path)
        if len(set(basename_keys)) == len(basename_keys):
            try:
                return (
                    _validate_direct_predictions(
                        payload,
                        expected_samples=normalized_expected,
                        expected_keys=basename_keys,
                        path=results_path,
                    ),
                    True,
                )
            except ValueError as canonical_error:
                if first_error is None:
                    first_error = canonical_error
        try:
            return (
                _validate_direct_predictions(
                    payload,
                    expected_samples=normalized_expected,
                    expected_keys=normalized_expected,
                    path=results_path,
                ),
                False,
            )
        except ValueError:
            continue

    if first_error is not None:
        raise first_error
    missing_paths = ", ".join(str(path) for path in _direct_results_candidate_paths(trial_dir))
    raise FileNotFoundError(f"Missing direct prediction results.json; checked {missing_paths}")


def _scenario_benchmark_variant(
    scenario_payload: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    context = str(
        scenario_payload.get("desc_level") or scenario_payload.get("context") or ""
    ).strip().lower()
    shot = str(scenario_payload.get("shot") or "").strip().lower()
    if not context or not shot:
        return None, None
    return "tsenv_cls", f"{context}_{shot}"


def _normalize_benchmark_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() == "simbench_cls":
        return "tsenv_cls"
    return normalized


def _entry_scores_payload(entry: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = entry.get("scores")
    if not isinstance(payload, dict):
        return None
    normalized = dict(payload)
    benchmark = _normalize_benchmark_name(normalized.get("benchmark"))
    if benchmark is not None:
        normalized["benchmark"] = benchmark
    try:
        return validate_scores_payload(
            normalized, path="results.json entry scores"
        ).model_dump()
    except Exception:
        return None


def _resolve_expected_label(
    labels_by_source: Mapping[str, str],
    source_sample_path: object,
) -> str | None:
    normalized_source = _normalize_path_key(source_sample_path)
    if normalized_source in labels_by_source:
        return labels_by_source[normalized_source]
    basename = Path(normalized_source).name
    matching = [
        value
        for key, value in labels_by_source.items()
        if Path(_normalize_path_key(key)).name == basename
    ]
    if len(matching) == 1:
        return matching[0]
    return None


def _parse_open_prediction_value(raw_value: Any, choices: Sequence[str]) -> str | None:
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text:
        return None
    try:
        return _resolve_classification_choice(text, choices)
    except ValueError:
        pass
    matches: list[str] = []
    lowered_text = text.lower()
    for choice in choices:
        normalized_choice = str(choice).strip()
        if not normalized_choice:
            continue
        if re.search(rf"(?<!\w){re.escape(normalized_choice.lower())}(?!\w)", lowered_text):
            matches.append(normalized_choice)
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def _extract_open_prediction_payload(
    agent_payload: Any,
    *,
    expected_samples: Sequence[str],
) -> dict[str, str]:
    if isinstance(agent_payload, dict) and "final_answer" not in agent_payload:
        return {
            str(key).strip(): str(value)
            for key, value in agent_payload.items()
            if str(key).strip()
        }
    final_answer = ""
    if isinstance(agent_payload, dict):
        final_answer = str(agent_payload.get("final_answer") or "").strip()
    elif isinstance(agent_payload, str):
        final_answer = agent_payload.strip()
    if not final_answer:
        return {}
    try:
        parsed = json.loads(final_answer)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return {
            str(key).strip(): str(value)
            for key, value in parsed.items()
            if str(key).strip()
        }
    if len(expected_samples) == 1:
        return {str(expected_samples[0]): final_answer}
    return {}


def _sample_entries_from_scores_payload(
    scores_payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    sample_results = scores_payload.get("sample_results")
    if not isinstance(sample_results, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in sample_results.items()
        if isinstance(value, dict)
    }


def load_question_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    run_results_path = run_dir / "results.json"
    run_metadata = _read_json(run_dir / "run_metadata.json")
    dataset_path_value = run_metadata.get("dataset_path")
    if not isinstance(dataset_path_value, str) or not dataset_path_value.strip():
        raise click.UsageError(f"dataset_path missing from {run_dir / 'run_metadata.json'}")
    dataset_root = Path(dataset_path_value).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = (run_dir.parents[1] / dataset_root).resolve()
    payload = _read_json(run_results_path)
    entries = payload.get("results")
    if not isinstance(entries, list):
        raise click.UsageError(f"Run results missing 'results' list: {run_results_path}")
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        task_id = str(entry.get("task_id") or "").strip()
        trial_name = str(entry.get("trial_name") or "").strip()
        task_hash = str(entry.get("task_hash") or "").strip()
        if not task_id or not trial_name:
            continue
        trial_dir = run_dir / task_id / trial_name
        scenario_path = trial_dir / "scenario_info.json"
        scenario_payload: dict[str, Any] = {}
        if scenario_path.exists():
            raw_scenario = _read_json(scenario_path)
            if isinstance(raw_scenario, dict):
                scenario_payload = raw_scenario
        embedded_scores = _entry_scores_payload(entry)
        question_row: dict[str, Any] = dict(scenario_payload)
        question_row["task_id"] = task_id
        question_row["trial_name"] = trial_name
        question_row["task_hash"] = task_hash
        question_row["question_root"] = str(dataset_root / task_id)
        question_row["question_id"] = str(
            question_row.get("question_id") or entry.get("question_id") or task_id
        ).strip()
        if embedded_scores is not None:
            question_row.setdefault("benchmark", embedded_scores.get("benchmark"))
            question_row.setdefault("variant", embedded_scores.get("variant"))
            question_row.setdefault(
                "exam_question_root", embedded_scores.get("exam_question_root")
            )
        benchmark = _normalize_benchmark_name(entry.get("benchmark"))
        if benchmark is not None:
            question_row["benchmark"] = benchmark
        variant = entry.get("variant")
        if isinstance(variant, str) and variant.strip():
            question_row["variant"] = variant.strip()
        exam_question_root = entry.get("exam_question_root")
        if isinstance(exam_question_root, str) and exam_question_root.strip():
            question_row["exam_question_root"] = exam_question_root.strip()
        multiple_choices = entry.get("multiple_choices")
        if isinstance(multiple_choices, list):
            question_row["multiple_choices"] = list(multiple_choices)
        if "benchmark" not in question_row or "variant" not in question_row:
            scenario_benchmark, scenario_variant = _scenario_benchmark_variant(
                scenario_payload
            )
            if scenario_benchmark and scenario_variant:
                question_row.setdefault("benchmark", scenario_benchmark)
                question_row.setdefault("variant", scenario_variant)
        question_row.setdefault("exam_question_root", "run_local")
        primary_question_id = question_row["question_id"]
        alias_keys: list[str] = []
        for candidate in (primary_question_id, task_id, task_hash):
            if isinstance(candidate, str):
                candidate_text = candidate.strip()
                if candidate_text and candidate_text not in alias_keys:
                    alias_keys.append(candidate_text)
        for idx_key, alias in enumerate(alias_keys):
            existing = index.get(alias)
            if existing is not None and existing is not question_row:
                label = "question_id" if idx_key == 0 else "question alias"
                click.echo(
                    f"Warning: duplicate {label} {alias} in {run_results_path}",
                    err=True,
                )
            index[alias] = question_row
    if not index:
        raise click.UsageError("No questions loaded from run-local metadata.")
    return index


def _ground_truth_description(metadata_payload: Mapping[str, Any]) -> str:
    ground_truth_information = metadata_payload.get("ground_truth_information")
    if isinstance(ground_truth_information, dict):
        description = str(ground_truth_information.get("model_description") or "").strip()
        shared_description = str(
            ground_truth_information.get("shared_description") or ""
        ).strip()
        if description and shared_description:
            return f"{description}\n\n{shared_description}"
        if description:
            return description
        if shared_description:
            return shared_description
        full_text = str(ground_truth_information.get("ground_truth_text") or "").strip()
        if full_text:
            return full_text
        return description
    return str(ground_truth_information or "").strip()


def _ground_truth_by_path_labels(
    question_row: Mapping[str, Any],
    question: Mapping[str, Any],
    metadata_payload: Mapping[str, Any],
) -> dict[str, str]:
    source_samples = question_sample_paths(metadata_payload, question=question, subset="test")
    if not source_samples:
        raise ValueError("Open eval-mode requires test_samples when deriving ground truth labels")
    answer_map: dict[str, str] = {}
    for source_sample in source_samples:
        try:
            label = label_for_question_sample(
                metadata_payload,
                question=question,
                sample_path=source_sample,
            )
        except Exception as exc:
            raise ValueError(
                f"Open eval-mode missing ground-truth label for sample {source_sample!r}: {exc}"
            ) from exc
        answer_map[f"test_samples/{Path(_normalize_path_key(source_sample)).name}"] = str(label).strip()
    return answer_map


def _resolve_open_answer(
    question_row: Mapping[str, Any],
    question: Mapping[str, Any],
    metadata_payload: Mapping[str, Any],
    agent_payload: Any,
) -> tuple[str, str]:
    explanation = str(question.get("ground_truth_explanation") or "").strip()
    if not explanation:
        explanation = str(question.get("ground_truth_context") or "").strip()
    if not explanation:
        question_text = question.get("question_text")
        if isinstance(question_text, dict):
            explanation = str(
                question_text.get("environment_description")
                or question_text.get("model_description")
                or ""
            ).strip()
    correct_answer = json.dumps(
        _ground_truth_by_path_labels(question_row, question, metadata_payload),
        ensure_ascii=True,
        sort_keys=True,
    )
    if not correct_answer:
        raise ValueError("Open eval-mode requires a resolved ground-truth answer")
    if not explanation:
        explanation = "No additional ground-truth explanation provided."
    if isinstance(agent_payload, dict) and "final_answer" in agent_payload:
        rendered_answer = str(agent_payload.get("final_answer") or "").strip()
    else:
        rendered_answer = json.dumps(agent_payload, ensure_ascii=True, sort_keys=True)
    if not rendered_answer:
        raise ValueError("Open eval-mode requires a non-empty agent answer")
    return correct_answer, explanation + "\n\nGround-truth answer: " + correct_answer


def _score_open_trial(
    *,
    entry: Mapping[str, Any],
    trial_dir: Path,
    question_row: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    scenario = _read_json(trial_dir / "scenario_info.json")
    question_id = str(question_row.get("question_id") or "").strip()
    if not question_id:
        raise ValueError("Missing question_id for open trial")
    _, metadata_payload, question = _load_tsenv_question_lookup(run_metadata, question_id)
    final_response_path = resolve_final_response_path(trial_dir)
    agent_payload = json.loads(final_response_path.read_text(encoding="utf-8"))
    choices = [str(choice) for choice in (question_row.get("multiple_choices") or [])]
    expected_samples = _test_samples(question_row)
    expected_ground_truth = _ground_truth_by_path_labels(question_row, question, metadata_payload)
    raw_predictions = _extract_open_prediction_payload(
        agent_payload,
        expected_samples=expected_samples,
    )
    sample_results: dict[str, dict[str, Any]] = {}
    for sample_path in expected_samples:
        sample_id = _sample_id_from_path(sample_path)
        ground_truth = str(expected_ground_truth.get(sample_path) or "").strip()
        raw_value = raw_predictions.get(sample_path)
        if raw_value is None:
            prediction = ""
            top1_correct = False
            shortlist_score = 0.0
            num_answers = 0.0
        else:
            raw_out = str(raw_value)
            parsed_prediction = _parse_open_prediction_value(raw_out, choices)
            if parsed_prediction is None:
                prediction = raw_out
                top1_correct = False
                shortlist_score = 0.0
            else:
                prediction = parsed_prediction
                top1_correct = parsed_prediction == ground_truth
                shortlist_score = 1.0 if top1_correct else 0.0
            num_answers = 1.0 if str(prediction).strip() else 0.0
        sample_results[sample_id] = _sample_result_entry(
            predictions=prediction,
            top1_correct=top1_correct,
            shortlist_score=shortlist_score,
            num_answers=num_answers,
            sample_type="test",
        )
    validated = _build_scores_payload(
        agent_run_id=str(run_metadata.get("run_id") or trial_dir.parents[1].name),
        sample_results=sample_results,
    )
    _write_json(trial_dir / "scores.json", validated)
    return validated


def _code_sample_results(solution_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    existing_sample_results = solution_payload.get("sample_results")
    if isinstance(existing_sample_results, dict) and existing_sample_results:
        return {
            str(key): dict(value)
            for key, value in existing_sample_results.items()
            if isinstance(value, dict)
            and str(value.get("sample_type") or "").strip() in {"test", "other"}
        }

    rows = solution_payload.get("results")
    if not isinstance(rows, list):
        raise ValueError("Code scoring did not return per-sample results.")
    sample_results: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_type = str(row.get("subset") or row.get("sample_type") or "").strip()
        if sample_type not in {"test", "other"}:
            continue
        sample_id = str(row.get("uuid") or row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        prediction = row.get("predictions")
        if prediction is None:
            prediction = row.get("predicted_label")
        if prediction is None:
            prediction = ""
        shortlist_score = float(row.get("shortlist_score") or 0.0)
        top1_correct = bool(row.get("top1_correct") is True)
        raw_num_answers = row.get("average_answers_per_entry")
        if raw_num_answers is None:
            raw_num_answers = row.get("num_answers")
        if raw_num_answers is None:
            raw_num_answers = _num_answers_from_prediction(prediction)
        sample_results[sample_id] = _sample_result_entry(
            predictions=prediction,
            top1_correct=top1_correct,
            shortlist_score=shortlist_score,
            num_answers=float(raw_num_answers or 0.0),
            sample_type=sample_type,
        )
    if not sample_results:
        raise ValueError("Code scoring did not return test or other sample results.")
    return sample_results


def _num_answers_from_prediction(prediction: object) -> float:
    if isinstance(prediction, list):
        return float(len(prediction))
    if str(prediction or "").strip():
        return 1.0
    return 0.0


def _score_code_rule_artifact(
    *,
    trial_dir: Path,
    question_row: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    resolved_noise_level: str,
    resolved_seed: int,
) -> dict[str, dict[str, Any]]:
    question_id = str(question_row.get("question_id") or "").strip()
    if not question_id:
        raise ValueError("Missing question_id for code trial")
    model_name, metadata_payload, question = _load_tsenv_question_lookup(
        run_metadata,
        question_id,
    )
    rule_path = trial_dir / "rule.py"
    model_record_path = TSENV_ROOT / model_name / "model_record.json"
    if not model_record_path.exists():
        raise FileNotFoundError(f"Missing model_record.json at {model_record_path}")
    if not (TSENV_ROOT / model_name / "sample_manifest.json").exists():
        raise FileNotFoundError(
            f"Missing sample_manifest.json at {TSENV_ROOT / model_name / 'sample_manifest.json'}"
        )

    choices = question_row.get("multiple_choices")
    allowed_choices = (
        [str(choice).strip() for choice in choices if str(choice).strip()]
        if isinstance(choices, list)
        else label_choices_from_payload(metadata_payload)
    )
    manifest_item = _question_manifest_item(metadata_payload, question)
    samples_by_type = {
        "test": [f"dataframes/{sample_id}.parquet" for sample_id in manifest_item.test_samples],
        "other": [f"dataframes/{sample_id}.parquet" for sample_id in manifest_item.other_samples],
    }
    sample_results: dict[str, dict[str, Any]] = {}
    for sample_type in ("test", "other"):
        for source_path in samples_by_type[sample_type]:
            source_path_text = _normalize_path_key(source_path)
            sample_uuid = Path(source_path_text).stem
            try:
                truth_label = label_for_question_sample(
                    metadata_payload,
                    question=question,
                    sample_path=source_path_text,
                )
                dataframe = materialize(
                    model_name,
                    sample_uuid,
                    resolved_noise_level,
                    int(resolved_seed),
                    tsenv_model_root=TSENV_ROOT / model_name,
                    models_root=MODELS_ROOT,
                )
                result = evaluate_rule_on_dataframe(
                    model_id=model_name,
                    child_df=dataframe,
                    truth_label=str(truth_label),
                    rule_path=rule_path,
                    models_root=MODELS_ROOT,
                    run_id=sample_uuid,
                    noise_level="none",
                    seed=0,
                    allowed_choices=allowed_choices,
                    data_path=TSENV_ROOT / model_name / "dataframes" / f"{sample_uuid}.parquet",
                )
                error = result.get("error")
                prediction = result.get("predicted_label") or ""
                sample_results[sample_uuid] = _sample_result_entry(
                    predictions=prediction,
                    top1_correct=bool(result.get("top1_correct")) and not error,
                    shortlist_score=float(result.get("shortlist_score") or 0.0),
                    num_answers=(
                        None
                        if error
                        else float(result.get("answer_count") or _num_answers_from_prediction(prediction))
                    ),
                    sample_type=sample_type,
                    error=str(error) if error else None,
                )
            except Exception as exc:
                sample_results[sample_uuid] = _sample_result_entry(
                    predictions="",
                    top1_correct=False,
                    shortlist_score=0.0,
                    num_answers=None,
                    sample_type=sample_type,
                    error=f"{type(exc).__name__}: {exc}",
                )
    if not sample_results:
        raise ValueError("Code scoring did not resolve any test or other samples.")
    return sample_results


def _score_non_open_trial(
    *,
    entry: Mapping[str, Any],
    trial_dir: Path,
    question_row: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    scores_path = trial_dir / "scores.json"
    question_root_value = question_row.get("question_root")
    if not isinstance(question_root_value, str) or not question_root_value:
        raise ValueError(f"Missing question_root for {trial_dir}")
    question_root = Path(question_root_value)
    benchmark = question_row.get("benchmark")
    variant = question_row.get("variant")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise ValueError(f"Missing benchmark for question {question_row.get('question_id')!r}")
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError(f"Missing variant for question {question_row.get('question_id')!r}")
    benchmark_id = f"{benchmark}_{variant}"
    benchmark_root = benchmark_root_from_label(benchmark)
    agent_name = run_metadata.get("agent_name") or run_metadata.get("agent")
    if not isinstance(agent_name, str) or not agent_name:
        raise KeyError("agent_name missing from run_metadata.json")
    model_name = run_metadata.get("model_name") or run_metadata.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise KeyError("model_name missing from run_metadata.json")
    eval_mode = normalize_tsenv_eval_mode(question_row.get("eval_mode"))
    sample_results: dict[str, dict[str, Any]] = {}

    if not (is_classification_run(benchmark_id) or is_equation_run(benchmark_id)):
        raise NotImplementedError(f"Unsupported benchmark category for {benchmark_id}.")

    if is_simbench_cls_benchmark(benchmark_root) and eval_mode == "code":
        final_response_path = resolve_final_response_path(trial_dir)
        agent_json_answer: dict[str, Any] = {}
        if final_response_path.exists():
            raw_agent_json_answer = json.loads(final_response_path.read_text(encoding="utf-8"))
            if not isinstance(raw_agent_json_answer, dict):
                raise ValueError(f"{final_response_path} must contain a JSON object")
            agent_json_answer = raw_agent_json_answer
        rule_file = str(agent_json_answer.get("rule_file") or "").strip()
        if not rule_file and (trial_dir / "rule.py").exists():
            rule_file = "rule.py"
        if not rule_file:
            raise ValueError(
                "Code response must create rule.py or include non-empty rule_file in agentic-final-response.json."
            )
        if Path(rule_file).name != "rule.py":
            raise ValueError(
                f"Unsupported rule_file value {rule_file!r}; expected 'rule.py'."
            )
        agent_rule_path = trial_dir / "rule.py"
        if not agent_rule_path.exists():
            raise FileNotFoundError(f"Missing persisted rule artifact at {agent_rule_path}")
        resolved_noise_level = resolve_run_noise_level(
            scenario=question_row,
            run_metadata=run_metadata,
            override=None,
        )
        resolved_seed = resolve_run_base_seed(
            scenario=question_row,
            run_metadata=run_metadata,
            override=None,
        )
        sample_results = _score_code_rule_artifact(
            trial_dir=trial_dir,
            question_row=question_row,
            run_metadata=run_metadata,
            resolved_noise_level=resolved_noise_level,
            resolved_seed=resolved_seed,
        )
        validated = _build_scores_payload(
            agent_run_id=str(run_metadata.get("run_id") or trial_dir.parents[1].name),
            sample_results=sample_results,
            include_final_metric_other=True,
        )
        _write_json(scores_path, validated)
        return validated
    elif is_simbench_cls_benchmark(benchmark_root) and eval_mode == "direct":
        test_samples = _test_samples(question_row)
        source_samples = _test_source_samples(question_row)
        prediction_map, is_correct_format = _load_direct_prediction_map(
            trial_dir=trial_dir,
            expected_samples=test_samples,
        )
        label_map = _test_label_map(question_row)
        choices = [str(choice) for choice in (question_row.get("multiple_choices") or [])]
        for idx, rel_path in enumerate(test_samples):
            normalized_rel_path = _normalize_path_key(rel_path)
            predicted_raw = prediction_map.get(normalized_rel_path)
            if predicted_raw is None:
                raise ValueError(f"Missing direct prediction for {rel_path}")
            predicted = _resolve_classification_choices(predicted_raw, choices)
            source_path = source_samples[idx] if idx < len(source_samples) else rel_path
            truth = str(
                _resolve_expected_label(label_map, source_path) or label_map.get(rel_path) or ""
            ).strip()
            if not truth:
                raise ValueError(f"Missing ground-truth label for {rel_path}")
            sample_score = _score_direct_prediction(predicted, truth, choices)
            sample_results[_sample_id_from_path(rel_path)] = _sample_result_entry(
                predictions=predicted[0] if len(predicted) == 1 else list(predicted),
                top1_correct=bool(predicted) and predicted[0] == truth,
                shortlist_score=sample_score,
                num_answers=float(len(predicted)),
                sample_type="test",
            )
    else:
        if not is_simbench_cls_benchmark(benchmark_root):
            raise ValueError(
                f"evaluate_accuracy.py no longer supports legacy scalar-answer scoring for benchmark {benchmark!r}"
            )
        raise ValueError(
            f"Unsupported tsENV non-open eval_mode {eval_mode!r}; expected 'direct' or 'code'"
        )

    validated = _build_scores_payload(
        agent_run_id=str(run_metadata.get("run_id") or trial_dir.parents[1].name),
        sample_results=sample_results,
        is_correct_format=is_correct_format if eval_mode == "direct" else None,
    )
    _write_json(scores_path, validated)
    return validated


def _summarize_scored_trial(
    *,
    entry: Mapping[str, Any],
    scenario: Mapping[str, Any],
    scores_payload: Mapping[str, Any],
) -> dict[str, Any]:
    eval_mode = normalize_tsenv_eval_mode(scenario.get("eval_mode"))
    sample_entries = _sample_entries_from_scores_payload(scores_payload)
    metric_accuracy = float(
        (
            (scores_payload.get("final_metric_test") or {}).get(
                "average_top1_accuracy"
            )
            or 0.0
        )
    )
    top1_counts: Counter[str] = Counter()
    if sample_entries:
        test_entries = [
            payload
            for payload in sample_entries.values()
            if str(payload.get("sample_type") or "").strip() == "test"
        ]
        for payload in test_entries:
            top1_counts["correct" if payload.get("top1_correct") is True else "wrong"] += 1
        sample_count = len(test_entries)
        evaluable_answers = int(sample_count)
        correct_answers = float(metric_accuracy) * float(sample_count)
    elif eval_mode == "code":
        sample_count = len(_test_samples(scenario))
        evaluable_answers = int(sample_count)
        correct_answers = float(metric_accuracy) * float(sample_count)
    else:
        sample_count = 0
        evaluable_answers = 0
        correct_answers = 0.0
    return {
        "task_id": str(entry["task_id"]),
        "trial_name": str(entry["trial_name"]),
        "question_id": str(scenario.get("question_id") or entry.get("question_id") or entry.get("task_hash") or ""),
        "eval_mode": eval_mode,
        "desc_level": scenario.get("desc_level") or scenario.get("context"),
        "context": scenario.get("context") or scenario.get("desc_level"),
        "shot": scenario.get("shot"),
        "recipe": scenario.get("recipe"),
        "sample_count": sample_count,
        "evaluable_answers": evaluable_answers,
        "correct_answers": correct_answers,
        "accuracy": metric_accuracy if evaluable_answers else None,
        "top1_counts": dict(top1_counts),
    }


@click.command()
@click.argument("agentic_run_id", required=False)
@click.option(
    "--path",
    "run_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
)
def main(agentic_run_id: str | None, run_path: Path | None) -> None:
    run_dir = _resolve_run_dir(agentic_run_id, run_path)
    run_metadata = _read_json(run_dir / "run_metadata.json")
    resolved_run_id = str(run_metadata.get("run_id") or run_dir.name).strip()
    run_results = _read_json(run_dir / "results.json")
    entries = run_results.get("results")
    if not isinstance(entries, list):
        raise click.ClickException(f"Run results missing results list: {run_dir / 'results.json'}")
    question_index = load_question_index(run_dir)
    trial_summaries: list[dict[str, Any]] = []
    mode_counts: Counter[str] = Counter()
    top1_counts: Counter[str] = Counter()
    total_evaluable_answers = 0
    total_correct_answers = 0.0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        task_id = str(entry.get("task_id") or "").strip()
        trial_name = str(entry.get("trial_name") or "").strip()
        if not task_id or not trial_name:
            continue
        trial_dir = run_dir / task_id / trial_name
        scenario_path = trial_dir / "scenario_info.json"
        if not scenario_path.exists():
            raise click.ClickException(f"missing scenario_info.json at {scenario_path}")
        scenario = _read_json(scenario_path)
        eval_mode = _normalize_eval_mode(scenario.get("eval_mode"))
        mode_counts[eval_mode] += 1
        question_key = _question_key_from_entry(entry, scenario)
        question_row = question_index.get(question_key)
        if question_row is None:
            raise click.ClickException(
                f"question metadata not found for key {question_key!r}"
            )
        try:
            if eval_mode == "open-ended":
                scores_payload = _score_open_trial(
                    entry=entry,
                    trial_dir=trial_dir,
                    question_row=question_row,
                    run_metadata=run_metadata,
                )
            else:
                scores_payload = _score_non_open_trial(
                    entry=entry,
                    trial_dir=trial_dir,
                    question_row=question_row,
                    run_metadata=run_metadata,
                )
            summary = _summarize_scored_trial(
                entry=entry,
                scenario=scenario,
                scores_payload=scores_payload,
            )
            summary["status"] = "evaluated"
            summary["scores_path"] = str(trial_dir / "scores.json")
            trial_summaries.append(summary)
            top1_counts.update(summary.get("top1_counts") or {})
            total_evaluable_answers += int(summary["evaluable_answers"])
            total_correct_answers += float(summary["correct_answers"])
        except Exception as exc:
            trial_summaries.append(
                {
                    "task_id": task_id,
                    "trial_name": trial_name,
                    "question_id": str(
                        scenario.get("question_id")
                        or entry.get("question_id")
                        or entry.get("task_hash")
                        or ""
                    ),
                    "eval_mode": eval_mode,
                    "desc_level": scenario.get("desc_level") or scenario.get("context"),
                    "context": scenario.get("context") or scenario.get("desc_level"),
                    "shot": scenario.get("shot"),
                    "recipe": scenario.get("recipe"),
                    "status": "skipped",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            click.echo(
                f"Skipped {task_id}/{trial_name}: {type(exc).__name__}: {exc}",
                err=True,
            )

    accuracy_summary = {
        "agentic_run_id": resolved_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_metadata": run_metadata,
        "total_trials": len([entry for entry in entries if isinstance(entry, dict)]),
        "evaluated_trials": sum(1 for item in trial_summaries if item.get("status") == "evaluated"),
        "errored_trials": sum(1 for item in trial_summaries if item.get("status") == "error"),
        "skipped_trials": sum(1 for item in trial_summaries if item.get("status") == "skipped"),
        "mode_counts": dict(mode_counts),
        "top1_counts": dict(top1_counts),
        "total_evaluable_answers": total_evaluable_answers,
        "total_correct_answers": total_correct_answers,
        "batch_accuracy": (total_correct_answers / total_evaluable_answers) if total_evaluable_answers else None,
        "trials": trial_summaries,
    }
    output_path = run_dir / ACCURACY_SUMMARY_NAME
    _write_json(output_path, accuracy_summary)
    for summary in trial_summaries:
        if summary.get("status") != "evaluated":
            continue
        scores_path = str(summary.get("scores_path") or "").strip()
        if not scores_path:
            continue
        click.echo(f"Saved scores.json to {scores_path}")


if __name__ == "__main__":
    main()
