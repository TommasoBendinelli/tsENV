#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import click

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.agentic_rollout_paths import resolve_final_response_path
from shared.model_noise_adder import normalize_noise_profile
from shared.tsenv_eval_mode import normalize_tsenv_eval_mode
from shared.tsenv_metadata import (
    label_for_question_sample,
    load_metadata_payload,
    metadata_questions_by_id,
    question_sample_paths,
    resolve_tsenv_payload_path,
)
from shared.tsenv_task_materialization import materialize_dataframe as materialize
from workflows.metrics.evaluate_rule import evaluate_rule_on_dataframe

RUNS_ROOT = root_dir / "terminal-bench" / "runs"
TSENV_ROOT = root_dir / "tsENV_questions"
DEFAULT_EXTRA_SEED_OFFSETS = (101, 102, 103, 104)


def _resolve_runs_root(runs_root: Optional[Path]) -> Path:
    if runs_root is not None:
        return Path(runs_root).expanduser().resolve()
    return RUNS_ROOT


def _resolve_tsenv_root(tsenv_root: Optional[Path]) -> Path:
    if tsenv_root is not None:
        return Path(tsenv_root).expanduser().resolve()
    return TSENV_ROOT


def _resolve_models_root(*, models_root: Optional[Path], tsenv_root: Path) -> Path:
    if models_root is not None:
        return Path(models_root).expanduser().resolve()
    sibling_root = (Path(tsenv_root).expanduser().resolve().parent / "models" / "simulink").resolve()
    if sibling_root.exists():
        return sibling_root
    return (root_dir / "models" / "simulink").resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


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


def _find_single_code_trial(
    *,
    run_dir: Path,
    run_results: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    entries = run_results.get("results")
    if not isinstance(entries, list):
        raise click.ClickException(f"Run results missing 'results' list: {run_dir / 'results.json'}")

    code_trials: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []
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
            continue
        scenario = _read_json(scenario_path)
        if normalize_tsenv_eval_mode(scenario.get("eval_mode")) != "code":
            continue
        code_trials.append((entry, trial_dir, scenario))

    if not code_trials:
        raise click.ClickException(f"No code-mode trial found under {run_dir}.")
    if len(code_trials) > 1:
        labels = [f"{str(entry.get('task_id') or '').strip()}/{trial_dir.name}" for entry, trial_dir, _ in code_trials]
        raise click.ClickException(
            "Expected exactly one code-mode trial, found multiple: " + ", ".join(labels)
        )
    return code_trials[0]


def _resolve_run_trial_context(
    *,
    run_id: str,
    runs_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    run_dir = (runs_root / str(run_id).strip()).resolve()
    if not run_dir.exists():
        raise click.ClickException(f"Run directory not found: {run_dir}")
    run_metadata = _read_json(run_dir / "run_metadata.json")
    run_results = _read_json(run_dir / "results.json")
    entry, trial_dir, scenario = _find_single_code_trial(run_dir=run_dir, run_results=run_results)
    return run_dir, run_metadata, run_results, entry, trial_dir, scenario


def _load_question_lookup(
    *,
    run_metadata: dict[str, Any],
    question_id: str,
    tsenv_root: Path,
) -> tuple[str, Path, dict[str, Any], dict[str, Any]]:
    dataset_path_value = run_metadata.get("dataset_path")
    dataset_path = Path(str(dataset_path_value or "")).expanduser().resolve()
    model_name = dataset_path.parent.name
    metadata_path = resolve_tsenv_payload_path(tsenv_root / model_name)
    payload = load_metadata_payload(metadata_path)
    questions = metadata_questions_by_id(payload)
    question = questions.get(question_id)
    if question is None:
        raise KeyError(f"question_id {question_id!r} not found in {metadata_path}")
    return model_name, metadata_path, payload, question


def _candidate_seed_noise_mappings(
    *,
    scenario: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    recipe = scenario.get("recipe")
    if isinstance(recipe, dict):
        resolved = recipe.get("resolved")
        if isinstance(resolved, dict):
            candidates.append(resolved)
    recipe_info = scenario.get("recipe_info")
    if isinstance(recipe_info, dict):
        candidates.append(recipe_info)
    candidates.append(scenario)
    candidates.append(run_metadata)
    return candidates


def _noise_level_from_mapping(item: Mapping[str, Any]) -> Optional[str]:
    profile_name = str(item.get("noise_profile") or "").strip().lower()
    if profile_name:
        if profile_name == "medium" or profile_name.endswith("_medium"):
            return "low"
        for level in ("none", "low", "high"):
            if profile_name == level or profile_name.endswith(f"_{level}"):
                return level
    values: list[float] = []
    for key in ("noise_local", "noise_global", "noise_abs"):
        if key not in item:
            continue
        try:
            values.append(float(item.get(key, 0.0)))
        except (TypeError, ValueError) as exc:
            raise click.ClickException(
                f"Invalid {key} value in scenario metadata: {item.get(key)!r}"
            ) from exc
    if not values:
        return None
    if max(values) <= 0.0:
        return "none"
    if max(values) <= 0.01:
        return "low"
    return "high"


def resolve_run_noise_level(
    *,
    scenario: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    override: Optional[str],
) -> str:
    if override is not None:
        return normalize_noise_profile(override)
    for candidate in _candidate_seed_noise_mappings(scenario=scenario, run_metadata=run_metadata):
        resolved = _noise_level_from_mapping(candidate)
        if resolved is not None:
            return normalize_noise_profile(resolved)
    raise click.ClickException(
        "Could not resolve noise_profile for this run from scenario_info.json or run_metadata.json."
    )


def resolve_run_base_seed(
    *,
    scenario: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    override: Optional[int],
) -> int:
    if override is not None:
        return int(override)
    for candidate in _candidate_seed_noise_mappings(scenario=scenario, run_metadata=run_metadata):
        for key in ("noise_seed", "seed"):
            if key not in candidate:
                continue
            try:
                return int(candidate.get(key))
            except (TypeError, ValueError) as exc:
                raise click.ClickException(
                    f"Invalid {key} value in scenario metadata: {candidate.get(key)!r}"
                ) from exc
    raise click.ClickException(
        "Could not resolve base seed for this run from scenario_info.json or run_metadata.json."
    )


def _resolve_seed_schedule(
    *,
    base_seed: int,
    extra_seed_offsets: Sequence[int],
) -> tuple[list[int], list[int]]:
    offsets: list[int] = [0]
    for raw in extra_seed_offsets:
        offset = int(raw)
        if offset not in offsets:
            offsets.append(offset)
    return offsets, [int(base_seed) + int(offset) for offset in offsets]


def _sample_uuid_from_path(*, source_sample_path: str, sample_path: str) -> str:
    source_name = Path(str(source_sample_path or "")).stem
    if source_name:
        return source_name
    return Path(str(sample_path or "")).stem


def _subset_for_source_path(
    *,
    source_sample_path: str,
    train_samples: set[str],
    test_samples: set[str],
) -> str:
    normalized = _normalize_path_key(source_sample_path)
    if normalized in train_samples:
        return "train"
    if normalized in test_samples:
        return "test"
    return "other"


def _documented_subset_metric(
    stats: dict[str, float | int] | None,
) -> dict[str, float]:
    if not stats or not int(stats.get("evaluable_count") or 0):
        return {
            "average_top1_accuracy": 0.0,
            "average_shortlist_score": 0.0,
            "average_num_answers": 0.0,
        }
    evaluable_count = float(stats["evaluable_count"])
    return {
        "average_top1_accuracy": float(stats["correct_count"]) / evaluable_count,
        "average_shortlist_score": float(stats["shortlist_score_sum"])
        / evaluable_count,
        "average_num_answers": float(stats["answer_count_sum"]) / evaluable_count,
    }


def _prediction_list(predictions: object) -> list[str]:
    if isinstance(predictions, list):
        return [str(item).strip() for item in predictions if str(item).strip()]
    text = str(predictions or "").strip()
    return [text] if text else []


def evaluate_rule_solution(
    run_id: str,
    seed: int,
    noise_level: str,
    *,
    runs_root: Optional[Path] = None,
    tsenv_root: Optional[Path] = None,
    models_root: Optional[Path] = None,
    extra_seed_offsets: Sequence[int] = DEFAULT_EXTRA_SEED_OFFSETS,
) -> dict[str, object]:
    resolved_runs_root = _resolve_runs_root(runs_root)
    resolved_tsenv_root = _resolve_tsenv_root(tsenv_root)
    resolved_models_root = _resolve_models_root(models_root=models_root, tsenv_root=resolved_tsenv_root)
    resolved_noise_level = normalize_noise_profile(noise_level)
    resolved_seed = int(seed)
    _, run_metadata, _run_results, entry, trial_dir, scenario = _resolve_run_trial_context(
        run_id=run_id,
        runs_root=resolved_runs_root,
    )

    rule_path = trial_dir / "rule.py"
    if not rule_path.exists():
        raise click.ClickException(f"Missing persisted rule artifact at {rule_path}")

    final_response_path = resolve_final_response_path(trial_dir)
    agent_payload = _read_json(final_response_path)
    rule_file = str(agent_payload.get("rule_file") or "").strip()
    if Path(rule_file).name != "rule.py":
        raise click.ClickException(
            f"Unsupported rule_file value {rule_file!r}; expected 'rule.py'."
        )

    question_id = _question_key_from_entry(entry, scenario)
    model_name, _metadata_path, metadata_payload, question = _load_question_lookup(
        run_metadata=run_metadata,
        question_id=question_id,
        tsenv_root=resolved_tsenv_root,
    )
    choices = scenario.get("multiple_choices")
    allowed_choices = [str(choice).strip() for choice in choices] if isinstance(choices, list) else None
    question_train_samples = {
        _normalize_path_key(str(item))
        for item in question_sample_paths(metadata_payload, question=question, subset="train")
        if str(item).strip()
    }
    question_test_samples = {
        _normalize_path_key(str(item))
        for item in question_sample_paths(metadata_payload, question=question, subset="test")
        if str(item).strip()
    }
    if not question_test_samples:
        raise click.ClickException("No code test_samples found for evaluation.")
    _, seeds = _resolve_seed_schedule(
        base_seed=resolved_seed,
        extra_seed_offsets=extra_seed_offsets,
    )
    normalized_ground_truth_by_path: dict[str, str] = {}
    for sample_path in question_sample_paths(metadata_payload, question=question):
        path_key = _normalize_path_key(sample_path)
        if not path_key:
            continue
        normalized_ground_truth_by_path[path_key] = label_for_question_sample(
            metadata_payload,
            question=question,
            sample_path=path_key,
        )

    results: list[dict[str, object]] = []
    evaluable_count = 0
    correct_count = 0
    shortlist_score_sum = 0.0
    answer_count_sum = 0.0
    single_seed_test_evaluable_count = 0
    single_seed_test_correct_count = 0
    single_seed_test_shortlist_score_sum = 0.0
    single_seed_test_answer_count_sum = 0.0
    subset_stats: dict[str, dict[str, float | int]] = {}
    for source_rel_text, expected_label in sorted(normalized_ground_truth_by_path.items()):
        subset = _subset_for_source_path(
            source_sample_path=source_rel_text,
            train_samples=question_train_samples,
            test_samples=question_test_samples,
        )
        sample_uuid = _sample_uuid_from_path(
            source_sample_path=source_rel_text,
            sample_path=source_rel_text,
        )
        stats = subset_stats.setdefault(
            subset,
            {
                "sample_count": 0,
                "rerun_count": 0,
                "evaluable_count": 0,
                "correct_count": 0,
                "shortlist_score_sum": 0.0,
                "answer_count_sum": 0.0,
            },
        )
        stats["sample_count"] += 1

        row: dict[str, object] = {
            "uuid": sample_uuid,
            "subset": subset,
            "ground_truth_label": expected_label,
            "accuracy": None,
        }

        try:
            sample_correct_count = 0
            sample_evaluable_count = 0
            sample_shortlist_score_sum = 0.0
            sample_answer_count_sum = 0.0
            representative_prediction: object = None
            representative_is_correct: bool | None = None
            for rerun_seed in seeds:
                dataframe = materialize(
                    model_name,
                    sample_uuid,
                    resolved_noise_level,
                    rerun_seed,
                    tsenv_model_root=resolved_tsenv_root / model_name,
                    models_root=resolved_models_root,
                )
                payload = evaluate_rule_on_dataframe(
                    model_id=model_name,
                    child_df=dataframe,
                    truth_label=str(expected_label),
                    rule_path=rule_path,
                    models_root=resolved_models_root,
                    run_id=sample_uuid,
                    noise_level="none",
                    seed=0,
                    allowed_choices=allowed_choices,
                    data_path=(resolved_tsenv_root / model_name / "dataframes" / f"{sample_uuid}.parquet"),
                )
                sample_shortlist_score = float(
                    payload.get("shortlist_score") or 0.0
                )
                sample_answer_count = int(payload.get("answer_count") or 0)
                if representative_prediction is None or rerun_seed == resolved_seed:
                    representative_prediction = payload.get("predicted_label")
                    representative_is_correct = bool(payload.get("is_correct"))
                evaluable_count += 1
                sample_evaluable_count += 1
                shortlist_score_sum += sample_shortlist_score
                answer_count_sum += float(sample_answer_count)
                sample_shortlist_score_sum += sample_shortlist_score
                sample_answer_count_sum += float(sample_answer_count)
                stats["rerun_count"] += 1
                stats["evaluable_count"] += 1
                stats["shortlist_score_sum"] += sample_shortlist_score
                stats["answer_count_sum"] += float(sample_answer_count)
                if subset == "test" and rerun_seed == resolved_seed:
                    single_seed_test_evaluable_count += 1
                    single_seed_test_shortlist_score_sum += sample_shortlist_score
                    single_seed_test_answer_count_sum += float(sample_answer_count)
                if bool(payload.get("is_correct")):
                    sample_correct_count += 1
                    correct_count += 1
                    stats["correct_count"] += 1
                    if subset == "test" and rerun_seed == resolved_seed:
                        single_seed_test_correct_count += 1
            row["accuracy"] = (
                float(sample_correct_count) / float(sample_evaluable_count)
                if sample_evaluable_count
                else None
            )
            row["shortlist_score"] = (
                float(sample_shortlist_score_sum) / float(sample_evaluable_count)
                if sample_evaluable_count
                else None
            )
            row["average_answers_per_entry"] = (
                float(sample_answer_count_sum) / float(sample_evaluable_count)
                if sample_evaluable_count
                else None
            )
            row["predictions"] = representative_prediction or ""
            row["top1_correct"] = representative_is_correct is True
        except Exception:
            pass
        results.append(row)

    accuracy_value = (float(correct_count) / float(evaluable_count)) if evaluable_count else None
    subset_accuracy = {
        subset: {
            "sample_count": int(stats["sample_count"]),
            "rerun_count": int(stats["rerun_count"]),
            "evaluable_count": int(stats["evaluable_count"]),
            "correct_count": int(stats["correct_count"]),
            "accuracy": (
                float(stats["correct_count"]) / float(stats["evaluable_count"])
                if int(stats["evaluable_count"])
                else None
            ),
            "shortlist_score": (
                float(stats["shortlist_score_sum"]) / float(stats["evaluable_count"])
                if int(stats["evaluable_count"])
                else None
            ),
        }
        for subset, stats in subset_stats.items()
        if int(stats["sample_count"]) > 0
    }
    final_metric_accuracy = (
        float(single_seed_test_correct_count) / float(single_seed_test_evaluable_count)
        if single_seed_test_evaluable_count
        else 0.0
    )
    final_metric_shortlist_score = (
        float(single_seed_test_shortlist_score_sum)
        / float(single_seed_test_evaluable_count)
        if single_seed_test_evaluable_count
        else 0.0
    )
    final_metric_average_answers_per_entry = (
        float(single_seed_test_answer_count_sum)
        / float(single_seed_test_evaluable_count)
        if single_seed_test_evaluable_count
        else 0.0
    )
    sample_results = {
        str(row["uuid"]): {
            "predictions": _prediction_list(row.get("predictions")),
            "top1_correct": bool(row.get("top1_correct") is True),
            "shortlist_score": float(row.get("shortlist_score") or 0.0),
            "num_answers": float(row.get("average_answers_per_entry") or 0.0),
            "sample_type": str(row.get("subset")),
        }
        for row in results
        if str(row.get("subset") or "") in {"test", "other"}
    }
    final_metric_other = _documented_subset_metric(subset_stats.get("other"))
    return {
        "agentic_run_id": str(run_id).strip(),
        "model_id": model_name,
        "noise_level": resolved_noise_level,
        "final_metric_test": {
            "average_top1_accuracy": final_metric_accuracy,
            "average_shortlist_score": final_metric_shortlist_score,
            "average_num_answers": final_metric_average_answers_per_entry,
        },
        "final_metric_other": final_metric_other,
        "sample_results": sample_results,
        "final_metric": {
            "accuracy": final_metric_accuracy,
            "shortlist_score": final_metric_shortlist_score,
            "average_answers_per_entry": final_metric_average_answers_per_entry,
        },
        "sample_count": len(normalized_ground_truth_by_path),
        "rerun_count": int(evaluable_count),
        "evaluable_count": int(evaluable_count),
        "correct_count": int(correct_count),
        "accuracy": accuracy_value,
        "shortlist_score": (
            float(shortlist_score_sum) / float(evaluable_count)
            if evaluable_count
            else None
        ),
        "average_answers_per_entry": (
            float(answer_count_sum) / float(evaluable_count)
            if evaluable_count
            else None
        ),
        "subset_accuracy": subset_accuracy,
        "results": results,
    }


def evaluate_on_run_params(
    agentic_run_id: str,
    *,
    runs_root: Optional[Path] = None,
    tsenv_root: Optional[Path] = None,
    models_root: Optional[Path] = None,
    base_seed: Optional[int] = None,
    noise_level: Optional[str] = None,
    extra_seed_offsets: Sequence[int] = DEFAULT_EXTRA_SEED_OFFSETS,
) -> dict[str, object]:
    resolved_runs_root = _resolve_runs_root(runs_root)
    _, run_metadata, _run_results, _entry, _trial_dir, scenario = _resolve_run_trial_context(
        run_id=agentic_run_id,
        runs_root=resolved_runs_root,
    )
    resolved_noise_level = resolve_run_noise_level(
        scenario=scenario,
        run_metadata=run_metadata,
        override=noise_level,
    )
    resolved_base_seed = resolve_run_base_seed(
        scenario=scenario,
        run_metadata=run_metadata,
        override=base_seed,
    )
    return evaluate_rule_solution(
        agentic_run_id,
        resolved_base_seed,
        resolved_noise_level,
        runs_root=resolved_runs_root,
        tsenv_root=tsenv_root,
        models_root=models_root,
        extra_seed_offsets=extra_seed_offsets,
    )


@click.command(help="Evaluate one agentic run by applying its exported rule.py to all ground-truth paths.")
@click.argument("agentic_run_id")
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override terminal-bench runs root.",
)
@click.option(
    "--tsenv-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override tsENV metadata root.",
)
@click.option(
    "--models-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override simulink models root.",
)
@click.option(
    "--base-seed",
    type=int,
    default=None,
    help="Override the base seed recorded on the run.",
)
@click.option(
    "--noise-level",
    type=str,
    default=None,
    help="Override the recorded run noise level.",
)
@click.option(
    "--extra-seed-offset",
    "extra_seed_offsets",
    type=int,
    multiple=True,
    default=DEFAULT_EXTRA_SEED_OFFSETS,
    show_default=True,
    help="Extra offsets added to the base seed. Base seed itself is always included.",
)
@click.option(
    "--json-indent",
    type=int,
    default=2,
    show_default=True,
    help="Indent level for JSON output.",
)
def main(
    agentic_run_id: str,
    runs_root: Optional[Path],
    tsenv_root: Optional[Path],
    models_root: Optional[Path],
    base_seed: Optional[int],
    noise_level: Optional[str],
    extra_seed_offsets: tuple[int, ...],
    json_indent: int,
) -> None:
    payload = evaluate_on_run_params(
        agentic_run_id,
        runs_root=runs_root,
        tsenv_root=tsenv_root,
        models_root=models_root,
        base_seed=base_seed,
        noise_level=noise_level,
        extra_seed_offsets=extra_seed_offsets,
    )
    click.echo(json.dumps(payload, indent=int(json_indent)))


if __name__ == "__main__":
    main()
