#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import click
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from shared.exam_questions_paths import TSENV_QUESTIONS_ROOT, resolve_exam_questions_output_dir  # noqa: E402
from shared.interface.distribution_json import load_experiment_config_json  # noqa: E402
from shared.interface.model_record_json import (  # noqa: E402
    load_model_record_json,
    load_model_run_specs_json,
)
from shared.interface.sample_manifest_json import (  # noqa: E402
    compute_train_test_sample_hash,
    validate_sample_manifest_payload,
)
from shared.interface.similarity_metrics_json import load_similarity_metrics_json  # noqa: E402
from shared.run_artifacts import resolve_model_record_path, resolve_runs_root, resolve_similarity_metrics_path  # noqa: E402
from shared.model_run_specs_runtime import build_model_record_registry  # noqa: E402
from shared.tsenv_combinations import (  # noqa: E402
    ALL_POSSIBLE_COMBINATIONS_PATH,
    ShotSelection,
    TIME0_BASELINE_LABEL,
    group_shot_selections,
    load_combination_rows,
)
from workflows.baselines.baseline_classifier_few_shot import (  # noqa: E402
    DEFAULT_TSENV_CLASSIFIERS,
    run_baseline_classifier,
)

MAX_RESAMPLE_ATTEMPTS = 500


@dataclass(frozen=True)
class _Candidate:
    intervention_uuid: str
    baseline_uuid: str
    label: str


@dataclass(frozen=True)
class _Shot:
    id: str
    test_set_slug: str
    is_adversarial: bool | None
    train_samples: int
    test_samples: int
    seeds: tuple[int, ...]


def _load_sampling_recipe(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    axes = payload.get("axes")
    evaluation = payload.get("evaluation")
    if not isinstance(axes, dict) or not isinstance(evaluation, dict):
        raise ValueError(f"Legacy recipe at {path} must contain axes and evaluation objects")
    test_count = int(evaluation.get("test_sample_count"))
    seeds = tuple(int(seed) for seed in evaluation.get("seeds") or [])
    shots_payload = axes.get("shot")
    if not isinstance(shots_payload, list) or not shots_payload:
        raise ValueError(f"Legacy recipe at {path} must contain a non-empty axes.shot list")
    shots: List[_Shot] = []
    for idx, shot in enumerate(shots_payload):
        if not isinstance(shot, dict):
            raise ValueError(f"recipe.axes.shot[{idx}] must be an object")
        train_samples = int(shot.get("train_samples"))
        if train_samples < 0:
            raise ValueError(f"recipe.axes.shot[{idx}].train_samples must be an integer >= 0")
        is_adversarial = None if train_samples == 0 else bool(shot.get("is_adversarial", False))
        shots.append(
            _Shot(
                id=str(shot.get("id") or "").strip(),
                test_set_slug=str(shot.get("test_set_slug") or "home").strip(),
                is_adversarial=is_adversarial,
                train_samples=train_samples,
                test_samples=test_count,
                seeds=seeds,
            )
        )
    return {
        "test_sample_count": test_count,
        "seeds": list(seeds),
        "shots": shots,
    }


def _find_run_data_path(runs_dir: Path, run_id: str) -> Optional[Path]:
    run_dir = runs_dir / run_id
    for name in ("data.parquet", "data.csv"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _read_matrix(
    runs_dir: Path,
    run_id: str,
    signals: Sequence[str],
    cache: Dict[tuple[str, tuple[str, ...]], Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    key = (run_id, tuple(signals))
    if key in cache:
        return cache[key]
    path = _find_run_data_path(runs_dir, run_id)
    if path is None:
        cache[key] = None
        return None
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "time" in df.columns:
        df = df.sort_values("time")
    if any(signal not in df.columns for signal in signals):
        cache[key] = None
        return None
    mat = df[list(signals)].to_numpy(dtype=np.float32, copy=True)
    if mat.size == 0:
        cache[key] = None
        return None
    if np.isnan(mat).any():
        means = np.nanmean(mat, axis=0)
        inds = np.where(np.isnan(mat))
        mat[inds] = np.take(means, inds[1])
    cache[key] = mat
    return mat


def _infer_signals(runs_dir: Path, registry: Sequence[Mapping[str, Any]]) -> List[str]:
    for run in registry:
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or run.get("baseline_uuid") or "").strip()
        path = _find_run_data_path(runs_dir, run_id) if run_id else None
        if path is None:
            continue
        df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        signals = [
            str(col).strip()
            for col in df.columns
            if str(col).strip() and str(col).strip().lower() not in {"time", "timestamp", "index", "idx"}
        ]
        if signals:
            return list(dict.fromkeys(signals))
    raise ValueError(f"Unable to infer signals from {runs_dir}")


def _load_class_labels_from_experiment_config(path: Path) -> tuple[List[str], bool]:
    config = load_experiment_config_json(path)
    labels = list(config.intervention_parameter_names)
    if not labels:
        raise ValueError(f"experiment_config.json has no exposed_variables.parameters at {path}")
    return labels, True


def _load_eligible_runs(path: Path) -> tuple[set[str], Dict[str, set[str]]]:
    payload = load_similarity_metrics_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"eligibility_metrics.json must be a JSON object in {path}")
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        raise TypeError(f"eligibility_metrics.json must contain a baselines object in {path}")
    eligible_baselines: set[str] = set()
    eligible_children: Dict[str, set[str]] = {}
    for baseline_uuid, summary in baselines.items():
        baseline_id = str(baseline_uuid).strip()
        if not baseline_id or not isinstance(summary, dict):
            continue
        if summary.get("family_eligible") is True:
            eligible_baselines.add(baseline_id)
        if baseline_id not in eligible_baselines:
            continue
        children = summary.get("children")
        if not isinstance(children, dict):
            continue
        ids = {
            str(child_id).strip()
            for child_id, child in children.items()
            if str(child_id).strip()
            and isinstance(child, dict)
            and child.get("eligible") is True
        }
        if ids:
            eligible_children[baseline_id] = ids
    if not eligible_baselines and not eligible_children:
        raise ValueError(f"No eligible runs found in {path}")
    return eligible_baselines, eligible_children


def _selection_jobs(shots: Sequence[_Shot]) -> List[tuple[int, _Shot]]:
    jobs: List[tuple[int, _Shot]] = []
    for shot in shots:
        for seed in shot.seeds:
            jobs.append((int(seed), shot))
    return jobs


def _baseline_accuracy_threshold(num_classes: int) -> float:
    if int(num_classes) < 1:
        raise ValueError("num_classes must be >= 1")
    random_chance = 1.0 / float(num_classes)
    return random_chance + 0.1 * (1.0 - random_chance)


def _build_candidates_from_registry(
    *,
    registry: Sequence[Mapping[str, Any]],
    eligible_baselines: set[str],
    eligible_children_by_baseline: Mapping[str, set[str]],
    allowed_raw: Sequence[str],
    include_time0_baseline_class: bool,
    signals: Sequence[str],
    get_matrix_for_run_id: Any,
) -> tuple[Dict[str, List[_Candidate]], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    allowed = set(allowed_raw)
    candidates: Dict[str, List[_Candidate]] = defaultdict(list)
    matrices: Dict[str, np.ndarray] = {}
    skipped: List[Dict[str, Any]] = []
    seen_sample_ids: set[str] = set()

    def append(sample_id: str, data_run_id: str, baseline_uuid: str, label: str, reason: str) -> None:
        sample_id = str(sample_id).strip()
        if not sample_id or sample_id in seen_sample_ids:
            return
        matrix = get_matrix_for_run_id(data_run_id, signals)
        if matrix is None:
            skipped.append({"baseline_uuid": baseline_uuid, "intervention_uuid": sample_id, "reason": reason})
            return
        seen_sample_ids.add(sample_id)
        matrices[sample_id] = matrix
        candidates[label].append(_Candidate(sample_id, baseline_uuid, label))

    for run in registry:
        if not isinstance(run, dict):
            continue
        baseline_uuid = str(run.get("baseline_uuid") or run.get("run_id") or "").strip()
        allowed_children = (
            eligible_children_by_baseline.get(baseline_uuid, set())
            if baseline_uuid in eligible_baselines
            else set()
        )
        if include_time0_baseline_class and baseline_uuid in eligible_baselines:
            append(
                baseline_uuid,
                baseline_uuid,
                baseline_uuid,
                TIME0_BASELINE_LABEL,
                "baseline_missing_matrix",
            )
        for intervention in [item for item in (run.get("interventions") or []) if isinstance(item, dict)]:
            intervention_uuid = str(intervention.get("intervention_uuid") or intervention.get("name") or "").strip()
            if intervention_uuid not in allowed_children:
                continue
            label = next(
                (
                    candidate
                    for candidate in (
                        str(intervention.get("variable") or "").strip(),
                        str(intervention.get("value") or "").strip(),
                    )
                    if candidate in allowed
                ),
                None,
            )
            if not label:
                skipped.append(
                    {
                        "baseline_uuid": baseline_uuid,
                        "intervention_uuid": intervention_uuid,
                        "reason": "missing_label",
                    }
                )
                continue
            data_run_id = str(intervention.get("data_source_id") or "").strip() or intervention_uuid
            append(intervention_uuid, data_run_id, baseline_uuid, label, "intervention_missing_matrix")
    return candidates, matrices, skipped


def _balanced_label_counts(*, label_order: Sequence[str], test_count: int) -> Dict[str, int]:
    base, remainder = divmod(int(test_count), len(label_order))
    return {label: base + int(index < remainder) for index, label in enumerate(label_order)}


def _validate_sampling_feasibility(
    *,
    model_name: str,
    class_labels: Sequence[str],
    candidates_by_label: Mapping[str, Sequence[_Candidate]],
    shots: Sequence[_Shot],
) -> None:
    if not class_labels:
        raise ValueError("class_labels must be non-empty")
    all_baselines = {
        candidate.baseline_uuid
        for pool in candidates_by_label.values()
        for candidate in pool
        if str(candidate.baseline_uuid).strip()
    }
    total_unique_baselines = len(all_baselines)
    for shot in shots:
        seed_count = len(set(int(seed) for seed in shot.seeds))
        label_test_counts = _balanced_label_counts(
            label_order=class_labels,
            test_count=shot.test_samples,
        )
        problems: List[str] = []
        if total_unique_baselines < shot.test_samples:
            problems.append(
                f"test_samples={shot.test_samples} requires at least {shot.test_samples} "
                f"eligible unique baselines per seed, but only {total_unique_baselines} are available"
            )
        for label in class_labels:
            pool = list(candidates_by_label.get(label, []))
            unique_baselines = {candidate.baseline_uuid for candidate in pool}
            per_seed_test_count = int(label_test_counts[label])
            required_test_candidates = per_seed_test_count * seed_count
            required_train_candidates = int(shot.train_samples) * seed_count
            if len(unique_baselines) < per_seed_test_count:
                problems.append(
                    f"label={label!r} needs {per_seed_test_count} unique baselines per test seed, "
                    f"but has {len(unique_baselines)}"
                )
            if len(pool) < required_test_candidates:
                problems.append(
                    f"label={label!r} needs {required_test_candidates} non-reused test candidates "
                    f"across {seed_count} seed(s), but has {len(pool)}"
                )
            if len(pool) < required_train_candidates:
                problems.append(
                    f"label={label!r} needs {required_train_candidates} non-reused train candidates "
                    f"across {seed_count} seed(s), but has {len(pool)}"
                )
        if problems:
            candidate_counts = {
                label: {
                    "candidates": len(candidates_by_label.get(label, [])),
                    "unique_baselines": len(
                        {candidate.baseline_uuid for candidate in candidates_by_label.get(label, [])}
                    ),
                }
                for label in class_labels
            }
            raise ValueError(
                f"Sampling request is infeasible for model={model_name!r} shot_slug={shot.id!r}: "
                + "; ".join(problems)
                + f"; candidate_counts={candidate_counts}"
            )


def _sample_train_candidates(
    *,
    candidates_by_label: Mapping[str, Sequence[_Candidate]],
    label_order: Sequence[str],
    examples_per_label: int,
    used_train_uuids: Sequence[str],
    allowed_baselines: Optional[Sequence[str]],
    rng: np.random.Generator,
) -> Optional[List[_Candidate]]:
    if examples_per_label <= 0:
        return []
    used_uuids = {str(item).strip() for item in used_train_uuids if str(item).strip()}
    allowed = None if allowed_baselines is None else {str(item).strip() for item in allowed_baselines}
    selected: List[_Candidate] = []
    for label in label_order:
        pool = [
            candidate
            for candidate in candidates_by_label.get(label, [])
            if candidate.intervention_uuid not in used_uuids and (allowed is None or candidate.baseline_uuid in allowed)
        ]
        if len(pool) < examples_per_label:
            return None
        chosen = rng.choice(len(pool), size=examples_per_label, replace=False).tolist()
        selected.extend(pool[int(index)] for index in chosen)
    return selected


def _sample_test_candidates(
    *,
    candidates_by_label: Mapping[str, Sequence[_Candidate]],
    label_order: Sequence[str],
    test_count: int,
    used_test_uuids: Sequence[str],
    excluded_baselines: Sequence[str],
    rng: np.random.Generator,
) -> Optional[List[_Candidate]]:
    used_uuids = {str(item).strip() for item in used_test_uuids if str(item).strip()}
    excluded = {str(item).strip() for item in excluded_baselines if str(item).strip()}
    label_counts = _balanced_label_counts(label_order=label_order, test_count=test_count)
    pools: Dict[str, List[_Candidate]] = {}
    for label in label_order:
        pool = [
            candidate
            for candidate in candidates_by_label.get(label, [])
            if candidate.intervention_uuid not in used_uuids and candidate.baseline_uuid not in excluded
        ]
        if len({candidate.baseline_uuid for candidate in pool}) < label_counts[label]:
            return None
        order = rng.permutation(len(pool)).tolist() if pool else []
        pools[label] = [pool[int(index)] for index in order]
    if len({candidate.baseline_uuid for pool in pools.values() for candidate in pool}) < test_count:
        return None

    slots = [label for label in label_order for _ in range(label_counts[label])]
    slots.sort(key=lambda label: len(pools[label]))
    selected: List[_Candidate] = []
    chosen_uuids: set[str] = set()
    used_baselines: set[str] = set()

    def backtrack(index: int) -> bool:
        if index == len(slots):
            return True
        for candidate in pools[slots[index]]:
            if candidate.intervention_uuid in chosen_uuids or candidate.baseline_uuid in used_baselines:
                continue
            selected.append(candidate)
            chosen_uuids.add(candidate.intervention_uuid)
            used_baselines.add(candidate.baseline_uuid)
            if backtrack(index + 1):
                return True
            selected.pop()
            chosen_uuids.remove(candidate.intervention_uuid)
            used_baselines.remove(candidate.baseline_uuid)
        return False

    return list(selected) if backtrack(0) else None


def _sorted_manifest_group(
    *,
    candidates: Sequence[_Candidate],
    sample_hash_by_id: Mapping[str, str],
) -> tuple[List[str], List[str], List[str]]:
    rows: List[tuple[str, str, str]] = []
    for candidate in candidates:
        sample_id = str(candidate.intervention_uuid).strip()
        sample_hash = str(sample_hash_by_id.get(sample_id) or "").strip()
        if not sample_hash:
            raise ValueError(f"Missing sample hash for sampled uuid={sample_id!r}")
        rows.append((sample_id, str(candidate.baseline_uuid).strip(), sample_hash))
    rows.sort(key=lambda item: item[0])
    return (
        [sample_id for sample_id, _baseline_uuid, _sample_hash in rows],
        [baseline_uuid for _sample_id, baseline_uuid, _sample_hash in rows],
        [sample_hash for _sample_id, _baseline_uuid, sample_hash in rows],
    )


def _seeded_rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], byteorder="big", signed=False))


def _sample_manifest_rows(
    *,
    model_name: str,
    class_labels: Sequence[str],
    candidates_by_label: Mapping[str, Sequence[_Candidate]],
    matrix_by_uuid: Mapping[str, np.ndarray],
    sample_hash_by_id: Mapping[str, str],
    label_by_sample_id: Mapping[str, str],
    jobs: Sequence[tuple[int, _Shot]],
) -> Dict[str, List[Dict[str, Any]]]:
    manifest: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    train_seed_by_uuid_by_shot: Dict[str, Dict[str, int]] = defaultdict(dict)
    test_seed_by_uuid_by_test_set: Dict[str, Dict[str, int]] = defaultdict(dict)
    test_candidates_by_key: Dict[tuple[str, int], List[_Candidate]] = {}
    test_specs_by_key: Dict[tuple[str, int], _Shot] = {}
    ordered_test_keys: List[tuple[str, int]] = []
    for seed, shot in jobs:
        key = (shot.test_set_slug, int(seed))
        existing = test_specs_by_key.get(key)
        if existing is None:
            test_specs_by_key[key] = shot
            ordered_test_keys.append(key)
            continue
        if int(existing.test_samples) != int(shot.test_samples):
            raise ValueError(
                f"test_set_slug={shot.test_set_slug!r} seed={seed} maps to conflicting number_test_samples"
            )

    for test_set_slug, seed in ordered_test_keys:
        shot = test_specs_by_key[(test_set_slug, seed)]
        test_seed_by_uuid = test_seed_by_uuid_by_test_set[test_set_slug]
        selected_test_candidates = None
        for attempt in range(MAX_RESAMPLE_ATTEMPTS):
            rng = _seeded_rng(model_name, "test_set", test_set_slug, seed, attempt)
            used_test = sorted(
                uuid for uuid, assigned_seed in test_seed_by_uuid.items() if assigned_seed != int(seed)
            )
            selected_test_candidates = _sample_test_candidates(
                candidates_by_label=candidates_by_label,
                label_order=class_labels,
                test_count=shot.test_samples,
                used_test_uuids=used_test,
                excluded_baselines=[],
                rng=rng,
            )
            if selected_test_candidates is not None:
                for sample_id in [candidate.intervention_uuid for candidate in selected_test_candidates]:
                    test_seed_by_uuid[sample_id] = int(seed)
                break
        if selected_test_candidates is None:
            raise ValueError(
                f"Unable to generate shared test set for model={model_name!r} "
                f"test_set_slug={test_set_slug!r} seed={seed} after {MAX_RESAMPLE_ATTEMPTS} attempts"
            )
        test_candidates_by_key[(test_set_slug, seed)] = selected_test_candidates

    with click.progressbar(
        jobs,
        label=f"Sampling train splits for {model_name}",
        file=sys.stderr,
        item_show_func=lambda job: f"seed={job[0]} shot_slug={job[1].id} test_set_slug={job[1].test_set_slug}" if job else "",
    ) as iterator:
        for seed, shot in iterator:
            row = None
            train_seed_by_uuid = train_seed_by_uuid_by_shot[shot.id]
            test_candidates = test_candidates_by_key[(shot.test_set_slug, int(seed))]
            for attempt in range(MAX_RESAMPLE_ATTEMPTS):
                rng = _seeded_rng(model_name, seed, shot.id, shot.test_set_slug, attempt)
                used_train = sorted(
                    uuid for uuid, assigned_seed in train_seed_by_uuid.items() if assigned_seed != int(seed)
                )
                available_baselines = sorted(
                    {candidate.baseline_uuid for pool in candidates_by_label.values() for candidate in pool}
                    - {candidate.baseline_uuid for candidate in test_candidates}
                )
                train_candidates = _sample_train_candidates(
                    candidates_by_label=candidates_by_label,
                    label_order=class_labels,
                    examples_per_label=shot.train_samples,
                    used_train_uuids=used_train,
                    allowed_baselines=available_baselines,
                    rng=rng,
                )
                if train_candidates is None:
                    continue
                train_samples = [candidate.intervention_uuid for candidate in train_candidates]
                test_samples = [candidate.intervention_uuid for candidate in test_candidates]
                accuracy = run_baseline_classifier(
                    train_uuids=train_samples,
                    test_uuids=test_samples,
                    classifier_type=DEFAULT_TSENV_CLASSIFIERS,
                    runs_dir=None,
                    label_by_sample_id=label_by_sample_id,
                    matrix_loader=lambda sample_id: matrix_by_uuid.get(str(sample_id).strip()),
                )
                sorted_train_samples, sorted_train_baselines, sorted_train_hashes = _sorted_manifest_group(
                    candidates=train_candidates,
                    sample_hash_by_id=sample_hash_by_id,
                )
                sorted_test_samples, sorted_test_baselines, sorted_test_hashes = _sorted_manifest_group(
                    candidates=test_candidates,
                    sample_hash_by_id=sample_hash_by_id,
                )
                row = {
                    "train_samples_baselines": sorted_train_baselines,
                    "train_samples": sorted_train_samples,
                    "train_samples_hashes": sorted_train_hashes,
                    "test_samples_baselines": sorted_test_baselines,
                    "test_samples": sorted_test_samples,
                    "test_samples_hashes": sorted_test_hashes,
                    "seed": int(seed),
                    "test_set_slug": shot.test_set_slug,
                    "shot_slug_recipe": {
                        "is_adversarial": shot.is_adversarial,
                        "number_train_samples_per_class": int(shot.train_samples),
                        "number_test_samples": int(shot.test_samples),
                    },
                    "accuracy_with_baselines": accuracy,
                    "train_test_sample_hash": compute_train_test_sample_hash(
                        train_samples_hashes=sorted_train_hashes,
                        test_samples_hashes=sorted_test_hashes,
                    ),
                }
                if shot.is_adversarial is True:
                    adversarial_score = max(accuracy.values())
                    if adversarial_score > _baseline_accuracy_threshold(len(class_labels)):
                        row = None
                        continue
                for sample_id in train_samples:
                    train_seed_by_uuid[sample_id] = int(seed)
                break
            if row is None:
                raise ValueError(
                    f"Unable to generate sample_manifest row for model={model_name!r} seed={seed} "
                    f"shot_slug={shot.id!r} after {MAX_RESAMPLE_ATTEMPTS} attempts"
                )
            manifest[shot.id].append(row)

    all_manifest_selected = sorted(
        {
            str(sample_id).strip()
            for rows in manifest.values()
            for row in rows
            for sample_id in [*row["train_samples"], *row["test_samples"]]
            if str(sample_id).strip()
        }
    )
    for rows in manifest.values():
        for row in rows:
            current_samples = set(row["train_samples"]) | set(row["test_samples"])
            other_samples = [
                sample_id
                for sample_id in all_manifest_selected
                if sample_id not in current_samples
            ]
            row["other_samples"] = other_samples
            row["accuracy_with_baselines_all_samples"] = run_baseline_classifier(
                train_uuids=sorted(set(row["train_samples"]) | set(other_samples)),
                test_uuids=row["test_samples"],
                classifier_type=DEFAULT_TSENV_CLASSIFIERS,
                runs_dir=None,
                label_by_sample_id=label_by_sample_id,
                matrix_loader=lambda sample_id: matrix_by_uuid.get(str(sample_id).strip()),
                require_balanced_support=False,
            )
    return {shot_slug: items for shot_slug, items in manifest.items()}


def _resolve_model_dir(*, model: Optional[str], model_dir: Optional[Path]) -> Path:
    if model_dir is not None:
        return model_dir.expanduser().resolve()
    model_name = str(model or "").strip()
    if not model_name:
        raise click.UsageError("Provide --model or --model-dir.")
    return (WORKSPACE_ROOT / "models" / "simulink" / model_name).resolve()


def _normalize_shots(rows: Sequence[Any]) -> List[_Shot]:
    shots = group_shot_selections(rows)
    return [
        _Shot(
            id=shot.shot_slug,
            test_set_slug=shot.test_set_slug,
            is_adversarial=shot.is_adversarial,
            train_samples=shot.number_train_samples_per_class,
            test_samples=shot.number_test_samples,
            seeds=tuple(int(seed) for seed in shot.seeds),
        )
        for shot in shots
    ]


@click.command()
@click.option("--model", type=str, default=None, help="tsENV model name under models/simulink/.")
@click.option("--model-dir", type=click.Path(path_type=Path), default=None, help="Explicit model directory.")
@click.option("--runs-dir-name", type=str, default=None, help="Model-local run-artifact directory name.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=TSENV_QUESTIONS_ROOT,
    show_default=True,
    help="Canonical tsENV bundle root for sample_manifest.json.",
)
@click.option(
    "--combinations-csv",
    type=click.Path(path_type=Path),
    default=ALL_POSSIBLE_COMBINATIONS_PATH,
    show_default=True,
    help="Path to all_possible_combinations.csv.",
)
@click.option(
    "--row_slug",
    "--row-slug",
    "row_slugs",
    multiple=True,
    help="Optional row_slug filter(s) from all_possible_combinations.csv.",
)
def main(
    model: Optional[str],
    model_dir: Optional[Path],
    runs_dir_name: Optional[str],
    output_dir: Path,
    combinations_csv: Path,
    row_slugs: Sequence[str],
) -> None:
    model_dir = _resolve_model_dir(model=model, model_dir=model_dir)
    output_dir = output_dir.expanduser().resolve()
    combinations_csv = combinations_csv.expanduser().resolve()
    runs_dir = resolve_runs_root(model_dir, runs_dir_name=runs_dir_name)
    registry_path = resolve_model_record_path(model_dir, runs_dir_name=runs_dir_name)
    sample_manifest_path = resolve_exam_questions_output_dir(output_dir, model_dir.name) / "sample_manifest.json"
    summary_metrics_path = resolve_similarity_metrics_path(model_dir, runs_dir_name=runs_dir_name)
    config_path = model_dir / "experiment_config.json"
    specs_path = model_dir / "model_run_specs.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing {registry_path}")
    if not specs_path.exists():
        raise FileNotFoundError(f"Missing {specs_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    rows = load_combination_rows(combinations_csv, row_slugs=row_slugs)
    shots = _normalize_shots(rows)
    if not shots:
        raise ValueError("No shot_slug rows were resolved from all_possible_combinations.csv")

    runtime_map = load_model_record_json(registry_path)
    experiment_config = load_experiment_config_json(config_path)
    registry = build_model_record_registry(
        model_id=model_dir.name,
        specs=load_model_run_specs_json(
            specs_path,
            enforce_baseline_pair_diversity=False,
        ),
        runtime_map=runtime_map,
        experiment_config=experiment_config,
    ).get("baselines", [])
    if not isinstance(registry, list):
        raise TypeError("derived registry baselines must be a JSON array")

    labels, include_baseline_class = _load_class_labels_from_experiment_config(config_path)
    signals = _infer_signals(runs_dir, registry)
    matrix_cache: Dict[tuple[str, tuple[str, ...]], Optional[np.ndarray]] = {}
    eligible_baselines, eligible_children_by_baseline = _load_eligible_runs(summary_metrics_path)
    candidates_by_label, matrix_by_uuid, _ = _build_candidates_from_registry(
        registry=registry,
        eligible_baselines=eligible_baselines,
        eligible_children_by_baseline=eligible_children_by_baseline,
        allowed_raw=labels,
        include_time0_baseline_class=include_baseline_class,
        signals=signals,
        get_matrix_for_run_id=lambda run_id, cols: _read_matrix(runs_dir, run_id, cols, matrix_cache),
    )
    class_labels = list(labels) + ([TIME0_BASELINE_LABEL] if include_baseline_class else [])
    missing_labels = [label for label in class_labels if not candidates_by_label.get(label)]
    if missing_labels:
        raise ValueError(
            f"Missing sampling candidates for required labels {missing_labels!r} in model={model_dir.name!r}"
        )
    candidate_lists = {
        label: sorted(candidates_by_label[label], key=lambda item: (item.baseline_uuid, item.intervention_uuid))
        for label in class_labels
    }
    label_by_sample_id = {
        candidate.intervention_uuid: candidate.label
        for pool in candidate_lists.values()
        for candidate in pool
    }
    sample_hash_by_id = {
        str(run_id).strip(): str(runtime.get("parameters_hash") or "").strip()
        for run_id, runtime in runtime_map.items()
        if str(run_id).strip() and isinstance(runtime, dict) and str(runtime.get("parameters_hash") or "").strip()
    }
    _validate_sampling_feasibility(
        model_name=model_dir.name,
        class_labels=class_labels,
        candidates_by_label=candidate_lists,
        shots=shots,
    )
    sample_manifest = _sample_manifest_rows(
        model_name=model_dir.name,
        class_labels=class_labels,
        candidates_by_label=candidate_lists,
        matrix_by_uuid=matrix_by_uuid,
        sample_hash_by_id=sample_hash_by_id,
        label_by_sample_id=label_by_sample_id,
        jobs=_selection_jobs(shots),
    )

    sample_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    validate_sample_manifest_payload(
        sample_manifest,
        num_labels=len(class_labels),
        label_by_sample_id=label_by_sample_id,
        path=str(sample_manifest_path),
    )
    sample_manifest_path.write_text(json.dumps(sample_manifest, indent=2), encoding="utf-8")
    click.echo(f"Saved in {sample_manifest_path}")


if __name__ == "__main__":
    main()
