#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import click
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from shared.interface.distribution_json import load_experiment_config_json  # noqa: E402
from shared.interface.model_record_json import load_model_run_specs_json  # noqa: E402
from shared.interface.similarity_metrics_json import load_similarity_metrics_json  # noqa: E402
from shared.run_artifacts import resolve_runs_root  # noqa: E402

DEFAULT_CV = 5
NO_CHANGE_LABEL = "no_parameter_change"
SPLIT_MANIFEST_FILENAME = "split_manifest.json"
_RUN_DATA_FILENAMES = ("data.parquet", "data.csv")
_SAMPLE_MANIFEST_FILENAME = "sample_manifest.json"


def _timestamp_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")


def _resolve_model_dir(*, model: Optional[str], model_dir: Optional[Path]) -> Path:
    if model_dir is not None:
        return model_dir.expanduser().resolve()
    model_name = str(model or "").strip()
    if not model_name:
        raise click.ClickException("Provide --model or --model-dir.")
    candidate = (Path("models") / "simulink" / model_name).resolve()
    if not candidate.exists():
        raise click.ClickException(f"Model directory not found: {candidate}")
    return candidate


def _default_output_path(*, model_dir: Path, run_id: str) -> Path:
    return (
        model_dir
        / "supervised_baselines"
        / str(run_id).strip()
        / SPLIT_MANIFEST_FILENAME
    )


def _default_sample_manifest_path(*, model_dir: Path) -> Path:
    return WORKSPACE_ROOT / "tsENV_questions" / model_dir.name / _SAMPLE_MANIFEST_FILENAME


def _resolve_sample_manifest_path(
    *,
    model_dir: Path,
    sample_manifest: Optional[Path],
) -> Path:
    candidate = (
        sample_manifest.expanduser().resolve()
        if sample_manifest is not None
        else _default_sample_manifest_path(model_dir=model_dir)
    )
    if not candidate.exists():
        raise click.ClickException(f"sample_manifest.json not found: {candidate}")
    return candidate


def _sample_manifest_includes_baselines(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"sample_manifest.json must be a JSON object keyed by shot_slug: {path}")
    for shot_slug, rows in payload.items():
        if not isinstance(rows, list):
            raise ValueError(
                f"sample_manifest.json row {shot_slug!r} must be a list of selections: {path}"
            )
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"sample_manifest.json row {shot_slug!r} must contain object selections: {path}"
                )
            for sample_key, baseline_key in (
                ("train_samples", "train_samples_baselines"),
                ("test_samples", "test_samples_baselines"),
            ):
                samples = row.get(sample_key)
                baselines = row.get(baseline_key)
                if not isinstance(samples, list) or not isinstance(baselines, list):
                    raise ValueError(
                        f"sample_manifest.json row {shot_slug!r} must define {sample_key} and {baseline_key} lists: {path}"
                    )
                if len(samples) != len(baselines):
                    raise ValueError(
                        f"sample_manifest.json row {shot_slug!r} must align {sample_key} with {baseline_key}: {path}"
                    )
                for sample_id, baseline_id in zip(samples, baselines):
                    if str(sample_id).strip() == str(baseline_id).strip():
                        return True
    return False


def _load_eligible_children_by_baseline(path: Path) -> dict[str, set[str]]:
    payload = load_similarity_metrics_json(path)
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        raise ValueError(f"eligibility_metrics.json must contain a baselines object: {path}")
    eligible: dict[str, set[str]] = {}
    for baseline_uuid, summary in baselines.items():
        if not isinstance(summary, Mapping) or summary.get("family_eligible") is not True:
            continue
        children = summary.get("children")
        if not isinstance(children, Mapping):
            continue
        eligible_children = {
            str(child_uuid).strip()
            for child_uuid in children.keys()
            if str(child_uuid).strip()
        }
        if eligible_children:
            eligible[str(baseline_uuid).strip()] = eligible_children
    if not eligible:
        raise ValueError(f"No eligible baselines found in {path}")
    return dict(sorted(eligible.items()))


def _child_parameter(raw_child: Mapping[str, Any], *, baseline_uuid: str, child_uuid: str) -> str:
    parameters = raw_child.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError(
            f"baseline={baseline_uuid!r} child={child_uuid!r} must define a non-empty parameters object"
        )
    if len(parameters) != 1:
        raise ValueError(
            f"baseline={baseline_uuid!r} child={child_uuid!r} must define exactly one intervened parameter"
        )
    parameter = str(next(iter(parameters.keys()))).strip()
    if not parameter:
        raise ValueError(
            f"baseline={baseline_uuid!r} child={child_uuid!r} has an empty parameter name"
        )
    return parameter


def _find_run_data_path(runs_dir: Path, run_id: str) -> Optional[Path]:
    run_dir = runs_dir / str(run_id).strip()
    for filename in _RUN_DATA_FILENAMES:
        candidate = run_dir / filename
        if candidate.exists():
            return candidate
    return None


def _require_run_data(runs_dir: Path, run_id: str) -> None:
    if _find_run_data_path(runs_dir, run_id) is None:
        raise ValueError(
            f"Missing run data for sample {run_id!r} under {runs_dir}. "
            f"Expected one of {list(_RUN_DATA_FILENAMES)!r}."
        )


def _baseline_parameter_children(
    *,
    specs: Mapping[str, Any],
    eligible_children_by_baseline: Mapping[str, set[str]],
    parameter_names: Sequence[str],
    runs_dir: Path,
) -> dict[str, dict[str, str]]:
    expected_parameters = tuple(str(name).strip() for name in parameter_names if str(name).strip())
    by_baseline: dict[str, dict[str, str]] = {}
    for baseline_uuid, eligible_children in eligible_children_by_baseline.items():
        baseline = specs.get(baseline_uuid)
        if not isinstance(baseline, Mapping):
            raise ValueError(f"Eligible baseline {baseline_uuid!r} is missing from model_run_specs.json")
        children = baseline.get("children")
        if not isinstance(children, Mapping):
            raise ValueError(f"Eligible baseline {baseline_uuid!r} has no children in model_run_specs.json")

        param_to_child: dict[str, str] = {}
        for child_uuid, raw_child in children.items():
            child_id = str(child_uuid).strip()
            if child_id not in eligible_children or not isinstance(raw_child, Mapping):
                continue
            parameters = raw_child.get("parameters")
            if not isinstance(parameters, Mapping):
                continue
            if next(iter(parameters.values()), None) is None:
                continue
            parameter = _child_parameter(
                raw_child,
                baseline_uuid=baseline_uuid,
                child_uuid=child_id,
            )
            _require_run_data(runs_dir, child_id)
            previous = param_to_child.get(parameter)
            if previous is None or child_id < previous:
                param_to_child[parameter] = child_id

        missing_parameters = sorted(set(expected_parameters) - set(param_to_child))
        extra_parameters = sorted(set(param_to_child) - set(expected_parameters))
        if missing_parameters or extra_parameters:
            details: list[str] = []
            if missing_parameters:
                details.append(f"missing={missing_parameters}")
            if extra_parameters:
                details.append(f"extra={extra_parameters}")
            raise ValueError(
                f"Eligible baseline {baseline_uuid!r} does not match the expected intervention parameter set "
                f"({', '.join(details)})"
            )
        by_baseline[baseline_uuid] = {
            parameter: param_to_child[parameter]
            for parameter in expected_parameters
        }
    return by_baseline


def _label_int_mapping(
    parameter_names: Sequence[str],
    *,
    is_baseline_class: bool,
) -> dict[str, int]:
    labels = [str(name).strip() for name in parameter_names if str(name).strip()]
    if is_baseline_class:
        labels.append(NO_CHANGE_LABEL)
    return {label: idx for idx, label in enumerate(labels)}


def _kfold_partitions(
    parent_ids: Sequence[str],
    *,
    cv: int,
    seed: int,
) -> list[tuple[str, ...]]:
    items = [str(parent_id).strip() for parent_id in parent_ids if str(parent_id).strip()]
    if len(items) != len(parent_ids):
        raise ValueError("parent_ids must contain only non-empty baseline UUIDs")
    if cv < 2:
        raise ValueError("cv must be >= 2")
    if len(items) < cv:
        raise ValueError(
            f"Need at least {cv} eligible baselines for cv={cv}, got {len(items)}"
        )
    order = np.random.default_rng(int(seed)).permutation(len(items)).tolist()
    shuffled = [items[int(index)] for index in order]
    return [tuple(str(item).strip() for item in fold.tolist()) for fold in np.array_split(shuffled, cv)]


def _expand_split_samples(
    parent_ids: Sequence[str],
    *,
    baseline_children_by_parameter: Mapping[str, Mapping[str, str]],
    parameter_names: Sequence[str],
    is_baseline_class: bool,
) -> tuple[str, ...]:
    ordered_parents = sorted(str(parent_id).strip() for parent_id in parent_ids if str(parent_id).strip())
    sample_ids: list[str] = []
    for baseline_uuid in ordered_parents:
        if baseline_uuid not in baseline_children_by_parameter:
            raise ValueError(f"Unknown baseline in split expansion: {baseline_uuid!r}")
        if is_baseline_class:
            sample_ids.append(baseline_uuid)
        child_map = baseline_children_by_parameter[baseline_uuid]
        for parameter in parameter_names:
            child_uuid = str(child_map.get(str(parameter).strip()) or "").strip()
            if not child_uuid:
                raise ValueError(
                    f"baseline={baseline_uuid!r} missing expanded child for parameter={parameter!r}"
                )
            sample_ids.append(child_uuid)
    return tuple(sample_ids)


def build_split_manifest(
    *,
    model_name: str,
    eligible_parent_ids: Sequence[str],
    baseline_children_by_parameter: Mapping[str, Mapping[str, str]],
    parameter_names: Sequence[str],
    cv: int,
    seed: int,
    is_baseline_class: bool,
) -> dict[str, Any]:
    outer_folds = _kfold_partitions(
        eligible_parent_ids,
        cv=cv,
        seed=seed,
    )

    label_by_sample_id: dict[str, str] = {}
    for baseline_uuid, child_map in baseline_children_by_parameter.items():
        if is_baseline_class:
            label_by_sample_id[baseline_uuid] = NO_CHANGE_LABEL
        for parameter, child_uuid in child_map.items():
            label_by_sample_id[str(child_uuid).strip()] = str(parameter).strip()

    splits: list[dict[str, Any]] = []
    for fold_index, test_fold in enumerate(outer_folds):
        test_parents = tuple(sorted(test_fold))
        train_parents = tuple(
            sorted(
                parent_id
                for parent_id in eligible_parent_ids
                if str(parent_id).strip() not in set(test_fold)
            )
        )
        if not train_parents:
            raise ValueError(f"Fold {fold_index} produced an empty training parent partition")
        if not test_parents:
            raise ValueError(f"Fold {fold_index} produced an empty test parent partition")

        train_samples = _expand_split_samples(
            train_parents,
            baseline_children_by_parameter=baseline_children_by_parameter,
            parameter_names=parameter_names,
            is_baseline_class=is_baseline_class,
        )
        test_samples = _expand_split_samples(
            test_parents,
            baseline_children_by_parameter=baseline_children_by_parameter,
            parameter_names=parameter_names,
            is_baseline_class=is_baseline_class,
        )

        splits.append(
            {
                "split_key": f"fold_{fold_index}",
                "train_parents": list(train_parents),
                "test_parents": list(test_parents),
                "train_samples": list(train_samples),
                "test_samples": list(test_samples),
            }
        )

    return {
        "model": str(model_name).strip(),
        "seed": int(seed),
        "cv": int(cv),
        "is_baseline_class": bool(is_baseline_class),
        "label_int_mapping": _label_int_mapping(
            parameter_names,
            is_baseline_class=is_baseline_class,
        ),
        "label_by_sample_id": dict(sorted(label_by_sample_id.items())),
        "splits": splits,
    }


@click.command()
@click.option("--model", type=str, default=None, help="Model name under models/simulink.")
@click.option("--model-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--runs-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--runs-dir-name", type=str, default=None)
@click.option(
    "--sample-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional tsENV_questions/<MODEL>/sample_manifest.json override.",
)
@click.option("--run-id", type=str, default=None, help="Run id used under supervised_baselines/.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional explicit split_manifest.json output path.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--cv", type=int, default=DEFAULT_CV, show_default=True)
def main(
    model: Optional[str],
    model_dir: Optional[Path],
    runs_dir: Optional[Path],
    runs_dir_name: Optional[str],
    sample_manifest: Optional[Path],
    run_id: Optional[str],
    output: Optional[Path],
    seed: int,
    cv: int,
) -> None:
    resolved_model_dir = _resolve_model_dir(model=model, model_dir=model_dir)
    resolved_runs_dir = resolve_runs_root(
        resolved_model_dir,
        runs_dir=runs_dir,
        runs_dir_name=runs_dir_name,
    )
    experiment_config = load_experiment_config_json(
        resolved_model_dir / "experiment_config.json"
    )
    parameter_names = list(experiment_config.intervention_parameter_names)
    if not parameter_names:
        raise click.ClickException(
            f"No exposed intervention parameters found in {resolved_model_dir / 'experiment_config.json'}"
        )
    sample_manifest_path = _resolve_sample_manifest_path(
        model_dir=resolved_model_dir,
        sample_manifest=sample_manifest,
    )
    is_baseline_class = _sample_manifest_includes_baselines(sample_manifest_path)

    eligible_children_by_baseline = _load_eligible_children_by_baseline(
        resolved_runs_dir / "eligibility_metrics.json"
    )
    specs = load_model_run_specs_json(
        resolved_model_dir / "model_run_specs.json",
        enforce_baseline_pair_diversity=False,
    )
    for baseline_uuid in eligible_children_by_baseline:
        if is_baseline_class:
            _require_run_data(resolved_runs_dir, baseline_uuid)
    baseline_children_by_parameter = _baseline_parameter_children(
        specs=specs,
        eligible_children_by_baseline=eligible_children_by_baseline,
        parameter_names=parameter_names,
        runs_dir=resolved_runs_dir,
    )
    eligible_parent_ids = tuple(sorted(eligible_children_by_baseline.keys()))

    manifest = build_split_manifest(
        model_name=resolved_model_dir.name,
        eligible_parent_ids=eligible_parent_ids,
        baseline_children_by_parameter=baseline_children_by_parameter,
        parameter_names=parameter_names,
        cv=int(cv),
        seed=int(seed),
        is_baseline_class=bool(is_baseline_class),
    )

    resolved_run_id = str(run_id or _timestamp_run_id()).strip()
    output_path = (
        output.expanduser().resolve()
        if output is not None
        else _default_output_path(model_dir=resolved_model_dir, run_id=resolved_run_id)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    click.echo(
        "Saved split manifest for model={model} folds={folds} eligible_parents={parents} to {path}".format(
            model=resolved_model_dir.name,
            folds=len(manifest["splits"]),
            parents=len(eligible_parent_ids),
            path=output_path,
        )
    )


if __name__ == "__main__":
    main()
