#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import click
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from shared.exam_question_baselines import (  # noqa: E402
    predict_correlation_nn_truncated,
    predict_euclidean_centroid_truncated,
    predict_euclidean_knn_truncated,
)
from shared.run_artifacts import resolve_runs_root  # noqa: E402

DEFAULT_TSENV_CLASSIFIERS = (
    "euclidean_knn",
    "euclidean_centroid",
    "correlation_nn",
)
_NON_SIGNAL_COLUMNS = {"time", "timestamp", "index", "idx"}

MatrixLoader = Callable[[str], Optional[np.ndarray]]


def _normalize_sample_ids(sample_ids: Sequence[str], *, field_name: str) -> list[str]:
    raw_sample_ids = list(sample_ids)
    normalized = [str(sample_id).strip() for sample_id in raw_sample_ids if str(sample_id).strip()]
    if len(normalized) != len(raw_sample_ids):
        raise ValueError(f"{field_name} must contain only non-empty sample UUIDs")
    return normalized


def _normalize_classifier_names(classifier_type: object) -> list[str]:
    if classifier_type is None:
        requested = list(DEFAULT_TSENV_CLASSIFIERS)
    elif isinstance(classifier_type, str):
        requested = [classifier_type]
    elif isinstance(classifier_type, Mapping):
        requested = [str(name).strip() for name in classifier_type.keys() if str(name).strip()]
    elif isinstance(classifier_type, Sequence):
        requested = [str(name).strip() for name in classifier_type if str(name).strip()]
    else:
        raise TypeError("classifier_type must be a string, sequence, mapping, or None")
    if not requested:
        requested = list(DEFAULT_TSENV_CLASSIFIERS)
    invalid = sorted({name for name in requested if name not in DEFAULT_TSENV_CLASSIFIERS})
    if invalid:
        raise ValueError(f"Unsupported classifier_type entries: {invalid!r}")
    return [name for name in DEFAULT_TSENV_CLASSIFIERS if name in set(requested)]


def _find_run_data_path(runs_dir: Path, run_id: str) -> Optional[Path]:
    run_dir = runs_dir / run_id
    for name in ("data.parquet", "data.csv"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _load_matrix_from_path(path: Path, *, signals: Optional[Sequence[str]]) -> np.ndarray:
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    if "time" in df.columns:
        df = df.sort_values("time")
    available_signals = [
        str(col).strip()
        for col in df.columns
        if str(col).strip() and str(col).strip().lower() not in _NON_SIGNAL_COLUMNS
    ]
    selected_signals = list(signals or available_signals)
    if not selected_signals:
        raise ValueError(f"No usable signal columns found in {path}")
    missing = [signal for signal in selected_signals if signal not in df.columns]
    if missing:
        raise ValueError(f"Missing required signals {missing!r} in {path}")
    matrix = df[selected_signals].to_numpy(dtype=np.float32, copy=True)
    if matrix.size == 0:
        raise ValueError(f"Empty matrix in {path}")
    if np.isnan(matrix).any():
        means = np.nanmean(matrix, axis=0)
        inds = np.where(np.isnan(matrix))
        matrix[inds] = np.take(means, inds[1])
    return matrix


def build_runs_dir_matrix_loader(
    runs_dir: Path,
    *,
    signals: Optional[Sequence[str]] = None,
) -> MatrixLoader:
    runs_dir = runs_dir.expanduser().resolve()
    cache: Dict[str, Optional[np.ndarray]] = {}
    resolved_signals = tuple(str(signal).strip() for signal in (signals or []) if str(signal).strip()) or None

    def load(sample_id: str) -> Optional[np.ndarray]:
        nonlocal resolved_signals
        sample_id = str(sample_id).strip()
        if sample_id in cache:
            return cache[sample_id]
        path = _find_run_data_path(runs_dir, sample_id)
        if path is None:
            cache[sample_id] = None
            return None
        if resolved_signals is None:
            preview = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, nrows=1)
            resolved_signals = tuple(
                str(col).strip()
                for col in preview.columns
                if str(col).strip() and str(col).strip().lower() not in _NON_SIGNAL_COLUMNS
            )
        matrix = _load_matrix_from_path(path, signals=resolved_signals)
        cache[sample_id] = matrix
        return matrix

    return load


def _label_for_sample(sample_id: str, label_by_sample_id: Mapping[str, str]) -> str:
    label = str(label_by_sample_id.get(sample_id) or "").strip()
    if not label:
        raise ValueError(f"Missing label for sample {sample_id!r}")
    return label


def _build_support(
    *,
    train_uuids: Sequence[str],
    label_by_sample_id: Mapping[str, str],
    matrix_loader: MatrixLoader,
) -> tuple[Dict[str, list[str]], Dict[str, np.ndarray]]:
    shots_by_label: Dict[str, list[str]] = defaultdict(list)
    samples_by_id: Dict[str, np.ndarray] = {}
    for sample_id in train_uuids:
        label = _label_for_sample(sample_id, label_by_sample_id)
        matrix = matrix_loader(sample_id)
        if matrix is None:
            raise ValueError(f"Missing training matrix for {sample_id!r}")
        shots_by_label[label].append(sample_id)
        samples_by_id[sample_id] = matrix
    return dict(shots_by_label), samples_by_id


def _validate_support(
    *,
    shots_by_label: Mapping[str, Sequence[str]],
    test_uuids: Sequence[str],
    label_by_sample_id: Mapping[str, str],
    require_balanced_support: bool = True,
) -> None:
    if not shots_by_label:
        return
    support_counts = {str(label): len(sample_ids) for label, sample_ids in shots_by_label.items()}
    if not require_balanced_support:
        return
    if len(set(support_counts.values())) > 1:
        raise ValueError(
            "train_uuids must be strictly balanced across labels, "
            f"got counts={dict(sorted(support_counts.items()))}"
        )
    test_labels = {_label_for_sample(sample_id, label_by_sample_id) for sample_id in test_uuids}
    missing_labels = sorted(test_labels - set(shots_by_label.keys()))
    if missing_labels:
        raise ValueError(
            "train_uuids do not cover all test labels, "
            f"missing support for labels={missing_labels!r}"
        )


def run_baseline_classifier(
    train_uuids: Sequence[str],
    test_uuids: Sequence[str],
    classifier_type: object,
    *,
    runs_dir: Optional[Path] = None,
    label_by_sample_id: Mapping[str, str],
    matrix_loader: Optional[MatrixLoader] = None,
    signals: Optional[Sequence[str]] = None,
    require_balanced_support: bool = True,
) -> dict[str, float]:
    train_ids = _normalize_sample_ids(train_uuids, field_name="train_uuids")
    test_ids = _normalize_sample_ids(test_uuids, field_name="test_uuids")
    if not test_ids:
        raise ValueError("test_uuids must be non-empty")
    classifier_names = _normalize_classifier_names(classifier_type)
    if matrix_loader is None:
        if runs_dir is None:
            raise ValueError("runs_dir is required when matrix_loader is not provided")
        matrix_loader = build_runs_dir_matrix_loader(runs_dir, signals=signals)
    shots_by_label, samples_by_id = _build_support(
        train_uuids=train_ids,
        label_by_sample_id=label_by_sample_id,
        matrix_loader=matrix_loader,
    )
    _validate_support(
        shots_by_label=shots_by_label,
        test_uuids=test_ids,
        label_by_sample_id=label_by_sample_id,
        require_balanced_support=require_balanced_support,
    )
    correct = {name: 0 for name in classifier_names}
    for sample_id in test_ids:
        label = _label_for_sample(sample_id, label_by_sample_id)
        matrix = matrix_loader(sample_id)
        if matrix is None:
            raise ValueError(f"Missing test matrix for {sample_id!r}")
        if "euclidean_knn" in correct:
            correct["euclidean_knn"] += int(
                predict_euclidean_knn_truncated(matrix, samples_by_id, shots_by_label) == label
            )
        if "euclidean_centroid" in correct:
            correct["euclidean_centroid"] += int(
                predict_euclidean_centroid_truncated(matrix, samples_by_id, shots_by_label) == label
            )
        if "correlation_nn" in correct:
            correct["correlation_nn"] += int(
                predict_correlation_nn_truncated(matrix, samples_by_id, shots_by_label) == label
            )
    return {name: correct[name] / float(len(test_ids)) for name in classifier_names}


def baseline_classifier(
    train_uuids: Sequence[str],
    test_uuids: Sequence[str],
    classifier_type: object,
    *,
    runs_dir: Optional[Path] = None,
    label_by_sample_id: Mapping[str, str],
    matrix_loader: Optional[MatrixLoader] = None,
    signals: Optional[Sequence[str]] = None,
    require_balanced_support: bool = True,
) -> dict[str, float]:
    return run_baseline_classifier(
        train_uuids=train_uuids,
        test_uuids=test_uuids,
        classifier_type=classifier_type,
        runs_dir=runs_dir,
        label_by_sample_id=label_by_sample_id,
        matrix_loader=matrix_loader,
        signals=signals,
        require_balanced_support=require_balanced_support,
    )


def _load_json_value(raw_value: str) -> Any:
    candidate = Path(raw_value).expanduser()
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(raw_value)


def _load_json_array(raw_value: str, *, field_name: str) -> list[str]:
    payload = _load_json_value(raw_value)
    if not isinstance(payload, list):
        raise click.ClickException(f"{field_name} must be a JSON array or a path to one")
    return _normalize_sample_ids([str(item) for item in payload], field_name=field_name)


def _load_label_mapping(raw_value: str) -> dict[str, str]:
    payload = _load_json_value(raw_value)
    if not isinstance(payload, dict):
        raise click.ClickException("label_by_sample_id must be a JSON object or a path to one")
    return {
        str(sample_id).strip(): str(label).strip()
        for sample_id, label in payload.items()
        if str(sample_id).strip() and str(label).strip()
    }


@click.command()
@click.option("--model-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--runs-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--runs-dir-name", type=str, default=None)
@click.option("--train-uuids", required=True, type=str, help="JSON array or path to JSON array.")
@click.option("--test-uuids", required=True, type=str, help="JSON array or path to JSON array.")
@click.option(
    "--classifier-type",
    "classifier_types",
    multiple=True,
    type=click.Choice(list(DEFAULT_TSENV_CLASSIFIERS), case_sensitive=False),
    help="Requested classifier(s). Defaults to all documented tsENV baselines.",
)
@click.option(
    "--label-by-sample-id",
    required=True,
    type=str,
    help="JSON object or path to JSON object mapping sample UUID -> class label.",
)
@click.option("--signal", "signals", multiple=True, type=str, help="Optional signal columns to use.")
def main(
    model_dir: Optional[Path],
    runs_dir: Optional[Path],
    runs_dir_name: Optional[str],
    train_uuids: str,
    test_uuids: str,
    classifier_types: Sequence[str],
    label_by_sample_id: str,
    signals: Sequence[str],
) -> None:
    if runs_dir is None:
        if model_dir is None:
            raise click.ClickException("Provide either --runs-dir or --model-dir")
        runs_dir = resolve_runs_root(model_dir, runs_dir_name=runs_dir_name)
    payload = run_baseline_classifier(
        train_uuids=_load_json_array(train_uuids, field_name="train_uuids"),
        test_uuids=_load_json_array(test_uuids, field_name="test_uuids"),
        classifier_type=list(classifier_types) or list(DEFAULT_TSENV_CLASSIFIERS),
        runs_dir=runs_dir,
        label_by_sample_id=_load_label_mapping(label_by_sample_id),
        signals=list(signals),
    )
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_TSENV_CLASSIFIERS",
    "baseline_classifier",
    "build_runs_dir_matrix_loader",
    "main",
    "run_baseline_classifier",
]
