#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from baseline_utils import (  # noqa: E402
    assert_strict_balanced_support,
    build_label_to_index,
    infer_few_shot_labels,
    load_classification_questions,
    parse_context_and_shot,
    question_train_paths,
)
from shared.interfaces import question_variant

METRICS_COLUMNS = [
    "run_dir",
    "run_id",
    "agent_name",
    "model_name",
    "metric_accuracy",
    "dataset_label",
    "context_level",
    "shot_level",
    "is_few_shot",
    "benchmark",
    "variant",
    "benchmark_id",
    "question_id",
    "exam_question_root",
    "task_id",
    "trial_name",
    "dataset",
    "label_path",
]

DATA_ROOT_DEFAULT = Path("exam_questions_complete/classification_univariate")
PATH_DIR = click.Path(file_okay=False, path_type=Path)
PATH_FILE = click.Path(dir_okay=False, path_type=Path)
DISTANCE = click.Choice(["dtw", "euclidean", "xcorr"], case_sensitive=False)
CLASSIFIER = click.Choice(["knn", "centroid"], case_sensitive=False)
DTW_BACKEND = click.Choice(["auto", "dtaidistance"], case_sensitive=False)
WINDOW_GRID_DEFAULT = (0, 5, 10, 25, 50)
KNN_GRID_DEFAULT = (1, 3, 5)

def _metrics_row(
    q, run_id, model_name, dataset_label, context_level, shot_level, is_few_shot, value
):
    question_id = str(q.get("question_id") or "")
    safe_id = question_id.replace(os.sep, "_") if os.sep else question_id
    if os.altsep:
        safe_id = safe_id.replace(os.altsep, "_")
    benchmark = q.get("benchmark")
    variant = question_variant(q)
    return {
        "run_dir": "",
        "run_id": run_id,
        "agent_name": "baseline",
        "model_name": model_name,
        "metric_accuracy": value,
        "dataset_label": dataset_label,
        "context_level": context_level,
        "shot_level": shot_level,
        "is_few_shot": is_few_shot,
        "benchmark": benchmark,
        "variant": variant,
        "benchmark_id": f"{benchmark}_{variant}",
        "question_id": question_id,
        "exam_question_root": q.get("exam_question_root"),
        "task_id": question_id,
        "trial_name": f"{safe_id}.1-of-1.{run_id}",
        "dataset": q.get("dataset"),
        "label_path": None,
    }


def _import_dtw_ndim():
    try:
        from dtaidistance import dtw_ndim  # type: ignore[import-not-found]

        return dtw_ndim
    except Exception as exc:
        raise RuntimeError(
            "DTW requires 'dtaidistance'. Install with: env/bin/pip install dtaidistance"
        ) from exc


def _downsample_block_mean(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by block mean to reduce aliasing (time axis is last)."""
    factor = max(1, int(factor))
    if factor <= 1:
        return x
    t = x.shape[1]
    t2 = (t // factor) * factor
    if t2 <= 0:
        return x[:, :1]
    x2 = x[:, :t2].reshape(x.shape[0], -1, factor)
    return x2.mean(axis=2).astype(np.float32, copy=False)


def _prepare_series(parquet_path, *, downsample, max_len, normalize):
    df = pd.read_parquet(parquet_path)
    if "time" in df.columns:
        df = df.sort_values("time")
    value_cols = [
        c for c in df.columns if c.lower() not in ("time", "timestamp", "index", "idx")
    ]
    if not value_cols:
        raise ValueError(f"No value columns found in {parquet_path}")

    # channel-first: (channels, time)
    ch_first = df[value_cols].to_numpy().T.astype(np.float32)
    ch_first = np.where(np.isfinite(ch_first), ch_first, np.nan)
    means = np.nanmean(ch_first, axis=1, keepdims=True)
    if not np.all(np.isfinite(means)):
        bad = np.flatnonzero(~np.isfinite(means[:, 0])).tolist()
        raise ValueError(
            f"All-NaN channel(s) in {parquet_path}: {[value_cols[i] for i in bad]}"
        )
    ch_first = np.where(np.isnan(ch_first), means, ch_first)

    # Downsample (block mean) if requested
    if downsample > 1:
        ch_first = _downsample_block_mean(ch_first, downsample)

    # Optional resampling to fixed length (only if max_len is set and > 0)
    if max_len and max_len > 0 and ch_first.shape[1] != int(max_len):
        target = int(max_len)
        orig = int(ch_first.shape[1])
        if orig < 2:
            ch_first = np.repeat(ch_first, target, axis=1).astype(np.float32)
        else:
            x_old = np.linspace(0, 1, orig, dtype=np.float32)
            x_new = np.linspace(0, 1, target, dtype=np.float32)
            ch_first = np.vstack([np.interp(x_new, x_old, ch) for ch in ch_first]).astype(
                np.float32
            )

    # Per-series, per-channel z-normalization (standard UCR-style option)
    if normalize:
        mean = ch_first.mean(axis=1, keepdims=True)
        std = ch_first.std(axis=1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        ch_first = ((ch_first - mean) / std).astype(np.float32)

    return ch_first.T  # (time, channels)


def _vote_knn(labels: np.ndarray, distances: np.ndarray, k: int) -> int:
    labels = np.asarray(labels, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float64)
    if labels.shape[0] != distances.shape[0]:
        raise ValueError("labels/distances length mismatch")
    k_eff = min(max(int(k), 1), int(labels.shape[0]))
    idx = np.argpartition(distances, k_eff - 1)[:k_eff]
    lbls = labels[idx]
    dists = distances[idx]
    counts: dict[int, int] = defaultdict(int)
    sums: dict[int, float] = defaultdict(float)
    for lbl, dist in zip(lbls.tolist(), dists.tolist()):
        counts[int(lbl)] += 1
        sums[int(lbl)] += float(dist)
    # tie-break by (more votes, lower total distance, then label)
    return max(counts, key=lambda lbl: (counts[lbl], -sums[lbl], -lbl))


def _loocv_accuracy(dist_mat: np.ndarray, y: np.ndarray, k: int) -> float:
    y = np.asarray(y, dtype=np.int64)
    n = int(y.shape[0])
    if n < 2:
        return 0.0
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pred = _vote_knn(y[mask], dist_mat[i, mask], k)
        correct += int(pred == int(y[i]))
    return float(correct / n)


def _predict_dtw_knn(dtw_ndim, train, y, query, *, k, window):
    k = max(1, int(k))
    best: list[tuple[float, int]] = []
    cutoff = float("inf")
    for series, lbl in zip(train, y):
        max_dist = None if cutoff == float("inf") else float(cutoff)
        dist = float(
            dtw_ndim.distance_fast(
                query,
                series,
                window=window,
                max_dist=max_dist,
                use_pruning=max_dist is not None,
                inner_dist="squared euclidean",
            )
        )
        lbl_i = int(lbl)
        if len(best) < k:
            best.append((dist, lbl_i))
            if len(best) == k:
                cutoff = max(d for d, _ in best)
            continue
        if dist >= cutoff:
            continue
        worst = max(range(k), key=lambda i: best[i][0])
        best[worst] = (dist, lbl_i)
        cutoff = max(d for d, _ in best)
    d = np.asarray([x[0] for x in best], dtype=np.float64)
    l = np.asarray([x[1] for x in best], dtype=np.int64)
    return _vote_knn(l, d, k)


def _tune_dtw_knn(
    dtw_ndim,
    train,
    y,
    *,
    base_window,
    base_k,
    tune_window,
    window_grid,
    tune_knn,
    knn_grid,
):
    y = np.asarray(y, dtype=np.int64)
    n = int(y.shape[0])
    if n < 2:
        return base_window, max(1, int(base_k)), 0.0

    windows = [base_window]
    if tune_window:
        if not window_grid:
            raise ValueError("--window-grid must be non-empty when --tune-window is set")
        windows = [None if int(w) <= 0 else int(w) for w in window_grid]

    ks = [max(1, int(base_k))]
    if tune_knn:
        if not knn_grid:
            raise ValueError("--knn-grid must be non-empty when --tune-knn is set")
        ks = sorted({max(1, int(k)) for k in knn_grid})

    def wv(w):
        return int(10**9) if w is None else int(w)

    best_key = (float("inf"), float("inf"), float("inf"))
    best = (base_window, max(1, int(base_k)), -1.0)
    for win in windows:
        dist = dtw_ndim.distance_matrix_fast(
            list(train),
            window=win,
            parallel=True,
            compact=False,
            inner_dist="squared euclidean",
        ).astype(np.float64, copy=False)
        np.fill_diagonal(dist, float("inf"))
        for k in ks:
            k_eff = min(int(k), n - 1)
            acc = _loocv_accuracy(dist, y, k_eff)
            key = (-float(acc), wv(win), int(k_eff))
            if key < best_key:
                best_key = key
                best = (win, int(k_eff), float(acc))
    return best


def _tune_euclidean_knn(train_flat, y, *, base_k, tune_knn, knn_grid):
    y = np.asarray(y, dtype=np.int64)
    n = int(y.shape[0])
    if n < 2:
        return max(1, int(base_k)), 0.0
    ks = [max(1, int(base_k))]
    if tune_knn:
        if not knn_grid:
            raise ValueError("--knn-grid must be non-empty when --tune-knn is set")
        ks = sorted({max(1, int(k)) for k in knn_grid})
    X = np.asarray(train_flat, dtype=np.float64)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    dist = x2 + x2.T - 2.0 * (X @ X.T)
    np.fill_diagonal(dist, float("inf"))
    best_key = (float("inf"), float("inf"))
    best = (int(ks[0]), -1.0)
    for k in ks:
        k_eff = min(int(k), n - 1)
        acc = _loocv_accuracy(dist, y, k_eff)
        key = (-float(acc), int(k_eff))
        if key < best_key:
            best_key = key
            best = (int(k_eff), float(acc))
    return best


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _xcorr_distance(x: np.ndarray, y: np.ndarray, *, max_lag: int) -> float:
    """
    Lag-tolerant distance: 1 - max corr over lags in [-max_lag, max_lag].
    Works for multivariate by averaging per-channel corr, then taking max over lags.
    x, y are (time, channels).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("xcorr expects (time, channels) arrays")
    c = x.shape[1]
    if y.shape[1] != c:
        raise ValueError("channel mismatch for xcorr distance")

    L = max(0, int(max_lag))
    best = -1.0

    # For each lag, compute mean correlation across channels on the overlapping region
    for lag in range(-L, L + 1):
        if lag < 0:
            xs = x[-lag:]
            ys = y[: y.shape[0] + lag]
        elif lag > 0:
            xs = x[: x.shape[0] - lag]
            ys = y[lag:]
        else:
            xs = x
            ys = y

        t = min(xs.shape[0], ys.shape[0])
        if t < 2:
            continue
        xs = xs[:t]
        ys = ys[:t]

        corr_sum = 0.0
        for ch in range(c):
            corr_sum += _corrcoef_safe(xs[:, ch], ys[:, ch])
        corr = corr_sum / float(c)
        if corr > best:
            best = corr

    # Clamp numeric noise
    best = max(-1.0, min(1.0, float(best)))
    return float(1.0 - best)


def _tune_xcorr_knn(train: list[np.ndarray], y, *, base_k, tune_knn, knn_grid, max_lag):
    y = np.asarray(y, dtype=np.int64)
    n = int(y.shape[0])
    if n < 2:
        return max(1, int(base_k)), 0.0
    ks = [max(1, int(base_k))]
    if tune_knn:
        if not knn_grid:
            raise ValueError("--knn-grid must be non-empty when --tune-knn is set")
        ks = sorted({max(1, int(k)) for k in knn_grid})

    dist = np.full((n, n), float("inf"), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _xcorr_distance(train[i], train[j], max_lag=max_lag)
            dist[i, j] = d
            dist[j, i] = d

    best_key = (float("inf"), float("inf"))
    best = (int(ks[0]), -1.0)
    for k in ks:
        k_eff = min(int(k), n - 1)
        acc = _loocv_accuracy(dist, y, k_eff)
        key = (-float(acc), int(k_eff))
        if key < best_key:
            best_key = key
            best = (int(k_eff), float(acc))
    return best


def _write_metrics_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=METRICS_COLUMNS)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path)
        if "agent_name" not in existing.columns:
            existing["agent_name"] = "baseline"
        if "metric_accuracy" not in existing.columns:
            if "metric_value" in existing.columns:
                existing["metric_accuracy"] = existing["metric_value"]
            elif "accuracy" in existing.columns:
                existing["metric_accuracy"] = existing["accuracy"]
            else:
                existing["metric_accuracy"] = np.nan
        existing = existing.reindex(columns=METRICS_COLUMNS, fill_value=np.nan)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(path, index=False, na_rep="nan")


@click.command()
@click.option("--data-root", type=PATH_DIR, default=DATA_ROOT_DEFAULT)
@click.option("--output", type=PATH_FILE, default=None)
@click.option("--metrics-output", type=PATH_FILE, default=None)
@click.option("--metrics-model-name", type=str, default=None)
@click.option("--limit-categories", multiple=True)
@click.option("--run-id", type=str, default=None)
@click.option("--distance", type=DISTANCE, default="dtw")
@click.option("--classifier", type=CLASSIFIER, default="knn")
@click.option("--knn-neighbors", type=int, default=1)
@click.option("--tune-window", is_flag=True, default=False)
@click.option("--window-grid", type=int, multiple=True, default=WINDOW_GRID_DEFAULT)
@click.option("--tune-knn", is_flag=True, default=False)
@click.option("--knn-grid", type=int, multiple=True, default=KNN_GRID_DEFAULT)
@click.option("--normalize/--no-normalize", default=False)
@click.option("--downsample", type=int, default=1)
# IMPORTANT CHANGE: default max-len is 0 => no resampling by default (esp. for DTW/xcorr).
@click.option("--max-len", type=int, default=0)
@click.option("--window", type=int, default=0)
@click.option("--dtw-backend", type=DTW_BACKEND, default="auto")
@click.option(
    "--xcorr-max-lag",
    type=int,
    default=25,
    help="Max lag (in samples) for xcorr distance (distance = 1 - max corr over +/-lag).",
)
def main(**kwargs) -> None:
    data_root = kwargs["data_root"]
    output = kwargs["output"]
    metrics_output = kwargs["metrics_output"]
    metrics_model_name = kwargs["metrics_model_name"]
    limit_categories = kwargs["limit_categories"]
    run_id = kwargs["run_id"]
    distance = kwargs["distance"].lower()
    classifier = kwargs["classifier"].lower()
    knn_neighbors = int(kwargs["knn_neighbors"])
    tune_window = bool(kwargs["tune_window"])
    window_grid = kwargs["window_grid"]
    tune_knn = bool(kwargs["tune_knn"])
    knn_grid = kwargs["knn_grid"]
    normalize = bool(kwargs["normalize"])
    downsample = max(1, int(kwargs["downsample"]))
    max_len = int(kwargs["max_len"])
    window = int(kwargs["window"])
    dtw_backend = kwargs["dtw_backend"].lower()
    xcorr_max_lag = max(0, int(kwargs["xcorr_max_lag"]))

    max_len = None if max_len <= 0 else int(max_len)
    window = None if window <= 0 else int(window)

    # Argument validation
    if distance != "dtw":
        if tune_window or window is not None:
            raise click.UsageError("--window/--tune-window only valid with --distance dtw")
        if dtw_backend != "auto":
            raise click.UsageError("--dtw-backend only valid with --distance dtw")

    if distance == "euclidean" and max_len is None:
        raise click.UsageError("--max-len must be > 0 with --distance euclidean")

    if classifier == "centroid" and distance != "euclidean":
        raise click.UsageError("--classifier centroid requires --distance euclidean")

    if distance in ("dtw", "xcorr") and classifier != "knn":
        raise click.UsageError("--classifier centroid is not compatible with DTW/xcorr")

    dtw_ndim = _import_dtw_ndim() if distance == "dtw" else None

    questions = load_classification_questions(data_root)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")
    dataset_label = data_root.name
    context_level, shot_level, is_few_shot = parse_context_and_shot(dataset_label)

    model_name = metrics_model_name or f"{distance}-{classifier}"
    limit_set = set(limit_categories)

    requested_k = int(knn_neighbors)
    base_k = max(1, requested_k) if requested_k > 0 else 1

    total_correct = 0.0
    total_count = 0
    results: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    def record(q, value) -> None:
        if metrics_output is not None:
            metrics_rows.append(
                _metrics_row(
                    q,
                    run_id,
                    model_name,
                    dataset_label,
                    context_level,
                    shot_level,
                    is_few_shot,
                    value,
                )
            )

    by_dataset: dict[str, list[Any]] = defaultdict(list)
    for q in questions:
        if limit_set and q.get("dataset") not in limit_set:
            continue
        by_dataset[str(q.get("dataset") or "")].append(q)

    name_width = max((len(name) for name in by_dataset.keys()), default=25)

    for dataset_name, ds_questions in sorted(by_dataset.items()):
        choices = list(ds_questions[0].get("multiple_choices") or [])
        label_to_index = build_label_to_index(choices)
        num_classes = len(choices)

        train_count = None
        correct = 0.0
        zero_shot = True
        disp_k = None
        disp_window = None
        cv_sum = 0.0
        cv_n = 0

        for q in ds_questions:
            if list(q.get("multiple_choices") or []) != choices:
                raise ValueError(f"Inconsistent multiple_choices for dataset {dataset_name}")

            support_paths = question_train_paths(q)

            # Zero-shot: report expected chance accuracy for aggregation
            if not support_paths:
                correct += 1.0 / num_classes if num_classes else 0.0
                record(q, np.nan)
                continue

            zero_shot = False
            if train_count is None:
                train_count = len(support_paths)

            y_support = infer_few_shot_labels(q, label_to_index=label_to_index)
            assert_strict_balanced_support(
                labels=y_support,
                num_classes=num_classes,
                choices=choices,
                question_id=str(q.get("question_id") or ""),
                dataset=dataset_name,
            )
            y_train = np.asarray(y_support, dtype=np.int64)
            y_true = int(label_to_index[str(q.get("label"))])

            scenario_k = base_k
            scenario_window = window
            cv_acc = float("nan")

            if distance == "dtw":
                if dtw_ndim is None:
                    raise RuntimeError("DTW requested but dtaidistance is unavailable.")

                train: list[np.ndarray] = []
                for raw_path in support_paths:
                    s = _prepare_series(
                        data_root / raw_path,
                        downsample=downsample,
                        max_len=max_len,  # default None => no resample
                        normalize=normalize,
                    )
                    train.append(np.asarray(s, dtype=np.float64, order="C"))

                if requested_k <= 0:
                    scenario_k = max(1, len(train))
                else:
                    scenario_window, scenario_k, cv_acc = _tune_dtw_knn(
                        dtw_ndim,
                        train,
                        y_train,
                        base_window=window,
                        base_k=base_k,
                        tune_window=tune_window,
                        window_grid=list(window_grid),
                        tune_knn=tune_knn,
                        knn_grid=list(knn_grid),
                    )
                    scenario_k = max(1, min(int(scenario_k), len(train)))

                if not q.get("test_samples"):
                    raise ValueError(f"Missing test_samples for {q.get('question_id')}")
                q_series = _prepare_series(
                    data_root / list(q.get("test_samples") or [])[0],
                    downsample=downsample,
                    max_len=max_len,
                    normalize=normalize,
                )
                pred = _predict_dtw_knn(
                    dtw_ndim,
                    train,
                    y_train,
                    np.asarray(q_series, dtype=np.float64, order="C"),
                    k=scenario_k,
                    window=scenario_window,
                )

            elif distance == "xcorr":
                train = [
                    _prepare_series(
                        data_root / raw_path,
                        downsample=downsample,
                        max_len=max_len,  # usually None => no resample
                        normalize=normalize,
                    )
                    for raw_path in support_paths
                ]
                if not q.get("test_samples"):
                    raise ValueError(f"Missing test_samples for {q.get('question_id')}")
                q_series = _prepare_series(
                    data_root / list(q.get("test_samples") or [])[0],
                    downsample=downsample,
                    max_len=max_len,
                    normalize=normalize,
                )

                if requested_k <= 0:
                    scenario_k = max(1, len(train))
                else:
                    scenario_k, cv_acc = _tune_xcorr_knn(
                        train,
                        y_train,
                        base_k=base_k,
                        tune_knn=tune_knn,
                        knn_grid=list(knn_grid),
                        max_lag=xcorr_max_lag,
                    )
                    scenario_k = max(1, min(int(scenario_k), len(train)))

                dists = np.asarray(
                    [_xcorr_distance(q_series, s, max_lag=xcorr_max_lag) for s in train],
                    dtype=np.float64,
                )
                pred = _vote_knn(y_train, dists, scenario_k)

            else:  # euclidean
                train_flat = np.stack(
                    [
                        _prepare_series(
                            data_root / raw_path,
                            downsample=downsample,
                            max_len=max_len,  # required for euclidean
                            normalize=normalize,
                        ).reshape(-1)
                        for raw_path in support_paths
                    ],
                    axis=0,
                )
                if not q.get("test_samples"):
                    raise ValueError(f"Missing test_samples for {q.get('question_id')}")
                query_flat = _prepare_series(
                    data_root / list(q.get("test_samples") or [])[0],
                    downsample=downsample,
                    max_len=max_len,
                    normalize=normalize,
                ).reshape(-1)

                if classifier == "knn":
                    if requested_k <= 0:
                        scenario_k = max(1, int(train_flat.shape[0]))
                    else:
                        scenario_k, cv_acc = _tune_euclidean_knn(
                            train_flat,
                            y_train,
                            base_k=base_k,
                            tune_knn=tune_knn,
                            knn_grid=list(knn_grid),
                        )
                        scenario_k = max(1, min(int(scenario_k), int(train_flat.shape[0])))

                    d = np.sum((train_flat - query_flat) ** 2, axis=1, dtype=np.float64)
                    pred = _vote_knn(y_train, d, scenario_k)
                else:
                    means: dict[int, np.ndarray] = {}
                    for lbl in np.unique(y_train):
                        means[int(lbl)] = train_flat[y_train == int(lbl)].mean(axis=0)
                    best = min(
                        (float(np.dot(m - query_flat, m - query_flat)), int(lbl))
                        for lbl, m in means.items()
                    )
                    pred = int(best[1])

            metric_value = int(int(pred) == int(y_true))
            correct += float(metric_value)
            record(q, metric_value)

            if disp_k is None:
                disp_k = int(scenario_k)
            if disp_window is None:
                disp_window = scenario_window
            if np.isfinite(cv_acc):
                cv_sum += float(cv_acc)
                cv_n += 1

        test_count = len(ds_questions)
        total_correct += correct
        total_count += int(test_count)
        acc = float(correct / test_count) if test_count else 0.0
        train_count_int = int(train_count or 0)
        prefix = (
            f"{dataset_name:<{name_width}} | train {train_count_int:3d} | "
            f"test {test_count:3d} | classes {num_classes:2d} | acc {acc:.3f}"
        )

        if zero_shot:
            print(f"{prefix} | zero-shot (expected chance)")
        elif distance == "dtw":
            win_disp = 0 if disp_window is None else int(disp_window)
            cv_str = "-" if cv_n == 0 else f"{(cv_sum / cv_n):.3f}"
            print(
                f"{prefix} | win {win_disp:3d} | k {int(disp_k):2d}/{train_count_int:2d} | cv {cv_str:>5s}"
            )
        elif distance == "xcorr":
            cv_str = "-" if cv_n == 0 else f"{(cv_sum / cv_n):.3f}"
            print(
                f"{prefix} | lag {xcorr_max_lag:3d} | k {int(disp_k):2d}/{train_count_int:2d} | cv {cv_str:>5s}"
            )
        elif classifier == "knn":
            cv_str = "-" if cv_n == 0 else f"{(cv_sum / cv_n):.3f}"
            print(f"{prefix} | k {int(disp_k):2d}/{train_count_int:2d} | cv {cv_str:>5s}")
        else:
            print(f"{prefix} | centroid")

        results.append({"dataset": dataset_name, "train": train_count_int, "test": test_count, "accuracy": acc})

    overall_acc = total_correct / total_count if total_count else 0.0
    print(f"\nOverall accuracy ({model_name}): {overall_acc:.3f} over {total_count} questions\n\n")

    if output:
        params = {
            "distance": distance,
            "classifier": classifier,
            "knn_neighbors": int(base_k) if classifier == "knn" else None,
            "normalize": bool(normalize),
            "downsample": downsample,
            "max_len": max_len,
            "window": window if distance == "dtw" else None,
            "tune_window": bool(tune_window) if distance == "dtw" else False,
            "window_grid": list(window_grid) if (distance == "dtw" and tune_window) else None,
            "tune_knn": bool(tune_knn) if classifier == "knn" else False,
            "knn_grid": list(knn_grid) if (tune_knn and classifier == "knn") else None,
            "dtw_backend": "dtaidistance" if distance == "dtw" else None,
            "xcorr_max_lag": xcorr_max_lag if distance == "xcorr" else None,
        }
        payload = {
            "overall_accuracy": float(overall_acc),
            "total_test": total_count,
            "per_category": results,
            "classifier": f"{distance}-{classifier}",
            "classifier_params": params,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if metrics_output is not None and metrics_rows:
        _write_metrics_csv(metrics_output, metrics_rows)


if __name__ == "__main__":
    main()
