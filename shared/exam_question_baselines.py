"""Baseline utilities for exam-question generation.

These baselines are used to populate question is_correct_* fields for
classification tasks. They are intentionally lightweight (numpy-only) so they
can be shared across generators.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation between two same-shaped arrays.

    Returns -inf when correlation is undefined (e.g. zero-variance vector).
    """

    if a.shape != b.shape:
        raise ValueError(f"correlation expects equal shapes, got {a.shape} vs {b.shape}")
    a_arr = np.asarray(a, dtype=np.float32).ravel()
    b_arr = np.asarray(b, dtype=np.float32).ravel()
    a_centered = a_arr - float(a_arr.mean())
    b_centered = b_arr - float(b_arr.mean())
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 0.0:
        return float("-inf")
    return float(np.dot(a_centered, b_centered) / denom)


def predict_correlation_nn(
    train_flat: np.ndarray,
    y_train: np.ndarray,
    query_flat: np.ndarray,
) -> int:
    """1-NN classifier using Pearson correlation (higher is closer)."""

    best_label = int(y_train[0])
    best_score = float("-inf")
    for row, label in zip(train_flat, y_train):
        score = pearson_correlation(row, query_flat)
        if score > best_score:
            best_score = score
            best_label = int(label)
    return best_label


def predict_euclidean_knn(train_flat: np.ndarray, y_train: np.ndarray, query_flat: np.ndarray) -> int:
    """1-NN classifier using squared Euclidean distance (lower is closer)."""

    distances = np.sum((train_flat - query_flat) ** 2, axis=1, dtype=np.float64)
    nn_idx = int(np.argmin(distances))
    return int(y_train[nn_idx])


def predict_euclidean_centroid(class_means: Mapping[int, np.ndarray], query_flat: np.ndarray) -> int:
    """Nearest class-mean classifier using squared Euclidean distance."""

    best_label: Optional[int] = None
    best_dist = float("inf")
    for label, mean_vec in class_means.items():
        diff = mean_vec - query_flat
        dist = float(np.dot(diff, diff))
        if dist < best_dist:
            best_label = int(label)
            best_dist = dist
        elif dist == best_dist and best_label is not None and int(label) < best_label:
            best_label = int(label)
    if best_label is None:
        raise ValueError("No class means available for centroid prediction.")
    return best_label


def build_class_means(train_flat: np.ndarray, y_train: np.ndarray) -> Dict[int, np.ndarray]:
    """Return label -> mean vector for centroid baseline."""

    y_train_arr = np.asarray(y_train, dtype=np.int64)
    means: Dict[int, np.ndarray] = {}
    for label_int in sorted({int(v) for v in y_train_arr.tolist()}):
        mask = y_train_arr == label_int
        means[label_int] = np.asarray(train_flat[mask]).mean(axis=0)
    return means


def baseline_correctness_fixed_length(
    *,
    train_flat: np.ndarray,
    y_train: np.ndarray,
    class_means: Mapping[int, np.ndarray],
    query_flat: np.ndarray,
    correct_label: int,
) -> Dict[str, bool]:
    pred_knn = predict_euclidean_knn(train_flat, y_train, query_flat)
    pred_centroid = predict_euclidean_centroid(class_means, query_flat)
    pred_corr = predict_correlation_nn(train_flat, y_train, query_flat)
    return {
        "is_correct_euclidean_knn": bool(pred_knn == int(correct_label)),
        "is_correct_euclidean_centroid": bool(pred_centroid == int(correct_label)),
        "is_correct_correlation": bool(pred_corr == int(correct_label)),
    }


def euclidean_truncated_rmse(question: np.ndarray, sample: np.ndarray) -> float:
    """RMSE computed after truncating to the shortest run length (time axis)."""

    rows = min(int(question.shape[0]), int(sample.shape[0]))
    if rows <= 0:
        return float("inf")
    diff = question[:rows] - sample[:rows]
    return float(np.sqrt(np.mean(diff * diff)))


def _correlation_score_truncated(question: np.ndarray, sample: np.ndarray) -> float:
    rows = min(int(question.shape[0]), int(sample.shape[0]))
    if rows <= 0:
        return float("-inf")
    question_flat = question[:rows].reshape(-1)
    sample_flat = sample[:rows].reshape(-1)
    return pearson_correlation(question_flat, sample_flat)


def predict_euclidean_knn_truncated(
    question: np.ndarray,
    samples_by_id: Mapping[str, np.ndarray],
    shots_by_label: Mapping[str, Sequence[str]],
    *,
    distance_fn=euclidean_truncated_rmse,
) -> Optional[str]:
    best_label: Optional[str] = None
    best_distance = float("inf")
    for label, sample_ids in shots_by_label.items():
        for sample_id in sample_ids:
            sample = samples_by_id.get(sample_id)
            if sample is None:
                continue
            dist = float(distance_fn(question, sample))
            if dist < best_distance or (
                dist == best_distance and best_label is not None and str(label) < best_label
            ):
                best_distance = dist
                best_label = str(label)
    return best_label


def predict_euclidean_centroid_truncated(
    question: np.ndarray,
    samples_by_id: Mapping[str, np.ndarray],
    shots_by_label: Mapping[str, Sequence[str]],
    *,
    distance_fn=euclidean_truncated_rmse,
) -> Optional[str]:
    best_label: Optional[str] = None
    best_distance = float("inf")
    for label, sample_ids in shots_by_label.items():
        mats: List[np.ndarray] = []
        for sample_id in sample_ids:
            mat = samples_by_id.get(sample_id)
            if mat is not None:
                mats.append(mat)
        if not mats:
            continue
        min_rows = min(int(mat.shape[0]) for mat in mats)
        if min_rows <= 0:
            continue
        stacked = np.stack([mat[:min_rows] for mat in mats], axis=0)
        centroid = stacked.mean(axis=0)
        dist = float(distance_fn(question, centroid))
        if dist < best_distance or (
            dist == best_distance and best_label is not None and str(label) < best_label
        ):
            best_distance = dist
            best_label = str(label)
    return best_label


def predict_correlation_nn_truncated(
    question: np.ndarray,
    samples_by_id: Mapping[str, np.ndarray],
    shots_by_label: Mapping[str, Sequence[str]],
    *,
    score_fn=_correlation_score_truncated,
) -> Optional[str]:
    best_label: Optional[str] = None
    best_score = float("-inf")
    for label, sample_ids in shots_by_label.items():
        for sample_id in sample_ids:
            sample = samples_by_id.get(sample_id)
            if sample is None:
                continue
            score = float(score_fn(question, sample))
            if score > best_score or (
                score == best_score and best_label is not None and str(label) < best_label
            ):
                best_score = score
                best_label = str(label)
    return best_label


def baseline_correctness_truncated(
    *,
    question: np.ndarray,
    correct_label: str,
    samples_by_id: Mapping[str, np.ndarray],
    shots_by_label: Mapping[str, Sequence[str]],
) -> Dict[str, bool]:
    pred_knn = predict_euclidean_knn_truncated(question, samples_by_id, shots_by_label)
    pred_centroid = predict_euclidean_centroid_truncated(question, samples_by_id, shots_by_label)
    pred_corr = predict_correlation_nn_truncated(question, samples_by_id, shots_by_label)
    return {
        "is_correct_euclidean_knn": pred_knn == correct_label if pred_knn is not None else False,
        "is_correct_euclidean_centroid": pred_centroid == correct_label if pred_centroid is not None else False,
        "is_correct_correlation": pred_corr == correct_label if pred_corr is not None else False,
    }


__all__ = [
    "baseline_correctness_fixed_length",
    "baseline_correctness_truncated",
    "build_class_means",
    "euclidean_truncated_rmse",
    "pearson_correlation",
    "predict_correlation_nn",
    "predict_euclidean_centroid",
    "predict_euclidean_centroid_truncated",
    "predict_euclidean_knn",
    "predict_euclidean_knn_truncated",
    "predict_correlation_nn_truncated",
]
