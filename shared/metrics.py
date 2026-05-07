"""
Metrics and time-series helpers shared across workflows.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


def correlation(
    a: np.ndarray,
    b: np.ndarray,
    eps: float = 1e-20,
    *,
    context: str = "No Context",
    raise_on_numerical_warnings: bool = False,
    ignore_when_signals_are_equal_to_baseline: bool = True,
    logger_prefix: str = "Jacobian",
) -> float:
    """
    Pearson correlation with guards for flat/zero signals.
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {a.shape} vs {b.shape}")

    if a.size == 0:
        return np.nan
    if ignore_when_signals_are_equal_to_baseline:
        non_zero_mask = np.logical_or(a != 0, b != 0)
        if non_zero_mask.any():
            first_change = int(np.argmax(non_zero_mask))
            a = a[first_change:]
            b = b[first_change:]
        else:
            return np.nan

    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)

    na = np.linalg.norm(a_centered)
    nb = np.linalg.norm(b_centered)

    if na < eps and nb < eps:
        return 1.0
    if na < eps or nb < eps:
        return 0.0

    numerator = float(np.dot(a_centered, b_centered))
    denominator = na * nb

    if not raise_on_numerical_warnings:
        return abs(float(numerator / denominator))

    errstate_kwargs = {
        "divide": "raise",
        "invalid": "raise",
        "over": "raise",
    }
    try:
        with np.errstate(**errstate_kwargs):
            value = numerator / denominator
    except FloatingPointError as exc:  # pragma: no cover - defensive, env specific
        raise RuntimeError(
            f"{logger_prefix}: numerical warning while computing {context}"
        ) from exc
    if not np.isfinite(value):
        raise RuntimeError(
            f"{logger_prefix}: non-finite correlation while computing {context}"
        )
    return abs(float(value))


def downsample_sequence(sequence: np.ndarray, length: int) -> np.ndarray:
    """
    Impulse-aware thinning/repetition to a fixed length.

    Assumptions:
      - Last column is time (seconds) and MUST remain unchanged.
      - This function NEVER edits time values and NEVER reorders rows.
      - If upsampling is needed (T < length), it repeats existing rows
        (evenly spread) rather than creating synthetic/interpolated rows.
    """
    T, D = sequence.shape
    if T == 0 or length <= 0:
        return sequence[:0]

    if T == length:
        return sequence.copy()

    time = sequence[:, -1].astype(np.float64, copy=False)
    X = sequence[:, :-1].astype(np.float64, copy=False)

    def pick_even(idx: np.ndarray, k: int) -> np.ndarray:
        if k <= 0:
            return np.empty(0, dtype=idx.dtype)
        buckets = np.array_split(idx, k)
        out = [b[0] for b in buckets if b.size]
        if len(out) < k and idx.size:
            need = k - len(out)
            out += list(idx[:need])
        return np.asarray(out, dtype=idx.dtype)

    if T > 1 and X.shape[1] > 0:
        dX = np.diff(X, axis=0, prepend=X[:1])
        med = np.nanmedian(dX, axis=0)
        mad = np.nanmedian(np.abs(dX - med), axis=0)
        denom = 1.4826 * np.where(mad > 0.0, mad, np.inf)
        z_per_ch = np.abs(dX - med) / denom
        z = np.nanmax(np.nan_to_num(z_per_ch, nan=0.0), axis=1)
        impulse_mask = z > 8.0
    else:
        impulse_mask = np.zeros(T, dtype=bool)

    keep = np.zeros(T, dtype=bool)
    keep[0] = True
    keep[-1] = True

    if impulse_mask.any():
        dt = np.diff(time)
        dt_pos = dt[dt > 0]
        if dt_pos.size > 0:
            step = float(np.median(dt_pos))
            halfwin_t = 5.0 * step
            imp_times = time[impulse_mask]
            j = np.searchsorted(imp_times, time)
            j0 = np.clip(j - 1, 0, imp_times.size - 1)
            j1 = np.clip(j, 0, imp_times.size - 1)
            dist = np.minimum(np.abs(time - imp_times[j0]), np.abs(time - imp_times[j1]))
            keep |= dist <= halfwin_t
        else:
            halfwin_n = 5
            imp_idx = np.flatnonzero(impulse_mask)
            for i in imp_idx:
                lo = max(0, i - halfwin_n)
                hi = min(T, i + halfwin_n + 1)
                keep[lo:hi] = True

    keep_idx = np.flatnonzero(keep)

    if keep_idx.size >= length:
        chosen = np.sort(pick_even(keep_idx, length))
        return sequence[chosen]

    remaining = length - keep_idx.size
    rest_idx = np.setdiff1d(np.arange(T, dtype=int), keep_idx, assume_unique=True)

    if remaining > 0 and rest_idx.size > 0:
        targets = np.linspace(0, T - 1, num=remaining + 2)[1:-1]
        nearest = np.rint(targets).astype(int)
        nearest = np.clip(nearest, 0, T - 1)
        nearest = np.intersect1d(nearest, rest_idx, assume_unique=False)
        if nearest.size < remaining:
            need = remaining - nearest.size
            topup_pool = np.setdiff1d(rest_idx, nearest, assume_unique=False)
            topup = pick_even(topup_pool, need) if topup_pool.size else np.empty(0, dtype=int)
            fill_idx = np.unique(np.concatenate([nearest, topup]))
            if fill_idx.size < remaining:
                fill_idx = pick_even(rest_idx, remaining)
        else:
            _, first = np.unique(nearest, return_index=True)
            fill_idx = nearest[np.sort(first)][:remaining]
    else:
        fill_idx = np.empty(0, dtype=int)

    chosen = np.sort(np.concatenate([keep_idx, fill_idx]))

    if chosen.size != length:
        chosen = np.sort(pick_even(np.arange(T, dtype=int), length))

    return sequence[chosen]


def step(df: pd.DataFrame, target_column: str, time: float, end_value: float) -> pd.DataFrame:
    """Hard step for target_column: for t >= time, set to end_value."""
    assert "time" in df.columns, "DataFrame must contain a 'time' column."
    assert target_column in df.columns, f"Missing column: {target_column}"
    assert target_column != "time", "target_column must not be 'time'."

    out = df.copy()
    t = out["time"].to_numpy()

    mask = t >= time
    out.loc[mask, target_column] = end_value
    return out


def compute_channel_tolerance(
    b: pd.DataFrame, h: pd.DataFrame, identical_until: float
) -> Tuple[pd.Series, pd.Series]:
    """
    Infer per-channel tolerance from rows where index <= identical_until.
    Returns (tolerance, normalized_tolerance).
    """
    assert isinstance(identical_until, float)
    try:
        calib_mask = np.asarray(b.index.to_numpy() <= identical_until, dtype=bool)
    except Exception:
        calib_mask = np.zeros(len(b.index), dtype=bool)

    if calib_mask.any():
        cal_diff = (b[calib_mask] - h[calib_mask]).abs()
        tol_per_col = (
            cal_diff.max(axis=0, skipna=True)
            .reindex(b.columns)
            .fillna(0.0)
            .astype(float)
        )
    else:
        tol_per_col = pd.Series(0.0, index=b.columns, dtype=float)

    normalized_tol_per_col = tol_per_col / b[calib_mask].abs().max()
    return tol_per_col, normalized_tol_per_col


@dataclass
class Run:
    run_id: str
    data_path: Path
    df: pd.DataFrame
    scenario_id: Optional[str] = None
    modification_start_time: Optional[float] = None
    label: Optional[str] = None
    multiple_choices: Optional[List[str]] = None


@dataclass
class ScenarioPairwiseResult:
    scenario_id: Optional[str]
    per_signal: Dict[str, pl.DataFrame]
    aggregate: pl.DataFrame
    runs: List[Run]


_NO_SCENARIO_KEY = "__no_scenario__"


def _compute_vectorized_mse_with_tolerance(
    data_i_norm: np.ndarray,
    data_j_norm: np.ndarray,
    cutoff_time: float,
    common_grid: np.ndarray,
) -> np.ndarray:
    """Compute MSE per signal using vectorized operations."""

    n_points, n_signals = data_i_norm.shape

    start_idx = int(np.searchsorted(common_grid, cutoff_time, side="left"))
    if start_idx >= n_points:
        return np.full(n_signals, np.nan, dtype=float)

    diffs_norm = data_i_norm[start_idx:] - data_j_norm[start_idx:]
    mse_values = np.nanmean(diffs_norm**2, axis=0)
    return mse_values


def _preprocess_runs(
    runs: List[Run],
    normalize: bool,
    identical_until: float,
    *,
    raise_on_warning: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], List[str]]:
    """Pre-process runs into aligned arrays, with optional early-window normalization."""

    all_signals = set()
    for r in runs:
        all_signals.update(r.df.columns)
    signal_names = sorted(all_signals)

    common_grid = runs[0].df.index.to_numpy()

    raw_data: Dict[str, np.ndarray] = {}
    normalized_data: Dict[str, np.ndarray] = {}

    for r in runs:
        raw_array = r.df[signal_names].to_numpy()
        raw_data[r.run_id] = raw_array

        if normalize:
            time_values = r.df.index.to_numpy()
            baseline_mask = time_values <= identical_until
            baseline_slice = raw_array[baseline_mask]

            if baseline_slice.size == 0:
                baseline_slice = raw_array

            with warnings.catch_warnings():
                if raise_on_warning:
                    warnings.simplefilter("error", category=RuntimeWarning)
                baseline_mean = np.nanmean(baseline_slice, axis=0)
                baseline_std = np.nanstd(baseline_slice, axis=0)

                baseline_std = np.where(
                    np.isnan(baseline_std) | (baseline_std == 0.0), 1.0, baseline_std
                )
                normalized_array = (raw_array - baseline_mean) / baseline_std

            normalized_data[r.run_id] = normalized_array
        else:
            normalized_data[r.run_id] = raw_array

    return common_grid, normalized_data, raw_data, signal_names


def build_pairwise_mse_fast(
    runs: List[Run],
    normalize: bool = True,
    *,
    raise_on_warning: bool = False,
) -> Tuple[Dict[str, pl.DataFrame], pl.DataFrame]:
    """Vectorized pairwise MSE computation on a shared time grid."""

    if not runs:
        return {}, pl.DataFrame()

    modification_start_times = [
        r.modification_start_time for r in runs if r.modification_start_time is not None
    ]
    if modification_start_times:
        identical_until = float(min(modification_start_times))
    else:
        identical_until = float(runs[0].df.index.max())

    print(f"Pre-processing {len(runs)} runs to common grid...")
    common_grid, normalized_data, raw_data, signal_names = _preprocess_runs(
        runs,
        normalize,
        identical_until,
        raise_on_warning=raise_on_warning,
    )
    print(f"Using {len(common_grid)} unique time points across all runs")

    run_ids = [r.run_id for r in runs]
    n_runs = len(runs)
    n_signals = len(signal_names)

    norm_stack = np.stack([normalized_data[rid] for rid in run_ids], axis=0)
    mod_times = np.array(
        [
            r.modification_start_time
            if r.modification_start_time is not None
            else float(r.df.index.max())
            for r in runs
        ],
        dtype=float,
    )

    per_signal_values = np.full((n_signals, n_runs, n_runs), np.nan, dtype=float)
    diag_indices = np.arange(n_runs)
    per_signal_values[:, diag_indices, diag_indices] = 0.0

    total_pairs = n_runs * (n_runs - 1) // 2
    pair_count = 0

    print(f"Computing {total_pairs} pairwise comparisons...")
    for i in range(n_runs - 1):
        data_i_norm = norm_stack[i]
        mod_i = mod_times[i]

        for j in range(i + 1, n_runs):
            pair_count += 1

            if pair_count % 100 == 0 or pair_count == total_pairs:
                print(f"  Progress: {pair_count}/{total_pairs} pairs")

            cutoff = mod_i if mod_i < mod_times[j] else mod_times[j]

            mse_values = _compute_vectorized_mse_with_tolerance(
                data_i_norm,
                norm_stack[j],
                cutoff,
                common_grid,
            )

            per_signal_values[:, i, j] = mse_values
            per_signal_values[:, j, i] = mse_values

    print(f"Aggregating results across {n_signals} signals...")
    aggregate_values = np.nanmean(per_signal_values, axis=0)

    per_signal: Dict[str, pl.DataFrame] = {
        sig: pl.DataFrame(per_signal_values[idx], schema=run_ids)
        for idx, sig in enumerate(signal_names)
    }
    aggregate = pl.DataFrame(aggregate_values, schema=run_ids)
    return per_signal, aggregate


def compute_pairwise_mse_by_scenario(
    runs: List[Run],
    *,
    normalize: bool = True,
    use_fast: bool = True,
    raise_on_warning: bool = False,
) -> Dict[str, ScenarioPairwiseResult]:
    """Group runs by scenario and compute pairwise MSE matrices within each group."""

    if not runs:
        return {}

    grouped: Dict[str, List[Run]] = {}
    for r in runs:
        key = r.scenario_id if r.scenario_id else _NO_SCENARIO_KEY
        grouped.setdefault(key, []).append(r)

    results: Dict[str, ScenarioPairwiseResult] = {}
    for scenario_key, scenario_runs in grouped.items():
        if len(scenario_runs) < 2:
            continue

        print(f"\n[{scenario_key}] Processing {len(scenario_runs)} runs...")

        if use_fast:
            per_signal, aggregate = build_pairwise_mse_fast(
                scenario_runs,
                normalize=normalize,
                raise_on_warning=raise_on_warning,
            )
        else:
            per_signal, aggregate = build_pairwise_mse_fast(
                scenario_runs,
                normalize=normalize,
                raise_on_warning=raise_on_warning,
            )

        results[scenario_key] = ScenarioPairwiseResult(
            scenario_id=None if scenario_key == _NO_SCENARIO_KEY else scenario_key,
            per_signal=per_signal,
            aggregate=aggregate,
            runs=scenario_runs,
        )

    return results


def select_best_mse_matrix(
    mse_matrices: Union[List[pd.DataFrame], np.ndarray]
) -> Tuple[np.ndarray, float, int]:
    """Return the matrix with the largest minimum off-diagonal separation."""

    if isinstance(mse_matrices, list):
        if not mse_matrices:
            raise ValueError("mse_matrices list cannot be empty")
        arr = np.stack([m.to_numpy(copy=False) for m in mse_matrices])
    else:
        arr = np.asarray(mse_matrices, dtype=float)

    if arr.ndim != 3:
        raise ValueError("mse_matrices must be a 3D array")

    n_combos, n_rows, n_cols = arr.shape
    if n_rows != n_cols:
        raise ValueError("MSE matrices must be square")
    if n_rows <= 1:
        raise ValueError("Need at least two choices to compare")

    off_diag_mask = ~np.eye(n_rows, dtype=bool)
    off_diag_vals = arr[:, off_diag_mask]
    finite_mask = np.isfinite(off_diag_vals)
    valid_combo_mask = np.any(finite_mask, axis=1)

    if not np.any(valid_combo_mask):
        raise ValueError("No valid matrices found with finite off-diagonal values")

    safe_vals = np.where(finite_mask, off_diag_vals, np.inf)
    min_distances = np.full(n_combos, -np.inf, dtype=float)
    min_distances[valid_combo_mask] = safe_vals[valid_combo_mask].min(axis=1)

    best_index = int(np.argmax(min_distances))
    best_min_distance = float(min_distances[best_index])

    if not math.isfinite(best_min_distance):
        raise ValueError("Unable to determine best matrix due to non-finite distances")

    return arr[best_index], best_min_distance, best_index


def build_choice_distance_matrices(
    runs: List[Run],
    aggregate: Union[pd.DataFrame, pl.DataFrame],
    exclude_self: bool = True,
) -> Tuple[np.ndarray, List[Tuple[str, ...]], List[str]]:
    """Vectorized construction of choice×choice distance matrices."""

    run_to_choice: Dict[str, str] = {}
    all_choices_set = set()

    if isinstance(aggregate, pd.DataFrame):
        available_run_ids = set(map(str, aggregate.index))
        agg_array = aggregate.to_numpy()
        run_ids_in_agg = list(map(str, aggregate.index))
    elif isinstance(aggregate, pl.DataFrame):
        available_run_ids = set(map(str, aggregate.columns))
        agg_array = aggregate.to_numpy()
        run_ids_in_agg = [str(r.run_id) for r in runs]
        if agg_array.shape[0] != len(run_ids_in_agg):
            run_ids_in_agg = list(map(str, aggregate.columns))
    else:
        raise TypeError("aggregate must be a pandas or polars DataFrame")

    agg_array = agg_array.astype(float, copy=False)

    for run in runs:
        run_id = str(run.run_id)
        if run.label and run_id in available_run_ids:
            run_to_choice[run_id] = run.label
            all_choices_set.add(run.label)

    if not run_to_choice:
        raise ValueError("No runs with label information found")

    choices = sorted(all_choices_set)
    n_choices = len(choices)

    runs_by_choice: Dict[str, List[str]] = {c: [] for c in choices}
    for run_id, choice in run_to_choice.items():
        runs_by_choice[choice].append(run_id)

    run_id_to_idx = {rid: idx for idx, rid in enumerate(run_ids_in_agg)}
    choice_run_lists = [runs_by_choice[c] for c in choices]
    all_combinations = list(product(*choice_run_lists))
    n_combos = len(all_combinations)

    print(
        f"Building {n_combos} combination matrices from {n_choices} choices (vectorized mode)..."
    )

    if n_combos == 0:
        return np.empty((0, 0, 0), dtype=float), [], choices

    combo_indices = np.array(
        [[run_id_to_idx.get(rid, -1) for rid in combo] for combo in all_combinations],
        dtype=int,
    )

    valid_mask = np.all(combo_indices != -1, axis=1)
    valid_positions = np.flatnonzero(valid_mask)

    all_matrices = np.full((n_combos, n_choices, n_choices), np.nan, dtype=float)

    if valid_positions.size:
        valid_indices = combo_indices[valid_mask]
        sub_matrices = agg_array[
            valid_indices[:, :, None], valid_indices[:, None, :]
        ].astype(float, copy=False)

        if exclude_self:
            same_run_mask = valid_indices[:, :, None] == valid_indices[:, None, :]
            sub_matrices = np.where(same_run_mask, np.nan, sub_matrices)

        all_matrices[valid_positions] = sub_matrices

    return all_matrices, all_combinations, choices


__all__ = [
    "Run",
    "ScenarioPairwiseResult",
    "_NO_SCENARIO_KEY",
    "build_pairwise_mse_fast",
    "build_choice_distance_matrices",
    "compute_channel_tolerance",
    "compute_pairwise_mse_by_scenario",
    "correlation",
    "downsample_sequence",
    "select_best_mse_matrix",
    "step",
]
