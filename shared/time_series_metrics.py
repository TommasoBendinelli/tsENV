from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _quantize_time_to_saved_precision(time: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(time, dtype=np.float32)


def load_run_df(run_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load a run's time series DataFrame from a run directory.

    Expects either data.parquet or data.csv. Ensures a numeric 'time' column.
    """
    parquet_path = run_dir / "data.parquet"
    csv_path = run_dir / "data.csv"
    try:
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            return None
    except Exception:
        return None

    if "time" not in df.columns:
        return None
    time = pd.to_numeric(df["time"], errors="coerce")
    if time.isna().any():
        return None
    df = df.copy()
    df["time"] = time.astype(float)
    return df


def df_with_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        out = df.copy()
        out["time"] = pd.to_numeric(out["time"], errors="coerce")
        return out
    out = df.copy()
    out["time"] = pd.to_numeric(out.index, errors="coerce")
    out = out.reset_index(drop=True)
    return out


def validate_time_series_frame(df: pd.DataFrame, *, context: str) -> None:
    if "time" not in df.columns:
        raise AssertionError(f"Missing time column for {context}")
    time = pd.to_numeric(df["time"], errors="coerce")
    if time.isna().any():
        raise AssertionError(f"Non-numeric time column for {context}")
    if not np.isfinite(time.to_numpy(dtype=float)).all():
        raise AssertionError(f"Non-finite time column for {context}")


def _collapse_by_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_time = pd.to_numeric(out["time"], errors="coerce")
    out["time"] = _quantize_time_to_saved_precision(numeric_time)
    out = out.dropna(subset=["time"])
    out = out.groupby("time", as_index=False).mean(numeric_only=True).sort_values("time")
    return out.reset_index(drop=True)


def _prepare_aligned_numeric_frames(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    align_on_overlap: bool,
    time_round_decimals: int,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]]:
    if baseline_df.empty or run_df.empty:
        return None

    validate_time_series_frame(baseline_df, context="baseline_df")
    validate_time_series_frame(run_df, context="run_df")

    base_work = _collapse_by_time(baseline_df)
    run_work = _collapse_by_time(run_df)
    if base_work.empty or run_work.empty:
        return None

    base_time = base_work["time"].to_numpy(dtype=float)
    run_time = run_work["time"].to_numpy(dtype=float)

    if align_on_overlap:
        base_key = np.round(base_time, int(time_round_decimals))
        run_key = np.round(run_time, int(time_round_decimals))
        common = np.intersect1d(base_key, run_key)
        if common.size == 0:
            return None
        base_idx = np.searchsorted(base_key, common)
        run_idx = np.searchsorted(run_key, common)
        base_work = base_work.iloc[base_idx].reset_index(drop=True)
        run_work = run_work.iloc[run_idx].reset_index(drop=True)
    else:
        if base_time.shape != run_time.shape or not np.allclose(
            base_time, run_time, rtol=0.0, atol=0.0
        ):
            return None

    base_numeric = base_work.select_dtypes(include=[np.number]).drop(
        columns=["time"], errors="ignore"
    )
    run_numeric = run_work.select_dtypes(include=[np.number]).drop(
        columns=["time"], errors="ignore"
    )
    common_cols = [str(c) for c in base_numeric.columns if c in run_numeric.columns]
    if not common_cols:
        return None

    aligned_time = run_work["time"].to_numpy(dtype=float)
    return base_work, run_work, aligned_time, common_cols


def prepare_common_numeric_arrays(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    align_on_overlap: bool = False,
    time_round_decimals: int = 9,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    """
    Prepare aligned numeric arrays for (baseline_df, run_df).

    Returns (baseline_vals, run_vals, finite_mask, signals).
    - Uses only columns common to both frames.
    - Requires identical 'time' columns (shape and values).
    """
    if baseline_df.empty or run_df.empty:
        return None

    aligned = _prepare_aligned_numeric_frames(
        baseline_df=baseline_df,
        run_df=run_df,
        align_on_overlap=align_on_overlap,
        time_round_decimals=time_round_decimals,
    )
    if aligned is None:
        return None
    base_work, run_work, _, common_cols = aligned

    base_vals = base_work[common_cols].to_numpy(dtype=float)
    run_vals = run_work[common_cols].to_numpy(dtype=float)
    mask = np.isfinite(base_vals) & np.isfinite(run_vals)
    if not np.any(mask):
        return None
    return base_vals, run_vals, mask, common_cols


def compute_euclid_distance_and_norm(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    align_on_overlap: bool = False,
    time_round_decimals: int = 9,
) -> Optional[tuple[float, float]]:
    prepared = prepare_common_numeric_arrays(
        baseline_df=baseline_df,
        run_df=run_df,
        align_on_overlap=align_on_overlap,
        time_round_decimals=time_round_decimals,
    )
    if prepared is None:
        return None
    base_vals, run_vals, mask, _ = prepared
    diff = run_vals - base_vals
    diff[~mask] = 0.0
    base_masked = base_vals.copy()
    base_masked[~mask] = 0.0
    dist = float(np.sqrt(np.sum(np.square(diff, dtype=np.float64), dtype=np.float64)))
    norm = float(np.sqrt(np.sum(np.square(base_masked, dtype=np.float64), dtype=np.float64)))
    return dist, norm


def compute_first_detectable_time(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    first_detectable_minimum_symmetric_distance: float,
    first_detectable_epsilon: float = 0.001,
    time_round_decimals: int = 9,
) -> Optional[float]:
    aligned = _prepare_aligned_numeric_frames(
        baseline_df=baseline_df,
        run_df=run_df,
        align_on_overlap=False,
        time_round_decimals=time_round_decimals,
    )
    if aligned is None:
        return None
    base_work, run_work, run_time, common_cols = aligned
    base_time = base_work["time"].to_numpy(dtype=float)
    if base_time.shape != run_time.shape or not np.allclose(
        base_time, run_time, rtol=0.0, atol=0.0
    ):
        raise AssertionError("baseline_df and run_df time columns must be identical")

    base_vals = base_work[common_cols].to_numpy(dtype=float)
    run_vals = run_work[common_cols].to_numpy(dtype=float)
    denominator = np.abs(base_vals) + np.abs(run_vals) + float(first_detectable_epsilon)
    with np.errstate(divide="ignore", invalid="ignore"):
        symmetric_distance = (2.0 * np.abs(base_vals - run_vals)) / denominator
    valid = (
        np.isfinite(base_vals)
        & np.isfinite(run_vals)
        & np.isfinite(symmetric_distance)
    )
    tolerance = float(first_detectable_minimum_symmetric_distance)
    detectable = np.any((symmetric_distance >= tolerance) & valid, axis=1)
    if not np.any(detectable):
        # -1 is the convention in model_record.schema.json (see shared/interface/model_record_json.py).
        return -1.0
    first_idx = int(np.argmax(detectable))
    return float(run_time[first_idx])


def _detectability_error_payload() -> dict[str, Any]:
    return {
        "environment_specific_detectability": "error",
        "max_SRD_detectability": "error",
        "detectability": "no",
        "detectable": "error",
        "max_SRD": [],
        "euclidean_distance": [],
        "mean_euclidean_distance_clean_dirty": [],
        "mean_euclidean_distance_clean_baseline": [],
        "mean_SNR": [],
        "first_diff": [],
    }


def _coerce_minimum_consecutive_srd_steps(value: object) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("minimum_consecutive_srd_steps must be an integer") from exc
    if parsed < 1:
        return 1
    return parsed


def _first_consecutive_true_index(
    mask: np.ndarray,
    *,
    minimum_steps: int,
) -> Optional[int]:
    if minimum_steps <= 1:
        if not np.any(mask):
            return None
        return int(np.argmax(mask))
    if mask.size < minimum_steps:
        return None
    for start_idx in range(0, int(mask.size) - int(minimum_steps) + 1):
        if bool(np.all(mask[start_idx : start_idx + minimum_steps])):
            return start_idx
    return None


def _compute_detectability_values(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    first_detectable_minimum_symmetric_distance: float,
    first_detectable_epsilon: float = 0.001,
    minimum_consecutive_srd_steps: int = 1,
    intervention_time: object = 0.0,
    signal_detectability_specs: Optional[Mapping[str, Mapping[str, object]]] = None,
    require_signal_detectability_specs: bool = False,
    time_round_decimals: int = 9,
) -> tuple[list[float], list[Optional[float]], list[float]]:
    _ = time_round_decimals
    consecutive_steps = _coerce_minimum_consecutive_srd_steps(
        minimum_consecutive_srd_steps
    )
    if baseline_df.empty or run_df.empty:
        raise ValueError("baseline_df and run_df must be non-empty")
    if not baseline_df.columns.size or not run_df.columns.size:
        raise ValueError("baseline_df and run_df must contain columns")
    if str(baseline_df.columns[-1]) != "time" or str(run_df.columns[-1]) != "time":
        raise ValueError("last column for baseline_df and run_df must be time")

    validate_time_series_frame(baseline_df, context="baseline_df")
    validate_time_series_frame(run_df, context="run_df")

    base_work = baseline_df.copy()
    run_work = run_df.copy()
    base_time = pd.to_numeric(base_work["time"], errors="coerce").to_numpy(dtype=float)
    run_time = pd.to_numeric(run_work["time"], errors="coerce").to_numpy(dtype=float)
    if base_time.shape != run_time.shape or not np.allclose(
        base_time, run_time, rtol=0.0, atol=0.0
    ):
        raise ValueError("baseline_df and run_df time columns must be identical")

    signal_columns = [str(column) for column in baseline_df.columns[:-1]]
    if signal_columns != [str(column) for column in run_df.columns[:-1]]:
        raise ValueError("baseline_df and run_df signal columns must be identical")
    if not signal_columns:
        raise ValueError("baseline_df and run_df must contain signal columns")

    base_vals = (
        base_work[signal_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    run_vals = (
        run_work[signal_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    threshold = float(first_detectable_minimum_symmetric_distance)
    epsilon = float(first_detectable_epsilon)
    intervention_time_value = _intervention_time_value(intervention_time)

    first_diff: list[Optional[float]] = []
    max_srd: list[float] = []
    euclidean_distance: list[float] = []
    for col_idx in range(len(signal_columns)):
        signal_name = signal_columns[col_idx]
        signal_spec = (
            signal_detectability_specs.get(signal_name)
            if signal_detectability_specs is not None
            else None
        )
        if require_signal_detectability_specs and not isinstance(signal_spec, Mapping):
            raise ValueError(f"missing detectability config for signal {signal_name!r}")
        signal_threshold = (
            float(signal_spec.get("min_srd_distance"))
            if isinstance(signal_spec, Mapping) and "min_srd_distance" in signal_spec
            else threshold
        )
        signal_epsilon = (
            float(signal_spec.get("epsilon_SRD"))
            if isinstance(signal_spec, Mapping) and "epsilon_SRD" in signal_spec
            else epsilon
        )
        signal_consecutive_steps = (
            _coerce_minimum_consecutive_srd_steps(
                signal_spec.get("minimum_consecutive_srd_steps")
            )
            if isinstance(signal_spec, Mapping)
            and "minimum_consecutive_srd_steps" in signal_spec
            else consecutive_steps
        )
        signal_denominator = (
            np.abs(base_vals[:, col_idx])
            + np.abs(run_vals[:, col_idx])
            + signal_epsilon
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            column_distance = (
                2.0 * np.abs(base_vals[:, col_idx] - run_vals[:, col_idx])
            ) / signal_denominator
        column_valid = (
            np.isfinite(base_vals[:, col_idx])
            & np.isfinite(run_vals[:, col_idx])
            & np.isfinite(column_distance)
        )
        finite_distance = column_distance[column_valid]
        max_srd.append(
            0.0 if finite_distance.size == 0 else float(np.max(finite_distance))
        )
        column_delta = run_vals[:, col_idx] - base_vals[:, col_idx]
        finite_delta = column_delta[column_valid]
        euclidean_distance.append(
            0.0
            if finite_delta.size == 0
            else float(np.sqrt(np.sum(np.square(finite_delta, dtype=np.float64))))
        )
        detectable_mask = (
            column_valid
            & (base_time >= intervention_time_value)
            & (column_distance >= signal_threshold)
        )
        first_idx = _first_consecutive_true_index(
            detectable_mask,
            minimum_steps=signal_consecutive_steps,
        )
        if first_idx is None:
            first_diff.append(None)
            continue
        first_diff.append(float(base_time[first_idx]))

    return max_srd, first_diff, euclidean_distance


def _intervention_time_value(intervention_time: object) -> float:
    try:
        value = float(intervention_time)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _rms_threshold_passes(
    *,
    signal_columns: Sequence[str],
    mean_euclidean_distance_clean_baseline: Optional[Sequence[float]],
    RMS_thresholds: Optional[Mapping[str, float]],
) -> bool:
    if not RMS_thresholds:
        return False
    for signal, distance in zip(
        signal_columns,
        list(mean_euclidean_distance_clean_baseline or []),
    ):
        if signal not in RMS_thresholds:
            continue
        try:
            threshold = float(RMS_thresholds[signal])
            parsed_distance = float(distance)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(threshold) and np.isfinite(parsed_distance)):
            continue
        if parsed_distance > threshold:
            return True
    return False


def _signal_to_noise_ratio_db_values(
    *,
    mean_euclidean_distance_clean_dirty: Optional[Sequence[float]],
    mean_euclidean_distance_clean_baseline: Optional[Sequence[float]],
    length: Optional[int] = None,
) -> list[Optional[float]]:
    dirty_distances = list(mean_euclidean_distance_clean_dirty or [])
    baseline_distances = list(mean_euclidean_distance_clean_baseline or [])
    if length is None:
        length = max(len(dirty_distances), len(baseline_distances))
    out: list[Optional[float]] = []
    for idx in range(length):
        try:
            noise = float(dirty_distances[idx])
            signal_norm = float(baseline_distances[idx])
        except (IndexError, TypeError, ValueError):
            out.append(None)
            continue
        if not (np.isfinite(noise) and np.isfinite(signal_norm)):
            out.append(None)
            continue
        if noise <= 0.0 or signal_norm <= 0.0:
            out.append(None)
            continue
        snr_db = float(20.0 * np.log10(signal_norm / noise))
        out.append(snr_db if np.isfinite(snr_db) else None)
    return out


def _detectability_payload(
    *,
    detectable: str,
    max_srd: list[float],
    first_diff: list[Optional[float]],
    euclidean_distance: list[float],
    mean_euclidean_distance_clean_dirty: Optional[Sequence[float]] = None,
    mean_euclidean_distance_clean_baseline: Optional[Sequence[float]] = None,
    mean_SNR: Optional[Sequence[Optional[float]]] = None,
    environment_specific_detectability: Optional[str] = None,
) -> dict[str, Any]:
    max_status = str(detectable or "error").strip().lower() or "error"
    env_status = (
        str(environment_specific_detectability).strip().lower()
        if environment_specific_detectability is not None
        else ("yes" if max_status == "yes" else ("error" if max_status == "error" else "no"))
    )
    if env_status not in {"yes", "no", "error"}:
        env_status = "error"
    final_detectability = "yes" if max_status == "yes" and env_status == "yes" else "no"
    return {
        "environment_specific_detectability": env_status,
        "max_SRD_detectability": max_status,
        "detectability": final_detectability,
        "detectable": final_detectability,
        "max_SRD": max_srd,
        "euclidean_distance": euclidean_distance,
        "mean_euclidean_distance_clean_dirty": list(
            mean_euclidean_distance_clean_dirty or []
        ),
        "mean_euclidean_distance_clean_baseline": list(
            mean_euclidean_distance_clean_baseline or []
        ),
        "mean_SNR": list(mean_SNR)
        if mean_SNR is not None
        else _signal_to_noise_ratio_db_values(
            mean_euclidean_distance_clean_dirty=mean_euclidean_distance_clean_dirty,
            mean_euclidean_distance_clean_baseline=mean_euclidean_distance_clean_baseline,
            length=len(first_diff),
        ),
        "first_diff": first_diff,
    }


def compute_detectability_baseline(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    first_detectable_minimum_symmetric_distance: float,
    first_detectable_epsilon: float = 0.001,
    minimum_consecutive_srd_steps: int = 1,
    intervention_time: object = 0.0,
    signal_detectability_specs: Optional[Mapping[str, Mapping[str, object]]] = None,
    require_signal_detectability_specs: bool = False,
    mean_euclidean_distance_clean_dirty: Optional[Sequence[float]] = None,
    mean_euclidean_distance_clean_baseline: Optional[Sequence[float]] = None,
    mean_SNR: Optional[Sequence[Optional[float]]] = None,
    RMS_thresholds: Optional[Mapping[str, float]] = None,
    time_round_decimals: int = 9,
) -> dict[str, Any]:
    error_payload = _detectability_error_payload()
    try:
        max_srd, first_diff, euclidean_distance = _compute_detectability_values(
            baseline_df=baseline_df,
            run_df=run_df,
            first_detectable_minimum_symmetric_distance=(
                first_detectable_minimum_symmetric_distance
            ),
            first_detectable_epsilon=first_detectable_epsilon,
            minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
            intervention_time=intervention_time,
            signal_detectability_specs=signal_detectability_specs,
            require_signal_detectability_specs=require_signal_detectability_specs,
            time_round_decimals=time_round_decimals,
        )
        signal_columns = [str(column) for column in baseline_df.columns[:-1]]
        resolved_mean_snr = (
            list(mean_SNR)
            if mean_SNR is not None
            else _signal_to_noise_ratio_db_values(
                mean_euclidean_distance_clean_dirty=mean_euclidean_distance_clean_dirty,
                mean_euclidean_distance_clean_baseline=mean_euclidean_distance_clean_baseline,
            )
        )
        rms_passes = _rms_threshold_passes(
            signal_columns=signal_columns,
            mean_euclidean_distance_clean_baseline=(
                mean_euclidean_distance_clean_baseline
            ),
            RMS_thresholds=RMS_thresholds,
        )
        detectable = "yes" if rms_passes else "no"
        return _detectability_payload(
            detectable=detectable,
            max_srd=max_srd,
            euclidean_distance=euclidean_distance,
            mean_euclidean_distance_clean_dirty=mean_euclidean_distance_clean_dirty,
            mean_euclidean_distance_clean_baseline=mean_euclidean_distance_clean_baseline,
            mean_SNR=resolved_mean_snr,
            first_diff=first_diff,
        )
    except Exception:
        return error_payload


def _documented_detectability_output(payload: Mapping[str, Any]) -> dict[str, Any]:
    first_diff = list(payload.get("first_diff") or [])
    clean_dirty = list(payload.get("mean_euclidean_distance_clean_dirty") or [])
    clean_baseline = list(payload.get("mean_euclidean_distance_clean_baseline") or [])
    snr = list(payload.get("mean_SNR") or [])
    if first_diff and not clean_dirty:
        clean_dirty = [0.0 for _ in first_diff]
    if first_diff and not clean_baseline:
        clean_baseline = list(payload.get("euclidean_distance") or [])
    if len(clean_baseline) < len(first_diff):
        clean_baseline = [
            *clean_baseline,
            *([0.0] * (len(first_diff) - len(clean_baseline))),
        ]
    if len(clean_dirty) < len(first_diff):
        clean_dirty = [
            *clean_dirty,
            *([0.0] * (len(first_diff) - len(clean_dirty))),
        ]
    if len(snr) < len(first_diff):
        computed_snr = _signal_to_noise_ratio_db_values(
            mean_euclidean_distance_clean_dirty=clean_dirty,
            mean_euclidean_distance_clean_baseline=clean_baseline,
            length=len(first_diff),
        )
        snr = [*(snr or []), *computed_snr[len(snr) :]]
    return {
        "mean_euclidean_distance_clean_dirty": clean_dirty[: len(first_diff)],
        "mean_euclidean_distance_clean_baseline": clean_baseline[: len(first_diff)],
        "mean_SNR": snr[: len(first_diff)],
        "first_diff": first_diff,
    }


def compute_baseline_detectability(**kwargs: Any) -> dict[str, Any]:
    payload = compute_detectability_baseline(**kwargs)
    env_status = str(
        payload.get("environment_specific_detectability") or "error"
    ).strip().lower()
    if env_status not in {"yes", "no", "error"}:
        env_status = "error"
    detectable = str(payload.get("detectable") or "error").strip().lower()
    if str(payload.get("max_SRD_detectability") or "").strip().lower() == "error":
        detectable = "error"
    if detectable not in {"yes", "no", "error"}:
        detectable = "error"
    return {
        "environment_specific_detectability": env_status,
        "detectable": detectable,
        "detectability_output": _documented_detectability_output(payload),
    }


def compute_time0_baseline_detectability(**kwargs: Any) -> dict[str, Any]:
    payload = compute_detectability_time0_baseline(**kwargs)
    detectable = str(payload.get("detectable") or "error").strip().lower()
    if str(payload.get("max_SRD_detectability") or "").strip().lower() == "error":
        detectable = "error"
    if detectable not in {"yes", "no", "error"}:
        detectable = "error"
    return {
        "detectable": detectable,
        "detectability_output": _documented_detectability_output(payload),
    }


def compute_detectability_time0_baseline(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    first_detectable_minimum_symmetric_distance: float,
    first_detectable_epsilon: float = 0.001,
    minimum_consecutive_srd_steps: int = 1,
    intervention_time: object = 0.0,
    signal_detectability_specs: Optional[Mapping[str, Mapping[str, object]]] = None,
    require_signal_detectability_specs: bool = False,
    mean_euclidean_distance_clean_dirty: Optional[Sequence[float]] = None,
    mean_euclidean_distance_clean_baseline: Optional[Sequence[float]] = None,
    mean_SNR: Optional[Sequence[Optional[float]]] = None,
    RMS_thresholds: Optional[Mapping[str, float]] = None,
    time_round_decimals: int = 9,
) -> dict[str, Any]:
    error_payload = _detectability_error_payload()
    try:
        max_srd, first_diff, euclidean_distance = _compute_detectability_values(
            baseline_df=baseline_df,
            run_df=run_df,
            first_detectable_minimum_symmetric_distance=(
                first_detectable_minimum_symmetric_distance
            ),
            first_detectable_epsilon=first_detectable_epsilon,
            minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
            intervention_time=intervention_time,
            signal_detectability_specs=signal_detectability_specs,
            require_signal_detectability_specs=require_signal_detectability_specs,
            time_round_decimals=time_round_decimals,
        )
        signal_columns = [str(column) for column in baseline_df.columns[:-1]]
        resolved_mean_snr = (
            list(mean_SNR)
            if mean_SNR is not None
            else _signal_to_noise_ratio_db_values(
                mean_euclidean_distance_clean_dirty=mean_euclidean_distance_clean_dirty,
                mean_euclidean_distance_clean_baseline=mean_euclidean_distance_clean_baseline,
            )
        )
        rms_passes = _rms_threshold_passes(
            signal_columns=signal_columns,
            mean_euclidean_distance_clean_baseline=(
                mean_euclidean_distance_clean_baseline
            ),
            RMS_thresholds=RMS_thresholds,
        )
        detectable = "yes" if rms_passes else "no"
        return _detectability_payload(
            detectable=detectable,
            max_srd=max_srd,
            euclidean_distance=euclidean_distance,
            mean_euclidean_distance_clean_dirty=mean_euclidean_distance_clean_dirty,
            mean_euclidean_distance_clean_baseline=mean_euclidean_distance_clean_baseline,
            mean_SNR=resolved_mean_snr,
            first_diff=first_diff,
        )
    except Exception:
        return error_payload


def compute_detectability(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    first_detectable_minimum_symmetric_distance: float,
    first_detectable_epsilon: float = 0.001,
    minimum_consecutive_srd_steps: int = 1,
    intervention_time: object = 0.0,
    signal_detectability_specs: Optional[Mapping[str, Mapping[str, object]]] = None,
    require_signal_detectability_specs: bool = False,
    time_round_decimals: int = 9,
) -> dict[str, Any]:
    return compute_detectability_baseline(
        baseline_df=baseline_df,
        run_df=run_df,
        first_detectable_minimum_symmetric_distance=(
            first_detectable_minimum_symmetric_distance
        ),
        first_detectable_epsilon=first_detectable_epsilon,
        minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
        intervention_time=intervention_time,
        signal_detectability_specs=signal_detectability_specs,
        require_signal_detectability_specs=require_signal_detectability_specs,
        time_round_decimals=time_round_decimals,
    )


def compute_detectability_euclid(
    *,
    baseline_df: pd.DataFrame,
    run_df: pd.DataFrame,
    k: float,
    align_on_overlap: bool = False,
    time_round_decimals: int = 9,
) -> dict[str, Any]:
    result = compute_euclid_distance_and_norm(
        baseline_df=baseline_df,
        run_df=run_df,
        align_on_overlap=align_on_overlap,
        time_round_decimals=time_round_decimals,
    )
    if result is None:
        return {
            "detectability_passed": False,
            "distance_baseline": None,
            "distance_parent": None,
            "threshold": None,
            "threshold_parent": None,
            "failure_reason": "non_detectable",
        }
    dist_baseline, norm_baseline = result
    threshold = float(k) * float(norm_baseline)
    passed = bool(float(dist_baseline) > float(threshold))
    return {
        "detectability_passed": passed,
        "distance_baseline": float(dist_baseline),
        "distance_parent": None,
        "threshold": float(threshold),
        "threshold_parent": None,
        "failure_reason": None if passed else "non_detectable",
    }


def assert_signal_prefix(
    df: pd.DataFrame,
    expected_prefix: Sequence[str],
    *,
    context: str,
) -> None:
    cols = list(df.columns)
    missing = [signal for signal in expected_prefix if signal not in cols]
    if missing:
        raise AssertionError(f"Missing Simulink signals for {context}: {missing}")
    prefix = cols[: len(expected_prefix)]
    if prefix != list(expected_prefix):
        raise AssertionError(
            "Simulink signal order mismatch for "
            f"{context}: expected_prefix={list(expected_prefix)} got_prefix={prefix}"
        )
