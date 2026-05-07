from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_child_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported path '{path}'. Only .parquet inputs are allowed.")
    return pd.read_parquet(path)


def validate_frame(
    df: pd.DataFrame,
    *,
    required_signals: tuple[str, ...],
    name: str,
) -> pd.DataFrame:
    out = df.copy()
    if "time" not in out.columns:
        raise ValueError(f"{name} is missing required 'time' column")
    for signal in required_signals:
        if signal not in out.columns:
            raise ValueError(f"{name} is missing required signal '{signal}'")
    out = out[["time", *required_signals]]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().sort_values("time").drop_duplicates("time")
    if out.shape[0] < 12:
        raise ValueError(f"{name} has too few valid samples")
    return out.reset_index(drop=True)


def median_dt(time_values: np.ndarray) -> float:
    dt = np.diff(time_values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 0.0
    return float(np.median(dt))


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float, copy=True)
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def seconds_to_odd_window(
    *,
    dt: float,
    duration_seconds: float,
    min_window: int = 3,
    max_window: int = 101,
    fallback_window: int = 9,
) -> int:
    window = (
        int(round(float(duration_seconds) / max(float(dt), 1e-9)))
        if float(dt) > 0.0
        else int(fallback_window)
    )
    window = int(min(int(max_window), max(int(min_window), int(window))))
    if window % 2 == 0:
        window += 1
    return int(window)


def rel_change(prev: float, curr: float, *, floor: float = 1e-9) -> float:
    if not (np.isfinite(prev) and np.isfinite(curr)):
        return 0.0
    den = max(abs(float(prev)), abs(float(curr)), float(floor))
    return abs(float(curr) - float(prev)) / den
