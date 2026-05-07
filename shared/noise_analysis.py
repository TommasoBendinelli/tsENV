from __future__ import annotations

from collections.abc import Mapping as MappingABC
from typing import Any, Mapping, Optional

import numpy as np


DEFAULT_LOCAL_NOISE_ANALYSIS_RADIUS_ROWS = 5
DOCUMENTED_LOCAL_NOISE_ANALYSIS_PRE_ROWS = 10
DOCUMENTED_LOCAL_NOISE_ANALYSIS_POST_ROWS = 30
SNR_NEG_INF = "-inf"
SNRValue = float | str | None


def _signal_columns(df: Any, signal_type: Optional[Mapping[str, Any]] = None) -> list[str]:
    if not hasattr(df, "columns"):
        return []
    columns = [str(column).strip() for column in list(df.columns) if str(column).strip()]
    if signal_type:
        signal_count = len(signal_type)
        if signal_count > 0 and len(columns) >= signal_count:
            return columns[:signal_count]
    out: list[str] = []
    for name in columns:
        if name.lower() in {"time", "timestamp", "index", "idx"}:
            continue
        out.append(name)
    return out


def snr_by_signal(
    *,
    clean_df: Any,
    noisy_df: Any,
    reference_df: Any = None,
    signal_type: Optional[Mapping[str, Any]] = None,
) -> dict[str, SNRValue]:
    ratios = _effect_to_noise_ratio_by_signal(
        clean_df=clean_df,
        noisy_df=noisy_df,
        reference_df=reference_df,
        signal_type=signal_type,
    )
    return {signal: _ratio_to_snr_db(ratio) for signal, ratio in ratios.items()}


def _effect_to_noise_ratio_by_signal(
    *,
    clean_df: Any,
    noisy_df: Any,
    reference_df: Any = None,
    signal_type: Optional[Mapping[str, Any]] = None,
) -> dict[str, Optional[float]]:
    if not hasattr(clean_df, "__getitem__") or not hasattr(noisy_df, "__getitem__"):
        raise TypeError("snr estimation requires dataframe-like inputs")
    values: dict[str, Optional[float]] = {}
    for column in _signal_columns(clean_df, signal_type=signal_type):
        if column not in getattr(noisy_df, "columns", []):
            values[column] = None
            continue
        if reference_df is not None and not hasattr(reference_df, "__getitem__"):
            raise TypeError("reference_df must be dataframe-like when provided")
        if reference_df is not None and column not in getattr(reference_df, "columns", []):
            values[column] = None
            continue
        clean = np.asarray(clean_df[column], dtype=np.float64)
        noisy = np.asarray(noisy_df[column], dtype=np.float64)
        reference = (
            None
            if reference_df is None
            else np.asarray(reference_df[column], dtype=np.float64)
        )
        width = min(clean.size, noisy.size, clean.size if reference is None else reference.size)
        if width <= 0:
            values[column] = None
            continue
        clean = clean[:width]
        noisy = noisy[:width]
        finite = np.isfinite(clean) & np.isfinite(noisy)
        if reference is not None:
            reference = reference[:width]
            finite = finite & np.isfinite(reference)
        if not finite.any():
            values[column] = None
            continue
        clean = clean[finite]
        noisy = noisy[finite]
        if reference is None:
            effect = clean
        else:
            effect = clean - reference[finite]
        noise = noisy - clean
        effect_rms = float(np.sqrt(np.mean(np.square(effect)))) if effect.size else 0.0
        noise_rms = float(np.sqrt(np.mean(np.square(noise)))) if noise.size else 0.0
        if not np.isfinite(effect_rms) or not np.isfinite(noise_rms):
            values[column] = None
        elif noise_rms <= 0.0:
            values[column] = None
        elif effect_rms <= 0.0:
            values[column] = 0.0
        else:
            values[column] = effect_rms / noise_rms
    return values


def _ratio_to_snr_db(ratio: Optional[float]) -> SNRValue:
    if ratio is None:
        return None
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio < 0.0:
        return None
    if ratio == 0.0:
        return SNR_NEG_INF
    estimate = float(20.0 * np.log10(ratio))
    return estimate if np.isfinite(estimate) else None


def average_noise_signal_values(
    values_by_seed: list[dict[str, Optional[float]]],
) -> dict[str, Optional[float]]:
    if not values_by_seed:
        return {}
    signal_names = list(values_by_seed[0].keys())
    expected = set(signal_names)
    if any(set(values.keys()) != expected for values in values_by_seed):
        raise ValueError("noise_analysis seed payloads must have consistent signals")
    averaged: dict[str, Optional[float]] = {}
    for signal in signal_names:
        numeric_values: list[float] = []
        for values in values_by_seed:
            value = values.get(signal)
            if value is None:
                continue
            parsed = float(value)
            if np.isfinite(parsed):
                numeric_values.append(parsed)
        averaged[signal] = (
            None
            if not numeric_values
            else float(np.mean(np.asarray(numeric_values, dtype=np.float64)))
        )
    return averaged


def _signal_values_as_list(values: Mapping[str, SNRValue]) -> list[SNRValue]:
    out: list[SNRValue] = []
    for value in values.values():
        if value is None:
            out.append(None)
        elif value == SNR_NEG_INF:
            out.append(SNR_NEG_INF)
        else:
            out.append(float(value))
    return out


def _time_column_name(df: Any, signal_type: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    if not hasattr(df, "columns"):
        return None
    columns = [str(column).strip() for column in list(df.columns) if str(column).strip()]
    for name in columns:
        if name.lower() in {"time", "timestamp"}:
            return name
    if signal_type:
        signal_count = len(signal_type)
        if len(columns) == signal_count + 1:
            return columns[-1]
    return None


def _local_noise_window_bounds(
    *,
    clean_df: Any,
    noisy_df: Any,
    local_time: Optional[float],
    radius_rows: int,
    signal_type: Optional[Mapping[str, Any]] = None,
    pre_rows: Optional[int] = None,
    post_rows: Optional[int] = None,
) -> Optional[tuple[Any, Any]]:
    if local_time is None:
        return None
    local_time = float(local_time)
    if not np.isfinite(local_time) or local_time < 0.0:
        return None
    if not hasattr(clean_df, "iloc") or not hasattr(noisy_df, "iloc"):
        return None
    time_column = _time_column_name(clean_df, signal_type=signal_type)
    if time_column is None:
        return None
    clean_time = np.asarray(clean_df[time_column], dtype=np.float64)
    width = min(clean_time.size, len(clean_df), len(noisy_df))
    if width <= 0:
        return None
    clean_time = clean_time[:width]
    finite_indices = np.flatnonzero(np.isfinite(clean_time))
    if finite_indices.size == 0:
        return None
    nearest_position = int(
        finite_indices[
            int(np.argmin(np.abs(clean_time[finite_indices] - local_time)))
        ]
    )
    pre = max(0, int(radius_rows if pre_rows is None else pre_rows))
    post = max(0, int(radius_rows if post_rows is None else post_rows))
    start = max(0, nearest_position - pre)
    stop = min(width, nearest_position + post + 1)
    if stop <= start:
        return None
    return start, stop


def local_noise_window(
    *,
    clean_df: Any,
    noisy_df: Any,
    local_time: Optional[float],
    radius_rows: int,
    signal_type: Optional[Mapping[str, Any]] = None,
    pre_rows: Optional[int] = None,
    post_rows: Optional[int] = None,
) -> Optional[tuple[Any, Any]]:
    bounds = _local_noise_window_bounds(
        clean_df=clean_df,
        noisy_df=noisy_df,
        local_time=local_time,
        radius_rows=radius_rows,
        signal_type=signal_type,
        pre_rows=pre_rows,
        post_rows=post_rows,
    )
    if bounds is None:
        return None
    start, stop = bounds
    return clean_df.iloc[start:stop], noisy_df.iloc[start:stop]


def _local_noise_window_with_reference(
    *,
    clean_df: Any,
    noisy_df: Any,
    reference_df: Any = None,
    local_time: Optional[float],
    radius_rows: int,
    signal_type: Optional[Mapping[str, Any]] = None,
    pre_rows: Optional[int] = None,
    post_rows: Optional[int] = None,
) -> Optional[tuple[Any, Any, Any]]:
    bounds = _local_noise_window_bounds(
        clean_df=clean_df,
        noisy_df=noisy_df,
        local_time=local_time,
        radius_rows=radius_rows,
        signal_type=signal_type,
        pre_rows=pre_rows,
        post_rows=post_rows,
    )
    if bounds is None:
        return None
    start, stop = bounds
    local_reference_df = None
    if reference_df is not None:
        if not hasattr(reference_df, "iloc"):
            return None
        local_reference_df = reference_df.iloc[start:stop]
    return clean_df.iloc[start:stop], noisy_df.iloc[start:stop], local_reference_df


def _is_frame_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_reference_argument(value: Any) -> bool:
    if value is None or isinstance(value, MappingABC):
        return False
    return _is_frame_sequence(value) or hasattr(value, "columns")


def _validate_sequence_inputs(
    clean_df: Any,
    noisy_df: Any,
    reference_df: Any,
) -> Optional[tuple[list[Any], list[Any], list[Any]]]:
    any_sequence = any(
        _is_frame_sequence(value) for value in (clean_df, noisy_df, reference_df)
    )
    if not any_sequence:
        return None
    if not _is_frame_sequence(clean_df) or not _is_frame_sequence(noisy_df):
        raise ValueError("clean_df and noisy_df must both be lists when either is a list")
    clean_items = list(clean_df)
    noisy_items = list(noisy_df)
    if reference_df is None:
        reference_items = [None] * len(clean_items)
    elif _is_frame_sequence(reference_df):
        reference_items = list(reference_df)
    else:
        raise ValueError("reference_df must be a list when clean_df and noisy_df are lists")
    if len(clean_items) != len(noisy_items) or len(clean_items) != len(reference_items):
        raise ValueError("clean_df, noisy_df, and reference_df lists must have equal length")
    return clean_items, noisy_items, reference_items


def _first_diff_at(first_diff: Any, index: int) -> Optional[float]:
    if _is_frame_sequence(first_diff):
        if index >= len(first_diff):
            raise ValueError("first_diff list must match dataframe list length")
        return first_diff[index]
    return first_diff


def _none_ratio_map(
    clean_df: Any,
    signal_type: Optional[Mapping[str, Any]] = None,
) -> dict[str, Optional[float]]:
    return {signal: None for signal in _signal_columns(clean_df, signal_type=signal_type)}


def _mean_ratio_maps(
    ratio_maps: list[dict[str, Optional[float]]],
) -> dict[str, Optional[float]]:
    if not ratio_maps:
        return {}
    signal_names = list(ratio_maps[0].keys())
    expected = set(signal_names)
    if any(set(values.keys()) != expected for values in ratio_maps):
        raise ValueError("noise_analysis list inputs must expose consistent signals")
    averaged: dict[str, Optional[float]] = {}
    for signal in signal_names:
        ratios: list[float] = []
        for values in ratio_maps:
            ratio = values.get(signal)
            if ratio is None:
                continue
            parsed = float(ratio)
            if np.isfinite(parsed) and parsed >= 0.0:
                ratios.append(parsed)
        averaged[signal] = (
            None
            if len(ratios) != len(ratio_maps)
            else float(np.mean(np.asarray(ratios, dtype=np.float64)))
        )
    return averaged


def _ratio_values_as_list(values: Mapping[str, Optional[float]]) -> list[SNRValue]:
    return [_ratio_to_snr_db(value) for value in values.values()]


def quantify_analysis(
    clean_df: Any,
    noisy_df: Any,
    signal_type: Optional[Mapping[str, Any]] = None,
    first_diff: Optional[float] = None,
    *,
    reference_df: Any = None,
    local_radius_rows: int = DEFAULT_LOCAL_NOISE_ANALYSIS_RADIUS_ROWS,
    local_pre_rows: Optional[int] = None,
    local_post_rows: Optional[int] = None,
) -> dict[str, list[SNRValue]]:
    if reference_df is None and _is_reference_argument(signal_type):
        reference_df = signal_type
        signal_type = None

    sequence_inputs = _validate_sequence_inputs(clean_df, noisy_df, reference_df)
    if sequence_inputs is not None:
        clean_items, noisy_items, reference_items = sequence_inputs
        if _is_frame_sequence(first_diff) and len(first_diff) != len(clean_items):
            raise ValueError("first_diff list must match dataframe list length")
        global_ratio_maps: list[dict[str, Optional[float]]] = []
        local_ratio_maps: list[dict[str, Optional[float]]] = []
        for index, (clean_item, noisy_item, reference_item) in enumerate(
            zip(clean_items, noisy_items, reference_items)
        ):
            global_ratio_maps.append(
                _effect_to_noise_ratio_by_signal(
                    clean_df=clean_item,
                    noisy_df=noisy_item,
                    reference_df=reference_item,
                    signal_type=signal_type,
                )
            )
            local_window = _local_noise_window_with_reference(
                clean_df=clean_item,
                noisy_df=noisy_item,
                reference_df=reference_item,
                local_time=_first_diff_at(first_diff, index),
                radius_rows=local_radius_rows,
                signal_type=signal_type,
                pre_rows=local_pre_rows,
                post_rows=local_post_rows,
            )
            if local_window is None:
                local_ratio_maps.append(_none_ratio_map(clean_item, signal_type=signal_type))
            else:
                local_clean_df, local_noisy_df, local_reference_df = local_window
                local_ratio_maps.append(
                    _effect_to_noise_ratio_by_signal(
                        clean_df=local_clean_df,
                        noisy_df=local_noisy_df,
                        reference_df=local_reference_df,
                        signal_type=signal_type,
                    )
                )
        return {
            "global": _ratio_values_as_list(_mean_ratio_maps(global_ratio_maps)),
            "local": _ratio_values_as_list(_mean_ratio_maps(local_ratio_maps)),
        }

    out: dict[str, list[SNRValue]] = {
        "global": _signal_values_as_list(
            snr_by_signal(
                clean_df=clean_df,
                noisy_df=noisy_df,
                reference_df=reference_df,
                signal_type=signal_type,
            )
        )
    }
    local_window = _local_noise_window_with_reference(
        clean_df=clean_df,
        noisy_df=noisy_df,
        reference_df=reference_df,
        local_time=first_diff,
        radius_rows=local_radius_rows,
        signal_type=signal_type,
        pre_rows=local_pre_rows,
        post_rows=local_post_rows,
    )
    out["local"] = [None] * len(out.get("global", []))
    if local_window is not None:
        local_clean_df, local_noisy_df, local_reference_df = local_window
        out["local"] = _signal_values_as_list(
            snr_by_signal(
                clean_df=local_clean_df,
                noisy_df=local_noisy_df,
                reference_df=local_reference_df,
                signal_type=signal_type,
            )
        )
    return out


def first_detectable_time_from_baseline(
    clean_df: Any,
    baseline_df: Any,
    *,
    epsilon: float = 0.001,
) -> Optional[float]:
    if baseline_df is None or baseline_df is clean_df:
        return None
    if not hasattr(clean_df, "columns") or not hasattr(baseline_df, "columns"):
        return None
    time_column = _time_column_name(clean_df)
    if time_column is None or time_column not in getattr(baseline_df, "columns", []):
        return None
    signal_columns = [
        column
        for column in _signal_columns(clean_df)
        if column in getattr(baseline_df, "columns", [])
    ]
    if not signal_columns:
        return None
    clean_time = np.asarray(clean_df[time_column], dtype=np.float64)
    baseline_time = np.asarray(baseline_df[time_column], dtype=np.float64)
    width = min(len(clean_df), len(baseline_df), clean_time.size, baseline_time.size)
    if width <= 0:
        return None
    if not np.allclose(clean_time[:width], baseline_time[:width], equal_nan=False):
        return None
    eps = float(epsilon)
    best: Optional[float] = None
    for column in signal_columns:
        clean_values = np.asarray(clean_df[column], dtype=np.float64)[:width]
        baseline_values = np.asarray(baseline_df[column], dtype=np.float64)[:width]
        finite = np.isfinite(clean_values) & np.isfinite(baseline_values)
        if not finite.any():
            continue
        distance = np.abs(clean_values - baseline_values) / (
            np.abs(clean_values) + np.abs(baseline_values) + eps
        )
        candidate_indices = np.flatnonzero(finite & (distance > 0.0))
        if candidate_indices.size == 0:
            continue
        candidate = float(clean_time[int(candidate_indices[0])])
        if np.isfinite(candidate) and (best is None or candidate < best):
            best = candidate
    return best


__all__ = [
    "DEFAULT_LOCAL_NOISE_ANALYSIS_RADIUS_ROWS",
    "DOCUMENTED_LOCAL_NOISE_ANALYSIS_POST_ROWS",
    "DOCUMENTED_LOCAL_NOISE_ANALYSIS_PRE_ROWS",
    "average_noise_signal_values",
    "first_detectable_time_from_baseline",
    "local_noise_window",
    "quantify_analysis",
    "snr_by_signal",
]
