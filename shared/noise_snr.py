from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

_NOISE_RULES_REL_PATH = Path("web_model_explorer/config/noise_rules.json")


@dataclass(frozen=True)
class NoiseRules:
    rolling_window_points: int = 11
    sigma_floor_ratio: float = 1e-4


DEFAULT_NOISE_RULES = NoiseRules()


def _to_finite_float(value: object, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if not np.isfinite(out):
        return float(fallback)
    return float(out)


def _normalize_window_points(value: object, fallback: int) -> int:
    try:
        out = int(value)
    except Exception:
        out = int(fallback)
    if out < 1:
        out = int(fallback)
    if out % 2 == 0:
        out += 1
    return out


def load_noise_rules(repo_root: Optional[Path] = None) -> NoiseRules:
    root = repo_root or Path(__file__).resolve().parents[1]
    path = root / _NOISE_RULES_REL_PATH
    if not path.exists():
        return DEFAULT_NOISE_RULES
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_NOISE_RULES
    if not isinstance(payload, dict):
        return DEFAULT_NOISE_RULES
    return NoiseRules(
        rolling_window_points=_normalize_window_points(
            payload.get("rolling_window_points"),
            DEFAULT_NOISE_RULES.rolling_window_points,
        ),
        sigma_floor_ratio=_to_finite_float(
            payload.get("sigma_floor_ratio"),
            DEFAULT_NOISE_RULES.sigma_floor_ratio,
        ),
    )


def noise_multiplier_from_snr_db(snr_db: float) -> float:
    snr = _to_finite_float(snr_db, 20.0)
    return float(10.0 ** (-snr / 20.0))


def snr_db_from_noise_multiplier(noise_multiplier: float) -> float:
    mult = _to_finite_float(noise_multiplier, 0.0)
    if mult <= 0.0:
        return float("inf")
    return float(-20.0 * math.log10(mult))


def hash_string(value: str) -> int:
    h = 2166136261
    for ch in str(value):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value >= 0x80000000:
        value -= 0x100000000
    return int(value)


def _to_uint32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return _to_int32(_to_int32(a) * _to_int32(b))


def make_rng(seed: int):
    t = _to_uint32(seed)

    def _next() -> float:
        nonlocal t
        t = _to_uint32(t + 0x6D2B79F5)
        result = _imul(
            _to_int32(t ^ (t >> 15)),
            _to_int32(1 | _to_int32(t)),
        )
        result = _to_int32(
            result
            ^ _to_int32(
                result
                + _imul(
                    _to_int32(result ^ (_to_uint32(result) >> 7)),
                    _to_int32(61 | result),
                )
            )
        )
        return float(
            _to_uint32(result ^ (_to_uint32(result) >> 14)) / 4294967296.0
        )

    return _next


def gaussian_sample(rng) -> float:
    u = 0.0
    v = 0.0
    while u == 0.0:
        u = float(rng())
    while v == 0.0:
        v = float(rng())
    return float(math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v))


def rolling_rms(values: np.ndarray, *, window_points: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    w = int(window_points)
    if w < 1:
        w = 1
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        chunk = arr[start:end]
        finite = np.isfinite(chunk)
        if not finite.any():
            out[i] = 0.0
            continue
        sq = chunk[finite] ** 2
        out[i] = float(np.sqrt(float(np.mean(sq))))
    return out


def sample_adaptive_gaussian_noise(
    values: np.ndarray,
    *,
    noise_multiplier: float,
    seed_key: str,
    seed: int = 0,
    rules: Optional[NoiseRules] = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mult = _to_finite_float(noise_multiplier, 0.0)
    if arr.shape[0] == 0 or mult <= 0.0:
        return np.zeros(arr.shape[0], dtype=float)

    cfg = rules or DEFAULT_NOISE_RULES
    local_rms = rolling_rms(arr, window_points=cfg.rolling_window_points)
    finite = np.isfinite(arr)
    if finite.any():
        global_rms = float(np.sqrt(float(np.mean(arr[finite] ** 2))))
    else:
        global_rms = 0.0
    sigma_floor = float(cfg.sigma_floor_ratio) * global_rms
    local_scale = np.maximum(local_rms, sigma_floor)

    rng = make_rng(hash_string(f"{seed_key}:{int(seed)}"))
    noise = np.zeros(arr.shape[0], dtype=float)
    for i in range(arr.shape[0]):
        if not np.isfinite(arr[i]):
            continue
        sigma = mult * float(local_scale[i])
        if sigma <= 0.0 or not np.isfinite(sigma):
            continue
        noise[i] = gaussian_sample(rng) * sigma
    return noise


def apply_noise_to_series(
    values: np.ndarray,
    *,
    run_noise_multiplier: float = 0.0,
    signal_noise_multiplier: float = 0.0,
    run_seed_key: str,
    signal_seed_key: str,
    seed: int = 0,
    rules: Optional[NoiseRules] = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    run_noise = sample_adaptive_gaussian_noise(
        arr,
        noise_multiplier=run_noise_multiplier,
        seed_key=run_seed_key,
        seed=seed,
        rules=rules,
    )
    signal_noise = sample_adaptive_gaussian_noise(
        arr,
        noise_multiplier=signal_noise_multiplier,
        seed_key=signal_seed_key,
        seed=seed,
        rules=rules,
    )
    return arr + run_noise + signal_noise


def sample_adaptive_and_base_gaussian_noise(
    values: np.ndarray,
    *,
    adaptive_noise_multiplier: float = 0.0,
    base_noise_multiplier: float = 0.0,
    abs_noise_sigma: float = 0.0,
    seed_key: str,
    seed: int = 0,
    rules: Optional[NoiseRules] = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    adaptive = _to_finite_float(adaptive_noise_multiplier, 0.0)
    base = _to_finite_float(base_noise_multiplier, 0.0)
    abs_sigma = _to_finite_float(abs_noise_sigma, 0.0)
    if arr.shape[0] == 0 or (adaptive <= 0.0 and base <= 0.0 and abs_sigma <= 0.0):
        return np.zeros(arr.shape[0], dtype=float)

    cfg = rules or DEFAULT_NOISE_RULES
    local_rms = rolling_rms(arr, window_points=cfg.rolling_window_points)
    finite = np.isfinite(arr)
    if finite.any():
        global_rms = float(np.sqrt(float(np.mean(arr[finite] ** 2))))
    else:
        global_rms = 0.0
    sigma_floor = float(cfg.sigma_floor_ratio) * global_rms
    sigma_base = base * global_rms if base > 0.0 else 0.0

    rng = make_rng(hash_string(f"{seed_key}:{int(seed)}"))
    noise = np.zeros(arr.shape[0], dtype=float)
    for i in range(arr.shape[0]):
        if not np.isfinite(arr[i]):
            continue
        sigma_adaptive = adaptive * float(max(local_rms[i], sigma_floor))
        sigma = float(math.hypot(sigma_adaptive, sigma_base, abs_sigma))
        if sigma <= 0.0 or not np.isfinite(sigma):
            continue
        noise[i] = gaussian_sample(rng) * sigma
    return noise


def apply_adaptive_and_base_noise(
    values: np.ndarray,
    *,
    adaptive_noise_multiplier: float = 0.0,
    base_noise_multiplier: float = 0.0,
    abs_noise_sigma: float = 0.0,
    seed_key: str,
    seed: int = 0,
    rules: Optional[NoiseRules] = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    noise = sample_adaptive_and_base_gaussian_noise(
        arr,
        adaptive_noise_multiplier=adaptive_noise_multiplier,
        base_noise_multiplier=base_noise_multiplier,
        abs_noise_sigma=abs_noise_sigma,
        seed_key=seed_key,
        seed=seed,
        rules=rules,
    )
    return arr + noise


def sample_global_rms_gaussian_noise(
    values: np.ndarray,
    *,
    noise_multiplier: float = 0.0,
    rms_scale: float = 0.03,
    seed_key: str,
    seed: int = 0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mult = _to_finite_float(noise_multiplier, 0.0)
    scale = _to_finite_float(rms_scale, 0.03)
    if arr.shape[0] == 0 or mult <= 0.0 or scale <= 0.0:
        return np.zeros(arr.shape[0], dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape[0], dtype=float)
    global_rms = float(np.sqrt(float(np.mean(arr[finite] ** 2))))
    sigma = float(mult) * float(scale) * float(global_rms)
    if sigma <= 0.0 or not np.isfinite(sigma):
        return np.zeros(arr.shape[0], dtype=float)

    rng = make_rng(hash_string(f"{seed_key}:{int(seed)}"))
    noise = np.zeros(arr.shape[0], dtype=float)
    for i in range(arr.shape[0]):
        if not np.isfinite(arr[i]):
            continue
        noise[i] = gaussian_sample(rng) * sigma
    return noise


def apply_global_rms_noise(
    values: np.ndarray,
    *,
    noise_multiplier: float = 0.0,
    rms_scale: float = 0.03,
    seed_key: str,
    seed: int = 0,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    noise = sample_global_rms_gaussian_noise(
        arr,
        noise_multiplier=noise_multiplier,
        rms_scale=rms_scale,
        seed_key=seed_key,
        seed=seed,
    )
    return arr + noise


def apply_noise_to_dataframe(
    df: pd.DataFrame,
    *,
    run_noise_multiplier: float = 0.0,
    signal_noise_mapping: Optional[Mapping[str, float]] = None,
    seed: int = 0,
    rules: Optional[NoiseRules] = None,
) -> pd.DataFrame:
    out = df.copy()
    mapping = signal_noise_mapping or {}
    for col in out.columns:
        if col == "time":
            continue
        series = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        if series.size == 0:
            continue
        sig_mult = _to_finite_float(mapping.get(col, 0.0), 0.0)
        noisy = apply_noise_to_series(
            series,
            run_noise_multiplier=_to_finite_float(run_noise_multiplier, 0.0),
            signal_noise_multiplier=sig_mult,
            run_seed_key=f"run:{col}",
            signal_seed_key=f"signal:{col}",
            seed=seed,
            rules=rules,
        )
        out[col] = noisy
    return out


def apply_signal_noise_only(
    df: pd.DataFrame,
    *,
    noise_multiplier: float,
    seed: int = 0,
    rules: Optional[NoiseRules] = None,
    include_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    cols = set(include_columns or [])
    mapping: dict[str, float] = {}
    for col in df.columns:
        if col == "time":
            continue
        if cols and col not in cols:
            continue
        mapping[col] = float(noise_multiplier)
    return apply_noise_to_dataframe(
        df,
        run_noise_multiplier=0.0,
        signal_noise_mapping=mapping,
        seed=seed,
        rules=rules,
    )
