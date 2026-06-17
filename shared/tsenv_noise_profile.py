from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from shared.noise_snr import NoiseRules, apply_adaptive_and_base_noise


@dataclass(frozen=True)
class TsenvNoiseProfile:
    noise_local: float
    noise_global: float
    noise_abs: float
    noise_seed: int
    profile_name: str = ""

    @property
    def enabled(self) -> bool:
        return (
            float(self.noise_local) > 0.0
            or float(self.noise_global) > 0.0
            or float(self.noise_abs) > 0.0
        )


def seed_for_run(base_seed: int, run_id: str) -> int:
    h = 2166136261
    for ch in str(run_id):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int((int(base_seed) ^ h) & 0xFFFFFFFF)


def _finite_non_negative(value: object, *, field_name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc
    if not np.isfinite(out):
        raise ValueError(f"{field_name} must be finite.")
    if out < 0.0:
        raise ValueError(f"{field_name} must be >= 0.")
    return float(out)


def resolve_noise_profile_from_resolved(
    resolved: object,
    *,
    default_seed: int = 0,
) -> TsenvNoiseProfile:
    if not isinstance(resolved, dict):
        return TsenvNoiseProfile(0.0, 0.0, 0.0, int(default_seed), "")
    profile_name = str(resolved.get("noise_profile") or "").strip()
    try:
        seed = int(resolved.get("seed", resolved.get("noise_seed", default_seed)))
    except (TypeError, ValueError):
        seed = int(default_seed)

    has_explicit = any(
        key in resolved for key in ("noise_local", "noise_global", "noise_abs")
    )
    if has_explicit:
        return TsenvNoiseProfile(
            noise_local=_finite_non_negative(
                resolved.get("noise_local", 0.0), field_name="recipe.resolved.noise_local"
            ),
            noise_global=_finite_non_negative(
                resolved.get("noise_global", 0.0), field_name="recipe.resolved.noise_global"
            ),
            noise_abs=_finite_non_negative(
                resolved.get("noise_abs", 0.0), field_name="recipe.resolved.noise_abs"
            ),
            noise_seed=int(seed),
            profile_name=profile_name,
        )

    return TsenvNoiseProfile(0.0, 0.0, 0.0, int(seed), profile_name)


def apply_noise_profile_to_dataframe(
    df: pd.DataFrame,
    *,
    run_id: str,
    profile: TsenvNoiseProfile,
    signal_columns: Iterable[str],
    rules: NoiseRules,
) -> pd.DataFrame:
    if not profile.enabled:
        return df
    out = df.copy()
    derived_seed = seed_for_run(int(profile.noise_seed), str(run_id))
    for col in signal_columns:
        name = str(col)
        if name not in out.columns:
            continue
        values = pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float)
        out[name] = apply_adaptive_and_base_noise(
            values,
            adaptive_noise_multiplier=float(profile.noise_local),
            base_noise_multiplier=float(profile.noise_global),
            abs_noise_sigma=float(profile.noise_abs),
            seed_key=f"signal:{run_id}:{name}",
            seed=derived_seed,
            rules=rules,
        )
    return out
