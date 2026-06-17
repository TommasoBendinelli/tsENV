from __future__ import annotations

import functools
import inspect
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Literal, Mapping, cast

import pandas as pd

NoiseProfile = Literal["none", "low", "high"]
VALID_NOISE_PROFILES = ("none", "low", "high")


def normalize_noise_profile(value: object, *, default: NoiseProfile = "none") -> NoiseProfile:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return default
    if normalized not in VALID_NOISE_PROFILES:
        raise ValueError(
            f"Invalid noise profile '{value}'. Expected one of {', '.join(VALID_NOISE_PROFILES)}."
        )
    return cast(NoiseProfile, normalized)


def _validate_noise_thresholds(raw_value: Any, *, source: Path) -> None:
    if not isinstance(raw_value, Mapping):
        raise TypeError(f"noise_adder.py at {source} must export SNR_THR_DICT as an object.")
    for profile in ("low", "high"):
        profile_payload = raw_value.get(profile)
        if not isinstance(profile_payload, Mapping):
            raise TypeError(
                f"noise_adder.py at {source} SNR_THR_DICT must define {profile!r}."
            )
        for scope in ("global", "local"):
            values = profile_payload.get(scope)
            if not isinstance(values, list):
                raise TypeError(
                    f"noise_adder.py at {source} SNR_THR_DICT[{profile!r}]"
                    f"[{scope!r}] must be a list."
                )


def _validate_noise_dict(raw_value: Any, *, source: Path) -> None:
    if not isinstance(raw_value, Mapping):
        raise TypeError(f"noise_adder.py at {source} must export NOISE_DICT as an object.")
    missing = [profile for profile in ("low", "high") if profile not in raw_value]
    if missing:
        raise TypeError(
            f"noise_adder.py at {source} NOISE_DICT is missing profiles: {missing}."
        )


def _validate_noise_adder_module(module: ModuleType, *, source: Path) -> Callable[..., Any]:
    _validate_noise_dict(getattr(module, "NOISE_DICT", None), source=source)
    _validate_noise_thresholds(getattr(module, "SNR_THR_DICT", None), source=source)
    quantify_noise = getattr(module, "quantify_noise", None)
    if not callable(quantify_noise):
        raise TypeError(
            f"noise_adder.py at {source} does not export quantify_noise(clean, noisy, baseline)."
        )
    if list(inspect.signature(quantify_noise).parameters) != ["clean", "noisy", "reference"]:
        raise TypeError(
            f"noise_adder.py at {source} quantify_noise must have "
            "(clean, noisy, reference)."
        )
    add_noise = getattr(module, "add_noise", None)
    if not callable(add_noise):
        raise TypeError(
            f"noise_adder.py at {source} does not export add_noise(src, seed, noise_level, ref)."
        )
    if list(inspect.signature(add_noise).parameters) != ["src", "seed", "noise_level", "ref"]:
        raise TypeError(
            f"noise_adder.py at {source} add_noise must have "
            "(src, seed, noise_level, ref)."
        )
    return add_noise


@functools.lru_cache(maxsize=None)
def _load_noise_adder_from_path(path_str: str) -> Callable[..., Any]:
    resolved = Path(path_str).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"noise_adder.py not found at {resolved}")
    spec = importlib.util.spec_from_file_location(
        f"model_noise_adder_{abs(hash(str(resolved)))}",
        resolved,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load noise adder from {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _validate_noise_adder_module(module, source=resolved)


def load_model_noise_adder(*, model_id: str, models_root: Path) -> Callable[..., Any]:
    return _load_noise_adder_from_path(str(Path(models_root) / str(model_id) / "noise_adder.py"))


def load_noise_adder_from_path(path: Path | str) -> Callable[..., Any]:
    return _load_noise_adder_from_path(str(path))


def call_noise_adder(
    add_noise: Callable[..., Any],
    df: pd.DataFrame,
    *,
    baseline_df: pd.DataFrame | None = None,
    first_diff: float | None = None,
    seed: int,
    noise_level: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = add_noise(
        df.copy(),
        seed=int(seed),
        noise_level=str(noise_level),
        ref=baseline_df.copy() if hasattr(baseline_df, "copy") else baseline_df,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "noise_adder.add_noise must return (pandas.DataFrame, noise_analysis)."
        )
    noisy_df, noise_analysis = result
    if not isinstance(noisy_df, pd.DataFrame):
        raise TypeError("noise_adder.add_noise first return value must be a pandas DataFrame.")
    if not isinstance(noise_analysis, dict):
        raise TypeError("noise_adder.add_noise second return value must be a dict.")
    return noisy_df, noise_analysis


def apply_model_noise_profile(
    df: pd.DataFrame,
    *,
    model_id: str,
    models_root: Path,
    noise_profile: object,
    noise_seed: int,
    baseline_df: pd.DataFrame | None = None,
    first_diff: float | None = None,
) -> pd.DataFrame:
    profile = normalize_noise_profile(noise_profile)
    if profile == "none":
        return df
    add_noise = load_model_noise_adder(model_id=model_id, models_root=Path(models_root))
    out, _ = call_noise_adder(
        add_noise,
        df,
        baseline_df=baseline_df,
        first_diff=first_diff,
        seed=int(noise_seed),
        noise_level=profile,
    )
    return out


__all__ = [
    "NoiseProfile",
    "VALID_NOISE_PROFILES",
    "apply_model_noise_profile",
    "call_noise_adder",
    "load_noise_adder_from_path",
    "load_model_noise_adder",
    "normalize_noise_profile",
]
