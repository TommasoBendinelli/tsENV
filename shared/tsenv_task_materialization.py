from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from shared.model_noise_adder import call_noise_adder, load_noise_adder_from_path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_tsenv_model_root(
    *,
    model: str,
    tsenv_model_root: Optional[Path] = None,
    tsenv_root: Optional[Path] = None,
) -> Path:
    if tsenv_model_root is not None:
        return Path(tsenv_model_root).expanduser().resolve()
    base = (
        Path(tsenv_root).expanduser().resolve()
        if tsenv_root is not None
        else (ROOT_DIR / "tsENV_questions").resolve()
    )
    return (base / str(model)).resolve()


def _resolve_models_root(models_root: Optional[Path] = None) -> Path:
    if models_root is not None:
        return Path(models_root).expanduser().resolve()
    return (ROOT_DIR / "models" / "simulink").resolve()


def _normalize_materialization_noise_level(noise_level: object) -> str:
    normalized = str(noise_level or "").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "noise_none": "none",
        "low": "low",
        "noise_low": "low",
        "high": "high",
        "noise_high": "high",
        "medium": "medium",
        "noise_medium": "medium",
    }
    resolved = aliases.get(normalized, normalized)
    if resolved == "medium":
        raise ValueError("Unsupported noise profile 'medium' for task materialization.")
    if resolved not in {"none", "low", "high"}:
        raise ValueError(f"Unsupported noise profile '{noise_level}'.")
    return resolved


def _source_dataframe_path(
    *,
    model: str | None,
    run_uuid: str,
    tsenv_model_root: Optional[Path] = None,
    tsenv_root: Optional[Path] = None,
    runs_root: Optional[Path] = None,
) -> Path:
    if runs_root is not None:
        return (
            Path(runs_root).expanduser().resolve()
            / str(run_uuid).strip()
            / "data.parquet"
        )
    if model is None and tsenv_model_root is None:
        raise ValueError("materialize(UUID, noise_level, seed) requires tsenv_model_root or model.")
    model_root = _resolve_tsenv_model_root(
        model=str(model or Path(tsenv_model_root or "").name),
        tsenv_model_root=tsenv_model_root,
        tsenv_root=tsenv_root,
    )
    run_id = str(run_uuid).strip()
    documented_path = (model_root / run_id).resolve()
    if documented_path.is_dir() and (documented_path / "data.parquet").exists():
        return (documented_path / "data.parquet").resolve()
    candidates = [
        *([] if documented_path.is_dir() else [documented_path]),
        (model_root / f"{run_id}.parquet").resolve(),
        (model_root / "dataframes" / f"{run_id}.parquet").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return documented_path


def _anonymize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"col{idx + 1}" for idx, _ in enumerate(out.columns)]
    return out


def _looks_like_noise_level(value: object) -> bool:
    try:
        _normalize_materialization_noise_level(value)
    except ValueError:
        return False
    return True


def _parse_materialize_args(
    args: tuple[object, ...],
    model: str | None,
) -> tuple[str | None, str, object, int, object | None]:
    if len(args) == 3:
        run_uuid, noise_level, seed = args
        return model, str(run_uuid), noise_level, int(seed), None
    if len(args) == 4:
        if _looks_like_noise_level(args[1]):
            run_uuid, noise_level, seed, uuid_baseline_path = args
            return model, str(run_uuid), noise_level, int(seed), uuid_baseline_path
        legacy_model, run_uuid, noise_level, seed = args
        return str(legacy_model), str(run_uuid), noise_level, int(seed), None
    if len(args) == 5:
        legacy_model, run_uuid, noise_level, seed, uuid_baseline_path = args
        return str(legacy_model), str(run_uuid), noise_level, int(seed), uuid_baseline_path
    raise TypeError(
        "materialize expects materialize(UUID, noise_level, seed, uuid_baseline_path), "
        "materialize(UUID, noise_level, seed), or the legacy "
        "materialize(model, UUID, noise_level, seed[, uuid_baseline_path]) form."
    )


def _coerce_optional_baseline_path(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None
    text = str(value).strip()
    if not text or text == "-1.0":
        return None
    try:
        float(text)
    except ValueError:
        return text
    return None


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _baseline_uuid_from_model_run_specs(model_root: Path, run_uuid: str) -> str | None:
    payload = _load_json_object(model_root / "model_run_specs.json")
    run_id = str(run_uuid).strip()
    for baseline_uuid, raw_baseline in payload.items():
        if run_id == str(baseline_uuid).strip():
            return None
        if not isinstance(raw_baseline, dict):
            continue
        children = raw_baseline.get("children")
        if not isinstance(children, dict):
            continue
        child = children.get(run_id)
        if isinstance(child, dict):
            baseline = str(child.get("time0_baseline_uuid") or "").strip()
            return baseline or None
    return None


def _baseline_uuid_from_sample_manifest(model_root: Path, run_uuid: str) -> str | None:
    payload = _load_json_object(model_root / "sample_manifest.json")
    run_id = str(run_uuid).strip()
    for raw_entries in payload.values():
        if not isinstance(raw_entries, list):
            continue
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            for samples_key, baselines_key in (
                ("train_samples", "train_samples_baselines"),
                ("test_samples", "test_samples_baselines"),
            ):
                samples = raw_entry.get(samples_key)
                baselines = raw_entry.get(baselines_key)
                if not isinstance(samples, list) or not isinstance(baselines, list):
                    continue
                for idx, sample in enumerate(samples):
                    if str(sample).strip() != run_id or idx >= len(baselines):
                        continue
                    baseline = str(baselines[idx] or "").strip()
                    return baseline or None
    return None


def _model_root_for_metadata(
    *,
    parsed_model: str | None,
    tsenv_model_root: Optional[Path],
    tsenv_root: Optional[Path],
    models_root: Optional[Path],
    runs_root: Optional[Path],
) -> Path | None:
    if tsenv_model_root is not None:
        return Path(tsenv_model_root).expanduser().resolve()
    if runs_root is not None:
        candidate = Path(runs_root).expanduser().resolve()
        if candidate.name.startswith("runs"):
            return candidate.parent
    if parsed_model:
        if tsenv_root is not None:
            return _resolve_tsenv_model_root(
                model=parsed_model,
                tsenv_root=tsenv_root,
            )
        return _resolve_models_root(models_root) / parsed_model
    return None


def _resolve_documented_baseline_path(
    *,
    parsed_model: str | None,
    run_uuid: str,
    uuid_baseline_path: object | None,
    tsenv_model_root: Optional[Path],
    tsenv_root: Optional[Path],
    models_root: Optional[Path],
    runs_root: Optional[Path],
) -> Path | None:
    explicit = _coerce_optional_baseline_path(uuid_baseline_path)
    if explicit:
        return _source_dataframe_path(
            model=parsed_model,
            run_uuid=explicit,
            tsenv_model_root=tsenv_model_root,
            tsenv_root=tsenv_root,
            runs_root=runs_root,
        )

    model_root = _model_root_for_metadata(
        parsed_model=parsed_model,
        tsenv_model_root=tsenv_model_root,
        tsenv_root=tsenv_root,
        models_root=models_root,
        runs_root=runs_root,
    )
    if model_root is None:
        return None

    baseline_uuid = _baseline_uuid_from_model_run_specs(model_root, run_uuid)
    if baseline_uuid is None:
        baseline_uuid = _baseline_uuid_from_sample_manifest(model_root, run_uuid)
    if not baseline_uuid:
        return None
    return _source_dataframe_path(
        model=parsed_model,
        run_uuid=baseline_uuid,
        tsenv_model_root=tsenv_model_root,
        tsenv_root=tsenv_root,
        runs_root=runs_root,
    )


def _existing_optional_baseline_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    warnings.warn(
        f"materialize reference parquet not found at {path}; proceeding without reference.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def materialize(
    *args: object,
    model: str | None = None,
    tsenv_model_root: Optional[Path] = None,
    tsenv_root: Optional[Path] = None,
    models_root: Optional[Path] = None,
    runs_root: Optional[Path] = None,
    noise_adder_path: Optional[Path] = None,
    uuid_baseline_path: object | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parsed_model, run_uuid, noise_level, seed, positional_baseline_path = _parse_materialize_args(args, model)
    baseline_path_arg = uuid_baseline_path if uuid_baseline_path is not None else positional_baseline_path
    source_path = _source_dataframe_path(
        model=parsed_model,
        run_uuid=run_uuid,
        tsenv_model_root=tsenv_model_root,
        tsenv_root=tsenv_root,
        runs_root=runs_root,
    )
    df = pd.read_parquet(source_path)
    resolved_noise_level = _normalize_materialization_noise_level(noise_level)
    noise_analysis: dict[str, Any] = {}
    if resolved_noise_level != "none":
        baseline_path = _resolve_documented_baseline_path(
            parsed_model=parsed_model,
            run_uuid=run_uuid,
            uuid_baseline_path=baseline_path_arg,
            tsenv_model_root=tsenv_model_root,
            tsenv_root=tsenv_root,
            models_root=models_root,
            runs_root=runs_root,
        )
        baseline_path = _existing_optional_baseline_path(baseline_path)
        baseline_df = pd.read_parquet(baseline_path) if baseline_path is not None else None
        resolved_noise_adder_path = (
            Path(noise_adder_path).expanduser().resolve()
            if noise_adder_path is not None
            else _resolve_tsenv_model_root(
                model=model,
                tsenv_model_root=tsenv_model_root,
                tsenv_root=tsenv_root,
            )
            / "noise_adder.py"
        )
        add_noise = load_noise_adder_from_path(resolved_noise_adder_path)
        df, noise_analysis = call_noise_adder(
            add_noise,
            df,
            baseline_df=baseline_df,
            seed=int(seed),
            noise_level=resolved_noise_level,
        )
    return _anonymize_columns(df), noise_analysis


def materialize_dataframe(
    *args: object,
    model: str | None = None,
    tsenv_model_root: Optional[Path] = None,
    tsenv_root: Optional[Path] = None,
    models_root: Optional[Path] = None,
    runs_root: Optional[Path] = None,
    noise_adder_path: Optional[Path] = None,
    uuid_baseline_path: object | None = None,
) -> pd.DataFrame:
    df, _ = materialize(
        *args,
        model=model,
        tsenv_model_root=tsenv_model_root,
        tsenv_root=tsenv_root,
        models_root=models_root,
        runs_root=runs_root,
        noise_adder_path=noise_adder_path,
        uuid_baseline_path=uuid_baseline_path,
    )
    return df


__all__ = ["materialize", "materialize_dataframe"]
