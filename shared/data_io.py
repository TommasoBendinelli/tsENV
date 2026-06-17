"""
Data loading and lightweight serialization helpers.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import pandas as pd

DEFAULT_BASE_DIR = "outputs"
SPLITS = ("train", "val", "test")
DATA_ROOT_CANDIDATES = (
    Path("data"),
    Path("new_data"),
    Path("old_data"),
    Path("data_failure"),
    Path("baseline_training"),
)

PREDICTION_FILE_RE = re.compile(
    r"epoch_(?P<epoch>best|\d+)_(?P<split>train|val|test)", re.IGNORECASE
)
HEALTHY_STATE = "There is no parameter change across the entire simulation."


def _pandas_matrix_to_dict(matrix: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """Convert a square pandas DataFrame into a JSON-ready nested dict."""

    pdf = matrix.copy()
    pdf.index = pdf.index.astype(str)
    pdf.columns = pdf.columns.astype(str)

    serialized: Dict[str, Dict[str, Optional[float]]] = {}
    for idx, row in pdf.iterrows():
        row_dict: Dict[str, Optional[float]] = {}
        for col, value in row.items():
            row_dict[str(col)] = None if pd.isna(value) else float(value)
        serialized[str(idx)] = row_dict
    return serialized


def _dict_to_pandas_matrix(payload: Mapping[str, Mapping[str, Optional[float]]]) -> pd.DataFrame:
    """Rebuild a pandas DataFrame from `_pandas_matrix_to_dict` output."""

    if not payload:
        return pd.DataFrame()

    ordered_rows = list(payload.keys())
    df = pd.DataFrame.from_dict(payload, orient="index")
    df = df.loc[ordered_rows]
    df.index = pd.Index([str(idx) for idx in df.index])
    df.columns = pd.Index([str(col) for col in df.columns])

    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def should_ignore_preserved(path: Path) -> bool:
    """Return True when any component indicates a preserved/archived attempt."""
    parts = [part.lower() for part in path.parts]
    return any("__attempt" in part or part.endswith("_preserved") for part in parts)


def _load_run_dataframe(run_path: str) -> pd.DataFrame:
    """Load a run directory into a single DataFrame if possible."""
    run_root = Path(run_path).expanduser()
    predictions_dir = run_root / "predictions"
    prediction_files = sorted(predictions_dir.glob("*.csv")) if predictions_dir.exists() else []
    frames: list[pd.DataFrame] = []

    if prediction_files:
        for csv_path in prediction_files:
            df_part = pd.read_csv(csv_path)
            if df_part.empty:
                continue
            df_part = df_part.copy()
            match = PREDICTION_FILE_RE.search(csv_path.stem)
            epoch = match.group("epoch") if match else None
            split = match.group("split") if match else None
            if split:
                df_part["split"] = split
            else:
                raise ValueError("Something wrong with the file name!")
            if epoch:
                df_part["epoch"] = epoch
            elif "epoch" not in df_part.columns:
                df_part["epoch"] = pd.NA
            df_part["source_file"] = str(csv_path)
            df_part["run_path"] = str(run_root)
            frames.append(df_part)
    else:
        for split in SPLITS:
            file_path = run_root / "errors" / f"{split}_errors.json"
            if not file_path.exists():
                continue
            try:
                payload = json.loads(file_path.read_text())
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Failed to load {file_path}: {exc}")
                continue
            if not isinstance(payload, list):
                warnings.warn(f"Expected a list in {file_path}; skipping.")
                continue
            rows = []
            for item in payload:
                rows.append(
                    {
                        "split": split,
                        "dataset_index": item.get("dataset_index"),
                        "sample_id": item.get("sample_id"),
                        "scenario": item.get("scenario"),
                        "true_label": item.get("true_label"),
                        "pred_label": item.get("pred_label"),
                        "true_choice": item.get("true_choice"),
                        "pred_choice": item.get("pred_choice"),
                        "source_file": str(file_path),
                        "run_path": str(run_root),
                    }
                )
            if rows:
                frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["dataset_index", "true_label", "pred_label"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    if "epoch" in df.columns:
        df["epoch"] = df["epoch"].astype(str).where(pd.notna(df["epoch"]), pd.NA)
    else:
        df["epoch"] = pd.NA
    return df


def _flatten_metadata(meta: dict, run_dir: Path) -> dict:
    """Flatten a single metadata_task.json payload into a row dict."""

    payload = meta or {}
    modifications = payload.get("modifications")
    if isinstance(modifications, dict):
        modifications = [modifications]
    modifications = modifications or []
    primary = modifications[0] if modifications else {}

    columns = payload.get("columns")
    if isinstance(columns, list):
        columns_str = ", ".join(map(str, columns))
    else:
        columns_str = str(columns)
    row = {
        "run_id": run_dir.name,
        "scenario_id": payload.get("scenario_id") or run_dir.parent.name,
        "run_path": str(run_dir),
        "model": payload.get("model"),
        "recipe_id": payload.get("recipe_id"),
        "n_rows": payload.get("n_rows"),
        "n_signals": payload.get("n_signals"),
        "time_start_s": payload.get("time_start_s"),
        "time_end_s": payload.get("time_end_s"),
        "stop_time_s": payload.get("stop_time_s"),
        "first_diff_time": payload.get("first_diff_time"),
        "valid": payload.get("valid"),
        "question": payload.get("question"),
        "columns": columns_str,
        "modifications": modifications,
        "modification_count": len(modifications),
        "correct_identifier": primary.get("identifier"),
        "correct_key": primary.get("key"),
        "correct_transition_type": primary.get("transition_type"),
        "correct_start_time": primary.get("start_time"),
        "correct_end_time": primary.get("end_time"),
        "correct_old_value": primary.get("old_value"),
        "correct_new_value": primary.get("new_value"),
        "has_healthy_csv": (run_dir / "data_healthy.csv").exists(),
        "has_broken_csv": (run_dir / "data_broken.csv").exists(),
    }
    return row


def _load_folder_metadata(folder_path: str, recursive: bool = True):
    """Scan a parent folder for many runs and aggregate their metadata into a DataFrame."""

    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        return pd.DataFrame(), f"Path not found or not a directory: {folder_path}"

    files = list(p.rglob("metadata_task.json") if recursive else p.glob("*/metadata_task.json"))
    if not files:
        return pd.DataFrame(), "No metadata_task.json files found."

    rows, errs = [], []
    for f in files:
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("metadata is not a JSON object")
        except Exception as e:
            errs.append(f"Failed to parse {f}: {e}")
            meta = {}
        rows.append(_flatten_metadata(meta, f.parent))

    df = pd.DataFrame(rows)

    preferred = [
        "run_id",
        "run_path",
        "model",
        "recipe_id",
        "valid",
        "n_rows",
        "n_signals",
        "time_start_s",
        "time_end_s",
        "stop_time_s",
        "first_diff_time",
        "correct_identifier",
        "correct_key",
        "correct_transition_type",
        "correct_start_time",
        "correct_end_time",
        "correct_new_value",
        "has_healthy_csv",
        "has_broken_csv",
        "columns",
        "question",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    if "run_id" in df.columns:
        df = df.sort_values("run_id")

    warn = "\n".join(errs) if errs else None
    return df, warn


__all__ = [
    "DATA_ROOT_CANDIDATES",
    "DEFAULT_BASE_DIR",
    "HEALTHY_STATE",
    "PREDICTION_FILE_RE",
    "SPLITS",
    "_dict_to_pandas_matrix",
    "_flatten_metadata",
    "_load_folder_metadata",
    "_load_run_dataframe",
    "_pandas_matrix_to_dict",
    "should_ignore_preserved",
]
