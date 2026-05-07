import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from shared.benchmark_utils import ALLOWED_TSENV_MODELS
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def load_run_dataframe(
    run_dir: Path,
    upcast_to_float32: bool = True,
) -> pd.DataFrame:
    fp = run_dir / "data.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing data.parquet at {fp}")
        
    df = pd.read_parquet(fp)

    if df.empty:
        raise ValueError(f"Empty dataframe in {fp}")
        
    if "time" not in df.columns:
        if df.index.name and df.index.name.lower().startswith("time"):
            df = df.reset_index().rename(columns={df.index.name: "time"})
        elif df.index.dtype.kind in {"f", "i"}:
            df = df.reset_index().rename(columns={"index": "time"})
        else:
            raise ValueError(f"'time' column missing in {fp}")
            
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    df = df.reset_index(drop=True)
    
    if upcast_to_float32:
        float_cols = df.select_dtypes(include=[np.floating]).columns
        if not float_cols.empty:
            df[float_cols] = df[float_cols].astype(np.float32)
            
    return df

def calculate_signal_ratio(baseline_abs: np.ndarray, delta_abs: pd.Series) -> float:
    """
    Calculate the ratio of max(delta) to mean(baseline) robustly.
    Returns 0.0 if baseline is corrupted (infinite) or non-positive.
    Returns inf if delta is infinite.
    """
    baseline_mean = np.nanmean(baseline_abs, dtype=np.float32)
    
    if not np.isfinite(baseline_mean):
        return 0.0
    
    if baseline_mean <= 0:
        if delta_abs.gt(0).any():
            return float('inf')
        return 0.0
        
    delta_max = delta_abs.max()
    if not np.isfinite(delta_max):
        return float('inf')
        
    return delta_max / baseline_mean

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
