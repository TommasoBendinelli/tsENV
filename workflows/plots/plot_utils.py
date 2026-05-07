from pathlib import Path

import pandas as pd


BENCHMARKS = {
    "ucr": {
        "title": "UCR",
        "context_order": ["none"],
        "group": "accuracy",
    },
    "uea": {
        "title": "UEA",
        "context_order": ["none"],
        "group": "accuracy",
    },
    "equations": {
        "title": "Equations",
        "context_order": ["low", "none"],
        "group": "accuracy",
    },
    "tsb_ad_u": {
        "title": "TSB-AD-U",
        "context_order": ["none"],
        "group": "anomaly",
    },
    "tsb_ad_m": {
        "title": "TSB-AD-M",
        "context_order": ["none"],
        "group": "anomaly",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def latest_metrics_csv(metrics_dir: Path) -> Path:
    candidates = sorted(metrics_dir.glob("metrics__*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics__*.csv files in {metrics_dir}")
    return candidates[-1]


def load_metrics(
    results_dir: Path,
    variant: str,
    skip_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    data = {}
    for bench in BENCHMARKS:
        metrics_dir = results_dir / bench / variant
        try:
            df = pd.read_csv(latest_metrics_csv(metrics_dir))
        except FileNotFoundError:
            if not skip_missing:
                raise
            print(f"Skipping missing metrics: {metrics_dir}")
            df = pd.DataFrame(
                columns=[
                    "context_level",
                    "shot_level",
                    "is_few_shot",
                    "model_name",
                    "metric_label",
                    "metric_value",
                ]
            )
        metric_columns = [
            column
            for column in df.columns
            if str(column).strip().lower().startswith("metric_")
            and not str(column).strip().lower().startswith("metric__")
        ]
        if metric_columns:
            id_vars = [column for column in df.columns if column not in metric_columns]
            df = df.melt(
                id_vars=id_vars,
                value_vars=metric_columns,
                var_name="metric_name",
                value_name="metric_value",
            )
            df["metric_label"] = (
                df["metric_name"]
                .astype(str)
                .str.replace(r"^metric_", "", regex=True)
                .str.replace("-", "_")
            )
        for column in ("context_level", "shot_level", "is_few_shot"):
            if column in df.columns:
                df[column] = df[column].replace({"": pd.NA, "nan": pd.NA})
        if "shot_level" in df.columns:
            df["shot_level"] = df["shot_level"].where(
                df["shot_level"].notna(), pd.NA
            )
        df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
        if "is_few_shot" in df.columns:
            df["is_few_shot"] = df["is_few_shot"].map(
                {True: True, False: False, "True": True, "False": False}
            )
        if "shot_level" not in df.columns and "is_few_shot" in df.columns:
            df["shot_level"] = df["is_few_shot"].map(
                {True: "few_shot", False: "zero_shot"}
            )
        accuracy_mask = df["metric_label"].str.lower() == "accuracy"
        df.loc[accuracy_mask, "metric_value"] = df.loc[
            accuracy_mask, "metric_value"
        ] * 100.0
        data[bench] = df
    return data
