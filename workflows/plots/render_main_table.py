#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import click
import pandas as pd

VARIANT = "separable"
MODEL_ORDER = [
    "claude-sonnet-4-5-20250929",
    "gemini-3-pro",
    "gpt-5-codex",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "moonshotai/kimi-k2-thinking",
]

DATASET_CONFIGS = {
    "simbench_cls": {
        "title": "SimBench CLS",
        "context_order": ["none", "low", "high", "ground_truth"],
        "shot_order": ["few_shot", "one_shot", "zero_shot"],
    },
    "ucr": {
        "title": "UCR",
        "context_order": ["none"],
        "shot_order": ["few_shot", "zero_shot"],
    },
}

SHOT_LABELS = {
    "few_shot": "Few-shot",
    "one_shot": "One-shot",
    "zero_shot": "Zero-shot",
}

CONTEXT_LABELS = {
    "ground_truth": "Ground Truth",
    "high": "High",
    "low": "Low",
    "none": "None",
}

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "overleaf_paper" / "tables"


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _match_token(value: str, tokens: list[str]) -> str | None:
    for token in tokens:
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", value):
            return token
    return None


def _infer_context_and_shot(
    row: pd.Series, context_order: list[str], shot_order: list[str]
) -> tuple[str | None, str | None]:
    candidates = " ".join(
        str(row.get(col) or "")
        for col in ("variant", "dataset_label", "benchmark_id", "run_id")
    )
    normalized = _normalize_label(candidates)
    context = _match_token(normalized, context_order)
    shot = _match_token(normalized, shot_order)
    if not context:
        fallback = str(row.get("context_level") or "").strip().lower()
        context = fallback or None
    if not shot:
        fallback = str(row.get("shot_level") or "").strip().lower()
        shot = fallback or None
    return context, shot


def _format_model_name(model_id: str) -> str:
    if "/" in model_id:
        left, right = model_id.split("/", 1)
        return f"\\makecell[l]{{{left}\\\\{right}}}"
    return model_id


def _format_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.2f}"


def _format_setting(shot: str, context: str) -> str:
    shot_label = SHOT_LABELS.get(shot, shot)
    context_label = CONTEXT_LABELS.get(context, context)
    return f"{shot_label} / {context_label}"


def _metric_title(metric_label: str) -> str:
    return metric_label.replace("_", " ").title()


def _metric_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if str(column).strip().lower().startswith("metric_")
        and not str(column).strip().lower().startswith("metric__")
    ]


def _load_metrics(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    metric_columns = _metric_columns(df)
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
    if "metric_label" not in df.columns:
        raise click.ClickException("Metrics CSV missing metric columns.")
    df["metric_label"] = df["metric_label"].astype(str).str.lower()
    return df


def _normalize_model_name(model_name: str) -> str:
    model_name = model_name.strip()
    if model_name.startswith("nvidia/"):
        return model_name[len("nvidia/") :]
    return model_name


def _model_id(agent_name: str, model_name: str) -> str | None:
    agent_name = str(agent_name or "").strip()
    model_name = _normalize_model_name(str(model_name or ""))
    if not agent_name or not model_name:
        return None
    return f"{agent_name}/{model_name}"


def _unique_models(df: pd.DataFrame) -> list[str]:
    models: list[str] = []
    seen = set()
    for _, row in df.iterrows():
        model_id = _model_id(row.get("agent_name"), row.get("model_name"))
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(model_id)
    unknown_order = {model_id: idx for idx, model_id in enumerate(models)}
    order_map = {name: idx for idx, name in enumerate(MODEL_ORDER)}

    def _sort_key(model_id: str) -> tuple[int, int]:
        model_name = model_id.split("/", 1)[-1]
        normalized = _normalize_model_name(model_name)
        return (order_map.get(normalized, len(order_map)), unknown_order[model_id])

    return sorted(models, key=_sort_key)


def _build_table(
    dataset_key: str,
    metric_label: str,
    models: list[str],
    values: dict[tuple[str, str, str], float],
    dataset_name: str | None = None,
) -> str:
    config = DATASET_CONFIGS[dataset_key]
    col_spec = "l|" + ("c" * len(models))
    dataset_suffix = f" ({dataset_name})" if dataset_name else ""
    caption = (
        f"{config['title']} {_metric_title(metric_label)} on the {VARIANT} split"
        f"{dataset_suffix}."
    )
    label_suffix = f"_{_normalize_label(dataset_name)}" if dataset_name else ""
    label = f"tab:{dataset_key}{label_suffix}_{metric_label}"
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{3pt}",
        "    \\renewcommand{\\arraystretch}{1.1}",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        "    Setting & " + " & ".join(_format_model_name(model) for model in models) + " \\\\",
        "    \\midrule",
    ]
    for shot in config["shot_order"]:
        for context in config["context_order"]:
            row_label = _format_setting(shot, context)
            cells = []
            for model in models:
                cells.append(_format_value(values.get((model, shot, context))))
            lines.append(f"    {row_label} & " + " & ".join(cells) + " \\\\")
    lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabular}",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def _latest_metrics_csv(metrics_dir: Path) -> Path:
    candidates = sorted(metrics_dir.glob("metrics__*.csv"))
    if not candidates:
        raise click.ClickException(f"No metrics__*.csv in {metrics_dir}")
    return candidates[-1]


def _metric_values(
    df: pd.DataFrame,
    metric_label: str,
    dataset_key: str,
) -> dict[tuple[str, str, str], float]:
    config = DATASET_CONFIGS[dataset_key]
    df_metric = df.loc[df["metric_label"] == metric_label].copy()
    if df_metric.empty:
        return {}
    df_metric["metric_value"] = pd.to_numeric(
        df_metric["metric_value"], errors="coerce"
    )
    if metric_label == "accuracy":
        df_metric["metric_value"] = df_metric["metric_value"] * 100.0
    rows = []
    for _, row in df_metric.iterrows():
        context, shot = _infer_context_and_shot(
            row, config["context_order"], config["shot_order"]
        )
        if context not in config["context_order"] or shot not in config["shot_order"]:
            continue
        model_id = _model_id(row.get("agent_name"), row.get("model_name"))
        if not model_id:
            continue
        value = row.get("metric_value")
        if pd.isna(value):
            continue
        rows.append(
            {
                "model_id": model_id,
                "context": context,
                "shot": shot,
                "metric_value": float(value),
            }
        )
    if not rows:
        return {}
    agg = (
        pd.DataFrame(rows)
        .groupby(["model_id", "shot", "context"], sort=False)["metric_value"]
        .mean()
        .reset_index()
    )
    values: dict[tuple[str, str, str], float] = {}
    for _, row in agg.iterrows():
        values[(row["model_id"], row["shot"], row["context"])] = float(
            row["metric_value"]
        )
    return values


@click.command()
@click.option(
    "--results-root",
    type=click.Path(path_type=Path, exists=True),
    default=Path("results/report_numerical_metrics"),
    show_default=True,
    help="Root directory with report_numerical_metrics outputs.",
)
def main(results_root: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset_key in DATASET_CONFIGS:
        metrics_dir = results_root / dataset_key / VARIANT
        metrics_csv = _latest_metrics_csv(metrics_dir)
        df = _load_metrics(metrics_csv)
        models = _unique_models(df)
        metric_labels = sorted(
            str(label)
            for label in df["metric_label"].dropna().unique()
            if str(label).strip()
        )
        for metric_label in metric_labels:
            values = _metric_values(df, metric_label, dataset_key)
            table = _build_table(dataset_key, metric_label, models, values)
            output_path = OUTPUT_DIR / f"{dataset_key}_{metric_label}.tex"
            output_path.write_text(table)
        dataset_series = df["dataset"].astype(str)
        dataset_names = sorted(
            {name.strip() for name in dataset_series if name.strip() and name != "nan"}
        )
        for dataset_name in dataset_names:
            dataset_df = df.loc[dataset_series == dataset_name]
            models = _unique_models(dataset_df)
            for metric_label in metric_labels:
                values = _metric_values(dataset_df, metric_label, dataset_key)
                table = _build_table(
                    dataset_key,
                    metric_label,
                    models,
                    values,
                    dataset_name=dataset_name,
                )
                dataset_token = _normalize_label(dataset_name)
                output_path = (
                    OUTPUT_DIR / f"{dataset_key}_{dataset_token}_{metric_label}.tex"
                )
                output_path.write_text(table)


if __name__ == "__main__":
    main()
