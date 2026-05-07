#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import click
import pandas as pd

CONTEXT_ORDER = ["none", "low", "high", "ground_truth"]
SHOT_ORDER = ["few_shot", "one_shot", "zero_shot"]
CAPTION = (
    "tsENV classification accuracy on the lite suite (6 questions per setting). "
    "Dashes indicate settings not present in the current report."
)
LABEL = "tab:tsenv_cls"


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _match_token(value: str, tokens: list[str]) -> str | None:
    for token in tokens:
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", value):
            return token
    return None


def _infer_context_and_shot(row: pd.Series) -> tuple[str | None, str | None]:
    candidates = " ".join(
        str(row.get(col) or "") for col in ("variant", "dataset_label", "benchmark_id")
    )
    normalized = _normalize_label(candidates)
    context = _match_token(normalized, CONTEXT_ORDER)
    shot = _match_token(normalized, SHOT_ORDER)
    if not context:
        fallback = str(row.get("context_level") or "").strip().lower()
        context = fallback or None
    if not shot:
        fallback = str(row.get("shot_level") or "").strip().lower()
        shot = fallback or None
    return context, shot


def _format_model_name(model_name: str) -> str:
    parts = model_name.split("/")
    if len(parts) >= 3:
        left = "/".join(parts[:2]) + "/"
        right = "/".join(parts[2:])
        return f"\\makecell[l]{{{left}\\\\{right}}}"
    return model_name


def _format_value(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.2f}"


def _build_table(models: list[str], values: dict[tuple[str, str, str], float]) -> str:
    context_count = len(CONTEXT_ORDER)
    shot_group_spec = "|".join(["c" * context_count for _ in SHOT_ORDER])
    lines = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\small",
        "    \\setlength{\\tabcolsep}{3pt}",
        "    \\renewcommand{\\arraystretch}{1.1}",
        f"    \\begin{{tabular}}{{l|{shot_group_spec}}}",
        "    \\toprule",
        "    \\makecell{Model} &",
        "    \\multicolumn{4}{c}{Few-shot Accuracy} &",
        "    \\multicolumn{4}{c}{One-shot Accuracy} &",
        "    \\multicolumn{4}{c}{Zero-shot Accuracy} \\\\",
        "    & None & Low & High & Ground Truth & None & Low & High & Ground Truth & None & Low & High & Ground Truth \\\\",
        "    \\midrule",
    ]
    for model_name in models:
        cells = []
        for shot in SHOT_ORDER:
            for context in CONTEXT_ORDER:
                key = (model_name, shot, context)
                cells.append(_format_value(values.get(key)))
        line = f"    {_format_model_name(model_name)} & " + " & ".join(cells) + " \\\\"
        lines.append(line)
    lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabular}",
            f"    \\caption{{{CAPTION}}}",
            f"    \\label{{{LABEL}}}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


@click.command()
@click.option(
    "--metrics-csv",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to a tsenv_cls metrics__*.csv file.",
)
def main(metrics_csv: Path) -> None:
    df = pd.read_csv(metrics_csv)
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
    accuracy_mask = df["metric_label"].astype(str).str.lower() == "accuracy"
    df_classification = df.loc[accuracy_mask].copy()
    if df_classification.empty:
        raise click.ClickException("No accuracy rows found in metrics CSV.")

    agent_models = (
        df_classification[["agent_name", "model_name"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .apply(
            lambda row: f"{row['agent_name']}/{row['model_name'].strip('/nvidia')}",
            axis=1,
        )
        .tolist()
    )

    rows = []
    for _, row in df_classification.iterrows():
        context, shot = _infer_context_and_shot(row)
        if context not in CONTEXT_ORDER or shot not in SHOT_ORDER:
            continue
        value = pd.to_numeric(row.get("metric_value"), errors="coerce")
        if pd.isna(value):
            continue
        agent_name = str(row.get("agent_name") or "").strip()
        model_name = str(row.get("model_name") or "").strip("/nvidia")
        if not agent_name or not model_name:
            continue
        model_id = f"{agent_name}/{model_name}"
        rows.append(
            {
                "model_id": model_id,
                "context": context,
                "shot": shot,
                "metric_value": float(value),
            }
        )

    values: dict[tuple[str, str, str], float] = {}
    if rows:
        agg = (
            pd.DataFrame(rows)
            .groupby(["model_id", "shot", "context"], sort=False)["metric_value"]
            .mean()
            .reset_index()
        )
        for _, row in agg.iterrows():
            values[(row["model_id"], row["shot"], row["context"])] = float(
                row["metric_value"]
            )
    print(_build_table(agent_models, values))


if __name__ == "__main__":
    main()
