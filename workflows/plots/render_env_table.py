#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import click
import pandas as pd

VARIANT = "separable"
SHOT_LEVELS = ["zero_shot", "one_shot", "few_shot"]
STACKED_SHOT_LABELS = {
    "zero_shot": "Zero-shot",
    "one_shot": "One-shot",
    "few_shot": "Few-shot",
}
CONTEXT_LABELS = {
    "ground_truth": "Ground Truth",
    "low": "Low",
    "high": "High",
    "none": "None",
}
SIMBENCH_DATASETS = [
    "DampedMassBetweenWalls",
    "BallDrop",
    "DoubleMassSpringDamperSameCoeffs",
    "InclinedPlane",
    "MassSpringDamperWithPID",
    "TransmissionLine",
]
SIMBENCH_DATASET_LABELS = {
    "BallDrop": "BounceBall",
    "DampedMassBetweenWalls": "WallMass",
    "InclinedPlane": "MassSlide",
    "DoubleMassSpringDamperSameCoeffs": "2MSD",
    "MassSpringDamperWithPID": "MSD\\_PID",
    "TransmissionLine": "TransLine",
}
SIMBENCH_STACKED_LABELS = {
    "DampedMassBetweenWalls": r"\makecell{\texttt{Wall}\\\texttt{Mass}}",
    "BallDrop": r"\makecell{\texttt{Bounce}\\\texttt{Ball}}",
    "DoubleMassSpringDamperSameCoeffs": r"\texttt{2MSD}",
    "InclinedPlane": r"\makecell{\texttt{Mass}\\\texttt{Slide}}",
    "MassSpringDamperWithPID": r"\makecell{\texttt{MSD}\\\texttt{\_PID}}",
    "TransmissionLine": r"\makecell{\texttt{Trans}\\\texttt{Line}}",
}
UCR_DATASETS = [
    "Chinatown",
    "BME",
    "SyntheticControl",
    "GunPoint",
    "Wafer",
]
UCR_STACKED_LABELS = {
    "Chinatown": r"\texttt{Chinatown}",
    "BME": r"\texttt{BME}",
    "SyntheticControl": r"\makecell{\texttt{Synthetic}\\\texttt{Control}}",
    "GunPoint": r"\texttt{GunPoint}",
    "Wafer": r"\texttt{Wafer}",
}
UCR_SHOT_LABELS = {
    "zero_shot": "Zero-shot",
    "one_shot": "One-shot",
    "few_shot": "Few-shot",
}
BATCH_CONFIGS = [
    {
        "key": "simbench_cls",
        "title": "SimBench",
        "datasets": SIMBENCH_DATASETS,
        "dataset_labels": SIMBENCH_DATASET_LABELS,
        "context_order": ["none", "low", "high", "ground_truth"],
        "shot_labels": {
            "zero_shot": "Zero-shot",
            "one_shot": "One-shot",
            "few_shot": "Few-shot",
        },
        "output_prefix": "simbench_env_context",
        "label_prefix": "simbench-env-context",
        "decimals": 1,
        "missing": "--",
    },
    {
        "key": "ucr",
        "title": "UCR",
        "datasets": UCR_DATASETS,
        "dataset_labels": {},
        "context_order": ["none"],
        "shot_labels": UCR_SHOT_LABELS,
        "output_prefix": "ucr_env_context",
        "label_prefix": "ucr-env-context",
        "decimals": 1,
        "missing": "-",
    },
]
MODELS = [
    "claude-sonnet-4-5-20250929",
    "gemini-3-pro",
    "gpt-5-codex",
    "nvidia/qwen/qwen3-coder-480b-a35b-instruct",
    "nvidia/moonshotai/kimi-k2-thinking",
]
QWEN_MODEL_NAME = "nvidia/qwen/qwen3-coder-480b-a35b-instruct"
RANK_DECIMALS = 2
GAIN_DECIMALS = 2
REMAPPING = {
    "claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "gemini-3-pro": "Gemini 3 Pro",
    "gpt-5-codex": "GPT 5 Codex",
    "nvidia/qwen/qwen3-coder-480b-a35b-instruct": "Qwen Coder 480B",
    "nvidia/moonshotai/kimi-k2-thinking": "Kimi K2 Thinking",
}
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "68e3980bb2dcb29a5a67ec17" / "tables"
)
METRICS_CSV_OVERRIDES = {
    "simbench_cls": Path(
        "results/report_numerical_metrics/simbench_cls/separable/metrics__2026-01-28__22-55-55.csv"
    ),
    "ucr": Path(
        "results/report_numerical_metrics/ucr/separable/metrics__2026-01-28__22-58-07.csv"
    ),
}
WITHIN_TIME_SUFFIX = "_withing_1_sec"
WITHIN_TIME_EXTENSION = ".text"
PROMPT_GAIN_NONE_AND_PICTURE_CSV = Path(
    "results/report_numerical_metrics/simbench_cls/none_and_picture/metrics__2026-01-28__22-57-08.csv"
)
PROMPT_GAIN_SEPARABLE_CSV = Path(
    "results/report_numerical_metrics/simbench_cls/separable/metrics__2026-01-28__22-55-55.csv"
)
PROMPT_GAIN_MODELS = [
    "claude-sonnet-4-5-20250929",
    "gpt-5-codex",
]
PROMPT_GAIN_CONTEXTS = ["none", "low", "high", "ground_truth"]
PROMPT_GAIN_DECIMALS = 2
ZERO_SHOT_EXCLUDE_DATASETS = {"Chinatown", "GunPoint", "Wafer"}


def _latest_metrics_csv(metrics_dir: Path) -> Path:
    candidates = sorted(metrics_dir.glob("metrics__*.csv"))
    if not candidates:
        raise click.ClickException(f"No metrics__*.csv in {metrics_dir}")
    return candidates[-1]


def _metrics_csv_for_config(metrics_dir: Path, key: str) -> Path:
    if key in METRICS_CSV_OVERRIDES:
        return METRICS_CSV_OVERRIDES[key]
    return _latest_metrics_csv(metrics_dir)


def _apply_output_suffix(path: Path, output_suffix: str, output_extension: str) -> Path:
    if not output_suffix:
        return path
    return path.with_name(f"{path.stem}{output_suffix}{output_extension}")


def _apply_metric_column(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    if metric_column == "metric_accuracy":
        return df
    updated = df.copy()
    updated["metric_accuracy"] = updated[metric_column]
    return updated


def _dataset_order(df: pd.DataFrame, preferred: list[str]) -> list[str]:
    present = [str(item) for item in pd.unique(df["dataset"]) if str(item).strip()]
    order = [name for name in preferred if name in present]
    order.extend(name for name in present if name not in order)
    return order


def _model_order(df: pd.DataFrame) -> list[str]:
    present = [str(item) for item in pd.unique(df["model_name"]) if str(item).strip()]
    order = [name for name in MODELS if name in present]
    order.extend(name for name in present if name not in order)
    return order


def _format_value(
    value: float | None,
    best_value: float | None,
    decimals: int,
    missing: str,
    strip_leading_zero: bool = False,
    math_mode: bool = True,
    highlight_best: bool = True,
) -> str:
    if value is None or pd.isna(value):
        return missing
    if strip_leading_zero and math.isclose(
        float(value), 0.0, abs_tol=0.5 * (10 ** -decimals)
    ):
        value = 0.0
    display = f"{value:.{decimals}f}"
    if strip_leading_zero:
        if display.startswith("-0."):
            display = "-." + display[3:]
        elif display.startswith("0."):
            display = "." + display[2:]
    if highlight_best and best_value is not None and not pd.isna(best_value):
        if math.isclose(float(value), float(best_value), rel_tol=1e-12, abs_tol=1e-12):
            if math_mode:
                return rf"$\boldsymbol{{{display}}}$"
            return rf"\textbf{{{display}}}"
    if math_mode:
        return f"${display}$"
    return display


def _cmidrules(
    dataset_count: int,
    context_count: int,
    start_col: int = 2,
) -> str:
    rules = []
    for idx in range(dataset_count):
        start = start_col + idx * context_count
        end = start + context_count - 1
        rules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    return "".join(rules)


def _stacked_label(dataset: str, labels: dict[str, str]) -> str:
    if dataset in labels:
        return labels[dataset]
    return rf"\texttt{{{dataset}}}"


def _stacked_block_lines(
    title_line: str,
    dataset_order: list[str],
    context_order: list[str],
    dataset_labels: dict[str, str],
    model_order: list[str],
    values: dict[tuple[str, str, str, str], float],
    missing: str,
    decimals: int,
    model_labels: dict[str, str] | None = None,
) -> list[str]:
    column_count = len(dataset_order) * len(context_order)
    col_spec = f"S M*{{{column_count}}}{{Y}}"
    header = r"\multirow{2}{*}{Setting} & \multirow{2}{*}{Model} & " + " & ".join(
        rf"\multicolumn{{{len(context_order)}}}{{c}}{{{_stacked_label(dataset, dataset_labels)}}}"
        for dataset in dataset_order
    ) + r" \\"
    context_labels = " & ".join(
        CONTEXT_LABELS.get(context, context) for context in context_order
    )
    subheader = " &  & " + " & ".join(
        [context_labels for _ in dataset_order]
    ) + r" \\"

    lines = [
        title_line,
        "",
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_spec}}}",
        r"\toprule",
        header,
        _cmidrules(len(dataset_order), len(context_order), start_col=3),
        subheader,
        r"\midrule",
    ]

    for shot_idx, shot_level in enumerate(SHOT_LEVELS):
        column_bests: dict[tuple[str, str], float] = {}
        for dataset in dataset_order:
            for context in context_order:
                column_values = [
                    values[(shot_level, model, dataset, context)]
                    for model in model_order
                    if (shot_level, model, dataset, context) in values
                ]
                if column_values:
                    column_bests[(dataset, context)] = max(column_values)

        rows = []
        for model in model_order:
            model_label = (model_labels or {}).get(model, REMAPPING.get(model, model))
            row_cells = []
            for dataset in dataset_order:
                for context in context_order:
                    value = values.get((shot_level, model, dataset, context))
                    best_value = column_bests.get((dataset, context))
                    row_cells.append(
                        _format_value(value, best_value, decimals, missing)
                    )
            rows.append(f"{model_label} & " + " & ".join(row_cells) + r" \\")
        if not rows:
            raise click.ClickException(
                f"No matching rows for shot_level={shot_level}."
            )

        shot_label = STACKED_SHOT_LABELS[shot_level]
        lines.append(
            rf"\multirow{{{len(rows)}}}{{*}}{{\shortstack[l]{{{shot_label}}}}} & {rows[0]}"
        )
        lines.extend(f"& {row}" for row in rows[1:])
        if shot_idx < len(SHOT_LEVELS) - 1:
            lines.append(r"\midrule")

    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return lines


def _gain_block_lines(
    title_line: str,
    model_order: list[str],
    shot_labels: dict[str, str],
    values: dict[tuple[str, str], float],
    missing: str,
    decimals: int,
) -> list[str]:
    col_spec = f"M*{{{len(SHOT_LEVELS)}}}{{Y}}"
    header = "Model & " + " & ".join(
        shot_labels.get(shot_level, shot_level) for shot_level in SHOT_LEVELS
    ) + r" \\"

    column_bests: dict[str, float] = {}
    for shot_level in SHOT_LEVELS:
        column_values = [
            values[(shot_level, model)]
            for model in model_order
            if (shot_level, model) in values
        ]
        if column_values:
            column_bests[shot_level] = max(column_values)

    lines = [
        title_line,
        "",
        rf"\begin{{tabularx}}{{\columnwidth}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for model in model_order:
        model_label = REMAPPING.get(model, model)
        row_cells = []
        for shot_level in SHOT_LEVELS:
            value = values.get((shot_level, model))
            best_value = column_bests.get(shot_level)
            row_cells.append(
                _format_value(
                    value,
                    best_value,
                    decimals,
                    missing,
                    strip_leading_zero=True,
                    highlight_best=False,
                )
            )
        lines.append(f"{model_label} & " + " & ".join(row_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return lines


def _gain_tabular_lines(
    model_order: list[str],
    shot_labels: dict[str, str],
    values: dict[tuple[str, str], float],
    missing: str,
    decimals: int,
    width: str,
) -> list[str]:
    col_spec = f"M*{{{len(SHOT_LEVELS)}}}{{Y}}"
    header = "Model & " + " & ".join(
        shot_labels.get(shot_level, shot_level) for shot_level in SHOT_LEVELS
    ) + r" \\"
    lines = [
        rf"\begin{{tabularx}}{{{width}}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for model in model_order:
        model_label = REMAPPING.get(model, model)
        row_cells = []
        for shot_level in SHOT_LEVELS:
            value = values.get((shot_level, model))
            row_cells.append(
                _format_value(
                    value,
                    None,
                    decimals,
                    missing,
                    strip_leading_zero=True,
                    highlight_best=False,
                )
            )
        lines.append(f"{model_label} & " + " & ".join(row_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return lines


def _shot_gain_block_lines(
    title_line: str,
    model_order: list[str],
    gain_labels: dict[str, str],
    gain_keys: list[str],
    values: dict[tuple[str, str], float],
    missing: str,
    decimals: int,
) -> list[str]:
    col_spec = f"M*{{{len(gain_keys)}}}{{Y}}"
    header = "Model & " + " & ".join(
        gain_labels.get(gain_key, gain_key) for gain_key in gain_keys
    ) + r" \\"

    lines = [
        title_line,
        "",
        rf"\begin{{tabularx}}{{\columnwidth}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for model in model_order:
        model_label = REMAPPING.get(model, model)
        row_cells = []
        for gain_key in gain_keys:
            value = values.get((gain_key, model))
            row_cells.append(
                _format_value(
                    value,
                    None,
                    decimals,
                    missing,
                    strip_leading_zero=True,
                    highlight_best=False,
                )
            )
        lines.append(f"{model_label} & " + " & ".join(row_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return lines


def _shot_gain_tabular_lines(
    model_order: list[str],
    gain_labels: dict[str, str],
    gain_keys: list[str],
    values: dict[tuple[str, str], float],
    missing: str,
    decimals: int,
    width: str,
) -> list[str]:
    col_spec = f"M*{{{len(gain_keys)}}}{{Y}}"
    header = "Model & " + " & ".join(
        gain_labels.get(gain_key, gain_key) for gain_key in gain_keys
    ) + r" \\"
    lines = [
        rf"\begin{{tabularx}}{{{width}}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for model in model_order:
        model_label = REMAPPING.get(model, model)
        row_cells = []
        for gain_key in gain_keys:
            value = values.get((gain_key, model))
            row_cells.append(
                _format_value(
                    value,
                    None,
                    decimals,
                    missing,
                    strip_leading_zero=True,
                    highlight_best=False,
                )
            )
        lines.append(f"{model_label} & " + " & ".join(row_cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    return lines


def _average_rank_values(
    df: pd.DataFrame,
    dataset_order: list[str],
    context_order: list[str],
) -> dict[tuple[str, str, str], float]:
    filtered = df.loc[
        df["dataset"].isin(dataset_order)
        & df["context_level"].isin(context_order)
        & df["shot_level"].isin(SHOT_LEVELS)
    ].copy()
    if filtered.empty:
        raise click.ClickException("No data available for average rank table.")

    grouped = (
        filtered.groupby(
            ["shot_level", "context_level", "dataset", "model_name"],
            as_index=False,
        )["metric_accuracy"]
        .mean()
    )
    if grouped.empty:
        raise click.ClickException("No grouped data for average rank table.")

    grouped["rank"] = grouped.groupby(
        ["shot_level", "context_level", "dataset"]
    )["metric_accuracy"].rank(ascending=False, method="average")
    averaged = (
        grouped.groupby(
            ["shot_level", "context_level", "model_name"], as_index=False
        )["rank"]
        .mean()
    )

    values: dict[tuple[str, str, str], float] = {}
    for _, row in averaged.iterrows():
        values[
            (row["shot_level"], row["context_level"], row["model_name"])
        ] = float(row["rank"])
    return values


def _shot_gain_values(
    df: pd.DataFrame,
    dataset_order: list[str],
    context_order: list[str],
) -> dict[tuple[str, str], float]:
    filtered = df.loc[
        df["dataset"].isin(dataset_order)
        & df["context_level"].isin(context_order)
        & df["shot_level"].isin(SHOT_LEVELS)
    ].copy()
    if filtered.empty:
        raise click.ClickException("No data available for shot gain table.")

    grouped = (
        filtered.groupby(
            ["shot_level", "model_name", "dataset", "context_level"],
            as_index=False,
        )["metric_accuracy"]
        .mean()
    )
    pivot = grouped.pivot_table(
        index=["model_name", "dataset", "context_level"],
        columns="shot_level",
        values="metric_accuracy",
    )
    if "zero_shot" not in pivot.columns:
        raise click.ClickException("Missing zero_shot results for shot gain table.")

    gains = []
    pivot = pivot.reset_index()
    for gain_key, target in [("gain_01", "one_shot"), ("gain_03", "few_shot")]:
        if target not in pivot.columns:
            continue
        subset = pivot.dropna(subset=["zero_shot", target]).copy()
        if subset.empty:
            continue
        subset["gain_key"] = gain_key
        subset["gain_value"] = subset[target] - subset["zero_shot"]
        gains.append(subset[["model_name", "gain_key", "gain_value"]])

    if not gains:
        raise click.ClickException("No paired shot results for shot gain table.")

    combined = pd.concat(gains, ignore_index=True)
    averaged = (
        combined.groupby(["gain_key", "model_name"], as_index=False)["gain_value"]
        .mean()
    )
    values: dict[tuple[str, str], float] = {}
    for _, row in averaged.iterrows():
        values[(row["gain_key"], row["model_name"])] = float(row["gain_value"])
    return values


def _context_gain_values(
    df: pd.DataFrame,
    dataset_order: list[str],
    context_order: list[str],
) -> dict[tuple[str, str], float]:
    if len(context_order) < 2:
        raise click.ClickException(
            "Context gain table requires at least two context levels."
        )
    base_context = context_order[0]
    target_context = context_order[-1]
    filtered = df.loc[
        df["dataset"].isin(dataset_order)
        & df["context_level"].isin([base_context, target_context])
        & df["shot_level"].isin(SHOT_LEVELS)
    ].copy()
    if filtered.empty:
        raise click.ClickException("No data available for context gain table.")

    grouped = (
        filtered.groupby(
            ["shot_level", "model_name", "dataset", "context_level"],
            as_index=False,
        )["metric_accuracy"]
        .mean()
    )
    pivot = grouped.pivot_table(
        index=["shot_level", "model_name", "dataset"],
        columns="context_level",
        values="metric_accuracy",
    )
    if base_context not in pivot.columns or target_context not in pivot.columns:
        raise click.ClickException(
            "Missing context levels for context gain table."
        )
    pivot = pivot.dropna(subset=[base_context, target_context])
    if pivot.empty:
        raise click.ClickException(
            "No paired context results for context gain table."
        )
    pivot["gain"] = pivot[target_context] - pivot[base_context]
    pivot = pivot.reset_index()
    averaged = (
        pivot.groupby(["shot_level", "model_name"], as_index=False)["gain"]
        .mean()
    )

    values: dict[tuple[str, str], float] = {}
    for _, row in averaged.iterrows():
        values[(row["shot_level"], row["model_name"])] = float(row["gain"])
    return values


def _prompt_gain_zero_shot_table() -> None:
    df_none = pd.read_csv(PROMPT_GAIN_NONE_AND_PICTURE_CSV)
    df_sep = pd.read_csv(PROMPT_GAIN_SEPARABLE_CSV)
    df_none["model_name"] = df_none["model_name"].replace(
        {"claude-sonnet-4-5": "claude-sonnet-4-5-20250929"}
    )

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        updated = df.loc[df["shot_level"] == "zero_shot"].copy()
        updated = updated.loc[updated["context_level"].isin(PROMPT_GAIN_CONTEXTS)]
        updated = updated.loc[updated["model_name"].isin(PROMPT_GAIN_MODELS)]
        updated = updated.loc[
            (updated["model_name"] != QWEN_MODEL_NAME)
            | (updated["agent_name"] == "terminus-ts-xml-think")
        ]
        return updated

    none_ready = _prep(df_none)
    sep_ready = _prep(df_sep)
    if none_ready.empty or sep_ready.empty:
        raise click.ClickException("Missing zero-shot data for prompt gain table.")

    none_grouped = (
        none_ready.groupby(
            ["model_name", "dataset", "context_level"], as_index=False
        )["metric_accuracy"]
        .mean()
    )
    sep_grouped = (
        sep_ready.groupby(
            ["model_name", "dataset", "context_level"], as_index=False
        )["metric_accuracy"]
        .mean()
    )
    merged = none_grouped.merge(
        sep_grouped,
        on=["model_name", "dataset", "context_level"],
        how="inner",
        suffixes=("_none", "_sep"),
    )
    if merged.empty:
        raise click.ClickException("No overlapping rows for prompt gain table.")

    merged["gain"] = merged["metric_accuracy_none"] - merged["metric_accuracy_sep"]
    averaged = (
        merged.groupby(["model_name", "context_level"], as_index=False)["gain"]
        .mean()
    )

    values: dict[tuple[str, str], float] = {}
    for _, row in averaged.iterrows():
        values[(row["model_name"], row["context_level"])] = float(row["gain"])

    for model in PROMPT_GAIN_MODELS:
        for context in PROMPT_GAIN_CONTEXTS:
            if (model, context) not in values:
                raise click.ClickException(
                    f"Missing gain for model={model}, context={context}."
                )

    col_spec = f"M*{{{len(PROMPT_GAIN_CONTEXTS)}}}{{Y}}"
    header = "Model & " + " & ".join(
        CONTEXT_LABELS.get(context, context) for context in PROMPT_GAIN_CONTEXTS
    ) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\begin{{tabularx}}{{\columnwidth}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for model in PROMPT_GAIN_MODELS:
        model_label = REMAPPING.get(model, model)
        row_cells = []
        for context in PROMPT_GAIN_CONTEXTS:
            row_cells.append(
                _format_value(
                    values[(model, context)],
                    None,
                    PROMPT_GAIN_DECIMALS,
                    "--",
                    highlight_best=False,
                )
            )
        lines.append(f"{model_label} & " + " & ".join(row_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{Prompt gain (none\_and\_picture $-$ separable) in zero-shot accuracy, averaged across SimBench datasets.}",
            r"\label{tab:simbench-prompt-gain-zero-shot}",
            r"\end{table}",
            "",
        ]
    )

    output_path = OUTPUT_DIR / "simbench_prompt_gain_zero_shot.tex"
    output_path.write_text("\n".join(lines))
    print(f"Wrote LaTeX table to {output_path}")


def _render_tables(
    results_root: Path,
    output_path: Path | None,
    group_by_setting: bool,
    metric_column: str,
    output_suffix: str,
    output_extension: str,
) -> None:
    base_output_path = output_path
    stacked_configs: dict[str, dict[str, object]] = {}
    gain_configs: dict[str, dict[str, object]] = {}
    simbench_qwen_df: pd.DataFrame | None = None
    simbench_config: dict[str, object] | None = None

    for config in BATCH_CONFIGS:
        metrics_dir = results_root / config["key"] / VARIANT
        metrics_csv = _metrics_csv_for_config(metrics_dir, config["key"])
        df = pd.read_csv(metrics_csv)
        df = df.loc[df["context_level"].isin(config["context_order"])].copy()
        df = df.loc[df["model_name"].isin(MODELS)].copy()
        df = _apply_metric_column(df, metric_column)
        if config["key"] == "simbench_cls":
            simbench_qwen_df = df.copy()
            simbench_config = config
        df = df.loc[
            (df["model_name"] != QWEN_MODEL_NAME)
            | (df["agent_name"] == "terminus-ts-xml-think")
        ]
        df = df.loc[
            ~(
                (df["shot_level"] == "zero_shot")
                & (df["dataset"].isin(ZERO_SHOT_EXCLUDE_DATASETS))
            )
        ].copy()
        config_output_path = (
            base_output_path if config["key"] == "simbench_cls" else None
        )

        gain_dataset_order = _dataset_order(df, config["datasets"])
        gain_model_order = _model_order(df)
        if not gain_dataset_order or not gain_model_order:
            raise click.ClickException(
                f"No matching rows for {config['key']} context gain table."
            )
        gain_values = None
        if len(config["context_order"]) >= 2:
            gain_values = _context_gain_values(
                df, gain_dataset_order, config["context_order"]
            )
        shot_gain_values = _shot_gain_values(
            df, gain_dataset_order, config["context_order"]
        )
        gain_configs[config["key"]] = {
            "model_order": gain_model_order,
            "values": gain_values,
            "shot_gain_values": shot_gain_values,
            "missing": config["missing"],
            "decimals": config["decimals"],
            "shot_labels": config["shot_labels"],
            "context_order": config["context_order"],
        }

        if group_by_setting:
            dataset_order = _dataset_order(df, config["datasets"])
            model_order = _model_order(df)
            if not dataset_order or not model_order:
                raise click.ClickException(
                    f"No matching rows for {config['key']} grouped table."
                )

            grouped = (
                df.groupby(
                    ["shot_level", "model_name", "dataset", "context_level"],
                    as_index=False,
                )["metric_accuracy"]
                .mean()
            )
            values: dict[tuple[str, str, str, str], float] = {}
            for _, row in grouped.iterrows():
                values[
                    (
                        row["shot_level"],
                        row["model_name"],
                        row["dataset"],
                        row["context_level"],
                    )
                ] = float(row["metric_accuracy"])
            stacked_configs[config["key"]] = {
                "dataset_order": dataset_order,
                "model_order": model_order,
                "context_order": config["context_order"],
                "values": values,
                "missing": config["missing"],
                "decimals": config["decimals"],
            }

            column_count = len(config["context_order"]) * len(dataset_order)
            col_spec = f"S M*{{{column_count}}}{{Y}}"
            header = r"\multirow{2}{*}{Setting} & \multirow{2}{*}{Model} & " + " & ".join(
                rf"\multicolumn{{{len(config['context_order'])}}}{{c}}{{\texttt{{{config['dataset_labels'].get(dataset, dataset)}}}}}"
                for dataset in dataset_order
            ) + r" \\"
            context_labels = " & ".join(
                CONTEXT_LABELS.get(context, context)
                for context in config["context_order"]
            )
            subheader = " &  & " + " & ".join(
                [context_labels for _ in dataset_order]
            ) + r" \\"

            lines = [
                r"\begin{table*}[t]",
                r"\centering",
                r"\begin{tabularx}{\textwidth}{" + col_spec + r"}",
                r"\toprule",
                header,
                _cmidrules(
                    len(dataset_order),
                    len(config["context_order"]),
                    start_col=3,
                ),
                subheader,
                r"\midrule",
            ]

            for shot_idx, shot_level in enumerate(SHOT_LEVELS):
                shot_df = df.loc[df["shot_level"] == shot_level].copy()
                if shot_df.empty:
                    raise click.ClickException(
                        f"No matching rows for {config['key']} shot_level={shot_level}."
                    )

                column_bests: dict[tuple[str, str], float] = {}
                for dataset in dataset_order:
                    for context in config["context_order"]:
                        column_values = [
                            values[(shot_level, model, dataset, context)]
                            for model in model_order
                            if (shot_level, model, dataset, context) in values
                        ]
                        if column_values:
                            column_bests[(dataset, context)] = max(column_values)

                rows = []
                for model in model_order:
                    model_label = REMAPPING.get(model, model)
                    row_cells = []
                    for dataset in dataset_order:
                        for context in config["context_order"]:
                            value = values.get(
                                (shot_level, model, dataset, context)
                            )
                            best_value = column_bests.get((dataset, context))
                            row_cells.append(
                                _format_value(
                                    value,
                                    best_value,
                                    config["decimals"],
                                    config["missing"],
                                )
                            )
                    rows.append(
                        f"{model_label} & " + " & ".join(row_cells) + r" \\"
                    )
                if not rows:
                    raise click.ClickException(
                        f"No matching rows for {config['key']} shot_level={shot_level}."
                    )

                shot_label = config["shot_labels"].get(shot_level, shot_level)
                lines.append(
                    rf"\multirow{{{len(rows)}}}{{*}}{{\shortstack[l]{{{shot_label}}}}} & {rows[0]}"
                )
                lines.extend(f"& {row}" for row in rows[1:])
                if shot_idx < len(SHOT_LEVELS) - 1:
                    lines.append(r"\midrule")

            lines.extend(
                [
                    r"\bottomrule",
                    r"\end{tabularx}",
                    rf"\caption{{Accuracy by \texttt{{{config['title']}}} environment and context level, grouped by evaluation setting.}}",
                    rf"\label{{tab:{config['label_prefix']}-all-left}}",
                    r"\end{table*}",
                    "",
                ]
            )

            grouped_output = OUTPUT_DIR / f"{config['output_prefix']}_all.tex"
            grouped_output = _apply_output_suffix(
                grouped_output, output_suffix, output_extension
            )
            grouped_output.write_text("\n".join(lines))
            print(f"Wrote LaTeX table to {grouped_output}")

        for shot_level in SHOT_LEVELS:
            shot_df = df.loc[df["shot_level"] == shot_level].copy()
            dataset_order = _dataset_order(shot_df, config["datasets"])
            model_order = _model_order(shot_df)
            if not dataset_order or not model_order:
                raise click.ClickException(
                    f"No matching rows for {config['key']} shot_level={shot_level}."
                )

            grouped = (
                shot_df.groupby(
                    ["model_name", "dataset", "context_level"], as_index=False
                )["metric_accuracy"]
                .mean()
            )
            values: dict[tuple[str, str, str], float] = {}
            for _, row in grouped.iterrows():
                values[
                    (row["model_name"], row["dataset"], row["context_level"])
                ] = float(row["metric_accuracy"])

            column_bests: dict[tuple[str, str], float] = {}
            for dataset in dataset_order:
                for context in config["context_order"]:
                    column_values = [
                        values.get((model, dataset, context))
                        for model in model_order
                        if (model, dataset, context) in values
                    ]
                    if column_values:
                        column_bests[(dataset, context)] = max(column_values)

            column_count = len(config["context_order"]) * len(dataset_order)
            col_spec = f"M*{{{column_count}}}{{Y}}"
            header = "Model & " + " & ".join(
                rf"\multicolumn{{{len(config['context_order'])}}}{{c}}{{\texttt{{{config['dataset_labels'].get(dataset, dataset)}}}}}"
                for dataset in dataset_order
            ) + r" \\"
            context_labels = " & ".join(
                CONTEXT_LABELS.get(context, context)
                for context in config["context_order"]
            )
            subheader = " & " + " & ".join(
                [context_labels for _ in dataset_order]
            ) + r" \\"

            lines = [
                r"\begin{table*}[t]",
                r"\centering",
                r"\begin{tabularx}{\textwidth}{" + col_spec + r"}",
                r"\toprule",
                header,
                _cmidrules(len(dataset_order), len(config["context_order"])),
                subheader,
                r"\midrule",
            ]

            for model in model_order:
                model_label = REMAPPING.get(model, model)
                row_cells = []
                for dataset in dataset_order:
                    for context in config["context_order"]:
                        value = values.get((model, dataset, context))
                        best_value = column_bests.get((dataset, context))
                        row_cells.append(
                            _format_value(
                                value,
                                best_value,
                                config["decimals"],
                                config["missing"],
                            )
                        )
                lines.append(f"{model_label} & " + " & ".join(row_cells) + r" \\")

            shot_label = config["shot_labels"].get(shot_level, shot_level)
            shot_token = shot_level.replace("_", "-")
            lines.extend(
                [
                    r"\bottomrule",
                    r"\end{tabularx}",
                    rf"\caption{{{shot_label} accuracy by {config['title']} environment and context level.}}",
                    rf"\label{{tab:{config['label_prefix']}-{shot_token}}}",
                    r"\end{table*}",
                    "",
                ]
            )

            if config_output_path is None:
                shot_output = (
                    OUTPUT_DIR / f"{config['output_prefix']}_{shot_level}.tex"
                )
            elif shot_level == "zero_shot":
                shot_output = config_output_path
            else:
                shot_output = config_output_path.with_name(
                    f"{config_output_path.stem}_{shot_level}{config_output_path.suffix}"
                )
            shot_output = _apply_output_suffix(
                shot_output, output_suffix, output_extension
            )
            shot_output.write_text("\n".join(lines))
            print(f"Wrote LaTeX table to {shot_output}")

        rank_dataset_order = _dataset_order(df, config["datasets"])
        rank_model_order = _model_order(df)
        if not rank_dataset_order or not rank_model_order:
            raise click.ClickException(
                f"No matching rows for {config['key']} average rank table."
            )

        rank_values = _average_rank_values(
            df, rank_dataset_order, config["context_order"]
        )
        column_bests: dict[tuple[str, str], float] = {}
        for shot_level in SHOT_LEVELS:
            for context in config["context_order"]:
                column_values = [
                    rank_values[(shot_level, context, model)]
                    for model in rank_model_order
                    if (shot_level, context, model) in rank_values
                ]
                if column_values:
                    column_bests[(shot_level, context)] = min(column_values)

        column_count = len(config["context_order"]) * len(SHOT_LEVELS)
        col_spec = f"M*{{{column_count}}}{{Y}}"
        header = "Model & " + " & ".join(
            rf"\multicolumn{{{len(config['context_order'])}}}{{c}}{{{config['shot_labels'].get(shot_level, shot_level)}}}"
            for shot_level in SHOT_LEVELS
        ) + r" \\"
        context_labels = " & ".join(
            CONTEXT_LABELS.get(context, context)
            for context in config["context_order"]
        )
        subheader = " & " + " & ".join(
            [context_labels for _ in SHOT_LEVELS]
        ) + r" \\"

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{0.95}",
            r"\begin{tabularx}{\columnwidth}{" + col_spec + r"}",
            r"\toprule",
            header,
            _cmidrules(len(SHOT_LEVELS), len(config["context_order"]), start_col=2),
            subheader,
            r"\midrule",
        ]

        for model in rank_model_order:
            model_label = REMAPPING.get(model, model)
            row_cells = []
            for shot_level in SHOT_LEVELS:
                for context in config["context_order"]:
                    value = rank_values.get((shot_level, context, model))
                    best_value = column_bests.get((shot_level, context))
                    row_cells.append(
                        _format_value(
                            value, best_value, RANK_DECIMALS, config["missing"]
                        )
                    )
            lines.append(f"{model_label} & " + " & ".join(row_cells) + r" \\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabularx}",
                rf"\caption{{Average model rank (1 = best) across datasets by shot setting and context level for {config['title']}.}}",
                rf"\label{{tab:{config['label_prefix']}-avg-rank}}",
                r"\end{table}",
                "",
            ]
        )

        rank_output = OUTPUT_DIR / f"{config['output_prefix']}_avg_rank.tex"
        rank_output = _apply_output_suffix(
            rank_output, output_suffix, output_extension
        )
        rank_output.write_text("\n".join(lines))
        print(f"Wrote LaTeX table to {rank_output}")

    if group_by_setting:
        if "simbench_cls" not in stacked_configs or "ucr" not in stacked_configs:
            raise click.ClickException("Missing simbench_cls or ucr metrics.")

        simbench = stacked_configs["simbench_cls"]
        ucr = stacked_configs["ucr"]
        stacked_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\setlength{\tabcolsep}{4pt}",
            "",
        ]
        stacked_lines.extend(
            _stacked_block_lines(
                r"\noindent\textbf{\name\ Synthetic}\par\vspace{4pt}",
                simbench["dataset_order"],
                simbench["context_order"],
                SIMBENCH_STACKED_LABELS,
                simbench["model_order"],
                simbench["values"],
                simbench["missing"],
                simbench["decimals"],
            )
        )
        stacked_lines.extend(["", r"\vspace{10pt}", ""])
        stacked_lines.extend(
            _stacked_block_lines(
                r"\noindent\textbf{\name\ Real}\par\vspace{4pt}",
                ucr["dataset_order"],
                ucr["context_order"],
                UCR_STACKED_LABELS,
                ucr["model_order"],
                ucr["values"],
                ucr["missing"],
                ucr["decimals"],
            )
        )
        stacked_lines.extend(
            [
                "",
                r"\caption{Accuracy by environment and context level across \name\ Synthetic and \name\ Real, grouped by evaluation setting. We omit \texttt{Chinatown}, \texttt{BME} and \texttt{GunPoint} as they cannot be solved reliably without examples.}",
                r"\label{tab:simbench-ucr-env-context-stacked}",
                r"\end{table*}",
                "",
            ]
        )
        stacked_output = OUTPUT_DIR / "simbench_ucr_env_context_stacked.tex"
        stacked_output = _apply_output_suffix(
            stacked_output, output_suffix, output_extension
        )
        stacked_output.write_text("\n".join(stacked_lines))
        print(f"Wrote LaTeX table to {stacked_output}")

        if simbench_qwen_df is not None and simbench_config is not None:
            qwen_df = simbench_qwen_df.loc[
                simbench_qwen_df["model_name"] == QWEN_MODEL_NAME
            ].copy()
            qwen_df = qwen_df.loc[
                qwen_df["agent_name"].isin(["terminus-ts-xml-think", "opencode"])
            ].copy()
            if qwen_df.empty:
                raise click.ClickException(
                    "No Qwen rows available for SimBench agent comparison table."
                )
            dataset_order = _dataset_order(qwen_df, SIMBENCH_DATASETS)
            if not dataset_order:
                raise click.ClickException(
                    "No SimBench datasets available for Qwen agent comparison table."
                )
            model_order = [
                f"terminus-ts-xml-think::{QWEN_MODEL_NAME}",
                f"opencode::{QWEN_MODEL_NAME}",
            ]
            grouped = (
                qwen_df.groupby(
                    ["shot_level", "agent_name", "dataset", "context_level"],
                    as_index=False,
                )["metric_accuracy"]
                .mean()
            )
            values: dict[tuple[str, str, str, str], float] = {}
            for _, row in grouped.iterrows():
                model_key = f"{row['agent_name']}::{QWEN_MODEL_NAME}"
                values[
                    (row["shot_level"], model_key, row["dataset"], row["context_level"])
                ] = float(row["metric_accuracy"])
            model_labels = {
                f"terminus-ts-xml-think::{QWEN_MODEL_NAME}": "Qwen Coder 480B (Terminus)",
                f"opencode::{QWEN_MODEL_NAME}": "Qwen Coder 480B (Opencode)",
            }
            compare_lines = [
                r"\begin{table*}[t]",
                r"\centering",
                r"\setlength{\tabcolsep}{4pt}",
                "",
            ]
            compare_lines.extend(
                _stacked_block_lines(
                    r"\noindent\textbf{\name\ Synthetic}\par\vspace{4pt}",
                    dataset_order,
                    simbench_config["context_order"],
                    SIMBENCH_STACKED_LABELS,
                    model_order,
                    values,
                    simbench_config["missing"],
                    simbench_config["decimals"],
                    model_labels=model_labels,
                )
            )
            compare_lines.extend(
                [
                    "",
                    r"\caption{Accuracy by environment and context level for Qwen Coder 480B on \name\ Synthetic, comparing Terminus and OpenCode agents.}",
                    r"\label{tab:simbench-qwen-agent-compare}",
                    r"\end{table*}",
                    "",
                ]
            )
            compare_output = OUTPUT_DIR / "simbench_qwen_agent_compare.tex"
            compare_output = _apply_output_suffix(
                compare_output, output_suffix, output_extension
            )
            compare_output.write_text("\n".join(compare_lines))
            print(f"Wrote LaTeX table to {compare_output}")

    if (
        "simbench_cls" in gain_configs
        and "ucr" in gain_configs
        and gain_configs["simbench_cls"]["values"] is not None
        and gain_configs["ucr"]["values"] is not None
    ):
        simbench_gain = gain_configs["simbench_cls"]
        ucr_gain = gain_configs["ucr"]
        gain_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{0.95}",
            "",
        ]
        gain_lines.extend(
            _gain_block_lines(
                r"\noindent\textbf{\name\ Synthetic (Low $\rightarrow$ High)}\par\vspace{4pt}",
                simbench_gain["model_order"],
                simbench_gain["shot_labels"],
                simbench_gain["values"],
                simbench_gain["missing"],
                GAIN_DECIMALS,
            )
        )
        gain_lines.extend(["", r"\vspace{10pt}", ""])
        gain_lines.extend(
            _gain_block_lines(
                r"\noindent\textbf{\name\ Real}\par\vspace{4pt}",
                ucr_gain["model_order"],
                ucr_gain["shot_labels"],
                ucr_gain["values"],
                ucr_gain["missing"],
                GAIN_DECIMALS,
            )
        )
        gain_lines.extend(
            [
                "",
                r"\caption{Average context gain across datasets by shot setting.}",
                r"\label{tab:simbench-ucr-context-gain}",
                r"\end{table}",
                "",
            ]
        )
        gain_output = OUTPUT_DIR / "simbench_ucr_context_gain.tex"
        gain_output = _apply_output_suffix(
            gain_output, output_suffix, output_extension
        )
        gain_output.write_text("\n".join(gain_lines))
        print(f"Wrote LaTeX table to {gain_output}")

    if (
        "simbench_cls" in gain_configs
        and "ucr" in gain_configs
        and gain_configs["simbench_cls"]["values"] is not None
        and gain_configs["ucr"]["values"] is not None
    ):
        simbench_gain = gain_configs["simbench_cls"]
        ucr_gain = gain_configs["ucr"]
        shot_gain_lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{0.95}",
            "",
        ]
        shot_gain_lines.extend(
            _shot_gain_block_lines(
                r"\noindent\textbf{\name\ Synthetic (0 $\rightarrow$ 1, 0 $\rightarrow$ 3)}\par\vspace{4pt}",
                simbench_gain["model_order"],
                {"gain_01": r"$0 \rightarrow 1$", "gain_03": r"$0 \rightarrow 3$"},
                ["gain_01", "gain_03"],
                simbench_gain["shot_gain_values"],
                simbench_gain["missing"],
                GAIN_DECIMALS,
            )
        )
        shot_gain_lines.extend(["", r"\vspace{10pt}", ""])
        shot_gain_lines.extend(
            _shot_gain_block_lines(
                r"\noindent\textbf{\name\ Real (0 $\rightarrow$ 1, 0 $\rightarrow$ 3)}\par\vspace{4pt}",
                ucr_gain["model_order"],
                {"gain_01": r"$0 \rightarrow 1$", "gain_03": r"$0 \rightarrow 3$"},
                ["gain_01", "gain_03"],
                ucr_gain["shot_gain_values"],
                ucr_gain["missing"],
                GAIN_DECIMALS,
            )
        )
        shot_gain_lines.extend(
            [
                "",
                r"\caption{Average shot gain across datasets by shot transition.}",
                r"\label{tab:simbench-ucr-shot-gain}",
                r"\end{table}",
                "",
            ]
        )
        shot_gain_output = OUTPUT_DIR / "simbench_ucr_shot_gain.tex"
        shot_gain_output = _apply_output_suffix(
            shot_gain_output, output_suffix, output_extension
        )
        shot_gain_output.write_text("\n".join(shot_gain_lines))
        print(f"Wrote LaTeX table to {shot_gain_output}")

    if "simbench_cls" in gain_configs and "ucr" in gain_configs:
        simbench_gain = gain_configs["simbench_cls"]
        ucr_gain = gain_configs["ucr"]
        gain_labels = {"gain_01": r"$0 \rightarrow 1$", "gain_03": r"$0 \rightarrow 3$"}
        gain_keys = ["gain_01", "gain_03"]
        combined_lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\footnotesize",
            r"\renewcommand{\arraystretch}{0.95}",
            "",
            r"\begin{tabular}{@{}p{0.50\textwidth}p{0.50\textwidth}@{}}",
            "",
            r"% ---------- LEFT ----------",
            r"\begin{minipage}[t]{\linewidth}",
            r"\setlength{\tabcolsep}{3pt}",
            "",
            r"\begin{center}\textbf{\name\ Synthetic (Low $\rightarrow$ High)}\end{center}\vspace{4pt}",
        ]
        combined_lines.extend(
            _gain_tabular_lines(
                simbench_gain["model_order"],
                simbench_gain["shot_labels"],
                simbench_gain["values"],
                simbench_gain["missing"],
                GAIN_DECIMALS,
                r"\linewidth",
            )
        )
        combined_lines.extend(
            [
                "",
                r"\vspace{10pt}",
                "",
                r"\begin{center}\textbf{\name\ Real}\end{center}\vspace{4pt}",
            ]
        )
        combined_lines.extend(
            _gain_tabular_lines(
                ucr_gain["model_order"],
                ucr_gain["shot_labels"],
                ucr_gain["values"],
                ucr_gain["missing"],
                GAIN_DECIMALS,
                r"\linewidth",
            )
        )
        combined_lines.extend(
            [
                r"\end{minipage}",
                "",
                r"&",
                r"% ---------- RIGHT ----------",
                r"\begin{minipage}[t]{\linewidth}",
                r"\setlength{\tabcolsep}{3pt}",
                "",
                r"\begin{center}\textbf{\name\ Synthetic (0 $\rightarrow$ 1, 0 $\rightarrow$ 3)}\end{center}\vspace{4pt}",
            ]
        )
        combined_lines.extend(
            _shot_gain_tabular_lines(
                simbench_gain["model_order"],
                gain_labels,
                gain_keys,
                simbench_gain["shot_gain_values"],
                simbench_gain["missing"],
                GAIN_DECIMALS,
                r"\linewidth",
            )
        )
        combined_lines.extend(
            [
                "",
                r"\vspace{10pt}",
                "",
                r"\begin{center}\textbf{\name\ Real (0 $\rightarrow$ 1, 0 $\rightarrow$ 3)}\end{center}\vspace{4pt}",
            ]
        )
        combined_lines.extend(
            _shot_gain_tabular_lines(
                ucr_gain["model_order"],
                gain_labels,
                gain_keys,
                ucr_gain["shot_gain_values"],
                ucr_gain["missing"],
                GAIN_DECIMALS,
                r"\linewidth",
            )
        )
        combined_lines.extend(
            [
                r"\end{minipage}",
                "",
                r"\end{tabular}",
                "",
                r"\caption{Context help: average context gain (left) and average shot gain (right).}",
                r"\label{tab:simbench-context-help-combined}",
                r"\end{table*}",
                "",
            ]
        )
        combined_output = OUTPUT_DIR / "simbench_context_help_combined.tex"
        combined_output = _apply_output_suffix(
            combined_output, output_suffix, output_extension
        )
        combined_output.write_text("\n".join(combined_lines))
        print(f"Wrote LaTeX table to {combined_output}")


@click.command()
@click.option(
    "--results-root",
    type=click.Path(path_type=Path, exists=True),
    default=Path("results/report_numerical_metrics"),
    show_default=True,
    help="Root directory with report_numerical_metrics outputs.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional output .tex path override.",
)
@click.option(
    "--group-by-setting/--no-group-by-setting",
    default=False,
    show_default=True,
    help="Render a combined table grouped by shot setting.",
)
def main(
    results_root: Path,
    output_path: Path | None,
    group_by_setting: bool,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _render_tables(
        results_root,
        output_path,
        group_by_setting,
        "metric_accuracy",
        "",
        ".tex",
    )
    _render_tables(
        results_root,
        output_path,
        group_by_setting,
        "metric_is_correct_within_time",
        WITHIN_TIME_SUFFIX,
        WITHIN_TIME_EXTENSION,
    )
    _prompt_gain_zero_shot_table()


if __name__ == "__main__":
    main()
