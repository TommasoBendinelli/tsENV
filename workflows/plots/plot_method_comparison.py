from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_heatmaps import GROUP_SETTINGS
from plot_utils import BENCHMARKS, load_metrics, repo_root


SHOT_CHOICES = ["zero-shot", "few-shot", "many-shot"]
SHOT_LEVEL_BY_CHOICE = {
    "few-shot": "few_shot",
    "zero-shot": "zero_shot",
    "many-shot": "many_shot",
}
CONTEXT_CHOICES = ["rich", "none"]


def context_level_for_benchmark(bench: str, context_choice: str) -> str:
    context_order = BENCHMARKS[bench]["context_order"]
    if context_choice == "rich":
        return context_order[0]
    if context_choice == "none":
        if "none" not in context_order:
            raise ValueError(f"Benchmark {bench} has no 'none' context")
        return "none"
    raise ValueError(f"Unknown context choice: {context_choice}")


def metric_stats_for(
    df: pd.DataFrame,
    model_name: str,
    context_level: str,
    shot_level: str,
    allow_multiple: bool,
) -> tuple[float, float]:
    model_df = df[df["model_name"] == model_name]
    if "shot_level" in model_df.columns:
        baseline_mask = model_df["context_level"].isna() & model_df["shot_level"].isna()
        match = model_df[
            ((model_df["context_level"] == context_level) & (model_df["shot_level"] == shot_level))
            | baseline_mask
        ]
    else:
        baseline_mask = model_df["context_level"].isna() & model_df["is_few_shot"].isna()
        if shot_level == "many_shot":
            match = model_df[baseline_mask]
        else:
            is_few_shot = shot_level == "few_shot"
            match = model_df[
                ((model_df["context_level"] == context_level) & (model_df["is_few_shot"] == is_few_shot))
                | baseline_mask
            ]
    if match.empty:
        raise ValueError(
            f"No rows for model={model_name}, context={context_level}, shot={shot_level}",
        )
    values = match["metric_value"].astype(float).to_numpy()
    mean_value = float(values.mean())
    if allow_multiple and len(values) > 1:
        std_value = float(values.std(ddof=0))
    else:
        std_value = 0.0
    return mean_value, std_value


def filter_best_baselines(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    for bench, df in data.items():
        if "context_level" not in df.columns or (
            "shot_level" not in df.columns and "is_few_shot" not in df.columns
        ):
            filtered[bench] = df
            continue
        if "shot_level" in df.columns:
            baseline_mask = df["context_level"].isna() & df["shot_level"].isna()
        else:
            baseline_mask = df["context_level"].isna() & df["is_few_shot"].isna()
        if not baseline_mask.any():
            filtered[bench] = df
            continue
        baseline_df = df.loc[baseline_mask]
        baseline_scores = baseline_df.groupby("model_name", dropna=False)[
            "metric_value"
        ].mean()
        baseline_scores = baseline_scores.dropna()
        if baseline_scores.empty:
            filtered[bench] = df.copy()
            continue
        best_score = baseline_scores.max()
        best_models = baseline_scores[baseline_scores == best_score].index.tolist()
        best_model = sorted(best_models)[0]
        keep_mask = ~baseline_mask | (df["model_name"] == best_model)
        trimmed = df.loc[keep_mask].copy()
        if "shot_level" in trimmed.columns:
            baseline_keep_mask = trimmed["context_level"].isna() & trimmed["shot_level"].isna()
        else:
            baseline_keep_mask = trimmed["context_level"].isna() & trimmed["is_few_shot"].isna()
        trimmed.loc[baseline_keep_mask, "model_name"] = "Best Baseline"
        filtered[bench] = trimmed
    return filtered


def benchmarks_for_group(group: str) -> list[str]:
    return [bench for bench, meta in BENCHMARKS.items() if meta["group"] == group]


def benchmarks_with_data(
    data: dict[str, pd.DataFrame],
    group: str,
    context_choice: str,
    shot_level: str,
) -> list[str]:
    benchmarks = []
    for bench, meta in BENCHMARKS.items():
        if meta["group"] != group:
            continue
        context_level = context_level_for_benchmark(bench, context_choice)
        df = data[bench]
        if "shot_level" in df.columns:
            subset = df[
                (df["context_level"] == context_level)
                & (df["shot_level"] == shot_level)
            ]
            baseline_subset = df[
                df["context_level"].isna() & df["shot_level"].isna()
            ]
        else:
            if shot_level == "many_shot":
                subset = df.iloc[0:0]
            else:
                is_few_shot = shot_level == "few_shot"
                subset = df[
                    (df["context_level"] == context_level)
                    & (df["is_few_shot"] == is_few_shot)
                ]
            baseline_subset = df[
                df["context_level"].isna() & df["is_few_shot"].isna()
            ]
        if subset.empty and baseline_subset.empty:
            continue
        benchmarks.append(bench)
    return benchmarks


def resolve_models(
    data: dict[str, pd.DataFrame],
    benchmarks: list[str],
    context_choice: str,
    shot_level: str,
    models: tuple[str, ...],
    allow_multiple: bool,
) -> list[str]:
    def _baseline_models_for_bench(df: pd.DataFrame) -> set[str]:
        if "context_level" not in df.columns or (
            "shot_level" not in df.columns and "is_few_shot" not in df.columns
        ):
            return set()
        if "shot_level" in df.columns:
            baseline_subset = df[
                df["context_level"].isna() & df["shot_level"].isna()
            ]
        else:
            baseline_subset = df[
                df["context_level"].isna() & df["is_few_shot"].isna()
            ]
        return set(baseline_subset["model_name"].dropna())

    if models:
        model_list = list(models)
        for bench in benchmarks:
            context_level = context_level_for_benchmark(bench, context_choice)
            df = data[bench]
            for model_name in model_list:
                metric_stats_for(
                    df,
                    model_name,
                    context_level,
                    shot_level,
                    allow_multiple,
                )
        baseline_models = set()
        for bench in benchmarks:
            baseline_models |= _baseline_models_for_bench(data[bench])
        for model_name in sorted(baseline_models):
            if model_name not in model_list:
                model_list.append(model_name)
        return model_list

    common_models = None
    for bench in benchmarks:
        context_level = context_level_for_benchmark(bench, context_choice)
        df = data[bench]
        if "shot_level" in df.columns:
            subset = df[
                (df["context_level"] == context_level)
                & (df["shot_level"] == shot_level)
            ]
        else:
            if shot_level == "many_shot":
                subset = df.iloc[0:0]
            else:
                is_few_shot = shot_level == "few_shot"
                subset = df[
                    (df["context_level"] == context_level)
                    & (df["is_few_shot"] == is_few_shot)
                ]
        models_in_bench = set(subset["model_name"])
        models_in_bench |= _baseline_models_for_bench(df)
        if not models_in_bench:
            raise ValueError(
                f"No models found for benchmark={bench}, context={context_level}, "
                f"shot={shot_level}",
            )
        if common_models is None:
            common_models = models_in_bench
        else:
            common_models &= models_in_bench

    if not common_models:
        raise ValueError("No overlapping models across benchmarks for the selection")
    return sorted(common_models)


def plot_group(
    group: str,
    benchmarks: list[str],
    models: list[str],
    data: dict[str, pd.DataFrame],
    context_choice: str,
    shot_level: str,
    variant: str,
    output_dir: Path,
    with_variance: bool,
) -> None:
    settings = GROUP_SETTINGS[group]
    n_models = len(models)
    x = np.arange(len(benchmarks))
    width = 0.8 / max(n_models, 1)
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(3.4 * len(benchmarks), 3.8),
        constrained_layout=True,
    )
    for idx, model_name in enumerate(models):
        values = []
        errors = []
        for bench in benchmarks:
            context_level = context_level_for_benchmark(bench, context_choice)
            value, std = metric_stats_for(
                data[bench],
                model_name,
                context_level,
                shot_level,
                with_variance,
            )
            values.append(value)
            if with_variance:
                errors.append(std)
        offset = (idx - (n_models - 1) / 2) * width
        bar_kwargs = {
            "x": x + offset,
            "height": values,
            "width": width,
            "label": model_name,
            "color": colors[idx % len(colors)],
        }
        if with_variance:
            bar_kwargs["yerr"] = errors
            bar_kwargs["capsize"] = 3
        ax.bar(
            **bar_kwargs,
        )

    shot_label = {
        "few_shot": "Few-shot",
        "zero_shot": "Zero-shot",
        "many_shot": "Many-shot",
    }.get(shot_level, "Shot")
    context_label = "Rich context" if context_choice == "rich" else "No context"
    ax.set_title(f"{group.capitalize()} - {shot_label}, {context_label}")
    ax.set_ylabel(settings["label"])
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARKS[bench]["title"] for bench in benchmarks])
    ax.set_ylim(settings["vmin"], settings["vmax"])
    ax.legend(title="Model", fontsize=8, title_fontsize=9)

    shot_tag = shot_level
    stem = output_dir / f"method_comparison_{group}_{variant}_{context_choice}_{shot_tag}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Root of results/report_numerical_metrics.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for saving comparison plots.",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    help="Model name as it appears in metrics CSVs (repeatable).",
)
@click.option(
    "--variant",
    type=click.Choice(["lite", "ready"], case_sensitive=False),
    default="lite",
    show_default=True,
    help="Metrics variant to load from results.",
)
@click.option(
    "--with-variance",
    is_flag=True,
    help="Add variance bars from repeated runs per benchmark.",
)
@click.option(
    "--skip-missing",
    is_flag=True,
    help="Skip benchmarks that have no metrics CSVs.",
)
def main(
    results_dir: Path | None,
    output_dir: Path | None,
    models: tuple[str, ...],
    variant: str,
    with_variance: bool,
    skip_missing: bool,
) -> None:
    root = repo_root()
    resolved_results = results_dir or root / "results" / "report_numerical_metrics"
    resolved_output = output_dir or root / "overleaf_paper" / "pictures" / "comparison"
    resolved_output.mkdir(parents=True, exist_ok=True)

    data = filter_best_baselines(load_metrics(resolved_results, variant, skip_missing))
    for context_choice in CONTEXT_CHOICES:
        for shot in SHOT_CHOICES:
            shot_level = SHOT_LEVEL_BY_CHOICE[shot]
            for group in GROUP_SETTINGS:
                benchmarks = benchmarks_with_data(
                    data,
                    group,
                    context_choice,
                    shot_level,
                )
                if not benchmarks:
                    click.echo(
                        f"Skipping {group} for {context_choice}/{shot}: no data.",
                    )
                    continue
                model_list = resolve_models(
                    data,
                    benchmarks,
                    context_choice,
                    shot_level,
                    models,
                    with_variance,
                )
                plot_group(
                    group,
                    benchmarks,
                    model_list,
                    data,
                    context_choice,
                    shot_level,
                    variant,
                    resolved_output,
                    with_variance,
                )
    click.echo(f"Saved plots to {resolved_output}")


if __name__ == "__main__":
    main()
