from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import BENCHMARKS, load_metrics, repo_root

CONTEXT_LABELS = {
    "none": "None",
    "low": "Low",
    "high": "High",
    "ground_truth": "Ground Truth",
}

SHOT_ORDER = ["zero_shot", "few_shot", "many_shot"]
SHOT_LABELS = {
    "zero_shot": "Zero",
    "few_shot": "Few",
    "many_shot": "Many",
}

GROUP_SETTINGS = {
    "accuracy": {
        "vmin": 0.0,
        "vmax": 100.0,
        "label": "Accuracy (%)",
        "fmt": "{:.0f}",
    },
    "anomaly": {
        "vmin": 0.0,
        "vmax": 0.55,
        "label": "VUS-PR",
        "fmt": "{:.3f}",
    },
}

def model_slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def grid_for_benchmark(
    df: pd.DataFrame,
    model_name: str,
    context_order: list[str],
    with_variance: bool,
) -> np.ndarray:
    grid = np.full((len(context_order), len(SHOT_ORDER)), np.nan, dtype=float)
    std_grid = np.full((len(context_order), len(SHOT_ORDER)), np.nan, dtype=float)
    has_shot_level = "shot_level" in df.columns
    for row_idx, context in enumerate(context_order):
        for col_idx, shot in enumerate(SHOT_ORDER):
            if has_shot_level:
                match = df[
                    (df["model_name"] == model_name)
                    & (df["context_level"] == context)
                    & (df["shot_level"] == shot)
                ]
            else:
                if shot == "many_shot":
                    continue
                is_few_shot = shot == "few_shot"
                match = df[
                    (df["model_name"] == model_name)
                    & (df["context_level"] == context)
                    & (df["is_few_shot"] == is_few_shot)
                ]
            if not match.empty:
                values = match["metric_value"].astype(float).to_numpy()
                grid[row_idx, col_idx] = float(values.mean())
                if with_variance:
                    std_grid[row_idx, col_idx] = float(values.std(ddof=0))
    return grid, std_grid


def plot_group(
    model_name: str,
    group: str,
    benchmarks: list[str],
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    variant: str,
    with_variance: bool,
) -> None:
    settings = GROUP_SETTINGS[group]
    ncols = len(benchmarks)
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.2), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="#f0f0f0")
    last_im = None

    for ax, bench in zip(axes, benchmarks):
        bench_meta = BENCHMARKS[bench]
        grid, std_grid = grid_for_benchmark(
            data[bench],
            model_name,
            bench_meta["context_order"],
            with_variance,
        )
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(
            masked,
            vmin=settings["vmin"],
            vmax=settings["vmax"],
            cmap=cmap,
            aspect="auto",
            origin="upper",
        )
        last_im = im

        ax.set_title(bench_meta["title"])
        ax.set_xticks(np.arange(len(SHOT_ORDER)))
        ax.set_xticklabels([SHOT_LABELS[s] for s in SHOT_ORDER])
        ax.set_yticks(np.arange(len(bench_meta["context_order"])))
        ax.set_yticklabels([CONTEXT_LABELS[c] for c in bench_meta["context_order"]])
        ax.set_xlabel("Shot")
        ax.set_ylabel("Context")

        for (row_idx, col_idx), value in np.ndenumerate(grid):
            if np.isfinite(value):
                text_value = settings["fmt"].format(value)
                if with_variance:
                    std_value = std_grid[row_idx, col_idx]
                    if np.isfinite(std_value):
                        text_value = f"{text_value}\n(+/-{std_value:.3g})"
                ax.text(
                    col_idx,
                    row_idx,
                    text_value,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax.set_xticks(np.arange(-0.5, len(SHOT_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(bench_meta["context_order"]), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(f"{model_name} — {group.capitalize()} Heatmaps", fontsize=11)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.85, label=settings["label"])

    stem = output_dir / f"{variant}_heatmap_{group}_{model_slug(model_name)}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300)
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def benchmarks_for_model(
    data: dict[str, pd.DataFrame],
    model_name: str,
    group: str,
) -> list[str]:
    benchmarks = []
    for bench, meta in BENCHMARKS.items():
        if meta["group"] != group:
            continue
        subset = data[bench][data[bench]["model_name"] == model_name]
        subset = subset[
            subset["context_level"].notna() & subset["is_few_shot"].notna()
        ]
        if subset.empty:
            continue
        benchmarks.append(bench)
    return benchmarks


def discover_models(data: dict[str, pd.DataFrame]) -> list[str]:
    models = set()
    for df in data.values():
        if "context_level" not in df.columns or "is_few_shot" not in df.columns:
            continue
        subset = df[df["context_level"].notna() & df["is_few_shot"].notna()]
        models.update(subset["model_name"].dropna().unique())
    return sorted(models)


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
    help="Directory for saving heatmap PDFs.",
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
    help="Ignored; kept for CLI parity with plot_method_comparison.py.",
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
    resolved_output = (
        output_dir
        or root / "68e3980bb2dcb29a5a67ec17" / "pictures" / "heatmap"
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    data = load_metrics(resolved_results, variant, skip_missing)
    model_list = list(models) or discover_models(data)
    if not model_list:
        raise click.UsageError("No models found in metrics data.")

    for model_name in model_list:
        accuracy_benchmarks = benchmarks_for_model(data, model_name, "accuracy")
        anomaly_benchmarks = benchmarks_for_model(data, model_name, "anomaly")
        if accuracy_benchmarks:
            plot_group(
                model_name,
                "accuracy",
                accuracy_benchmarks,
                data,
                resolved_output,
                variant,
                with_variance,
            )
        else:
            click.echo(f"Skipping accuracy heatmap for {model_name}: no data.")
        if anomaly_benchmarks:
            plot_group(
                model_name,
                "anomaly",
                anomaly_benchmarks,
                data,
                resolved_output,
                variant,
                with_variance,
            )
        else:
            click.echo(f"Skipping anomaly heatmap for {model_name}: no data.")
    click.echo(f"Saved plots to {resolved_output}")


if __name__ == "__main__":
    main()
