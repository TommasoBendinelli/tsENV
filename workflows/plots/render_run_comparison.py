#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from workflows.plots.render_bookmarked_plots import (
    _append_units,
    _get_time_column,
    _normalize_dataframe,
    _normalize_label_key,
    _read_dataframe,
    _safe_name,
)


DEFAULT_TSENV_QUESTIONS = Path("tsENV_questions")
DEFAULT_MODELS_ROOT = Path("models/simulink")
DEFAULT_OUTPUT_DIR = Path("workflows/plots/run_comparisons")


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _find_baseline(specs: dict, baseline_parameters_hash: str) -> Tuple[str, dict]:
    for baseline_uuid, entry in specs.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("baseline_parameters_hash") == baseline_parameters_hash:
            return baseline_uuid, entry
    raise KeyError(
        f"No baseline entry with baseline_parameters_hash={baseline_parameters_hash}"
    )


def _child_parameter_names(child: dict) -> List[str]:
    params = child.get("parameters") or {}
    return [k for k, v in params.items() if v is not None]


def _select_children(
    baseline_entry: dict,
    parameter_hashes: Optional[List[str]],
    parameter_names: Optional[List[str]],
) -> List[Tuple[str, dict]]:
    children = baseline_entry.get("children") or {}
    wanted_hashes = set(parameter_hashes) if parameter_hashes else None
    wanted_names = set(parameter_names) if parameter_names else None
    selected: List[Tuple[str, dict]] = []
    matched_hashes: set = set()
    matched_names: set = set()
    for child_uuid, child in children.items():
        if not isinstance(child, dict):
            continue
        ph = child.get("parameter_hash")
        if ph is None:
            continue
        names = set(_child_parameter_names(child))
        hash_hit = wanted_hashes is not None and ph in wanted_hashes
        name_hit = wanted_names is not None and bool(names & wanted_names)
        if wanted_hashes is None and wanted_names is None:
            selected.append((child_uuid, child))
            continue
        if hash_hit or name_hit:
            selected.append((child_uuid, child))
            if hash_hit:
                matched_hashes.add(ph)
            if name_hit:
                matched_names.update(names & wanted_names)
    if wanted_hashes:
        missing = wanted_hashes - matched_hashes
        if missing:
            raise KeyError(
                f"No child with parameter_hash in {sorted(missing)} under selected baseline"
            )
    if wanted_names:
        missing = wanted_names - matched_names
        if missing:
            available = sorted({n for _, c in children.items() if isinstance(c, dict)
                                for n in _child_parameter_names(c)
                                if c.get("parameter_hash") is not None})
            raise KeyError(
                f"No child with parameter name in {sorted(missing)} under selected baseline. "
                f"Available: {available}"
            )
    return selected


def _parquet_path(dataframes_dir: Path, uuid: str) -> Path:
    return dataframes_dir / f"{uuid}.parquet"


def _load_noise_adder(model_dir: Path, model: str) -> ModuleType:
    path = model_dir / "noise_adder.py"
    if not path.exists():
        raise FileNotFoundError(f"No noise_adder.py at {path}")
    spec = importlib.util.spec_from_file_location(
        f"_noise_adder_{model}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load noise_adder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "add_noise"):
        raise AttributeError(f"{path} does not expose add_noise()")
    return module


def _load_signals(
    df,
    requested_signals: Optional[List[str]],
) -> Tuple[Optional[str], List[str]]:
    time_col = _get_time_column(df)
    candidates = [c for c in df.columns if c != time_col]
    if requested_signals:
        candidates = [c for c in candidates if c in requested_signals]
    import pandas as pd
    candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return time_col, candidates


def _intervention_class_label(
    child: dict,
    record: Optional[Dict[str, dict]],
    child_uuid: str,
) -> str:
    base = None
    if record:
        entry = record.get(child_uuid)
        if isinstance(entry, dict):
            base = entry.get("class_agent_facing_name") or entry.get("class_internal")
    if not base:
        keys = _child_parameter_names(child)
        base = ", ".join(keys) if keys else "intervention"
    return str(base)


def _render_overlay(
    baseline_df,
    interventions: List[dict],
    signal_cols: List[str],
    output_base: Path,
) -> None:
    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.0,
        rc={
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        },
    )
    if len(signal_cols) > 1:
        fig_height = max(2.6, 1.1 * len(signal_cols) + 1.0) * 1.5
        fig, axes = plt.subplots(
            nrows=len(signal_cols),
            ncols=1,
            sharex=True,
            figsize=(9.4, fig_height),
        )
        axes_list = list(axes)
    else:
        fig, ax = plt.subplots(figsize=(9.4, 5.1))
        axes_list = [ax]

    baseline_color = "#1f1f1f"
    palette = sns.color_palette("bright", n_colors=max(3, len(interventions)))
    for i, item in enumerate(interventions):
        item["color"] = palette[i % len(palette)]

    baseline_time = _get_time_column(baseline_df)
    unique_intervention_times = sorted({item["intervention_time"] for item in interventions})
    baseline_split = unique_intervention_times[0] if unique_intervention_times else None

    for idx, col in enumerate(signal_cols):
        ax = axes_list[idx]

        if baseline_time and col in baseline_df.columns:
            x = baseline_df[baseline_time]
            y = baseline_df[col]
            base_kwargs = dict(
                color=baseline_color,
                linestyle="None",
                marker=".",
                markeredgewidth=0,
                zorder=4,
            )
            if baseline_split is None:
                ax.plot(x, y, alpha=1.0, markersize=3.0, **base_kwargs)
            else:
                pre_mask = x <= baseline_split
                post_mask = x >= baseline_split
                ax.plot(
                    x[pre_mask], y[pre_mask],
                    alpha=1.0, markersize=3.0, **base_kwargs,
                )
                ax.plot(
                    x[post_mask], y[post_mask],
                    alpha=0.55, markersize=2.5, **base_kwargs,
                )

        for item in interventions:
            df = item["df"]
            t_col = _get_time_column(df)
            if not t_col or col not in df.columns:
                continue
            x = df[t_col]
            post_mask = x >= item["intervention_time"]
            ax.plot(
                x[post_mask],
                df[col][post_mask],
                color=item["color"],
                linestyle="None",
                marker=".",
                markersize=5.0,
                markeredgewidth=0,
                alpha=1.0,
                zorder=5,
            )

        label = col
        normalized = _normalize_label_key(col)
        if normalized == "position":
            label = "Position (m)"
        elif normalized == "velocity":
            label = "Velocity (m/s)"
        ax.set_ylabel(_append_units(label), fontsize=13)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
        for t in unique_intervention_times:
            ax.axvline(
                t,
                color="#8f8f8f",
                linestyle="--",
                linewidth=0.9,
                alpha=0.6,
                zorder=1,
            )
        if ax.get_legend():
            ax.get_legend().remove()
        ax.margins(x=0)

    for ax in axes_list[:-1]:
        ax.set_xlabel("")
    axes_list[-1].set_xlabel("Time (s)", fontsize=13)
    if len(axes_list) > 1:
        fig.align_ylabels(axes_list)
        fig.subplots_adjust(hspace=0.1)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render baseline-vs-intervention plots for a tsENV simulation run.",
    )
    parser.add_argument("--model", required=True, help="e.g. BallDrop")
    parser.add_argument("--baseline-parameters-hash", required=True)
    parser.add_argument(
        "--parameter-hash",
        default=None,
        help="Comma-separated parameter_hash values to include.",
    )
    parser.add_argument(
        "--parameter",
        default=None,
        help=(
            "Comma-separated parameter names to include (e.g. drag_coeff,gravity). "
            "Matches every child whose 'parameters' dict touches that key. "
            "If both --parameter and --parameter-hash are omitted, all children render."
        ),
    )
    parser.add_argument("--tsenv-questions", type=Path, default=DEFAULT_TSENV_QUESTIONS)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--signals",
        default=None,
        help="Comma-separated signal names to plot (default: all numeric).",
    )
    parser.add_argument(
        "--separate-signals",
        action="store_true",
        help="Render one PDF per signal instead of stacking them as subplots.",
    )
    parser.add_argument(
        "--noise",
        choices=["low", "high"],
        default=None,
        help=(
            "Add noise to baseline + interventions before plotting, using the "
            "model's noise_adder.py. Default: no noise."
        ),
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Seed for the noise RNG (default 0).",
    )
    args = parser.parse_args()

    model_dir = args.models_root / args.model
    specs_path = model_dir / "model_run_specs.json"
    dataframes_dir = args.tsenv_questions / args.model / "dataframes"
    record_path = args.tsenv_questions / args.model / "model_record.json"

    specs = _load_json(specs_path)
    record = _load_json(record_path) if record_path.exists() else None

    baseline_uuid, baseline_entry = _find_baseline(specs, args.baseline_parameters_hash)
    parameter_hashes = (
        [h.strip() for h in args.parameter_hash.split(",") if h.strip()]
        if args.parameter_hash
        else None
    )
    parameter_names = (
        [n.strip() for n in args.parameter.split(",") if n.strip()]
        if args.parameter
        else None
    )
    children = _select_children(baseline_entry, parameter_hashes, parameter_names)
    if not children:
        print("No children to render.")
        return 1

    requested_signals = (
        [s.strip() for s in args.signals.split(",") if s.strip()]
        if args.signals
        else None
    )

    baseline_clean = _normalize_dataframe(_read_dataframe(_parquet_path(dataframes_dir, baseline_uuid)))

    noise_adder = _load_noise_adder(model_dir, args.model) if args.noise else None

    if noise_adder:
        baseline_df, _ = noise_adder.add_noise(
            baseline_clean, baseline_clean, seed=args.noise_seed, noise_level=args.noise
        )
    else:
        baseline_df = baseline_clean

    interventions: List[dict] = []
    signal_cols: List[str] = []
    for child_uuid, child in children:
        intervention_path = _parquet_path(dataframes_dir, child_uuid)
        if not intervention_path.exists():
            print(f"Missing intervention parquet for {child_uuid}")
            continue
        intervention_df = _normalize_dataframe(_read_dataframe(intervention_path))
        if noise_adder:
            intervention_df, _ = noise_adder.add_noise(
                intervention_df, baseline_clean,
                seed=args.noise_seed, noise_level=args.noise,
            )
        if not signal_cols:
            _, signal_cols = _load_signals(intervention_df, requested_signals)
        interventions.append({
            "df": intervention_df,
            "intervention_time": float(child.get("intervention_time") or 0.0),
            "label": _intervention_class_label(child, record, child_uuid),
        })

    if not interventions:
        print("No interventions to render.")
        return 1
    if not signal_cols:
        print("No numeric signals to plot.")
        return 1

    if parameter_names:
        suffix = "_".join(_safe_name(n) for n in parameter_names)
    elif parameter_hashes and len(parameter_hashes) == 1:
        suffix = _safe_name(parameter_hashes[0])
    elif parameter_hashes:
        suffix = f"{len(parameter_hashes)}_interventions"
    else:
        suffix = "all_interventions"
    base_dir = args.output_dir / args.model
    base_stem = f"{_safe_name(args.baseline_parameters_hash)}__{suffix}"
    if args.noise:
        base_stem = f"{base_stem}__noise_{args.noise}"

    if args.separate_signals:
        for col in signal_cols:
            output_base = base_dir / f"{base_stem}__{_safe_name(col)}"
            _render_overlay(baseline_df, interventions, [col], output_base)
            print(f"Saved {output_base.with_suffix('.pdf')}")
    else:
        output_base = base_dir / base_stem
        _render_overlay(baseline_df, interventions, signal_cols, output_base)
        print(f"Saved {output_base.with_suffix('.pdf')} ({len(interventions)} interventions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
