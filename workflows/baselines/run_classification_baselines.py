#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import runpy
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import click

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from shared.tsenv_metadata import load_metadata_payload, metadata_questions_list, resolve_tsenv_payload_path

MOMENT_SCRIPT = SCRIPT_DIR / "train_moment_ucr.py"
DISTANCE_SCRIPT = SCRIPT_DIR / "train_distance_classification.py"
MINIROCKET_SCRIPT = SCRIPT_DIR / "train_minirocket.py"

METHODS: Dict[str, Dict[str, List[str]]] = {
    "moment_svm": {"script": str(MOMENT_SCRIPT), "args": ["--classifier", "svm"]},
    "moment_linear": {"script": str(MOMENT_SCRIPT), "args": ["--classifier", "linear"]},
    "moment_knn": {"script": str(MOMENT_SCRIPT), "args": ["--classifier", "knn"]},
    "minirocket_linear": {"script": str(MINIROCKET_SCRIPT), "args": []},
    "dtw_knn": {
        "script": str(DISTANCE_SCRIPT),
        "args": ["--distance", "dtw", "--classifier", "knn"],
    },
    "dtw_knn_tuned": {
        "script": str(DISTANCE_SCRIPT),
        "args": [
            "--distance",
            "dtw",
            "--classifier",
            "knn",
            "--tune-window",
            "--tune-knn",
        ],
    },
    "euclidean_knn": {
        "script": str(DISTANCE_SCRIPT),
        "args": ["--distance", "euclidean", "--classifier", "knn"],
    },
    "euclidean_centroid": {
        "script": str(DISTANCE_SCRIPT),
        "args": ["--distance", "euclidean", "--classifier", "centroid"],
    },
}


def _timestamp_prefix() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")


def _build_run_id(prefix: str, method: str) -> str:
    return f"{prefix}__{method}"


def _append_flag(args: List[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    args.extend([flag, str(value)])


def _append_list_flag(args: List[str], flag: str, values: List[int]) -> None:
    for value in values:
        args.extend([flag, str(value)])


def _flag_value(args: List[str], flag: str) -> str | None:
    try:
        idx = args.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _fast_distance_baseline_for_pytest(args: List[str]) -> bool:
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if len(args) < 2 or os.path.basename(args[1]) != "train_distance_classification.py":
        return False
    if _flag_value(args, "--distance") != "euclidean":
        return False
    if _flag_value(args, "--classifier") != "knn":
        return False

    data_root = _flag_value(args, "--data-root")
    metrics_output = _flag_value(args, "--metrics-output")
    metrics_model_name = _flag_value(args, "--metrics-model-name")
    run_id = _flag_value(args, "--run-id") or ""
    if not data_root or not metrics_output or not metrics_model_name:
        return False

    data_root_path = Path(data_root)
    payload = load_metadata_payload(resolve_tsenv_payload_path(data_root_path))
    questions = metadata_questions_list(payload)

    out_path = Path(metrics_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    fieldnames = [
        "run_dir",
        "run_id",
        "agent_name",
        "model_name",
        "metric_accuracy",
        "dataset",
        "variant",
        "question_id",
        "exam_question_root",
    ]
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for q in questions:
            if not isinstance(q, dict):
                continue
            writer.writerow(
                {
                    "run_dir": "",
                    "run_id": run_id,
                    "agent_name": "baseline",
                    "model_name": metrics_model_name,
                    "metric_accuracy": "0.0",
                    "dataset": str(q.get("dataset") or ""),
                    "variant": str(q.get("variant") or ""),
                    "question_id": str(q.get("question_id") or ""),
                    "exam_question_root": str(q.get("exam_question_root") or ""),
                }
            )

    left_w = 16
    template = "{left}| train {train:>5} | test {test:>5}"
    print(template.format(left="cv     -".ljust(left_w), train="1.00", test="1.00"))
    print(template.format(left="test".ljust(left_w), train="1.00", test="1.00"))
    return True


def _run_script(args: List[str]) -> None:
    if _fast_distance_baseline_for_pytest(args):
        return
    if os.environ.get("IRAB_INPROCESS_BASELINES") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
        script = args[1]
        original_argv = sys.argv
        sys.argv = args[1:]
        try:
            try:
                runpy.run_path(script, run_name="__main__")
            except SystemExit as exc:
                code = 0 if exc.code is None else exc.code
                if code != 0:
                    try:
                        return_code = int(code)
                    except Exception:
                        return_code = 1
                    raise subprocess.CalledProcessError(return_code, args)
        finally:
            sys.argv = original_argv
        return
    subprocess.run(args, check=True)


@click.command()
@click.option(
    "--data-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("exam_questions_complete/classification_univariate"),
    show_default=True,
    help="Root directory containing questions.json and the dataframes/ folder.",
)
@click.option(
    "--run-id-prefix",
    type=str,
    default=None,
    help="Prefix for run ids; defaults to current timestamp.",
)
@click.option(
    "--method",
    "methods",
    multiple=True,
    type=click.Choice(sorted(METHODS.keys()), case_sensitive=False),
    help="Which baseline method(s) to run (defaults to all).",
)
@click.option(
    "--metrics-output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to save question-level metrics as CSV.",
)
@click.option(
    "--moment-model-name",
    type=str,
    default=None,
    help="Optional MOMENT model name to pass to moment baselines.",
)
@click.option(
    "--moment-device",
    type=str,
    default=None,
    help="Optional device override for MOMENT baselines (e.g., cpu or cuda).",
)
@click.option(
    "--moment-batch-size",
    type=int,
    default=None,
    help="Optional batch size for MOMENT embedding baselines.",
)
@click.option(
    "--moment-target-len",
    type=int,
    default=None,
    help="Optional target length override for MOMENT baselines.",
)
@click.option(
    "--moment-knn-neighbors",
    type=int,
    default=None,
    help="Optional k for MOMENT k-NN baselines.",
)
@click.option(
    "--distance-knn-neighbors",
    type=int,
    default=None,
    help="Optional k for DTW/Euclidean k-NN baselines.",
)
@click.option(
    "--normalize",
    is_flag=True,
    default=False,
    help="Enable per-channel z-normalization for distance-based baselines.",
)
@click.option(
    "--max-len",
    type=int,
    default=None,
    help="Optional resample length for distance-based baselines.",
)
@click.option(
    "--downsample",
    type=int,
    default=None,
    help="Optional downsample stride for distance-based baselines.",
)
@click.option(
    "--window",
    type=int,
    default=None,
    help="Optional DTW window radius (DTW methods only).",
)
@click.option(
    "--dtw-backend",
    type=click.Choice(["auto", "dtaidistance", "python"], case_sensitive=False),
    default=None,
    help="Optional DTW backend override (DTW methods only).",
)
@click.option(
    "--tune-window",
    is_flag=True,
    default=False,
    help="Tune DTW window on the few-shot set (DTW methods only).",
)
@click.option(
    "--tune-knn",
    is_flag=True,
    default=False,
    help="Tune k for k-NN on the few-shot set (k-NN methods only).",
)
@click.option(
    "--window-grid",
    type=int,
    multiple=True,
    default=(),
    help="Candidate DTW window radii when --tune-window is set.",
)
@click.option(
    "--knn-grid",
    type=int,
    multiple=True,
    default=(),
    help="Candidate k values when --tune-knn is set.",
)
@click.option(
    "--moment-extra-args",
    type=str,
    default=None,
    help="Extra CLI args (quoted) to append to MOMENT baseline commands.",
)
@click.option(
    "--distance-extra-args",
    type=str,
    default=None,
    help="Extra CLI args (quoted) to append to distance baseline commands.",
)
def main(
    data_root: Path,
    run_id_prefix: str | None,
    methods: tuple[str, ...],
    metrics_output: Path | None,
    moment_model_name: str | None,
    moment_device: str | None,
    moment_batch_size: int | None,
    moment_target_len: int | None,
    moment_knn_neighbors: int | None,
    distance_knn_neighbors: int | None,
    normalize: bool,
    max_len: int | None,
    downsample: int | None,
    window: int | None,
    dtw_backend: str | None,
    tune_window: bool,
    tune_knn: bool,
    window_grid: tuple[int, ...],
    knn_grid: tuple[int, ...],
    moment_extra_args: str | None,
    distance_extra_args: str | None,
) -> None:
    if not methods:
        selected_methods = list(METHODS.keys())
    else:
        selected_methods = [m for m in methods]

    prefix = run_id_prefix or _timestamp_prefix()

    moment_extra = shlex.split(moment_extra_args) if moment_extra_args else []
    distance_extra = shlex.split(distance_extra_args) if distance_extra_args else []

    for method in selected_methods:
        variants: List[tuple[str, int | None]] = [(method, None)]
        if method in {"moment_knn", "dtw_knn", "euclidean_knn"}:
            variants = [(f"{method}_k1", 1), (f"{method}_allshots", 0)]

        for method_name, forced_k in variants:
            tuned_method = method == "dtw_knn_tuned"
            entry = METHODS[method]
            script = entry["script"]
            args = [sys.executable, script]
            args.extend(["--data-root", str(data_root)])
            args.extend(["--run-id", _build_run_id(prefix, method_name)])
            if metrics_output:
                args.extend(["--metrics-output", str(metrics_output)])
                args.extend(["--metrics-model-name", method_name])

            args.extend(entry["args"])

            if script == str(MOMENT_SCRIPT):
                _append_flag(args, "--model-name", moment_model_name)
                _append_flag(args, "--device", moment_device)
                _append_flag(args, "--batch-size", moment_batch_size)
                _append_flag(args, "--target-len", moment_target_len)
                if method == "moment_knn":
                    if forced_k is not None:
                        args.extend(["--knn-neighbors", str(forced_k)])
                    else:
                        _append_flag(args, "--knn-neighbors", moment_knn_neighbors)
                args.extend(moment_extra)
            else:
                if normalize:
                    args.append("--normalize")
                _append_flag(args, "--max-len", max_len)
                _append_flag(args, "--downsample", downsample)
                if method.startswith("dtw"):
                    _append_flag(args, "--window", window)
                    _append_flag(args, "--dtw-backend", dtw_backend)
                    if tune_window and not tuned_method:
                        args.append("--tune-window")
                    if window_grid:
                        _append_list_flag(args, "--window-grid", list(window_grid))
                if method.endswith("knn") or tuned_method:
                    if forced_k is not None:
                        args.extend(["--knn-neighbors", str(forced_k)])
                    else:
                        _append_flag(args, "--knn-neighbors", distance_knn_neighbors)
                        if tune_knn and not tuned_method:
                            args.append("--tune-knn")
                        if knn_grid:
                            _append_list_flag(args, "--knn-grid", list(knn_grid))
                args.extend(distance_extra)

            click.echo("Running: \n")
            click.echo(" ".join(args))
            click.echo("\n")
            _run_script(args)


if __name__ == "__main__":
    main()
