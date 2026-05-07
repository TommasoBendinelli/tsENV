#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Set

import click

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from run_anomaly_baselines import METHODS as ANOMALY_METHODS
from run_classification_baselines import METHODS as CLASSIFICATION_METHODS
from shared.benchmark_utils import benchmark_root_from_label
from shared.exam_questions_paths import detect_exam_questions_variant
from shared.tsenv_metadata import load_metadata_payload, metadata_questions_list, resolve_tsenv_payload_path

CLASSIFICATION_RUNNER = SCRIPT_DIR / "run_classification_baselines.py"
ANOMALY_RUNNER = SCRIPT_DIR / "run_anomaly_baselines.py"


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _is_torch_classification_method(method: str) -> bool:
    return method.lower().startswith("moment_")


def _is_torch_anomaly_method(method: str) -> bool:
    return method.lower().startswith("moment_")


def _timestamp_prefix() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d__%H-%M-%S")


def _load_task_types(meta_path: Path) -> Set[str]:
    payload = load_metadata_payload(meta_path)
    questions = metadata_questions_list(payload)
    tasks = {q.get("task") for q in questions if isinstance(q, dict)}
    normalized = {task for task in tasks if isinstance(task, str) and task}
    if normalized:
        return normalized
    if isinstance(payload.get("label_int_mapping"), dict):
        return {"classification"}
    return set()


def _append_methods(args: list[str], methods: Iterable[str]) -> None:
    for method in methods:
        args.extend(["--method", method])


def _variant_from_root(data_root: Path) -> str:
    variant = detect_exam_questions_variant(data_root)
    if variant is None:
        raise click.ClickException(
            f"Unable to determine exam_questions variant from data_root: {data_root}. "
            "Expected a path under exam_questions_<variant>."
        )
    return variant


def _run_baselines(
    runner: Path,
    *,
    data_root: Path,
    run_id_prefix: str,
    methods: Iterable[str],
    metrics_output: Path | None,
    extra_args: str | None,
) -> None:
    args = [
        sys.executable,
        str(runner),
        "--data-root",
        str(data_root),
        "--run-id-prefix",
        run_id_prefix,
    ]
    _append_methods(args, methods)
    if metrics_output is not None:
        args.extend(["--metrics-output", str(metrics_output)])
    if extra_args:
        args.extend(shlex.split(extra_args))

    subprocess.run(args, check=True)


def _run_anomaly_baselines(
    runner: Path,
    *,
    data_root: Path,
    methods: Iterable[str],
    extra_args: str | None,
) -> None:
    args = [
        sys.executable,
        str(runner),
        "--data-root",
        str(data_root),
    ]
    _append_methods(args, methods)
    if extra_args:
        args.extend(shlex.split(extra_args))
    click.echo("Running: " + " ".join(args))
    subprocess.run(args, check=True)


@click.command()
@click.option(
    "--data-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("exam_questions_ready"),
    show_default=True,
    help="Root directory containing exam question folders.",
)
@click.option(
    "--run-id-prefix",
    type=str,
    default=None,
    help="Prefix for run ids; defaults to current timestamp.",
)
@click.option(
    "--only",
    "only_sets",
    multiple=True,
    help="Only run baselines for these dataset folder names.",
)
@click.option(
    "--skip-classification",
    is_flag=True,
    default=False,
    help="Skip classification baselines.",
)
@click.option(
    "--skip-anomaly",
    is_flag=True,
    default=False,
    help="Skip anomaly baselines.",
)
@click.option(
    "--classification-method",
    "classification_methods",
    multiple=True,
    type=click.Choice(sorted(CLASSIFICATION_METHODS.keys()), case_sensitive=False),
    help="Which classification baseline method(s) to run (defaults to all).",
)
@click.option(
    "--anomaly-method",
    "anomaly_methods",
    multiple=True,
    type=click.Choice(sorted(ANOMALY_METHODS.keys()), case_sensitive=False),
    help="Which anomaly baseline method(s) to run (defaults to all).",
)
@click.option(
    "--classification-extra-args",
    type=str,
    default=None,
    help="Extra CLI args (quoted) to append to classification baseline commands.",
)
@click.option(
    "--anomaly-extra-args",
    type=str,
    default=None,
    help="Extra CLI args (quoted) to append to anomaly baseline commands.",
)
@click.option(
    "--skip-torch",
    is_flag=True,
    default=False,
    help=(
        "Skip baselines that require torch (MOMENT classification baselines and "
        "all anomaly baselines). If torch is not installed and no explicit torch "
        "methods are requested, this is enabled automatically."
    ),
)
def main(
    data_root: Path,
    run_id_prefix: str | None,
    only_sets: tuple[str, ...],
    skip_classification: bool,
    skip_anomaly: bool,
    classification_methods: tuple[str, ...],
    anomaly_methods: tuple[str, ...],
    classification_extra_args: str | None,
    anomaly_extra_args: str | None,
    skip_torch: bool,
) -> None:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise click.ClickException(f"Data root not found: {data_root}")

    torch_available = _torch_available()
    if not torch_available and not skip_torch:
        requested_torch_methods = [
            method
            for method in classification_methods
            if _is_torch_classification_method(method)
        ]
        if requested_torch_methods:
            raise click.ClickException(
                "torch is not installed, but torch baselines were requested via "
                f"--classification-method: {', '.join(requested_torch_methods)}"
            )
        requested_torch_anomaly = [
            method for method in anomaly_methods if _is_torch_anomaly_method(method)
        ]
        if requested_torch_anomaly:
            raise click.ClickException(
                "torch is not installed, but torch anomaly baselines were requested via "
                f"--anomaly-method: {', '.join(requested_torch_anomaly)}"
            )
        click.echo("torch not installed; skipping torch baselines.")
        skip_torch = True

    prefix = run_id_prefix or _timestamp_prefix()
    allow_set = set(only_sets) if only_sets else None
    repo_root = Path(__file__).resolve().parents[2]
    exam_question_root_name_only = _variant_from_root(data_root)
    baseline_outputs: dict[str, Path] = {}

    for benchmark_variant_dir in sorted(data_root.iterdir()):
        if not benchmark_variant_dir.is_dir():
            continue
        if (benchmark_variant_dir / ".hidden").exists():
            continue
        if allow_set and benchmark_variant_dir.name not in allow_set:
            continue
        try:
            meta_path = resolve_tsenv_payload_path(benchmark_variant_dir)
        except FileNotFoundError:
            continue

        tasks = _load_task_types(meta_path)
        if not tasks:
            click.echo(f"Skipping {benchmark_variant_dir.name}: no task type found.")
            continue
        if len(tasks) > 1:
            click.echo(
                f"Skipping {benchmark_variant_dir.name}: mixed task types {sorted(tasks)}."
            )
            continue

        task_type = next(iter(tasks))
        run_prefix = f"{prefix}_{benchmark_variant_dir.name}"
        if task_type == "classification":
            if skip_classification:
                continue
            benchmark_root = benchmark_root_from_label(benchmark_variant_dir.name)
            metrics_dir = (
                repo_root
                / "results"
                / "report_numerical_metrics"
                / benchmark_root
                / exam_question_root_name_only
            )
            metrics_output = baseline_outputs.get(benchmark_root)
            if metrics_output is None:
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_output = (
                    metrics_dir / f"metrics_baseline_{prefix}_{benchmark_root}.csv"
                )
                baseline_outputs[benchmark_root] = metrics_output
            methods = classification_methods or tuple(CLASSIFICATION_METHODS.keys())
            if skip_torch:
                methods = tuple(
                    method
                    for method in methods
                    if not _is_torch_classification_method(method)
                )
                if not methods:
                    click.echo(
                        f"Skipping {benchmark_variant_dir.name}: no non-torch classification baselines selected."
                    )
                    continue
            _run_baselines(
                CLASSIFICATION_RUNNER,
                data_root=benchmark_variant_dir,
                run_id_prefix=run_prefix,
                methods=methods,
                metrics_output=metrics_output,
                extra_args=classification_extra_args,
            )
        elif task_type in {"anomaly_localization", "change_point_detection"}:
            if skip_anomaly:
                continue
            methods = anomaly_methods or tuple(ANOMALY_METHODS.keys())
            if skip_torch:
                methods = tuple(
                    method for method in methods if not _is_torch_anomaly_method(method)
                )
                if not methods:
                    click.echo(
                        f"Skipping {benchmark_variant_dir.name}: no non-torch anomaly baselines selected."
                    )
                    continue
            _run_anomaly_baselines(
                ANOMALY_RUNNER,
                data_root=benchmark_variant_dir,
                methods=methods,
                extra_args=anomaly_extra_args,
            )
        else:
            click.echo(
                f"Skipping {benchmark_variant_dir.name}: unsupported task {task_type!r}."
            )


if __name__ == "__main__":
    main()
