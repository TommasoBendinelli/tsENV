from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import click
from tqdm import tqdm

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.interface.distribution_json import load_experiment_config_json  # noqa: E402
from shared.interface.model_record_json import load_model_run_specs_json  # noqa: E402
from shared.interface.similarity_metrics_json import load_similarity_metrics_json  # noqa: E402
from shared.run_artifacts import (  # noqa: E402
    SIMILARITY_METRICS_FILENAME,
    resolve_runs_root,
)
from workflows.metrics import parameter_optimisation  # noqa: E402


_BATCH_MAX_ITER = 5
_BATCH_COARSE_GRID_POINTS = 9
_ELIGIBILITY_FILENAME = SIMILARITY_METRICS_FILENAME
_WEBAPP_BASE_URL = "http://localhost:3001/"
_OPTIMISATION_REPORT_FILENAME = "optimisation_results.json"
_OPTIMISATION_SUMMARY_FILENAME = "optimisation_summary.json"
_NO_PARAMETER_CHANGE = "no_parameter_change"


def _resolve_model_dir(model_id: str) -> Path:
    model_name = str(model_id or "").strip()
    if not model_name or "/" in model_name or "\\" in model_name:
        raise ValueError(f"Expected a model id under models/simulink, got {model_id!r}")
    model_dir = parameter_optimisation._resolve_models_root() / model_name
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    return model_dir


def _now_iso8601_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _webapp_run_url(*, model_id: str, run_id: str) -> str:
    query = urlencode(
        {
            "model": str(model_id or "").strip(),
            "run": str(run_id or "").strip(),
            "compare": "none",
        }
    )
    return f"{_WEBAPP_BASE_URL}?{query}"


def _child_parameter(raw_child: Mapping[str, Any], *, child_id: str) -> str:
    child_parameters = dict(raw_child.get("parameters") or {})
    non_null_parameters = [
        str(key).strip()
        for key, value in child_parameters.items()
        if str(key).strip() and value is not None
    ]
    if len(non_null_parameters) != 1:
        raise ValueError(f"Child run {child_id!r} must define exactly one changed parameter")
    return non_null_parameters[0]


def _observable_signals_for_child(
    *,
    specs: Mapping[str, Any],
    experiment_config: Any,
    baseline_id: str,
    child_id: str,
) -> list[str]:
    raw_parent = specs.get(baseline_id)
    if not isinstance(raw_parent, Mapping):
        raise ValueError(f"Eligible baseline {baseline_id!r} is missing from model_run_specs.json")
    raw_children = raw_parent.get("children")
    if not isinstance(raw_children, Mapping):
        raise ValueError(f"Eligible baseline {baseline_id!r} has no children in model_run_specs.json")
    raw_child = raw_children.get(child_id)
    if not isinstance(raw_child, Mapping):
        raise ValueError(f"Eligible child {child_id!r} is missing from model_run_specs.json")
    parameter = _child_parameter(raw_child, child_id=child_id)
    return parameter_optimisation._observable_signals_for_parameter(
        experiment_config,
        parameter,
    )


def _eligible_children_by_baseline(metrics: Mapping[str, Any]) -> dict[str, list[str]]:
    baselines = metrics.get("baselines")
    if not isinstance(baselines, Mapping):
        raise ValueError("eligibility_metrics.json must contain a baselines object")
    eligible: dict[str, list[str]] = {}
    for raw_baseline_id, raw_summary in baselines.items():
        baseline_id = str(raw_baseline_id).strip().lower()
        if not baseline_id or not isinstance(raw_summary, Mapping):
            continue
        if raw_summary.get("family_eligible") is not True:
            continue
        raw_children = raw_summary.get("children")
        if not isinstance(raw_children, Mapping):
            continue
        child_ids = sorted(
            str(raw_child_id).strip().lower()
            for raw_child_id, raw_child in raw_children.items()
            if str(raw_child_id).strip()
            and isinstance(raw_child, Mapping)
            and raw_child.get("eligible") is True
        )
        if child_ids:
            eligible[baseline_id] = child_ids
    if not eligible:
        raise ValueError("No eligible children found in eligible families")
    return dict(sorted(eligible.items()))


def _normalise_model_ids(models: Sequence[str]) -> tuple[str, ...]:
    normalised: list[str] = []
    seen: set[str] = set()
    for raw_model in models:
        model_id = str(raw_model or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        normalised.append(model_id)
    if not normalised:
        raise ValueError("Pass at least one --model.")
    return tuple(normalised)


def _normalise_model_args(
    option_models: Sequence[str],
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    return _normalise_model_ids([*option_models, *extra_args])


def _resolve_tsenv_questions_root() -> Path:
    cwd_root = Path(os.getcwd()).resolve() / "tsENV_questions"
    if cwd_root.exists():
        return cwd_root
    return root_dir / "tsENV_questions"


def _read_json_object(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _question_intervention_labels(questions_path: Path) -> Mapping[str, str]:
    payload = _read_json_object(questions_path)
    ground_truth = payload.get("ground_truth_information")
    if not isinstance(ground_truth, Mapping):
        raise ValueError(f"{questions_path} is missing ground_truth_information")
    interventions = ground_truth.get("interventions")
    if interventions is None:
        interventions = ground_truth.get("intervention")
    if not isinstance(interventions, Mapping):
        raise ValueError(
            f"{questions_path} must define ground_truth_information.interventions"
        )

    labels: dict[str, str] = {}
    for raw_child_id, raw_info in interventions.items():
        child_id = str(raw_child_id or "").strip().lower()
        if not child_id or not isinstance(raw_info, Mapping):
            continue
        labels[child_id] = str(raw_info.get("changed_parameter") or "").strip()
    return labels


def _tsenv_question_child_ids(model_id: str) -> list[str]:
    questions_root = _resolve_tsenv_questions_root()
    model_questions_root = questions_root / model_id
    sample_manifest_path = model_questions_root / "sample_manifest.json"
    questions_path = model_questions_root / "questions.json"
    sample_manifest = _read_json_object(sample_manifest_path)
    changed_parameter_by_child_id = _question_intervention_labels(questions_path)

    child_ids: list[str] = []
    seen: set[str] = set()
    for _shot_slug, raw_rows in sample_manifest.items():
        rows = raw_rows if isinstance(raw_rows, list) else []
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                continue
            for field_name in ("train_samples", "test_samples"):
                raw_samples = raw_row.get(field_name)
                if not isinstance(raw_samples, list):
                    continue
                for raw_child_id in raw_samples:
                    child_id = str(raw_child_id or "").strip().lower()
                    if not child_id or child_id in seen:
                        continue
                    changed_parameter = changed_parameter_by_child_id.get(child_id, "")
                    if not changed_parameter or changed_parameter == _NO_PARAMETER_CHANGE:
                        continue
                    seen.add(child_id)
                    child_ids.append(child_id)
    if not child_ids:
        raise ValueError(f"No changed train/test sample runs found in {sample_manifest_path}")
    return child_ids


def _children_by_baseline_from_child_ids(
    *,
    specs: Mapping[str, Any],
    child_ids: Sequence[str],
) -> dict[str, list[str]]:
    child_to_baseline: dict[str, str] = {}
    for raw_baseline_id, raw_parent in specs.items():
        baseline_id = str(raw_baseline_id or "").strip().lower()
        if not baseline_id or not isinstance(raw_parent, Mapping):
            continue
        raw_children = raw_parent.get("children")
        if not isinstance(raw_children, Mapping):
            continue
        for raw_child_id in raw_children:
            child_id = str(raw_child_id or "").strip().lower()
            if child_id:
                child_to_baseline[child_id] = baseline_id

    grouped: dict[str, list[str]] = {}
    missing: list[str] = []
    for raw_child_id in child_ids:
        child_id = str(raw_child_id or "").strip().lower()
        baseline_id = child_to_baseline.get(child_id)
        if baseline_id is None:
            missing.append(child_id)
            continue
        grouped.setdefault(baseline_id, []).append(child_id)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"tsENV question run ids are missing from model_run_specs.json: {missing_text}")
    return dict(sorted(grouped.items()))


def _optimisation_report_path(*, runs_root: Path, child_id: str) -> Path:
    return runs_root / child_id / _OPTIMISATION_REPORT_FILENAME


def _optimisation_report_is_fresh(*, runs_root: Path, child_id: str) -> bool:
    report_path = _optimisation_report_path(runs_root=runs_root, child_id=child_id)
    data_path = runs_root / child_id / "data.parquet"
    if not report_path.exists() or not data_path.exists():
        return False
    return report_path.stat().st_mtime > data_path.stat().st_mtime


def _resolve_eligibility_path(runs_root: Path) -> Path:
    return runs_root / _ELIGIBILITY_FILENAME


def _build_oracle_summary_template() -> dict[str, Any]:
    return {"timestamp": _now_iso8601_utc()}


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in (float("inf"), float("-inf")) else None


def _report_from_result_or_path(
    *,
    output: Path,
    result: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    if isinstance(result, Mapping):
        documented_report = result.get("documented_report")
        if isinstance(documented_report, Mapping):
            return documented_report
        report = result.get("report")
        if isinstance(report, Mapping):
            return report
    if output.exists():
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return payload
    return None


def _child_summary_from_report(
    *,
    model_id: str,
    child_id: str,
    report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "url": _webapp_run_url(model_id=model_id, run_id=child_id),
        "parameters": {},
    }
    if not isinstance(report, Mapping):
        return summary

    if isinstance(report.get("url"), str) and str(report.get("url")).strip():
        summary["url"] = str(report["url"])
    ground_truth = report.get("ground_truth_intervention")
    ground_truth_parameter = ""
    ground_truth_value: Any = None
    if isinstance(ground_truth, Mapping):
        ground_truth_parameter = str(ground_truth.get("parameter") or "").strip()
        ground_truth_value = ground_truth.get("value")
    else:
        ground_truth_parameter = str(
            report.get("parameter_ground_truth", report.get("ground_truth_parameter", ""))
            or ""
        ).strip()
        ground_truth_value = report.get(
            "parameter_ground_truth_value",
            report.get("ground_truth_value"),
        )

    parameters: dict[str, dict[str, Any]] = {}
    for raw_candidate in report.get("candidates", []):
        if not isinstance(raw_candidate, Mapping):
            continue
        parameter = str(raw_candidate.get("parameter") or "").strip()
        if not parameter:
            continue
        is_ground_truth = parameter == ground_truth_parameter
        entry: dict[str, Any] = {
            "loss": _finite_or_none(
                raw_candidate.get("optimisation_loss", raw_candidate.get("loss"))
            ),
            "value_found": _finite_or_none(
                raw_candidate.get("param_opt", raw_candidate.get("p_opt"))
            ),
            "is_ground_truth": is_ground_truth,
        }
        if is_ground_truth:
            entry["ground_truth_value"] = ground_truth_value
        parameters[parameter] = entry

    if ground_truth_parameter and ground_truth_parameter not in parameters:
        parameters[ground_truth_parameter] = {
            "loss": None,
            "value_found": None,
            "ground_truth_value": ground_truth_value,
            "is_ground_truth": True,
        }
    summary["parameters"] = dict(sorted(parameters.items()))
    return summary


def _set_child_summary(
    summary: dict[str, Any],
    *,
    model_id: str,
    child_id: str,
    output: Path,
    result: Mapping[str, Any] | None = None,
) -> None:
    report = _report_from_result_or_path(output=output, result=result)
    summary[str(child_id).strip().lower()] = _child_summary_from_report(
        model_id=model_id,
        child_id=child_id,
        report=report,
    )


def run_over_model(
    *,
    model_id: str,
    overwrite: bool = False,
    only_tsenv_questions: bool = False,
    max_iter: int | None = _BATCH_MAX_ITER,
    tol_x: float | None = None,
    coarse_grid_points: int = _BATCH_COARSE_GRID_POINTS,
    countinous_only: bool = False,
    matlab_workers: int | None = None,
    runner=None,
) -> dict[str, Any]:
    if runner is None:
        runner = parameter_optimisation.run_for_model
    use_shared_matlab = runner is parameter_optimisation.run_for_model
    model_dir = _resolve_model_dir(model_id)
    runs_root = resolve_runs_root(model_dir)
    specs = load_model_run_specs_json(
        model_dir / "model_run_specs.json",
        enforce_baseline_pair_diversity=False,
    )
    experiment_config = load_experiment_config_json(model_dir / "experiment_config.json")
    metrics = load_similarity_metrics_json(_resolve_eligibility_path(runs_root))
    eligible = (
        _children_by_baseline_from_child_ids(
            specs=specs,
            child_ids=_tsenv_question_child_ids(model_id),
        )
        if only_tsenv_questions
        else _eligible_children_by_baseline(metrics)
    )
    summary = _build_oracle_summary_template()

    successes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    progress_total = sum(len(child_ids) for child_ids in eligible.values())
    progress = tqdm(
        total=progress_total,
        desc="parameter optimisation",
        unit="run",
    )

    matlab_session_cm = None
    shared_matlab_engine = None

    def get_shared_matlab_engine():
        nonlocal matlab_session_cm, shared_matlab_engine
        if not use_shared_matlab:
            return None
        if shared_matlab_engine is None:
            matlab_session_cm = parameter_optimisation.build_metadata.matlab_session(Path.cwd())
            shared_matlab_engine = matlab_session_cm.__enter__()
            if shared_matlab_engine is None:
                raise RuntimeError("MATLAB session did not start")
            resolved_workers = parameter_optimisation._resolve_matlab_workers(matlab_workers)
            parameter_optimisation._ensure_matlab_parallel_pool(
                shared_matlab_engine,
                resolved_workers,
            )
        return shared_matlab_engine

    try:
        for baseline_id, child_ids in eligible.items():
            for child_id in child_ids:
                try:
                    signals = _observable_signals_for_child(
                        specs=specs,
                        experiment_config=experiment_config,
                        baseline_id=baseline_id,
                        child_id=child_id,
                    )
                except Exception as exc:  # noqa: BLE001 - collect all batch failures
                    failures.append(
                        {
                            "baseline_id": baseline_id,
                            "run_id": child_id,
                            "error": str(exc),
                        }
                    )
                    progress.update(1)
                    continue
                output = _optimisation_report_path(
                    runs_root=runs_root,
                    child_id=child_id,
                )
                if not overwrite and _optimisation_report_is_fresh(
                    runs_root=runs_root,
                    child_id=child_id,
                ):
                    _set_child_summary(
                        summary,
                        model_id=model_id,
                        child_id=child_id,
                        output=output,
                    )
                    skipped.append(
                        {
                            "baseline_id": baseline_id,
                            "run_id": child_id,
                            "path": str(output),
                        }
                    )
                    progress.update(1)
                    continue
                try:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    runner_kwargs: dict[str, Any] = {
                        "model_id": model_id,
                        "run_id": child_id,
                        "max_iter": max_iter,
                        "coarse_grid_points": coarse_grid_points,
                        "signals": signals,
                        "output": output,
                        "countinous_only": countinous_only,
                        "matlab_workers": matlab_workers,
                    }
                    if tol_x is not None:
                        runner_kwargs["tol_x"] = tol_x
                    if use_shared_matlab:
                        runner_kwargs["matlab_engine"] = get_shared_matlab_engine()
                    result = runner(**runner_kwargs)
                    os.utime(output.parent, None)
                    _set_child_summary(
                        summary,
                        model_id=model_id,
                        child_id=child_id,
                        output=output,
                        result=result,
                    )
                    successes.append(
                        {
                            "baseline_id": baseline_id,
                            "run_id": child_id,
                            "path": result["path"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - keep remaining children running
                    failures.append(
                        {
                            "baseline_id": baseline_id,
                            "run_id": child_id,
                            "path": str(output),
                            "error": str(exc),
                        }
                    )
                finally:
                    progress.update(1)
    finally:
        progress.close()
        if matlab_session_cm is not None:
            try:
                matlab_session_cm.__exit__(None, None, None)
            except Exception:
                pass

    summary_path = runs_root / _OPTIMISATION_SUMMARY_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": not failures,
        "model": model_id,
        "successes": successes,
        "skipped": skipped,
        "failures": failures,
        "summary_path": str(summary_path),
        "summary": summary,
    }


@click.command()
@click.option(
    "--model",
    "models",
    required=True,
    multiple=True,
    type=str,
    help="Model id under models/simulink/; repeat for multiple models.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Run even when the optimisation report is newer than the child data.parquet.",
)
@click.option(
    "--only-tsENV-questions",
    "--only-tsenv-questions",
    "only_tsenv_questions",
    is_flag=True,
    default=False,
    help="Run only changed intervention runs listed in tsENV_questions/<MODEL>/questions.json.",
)
@click.option(
    "--max-iter",
    type=click.IntRange(min=1),
    default=_BATCH_MAX_ITER,
    show_default=True,
    help="MATLAB fminbnd MaxIter for each candidate refinement.",
)
@click.option(
    "--tol-x",
    type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help="Optional MATLAB fminbnd TolX for each candidate refinement.",
)
@click.option(
    "--coarse-grid-points",
    type=click.IntRange(min=1),
    default=_BATCH_COARSE_GRID_POINTS,
    show_default=True,
    help="Number of parsim coarse-grid points per candidate before fminbnd.",
)
@click.option(
    "--matlab-workers",
    type=click.IntRange(min=0),
    default=None,
    help=(
        "MATLAB parallel pool workers. Defaults to TSENV_MATLAB_WORKERS or "
        f"{parameter_optimisation._DEFAULT_MATLAB_WORKERS}; use 0 to disable explicit pool startup."
    ),
)
@click.option(
    "--countinous-only",
    "--continuous-only",
    "countinous_only",
    is_flag=True,
    default=False,
    help="Use only continuous observable signals for coarse grid and fminbnd.",
)
@click.argument("model_args", nargs=-1)
def cli(
    models: tuple[str, ...],
    model_args: tuple[str, ...],
    overwrite: bool,
    only_tsenv_questions: bool,
    max_iter: int,
    tol_x: float | None,
    coarse_grid_points: int,
    matlab_workers: int | None,
    countinous_only: bool,
) -> None:
    """Run the parameter optimisation diagnostic for all eligible children."""
    try:
        model_ids = _normalise_model_args(models, model_args)
    except Exception as exc:  # noqa: BLE001 - Click entrypoint should report cleanly
        raise click.ClickException(str(exc)) from exc

    results: list[dict[str, Any]] = []
    top_level_failures: list[dict[str, str]] = []
    for model_id in model_ids:
        try:
            results.append(
                run_over_model(
                    model_id=model_id,
                    overwrite=overwrite,
                    only_tsenv_questions=only_tsenv_questions,
                    max_iter=max_iter,
                    tol_x=tol_x,
                    coarse_grid_points=coarse_grid_points,
                    matlab_workers=matlab_workers,
                    countinous_only=countinous_only,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep processing requested models
            top_level_failures.append({"model": model_id, "error": str(exc)})

    for result in results:
        model_id = str(result.get("model") or "")
        for item in result["successes"]:
            click.echo(
                "{model}/{run_id}: wrote {path}".format(
                    model=model_id,
                    run_id=item["run_id"],
                    path=item["path"],
                )
            )
        for item in result["skipped"]:
            click.echo(
                "{model}/{run_id}: skipped fresh optimisation report {path}".format(
                    model=model_id,
                    run_id=item["run_id"],
                    path=item["path"],
                )
            )
        for item in result["failures"]:
            click.echo(
                "{model}/{run_id}: failed: {error}".format(
                    model=model_id,
                    run_id=item["run_id"],
                    error=item["error"],
                ),
                err=True,
            )
    for item in top_level_failures:
        click.echo(
            "{model}: failed: {error}".format(
                model=item["model"],
                error=item["error"],
            ),
            err=True,
        )
    failure_count = sum(len(result["failures"]) for result in results) + len(top_level_failures)
    attempted_count = (
        sum(len(result["successes"]) + len(result["failures"]) for result in results)
        + len(top_level_failures)
    )
    if failure_count:
        raise click.ClickException(
            "parameter optimisation failed for "
            f"{failure_count} of "
            f"{attempted_count} runs"
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
