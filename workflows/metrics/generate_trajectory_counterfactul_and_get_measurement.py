from __future__ import annotations

import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlencode

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import shared.simulation as simulation_api  # noqa: E402
from shared.interface.distribution_json import load_experiment_config_json  # noqa: E402
from shared.interface.model_record_json import (  # noqa: E402
    load_model_record_json,
    load_model_run_specs_json,
)
from shared.interface.simulink_metadata_json import load_simulink_generated_metadata  # noqa: E402
from shared.run_artifacts import resolve_model_record_path, resolve_runs_root  # noqa: E402
from shared.time_series_metrics import compute_detectability_baseline, load_run_df  # noqa: E402
from workflows.metrics import compute_metrics, parameter_optimisation  # noqa: E402
import workflows.simulate.build_metadata as build_metadata  # noqa: E402


REPORT_FILENAME = "optimisation_results.json"
COUNTERFACTUAL_FILENAME = "data_counterfactual.parquet"
DETECTABILITY_FILENAME = "eligibility_metric_detectable.json"
_WEBAPP_BASE_URL = "http://localhost:3001/"


def _webapp_run_url(*, model_id: str, run_id: str) -> str:
    query = urlencode(
        {
            "model": str(model_id or "").strip(),
            "run": str(run_id or "").strip(),
            "compare": "none",
        }
    )
    return f"{_WEBAPP_BASE_URL}?{query}"


def _resolve_model_dir(model_id: str) -> Path:
    model_name = str(model_id or "").strip()
    if not model_name or "/" in model_name or "\\" in model_name:
        raise ValueError(f"Expected a model id under models/simulink, got {model_id!r}")
    model_dir = parameter_optimisation._resolve_models_root() / model_name
    if not model_dir.exists():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    return model_dir


def _read_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _best_non_ground_truth_candidate(report: Mapping[str, Any]) -> Mapping[str, Any]:
    ground_truth = report.get("ground_truth_intervention")
    ground_truth_parameter = ""
    if isinstance(ground_truth, Mapping):
        ground_truth_parameter = str(ground_truth.get("parameter") or "").strip()
    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for raw_candidate in report.get("candidates", []):
        if not isinstance(raw_candidate, Mapping):
            continue
        parameter = str(raw_candidate.get("parameter") or "").strip()
        if not parameter or parameter == ground_truth_parameter:
            continue
        loss = _finite_float(raw_candidate.get("optimisation_loss", raw_candidate.get("loss")))
        value = _finite_float(raw_candidate.get("param_opt", raw_candidate.get("p_opt")))
        if loss is None or value is None:
            continue
        candidates.append((loss, parameter, raw_candidate))
    if not candidates:
        raise ValueError("No finite non-ground-truth optimisation candidate found")
    return sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]


def _counterfactual_is_fresh(*, counterfactual_path: Path, report_path: Path) -> bool:
    if not counterfactual_path.exists() or not report_path.exists():
        return False
    return counterfactual_path.stat().st_mtime > report_path.stat().st_mtime


@contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _observable_signal_types(experiment_config: Any) -> dict[str, str]:
    raw_types = getattr(experiment_config, "observable_signal_types", {})
    if not isinstance(raw_types, Mapping):
        return {}
    return {
        str(signal): str(signal_type)
        for signal, signal_type in raw_types.items()
    }


def _write_counterfactual_parquet(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    if "time" in out.columns:
        time_values = pd.to_numeric(out.pop("time"), errors="coerce")
    else:
        time_values = pd.to_numeric(out.index, errors="coerce")
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(np.float32)
    out["time"] = time_values.astype(np.float32)
    out.reset_index(drop=True, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, compression="zstd", index=False)


def _simulate_counterfactual_dataframe(
    *,
    model_id: str,
    model_dir: Path,
    context: parameter_optimisation.ChildRunContext,
    candidate: Mapping[str, Any],
    experiment_config: Any,
    metadata: Mapping[str, Any],
    matlab_engine: Any,
) -> pd.DataFrame:
    observable_signals = list(experiment_config.observable_signal_names)
    parameter = str(candidate.get("parameter") or "").strip()
    value = _finite_float(candidate.get("param_opt", candidate.get("p_opt")))
    if not parameter or value is None:
        raise ValueError("Counterfactual candidate must define parameter and param_opt")
    rel_model = (
        str(model_dir.relative_to(root_dir / "models"))
        if model_dir.is_relative_to(root_dir / "models")
        else str(model_dir)
    )
    recipe = {
        "id": context.run_id,
        "run_id": context.run_id,
        "model": rel_model,
        "parent_id": context.baseline_run_id,
        "baseline_parameters": dict(context.baseline_parameters),
        "intervention_mode": "at_intervention_time",
        "intervention_time": float(context.intervention_time),
        "end_time_input_s": float(experiment_config.end_time_input_s),
        "intervention_parameters": {parameter: value},
    }
    signal_dict = simulation_api.simulate_recipe(
        recipe,
        observable_signals,
        matlab_engine=matlab_engine,
        metadata=dict(metadata),
        run_dir=resolve_runs_root(model_dir) / context.run_id,
        internal_sampling_rate_hz=float(experiment_config.sampling_rate_hz),
        sampling_rate_hz=float(experiment_config.sampling_rate_hz),
        sim_script=model_dir.parent / "sim_the_model.m",
        all_signal_names=observable_signals,
        feature_model_dir=model_dir,
        return_features=False,
        compute_features=False,
    )
    if isinstance(signal_dict, tuple):
        signal_dict = signal_dict[0]
    df = simulation_api.resample_signal_dict(
        signal_dict,
        _observable_signal_types(experiment_config),
        internal_sampling_rate_hz=float(experiment_config.sampling_rate_hz),
        sampling_rate_hz=float(experiment_config.sampling_rate_hz),
        end_time_input_s=float(experiment_config.end_time_input_s),
    )
    simulation_api.validate(
        context.run_id,
        df=df,
        observable_signals=observable_signals,
        sampling_rate_hz=float(experiment_config.sampling_rate_hz),
        end_time_input_s=float(experiment_config.end_time_input_s),
    )
    return df


def _detectability_summary_entry(
    *,
    model_id: str,
    run_id: str,
    child_df: pd.DataFrame,
    counterfactual_df: pd.DataFrame,
    experiment_config: Any,
    model_dir: Path,
    intervention_time: float,
) -> dict[str, Any]:
    counterfactual_with_time = counterfactual_df.copy()
    if "time" not in counterfactual_with_time.columns:
        counterfactual_with_time["time"] = pd.to_numeric(
            counterfactual_with_time.index,
            errors="coerce",
        )
    counterfactual_with_time = counterfactual_with_time.reset_index(drop=True)
    child_aligned = child_df[[*list(counterfactual_with_time.columns[:-1]), "time"]]
    noise_adder_path = model_dir / "noise_adder.py"
    (
        _noisy_counterfactual_seed0,
        _noisy_child_seed0,
        mean_dirty,
        mean_clean_baseline,
        mean_snr,
    ) = compute_metrics._detectability_noise_distance_summary(
        model_id=model_id,
        baseline_df=counterfactual_with_time,
        run_df=child_aligned,
        noise_baseline_df=counterfactual_with_time,
        clean_baseline_df=counterfactual_with_time,
        noise_adder_path=noise_adder_path if noise_adder_path.exists() else None,
    )
    clean_counterfactual = compute_metrics._preprocess_detectability_frame(
        counterfactual_with_time,
        model_id=model_id,
    )
    clean_child = compute_metrics._preprocess_detectability_frame(
        child_aligned,
        model_id=model_id,
    )
    payload = compute_detectability_baseline(
        baseline_df=clean_counterfactual,
        run_df=clean_child,
        first_detectable_minimum_symmetric_distance=float(experiment_config.min_srd_distance),
        first_detectable_epsilon=float(experiment_config.epsilon_SRD),
        minimum_consecutive_srd_steps=max(
            1,
            int(getattr(experiment_config, "minimum_consecurive_below_SRD", 1)),
        ),
        intervention_time=float(intervention_time),
        signal_detectability_specs=compute_metrics._signal_detectability_specs(experiment_config),
        require_signal_detectability_specs=True,
        mean_euclidean_distance_clean_dirty=mean_dirty,
        mean_euclidean_distance_clean_baseline=mean_clean_baseline,
        mean_SNR=mean_snr,
        signal_to_noise_ratio_db_thresholds=(
            compute_metrics._signal_to_noise_ratio_thresholds(experiment_config, profile="high")
        ),
    )
    return {
        "url": _webapp_run_url(model_id=model_id, run_id=run_id),
        "detectable": str(payload.get("detectable") or "error"),
        "mean_SNR": list(payload.get("mean_SNR") or []),
        "mean_euclidean_distance_clean_baseline": list(
            payload.get("mean_euclidean_distance_clean_baseline") or []
        ),
    }


def _failure_detectability_entry(*, model_id: str, run_id: str, error: Exception) -> dict[str, Any]:
    return {
        "url": _webapp_run_url(model_id=model_id, run_id=run_id),
        "detectable": "error",
        "mean_SNR": [],
        "mean_euclidean_distance_clean_baseline": [],
        "error": str(error),
    }


def run_for_model(
    *,
    model_id: str,
    overwrite: bool = False,
    matlab_engine: Any = None,
) -> dict[str, Any]:
    model_dir = _resolve_model_dir(model_id)
    runs_root = resolve_runs_root(model_dir)
    specs = load_model_run_specs_json(
        model_dir / "model_run_specs.json",
        enforce_baseline_pair_diversity=False,
    )
    runtime_record = load_model_record_json(resolve_model_record_path(model_dir))
    experiment_config = load_experiment_config_json(model_dir / "experiment_config.json")
    metadata_model = load_simulink_generated_metadata(model_dir / "generated" / "metadata.json")
    metadata = metadata_model.model_dump(mode="python")
    summary: dict[str, Any] = {}
    successes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    report_paths = sorted(runs_root.glob(f"*/{REPORT_FILENAME}"))

    matlab_session_cm = None

    temp_source = model_dir / "simulink_model.mdl"
    created_temp = False

    def get_matlab_engine() -> Any:
        nonlocal matlab_engine, matlab_session_cm
        if matlab_engine is None:
            matlab_session_cm = build_metadata.matlab_session(Path.cwd())
            matlab_engine = matlab_session_cm.__enter__()
        return matlab_engine

    def ensure_working_model_copy() -> None:
        nonlocal created_temp
        if not temp_source.exists() and (model_dir / "simulink_model_original.mdl").exists():
            shutil.copy(model_dir / "simulink_model_original.mdl", temp_source)
            created_temp = True

    try:
        with _pushd(model_dir):
            for report_path in tqdm(report_paths, desc="counterfactual trajectory", unit="run"):
                run_id = report_path.parent.name
                counterfactual_path = report_path.parent / COUNTERFACTUAL_FILENAME
                try:
                    if not overwrite and _counterfactual_is_fresh(
                        counterfactual_path=counterfactual_path,
                        report_path=report_path,
                    ):
                        child_df = load_run_df(report_path.parent)
                        counterfactual_df = pd.read_parquet(counterfactual_path)
                        if child_df is None:
                            raise ValueError(f"Missing child data.parquet for {run_id}")
                        context = parameter_optimisation._find_child_context(
                            run_id=run_id,
                            specs=specs,
                            runtime_record=runtime_record,
                        )
                        summary[run_id] = _detectability_summary_entry(
                            model_id=model_id,
                            run_id=run_id,
                            child_df=child_df,
                            counterfactual_df=counterfactual_df,
                            experiment_config=experiment_config,
                            model_dir=model_dir,
                            intervention_time=context.intervention_time,
                        )
                        skipped.append({"run_id": run_id, "path": str(counterfactual_path)})
                        continue

                    report = _read_json_object(report_path)
                    candidate = _best_non_ground_truth_candidate(report)
                    context = parameter_optimisation._find_child_context(
                        run_id=run_id,
                        specs=specs,
                        runtime_record=runtime_record,
                    )
                    ensure_working_model_copy()
                    df = _simulate_counterfactual_dataframe(
                        model_id=model_id,
                        model_dir=model_dir,
                        context=context,
                        candidate=candidate,
                        experiment_config=experiment_config,
                        metadata=metadata,
                        matlab_engine=get_matlab_engine(),
                    )
                    _write_counterfactual_parquet(df, counterfactual_path)
                    child_df = load_run_df(report_path.parent)
                    if child_df is None:
                        raise ValueError(f"Missing child data.parquet for {run_id}")
                    summary[run_id] = _detectability_summary_entry(
                        model_id=model_id,
                        run_id=run_id,
                        child_df=child_df,
                        counterfactual_df=df,
                        experiment_config=experiment_config,
                        model_dir=model_dir,
                        intervention_time=context.intervention_time,
                    )
                    successes.append({"run_id": run_id, "path": str(counterfactual_path)})
                except Exception as exc:  # noqa: BLE001 - collect all runs
                    summary[run_id] = _failure_detectability_entry(
                        model_id=model_id,
                        run_id=run_id,
                        error=exc,
                    )
                    failures.append(
                        {
                            "run_id": run_id,
                            "path": str(counterfactual_path),
                            "error": str(exc),
                        }
                    )
    finally:
        if created_temp:
            try:
                temp_source.unlink()
            except FileNotFoundError:
                pass
        if matlab_session_cm is not None:
            matlab_session_cm.__exit__(None, None, None)

    summary_path = runs_root / DETECTABILITY_FILENAME
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
@click.option("--overwrite", is_flag=True, default=False)
def cli(models: tuple[str, ...], overwrite: bool) -> None:
    failure_count = 0
    for model_id in models:
        try:
            result = run_for_model(model_id=model_id, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            failure_count += 1
            click.echo(f"{model_id}: failed: {exc}", err=True)
            continue
        for item in result["successes"]:
            click.echo(f"{model_id}/{item['run_id']}: wrote {item['path']}")
        for item in result["skipped"]:
            click.echo(f"{model_id}/{item['run_id']}: skipped fresh {item['path']}")
        for item in result["failures"]:
            failure_count += 1
            click.echo(
                f"{model_id}/{item['run_id']}: failed: {item['error']}",
                err=True,
            )
    if failure_count:
        raise click.ClickException(
            f"counterfactual generation failed for {failure_count} run(s)"
        )


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
