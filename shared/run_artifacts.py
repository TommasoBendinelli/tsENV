from __future__ import annotations

from pathlib import Path
from typing import Optional


DEFAULT_RUNS_DIR_NAME = "runs"
MODEL_RECORD_FILENAME = "model_record.json"
QUESTIONS_FILENAME = "questions.json"
SAMPLE_MANIFEST_FILENAME = "sample_manifest.json"
SIMILARITY_METRICS_FILENAME = "eligibility_metrics.json"
SIMILARITY_METRICS_RUN_SUMMARY_FILENAME = "summary_metrics.json"


def _resolve_runs_dir_name(runs_dir_name: Optional[str]) -> str:
    dir_name = str(runs_dir_name or DEFAULT_RUNS_DIR_NAME).strip()
    if not dir_name:
        raise ValueError("runs_dir_name must be non-empty")
    if "/" in dir_name or "\\" in dir_name:
        raise ValueError("runs_dir_name must be a simple directory name")
    return dir_name


def resolve_runs_root(
    model_dir: Path,
    runs_dir: Optional[Path] = None,
    runs_dir_name: Optional[str] = None,
) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if runs_dir is not None:
        return runs_dir.expanduser().resolve()
    return model_dir / _resolve_runs_dir_name(runs_dir_name)


def resolve_model_artifact_root(
    model_dir: Path,
    runs_dir: Optional[Path] = None,
    runs_dir_name: Optional[str] = None,
) -> Path:
    return resolve_runs_root(
        model_dir,
        runs_dir=runs_dir,
        runs_dir_name=runs_dir_name,
    )


def resolve_model_record_path(
    model_dir: Path,
    runs_dir: Optional[Path] = None,
    runs_dir_name: Optional[str] = None,
) -> Path:
    return resolve_model_artifact_root(
        model_dir,
        runs_dir=runs_dir,
        runs_dir_name=runs_dir_name,
    ) / MODEL_RECORD_FILENAME


def resolve_questions_path(
    model_dir: Path,
    runs_dir: Optional[Path] = None,
    runs_dir_name: Optional[str] = None,
) -> Path:
    return resolve_model_artifact_root(
        model_dir,
        runs_dir=runs_dir,
        runs_dir_name=runs_dir_name,
    ) / QUESTIONS_FILENAME


def resolve_sample_manifest_path(
    model_dir: Path,
    runs_dir: Optional[Path] = None,
    runs_dir_name: Optional[str] = None,
) -> Path:
    return resolve_model_artifact_root(
        model_dir,
        runs_dir=runs_dir,
        runs_dir_name=runs_dir_name,
    ) / SAMPLE_MANIFEST_FILENAME


def resolve_similarity_metrics_path(
    model_dir: Path,
    runs_dir: Optional[Path] = None,
    runs_dir_name: Optional[str] = None,
) -> Path:
    return resolve_model_artifact_root(
        model_dir,
        runs_dir=runs_dir,
        runs_dir_name=runs_dir_name,
    ) / SIMILARITY_METRICS_FILENAME


def resolve_similarity_metrics_run_summary_path(
    model_dir: Path,
    runs_dir: Optional[Path] = None,
    runs_dir_name: Optional[str] = None,
) -> Path:
    return resolve_model_artifact_root(
        model_dir,
        runs_dir=runs_dir,
        runs_dir_name=runs_dir_name,
    ) / SIMILARITY_METRICS_RUN_SUMMARY_FILENAME
