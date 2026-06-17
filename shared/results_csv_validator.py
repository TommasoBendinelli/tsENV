from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from shared.benchmark_utils import AVAILABLE_AGENT_MODELS

QUESTION_COLUMNS = ("question_id", "task_id", "question", "question_ref")
MODEL_NAME_COLUMN = "model_name"
AGENT_NAME_COLUMN = "agent_name"
DATASET_COLUMNS = ("dataset", "dataset_name", "category")
BENCHMARK_COLUMNS = ("benchmark_id", "benchmark_name", "benchmark", "dataset_label")
QUESTION_REQUIRED_COLUMNS = (
    "exam_question_root",
    "variant",
    "dataset",
    "question_id",
)

METRIC_PREFIX = "metric_"


@dataclass
class ResultsCsvValidation:
    path: Path
    mode: str
    errors: List[str]
    warnings: List[str]

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("; ".join(self.errors))


def _normalize_headers(headers: Sequence[str]) -> List[str]:
    return [str(name).strip() for name in headers if str(name).strip()]


def _header_set(headers: Sequence[str]) -> set[str]:
    return {name.strip().lower() for name in headers if str(name).strip()}


def _has_any(headers: set[str], candidates: Sequence[str]) -> bool:
    return any(candidate in headers for candidate in candidates)


def _header_index_map(headers: Sequence[str]) -> dict[str, int]:
    return {name.strip().lower(): idx for idx, name in enumerate(headers) if str(name).strip()}


def _iter_metrics_csv_paths(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("metrics__*.csv"))
        else:
            yield path


def _metric_columns(headers: Sequence[str]) -> List[str]:
    columns = []
    for name in headers:
        label = str(name).strip()
        lowered = label.lower()
        if not lowered.startswith(METRIC_PREFIX):
            continue
        suffix = label[len(METRIC_PREFIX):].strip()
        if not suffix or suffix.startswith("_"):
            continue
        columns.append(label)
    return columns


def validate_results_csv(
    path: Path,
    *,
    mode: str = "auto",
    max_rows: int = 200,
) -> ResultsCsvValidation:
    errors: List[str] = []
    warnings: List[str] = []
    resolved_mode = mode

    if mode not in {"auto", "question", "summary"}:
        raise ValueError("mode must be one of: auto, question, summary")

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            headers = next(reader, None)
            if not headers:
                return ResultsCsvValidation(path=path, mode=resolved_mode, errors=["missing header row"], warnings=[])
            headers = _normalize_headers(headers)
            header_set = _header_set(headers)
            header_index = _header_index_map(headers)

            metric_columns = _metric_columns(headers)
            has_metrics = bool(metric_columns)
            has_model_name = MODEL_NAME_COLUMN in header_set
            has_agent_name = AGENT_NAME_COLUMN in header_set
            has_question = _has_any(header_set, QUESTION_COLUMNS)
            has_exam_question_root = "exam_question_root" in header_set
            has_variant = "variant" in header_set
            has_dataset_exact = "dataset" in header_set
            has_question_id = "question_id" in header_set
            has_dataset = _has_any(header_set, DATASET_COLUMNS)
            has_benchmark = _has_any(header_set, BENCHMARK_COLUMNS)
            has_run = "run_id" in header_set or "run_dir" in header_set

            if mode == "auto":
                resolved_mode = "question" if has_question else "summary"

            if not has_metrics:
                errors.append("missing metric_* columns")
            if not has_model_name:
                errors.append("missing model_name column")
            if not has_agent_name:
                errors.append("missing agent_name column")
            if resolved_mode == "question":
                required_present = {
                    "exam_question_root": has_exam_question_root,
                    "variant": has_variant,
                    "dataset": has_dataset_exact,
                    "question_id": has_question_id,
                }
                missing = [name for name in QUESTION_REQUIRED_COLUMNS if not required_present[name]]
                if missing:
                    errors.append(f"question mode requires {', '.join(missing)}")
            if resolved_mode == "summary" and not (has_dataset or has_benchmark):
                errors.append("summary mode requires dataset or benchmark columns")

            if resolved_mode == "question" and not has_run:
                warnings.append("missing run_id/run_dir (trajectory links will be unavailable)")
            if resolved_mode == "question" and not has_dataset:
                warnings.append("missing dataset column (dataset averages may be unavailable)")

            if max_rows != 0:
                metric_indices = [
                    header_index.get(name.strip().lower())
                    for name in metric_columns
                ]
                model_name_idx = header_index.get(MODEL_NAME_COLUMN)
                agent_name_idx = header_index.get(AGENT_NAME_COLUMN)
                allowed_pairs = {(agent, model) for agent, model in AVAILABLE_AGENT_MODELS}
                limit = None if max_rows < 0 else max_rows
                checked = 0
                for row in reader:
                    if limit is not None and checked >= limit:
                        break
                    if not metric_indices or model_name_idx is None or agent_name_idx is None:
                        break
                    if max(
                        [idx for idx in metric_indices if idx is not None] + [model_name_idx, agent_name_idx]
                    ) >= len(row):
                        errors.append("row is missing required columns")
                        break
                    for idx in metric_indices:
                        if idx is None or idx >= len(row):
                            errors.append("row is missing required columns")
                            break
                        metric_value = str(row[idx]).strip()
                        if metric_value:
                            try:
                                float(metric_value)
                            except ValueError:
                                errors.append(
                                    f"metric values must be floats, got {metric_value!r}"
                                )
                                break
                    if errors:
                        break
                    agent_name = str(row[agent_name_idx]).strip()
                    model_name = str(row[model_name_idx]).strip()
                    if agent_name.lower() == "baseline":
                        checked += 1
                        continue
                    if (agent_name, model_name) not in allowed_pairs:
                        errors.append(
                            "model_name and agent_name must match AVAILABLE_AGENT_MODELS "
                            f"(got {(agent_name, model_name)!r})"
                        )
                        break
                    checked += 1
    except FileNotFoundError:
        errors.append("file not found")
    except Exception as exc:
        errors.append(f"failed to read csv: {exc}")

    return ResultsCsvValidation(path=path, mode=resolved_mode, errors=errors, warnings=warnings)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate results metrics CSV files.")
    parser.add_argument("paths", nargs="+", help="CSV file or directory path(s).")
    parser.add_argument("--mode", choices=("auto", "question", "summary"), default="auto")
    parser.add_argument("--max-rows", type=int, default=200, help="Rows to sample for metric parsing (0 to skip).")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args()

    validations: List[ResultsCsvValidation] = []
    for csv_path in _iter_metrics_csv_paths([Path(p) for p in args.paths]):
        validations.append(validate_results_csv(csv_path, mode=args.mode, max_rows=args.max_rows))

    exit_code = 0
    for result in validations:
        rel_path = result.path
        if result.errors:
            exit_code = 1
            print(f"[ERROR] {rel_path}: " + "; ".join(result.errors))
        elif result.warnings:
            if args.strict:
                exit_code = 1
            print(f"[WARN] {rel_path}: " + "; ".join(result.warnings))
        else:
            print(f"[OK] {rel_path} ({result.mode})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
