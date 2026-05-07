"""Helpers for generating lightweight exam question outputs."""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from shared.exam_questions_paths import (
    EXAM_QUESTIONS_LITE_ROOT,
    resolve_exam_questions_root_from_values,
)
from shared.tsenv_metadata import load_metadata_payload, metadata_questions_by_id, resolve_tsenv_payload_path
REMOTE_EXAM_QUESTIONS_LITE_PATH = "tommaso@t7144:~/repo/tsENV/exam_questions_lite/"


def _sample_questions(
    questions: Sequence[Dict[str, Any]],
    sample_count: int,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not questions:
        raise ValueError("metadata questions are empty")
    if len(questions) <= sample_count:
        return list(questions)
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(questions)), sample_count))
    return [questions[idx] for idx in selected_indices]


def _select_questions_by_id(
    questions: Sequence[Dict[str, Any]],
    question_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    if not question_ids:
        raise ValueError("question_ids must be non-empty")
    id_set: Set[str] = set()
    for question_id in question_ids:
        if not isinstance(question_id, str) or not question_id:
            raise TypeError("question_ids must contain non-empty strings")
        if question_id in id_set:
            raise ValueError(f"Duplicate question_id requested: {question_id}")
        id_set.add(question_id)
    selected: List[Dict[str, Any]] = []
    found: Set[str] = set()
    for question in questions:
        question_id = question["question_id"]
        if question_id in id_set:
            selected.append(question)
            found.add(question_id)
    missing = id_set - found
    if missing:
        raise KeyError(f"Requested question_ids missing: {', '.join(sorted(missing))}")
    return selected


def _normalize_paths(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("train_paths JSON must be a list")
            return [str(item) for item in parsed if str(item)]
        return [text]
    raise TypeError(f"Unsupported path container type: {type(value)}")


def _collect_question_paths(questions: Sequence[Dict[str, Any]]) -> Set[str]:
    paths: Set[str] = set()
    for question in questions:
        if "label_path" in question and question["label_path"]:
            paths.add(str(question["label_path"]))
        for key in ("test_samples", "train_samples", "train_paths"):
            if key not in question:
                continue
            for path in _normalize_paths(question[key]):
                paths.add(path)
    return paths


def _filter_baselines(
    baselines: Any,
    referenced_paths: Set[str],
) -> Any:
    if not isinstance(baselines, list):
        return baselines
    filtered: List[Dict[str, Any]] = []
    for entry in baselines:
        if not isinstance(entry, dict):
            raise ValueError("Baseline entries must be JSON objects")
        if "path" not in entry:
            raise KeyError("Baseline entry missing path")
        if entry["path"] in referenced_paths:
            filtered.append(entry)
    return filtered


def _filter_dataset_stats(
    dataset_stats: Any,
    questions: Sequence[Dict[str, Any]],
) -> Any:
    if not isinstance(dataset_stats, dict):
        return dataset_stats
    question_ids = {q["question_id"] for q in questions if "question_id" in q and q["question_id"]}
    datasets = {
        q.get("dataset")
        for q in questions
        if isinstance(q, dict) and q.get("dataset")
    }
    keys = set(dataset_stats.keys())
    if question_ids and question_ids.issubset(keys):
        return {key: dataset_stats[key] for key in keys if key in question_ids}
    if datasets and datasets.issubset(keys):
        return {key: dataset_stats[key] for key in keys if key in datasets}
    return dataset_stats


def _copy_relative_paths(paths: Iterable[str], source_dir: Path, dest_dir: Path) -> None:
    for rel_path in sorted(set(paths)):
        if not rel_path:
            continue
        path = Path(rel_path)
        if path.is_absolute():
            raise ValueError(f"Expected relative path, got {path}")
        src_path = source_dir / path
        if not src_path.exists():
            raise FileNotFoundError(f"Missing referenced file: {src_path}")
        dest_path = dest_dir / path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)


def create_lite_exam_questions(
    source_dir: Path,
    *,
    sample_count: int = 5,
    seed: Optional[int] = None,
    question_ids: Optional[Sequence[str]] = None,
    overwrite: bool = True,
) -> Path:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    metadata_path = resolve_tsenv_payload_path(source_dir)

    payload = load_metadata_payload(metadata_path)
    questions_by_id = metadata_questions_by_id(payload)
    questions = list(questions_by_id.values())

    if question_ids is None:
        selected_questions = _sample_questions(questions, sample_count, seed)
    else:
        selected_questions = _select_questions_by_id(questions, question_ids)
    payload["questions"] = {
        question["question_id"]: {
            key: value
            for key, value in question.items()
            if key != "question_id"
        }
        for question in selected_questions
    }
    if "summary_by_correct_answer" in payload:
        counts = Counter(
            str(q["label"]) if "label" in q and q["label"] is not None else ""
            for q in selected_questions
        )
        payload["summary_by_correct_answer"] = {key: count for key, count in sorted(counts.items())}

    detected_root = resolve_exam_questions_root_from_values(source_dir)
    if detected_root is not None:
        rel_path = source_dir.relative_to(detected_root)
        lite_dir = EXAM_QUESTIONS_LITE_ROOT / rel_path
    else:
        lite_dir = EXAM_QUESTIONS_LITE_ROOT / source_dir.name
    if lite_dir.exists() and overwrite:
        shutil.rmtree(lite_dir)
    lite_dir.mkdir(parents=True, exist_ok=True)

    for entry in source_dir.iterdir():
        if entry.name == "dataframes":
            continue
        if entry.is_dir():
            dest_dir = lite_dir / entry.name
            if dest_dir.exists() and overwrite:
                shutil.rmtree(dest_dir)
            shutil.copytree(entry, dest_dir, dirs_exist_ok=not overwrite)
        elif entry.is_file():
            if entry.name in {"questions.json", "dataset_stats.json", "baselines.json"}:
                continue
            shutil.copy2(entry, lite_dir / entry.name)

    referenced_paths = _collect_question_paths(selected_questions)

    baselines_path = source_dir / "baselines.json"
    if baselines_path.exists():
        baselines_payload = json.loads(baselines_path.read_text(encoding="utf-8"))
        baselines_payload = _filter_baselines(baselines_payload, referenced_paths)
        (lite_dir / "baselines.json").write_text(
            json.dumps(baselines_payload, indent=2), encoding="utf-8"
        )
        if isinstance(baselines_payload, list):
            for entry in baselines_payload:
                referenced_paths.add(str(entry["path"]))

    dataset_stats_path = source_dir / "dataset_stats.json"
    if dataset_stats_path.exists():
        dataset_stats_payload = json.loads(dataset_stats_path.read_text(encoding="utf-8"))
        dataset_stats_payload = _filter_dataset_stats(dataset_stats_payload, selected_questions)
        (lite_dir / "dataset_stats.json").write_text(
            json.dumps(dataset_stats_payload, indent=2), encoding="utf-8"
        )

    (lite_dir / metadata_path.name).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    _copy_relative_paths(referenced_paths, source_dir, lite_dir)
    return lite_dir


__all__ = [
    "create_lite_exam_questions",
    "EXAM_QUESTIONS_LITE_ROOT",
    "REMOTE_EXAM_QUESTIONS_LITE_PATH",
]
