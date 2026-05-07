from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from shared.benchmark_utils import parse_context_and_shot
from shared.interfaces import validate_question_payload
from shared.prompts import join_message_lines, tsenv_prompt_field_entries
from shared.tsenv_metadata import (
    label_for_question_sample,
    label_choices_from_payload,
    resolve_tsenv_payload_path,
    load_metadata_payload,
    metadata_questions_list,
    question_sample_paths,
)
from shared.tsenv_eval_mode import normalize_tsenv_eval_mode


def resolve_dataset_path(data_root: Path) -> Path:
    data_root = data_root.expanduser().resolve()
    if any(part.startswith("exam_questions_") for part in data_root.parts):
        if not data_root.exists():
            raise FileNotFoundError(f"Exam questions dataset not found: {data_root}")
        return data_root
    dataset_name = data_root.name
    if dataset_name.endswith("_desc"):
        dataset_name = dataset_name[: -len("_desc")]
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / "terminal-bench" / "tasks" / "tsenv" / dataset_name
    if dataset_path.exists():
        return dataset_path.resolve()
    if data_root.exists():
        return data_root
    raise FileNotFoundError(f"Terminal-Bench dataset not found: {dataset_path}")


def safe_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "unknown"


def extend_label_to_index(label_to_index: Dict[str, int]) -> None:
    for label, idx in list(label_to_index.items()):
        safe_label = safe_filename_component(label)
        if safe_label not in label_to_index:
            label_to_index[safe_label] = idx
        elif label_to_index[safe_label] != idx:
            raise ValueError(
                f"Safe label collision: {safe_label!r} maps to multiple indices"
            )


def label_from_path(
    path: Path, label_to_index: Optional[Dict[str, int]] = None
) -> str:
    stem = path.stem
    if stem.startswith("baseline_"):
        stem = stem[len("baseline_") :]
    for token in ("_train_", "_example_"):
        if token in stem:
            stem = stem.split(token, 1)[0]
            break
    if label_to_index:
        candidates = sorted(label_to_index.keys(), key=len, reverse=True)
        for label in candidates:
            if stem == label or stem.endswith(f"_{label}"):
                return label
    return stem.split("_")[-1]
def _parse_few_shot(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
        return []
    if isinstance(raw, Sequence):
        return [str(item) for item in raw if str(item)]
    return []


def _ground_truth_description(payload: Mapping[str, Any]) -> str:
    ground_truth_information = payload.get("ground_truth_information")
    if isinstance(ground_truth_information, dict):
        description = str(ground_truth_information.get("model_description") or "")
        shared_description = str(
            ground_truth_information.get("shared_description") or ""
        )
        if description and shared_description:
            return f"{description}\n\n{shared_description}"
        if description:
            return description
        if shared_description:
            return shared_description
        full_text = str(ground_truth_information.get("ground_truth_text") or "")
        if full_text:
            return full_text
        return description
    return str(ground_truth_information or "")


def _question_context_from_recipe_info(recipe_info: Mapping[str, Any]) -> str:
    return str(recipe_info.get("desc_level") or "none").strip().lower() or "none"


def _shot_from_recipe_info(recipe_info: Mapping[str, Any]) -> str:
    num_examples = int(recipe_info.get("number_train_samples_per_class") or 0)
    if num_examples == 0:
        return "zero_shot"
    if num_examples == 1:
        return "one_shot"
    if num_examples == 3:
        return "few_shot"
    if num_examples == 20:
        return "many_shots"
    raise ValueError(f"Unsupported number_train_samples_per_class={num_examples}")


def load_classification_questions(data_root: Path) -> List[Dict[str, Any]]:
    meta_path = resolve_tsenv_payload_path(data_root)
    meta = load_metadata_payload(meta_path)
    raw_questions = metadata_questions_list(meta)
    label_choices = label_choices_from_payload(meta)
    dataset_name = data_root.name
    exam_root = next(
        (part for part in data_root.parts if part.startswith("exam_questions_") or part == "tsENV_questions"),
        "tsENV_questions",
    )
    questions: List[Dict[str, Any]] = []
    for entry in raw_questions:
        qid = str(entry.get("question_id") or "").strip()
        if not qid:
            raise ValueError("questions.json question missing question_id")
        recipe_info = entry.get("recipe_info")
        question_text = entry.get("question_text")
        if not isinstance(recipe_info, dict) or not isinstance(question_text, dict):
            fallback_payload = dict(entry)
            fallback_payload.setdefault("task", "classification")
            questions.append(validate_question_payload(fallback_payload))
            continue
        train_samples = question_sample_paths(meta, question=entry, subset="train")
        test_samples = question_sample_paths(meta, question=entry, subset="test")
        if not test_samples:
            raise ValueError(f"{meta_path.name} question {qid!r} missing test_samples")
        first_test_sample = test_samples[0]
        label = label_for_question_sample(meta, question=entry, sample_path=first_test_sample)
        context = _question_context_from_recipe_info(recipe_info)
        shot = _shot_from_recipe_info(recipe_info)
        requested_mode = entry.get("eval_mode") or recipe_info.get("eval_mode")
        eval_mode = normalize_tsenv_eval_mode(requested_mode)
        instruction_parts: List[str] = []
        pending_separator = ""
        for field, separator in tsenv_prompt_field_entries(question_text):
            rendered = str(question_text.get(field) or "").strip()
            if not rendered:
                continue
            if instruction_parts:
                instruction_parts.append(pending_separator)
            instruction_parts.append(rendered)
            pending_separator = separator
        instruction = join_message_lines(["".join(instruction_parts)])
        questions.append(
            validate_question_payload(
                {
                    "task": "classification",
                    "question_id": qid,
                    "benchmark": "tsenv_cls",
                    "context": context,
                    "shot": shot,
                    "parent_variant_key": "tsenv_cls",
                    "instruction_agent_format": instruction,
                    "ground_truth_context": _ground_truth_description(meta),
                    "dataset": dataset_name,
                    "exam_question_root": exam_root,
                    "dataframe_name": "dataframe.parquet",
                    "eval_mode": eval_mode,
                    "test_samples": list(test_samples),
                    "train_samples": list(train_samples),
                    "train_paths": list(train_samples),
                    "label": label,
                    "multiple_choices": list(label_choices),
                    "metadata": {
                        "sample_id": entry.get("sample_id"),
                        "recipe_id": recipe_info.get("recipe_id"),
                    },
                }
            )
        )
    return questions



_CLASS_EXAMPLE_RE = re.compile(r"^class_(\d+)_example_\d+$", flags=re.IGNORECASE)
_CLASS_UNKNOWN_EXAMPLE_RE = re.compile(r"^class_unknown_example_\d+$", flags=re.IGNORECASE)


def build_label_to_index(choices: Sequence[str]) -> Dict[str, int]:
    label_to_index = {str(choice): idx for idx, choice in enumerate(list(choices))}
    extend_label_to_index(label_to_index)
    return label_to_index


def question_train_paths(question: Mapping[str, Any]) -> List[str]:
    return _parse_few_shot(question.get("train_paths"))


def infer_few_shot_labels(
    question: Mapping[str, Any], *, label_to_index: Dict[str, int]
) -> List[int]:
    paths = question_train_paths(question)
    if not paths:
        return []

    names_no_desc = list(question.get("few_shot_path_name_no_description") or [])
    names_desc = list(question.get("few_shot_path_name_description") or [])

    def _norm(text: str) -> str:
        return safe_filename_component(text).lower()

    normalized_to_index: Dict[str, int] = {}
    for key, idx in label_to_index.items():
        normalized_to_index.setdefault(_norm(str(key)), int(idx))

    def _lookup_label(stem: str) -> int | None:
        raw = (stem or "").strip()
        if not raw:
            return None
        if raw in label_to_index:
            return int(label_to_index[raw])
        safe = safe_filename_component(raw)
        if safe in label_to_index:
            return int(label_to_index[safe])
        norm = _norm(raw)
        if norm in normalized_to_index:
            return int(normalized_to_index[norm])
        stripped = re.sub(r"^\\d+_", "", raw)
        if stripped != raw:
            return _lookup_label(stripped)
        return None

    choices_norm = sorted(
        {(_norm(str(key)), int(idx)) for key, idx in label_to_index.items()},
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def _infer_from_path(raw_path: str) -> int | None:
        path_norm = _norm(Path(raw_path).stem)
        hits = [(tok, idx) for tok, idx in choices_norm if tok and tok in path_norm]
        if not hits:
            return None
        best_tok, best_idx = hits[0]
        if any(idx != best_idx and tok == best_tok for tok, idx in hits[1:]):
            return None
        return int(best_idx)

    labels: List[int] = []
    for i, raw_path in enumerate(paths):
        label_idx: int | None = None

        if len(names_no_desc) == len(paths):
            name = names_no_desc[i] or ""
            match = _CLASS_EXAMPLE_RE.match(name)
            if match is not None:
                label_idx = int(match.group(1)) - 1
            elif _CLASS_UNKNOWN_EXAMPLE_RE.match(name):
                label_idx = None

        if label_idx is None and len(names_desc) == len(paths):
            desc_name = names_desc[i] or ""
            stem = desc_name.split("_example_", 1)[0]
            label_idx = _lookup_label(stem)

        if label_idx is None:
            label_idx = _infer_from_path(raw_path)

        if label_idx is None:
            raise ValueError(
                "Unable to infer few-shot label for "
                f"{question.get('dataset')}/{question.get('question_id')}: "
                f"path={raw_path!r}, "
                f"few_shot_path_name_no_description={names_no_desc[i] if len(names_no_desc)==len(paths) else None!r}, "
                f"few_shot_path_name_description={names_desc[i] if len(names_desc)==len(paths) else None!r}"
            )

        labels.append(int(label_idx))

    return labels


def assert_strict_balanced_support(
    *,
    labels: Sequence[int],
    num_classes: int,
    choices: Sequence[str],
    question_id: str,
    dataset: str,
) -> None:
    if not labels:
        return
    counts = [0] * int(num_classes)
    for label_idx in labels:
        idx = int(label_idx)
        if idx < 0 or idx >= int(num_classes):
            raise ValueError(
                f"Support label index out of range for {dataset}/{question_id}: {idx} not in [0,{num_classes})"
            )
        counts[idx] += 1
    if any(c == 0 for c in counts) or len(set(counts)) != 1:
        labeled = ", ".join(f"{choices[i]}={counts[i]}" for i in range(int(num_classes)))
        raise ValueError(f"Unbalanced support set for {dataset}/{question_id}: {labeled}")
