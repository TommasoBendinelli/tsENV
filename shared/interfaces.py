from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from shared.context_levels import QUESTION_CONTEXT_VALUES, TSENV_CONTEXT_VALUES
from shared.benchmark_utils import (
    ALLOWED_TSENV_MODELS,
    BENCHMARK_CHOICES,
    is_simbench_cls_benchmark,
)
from shared.tsenv_combinations import TIME0_BASELINE_AGENT_FACING_LABEL
from shared.tsenv_eval_mode import normalize_tsenv_eval_mode

QuestionPayload = Dict[str, Any]

_BANNED_SIGNALS_SENTENCE_RE = re.compile(
    r"\bthe signals you are seeing are\s*:", flags=re.IGNORECASE
)
_TSENV_NOTE_SENTENCE = (
    "Note that the underlying simulated model is the same, but the initial conditions, "
    "model parameters, and intervention time differ."
)
_TSENV_ZERO_SHOT_DATA_LOCATION_SENTENCE = (
    "The data to be classified is stored in the test_samples folder."
)
_TSENV_NONZERO_DATA_LOCATION_SENTENCE = (
    "The labeled examples are stored in the train_samples folder, with "
)
_TSENV_NONZERO_DATA_LOCATION_RE = re.compile(
    r"The labeled examples are stored in the train_samples folder, with \d+ labeled examples per class\.\s+"
    r"The label is included in the file name\. "
    r"The data to be classified is stored in the test_samples folder\."
)
_TSENV_ZERO_SHOT_DATA_LOCATION_RE = re.compile(
    rf"(?m)^{re.escape(_TSENV_ZERO_SHOT_DATA_LOCATION_SENTENCE)}$"
)
_TSENV_PARAM_LIST_PATTERNS = (
    re.compile(
        r"For each simulation,\s*either no parameter changes,\s*or exactly one parameter among\s*(.+?)\s*"
        r"changes during the observed simulation interval\.",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"For each test sample,\s*return the smallest set of labels that you believe contains the true label "
        r"with high confidence,\s*based on the observed time series\.\s*"
        r"For each sample,\s*either no parameter changed or exactly one candidate parameter from\s*(.+?)\s*"
        r"changed during the observed simulation interval\.\s*"
        r"(?=.*If you cannot detect any parameter change,\s*return\s*\[\"[^\"]+\"\]\.)"
        r"(?=.*If a parameter changed,\s*assume it changed abruptly at a single time point and produced an observable effect within the recorded interval\.)"
        r"(?=.*You may assume that,\s*in the corresponding noiseless trajectory,\s*the correct changed parameter would be clearly distinguishable from the other candidate parameters and from the \"no parameter chang(?:e|ed)\" case\.)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"For each test sample,\s*(?:determine which single outcome is best supported by the observed time series|"
        r"create an interpretable stand-alone python script that can determine which single outcome is best supported by the observed time series)\.\s*"
        r"For each sample,\s*either no parameter changed or exactly one candidate parameter from\s*(\[[^\]]*\])\s*"
        r"changed during the observed simulation interval\.\s*"
        r"(?=.*If a parameter changed,\s*assume it changed abruptly at a single time point and produced an observable effect within the recorded interval\.)"
        r"(?=.*You may assume that,\s*in the corresponding noiseless trajectory,\s*the correct changed parameter would be clearly distinguishable from the other candidate parameters and from the \"no parameter changed\" case\.)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"For each test sample,\s*(?:determine which single outcome is best supported by the observed time series|"
        r"create an interpretable stand-alone python script that can determine which single outcome is best supported by the observed time series)\.\s*"
        r"For each sample,\s*either no parameter changed or exactly one candidate parameter from\s*(.+?)\s*"
        r"changed during the observed simulation interval\.\s*"
        r"(?=.*If a parameter changed,\s*assume it changed abruptly at a single time point and produced an observable effect within the recorded interval\.)"
        r"(?=.*You may assume that,\s*in the corresponding noiseless trajectory,\s*the correct changed parameter would be clearly distinguishable from the other candidate parameters and from the \"no parameter changed\" case\.)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"For each test sample,\s*determine which single outcome is best supported by the observed time series\.\s*"
        r"For each sample,\s*either no parameter changed or exactly one candidate parameter from\s*(\[[^\]]*\])\s*"
        r"changed during the observed simulation interval\.\s*"
        r"If a parameter changed,\s*assume it changed abruptly at a single time point where it produces an observable effect in the recorded interval\.\s*"
        r"You can assume that if the trajectory was completely noiseless,\s*the correct changed would be clearly distinguishable from the other candidate parameters and from the [\u201c\"]no parameter changed[\u201d\"] case\s*"
        rf"If you cannot detect any parameter change,\s*return\s*\[\"{re.escape(TIME0_BASELINE_AGENT_FACING_LABEL)}\"\]\.",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"For each test sample,\s*determine which single outcome is best supported by the observed time series\.\s*"
        r"For each sample,\s*either no parameter changed or exactly one candidate parameter from\s*(\[[^\]]*\])\s*"
        r"changed during the observed simulation interval\.\s*"
        r"If a parameter changed,\s*assume it changed abruptly at a single time point\.\s*"
        rf"If you cannot detect any parameter change,\s*return\s*\[\"{re.escape(TIME0_BASELINE_AGENT_FACING_LABEL)}\"\]\.",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"For each test sample,\s*determine whether there is evidence that one of the candidate parameters\s*(\[[^\]]*\])\s*changed during the simulation\.\s*"
        r"At most one candidate parameter changed\.\s*If a change occurred,\s*assume it happened abruptly at a single time point and caused an observable effect in the time series from the time its value changed onward\.\s*"
        rf"If you can't detect any parameter change,\s*return\s*\"{re.escape(TIME0_BASELINE_AGENT_FACING_LABEL)}\"\.",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"For each test sample,\s*exactly one of the following parameters changes suddenly once during the simulation:\s*(\[[^\]]*\])\.\s*"
        r"Your task is to determine which parameter changed\.\s*"
        r"If you think the available evidence in a sample is insufficient to determine the changed parameter,\s*"
        rf"return\s*\"{re.escape(TIME0_BASELINE_AGENT_FACING_LABEL)}\"\s*for that sample\.",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"Allowed labels:\s*(\[[^\]]*\])",
        flags=re.IGNORECASE,
    ),
)
_TSENV_QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_TSENV_CONTEXT_VALUES = TSENV_CONTEXT_VALUES
_CONTEXT_VALUES = QUESTION_CONTEXT_VALUES
_SUPPORTED_TASKS = {"classification", "anomaly_localization", "change_point_detection"}
_SUPPORTED_SHOTS = {"zero_shot", "one_shot", "few_shot", "many_shots", "many_shot", "baseline", "unknown"}


def _now_iso8601_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_list_str(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    return [str(value)]


def _parse_json_list(text: str) -> Optional[List[str]]:
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None
    return [str(x) for x in parsed]


def _parse_tsenv_choice_fragment(text: str) -> Optional[List[str]]:
    parsed_json = _parse_json_list(text)
    if parsed_json is not None:
        return parsed_json

    quoted = re.findall(r'"([^"]+)"', str(text or ""))
    if quoted:
        return [str(item) for item in quoted]
    return None


def _infer_tsenv_context_from_variant(variant: str) -> Optional[str]:
    lowered = str(variant or "").strip().lower()
    if lowered.startswith("tsenv_cls_"):
        lowered = lowered[len("tsenv_cls_") :]
    for token in _TSENV_CONTEXT_VALUES:
        if lowered.startswith(f"{token}_"):
            return token
    return None


def _infer_shot_from_variant(variant: str) -> Optional[str]:
    lowered = str(variant or "").strip().lower()
    for token in ("zero_shot", "one_shot", "few_shot", "many_shots", "many_shot", "baseline"):
        if lowered.endswith(token):
            return token
    return None


def _normalize_optional_sample_list(
    value: Optional[Sequence[object]],
    field_name: str,
) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        raise ValueError(f"Question.{field_name} must be a list of sample paths/ids, not a string")
    if not isinstance(value, Sequence):
        raise ValueError(f"Question.{field_name} must be a list of sample paths/ids")
    normalized: List[str] = []
    for idx, item in enumerate(value):
        text = str(item or "").strip()
        if not text:
            raise ValueError(f"Question.{field_name}[{idx}] must be non-empty")
        normalized.append(text)
    return normalized


def _normalize_eval_mode(payload: QuestionPayload) -> str:
    benchmark = str(payload.get("benchmark") or "").strip()
    raw_mode = payload.get("eval_mode")
    if is_simbench_cls_benchmark(benchmark):
        return normalize_tsenv_eval_mode(raw_mode)
    normalized = str(raw_mode or "").strip().lower()
    return normalized or "online"


def _validate_tsenv_prompt(
    *,
    eval_mode: str,
    instruction_agent_format: str,
    choices: Sequence[str],
) -> None:
    if str(eval_mode or "").strip().lower() == "open-ended":
        return
    text = str(instruction_agent_format or "").replace("\r\n", "\n").replace("\r", "\n")
    match = None
    for pattern in _TSENV_PARAM_LIST_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            break
    if match is None:
        raise ValueError(
            "tsENV prompts must include the classification question sentence."
        )

    parsed = _parse_tsenv_choice_fragment(match.group(1))
    if parsed is None:
        raise ValueError("tsENV intervention sentence must contain a parseable list of choices.")
    expected_choices = [
        str(choice)
        for choice in choices
        if str(choice).strip() != TIME0_BASELINE_AGENT_FACING_LABEL
    ]
    allowed_choice_sets = (
        Counter(str(choice) for choice in choices),
        Counter(expected_choices),
    )
    if Counter(parsed) not in allowed_choice_sets:
        raise ValueError("tsENV intervention sentence choices must match multiple_choices.")


def question_variant(question: Mapping[str, Any]) -> str:
    explicit = str(question.get("variant") or "").strip()
    if explicit:
        return explicit
    context = str(question.get("context") or "unknown")
    shot = str(question.get("shot") or "unknown")
    return f"{context}_{shot}"


def validate_question_payload(payload: Mapping[str, Any]) -> QuestionPayload:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    question: QuestionPayload = dict(payload)
    task = str(question.get("task") or "").strip().lower()
    if not task:
        raise ValueError("payload.task is required")
    if task not in _SUPPORTED_TASKS:
        raise ValueError("Question.task must be a supported task type")
    question["task"] = task

    if "episode_id" in question:
        raise ValueError("Question.episode_id is not a supported field")

    if "dataframe_path" in question:
        raise ValueError("Question.dataframe_path is no longer supported; use Question.test_samples.")
    if "instruction_human_format" in question or "base_instruction_human_format" in question:
        raise ValueError(
            "Question.instruction_human_format and Question.base_instruction_human_format "
            "are no longer supported; use Question.instruction_agent_format."
        )

    question.setdefault("dataframe_name", "dataframe.parquet")
    question.setdefault("timestamp", _now_iso8601_utc())
    question["signals"] = _as_list_str(question.get("signals"))
    if "multiple_choices" in question:
        question["multiple_choices"] = _as_list_str(question.get("multiple_choices"))

    context_raw = str(question.get("context") or "").strip().lower()
    if not context_raw:
        raise ValueError("Question.context must be non-empty")
    if context_raw not in _CONTEXT_VALUES:
        raise ValueError(f"Question.context must be one of {', '.join(_CONTEXT_VALUES)}")
    question["context"] = context_raw

    shot_raw = str(question.get("shot") or "").strip().lower()
    if not shot_raw:
        raise ValueError("Question.shot must be non-empty")
    if shot_raw not in _SUPPORTED_SHOTS:
        raise ValueError("Question.shot must be one of zero_shot, one_shot, few_shot, many_shots, many_shot, baseline, unknown")
    question["shot"] = shot_raw

    benchmark = str(question.get("benchmark") or "").strip()
    if not benchmark or benchmark not in BENCHMARK_CHOICES:
        raise ValueError(
            "Question.benchmark must be one of "
            f"{', '.join(BENCHMARK_CHOICES)}; update shared/benchmark_utils.py if adding new benchmarks."
        )
    question["benchmark"] = benchmark

    if not str(question.get("parent_variant_key") or "").strip():
        raise ValueError("Question.parent_variant_key must be non-empty")
    if not str(question.get("dataset") or "").strip():
        raise ValueError("Question.dataset must be non-empty")

    question_id = str(question.get("question_id") or "").strip()
    if not question_id:
        raise ValueError("Question.question_id must be non-empty")
    question["question_id"] = question_id

    eval_mode = _normalize_eval_mode(question)
    question["eval_mode"] = eval_mode

    if is_simbench_cls_benchmark(benchmark):
        if not _TSENV_QUESTION_ID_RE.match(question_id):
            raise ValueError("Question.question_id must be a simple slug for tsENV.")
        if context_raw not in _TSENV_CONTEXT_VALUES:
            raise ValueError("Question.context must be one of {'none','low','high','ground_truth'} for tsENV.")
        if shot_raw not in {"zero_shot", "one_shot", "few_shot", "many_shots"}:
            raise ValueError("Question.shot must be one of {'zero_shot','one_shot','few_shot','many_shots'} for tsENV.")
        dataset = str(question["dataset"]).strip()
        if Path(dataset).name != dataset:
            raise ValueError(f"tsENV dataset must be a single directory name; got {question['dataset']!r}")
        if dataset not in ALLOWED_TSENV_MODELS:
            raise ValueError(
                "tsENV dataset must be one of ALLOWED_TSENV_MODELS defined in shared/benchmark_utils.py; "
                f"got {question['dataset']!r}"
            )
    exam_question_root = str(question.get("exam_question_root") or "").strip()
    if not exam_question_root:
        raise ValueError("Question.exam_question_root must be non-empty")
    question["exam_question_root"] = exam_question_root

    dataframe_name = str(question.get("dataframe_name") or "").strip()
    if not dataframe_name:
        raise ValueError("Question.dataframe_name must be non-empty")
    if dataframe_name != "dataframe.parquet":
        raise ValueError("Question.dataframe_name must be 'dataframe.parquet'")
    question["dataframe_name"] = dataframe_name

    instruction_agent_format = str(question.get("instruction_agent_format") or "")
    if _BANNED_SIGNALS_SENTENCE_RE.search(instruction_agent_format):
        raise ValueError("Question prompts must not include the sentence prefix about what signals are shown")

    if benchmark == "tsenv_cls":
        if shot_raw == "zero_shot":
            if _TSENV_NONZERO_DATA_LOCATION_SENTENCE in instruction_agent_format:
                raise ValueError(
                    "tsENV zero-shot Question.instruction_agent_format must not use "
                    f"{_TSENV_NONZERO_DATA_LOCATION_SENTENCE!r}; use "
                    f"{_TSENV_ZERO_SHOT_DATA_LOCATION_SENTENCE!r} instead."
                )
            if len(_TSENV_ZERO_SHOT_DATA_LOCATION_RE.findall(instruction_agent_format)) != 1:
                raise ValueError(
                    "tsENV zero-shot Question.instruction_agent_format must include "
                    f"exactly once: {_TSENV_ZERO_SHOT_DATA_LOCATION_SENTENCE!r}"
                )
        else:
            if _TSENV_ZERO_SHOT_DATA_LOCATION_RE.search(instruction_agent_format):
                raise ValueError(
                    "tsENV non-zero-shot Question.instruction_agent_format must not "
                    f"use {_TSENV_ZERO_SHOT_DATA_LOCATION_SENTENCE!r}; use "
                    f"{_TSENV_NONZERO_DATA_LOCATION_SENTENCE!r} instead."
                )
            matches = _TSENV_NONZERO_DATA_LOCATION_RE.findall(instruction_agent_format)
            if len(matches) != 1:
                raise ValueError(
                    "tsENV Question.instruction_agent_format must include exactly once: "
                    f"{_TSENV_NONZERO_DATA_LOCATION_SENTENCE!r}"
                )

    if ".png" in instruction_agent_format.lower():
        raise ValueError("Question.instruction_agent_format must not mention .png files")

    ground_truth_context = question.get("ground_truth_context")
    if ground_truth_context is None or not isinstance(ground_truth_context, str):
        raise ValueError("Question.ground_truth_context must be a string")

    metadata = question.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Question.metadata must be a dict when provided")
    if question.get("is_solved_by_inference") is not None and not isinstance(
        question.get("is_solved_by_inference"), bool
    ):
        raise ValueError("Question.is_solved_by_inference must be a bool when provided")

    question["test_samples"] = _normalize_optional_sample_list(question.get("test_samples"), "test_samples")
    question["train_samples"] = _normalize_optional_sample_list(question.get("train_samples"), "train_samples")

    if not question["test_samples"]:
        raise ValueError(f"Question.test_samples is required for eval_mode={eval_mode!r}")

    for field_name in ("instruction_agent_format", "ground_truth_context"):
        text = str(question.get(field_name) or "").replace("\r\n", "\n").replace("\r", "\n")
        if text.startswith("\n") and text:
            raise ValueError(f"Question.{field_name} must not start with a newline")
        if re.search(r"\n{3,}", text):
            raise ValueError(f"Question.{field_name} must not contain consecutive empty lines.")

    if task == "classification":
        label = str(question.get("label") or "").strip()
        if not label:
            raise ValueError("Classification question.label must be non-empty")
        question["label"] = label
        choices = [str(choice) for choice in question.get("multiple_choices") or []]
        if not choices:
            raise ValueError("Classification question.multiple_choices must be non-empty")
        question["multiple_choices"] = choices
        if is_simbench_cls_benchmark(benchmark):
            _validate_tsenv_prompt(
                eval_mode=eval_mode,
                instruction_agent_format=instruction_agent_format,
                choices=choices,
            )

    return question


def assert_balanced_class_distribution(
    questions: Sequence[Mapping[str, Any]],
    *,
    max_diff: int = 1,
    benchmark_id: str = "",
) -> None:
    labels: List[str] = []
    for question in questions:
        if str(question.get("task") or "").strip().lower() != "classification":
            continue
        label = question.get("label")
        if label is not None:
            labels.append(str(label))
    if not labels:
        return
    counts = Counter(labels)
    if not counts:
        return
    lo = min(counts.values())
    hi = max(counts.values())
    if hi - lo > int(max_diff):
        raise ValueError(
            f"Unbalanced class distribution for {benchmark_id or 'questions'}: {dict(counts)}"
        )


__all__ = [
    "QuestionPayload",
    "_TSENV_NOTE_SENTENCE",
    "assert_balanced_class_distribution",
    "question_variant",
    "validate_question_payload",
]
