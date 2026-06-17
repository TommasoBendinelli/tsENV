"""Shared helpers for loading exam question metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

from shared.tsenv_metadata import load_metadata_payload, metadata_questions_list


@dataclass(frozen=True)
class PreviousExamQuestions:
    questions: List[dict]
    questions_by_id: Dict[str, dict]

DEFAULT_FEEDBACK_PATH = Path("web_human_study/backend/feedback.json")


def load_feedback_question_ids(feedback_path: Path) -> Set[str]:
    payload = json.loads(feedback_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("feedback.json must contain a list")
    question_ids: Set[str] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise TypeError("feedback.json entries must be objects")
        question_id = entry["question_ref"] if "question_ref" in entry else entry["question_id"]
        if not isinstance(question_id, str) or not question_id:
            raise TypeError("feedback.json question_id must be a non-empty string")
        question_ids.add(question_id)
    return question_ids


def load_previous_exam_questions(metadata_path: Path) -> PreviousExamQuestions:
    payload = load_metadata_payload(metadata_path)
    questions = metadata_questions_list(payload)
    questions_by_id: Dict[str, dict] = {}
    for question in questions:
        question_id = question["question_id"]
        if question_id in questions_by_id:
            raise ValueError(f"Duplicate question_id in metadata: {question_id}")
        questions_by_id[question_id] = question
    return PreviousExamQuestions(questions=questions, questions_by_id=questions_by_id)


__all__ = [
    "DEFAULT_FEEDBACK_PATH",
    "PreviousExamQuestions",
    "load_feedback_question_ids",
    "load_previous_exam_questions",
]
