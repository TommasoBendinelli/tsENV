from __future__ import annotations

from collections.abc import Sequence


def sample_score(predicted_labels: Sequence[str], correct_label: str) -> float:
    """Score one ranked-label shortlist sample using the paper convention."""
    correct = str(correct_label).strip()
    if not correct:
        raise ValueError("correct_label must be non-empty")

    predicted: list[str] = []
    for label in predicted_labels:
        text = str(label).strip()
        if not text:
            continue
        if text not in predicted:
            predicted.append(text)
    if not predicted:
        return 0.0
    if correct not in predicted:
        return 0.0
    return 1.0 / float(len(predicted))


__all__ = ["sample_score"]
