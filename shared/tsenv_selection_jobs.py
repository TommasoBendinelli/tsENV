from __future__ import annotations

from typing import List, Sequence, Tuple


def resolve_selection_job_ids(
    *, selection_seeds: Sequence[int], shot_ids: Sequence[str]
) -> List[Tuple[int, str]]:
    resolved_seeds = [int(seed) for seed in selection_seeds]
    if not resolved_seeds:
        return []
    normalized_shot_ids = [str(shot_id or "").strip() for shot_id in shot_ids]
    jobs: List[Tuple[int, str]] = []
    for seed in resolved_seeds:
        for shot_id in normalized_shot_ids:
            if not shot_id:
                continue
            jobs.append((int(seed), shot_id))
    return jobs
