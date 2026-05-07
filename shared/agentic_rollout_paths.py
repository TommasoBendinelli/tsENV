from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "terminal-bench" / "runs"
CANONICAL_AGENT_LOGS_DIRNAME = "agent_logs"
CANONICAL_ATIF_PROCESSED_DIRNAME = "atif_processed"
CANONICAL_FINAL_RESPONSE_FILENAME = "agentic-final-response.json"
TRAJECTORY_FILENAME = "atif_trajectory.json"
LIGHT_TRAJECTORY_FILENAME = "atif_trajectory_light.json"


def canonical_agent_logs_dir(trial_dir: Path) -> Path:
    return trial_dir / CANONICAL_AGENT_LOGS_DIRNAME


def resolve_agent_logs_dir(trial_dir: Path) -> Path:
    return canonical_agent_logs_dir(trial_dir)


def canonical_atif_processed_dir(trial_dir: Path) -> Path:
    return canonical_agent_logs_dir(trial_dir) / CANONICAL_ATIF_PROCESSED_DIRNAME


def canonical_trajectory_path(trial_dir: Path) -> Path:
    return canonical_atif_processed_dir(trial_dir) / TRAJECTORY_FILENAME


def _resolve_run_dir(agentic_run_id: str) -> Path:
    run_id = str(agentic_run_id).strip()
    if not run_id:
        raise ValueError("agentic_run_id must be a non-empty string")
    run_dir = (RUNS_ROOT / run_id).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def _candidate_trial_dirs(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for task_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        candidates.extend(sorted(path for path in task_dir.iterdir() if path.is_dir()))
    return candidates


def _resolve_single_trial_dir(agentic_run_id: str) -> Path:
    run_dir = _resolve_run_dir(agentic_run_id)
    trial_dirs = _candidate_trial_dirs(run_dir)
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories found under run directory: {run_dir}")
    if len(trial_dirs) != 1:
        rendered = ", ".join(str(path.relative_to(run_dir)) for path in trial_dirs[:5])
        if len(trial_dirs) > 5:
            rendered += ", ..."
        raise ValueError(
            f"Expected exactly one trial directory under {run_dir}, found {len(trial_dirs)}: {rendered}"
        )
    return trial_dirs[0]


def resolve_trajectory_path(agentic_run_id: str) -> Path:
    trial_dir = _resolve_single_trial_dir(agentic_run_id)
    trajectory_path = canonical_trajectory_path(trial_dir)
    if trajectory_path.is_file():
        return trajectory_path
    raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")


def canonical_light_trajectory_path(trial_dir: Path) -> Path:
    return canonical_atif_processed_dir(trial_dir) / LIGHT_TRAJECTORY_FILENAME


def resolve_light_trajectory_path(trial_dir: Path) -> Path:
    return canonical_light_trajectory_path(trial_dir)


def canonical_final_response_path(trial_dir: Path) -> Path:
    return trial_dir / CANONICAL_FINAL_RESPONSE_FILENAME


def resolve_final_response_path(trial_dir: Path) -> Path:
    return canonical_final_response_path(trial_dir)
