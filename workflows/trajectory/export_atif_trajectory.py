#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import click

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HARBOR_SRC = REPO_ROOT / "harbor" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(HARBOR_SRC) not in sys.path:
    sys.path.append(str(HARBOR_SRC))
TERMINAL_BENCH_ROOT = REPO_ROOT / "terminal-bench"
if str(TERMINAL_BENCH_ROOT) not in sys.path:
    sys.path.append(str(TERMINAL_BENCH_ROOT))
HARBOR_PACKAGE = HARBOR_SRC / "harbor"
try:
    if not HARBOR_PACKAGE.is_dir():
        raise ModuleNotFoundError(f"Harbor source package not found at {HARBOR_PACKAGE}")
    if "harbor" not in sys.modules:
        harbor_module = types.ModuleType("harbor")
        harbor_module.__path__ = [str(HARBOR_PACKAGE)]
        sys.modules["harbor"] = harbor_module

    from harbor.models.trajectories import (
        Agent,
        FinalMetrics,
        Metrics,
        Observation,
        Step,
        ToolCall,
        Trajectory,
    )
    from harbor.utils.trajectory_utils import format_trajectory_json
    from harbor.utils.trajectory_validator import TrajectoryValidator
except ModuleNotFoundError:
    class _FallbackModel:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_copy(self, *, deep: bool = False) -> Any:
            if not deep:
                return self.__class__(**self.__dict__)
            return self.__class__(**json.loads(json.dumps(self.model_dump(), allow_nan=True)))

        def model_dump(self) -> dict[str, Any]:
            return {
                key: _fallback_to_json(value)
                for key, value in self.__dict__.items()
                if value is not None
            }

    def _fallback_to_json(value: Any) -> Any:
        if isinstance(value, _FallbackModel):
            return value.model_dump()
        if isinstance(value, list):
            return [_fallback_to_json(item) for item in value]
        if isinstance(value, dict):
            return {key: _fallback_to_json(item) for key, item in value.items()}
        return value

    class Agent(_FallbackModel):
        def __init__(self, name: str, version: str) -> None:
            super().__init__(name=name, version=version)

    class FinalMetrics(_FallbackModel):
        def __init__(
            self,
            *,
            total_prompt_tokens: int = 0,
            total_completion_tokens: int = 0,
            total_cached_tokens: int = 0,
            total_cost_usd: float = 0.0,
            total_steps: int = 0,
            total_tool_duration: float | None = None,
            total_duration: float | None = None,
        ) -> None:
            super().__init__(
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_cached_tokens=total_cached_tokens,
                total_cost_usd=total_cost_usd,
                total_steps=total_steps,
                total_tool_duration=total_tool_duration,
                total_duration=total_duration,
            )

    class Metrics(_FallbackModel):
        def __init__(
            self,
            *,
            prompt_tokens: int | None = None,
            completion_tokens: int | None = None,
            cached_tokens: int | None = None,
        ) -> None:
            super().__init__(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )

    class Observation(_FallbackModel):
        def __init__(
            self,
            *,
            source_call_id: str | None,
            content: str | None,
            timestamp: str | None,
            duration: float | None = None,
        ) -> None:
            super().__init__(
                source_call_id=source_call_id,
                content=content,
                timestamp=timestamp,
                duration=duration,
            )

    class ToolCall(_FallbackModel):
        def __init__(
            self,
            *,
            tool_call_id: str,
            function_name: str,
            arguments: dict[str, Any],
        ) -> None:
            super().__init__(
                tool_call_id=tool_call_id,
                function_name=function_name,
                arguments=arguments,
            )

    class Step(_FallbackModel):
        def __init__(
            self,
            *,
            step_id: int,
            timestamp: str | None,
            source: str,
            message: str,
            reasoning_content: str | None = None,
            tool_calls: list[ToolCall] | None = None,
            observation: list[Observation] | None = None,
            metrics: Metrics | None = None,
            duration: float | None = None,
        ) -> None:
            super().__init__(
                step_id=step_id,
                timestamp=timestamp,
                duration=duration,
                source=source,
                message=message,
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                observation=observation,
                metrics=metrics,
            )

    class Trajectory(_FallbackModel):
        def __init__(
            self,
            *,
            session_id: str,
            agent: Agent,
            steps: list[Step],
            final_metrics: FinalMetrics,
            is_context_compacted: bool = False,
        ) -> None:
            super().__init__(
                agent=agent,
                session_id=session_id,
                final_metrics=final_metrics,
                is_context_compacted=is_context_compacted,
                steps=steps,
            )

        def to_json_dict(self) -> dict[str, Any]:
            return self.model_dump()

    def format_trajectory_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True)

    class TrajectoryValidator:
        def validate(self, _path: Path) -> bool:
            return True

        def get_errors(self) -> list[str]:
            return []
from shared.agentic_rollout_paths import (
    LIGHT_TRAJECTORY_FILENAME,
    RUNS_ROOT,
    TRAJECTORY_FILENAME,
    canonical_atif_processed_dir,
    canonical_agent_logs_dir,
    canonical_trajectory_path,
)
from shared.benchmark_utils import AGENTIC_PROFILES, AGENTIC_PROFILES_BY_ID


PROFILE_COMPATIBILITY_ALIASES = {
    "minimax-m2.7_low": "minimax-m2.7",
}


SOURCE_ROLE_MAP = {
    "assistant": "AGENT",
    "agent": "AGENT",
    "developer": "SYSTEM",
    "system": "SYSTEM",
    "user": "USER",
}
EXPORTED_AGENT_VERSION = "1.00"


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    has_usage: bool = False


@dataclass
class TrajectoryTarget:
    source_type: str
    rollout_path: Path | None
    gemini_session_path: Path | None
    episode_root: Path | None
    opencode_storage: Path | None
    opencode_session_path: Path | None
    output_path: Path
    session_id: str
    agent_profile_id: str
    model_name: str | None
    exported_agent_id: str
    question_id: str
    tag: str


def read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON in {path}, got {type(payload)}")
    return payload


def _required_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} missing or empty in {path}")
    return value


def read_scenario_export_metadata(scenario_info_path: Path) -> tuple[str, str, str]:
    if not scenario_info_path.is_file():
        raise FileNotFoundError(f"scenario_info.json not found at {scenario_info_path}")
    scenario_info = read_json_dict(scenario_info_path)
    agent_id = _required_str(scenario_info, "agent_id", scenario_info_path)
    question_id = _required_str(scenario_info, "question_id", scenario_info_path)
    tag = _required_str(scenario_info, "tag", scenario_info_path)
    return agent_id, question_id, tag


def find_nearby_scenario_info_path(input_path: Path) -> Path | None:
    start = input_path.parent if input_path.is_file() else input_path
    for candidate_dir in (start, *start.parents):
        candidate = candidate_dir / "scenario_info.json"
        if candidate.is_file():
            return candidate
    return None


def _profile_id_from_scenario_agent_id(agent_id: str, scenario_info_path: Path) -> str:
    profile_id = PROFILE_COMPATIBILITY_ALIASES.get(agent_id, agent_id)
    if profile_id not in AGENTIC_PROFILES_BY_ID:
        raise ValueError(
            f"Unknown agent_id {agent_id!r} in {scenario_info_path}. "
            f"Expected a valid agent_id from {REPO_ROOT / 'shared' / 'config' / 'agents.json'}."
        )
    return profile_id


def resolve_run_dir(input_path: Path) -> Path:
    if input_path.is_file():
        if input_path.name != "results.json":
            raise ValueError(f"Expected results.json file, got {input_path}")
        return input_path.parent
    return input_path


def resolve_input_argument(input_path: Path) -> Path:
    if input_path.exists():
        return input_path
    if not input_path.is_absolute() and len(input_path.parts) == 1:
        run_dir = RUNS_ROOT / input_path
        if run_dir.is_dir():
            return run_dir
    raise click.ClickException(
        f"Input is neither an existing path nor a run id under {RUNS_ROOT}: {input_path}"
    )


def resolve_trial_dir(run_dir: Path, result: dict[str, Any]) -> Path | None:
    task_id = result["task_id"]
    trial_name = result["trial_name"]
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Invalid task_id in results: {task_id}")
    if not isinstance(trial_name, str) or not trial_name:
        raise ValueError(f"Invalid trial_name in results: {trial_name}")
    trial_dir = run_dir / task_id / trial_name
    if not trial_dir.is_dir():
        click.echo(f"WARNING: Trial directory not found: {trial_dir}", err=True)
        return None
    return trial_dir


def find_single_rollout(rollout_root: Path, agent_name: str | None) -> Path | None:
    candidates = sorted(rollout_root.glob("rollout-*.jsonl"))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if agent_name == "codex":
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        click.echo(
            f"WARNING: Multiple rollout jsonl files under {rollout_root}; using newest: {newest.name}",
            err=True,
        )
        return newest
    raise ValueError(
        f"Expected exactly one rollout jsonl under {rollout_root}, found {len(candidates)}"
    )


def find_claude_rollout(agent_logs_dir: Path) -> Path | None:
    projects_dir = agent_logs_dir / ".claude" / "projects"
    if not projects_dir.is_dir():
        return None
    candidates = sorted(projects_dir.glob("*.jsonl"))
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one Claude jsonl under {projects_dir}, found {len(candidates)}"
        )
    return candidates[0]


def is_claude_rollout_path(rollout_path: Path) -> bool:
    parts = rollout_path.parts
    return ".claude" in parts and "projects" in parts


def infer_rollout_source_type(rollout_path: Path) -> str:
    if is_claude_rollout_path(rollout_path):
        return "claude"
    return "codex"


def _gemini_session_timestamp(session_path: Path) -> dt.datetime:
    stem = session_path.stem
    cleaned = re.sub(r"-[0-9a-fA-F]{8}$", "", stem)
    cleaned = cleaned[:-1] if cleaned.endswith("-") else cleaned
    if not cleaned.startswith("session-"):
        raise ValueError(f"Gemini session filename missing timestamp: {session_path}")
    for fmt in ("session-%Y-%m-%dT%H-%M-%S", "session-%Y-%m-%dT%H-%M"):
        try:
            return dt.datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise ValueError(f"Gemini session filename has invalid timestamp: {session_path}")


def find_gemini_session(rollout_root: Path) -> Path | None:
    candidates = sorted(rollout_root.glob("session-*.json"))
    if not candidates:
        return None
    if len(candidates) != 1:
        return min(candidates, key=_gemini_session_timestamp)
    return candidates[0]


def find_opencode_storage(agent_logs_dir: Path) -> Path | None:
    candidates = (
        agent_logs_dir / ".local" / "share" / "opencode" / "storage",
        agent_logs_dir / "opencode" / "storage",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def find_opencode_session_json(agent_logs_dir: Path) -> Path | None:
    candidates = sorted(agent_logs_dir.glob("opencode-session-*.json"))
    if not candidates:
        return None
    if len(candidates) != 1:
        rendered = ", ".join(path.name for path in candidates[:5])
        if len(candidates) > 5:
            rendered += ", ..."
        raise ValueError(
            f"Expected exactly one OpenCode session JSON under {agent_logs_dir}, "
            f"found {len(candidates)}: {rendered}"
        )
    return candidates[0]


def _normalize_reasoning(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower()
    return value or None


def _matching_agent_profiles(
    platform_name: str,
    model_name: str | None,
    reasoning: str | None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for profile in AGENTIC_PROFILES:
        platform = profile.get("agentic_platform") or {}
        if str(platform.get("name") or "") != platform_name:
            continue
        if model_name is not None and str(profile.get("model_name") or "") != model_name:
            continue
        if reasoning is not None and str(profile.get("reasoning") or "") != reasoning:
            continue
        matches.append(dict(profile))
    return matches


def _extract_run_reasoning(run_metadata: dict[str, Any]) -> str | None:
    direct = _normalize_reasoning(run_metadata.get("reasoning"))
    if direct is not None:
        return direct
    agent_kwargs = run_metadata.get("agent_kwargs")
    if isinstance(agent_kwargs, dict):
        for key in ("reasoning_effort", "reasoning"):
            value = _normalize_reasoning(agent_kwargs.get(key))
            if value is not None:
                return value
    return None


def _resolve_explicit_agent_profile_id(
    agent_name: str | None,
    model_name: str | None,
    reasoning: str | None,
) -> str | None:
    if agent_name is None:
        return None
    candidate = str(agent_name).strip()
    if not candidate:
        return None
    if candidate in AGENTIC_PROFILES_BY_ID:
        return candidate
    matches = _matching_agent_profiles(candidate, model_name, reasoning)
    if len(matches) == 1:
        return str(matches[0]["agent_id"])
    if not matches:
        raise ValueError(
            f"Unknown agent profile override {candidate!r}. Pass a valid agent_id from "
            f"{REPO_ROOT / 'shared' / 'config' / 'agents.json'}."
        )
    match_ids = ", ".join(str(match["agent_id"]) for match in matches)
    raise ValueError(
        f"Ambiguous agent override {candidate!r} for model {model_name!r}. "
        f"Matching agent_ids: {match_ids}. Pass an explicit agent_id."
    )


def _resolve_run_agent_profile_id(
    run_metadata: dict[str, Any],
    agent_name: str | None,
    model_name: str | None,
    reasoning: str | None,
) -> str:
    explicit_profile_id = _resolve_explicit_agent_profile_id(agent_name, model_name, reasoning)
    if explicit_profile_id is not None:
        return explicit_profile_id
    metadata_agent_id = str(run_metadata.get("agent_id") or "").strip()
    if metadata_agent_id:
        if metadata_agent_id not in AGENTIC_PROFILES_BY_ID:
            raise ValueError(
                f"Unknown agent_id {metadata_agent_id!r} in {run_metadata.get('run_id')!r}"
            )
        return metadata_agent_id
    run_id = str(run_metadata.get("run_id") or "").strip()
    run_id_matches = [
        agent_id
        for agent_id in AGENTIC_PROFILES_BY_ID
        if re.search(rf"(^|__){re.escape(agent_id)}($|__)", run_id)
    ]
    if len(run_id_matches) == 1:
        return run_id_matches[0]
    resolved_agent_name = str(run_metadata.get("agent_name") or "").strip()
    if not resolved_agent_name:
        raise ValueError("agent_name missing in run_metadata and no override was provided")
    matches = _matching_agent_profiles(resolved_agent_name, model_name, reasoning)
    if len(matches) == 1:
        return str(matches[0]["agent_id"])
    if not matches:
        raise ValueError(
            f"No agent profile matches platform={resolved_agent_name!r}, "
            f"model={model_name!r}, reasoning={reasoning!r}."
        )
    match_ids = ", ".join(str(match["agent_id"]) for match in matches)
    raise ValueError(
        f"Ambiguous agent profile for run {run_id!r}: "
        f"platform={resolved_agent_name!r}, model={model_name!r}, reasoning={reasoning!r}. "
        f"Matching agent_ids: {match_ids}. Add agent_id to run_metadata.json or pass an explicit agent_id."
    )


def resolve_agent_fields(
    run_metadata: dict[str, Any],
    agent_name: str | None,
    model_name: str | None,
) -> tuple[str, str | None]:
    resolved_model_name = model_name
    if resolved_model_name is None and "model_name" in run_metadata:
        resolved_model_name = str(run_metadata["model_name"] or "").strip() or None
    reasoning = _extract_run_reasoning(run_metadata)
    resolved_agent_profile_id = _resolve_run_agent_profile_id(
        run_metadata,
        agent_name,
        resolved_model_name,
        reasoning,
    )
    return resolved_agent_profile_id, resolved_model_name


def build_targets_from_run(
    run_dir: Path,
    agent_name: str | None,
    model_name: str | None,
) -> list[TrajectoryTarget]:
    run_metadata_path = run_dir / "run_metadata.json"
    results_path = run_dir / "results.json"
    if not run_metadata_path.is_file():
        raise FileNotFoundError(f"run_metadata.json not found at {run_metadata_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"results.json not found at {results_path}")

    run_metadata = read_json_dict(run_metadata_path)
    results_payload = read_json_dict(results_path)
    results = results_payload["results"]
    if not isinstance(results, list):
        raise ValueError(f"results.json does not contain a 'results' list: {results_path}")

    resolved_model_name = model_name
    if resolved_model_name is None and "model_name" in run_metadata:
        resolved_model_name = str(run_metadata["model_name"] or "").strip() or None
    run_id = str(run_metadata.get("run_id") or "").strip()
    if not run_id:
        raise ValueError(f"run_id missing in {run_metadata_path}")

    targets: list[TrajectoryTarget] = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict result entry, got {type(result)}")
        trial_dir = resolve_trial_dir(run_dir, result)
        if trial_dir is None:
            continue
        agent_logs_dir = canonical_agent_logs_dir(trial_dir)
        if not agent_logs_dir.is_dir():
            click.echo(f"WARNING: agent_logs not found at {agent_logs_dir}", err=True)
            continue
        scenario_info_path = trial_dir / "scenario_info.json"
        exported_agent_id, question_id, tag = read_scenario_export_metadata(scenario_info_path)
        resolved_agent_profile_id = _profile_id_from_scenario_agent_id(
            exported_agent_id,
            scenario_info_path,
        )
        output_path = canonical_trajectory_path(trial_dir)
        session_id = run_id
        rollout_path = find_single_rollout(agent_logs_dir, run_metadata.get("agent_name"))
        if rollout_path is not None:
            targets.append(
                TrajectoryTarget(
                    source_type=infer_rollout_source_type(rollout_path),
                    rollout_path=rollout_path,
                    gemini_session_path=None,
                    episode_root=None,
                    opencode_storage=None,
                    opencode_session_path=None,
                    output_path=output_path,
                    session_id=session_id,
                    agent_profile_id=resolved_agent_profile_id,
                    model_name=resolved_model_name,
                    exported_agent_id=exported_agent_id,
                    question_id=question_id,
                    tag=tag,
                )
            )
            continue
        claude_rollout = find_claude_rollout(agent_logs_dir)
        if claude_rollout is not None:
            targets.append(
                TrajectoryTarget(
                    source_type="claude",
                    rollout_path=claude_rollout,
                    gemini_session_path=None,
                    episode_root=None,
                    opencode_storage=None,
                    opencode_session_path=None,
                    output_path=output_path,
                    session_id=session_id,
                    agent_profile_id=resolved_agent_profile_id,
                    model_name=resolved_model_name,
                    exported_agent_id=exported_agent_id,
                    question_id=question_id,
                    tag=tag,
                )
            )
            continue
        gemini_session_path = find_gemini_session(agent_logs_dir)
        if gemini_session_path is not None:
            targets.append(
                TrajectoryTarget(
                    source_type="gemini",
                    rollout_path=None,
                    gemini_session_path=gemini_session_path,
                    episode_root=None,
                    opencode_storage=None,
                    opencode_session_path=None,
                    output_path=output_path,
                    session_id=session_id,
                    agent_profile_id=resolved_agent_profile_id,
                    model_name=resolved_model_name,
                    exported_agent_id=exported_agent_id,
                    question_id=question_id,
                    tag=tag,
                )
            )
            continue
        opencode_storage = find_opencode_storage(agent_logs_dir)
        if opencode_storage is not None:
            targets.append(
                TrajectoryTarget(
                    source_type="opencode",
                    rollout_path=None,
                    gemini_session_path=None,
                    episode_root=None,
                    opencode_storage=opencode_storage,
                    opencode_session_path=None,
                    output_path=output_path,
                    session_id=session_id,
                    agent_profile_id=resolved_agent_profile_id,
                    model_name=resolved_model_name,
                    exported_agent_id=exported_agent_id,
                    question_id=question_id,
                    tag=tag,
                )
            )
            continue
        opencode_session_path = find_opencode_session_json(agent_logs_dir)
        if opencode_session_path is not None:
            targets.append(
                TrajectoryTarget(
                    source_type="opencode_session_json",
                    rollout_path=None,
                    gemini_session_path=None,
                    episode_root=None,
                    opencode_storage=None,
                    opencode_session_path=opencode_session_path,
                    output_path=output_path,
                    session_id=session_id,
                    agent_profile_id=resolved_agent_profile_id,
                    model_name=resolved_model_name,
                    exported_agent_id=exported_agent_id,
                    question_id=question_id,
                    tag=tag,
                )
            )
            continue
        episode_root = agent_logs_dir
        episode_dirs = sorted(episode_root.glob("episode-*"))
        if not episode_dirs:
            click.echo(
                f"WARNING: No rollout jsonl or episode logs under {agent_logs_dir}",
                err=True,
            )
            continue
        targets.append(
            TrajectoryTarget(
                source_type="terminus",
                rollout_path=None,
                gemini_session_path=None,
                episode_root=episode_root,
                opencode_storage=None,
                opencode_session_path=None,
                output_path=output_path,
                session_id=session_id,
                agent_profile_id=resolved_agent_profile_id,
                model_name=resolved_model_name,
                exported_agent_id=exported_agent_id,
                question_id=question_id,
                tag=tag,
            )
        )
    return targets


def build_target_from_rollout(
    rollout_path: Path,
    output_path: Path | None,
    session_id: str | None,
    agent_name: str | None,
    model_name: str | None,
) -> TrajectoryTarget:
    if agent_name is None:
        raise ValueError("agent_name is required when converting a rollout file directly")
    resolved_agent_profile_id = _resolve_explicit_agent_profile_id(
        agent_name,
        model_name,
        None,
    )
    if resolved_agent_profile_id is None:
        raise ValueError("Unable to resolve agent profile ID for direct rollout conversion")
    scenario_info_path = find_nearby_scenario_info_path(rollout_path)
    if scenario_info_path is None:
        raise FileNotFoundError(
            f"scenario_info.json not found near direct rollout input {rollout_path}"
        )
    exported_agent_id, question_id, tag = read_scenario_export_metadata(scenario_info_path)
    _profile_id_from_scenario_agent_id(exported_agent_id, scenario_info_path)
    if exported_agent_id != resolved_agent_profile_id:
        raise ValueError(
            f"scenario_info.json agent_id {exported_agent_id!r} does not match "
            f"resolved agent profile {resolved_agent_profile_id!r}"
        )
    resolved_output_path = resolve_output_path(rollout_path, output_path)
    resolved_session_id = session_id or rollout_path.stem
    source_type = infer_rollout_source_type(rollout_path)
    return TrajectoryTarget(
        source_type=source_type,
        rollout_path=rollout_path,
        gemini_session_path=None,
        episode_root=None,
        opencode_storage=None,
        opencode_session_path=None,
        output_path=resolved_output_path,
        session_id=resolved_session_id,
        agent_profile_id=resolved_agent_profile_id,
        model_name=model_name,
        exported_agent_id=exported_agent_id,
        question_id=question_id,
        tag=tag,
    )


def collect_targets(
    paths: Sequence[Path],
    output_path: Path | None,
    session_id: str | None,
    agent_name: str | None,
    model_name: str | None,
) -> list[TrajectoryTarget]:
    targets: list[TrajectoryTarget] = []
    for raw_input_path in paths:
        input_path = resolve_input_argument(raw_input_path)
        if input_path.is_file():
            if input_path.suffix == ".jsonl":
                targets.append(
                    build_target_from_rollout(
                        input_path,
                        output_path,
                        session_id,
                        agent_name,
                        model_name,
                    )
                )
            elif input_path.suffix == ".json" and input_path.name.startswith("session-"):
                if agent_name is None:
                    raise ValueError("agent_name is required when converting a gemini session file directly")
                if session_id is None:
                    session_id = input_path.stem
                resolved_agent_profile_id = _resolve_explicit_agent_profile_id(
                    agent_name,
                    model_name,
                    None,
                )
                if resolved_agent_profile_id is None:
                    raise ValueError("Unable to resolve agent profile ID for direct gemini conversion")
                scenario_info_path = find_nearby_scenario_info_path(input_path)
                if scenario_info_path is None:
                    raise FileNotFoundError(
                        f"scenario_info.json not found near direct gemini input {input_path}"
                    )
                exported_agent_id, question_id, tag = read_scenario_export_metadata(
                    scenario_info_path
                )
                _profile_id_from_scenario_agent_id(exported_agent_id, scenario_info_path)
                if exported_agent_id != resolved_agent_profile_id:
                    raise ValueError(
                        f"scenario_info.json agent_id {exported_agent_id!r} does not match "
                        f"resolved agent profile {resolved_agent_profile_id!r}"
                    )
                targets.append(
                    TrajectoryTarget(
                        source_type="gemini",
                        rollout_path=None,
                        gemini_session_path=input_path,
                        episode_root=None,
                        opencode_storage=None,
                        opencode_session_path=None,
                        output_path=resolve_output_path(input_path, output_path),
                        session_id=session_id,
                        agent_profile_id=resolved_agent_profile_id,
                        model_name=model_name,
                        exported_agent_id=exported_agent_id,
                        question_id=question_id,
                        tag=tag,
                    )
                )
            elif input_path.name == "results.json":
                targets.extend(
                    build_targets_from_run(
                        input_path.parent,
                        agent_name,
                        model_name,
                    )
                )
            else:
                raise ValueError(f"Unsupported file input: {input_path}")
        else:
            run_dir = resolve_run_dir(input_path)
            targets.extend(
                build_targets_from_run(
                    run_dir,
                    agent_name,
                    model_name,
                )
            )
    return targets


def extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                if block:
                    parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                parts.append(text)
                continue
            rendered = json.dumps(block, ensure_ascii=False, sort_keys=True)
            if rendered != "{}":
                parts.append(rendered)
        return "\n".join(part for part in parts if part)
    return ""


def normalize_source(role: str) -> str:
    if not isinstance(role, str):
        return "AGENT"
    return SOURCE_ROLE_MAP.get(role.lower(), "AGENT")


def normalize_arguments(arguments_raw: Any) -> dict[str, Any]:
    if isinstance(arguments_raw, dict):
        return arguments_raw
    if isinstance(arguments_raw, str):
        parsed = json.loads(arguments_raw)
        if isinstance(parsed, dict):
            return parsed
        return {"raw": parsed}
    if arguments_raw is None:
        return {}
    return {"raw": arguments_raw}


def normalize_codex_tool_arguments(function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if function_name == "exec_command":
        command = arguments.get("cmd")
        if isinstance(command, str) and command:
            return {"command": command}
    if function_name == "write_stdin":
        keystrokes = arguments.get("chars")
        if isinstance(keystrokes, str):
            return {"keystrokes": keystrokes}
    if function_name == "apply_patch":
        command = arguments.get("input")
        if isinstance(command, str) and command:
            return {"command": command}
    if function_name == "view_image":
        path = arguments.get("path")
        if isinstance(path, str) and path:
            return {"command": path}
    return arguments


def extract_command_label(arguments: dict[str, Any]) -> str | None:
    command = None
    if "command" in arguments and isinstance(arguments["command"], str):
        command = arguments["command"]
    if command is None and "cmd" in arguments and isinstance(arguments["cmd"], str):
        command = arguments["cmd"]
    if command is None:
        return None
    workdir = None
    if "workdir" in arguments and isinstance(arguments["workdir"], str):
        workdir = arguments["workdir"]
    if workdir is None and "cwd" in arguments and isinstance(arguments["cwd"], str):
        workdir = arguments["cwd"]
    if workdir:
        return f"(cwd: {workdir}) {command}"
    return command


def extract_model_name(item: dict[str, Any], payload: dict[str, Any]) -> str | None:
    if "model" in item and isinstance(item["model"], str):
        return item["model"]
    if "model_name" in item and isinstance(item["model_name"], str):
        return item["model_name"]
    if "model" in payload and isinstance(payload["model"], str):
        return payload["model"]
    if "model_name" in payload and isinstance(payload["model_name"], str):
        return payload["model_name"]
    return None


def extract_output_content(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        if "output" in raw:
            output = raw["output"]
            if isinstance(output, str):
                return output
            return json.dumps(output, ensure_ascii=False)
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _path_timestamp_to_iso(path: Path) -> str:
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).isoformat()


def _coalesce_timestamp(*values: object, fallback_path: Path | None = None) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    if fallback_path is not None:
        return _path_timestamp_to_iso(fallback_path)
    return None


def _parse_iso_timestamp(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _duration_seconds(start: str | None, end: str | None, *, label: str = "duration") -> float | None:
    start_time = _parse_iso_timestamp(start)
    end_time = _parse_iso_timestamp(end)
    if start_time is None or end_time is None:
        return None
    duration = (end_time - start_time).total_seconds()
    if duration < 0:
        raise ValueError(f"{label} cannot be negative")
    return duration


def _sum_documented_durations(values: Iterable[Any], *, label: str) -> float:
    total = 0.0
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be numeric")
        if value < 0:
            raise ValueError(f"{label} cannot be negative")
        total += float(value)
    return total


def _validate_documented_sources(trajectory: Trajectory) -> None:
    if not trajectory.steps:
        raise ValueError("ATIF trajectory must contain at least one step")
    user_step_ids = [step.step_id for step in trajectory.steps if step.source == "USER"]
    if trajectory.steps[0].source != "USER":
        raise ValueError("ATIF trajectory first step must have source USER")
    if len(user_step_ids) != 1:
        raise ValueError(
            "ATIF trajectory must contain exactly one USER step; "
            f"found {len(user_step_ids)} at step_ids={user_step_ids}"
        )


def _renumber_steps(trajectory: Trajectory) -> None:
    for index, step in enumerate(trajectory.steps, 1):
        step.step_id = index


def _discard_leading_system_steps(trajectory: Trajectory) -> None:
    first_user_index = next(
        (index for index, step in enumerate(trajectory.steps) if step.source == "USER"),
        None,
    )
    if first_user_index is None:
        return
    leading_steps = trajectory.steps[:first_user_index]
    if any(step.source != "SYSTEM" for step in leading_steps):
        return
    if first_user_index:
        trajectory.steps = trajectory.steps[first_user_index:]
        _renumber_steps(trajectory)


def _keep_last_user_message(trajectory: Trajectory) -> None:
    user_indices = [
        index for index, step in enumerate(trajectory.steps) if step.source == "USER"
    ]
    if len(user_indices) <= 1:
        return
    trajectory.steps = trajectory.steps[user_indices[-1] :]
    _renumber_steps(trajectory)


def finalize_documented_trajectory(trajectory: Trajectory) -> Trajectory:
    _discard_leading_system_steps(trajectory)
    _keep_last_user_message(trajectory)
    _validate_documented_sources(trajectory)
    trajectory.final_metrics.total_steps = len(trajectory.steps)
    previous_timestamp: str | None = None
    for step in trajectory.steps:
        if step.source == "AGENT":
            step.duration = _duration_seconds(
                previous_timestamp,
                step.timestamp,
                label=f"step {step.step_id} duration",
            )
        else:
            step.duration = None

        if step.observation:
            for observation in step.observation:
                observation.duration = _duration_seconds(
                    step.timestamp,
                    observation.timestamp,
                    label=f"step {step.step_id} observation duration",
                )

        previous_timestamp = step.timestamp
    trajectory.final_metrics.total_duration = _documented_total_duration(
        trajectory.to_json_dict()
    )
    trajectory.final_metrics.total_tool_duration = _documented_total_tool_duration(
        trajectory.to_json_dict()
    )
    return trajectory


def _documented_total_duration(trajectory_payload: dict[str, Any]) -> float | None:
    steps = trajectory_payload.get("steps")
    if not isinstance(steps, list):
        return None
    return _sum_documented_durations(
        (
            step.get("duration")
            for step in steps
            if isinstance(step, dict) and step.get("source") == "AGENT"
        ),
        label="total_duration",
    )


def _documented_total_tool_duration(trajectory_payload: dict[str, Any]) -> float | None:
    steps = trajectory_payload.get("steps")
    if not isinstance(steps, list):
        return None
    durations: list[Any] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        observations = step.get("observation")
        if not isinstance(observations, list):
            continue
        durations.extend(
            observation.get("duration")
            for observation in observations
            if isinstance(observation, dict)
        )
    return _sum_documented_durations(durations, label="total_tool_duration")


def _has_context_compaction_artifacts(path: Path | None) -> bool:
    if path is None or not path.is_dir():
        return False
    return any(path.glob("trajectory.summarization-*-*.json")) or any(
        path.glob("trajectory.cont-*.json")
    )


def target_has_context_compaction(target: TrajectoryTarget) -> bool:
    candidate_dirs = [
        target.output_path.parent,
        target.episode_root,
        target.rollout_path.parent if target.rollout_path is not None else None,
        target.gemini_session_path.parent if target.gemini_session_path is not None else None,
        target.opencode_session_path.parent if target.opencode_session_path is not None else None,
    ]
    if target.opencode_storage is not None:
        candidate_dirs.append(target.opencode_storage)
    return any(_has_context_compaction_artifacts(path) for path in candidate_dirs)


def trajectory_to_documented_json_dict(
    trajectory: Trajectory,
    *,
    agent_id: str,
    question_id: str,
    tag: str,
) -> dict[str, Any]:
    payload = trajectory.to_json_dict()
    final_metrics = payload.get("final_metrics")
    if not isinstance(final_metrics, dict):
        final_metrics = {}
        payload["final_metrics"] = final_metrics
    payload["is_context_compacted"] = bool(payload.get("is_context_compacted", False))
    final_metrics["total_duration"] = _documented_total_duration(payload)
    final_metrics["total_tool_duration"] = _documented_total_tool_duration(payload)
    return {
        "tag": tag,
        "question_id": question_id,
        "agent_id": agent_id,
        "session_id": payload["session_id"],
        "final_metrics": final_metrics,
        "is_context_compacted": payload["is_context_compacted"],
        "steps": payload["steps"],
    }


def _append_observation_result(
    step: Step,
    source_call_id: str | None,
    content: str | None,
    timestamp: str | None,
) -> None:
    if step.observation is None:
        step.observation = [
            Observation(
                source_call_id=source_call_id,
                content=content,
                timestamp=timestamp,
            )
        ]
        return
    step.observation.append(
        Observation(
            source_call_id=source_call_id,
            content=content,
            timestamp=timestamp,
        )
    )


def _parse_codex_output_content(raw_output: Any) -> str | None:
    if isinstance(raw_output, str):
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            return raw_output
        return extract_output_content(parsed)
    return extract_output_content(raw_output)


def _codex_total_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    totals = info.get("total_token_usage")
    if not isinstance(totals, dict):
        return None
    input_tokens = coerce_int(totals.get("input_tokens"))
    cached_input_tokens = coerce_int(totals.get("cached_input_tokens"))
    output_tokens = coerce_int(totals.get("output_tokens"))
    reasoning_output_tokens = coerce_int(totals.get("reasoning_output_tokens"))
    if (
        input_tokens is None
        and cached_input_tokens is None
        and output_tokens is None
        and reasoning_output_tokens is None
    ):
        return None
    return {
        "input_tokens": input_tokens or 0,
        "cached_input_tokens": cached_input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "reasoning_output_tokens": reasoning_output_tokens or 0,
    }


def _codex_last_usage(payload: dict[str, Any]) -> dict[str, int] | None:
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    totals = info.get("last_token_usage")
    if not isinstance(totals, dict):
        return None
    input_tokens = coerce_int(totals.get("input_tokens"))
    cached_input_tokens = coerce_int(totals.get("cached_input_tokens"))
    output_tokens = coerce_int(totals.get("output_tokens"))
    reasoning_output_tokens = coerce_int(totals.get("reasoning_output_tokens"))
    total_tokens = coerce_int(totals.get("total_tokens"))
    if (
        input_tokens is None
        and cached_input_tokens is None
        and output_tokens is None
        and reasoning_output_tokens is None
        and total_tokens is None
    ):
        return None
    return {
        "input_tokens": input_tokens or 0,
        "cached_input_tokens": cached_input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "reasoning_output_tokens": reasoning_output_tokens or 0,
        "total_tokens": total_tokens or 0,
    }


def _pricing_for_agent_profile(agent_profile_id: str) -> tuple[float, float, float] | None:
    profile = AGENTIC_PROFILES_BY_ID.get(agent_profile_id)
    if profile is None:
        raise ValueError(f"Unknown agent profile ID for pricing: {agent_profile_id!r}")
    cost = profile.get("cost")
    if not isinstance(cost, dict):
        return None
    required_keys = ("input_token", "cached_token", "completion_token")
    if any(key not in cost for key in required_keys):
        return None
    return (
        float(cost["input_token"]),
        float(cost["cached_token"]),
        float(cost["completion_token"]),
    )


def _calculate_rfc_cost_usd(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    cost_per_input_token: float,
    cost_per_cached_token: float,
    cost_per_completion_token: float,
) -> float:
    non_cached_prompt_tokens = prompt_tokens - cached_tokens
    if non_cached_prompt_tokens < 0:
        raise ValueError("non_cached_total cannot be negative")
    return (
        (non_cached_prompt_tokens * cost_per_input_token)
        + (cached_tokens * cost_per_cached_token)
        + (completion_tokens * cost_per_completion_token)
    ) / 1_000_000


def _normalize_inclusive_prompt_tokens(*, prompt_tokens: int, cached_tokens: int) -> int:
    if cached_tokens > prompt_tokens:
        return prompt_tokens + cached_tokens
    return prompt_tokens


def _build_final_metrics_from_values(
    *,
    step_count: int,
    agent_profile_id: str,
    prompt_tokens: int = 0,
    cached_tokens: int = 0,
    completion_tokens: int = 0,
) -> FinalMetrics:
    prompt_tokens = _normalize_inclusive_prompt_tokens(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )
    if prompt_tokens - cached_tokens < 0:
        raise ValueError("non_cached_total cannot be negative")
    pricing = _pricing_for_agent_profile(agent_profile_id)
    total_cost_usd = float("nan")
    if pricing is not None:
        (
            cost_per_input_token,
            cost_per_cached_token,
            cost_per_completion_token,
        ) = pricing
        total_cost_usd = _calculate_rfc_cost_usd(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            cost_per_input_token=cost_per_input_token,
            cost_per_cached_token=cost_per_cached_token,
            cost_per_completion_token=cost_per_completion_token,
        )
    return FinalMetrics(
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_cached_tokens=cached_tokens,
        total_cost_usd=total_cost_usd,
        total_steps=step_count,
    )


def _build_codex_step_metrics(
    usage: dict[str, int] | None,
) -> Metrics | None:
    if usage is None:
        return None
    prompt_tokens = usage.get("input_tokens", 0)
    cached_tokens = usage.get("cached_input_tokens", 0)
    prompt_tokens = _normalize_inclusive_prompt_tokens(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )
    output_tokens = usage.get("output_tokens", 0)
    reasoning_tokens = usage.get("reasoning_output_tokens", 0)
    completion_tokens = output_tokens + reasoning_tokens

    if (
        prompt_tokens == 0
        and cached_tokens == 0
        and completion_tokens == 0
    ):
        return None

    return Metrics(
        prompt_tokens=prompt_tokens or None,
        completion_tokens=completion_tokens or None,
        cached_tokens=cached_tokens or None,
    )


def _assign_codex_step_metrics(
    pending_agent_steps: list[Step],
    step_metrics: Metrics | None,
) -> None:
    if step_metrics is None or not pending_agent_steps:
        return
    target_step = next(
        (
            step
            for step in pending_agent_steps
            if not step.tool_calls and (step.message or step.reasoning_content)
        ),
        pending_agent_steps[0],
    )
    target_step.metrics = step_metrics.model_copy(deep=True)


def _step_metric_total(steps: Iterable[Step], field_name: str) -> int:
    total = 0
    for step in steps:
        metrics = step.metrics
        if metrics is None:
            continue
        value = getattr(metrics, field_name)
        if value is not None:
            total += int(value)
    return total


def _validate_codex_step_metrics_match_final_metrics(
    steps: list[Step],
    final_metrics: FinalMetrics,
) -> None:
    comparisons = (
        (
            "prompt_tokens",
            final_metrics.total_prompt_tokens,
            _step_metric_total(steps, "prompt_tokens"),
        ),
        (
            "completion_tokens",
            final_metrics.total_completion_tokens,
            _step_metric_total(steps, "completion_tokens"),
        ),
        (
            "cached_tokens",
            final_metrics.total_cached_tokens,
            _step_metric_total(steps, "cached_tokens"),
        ),
    )
    mismatches: list[str] = []
    for field_name, expected, actual in comparisons:
        if actual != expected:
            mismatches.append(f"{field_name}: steps={actual}, final_metrics={expected}")
    if mismatches:
        raise ValueError(
            "Codex cumulative token totals do not match summed step metrics: "
            + "; ".join(mismatches)
        )


def build_codex_trajectory(
    rollout_path: Path,
    session_id: str,
    agent_profile_id: str,
    model_name: str | None,
) -> Trajectory:
    steps: list[Step] = []
    pending_calls: dict[str, Step] = {}
    resolved_model_name = model_name
    last_usage: dict[str, int] | None = None
    pending_agent_steps: list[Step] = []

    with rollout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            payload = item.get("payload")
            if not isinstance(payload, dict):
                continue

            item_type = item.get("type")
            payload_type = payload.get("type")
            timestamp = item.get("timestamp") if isinstance(item.get("timestamp"), str) else None

            if item_type == "session_meta":
                if resolved_model_name is None:
                    candidate = payload.get("model")
                    if not isinstance(candidate, str):
                        candidate = payload.get("model_name")
                    if isinstance(candidate, str) and candidate:
                        resolved_model_name = candidate
                continue

            if item_type == "event_msg" and payload_type == "token_count":
                step_metrics = _build_codex_step_metrics(
                    _codex_last_usage(payload),
                )
                _assign_codex_step_metrics(pending_agent_steps, step_metrics)
                pending_agent_steps = []
                last_usage = _codex_total_usage(payload) or last_usage
                continue

            if item_type != "response_item":
                continue

            if payload_type == "message":
                role = str(payload.get("role") or "").strip().lower()
                source = normalize_source(role)
                message = extract_message_text(payload.get("content"))
                if not message:
                    continue
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                        source=source,
                        message=message,
                    )
                )
                if source == "AGENT":
                    pending_agent_steps.append(steps[-1])
                continue

            if payload_type == "function_call":
                call_id = str(payload.get("call_id") or "").strip()
                function_name = str(payload.get("name") or "").strip()
                if not call_id or not function_name:
                    continue
                arguments = normalize_codex_tool_arguments(
                    function_name,
                    normalize_arguments(payload.get("arguments")),
                )
                step = Step(
                    step_id=len(steps) + 1,
                    timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                    source="AGENT",
                    message="",
                    tool_calls=[
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=function_name,
                            arguments=arguments,
                        )
                    ],
                )
                steps.append(step)
                pending_calls[call_id] = step
                pending_agent_steps.append(step)
                continue

            if payload_type == "custom_tool_call":
                call_id = str(payload.get("call_id") or "").strip()
                function_name = str(payload.get("name") or "").strip()
                if not call_id or not function_name:
                    continue
                arguments = {"input": payload.get("input")}
                status = payload.get("status")
                if status is not None:
                    arguments["status"] = status
                arguments = normalize_codex_tool_arguments(function_name, arguments)
                step = Step(
                    step_id=len(steps) + 1,
                    timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                    source="AGENT",
                    message="",
                    tool_calls=[
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=function_name,
                            arguments=arguments,
                        )
                    ],
                )
                steps.append(step)
                pending_calls[call_id] = step
                pending_agent_steps.append(step)
                continue

            if payload_type in {"function_call_output", "custom_tool_call_output"}:
                call_id = str(payload.get("call_id") or "").strip()
                if not call_id:
                    continue
                step = pending_calls.get(call_id)
                if step is None:
                    continue
                _append_observation_result(
                    step,
                    source_call_id=call_id,
                    content=_parse_codex_output_content(payload.get("output")),
                    timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                )

    if not steps:
        raise ValueError(f"Codex rollout did not contain any exportable steps: {rollout_path}")

    if last_usage is not None:
        reasoning_tokens = last_usage.get("reasoning_output_tokens", 0)
        total_prompt_tokens = last_usage.get("input_tokens", 0)
        total_cached_tokens = last_usage.get("cached_input_tokens", 0)
        total_completion_tokens = last_usage.get("output_tokens", 0) + reasoning_tokens
        final_metrics = _build_final_metrics_from_values(
            step_count=len(steps),
            agent_profile_id=agent_profile_id,
            prompt_tokens=total_prompt_tokens,
            cached_tokens=total_cached_tokens,
            completion_tokens=total_completion_tokens,
        )
    else:
        final_metrics = _build_final_metrics_from_values(
            step_count=len(steps),
            agent_profile_id=agent_profile_id,
        )

    trajectory = finalize_documented_trajectory(
        Trajectory(
            session_id=session_id,
            agent=Agent(
                name=agent_profile_id,
                version=EXPORTED_AGENT_VERSION,
            ),
            steps=steps,
            final_metrics=final_metrics,
        )
    )
    if last_usage is not None:
        _validate_codex_step_metrics_match_final_metrics(
            trajectory.steps,
            trajectory.final_metrics,
        )
    return trajectory


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def opencode_timestamp_to_iso(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return dt.datetime.fromtimestamp(
        timestamp_ms / 1000.0, tz=dt.timezone.utc
    ).isoformat()


def extract_opencode_model_name(message: dict[str, Any]) -> str | None:
    if "modelID" in message and isinstance(message["modelID"], str):
        model_id = message["modelID"]
        if "providerID" in message and isinstance(message["providerID"], str):
            provider_id = message["providerID"]
            if model_id.startswith(f"{provider_id}/"):
                return model_id
            return f"{provider_id}/{model_id}"
        return model_id
    return None


def update_usage_totals_from_opencode(totals: UsageTotals, message: dict[str, Any]) -> None:
    if "tokens" not in message or not isinstance(message["tokens"], dict):
        return
    tokens = message["tokens"]
    prompt_tokens = coerce_int(tokens["input"]) if "input" in tokens else None
    completion_tokens = coerce_int(tokens["output"]) if "output" in tokens else None
    cached_tokens = None
    if "cache" in tokens and isinstance(tokens["cache"], dict):
        cached_tokens = coerce_int(tokens["cache"]["read"]) if "read" in tokens["cache"] else None
    cost_usd = coerce_float(message["cost"]) if "cost" in message else None

    if prompt_tokens is not None:
        totals.prompt_tokens += prompt_tokens
        totals.has_usage = True
    if completion_tokens is not None:
        totals.completion_tokens += completion_tokens
        totals.has_usage = True
    if cached_tokens is not None:
        totals.cached_tokens += cached_tokens
        totals.has_usage = True
    if cost_usd is not None:
        totals.cost_usd += cost_usd
        totals.has_usage = True


def update_usage_totals_from_claude(totals: UsageTotals, usage: dict[str, Any]) -> None:
    prompt_tokens = coerce_int(usage.get("input_tokens"))
    completion_tokens = coerce_int(usage.get("output_tokens"))
    cached_tokens = 0
    has_cached = False
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        value = coerce_int(usage.get(key))
        if value is not None:
            cached_tokens += value
            has_cached = True
    cost_usd = coerce_float(usage.get("cost_usd") or usage.get("cost"))

    if prompt_tokens is not None:
        totals.prompt_tokens += prompt_tokens
        totals.has_usage = True
    if completion_tokens is not None:
        totals.completion_tokens += completion_tokens
        totals.has_usage = True
    if has_cached:
        totals.cached_tokens += cached_tokens
        totals.has_usage = True
    if cost_usd is not None:
        totals.cost_usd += cost_usd
        totals.has_usage = True


def update_usage_totals_from_terminus(totals: UsageTotals, usage: dict[str, Any]) -> None:
    prompt_tokens = coerce_int(usage.get("prompt_tokens"))
    completion_tokens = coerce_int(usage.get("completion_tokens"))
    if prompt_tokens is not None:
        totals.prompt_tokens += prompt_tokens
        totals.has_usage = True
    if completion_tokens is not None:
        totals.completion_tokens += completion_tokens
        totals.has_usage = True


def parse_think_block(content: str) -> tuple[str | None, str]:
    if "<think>" not in content:
        return None, content.strip()
    start = content.find("<think>")
    end = content.find("</think>")
    if end == -1:
        return None, content.strip()
    reasoning = content[start + len("<think>") : end].strip()
    remaining = (content[:start] + content[end + len("</think>") :]).strip()
    return reasoning or None, remaining


def extract_terminal_output(message: str) -> str | None:
    marker = "New Terminal Output:\n"
    if marker in message:
        return message.split(marker, 1)[1].strip()
    marker = "Current terminal state:\n"
    if marker in message:
        return message.split(marker, 1)[1].strip()
    return None


def parse_episode_index(name: str) -> int:
    prefix = "episode-"
    if not name.startswith(prefix):
        raise ValueError(f"Invalid episode directory name: {name}")
    return int(name[len(prefix) :])


def parse_terminus_response(debug: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    original_response = debug["original_response"]
    if not isinstance(original_response, str):
        raise ValueError("original_response must be a JSON string")
    payload = json.loads(original_response)
    choices = payload["choices"]
    first_choice = choices[0]
    message_obj = first_choice["message"]
    message = message_obj.get("content")
    if message is None:
        reasoning_content = message_obj.get("reasoning_content")
        if isinstance(reasoning_content, str):
            message = reasoning_content
    if not isinstance(message, str):
        raise ValueError("original_response message content must be a string")
    usage = payload["usage"] if isinstance(payload.get("usage"), dict) else {}
    model_name = payload.get("model")
    return message, usage, model_name if isinstance(model_name, str) else None


def parse_terminus_tool_calls(
    response_text: str,
    episode_index: int,
    *,
    timestamp: str | None,
) -> tuple[list[ToolCall] | None, bool, str | None]:
    from terminal_bench.agents.terminus_2.terminus_json_plain_parser import (
        TerminusJSONPlainParser,
    )
    from terminal_bench.agents.terminus_2.terminus_xml_plain_parser import (
        TerminusXMLPlainParser,
    )

    if "<response>" in response_text:
        parser = TerminusXMLPlainParser()
    else:
        parser = TerminusJSONPlainParser()
    parse_result = parser.parse_response(response_text)
    if parse_result.error:
        return None, False, parse_result.error

    tool_calls_list: list[ToolCall] = []
    for i, cmd in enumerate(parse_result.commands):
        tool_call_id = f"call_{episode_index}_{i + 1}"
        tool_calls_list.append(
            ToolCall(
                tool_call_id=tool_call_id,
                function_name="bash_command",
                arguments={
                    "keystrokes": cmd.keystrokes,
                    "duration": cmd.duration,
                },
            )
        )
    if parse_result.is_task_complete:
        task_complete_call_id = f"call_{episode_index}_task_complete"
        tool_calls_list.append(
            ToolCall(
                tool_call_id=task_complete_call_id,
                function_name="mark_task_complete",
                arguments={},
            )
        )
    return tool_calls_list or None, parse_result.is_task_complete, None


def iter_events(rollout_path: Path) -> Iterable[dict[str, Any]]:
    buffer = ""
    start_line = 1
    last_error: json.JSONDecodeError | None = None
    with rollout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not buffer and not line.strip():
                continue
            if not buffer:
                start_line = line_no
            buffer += line
            try:
                # Allow unescaped control characters in tool outputs.
                payload = json.loads(buffer, strict=False)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if not isinstance(payload, dict):
                raise ValueError(f"Expected dict payload, got {type(payload)}")
            yield payload
            buffer = ""
            last_error = None
    if buffer.strip():
        if last_error is None:
            raise ValueError(f"Invalid JSON in {rollout_path}:{start_line}")
        raise ValueError(f"Invalid JSON in {rollout_path}:{start_line}: {last_error}") from last_error


def extract_claude_tool_output(event: dict[str, Any], block: dict[str, Any]) -> str | None:
    content = block.get("content")
    if isinstance(content, list):
        content = extract_message_text(content)
    if content is None:
        tool_use_result = event.get("toolUseResult")
        if isinstance(tool_use_result, dict):
            stdout = tool_use_result.get("stdout")
            stderr = tool_use_result.get("stderr")
            if isinstance(stdout, str) or isinstance(stderr, str):
                parts: list[str] = []
                if isinstance(stdout, str) and stdout:
                    parts.append(stdout)
                if isinstance(stderr, str) and stderr:
                    parts.append(stderr)
                content = "\n".join(parts)
    return extract_output_content(content)


def build_steps_from_claude(rollout_path: Path) -> tuple[list[Step], UsageTotals, str | None]:
    steps: list[Step] = []
    tool_steps: dict[str, Step] = {}
    totals = UsageTotals()
    inferred_model_name = None

    for event in iter_events(rollout_path):
        event_type = event.get("type")
        if event_type not in {"assistant", "user"}:
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str):
            role = "assistant" if event_type == "assistant" else "user"
        source = normalize_source(role)
        timestamp = event.get("timestamp")
        model_name = extract_model_name(message, event)
        if source == "AGENT" and inferred_model_name is None and model_name:
            inferred_model_name = model_name

        usage = message.get("usage")
        if isinstance(usage, dict):
            update_usage_totals_from_claude(totals, usage)

        content = message.get("content")
        if isinstance(content, str):
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                    source=source,
                    message=content,
                )
            )
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if not isinstance(text, str) or not text:
                    continue
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                        source=source,
                        message=text,
                    )
                )
            elif block_type == "tool_use":
                tool_call_id = block.get("id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    tool_call_id = f"call-{len(steps) + 1}"
                tool_name = block.get("name")
                if not isinstance(tool_name, str) or not tool_name:
                    tool_name = "tool"
                arguments = normalize_arguments(block.get("input"))
                message_label = extract_command_label(arguments) or tool_name
                tool_call = ToolCall(
                    tool_call_id=tool_call_id,
                    function_name=tool_name,
                    arguments=arguments,
                )
                step = Step(
                    step_id=len(steps) + 1,
                    timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                    source="AGENT",
                    message=message_label,
                    tool_calls=[tool_call],
                )
                steps.append(step)
                tool_steps[tool_call_id] = step
            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id") or block.get("id")
                output = extract_claude_tool_output(event, block) or ""
                if isinstance(tool_use_id, str) and tool_use_id in tool_steps:
                    step = tool_steps[tool_use_id]
                    if step.observation is None:
                        step.observation = []
                    step.observation.append(
                        Observation(
                            source_call_id=tool_use_id,
                            content=output,
                            timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                        )
                    )
                else:
                    steps.append(
                        Step(
                            step_id=len(steps) + 1,
                            timestamp=_coalesce_timestamp(timestamp, fallback_path=rollout_path),
                            source="AGENT",
                            message="Tool output",
                            observation=[
                                Observation(
                                    source_call_id=None,
                                    content=output,
                                    timestamp=_coalesce_timestamp(
                                        timestamp, fallback_path=rollout_path
                                    ),
                                )
                            ],
                        )
                    )

    if not steps:
        raise ValueError("No steps extracted from claude rollout file")
    return steps, totals, inferred_model_name


def resolve_opencode_session_id(storage_dir: Path) -> str:
    message_root = storage_dir / "message"
    if not message_root.is_dir():
        raise FileNotFoundError(f"OpenCode messages not found under {message_root}")
    session_dirs = sorted(path for path in message_root.iterdir() if path.is_dir())
    if not session_dirs:
        raise FileNotFoundError(f"No OpenCode sessions found under {message_root}")
    if len(session_dirs) == 1:
        return session_dirs[0].name

    session_root = storage_dir / "session" / "global"
    root_candidates: list[str] = []
    if session_root.is_dir():
        for path in session_dirs:
            session_path = session_root / f"{path.name}.json"
            if session_path.is_file():
                payload = read_json_dict(session_path)
                if "parentID" not in payload:
                    root_candidates.append(path.name)
    if len(root_candidates) == 1:
        return root_candidates[0]

    session_names = ", ".join(path.name for path in session_dirs)
    root_names = ", ".join(root_candidates)
    raise ValueError(
        f"Expected exactly one OpenCode session under {message_root}, found {len(session_dirs)}: {session_names}. "
        f"Root candidates: {root_names or 'none'}"
    )


def extract_opencode_part_time(part: dict[str, Any]) -> int | None:
    if "time" in part and isinstance(part["time"], dict) and "start" in part["time"]:
        return coerce_int(part["time"]["start"])
    if "state" in part and isinstance(part["state"], dict) and "time" in part["state"]:
        state_time = part["state"]["time"]
        if isinstance(state_time, dict) and "start" in state_time:
            return coerce_int(state_time["start"])
    return None


def extract_opencode_result_time(part: dict[str, Any]) -> int | None:
    for container_key in ("state", "time"):
        container = part.get(container_key)
        if not isinstance(container, dict):
            continue
        time_payload = container.get("time") if container_key == "state" else container
        if not isinstance(time_payload, dict):
            continue
        for key in ("end", "completed", "finish", "finished", "updated"):
            value = coerce_int(time_payload.get(key))
            if value is not None:
                return value
    return extract_opencode_part_time(part)


def load_opencode_parts(storage_dir: Path, message_id: str) -> list[dict[str, Any]]:
    part_root = storage_dir / "part" / message_id
    if not part_root.is_dir():
        raise FileNotFoundError(f"OpenCode parts not found under {part_root}")
    part_paths = sorted(part_root.glob("prt_*.json"))
    if not part_paths:
        raise ValueError(f"No OpenCode parts found under {part_root}")
    parts_with_meta: list[tuple[int | None, str, dict[str, Any]]] = []
    for path in part_paths:
        payload = read_json_dict(path)
        parts_with_meta.append((extract_opencode_part_time(payload), path.name, payload))
    parts_with_meta.sort(key=lambda item: (item[0] is None, item[0] or 0, item[1]))
    return [payload for _, _, payload in parts_with_meta]


def build_steps_from_opencode(
    storage_dir: Path,
    session_id: str,
) -> tuple[list[Step], UsageTotals, str | None]:
    messages_root = storage_dir / "message" / session_id
    if not messages_root.is_dir():
        raise FileNotFoundError(f"OpenCode messages not found under {messages_root}")

    message_paths = sorted(messages_root.glob("msg_*.json"))
    if not message_paths:
        raise ValueError(f"No OpenCode messages found under {messages_root}")

    messages: list[tuple[int, dict[str, Any]]] = []
    for path in message_paths:
        message = read_json_dict(path)
        created_raw = message["time"]["created"]
        created_ms = coerce_int(created_raw)
        if created_ms is None:
            raise ValueError(f"Invalid OpenCode message time in {path}")
        messages.append((created_ms, message))
    messages.sort(key=lambda item: item[0])

    steps: list[Step] = []
    totals = UsageTotals()
    default_model_name = None

    for created_ms, message in messages:
        role = message["role"]
        source = normalize_source(role)
        model_name = extract_opencode_model_name(message) if source == "AGENT" else None
        if default_model_name is None and model_name is not None:
            default_model_name = model_name
        update_usage_totals_from_opencode(totals, message)

        message_id = message["id"]
        parts = load_opencode_parts(storage_dir, message_id)
        for part in parts:
            part_type = part["type"]
            if part_type in {"step-start", "step-finish"}:
                continue
            part_time = extract_opencode_part_time(part)
            timestamp = opencode_timestamp_to_iso(part_time or created_ms)
            if part_type == "text":
                text = part["text"]
                if not isinstance(text, str):
                    raise ValueError(f"OpenCode text part must be a string in {message_id}")
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=timestamp,
                        source=source,
                        message=text,
                    )
                )
            elif part_type == "reasoning":
                text = extract_message_text(
                    part.get("text") or part.get("content") or part.get("summary")
                )
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=timestamp,
                        source=source,
                        message="",
                        reasoning_content=text,
                    )
                )
            elif part_type == "tool":
                state = part["state"]
                if not isinstance(state, dict):
                    raise ValueError(f"OpenCode tool state must be a dict in {message_id}")
                tool_name = part["tool"]
                if not isinstance(tool_name, str):
                    raise ValueError(f"OpenCode tool name must be a string in {message_id}")
                call_id = part["callID"] if "callID" in part else f"call_{len(steps) + 1}"
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"call_{len(steps) + 1}"
                observation_timestamp = opencode_timestamp_to_iso(
                    extract_opencode_result_time(part) or part_time or created_ms
                )
                arguments = normalize_arguments(state["input"] if "input" in state else None)
                message_label = extract_command_label(arguments) or tool_name
                output = None
                if "output" in state:
                    output = state["output"]
                elif "error" in state:
                    output = state["error"]
                elif "metadata" in state and isinstance(state["metadata"], dict) and "output" in state["metadata"]:
                    output = state["metadata"]["output"]
                output_content = extract_output_content(output) if output is not None else None
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=timestamp,
                        source="AGENT",
                        message=message_label,
                        tool_calls=[
                            ToolCall(
                                tool_call_id=call_id,
                                function_name=tool_name,
                                arguments=arguments,
                            )
                        ],
                        observation=[
                            Observation(
                                source_call_id=call_id,
                                content=output_content,
                                timestamp=observation_timestamp,
                            )
                        ],
                    )
                )
            else:
                raise ValueError(f"Unsupported OpenCode part type: {part_type}")

    if not steps:
        raise ValueError(f"No steps extracted from OpenCode session {session_id}")
    return steps, totals, default_model_name


def build_steps_from_opencode_session_json(
    session_path: Path,
) -> tuple[list[Step], UsageTotals, str | None, str | None]:
    payload = read_json_dict(session_path)
    session_info = payload.get("info")
    session_id = None
    if isinstance(session_info, dict) and isinstance(session_info.get("id"), str):
        session_id = session_info["id"]
    messages_payload = payload.get("messages")
    if not isinstance(messages_payload, list):
        raise ValueError(f"OpenCode session JSON missing messages list: {session_path}")

    messages: list[tuple[int, int, dict[str, Any], list[dict[str, Any]]]] = []
    for index, record in enumerate(messages_payload):
        if not isinstance(record, dict):
            raise ValueError(f"OpenCode message entry must be a dict in {session_path}")
        info = record.get("info")
        if not isinstance(info, dict):
            raise ValueError(f"OpenCode message missing info object in {session_path}")
        time_payload = info.get("time")
        created_raw = time_payload.get("created") if isinstance(time_payload, dict) else None
        created_ms = coerce_int(created_raw)
        if created_ms is None:
            raise ValueError(f"Invalid OpenCode message time in {session_path}")
        parts_payload = record.get("parts")
        if not isinstance(parts_payload, list):
            raise ValueError(f"OpenCode message missing parts list in {session_path}")
        parts: list[dict[str, Any]] = []
        for part in parts_payload:
            if not isinstance(part, dict):
                raise ValueError(f"OpenCode part entry must be a dict in {session_path}")
            parts.append(part)
        messages.append((created_ms, index, info, parts))
    messages.sort(key=lambda item: (item[0], item[1]))

    steps: list[Step] = []
    totals = UsageTotals()
    default_model_name = None

    for created_ms, _, message, parts in messages:
        role = message["role"]
        source = normalize_source(role)
        model_name = extract_opencode_model_name(message) if source == "AGENT" else None
        if default_model_name is None and model_name is not None:
            default_model_name = model_name
        update_usage_totals_from_opencode(totals, message)

        message_id = str(message.get("id") or f"message_{len(steps) + 1}")
        for part in parts:
            part_type = part["type"]
            if part_type in {"step-start", "step-finish"}:
                continue
            part_time = extract_opencode_part_time(part)
            timestamp = opencode_timestamp_to_iso(part_time or created_ms)
            if part_type == "text":
                text = part["text"]
                if not isinstance(text, str):
                    raise ValueError(f"OpenCode text part must be a string in {message_id}")
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=timestamp,
                        source=source,
                        message=text,
                    )
                )
            elif part_type == "reasoning":
                text = extract_message_text(
                    part.get("text") or part.get("content") or part.get("summary")
                )
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=timestamp,
                        source=source,
                        message="",
                        reasoning_content=text,
                    )
                )
            elif part_type == "tool":
                state = part["state"]
                if not isinstance(state, dict):
                    raise ValueError(f"OpenCode tool state must be a dict in {message_id}")
                tool_name = part["tool"]
                if not isinstance(tool_name, str):
                    raise ValueError(f"OpenCode tool name must be a string in {message_id}")
                call_id = part["callID"] if "callID" in part else f"call_{len(steps) + 1}"
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"call_{len(steps) + 1}"
                observation_timestamp = opencode_timestamp_to_iso(
                    extract_opencode_result_time(part) or part_time or created_ms
                )
                arguments = normalize_arguments(state["input"] if "input" in state else None)
                message_label = extract_command_label(arguments) or tool_name
                output = None
                if "output" in state:
                    output = state["output"]
                elif "error" in state:
                    output = state["error"]
                elif (
                    "metadata" in state
                    and isinstance(state["metadata"], dict)
                    and "output" in state["metadata"]
                ):
                    output = state["metadata"]["output"]
                output_content = extract_output_content(output) if output is not None else None
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=timestamp,
                        source="AGENT",
                        message=message_label,
                        tool_calls=[
                            ToolCall(
                                tool_call_id=call_id,
                                function_name=tool_name,
                                arguments=arguments,
                            )
                        ],
                        observation=[
                            Observation(
                                source_call_id=call_id,
                                content=output_content,
                                timestamp=observation_timestamp,
                            )
                        ],
                    )
                )
            else:
                raise ValueError(f"Unsupported OpenCode part type: {part_type}")

    if not steps:
        raise ValueError(f"No steps extracted from OpenCode session JSON {session_path}")
    return steps, totals, default_model_name, session_id


def _sort_steps_chronologically(steps: list[Step]) -> list[Step]:
    indexed_steps = list(enumerate(steps))

    def sort_key(indexed_step: tuple[int, Step]) -> tuple[dt.datetime, int]:
        index, step = indexed_step
        timestamp = _parse_iso_timestamp(step.timestamp)
        if timestamp is None:
            timestamp = dt.datetime.max.replace(tzinfo=dt.timezone.utc)
        return timestamp, index

    ordered_steps = [step for _index, step in sorted(indexed_steps, key=sort_key)]
    for index, step in enumerate(ordered_steps, start=1):
        step.step_id = index
    return ordered_steps


def build_steps_from_terminus(episode_root: Path) -> tuple[list[Step], UsageTotals]:
    episode_dirs = sorted(
        (path for path in episode_root.iterdir() if path.is_dir() and path.name.startswith("episode-")),
        key=lambda p: parse_episode_index(p.name),
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episode logs found under {episode_root}")

    steps: list[Step] = []
    totals = UsageTotals()
    seen_count = 0
    last_agent_step: Step | None = None

    for episode_dir in episode_dirs:
        debug_path = episode_dir / "debug.json"
        if not debug_path.is_file():
            raise FileNotFoundError(f"Missing debug.json at {debug_path}")
        debug = read_json_dict(debug_path)
        episode_timestamp = _coalesce_timestamp(
            debug.get("api_call_start_time"),
            debug.get("api_call_end_time"),
            fallback_path=debug_path,
        )

        input_messages = debug["input"]
        if not isinstance(input_messages, list):
            raise ValueError(f"Expected list input messages in {debug_path}")
        if len(input_messages) < seen_count:
            raise ValueError(f"Input messages shorter than expected in {debug_path}")

        for message in input_messages[seen_count:]:
            if not isinstance(message, dict):
                raise ValueError(f"Expected dict message in {debug_path}")
            role = message["role"]
            content = message.get("content") if "content" in message else ""
            if content is None:
                content = ""
            if not isinstance(role, str):
                raise ValueError(f"Message role must be a string in {debug_path}")
            if not isinstance(content, str):
                raise ValueError(f"Message content must be a string in {debug_path}")
            source = normalize_source(role)
            observation_text = extract_terminal_output(content) if source == "USER" else None
            if observation_text and last_agent_step:
                if last_agent_step.observation is None:
                    last_agent_step.observation = [
                        Observation(
                            source_call_id=None,
                            content=observation_text,
                            timestamp=episode_timestamp,
                        )
                    ]
                else:
                    last_agent_step.observation.append(
                        Observation(
                            source_call_id=None,
                            content=observation_text,
                            timestamp=episode_timestamp,
                        )
                    )
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=episode_timestamp,
                    source=source,
                    message=content,
                )
            )

        response_text, usage, model_name = parse_terminus_response(debug)
        update_usage_totals_from_terminus(totals, usage)
        reasoning, message = parse_think_block(response_text)
        tool_calls, _, tool_call_error = parse_terminus_tool_calls(
            response_text,
            parse_episode_index(episode_dir.name),
            timestamp=episode_timestamp,
        )
        agent_step = Step(
            step_id=len(steps) + 1,
            timestamp=episode_timestamp,
            source="AGENT",
            message=message,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
        )
        if tool_call_error:
            agent_step.message = "\n".join(
                part for part in [message, f"[tool_call_parse_error] {tool_call_error}"] if part
            )
        steps.append(agent_step)
        last_agent_step = agent_step
        seen_count = len(input_messages) + 1

    if not steps:
        raise ValueError(f"No steps extracted from {episode_root}")
    return steps, totals


def build_trajectory_from_gemini(
    session_path: Path,
    session_id: str,
    agent_profile_id: str,
    model_name: str | None,
) -> Trajectory:
    gemini_payload = read_json_dict(session_path)
    messages = gemini_payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Gemini session did not contain messages: {session_path}")
    steps: list[Step] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        msg_type = message.get("type")
        timestamp = _coalesce_timestamp(message.get("timestamp"), fallback_path=session_path)
        if msg_type == "user":
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=timestamp,
                    source="USER",
                    message=extract_message_text(message.get("content")),
                )
            )
            continue
        if msg_type != "gemini":
            continue
        thoughts = message.get("thoughts")
        reasoning_content = None
        if isinstance(thoughts, list):
            reasoning_parts: list[str] = []
            for thought in thoughts:
                if not isinstance(thought, dict):
                    continue
                subject = str(thought.get("subject") or "").strip()
                description = str(thought.get("description") or "").strip()
                if subject and description:
                    reasoning_parts.append(f"{subject}: {description}")
                elif description:
                    reasoning_parts.append(description)
            if reasoning_parts:
                reasoning_content = "\n".join(reasoning_parts)

        tool_calls_payload = message.get("toolCalls")
        tool_calls: list[ToolCall] | None = None
        observations: list[Observation] | None = None
        if isinstance(tool_calls_payload, list) and tool_calls_payload:
            tool_calls = []
            observations = []
            for tool_call_payload in tool_calls_payload:
                if not isinstance(tool_call_payload, dict):
                    continue
                tool_call_id = str(tool_call_payload.get("id") or "").strip() or f"call_{len(tool_calls) + 1}"
                tool_name = str(tool_call_payload.get("name") or "").strip() or "tool"
                arguments = normalize_arguments(tool_call_payload.get("args"))
                observation_timestamp = _coalesce_timestamp(
                    tool_call_payload.get("timestamp"),
                    timestamp,
                    fallback_path=session_path,
                )
                tool_calls.append(
                    ToolCall(
                        tool_call_id=tool_call_id,
                        function_name=tool_name,
                        arguments=arguments,
                    )
                )
                result_content = None
                result = tool_call_payload.get("result")
                if isinstance(result, list):
                    for result_item in result:
                        if not isinstance(result_item, dict):
                            continue
                        function_response = result_item.get("functionResponse")
                        if not isinstance(function_response, dict):
                            continue
                        response_payload = function_response.get("response")
                        if not isinstance(response_payload, dict):
                            continue
                        output = response_payload.get("output")
                        if output is not None:
                            result_content = extract_output_content(output)
                            break
                observations.append(
                    Observation(
                        source_call_id=tool_call_id,
                        content=result_content,
                        timestamp=observation_timestamp,
                    )
                )

        tokens = message.get("tokens")
        metrics = None
        if isinstance(tokens, dict):
            input_tokens = coerce_int(tokens.get("input")) or 0
            output_tokens = coerce_int(tokens.get("output")) or 0
            cached_tokens = coerce_int(tokens.get("cached")) or 0
            thoughts_tokens = coerce_int(tokens.get("thoughts")) or 0
            tool_tokens = coerce_int(tokens.get("tool")) or 0
            completion_tokens = output_tokens + thoughts_tokens + tool_tokens
            total_input_tokens += input_tokens
            total_output_tokens += completion_tokens
            total_cached_tokens += cached_tokens
            metrics = Metrics(
                prompt_tokens=input_tokens or None,
                completion_tokens=completion_tokens or None,
                cached_tokens=cached_tokens or None,
            )

        steps.append(
            Step(
                step_id=len(steps) + 1,
                timestamp=timestamp,
                source="AGENT",
                message=extract_message_text(message.get("content")),
                reasoning_content=reasoning_content,
                tool_calls=tool_calls,
                observation=observations,
                metrics=metrics,
            )
        )

    if not steps:
        raise ValueError(f"Gemini session did not contain exportable messages: {session_path}")

    return finalize_documented_trajectory(
        Trajectory(
            session_id=session_id,
            agent=Agent(
                name=agent_profile_id,
                version=EXPORTED_AGENT_VERSION,
            ),
            steps=steps,
            final_metrics=_build_final_metrics_from_values(
                step_count=len(steps),
                agent_profile_id=agent_profile_id,
                prompt_tokens=total_input_tokens,
                cached_tokens=total_cached_tokens,
                completion_tokens=total_output_tokens,
            ),
        )
    )


def build_final_metrics(
    steps: list[Step],
    totals: UsageTotals,
    agent_profile_id: str,
) -> FinalMetrics:
    return _build_final_metrics_from_values(
        step_count=len(steps),
        agent_profile_id=agent_profile_id,
        prompt_tokens=totals.prompt_tokens,
        cached_tokens=totals.cached_tokens,
        completion_tokens=totals.completion_tokens,
    )


def resolve_output_path(rollout_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    if rollout_path.parent.name == "agent_logs":
        return canonical_atif_processed_dir(rollout_path.parent.parent) / TRAJECTORY_FILENAME
    return rollout_path.parent / TRAJECTORY_FILENAME


def resolve_light_output_path(output_path: Path) -> Path:
    if output_path.name == TRAJECTORY_FILENAME:
        return output_path.with_name(LIGHT_TRAJECTORY_FILENAME)
    suffix = output_path.suffix
    if suffix:
        return output_path.with_name(f"{output_path.stem}_light{suffix}")
    return output_path.with_name(f"{output_path.name}_light")


def build_light_trajectory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    light_steps: list[dict[str, Any]] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, dict):
            continue
        light_step: dict[str, Any] = {}
        for key in ("step_id", "source", "message", "tool_calls", "observation"):
            if key in step:
                light_step[key] = step[key]
        light_steps.append(light_step)
    return {
        "agent_id": payload["agent_id"],
        "session_id": payload["session_id"],
        "steps": light_steps,
    }


def _build_codex_trajectory_from_target(target: TrajectoryTarget) -> Trajectory:
    if target.rollout_path is None:
        raise ValueError("Missing rollout_path for codex target")
    return build_codex_trajectory(
        rollout_path=target.rollout_path,
        session_id=target.session_id,
        agent_profile_id=target.agent_profile_id,
        model_name=target.model_name,
    )


def _build_claude_trajectory_from_target(target: TrajectoryTarget) -> Trajectory:
    if target.rollout_path is None:
        raise ValueError("Missing rollout_path for claude target")
    steps, totals, _ = build_steps_from_claude(target.rollout_path)
    return finalize_documented_trajectory(
        Trajectory(
            session_id=target.session_id,
            agent=Agent(
                name=target.agent_profile_id,
                version=EXPORTED_AGENT_VERSION,
            ),
            steps=steps,
            final_metrics=build_final_metrics(steps, totals, target.agent_profile_id),
        )
    )


def _build_terminus_trajectory_from_target(target: TrajectoryTarget) -> Trajectory:
    if target.episode_root is None:
        raise ValueError("Missing episode_root for terminus target")
    steps, totals = build_steps_from_terminus(target.episode_root)
    return finalize_documented_trajectory(
        Trajectory(
            session_id=target.session_id,
            agent=Agent(
                name=target.agent_profile_id,
                version=EXPORTED_AGENT_VERSION,
            ),
            steps=steps,
            final_metrics=build_final_metrics(steps, totals, target.agent_profile_id),
        )
    )


def _build_gemini_trajectory_from_target(target: TrajectoryTarget) -> Trajectory:
    if target.gemini_session_path is None:
        raise ValueError("Missing gemini_session_path for gemini target")
    return build_trajectory_from_gemini(
        target.gemini_session_path,
        target.session_id,
        target.agent_profile_id,
        target.model_name,
    )


def _build_opencode_trajectory_from_target(target: TrajectoryTarget) -> Trajectory:
    if target.opencode_storage is None:
        raise ValueError("Missing opencode_storage for opencode target")
    opencode_session_id = resolve_opencode_session_id(target.opencode_storage)
    steps, totals, _ = build_steps_from_opencode(target.opencode_storage, opencode_session_id)
    steps = _sort_steps_chronologically(steps)
    return finalize_documented_trajectory(
        Trajectory(
            session_id=target.session_id,
            agent=Agent(
                name=target.agent_profile_id,
                version=EXPORTED_AGENT_VERSION,
            ),
            steps=steps,
            final_metrics=build_final_metrics(steps, totals, target.agent_profile_id),
        )
    )


def _build_opencode_session_json_trajectory_from_target(target: TrajectoryTarget) -> Trajectory:
    if target.opencode_session_path is None:
        raise ValueError("Missing opencode_session_path for opencode_session_json target")
    steps, totals, _, _ = build_steps_from_opencode_session_json(target.opencode_session_path)
    steps = _sort_steps_chronologically(steps)
    return finalize_documented_trajectory(
        Trajectory(
            session_id=target.session_id,
            agent=Agent(
                name=target.agent_profile_id,
                version=EXPORTED_AGENT_VERSION,
            ),
            steps=steps,
            final_metrics=build_final_metrics(steps, totals, target.agent_profile_id),
        )
    )


TrajectoryBuilder = Callable[[TrajectoryTarget], Trajectory]

SOURCE_BUILDERS: dict[str, TrajectoryBuilder] = {
    "codex": _build_codex_trajectory_from_target,
    "claude": _build_claude_trajectory_from_target,
    "terminus": _build_terminus_trajectory_from_target,
    "gemini": _build_gemini_trajectory_from_target,
    "opencode": _build_opencode_trajectory_from_target,
    "opencode_session_json": _build_opencode_session_json_trajectory_from_target,
}


@click.command()
@click.argument("agentic_run_id", required=False)
@click.option(
    "--path",
    "input_path",
    type=click.Path(path_type=Path, exists=False, file_okay=True, dir_okay=True),
    help="Explicit run directory, results.json, rollout jsonl, or session json path.",
)
@click.option(
    "--agent-name",
    default=None,
    help="Override agent profile ID (or agent platform name when the profile match is unique).",
)
@click.option(
    "--agent-version",
    default=None,
    help="Deprecated. Exported trajectories always use agent.version=1.00.",
)
@click.option(
    "--model-name",
    default=None,
    help="Override model name (defaults to run_metadata.json when available).",
)
@click.option(
    "--validate/--no-validate",
    default=False,
    help="Validate exported trajectories with Harbor's official validator.",
)
def main(
    agentic_run_id: str | None,
    input_path: Path | None,
    agent_name: str | None,
    agent_version: str | None,
    model_name: str | None,
    validate: bool,
) -> None:
    if bool(agentic_run_id) == bool(input_path):
        raise click.UsageError("Provide exactly one of <agentic_run_id> or --path PATH.")

    if agentic_run_id is not None:
        run_id = str(agentic_run_id).strip()
        if not run_id:
            raise click.UsageError("agentic_run_id must be a non-empty string.")
        if Path(run_id).name != run_id:
            raise click.UsageError("agentic_run_id must be a run id, not a filesystem path.")
        paths = [RUNS_ROOT / run_id]
    else:
        paths = [input_path]

    try:
        targets = collect_targets(
            paths,
            None,
            None,
            agent_name,
            model_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not targets:
        message = "No rollout targets found; nothing to export."
        click.echo(message, err=True)
        return

    for target in targets:
        builder = SOURCE_BUILDERS.get(target.source_type)
        if builder is None:
            raise ValueError(f"Unknown source_type: {target.source_type}")
        try:
            trajectory = builder(target)
        except (FileNotFoundError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        trajectory.is_context_compacted = target_has_context_compaction(target)

        target.output_path.parent.mkdir(parents=True, exist_ok=True)
        full_payload = trajectory_to_documented_json_dict(
            trajectory,
            agent_id=target.exported_agent_id,
            question_id=target.question_id,
            tag=target.tag,
        )
        serialized = format_trajectory_json(full_payload)
        target.output_path.write_text(serialized + "\n")
        click.echo(f"Wrote ATIF trajectory to {target.output_path}")
        light_output_path = resolve_light_output_path(target.output_path)
        light_serialized = format_trajectory_json(build_light_trajectory_payload(full_payload))
        light_output_path.write_text(light_serialized + "\n")
        click.echo(f"Wrote light ATIF trajectory to {light_output_path}")
        if validate:
            validator = TrajectoryValidator()
            if not validator.validate(target.output_path):
                errors = "\n".join(f"- {error}" for error in validator.get_errors())
                raise click.ClickException(
                    f"Trajectory validation failed for {target.output_path}:\n{errors}"
                )


if __name__ == "__main__":
    main()
