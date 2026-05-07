#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import click

_SCRIPT_DIR = Path(__file__).resolve().parent


def _find_repo_root(start_dir: Path) -> Path:
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "terminal-bench").is_dir():
            return candidate
    return start_dir.parents[1] if len(start_dir.parents) > 1 else start_dir


ROOT = _find_repo_root(_SCRIPT_DIR)
TSENV_QUESTIONS_ROOT = (ROOT / "tsENV_questions").resolve()
DRIVER_SCRIPT = ROOT / "workflows" / "rollout" / "run_single_question.py"
ADAPTER_SCRIPT = ROOT / "terminal-bench" / "adapters" / "tsENV" / "run_adapter.py"
DEFAULT_CONFIGURATION_DIR = ROOT / "run_configurations"
QUESTION_TIMEOUT_SEC = "3600"
DEFAULT_PARALLEL_PER_AGENT = 1
PLAN_VERSION = 1
PLAN_STATUS_PENDING = "PENDING"
PLAN_STATUS_RUNNING = "RUNNING"
PLAN_STATUS_DONE = "DONE"
PLAN_STATUS_ERROR = "ERROR"
RESUMABLE_PLAN_STATUSES = {PLAN_STATUS_PENDING, PLAN_STATUS_ERROR}
ALL_PLAN_STATUSES = {
    PLAN_STATUS_PENDING,
    PLAN_STATUS_RUNNING,
    PLAN_STATUS_DONE,
    PLAN_STATUS_ERROR,
}
VERBOSE = True

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.benchmark_utils import AGENTIC_PROFILES_BY_ID  # noqa: E402
from shared.prompts import render_tsenv_agent_prompt  # noqa: E402
from shared.scores_schema import validate_scores_payload  # noqa: E402
from shared.tsenv_metadata import (  # noqa: E402
    load_metadata_payload,
    metadata_questions_by_id,
    resolve_tsenv_payload_path,
)
from workflows.rollout.run_single_question import (  # noqa: E402
    write_noise_analysis_for_materialized_question as _driver_write_noise_analysis_for_materialized_question,
)


class MultiValueOption(click.Option):
    """Click option that accepts one or more values per occurrence."""

    def __init__(self, *args, **kwargs) -> None:
        nargs = kwargs.pop("nargs", -1)
        if nargs != -1:
            raise TypeError(f"{self.__class__.__name__} only supports nargs=-1")
        super().__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._multi_parser = None

    def add_to_parser(self, parser, ctx):  # type: ignore[override]
        def parser_process(value, state):
            collected = [value]
            while state.rargs:
                next_arg = state.rargs[0]
                if any(next_arg.startswith(prefix) for prefix in self._multi_parser.prefixes):
                    break
                collected.append(state.rargs.pop(0))
            self._previous_parser_process(tuple(collected), state)

        retval = super().add_to_parser(parser, ctx)
        for name in [*self.opts, *self.secondary_opts]:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser is not None:
                self._multi_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval

    def type_cast_value(self, ctx, value):  # type: ignore[override]
        if value is None:
            return ()
        flattened: list[Any] = []

        def _flatten(raw: Any) -> None:
            if isinstance(raw, (tuple, list)):
                for item in raw:
                    _flatten(item)
                return
            flattened.append(raw)

        _flatten(value)
        return tuple(self.type(item, self, ctx) for item in flattened)


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    if VERBOSE:
        prefix = f"$ (cwd={cwd}) " if cwd else "$ "
        print(prefix + " ".join(shlex.quote(part) for part in cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if VERBOSE:
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise click.ClickException(
            f"Command failed (exit={proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stdout or ''}{proc.stderr or ''}"
        )
    return proc


def _sanitize_run_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


def _compute_run_id(agent_id: str, model_name: str, question_slug: str) -> str:
    raw = (
        f"{dt.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
        f"__{agent_id}__{model_name}__{question_slug}"
    )
    return _sanitize_run_id(raw)


def _resolve_run_name(value: str | None) -> str:
    name = str(value or "").strip()
    if name:
        return Path(_resolve_configuration_file_name(name)).stem
    return _resolve_configuration_file_name(None).removesuffix(".json").removesuffix(".csv")


def _validate_parallelism(value: int | None) -> int:
    if value is None:
        return DEFAULT_PARALLEL_PER_AGENT
    if value <= 0:
        raise click.ClickException(
            "--number-of-runs-in-parallel-per-agent must be a positive integer."
        )
    return value


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _flatten_cli_values(values: tuple[Any, ...]) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, (tuple, list)):
            out.extend(_flatten_cli_values(tuple(value)))
            continue
        text = str(value or "").strip()
        if text and text != "Sentinel.UNSET":
            out.append(text)
    return out


def _validate_agent_ids(agent_ids: tuple[Any, ...]) -> list[str]:
    cleaned = _dedupe_preserve_order(
        _flatten_cli_values(agent_ids)
    )
    for agent_id in cleaned:
        if agent_id not in AGENTIC_PROFILES_BY_ID:
            valid_ids = ", ".join(sorted(AGENTIC_PROFILES_BY_ID))
            raise click.ClickException(
                f"Unknown --agent-id {agent_id!r}. Valid values: {valid_ids}."
            )
    return cleaned


def _validate_models(models: tuple[Any, ...]) -> list[str]:
    cleaned = _dedupe_preserve_order(
        _flatten_cli_values(models)
    )
    if not cleaned:
        raise click.UsageError("Pass at least one --model.")
    return cleaned


def _discover_tsenv_model_dirs(tasks_dir: Path) -> list[Path]:
    tasks_dir = tasks_dir.expanduser().resolve()
    root_dir = TSENV_QUESTIONS_ROOT.expanduser().resolve()
    if tasks_dir == root_dir:
        return sorted(
            path
            for path in root_dir.iterdir()
            if path.is_dir() and (path / "questions.json").is_file()
        )
    if (
        tasks_dir.parent == root_dir
        and tasks_dir.is_dir()
        and (tasks_dir / "questions.json").is_file()
    ):
        return [tasks_dir]
    raise click.ClickException(
        f"--tasks-dir must be {root_dir} or a model directory directly under it."
    )


def _resolve_model_dirs(tasks_dir: Path, requested_models: list[str]) -> list[Path]:
    available_dirs = _discover_tsenv_model_dirs(tasks_dir)
    by_name = {path.name: path for path in available_dirs}
    missing = [model for model in requested_models if model not in by_name]
    if missing:
        raise click.ClickException(
            f"Unknown --model value(s): {', '.join(sorted(missing))}."
        )
    return [by_name[model] for model in requested_models]


def _catalog_key(model_name: str, question_slug: str) -> str:
    return f"{model_name}::{question_slug}"


def _build_question_catalog(
    tasks_dir: Path,
    *,
    requested_models: list[str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    question_catalog: dict[str, dict[str, Any]] = {}
    row_to_question_keys: dict[str, list[str]] = defaultdict(list)
    shot_to_question_keys: dict[str, list[str]] = defaultdict(list)
    question_slug_to_keys: dict[str, list[str]] = defaultdict(list)
    model_dirs = _resolve_model_dirs(tasks_dir, requested_models)
    for model_dir in model_dirs:
        payload_path = resolve_tsenv_payload_path(model_dir)
        payload = load_metadata_payload(payload_path)
        questions_by_id = metadata_questions_by_id(payload)
        items = sorted(questions_by_id.items())
        for question_slug, question in items:
            qslug = str(question_slug or "").strip()
            if not qslug:
                raise click.ClickException(f"Empty question_slug found in {payload_path}")
            catalog_key = _catalog_key(model_dir.name, qslug)
            recipe_info = question.get("recipe_info") or {}
            question_text = question.get("question_text") or {}
            row_slug = str(recipe_info.get("row_slug") or "").strip() if isinstance(recipe_info, dict) else ""
            shot_slug = str(recipe_info.get("shot_slug") or "").strip() if isinstance(recipe_info, dict) else ""
            question_catalog[catalog_key] = {
                "question_slug": qslug,
                "model": model_dir.name,
                "row_slug": row_slug,
                "shot_slug": shot_slug,
                "question_schema": dict(question),
                "recipe_info": dict(recipe_info) if isinstance(recipe_info, dict) else {},
                "agent_prompt": (
                    render_tsenv_agent_prompt(
                        question_text,
                        question_slug=qslug,
                        questions_by_id=questions_by_id,
                        questions_metadata=payload,
                    )
                    if isinstance(question_text, dict)
                    else ""
                ),
            }
            if row_slug:
                row_to_question_keys[row_slug].append(catalog_key)
            if shot_slug:
                shot_to_question_keys[shot_slug].append(catalog_key)
            question_slug_to_keys[qslug].append(catalog_key)
    if not question_catalog:
        raise click.ClickException(
            f"No question_slug entries found under {tasks_dir} for models {requested_models!r}."
        )
    return (
        question_catalog,
        {key: list(value) for key, value in row_to_question_keys.items()},
        {key: list(value) for key, value in shot_to_question_keys.items()},
        {key: list(value) for key, value in question_slug_to_keys.items()},
    )


def _resolve_selector(
    *,
    question_slugs: tuple[Any, ...],
    row_slugs: tuple[Any, ...],
    shot_slugs: tuple[Any, ...],
    resume: str | None,
) -> tuple[str, list[str]]:
    provided = [
        ("question-slug", _flatten_cli_values(question_slugs)),
        ("row-slug", _flatten_cli_values(row_slugs)),
        ("shot-slug", _flatten_cli_values(shot_slugs)),
        ("resume", [str(resume or "").strip()] if str(resume or "").strip() else []),
    ]
    selected = [(name, values) for name, values in provided if values]
    if len(selected) != 1:
        raise click.UsageError(
            "Pass exactly one of --question-slug, --row-slug, --shot-slug, or --resume."
        )
    selector_name, selector_values = selected[0]
    return selector_name, _dedupe_preserve_order(selector_values)


def _resolve_selected_question_slugs(
    *,
    selector_name: str,
    selector_values: list[str],
    question_catalog: dict[str, dict[str, Any]],
    row_to_question_keys: dict[str, list[str]],
    shot_to_question_keys: dict[str, list[str]],
    question_slug_to_keys: dict[str, list[str]],
) -> list[str]:
    selected_question_keys: list[str] = []
    if selector_name == "question-slug":
        for question_slug in selector_values:
            matched = question_slug_to_keys.get(question_slug)
            if not matched:
                raise click.ClickException(f"Unknown --question-slug {question_slug!r}.")
            selected_question_keys.extend(matched)
    elif selector_name == "row-slug":
        for row_slug in selector_values:
            matched = row_to_question_keys.get(row_slug)
            if not matched:
                raise click.ClickException(f"Unknown --row-slug {row_slug!r}.")
            selected_question_keys.extend(matched)
    elif selector_name == "shot-slug":
        for shot_slug in selector_values:
            matched = shot_to_question_keys.get(shot_slug)
            if not matched:
                raise click.ClickException(f"Unknown --shot-slug {shot_slug!r}.")
            selected_question_keys.extend(matched)
    else:
        raise click.ClickException(f"Unsupported selector {selector_name!r}.")
    return _dedupe_preserve_order(selected_question_keys)


def _run_path_for_id(run_id: str) -> str:
    return str((ROOT / "terminal-bench" / "runs" / run_id).resolve())


def _valid_scores_path_for_run(run: dict[str, Any]) -> Path | None:
    run_id = str(run.get("agentic_run_id") or "").strip()
    if not run_id:
        return None
    candidate_paths: list[Path] = []
    recorded_path = str(run.get("path_to_the_run") or "").strip()
    if recorded_path:
        candidate_paths.append(Path(recorded_path).expanduser().resolve())
    candidate_paths.append(Path(_run_path_for_id(run_id)))

    seen: set[Path] = set()
    for run_path in candidate_paths:
        if run_path in seen:
            continue
        seen.add(run_path)
        if not run_path.is_dir():
            continue
        for scores_path in sorted(run_path.rglob("scores.json")):
            try:
                payload = json.loads(scores_path.read_text(encoding="utf-8"))
                scores = validate_scores_payload(payload, path=str(scores_path))
            except Exception:
                continue
            if scores.agent_run_id == run_id:
                return scores_path
    return None


def _reconcile_scored_error_runs(plan: dict[str, Any]) -> int:
    reconciled_count = 0
    for run in plan.get("runs") or []:
        if not isinstance(run, dict) or run.get("status") != PLAN_STATUS_ERROR:
            continue
        scores_path = _valid_scores_path_for_run(run)
        if scores_path is None:
            continue
        run["status"] = PLAN_STATUS_DONE
        run.pop("error", None)
        run["path_to_the_run"] = _run_path_for_id(str(run.get("agentic_run_id") or ""))
        reconciled_count += 1
    return reconciled_count


def _plan_entry_for_run(
    *,
    agent_id: str,
    question_slug: str,
    model_name: str,
    tag: str,
    recipe_info: dict[str, Any],
    agent_prompt: str,
    question_schema: dict[str, Any],
) -> dict[str, Any]:
    run_id = _compute_run_id(agent_id, model_name, question_slug)
    entry: dict[str, Any] = {
        "question_slug": question_slug,
        "question_schema": question_schema,
        "model": model_name,
        "agent_id": agent_id,
        "agentic_run_id": run_id,
        "status": PLAN_STATUS_PENDING,
        "path_to_the_run": "",
        "tag": tag,
        "agent_prompt": agent_prompt,
    }
    for key, value in sorted(recipe_info.items()):
        entry[f"question.recipe_info.{key}"] = value
    return entry


def _build_plan(
    *,
    tasks_dir: Path,
    agent_ids: list[str],
    requested_models: list[str],
    selector_name: str,
    selector_values: list[str],
    selected_question_keys: list[str],
    question_catalog: dict[str, dict[str, Any]],
    number_runs_in_parallel_per_agent: int,
    tag: str,
    run_name: str,
    timestamp: str,
    call_adapter: bool = False,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for agent_id in agent_ids:
        for question_key in selected_question_keys:
            question_info = question_catalog[question_key]
            runs.append(
                _plan_entry_for_run(
                    agent_id=agent_id,
                    question_slug=question_info["question_slug"],
                    model_name=question_info["model"],
                    tag=tag,
                    recipe_info=dict(question_info.get("recipe_info") or {}),
                    agent_prompt=str(question_info.get("agent_prompt") or ""),
                    question_schema=dict(question_info.get("question_schema") or {}),
                )
            )
    return {
        "timestamp": timestamp,
        "name": run_name,
        "models": list(requested_models),
        "number_of_runs_in_parallel_per_agent": number_runs_in_parallel_per_agent,
        "total_runs": len(runs),
        "runs": runs,
    }


def _resolve_configuration_file_name(value: str | None) -> str:
    if value is None:
        return f"{dt.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}.json"
    file_name = str(value or "").strip()
    if not file_name:
        raise click.ClickException("--resume requires a configuration file name.")
    if Path(file_name).name != file_name:
        raise click.ClickException("--resume expects a file name, not a path.")
    if not file_name.endswith((".json", ".csv")):
        file_name += ".json"
    return file_name


def _current_timestamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")


def _default_plan_path(configuration_file_name: str | None) -> Path:
    return DEFAULT_CONFIGURATION_DIR / _resolve_configuration_file_name(configuration_file_name)


def _resolve_resume_plan_path(value: str) -> Path:
    raw_value = str(value or "").strip()
    if not raw_value:
        raise click.ClickException("--resume requires a configuration file path.")
    candidate = Path(raw_value).expanduser()
    if not candidate.name.endswith((".json", ".csv")):
        candidate = candidate.with_name(candidate.name + ".json")
    if candidate.is_absolute():
        return candidate
    if candidate.parent == Path("."):
        return DEFAULT_CONFIGURATION_DIR / candidate.name
    return ROOT / candidate


def _normalize_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload.get("runs")
    normalized_runs = list(runs) if isinstance(runs, list) else []
    return {
        "timestamp": str(payload.get("timestamp") or "").strip(),
        "name": str(payload.get("name") or "").strip(),
        "models": [
            str(item).strip()
            for item in payload.get("models") or []
            if str(item).strip()
        ],
        "number_of_runs_in_parallel_per_agent": int(
            payload.get("number_of_runs_in_parallel_per_agent")
            or DEFAULT_PARALLEL_PER_AGENT
        ),
        "total_runs": len(normalized_runs),
        "runs": normalized_runs,
    }


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    normalized_payload = _normalize_plan_payload(payload)
    tmp_path.write_text(json.dumps(normalized_payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _validate_run_entry(raw_run: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw_run, dict):
        raise click.ClickException(f"Configuration run entry at index {index} must be an object.")

    def _field(name: str, legacy_name: str | None = None) -> Any:
        if name in raw_run:
            return raw_run.get(name)
        if legacy_name is not None and legacy_name in raw_run:
            return raw_run.get(legacy_name)
        return None

    required = (
        ("question_slug", "QUESTION_SLUG"),
        ("model", "MODEL"),
        ("agent_id", "AGENT_ID"),
        ("agentic_run_id", "AGENTIC_RUN_ID"),
        ("status", None),
    )
    missing = [
        name
        for name, legacy_name in required
        if name not in raw_run and (legacy_name is None or legacy_name not in raw_run)
    ]
    if missing:
        raise click.ClickException(
            f"Configuration run entry at index {index} is missing required keys: {missing}."
        )
    status = str(_field("status") or "").strip()
    if status not in ALL_PLAN_STATUSES:
        raise click.ClickException(
            f"Unsupported plan status {status!r} at run index {index}."
        )
    run = {
        "question_slug": str(_field("question_slug", "QUESTION_SLUG") or "").strip(),
        "model": str(_field("model", "MODEL") or "").strip(),
        "agent_id": str(_field("agent_id", "AGENT_ID") or "").strip(),
        "agentic_run_id": str(_field("agentic_run_id", "AGENTIC_RUN_ID") or "").strip(),
        "status": status,
        "path_to_the_run": str(_field("path_to_the_run") or "").strip(),
        "agent_prompt": str(raw_run.get("agent_prompt") or ""),
    }
    if isinstance(raw_run.get("question_schema"), dict):
        run["question_schema"] = dict(raw_run["question_schema"])
    if raw_run.get("tag") is not None:
        run["tag"] = str(raw_run.get("tag") or "DEBUG").strip() or "DEBUG"
    if raw_run.get("error") is not None:
        run["error"] = raw_run.get("error")
    for key, value in raw_run.items():
        if str(key).startswith("question.recipe_info."):
            run[str(key)] = value
    return run


def _load_existing_plan(plan_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise click.ClickException(f"Configuration file not found: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in configuration file {plan_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"Configuration file must contain a JSON object: {plan_path}")
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list):
        raise click.ClickException(f"Configuration file {plan_path} must contain a 'runs' array.")
    runs = [_validate_run_entry(run, index=idx) for idx, run in enumerate(raw_runs)]
    if any(run["status"] == PLAN_STATUS_RUNNING for run in runs):
        raise click.ClickException(
            f"Configuration file {plan_path} still contains run(s) with status 'RUNNING'."
        )
    return _normalize_plan_payload({
        "timestamp": str(payload.get("timestamp") or "").strip(),
        "name": _resolve_run_name(str(payload.get("name") or "").strip()),
        "models": [str(item).strip() for item in payload.get("models") or [] if str(item).strip()],
        "number_of_runs_in_parallel_per_agent": int(
            payload.get("number_of_runs_in_parallel_per_agent") or DEFAULT_PARALLEL_PER_AGENT
        ),
        "runs": runs,
    })


def _adapter_cmd(
    *,
    model_name: str,
    question_slug: str,
) -> list[str]:
    source_dir = TSENV_QUESTIONS_ROOT / model_name
    dataset_dir = ROOT / "terminal-bench" / "tasks_runtime" / "manual" / model_name / question_slug
    return [
        str(ROOT / "env" / "bin" / "python"),
        str(ADAPTER_SCRIPT),
        str(source_dir),
        "--output-dir",
        str(dataset_dir),
        "--overwrite",
        "--only-question-id",
        question_slug,
    ]


def _write_noise_analysis_for_materialized_question(
    *,
    model_name: str,
    question_slug: str,
) -> Path:
    try:
        artifact_path = _driver_write_noise_analysis_for_materialized_question(
            model_name=model_name,
            question_id=question_slug,
            repo_root=ROOT,
        )
    except Exception as exc:
        raise click.ClickException(
            "Failed to write noise_analysis.json for "
            f"{model_name}/{question_slug}: {exc}"
        ) from exc
    print(f"Wrote noise analysis for {model_name}/{question_slug}: {artifact_path}")
    return artifact_path


def _materialize_selected_questions_with_adapter(
    *,
    selected_question_keys: list[str],
    question_catalog: dict[str, dict[str, Any]],
) -> None:
    for question_key in selected_question_keys:
        question_info = question_catalog[question_key]
        model_name = str(question_info["model"])
        question_slug = str(question_info["question_slug"])
        _run_cmd(
            _adapter_cmd(
                model_name=model_name,
                question_slug=question_slug,
            ),
            cwd=ROOT,
        )
        _write_noise_analysis_for_materialized_question(
            model_name=model_name,
            question_slug=question_slug,
        )


def _delete_healed_run_path(raw_path: str) -> None:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return
    target = Path(path_text).expanduser().resolve()
    runs_root = (ROOT / "terminal-bench" / "runs").resolve()
    try:
        target.relative_to(runs_root)
    except ValueError as exc:
        raise click.ClickException(
            f"Refusing to delete path_to_the_run outside terminal-bench/runs during --heal: {target}"
        ) from exc
    if target == runs_root:
        raise click.ClickException(
            f"Refusing to delete terminal-bench runs root during --heal: {target}"
        )
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def _heal_error_runs(plan: dict[str, Any]) -> int:
    healed_count = 0
    for run in plan.get("runs") or []:
        if not isinstance(run, dict) or run.get("status") != PLAN_STATUS_ERROR:
            continue
        _delete_healed_run_path(str(run.get("path_to_the_run") or ""))
        run["status"] = PLAN_STATUS_PENDING
        run.pop("error", None)
        run["path_to_the_run"] = ""
        healed_count += 1
    return healed_count


def _driver_cmd(
    *,
    model_name: str,
    question_slug: str,
    agent_id: str,
    run_id: str,
) -> list[str]:
    return [
        str(ROOT / "env" / "bin" / "python"),
        str(DRIVER_SCRIPT),
        "--model",
        model_name,
        "--question-slug",
        question_slug,
        "--agent-id",
        agent_id,
        "--agentic-run-id",
        run_id,
    ]


def _update_run_status(
    plan: dict[str, Any],
    *,
    run: dict[str, Any],
    plan_path: Path,
    status: str,
    error: str | None,
    lock: threading.Lock,
) -> None:
    with lock:
        run["status"] = status
        if error is None:
            run.pop("error", None)
        else:
            run["error"] = error
        run["path_to_the_run"] = _run_path_for_id(run["agentic_run_id"])
        _write_json_file(plan_path, plan)


def _run_plan(
    plan: dict[str, Any],
    *,
    plan_path: Path,
    number_runs_in_parallel_per_agent: int,
    tag: str = "DEBUG",
) -> None:
    runs = [run for run in plan["runs"] if run["status"] in RESUMABLE_PLAN_STATUSES]
    if not runs:
        print(f"No pending runs found in {plan_path}.")
        return

    runs_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        runs_by_agent[run["agent_id"]].append(run)

    lock = threading.Lock()
    errors: list[str] = []

    def _execute_run(run: dict[str, Any]) -> str | None:
        _update_run_status(
            plan,
            run=run,
            plan_path=plan_path,
            status=PLAN_STATUS_RUNNING,
            error=None,
            lock=lock,
        )
        print(
            "=== Running question: "
            f"{run['model']}/{run['question_slug']} "
            f"(agent_id={run['agent_id']}, run_id={run['agentic_run_id']}) ==="
        )
        try:
            env = os.environ.copy()
            env["TSENV_AGENTIC_ROLLOUT_TAG"] = str(run.get("tag") or tag or "DEBUG")
            env["TSENV_AGENTIC_CONFIGURATION_FILE_NAME"] = plan_path.name
            _run_cmd(
                [
                    *_driver_cmd(
                        model_name=run["model"],
                        question_slug=run["question_slug"],
                        agent_id=run["agent_id"],
                        run_id=run["agentic_run_id"],
                    ),
                ],
                cwd=ROOT,
                env=env,
            )
        except Exception as exc:
            error_text = str(exc)
            if _valid_scores_path_for_run(run) is not None:
                _update_run_status(
                    plan,
                    run=run,
                    plan_path=plan_path,
                    status=PLAN_STATUS_DONE,
                    error=None,
                    lock=lock,
                )
                return None
            _update_run_status(
                plan,
                run=run,
                plan_path=plan_path,
                status=PLAN_STATUS_ERROR,
                error=error_text,
                lock=lock,
            )
            return error_text
        _update_run_status(
            plan,
            run=run,
            plan_path=plan_path,
            status=PLAN_STATUS_DONE,
            error=None,
            lock=lock,
        )
        return None

    executors: list[concurrent.futures.ThreadPoolExecutor] = []
    futures: dict[concurrent.futures.Future[str | None], dict[str, Any]] = {}
    try:
        for agent_id in _dedupe_preserve_order([run["agent_id"] for run in plan["runs"]]):
            agent_runs = runs_by_agent.get(agent_id)
            if not agent_runs:
                continue
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=number_runs_in_parallel_per_agent
            )
            executors.append(executor)
            for run in agent_runs:
                futures[executor.submit(_execute_run, run)] = run
        for future in concurrent.futures.as_completed(futures):
            run = futures[future]
            error_text = future.result()
            if error_text:
                errors.append(
                    f"{run['agent_id']} {run['question_slug']} {run['agentic_run_id']}: {error_text}"
                )
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    if errors:
        raise click.ClickException("One or more configured runs failed: " + "; ".join(errors))


@click.command(
    help="Create and run rollout configurations via workflows/rollout/run_single_question.py."
)
@click.option(
    "--tasks-dir",
    type=click.Path(path_type=Path),
    default=TSENV_QUESTIONS_ROOT,
    show_default=True,
    help="tsENV question root or a single model directory under tsENV_questions.",
)
@click.option(
    "--agent-id",
    cls=MultiValueOption,
    multiple=True,
    help="One or more agent profile IDs from shared/config/agents.json.",
)
@click.option(
    "--question-slug",
    cls=MultiValueOption,
    multiple=True,
    help="One or more question slugs to run.",
)
@click.option(
    "--row-slug",
    cls=MultiValueOption,
    multiple=True,
    help="One or more row slugs to run.",
)
@click.option(
    "--shot-slug",
    cls=MultiValueOption,
    multiple=True,
    help="One or more shot slugs to run.",
)
@click.option(
    "--model",
    cls=MultiValueOption,
    multiple=True,
    help="One or more model directory names under tsENV_questions.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Write the configuration JSON file and stop before launching runs.",
)
@click.option(
    "--resume",
    default=None,
    metavar="CONFIGURATION_FILE_PATH",
    help="Resume an existing configuration path. Bare filenames resolve under run_configurations/.",
)
@click.option(
    "--heal",
    is_flag=True,
    help="When resuming, delete errored run directories and reset ERROR runs to PENDING.",
)
@click.option(
    "--number-of-runs-in-parallel-per-agent",
    "number_runs_in_parallel_per_agent",
    type=int,
    default=None,
    help="Maximum number of concurrent runs to launch for each agent.",
)
@click.option(
    "--tag",
    default="DEBUG",
    show_default=True,
    help="Tag to attach to the run.",
)
@click.option(
    "--call-adapter",
    is_flag=True,
    help="Materialize selected Terminal-Bench task folders during orchestration.",
)
@click.option(
    "--name",
    "run_name",
    default=None,
    help="Configuration file name under run_configurations/. Defaults to the current timestamp.",
)
def main(
    tasks_dir: Path,
    agent_id: tuple[Any, ...],
    question_slug: tuple[Any, ...],
    row_slug: tuple[Any, ...],
    shot_slug: tuple[Any, ...],
    model: tuple[Any, ...],
    dry_run: bool,
    resume: str | None,
    heal: bool,
    number_runs_in_parallel_per_agent: int | None,
    tag: str,
    call_adapter: bool,
    run_name: str | None,
) -> None:
    if not DRIVER_SCRIPT.exists():
        raise click.ClickException(f"Driver script not found: {DRIVER_SCRIPT}")
    if call_adapter and not ADAPTER_SCRIPT.exists():
        raise click.ClickException(f"Adapter script not found: {ADAPTER_SCRIPT}")

    selector_name, selector_values = _resolve_selector(
        question_slugs=question_slug,
        row_slugs=row_slug,
        shot_slugs=shot_slug,
        resume=resume,
    )
    normalized_tag = str(tag or "").strip() or "DEBUG"
    effective_parallelism = _validate_parallelism(number_runs_in_parallel_per_agent)
    if heal and selector_name != "resume":
        raise click.UsageError("--heal can only be used with --resume.")

    if selector_name == "resume":
        if (
            _flatten_cli_values(agent_id)
            or _flatten_cli_values(question_slug)
            or _flatten_cli_values(row_slug)
            or _flatten_cli_values(shot_slug)
            or _flatten_cli_values(model)
            or dry_run
            or call_adapter
            or run_name
            or number_runs_in_parallel_per_agent is not None
            or str(tag or "").strip() != "DEBUG"
        ):
            raise click.UsageError(
                "When --resume is set, do not pass any other option except --heal."
            )
        plan_path = _resolve_resume_plan_path(selector_values[0]).resolve()
        if not plan_path.exists():
            raise click.ClickException(f"Configuration file not found: {plan_path}")
        plan = _load_existing_plan(plan_path)
        reconciled_count = _reconcile_scored_error_runs(plan)
        if reconciled_count:
            print(
                f"Reconciled {reconciled_count} ERROR run(s) with valid scores.json in {plan_path}"
            )
        if heal:
            healed_count = _heal_error_runs(plan)
            print(f"Healed {healed_count} ERROR run(s) in {plan_path}")
        effective_parallelism = int(
            plan.get("number_of_runs_in_parallel_per_agent")
            or DEFAULT_PARALLEL_PER_AGENT
        )
        _write_json_file(plan_path, plan)
        print(f"Configuration loaded from {plan_path}")
        _run_plan(
            plan,
            plan_path=plan_path,
            number_runs_in_parallel_per_agent=effective_parallelism,
            tag=normalized_tag,
        )
        return

    agent_ids = _validate_agent_ids(agent_id)
    if not agent_ids:
        raise click.UsageError("Pass at least one --agent-id.")
    requested_models = _validate_models(model)
    plan_path = _default_plan_path(run_name).resolve()
    resolved_run_name = plan_path.stem
    tasks_dir = tasks_dir.expanduser().resolve()
    question_catalog, row_to_question_keys, shot_to_question_keys, question_slug_to_keys = _build_question_catalog(
        tasks_dir,
        requested_models=requested_models,
    )
    selected_question_keys = _resolve_selected_question_slugs(
        selector_name=selector_name,
        selector_values=selector_values,
        question_catalog=question_catalog,
        row_to_question_keys=row_to_question_keys,
        shot_to_question_keys=shot_to_question_keys,
        question_slug_to_keys=question_slug_to_keys,
    )
    plan = _build_plan(
        tasks_dir=tasks_dir,
        agent_ids=agent_ids,
        requested_models=requested_models,
        selector_name=selector_name,
        selector_values=selector_values,
        selected_question_keys=selected_question_keys,
        question_catalog=question_catalog,
        number_runs_in_parallel_per_agent=effective_parallelism,
        tag=normalized_tag,
        run_name=resolved_run_name,
        timestamp=resolved_run_name if run_name is None else _current_timestamp(),
        call_adapter=call_adapter,
    )
    if call_adapter:
        _materialize_selected_questions_with_adapter(
            selected_question_keys=selected_question_keys,
            question_catalog=question_catalog,
        )
    if plan_path.exists():
        raise click.ClickException(
            f"Configuration file already exists: {plan_path}. Wait a second and retry."
        )
    _write_json_file(plan_path, plan)
    print(f"Configuration written to {plan_path}")
    if dry_run:
        return
    _run_plan(
        plan,
        plan_path=plan_path,
        number_runs_in_parallel_per_agent=effective_parallelism,
        tag=normalized_tag,
    )


if __name__ == "__main__":
    main()
