#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import os
import re
import shlex
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.agentic_rollout_paths import (
    canonical_trajectory_path,
    resolve_light_trajectory_path,
)
from shared.benchmark_utils import AGENTIC_PROFILES_BY_ID, agentic_profile_by_id
from shared.noise_analysis import (
    DEFAULT_LOCAL_NOISE_ANALYSIS_RADIUS_ROWS,
    quantify_analysis,
)
from shared.prompts import render_tsenv_agent_prompt
from shared.tsenv_task_materialization import materialize
from shared.tsenv_eval_mode import normalize_tsenv_eval_mode
from shared.tsenv_metadata import (
    label_for_question_sample,
    load_metadata_payload,
    metadata_questions_by_id,
    question_sample_paths,
    resolve_tsenv_payload_path,
)

TERMINAL_BENCH_DIR = ROOT_DIR / "terminal-bench"
PYTHON_EXE = ROOT_DIR / "env" / "bin" / "python"
DOCKER_TIMEOUT_KILL_GRACE_SEC = 10
QUESTION_TIMEOUT_SEC_DEFAULT = 3600


def _usage_text() -> str:
    return """Usage:
  workflows/rollout/run_single_question.py --model MODEL --question-slug QUESTION_SLUG --agent-id AGENT_ID --agentic-run-id AGENTIC_RUN_ID [--keep-container] [--single-run]

Examples:
  workflows/rollout/run_single_question.py --model BallDrop --question-slug q_8169c2ed7a762d74 --agent-id gpt_5_5_codex_high --agentic-run-id my_run

Options:
  --keep-container   Keep the client container alive after completion
  --single-run       Isolated debugging mode; allow Docker image cleanup after completion

Environment overrides:
  Uses the selected agent profile's required environment variables.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workflows/rollout/run_single_question.py",
        add_help=False,
        usage=argparse.SUPPRESS,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=_usage_text(),
    )
    parser.add_argument(
        "--keep-container",
        dest="keep_container",
        action="store_true",
    )
    parser.add_argument(
        "--single-run",
        dest="single_run",
        action="store_true",
    )
    parser.add_argument("--model", dest="model_name")
    parser.add_argument("--question-slug", dest="question_slug")
    parser.add_argument("--agent-id", dest="agent_id")
    parser.add_argument("--agentic-run-id", dest="agentic_run_id")
    parser.add_argument("-h", "--help", action="help", help="show this help message and exit")
    return parser


def _resolve_cli_contract(
    args: argparse.Namespace,
) -> tuple[str, str, str, str | None, int | None]:
    flagged_values = {
        "model": args.model_name,
        "question_slug": args.question_slug,
        "agent_id": args.agent_id,
        "agentic_run_id": args.agentic_run_id,
    }
    missing = [
        f"--{name.replace('_', '-')}"
        for name, value in flagged_values.items()
        if not str(value or "").strip()
    ]
    if missing:
        raise SystemExit(
            "Error: documented invocation requires all of "
            "--model, --question-slug, --agent-id, and --agentic-run-id. "
            f"Missing: {', '.join(missing)}."
        )
    return (
        str(args.question_slug).strip(),
        str(args.agent_id).strip(),
        str(args.agentic_run_id).strip(),
        str(args.model_name).strip(),
        None,
    )


def _validate_agent(agent: str) -> None:
    agent_name_path = TERMINAL_BENCH_DIR / "terminal_bench" / "agents" / "agent_name.py"
    if not agent_name_path.exists():
        raise SystemExit(f"Error: could not find agent enum file: {agent_name_path}")
    tree = ast.parse(agent_name_path.read_text(encoding="utf-8"), filename=str(agent_name_path))
    valid_agents: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AgentName":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    value_node = stmt.value
                    if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                        valid_agents.append(value_node.value)
            break
    if agent not in valid_agents:
        valid_text = "\n".join(f"  - {item}" for item in valid_agents)
        raise SystemExit(
            f"Error: unsupported --agent '{agent}'.\nValid agents:\n{valid_text}"
        )


def _load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_question_source(
    question_id: str,
    expected_model: str | None = None,
) -> tuple[str, Path]:
    tsenv_root = ROOT_DIR / "tsENV_questions"
    if not tsenv_root.is_dir():
        raise SystemExit(f"Error: tsENV question root not found: {tsenv_root}")

    matches: list[tuple[str, Path]] = []
    examples: list[str] = []
    model_filter = str(expected_model or "").strip()
    question_glob = (
        [tsenv_root / model_filter / "questions.json"]
        if model_filter
        else sorted(tsenv_root.glob("*/questions.json"))
    )
    for questions_path in question_glob:
        if not questions_path.exists():
            continue
        source_dir = questions_path.parent
        payload = load_metadata_payload(questions_path)
        for qid in sorted(metadata_questions_by_id(payload).keys()):
            if len(examples) < 10:
                examples.append(qid)
            if qid == question_id:
                matches.append((source_dir.name, source_dir))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        locations = ", ".join(f"{model} ({path})" for model, path in matches)
        raise SystemExit(
            f"Error: question_id '{question_id}' is ambiguous across tsENV_questions: {locations}"
        )

    if model_filter:
        lines = [
            f"Error: question_id '{question_id}' not found under {tsenv_root / model_filter}"
        ]
    else:
        lines = [f"Error: question_id '{question_id}' not found under {tsenv_root}"]
    if examples:
        contains = [item for item in examples if question_id in item]
        sample = contains[:10] if contains else examples[:10]
        lines.append("Available question_id examples:")
        lines.extend(f"  - {item}" for item in sample)
    raise SystemExit("\n".join(lines))


def _load_question_payload(source_dir: Path, question_id: str) -> tuple[dict, dict]:
    payload_path = resolve_tsenv_payload_path(source_dir)
    payload = load_metadata_payload(payload_path)
    questions = metadata_questions_by_id(payload)
    question = questions.get(question_id)
    if not isinstance(question, dict):
        raise SystemExit(f"Error: question_id '{question_id}' not found in {payload_path}")
    return payload, dict(question)


def _question_context(question: dict[str, Any]) -> str:
    desc_level = str(question.get("desc_level") or "").strip().lower()
    if desc_level:
        return desc_level
    context = str(question.get("context") or "").strip().lower()
    if context:
        return context
    recipe_info = question.get("recipe_info")
    if isinstance(recipe_info, dict):
        desc_level = str(recipe_info.get("desc_level") or "").strip().lower()
        if desc_level:
            return desc_level
        return str(recipe_info.get("context") or "").strip().lower()
    recipe = question.get("recipe")
    if isinstance(recipe, dict):
        resolved = recipe.get("resolved")
        if isinstance(resolved, dict):
            return str(resolved.get("context") or "").strip().lower()
    return ""


def _uses_ground_truth_context(question: dict[str, Any]) -> bool:
    recipe_info = question.get("recipe_info")
    if isinstance(recipe_info, dict):
        textual_context = str(recipe_info.get("textual_context") or "").strip().lower()
        if textual_context:
            return textual_context == "ground_truth"
        desc_level = str(recipe_info.get("desc_level") or "").strip().lower()
        if desc_level == "ground_truth":
            return True
    return _question_context(question) == "ground_truth"


def _resolve_noise_adder_source(source_dir: Path) -> Path:
    noise_adder_path = source_dir / "noise_adder.py"
    if not noise_adder_path.exists():
        raise SystemExit(f"Error: noise_adder.py not found at {noise_adder_path}")
    return noise_adder_path


def _normalize_noise_profile(raw_profile: object) -> str:
    normalized = str(raw_profile or "").strip().lower()
    aliases = {
        "": "none",
        "noise_none": "none",
        "noise_low": "low",
        "noise_medium": "medium",
        "noise_high": "high",
    }
    return aliases.get(normalized, normalized) or "none"


def _active_materialized_noise_profile(question: dict, scenario_info: dict) -> str:
    recipe_info = question.get("recipe_info")
    if isinstance(recipe_info, dict):
        for key in ("noise_level", "noise"):
            if recipe_info.get(key) is not None:
                return _normalize_noise_profile(recipe_info.get(key))
    recipe = question.get("recipe")
    if isinstance(recipe, dict):
        resolved = recipe.get("resolved")
        if isinstance(resolved, dict):
            for key in ("noise_profile", "noise"):
                if resolved.get(key) is not None:
                    return _normalize_noise_profile(resolved.get(key))
    return _normalize_noise_profile(scenario_info.get("noise_level"))


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if check and proc.returncode != 0:
        raise SystemExit(
            f"Error: command failed with exit code {proc.returncode}: {' '.join(cmd)}"
    )
    return proc


def _populate_required_environment_variables(
    env: dict[str, str],
    required_environmental_variable: dict[str, str],
) -> None:
    for key, command in required_environmental_variable.items():
        if str(env.get(key) or "").strip():
            continue
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(ROOT_DIR),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip()
            raise SystemExit(
                f"Error: failed to derive required environment variable {key}."
                + (f" {detail}" if detail else "")
            )
        value = (proc.stdout or "").strip()
        if not value:
            raise SystemExit(
                f"Error: command for required environment variable {key} produced an empty value."
            )
        env[key] = value


def _read_dotenv_values(keys: Iterable[str], env: dict[str, str]) -> dict[str, str]:
    names = [str(key).strip() for key in keys if str(key).strip()]
    if not names:
        return {}
    script = (
        "if [ -f ./.env ]; then set -a; source ./.env; set +a; fi\n"
        f"{shlex.quote(sys.executable)} - <<'PY'\n"
        "import os\n"
        f"keys = {names!r}\n"
        "for key in keys:\n"
        "    print(f'{key}={os.environ.get(key, \"\")}')\n"
        "PY"
    )
    proc = subprocess.run(
        ["bash", "-lc", script],
        cwd=str(ROOT_DIR),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip()
        raise SystemExit(
            "Error: failed to source Gemini auth settings from .env."
            + (f" {detail}" if detail else "")
        )
    values: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in names:
            values[key] = value.strip()
    return values


def _populate_gemini_auth_environment(env: dict[str, str]) -> None:
    dotenv_values = _read_dotenv_values(
        ("GEMINI_AUTH_MODE", "GEMINI_CLI_AUTH", "GEMINI_API_KEY"),
        env,
    )
    if not str(env.get("GEMINI_AUTH_MODE") or "").strip():
        value = str(dotenv_values.get("GEMINI_AUTH_MODE") or "").strip()
        env["GEMINI_AUTH_MODE"] = value or "cli"

    mode = str(env.get("GEMINI_AUTH_MODE") or "").strip().lower()
    env["GEMINI_AUTH_MODE"] = mode
    if mode not in {"api", "cli", "both"}:
        return

    if mode in {"cli", "both"} and not str(env.get("GEMINI_CLI_AUTH") or "").strip():
        value = str(dotenv_values.get("GEMINI_CLI_AUTH") or "").strip()
        env["GEMINI_CLI_AUTH"] = value or "~/.gemini"
    if mode in {"api", "both"} and not str(env.get("GEMINI_API_KEY") or "").strip():
        value = str(dotenv_values.get("GEMINI_API_KEY") or "").strip()
        if value:
            env["GEMINI_API_KEY"] = value
    if mode == "cli":
        env.pop("GEMINI_API_KEY", None)
    elif mode == "api":
        env.pop("GEMINI_CLI_AUTH", None)


def _populate_claude_auth_environment(
    env: dict[str, str],
    required_environmental_variable: dict[str, str],
) -> None:
    dotenv_values = _read_dotenv_values(
        ("CLAUDE_AUTH_MODE", "CLAUDE_CODE_AUTH_DIR"),
        env,
    )
    if not str(env.get("CLAUDE_AUTH_MODE") or "").strip():
        value = str(dotenv_values.get("CLAUDE_AUTH_MODE") or "").strip()
        if value:
            env["CLAUDE_AUTH_MODE"] = value

    mode = str(env.get("CLAUDE_AUTH_MODE") or "").strip().lower()
    if mode not in {"subscription", "both"}:
        return

    env["CLAUDE_AUTH_MODE"] = mode
    if mode == "subscription":
        required_environmental_variable.pop("ANTHROPIC_API_KEY", None)
    if not str(env.get("CLAUDE_CODE_AUTH_DIR") or "").strip():
        value = str(dotenv_values.get("CLAUDE_CODE_AUTH_DIR") or "").strip()
        if value:
            env["CLAUDE_CODE_AUTH_DIR"] = value


def _ensure_auth(agent: str, env: dict[str, str]) -> None:
    if agent == "codex":
        if not str(env.get("CODEX_AUTH_JSON_BASE64") or "").strip():
            raise SystemExit(
                "Error: codex requires CODEX_AUTH_JSON_BASE64."
            )
        return
    if agent == "gemini-cli":
        has_api = bool(str(env.get("GEMINI_API_KEY") or "").strip())
        has_cli = bool(str(env.get("GEMINI_CLI_AUTH") or "").strip())
        mode = str(env.get("GEMINI_AUTH_MODE") or "").strip().lower()
        if mode and mode not in {"api", "cli", "both"}:
            raise SystemExit(
                "Error: GEMINI_AUTH_MODE must be one of: api, cli, both."
            )
        if not has_api and not has_cli:
            raise SystemExit(
                "Error: gemini-cli requires GEMINI_API_KEY or GEMINI_CLI_AUTH."
            )
        if mode:
            if mode == "api":
                if not has_api:
                    raise SystemExit(
                        "Error: GEMINI_AUTH_MODE=api requires GEMINI_API_KEY."
                    )
            elif mode == "cli":
                if not has_cli:
                    raise SystemExit(
                        "Error: GEMINI_AUTH_MODE=cli requires GEMINI_CLI_AUTH."
                    )
                if has_api:
                    raise SystemExit(
                        "Error: GEMINI_AUTH_MODE=cli forbids GEMINI_API_KEY."
                    )
            else:
                if not has_api or not has_cli:
                    raise SystemExit(
                        "Error: GEMINI_AUTH_MODE=both requires GEMINI_API_KEY and GEMINI_CLI_AUTH."
                    )
        elif has_api and has_cli:
            raise SystemExit(
                "Error: both GEMINI_API_KEY and GEMINI_CLI_AUTH are set. Set GEMINI_AUTH_MODE to 'api', 'cli', or 'both'."
            )
        return
    if agent == "claude-code":
        _resolve_claude_auth_mode(env)


def _resolve_claude_auth_mode(env: dict[str, str]) -> str:
    raw_mode = str(env.get("CLAUDE_AUTH_MODE") or "").strip().lower()
    api_key = str(env.get("ANTHROPIC_API_KEY") or "").strip()
    auth_dir = str(env.get("CLAUDE_CODE_AUTH_DIR") or "").strip()
    has_api = bool(api_key)
    has_subscription = bool(auth_dir)

    def _validate_auth_dir() -> None:
        auth_path = Path(auth_dir).expanduser()
        if not auth_path.is_dir():
            raise SystemExit(
                f"Error: CLAUDE_CODE_AUTH_DIR must point to an existing directory, got {auth_path}."
            )

    if raw_mode:
        if raw_mode not in {"subscription", "api", "both"}:
            raise SystemExit(
                "Error: CLAUDE_AUTH_MODE must be one of: subscription, api, both."
            )
        if raw_mode == "subscription":
            if not has_subscription:
                raise SystemExit(
                    "Error: CLAUDE_AUTH_MODE=subscription requires CLAUDE_CODE_AUTH_DIR."
                )
            _validate_auth_dir()
        elif raw_mode == "api":
            if not has_api:
                raise SystemExit(
                    "Error: CLAUDE_AUTH_MODE=api requires ANTHROPIC_API_KEY."
                )
        else:
            if not has_api or not has_subscription:
                raise SystemExit(
                    "Error: CLAUDE_AUTH_MODE=both requires ANTHROPIC_API_KEY and CLAUDE_CODE_AUTH_DIR."
                )
            _validate_auth_dir()
        return raw_mode

    if has_api:
        return "api"
    raise SystemExit(
        "Error: claude-code requires ANTHROPIC_API_KEY or CLAUDE_CODE_AUTH_DIR. "
        "Set ANTHROPIC_API_KEY, or set CLAUDE_AUTH_MODE=subscription with CLAUDE_CODE_AUTH_DIR."
    )


def _configure_agent_auth_env(agent: str, env: dict[str, str]) -> None:
    if agent != "claude-code":
        return
    mode = _resolve_claude_auth_mode(env)
    env["CLAUDE_AUTH_MODE"] = mode
    if mode == "subscription":
        env.pop("ANTHROPIC_API_KEY", None)
    elif mode == "api":
        env.pop("CLAUDE_CODE_AUTH_DIR", None)


def _validate_explicit_run_id(value: str) -> str:
    run_id = str(value).strip()
    if not run_id:
        raise SystemExit("Error: AGENTIC_RUN_ID must be non-empty.")
    if Path(run_id).name != run_id or any(part == ".." for part in Path(run_id).parts):
        raise SystemExit(f"Error: invalid AGENTIC_RUN_ID {run_id!r}.")
    return run_id


def _parse_batch_size(value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise SystemExit(f"Error: --batch-size must be positive, got {value}.")
    return int(value)


def _sanitize_run_id(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


def _compute_run_id(agent_id: str, dataset_model: str, question_id: str) -> str:
    raw = (
        f"{dt.datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"
        f"__{agent_id}__{dataset_model}__{question_id}"
    )
    run_id = _sanitize_run_id(raw)
    if not run_id:
        raise SystemExit("Error: failed to compute a non-empty AGENTIC_RUN_ID.")
    return run_id


def _materialize_dataset(source_dir: Path, dataset_dir: Path, question_id: str) -> None:
    adapter = TERMINAL_BENCH_DIR / "adapters" / "tsENV" / "run_adapter.py"
    _run_cmd(
        [
            str(PYTHON_EXE),
            str(adapter),
            str(source_dir),
            "--output-dir",
            str(dataset_dir),
            "--overwrite",
            "--only-question-id",
            question_id,
        ],
        cwd=ROOT_DIR,
    )


def _question_schema_for_comparison(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    return {key: value for key, value in schema.items() if key != "question_id"}


def _question_schema_version_matches(source_schema: object, materialized_schema: object) -> bool:
    comparable_source = _question_schema_for_comparison(source_schema)
    comparable_materialized = _question_schema_for_comparison(materialized_schema)
    if comparable_source == comparable_materialized:
        return True
    if not isinstance(comparable_source, dict):
        return False
    source_version = comparable_source.get("version")
    if source_version is None:
        return False
    if comparable_materialized == source_version:
        return True
    return (
        isinstance(comparable_materialized, dict)
        and comparable_materialized.get("version") == source_version
    )


def _materialized_question_matches_source(question_root: Path, question: dict) -> bool:
    scenario_info_path = question_root / "scenario_info.json"
    if not question_root.is_dir() or not scenario_info_path.exists():
        return False
    try:
        scenario_info = _load_json(scenario_info_path)
    except Exception:
        return False
    if not isinstance(scenario_info, dict):
        return False
    return _question_schema_version_matches(question, scenario_info.get("question_schema"))


def _validate_relative_sample_path(raw_path: object) -> Path:
    rel_path = Path(str(raw_path).strip())
    if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
        raise SystemExit(f"Error: invalid relative sample path {raw_path!r}.")
    return rel_path


def _safe_tmp_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip()).strip("._")
    if not safe:
        raise SystemExit("Error: question slug must be non-empty for tmp export.")
    return safe


def _build_train_labels_payload(
    *,
    question: dict,
    payload: dict,
    materialized_train_samples: object,
) -> dict[str, str]:
    source_train_samples = question_sample_paths(payload, question=question, subset="train")
    basename_to_label: dict[str, str] = {}
    for raw_sample in source_train_samples:
        source_ref = str(raw_sample).strip()
        if not source_ref:
            raise SystemExit("Error: questions.json question contains empty train_samples entry.")
        basename = Path(source_ref).name
        if basename in basename_to_label:
            raise SystemExit(
                f"Error: duplicate training sample basename {basename!r} prevents label export."
            )
        try:
            label = label_for_question_sample(
                payload,
                question=question,
                sample_path=source_ref,
            )
        except Exception as exc:
            raise SystemExit(
                f"Error: missing ground-truth label for training sample {source_ref!r}: {exc}"
            ) from exc
        basename_to_label[basename] = label

    if not isinstance(materialized_train_samples, list):
        raise SystemExit("Error: scenario_info.json train_samples must be a list.")
    train_labels: dict[str, str] = {}
    for raw_sample in materialized_train_samples:
        rel_path = _validate_relative_sample_path(raw_sample)
        basename = rel_path.name
        label = basename_to_label.get(basename)
        if not label:
            raise SystemExit(
                f"Error: could not map materialized training sample {basename!r} back to a label."
            )
        sample_key = rel_path.name
        if sample_key in train_labels:
            raise SystemExit(
                f"Error: duplicate materialized training sample filename {sample_key!r}."
            )
        train_labels[sample_key] = label
    return train_labels


def _documented_tmp_export_dir(agentic_run_id: str) -> Path:
    return ROOT_DIR / "tmp" / _safe_tmp_name(agentic_run_id) / "agent_payload"


def _export_materialized_samples(
    *,
    question_root: Path,
    question: dict,
    payload: dict,
    question_slug: str,
    agentic_run_id: str,
) -> Path:
    scenario_info_path = question_root / "scenario_info.json"
    if not scenario_info_path.exists():
        raise SystemExit(f"Error: scenario_info.json not found at {scenario_info_path}")
    scenario_info = _load_json(scenario_info_path)
    if not isinstance(scenario_info, dict):
        raise SystemExit(f"Error: scenario_info.json must contain an object: {scenario_info_path}")

    source_payload_dir = question_root / "agent_payload"
    if not source_payload_dir.is_dir():
        raise SystemExit(f"Error: agent_payload directory not found at {source_payload_dir}")

    export_dir = _documented_tmp_export_dir(agentic_run_id)
    export_root = export_dir.parent
    tmp_root = export_root.with_name(
        f".{export_root.name}.tmp.{os.getpid()}.{_safe_tmp_name(agentic_run_id)}"
    )
    tmp_dir = tmp_root / "agent_payload"
    shutil.rmtree(export_root, ignore_errors=True)
    shutil.rmtree(tmp_root, ignore_errors=True)

    try:
        tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_payload_dir, tmp_dir)
        tmp_root.rename(export_root)
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    return export_dir


def _render_agent_prompt_for_question(
    *,
    payload: dict,
    question: dict,
    question_slug: str,
) -> str:
    question_text = question.get("question_text")
    if not isinstance(question_text, dict):
        return ""
    try:
        questions_by_id = metadata_questions_by_id(payload)
    except TypeError:
        questions_by_id = {question_slug: question}
    return render_tsenv_agent_prompt(
        question_text,
        question_slug=question_slug,
        questions_by_id=questions_by_id,
        questions_metadata=payload,
    )


def _write_agent_prompt_log(
    *,
    payload: dict,
    question: dict,
    question_slug: str,
) -> Path | None:
    prompt = _render_agent_prompt_for_question(
        payload=payload,
        question=question,
        question_slug=question_slug,
    )
    if not prompt:
        return None
    logs_dir = ROOT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = logs_dir / f"{_safe_tmp_name(question_slug)}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    return prompt_path


def _load_rollout_signal_type(
    model_name: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    root_dir = repo_root or ROOT_DIR
    config_path = root_dir / "models" / "simulink" / model_name / "experiment_config.json"
    if not config_path.exists():
        return None
    try:
        payload = _load_json(config_path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    observable = payload.get("observable_signals")
    if not isinstance(observable, dict):
        return None
    signal_type = observable.get("signal_type")
    return dict(signal_type) if isinstance(signal_type, dict) else None


def _local_noise_radius_rows_from_signal_type(
    signal_type: dict[str, Any] | None,
) -> int:
    if not isinstance(signal_type, dict):
        return DEFAULT_LOCAL_NOISE_ANALYSIS_RADIUS_ROWS
    sizes: list[int] = []
    for raw_spec in signal_type.values():
        if not isinstance(raw_spec, dict):
            continue
        try:
            envelope_size = int(raw_spec.get("envelope_size"))
        except (TypeError, ValueError):
            continue
        if envelope_size >= 0:
            sizes.append(envelope_size)
    if not sizes:
        return DEFAULT_LOCAL_NOISE_ANALYSIS_RADIUS_ROWS
    return max(sizes)


def _first_diff_for_sample(payload: dict, sample_uuid: str) -> float | None:
    ground_truth = payload.get("ground_truth_information")
    if not isinstance(ground_truth, dict):
        return None
    interventions = ground_truth.get("interventions")
    if not isinstance(interventions, dict):
        return None
    entry = interventions.get(sample_uuid)
    if not isinstance(entry, dict):
        return None
    value = entry.get("first_diff")
    if value is None:
        return None
    if isinstance(value, list):
        parsed_values: list[float] = []
        for item in value:
            if item is None:
                continue
            try:
                parsed = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed >= 0.0:
                parsed_values.append(parsed)
        return min(parsed_values) if parsed_values else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_noise_analysis_payload(value: Any) -> dict[str, Any] | None:
    def _is_snr_value(item: Any) -> bool:
        if item is None or item == "-inf":
            return True
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        return math.isfinite(float(item))

    if not isinstance(value, dict):
        return None
    global_values = value.get("global")
    local_values = value.get("local")
    if not isinstance(global_values, list) and not isinstance(local_values, list):
        return None
    for raw_values in (global_values, local_values):
        if isinstance(raw_values, list) and not all(_is_snr_value(item) for item in raw_values):
            return None
    out = {
        "global": list(global_values) if isinstance(global_values, list) else [],
        "local": list(local_values) if isinstance(local_values, list) else [],
    }
    return out if out["global"] or out["local"] else None


def _write_noise_analysis_artifact(
    *,
    question_root: Path,
    source_dir: Path,
    model_name: str,
    question_slug: str,
    question: dict,
    payload: dict,
    scenario_info: dict,
    repo_root: Path | None = None,
) -> Path:
    recipe_info = question.get("recipe_info")
    recipe_info = recipe_info if isinstance(recipe_info, dict) else {}
    try:
        seed = int(recipe_info.get("question_seed") or 0)
    except (TypeError, ValueError):
        seed = 0
    signal_type = _load_rollout_signal_type(model_name, repo_root=repo_root)
    local_radius_rows = _local_noise_radius_rows_from_signal_type(signal_type)
    active_noise_profile = _active_materialized_noise_profile(question, scenario_info)
    artifacts: dict[str, Any] = {}
    for group_name in ("train_samples", "test_samples"):
        raw_paths = scenario_info.get(group_name)
        if not isinstance(raw_paths, list):
            continue
        for raw_path in raw_paths:
            rel_path = _validate_relative_sample_path(raw_path)
            sample_uuid = rel_path.stem
            try:
                first_diff = _first_diff_for_sample(payload, sample_uuid)
                clean_df, _ = materialize(
                    sample_uuid,
                    "none",
                    seed,
                    first_diff if first_diff is not None else -1.0,
                    tsenv_model_root=source_dir,
                )
                sample_analysis: dict[str, Any] = {
                    "global": [],
                    "local": [],
                }
                noisy_df, materialized_noise_analysis = materialize(
                    sample_uuid,
                    active_noise_profile,
                    seed,
                    first_diff if first_diff is not None else -1.0,
                    tsenv_model_root=source_dir,
                )
                if active_noise_profile != "none":
                    normalized = _normalized_noise_analysis_payload(materialized_noise_analysis)
                    if normalized is not None:
                        sample_analysis = normalized
                    else:
                        noise_analysis = quantify_analysis(
                            clean_df,
                            noisy_df,
                            signal_type=signal_type,
                            first_diff=first_diff,
                            local_radius_rows=local_radius_rows,
                        )
                        sample_analysis["global"] = noise_analysis.get("global", [])
                        sample_analysis["local"] = noise_analysis.get("local", [])
                else:
                    noise_analysis = quantify_analysis(
                        clean_df,
                        noisy_df,
                        signal_type=signal_type,
                        first_diff=first_diff,
                        local_radius_rows=local_radius_rows,
                    )
                    sample_analysis["global"] = noise_analysis.get("global", [])
                    sample_analysis["local"] = noise_analysis.get("local", [])
                artifacts[str(rel_path)] = sample_analysis
            except Exception as exc:
                artifacts[str(rel_path)] = {"error": str(exc)}
    artifact_path = question_root / "noise_analysis.json"
    artifact_path.write_text(
        json.dumps(artifacts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_path


def _write_train_labels_file(
    *,
    question_root: Path,
    question: dict,
    payload: dict,
    scenario_info: dict,
) -> Path | None:
    train_samples = scenario_info.get("train_samples")
    if not isinstance(train_samples, list) or not train_samples:
        return None
    train_labels = _build_train_labels_payload(
        question=question,
        payload=payload,
        materialized_train_samples=train_samples,
    )
    if not train_labels:
        return None
    labels_path = question_root / "train_labels.json"
    labels_path.write_text(
        json.dumps(train_labels, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return labels_path


def write_noise_analysis_for_materialized_question(
    *,
    model_name: str,
    question_id: str,
    repo_root: Path | None = None,
) -> Path:
    root_dir = repo_root or ROOT_DIR
    source_dir = root_dir / "tsENV_questions" / model_name
    payload, question = _load_question_payload(source_dir, question_id)
    question_root = (
        root_dir
        / "terminal-bench"
        / "tasks_runtime"
        / "manual"
        / model_name
        / question_id
        / "question_0"
    )
    if not question_root.is_dir():
        raise FileNotFoundError(
            f"Materialized question folder not found: {question_root}"
        )
    scenario_info_path = question_root / "scenario_info.json"
    if not scenario_info_path.exists():
        raise FileNotFoundError(f"scenario_info.json not found at {scenario_info_path}")
    scenario_info = _load_json(scenario_info_path)
    if not isinstance(scenario_info, dict):
        raise ValueError(f"scenario_info.json must contain an object: {scenario_info_path}")
    _write_train_labels_file(
        question_root=question_root,
        question=question,
        payload=payload,
        scenario_info=scenario_info,
    )
    _sync_agent_payload(question_root)
    return _write_noise_analysis_artifact(
        question_root=question_root,
        source_dir=source_dir,
        model_name=model_name,
        question_slug=question_id,
        question=question,
        payload=payload,
        scenario_info=scenario_info,
        repo_root=root_dir,
    )


def _sync_agent_payload(question_root: Path) -> None:
    payload_dir = question_root / "agent_payload"
    shutil.rmtree(payload_dir, ignore_errors=True)
    payload_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train_samples", "test_samples", "train_labels.json"):
        source = question_root / name
        if not source.exists():
            continue
        destination = payload_dir / name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _subset_mapping_by_basename(mapping: object, basenames: set[str]) -> object:
    if not isinstance(mapping, dict):
        return mapping
    subset: dict[str, Any] = {}
    for key, value in mapping.items():
        key_text = str(key)
        if Path(key_text).name in basenames:
            subset[key_text] = value
    return subset


def _chunk_sample_paths(sample_paths: list[str], batch_size: int) -> list[list[str]]:
    if not sample_paths:
        raise SystemExit("Error: --batch-size requires at least one test sample.")
    if len(sample_paths) % batch_size != 0:
        raise SystemExit(
            f"Error: --batch-size={batch_size} must divide the total test sample count {len(sample_paths)}."
        )
    return [
        sample_paths[index : index + batch_size]
        for index in range(0, len(sample_paths), batch_size)
    ]


def _materialize_batches(dataset_dir: Path, *, batch_size: int) -> tuple[int, Path]:
    question_root = dataset_dir / "question_0"
    if not question_root.is_dir():
        raise SystemExit(
            f"Error: no task materialized under {dataset_dir} (expected question_0)."
        )
    scenario_info_path = question_root / "scenario_info.json"
    if not scenario_info_path.exists():
        raise SystemExit(f"Error: scenario_info.json not found at {scenario_info_path}")
    scenario_info = _load_json(scenario_info_path)
    test_samples = scenario_info.get("test_samples")
    if not isinstance(test_samples, list):
        raise SystemExit(f"Error: scenario_info.json test_samples must be a list: {scenario_info_path}")
    test_sample_paths = [str(item).strip() for item in test_samples if str(item).strip()]
    batches = _chunk_sample_paths(test_sample_paths, batch_size)
    if len(batches) == 1:
        return 1, question_root

    source_root = question_root
    source_test_samples = source_root / "test_samples"
    if not source_test_samples.is_dir():
        raise SystemExit(f"Error: missing test_samples directory at {source_test_samples}")
    template_root = dataset_dir / "__question_template__"
    shutil.rmtree(template_root, ignore_errors=True)
    shutil.copytree(source_root, template_root)

    try:
        for index, batch_paths in enumerate(batches):
            target_root = dataset_dir / f"question_{index}"
            shutil.rmtree(target_root, ignore_errors=True)
            shutil.copytree(template_root, target_root)
            target_scenario_path = target_root / "scenario_info.json"
            target_scenario = _load_json(target_scenario_path)
            basenames = {Path(item).name for item in batch_paths}

            target_test_dir = target_root / "test_samples"
            for candidate in target_test_dir.rglob("*.parquet"):
                if candidate.name not in basenames:
                    candidate.unlink()

            target_scenario["test_samples"] = batch_paths
            source_paths = target_scenario.get("test_samples_source_paths")
            if isinstance(source_paths, list):
                filtered_source_paths = [
                    str(item)
                    for item in source_paths
                    if Path(str(item)).name in basenames
                ]
                target_scenario["test_samples_source_paths"] = filtered_source_paths
            target_scenario["num_samples"] = len(batch_paths)
            target_scenario["test_sample_labels"] = _subset_mapping_by_basename(
                target_scenario.get("test_sample_labels"),
                basenames,
            )
            _write_json(target_scenario_path, target_scenario)
            _sync_agent_payload(target_root)
    finally:
        shutil.rmtree(template_root, ignore_errors=True)

    return len(batches), source_root


def _load_eval_mode(scenario_info_path: Path) -> str:
    payload = _load_json(scenario_info_path)
    if not isinstance(payload, dict):
        return "direct"
    return normalize_tsenv_eval_mode(payload.get("eval_mode"))


def _annotate_run_scenarios(
    run_dir: Path,
    *,
    agent_id: str,
    tag: str,
    configuration_file_name: str | None = None,
) -> None:
    run_results_path = run_dir / "results.json"
    if not run_results_path.exists():
        return
    payload = _load_json(run_results_path)
    results = payload.get("results")
    if not isinstance(results, list):
        return
    for entry in results:
        if not isinstance(entry, dict):
            continue
        task_id = str(entry.get("task_id") or "").strip()
        trial_name = str(entry.get("trial_name") or "").strip()
        if not task_id or not trial_name:
            continue
        scenario_path = run_dir / task_id / trial_name / "scenario_info.json"
        if not scenario_path.exists():
            continue
        scenario = _load_json(scenario_path)
        scenario["agent_id"] = agent_id
        scenario["tag"] = tag
        if str(configuration_file_name or "").strip():
            scenario["configuration_file_name"] = str(configuration_file_name).strip()
        scenario.setdefault(
            "question_id",
            str(entry.get("question_id") or entry.get("task_hash") or "").strip(),
        )
        _write_json(scenario_path, scenario)


def _tb_run_cmd(
    *,
    run_id: str,
    model: str,
    question_id: str,
    agent: str,
    agent_model: str,
    installation_command: str | None,
    reasoning: str | None,
    timeout_sec: int,
    keep_container: bool,
    cleanup_images: bool = False,
) -> list[str]:
    cleanup_arg = "--no-cleanup" if keep_container else "--cleanup"
    cmd = [
        "uv",
        "run",
        "tb",
        "run",
        "--run-id",
        run_id,
        "--dataset-path",
        f"tasks_runtime/manual/{model}/{question_id}/",
        "--agent",
        agent,
        "--model",
        agent_model,
        "--n-concurrent",
        "1",
        "--global-agent-timeout-sec",
        str(timeout_sec),
        cleanup_arg,
    ]
    if cleanup_images and not keep_container:
        cmd.append("--cleanup-images")
    if installation_command:
        cmd.extend(["--agent-kwarg", f"installation_command={installation_command}"])
    if reasoning:
        cmd.extend(["--agent-kwarg", f"reasoning={reasoning}"])
    if keep_container:
        cmd.append("--keep-client-container")
    return cmd


def _installation_command_from_npm_package(npm_package: str | None) -> str | None:
    if npm_package is None:
        return None
    value = str(npm_package).strip()
    if not value:
        return None
    return f"npm install -g {value}"


def _iter_named_lines(cmd: list[str]) -> list[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _cleanup_run_resources(run_id: str) -> None:
    removed_containers = 0
    removed_networks = 0
    container_names = _iter_named_lines(["docker", "ps", "-a", "--format", "{{.Names}}"])
    for name in container_names:
        if run_id in name:
            proc = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                removed_containers += 1
    network_names = _iter_named_lines(["docker", "network", "ls", "--format", "{{.Name}}"])
    for name in network_names:
        if run_id in name:
            proc = subprocess.run(["docker", "network", "rm", name], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                removed_networks += 1
    print(f"cleanup removed_containers={removed_containers} removed_networks={removed_networks}")


def _run_tb_with_timeout(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout_sec: int) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=DOCKER_TIMEOUT_KILL_GRACE_SEC)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        if stdout:
            sys.stdout.write(stdout)
        if stderr:
            sys.stderr.write(stderr)
        return 124
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    return proc.returncode


def _find_first(paths: Iterable[Path]) -> Path | None:
    for path in sorted(paths):
        return path
    return None


def _resolve_trial_dir_from_paths(
    *,
    run_dir: Path,
    trial_json: Path | None,
    scores_json: Path | None,
) -> Path | None:
    if trial_json is not None:
        return trial_json.parent
    if scores_json is not None:
        return scores_json.parent
    for candidate in sorted(run_dir.rglob("scenario_info.json")):
        if candidate.parent != run_dir:
            return candidate.parent
    return None


def _tail_lines(path: Path, n: int = 40) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-n:]


def _detect_agent_timeout(trial_json: Path | None, run_log: Path | None) -> tuple[bool, str]:
    failure_mode = ""
    if trial_json and trial_json.exists():
        payload = _load_json(trial_json)
        if isinstance(payload, dict):
            failure_mode = str(payload.get("failure_mode") or "")
    if failure_mode == "agent_timeout":
        return True, failure_mode
    if run_log and run_log.exists():
        text = run_log.read_text(encoding="utf-8")
        if re.search(r"Agent timed out after [0-9.]+s for task", text):
            return True, failure_mode
    return False, failure_mode


def _print_json_summary(title: str, payload: dict[str, object]) -> None:
    print()
    print(title)
    print(json.dumps(payload, indent=2))


def _run_postprocessor(cmd: list[str], *, cwd: Path, label: str) -> subprocess.CompletedProcess:
    print()
    print(label)
    return _run_cmd(cmd, cwd=cwd, check=False)


def _warn_postprocessor_failure(name: str, run_id: str, returncode: int) -> None:
    print()
    print(
        f"Warning: {name} failed for run {run_id} with exit code {returncode}.",
        file=sys.stderr,
    )


def _persist_profile_run_metadata(
    run_dir: Path,
    *,
    agent_profile_id: str | None,
    reasoning: str | None,
) -> None:
    if not agent_profile_id:
        return
    run_metadata_path = run_dir / "run_metadata.json"
    if not run_metadata_path.exists():
        raise SystemExit(f"Error: run_metadata.json not found at {run_metadata_path}")
    payload = _load_json(run_metadata_path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Error: run_metadata.json must contain an object: {run_metadata_path}")
    payload["agent_id"] = agent_profile_id
    if reasoning is not None:
        payload["reasoning"] = reasoning
    _write_json(run_metadata_path, payload)


def _copytree_replace(source: Path, destination: Path) -> None:
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(source, destination)


def _materialize_documented_output_layout(run_dir: Path, trial_dir: Path | None) -> None:
    if trial_dir is None or not trial_dir.is_dir():
        return
    question_dir = trial_dir.parent
    if question_dir == run_dir:
        return

    for filename in ("scenario_info.json", "results.json", "agentic-final-response.json"):
        source = trial_dir / filename
        if not source.exists():
            continue
        destination = question_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    artifacts_dir = trial_dir / "artifacts"
    if artifacts_dir.is_dir():
        _copytree_replace(artifacts_dir, question_dir / "artifacts")

    agent_logs_dir = trial_dir / "agent_logs"
    if agent_logs_dir.is_dir():
        target_agent_logs_dir = question_dir / "agent_logs"
        _copytree_replace(agent_logs_dir, target_agent_logs_dir)
        rollout_dir = target_agent_logs_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        for child in sorted(target_agent_logs_dir.iterdir()):
            if child.name == "rollout":
                continue
            destination = rollout_dir / child.name
            if child.is_dir():
                _copytree_replace(child, destination)
            else:
                shutil.copy2(child, destination)


def run_single_question(
    *,
    question_id: str,
    agent_profile_id: str,
    run_id: str,
    model_name: str | None = None,
    timeout_sec: int | None = None,
    keep_container: bool = False,
    single_run: bool = False,
    batch_size: int | None = None,
    tag: str = "DEBUG",
    configuration_file_name: str | None = None,
) -> int:
    if not PYTHON_EXE.exists():
        raise SystemExit(f"Error: python not found at {PYTHON_EXE}")

    env = os.environ.copy()
    reasoning: str | None = None
    required_environmental_variable: dict[str, str] = {}
    installation_command: str | None = None
    normalized_batch_size = _parse_batch_size(batch_size)
    effective_agent_id = str(agent_profile_id).strip()
    requested_run_id = _validate_explicit_run_id(run_id)
    if effective_agent_id not in AGENTIC_PROFILES_BY_ID:
        valid_ids = ", ".join(sorted(AGENTIC_PROFILES_BY_ID))
        raise SystemExit(
            f"Error: unknown AGENT_ID {effective_agent_id!r}. Valid values: {valid_ids}."
        )
    effective_timeout_sec = (
        int(timeout_sec)
        if timeout_sec is not None
        else QUESTION_TIMEOUT_SEC_DEFAULT
    )
    profile = agentic_profile_by_id(effective_agent_id)
    platform = dict(profile["agentic_platform"])
    agent = str(platform["name"])
    installation_command = _installation_command_from_npm_package(
        platform.get("npm_package")
    )
    agent_model = str(profile["model_name"])
    reasoning = profile["reasoning"]
    required_environmental_variable = dict(
        profile.get("required_environmental_variable") or {}
    )
    if agent == "claude-code":
        _populate_claude_auth_environment(env, required_environmental_variable)
    _populate_required_environment_variables(env, required_environmental_variable)
    if agent == "gemini-cli":
        _populate_gemini_auth_environment(env)
    _validate_agent(agent)
    _ensure_auth(agent, env)
    _configure_agent_auth_env(agent, env)

    model, source_dir = _resolve_question_source(question_id, expected_model=model_name)
    payload, question = _load_question_payload(source_dir, question_id)
    dataset_dir = TERMINAL_BENCH_DIR / "tasks_runtime" / "manual" / model / question_id
    effective_run_id = requested_run_id or _compute_run_id(effective_agent_id, model, question_id)
    initial_question_root = dataset_dir / "question_0"
    reused_materialized_task = _materialized_question_matches_source(initial_question_root, question)
    if reused_materialized_task:
        print("==> Reusing materialized single-question dataset")
        question_count, question_root = 1, initial_question_root
        export_dir: Path | None = None
    else:
        print("==> Materializing single-question dataset")
        _materialize_dataset(source_dir, dataset_dir, question_id)
        if _uses_ground_truth_context(question):
            if not initial_question_root.is_dir():
                raise SystemExit(
                    f"Error: no task materialized under {dataset_dir} (expected question_0)."
                )
            shutil.copy2(
                _resolve_noise_adder_source(source_dir),
                initial_question_root / "noise_adder.py",
            )
            _sync_agent_payload(initial_question_root)

        question_count, question_root = (
            _materialize_batches(dataset_dir, batch_size=normalized_batch_size)
            if normalized_batch_size is not None
            else (1, initial_question_root)
        )
        export_dir = None
    if not question_root.is_dir():
        raise SystemExit(f"Error: no task materialized under {dataset_dir} (expected question_0).")
    scenario_info_path = question_root / "scenario_info.json"
    if not scenario_info_path.exists():
        raise SystemExit(f"Error: scenario_info.json not found at {scenario_info_path}")
    scenario_info = _load_json(scenario_info_path)
    if not isinstance(scenario_info, dict):
        raise SystemExit(f"Error: scenario_info.json must contain an object: {scenario_info_path}")
    eval_mode = _load_eval_mode(scenario_info_path)
    _write_agent_prompt_log(
        payload=payload,
        question=question,
        question_slug=question_id,
    )
    _write_train_labels_file(
        question_root=question_root,
        question=question,
        payload=payload,
        scenario_info=scenario_info,
    )
    export_dir = _export_materialized_samples(
        question_root=question_root,
        question=question,
        payload=payload,
        question_slug=question_id,
        agentic_run_id=effective_run_id,
    )
    _write_noise_analysis_artifact(
        question_root=question_root,
        source_dir=source_dir,
        model_name=model,
        question_slug=question_id,
        question=question,
        payload=payload,
        scenario_info=scenario_info,
        repo_root=ROOT_DIR,
    )
    env["T_BENCH_TASK_PAYLOAD_PATH"] = str(export_dir)

    print("==> Running agent")
    print(
        f"agent={agent} model={agent_model} dataset_model={model} "
        f"keep_container={1 if keep_container else 0} timeout_sec={effective_timeout_sec} "
        f"questions={question_count}"
    )
    print(f"cleanup_images={1 if single_run and not keep_container else 0}")
    print(f"agent_profile_id={effective_agent_id} reasoning={reasoning or '<none>'}")
    print(
        f"hard_timeout_sec={effective_timeout_sec} "
        f"kill_grace_sec={DOCKER_TIMEOUT_KILL_GRACE_SEC}"
    )
    print(f"tag={tag}")
    print(f"materialized_samples_dir={export_dir or '<task-local agent_payload>'}")

    tb_cmd = _tb_run_cmd(
        run_id=effective_run_id,
        model=model,
        question_id=question_id,
        agent=agent,
        agent_model=agent_model,
        installation_command=installation_command,
        reasoning=reasoning,
        timeout_sec=effective_timeout_sec,
        keep_container=keep_container,
        cleanup_images=single_run,
    )
    tb_exit = _run_tb_with_timeout(
        tb_cmd,
        cwd=TERMINAL_BENCH_DIR,
        env=env,
        timeout_sec=effective_timeout_sec,
    )
    run_dir = TERMINAL_BENCH_DIR / "runs" / effective_run_id
    run_log = run_dir / "run.log"

    if tb_exit == 124:
        print()
        print("== Timeout summary ==")
        print("Run timed out before task resolution.")
        print(f"Run ID: {effective_run_id}")
        print(f"timeout_sec: {effective_timeout_sec}")
        print(f"kill_grace_sec: {DOCKER_TIMEOUT_KILL_GRACE_SEC}")
        print(f"tb_exit: {tb_exit}")
        _cleanup_run_resources(effective_run_id)
        return 124
    if tb_exit != 0:
        raise SystemExit(f"Error: tb run exited with code {tb_exit}.")

    run_results_json = run_dir / "results.json"
    _persist_profile_run_metadata(
        run_dir,
        agent_profile_id=effective_agent_id,
        reasoning=reasoning,
    )
    _annotate_run_scenarios(
        run_dir,
        agent_id=effective_agent_id,
        tag=str(tag).strip() or "DEBUG",
        configuration_file_name=configuration_file_name,
    )
    trial_json = _find_first(
        path
        for path in run_dir.rglob("results.json")
        if path != run_results_json
    )
    scores_json = _find_first(run_dir.rglob("scores.json"))

    print()
    print(f"Run ID: {effective_run_id}")
    print(f"eval_mode:  {eval_mode}")
    print(f"run results: {run_results_json if run_results_json.exists() else '<not found>'}")
    print(f"trial results: {trial_json if trial_json else '<not found>'}")
    print(f"scores.json:  {scores_json if scores_json else '<not found>'}")

    if not run_results_json.exists():
        raise SystemExit(f"Error: run results file not found at {run_results_json}")

    run_results = _load_json(run_results_json)
    n_resolved = int(run_results.get("n_resolved", 0)) if isinstance(run_results, dict) else 0
    n_unresolved = int(run_results.get("n_unresolved", 0)) if isinstance(run_results, dict) else 0
    _print_json_summary(
        "== Run summary ==",
        {
            "n_resolved": n_resolved,
            "n_unresolved": n_unresolved,
            "accuracy": run_results.get("accuracy") if isinstance(run_results, dict) else None,
        },
    )
    if trial_json and trial_json.exists():
        trial_payload = _load_json(trial_json)
        if isinstance(trial_payload, dict):
            _print_json_summary(
                "== Trial status ==",
                {
                    "is_resolved": trial_payload.get("is_resolved"),
                    "failure_mode": trial_payload.get("failure_mode"),
                    "parser_results": trial_payload.get("parser_results"),
                },
            )
    if scores_json and scores_json.exists():
        scores_payload = _load_json(scores_json)
        if isinstance(scores_payload, dict):
            _print_json_summary(
                "== Scored result ==",
                {
                    "metrics": scores_payload.get("metrics"),
                    "agent_answer": scores_payload.get("agent_answer"),
                    "ground_truth": scores_payload.get("ground_truth"),
                },
            )

    has_timeout, failure_mode = _detect_agent_timeout(trial_json, run_log)
    if has_timeout:
        print()
        print("== Timeout summary ==")
        print("Run timed out before task resolution.")
        print(f"Run ID: {effective_run_id}")
        print(f"timeout_sec: {effective_timeout_sec}")
        print(f"failure_mode: {failure_mode or '<unknown>'}")
        print(f"run.log: {run_log}")
        print(f"trial results: {trial_json if trial_json else '<not found>'}")
        _cleanup_run_resources(effective_run_id)
        if run_log.exists():
            print()
            print("Last lines from run.log:")
            for line in _tail_lines(run_log):
                print(line)
        return 124

    if n_resolved < 1:
        print()
        print(
            f"Run did not resolve successfully (resolved={n_resolved} unresolved={n_unresolved}).",
            file=sys.stderr,
        )
        if run_log.exists():
            print("Last lines from run.log:", file=sys.stderr)
            for line in _tail_lines(run_log):
                print(line, file=sys.stderr)
        return 2

    print()
    accuracy_summary_path = run_dir / "accuracy_summary.json"
    accuracy_cmd = [
        str(PYTHON_EXE),
        str(ROOT_DIR / "workflows" / "rollout" / "evaluate_artifact.py"),
        effective_run_id,
    ]
    accuracy_proc = _run_postprocessor(
        accuracy_cmd,
        cwd=ROOT_DIR,
        label="== Accuracy evaluation ==",
    )
    if accuracy_proc.returncode != 0:
        _warn_postprocessor_failure("accuracy evaluation", effective_run_id, accuracy_proc.returncode)
    elif accuracy_summary_path.exists():
        accuracy_payload = _load_json(accuracy_summary_path)
        if isinstance(accuracy_payload, dict):
            _print_json_summary(
                "== Accuracy summary ==",
                {
                    "evaluated_trials": accuracy_payload.get("evaluated_trials"),
                    "errored_trials": accuracy_payload.get("errored_trials"),
                    "total_evaluable_answers": accuracy_payload.get("total_evaluable_answers"),
                    "total_correct_answers": accuracy_payload.get("total_correct_answers"),
                    "batch_accuracy": accuracy_payload.get("batch_accuracy"),
                },
            )

    trajectory_proc = _run_postprocessor(
        [
            str(PYTHON_EXE),
            str(ROOT_DIR / "workflows" / "rollout" / "export_atif_trajectory.py"),
            effective_run_id,
        ],
        cwd=ROOT_DIR,
        label="== Trajectory export ==",
    )
    if trajectory_proc.returncode != 0:
        _warn_postprocessor_failure("trajectory export", effective_run_id, trajectory_proc.returncode)

    trajectory_evaluation_proc = _run_postprocessor(
        [
            str(PYTHON_EXE),
            str(ROOT_DIR / "workflows" / "trajectories" / "trajectory_evaluation_programmatic.py"),
            effective_run_id,
        ],
        cwd=ROOT_DIR,
        label="== Trajectory evaluation ==",
    )
    if trajectory_evaluation_proc.returncode != 0:
        _warn_postprocessor_failure(
            "trajectory evaluation",
            effective_run_id,
            trajectory_evaluation_proc.returncode,
        )

    scores_json = _find_first(run_dir.rglob("scores.json"))
    trial_dir = _resolve_trial_dir_from_paths(
        run_dir=run_dir,
        trial_json=trial_json,
        scores_json=scores_json,
    )
    _materialize_documented_output_layout(run_dir, trial_dir)
    atif_trajectory_path = canonical_trajectory_path(trial_dir) if trial_dir is not None else None
    atif_trajectory_light_path = (
        resolve_light_trajectory_path(trial_dir) if trial_dir is not None else None
    )
    if scores_json and scores_json.exists():
        scores_payload = _load_json(scores_json)
        if isinstance(scores_payload, dict):
            _print_json_summary(
                "== Scored result ==",
                {
                    "metrics": scores_payload.get("metrics"),
                    "agent_answer": scores_payload.get("agent_answer"),
                    "ground_truth": scores_payload.get("ground_truth"),
                },
            )

    print(f"accuracy_summary.json:   {accuracy_summary_path if accuracy_summary_path.exists() else '<not found>'}")
    print(
        "atif_trajectory.json:    "
        f"{atif_trajectory_path if atif_trajectory_path and atif_trajectory_path.exists() else '<not found>'}"
    )
    print(
        "atif_trajectory_light.json: "
        f"{atif_trajectory_light_path if atif_trajectory_light_path and atif_trajectory_light_path.exists() else '<not found>'}"
    )
    print(f"scores.json:             {scores_json if scores_json and scores_json.exists() else '<not found>'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    question_id, agent_profile_id, run_id, model_name, timeout_sec = _resolve_cli_contract(args)
    kwargs: dict[str, Any] = {
        "question_id": question_id,
        "agent_profile_id": agent_profile_id,
        "run_id": run_id,
        "model_name": model_name,
        "keep_container": args.keep_container,
        "single_run": args.single_run,
        "tag": str(os.environ.get("TSENV_AGENTIC_ROLLOUT_TAG") or "DEBUG").strip() or "DEBUG",
        "configuration_file_name": str(
            os.environ.get("TSENV_AGENTIC_CONFIGURATION_FILE_NAME") or ""
        ).strip()
        or None,
    }
    if timeout_sec is not None:
        kwargs["timeout_sec"] = timeout_sec
    return internal_runner.run_single_question(**kwargs)


internal_runner = sys.modules[__name__]


if __name__ == "__main__":
    raise SystemExit(main())
