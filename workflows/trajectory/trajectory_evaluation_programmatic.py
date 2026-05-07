#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import click
try:
    from tqdm import tqdm as _tqdm
except ModuleNotFoundError:
    _tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "terminal-bench" / "runs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.agentic_rollout_paths import CANONICAL_ATIF_PROCESSED_DIRNAME, TRAJECTORY_FILENAME


_SHELL_NESTED_NODE_TYPES = {
    "command_substitution",
    "process_substitution",
    "subshell",
    "if_statement",
    "for_statement",
    "while_statement",
    "case_statement",
    "function_definition",
}
_CAT_PATTERN_KEYS = (
    "heredoc_write",
    "file_concat_to_stdout",
    "file_concat_to_file",
    "pipe_into_cat",
    "other",
)
_INTERACTION_SOURCE_KEYS = ("user", "system", "agent")
_PYTHON_INVOCATION_MODE_KEYS = ("dash_c", "heredoc", "script_path", "other")
_SHELL_COMMAND_ALIASES = {"python3": "python"}
_PYTHON_FAILURE_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\):"),
    re.compile(r"\bCommand failed\b", re.IGNORECASE),
    re.compile(r"\breturned non-zero exit status\b", re.IGNORECASE),
    re.compile(r"\bexit code\b", re.IGNORECASE),
)
_PYTHON_EXCEPTION_LINE_PATTERN = re.compile(
    r"^\s*(?:ModuleNotFoundError|SyntaxError|NameError|ImportError|FileNotFoundError|RuntimeError|ValueError|TypeError|AssertionError|AttributeError|KeyError|IndexError|ZeroDivisionError)\b.*$",
    re.MULTILINE,
)
_CODEX_DESCRIPTION_RAW_DIRNAME = "codex_description_raw"
PROFILE_COMPATIBILITY_ALIASES = {
    "minimax-m2.7_low": "minimax-m2.7",
}
_PLOT_FILE_SUFFIXES = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".webp"}


class _NullProgressBar:
    def update(self, value: int) -> None:
        return None

    def close(self) -> None:
        return None


def tqdm(*args: Any, **kwargs: Any) -> Any:
    if _tqdm is not None:
        return _tqdm(*args, **kwargs)
    if args:
        return args[0]
    return _NullProgressBar()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"{path} must contain a JSON object")
    return payload


def _resolve_run_dir(agentic_run_id: str | None, run_path: Path | None) -> Path:
    has_run_id = bool(str(agentic_run_id or "").strip())
    has_run_path = run_path is not None
    if has_run_id == has_run_path:
        raise click.ClickException("Pass exactly one of <agentic_run_id> or --path.")
    if has_run_id:
        run_dir = (RUNS_ROOT / str(agentic_run_id).strip()).resolve()
    else:
        run_dir = Path(run_path).expanduser().resolve()
    if not run_dir.exists():
        raise click.ClickException(f"Run directory not found: {run_dir}")
    if not run_dir.is_dir():
        raise click.ClickException(f"Run path is not a directory: {run_dir}")
    return run_dir


def _candidate_trajectory_paths(run_dir: Path) -> list[Path]:
    return sorted(
        run_dir.glob(
            f"*/*/agent_logs/{CANONICAL_ATIF_PROCESSED_DIRNAME}/{TRAJECTORY_FILENAME}"
        )
    )


def _resolve_trajectory_path(agentic_run_id: str | None, run_dir: Path) -> Path:
    trajectory_paths = _candidate_trajectory_paths(run_dir)
    if not trajectory_paths:
        raise click.ClickException(f"Trajectory file not found under run directory: {run_dir}")
    if len(trajectory_paths) != 1:
        rendered = ", ".join(str(path.relative_to(run_dir)) for path in trajectory_paths[:5])
        if len(trajectory_paths) > 5:
            rendered += ", ..."
        raise click.ClickException(
            f"Expected exactly one trajectory file under {run_dir}, "
            f"found {len(trajectory_paths)}: {rendered}"
        )
    trajectory_path = trajectory_paths[0].resolve()
    if not trajectory_path.exists():
        raise click.ClickException(f"Trajectory file not found: {trajectory_path}")
    return trajectory_path


def _resolve_run_metadata_path(run_dir: Path) -> Path:
    return (run_dir / "run_metadata.json").resolve()


def _trial_dir_for_trajectory_path(trajectory_path: Path) -> Path:
    return trajectory_path.parents[2]


def _analyze_artifacts(trajectory_path: Path) -> dict[str, list[str]]:
    artifacts_dir = _trial_dir_for_trajectory_path(trajectory_path) / "artifacts"
    buckets: dict[str, list[str]] = {
        "python_scripts": [],
        "json_files": [],
        "plots": [],
    }
    if not artifacts_dir.is_dir():
        return buckets
    for path in sorted(candidate for candidate in artifacts_dir.rglob("*") if candidate.is_file()):
        try:
            rendered_path = str(path.relative_to(artifacts_dir))
        except ValueError:
            rendered_path = str(path)
        suffix = path.suffix.lower()
        if suffix == ".py":
            buckets["python_scripts"].append(rendered_path)
        if suffix == ".json":
            buckets["json_files"].append(rendered_path)
        if suffix in _PLOT_FILE_SUFFIXES:
            buckets["plots"].append(rendered_path)
    return buckets


def _resolve_output_run_id(
    agentic_run_id: str | None,
    run_metadata: Mapping[str, Any],
    run_dir: Path,
) -> str:
    cli_run_id = str(agentic_run_id or "").strip()
    if cli_run_id:
        return cli_run_id
    return str(run_metadata.get("run_id") or run_dir.name).strip()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _load_tree_sitter_bash_parser() -> Any:
    try:
        from tree_sitter import Language, Parser  # type: ignore[import-not-found]
        import tree_sitter_bash  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    return Parser(Language(tree_sitter_bash.language()))


def _agentic_profile_by_id(agent_id: str) -> Mapping[str, Any]:
    profile_id = PROFILE_COMPATIBILITY_ALIASES.get(agent_id, agent_id)
    try:
        from shared.benchmark_utils import agentic_profile_by_id
    except Exception as exc:
        raise click.ClickException(f"Unable to load agent registry: {exc}") from exc
    try:
        return agentic_profile_by_id(profile_id)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def _node_text(source: bytes, node: Any) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8").strip()


def _normalize_shell_token(text: object) -> str:
    value = str(text or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _canonical_shell_command_name(name: object) -> str:
    value = str(name or "").strip()
    if not value:
        return ""
    return _SHELL_COMMAND_ALIASES.get(value, value)


def _iter_tree_sitter_nodes(node: Any) -> Any:
    yield node
    for child in node.children:
        yield from _iter_tree_sitter_nodes(child)


def _iter_command_nodes(node: Any) -> list[Any]:
    return [candidate for candidate in _iter_tree_sitter_nodes(node) if candidate.type == "command"]


def _command_name_from_tree_sitter_node(node: Any, source: bytes) -> str | None:
    for child in node.children:
        if child.type == "command_name":
            text = _normalize_shell_token(_node_text(source, child))
            if text:
                return text
    for child in node.named_children:
        text = _normalize_shell_token(_node_text(source, child))
        if text:
            return text
    return None


def _command_argument_texts(node: Any, source: bytes) -> list[str]:
    arguments: list[str] = []
    for child in node.named_children:
        if child.type == "command_name":
            continue
        text = _normalize_shell_token(_node_text(source, child))
        if text:
            arguments.append(text)
    return arguments


def _parse_shell_program(
    command: str,
    *,
    tool_name: str,
    trajectory_path: Path,
) -> tuple[bytes, Any]:
    parser = _load_tree_sitter_bash_parser()
    source = str(command).encode("utf-8")
    if parser is None:
        raise click.ClickException(
            "Shell parsing requires tree_sitter and tree_sitter_bash to be installed "
            f"for tool {tool_name!r} in {trajectory_path}."
        )
    try:
        tree = parser.parse(source)
    except Exception as exc:
        preview = str(command).strip().replace("\n", "\\n")
        raise click.ClickException(
            f"Failed to parse {tool_name} in {trajectory_path}: {preview}"
        ) from exc
    root = tree.root_node
    if root.has_error:
        preview = str(command).strip().replace("\n", "\\n")
        raise click.ClickException(
            f"Failed to parse {tool_name} in {trajectory_path}: {preview}"
        )
    return source, root


def _split_top_level_shell(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    single_quote = False
    double_quote = False
    paren_depth = 0
    i = 0
    while i < len(text):
        char = text[i]
        next_two = text[i : i + 2]
        if not single_quote and char == '"' and (i == 0 or text[i - 1] != "\\"):
            double_quote = not double_quote
            current.append(char)
            i += 1
            continue
        if not double_quote and char == "'" and (i == 0 or text[i - 1] != "\\"):
            single_quote = not single_quote
            current.append(char)
            i += 1
            continue
        if not single_quote and not double_quote and next_two == "$(":
            paren_depth += 1
            current.append(next_two)
            i += 2
            continue
        if not single_quote and not double_quote and char == ")" and paren_depth > 0:
            paren_depth -= 1
            current.append(char)
            i += 1
            continue
        if not single_quote and not double_quote and paren_depth == 0:
            if next_two in {"&&", "||"}:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 2
                continue
            if char in {"|", ";"}:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                continue
        current.append(char)
        i += 1
    if single_quote or double_quote or paren_depth != 0:
        raise ValueError("unbalanced shell syntax")
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _extract_command_substitutions(text: str) -> list[str]:
    substitutions: list[str] = []
    single_quote = False
    double_quote = False
    i = 0
    while i < len(text):
        char = text[i]
        next_two = text[i : i + 2]
        if not single_quote and char == '"' and (i == 0 or text[i - 1] != "\\"):
            double_quote = not double_quote
            i += 1
            continue
        if not double_quote and char == "'" and (i == 0 or text[i - 1] != "\\"):
            single_quote = not single_quote
            i += 1
            continue
        if not single_quote and not double_quote and next_two == "$(":
            start = i + 2
            depth = 1
            i += 2
            inner_single = False
            inner_double = False
            while i < len(text):
                inner_char = text[i]
                inner_next_two = text[i : i + 2]
                if not inner_single and inner_char == '"' and text[i - 1] != "\\":
                    inner_double = not inner_double
                    i += 1
                    continue
                if not inner_double and inner_char == "'" and text[i - 1] != "\\":
                    inner_single = not inner_single
                    i += 1
                    continue
                if not inner_single and not inner_double and inner_next_two == "$(":
                    depth += 1
                    i += 2
                    continue
                if not inner_single and not inner_double and inner_char == ")":
                    depth -= 1
                    if depth == 0:
                        substitutions.append(text[start:i])
                        break
                i += 1
            else:
                raise ValueError("unbalanced command substitution")
        i += 1
    return substitutions


def _extract_output_redirection(segment: str) -> tuple[str, str | None]:
    match = re.search(r">\s*([^\s]+)\s*$", segment)
    if match is None:
        return segment.strip(), None
    output_target = _normalize_shell_token(match.group(1))
    return segment[: match.start()].strip(), output_target or None


def _build_fallback_command_info(
    *,
    segment: str,
    start_byte: int,
    end_byte: int | None = None,
    heredoc_body: str | None = None,
    heredoc_delimiter: str | None = None,
    force_pattern: str | None = None,
) -> dict[str, Any] | None:
    command_without_redirect, output_target = _extract_output_redirection(segment)
    try:
        tokens = shlex.split(command_without_redirect)
    except ValueError as exc:
        raise ValueError(f"invalid shell tokens: {exc}") from exc
    if not tokens:
        return None
    name = _canonical_shell_command_name(tokens[0])
    args = tokens[1:]
    pattern = force_pattern
    if name == "cat":
        if pattern is None and heredoc_body is not None and output_target:
            pattern = "heredoc_write"
        elif pattern is None and "|" in segment and output_target:
            pattern = "pipe_into_cat"
        elif pattern is None and output_target:
            pattern = "file_concat_to_file"
        elif pattern is None and args:
            pattern = "file_concat_to_stdout"
        elif pattern is None:
            pattern = "other"
    return {
        "node": {
            "fallback": True,
            "name": name,
            "args": args,
            "start_byte": start_byte,
            "end_byte": start_byte + len(segment) if end_byte is None else int(end_byte),
            "raw_text": segment,
            "output_target": output_target,
            "heredoc_delimiter": heredoc_delimiter,
            "heredoc_body": heredoc_body,
            "pattern": pattern,
        },
        "name": name,
        "args": args,
        "start_byte": start_byte,
        "source": None,
    }


def _parse_shell_program_fallback(
    command: str,
    *,
    tool_name: str,
    trajectory_path: Path,
) -> dict[str, Any]:
    command_infos: list[dict[str, Any]] = []
    lines = str(command).splitlines()
    byte_offset = 0
    i = 0
    heredoc_pattern = re.compile(r"^\s*(?P<prefix>.+?)<<\s*['\"]?(?P<delimiter>[A-Za-z0-9_]+)['\"]?")
    try:
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                byte_offset += len(line) + 1
                i += 1
                continue
            heredoc_match = heredoc_pattern.match(line)
            if heredoc_match:
                delimiter = heredoc_match.group("delimiter")
                body_lines: list[str] = []
                end_index = i + 1
                while end_index < len(lines) and lines[end_index].strip() != delimiter:
                    body_lines.append(lines[end_index])
                    end_index += 1
                if end_index >= len(lines):
                    raise ValueError("unterminated heredoc")
                heredoc_body = "\n".join(body_lines)
                info = _build_fallback_command_info(
                    segment=line,
                    start_byte=byte_offset,
                    end_byte=byte_offset + sum(len(lines[idx]) + 1 for idx in range(i, end_index + 1)),
                    heredoc_body=heredoc_body,
                    heredoc_delimiter=delimiter,
                )
                if info is not None:
                    command_infos.append(info)
                consumed = sum(len(lines[idx]) + 1 for idx in range(i, end_index + 1))
                byte_offset += consumed
                i = end_index + 1
                continue
            for segment in _split_top_level_shell(line):
                force_pattern = None
                if "|" in line and segment.lstrip().startswith("cat"):
                    force_pattern = "pipe_into_cat"
                info = _build_fallback_command_info(
                    segment=segment,
                    start_byte=byte_offset + line.find(segment),
                    force_pattern=force_pattern,
                )
                if info is not None:
                    command_infos.append(info)
                for substitution in _extract_command_substitutions(segment):
                    nested_infos = _parse_shell_program_fallback(
                        substitution,
                        tool_name=tool_name,
                        trajectory_path=trajectory_path,
                    )["command_infos"]
                    command_infos.extend(nested_infos)
            byte_offset += len(line) + 1
            i += 1
    except ValueError as exc:
        preview = str(command).strip().replace("\n", "\\n")
        raise click.ClickException(
            f"Failed to parse {tool_name} in {trajectory_path}: {preview}"
        ) from exc
    return {"fallback": True, "command_infos": sorted(command_infos, key=lambda item: int(item["start_byte"]))}


def _update_shell_command_counts(
    node: Any,
    *,
    source: bytes,
    counts: Counter[str],
) -> None:
    if isinstance(node, dict) and node.get("fallback"):
        for info in node.get("command_infos", []):
            command_name = _canonical_shell_command_name(info.get("name"))
            if command_name:
                counts[command_name] += 1
        return
    if node.type == "command":
        command_name = _canonical_shell_command_name(
            _command_name_from_tree_sitter_node(node, source)
        )
        if command_name:
            counts[command_name] += 1
    for child in node.children:
        _update_shell_command_counts(
            child,
            source=source,
            counts=counts,
        )


def _render_shell_command_objects(
    counts: Counter[str],
    call_unique_counts: Counter[str],
) -> dict[str, dict[str, int]]:
    names = sorted(set(counts) | set(call_unique_counts))
    return {
        str(name): {
            "count": int(counts.get(name, 0)),
            "call_unique": int(call_unique_counts.get(name, 0)),
        }
        for name in names
    }


def _render_tool_call_counts(
    *,
    available_tools: Mapping[str, Mapping[str, Any]],
    simple_counts: Counter[str],
    file_path_tool_counts: Mapping[str, Counter[str]],
    parsed_tool_counts: Mapping[str, Counter[str]],
    parsed_tool_call_unique_counts: Mapping[str, Counter[str]],
) -> dict[str, Any]:
    rendered_counts: dict[str, Any] = {}
    all_tool_names = sorted(
        set(available_tools)
        | set(simple_counts)
        | set(file_path_tool_counts)
        | set(parsed_tool_counts)
    )
    for function_name in all_tool_names:
        tool_config = available_tools.get(function_name) or {}
        parse_method = tool_config.get("parse_method", False)
        if parse_method == "file_path":
            rendered_parse = dict(
                sorted(
                    (str(path), int(count))
                    for path, count in file_path_tool_counts.get(function_name, Counter()).items()
                )
            )
            rendered_counts[function_name] = {
                "count": sum(rendered_parse.values()),
                "read": rendered_parse,
            }
            continue
        if parse_method == "tree_sitter_bash":
            rendered_parse = _render_shell_command_objects(
                parsed_tool_counts.get(function_name, Counter()),
                parsed_tool_call_unique_counts.get(function_name, Counter()),
            )
            rendered_counts[function_name] = {
                "count": sum(item["count"] for item in rendered_parse.values()),
                "commands": rendered_parse,
            }
            continue
        rendered_counts[function_name] = {
            "count": int(simple_counts.get(function_name, 0)),
        }
    return rendered_counts


def _empty_cat_analysis() -> dict[str, Any]:
    return {
        "total_invocations": 0,
        "pattern_counts": Counter({key: 0 for key in _CAT_PATTERN_KEYS}),
        "heredoc_write_create_files": Counter(),
        "followup_command_counts": Counter(),
        "execution_after_creation": Counter(
            {
                "executed_in_same_shell_snippet": 0,
                "not_executed_in_same_shell_snippet": 0,
            }
        ),
        "heredoc_delimiters": Counter(),
    }


def _render_cat_analysis(cat_analysis: Mapping[str, Any]) -> dict[str, Any]:
    python_files = {
        str(key): int(value)
        for key, value in cat_analysis["heredoc_write_create_files"].items()
        if Path(str(key)).suffix == ".py"
    }
    other_files = {
        str(key): int(value)
        for key, value in cat_analysis["heredoc_write_create_files"].items()
        if Path(str(key)).suffix != ".py"
    }
    return {
        "total_invocations": int(cat_analysis["total_invocations"]),
        "pattern": {
            "heredoc_write": {
                "create_files": {
                    "python_files": dict(sorted(python_files.items())),
                    "other": dict(sorted(other_files.items())),
                },
            },
            "file_concat_to_stdout": int(cat_analysis["pattern_counts"].get("file_concat_to_stdout", 0)),
            "file_concat_to_file": int(cat_analysis["pattern_counts"].get("file_concat_to_file", 0)),
            "pipe_into_cat": int(cat_analysis["pattern_counts"].get("pipe_into_cat", 0)),
            "other": int(cat_analysis["pattern_counts"].get("other", 0)),
        },
        "followup_command_counts": dict(
            sorted((str(key), int(value)) for key, value in cat_analysis["followup_command_counts"].items())
        ),
        "execution_after_creation": {
            "executed_in_same_shell_snippet": int(
                cat_analysis["execution_after_creation"].get("executed_in_same_shell_snippet", 0)
            ),
            "not_executed_in_same_shell_snippet": int(
                cat_analysis["execution_after_creation"].get("not_executed_in_same_shell_snippet", 0)
            ),
        },
        "heredoc_delimiters": dict(
            sorted((str(key), int(value)) for key, value in cat_analysis["heredoc_delimiters"].items())
        ),
    }


def _empty_python_analysis() -> dict[str, Any]:
    return {
        "total_invocations": 0,
        "failed_invocations": 0,
        "total_stdout_lines": 0,
        "total_stderr_lines": 0,
        "total_output_bytes": 0,
        "invocation_modes": Counter({key: 0 for key in _PYTHON_INVOCATION_MODE_KEYS}),
        "lines_of_code": {
            "inline_total": 0,
            "inline_by_invocation": [],
            "script_total": 0,
            "script_by_path": {},
        },
        "library_calls": {},
        "detailed": {},
        "detailed_inline_counters": Counter(),
        "script_targets": Counter(),
        "artifact_script_targets": Counter(),
        "unresolved_script_targets": Counter(),
    }


def _empty_python_detailed_entry() -> dict[str, Any]:
    return {
        "executions": 0,
        "lines_of_code": None,
        "library_calls": {},
        "total_stdout_lines": 0,
        "total_stderr_lines": 0,
        "total_output_bytes": 0,
        "_code_text": None,
        "_llm_tasks": [],
    }


def _render_library_calls(
    library_calls: Mapping[str, Mapping[str, int] | Counter[str]],
    *,
    verbose: bool,
) -> dict[str, int] | dict[str, dict[str, int]]:
    rendered = {
        str(library): dict(
            sorted((str(function_name), int(count)) for function_name, count in functions.items())
        )
        for library, functions in sorted((str(key), value) for key, value in library_calls.items())
    }
    if verbose:
        return rendered
    return {
        str(library): sum(int(count) for count in functions.values())
        for library, functions in rendered.items()
    }


def _render_python_detailed_entry(
    script_entry: Mapping[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    rendered = {
        "executions": int(script_entry["executions"]),
        "lines_of_code": (
            None if script_entry["lines_of_code"] is None else int(script_entry["lines_of_code"])
        ),
        "library_calls": _render_library_calls(script_entry["library_calls"], verbose=verbose),
        "total_stdout_lines": int(script_entry["total_stdout_lines"]),
        "total_stderr_lines": int(script_entry["total_stderr_lines"]),
        "total_output_bytes": int(script_entry["total_output_bytes"]),
    }
    if "description" in script_entry:
        rendered["description"] = script_entry["description"]
    if "number_of_semantically_relevant_numbers" in script_entry:
        value = script_entry["number_of_semantically_relevant_numbers"]
        rendered["number_of_semantically_relevant_numbers"] = (
            None if value is None else int(value)
        )
    if "intent_class" in script_entry:
        rendered["intent_class"] = script_entry["intent_class"]
    if "specific_subtype" in script_entry:
        rendered["specific_subtype"] = script_entry["specific_subtype"]
    return rendered


def _render_python_analysis(
    python_analysis: Mapping[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    lines_of_code = python_analysis["lines_of_code"]
    return {
        "total_invocations": int(python_analysis["total_invocations"]),
        "failed_invocations": int(python_analysis["failed_invocations"]),
        "total_stdout_lines": int(python_analysis["total_stdout_lines"]),
        "total_stderr_lines": int(python_analysis["total_stderr_lines"]),
        "total_output_bytes": int(python_analysis["total_output_bytes"]),
        "invocation_modes": {
            key: int(python_analysis["invocation_modes"].get(key, 0))
            for key in _PYTHON_INVOCATION_MODE_KEYS
        },
        "lines_of_code": {
            "script_total": int(lines_of_code["script_total"]),
        },
        "library_calls": _render_library_calls(python_analysis["library_calls"], verbose=verbose),
        "detailed": {
            str(target): _render_python_detailed_entry(entry, verbose=verbose)
            for target, entry in python_analysis["detailed"].items()
        },
    }


def _empty_subcommand_analysis(enabled_subcommands: set[str]) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    if "cat" in enabled_subcommands:
        analysis["cat"] = _empty_cat_analysis()
    if "python" in enabled_subcommands:
        analysis["python"] = _empty_python_analysis()
    return analysis


def _render_subcommand_analysis(
    subcommand_analysis: Mapping[str, Any],
    *,
    verbose: bool,
) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for subcommand in sorted(subcommand_analysis):
        if subcommand == "cat":
            if int(subcommand_analysis[subcommand].get("total_invocations", 0)) > 0:
                rendered[subcommand] = _render_cat_analysis(subcommand_analysis[subcommand])
            continue
        if subcommand == "python":
            if int(subcommand_analysis[subcommand].get("total_invocations", 0)) > 0:
                rendered[subcommand] = _render_python_analysis(
                    subcommand_analysis[subcommand],
                    verbose=verbose,
                )
    return rendered


def _nearest_ancestor_of_type(node: Any, target_type: str) -> Any | None:
    current = getattr(node, "parent", None)
    while current is not None:
        if current.type == target_type:
            return current
        current = getattr(current, "parent", None)
    return None


def _find_output_target_for_command(node: Any, source: bytes) -> str | None:
    if isinstance(node, dict) and node.get("fallback"):
        value = node.get("output_target")
        return str(value).strip() or None if value is not None else None
    redirected_statement = _nearest_ancestor_of_type(node, "redirected_statement")
    if redirected_statement is None:
        return None
    for candidate in _iter_tree_sitter_nodes(redirected_statement):
        if candidate.type != "file_redirect":
            continue
        for child in candidate.named_children:
            if child.type in {"word", "raw_string", "string"}:
                text = _normalize_shell_token(_node_text(source, child))
                if text:
                    return text
    return None


def _find_heredoc_delimiter_for_command(node: Any, source: bytes) -> str | None:
    if isinstance(node, dict) and node.get("fallback"):
        value = node.get("heredoc_delimiter")
        return str(value).strip() or None if value is not None else None
    redirected_statement = _nearest_ancestor_of_type(node, "redirected_statement")
    if redirected_statement is None:
        return None
    for candidate in _iter_tree_sitter_nodes(redirected_statement):
        if candidate.type == "heredoc_start":
            text = _normalize_shell_token(_node_text(source, candidate))
            if text:
                return text
    return None


def _is_inside_pipeline(node: Any) -> bool:
    if isinstance(node, dict) and node.get("fallback"):
        return node.get("pattern") == "pipe_into_cat"
    current = getattr(node, "parent", None)
    while current is not None:
        if current.type == "pipeline":
            return True
        current = getattr(current, "parent", None)
    return False


def _classify_cat_pattern(node: Any, source: bytes) -> tuple[str, str | None, str | None]:
    if isinstance(node, dict) and node.get("fallback"):
        return (
            str(node.get("pattern") or "other"),
            node.get("output_target"),
            node.get("heredoc_delimiter"),
        )
    output_target = _find_output_target_for_command(node, source)
    heredoc_delimiter = _find_heredoc_delimiter_for_command(node, source)
    arguments = _command_argument_texts(node, source)
    if heredoc_delimiter and output_target:
        return "heredoc_write", output_target, heredoc_delimiter
    if _is_inside_pipeline(node) and output_target:
        return "pipe_into_cat", output_target, heredoc_delimiter
    if output_target:
        return "file_concat_to_file", output_target, heredoc_delimiter
    if arguments:
        return "file_concat_to_stdout", None, heredoc_delimiter
    return "other", None, heredoc_delimiter


def _command_references_created_file(command_info: Mapping[str, Any], target_path: str) -> bool:
    normalized_target = str(target_path).strip()
    if not normalized_target:
        return False
    basename = Path(normalized_target).name
    candidates = {normalized_target, basename, f"./{basename}"}
    for token in [command_info["name"], *command_info["args"]]:
        normalized_token = str(token).strip()
        if not normalized_token:
            continue
        if normalized_token in candidates:
            return True
        if Path(normalized_token).name == basename:
            return True
    return False


def _command_infos(root: Any, source: bytes) -> list[dict[str, Any]]:
    if isinstance(root, dict) and root.get("fallback"):
        infos = []
        for info in root.get("command_infos", []):
            infos.append(
                {
                    "node": info["node"],
                    "name": info["name"],
                    "args": list(info["args"]),
                    "start_byte": int(info["start_byte"]),
                    "source": source,
                }
            )
        return infos
    infos: list[dict[str, Any]] = []
    for node in _iter_command_nodes(root):
        name = _canonical_shell_command_name(
            _command_name_from_tree_sitter_node(node, source)
        )
        if not name:
            continue
        infos.append(
            {
                "node": node,
                "name": name,
                "args": _command_argument_texts(node, source),
                "start_byte": int(node.start_byte),
                "source": source,
            }
        )
    return sorted(infos, key=lambda item: int(item["start_byte"]))


def _find_heredoc_body_for_command(node: Any, source: bytes) -> str | None:
    if isinstance(node, dict) and node.get("fallback"):
        value = node.get("heredoc_body")
        return None if value is None else str(value)
    redirected_statement = _nearest_ancestor_of_type(node, "redirected_statement")
    if redirected_statement is None:
        return None
    for candidate in _iter_tree_sitter_nodes(redirected_statement):
        if candidate.type == "heredoc_body":
            return _node_text(source, candidate)
    return None


def _count_non_empty_lines(text: str) -> int:
    return sum(1 for line in str(text or "").splitlines() if line.strip())


def _extract_python_library_calls(code: str) -> dict[str, Counter[str]]:
    source = str(code or "").strip()
    if not source:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    module_aliases: dict[str, str] = {}
    imported_names: dict[str, tuple[str, str]] = {}
    library_calls: dict[str, Counter[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = str(alias.name or "").split(".", 1)[0].strip()
                local_name = str(alias.asname or root).strip()
                if root and local_name:
                    module_aliases[local_name] = root
        elif isinstance(node, ast.ImportFrom):
            root = str(node.module or "").split(".", 1)[0].strip()
            if not root:
                continue
            for alias in node.names:
                imported_name = str(alias.name or "").strip()
                local_name = str(alias.asname or imported_name).strip()
                if imported_name and local_name:
                    imported_names[local_name] = (root, imported_name)

    def _attribute_parts(node: ast.AST) -> list[str]:
        if isinstance(node, ast.Name):
            return [str(node.id)]
        if isinstance(node, ast.Attribute):
            parent_parts = _attribute_parts(node.value)
            if not parent_parts:
                return []
            return [*parent_parts, str(node.attr)]
        return []

    def _record_call(library: str, function_name: str) -> None:
        library_key = str(library).strip()
        function_key = str(function_name).strip()
        if not library_key or not function_key:
            return
        library_calls.setdefault(library_key, Counter())[function_key] += 1

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            local_name = str(node.func.id or "").strip()
            imported = imported_names.get(local_name)
            if imported is not None:
                library, imported_name = imported
                _record_call(library, imported_name)
            continue
        if isinstance(node.func, ast.Attribute):
            parts = _attribute_parts(node.func)
            if len(parts) < 2:
                continue
            local_name = parts[0]
            if local_name in module_aliases:
                _record_call(module_aliases[local_name], ".".join(parts[1:]))
                continue
            imported = imported_names.get(local_name)
            if imported is not None:
                library, imported_name = imported
                _record_call(library, ".".join([imported_name, *parts[1:]]))
    return library_calls


def _merge_library_calls(
    target: dict[str, Counter[str]],
    source: Mapping[str, Mapping[str, int] | Counter[str]],
) -> None:
    for library, functions in source.items():
        library_key = str(library).strip()
        if not library_key:
            continue
        entry = target.setdefault(library_key, Counter())
        for function_name, count in functions.items():
            function_key = str(function_name).strip()
            if not function_key:
                continue
            entry[function_key] += int(count)


def _candidate_script_paths(script_target: str, *, trial_dir: Path) -> list[tuple[Path, bool]]:
    normalized = str(script_target or "").strip()
    if not normalized:
        return []
    basename = Path(normalized).name
    raw_candidates: list[str] = []
    if normalized.startswith("/app/"):
        raw_candidates.append(normalized[len("/app/"):])
    if normalized.startswith("./"):
        raw_candidates.append(normalized[2:])
    if normalized.startswith("/"):
        raw_candidates.append(normalized.lstrip("/"))
    raw_candidates.extend([normalized, basename])
    seen: set[tuple[str, bool]] = set()
    candidates: list[tuple[Path, bool]] = []
    artifacts_dir = trial_dir / "artifacts"
    for raw_candidate in raw_candidates:
        candidate_text = str(raw_candidate or "").strip()
        if not candidate_text:
            continue
        candidate_path = Path(candidate_text)
        for base_dir, is_artifact in ((artifacts_dir, True), (trial_dir, False)):
            resolved = base_dir / candidate_path
            key = (str(resolved), is_artifact)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((resolved, is_artifact))
    return candidates


def _resolve_script_target_path(
    script_target: str,
    *,
    trajectory_path: Path,
) -> tuple[Path | None, bool]:
    trial_dir = _trial_dir_for_trajectory_path(trajectory_path)
    for candidate_path, is_artifact in _candidate_script_paths(script_target, trial_dir=trial_dir):
        if candidate_path.is_file():
            return candidate_path, is_artifact
    return None, False


def _read_text_if_possible(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _trim_python_code_text(code: str) -> str:
    return str(code or "").strip()


def _python_code_metadata(code: str) -> dict[str, Any]:
    normalized_code = _trim_python_code_text(code)
    loc = _count_non_empty_lines(code)
    library_calls = _extract_python_library_calls(code)
    return {
        "lines_of_code": loc,
        "library_calls": library_calls,
        "code_text": normalized_code,
    }


def _update_python_analysis_from_code(
    python_analysis: dict[str, Any],
    *,
    code: str,
) -> dict[str, Any]:
    metadata = _python_code_metadata(code)
    loc = int(metadata["lines_of_code"])
    python_analysis["lines_of_code"]["inline_total"] += loc
    python_analysis["lines_of_code"]["inline_by_invocation"].append(loc)
    _merge_library_calls(
        python_analysis["library_calls"],
        metadata["library_calls"],
    )
    return metadata


def _update_python_analysis_from_script_target(
    python_analysis: dict[str, Any],
    *,
    script_target: str,
    trajectory_path: Path,
) -> dict[str, Any]:
    target = str(script_target or "").strip()
    metadata = {
        "script_target": target,
        "lines_of_code": None,
        "library_calls": {},
        "code_text": None,
    }
    if not target:
        return metadata
    python_analysis["script_targets"][target] += 1
    resolved_path, is_artifact = _resolve_script_target_path(
        target,
        trajectory_path=trajectory_path,
    )
    if resolved_path is None:
        python_analysis["unresolved_script_targets"][target] += 1
        return metadata
    if is_artifact:
        python_analysis["artifact_script_targets"][target] += 1
    code = _read_text_if_possible(resolved_path)
    if code is None:
        python_analysis["unresolved_script_targets"][target] += 1
        return metadata
    code_metadata = _python_code_metadata(code)
    loc = int(code_metadata["lines_of_code"])
    library_calls = code_metadata["library_calls"]
    python_analysis["lines_of_code"]["script_total"] += loc
    python_analysis["lines_of_code"]["script_by_path"][target] = loc
    _merge_library_calls(
        python_analysis["library_calls"],
        library_calls,
    )
    metadata["lines_of_code"] = loc
    metadata["library_calls"] = library_calls
    metadata["code_text"] = code_metadata["code_text"]
    return metadata


def _next_python_inline_detailed_key(
    python_analysis: dict[str, Any],
    *,
    mode: str,
) -> str:
    python_analysis["detailed_inline_counters"][mode] += 1
    return f"{mode}_{int(python_analysis['detailed_inline_counters'][mode])}"


def _python_invocation_text(info: Mapping[str, Any], source: bytes, *, mode: str) -> str:
    node = info["node"]
    if isinstance(node, dict) and node.get("fallback"):
        return str(node.get("raw_text") or "").strip()
    if mode == "heredoc":
        redirected_statement = _nearest_ancestor_of_type(node, "redirected_statement")
        if redirected_statement is not None:
            return _node_text(source, redirected_statement)
    return _node_text(source, node)


def _update_python_detailed_entry(
    python_analysis: dict[str, Any],
    *,
    key: str,
    lines_of_code: int | None,
    library_calls: Mapping[str, Mapping[str, int] | Counter[str]],
    stdout_lines: int,
    stderr_lines: int,
    output_bytes: int,
    code_text: str | None = None,
    invocation_text: str | None = None,
    observation_text: str | None = None,
) -> None:
    target = str(key or "").strip()
    if not target:
        return
    entry = python_analysis["detailed"].setdefault(target, _empty_python_detailed_entry())
    entry["executions"] += 1
    if lines_of_code is not None:
        entry["lines_of_code"] = int(lines_of_code)
    if library_calls:
        _merge_library_calls(entry["library_calls"], library_calls)
    entry["total_stdout_lines"] += int(stdout_lines)
    entry["total_stderr_lines"] += int(stderr_lines)
    entry["total_output_bytes"] += int(output_bytes)
    if code_text and not entry["_code_text"]:
        entry["_code_text"] = str(code_text).strip()
    entry["_llm_tasks"].append(
        {
            "executions": 1,
            "invocation_text": str(invocation_text or "").strip(),
            "observation_text": str(observation_text or "").strip(),
        }
    )


def _update_python_subcommand_analysis_for_shell(
    python_analysis: dict[str, Any],
    *,
    command_infos: list[dict[str, Any]],
    source: bytes,
    trajectory_path: Path,
    subcommand_name: str,
) -> list[dict[str, Any]]:
    python_invocations: list[dict[str, Any]] = []
    for info in command_infos:
        if info["name"] != subcommand_name:
            continue
        python_analysis["total_invocations"] += 1
        args = [str(arg) for arg in info["args"]]
        code: str | None = None
        if args[:1] == ["-c"]:
            python_analysis["invocation_modes"]["dash_c"] += 1
            invocation_text = _python_invocation_text(info, source, mode="dash_c")
            python_invocations.append(
                {
                    "mode": "dash_c",
                    "invocation_text": invocation_text,
                    "detailed_key": _next_python_inline_detailed_key(
                        python_analysis,
                        mode="dash_c",
                    ),
                }
            )
            if len(args) >= 2:
                code = args[1]
        else:
            heredoc_body = _find_heredoc_body_for_command(info["node"], source)
            if heredoc_body:
                python_analysis["invocation_modes"]["heredoc"] += 1
                invocation_text = _python_invocation_text(info, source, mode="heredoc")
                python_invocations.append(
                    {
                        "mode": "heredoc",
                        "invocation_text": invocation_text,
                        "detailed_key": _next_python_inline_detailed_key(
                            python_analysis,
                            mode="heredoc",
                        ),
                    }
                )
                code = heredoc_body
            elif args:
                python_analysis["invocation_modes"]["script_path"] += 1
                invocation_text = _python_invocation_text(info, source, mode="script_path")
                script_metadata = _update_python_analysis_from_script_target(
                    python_analysis,
                    script_target=args[0],
                    trajectory_path=trajectory_path,
                )
                python_invocations.append(
                    {
                        "mode": "script_path",
                        "invocation_text": invocation_text,
                        "detailed_key": script_metadata["script_target"],
                        "lines_of_code": script_metadata["lines_of_code"],
                        "library_calls": script_metadata["library_calls"],
                        "code_text": script_metadata["code_text"],
                    }
                )
            else:
                python_analysis["invocation_modes"]["other"] += 1
                python_invocations.append({"mode": "other"})
        if code:
            code_metadata = _update_python_analysis_from_code(
                python_analysis,
                code=code,
            )
            python_invocations[-1]["lines_of_code"] = code_metadata["lines_of_code"]
            python_invocations[-1]["library_calls"] = code_metadata["library_calls"]
            python_invocations[-1]["code_text"] = code_metadata["code_text"]
    return python_invocations


def _extract_observation_text_for_call(
    step: Mapping[str, Any],
    *,
    tool_call_id: str,
) -> str:
    observation = step.get("observation")
    if isinstance(observation, list):
        results = observation
    else:
        return ""
    chunks: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if str(result.get("source_call_id") or "").strip() != tool_call_id:
            continue
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            chunks.append(content)
    return "\n".join(chunks)


def _count_python_failures_in_text(output_text: str, *, max_failures: int) -> int:
    if max_failures <= 0:
        return 0
    text = str(output_text or "")
    if not text.strip():
        return 0
    explicit_failures = sum(len(pattern.findall(text)) for pattern in _PYTHON_FAILURE_PATTERNS)
    if explicit_failures > 0:
        return min(max_failures, explicit_failures)
    exception_lines = len(_PYTHON_EXCEPTION_LINE_PATTERN.findall(text))
    return min(max_failures, exception_lines)


def _is_python_stderr_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if "Traceback (most recent call last):" in text:
        return True
    if _PYTHON_EXCEPTION_LINE_PATTERN.match(text):
        return True
    lowered = text.lower()
    return (
        "command failed" in lowered
        or "returned non-zero exit status" in lowered
        or "exit code" in lowered
    )


def _summarize_python_output_text(output_text: str) -> tuple[int, int, int]:
    text = str(output_text or "")
    if not text:
        return 0, 0, 0
    stdout_lines = 0
    stderr_lines = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        if _is_python_stderr_line(line):
            stderr_lines += 1
            continue
        stdout_lines += 1
    return stdout_lines, stderr_lines, len(text.encode("utf-8"))


def _update_cat_analysis_for_shell(
    cat_analysis: dict[str, Any],
    *,
    command_infos: list[dict[str, Any]],
) -> None:
    for info in command_infos:
        node = info["node"]
        if info["name"] != "cat":
            continue
        node_end_byte = (
            int(node.get("end_byte", info["start_byte"])) if isinstance(node, dict) else int(node.end_byte)
        )
        cat_analysis["total_invocations"] += 1
        source = info["source"]
        pattern, output_target, heredoc_delimiter = _classify_cat_pattern(node, source)
        cat_analysis["pattern_counts"][pattern] += 1
        if heredoc_delimiter:
            cat_analysis["heredoc_delimiters"][heredoc_delimiter] += 1
        if not output_target:
            continue
        if pattern == "heredoc_write":
            cat_analysis["heredoc_write_create_files"][output_target] += 1
        followup = next(
            (
                later
                for later in command_infos
                if int(later["start_byte"]) >= node_end_byte
                and later["name"] != "cat"
                and _command_references_created_file(later, output_target)
            ),
            None,
        )
        if followup is None:
            cat_analysis["execution_after_creation"]["not_executed_in_same_shell_snippet"] += 1
            continue
        cat_analysis["execution_after_creation"]["executed_in_same_shell_snippet"] += 1
        cat_analysis["followup_command_counts"][str(followup["name"])] += 1


def _analyze_tool_calls(
    trajectory_payload: Mapping[str, Any],
    *,
    available_tools: Mapping[str, Mapping[str, Any]],
    trajectory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    counts: Counter[str] = Counter()
    parsed_tool_counts: dict[str, Counter[str]] = defaultdict(Counter)
    parsed_tool_call_unique_counts: dict[str, Counter[str]] = defaultdict(Counter)
    file_path_tool_counts: dict[str, Counter[str]] = defaultdict(Counter)
    enabled_subcommands = {
        str(subcommand)
        for tool_config in available_tools.values()
        if str(tool_config.get("parse_method") or "").strip() == "tree_sitter_bash"
        for subcommand in (tool_config.get("subcommands") or [])
        if str(subcommand or "").strip()
    }
    subcommand_analysis = _empty_subcommand_analysis(enabled_subcommands)
    steps = trajectory_payload.get("steps")
    if not isinstance(steps, list):
        return _render_tool_call_counts(
            available_tools=available_tools,
            simple_counts=counts,
            file_path_tool_counts=file_path_tool_counts,
            parsed_tool_counts=parsed_tool_counts,
            parsed_tool_call_unique_counts=parsed_tool_call_unique_counts,
        ), subcommand_analysis
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_calls = step.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_name = str(tool_call.get("function_name") or "").strip()
            if not function_name:
                continue
            tool_call_id = str(tool_call.get("tool_call_id") or "").strip()
            tool_config = available_tools.get(function_name) or {}
            parse_method = tool_config.get("parse_method", False)
            if parse_method is False:
                counts[function_name] += 1
                continue
            arguments = tool_call.get("arguments")
            if not isinstance(arguments, dict):
                continue
            if parse_method == "file_path":
                file_path = str(arguments.get("file_path") or "").strip()
                if not file_path:
                    continue
                file_path_tool_counts[function_name][file_path] += 1
                continue
            if parse_method != "tree_sitter_bash":
                raise click.ClickException(
                    f"Unsupported parse_method {parse_method!r} for tool {function_name!r} "
                    f"in {trajectory_path}"
                )
            command = arguments.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            source, root = _parse_shell_program(
                command,
                tool_name=function_name,
                trajectory_path=trajectory_path,
            )
            tool_call_command_counts: Counter[str] = Counter()
            _update_shell_command_counts(
                root,
                source=source,
                counts=tool_call_command_counts,
            )
            parsed_tool_counts[function_name].update(tool_call_command_counts)
            parsed_tool_call_unique_counts[function_name].update(tool_call_command_counts.keys())
            command_infos = _command_infos(root, source)
            if "cat" in subcommand_analysis:
                _update_cat_analysis_for_shell(
                    subcommand_analysis["cat"],
                    command_infos=command_infos,
                )
            if "python" in subcommand_analysis:
                python_count_before = int(subcommand_analysis["python"]["total_invocations"])
                python_invocations = _update_python_subcommand_analysis_for_shell(
                    subcommand_analysis["python"],
                    command_infos=command_infos,
                    source=source,
                    trajectory_path=trajectory_path,
                    subcommand_name="python",
                )
                python_count_after = int(subcommand_analysis["python"]["total_invocations"])
                python_invocations_for_call = python_count_after - python_count_before
                if python_invocations_for_call > 0 and tool_call_id:
                    observation_text = _extract_observation_text_for_call(
                        step,
                        tool_call_id=tool_call_id,
                    )
                    stdout_lines, stderr_lines, output_bytes = _summarize_python_output_text(
                        observation_text
                    )
                    subcommand_analysis["python"]["total_stdout_lines"] += stdout_lines
                    subcommand_analysis["python"]["total_stderr_lines"] += stderr_lines
                    subcommand_analysis["python"]["total_output_bytes"] += output_bytes
                    subcommand_analysis["python"]["failed_invocations"] += (
                        _count_python_failures_in_text(
                            observation_text,
                            max_failures=python_invocations_for_call,
                        )
                    )
                    (
                        invocation_stdout_lines,
                        invocation_stderr_lines,
                        invocation_output_bytes,
                    ) = _per_invocation_output_stats(
                        python_invocation_count=python_invocations_for_call,
                        stdout_lines=stdout_lines,
                        stderr_lines=stderr_lines,
                        output_bytes=output_bytes,
                    )
                    for python_invocation in python_invocations:
                        if python_invocation.get("mode") == "other":
                            continue
                        _update_python_detailed_entry(
                            subcommand_analysis["python"],
                            key=str(python_invocation.get("detailed_key") or ""),
                            lines_of_code=python_invocation.get("lines_of_code"),
                            library_calls=python_invocation.get("library_calls")
                            or {},
                            stdout_lines=invocation_stdout_lines,
                            stderr_lines=invocation_stderr_lines,
                            output_bytes=invocation_output_bytes,
                            code_text=python_invocation.get("code_text"),
                            invocation_text=python_invocation.get("invocation_text"),
                            observation_text=observation_text,
                        )
                else:
                    for python_invocation in python_invocations:
                        if python_invocation.get("mode") == "other":
                            continue
                        _update_python_detailed_entry(
                            subcommand_analysis["python"],
                            key=str(python_invocation.get("detailed_key") or ""),
                            lines_of_code=python_invocation.get("lines_of_code"),
                            library_calls=python_invocation.get("library_calls")
                            or {},
                            stdout_lines=0,
                            stderr_lines=0,
                            output_bytes=0,
                            code_text=python_invocation.get("code_text"),
                            invocation_text=python_invocation.get("invocation_text"),
                            observation_text=None,
                        )
    rendered_counts = _render_tool_call_counts(
        available_tools=available_tools,
        simple_counts=counts,
        file_path_tool_counts=file_path_tool_counts,
        parsed_tool_counts=parsed_tool_counts,
        parsed_tool_call_unique_counts=parsed_tool_call_unique_counts,
    )
    return rendered_counts, subcommand_analysis


def _count_interactions_by_source(
    trajectory_payload: Mapping[str, Any],
) -> dict[str, int | dict[str, int]]:
    return _resolve_number_of_interactions(trajectory_payload)


def _resolve_number_of_interactions(
    trajectory_payload: Mapping[str, Any],
) -> dict[str, int | dict[str, int]]:
    counts: Counter[str] = Counter({key: 0 for key in _INTERACTION_SOURCE_KEYS})
    tool_call_steps = 0
    tool_observations = 0
    steps = trajectory_payload.get("steps")
    if not isinstance(steps, list):
        steps = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        source = str(step.get("source") or "").strip().lower()
        if source in _INTERACTION_SOURCE_KEYS:
            counts[source] += 1
        if source != "agent":
            continue
        tool_calls = step.get("tool_calls")
        if isinstance(tool_calls, list):
            tool_call_steps += sum(
                1
                for tool_call in tool_calls
                if isinstance(tool_call, dict)
                and str(tool_call.get("function_name") or "").strip()
            )
        observations = step.get("observation")
        if isinstance(observations, list):
            tool_observations += sum(1 for observation in observations if isinstance(observation, dict))
    if int(counts["user"]) != 1:
        raise click.ClickException(
            f"number_of_interactions.user must be 1; found {int(counts['user'])}."
        )
    if int(counts["system"]) != 0:
        raise click.ClickException(
            f"number_of_interactions.system must be 0; found {int(counts['system'])}."
        )
    if tool_call_steps != tool_observations:
        raise click.ClickException(
            "number_of_interactions.agent tool_call_steps and tool_observations must match; "
            f"found tool_call_steps={tool_call_steps}, tool_observations={tool_observations}."
        )
    return {
        "user": int(counts["user"]),
        "system": int(counts["system"]),
        "agent": {
            "tool_call_steps": int(tool_call_steps),
            "tool_observations": int(tool_observations),
            "steps": int(counts["agent"]),
        },
    }


def _validate_tool_calls(
    *,
    available_tools: Mapping[str, Mapping[str, Any]],
    agent_id: str,
    tool_calls: Mapping[str, Any],
    trajectory_path: Path,
) -> None:
    allowed_tools = set(available_tools)
    invalid_tools = sorted(
        tool_name for tool_name in tool_calls if tool_name not in allowed_tools
    )
    if not invalid_tools:
        return
    allowed_text = ", ".join(sorted(allowed_tools)) if allowed_tools else "<none>"
    invalid_text = ", ".join(invalid_tools)
    raise click.ClickException(
        "atif_trajectory.json contains tool calls not allowed for agent profile "
        f"{agent_id} at {trajectory_path}. "
        f"Invalid tools: {invalid_text}. Allowed tools: {allowed_text}."
    )


def _resolve_iteration_count(
    trajectory_payload: Mapping[str, Any],
) -> int:
    final_metrics = trajectory_payload.get("final_metrics")
    final_metrics_dict = final_metrics if isinstance(final_metrics, dict) else {}
    total_steps = _safe_int(final_metrics_dict.get("total_steps"))
    if total_steps is not None and total_steps > 0:
        return total_steps
    raise click.ClickException(
        "atif_trajectory.json final_metrics.total_steps must be a positive integer "
        "to compute documented token averages."
    )


def _resolve_tokens(
    trajectory_payload: Mapping[str, Any],
) -> dict[str, float | int]:
    final_metrics = trajectory_payload.get("final_metrics")
    final_metrics_dict = final_metrics if isinstance(final_metrics, dict) else {}
    total_prompt_tokens = _safe_int(final_metrics_dict.get("total_prompt_tokens")) or 0
    total_cached_tokens = _safe_int(final_metrics_dict.get("total_cached_tokens")) or 0
    total_prompt_tokens = _normalize_inclusive_prompt_tokens(
        prompt_tokens=total_prompt_tokens,
        cached_tokens=total_cached_tokens,
    )
    total_completion_tokens = _safe_int(final_metrics_dict.get("total_completion_tokens")) or 0
    iteration_count = _resolve_iteration_count(trajectory_payload)
    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_prompt_tokens_per_iteration": total_prompt_tokens / iteration_count,
        "avg_completion_tokens_per_iteration": total_completion_tokens / iteration_count,
    }


def _pricing_for_agent_profile(agent_id: str) -> tuple[float, float, float]:
    profile = _agentic_profile_by_id(agent_id)
    cost = profile["cost"]
    return (
        float(cost["input_token"]),
        float(cost["cached_token"]),
        float(cost["completion_token"]),
    )


def _calculate_total_cost_usd(
    *,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    cost_per_input_token: float,
    cost_per_cached_token: float,
    cost_per_completion_token: float,
) -> float:
    non_cached_prompt_tokens = prompt_tokens - cached_tokens
    return (
        (non_cached_prompt_tokens * cost_per_input_token)
        + (cached_tokens * cost_per_cached_token)
        + (completion_tokens * cost_per_completion_token)
    ) / 1_000_000


def _normalize_inclusive_prompt_tokens(*, prompt_tokens: int, cached_tokens: int) -> int:
    if cached_tokens > prompt_tokens:
        return prompt_tokens + cached_tokens
    return prompt_tokens


def _resolve_agent_profile_id(
    run_metadata: Mapping[str, Any],
    trajectory_payload: Mapping[str, Any],
) -> str:
    candidates = [
        str(run_metadata.get("agent_id") or "").strip(),
        str(trajectory_payload.get("agent_id") or "").strip(),
    ]
    agent_payload = trajectory_payload.get("agent")
    if isinstance(agent_payload, dict):
        candidates.append(str(agent_payload.get("name") or "").strip())
    tried: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        tried.append(candidate)
        try:
            _agentic_profile_by_id(candidate)
        except click.ClickException as exc:
            if str(exc).startswith("Unable to load agent registry:"):
                raise
            continue
        return candidate
    if tried:
        raise click.ClickException(
            "Could not resolve a known agentic profile from run_metadata/trajectory "
            f"candidates: {tried}"
        )
    raise click.ClickException(
        "Could not resolve agentic profile id from run_metadata.agent_id or trajectory.agent.name"
    )


def _required_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    errors: list[str],
) -> int | None:
    value = _safe_int(payload.get(key))
    if value is None:
        errors.append(key)
    return value


def _resolve_cost_usd(
    *,
    run_metadata: Mapping[str, Any],
    trajectory_payload: Mapping[str, Any],
    trajectory_path: Path,
) -> float:
    final_metrics = trajectory_payload.get("final_metrics")
    final_metrics_dict = final_metrics if isinstance(final_metrics, dict) else {}
    agent_profile_id = _resolve_agent_profile_id(run_metadata, trajectory_payload)
    missing_fields: list[str] = []
    prompt_tokens = _required_int(
        final_metrics_dict,
        "total_prompt_tokens",
        errors=missing_fields,
    )
    cached_tokens = _required_int(
        final_metrics_dict,
        "total_cached_tokens",
        errors=missing_fields,
    )
    completion_tokens = _required_int(
        final_metrics_dict,
        "total_completion_tokens",
        errors=missing_fields,
    )
    if missing_fields:
        raise click.ClickException(
            "atif_trajectory.json is missing required final_metrics fields to compute cost "
            f"for agent profile {agent_profile_id}: {', '.join(missing_fields)} "
            f"at {trajectory_path}"
        )
    prompt_tokens = _normalize_inclusive_prompt_tokens(
        prompt_tokens=prompt_tokens or 0,
        cached_tokens=cached_tokens or 0,
    )
    (
        cost_per_input_token,
        cost_per_cached_token,
        cost_per_completion_token,
    ) = _pricing_for_agent_profile(agent_profile_id)
    return _calculate_total_cost_usd(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens or 0,
        completion_tokens=completion_tokens or 0,
        cost_per_input_token=cost_per_input_token,
        cost_per_cached_token=cost_per_cached_token,
        cost_per_completion_token=cost_per_completion_token,
    )


def _resolve_codex_bin() -> str:
    codex_exe = shutil.which("codex")
    if not codex_exe:
        raise click.ClickException("Could not find 'codex' in PATH.")
    return codex_exe


def _sanitize_filesystem_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return sanitized or "entry"


def _resolve_codex_description_raw_dir(trajectory_path: Path) -> Path:
    return trajectory_path.parent / _CODEX_DESCRIPTION_RAW_DIRNAME


def _truncate_words(text: str, *, max_words: int) -> str:
    words = str(text or "").strip().split()
    if not words:
        return ""
    return " ".join(words[:max_words])


def _extract_text_payload(obj: object) -> list[str]:
    collected: list[str] = []
    if isinstance(obj, str):
        collected.append(obj)
        return collected
    if isinstance(obj, dict):
        for key in ("message", "content", "text", "response", "result", "data", "delta"):
            if key in obj:
                collected.extend(_extract_text_payload(obj[key]))
        item = obj.get("item")
        if item is not None:
            collected.extend(_extract_text_payload(item))
        part = obj.get("part")
        if part is not None:
            collected.extend(_extract_text_payload(part))
        parts = obj.get("parts")
        if parts is not None:
            collected.extend(_extract_text_payload(parts))
        msg = obj.get("msg")
        if msg is not None:
            collected.extend(_extract_text_payload(msg))
        return collected
    if isinstance(obj, list):
        for item in obj:
            collected.extend(_extract_text_payload(item))
    return collected


def _extract_codex_text(stdout: str) -> str:
    messages: list[str] = []
    for raw_line in str(stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages.extend(_extract_text_payload(event))
    if messages:
        return "\n".join(text for text in messages if str(text).strip()).strip()
    return str(stdout or "").strip()


def _build_python_detailed_description_prompt(
    entry_key: str,
    invocation_payload: Mapping[str, Any],
) -> str:
    invocation_text = str(invocation_payload.get("invocation_text") or "").strip() or "<unavailable>"
    observation_text = str(invocation_payload.get("observation_text") or "").strip() or "<none>"
    return (
        "Return only a JSON object with exactly these keys:\n"
        "description, number_of_semantically_relevant_numbers, intent_class, specific_subtype\n\n"
        "Guidelines:\n"
        '- description: In at most 20 words, describe what the Python code does and what it outputs. Avoid boilerplate such as "Run python3 script.py."\n'
        "- number_of_semantically_relevant_numbers: Count the meaningful numeric values visible in the execution context. Prioritize numbers from outputs. Ignore incidental numbers such as warnings, file paths, version numbers, timestamps, and library-internal values.\n"
        "- intent_class: Use exactly one of these values:\n"
        "  exploring: inspection, plotting, debugging, exploratory statistics, or hypothesis generation.\n"
        "  exploiting: inferring the label, producing the final answer, verifying a guessed class, or otherwise solving the task.\n"
        "  artifact: work that is not related to solving the task, such as metadata or file-management work.\n"
        "- specific_subtype: \n"
        "Heuristics:\n"
        "- If the output includes labels, candidate classes, or answers, classify it as exploiting.\n"
        "- If the code is trying to train a black-box algorithm on the training data, classify it as exploiting.\n\n"
        "Output requirements:\n"
        "- Return JSON only.\n"
        "- Do not include any extra keys.\n"
        "- Use the exact key names shown above.\n\n"
        f"Entry key: {entry_key}\n"
        f"Executions: {int(invocation_payload.get('executions') or 1)}\n"
        "Calls:\n"
        f"- {invocation_text}\n"
        "Observed output:\n"
        f"- {observation_text}"
    )


def _parse_codex_description_payload(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise click.ClickException("codex returned an empty description payload")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"codex returned invalid JSON payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("codex returned a non-object description payload")
    description = payload.get("description")
    number_value = payload.get("number_of_semantically_relevant_numbers")
    intent_class = payload.get("intent_class")
    specific_subtype = payload.get("specific_subtype")
    if description is not None and not isinstance(description, str):
        raise click.ClickException("codex description payload contains a non-string description")
    if intent_class is not None and not isinstance(intent_class, str):
        raise click.ClickException("codex description payload contains a non-string intent_class")
    parsed_number: int | None = None
    if number_value is not None:
        parsed_number = _safe_int(number_value)
        if parsed_number is None:
            raise click.ClickException(
                "codex description payload contains a non-integer number_of_semantically_relevant_numbers"
            )
    parsed_intent_class = str(intent_class or "").strip() or None
    if parsed_intent_class not in {"exploring", "exploiting", "artifact"}:
        raise click.ClickException(
            "codex description payload contains an invalid intent_class"
        )
    if isinstance(specific_subtype, (dict, list)):
        raise click.ClickException("codex description payload contains an invalid specific_subtype")
    return {
        "description": _truncate_words(str(description or "").strip(), max_words=20) or None,
        "number_of_semantically_relevant_numbers": parsed_number,
        "intent_class": parsed_intent_class,
        "specific_subtype": specific_subtype,
    }


def _run_codex_description(prompt: str) -> dict[str, Any]:
    cmd = [
        _resolve_codex_bin(),
        "exec",
        "--json",
        "--model",
        "gpt-5.3-codex",
        "--",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            check=False,
            capture_output=True,
            input=prompt,
        )
    except OSError as exc:
        raise click.ClickException(f"Failed to invoke codex: {exc}") from exc
    description: str | None = None
    number_of_semantically_relevant_numbers: int | None = None
    intent_class: str | None = None
    specific_subtype: Any = None
    if proc.returncode == 0:
        text = _extract_codex_text(proc.stdout)
        parsed_payload = _parse_codex_description_payload(text)
        description = parsed_payload["description"]
        number_of_semantically_relevant_numbers = parsed_payload[
            "number_of_semantically_relevant_numbers"
        ]
        intent_class = parsed_payload["intent_class"]
        specific_subtype = parsed_payload["specific_subtype"]
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
        "description": description,
        "number_of_semantically_relevant_numbers": number_of_semantically_relevant_numbers,
        "intent_class": intent_class,
        "specific_subtype": specific_subtype,
    }


def _write_codex_description_raw_artifacts(
    entry_dir: Path,
    *,
    entry_key: str,
    entry_payload: Mapping[str, Any],
    prompt: str,
    result: Mapping[str, Any],
) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "prompt.txt").write_text(str(prompt or ""), encoding="utf-8")
    (entry_dir / "stdout.txt").write_text(str(result.get("stdout") or ""), encoding="utf-8")
    (entry_dir / "stderr.txt").write_text(str(result.get("stderr") or ""), encoding="utf-8")
    metadata = {
        "entry_key": entry_key,
        "description": result.get("description"),
        "number_of_semantically_relevant_numbers": result.get(
            "number_of_semantically_relevant_numbers"
        ),
        "intent_class": result.get("intent_class"),
        "specific_subtype": result.get("specific_subtype"),
        "returncode": int(result.get("returncode") or 0),
        "codex_command": list(result.get("cmd") or []),
        "executions": int(entry_payload.get("executions") or 0),
    }
    if str(result.get("error") or "").strip():
        metadata["error"] = str(result.get("error"))
    (entry_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _safe_run_codex_description(prompt: str) -> dict[str, Any]:
    try:
        return _run_codex_description(prompt)
    except click.ClickException as exc:
        return {
            "cmd": [],
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "description": None,
            "number_of_semantically_relevant_numbers": None,
            "intent_class": None,
            "specific_subtype": None,
            "error": str(exc),
        }


def _write_trajectory_evaluation_json(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    output_path = run_dir / "trajectory_evaluation.json"
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)
    return output_path


def _per_invocation_output_stats(
    *,
    python_invocation_count: int,
    stdout_lines: int,
    stderr_lines: int,
    output_bytes: int,
) -> tuple[int, int, int]:
    if python_invocation_count == 1:
        return stdout_lines, stderr_lines, output_bytes
    return 0, 0, 0


def _result_has_valid_enrichment(result: Mapping[str, Any]) -> bool:
    return (
        result.get("error") in {None, ""}
        and int(result.get("returncode") or 0) == 0
        and result.get("description") is not None
        and result.get("number_of_semantically_relevant_numbers") is not None
        and result.get("intent_class") is not None
        and result.get("specific_subtype") is not None
    )


def _apply_codex_description_result(
    *,
    entry_key: str,
    entry_payload: dict[str, Any],
    invocation_payload: Mapping[str, Any],
    prompt: str,
    entry_dir: Path,
    result: Mapping[str, Any],
) -> None:
    if _result_has_valid_enrichment(result):
        entry_payload["description"] = result.get("description")
        entry_payload["number_of_semantically_relevant_numbers"] = result.get(
            "number_of_semantically_relevant_numbers"
        )
        entry_payload["intent_class"] = result.get("intent_class")
        entry_payload["specific_subtype"] = result.get("specific_subtype")
    _write_codex_description_raw_artifacts(
        entry_dir,
        entry_key=entry_key,
        entry_payload=invocation_payload,
        prompt=prompt,
        result=result,
    )


def _enrich_python_descriptions(
    subcommand_analysis: dict[str, Any],
    *,
    trajectory_path: Path,
    parallel: int = 1,
) -> None:
    python_analysis = subcommand_analysis.get("python")
    if not isinstance(python_analysis, dict):
        return
    detailed = python_analysis.get("detailed")
    if not isinstance(detailed, dict):
        return
    raw_dir = _resolve_codex_description_raw_dir(trajectory_path)
    tasks: list[dict[str, Any]] = []
    task_index = 0
    for entry_key, entry_payload in detailed.items():
        if not isinstance(entry_payload, dict):
            continue
        entry_payload["description"] = None
        entry_payload["number_of_semantically_relevant_numbers"] = None
        entry_payload["intent_class"] = None
        entry_payload["specific_subtype"] = None
        llm_tasks = entry_payload.get("_llm_tasks")
        if not isinstance(llm_tasks, list):
            continue
        for invocation_payload in llm_tasks:
            if not isinstance(invocation_payload, dict):
                continue
            task_index += 1
            tasks.append(
                {
                    "index": task_index,
                    "entry_key": str(entry_key),
                    "entry_payload": entry_payload,
                    "invocation_payload": invocation_payload,
                    "prompt": _build_python_detailed_description_prompt(
                        str(entry_key),
                        invocation_payload,
                    ),
                    "entry_dir": raw_dir / f"{task_index:03d}_{_sanitize_filesystem_component(str(entry_key))}",
                }
            )
    if not tasks:
        return
    if int(parallel) <= 1:
        progress_iter = tqdm(
            tasks,
            desc="Codex descriptions",
            unit="entry",
            file=sys.stderr,
            disable=not sys.stderr.isatty(),
        )
        for task in progress_iter:
            result = _safe_run_codex_description(str(task["prompt"]))
            _apply_codex_description_result(
                entry_key=str(task["entry_key"]),
                entry_payload=task["entry_payload"],
                invocation_payload=task["invocation_payload"],
                prompt=str(task["prompt"]),
                entry_dir=task["entry_dir"],
                result=result,
            )
        return
    progress = tqdm(
        total=len(tasks),
        desc="Codex descriptions",
        unit="entry",
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    )
    try:
        with ThreadPoolExecutor(max_workers=int(parallel)) as executor:
            future_to_task = {
                executor.submit(_safe_run_codex_description, str(task["prompt"])): task
                for task in tasks
            }
            completed_results: dict[int, dict[str, Any]] = {}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                completed_results[int(task["index"])] = dict(future.result())
                progress.update(1)
        for task in tasks:
            result = completed_results.get(int(task["index"])) or {
                "cmd": [],
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "description": None,
                "number_of_semantically_relevant_numbers": None,
                "intent_class": None,
                "specific_subtype": None,
                "error": "Missing completed Codex result.",
            }
            _apply_codex_description_result(
                entry_key=str(task["entry_key"]),
                entry_payload=task["entry_payload"],
                invocation_payload=task["invocation_payload"],
                prompt=str(task["prompt"]),
                entry_dir=task["entry_dir"],
                result=result,
            )
    finally:
        progress.close()


@click.command()
@click.argument("agentic_run_id", required=False)
@click.option(
    "--path",
    "run_path",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
)
@click.option("--llm", is_flag=True, help="Enrich Python detailed entries with Codex-generated metadata.")
@click.option("--parallel", type=click.IntRange(min=1), default=None, help="Run --llm Codex enrichment with N concurrent workers.")
def main(
    agentic_run_id: str | None,
    run_path: Path | None,
    llm: bool,
    parallel: int | None,
) -> None:
    if parallel is not None and not llm:
        raise click.ClickException("--parallel requires --llm.")
    run_dir = _resolve_run_dir(agentic_run_id, run_path)
    run_metadata_path = _resolve_run_metadata_path(run_dir)
    trajectory_path = _resolve_trajectory_path(agentic_run_id, run_dir)
    run_metadata = _read_json(run_metadata_path)
    output_run_id = _resolve_output_run_id(agentic_run_id, run_metadata, run_dir)
    trajectory_payload = _read_json(trajectory_path)
    agent_id = _resolve_agent_profile_id(run_metadata, trajectory_payload)
    profile = _agentic_profile_by_id(agent_id)
    available_tools = dict(profile.get("available_tools") or {})
    cost_usd = _resolve_cost_usd(
        run_metadata=run_metadata,
        trajectory_payload=trajectory_payload,
        trajectory_path=trajectory_path,
    )
    tokens = _resolve_tokens(trajectory_payload)
    artifact_analysis = _analyze_artifacts(trajectory_path)
    number_of_interactions = _resolve_number_of_interactions(trajectory_payload)
    tool_calls, subcommand_analysis = _analyze_tool_calls(
        trajectory_payload,
        available_tools=available_tools,
        trajectory_path=trajectory_path,
    )
    _validate_tool_calls(
        available_tools=available_tools,
        agent_id=agent_id,
        tool_calls=tool_calls,
        trajectory_path=trajectory_path,
    )
    if llm:
        _enrich_python_descriptions(
            subcommand_analysis,
            trajectory_path=trajectory_path,
            parallel=int(parallel or 1),
        )
    rendered_subcommand_analysis = _render_subcommand_analysis(
        subcommand_analysis,
        verbose=False,
    )
    payload = {
        "agentic_run_id": output_run_id,
        "agent_id": agent_id,
        "atif_trajectory_path": str(trajectory_path),
        "tokens": tokens,
        "artifact_analysis": artifact_analysis,
        "cost_usd": cost_usd,
        "number_of_interactions": number_of_interactions,
        "tool_calls": tool_calls,
        "subcommand_analysis": rendered_subcommand_analysis,
    }
    _write_trajectory_evaluation_json(run_dir, payload)
    click.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
