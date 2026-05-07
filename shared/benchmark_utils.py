from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

from shared.context_levels import (
    BENCHMARK_CONTEXT_VALUES as CONTEXT_LEVELS,
    PARSEABLE_CONTEXT_VALUES,
)

TSENV_CLS_CANONICAL = "tsenv_cls"

CLASSIFICATION_PREFIXES = (
    "classification_",
    "equations_",
    "ucr_",
    "uea_",
    "tsenv_cls",
)
ANOMALY_PREFIXES = ("anomaly_", "tsb_ad_")

# Central benchmark list used by CLIs and question-payload validation.
BENCHMARK_CHOICES = (
    "ucr",
    "uea",
    TSENV_CLS_CANONICAL,
)
BENCHMARK_ALIAS_TO_CANONICAL = {
    "ucr": "ucr",
    "uea": "uea",
    TSENV_CLS_CANONICAL: TSENV_CLS_CANONICAL,
}

# Allowed tsENV dataset/model directory names.
ALLOWED_TSENV_MODELS = (
    "DampedMassBetweenWalls",
    "BallDrop",
    "InclinedPlane",
)

# Shot level conventions used across exam question generation and evaluation.
SHOT_LEVELS = ("zero_shot", "one_shot", "few_shot", "many_shots")
SHOT_EXAMPLES_TOTAL = {
    "zero_shot": 0,
    "one_shot": 1,
    "few_shot": 3,
    "many_shots": 30,
}
SHOT_DEFINITIONS = {
    "zero_shot": "No in-context examples are provided.",
    "one_shot": "One in-context example is provided.",
    "few_shot": "Three in-context examples are provided.",
    "many_shots": "Thirty in-context examples are provided.",
}

# Full set of shot suffixes seen in run/model IDs (includes legacy/extra variants).
SHOT_SUFFIXES = ("zero_shot", "one_shot", "few_shot", "many_shots", "many_shot", "baseline")
_AGENT_REGISTRY_PATH = Path(__file__).resolve().parent / "config" / "agents.json"
_PLATFORM_REGISTRY_PATH = Path(__file__).resolve().parent / "platform.json"
_ALLOWED_AGENTIC_PLATFORMS = ("codex", "gemini-cli", "claude-code", "opencode")
_ALLOWED_REASONING = ("high", "low", "medium")
_ALLOWED_REASONING_PLATFORMS = ("codex", "claude-code")
_ALLOWED_TOOL_PARSE_METHODS = ("tree_sitter_bash", "file_path")


def canonical_benchmark_name(benchmark: str) -> str:
    lowered = str(benchmark or "").strip().lower()
    if not lowered:
        raise ValueError("benchmark must be non-empty")
    return BENCHMARK_ALIAS_TO_CANONICAL.get(lowered, lowered)


def is_simbench_cls_benchmark(benchmark: object) -> bool:
    text = str(benchmark or "").strip().lower()
    if not text:
        return False
    return canonical_benchmark_name(text) == TSENV_CLS_CANONICAL


def _normalize_dataset_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower())


def _match_label_token(label: str, tokens: Tuple[str, ...]) -> Optional[str]:
    for token in tokens:
        if re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", label):
            return token
    return None


def parse_context_and_shot(
    dataset_label: str,
) -> Tuple[Optional[str], Optional[str], Optional[bool]]:
    """Parse context/shot annotations embedded in dataset labels."""
    if not dataset_label:
        return None, None, None
    normalized = _normalize_dataset_label(dataset_label)
    shot = _match_label_token(
        normalized,
        ("many_shots", "many_shot", "few_shot", "one_shot", "varying_shot", "zero_shot", "baseline"),
    )
    if shot in ("zero_shot", "baseline"):
        is_few_shot = False
    elif shot in ("one_shot", "few_shot", "many_shots", "many_shot", "varying_shot"):
        is_few_shot = True
    else:
        is_few_shot = None
    context = _match_label_token(
        normalized,
        PARSEABLE_CONTEXT_VALUES,
    )
    return context, shot, is_few_shot


def _normalize_required_environment_variables(
    raw_value: Any,
    *,
    source_path: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(raw_value, dict):
        raise TypeError(
            f"{source_path} {label} required_environmental_variable must be an object"
        )
    required_env: dict[str, str] = {}
    for raw_key, raw_item in raw_value.items():
        key = str(raw_key).strip()
        value = str(raw_item).strip()
        if not key or not value:
            raise ValueError(
                f"{source_path} {label} contains empty required_environmental_variable entry"
            )
        required_env[key] = value
    return required_env


def _normalize_default_tools(
    raw_value: Any,
    *,
    source_path: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise TypeError(
            f"{source_path} {label} default_tools must be an object"
        )
    return dict(raw_value)


def _normalize_platform_entry(raw_value: Any, *, platform_id: str) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise TypeError(
            f"{_PLATFORM_REGISTRY_PATH} entry {platform_id!r} must be an object"
        )
    keys = set(raw_value)
    expected = {
        "name",
        "npm_package",
        "required_environmental_variable",
        "default_tools",
    }
    if keys != expected:
        raise ValueError(
            f"{_PLATFORM_REGISTRY_PATH} entry {platform_id!r} must define exactly "
            f"{sorted(expected)!r}, got {sorted(keys)!r}"
        )
    name = str(raw_value["name"]).strip()
    npm_package = str(raw_value["npm_package"]).strip()
    if not name:
        raise ValueError(
            f"{_PLATFORM_REGISTRY_PATH} entry {platform_id!r} name must be non-empty"
        )
    if not npm_package:
        raise ValueError(
            f"{_PLATFORM_REGISTRY_PATH} entry {platform_id!r} npm_package must be non-empty"
        )
    if name != platform_id:
        raise ValueError(
            f"{_PLATFORM_REGISTRY_PATH} entry {platform_id!r} name must match the platform_id"
        )
    if name not in _ALLOWED_AGENTIC_PLATFORMS:
        raise ValueError(
            f"Unsupported platform_id in {_PLATFORM_REGISTRY_PATH}: {name!r}"
        )
    return {
        "name": name,
        "npm_package": npm_package,
        "required_environmental_variable": _normalize_required_environment_variables(
            raw_value["required_environmental_variable"],
            source_path=_PLATFORM_REGISTRY_PATH,
            label=f"entry {platform_id!r}",
        ),
        "default_tools": _normalize_default_tools(
            raw_value["default_tools"],
            source_path=_PLATFORM_REGISTRY_PATH,
            label=f"entry {platform_id!r}",
        ),
    }


def _normalize_non_negative_float(raw_value: Any, *, idx: int, key: str) -> float:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise TypeError(
            f"{_AGENT_REGISTRY_PATH} entry {idx} {key} must be a number"
        )
    value = float(raw_value)
    if value < 0:
        raise ValueError(
            f"{_AGENT_REGISTRY_PATH} entry {idx} {key} must be non-negative"
        )
    return value


def _normalize_cost(raw_value: Any, *, idx: int) -> dict[str, float]:
    if not isinstance(raw_value, dict):
        raise TypeError(
            f"{_AGENT_REGISTRY_PATH} entry {idx} cost must be an object"
        )
    expected = {"input_token", "cached_token", "completion_token"}
    keys = set(raw_value)
    if keys != expected:
        raise ValueError(
            f"{_AGENT_REGISTRY_PATH} entry {idx} cost must define exactly "
            f"{sorted(expected)!r}, got {sorted(keys)!r}"
        )
    cost: dict[str, float] = {}
    for key in ("input_token", "cached_token", "completion_token"):
        cost[key] = _normalize_non_negative_float(
            raw_value[key],
            idx=idx,
            key=f"cost.{key}",
        )
    return cost


def _normalize_platform_id(raw_value: Any, *, idx: int) -> str:
    value = str(raw_value).strip()
    if not value:
        raise ValueError(
            f"{_AGENT_REGISTRY_PATH} entry {idx} contains empty value for 'platform_id'"
        )
    if value not in _ALLOWED_AGENTIC_PLATFORMS:
        raise ValueError(
            f"Unsupported platform_id in {_AGENT_REGISTRY_PATH}: {value!r}"
        )
    return value


def _normalize_bool(raw_value: Any, *, idx: int, key: str) -> bool:
    if not isinstance(raw_value, bool):
        raise TypeError(
            f"{_AGENT_REGISTRY_PATH} entry {idx} {key} must be a boolean"
        )
    return raw_value


def _normalize_available_tools(
    raw_value: Any,
    *,
    idx: int,
    key: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_value, dict):
        raise TypeError(
            f"{_AGENT_REGISTRY_PATH} entry {idx} {key} must be an object"
        )
    items: dict[str, dict[str, Any]] = {}
    for raw_tool_name, raw_tool_payload in raw_value.items():
        tool_name = str(raw_tool_name).strip()
        if not tool_name:
            raise ValueError(
                f"{_AGENT_REGISTRY_PATH} entry {idx} {key} must not contain empty tool names"
            )
        if not isinstance(raw_tool_payload, dict):
            raise TypeError(
                f"{_AGENT_REGISTRY_PATH} entry {idx} {key}.{tool_name} must be an object"
            )
        payload_keys = set(raw_tool_payload)
        if payload_keys != {"parse_method", "subcommands"}:
            raise ValueError(
                f"{_AGENT_REGISTRY_PATH} entry {idx} {key}.{tool_name} must define exactly "
                "['parse_method', 'subcommands']"
            )
        parse_method = raw_tool_payload["parse_method"]
        if isinstance(parse_method, bool):
            if parse_method is not False:
                raise ValueError(
                    f"{_AGENT_REGISTRY_PATH} entry {idx} {key}.{tool_name}.parse_method "
                    "must be false or one of the supported parser names"
                )
            normalized_parse_method: str | bool = False
        else:
            normalized_parse_method = str(parse_method).strip()
            if normalized_parse_method not in _ALLOWED_TOOL_PARSE_METHODS:
                raise ValueError(
                    f"{_AGENT_REGISTRY_PATH} entry {idx} {key}.{tool_name}.parse_method "
                    f"must be false or one of {list(_ALLOWED_TOOL_PARSE_METHODS)!r}"
                )
        raw_subcommands = raw_tool_payload["subcommands"]
        if not isinstance(raw_subcommands, list):
            raise TypeError(
                f"{_AGENT_REGISTRY_PATH} entry {idx} {key}.{tool_name}.subcommands must be a list"
            )
        subcommands: list[str] = []
        for raw_subcommand in raw_subcommands:
            subcommand = str(raw_subcommand).strip()
            if not subcommand:
                raise ValueError(
                    f"{_AGENT_REGISTRY_PATH} entry {idx} {key}.{tool_name}.subcommands must not contain empty values"
                )
            subcommands.append(subcommand)
        items[tool_name] = {
            "parse_method": normalized_parse_method,
            "subcommands": subcommands,
        }
    return items


def _load_platform_registry() -> dict[str, dict[str, Any]]:
    payload = json.loads(_PLATFORM_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{_PLATFORM_REGISTRY_PATH} must contain a non-empty JSON object")
    platforms: dict[str, dict[str, Any]] = {}
    for raw_platform_id, raw_entry in payload.items():
        platform_id = str(raw_platform_id).strip()
        if not platform_id:
            raise ValueError(
                f"{_PLATFORM_REGISTRY_PATH} must not contain empty platform ids"
            )
        if platform_id in platforms:
            raise ValueError(f"Duplicate platform_id in {_PLATFORM_REGISTRY_PATH}: {platform_id!r}")
        platforms[platform_id] = _normalize_platform_entry(
            raw_entry,
            platform_id=platform_id,
        )
    return platforms


def _load_agent_registry() -> tuple[dict[str, Any], ...]:
    platforms = _load_platform_registry()
    payload = json.loads(_AGENT_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{_AGENT_REGISTRY_PATH} must contain a non-empty JSON array")
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, raw_entry in enumerate(payload):
        if not isinstance(raw_entry, dict):
            raise TypeError(f"{_AGENT_REGISTRY_PATH} entries must be objects (index {idx})")
        keys = set(raw_entry)
        expected = {
            "agent_id",
            "available_tools",
            "cost",
            "model_name",
            "platform_id",
            "reasoning",
            "support_images",
        }
        if keys != expected:
            raise ValueError(
                f"{_AGENT_REGISTRY_PATH} entry {idx} must define exactly {sorted(expected)!r}, "
                f"got {sorted(keys)!r}"
            )
        entry: dict[str, Any] = {}
        for key in ("agent_id", "model_name", "reasoning"):
            raw_value = raw_entry[key]
            if key == "reasoning" and raw_value is None:
                entry[key] = None
                continue
            value = str(raw_value).strip()
            if not value:
                raise ValueError(
                    f"{_AGENT_REGISTRY_PATH} entry {idx} contains empty value for {key!r}"
                )
            entry[key] = value
        platform_id = _normalize_platform_id(
            raw_entry["platform_id"],
            idx=idx,
        )
        platform = platforms.get(platform_id)
        if platform is None:
            raise ValueError(
                f"Unknown platform_id in {_AGENT_REGISTRY_PATH}: {platform_id!r}"
            )
        entry["agentic_platform"] = {
            "name": str(platform["name"]),
            "npm_package": str(platform["npm_package"]),
            "default_tools": dict(platform["default_tools"]),
        }
        entry["required_environmental_variable"] = dict(
            platform["required_environmental_variable"]
        )
        entry["cost"] = _normalize_cost(
            raw_entry["cost"],
            idx=idx,
        )
        entry["support_images"] = _normalize_bool(
            raw_entry["support_images"],
            idx=idx,
            key="support_images",
        )
        entry["available_tools"] = _normalize_available_tools(
            raw_entry["available_tools"],
            idx=idx,
            key="available_tools",
        )
        if entry["agent_id"] in seen_ids:
            raise ValueError(f"Duplicate agent_id in {_AGENT_REGISTRY_PATH}: {entry['agent_id']!r}")
        seen_ids.add(str(entry["agent_id"]))
        reasoning = entry["reasoning"]
        if reasoning is not None and reasoning not in _ALLOWED_REASONING:
            raise ValueError(
                f"Unsupported reasoning in {_AGENT_REGISTRY_PATH}: {reasoning!r}"
            )
        if (
            reasoning is not None
            and entry["agentic_platform"]["name"] not in _ALLOWED_REASONING_PLATFORMS
        ):
            raise ValueError(
                "Only codex and claude-code profiles may define reasoning in "
                f"{_AGENT_REGISTRY_PATH}: "
                f"{entry['agent_id']!r}"
            )
        entries.append(entry)
    return tuple(entries)


def _available_agent_models_from_registry(
    registry: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in registry:
        platform = entry.get("agentic_platform") or {}
        pair = (str(platform.get("name") or ""), str(entry["model_name"]))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return tuple(out)


AGENTIC_PROFILES = _load_agent_registry()
AGENTIC_PROFILES_BY_ID = {str(entry["agent_id"]): entry for entry in AGENTIC_PROFILES}
AVAILABLE_AGENT_MODELS = _available_agent_models_from_registry(AGENTIC_PROFILES)


def agentic_profile_by_id(agent_id: str) -> dict[str, Any]:
    profile = AGENTIC_PROFILES_BY_ID.get(str(agent_id).strip())
    if profile is None:
        valid_ids = ", ".join(AGENTIC_PROFILES_BY_ID)
        raise ValueError(f"Unknown agent_id {agent_id!r}. Valid values: {valid_ids}.")
    return dict(profile)


def normalize_shot_level(shot_level: Optional[str]) -> Optional[str]:
    if not shot_level:
        return None
    lowered = str(shot_level).strip().lower()
    if lowered in SHOT_LEVELS:
        return lowered
    return None


def expected_shot_examples_total(shot_level: str) -> int:
    normalized = normalize_shot_level(shot_level)
    if normalized is None:
        raise ValueError(f"Unknown shot level: {shot_level!r}")
    return SHOT_EXAMPLES_TOTAL[normalized]


def assert_shot_example_count(
    shot_level: str,
    num_classes: int,
    paths: Sequence[object],
    *,
    per_class: Optional[int] = None,
    dataset_label: Optional[str] = None,
) -> None:
    normalized = normalize_shot_level(shot_level)
    if normalized is None:
        raise ValueError(f"Unknown shot level: {shot_level!r}")
    if per_class is not None and normalized != "zero_shot":
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1 when per_class is provided")
        if per_class < 0:
            raise ValueError("per_class must be >= 0")
        expected_total = per_class * num_classes
    else:
        expected_total = SHOT_EXAMPLES_TOTAL[normalized]
    if len(paths) != expected_total:
        suffix = f" for {dataset_label}" if dataset_label else ""
        raise ValueError(
            f"{normalized} expects {expected_total} example(s) total{suffix}, got {len(paths)}."
        )




def _benchmark_id_from_question(question: object) -> Optional[str]:
    if isinstance(question, str):
        return question
    if isinstance(question, Mapping):
        benchmark_id = question.get("benchmark_id")
        if isinstance(benchmark_id, str) and benchmark_id:
            return benchmark_id
        benchmark = question.get("benchmark")
        variant = question.get("variant")
        if isinstance(benchmark, str) and benchmark:
            if isinstance(variant, str) and variant:
                return f"{benchmark}_{variant}"
            return benchmark
    return None


def _task_from_question(question: object) -> Optional[str]:
    if isinstance(question, Mapping):
        task = question.get("task")
        if isinstance(task, str) and task:
            return task
    return None


def _matches_prefix(value: str, prefixes: Sequence[str]) -> bool:
    lowered = value.lower()
    for prefix in prefixes:
        prefix_lower = prefix.lower()
        if lowered.startswith(prefix_lower):
            return True
        if prefix_lower.endswith("_") and lowered == prefix_lower[:-1]:
            return True
    return False


def is_classification_run(question: object) -> bool:
    task = _task_from_question(question)
    if task is not None:
        return task == "classification"
    benchmark_id = _benchmark_id_from_question(question)
    if not benchmark_id:
        return False
    return _matches_prefix(benchmark_id, CLASSIFICATION_PREFIXES)


def is_few_shot_run(question: dict) -> bool:
    return question["variant"].endswith(
        (
            "_one_shot",
            "_few_shot",
            "_many_shots",
            "_many_shot",
        )
    )




def is_equation_run(question: object) -> bool:
    benchmark_id = _benchmark_id_from_question(question)
    if not benchmark_id:
        return False
    return _matches_prefix(benchmark_id, ("equations_",))


def is_anomaly_run(question: object) -> bool:
    task = _task_from_question(question)
    if task is not None:
        return task in {"anomaly_localization", "change_point_detection"}
    benchmark_id = _benchmark_id_from_question(question)
    if not benchmark_id:
        return False
    return _matches_prefix(benchmark_id, ANOMALY_PREFIXES)


def benchmark_root_from_label(label: str) -> str:
    """Map e.g. `tsenv_cls_high_few_shot` -> `tsenv_cls`.

    Raises ValueError if the label doesn't start with any known benchmark root.
    """

    lowered = str(label or "").strip().lower()
    if not lowered:
        raise ValueError("benchmark label must be non-empty")
    for root in sorted(BENCHMARK_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True):
        root_lower = root.lower()
        if lowered == root_lower or lowered.startswith(f"{root_lower}_"):
            return canonical_benchmark_name(root)
    raise ValueError(f"Unrecognized benchmark label: {label!r}")
