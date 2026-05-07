#!/usr/bin/env python3
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import click
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from shared.exam_questions_paths import TSENV_QUESTIONS_ROOT, commit_exam_questions_output, prepare_exam_questions_tmp_dir, resolve_exam_questions_output_dir  # noqa: E402
from shared.interface.distribution_json import load_experiment_config_json  # noqa: E402
from shared.interface.model_record_json import load_model_run_specs_json  # noqa: E402
from shared.interface.sample_manifest_json import validate_sample_manifest_payload  # noqa: E402
from shared.model_intervention_interface import load_allowed_interventions  # noqa: E402
from shared.run_artifacts import (  # noqa: E402
    QUESTIONS_FILENAME,
    resolve_model_record_path,
    resolve_runs_root,
    resolve_similarity_metrics_path,
)
from shared.tsenv_combinations import (  # noqa: E402
    ALL_POSSIBLE_COMBINATIONS_PATH,
    CombinationRow,
    TIME0_BASELINE_AGENT_FACING_LABEL,
    TIME0_BASELINE_LABEL,
    load_combination_rows,
)
from workflows.exam_questions import tsenv_shared  # noqa: E402

_SHARED_DESCRIPTION_CONFIG_PATH = (
    WORKSPACE_ROOT / "shared" / "config" / "tsenv_shared_description.json"
)
_QUESTION_VERSION_STATE_DIR = WORKSPACE_ROOT / "tmp" / "tsenv_question_versions"
_ALLOWED_QUESTION_TYPES = ("direct", "code", "open-ended")
_LABEL_CHOICES_PLACEHOLDER = "{label_choices_json}"
_POSSIBLE_INTERVENTIONS_PARAMETER_PLACEHOLDER = "{possible_interventions_parameter}"
_POSSIBLE_INTERVENTION_PARAMETERS_PLACEHOLDER = "{possible_intervention_parameters}"
_POSSIBLE_WITH_NO_CHANGE_PLACEHOLDER = "{possible_interventions_parameter_plus_no_parameter}"
_POSSIBLE_WITH_NO_CHANGE_LEN_PLACEHOLDER = "{len(possible_interventions_parameter_plus_no_parameter)}"
_LEN_TOTAL_OPTIONS_PLACEHOLDER = "{len_total_options}"
_BASELINE_CASE_PLACEHOLDER = "{baseline_case}"
_LEGACY_TIME0_BASELINE_PLACEHOLDER = "{time0_baseline_label}"
_CODE_FILE_DEPENDENCY_REQUIREMENT_PLACEHOLDER = "{code_file_dependency_requirement}"
_TIME0_BASELINE_DISPLAY_LABEL = TIME0_BASELINE_AGENT_FACING_LABEL
_SHARED_PROMPT_STRING_FIELDS = (
    "intervention_semantics",
    "no_change_guidance",
    "evaluation",
    "runtime_constraints",
)
_SHARED_PROMPT_MODE_FIELDS = (
    "task_artifact",
    "prediction_format",
    "mode_specific_requirements",
    "fewshot_context",
)
_NONE_DESCRIPTION_COLUMN_TEXT = (
    "The column meanings are unknown, except for the last column, which represents time."
)
_NONE_DESCRIPTION_INTERVENTION_SEMANTICS = (
    "For each simulation, either no parameter changes occur, or exactly one parameter "
    "corresponding to one of the allowed changes during the observed simulation interval.\n"
    "If a parameter changes, it undergoes a single instantaneous step change at an unknown "
    "time during the observed interval."
)
_LEGACY_CODE_SELF_CONTAINED_REQUIREMENT = "- rule.py must be self-contained."
_ZERO_SHOT_CODE_FILE_DEPENDENCY_REQUIREMENT = (
    "- You may inspect any file while developing rule.py, but the final submitted rule.py "
    "must not read, open, import, or depend on any files at prediction time, including "
    "test_samples/"
)
_FEW_SHOT_CODE_FILE_DEPENDENCY_REQUIREMENT = (
    "- You may inspect train_samples/ and train_labels.json while developing rule.py, but "
    "the final submitted rule.py must not read, open, import, or depend on any files at "
    "prediction time, including train_samples/, test_samples/ or train_labels.json"
)


def _feature_docstring_path(model_dir: Path) -> Optional[Path]:
    canonical_path = model_dir / "ground_truth_features.py"
    if canonical_path.exists():
        return canonical_path
    fallback_path = model_dir / "features.py"
    return fallback_path if fallback_path.exists() else None


def _parse_feature_docstring_patterns(docstring: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw_line in str(docstring or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("-*").strip()
        if ":" not in line:
            continue
        raw_key, raw_description = line.split(":", 1)
        key = raw_key.strip().strip("`")
        description = raw_description.strip()
        if key and description:
            out[key] = description
    return out


def _load_feature_docstring_patterns(model_dir: Path) -> Dict[str, str]:
    features_path = _feature_docstring_path(model_dir)
    if features_path is None:
        return {}
    module = ast.parse(features_path.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(module)
    if not docstring:
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and node.name == "compute_problem_specific_features":
                docstring = ast.get_docstring(node)
                break
    if not docstring:
        return {}
    parsed = _parse_feature_docstring_patterns(docstring)
    if parsed:
        return parsed
    return {"ground_truth_features": docstring.strip()}


def _observed_columns_from_signal_mapping(signal_display_mapping: Mapping[str, str]) -> str:
    rows = [
        (str(internal_name).strip(), str(agent_name).strip())
        for internal_name, agent_name in signal_display_mapping.items()
        if str(internal_name).strip() and str(agent_name).strip()
    ]
    if not rows:
        raise ValueError("internal_naming_to_agent_facing_signal must contain at least one non-empty mapping")
    return "Observed columns:\n" + "\n".join(
        f"- {internal_name}: {agent_name}" for internal_name, agent_name in rows
    )


def _load_description_levels_payload(
    model_dir: Path,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], str, Dict[str, str]]:
    levels_path = model_dir / "description_levels.json"
    if not levels_path.exists():
        raise FileNotFoundError(f"Missing description levels at {levels_path}")
    payload = json.loads(levels_path.read_text(encoding="utf-8"))
    textual_description = payload.get("textual_description")
    if not isinstance(textual_description, dict):
        raise TypeError("description_levels.json must contain a textual_description object")
    signal_display_mapping = payload.get("internal_naming_to_agent_facing_signal")
    if not isinstance(signal_display_mapping, dict) or not signal_display_mapping:
        raise TypeError(
            "description_levels.json must contain a non-empty internal_naming_to_agent_facing_signal object"
        )
    parameter_display_mapping = payload.get("internal_naming_to_agent_facing_parameter")
    if not isinstance(parameter_display_mapping, dict) or not parameter_display_mapping:
        raise TypeError(
            "description_levels.json must contain a non-empty internal_naming_to_agent_facing_parameter object"
        )
    levels = {
        "low": str(textual_description.get("low") or "").strip(),
        "high": str(textual_description.get("high") or "").strip(),
    }
    normalized_signal_display_mapping = {
        str(key).strip(): str(value).strip()
        for key, value in signal_display_mapping.items()
        if str(key).strip() and str(value).strip()
    }
    observed_signals_description = str(payload.get("observed_signals_description") or "").strip()
    observed_signals = (
        observed_signals_description
        or _observed_columns_from_signal_mapping(normalized_signal_display_mapping)
    )
    pattern_observed = _load_feature_docstring_patterns(model_dir)
    return (
        levels,
        normalized_signal_display_mapping,
        {
            str(key).strip(): str(value).strip()
            for key, value in parameter_display_mapping.items()
            if str(key).strip() and str(value).strip()
        },
        observed_signals,
        pattern_observed,
    )


def _load_agent_facing_parameter_order(model_dir: Path) -> List[str]:
    levels_path = model_dir / "description_levels.json"
    if not levels_path.exists():
        return []
    payload = json.loads(levels_path.read_text(encoding="utf-8"))
    raw_order = payload.get("agent_facing_parameter_order")
    if not isinstance(raw_order, list):
        return []
    return [str(item).strip() for item in raw_order if str(item).strip()]


def _validate_ordered_field_agent_prompt(
    value: Any,
    *,
    config_path: Path,
) -> List[List[str]]:
    if not isinstance(value, list) or not value:
        raise TypeError(
            f"{config_path} must define a non-empty ordered_field_agent_prompt list"
        )
    entries: List[List[str]] = []
    for index, raw_entry in enumerate(value):
        if not isinstance(raw_entry, list) or len(raw_entry) != 2:
            raise TypeError(
                f"{config_path} ordered_field_agent_prompt[{index}] must be a "
                "two-item list [field_name, separator]"
            )
        field_name, separator = raw_entry
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError(
                f"{config_path} ordered_field_agent_prompt[{index}][0] must be a non-empty string"
            )
        field_text = field_name.strip()
        if not isinstance(separator, str):
            raise TypeError(
                f"{config_path} ordered_field_agent_prompt[{index}][1] must be a string"
            )
        entries.append([field_text, separator])
    return entries


def _load_tsenv_shared_prompt_config(
    path: Path = _SHARED_DESCRIPTION_CONFIG_PATH,
) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing shared description config at {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{config_path} must be a JSON object")
    try:
        version = int(payload.get("version", 1))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{config_path} has an invalid version value") from exc
    if version != 1:
        raise ValueError(f"Unsupported shared description config version={version}; expected 1.")
    raw_textual_description = payload.get("textual_description")
    if not isinstance(raw_textual_description, dict):
        raise TypeError(f"{config_path} must define a textual_description object")
    textual_description_none = raw_textual_description.get("none")
    if not isinstance(textual_description_none, str):
        raise TypeError(f"{config_path} textual_description.none must be a string")
    ordered_field_agent_prompt = _validate_ordered_field_agent_prompt(
        payload.get("ordered_field_agent_prompt"),
        config_path=config_path,
    )
    shared_fields: Dict[str, str] = {}
    for field_name in _SHARED_PROMPT_STRING_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str):
            raise TypeError(f"{config_path} must define string field {field_name}")
        shared_fields[field_name] = value.strip()

    mode_fields: Dict[str, Dict[str, str]] = {}
    allowed_request_types = set(_ALLOWED_QUESTION_TYPES)
    for field_name in _SHARED_PROMPT_MODE_FIELDS:
        raw_mapping = payload.get(field_name)
        if not isinstance(raw_mapping, dict):
            raise TypeError(f"{config_path} must define a {field_name} object")
        extra_types = sorted(str(key) for key in raw_mapping if str(key) not in allowed_request_types)
        if extra_types:
            raise ValueError(
                f"{config_path} contains unsupported {field_name} keys: "
                f"{extra_types!r}; allowed={list(_ALLOWED_QUESTION_TYPES)!r}"
            )
        mode_fields[field_name] = {}
        for request_type in _ALLOWED_QUESTION_TYPES:
            value = raw_mapping.get(request_type, "")
            if not isinstance(value, str):
                raise TypeError(f"{config_path} {field_name}.{request_type} must be a string")
            mode_fields[field_name][request_type] = value.strip()
    return {
        "version": version,
        "textual_description": {
            "none": textual_description_none.strip(),
        },
        "ordered_field_agent_prompt": ordered_field_agent_prompt,
        **shared_fields,
        **mode_fields,
    }


def _load_shared_description_prompts(
    path: Path = _SHARED_DESCRIPTION_CONFIG_PATH,
) -> Dict[str, str]:
    payload = _load_tsenv_shared_prompt_config(path)
    return {
        request_type: str(payload["intervention_semantics"])
        for request_type in _ALLOWED_QUESTION_TYPES
    }


def _load_question_templates(
    path: Path = _SHARED_DESCRIPTION_CONFIG_PATH,
) -> Dict[str, str]:
    payload = _load_tsenv_shared_prompt_config(path)
    question_templates = payload.get("task_artifact")
    if not isinstance(question_templates, dict):
        raise TypeError("task_artifact payload must be an object")
    return {
        str(key): str(value)
        for key, value in question_templates.items()
    }


def _load_first_sentence_templates(
    path: Path = _SHARED_DESCRIPTION_CONFIG_PATH,
) -> Dict[str, str]:
    del path
    return {
        "zero_shot": "The time series in the test_samples/ folder were generated",
        "one_shot": "The time series in the test_samples/ and train_samples/ folders were generated",
        "few_shot": "The time series in the test_samples/ and train_samples/ folders were generated",
        "many_shot": "The time series in the test_samples/ and train_samples/ folders were generated",
    }


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _full_parent_parameters(spec_row: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(spec_row.get("baseline_parameters") or {}))


def _load_eligibility_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        raise TypeError(f"{path} must contain a baselines object")
    return payload


def _detectability_first_diff_for_child(
    *,
    eligibility_metrics: Mapping[str, Any],
    baseline_uuid: str,
    child_uuid: str,
    intervention_time: Optional[float],
) -> List[Optional[float]]:
    baselines = eligibility_metrics.get("baselines")
    if not isinstance(baselines, Mapping):
        raise TypeError("eligibility_metrics must contain a baselines object")
    baseline_entry = baselines.get(baseline_uuid)
    if not isinstance(baseline_entry, Mapping):
        raise ValueError(
            f"Missing eligibility_metrics baseline entry for baseline_uuid={baseline_uuid!r}"
        )
    children = baseline_entry.get("children")
    if not isinstance(children, Mapping):
        raise ValueError(
            f"Missing eligibility_metrics children object for baseline_uuid={baseline_uuid!r}"
        )
    child_entry = children.get(child_uuid)
    if not isinstance(child_entry, Mapping):
        raise ValueError(
            "Missing eligibility_metrics child entry for "
            f"baseline_uuid={baseline_uuid!r} child_uuid={child_uuid!r}"
        )
    detectability = child_entry.get("detectability")
    if not isinstance(detectability, Mapping):
        raise ValueError(
            f"Missing detectability for eligibility_metrics child_uuid={child_uuid!r}"
        )
    vs_baseline = detectability.get("vs_baseline")
    if not isinstance(vs_baseline, Mapping):
        raise ValueError(
            f"Missing detectability.vs_baseline for eligibility_metrics child_uuid={child_uuid!r}"
        )
    raw_first_diff = vs_baseline.get("first_diff")
    if raw_first_diff is None:
        detectability_output = vs_baseline.get("detectability_output")
        if isinstance(detectability_output, Mapping):
            raw_first_diff = detectability_output.get("first_diff")
    if raw_first_diff is None:
        raise ValueError(
            "Missing detectability.vs_baseline.first_diff "
            f"for child_uuid={child_uuid!r}"
        )
    if not isinstance(raw_first_diff, list):
        raise ValueError(
            "detectability.vs_baseline.first_diff must be a list for "
            f"child_uuid={child_uuid!r}: {raw_first_diff!r}"
        )
    values: List[Optional[float]] = []
    detectable_times: list[float] = []
    for raw_value in raw_first_diff:
        if raw_value is None:
            values.append(None)
            continue
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid detectability.vs_baseline.first_diff for "
                f"child_uuid={child_uuid!r}: {raw_first_diff!r}"
            ) from exc
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(
                "detectability.vs_baseline.first_diff must contain only nullable "
                f"non-negative finite floats for child_uuid={child_uuid!r}: {raw_first_diff!r}"
            )
        values.append(parsed)
        detectable_times.append(parsed)
    if not detectable_times:
        raise ValueError(
            "detectability.vs_baseline.first_diff must contain a non-null "
            f"detectable time for child_uuid={child_uuid!r}: {raw_first_diff!r}"
        )
    if intervention_time is None or not any(
        first_diff >= float(intervention_time) for first_diff in detectable_times
    ):
        raise ValueError(
            "detectability.vs_baseline.first_diff must contain at least one value "
            "greater than or equal to intervention_time "
            f"for child_uuid={child_uuid!r}; got first_diff={values!r}, "
            f"intervention_time={intervention_time!r}"
        )
    return values


def _null_first_diff_like(first_diff: Sequence[Any]) -> List[None]:
    return [None for _ in first_diff]


def _eligibility_child_is_explicitly_ineligible(
    *,
    eligibility_metrics: Mapping[str, Any],
    baseline_uuid: str,
    child_uuid: str,
) -> bool:
    baselines = eligibility_metrics.get("baselines")
    if not isinstance(baselines, Mapping):
        return False
    baseline_entry = baselines.get(baseline_uuid)
    if not isinstance(baseline_entry, Mapping):
        return False
    children = baseline_entry.get("children")
    if not isinstance(children, Mapping):
        return False
    child_entry = children.get(child_uuid)
    if not isinstance(child_entry, Mapping):
        return False
    return child_entry.get("eligible") is False


def _build_ground_truth_information(
    *,
    description: str,
    observed_signals: str,
    pattern_observed: Mapping[str, str],
    parameter_display_mapping: Mapping[str, str],
    model_specs: Mapping[str, Any],
    eligibility_metrics: Mapping[str, Any],
    selected_sample_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    selected_ids = (
        None
        if selected_sample_ids is None
        else {str(sample_id).strip() for sample_id in selected_sample_ids if str(sample_id).strip()}
    )

    def is_selected(sample_id: str) -> bool:
        return selected_ids is None or str(sample_id).strip() in selected_ids

    interventions_by_uuid: Dict[str, Dict[str, Any]] = {}
    for baseline_uuid, spec_row in model_specs.items():
        baseline_uuid = str(baseline_uuid or "").strip()
        if not isinstance(spec_row, dict):
            continue
        initial_parameters = _full_parent_parameters(spec_row)
        raw_interventions = spec_row.get("children")
        child_intervention_times = [
            _parse_float(intervention.get("intervention_time"))
            for intervention in (raw_interventions.values() if isinstance(raw_interventions, dict) else [])
            if isinstance(intervention, dict)
        ]
        child_intervention_times = [
            value for value in child_intervention_times if value is not None
        ]
        baseline_intervention_time = (
            child_intervention_times[0] if child_intervention_times else None
        )
        if selected_ids is not None and not is_selected(baseline_uuid):
            if not isinstance(raw_interventions, dict):
                continue
            has_selected_descendant = any(
                str(child_uuid or "").strip() in selected_ids
                or str(intervention.get("time0_baseline_uuid") or "").strip() in selected_ids
                for child_uuid, intervention in raw_interventions.items()
                if isinstance(intervention, dict)
            )
            if not has_selected_descendant:
                continue
        if baseline_uuid and is_selected(baseline_uuid):
            interventions_by_uuid[baseline_uuid] = {
                "initial_parameters": copy.deepcopy(initial_parameters),
                "intervention_time": baseline_intervention_time,
                "changed_parameter": TIME0_BASELINE_LABEL,
                "new_value": None,
                "first_diff": [],
            }
        if not isinstance(raw_interventions, dict):
            raise TypeError(
                f"model_run_specs.json children must be an object for baseline_uuid={baseline_uuid!r}"
            )
        for child_uuid, intervention in raw_interventions.items():
            if not isinstance(intervention, dict):
                continue
            child_uuid = str(child_uuid or "").strip()
            time0_uuid = str(intervention.get("time0_baseline_uuid") or "").strip()
            needs_baseline_first_diff = (
                baseline_uuid in interventions_by_uuid
                and not interventions_by_uuid[baseline_uuid]["first_diff"]
            )
            if (
                not needs_baseline_first_diff
                and not is_selected(child_uuid)
                and not (time0_uuid and is_selected(time0_uuid))
            ):
                continue
            parameters = intervention.get("parameters")
            if not isinstance(parameters, dict) or len(parameters) != 1:
                raise ValueError(
                    f"Expected one child parameter for intervention_uuid={child_uuid!r}"
                )
            changed_parameter = str(next(iter(parameters.keys())) or "").strip()
            new_value = copy.deepcopy(next(iter(parameters.values())))
            if new_value is None:
                continue
            if not child_uuid:
                continue
            if not changed_parameter:
                raise ValueError(f"Missing intervention parameter for intervention_uuid={child_uuid!r}")
            intervention_time = _parse_float(intervention.get("intervention_time"))
            if _eligibility_child_is_explicitly_ineligible(
                eligibility_metrics=eligibility_metrics,
                baseline_uuid=baseline_uuid,
                child_uuid=child_uuid,
            ):
                continue
            first_diff = _detectability_first_diff_for_child(
                eligibility_metrics=eligibility_metrics,
                baseline_uuid=baseline_uuid,
                child_uuid=child_uuid,
                intervention_time=intervention_time,
            )
            null_first_diff = _null_first_diff_like(first_diff)
            if baseline_uuid and baseline_uuid in interventions_by_uuid:
                existing_baseline_first_diff = interventions_by_uuid[baseline_uuid]["first_diff"]
                if not existing_baseline_first_diff:
                    interventions_by_uuid[baseline_uuid]["first_diff"] = null_first_diff
                elif len(existing_baseline_first_diff) != len(null_first_diff):
                    raise ValueError(
                        "Inconsistent detectability.vs_baseline.first_diff lengths for "
                        f"baseline_uuid={baseline_uuid!r}"
                    )
            if is_selected(child_uuid):
                interventions_by_uuid[child_uuid] = {
                    "initial_parameters": copy.deepcopy(initial_parameters),
                    "intervention_time": intervention_time,
                    "changed_parameter": changed_parameter,
                    "new_value": new_value,
                    "first_diff": first_diff,
                }
            if time0_uuid and is_selected(time0_uuid):
                interventions_by_uuid[time0_uuid] = {
                    "initial_parameters": copy.deepcopy(initial_parameters),
                    "intervention_time": intervention_time,
                    "changed_parameter": TIME0_BASELINE_LABEL,
                    "new_value": None,
                    "first_diff": null_first_diff,
                }
    return {
        "pattern_observed": {
            str(key).strip(): str(value).strip()
            for key, value in pattern_observed.items()
            if str(key).strip() and str(value).strip()
        },
        "interventions": interventions_by_uuid,
    }


def _select_sample_manifest_item(
    *,
    sample_manifest_payload: Any,
    shot_slug: str,
    test_set_slug: str,
    seed: int,
    num_labels: int,
    manifest_path: str,
) -> Any:
    validated = validate_sample_manifest_payload(
        sample_manifest_payload,
        num_labels=num_labels,
        path=manifest_path,
    )
    matches = [
        item
        for item in validated.get(str(shot_slug).strip(), [])
        if item.test_set_slug == str(test_set_slug).strip() and int(item.seed) == int(seed)
    ]
    if not matches:
        raise ValueError(
            f"Missing sample_manifest row for shot_slug={shot_slug!r} "
            f"test_set_slug={test_set_slug!r} seed={seed} in {manifest_path}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one sample_manifest row for shot_slug={shot_slug!r} "
            f"test_set_slug={test_set_slug!r} seed={seed} in {manifest_path}"
        )
    return matches[0]


def _ensure_dataframe_copy(run_dir: Path, output_path: Path) -> None:
    parquet_path = run_dir / "data.parquet"
    csv_path = run_dir / "data.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No data.parquet or data.csv in {run_dir}")
    if df.empty:
        raise ValueError(f"Cannot export empty dataframe from {run_dir}")
    if list(df.columns)[-1] != "time":
        raise ValueError(f'Expected last column to be "time" in {run_dir}')
    df = df.astype({column: "float32" for column in df.columns})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def _copy_noise_adder_to_export_bundle(*, model_dir: Path, output_dir: Path) -> None:
    source_path = model_dir / "noise_adder.py"
    if not source_path.exists():
        return
    shutil.copy2(source_path, output_dir / "noise_adder.py")


def _copy_model_record_to_export_bundle(
    *,
    model_dir: Path,
    output_dir: Path,
    runs_dir_name: Optional[str],
) -> None:
    source_path = resolve_model_record_path(model_dir, runs_dir_name=runs_dir_name)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing model_record.json at {source_path}")
    shutil.copy2(source_path, output_dir / "model_record.json")


def _mode_prompt_block(
    shared_prompt_config: Mapping[str, Any],
    field_name: str,
    row: CombinationRow,
    *,
    label_choices: Sequence[str],
    candidate_parameter_choices: Sequence[str],
) -> str:
    raw_mapping = shared_prompt_config.get(field_name)
    if not isinstance(raw_mapping, Mapping):
        raise TypeError(f"shared prompt config field {field_name!r} must be an object")
    try:
        template = str(raw_mapping[str(row.type_of_request)]).strip()
    except KeyError as exc:
        raise ValueError(
            f"Missing {field_name} prompt for type_of_request={row.type_of_request!r}"
        ) from exc
    return _render_prompt_template(
        template,
        label_choices=label_choices,
        candidate_parameter_choices=candidate_parameter_choices,
    )


def _code_file_dependency_requirement_for_row(row: CombinationRow) -> str:
    if int(row.train_samples_per_class) <= 0:
        return _ZERO_SHOT_CODE_FILE_DEPENDENCY_REQUIREMENT
    return _FEW_SHOT_CODE_FILE_DEPENDENCY_REQUIREMENT


def _mode_specific_requirements_for_row(
    shared_prompt_config: Mapping[str, Any],
    row: CombinationRow,
    *,
    label_choices: Sequence[str],
    candidate_parameter_choices: Sequence[str],
) -> str:
    rendered = _mode_prompt_block(
        shared_prompt_config,
        "mode_specific_requirements",
        row,
        label_choices=label_choices,
        candidate_parameter_choices=candidate_parameter_choices,
    )
    if str(row.type_of_request).strip() not in {"code", "open-ended"}:
        return rendered

    dependency_requirement = _code_file_dependency_requirement_for_row(row)
    if _CODE_FILE_DEPENDENCY_REQUIREMENT_PLACEHOLDER in rendered:
        return rendered.replace(
            _CODE_FILE_DEPENDENCY_REQUIREMENT_PLACEHOLDER,
            dependency_requirement,
        )
    return rendered.replace(
        _LEGACY_CODE_SELF_CONTAINED_REQUIREMENT,
        dependency_requirement,
    )


def _shared_prompt_block(
    shared_prompt_config: Mapping[str, Any],
    field_name: str,
    *,
    label_choices: Sequence[str],
    candidate_parameter_choices: Sequence[str],
) -> str:
    template = str(shared_prompt_config.get(field_name) or "").strip()
    return _render_prompt_template(
        template,
        label_choices=label_choices,
        candidate_parameter_choices=candidate_parameter_choices,
    )


def _candidate_parameter_choices(
    *,
    row: CombinationRow,
    label_choices: Sequence[str],
    parameter_display_mapping: Mapping[str, str],
    question_seed: int,
) -> List[str]:
    if str(row.desc_level).strip().lower() == "none":
        return [
            str(label).strip()
            for label in list(label_choices)[:-1]
            if str(label).strip()
        ]
    interventions = [
        str(value).strip()
        for value in parameter_display_mapping.values()
        if str(value).strip()
    ]
    shuffled = list(interventions)
    random.Random(int(question_seed)).shuffle(shuffled)
    return shuffled


def _format_natural_quoted_list(values: Sequence[str]) -> str:
    quoted = [json.dumps(str(value)) for value in values]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _render_prompt_template(
    template: str,
    *,
    label_choices: Sequence[str],
    candidate_parameter_choices: Sequence[str],
) -> str:
    rendered = str(template).strip()
    prompt_label_choices = [str(label) for label in label_choices if str(label).strip()]
    rendered = rendered.replace(
        _POSSIBLE_WITH_NO_CHANGE_LEN_PLACEHOLDER,
        str(len(prompt_label_choices)),
    )
    rendered = rendered.replace(
        _LEN_TOTAL_OPTIONS_PLACEHOLDER,
        str(len(prompt_label_choices)),
    )
    rendered = rendered.replace(
        _POSSIBLE_INTERVENTION_PARAMETERS_PLACEHOLDER,
        _format_natural_quoted_list(candidate_parameter_choices),
    )
    rendered = rendered.replace(
        _POSSIBLE_INTERVENTIONS_PARAMETER_PLACEHOLDER,
        _format_natural_quoted_list(candidate_parameter_choices),
    )
    rendered = rendered.replace(
        _POSSIBLE_WITH_NO_CHANGE_PLACEHOLDER,
        json.dumps(prompt_label_choices),
    )
    rendered = rendered.replace(
        _LABEL_CHOICES_PLACEHOLDER,
        json.dumps(prompt_label_choices),
    )
    rendered = rendered.replace(_BASELINE_CASE_PLACEHOLDER, _TIME0_BASELINE_DISPLAY_LABEL)
    rendered = rendered.replace(_LEGACY_TIME0_BASELINE_PLACEHOLDER, _TIME0_BASELINE_DISPLAY_LABEL)
    return rendered


def _sample_source_for_row(row: CombinationRow) -> str:
    folders = (
        "test_samples/ folder"
        if int(row.train_samples_per_class) <= 0
        else "test_samples/ and train_samples/ folders"
    )
    if _is_none_description_row(row):
        return (
            f"Context:\nThe time series in the {folders} were generated by a simulator "
            "of an unknown physical phenomenon."
        )
    prefix = "Context:\n" if row.type_of_request in {"code", "open-ended"} else ""
    return f"{prefix}The time series in the {folders} were generated"


def _is_none_description_row(row: CombinationRow) -> bool:
    return str(row.desc_level).strip().lower() == "none"


def _environment_description_for_row(
    *,
    row: CombinationRow,
    description_levels: Mapping[str, str],
    shared_textual_description_none: str,
) -> str:
    if _is_none_description_row(row):
        configured = str(shared_textual_description_none or "").strip()
        return configured or _NONE_DESCRIPTION_COLUMN_TEXT
    return str(description_levels[row.desc_level]).strip()


def _label_space(label_choices: Sequence[str]) -> str:
    return "Allowed labels:\n" + json.dumps([str(label) for label in label_choices])


def _none_description_label_space(
    label_choices: Sequence[str],
    *,
    type_of_request: str,
) -> str:
    labels = [str(label).strip() for label in label_choices if str(label).strip()]
    if len(labels) < 2:
        return json.dumps(labels)
    parameter_labels = labels[:-1]
    no_change_label = labels[-1]
    separator = "  " if type_of_request in {"code", "open-ended"} else " "
    return (
        f"{json.dumps(parameter_labels)}{separator}denote different parameter changes, "
        f'while "{no_change_label}" denotes that no parameter changed.'
    )


def _fewshot_context_for_row(
    *,
    row: CombinationRow,
    shared_prompt_config: Mapping[str, Any],
    label_choices: Sequence[str],
    candidate_parameter_choices: Sequence[str],
) -> str:
    if int(row.train_samples_per_class) <= 0:
        return ""
    return _mode_prompt_block(
        shared_prompt_config,
        "fewshot_context",
        row,
        label_choices=label_choices,
        candidate_parameter_choices=candidate_parameter_choices,
    )


def _evaluation_for_row(
    *,
    row: CombinationRow,
    shared_prompt_config: Mapping[str, Any],
    label_choices: Sequence[str],
    candidate_parameter_choices: Sequence[str],
) -> str:
    rendered = _shared_prompt_block(
        shared_prompt_config,
        "evaluation",
        label_choices=label_choices,
        candidate_parameter_choices=candidate_parameter_choices,
    )
    if row.type_of_request in {"code", "open-ended"}:
        return rendered.replace(
            "the first label in each returned list is compared",
            "the first label returned by predict(df) is compared",
        )
    return rendered


def _documented_label_order(
    *,
    internal_labels: Sequence[str],
    parameter_display_mapping: Mapping[str, str],
    preferred_internal_order: Sequence[str] = (),
) -> List[str]:
    display_by_internal = {
        str(label).strip(): str(parameter_display_mapping.get(label, label)).strip()
        for label in internal_labels
        if label != TIME0_BASELINE_LABEL and str(label).strip()
    }
    preferred = [str(label).strip() for label in preferred_internal_order if str(label).strip()]
    if preferred:
        labels = [
            display_by_internal.pop(label)
            for label in preferred
            if label in display_by_internal and display_by_internal[label]
        ]
        labels.extend(sorted({label for label in display_by_internal.values() if label}))
    else:
        labels = sorted({label for label in display_by_internal.values() if label})
    return [*labels, _TIME0_BASELINE_DISPLAY_LABEL]


def _documented_none_label_order(label_choices: Sequence[str]) -> List[str]:
    parameter_labels = [
        str(label).strip()
        for label in label_choices
        if str(label).strip() and str(label).strip() != _TIME0_BASELINE_DISPLAY_LABEL
    ]
    return [f"label_{idx}" for idx in range(len(parameter_labels) + 1)]


def _prompt_label_choices_for_row(
    *,
    row: CombinationRow,
    documented_label_choices: Sequence[str],
) -> List[str]:
    if str(row.desc_level).strip().lower() == "none":
        return _documented_none_label_order(documented_label_choices)
    return [str(label).strip() for label in documented_label_choices if str(label).strip()]


def _ordered_field_agent_prompt_for_row(
    *,
    row: CombinationRow,
    shared_prompt_config: Mapping[str, Any],
) -> List[List[str]]:
    if not _is_none_description_row(row):
        return copy.deepcopy(shared_prompt_config["ordered_field_agent_prompt"])
    return [
        ["sample_source", "\n"],
        ["environment_description", "\n\n"],
        ["intervention_semantics", "\n\n"],
        ["label_space", "\n\n"],
        ["task_artifact", "\n"],
        ["prediction_format", "\n\n"],
        ["fewshot_context", "\n\n"],
        ["mode_specific_requirements", "\n\n"],
        ["evaluation", "\n\n"],
        ["runtime_constraints", ""],
    ]


def _build_question_text_payload(
    *,
    row: CombinationRow,
    description_levels: Mapping[str, str],
    observed_signals: str,
    label_choices: Sequence[str],
    parameter_display_mapping: Mapping[str, str],
    question_seed: int,
    shared_prompt_config: Mapping[str, Any],
    shared_textual_description_none: str,
) -> Dict[str, Any]:
    prompt_label_choices = _prompt_label_choices_for_row(
        row=row,
        documented_label_choices=label_choices,
    )
    candidate_parameter_choices = _candidate_parameter_choices(
        row=row,
        label_choices=prompt_label_choices,
        parameter_display_mapping=parameter_display_mapping,
        question_seed=int(question_seed),
    )
    is_none_description = _is_none_description_row(row)
    return {
        "sample_source": _sample_source_for_row(row),
        "environment_description": _environment_description_for_row(
            row=row,
            description_levels=description_levels,
            shared_textual_description_none=shared_textual_description_none,
        ),
        "observed_columns": "" if is_none_description else str(observed_signals or "").strip(),
        "intervention_semantics": (
            _NONE_DESCRIPTION_INTERVENTION_SEMANTICS
            if is_none_description
            else _shared_prompt_block(
                shared_prompt_config,
                "intervention_semantics",
                label_choices=prompt_label_choices,
                candidate_parameter_choices=candidate_parameter_choices,
            )
        ),
        "allowed_labels": prompt_label_choices,
        "label_space": (
            _none_description_label_space(
                prompt_label_choices,
                type_of_request=str(row.type_of_request),
            )
            if is_none_description
            else _label_space(prompt_label_choices)
        ),
        "no_change_guidance": "" if is_none_description else _shared_prompt_block(
            shared_prompt_config,
            "no_change_guidance",
            label_choices=prompt_label_choices,
            candidate_parameter_choices=candidate_parameter_choices,
        ),
        "fewshot_context": _fewshot_context_for_row(
            row=row,
            shared_prompt_config=shared_prompt_config,
            label_choices=prompt_label_choices,
            candidate_parameter_choices=candidate_parameter_choices,
        ),
        "task_artifact": _mode_prompt_block(
            shared_prompt_config,
            "task_artifact",
            row,
            label_choices=prompt_label_choices,
            candidate_parameter_choices=candidate_parameter_choices,
        ),
        "prediction_format": _mode_prompt_block(
            shared_prompt_config,
            "prediction_format",
            row,
            label_choices=prompt_label_choices,
            candidate_parameter_choices=candidate_parameter_choices,
        ),
        "mode_specific_requirements": _mode_specific_requirements_for_row(
            shared_prompt_config,
            row,
            label_choices=prompt_label_choices,
            candidate_parameter_choices=candidate_parameter_choices,
        ),
        "evaluation": _evaluation_for_row(
            row=row,
            shared_prompt_config=shared_prompt_config,
            label_choices=prompt_label_choices,
            candidate_parameter_choices=candidate_parameter_choices,
        ),
        "runtime_constraints": _shared_prompt_block(
            shared_prompt_config,
            "runtime_constraints",
            label_choices=prompt_label_choices,
            candidate_parameter_choices=candidate_parameter_choices,
        ),
        "ordered_field_agent_prompt": _ordered_field_agent_prompt_for_row(
            row=row,
            shared_prompt_config=shared_prompt_config,
        ),
    }


def _recipe_info_payload(row: CombinationRow, *, seed: int) -> Dict[str, Any]:
    return {
        "type_of_request": row.type_of_request,
        "desc_level": row.desc_level,
        "noise_level": row.noise_level,
        "number_train_samples_per_class": int(row.number_train_samples_per_class),
        "number_test_samples": int(row.number_test_samples),
        "is_adversarial": row.is_adversarial,
        "question_seed": int(seed),
        "test_set_slug": row.test_set_slug,
        "shot_slug": row.shot_slug,
        "row_slug": row.row_slug,
    }


def _question_slug_for_row(row: CombinationRow, *, seed: int) -> str:
    return f"{row.row_slug}_{int(seed)}"


def _copy_selected_samples(
    *,
    model_name: str,
    runs_dir: Path,
    output_dir: Path,
    sample_ids: Sequence[str],
) -> Dict[str, str]:
    sample_paths_by_id: Dict[str, str] = {}
    for sample_id in sample_ids:
        sample_uuid = str(sample_id).strip()
        if not sample_uuid:
            continue
        run_dir = runs_dir / sample_uuid
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Missing run folder for sample {sample_uuid!r} under {runs_dir} (model={model_name!r})"
            )
        rel_path = f"dataframes/{sample_uuid}.parquet"
        _ensure_dataframe_copy(run_dir, output_dir / rel_path)
        sample_paths_by_id[sample_uuid] = rel_path
    return sample_paths_by_id


def _copy_sample_manifest_to_export_bundle(*, sample_manifest_path: Path, output_dir: Path) -> None:
    shutil.copy2(sample_manifest_path, output_dir / "sample_manifest.json")


def _selected_sample_ids_from_manifest_items(items: Sequence[Any]) -> List[str]:
    return sorted(
        {
            str(sample_id).strip()
            for item in items
            for sample_id in [
                *(getattr(item, "train_samples", None) or []),
                *(getattr(item, "test_samples", None) or []),
                *(getattr(item, "other_samples", None) or []),
            ]
            if str(sample_id).strip()
        }
    )


def _question_version_state_path(
    existing_questions_path: Path,
    *,
    state_dir: Optional[Path] = None,
) -> Path:
    resolved = Path(existing_questions_path).expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
    root = _QUESTION_VERSION_STATE_DIR if state_dir is None else Path(state_dir).expanduser().resolve()
    return root / f"{digest}.json"


def _read_questions_version(path: Path) -> Optional[int]:
    path = Path(path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    raw_version = payload.get("version")
    if raw_version is None:
        return None
    if isinstance(raw_version, bool):
        raise TypeError(f"{path} version must be an integer, got {raw_version!r}")
    try:
        return int(raw_version)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{path} version must be an integer, got {raw_version!r}") from exc


def _next_questions_version(
    existing_questions_path: Path,
    *,
    persist: bool = False,
    state_dir: Optional[Path] = None,
) -> int:
    path = Path(existing_questions_path)
    versions = [_read_questions_version(path)]
    state_path = _question_version_state_path(path, state_dir=state_dir)
    if state_path.exists():
        versions.append(_read_questions_version(state_path))
    next_version = max([version for version in versions if version is not None] or [0]) + 1
    if persist:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"version": next_version}, indent=2),
            encoding="utf-8",
        )
    return next_version


def _export_model(
    *,
    model_name: str,
    row_slugs: Sequence[str],
    combinations_csv: Path,
    output_root: Path,
    runs_dir: Optional[Path],
    runs_dir_name: Optional[str],
) -> None:
    model_dir = Path("models") / "simulink" / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory at {model_dir}")
    rows = load_combination_rows(combinations_csv, row_slugs=row_slugs)
    model_output_dir = resolve_exam_questions_output_dir(output_root, model_name)
    questions_version = _next_questions_version(
        model_output_dir / QUESTIONS_FILENAME,
        persist=True,
    )
    sample_manifest_path = model_output_dir / "sample_manifest.json"
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"Missing {sample_manifest_path}")
    sample_manifest_payload = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    experiment_config = load_experiment_config_json(model_dir / "experiment_config.json")
    internal_labels = list(load_allowed_interventions(model_id=model_name, models_root=Path("models") / "simulink"))
    internal_labels.append(TIME0_BASELINE_LABEL)
    (
        description_levels,
        _signal_display_mapping,
        parameter_display_mapping,
        observed_signals,
        pattern_observed,
    ) = _load_description_levels_payload(model_dir)
    shared_prompt_config = _load_tsenv_shared_prompt_config()
    shared_textual_description_none = str(shared_prompt_config["textual_description"]["none"])
    preferred_parameter_order = _load_agent_facing_parameter_order(model_dir)
    model_specs = load_model_run_specs_json(
        model_dir / "model_run_specs.json",
        enforce_baseline_pair_diversity=False,
    )
    resolved_runs_dir = resolve_runs_root(model_dir, runs_dir=runs_dir, runs_dir_name=runs_dir_name)
    eligibility_metrics = _load_eligibility_metrics(
        resolve_similarity_metrics_path(model_dir, runs_dir=runs_dir, runs_dir_name=runs_dir_name)
    )
    selected_manifest_items = [
        _select_sample_manifest_item(
            sample_manifest_payload=sample_manifest_payload,
            shot_slug=row.shot_slug,
            test_set_slug=row.test_set_slug,
            seed=seed,
            num_labels=len(internal_labels),
            manifest_path=str(sample_manifest_path),
        )
        for row in rows
        for seed in row.seeds
    ]
    selected_sample_ids = _selected_sample_ids_from_manifest_items(selected_manifest_items)
    ground_truth_information = _build_ground_truth_information(
        description=str(description_levels["high"]),
        observed_signals=observed_signals,
        pattern_observed=pattern_observed,
        parameter_display_mapping=parameter_display_mapping,
        model_specs=model_specs,
        eligibility_metrics=eligibility_metrics,
        selected_sample_ids=selected_sample_ids,
    )
    staging_dir = prepare_exam_questions_tmp_dir(resolve_exam_questions_output_dir(output_root, model_name))
    _copy_noise_adder_to_export_bundle(model_dir=model_dir, output_dir=staging_dir)
    _copy_model_record_to_export_bundle(model_dir=model_dir, output_dir=staging_dir, runs_dir_name=runs_dir_name)
    _copy_sample_manifest_to_export_bundle(sample_manifest_path=sample_manifest_path, output_dir=staging_dir)

    sample_paths_by_id = _copy_selected_samples(
        model_name=model_name,
        runs_dir=resolved_runs_dir,
        output_dir=staging_dir,
        sample_ids=selected_sample_ids,
    )

    questions: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        label_choices = _documented_label_order(
            internal_labels=internal_labels,
            parameter_display_mapping=parameter_display_mapping,
            preferred_internal_order=preferred_parameter_order,
        )
        for seed in row.seeds:
            manifest_item = _select_sample_manifest_item(
                sample_manifest_payload=sample_manifest_payload,
                shot_slug=row.shot_slug,
                test_set_slug=row.test_set_slug,
                seed=seed,
                num_labels=len(internal_labels),
                manifest_path=str(sample_manifest_path),
            )
            question_slug = _question_slug_for_row(row, seed=int(seed))
            if question_slug in questions:
                raise ValueError(f"Duplicate generated question_slug={question_slug!r}")
            questions[question_slug] = {
                "question_hash": str(manifest_item.train_test_sample_hash),
                "train_test_sample_hash": str(manifest_item.train_test_sample_hash),
                "question_text": _build_question_text_payload(
                    row=row,
                    description_levels=description_levels,
                    observed_signals=observed_signals,
                    label_choices=label_choices,
                    parameter_display_mapping=parameter_display_mapping,
                    question_seed=int(seed),
                    shared_prompt_config=shared_prompt_config,
                    shared_textual_description_none=shared_textual_description_none,
                ),
                "recipe_info": _recipe_info_payload(row, seed=seed),
            }

    payload = {
        "version": questions_version,
        "questions": questions,
        "label_int_mapping": {
            label: idx
            for idx, label in enumerate(
                _documented_label_order(
                    internal_labels=internal_labels,
                    parameter_display_mapping=parameter_display_mapping,
                    preferred_internal_order=preferred_parameter_order,
                )
            )
        },
        "ground_truth_information": ground_truth_information,
        "environment_name": model_name,
    }
    (staging_dir / QUESTIONS_FILENAME).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    commit_exam_questions_output(
        staging_dir,
        resolve_exam_questions_output_dir(output_root, model_name),
        overwrite=True,
    )
    click.echo(f"Wrote {len(questions)} questions to {resolve_exam_questions_output_dir(output_root, model_name) / QUESTIONS_FILENAME}")


@click.command()
@click.option("--model", "models", multiple=True, help="tsENV model name to export.")
@click.option("--all-models", is_flag=True, help="Export all known tsENV models.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=TSENV_QUESTIONS_ROOT,
    show_default=True,
    help="Output directory under tsENV_questions.",
)
@click.option(
    "--combinations-csv",
    type=click.Path(path_type=Path),
    default=ALL_POSSIBLE_COMBINATIONS_PATH,
    show_default=True,
    help="Path to all_possible_combinations.csv.",
)
@click.option(
    "--row-slug",
    "row_slugs",
    multiple=True,
    required=True,
    help="row_slug values from all_possible_combinations.csv to materialize.",
)
@click.option(
    "--runs-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Override the run-artifact root for this model. Default: models/simulink/<MODEL>/runs.",
)
@click.option(
    "--runs-dir-name",
    type=str,
    default=None,
    help="Model-local run-artifact directory name (for example: runs_7161).",
)
def main(
    models: Sequence[str],
    all_models: bool,
    output_dir: Path,
    combinations_csv: Path,
    row_slugs: Sequence[str],
    runs_dir: Optional[Path],
    runs_dir_name: Optional[str],
) -> None:
    if all_models:
        target_models = list(tsenv_shared.ALLOWED_TSENV_MODELS)
    else:
        target_models = [str(model).strip() for model in models if str(model).strip()]
    if not target_models:
        raise click.UsageError("Provide --model or --all-models.")
    if runs_dir is not None and len(target_models) != 1:
        raise click.UsageError("--runs-dir requires exactly one model.")

    filtered_rows = load_combination_rows(combinations_csv, row_slugs=row_slugs)
    grouped_by_model = {model_name: list(filtered_rows) for model_name in target_models}
    for model_name in target_models:
        _export_model(
            model_name=model_name,
            row_slugs=[row.row_slug for row in grouped_by_model[model_name]],
            combinations_csv=combinations_csv,
            output_root=output_dir.expanduser().resolve(),
            runs_dir=runs_dir,
            runs_dir_name=runs_dir_name,
        )


if __name__ == "__main__":
    main()
