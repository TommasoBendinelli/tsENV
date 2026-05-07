from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional

from shared.interface.sample_manifest_json import (
    ManifestItem,
    validate_sample_manifest_payload,
)
from shared.model_intervention_interface import load_allowed_interventions
from shared.run_artifacts import QUESTIONS_FILENAME, SAMPLE_MANIFEST_FILENAME
from shared.tsenv_combinations import (
    TIME0_BASELINE_AGENT_FACING_LABEL,
    TIME0_BASELINE_LABEL,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODELS_ROOT = _REPO_ROOT / "models" / "simulink"


def resolve_tsenv_payload_path(path: Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        questions_path = candidate / QUESTIONS_FILENAME
        if questions_path.exists():
            return questions_path
        raise FileNotFoundError(f"Missing {QUESTIONS_FILENAME} under {candidate}")
    return candidate


def _payload_dir(payload_path: Path) -> Path:
    return payload_path.parent if payload_path.name == QUESTIONS_FILENAME else payload_path.parent


def _load_sample_manifest_if_present(payload_dir: Path) -> Optional[Dict[str, Any]]:
    sample_manifest_path = payload_dir / SAMPLE_MANIFEST_FILENAME
    if not sample_manifest_path.exists():
        return None
    payload = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{sample_manifest_path} must contain a JSON object keyed by shot_slug")
    return _normalize_loaded_sample_manifest(payload, payload_dir=payload_dir)


def _normalize_loaded_sample_manifest(
    payload: Mapping[str, Any],
    *,
    payload_dir: Path,
) -> Dict[str, Any]:
    dataframe_sample_ids = sorted(
        path.stem for path in (payload_dir / "dataframes").glob("*.parquet")
    )
    normalized: Dict[str, Any] = {}
    for shot_slug, raw_rows in payload.items():
        if not isinstance(raw_rows, list):
            normalized[str(shot_slug)] = raw_rows
            continue
        rows: List[Any] = []
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                rows.append(raw_row)
                continue
            row = dict(raw_row)
            if "other_samples" not in row:
                explicit_others = row.pop("others", None)
                if isinstance(explicit_others, list):
                    row["other_samples"] = [
                        str(sample_id).strip()
                        for sample_id in explicit_others
                        if str(sample_id).strip()
                    ]
                else:
                    selected = {
                        str(sample_id).strip()
                        for field_name in ("train_samples", "test_samples")
                        for sample_id in row.get(field_name, [])
                        if str(sample_id).strip()
                    }
                    row["other_samples"] = [
                        sample_id
                        for sample_id in dataframe_sample_ids
                        if sample_id not in selected
                    ]
            rows.append(row)
        normalized[str(shot_slug)] = rows
    return normalized


def load_metadata_payload(metadata_path: Path) -> Dict[str, Any]:
    payload_path = resolve_tsenv_payload_path(metadata_path)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{payload_path} must contain a JSON object")
    payload["_payload_dir"] = str(_payload_dir(payload_path))
    sample_manifest_payload = _load_sample_manifest_if_present(_payload_dir(payload_path))
    if sample_manifest_payload is not None:
        payload["_sample_manifest_payload"] = sample_manifest_payload
    return payload


def metadata_questions_by_id(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_questions = payload.get("questions")
    if isinstance(raw_questions, dict):
        out: Dict[str, Dict[str, Any]] = {}
        for question_id, question in raw_questions.items():
            qid = str(question_id or "").strip()
            if not qid:
                raise ValueError("questions contains an empty question_id key")
            if not isinstance(question, dict):
                raise TypeError(f"questions[{qid!r}] must be an object")
            out[qid] = {"question_id": qid, **question}
        return out
    raise TypeError("questions must be an object")


def metadata_questions_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(metadata_questions_by_id(payload).values())


def label_choices_from_payload(payload: Dict[str, Any]) -> List[str]:
    raw_mapping = payload.get("label_int_mapping")
    if not isinstance(raw_mapping, dict):
        return []
    indexed: List[tuple[int, str]] = []
    for raw_label, raw_idx in raw_mapping.items():
        label = str(raw_label or "").strip()
        if not label:
            continue
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label_int_mapping index for {label!r}: {raw_idx!r}") from exc
        indexed.append((idx, label))
    indexed.sort(key=lambda item: (item[0], item[1]))
    return [label for _, label in indexed]


def _question_recipe_info(question: Mapping[str, Any]) -> Mapping[str, Any]:
    recipe_info = question.get("recipe_info")
    if isinstance(recipe_info, Mapping):
        return recipe_info
    return {}


def _question_textual_info(question: Mapping[str, Any]) -> str:
    recipe_info = _question_recipe_info(question)
    return str(recipe_info.get("desc_level") or "none").strip().lower() or "none"


def _load_parameter_display_mapping_from_model(payload: Mapping[str, Any]) -> Dict[str, str]:
    payload_dir_text = str(payload.get("_payload_dir") or "").strip()
    environment_name = str(payload.get("environment_name") or "").strip()
    if not payload_dir_text or not environment_name:
        return {}
    description_levels_path = _MODELS_ROOT / environment_name / "description_levels.json"
    if not description_levels_path.exists():
        return {}
    description_levels = json.loads(description_levels_path.read_text(encoding="utf-8"))
    raw_mapping = description_levels.get("internal_naming_to_agent_facing_parameter")
    if not isinstance(raw_mapping, Mapping):
        return {}
    return {
        str(key).strip(): str(value).strip()
        for key, value in raw_mapping.items()
        if str(key).strip() and str(value).strip()
    }


def _parameter_display_mapping(payload: Mapping[str, Any]) -> Dict[str, str]:
    ground_truth_information = payload.get("ground_truth_information")
    if isinstance(ground_truth_information, Mapping):
        raw_mapping = ground_truth_information.get("parameter_display_mapping")
        if isinstance(raw_mapping, Mapping):
            return {
                str(key).strip(): str(value).strip()
                for key, value in raw_mapping.items()
                if str(key).strip() and str(value).strip()
            }
    cached = payload.get("_parameter_display_mapping")
    if isinstance(cached, Mapping):
        return {
            str(key).strip(): str(value).strip()
            for key, value in cached.items()
            if str(key).strip() and str(value).strip()
        }
    return _load_parameter_display_mapping_from_model(payload)


def _label_agnostic_choices(
    payload: Mapping[str, Any],
    question: Mapping[str, Any],
) -> List[str]:
    question_text = question.get("question_text")
    if isinstance(question_text, Mapping):
        allowed_labels = question_text.get("allowed_labels")
        if isinstance(allowed_labels, list):
            choices = [
                str(choice).strip()
                for choice in allowed_labels
                if str(choice).strip()
            ]
            if choices:
                return choices
        label_choices_json = question_text.get("label_choices_json")
        if isinstance(label_choices_json, list):
            choices = [
                str(choice).strip()
                for choice in label_choices_json
                if str(choice).strip()
            ]
            if choices:
                return choices
    payload_label_choices = label_choices_from_payload(dict(payload))
    label_agnostic = [
        label
        for label in payload_label_choices
        if re.match(r"^(?:class|label)_\d+$", str(label or "").strip())
    ]
    sorted_labels = sorted(label_agnostic, key=lambda label: int(label.rsplit("_", 1)[1]))
    if (
        sorted_labels
        and TIME0_BASELINE_AGENT_FACING_LABEL in payload_label_choices
        and TIME0_BASELINE_AGENT_FACING_LABEL not in sorted_labels
    ):
        sorted_labels.append(TIME0_BASELINE_AGENT_FACING_LABEL)
    return sorted_labels


def _label_agnostic_internal_order(payload: Mapping[str, Any]) -> List[str]:
    environment_name = str(payload.get("environment_name") or "").strip()
    if not environment_name:
        return []
    try:
        labels = load_allowed_interventions(
            model_id=environment_name,
            models_root=_MODELS_ROOT,
        )
    except Exception:
        return []
    parameter_display_mapping = _parameter_display_mapping(payload)
    ordered_labels = sorted(
        (label for label in labels if str(label).strip()),
        key=lambda label: str(parameter_display_mapping.get(label, label)).strip(),
    )
    return [*ordered_labels, TIME0_BASELINE_LABEL]


def _internal_label_by_uuid(payload: Mapping[str, Any]) -> Dict[str, str]:
    ground_truth_information = payload.get("ground_truth_information")
    if not isinstance(ground_truth_information, Mapping):
        return {}
    interventions = ground_truth_information.get("interventions")
    if not isinstance(interventions, Mapping):
        return {}
    out: Dict[str, str] = {}
    for sample_uuid, info in interventions.items():
        uuid_text = str(sample_uuid or "").strip()
        if not uuid_text or not isinstance(info, Mapping):
            continue
        changed_parameter = str(info.get("changed_parameter") or "").strip()
        out[uuid_text] = changed_parameter or TIME0_BASELINE_LABEL
    return out


def _validated_sample_manifest(payload: Mapping[str, Any]) -> Dict[str, List[ManifestItem]]:
    cached = payload.get("_validated_sample_manifest")
    if isinstance(cached, dict):
        return cached  # type: ignore[return-value]
    raw_manifest = payload.get("_sample_manifest_payload")
    if not isinstance(raw_manifest, dict):
        raise ValueError("sample_manifest.json is required to resolve question samples")
    validated = validate_sample_manifest_payload(
        raw_manifest,
        num_labels=max(1, len(label_choices_from_payload(dict(payload)))),
        path=SAMPLE_MANIFEST_FILENAME,
    )
    if isinstance(payload, dict):
        payload["_validated_sample_manifest"] = validated
    return validated


def _question_manifest_item(payload: Mapping[str, Any], question: Mapping[str, Any]) -> ManifestItem:
    recipe_info = _question_recipe_info(question)
    shot_slug = str(recipe_info.get("shot_slug") or "").strip()
    test_set_slug = str(recipe_info.get("test_set_slug") or "").strip()
    question_seed = recipe_info.get("question_seed")
    if not shot_slug:
        raise KeyError("question.recipe_info.shot_slug is required to resolve question samples")
    if not test_set_slug:
        raise KeyError("question.recipe_info.test_set_slug is required to resolve question samples")
    try:
        seed = int(question_seed)
    except (TypeError, ValueError) as exc:
        raise KeyError("question.recipe_info.question_seed is required to resolve question samples") from exc
    matches = [
        item
        for item in _validated_sample_manifest(payload).get(shot_slug, [])
        if item.test_set_slug == test_set_slug and int(item.seed) == seed
    ]
    if not matches:
        raise KeyError(
            f"Could not resolve sample_manifest row for shot_slug={shot_slug!r} "
            f"test_set_slug={test_set_slug!r} seed={seed}"
        )
    if len(matches) != 1:
        raise KeyError(
            f"Found multiple sample_manifest rows for shot_slug={shot_slug!r} "
            f"test_set_slug={test_set_slug!r} seed={seed}"
        )
    return matches[0]


def question_sample_paths(
    payload: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    subset: Optional[str] = None,
) -> List[str]:
    subset_key = str(subset or "").strip().lower()
    if subset_key in {"train", "test"}:
        raw_paths = question.get(f"{subset_key}_samples")
        if isinstance(raw_paths, list):
            return [str(sample_path).strip() for sample_path in raw_paths if str(sample_path).strip()]
    elif not subset_key:
        explicit_paths = []
        saw_explicit_field = False
        for field_name in ("train_samples", "test_samples"):
            raw_paths = question.get(field_name)
            if isinstance(raw_paths, list):
                saw_explicit_field = True
                explicit_paths.extend(str(sample_path).strip() for sample_path in raw_paths if str(sample_path).strip())
        if saw_explicit_field:
            return explicit_paths

    manifest_item = _question_manifest_item(payload, question)
    if not subset_key:
        sample_ids = [*manifest_item.train_samples, *manifest_item.test_samples]
    elif subset_key == "train":
        sample_ids = list(manifest_item.train_samples)
    elif subset_key == "test":
        sample_ids = list(manifest_item.test_samples)
    elif subset_key == "other":
        sample_ids = list(manifest_item.other_samples)
    else:
        raise ValueError(f"Unsupported subset {subset!r}")
    return [f"dataframes/{str(sample_id).strip()}.parquet" for sample_id in sample_ids if str(sample_id).strip()]


def label_for_question_sample(
    payload: Mapping[str, Any],
    *,
    question: Mapping[str, Any],
    sample_path: str,
) -> str:
    internal_label = str(_internal_label_by_uuid(payload).get(Path(str(sample_path)).stem) or "").strip()
    if not internal_label:
        raise KeyError(f"Could not resolve ground-truth label for sample {sample_path!r}")
    if _question_textual_info(question) == "none":
        internal_order = _label_agnostic_internal_order(payload)
        label_choices = _label_agnostic_choices(payload, question)
        if internal_label in internal_order:
            idx = internal_order.index(internal_label)
            if idx < len(label_choices):
                return label_choices[idx]
        return internal_label
    if internal_label == TIME0_BASELINE_LABEL:
        return TIME0_BASELINE_AGENT_FACING_LABEL
    return _parameter_display_mapping(payload).get(internal_label, internal_label)


def ground_truth_by_path_from_payload(payload: Mapping[str, Any]) -> Dict[str, str]:
    existing = payload.get("ground_truth_by_path")
    if isinstance(existing, Mapping):
        return {
            str(path).strip(): str(label).strip()
            for path, label in existing.items()
            if str(path).strip() and str(label).strip()
        }
    questions = metadata_questions_by_id(dict(payload))
    out: Dict[str, str] = {}
    for question in questions.values():
        for sample_path in question_sample_paths(payload, question=question):
            path_text = str(sample_path).strip()
            if not path_text:
                continue
            label = label_for_question_sample(payload, question=question, sample_path=path_text)
            previous = out.get(path_text)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Conflicting labels for {path_text!r} across questions: {previous!r} vs {label!r}"
                )
            out[path_text] = label
    return out


__all__ = [
    "ground_truth_by_path_from_payload",
    "label_choices_from_payload",
    "label_for_question_sample",
    "load_metadata_payload",
    "metadata_questions_by_id",
    "metadata_questions_list",
    "question_sample_paths",
    "resolve_tsenv_payload_path",
]
