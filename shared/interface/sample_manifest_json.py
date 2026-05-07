from __future__ import annotations

import functools
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "schemas" / "sample_manifest.schema.json"
)


class SampleManifestSchemaError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def _schema_validator() -> Draft7Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft7Validator(schema)


def validate_sample_manifest_schema(payload: Any) -> None:
    try:
        _schema_validator().validate(payload)
    except JsonSchemaValidationError as exc:
        loc = list(exc.path)
        raise SampleManifestSchemaError(
            f"sample_manifest.json schema validation failed at {loc}: {exc.message}"
        ) from exc


def compute_train_test_sample_hash(
    *,
    train_samples_hashes: List[str],
    test_samples_hashes: List[str],
) -> str:
    payload = {
        "train_samples_hashes": sorted(str(item).strip() for item in train_samples_hashes if str(item).strip()),
        "test_samples_hashes": sorted(str(item).strip() for item in test_samples_hashes if str(item).strip()),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


class Accuracy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    euclidean_knn: float
    euclidean_centroid: float
    correlation_nn: float


class ManifestShotSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_adversarial: Optional[bool]
    number_train_samples_per_class: int
    number_test_samples: int

    @property
    def train_samples_per_class(self) -> int:
        return self.number_train_samples_per_class

    @property
    def test_samples(self) -> int:
        return self.number_test_samples

    @field_validator("number_train_samples_per_class", "number_test_samples")
    @classmethod
    def _validate_non_negative_int(cls, value: int, info: Any) -> int:
        parsed = int(value)
        minimum = 1 if info.field_name == "number_test_samples" else 0
        if parsed < minimum:
            raise ValueError(f"shot_slug_recipe.{info.field_name} must be >= {minimum}")
        return parsed

    @model_validator(mode="after")
    def _validate_nullable_adversarial_contract(self) -> "ManifestShotSelection":
        if self.number_train_samples_per_class == 0 and self.is_adversarial is not None:
            raise ValueError(
                "shot_slug_recipe.is_adversarial must be null when number_train_samples_per_class == 0"
            )
        if self.number_train_samples_per_class > 0 and self.is_adversarial is None:
            raise ValueError(
                "shot_slug_recipe.is_adversarial must be true or false when number_train_samples_per_class > 0"
            )
        return self


class ManifestItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_samples_baselines: List[str]
    train_samples: List[str]
    train_samples_hashes: List[str]
    test_samples_baselines: List[str]
    test_samples: List[str]
    test_samples_hashes: List[str]
    other_samples: List[str]
    seed: int
    test_set_slug: str
    shot_slug_recipe: ManifestShotSelection
    accuracy_with_baselines: Accuracy
    accuracy_with_baselines_all_samples: Accuracy
    train_test_sample_hash: str

    @field_validator(
        "train_samples_baselines",
        "train_samples",
        "train_samples_hashes",
        "test_samples_baselines",
        "test_samples",
        "test_samples_hashes",
        "other_samples",
        mode="before",
    )
    @classmethod
    def _require_str_list(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            raise TypeError("Expected a list of strings")
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise TypeError("Expected a list of non-empty strings")
        return normalized

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        return int(value)

    @field_validator("test_set_slug")
    @classmethod
    def _validate_test_set_slug(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("test_set_slug must be non-empty")
        return text

    @field_validator("train_test_sample_hash")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("train_test_sample_hash must be non-empty")
        return text


def _require_sorted_list(
    *,
    values: List[str],
    field_name: str,
    shot_slug: str,
    item: ManifestItem,
    path: str,
) -> None:
    if values != sorted(values):
        raise ValueError(
            f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
            f"{field_name} must be sorted"
        )


def _require_disjoint(
    *,
    left: set[str],
    right: set[str],
    left_name: str,
    right_name: str,
    shot_slug: str,
    item: ManifestItem,
    path: str,
) -> None:
    overlap = sorted(left & right)
    if overlap:
        raise ValueError(
            f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
            f"has overlapping {left_name}/{right_name}: {overlap!r}"
        )


def validate_sample_manifest_payload(
    payload: Any,
    *,
    num_labels: int,
    label_by_sample_id: Optional[Mapping[str, str]] = None,
    path: str = "sample_manifest.json",
) -> Dict[str, List[ManifestItem]]:
    try:
        validate_sample_manifest_schema(payload)
    except SampleManifestSchemaError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid sample_manifest schema at {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sample_manifest schema at {path}: expected an object keyed by shot_slug")

    label_count = int(num_labels)
    if label_count < 1:
        raise ValueError("num_labels must be >= 1 for sample_manifest validation")

    validated: Dict[str, List[ManifestItem]] = {}
    for shot_slug, raw_items in payload.items():
        shot_key = str(shot_slug or "").strip()
        if not shot_key:
            raise ValueError(f"Invalid sample_manifest at {path}: shot_slug keys must be non-empty strings")
        if not isinstance(raw_items, list):
            raise ValueError(
                f"Invalid sample_manifest at {path}: shot_slug={shot_key!r} must map to a list of rows"
            )
        try:
            items = [ManifestItem.model_validate(item) for item in raw_items]
        except ValidationError as exc:
            raise ValueError(f"Invalid sample_manifest schema at {path}: {exc}") from exc
        if not items:
            raise ValueError(
                f"Invalid sample_manifest at {path}: shot_slug={shot_key!r} must contain at least one row"
            )
        validated[shot_key] = items

    for shot_slug, items in validated.items():
        seen_keys: set[tuple[str, int]] = set()
        train_seed_by_uuid: Dict[str, int] = {}
        for item in items:
            item_key = (item.test_set_slug, int(item.seed))
            if item_key in seen_keys:
                raise ValueError(
                    f"Invalid sample_manifest at {path}: duplicate row for shot_slug={shot_slug!r} "
                    f"test_set_slug={item.test_set_slug!r} seed={item.seed}"
                )
            seen_keys.add(item_key)

            if len(item.test_samples) != item.shot_slug_recipe.test_samples:
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    f"has {len(item.test_samples)} test_samples but expected {item.shot_slug_recipe.test_samples}"
                )
            if len(item.test_samples_baselines) != len(item.test_samples):
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    "must provide one test_samples_baselines entry per test sample"
                )
            if len(item.test_samples_hashes) != len(item.test_samples):
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    "must provide one test_samples_hashes entry per test sample"
                )
            expected_train_count = item.shot_slug_recipe.train_samples_per_class * label_count
            if len(item.train_samples) != expected_train_count:
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    f"has {len(item.train_samples)} train_samples but expected {expected_train_count}"
                )
            if len(item.train_samples_baselines) != len(item.train_samples):
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    "must provide one train_samples_baselines entry per train sample"
                )
            if len(item.train_samples_hashes) != len(item.train_samples):
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    "must provide one train_samples_hashes entry per train sample"
                )
            _require_sorted_list(
                values=list(item.train_samples),
                field_name="train_samples",
                shot_slug=shot_slug,
                item=item,
                path=path,
            )
            _require_sorted_list(
                values=list(item.test_samples),
                field_name="test_samples",
                shot_slug=shot_slug,
                item=item,
                path=path,
            )
            _require_sorted_list(
                values=list(item.other_samples),
                field_name="other_samples",
                shot_slug=shot_slug,
                item=item,
                path=path,
            )
            if len(set(item.test_samples_baselines)) != len(item.test_samples_baselines):
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    "test_samples_baselines must be unique"
                )
            _require_disjoint(
                left=set(item.train_samples),
                right=set(item.test_samples),
                left_name="train_samples",
                right_name="test_samples",
                shot_slug=shot_slug,
                item=item,
                path=path,
            )
            _require_disjoint(
                left=set(item.train_samples_baselines),
                right=set(item.test_samples_baselines),
                left_name="train_samples_baselines",
                right_name="test_samples_baselines",
                shot_slug=shot_slug,
                item=item,
                path=path,
            )
            for sample_id in item.train_samples:
                previous_seed = train_seed_by_uuid.get(sample_id)
                if previous_seed is not None and previous_seed != int(item.seed):
                    raise ValueError(
                        f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} "
                        f"train sample {sample_id!r} repeats across seed buckets "
                        f"{previous_seed} and {item.seed}"
                    )
                train_seed_by_uuid[sample_id] = int(item.seed)
            selected_sample_ids = set(item.train_samples) | set(item.test_samples)
            overlapping_other_samples = sorted(selected_sample_ids & set(item.other_samples))
            if overlapping_other_samples:
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    f"has other_samples overlapping train/test samples: {overlapping_other_samples!r}"
                )
            computed_hash = compute_train_test_sample_hash(
                train_samples_hashes=list(item.train_samples_hashes),
                test_samples_hashes=list(item.test_samples_hashes),
            )
            if item.train_test_sample_hash != computed_hash:
                raise ValueError(
                    f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} seed={item.seed} "
                    "has a train_test_sample_hash that does not match train_samples_hashes/test_samples_hashes"
                )
            if label_by_sample_id is not None:
                counts = Counter()
                for sample_id in item.test_samples:
                    label = str(label_by_sample_id.get(sample_id) or "").strip()
                    if not label:
                        raise ValueError(
                            f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} "
                            f"seed={item.seed} has unmapped test sample {sample_id!r}"
                        )
                    counts[label] += 1
                if len(counts) > label_count:
                    raise ValueError(
                        f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} "
                        f"seed={item.seed} has {len(counts)} test labels but expected at most {label_count}"
                    )
                balanced_counts = list(counts.values()) + ([0] * (label_count - len(counts)))
                if balanced_counts and (max(balanced_counts) - min(balanced_counts) > 1):
                    raise ValueError(
                        f"Invalid sample_manifest at {path}: shot_slug={shot_slug!r} "
                        f"seed={item.seed} has unbalanced test label counts {dict(sorted(counts.items()))}"
                    )

    test_panel_by_key: Dict[tuple[str, int], tuple[List[str], List[str], List[str]]] = {}
    test_seed_by_uuid_by_test_set: Dict[str, Dict[str, int]] = {}
    for shot_slug, items in validated.items():
        for item in items:
            key = (item.test_set_slug, int(item.seed))
            existing = test_panel_by_key.get(key)
            current = (
                list(item.test_samples),
                list(item.test_samples_baselines),
                list(item.test_samples_hashes),
            )
            if existing is None:
                test_panel_by_key[key] = current
            elif existing != current:
                raise ValueError(
                    f"Invalid sample_manifest at {path}: entries sharing test_set_slug={item.test_set_slug!r} "
                    f"seed={item.seed} must have identical test_samples, test_samples_baselines, "
                    "and test_samples_hashes"
                )
            test_seed_by_uuid = test_seed_by_uuid_by_test_set.setdefault(item.test_set_slug, {})
            for sample_id in item.test_samples:
                previous_seed = test_seed_by_uuid.get(sample_id)
                if previous_seed is not None and previous_seed != int(item.seed):
                    raise ValueError(
                        f"Invalid sample_manifest at {path}: test_set_slug={item.test_set_slug!r} "
                        f"test sample {sample_id!r} repeats across seed buckets "
                        f"{previous_seed} and {item.seed}"
                    )
                test_seed_by_uuid[sample_id] = int(item.seed)

    return validated


__all__ = [
    "Accuracy",
    "ManifestItem",
    "ManifestShotSelection",
    "SampleManifestSchemaError",
    "compute_train_test_sample_hash",
    "validate_sample_manifest_payload",
    "validate_sample_manifest_schema",
]
