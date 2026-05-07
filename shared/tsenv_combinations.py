from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ALL_POSSIBLE_COMBINATIONS_PATH = (
    Path(__file__).resolve().parent.parent / "all_possible_combinations.csv"
)
TIME0_BASELINE_LABEL = "no_parameter_change"
TIME0_BASELINE_AGENT_FACING_LABEL = "no parameter change"
_ALLOWED_REQUEST_TYPES = {"direct", "code", "open-ended"}
_ALLOWED_DESC_LEVEL = {"none", "low", "high"}
_ALLOWED_NOISE = {"none", "low", "high"}


@dataclass(frozen=True)
class CombinationRow:
    row_slug: str
    shot_slug: str
    test_set_slug: str
    type_of_request: str
    desc_level: str
    noise_level: str
    number_train_samples_per_class: int
    number_test_samples: int
    is_adversarial: Optional[bool]
    seeds: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.number_train_samples_per_class == 0 and self.is_adversarial is not None:
            raise ValueError("is_adversarial must be None when number_train_samples_per_class == 0")
        if self.number_train_samples_per_class > 0 and self.is_adversarial is None:
            raise ValueError("is_adversarial must be true or false when number_train_samples_per_class > 0")

    @property
    def train_samples_per_class(self) -> int:
        return self.number_train_samples_per_class

    @property
    def test_samples(self) -> int:
        return self.number_test_samples


@dataclass(frozen=True)
class ShotSelection:
    shot_slug: str
    test_set_slug: str
    is_adversarial: Optional[bool]
    number_train_samples_per_class: int
    number_test_samples: int
    seeds: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.number_train_samples_per_class == 0 and self.is_adversarial is not None:
            raise ValueError("is_adversarial must be None when number_train_samples_per_class == 0")
        if self.number_train_samples_per_class > 0 and self.is_adversarial is None:
            raise ValueError("is_adversarial must be true or false when number_train_samples_per_class > 0")

    @property
    def train_samples_per_class(self) -> int:
        return self.number_train_samples_per_class

    @property
    def test_samples(self) -> int:
        return self.number_test_samples


def _normalize_slug(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _normalize_choice(value: Any, *, field_name: str, allowed: Sequence[str]) -> str:
    text = str(value or "").strip().lower()
    if text not in set(allowed):
        raise ValueError(f"{field_name} must be one of {list(allowed)!r}; got {value!r}")
    return text


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise ValueError(f"{field_name} must be a boolean-like value; got {value!r}")


def _parse_nullable_bool(value: Any, *, field_name: str) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return None
    return _parse_bool(value, field_name=field_name)


def _parse_int(value: Any, *, field_name: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer; got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}; got {parsed}")
    return parsed


def _parse_seed_list(value: Any) -> Tuple[int, ...]:
    if isinstance(value, list):
        values = value
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("seeds must be a non-empty list<integer>")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [item.strip() for item in text.split(",") if item.strip()]
        values = parsed
    if not isinstance(values, list) or not values:
        raise ValueError("seeds must be a non-empty list<integer>")
    seeds: List[int] = []
    seen: set[int] = set()
    for item in values:
        seed = _parse_int(item, field_name="seeds", minimum=0)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return tuple(seeds)


def _normalize_row(raw: Dict[str, Any], *, row_index: int) -> CombinationRow:
    row_slug = _normalize_slug(raw.get("row_slug"), field_name=f"row[{row_index}].row_slug")
    shot_slug = _normalize_slug(raw.get("shot_slug"), field_name=f"row[{row_index}].shot_slug")
    test_set_slug = _normalize_slug(raw.get("test_set_slug"), field_name=f"row[{row_index}].test_set_slug")
    if not row_slug.startswith(shot_slug):
        raise ValueError(
            f"row[{row_index}].row_slug={row_slug!r} must start with shot_slug={shot_slug!r}"
        )
    number_train_samples_per_class = _parse_int(
        raw.get("number_train_samples_per_class"),
        field_name=f"row[{row_index}].number_train_samples_per_class",
        minimum=0,
    )
    is_adversarial = _parse_nullable_bool(
        raw.get("is_adversarial"),
        field_name=f"row[{row_index}].is_adversarial",
    )
    if number_train_samples_per_class == 0 and is_adversarial is not None:
        raise ValueError(
            f"row[{row_index}].is_adversarial must be null when number_train_samples_per_class == 0"
        )
    if number_train_samples_per_class > 0 and is_adversarial is None:
        raise ValueError(
            f"row[{row_index}].is_adversarial must be true or false when number_train_samples_per_class > 0"
        )
    return CombinationRow(
        row_slug=row_slug,
        shot_slug=shot_slug,
        test_set_slug=test_set_slug,
        type_of_request=_normalize_choice(
            raw.get("type_of_request"),
            field_name=f"row[{row_index}].type_of_request",
            allowed=sorted(_ALLOWED_REQUEST_TYPES),
        ),
        desc_level=_normalize_choice(
            raw.get("desc_level"),
            field_name=f"row[{row_index}].desc_level",
            allowed=sorted(_ALLOWED_DESC_LEVEL),
        ),
        noise_level=_normalize_choice(
            raw.get("noise_level"),
            field_name=f"row[{row_index}].noise_level",
            allowed=sorted(_ALLOWED_NOISE),
        ),
        number_train_samples_per_class=number_train_samples_per_class,
        number_test_samples=_parse_int(
            raw.get("number_test_samples"),
            field_name=f"row[{row_index}].number_test_samples",
            minimum=1,
        ),
        is_adversarial=is_adversarial,
        seeds=_parse_seed_list(raw.get("seeds")),
    )


def load_combination_rows(
    path: Path,
    *,
    row_slugs: Optional[Sequence[str]] = None,
) -> List[CombinationRow]:
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing all_possible_combinations.csv at {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [_normalize_row(dict(row), row_index=idx) for idx, row in enumerate(reader, start=2)]
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    requested = [str(value or "").strip() for value in (row_slugs or []) if str(value or "").strip()]
    if requested:
        requested_set = set(requested)
        rows = [row for row in rows if row.row_slug in requested_set]
        missing = [row_slug for row_slug in requested if row_slug not in {row.row_slug for row in rows}]
        if missing:
            raise ValueError(
                f"Requested row_slug values were not found in {csv_path}: {sorted(missing)!r}"
            )
    seen_row_slugs: set[str] = set()
    for row in rows:
        if row.row_slug in seen_row_slugs:
            raise ValueError(f"Duplicate row_slug={row.row_slug!r} in {csv_path}")
        seen_row_slugs.add(row.row_slug)
    return rows


def group_shot_selections(rows: Sequence[CombinationRow]) -> List[ShotSelection]:
    grouped: Dict[str, Dict[str, Any]] = {}
    ordered_shot_slugs: List[str] = []
    for row in rows:
        group_key = f"{row.shot_slug}\0{row.test_set_slug}"
        existing = grouped.get(group_key)
        if existing is None:
            grouped[group_key] = {
                "shot_slug": row.shot_slug,
                "test_set_slug": row.test_set_slug,
                "is_adversarial": row.is_adversarial,
                "number_train_samples_per_class": int(row.number_train_samples_per_class),
                "number_test_samples": int(row.number_test_samples),
                "seeds": list(row.seeds),
            }
            ordered_shot_slugs.append(group_key)
            continue
        if (
            existing["is_adversarial"] != row.is_adversarial
            or int(existing["number_train_samples_per_class"]) != int(row.number_train_samples_per_class)
            or int(existing["number_test_samples"]) != int(row.number_test_samples)
        ):
            raise ValueError(
                f"shot_slug={row.shot_slug!r} test_set_slug={row.test_set_slug!r} maps to conflicting train/test/adversarial settings"
            )
        for seed in row.seeds:
            if int(seed) not in existing["seeds"]:
                existing["seeds"].append(int(seed))
    return [
        ShotSelection(
            shot_slug=str(grouped[group_key]["shot_slug"]),
            test_set_slug=str(grouped[group_key]["test_set_slug"]),
            is_adversarial=grouped[group_key]["is_adversarial"],
            number_train_samples_per_class=int(grouped[group_key]["number_train_samples_per_class"]),
            number_test_samples=int(grouped[group_key]["number_test_samples"]),
            seeds=tuple(int(seed) for seed in grouped[group_key]["seeds"]),
        )
        for group_key in ordered_shot_slugs
    ]


def all_requested_seeds(rows: Sequence[CombinationRow]) -> List[int]:
    seen: set[int] = set()
    ordered: List[int] = []
    for row in rows:
        for seed in row.seeds:
            if seed in seen:
                continue
            seen.add(seed)
            ordered.append(int(seed))
    return ordered


__all__ = [
    "ALL_POSSIBLE_COMBINATIONS_PATH",
    "CombinationRow",
    "ShotSelection",
    "TIME0_BASELINE_LABEL",
    "all_requested_seeds",
    "group_shot_selections",
    "load_combination_rows",
]
