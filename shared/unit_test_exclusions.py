from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Set

_RUN_ID_RE = re.compile(r"\b[0-9a-f]{32}\b")
_EXCLUSIONS_REL_PATH = Path("shared/config/unit_test_exclusions.json")

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_unit_test_exclusions_payload(repo_root: Optional[Path] = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else _repo_root()
    path = root / _EXCLUSIONS_REL_PATH
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return {"fixtures": []}


def load_unit_test_excluded_run_ids(repo_root: Optional[Path] = None) -> Set[str]:
    payload = load_unit_test_exclusions_payload(repo_root=repo_root)
    return set(_RUN_ID_RE.findall(json.dumps(payload)))


def load_unit_test_excluded_baseline_spec_ids(
    repo_root: Optional[Path] = None,
) -> Set[str]:
    payload = load_unit_test_exclusions_payload(repo_root=repo_root)
    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list):
        return set()
    out: Set[str] = set()
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            continue
        baseline = str(fixture.get("baseline") or "").strip()
        if baseline:
            out.add(baseline)
    return out
