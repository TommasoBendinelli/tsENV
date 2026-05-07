from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
from typing import Any, Optional

_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def normalize_run_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    run_id = str(value).strip()
    if not run_id:
        return None
    return run_id if _RUN_ID_RE.match(run_id) else None


def infer_run_id_from_child_path(child_path: Path) -> Optional[str]:
    return normalize_run_id(child_path.parent.name)


def result_to_pretty_json(result: Any) -> str:
    return json.dumps(asdict(result), indent=2)
