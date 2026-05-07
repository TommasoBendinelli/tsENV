from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from shared.run_artifacts import resolve_similarity_metrics_path

OVERLAY_SIMILARITY_MIN_DEFAULT = 0.99
OVERLAY_MAX_EUCLIDEAN_DISTANCE_DEFAULT = 0.01
SIMILARITY_RULES_REL_PATH = Path("web_model_explorer/config/similarity_rules.json")


def _to_finite_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def load_overlay_similarity_min(repo_root: Path) -> float:
    rules_path = repo_root / SIMILARITY_RULES_REL_PATH
    try:
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
    except Exception:
        return float(OVERLAY_SIMILARITY_MIN_DEFAULT)
    value = _to_finite_float(
        payload.get("overlay_min_similarity")
        if isinstance(payload, dict)
        else None
    )
    if value is None:
        return float(OVERLAY_SIMILARITY_MIN_DEFAULT)
    return float(value)


def is_overlaying_ui_rule(
    score: object,
    *,
    overlay_min_similarity: float,
    overlay_max_euclidean_distance: float = OVERLAY_MAX_EUCLIDEAN_DISTANCE_DEFAULT,
) -> bool:
    if not isinstance(score, dict):
        return False
    pearson = _to_finite_float(score.get("pearson_min"))
    cosine = _to_finite_float(score.get("cosine_min"))
    euclidean_distance = _to_finite_float(
        score.get("euclidean_distance_mean")
        if "euclidean_distance_mean" in score
        else score.get("euclidean_distance")
    )
    similarity_ok = bool(
        pearson is not None
        and cosine is not None
        and pearson >= float(overlay_min_similarity)
        and cosine >= float(overlay_min_similarity)
    )
    euclidean_ok = bool(
        euclidean_distance is not None
        and euclidean_distance <= float(overlay_max_euclidean_distance)
    )
    return bool(similarity_ok or euclidean_ok)


def load_similarity_intervention_map(
    model_dir: Path,
    *,
    runs_dir: Optional[Path] = None,
    runs_dir_name: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    path = resolve_similarity_metrics_path(
        model_dir,
        runs_dir=runs_dir,
        runs_dir_name=runs_dir_name,
    )
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    baselines = payload.get("baselines")
    if not isinstance(baselines, dict):
        return {}
    out: Dict[str, Dict[str, Dict[str, object]]] = {}
    for baseline_id, baseline_payload in baselines.items():
        if not isinstance(baseline_payload, dict):
            continue
        children = baseline_payload.get("children")
        if not isinstance(children, dict):
            continue
        cast_map: Dict[str, Dict[str, object]] = {}
        for run_id, iv_metrics in children.items():
            if isinstance(iv_metrics, dict):
                cast_map[str(run_id)] = iv_metrics
        out[str(baseline_id)] = cast_map
    return out


def is_skipped_reason(reason: Optional[str]) -> bool:
    text = str(reason or "").strip().lower()
    return text.startswith("skipped:")


def _detectability_status(value: object) -> str:
    if isinstance(value, dict):
        return str(
            value.get("detectability") or value.get("detectable") or ""
        ).strip().lower()
    return str(value or "").strip().lower()


def skip_reason_for_identical_baseline_or_time0(
    *,
    run_id: str,
    baseline_spec_id: str,
    similarity_by_baseline: Dict[str, Dict[str, Dict[str, object]]],
    overlay_min_similarity: float,
) -> Optional[str]:
    run_id = str(run_id or "").strip()
    baseline_spec_id = str(baseline_spec_id or "").strip()
    if not run_id or not baseline_spec_id:
        return None
    iv_metrics = similarity_by_baseline.get(baseline_spec_id, {}).get(run_id, {})
    detectability = iv_metrics.get("detectability") if isinstance(iv_metrics, dict) else None
    if not isinstance(detectability, dict):
        return None
    baseline_status = _detectability_status(detectability.get("vs_baseline"))
    time0_status = _detectability_status(detectability.get("vs_time0_baseline"))
    if baseline_status != "yes" or time0_status != "yes":
        return "skipped: child not detectable vs baseline_or_time0"
    return None


def resolve_similarity_skip_reason(
    *,
    run_id: str,
    baseline_spec_id: str,
    similarity_by_baseline: Dict[str, Dict[str, Dict[str, object]]],
    overlay_min_similarity: float,
) -> Optional[str]:
    return skip_reason_for_identical_baseline_or_time0(
        run_id=run_id,
        baseline_spec_id=baseline_spec_id,
        similarity_by_baseline=similarity_by_baseline,
        overlay_min_similarity=overlay_min_similarity,
    )
