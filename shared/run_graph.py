from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

RUN_KINDS = {"baseline", "intervention", "time0_baseline"}
EDGE_TYPES = {"baseline_to_intervention", "intervention_to_time0_baseline"}


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def normalize_identity_numbers(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        if value == 0.0:
            return 0
        if value.is_integer():
            return int(value)
        return float(value)
    if isinstance(value, (list, tuple)):
        return [normalize_identity_numbers(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): normalize_identity_numbers(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize_identity_numbers(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    )


def hash_object(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:32]


def recipe_hash(recipe: Mapping[str, Any]) -> str:
    return hash_object(dict(recipe))


def family_id_for_recipe(recipe: Mapping[str, Any]) -> str:
    payload = {
        "model": recipe.get("model"),
        "baseline_parameters": recipe.get("baseline_parameters") or {},
    }
    return f"fam_{hash_object(payload)}"


@dataclass(frozen=True)
class RunGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

    @property
    def nodes_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {str(node["run_id"]): node for node in self.nodes}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")


def load_run_graph(plan_dir: Path) -> RunGraph:
    graph = RunGraph(
        nodes=read_jsonl(plan_dir / "run_nodes.jsonl"),
        edges=read_jsonl(plan_dir / "run_edges.jsonl"),
    )
    validate_run_graph(graph)
    return graph


def _require_node_field(node: Mapping[str, Any], field: str) -> Any:
    if field not in node:
        raise ValueError(f"run node is missing required field {field!r}")
    return node[field]


def validate_run_graph(graph: RunGraph) -> None:
    if not graph.nodes:
        raise ValueError("run_nodes.jsonl must be non-empty")
    seen: set[str] = set()
    for node in graph.nodes:
        run_id = str(_require_node_field(node, "run_id") or "").strip()
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if run_id in seen:
            raise ValueError(f"duplicate run_id {run_id!r}")
        seen.add(run_id)
        kind = str(_require_node_field(node, "kind") or "").strip()
        if kind not in RUN_KINDS:
            raise ValueError(f"run_id {run_id!r} has invalid kind {kind!r}")
        recipe = _require_node_field(node, "recipe")
        if not isinstance(recipe, Mapping):
            raise ValueError(f"run_id {run_id!r} recipe must be an object")
        baseline_parameters = recipe.get("baseline_parameters")
        if not isinstance(baseline_parameters, Mapping):
            raise ValueError(
                f"run_id {run_id!r} recipe.baseline_parameters must be an object"
            )
        intervention = recipe.get("intervention")
        if not isinstance(intervention, Mapping):
            raise ValueError(f"run_id {run_id!r} recipe.intervention must be an object")
        expected_hash = recipe_hash(recipe)
        if str(node.get("recipe_hash") or "").strip() != expected_hash:
            raise ValueError(f"run_id {run_id!r} recipe_hash mismatch")
    for edge in graph.edges:
        edge_type = str(edge.get("edge_type") or "").strip()
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"invalid edge_type {edge_type!r}")
        source = str(edge.get("source_run_id") or "").strip()
        target = str(edge.get("target_run_id") or "").strip()
        if source not in seen or target not in seen:
            raise ValueError(
                f"edge {edge_type!r} points to missing run_id {source!r}->{target!r}"
            )


def write_run_graph(plan_dir: Path, graph: RunGraph) -> None:
    validate_run_graph(graph)
    plan_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(plan_dir / "run_nodes.jsonl", graph.nodes)
    write_jsonl(plan_dir / "run_edges.jsonl", graph.edges)


def intervention_time_by_time0_run_id(edges: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for edge in edges:
        if str(edge.get("edge_type") or "") != "intervention_to_time0_baseline":
            continue
        target = str(edge.get("target_run_id") or "").strip()
        metadata = edge.get("metadata")
        if not target or not isinstance(metadata, Mapping):
            continue
        try:
            out[target] = float(metadata.get("intervention_time"))
        except Exception:
            continue
    return out


def runtime_record_from_run_graph(graph: RunGraph) -> Dict[str, Dict[str, Any]]:
    expected: Dict[str, Dict[str, Any]] = {}
    for node in graph.nodes:
        run_id = str(node["run_id"])
        kind = str(node["kind"])
        recipe = node["recipe"]
        intervention = recipe.get("intervention") if isinstance(recipe, Mapping) else {}
        parameter = ""
        if isinstance(intervention, Mapping) and intervention.get("parameter") is not None:
            parameter = str(intervention.get("parameter") or "").strip()
        if kind == "baseline":
            class_internal = "no_parameter_change"
            class_agent_facing_name = "no parameter changed"
        elif kind == "intervention":
            class_internal = parameter
            class_agent_facing_name = parameter
        else:
            class_internal = ""
            class_agent_facing_name = ""
        metadata = node.get("metadata")
        policy_id = ""
        if isinstance(metadata, Mapping):
            policy_id = str(metadata.get("policy_id") or "").strip()
        expected[run_id] = {
            "parameters_hash": str(node["recipe_hash"]),
            "recipe_hash": str(node["recipe_hash"]),
            "run_type": kind,
            "family_id": str(node.get("family_id") or ""),
            "last_policy_id": policy_id,
            "class_internal": class_internal,
            "class_agent_facing_name": class_agent_facing_name,
            "status": "not_run",
        }
    return expected


def selected_run_ids(graph: RunGraph) -> set[str]:
    return {str(node["run_id"]) for node in graph.nodes}
