from __future__ import annotations

from typing import Any, Dict, Mapping

from shared.interface.model_record_json import (
    compute_child_parameters_hash,
    compute_parameters_hash,
    compute_time0_baseline_hash,
)

_PARENT_RESERVED_PARAMETER_KEYS = (
    "intervention_time",
    "end_time_input_s",
    "sampling_rate_hz",
)
_RUNTIME_ENTRY_RESERVED_KEYS = {
    "parameters_hash",
    "run_type",
    "class_internal",
    "class_agent_facing_name",
    "status",
    "timestamp",
    "end_time_simulation",
    "error",
}


def _iter_children(specs: Mapping[str, Any]):
    for baseline_uuid, raw_parent in specs.items():
        if not isinstance(raw_parent, dict):
            continue
        children = raw_parent.get("children")
        if not isinstance(children, dict):
            continue
        yield str(baseline_uuid), raw_parent, children


def _is_runnable_child(raw_child: Any) -> bool:
    if not isinstance(raw_child, dict):
        return False
    parameters = raw_child.get("parameters")
    if not isinstance(parameters, dict) or len(parameters) != 1:
        return False
    return next(iter(parameters.values()), None) is not None


def _child_intervention_time(raw_child: Any) -> float | None:
    if not isinstance(raw_child, dict):
        return None
    try:
        return float(raw_child.get("intervention_time"))
    except Exception:
        return None


def _common_child_intervention_time(children: Mapping[str, Any]) -> float | None:
    values = [
        value
        for raw_child in children.values()
        for value in [_child_intervention_time(raw_child)]
        if value is not None
    ]
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values) else None


def expected_run_ids(specs: Mapping[str, Any]) -> set[str]:
    return set(build_expected_runtime_model_record(specs).keys())


def build_expected_runtime_model_record(specs: Mapping[str, Any]) -> Dict[str, Any]:
    expected: Dict[str, Any] = {}
    for baseline_uuid, raw_parent, children in _iter_children(specs):
        baseline_parameters = dict(raw_parent.get("baseline_parameters") or {})
        baseline_hash = compute_parameters_hash(parameters=baseline_parameters)
        expected[baseline_uuid] = {
            "parameters_hash": baseline_hash,
            "run_type": "baseline",
            "class_internal": "no_parameter_change",
            "class_agent_facing_name": "no parameter changed",
            "status": "not_run",
        }
        for child_uuid, raw_child in children.items():
            if not _is_runnable_child(raw_child):
                continue
            child_parameters = dict(raw_child.get("parameters") or {})
            child_hash = compute_child_parameters_hash(
                parent_parameters_hash=baseline_hash,
                child_parameters=child_parameters,
                intervention_time=raw_child.get("intervention_time"),
            )
            parameter = str(next(iter(child_parameters.keys()), "")).strip()
            child_id = str(child_uuid)
            expected[child_id] = {
                "parameters_hash": child_hash,
                "run_type": "intervention",
                "class_internal": parameter,
                "class_agent_facing_name": parameter,
                "status": "not_run",
            }
            time0_uuid = str(raw_child.get("time0_baseline_uuid") or "").strip()
            if time0_uuid:
                expected[time0_uuid] = {
                    "parameters_hash": compute_time0_baseline_hash(
                        child_parameters_hash=child_hash
                    ),
                    "run_type": "time0_baseline",
                    "class_internal": "",
                    "class_agent_facing_name": "",
                    "status": "not_run",
                }
    return expected


def reconcile_runtime_model_record(
    *,
    specs: Mapping[str, Any],
    runtime_map: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = build_expected_runtime_model_record(specs)
    reconciled: Dict[str, Any] = {}
    for run_id, expected_entry in expected.items():
        existing_entry = runtime_map.get(run_id)
        status = "not_run"
        runtime_fields: Dict[str, Any] = {}
        if isinstance(existing_entry, dict):
            status = str(existing_entry.get("status") or "not_run")
            runtime_fields = {
                key: value
                for key, value in {
                    "timestamp": existing_entry.get("timestamp"),
                    "end_time_simulation": existing_entry.get("end_time_simulation"),
                    "error": existing_entry.get("error"),
                }.items()
                if value not in (None, "", [])
            }
            runtime_fields.update(
                {
                    str(key): value
                    for key, value in existing_entry.items()
                    if key not in _RUNTIME_ENTRY_RESERVED_KEYS
                }
            )
        payload = dict(expected_entry)
        payload["status"] = status
        payload.update(runtime_fields)
        reconciled[run_id] = payload
    return reconciled


def build_model_record_registry(
    *,
    model_id: str,
    specs: Mapping[str, Any],
    runtime_map: Mapping[str, Any],
    experiment_config: Any,
) -> Dict[str, Any]:
    baselines: list[Dict[str, Any]] = []
    sampling_rate_hz = float(getattr(experiment_config, "sampling_rate_hz"))
    end_time_input_s = float(getattr(experiment_config, "end_time_input_s"))
    for baseline_uuid, raw_parent, children in _iter_children(specs):
        baseline_parameters_full = dict(raw_parent.get("baseline_parameters") or {})
        baseline_intervention_time = _common_child_intervention_time(children)
        for key in _PARENT_RESERVED_PARAMETER_KEYS:
            baseline_parameters_full.pop(key, None)
        parent_runtime = (
            runtime_map.get(baseline_uuid)
            if isinstance(runtime_map.get(baseline_uuid), dict)
            else {}
        )
        interventions: list[Dict[str, Any]] = []
        for child_uuid, raw_child in children.items():
            if not _is_runnable_child(raw_child):
                continue
            child_runtime = (
                runtime_map.get(str(child_uuid))
                if isinstance(runtime_map.get(str(child_uuid)), dict)
                else {}
            )
            child_parameters = dict(raw_child.get("parameters") or {})
            parameter = str(next(iter(child_parameters.keys()), "")).strip()
            set_value = next(iter(child_parameters.values()), None)
            intervention_time = _child_intervention_time(raw_child)
            time0_uuid = str(raw_child.get("time0_baseline_uuid") or "").strip()
            time0_runtime = (
                runtime_map.get(time0_uuid)
                if time0_uuid and isinstance(runtime_map.get(time0_uuid), dict)
                else {}
            )
            interventions.append(
                {
                    "name": str(child_uuid),
                    "parent_id": baseline_uuid,
                    "depth": 1,
                    "intervention_time": intervention_time,
                    "parameter": parameter,
                    "set_value": set_value,
                    "intervention_uuid": str(child_uuid),
                    "time0_baseline_uuid": time0_uuid or None,
                    "variable": parameter,
                    "value": set_value,
                    "end_time_input_s": end_time_input_s,
                    "status": child_runtime.get("status", "not_run"),
                    **{
                        key: child_runtime.get(key)
                        for key in (
                            "timestamp",
                            "end_time_simulation",
                            "error",
                            "noise",
                            "classification",
                        )
                        if key in child_runtime
                    },
                    "time0_baseline_status": time0_runtime.get("status", "not_run"),
                    "time0_baseline_timestamp": str(
                        time0_runtime.get("timestamp") or ""
                    ),
                    "time0_baseline_end_time_simulation": time0_runtime.get(
                        "end_time_simulation"
                    ),
                    "time0_baseline_error": time0_runtime.get("error"),
                }
            )
        baselines.append(
            {
                "run_id": baseline_uuid,
                "baseline_uuid": baseline_uuid,
                "parent_id": None,
                "parameters": baseline_parameters_full,
                "intervention_time": baseline_intervention_time,
                "baseline_intervention_time": baseline_intervention_time,
                "interventions": interventions,
                "sampling_rate_hz": sampling_rate_hz,
                "end_time_input_s": end_time_input_s,
                "status": parent_runtime.get("status", "not_run"),
                **{
                    key: parent_runtime.get(key)
                    for key in (
                        "timestamp",
                        "end_time_simulation",
                        "error",
                        "noise",
                        "classification",
                    )
                    if key in parent_runtime
                },
            }
        )
    return {
        "version": 1,
        "model_id": str(model_id or "").strip(),
        "metadata": {},
        "baselines": baselines,
    }


__all__ = [
    "build_expected_runtime_model_record",
    "build_model_record_registry",
    "expected_run_ids",
    "reconcile_runtime_model_record",
]
