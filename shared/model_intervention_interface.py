from __future__ import annotations

import json
from pathlib import Path

INTERFACE_FILENAME = "experiment_config.json"
INTERFACE_KEY = "exposed_variables.parameters"


def _derive_allowed_interventions(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    exposed_variables = payload.get("exposed_variables")
    if not isinstance(exposed_variables, dict):
        return []
    parameters = exposed_variables.get("parameters")
    if not isinstance(parameters, dict):
        return []
    ordered = sorted(
        (
            str(parameter_name).strip()
            for parameter_name in parameters.keys()
            if str(parameter_name).strip()
        ),
        key=str.casefold,
    )
    return ordered


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _models_root(models_root: Path | None = None) -> Path:
    if models_root is not None:
        return Path(models_root)
    return _repo_root() / "models" / "simulink"


def list_models_with_allowed_interventions(models_root: Path | None = None) -> list[str]:
    root = _models_root(models_root)
    out: list[str] = []
    for p in sorted(root.glob(f"*/{INTERFACE_FILENAME}")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if _derive_allowed_interventions(payload):
            out.append(p.parent.name)
    return out


def load_allowed_interventions(
    *,
    model_id: str,
    models_root: Path | None = None,
) -> list[str]:
    root = _models_root(models_root)
    p = root / str(model_id) / INTERFACE_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Missing {INTERFACE_FILENAME} for model '{model_id}': {p}")

    payload = json.loads(p.read_text(encoding="utf-8"))
    params = _derive_allowed_interventions(payload)
    if not params:
        raise ValueError(f"Invalid '{INTERFACE_KEY}' in {p}")
    return list(params)
