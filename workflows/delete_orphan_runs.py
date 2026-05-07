#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, List, Optional, Set

import click

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.benchmark_utils import ALLOWED_TSENV_MODELS  # noqa: E402
from shared.interface.model_record_json import load_model_record_json, load_model_run_specs_json  # noqa: E402
from shared.model_lock import ModelLockError, model_lock  # noqa: E402
from shared.run_artifacts import resolve_model_record_path, resolve_runs_root  # noqa: E402


def _is_hash_dir(name: str) -> bool:
    if len(name) != 32:
        return False
    return all(c in "0123456789abcdef" for c in name)


def _referenced_run_ids(
    specs: Mapping[str, Any],
    model_record: Dict[str, Any],
) -> Set[str]:
    referenced: Set[str] = set()
    for baseline_uuid, baseline in specs.items():
        if not isinstance(baseline, dict):
            continue
        run_id = str(baseline_uuid or "").strip()
        if run_id:
            referenced.add(run_id)
        children = baseline.get("children")
        if not isinstance(children, dict):
            continue
        for child_uuid, raw_child in children.items():
            if not isinstance(raw_child, dict):
                continue
            name = str(child_uuid or "").strip()
            if name:
                referenced.add(name)
            time0 = str(raw_child.get("time0_baseline_uuid") or "").strip()
            if time0:
                referenced.add(time0)
    for run_id in model_record.keys():
        normalized = str(run_id or "").strip()
        if normalized:
            referenced.add(normalized)
    return referenced


def _list_hash_dirs(path: Path) -> List[str]:
    if not path.exists():
        return []
    return sorted(
        [
            p.name
            for p in path.iterdir()
            if p.is_dir() and not p.name.startswith(".") and _is_hash_dir(p.name)
        ]
    )


def _delete_tree_or_link(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.exists():
        shutil.rmtree(path)


@click.command()
@click.argument("model", required=False)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview orphan targets without deleting them.",
)
@click.option(
    "--apply",
    "apply_flag",
    is_flag=True,
    default=False,
    help="Deprecated compatibility flag. Apply is now default unless --dry-run is set.",
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
def cli(
    model: Optional[str],
    dry_run: bool,
    apply_flag: bool,
    runs_dir: Optional[Path],
    runs_dir_name: Optional[str],
) -> None:
    """Delete orphan folders under the configured runs root for a model.

    Orphan = a hashed run directory that is not referenced by model_record.json.
    """
    if dry_run and apply_flag:
        raise SystemExit("Use either --dry-run or --apply, not both.")

    apply_changes = not dry_run

    models_root = REPO_ROOT / "models" / "simulink"
    if model is None:
        model_ids = list(ALLOWED_TSENV_MODELS)
    else:
        model_id = str(model).strip()
        if model_id not in ALLOWED_TSENV_MODELS:
            raise SystemExit(
                f"Model '{model_id}' is not an allowed tsENV model. "
                "Update shared/benchmark_utils.py (ALLOWED_TSENV_MODELS) to add it."
            )
        model_ids = [model_id]
    if runs_dir is not None and len(model_ids) != 1:
        raise click.UsageError("--runs-dir requires a specific model.")
    resolved_runs_dir = runs_dir.expanduser().resolve() if runs_dir is not None else None
    resolved_runs_dir_name = (
        str(runs_dir_name).strip() if runs_dir_name is not None else None
    )

    per_model: List[Dict[str, Any]] = []
    total_orphans = 0
    total_deleted = 0

    for model_id in sorted(model_ids):
        model_dir = models_root / model_id
        lock_ctx = (
            model_lock(
                model_id,
                purpose="delete_orphan_runs --apply",
                lock_root=model_dir / ".locks",
            )
            if apply_changes
            else nullcontext()
        )
        try:
            with lock_ctx:
                runs_dir = resolve_runs_root(
                    model_dir,
                    runs_dir=resolved_runs_dir,
                    runs_dir_name=resolved_runs_dir_name,
                )
                model_record_path = resolve_model_record_path(
                    model_dir,
                    runs_dir=resolved_runs_dir,
                    runs_dir_name=resolved_runs_dir_name,
                )
                if not runs_dir.exists():
                    per_model.append(
                        {
                            "model_id": model_id,
                            "orphans": [],
                            "deleted": [],
                            "runs_orphans": [],
                            "runs_deleted": [],
                            "skipped_reason": "runs_dir_missing",
                        }
                    )
                    continue
                if not model_record_path.exists():
                    per_model.append(
                        {
                            "model_id": model_id,
                            "orphans": [],
                            "deleted": [],
                            "runs_orphans": [],
                            "runs_deleted": [],
                            "skipped_reason": "model_record_missing",
                        }
                    )
                    continue

                specs_path = model_dir / "model_run_specs.json"
                if not specs_path.exists():
                    per_model.append(
                        {
                            "model_id": model_id,
                            "orphans": [],
                            "deleted": [],
                            "runs_orphans": [],
                            "runs_deleted": [],
                            "skipped_reason": "model_run_specs_missing",
                        }
                    )
                    continue

                specs = load_model_run_specs_json(
                    specs_path,
                    enforce_baseline_pair_diversity=False,
                )
                model_record = load_model_record_json(model_record_path)
                referenced = _referenced_run_ids(specs, model_record)
                runs_on_disk = _list_hash_dirs(runs_dir)
                runs_orphans = sorted(set(runs_on_disk) - referenced)

                runs_deleted: List[str] = []
                if apply_changes:
                    for run_id in runs_orphans:
                        _delete_tree_or_link(runs_dir / run_id)
                        runs_deleted.append(run_id)

                model_orphans = list(runs_orphans)
                model_deleted = list(runs_deleted)
                total_orphans += len(model_orphans)
                total_deleted += len(model_deleted)
                per_model.append(
                    {
                        "model_id": model_id,
                        # Backward compatibility aliases for existing UI callers.
                        "orphans": runs_orphans,
                        "deleted": runs_deleted,
                        "runs_orphans": runs_orphans,
                        "runs_deleted": runs_deleted,
                    }
                )
        except ModelLockError as exc:
            raise SystemExit(str(exc))

    print(
        json.dumps(
            {
                "apply": bool(apply_changes),
                "total_orphans": total_orphans,
                "total_deleted": total_deleted,
                "models": per_model,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli()
