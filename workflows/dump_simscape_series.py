#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import click

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.run_artifacts import resolve_runs_root


def _resolve_repo_root(user_repo_root: str | None) -> Path:
    if user_repo_root:
        return Path(user_repo_root).expanduser().resolve()
    cwd = Path(os.getcwd()).resolve()
    if (cwd / "models" / "simulink").exists():
        return cwd
    return root_dir


def _simlog_file_path(
    repo_root: Path,
    model_id: str,
    run_id: str,
    runs_dir_name: str | None,
) -> Path:
    model_dir = repo_root / "models" / "simulink" / model_id
    return (
        resolve_runs_root(model_dir, runs_dir_name=runs_dir_name)
        / run_id
        / "debug"
        / "simlog_segments.mat"
    )


def _resolve_model_id_from_run_id(
    repo_root: Path,
    run_id: str,
    runs_dir_name: str | None,
) -> str:
    models_root = repo_root / "models" / "simulink"
    if not models_root.exists():
        raise click.ClickException(f"Missing models root: {models_root}")
    matches: List[str] = []
    for model_dir in sorted(models_root.iterdir()):
        if not model_dir.is_dir():
            continue
        runs_dir = resolve_runs_root(model_dir, runs_dir_name=runs_dir_name)
        if not runs_dir.exists():
            continue
        candidate = runs_dir / run_id
        if candidate.is_dir():
            matches.append(model_dir.name)
    if not matches:
        raise click.ClickException(
            f"Run id not found under models/simulink/* run root '{runs_dir_name or 'runs'}': {run_id}"
        )
    if len(matches) > 1:
        raise click.ClickException(
            f"Ambiguous run id '{run_id}' found in models: {matches}. "
            "Pass --model-id to disambiguate."
        )
    return matches[0]


def _rows_to_table(rows: List[Dict[str, Any]]) -> str:
    headers = ["idx", "path", "points", "dims", "t_start", "t_end"]
    lines: List[str] = []
    lines.append(f"{headers[0]:>4} | {headers[1]:<90} | {headers[2]:>8} | {headers[3]:>4} | {headers[4]:>12} | {headers[5]:>12}")
    lines.append("-" * 150)
    for i, row in enumerate(rows, start=1):
        path = str(row.get("path", ""))
        points = int(row.get("points", 0))
        dims = int(row.get("dims", 0))
        t_start = float(row.get("t_start", float("nan")))
        t_end = float(row.get("t_end", float("nan")))
        lines.append(
            f"{i:4d} | {path:<90} | {points:8d} | {dims:4d} | {t_start:12.6g} | {t_end:12.6g}"
        )
    return "\n".join(lines)


def _rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    header = ["path", "points", "dims", "t_start", "t_end"]
    out.append(",".join(header))
    for row in rows:
        rec = [
            str(row.get("path", "")),
            str(int(row.get("points", 0))),
            str(int(row.get("dims", 0))),
            str(float(row.get("t_start", float("nan")))),
            str(float(row.get("t_end", float("nan")))),
        ]
        sio = []
        for cell in rec:
            if "," in cell or '"' in cell:
                cell = '"' + cell.replace('"', '""') + '"'
            sio.append(cell)
        out.append(",".join(sio))
    return "\n".join(out)


@click.command()
@click.option("--run-id", required=True, type=str, help="Run id under models/simulink/<model>/<runs-dir-name>/<run-id>.")
@click.option("--model-id", default=None, type=str, help="Model folder name. If omitted, inferred from --run-id.")
@click.option("--repo-root", default=None, type=click.Path(path_type=Path), help="Repository root path.")
@click.option(
    "--runs-dir-name",
    type=str,
    default=None,
    help="Model-local run-artifact directory name (for example: runs_7161).",
)
@click.option("--use-raw-segments", is_flag=True, help="Use raw simlogSegments traversal instead of simscapeMerged.")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option("--out", "out_path", default=None, type=click.Path(path_type=Path), help="Optional output file path.")
@click.option("--quiet", is_flag=True, help="Suppress banner output.")
def main(
    run_id: str,
    model_id: str | None,
    repo_root: Path | None,
    runs_dir_name: str | None,
    use_raw_segments: bool,
    output_format: str,
    out_path: Path | None,
    quiet: bool,
) -> None:
    repo = _resolve_repo_root(str(repo_root) if repo_root is not None else None)
    if model_id is None or not str(model_id).strip():
        model_id = _resolve_model_id_from_run_id(repo, str(run_id), runs_dir_name)
    mat_file = _simlog_file_path(repo, model_id, run_id, runs_dir_name)
    if not mat_file.exists():
        raise click.ClickException(
            f"Missing debug Simscape file: {mat_file}\n"
            "Run `env/bin/python workflows/simulate/run_pending_sims.py "
            f"{model_id} --save-simscape-mat` first."
        )

    try:
        import matlab.engine  # type: ignore
    except Exception as exc:
        raise click.ClickException(
            "MATLAB Engine for Python is not available in this environment."
        ) from exc

    eng = None
    try:
        eng = matlab.engine.start_matlab()
        eng.addpath(str(repo / "workflows"), nargout=0)
        eng.workspace["run_id_py"] = str(run_id)
        eng.workspace["model_id_py"] = str(model_id)
        eng.workspace["repo_root_py"] = str(repo)
        eng.workspace["use_raw_py"] = bool(use_raw_segments)
        eng.eval(
            "rows_tmp = dump_simscape_series(run_id_py, "
            "'ModelId', model_id_py, "
            "'RepoRoot', repo_root_py, "
            "'UseRawSegments', use_raw_py, "
            "'PrintTable', false);",
            nargout=0,
        )
        eng.eval("rows_json_tmp = jsonencode(rows_tmp);", nargout=0)
        rows_json = eng.workspace["rows_json_tmp"]
    except Exception as exc:
        raise click.ClickException(f"Failed to read Simscape series via MATLAB: {exc}") from exc
    finally:
        if eng is not None:
            try:
                eng.quit()
            except Exception:
                pass

    try:
        rows_raw = json.loads(str(rows_json))
    except Exception as exc:
        raise click.ClickException(f"Failed to parse MATLAB JSON output: {exc}") from exc

    if isinstance(rows_raw, dict):
        rows: List[Dict[str, Any]] = [rows_raw]
    elif isinstance(rows_raw, list):
        rows = [r for r in rows_raw if isinstance(r, dict)]
    else:
        rows = []

    rows.sort(key=lambda r: str(r.get("path", "")))

    if output_format == "json":
        rendered = json.dumps(rows, indent=2)
    elif output_format == "csv":
        rendered = _rows_to_csv(rows)
    else:
        rendered = _rows_to_table(rows)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
        if not quiet:
            click.echo(f"Wrote {len(rows)} rows to {out_path}")
    else:
        if not quiet:
            click.echo(
                f"Run: {run_id} | Model: {model_id} | Rows: {len(rows)} | Source: {mat_file}"
            )
        click.echo(rendered)


if __name__ == "__main__":
    main()
