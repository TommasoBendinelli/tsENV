#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from shared.scores_schema import validate_scores_payload


ROOT = Path(__file__).resolve().parent
RUN_CONFIGURATIONS_DIR = ROOT / "run_configurations"
OUTPUT_DIR = ROOT / "results" / "consolidated"
LOGS_DIR = ROOT / "logs"
DONE_STATUS = "DONE"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise click.ClickException(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _run_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d__%H-%M-%S")


def _resolve_configuration_path(configuration_name: str) -> Path:
    raw_value = str(configuration_name or "").strip()
    if not raw_value:
        raise click.ClickException("CONFIGURATION_NAME must be non-empty.")
    candidate = Path(raw_value).expanduser()
    if candidate.suffix == ".json" or candidate.parent != Path("."):
        return candidate.resolve()
    return (RUN_CONFIGURATIONS_DIR / f"{raw_value}.json").resolve()


def _safe_output_name(value: object, *, fallback: str) -> str:
    raw_value = str(value or "").strip() or fallback
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_value).strip("._")
    return safe or fallback


def _configuration_identity(configuration_path: Path, payload: dict[str, Any]) -> tuple[str, str]:
    configuration_stem = configuration_path.stem
    configuration_name = _safe_output_name(
        payload.get("name"),
        fallback=configuration_stem,
    )
    configuration_timestamp = _safe_output_name(
        payload.get("timestamp"),
        fallback=configuration_stem,
    )
    return configuration_name, configuration_timestamp


def _output_summary_path(row_slug: str) -> Path:
    safe_row_slug = _safe_output_name(row_slug, fallback="row")
    return OUTPUT_DIR / f"{safe_row_slug}_summary.json"


def _output_sample_path(row_slug: str) -> Path:
    safe_row_slug = _safe_output_name(row_slug, fallback="row")
    return OUTPUT_DIR / f"{safe_row_slug}.json"


def _required_text(run: dict[str, Any], key: str, *, run_index: int) -> str:
    value = str(run.get(key) or "").strip()
    if not value:
        raise click.ClickException(f"Run entry {run_index} is missing required key {key!r}.")
    return value


def _recipe_from_run(run: dict[str, Any]) -> dict[str, Any]:
    prefix = "question.recipe_info."
    return {
        key[len(prefix):]: value
        for key, value in sorted(run.items())
        if str(key).startswith(prefix)
    }


def _row_slug_from_recipe_or_question_slug(recipe: dict[str, Any], question_slug: str) -> str:
    return str(recipe.get("row_slug") or question_slug).strip()


def _entry_seed_from_recipe_or_slug(recipe: dict[str, Any], question_slug: str) -> int | None:
    raw_seed = recipe.get("question_seed")
    if not isinstance(raw_seed, bool) and isinstance(raw_seed, int):
        return raw_seed
    if isinstance(raw_seed, str) and raw_seed.strip().isdigit():
        return int(raw_seed)
    match = re.search(r"_(\d+)$", str(question_slug or ""))
    if match is None:
        return None
    return int(match.group(1))


def _canonical_scores_paths(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.rglob("scores.json")
        if "artifacts" not in path.relative_to(run_dir).parts
    )


def _find_single_scores_json(run_dir: Path, *, run_index: int) -> Path:
    scores_paths = _canonical_scores_paths(run_dir)
    if not scores_paths:
        raise click.ClickException(
            f"Run entry {run_index} has no scores.json under {run_dir}."
        )
    if len(scores_paths) > 1:
        formatted = ", ".join(str(path) for path in scores_paths)
        raise click.ClickException(
            f"Run entry {run_index} has multiple scores.json files under {run_dir}: {formatted}"
        )
    return scores_paths[0]


def _required_mapping(payload: Any, *, path: str, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise click.ClickException(f"{path}::{label} must be a JSON object.")
    return payload


def _required_number(payload: dict[str, Any], key: str, *, path: str) -> int | float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise click.ClickException(f"{path}::{key} must be numeric.")
    return value


def _trajectory_summary(trajectory_path: Path) -> dict[str, Any]:
    payload = _required_mapping(
        _read_json(trajectory_path),
        path=str(trajectory_path),
        label="",
    )
    tokens = _required_mapping(
        payload.get("tokens"),
        path=str(trajectory_path),
        label="tokens",
    )
    number_of_interactions = _required_mapping(
        payload.get("number_of_interactions"),
        path=str(trajectory_path),
        label="number_of_interactions",
    )
    agent_interactions = _required_mapping(
        number_of_interactions.get("agent"),
        path=str(trajectory_path),
        label="number_of_interactions.agent",
    )
    summary = {
        "tokens": tokens,
        "number_of_tool_calls": _required_number(
            agent_interactions,
            "tool_call_steps",
            path=str(trajectory_path),
        ),
        "total_steps": _required_number(
            agent_interactions,
            "steps",
            path=str(trajectory_path),
        ),
        "cost_usd": _required_number(payload, "cost_usd", path=str(trajectory_path)),
    }
    subcommand_analysis = payload.get("subcommand_analysis")
    if isinstance(subcommand_analysis, dict):
        python_analysis = subcommand_analysis.get("python")
        if isinstance(python_analysis, dict) and "total_invocations" in python_analysis:
            summary["python_calls"] = _required_number(
                python_analysis,
                "total_invocations",
                path=str(trajectory_path),
            )
    artifact_analysis = payload.get("artifact_analysis")
    if isinstance(artifact_analysis, dict):
        plots = artifact_analysis.get("plots")
        if isinstance(plots, list):
            summary["plots_generated"] = len(plots)
    return summary


def _wrong_uuids(question_slug: str, sample_result_batches: list[Any]) -> list[list[str]]:
    wrong: list[list[str]] = []
    for batch in sample_result_batches:
        if not isinstance(batch, dict):
            continue
        for sample_uuid, sample_result in batch.items():
            if not isinstance(sample_result, dict):
                continue
            if str(sample_result.get("sample_type") or "") != "test":
                continue
            if sample_result.get("top1_correct") is False:
                wrong.append([question_slug, str(sample_uuid)])
    return wrong


def _load_run_artifacts_if_eligible(
    run: dict[str, Any],
    *,
    run_index: int,
) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    status = str(run.get("status") or "").strip()
    if status != DONE_STATUS:
        return None
    run_dir = Path(_required_text(run, "path_to_the_run", run_index=run_index)).expanduser()
    if not run_dir.is_dir():
        return None
    trajectory_path = run_dir / "trajectory_evaluation.json"
    if not trajectory_path.is_file():
        return None
    trajectory_payload = _trajectory_summary(trajectory_path)
    if not _canonical_scores_paths(run_dir):
        return None
    scores_path = _find_single_scores_json(run_dir, run_index=run_index)
    scores_payload = _read_json(scores_path)
    try:
        validated_scores = validate_scores_payload(scores_payload, path=str(scores_path)).model_dump()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if validated_scores.get("final_metric_other") is None:
        validated_scores.pop("final_metric_other", None)
    return scores_path, validated_scores, trajectory_payload


def _load_configuration_payload(configuration_path: Path) -> dict[str, Any]:
    payload = _read_json(configuration_path)
    if not isinstance(payload, dict):
        raise click.ClickException(f"Configuration must be a JSON object: {configuration_path}")
    return payload


def _run_tag(run: dict[str, Any]) -> str:
    return str(run.get("tag") or "").strip()


def _configuration_has_run_tag(payload: dict[str, Any], tag: str) -> bool:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return False
    return any(isinstance(run, dict) and _run_tag(run) == tag for run in runs)


def _configuration_legacy_matches_tag(configuration_path: Path, payload: dict[str, Any], tag: str) -> bool:
    if tag in configuration_path.stem:
        return True
    configuration_name = str(payload.get("name") or "").strip()
    return tag in configuration_name


def _configuration_matches_tag(configuration_path: Path, payload: dict[str, Any], tag: str | None) -> bool:
    raw_tag = str(tag or "").strip()
    if not raw_tag:
        return True
    return (
        _configuration_has_run_tag(payload, raw_tag)
        or _configuration_legacy_matches_tag(configuration_path, payload, raw_tag)
    )


def _resolve_configuration_inputs(tag: str | None) -> list[tuple[Path, dict[str, Any], str, str, str | None]]:
    raw_tag = str(tag or "").strip()
    explicit_path = Path(raw_tag).expanduser() if raw_tag else None
    if explicit_path is not None and (explicit_path.suffix == ".json" or explicit_path.parent != Path(".")):
        configuration_path = explicit_path.resolve()
        payload = _load_configuration_payload(configuration_path)
        configuration_name, configuration_timestamp = _configuration_identity(configuration_path, payload)
        return [(configuration_path, payload, configuration_name, configuration_timestamp, None)]

    configuration_inputs: list[tuple[Path, dict[str, Any], str, str, str | None]] = []
    for configuration_path in sorted(RUN_CONFIGURATIONS_DIR.glob("*.json")):
        payload = _load_configuration_payload(configuration_path)
        if not _configuration_matches_tag(configuration_path, payload, raw_tag or None):
            continue
        configuration_name, configuration_timestamp = _configuration_identity(configuration_path, payload)
        run_tag_filter = raw_tag if raw_tag and _configuration_has_run_tag(payload, raw_tag) else None
        configuration_inputs.append(
            (
                configuration_path.resolve(),
                payload,
                configuration_name,
                configuration_timestamp,
                run_tag_filter,
            )
        )
    if not configuration_inputs:
        suffix = f" matching tag {raw_tag!r}" if raw_tag else ""
        raise click.ClickException(f"No run configuration files found in {RUN_CONFIGURATIONS_DIR}{suffix}.")
    return configuration_inputs


def _consolidate_configuration_entries(
    configuration_path: Path,
    payload: dict[str, Any],
    *,
    configuration_name: str,
    configuration_timestamp: str,
    run_tag_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise click.ClickException(f"Configuration missing runs list: {configuration_path}")

    grouped: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
    family_expected_seeds: dict[tuple[str, str, str], set[int | None]] = {}
    family_valid_seeds: dict[tuple[str, str, str], set[int | None]] = {}
    family_question_slug_by_seed: dict[tuple[str, str, str], dict[int | None, str]] = {}
    entry_family: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for run_index, raw_run in enumerate(runs):
        if not isinstance(raw_run, dict):
            raise click.ClickException(f"Run entry {run_index} must be a JSON object.")
        run_tag = _run_tag(raw_run)
        if run_tag_filter is not None and run_tag != run_tag_filter:
            continue
        agent_id = _required_text(raw_run, "agent_id", run_index=run_index)
        model = _required_text(raw_run, "model", run_index=run_index)
        question_slug = _required_text(raw_run, "question_slug", run_index=run_index)
        recipe = _recipe_from_run(raw_run)
        row_slug = _row_slug_from_recipe_or_question_slug(recipe, question_slug)
        seed = _entry_seed_from_recipe_or_slug(recipe, question_slug)
        family_key = (agent_id, model, row_slug)
        family_expected_seeds.setdefault(family_key, set()).add(seed)
        family_question_slug_by_seed.setdefault(family_key, {})[seed] = question_slug
        artifacts = _load_run_artifacts_if_eligible(
            raw_run,
            run_index=run_index,
        )
        if artifacts is None:
            continue
        scores_path, scores_payload, trajectory_payload = artifacts
        family_valid_seeds.setdefault(family_key, set()).add(seed)
        group_key = (agent_id, model, question_slug)
        entry_family[group_key] = family_key
        if group_key not in grouped:
            grouped[group_key] = {
                "agent_id": agent_id,
                "paths": [],
                "configuration_name": configuration_name,
                "simulator": model,
                "question_slug": question_slug,
                "recipe": recipe,
                "score_batch": [],
                "sample_results": [],
                "trajectory_evaluation": [],
                "tags": [],
            }
        entry = grouped[group_key]
        if entry["recipe"] != recipe:
            raise click.ClickException(
                f"Conflicting recipe values for model={model!r}, question_slug={question_slug!r}."
            )
        entry["paths"].append(str(scores_path))
        entry["score_batch"].append(scores_payload["final_metric_test"])
        if "final_metric_other" in scores_payload:
            entry.setdefault("score_other_batch", []).append(scores_payload["final_metric_other"])
        entry["sample_results"].append(scores_payload["sample_results"])
        entry["trajectory_evaluation"].append(trajectory_payload)
        if run_tag:
            entry["tags"].append(run_tag)

    summary_entries: list[dict[str, Any]] = []
    sample_entries: list[dict[str, Any]] = []
    incomplete_entries: list[dict[str, Any]] = []
    logged_incomplete_families: set[tuple[str, str]] = set()
    for group_key, entry in grouped.items():
        family_key = entry_family.get(group_key)
        if family_key is not None:
            expected_seeds = family_expected_seeds.get(family_key, set())
            valid_seeds = family_valid_seeds.get(family_key, set())
            if not expected_seeds.issubset(valid_seeds):
                if valid_seeds and family_key not in logged_incomplete_families:
                    logged_incomplete_families.add(family_key)
                    missing_seeds = sorted(
                        seed for seed in expected_seeds - valid_seeds if seed is not None
                    )
                    question_slug_by_seed = family_question_slug_by_seed.get(family_key, {})
                    row_slug = family_key[2]
                    incomplete_entries.append(
                        {
                            "agent_id": family_key[0],
                            "simulator": family_key[1],
                            "row_slug": row_slug,
                            "configuration_name": configuration_name,
                            "timestamp": configuration_timestamp,
                            "present_seeds": sorted(seed for seed in valid_seeds if seed is not None),
                            "missing_seeds": missing_seeds,
                            "present_question_slugs": [
                                question_slug_by_seed[seed]
                                for seed in sorted(valid_seeds, key=lambda value: -1 if value is None else value)
                                if seed in question_slug_by_seed
                            ],
                            "missing_question_slugs": [
                                question_slug_by_seed.get(seed, f"{row_slug}_{seed}")
                                for seed in missing_seeds
                            ],
                        }
                    )
                continue
        identity_entries = {
            "timestamp": configuration_timestamp,
            "configuration_name": entry["configuration_name"],
            "agent_id": entry["agent_id"],
            "paths": entry["paths"],
            "simulator": entry["simulator"],
            "question_slug": entry["question_slug"],
            "recipe": entry["recipe"],
        }
        unique_tags = sorted(set(entry.get("tags") or []))
        if len(unique_tags) == 1:
            identity_entries["tag"] = unique_tags[0]
        elif unique_tags:
            identity_entries["tags"] = unique_tags
        summary_entry = {
            "identity_entries": identity_entries,
            "score_batch": entry["score_batch"],
            "trajectory_evaluation": entry["trajectory_evaluation"],
        }
        if "score_other_batch" in entry:
            summary_entry["score_other_batch"] = entry["score_other_batch"]
        summary_entries.append(summary_entry)
        sample_entries.append({
            "wrong_uuids": _wrong_uuids(entry["question_slug"], entry["sample_results"]),
            "identity_entries": identity_entries,
            "sample_results": entry["sample_results"],
        })

    return summary_entries, sample_entries, incomplete_entries


def _newer_configuration_sort_key(configuration_timestamp: str, configuration_name: str) -> tuple[str, str]:
    return configuration_timestamp, configuration_name


def _row_slug_from_entry(entry: dict[str, Any]) -> str:
    identity_entries = entry.get("identity_entries")
    if not isinstance(identity_entries, dict):
        identity_entries = entry
    recipe = identity_entries.get("recipe")
    if isinstance(recipe, dict):
        row_slug = str(recipe.get("row_slug") or "").strip()
        if row_slug:
            return row_slug
    return str(identity_entries.get("question_slug") or "").strip()


def _identity_value(entry: dict[str, Any], key: str) -> Any:
    identity_entries = entry.get("identity_entries")
    if isinstance(identity_entries, dict):
        return identity_entries.get(key)
    return entry.get(key)


def _identity_group_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(_identity_value(entry, "agent_id") or ""),
        str(_identity_value(entry, "simulator") or _identity_value(entry, "model") or ""),
        str(_identity_value(entry, "question_slug") or ""),
    )


def _write_incomplete_log(entries: list[dict[str, Any]], *, log_timestamp: str) -> Path | None:
    if not entries:
        return None
    log_path = LOGS_DIR / f"consolidate_logs_{_safe_output_name(log_timestamp, fallback='log')}"
    _write_json(log_path, entries)
    return log_path


def consolidate_configurations(
    configuration_inputs: list[tuple[Path, dict[str, Any], str, str, str | None]],
) -> tuple[dict[str, Path], dict[str, Path], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], Path | None]:
    selected_family_by_key: OrderedDict[
        tuple[str, str, str],
        tuple[tuple[str, str], str, list[dict[str, Any]], list[dict[str, Any]]],
    ] = OrderedDict()
    incomplete_entries: list[dict[str, Any]] = []
    for configuration_path, payload, configuration_name, configuration_timestamp, run_tag_filter in configuration_inputs:
        summary_entries, sample_entries, configuration_incomplete_entries = _consolidate_configuration_entries(
            configuration_path,
            payload,
            configuration_name=configuration_name,
            configuration_timestamp=configuration_timestamp,
            run_tag_filter=run_tag_filter,
        )
        incomplete_entries.extend(configuration_incomplete_entries)
        sample_by_key = {_identity_group_key(entry): entry for entry in sample_entries}
        summary_entries_by_family: OrderedDict[tuple[str, str, str], list[dict[str, Any]]] = OrderedDict()
        sample_entries_by_family: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for summary_entry in summary_entries:
            agent_id = str(_identity_value(summary_entry, "agent_id") or "")
            model = str(
                _identity_value(summary_entry, "simulator")
                or _identity_value(summary_entry, "model")
                or ""
            )
            row_slug = _row_slug_from_entry(summary_entry)
            family_key = (agent_id, model, row_slug)
            summary_entries_by_family.setdefault(family_key, []).append(summary_entry)
            sample_entry = sample_by_key.get(_identity_group_key(summary_entry))
            if sample_entry is not None:
                sample_entries_by_family.setdefault(family_key, []).append(sample_entry)

        sort_key = _newer_configuration_sort_key(configuration_timestamp, configuration_name)
        for family_key, family_summary_entries in summary_entries_by_family.items():
            current = selected_family_by_key.get(family_key)
            if current is not None and current[0] >= sort_key:
                continue
            selected_family_by_key[family_key] = (
                sort_key,
                family_key[2],
                family_summary_entries,
                sample_entries_by_family.get(family_key, []),
            )

    summary_entries_by_row_slug: dict[str, list[dict[str, Any]]] = OrderedDict()
    sample_entries_by_row_slug: dict[str, list[dict[str, Any]]] = OrderedDict()
    for _family_key, (_sort_key, row_slug, family_summary_entries, family_sample_entries) in selected_family_by_key.items():
        if not family_summary_entries or not family_sample_entries:
            continue
        summary_entries_by_row_slug.setdefault(row_slug, []).extend(family_summary_entries)
        sample_entries_by_row_slug.setdefault(row_slug, []).extend(family_sample_entries)

    summary_paths: dict[str, Path] = OrderedDict()
    sample_paths: dict[str, Path] = {}
    for row_slug, summary_entries in summary_entries_by_row_slug.items():
        if not summary_entries:
            continue
        sample_entries = sample_entries_by_row_slug.get(row_slug, [])
        if not sample_entries:
            continue
        summary_path = _output_summary_path(row_slug)
        sample_path = _output_sample_path(row_slug)
        _write_json(summary_path, summary_entries)
        _write_json(sample_path, sample_entries)
        summary_paths[row_slug] = summary_path
        sample_paths[row_slug] = sample_path
    log_path = _write_incomplete_log(incomplete_entries, log_timestamp=_run_timestamp())
    return summary_paths, sample_paths, summary_entries_by_row_slug, sample_entries_by_row_slug, log_path


def consolidate_configuration(
    configuration_path: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], Path | None]:
    payload = _load_configuration_payload(configuration_path)
    configuration_name, configuration_timestamp = _configuration_identity(configuration_path, payload)
    return consolidate_configurations(
        [(configuration_path, payload, configuration_name, configuration_timestamp, None)],
    )


@click.command()
@click.argument("tag", required=False)
def main(tag: str | None) -> None:
    """Consolidate completed rollout configurations into documented results JSON files."""
    if tag and (Path(str(tag)).suffix == ".json" or Path(str(tag)).parent != Path(".")):
        configuration_path = _resolve_configuration_path(str(tag))
        payload = _load_configuration_payload(configuration_path)
        configuration_name, configuration_timestamp = _configuration_identity(configuration_path, payload)
        configuration_inputs = [(configuration_path, payload, configuration_name, configuration_timestamp, None)]
    else:
        configuration_inputs = _resolve_configuration_inputs(tag)
    summary_paths, sample_paths, summary_entries_by_row_slug, _sample_entries, log_path = consolidate_configurations(configuration_inputs)
    entry_count = sum(len(entries) for entries in summary_entries_by_row_slug.values())
    entry_word = "entry" if entry_count == 1 else "entries"
    if not summary_paths:
        click.echo(f"Wrote 0 consolidated {entry_word}; no non-empty row_slug outputs created.")
        if log_path is not None:
            click.echo(f"Wrote incomplete seed log to {log_path}")
        return
    click.echo(f"Wrote {entry_count} consolidated {entry_word} across {len(summary_paths)} row_slug file(s).")
    for row_slug, summary_path in summary_paths.items():
        click.echo(f"Wrote summary results for {row_slug} to {summary_path}")
    for row_slug, sample_path in sample_paths.items():
        click.echo(f"Wrote sample results for {row_slug} to {sample_path}")
    if log_path is not None:
        click.echo(f"Wrote incomplete seed log to {log_path}")


if __name__ == "__main__":
    main()
