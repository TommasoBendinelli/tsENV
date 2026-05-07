"""Build a Croissant 1.1 metadata file for one or all tsENV environments.

Usage:
    # Single environment (legacy behaviour)
    python build_croissant.py BallDrop
    python build_croissant.py BallDrop --output BallDrop.croissant.json

    # Combined file covering every environment under tsENV_questions/
    python build_croissant.py --all
    python build_croissant.py --all --output tsENV.croissant.json

The script discovers the Parquet files under
``tsENV_questions/<env>/dataframes/`` and the per-run metadata in
``model_record.json``, and emits a JSON-LD file laid out per the Croissant
spec: https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html#resources

Resource layout produced (per environment):
- FileObject  -> single auxiliary files (model_record.json, sample_manifest.json, questions.json, noise_adder.py)
- FileSet     -> the collection of per-run Parquet files in dataframes/
- RecordSet   -> the time-series rows extracted from those Parquet files
- RecordSet   -> a "questions" view of questions.json
- RecordSet   -> a "labels" enumeration of the label space

In combined mode, every per-env @id is namespaced with ``<env>/`` to avoid
collisions, and a single shared ``repo`` FileObject is emitted at the top.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "tsENV_questions"
CROISSANT_VERSION = "1.1"

GENERAL_DESCRIPTION = (
    "tsENV environment: a structured benchmark dataset of simulated time-series "
    "from dynamical systems with controlled interventions on system parameters. "
    "Each sample represents either a baseline or an intervention where a single "
    "parameter is changed, enabling evaluation of models that identify the "
    "intervened parameter from observed trajectories."
)

RAI_FIELDS: dict[str, Any] = {
    "rai:dataLimitations": (
        "Limited to the specific simulator designs, parameter ranges, observable "
        "signals, and intervention mechanisms defined in tsENV. It is not suitable "
        "for real-world system identification, control, robotics deployment, or "
        "safety-critical decision-making."
    ),
    "rai:dataBiases": (
        "The dataset reflects the structural assumptions of the tsENV simulation "
        "pipeline, including parameter sampling strategies, detectability "
        "thresholds (e.g., SRD-based criteria), and predefined observable signals."
    ),
    "rai:personalSensitiveInformation": (
        "The dataset contains no personal or sensitive information. All data is "
        "generated from physics-based simulations and includes only numerical "
        "time-series signals, parameters, and derived metadata. No human subjects, "
        "demographic attributes, geographic identifiers, health data, or other "
        "sensitive categories are present."
    ),
    "rai:dataUseCases": (
        "The dataset is designed to evaluate models on identifying which system "
        "parameter was intervened upon from observed time-series trajectories. "
        "Validated use cases include benchmark evaluation of e.g. agent-based "
        "reasoning systems within the tsENV framework. It is not validated for "
        "real-world prediction, physical system control, scientific discovery, or "
        "causal inference outside the simulator setting."
    ),
    "rai:dataSocialImpact": (
        "The dataset promotes reproducible evaluation of agents on structured "
        "reasoning tasks involving interventions in dynamical systems, supporting "
        "research in interpretability and causal reasoning. Potential risks "
        "include overinterpreting benchmark performance as evidence of real-world "
        "physical reasoning or deployment readiness, especially in safety-critical "
        "domains such as robotics or engineering. These risks are mitigated by "
        "the synthetic nature of the data, explicit detectability constraints, "
        "and clear documentation of simulator assumptions and limitations."
    ),
    "rai:hasSyntheticData": True,
}

PROV_FIELDS: dict[str, Any] = {
    "prov:wasDerivedFrom": (
        "This dataset is fully synthetic and generated from simulation code "
        "defined in the tsENV repository. It is not derived from any external "
        "or real-world datasets."
    ),
    "prov:wasGeneratedBy": [
        {
            "@type": "prov:Activity",
            "@id": "tsENV-pipeline",
            "name": "tsENV simulation pipeline",
            "description": "The dataset was generated using the tsENV pipeline.",
        }
    ],
}

ENV_DESCRIPTIONS = {
    "BallDrop": (
        "In the BallDrop environment, the system models a bouncing ball under "
        "gravity with parameters such as mass, drag coefficient, and restitution "
        "determining the resulting motion trajectories."
    ),
    "InclinedPlane": (
        "In the InclinedPlane environment, the system models an object sliding "
        "along an inclined surface under gravity, where parameters such as "
        "inclination angle, friction coefficient, and mass determine the "
        "resulting motion trajectories."
    ),
    "DampedMassBetweenWalls": (
        "In the DampedMassBetweenWalls environment, the system models a mass "
        "moving between two rigid boundaries with damping, where parameters such "
        "as mass, damping coefficient, and boundary stiffness determine the "
        "resulting motion trajectories."
    ),
}

DTYPE_TO_CROISSANT = {
    "float16": "sc:Float",
    "float32": "sc:Float",
    "float64": "sc:Float",
    "int8": "sc:Integer",
    "int16": "sc:Integer",
    "int32": "sc:Integer",
    "int64": "sc:Integer",
    "bool": "sc:Boolean",
    "object": "sc:Text",
    "string": "sc:Text",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def croissant_context() -> dict[str, Any]:
    """Canonical Croissant 1.1 @context (mirrors mlcroissant._src.core.rdf.make_context)."""
    return {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "arrayShape": "cr:arrayShape",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "containedIn": "cr:containedIn",
        "cr": "http://mlcommons.org/croissant/",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "equivalentProperty": "cr:equivalentProperty",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isArray": "cr:isArray",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "prov": "http://www.w3.org/ns/prov#",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "samplingRate": "cr:samplingRate",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
    }


REPO_FILE_OBJECT: dict[str, Any] = {
    "@type": "cr:FileObject",
    "@id": "repo",
    "name": "repo",
    "description": "tsENV git repository containing all environment data.",
    "contentUrl": "https://github.com/TommasoBendinelli/tsENV",
    "encodingFormat": "git+https",
    "sha256": "https://github.com/mlcommons/croissant/issues/80",
}


def parquet_field_specs(
    sample_parquet: Path, fileset_id: str, id_prefix: str
) -> list[dict[str, Any]]:
    df = pd.read_parquet(sample_parquet)
    fields = []
    for col, dtype in df.dtypes.items():
        cr_type = DTYPE_TO_CROISSANT.get(str(dtype), "sc:Text")
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"{id_prefix}timeseries/{col}",
                "name": col,
                "description": f"Column '{col}' from each Parquet run file (dtype: {dtype}).",
                "dataType": cr_type,
                "source": {
                    "fileSet": {"@id": fileset_id},
                    "extract": {"column": col},
                },
            }
        )
    return fields


def build_env_distribution(
    env_dir: Path, env_name: str, id_prefix: str
) -> tuple[list[dict[str, Any]], list[Path]]:
    distribution: list[dict[str, Any]] = []

    aux_files = {
        "model-record": (
            "model_record.json",
            "Per-run metadata: parameters_hash, run_type, class label, status, timestamp.",
            "application/json",
        ),
        "sample-manifest": (
            "sample_manifest.json",
            "Train/test split assignment per question variant.",
            "application/json",
        ),
        "questions-file": (
            "questions.json",
            "Question definitions, prompts, label space, and evaluation rules.",
            "application/json",
        ),
        "noise-adder": (
            "noise_adder.py",
            "Model-specific script for adding low/high noise profiles and reporting noise analysis.",
            "text/x-python",
        ),
    }
    for rid, (fname, desc, encoding) in aux_files.items():
        fpath = env_dir / fname
        if not fpath.exists():
            continue
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": f"{id_prefix}{rid}",
                "name": fname,
                "description": desc,
                "containedIn": {"@id": "repo"},
                "contentUrl": f"tsENV_questions/{env_name}/{fname}",
                "encodingFormat": encoding,
                "sha256": sha256_of(fpath),
            }
        )

    parquet_files = sorted((env_dir / "dataframes").glob("*.parquet"))
    if parquet_files:
        distribution.append(
            {
                "@type": "cr:FileSet",
                "@id": f"{id_prefix}parquet-files",
                "name": "parquet-files",
                "description": "One Parquet file per simulation run; filename stem is the run UUID.",
                "containedIn": {"@id": "repo"},
                "encodingFormat": "application/vnd.apache.parquet",
                "includes": f"tsENV_questions/{env_name}/dataframes/*.parquet",
            }
        )

    return distribution, parquet_files


def build_env_record_sets(
    parquet_files: list[Path], env_dir: Path, id_prefix: str
) -> list[dict[str, Any]]:
    record_sets: list[dict[str, Any]] = []

    fileset_id = f"{id_prefix}parquet-files"
    questions_file_id = f"{id_prefix}questions-file"

    if parquet_files:
        ts_fields = parquet_field_specs(
            parquet_files[0], fileset_id=fileset_id, id_prefix=id_prefix
        )
        ts_fields.insert(
            0,
            {
                "@type": "cr:Field",
                "@id": f"{id_prefix}timeseries/run_uuid",
                "name": "run_uuid",
                "description": "Run UUID; the Parquet file stem and canonical run identity.",
                "dataType": "sc:Text",
                "source": {
                    "fileSet": {"@id": fileset_id},
                    "extract": {"fileProperty": "filename"},
                    "transform": {"regex": "^(.+)\\.parquet$"},
                },
            },
        )
        record_sets.append(
            {
                "@type": "cr:RecordSet",
                "@id": f"{id_prefix}timeseries",
                "name": "timeseries",
                "description": "Time-series rows from each simulation run.",
                "field": ts_fields,
            }
        )

    questions_path = env_dir / "questions.json"
    if questions_path.exists():
        record_sets.append(
            _build_questions_record_set(
                id_prefix=id_prefix, questions_file_id=questions_file_id
            )
        )
        label_mapping = _extract_label_int_mapping(env_dir)
        if label_mapping:
            record_sets.append(
                _build_labels_record_set(label_mapping, id_prefix=id_prefix)
            )

    return record_sets


def _build_labels_record_set(
    mapping: dict[str, int], id_prefix: str
) -> dict[str, Any]:
    """Enumerated RecordSet for the label space.

    The mapping in questions.json is a JSON object whose keys are the labels and
    whose values are integer codes. Dict keys are not extractable as record
    values via jsonpath_rw, so the mapping is inlined as Croissant ``data`` —
    the canonical pattern for small enumerations. The label set is the union of
    the intervened parameter names and the class "no parameter change".
    """
    label_id = f"{id_prefix}labels/label"
    integer_id = f"{id_prefix}labels/integer_id"
    rows = [
        {label_id: label, integer_id: int(idx)}
        for label, idx in sorted(mapping.items(), key=lambda kv: kv[1])
    ]
    return {
        "@type": "cr:RecordSet",
        "@id": f"{id_prefix}labels",
        "name": "labels",
        "description": (
            "Label space for this environment: intervened parameter names plus "
            "the class 'no parameter change'. Each label has a stable integer "
            "code shared across all questions in the environment."
        ),
        "key": {"@id": integer_id},
        "field": [
            {
                "@type": "cr:Field",
                "@id": label_id,
                "name": "label",
                "description": "Human-readable label name.",
                "dataType": "sc:Text",
            },
            {
                "@type": "cr:Field",
                "@id": integer_id,
                "name": "integer_id",
                "description": "Stable integer code for the label.",
                "dataType": "sc:Integer",
            },
        ],
        "data": rows,
    }


def _extract_label_int_mapping(env_dir: Path) -> dict[str, int]:
    qpath = env_dir / "questions.json"
    if not qpath.exists():
        return {}
    data = json.loads(qpath.read_text())
    mapping = data.get("label_int_mapping")
    if not isinstance(mapping, dict):
        return {}
    return {str(k): int(v) for k, v in mapping.items()}


def _build_questions_record_set(
    id_prefix: str, questions_file_id: str
) -> dict[str, Any]:
    """RecordSet describing questions.json.

    questions.json is a dict keyed by question id. The dict key (slug) is not
    exposed as a field because jsonpath_rw cannot reliably extract dict keys as
    record values. Only fields that live as values inside each question object
    are emitted; consumers needing the slug should load questions.json directly.
    """
    file_ref = {"fileObject": {"@id": questions_file_id}}

    def field(field_name: str, description: str, dtype: str, jpath: str, repeated: bool = False) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "@type": "cr:Field",
            "@id": f"{id_prefix}questions/{field_name}",
            "name": field_name,
            "description": description,
            "dataType": dtype,
            "source": {**file_ref, "extract": {"jsonPath": jpath}},
        }
        if repeated:
            spec["repeated"] = True
        return spec

    return {
        "@type": "cr:RecordSet",
        "@id": f"{id_prefix}questions",
        "name": "questions",
        "description": (
            "One record per question. Each question fixes a train/test split "
            "and the label space (intervened parameter names plus 'no parameter change')."
        ),
        "field": [
            field("question_hash", "Hash uniquely identifying this question.", "sc:Text", "$.questions.*.question_hash"),
            field("train_test_sample_hash", "Hash uniquely identifying the train/test sample assignment for this question.", "sc:Text", "$.questions.*.train_test_sample_hash"),
            field("allowed_labels", "Label space for this question: intervened parameter names plus the class 'no parameter change'.", "sc:Text", "$.questions.*.question_text.allowed_labels", repeated=True),
            field("desc_level", "Verbosity level of the environment description shown to the agent.", "sc:Text", "$.questions.*.recipe_info.desc_level"),
            field("noise_level", "Noise profile applied to the time-series ('none', 'low', 'high').", "sc:Text", "$.questions.*.recipe_info.noise_level"),
            field("is_adversarial", "Whether the question uses an adversarial recipe (may be null).", "sc:Boolean", "$.questions.*.recipe_info.is_adversarial"),
            field("number_test_samples", "Number of test samples included in the question.", "sc:Integer", "$.questions.*.recipe_info.number_test_samples"),
            field("number_train_samples_per_class", "Number of few-shot training samples per class (0 for zero-shot).", "sc:Integer", "$.questions.*.recipe_info.number_train_samples_per_class"),
            field("question_seed", "Random seed used when sampling the question variant.", "sc:Integer", "$.questions.*.recipe_info.question_seed"),
            field("row_slug", "Identifier for the row (recipe row) the question instantiates.", "sc:Text", "$.questions.*.recipe_info.row_slug"),
            field("shot_slug", "Identifier for the shot (few-shot configuration) the question uses.", "sc:Text", "$.questions.*.recipe_info.shot_slug"),
            field("test_set_slug", "Identifier for the test set used by this question.", "sc:Text", "$.questions.*.recipe_info.test_set_slug"),
            field("type_of_request", "How the question is posed to the agent (e.g. 'direct').", "sc:Text", "$.questions.*.recipe_info.type_of_request"),
        ],
    }


def _extract_allowed_labels(env_dir: Path) -> list[str]:
    """Read the label space from questions.json (label_int_mapping is canonical)."""
    qpath = env_dir / "questions.json"
    if not qpath.exists():
        return []
    data = json.loads(qpath.read_text())
    mapping = data.get("label_int_mapping")
    if isinstance(mapping, dict) and mapping:
        return list(mapping.keys())
    questions = data.get("questions", {})
    for q in questions.values():
        labels = q.get("question_text", {}).get("allowed_labels")
        if labels:
            return list(labels)
    return []


def _resolve_env_dir(env_name: str) -> Path:
    env_dir = QUESTIONS_DIR / env_name
    if not env_dir.is_dir():
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")
    return env_dir


def _env_description_block(env_name: str, env_dir: Path) -> tuple[str, list[str]]:
    """Return the description sentence(s) for an env and its labels."""
    env_specific = ENV_DESCRIPTIONS.get(env_name, "")
    allowed_labels = _extract_allowed_labels(env_dir)
    desc = env_specific
    if allowed_labels:
        labels_str = ", ".join(f"'{lbl}'" for lbl in allowed_labels)
        desc += (
            f" Label space for {env_name}: {labels_str} — i.e. the names of the "
            "intervened parameters plus the class 'no parameter change'."
        )
    return desc, allowed_labels


def build_croissant(env_names: list[str]) -> dict[str, Any]:
    if not env_names:
        raise ValueError("At least one environment is required.")

    combined = len(env_names) > 1

    distribution: list[dict[str, Any]] = [REPO_FILE_OBJECT]
    record_sets: list[dict[str, Any]] = []
    description_parts: list[str] = [GENERAL_DESCRIPTION]
    keywords: list[str] = []

    for env_name in env_names:
        env_dir = _resolve_env_dir(env_name)
        id_prefix = f"{env_name}/" if combined else ""

        env_distribution, parquet_files = build_env_distribution(
            env_dir, env_name, id_prefix
        )
        env_record_sets = build_env_record_sets(parquet_files, env_dir, id_prefix)
        env_desc, env_labels = _env_description_block(env_name, env_dir)

        distribution.extend(env_distribution)
        record_sets.extend(env_record_sets)
        if env_desc:
            description_parts.append(env_desc)
        for lbl in env_labels:
            if lbl not in keywords:
                keywords.append(lbl)

    if combined:
        dataset_name = "tsENV"
        cite_subject = "tsENV: simulated time-series interventions across all environments"
    else:
        dataset_name = f"tsENV-{env_names[0]}"
        cite_subject = (
            f"tsENV: {env_names[0]} environment (simulated time-series interventions)"
        )

    metadata: dict[str, Any] = {
        "@context": croissant_context(),
        "@type": "sc:Dataset",
        "name": dataset_name,
        "description": " ".join(description_parts),
        "conformsTo": f"http://mlcommons.org/croissant/{CROISSANT_VERSION}",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": "https://github.com/TommasoBendinelli/tsENV",
        "version": "1.0.0",
        "datePublished": datetime.date.today().isoformat(),
        "citeAs": (
            f"Bendinelli, T. {cite_subject}. https://github.com/TommasoBendinelli/tsENV"
        ),
        "distribution": distribution,
        "recordSet": record_sets,
    }
    if keywords:
        metadata["keywords"] = keywords

    metadata.update(RAI_FIELDS)
    metadata.update(PROV_FIELDS)

    return metadata


def _discover_envs() -> list[str]:
    return sorted(p.name for p in QUESTIONS_DIR.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "env",
        nargs="?",
        help="Environment name, e.g. BallDrop. Omit and pass --all to combine every environment.",
    )
    parser.add_argument(
        "--all",
        dest="all_envs",
        action="store_true",
        help="Build a single Croissant file covering every environment under tsENV_questions/.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (defaults: <env>.croissant.json or tsENV.croissant.json).",
    )
    args = parser.parse_args()

    if args.all_envs and args.env:
        parser.error("Pass either an env name OR --all, not both.")
    if not args.all_envs and not args.env:
        parser.error("Pass an env name or --all.")

    if args.all_envs:
        env_names = _discover_envs()
        if not env_names:
            parser.error(f"No environments discovered under {QUESTIONS_DIR}.")
        default_output = "tsENV.croissant.json"
    else:
        env_names = [args.env]
        default_output = f"{args.env}.croissant.json"

    croissant = build_croissant(env_names)
    output = args.output or Path(__file__).parent / default_output
    output.write_text(json.dumps(croissant, indent=2))
    print(f"Wrote Croissant metadata for {env_names} to {output}")


if __name__ == "__main__":
    main()
