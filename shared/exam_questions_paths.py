"""Shared helpers for enforcing exam_questions output locations."""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TSENV_QUESTIONS_ROOT = REPO_ROOT / "tsENV_questions"
EXAM_QUESTIONS_COMPLETE_ROOT = REPO_ROOT / "exam_questions_complete"
EXAM_QUESTIONS_READY_ROOT = REPO_ROOT / "exam_questions_ready"
EXAM_QUESTIONS_TO_BE_CHECKED_ROOT = REPO_ROOT / "exam_questions_to_be_checked"
EXAM_QUESTIONS_SEPARABLE_ROOT = REPO_ROOT / "exam_questions_separable"
EXAM_QUESTIONS_NONE_AND_PICTURE_ROOT = REPO_ROOT / "exam_question_none_and_picture"
EXAM_QUESTIONS_LITE_ROOT = REPO_ROOT / "exam_questions_lite"
TSENV_QUESTIONS_ROOT = REPO_ROOT / "tsENV_questions"
EXAM_QUESTIONS_ROOTS = (
    TSENV_QUESTIONS_ROOT,
    EXAM_QUESTIONS_COMPLETE_ROOT,
    EXAM_QUESTIONS_READY_ROOT,
    EXAM_QUESTIONS_TO_BE_CHECKED_ROOT,
    EXAM_QUESTIONS_SEPARABLE_ROOT,
    EXAM_QUESTIONS_NONE_AND_PICTURE_ROOT,
    EXAM_QUESTIONS_LITE_ROOT,
    TSENV_QUESTIONS_ROOT,
)
EXAM_QUESTIONS_COMPLETE_TMP_ROOT = REPO_ROOT / "tmp" / "exam_questions_complete"
EXAM_QUESTIONS_READY_TMP_ROOT = REPO_ROOT / "tmp" / "exam_questions_ready"
EXAM_QUESTIONS_TO_BE_CHECKED_TMP_ROOT = REPO_ROOT / "tmp" / "exam_questions_to_be_checked"
EXAM_QUESTIONS_SEPARABLE_TMP_ROOT = REPO_ROOT / "tmp" / "exam_questions_separable"
EXAM_QUESTIONS_NONE_AND_PICTURE_TMP_ROOT = REPO_ROOT / "tmp" / "exam_question_none_and_picture"
EXAM_QUESTIONS_LITE_TMP_ROOT = REPO_ROOT / "tmp" / "exam_questions_lite"
TSENV_QUESTIONS_TMP_ROOT = REPO_ROOT / "tmp" / "tsENV_questions"

EXAM_QUESTIONS_PREFIX = "exam_questions"
EXAM_QUESTIONS_VARIANTS = (
    "separable",
    "none_and_picture",
    "ready",
    "to_be_checked",
    "lite",
    "complete",
    "tsenv",
)
EXAM_QUESTIONS_VARIANT_ROOTS = {
    "complete": EXAM_QUESTIONS_COMPLETE_ROOT,
    "ready": EXAM_QUESTIONS_READY_ROOT,
    "to_be_checked": EXAM_QUESTIONS_TO_BE_CHECKED_ROOT,
    "separable": EXAM_QUESTIONS_SEPARABLE_ROOT,
    "none_and_picture": EXAM_QUESTIONS_NONE_AND_PICTURE_ROOT,
    "lite": EXAM_QUESTIONS_LITE_ROOT,
    "tsenv": TSENV_QUESTIONS_ROOT,
}
EXAM_ROOT_NAME_TO_ROOT = {root.name.lower(): root for root in EXAM_QUESTIONS_ROOTS}
EXAM_ROOT_NAME_TO_VARIANT = {
    root.name.lower(): variant for variant, root in EXAM_QUESTIONS_VARIANT_ROOTS.items()
}

logger = logging.getLogger(__name__)


def _log_unknown_variant(variant: str, context: str) -> None:
    logger.warning("Unknown exam questions variant in %s: %s", context, variant)


def _log_invalid_output_dir(path: Path) -> None:
    logger.warning("Exam questions output dir not under an allowed variant: %s", path)


def _normalize_exam_questions_segment(segment: str) -> str:
    return segment.lower().replace("-", "_")


def _normalize_for_path_matching(value: object) -> str:
    return str(value).replace("\\", "/").strip().lower()


def _find_exam_root_from_text(text: str) -> Path | None:
    if not text:
        return None
    normalized = text.replace("\\", "/").lower()
    for root in EXAM_QUESTIONS_ROOTS:
        root_path = root.expanduser().resolve().as_posix().lower()
        if normalized == root_path or normalized.startswith(f"{root_path}/"):
            return root
    for root_name, root in EXAM_ROOT_NAME_TO_ROOT.items():
        if re.search(rf"(?:^|/){re.escape(root_name)}(?:/|$)", normalized):
            return root
    return None


def extract_exam_questions_segment(value: object) -> str | None:
    if value is None:
        return None
    root = _find_exam_root_from_text(_normalize_for_path_matching(value))
    if root is None:
        return None
    return root.name


def exam_questions_variant_from_segment(segment: str) -> str | None:
    normalized_segment = _normalize_exam_questions_segment(segment)
    root_variant = EXAM_ROOT_NAME_TO_VARIANT.get(normalized_segment)
    if root_variant is not None:
        return root_variant
    prefix = f"{EXAM_QUESTIONS_PREFIX}_"
    if normalized_segment.startswith(prefix):
        variant = normalized_segment[len(prefix) :]
    elif normalized_segment.startswith("exam_question_"):
        variant = normalized_segment[len("exam_question_") :]
    else:
        return None
    if variant not in EXAM_QUESTIONS_VARIANTS:
        _log_unknown_variant(variant, "segment")
        return None
    return variant


def detect_exam_questions_variant(
    *values: object,
) -> str | None:
    for value in values:
        root = _find_exam_root_from_text(_normalize_for_path_matching(value))
        if root is not None:
            variant = EXAM_ROOT_NAME_TO_VARIANT.get(root.name.lower())
            if variant is not None:
                return variant
        segment = extract_exam_questions_segment(value)
        if segment is not None:
            variant = exam_questions_variant_from_segment(segment)
            if variant is not None:
                return variant
    return None


def resolve_exam_questions_root_from_values(*values: object) -> Path | None:
    for value in values:
        root = _find_exam_root_from_text(_normalize_for_path_matching(value))
        if root is not None:
            return root
        segment = extract_exam_questions_segment(value)
        if segment is None:
            continue
        variant = exam_questions_variant_from_segment(segment)
        if variant is None:
            continue
        return EXAM_QUESTIONS_VARIANT_ROOTS[variant]
    return None


def resolve_exam_questions_root_for_variant(
    variant: str,
    *,
    allow_ready_fallback: bool = True,
) -> Path:
    if not variant:
        raise ValueError("variant must be non-empty")
    if variant not in EXAM_QUESTIONS_VARIANTS:
        _log_unknown_variant(variant, "resolve_exam_questions_root_for_variant")
        raise ValueError(
            f"Unknown exam questions variant: {variant}. "
            "Either create a new variant or change the output directory."
        )
    root = EXAM_QUESTIONS_VARIANT_ROOTS[variant]
    if variant == "ready" and allow_ready_fallback and not root.exists():
        return EXAM_QUESTIONS_COMPLETE_ROOT
    return root


def available_exam_questions_variants(*, base_variant: str = "ready") -> tuple[str, ...]:
    if base_variant is not None and base_variant not in EXAM_QUESTIONS_VARIANTS:
        _log_unknown_variant(base_variant, "available_exam_questions_variants")
        raise ValueError(f"Unknown base_variant: {base_variant}")
    variants = []
    for variant in EXAM_QUESTIONS_VARIANTS:
        root = EXAM_QUESTIONS_VARIANT_ROOTS.get(variant)
        if root is not None and root.exists():
            variants.append(variant)
    if base_variant and base_variant not in variants:
        variants.append(base_variant)
    return tuple(sorted(variants))


def _resolve_exam_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    for root in EXAM_QUESTIONS_ROOTS:
        exam_root = root.resolve()
        if resolved == exam_root or exam_root in resolved.parents:
            return exam_root
    _log_invalid_output_dir(resolved)
    raise ValueError(
        "Output dir must live under one of: "
        f"{', '.join(str(r) for r in EXAM_QUESTIONS_ROOTS)}; got {resolved}. "
        "Either create a new variant or change the output directory."
    )


def _resolve_tmp_root(exam_root: Path) -> Path:
    exam_root = exam_root.resolve()
    if exam_root == TSENV_QUESTIONS_ROOT.resolve():
        return TSENV_QUESTIONS_TMP_ROOT
    if exam_root == EXAM_QUESTIONS_COMPLETE_ROOT.resolve():
        return EXAM_QUESTIONS_COMPLETE_TMP_ROOT
    if exam_root == EXAM_QUESTIONS_READY_ROOT.resolve():
        return EXAM_QUESTIONS_READY_TMP_ROOT
    if exam_root == EXAM_QUESTIONS_TO_BE_CHECKED_ROOT.resolve():
        return EXAM_QUESTIONS_TO_BE_CHECKED_TMP_ROOT
    if exam_root == EXAM_QUESTIONS_SEPARABLE_ROOT.resolve():
        return EXAM_QUESTIONS_SEPARABLE_TMP_ROOT
    if exam_root == EXAM_QUESTIONS_NONE_AND_PICTURE_ROOT.resolve():
        return EXAM_QUESTIONS_NONE_AND_PICTURE_TMP_ROOT
    if exam_root == EXAM_QUESTIONS_LITE_ROOT.resolve():
        return EXAM_QUESTIONS_LITE_TMP_ROOT
    if exam_root == TSENV_QUESTIONS_ROOT.resolve():
        return TSENV_QUESTIONS_TMP_ROOT
    raise ValueError(f"Unknown exam root: {exam_root}")


def resolve_exam_questions_output_dir(output_dir: Path, model_name: str) -> Path:
    if not model_name:
        raise ValueError("model_name must be non-empty")

    resolved = output_dir.expanduser().resolve()
    exam_root = _resolve_exam_root(resolved)

    if resolved == exam_root:
        return exam_root / model_name
    if exam_root in resolved.parents:
        if resolved.name == model_name:
            return resolved
        return resolved / model_name
    raise ValueError(f"Output dir must live under {exam_root}; got {resolved}")


def resolve_exam_questions_tmp_dir(output_dir: Path) -> Path:
    resolved = output_dir.expanduser().resolve()
    exam_root = _resolve_exam_root(resolved)
    if resolved == exam_root:
        raise ValueError("Output dir must be a model-specific directory under the exam root.")
    if exam_root not in resolved.parents:
        raise ValueError(f"Output dir must live under {exam_root}; got {resolved}")
    rel_path = resolved.relative_to(exam_root)
    return _resolve_tmp_root(exam_root) / rel_path


def prepare_exam_questions_tmp_dir(output_dir: Path) -> Path:
    tmp_dir = resolve_exam_questions_tmp_dir(output_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


def commit_exam_questions_output(
    staging_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    staging_dir = staging_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not staging_dir.exists():
        raise FileNotFoundError(f"Staging directory not found: {staging_dir}")
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        shutil.copytree(
            staging_dir,
            output_dir,
            dirs_exist_ok=True,
            symlinks=True,
        )
        shutil.rmtree(staging_dir)
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging_dir), str(output_dir))


def resolve_exam_questions_root_for_path(path: Path) -> Path:
    """Return the exam_questions_* root that contains `path`."""
    return _resolve_exam_root(path)


def model_key_from_dir(model_dir: Path) -> str:
    """Return the model key (path relative to the exam root)."""
    resolved = model_dir.expanduser().resolve()
    exam_root = resolve_exam_questions_root_for_path(resolved).resolve()
    return str(resolved.relative_to(exam_root))


def parent_variant_key_from_dir(model_dir: Path) -> str:
    """Return the key of the directory that owns real (non-symlink) dataframes/."""
    resolved = model_dir.expanduser().resolve()
    exam_root = resolve_exam_questions_root_for_path(resolved).resolve()
    current = resolved
    visited: set[Path] = set()

    while True:
        if current in visited:
            break
        visited.add(current)

        dataframes_dir = current / "dataframes"
        if dataframes_dir.is_symlink():
            target = dataframes_dir.resolve()
            current = target.parent
            continue
        break

    try:
        return str(current.relative_to(exam_root))
    except ValueError:
        return current.name
