#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from shared.tsenv_metadata import load_metadata_payload, metadata_questions_list


DEFAULT_BOOKMARKS = Path("web_human_study/backend/web_human_study/backend/bookmark.json")
DEFAULT_EXAM_ROOT = Path("exam_questions_separable")
DEFAULT_OUTPUT_DIR = Path("overleaf_paper/plots")


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _load_bookmark_hashes(path: Path, nickname: Optional[str]) -> List[str]:
    with path.open("r") as f:
        payload = json.load(f)
    hashes: List[str] = []
    def add_entry(entry: object) -> None:
        if isinstance(entry, dict):
            value = entry.get("question_id")
        else:
            value = entry
        text = str(value or "").strip()
        if text:
            hashes.append(text)
    if isinstance(payload, dict):
        if nickname:
            entries = payload.get(nickname, [])
            if isinstance(entries, list):
                for entry in entries:
                    add_entry(entry)
        else:
            for entries in payload.values():
                if isinstance(entries, list):
                    for entry in entries:
                        add_entry(entry)
    elif isinstance(payload, list):
        for entry in payload:
            add_entry(entry)
    else:
        raise TypeError("bookmark.json must be a dict or list")
    return _dedupe_keep_order(hashes)


def _flatten_column_name(col: object) -> str:
    parts: List[str] = []
    if isinstance(col, tuple):
        parts = [str(p).strip() for p in col if str(p).strip()]
    else:
        raw = str(col).strip()
        if raw.startswith("(") and raw.endswith(")") and "," in raw:
            raw = raw[1:-1]
            parts = [p.strip().strip("'\"") for p in raw.split(",")]
        else:
            parts = [raw]
    parts = [p for p in parts if p and p.lower() != "nan"]
    if not parts:
        return "value"
    if len(parts) == 1:
        val = parts[0]
        return "time" if val.lower().startswith("time") else val
    if parts[0].lower().startswith("time"):
        return "time"
    return " - ".join(parts)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flattened: List[str] = []
    seen: Dict[str, int] = {}
    for col in df.columns:
        flat = _flatten_column_name(col)
        if flat in seen:
            seen[flat] += 1
            flat = f"{flat} ({seen[flat]})"
        else:
            seen[flat] = 0
        flattened.append(flat)
    df.columns = flattened
    time_cols = [c for c in df.columns if c.lower() == "time"]
    if time_cols:
        time_col = time_cols[0]
        remaining = [c for c in df.columns if c != time_col]
        df = df[[time_col, *remaining]]
    return df


def _get_time_column(df: pd.DataFrame) -> Optional[str]:
    return next((c for c in df.columns if c.lower() == "time"), None)


def _safe_name(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _normalize_label_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _append_units(label: str) -> str:
    if "(" in label:
        return label
    normalized = _normalize_label_key(label)
    if normalized == "position":
        return "Position (m)"
    if normalized == "velocity":
        return "Velocity (m/s)"
    if normalized == "hardstopf" or normalized == "impulseforce":
        return "Impulse Force (N)"
    return label


def _read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".pkl", ".pickle", ".p"):
        return pd.read_pickle(path)
    return pd.read_parquet(path)


def _parse_signals(raw_value: object) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if str(v).strip()]
    if isinstance(raw_value, tuple):
        return [str(v).strip() for v in raw_value if str(v).strip()]
    if isinstance(raw_value, str):
        return [s.strip() for s in raw_value.split(",") if s.strip()]
    return []


def _iter_metadata_paths(exam_root: Path) -> Iterable[Path]:
    return sorted(p for p in exam_root.rglob("questions.json") if p.is_file())


def _find_questions(
    exam_root: Path,
    target_hashes: Iterable[str],
) -> Dict[str, Tuple[dict, Path]]:
    targets = set(target_hashes)
    matches: Dict[str, Tuple[dict, Path]] = {}
    for metadata_path in _iter_metadata_paths(exam_root):
        try:
            payload = load_metadata_payload(metadata_path)
        except Exception:
            continue
        questions = metadata_questions_list(payload)
        for question in questions:
            if not isinstance(question, dict):
                continue
            question_id = question.get("question_id")
            if question_id in targets and question_id not in matches:
                matches[question_id] = (question, metadata_path.parent)
        if len(matches) == len(targets):
            break
    return matches


def _candidate_paths(
    filename: str,
    model_dir: Path,
    exam_root: Path,
    parent_variant_key: Optional[str],
) -> Iterable[Path]:
    path = Path(filename)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
        candidates.append(model_dir / path.name)
    else:
        candidates.append(model_dir / path)
        candidates.append(model_dir / "dataframes" / path)
        candidates.append(model_dir / "dataframes" / path.name)
    if parent_variant_key:
        parent_dir = exam_root / parent_variant_key
        if parent_dir.exists():
            candidates.append(parent_dir / path)
            candidates.append(parent_dir / "dataframes" / path)
            candidates.append(parent_dir / "dataframes" / path.name)
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _resolve_dataframe_path(
    question: dict,
    model_dir: Path,
    exam_root: Path,
) -> Optional[Path]:
    test_samples = question.get("test_samples")
    filename = None
    if isinstance(test_samples, list):
        for item in test_samples:
            text = str(item or "").strip()
            if text:
                filename = text
                break
    if not filename:
        filename = question.get("dataframe_name")
    if not filename:
        return None
    parent_variant_key = None
    if isinstance(question.get("parent_variant_key"), str):
        parent_variant_key = question["parent_variant_key"].strip() or None
    for candidate in _candidate_paths(str(filename), model_dir, exam_root, parent_variant_key):
        if candidate.exists():
            return candidate
    return None


def _extract_plot_frame(
    question: dict,
    dataframe_path: Path,
) -> Tuple[pd.DataFrame, Optional[str], List[str], Optional[Dict[str, str]], float]:
    df = _read_dataframe(dataframe_path)
    df = _normalize_dataframe(df)
    time_col = next((c for c in df.columns if c.lower() == "time"), None)
    dataset = str(question.get("dataset") or "").strip()
    question_id = str(question.get("question_id") or "").strip()
    label_map = None
    time_offset = 0.0
    if dataset == "DampedMassBetweenWalls":
        if time_col is None:
            raise ValueError(
                f"Missing time column for DampedMassBetweenWalls question_id={question.get('question_id')}"
            )
        df = df[df[time_col] <= 20]
        label_map = {
            "ballcenterposition": "Ball Position (m)",
            "ballcenterspeed": "Ball Speed (m/s)",
        }
    elif dataset == "BallDrop":
        if time_col is None:
            raise ValueError(
                f"Missing time column for BallDrop question_id={question.get('question_id')}"
            )
        df = df[df[time_col] <= 8].copy()
    elif dataset == "InclinedPlane":
        if time_col is None:
            raise ValueError(
                f"Missing time column for InclinedPlane question_id={question.get('question_id')}"
            )
        df = df[df[time_col] >= 2].copy()
        df[time_col] = df[time_col] - 2
        time_offset = 2.0
        label_map = {
            "massvelocity": "Mass Velocity (m/s)",
            "frictionforce": "Friction Force (N)",
            "normalforce": "Normal Force (N)",
        }
    elif dataset == "DoubleMassSpringDamperSameCoeffs":
        label_map = {
            "positionmass1": "Position Mass 1 (m)",
            "positionmass2": "Position Mass 2 (m)",
        }
    elif len(df) > 1 and question_id != "cls__BallDrop__0__drag_coeff_000":
        df = df.iloc[: max(1, len(df) // 2)]
    signal_cols = [c for c in df.columns if c != time_col]
    signals = _parse_signals(question.get("signals"))
    if signals:
        filtered = [c for c in signal_cols if c in signals]
        if filtered:
            signal_cols = filtered
    signal_cols = [c for c in signal_cols if pd.api.types.is_numeric_dtype(df[c])]
    return df, time_col, signal_cols, label_map, time_offset


def _coerce_first_detectable_time(question: dict) -> float:
    value = question.get("first_detectable_time")
    if value is None or value == "":
        raise ValueError(
            f"Missing first_detectable_time for question_id={question.get('question_id')}"
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid first_detectable_time for question_id={question.get('question_id')}"
        ) from exc


def _render_plot(
    df: pd.DataFrame,
    time_col: Optional[str],
    signal_cols: List[str],
    first_detectable_time: float,
    output_base: Path,
    label_map: Optional[Dict[str, str]],
) -> None:
    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.0,
        rc={
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )
    if len(signal_cols) > 1:
        fig_height = max(2.6, 1.1 * len(signal_cols) + 1.0) * 1.5
        fig, axes = plt.subplots(
            nrows=len(signal_cols),
            ncols=1,
            sharex=True,
            figsize=(9.4, fig_height),
        )
        axes_list = list(axes)
    else:
        fig, ax = plt.subplots(figsize=(9.4, 5.1))
        axes_list = [ax]
    if time_col:
        x_values = df[time_col]
    else:
        x_values = df.index
    palette = sns.color_palette("tab10", n_colors=max(1, len(signal_cols)))
    for idx, col in enumerate(signal_cols):
        ax = axes_list[idx]
        sns.lineplot(
            x=x_values,
            y=df[col],
            ax=ax,
            color=palette[idx % len(palette)],
            zorder=3,
        )
        ax.set_xlabel("")
        label = col
        if label_map:
            mapped = label_map.get(_normalize_label_key(col))
            if mapped:
                label = mapped
        ax.set_ylabel(_append_units(label), fontsize=13)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
        ax.grid(True, which="minor", linewidth=0.2, alpha=0.25)
        ax.axvline(
            first_detectable_time,
            color="#8f8f8f",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            zorder=1,
        )
        if ax.get_legend():
            ax.get_legend().remove()
        ax.margins(x=0)
        sns.despine(ax=ax)
    for ax in axes_list[:-1]:
        ax.set_xlabel("")
    x_label = "Time (s)" if time_col else "Index"
    axes_list[-1].set_xlabel(x_label, fontsize=13)
    if len(axes_list) > 1:
        fig.align_ylabels(axes_list)
        fig.subplots_adjust(hspace=0.1)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _ball_drop_run_id(question: dict, dataframe_path: Path) -> Optional[str]:
    for value in (question.get("question_id"), dataframe_path.name):
        if isinstance(value, str):
            match = re.search(r"cls__BallDrop__(\d+)__", value)
            if match:
                return match.group(1)
    return None


def _ball_drop_intervention_label(path: Path) -> Optional[str]:
    name = path.stem
    if "drag_coeff" in name:
        return "Drag coef."
    if "gravity" in name:
        return "Gravity"
    if "restitution" in name:
        return "Restitution"
    return None


def _ball_between_walls_run_id(question: dict, dataframe_path: Path) -> Optional[str]:
    for value in (question.get("question_id"), dataframe_path.name):
        if isinstance(value, str):
            match = re.search(r"cls__DampedMassBetweenWalls__(\d+)__", value)
            if match:
                return match.group(1)
    return None


def _ball_between_walls_intervention_label(path: Path) -> Optional[str]:
    name = path.stem
    if "restitution_left" in name:
        return "Restitution left"
    if "restitution_right" in name:
        return "Restitution right"
    if "viscous_damping" in name:
        return "Viscous damping"
    return None


def _ball_between_walls_overlay_frames(
    question: dict,
    dataframe_path: Path,
) -> List[Tuple[str, pd.DataFrame]]:
    run_id = _ball_between_walls_run_id(question, dataframe_path)
    if not run_id:
        return []
    frames: List[Tuple[str, pd.DataFrame]] = []
    pattern = f"cls__DampedMassBetweenWalls__{run_id}__*.parquet"
    for path in sorted(dataframe_path.parent.glob(pattern)):
        if path.resolve() == dataframe_path.resolve():
            continue
        label = _ball_between_walls_intervention_label(path)
        if not label:
            continue
        df = _normalize_dataframe(_read_dataframe(path))
        time_col = _get_time_column(df)
        if time_col:
            df = df[df[time_col] <= 20].copy()
        frames.append((label, df))
    order = {"Restitution left": 0, "Restitution right": 1, "Viscous damping": 2}
    frames.sort(key=lambda item: order.get(item[0], 99))
    return frames


def _ball_drop_overlay_frames(
    question: dict,
    dataframe_path: Path,
) -> List[Tuple[str, pd.DataFrame]]:
    run_id = _ball_drop_run_id(question, dataframe_path)
    if not run_id:
        return []
    frames: List[Tuple[str, pd.DataFrame]] = []
    pattern = f"cls__BallDrop__{run_id}__*.parquet"
    for path in sorted(dataframe_path.parent.glob(pattern)):
        if path.resolve() == dataframe_path.resolve():
            continue
        label = _ball_drop_intervention_label(path)
        if not label:
            continue
        df = _normalize_dataframe(_read_dataframe(path))
        time_col = _get_time_column(df)
        if time_col:
            df = df[df[time_col] <= 8].copy()
        frames.append((label, df))
    order = {"Gravity": 0, "Drag coef.": 1, "Restitution": 2}
    frames.sort(key=lambda item: order.get(item[0], 99))
    return frames


def _inclined_plane_run_id(question: dict, dataframe_path: Path) -> Optional[str]:
    for value in (question.get("question_id"), dataframe_path.name):
        if isinstance(value, str):
            match = re.search(r"cls__InclinedPlane__(\d+)__", value)
            if match:
                return match.group(1)
    return None


def _inclined_plane_intervention_label(path: Path) -> Optional[str]:
    name = path.stem
    if "gravity_constant" in name:
        return "Gravity"
    if "plane_inclination" in name:
        return "Plane inclination"
    if "coulomb_friction" in name:
        return "Coulomb friction"
    return None


def _inclined_plane_overlay_frames(
    question: dict,
    dataframe_path: Path,
) -> List[Tuple[str, pd.DataFrame]]:
    run_id = _inclined_plane_run_id(question, dataframe_path)
    if not run_id:
        return []
    frames: List[Tuple[str, pd.DataFrame]] = []
    pattern = f"cls__InclinedPlane__{run_id}__*.parquet"
    for path in sorted(dataframe_path.parent.glob(pattern)):
        if path.resolve() == dataframe_path.resolve():
            continue
        label = _inclined_plane_intervention_label(path)
        if not label:
            continue
        df = _normalize_dataframe(_read_dataframe(path))
        time_col = _get_time_column(df)
        if time_col:
            df = df[df[time_col] >= 2].copy()
            df[time_col] = df[time_col] - 2
        frames.append((label, df))
    order = {"Gravity": 0, "Plane inclination": 1, "Coulomb friction": 2}
    frames.sort(key=lambda item: order.get(item[0], 99))
    return frames


def _double_mass_spring_run_id(question: dict, dataframe_path: Path) -> Optional[str]:
    for value in (question.get("question_id"), dataframe_path.name):
        if isinstance(value, str):
            match = re.search(r"cls__DoubleMassSpringDamperSameCoeffs__(\d+)__", value)
            if match:
                return match.group(1)
    return None


def _double_mass_spring_intervention_label(path: Path) -> Optional[str]:
    name = path.stem
    if "spring_constant" in name:
        return "Spring constant"
    if "damping_constant" in name:
        return "Damping constant"
    if re.search(r"__\d+_mass_", name):
        return "Mass"
    return None


def _double_mass_spring_overlay_frames(
    question: dict,
    dataframe_path: Path,
) -> List[Tuple[str, pd.DataFrame]]:
    run_id = _double_mass_spring_run_id(question, dataframe_path)
    if not run_id:
        return []
    frames: List[Tuple[str, pd.DataFrame]] = []
    pattern = f"cls__DoubleMassSpringDamperSameCoeffs__{run_id}__*.parquet"
    for path in sorted(dataframe_path.parent.glob(pattern)):
        if path.resolve() == dataframe_path.resolve():
            continue
        label = _double_mass_spring_intervention_label(path)
        if not label:
            continue
        df = _normalize_dataframe(_read_dataframe(path))
        frames.append((label, df))
    order = {"Spring constant": 0, "Damping constant": 1, "Mass": 2}
    frames.sort(key=lambda item: order.get(item[0], 99))
    return frames


def _render_ball_between_walls_plot(
    question: dict,
    df: pd.DataFrame,
    time_col: Optional[str],
    signal_cols: List[str],
    first_detectable_time: float,
    output_base: Path,
    label_map: Optional[Dict[str, str]],
    dataframe_path: Path,
) -> None:
    overlays = _ball_between_walls_overlay_frames(question, dataframe_path)
    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.0,
        rc={
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )
    if len(signal_cols) > 1:
        fig_height = max(2.6, 1.1 * len(signal_cols) + 1.0) * 1.5
        fig, axes = plt.subplots(
            nrows=len(signal_cols),
            ncols=1,
            sharex=True,
            figsize=(9.4, fig_height),
        )
        axes_list = list(axes)
    else:
        fig, ax = plt.subplots(figsize=(9.4, 5.1))
        axes_list = [ax]
    if time_col:
        x_values = df[time_col]
    else:
        x_values = df.index
    base_palette = sns.color_palette("tab10", n_colors=max(1, len(signal_cols)))
    palette = sns.color_palette("tab10", 5)
    color_map = {
        "Restitution left": palette[2],
        "Restitution right": palette[3],
        "Viscous damping": palette[4],
    }
    viscous_linewidth = 2.0
    primary_label = _ball_between_walls_intervention_label(dataframe_path)
    for idx, col in enumerate(signal_cols):
        ax = axes_list[idx]
        if primary_label:
            pre_mask = x_values <= first_detectable_time
            post_mask = x_values >= first_detectable_time
            ax.plot(
                x_values[pre_mask],
                df[col][pre_mask],
                color=base_palette[idx % len(base_palette)],
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )
            ax.plot(
                x_values[post_mask],
                df[col][post_mask],
                color=color_map.get(primary_label, palette[2]),
                linestyle="-",
                linewidth=viscous_linewidth if primary_label == "Viscous damping" else 1.2,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                df[col],
                color=base_palette[idx % len(base_palette)],
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )
        for label, overlay_df in overlays:
            overlay_time = _get_time_column(overlay_df)
            if overlay_time:
                overlay_df = overlay_df[overlay_df[overlay_time] >= first_detectable_time]
                x_overlay = overlay_df[overlay_time]
            else:
                x_overlay = overlay_df.index
            if col not in overlay_df.columns:
                continue
            ax.plot(
                x_overlay,
                overlay_df[col],
                color=color_map.get(label, palette[2]),
                linestyle="-",
                linewidth=viscous_linewidth if label == "Viscous damping" else 1.2,
                zorder=2,
            )
        ax.set_xlabel("")
        label = col
        if label_map:
            mapped = label_map.get(_normalize_label_key(col))
            if mapped:
                label = mapped
        ax.set_ylabel(_append_units(label), fontsize=13)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
        ax.grid(True, which="minor", linewidth=0.2, alpha=0.25)
        ax.axvline(
            first_detectable_time,
            color="#8f8f8f",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            zorder=1,
        )
        ax.margins(x=0)
        sns.despine(ax=ax)
    for ax in axes_list[:-1]:
        ax.set_xlabel("")
    x_label = "Time (s)" if time_col else "Index"
    axes_list[-1].set_xlabel(x_label, fontsize=13)
    if len(axes_list) > 1:
        fig.align_ylabels(axes_list)
        fig.subplots_adjust(hspace=0.1)
    handles = [
        Line2D([0], [0], color=color_map["Restitution left"], linestyle="-", linewidth=1.2),
        Line2D([0], [0], color=color_map["Restitution right"], linestyle="-", linewidth=1.2),
        Line2D([0], [0], color=color_map["Viscous damping"], linestyle="-", linewidth=viscous_linewidth),
    ]
    labels = ["Restitution left", "Restitution right", "Viscous damping"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
        borderaxespad=0.0,
    )
    legend = fig.legends[-1]
    for text in legend.get_texts():
        if text.get_text() == "Viscous damping":
            text.set_fontweight("bold")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _render_inclined_plane_plot(
    question: dict,
    df: pd.DataFrame,
    time_col: Optional[str],
    signal_cols: List[str],
    first_detectable_time: float,
    output_base: Path,
    label_map: Optional[Dict[str, str]],
    dataframe_path: Path,
) -> None:
    overlays = _inclined_plane_overlay_frames(question, dataframe_path)
    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.0,
        rc={
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )
    if len(signal_cols) > 1:
        fig_height = max(2.6, 1.1 * len(signal_cols) + 1.0) * 1.5
        fig, axes = plt.subplots(
            nrows=len(signal_cols),
            ncols=1,
            sharex=True,
            figsize=(9.4, fig_height),
        )
        axes_list = list(axes)
    else:
        fig, ax = plt.subplots(figsize=(9.4, 5.1))
        axes_list = [ax]
    if time_col:
        x_values = df[time_col]
    else:
        x_values = df.index
    base_palette = sns.color_palette("tab10", n_colors=max(1, len(signal_cols)))
    palette = sns.color_palette("tab10", 5)
    color_map = {
        "Gravity": palette[2],
        "Plane inclination": palette[3],
        "Coulomb friction": palette[4],
    }
    gravity_linewidth = 2.0
    primary_label = _inclined_plane_intervention_label(dataframe_path)
    for idx, col in enumerate(signal_cols):
        ax = axes_list[idx]
        if primary_label:
            pre_mask = x_values <= first_detectable_time
            post_mask = x_values >= first_detectable_time
            ax.plot(
                x_values[pre_mask],
                df[col][pre_mask],
                color=base_palette[idx % len(base_palette)],
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )
            ax.plot(
                x_values[post_mask],
                df[col][post_mask],
                color=color_map.get(primary_label, palette[2]),
                linestyle="-",
                linewidth=gravity_linewidth if primary_label == "Gravity" else 1.2,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                df[col],
                color=base_palette[idx % len(base_palette)],
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )
        for label, overlay_df in overlays:
            overlay_time = _get_time_column(overlay_df)
            if overlay_time:
                overlay_df = overlay_df[overlay_df[overlay_time] >= first_detectable_time]
                x_overlay = overlay_df[overlay_time]
            else:
                x_overlay = overlay_df.index
            if col not in overlay_df.columns:
                continue
            ax.plot(
                x_overlay,
                overlay_df[col],
                color=color_map.get(label, palette[2]),
                linestyle="-",
                linewidth=gravity_linewidth if label == "Gravity" else 1.2,
                zorder=2,
            )
        ax.set_xlabel("")
        label = col
        if label_map:
            mapped = label_map.get(_normalize_label_key(col))
            if mapped:
                label = mapped
        ax.set_ylabel(_append_units(label), fontsize=13)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
        ax.grid(True, which="minor", linewidth=0.2, alpha=0.25)
        ax.axvline(
            first_detectable_time,
            color="#8f8f8f",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            zorder=1,
        )
        ax.margins(x=0)
        sns.despine(ax=ax)
    for ax in axes_list[:-1]:
        ax.set_xlabel("")
    x_label = "Time (s)" if time_col else "Index"
    axes_list[-1].set_xlabel(x_label, fontsize=13)
    if len(axes_list) > 1:
        fig.align_ylabels(axes_list)
        fig.subplots_adjust(hspace=0.1)
    handles = [
        Line2D([0], [0], color=color_map["Gravity"], linestyle="-", linewidth=gravity_linewidth),
        Line2D([0], [0], color=color_map["Plane inclination"], linestyle="-", linewidth=1.2),
        Line2D([0], [0], color=color_map["Coulomb friction"], linestyle="-", linewidth=1.2),
    ]
    labels = ["Gravity", "Plane inclination", "Coulomb friction"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
        borderaxespad=0.0,
    )
    legend = fig.legends[-1]
    for text in legend.get_texts():
        if text.get_text() == "Gravity":
            text.set_fontweight("bold")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _render_double_mass_spring_plot(
    question: dict,
    df: pd.DataFrame,
    time_col: Optional[str],
    signal_cols: List[str],
    first_detectable_time: float,
    output_base: Path,
    label_map: Optional[Dict[str, str]],
    dataframe_path: Path,
) -> None:
    overlays = _double_mass_spring_overlay_frames(question, dataframe_path)
    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.0,
        rc={
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )
    if len(signal_cols) > 1:
        fig_height = max(2.6, 1.1 * len(signal_cols) + 1.0) * 1.5
        fig, axes = plt.subplots(
            nrows=len(signal_cols),
            ncols=1,
            sharex=True,
            figsize=(9.4, fig_height),
        )
        axes_list = list(axes)
    else:
        fig, ax = plt.subplots(figsize=(9.4, 5.1))
        axes_list = [ax]
    if time_col:
        x_values = df[time_col]
    else:
        x_values = df.index
    base_palette = sns.color_palette("tab10", n_colors=max(1, len(signal_cols)))
    palette = sns.color_palette("tab10", 5)
    color_map = {
        "Spring constant": palette[2],
        "Damping constant": palette[3],
        "Mass": palette[4],
    }
    spring_linewidth = 2.0
    primary_label = _double_mass_spring_intervention_label(dataframe_path)
    for idx, col in enumerate(signal_cols):
        ax = axes_list[idx]
        if primary_label:
            pre_mask = x_values <= first_detectable_time
            post_mask = x_values >= first_detectable_time
            ax.plot(
                x_values[pre_mask],
                df[col][pre_mask],
                color=base_palette[idx % len(base_palette)],
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )
            ax.plot(
                x_values[post_mask],
                df[col][post_mask],
                color=color_map.get(primary_label, palette[2]),
                linestyle="-",
                linewidth=spring_linewidth if primary_label == "Spring constant" else 1.2,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                df[col],
                color=base_palette[idx % len(base_palette)],
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )
        for label, overlay_df in overlays:
            overlay_time = _get_time_column(overlay_df)
            if overlay_time:
                overlay_df = overlay_df[overlay_df[overlay_time] >= first_detectable_time]
                x_overlay = overlay_df[overlay_time]
            else:
                x_overlay = overlay_df.index
            if col not in overlay_df.columns:
                continue
            ax.plot(
                x_overlay,
                overlay_df[col],
                color=color_map.get(label, palette[2]),
                linestyle="-",
                linewidth=spring_linewidth if label == "Spring constant" else 1.2,
                zorder=2,
            )
        ax.set_xlabel("")
        label = col
        if label_map:
            mapped = label_map.get(_normalize_label_key(col))
            if mapped:
                label = mapped
        ax.set_ylabel(_append_units(label), fontsize=13)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
        ax.grid(True, which="minor", linewidth=0.2, alpha=0.25)
        ax.axvline(
            first_detectable_time,
            color="#8f8f8f",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            zorder=1,
        )
        ax.margins(x=0)
        sns.despine(ax=ax)
    for ax in axes_list[:-1]:
        ax.set_xlabel("")
    x_label = "Time (s)" if time_col else "Index"
    axes_list[-1].set_xlabel(x_label, fontsize=13)
    if len(axes_list) > 1:
        fig.align_ylabels(axes_list)
        fig.subplots_adjust(hspace=0.1)
    handles = [
        Line2D([0], [0], color=color_map["Spring constant"], linestyle="-", linewidth=spring_linewidth),
        Line2D([0], [0], color=color_map["Damping constant"], linestyle="-", linewidth=1.2),
        Line2D([0], [0], color=color_map["Mass"], linestyle="-", linewidth=1.2),
    ]
    labels = ["Spring constant", "Damping constant", "Mass"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
        borderaxespad=0.0,
    )
    legend = fig.legends[-1]
    for text in legend.get_texts():
        if text.get_text() == "Spring constant":
            text.set_fontweight("bold")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _render_ball_drop_plot(
    question: dict,
    df: pd.DataFrame,
    time_col: Optional[str],
    signal_cols: List[str],
    first_detectable_time: float,
    output_base: Path,
    label_map: Optional[Dict[str, str]],
    dataframe_path: Path,
) -> None:
    overlays = _ball_drop_overlay_frames(question, dataframe_path)
    sns.set_theme(
        context="paper",
        style="white",
        font_scale=1.0,
        rc={
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        },
    )
    if len(signal_cols) > 1:
        fig_height = max(2.6, 1.1 * len(signal_cols) + 1.0) * 1.5
        fig, axes = plt.subplots(
            nrows=len(signal_cols),
            ncols=1,
            sharex=True,
            figsize=(9.4, fig_height),
        )
        axes_list = list(axes)
    else:
        fig, ax = plt.subplots(figsize=(9.4, 5.1))
        axes_list = [ax]
    if time_col:
        x_values = df[time_col]
    else:
        x_values = df.index
    palette = sns.color_palette("tab10", 5)
    color_map = {
        "Position": palette[0],
        "Velocity": palette[1],
        "Gravity": palette[2],
        "Drag coef.": palette[3],
        "Restitution": palette[4],
    }
    drag_linewidth = 2.0
    drag_coeff_primary = _ball_drop_intervention_label(dataframe_path) == "Drag coef."
    for idx, col in enumerate(signal_cols):
        ax = axes_list[idx]
        if drag_coeff_primary:
            pre_mask = x_values <= first_detectable_time
            post_mask = x_values >= first_detectable_time
            ax.plot(
                x_values[pre_mask],
                df[col][pre_mask],
                color=color_map.get(col, palette[0]),
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )
            ax.plot(
                x_values[post_mask],
                df[col][post_mask],
                color=color_map["Drag coef."],
                linestyle="-",
                linewidth=drag_linewidth,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                df[col],
                color=color_map.get(col, palette[0]),
                linestyle=":",
                linewidth=1.2,
                zorder=4,
            )
        for label, overlay_df in overlays:
            overlay_time = _get_time_column(overlay_df)
            if overlay_time:
                overlay_df = overlay_df[overlay_df[overlay_time] >= first_detectable_time]
                x_overlay = overlay_df[overlay_time]
            else:
                x_overlay = overlay_df.index
            if col not in overlay_df.columns:
                continue
            ax.plot(
                x_overlay,
                overlay_df[col],
                color=color_map.get(label, palette[2]),
                linestyle="-",
                linewidth=drag_linewidth if label == "Drag coef." else 1.2,
                zorder=2,
            )
        ax.set_xlabel("")
        label = col
        if label_map:
            mapped = label_map.get(_normalize_label_key(col))
            if mapped:
                label = mapped
        ax.set_ylabel(_append_units(label), fontsize=13)
        ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
        ax.grid(True, which="major", linewidth=0.4, alpha=0.4)
        ax.grid(True, which="minor", linewidth=0.2, alpha=0.25)
        ax.axvline(
            first_detectable_time,
            color="#8f8f8f",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            zorder=1,
        )
        ax.margins(x=0)
        sns.despine(ax=ax)
    for ax in axes_list[:-1]:
        ax.set_xlabel("")
    x_label = "Time (s)" if time_col else "Index"
    axes_list[-1].set_xlabel(x_label, fontsize=13)
    if len(axes_list) > 1:
        fig.align_ylabels(axes_list)
        fig.subplots_adjust(hspace=0.35)
    handles = [
        Line2D([0], [0], color=color_map["Gravity"], linestyle="-", linewidth=1.2),
        Line2D([0], [0], color=color_map["Drag coef."], linestyle="-", linewidth=drag_linewidth),
        Line2D([0], [0], color=color_map["Restitution"], linestyle="-", linewidth=1.2),
    ]
    labels = ["Gravity", "Drag coef.", "Restitution"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
        borderaxespad=0.0,
    )
    legend = fig.legends[-1]
    for text in legend.get_texts():
        if text.get_text() == "Drag coef.":
            text.set_fontweight("bold")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render plots for bookmarked questions from exam_questions_separable.",
    )
    parser.add_argument("--bookmarks", type=Path, default=DEFAULT_BOOKMARKS)
    parser.add_argument("--exam-root", type=Path, default=DEFAULT_EXAM_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nickname", type=str, default=None)
    args = parser.parse_args()

    hashes = _load_bookmark_hashes(args.bookmarks, args.nickname)
    if not hashes:
        print("No bookmarks found.")
        return 1

    matches = _find_questions(args.exam_root, hashes)
    missing = [h for h in hashes if h not in matches]
    if missing:
        print(f"Missing {len(missing)} question_id entries.")
        for h in missing:
            print(f"- {h}")

    rendered = 0
    for question_id in hashes:
        if question_id not in matches:
            continue
        question, model_dir = matches[question_id]
        dataframe_path = _resolve_dataframe_path(question, model_dir, args.exam_root)
        if not dataframe_path:
            print(f"Dataframe missing for {question_id}")
            continue
        df, time_col, signal_cols, label_map, time_offset = _extract_plot_frame(
            question, dataframe_path
        )
        first_detectable_time = _coerce_first_detectable_time(question) - time_offset
        if time_col is None:
            raise ValueError(
                f"Missing time column for question_id={question.get('question_id')}"
            )
        if not signal_cols:
            print(f"No numeric signals for {question_id}")
            continue
        dataset_raw = str(question.get("dataset") or "").strip()
        dataset = _safe_name(dataset_raw)
        question_id = _safe_name(question.get("question_id"))
        output_base = args.output_dir / f"{dataset}_{question_id}"
        if dataset_raw == "BallDrop":
            _render_ball_drop_plot(
                question,
                df,
                time_col,
                signal_cols,
                first_detectable_time,
                output_base,
                label_map,
                dataframe_path,
            )
        elif dataset_raw == "DampedMassBetweenWalls":
            _render_ball_between_walls_plot(
                question,
                df,
                time_col,
                signal_cols,
                first_detectable_time,
                output_base,
                label_map,
                dataframe_path,
            )
        elif dataset_raw == "InclinedPlane":
            _render_inclined_plane_plot(
                question,
                df,
                time_col,
                signal_cols,
                first_detectable_time,
                output_base,
                label_map,
                dataframe_path,
            )
        elif dataset_raw == "DoubleMassSpringDamperSameCoeffs":
            _render_double_mass_spring_plot(
                question,
                df,
                time_col,
                signal_cols,
                first_detectable_time,
                output_base,
                label_map,
                dataframe_path,
            )
        else:
            _render_plot(
                df,
                time_col,
                signal_cols,
                first_detectable_time,
                output_base,
                label_map,
            )
        rendered += 1
        print(f"Saved {output_base.with_suffix('.png')} and .pdf")

    print(f"Rendered {rendered} plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
