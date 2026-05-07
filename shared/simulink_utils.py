"""
Simulink/Simscape specific helpers and XML cleaners.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

WANTED = {"/simulink/systems/system_root.xml"}
# Matches MATLAB identifiers: leading letter/underscore then word chars.
MATLAB_IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
PRIMARY_STOP_TIME_WORKSPACE_VARIABLE = "end_time_input_s"
LOCAL_SOLVER_STEP_SIZE_VARIABLE = "local_solver_step_size"


def _get_all_simscape_blocks(mle, model_name: str) -> List[Dict[str, Any]]:
    """Return basic metadata for all Simscape blocks in a model."""

    def _try_get_component_path(ident: str) -> Optional[str]:
        for prop in ("ComponentPath", "ReferenceBlock"):
            try:
                val = mle.get_param(ident, prop, nargout=1)
                if isinstance(val, str) and val:
                    return val
            except Exception:
                continue
        return None

    blocks: List[Dict[str, Any]] = []

    try:
        all_blocks = mle.find_system(
            model_name,
            "LookUnderMasks",
            "all",
            "FollowLinks",
            "on",
            "Type",
            "Block",
            nargout=1,
        )

        if not isinstance(all_blocks, (list, tuple)):
            all_blocks = [all_blocks]

        for block_path in all_blocks:
            if not isinstance(block_path, str):
                continue

            try:
                block_type = mle.get_param(block_path, "BlockType", nargout=1)
                if block_type in ("SimscapeBlock", "SubSystem"):
                    comp_path = _try_get_component_path(block_path)
                    if comp_path:
                        blocks.append(
                            {
                                "path": block_path,
                                "component": comp_path,
                                "block_type": block_type,
                            }
                        )
            except Exception:
                continue

    except Exception as exc:
        print(f"Error finding blocks: {exc}")

    return blocks


def get_all_blocks(
    mle, model: str = "simulink_model_original", filter_based_on_colors: Iterable[str] = ("magenta", "pink")
) -> List[Dict[str, Any]]:
    """
    Find blocks whose name/frame color is pink/magenta, classify them
    as 'simscape' or 'simulink', and (for simscape) list their parameters.
    """

    def _is_simscape_by_ports(block):
        """Simscape blocks have physical conserving ports (LConn field)."""
        try:
            ph = mle.get_param(block, "PortHandles", nargout=1)
            if isinstance(ph, dict):
                lconn = None
                for k in ph.keys():
                    if k.lower() == "lconn":
                        lconn = ph[k]
                        break
                if lconn is not None:
                    try:
                        return len(lconn) > 0
                    except Exception:
                        return False
        except Exception:
            pass
        return False

    SIMSCAPE_LIB_TOKENS = {
        "simscape",
        "sm_lib",
        "ee_lib",
        "fl_lib",
        "sdl_lib",
        "hdlib",
        "powerlib",
        "sps_lib",
    }

    def _classify(block: str):
        if _is_simscape_by_ports(block):
            return "simscape"
        try:
            ref = mle.get_param(block, "ReferenceBlock", nargout=1)
            if isinstance(ref, str) and ref:
                rl = ref.lower()
                if any(tok in rl for tok in SIMSCAPE_LIB_TOKENS):
                    return "simscape"
                if rl.startswith("simulink/") or rl.startswith("built-in/"):
                    return "simulink"
        except Exception:
            pass
        return "simulink"

    _num_unit_re = re.compile(r'^\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)\\s*(.*?)\\s*$')

    def _simscape_parameters(block):
        out = []
        try:
            dp = mle.get_param(block, "DialogParameters", nargout=1)
        except Exception:
            dp = None

        if not isinstance(dp, dict) or not dp:
            return out
        ignore_names = {"LabelModeActiveChoice"}

        for pname, values in dp.items():
            if pname in ignore_names:
                continue
            try:
                val = mle.get_param(block, pname, nargout=1)
            except Exception:
                continue

            rec = {"name": pname}
            sval = str(val)
            m = _num_unit_re.match(sval)
            if m:
                try:
                    rec["value"] = float(m.group(1))
                except Exception:
                    rec["value"] = None
                unit = m.group(2) or None
                rec["unit"] = unit
                rec["raw"] = sval
            else:
                rec["raw"] = sval
            rec["prompt"] = values["Prompt"]
            out.append(rec)
        return out

    def _simulink_parameters(block):
        out = []
        ignore_names = {
            "LabelModeActiveChoice",
            "Position",
            "ZOrder",
            "Priority",
            "ForegroundColor",
            "BackgroundColor",
            "DropShadow",
            "ShowName",
            "NameLocation",
            "FontSize",
            "FontWeight",
            "Orientation",
            "BlockRotation",
            "BlockMirror",
            "Tag",
            "UserData",
            "UserDataPersistent",
            "AttributesFormatString",
        }

        dp = None
        try:
            dp = mle.get_param(block, "DialogParameters", nargout=1)
        except Exception:
            dp = None

        def _record_for_param(pname, pinfo, value_obj):
            if pname in ignore_names:
                return None
            rec = {"name": pname}
            sval = str(value_obj)

            m = _num_unit_re.match(sval)
            if m:
                try:
                    rec["value"] = float(m.group(1))
                except Exception:
                    rec["value"] = None
                unit = m.group(2) or None
                rec["unit"] = unit
                rec["raw"] = sval
            else:
                rec["raw"] = sval

            if isinstance(pinfo, dict):
                rec["prompt"] = pinfo.get("Prompt")
            else:
                rec["prompt"] = None
            return rec

        if isinstance(dp, dict) and dp:
            for pname, pinfo in dp.items():
                if pname in ignore_names:
                    continue
                try:
                    val = mle.get_param(block, pname, nargout=1)
                except Exception:
                    continue
                rec = _record_for_param(pname, pinfo, val)
                if rec is not None:
                    out.append(rec)

            if out:
                return out

        try:
            op = mle.get_param(block, "ObjectParameters", nargout=1)
        except Exception:
            op = None

        if isinstance(op, dict) and op:
            for pname, pmeta in op.items():
                if pname in ignore_names:
                    continue
                try:
                    writable = None
                    if isinstance(pmeta, dict):
                        writable = pmeta.get("Writable", pmeta.get("writable", None))
                    if writable is not None and str(writable).lower() not in ("on", "true", "1", "yes"):
                        continue

                    val = mle.get_param(block, pname, nargout=1)
                except Exception:
                    continue

                rec = _record_for_param(pname, pmeta if isinstance(pmeta, dict) else {}, val)
                if rec is not None:
                    out.append(rec)

        return out

    color_set = {c.lower() for c in filter_based_on_colors} if filter_based_on_colors else None
    found = set()

    try:
        all_blks = list(
            mle.find_system(
                model,
                "LookUnderMasks",
                "all",
                "FollowLinks",
                "on",
                "Type",
                "block",
                nargout=1,
            )
        )
    except Exception as e:
        raise RuntimeError(f"Unable to enumerate blocks in '{model}': {e}")

    if color_set is None:
        found.update(blk for blk in all_blks if blk != model)
    else:
        for cname in filter_based_on_colors:
            try:
                res = mle.find_system(
                    model,
                    "LookUnderMasks",
                    "all",
                    "FollowLinks",
                    "on",
                    "Type",
                    "block",
                    "ForegroundColor",
                    cname,
                    nargout=1,
                )
                for b in list(res):
                    if isinstance(b, str) and b != model:
                        found.add(b)
            except Exception:
                pass

        for blk in all_blks:
            if blk in found:
                continue
            try:
                col = mle.get_param(blk, "ForegroundColor", nargout=1)
            except Exception:
                continue

            if isinstance(col, str):
                if col.lower() in color_set:
                    found.add(blk)
                continue

            try:
                arr = np.array(col, dtype=float).ravel()
                if arr.size == 3 and (
                    np.allclose(arr, [1.0, 0.0, 1.0], atol=1e-6)
                    or np.allclose(arr * 255.0, [255.0, 0.0, 255.0], atol=1e-3)
                ):
                    found.add(blk)
            except Exception:
                pass

    result = []
    for blk in sorted(found):
        kind = _classify(blk)
        if kind == "simscape":
            params = _simscape_parameters(blk)
            result.append({"path": blk, "kind": kind, "parameters": params})
        else:
            params = _simulink_parameters(blk)
            result.append({"path": blk, "kind": kind, "parameters": params})
    return result


def extract_declared_variables(raw_value: Any, declared_names: Iterable[str]) -> Tuple[List[str], bool]:
    """
    Return declared variable identifiers referenced in ``raw_value`` and a flag
    indicating whether the raw value matches a declared identifier exactly.
    """

    if not declared_names:
        return [], False

    declared_set = set(declared_names) if not isinstance(declared_names, set) else declared_names
    if not declared_set:
        return [], False

    if not isinstance(raw_value, str) or not raw_value:
        return [], False

    matches: List[str] = []
    for token in MATLAB_IDENTIFIER_RE.findall(raw_value):
        if token in declared_set and token not in matches:
            matches.append(token)

    is_exact_match = bool(matches) and raw_value == matches[0]
    return matches, is_exact_match


def get_configured_stop_time(mle, model_name: str) -> float:
    """Return the numeric stop time resolved from model-workspace variable `end_time_input_s`."""

    configured = mle.get_param(model_name, "StopTime", nargout=1)
    configured_text = str(configured or "").strip()
    if configured_text != PRIMARY_STOP_TIME_WORKSPACE_VARIABLE:
        raise ValueError(
            f"Model '{model_name}' must set StopTime to '{PRIMARY_STOP_TIME_WORKSPACE_VARIABLE}', got {configured_text!r}"
        )

    try:
        model_workspace = mle.get_param(model_name, "ModelWorkspace", nargout=1)
    except Exception as exc:
        raise ValueError(
            f"Model '{model_name}' must expose a ModelWorkspace containing '{PRIMARY_STOP_TIME_WORKSPACE_VARIABLE}'"
        ) from exc

    try:
        resolved = mle.evalin(model_workspace, PRIMARY_STOP_TIME_WORKSPACE_VARIABLE, nargout=1)
    except Exception as exc:
        raise ValueError(
            f"Model '{model_name}' must define a model-workspace variable '{PRIMARY_STOP_TIME_WORKSPACE_VARIABLE}'"
        ) from exc

    try:
        parsed = float(resolved)
    except Exception as exc:
        raise ValueError(
            f"Model '{model_name}' model-workspace variable '{PRIMARY_STOP_TIME_WORKSPACE_VARIABLE}' must resolve to a finite number"
        ) from exc

    if not math.isfinite(parsed):
        raise ValueError(
            f"Model '{model_name}' model-workspace variable '{PRIMARY_STOP_TIME_WORKSPACE_VARIABLE}' must resolve to a finite number"
        )
    return parsed


def iter_mwopc_parts(text: str):
    """Yield (path, content) chunks from a __MWOPC_PART_BEGIN__ file."""
    marker = "__MWOPC_PART_BEGIN__"
    chunks = text.split(marker)
    for chunk in chunks[1:]:
        chunk = chunk.lstrip()
        if not chunk:
            continue
        first_line, _, rest = chunk.partition("\n")
        path = first_line.strip()
        content = rest.lstrip("\r\n")
        yield path, content


def extract_simulink_parts_from_file(
    file_path: Path,
    wanted_paths: Iterable[str] = WANTED,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> Dict[str, str]:
    """Return a dict of wanted XML parts from a packed Simulink file."""
    raw = file_path.read_text(encoding=encoding, errors=errors)
    out: Dict[str, str] = {}
    wanted_set = set(wanted_paths)
    for path, content in iter_mwopc_parts(raw):
        print(path)
        if path in wanted_set:
            out[path] = content
    return out


TOP_LEVEL_P_REMOVE = {
    "Location",
    "Open",
    "ZoomFactor",
    "ReportName",
    "SIDHighWatermark",
}

BLOCK_P_REMOVE = {
    "Position",
    "ZOrder",
    "BlockRotation",
    "BlockMirror",
    "BackgroundColor",
    "NameLocation",
    "ShowName",
    "HideAutomaticName",
    "FontSize",
    "LibraryVersion",
    "SchemaVersion",
    "ClassName",
    "ComponentPath",
    "ComponentVariants",
    "ComponentVariantNames",
    "SourceFile",
}

INSTANCE_P_REMOVE = {
    "LogSimulationData",
    "SimscapeInstrumentationLogging",
    "SimscapeInstrumentationVariables",
    "InternalSimscapePortConfiguration",
    "RTWMemSecFuncInitTerm",
    "RTWMemSecFuncExecute",
    "RTWMemSecDataConstants",
    "RTWMemSecDataInternal",
    "RTWMemSecDataParameters",
    "ContentPreviewEnabled",
}

SCOPE_P_REMOVE = {
    "GraphicalSettings",
    "WindowPosition",
    "ScopeFrameLocation",
    "WasSavedAsWebScope",
    "Floating",
    "MultipleDisplayCache",
    "Title",
    "ActiveDisplayYMinimum",
    "ActiveDisplayYMaximum",
    "NumInputPorts",
}

PORT_P_REMOVE = {"DataLogging"}
ELEMENT_TAGS_REMOVE_UNDER_BLOCK = {"PortCounts"}
LINE_BRANCH_KEEP_P = {"Src", "Dst", "Name"}
LINE_BRANCH_ATTR_REMOVE = {"ConnectType"}


def is_pid_block(block: ET.Element) -> bool:
    for p in block.findall("P"):
        if p.get("Name") == "SourceBlock" and "slpidlib/PID Controller" in (p.text or ""):
            return True
    return False


def prune_pid_boilerplate(block: ET.Element) -> None:
    inst = block.find("InstanceData")
    if inst is None:
        return
    to_delete = []
    for p in inst.findall("P"):
        name = p.get("Name", "")
        if (
            name.endswith("DataTypeStr")
            or name.endswith("OutMin")
            or name.endswith("OutMax")
            or name.endswith("ParamMin")
            or name.endswith("ParamMax")
            or name.endswith("AccumDataTypeStr")
            or name.endswith("ICVariant")
            or name.endswith("Variant")
            or name
            in {
                "RndMeth",
                "SaturateOnIntegerOverflow",
                "LockScale",
                "IntegratorRTWStateStorageClass",
                "FilterRTWStateStorageClass",
                "LinearizeAsGain",
            }
        ):
            to_delete.append(p)

    for p in to_delete:
        inst.remove(p)


def remove_selected_Ps(elem: ET.Element, names: set) -> None:
    for p in list(elem.findall("P")):
        if p.get("Name") in names:
            elem.remove(p)


def clean_block(block: ET.Element) -> None:
    if "SID" in block.attrib:
        del block.attrib["SID"]

    for tag in list(block):
        if tag.tag in ELEMENT_TAGS_REMOVE_UNDER_BLOCK:
            block.remove(tag)

    remove_selected_Ps(block, BLOCK_P_REMOVE)

    inst = block.find("InstanceData")
    if inst is not None:
        remove_selected_Ps(inst, INSTANCE_P_REMOVE)

    if is_pid_block(block):
        prune_pid_boilerplate(block)

    name = block.get("Name", "")
    if block.tag == "Block" and name == "Scope":
        remove_selected_Ps(block, SCOPE_P_REMOVE)
        for child in list(block):
            if child.tag == "List":
                block.remove(child)

    portprops = block.find("PortProperties")
    if portprops is not None:
        for port in portprops.findall("Port"):
            remove_selected_Ps(port, PORT_P_REMOVE)


def clean_line_or_branch(elem: ET.Element) -> None:
    for attr in LINE_BRANCH_ATTR_REMOVE:
        if attr in elem.attrib:
            del elem.attrib[attr]

    for p in list(elem.findall("P")):
        if p.get("Name") not in LINE_BRANCH_KEEP_P:
            elem.remove(p)

    for br in elem.findall("Branch"):
        clean_line_or_branch(br)


def pretty_indent(elem: ET.Element, level: int = 0) -> None:
    """Indent an ElementTree for readability."""
    indent_str = "\n" + ("  " * level)
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent_str + "  "
        for child in elem:
            pretty_indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent_str
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent_str


def clean_tree(tree: ET.ElementTree) -> None:
    root = tree.getroot()

    remove_selected_Ps(root, TOP_LEVEL_P_REMOVE)

    for block in root.findall("Block"):
        clean_block(block)

    for line in root.findall("Line"):
        clean_line_or_branch(line)


def obtain_xml_system_description(mld_file_path: Path):
    parts = extract_simulink_parts_from_file(mld_file_path)
    item = parts["/simulink/systems/system_root.xml"]
    root = ET.fromstring(item)
    tree = ET.ElementTree(root)
    clean_tree(tree)
    pretty_indent(tree.getroot())
    cleaned_xml = ET.tostring(tree.getroot(), encoding="utf-8").decode("utf-8")
    return cleaned_xml


def fix_simulink_model_path(target_folder: Path) -> str:
    files = [p for p in target_folder.iterdir() if p.is_file()]

    if Path("simulink_model_original.mdl") in files:
        return str(target_folder / "simulink_model_original.mdl")
    elif Path("ssimulink_model_original.mdl.r2025a") in files:
        os.rename("simulink_model_original.mdl.r2025a", "simulink_model_original.mdl")
        return str(target_folder / "simulink_model_original.mdl")
    elif Path("simulink_model_original.mdl.r2025a") in files:
        os.rename("simulink_model_original.mdl.r2025a", "simulink_model_original.mdl")
        return str(target_folder / "simulink_model_original.mdl")
    else:
        raise ValueError("Something wrong with the path")


def _default_parameter_change_ranges(stop_time: float) -> Dict[str, Any]:
    """Fallback window for the time at which a parameter change occurs."""
    min_duration = 1e-6
    try:
        stop = float(stop_time)
    except (TypeError, ValueError):
        stop = math.inf

    if not math.isfinite(stop) or stop <= 0:
        return {
            "start_range": [0.25, 0.5],
            "distribution": "uniform",
        }

    start_lo = max(0.0, 0.5 * stop)
    start_hi = min(0.6 * stop, stop - min_duration)
    if start_hi <= start_lo:
        start_lo = max(0.0, 0.25 * stop)
        start_hi = min(stop - min_duration, max(start_lo + 0.1 * stop, start_lo + 10 * min_duration))

    start_hi = min(start_hi, stop - min_duration)
    if start_hi <= start_lo:
        available = max(stop - min_duration, min_duration)
        start_lo = max(0.0, available - max(min_duration, 0.1 * stop))
        start_hi = min(available, max(start_lo + min_duration, available))

    if start_hi <= start_lo:
        start_hi = start_lo + min_duration

    return {
        "start_range": [float(start_lo), float(start_hi)],
        "distribution": "uniform",
    }


def collect_all_models(model_path: str = "models") -> List[str]:
    """Collect available models relative to `model_path`."""

    root = Path(model_path)
    if not root.exists():
        return []

    models: List[str] = []

    for category in sorted(root.iterdir(), key=lambda p: p.name):
        if not category.is_dir() or category.name == ".DS_Store":
            continue

        subdirs = [child for child in category.iterdir() if child.is_dir()]
        if subdirs:
            for child in sorted(subdirs, key=lambda p: p.name):
                models.append(f"{category.name}/{child.name}")
        else:
            models.append(category.name)

    return models


def short_block_name(ident: str) -> str:
    name = ident.split("/")[-1].replace("\n", " ").strip()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _ident_to_working(ident: Optional[str]) -> Optional[str]:
    if ident is None:
        return ident
    return ident.replace("simulink_model_original", "simulink_model")


def _ensure_model_stopped(mle, model_name: str = "simulink_model", timeout_s: float = 10.0):
    """Best-effort: stop a running/compiled model and wait until status == 'stopped'."""
    try:
        status = mle.get_param(model_name, "SimulationStatus")
    except Exception:
        return

    if status != "stopped":
        try:
            mle.set_param(model_name, "SimulationCommand", "stop", nargout=0)
        except Exception:
            pass
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                status = mle.get_param(model_name, "SimulationStatus")
            except Exception:
                break
            if status == "stopped":
                break
            time.sleep(0.1)


def save_model_with_workspace_values(
    dest_path: Path,
    workspace_values: Mapping[str, Any],
    *,
    source_path: Path = Path("simulink_model.mdl"),
) -> Dict[str, str]:
    """Persist a Simulink model file with updated WSMATLABCode workspace values."""
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    same_file = False
    try:
        same_file = source_path.resolve() == dest_path.resolve()
    except FileNotFoundError:
        same_file = source_path == dest_path

    text = source_path.read_text(encoding="utf-8")
    match = re.search(r'(<P Name="WSMATLABCode">)(.*?)(</P>)', text, flags=re.DOTALL)
    if match is None or not workspace_values:
        if not same_file:
            shutil.copy(source_path, dest_path)
        return {"model_with_workspace": str(dest_path)}

    workspace_code = match.group(2)

    def _format_value(val: Any) -> str:
        if isinstance(val, bool):
            return "1" if val else "0"
        if isinstance(val, (np.floating, np.integer)):
            val = val.item()
        if isinstance(val, (int, float)):
            return f"{val:.15g}" if math.isfinite(val) else str(val)
        return str(val)

    patterns = {
        str(name): re.compile(
            rf"(\s*{re.escape(str(name))}\s*=\s*)([^;]*)(;.*)",
            flags=re.DOTALL,
        )
        for name in workspace_values
    }

    new_lines: List[str] = []
    updated: Set[str] = set()
    for line in workspace_code.splitlines():
        new_line = line
        for name, pattern in patterns.items():
            m_line = pattern.match(line)
            if m_line:
                new_line = (
                    f"{m_line.group(1)}{_format_value(workspace_values[name])}{m_line.group(3)}"
                )
                updated.add(name)
                break
        new_lines.append(new_line)

    missing = [str(name) for name in workspace_values if str(name) not in updated]
    if missing:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        for name in missing:
            new_lines.append(f"{name} = {_format_value(workspace_values[name])};")

    updated_code = "\n".join(new_lines)
    if workspace_code.endswith("\n") and not updated_code.endswith("\n"):
        updated_code += "\n"

    new_text = text[: match.start(2)] + updated_code + text[match.end(2) :]
    dest_path.write_text(new_text, encoding="utf-8")
    return {"model_with_workspace": str(dest_path)}


def save_model_with_new_initial_variables(run_dir: Path, recipe: Dict[str, Any]) -> Dict[str, str]:
    """Persist a copy of the working Simulink model with updated workspace values."""
    initial_state = recipe.get("initial_state")
    variable_values = getattr(initial_state, "variable_values", {}) or {}

    source_path = Path("simulink_model.mdl")
    dest_path = run_dir / source_path.name
    save_model_with_workspace_values(
        dest_path,
        variable_values,
        source_path=source_path,
    )
    return {"model_with_initial_state": str(dest_path)}

    return {"model_with_initial_state": str(dest_path)}


__all__ = [
    "LOCAL_SOLVER_STEP_SIZE_VARIABLE",
    "MATLAB_IDENTIFIER_RE",
    "PRIMARY_STOP_TIME_WORKSPACE_VARIABLE",
    "WANTED",
    "_default_parameter_change_ranges",
    "_ensure_model_stopped",
    "_get_all_simscape_blocks",
    "_ident_to_working",
    "clean_block",
    "clean_line_or_branch",
    "clean_tree",
    "collect_all_models",
    "extract_declared_variables",
    "extract_simulink_parts_from_file",
    "fix_simulink_model_path",
    "get_all_blocks",
    "get_configured_stop_time",
    "is_pid_block",
    "iter_mwopc_parts",
    "obtain_xml_system_description",
    "pretty_indent",
    "remove_selected_Ps",
    "save_model_with_new_initial_variables",
    "short_block_name",
]
