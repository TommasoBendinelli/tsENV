#!/usr/bin/env python3
"""
build_metadata.py

Generates metadata.json for a Simulink/Simscape model. This is the first step to generate data that can be used for generating data programmatically.

"""

import html
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import suppress, contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Mapping

import click
import matlab.engine  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_repo_dotenv(env_path: Path) -> None:
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_repo_dotenv(REPO_ROOT / ".env")

from shared.benchmark_utils import ALLOWED_TSENV_MODELS
from shared.matlab_runtime import (
    _ProcessCleanupGuard,
    _reset_working_copy,
    force_stop_matlab_processes,
)
from shared.simulink_utils import (
    LOCAL_SOLVER_STEP_SIZE_VARIABLE,
    MATLAB_IDENTIFIER_RE,
    extract_declared_variables,
    fix_simulink_model_path,
    get_all_blocks,
    get_configured_stop_time,
)
from workflows.simulate_core import get_time_series_simscapes, run_simulation

logger = logging.getLogger(__name__)
REQUIRED_LOCAL_SOLVER_CHOICE = "NE_BACKWARD_EULER_ADVANCER"
_RESERVED_MODEL_WORKSPACE_VARIABLES = frozenset(
    {"end_time_input_s", "stop_time", LOCAL_SOLVER_STEP_SIZE_VARIABLE}
)


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_intervention_parameters_from_sampling_file(model_name: str) -> List[str]:
    del model_name  # The helper reads the local experiment_config.json in the current workspace.
    config_path = Path("experiment_config.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    exposed_variables = payload.get("exposed_variables")
    if not isinstance(exposed_variables, dict):
        raise KeyError("experiment_config.json missing exposed_variables")
    parameters = exposed_variables.get("parameters")
    if not isinstance(parameters, dict):
        raise KeyError("experiment_config.json missing exposed_variables.parameters")
    return [str(name).strip() for name in parameters.keys() if str(name).strip()]


def get_signaled_signals(mdl_file: Path) -> List[str]:
    """Extract logged/test-pointed signal names from a Simulink text .mdl (R2025a OPC format)."""
    path = Path(mdl_file)
    try:
        opc_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("Unable to read %s: %s", path, exc)
        return []

    signals: List[str] = []

    part_marker = "__MWOPC_PART_BEGIN__ /simulink/graphicalInterface.json"
    start = opc_text.find(part_marker)
    if start != -1:
        start += len(part_marker)
        end = opc_text.find("__MWOPC_PART_BEGIN__", start)
        if end == -1:
            end = len(opc_text)

        json_str = opc_text[start:end].strip()
        if json_str:
            try:
                gi = json.loads(json_str)
            except json.JSONDecodeError as exc:
                logger.debug("Unable to parse graphicalInterface JSON from %s: %s", path, exc)
            else:
                tp_list = gi.get("GraphicalInterface", {}).get("TestPointedSignals", [])
                for tp in tp_list:
                    if not tp.get("LogSignal"):
                        continue
                    name = (
                        tp.get("SignalName")
                        or tp.get("LogName")
                        or tp.get("FullBlockPath", "")
                    )
                    if name:
                        signals.append(name)

    if signals:
        return signals

    sys_marker = "__MWOPC_PART_BEGIN__ /simulink/systems/system_root.xml"
    sys_start = opc_text.find(sys_marker)
    if sys_start == -1:
        return []
    sys_start += len(sys_marker)
    sys_end = opc_text.find("__MWOPC_PART_BEGIN__", sys_start)
    if sys_end == -1:
        sys_end = len(opc_text)
    sys_xml = opc_text[sys_start:sys_end]

    seen = set()
    for match in re.finditer(r'<Block BlockType="Outport" Name="([^"]+)"', sys_xml):
        name = str(match.group(1) or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        signals.append(name)

    return signals


def validate_simulink_signal_names(signal_names: Iterable[str]) -> None:
    invalid = [str(name) for name in signal_names if " " in str(name)]
    if invalid:
        raise ValueError(
            "All entries in simulink_signals_available must contain no spaces: "
            + ", ".join(repr(name) for name in invalid)
        )


def _model_workspace_defines_variable(model_path: Path, variable_name: str) -> bool:
    workspace_code = _extract_workspace_code(model_path)
    if workspace_code is None:
        return False
    return variable_name in _parse_symbolic_assignments(workspace_code)


def _matlab_value_list(raw_value: Any) -> List[Any]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    try:
        return list(raw_value)
    except Exception:
        return [raw_value]


def find_simscape_solver_configuration_blocks(mle, model_name: str) -> List[str]:
    sc_blocks = mle.find_system(
        model_name,
        "LookUnderMasks",
        "all",
        "FollowLinks",
        "on",
        "MaskType",
        "Solver Configuration",
        nargout=1,
    )
    block_paths: List[str] = []
    seen_paths: Set[str] = set()
    for raw_block in _matlab_value_list(sc_blocks):
        block_path = str(raw_block or "").strip()
        if not block_path or block_path in seen_paths:
            continue
        seen_paths.add(block_path)
        block_paths.append(block_path)
    return block_paths


def _assert_model_workspace_variable_exists(mle, model_name: str, variable_name: str) -> None:
    try:
        model_workspace = mle.get_param(model_name, "ModelWorkspace", nargout=1)
        mle.evalin(model_workspace, variable_name, nargout=1)
    except Exception as exc:
        raise RuntimeError(
            f"Model '{model_name}' must define model-workspace variable "
            f"{variable_name!r}"
        ) from exc


def _validate_simscape_local_solver_blocks(
    mle,
    model_name: str,
    solver_blocks: Iterable[str],
) -> None:
    _assert_model_workspace_variable_exists(
        mle, model_name, LOCAL_SOLVER_STEP_SIZE_VARIABLE
    )

    for block_path in solver_blocks:
        use_local_solver = str(
            mle.get_param(block_path, "UseLocalSolver", nargout=1) or ""
        ).strip()
        if use_local_solver.lower() != "on":
            raise RuntimeError(
                "Simscape Solver Configuration block must use a local solver: "
                f"{block_path} has UseLocalSolver={use_local_solver!r}"
            )
        local_solver_choice = str(
            mle.get_param(block_path, "LocalSolverChoice", nargout=1) or ""
        ).strip()
        if local_solver_choice != REQUIRED_LOCAL_SOLVER_CHOICE:
            raise RuntimeError(
                "Simscape Solver Configuration block must use the documented "
                "local implicit fixed-step solver: "
                f"{block_path} has LocalSolverChoice={local_solver_choice!r}"
            )
        local_solver_sample_time = str(
            mle.get_param(block_path, "LocalSolverSampleTime", nargout=1) or ""
        ).strip()
        if local_solver_sample_time != LOCAL_SOLVER_STEP_SIZE_VARIABLE:
            raise RuntimeError(
                "Simscape Solver Configuration block must set "
                f"LocalSolverSampleTime to {LOCAL_SOLVER_STEP_SIZE_VARIABLE!r}: "
                f"{block_path} has LocalSolverSampleTime={local_solver_sample_time!r}"
            )


def _validate_global_fixed_step_solver(mle, model_name: str, model_path: Path) -> None:
    if not _model_workspace_defines_variable(model_path, LOCAL_SOLVER_STEP_SIZE_VARIABLE):
        raise RuntimeError(
            f"Non-Simscape model '{model_name}' must define "
            f"{LOCAL_SOLVER_STEP_SIZE_VARIABLE!r} in WSMATLABCode"
        )
    _assert_model_workspace_variable_exists(
        mle, model_name, LOCAL_SOLVER_STEP_SIZE_VARIABLE
    )

    solver_type = str(mle.get_param(model_name, "SolverType", nargout=1) or "").strip()
    if solver_type.lower() != "fixed-step":
        raise RuntimeError(
            "Non-Simscape model must use the documented global fixed-step solver: "
            f"{model_name} has SolverType={solver_type!r}"
        )

    fixed_step = str(mle.get_param(model_name, "FixedStep", nargout=1) or "").strip()
    if fixed_step != LOCAL_SOLVER_STEP_SIZE_VARIABLE:
        raise RuntimeError(
            "Non-Simscape model must set global FixedStep to "
            f"{LOCAL_SOLVER_STEP_SIZE_VARIABLE!r}: "
            f"{model_name} has FixedStep={fixed_step!r}"
        )


def validate_solver_contract(mle, model_name: str, model_path: Path) -> None:
    solver_blocks = find_simscape_solver_configuration_blocks(mle, model_name)
    if solver_blocks:
        _validate_simscape_local_solver_blocks(mle, model_name, solver_blocks)
    else:
        _validate_global_fixed_step_solver(mle, model_name, Path(model_path))


def validate_simscape_local_solver(mle, model_name: str) -> None:
    """Backward-compatible wrapper for callers/tests that validate Simscape blocks only."""
    solver_blocks = find_simscape_solver_configuration_blocks(mle, model_name)
    if solver_blocks:
        _validate_simscape_local_solver_blocks(mle, model_name, solver_blocks)


SIMSCAPE_CUSTOM_DIRNAME = "simscape_custom"
SIMSCAPE_CUSTOM_ENV_VAR = "SIMSCAPE_CUSTOM_PATH"


def resolve_simscape_custom_dir() -> Path:
    configured = os.environ.get(SIMSCAPE_CUSTOM_ENV_VAR, "").strip()
    if not configured:
        raise FileNotFoundError(
            f"{SIMSCAPE_CUSTOM_ENV_VAR} must be set to the custom Simscape directory."
        )
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"{SIMSCAPE_CUSTOM_ENV_VAR} resolved to '{resolved}', but that directory does not exist."
        )
    return resolved


def _matlab_quote(value: Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _set_simulink_file_gen_root(mle: Any, cache_dir: Path) -> None:
    cache_folder = cache_dir / "simulink_cache"
    codegen_folder = cache_dir / "simulink_codegen"
    cache_folder.mkdir(parents=True, exist_ok=True)
    codegen_folder.mkdir(parents=True, exist_ok=True)
    command = (
        "Simulink.fileGenControl('set',"
        f"'CacheFolder',{_matlab_quote(cache_folder)},"
        f"'CodeGenFolder',{_matlab_quote(codegen_folder)},"
        "'createDir',true);"
    )
    mle.eval(command, nargout=0)


@contextmanager
def matlab_session(
    original_cwd,
    *,
    cache_root: Optional[Path] = None,
    keep_cache: bool = False,
):
    """Start a MATLAB engine session and tear it down reliably."""
    session_started_at = time.time()
    cache_dir: Optional[Path] = None
    old_tmpdir = os.environ.get("TMPDIR")
    if cache_root is not None:
        resolved_cache_root = Path(cache_root).expanduser().resolve()
        resolved_cache_root.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(
            tempfile.mkdtemp(
                prefix="tsenv-matlab-",
                dir=str(resolved_cache_root),
            )
        )
        os.environ["TMPDIR"] = str(cache_dir / "tmp")
        Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    mle = None
    interrupted = False
    try:
        with _ProcessCleanupGuard("matlab_session"):
            mle = matlab.engine.start_matlab()
            try:
                simscape_custom_dir = resolve_simscape_custom_dir()
                mle.addpath(str(simscape_custom_dir), nargout=0)
                if cache_dir is not None:
                    _set_simulink_file_gen_root(mle, cache_dir)
                with suppress(Exception):
                    setattr(mle, "_tsenv_session_started_at", session_started_at)
                yield mle
            except KeyboardInterrupt:
                interrupted = True
                force_stop_matlab_processes(
                    started_at=session_started_at,
                    reason="matlab_session interrupt",
                )
                raise
            finally:
                if not interrupted and mle is not None:
                    with suppress(Exception):
                        mle.eval("Simulink.fileGenControl('reset');", nargout=0)
                    with suppress(Exception):
                        mle.quit()
    finally:
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir
        if cache_dir is not None and not keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)


@contextmanager
def temporary_working_model_copy(
    *,
    source_model: Path = Path("simulink_model_original.mdl"),
    working_model: Path = Path("simulink_model.mdl"),
):
    _reset_working_copy(working_model=str(working_model), source_model=str(source_model))
    try:
        yield working_model
    finally:
        with suppress(FileNotFoundError):
            working_model.unlink()


def discover_simscape_signals(
    mle: Any,
    *,
    stop_time: float,
    sim_script: Path,
) -> List[str]:
    with suppress(Exception):
        mle.set_param("simulink_model_original", "FastRestart", "off", nargout=0)
    with suppress(Exception):
        mle.close_system("simulink_model_original", 0, nargout=0)
    with suppress(Exception):
        mle.close_system("simulink_model", 0, nargout=0)

    debug_dir = Path("generated") / "metadata_simscape_discovery"
    try:
        with temporary_working_model_copy():
            result = run_simulation(
                mle,
                stop_time=float(stop_time),
                debug_dir=debug_dir,
                debug=False,
                sim_the_model_path=str(sim_script),
                save_simscape_mat=False,
            )
            simscape_dict = get_time_series_simscapes(result, matlab_engine=mle)
    finally:
        shutil.rmtree(debug_dir, ignore_errors=True)

    discovered = {str(name).strip() for name in simscape_dict.keys() if str(name).strip()}
    return sorted(discovered)


def _get_workspace_variable_names(mle, mws) -> Dict[str, Any]:
    names = mle.evalin(mws, "who", nargout=1) or []
    out: Dict[str, Any] = {}

    for name in names:
        if not isinstance(name, str):
            continue
        try:
            out[name] = mle.evalin(mws, name, nargout=1)  # fetch value
        except Exception:
            continue

    return out


_MODEL_WORKSPACE_RE = re.compile(r"<ModelWorkspace>(.*?)</ModelWorkspace>", re.DOTALL)
_WSMATLABCODE_RE = re.compile(r'<P Name="WSMATLABCode">(.*?)</P>', re.DOTALL)
_ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z]\w*)\s*=\s*(.+?)\s*;?\s*$")


def _extract_workspace_code(mdl_path: Path) -> Optional[str]:
    """Pull the raw WSMATLABCode block from a Simulink .mdl file."""
    try:
        mdl_text = Path(mdl_path).read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("Unable to read %s: %s", mdl_path, exc)
        return None
    workspace_match = _MODEL_WORKSPACE_RE.search(mdl_text)
    if not workspace_match:
        return None

    code_match = _WSMATLABCODE_RE.search(workspace_match.group(1))
    if not code_match:
        return None

    return html.unescape(code_match.group(1))


def _parse_symbolic_assignments(
    workspace_code: str, *, limit_to: Optional[Iterable[str]] = None
) -> Dict[str, str]:
    """Return a mapping of variable names to their symbolic (unevaluated) expressions."""
    limit: Optional[Set[str]] = set(limit_to) if limit_to is not None else None
    symbolic: Dict[str, str] = {}

    for raw_line in workspace_code.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("%") or "=" not in stripped:
            continue

        code_without_comment = stripped.split("%", 1)[0].strip()
        if not code_without_comment:
            continue

        match = _ASSIGNMENT_RE.match(code_without_comment)
        if not match:
            continue

        name, expr = match.groups()
        if limit is not None and name not in limit:
            continue

        symbolic[name] = expr.strip().rstrip(";")

    return symbolic


def get_symbolic_workspace_assignments(
    mdl_path: Path, *, variable_names: Optional[Iterable[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Return {var: {expression, is_primitive}} for Model Workspace assignments in the .mdl file."""
    workspace_code = _extract_workspace_code(mdl_path)
    if not workspace_code:
        return {}

    all_assignments = _parse_symbolic_assignments(workspace_code)
    selected_assignments = (
        {k: v for k, v in all_assignments.items() if k in set(variable_names)}
        if variable_names is not None
        else all_assignments
    )
    workspace_variables = set(all_assignments.keys())

    symbolic_with_flags: Dict[str, Dict[str, Any]] = {}
    for name, expr in selected_assignments.items():
        identifiers = set(re.findall(r"[A-Za-z]\w*", expr))
        dependencies = (identifiers & workspace_variables) - {name}
        symbolic_with_flags[name] = {
            "expression": expr,
            "is_primitive": len(dependencies) == 0,
        }

    return symbolic_with_flags


def _set_runtime_configurability(mle, block_path: str, param_name: str) -> None:
    """Best effort: switch the block parameter to run-time configurability."""
    param_name_conf = param_name + "_conf"
    try:
        mle.set_param(block_path, param_name_conf, "runtime", nargout=0)
        return True
    except Exception:
        return False


def build_metadata(
    mle,
    model_path: str = "simulink_model_original.mdl",
) -> Dict:
    model_path = Path(model_path)
    model_name = model_path.stem

    mle.load_system(str(model_name))
    mws = mle.get_param(str(model_name), "ModelWorkspace")
    default_variable_values = {
        key: value
        for key, value in _get_workspace_variable_names(mle, mws).items()
        if key not in _RESERVED_MODEL_WORKSPACE_VARIABLES
    }
    intervention_parameters = list(default_variable_values.keys())
    stop_time = get_configured_stop_time(mle, model_name=model_name)

    symbolic_workspace_assignments = get_symbolic_workspace_assignments(
        model_path, variable_names=intervention_parameters
    )
    for name, raw_value in default_variable_values.items():
        if name in symbolic_workspace_assignments:
            continue
        numeric_value = _to_float_or_none(raw_value)
        if numeric_value is None:
            continue
        symbolic_workspace_assignments[name] = {
            "expression": repr(float(numeric_value)),
            "is_primitive": True,
        }
    primitive_parameter_names = {
        key for key, value in symbolic_workspace_assignments.items() if value["is_primitive"]
    }
    non_primitive_parameter_names = {
        key for key, value in symbolic_workspace_assignments.items() if not value["is_primitive"]
    }
    intervention_params_to_blocks_map = map_intervention_parameters_to_simulink_blocks(
        mle,
        mws,
        include_simulink=True,
        parameter_set=primitive_parameter_names,
        intervention_parameters=intervention_parameters,
        symbolic_workspace_assignments=symbolic_workspace_assignments,
        model_name=model_name,
    )
    from_build_metadata: Set[str] = set(intervention_params_to_blocks_map.keys())
    raw_expressions = [
        param["expression"]
        for params in intervention_params_to_blocks_map.values()
        for param in params["parameters"]
    ]
    for raw_expression in raw_expressions:
        for non_primitive_variable in non_primitive_parameter_names:
            assert not non_primitive_variable in raw_expression

    for key, bindings in intervention_params_to_blocks_map.items():
        for idx, entry in enumerate(bindings["parameters"]):  # .get("parameters", []):
            path = entry.get("path")
            param = entry.get("name")
            if path and param:
                runtime_type = _set_runtime_configurability(mle, path, param)
                intervention_params_to_blocks_map[key]["parameters"][idx][
                    "runtime_type"
                ] = runtime_type

    mle.save_system(str(model_path))
    assert set(symbolic_workspace_assignments.keys()) == set(
        primitive_parameter_names
    ) | set(non_primitive_parameter_names)
    assert set(primitive_parameter_names) & set(non_primitive_parameter_names) == set()
    unmapped_primitive_parameters = primitive_parameter_names - from_build_metadata
    if unmapped_primitive_parameters:
        logger.warning(
            "Primitive workspace variables without external block bindings in %s: %s",
            os.getcwd(),
            sorted(unmapped_primitive_parameters),
        )
    parameter_list = sorted(from_build_metadata)
    assert parameter_list, "No externally exposed primitive parameters found in model"
    filtered_default_variable_values = {
        key: default_variable_values[key]
        for key in parameter_list
        if key in default_variable_values
    }
    assert set(filtered_default_variable_values.keys()) == set(
        parameter_list
    ), "Missing intervention default values in model workspace"
    default_values = {
        **filtered_default_variable_values,
        "end_time_input_s": float(stop_time),
    }
    return {
        "time_grid": {
            "uST": stop_time / 10_000,
            "stop_time": float(stop_time),
        },
        "parameter_set": parameter_list,
        "intervention_block_map": intervention_params_to_blocks_map,
        "default_values": default_values,
    }


def map_intervention_parameters_to_simulink_blocks(
    mle,
    mws,
    model_name: str = "simulink_model_original",
    include_simulink: bool = False,
    parameter_set: Optional[Iterable[str]] = None,
    intervention_parameters: Optional[Iterable[str]] = None,
    symbolic_workspace_assignments: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    workspace_parameters_set: Set[str] = (
        {str(p) for p in intervention_parameters}
        if intervention_parameters is not None
        else _get_workspace_variable_names(mle, mws)
    )
    configured_parameters_set: Set[str] = {str(p) for p in (parameter_set or [])}
    candidate_parameters_set: Set[str] = (
        set(configured_parameters_set)
        if configured_parameters_set
        else set(workspace_parameters_set)
    )
    if (
        configured_parameters_set
        and configured_parameters_set != workspace_parameters_set
    ):
        only_in_workspace = sorted(workspace_parameters_set - configured_parameters_set)
        only_in_distribution = sorted(
            configured_parameters_set - workspace_parameters_set
        )
        assert (
            len(only_in_distribution) == 0
        ), f"Missing parameters {only_in_distribution}"
        logger.warning(
            "Intervention parameters mismatch in %s: only in workspace=%s, only in distribution=%s",
            os.getcwd(),
            only_in_workspace,
            only_in_distribution,
        )

    symbolic_assignments: Dict[str, Dict[str, Any]] = (
        dict(symbolic_workspace_assignments) if symbolic_workspace_assignments else {}
    )
    resolved_expression_cache: Dict[str, str] = {}

    def resolve_symbol_expression(var_name: str, seen: Optional[Set[str]] = None) -> str:
        if var_name in resolved_expression_cache:
            return resolved_expression_cache[var_name]

        info = symbolic_assignments.get(var_name)
        if info is None:
            resolved_expression_cache[var_name] = var_name
            return var_name

        # Keep selected intervention parameters symbolic in expressions.
        if var_name in candidate_parameters_set:
            resolved_expression_cache[var_name] = var_name
            return var_name

        expr = str(info.get("expression") or var_name)
        visited: Set[str] = set(seen or [])
        if var_name in visited:
            logger.warning("Cyclic dependency detected while resolving %s", var_name)
            resolved_expression_cache[var_name] = expr
            return expr
        visited.add(var_name)

        def repl(match: re.Match) -> str:
            token = match.group(0)
            replacement = resolve_symbol_expression(token, visited)
            return f"({replacement})" if replacement != token else replacement

        expanded = MATLAB_IDENTIFIER_RE.sub(repl, expr)
        resolved_expression_cache[var_name] = expanded
        return expanded

    def substitute_non_candidate_symbols(raw_expr: str) -> str:
        if not symbolic_assignments:
            return raw_expr

        def repl(match: re.Match) -> str:
            token = match.group(0)
            if token not in symbolic_assignments:
                return token
            replacement = resolve_symbol_expression(token)
            return f"({replacement})" if replacement != token else replacement

        return MATLAB_IDENTIFIER_RE.sub(repl, raw_expr)

    blocks = get_all_blocks(mle, model=model_name, filter_based_on_colors=None)
    intervention_parameter_to_blocks: Dict[str, Dict[str, Any]] = {}
    for blk in blocks:
        kind = blk.get("kind")
        parameters = blk.get("parameters")

        if not include_simulink and kind == "simulink":
            continue

        for parameter in parameters or []:
            raw_name = parameter["raw"]

            if parameter["prompt"] == "S-function parameters:":
                raw_name_list = raw_name.split(",")
            else:
                raw_name_list = [raw_name]

            for rn in raw_name_list:
                rn = substitute_non_candidate_symbols(rn)
                matched_names, _ = extract_declared_variables(
                    rn, candidate_parameters_set
                )
                if not matched_names:
                    continue

                for matched_name in matched_names:
                    entry = intervention_parameter_to_blocks.get(matched_name)
                    if entry is None:
                        entry = {
                            "parameters": [],
                        }
                        intervention_parameter_to_blocks[matched_name] = entry
                    parameters_for_var = entry.setdefault("parameters", [])
                    parameters_for_var.append(
                        {
                            "path": blk["path"],
                            "name": parameter["name"],
                            "expression": rn,
                        }
                    )
    return intervention_parameter_to_blocks
def get_optimizer_info(mle, model_name: str) -> Dict:
    """Collect solver settings from the model config and Simscape Solver Configuration blocks."""
    info: Dict[str, object] = {}

    cs = None
    try:
        cs = mle.getActiveConfigSet(model_name, nargout=1)
    except Exception:
        cs = None

    def _gp(obj, param: str):
        try:
            return mle.get_param(obj, param, nargout=1)
        except Exception:
            try:
                return mle.get_param(model_name, param, nargout=1)
            except Exception:
                return None

    target = cs if cs is not None else model_name

    for key in [
        "SolverType",
        "Solver",
        "FixedStep",
        "MaxStep",
        "MinStep",
        "InitialStep",
        "RelTol",
        "AbsTol",
        "StartTime",
        "StopTime",
    ]:

        val = _gp(target, key)
        if key == "StartTime":
            assert float(val) == 0
            info[key] = val
            continue

        else:
            if val is not None:
                if key in {
                    "FixedStep",
                    "MaxStep",
                    "MinStep",
                    "InitialStep",
                    "RelTol",
                    "StartTime",
                    "StopTime",
                }:
                    if val == "auto":
                        info[key] = val
                    else:
                        f = float(eval(val))
                        info[key] = f
                else:
                    info[key] = val

    stop_time_raw = info.get("StopTime")
    if stop_time_raw is None:
        raise RuntimeError(
            f"Unable to determine StopTime while extracting optimizer info for model '{model_name}'."
        )
    try:
        stop_time_float = float(stop_time_raw)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid StopTime value '{stop_time_raw}' for model '{model_name}'."
        ) from exc

    for key in [
        "SampleTimeConstraint",
        "AutoInsertRateTranBlk",
        "ZeroCrossControl",
        "ZeroCrossAlgorithm",
        "MaxConsecutiveMinStep",
        "MaxOrder",
        "ConsistencyChecking",
    ]:
        val = _gp(target, key)
        if val is not None:
            info[key] = val

    simscape_blocks: Dict[str, Dict[str, object]] = {}
    try:
        sc_blocks = mle.find_system(
            model_name,
            "LookUnderMasks",
            "all",
            "FollowLinks",
            "on",
            "MaskType",
            "Solver Configuration",
            nargout=1,
        )
        raw_blocks: List[Any]
        if sc_blocks is None:
            raw_blocks = []
        elif isinstance(sc_blocks, str):
            raw_blocks = [sc_blocks]
        elif isinstance(sc_blocks, (list, tuple)):
            raw_blocks = list(sc_blocks)
        else:
            try:
                raw_blocks = list(sc_blocks)
            except Exception:
                raw_blocks = [sc_blocks]

        block_paths: List[str] = []
        seen_paths: Set[str] = set()
        for raw_block in raw_blocks:
            block_path = str(raw_block or "").strip()
            if not block_path or block_path in seen_paths:
                continue
            seen_paths.add(block_path)
            block_paths.append(block_path)

        for blk in block_paths:
            try:
                dp = mle.get_param(blk, "DialogParameters", nargout=1)
                try:
                    keys = list(dp.keys())
                except Exception:
                    keys = [k for k in dir(dp) if not k.startswith("_")]
                values: Dict[str, object] = {}
                for p in keys:
                    try:
                        v = mle.get_param(blk, p, nargout=1)
                        fv = _to_float_or_none(v)
                        values[p] = fv if fv is not None else v
                    except Exception:
                        continue
                simscape_blocks[blk] = values
            except Exception:
                continue
    except Exception:
        pass

    if not simscape_blocks:
        return info
    if len(simscape_blocks) > 1:
        block_list = ", ".join(sorted(simscape_blocks.keys()))
        raise RuntimeError(
            "Expected exactly one Simscape Solver Configuration block in model "
            f"'{model_name}', found {len(simscape_blocks)}: {block_list}"
        )

    # Persist only the canonical subset used by downstream schema/workflows.
    simscape_solver = {
        "UseLocalSolver": "on",
        "LocalSolverChoice": REQUIRED_LOCAL_SOLVER_CHOICE,
        "ResolveIndetEquations": "on",
        "FunctionEvalNumThread": 1.0,
        "LocalSolverSampleTime": LOCAL_SOLVER_STEP_SIZE_VARIABLE,
    }
    info["simscape_solver"] = simscape_solver
    return info


@click.command()
@click.argument("model_arg", required=False, metavar="MODEL")
@click.option(
    "--model",
    "model_option",
    metavar="MODEL",
    help="Generate metadata for a single allowed model.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing metadata.json files",
)
def cli(model_arg: Optional[str], model_option: Optional[str], overwrite: bool) -> None:
    if model_arg and model_option:
        raise click.UsageError("Use either --model MODEL or positional MODEL, not both.")
    model = model_option or model_arg
    main(model, overwrite)


def main(
    target_model: Optional[str] = None,
    overwrite: bool = False,
):
    """If target_model is None, process all ALLOWED_TSENV_MODELS. Else target_model is a model id."""
    original_cwd = Path(os.getcwd()).resolve()
    models_root = original_cwd / "models" / "simulink"

    if target_model is None:
        model_ids = list(ALLOWED_TSENV_MODELS)
    else:
        model_id = str(target_model).strip()
        if model_id not in ALLOWED_TSENV_MODELS:
            raise SystemExit(
                f"Model '{model_id}' is not an allowed tsENV model. "
                "Update shared/benchmark_utils.py (ALLOWED_TSENV_MODELS) to add it."
            )
        model_ids = [model_id]

    simulink_model_name = "simulink_model_original"

    for model_id in sorted(model_ids):
        model = f"simulink/{model_id}"
        model_dir = models_root / model_id

        if not model_dir.exists():
            print(f"Model directory {model_dir} missing, skipping")
            continue

        print(f"Generating simulink model {model}")
        generated = model_dir / "generated"
        generated.mkdir(parents=True, exist_ok=True)

        metadata_path = generated / "metadata.json"
        if metadata_path.exists() and not overwrite:
            print(f"For model {model}, metadata already found, skipping")
            continue
        if metadata_path.exists() and overwrite:
            metadata_path.unlink()

        if not (model_dir / "simulink_model_original.mdl").exists():
            print(f"For model {model}, .mdl file not found, skipping!")
            continue

        try:
            os.chdir(str(model_dir))
            sim_script = original_cwd / "models" / "simulink" / "sim_the_model.m"

            with matlab_session(original_cwd) as mle:
                simulink_path = fix_simulink_model_path(Path("."))
                mle.load_system(simulink_path)
                model_path = f"{simulink_model_name}.mdl"

                with suppress(Exception):
                    mle.set_param(simulink_model_name, "FastRestart", "off", nargout=0)
                mle.close_system(simulink_model_name, 0, nargout=0)

            with matlab_session(original_cwd) as mle:
                metadata = build_metadata(mle, model_path=model_path)
                validate_solver_contract(
                    mle,
                    simulink_model_name,
                    Path(model_path),
                )
                metadata["simscape_signals_available"] = discover_simscape_signals(
                    mle,
                    stop_time=float(metadata["time_grid"]["stop_time"]),
                    sim_script=sim_script,
                )

            labelled_signals = get_signaled_signals("simulink_model_original.mdl")
            validate_simulink_signal_names(labelled_signals)
            metadata["simulink_signals_available"] = labelled_signals
            metadata.setdefault("simscape_signals_available", [])

            metadata.pop("time_grid", None)

            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Wrote {metadata_path.resolve()}")

        finally:
            os.chdir(str(original_cwd))


if __name__ == "__main__":
    cli()
