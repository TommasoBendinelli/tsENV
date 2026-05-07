"""Helpers for managing MATLAB/MathWorks processes and engine streams."""

from __future__ import annotations

import atexit
import getpass
import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil optional in some environments
    psutil = None


logger = logging.getLogger(__name__)

_SCRIPT_START_TIME = time.time()
_CURRENT_USER = getpass.getuser()
_MATHWORKS_CLEANUP_WINDOW_S = float(
    os.environ.get("MATLAB_CLEANUP_WINDOW_SECONDS", "4")
)
_MATHWORKS_MATCHERS = (
    "matlab",
    "mathworks",
    "matlabwindow",
    "matlabwindowhelper",
    "mathworksservicehost",
    "fluxbox",
    "xvfb",
)
_MATLAB_INTERRUPT_PATTERNS = (
    "the matlab function has been cancelled",
    "program interruption (ctrl-c) has been detected",
    "operation terminated by user",
)


class _MatlabStream(io.StringIO):
    """Route MATLAB engine output through the module logger."""

    def __init__(self, level: int = logging.INFO, prefix: str = "MATLAB") -> None:
        super().__init__()
        self._level = level
        self._prefix = prefix
        self._buffer: List[str] = []
        self._captured: List[str] = []

    def write(self, s: str) -> int:  # pragma: no cover - exercised via MATLAB engine
        if not s:
            return 0
        written = super().write(s)
        parts = s.splitlines(keepends=True)
        for part in parts:
            self._buffer.append(part)
            if part.endswith("\n"):
                self._flush_buffer()
        return written

    def flush(self) -> None:  # pragma: no cover - exercised via MATLAB engine
        self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        message = text.rstrip()
        if message:
            logger.log(self._level, "%s: %s", self._prefix, message)
            self._captured.append(message)
        # reset underlying buffer so repeated writes don't grow without bound
        self.seek(0)
        self.truncate(0)

    def captured_text(self) -> str:
        """Return all text seen so far (joined by newlines)."""

        return "\n".join(self._captured).strip()


class MatlabRecipeErrorBudgetExceeded(RuntimeError):
    """Raised when a scenario exceeds its MATLAB error skip allowance."""

    def __init__(self, scenario_id: str, allowed_errors: int) -> None:
        msg = (
            f"Scenario '{scenario_id}' exceeded the MATLAB recipe error budget"
            f" (allowed={allowed_errors})."
        )
        super().__init__(msg)
        self.scenario_id = scenario_id
        self.allowed_errors = allowed_errors


class MatlabSegmentFailure(RuntimeError):
    """Raised when a segmented MATLAB run fails mid-segment."""

    def __init__(
        self,
        message: str,
        *,
        segment_failure: Optional[int],
        details: Optional[str],
        original_exception: Exception,
        failed_to_converge: Optional[bool] = False
    ) -> None:
        super().__init__(message)
        self.segment_failure = segment_failure
        self.details = details
        self.original_exception = original_exception
        self.failed_to_converge = failed_to_converge


class MatlabUserInterrupt(KeyboardInterrupt):
    """Raised when MATLAB reports a user-triggered cancellation."""


def is_matlab_user_interrupt_text(text: Any) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(pattern in normalized for pattern in _MATLAB_INTERRUPT_PATTERNS)


def _terminate_processes(processes: List["psutil.Process"], *, reason: str) -> None:
    if psutil is None or not processes:
        return

    aggregated: Dict[int, psutil.Process] = {}
    for proc in processes:
        if proc is None:
            continue
        if proc.pid in aggregated:
            continue
        aggregated[proc.pid] = proc
        try:
            for child in proc.children(recursive=True):
                if child.pid not in aggregated:
                    aggregated[child.pid] = child
        except Exception:
            continue

    ordered = list(aggregated.values())
    if not ordered:
        return

    for proc in ordered:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        _, alive = psutil.wait_procs(ordered, timeout=5.0)
    except Exception:
        alive = ordered

    for proc in alive:
        try:
            proc.kill()
        except Exception:
            pass

    if alive:
        info = ", ".join(f"{p.pid}:{p.name()}" for p in alive[:5])
        suffix = "" if len(alive) <= 5 else ", ..."
        logger.warning(
            "Forced kill of lingering processes for %s: %s%s",
            reason,
            info,
            suffix,
        )


def _collect_descendants_since(start_time: float) -> List["psutil.Process"]:
    if psutil is None:
        return []
    try:
        current = psutil.Process(os.getpid())
    except Exception:
        return []

    descendants: List[psutil.Process] = []
    try:
        for proc in current.children(recursive=True):
            try:
                created = proc.create_time()
            except Exception:
                created = None
            if created is None or created >= start_time - 1.0:
                descendants.append(proc)
    except Exception:
        return []
    return descendants


def _is_mathworks_process(info: Dict[str, Any]) -> bool:
    name = (info["name"] or "").lower()
    cmdline_raw = info["cmdline"]
    if isinstance(cmdline_raw, (list, tuple)):
        cmdline = " ".join(str(part) for part in cmdline_raw).lower()
    else:
        cmdline = str(cmdline_raw or "").lower()
    for token in _MATHWORKS_MATCHERS:
        if token in name or token in cmdline:
            return True
    return False


def _collect_mathworks_processes_since(start_time: float) -> List["psutil.Process"]:
    if psutil is None:
        return []
    lower_bound = start_time - _MATHWORKS_CLEANUP_WINDOW_S
    upper_bound = start_time + _MATHWORKS_CLEANUP_WINDOW_S
    matches: List[psutil.Process] = []
    for proc in psutil.process_iter(
        ["pid", "name", "cmdline", "username", "create_time"]
    ):
        try:
            info = proc.info
            if info["username"] != _CURRENT_USER:
                continue
            created = info["create_time"]
            if created is None:
                continue
            if created < lower_bound or created > upper_bound:
                continue
            if not _is_mathworks_process(info):
                continue
            matches.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def _reset_working_copy(
    *, working_model: str = "simulink_model.mdl", source_model: str = "simulink_model_original.mdl"
) -> None:
    """Refresh the working Simulink model from its pristine original copy."""

    working_path = Path(working_model)
    source_path = Path(source_model)
    if working_path.exists():
        working_path.unlink()
    shutil.copy(source_path, working_path)


def _purge_autosaves(model_base: str, where: Path = Path(".")) -> int:
    """Remove Simulink autosave artifacts for the given model within ``where``."""

    pats = [
        f"{model_base}_autosave.mdl*",
        f"{model_base}.mdl.autosave*",
        f"{model_base}.autosave*",
        f"{model_base}.asv",
        f"{model_base}*autosave*",
    ]
    removed = 0
    for pat in pats:
        for candidate in where.glob(pat):
            if candidate.name == f"{model_base}.mdl":
                continue
            try:
                candidate.unlink()
                removed += 1
            except Exception as exc:  # pragma: no cover - filesystem dependent
                logger.warning("Could not remove autosave candidate %s: %s", candidate, exc)
    return removed


def _global_process_cleanup() -> None:
    if psutil is None:
        return

    descendants = _collect_descendants_since(_SCRIPT_START_TIME)
    mathworks = _collect_mathworks_processes_since(_SCRIPT_START_TIME)

    aggregated: Dict[int, psutil.Process] = {proc.pid: proc for proc in descendants}
    for proc in mathworks:
        aggregated.setdefault(proc.pid, proc)

    if not aggregated:
        return

    logger.warning(
        "Cleaning up %d lingering processes left from this session.",
        len(aggregated),
    )
    _terminate_processes(list(aggregated.values()), reason="session shutdown")


def force_stop_matlab_processes(
    *,
    started_at: Optional[float] = None,
    reason: str,
) -> None:
    if psutil is None:
        return

    lower_bound = float(started_at) if started_at is not None else _SCRIPT_START_TIME
    descendants = _collect_descendants_since(lower_bound)
    mathworks = _collect_mathworks_processes_since(lower_bound)
    aggregated: Dict[int, psutil.Process] = {}
    for proc in descendants:
        aggregated.setdefault(proc.pid, proc)
    for proc in mathworks:
        aggregated.setdefault(proc.pid, proc)
    if not aggregated:
        return
    _terminate_processes(list(aggregated.values()), reason=reason)


if psutil is not None:
    atexit.register(_global_process_cleanup)


class _ProcessCleanupGuard:
    """Track descendant processes and clean up any that linger after work completes."""

    def __init__(self, label: str):
        self._label = label
        self._process = None
        self._baseline: set[int] = set()
        self._enabled = psutil is not None

    def __enter__(self) -> "_ProcessCleanupGuard":
        if self._enabled:
            try:
                self._process = psutil.Process(os.getpid())
                self._baseline = self._snapshot_descendants()
            except (
                Exception
            ) as exc:  # pragma: no cover - defensive: psutil may fail unexpectedly
                self._enabled = False
                logger.warning(
                    "Process tracking disabled for %s: %s",
                    self._label,
                    exc,
                )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def _snapshot_descendants(self) -> set[int]:
        if not self._process:
            return set()
        try:
            with self._process.oneshot():
                return {child.pid for child in self._process.children(recursive=True)}
        except Exception:
            return set()

    def cleanup(self) -> None:
        if not self._enabled or not self._process:
            return

        remaining = self._snapshot_descendants()
        leaked_pids = remaining - self._baseline
        if not leaked_pids:
            return

        leaked_processes = []
        for pid in leaked_pids:
            try:
                leaked_processes.append(psutil.Process(pid))
            except Exception:
                continue

        if not leaked_processes:
            return

        leaked_processes.sort(key=lambda proc: proc.pid)
        leaked_info = ", ".join(
            f"{proc.pid}:{proc.name()}" for proc in leaked_processes[:5]
        )
        extra = "" if len(leaked_processes) <= 5 else ", ..."
        logger.warning(
            "Cleaning up %d leaked processes for %s: %s%s",
            len(leaked_processes),
            self._label,
            leaked_info,
            extra,
        )

        _terminate_processes(leaked_processes, reason=self._label)


__all__ = [
    "MatlabUserInterrupt",
    "_MatlabStream",
    "_ProcessCleanupGuard",
    "_collect_descendants_since",
    "_collect_mathworks_processes_since",
    "_global_process_cleanup",
    "_is_mathworks_process",
    "_terminate_processes",
    "force_stop_matlab_processes",
    "is_matlab_user_interrupt_text",
]
