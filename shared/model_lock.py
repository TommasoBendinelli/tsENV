from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, TextIO, Union


class ModelLockError(RuntimeError):
    """Raised when a per-model lock cannot be acquired."""


def _lock_path_for_model(
    model_id: str, lock_root: Optional[Union[str, Path]] = None
) -> Path:
    model_key = str(model_id or "").strip()
    if not model_key:
        raise ValueError("model_id must be non-empty")
    root = (
        Path(lock_root)
        if lock_root is not None
        else Path(tempfile.gettempdir()) / "web_model_explorer" / "locks"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{model_key}.lock"


def _acquire_nonblocking_lock(lock_file: TextIO) -> None:
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ModelLockError("lock_busy") from exc


@contextmanager
def model_lock(
    model_id: str,
    *,
    purpose: str,
    lock_root: Optional[Union[str, Path]] = None,
) -> Iterator[Path]:
    """Acquire an exclusive per-model lock.

    The lock is process-wide (filesystem flock). If another process already holds it,
    raises ModelLockError.
    """
    lock_path = _lock_path_for_model(model_id, lock_root=lock_root)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        _acquire_nonblocking_lock(lock_file)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} purpose={purpose}\n")
        lock_file.flush()
        yield lock_path
    except ModelLockError:
        raise ModelLockError(
            f"Model '{model_id}' is locked by another process; cannot run {purpose}."
        )
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()
