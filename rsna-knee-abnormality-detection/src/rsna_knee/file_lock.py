"""Cross-process locks for append-only generated files."""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


@contextmanager
def output_file_lock(output_path: str | Path) -> Iterator[None]:
    """Prevent multiple processes from writing one output file."""
    output_path = Path(output_path)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        try:
            _lock(handle)
        except OSError as exc:
            raise RuntimeError(
                f"another labeling process is already using {output_path}"
            ) from exc
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()}\n".encode())
            handle.flush()
            yield
        finally:
            handle.seek(0)
            _unlock(handle)


def lock_output_argument(
    function: Callable[..., T],
) -> Callable[..., T]:
    """Lock the path supplied as the function's output_path argument."""
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args: object, **kwargs: object) -> T:
        arguments = signature.bind(*args, **kwargs)
        with output_file_lock(arguments.arguments["output_path"]):
            return function(*args, **kwargs)

    return wrapped


if os.name == "nt":
    import msvcrt

    def _lock(handle: object) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle: object) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle: object) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: object) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
