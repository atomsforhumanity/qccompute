"""The worker wire's encoding, shared by the worker (`qccompute.serve`) and its client.

One JSON message per line. Models cross as their Python dump; with an array directory
(the connection's, named by the client in the ``options`` message), an array field over
1 MiB crosses as a ``.npy`` file there, with qcdata's pointer (``path``, ``shape``,
``dtype``, ``sha256``) in its place, exactly as a saved model's arrays do. The reader
checks each file against its pointer and deletes it once read.
"""

from __future__ import annotations

import json
import os
import secrets
import select
import time
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel
from qcdata.array_files import DIRECTORY, POINTER_KEYS, SINK, ArraySink

M = TypeVar("M", bound=BaseModel)


def plain(x: Any) -> Any:
    """``json``'s fallback: numpy values, paths, enums and sets as JSON values."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, set | frozenset):
        return sorted(x, key=str)
    raise TypeError(f"{type(x).__name__} is not JSON serializable")


def encode(message: dict[str, Any]) -> str:
    """One protocol line (without its newline); ValueError on NaN or Infinity."""
    return json.dumps(message, allow_nan=False, separators=(",", ":"), default=plain)


def _python(value: Any, context: dict[str, Any] | None) -> Any:
    if isinstance(value, list):
        return [_python(v, context) for v in value]
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python", context=context)
    return value


def dump(value: Any, directory: Path | None) -> Any:
    """A model, or a list of them, as values for `encode`.

    With a directory, array fields over 1 MiB are written there as ``.npy`` files and
    replaced by their pointers. Arrays holding NaN or infinity stay inline, so the
    wire refuses them as it refuses any non-finite number.
    """
    if directory is None:
        return _python(value, None)
    sink = ArraySink()
    document = _python(value, {SINK: sink})
    if not sink.arrays:
        return document
    if not all(np.isfinite(a).all() for a in sink.arrays):
        return _python(value, None)
    placed, _ = sink.place(document, directory, secrets.token_hex(8))
    return placed


def load(cls: type[M], data: Any, directory: Path | None) -> M:
    """Validate a model from the wire, reading (then deleting) its array files."""
    try:
        return cls.model_validate(
            data, context=None if directory is None else {DIRECTORY: directory}
        )
    finally:
        if directory is not None:
            for name in _pointers(data):
                (directory / name).unlink(missing_ok=True)


def _pointers(node: Any) -> Iterator[str]:
    """The file names of the array pointers in a document."""
    if isinstance(node, dict):
        if set(node) == POINTER_KEYS:
            name = node["path"]
            if isinstance(name, str) and Path(name).name == name:
                yield name
            return
        for v in node.values():
            yield from _pointers(v)
    elif isinstance(node, list) and node and isinstance(node[0], dict | list):
        for v in node:  # not a list of numbers (an array's rows end in numbers)
            yield from _pointers(v)


class Lines:
    """Lines from a pipe's file descriptor, with an optional deadline per line.

    Reads the descriptor directly, so a line already read past is never waited on
    (``select`` on a buffered file object can miss it).
    """

    def __init__(self, fd: int) -> None:
        """Read lines from ``fd``."""
        self._fd = fd
        self._buffer = bytearray()
        self._scanned = 0

    def readline(self, timeout: float | None = None) -> str | None:
        """The next line (with its newline); ``""`` at end of file; None on timeout.

        Args:
            timeout: Seconds to wait for the line; None: no limit.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            end = self._buffer.find(b"\n", self._scanned)
            if end >= 0:
                line = bytes(self._buffer[: end + 1])
                del self._buffer[: end + 1]
                self._scanned = 0
                return line.decode("utf-8")
            self._scanned = len(self._buffer)
            if deadline is not None:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([self._fd], [], [], left)[0]:
                    return None
            chunk = os.read(self._fd, 1 << 20)
            if not chunk:
                return ""
            self._buffer += chunk
