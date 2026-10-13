"""Execute prepared invocations without knowing any scientific program conventions."""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from time import monotonic
from typing import TextIO

from qccodec import PythonInput, PythonOutput, SubprocessInput, SubprocessOutput
from qccodec.native import ComputeChild

from .exceptions import ProgramNotFoundError

_python_lock = threading.RLock()


def _flush_native_output() -> None:
    """Flush C stdio while the intended file descriptors are still installed."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError):
        pass


def run_subprocess(
    request: SubprocessInput,
    directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    update_func: Callable[[str, str], None] | None = None,
    update_interval: float | None = None,
) -> SubprocessOutput:
    """Execute a native command and monitor its merged stdout/stderr.

    Run in the calculation directory without changing the parent process's working
    directory or environment. Standard error is redirected into standard output,
    and stdin is closed. Output is read until EOF, including any final text without
    a trailing newline.

    Args:
        request: The prepared command and invocation environment defaults. Input
            files must already have been written to the calculation directory.
        directory: The working directory for the child process.
        environment: Overrides applied after the inherited environment and request
            defaults. These affect only the child process.
        update_func: Optional callback accepting two strings: all captured output
            and the new output since the previous callback. For console output,
            use ``lambda all_logs, new_logs: print(new_logs, end="")``. Any remaining
            output is delivered after the child exits, including on a nonzero exit.
            No callback is made if the child produces no output.
        update_interval: Minimum seconds between live callbacks; defaults to 0.5.
            Updates are line-based, so an unterminated line waits until a newline
            or EOF. This interval is not a heartbeat. The final callback is not
            delayed by the interval.

    Returns:
        Native execution observations with merged output in ``stdout``, the working directory, and the return code. Launch, reading,
        or callback exceptions are recorded in ``error`` and ``traceback``, with
        output read so far retained. A nonzero exit is returned for the caller to
        interpret, allowing decoding of partial scientific results.

    The immediate child is terminated and waited for if execution is interrupted
    before it exits. Scientific decoding and scratch-directory cleanup belong to
    the caller.
    """
    output = SubprocessOutput(directory=directory)
    proc = None
    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            request.command,
            cwd=directory,
            env={**os.environ, **request.environment, **(environment or {})},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        last_update = monotonic()
        reported = 0
        interval = 0.5 if update_interval is None else update_interval
        for line in proc.stdout:
            lines.append(line)
            if update_func and monotonic() - last_update >= interval:
                update_func("".join(lines), "".join(lines[reported:]))
                reported = len(lines)
                last_update = monotonic()
        output.returncode = proc.wait()
        if update_func and len(lines) > reported:
            update_func("".join(lines), "".join(lines[reported:]))
    except Exception as exc:
        output.error = exc
        output.traceback = traceback.format_exc()
        if (
            proc is None
            and isinstance(exc, FileNotFoundError)
            and exc.filename == request.command[0]
        ):
            # Popen identifies a missing executable separately from a missing cwd.
            output.error = ProgramNotFoundError(request.command[0], str(exc))
            output.error.__cause__ = exc
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            output.returncode = proc.wait()
            if proc.stdout is not None:
                proc.stdout.close()
        output.stdout = "".join(lines)
    return output


@contextmanager
def _python_context(
    directory: Path,
    environment: Mapping[str, str],
    loggers: tuple[str, ...],
    capture: TextIO,
) -> Iterator[None]:
    """Capture Python and native output while temporarily preparing process state.

    Args:
        directory: Working directory for the invocation.
        environment: Environment overrides to restore afterward.
        loggers: Named loggers whose handlers temporarily write into capture.
        capture: Caller-owned file receiving both stdout and stderr. It remains
            open after this context exits so the caller can read the completed log.

    Yields:
        Control with Python streams, native descriptors, and loggers redirected.

    The reentrant lock permits synchronous nested calculations. It serializes
    Python runners within this process, but does not isolate unrelated threads.
    Each restoration is registered separately so cleanup continues if one fails.
    """
    with _python_lock, ExitStack() as restore:
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        _flush_native_output()
        restore.callback(os.chdir, Path.cwd())
        os.chdir(directory)
        for name, value in environment.items():
            if name in os.environ:
                restore.callback(os.environ.__setitem__, name, os.environ[name])
            else:
                restore.callback(os.environ.pop, name, None)
            os.environ[name] = value
        for fd in (1, 2):
            saved = os.dup(fd)
            restore.callback(os.close, saved)
            restore.callback(os.dup2, saved, fd)
            os.dup2(capture.fileno(), fd)
        for name in loggers:
            logger = logging.getLogger(name)
            restore.callback(setattr, logger, "handlers", logger.handlers[:])
            restore.callback(logger.setLevel, logger.level)
            restore.callback(setattr, logger, "propagate", logger.propagate)
            logger.handlers = [logging.StreamHandler(capture)]
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
        restore.enter_context(redirect_stdout(capture))
        restore.enter_context(redirect_stderr(capture))
        try:
            yield
        finally:
            _flush_native_output()
            capture.flush()


def run_python(
    request: PythonInput,
    directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    compute_child: ComputeChild | None = None,
    update_func: Callable[[str, str], None] | None = None,
) -> PythonOutput:
    """Invoke a callable and retain its return value, merged logs, or failure.

    Args:
        request: Prepared callable, arguments, and capture requirements.
        directory: Calculation directory, with input files already staged.
        environment: Overrides applied after the invocation's environment defaults.
        compute_child: Synchronous calculation callback injected into the keyword
            named by request.child_callback, when requested by the integration.
        update_func: Optional callback receiving the completed log twice, as both
            all output and new output. Called once for nonempty logs after capture
            is restored; Python execution does not provide live updates.

    Returns:
        Native observations with the callable's value, merged output in stdout,
        and exceptions. Exceptions are recorded with their traceback and logs;
        a reporting or cleanup error does not replace an earlier invocation error.

    Native writes to descriptors 1 and 2 are captured alongside Python output.
    C stdio is flushed before restoration; bindings remain responsible for other
    runtime-specific buffers. Use separate processes for concurrent calculations.
    """
    output = PythonOutput(directory=directory)
    try:
        overrides = {**request.environment, **(environment or {})}
        with tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        ) as capture:
            try:
                with _python_context(directory, overrides, request.loggers, capture):
                    try:
                        kwargs = dict(request.kwargs)
                        if request.child_callback:
                            if (
                                request.child_callback in kwargs
                                or compute_child is None
                            ):
                                raise ValueError(
                                    "Missing or conflicting child calculation callback."
                                )
                            kwargs[request.child_callback] = compute_child
                        output.value = request.function(*request.args, **kwargs)
                    except Exception as exc:
                        output.error = exc
                        output.traceback = traceback.format_exc()
            finally:
                capture.seek(0)
                output.stdout = capture.read()
        if update_func and output.stdout:
            update_func(output.stdout, output.stdout)
    except Exception as exc:
        if output.error is None:
            output.error = exc
            output.traceback = traceback.format_exc()
    return output
