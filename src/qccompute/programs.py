"""Running programs that implement ``qcdata.program.Program`` directly, without a codec.

AFH's own programs (linus, axtb, qcopt, ...) register an object under the
``qcdata.programs`` entry points. ``compute`` finds them before codecs and calls
``program.compute(input, context)`` in this process, with a `RunContext` carrying the
thread and GPU grant, a scratch directory, a callback for child calculations, and child
sessions. `session` opens a session for any program: its own when it has them, else one
over ``compute``.
"""

from __future__ import annotations

import tempfile
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any

from qcdata import (
    Data,
    ErrorInfo,
    ErrorKind,
    InputType,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Structure,
)
from qcdata.program import (
    ComputeSession,
    Program,
    ProgramError,
    RunContext,
    Session,
    SessionProgram,
    find_program,
)

from .exceptions import (
    ExternalProgramError,
    InputError,
    ProgramNotFoundError,
    ProgramTimeoutError,
    QCComputeBaseError,
)
from .execution import construct_execution, empty_results

__all__ = [
    "register_program",
    "get_program",
    "run_program",
    "session",
    "NativeProgramRequest",
    "native_program_request",
]

_registered: dict[str, Program] = {}
_found: dict[str, Program | None] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class NativeProgramRequest:
    """What a native calculation needs from the list scheduler, from its plan.

    Attributes:
        threads: CPU threads.
        gpus: Number of GPUs.
    """

    threads: int
    gpus: int


def native_program_request(input_data: InputType) -> NativeProgramRequest | None:
    """Plan a native calculation for running beside others in threads.

    Returns:
        Its resources when a thread-safe native program computes it and its plan
        succeeds; None otherwise (it then runs on its own, which reports any error).
    """
    if not isinstance(input_data, ProgramInput):
        return None
    program = get_program(input_data.program)
    if program is None or not program.capabilities().thread_safe:
        return None
    try:
        resources = program.plan(input_data).resources
    except Exception:  # noqa: BLE001 - computing it reports the error
        return None
    return NativeProgramRequest(threads=resources.threads, gpus=resources.gpus)


def register_program(program: Program) -> None:
    """Make a program available to ``compute`` in this process, ahead of entry points.

    Args:
        program: An object implementing ``qcdata.program.Program``.

    Raises:
        TypeError: The object does not implement the protocol.
    """
    if not isinstance(program, Program):
        raise TypeError(f"{program!r} does not implement qcdata.program.Program.")
    with _lock:
        _registered[program.name] = program


def get_program(name: str) -> Program | None:
    """The native program of this name: registered here, else installed.

    Args:
        name: The program's name, as in ``ProgramInput.program``.

    Returns:
        The program, or None when only a codec (or nothing) provides it.
    """
    with _lock:
        if name in _registered:
            return _registered[name]
        if name not in _found:
            _found[name] = find_program(name)
        return _found[name]


def _error_for(output: ProgramOutput[Any, Any]) -> QCComputeBaseError:
    """The exception ``compute`` raises for a failed native calculation."""
    program = output.input_data.program
    error = output.error
    message = error.message if error is not None else (output.traceback or "failed")
    kind = error.kind if error is not None else ErrorKind.internal
    exc: QCComputeBaseError
    if kind in (ErrorKind.input, ErrorKind.unsupported):
        exc = InputError(program, message)
    elif kind == ErrorKind.timeout:
        exc = ProgramTimeoutError(program, message)
    elif kind == ErrorKind.environment:
        exc = ProgramNotFoundError(program, message)
    else:
        exc = ExternalProgramError(program, message)
    exc.prog_output = output
    exc.data = output.results
    return exc


def _failed_output(
    input_data: InputType, error: ErrorInfo, trace: str, start: float
) -> ProgramOutput[InputType, Data]:
    return ProgramOutput(
        input_data=input_data,
        results=empty_results(input_data),
        success=False,
        traceback=trace,
        error=error,
        execution=construct_execution(None, monotonic() - start),
    )


def run_program(
    program: Program,
    input_data: ProgramInput,
    *,
    raise_exc: bool = True,
    collect_logs: bool = True,
    threads: int | None = None,
    gpus: tuple[str, ...] = (),
    scratch_dir: Path | None = None,
    child_compute: Callable[..., Any] | None = None,
) -> ProgramOutput[ProgramInput, Data]:
    """Run a native program in this process.

    Args:
        program: The program.
        input_data: Its input.
        raise_exc: Raise for a failed calculation, with the output attached.
        collect_logs: Keep the program's logs on success too.
        threads: CPU threads granted (None: the program's resolved settings).
        gpus: GPU identifiers granted.
        scratch_dir: A directory to work in; a temporary one is made otherwise and
            removed afterwards.
        child_compute: Computes child calculations (``compute`` with this
            calculation's options); also backs child sessions of programs without
            their own.

    Returns:
        The program's output, with execution information filled in.

    Raises:
        QCComputeBaseError: The calculation failed and ``raise_exc`` is True.
    """
    start = monotonic()

    def open_child_session(
        spec: ProgramSpec, structure: Structure, share: int | None
    ) -> Session:
        return session(spec, structure, threads=share, compute=child_compute)

    with tempfile.TemporaryDirectory(prefix="qccompute-") as tmp:
        directory = scratch_dir or Path(tmp)
        ctx = RunContext(
            threads=threads,
            gpus=gpus,
            scratch=directory,
            compute=child_compute,  # type: ignore[arg-type]
            open_session=open_child_session,
        )
        try:
            output = program.compute(input_data, ctx)
        except ProgramError as exc:  # a program may raise instead of returning
            output = _failed_output(
                input_data, exc.error, "".join(traceback.format_exception(exc)), start
            )
        except Exception as exc:  # noqa: BLE001 - a bug in the program
            output = _failed_output(
                input_data,
                ErrorInfo(
                    kind=ErrorKind.internal, message=f"{type(exc).__name__}: {exc}"
                ),
                "".join(traceback.format_exception(exc)),
                start,
            )
    # Keep the program's own timings and extras; add what the runner measured.
    measured = construct_execution(scratch_dir, monotonic() - start)
    execution = output.execution.model_copy(
        update={
            "wall_time": measured.wall_time,
            "hostname": measured.hostname,
            "host_cpu": measured.host_cpu,
            "scratch_dir": scratch_dir,
        }
    )
    update: dict[str, Any] = {"execution": execution}
    if output.success and not collect_logs:
        update["logs"] = None
    output = output.model_copy(update=update)
    if not output.success and raise_exc:
        raise _error_for(output)
    return output


def session(
    spec: ProgramSpec,
    structure: Structure,
    *,
    threads: int | None = None,
    compute: Callable[..., Any] | None = None,
    **compute_options: Any,
) -> Session:
    """Open a session: one program and molecule at many geometries.

    Programs with sessions of their own (``capabilities().sessions``) open one, pinned
    to the spec's settings and warm-started between geometries. For any other program
    the session runs each geometry through ``compute``.

    Args:
        spec: The program, model, and keywords; ``spec.calctype`` is the default
            request type.
        structure: The atoms, charge, multiplicity, and first geometry.
        threads: The session's CPU threads (None: the program's settings).
        compute: Computes inputs for a session over ``compute``; by default
            ``qccompute.compute`` with ``compute_options`` and ``raise_exc=False``.
        **compute_options: Options for ``qccompute.compute`` (``environment``,
            ``timeout``, ...), for sessions over it.

    Returns:
        The session; use it as a context manager, or close it.

    Raises:
        qcdata.program.ProgramError: The program rejects the spec.
    """
    program = get_program(spec.program)
    if program is not None and isinstance(program, SessionProgram):
        if program.capabilities().sessions:
            return program.open_session(spec, structure, RunContext(threads=threads))
    if compute is None:
        from .main import compute as qccompute

        compute = partial(qccompute, raise_exc=False, **compute_options)
    return ComputeSession(compute, spec, structure)


def children_callback(
    options: Mapping[str, Any], threads: int | None
) -> Callable[..., Any]:
    """The child callback of a native calculation: ``compute`` within its grant."""
    from .main import compute as qccompute

    inherited = {
        k: options[k]
        for k in ("environment", "timeout", "executables", "collect_logs")
        if k in options and options[k] is not None
    }
    return partial(qccompute, raise_exc=False, max_workers=threads, **inherited)
