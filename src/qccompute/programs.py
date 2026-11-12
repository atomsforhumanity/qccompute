"""Running programs that implement ``qcdata.program.Program`` directly, without a codec.

AFH's own programs (linus, axtb, qcopt, ...) register an object under the
``qcdata.programs`` entry points; tests and notebooks add others with
``qcdata.program.register_program`` (also exported here). ``compute`` finds them with
``qcdata.program.find_program``, the same lookup drivers use, before codecs, and calls
``program.compute(input, context)`` in this process, with a `RunContext` carrying the
thread and GPU grant, a scratch directory, a callback for child calculations, and child
sessions. `session` opens a session for any program: its own when it has them, else one
over ``compute``.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any

from qcdata import (
    CalcType,
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
    Capabilities,
    ComputeSession,
    Plan,
    Program,
    ProgramError,
    RunContext,
    Session,
    SessionProgram,
    find_program,
    register_program,
    session_calctypes,
)

from .exceptions import (
    ExternalProgramError,
    InputError,
    ProgramNotFoundError,
    ProgramTimeoutError,
    QCComputeBaseError,
)
from .execution import construct_execution, empty_results, host_info

__all__ = [
    "register_program",
    "get_program",
    "run_program",
    "session",
    "NativeProgramRequest",
    "native_program_request",
]

# Keyed by the program object's id; the object is kept with its capabilities, so an id
# is never reused while cached and a newly registered program is read afresh.
_capabilities: dict[int, tuple[Program, Capabilities]] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class NativeProgramRequest:
    """What a native calculation needs from the list scheduler, from its plan.

    Attributes:
        threads: CPU threads.
        gpus: Number of GPUs.
        plan: The plan the resources came from, handed to the program so it does not
            plan again.
    """

    threads: int
    gpus: int
    plan: Plan


def native_program_request(input_data: InputType) -> NativeProgramRequest | None:
    """Plan a native calculation for running beside others in threads.

    Returns:
        Its resources when a thread-safe native program computes it and its plan
        succeeds; None otherwise (it then runs on its own, which reports any error).
    """
    if not isinstance(input_data, ProgramInput):
        return None
    program = get_program(input_data.program)
    if program is None or not capabilities(program).thread_safe:
        return None
    try:
        plan = program.plan(input_data)
    except Exception:  # noqa: BLE001 - computing it reports the error
        return None
    return NativeProgramRequest(
        threads=plan.resources.threads, gpus=plan.resources.gpus, plan=plan
    )


def capabilities(program: Program) -> Capabilities:
    """A program object's capabilities, read once per process (they do not change)."""
    with _lock:
        cached = _capabilities.get(id(program))
    if cached is not None and cached[0] is program:
        return cached[1]
    caps = program.capabilities()
    with _lock:
        _capabilities[id(program)] = (program, caps)
    return caps


def get_program(name: str) -> Program | None:
    """The native program of this name: registered in this process, else installed.

    The same lookup as ``qcdata.program.find_program``, which drivers use for their
    children.

    Args:
        name: The program's name, as in ``ProgramInput.program``.

    Returns:
        The program, or None when only a codec (or nothing) provides it.
    """
    return find_program(name)


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
    plan: Plan | None = None,
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
        plan: The plan already made for this input, handed to the program.

    Returns:
        The program's output, with execution information filled in.

    Raises:
        QCComputeBaseError: The calculation failed and ``raise_exc`` is True.
    """
    start = monotonic()

    def open_child_session(
        spec: ProgramSpec,
        structure: Structure,
        share: int | None,
        calctypes: frozenset[CalcType] | None,
    ) -> Session:
        return session(
            spec, structure, threads=share, calctypes=calctypes, compute=child_compute
        )

    made: list[Path] = []

    def make_scratch() -> Path:
        made.append(Path(tempfile.mkdtemp(prefix="qccompute-")))
        return made[-1]

    ctx = RunContext(
        threads=threads,
        gpus=gpus,
        scratch_dir=scratch_dir,
        make_scratch=make_scratch,
        plan=plan,
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
            ErrorInfo(kind=ErrorKind.internal, message=f"{type(exc).__name__}: {exc}"),
            "".join(traceback.format_exception(exc)),
            start,
        )
    finally:
        for directory in made:  # only directories this run asked for
            shutil.rmtree(directory, ignore_errors=True)
    # Keep the program's own timings and extras; add what the runner measured.
    # Keep the program's own timings and extras; add what the runner measured. The
    # program's output is trusted: copy it once, without validating it again.
    hostname, host_cpu = host_info()
    execution = type(output.execution).model_construct(
        _fields_set=output.execution.model_fields_set
        | {"wall_time", "hostname", "host_cpu", "scratch_dir"},
        **{
            **output.execution.__dict__,
            "wall_time": round(monotonic() - start, 6),
            "hostname": hostname,
            "host_cpu": host_cpu,
            "scratch_dir": scratch_dir,
        },
    )
    values: dict[str, Any] = {**output.__dict__, "execution": execution}
    if output.success and not collect_logs:
        values["logs"] = None
    output = type(output).model_construct(
        _fields_set=output.model_fields_set | {"execution"}, **values
    )
    if not output.success and raise_exc:
        raise _error_for(output)
    return output


def session(
    spec: ProgramSpec,
    structure: Structure,
    *,
    threads: int | None = None,
    calctypes: Iterable[CalcType | str] | None = None,
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
        calctypes: The calctypes the session may run (default: the spec's, with lower
            derivatives); the keywords are checked against them.
        compute: Computes inputs for a session over ``compute``; by default
            ``qccompute.compute`` with ``compute_options`` and ``raise_exc=False``.
        **compute_options: Options for ``qccompute.compute`` (``environment``,
            ``timeout``, ...), for sessions over it.

    Returns:
        The session; use it as a context manager, or close it.

    Raises:
        qcdata.program.ProgramError: The program rejects the spec.
    """
    allowed = session_calctypes(spec, calctypes)
    program = get_program(spec.program)
    if program is not None and isinstance(program, SessionProgram):
        if program.capabilities().sessions:
            return program.open_session(
                spec, structure, RunContext(threads=threads), allowed
            )
    if compute is None:
        from .main import compute as qccompute

        compute = partial(qccompute, raise_exc=False, **compute_options)
    return ComputeSession(compute, spec, structure, calctypes=allowed)


class _Children:
    """The child callback of a native calculation: ``compute`` within its grant.

    Cheap to make (most programs never call it); the inherited options are read on
    first use. Lists of children never go to worker processes (``compute_children``).
    """

    __slots__ = ("_options", "_threads", "_inherited")

    def __init__(self, options: Mapping[str, Any], threads: int | None) -> None:
        self._options = options
        self._threads = threads
        self._inherited: dict[str, Any] | None = None

    def __call__(self, inputs: Any, previous: Any = None) -> Any:
        from .main import compute as qccompute
        from .main import compute_children

        if self._inherited is None:
            self._inherited = {
                k: self._options[k]
                for k in ("environment", "timeout", "executables", "collect_logs")
                if k in self._options and self._options[k] is not None
            }
        if isinstance(inputs, ProgramInput):
            return qccompute(inputs, raise_exc=False, **self._inherited)
        return compute_children(inputs, max_workers=self._threads, **self._inherited)


def children_callback(
    options: Mapping[str, Any], threads: int | None
) -> Callable[..., Any]:
    """The child callback of a native calculation: ``compute`` within its grant."""
    return _Children(options, threads)
