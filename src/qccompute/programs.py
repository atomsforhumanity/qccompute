"""Running programs that implement `qccompute.program.Program` directly, without a codec.

AFH's own programs (linus, axtb, qcopt, ...) register an object under the
``qccompute.programs`` entry points; tests and notebooks add others with
`register_program` (also exported here). ``compute`` finds them with
`qccompute.program.find_program`, the same lookup drivers use, before codecs, and calls
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
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    Frame,
    InputType,
    ProgramError,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Request,
    Structure,
)

from .exceptions import (
    ExternalProgramError,
    InputError,
    ProgramNotFoundError,
    ProgramTimeoutError,
    QCComputeBaseError,
)
from .execution import construct_execution, empty_results, host_info
from .program import (
    Capabilities,
    ComputeSession,
    Plan,
    Program,
    RunContext,
    Session,
    SessionProgram,
    capabilities_of,
    find_program,
    plan_of,
    register_program,
    session_calctypes,
)

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
_locks: dict[int, tuple[Program, threading.Lock]] = {}
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
        plan = plan_of(program, input_data)
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
    caps = capabilities_of(program)
    with _lock:
        _capabilities[id(program)] = (program, caps)
    return caps


def get_program(name: str) -> Program | None:
    """The native program of this name: registered in this process, else installed.

    The same lookup as `qccompute.program.find_program`, which drivers use for their
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
        inp: ProgramInput, share: int | None, calctypes: frozenset[CalcType] | None
    ) -> Session:
        return session(inp, threads=share, calctypes=calctypes, compute=child_compute)

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
        | {"wall_time", "hostname", "host_cpu", "scratch_dir", "threads"},
        **{
            **output.execution.__dict__,
            "wall_time": round(monotonic() - start, 6),
            "hostname": hostname,
            "host_cpu": host_cpu,
            "scratch_dir": scratch_dir,
            "threads": output.execution.threads
            if output.execution.threads is not None
            else threads,
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
    spec: ProgramInput | ProgramSpec,
    structure: Structure | None = None,
    *,
    threads: int | None = None,
    calctypes: Iterable[CalcType | str] | None = None,
    compute: Callable[..., Any] | None = None,
    **compute_options: Any,
) -> Session:
    """Open a session: one program and molecule at many geometries.

    Programs with sessions of their own (``capabilities().sessions``) open one, pinned
    to the input's settings and warm-started between geometries; it is the program's
    own session object (``linus.Session``), exactly what a direct user gets. For any
    other program the session runs each geometry through ``compute``.

    Args:
        spec: The input (program, model, keywords, structure; its calctype and
            geometry are the defaults of a request), or a spec with ``structure``.
        structure: With a spec: the atoms, charge, multiplicity and first geometry.
        threads: The session's CPU threads (None: the program's settings).
        calctypes: The calctypes the session may run (default: everything the method
            supports; declaring narrows it); the keywords are checked against them.
        compute: Computes inputs for a session over ``compute``; by default
            ``qccompute.compute`` with ``compute_options`` and ``raise_exc=False``.
        **compute_options: Options for ``qccompute.compute`` (``environment``,
            ``timeout``, ...), for sessions over it.

    Returns:
        The session; use it as a context manager, or close it.

    Raises:
        qcdata.ProgramError: The program rejects the input.
    """
    if isinstance(spec, ProgramInput):
        inp = spec
    else:
        if structure is None:
            raise TypeError("A session from a ProgramSpec needs its structure.")
        inp = ProgramInput.from_spec(spec, structure)
    allowed = session_calctypes(calctypes) if calctypes else None
    program = get_program(inp.program)
    opened: Session
    if (
        program is not None
        and isinstance(program, SessionProgram)
        and capabilities(program).sessions
    ):
        opened = program.open_session(inp, RunContext(threads=threads), allowed)
    else:
        if program is not None and allowed is None:
            caps = capabilities(program)
            method = inp.model.method if inp.model is not None else None
            supported = (caps.methods or {}).get(method or "", caps.calctypes)
            allowed = session_calctypes(None, supported & session_calctypes())
        if compute is None:
            from .main import compute as qccompute

            compute = partial(qccompute, raise_exc=False, **compute_options)
        opened = ComputeSession(compute, inp, calctypes=allowed)
    if program is not None and not capabilities(program).thread_safe:
        return _Serialized(opened, _program_lock(program))
    return opened


def _program_lock(program: Program) -> threading.Lock:
    """One lock per program object, shared by all its sessions in this process."""
    with _lock:
        cached = _locks.get(id(program))
        if cached is None or cached[0] is not program:
            cached = (program, threading.Lock())
            _locks[id(program)] = cached
    return cached[1]


class _Serialized:
    """A session of a program that is not thread-safe, safe to call from threads.

    Drivers may call the sessions they open from several threads at once (ten TS
    searches side by side); the runner, not the driver, makes that safe. Sessions of a
    program that does not declare ``thread_safe`` take turns on one lock per program;
    codec children (separate processes) and thread-safe programs run side by side.
    """

    def __init__(self, session: Session, lock: threading.Lock):
        self._session = session
        self._lock = lock

    @property
    def program(self) -> str:
        """The session's program."""
        return self._session.program

    @property
    def program_version(self) -> str:
        """The session's program version."""
        return self._session.program_version

    def compute(self, *args: Any, **kwargs: Any) -> Frame:
        """See `qccompute.program.Session.compute`."""
        with self._lock:
            return self._session.compute(*args, **kwargs)

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """See `qccompute.program.Session.compute_many`."""
        with self._lock:
            return self._session.compute_many(requests)

    def reset(self, lineage: Any = ...) -> None:
        """See `qccompute.program.Session.reset`."""
        with self._lock:
            self._session.reset(lineage)

    def plan(self) -> Plan:
        """See `qccompute.program.Session.plan`."""
        return self._session.plan()

    def close(self) -> None:
        """Close the session."""
        with self._lock:
            self._session.close()

    def __enter__(self) -> _Serialized:
        """Use the session as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the session."""
        self.close()

    def __getattr__(self, name: str) -> Any:
        """The program's own extras on its session (``calctypes``, ...)."""
        return getattr(self._session, name)


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
