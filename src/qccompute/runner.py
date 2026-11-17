"""The Runner: one way to run programs, one input or many, in this process or in workers.

``qccompute.compute`` and ``qccompute.session`` are the methods of the default runner,
made on first use for this machine (all its usable cores and visible GPUs). A `Runner`
of your own takes a share of the machine and closes its worker processes when you are
done with it::

    with qccompute.Runner(threads=16) as runner:
        outputs = runner.compute(inputs, store=store, tags=["screen"])

Every input goes one path: find the program by name, plan it (the threads it asks for,
and how to pack a list), place it (this process, or a worker process for programs that
cannot share one), call ``compute`` or ``compute_batch`` with a
`qccompute.program.RunContext`, and record the execution. A calculation gets the threads
it asks for: its ``threads`` keyword, else its plan's (1 unless the program says
otherwise; None: all of them), at most the runner's. A failed calculation raises `qcdata.ProgramError` with
the failed output as its ``output``; a list runs everything first and raises
`qcdata.ProgramErrors`.
"""

from __future__ import annotations

import atexit
import io
import math
import os
import pickle
import platform
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
import warnings
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from pydantic import ValidationError
from qcdata import (
    CalcType,
    ErrorInfo,
    ErrorKind,
    FileInput,
    Frame,
    ProgramError,
    ProgramErrors,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Request,
    Structure,
)

from .program import (
    FILES,
    BatchProgram,
    Capabilities,
    ComputeSession,
    Plan,
    Program,
    ProgressCallback,
    RunContext,
    Session,
    SessionProgram,
    _cuda_visible_devices,
    _entry_points,
    _gpu_row,
    _nvidia_smi_gpus,
    _Progress,
    capabilities_of,
    failed_frame,
    find_program,
    numerics_of,
    plan_of,
    session_calctypes,
)

if TYPE_CHECKING:
    from .worker import Worker

__all__ = [
    "Runner",
    "Store",
    "FILES",
    "compute",
    "submit",
    "session",
    "default_runner",
]

Input = ProgramInput | FileInput
Output = ProgramOutput[Any, Any]

# Kinds a stored failure is reused for: determined by the input, so the same code fails
# them the same way (kind program counts when its cause is one of these).
_DETERMINISTIC = frozenset(
    {ErrorKind.input, ErrorKind.unsupported, ErrorKind.convergence}
)


def _deterministic(error: ErrorInfo | None) -> bool:
    """Whether a stored failure is determined by its input, and so reusable."""
    while error is not None and error.kind is ErrorKind.program:
        error = error.cause
    return error is not None and error.kind in _DETERMINISTIC


class Store(Protocol):
    """What a runner needs of a calculation store (``qcstore.Store`` implements it).

    A key names a calculation: ``qcdata.calculation_key(inp)``, the input as given.
    Which code produced a run's numbers is kept apart from the key, as the run's
    identities: ``{program: numerics}`` for the input's program and every subprogram it
    names, which the runner sets on every output it returns
    (``output.execution.identities``).
    """

    def key(self, inp: Input) -> str:
        """The key of an input."""
        ...

    def get(self, key: str, identities: Mapping[str, str]) -> Output | None:
        """The preferred run of a key made with exactly these identities, if any."""
        ...

    def put(
        self,
        output: Output,
        *,
        tags: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        """Store a run under its input's key and ``output.execution.identities``."""
        ...

    def tag(self, key: str, *tags: str) -> None:
        """Add tags to a key."""
        ...


# ----------------------------------------------------------------------- the machine


def _cgroup_dirs() -> list[Path]:
    """This process's cgroup v2 directory and its ancestors, innermost first."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return []
    for line in lines:
        if line.startswith("0::"):
            path = Path("/sys/fs/cgroup") / line[3:].strip().lstrip("/")
            return [path, *path.parents][: len(path.parts) - 3]
    return []


def usable_cores() -> int:
    """CPU cores this process may use: its affinity, capped by any cgroup CPU quota."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not on macOS or Windows
        cores = os.cpu_count() or 1
    for directory in _cgroup_dirs():
        try:
            quota, period = (directory / "cpu.max").read_text().split()
        except (OSError, ValueError):
            continue
        if quota != "max":
            cores = min(cores, max(1, math.ceil(int(quota) / int(period))))
    return cores


def memory_gib() -> float | None:
    """Memory this process may use (GiB): its cgroup limit, else the machine's RAM."""
    limit: float | None = None
    for directory in _cgroup_dirs():
        try:
            value = (directory / "memory.max").read_text().strip()
        except OSError:
            continue
        if value != "max":
            limit = min(limit or math.inf, int(value) / 2**30)
    if limit is not None:
        return limit
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (AttributeError, ValueError, OSError):
        return None


def _pci_bus_order() -> None:
    """Have CUDA number GPUs as ``nvidia-smi`` does, when the runner names them so.

    With ``CUDA_VISIBLE_DEVICES`` unset, a runner's GPUs are ``nvidia-smi``'s indices,
    which follow PCI bus order; CUDA's own order may differ unless
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``. Set it, unless it is set, so the indices mean
    the same devices in CUDA, in this process and its workers. With
    ``CUDA_VISIBLE_DEVICES`` set, the dispatcher's choice stands untouched.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def visible_gpus() -> dict[str, float | None]:
    """The GPUs this process may use and their memory (GiB, if known).

    ``CUDA_VISIBLE_DEVICES`` wins when set (its indices or UUIDs); otherwise
    ``nvidia-smi`` lists the NVIDIA GPUs by index, and ``CUDA_DEVICE_ORDER`` is set to
    ``PCI_BUS_ID`` (unless set) so CUDA numbers them alike. Empty when there are none
    or they cannot be found.
    """
    listed = _cuda_visible_devices()
    if listed == []:
        return {}
    if listed is None:
        _pci_bus_order()
    if listed is None:
        return {row[0]: row[2] for row in _nvidia_smi_gpus()}
    return {gpu: (row[2] if (row := _gpu_row(gpu)) else None) for gpu in listed}


def _cpu_model() -> str | None:
    """The CPU's model name as the operating system reports it, if it does."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                name, _, value = line.partition(":")
                if name.strip() == "model name" and value.strip():
                    return value.strip()
    except OSError:
        pass
    return platform.processor() or None


def _host() -> dict[str, Any]:
    """This machine's environment for `ExecutionInfo`, read once per process.

    Its name, logical CPU count, RAM (GiB), CPU model and C library (``glibc 2.39``);
    each None when it cannot be read.
    """
    global _HOST
    if _HOST is None:
        try:
            ram: float | None = (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
            )
        except (AttributeError, ValueError, OSError):
            ram = None
        libc = " ".join(part for part in platform.libc_ver() if part)
        _HOST = {
            "hostname": platform.node() or None,
            "host_cpu": os.cpu_count(),
            "host_mem_gib": ram,
            "cpu_model": _cpu_model(),
            "libc": libc or None,
        }
    return _HOST


_HOST: dict[str, Any] | None = None


def _device_names(devices: Sequence[str]) -> list[str]:
    """The granted GPUs' names, as ``nvidia-smi`` gives them (else their identifiers)."""
    return [row[3] if (row := _gpu_row(gpu)) else gpu for gpu in devices]


# --------------------------------------------------------------------- the programs


class _Entry:
    """A program as a runner places it: its capabilities, lock and worker spec."""

    __slots__ = ("program", "caps", "lock", "_spec")

    def __init__(self, program: Program, caps: Capabilities) -> None:
        self.program = program
        self.caps = caps
        self.lock = (
            None if caps.thread_safe else _Turns(_program_lock(program), program.name)
        )
        self._spec: str | None = None

    @property
    def spec(self) -> str:
        """How a worker process finds the program (`_worker_spec`), on first use."""
        if self._spec is None:
            self._spec = _worker_spec(self.program)
        return self._spec


class _Turns:
    """A program's lock as one runner takes it: the first wait is a warning.

    Calculations of a program that is not thread-safe take turns in this process.
    When one has to wait for another (threads calling ``compute``, futures, a list),
    they run one at a time where the caller may expect them side by side, so the
    runner warns once per program, naming the way out.
    """

    __slots__ = ("_lock", "_name", "_warned")

    def __init__(self, lock: threading.Lock, name: str) -> None:
        self._lock = lock
        self._name = name
        self._warned = False

    def __enter__(self) -> None:
        """Take the program's turn, warning the first time it has to wait."""
        if self._lock.acquire(blocking=False):
            return
        if not self._warned:
            self._warned = True
            warnings.warn(
                f"Calculations of {self._name} are waiting their turn: it is not "
                "thread-safe, so this process runs them one at a time. "
                "Runner(isolation='process') runs them side by side, each in a "
                "worker process.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        """Give the turn back."""
        self._lock.release()


_locks: dict[int, tuple[Program, threading.Lock]] = {}
_locks_guard = threading.Lock()


def _program_lock(program: Program) -> threading.Lock:
    """One lock per program object in this process, shared by every runner."""
    with _locks_guard:
        held = _locks.get(id(program))
        if held is None or held[0] is not program:
            held = (program, threading.Lock())
            _locks[id(program)] = held
    return held[1]


#: A worker spec naming a file that holds the program pickled (`_worker_spec`).
PICKLED = "pickle:"

_specs: dict[int, tuple[Program, str]] = {}
_pickles: str | None = None  # this process's directory of pickled programs


def _worker_spec(program: Program) -> str:
    """How a worker process finds a program: its name, ``module:attribute``, or a pickle.

    An installed program (the object its entry point names) is found by name; a
    registered one that is a top-level attribute of its class's module, by that.
    Any other instance (one made in a script, ``register_program(Program(...))``) is
    pickled once per process into a private directory, removed at exit, and the
    worker loads it: its class and everything it holds must be importable there.

    Raises:
        ProgramError: Kind ``environment``: the program cannot be pickled for a
            worker (it holds a lambda, a closure, a lock, or something defined in
            the script being run).
    """
    with _locks_guard:
        held = _specs.get(id(program))
        if held is not None and held[0] is program:
            return held[1]
        spec = _find_spec(program)
        _specs[id(program)] = (program, spec)
        return spec


def _find_spec(program: Program) -> str:
    """`_worker_spec`, uncached (call under ``_locks_guard``)."""
    global _pickles
    entry = _entry_points().get(program.name)
    if entry is not None and entry.load() is program:
        return program.name
    module = type(program).__module__
    if module != "__main__":
        for name, value in vars(sys.modules.get(module, object())).items():
            if value is program:
                return f"{module}:{name}"
    try:
        data = _portable_pickle(program)
    except Exception as e:  # noqa: BLE001 - PicklingError, TypeError, AttributeError
        raise ProgramError.of(
            "environment",
            f"{program.name} is registered in this process, but a worker process "
            f"cannot rebuild it: {e}. Run it in this process (isolation='auto', for "
            "a program that may share it), make it a top-level attribute of an "
            "importable module, or install it under the qccompute.programs entry "
            "points.",
        ) from None
    if _pickles is None:
        _pickles = tempfile.mkdtemp(prefix="qccompute-programs-")
        atexit.register(shutil.rmtree, _pickles, ignore_errors=True)
    path = Path(_pickles) / f"{len(_specs)}-{program.name}.pickle"
    path.write_bytes(data)
    return f"{PICKLED}{path}"


class _Portable(pickle.Pickler):
    """A pickler refusing references a worker process could not import.

    Pickle stores classes and functions by reference, which a worker resolves by
    importing them; the script this process runs (``__main__``) is not importable
    there, so a reference into it is refused here rather than in the worker.
    """

    def reducer_override(self, obj: Any) -> Any:
        """Refuse a class or function of ``__main__``; pickle the rest as usual."""
        if (
            isinstance(obj, type | types.FunctionType)
            and getattr(obj, "__module__", None) == "__main__"
        ):
            raise pickle.PicklingError(
                f"{obj.__qualname__} is defined in the script being run (__main__), "
                "which a worker cannot import"
            )
        return NotImplemented


def _portable_pickle(program: Program) -> bytes:
    """The program pickled for a worker process (`_Portable`)."""
    buffer = io.BytesIO()
    _Portable(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(program)
    return buffer.getvalue()


def _failure(
    inp: Input,
    error: ErrorInfo,
    trace: str | None = None,
    results: Any = None,
) -> Output:
    """A failed output; partial results of the wrong type are dropped."""
    try:
        return ProgramOutput.failure(inp, error, results=results, traceback=trace)
    except ValidationError:
        return ProgramOutput.failure(inp, error, traceback=trace)


def _name(inp: Input) -> str:
    """The name of the program that runs an input."""
    return inp.program if isinstance(inp, ProgramInput) else FILES


def _not_found(inp: Input) -> ProgramError:
    name = inp.program if isinstance(inp, ProgramInput) else FILES
    hint = (
        " (a FileInput runs under qccodec's 'files' program: pip install "
        "qccompute[qccodec])"
        if isinstance(inp, FileInput) and not isinstance(inp, ProgramInput)
        else ""
    )
    return ProgramError.of(
        "environment",
        f"No program {name!r} is installed or registered{hint}. Register it in this "
        "process with qccompute.register_program(program), or install a package that "
        "lists it under the 'qccompute.programs' entry points.",
        code="not_installed",
    )


def _error(out: Output, cause: BaseException | None = None) -> ProgramError:
    """The exception for a failed output, with the output attached as ``output``."""
    if isinstance(cause, ProgramError) and cause.output is out:
        return cause
    error = out.error or ErrorInfo(
        kind=ErrorKind.internal, message=out.traceback or "failed"
    )
    exc = ProgramError(error, output=out)
    if cause is not None:
        exc.__cause__ = cause
    return exc


def _as_failure(
    inp: Input, exc: BaseException, program: str
) -> tuple[Output, ProgramError]:
    """A program's exception as its failed output and the ProgramError to raise.

    A `ProgramError` raised by a subprogram (through ``ctx``) that the program let
    escape fails it with kind ``program`` and the subprogram's error as ``cause``. Any
    other exception than a ProgramError is a bug: kind ``internal``.
    """
    trace = "".join(traceback.format_exception(exc))
    if isinstance(exc, ProgramError):
        child = getattr(exc, "_subprogram", None)
        if child is not None:
            error = ErrorInfo(
                kind=ErrorKind.program,
                message=f"subprogram {child} failed: {exc.error.message}",
                program=child,
                cause=exc.error,
            )
            out = _failure(inp, error, trace)
            wrapped = ProgramError(error, output=out)
            wrapped.__cause__ = exc
            return out, wrapped
        attached = exc.output
        out = (
            attached
            if isinstance(attached, ProgramOutput)
            else _failure(inp, exc.error, trace, exc.results)
        )
        exc.output = out
        return out, exc
    if isinstance(exc, ExceptionGroup):  # a list's or a session's group
        first = exc.exceptions[0]
        if isinstance(first, ProgramError):
            return _as_failure(inp, first, program)
    error = ErrorInfo(kind=ErrorKind.internal, message=f"{type(exc).__name__}: {exc}")
    out = _failure(inp, error, trace)
    wrapped = ProgramError(error, output=out)
    wrapped.__cause__ = exc
    return out, wrapped


def _threads(inp: Input, plan: Plan, cap: int) -> int:
    """The threads a calculation gets: what it asks for, at most ``cap``.

    It asks with its ``threads`` keyword, else through its plan
    (``Plan.resources.threads``: 1 unless the program says otherwise; None for all of
    ``cap``, the runner's threads or a subprogram's parent grant).
    """
    requested = inp.keywords.get("threads") if isinstance(inp, ProgramInput) else None
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        requested = plan.resources.threads
    if requested is None:
        return cap
    return max(1, min(requested, cap))


def _asked(
    inp: ProgramInput, key: str, share: float | None, limit: float | None
) -> float | None:
    """A memory keyword (GiB) in place of the share, at most ``limit``; else the share.

    Like ``threads``, a resource keyword is what the calculation asks for and the
    runner grants it, at most what it has. A value that is not a positive number is
    the program's to refuse; the share stands.
    """
    asked = inp.keywords.get(key)
    if isinstance(asked, bool) or not isinstance(asked, int | float):
        return share
    if not (0 < asked < math.inf):
        return share
    return float(asked) if limit is None else min(float(asked), limit)


def _refused(
    inp: Input, exc: BaseException, identities: Mapping[str, str]
) -> tuple[Output, ProgramError]:
    """The failed output of an input that failed before it ran (not found, refused)."""
    out, failure = _as_failure(inp, exc, inp.program)
    out = _with_execution(out, 0.0, None, identities)
    failure.output = out
    return out, failure


def _out_of_time(program: str) -> ProgramError:
    """The failure of a subprogram its parent started after its time ran out."""
    return ProgramError.of(
        "timeout",
        f"The calculation's time ran out before its subprogram {program} started.",
    )


def _subprogram_results(
    inputs: Input | Sequence[Input],
    results: Sequence[tuple[Output, ProgramError | None]],
    raise_exc: bool,
) -> Any:
    """Subprograms' outputs as ``RunContext.compute`` returns them.

    Each failure is marked with its subprogram, so a program that lets it escape fails
    with kind ``program`` (`_as_failure`). With ``raise_exc``, one input's failure
    raises, and a list's failures raise together as `qcdata.ProgramErrors`.

    Args:
        inputs: One input, or the list.
        results: Each input's output and failure, in order.
        raise_exc: Raise for failures.

    Returns:
        The output, or the outputs in input order.
    """
    one = isinstance(inputs, ProgramInput | FileInput)
    listed = [inputs] if one else list(inputs)  # type: ignore[list-item]
    for inp, (_, failure) in zip(listed, results, strict=True):
        if failure is not None:
            failure._subprogram = inp.program  # type: ignore[attr-defined]
    outputs = [out for out, _ in results]
    failures = [failure for _, failure in results if failure is not None]
    if failures and raise_exc:
        if one:
            raise failures[0]
        raise ProgramErrors(
            f"{len(failures)} of {len(outputs)} subprogram calculations failed",
            failures,
            outputs,
        )
    return outputs[0] if one else outputs


def _resolved(out: Output, failure: ProgramError | None) -> Future[Output]:
    """A future already done: the output, or its failure."""
    future: Future[Output] = Future()
    if failure is not None:
        future.set_exception(failure)
    else:
        future.set_result(out)
    return future


def _put(
    store: Store,
    key: str,
    out: Output,
    tags: Sequence[str],
    meta: Mapping[str, Any] | None,
) -> None:
    """Store a run, unless it was warm-started (its numbers depend on history).

    The output then names its run, ``execution.store_run = "<key>/<run>"``, as an
    output read from the store does, so a fresh result can be cited exactly too. The
    output's `ExecutionInfo` is the runner's own copy (`_with_execution`).
    """
    if out.record.get("warm_start") is True:
        return
    run = store.put(out, tags=tuple(tags), meta=meta)
    out.execution.__dict__["store_run"] = f"{key}/{run}"


def _with_execution(
    out: Output,
    wall_time: float,
    threads: int | None,
    identities: Mapping[str, str],
    devices: Sequence[str] = (),
    batch: bool = False,
) -> Output:
    """The output with what the runner measured, where it ran and the code that made it.

    ``wall_time`` is the call's. For an output of a batch it is the whole batch's,
    since its inputs ran together: it goes to ``timings["batch"]``, and to
    ``wall_time`` only when the program left that unset (a lockstep batch knows when
    each of its tasks finished). ``threads`` stays the program's when it set it; else
    it is the grant, and for an output of a batch its own part of it
    (`_Task.threads_of`), so ``wall_time * threads`` adds up across a batch's outputs
    to the batch's cost. The environment (host, CPU model, libc, and ``devices``: the
    granted GPUs' names) fills each field the program left unset. The program's own
    timings stay. ``identities`` are those of the programs that ran
    (`Runner.identities`). The output is the caller's from here on, so it is updated
    in place; its `ExecutionInfo` is copied (a program may share one between outputs)
    without validating it again.
    """
    execution = out.execution
    host = _host()
    if execution.model_fields_set.isdisjoint(host):  # the usual case: none set
        update = {**host, "identities": dict(identities)}
    else:
        update = {k: v for k, v in host.items() if getattr(execution, k) is None}
        update["identities"] = dict(identities)
    if devices and not execution.devices:
        update["devices"] = _device_names(devices)
    if not batch:
        update["wall_time"] = round(wall_time, 6)
    else:
        update["timings"] = {**execution.timings, "batch": round(wall_time, 6)}
        if execution.wall_time is None:
            update["wall_time"] = round(wall_time, 6)
    if execution.threads is None:
        update["threads"] = threads
    out.__dict__["execution"] = execution.model_copy(update=update)
    return out


# ------------------------------------------------------------------------ resources


class _Grant:
    """A budget of threads and GPUs, handed to the tasks of a list in input order.

    A task's share is the larger of its share of the threads and of the GPUs (with k
    calculations per GPU, each counts 1/k of a GPU); tasks are admitted while the
    running shares sum to at most 1, and host memory follows the share. Granting in
    order keeps a task that needs much from waiting forever behind a stream of small
    ones. A task asking for more than the budget gets all of it.
    """

    def __init__(self, threads: int, gpus: Sequence[str], per_gpu: int) -> None:
        self.threads = threads
        self._slots = {g: per_gpu for g in gpus}
        self._per_gpu = per_gpu
        self._used = 0.0
        self._turn = 0
        self._skipped: set[int] = set()
        self._ready = threading.Condition()

    def add_gpus(self, gpus: Sequence[str]) -> None:
        """Give a budget made without GPUs these, when a task first asks for one."""
        with self._ready:
            if not self._slots:
                self._slots = {g: self._per_gpu for g in gpus}

    def share(self, threads: int, gpus: int) -> float:
        """The share of the budget a task with these threads and GPUs takes."""
        gpu_share = gpus / self._per_gpu / len(self._slots) if self._slots else 0.0
        return max(threads / self.threads, gpu_share)

    def acquire(
        self, turn: int, threads: int, gpus: int
    ) -> tuple[int, tuple[str, ...], float]:
        """Wait for a task's turn and room; its threads, GPUs and share."""
        threads = max(1, min(threads, self.threads))
        gpus = min(gpus, len(self._slots))
        share = self.share(threads, gpus)
        with self._ready:
            self._ready.wait_for(
                lambda: (
                    self._turn == turn
                    and self._used + share <= 1 + 1e-9
                    and sum(1 for n in self._slots.values() if n > 0) >= gpus
                )
            )
            self._next()
            self._used += share
            devices = tuple(g for g, n in self._slots.items() if n > 0)[:gpus]
            for g in devices:
                self._slots[g] -= 1
            self._ready.notify_all()
        return threads, devices, share

    def _next(self) -> None:
        """Pass the turn on, past turns given up (call with the condition held)."""
        self._turn += 1
        while self._turn in self._skipped:
            self._skipped.discard(self._turn)
            self._turn += 1

    def skip(self, turn: int) -> None:
        """Give up a turn that will never be taken (a future cancelled before it ran)."""
        with self._ready:
            if turn == self._turn:
                self._next()
            elif turn > self._turn:
                self._skipped.add(turn)
            self._ready.notify_all()

    def release(self, devices: tuple[str, ...], share: float) -> None:
        """Return a finished task's room."""
        with self._ready:
            self._used -= share
            for g in devices:
                self._slots[g] += 1
            self._ready.notify_all()


class _Task:
    """One unit of a list: an input, or a batch of one program's inputs.

    A batch asks for what its inputs ask for together: the sum of their threads, at
    most the list's (the batch program divides them among its inputs), and the most
    GPUs any of them asks for (a batch runs its inputs together on its devices).
    """

    __slots__ = ("indices", "inputs", "entry", "threads", "asked", "gpus", "plan")

    def __init__(
        self, index: int, inp: Input, entry: _Entry, plan: Plan, threads: int
    ) -> None:
        self.indices = [index]
        self.inputs = [inp]
        self.entry = entry
        self.threads = threads
        self.asked = [threads]
        self.gpus = plan.resources.gpus
        self.plan = plan

    def add(self, index: int, inp: Input, plan: Plan, threads: int, cap: int) -> None:
        """Add an input to a batch, with the threads it asks for (at most ``cap``)."""
        self.indices.append(index)
        self.inputs.append(inp)
        self.asked.append(threads)
        self.threads = min(cap, self.threads + threads)
        self.gpus = max(self.gpus, plan.resources.gpus)

    def threads_of(self, granted: int) -> list[int]:
        """Each input's part of the batch's ``granted`` threads, for its output.

        Its own term in the sum the batch asked for, scaled down when the batch got
        less: ``max(1, asked * granted // total asked)``. They add up to the grant,
        to rounding.
        """
        total = sum(self.asked)
        return [max(1, asked * granted // total) for asked in self.asked]


# ---------------------------------------------------------------------------- runner


class Runner:
    """Runs programs on a share of this machine: one input, a list, or a session.

    Args:
        threads: CPU threads the runner may use at once, and the most one
            calculation gets; None: every core this process may use (its affinity and
            cgroup quota).
        gpus: GPUs it may use: ``"auto"`` for every visible GPU (``CUDA_VISIBLE_DEVICES``
            or ``nvidia-smi``, see `visible_gpus`), or a list of device identifiers as
            for ``CUDA_VISIBLE_DEVICES``; ``()`` for none. With ``"auto"`` and
            ``CUDA_VISIBLE_DEVICES`` unset, the runner sets
            ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (unless set), so CUDA numbers the GPUs as
            ``nvidia-smi`` does.
        backend: Where calculations run. Only ``"local"`` (this machine) for now.
        isolation: ``"auto"`` runs programs in this process unless they cannot share
            it (``in_process=False``), which run in worker processes; ``"process"``
            runs every program in a worker process, so a crash ends only the worker.
        per_gpu: Calculations that may share one GPU at once; each gets 1/per_gpu of
            its memory. A program in worker processes keeps this many workers per
            GPU, which its calculations and sessions there share.

    A runner never changes a number: it hands each calculation its share of the
    machine (`qccompute.program.RunContext`) and the program decides. Close a runner
    (or use it as a context manager) to stop its worker processes.
    """

    def __init__(
        self,
        threads: int | None = None,
        gpus: Literal["auto"] | Sequence[str] = "auto",
        backend: Literal["local"] = "local",
        isolation: Literal["auto", "process"] = "auto",
        *,
        per_gpu: int = 1,
    ) -> None:
        """Create a runner."""
        if backend != "local":
            raise ValueError(f"Unknown backend {backend!r}; there is only 'local'.")
        if isolation not in ("auto", "process"):
            raise ValueError(f"isolation is 'auto' or 'process', not {isolation!r}.")
        if threads is not None and (isinstance(threads, bool) or threads < 1):
            raise ValueError("threads must be a positive integer.")
        if per_gpu < 1:
            raise ValueError("per_gpu must be a positive integer.")
        self.threads: int = threads or usable_cores()
        self.backend = backend
        self.isolation = isolation
        self.per_gpu = per_gpu
        self._gpus_arg = gpus
        self._gpus: dict[str, float | None] | None = (
            None if gpus == "auto" else {str(g): None for g in gpus}
        )
        if gpus == "auto":  # before any program starts CUDA (its plan may)
            _pci_bus_order()
        self._memory: float | None | Literal[False] = False
        self._entries: dict[int, _Entry] = {}
        self._entries_lock = threading.Lock()
        # Workers by pool (program and GPUs), each with its users: calls in flight
        # and open sessions; workers being started, by pool.
        self._pools: dict[tuple[str, tuple[str, ...]], dict[Worker, int]] = {}
        self._starting: dict[tuple[str, tuple[str, ...]], int] = {}
        self._pools_changed = threading.Condition()
        self._all_workers: weakref.WeakSet[Worker] = weakref.WeakSet()
        self._closed = False
        # Submitted calculations: their threads, the runner's budget, the next turn.
        self._submitting = threading.Lock()
        self._futures: ThreadPoolExecutor | None = None
        self._budget: _Grant | None = None
        self._turns = 0

    def __repr__(self) -> str:
        """Show the runner's share of the machine."""
        return (
            f"Runner(threads={self.threads}, gpus={self._gpus_arg!r}, "
            f"isolation={self.isolation!r}, per_gpu={self.per_gpu})"
        )

    # ------------------------------------------------------------------ the machine

    @property
    def gpus(self) -> tuple[str, ...]:
        """The GPUs this runner uses (found on first use for ``"auto"``)."""
        if self._gpus is None:
            self._gpus = visible_gpus()
        return tuple(self._gpus)

    def _gpu_memory(self, devices: Sequence[str]) -> float | None:
        """The memory of the smallest of these GPUs (GiB), if known."""
        if not devices or self._gpus is None:
            return None
        sizes = [self._gpus.get(d) for d in devices]
        if any(s is None for s in sizes):
            return None
        return min(s for s in sizes if s is not None)

    def _host_memory(self) -> float | None:
        """The host memory of this process (GiB), read on first use."""
        if self._memory is False:
            self._memory = memory_gib()
        return self._memory

    # ------------------------------------------------------------------ programs

    def _entry(self, inp: Input) -> _Entry | None:
        """The program of an input, with what the runner needs to place it."""
        name = inp.program if isinstance(inp, ProgramInput) else FILES
        program = find_program(name)
        if program is None:
            return None
        return self._entry_of(program)

    def _plan(self, inp: Input) -> tuple[_Entry, Plan]:
        """The program of an input and its plan.

        Raises:
            ProgramError: The program is not installed, or refuses the input.
        """
        entry = self._entry(inp)
        if entry is None:
            raise _not_found(inp)
        return entry, plan_of(entry.program, inp)  # type: ignore[arg-type]

    def _entry_of(self, program: Program) -> _Entry:
        entry = self._entries.get(id(program))
        if entry is not None and entry.program is program:
            return entry
        with self._entries_lock:
            entry = _Entry(program, capabilities_of(program))
            self._entries[id(program)] = entry
        return entry

    def _in_worker(self, entry: _Entry) -> bool:
        return self.isolation == "process" or not entry.caps.in_process

    def _devices(self, plan: Plan, devices: Sequence[str] | None) -> tuple[str, ...]:
        """The GPUs of a calculation: as many as its plan asks for, of ``devices``.

        ``devices`` are those it may use (a parent's grant); None: the runner's. The
        plan decides: a program whose subprograms use GPUs plans theirs.
        """
        wanted = plan.resources.gpus
        if not wanted:
            return ()
        return (tuple(devices) if devices is not None else self.gpus)[:wanted]

    def _context(
        self,
        threads: int,
        devices: tuple[str, ...],
        memory: float | None,
        timeout: float | None,
        plan: Plan | None,
        inp: Input | None = None,
        parent: RunContext | None = None,
        batch: Sequence[Input] | None = None,
        progress: ProgressCallback | None = None,
    ) -> RunContext:
        """A calculation's context: its grant, its memory, the programs it may run.

        Host memory is ``memory``: the runner's (or a parent's grant) for a
        calculation alone, its share of it in a list. Device memory is 1/k of each
        GPU with k calculations per GPU, at most a parent's device grant. The input's
        ``memory_gib`` or ``device_memory_gib`` keyword replaces either, at most the
        runner's memory or the GPU's, or for a subprogram (``parent``) the parent's
        grant: a parent's grant holds its children. Memory is never scheduled.

        Its subprograms may be the programs its input names (``inp``, or for a batch
        each of ``batch``), as `RunContext.compute` checks. Its progress events go to
        the caller's ``progress`` with the program's name as their path, or for a
        subprogram to its parent's, the path one name longer.
        """
        host_limit, device_limit = self._limits(parent)
        device: float | None = None
        if devices:
            gpu = self._gpu_memory(devices)
            device = None if gpu is None else gpu / self.per_gpu
            if device_limit is not None:
                device = device_limit if device is None else min(device, device_limit)
                gpu = device_limit if gpu is None else min(gpu, device_limit)
        if isinstance(inp, ProgramInput):
            memory = _asked(inp, "memory_gib", memory, host_limit)
            if devices:
                device = _asked(inp, "device_memory_gib", device, gpu)
        ctx = RunContext(
            threads=threads,
            gpus=devices,
            memory_gib=None if memory is None else round(memory, 3),
            device_memory_gib=None if device is None else round(device, 3),
            timeout=timeout,
            plan=plan,
            runner=self,
        )
        ctx._calling = inp if inp is not None else batch
        if parent is not None:
            up = parent.progress
            if up is not None:
                name = _name(inp if inp is not None else batch[0])  # type: ignore[index]
                ctx.progress = up.child(name) if isinstance(up, _Progress) else up
        elif progress is not None:
            name = _name(inp if inp is not None else batch[0])  # type: ignore[index]
            ctx.progress = _Progress(progress, (name,))
        return ctx

    def _limits(self, parent: RunContext | None) -> tuple[float | None, float | None]:
        """What a calculation's memory may reach: host (GiB) and per GPU (None: all).

        A subprogram's are its parent's grant (made by a runner: it has threads); a
        calculation of the caller's own, the runner's memory and whole GPUs.
        """
        if parent is not None and parent.threads is not None:
            return parent.memory_gib, parent.device_memory_gib
        return self._host_memory(), None

    # ------------------------------------------------------------------ workers

    def _worker(self, entry: _Entry, devices: tuple[str, ...], threads: int) -> Worker:
        """A worker for this program and grant: idle, newly started, else shared.

        Workers are pooled by program and GPUs, whatever the threads of the calls
        they serve: a program's state (a model loaded, a GPU context) is costly to
        repeat, so calls of different threads (a list's batches of different sizes)
        share its workers. Each call's threads reach the program in its ``ctx``; a
        worker's numerical libraries start with the threads of the call that started
        it. A pool holds ``per_gpu`` workers on GPUs; without GPUs, as many as fit
        side by side in the runner's threads at this call's threads.

        An idle worker of the pool is used first; else a new one starts while the
        pool has room; else the calculation shares the worker with the fewest users,
        taking turns with them. Release it with `_release`.
        """
        from .worker import Worker

        key = (entry.spec, devices)
        room = self.per_gpu if devices else max(1, self.threads // threads)
        dead: list[Worker] = []
        worker: Worker | None
        with self._pools_changed:
            while True:
                if self._closed:
                    raise RuntimeError("This runner is closed.")
                pool = self._pools.setdefault(key, {})
                dead += [w for w in pool if not w.alive]
                for w in dead:
                    pool.pop(w, None)
                starting = self._starting.get(key, 0)
                if pool:
                    worker = min(pool, key=pool.__getitem__)
                    if pool[worker] == 0 or len(pool) + starting >= room:
                        pool[worker] += 1
                        break
                if len(pool) + starting < room:
                    self._starting[key] = starting + 1
                    worker = None
                    break
                self._pools_changed.wait()  # every worker of the pool is starting
        for w in dead:
            w.close()
        if worker is not None:
            return worker
        try:
            worker = Worker(entry.spec, threads=threads, gpus=devices)
        finally:
            with self._pools_changed:
                self._starting[key] -= 1
                self._pools_changed.notify_all()
        with self._pools_changed:
            self._all_workers.add(worker)
            if not self._closed:
                self._pools.setdefault(key, {})[worker] = 1
                return worker
        worker.close()
        raise RuntimeError("This runner is closed.")

    def _release(self, worker: Worker, devices: tuple[str, ...]) -> None:
        """A call or session is done with its worker; a dead worker leaves its pool."""
        key = (worker.spec, devices)
        with self._pools_changed:
            pool = self._pools.get(key, {})
            if worker in pool:
                pool[worker] -= 1
                if worker.alive:
                    return
                del pool[worker]
            self._pools_changed.notify_all()
        worker.close()

    def close(self) -> None:
        """Stop this runner's worker processes; it cannot run in workers afterwards.

        Submitted calculations that have not started are cancelled.
        """
        with self._submitting:
            futures, self._futures = self._futures, None
        if futures is not None:
            futures.shutdown(wait=False, cancel_futures=True)
        with self._pools_changed:
            self._closed = True
            self._pools.clear()
            self._pools_changed.notify_all()
        for worker in list(self._all_workers):
            worker.close()

    def __enter__(self) -> Runner:
        """Use the runner as a context manager that closes its workers."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the runner."""
        self.close()

    # ------------------------------------------------------------------ one input

    def _execute(
        self,
        inp: Input,
        threads: int,
        devices: tuple[str, ...],
        memory: float | None,
        timeout: float | None,
        plan: Plan | None = None,
        entry: _Entry | None = None,
        parent: RunContext | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Output, ProgramError | None]:
        """Run one input with its grant (``devices``: its GPUs) to its output.

        ``parent`` is the grant of the calculation whose subprogram it is; else
        ``progress`` is the caller's progress callable, if any.
        Returns the output, failed or not, and the error of a failure.
        """
        identities = self.identities(inp)
        start = time.perf_counter()
        if entry is None:
            entry = self._entry(inp)
            if entry is None:
                exc = _not_found(inp)
                out = _failure(inp, exc.error)
                exc.output = out
                return out, exc
        ctx = self._context(
            threads, devices, memory, timeout, plan, inp, parent, None, progress
        )
        worker = None
        try:
            if self._in_worker(entry):
                worker = self._worker(entry, devices, threads)
                out = worker.compute(inp, ctx)
            elif entry.lock is not None:
                with entry.lock:
                    out = entry.program.compute(inp, ctx)  # type: ignore[arg-type]
            else:
                out = entry.program.compute(inp, ctx)  # type: ignore[arg-type]
            failure = None
            if not out.success:  # a program that returned its failure
                failure = _error(out)
        except Exception as e:  # noqa: BLE001 - every failure becomes an output
            out, failure = _as_failure(inp, e, entry.program.name)
        finally:
            ctx.cleanup()
            if worker is not None:
                self._release(worker, devices)
        out = _with_execution(
            out, time.perf_counter() - start, threads, identities, devices
        )
        if failure is not None:
            failure.output = out
        return out, failure

    def _run(
        self,
        inp: Input,
        cap: int,
        devices: Sequence[str] | None,
        memory: float | None,
        timeout: float | None,
        parent: RunContext | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Output, ProgramError | None]:
        """Plan and run one input with the threads it asks for, at most ``cap``.

        It gets the GPUs its plan asks for, of ``devices`` (None: the runner's), and
        all of ``memory``; ``parent`` is the grant of the calculation whose subprogram
        it is, else ``progress`` the caller's progress callable.
        """
        try:
            entry, plan = self._plan(inp)
        except Exception as e:  # noqa: BLE001 - refused before running
            return _refused(inp, e, self.identities(inp))
        threads = _threads(inp, plan, cap)
        granted = self._devices(plan, devices)
        return self._execute(
            inp, threads, granted, memory, timeout, plan, entry, parent, progress
        )

    def identities(self, inp: Input) -> dict[str, str]:
        """The numerics identity of the input's program and of every subprogram.

        ``{name: numerics}``, by name, recursively through the subprograms' own
        subprograms, from the programs this runner would run now, each asked with its
        own spec (`qccompute.program.numerics_of`). A program named with several specs
        gets its distinct identities sorted and joined by ``";"``
        (``"gfn2:5;gxtb-2.0.1:7"``), so the map stays ``{program: string}``. A program
        that is not installed is left out.

        Args:
            inp: The input.

        Returns:
            The identities the runner sets on the input's output
            (``execution.identities``), and a store files its run under.
        """
        if not isinstance(inp, ProgramInput):
            entry = self._entry(inp)
            return {} if entry is None else {FILES: numerics_of(entry.program, inp)}
        found: dict[str, set[str]] = {}
        pending: list[ProgramSpec] = [inp]
        while pending:
            spec = pending.pop()
            program = find_program(spec.program)
            if program is not None:
                found.setdefault(spec.program, set()).add(numerics_of(program, spec))
            pending.extend(spec.subprograms)
        return {name: ";".join(sorted(found[name])) for name in sorted(found)}

    def _lookup(
        self, store: Store, inp: Input, rerun: bool
    ) -> tuple[str, Output | None]:
        """The key of an input, and its stored output if reusable.

        A stored run is reused only when it was made with the identities of the
        programs this runner would run now: a success always, a failure only when the
        input determines it (kind ``input``, ``unsupported`` or ``convergence``, or
        ``program`` with such a cause).
        """
        key = store.key(inp)
        if rerun:
            return key, None
        hit = store.get(key, self.identities(inp))
        if hit is None or hit.success or _deterministic(hit.error):
            return key, hit
        return key, None

    def _one(
        self,
        inp: Input,
        store: Store | None,
        timeout: float | None,
        rerun: bool,
        tags: Sequence[str],
        meta: Mapping[str, Any] | None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Output, ProgramError | None]:
        if store is not None:
            key, hit = self._lookup(store, inp, rerun)
            if hit is not None:
                if tags:
                    store.tag(key, *tags)
                return hit, None if hit.success else _error(hit)
        out, failure = self._run(
            inp, self.threads, None, self._host_memory(), timeout, None, progress
        )
        if store is not None:
            _put(store, key, out, tags, meta)
        return out, failure

    # ------------------------------------------------------------------ public API

    @overload
    def compute(
        self,
        inputs: Input,
        *,
        store: Store | None = None,
        timeout: float | None = None,
        rerun: bool = False,
        tags: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
        raise_exc: bool = True,
        progress: ProgressCallback | None = None,
    ) -> Output: ...

    @overload
    def compute(
        self,
        inputs: Sequence[Input],
        *,
        store: Store | None = None,
        timeout: float | None = None,
        rerun: bool = False,
        tags: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
        raise_exc: bool = True,
        progress: ProgressCallback | None = None,
    ) -> list[Output]: ...

    def compute(
        self,
        inputs: Input | Sequence[Input],
        *,
        store: Store | None = None,
        timeout: float | None = None,
        rerun: bool = False,
        tags: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
        raise_exc: bool = True,
        progress: ProgressCallback | None = None,
    ) -> Output | list[Output]:
        """Run one input, or a list of independent inputs side by side.

        Every input is planned first: an input its program refuses fails at once,
        without running. A calculation gets the threads it asks for: its ``threads``
        keyword, else its plan's (``Plan.resources.threads``: 1 unless the program
        says otherwise, None for all of them), at most the runner's ``threads``. In a
        list, a calculation that asks for all of them runs alone. A list's calculations are
        packed into the runner's threads and GPUs by what they ask for; inputs of a
        program with batches go to it together, with the sum of their threads (at most
        the runner's), which the program divides among them.

        Args:
            inputs: One input, or a list of them (of any programs).
            store: Look each input up first and return a reusable stored output; store
                every output that ran, with the identities of the programs that made
                it (`identities`). A stored run is reused only when its identities
                are those of the programs the runner would run now: a success always,
                a failure only when the input determines it (kind ``input``,
                ``unsupported`` or ``convergence``, or ``program`` with such a cause);
                ``rerun=True`` runs past it.
            timeout: Seconds each calculation may run; past it, the calculation
                fails with kind ``timeout``. Programs that start processes enforce it,
                and the runner kills programs in worker processes. In this process it
                is the program's to honour (``ctx.timeout``): the runner cannot stop a
                thread, and only starts none of its subprograms after it. For a hard
                limit on any program, use ``isolation="process"``.
            rerun: Run even when the store has a reusable output.
            tags: Tags for every key computed or reused (needs ``store``).
            meta: Metadata stored with each run (needs ``store``).
            raise_exc: Raise for failures (after every input of a list ran); False
                returns failed outputs in place.
            progress: Called with the events programs emit while they run (qcopt:
                one per step): small JSON objects, each with ``"path"``, the program
                names from the input down to the program that emitted it (a
                subprogram's events come through its parent). It may be called from
                several threads at once, and must be quick; an exception it raises
                is a warning. Stored outputs emit nothing; nothing is stored. A
                list's events do not say which input they are from: `submit` each
                input with a callable of its own for that.

        Returns:
            The output, or the outputs in input order.

        Raises:
            ProgramError: The calculation failed; its ``output`` is the failed
                output, with partial results and logs.
            ProgramErrors: Some calculations of a list failed; ``results`` holds
                every output in input order.
        """
        if (tags or meta) and store is None:
            raise ValueError("tags and meta are stored with runs: pass a store.")
        if timeout is not None and not timeout > 0:
            raise ValueError("timeout must be a positive number of seconds.")
        if isinstance(inputs, ProgramInput | FileInput):
            out, failure = self._one(
                inputs, store, timeout, rerun, tags, meta, progress
            )
            if failure is not None and raise_exc:
                raise failure
            return out
        inputs = list(inputs)
        outputs: list[Output | None] = [None] * len(inputs)
        failures: dict[int, ProgramError] = {}
        grant = (self.threads, None, self._host_memory(), timeout)
        listed = self._run_list(inputs, store, rerun, tags, meta, grant, None, progress)
        for i, out, failure in listed:
            outputs[i] = out
            if failure is not None:
                failures[i] = failure
        done = [o for o in outputs if o is not None]
        if failures and raise_exc:
            raise ProgramErrors(
                f"{len(failures)} of {len(inputs)} calculations failed",
                [failures[i] for i in sorted(failures)],
                done,
            )
        return done

    def submit(
        self,
        inp: Input,
        *,
        store: Store | None = None,
        timeout: float | None = None,
        rerun: bool = False,
        tags: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
        progress: ProgressCallback | None = None,
    ) -> Future[Output]:
        """Start one calculation and return at once: a future of its output.

        For workflows whose next calculations depend on earlier ones (per reaction, a
        transition-state search, then the IRC from its saddle, then both ends
        minimized), many at once and with no stage barriers: submit each calculation
        when its inputs are known, and the next as each future finishes
        (``concurrent.futures.wait``, ``as_completed`` or ``add_done_callback``).

        A stored output, or an input its program refuses, resolves the future before
        this returns. The others are planned now and run in the runner's threads:
        every submitted calculation of the runner is packed into its threads and GPUs
        as a list's are, admitted in the order submitted. They share the machine
        with each other, not with `compute` calls, which take their own share.

        Args:
            inp: The input.
            store: See `compute`.
            timeout: See `compute`; counted from when the calculation starts.
            rerun: See `compute`.
            tags: See `compute`.
            meta: See `compute`.
            progress: See `compute`; the calculation's events, from the threads
                that run it.

        Returns:
            The future: its ``result()`` is the output, or raises
            `qcdata.ProgramError` with the failed output, as `compute` does.
            Cancelling it before it starts (``future.cancel()``) skips the
            calculation.
        """
        if (tags or meta) and store is None:
            raise ValueError("tags and meta are stored with runs: pass a store.")
        if timeout is not None and not timeout > 0:
            raise ValueError("timeout must be a positive number of seconds.")
        key = ""
        if store is not None:
            key, hit = self._lookup(store, inp, rerun)
            if hit is not None:
                if tags:
                    store.tag(key, *tags)
                return _resolved(hit, None if hit.success else _error(hit))
        try:
            entry, plan = self._plan(inp)
        except Exception as e:  # noqa: BLE001 - refused before running
            out, failure = _refused(inp, e, self.identities(inp))
            if store is not None:
                _put(store, key, out, tags, meta)
            return _resolved(out, failure)
        task = _Task(0, inp, entry, plan, _threads(inp, plan, self.threads))
        with self._submitting:
            if self._futures is None:
                self._futures = ThreadPoolExecutor(
                    self.threads, thread_name_prefix="qccompute-submit"
                )
            if self._budget is None:
                self._budget = _Grant(self.threads, (), self.per_gpu)
            budget, turn = self._budget, self._turns
            if task.gpus:
                budget.add_gpus(self.gpus)
            self._turns += 1
            future = self._futures.submit(
                self._submitted,
                budget,
                turn,
                task,
                timeout,
                progress,
                (store, key, tags, meta),
            )

        def cancelled(done: Future[Output]) -> None:  # its turn is never taken
            if done.cancelled():
                budget.skip(turn)

        future.add_done_callback(cancelled)
        return future

    def _submitted(
        self,
        budget: _Grant,
        turn: int,
        task: _Task,
        timeout: float | None,
        progress: ProgressCallback | None,
        stored: tuple[Store | None, str, Sequence[str], Mapping[str, Any] | None],
    ) -> Output:
        """Run a submitted calculation in its turn, within the runner's budget.

        ``stored`` is the store, the input's key, the tags and meta to store the run
        with (no store: None).
        """
        held, granted, share = budget.acquire(turn, task.threads, task.gpus)
        memory = self._host_memory()
        try:
            out, failure = self._execute(
                task.inputs[0],
                held,
                granted,
                None if memory is None else memory * share,
                timeout,
                task.plan,
                task.entry,
                None,
                progress,
            )
        finally:
            budget.release(granted, share)
        store, key, tags, meta = stored
        if store is not None:
            _put(store, key, out, tags, meta)
        if failure is not None:
            raise failure
        return out

    def as_completed(
        self,
        inputs: Iterable[Input],
        *,
        store: Store | None = None,
        timeout: float | None = None,
        rerun: bool = False,
        tags: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
        raise_exc: bool = True,
        progress: ProgressCallback | None = None,
    ) -> Iterator[Output]:
        """Run a list like `compute` and yield each output as it finishes.

        Stored outputs come first. Each output's ``input_data`` says which input it
        is. With ``raise_exc``, failures are held back and raised together as
        `qcdata.ProgramErrors` once everything ran (``results`` in the order they
        finished); otherwise they are yielded in place.

        Args:
            inputs: The inputs.
            store: See `compute`.
            timeout: See `compute`.
            rerun: See `compute`.
            tags: See `compute`.
            meta: See `compute`.
            raise_exc: See above.
            progress: See `compute`.

        Yields:
            Outputs in the order they finish.

        Raises:
            ProgramErrors: Some calculations failed, after the last output.
        """
        if (tags or meta) and store is None:
            raise ValueError("tags and meta are stored with runs: pass a store.")
        inputs = list(inputs)
        failures: list[ProgramError] = []
        finished: list[Output] = []
        grant = (self.threads, None, self._host_memory(), timeout)
        listed = self._run_list(inputs, store, rerun, tags, meta, grant, None, progress)
        for _, out, failure in listed:
            finished.append(out)
            if failure is not None and raise_exc:
                failures.append(failure)
                continue
            yield out
        if failures:
            raise ProgramErrors(
                f"{len(failures)} of {len(inputs)} calculations failed",
                failures,
                finished,
            )

    def session(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None = None,
        *,
        threads: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Session:
        """Open a session: one program and molecule at many geometries.

        A program with sessions of its own (``capabilities().sessions``) opens one,
        pinned to the input's settings and warm-started between geometries: in this
        process it is the program's own session object (``mychem.Session``), exactly
        what a direct user gets. Its grant is set when it opens: ``threads``, else the
        threads its input asks for (as in `compute`), at most the runner's; the
        program splits it across the lineages it runs at once. Programs in worker
        processes give a session over the worker's pipe, with the same methods;
        sessions of a program on a GPU share its worker there.

        For any other program the session runs each geometry through ``compute``,
        from scratch (``record["warm_start"]`` is false): it is a convenience, not a
        speed-up. Each request gets the threads its input asks for (``threads``, if
        given); a ``compute_many`` of n requests runs as a list does, side by side,
        with the sum of their threads, at most ``threads`` or else the runner's.

        A session gets the GPUs its plan asks for, and all of the runner's memory.

        Args:
            inp: The program, model, keywords and structure; its calctype and
                geometry are the defaults of a request.
            calctypes: The calctypes the session may run (default: everything the
                method supports; declaring narrows it).
            threads: The session's grant of threads (at most the runner's): 13 path
                images evaluated at once want 13. None: what its input asks for.
            progress: Called with the events the program emits while the session
                runs requests; see `compute`.

        Returns:
            The session; use it as a context manager, or close it.

        Raises:
            ProgramError: The program is not installed or rejects the input.
        """
        if threads is not None:
            if isinstance(threads, bool) or threads < 1:
                raise ValueError("threads must be a positive integer.")
            threads = min(threads, self.threads)
        return self._open_session(
            inp,
            calctypes,
            threads,
            self.threads,
            None,
            self._host_memory(),
            None,
            progress,
        )

    # ------------------------------------------------------------------ lists

    def _run_list(
        self,
        inputs: list[Input],
        store: Store | None,
        rerun: bool,
        tags: Sequence[str],
        meta: Mapping[str, Any] | None,
        grant: tuple[int, Sequence[str] | None, float | None, float | None],
        parent: RunContext | None = None,
        progress: ProgressCallback | None = None,
    ) -> Iterator[tuple[int, Output, ProgramError | None]]:
        """Yield (index, output, failure) for every input, as each finishes.

        ``grant`` is what the list may use: threads, GPUs (None: the runner's), host
        memory (GiB), which each calculation gets its share of, and the timeout of
        each calculation. ``parent`` is the grant of the calculation whose
        subprograms they are; else ``progress`` the caller's progress callable.
        """
        threads, devices, memory, timeout = grant
        tasks: list[_Task] = []
        batches: dict[int, _Task] = {}
        keys: dict[int, str] = {}
        for i, inp in enumerate(inputs):
            if store is not None:
                keys[i], hit = self._lookup(store, inp, rerun)
                if hit is not None:
                    if tags:
                        store.tag(keys[i], *tags)
                    yield i, hit, None if hit.success else _error(hit)
                    continue
            try:
                entry, plan = self._plan(inp)
            except Exception as e:  # noqa: BLE001 - refused before running
                out, failure = _refused(inp, e, self.identities(inp))
                if store is not None:
                    _put(store, keys[i], out, tags, meta)
                yield i, out, failure
                continue
            wanted = _threads(inp, plan, threads)
            if entry.caps.batches and isinstance(entry.program, BatchProgram):
                task = batches.get(id(entry))
                if task is not None:
                    task.add(i, inp, plan, wanted, threads)
                    continue
                task = batches[id(entry)] = _Task(i, inp, entry, plan, wanted)
            else:
                task = _Task(i, inp, entry, plan, wanted)
            tasks.append(task)
        if not tasks:
            return
        if devices is None:
            devices = self.gpus if any(t.gpus for t in tasks) else ()
        budget = _Grant(threads, devices, self.per_gpu)

        def run(
            turn: int, task: _Task
        ) -> list[tuple[int, Output, ProgramError | None]]:
            held, granted, share = budget.acquire(turn, task.threads, task.gpus)
            mine = None if memory is None else memory * share
            try:
                return list(
                    self._run_task(task, held, granted, mine, timeout, parent, progress)
                )
            finally:
                budget.release(granted, share)

        if len(tasks) == 1:  # nothing to run beside: no thread
            for item in run(0, tasks[0]):
                if store is not None:
                    _put(store, keys[item[0]], item[1], tags, meta)
                yield item
            return
        pool = ThreadPoolExecutor(max_workers=min(len(tasks), budget.threads))
        try:
            futures = [pool.submit(run, turn, task) for turn, task in enumerate(tasks)]
            for future in as_completed(futures):
                for item in future.result():
                    if store is not None:
                        _put(store, keys[item[0]], item[1], tags, meta)
                    yield item
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def _run_task(
        self,
        task: _Task,
        threads: int,
        devices: tuple[str, ...],
        memory: float | None,
        timeout: float | None,
        parent: RunContext | None = None,
        progress: ProgressCallback | None = None,
    ) -> Iterator[tuple[int, Output, ProgramError | None]]:
        """Run one task of a list with its grant."""
        if len(task.inputs) == 1:
            out, failure = self._execute(
                task.inputs[0],
                threads,
                devices,
                memory,
                timeout,
                task.plan,
                task.entry,
                parent,
                progress,
            )
            yield task.indices[0], out, failure
            return
        entry = task.entry
        ctx = self._context(
            threads, devices, memory, timeout, None, None, parent, task.inputs, progress
        )
        start = time.perf_counter()
        worker = None
        try:
            if self._in_worker(entry):
                worker = self._worker(entry, devices, threads)
                outs = worker.compute_batch(task.inputs, ctx)
            elif entry.lock is not None:
                with entry.lock:
                    outs = entry.program.compute_batch(task.inputs, ctx)  # type: ignore[attr-defined]
            else:
                outs = entry.program.compute_batch(task.inputs, ctx)  # type: ignore[attr-defined]
            if len(outs) != len(task.inputs):
                raise RuntimeError(
                    f"{entry.program.name}.compute_batch returned {len(outs)} "
                    f"outputs for {len(task.inputs)} inputs"
                )
        except Exception as e:  # noqa: BLE001 - every input gets the failure
            outs = [_as_failure(inp, e, entry.program.name)[0] for inp in task.inputs]
        finally:
            ctx.cleanup()
            if worker is not None:
                self._release(worker, devices)
        wall = time.perf_counter() - start  # the batch's: its inputs ran together
        parts = task.threads_of(threads)
        for i, inp, out, part in zip(
            task.indices, task.inputs, outs, parts, strict=True
        ):
            out = _with_execution(
                out, wall, part, self.identities(inp), devices, batch=True
            )
            yield i, out, None if out.success else _error(out)

    # ------------------------------------------------------------------ subprograms

    def _subcompute(self, inputs: Any, ctx: RunContext, raise_exc: bool) -> Any:
        """``RunContext.compute``: subprograms within a calculation's grant.

        Each gets the threads it asks for (`_threads`), at most the grant's, and the
        time the calculation has left; once it has none, they fail with kind
        ``timeout`` without running.
        """
        threads = ctx.threads or self.threads
        devices = ctx.gpus if ctx.threads is not None else None
        memory = ctx.memory_gib if ctx.threads is not None else self._host_memory()
        timeout = ctx.timeout
        one = isinstance(inputs, ProgramInput | FileInput)
        if timeout == 0:  # the calculation's time ran out: none starts
            listed = [inputs] if one else list(inputs)
            failed = [
                _refused(i, _out_of_time(i.program), self.identities(i)) for i in listed
            ]
            return _subprogram_results(inputs if one else listed, failed, raise_exc)
        if one:
            ran = [self._run(inputs, threads, devices, memory, timeout, ctx)]
            return _subprogram_results(inputs, ran, raise_exc)
        inputs = list(inputs)
        results: list[Any] = [None] * len(inputs)
        grant = (threads, devices, memory, timeout)
        for i, out, failure in self._run_list(
            inputs, None, False, (), None, grant, ctx
        ):
            results[i] = (out, failure)
        return _subprogram_results(inputs, results, raise_exc)

    def _subsession(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None,
        threads: int | None,
        ctx: RunContext,
    ) -> Session:
        """``RunContext.session``: a subprogram's session within a grant.

        ``threads`` is the parent's own split of its grant, and host memory follows
        it; None: the threads the subprogram asks for, at most the grant. Once the
        calculation's time has run out, the session neither opens nor computes (kind
        ``timeout``).
        """
        if ctx.timeout == 0:
            error = _out_of_time(inp.program)
            error._subprogram = inp.program  # type: ignore[attr-defined]
            raise error
        cap = ctx.threads or self.threads
        devices = ctx.gpus if ctx.threads is not None else None
        memory = ctx.memory_gib if ctx.threads is not None else self._host_memory()
        if threads is not None:
            threads = max(1, min(threads, cap))
            if memory is not None and ctx.threads:
                memory = memory * min(1.0, threads / ctx.threads)
        return self._open_session(inp, calctypes, threads, cap, devices, memory, ctx)

    def _open_session(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None,
        threads: int | None,
        cap: int,
        devices: Sequence[str] | None,
        memory: float | None,
        parent: RunContext | None,
        progress: ProgressCallback | None = None,
    ) -> Session:
        """Open a session with ``threads``, or (None) the threads its plan asks for.

        A session over compute caps its lists at ``threads`` when given, else at
        ``cap``. ``parent`` is the grant of the calculation whose subprogram it is (its
        time left is the session's), None for a session of the caller's own.
        """
        timeout = None if parent is None else parent.timeout
        entry, plan = self._plan(inp)
        lists = cap if threads is None else threads
        if threads is None:
            threads = _threads(inp, plan, cap)
        allowed = session_calctypes(calctypes) if calctypes else None
        granted = self._devices(plan, devices)
        program = entry.program
        opened: Session
        lock = entry.lock
        native = entry.caps.sessions and (
            self._in_worker(entry) or isinstance(program, SessionProgram)
        )
        if not native:  # each request is a calculation, placed as any other
            if allowed is None:
                method = inp.model.method if inp.model is not None else None
                supported = (entry.caps.methods or {}).get(
                    method or "", entry.caps.calctypes
                )
                allowed = session_calctypes(None, supported & session_calctypes())
            compute = partial(
                self._session_compute,
                (threads, lists, granted, memory, timeout),
                parent,
                progress,
            )
            over = ComputeSession(
                compute, inp, allowed, program_version=program.version()
            )
            # Each compute takes the program's lock itself.
            return over if parent is None else _Guarded(over, None, parent)
        ctx = self._context(
            threads, granted, memory, timeout, None, inp, parent, None, progress
        )
        if self._in_worker(entry):
            worker = self._worker(entry, granted, threads)
            release = partial(self._release, worker, granted)
            opened = worker.session(inp, allowed, ctx, on_close=release)
            lock = None  # the worker is this session's own process
        else:
            opened = program.open_session(inp, ctx, allowed)  # type: ignore[attr-defined]
        try:
            weakref.finalize(opened, ctx.cleanup)
        except TypeError:  # not weak-referenceable
            pass
        if lock is None and parent is None:
            return opened
        return _Guarded(opened, lock, parent)

    def _session_compute(
        self,
        grant: tuple[int, int, tuple[str, ...], float | None, float | None],
        parent: RunContext | None,
        progress: ProgressCallback | None,
        inputs: Any,
        raise_exc: bool = True,
    ) -> Any:
        """Compute for a session over compute, within the session's grant.

        ``grant`` is the threads of one request, the threads a list of them may use
        together, the GPUs, host memory (GiB) and the timeout. A list runs as
        `compute` runs one: each request gets the threads it asks for, packed into
        the list's (#54).
        """
        threads, lists, devices, memory, timeout = grant
        if isinstance(inputs, ProgramInput):
            out, failure = self._execute(
                inputs, threads, devices, memory, timeout, None, None, parent, progress
            )
            if failure is not None and raise_exc:
                raise failure
            return out
        outputs: list[Output | None] = [None] * len(inputs)
        listed = self._run_list(
            list(inputs),
            None,
            False,
            (),
            None,
            (lists, devices, memory, timeout),
            parent,
            progress,
        )
        for i, out, _ in listed:
            outputs[i] = out
        return outputs


class _Guarded:
    """A session that takes turns on its program's lock; a subprogram's, within its time.

    Sessions of a program that is not thread-safe may be called from several threads
    (ten TS searches side by side): they take turns on one lock per program. A
    subprogram's session (``parent``: the grant of the calculation it serves) marks its
    failures, so a program that lets one escape fails with kind ``program``, and
    computes nothing once the calculation's time has run out (kind ``timeout``).
    """

    def __init__(
        self,
        session: Session,
        lock: _Turns | None,
        parent: RunContext | None = None,
    ) -> None:
        self._session = session
        self._lock = lock if lock is not None else _NoLock()
        self._parent = parent
        self._subprogram = None if parent is None else session.program

    def _out_of_time(self) -> ProgramError | None:
        """The failure of a request made after the calculation's time ran out."""
        if self._parent is None or self._parent.timeout != 0:
            return None
        return _out_of_time(self._subprogram or self.program)

    @property
    def program(self) -> str:
        """The session's program."""
        return self._session.program

    @property
    def program_version(self) -> str:
        """The session's program version."""
        return self._session.program_version

    @property
    def input_data(self) -> ProgramInput:
        """The input the session was opened with."""
        return self._session.input_data

    def compute(self, *args: Any, **kwargs: Any) -> Frame:
        """See `qccompute.program.Session.compute`."""
        try:
            late = self._out_of_time()
            if late is not None:
                raise late
            with self._lock:
                return self._session.compute(*args, **kwargs)
        except ProgramError as e:
            if self._subprogram is not None:
                e._subprogram = self._subprogram  # type: ignore[attr-defined]
            raise

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """See `qccompute.program.Session.compute_many`."""
        try:
            late = self._out_of_time()
            if late is not None:  # every request fails, as a group
                return ProgramErrors.check(
                    [
                        failed_frame(self._structure(r), late, self.program)
                        for r in requests
                    ]
                )
            with self._lock:
                return self._session.compute_many(requests)
        except ExceptionGroup as group:
            if self._subprogram is not None:
                for e in group.exceptions:
                    setattr(e, "_subprogram", self._subprogram)  # noqa: B010
            raise

    def _structure(self, request: Request) -> Structure:
        """The structure a request is for."""
        structure = self._session.input_data.require_structure()
        if request.positions is None:
            return structure
        return structure.model_copy(update={"geometry": request.positions})

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

    def __enter__(self) -> _Guarded:
        """Use the session as a context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the session."""
        self.close()

    def __getattr__(self, name: str) -> Any:
        """The program's own extras on its session (``calctypes``, ``lineages``)."""
        return getattr(self._session, name)


class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


# ------------------------------------------------------------------ the default runner

_default: Runner | None = None
_default_lock = threading.Lock()


def default_runner() -> Runner:
    """The runner of ``qccompute.compute``: this machine, made on first use.

    It shares every core this process may use and every visible GPU (each calculation
    gets the threads it asks for: 1 unless its input or program says more), runs
    programs in this process unless they cannot share it, and closes its workers at
    exit. It cannot be reconfigured; make a `Runner` for anything else.
    """
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Runner()
                atexit.register(_default.close)
    return _default


@overload
def compute(
    inputs: Input,
    *,
    store: Store | None = None,
    timeout: float | None = None,
    rerun: bool = False,
    tags: Sequence[str] = (),
    meta: Mapping[str, Any] | None = None,
    raise_exc: bool = True,
    progress: ProgressCallback | None = None,
) -> Output: ...


@overload
def compute(
    inputs: Sequence[Input],
    *,
    store: Store | None = None,
    timeout: float | None = None,
    rerun: bool = False,
    tags: Sequence[str] = (),
    meta: Mapping[str, Any] | None = None,
    raise_exc: bool = True,
    progress: ProgressCallback | None = None,
) -> list[Output]: ...


def compute(
    inputs: Input | Sequence[Input],
    *,
    store: Store | None = None,
    timeout: float | None = None,
    rerun: bool = False,
    tags: Sequence[str] = (),
    meta: Mapping[str, Any] | None = None,
    raise_exc: bool = True,
    progress: ProgressCallback | None = None,
) -> Output | list[Output]:
    """Run one input or a list on this machine: `Runner.compute` of the default runner.

    Args:
        inputs: One input, or a list of them (of any programs).
        store: See `Runner.compute`.
        timeout: See `Runner.compute`.
        rerun: See `Runner.compute`.
        tags: See `Runner.compute`.
        meta: See `Runner.compute`.
        raise_exc: See `Runner.compute`.
        progress: See `Runner.compute`.

    Returns:
        The output, or the outputs in input order.

    Raises:
        ProgramError: The calculation failed; ``output`` is the failed output.
        ProgramErrors: Some calculations of a list failed.
    """
    return default_runner().compute(
        inputs,
        store=store,
        timeout=timeout,
        rerun=rerun,
        tags=tags,
        meta=meta,
        raise_exc=raise_exc,
        progress=progress,
    )


def submit(
    inp: Input,
    *,
    store: Store | None = None,
    timeout: float | None = None,
    rerun: bool = False,
    tags: Sequence[str] = (),
    meta: Mapping[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> Future[Output]:
    """Start one calculation on this machine: `Runner.submit` of the default runner.

    Args:
        inp: The input.
        store: See `Runner.compute`.
        timeout: See `Runner.compute`.
        rerun: See `Runner.compute`.
        tags: See `Runner.compute`.
        meta: See `Runner.compute`.
        progress: See `Runner.compute`.

    Returns:
        A future of the output; ``result()`` raises `qcdata.ProgramError` for a
        failed calculation.
    """
    return default_runner().submit(
        inp,
        store=store,
        timeout=timeout,
        rerun=rerun,
        tags=tags,
        meta=meta,
        progress=progress,
    )


def session(
    inp: ProgramInput,
    calctypes: Iterable[CalcType | str] | None = None,
    *,
    threads: int | None = None,
    progress: ProgressCallback | None = None,
) -> Session:
    """Open a session on this machine: `Runner.session` of the default runner.

    Args:
        inp: The program, model, keywords, structure, and the default calctype.
        calctypes: The calctypes the session may run.
        threads: See `Runner.session`.
        progress: See `Runner.session`.

    Returns:
        The session.
    """
    return default_runner().session(inp, calctypes, threads=threads, progress=progress)
