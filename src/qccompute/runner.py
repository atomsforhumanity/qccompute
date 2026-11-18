"""The Runner: one way to run programs, one input or many, in this process or in workers.

``qccompute.compute`` and ``qccompute.session`` are the methods of the default runner,
made on first use for this machine (all its usable cores and visible GPUs). A `Runner`
of your own takes a share of the machine and closes its worker processes when you are
done with it::

    with qccompute.Runner(threads=16) as runner:
        outputs = runner.compute(inputs, store=store, tags=["screen"])

Every input goes one path: find the program by name, plan it, place it (this process,
or a worker process for programs that cannot share one), call ``compute`` or
``compute_batch`` with a `qccompute.program.RunContext`, and record the execution. Each
calculation runs with one thread count: its ``threads`` keyword, else its caller's whole
count, and never more runs under a count than it holds (the Threads page of the docs). A
failed calculation raises `qcdata.ProgramError` with the failed output as its
``output``; a list runs everything first and raises `qcdata.ProgramErrors`.
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
from concurrent.futures import (
    Future,
    InvalidStateError,
    ThreadPoolExecutor,
    as_completed,
)
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

from ._count import Closed, Count, check_free, holding
from ._pools import hold_threads, own_threads
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
    _Control,
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
    "stop",
    "session",
    "default_runner",
]

#: Guards the lazy making of a running calculation's count (`_children`).
_children_lock = threading.Lock()

Input = ProgramInput | FileInput
Output = ProgramOutput[Any, Any]

# Kinds a stored failure is reused for: determined by the input, so the same code fails
# them the same way (kind program counts when its cause is one of these).
_DETERMINISTIC = frozenset(
    {ErrorKind.input, ErrorKind.unsupported, ErrorKind.convergence}
)


def _root(error: ErrorInfo | None) -> ErrorInfo | None:
    """A failure's own error: through kind ``program`` to the subprogram's cause."""
    while error is not None and error.kind is ErrorKind.program:
        error = error.cause
    return error


def _merged(identities: Iterable[Mapping[str, str]]) -> dict[str, str]:
    """Several inputs' identities as one: each program's distinct strings, joined."""
    found: dict[str, set[str]] = {}
    for one in identities:
        for name, numerics in one.items():
            found.setdefault(name, set()).update(numerics.split(";"))
    return {name: ";".join(sorted(found[name])) for name in sorted(found)}


def _deterministic(error: ErrorInfo | None) -> bool:
    """Whether a stored failure is determined by its input, and so reusable."""
    error = _root(error)
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


def default_threads() -> int:
    """A runner's threads when none are given: ``OMP_NUM_THREADS``, else `usable_cores`.

    A dispatcher that sets ``OMP_NUM_THREADS`` for a task (the job runner's
    ``threads_per_task``) sets the runner's count with it.
    """
    value = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
    if value.isdigit() and int(value) > 0:
        return int(value)
    return usable_cores()


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
    name = _name(inp)
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
    escape fails it with kind ``program`` and the subprogram's error as ``cause``, or
    kind ``stopped`` when the subprogram stopped on request (the request covers the
    whole calculation). Any other exception than a ProgramError is a bug: kind
    ``internal``.
    """
    trace = "".join(traceback.format_exception(exc))
    if isinstance(exc, ProgramError):
        child = getattr(exc, "_subprogram", None)
        if child is not None:
            stopped = exc.error.kind is ErrorKind.stopped
            error = ErrorInfo(
                kind=ErrorKind.stopped if stopped else ErrorKind.program,
                message=f"subprogram {child} "
                f"{'stopped' if stopped else 'failed'}: {exc.error.message}",
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


def _threads(inp: Input, count: int, default: int) -> int:
    """A calculation's thread count: its ``threads`` keyword, at most ``count``.

    Without the keyword it is ``default``: the caller's whole count, its part of it in
    a list, or 1 for a calculation that runs on GPUs (`_unasked`). A keyword above
    ``count`` is lowered to it, with a warning (once per message).
    """
    asked = inp.keywords.get("threads") if isinstance(inp, ProgramInput) else None
    if isinstance(asked, bool) or not isinstance(asked, int) or asked < 1:
        return default
    if asked > count:
        warnings.warn(
            f"A calculation asks for threads={asked} and runs on {count}, all its "
            "caller has. A top-level calculation's caller is the runner: "
            "Runner(threads=), else OMP_NUM_THREADS, else this process's cores; a "
            "subprogram's is its parent. To keep NumPy on one thread without capping "
            "calculations, set OPENBLAS_NUM_THREADS or MKL_NUM_THREADS instead of "
            "OMP_NUM_THREADS.",
            RuntimeWarning,
            stacklevel=2,
        )
        return count
    return asked


def _unasked(inp: object, gpus: int, whole: int) -> int:
    """A calculation's threads without a ``threads`` keyword: ``whole``, or 1 on GPUs.

    A calculation given GPUs runs on them and gets 1 thread, unless its input names
    subprograms: then its plan asks for their GPUs, its threads are theirs, and it gets
    ``whole``.
    """
    return 1 if gpus and not getattr(inp, "subprograms", None) else whole


def _children(ctx: RunContext) -> Count:
    """The count a running calculation's subprograms run under: its own threads.

    Made on its first subprogram call, with the calculation's GPUs and memory, which
    its subprograms take their shares of.
    """
    count: Count | None = ctx._count
    if count is None:
        with _children_lock:
            count = ctx._count
            if count is None:
                assert ctx.threads is not None
                count = ctx._count = Count(ctx.threads, ctx.gpus, ctx.memory_gib)
    return count


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


def _unnamed(
    inp: Input, refusal: ProgramError, out: Output
) -> tuple[Output, ProgramError]:
    """The failure of a calculation whose program called a program its input does not name.

    The refusal (kind ``input``, code ``unnamed_subprogram``) is the calculation's
    error, whatever the program made of it (an output, its own failure); the
    program's partial results stay where they fit.
    """
    trace = "".join(traceback.format_exception(refusal))
    failed = _failure(inp, refusal.error, trace, out.results)
    return failed, ProgramError(refusal.error, output=failed)


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
    """Store a run, unless it was warm-started (its numbers depend on history) or stopped.

    A stopped output is where the caller chose to end the run, not what the input
    gives: never stored (nor one that failed because a subprogram stopped). The output
    then names its run, ``execution.store_run = "<key>/<run>"``, as an output read from
    the store does, so a fresh result can be cited exactly too.
    """
    if out.record.get("warm_start") is True:
        return
    if not out.success and (root := _root(out.error)) is not None:
        if root.kind is ErrorKind.stopped:
            return
    run = store.put(out, tags=tuple(tags), meta=meta)
    cited = out.execution.model_copy(update={"store_run": f"{key}/{run}"})
    out.__dict__["execution"] = cited


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
    granted GPUs' names) fills each field the program left unset; a program's explicit
    ``devices=[]`` says it ran on no device (a CPU fallback) and stays. The program's
    own timings stay. ``identities`` are those of the programs that ran
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
    if devices and "devices" not in execution.model_fields_set:
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


class _Task:
    """One unit of a list: an input, or a batch of one program's inputs.

    A batch asks for what its inputs get together: the sum of their threads, at most
    the list's count (the batch program divides them among its inputs), and the most
    GPUs any of them gets (a batch runs its inputs together on its devices).
    """

    __slots__ = ("indices", "inputs", "entry", "threads", "asked", "gpus", "plan")

    def __init__(
        self, index: int, inp: Input, entry: _Entry, plan: Plan, threads: int, gpus: int
    ) -> None:
        self.indices = [index]
        self.inputs = [inp]
        self.entry = entry
        self.threads = threads
        self.asked = [threads]
        self.gpus = gpus
        self.plan = plan

    def add(self, index: int, inp: Input, threads: int, gpus: int, cap: int) -> None:
        """Add an input to a batch, with its threads and GPUs (at most ``cap`` in all)."""
        self.indices.append(index)
        self.inputs.append(inp)
        self.asked.append(threads)
        self.threads = min(cap, self.threads + threads)
        self.gpus = max(self.gpus, gpus)

    def threads_of(self, granted: int) -> list[int]:
        """Each input's part of the batch's ``granted`` threads, for its output.

        Its own term in the sum the batch asked for, scaled down when the batch got
        less: ``max(1, asked * granted // total asked)``. They add up to the grant,
        to rounding.
        """
        total = sum(self.asked)
        return [max(1, asked * granted // total) for asked in self.asked]


class _Future(Future[Output]):
    """A submitted calculation's future, whose `cancel` ends it even while it runs.

    It stays pending until the calculation's output is in, so ``cancel()`` succeeds
    while it runs (the standard library's refuses then), and `running` says whether
    it has started: taken its threads.
    """

    def __init__(self, control: _Control) -> None:
        super().__init__()
        self._control = control

    def running(self) -> bool:
        """Whether the calculation is running."""
        return self._control.started and not self.done()

    def cancel(self) -> bool:
        """Cancel the calculation, waiting or running; False once it is done.

        One that waits never runs. One that runs ends: a worker running it is killed
        with every process it started; in this process it ends at its next look at
        its time (``ctx.timeout`` is 0) and starts no more subprograms. Its output
        is discarded and never stored.
        """
        if not super().cancel():
            return False
        self._control.cancel()
        return True

    def _finish(self, out: Output | None, failure: BaseException | None) -> None:
        """Resolve with the output or the failure, unless cancelled meanwhile."""
        try:
            if failure is not None:
                self.set_exception(failure)
            else:
                self.set_result(out)  # type: ignore[arg-type]
        except InvalidStateError:  # cancelled as it finished
            pass


# ---------------------------------------------------------------------------- runner


class Runner:
    """Runs programs on a share of this machine: one input, a list, or a session.

    Args:
        threads: The runner's thread count: the most threads its calculations run at
            once, and what a calculation that sets no ``threads`` gets. None:
            ``OMP_NUM_THREADS`` if set (a dispatcher's choice), else every core this
            process may use (its affinity and cgroup quota).
        gpus: GPUs it may use: ``"auto"`` for every visible GPU (``CUDA_VISIBLE_DEVICES``
            or ``nvidia-smi``, see `visible_gpus`), or a list of device identifiers as
            for ``CUDA_VISIBLE_DEVICES``; ``()`` for none. With ``"auto"`` and
            ``CUDA_VISIBLE_DEVICES`` unset, the runner sets
            ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (unless set), so CUDA numbers the GPUs as
            ``nvidia-smi`` does. A calculation gets whole GPUs.
        isolation: ``"auto"`` runs programs in this process unless they cannot share
            it (``in_process=False``), which run in worker processes; ``"process"``
            runs every program in a worker process, so a crash ends only the worker.

    A runner never changes a number: it hands each calculation its thread count, GPUs
    and memory (`qccompute.program.RunContext`) and the program decides. It never runs
    more threads at once than its own count; work beyond it waits its turn (the docs'
    Threads page). Close a runner (or use it as a context manager) to stop its worker
    processes.
    """

    def __init__(
        self,
        threads: int | None = None,
        gpus: Literal["auto"] | Sequence[str] = "auto",
        isolation: Literal["auto", "process"] = "auto",
    ) -> None:
        """Create a runner."""
        if isolation not in ("auto", "process"):
            raise ValueError(f"isolation is 'auto' or 'process', not {isolation!r}.")
        if threads is not None and (isinstance(threads, bool) or threads < 1):
            raise ValueError("threads must be a positive integer.")
        self.threads: int = threads or default_threads()
        self.isolation = isolation
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
        # The count top-level work runs under; submitted calculations' threads and
        # futures not yet done.
        self._count = Count(self.threads, warn=True)
        self._submitting = threading.Lock()
        self._futures: ThreadPoolExecutor | None = None
        self._submitted: set[_Future] = set()

    def __repr__(self) -> str:
        """Show the runner's share of the machine."""
        return (
            f"Runner(threads={self.threads}, gpus={self._gpus_arg!r}, "
            f"isolation={self.isolation!r})"
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

    def _top(self, gpus: int = 0) -> Count:
        """The count top-level work runs under: the runner's.

        With the runner's memory, and its GPUs once a calculation asks for one.
        """
        count = self._count
        count.memory = self._host_memory()
        if gpus and not count.gpus:
            count.add_gpus(self.gpus)
        return count

    def _gpus_of(self, holder: Count, plan: Plan) -> int:
        """The GPUs a calculation gets: as many as its plan asks for, of the holder's."""
        wanted = plan.resources.gpus
        if wanted and holder is self._count:
            self._top(wanted)
        return min(wanted, holder.gpus)

    # ------------------------------------------------------------------ programs

    def _entry(self, inp: Input) -> _Entry | None:
        """The program of an input, with what the runner needs to place it."""
        program = find_program(_name(inp))
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
        limit: float | None,
        timeout: float | None,
        plan: Plan | None,
        inp: Input | None = None,
        parent: RunContext | None = None,
        batch: Sequence[Input] | None = None,
        progress: ProgressCallback | None = None,
        control: _Control | None = None,
    ) -> RunContext:
        """A calculation's context: its threads, its memory, the programs it may run.

        Host memory is ``memory``, its share of its holder's (``limit``: the runner's,
        a parent's, a session's). Device memory is each GPU's, at most a subprogram's
        parent's. The input's ``memory_gib`` or ``device_memory_gib`` keyword replaces
        either, at most ``limit`` or the GPU's. Memory is never scheduled, and never
        rounded: a keyword reaches the program as given, a share as computed.

        Its subprograms may be the programs its input names (``inp``, or for a batch
        each of ``batch``), as `RunContext.compute` checks. Its progress events go to
        the caller's ``progress`` with the program's name as their path, or for a
        subprogram to its parent's, the path one name longer. A submitted
        calculation's ``control`` (its cancel) is shared with its subprograms.
        """
        device: float | None = None
        if devices:
            device = self._gpu_memory(devices)
            below = parent.device_memory_gib if parent is not None else None
            if below is not None:
                device = below if device is None else min(device, below)
        if isinstance(inp, ProgramInput):
            memory = _asked(inp, "memory_gib", memory, limit)
            if devices:
                device = _asked(inp, "device_memory_gib", device, device)
        ctx = RunContext(
            threads=threads,
            gpus=devices,
            memory_gib=memory,
            device_memory_gib=device,
            timeout=timeout,
            plan=plan,
            runner=self,
        )
        ctx._calling = inp if inp is not None else batch
        ctx._control = parent._control if parent is not None else control
        if parent is not None:
            up = parent.progress
            if up is not None:
                name = _name(inp if inp is not None else batch[0])  # type: ignore[index]
                ctx.progress = up.child(name) if isinstance(up, _Progress) else up
        elif progress is not None:
            name = _name(inp if inp is not None else batch[0])  # type: ignore[index]
            ctx.progress = _Progress(progress, (name,))
        return ctx

    # ------------------------------------------------------------------ workers

    def _worker(self, entry: _Entry, devices: tuple[str, ...], threads: int) -> Worker:
        """A worker for this program and grant: idle, newly started, else shared.

        Workers are pooled by program and GPUs, whatever the threads of the calls
        they serve: a program's state (a model loaded, a GPU context) is costly to
        repeat, so calls of different threads (a list's batches of different sizes)
        share its workers. Each call's threads reach the program in its ``ctx``, and
        the worker holds its BLAS and OpenMP pools at them while the call runs. A
        pool holds one worker on GPUs; without GPUs, as many as fit side by side in
        the runner's threads at this call's threads.

        An idle worker of the pool is used first; else a new one starts while the
        pool has room; else the calculation shares the worker with the fewest users,
        taking turns with them. Release it with `_release`.
        """
        from .worker import Worker

        key = (entry.spec, devices)
        room = 1 if devices else max(1, self.threads // threads)
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

        Submitted calculations that have not started are cancelled, and so is work
        waiting for the runner's threads. A worker still running a calculation (from
        another thread) is killed with every process it started; that calculation
        fails with kind ``environment``.
        """
        with self._submitting:
            futures, self._futures = self._futures, None
            waiting = [f for f in self._submitted if not f._control.started]
        for future in waiting:
            future.cancel()
        self._count.close()
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
        gpus: int,
        holder: Count,
        timeout: float | None,
        plan: Plan | None = None,
        entry: _Entry | None = None,
        parent: RunContext | None = None,
        progress: ProgressCallback | None = None,
        control: _Control | None = None,
    ) -> tuple[Output, ProgramError | None]:
        """Run one input with ``threads`` and ``gpus`` GPUs taken from ``holder``.

        It waits its turn for them, then runs with its share of the holder's memory;
        its time (``timeout``; a subprogram's is its parent's time left) counts from
        when it starts. ``parent`` is the context of the calculation whose subprogram
        it is; else ``progress`` is the caller's progress callable, if any, and
        ``control`` a submitted calculation's.
        Returns the output, failed or not, and the error of a failure.
        """
        identities = self.identities(inp)
        if entry is None:
            entry = self._entry(inp)
            if entry is None:
                exc = _not_found(inp)
                out = _failure(inp, exc.error)
                exc.output = out
                return out, exc
        if parent is not None:
            control = parent._control
        try:
            devices = holder.take(threads, gpus, control)
        except ProgramError as e:  # cancelled or stopped while it waited
            return _refused(inp, e, identities)
        if control is not None:
            control.started = True
        try:
            if parent is not None:
                timeout = parent.timeout  # what is left of it once this starts
                if timeout == 0:
                    return _refused(inp, _out_of_time(_name(inp)), identities)
            share = holder.share(threads, devices)
            memory = None if holder.memory is None else holder.memory * share
            ctx = self._context(
                threads,
                devices,
                memory,
                holder.memory,
                timeout,
                plan,
                inp,
                parent,
                None,
                progress,
                control,
            )
            return self._call(inp, ctx, entry, identities)
        finally:
            holder.give(threads, devices)

    def _call(
        self, inp: Input, ctx: RunContext, entry: _Entry, identities: dict[str, str]
    ) -> tuple[Output, ProgramError | None]:
        """Call the program with its context: in this process, or in a worker."""
        threads, devices = ctx.threads, ctx.gpus
        assert threads is not None
        start = time.perf_counter()
        worker = None
        try:
            with holding():
                if self._in_worker(entry):
                    ctx._identities = identities
                    worker = self._worker(entry, devices, threads)
                    out = worker.compute(inp, ctx)
                elif entry.lock is not None:
                    with entry.lock, hold_threads(own_threads(threads, (inp,))):
                        out = entry.program.compute(inp, ctx)  # type: ignore[arg-type]
                else:
                    with hold_threads(own_threads(threads, (inp,))):
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
        if ctx._refusal is not None:  # it called a program its input does not name
            out, failure = _unnamed(inp, ctx._refusal, out)
        out = _with_execution(
            out, time.perf_counter() - start, threads, identities, devices
        )
        if failure is not None:
            failure.output = out
        return out, failure

    def _run(
        self,
        inp: Input,
        holder: Count,
        timeout: float | None,
        parent: RunContext | None = None,
        progress: ProgressCallback | None = None,
    ) -> tuple[Output, ProgramError | None]:
        """Plan and run one input under ``holder``.

        Its threads are its ``threads`` keyword, else the holder's whole count (1 when
        it is given GPUs); ``parent`` is the context of the calculation whose
        subprogram it is, else ``progress`` the caller's progress callable.
        """
        try:
            entry, plan = self._plan(inp)
        except Exception as e:  # noqa: BLE001 - refused before running
            return _refused(inp, e, self.identities(inp))
        gpus = self._gpus_of(holder, plan)
        threads = _threads(inp, holder.threads, _unasked(inp, gpus, holder.threads))
        return self._execute(
            inp, threads, gpus, holder, timeout, plan, entry, parent, progress
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
        out, failure = self._run(inp, self._top(), timeout, None, progress)
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
        without running. One calculation gets the runner's whole thread count unless
        it sets ``threads``; a list shares it, each input getting the count divided by
        the inputs that run (store hits don't count), and inputs of a program with
        batches go to it together in one call. Work beyond the runner's count waits
        its turn. The docs' Threads page explains the rules.

        Args:
            inputs: One input, or a list of them (of any programs).
            store: Look each input up first and return a reusable stored output; store
                every output that ran, with the identities of the programs that made
                it (`identities`). A stored run is reused only when its identities
                are those of the programs the runner would run now: a success always,
                a failure only when the input determines it (kind ``input``,
                ``unsupported`` or ``convergence``, or ``program`` with such a cause);
                ``rerun=True`` runs past it.
            timeout: Seconds each calculation may run, from when it starts; past it,
                the calculation fails with kind ``timeout``. Programs that start
                processes enforce it, and the runner kills programs in worker
                processes. In this process it is the program's to honour
                (``ctx.timeout``): the runner cannot stop a thread, and only starts
                none of its subprograms after it. For a hard limit on any program, use
                ``isolation="process"``.
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
            RuntimeError: Called from a thread that is running a calculation or a
                session request (a program uses ``ctx.compute``).
        """
        check_free()
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
        listed = self._run_list(
            inputs, store, rerun, tags, meta, self._top(), timeout, None, progress
        )
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
        minimized): submit each calculation when its inputs are known, and the next as
        each future finishes (``concurrent.futures.wait``, ``as_completed`` or
        ``add_done_callback``).

        A stored output, or an input its program refuses, resolves the future before
        this returns. The others start in the order submitted, as the runner's
        threads allow: a calculation without a ``threads`` keyword takes all of them,
        so to run many at once, set ``threads`` (usually 1) on each input (the docs'
        Threads page). A calculation never waits on a future itself: a program runs
        its subprograms through ``ctx.compute``.

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
            ``future.cancel()`` cancels the calculation even while it runs: a
            waiting one never starts; a running one in a worker process is killed
            with every process it started, and one in this process ends at its next
            look at ``ctx.timeout`` (which reads 0) and starts no more subprograms,
            as for a timeout. A cancelled calculation's output is never stored.

        Raises:
            RuntimeError: Called from a thread that is running a calculation or a
                session request.
        """
        check_free()
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
        holder = self._top()
        gpus = self._gpus_of(holder, plan)
        threads = _threads(inp, holder.threads, _unasked(inp, gpus, holder.threads))
        task = _Task(0, inp, entry, plan, threads, gpus)
        control = _Control()
        future = _Future(control)
        stored = (store, key, tags, meta)
        with self._submitting:
            if self._futures is None:
                self._futures = ThreadPoolExecutor(
                    self.threads, thread_name_prefix="qccompute-submit"
                )
            self._submitted.add(future)
            self._futures.submit(
                self._run_submitted, task, future, timeout, progress, stored, control
            )
        future.add_done_callback(self._submitted.discard)
        return future

    def _run_submitted(
        self,
        task: _Task,
        future: _Future,
        timeout: float | None,
        progress: ProgressCallback | None,
        stored: tuple[Store | None, str, Sequence[str], Mapping[str, Any] | None],
        control: _Control,
    ) -> None:
        """Run a submitted calculation once its turn for the runner's threads comes."""
        if future.cancelled():
            return
        try:
            out, failure = self._execute(
                task.inputs[0],
                task.threads,
                task.gpus,
                self._top(task.gpus),
                timeout,
                task.plan,
                task.entry,
                None,
                progress,
                control,
            )
            if future.cancelled():  # its output is nobody's
                return
            store, key, tags, meta = stored
            if store is not None:
                _put(store, key, out, tags, meta)
        except Closed:  # the runner closed while it waited
            future.cancel()
            return
        except BaseException as e:  # noqa: BLE001 - a bug of the runner's: raise it
            future._finish(None, e)
            return
        future._finish(out, failure)

    def stop(self, future: Future[Output]) -> bool:
        """Ask a submitted calculation to stop at its next step, keeping what it reached.

        The program sees the request as ``ctx.stop_requested()`` (in a worker process
        too), ends at its next step and returns its partial output (a search's frames
        so far) as a failure of kind ``stopped``: ``future.result()`` raises
        `qcdata.ProgramError` with it as ``output``. The request covers the
        calculation's subprograms, and one that has not started yet stops at its
        first step. A program with no steps runs to its end. A stopped output is never
        stored. Unlike ``future.cancel()``, which discards the calculation, a stop
        keeps what it reached.

        Args:
            future: A future from `submit`.

        Returns:
            Whether the request was made: False once the calculation is done.

        Raises:
            ValueError: The future is not one `submit` made.
        """
        return _stop(future)

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
        process it is the program's own session, behind a thin wrapper that counts
        its requests' threads. Programs in worker processes give a session over the
        worker's pipe, with the same methods; sessions of a program on a GPU share its
        worker there. For any other program the session runs each geometry through
        ``compute``, from scratch (``record["warm_start"]`` is false): it is a
        convenience, not a speed-up.

        The session's thread count is ``threads``, else its input's ``threads``
        keyword, else the runner's whole count (1 for a session given GPUs). Each
        request runs on it: the requests of a ``compute_many``, and requests from
        several threads at once, share it. An open session that computes nothing
        holds no threads. A session gets the GPUs its plan asks for.

        Args:
            inp: The program, model, keywords and structure; its calctype and
                geometry are the defaults of a request.
            calctypes: The calctypes the session may run (default: everything the
                method supports; declaring narrows it).
            threads: The session's thread count (at most the runner's).
            progress: Called with the events the program emits while the session
                runs requests; see `compute`.

        Returns:
            The session; use it as a context manager, or close it.

        Raises:
            ProgramError: The program is not installed or rejects the input.
        """
        if threads is not None and (isinstance(threads, bool) or threads < 1):
            raise ValueError("threads must be a positive integer.")
        return self._open_session(inp, calctypes, threads, self._top(), None, progress)

    # ------------------------------------------------------------------ lists

    def _run_list(
        self,
        inputs: list[Input],
        store: Store | None,
        rerun: bool,
        tags: Sequence[str],
        meta: Mapping[str, Any] | None,
        holder: Count,
        timeout: float | None,
        parent: RunContext | None = None,
        progress: ProgressCallback | None = None,
        shared: bool = False,
    ) -> Iterator[tuple[int, Output, ProgramError | None]]:
        """Yield (index, output, failure) for every input, as each finishes.

        The inputs share ``holder``'s count: each that runs gets the count divided by
        how many run (store hits and refused inputs don't), at least 1, unless it sets
        ``threads`` (1 when it is given GPUs); a batch program's inputs run as one task
        with the sum of theirs. ``timeout`` is each calculation's; ``parent`` is the
        context of the calculation whose subprograms they are, else ``progress`` the
        caller's progress callable. ``shared``: every input gets its share, whatever
        its ``threads`` (a session's requests, whose keyword is the session's count).
        """
        keys: dict[int, str] = {}
        planned: list[tuple[int, Input, _Entry, Plan]] = []
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
            planned.append((i, inp, entry, plan))
        if not planned:
            return
        count = holder.threads
        share = max(1, count // len(planned))
        tasks: list[_Task] = []
        batches: dict[int, _Task] = {}
        for i, inp, entry, plan in planned:
            gpus = self._gpus_of(holder, plan)
            wanted = _unasked(inp, gpus, share)
            if not shared:
                wanted = _threads(inp, count, wanted)
            if entry.caps.batches and isinstance(entry.program, BatchProgram):
                task = batches.get(id(entry))
                if task is not None:
                    task.add(i, inp, wanted, gpus, count)
                    continue
                task = batches[id(entry)] = _Task(i, inp, entry, plan, wanted, gpus)
            else:
                task = _Task(i, inp, entry, plan, wanted, gpus)
            tasks.append(task)

        def run(task: _Task) -> list[tuple[int, Output, ProgramError | None]]:
            return self._run_task(task, holder, timeout, parent, progress)

        pool = None
        if len(tasks) == 1:  # nothing to run beside: no thread
            finished: Iterable[list[Any]] = [run(tasks[0])]
        else:
            pool = ThreadPoolExecutor(max_workers=min(len(tasks), count))
            futures = [pool.submit(run, task) for task in tasks]
            finished = (f.result() for f in as_completed(futures))
        try:
            for items in finished:
                for item in items:
                    if store is not None:
                        _put(store, keys[item[0]], item[1], tags, meta)
                    yield item
        finally:
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)

    def _run_task(
        self,
        task: _Task,
        holder: Count,
        timeout: float | None,
        parent: RunContext | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[tuple[int, Output, ProgramError | None]]:
        """Run one task of a list under ``holder``."""
        if len(task.inputs) == 1:
            out, failure = self._execute(
                task.inputs[0],
                task.threads,
                task.gpus,
                holder,
                timeout,
                task.plan,
                task.entry,
                parent,
                progress,
            )
            return [(task.indices[0], out, failure)]
        control = parent._control if parent is not None else None
        try:
            devices = holder.take(task.threads, task.gpus, control)
        except ProgramError as e:  # cancelled or stopped while it waited
            return [
                (i, *_refused(inp, e, self.identities(inp)))
                for i, inp in zip(task.indices, task.inputs, strict=True)
            ]
        try:
            if parent is not None:
                timeout = parent.timeout
            share = holder.share(task.threads, devices)
            memory = None if holder.memory is None else holder.memory * share
            ctx = self._context(
                task.threads,
                devices,
                memory,
                holder.memory,
                timeout,
                None,
                None,
                parent,
                task.inputs,
                progress,
            )
            return self._call_batch(task, ctx)
        finally:
            holder.give(task.threads, devices)

    def _call_batch(
        self, task: _Task, ctx: RunContext
    ) -> list[tuple[int, Output, ProgramError | None]]:
        """Call a batch program with its context; one result per input, in order."""
        threads, devices = ctx.threads, ctx.gpus
        assert threads is not None
        entry = task.entry
        start = time.perf_counter()
        worker = None
        try:
            with holding():
                if self._in_worker(entry):
                    ctx._identities = _merged(self.identities(i) for i in task.inputs)
                    worker = self._worker(entry, devices, threads)
                    outs = worker.compute_batch(task.inputs, ctx)
                elif entry.lock is not None:
                    with entry.lock, hold_threads(own_threads(threads, task.inputs)):
                        outs = entry.program.compute_batch(task.inputs, ctx)  # type: ignore[attr-defined]
                else:
                    with hold_threads(own_threads(threads, task.inputs)):
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
        if ctx._refusal is not None:  # it called a program no input names
            refusal = ctx._refusal
            outs = [
                _unnamed(i, refusal, o)[0]
                for i, o in zip(task.inputs, outs, strict=True)
            ]
        wall = time.perf_counter() - start  # the batch's: its inputs ran together
        parts = task.threads_of(threads)
        done: list[tuple[int, Output, ProgramError | None]] = []
        for i, inp, out, part in zip(
            task.indices, task.inputs, outs, parts, strict=True
        ):
            out = _with_execution(
                out, wall, part, self.identities(inp), devices, batch=True
            )
            done.append((i, out, None if out.success else _error(out)))
        return done

    # ------------------------------------------------------------------ subprograms

    def _subcompute(self, inputs: Any, ctx: RunContext, raise_exc: bool) -> Any:
        """``RunContext.compute``: subprograms under the calculation's count.

        Each gets the calculation's threads unless it sets ``threads`` (a list shares
        them), and the time the calculation has left; once it has none, they fail
        with kind ``timeout`` without running. A context no runner made (``threads``
        None) runs them under this runner's count.
        """
        holder = self._top() if ctx.threads is None else _children(ctx)
        timeout = ctx.timeout
        one = isinstance(inputs, ProgramInput | FileInput)
        if timeout == 0:  # the calculation's time ran out: none starts
            listed = [inputs] if one else list(inputs)
            failed = [
                _refused(i, _out_of_time(i.program), self.identities(i)) for i in listed
            ]
            return _subprogram_results(inputs if one else listed, failed, raise_exc)
        if one:
            ran = [self._run(inputs, holder, timeout, ctx)]
            return _subprogram_results(inputs, ran, raise_exc)
        inputs = list(inputs)
        results: list[Any] = [None] * len(inputs)
        for i, out, failure in self._run_list(
            inputs, None, False, (), None, holder, timeout, ctx
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
        """``RunContext.session``: a subprogram's session under the calculation's count.

        Once the calculation's time has run out, the session neither opens nor
        computes (kind ``timeout``).
        """
        if ctx.timeout == 0:
            error = _out_of_time(inp.program)
            error._subprogram = inp.program  # type: ignore[attr-defined]
            raise error
        holder = self._top() if ctx.threads is None else _children(ctx)
        return self._open_session(inp, calctypes, threads, holder, ctx)

    def _open_session(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None,
        threads: int | None,
        outer: Count,
        parent: RunContext | None,
        progress: ProgressCallback | None = None,
    ) -> Session:
        """Open a session under ``outer``, the count its requests also run under.

        Its count is ``threads``, else its input's ``threads`` keyword, else
        ``outer``'s whole count (1 when it is given GPUs), at most ``outer``'s; its
        host memory follows. ``parent`` is the context of the calculation whose
        subprogram it is (its time left is the session's), None for a session of the
        caller's own.
        """
        entry, plan = self._plan(inp)
        allowed = session_calctypes(calctypes) if calctypes else None
        below = parent if parent is not None and parent.threads is not None else None
        granted = self._devices(plan, None if below is None else below.gpus)
        if threads is not None:
            count = max(1, min(threads, outer.threads))
        else:
            count = _threads(
                inp, outer.threads, _unasked(inp, len(granted), outer.threads)
            )
        memory = outer.memory
        if memory is not None:
            memory *= min(1.0, count / outer.threads)
        session = Count(count, granted, memory, outer)  # its requests run under it
        program = entry.program
        opened: Session
        lock = entry.lock
        native = entry.caps.sessions and (
            self._in_worker(entry) or isinstance(program, SessionProgram)
        )
        timeout = None if parent is None else parent.timeout
        if not native:  # each request is a calculation, placed as any other
            if allowed is None:
                method = inp.model.method if inp.model is not None else None
                supported = (entry.caps.methods or {}).get(
                    method or "", entry.caps.calctypes
                )
                allowed = session_calctypes(None, supported & session_calctypes())
            compute = partial(self._session_compute, session, timeout, parent, progress)
            over = ComputeSession(
                compute, inp, allowed, program_version=program.version()
            )
            # Each compute takes the program's lock itself.
            return over if parent is None else _Guarded(over, None, parent)
        ctx = self._context(
            count,
            granted,
            memory,
            outer.memory,
            timeout,
            None,
            inp,
            parent,
            None,
            progress,
        )
        held: int | None = None  # the threads its requests hold the pools at here
        if self._in_worker(entry):
            ctx._identities = self.identities(inp)
            worker = self._worker(entry, granted, count)
            release = partial(self._release, worker, granted)
            opened = worker.session(inp, allowed, ctx, on_close=release)
            lock = None  # the worker is this session's own process
        else:
            held = own_threads(count, (inp,))
            with hold_threads(held):
                opened = program.open_session(inp, ctx, allowed)  # type: ignore[attr-defined]
        try:
            weakref.finalize(opened, ctx.cleanup)
        except TypeError:  # not weak-referenceable
            pass
        return _Guarded(opened, lock, parent, held, session)

    def _session_compute(
        self,
        session: Count,
        timeout: float | None,
        parent: RunContext | None,
        progress: ProgressCallback | None,
        inputs: Any,
        raise_exc: bool = True,
    ) -> Any:
        """Compute for a session over compute, under the session's count.

        One request gets the whole count; a list of them runs as `compute` runs one,
        sharing it.
        """
        if parent is None:
            check_free()
        if isinstance(inputs, ProgramInput):
            threads = _threads(inputs, session.threads, session.threads)
            out, failure = self._execute(
                inputs,
                threads,
                session.gpus,
                session,
                timeout,
                None,
                None,
                parent,
                progress,
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
            session,
            timeout,
            parent,
            progress,
            True,
        )
        for i, out, _ in listed:
            outputs[i] = out
        return outputs


class _Guarded:
    """A session whose requests run under its count, and take turns on its program's lock.

    A program's own session (``count``: the session's) has each request take the
    session's threads for as long as it runs, so requests from several threads take
    turns within the session's count and the count it runs under; a session over
    compute (no ``count``) has its requests counted as the calculations they are.
    Sessions of a program that is not thread-safe may be called from several threads
    (ten TS searches side by side): they take turns on one lock per program. A
    subprogram's session (``parent``: the context of the calculation it serves)
    marks its failures, so a program that lets one escape fails with kind
    ``program``, and computes nothing once the calculation's time has run out (kind
    ``timeout``). A program's own session in this process has its requests hold the
    BLAS and OpenMP pools at ``threads`` (None: its requests run in a worker).
    """

    def __init__(
        self,
        session: Session,
        lock: _Turns | None,
        parent: RunContext | None = None,
        threads: int | None = None,
        count: Count | None = None,
    ) -> None:
        self._session = session
        self._lock = lock if lock is not None else _NoLock()
        self._parent = parent
        self._threads = threads
        self._count = count
        self._control = None if parent is None else parent._control
        self._subprogram = None if parent is None else session.program

    def _out_of_time(self) -> ProgramError | None:
        """The failure of a request made after the calculation's time ran out."""
        if self._parent is None or self._parent.timeout != 0:
            return None
        return _out_of_time(self._subprogram or self.program)

    def _take(self) -> ProgramError | None:
        """Wait for the session's threads; the failure if cancelled or stopped meanwhile."""
        if self._parent is None:
            check_free()
        count = self._count
        if count is not None:
            try:
                count.take(count.threads, 0, self._control)
            except ProgramError as e:
                return e
        return None

    def _give(self) -> None:
        """Give back what `_take` took."""
        if self._count is not None:
            self._count.give(self._count.threads)

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
            late = self._out_of_time() or self._take()
            if late is not None:
                raise late
            try:
                with holding(), self._lock, hold_threads(self._threads):
                    return self._session.compute(*args, **kwargs)
            finally:
                self._give()
        except ProgramError as e:
            if self._subprogram is not None:
                e._subprogram = self._subprogram  # type: ignore[attr-defined]
            raise

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """See `qccompute.program.Session.compute_many`."""
        try:
            late = self._out_of_time() or self._take()
            if late is not None:  # every request fails, as a group
                return ProgramErrors.check(
                    [
                        failed_frame(self._structure(r), late, self.program)
                        for r in requests
                    ]
                )
            try:
                with holding(), self._lock, hold_threads(self._threads):
                    return self._session.compute_many(requests)
            finally:
                self._give()
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
        failed calculation, and ``cancel()`` ends it even while it runs (see
        `Runner.submit`).
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


def stop(future: Future[Output]) -> bool:
    """Ask a submitted calculation to stop at its next step: `Runner.stop`.

    Args:
        future: A future from `submit` (any runner's).

    Returns:
        Whether the request was made: False once the calculation is done.

    Raises:
        ValueError: The future is not one `submit` made.
    """
    return _stop(future)


def _stop(future: Future[Output]) -> bool:
    """`Runner.stop`, which needs no runner: the future holds its calculation's control."""
    if future.done():
        return False
    if not isinstance(future, _Future):
        raise ValueError("Only a future from submit() can be stopped.")
    future._control.stop()
    return True


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
