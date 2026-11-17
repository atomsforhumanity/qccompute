"""The Runner: one way to run programs, one input or many, in this process or in workers.

``qccompute.compute`` and ``qccompute.session`` are the methods of the default runner,
made on first use for this machine (all its usable cores and visible GPUs). A `Runner`
of your own takes a share of the machine and closes its worker processes when you are
done with it::

    with qccompute.Runner(threads=16) as runner:
        outputs = runner.compute(inputs, store=store, tags=["screen"])

Every input goes one path: find the program by name, plan it (lists only, to pack them),
place it (this process, or a worker process for programs that cannot share one), call
``compute`` or ``compute_batch`` with a `qccompute.program.RunContext`, and record the
execution. A failed calculation raises `qcdata.ProgramError` with the failed output as
its ``output``; a list runs everything first and raises `qcdata.ProgramErrors`.
"""

from __future__ import annotations

import atexit
import math
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
)

from .program import (
    BatchProgram,
    Capabilities,
    ComputeSession,
    Plan,
    Program,
    RunContext,
    Session,
    SessionProgram,
    _entry_points,
    capabilities_of,
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
    "session",
    "default_runner",
]

Input = ProgramInput | FileInput
Output = ProgramOutput[Any, Any]

#: The program that runs a `qcdata.FileInput` (a command with its files): qccodec's.
FILES = "files"

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
    ``identities``: ``{program: numerics}`` for the input's program and every
    subprogram it names.
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
        identities: Mapping[str, str],
        *,
        tags: Sequence[str] = (),
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        """Store a run under its input's key, with the identities that produced it."""
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


def visible_gpus() -> dict[str, float | None]:
    """The GPUs this process may use and their memory (GiB, if known).

    ``CUDA_VISIBLE_DEVICES`` wins when set; otherwise ``nvidia-smi`` lists the NVIDIA
    GPUs. Empty when there are none or they cannot be found.
    """
    listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    memory: dict[str, float | None] = {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        for line in result.stdout.splitlines():
            index, _, total = line.partition(",")
            if index.strip():
                memory[index.strip()] = float(total) / 1024 if total.strip() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    if listed is not None:
        ids = [d.strip() for d in listed.split(",") if d.strip() and d.strip() != "-1"]
        return {d: memory.get(d) for d in ids}
    return memory


@overload
def _host() -> tuple[str, int | None]: ...
@overload
def _host(memory: Literal[True]) -> tuple[str, int | None, float | None]: ...
def _host(memory: bool = False) -> Any:
    """This machine's name, logical CPU count and RAM, read once per process."""
    global _HOST
    if _HOST is None:
        try:
            ram: float | None = (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
            )
        except (AttributeError, ValueError, OSError):
            ram = None
        _HOST = (platform.node(), os.cpu_count(), ram)
    return _HOST if memory else _HOST[:2]


_HOST: tuple[str, int | None, float | None] | None = None


# --------------------------------------------------------------------- the programs


class _Entry:
    """A program as a runner places it: its capabilities, lock and worker spec."""

    __slots__ = ("program", "caps", "lock", "spec")

    def __init__(self, program: Program, caps: Capabilities, spec: str) -> None:
        self.program = program
        self.caps = caps
        self.lock = None if caps.thread_safe else _program_lock(program)
        self.spec = spec


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


def _worker_spec(program: Program) -> str:
    """How a worker process finds a program: its name, or ``module:attribute``.

    Installed programs are found by name. A program registered in this process is
    found in a worker only if it is a top-level attribute of an importable module.
    """
    if program.name in _entry_points():
        return program.name
    module = type(program).__module__
    if module != "__main__":
        for name, value in vars(sys.modules.get(module, object())).items():
            if value is program:
                return f"{module}:{name}"
    return program.name


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


def _put(
    store: Store,
    out: Output,
    identities: Mapping[str, str],
    tags: Sequence[str],
    meta: Mapping[str, Any] | None,
) -> None:
    """Store a run, unless it was warm-started (its numbers depend on history)."""
    if out.record.get("warm_start") is True:
        return
    store.put(out, identities, tags=tuple(tags), meta=meta)


def _with_execution(out: Output, wall_time: float, threads: int | None) -> Output:
    """The output with what the runner measured; the program's own timings stay.

    The output is the caller's from here on, so it is updated in place; its
    `ExecutionInfo` is copied (a program may share one between outputs) without
    validating it again.
    """
    hostname, host_cpu, host_mem = _host(True)
    execution = out.execution
    new = execution.model_copy(
        update={
            "wall_time": round(wall_time, 6),
            "hostname": hostname,
            "host_cpu": host_cpu,
            "host_mem_gib": host_mem,
            "threads": execution.threads if execution.threads is not None else threads,
        }
    )
    out.__dict__["execution"] = new
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
        self._ready = threading.Condition()

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
            self._turn += 1
            self._used += share
            devices = tuple(g for g, n in self._slots.items() if n > 0)[:gpus]
            for g in devices:
                self._slots[g] -= 1
            self._ready.notify_all()
        return threads, devices, share

    def release(self, devices: tuple[str, ...], share: float) -> None:
        """Return a finished task's room."""
        with self._ready:
            self._used -= share
            for g in devices:
                self._slots[g] += 1
            self._ready.notify_all()


class _Task:
    """One unit of a list: an input, or a batch of one program's inputs."""

    __slots__ = ("indices", "inputs", "entry", "threads", "gpus", "plan")

    def __init__(
        self,
        indices: list[int],
        inputs: list[Input],
        entry: _Entry,
        plan: Plan,
    ) -> None:
        self.indices = indices
        self.inputs = inputs
        self.entry = entry
        self.threads = plan.resources.threads
        self.gpus = plan.resources.gpus
        self.plan = plan


# ---------------------------------------------------------------------------- runner


class Runner:
    """Runs programs on a share of this machine: one input, a list, or a session.

    Args:
        threads: CPU threads the runner may use at once; None: every core this process
            may use (its affinity and cgroup quota).
        gpus: GPUs it may use: ``"auto"`` for every visible GPU (``CUDA_VISIBLE_DEVICES``
            or ``nvidia-smi``), or a list of device identifiers; ``()`` for none.
        backend: Where calculations run. Only ``"local"`` (this machine) for now.
        isolation: ``"auto"`` runs programs in this process unless they cannot share
            it (``in_process=False``), which run in worker processes; ``"process"``
            runs every program in a worker process, so a crash ends only the worker.
        per_gpu: Calculations that may share one GPU at once; each gets 1/per_gpu of
            its memory.

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
        self._memory: float | None | Literal[False] = False
        self._entries: dict[int, _Entry] = {}
        self._entries_lock = threading.Lock()
        self._workers: dict[tuple[str, tuple[str, ...], int], list[Worker]] = {}
        self._workers_lock = threading.Lock()
        self._all_workers: weakref.WeakSet[Worker] = weakref.WeakSet()
        self._closed = False

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

    def _device_memory(self, devices: Sequence[str]) -> float | None:
        """Each calculation's share of its GPUs' memory (GiB), if known."""
        if not devices or self._gpus is None:
            return None
        sizes = [self._gpus.get(d) for d in devices]
        # 1/k of each GPU per calculation, with k calculations per GPU.
        if any(s is None for s in sizes):
            return None
        return min(s for s in sizes if s is not None) / self.per_gpu

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

    def _entry_of(self, program: Program) -> _Entry:
        entry = self._entries.get(id(program))
        if entry is not None and entry.program is program:
            return entry
        with self._entries_lock:
            entry = _Entry(program, capabilities_of(program), _worker_spec(program))
            self._entries[id(program)] = entry
        return entry

    def _in_worker(self, entry: _Entry) -> bool:
        return self.isolation == "process" or not entry.caps.in_process

    def _devices(self, entry: _Entry, devices: Sequence[str] | None) -> tuple[str, ...]:
        """The GPUs of a calculation: those granted, else all of the runner's."""
        if not entry.caps.gpu:
            return ()
        return tuple(devices) if devices is not None else self.gpus

    def _context(
        self,
        threads: int,
        devices: tuple[str, ...],
        memory: float | None,
        timeout: float | None,
        plan: Plan | None,
    ) -> RunContext:
        return RunContext(
            threads=threads,
            gpus=devices,
            memory_gib=None if memory is None else round(memory, 3),
            device_memory_gib=self._device_memory(devices),
            timeout=timeout,
            plan=plan,
            runner=self,
        )

    # ------------------------------------------------------------------ workers

    def _worker(self, entry: _Entry, devices: tuple[str, ...], threads: int) -> Worker:
        """An idle worker for this program, GPUs and threads; started if none is."""
        from .worker import Worker

        if self._closed:
            raise RuntimeError("This runner is closed.")
        key = (entry.spec, devices, threads)
        with self._workers_lock:
            idle = self._workers.setdefault(key, [])
            while idle:
                worker = idle.pop()
                if worker.alive:
                    return worker
        worker = Worker(
            entry.spec, threads=threads, gpus=devices if entry.caps.gpu else None
        )
        self._all_workers.add(worker)
        return worker

    def _release(self, worker: Worker, devices: tuple[str, ...], threads: int) -> None:
        if not worker.alive or self._closed:
            worker.close()
            return
        with self._workers_lock:
            self._workers.setdefault((worker.spec, devices, threads), []).append(worker)

    def close(self) -> None:
        """Stop this runner's worker processes; it cannot run in workers afterwards."""
        self._closed = True
        with self._workers_lock:
            self._workers.clear()
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
        devices: Sequence[str] | None,
        memory: float | None,
        timeout: float | None,
        plan: Plan | None = None,
        entry: _Entry | None = None,
    ) -> tuple[Output, ProgramError | None]:
        """Run one input to its output, failed or not, and the error of a failure."""
        start = time.perf_counter()
        if entry is None:
            entry = self._entry(inp)
            if entry is None:
                exc = _not_found(inp)
                out = _failure(inp, exc.error)
                exc.output = out
                return out, exc
        granted = self._devices(entry, devices)
        ctx = self._context(threads, granted, memory, timeout, plan)
        worker = None
        try:
            if self._in_worker(entry):
                worker = self._worker(entry, granted, threads)
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
                self._release(worker, granted, threads)
        out = _with_execution(out, time.perf_counter() - start, threads)
        if failure is not None:
            failure.output = out
        return out, failure

    def identities(self, inp: Input) -> dict[str, str]:
        """The numerics identity of the input's program and of every subprogram.

        ``{name: numerics}``, by name, recursively through the subprograms' own
        subprograms, from the programs this runner would run now, each asked with its
        own spec (`qccompute.program.numerics_of`). A program named twice with
        different identities gets them all, sorted and joined by ``" + "``. A program
        that is not installed is left out.

        Args:
            inp: The input.

        Returns:
            The identities a store files a run of this input under.
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
        return {name: " + ".join(sorted(found[name])) for name in sorted(found)}

    def _lookup(
        self, store: Store, inp: Input, rerun: bool
    ) -> tuple[str, dict[str, str], Output | None]:
        """The key and identities of an input, and its stored output if reusable.

        A stored run is reused only when it was made with the identities of the
        programs this runner would run now: a success always, a failure only when the
        input determines it (kind ``input``, ``unsupported`` or ``convergence``, or
        ``program`` with such a cause).
        """
        key = store.key(inp)
        identities = self.identities(inp)
        if rerun:
            return key, identities, None
        hit = store.get(key, identities)
        if hit is None or hit.success or _deterministic(hit.error):
            return key, identities, hit
        return key, identities, None

    def _one(
        self,
        inp: Input,
        store: Store | None,
        timeout: float | None,
        rerun: bool,
        tags: Sequence[str],
        meta: Mapping[str, Any] | None,
    ) -> tuple[Output, ProgramError | None]:
        if store is not None:
            key, identities, hit = self._lookup(store, inp, rerun)
            if hit is not None:
                if tags:
                    store.tag(key, *tags)
                return hit, None if hit.success else _error(hit)
        out, failure = self._execute(
            inp, self.threads, None, self._host_memory(), timeout
        )
        if store is not None:
            _put(store, out, identities, tags, meta)
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
    ) -> Output | list[Output]:
        """Run one input, or a list of independent inputs side by side.

        One input gets the runner's whole share of the machine. A list's inputs are
        planned first (an input its program refuses fails at once, without running),
        then packed by the threads and GPUs their plans ask for; inputs of a program
        with batches go to it together.

        Args:
            inputs: One input, or a list of them (of any programs).
            store: Look each input up first and return a reusable stored output; store
                every output that ran, with the identities of the programs that made
                it (`identities`). A stored run is reused only when its identities
                are those of the programs the runner would run now: a success always,
                a failure only when the input determines it (kind ``input``,
                ``unsupported`` or ``convergence``, or ``program`` with such a cause);
                ``rerun=True`` runs past it.
            timeout: Seconds each calculation may run. Programs that start processes
                enforce it, and so does the runner for programs in worker processes;
                it fails with kind ``timeout``.
            rerun: Run even when the store has a reusable output.
            tags: Tags for every key computed or reused (needs ``store``).
            meta: Metadata stored with each run (needs ``store``).
            raise_exc: Raise for failures (after every input of a list ran); False
                returns failed outputs in place.

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
            out, failure = self._one(inputs, store, timeout, rerun, tags, meta)
            if failure is not None and raise_exc:
                raise failure
            return out
        inputs = list(inputs)
        outputs: list[Output | None] = [None] * len(inputs)
        failures: dict[int, ProgramError] = {}
        grant = (self.threads, None, self._host_memory(), timeout)
        for i, out, failure in self._run_list(inputs, store, rerun, tags, meta, grant):
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
        for _, out, failure in self._run_list(inputs, store, rerun, tags, meta, grant):
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
    ) -> Session:
        """Open a session: one program and molecule at many geometries.

        A program with sessions of its own (``capabilities().sessions``) opens one,
        pinned to the input's settings and warm-started between geometries: in this
        process it is the program's own session object (``linus.Session``), exactly
        what a direct user gets. For any other program the session runs each
        geometry through ``compute``. Programs in worker processes give a session
        over the worker's pipe, with the same methods.

        Args:
            inp: The program, model, keywords and structure; its calctype and
                geometry are the defaults of a request.
            calctypes: The calctypes the session may run (default: everything the
                method supports; declaring narrows it).

        Returns:
            The session; use it as a context manager, or close it.

        Raises:
            ProgramError: The program is not installed or rejects the input.
        """
        return self._open_session(
            inp, calctypes, self.threads, None, self._host_memory(), None, False
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
    ) -> Iterator[tuple[int, Output, ProgramError | None]]:
        """Yield (index, output, failure) for every input, as each finishes.

        ``grant`` is what the list may use: threads, GPUs (None: the runner's), host
        memory (GiB) and the timeout of each calculation.
        """
        threads, devices, memory, timeout = grant
        keys: dict[int, str] = {}
        identities: dict[int, dict[str, str]] = {}
        tasks: list[_Task] = []
        batches: dict[int, _Task] = {}
        for i, inp in enumerate(inputs):
            if store is not None:
                keys[i], identities[i], hit = self._lookup(store, inp, rerun)
                if hit is not None:
                    if tags:
                        store.tag(keys[i], *tags)
                    yield i, hit, None if hit.success else _error(hit)
                    continue
            entry = self._entry(inp)
            try:
                if entry is None:
                    raise _not_found(inp)
                plan = plan_of(entry.program, inp)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001 - refused before running
                out, failure = _as_failure(inp, e, inp.program)
                out = _with_execution(out, 0.0, None)
                failure.output = out
                if store is not None:
                    _put(store, out, identities[i], tags, meta)
                yield i, out, failure
                continue
            assert entry is not None
            if entry.caps.batches and isinstance(entry.program, BatchProgram):
                task = batches.get(id(entry))
                if task is not None:
                    task.indices.append(i)
                    task.inputs.append(inp)
                    continue
                task = batches[id(entry)] = _Task([i], [inp], entry, plan)
            else:
                task = _Task([i], [inp], entry, plan)
            tasks.append(task)
        if not tasks:
            return
        if devices is None:
            needs_gpus = any(t.entry.caps.gpu and t.gpus for t in tasks)
            devices = self.gpus if needs_gpus else ()
        budget = _Grant(threads, devices, self.per_gpu)

        def run(
            turn: int, task: _Task
        ) -> list[tuple[int, Output, ProgramError | None]]:
            held, granted, share = budget.acquire(turn, task.threads, task.gpus)
            mine = None if memory is None else memory * share
            try:
                return list(self._run_task(task, held, granted, mine, timeout))
            finally:
                budget.release(granted, share)

        if len(tasks) == 1:  # nothing to run beside: no thread
            for item in run(0, tasks[0]):
                if store is not None:
                    _put(store, item[1], identities[item[0]], tags, meta)
                yield item
            return
        pool = ThreadPoolExecutor(max_workers=min(len(tasks), budget.threads))
        try:
            futures = [pool.submit(run, turn, task) for turn, task in enumerate(tasks)]
            for future in as_completed(futures):
                for item in future.result():
                    if store is not None:
                        _put(store, item[1], identities[item[0]], tags, meta)
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
    ) -> Iterator[tuple[int, Output, ProgramError | None]]:
        """Run one task of a list with its grant."""
        if len(task.inputs) == 1:
            out, failure = self._execute(
                task.inputs[0], threads, devices, memory, timeout, task.plan, task.entry
            )
            yield task.indices[0], out, failure
            return
        entry = task.entry
        granted = self._devices(entry, devices)
        ctx = self._context(threads, granted, memory, timeout, None)
        start = time.perf_counter()
        worker = None
        try:
            if self._in_worker(entry):
                worker = self._worker(entry, granted, threads)
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
                self._release(worker, granted, threads)
        wall = time.perf_counter() - start
        for i, out in zip(task.indices, outs, strict=True):
            out = _with_execution(out, wall, threads)
            yield i, out, None if out.success else _error(out)

    # ------------------------------------------------------------------ subprograms

    def _subcompute(self, inputs: Any, ctx: RunContext, raise_exc: bool) -> Any:
        """``RunContext.compute``: subprograms within a calculation's grant."""
        threads = ctx.threads or self.threads
        devices = ctx.gpus if ctx.threads is not None else None
        memory = ctx.memory_gib if ctx.threads is not None else self._host_memory()
        timeout = ctx.timeout
        if isinstance(inputs, ProgramInput | FileInput):
            out, failure = self._execute(inputs, threads, devices, memory, timeout)
            if failure is not None and raise_exc:
                failure._subprogram = inputs.program  # type: ignore[attr-defined]
                raise failure
            return out
        inputs = list(inputs)
        outputs: list[Output | None] = [None] * len(inputs)
        failures: dict[int, ProgramError] = {}
        grant = (threads, devices, memory, timeout)
        for i, out, failure in self._run_list(inputs, None, False, (), None, grant):
            outputs[i] = out
            if failure is not None:
                failure._subprogram = inputs[i].program  # type: ignore[attr-defined]
                failures[i] = failure
        done = [o for o in outputs if o is not None]
        if failures and raise_exc:
            raise ProgramErrors(
                f"{len(failures)} of {len(inputs)} subprogram calculations failed",
                [failures[i] for i in sorted(failures)],
                done,
            )
        return done

    def _subsession(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None,
        threads: int | None,
        ctx: RunContext,
    ) -> Session:
        """``RunContext.session``: a subprogram's session within a grant."""
        devices = ctx.gpus if ctx.threads is not None else None
        share = threads or ctx.threads or self.threads
        memory = ctx.memory_gib if ctx.threads is not None else self._host_memory()
        if memory is not None and ctx.threads:
            memory = memory * min(1.0, share / ctx.threads)
        return self._open_session(
            inp, calctypes, share, devices, memory, ctx.timeout, True
        )

    def _open_session(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None,
        threads: int,
        devices: Sequence[str] | None,
        memory: float | None,
        timeout: float | None,
        subprogram: bool,
    ) -> Session:
        entry = self._entry(inp)
        if entry is None:
            raise _not_found(inp)
        allowed = session_calctypes(calctypes) if calctypes else None
        granted = self._devices(entry, devices)
        ctx = self._context(threads, granted, memory, timeout, None)
        program = entry.program
        opened: Session
        lock = entry.lock
        if self._in_worker(entry):
            worker = self._worker(entry, granted, threads)
            release = partial(self._release, worker, granted, threads)
            opened = worker.session(inp, allowed, ctx, on_close=release)
            lock = None  # the worker is this session's own process
        elif entry.caps.sessions and isinstance(program, SessionProgram):
            opened = program.open_session(inp, ctx, allowed)
        else:
            if allowed is None:
                method = inp.model.method if inp.model is not None else None
                supported = (entry.caps.methods or {}).get(
                    method or "", entry.caps.calctypes
                )
                allowed = session_calctypes(None, supported & session_calctypes())
            compute = partial(self._session_compute, threads, granted, memory, timeout)
            opened = ComputeSession(compute, inp, calctypes=allowed)
            lock = None  # each compute takes the program's lock itself
        try:
            weakref.finalize(opened, ctx.cleanup)
        except TypeError:  # not weak-referenceable
            pass
        if lock is None and not subprogram:
            return opened
        return _Guarded(opened, lock, inp.program if subprogram else None)

    def _session_compute(
        self,
        threads: int,
        devices: Sequence[str],
        memory: float | None,
        timeout: float | None,
        inputs: Any,
        raise_exc: bool = True,
    ) -> Any:
        """Compute for a session over compute, within the session's grant."""
        if isinstance(inputs, ProgramInput):
            out, failure = self._execute(inputs, threads, devices, memory, timeout)
            if failure is not None and raise_exc:
                raise failure
            return out
        grant = (threads, devices, memory, timeout)
        outputs: list[Output | None] = [None] * len(inputs)
        for i, out, _ in self._run_list(list(inputs), None, False, (), None, grant):
            outputs[i] = out
        return outputs


class _Guarded:
    """A session that takes turns on its program's lock and marks subprogram failures.

    Sessions of a program that is not thread-safe may be called from several threads
    (ten TS searches side by side): they take turns on one lock per program. A
    subprogram's session marks its failures, so a program that lets one escape fails
    with kind ``program``.
    """

    def __init__(
        self, session: Session, lock: threading.Lock | None, subprogram: str | None
    ) -> None:
        self._session = session
        self._lock = lock if lock is not None else _NoLock()
        self._subprogram = subprogram

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
            with self._lock:
                return self._session.compute(*args, **kwargs)
        except ProgramError as e:
            if self._subprogram is not None:
                e._subprogram = self._subprogram  # type: ignore[attr-defined]
            raise

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """See `qccompute.program.Session.compute_many`."""
        try:
            with self._lock:
                return self._session.compute_many(requests)
        except ExceptionGroup as group:
            if self._subprogram is not None:
                for e in group.exceptions:
                    setattr(e, "_subprogram", self._subprogram)  # noqa: B010
            raise

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

    It uses every core this process may use and every visible GPU, runs programs in
    this process unless they cannot share it, and closes its workers at exit. It
    cannot be reconfigured; make a `Runner` for anything else.
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
    )


def session(
    inp: ProgramInput, calctypes: Iterable[CalcType | str] | None = None
) -> Session:
    """Open a session on this machine: `Runner.session` of the default runner.

    Args:
        inp: The program, model, keywords, structure, and the default calctype.
        calctypes: The calctypes the session may run.

    Returns:
        The session.
    """
    return default_runner().session(inp, calctypes)
