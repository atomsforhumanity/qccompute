"""The client of a worker process: a program hosted by ``python -m qccompute.serve``.

A `Worker` starts the generic worker for one program and talks to it over its pipes,
one JSON line per message (``qccompute.serve`` documents the wire). It offers what the
program offers in process (``compute``, ``compute_batch``, ``plan`` and sessions), so a
runner can place a program in this process or in a worker without callers noticing:
``Worker.session`` returns an object with the `qccompute.program.Session` methods.

The worker's program runs its subprograms here: each ``ctx.compute`` or ``ctx.session``
in the worker comes back as a call that this process's runner answers within the
calculation's grant, so a subprogram may be any program this process finds, registered
ones included. Large arrays cross as ``.npy`` files in a directory of the worker's own,
removed when it closes.

A worker isolates its program: a crash ends the worker, not the caller, and is a
`qcdata.ProgramError` of kind ``environment``, code ``worker_died``, with the exit
status and the end of the worker's stderr. A worker that reports a lost GPU context
(code ``device_lost``) exits after answering; the client then starts no more work on it.
The runner keeps workers alive between calls and closes them with itself.

A worker whose calculation runs out of time, or is cancelled, is killed with every
process it started, however deep and in whatever session (an external program and the
processes it starts in turn): the tree is frozen, then killed.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel
from qcdata import (
    Accuracy,
    CalcType,
    Data,
    ErrorInfo,
    ErrorKind,
    FileInput,
    Frame,
    Lineage,
    ProgramError,
    ProgramErrors,
    ProgramInput,
    ProgramOutput,
    Request,
)

from . import _wire
from .program import (
    Capabilities,
    ComputeSession,
    Plan,
    RunContext,
    Session,
    _Control,
    failed_frame,
    find_program,
    plan_of,
)
from .serve import Host, MessageError

__all__ = ["Worker", "WorkerSession", "THREAD_VARIABLES", "PATH_VARIABLE"]

M = TypeVar("M", bound=BaseModel)

#: The client's ``sys.path`` for the worker, which takes it out of its environment.
PATH_VARIABLE = "QCCOMPUTE_WORKER_SYS_PATH"

#: Thread counts of numerical libraries, set for a worker to the threads of the call
#: that starts it unless the environment already sets them: they size the pools the
#: libraries make when they load. The worker holds each call's BLAS and OpenMP pools at
#: its own threads (as the runner does in process), whatever started the worker.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _children() -> dict[int, list[int]]:
    """Every process's children, by ``/proc`` (Linux); empty where it cannot be read."""
    children: dict[int, list[int]] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return children
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                stat = f.read()
            ppid = int(stat[stat.rindex(b")") + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue  # gone meanwhile
        children.setdefault(ppid, []).append(int(name))
    return children


def _kill_tree(pid: int) -> None:
    """Kill a process and every process it started, however deep.

    The tree is frozen first (``SIGSTOP``), found again until nothing new appears, then
    killed, so a process cannot start another meanwhile. Descendants in sessions of
    their own (qccodec runs each external program in one) are found by their parent,
    through ``/proc``; where there is none, only the process itself is killed.

    Args:
        pid: The process.
    """
    stop = getattr(signal, "SIGSTOP", None)
    tree = [pid]
    seen = {pid}
    while tree:
        for p in tree:
            try:
                os.kill(p, stop if stop is not None else signal.SIGTERM)
            except OSError:  # gone, or not ours
                pass
        if stop is None:
            return
        children = _children()
        tree = [c for p in list(seen) for c in children.get(p, ()) if c not in seen]
        seen.update(tree)
    for p in seen:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


class Worker:
    """One program in a worker process of its own.

    Args:
        program: The program's name (an installed program) or ``module:attribute``.
        threads: CPU threads the process's numerical libraries start with (the
            `THREAD_VARIABLES`, unless the environment sets them); each call's own
            threads come in its ``ctx``, at which the worker holds the BLAS and
            OpenMP pools while the call runs. None: not set.
        gpus: GPUs the process sees (``CUDA_VISIBLE_DEVICES``); None: as this one.
        environment: More environment variables for the process.

    Raises:
        ProgramError: The worker did not start (kind ``environment``, code
            ``worker_died``): the program is not installed, or its import failed.
    """

    def __init__(
        self,
        program: str,
        *,
        threads: int | None = None,
        gpus: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Start the worker, read its hello and set the connection's options."""
        env = dict(os.environ)
        # The worker imports what this process can import; not through PYTHONPATH,
        # which programs the worker starts (Psi4, other Pythons) would inherit.
        env[PATH_VARIABLE] = os.pathsep.join(p for p in sys.path if p)
        if threads is not None:
            for name in THREAD_VARIABLES:
                env.setdefault(name, str(threads))
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        env.update(environment or {})
        self.spec = program
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "qccompute.serve", program],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._stderr: collections.deque[str] = collections.deque(maxlen=40)
        threading.Thread(target=self._drain, daemon=True).start()
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._ids = 0
        self._dead: ProgramError | None = None
        # The caller's control of the message in flight, the file that asks it to
        # stop, and whether it cancelled it; the controls this worker has served.
        self._flight_lock = threading.Lock()
        self._flight: _Control | None = None
        self._stop_file: Path | None = None
        self._stops: Path | None = None  # a directory for stop files, made on use
        self._cancelled = False
        self._watched: weakref.WeakSet[_Control] = weakref.WeakSet()
        self._dir: Path | None = None
        assert self._proc.stdout is not None and self._proc.stdin is not None
        self._lines = _wire.Lines(self._proc.stdout.fileno())
        self._in: IO[bytes] = self._proc.stdin
        self._parent = _Parent()
        self._calls = Host(self._parent)
        line = self._lines.readline()
        if not line:
            raise self._died("did not start")
        hello = json.loads(line)
        self.hello: dict[str, Any] = hello
        self.name: str = hello["program"]
        self.capabilities: Capabilities = Capabilities.model_validate(
            hello["capabilities"]
        )
        if hello.get("array_files") or hello.get("calls"):
            if hello.get("array_files"):
                self._dir = Path(tempfile.mkdtemp(prefix="qccompute-worker-"))
                self._calls.array_dir = self._dir
            self._ask(
                "options",
                array_dir=None if self._dir is None else str(self._dir),
                calls=bool(hello.get("calls")),
            )

    def __repr__(self) -> str:
        """Show the program and process."""
        return f"Worker({self.spec!r}, pid={self._proc.pid}, alive={self.alive})"

    def _drain(self) -> None:
        """Keep the end of the worker's stderr (its logs), so its pipe never fills."""
        assert self._proc.stderr is not None
        with self._proc.stderr:  # closed at the worker's end
            for line in self._proc.stderr:
                self._stderr.append(line.decode("utf-8", "replace"))

    @property
    def alive(self) -> bool:
        """Whether the worker can take work."""
        return self._dead is None and self._proc.poll() is None

    def version(self) -> str:
        """The program's version, from the worker's hello."""
        return str(self.hello["version"])

    def _kill(self) -> None:
        """Kill the worker and every process it started."""
        if self._proc.poll() is None:
            _kill_tree(self._proc.pid)

    def _stop(self, control: _Control) -> None:
        """Ask the message in flight to stop, if it is that calculation's."""
        with self._flight_lock:
            if self._flight is control and self._stop_file is not None:
                self._stop_file.touch()

    def _cancel(self, control: _Control) -> None:
        """Kill the worker if the message in flight is that of a cancelled calculation."""
        with self._flight_lock:
            if self._flight is control:
                self._cancelled = True
                self._kill()

    def _died(self, what: str) -> ProgramError:
        """The failure of a worker that exited, with its status and stderr's end."""
        if self._cancelled:
            what = "was killed: its calculation was cancelled"
        try:
            status = self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._kill()
            status = self._proc.wait()
        tail = "".join(self._stderr).strip()
        self._dead = ProgramError(
            ErrorInfo(
                kind=ErrorKind.environment,
                code="worker_died",
                message=f"The worker for {self.spec} {what} (exit status {status})"
                + (f":\n{tail}" if tail else "."),
            )
        )
        return self._dead

    def _send(self, message: dict[str, Any]) -> None:
        """Write one line to the worker."""
        data = (_wire.encode(message) + "\n").encode("utf-8")
        with self._write_lock:
            self._in.write(data)
            self._in.flush()

    def _ask(
        self,
        op: str,
        timeout: float | None = None,
        grant: RunContext | None = None,
        **fields: Any,
    ) -> Any:
        """Send one message and return its result.

        While it is in flight, the worker's program may call back for subprograms;
        each call is answered in a thread of its own, within ``grant``. Its progress
        events go to ``grant.progress``, in this thread.

        Raises:
            ProgramError: The worker refused the message (``ok: false``), died, or
                did not answer within ``timeout`` (then it is killed with every
                process it started: kind ``timeout``). A cancelled calculation's
                worker is killed the same way.
        """
        control = None if grant is None else grant._control
        with self._lock:
            if self._dead is not None:
                raise ProgramError(self._dead.error)
            self._ids += 1
            self._parent.ctx = grant
            if control is not None:
                stop_file = self._stop_file_for(op, fields)
                with self._flight_lock:
                    cancelled = control.cancelled
                    self._flight = None if cancelled else control
                    self._stop_file = stop_file
                    if stop_file is not None and control.stopping:
                        stop_file.touch()
                if cancelled:  # never sent: the worker serves on
                    raise ProgramError.of(
                        "timeout", "The calculation was cancelled before it ran."
                    )
                if control not in self._watched:
                    self._watched.add(control)
                    control.on_cancel(partial(self._cancel, control))
                    control.on_stop(partial(self._stop, control))
            try:
                reply = self._exchange(op, timeout, grant, fields)
            finally:
                with self._flight_lock:
                    self._flight = None
                    if self._stop_file is not None:
                        self._stop_file.unlink(missing_ok=True)
                        self._stop_file = None
        if not reply.get("ok"):
            raise ProgramError(ErrorInfo.model_validate(reply["error"]))
        if self._proc.poll() is not None or _lost(reply.get("result")):
            self._died("exited after losing its GPU context")
        return reply.get("result")

    def _stop_file_for(self, op: str, fields: dict[str, Any]) -> Path | None:
        """The file that asks a one-off calculation to stop, named in its wire ``ctx``.

        A submitted calculation's ``compute`` and ``compute_batch`` carry one (session
        requests are steps of their caller, which sees its own stop).
        """
        ctx = fields.get("ctx")
        if op not in ("compute", "compute_batch") or not isinstance(ctx, dict):
            return None
        if self._stops is None:
            self._stops = Path(tempfile.mkdtemp(prefix="qccompute-stop-"))
        path = self._stops / str(self._ids)
        ctx["stop_file"] = str(path)
        return path

    def _exchange(
        self,
        op: str,
        timeout: float | None,
        grant: RunContext | None,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Send the message and read lines to its reply (`_ask`, under its lock)."""
        try:
            self._send({"id": self._ids, "op": op, **fields})
        except (BrokenPipeError, OSError):
            raise self._died("died") from None
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if deadline is None else deadline - time.monotonic()
            line = self._lines.readline(None if left is None else max(0.0, left))
            if line is None:
                self._kill()
                self._died("was killed")
                raise ProgramError.of(
                    "timeout",
                    f"{self.name} did not finish within {timeout:g} s; its worker "
                    "was killed.",
                )
            if not line:
                raise self._died("died")
            reply: dict[str, Any] = json.loads(line)
            if "call" in reply and "id" not in reply:
                self._answer(reply)
                continue
            if "progress" in reply:  # an event of the message in flight
                if reply.get("id") == self._ids and grant is not None:
                    if grant.progress is not None:
                        grant.progress(reply["progress"])
                continue
            return reply

    def _answer(self, call: dict[str, Any]) -> None:
        """Answer a subprogram call of the worker's program, in a thread of its own."""

        def answer() -> None:
            line = self._calls.answer(call, key="call")
            try:
                with self._write_lock:
                    self._in.write((line + "\n").encode("utf-8"))
                    self._in.flush()
            except (OSError, ValueError):  # the worker is gone
                pass

        threading.Thread(target=answer, daemon=True, name="qccompute-call").start()

    def _dump(self, value: Any) -> Any:
        return _wire.dump(value, self._dir)

    def _load(self, cls: type[M], data: Any) -> M:
        return _wire.load(cls, data, self._dir)

    @staticmethod
    def _ctx(ctx: RunContext | None) -> dict[str, Any]:
        if ctx is None:
            return {}
        wire: dict[str, Any] = {
            "threads": ctx.threads,
            "memory_gib": ctx.memory_gib,
            "device_memory_gib": ctx.device_memory_gib,
            "timeout": ctx.timeout,
        }
        if ctx.gpus:
            wire["gpus"] = list(ctx.gpus)
        if ctx.progress is not None:
            wire["progress"] = True
        return {k: v for k, v in wire.items() if v is not None}

    def compute(
        self, inp: ProgramInput | FileInput, ctx: RunContext | None = None
    ) -> ProgramOutput[Any, Data]:
        """Run one calculation in the worker.

        Args:
            inp: The input.
            ctx: The grant; its ``timeout`` is enforced by killing the worker, and the
                program's subprograms run within it.

        Returns:
            The successful output.

        Raises:
            ProgramError: The calculation failed (``output`` is the failed output),
                timed out, or the worker died.
        """
        timeout = ctx.timeout if ctx is not None else None
        result = self._ask(
            "compute", timeout, ctx, input=self._dump(inp), ctx=self._ctx(ctx)
        )
        out: ProgramOutput[Any, Data] = self._load(ProgramOutput, result)
        if not out.success:
            error = out.error or ErrorInfo(kind=ErrorKind.internal, message="failed")
            raise ProgramError(error, output=out)
        return out

    def compute_batch(
        self, inputs: Sequence[ProgramInput | FileInput], ctx: RunContext | None = None
    ) -> list[ProgramOutput[Any, Data]]:
        """Run many calculations together in the worker; failures in place.

        Raises:
            ProgramError: The worker died or timed out.
        """
        timeout = ctx.timeout if ctx is not None else None
        result = self._ask(
            "compute_batch",
            timeout,
            ctx,
            inputs=[self._dump(i) for i in inputs],
            ctx=self._ctx(ctx),
        )
        return [self._load(ProgramOutput, r) for r in result]

    def plan(self, inp: ProgramInput) -> Plan:
        """The program's plan of an input.

        Raises:
            ProgramError: The program refuses the input.
        """
        return Plan.model_validate(self._ask("plan", input=self._dump(inp)))

    def session(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None = None,
        ctx: RunContext | None = None,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> WorkerSession | ComputeSession:
        """Open a session in the worker: the program's own, else one over compute.

        Args:
            inp: The program, model, keywords, structure, and the default calctype.
            calctypes: The calctypes the session may run.
            ctx: The session's grant.
            on_close: Called when the session closes (a runner returns the worker
                to its pool).

        Returns:
            The session.

        Raises:
            ProgramError: The program refuses the input, or the worker died.
        """
        if not self.capabilities.sessions:
            opened = _ClosingComputeSession(
                lambda i, raise_exc=True: self._compute_any(i, ctx, raise_exc),
                inp,
                calctypes,
                program_version=self.version(),
            )
            opened.on_close = on_close
            return opened
        reply = self._ask(
            "open",
            None,
            ctx,
            input=self._dump(inp),
            calctypes=sorted(CalcType(c).value for c in calctypes)
            if calctypes
            else None,
            ctx=self._ctx(ctx),
        )
        version = str(reply.get("program_version") or self.version())
        return WorkerSession(self, reply["session"], inp, version, on_close, ctx)

    def _compute_any(self, inputs: Any, ctx: RunContext | None, raise_exc: bool) -> Any:
        """Compute for a session over compute: one input raises, a list does not."""
        if isinstance(inputs, ProgramInput):
            return self.compute(inputs, ctx)
        outputs = []
        for inp in inputs:
            try:
                outputs.append(self.compute(inp, ctx))
            except ProgramError as e:
                assert e.output is not None
                outputs.append(e.output)
        return outputs

    def close(self) -> None:
        """Stop the worker: end of input closes its sessions and it exits.

        A worker with a message in flight (a calculation running in another thread)
        is killed with every process it started, rather than waited for.
        """
        if self._lock.locked():
            self._kill()
        if self._proc.poll() is None:
            try:
                self._in.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._kill()
                self._proc.wait()
        self._dead = self._dead or ProgramError.of(
            "environment", f"The worker for {self.spec} is closed.", code="worker_died"
        )
        self._calls.close_all()  # sessions its program opened here
        if self._proc.poll() is not None and self._proc.stdout is not None:
            self._proc.stdout.close()
        for directory in (self._dir, self._stops):
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)

    def __enter__(self) -> Worker:
        """Use the worker as a context manager that stops it."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the worker."""
        self.close()

    def __del__(self) -> None:
        """Do not leave a worker process or its array directory behind."""
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            try:
                _kill_tree(proc.pid)
            except Exception:  # noqa: BLE001 - at interpreter exit: at least the worker
                proc.kill()
        directory = getattr(self, "_dir", None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)


class _Parent:
    """This process's runner, as the program that answers a worker's subprogram calls.

    A `qccompute.serve.Host` answers the calls with it. ``ctx`` is the grant of the
    message in flight: the worker program's subprograms run within it, exactly as
    they would for the program in this process.
    """

    name = "subprograms"

    def __init__(self) -> None:
        self.ctx: RunContext | None = None

    def version(self) -> str:
        return ""

    def capabilities(self) -> Capabilities:
        return Capabilities(calctypes=frozenset(CalcType), sessions=True, batches=True)

    def settings_schema(self, calctype: CalcType) -> dict[str, Any]:
        return {}

    def record_schema(self, calctype: CalcType) -> dict[str, Any]:
        return {}

    def plan(self, inp: ProgramInput) -> Plan:
        from .runner import _not_found

        program = find_program(inp.program)
        if program is None:
            raise _not_found(inp)
        return plan_of(program, inp)

    def _grant(self, inputs: Sequence[ProgramInput] = ()) -> RunContext:
        """The grant of the calculation in flight, once it may run these inputs.

        A program it does not name is refused as the call itself (``ok: false``), as
        `RunContext.compute` refuses it in process: not a subprogram's failure.
        """
        if self.ctx is None:
            raise ProgramError.of(
                "internal", "a subprogram call came with no calculation in flight"
            )
        try:
            self.ctx._check_named(inputs)
        except ProgramError as e:
            raise MessageError.of(e) from None
        return self.ctx

    def compute(
        self, inp: ProgramInput, ctx: RunContext | None = None
    ) -> ProgramOutput[Any, Data]:
        return self._grant([inp]).compute(inp, raise_exc=False)

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext | None = None
    ) -> list[ProgramOutput[Any, Data]]:
        return self._grant(inputs).compute(list(inputs), raise_exc=False)

    def open_session(
        self,
        inp: ProgramInput,
        ctx: RunContext | None = None,
        calctypes: frozenset[CalcType] | None = None,
    ) -> Session:
        threads = None if ctx is None else ctx.threads
        return self._grant().session(inp, threads=threads, calctypes=calctypes)


_REQUEST_DEFAULTS = Request().model_dump(mode="json")


def _request_fields(request: Request) -> dict[str, Any]:
    """A request's fields for the wire, leaving out those at their defaults."""
    return {
        k: v
        for k, v in request.model_dump(mode="json").items()
        if v != _REQUEST_DEFAULTS[k]
    }


def _lost(result: Any) -> bool:
    """Whether a reply reports a lost GPU context (the worker then exits)."""
    items = result if isinstance(result, list) else [result]
    return any(
        isinstance(r, dict)
        and isinstance(r.get("error"), dict)
        and r["error"].get("code") == "device_lost"
        for r in items
    )


class _ClosingComputeSession(ComputeSession):
    """A session over a worker's compute that tells its runner when it closes."""

    on_close: Callable[[], None] | None = None

    def close(self) -> None:
        """Close the session and release its worker."""
        if not self._closed and self.on_close is not None:
            self.on_close()
        super().close()


class _Peer(Protocol):
    """The other end of a session's pipe: a `Worker`, or (in a worker) its client."""

    @property
    def alive(self) -> bool: ...

    def _ask(
        self,
        op: str,
        timeout: float | None = None,
        grant: RunContext | None = None,
        **fields: Any,
    ) -> Any: ...

    def _load(self, cls: type[M], data: Any) -> M: ...


class WorkerSession:
    """A session in another process; see `qccompute.program.Session`.

    A program's own session in a worker, or, inside a worker, a subprogram's session
    in its client. Positions and results cross the pipe as shortest round-trip
    decimals (large arrays as ``.npy`` files), so they are bit for bit those of the
    session in process.
    """

    def __init__(
        self,
        worker: _Peer,
        key: str,
        inp: ProgramInput,
        program_version: str = "",
        on_close: Callable[[], None] | None = None,
        grant: RunContext | None = None,
    ) -> None:
        """Wrap an open session of a worker."""
        self._worker = worker
        self._key = key
        self._inp = inp
        self._on_close = on_close
        self._grant = grant
        self._closed = False
        self.program: str = inp.program
        self.program_version: str = program_version

    @property
    def input_data(self) -> ProgramInput:
        """The input the session was opened with."""
        return self._inp

    def _ask(self, op: str, **fields: Any) -> Any:
        return self._worker._ask(op, None, self._grant, session=self._key, **fields)

    def compute(
        self,
        positions: np.ndarray | None = None,
        calctype: CalcType | str | None = None,
        *,
        accuracy: Accuracy | float | None = None,
        lineage: Lineage = None,
        outputs: Iterable[str] = (),
        seed: Iterable[Lineage] = (),
        commit: bool = True,
    ) -> Frame:
        """Evaluate one geometry; see `qccompute.program.Session.compute`.

        Raises:
            ProgramError: The evaluation failed (with its failed frame), or the worker
                died.
        """
        request = Request(
            positions=positions,
            calctype=None if calctype is None else CalcType(calctype),
            accuracy=accuracy,
            lineage=lineage,
            outputs=frozenset(outputs),
            seed=list(seed),
            commit=commit,
        )
        if request.positions is not None and not np.all(np.isfinite(request.positions)):
            raise ProgramError.of(
                "numerical", "positions are not finite", code="non_finite"
            )
        frame = self._frame(self._ask("compute", **_request_fields(request)), request)
        if not frame.success:
            error = frame.error or ErrorInfo(kind=ErrorKind.internal, message="failed")
            raise ProgramError(error, frame=frame)
        return frame

    def _frame(self, result: dict[str, Any], request: Request) -> Frame:
        """A frame from the wire; a non-finite result's failed frame lacks a structure."""
        if "structure" not in result:
            structure = self._inp.require_structure()
            if request.positions is not None:
                structure = structure.model_copy(update={"geometry": request.positions})
            result = {**result, "structure": structure}
        return self._worker._load(Frame, result)

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """Evaluate many geometries; see `qccompute.program.Session.compute_many`.

        Requests with non-finite positions fail here (the wire carries no NaN); the
        others run in the worker.

        Raises:
            ProgramErrors: Some requests failed; ``results`` holds every frame.
        """
        frames: list[Frame | None] = [None] * len(requests)
        sent: list[int] = []
        for i, r in enumerate(requests):
            if r.positions is not None and not np.all(np.isfinite(r.positions)):
                structure = self._inp.require_structure().model_copy(
                    update={"geometry": r.positions}
                )
                error = ProgramError.of(
                    "numerical", "positions are not finite", code="non_finite"
                )
                frames[i] = failed_frame(structure, error, self.program)
            else:
                sent.append(i)
        if sent:
            result = self._ask(
                "compute_many", requests=[_request_fields(requests[i]) for i in sent]
            )
            for i, f in zip(sent, result, strict=True):
                frames[i] = self._frame(f, requests[i])
        return ProgramErrors.check([f for f in frames if f is not None])

    def reset(self, lineage: Lineage = ...) -> None:
        """Forget one lineage's history (None: the default one); omitted: all."""
        if lineage is ...:
            self._ask("reset")
        else:
            self._ask("reset", lineage=lineage)

    def plan(self) -> Plan:
        """The session's pinned plan."""
        return Plan.model_validate(self._ask("plan"))

    def close(self) -> None:
        """Close the session in the worker."""
        if self._closed:
            return
        self._closed = True
        if self._worker.alive:
            try:
                self._ask("close")
            except ProgramError:
                pass
        if self._on_close is not None:
            self._on_close()

    def __enter__(self) -> WorkerSession:
        """Use the session as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session."""
        self.close()
