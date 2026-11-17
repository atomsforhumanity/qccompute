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
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    failed_frame,
    find_program,
    plan_of,
)
from .serve import Host

__all__ = ["Worker", "WorkerSession", "THREAD_VARIABLES", "PATH_VARIABLE"]

M = TypeVar("M", bound=BaseModel)

#: The client's ``sys.path`` for the worker, which takes it out of its environment.
PATH_VARIABLE = "QCCOMPUTE_WORKER_SYS_PATH"

#: Thread counts of numerical libraries, set for a worker to its grant unless the
#: environment already sets them.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class Worker:
    """One program in a worker process of its own.

    Args:
        program: The program's name (an installed program) or ``module:attribute``.
        threads: CPU threads of the process; numerical libraries are limited to
            these. None: not limited.
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
        for line in self._proc.stderr:
            self._stderr.append(line.decode("utf-8", "replace"))

    @property
    def alive(self) -> bool:
        """Whether the worker can take work."""
        return self._dead is None and self._proc.poll() is None

    def version(self) -> str:
        """The program's version, from the worker's hello."""
        return str(self.hello["version"])

    def _died(self, what: str) -> ProgramError:
        """The failure of a worker that exited, with its status and stderr's end."""
        try:
            status = self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
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
        each call is answered in a thread of its own, within ``grant``.

        Raises:
            ProgramError: The worker refused the message (``ok: false``), died, or
                did not answer within ``timeout`` (then it is killed: kind
                ``timeout``).
        """
        with self._lock:
            if self._dead is not None:
                raise ProgramError(self._dead.error)
            self._ids += 1
            self._parent.ctx = grant
            try:
                self._send({"id": self._ids, "op": op, **fields})
            except (BrokenPipeError, OSError):
                raise self._died("died") from None
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                left = None if deadline is None else deadline - time.monotonic()
                line = self._lines.readline(None if left is None else max(0.0, left))
                if line is None:
                    self._proc.kill()
                    self._died("was killed")
                    raise ProgramError.of(
                        "timeout",
                        f"{self.name} did not finish within {timeout:g} s; its worker "
                        "was killed.",
                    )
                if not line:
                    raise self._died("died")
                reply = json.loads(line)
                if "call" in reply and "id" not in reply:
                    self._answer(reply)
                    continue
                break
        if not reply.get("ok"):
            raise ProgramError(ErrorInfo.model_validate(reply["error"]))
        if self._proc.poll() is not None or _lost(reply.get("result")):
            self._died("exited after losing its GPU context")
        return reply.get("result")

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
        """Stop the worker: end of input closes its sessions and it exits."""
        if self._proc.poll() is None:
            try:
                self._in.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._dead = self._dead or ProgramError.of(
            "environment", f"The worker for {self.spec} is closed.", code="worker_died"
        )
        self._calls.close_all()  # sessions its program opened here
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)

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

    def _grant(self) -> RunContext:
        if self.ctx is None:
            raise ProgramError.of(
                "internal", "a subprogram call came with no calculation in flight"
            )
        return self.ctx

    def compute(
        self, inp: ProgramInput, ctx: RunContext | None = None
    ) -> ProgramOutput[Any, Data]:
        return self._grant().compute(inp, raise_exc=False)

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext | None = None
    ) -> list[ProgramOutput[Any, Data]]:
        return self._grant().compute(list(inputs), raise_exc=False)

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
