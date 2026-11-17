"""The client of a worker process: a program hosted by ``python -m qccompute.serve``.

A `Worker` starts the generic worker for one program and talks to it over its pipes,
one JSON line per message (``qccompute.serve`` documents the wire). It offers what the
program offers in process (``compute``, ``compute_batch``, ``plan`` and sessions), so a
runner can place a program in this process or in a worker without callers noticing:
``Worker.session`` returns an object with the `qccompute.program.Session` methods.

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
import select
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import IO, Any

import numpy as np
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

from .program import (
    Capabilities,
    ComputeSession,
    Plan,
    RunContext,
    failed_frame,
)

__all__ = ["Worker", "WorkerSession", "THREAD_VARIABLES", "PATH_VARIABLE"]

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


def _plain(x: Any) -> Any:
    """``json``'s fallback for numpy values and sets."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, set | frozenset):
        return sorted(x, key=str)
    raise TypeError(f"{type(x).__name__} is not JSON serializable")


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
        """Start the worker and read its hello."""
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
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._stderr: collections.deque[str] = collections.deque(maxlen=40)
        threading.Thread(target=self._drain, daemon=True).start()
        self._lock = threading.Lock()
        self._ids = 0
        self._dead: ProgramError | None = None
        assert self._proc.stdout is not None and self._proc.stdin is not None
        self._out: IO[str] = self._proc.stdout
        self._in: IO[str] = self._proc.stdin
        line = self._out.readline()
        if not line:
            raise self._died("did not start")
        hello = json.loads(line)
        self.hello: dict[str, Any] = hello
        self.name: str = hello["program"]
        self.capabilities: Capabilities = Capabilities.model_validate(
            hello["capabilities"]
        )

    def __repr__(self) -> str:
        """Show the program and process."""
        return f"Worker({self.spec!r}, pid={self._proc.pid}, alive={self.alive})"

    def _drain(self) -> None:
        """Keep the end of the worker's stderr (its logs), so its pipe never fills."""
        assert self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr.append(line)

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

    def _ask(self, op: str, timeout: float | None = None, **fields: Any) -> Any:
        """Send one message and return its result.

        Raises:
            ProgramError: The worker refused the message (``ok: false``), died, or
                did not answer within ``timeout`` (then it is killed: kind
                ``timeout``).
        """
        with self._lock:
            if self._dead is not None:
                raise ProgramError(self._dead.error)
            self._ids += 1
            message = json.dumps(
                {"id": self._ids, "op": op, **fields},
                allow_nan=False,
                separators=(",", ":"),
                default=_plain,
            )
            try:
                self._in.write(message + "\n")
                self._in.flush()
            except (BrokenPipeError, OSError):
                raise self._died("died") from None
            if timeout is not None:
                ready, _, _ = select.select([self._out], [], [], timeout)
                if not ready:
                    self._proc.kill()
                    self._died("was killed")
                    raise ProgramError.of(
                        "timeout",
                        f"{self.name} did not finish within {timeout:g} s; its worker "
                        "was killed.",
                    )
            line = self._out.readline()
            if not line:
                raise self._died("died")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise ProgramError(ErrorInfo.model_validate(reply["error"]))
        if self._proc.poll() is not None or _lost(reply.get("result")):
            self._died("exited after losing its GPU context")
        return reply.get("result")

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
            ctx: The grant; its ``timeout`` is enforced by killing the worker.

        Returns:
            The successful output.

        Raises:
            ProgramError: The calculation failed (``output`` is the failed output),
                timed out, or the worker died.
        """
        timeout = ctx.timeout if ctx is not None else None
        result = self._ask(
            "compute",
            timeout,
            input=inp.model_dump(mode="json"),
            ctx=self._ctx(ctx),
        )
        out: ProgramOutput[Any, Data] = ProgramOutput.model_validate(result)
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
            inputs=[i.model_dump(mode="json") for i in inputs],
            ctx=self._ctx(ctx),
        )
        return [ProgramOutput.model_validate(r) for r in result]

    def plan(self, inp: ProgramInput) -> Plan:
        """The program's plan of an input.

        Raises:
            ProgramError: The program refuses the input.
        """
        return Plan.model_validate(self._ask("plan", input=inp.model_dump(mode="json")))

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
            input=inp.model_dump(mode="json"),
            calctypes=sorted(CalcType(c).value for c in calctypes)
            if calctypes
            else None,
            ctx=self._ctx(ctx),
        )
        return WorkerSession(self, reply["session"], inp, on_close)

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

    def __enter__(self) -> Worker:
        """Use the worker as a context manager that stops it."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the worker."""
        self.close()

    def __del__(self) -> None:
        """Do not leave a worker process behind."""
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()


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


class WorkerSession:
    """A program's own session in a worker process; see `qccompute.program.Session`.

    Positions and results cross the pipe as shortest round-trip decimals, so they are
    bit for bit those of the session in process.
    """

    def __init__(
        self,
        worker: Worker,
        key: str,
        inp: ProgramInput,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Wrap an open session of a worker."""
        self._worker = worker
        self._key = key
        self._inp = inp
        self._on_close = on_close
        self._closed = False
        self.program: str = worker.name
        self.program_version: str = worker.version()

    @property
    def input_data(self) -> ProgramInput:
        """The input the session was opened with."""
        return self._inp

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
        fields = _request_fields(request)
        frame = self._frame(
            self._worker._ask("compute", session=self._key, **fields), request
        )
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
        return Frame.model_validate(result)

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
            result = self._worker._ask(
                "compute_many",
                session=self._key,
                requests=[_request_fields(requests[i]) for i in sent],
            )
            for i, f in zip(sent, result, strict=True):
                frames[i] = self._frame(f, requests[i])
        return ProgramErrors.check([f for f in frames if f is not None])

    def reset(self, lineage: Lineage = ...) -> None:
        """Forget one lineage's history (None: the default one); omitted: all."""
        if lineage is ...:
            self._worker._ask("reset", session=self._key)
        else:
            self._worker._ask("reset", session=self._key, lineage=lineage)

    def plan(self) -> Plan:
        """The session's pinned plan."""
        return Plan.model_validate(self._worker._ask("plan", session=self._key))

    def close(self) -> None:
        """Close the session in the worker."""
        if self._closed:
            return
        self._closed = True
        if self._worker.alive:
            try:
                self._worker._ask("close", session=self._key)
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
