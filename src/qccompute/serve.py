"""The worker: one loop that hosts any program's sessions in its own process.

    python -m qccompute.serve mychem                  # stdin/stdout, one client
    python -m qccompute.serve mychem --socket PATH    # a Unix socket, one client at a time

A worker keeps a program alive between calls (warm starts, a live GPU context) and
isolates it: a crash ends the worker, not its caller. The program is found by name
(`qccompute.program.find_program`: registered in this process, else the
``qccompute.programs`` entry points), given as ``module:attribute``, or as
``pickle:PATH``, a file holding it pickled (a runner's, for a program registered in
its process that no module holds). The loop uses only
qcdata and the protocol, and knows nothing of the program beyond it. Design:
``docs/design/program-interface.md``, "Workers".

Wire, one JSON message per line:

- The first line is the hello: ``{"protocol": 1, "program", "version", "capabilities",
  "array_files": true, "calls": true}``. ``array_files`` and ``calls`` say what the
  ``options`` message may turn on.
- Every message carries an ``id``; replies are ``{"id", "ok": true, "result"}`` or
  ``{"id", "ok": false, "error": ErrorInfo}``. ``ok: false`` is for the message itself
  (bad JSON, an unknown op or session, a failed open or plan); a failed calculation is a
  result: a failed Frame or ProgramOutput.
- Ops: ``open {input, calctypes, ctx}`` -> ``{"session", "program_version"}``; ``compute
  {session, positions, calctype, accuracy, lineage, outputs, seed, commit}`` -> Frame;
  ``compute_many {session, requests}`` -> [Frame]; ``reset {session, lineage}`` (absent:
  every lineage; null: the default one); ``plan {session}``, ``{session, calctype}`` or
  ``{input}``; ``close {session}``; ``compute {input, ctx}`` (a one-off) ->
  ProgramOutput; ``compute_batch {inputs, ctx}`` -> [ProgramOutput]; ``lineages
  {session}`` where the session offers them; ``options {array_dir, calls}``.
- ``ctx`` is ``{threads, memory_gib, device_memory_gib, timeout, scratch_dir, gpus,
  progress, stop_file, identities}``. The GPU is fixed when the worker is spawned, so ``gpus`` must
  be empty or match ``CUDA_VISIBLE_DEVICES``. ``stop_file`` names a file whose
  existence asks the calculation to stop at its next step (``ctx.stop_requested()``):
  the client creates it while the message is in flight, and removes it after. While
  the message runs, the worker holds its BLAS and OpenMP pools at ``threads``. With
  ``progress: true`` the program's progress events (``ctx.progress``), each with its
  ``path`` from the worker's program down (``[]`` for its own; a subprogram's name for
  one the worker ran itself), stream to the client while a message is in flight, as lines
  ``{"id", "progress": event}`` naming the message (for a session, each request's
  message); an event that is not JSON is dropped, with a note on stderr. A plain
  client never asks for them and never gets one.
- Numbers are shortest round-trip decimals, so positions and results cross bit for bit.
  No NaN or Infinity: a message holding one is refused, and a non-finite result becomes a
  failed frame (kind ``numerical``, code ``non_finite``).
- ``options`` sets what this connection uses; a plain JSON-lines client never sends it
  and gets neither:

  - ``array_dir``: a directory both sides can read and write (workers are local). An
    array field over 1 MiB, in a message either way, crosses as a ``.npy`` file there,
    with qcdata's pointer ``{path, shape, dtype, sha256}`` in its place (the format of a
    saved model's arrays); the reader checks it and deletes the file.
  - ``calls: true``: the client runs the subprograms of this worker's program (its
    ``ctx.compute`` and ``ctx.session``) that the worker cannot run exactly as the
    client would, so they see the programs the client sees, registered ones included,
    and share its grant. The worker runs a call itself when every program its input
    names is found here with the ``identities`` the client gave in the message's
    ``ctx``, may run in process, and its plan uses no GPU. While a message is in
    flight the worker may write call lines, ``{"call", "op", ...}`` with the ops above (``compute
    {input}``, ``compute_batch {inputs}``, ``open {input, calctypes, ctx: {threads}}``
    and the session ops); the client answers each on stdin, ``{"call", "ok", "result"}``
    or ``{"call", "ok": false, "error"}``, in any order. Without it, subprograms run in
    the worker's own default runner.
- End of input closes every session and exits 0. After a reply whose error has the code
  ``device_lost`` (a GPU fault that poisoned the device context) the worker exits 3:
  nothing more can run in this process.

Hygiene: stdout carries protocol lines only. At start the worker writes the protocol to a
dup of the original stdout and points file descriptor 1 at stderr, because CUDA, BLAS and
other libraries print.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import pickle
import socket
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import IO, Any

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

from . import _wire
from ._pools import hold_threads
from .program import (
    BatchProgram,
    Program,
    RunContext,
    Session,
    SessionProgram,
    _Control,
    _cuda_visible_devices,
    _Progress,
    capabilities_of,
    failed_frame,
    find_program,
    numerics_of,
    plan_of,
)

__all__ = ["PROTOCOL", "EXIT_DEVICE_LOST", "Host", "serve", "main"]

#: The wire protocol's version, in the hello line.
PROTOCOL = 1
#: The exit status after a ``device_lost`` reply.
EXIT_DEVICE_LOST = 3
_OPS = (
    "open, compute, compute_batch, compute_many, reset, plan, close, lineages, options"
)


class MessageError(Exception):
    """A message the worker cannot act on: an ``ok: false`` reply."""

    def __init__(
        self, message: str, kind: ErrorKind = ErrorKind.input, code: str | None = None
    ):
        """Create the error.

        Args:
            message: What is wrong with the message.
            kind: The error kind.
            code: A code refining the kind.
        """
        super().__init__(message)
        self.error = ErrorInfo(kind=kind, message=message, code=code)

    @classmethod
    def of(cls, e: ProgramError) -> MessageError:
        """A program's refusal (of an open or a plan) as a message error."""
        return cls(e.error.message, e.error.kind, e.error.code)


def load_program(spec: str) -> Program:
    """The program a spec names.

    A name (`find_program`), the object at ``module:attribute``, or ``pickle:PATH``:
    a file holding the program pickled, which a runner writes for a program
    registered in its process that no module holds (`qccompute.runner._worker_spec`).

    Raises:
        SystemExit: No such program.
    """
    if spec.startswith("pickle:"):
        with open(spec.removeprefix("pickle:"), "rb") as f:
            loaded: Program = pickle.load(f)
        return loaded
    if ":" in spec:
        module, attr = spec.split(":", 1)
        program: Program = getattr(importlib.import_module(module), attr)
        return program
    found = find_program(spec)
    if found is None:
        raise SystemExit(
            f"qccompute.serve: no program {spec!r} (registered or in the "
            "qccompute.programs entry points)"
        )
    return found


def _finite(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False, default=_wire.plain)
    except ValueError:
        return False
    return True


def _non_finite(result: Any) -> Any:
    """A result whose non-finite numbers made it unsendable, as failed frames."""
    if isinstance(result, list):
        return [_non_finite(r) for r in result]
    if not isinstance(result, dict) or _finite(result):
        return result
    error = ErrorInfo(
        kind=ErrorKind.numerical,
        code="non_finite",
        message="the result has a non-finite number (NaN or Infinity), which the "
        "wire does not carry",
    )
    failed: dict[str, Any] = {
        k: result[k]
        for k in ("structure", "input_data", "warnings", "record")
        if k in result and _finite(result[k])
    }
    provenance = (result.get("results") or {}).get("provenance")
    failed.update(
        success=False,
        error=error.model_dump(mode="json"),
        traceback=error.message,
        results={"provenance": provenance} if provenance else {},
    )
    return failed


def _no_constant(name: str) -> Any:
    raise MessageError(f"{name} is not a number the wire carries (no NaN or Infinity)")


def _request(fields: Any) -> Request:
    """A request from a message's fields; a MessageError when they are not one."""
    try:
        return Request.model_validate(fields)
    except Exception as e:  # noqa: BLE001 - pydantic's ValidationError
        raise MessageError(f"invalid request: {e}") from None


class Host:
    """The sessions of one program in this process, and the ops on them.

    Args:
        program: The program the worker hosts.
        array_dir: Where array files cross (the ``options`` message sets it per
            connection); None: arrays cross inline.

    Attributes:
        connection: The client being served, through which the program's subprograms
            run when it asked for ``calls``.
    """

    def __init__(self, program: Program, array_dir: Path | None = None):
        """Host a program."""
        self.program = program
        self.array_dir = array_dir
        self.connection: _Connection | None = None
        self.sessions: dict[str, Session] = {}
        self._threads: dict[str, int | None] = {}  # each session's grant
        self._ids = itertools.count(1)
        self._subprograms = _ClientRunner(self)
        self.device_lost = False
        self._message: Any = None  # the id of the message in flight

    def hello(self) -> dict[str, Any]:
        """The worker's first line: protocol, program, version, capabilities, options."""
        return {
            "protocol": PROTOCOL,
            "program": self.program.name,
            "version": self.program.version(),
            "capabilities": capabilities_of(self.program).model_dump(mode="json"),
            "array_files": True,
            "calls": True,
        }

    def handle(self, msg: dict[str, Any]) -> Any:
        """The result of one message: models, lists of them, or plain values.

        Raises:
            MessageError: The message cannot be acted on.
        """
        op = msg.get("op")
        fn: Callable[[dict[str, Any]], Any] | None = (
            getattr(self, f"op_{op}", None) if isinstance(op, str) else None
        )
        if fn is None:
            raise MessageError(f"unknown op {op!r}; ops: {_OPS}")
        return fn(msg)

    def answer(self, msg: dict[str, Any], key: str = "id") -> str:
        """The reply line to one parsed message, under ``key`` (``id``, or ``call``)."""
        mid = msg.get(key)
        if key == "id":
            self._message = mid
        try:
            result = _wire.dump(self.handle(msg), self.array_dir)
            try:
                return _wire.encode({key: mid, "ok": True, "result": result})
            except ValueError:
                return _wire.encode(
                    {key: mid, "ok": True, "result": _non_finite(result)}
                )
        except (MessageError, ProgramError) as e:  # ProgramError: a closed session
            error = e.error
        except Exception as e:  # noqa: BLE001 - a bug in the loop or a program: keep serving
            traceback.print_exc(file=sys.stderr)
            error = ErrorInfo(
                kind=ErrorKind.internal, message=f"{type(e).__name__}: {e}"
            )
        return _wire.encode(
            {key: mid, "ok": False, "error": error.model_dump(mode="json")}
        )

    def op_options(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Set this connection's array directory and calls; the options in effect."""
        directory = msg.get("array_dir")
        if directory is not None and not Path(directory).is_dir():
            raise MessageError(f"array_dir {directory!r} is not a directory")
        self.array_dir = None if directory is None else Path(directory)
        if self.connection is not None:
            self.connection.calls = bool(msg.get("calls"))
        return {
            "array_dir": directory,
            "calls": self.connection is not None and self.connection.calls,
        }

    def op_open(self, msg: dict[str, Any]) -> dict[str, str]:
        """Open a session; its id and the program's version."""
        if not isinstance(self.program, SessionProgram):
            raise MessageError(
                f"{self.program.name} has no sessions", ErrorKind.unsupported
            )
        inp = self._input(msg)
        if not isinstance(inp, ProgramInput):
            raise MessageError("a session opens from a ProgramInput")
        calctypes = msg.get("calctypes")
        ctx = self._ctx(msg.get("ctx"))
        ctx._calling = inp
        try:
            with hold_threads(ctx.threads):
                session = self.program.open_session(
                    inp,
                    ctx,
                    frozenset(CalcType(c) for c in calctypes) if calctypes else None,
                )
        except ProgramError as e:
            raise MessageError.of(e) from None
        key = f"s{next(self._ids)}"
        self.sessions[key] = session
        self._threads[key] = ctx.threads
        return {"session": key, "program_version": session.program_version}

    def op_compute(self, msg: dict[str, Any]) -> Any:
        """A session's frame, or a one-off output for ``input``."""
        if "input" in msg:
            out = self._compute(self._input(msg), self._ctx(msg.get("ctx")))
            self._watch(out)
            return out
        session = self._session(msg)
        fields = ("positions", "calctype", "accuracy", "lineage", "outputs", "seed")
        r = _request({k: msg[k] for k in (*fields, "commit") if k in msg})
        try:
            with hold_threads(self._threads.get(msg["session"])):
                frame = session.compute(
                    r.positions,
                    r.calctype,
                    accuracy=r.accuracy,
                    lineage=r.lineage,
                    outputs=r.outputs,
                    seed=r.seed,
                    commit=r.commit,
                )
        except ProgramError as e:
            frame = e.frame or failed_frame(
                self._structure(session, r), e, self.program.name
            )
        self._watch(frame)
        return frame

    def op_compute_batch(self, msg: dict[str, Any]) -> Any:
        """Outputs of many one-off inputs, computed together; failures in place."""
        inputs = [self._input({"input": i}) for i in msg.get("inputs") or []]
        ctx = self._ctx(msg.get("ctx"))
        ctx._calling = inputs
        if isinstance(self.program, BatchProgram):
            try:
                with hold_threads(ctx.threads):
                    outs = list(self.program.compute_batch(inputs, ctx))  # type: ignore[arg-type]
            except MessageError:  # the message refused (a client's subprogram call)
                raise
            except Exception as e:  # noqa: BLE001 - a bug: every input fails with it
                outs = [_failed(i, e, self.program.name) for i in inputs]
        else:
            outs = [self._compute(i, ctx) for i in inputs]
        if ctx._refusal is not None:  # it ran a program no input names: all fail
            outs = [
                _refused(i, ctx._refusal, o) for i, o in zip(inputs, outs, strict=True)
            ]
        outs = [_with_devices(out, ctx.gpus) for out in outs]
        for out in outs:
            self._watch(out)
        return outs

    def _compute(
        self, inp: ProgramInput | FileInput, ctx: RunContext
    ) -> ProgramOutput[Any, Any]:
        """One output, failed or not: a ProgramError or a bug becomes a failed output.

        As in a runner, a call of a program the input does not name fails it.
        """
        if ctx._calling is None:
            ctx._calling = inp
        try:
            with hold_threads(ctx.threads):
                out = self.program.compute(inp, ctx)  # type: ignore[arg-type]
        except MessageError:  # the message refused (a client's subprogram call)
            raise
        except Exception as e:  # noqa: BLE001 - failures are results on the wire
            out = _failed(inp, e, self.program.name)
        finally:
            ctx.cleanup()
        if ctx._refusal is not None:
            out = _refused(inp, ctx._refusal, out)
        return _with_devices(out, ctx.gpus)

    def op_compute_many(self, msg: dict[str, Any]) -> Any:
        """A session's frames for many requests, in order."""
        session = self._session(msg)
        frames: list[Frame]
        requests = [_request(r) for r in msg.get("requests") or []]
        try:
            with hold_threads(self._threads.get(msg["session"])):
                frames = session.compute_many(requests)
        except ProgramErrors as group:  # every frame, in order
            frames = group.results
        for frame in frames:
            self._watch(frame)
        return list(frames)

    def op_reset(self, msg: dict[str, Any]) -> None:
        """Forget one lineage (null: the default one), or every lineage when absent."""
        session = self._session(msg)
        if "lineage" in msg:
            session.reset(msg["lineage"])
        else:
            session.reset()

    def op_lineages(self, msg: dict[str, Any]) -> Any:
        """The session's lineages with history, where the session offers them."""
        lineages = getattr(self._session(msg), "lineages", None)
        if lineages is None:
            raise MessageError(
                "this session does not list lineages", ErrorKind.unsupported
            )
        return list(lineages())

    def op_plan(self, msg: dict[str, Any]) -> Any:
        """A session's plan, its input's plan at another calctype, or an input's plan."""
        try:
            if "input" in msg:
                return plan_of(self.program, self._input(msg))  # type: ignore[arg-type]
            session = self._session(msg)
            if msg.get("calctype"):
                try:
                    calctype = CalcType(msg["calctype"])
                except ValueError as e:
                    raise MessageError(str(e)) from None
                inp = session.input_data.model_copy(update={"calctype": calctype})
                return plan_of(self.program, inp)
            return session.plan()
        except ProgramError as e:
            raise MessageError.of(e) from None

    def op_close(self, msg: dict[str, Any]) -> None:
        """Close a session."""
        self._session(msg)
        self._threads.pop(msg["session"], None)
        self.sessions.pop(msg["session"]).close()

    def close_all(self) -> None:
        """Close every session, reporting (not raising) failures."""
        for session in list(self.sessions.values()):
            try:
                session.close()
            except Exception as e:  # noqa: BLE001 - closing at exit: report, go on
                print(f"qccompute.serve: closing a session: {e}", file=sys.stderr)
        self.sessions.clear()
        self._threads.clear()

    def _input(self, msg: dict[str, Any]) -> ProgramInput | FileInput:
        try:
            data = msg["input"]
            if isinstance(data, dict) and "calctype" not in data:
                return _wire.load(FileInput, data, self.array_dir)
            return _wire.load(ProgramInput, data, self.array_dir)
        except Exception as e:  # noqa: BLE001 - a ValidationError or a missing key
            raise MessageError(f"invalid input: {e}") from None

    def _session(self, msg: dict[str, Any]) -> Session:
        key = msg.get("session")
        if not isinstance(key, str) or key not in self.sessions:
            raise MessageError(f"unknown session {key!r}", code="unknown_session")
        return self.sessions[key]

    @staticmethod
    def _structure(session: Session, r: Request) -> Any:
        """The structure a request was for, to build its failed frame."""
        structure = session.input_data.structure
        if structure is None or r.positions is None:
            return structure
        return structure.model_copy(update={"geometry": r.positions})

    def _ctx(self, ctx: dict[str, Any] | None) -> RunContext:
        fields = dict(ctx or {})
        gpus = tuple(str(g) for g in fields.pop("gpus", None) or ())
        visible = tuple(_cuda_visible_devices() or ())
        if gpus and gpus != visible:
            raise MessageError(
                f"this worker's GPUs are {list(visible)}, fixed at its start; ctx asks "
                f"for {list(gpus)}",
                ErrorKind.environment,
            )
        known = {
            "threads",
            "memory_gib",
            "device_memory_gib",
            "timeout",
            "scratch_dir",
            "progress",
            "stop_file",
            "identities",
        }
        unknown = sorted(set(fields) - known)
        if unknown:
            raise MessageError(
                f"unknown ctx fields {unknown}; ctx has {', '.join(sorted(known))}, gpus"
            )
        scratch = fields.get("scratch_dir")
        made = RunContext(
            threads=fields.get("threads"),
            memory_gib=fields.get("memory_gib"),
            device_memory_gib=fields.get("device_memory_gib"),
            timeout=fields.get("timeout"),
            gpus=visible,
            scratch=Path(scratch) if scratch else None,
            runner=self._subprograms,
            progress=_Progress(self._progress, ()) if fields.get("progress") else None,
        )
        stop_file = fields.get("stop_file")
        if isinstance(stop_file, str):
            made._control = _Control(stop_file=stop_file)
        identities = fields.get("identities")
        if isinstance(identities, dict):
            made._identities = {str(k): str(v) for k, v in identities.items()}
        return made

    def _progress(self, event: Any) -> None:
        """Stream a progress event to the client, naming the message in flight."""
        connection = self.connection
        if connection is None:
            return
        try:
            line = _wire.encode({"id": self._message, "progress": event})
        except (TypeError, ValueError) as e:
            print(f"qccompute.serve: a progress event dropped: {e}", file=sys.stderr)
            return
        try:
            connection.write(line)
        except (OSError, ValueError):  # the client is gone
            pass

    def _watch(self, out: Frame | ProgramOutput[Any, Any]) -> None:
        if (
            not out.success
            and out.error is not None
            and out.error.code == "device_lost"
        ):
            self.device_lost = True


def _with_devices(
    out: ProgramOutput[Any, Any], gpus: tuple[str, ...]
) -> ProgramOutput[Any, Any]:
    """The output with its GPUs' names in ``execution.devices``, if the program left it unset.

    As the runner does for a program in its own process; a worker does it before the
    output crosses the wire, where every field arrives set. A program's explicit
    ``devices=[]`` (it ran on no device) stays.
    """
    if gpus and "devices" not in out.execution.model_fields_set:
        from .runner import _device_names

        execution = out.execution.model_copy(update={"devices": _device_names(gpus)})
        out.__dict__["execution"] = execution
    return out


def _refused(
    inp: ProgramInput | FileInput, refusal: ProgramError, out: ProgramOutput[Any, Any]
) -> ProgramOutput[Any, Any]:
    """The output of a calculation that called a program its input does not name."""
    from .runner import _unnamed

    return _unnamed(inp, refusal, out)[0]


def _failed(
    inp: ProgramInput | FileInput, e: Exception, program: str
) -> ProgramOutput[Any, Any]:
    """The failed output of a one-off compute that raised.

    As in the runner: a subprogram's failure the program let escape is kind
    ``program`` with the subprogram's error as its cause; any other exception than a
    `ProgramError` is a bug, kind ``internal``.
    """
    from .runner import _as_failure

    if not isinstance(e, ProgramError | ExceptionGroup):
        traceback.print_exc(file=sys.stderr)
    return _as_failure(inp, e, program)[0]


class _Connection:
    """One client of the worker: its message lines, and calls to it for subprograms.

    The serve loop reads messages; while one is handled, the threads of the program
    that make calls read the client's answers, one thread at a time, and hand each to
    the thread waiting for it.
    """

    def __init__(self, lines: Iterator[str], write: Callable[[str], None]) -> None:
        self._lines = lines
        self._write = write
        self.calls = False
        self._write_lock = threading.Lock()
        self._read = threading.Condition()
        self._reading = False
        self._eof = False
        self._answers: dict[int, dict[str, Any]] = {}
        self._backlog: deque[str] = deque()
        self._ids = itertools.count(1)

    def write(self, line: str) -> None:
        """Write one line to the client."""
        with self._write_lock:
            self._write(line)

    def messages(self) -> Iterator[str]:
        """The client's message lines, for the serve loop."""
        while True:
            with self._read:
                line = self._backlog.popleft() if self._backlog else None
                if line is None and self._eof:
                    return
            if line is None:
                line = next(self._lines, None)
                if line is None:
                    return
            yield line

    def ask(self, op: str, **fields: Any) -> Any:
        """Send a call to the client and wait for its answer's result.

        Raises:
            ProgramError: The client refused the call, or is gone.
        """
        n = next(self._ids)
        try:
            self.write(_wire.encode({"call": n, "op": op, **fields}))
        except (OSError, ValueError):  # the client is gone
            self._eof = True
        with self._read:
            while n not in self._answers:
                if self._eof:
                    raise ProgramError.of(
                        "environment",
                        "the worker's client closed before answering a subprogram call",
                        code="worker_died",
                    )
                if self._reading:
                    self._read.wait()
                    continue
                self._reading = True
                self._read.release()
                try:
                    line = next(self._lines, None)
                finally:
                    self._read.acquire()
                    self._reading = False
                    self._read.notify_all()
                self._take(line)
            answer = self._answers.pop(n)
        if not answer.get("ok"):
            raise ProgramError(ErrorInfo.model_validate(answer.get("error")))
        return answer.get("result")

    def _take(self, line: str | None) -> None:
        """File a line read while waiting: an answer, or a message for the loop."""
        if line is None:
            self._eof = True
            return
        try:
            msg = json.loads(line)
        except ValueError:
            msg = None
        if isinstance(msg, dict) and "call" in msg and "op" not in msg:
            self._answers[msg["call"]] = msg
        elif line.strip():
            self._backlog.append(line)


class _ClientRunner:
    """A worker program's subprograms: run here when they can, else by the client.

    With ``calls``, a subprogram runs in this process's default runner, within the
    calculation's grant, when it can run here exactly as the client would run it
    (`_here`); every other runs in the client, whose runner finds it (programs
    registered there included) and runs it within its grant. Without ``calls``, they
    all run here. Either way the context's grant, time left, stop, progress and the
    check that the input names the program apply alike.
    """

    def __init__(self, host: Host) -> None:
        self._host = host

    def _connection(self) -> _Connection | None:
        connection = self._host.connection
        return connection if connection is not None and connection.calls else None

    @staticmethod
    def _here(inputs: Iterable[Any], ctx: RunContext) -> bool:
        """Whether subprogram calls run in this worker rather than in its client.

        Each input's programs, its own and its subprograms', must be found here with
        the very identities the client gave for the calculation (so they are the
        programs the client would run), may run in process, and its plan uses no GPU
        of its own. A program found here alone, one that must run in a worker of its
        own, a GPU calculation, or one its program refuses go to the client.
        """
        identities = ctx._identities
        if not identities:
            return False
        for inp in inputs:
            if not isinstance(inp, ProgramInput):
                return False
            pending: list[ProgramSpec] = [inp]
            while pending:
                spec = pending.pop()
                program = find_program(spec.program)
                if program is None or not capabilities_of(program).in_process:
                    return False
                known = identities.get(spec.program, "").split(";")
                if numerics_of(program, spec) not in known:
                    return False
                pending.extend(spec.subprograms)
            try:
                if plan_of(find_program(inp.program), inp).resources.gpus:  # type: ignore[arg-type]
                    return False
            except ProgramError:  # refused: as the client refuses it
                return False
        return True

    def _subcompute(self, inputs: Any, ctx: RunContext, raise_exc: bool) -> Any:
        from .runner import _error, _subprogram_results, default_runner

        one = isinstance(inputs, ProgramInput | FileInput)
        if not one:
            inputs = list(inputs)
        connection = self._connection()
        if connection is None or self._here([inputs] if one else inputs, ctx):
            return default_runner()._subcompute(inputs, ctx, raise_exc)
        directory = self._host.array_dir
        if one:
            wire = [connection.ask("compute", input=self._dump(inputs))]
        else:
            wire = connection.ask(
                "compute_batch", inputs=[self._dump(i) for i in inputs]
            )
        outputs = [_wire.load(ProgramOutput, r, directory) for r in wire]
        results = [(o, None if o.success else _error(o)) for o in outputs]
        return _subprogram_results(inputs, results, raise_exc)

    def _subsession(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None,
        threads: int | None,
        ctx: RunContext,
    ) -> Session:
        from .runner import _Guarded, default_runner
        from .worker import WorkerSession

        connection = self._connection()
        if connection is None or self._here([inp], ctx):
            return default_runner()._subsession(inp, calctypes, threads, ctx)
        reply = connection.ask(
            "open",
            input=self._dump(inp),
            calctypes=sorted(CalcType(c).value for c in calctypes)
            if calctypes
            else None,
            ctx={} if threads is None else {"threads": threads},
        )
        peer = _ClientPeer(connection, self._host)
        session = WorkerSession(
            peer, reply["session"], inp, str(reply.get("program_version") or "")
        )
        return _Guarded(session, None, ctx)

    def _dump(self, inp: ProgramInput | FileInput) -> Any:
        return _wire.dump(inp, self._host.array_dir)


class _ClientPeer:
    """The client, as the peer a `qccompute.worker.WorkerSession` talks to."""

    def __init__(self, connection: _Connection, host: Host) -> None:
        self._connection = connection
        self._host = host
        self.alive = True

    def _ask(
        self,
        op: str,
        timeout: float | None = None,
        grant: RunContext | None = None,
        **fields: Any,
    ) -> Any:
        return self._connection.ask(op, **fields)

    def _load(self, cls: Any, data: Any) -> Any:
        return _wire.load(cls, data, self._host.array_dir)


def _reply(host: Host, line: str) -> str | None:
    """The reply line to one message line; None for a late answer to a call."""
    try:
        msg = json.loads(line, parse_constant=_no_constant)
        if not isinstance(msg, dict):
            raise MessageError("a message is a JSON object")
    except (ValueError, MessageError) as e:
        error = (
            e.error
            if isinstance(e, MessageError)
            else ErrorInfo(kind=ErrorKind.input, message=f"bad JSON: {e}")
        )
        return _wire.encode(
            {"id": _id_of(line), "ok": False, "error": error.model_dump(mode="json")}
        )
    if "call" in msg and "op" not in msg:
        return None
    return host.answer(msg)


def _id_of(line: str) -> Any:
    """The id of a message refused while parsing (a NaN in it), if it has one."""
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    return msg.get("id") if isinstance(msg, dict) else None


def serve(host: Host, lines: Iterable[str], write: Callable[[str], None]) -> int:
    """Answer every message of ``lines`` through ``write``, after the hello.

    Each call is one connection: its ``options`` start unset.

    Args:
        host: The program's sessions.
        lines: Message lines.
        write: Writes one reply line (without its newline).

    Returns:
        The exit status: 0 at the end of input, `EXIT_DEVICE_LOST` after a
        ``device_lost`` reply.
    """
    connection = _Connection(iter(lines), write)
    host.connection = connection
    host.array_dir = None
    connection.write(_wire.encode(host.hello()))
    for line in connection.messages():
        if not line.strip():
            continue
        reply = _reply(host, line)
        if reply is None:
            continue
        connection.write(reply)
        if host.device_lost:
            return EXIT_DEVICE_LOST
    return 0


def _protocol_stdout() -> IO[str]:
    """The protocol stream (a dup of the original stdout), with fd 1 pointed at stderr."""
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def _lines(f: IO[str]) -> Iterator[str]:
    yield from iter(f.readline, "")


def _writer(f: IO[str]) -> Callable[[str], None]:
    def write(line: str) -> None:
        f.write(line + "\n")
        f.flush()

    return write


def main(argv: list[str] | None = None) -> int:
    """Run a worker; the exit status.

    Args:
        argv: Command-line arguments (default: ``sys.argv``).
    """
    parser = argparse.ArgumentParser(
        prog="python -m qccompute.serve",
        description="Host a program's sessions in this process, over JSON lines.",
    )
    parser.add_argument("program", help="a program name, or module:attribute")
    parser.add_argument(
        "--socket", help="serve a Unix socket at this path, one client at a time"
    )
    args = parser.parse_args(argv)
    paths = os.environ.pop("QCCOMPUTE_WORKER_SYS_PATH", "")
    sys.path.extend(p for p in paths.split(os.pathsep) if p and p not in sys.path)
    if args.socket is None:
        proto = _protocol_stdout()
        host = Host(load_program(args.program))
        try:
            return serve(host, _lines(sys.stdin), _writer(proto))
        finally:
            host.close_all()
    host = Host(load_program(args.program))
    sys.stdout = sys.stderr
    path = Path(args.socket)
    path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen()
    try:
        while True:
            conn, _ = server.accept()
            with (
                conn,
                conn.makefile("r", encoding="utf-8") as reader,
                conn.makefile("w", encoding="utf-8") as writer,
            ):
                status = serve(host, _lines(reader), _writer(writer))
            if status:
                return status
    finally:
        host.close_all()
        server.close()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
