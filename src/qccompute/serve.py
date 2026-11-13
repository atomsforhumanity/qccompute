"""The worker: one loop that hosts any program's sessions in its own process.

    python -m qccompute.serve linus                  # stdin/stdout, one client
    python -m qccompute.serve linus --socket PATH    # a Unix socket, one client at a time

A worker keeps a program alive between calls (warm starts, a live GPU context) and
isolates it: a crash ends the worker, not its caller. The program is found by name
(`qccompute.program.find_program`: registered in this process, else the
``qccompute.programs`` entry points) or given as ``module:attribute``. The loop uses only
qcdata and the protocol, and knows nothing of the program beyond it. Design:
``docs/design/program-interface.md``, "Workers".

Wire, one JSON message per line:

- The first line is the hello: ``{"protocol": 1, "program", "version", "capabilities",
  "extras"}``; ``extras`` are the capabilities' (``{"numerics_epoch": 8}``), so a store
  guard can check them without importing the program.
- Every message carries an ``id``; replies are ``{"id", "ok": true, "result"}`` or
  ``{"id", "ok": false, "error": ErrorInfo}``. ``ok: false`` is for the message itself
  (bad JSON, an unknown op or session, a failed open or plan); a failed calculation is a
  result: a failed Frame or ProgramOutput.
- Ops: ``open {input, calctypes, ctx}`` -> ``{"session"}``; ``compute {session,
  positions, calctype, accuracy, lineage, outputs, seed, commit}`` -> Frame;
  ``compute_many {session, requests}`` -> [Frame]; ``reset {session, lineage}`` (absent:
  every lineage; null: the default one); ``plan {session}``, ``{session, calctype}`` or
  ``{input}``; ``close {session}``; ``compute {input, ctx}`` (a one-off) ->
  ProgramOutput; ``lineages {session}`` where the session offers them.
- ``ctx`` is ``{threads, memory_gib, scratch_dir, gpus}``. The GPU is fixed when the
  worker is spawned, so ``gpus`` must be empty or match ``CUDA_VISIBLE_DEVICES``.
- Numbers are shortest round-trip decimals, so positions and results cross bit for bit.
  No NaN or Infinity: a message holding one is refused, and a non-finite result becomes a
  failed frame (kind ``numerical``, code ``non_finite``).
- End of input closes every session and exits 0. After a reply whose error has the code
  ``device_lost`` (a GPU fault that poisoned the device context) the worker exits 3:
  nothing more can run in this process.

Hygiene: stdout carries protocol lines only. At start the worker writes the protocol to a
dup of the original stdout and points file descriptor 1 at stderr, because CUDA, BLAS and
other libraries print.

First written by the linus session (SESSION package) against this protocol.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import os
import socket
import sys
import traceback
from collections.abc import Callable, Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import IO, Any

from qcdata import (
    CalcType,
    ErrorInfo,
    ErrorKind,
    Frame,
    ProgramError,
    ProgramInput,
    ProgramOutput,
    Request,
)

from .program import (
    Program,
    RunContext,
    Session,
    SessionProgram,
    capabilities_of,
    failed_frame,
    find_program,
    plan_of,
)

__all__ = ["PROTOCOL", "EXIT_DEVICE_LOST", "Host", "serve", "main"]

#: The wire protocol's version, in the hello line.
PROTOCOL = 1
#: The exit status after a ``device_lost`` reply.
EXIT_DEVICE_LOST = 3
_OPS = "open, compute, compute_many, reset, plan, close, lineages"


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
    """The program named ``spec`` (`find_program`), or the object at ``module:attribute``.

    Raises:
        SystemExit: No such program.
    """
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


def _dump(value: Any) -> Any:
    """A model, or a list of them, as Python values for `_encode`."""
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _plain(x: Any) -> Any:
    """``json``'s fallback: numpy values, paths, enums and sets as JSON values."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, set | frozenset):
        return sorted(x, key=str)
    raise TypeError(f"{type(x).__name__} is not JSON serializable")


def _encode(reply: dict[str, Any]) -> str:
    """One protocol line; ValueError on NaN or Infinity."""
    return json.dumps(reply, allow_nan=False, separators=(",", ":"), default=_plain)


def _finite(value: Any) -> bool:
    try:
        json.dumps(value, allow_nan=False, default=_plain)
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
    """

    def __init__(self, program: Program):
        """Host a program."""
        self.program = program
        self.sessions: dict[str, Session] = {}
        self._ids = itertools.count(1)
        self.device_lost = False

    def hello(self) -> dict[str, Any]:
        """The worker's first line: protocol, program, version, capabilities, extras."""
        caps = capabilities_of(self.program).model_dump(mode="json")
        return {
            "protocol": PROTOCOL,
            "program": self.program.name,
            "version": self.program.version(),
            "capabilities": caps,
            "extras": dict(caps.get("extras") or {}),
        }

    def handle(self, msg: dict[str, Any]) -> Any:
        """The result of one message, as Python values.

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

    def op_open(self, msg: dict[str, Any]) -> dict[str, str]:
        """Open a session; its id."""
        if not isinstance(self.program, SessionProgram):
            raise MessageError(
                f"{self.program.name} has no sessions", ErrorKind.unsupported
            )
        inp = self._input(msg)
        calctypes = msg.get("calctypes")
        try:
            session = self.program.open_session(
                inp,
                self._ctx(msg.get("ctx")),
                frozenset(calctypes) if calctypes else None,
            )
        except ProgramError as e:
            raise MessageError.of(e) from None
        key = f"s{next(self._ids)}"
        self.sessions[key] = session
        return {"session": key}

    def op_compute(self, msg: dict[str, Any]) -> Any:
        """A session's frame, or a one-off output for ``input``."""
        if "input" in msg:
            out = self.program.compute(self._input(msg), self._ctx(msg.get("ctx")))
            self._watch(out)
            return _dump(out)
        session = self._session(msg)
        fields = ("positions", "calctype", "accuracy", "lineage", "outputs", "seed")
        r = _request({k: msg[k] for k in (*fields, "commit") if k in msg})
        try:
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
        return _dump(frame)

    def op_compute_many(self, msg: dict[str, Any]) -> Any:
        """A session's frames for many requests, in order."""
        session = self._session(msg)
        frames = session.compute_many([_request(r) for r in msg.get("requests") or []])
        for frame in frames:
            self._watch(frame)
        return _dump(list(frames))

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
                return _dump(plan_of(self.program, self._input(msg)))
            session = self._session(msg)
            if msg.get("calctype"):
                try:
                    calctype = CalcType(msg["calctype"])
                except ValueError as e:
                    raise MessageError(str(e)) from None
                inp = session.input_data.model_copy(update={"calctype": calctype})
                return _dump(plan_of(self.program, inp))
            return _dump(session.plan())
        except ProgramError as e:
            raise MessageError.of(e) from None

    def op_close(self, msg: dict[str, Any]) -> None:
        """Close a session."""
        self._session(msg)
        self.sessions.pop(msg["session"]).close()

    def close_all(self) -> None:
        """Close every session, reporting (not raising) failures."""
        for session in self.sessions.values():
            try:
                session.close()
            except Exception as e:  # noqa: BLE001 - closing at exit: report, go on
                print(f"qccompute.serve: closing a session: {e}", file=sys.stderr)
        self.sessions.clear()

    def _input(self, msg: dict[str, Any]) -> ProgramInput:
        try:
            return ProgramInput.model_validate(msg["input"])
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
        visible = tuple(
            g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g
        )
        if gpus and gpus != visible:
            raise MessageError(
                f"this worker's GPUs are {list(visible)}, fixed at its start; ctx asks "
                f"for {list(gpus)}",
                ErrorKind.environment,
            )
        unknown = sorted(set(fields) - {"threads", "memory_gib", "scratch_dir"})
        if unknown:
            raise MessageError(
                f"unknown ctx fields {unknown}; ctx has threads, memory_gib, "
                "scratch_dir, gpus"
            )
        scratch = fields.get("scratch_dir")
        return RunContext(
            threads=fields.get("threads"),
            memory_gib=fields.get("memory_gib"),
            gpus=visible,
            scratch_dir=Path(scratch) if scratch else None,
        )

    def _watch(self, out: Frame | ProgramOutput[Any, Any]) -> None:
        if (
            not out.success
            and out.error is not None
            and out.error.code == "device_lost"
        ):
            self.device_lost = True


def _id_of(line: str) -> Any:
    """The id of a message refused while parsing (a NaN in it), if it has one."""
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    return msg.get("id") if isinstance(msg, dict) else None


def _reply(host: Host, line: str) -> str:
    """The reply line to one message line."""
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
        return _encode(
            {"id": _id_of(line), "ok": False, "error": error.model_dump(mode="json")}
        )
    mid = msg.get("id")
    try:
        result = host.handle(msg)
        try:
            return _encode({"id": mid, "ok": True, "result": result})
        except ValueError:
            return _encode({"id": mid, "ok": True, "result": _non_finite(result)})
    except (MessageError, ProgramError) as e:  # ProgramError: e.g. a closed session
        error = e.error
    except Exception as e:  # noqa: BLE001 - a bug in the loop or a program: keep serving
        traceback.print_exc(file=sys.stderr)
        error = ErrorInfo(kind=ErrorKind.internal, message=f"{type(e).__name__}: {e}")
    return _encode({"id": mid, "ok": False, "error": error.model_dump(mode="json")})


def serve(host: Host, lines: Iterable[str], write: Callable[[str], None]) -> int:
    """Answer every message of ``lines`` through ``write``, after the hello.

    Args:
        host: The program's sessions.
        lines: Message lines.
        write: Writes one reply line (without its newline).

    Returns:
        The exit status: 0 at the end of input, `EXIT_DEVICE_LOST` after a
        ``device_lost`` reply.
    """
    write(_encode(host.hello()))
    for line in lines:
        if not line.strip():
            continue
        write(_reply(host, line))
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
