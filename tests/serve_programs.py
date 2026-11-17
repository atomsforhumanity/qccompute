"""Programs for the worker's tests, loaded in a worker process as module:attribute."""

import os
import time

import numpy as np
from qcdata import (
    ErrorInfo,
    ErrorKind,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    Provenance,
    Resolved,
)

import qccompute
from qccompute.program import Plan, Resources
from qccompute.testing.toy import ToyProgram


class DeviceLost(ToyProgram):
    """Every one-off compute reports a poisoned GPU context."""

    name = "device-lost"

    def compute(self, inp, ctx):
        error = ErrorInfo(
            kind=ErrorKind.backend, code="device_lost", message="context poisoned"
        )
        return ProgramOutput.failure(inp, error)


class Noisy(ToyProgram):
    """Prints to stdout from Python and from the file descriptor, as libraries do."""

    name = "noisy"

    def compute(self, inp, ctx):
        print("python print on stdout")
        os.write(1, b"fd 1 write\n")
        return super().compute(inp.model_copy(update={"program": "toy"}), ctx)


class Isolated(ToyProgram):
    """The toy in a process of its own; ``sleep`` (s) and ``crash`` keywords for tests."""

    name = "isolated"

    def capabilities(self):
        return super().capabilities().model_copy(update={"in_process": False})

    def _own(self, inp):
        keywords = {
            k: v for k, v in inp.keywords.items() if k not in ("sleep", "crash")
        }
        return inp.model_copy(update={"program": "toy", "keywords": keywords})

    def plan(self, inp):
        return super().plan(self._own(inp))

    def compute(self, inp, ctx=None):
        if inp.keywords.get("crash"):
            os._exit(9)
        time.sleep(inp.keywords.get("sleep", 0))
        out = super().compute(self._own(inp), ctx)
        results = out.results.model_copy(update={"extras": {"pid": os.getpid()}})
        return out.model_copy(update={"input_data": inp, "results": results})

    def open_session(self, inp, ctx=None, calctypes=None):
        return super().open_session(self._own(inp), ctx, calctypes)


class Relay(ToyProgram):
    """In a worker of its own, runs its subprogram through ctx: one, a list, a session.

    Like geomeTRIC: the subprogram is whatever program the runner finds.
    """

    name = "relay"

    def capabilities(self):
        caps = super().capabilities()
        return caps.model_copy(update={"in_process": False, "sessions": False})

    def plan(self, inp):
        return Plan(resolved=Resolved(), resources=Resources(threads=2))

    def compute(self, inp, ctx=None):
        if ctx is None:
            return qccompute.compute(inp)
        spec = inp.subprograms[0]
        child = ProgramInput.from_spec(spec, inp.structure)
        one = ctx.compute(child)
        many = ctx.compute([child, child], raise_exc=False)
        x = np.asarray(inp.structure.geometry)
        with ctx.session(spec, inp.structure) as s:
            frames = [s.compute(), s.compute(x * 1.001)]
        extras = {
            "pid": os.getpid(),
            "list": [o.success for o in many],
            "warm": [f.record.get("warm_start") for f in frames],
            "threads": one.execution.threads,
        }
        results = one.results.model_copy(
            update={"provenance": Provenance(program="relay"), "extras": extras}
        )
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=results,
            execution=ExecutionInfo(threads=ctx.threads),
        )


device_lost = DeviceLost()
relay = Relay()
noisy = Noisy()
isolated = Isolated()
