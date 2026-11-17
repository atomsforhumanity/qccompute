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
from qccompute.testing.toy import ToyProgram, ToySession


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
        extras = {
            "pid": os.getpid(),
            "devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "ordinals": ctx.device_ordinals() if ctx is not None else None,
        }
        results = out.results.model_copy(update={"extras": extras})
        return out.model_copy(update={"input_data": inp, "results": results})

    def open_session(self, inp, ctx=None, calctypes=None):
        return PidSession(self._own(inp), calctypes=calctypes, ctx=ctx)


class PidSession(ToySession):
    """The toy's session, with its process in each frame's record."""

    def compute(self, *args, **kwargs):
        frame = super().compute(*args, **kwargs)
        return frame.model_copy(update={"record": {**frame.record, "pid": os.getpid()}})


class Stateless(Isolated):
    """The isolated toy without sessions or batches: a session runs its compute."""

    name = "stateless"

    def capabilities(self):
        caps = super().capabilities()
        return caps.model_copy(update={"sessions": False, "batches": False})


class Stepper(ToyProgram):
    """The toy, reporting three steps of each calculation and each session request."""

    name = "stepper"

    def compute(self, inp, ctx=None):
        if ctx is not None and ctx.progress is not None:
            for step in range(3):
                ctx.progress({"event": "step", "step": step, "pid": os.getpid()})
        own = inp.model_copy(update={"program": "toy"})
        return super().compute(own, ctx).model_copy(update={"input_data": inp})

    def compute_batch(self, inputs, ctx=None):
        return [self.compute(i, ctx) for i in inputs]

    def open_session(self, inp, ctx=None, calctypes=None):
        own = inp.model_copy(update={"program": "toy"})
        return StepperSession(own, calctypes=calctypes, ctx=ctx)


class StepperSession(ToySession):
    """The toy's session, reporting each request."""

    def __init__(self, inp, *, calctypes=None, ctx=None):
        super().__init__(inp, calctypes=calctypes, ctx=ctx)
        self._progress = None if ctx is None else ctx.progress

    def compute(self, *args, **kwargs):
        if self._progress is not None:
            self._progress({"event": "request", "pid": os.getpid()})
        return super().compute(*args, **kwargs)

    def compute_many(self, requests):
        if self._progress is not None:
            for _ in requests:
                self._progress({"event": "request", "pid": os.getpid()})
        return super().compute_many(requests)


class Serial(ToyProgram):
    """The toy, not thread-safe (like qcopt), sleeping ``sleep`` seconds per call."""

    name = "serial"

    def capabilities(self):
        caps = super().capabilities()
        return caps.model_copy(update={"thread_safe": False, "batches": False})

    def _own(self, inp):
        keywords = {k: v for k, v in inp.keywords.items() if k != "sleep"}
        return inp.model_copy(update={"program": "toy", "keywords": keywords})

    def plan(self, inp):
        return super().plan(self._own(inp))

    def compute(self, inp, ctx=None):
        start = time.monotonic()
        time.sleep(inp.keywords.get("sleep", 0))
        out = super().compute(self._own(inp), ctx)
        extras = {"pid": os.getpid(), "start": start, "end": time.monotonic()}
        results = out.results.model_copy(update={"extras": extras})
        return out.model_copy(update={"input_data": inp, "results": results})


class IsolatedGPU(Isolated):
    """The isolated toy, planning one GPU (a GPU program in a worker of its own)."""

    name = "isolated-gpu"

    def plan(self, inp):
        plan = super().plan(inp)
        return plan.model_copy(update={"resources": Resources(gpus=1)})


def blas_threads() -> int | None:
    """The thread count of this process's first BLAS library (numpy's), if any."""
    from threadpoolctl import threadpool_info

    counts = [i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"]
    return counts[0] if counts else None


class Pools(ToyProgram):
    """The toy, reporting the BLAS threads each calculation and request ran with."""

    name = "pools"

    def compute(self, inp, ctx=None):
        own = inp.model_copy(update={"program": "toy"})
        out = super().compute(own, ctx)
        extras = {"blas": blas_threads(), "pid": os.getpid()}
        results = out.results.model_copy(update={"extras": extras})
        return out.model_copy(update={"input_data": inp, "results": results})

    def compute_batch(self, inputs, ctx=None):
        return [self.compute(i, ctx) for i in inputs]

    def open_session(self, inp, ctx=None, calctypes=None):
        own = inp.model_copy(update={"program": "toy"})
        return PoolsSession(own, calctypes=calctypes, ctx=ctx)


class PoolsSession(ToySession):
    """The toy's session, with the BLAS threads of each request in its record."""

    def compute(self, *args, **kwargs):
        frame = super().compute(*args, **kwargs)
        record = {**frame.record, "blas": blas_threads()}
        return frame.model_copy(update={"record": record})


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


class Stray(Relay):
    """In a worker of its own, runs a toy calculation its input does not name."""

    name = "stray"

    def compute(self, inp, ctx=None):
        child = inp.model_copy(update={"program": "toy", "subprograms": []})
        return ctx.compute(child).model_copy(update={"input_data": inp})


device_lost = DeviceLost()
pools = Pools()
stray = Stray()
relay = Relay()
noisy = Noisy()
isolated = Isolated()
stateless = Stateless()
stepper = Stepper()
serial = Serial()
isolated_gpu = IsolatedGPU()
