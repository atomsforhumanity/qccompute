"""The Runner: one path for one input or many, failures that raise, ctx, the store hook."""

import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qcdata import (
    CalcType,
    ErrorKind,
    ExecutionInfo,
    FileInput,
    Model,
    ProgramError,
    ProgramErrors,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    Request,
    Resolved,
    SinglePointData,
    Structure,
)

import qccompute
from qccompute import RunContext, Runner
from qccompute.program import (
    Capabilities,
    ComputeSession,
    Plan,
    Resources,
    available_programs,
    register_program,
    unregister_program,
)
from qccompute.runner import default_runner
from qccompute.testing.toy import ToyProgram, ToySession
from tests import serve_programs

ARGON3 = Structure(
    symbols=["Ar", "Ar", "Ar"],
    geometry=np.array([[0.0, 0.0, 0.0], [7.2, 0.0, 0.0], [3.0, 6.1, 0.4]]),
)


def toy_input(
    calctype: str = "gradient", program: str = "toy", **kw: Any
) -> ProgramInput:
    return ProgramInput.model_validate(
        {
            "program": program,
            "calctype": calctype,
            "structure": ARGON3,
            "model": {"method": "lj"},
            "keywords": kw,
        }
    )


class Recording(ToyProgram):
    """The toy, recording each run's context; optionally not thread-safe."""

    def __init__(self, name: str, thread_safe: bool = True):
        self.name = name
        self._thread_safe = thread_safe
        self.contexts: list[RunContext] = []
        self.scratches: list = []
        self.running = 0
        self.most = 0
        self._count = threading.Lock()

    def capabilities(self) -> Capabilities:
        return (
            super()
            .capabilities()
            .model_copy(
                update={
                    "thread_safe": self._thread_safe,
                    "batches": False,
                }
            )
        )

    def compute(self, inp, ctx=None):
        with self._count:
            self.running += 1
            self.most = max(self.most, self.running)
        self.contexts.append(ctx)
        self.scratches.append(ctx.scratch)
        time.sleep(0.01)
        with self._count:
            self.running -= 1
        return (
            super()
            .compute(inp.model_copy(update={"program": "toy"}), ctx)
            .model_copy(update={"input_data": inp})
        )


class NoSessions(Recording):
    def capabilities(self) -> Capabilities:
        return super().capabilities().model_copy(update={"sessions": False})


class Broken(ToyProgram):
    name = "broken"

    def compute(self, inp, ctx=None):
        raise RuntimeError("a bug")


class FiniteDifference(ToyProgram):
    """Its subprogram's gradient by central differences, through ctx."""

    name = "fd"

    def numerics(self, spec):
        return "0+fd"

    def capabilities(self):
        return super().capabilities().model_copy(update={"sessions": False})

    def plan(self, inp):
        return Plan(resolved=Resolved(settings={"step": 1e-4}), resources=Resources())

    def compute(self, inp, ctx=None):
        if ctx is None:  # a direct call: the default runner runs it, with a ctx
            return qccompute.compute(inp)
        child = inp.subprograms[0]
        x = np.asarray(inp.structure.geometry, dtype=float).reshape(-1)
        step = 1e-4
        requests = []
        for k in range(len(x)):
            for sign in (1, -1):
                d = np.zeros_like(x)
                d[k] = sign * step
                requests.append(
                    Request(positions=(x + d).reshape(-1, 3), calctype=CalcType.energy)
                )
        with ctx.session(child, inp.structure, threads=1) as s:
            frames = s.compute_many(requests)
        energies = np.array([f.results.energy for f in frames])
        center = ctx.compute(ProgramInput.from_spec(child, inp.structure))
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=SinglePointData(
                provenance=Provenance(program="fd", program_version="0"),
                energy=center.results.energy,
                gradient=(energies[0::2] - energies[1::2]) / (2 * step),
            ),
            resolved=self.plan(inp).resolved,
            execution=ExecutionInfo(threads=ctx.threads),
        )


def fd_input(child: str = "toy", **child_keywords: Any) -> ProgramInput:
    return ProgramInput.model_validate(
        {
            "program": "fd",
            "calctype": "gradient",
            "structure": ARGON3,
            "subprograms": [
                {
                    "program": child,
                    "calctype": "energy",
                    "model": {"method": "lj"},
                    "keywords": child_keywords,
                }
            ],
        }
    )


class FakeStore:
    """The minimal store a runner needs: key, get, put, tag; runs per identities."""

    def __init__(self):
        self.runs: dict[tuple[str, tuple], ProgramOutput] = {}
        self.tags: dict[str, set[str]] = {}
        self.meta: dict[str, Any] = {}
        self.puts = 0

    def key(self, inp):
        return inp.model_dump_json(exclude={"extras"})

    def get(self, key, identities):
        return self.runs.get((key, tuple(sorted(identities.items()))))

    def put(self, output, *, tags=(), meta=None):
        key = self.key(output.input_data)
        identities = output.execution.identities
        self.runs[key, tuple(sorted(identities.items()))] = output
        self.meta[key] = meta
        self.puts += 1
        self.tag(key, *tags)
        return f"run-{self.puts}"

    def tag(self, key, *tags):
        self.tags.setdefault(key, set()).update(tags)


@pytest.fixture(autouse=True)
def toy_registered():
    register_program(ToyProgram())
    yield
    for name in available_programs():
        unregister_program(name)  # installed programs stay


# --------------------------------------------------------------------- one input


def test_one_input_runs_in_process_with_the_threads_it_asks_for():
    out = qccompute.compute(toy_input("hessian", sigma="auto"))
    assert out.success
    assert out.resolved.settings["sigma"] == 6.0
    assert out.results.hessian.shape == (9, 9)
    assert out.results.gradient.shape == (3, 3)
    assert out.execution.wall_time is not None and out.execution.hostname
    assert out.execution.threads == default_runner().threads  # the whole count
    assert "total" in out.execution.timings


def test_threads_are_the_keyword_else_the_callers_whole_count():
    recording = Recording("rec")
    register_program(recording)
    with Runner(threads=4) as runner:
        runner.compute(toy_input(program="rec"))  # nothing asked: the runner's count
        runner.compute(toy_input(program="rec", threads=2))  # the keyword
        with pytest.warns(RuntimeWarning, match="threads=64 and runs on 4"):
            runner.compute(toy_input(program="rec", threads=64))  # at most the runner's
    assert [c.threads for c in recording.contexts] == [4, 2, 4]
    assert all(c.plan is not None for c in recording.contexts)  # handed on
    plain = NoSessions("plain")
    register_program(plain)
    with Runner(threads=4) as runner:
        with runner.session(toy_input(program="plain")) as s:  # a session: the count
            s.compute()
            s.compute_many([Request(), Request()])  # its requests share it
        with runner.session(toy_input(program="plain"), threads=3) as s:
            s.compute()
        with runner.session(toy_input(program="plain", threads=1)) as s:
            s.compute()
    assert [c.threads for c in plain.contexts] == [4, 2, 2, 3, 1]
    with pytest.raises(ProgramError) as e:  # refused by its plan, without running
        Runner(threads=4).compute(toy_input(program="rec", sigmaa=1.0))
    assert e.value.error.code == "unknown_key" and len(recording.contexts) == 3
    assert e.value.output.execution.wall_time == 0.0

    class Parent(FiniteDifference):
        """Runs its subprogram once, in a session, and in a session of 3 threads."""

        name = "parent"

        def compute(self, inp, ctx=None):
            child = inp.subprograms[0]
            ctx.compute(ProgramInput.from_spec(child, inp.structure))
            with ctx.session(child, inp.structure) as s:
                s.compute()
            with ctx.session(child, inp.structure, threads=3) as s:
                s.compute()
            return super().compute(inp, ctx)

    register_program(Parent())
    del plain.contexts[:]
    parent = fd_input("plain").model_copy(update={"program": "parent"})
    with Runner(threads=8) as runner:
        runner.compute(parent)
        assert [c.threads for c in plain.contexts][:3] == [8, 8, 3]
        del plain.contexts[:]
        runner.compute(parent.model_copy(update={"keywords": {"threads": 2}}))
        assert [c.threads for c in plain.contexts][:3] == [2, 2, 2]  # never more
    with pytest.raises(ValidationError):  # programs don't suggest threads
        Resources(threads=2)  # type: ignore[call-arg]


def test_the_default_count_is_omp_num_threads_else_the_cores(monkeypatch):
    from qccompute.runner import default_threads, usable_cores

    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert default_threads() == 3 and Runner().threads == 3
    monkeypatch.setenv("OMP_NUM_THREADS", "2,1")  # nested levels: the outer one
    assert default_threads() == 2
    for unset in ("", "0", "many"):
        monkeypatch.setenv("OMP_NUM_THREADS", unset)
        assert default_threads() == usable_cores()
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert Runner().threads == usable_cores() and Runner(threads=5).threads == 5


def test_failures_raise_program_errors_with_the_output():
    with pytest.raises(ProgramError, match="Did you mean 'sigma'") as e:
        qccompute.compute(toy_input(sigmaa=1.0))
    assert e.value.error.kind is ErrorKind.input
    assert e.value.output.error.kind is ErrorKind.input
    assert e.value.output.input_data.keywords == {"sigmaa": 1.0}
    with pytest.raises(ProgramError) as e:
        qccompute.compute(toy_input(fail="convergence"))
    assert e.value.output.error.kind is ErrorKind.convergence
    assert e.value.output.execution.wall_time is not None
    out = qccompute.compute(toy_input(fail="resource"), raise_exc=False)
    assert not out.success and out.error.kind is ErrorKind.resource


def test_program_bugs_are_internal_failures_caused_by_the_bug():
    register_program(Broken())
    out = qccompute.compute(toy_input(program="broken"), raise_exc=False)
    assert not out.success and out.error.kind is ErrorKind.internal
    assert "RuntimeError: a bug" in out.traceback
    with pytest.raises(ProgramError) as e:
        qccompute.compute(toy_input(program="broken"))
    assert isinstance(e.value.__cause__, RuntimeError)


def test_missing_programs_are_environment_failures():
    with pytest.raises(ProgramError) as e:
        qccompute.compute(toy_input(program="no-such-program"))
    assert e.value.error.kind is ErrorKind.environment
    assert e.value.error.code == "not_installed"
    from qccompute.program import available_programs

    if "files" not in available_programs():  # qccodec provides it when installed
        with pytest.raises(ProgramError, match="files"):
            qccompute.compute(FileInput(program="echo", cmdline_args=["hi"]))


def test_options_are_checked():
    with pytest.raises(ValueError, match="store"):
        qccompute.compute(toy_input(), tags=["a"])
    with pytest.raises(ValueError, match="timeout"):
        qccompute.compute(toy_input(), timeout=0)
    with pytest.raises(TypeError):
        qccompute.compute(toy_input(), collect_files=True)  # type: ignore[call-arg]


def test_the_default_runner_is_one_and_made_on_first_use():
    assert qccompute.default_runner is default_runner  # exported
    assert default_runner() is default_runner()
    assert default_runner().isolation == "auto"
    with pytest.raises(ValueError, match="isolation"):
        Runner(isolation="thread")  # type: ignore[arg-type]


# ------------------------------------------------------------------------- lists


def test_lists_run_side_by_side_with_their_planned_grants():
    recording = Recording("rec")
    register_program(recording)
    inputs = [
        toy_input(program="rec", threads=2, epsilon=0.01 * (i + 1)) for i in range(6)
    ]
    with Runner(threads=4) as runner:
        outs = runner.compute(inputs)
    assert [o.input_data.keywords["epsilon"] for o in outs] == [
        i.keywords["epsilon"] for i in inputs
    ]
    assert all(o.success for o in outs)
    assert {c.threads for c in recording.contexts} == {2}
    assert recording.most == 2  # 4 threads hold two 2-thread calculations
    assert {c.memory_gib for c in recording.contexts} != {None}
    assert not any(p.exists() for p in recording.scratches)  # removed afterwards


def test_lists_of_programs_that_are_not_thread_safe_take_turns():
    serial = NoSessions("serial", thread_safe=False)
    register_program(serial)
    with pytest.warns(RuntimeWarning, match="isolation='process'"):
        outs = Runner(threads=4).compute(
            [toy_input(program="serial") for _ in range(4)]
        )
    assert all(o.success for o in outs)
    assert serial.most == 1


def test_lists_plan_first_and_refuse_without_running():
    recording = Recording("rec")
    register_program(recording)
    outs = qccompute.compute(
        [toy_input(program="rec"), toy_input(program="rec", sigmaa=1)],
        raise_exc=False,
    )
    assert [o.success for o in outs] == [True, False]
    assert outs[1].error.code == "unknown_key"
    assert len(recording.contexts) == 1  # the refused input never ran
    assert recording.contexts[0].plan is not None  # the plan is handed on


def test_list_failures_are_one_group_holding_every_output():
    with pytest.raises(ProgramErrors) as e:
        qccompute.compute([toy_input(), toy_input(fail="numerical"), toy_input()])
    assert len(e.value.exceptions) == 1
    assert isinstance(e.value.exceptions[0], ProgramError)
    assert e.value.exceptions[0].output.error.kind is ErrorKind.numerical
    assert [o.success for o in e.value.results] == [True, False, True]
    outs = qccompute.compute(
        [toy_input(), toy_input(fail="numerical")], raise_exc=False
    )
    assert [o.success for o in outs] == [True, False]
    with pytest.raises(ProgramErrors) as e:
        qccompute.compute([toy_input(fail="numerical")])
    try:
        qccompute.compute([toy_input(), toy_input(fail="numerical")])
    except* ProgramError as group:
        assert len(group.exceptions) == 1
        assert len(group.results) == 2  # type: ignore[attr-defined]


def test_outputs_as_they_finish_come_from_submitted_futures():
    """``concurrent.futures.as_completed`` over `submit` (no runner method of its own)."""
    assert not hasattr(Runner, "as_completed")
    with Runner(threads=2) as runner:
        inputs = [toy_input(epsilon=0.01 * (i + 1), threads=1) for i in range(4)]
        futures = [runner.submit(i) for i in inputs]
        done = [f.result() for f in as_completed(futures)]
    seen = sorted(o.input_data.keywords["epsilon"] for o in done)
    assert seen == [0.01, 0.02, 0.03, 0.04]


def test_batch_programs_get_their_inputs_together():
    calls = []

    class Batching(ToyProgram):
        name = "batching"

        def compute_batch(self, inputs, ctx=None):
            calls.append(len(inputs))
            toy = [i.model_copy(update={"program": "toy"}) for i in inputs]
            return [
                o.model_copy(update={"input_data": i})
                for i, o in zip(inputs, super().compute_batch(toy, ctx), strict=True)
            ]

    register_program(Batching())
    inputs = [toy_input(program="batching", epsilon=0.01 * (i + 1)) for i in range(5)]
    inputs.append(toy_input(program="batching", fail="numerical"))
    outs = qccompute.compute(inputs, raise_exc=False)
    assert calls == [6]
    assert [o.success for o in outs] == [True] * 5 + [False]
    assert outs[-1].error.kind is ErrorKind.numerical


def test_a_batch_gets_the_threads_of_its_inputs_together():
    """A batch gets its inputs' shares of the list's count together (#39)."""
    granted = []

    class Batching(ToyProgram):
        name = "batching"

        def compute_batch(self, inputs, ctx=None):
            granted.append((len(inputs), ctx.threads))
            toy = [i.model_copy(update={"program": "toy"}) for i in inputs]
            return [
                o.model_copy(update={"input_data": i})
                for i, o in zip(inputs, super().compute_batch(toy, ctx), strict=True)
            ]

    register_program(Batching())

    def batch(n, **kw):
        return [
            toy_input(program="batching", epsilon=0.01 * (i + 1), **kw)
            for i in range(n)
        ]

    with Runner(threads=8) as runner:
        runner.compute(batch(8))  # 1 each
        runner.compute(batch(3))  # 2 each
        runner.compute(batch(20))  # 1 each, at most the runner's in all
        runner.compute(batch(3, threads=1))  # the keyword, summed
        runner.compute(batch(2) + [toy_input(), toy_input()])  # four share 8
    assert granted == [(8, 8), (3, 6), (20, 8), (3, 3), (2, 4)]


def test_a_batch_output_reports_its_own_inputs_threads():
    """Not the whole batch's: its term in the sum, scaled down if capped (#52)."""

    class Unrecorded(ToyProgram):  # a batch program that records no threads
        name = "unrecorded"

        def compute_batch(self, inputs, ctx=None):
            toy = [i.model_copy(update={"program": "toy"}) for i in inputs]
            outs = super().compute_batch(toy, ctx)
            return [
                o.model_copy(
                    update={
                        "input_data": i,
                        "execution": o.execution.model_copy(update={"threads": None}),
                    }
                )
                for i, o in zip(inputs, outs, strict=True)
            ]

    register_program(Unrecorded())

    def batch(*threads):
        return [
            toy_input(program="unrecorded", epsilon=0.01 * (i + 1), threads=t)
            for i, t in enumerate(threads)
        ]

    with Runner(threads=8) as runner:
        three = runner.compute(batch(2, 2, 2))  # 6 threads, uncapped
        capped = runner.compute(batch(8, 4, 4))  # 16 asked, 8 granted
        many = runner.compute(batch(*[1] * 20))  # 20 asked, 8 granted
    assert [o.execution.threads for o in three] == [2, 2, 2]
    assert [o.execution.threads for o in capped] == [4, 2, 2]  # adds up to 8
    assert {o.execution.threads for o in many} == {1}  # at least 1 each
    # A program that cannot tell each input's wall time leaves the batch's (#60).
    assert len({o.execution.wall_time for o in capped}) == 1
    assert {o.execution.timings["batch"] for o in capped} == {
        capped[0].execution.wall_time
    }


def test_a_batch_output_keeps_its_own_wall_time():
    """A lockstep batch knows when each task ended; the batch's is timings.batch (#60)."""

    class Lockstep(ToyProgram):
        name = "lockstep"

        def compute_batch(self, inputs, ctx=None):
            time.sleep(0.05)
            toy = [i.model_copy(update={"program": "toy"}) for i in inputs]
            outs = super().compute_batch(toy, ctx)
            return [
                o.model_copy(
                    update={
                        "input_data": i,
                        "execution": o.execution.model_copy(
                            update={"wall_time": 0.01 * (n + 1)}
                        ),
                    }
                )
                for n, (i, o) in enumerate(zip(inputs, outs, strict=True))
            ]

    register_program(Lockstep())
    inputs = [toy_input(program="lockstep", epsilon=0.01 * (i + 1)) for i in range(3)]
    outs = qccompute.compute(inputs)
    assert [o.execution.wall_time for o in outs] == [0.01, 0.02, 0.03]
    batch = {o.execution.timings["batch"] for o in outs}
    assert len(batch) == 1 and batch.pop() >= 0.05
    assert "batch" not in qccompute.compute(toy_input()).execution.timings


# ------------------------------------------------------------------------- store


def test_the_store_is_looked_up_first_and_written_after():
    store = FakeStore()
    recording = Recording("rec")
    register_program(recording)
    inp = toy_input(program="rec")
    first = qccompute.compute(inp, store=store, tags=["a"], meta={"by": "test"})
    again = qccompute.compute(inp, store=store, tags=["b"])
    assert len(recording.contexts) == 1 and again is first  # a stored success
    key = store.key(inp)
    assert store.tags[key] == {"a", "b"}  # tags go on hits too
    assert store.meta[key] == {"by": "test"}
    qccompute.compute(inp, store=store, rerun=True)
    assert len(recording.contexts) == 2
    outs = qccompute.compute([inp, toy_input(program="rec", epsilon=0.5)], store=store)
    assert len(recording.contexts) == 3  # only the miss ran
    assert all(o.success for o in outs)


def test_stored_failures_are_reused_only_when_the_input_determines_them():
    store = FakeStore()
    recording = Recording("rec")
    register_program(recording)
    parents = []

    class Counted(FiniteDifference):
        def compute(self, inp, ctx=None):
            parents.append(inp)
            return super().compute(inp, ctx)

    register_program(Counted())
    refused = toy_input(program="rec", fail="input")
    unconverged = toy_input(program="rec", fail="convergence")
    out_of_memory = toy_input(program="rec", fail="resource")
    for inp in (refused, unconverged, out_of_memory):
        with pytest.raises(ProgramError):
            qccompute.compute(inp, store=store)
    assert len(recording.contexts) == 3
    for inp, kind in ((refused, "input"), (unconverged, "convergence")):
        with pytest.raises(ProgramError) as e:
            qccompute.compute(inp, store=store)  # the same code fails it again
        assert e.value.error.kind == kind
    assert len(recording.contexts) == 3
    with pytest.raises(ProgramError):
        qccompute.compute(out_of_memory, store=store)  # a bigger grant may succeed
    assert len(recording.contexts) == 4
    with pytest.raises(ProgramError):
        qccompute.compute(unconverged, store=store, rerun=True)  # the way past one
    assert len(recording.contexts) == 5
    # A parent failed by its subprogram: reused when the subprogram's failure is.
    for child_kind, reused in (("convergence", True), ("resource", False)):
        parent = fd_input("rec", fail=child_kind)
        with pytest.raises(ProgramError):
            qccompute.compute(parent, store=store)
        ran = len(parents)
        with pytest.raises(ProgramError) as e:
            qccompute.compute(parent, store=store)
        assert e.value.error.kind is ErrorKind.program
        assert e.value.error.cause.kind == child_kind
        assert (len(parents) == ran) is reused
    recording.numerics = lambda spec: "epoch 2"  # new numbers: no stored run reused
    ran = len(recording.contexts)
    with pytest.raises(ProgramError):
        qccompute.compute(refused, store=store)
    assert len(recording.contexts) == ran + 1


def test_identities_name_every_program_a_calculation_runs():
    register_program(FiniteDifference())
    runner = Runner(threads=1)
    toy = default_runner()._entry(toy_input()).program
    assert runner.identities(fd_input("toy")) == {"fd": "0+fd", "toy": toy.version()}
    assert runner.identities(toy_input(program="no-such")) == {}
    store = FakeStore()
    out = runner.compute(fd_input("toy"), store=store)
    assert out.execution.identities == runner.identities(fd_input("toy"))
    assert (
        store.get(store.key(fd_input("toy")), runner.identities(fd_input("toy"))) is out
    )

    class Epochs(ToyProgram):
        name = "epochs"

        def numerics(self, spec):
            return {"lj": "lj-1", "x": "x-2"}[spec.model.method]

    register_program(Epochs())
    child = fd_input("epochs").subprograms[0]
    two = fd_input("epochs").model_copy(
        update={
            "subprograms": [
                child,
                child.model_copy(update={"model": Model(method="x")}),
            ]
        }
    )
    assert runner.identities(two)["epochs"] == "lj-1;x-2"


def test_every_output_says_which_code_made_it(isolated):
    """execution.identities on outputs that ran, failed, were refused or batched (#38)."""
    seen = []

    class Parent(FiniteDifference):
        def compute(self, inp, ctx=None):
            seen.append(ctx.compute(ProgramInput.from_spec(inp.subprograms[0], ARGON3)))
            return super().compute(inp, ctx)

    register_program(Parent())
    version = default_runner()._entry(toy_input()).program.version()
    toy = {"toy": version}
    with Runner(threads=2) as runner:
        assert runner.compute(toy_input()).execution.identities == toy
        failed = runner.compute(toy_input(fail="convergence"), raise_exc=False)
        refused = runner.compute(toy_input(sigmaa=1.0), raise_exc=False)
        assert failed.execution.identities == refused.execution.identities == toy
        outs = runner.compute(
            [toy_input(epsilon=0.02), toy_input(epsilon=0.03)]  # a batch
            + [toy_input(program="isolated"), toy_input(program="no-such")],
            raise_exc=False,
        )
        assert [o.execution.identities for o in outs] == [
            toy,
            toy,
            {"isolated": version},  # in a worker
            {},  # not installed
        ]
        out = runner.compute(fd_input("toy"))
        assert out.execution.identities == {"fd": "0+fd", **toy}
        assert seen[0].execution.identities == toy  # a subprogram's output
    # A stored run keeps the identities it was made with.
    store = FakeStore()
    first = qccompute.compute(toy_input(), store=store)
    assert qccompute.compute(toy_input(), store=store).execution.identities == toy
    assert store.runs[store.key(toy_input()), (("toy", version),)] is first


def test_every_output_with_a_store_names_its_run():
    """Computed or hit, so a result can be cited exactly (#57)."""
    store = FakeStore()
    out = qccompute.compute(toy_input(), store=store)
    assert out.execution.store_run == f"{store.key(toy_input())}/run-1"
    inputs = [toy_input(epsilon=0.02), toy_input(sigmaa=1.0), toy_input(fail="input")]
    outs = qccompute.compute(inputs, store=store, raise_exc=False)
    cited = [o.execution.store_run.rpartition("/") for o in outs]
    assert [key for key, _, _ in cited] == [store.key(i) for i in inputs]
    assert sorted(run for _, _, run in cited) == ["run-2", "run-3", "run-4"]
    hit = qccompute.compute(toy_input(), store=store)
    assert hit.execution.store_run == out.execution.store_run
    assert qccompute.compute(toy_input()).execution.store_run is None  # no store


def test_outputs_record_the_environment(monkeypatch):
    """CPU model, libc and the granted GPUs' names beside the host (#58)."""
    from qccompute import program
    from qccompute.runner import _host

    host = _host()
    out = qccompute.compute(toy_input())
    assert out.execution.hostname == host["hostname"]
    assert out.execution.cpu_model == host["cpu_model"]
    assert out.execution.libc == host["libc"]
    assert out.execution.devices == []
    if os.path.exists("/proc/cpuinfo"):
        assert out.execution.cpu_model and out.execution.libc.startswith("glibc ")

    class Elsewhere(ToyProgram):  # a program that knows better sets them itself
        name = "elsewhere"

        def plan(self, inp):
            plan = super().plan(inp.model_copy(update={"program": "toy"}))
            return plan.model_copy(update={"resources": Resources(gpus=1)})

        def compute(self, inp, ctx=None):
            out = super().compute(inp.model_copy(update={"program": "toy"}), ctx)
            mine = {"cpu_model": "Other CPU", "devices": ["Other GPU"]}
            if inp.keywords.get("epsilon") == 0.5:
                mine = {}
            execution = out.execution.model_copy(update=mine)
            return out.model_copy(update={"input_data": inp, "execution": execution})

    register_program(Elsewhere())
    rows = (("0", "GPU-aaaa", 80.0, "NVIDIA A100"), ("1", "GPU-bbbb", 24.0, "L4"))
    monkeypatch.setattr(program, "_nvidia_smi_gpus", lambda: rows)
    with Runner(threads=2, gpus=["GPU-bbbb", "7"]) as runner:
        set_by_program = runner.compute(toy_input(program="elsewhere"))
        assert set_by_program.execution.cpu_model == "Other CPU"
        assert set_by_program.execution.devices == ["Other GPU"]
        assert set_by_program.execution.libc == host["libc"]
        granted = runner.compute(toy_input(program="elsewhere", epsilon=0.5))
        assert granted.execution.devices == ["L4"]
        assert granted.execution.cpu_model == host["cpu_model"]
    with Runner(threads=2, gpus=["7"]) as runner:  # a GPU nvidia-smi does not list
        out = runner.compute(toy_input(program="elsewhere", epsilon=0.5))
        assert out.execution.devices == ["7"]


class FallbackHere(serve_programs.Fallback):
    name = "fallback-here"

    def capabilities(self):
        return super().capabilities().model_copy(update={"in_process": True})


@pytest.mark.parametrize("program", ["fallback-here", "fallback"])
def test_a_program_that_ran_on_no_device_says_so(program):
    """devices=[] stays: a GPU's name is filled only where the program left it unset."""
    register_program(serve_programs.fallback)
    register_program(FallbackHere())
    with Runner(threads=2, gpus=["7"]) as runner:
        on_cpu = runner.compute(toy_input(program=program, fallback=True))
        on_gpu = runner.compute(toy_input(program=program))
    assert on_cpu.execution.devices == []
    assert on_gpu.execution.devices == ["7"]  # nvidia-smi does not list it
    in_worker = program == "fallback"
    assert (on_gpu.results.extras["pid"] != os.getpid()) is in_worker


def test_warm_started_outputs_are_never_stored():
    class Warm(ToyProgram):
        name = "warm"

        def compute(self, inp, ctx=None):
            out = super().compute(inp.model_copy(update={"program": "toy"}), ctx)
            return out.model_copy(
                update={"input_data": inp, "record": {"warm_start": True}}
            )

    register_program(Warm())
    store = FakeStore()
    out = qccompute.compute(toy_input(program="warm"), store=store)
    qccompute.compute([toy_input(program="warm"), toy_input()], store=store)
    assert store.puts == 1  # only the toy's own run
    assert out.execution.store_run is None


# ---------------------------------------------------------------------- sessions


def test_sessions_are_the_programs_own_in_process():
    with qccompute.session(toy_input()) as s:
        assert type(s._session) is ToySession  # behind a wrapper that counts threads
        assert s.lineages() == []  # the program's own extras come through
        r = s.compute(np.asarray(ARGON3.geometry)).results
        assert r.gradient.shape == (3, 3)
    register_program(NoSessions("plain"))
    with qccompute.session(toy_input(program="plain")) as s:
        assert isinstance(s, ComputeSession)
        assert s.program_version == ToyProgram().version()  # filled at opening
        r2 = s.compute(np.asarray(ARGON3.geometry)).results
        assert r2.energy == pytest.approx(r.energy)
        with pytest.raises(ProgramError) as e:
            s.compute(np.full((3, 3), np.nan))
        assert e.value.error.kind is ErrorKind.numerical
        assert s.compute(np.asarray(ARGON3.geometry)).results.energy == r2.energy
        with pytest.raises(ProgramErrors) as group:
            s.compute_many([Request(), Request(positions=np.full((3, 3), np.nan))])
        assert [f.success for f in group.value.results] == [True, False]
    with pytest.raises(ProgramError) as e:
        qccompute.session(toy_input(program="no-such-program"))
    assert e.value.error.kind is ErrorKind.environment


def test_accuracy_targets_reach_programs():
    """The shared keyword accuracy, and a request's target in a session (#35, #41)."""
    from qcdata import Accuracy

    from qccompute.program import ComputeSession, FunctionProgram

    def met(**target):
        return {
            "target": target,
            "thresholds": {},
            "estimate": dict.fromkeys(target, 0.0),
        }

    # Programs with nothing to converge accept a target, keyword or request, and
    # record it as met: no thresholds, an estimate of 0.0 per target (#41).
    register_program(FunctionProgram("bowl", lambda x: (0.0, 0 * x)))
    target = {"accuracy": {"gradient": 1e-6}}
    for program in ("toy", "bowl"):
        out = qccompute.compute(toy_input(program=program, **target))
        assert out.record["accuracy"] == met(gradient=1e-6)
        assert out.resolved.settings["accuracy"] == {"gradient": 1e-6}
        assert "accuracy" not in qccompute.compute(toy_input(program=program)).record
        for bad, code in (({}, "no_effect"), ({"gradinet": 1e-6}, "unknown_key")):
            with pytest.raises(ProgramError) as e:
                qccompute.compute(toy_input(program=program, accuracy=bad))
            assert e.value.error.kind is ErrorKind.input
            assert e.value.error.code == code
    with qccompute.session(toy_input()) as s:  # the toy's own session
        assert "accuracy" not in s.compute().record
        assert s.compute(accuracy=1e-6).record["accuracy"] == met(gradient=1e-6)
    with qccompute.session(toy_input(**target)) as s:
        assert s.compute().record["accuracy"] == met(gradient=1e-6)  # the input's
        assert s.compute(accuracy=Accuracy(energy=1e-9)).record["accuracy"] == met(
            energy=1e-9
        )
    # A session over compute passes a request's target on as the keyword, which a
    # program with nothing to converge accepts: right end to end (#41).
    bowl = toy_input(program="bowl")
    with qccompute.session(bowl) as s:
        assert isinstance(s, ComputeSession)
        assert "accuracy" not in s.compute().record
        frame = s.compute(accuracy=1e-6)
        assert frame.record == {"warm_start": False, "accuracy": met(gradient=1e-6)}
        frames = s.compute_many(
            [Request(accuracy=Accuracy(energy=1e-8), calctype="energy"), Request()]
        )
        assert [f.record.get("accuracy") for f in frames] == [met(energy=1e-8), None]

    # A session over compute passes a request's target on as the input's keyword.
    seen = []

    class Converging(NoSessions):
        @staticmethod
        def _own(inp):
            keywords = {k: v for k, v in inp.keywords.items() if k != "accuracy"}
            return inp.model_copy(update={"keywords": keywords})

        def plan(self, inp):
            return super().plan(self._own(inp))

        def compute(self, inp, ctx=None):
            seen.append(inp.keywords.get("accuracy"))
            out = super().compute(self._own(inp), ctx)
            return out.model_copy(update={"input_data": inp})

    register_program(Converging("converging"))
    inp = toy_input(program="converging", accuracy={"energy": 1e-8})
    with qccompute.session(inp) as s:
        s.compute()  # the input's own target
        s.compute(accuracy=1e-6)  # a number targets the highest derivative
        s.compute_many([Request(accuracy=Accuracy(hessian=1e-4)), Request()])
    assert seen == [
        {"energy": 1e-8},
        {"gradient": 1e-6},
        {"hessian": 1e-4},
        {"energy": 1e-8},
    ]


@pytest.mark.parametrize("thread_safe", [False, True])
def test_sessions_may_be_called_from_threads(thread_safe):
    """The runner makes concurrent session calls safe; callers need not ask."""
    from concurrent.futures import ThreadPoolExecutor

    program = NoSessions(f"exclusive-{thread_safe}", thread_safe)
    register_program(program)
    runner = Runner(threads=4)
    one = toy_input(program=program.name)
    sessions = [runner.session(one, threads=1) for _ in range(4)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(4) as pool:
            frames = list(pool.map(lambda s: s.compute(), sessions))
    waited = [w for w in caught if "waiting their turn" in str(w.message)]
    assert len(waited) == (0 if thread_safe else 1)  # once per runner and program
    assert all(f.success for f in frames)
    assert program.most > 1 if thread_safe else program.most == 1


def test_a_sessions_compute_many_runs_its_requests_side_by_side():
    """They share the session's count, as a list does: 13 images, 8 at a time."""
    program = NoSessions("images")
    register_program(program)
    x = np.asarray(ARGON3.geometry)
    images = [Request(positions=x * (1 + 0.01 * i)) for i in range(13)]
    with Runner(threads=8) as runner:
        with runner.session(toy_input(program="images")) as s:
            frames = s.compute_many(images)
        assert program.most == 8  # 13 one-thread requests on 8 threads
        assert {c.threads for c in program.contexts} == {1}
        assert all(f.success for f in frames)
        program.most = 0
        with runner.session(toy_input(program="images"), threads=3) as s:
            s.compute_many(images)  # the session's own count caps them
        assert program.most == 3
        program.most = 0
        with runner.session(toy_input(program="images", threads=2)) as s:
            s.compute_many(images)  # its keyword is the session's count: 2 at once
            s.compute()
        assert program.most == 2 and program.contexts[-1].threads == 2
        # A native session's count is set at open, for the program to split.
        opened = []

        class Opening(ToyProgram):
            name = "opening"

            def open_session(self, inp, ctx=None, calctypes=None):
                opened.append(ctx.threads)
                own = inp.model_copy(update={"program": "toy"})
                return super().open_session(own, ctx, calctypes)

        register_program(Opening())
        runner.session(toy_input(program="opening"), threads=13).close()
        runner.session(toy_input(program="opening")).close()
        runner.session(toy_input(program="opening", threads=4)).close()
        assert opened == [8, 8, 4]  # threads, else the keyword, else the count
    with pytest.raises(ValueError):
        Runner(threads=2).session(toy_input(), threads=0)


def test_a_session_of_a_worker_program_without_sessions_uses_its_pool():
    """Its requests are calculations, placed in the program's workers (#54)."""
    register_program(serve_programs.stateless)
    x = np.asarray(ARGON3.geometry)
    images = [Request(positions=x * (1 + 0.01 * i)) for i in range(4)]
    with Runner(threads=2) as runner:
        with runner.session(toy_input(program="stateless", sleep=0.2)) as s:
            start = time.perf_counter()
            frames = s.compute_many(images)
            elapsed = time.perf_counter() - start
    assert all(f.success for f in frames)
    assert len({f.results.extras["pid"] for f in frames}) == 2  # two workers
    assert elapsed < 4 * 0.2 + 0.4  # two at a time, after the workers start


# ------------------------------------------------------------------------ submit


def test_submit_returns_a_future_of_the_output():
    recording = Recording("rec")
    register_program(recording)
    future = qccompute.submit(toy_input(program="rec"))
    assert isinstance(future, Future)
    out = future.result(timeout=10)
    assert out.success and out.input_data.program == "rec"
    failed = qccompute.submit(toy_input(program="rec", fail="convergence"))
    with pytest.raises(ProgramError) as e:
        failed.result(timeout=10)
    assert e.value.error.kind is ErrorKind.convergence
    assert e.value.output.input_data.keywords["fail"] == "convergence"
    # An input its program refuses resolves at once, without running.
    refused = qccompute.submit(toy_input(program="rec", sigmaa=1.0))
    assert refused.done() and refused.exception().error.code == "unknown_key"
    # A store hit resolves at once; a run is stored and names its run.
    store = FakeStore()
    first = qccompute.submit(toy_input(program="rec"), store=store, tags=["a"])
    assert first.result(timeout=10).execution.store_run.endswith("/run-1")
    ran = len(recording.contexts)
    hit = qccompute.submit(toy_input(program="rec"), store=store, tags=["b"])
    assert hit.done() and len(recording.contexts) == ran
    assert hit.result() is first.result()
    assert store.tags[store.key(toy_input(program="rec"))] == {"a", "b"}
    with pytest.raises(ValueError):
        qccompute.submit(toy_input(), tags=["x"])
    events = []
    register_program(serve_programs.stepper)
    qccompute.submit(toy_input(program="stepper"), progress=events.append).result()
    assert [e["path"] for e in events] == [["stepper"]] * 3


def test_submitted_calculations_share_the_runners_threads():
    """Each takes its count; they start in the order submitted as threads free up."""
    recording = Recording("rec")
    register_program(recording)
    with Runner(threads=4) as runner:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            whole = [
                runner.submit(toy_input(program="rec", epsilon=0.01 * (i + 1)))
                for i in range(4)
            ]
            assert all(f.result(timeout=10).success for f in whole)
        assert recording.most == 1  # each takes all 4 threads: one at a time
        named = [w for w in caught if "Set threads" in str(w.message)]
        assert len(named) == 1  # once per runner, naming the way out
        recording.most = 0
        futures = [
            runner.submit(toy_input(program="rec", threads=1, epsilon=0.01 * (i + 1)))
            for i in range(12)
        ]
        assert all(f.result(timeout=10).success for f in futures)
        assert recording.most == 4
        recording.most = 0
        two = [runner.submit(toy_input(program="rec", threads=2)) for _ in range(6)]
        assert all(f.result(timeout=10).success for f in two)
        assert recording.most == 2
        assert {c.threads for c in recording.contexts[-6:]} == {2}


def test_a_cancelled_submission_gives_up_its_turn():
    register_program(serve_programs.serial)
    with Runner(threads=1) as runner:
        slow = runner.submit(toy_input(program="serial", sleep=0.2))
        pending = [runner.submit(toy_input(program="serial")) for _ in range(3)]
        assert pending[1].cancel()
        assert slow.result(timeout=10).success
        assert pending[0].result(timeout=10).success
        assert pending[2].result(timeout=10).success  # no turn left waiting


def _alive(pid: int) -> bool:
    """Whether a process runs (a zombie, killed and not yet reaped, does not)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _gone(pid: int, within: float = 10.0) -> bool:
    deadline = time.monotonic() + within
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _pid(path: Path, within: float = 20.0) -> int:
    deadline = time.monotonic() + within
    while not path.exists() or not path.read_text():
        assert time.monotonic() < deadline, "the program never started"
        time.sleep(0.02)
    return int(path.read_text())


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_cancelling_a_running_calculation_kills_its_worker_and_what_it_started(
    tmp_path,
):
    """future.cancel() on a running calculation ends its whole process tree (#74)."""
    register_program(serve_programs.spawner)
    pidfile = tmp_path / "child.pid"
    with Runner(threads=2) as runner:
        future = runner.submit(
            toy_input(program="spawner", sleep=60, pidfile=str(pidfile))
        )
        child = _pid(pidfile)
        assert future.running()
        assert future.cancel() and future.cancelled() and not future.running()
        assert _gone(child)  # in a session of its own, and killed all the same
        with pytest.raises(CancelledError):
            future.result(timeout=10)
        pidfile.unlink()
        # The pool carries on: a new worker for the next calculation.
        again = runner.submit(toy_input(program="spawner", pidfile=str(pidfile)))
        assert again.result(timeout=30).success
        os.kill(_pid(pidfile), 9)  # this one ran to its end: its child is ours


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
def test_a_timeout_kills_the_workers_whole_tree(tmp_path):
    register_program(serve_programs.spawner)
    pidfile = tmp_path / "child.pid"
    with Runner(threads=1) as runner:
        with pytest.raises(ProgramError) as e:
            runner.compute(
                toy_input(program="spawner", sleep=60, pidfile=str(pidfile)),
                timeout=3,
            )
    assert e.value.error.kind is ErrorKind.timeout
    assert _gone(_pid(pidfile))


def test_a_calculations_scratch_is_kept_for_debugging_when_asked(monkeypatch):
    made: list[Path] = []

    class Scratchy(ToyProgram):
        name = "scratchy"

        def compute(self, inp, ctx=None):
            made.append(ctx.scratch)
            (ctx.scratch / "work").write_text("partial")
            own = inp.model_copy(update={"program": "toy"})
            return super().compute(own, ctx).model_copy(update={"input_data": inp})

    register_program(Scratchy())
    with Runner(threads=1) as runner:
        runner.compute(toy_input(program="scratchy"))
        assert not made[-1].exists()
        monkeypatch.setenv("QCCOMPUTE_KEEP_SCRATCH", "1")
        runner.compute(toy_input(program="scratchy"))
    kept = made[-1]
    assert kept.name.startswith("qccompute-") and (kept / "work").exists()
    shutil.rmtree(kept)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
@pytest.mark.parametrize("how", ["cancel", "timeout", "kept"])
def test_a_killed_workers_scratch_is_removed_unless_kept(tmp_path, monkeypatch, how):
    """Its calculation's scratch goes with it, at once; kept for debugging when asked."""
    if how == "kept":
        monkeypatch.setenv("QCCOMPUTE_KEEP_SCRATCH", "1")
    register_program(serve_programs.spawner)
    pidfile = tmp_path / "child.pid"
    inp = toy_input(program="spawner", sleep=60, pidfile=str(pidfile))
    with Runner(threads=1) as runner:
        future = runner.submit(inp, timeout=3 if how == "timeout" else None)
        child = _pid(pidfile)
        scratch = Path(pidfile.with_suffix(".scratch").read_text())
        root = scratch.parent  # the worker's own, never the system's temporary one
        assert root.name.startswith("qccompute-worker-"), root
        assert (scratch / "work").exists()
        if how != "timeout":
            assert future.cancel()
        assert _gone(child)
        deadline = time.monotonic() + 10
        while scratch.exists() and how != "kept":  # when the runner lets it go
            assert time.monotonic() < deadline, "the killed worker's scratch stayed"
            time.sleep(0.02)
        if how == "timeout":
            with pytest.raises(ProgramError) as e:
                future.result(timeout=10)
            assert e.value.error.kind is ErrorKind.timeout
    if how == "kept":
        assert (scratch / "work").read_text() == "partial"
        shutil.rmtree(root)
    else:
        assert not root.exists()


class Patient(ToyProgram):
    """In process, works in steps while it has time (as a program honouring
    ctx.timeout does), starting a subprogram when its time is up."""

    name = "patient"

    def __init__(self):
        self.started = threading.Event()
        self.saw: list[str] = []

    def compute(self, inp, ctx=None):
        self.started.set()
        for _ in range(1000):
            if ctx.timeout == 0:
                self.saw.append("no time left")
                try:
                    ctx.compute(toy_input())
                except ProgramError as e:
                    self.saw.append(e.error.kind)
                raise ProgramError.of("timeout", "stopped: no time left")
            time.sleep(0.01)
        self.saw.append("ran to the end")
        own = inp.model_copy(update={"program": "toy"})
        return super().compute(own, ctx).model_copy(update={"input_data": inp})


def test_cancelling_a_calculation_in_process_ends_it_as_its_time_running_out():
    patient = Patient()
    register_program(patient)
    store = FakeStore()
    with Runner(threads=2) as runner:
        inp = toy_input(program="patient").model_copy(
            update={"subprograms": [ProgramSpec(program="toy", calctype="gradient")]}
        )
        future = runner.submit(
            ProgramInput.model_validate(inp.model_dump()), store=store
        )
        assert patient.started.wait(10)
        assert future.cancel()
        deadline = time.monotonic() + 10
        while len(patient.saw) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert patient.saw == ["no time left", ErrorKind.timeout]
    assert future.cancelled() and store.puts == 0  # never stored


class Gated(ToyProgram):
    """In process, records the order calculations start; ``epsilon`` 0.9 waits for the gate."""

    name = "gated"

    def __init__(self):
        self.order: list[float] = []
        self.gate = threading.Event()

    def compute(self, inp, ctx=None):
        epsilon = inp.keywords.get("epsilon")
        self.order.append(epsilon)
        if epsilon == 0.9:
            self.gate.wait(10)
        own = inp.model_copy(update={"program": "toy"})
        return super().compute(own, ctx).model_copy(update={"input_data": inp})


def test_submitted_calculations_start_in_the_order_submitted():
    """First come, first served: no priorities."""
    gated = Gated()
    register_program(gated)
    with Runner(threads=2) as runner:
        first = runner.submit(toy_input(program="gated", epsilon=0.9))
        while not gated.order:
            time.sleep(0.01)
        submitted = [
            runner.submit(toy_input(program="gated", epsilon=e))
            for e in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        ]
        time.sleep(0.05)  # the first waits for the runner's threads, the rest behind
        assert submitted[1].cancel()  # waiting: it never runs
        assert not first.cancelled() and not submitted[0].running()
        gated.gate.set()
        assert all(f.result(timeout=10).success for f in [first, *submitted[2:]])
        assert submitted[0].result(timeout=10).success
    assert gated.order == [0.9, 0.1, 0.3, 0.4, 0.5, 0.6]
    with pytest.raises(TypeError):
        Runner().submit(toy_input(), priority=1)  # type: ignore[call-arg]


def test_closing_a_runner_cancels_what_waits():
    gated = Gated()
    register_program(gated)
    runner = Runner(threads=1)
    first = runner.submit(toy_input(program="gated", epsilon=0.9))
    while not gated.order:
        time.sleep(0.01)
    waiting = runner.submit(toy_input(program="gated", epsilon=0.1))
    runner.close()
    assert waiting.cancelled()
    gated.gate.set()
    assert first.result(timeout=10).success  # in process: it ran on


@pytest.mark.parametrize("isolation", ["auto", "process"])
def test_a_stopped_calculation_keeps_what_it_reached_and_is_never_stored(isolation):
    """runner.stop(future): the program ends at its next step, kind stopped (#75)."""
    register_program(serve_programs.stepping)
    store = FakeStore()
    with Runner(threads=2, isolation=isolation) as runner:
        many = toy_input(program="stepping", steps=2000, sleep=0.005)
        future = runner.submit(many, store=store)
        while not future.running():
            time.sleep(0.01)
        time.sleep(0.5)
        assert runner.stop(future)
        with pytest.raises(ProgramError) as e:
            future.result(timeout=30)
        assert not runner.stop(future)  # done: nothing to ask
        details = e.value.output.error.details
        assert e.value.error.kind is ErrorKind.stopped and details["steps"] < 2000
        assert (details["pid"] != os.getpid()) is (isolation == "process")
        assert store.puts == 0
        # The program's process carries on: the worker was asked, not killed.
        after = runner.submit(toy_input(program="stepping"), store=store)
        assert after.result(timeout=30).success and store.puts == 1


def test_a_stop_before_the_start_ends_the_calculation_at_its_first_step():
    register_program(serve_programs.stepping)
    with Runner(threads=1) as runner:
        first = runner.submit(toy_input(program="stepping", steps=20, sleep=0.01))
        waiting = runner.submit(toy_input(program="stepping", steps=20))
        assert qccompute.stop(waiting)
        assert first.result(timeout=10).success
        with pytest.raises(ProgramError) as e:
            waiting.result(timeout=10)
    assert e.value.error.details["steps"] == 0
    with pytest.raises(ValueError):
        qccompute.stop(Future())  # not a calculation's


class Driver(ToyProgram):
    """In process, runs its subprogram and lets a failure of it escape."""

    name = "driver"

    def plan(self, inp):
        own = inp.model_copy(update={"program": "toy", "subprograms": []})
        return super().plan(own)

    def compute(self, inp, ctx=None):
        child = ProgramInput.from_spec(inp.subprograms[0], inp.structure)
        ctx.compute(child)
        raise AssertionError("the subprogram was meant to stop")


def test_a_program_whose_subprogram_stopped_stops_too():
    """The request covers the whole calculation: kind stopped, not program (#75)."""
    register_program(serve_programs.stepping)
    register_program(Driver())
    child = {
        "program": "stepping",
        "calctype": "gradient",
        "model": {"method": "lj"},
        "keywords": {"steps": 2000, "sleep": 0.005},
    }
    inp = ProgramInput.model_validate(
        toy_input(program="driver").model_dump() | {"subprograms": [child]}
    )
    store = FakeStore()
    with Runner(threads=2) as runner:
        future = runner.submit(inp, store=store)
        while not future.running():
            time.sleep(0.01)
        time.sleep(0.2)
        runner.stop(future)
        with pytest.raises(ProgramError) as e:
            future.result(timeout=30)
    error = e.value.error
    assert (error.kind, error.program) == (ErrorKind.stopped, "stepping")
    assert error.cause.kind is ErrorKind.stopped
    assert store.puts == 0


def test_dynamic_workflows_need_no_stage_barriers():
    """Per reaction: search, then IRC, then both ends, as each step finishes (#55)."""
    register_program(serve_programs.serial)

    def step(stage, reaction, seconds):
        return toy_input(
            program="serial",
            epsilon=0.01 + reaction / 100 + stage / 10,
            sleep=seconds,
            threads=1,
        )

    stages = {0: [0.05, 0.6], 1: [0.05, 0.05]}  # per reaction: search, IRC
    with Runner(threads=4, isolation="process") as runner:
        futures = {runner.submit(step(0, r, stages[0][r])): (0, r) for r in (0, 1)}
        ends: dict[int, Any] = {}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                stage, r = futures.pop(future)
                out = future.result()
                ends.setdefault(r, []).append(out.results.extras)
                if stage == 0:
                    futures[runner.submit(step(1, r, stages[1][r]))] = (1, r)
                elif stage == 1:  # both ends of the IRC
                    for side in (2, 3):
                        futures[runner.submit(step(side, r, 0.0))] = (side, r)
    # Reaction 0 finished all its steps before reaction 1's search ended.
    assert max(e["end"] for e in ends[0]) < ends[1][0]["end"]
    assert len(ends[0]) == len(ends[1]) == 4


def test_waiting_on_a_programs_lock_warns_once_and_names_the_way_out():
    """8 threads chaining compute(one) of qcopt serialised silently (#55)."""
    from concurrent.futures import ThreadPoolExecutor

    register_program(serve_programs.serial)
    inputs = [
        toy_input(program="serial", epsilon=e, sleep=0.1, threads=1) for e in (1, 2, 3)
    ]
    with Runner(threads=3) as runner:
        with pytest.warns(RuntimeWarning, match="isolation='process'") as caught:
            with ThreadPoolExecutor(3) as pool:
                outs = list(pool.map(runner.compute, inputs))
        assert len([w for w in caught if "serial" in str(w.message)]) == 1
        spans = sorted(
            (o.results.extras["start"], o.results.extras["end"]) for o in outs
        )
        assert all(
            a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False)
        )  # one at a time
    with Runner(threads=3, isolation="process") as runner:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning: each in its own worker
            with ThreadPoolExecutor(3) as pool:
                outs = list(pool.map(runner.compute, inputs))
        assert len({o.results.extras["pid"] for o in outs}) > 1


# ---------------------------------------------------------------------- the guard


class Busy(ToyProgram):
    """The toy, busy a while in each calculation and session request, adding up the
    threads of those running at once (``most``: the sum at its peak)."""

    def __init__(self, name: str = "busy"):
        self.name = name
        self.lock = threading.Lock()
        self.running = 0
        self.most = 0
        self.seen: list[int] = []

    def capabilities(self):
        caps = super().capabilities()
        return caps.model_copy(update={"thread_safe": True, "batches": False})

    def busy(self, threads: int) -> None:
        with self.lock:
            self.running += threads
            self.most = max(self.most, self.running)
            self.seen.append(threads)
        time.sleep(0.05)
        with self.lock:
            self.running -= threads

    def compute(self, inp, ctx=None):
        self.busy(ctx.threads)
        own = inp.model_copy(update={"program": "toy"})
        return super().compute(own, ctx).model_copy(update={"input_data": inp})

    def open_session(self, inp, ctx=None, calctypes=None):
        program = self

        class Counted(ToySession):
            def compute(self, *args, **kwargs):
                program.busy(ctx.threads)
                return super().compute(*args, **kwargs)

        own = inp.model_copy(update={"program": "toy"})
        return Counted(own, calctypes=calctypes, ctx=ctx)


class Fan(ToyProgram):
    """Runs its subprogram four times at once from its own threads, then a session of
    it four times at once, splitting nothing: the runner holds them to its count."""

    name = "fan"

    def capabilities(self):
        caps = super().capabilities()
        return caps.model_copy(update={"sessions": False, "batches": False})

    def plan(self, inp):
        return super().plan(
            inp.model_copy(update={"program": "toy", "subprograms": []})
        )

    def compute(self, inp, ctx=None):
        from concurrent.futures import ThreadPoolExecutor

        spec = inp.subprograms[0]
        child = ProgramInput.from_spec(spec, inp.structure)
        sessions = [ctx.session(spec, inp.structure) for _ in range(4)]
        with ThreadPoolExecutor(4) as pool:
            assert all(
                o.success for o in pool.map(lambda _: ctx.compute(child), "1234")
            )
            assert all(f.success for f in pool.map(lambda s: s.compute(), sessions))
        for s in sessions:
            s.close()
        own = inp.model_copy(update={"program": "toy", "subprograms": []})
        return super().compute(own, ctx).model_copy(update={"input_data": inp})


def fan_input(**child_keywords: Any) -> ProgramInput:
    child = {"program": "busy", "calctype": "gradient", "model": {"method": "lj"}}
    return ProgramInput.model_validate(
        toy_input(program="fan", threads=2).model_dump()
        | {"subprograms": [child | {"keywords": child_keywords}]}
    )


def test_a_parent_running_children_side_by_side_never_exceeds_its_count():
    busy = Busy()
    register_program(busy)
    register_program(Fan())
    with Runner(threads=8) as runner:
        assert runner.compute(fan_input()).success  # the parent's count: 2
        assert busy.most == 2 and set(busy.seen) == {2}  # one at a time, 2 each
        busy.most = 0
        del busy.seen[:]
        assert runner.compute(fan_input(threads=1)).success
        assert busy.most == 2 and set(busy.seen) == {1}  # two at a time, 1 each


def test_a_worker_parents_children_sent_to_its_client_count_against_it():
    """Counted in the worker before they go, and run in the client (Busy is only here)."""
    busy = Busy()
    register_program(busy)
    register_program(serve_programs.spread)
    inp = fan_input().model_copy(update={"program": "spread"})
    with Runner(threads=8) as runner:
        out = runner.compute(inp)
    assert out.results.extras["pid"] != os.getpid()
    assert out.results.extras["threads"] == [2, 2, 2, 2]
    assert busy.most == 2  # one at a time under the parent's 2


def test_requests_on_one_session_from_several_threads_share_its_count():
    from concurrent.futures import ThreadPoolExecutor

    busy = Busy()
    register_program(busy)
    plain = NoSessions("plain")
    register_program(plain)
    with Runner(threads=8) as runner:
        with runner.session(toy_input(program="busy"), threads=2) as s:
            with ThreadPoolExecutor(6) as pool:
                frames = list(pool.map(lambda _: s.compute(), range(6)))
        assert all(f.success for f in frames)
        assert busy.most == 2 and set(busy.seen) == {2}  # in turn, never 12 threads
        with runner.session(toy_input(program="plain"), threads=2) as s:
            with ThreadPoolExecutor(6) as pool:
                frames = list(pool.map(lambda _: s.compute(), range(6)))
        assert all(f.success for f in frames)
        assert plain.most == 1 and {c.threads for c in plain.contexts} == {2}


def test_open_sessions_hold_no_threads_so_workflows_do_not_hang():
    """A pool of N chains on N threads, each holding a session open beside top-level
    calls, and a whole-count session held open while submitted work is waited on."""
    from concurrent.futures import ThreadPoolExecutor

    busy = Busy()
    register_program(busy)
    with Runner(threads=4) as runner:

        def chain(i: int) -> bool:
            one = toy_input(program="busy", threads=1, epsilon=0.01 * (i + 1))
            with runner.session(one, threads=1) as s:
                s.compute()
                done = runner.compute(one).success  # while the session is open
                runner.compute([one, one])
                s.compute()
            return done

        with ThreadPoolExecutor(4) as pool:
            chains = [pool.submit(chain, i) for i in range(4)]
            assert all(c.result(timeout=30) for c in chains)  # no hang
        with runner.session(toy_input(program="busy")) as s:  # all four threads
            s.compute()
            futures = [runner.submit(toy_input(program="busy")) for _ in range(3)]
            assert all(f.result(timeout=30).success for f in futures)
            s.compute()


def test_a_top_level_call_from_a_thread_holding_threads_fails_at_once():
    seen: list[str] = []

    class Inner(ToyProgram):  # calls a runner itself, inside its calculation
        name = "inner"

        def compute(self, inp, ctx=None):
            calls = (
                lambda: runner.compute(toy_input()),
                lambda: runner.submit(toy_input()),
                lambda: qccompute.compute(toy_input()),
            )
            for call in calls:
                try:
                    call()
                except RuntimeError as e:
                    seen.append(str(e))
            own = inp.model_copy(update={"program": "toy"})
            return super().compute(own, ctx).model_copy(update={"input_data": inp})

    register_program(Inner())
    with Runner(threads=2) as runner:
        start = time.perf_counter()
        assert runner.compute(toy_input(program="inner")).success
        assert time.perf_counter() - start < 5  # at once, no waiting
        with runner.session(toy_input()) as s:
            s.compute()  # from a thread that holds nothing: fine
    assert len(seen) == 3 and all("ctx.compute" in m for m in seen)


def test_waiting_work_can_be_cancelled_or_stopped():
    gated = Gated()
    register_program(gated)
    with Runner(threads=2) as runner:
        first = runner.submit(toy_input(program="gated", epsilon=0.9))
        while not gated.order:
            time.sleep(0.01)
        waiting = runner.submit(toy_input(program="gated", epsilon=0.1))
        stopped = runner.submit(toy_input(program="gated", epsilon=0.2))
        time.sleep(0.1)
        assert not waiting.running() and waiting.cancel()  # waiting for threads
        assert runner.stop(stopped)
        with pytest.raises(ProgramError) as e:
            stopped.result(timeout=10)
        assert e.value.error.kind is ErrorKind.stopped
        gated.gate.set()
        assert first.result(timeout=10).success
    assert gated.order == [0.9]  # neither ran
    # A count's waiter, alone: a cancel or a stop of its caller ends the wait.
    from qccompute._count import Count
    from qccompute.program import _Control

    for end, kind in ((_Control.cancel, ErrorKind.timeout), (_Control.stop, "stopped")):
        count = Count(1)
        count.take(1)
        control = _Control()
        failed: list[ProgramError] = []

        def wait(count=count, control=control, failed=failed):
            try:
                count.take(1, 0, control)
            except ProgramError as e:
                failed.append(e)

        waiter = threading.Thread(target=wait)
        waiter.start()
        time.sleep(0.05)
        end(control)
        waiter.join(1)
        assert failed and failed[0].error.kind == kind and not control.waiting


def test_a_timeout_counts_from_when_a_calculation_starts():
    left: list[float | None] = []

    class TimeLeft(Gated):
        name = "time-left"

        def compute(self, inp, ctx=None):
            left.append(ctx.timeout)
            return super().compute(inp, ctx)

    gated = TimeLeft()
    register_program(gated)
    with Runner(threads=1) as runner:
        first = runner.submit(toy_input(program="time-left", epsilon=0.9))
        while not gated.order:
            time.sleep(0.01)
        late = runner.submit(toy_input(program="time-left", epsilon=0.1), timeout=1.0)
        time.sleep(1.2)  # it waits longer than its timeout
        gated.gate.set()
        assert first.result(timeout=10).success and late.result(timeout=10).success
    assert left[1] > 0.9  # its whole second, from when it started


def test_a_store_hit_takes_no_threads():
    gated = Gated()
    register_program(gated)
    store = FakeStore()
    stored = toy_input(epsilon=0.3)
    with Runner(threads=1) as runner:
        runner.compute(stored, store=store)  # stored
        first = runner.submit(toy_input(program="gated", epsilon=0.9))
        while not gated.order:
            time.sleep(0.01)
        start = time.perf_counter()
        assert runner.compute(stored, store=store).success  # the runner is full
        assert runner.submit(stored, store=store).result(timeout=1).success
        assert runner.compute([stored, stored], store=store)
        assert time.perf_counter() - start < 1
        gated.gate.set()
        assert first.result(timeout=10).success


def test_a_list_shares_its_count_among_the_inputs_that_run():
    recording = Recording("rec")
    register_program(recording)
    store = FakeStore()
    inputs = [toy_input(program="rec", epsilon=0.01 * (i + 1)) for i in range(8)]
    with Runner(threads=8) as runner:
        runner.compute(inputs[:4])  # four share 8: 2 each
        runner.compute(inputs[4:6], store=store)
        del recording.contexts[:]
        runner.compute(inputs[4:7], store=store)  # two hits: the one that runs gets 8
        runner.compute(inputs[:3])  # 8 // 3: 2 each
    assert [c.threads for c in recording.contexts] == [8, 2, 2, 2]


# ---------------------------------------------------------------------- progress


def test_progress_events_reach_the_caller_with_their_path():
    """Programs emit; the runner adds the path; hits emit nothing (14a)."""
    register_program(serve_programs.stepper)
    events = []
    out = qccompute.compute(toy_input(program="stepper"), progress=events.append)
    assert out.success
    assert [e["step"] for e in events] == [0, 1, 2]
    assert all(e["event"] == "step" and e["path"] == ["stepper"] for e in events)
    # Without a callable, programs see None and emit nothing.
    seen = []

    class Looking(ToyProgram):
        name = "looking"

        def compute(self, inp, ctx=None):
            seen.append(ctx.progress)
            own = inp.model_copy(update={"program": "toy"})
            return super().compute(own, ctx).model_copy(update={"input_data": inp})

    register_program(Looking())
    qccompute.compute(toy_input(program="looking"))
    assert seen == [None]
    # A stored output emits nothing.
    store = FakeStore()
    qccompute.compute(toy_input(program="stepper"), store=store)
    del events[:]
    hit = qccompute.compute(
        toy_input(program="stepper"), store=store, progress=events.append
    )
    assert events == [] and hit.success
    # Lists: every calculation's events, batches included.
    inputs = [toy_input(program="stepper", epsilon=e) for e in (0.1, 0.2, 0.3)]
    qccompute.compute(inputs, progress=events.append)
    assert len(events) == 9
    # A session's requests, its own or over compute.
    del events[:]
    with qccompute.session(toy_input(program="stepper"), progress=events.append) as s:
        s.compute()
        s.compute_many([Request(), Request()])
    assert [e["event"] for e in events] == ["request"] * 3
    assert {tuple(e["path"]) for e in events} == {("stepper",)}
    del events[:]
    plain = NoSessions("plain-stepper")
    plain.compute = serve_programs.stepper.compute  # type: ignore[method-assign]
    register_program(plain)
    with qccompute.session(
        toy_input(program="plain-stepper"), progress=events.append
    ) as s:
        s.compute_many([Request(), Request()])
    assert len(events) == 6


def test_a_subprograms_progress_comes_through_its_parent():
    register_program(serve_programs.stepper)

    class Driver(FiniteDifference):  # reports its own step, then runs its child
        name = "driver"

        def compute(self, inp, ctx=None):
            ctx.progress({"event": "start"})
            ctx.compute(ProgramInput.from_spec(inp.subprograms[0], ARGON3))
            with ctx.session(inp.subprograms[0], ARGON3) as s:
                s.compute()
            return super().compute(inp, ctx)

    register_program(Driver())
    events = []
    inp = fd_input("stepper").model_copy(update={"program": "driver"})
    qccompute.compute(inp, progress=events.append)
    paths = [(e["event"], tuple(e["path"])) for e in events]
    assert paths[0] == ("start", ("driver",))
    assert ("step", ("driver", "stepper")) in paths
    assert ("request", ("driver", "stepper")) in paths  # its sessions too


def test_progress_streams_from_workers():
    """A worker's program streams its events over the wire, sessions included."""
    register_program(serve_programs.stepper)
    events = []
    with Runner(threads=2, isolation="process") as runner:
        out = runner.compute(toy_input(program="stepper"), progress=events.append)
        assert [e["step"] for e in events] == [0, 1, 2]
        assert {e["pid"] for e in events} != {os.getpid()}  # emitted in the worker
        assert all(e["path"] == ["stepper"] for e in events)
        del events[:]
        runner.compute(toy_input(program="stepper", epsilon=0.5))  # nobody listens
        assert events == []
        with runner.session(toy_input(program="stepper"), progress=events.append) as s:
            s.compute()
            s.compute()
        assert [e["event"] for e in events] == ["request", "request"]
    assert out.success
    # A worker's subprogram runs here, and its events come through the parent.
    register_program(serve_programs.relay)
    del events[:]
    relay = toy_input(program="relay").model_copy(
        update={
            "subprograms": [
                ProgramSpec.model_validate(
                    {
                        "program": "stepper",
                        "calctype": "gradient",
                        "model": {"method": "lj"},
                    }
                )
            ]
        }
    )
    with Runner(threads=2) as runner:
        runner.compute(relay, progress=events.append)
    assert {tuple(e["path"]) for e in events} == {("relay", "stepper")}


def test_a_failing_progress_callable_is_a_warning():
    register_program(serve_programs.stepper)

    def broken(event):
        raise RuntimeError("display gone")

    with pytest.warns(RuntimeWarning, match="display gone"):
        out = qccompute.compute(toy_input(program="stepper"), progress=broken)
    assert out.success


# ------------------------------------------------------------------- subprograms


@pytest.mark.parametrize("child", ["toy", "plain"])
def test_subprograms_run_through_ctx_within_the_grant(child):
    register_program(FiniteDifference())
    plain = NoSessions("plain")
    register_program(plain)
    with Runner(threads=4) as runner:
        out = runner.compute(fd_input(child))
    exact = qccompute.compute(toy_input("gradient"))
    assert np.allclose(out.results.gradient, exact.results.gradient, atol=1e-8)
    assert out.results.energy == pytest.approx(exact.results.energy)
    if child == "plain":
        assert {c.threads for c in plain.contexts} <= {1, 4}


def test_a_failed_subprogram_fails_its_parent_with_kind_program():
    register_program(FiniteDifference())
    with pytest.raises(ProgramError) as e:
        qccompute.compute(fd_input("toy", fail="convergence"))
    error = e.value.error
    assert error.kind is ErrorKind.program and error.program == "toy"
    assert error.cause.kind is ErrorKind.convergence
    assert e.value.output.error == error


def test_subprograms_must_be_named_by_the_input():
    """So a run's identities cover every program it used (#56)."""

    class PathSearch(
        ToyProgram
    ):  # like a path search that minimizes its ends through ctx
        name = "path"

        def capabilities(self):
            return super().capabilities().model_copy(update={"sessions": False})

        def plan(self, inp):
            return Plan(resolved=Resolved(), resources=Resources())

        def compute(self, inp, ctx=None):
            ends = fd_input("toy")
            if inp.keywords.get("session"):
                with ctx.session(ends.subprograms[0], ARGON3) as s:
                    s.compute()
            ctx.compute([ends] if inp.keywords.get("list") else ends)
            return ProgramOutput(
                input_data=inp,
                success=True,
                results=SinglePointData(
                    provenance=Provenance(program="path"), energy=0.0
                ),
            )

    register_program(PathSearch())
    register_program(FiniteDifference())
    toy_spec = fd_input("toy").subprograms[0]
    fd_spec = ProgramSpec(program="fd", calctype="gradient", subprograms=[toy_spec])
    names_toy = toy_input("energy", program="path").model_copy(
        update={"subprograms": [toy_spec]}
    )
    for keywords in ({}, {"list": True}):
        with pytest.raises(ProgramError) as e:
            qccompute.compute(names_toy.model_copy(update={"keywords": keywords}))
        assert e.value.error.kind is ErrorKind.input
        assert e.value.error.code == "unnamed_subprogram"
        assert "fd" in e.value.error.message
    # fd itself named, but not its own subprogram: refused too.
    fd_alone = fd_spec.model_copy(update={"subprograms": []})
    with pytest.raises(ProgramError) as e:
        qccompute.compute(names_toy.model_copy(update={"subprograms": [fd_alone]}))
    assert e.value.error.code == "unnamed_subprogram"
    assert "toy" in e.value.error.message
    # Named, it runs, and the run's identities cover both.
    named = names_toy.model_copy(update={"subprograms": [fd_spec]})
    out = qccompute.compute(named)
    assert set(out.execution.identities) == {"path", "fd", "toy"}
    assert qccompute.compute(named.model_copy(update={"keywords": {"session": True}}))
    # A session of an unnamed program is refused as well.
    fd_only = toy_input("energy", program="path", session=True).model_copy(
        update={"subprograms": [fd_alone]}
    )
    with pytest.raises(ProgramError) as e:
        qccompute.compute(fd_only)
    assert e.value.error.code == "unnamed_subprogram"
    # From a worker, its calls come back here and are checked here.
    register_program(serve_programs.stray)
    with Runner(threads=2) as runner:
        with pytest.raises(ProgramError) as e:
            runner.compute(toy_input(program="stray"))
        assert e.value.error.code == "unnamed_subprogram"
        out = runner.compute(
            toy_input(program="stray").model_copy(update={"subprograms": [toy_spec]})
        )
        assert out.success


class Swallowing(ToyProgram):
    """Calls a toy calculation its input does not name; makes of the refusal what it
    likes: a success, or a failure of its own."""

    def __init__(self, name, wrap):
        self.name = name
        self.wrap = wrap

    def plan(self, inp):
        return super().plan(
            inp.model_copy(update={"program": "toy", "subprograms": []})
        )

    def compute(self, inp, ctx=None):
        try:
            ctx.compute(toy_input())
        except ProgramError as e:
            if self.wrap:
                raise RuntimeError("the endpoint optimization failed") from e
        own = inp.model_copy(update={"program": "toy", "subprograms": []})
        return super().compute(own, ctx).model_copy(update={"input_data": inp})


@pytest.mark.parametrize("program", ["swallowing", "wrapping", "swallow"])
def test_a_refused_unnamed_subprogram_fails_the_calculation_whatever_follows(program):
    """Kind input, code unnamed_subprogram, so a store reuses the refusal (#95)."""
    register_program(Swallowing("swallowing", wrap=False))
    register_program(Swallowing("wrapping", wrap=True))
    register_program(serve_programs.swallow)  # in a worker: checked here all the same
    store = FakeStore()
    with Runner(threads=2) as runner:
        with pytest.raises(ProgramError) as e:
            runner.compute(toy_input(program=program), store=store)
        assert e.value.error.kind is ErrorKind.input
        assert e.value.error.code == "unnamed_subprogram"
        assert e.value.output.error.code == "unnamed_subprogram"
        again = runner.compute(toy_input(program=program), store=store, raise_exc=False)
    assert again.error.code == "unnamed_subprogram" and store.puts == 1  # reused


def nest_input(child: str = "pools", **extras: Any) -> ProgramInput:
    spec = {"program": child, "calctype": "gradient", "model": {"method": "lj"}}
    if child == "stepping":
        spec["keywords"] = {"steps": 2000, "sleep": 0.005}
    return ProgramInput.model_validate(
        toy_input(program="nest").model_dump()
        | {"subprograms": [spec], "extras": extras}
    )


def test_a_worker_runs_subprograms_it_can_run_itself():
    """In process, no GPU, found there with the client's identities (#93)."""
    for program in (serve_programs.nest, serve_programs.pools):
        register_program(program)
    with Runner(threads=4) as runner:
        out = runner.compute(nest_input())
        extras = out.results.extras
        assert extras["pid"] != os.getpid()
        assert extras["child"] == extras["pid"] == extras["session"]  # no hop back
        assert extras["identities"] == {"pools": serve_programs.pools.version()}
        assert extras["threads"] == 4  # its parent's count, as the client gives it
        assert out.execution.identities == runner.identities(nest_input())

        class Other(serve_programs.Pools):  # another program under the same name
            def version(self):
                return "other"

        register_program(Other())
        out = runner.compute(nest_input())
        extras = out.results.extras
        assert extras["child"] == extras["session"] == os.getpid()  # the client's
        assert extras["identities"] == {"pools": "other"}


def test_subprograms_a_worker_runs_itself_report_and_stop_as_through_its_client():
    """Progress paths, the named-program check and stops behave alike (#93)."""
    for program in (serve_programs.nest, serve_programs.stepper, serve_programs.pools):
        register_program(program)
    register_program(serve_programs.stepping)
    events = []
    with Runner(threads=4) as runner:
        runner.compute(nest_input("stepper"), progress=events.append)
        assert [e["path"] for e in events] == [["nest", "stepper"]] * 3
        assert {e["pid"] for e in events} != {os.getpid()}  # it ran in the worker
        # A program available there but not named: refused there, and it fails.
        with pytest.raises(ProgramError) as e:
            runner.compute(nest_input("stepper", call="pools"))
        assert e.value.error.code == "unnamed_subprogram"
        future = runner.submit(nest_input("stepping"))
        while not future.running():
            time.sleep(0.01)
        time.sleep(1.0)
        assert runner.stop(future)
        with pytest.raises(ProgramError) as e:
            future.result(timeout=30)
    error = e.value.error
    assert error.kind is ErrorKind.stopped and error.cause.kind is ErrorKind.stopped
    assert error.cause.details["pid"] != os.getpid()  # stopped in the worker


def test_subprogram_lists_and_their_failures():
    seen = []

    class Fanout(ToyProgram):
        name = "fanout"

        def plan(self, inp):
            return Plan(resolved=Resolved())

        def compute(self, inp, ctx=None):
            spec = inp.subprograms[0]
            children = [
                ProgramInput.from_spec(
                    spec.model_copy(update={"keywords": {"epsilon": 0.01 * (i + 1)}}),
                    inp.structure,
                )
                for i in range(4)
            ]
            outs = ctx.compute(children, raise_exc=False)
            seen.append([o.execution.threads for o in outs])
            ctx.compute(children)  # raises if any failed
            return ProgramOutput(
                input_data=inp,
                success=True,
                results=SinglePointData(
                    provenance=Provenance(program="fanout", program_version="0"),
                    energy=sum(o.results.energy for o in outs),
                ),
                execution=ExecutionInfo(threads=ctx.threads),
            )

    register_program(Fanout())
    inp = ProgramInput.model_validate(
        {
            "program": "fanout",
            "calctype": "energy",
            "structure": ARGON3,
            "subprograms": [
                {"program": "toy", "calctype": "energy", "model": {"method": "lj"}}
            ],
        }
    )
    with Runner(threads=2) as runner:
        assert runner.compute(inp).success
        # The toy batches: the four children share the parent's 2, together in one
        # call, and the toy runs them one after another with both.
        assert seen == [[2, 2, 2, 2]]
        failing = inp.model_copy(
            update={
                "subprograms": [
                    inp.subprograms[0].model_copy(update={"model": Model(method="x")})
                ]
            }
        )
        with pytest.raises(ProgramError) as e:
            runner.compute(failing)
    assert e.value.error.kind is ErrorKind.program
    assert e.value.error.cause.kind is ErrorKind.input


# ----------------------------------------------------------------------- workers


@pytest.fixture
def isolated():
    register_program(serve_programs.isolated)  # workers import it by module:attribute
    yield serve_programs.isolated


def test_programs_that_cannot_share_a_process_run_in_workers(isolated):
    with Runner(threads=2) as runner:
        out = runner.compute(toy_input(program="isolated"))
        assert out.success and out.results.extras["pid"] != os.getpid()
        outs = runner.compute([toy_input(program="isolated") for _ in range(3)])
        pids = {o.results.extras["pid"] for o in outs}
        assert os.getpid() not in pids and len(pids) <= 2  # live workers are reused
        with pytest.raises(ProgramError) as e:
            runner.compute(toy_input(program="isolated", fail="convergence"))
        assert e.value.output.error.kind is ErrorKind.convergence
        with runner.session(toy_input(program="isolated")) as s:
            assert s.program_version == isolated.version()
            frame = s.compute()
            assert frame.success and frame.record["warm_start"] is False
            assert s.compute(np.asarray(ARGON3.geometry) + 0.01).record["warm_start"]
            with pytest.raises(ProgramError):
                s.compute(np.full((3, 3), np.nan))
            with pytest.raises(ProgramErrors) as group:
                s.compute_many([Request(), Request(positions=np.full((3, 3), np.nan))])
            assert [f.success for f in group.value.results] == [True, False]
            s.reset()
            assert s.plan().resolved.settings["sigma"] == 6.0


def test_isolation_process_runs_every_program_in_a_worker():
    from qccompute.testing import toy

    register_program(toy.program)  # importable: workers load it as module:attribute
    with Runner(threads=2, isolation="process") as runner:
        out = runner.compute(toy_input(threads=2))
        assert out.success and out.execution.threads == 2
        with runner.session(toy_input()) as s:
            assert type(s._session).__name__ == "WorkerSession"
            assert s.compute().success


def test_registered_instances_reach_workers_pickled(tmp_path):
    """An instance no module holds, made in a script, runs in a worker (#62)."""
    from qccompute.program import FunctionProgram

    copy = serve_programs.Isolated()
    copy.name = "isolated-copy"  # held by no module: pickled for the worker
    register_program(copy)
    shadow = serve_programs.Isolated()
    shadow.name = "xtb"  # registered ahead of an installed program of that name
    register_program(shadow)
    with Runner(threads=1) as runner:
        out = runner.compute(toy_input(program="isolated-copy"))
        assert out.results.extras["pid"] != os.getpid()
        assert runner.compute(toy_input(program="xtb")).results.extras["pid"]

        class Scripted(serve_programs.Isolated):  # as if defined in the script run
            name = "scripted"

        Scripted.__module__ = "__main__"
        register_program(Scripted())
        with pytest.raises(ProgramError) as e:
            runner.compute(toy_input(program="scripted"))
        assert e.value.error.kind is ErrorKind.environment
        assert "script being run" in e.value.error.message
    register_program(FunctionProgram("lambda", lambda x: (0.0, 0 * x)))
    lam = toy_input(program="lambda").model_copy(update={"model": None})
    assert qccompute.compute(lam).success  # in process: fine
    with Runner(threads=1, isolation="process") as runner:
        with pytest.raises(ProgramError) as e:
            runner.compute(lam)
        assert "cannot rebuild it" in e.value.error.message
    # The issue's case, end to end: a script registers an instance it made.
    script = tmp_path / "script.py"
    script.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parents[1])!r})\n"
        "import qccompute\n"
        "from tests import serve_programs\n"
        "from tests.test_runner import toy_input\n"
        "program = serve_programs.Isolated()\n"
        "program.name = 'mine'\n"
        "qccompute.register_program(program)\n"
        "with qccompute.Runner(threads=1) as runner:\n"
        "    out = runner.compute(toy_input(program='mine'))\n"
        "print(out.results.extras['pid'] != os.getpid())\n"
    )
    ran = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )
    assert ran.stdout.strip() == "True", ran.stderr


def test_subprograms_of_worker_programs_run_in_the_client():
    """A program in a worker (like geomeTRIC) runs subprograms registered here."""
    from qccompute.program import FunctionProgram

    register_program(serve_programs.relay)
    k = 1.5

    def bowl(x):  # a closure in a test: no worker could import it
        return k * float(np.sum(x**2)), 2 * k * x

    def bowl_hessian(x):
        return 2 * k * np.eye(x.size)

    register_program(FunctionProgram("bowl", bowl, bowl_hessian))

    def relay_input(child: dict[str, Any], structure: Structure = ARGON3):
        return ProgramInput.model_validate(
            {
                "program": "relay",
                "calctype": child["calctype"],
                "structure": structure,
                "subprograms": [child],
            }
        )

    with Runner(threads=4) as runner:
        out = runner.compute(relay_input({"program": "bowl", "calctype": "gradient"}))
        assert out.results.extras["pid"] != os.getpid()
        assert out.results.energy == pytest.approx(k * np.sum(ARGON3.geometry**2))
        assert np.array_equal(out.results.gradient, 2 * k * ARGON3.geometry)
        assert out.results.extras["list"] == [True, True]
        assert out.results.extras["threads"] == 4  # the relay's count
        # A program with sessions of its own opens one here: warm-started
        toy = {"program": "toy", "calctype": "gradient", "model": {"method": "lj"}}
        assert runner.compute(relay_input(toy)).results.extras["warm"] == [False, True]
        # A failed subprogram fails the program in the worker with kind program
        with pytest.raises(ProgramError) as e:
            runner.compute(relay_input({**toy, "keywords": {"fail": "convergence"}}))
        assert e.value.error.kind is ErrorKind.program
        assert e.value.error.program == "toy"
        assert e.value.error.cause.kind is ErrorKind.convergence
        # Large arrays cross both ways as files: a 390 x 390 Hessian, bit for bit
        big = Structure(symbols=["Ar"] * 130, geometry=_lattice(130))
        out = runner.compute(
            relay_input({"program": "bowl", "calctype": "hessian"}, big)
        )
        assert np.array_equal(out.results.hessian, 2 * k * np.eye(390))


def _lattice(n: int) -> np.ndarray:
    """n argon positions (bohr) on a cubic lattice 7.2 bohr apart."""
    side = int(np.ceil(n ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    return 7.2 * np.array(points[:n], dtype=float)


def test_large_arrays_cross_workers_as_files(isolated, monkeypatch):
    """Arrays over 1 MiB cross as .npy files, bit for bit, deleted once read."""
    from qccompute import _wire

    read = []
    load = _wire.load

    def spy(cls, data, directory):
        read.append((directory, list(_wire._pointers(data))))
        return load(cls, data, directory)

    monkeypatch.setattr(_wire, "load", spy)
    big = Structure(symbols=["Ar"] * 130, geometry=_lattice(130))
    inp = toy_input("hessian", program="isolated").model_copy(update={"structure": big})
    with Runner(threads=1) as runner:
        out = runner.compute(inp)
        directory = next(d for d, names in read if names)
        assert directory.is_dir() and not any(directory.iterdir())  # all read
    assert not directory.exists()  # removed with the worker
    here = qccompute.compute(inp.model_copy(update={"program": "toy"}))
    assert out.results.hessian.shape == (390, 390)
    assert np.array_equal(out.results.hessian, here.results.hessian)


def test_timeouts_kill_workers(isolated):
    with Runner(threads=1) as runner:
        with pytest.raises(ProgramError) as e:
            runner.compute(toy_input(program="isolated", sleep=5.0), timeout=0.5)
        assert e.value.error.kind is ErrorKind.timeout
        assert runner.compute(toy_input(program="isolated")).success  # a new worker


def test_a_dead_worker_is_an_environment_failure(isolated):
    with Runner(threads=1) as runner:
        with pytest.raises(ProgramError) as e:
            runner.compute(toy_input(program="isolated", crash=True))
        assert e.value.error.kind is ErrorKind.environment
        assert e.value.error.code == "worker_died"
        assert "exit status" in e.value.error.message
        assert runner.compute(toy_input(program="isolated")).success


# ------------------------------------------------------------ BLAS and OpenMP pools

blas_threads = serve_programs.blas_threads


@pytest.fixture
def pools():
    register_program(serve_programs.pools)  # workers import it by module:attribute
    assert blas_threads() is not None, "numpy's BLAS is not one threadpoolctl finds"
    yield serve_programs.pools


def test_a_calculation_holds_blas_at_its_threads_and_gives_the_counts_back(pools):
    """In process, the pools follow each call's grant, and come back after (#66)."""
    outside = blas_threads()
    with Runner(threads=4) as runner:
        assert runner.compute(toy_input(program="pools")).results.extras["blas"] == 4
        three = runner.compute(toy_input(program="pools", threads=3))
        assert three.results.extras["blas"] == 3
        assert blas_threads() == outside
        batch = runner.compute([toy_input(program="pools") for _ in range(3)])
        assert [o.results.extras["blas"] for o in batch] == [3, 3, 3]  # the batch's
        with runner.session(toy_input(program="pools", threads=2)):
            assert blas_threads() == outside  # opened, it holds nothing
    assert blas_threads() == outside


def test_a_pooled_worker_holds_each_calls_threads(pools):
    """A worker started by a one-thread call gives a later call its own threads (#66)."""
    with Runner(threads=4, isolation="process") as runner:
        outs = [
            runner.compute(toy_input(program="pools", threads=n)) for n in (1, 3, 2)
        ]
        assert len({o.results.extras["pid"] for o in outs}) == 1  # one worker
        assert [o.results.extras["blas"] for o in outs] == [1, 3, 2]
        batch = runner.compute([toy_input(program="pools") for _ in range(3)])
        assert [o.results.extras["blas"] for o in batch] == [3, 3, 3]
        with runner.session(toy_input(program="pools", threads=2)) as s:
            assert s.compute().record["blas"] == 2
        assert runner.compute(toy_input(program="pools")).results.extras["blas"] == 4


class Pinning(ToyProgram):
    """A program with subprograms that raises its own pools while it works."""

    name = "pinning"

    def plan(self, inp):
        return super().plan(
            inp.model_copy(update={"program": "toy", "subprograms": []})
        )

    def compute(self, inp, ctx=None):
        from threadpoolctl import threadpool_limits

        seen = {"granted": ctx.threads, "held": blas_threads()}
        with threadpool_limits(limits=ctx.threads, user_api="blas"):
            seen["own"] = blas_threads()
        spec = inp.subprograms[0]
        child = ProgramInput.from_spec(spec, inp.structure)
        seen["child"] = ctx.compute(child).results.extras["blas"]
        seen["after child"] = blas_threads()
        with ctx.session(spec, inp.structure, threads=2) as s:
            seen["session"] = s.compute().record["blas"]
        seen["after session"] = blas_threads()
        own = inp.model_copy(update={"program": "toy", "subprograms": []})
        out = super().compute(own, ctx)
        results = out.results.model_copy(update={"extras": seen})
        return out.model_copy(update={"input_data": inp, "results": results})


def test_a_subprogram_holds_its_own_threads_and_gives_its_parent_its_back(pools):
    """A parent's own pools hold one thread: its grant is its children's, each of which
    holds its own and gives the parent's back; a program may raise its own between."""
    register_program(Pinning())
    outside = blas_threads()
    child = {"program": "pools", "calctype": "gradient", "model": {"method": "lj"}}
    inp = ProgramInput.model_validate(
        toy_input(program="pinning", threads=4).model_dump()
        | {"subprograms": [child | {"keywords": {"threads": 3}}]}
    )
    with Runner(threads=8) as runner:
        seen = runner.compute(inp).results.extras
    assert seen == {
        "granted": 4,
        "held": 1,
        "own": 4,
        "child": 3,
        "after child": 1,
        "session": 2,
        "after session": 1,
    }
    assert blas_threads() == outside


def test_overlapping_holds_restore_the_counts_from_before():
    """The latest running call holds a process-wide count; the last out restores."""
    from qccompute._pools import hold_threads

    outside = blas_threads()
    a, b, c = hold_threads(4), hold_threads(1), hold_threads(2)
    a.__enter__()
    b.__enter__()
    c.__enter__()
    assert blas_threads() == 2
    b.__exit__()  # c, which started later, still holds the count
    assert blas_threads() == 2
    c.__exit__()  # gives back what it found
    assert blas_threads() == 1
    a.__exit__()  # the last out: the outside, though a's own was never 1
    assert blas_threads() == outside
    for n in (outside, 1):  # equal to the outside count, the last out still checks
        first, second = hold_threads(n), hold_threads(5)
        first.__enter__()
        second.__enter__()
        first.__exit__()
        second.__exit__()
        assert blas_threads() == outside


def _openblas_spins() -> bool:
    """Whether numpy's BLAS is OpenBLAS on its own threads, which spin when idle."""
    from threadpoolctl import threadpool_info

    return sys.platform == "linux" and any(
        i["internal_api"] == "openblas" and i.get("threading_layer") == "pthreads"
        for i in threadpool_info()
    )


@pytest.mark.parametrize("route", ["in process", "worker", "worker, child in client"])
def test_threads_in_use_never_exceed_the_grant_children_included(pools, route):
    """A calculation whose child takes its whole grant keeps no threads of its own
    beside it, on every route; submitted ones that each take it never overlap."""
    for program in (serve_programs.orchestrator, serve_programs.census):
        register_program(program)
    if route == "worker, child in client":  # not the worker's: the client runs it
        register_program(serve_programs.other_census)
    child = {"program": "census", "calctype": "gradient", "model": {"method": "lj"}}
    inp = ProgramInput.model_validate(
        toy_input(program="orchestrator").model_dump() | {"subprograms": [child]}
    )
    isolation = "auto" if route == "in process" else "process"
    with Runner(threads=4, isolation=isolation) as runner:
        futures = [runner.submit(inp) for _ in range(3)]
        outs = [f.result(timeout=60) for f in futures]
    spins = _openblas_spins()
    for out in outs:
        extras = out.results.extras
        assert out.execution.threads == 4  # it asked for all of them
        assert extras["blas"] == 1  # its own pools: one thread, the grant its child's
        for seen in extras["children"]:
            assert seen["threads"] == seen["blas"] == 4
            if route == "worker, child in client":
                assert seen["pid"] != extras["pid"]
                continue
            assert seen["pid"] == extras["pid"]
            if spins:  # its parent's last step left no BLAS thread spinning
                assert seen["busy"] == 0
    spans = sorted(out.results.extras["span"] for out in outs)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_overlapping_calculations_in_threads_leave_the_counts_as_they_were(pools):
    outside = blas_threads()
    with Runner(threads=8) as runner:
        inputs = [toy_input(program="pools", threads=1 + i % 3) for i in range(24)]
        outs = runner.compute(inputs)
    assert all(o.success for o in outs)
    assert blas_threads() == outside


# ------------------------------------------------------------- the protocol's rules


def test_programs_with_subprograms_called_directly_go_to_the_default_runner():
    fd = FiniteDifference()
    with pytest.raises(ProgramError) as e:
        fd.compute(fd_input())  # not registered: the runner cannot find it
    assert e.value.error.kind is ErrorKind.environment
    assert "register_program" in e.value.error.message
    register_program(fd)
    out = fd.compute(fd_input())
    assert out.success and out.execution.threads == default_runner().threads


def test_the_runner_attaches_the_output_a_program_raised_without():
    class Bare(ToyProgram):
        name = "bare"

        def compute(self, inp, ctx=None):
            raise ProgramError.of("convergence", "no output attached")

    class Whole(ToyProgram):
        name = "whole"

        def compute(self, inp, ctx=None):
            out = ProgramOutput.failure(
                inp,
                ProgramError.of("convergence", "x").error,
                record={"scf": {"iterations": 99}},
            )
            raise ProgramError(out.error, output=out)

    register_program(Bare())
    register_program(Whole())
    with pytest.raises(ProgramError) as e:
        qccompute.compute(toy_input(program="bare"))
    assert e.value.output is not None and e.value.output.error.kind == "convergence"
    assert e.value.output.execution.wall_time is not None
    with pytest.raises(ProgramError) as e:
        qccompute.compute(toy_input(program="whole"))
    assert e.value.output.record == {"scf": {"iterations": 99}}  # the program's own


def test_function_programs():
    from qccompute.program import FunctionProgram
    from qccompute.testing import check_program

    def harmonic(x):
        return float(np.sum(x**2)), 2 * x

    def hessian(x):
        return 2 * np.eye(x.size)

    program = FunctionProgram("harmonic", harmonic, hessian, version="1")
    inp = toy_input("hessian", program="harmonic").model_copy(update={"model": None})
    target = inp.model_copy(update={"keywords": {"accuracy": {"hessian": 1e-4}}})
    assert "honours ctx.threads" in check_program(program, [inp, target])
    register_program(program)
    out = qccompute.compute(inp)
    assert out.results.energy == pytest.approx(np.sum(ARGON3.geometry**2))
    assert np.allclose(out.results.hessian, 2 * np.eye(9))
    with pytest.raises(ProgramError) as e:
        qccompute.compute(inp.model_copy(update={"keywords": {"k": 1}}))
    assert e.value.error.code == "unknown_key"

    def broken(x):
        raise FloatingPointError("overflow")

    register_program(FunctionProgram("broken-fn", broken))
    with pytest.raises(ProgramError) as e:
        qccompute.compute(
            inp.model_copy(update={"program": "broken-fn", "calctype": CalcType.energy})
        )
    assert e.value.error.kind is ErrorKind.internal
    assert "FloatingPointError" in e.value.output.traceback
    with qccompute.session(inp.model_copy(update={"calctype": CalcType.gradient})) as s:
        assert (
            s.compute(np.asarray(ARGON3.geometry) * 2).results.energy
            > out.results.energy
        )


def test_function_programs_may_take_the_input_and_declare_settings():
    """One program for any molecule and setting; threads always accepted (#59)."""
    from qccompute.program import FunctionProgram
    from qccompute.testing import check_program

    seen = []

    def pairs(x, inp):  # a harmonic bond between every pair, per element
        seen.append(inp)
        k = inp.keywords.get("k", 1.0)
        if k <= 0:
            raise ProgramError.of("input", f"k is positive, not {k}", code="bad_k")
        scale = 2.0 if inp.model.method == "stiff" else 1.0
        n = len(inp.structure.symbols)
        energy, gradient = 0.0, np.zeros_like(x)
        for i in range(n):
            for j in range(i + 1, n):
                d = x[i] - x[j]
                energy += scale * k * float(d @ d)
                gradient[i] += 2 * scale * k * d
                gradient[j] -= 2 * scale * k * d
        return energy, gradient

    settings = {
        "type": "object",
        "properties": {"k": {"type": "number", "exclusiveMinimum": 0, "x-unit": "Eh"}},
    }
    program = FunctionProgram(
        "pairs", pairs, version="1", takes_input=True, settings=settings
    )
    inp = toy_input("gradient", program="pairs", k=0.5)
    schema = program.settings_schema(CalcType.gradient)
    assert set(schema["properties"]) == {"k", "accuracy", "threads"}
    assert "honours ctx.threads" in check_program(program, [inp])
    register_program(program)
    out = qccompute.compute(inp)
    assert seen[-1] is inp and out.resolved.settings == {"k": 0.5}
    stiff = qccompute.compute(
        inp.model_copy(update={"model": Model(method="stiff")})
    ).results.energy
    assert stiff == pytest.approx(2 * out.results.energy)
    with pytest.raises(ProgramError) as e:
        qccompute.compute(toy_input(program="pairs", kk=1.0))
    assert e.value.error.code == "unknown_key"
    with pytest.raises(ProgramError) as e:  # the function's own refusal
        qccompute.compute(toy_input(program="pairs", k=-1.0))
    assert e.value.error.code == "bad_k"
    # threads: asked of the runner and recorded; every FunctionProgram takes it.
    with Runner(threads=4) as runner:
        assert (
            runner.compute(toy_input(program="pairs", threads=3)).execution.threads == 3
        )
    with pytest.raises(ProgramError) as e:
        qccompute.compute(toy_input(program="pairs", threads=0))
    assert e.value.error.kind is ErrorKind.input
    plain = FunctionProgram("plain", lambda x: (0.0, 0 * x))
    plain.plan(toy_input(program="plain", threads=2))
    with pytest.raises(ValueError, match="takes_input"):
        FunctionProgram("f", pairs, settings=settings)
    with pytest.raises(ValueError, match="threads"):
        shared = {"properties": {"threads": {"type": "integer"}}}
        FunctionProgram("f", pairs, takes_input=True, settings=shared)


SCALE = 1.0


def scaled(x):
    return SCALE * float(np.sum(x**2)), 2 * SCALE * x


def test_function_program_numerics_change_with_the_function():
    """An edited function (or a value it reads) is new numerics: no stale store hit."""
    from qccompute.program import FunctionProgram

    global SCALE
    store = FakeStore()
    program = FunctionProgram("scaled", scaled)
    register_program(program)
    inp = toy_input("gradient", program="scaled").model_copy(update={"model": None})
    first = qccompute.compute(inp, store=store)
    assert qccompute.compute(inp, store=store) is first  # unchanged: reused
    assert first.results.provenance.program_version == program.version()
    assert program.version().startswith("fn-")
    SCALE = 2.0
    try:
        second = qccompute.compute(inp, store=store)
    finally:
        SCALE = 1.0
    assert second.results.energy == pytest.approx(2 * first.results.energy)
    assert store.puts == 2

    def make(k):
        def f(x):
            return k * float(np.sum(x**2)), 2 * k * x

        return f

    def numerics(*functions, **kw):
        return FunctionProgram("f", *functions, **kw).numerics(inp)

    assert numerics(make(1.0)) == numerics(make(1.0))  # stable
    assert numerics(make(1.0)) != numerics(make(2.0))  # closures count
    assert numerics(lambda x: (1.0, x)) != numerics(lambda x: (2.0, x))  # constants
    assert numerics(make(1.0)) != numerics(make(1.0), lambda x: np.eye(x.size))
    assert numerics(make(1.0), version="v2") == "v2"  # an explicit override


def test_counts_hand_out_threads_and_whole_gpus_in_turn_and_memory_follows():
    from qccompute._count import Count

    count = Count(8, ["0", "1"], memory=64.0)
    assert count.share(2, ()) == 0.25 and count.share(1, ("0",)) == 0.5
    held = [count.take(1, 1), count.take(1, 1)]  # a GPU each, whole
    assert sorted(d for devices in held for d in devices) == ["0", "1"]
    late: list[tuple[str, ...]] = []
    waiting = threading.Thread(target=lambda: late.append(count.take(1, 1)))
    waiting.start()
    waiting.join(0.1)
    assert not late  # no GPU free, though threads are
    behind: list[tuple[str, ...]] = []
    cpu = threading.Thread(target=lambda: behind.append(count.take(1)))
    cpu.start()
    cpu.join(0.1)
    assert not behind  # first come, first served: it waits its turn
    count.give(1, held[0])
    waiting.join(5)
    cpu.join(5)
    assert late == [held[0]] and behind == [()]

    recording = Recording("rec")
    register_program(recording)
    with Runner(threads=4, gpus=()) as runner:
        runner.compute([toy_input(program="rec", threads=t) for t in (1, 3)])
        memory = runner._host_memory()
    if memory is not None:
        by_threads = {c.threads: c.memory_gib for c in recording.contexts}
        assert by_threads[1] == pytest.approx(memory / 4, rel=1e-3)
        assert by_threads[3] == pytest.approx(3 * memory / 4, rel=1e-3)


def test_memory_keywords_replace_the_share():
    """memory_gib and device_memory_gib are asked for and granted, like threads (#46)."""

    class Memory(NoSessions):  # takes the memory keywords, plans one GPU
        def _own(self, inp):
            keywords = {
                k: v
                for k, v in inp.keywords.items()
                if k not in ("memory_gib", "device_memory_gib")
            }
            return inp.model_copy(update={"keywords": keywords})

        def plan(self, inp):
            plan = super().plan(self._own(inp).model_copy(update={"program": "toy"}))
            return plan.model_copy(update={"resources": Resources(gpus=1)})

        def compute(self, inp, ctx=None):
            return super().compute(self._own(inp), ctx)

    memory = Memory("memory")
    register_program(memory)
    with Runner(threads=4, gpus=["0", "1"]) as runner:
        runner._gpus = {"0": 40.0, "1": 40.0}  # as nvidia-smi would list them
        runner._memory = 64.0
        given = {"memory_gib": 30, "device_memory_gib": 32.5}
        runner.compute(
            [toy_input(program="memory", epsilon=e, **given) for e in (0.1, 0.2)]
            + [toy_input(program="memory", epsilon=e) for e in (0.3, 0.4)]
        )
        # A GPU each, whole: half the runner's memory follows it.
        granted = sorted((c.memory_gib, c.device_memory_gib) for c in memory.contexts)
        assert granted == [(30.0, 32.5), (30.0, 32.5), (32.0, 40.0), (32.0, 40.0)]
        del memory.contexts[:]
        # At most what the runner has: the machine's memory, the GPU's.
        too_much = {"memory_gib": 100, "device_memory_gib": 100}
        runner.compute(toy_input(program="memory", **too_much))
        with runner.session(toy_input(program="memory", memory_gib=8)) as s:
            s.compute()
        assert [(c.memory_gib, c.device_memory_gib) for c in memory.contexts] == [
            (64.0, 40.0),
            (8.0, 40.0),
        ]

    # A subprogram's keywords are capped at its parent's grant, which holds them.
    class Parent(ToyProgram):
        name = "memory-parent"

        def capabilities(self):
            caps = super().capabilities()
            return caps.model_copy(update={"batches": False, "sessions": False})

        def _toy(self, inp):
            return inp.model_copy(update={"program": "toy", "subprograms": []})

        def plan(self, inp):
            plan = super().plan(self._toy(inp))
            return plan.model_copy(update={"resources": Resources(gpus=1)})

        def compute(self, inp, ctx=None):
            child = self._toy(inp).model_copy(update={"program": "memory"})
            asks = {"memory_gib": 50, "device_memory_gib": 30}
            ctx.compute(child.model_copy(update={"keywords": asks}))
            ctx.compute([child, child])
            return (
                super()
                .compute(self._toy(inp), ctx)
                .model_copy(update={"input_data": inp})
            )

    register_program(Parent())
    del memory.contexts[:]
    child = {"program": "memory", "calctype": "gradient", "model": {"method": "lj"}}
    with Runner(threads=4, gpus=["0", "1"]) as runner:
        runner._gpus = {"0": 40.0, "1": 40.0}
        runner._memory = 64.0
        # Each parent: a GPU of two, so 32 GiB, and its GPU's 40 GiB.
        runner.compute(
            [
                toy_input(program="memory-parent", epsilon=e).model_copy(
                    update={"subprograms": [ProgramSpec.model_validate(child)]}
                )
                for e in (1, 2)
            ]
        )
    granted = sorted((c.memory_gib, c.device_memory_gib) for c in memory.contexts)
    assert granted == [(32.0, 30.0)] * 2 + [(32.0, 40.0)] * 4


def test_memory_reaches_the_program_unrounded():
    """A keyword as given, a share as computed: never rounded, never to zero (#69)."""

    class Asking(Recording):  # takes the memory_gib keyword
        def _own(self, inp):
            keywords = {k: v for k, v in inp.keywords.items() if k != "memory_gib"}
            return inp.model_copy(update={"keywords": keywords})

        def plan(self, inp):
            return super().plan(self._own(inp).model_copy(update={"program": "toy"}))

        def compute(self, inp, ctx=None):
            return super().compute(self._own(inp), ctx)

    recording = Asking("rec")
    register_program(recording)
    with Runner(threads=8, gpus=()) as runner:
        runner._memory = 0.004
        runner.compute(toy_input(program="rec", memory_gib=0.0034))
        runner.compute(toy_input(program="rec", memory_gib=0.0004))
        runner.compute([toy_input(program="rec", epsilon=0.1 * i) for i in range(1, 9)])
    granted = [c.memory_gib for c in recording.contexts]
    assert granted[:2] == [0.0034, 0.0004]
    assert granted[2:] == [0.004 / 8] * 8  # 0.0005 each


def test_a_threads_keyword_above_the_count_is_lowered_with_a_warning(monkeypatch):
    """The count caps a keyword at top level too, and says so (OMP_NUM_THREADS=1)."""
    recording = Recording("rec")
    register_program(recording)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with Runner(gpus=()) as runner:
        with pytest.warns(RuntimeWarning, match=r"threads=8 and runs on 1.*OPENBLAS"):
            runner.compute(toy_input(program="rec", threads=8))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            runner.compute(toy_input(program="rec", threads=1))
    assert [c.threads for c in recording.contexts] == [1, 1]


def test_gpus_come_from_the_plan():
    """A calculation gets the GPUs its plan asks for (#40); no capability says (a4)."""

    class Planned(NoSessions):  # its sessions run through compute
        def __init__(self, name, gpus):
            super().__init__(name)
            self._wanted = gpus

        def plan(self, inp):
            plan = super().plan(inp.model_copy(update={"program": "toy"}))
            return plan.model_copy(update={"resources": Resources(gpus=self._wanted)})

    one = Planned("one-gpu", 1)  # a parent of GPU children, say
    none = Planned("no-gpu", 0)  # a GPU program on the CPU this time
    register_program(one)
    register_program(none)

    class Parent(FiniteDifference):
        name = "gpu-parent"

        def plan(self, inp):  # its child's GPU, as qcopt plans its children's
            return Plan(resolved=Resolved(), resources=Resources(gpus=1))

    register_program(Parent())
    register_program(FiniteDifference())
    with Runner(threads=4, gpus=["0", "1"]) as runner:
        runner.compute(toy_input(program="one-gpu"))
        runner.compute(toy_input(program="no-gpu"))
        with runner.session(toy_input(program="one-gpu")) as s:
            s.compute()
        assert [c.gpus for c in one.contexts] == [("0",), ("0",)]
        assert [c.gpus for c in none.contexts] == [()]
        # Given a GPU and no threads, 1 thread; on the CPU, the whole count.
        assert [c.threads for c in one.contexts] == [1, 1]
        assert [c.threads for c in none.contexts] == [4]
        runner.compute(toy_input(program="one-gpu", threads=3))
        assert one.contexts[-1].threads == 3
        # A list packs by them: two one-GPU calculations side by side, a GPU each
        del one.contexts[:]
        runner.compute([toy_input(program="one-gpu", epsilon=e) for e in (0.1, 0.2)])
        assert sorted(c.gpus for c in one.contexts) == [("0",), ("1",)]
        assert one.most == 2
        # A parent passes its grant on to its children. It runs no GPU work itself, so
        # it gets the whole count, and a child's own threads hold under it.
        del one.contexts[:]
        parent = fd_input("one-gpu", threads=3).model_copy(
            update={"program": "gpu-parent"}
        )
        out = runner.compute(parent)
        assert {c.gpus for c in one.contexts} == {("0",)}
        assert out.execution.threads == 4
        assert one.contexts[-1].threads == 3  # the center point: its keyword
        del one.contexts[:]
        runner.compute(fd_input("one-gpu"))  # a parent that plans none has none
        assert {c.gpus for c in one.contexts} == {()}


def test_workers_with_gpus_are_shared_by_sessions_and_calls():
    """One worker per program and GPUs; without GPUs, as many as fit."""
    register_program(serve_programs.isolated_gpu)
    register_program(serve_programs.isolated)
    gpu = "isolated-gpu"

    def pids(runner, program, n):
        one = toy_input(program=program)
        sessions = [runner.session(one, threads=1) for _ in range(n)]
        seen = [s.compute().record["pid"] for s in sessions]
        return sessions, seen

    with Runner(threads=4, gpus=["7"]) as runner:
        sessions, seen = pids(runner, gpu, 3)
        assert len(set(seen)) == 1  # one worker hosts all three sessions
        out = runner.compute(toy_input(program=gpu))  # a call beside them
        batch = runner.compute(  # a batch with 3 threads: the same worker
            [toy_input(program=gpu, epsilon=0.01 * (i + 1)) for i in range(3)]
        )
        assert {o.results.extras["pid"] for o in [out, *batch]} == set(seen)
        assert out.results.extras["devices"] == "7"  # it sees its grant
        assert out.results.extras["ordinals"] == [0]  # as CUDA device 0 (#49)
        assert [s.compute().success for s in sessions] == [True] * 3
        for s in sessions:
            s.close()
    # Without GPUs: a worker per session, as many as fit side by side, then shared.
    with Runner(threads=2, gpus=["7"]) as runner:
        sessions, seen = pids(runner, "isolated", 3)
        assert len(set(seen[:2])) == 2 and seen[2] in seen[:2]
        out = runner.compute(toy_input(program="isolated"))
        assert out.results.extras["devices"] == ""  # it sees no GPU: none granted
        assert out.results.extras["ordinals"] == []
        for s in sessions:
            s.close()


def test_workers_without_gpus_serve_calls_of_any_threads():
    """A list's batches of different sizes share the program's worker (alpha 4)."""
    register_program(serve_programs.isolated)
    with Runner(threads=4, gpus=()) as runner:
        three = runner.compute(
            [toy_input(program="isolated", epsilon=e) for e in (1, 2, 3)]
        )
        two = runner.compute([toy_input(program="isolated", epsilon=e) for e in (4, 5)])
        one = runner.compute(toy_input(program="isolated", threads=1))
    assert len({o.results.extras["pid"] for o in [*three, *two, one]}) == 1
    # Each call's threads reach the worker in its ctx.
    assert [o.execution.threads for o in [*three, *two, one]] == [3, 3, 3, 4, 4, 1]


def test_subprograms_start_only_within_the_calculations_time():
    """In process a timeout is the program's to honour; after it, no subprogram (#19)."""
    kinds = []

    class Slow(FiniteDifference):
        name = "slow"

        def compute(self, inp, ctx=None):
            child = ProgramInput.from_spec(inp.subprograms[0], inp.structure)
            session = ctx.session(child)  # opened in time
            session.compute()
            time.sleep(0.3)  # past its time: the runner cannot stop it
            for call in (
                lambda: ctx.compute(child),
                lambda: ctx.compute([child, child]),
                session.compute,
                lambda: session.compute_many([Request(), Request()]),
                lambda: ctx.session(child),
            ):
                try:
                    call()
                except ProgramErrors as group:
                    kinds.append([e.error.kind for e in group.exceptions])
                except ProgramError as e:
                    kinds.append(e.error.kind)
            return super().compute(inp, ctx)  # its subprograms fail, so it does

    register_program(Slow())
    register_program(FiniteDifference())
    timeout = ErrorKind.timeout
    with Runner(threads=2) as runner:
        with pytest.raises(ProgramError) as e:
            runner.compute(
                fd_input().model_copy(update={"program": "slow"}), timeout=0.2
            )
        assert e.value.error.kind is ErrorKind.program
        assert e.value.error.cause.kind is timeout
        assert kinds == [timeout, [timeout] * 2, timeout, [timeout] * 2, timeout]
        assert runner.compute(fd_input(), timeout=5).success  # within its time
