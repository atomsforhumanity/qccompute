"""The Runner: one path for one input or many, failures that raise, ctx, the store hook."""

import os
import threading
import time
from typing import Any

import numpy as np
import pytest
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

    def __init__(self, name: str, thread_safe: bool = True, gpu: bool = False):
        self.name = name
        self._thread_safe = thread_safe
        self._gpu = gpu
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
                    "gpu": self._gpu,
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

    def put(self, output, identities, *, tags=(), meta=None):
        key = self.key(output.input_data)
        self.runs[key, tuple(sorted(identities.items()))] = output
        self.meta[key] = meta
        self.puts += 1
        self.tag(key, *tags)
        return key

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
    assert out.execution.threads == 1  # the toy's plan asks for 1, not the machine
    assert "total" in out.execution.timings


def test_a_plan_may_ask_for_every_thread():
    """Resources(threads=None): all of the runner's threads, or of the parent's grant."""
    spans: list[tuple[str, float, float]] = []

    class Asks(NoSessions):
        def __init__(self, name, threads):
            super().__init__(name)
            self._asks = threads

        def plan(self, inp):
            plan = super().plan(inp.model_copy(update={"program": "toy"}))
            return plan.model_copy(update={"resources": Resources(threads=self._asks)})

        def compute(self, inp, ctx=None):
            start = time.perf_counter()
            out = super().compute(inp, ctx)
            spans.append((self.name, start, time.perf_counter()))
            return out

    every, one = Asks("every", None), Asks("one", 1)
    register_program(every)
    register_program(one)
    with Runner(threads=4) as runner:
        runner.compute(toy_input(program="every"))  # a single input: all 4
        runner.compute(toy_input(program="every", threads=2))  # the keyword wins
        with runner.session(toy_input(program="every")) as s:  # a session: all 4
            s.compute()
        assert [c.threads for c in every.contexts] == [4, 2, 4]
        # In a list it is a full share: it runs alone, the 1-thread ones side by side.
        spans.clear()
        inputs = [toy_input(program="one", epsilon=0.01 * (i + 1)) for i in range(6)]
        inputs.insert(3, toy_input(program="every"))
        assert all(o.success for o in runner.compute(inputs))
        assert one.most > 1
        (_, start, end), *_ = (span for span in spans if span[0] == "every")
        others = [span for span in spans if span[0] == "one"]
        assert all(e <= start or s >= end for _, s, e in others)

    class Parent(FiniteDifference):
        """Asks for 3 threads; its subprogram asks for all of them."""

        name = "parent3"

        def plan(self, inp):
            return Plan(resolved=Resolved(), resources=Resources(threads=3))

        def compute(self, inp, ctx=None):
            child = inp.subprograms[0]
            ctx.compute(ProgramInput.from_spec(child, inp.structure))
            with ctx.session(child, inp.structure) as session:
                session.compute()
            return super().compute(inp, ctx)

    register_program(Parent())
    del every.contexts[:]
    with Runner(threads=8) as runner:
        runner.compute(fd_input("every").model_copy(update={"program": "parent3"}))
    # compute and session get the parent's whole grant; fd's own session asks for 1
    assert [c.threads for c in every.contexts][:2] == [3, 3]


def test_threads_come_from_the_keyword_else_the_plan_capped_at_the_runner():
    class Planned(Recording):
        def plan(self, inp):
            plan = super().plan(inp.model_copy(update={"program": "toy"}))
            return plan.model_copy(update={"resources": Resources(threads=3)})

    recording = Recording("rec")
    planned = Planned("planned")
    register_program(recording)
    register_program(planned)
    with Runner(threads=4) as runner:
        runner.compute(toy_input(program="rec"))  # nothing asked: 1
        runner.compute(toy_input(program="rec", threads=2))  # the keyword
        runner.compute(toy_input(program="rec", threads=64))  # capped at the runner
        runner.compute(toy_input(program="planned"))  # the plan
        runner.compute(toy_input(program="planned", threads=1))  # the keyword wins
        with runner.session(toy_input(program="rec")) as s:  # sessions alike
            s.compute()
        with runner.session(toy_input(program="planned")) as s:
            s.compute()
    assert [c.threads for c in recording.contexts] == [1, 2, 4]
    assert [c.threads for c in planned.contexts] == [3, 1]
    assert all(c.plan is not None for c in recording.contexts)  # handed on
    plain = NoSessions("plain")
    register_program(plain)
    with Runner(threads=4) as runner, runner.session(toy_input(program="plain")) as s:
        s.compute_many([Request(), Request()])
    assert [c.threads for c in plain.contexts] == [1, 1]
    with pytest.raises(ProgramError) as e:  # refused by its plan, without running
        Runner(threads=4).compute(toy_input(program="rec", sigmaa=1.0))
    assert e.value.error.code == "unknown_key" and len(recording.contexts) == 3
    assert e.value.output.execution.wall_time == 0.0

    class Parent(FiniteDifference):
        """Asks for 4 threads; its subprograms ask for their own."""

        name = "parent"

        def plan(self, inp):
            return Plan(resolved=Resolved(), resources=Resources(threads=4))

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
    with Runner(threads=8) as runner:
        runner.compute(fd_input("plain").model_copy(update={"program": "parent"}))
    # compute and session ask for 1; the parent's split gives 3; fd's session 1
    assert [c.threads for c in plain.contexts][:3] == [1, 1, 3]


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
    with pytest.raises(ValueError, match="backend"):
        Runner(backend="slurm")  # type: ignore[arg-type]
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
    outs = Runner(threads=4).compute([toy_input(program="serial") for _ in range(4)])
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


def test_as_completed_yields_successes_then_raises_the_group():
    runner = Runner(threads=2)
    inputs = [toy_input(epsilon=0.01 * (i + 1)) for i in range(4)]
    seen = sorted(o.input_data.keywords["epsilon"] for o in runner.as_completed(inputs))
    assert seen == [0.01, 0.02, 0.03, 0.04]
    yielded = []
    with pytest.raises(ProgramErrors) as e:
        for out in runner.as_completed([toy_input(), toy_input(fail="input")]):
            yielded.append(out)
    assert len(yielded) == 1 and yielded[0].success
    assert len(e.value.results) == 2
    every = list(
        runner.as_completed([toy_input(), toy_input(fail="input")], raise_exc=False)
    )
    assert sorted(o.success for o in every) == [False, True]


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
    """N 1-thread inputs as one batch get N threads, at most the runner's (#39)."""
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
        runner.compute(batch(8))  # each asks for 1
        runner.compute(batch(3))
        runner.compute(batch(20))  # at most the runner's
        runner.compute(batch(3, threads=2))  # the keyword, summed
        runner.compute(batch(2) + [toy_input(), toy_input()])  # beside other tasks
    assert granted == [(8, 8), (3, 3), (20, 8), (3, 6), (2, 2)]


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
    qccompute.compute(toy_input(program="warm"), store=store)
    qccompute.compute([toy_input(program="warm"), toy_input()], store=store)
    assert store.puts == 1  # only the toy's own run


# ---------------------------------------------------------------------- sessions


def test_sessions_are_the_programs_own_in_process():
    with qccompute.session(toy_input()) as s:
        assert type(s) is ToySession
        r = s.compute(np.asarray(ARGON3.geometry)).results
        assert r.gradient.shape == (3, 3)
    register_program(NoSessions("plain"))
    with qccompute.session(toy_input(program="plain")) as s:
        assert isinstance(s, ComputeSession)
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


@pytest.mark.parametrize("thread_safe", [False, True])
def test_sessions_may_be_called_from_threads(thread_safe):
    """The runner makes concurrent session calls safe; callers need not ask."""
    from concurrent.futures import ThreadPoolExecutor

    program = NoSessions(f"exclusive-{thread_safe}", thread_safe)
    register_program(program)
    sessions = [qccompute.session(toy_input(program=program.name)) for _ in range(4)]
    with ThreadPoolExecutor(4) as pool:
        frames = list(pool.map(lambda s: s.compute(), sessions))
    assert all(f.success for f in frames)
    assert program.most > 1 if thread_safe else program.most == 1


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


def test_subprogram_lists_and_their_failures():
    seen = []

    class Fanout(ToyProgram):
        name = "fanout"

        def plan(self, inp):
            return Plan(resolved=Resolved(), resources=Resources(threads=2))

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
        # The toy batches: the four children ask for 4 threads together, capped at
        # the parent's 2, and the toy runs them one after another with both.
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
            assert type(s).__name__ == "WorkerSession"
            assert s.compute().success


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
        assert out.results.extras["threads"] == 1  # what the child asks for
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


# ------------------------------------------------------------- the protocol's rules


def test_programs_with_subprograms_called_directly_go_to_the_default_runner():
    fd = FiniteDifference()
    with pytest.raises(ProgramError) as e:
        fd.compute(fd_input())  # not registered: the runner cannot find it
    assert e.value.error.kind is ErrorKind.environment
    assert "register_program" in e.value.error.message
    register_program(fd)
    out = fd.compute(fd_input())
    assert out.success and out.execution.threads == 1  # what fd's plan asks for


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
    assert "honours ctx.threads" in check_program(program, [inp])
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


def test_shares_pack_threads_and_gpus_and_memory_follows():
    from qccompute.runner import _Grant

    budget = _Grant(8, ["0", "1"], per_gpu=2)
    assert budget.share(2, 0) == 0.25
    assert budget.share(1, 1) == 0.25  # half a GPU of two
    assert budget.share(8, 1) == 1.0
    held = [budget.acquire(i, 1, 1) for i in range(4)]  # four quarter shares
    assert sorted(d for _, devices, _ in held for d in devices) == ["0", "0", "1", "1"]
    late = []
    t = threading.Thread(target=lambda: late.append(budget.acquire(4, 1, 0)))
    t.start()
    t.join(0.1)
    assert not late  # the shares already sum to 1
    budget.release(*held[0][1:])
    t.join(5)
    assert late and late[0][2] == 0.125

    recording = Recording("rec")
    register_program(recording)
    with Runner(threads=4, gpus=()) as runner:
        runner.compute([toy_input(program="rec", threads=t) for t in (1, 3)])
        memory = runner._host_memory()
    if memory is not None:
        by_threads = {c.threads: c.memory_gib for c in recording.contexts}
        assert by_threads[1] == pytest.approx(memory / 4, rel=1e-3)
        assert by_threads[3] == pytest.approx(3 * memory / 4, rel=1e-3)
