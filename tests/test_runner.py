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


def test_one_input_runs_in_process_with_the_whole_grant():
    out = qccompute.compute(toy_input("hessian", sigma="auto"))
    assert out.success
    assert out.resolved.settings["sigma"] == 6.0
    assert out.results.hessian.shape == (9, 9)
    assert out.results.gradient.shape == (3, 3)
    assert out.execution.wall_time is not None and out.execution.hostname
    assert out.execution.threads == default_runner().threads
    assert "total" in out.execution.timings


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
    assert runner.identities(two)["epochs"] == "lj-1 + x-2"


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
        assert seen == [[1, 1, 1, 1]]  # each child got its planned thread
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
        out = runner.compute(toy_input())
        assert out.success and out.execution.threads == 2
        with runner.session(toy_input()) as s:
            assert type(s).__name__ == "WorkerSession"
            assert s.compute().success


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
