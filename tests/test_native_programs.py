"""Programs implementing qccompute.program.Program run without a codec."""

from typing import Any

import numpy as np
import pytest
from qcdata import (
    CalcType,
    ErrorKind,
    ExecutionInfo,
    ProgramError,
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
from qccompute import programs
from qccompute.exceptions import ComputeErrors, ExternalProgramError, InputError
from qccompute.program import (
    Capabilities,
    ComputeSession,
    Plan,
    Resources,
    RunContext,
    available_programs,
    unregister_program,
)
from qccompute.testing.toy import ToyProgram, ToySession

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
    """The toy, recording each run's grant; optionally not thread-safe."""

    def __init__(self, name: str, thread_safe: bool = True):
        self.name = name
        self._thread_safe = thread_safe
        self.grants: list[tuple[int | None, tuple[str, ...]]] = []
        self.scratches: list = []

    def capabilities(self) -> Capabilities:
        return (
            super().capabilities().model_copy(update={"thread_safe": self._thread_safe})
        )

    def compute(self, inp, ctx):
        self.grants.append((ctx.threads, ctx.gpus))
        assert ctx.scratch_dir is None  # nothing made until asked for
        assert ctx.scratch is not None and ctx.scratch.is_dir()
        self.scratches.append(ctx.scratch)
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

    def compute(self, inp, ctx):
        raise RuntimeError("a bug")


class FiniteDifference(ToyProgram):
    """A driver: the gradient of its subprogram by central differences, via a session."""

    name = "fd"

    def plan(self, inp):
        return Plan(resolved=Resolved(settings={"step": 1e-4}), resources=Resources())

    def compute(self, inp, ctx):
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
            results = s.compute_many(requests)
        energies = np.array([r.results.energy for r in results])
        gradient = (energies[0::2] - energies[1::2]) / (2 * step)
        center = ctx.compute(ProgramInput.from_spec(child, inp.structure))
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=SinglePointData(
                provenance=Provenance(program="fd", program_version="0"),
                energy=center.results.energy,
                gradient=gradient,
            ),
            resolved=self.plan(inp).resolved,
            execution=ExecutionInfo(timings={"total": 0.0}),
        )


@pytest.fixture(autouse=True)
def toy_registered():
    programs.register_program(ToyProgram())
    yield
    for name in available_programs():
        unregister_program(name)  # installed programs stay


def test_native_programs_run_without_a_codec():
    out = qccompute.compute(toy_input("hessian", sigma="auto"))
    assert out.success
    assert out.resolved.settings["sigma"] == 6.0
    assert out.results.hessian.shape == (9, 9)
    assert out.results.gradient.shape == (3, 3)
    assert out.execution.wall_time is not None and out.execution.hostname
    assert "total" in out.execution.timings


def test_native_failures_raise_or_return_with_kinds():
    with pytest.raises(InputError, match="Did you mean 'sigma'") as e:
        qccompute.compute(toy_input(sigmaa=1.0))
    assert e.value.prog_output.error.kind is ErrorKind.input
    with pytest.raises(ExternalProgramError) as e:
        qccompute.compute(toy_input(fail="convergence"))
    assert e.value.prog_output.error.kind is ErrorKind.convergence
    out = qccompute.compute(toy_input(fail="resource"), raise_exc=False)
    assert not out.success and out.error.kind is ErrorKind.resource


def test_program_bugs_become_internal_errors():
    programs.register_program(Broken())
    out = qccompute.compute(toy_input(program="broken"), raise_exc=False)
    assert not out.success
    assert out.error.kind is ErrorKind.internal
    assert "RuntimeError: a bug" in out.traceback


def test_codec_only_options_are_rejected():
    with pytest.raises(InputError, match="does not support timeout"):
        qccompute.compute(toy_input(), timeout=10)
    out = qccompute.compute(toy_input(), collect_files=True, raise_exc=False)
    assert out.error.kind is ErrorKind.input


def test_lists_of_thread_safe_programs_run_with_grants():
    recording = Recording("rec")
    programs.register_program(recording)
    inputs = [
        toy_input(program="rec", threads=2, epsilon=0.01 * (i + 1)) for i in range(6)
    ]
    outs = qccompute.compute(inputs, max_workers=4)
    assert [o.input_data.keywords["epsilon"] for o in outs] == [
        i.keywords["epsilon"] for i in inputs
    ]
    assert all(o.success for o in outs)
    assert {g[0] for g in recording.grants} == {2}
    assert not any(p.exists() for p in recording.scratches)  # removed afterwards


def test_lists_of_other_programs_run_one_at_a_time():
    serial = Recording("serial", thread_safe=False)
    programs.register_program(serial)
    outs = qccompute.compute([toy_input(program="serial") for _ in range(3)])
    assert all(o.success for o in outs)
    assert {g[0] for g in serial.grants} == {None}  # not granted: its own settings


def test_native_lists_stay_in_this_process(monkeypatch):
    """Program state (warm starts, GPU contexts) lives in the caller's process, so a
    top-level list of native programs never starts worker processes. axtb's study on
    the codec path: worker rounds lost every warm start and paid about 0.25 s each."""
    from qccompute import parallel

    def no_workers(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a list of native programs started a worker pool")

    monkeypatch.setattr(parallel, "WorkerPool", no_workers)
    programs.register_program(Recording("serial", thread_safe=False))
    inputs = [toy_input(), toy_input(program="serial"), toy_input(), toy_input()]
    outs = qccompute.compute(inputs, max_workers=4)
    assert all(o.success for o in outs)


def test_list_failures_are_grouped():
    with pytest.raises(ComputeErrors) as e:
        qccompute.compute([toy_input(), toy_input(fail="numerical")])
    assert len(e.value.exceptions) == 1
    outs = qccompute.compute(
        [toy_input(), toy_input(fail="numerical")], raise_exc=False
    )
    assert [o.success for o in outs] == [True, False]


def test_sessions_native_and_over_compute():
    spec = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})
    with qccompute.session(spec, ARGON3) as s:
        assert isinstance(s, ToySession)
        r = s.compute(np.asarray(ARGON3.geometry)).results
        assert r.gradient.shape == (3, 3)
    programs.register_program(NoSessions("plain"))
    plain = ProgramSpec(program="plain", calctype="gradient", model={"method": "lj"})
    with qccompute.session(plain, ARGON3) as s:
        assert isinstance(s, ComputeSession)
        r2 = s.compute(np.asarray(ARGON3.geometry)).results
        assert r2.energy == pytest.approx(r.energy)
        with pytest.raises(ProgramError) as e:
            s.compute(np.full((3, 3), np.nan))
        assert e.value.error.kind is ErrorKind.numerical
        assert s.compute(np.asarray(ARGON3.geometry)).results.energy == r2.energy


@pytest.mark.parametrize("child", ["toy", "plain"])
def test_drivers_get_child_sessions_and_compute(child):
    programs.register_program(FiniteDifference())
    programs.register_program(NoSessions("plain"))
    inp = ProgramInput.model_validate(
        {
            "program": "fd",
            "calctype": "gradient",
            "structure": ARGON3,
            "subprograms": [
                {"program": child, "calctype": "energy", "model": {"method": "lj"}}
            ],
        }
    )
    out = qccompute.compute(inp)
    exact = qccompute.compute(toy_input("gradient"))
    assert np.allclose(out.results.gradient, exact.results.gradient, atol=1e-8)
    assert out.results.energy == pytest.approx(exact.results.energy)


def test_registration_checks_the_protocol():
    with pytest.raises(TypeError, match="does not implement"):
        programs.register_program(object())  # type: ignore[arg-type]
    assert qccompute.get_program("toy") is not None
    assert qccompute.get_program("no-such-program") is None


def test_direct_calls_need_no_runner():
    out = ToyProgram().compute(toy_input(), RunContext())
    assert out.success


class Fanout(ToyProgram):
    """A driver that computes a list of children in one call, like a round of a driver."""

    name = "fanout"

    def compute(self, inp, ctx):
        spec = inp.subprograms[0].model_dump()
        children = [
            ProgramInput.model_validate(
                {**spec, "keywords": {"value": i}, "structure": inp.structure}
            )
            for i in range(4)
        ]
        outs = ctx.compute(children)
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=SinglePointData(
                provenance=Provenance(program="fanout", program_version="0"),
                energy=sum(o.results.energy for o in outs),
                extras={"pids": [o.results.extras["pid"] for o in outs]},
            ),
            resolved=Resolved(settings={}),
        )


def test_driver_children_never_go_to_worker_processes(monkeypatch):
    """A driver's rounds are small and frequent: a worker round trip (about 100 ms)
    would dwarf an in-process gradient (qcopt measured 30x on axtb through lists)."""
    import os

    from qccompute import parallel

    def no_workers(*args, **kwargs):
        raise AssertionError("a driver's children started a worker pool")

    monkeypatch.setattr(parallel, "WorkerPool", no_workers)
    import qccodec

    from tests.test_parallel import PYTHON  # picklable: workers could receive it

    qccodec.register(PYTHON)
    try:
        programs.register_program(Fanout())
        inp = ProgramInput.model_validate(
            {
                "program": "fanout",
                "calctype": "energy",
                "structure": ARGON3,
                "subprograms": [{"program": "library", "calctype": "energy"}],
            }
        )
        out = qccompute.compute(inp, max_workers=4)
    finally:
        qccodec.unregister("library")
    assert out.results.energy == 0 + 1 + 2 + 3
    assert set(out.results.extras["pids"]) == {os.getpid()}
