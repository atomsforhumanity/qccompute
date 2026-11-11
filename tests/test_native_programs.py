"""Programs implementing qcdata.program.Program run without a codec."""

from typing import Any

import numpy as np
import pytest
from qcdata import (
    CalcType,
    ErrorKind,
    ExecutionInfo,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    Resolved,
    SinglePointData,
    Structure,
)
from qcdata.program import (
    Capabilities,
    ComputeSession,
    Plan,
    ProgramError,
    Request,
    Resources,
    RunContext,
)
from qcdata.testing.toy import ToyProgram, ToySession

import qccompute
from qccompute import programs
from qccompute.exceptions import ComputeErrors, ExternalProgramError, InputError

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

    def capabilities(self) -> Capabilities:
        return (
            super().capabilities().model_copy(update={"thread_safe": self._thread_safe})
        )

    def compute(self, inp, ctx):
        self.grants.append((ctx.threads, ctx.gpus))
        assert ctx.scratch is not None and ctx.scratch.is_dir()
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
                requests.append(Request((x + d).reshape(-1, 3), CalcType.energy))
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
    programs._registered.clear()


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


def test_lists_of_other_programs_run_one_at_a_time():
    serial = Recording("serial", thread_safe=False)
    programs.register_program(serial)
    outs = qccompute.compute([toy_input(program="serial") for _ in range(3)])
    assert all(o.success for o in outs)
    assert {g[0] for g in serial.grants} == {None}  # not granted: its own settings


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
