"""The Program/Session protocol, its conformance suite, and the toy program."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qcdata import (
    Accuracy,
    CalcType,
    ErrorKind,
    ProgramError,
    ProgramErrors,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Request,
    Structure,
)

from qccompute.program import (
    BatchProgram,
    Capabilities,
    ComputeSession,
    Program,
    Resources,
    RunContext,
    Session,
    SessionProgram,
    available_programs,
    find_program,
    register_program,
    unregister_program,
)
from qccompute.testing import ConformanceError, check_program
from qccompute.testing.toy import ToyProgram, ToySession, program

ARGON3 = Structure(
    symbols=["Ar", "Ar", "Ar"],
    geometry=np.array([[0.0, 0.0, 0.0], [7.2, 0.0, 0.0], [3.0, 6.1, 0.4]]),
)


def toy_input(calctype: str = "gradient", **keywords: Any) -> ProgramInput:
    return ProgramInput.model_validate(
        {
            "program": "toy",
            "calctype": calctype,
            "structure": ARGON3,
            "model": {"method": "lj"},
            "keywords": keywords,
        }
    )


def test_toy_implements_every_protocol():
    assert isinstance(program, Program)
    assert isinstance(program, SessionProgram)
    assert isinstance(program, BatchProgram)


def test_toy_conforms():
    ran = check_program(
        program,
        [
            toy_input("gradient"),
            toy_input("energy", sigma=6.5, output={"pair_energies": True}),
            toy_input("hessian", epsilon=0.02),
            toy_input("gradient", accuracy={"gradient": 1e-6}),
        ],
    )
    assert "session survives a failed request" in ran
    assert "batch agrees with compute" in ran


def test_toy_gradient_and_hessian_are_derivatives():
    x = np.asarray(ARGON3.geometry)
    out = program.compute(toy_input("hessian"), RunContext())
    g, h = out.results.gradient.reshape(-1), out.results.hessian
    step = 1e-4
    with program.open_session(toy_input(), RunContext()) as s:
        for k in range(9):
            d = np.zeros(9)
            d[k] = step
            plus = s.compute((x.reshape(-1) + d).reshape(-1, 3), "gradient").results
            minus = s.compute((x.reshape(-1) - d).reshape(-1, 3), "gradient").results
            fd_g = (plus.energy - minus.energy) / (2 * step)
            fd_h = (plus.gradient - minus.gradient).reshape(-1) / (2 * step)
            assert fd_g == pytest.approx(g[k], abs=1e-9)
            assert fd_h == pytest.approx(h[k], abs=1e-7)


def test_toy_resolves_auto_and_reports_why():
    out = program.compute(toy_input(sigma="auto"), RunContext())
    assert out.resolved.settings["sigma"] == 6.0
    assert "sigma" in out.resolved.reasons
    assert out.execution.timings["total"] >= 0
    assert sum(out.results.energy_components.values()) == pytest.approx(
        out.results.energy
    )


def test_unknown_keys_get_suggestions_and_typed_errors():
    with pytest.raises(ProgramError, match="Did you mean 'sigma'") as e:
        program.plan(toy_input(sigmaa=1.0))
    assert e.value.error.kind is ErrorKind.input
    assert e.value.error.code == "unknown_key"
    with pytest.raises(ProgramError) as e:
        program.compute(toy_input(output={"pair_energie": True}))
    assert e.value.error.kind is ErrorKind.input
    assert "output.pair_energies" in e.value.error.message


@pytest.mark.parametrize("kind", ["convergence", "resource", "internal"])
def test_failures_raise_with_kinds(kind):
    with pytest.raises(ProgramError) as e:
        program.compute(toy_input(fail=kind))
    assert e.value.error.kind == kind
    assert e.value.results is not None  # partial results travel with the error


def test_programs_honour_the_thread_grant():
    assert program.compute(toy_input()).execution.threads == 1  # direct: 1
    assert program.compute(toy_input(), RunContext(threads=8)).execution.threads == 8
    capped = program.compute(toy_input(threads=4), RunContext(threads=2))
    assert capped.execution.threads == 2
    assert capped.resolved.settings["threads"] == 2
    assert [w.code for w in capped.warnings] == ["resource_capped"]

    class Greedy(ToyProgram):
        def compute(self, inp, ctx=None):
            out = super().compute(inp)
            return out.model_copy(
                update={"execution": out.execution.model_copy(update={"threads": 64})}
            )

    with pytest.raises(ConformanceError, match="honours ctx.threads"):
        check_program(Greedy(), [toy_input()])


def test_unsupported_calctype():
    with pytest.raises(ProgramError) as e:
        program.plan(toy_input("optimization"))
    assert e.value.error.kind is ErrorKind.unsupported


def test_check_program_reports_every_broken_contract():
    class Lenient(ToyProgram):
        """Accepts unknown keys and forgets the gradient of Hessian sessions."""

        def plan(self, inp):
            clean = {k: v for k, v in inp.keywords.items() if k != "zz_not_a_setting"}
            return super().plan(inp.model_copy(update={"keywords": clean}))

        def compute(self, inp, ctx=None):
            clean = {k: v for k, v in inp.keywords.items() if k != "zz_not_a_setting"}
            return super().compute(inp.model_copy(update={"keywords": clean}), ctx)

    with pytest.raises(ConformanceError) as e:
        check_program(Lenient(), [toy_input()])
    message = str(e.value)
    assert "unknown key rejected by plan" in message
    assert "unknown key fails compute" in message


def plain_compute(inputs, raise_exc=True):
    """What a runner's compute gives a ComputeSession: one raises, a list does not."""
    if isinstance(inputs, ProgramInput):
        return program.compute(inputs)
    return program.compute_batch(inputs)


def test_compute_session_over_a_program():
    compute = plain_compute

    spec = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})
    x = np.asarray(ARGON3.geometry)
    inp = ProgramInput.from_spec(spec, ARGON3)
    with ComputeSession(compute, inp, program_version=program.version()) as s:
        assert isinstance(s, Session)
        assert s.program_version == program.version()  # from its opening
        frame = s.compute(x)
        r = frame.results
        assert r.gradient.shape == (3, 3)
        assert np.allclose(frame.structure.geometry, x)
        with pytest.raises(ProgramError) as e:
            s.compute(np.full_like(x, np.nan))
        assert e.value.error.kind is ErrorKind.numerical
        assert s.compute(x).results.energy == r.energy  # still usable
        assert frame.record["warm_start"] is False  # stateless: always fresh
        assert np.allclose(s.compute().structure.geometry, x)  # the input's own
        with pytest.raises(ProgramErrors) as group:
            s.compute_many(
                [Request(positions=x), Request(positions=np.full_like(x, np.nan))]
            )
        many = group.value.results  # every frame, in request order
        assert len(group.value.exceptions) == 1
        assert many[0].success and not many[1].success
        assert many[1].error.kind is ErrorKind.numerical
        with pytest.raises(ProgramError) as e:
            s.compute(x, outputs=["dipole"])
        assert e.value.error.kind is ErrorKind.unsupported
        with pytest.raises(ProgramError, match="atoms"):
            s.compute(x[:2])
    with pytest.raises(RuntimeError, match="closed"):
        s.compute(x)


def test_compute_session_completes_hessians_without_gradients():
    def no_gradient(inp, raise_exc=True):
        out = program.compute(inp)
        if inp.calctype == CalcType.hessian:
            return out.model_copy(
                update={"results": out.results.model_copy(update={"gradient": None})}
            )
        return out

    spec = ProgramSpec(program="toy", calctype="hessian", model={"method": "lj"})
    s = ComputeSession(
        no_gradient, ProgramInput.from_spec(spec, ARGON3), program_version="1"
    )
    h = s.compute(np.asarray(ARGON3.geometry), "hessian").results
    assert h.gradient is not None and h.hessian is not None


def test_run_context_sessions_and_compute_use_the_default_runner():
    """Called directly, a program with subprograms runs them on the default runner."""
    register_program(program)
    try:
        spec = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})
        with RunContext().session(spec, ARGON3) as s:
            assert s.calctypes  # the program's own session, guarded for subprograms
            assert s.compute().success
        with RunContext(threads=16).session(spec, ARGON3, threads=2) as s:
            assert s.compute().success
        out = RunContext().compute(toy_input())
        assert out.success and out.execution.threads is not None
    finally:
        unregister_program("toy")


def test_discovery_of_unregistered_programs():
    assert find_program("no-such-program") is None
    assert isinstance(available_programs(), list)


def test_registered_programs_are_found_everywhere():
    """One registry: what a test or notebook registers, a driver's plan finds too."""
    toy = ToyProgram()
    toy.name = "toy-registered"
    register_program(toy)
    try:
        assert find_program("toy-registered") is toy
        assert "toy-registered" in available_programs()
        child = ProgramSpec(
            program="toy-registered", calctype="energy", model={"method": "lj"}
        )
        with RunContext().session(child, ARGON3) as s:
            assert s.compute(np.asarray(ARGON3.geometry), CalcType.energy).success
    finally:
        unregister_program("toy-registered")
    assert find_program("toy-registered") is None
    with pytest.raises(TypeError, match="does not implement"):
        register_program(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="class ToyProgram, not a program"):
        register_program(ToyProgram)  # type: ignore[arg-type]


def test_entry_points_must_name_program_instances(monkeypatch):
    """A Program class passes the protocol check; its first call would then fail
    obscurely (version() missing self), so the entry point is refused at once."""
    from importlib.metadata import EntryPoint

    from qccompute import program as module

    ep = EntryPoint(
        "toy-class", "qccompute.testing.toy:ToyProgram", "qccompute.programs"
    )
    monkeypatch.setattr(module, "entry_points", lambda **kwargs: [ep])
    module._entry_points.cache_clear()
    try:
        with pytest.raises(TypeError, match=r"'toy-class' .* is the class ToyProgram"):
            find_program("toy-class")
    finally:
        module._entry_points.cache_clear()


def test_installed_programs_are_read_once_per_process(monkeypatch):
    from qccompute import program as module

    calls = []

    def counting(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(module, "entry_points", counting)
    module._entry_points.cache_clear()
    try:
        assert find_program("not-installed-a") is None
        assert find_program("not-installed-b") is None
        available_programs()
    finally:
        module._entry_points.cache_clear()
    assert len(calls) == 1


def test_program_error_shorthand():
    e = ProgramError.of("convergence", "SCF", code="max_iter")
    assert e.error.code == "max_iter" and e.results is None
    assert "convergence: SCF" in str(e)


def test_toy_session_keeps_history_per_lineage():
    x = np.asarray(ARGON3.geometry)
    y = x + 0.1
    with program.open_session(toy_input(), RunContext()) as s:
        assert s.compute(x, lineage="left").record["warm_start"] is False
        right = s.compute(y, lineage="right")
        assert right.record["warm_start"] is False
        left = s.compute(y, lineage="left")
        assert left.record["step_from_previous"] == pytest.approx(np.linalg.norm(y - x))
        with pytest.raises(ProgramError):
            s.compute(np.full_like(x, np.nan), lineage="left")
        assert s.compute(y, lineage="left").record["step_from_previous"] == 0.0
        s.reset("left")
        assert s.compute(y, lineage="left").record["warm_start"] is False
        assert s.compute(y, lineage="right").record["warm_start"] is True
        pairs = s.compute(x, outputs={"pair_energies"}).record["pair_energies"]
        assert len(pairs) == 3


def test_accuracy_of_a_bare_number_targets_the_highest_derivative():
    assert Accuracy.of(1e-4, CalcType.gradient) == Accuracy(gradient=1e-4)
    assert Accuracy.of(1e-7, CalcType.energy) == Accuracy(energy=1e-7)
    assert Accuracy.of(None, CalcType.hessian) is None
    a = Accuracy(energy=1e-8, gradient=1e-5)
    assert Accuracy.of(a, CalcType.hessian) is a


class FiniteDifference(ToyProgram):
    """A driver: its subprogram's gradient by central differences, through a session."""

    name = "fd"

    def capabilities(self):
        return (
            super()
            .capabilities()
            .model_copy(update={"sessions": False, "batches": False})
        )

    def settings_schema(self, calctype):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "step": {"type": "number", "default": 1e-4, "x-unit": "bohr"}
            },
        }

    def plan(self, inp):
        from qcdata import Resolved

        from qccompute.program import Plan

        if set(inp.keywords) - {"step"}:
            raise ProgramError.of("input", "fd takes only 'step'")
        return Plan(
            resolved=Resolved(settings={"step": inp.keywords.get("step", 1e-4)})
        )

    def compute(self, inp, ctx=None):
        from qcdata import ExecutionInfo, ProgramOutput, Provenance, SinglePointData

        if ctx is None:  # a direct call: the default runner runs it, with a ctx
            import qccompute

            return qccompute.compute(inp)
        plan = self.plan(inp)
        step = plan.resolved.settings["step"]
        child = inp.subprograms[0]
        x = np.asarray(inp.structure.geometry, dtype=float).reshape(-1)
        requests = []
        for k in range(len(x)):
            for sign in (1, -1):
                d = np.zeros_like(x)
                d[k] = sign * step
                # Every displacement starts from the center; none is kept.
                requests.append(
                    Request(
                        positions=(x + d).reshape(-1, 3),
                        calctype=CalcType.energy,
                        commit=False,
                    )
                )
        with ctx.session(child, inp.structure) as s:
            center = s.compute(x.reshape(-1, 3), CalcType.energy).results
            frames = s.compute_many(requests)
        energies = np.array([f.results.energy for f in frames])
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=SinglePointData(
                provenance=Provenance(program="fd", program_version=self.version()),
                energy=center.energy,
                gradient=(energies[0::2] - energies[1::2]) / (2 * step),
            ),
            resolved=plan.resolved,
            execution=ExecutionInfo(threads=ctx.threads, timings={"total": 0.0}),
        )


def fd_input(child_program: str = "toy", **child_keywords: Any) -> ProgramInput:
    return ProgramInput.model_validate(
        {
            "program": "fd",
            "calctype": "gradient",
            "structure": ARGON3,
            "subprograms": [
                {
                    "program": child_program,
                    "calctype": "energy",
                    "model": {"method": "lj"},
                    "keywords": child_keywords,
                }
            ],
        }
    )


@pytest.fixture
def toy_registered():
    register_program(program)
    yield
    unregister_program("toy")


def test_programs_with_subprograms_conform(toy_registered):
    ran = check_program(FiniteDifference(), [fd_input()])
    assert "input 0 (gradient): compute succeeds" in ran


def test_subprograms_of_direct_calls_run_on_the_default_runner(toy_registered):
    fd = FiniteDifference()
    register_program(fd)  # called directly, it hands itself to the default runner
    out = fd.compute(fd_input())
    unregister_program("fd")
    exact = program.compute(toy_input("gradient"))
    assert np.allclose(out.results.gradient, exact.results.gradient, atol=1e-8)
    with pytest.raises(ProgramError) as e:
        RunContext().compute(toy_input().model_copy(update={"program": "no-such"}))
    assert e.value.error.kind is ErrorKind.environment


def test_toy_sessions_fail_on_request():
    spec = ProgramSpec(
        program="toy",
        calctype="gradient",
        model={"method": "lj"},
        keywords={"fail": "convergence"},
    )
    with program.open_session(ProgramInput.from_spec(spec, ARGON3)) as s:
        with pytest.raises(ProgramError) as e:
            s.compute(np.asarray(ARGON3.geometry))
        assert e.value.error.kind is ErrorKind.convergence
        with pytest.raises(ProgramErrors) as group:
            s.compute_many([Request()])
        frame = group.value.results[0]
        assert not frame.success and frame.error.kind is ErrorKind.convergence
        assert group.value.exceptions[0].frame is frame


def test_sessions_run_only_the_calctypes_they_were_opened_for(toy_registered):
    from qccompute.program import session_calctypes

    spec = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})
    assert session_calctypes() == {CalcType.energy, CalcType.gradient, CalcType.hessian}
    assert session_calctypes(None, ["energy"]) == {CalcType.energy}  # the method's
    assert session_calctypes(["gradient"]) == {CalcType.energy, CalcType.gradient}
    assert CalcType.gradient in session_calctypes(["hessian"])
    x = np.asarray(ARGON3.geometry)
    inp = ProgramInput.from_spec(spec, ARGON3)
    with program.open_session(inp, RunContext()) as s:  # everything the toy supports
        assert s.compute(x, "hessian").results.hessian is not None
    with program.open_session(inp, RunContext(), session_calctypes(["gradient"])) as s:
        assert s.compute(x, "energy").results.energy is not None  # lower derivative
        with pytest.raises(ProgramError) as e:
            s.compute(x, "hessian")
        assert e.value.error.kind is ErrorKind.input
        assert e.value.error.code == "not_applicable"
        with pytest.raises(ProgramErrors) as group:
            s.compute_many([Request(positions=x, calctype=CalcType.hessian)])
        frame = group.value.results[0]
        assert not frame.success and frame.error.code == "not_applicable"
    with program.open_session(
        inp, RunContext(), frozenset({CalcType.gradient, CalcType.hessian})
    ) as s:
        assert s.compute(x, "hessian").results.hessian is not None

    plain = ComputeSession(
        plain_compute, inp, calctypes=["gradient"], program_version="1"
    )
    with pytest.raises(ProgramError) as e:
        plain.compute(x, "hessian")
    assert e.value.error.code == "not_applicable"
    ctx = RunContext()
    assert (
        ctx.session(spec, ARGON3, calctypes=["hessian"]).compute(x, "hessian").success
    )


def test_scratch_is_made_on_first_use_and_removed_by_cleanup(tmp_path):
    ctx = RunContext()
    assert ctx._scratch is None  # nothing made until asked for
    scratch = ctx.scratch
    assert scratch.is_dir() and ctx.scratch == scratch
    ctx.cleanup()
    assert not scratch.exists()
    given = RunContext(scratch=tmp_path)
    assert given.scratch == tmp_path
    given.cleanup()
    assert tmp_path.is_dir()  # a given directory is kept


def test_sessions_open_from_an_input_and_default_to_it():
    """One object is the native API and the protocol's: ToySession(inp) directly."""
    inp = toy_input("gradient")
    with ToySession(inp) as s:
        assert isinstance(s, Session)
        frame = s.compute()  # the input's geometry and calctype
        assert np.allclose(frame.structure.geometry, ARGON3.geometry)
        assert frame.results.gradient is not None and frame.results.hessian is None
    with pytest.raises(TypeError, match="ProgramInput"):
        ToySession(ProgramSpec(program="toy", calctype="energy"))  # type: ignore[arg-type]
    opened = program.open_session(inp)
    assert type(opened) is ToySession


def test_lineages_seeds_and_commits():
    x = np.asarray(ARGON3.geometry)
    y, z = x + 0.1, x + 0.2
    with ToySession(toy_input()) as s:
        s.compute(x, lineage=("ts",))  # a tuple and its JSON list are one lineage
        frames = s.compute_many(
            [
                Request(positions=y, lineage="forward", seed=[None, ["ts"]]),
                Request(positions=z, lineage="forward", commit=False),
                Request(positions=z, lineage="backward", commit=False),
            ]
        )
        first, probe, unseeded = (f.record for f in frames)
        assert first["warm_start"] and first["from_lineage"] == ["ts"]
        assert probe["from_lineage"] == "forward"  # reads what the first committed
        assert probe["step_from_previous"] == pytest.approx(np.linalg.norm(z - y))
        assert unseeded["warm_start"] is False  # no seed: fresh, not an error
        again = s.compute(y + 1e-3, lineage="forward").record
        # The probe at z stored nothing: the step is measured from y.
        assert again["step_from_previous"] == pytest.approx(np.sqrt(9) * 1e-3)
        single = s.compute(z, lineage="branch", seed=["forward"], commit=False)
        assert single.record["from_lineage"] == "forward"
        s.compute(x)  # the default lineage
        s.reset(None)  # forgets only the default lineage
        assert s.compute(x).record["warm_start"] is False
        assert s.compute(y, lineage="forward").record["warm_start"] is True
        s.reset()  # forgets every lineage
        assert s.compute(y, lineage="forward").record["warm_start"] is False


def test_capabilities_and_plans_may_be_plain_mappings():
    """Programs never import qccompute: what they return is validated here."""
    from qccompute.program import capabilities_of, plan_of

    class Plain(ToyProgram):
        name = "plain-mappings"

        def capabilities(self):
            return {"calctypes": ["energy", "gradient"], "methods": {"lj": ["energy"]}}

        def plan(self, inp):
            return {
                "resolved": {"settings": {"sigma": 6.0}},
                "resources": {"threads": 2},
                "record": {"placement": "cpu"},
            }

    plain = Plain()
    caps = capabilities_of(plain)
    assert caps.calctypes == {CalcType.energy, CalcType.gradient}
    assert caps.methods == {"lj": {CalcType.energy}}
    plan = plan_of(plain, toy_input())
    assert plan.resources.threads == 2 and plan.record == {"placement": "cpu"}
    assert plan.resolved.settings == {"sigma": 6.0}


def test_program_errors_keep_the_failed_frame_through_compute_sessions():
    def unconverged(inputs, raise_exc=True):
        def one(inp):
            failing = inp.model_copy(update={"keywords": {"fail": "convergence"}})
            try:
                program.compute(failing)
            except ProgramError as e:
                out = ProgramOutput.failure(failing, e.error, results=e.results)
                return out.model_copy(update={"record": {"scf": {"iterations": 128}}})
            raise AssertionError("did not fail")

        if isinstance(inputs, ProgramInput):
            out = one(inputs)
            e = ProgramError(out.error, out.results)
            e.output = out
            raise e
        return [one(i) for i in inputs]

    s = ComputeSession(unconverged, toy_input(), program_version="1")
    with pytest.raises(ProgramError) as e:
        s.compute()
    assert e.value.frame is not None and not e.value.frame.success
    assert e.value.frame.record["scf"]["iterations"] == 128
    with pytest.raises(ProgramErrors) as group:
        s.compute_many([Request()])
    assert group.value.results[0].record["scf"]["iterations"] == 128


def test_plans_may_defer_what_the_run_decides():
    """linus decides DF storage from free memory at the SCF's start: auto in the plan,
    resolved in the output; anything else left auto is a conformance failure."""
    from qcdata import Resolved

    from qccompute.program import Plan

    class Deferring(ToyProgram):
        def __init__(self, declare: bool):
            self.declare = declare

        def plan(self, inp):
            own = {k: v for k, v in inp.keywords.items() if k != "storage"}
            plan = super().plan(inp.model_copy(update={"keywords": own}))
            settings = {**plan.resolved.settings, "storage": {"kind": "auto"}}
            resolved = Resolved(
                settings=settings,
                reasons={
                    **plan.resolved.reasons,
                    "storage.kind": "decided at run time",
                },
                deferred=plan.resolved.deferred
                + (["storage.kind"] if self.declare else []),
            )
            return Plan(resolved=resolved, resources=plan.resources)

        def compute(self, inp, ctx=None):
            out = super().compute(
                inp.model_copy(
                    update={
                        "keywords": {
                            k: v for k, v in inp.keywords.items() if k != "storage"
                        }
                    }
                ),
                ctx,
            )
            if not out.success:
                return out.model_copy(update={"input_data": inp})
            settings = {**out.resolved.settings, "storage": {"kind": "incore"}}
            return out.model_copy(
                update={"input_data": inp, "resolved": Resolved(settings=settings)}
            )

        def settings_schema(self, calctype):
            schema = super().settings_schema(calctype)
            props = {**schema["properties"], "storage": {"type": "object"}}
            return {**schema, "properties": props}

    check_program(Deferring(declare=True), [toy_input()])
    with pytest.raises(ConformanceError, match="storage.kind"):
        check_program(Deferring(declare=False), [toy_input()])


# ---------------------------------------------------------------- device ordinals (#49)


def test_device_ordinals_are_positions_in_cuda_visible_devices(monkeypatch):
    """With CUDA_VISIBLE_DEVICES set, a granted GPU's ordinal is its position there."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,5")
    assert RunContext().device_ordinals() == []
    assert RunContext(gpus=["2"]).device_ordinals() == [1]
    assert RunContext(gpus=["5", "1"]).device_ordinals() == [2, 0]
    # In a worker, CUDA_VISIBLE_DEVICES is the grant: its GPUs are 0, 1, ...
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert RunContext(gpus=["7"]).device_ordinals() == [0]
    # UUIDs match by value or a unique prefix; MIG instances likewise.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa-1111,GPU-bbbb-2222,MIG-cc-3")
    assert RunContext(gpus=["GPU-bbbb-2222"]).device_ordinals() == [1]
    assert RunContext(gpus=["GPU-bbbb"]).device_ordinals() == [1]
    assert RunContext(gpus=["MIG-cc-3"]).device_ordinals() == [2]
    # CUDA stops at the first entry that names no device.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,-1,1")
    with pytest.raises(ProgramError) as info:
        RunContext(gpus=["1"]).device_ordinals()
    assert info.value.error.kind is ErrorKind.environment
    assert "'1'" in info.value.error.message


def test_device_ordinals_without_cuda_visible_devices(monkeypatch):
    """Unset: an index is its own ordinal, a UUID its nvidia-smi index (PCI order)."""
    from qccompute import program as module

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rows = (("0", "GPU-aaaa-1111", 80.0, "A100"), ("1", "GPU-bbbb-2222", 24.0, "L4"))
    monkeypatch.setattr(module, "_nvidia_smi_gpus", lambda: rows)
    assert RunContext(gpus=["0", "3"]).device_ordinals() == [0, 3]
    assert RunContext(gpus=["GPU-bbbb"]).device_ordinals() == [1]
    with pytest.raises(ProgramError, match="nvidia-smi"):
        RunContext(gpus=["GPU-cccc"]).device_ordinals()


def test_runner_numbers_gpus_as_nvidia_smi_does(monkeypatch):
    """GPUs listed by nvidia-smi: CUDA_DEVICE_ORDER=PCI_BUS_ID unless set (#49)."""
    from qccompute import program
    from qccompute import runner as module
    from qccompute.runner import Runner, visible_gpus

    rows = (("0", "GPU-aaaa-1111", 80.0, "A100"), ("1", "GPU-bbbb-2222", None, "L4"))
    monkeypatch.setattr(module, "_nvidia_smi_gpus", lambda: rows)
    monkeypatch.setattr(program, "_nvidia_smi_gpus", lambda: rows)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    assert visible_gpus() == {"0": 80.0, "1": None}
    assert module.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")  # the user's stands
    Runner(threads=1)
    assert module.os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
    # A dispatcher's CUDA_VISIBLE_DEVICES is left as it is, UUIDs and all.
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb-2222,0")
    assert visible_gpus() == {"GPU-bbbb-2222": None, "0": 80.0}
    assert "CUDA_DEVICE_ORDER" not in module.os.environ
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert visible_gpus() == {}


def test_capabilities_have_no_gpu_flag():
    """GPUs are a plan's, not a capability: nothing read Capabilities.gpu (alpha 4)."""
    with pytest.raises(ValidationError):
        Capabilities.model_validate({"calctypes": ["energy"], "gpu": True})


def test_resources_have_no_exclusive_flag():
    """Nothing read Resources.exclusive, and the runner never honoured it (alpha 4)."""
    with pytest.raises(ValidationError):
        Resources.model_validate({"exclusive": True})
