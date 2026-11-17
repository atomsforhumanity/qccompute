"""Conformance tests for programs implementing `qccompute.program.Program`.

A program runs `check_program` in its CI with a few small inputs of its own (the one
place a program imports qccompute, as a test dependency)::

    from qccompute.testing import check_program

    def test_conforms(water):
        check_program(mychem.qc.program, [
            ProgramInput(program="mychem", calctype="gradient", structure=water,
                         model={"method": "hf", "basis": "sto-3g"}),
        ])

It checks the contract of ``docs/design/program-interface.md`` and qcdata's
conventions: schemas, plans that reproduce themselves, typed errors (raised) for
unknown settings, the thread grant honoured, standard fields filled in atomic-unit
shapes, energy components that add up, and, for programs with sessions or batches, that
those agree with plain computes and that a session survives a failed request. Schemas
and keywords are validated with ``jsonschema`` when it is installed. A program with
subprograms runs them on the default runner, so register the programs they name (or
install them) first.

`qccompute.testing.toy` is a complete example program.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import ValidationError
from qcdata import (
    CalcType,
    ErrorKind,
    Frame,
    ProgramError,
    ProgramErrors,
    ProgramInput,
    ProgramOutput,
    Request,
    SinglePointData,
)

from ..program import (
    BatchProgram,
    Capabilities,
    Program,
    RunContext,
    SessionProgram,
    _registered,
    capabilities_of,
    plan_of,
    register_program,
    unregister_program,
)

__all__ = ["check_program", "ConformanceError"]


class ConformanceError(AssertionError):
    """A program does not meet the contract; the message lists every failed check."""


def _auto_paths(value: Any, path: str = "") -> list[str]:
    """Dotted paths of every ``"auto"`` left in a settings tree."""
    if isinstance(value, str):
        return [path] if value.lower() == "auto" else []
    if isinstance(value, dict):
        return [
            p
            for k, v in value.items()
            for p in _auto_paths(v, f"{path}.{k}" if path else str(k))
        ]
    if isinstance(value, list | tuple):
        return [p for i, v in enumerate(value) for p in _auto_paths(v, f"{path}[{i}]")]
    return []


def _without(settings: Mapping[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    """The settings tree without the given dotted paths."""
    tree = {
        k: _without(v, []) if isinstance(v, dict) else v for k, v in settings.items()
    }
    for path in paths:
        *parents, leaf = path.split(".")
        node: Any = tree
        for key in parents:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(leaf, None)
    return tree


def _validator(schema: dict[str, Any]) -> Callable[[Any], list[str]] | None:
    """A function listing schema violations of a value, or None without jsonschema."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        return None
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    validator = cls(schema)
    return lambda value: [e.message for e in validator.iter_errors(value)]


def check_program(
    program: Program,
    inputs: Sequence[ProgramInput],
    *,
    energy_tol: float = 1e-8,
) -> list[str]:
    """Check that a program meets the AFH program contract.

    The program is registered while the checks run, so a program with subprograms
    called directly (``ctx=None``) reaches itself through ``qccompute.compute``.

    Args:
        program: The program object (what its ``qccompute.programs`` entry point loads).
        inputs: Small inputs the program can compute quickly, covering the calctypes
            to check; the first one also exercises the thread grant, sessions and
            batches.
        energy_tol: Tolerance (Eh) for energy components adding up and for sessions
            and batches agreeing with plain computes.

    Returns:
        The names of the checks that ran.

    Raises:
        ConformanceError: One or more checks failed; the message lists all of them.
    """
    previous = _registered.get(program.name) if isinstance(program, Program) else None
    if isinstance(program, Program) and not isinstance(program, type):
        register_program(program)
    try:
        return _check_program(program, inputs, energy_tol)
    finally:
        if previous is not None:
            register_program(previous)
        elif isinstance(program, Program) and not isinstance(program, type):
            unregister_program(program.name)


def _check_program(
    program: Program, inputs: Sequence[ProgramInput], energy_tol: float
) -> list[str]:
    failures: list[str] = []
    ran: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        ran.append(name)
        if not ok:
            failures.append(f"{name}: {detail}" if detail else name)
        return ok

    if not inputs:
        raise ValueError("check_program needs at least one input.")

    # The protocol itself.
    if not check(
        "protocol", isinstance(program, Program), "does not implement Program"
    ):
        raise ConformanceError("\n".join(failures))
    version = program.version()
    check("version", isinstance(version, str) and bool(version), repr(version))
    try:
        caps = capabilities_of(program)
    except ValidationError as e:
        check("capabilities", False, str(e))
        raise ConformanceError("\n".join(failures)) from None
    check("capabilities", bool(caps.calctypes), repr(caps))

    validators: dict[CalcType, Callable[[Any], list[str]] | None] = {}
    for calctype in sorted(caps.calctypes):
        schema = program.settings_schema(calctype)
        record = program.record_schema(calctype)
        ok = (
            isinstance(schema, dict)
            and schema.get("type") == "object"
            and isinstance(schema.get("properties"), dict)
            and isinstance(record, dict)
        )
        check(f"schemas[{calctype.value}]", ok, "settings schema must be an object")
        if ok:
            try:
                validators[calctype] = _validator(schema)
                _validator(record)
            except Exception as e:  # jsonschema.SchemaError
                check(f"schemas[{calctype.value}] valid", False, str(e))

    for n, inp in enumerate(inputs):
        tag = f"input {n} ({inp.calctype.value})"
        if not check(f"{tag}: program name", inp.program == program.name, inp.program):
            continue
        validate = validators.get(inp.calctype)
        if validate is not None:
            errors = validate(dict(inp.keywords))
            check(f"{tag}: keywords match schema", not errors, "; ".join(errors))

        # Plans resolve every auto the run does not decide, and reproduce themselves.
        try:
            plan = plan_of(program, inp)
        except ProgramError as e:
            check(f"{tag}: plan", False, str(e))
            continue
        except ValidationError as e:
            check(f"{tag}: plan type", False, str(e))
            continue
        settings = plan.resolved.settings
        deferred = plan.resolved.deferred
        undeferred = [p for p in _auto_paths(settings) if p not in deferred]
        check(
            f"{tag}: resolved has no auto but the deferred",
            not undeferred,
            f"{undeferred} (list settings the run decides in resolved.deferred)",
        )
        if validate is not None:
            errors = validate(settings)
            check(f"{tag}: resolved matches schema", not errors, "; ".join(errors))
        again = plan_of(program, inp.model_copy(update={"keywords": settings}))
        check(
            f"{tag}: resolved settings reproduce the plan",
            again.resolved.settings == settings,
            f"{again.resolved.settings} != {settings}",
        )

        # Unknown settings are input errors, in plan and compute.
        bad = inp.model_copy(
            update={"keywords": {**inp.keywords, "zz_not_a_setting": 1}}
        )
        try:
            program.plan(bad)
            check(f"{tag}: unknown key rejected by plan", False, "plan accepted it")
        except ProgramError as e:
            check(
                f"{tag}: unknown key rejected by plan",
                e.error.kind == ErrorKind.input,
                f"kind {e.error.kind}",
            )
        try:
            program.compute(bad)
            check(f"{tag}: unknown key fails compute", False, "compute accepted it")
        except ProgramError as e:
            check(
                f"{tag}: unknown key fails compute",
                e.error.kind == ErrorKind.input,
                f"kind {e.error.kind}",
            )

        # The calculation and its standard fields.
        try:
            out = program.compute(inp)
        except ProgramError as e:
            check(f"{tag}: compute succeeds", False, repr(e.error))
            continue
        if not check(
            f"{tag}: compute succeeds",
            isinstance(out, ProgramOutput) and out.success,
            repr(getattr(out, "error", out)),
        ):
            continue
        results = out.results
        prov = results.provenance
        check(
            f"{tag}: provenance",
            prov.program == program.name and prov.program_version == version,
            repr(prov),
        )
        check(
            f"{tag}: output resolved equals plan, deferred settings decided",
            out.resolved is not None
            and not _auto_paths(out.resolved.settings)
            and not out.resolved.deferred
            and _without(out.resolved.settings, deferred)
            == _without(settings, deferred),
            repr(out.resolved),
        )
        if isinstance(results, SinglePointData):
            failures.extend(_check_single_point(tag, inp, results, energy_tol, ran))

    first = inputs[0]
    try:
        reference = program.compute(first)
    except ProgramError as e:
        raise ConformanceError(
            f"{program.name} fails its first input: {e.error!r}\n- "
            + "\n- ".join(failures)
        ) from e

    # The thread grant: never more threads than granted, even when asked for more.
    ran.append("honours ctx.threads")
    for keywords in (first.keywords, {**first.keywords, "threads": 2}):
        try:
            granted = program.compute(
                first.model_copy(update={"keywords": keywords}), RunContext(threads=1)
            )
        except ProgramError as e:
            if keywords is first.keywords:
                failures.append(f"honours ctx.threads: compute failed: {e.error!r}")
            continue  # a program without a threads keyword refuses it
        used = granted.execution.threads
        if used is None or used > 1:
            failures.append(
                "honours ctx.threads: granted 1 thread, the output's "
                f"execution.threads is {used} (record the threads used)"
            )
            break

    if caps.sessions:
        failures.extend(
            _check_session(program, first, reference, caps, energy_tol, ran)
        )
    if caps.batches:
        name = "batch agrees with compute"
        if not isinstance(program, BatchProgram):
            failures.append(f"{name}: batches claimed but no compute_batch")
        else:
            outs = program.compute_batch([first, first])
            ran.append(name)
            if len(outs) != 2 or not all(o.success for o in outs):
                failures.append(f"{name}: {outs!r}")
            elif reference.success and isinstance(reference.results, SinglePointData):
                e0 = reference.results.energy
                energies = [getattr(o.results, "energy", None) for o in outs]
                if e0 is not None and any(
                    e is None or not math.isclose(e, e0, abs_tol=energy_tol)
                    for e in energies
                ):
                    failures.append(f"{name}: energies differ")

    if failures:
        raise ConformanceError(
            f"{program.name} fails {len(failures)} check(s):\n- "
            + "\n- ".join(failures)
        )
    return ran


def _check_single_point(
    tag: str,
    inp: ProgramInput,
    r: SinglePointData,
    tol: float,
    ran: list[str],
) -> list[str]:
    failures: list[str] = []
    assert inp.structure is not None
    natoms = len(inp.structure.symbols)
    ran.append(f"{tag}: standard fields")
    if r.energy is None:
        failures.append(f"{tag}: no energy")
    if inp.calctype in (CalcType.gradient, CalcType.hessian):
        if r.gradient is None or np.shape(r.gradient) != (natoms, 3):
            failures.append(f"{tag}: gradient must be {natoms} x 3 (Eh/bohr)")
    if inp.calctype == CalcType.hessian:
        if r.hessian is None or np.shape(r.hessian) != (3 * natoms, 3 * natoms):
            failures.append(f"{tag}: hessian must be {3 * natoms} x {3 * natoms}")
        va = r.vibrational_analysis
        if va is not None and len(va.masses) != natoms:
            failures.append(f"{tag}: vibrational_analysis has the wrong atom count")
    if r.energy_components and r.energy is not None:
        ran.append(f"{tag}: energy components add up")
        total = sum(r.energy_components.values())
        if not math.isclose(total, r.energy, abs_tol=tol):
            failures.append(
                f"{tag}: energy_components add up to {total}, not {r.energy}"
            )
    return failures


def _check_session(
    program: Program,
    inp: ProgramInput,
    reference: ProgramOutput[Any, Any],
    caps: Capabilities,
    tol: float,
    ran: list[str],
) -> list[str]:
    failures: list[str] = []
    if not isinstance(program, SessionProgram):
        return ["sessions claimed but no open_session"]
    assert inp.structure is not None
    positions = np.asarray(inp.structure.geometry, dtype=float)
    nan = np.full_like(positions, np.nan)
    ran.append("session")
    declared = {inp.calctype} | (
        {CalcType.hessian} if CalcType.hessian in caps.calctypes else set()
    )
    ran.append("session refuses calctypes it was not opened for")
    narrow = program.open_session(inp, None, frozenset({CalcType.energy}))
    try:
        if CalcType.gradient in caps.calctypes:
            try:
                narrow.compute(positions, CalcType.gradient)
                failures.append("session: ran a calctype it was not opened for")
            except ProgramError as e:
                if e.error.kind != ErrorKind.input:
                    failures.append(
                        f"session: a request outside its calctypes must be an input "
                        f"error, not {e.error.kind}"
                    )
    finally:
        narrow.close()
    with program.open_session(inp, None, frozenset(declared)) as session:
        if session.program != program.name or not session.program_version:
            failures.append("session: program and program_version must be set")
        try:
            frame = session.compute(positions, inp.calctype, lineage="a")
        except ProgramError as e:
            return failures + [f"session: compute failed: {e}"]
        if not isinstance(frame, Frame) or not frame.success:
            return failures + [
                f"session: compute must return a successful Frame: {frame!r}"
            ]
        if not np.allclose(frame.structure.geometry, positions):
            failures.append(
                "session: the frame's structure is not the evaluated geometry"
            )
        if "warm_start" not in frame.record:
            failures.append("session: the frame's record has no warm_start")
        try:
            default = session.compute()  # the input's own geometry and calctype
            if not np.allclose(default.structure.geometry, positions):
                failures.append("session: compute() is not at the input's geometry")
        except ProgramError as e:
            failures.append(f"session: compute() with no arguments failed: {e}")
        energy = frame.results.energy
        if reference.success and reference.results.energy is not None:
            if energy is None or not math.isclose(
                energy, reference.results.energy, abs_tol=tol
            ):
                failures.append(
                    f"session: energy {energy} differs from compute's "
                    f"{reference.results.energy}"
                )
        ran.append("session survives a failed request")
        try:
            session.compute(nan, inp.calctype, lineage="a")
            failures.append("session: non-finite positions were accepted")
        except ProgramError:
            pass
        try:
            again = session.compute(positions, inp.calctype, lineage="a")
            if energy is not None and not math.isclose(
                again.results.energy or math.nan, energy, abs_tol=tol
            ):
                failures.append("session: results changed after a failed request")
        except ProgramError as e:
            failures.append(f"session: unusable after a failed request: {e}")
        ran.append("session compute_many")
        many: Any = None
        try:
            session.compute_many(
                [
                    Request(positions=positions, calctype=inp.calctype, lineage="a"),
                    Request(positions=nan, calctype=inp.calctype, lineage="b"),
                    Request(positions=positions, calctype=inp.calctype, lineage="b"),
                ]
            )
            failures.append("session: compute_many must raise qcdata.ProgramErrors")
        except ProgramErrors as group:
            many = group.results
            if len(group.exceptions) != 1 or not isinstance(
                group.exceptions[0], ProgramError
            ):
                failures.append(
                    "session: compute_many's group must hold one ProgramError per "
                    f"failed request: {group.exceptions!r}"
                )
        if many is None:
            pass
        elif len(many) != 3 or not all(isinstance(m, Frame) for m in many):
            failures.append(
                f"session: compute_many must return one Frame per request: {many!r}"
            )
        elif not (many[0].success and many[2].success) or many[1].success:
            failures.append("session: compute_many must fail only the bad request")
        elif many[1].error is None:
            failures.append("session: a failed frame must carry its error")
        ran.append("session reset")
        session.reset("a")
        session.reset()
        try:
            session.compute(positions, inp.calctype)
        except ProgramError as e:
            failures.append(f"session: unusable after reset: {e}")
        if CalcType.hessian in caps.calctypes:
            ran.append("session hessian includes gradient")
            h = session.compute(positions, CalcType.hessian).results
            if h.energy is None or h.gradient is None or h.hessian is None:
                failures.append(
                    "session: a hessian request must return energy and gradient"
                )
    return failures
