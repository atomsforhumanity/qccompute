"""The protocol an AFH program implements to run under qccompute, and its sessions.

A program package registers one object (an instance, not its class) under the
``qccompute.programs`` entry point group::

    [project.entry-points."qccompute.programs"]
    linus = "linus.qc:program"

The object implements `Program`: it says what it can do (`capabilities`), publishes the
JSON Schema of its settings (`settings_schema`) and of its run record (`record_schema`),
plans a calculation without running it (`plan`: validated, every ``"auto"`` resolved, the
resources it needs), and computes it (`compute`) under a `RunContext` the runner fills.
Programs that keep state between geometries (warm starts, a pinned grid) also implement
`SessionProgram.open_session`; programs that evaluate many inputs at once implement
`BatchProgram.compute_batch`.

**Programs never import qccompute.** The protocols are structural (`typing.Protocol`), and
what a program returns for `capabilities` and `plan` may be a plain mapping (the document
its command line prints and its worker sends), which qccompute validates
(`capabilities_of`, `plan_of`). What programs and drivers share as data is in qcdata:
inputs and outputs, `Frame`, `qcdata.Request`, `qcdata.Accuracy` and
`qcdata.ProgramError`. So ``linus.Session(inp)`` and ``linus.compute(inp)`` work without
qccompute, and qccompute runs the same objects.

Everything here finds programs by name through `find_program`: qccompute, a driver
planning its children, `local_context`. A program that is not an installed package (in a
test, a notebook or a script) is added with `register_program`.

Design: ``docs/design/program-interface.md``; names and units: qcdata's conventions.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, overload, runtime_checkable

import numpy as np
from qcdata import (
    Accuracy,
    CalcType,
    Data,
    ErrorInfo,
    ErrorKind,
    Frame,
    Lineage,
    ProgramError,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    ProgramWarning,
    Provenance,
    Request,
    Resolved,
    SinglePointData,
    Structure,
    get_data_type,
)
from qcdata.models.base_models import QCDataBaseModel

__all__ = [
    "ENTRY_POINT_GROUP",
    "Capabilities",
    "Resources",
    "Plan",
    "Compute",
    "RunContext",
    "Program",
    "SessionProgram",
    "BatchProgram",
    "Session",
    "ComputeSession",
    "failed_frame",
    "session_calctypes",
    "check_session_calctype",
    "find_program",
    "register_program",
    "unregister_program",
    "local_context",
    "available_programs",
    "capabilities_of",
    "plan_of",
]

#: The entry point group programs register under.
ENTRY_POINT_GROUP = "qccompute.programs"


class Capabilities(QCDataBaseModel):
    """What a program can do and how it may be run.

    Attributes:
        calctypes: The calculation types it computes.
        methods: The methods it implements, each with the calctypes it computes,
            if it can list them (GFN-FF has no Hessian where GFN2 has one).
        sessions: Whether it implements `SessionProgram.open_session`.
        batches: Whether it implements `BatchProgram.compute_batch`.
        in_process: Whether it may run in the runner's own process (it neither writes
            to the process's stdout/stderr file descriptors nor changes process-global
            state); otherwise the runner uses a worker process.
        thread_safe: Whether separate calculations and sessions may run concurrently
            in threads of one process (the program releases the GIL and shares no
            mutable state); the runner then runs them side by side in threads instead
            of one at a time.
        gpu: Whether it can use GPUs.
        driver: Whether it computes child calculations (its input's subprograms)
            through `RunContext.compute` or `RunContext.session`.
    """

    calctypes: frozenset[CalcType]
    methods: dict[str, frozenset[CalcType]] | None = None
    sessions: bool = False
    batches: bool = False
    in_process: bool = True
    thread_safe: bool = False
    gpu: bool = False
    driver: bool = False


class Resources(QCDataBaseModel):
    """What one calculation needs from the machine; the runner schedules by it.

    Attributes:
        threads: CPU threads.
        memory_gib: Host memory (GiB), if known.
        gpus: Number of GPUs.
        device_memory_gib: Memory per GPU (GiB), if capped; several calculations may
            then share one device.
        exclusive: Whether the calculation must have the machine (or its GPUs) to
            itself, e.g. for clean timings.
    """

    threads: int = 1
    memory_gib: float | None = None
    gpus: int = 0
    device_memory_gib: float | None = None
    exclusive: bool = False


class Plan(QCDataBaseModel):
    """A calculation planned but not run.

    Attributes:
        resolved: The settings with every ``"auto"`` replaced, and why; exactly what
            the run's `ProgramOutput.resolved` will hold.
        resources: What the calculation needs. A driver's is the largest over its
            children's plans.
        warnings: What the program would warn about.
        record: The program's own detail of the plan (linus: placement, arithmetic
            and memory plan), shaped like a run's record.
    """

    resolved: Resolved
    resources: Resources = Resources()
    warnings: list[ProgramWarning] = []
    record: dict[str, Any] = {}


class Compute(Protocol):
    """Computes child calculations; the runner provides it in `RunContext.compute`.

    It takes what ``qccompute.compute`` takes: one input, or a list of independent
    inputs that the runner may batch or run side by side. It returns outputs, failed or
    not, and never raises for a failed calculation. ``previous`` holds outputs whose
    wavefunctions start the children.
    """

    @overload
    def __call__(
        self,
        inputs: ProgramInput,
        /,
        *,
        previous: ProgramOutput[ProgramInput, Data] | None = None,
    ) -> ProgramOutput[ProgramInput, Data]: ...

    @overload
    def __call__(
        self,
        inputs: Sequence[ProgramInput],
        /,
        *,
        previous: Sequence[ProgramOutput[ProgramInput, Data] | None] | None = None,
    ) -> list[ProgramOutput[ProgramInput, Data]]: ...


#: Opens a child session: (input, threads, calctypes) -> Session.
OpenSession = Callable[
    [ProgramInput, int | None, "frozenset[CalcType] | None"], "Session"
]

_LOWER = {
    CalcType.energy: (CalcType.energy,),
    CalcType.gradient: (CalcType.energy, CalcType.gradient),
    CalcType.hessian: (CalcType.energy, CalcType.gradient, CalcType.hessian),
}


def session_calctypes(
    calctypes: Iterable[CalcType | str] | None = None,
    supported: Iterable[CalcType | str] | None = None,
) -> frozenset[CalcType]:
    """The calculation types a session may run.

    By default everything the method supports; declaring calctypes narrows it to
    those, each with the lower derivatives it implies (a gradient session also returns
    energies, a Hessian session also gradients). A session checks its keywords against
    these at opening (a setting that applies to none of them is a ``not_applicable``
    input error; one that applies to an allowed calctype never requested is contingent,
    D41) and refuses requests of other types. Declaring changes validation and
    planning, not what is allocated: programs allocate per request.

    Args:
        calctypes: The calctypes declared at opening; None for the default.
        supported: The method's calctypes (the program's capabilities); None: energy,
            gradient and Hessian.

    Returns:
        The allowed calctypes.
    """
    if not calctypes:
        return frozenset(
            CalcType(c) for c in (supported if supported is not None else _LOWER)
        )
    declared = {CalcType(c) for c in calctypes}
    allowed: set[CalcType] = set()
    for c in declared:
        allowed.update(_LOWER.get(c, (c,)))
    return frozenset(allowed)


def check_session_calctype(
    calctype: CalcType | str, allowed: frozenset[CalcType]
) -> None:
    """Refuse a request of a calctype the session was not opened for.

    Raises:
        ProgramError: Kind ``input``, code ``not_applicable``.
    """
    if CalcType(calctype) not in allowed:
        raise ProgramError(
            ErrorInfo(
                kind=ErrorKind.input,
                code="not_applicable",
                message=f"This session was opened for {sorted(c.value for c in allowed)}, "
                f"not {CalcType(calctype).value}; open it with calctypes= that include it.",
            )
        )


@dataclass
class RunContext:
    """What the runner gives a calculation: its resource grant and its callbacks.

    `RunContext()` is the context of a direct call: nothing granted (the program uses
    its resolved settings) and no runner for children.

    Attributes:
        threads: CPU threads granted; use these, not the machine's count. None: not
            managed by a runner.
        gpus: Identifiers of the GPUs granted (as for ``CUDA_VISIBLE_DEVICES``).
        memory_gib: Host memory granted (GiB), if managed.
        scratch_dir: A directory the calculation may write to, if one was given.
        make_scratch: Makes a scratch directory on first use of `scratch`, so a run
            that never writes files never pays for one.
        plan: The plan the runner already made for this input (to schedule it or to
            key it in a store), if any. A program may use it instead of planning
            again; it is exactly what ``program.plan(input)`` returned.
        compute: Computes child calculations (drivers only).
        open_session: Opens child sessions; use `session`.
        progress: Receives streaming updates (any JSON-serializable value), if the
            caller wants them.
    """

    threads: int | None = None
    gpus: tuple[str, ...] = ()
    memory_gib: float | None = None
    scratch_dir: Path | None = None
    make_scratch: Callable[[], Path] | None = None
    plan: Plan | None = None
    compute: Compute | None = None
    open_session: OpenSession | None = None
    progress: Callable[[Any], None] | None = None

    @property
    def scratch(self) -> Path | None:
        """A directory the calculation may write to, made on first use; None if none.

        Returns:
            The directory, or None when the caller provides none.
        """
        if self.scratch_dir is None and self.make_scratch is not None:
            self.scratch_dir = self.make_scratch()
        return self.scratch_dir

    def session(
        self,
        spec: ProgramInput | ProgramSpec,
        structure: Structure | None = None,
        *,
        threads: int | None = None,
        calctypes: Iterable[CalcType | str] | None = None,
    ) -> Session:
        """Open a session for a child program under this calculation's grant.

        Args:
            spec: The child's input, or its spec with ``structure``.
            structure: With a spec: the child's atoms, charge, multiplicity and first
                geometry.
            threads: This session's share of the grant (a driver with 16 threads and
                8 tasks gives each session 2). None: the child's own settings.
            calctypes: The calctypes the session may run (see `session_calctypes`).

        Returns:
            The session; close it (or use it as a context manager) when done. A
            driver may call its sessions from several threads at once: qccompute makes
            that safe, serializing sessions of programs that are not thread-safe.

        Raises:
            RuntimeError: The runner provides neither sessions nor a compute callback.
        """
        if isinstance(spec, ProgramInput):
            inp = spec
        else:
            if structure is None:
                raise TypeError("A session from a ProgramSpec needs its structure.")
            inp = ProgramInput.from_spec(spec, structure)
        allowed = session_calctypes(calctypes) if calctypes else None
        if self.open_session is not None:
            return self.open_session(inp, threads, allowed)
        if self.compute is not None:
            return ComputeSession(self.compute, inp, calctypes=allowed)
        raise RuntimeError(
            "This calculation computes child calculations, but its runner provided "
            "neither sessions nor a compute callback."
        )


@runtime_checkable
class Session(Protocol):
    """The same program and molecule at many geometries, warm-started.

    Opened from a `ProgramInput` (``linus.Session(inp)`` directly, or
    `SessionProgram.open_session` under qccompute). The input pins the atoms, charge,
    multiplicity, model and keywords (method, basis, grids, precision) for the session's
    lifetime, so all its evaluations lie on one potential energy surface; anything else
    needs a new session. A request may vary only how tightly it converges
    (`qcdata.Request.accuracy`), which history it continues (``lineage``, ``seed``,
    ``commit``) and which optional outputs it adds (``outputs``). Contract:

    - Results are `Frame` objects: the structure evaluated, its `SinglePointData`, and
      the request's own ``warnings`` and ``record`` (an SCF escalation, a stability
      probe, a fallback), with the shared ``warm_start`` at the top of the record.
      ``program`` and ``program_version`` are the session's.
    - Lineages are any JSON value, keyed by `qcdata.lineage_key`; None is the default
      lineage. A lineage's history is its last committed success.
    - A failed request raises `ProgramError` (with its failed frame) from `compute`, and
      is a failed `Frame` from `compute_many`. The session stays usable and the
      lineage's history is untouched: a driver may retry a smaller step.
    - A ``hessian`` request returns the energy and gradient too.

    Attributes:
        program: The program's name.
        program_version: The program's version.
    """

    @property
    def program(self) -> str:
        """The program's name."""
        ...

    @property
    def program_version(self) -> str:
        """The program's version."""
        ...

    @property
    def input_data(self) -> ProgramInput:
        """The input the session was opened with: its atoms, settings and defaults."""
        ...

    def compute(
        self,
        positions: np.ndarray | None = None,
        calctype: CalcType | str | None = None,
        *,
        accuracy: Accuracy | float | None = None,
        lineage: Lineage = None,
        outputs: Iterable[str] = (),
        seed: Iterable[Lineage] = (),
        commit: bool = True,
    ) -> Frame:
        """Evaluate one geometry.

        Args:
            positions: Atomic positions (bohr), n_atoms x 3; None: the input's own.
            calctype: ``energy``, ``gradient`` or ``hessian``; None: the input's.
            accuracy: See `qcdata.Request.accuracy`.
            lineage: See `qcdata.Request.lineage`.
            outputs: See `qcdata.Request.outputs`.
            seed: See `qcdata.Request.seed`.
            commit: See `qcdata.Request.commit`.

        Returns:
            The evaluation, successful.

        Raises:
            ProgramError: The evaluation failed; the session stays usable.
        """
        ...

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """Evaluate many geometries (finite differences, path images, IRC branches).

        Requests of one lineage run in list order, each reading the lineage's history
        as it stands at its place (`qcdata.Request.commit`); requests of different
        lineages may run concurrently within the grant. The results equal running the
        requests one at a time, each with the thread count it got.

        Args:
            requests: The evaluations.

        Returns:
            One frame per request, in order; a failed request is a failed frame with
            its ``error``.
        """
        ...

    def reset(self, lineage: Lineage = ...) -> None:
        """Drop warm-start and extrapolation history, after a large jump or a failure.

        Args:
            lineage: The lineage to forget; None is the default lineage. Omitted: all.
        """
        ...

    def plan(self) -> Plan:
        """The settings pinned for the session's lifetime, its resources and record."""
        ...

    def close(self) -> None:
        """Release the session's resources."""
        ...

    def __enter__(self) -> Session:
        """Use the session as a context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session."""
        ...


def failed_frame(structure: Structure, error: ProgramError, program: str) -> Frame:
    """The failed `Frame` of a session request, for `Session.compute_many`.

    Args:
        structure: The structure that was evaluated.
        error: The request's failure.
        program: The session's program, for the empty results' provenance.

    Returns:
        The error's own frame when it has one; else a frame with ``success=False``,
        the error, and any partial results.
    """
    if error.frame is not None:
        return error.frame
    results = error.results or SinglePointData(provenance=Provenance(program=program))
    return Frame(
        structure=structure,
        results=results,
        success=False,
        traceback=error.error.message,
        error=error.error,
    )


@runtime_checkable
class Program(Protocol):
    """A program qccompute can run: the object registered under ``qccompute.programs``.

    Structural: a program implements these methods without importing qccompute.

    Attributes:
        name: The program's name, as in ``ProgramInput.program``.
    """

    name: str

    def version(self) -> str:
        """The program's version."""
        ...

    def capabilities(self) -> Capabilities | Mapping[str, Any]:
        """What the program can do and how it may be run (a mapping is validated)."""
        ...

    def settings_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The JSON Schema of ``keywords`` (and of ``resolved.settings``) for a calctype.

        Every key carries ``x-unit``, ``x-tier`` (common, expert), ``x-concern``
        (model, approximation, convergence, algorithm, resource, output) and
        ``x-applies-to`` where they apply.
        """
        ...

    def record_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The JSON Schema of ``ProgramOutput.record`` for a calctype."""
        ...

    def plan(self, inp: ProgramInput) -> Plan | Mapping[str, Any]:
        """Validate an input and resolve its settings without running it.

        A mapping is validated as a `Plan`: ``resolved`` (a `qcdata.Resolved`),
        ``resources``, ``warnings`` and the program's own ``record``.

        Raises:
            ProgramError: The input cannot run (kind ``input`` or ``unsupported``),
                with a did-you-mean suggestion for unknown keys.
        """
        ...

    def compute(
        self, inp: ProgramInput, ctx: RunContext
    ) -> ProgramOutput[ProgramInput, Data]:
        """Run a calculation.

        A failed calculation returns ``success=False`` with `ProgramOutput.error` and
        the partial results; it raises only on bugs.
        """
        ...


@runtime_checkable
class SessionProgram(Program, Protocol):
    """A program that keeps state between geometries (``capabilities().sessions``)."""

    def open_session(
        self,
        inp: ProgramInput,
        ctx: RunContext,
        calctypes: frozenset[CalcType] | None = None,
    ) -> Session:
        """Open a session at the input's atoms and first geometry.

        Usually ``return Session(inp, calctypes=calctypes, ctx=ctx)``: the program's
        own session class is the protocol object. ``calctypes`` are the calculation
        types the session may run (None: everything the method supports; see
        `session_calctypes`). The keywords are
        checked against them: a setting that applies to none of them is an input error,
        and so is a later request of another type.

        Raises:
            ProgramError: The input cannot run (kind ``input`` or ``unsupported``).
        """
        ...


@runtime_checkable
class BatchProgram(Program, Protocol):
    """A program that evaluates many inputs at once (``capabilities().batches``)."""

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext
    ) -> list[ProgramOutput[ProgramInput, Data]]:
        """Run many calculations together; one output per input, in order."""
        ...


class ComputeSession:
    """A session over a compute callback, for programs without sessions of their own.

    Each request becomes the session's input at the new positions and calctype,
    computed by `compute` (a program's `Program.compute` under a context, or a runner's
    callback). It keeps no history: every request starts fresh (``warm_start`` false),
    so ``lineage``, ``seed`` and ``commit`` change nothing. A ``hessian`` result without
    a gradient is completed by a gradient calculation.

    Args:
        compute: Computes one input (or a list) and returns outputs, failed or not.
        inp: The program, model, keywords and structure; its geometry and calctype are
            the defaults of a request.
        calctypes: The calctypes the session may run (see `session_calctypes`; by
            default energy, gradient and Hessian, which the program may still refuse).
    """

    def __init__(
        self,
        compute: Callable[..., Any],
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None = None,
    ):
        """Create the session."""
        if inp.structure is None:
            raise TypeError("A session's input needs a structure.")
        self._compute = compute
        self._inp = inp
        self._structure: Structure = inp.structure
        self.calctypes = session_calctypes(calctypes)
        self.program = inp.program
        self.program_version = ""
        self._closed = False

    @property
    def input_data(self) -> ProgramInput:
        """The input the session was opened with."""
        return self._inp

    def _input(
        self, positions: np.ndarray | None, calctype: CalcType | str | None
    ) -> ProgramInput:
        calctype = CalcType(calctype) if calctype is not None else self._inp.calctype
        check_session_calctype(calctype, self.calctypes)
        if positions is None:
            return self._inp.model_copy(update={"calctype": calctype})
        geometry = np.asarray(positions, dtype=float).reshape(-1, 3)
        if geometry.shape != np.asarray(self._structure.geometry).shape:
            raise ProgramError.of(
                "input",
                f"positions have {geometry.shape[0]} atoms; the session's structure "
                f"has {len(self._structure.symbols)}",
            )
        # The input is already valid; copies skip validating it again on every step.
        structure = self._structure.model_copy(update={"geometry": geometry})
        return self._inp.model_copy(
            update={"calctype": calctype, "structure": structure}
        )

    def _frame(self, out: ProgramOutput[Any, Any]) -> Frame:
        """The frame of a successful output; raise the failure of a failed one."""
        version = out.results.provenance.program_version
        if version and not self.program_version:
            self.program_version = version
        frame = Frame.model_validate(out)
        frame = frame.model_copy(
            update={"record": {"warm_start": False, **frame.record}}
        )
        if not out.success:
            error = out.error or ErrorInfo(
                kind=ErrorKind.internal, message=out.traceback or "failed"
            )
            raise ProgramError(error, frame=frame)
        return frame

    def _complete(self, frame: Frame, inp: ProgramInput) -> Frame:
        """Add the energy and gradient to a Hessian computed without them."""
        if frame.results.gradient is not None:
            return frame
        gradient = self._frame(
            self._compute(inp.model_copy(update={"calctype": CalcType.gradient}))
        ).results
        results = frame.results.model_copy(
            update={"gradient": gradient.gradient, "energy": gradient.energy}
        )
        return frame.model_copy(update={"results": results})

    def compute(
        self,
        positions: np.ndarray | None = None,
        calctype: CalcType | str | None = None,
        *,
        accuracy: Accuracy | float | None = None,
        lineage: Lineage = None,
        outputs: Iterable[str] = (),
        seed: Iterable[Lineage] = (),
        commit: bool = True,
    ) -> Frame:
        """Evaluate one geometry; see `Session.compute`.

        Args:
            positions: Atomic positions (bohr); None: the input's own.
            calctype: ``energy``, ``gradient`` or ``hessian``; None: the input's.
            accuracy: Ignored: a stateless session runs its input's settings.
            lineage: Ignored: a stateless session keeps no history.
            outputs: Not supported; must be empty.
            seed: Ignored, like ``lineage``.
            commit: Ignored, like ``lineage``.

        Returns:
            The evaluation.

        Raises:
            ProgramError: The calculation failed, or outputs were requested.
        """
        if self._closed:
            raise RuntimeError("The session is closed.")
        if tuple(outputs):
            raise ProgramError.of(
                "unsupported",
                "A session over compute takes optional outputs from the input's "
                "keywords, not per request.",
            )
        inp = self._input(positions, calctype)
        frame = self._frame(self._compute(inp))
        if inp.calctype == CalcType.hessian:
            frame = self._complete(frame, inp)
        return frame

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """Evaluate many geometries in one call to the callback.

        Args:
            requests: The evaluations.

        Returns:
            One frame per request, in order; failures as failed frames.
        """
        if self._closed:
            raise RuntimeError("The session is closed.")
        frames: list[Frame | None] = [None] * len(requests)
        inputs: list[ProgramInput] = []
        places: list[int] = []
        for i, r in enumerate(requests):
            try:
                if r.outputs:
                    raise ProgramError.of(
                        "unsupported", "per-request outputs need a program session"
                    )
                inputs.append(self._input(r.positions, r.calctype))
                places.append(i)
            except ProgramError as e:
                geometry = (
                    self._structure.geometry if r.positions is None else r.positions
                )
                structure = self._structure.model_copy(
                    update={
                        "geometry": np.asarray(geometry, dtype=float).reshape(-1, 3)
                    }
                )
                frames[i] = failed_frame(structure, e, self.program)
        outputs = self._compute(inputs) if inputs else []
        for i, inp, out in zip(places, inputs, outputs, strict=True):
            try:
                frame = self._frame(out)
                if inp.calctype == CalcType.hessian:
                    frame = self._complete(frame, inp)
                frames[i] = frame
            except ProgramError as e:
                assert inp.structure is not None
                frames[i] = failed_frame(inp.structure, e, self.program)
        return [f for f in frames if f is not None]

    def reset(self, lineage: Lineage = ...) -> None:
        """Nothing to reset: a stateless session keeps no history."""

    def plan(self) -> Plan:
        """Not known to a stateless session: the input's keywords, unresolved."""
        return Plan(resolved=Resolved(settings=dict(self._inp.keywords)))

    def close(self) -> None:
        """Close the session."""
        self._closed = True

    def __enter__(self) -> ComputeSession:
        """Use the session as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session."""
        self.close()


def local_context(
    programs: Mapping[str, Program] | None = None, *, threads: int | None = None
) -> RunContext:
    """A context that runs a driver's children in this process, one at a time.

    For calling a driver directly (tests, notebooks) without a runner. Children run
    through the given programs, else the installed ones (`find_program`): sessions of
    their own when they have them, else a `ComputeSession` over their `compute`.
    qccompute provides the full runner (parallel children, codecs, the cluster).

    Args:
        programs: Programs by name, ahead of the installed ones.
        threads: Threads granted to the driver; children get the same.

    Returns:
        The context.
    """
    known = dict(programs or {})

    def lookup(name: str) -> Program:
        program = known.get(name) or find_program(name)
        if program is None:
            raise ProgramError.of(
                "environment", f"No program {name!r} is installed or given."
            )
        return program

    def child_context() -> RunContext:
        return local_context(known, threads=threads)

    def compute_one(inp: ProgramInput) -> ProgramOutput[ProgramInput, Data]:
        try:
            program = lookup(inp.program)
        except ProgramError as e:
            return ProgramOutput(
                input_data=inp,
                success=False,
                traceback=e.error.message,
                error=e.error,
                results=SinglePointData(provenance=Provenance(program=inp.program))
                if inp.calctype
                in (CalcType.energy, CalcType.gradient, CalcType.hessian)
                else _empty_results(inp),
            )
        return program.compute(inp, child_context())

    def compute(inputs: Any, previous: Any = None) -> Any:
        if isinstance(inputs, ProgramInput):
            return compute_one(inputs)
        return [compute_one(i) for i in inputs]

    def open_session(
        inp: ProgramInput, share: int | None, calctypes: frozenset[CalcType] | None
    ) -> Session:
        program = lookup(inp.program)
        if isinstance(program, SessionProgram) and capabilities_of(program).sessions:
            return program.open_session(inp, RunContext(threads=share), calctypes)
        return ComputeSession(compute, inp, calctypes=calctypes)

    return RunContext(
        threads=threads,
        compute=compute,  # type: ignore[arg-type]
        open_session=open_session,
    )


def _empty_results(inp: ProgramInput) -> Any:
    return get_data_type(inp.calctype)(provenance=Provenance(program=inp.program))


_registered: dict[str, Program] = {}
_installed: dict[str, Program | None] = {}
_registry_lock = threading.Lock()


@functools.cache
def _entry_points() -> dict[str, EntryPoint]:
    """The ``qccompute.programs`` entry points by name, read once per process."""
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def _require_program(obj: object, what: str) -> Program:
    """`obj` if it implements `Program`; a class passes the protocol check, so refuse it."""
    if isinstance(obj, type):
        raise TypeError(
            f"{what} is the class {obj.__qualname__}, not a program: use an instance."
        )
    if not isinstance(obj, Program):
        raise TypeError(f"{what} does not implement qccompute.program.Program.")
    return obj


def register_program(program: Program) -> None:
    """Make a program findable by name in this process, ahead of installed ones.

    For programs that are not installed packages: in tests, notebooks and scripts. Drivers'
    plans, `local_context` and qccompute all find it through `find_program`.

    Args:
        program: An object implementing `Program`.

    Raises:
        TypeError: The object does not implement `Program`, or is a class.
    """
    _require_program(program, repr(program))
    with _registry_lock:
        _registered[program.name] = program


def unregister_program(name: str) -> None:
    """Forget a program added with `register_program`; installed programs stay.

    Args:
        name: The program's name.
    """
    with _registry_lock:
        _registered.pop(name, None)


def available_programs() -> list[str]:
    """Names of the registered and installed programs."""
    with _registry_lock:
        registered = set(_registered)
    return sorted(registered | set(_entry_points()))


def find_program(name: str) -> Program | None:
    """The program of this name: registered in this process, else installed.

    Installed programs are read from the ``qccompute.programs`` entry points and loaded
    once per process.

    Args:
        name: The program's name, as in ``ProgramInput.program``.

    Returns:
        The program object, or None if none is registered or installed.

    Raises:
        TypeError: The installed object does not implement `Program`, or is a class.
    """
    with _registry_lock:
        if name in _registered:
            return _registered[name]
        if name in _installed:
            return _installed[name]
    ep = _entry_points().get(name)
    program: Program | None = None
    if ep is not None:
        program = _require_program(
            ep.load(), f"The qccompute.programs entry point {name!r} ({ep.value})"
        )
    with _registry_lock:
        return _installed.setdefault(name, program)


def capabilities_of(program: Program) -> Capabilities:
    """A program's capabilities, validated: it may return a mapping or any object.

    Args:
        program: The program.

    Returns:
        Its `Capabilities`.
    """
    return Capabilities.model_validate(program.capabilities(), from_attributes=True)


def plan_of(program: Program, inp: ProgramInput) -> Plan:
    """A program's plan for an input, validated: it may return a mapping or any object.

    Args:
        program: The program.
        inp: The input.

    Returns:
        The `Plan`.

    Raises:
        ProgramError: The input cannot run.
    """
    return Plan.model_validate(program.plan(inp), from_attributes=True)
