"""The protocol a program implements to run under qccompute, and its sessions.

A program package registers one object (an instance, not its class) under the
``qccompute.programs`` entry point group::

    [project.entry-points."qccompute.programs"]
    mychem = "mychem.qc:program"

The object implements `Program`: it says what it can do (`capabilities`), publishes the
JSON Schema of its settings (`settings_schema`) and of its run record (`record_schema`),
plans a calculation without running it (`plan`: validated, every ``"auto"`` resolved, the
resources it needs), and computes it (`compute`). A failed calculation raises
`qcdata.ProgramError` with its partial results (``ProgramError(error, output=...)``
for a whole failed output). Programs that keep state between
geometries (warm starts, a pinned grid) also implement `SessionProgram.open_session`;
programs that evaluate many inputs at once implement `BatchProgram.compute_batch`.

``compute`` takes an optional second argument, a `RunContext`, which only a runner
passes, and a runner always passes one: the calculation's share of the machine
(threads, GPUs, memory, time) and the way to run subprograms (`RunContext.compute`,
`RunContext.session`). ``ctx is None`` means a direct call: the program uses its own
settings, and a program with subprograms hands itself to the default runner::

    def compute(self, inp, ctx=None):
        if ctx is None:
            import qccompute

            return qccompute.compute(inp)  # the default runner passes a ctx
        ...

The runner finds it by name, so it must be installed or registered
(`register_program`); otherwise the call fails with kind ``environment`` and says how.

The runner never changes a number by itself. A grant may change a program's resolved
choices (DF storage from the memory granted, a precision from the device); the program
records each one in its output's ``resolved``.

`FunctionProgram` makes a program of plain Python functions (a model surface, an ML
potential in a script): the smallest complete example of the protocol.

**A program without subprograms never imports qccompute.** The protocols are structural
(`typing.Protocol`), and what a program returns for `capabilities` and `plan` may be a
plain mapping (the document its command line prints and its worker sends), which
qccompute validates (`capabilities_of`, `plan_of`). What programs share as data is in
qcdata: inputs and outputs, `Frame`, `qcdata.Request`, `qcdata.Accuracy` and
`qcdata.ProgramError`. So ``mychem.Session(inp)`` and ``mychem.compute(inp)`` work without
qccompute, and qccompute runs the same objects.

Programs are found by name through `find_program`. A program that is not an installed
package (in a test, a notebook or a script) is added with `register_program`.

Design: ``docs/design/program-interface.md``; names and units: qcdata's conventions.
"""

from __future__ import annotations

import functools
import hashlib
import shutil
import tempfile
import threading
import time
import traceback
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    ExecutionInfo,
    FileInput,
    Frame,
    Lineage,
    ProgramError,
    ProgramErrors,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    ProgramWarning,
    Provenance,
    Request,
    Resolved,
    SinglePointData,
    Structure,
)
from qcdata.models.base_models import QCDataBaseModel

__all__ = [
    "ENTRY_POINT_GROUP",
    "FunctionProgram",
    "Capabilities",
    "Resources",
    "Plan",
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
    "available_programs",
    "capabilities_of",
    "numerics_of",
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
        gpu: Whether it can use GPUs; the runner then grants it GPUs.
    """

    calctypes: frozenset[CalcType]
    methods: dict[str, frozenset[CalcType]] | None = None
    sessions: bool = False
    batches: bool = False
    in_process: bool = True
    thread_safe: bool = False
    gpu: bool = False


class Resources(QCDataBaseModel):
    """What one calculation needs from the machine; the runner schedules by it.

    Attributes:
        threads: CPU threads the calculation asks for when its input has no
            ``threads`` keyword: the runner grants this many, at most its own. The
            default is 1. None asks for all the runner has (for a subprogram, all of
            its parent's grant): for a program that scales with whatever it is given
            (a DFT code). In a list such a calculation is a full share, so it runs
            alone. A small molecule rarely gains from more than a few.
        memory_gib: Host memory (GiB), if known.
        gpus: Number of GPUs.
        device_memory_gib: Memory per GPU (GiB), if capped; several calculations may
            then share one device.
        exclusive: Whether the calculation must have the machine (or its GPUs) to
            itself, e.g. for clean timings.
    """

    threads: int | None = 1
    memory_gib: float | None = None
    gpus: int = 0
    device_memory_gib: float | None = None
    exclusive: bool = False


class Plan(QCDataBaseModel):
    """A calculation planned but not run.

    Attributes:
        resolved: The settings with every ``"auto"`` replaced, and why; exactly what
            the run's `ProgramOutput.resolved` will hold.
        resources: What the calculation needs; the runner grants its threads (unless
            the input's ``threads`` keyword asks) and packs lists by it. A program with
            subprograms gives the largest over its subprograms' plans.
        warnings: What the program would warn about.
        record: The program's own detail of the plan (a placement, arithmetic
            and memory plan), shaped like a run's record.
    """

    resolved: Resolved
    resources: Resources = Resources()
    warnings: list[ProgramWarning] = []
    record: dict[str, Any] = {}


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


class _Subprograms(Protocol):
    """What runs a calculation's subprograms: a `Runner`, or a worker's client."""

    def _subcompute(self, inputs: Any, ctx: RunContext, raise_exc: bool) -> Any: ...

    def _subsession(
        self,
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None,
        threads: int | None,
        ctx: RunContext,
    ) -> Session: ...


class RunContext:
    """What a runner gives one calculation: its share of the machine and its subprograms.

    Runners make it and always pass one to `Program.compute`; a program reads its
    fields. A calculation's share of its runner is the larger of its share of the
    threads and its share of the GPUs (with k calculations per GPU, each counts 1/k of
    a GPU); host and device memory follow the share.

    A program uses the smaller of its own request (its ``threads`` or ``memory_gib``
    keyword) and the grant, or the grant when it requested nothing; a request capped by
    the grant is a ``resource_capped`` warning, not an error.

    Attributes:
        threads: CPU threads granted: what the input asks for (its ``threads``
            keyword, else ``plan.resources.threads``; all of them when that is
            None), at most the runner's. None: not
            granted (a direct call).
        gpus: Identifiers of the GPUs granted (as for ``CUDA_VISIBLE_DEVICES``).
        memory_gib: Host memory granted (GiB): the calculation's share of the
            runner's memory. None: not known.
        device_memory_gib: Memory granted on each GPU (GiB): 1/k of it when k
            calculations share a GPU. None: not known or not granted.
        plan: The runner's plan of this input, when it made one, so the program need
            not plan again; exactly what ``program.plan(input)``
            returned.
    """

    __slots__ = (
        "threads",
        "gpus",
        "memory_gib",
        "device_memory_gib",
        "plan",
        "_deadline",
        "_scratch",
        "_made",
        "_runner",
    )

    def __init__(
        self,
        *,
        threads: int | None = None,
        gpus: Sequence[str] = (),
        memory_gib: float | None = None,
        device_memory_gib: float | None = None,
        timeout: float | None = None,
        plan: Plan | None = None,
        scratch: Path | None = None,
        runner: _Subprograms | None = None,
    ) -> None:
        """Create a context; runners do this, programs read it.

        Args:
            threads: CPU threads granted.
            gpus: GPU identifiers granted.
            memory_gib: Host memory granted (GiB).
            device_memory_gib: Memory granted on each GPU (GiB).
            timeout: Seconds the calculation may run, from now.
            plan: The runner's plan of the input.
            scratch: A directory the calculation may write to, kept afterwards; None:
                a temporary one made on first use of `scratch`.
            runner: What runs subprograms (a runner, or in a worker its client); None:
                the default runner.
        """
        self.threads = threads
        self.gpus: tuple[str, ...] = tuple(gpus)
        self.memory_gib = memory_gib
        self.device_memory_gib = device_memory_gib
        self.plan = plan
        self._deadline = None if timeout is None else time.monotonic() + timeout
        self._scratch = scratch
        self._made = False
        self._runner = runner

    def __repr__(self) -> str:
        """Show the grant."""
        return (
            f"RunContext(threads={self.threads}, gpus={self.gpus}, "
            f"memory_gib={self.memory_gib}, timeout={self.timeout})"
        )

    @property
    def timeout(self) -> float | None:
        """Seconds the calculation has left, if the caller set a limit.

        A program that starts processes kills them when it runs out and fails with
        kind ``timeout``. The runner enforces it itself only for programs in worker
        processes. Subprograms get what is left.
        """
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    @property
    def scratch(self) -> Path:
        """A directory the calculation may write to, made on first use.

        A temporary directory made here is removed when the runner's call returns; a
        program called directly with ``RunContext()`` removes it itself (or uses its
        own temporary directory).
        """
        if self._scratch is None:
            self._scratch = Path(tempfile.mkdtemp(prefix="qccompute-"))
            self._made = True
        return self._scratch

    def cleanup(self) -> None:
        """Remove the scratch directory if this context made it (runners call this)."""
        if self._made and self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None
            self._made = False

    def _get_runner(self) -> _Subprograms:
        """The runner of this calculation's subprograms: its own, else the default."""
        if self._runner is None:
            from .runner import default_runner

            return default_runner()
        return self._runner

    @overload
    def compute(
        self, inputs: ProgramInput | FileInput, /, *, raise_exc: bool = True
    ) -> ProgramOutput[Any, Data]: ...

    @overload
    def compute(
        self,
        inputs: Sequence[ProgramInput | FileInput],
        /,
        *,
        raise_exc: bool = True,
    ) -> list[ProgramOutput[Any, Data]]: ...

    def compute(self, inputs: Any, /, *, raise_exc: bool = True) -> Any:
        """Run subprograms within this calculation's grant, as ``Runner.compute`` does.

        Each gets the threads it asks for (its ``threads`` keyword, else its plan's),
        at most the grant; a list runs side by side within the grant. Failures raise
        as from ``qccompute.compute``: `qcdata.ProgramError` for one input,
        `qcdata.ProgramErrors` for a list; ``raise_exc=False`` returns them in place.
        A subprogram's failure that the program lets escape fails it with kind
        ``program`` and the subprogram's error as ``cause``.

        Args:
            inputs: One input, or a list of independent inputs.
            raise_exc: Raise for failures; False returns failed outputs.

        Returns:
            The output, or the outputs in input order.
        """
        return self._get_runner()._subcompute(inputs, self, raise_exc)

    def session(
        self,
        spec: ProgramInput | ProgramSpec,
        structure: Structure | None = None,
        *,
        threads: int | None = None,
        calctypes: Iterable[CalcType | str] | None = None,
    ) -> Session:
        """Open a session of a subprogram under this calculation's grant.

        Args:
            spec: The subprogram's input, or its spec with ``structure``.
            structure: With a spec: the atoms, charge, multiplicity and first
                geometry.
            threads: This session's share of the grant (a program with 16 threads
                running 8 tasks gives each session 2), which its host memory follows;
                None: the threads the subprogram asks for (its ``threads`` keyword,
                else its plan's), at most the grant.
            calctypes: The calctypes the session may run (see `session_calctypes`).

        Returns:
            The session; close it (or use it as a context manager) when done. It may be
            called from several threads at once: the runner serializes sessions of
            programs that are not thread-safe.
        """
        if isinstance(spec, ProgramInput):
            inp = spec
        else:
            if structure is None:
                raise TypeError("A session from a ProgramSpec needs its structure.")
            inp = ProgramInput.from_spec(spec, structure)
        return self._get_runner()._subsession(inp, calctypes, threads, self)


@runtime_checkable
class Session(Protocol):
    """The same program and molecule at many geometries, warm-started.

    Opened from a `ProgramInput` (``mychem.Session(inp)`` directly, or
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
    - A failed request raises `ProgramError` (with its failed frame) from `compute`;
      `compute_many` runs every request and then raises `qcdata.ProgramErrors` holding
      the failures and every frame in order. The session stays usable and the lineage's
      history is untouched: a caller may retry a smaller step.
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

        Every request runs, failed ones included; then, if any failed, the call
        raises one `qcdata.ProgramErrors` group holding each failure's
        `qcdata.ProgramError` and every frame in request order (failed frames in
        place).

        Args:
            requests: The evaluations.

        Returns:
            One frame per request, in order, all successful.

        Raises:
            ProgramErrors: Some requests failed; ``results`` holds every frame.
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
    results = (
        error.results
        if isinstance(error.results, SinglePointData)
        else SinglePointData(provenance=Provenance(program=program))
    )
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

    Optional: ``numerics(spec) -> str``, which changes whenever results for an
    unchanged input may change (see `numerics_of`); without it the version stands for
    it.

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
        self, inp: ProgramInput, ctx: RunContext | None = None
    ) -> ProgramOutput[ProgramInput, Data]:
        """Run a calculation.

        Args:
            inp: The input.
            ctx: The runner's grant and subprograms; a runner always passes one. None:
                a direct call, after which a program with subprograms returns
                ``qccompute.compute(inp)``.

        Returns:
            The successful output.

        Raises:
            ProgramError: The calculation failed, with its partial results or its
                whole failed output (``output``).
        """
        ...


@runtime_checkable
class SessionProgram(Program, Protocol):
    """A program that keeps state between geometries (``capabilities().sessions``)."""

    def open_session(
        self,
        inp: ProgramInput,
        ctx: RunContext | None = None,
        calctypes: frozenset[CalcType] | None = None,
    ) -> Session:
        """Open a session at the input's atoms and first geometry.

        Usually ``return Session(inp, calctypes=calctypes, ctx=ctx)``: the program's
        own session class is the protocol object. ``calctypes`` are the calculation
        types the session may run (None: everything the method supports; see
        `session_calctypes`). The keywords are checked against them: a setting that
        applies to none of them is an input error, and so is a later request of
        another type.

        Raises:
            ProgramError: The input cannot run (kind ``input`` or ``unsupported``).
        """
        ...


@runtime_checkable
class BatchProgram(Program, Protocol):
    """A program that evaluates many inputs at once (``capabilities().batches``)."""

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext | None = None
    ) -> list[ProgramOutput[ProgramInput, Data]]:
        """Run many calculations together.

        Returns:
            One output per input, in order; a failed calculation is a failed output
            in its place (``ProgramOutput.failure``), and so is an input the program
            refuses (kind ``input``), so the other inputs still run. The runner
            raises the failures together, or returns them in place.
        """
        ...


class ComputeSession:
    """A session over compute, for programs without sessions of their own.

    Each request becomes the session's input at the new positions and calctype,
    computed by ``compute``: a runner's ``compute`` (one input raises
    `qcdata.ProgramError` on failure; a list with ``raise_exc=False`` returns failed
    outputs in place). It keeps no history: every request starts fresh
    (``warm_start`` false), so ``lineage``, ``seed`` and ``commit`` change nothing. A
    ``hessian`` result without a gradient is completed by a gradient calculation.

    Args:
        compute: Computes one input, or a list with ``raise_exc=False``.
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

    def _as_frame(self, out: ProgramOutput[Any, Any]) -> Frame:
        """An output as a frame, with ``warm_start`` false in its record."""
        version = out.results.provenance.program_version
        if version and not self.program_version:
            self.program_version = version
        frame = Frame.model_validate(out)
        return frame.model_copy(
            update={"record": {"warm_start": False, **frame.record}}
        )

    def _one(self, inp: ProgramInput) -> Frame:
        """Compute one input as a frame; a failure raises with its failed frame."""
        try:
            out = self._compute(inp)
        except ProgramError as e:
            failed = e.output
            if e.frame is None and isinstance(failed, ProgramOutput):
                e.frame = self._as_frame(failed)
            raise
        frame = self._as_frame(out)
        if inp.calctype == CalcType.hessian and frame.results.gradient is None:
            gradient = self._one(
                inp.model_copy(update={"calctype": CalcType.gradient})
            ).results
            results = frame.results.model_copy(
                update={"gradient": gradient.gradient, "energy": gradient.energy}
            )
            frame = frame.model_copy(update={"results": results})
        return frame

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
        return self._one(self._input(positions, calctype))

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """Evaluate many geometries in one call to compute.

        Args:
            requests: The evaluations.

        Returns:
            One frame per request, in order.

        Raises:
            ProgramErrors: Some requests failed; ``results`` holds every frame.
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
        outputs = self._compute(inputs, raise_exc=False) if inputs else []
        for i, inp, out in zip(places, inputs, outputs, strict=True):
            try:
                if not out.success:
                    frame = self._as_frame(out)
                elif inp.calctype == CalcType.hessian and out.results.gradient is None:
                    frame = self._one(inp)
                else:
                    frame = self._as_frame(out)
                frames[i] = frame
            except ProgramError as e:
                assert inp.structure is not None
                frames[i] = failed_frame(inp.structure, e, self.program)
        return ProgramErrors.check([f for f in frames if f is not None])

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

    For programs that are not installed packages: in tests, notebooks and scripts.
    Runners in this process find it through `find_program`, and so do the
    subprograms of a program in a worker process (geomeTRIC's gradients), which run
    back in this process. A registered program that itself needs a worker process
    (``in_process=False``, or a runner with ``isolation="process"``) must be a
    top-level attribute of an importable module, which the worker imports.

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
    """Names of the programs qccompute can find, without loading any.

    The programs registered in this process and those installed packages offer
    under the ``qccompute.programs`` entry points. A package may offer a program whose
    executable is not on this machine (qccodec offers every external program it has
    an adapter for: ORCA, Psi4, TeraChem, ...); running it then fails with kind
    ``environment``. An external program's ``numerics`` says ``not installed`` when its
    executable cannot be found.

    Returns:
        The names, sorted.
    """
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


def numerics_of(program: Program, spec: ProgramSpec | FileInput) -> str:
    """A program's numerics identity for a spec: ``program.numerics(spec)``, else its version.

    ``numerics(spec) -> str`` is optional in the protocol: a string that changes
    whenever the program's results for an unchanged input may change (a numerics
    epoch the program bumps; an external program: its executable's sha256; an ML potential: its
    model's checksum). It takes the spec (an input, or a subprogram's `ProgramSpec`),
    so a program of several methods can give each its own. A store files every run
    under the identities of the programs that made it, and a runner reuses a run only
    when they are those it would run now.

    Args:
        program: The program.
        spec: The input, or the subprogram's spec, it would run.

    Returns:
        The identity.
    """
    numerics = getattr(program, "numerics", None)
    if callable(numerics):
        return str(numerics(spec))
    return program.version()


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


_EMPTY_PLAN = Plan(resolved=Resolved())


class _FunctionDigest:
    """A sha256 of what Python functions compute with, for their numerics identity.

    It covers each function's bytecode and constants, its default arguments, the
    values in its closure, and the module globals its code reads, recursing into the
    Python functions among them (a helper the function calls). Plain data (numbers,
    strings, containers, numpy arrays) counts by value; modules, classes and built-in
    functions by name; any other object by its type and its attributes. The digest
    changes when the code or any of these values does, and with the Python version,
    whose bytecode differs.
    """

    def __init__(self) -> None:
        self._sha = hashlib.sha256()
        self._seen: set[int] = set()

    def hexdigest(self) -> str:
        return self._sha.hexdigest()

    def _text(self, text: str) -> None:
        self._sha.update(text.encode("utf-8", "backslashreplace") + b"\0")

    def feed(self, value: Any, depth: int = 0) -> None:
        """Add a value to the digest."""
        if depth > 40:
            self._text("<deep>")
        elif value is None or isinstance(value, bool | int | float | complex | str):
            self._text(f"{type(value).__name__}:{value!r}")
        elif isinstance(value, bytes | bytearray):
            self._text(f"bytes:{len(value)}")
            self._sha.update(value)
        elif isinstance(value, np.ndarray):
            self._text(f"ndarray:{value.dtype.str}:{value.shape}")
            if value.dtype.hasobject:
                self.feed(value.tolist(), depth + 1)
            else:
                self._sha.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, np.generic):
            self._text(f"{type(value).__name__}:{value!r}")
        elif isinstance(value, types.CodeType):
            self._text("code")
            self._sha.update(value.co_code)
            self._text(repr(value.co_names))
            for constant in value.co_consts:
                self.feed(constant, depth + 1)
        elif isinstance(value, tuple | list):
            self._text(f"{type(value).__name__}:{len(value)}")
            for item in value:
                self.feed(item, depth + 1)
        elif isinstance(value, set | frozenset):
            self._text(f"set:{len(value)}")
            for item in sorted(_digest_of(v, depth + 1) for v in value):
                self._text(item)
        elif isinstance(value, Mapping):
            self._text(f"mapping:{len(value)}")
            for k, v in sorted(value.items(), key=lambda kv: repr(kv[0])):
                self.feed(k, depth + 1)
                self.feed(v, depth + 1)
        elif id(value) in self._seen:
            self._text("<seen>")
        else:
            self._seen.add(id(value))
            self._object(value, depth)

    def _object(self, value: Any, depth: int) -> None:
        if isinstance(value, types.FunctionType):
            self._text(f"function:{value.__qualname__}")
            self.feed(value.__code__, depth + 1)
            self.feed(value.__defaults__, depth + 1)
            self.feed(value.__kwdefaults__, depth + 1)
            for cell in value.__closure__ or ():
                try:
                    self.feed(cell.cell_contents, depth + 1)
                except ValueError:  # an empty cell
                    self._text("<empty>")
            for name in sorted(_names(value.__code__)):
                if name in value.__globals__:
                    self._text(f"global:{name}")
                    self.feed(value.__globals__[name], depth + 1)
        elif isinstance(value, types.MethodType):
            self.feed(value.__func__, depth + 1)
            self.feed(value.__self__, depth + 1)
        elif isinstance(value, functools.partial):
            self._text("partial")
            self.feed(value.func, depth + 1)
            self.feed(value.args, depth + 1)
            self.feed(value.keywords, depth + 1)
        elif isinstance(value, types.ModuleType):
            self._text(f"module:{value.__name__}")
        elif isinstance(value, type):
            self._text(f"class:{value.__module__}.{value.__qualname__}")
        elif callable(value) and not hasattr(value, "__dict__"):  # built-in, ufunc
            label = getattr(value, "__qualname__", None) or getattr(
                value, "__name__", ""
            )
            self._text(f"builtin:{getattr(value, '__module__', None)}.{label}")
        else:
            kind = type(value)
            self._text(f"object:{kind.__module__}.{kind.__qualname__}")
            # A callable object's own code: its class's __call__.
            calls = (vars(k)["__call__"] for k in kind.__mro__ if "__call__" in vars(k))
            call = next(calls, None)
            if isinstance(call, types.FunctionType):
                self.feed(call, depth + 1)
            attributes = getattr(value, "__dict__", None)
            if isinstance(attributes, dict):
                self.feed(attributes, depth + 1)


def _names(code: types.CodeType) -> set[str]:
    """The names a code object and the code nested in it read (globals among them)."""
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names |= _names(constant)
    return names


def _digest_of(value: Any, depth: int = 0) -> str:
    digest = _FunctionDigest()
    digest.feed(value, depth)
    return digest.hexdigest()


class FunctionProgram:
    """A program made of Python functions of the positions.

    For a model surface, an ML potential in a script, or teaching: the smallest
    complete program. It has no settings (any keyword is an ``unknown_key`` input
    error) and ignores the model; it runs in process and is thread-safe::

        def morse(x):  # positions (bohr, n x 3) -> energy (Eh), gradient (Eh/bohr)
            ...
            return energy, gradient

        qccompute.register_program(FunctionProgram("morse", morse))
        out = qccompute.compute(ProgramInput(program="morse", calctype="gradient",
                                             structure=h2))

    An exception from a function is a failed calculation of kind ``internal``, with
    the function's traceback in the failed output.

    Args:
        name: The program's name, as in ``ProgramInput.program``.
        energy_gradient: Positions (bohr, n_atoms x 3) to the energy (Eh) and the
            gradient (Eh/bohr, n_atoms x 3).
        hessian: Positions to the Hessian (Eh/bohr^2, 3 n_atoms x 3 n_atoms); without
            it the program computes energies and gradients only.
        version: The program's version and numerics identity, for provenance and the
            store. None: a digest of the functions (``fn-`` and 16 hex digits), taken
            on each use, of their code, constants, defaults, closures and the module
            globals they read, recursing into the Python functions they call. Editing
            a function or a value it reads (an energy scale) changes it, so a store
            never serves results of the old function. An object the functions use
            that is not plain data (an ML model) counts by its type and attributes;
            give ``version`` when that is not enough (a model's checksum).
    """

    _SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(
        self,
        name: str,
        energy_gradient: Callable[[np.ndarray], tuple[float, Any]],
        hessian: Callable[[np.ndarray], Any] | None = None,
        *,
        version: str | None = None,
    ) -> None:
        """Make a program of the functions."""
        self.name = name
        self._energy_gradient = energy_gradient
        self._hessian = hessian
        self._version = version
        calctypes = {CalcType.energy, CalcType.gradient}
        if hessian is not None:
            calctypes.add(CalcType.hessian)
        self._caps = Capabilities(
            calctypes=frozenset(calctypes), in_process=True, thread_safe=True
        )

    def __repr__(self) -> str:
        """Show the program's name."""
        return f"FunctionProgram({self.name!r})"

    def version(self) -> str:
        """The version given, else the digest of the functions as they are now."""
        if self._version is not None:
            return self._version
        return "fn-" + _digest_of((self._energy_gradient, self._hessian))[:16]

    def numerics(self, spec: Any = None) -> str:
        """The numerics identity: the version (by default the functions' digest).

        Args:
            spec: The input or subprogram spec; the functions are the same for all.

        Returns:
            The identity.
        """
        return self.version()

    def capabilities(self) -> Capabilities:
        """Energies and gradients (and Hessians with a Hessian function)."""
        return self._caps

    def settings_schema(self, calctype: CalcType) -> dict[str, Any]:
        """No settings."""
        return self._SCHEMA

    def record_schema(self, calctype: CalcType) -> dict[str, Any]:
        """No record."""
        return {"type": "object", "properties": {}}

    def plan(self, inp: ProgramInput) -> Plan:
        """Check the input: a structure, a supported calctype, no keywords.

        Raises:
            ProgramError: Kind ``input`` (``unknown_key``, no structure) or
                ``unsupported``.
        """
        if inp.calctype not in self._caps.calctypes:
            raise ProgramError.of(
                "unsupported",
                f"{self.name} computes {sorted(c.value for c in self._caps.calctypes)},"
                f" not {inp.calctype.value}.",
            )
        if inp.structure is None:
            raise ProgramError.of("input", f"{self.name} needs a structure.")
        if inp.keywords:
            raise ProgramError.of(
                "input",
                f"{self.name} has no settings; unknown {sorted(inp.keywords)}.",
                code="unknown_key",
            )
        return _EMPTY_PLAN

    def compute(
        self, inp: ProgramInput, ctx: RunContext | None = None
    ) -> ProgramOutput[ProgramInput, Data]:
        """Evaluate the functions at the input's structure.

        Raises:
            ProgramError: The input is refused, or a function raised (kind
                ``internal``; the failed output holds its traceback).
        """
        start = time.perf_counter()
        if ctx is None or ctx.plan is None:  # a runner hands on the plan it made
            self.plan(inp)
        assert inp.structure is not None
        x = np.asarray(inp.structure.geometry, dtype=float).reshape(-1, 3)
        provenance = Provenance(program=self.name, program_version=self.version())
        try:
            energy, gradient = self._energy_gradient(x.copy())
            values: dict[str, Any] = {"energy": float(energy)}
            if inp.calctype != CalcType.energy:
                values["gradient"] = np.asarray(gradient, dtype=float).reshape(-1, 3)
            if inp.calctype == CalcType.hessian:
                assert self._hessian is not None
                values["hessian"] = np.asarray(self._hessian(x.copy()), dtype=float)
            results = SinglePointData(provenance=provenance, **values)
        except Exception as e:  # noqa: BLE001 - the function's failure, with its trace
            error = ErrorInfo(
                kind=ErrorKind.internal, message=f"{type(e).__name__}: {e}"
            )
            out = ProgramOutput.failure(
                inp,
                error,
                traceback="".join(traceback.format_exception(e)),
                execution=ExecutionInfo(threads=1),
            )
            raise ProgramError(error, output=out) from e
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=results,
            resolved=Resolved(),
            execution=ExecutionInfo(
                threads=1, timings={"total": time.perf_counter() - start}
            ),
        )
