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
import os
import shutil
import subprocess
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
from pydantic import BaseModel
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

#: The program that runs a `qcdata.FileInput` (a command with its files, not a
#: `qcdata.ProgramInput`): the one program name the protocol reserves; qccodec registers
#: it. Its numerics identity is a FileInput run's identity.
FILES = "files"


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

    Whether a calculation uses GPUs is its plan's (`Resources.gpus`), not a
    capability: a program whose subprograms use GPUs plans theirs.
    """

    calctypes: frozenset[CalcType]
    methods: dict[str, frozenset[CalcType]] | None = None
    sessions: bool = False
    batches: bool = False
    in_process: bool = True
    thread_safe: bool = False


class Resources(QCDataBaseModel):
    """What one calculation asks of the machine.

    The runner grants the threads and GPUs asked for and packs lists by them. Memory
    is informational: the runner records it in the plan and never schedules,
    packs or caps by it (qc* makes no memory estimates). What memory a calculation is
    granted is in `RunContext`: ``memory_gib`` and ``device_memory_gib``.

    Attributes:
        threads: CPU threads the calculation asks for when its input has no
            ``threads`` keyword: the runner grants this many, at most its own. The
            default is 1. None asks for all the runner has (for a subprogram, all of
            its parent's grant): for a program that scales with whatever it is given
            (a DFT code). In a list such a calculation is a full share, so it runs
            alone. A small molecule rarely gains from more than a few.
        memory_gib: Host memory the calculation expects to use (GiB), if known;
            informational, not scheduled.
        gpus: GPUs the calculation uses: the runner grants this many (a program
            whose subprograms use GPUs plans theirs). A program in a worker sees only
            those granted.
        device_memory_gib: Memory per GPU the calculation expects to use (GiB), if
            known; informational, not scheduled (``per_gpu`` sets how many
            calculations share a GPU).
    """

    threads: int | None = 1
    memory_gib: float | None = None
    gpus: int = 0
    device_memory_gib: float | None = None


class Plan(QCDataBaseModel):
    """A calculation planned but not run.

    Attributes:
        resolved: The settings with every ``"auto"`` replaced, and why; exactly what
            the run's `ProgramOutput.resolved` will hold.
        resources: What the calculation needs; the runner grants its threads (unless
            the input's ``threads`` keyword asks) and GPUs, and packs lists by them;
            its memory is informational. A program with subprograms gives the largest
            over its subprograms' plans.
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


def _cuda_visible_devices() -> list[str] | None:
    """This process's ``CUDA_VISIBLE_DEVICES`` as a list; None when it is unset.

    CUDA stops at the first entry that names no device, so the list does too: a
    negative index (``-1``) or an empty entry ends it (``"-1"`` hides every GPU).

    Returns:
        The identifiers in order (indices, ``GPU-`` or ``MIG-`` UUIDs), or None.
    """
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return None
    ids: list[str] = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry or entry.startswith("-"):
            break
        ids.append(entry)
    return ids


@functools.cache
def _nvidia_smi_gpus() -> tuple[tuple[str, str, float | None, str], ...]:
    """The NVIDIA GPUs ``nvidia-smi`` lists: index, UUID, memory (GiB, if known), name.

    ``nvidia-smi`` numbers GPUs in PCI bus order, whatever ``CUDA_VISIBLE_DEVICES``
    says. Read once per process; empty when there is no ``nvidia-smi`` or it fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    rows: list[tuple[str, str, float | None, str]] = []
    for line in result.stdout.splitlines():
        fields = [f.strip() for f in line.split(",", 3)]
        if len(fields) != 4 or not fields[0]:
            continue
        try:
            memory = float(fields[2]) / 1024 if fields[2] else None
        except ValueError:
            memory = None
        rows.append((fields[0], fields[1], memory, fields[3]))
    return tuple(rows)


def _gpu_row(gpu: str) -> tuple[str, str, float | None, str] | None:
    """The ``nvidia-smi`` row of a GPU identifier (an index or a UUID), if it lists it."""
    for row in _nvidia_smi_gpus():
        if gpu == row[0] or _same_gpu(gpu, row[1]):
            return row
    return None


def _same_gpu(a: str, b: str) -> bool:
    """Whether two GPU identifiers name one device: equal, or a UUID and its prefix."""
    if a == b:
        return True
    uuid = ("GPU-", "MIG-")
    return (
        a.startswith(uuid)
        and b.startswith(uuid)
        and (a.startswith(b) or b.startswith(a))
    )


def _ordinal(gpu: str, visible: list[str] | None) -> int | None:
    """The CUDA ordinal of one GPU identifier in this process, if it can be found."""
    if visible is not None:
        return next((i for i, v in enumerate(visible) if _same_gpu(gpu, v)), None)
    if gpu.isdigit():
        return int(gpu)
    rows = _nvidia_smi_gpus()
    return next((int(row[0]) for row in rows if _same_gpu(gpu, row[1])), None)


def _program_names(spec: ProgramSpec | FileInput) -> set[str]:
    """The programs an input or spec names: its own and its subprograms', recursively."""
    if not isinstance(spec, ProgramSpec):
        return {FILES}
    names: set[str] = set()
    pending = [spec]
    while pending:
        spec = pending.pop()
        names.add(spec.program)
        pending.extend(spec.subprograms)
    return names


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
    fields. In a list, a calculation's share of its runner is the larger of its share
    of the threads and its share of the GPUs (with k calculations per GPU, each counts
    1/k of a GPU). The resource keywords ``threads``, ``memory_gib`` and
    ``device_memory_gib`` are what a calculation asks for, and the runner grants
    them, at most what it has (for a subprogram, what its parent was granted).

    A program uses the smaller of its own request (a resource keyword) and the grant,
    or the grant when it requested nothing; a request capped by the grant is a
    ``resource_capped`` warning, not an error.

    Attributes:
        threads: CPU threads granted: what the input asks for (its ``threads``
            keyword, else ``plan.resources.threads``; all of them when that is
            None), at most the runner's. None: not
            granted (a direct call).
        gpus: Identifiers of the GPUs granted (as for ``CUDA_VISIBLE_DEVICES``:
            indices or UUIDs); `device_ordinals` gives this process's CUDA ordinals.
        memory_gib: Host memory granted (GiB). A calculation run alone (one input, a
            session) gets all of the runner's memory, or as a subprogram all of its
            parent's grant (a session opened with ``threads=`` gets the part of it
            those threads are of the parent's). In a list, each calculation, and each
            batch, gets its share of the list's memory. The input's ``memory_gib``
            keyword replaces it, at most the runner's memory, or for a subprogram its
            parent's grant. None: not known.
        device_memory_gib: Memory granted on each GPU (GiB): 1/k of the GPU when k
            calculations may share it (``per_gpu``), at most a subprogram's parent's
            grant. The input's ``device_memory_gib`` keyword replaces it, at most the
            GPU's memory or that grant. A batch gets the share. None: not known or no
            GPU granted.
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
        "_calling",
        "_named",
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
        # The input(s) whose subprogram tree names the programs ctx may run, set by a
        # runner (several for a batch); None: no check. Their names, on first use.
        self._calling: Sequence[ProgramInput | FileInput] | None = None
        self._named: frozenset[str] | None = None

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
        kind ``timeout``; one that loops (SCF cycles, optimizer steps) checks it. The
        runner kills a program in a worker process when it runs out, but cannot stop
        one in its own process: there the limit is the program's to honour, and the
        runner only refuses to start its subprograms afterwards (``ctx.compute``,
        ``ctx.session`` and the session's requests fail with kind ``timeout``).
        Subprograms get what is left.
        """
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    def device_ordinals(self) -> list[int]:
        """The granted GPUs as CUDA device ordinals in this process, in grant order.

        A program selects its devices with these (``torch.device("cuda", i)``,
        ``cudaSetDevice(i)``), so every program maps a grant alike:

        - When this process's ``CUDA_VISIBLE_DEVICES`` is set, a GPU's ordinal is its
          position there. A worker process is started with ``CUDA_VISIBLE_DEVICES``
          set to its grant, so its GPUs are 0, 1, ...; in the runner's process GPU
          ``"2"`` of ``CUDA_VISIBLE_DEVICES=1,2`` is ordinal 1. UUIDs (``GPU-...``,
          ``MIG-...``) match by value or unique prefix.
        - When it is unset, an index is its own ordinal, and a UUID is the index
          ``nvidia-smi`` gives it. ``nvidia-smi`` numbers GPUs in PCI bus order, so a
          runner that lists them with it sets ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` (unless
          set), for CUDA to number them alike. A process that starts CUDA before it
          makes its runner sets that itself.

        Returns:
            One ordinal per granted GPU; empty when none is granted.

        Raises:
            ProgramError: Kind ``environment``: a granted GPU is not one this process
                can see.
        """
        if not self.gpus:
            return []
        visible = _cuda_visible_devices()
        ordinals: list[int] = []
        for gpu in self.gpus:
            ordinal = _ordinal(gpu, visible)
            if ordinal is None:
                where = (
                    f"in this process's CUDA_VISIBLE_DEVICES ({','.join(visible)!r})"
                    if visible is not None
                    else "among the GPUs nvidia-smi lists"
                )
                raise ProgramError.of(
                    "environment", f"The granted GPU {gpu!r} is not {where}."
                )
            ordinals.append(ordinal)
        return ordinals

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

    def _check_named(self, inputs: Iterable[ProgramInput | FileInput]) -> None:
        """Refuse a subprogram that the calling input's subprogram tree does not name.

        A run's identities are those of the programs its input names, so a program it
        ran without naming would be missing from them, and a store would reuse the
        run after that program changed. Every program a subprogram's input names
        (its own and its subprograms') must be named.

        Raises:
            ProgramError: Kind ``input``, code ``unnamed_subprogram``.
        """
        if self._calling is None:
            return
        if self._named is None:
            self._named = frozenset().union(*map(_program_names, self._calling))
        for inp in inputs:
            missing = _program_names(inp) - self._named
            if missing:
                name = ", ".join(sorted(missing))
                raise ProgramError.of(
                    "input",
                    f"This calculation runs {name}, which its input does not name. "
                    f"Name {name} among the input's subprograms (it names "
                    f"{sorted(self._named)}): a run's identities are those of the "
                    "programs its input names, so a store could reuse it after "
                    f"{name} changed.",
                    code="unnamed_subprogram",
                )

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

        Every program a subprogram's input names must be named by this calculation's
        input, among its subprograms (recursively), so that the run's identities cover
        every program it used; another is refused, whatever ``raise_exc``.

        Args:
            inputs: One input, or a list of independent inputs.
            raise_exc: Raise for failures; False returns failed outputs.

        Returns:
            The output, or the outputs in input order.

        Raises:
            ProgramError: A subprogram failed (with ``raise_exc``); or kind ``input``,
                code ``unnamed_subprogram``: the input does not name the program.
            ProgramErrors: Some subprograms of a list failed (with ``raise_exc``).
        """
        if self._calling is not None:
            one = isinstance(inputs, ProgramInput | FileInput)
            inputs = [inputs] if one else list(inputs)
            self._check_named(inputs)
            if one:
                inputs = inputs[0]
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

        Raises:
            ProgramError: The program refuses the input; or kind ``input``, code
                ``unnamed_subprogram``: this calculation's input does not name it
                (see `compute`).
        """
        if isinstance(spec, ProgramInput):
            inp = spec
        else:
            if structure is None:
                raise TypeError("A session from a ProgramSpec needs its structure.")
            inp = ProgramInput.from_spec(spec, structure)
        self._check_named((inp,))
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

        Under a runner, ``ctx`` is the batch's grant: the sum of its inputs' threads
        (at most the runner's) and the most GPUs any of them asks for. The program
        divides it among the inputs (one thread each to N one-thread searches run
        side by side), and records in each output's ``execution.threads`` what that
        input used; left unset, the runner records the input's own part of the
        grant. Likewise ``execution.wall_time``: each input's own where the program
        can tell (a lockstep batch knows when each task finished), else left unset,
        and the runner records the whole batch's. The batch's wall time is always in
        ``timings["batch"]`` (s).

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
    request's ``accuracy`` becomes the ``accuracy`` keyword of the input it computes,
    replacing the session input's own; the program records it as it records the
    keyword (a program with no convergence settings, as met). A ``hessian`` result
    without a gradient is completed by a gradient calculation.

    Args:
        compute: Computes one input, or a list with ``raise_exc=False``.
        inp: The program, model, keywords and structure; its geometry and calctype are
            the defaults of a request.
        calctypes: The calctypes the session may run (see `session_calctypes`; by
            default energy, gradient and Hessian, which the program may still refuse).
        program_version: The program's version (``program.version()``), the
            session's from its opening.
    """

    def __init__(
        self,
        compute: Callable[..., Any],
        inp: ProgramInput,
        calctypes: Iterable[CalcType | str] | None = None,
        *,
        program_version: str,
    ):
        """Create the session."""
        if inp.structure is None:
            raise TypeError("A session's input needs a structure.")
        self._compute = compute
        self._inp = inp
        self._structure: Structure = inp.structure
        self.calctypes = session_calctypes(calctypes)
        self.program = inp.program
        self.program_version = program_version
        self._closed = False

    @property
    def input_data(self) -> ProgramInput:
        """The input the session was opened with."""
        return self._inp

    def _input(
        self,
        positions: np.ndarray | None,
        calctype: CalcType | str | None,
        accuracy: Accuracy | float | None = None,
    ) -> ProgramInput:
        """The session's input at a request's positions, calctype and accuracy."""
        calctype = CalcType(calctype) if calctype is not None else self._inp.calctype
        check_session_calctype(calctype, self.calctypes)
        # The input is already valid; copies skip validating it again on every step.
        update: dict[str, Any] = {"calctype": calctype}
        target = Accuracy.of(accuracy, calctype)
        if target is not None:  # plain JSON, as qcdata stores keywords
            plain = BaseModel.model_dump(target, mode="json", exclude_none=True)
            update["keywords"] = {**self._inp.keywords, "accuracy": plain}
        if positions is not None:
            geometry = np.asarray(positions, dtype=float).reshape(-1, 3)
            if geometry.shape != np.asarray(self._structure.geometry).shape:
                raise ProgramError.of(
                    "input",
                    f"positions have {geometry.shape[0]} atoms; the session's "
                    f"structure has {len(self._structure.symbols)}",
                )
            update["structure"] = self._structure.model_copy(
                update={"geometry": geometry}
            )
        return self._inp.model_copy(update=update)

    def _as_frame(self, out: ProgramOutput[Any, Any]) -> Frame:
        """An output as a frame, with ``warm_start`` false in its record."""
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
            accuracy: The input's ``accuracy`` keyword for this request (a number
                targets the calctype's highest derivative); None: the input's own.
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
        return self._one(self._input(positions, calctype, accuracy))

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
                inputs.append(self._input(r.positions, r.calctype, r.accuracy))
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

#: The JSON Schema of the shared keyword ``accuracy`` in a program with no convergence
#: settings (`FunctionProgram`, the toy), which accepts it and records it as met.
_ACCURACY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "minProperties": 1,
    "properties": {
        "energy": {"type": "number", "exclusiveMinimum": 0, "x-unit": "Eh"},
        "gradient": {"type": "number", "exclusiveMinimum": 0, "x-unit": "Eh/bohr"},
        "hessian": {"type": "number", "exclusiveMinimum": 0, "x-unit": "Eh/bohr^2"},
    },
    "description": "Accuracy targets, recorded as met: nothing here converges.",
    "x-tier": "common",
    "x-concern": "convergence",
}

#: The JSON Schema of the shared keyword ``threads`` of a program the runner grants
#: threads to and that has nothing of its own to say about them (`FunctionProgram`).
_THREADS_SCHEMA: dict[str, Any] = {
    "type": "integer",
    "minimum": 1,
    "default": 1,
    "description": "CPU threads the calculation asks for; the runner grants them, at "
    "most its own.",
    "x-tier": "common",
    "x-concern": "resource",
}

#: The JSON Schema of the shared record key ``accuracy`` of a target met with no
#: thresholds.
_ACCURACY_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "The accuracy target, met with no thresholds: an estimate of 0.0 "
    "per target.",
    "properties": {
        "target": {"type": "object"},
        "thresholds": {"type": "object", "maxProperties": 0},
        "estimate": {"type": "object"},
    },
    "required": ["target", "thresholds", "estimate"],
}


def _accuracy_met(target: Accuracy) -> dict[str, Any]:
    """The record of an accuracy target met by a program with no convergence settings.

    qcdata's conventions shape it: the targets given, no thresholds, and an error
    estimate of exactly zero per target, so a reader checks every program alike (each
    estimate at or below its target)::

        {"target": {"gradient": 1e-06}, "thresholds": {}, "estimate": {"gradient": 0.0}}

    Args:
        target: The calculation's target (the keyword's or a session request's).

    Returns:
        The value of the record key ``accuracy``.
    """
    given = BaseModel.model_dump(target, mode="json", exclude_none=True)
    return {
        "target": given,
        "thresholds": {},
        "estimate": dict.fromkeys(given, 0.0),
    }


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
    complete program. Its keywords are the shared ``threads`` (what it asks the runner
    for; 1 by default), the shared ``accuracy``, which it records as met, since the
    functions converge nothing (``{"target", "thresholds": {}, "estimate"}`` with an
    estimate of 0.0 per target, as qcdata's conventions give), and the settings it
    declares; any other keyword is an ``unknown_key`` input error. It runs in process
    and is thread-safe::

        def morse(x):  # positions (bohr, n x 3) -> energy (Eh), gradient (Eh/bohr)
            ...
            return energy, gradient

        qccompute.register_program(FunctionProgram("morse", morse))
        out = qccompute.compute(ProgramInput(program="morse", calctype="gradient",
                                             structure=h2))

    With ``takes_input=True`` the functions also get the input, so one program serves
    any molecule and setting: ``energy_gradient(positions, inp)`` reads the symbols,
    charge, multiplicity, model and keywords (an engine over a library's calculator).
    Without it they get the positions alone and the program ignores the model.

    An exception from a function is a failed calculation of kind ``internal``, with
    the function's traceback in the failed output; a `qcdata.ProgramError` it raises
    (kind ``input`` for a setting's value it refuses) is the calculation's failure.

    Args:
        name: The program's name, as in ``ProgramInput.program``.
        energy_gradient: Positions (bohr, n_atoms x 3), and with ``takes_input`` the
            input, to the energy (Eh) and the gradient (Eh/bohr, n_atoms x 3).
        hessian: The same arguments to the Hessian (Eh/bohr^2, 3 n_atoms x 3
            n_atoms); without it the program computes energies and gradients only.
        version: The program's version and numerics identity, for provenance and the
            store. None: a digest of the functions (``fn-`` and 16 hex digits), taken
            on each use, of their code, constants, defaults, closures and the module
            globals they read, recursing into the Python functions they call. Editing
            a function or a value it reads (an energy scale) changes it, so a store
            never serves results of the old function. An object the functions use
            that is not plain data (an ML model) counts by its type and attributes;
            give ``version`` when that is not enough (a model's checksum).
        takes_input: Call the functions with the input after the positions.
        settings: The JSON Schema of the keywords the functions read (an object
            schema; its ``properties`` name them, with ``x-unit``, ``x-tier`` and
            ``x-concern`` as for any program). Needs ``takes_input``: a keyword the
            functions never see would change nothing. The program refuses keys it
            does not name; checking their values is the functions' (or
            ``qccompute.testing``'s, against the schema).

    Raises:
        ValueError: ``settings`` without ``takes_input``, or naming ``accuracy`` or
            ``threads``, which every FunctionProgram accepts.
    """

    _RECORD_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {"accuracy": _ACCURACY_RECORD_SCHEMA},
    }

    def __init__(
        self,
        name: str,
        energy_gradient: Callable[..., tuple[float, Any]],
        hessian: Callable[..., Any] | None = None,
        *,
        version: str | None = None,
        takes_input: bool = False,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        """Make a program of the functions."""
        declared = dict((settings or {}).get("properties") or {})
        if declared and not takes_input:
            raise ValueError(
                f"{name}: settings need takes_input=True; the functions would never "
                "see a keyword, so it could not affect the calculation."
            )
        shared = sorted({"accuracy", "threads"} & set(declared))
        if shared:
            raise ValueError(
                f"{name}: {shared} are shared keywords every FunctionProgram accepts; "
                "do not declare them."
            )
        self.name = name
        self._energy_gradient = energy_gradient
        self._hessian = hessian
        self._version = version
        self._takes_input = takes_input
        self._schema: dict[str, Any] = {
            **(settings or {}),
            "type": "object",
            "properties": {
                **declared,
                "accuracy": _ACCURACY_SCHEMA,
                "threads": _THREADS_SCHEMA,
            },
            "additionalProperties": False,
        }
        self._keys = frozenset(self._schema["properties"])
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
        """The settings declared, and the shared ``accuracy`` and ``threads``."""
        return self._schema

    def record_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The shared ``accuracy``, when a target is given."""
        return self._RECORD_SCHEMA

    def plan(self, inp: ProgramInput) -> Plan:
        """Check the input: a structure, a supported calctype, keywords it knows.

        The resolved settings are the keywords given. An ``accuracy`` target is met:
        the plan's record holds the record key ``accuracy``, which the run records.

        Raises:
            ProgramError: Kind ``input`` (``unknown_key``; ``threads`` that is not a
                positive integer; a malformed ``accuracy``, ``no_effect`` for one with
                no target; no structure) or ``unsupported``.
        """
        if inp.calctype not in self._caps.calctypes:
            raise ProgramError.of(
                "unsupported",
                f"{self.name} computes {sorted(c.value for c in self._caps.calctypes)},"
                f" not {inp.calctype.value}.",
            )
        if inp.structure is None:
            raise ProgramError.of("input", f"{self.name} needs a structure.")
        if not inp.keywords:
            return _EMPTY_PLAN
        unknown = sorted(set(inp.keywords) - self._keys)
        if unknown:
            raise ProgramError.of(
                "input",
                f"{self.name} has the settings {sorted(self._keys)}; unknown {unknown}.",
                code="unknown_key",
            )
        threads = inp.keywords.get("threads", 1)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ProgramError.of(
                "input", f"threads is a positive integer, not {threads!r}."
            )
        settings = dict(inp.keywords)
        target = Accuracy.from_keywords(inp.keywords)
        if target is None:
            return Plan(resolved=Resolved(settings=settings))
        met = _accuracy_met(target)
        settings["accuracy"] = met["target"]
        return Plan(resolved=Resolved(settings=settings), record={"accuracy": met})

    def compute(
        self, inp: ProgramInput, ctx: RunContext | None = None
    ) -> ProgramOutput[ProgramInput, Data]:
        """Evaluate the functions at the input's structure.

        Raises:
            ProgramError: The input is refused, or a function raised (kind
                ``internal``; the failed output holds its traceback).
        """
        start = time.perf_counter()
        # A runner hands on the plan it made.
        plan = ctx.plan if ctx is not None and ctx.plan is not None else self.plan(inp)
        assert inp.structure is not None
        x = np.asarray(inp.structure.geometry, dtype=float).reshape(-1, 3)
        provenance = Provenance(program=self.name, program_version=self.version())
        # What the runner granted; called directly, what the input asks for.
        threads = (ctx.threads if ctx is not None else None) or inp.keywords.get(
            "threads", 1
        )
        given = (inp,) if self._takes_input else ()
        try:
            energy, gradient = self._energy_gradient(x.copy(), *given)
            values: dict[str, Any] = {"energy": float(energy)}
            if inp.calctype != CalcType.energy:
                values["gradient"] = np.asarray(gradient, dtype=float).reshape(-1, 3)
            if inp.calctype == CalcType.hessian:
                assert self._hessian is not None
                hessian = self._hessian(x.copy(), *given)
                values["hessian"] = np.asarray(hessian, dtype=float)
            results = SinglePointData(provenance=provenance, **values)
        except ProgramError:  # the function's own refusal or failure
            raise
        except Exception as e:  # noqa: BLE001 - the function's failure, with its trace
            error = ErrorInfo(
                kind=ErrorKind.internal, message=f"{type(e).__name__}: {e}"
            )
            out = ProgramOutput.failure(
                inp,
                error,
                traceback="".join(traceback.format_exception(e)),
                execution=ExecutionInfo(threads=threads),
            )
            raise ProgramError(error, output=out) from e
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=results,
            resolved=plan.resolved,
            record=plan.record,
            execution=ExecutionInfo(
                threads=threads, timings={"total": time.perf_counter() - start}
            ),
        )
