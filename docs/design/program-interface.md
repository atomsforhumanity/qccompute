# The AFH program interface: one way to run programs, one shape for their data

Status: **draft proposal** (2026-11-11), from the qc* stack, agreed in substance by the
linus, axtb and qcopt sessions (their amendments are folded in). The session interface is
v4 of linus's proposal, agreed by linus, axtb and qc* on 2026-11-13. The data parts live in
qcdata (branch `data-only`); the protocol in qccompute (`qccompute.program`, branch
`execution-protocol`). The qccodec adapter that wraps a Codec as a Program and the worker
(`qccompute.serve`, drafted by linus) are still to do. Items marked **[Colton]** need a
decision.

## Why

AFH now writes its own programs (linus, axtb, qcopt, more to come) faster than it wraps
other people's. What we need from qc* has changed accordingly:

1. **Standard data out.** Apps, visualizations, analyses and workflows should read one set of
   typed fields whatever program produced them. Today each of our programs puts its most
   useful output (energy components, resolved settings, vibrational analysis,
   thermochemistry, orbitals, warnings, error kinds, timings) in `extras`, under its own names.
2. **One way in.** A calculation is described the same way for every program, with each
   program's settings validated against a schema the program publishes.
3. **One way to run.** A program written at AFH implements one small protocol and gets
   qccompute's parallelism, batching, GPU scheduling, child calculations and sessions
   without the codec ceremony that exists for wrapping external binaries.

## The layers

```
apps · visualization · analysis · workflows        read qcdata, nothing else
──────────────────────────────────────────────
qcdata     the standard: inputs, outputs, typed result blocks, JSON Schema,
           the Program/Session protocols (typing only), conventions, conformance tests
──────────────────────────────────────────────
qccompute  the runner: discovery, schema validation, resources and scheduling
           (threads, GPUs, memory), lists, batches, sessions, children, logs, errors
──────────────────────────────────────────────
programs   native AFH programs (linus, axtb, qcopt, ...) implement qcdata.Program;
           they depend on qcdata only
qccodec    adapters that make external programs (ORCA, Psi4, TeraChem, CREST, xtb,
           geomeTRIC, Sella, ML potentials) look like a qcdata.Program
```

The role of qccodec shrinks to what it was built for: external programs. Native programs
stop importing it (today linus, axtb and qcopt each ship a qccodec codec whose encode
packs a `PythonInput` that calls themselves and whose decode unpacks it again).

## 1. Inputs

`ProgramInput` stays: `program`, `calctype`, `structure`, `model`, `keywords`, `subprograms`,
`files`.

- **`keywords` is the program's native settings tree**, unchanged: the same keys and nesting
  as its JSON request file and its Python keyword arguments (the D39 rules agreed by linus,
  axtb and qcopt: `group.key`, at most two levels, one canonical spelling, `"auto"` means
  the program decides). qcdata already round-trips nested keywords; nothing to flatten.
- **Validation from the program's schema.** Every program publishes
  `settings_schema(calctype) -> dict` (JSON Schema, with `x-unit`, `x-tier`, `x-concern`,
  `x-applies-to`). qccompute validates keywords before scheduling, so a typo fails at once
  for a list of 1000 inputs, and tools get completion and docs from the schema.
- **`model` and `structure` map onto native keys**: `model.method`/`model.basis` to
  `method`/`basis`; `structure` gives positions (bohr), numbers, charge and multiplicity.
  The native tree may not repeat them.
- **Resource keys** (`threads`, `memory_gib`, `backend`, `device_memory_gib`) stay keywords
  (as agreed), and their schema entries carry `x-concern: "resource"`; qccompute reads them
  for scheduling. Memory and backend can change numbers at round-off (linus picks its DF
  storage from the memory budget; capped GPU runs may take FP32 paths), so **result
  identity is computed from `resolved.settings`**, where every such choice appears resolved,
  never from the raw resource keys.
- **One shared vocabulary** for settings that mean the same in several programs lives in
  qcdata's docs (`docs/conventions.md`), which is the single place for names and units
  (axtb's request 6). The starting list is what the three programs already agreed:
  `threads`, `memory_gib`, `backend`; `scf.{max_iter, e_tol, g_tol, guess}`;
  `solvent.{name, model, eps}`; `freq.{temperature, pressure, scale, imaginary_cutoff}`;
  `output.{orbitals, density, molden, bond_orders}`; `convergence.{max_iter, g_tol, ...}`
  for optimizers. `accuracy` is reserved for the cross-program absolute target. **[Colton]**
  signs off the names.
- **Calculation types.** `hessian` means a Hessian *and*, when the program can do it, the
  harmonic analysis and thermochemistry at the same geometry (linus and axtb already do;
  it costs nothing next to the Hessian; `freq.*` keywords tune or switch it off). No separate
  `frequencies` calctype: linus maps qcdata `hessian` to its own `frequencies`. Analysis of
  a *given* Hessian is a function, not a program run:
  `qcinf.vibrational_analysis(structure, hessian, ...)`, a reference implementation any
  program's Hessian (ORCA, an ML potential) can go through. **[Colton]**

## 2. Outputs

### 2.1 Slots every output has (`ProgramOutput`; `warnings` and `error` also on each `Frame`)

| field | type | meaning |
|---|---|---|
| `resolved` | `Resolved \| None` | `settings`: the complete settings tree with every `auto` replaced, itself a valid `keywords` that reproduces the run; `reasons`: path → why |
| `warnings` | `list[Warning]` | `code` (stable, snake_case), `message`, optional `path` (the settings key concerned) |
| `error` | `ErrorInfo \| None` | on failure: `kind` (below), `message`, optional `code` (stable snake_case within a kind, e.g. `max_iter`, `energy_rose`, `not_first_order_saddle`: callers branch on it instead of parsing text), optional `program` (the child that failed), optional `cause: ErrorInfo` (the child's own error, for kind `program`) |
| `record` | `dict` | (also on each `Frame`) what happened during the run (fallbacks, escalations, residuals reached, per-phase decisions); JSON, shaped by the program's published result schema; diagnostic, not stable across versions |
| `execution.timings` | `dict[str, float]` | seconds per phase (keys are the program's; `total` for the whole call, drivers also `children` and their own time), on `ExecutionInfo` beside `wall_time` |
| `results.provenance` | `Provenance` | gains `program_commit` (and keeps `program`, `program_version`) |

`ErrorKind` (`StrEnum`): `input`, `unsupported`, `convergence`, `numerical`, `resource`,
`backend`, `program` (a child calculation failed), `internal` (the agreed vocabulary), plus
two qccompute sets itself: `timeout` and `environment` (program not installed, licence,
the process died). Partial results stay attached to failed outputs, as today.

`extras` remains for what has no standard home yet. **Rule:** a concept that two programs
report goes into a typed field, not `extras`.

### 2.2 Single points (`SinglePointData`, new optional fields)

| field | type | meaning |
|---|---|---|
| `energy_components` | `dict[str, float]` | Eh; a true partition: they add up to `energy`; what a component contains is the program's, and a hierarchy of overlapping terms goes in `record`. Standard keys: `nuclear_repulsion`, `one_electron`, `coulomb`, `exchange`, `xc`, `dispersion`, `vv10`, `solvation_electrostatic`, `solvation_cds`, `repulsion`, `electrostatic`, `halogen_bond`, `hydrogen_bond`, ... (listed in conventions.md; programs may add their own keys) |
| `scf_iterations` | `int` | |
| `scf_quadrupole_moment` | 3 × 3 | e·bohr², about the coordinate origin, with `quadrupole_kind`: `traceless` (Buckingham ½Σq(3rᵢrⱼ − r²δᵢⱼ)) or `second_moment` (Σq rᵢrⱼ); programs convert other forms to one of these |
| `polarizability` | 3 × 3 | e² bohr² / Eh |
| `atomic_charges_kind` | `str` | `mulliken`, `lowdin`, `eeq`, `eeqbc`, `cm5`, `hirshfeld`, ... (charges are not comparable without it) |
| `bond_orders` | n × n | with `bond_orders_kind`: `mayer` (Σ(PS)μν(PS)νμ, what xtb prints as "Wiberg") or `wiberg` (orthogonalized basis) |
| `vibrational_analysis` | `VibrationalAnalysis` | linus's proposal (masses, projection, all modes with `external` flags, imaginary frequencies negative and stored without a cutoff, reduced masses, force constants, IR intensities, dipole derivatives, rotor type, moments, rotational constants, point group, symmetry number); `n_imaginary(cutoff)` is a method |
| `thermochemistry` | `list[Thermochemistry]` | linus's proposal (one per temperature/pressure: ZPE, per-term E/H/S/Cv/Cp/G, corrections, entropy/enthalpy model, scale factor, excluded imaginary modes) |
| `wavefunction.orbitals` | `Orbitals` | linus's proposal (`BasisSet` of `Shell`s with a named `AOConvention`, per-spin coefficients, energies, occupations); filled on request (`output.orbitals`) |

The old `freqs_wavenumber`, `normal_modes_cartesian` and `gibbs_free_energy` are
**removed** (Colton, 2026-11-11: pre-v1, a final unified API rather than compatibility
for the few pre-release users). Files saved before 0.24 still load when those fields are
empty. HOMO-LUMO gaps are a property computed from `wavefunction`, not a stored field.

### 2.3 Trajectories (phase 4)

`OptimizationData`/`PathData` keep their reading interface (frames), but serialize
columnar when frames share atoms, charge and multiplicity: symbols once, positions,
energies and gradients as arrays, provenance once. A 400-frame IRC drops from a full
`Structure` and `Provenance` per frame to a few arrays. Also: a standard slot for a
frequency analysis made with a different method than the frames (qcopt's final check).

### 2.4 Published schema

`qcdata.output_schema(calctype)` gives the JSON Schema of the outputs; each program's
`settings_schema(calctype)` gives the shape of `keywords` and `resolved.settings`, and its
`record_schema(calctype)` the shape of `record`. Viewers and
apps generate from these, as the programs generate their docs.

## 3. Running: the Program protocol

qcdata holds only data (Colton, 2026-11-13: "strictly about quantum chemistry data input
and output, not about program execution"). The protocol below lives in qccompute
(`qccompute.program`). **Programs never import qccompute:** the protocols are structural
(`typing.Protocol`), and what a program returns for `capabilities` and `plan` may be a
plain mapping, which qccompute validates (`capabilities_of`, `plan_of`). It is the same
document the program's command line prints for `plan` and its worker sends in its hello
line. What programs and drivers share is in qcdata: `ProgramInput`, `ProgramOutput`,
`Frame`, `Request`, `Accuracy` and `ProgramError` (the exception form of `ErrorInfo`;
programs raise it or a subclass, drivers catch this one class).

A program registers an instance (not its class) under the entry point group
`qccompute.programs`:

```toml
[project.entry-points."qccompute.programs"]
linus = "linus.qc:program"
```

and the object implements:

```python
class Program(Protocol):
    name: str
    def version(self) -> str: ...
    def capabilities(self) -> Capabilities | Mapping: ...
        # calctypes; methods {method: calctypes}; sessions, batches, in_process,
        # thread_safe, gpu, driver
    def settings_schema(self, calctype: CalcType) -> dict: ...
    def record_schema(self, calctype: CalcType) -> dict: ...
    def plan(self, inp: ProgramInput) -> Plan | Mapping: ...
        # dry run: validates (ProgramError kind input, with did-you-mean), resolves every
        # auto. Plan(resolved: qcdata Resolved, resources: Resources(threads, memory_gib,
        # gpus, device_memory_gib, exclusive), warnings, record: the program's own detail,
        # e.g. linus's placement, arithmetic and memory plan). Estimates per calctype come
        # from planning the input at each calctype.
    def compute(self, inp: ProgramInput, ctx: RunContext) -> ProgramOutput: ...
        # a failed calculation returns success=False with error and partial results;
        # it raises only on bugs

    # optional
    def open_session(self, inp: ProgramInput, ctx: RunContext, calctypes=None) -> Session: ...
        # usually `return Session(inp, calctypes=calctypes, ctx=ctx)`
    def compute_batch(self, inputs: list[ProgramInput], ctx: RunContext) -> list[ProgramOutput]: ...
```

`RunContext` is what qccompute fills when it runs a program; a program reads its fields
and never constructs one:

- `threads`, `gpus`, `memory_gib`: the grant. A program uses the smaller of its keyword
  request and the grant, or the grant when nothing was requested; a capped request is a
  `resource_capped` warning, not an error. Without a context (direct use) the program
  behaves as under a runner that granted nothing.
- `scratch`: a directory, made on first use;
- `plan`: the runner's plan of this input, so the program need not plan twice;
- `compute`: the child callback (one input or a list, `previous=`);
- `session(inp)` or `session(spec, structure)`: open a child session with a share of the
  grant (qcopt with 16 threads and 8 tasks gives each session 2);
- `progress(...)`: optional streaming updates.

qccompute finds programs in `qccompute.programs` first and then in `qccodec.codecs`. The
native path skips encode/decode: qccompute calls `program.compute(inp, ctx)` in its own
process when `capabilities().in_process` is true.

### Sessions (agreed by linus, axtb and qc*, v4, 2026-11-13)

"The same program, the same molecule, many geometries, warm-started" is what optimizers,
IRC, MD, NEB and finite differences all need. One class is both the native API and the
protocol object: `linus.Session` and `axtb.Session` implement `Session` directly, and
`open_session` constructs them, so qccompute needs no adapter and a researcher uses the
object qccompute does:

```python
import qcdata, linus

inp = qcdata.ProgramInput(structure=water, calctype="gradient",
                          model={"method": "b3lyp-d3bj", "basis": "def2-svp"})
out = linus.compute(inp)                    # one calculation -> ProgramOutput; never raises

with linus.Session(inp) as s:               # calctypes: everything the method supports
    f = s.compute()                         # the input's geometry and calctype -> Frame
    f = s.compute(x2)                       # warm start from this lineage's last success
    h = s.compute(x2, calctype="hessian")   # energy, gradient, Hessian and its analysis
    g = s.compute(y, lineage=("irc", "forward"), seed=["ts"])   # any JSON value
    fd = s.compute_many([qcdata.Request(positions=p, commit=False) for p in displaced])
    s.reset("irc")                          # one lineage; s.reset() all; s.reset(None) default
    s.plan()
```

```python
class Session(Protocol):
    program: str
    program_version: str
    def compute(self, positions=None, calctype=None, *, accuracy=None, lineage=None,
                outputs=(), seed=(), commit=True) -> Frame: ...
    def compute_many(self, requests: list[qcdata.Request]) -> list[Frame]: ...
    def reset(self, lineage=...) -> None: ...   # omitted: all; None: the default lineage
    def plan(self) -> Plan: ...
    def close(self) -> None: ...                # also a context manager
```

- **Opening.** The `ProgramInput` pins atoms, charge, multiplicity, model and keywords;
  its geometry and calctype are the defaults of a request. `calctypes` defaults to
  everything the method supports; declaring narrows it. A setting that applies to no
  allowed calctype is an error at open (`not_applicable`); one that applies to an allowed
  calctype never requested is contingent (D41), and the record shows it never fired. A
  method that cannot represent the charge or multiplicity raises `not_modeled` at open.
  Declaring changes validation and planning, not memory: programs allocate per request.
- **Results are `Frame`s** (structure, `SinglePointData`, the request's own `warnings`,
  `record` and `execution`), with the shared `warm_start` (true, false, or null when the
  method keeps no lineage state) at the top of the record, beside the program's own
  detail (linus: `scf.guess {source, lineage, distance, orthonormalized}`). Results are
  read as `f.results.energy`; no shortcuts.
- **Lineages** are any JSON value, keyed by their canonical JSON text
  (`qcdata.lineage_key`), so a tuple in process and a list on the wire are one lineage.
  None is the default lineage. Per lineage a program keeps what speeds or steadies the
  next step (linus: last orbitals and probe state; axtb: charges and multipoles or
  densities); what is the method itself (GFN-FF's topology) is per session. A step
  farther than the program's warm-start distance starts fresh, with the reason recorded.
- **`seed`** lists lineages to start from, in order, when the request's own lineage has
  no history; with none, the request starts fresh (not an error). Both IRC branches
  start from the TS lineage this way. The record names the lineage used (`from_lineage`).
- **`commit`** (default true) makes a success the lineage's history; false stores
  nothing. A finite-difference fan-out runs its displacements in the center's lineage
  with `commit=False`: every one starts from the center, with no state kept per
  displacement (axtb: 2 GB of g-xTB densities for taxol otherwise). In a list, a request
  reads its lineage's history as it stands at its place: in [A, B (commit=False), C,
  D (commit=False)], B starts from A and D from C, and B may run alongside C if it reads
  the history from before C.
- **`compute_many`** runs requests of one lineage in list order; requests of different
  lineages may run concurrently within the grant. Results equal running them one at a
  time, each with the thread count it got (recorded in `execution.threads`: some
  programs' bits depend on it).
- **`accuracy`** is an `Accuracy(energy, gradient, hessian)` of targets (Eh, Eh/bohr and
  Eh/bohr² for the largest absolute component), or a bare number for the calctype's
  highest derivative: the convergence error the caller can tolerate. A target, not a
  guarantee: each program picks thresholds it expects to meet it and records target,
  thresholds and any estimate under the shared record key `accuracy`. Explicit
  convergence keywords are floors.
- **`outputs`** are per-request extras that do not change the surface; unsupported ones
  fail with kind `unsupported`. A name that fills a top-level `SinglePointData` field is
  that field's name (`scf_dipole_moment`, `polarizability`); others name what they add
  (`ir`: the vibrational analysis's intensities and dipole derivatives; `orbitals`,
  `density`: the wavefunction's).
- **Failures.** A failed request raises `ProgramError` with its failed `Frame` (partial
  results, warnings and record: an SCF's iteration history, its stability analyses), and
  is a failed frame in `compute_many` and over the wire. The session and its other
  lineages continue; the lineage's history is untouched (commit on success only: engines
  may drop or consume their state in a failed SCF). A program bug caught at the request
  boundary (a Rust panic) gives a failed frame and closes that session, in process and in
  a worker alike.
- **A `hessian` request returns energy and gradient too**, and behaves as a stateless
  `hessian` does (D42): linus runs the vibrational analysis by default.
- The session pins whatever the method derives from the structure it was opened with:
  GFN-FF's topology is part of the surface (re-deriving it moved the energy by up to
  4.4e-2 Eh along a 600-atom trajectory). A request's `accuracy` may make the program
  rebuild internals (GFN2's integral cutoffs, under 65 µs); the lineage state survives.
  The shape both studies arrived at: one engine object per session (model, parameters,
  topology, thread pool) shared by all lineages, plus a small state per lineage.
- **Native sessions matter for every engine.** The generic `ComputeSession` (one stateless
  calculation per request, `warm_start` false) is correct and adds about 13 µs per request
  in process. What it loses is the state a program carries between steps. linus measured
  it (real BFGS optimizations, B3LYP/def2-SVP and TZVP, 3-39 atoms and a radical): a fresh
  calculation per gradient cost 1.1-1.4× the session on 8 CPU cores (water 1.9×) and
  1.6-2.1× on a GPU (water 3.8×), from the warm start (about 15 → 10-11 SCF iterations),
  probe state carried across steps (+14-16% when lost) and, on a GPU, a live process
  (0.3-0.8 s of CUDA start-up per new process). axtb measured 1.1-1.75× on medium
  molecules, 5-75× on tiny ones, and 1.24-3.5× for GFN-FF. So runners keep programs in a
  live process and never start a process per calculation.
- **A warm start can find a different solution** than a fresh start (linus: a radical's
  UKS state 0.92 mEh lower; axtb: water at 3 Å), so every frame's record says which start
  produced it.
- `qccompute.session(inp)` opens one: the program's own when it has sessions, else a
  `ComputeSession` over `compute`. Inside a running program, `ctx.session(...)` opens child
  sessions under the parent's grant.
- Building a `Frame` with full validation costs 6-10 µs whatever the molecule's size
  (arrays pass through), less than `model_construct`; programs need no fast path.

### Workers: one live process per slot

A worker hosts a program in its own process, so a crash ends only the worker and the
caller survives. One generic loop in qccompute hosts any Session-shaped program in
process (`python -m qccompute.serve linus [--socket PATH]`); programs write no serve code.
`qccompute.Worker` is its client, whose `.Session(inp, ...)` has the Session API above, so
code moves between in-process and worker by one line.

- **Transport:** stdin/stdout, one JSON message per line, in order, one client per
  connection; a Unix socket on the same framing for several local clients. No TCP or HTTP
  inside programs. Workers are spawned, never forked after a GPU starts.
- **Hello:** the first line is `{"protocol": 1, "program", "version", "capabilities"
  (methods with their calctypes), "extras": {"numerics_epoch": 6}}`, so a store guard can
  check the epoch without importing the program.
- **Envelope:** every message carries an `id`; replies are `{"id", "ok": true, "result"}`
  or `{"id", "ok": false, "error": ErrorInfo}`. `ok: false` is only for the message itself
  (bad JSON, unknown op or session, a failed open); a failed calculation is a result (a
  failed Frame or ProgramOutput) from which the client raises `ProgramError`.
- **Ops** mirror the methods: `open {input, calctypes, ctx}`, `compute {session, positions,
  calctype, accuracy, lineage, outputs, seed, commit}`, `compute_many {session, requests}`,
  `reset {session, lineage}` (absent: all; null: the default), `plan {session | input}`,
  `close {session}`, and a one-off `compute` with `input` in place of `session`. End of
  input closes every session and exits 0.
- **ctx on the wire:** `{threads, memory_gib, scratch_dir}`; the GPU is fixed when the
  worker is spawned, so a `gpus` field must be empty or match.
- **Numbers:** shortest round-trip decimals, so positions and results cross bit for bit;
  no NaN or Inf on the wire (a non-finite result is a failed frame, kind `numerical`).
- **Hygiene:** stdout carries protocol lines only; the worker points fd 1 at stderr and
  writes the protocol to a dup of the original, because CUDA, BLAS and other libraries
  print. Logs go to stderr. Memory caps are cgroups, never `RLIMIT_AS` (CUDA reserves
  large address ranges).
- **Crashes:** a dead worker is `ErrorInfo(kind environment, code worker_died)` with the
  exit status and the tail of its stderr; the client can start a new one, and lineages
  on it are lost (the next step starts fresh, as its record shows). A worker that exits
  on purpose first answers the request in flight with its own error: a GPU fault that
  poisons the device context is kind `backend`, code `device_lost`, on which the serve loop
  exits without needing to know the program. In-process use has no such isolation, so qccompute prefers
  workers for GPU and long runs.

### Resources and GPUs (packing)

Workers are slots, not GPUs. Several processes per GPU raise throughput for small and
medium molecules (linus, against one process per GPU at N = 4/8/16: H200 2.0-4.0×, RTX PRO
6000 1.3-4.2×, bitwise identical results, 0.6-6.8 GB per process; MPS added only 2-11%).
On CPUs one 8-thread job per CCD gives the best throughput. So the scheduler starts K
workers per GPU, each with a grant (a share of device memory, one CCD's threads, a host
memory cap), and routes each session to one slot. K comes from the programs' plans. What
programs owe the scheduler: `plan()` estimates peak device memory, host memory and
useful threads; grants are honoured strictly (linus's `device_memory_gib` caps every
device allocation); out of memory is a typed error (`resource`, `device_memory` or
`host_memory`), falling back to the CPU only under `backend: auto`, with a
`backend_fallback` warning. The scheduler itself waits for Colton's review of the runner
design.

## 4. Conformance

`qccompute.testing.check_program(program, inputs)` (a test dependency of each program) runs a fixed set of small inputs and checks the
contract: the schema validates its own defaults, `plan` is reproducible from
`resolved.settings`, unknown keys fail with kind `input`, units are atomic, standard
fields are filled where the program claims the capability, errors carry kinds,
`energy_components` add up. Each program runs it in CI.

## 5. Phases (each a PR, in order)

1. **qcdata 0.24: output slots and typed blocks.** `Resolved`, `Warning`, `ErrorInfo`/
   `ErrorKind`, `record`, `execution.timings`, `Provenance.program_commit`; the new
   `SinglePointData` fields; `VibrationalAnalysis`, `Thermochemistry`; `Orbitals`, `BasisSet`,
   `Shell`, `AOConvention` (ported from `linus.models` with its tests); conventions.md.
   Stacked on #128.
2. **The protocol.** `qccompute.program` (Program, Session, RunContext, Plan, Resources,
   Capabilities) and `qccompute.testing`, with `Request`, `Accuracy` and `ProgramError`
   in qcdata; qccompute discovers `qccompute.programs`, validates by schema, runs native
   programs, opens sessions; qccodec wraps Codecs as Programs.
3. **Programs move.** linus, axtb and qcopt register `qccompute.programs`, fill the typed
   fields instead of `extras`, drop their qccodec codecs (kept one release as aliases);
   qcopt uses sessions and deletes its evaluators.
4. **Later.** Columnar trajectories, subprogram roles beyond calctype, GPU packing,
   session-aware batching, `qcinf.vibrational_analysis`.

## 6. Decisions (Colton, 2026-11-11, relayed by the linus session)

1. The protocol lives in qcdata. **Approved.**
2. One calctype `hessian` (with analysis); no `frequencies`. **Approved.**
3. No compatibility views: the old frequency fields are removed. **Decided.**
4. qccodec is for external programs only. **Approved.**
5. The shared names and glossary. **Approved.** Settings strictness (D41): settings that
   do not affect the calculation are errors; contingent settings are allowed and the
   record says whether they fired; sessions check against the calctypes they were opened
   for. See conventions.md rule 5.
