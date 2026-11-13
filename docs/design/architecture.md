# The AFH computing stack: architecture proposal

Status: **draft for discussion** (2026-11-11; moved from qcdata to qccompute on
2026-11-13, when qcdata became data only), from the qc* session, after a survey of
linus, axtb, qcopt, mepd (develop, 004f748) and the job runner. It builds on
[the program interface](program-interface.md), which the linus, axtb and qcopt sessions
agreed. Items marked **[Colton]** or **[Jan]** need their decision.

## 1. What we have, honestly

The qc* packages were designed for a world in which AFH wrapped other people's programs.
qccodec was the heart of that world: it held each external program's input format, output
parsers, defaults and quirks, and qccompute ran whatever qccodec encoded. That world is
going away. AFH now writes the programs that matter most to it:

| role | ours | still external |
|---|---|---|
| electronic structure | linus (HF/DFT, GPU), axtb (GFN2, g-xTB, GFN-FF) | ORCA, Psi4, TeraChem (benchmarks, methods linus lacks: CC, DLPNO, CAS); ML potentials (AIMNet2, UMA, ...) |
| geometry drivers | qcopt (minima, TS, IRC) | geomeTRIC, Sella (benchmarks only); CREST (conformers); GSM (paths) |
| workflows | mepd (reaction discovery, explorer, kinetics, web) | — |

Our programs already speak qcdata (each ships a qccodec codec that only packs a call to
itself and unpacks the result), and the program interface now lets them do so without
qccodec at all.

**mepd shows what the stack is missing.** It is 99k lines of Python (plus 16k of JS), and a
large share of it rebuilds what a shared stack should provide:

- **Its own engine layer** (`mepd/engines/`, 9.4k lines): an abstract `Engine`
  (`compute_gradients`, `compute_energies`, optional `compute_hessian`,
  `compute_transition_state`, ... found by `hasattr`), with adapters that bypass qc*:
  - g-xTB through the xtb binary with regex parsing, or through axtb;
  - Psi4 through a JSON-lines worker;
  - TeraChem through its own `tc.in`;
  - ORCA through `qccodec.encode` with mepd's own `subprocess.run`;
  - CREST, `xtb --bhess` and GSM through subprocesses and sockets.
  qcopt is driven through `qcopt.FunctionEvaluator` over that engine, so qcopt never sees
  a qcdata program.
- **Its own results:** `FakeQCIOOutput`, `FiniteDifferenceHessianOutput`, `IRCResult`,
  `TsIrcResult`, `ElemStepResults`. It never uses `SinglePointData`, `OptimizationData`
  or `Frame`.
- **Four network models that share no schema:** `pot.Pot` (a networkx graph),
  `network_expansion.Species`/`ExpansionResult`, `network_model.NetworkModel`,
  `microkinetics.Network`. Three kinetics solvers sit beside them. The live view expects a
  `step_list` schema that nothing on develop writes.
- **Its own orchestration and caching:**
  - a fork-based process pool with a priority queue (`HorizontalConnector`);
  - resume from a pickled `expand_state.pkl`;
  - half a dozen ad hoc hash caches (thermo, steering single points, Hessians) and
    `.energies`/`.gradients` sidecar files;
  - per-program special cases (`type(engine).__name__ == "GXTBCalculator"`).
- **Cluster use is one whole exploration per pod.** No individual calculation is ever
  dispatched, so an exploration cannot grow past one machine.

None of this is a criticism of mepd. It is what a research code does when the layer under
it does not give it sessions, standard results, a calculation store, or a way to run
calculations anywhere. **That layer is qc*'s job**, and the rest of this document is what I
think it should look like.

## 2. The shape: five layers and one standard

```
L4  apps and analysis      mepd web / Explorer, findings, notebooks, viewers
                           read networks and calculation records; never run programs
L3  research workflows     mepd explorer and channels, campaigns, benchmarks
                           domain logic and steering, in plain Python
                           data: ReactionNetwork, Protocol (levels of theory)
L2  drivers                qcopt (min, TS, IRC; later NEB/string/GSM), MD, conformer search,
                           path methods; programs whose children are programs
L1  engines                linus, axtb, ML potentials; external codes through adapters
L0  execution              qccompute: local threads/workers/GPUs, a calculation store,
                           cluster backends (afh, astera; ChemCloud)

qcdata spans every layer: inputs, outputs, frames, sessions, networks, JSON Schema
```

Five rules hold it together:

1. **Everything that computes is a `Program`.** Engines (linus, axtb) and drivers (qcopt)
   alike implement `qccompute.program.Program` and are addressed by a `ProgramInput`. So any
   calculation, from a g-xTB gradient to a DFT IRC, can run in this process, in a worker,
   or in a cluster pod, and can be cached, without the caller knowing which.
2. **Two granularities, two interfaces.** Inner loops (an optimizer's steps, NEB images,
   finite differences, MD) use a `Session`: warm-started, in-process, no serialization,
   frames out. Outer loops (an optimization, a TS search, an IRC, a Hessian, a DFT single
   point) are whole `ProgramInput` jobs: serializable, cacheable, distributable. Rule of
   thumb: if a unit of work is a result someone might want to look up later, it is a job.
   If it is a step inside one, it is a session call.
3. **Calculations are immutable and content-addressed.** Every `ProgramOutput` is stored
   under a key computed from what determines it. Everything above (species, reactions,
   free energies, figures) refers to calculations by key, and never re-parses or copies
   numbers ad hoc. Resume, deduplication across workflows and provenance all follow.
4. **Workflows are plain Python** against a small futures API (`submit`, `gather`,
   `as_completed`) whose execution goes through the store. A rerun is a sequence of cache
   hits, so workflows are resumable by construction: no pickled state, no workflow engine.
5. **Domain data has one schema.** Reaction networks, levels of theory and kinetics are
   qcdata models with a published JSON Schema. mepd writes them; the web, the landing,
   findings and new apps read them.

## 3. Is qccodec necessary?

**Not as a central layer.** Its job, encapsulating an external program's formats and
quirks, is now the job of an *adapter that makes an external program a `Program`*. That is
an optional plugin, not the middle of the stack. Concretely:

- **qccompute stops depending on qccodec.** It runs `Program`s only: native ones from the
  `qccompute.programs` entry points, external ones through the same entry points.
- **qccodec becomes the package of external programs.** It keeps its codecs and parsers as
  internals, and exports ORCA, Psi4, TeraChem, CREST and the ML potentials as `Program`s
  through an `ExternalProgram` adapter. The adapter runs the subprocess itself, so the
  subprocess runner moves from qccompute into qccodec. Installing qccodec adds those
  programs; not installing it leaves a lean stack of AFH programs.
- **Its legacy codecs retire as we replace them.** xtb gives way to axtb, and geomeTRIC and
  Sella give way to qcopt; they stay only as long as benchmarks need them. ML potentials
  could later move to a small native package (they are in-process Python with batching,
  which fits `Program` + `compute_batch` directly).
- **Dependencies become** qcref → qcdata → {qcinf, programs, qccodec} → qccompute.
  Workflows depend on qccompute, plus whichever programs they install.

The name could change to say what it is (for example "qcexternal"). I would keep the name
for continuity and revisit it later. **[Colton]**

### Standard run formats

"Speaking qcdata" should hold at the file level too, so no program needs Python glue to run
on the cluster:

- Every program has the same command-line contract:
  - `<program> run input.json -o output.json` reads a `ProgramInput` and writes a
    `ProgramOutput`;
  - `<program> plan input.json` prints the resolved settings and resources;
  - `<program> schema settings|record --calctype X` prints the program's JSON Schemas.
- linus and axtb, being Rust, can read and write these files natively against qcdata's
  published JSON Schema, with no Python in the container.
- `qccompute run input.json` dispatches any installed `Program` the same way. That is what
  a cluster task runs.
- A job-runner task is a `ProgramInput` file; its output is a `ProgramOutput` file that
  lands in the store.

## 4. Execution: one API from a laptop to the cluster

`qccompute.compute(inputs)` already runs lists in parallel locally. The proposal grows it
along two axes without changing how scripts read.

**The calculation store** (qcstore, its own package depending on qcdata only; the AFH
Stack Plan has the full design):

- **Key:** `qcdata.calculation_key(inp, key_context(inp))`, a hash of the input as given:
  keys sorted, numbers normalized, fields at their default left out, coordinates rounded,
  files by digest, subprograms recursively. Only `threads` and `memory_gib` are left out.
  Spellings are not normalized, so a synonym or an explicit default costs a cache miss,
  never a wrong hit. Programs are never consulted, so readers need none installed.
- **Numerics identity, not version** (linus's lesson from a year of its reference store). A
  version is too coarse and too fine at once: most bumps do not change numbers, and some
  changes the version does not track do (glibc 2.36 vs 2.43 moved linus results at
  round-off through erf/erfc). The store takes a `key_context` function, `inp →
  dict[str, str]`, that reads static configuration (image digest, numerics epochs, GPU
  class) and is mixed into the hash. For example, linus publishes `linus.NUMERICS_EPOCH`
  (6 since linus `cae0bad`, when `jk` began defaulting to `df`). The store's
  configuration says `{"linus": "6"}`, and the runner refuses to write when the installed
  epoch disagrees. The exact version and commit go in provenance, not the key. Reuse is
  then broad across releases, and never across builds that compute different bits.
- **Value:** the `ProgramOutput` (compressed JSON; large arrays later as sidecars). The
  index is SQLite: key, program, method, formula, calctype, success, wall time, campaign
  tags, created.
- **Backends:** a local directory, the shared NFS `/jobs/store` on the cluster, and B2 for
  the archive.
- **Policy:** `compute(..., store=...)` looks up first and writes after. Failures are
  stored and retried by default, unless the caller asks to reuse them.
- **What it replaces:** mepd's thermo, steering and Hessian caches, the pickled explorer
  state and the sidecar files all become store lookups.

**Execution backends** behind one futures API:

```python
with qccompute.Executor(backend="afh", store="/jobs/store", image="afh-runtime:2026.11") as ex:
    ts = ex.map(ts_inputs)                 # futures; cache hits resolve at once
    for out in ex.as_completed(ts):        # IRCs start as TS searches finish
        if out.success:
            irc = ex.submit(irc_input(out))
```

- **`local`:** today's thread pool, workers and GPU pinning.
- **`afh` / `astera`:** the job runner. Fan-outs become Indexed Jobs of `qccompute run`
  tasks, as today's job-runner already does with `tasks.jsonl`. For dynamic workflows (the
  explorer submits work as results arrive), long-lived worker pods pull `ProgramInput` files
  from a queue directory on the NFS share, claiming each by atomic rename. That needs no new
  infrastructure, and pods are sized by `Plan.resources` (threads to CPU requests, GPUs,
  memory).
- **`chemcloud`:** later, for TeraChem.

The image is one AFH runtime image (qc*, linus, axtb, qcopt, the external programs), built
per release. mepd's `mepd-orca-uma-axtb` image is already most of it.

## 5. Domain data: reaction networks and levels of theory in qcdata

mepd's four network models should become one, in qcdata (pure pydantic data, references by
key, no networkx dependency), with algorithms in qcinf and science in mepd. A first sketch,
to design together with Jan: **[Jan]**

| model | holds |
|---|---|
| `Species` | `id` (canonical: connectivity graph hash + charge + multiplicity; stereo when known), `smiles`, `formula`, `charge`, `multiplicity`, `conformers: list[Conformer]` |
| `Conformer` | `structure`, and calculation keys by role: `optimization`, `hessian`, `energy` (one per protocol) |
| `TransitionState` | `id`, `structure`, calculation keys (TS optimization, Hessian, energies), `imaginary_frequency`, conformers |
| `ElementaryStep` | `id`, `reactants`/`products` (species ids with stoichiometry), `transition_state`, `irc` (calculation key), `verified` (the IRC connects these ends), `source` (how it was found: generator, channel search, imported) |
| `Protocol` | a named level of theory: `geometry: ProgramSpec`, `energy: ProgramSpec \| None` (single-point refinement), `thermo: ProgramSpec \| None` (Hessian), `standard_state`, `temperature`. It defines how G is assembled from referenced calculations ("g-xTB//g-xTB + r2SCAN-3c SP") |
| `Energies` | per species, TS and protocol: E, ZPE, H, G, each with the calculation keys it came from (derived, recomputable) |
| `ReactionNetwork` | species, transition states, steps, protocols, energies, conditions, and the exploration's provenance (workflow version, settings, seeds); optionally its event log |
| `KineticModel` | rates per step at T, mechanism, concentration traces |

Why in qcdata: networks are *the* standard output of AFH's main research workflow, and the
user's goal is that apps and viewers build on standard outputs quickly. The models are data
only; kinetics solvers (mepd has three), species identity, geodesic interpolation and
`vibrational_analysis` belong in qcinf as shared algorithms. **[Colton]**: in qcdata, or a
separate `qcnet` package.

## 6. Where each program sits, and what changes for it

**linus, axtb (L1).** Implement `Program` and `SessionProgram`: plan, schemas, sessions
keyed by lineage, typed results. This is already agreed; linus's IFACE refactor and axtb 0.2
carry it. They also adopt the file-level `run`/`plan`/`schema` contract.

**qcopt (L2).** Becomes a `Program` whose children are sessions (`ctx.session`). This is
prototyped on qcopt's design-0.3 against these branches: it passes `check_program`, and
the 77 golden trajectories are bit for bit. The per-program evaluators go. qcopt keeps
its Evaluator protocol and `FunctionEvaluator`, so it can still optimize on a plain
Python function: model surfaces, tests, an in-process ML potential, and mepd's engines
until they are programs. qcopt is the natural home for path methods:
- NEB and string methods fit its generator/lockstep design: each image is a request of a
  round and a lineage of its session.
- FSM-TS composes from what exists: the path, a TS search from a guide direction (the
  path tangent), and the IRC. As a reaction-path calctype it would make mepd's default
  elementary-step search a single cacheable job.
- GSM through the external binary stays in mepd; a native growing string would be new
  work.

The order is after qcopt 0.4, one calctype at a time, each benchmarked against mepd's
current implementation on the Reaction-QM set before mepd switches. **[Colton, Jan]**

**mepd (L3/L4).** The biggest gains, with incremental steps that each pay on their own.
**[Jan]**

1. **Engines become sessions.** A single `SessionEngine` adapter maps mepd's `Engine`
   interface onto `qccompute.session(inp)`, so every caller keeps working.
   Then the g-xTB, xtb, ORCA, Psi4, TeraChem and MLIP engines collapse into `ProgramSpec`s
   in profiles, `StructureNode` caches `Frame`s instead of `FakeQCIOOutput`, and
   `xtb --bhess` gives way to the vibrational analysis in axtb's Hessians (symmetry number
   included). About 9k lines of adapters and parsers go, along with the
   `GXTBCalculator` name checks.
2. **qcopt through `Program`.** TS searches, IRCs and minimizations become `ProgramInput`
   jobs (`program="qcopt"`, `subprograms=[engine spec]`) instead of `FunctionEvaluator`
   callbacks over mepd's engines, so they can be cached and dispatched.
   (`FunctionEvaluator` itself stays in qcopt for Python-function surfaces.)
3. **One network schema.** `ExpansionResult`, `summary.json`, `Pot`, `NetworkModel` and the
   workspace become `qcdata.ReactionNetwork`. The live views and the landing read it; the
   kinetics read `Energies` under a `Protocol`.
4. **Orchestration on the executor and store.** `HorizontalConnector`'s policy (priorities,
   steering, stopping) is mepd's science and stays. Its jobs go through `Executor.submit`,
   so an exploration resumes from the store instead of a pickle, and can spread over the
   cluster instead of one pod.

**Apps (L4).** A viewer for any `ReactionNetwork` and any stored calculation (structures,
trajectories, spectra, orbitals from `Wavefunction.orbitals`, energies), generated against
the JSON Schemas. mepd's Explorer becomes one such app.

## 7. Workflows: what a research script looks like

```python
from qccompute import Executor
from qcdata import ProgramInput, ProgramSpec, Model, ReactionNetwork, Protocol

engine = ProgramSpec(program="axtb", calctype="gradient", model=Model(method="gxtb"))
dft = Protocol(name="r2scan-3c//gxtb", geometry=engine,
               energy=ProgramSpec(program="linus", calctype="energy",
                                  model=Model(method="r2scan-3c", basis="def2-mtzvpp")))

with Executor(backend="afh", store="/jobs/store") as ex:
    net = ReactionNetwork(protocols=[dft])
    guesses = propose_reactions(seed)                      # mepd: generators
    for ts in ex.as_completed(ex.map(qcopt_ts(g, engine) for g in guesses)):
        if ts.success and ts.results.final_frame_is_saddle():
            irc = ex.submit(qcopt_irc(ts, engine))
            net.add_step(ts, irc.result())                 # keys, not copies
    ex.map(dft.energy_inputs(net))                         # refine what kinetics need
    net.save("glycine.network.json")                       # the app reads this
```

Rerunning the script after a crash, or with a new protocol, recomputes only what is
missing.

## 8. Roadmap

| phase | what | who | status |
|---|---|---|---|
| 1 | qcdata typed slots: outcome, vibrations, thermochemistry, orbitals | qc* | done locally (`program-interface`) |
| 2 | `qccompute.program` protocol, sessions, `check_program` (qcdata keeps the data: `Request`, `Accuracy`, `ProgramError`); qccompute native path | qc* | done locally (qcdata `data-only`, qccompute `execution-protocol`) |
| 3 | programs adopt: linus IFACE, axtb 0.2, qcopt 0.3 (slots), then 0.4 (sessions) | program sessions | agreed |
| 4 | the store; the `Executor` with local and afh backends; `qccompute run`; qccodec as external `Program`s; qccompute without qccodec | qc* | proposed |
| 5 | network models and `Protocol` in qcdata, `qcinf.vibrational_analysis` and kinetics, designed with Jan | qc* + Jan | proposed |
| 6 | mepd steps 1–4; a generic viewer | Jan (+ qc* help) | proposed |

## 9. Decisions

For Colton:

1. qccodec leaves the middle of the stack and becomes the optional package of external
   programs; qccompute drops its dependency on it. *Recommended.*
2. A calculation store and an executor with cluster backends belong in qccompute (not in
   mepd or the job runner). *Recommended.*
3. Reaction networks and levels of theory become qcdata models (rather than a new package
   or staying in mepd). *Recommended, with Jan.*
4. Every program offers the `run`/`plan`/`schema` file contract. *Recommended.*
5. Path methods consolidate in qcopt over time. *Recommended, with Jan and qcopt.*

For Jan: whether the mepd steps in §6, in that order, fit the explorer's plans (its work is
paused since 2026-11-09, which makes this a good moment), and what the network schema must
hold for his science.
