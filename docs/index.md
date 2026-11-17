# qccompute

qccompute runs programs on [qcdata](https://qcdata.docs.afh.org/next/) inputs. Every program
runs the same way, whether it implements qccompute's protocol in Python or is an
external program that [qccodec](https://qccodec.docs.afh.org/next/) brings in:

```sh
uv pip install --prerelease=allow "qcdata==0.24.0a6" "qccompute[qccodec]==0.23.0a5" "qccodec==0.22.0a4" "qcstore==0.1.0a4"
```

This is the alpha line, pinned with its matching packages (`--prerelease=allow` lets the
resolver take pre-releases); a plain `pip install qccompute` resolves the previous stable
release. `qccompute` alone is the runner (`"qcdata==0.24.0a6" "qccompute==0.23.0a5"`);
the `qccodec` extra adds external programs (xtb, ORCA, Psi4, ...) and `qcstore` is the
calculation store. These pages document the alpha, at
[qccompute.docs.afh.org/next](https://qccompute.docs.afh.org/next/); the site root
documents the stable release. Coming from 0.22? See [the migration guide](migration.md).

## One calculation

```python
import qccompute
from qcdata import CalcType, Model, ProgramInput, Structure

water = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],  # bohr
)
inp = ProgramInput(
    program="xtb", calctype=CalcType.gradient, structure=water, model=Model(method="gfn2")
)
out = qccompute.compute(inp)
print(out.results.energy, out.results.gradient)
```

qccompute finds the program by name (`ProgramInput.program`) among the installed programs
(the `qccompute.programs` entry points) and those registered in this process with
`qccompute.register_program`. It hands the calculation its share of the machine, runs it,
and records the run in `out.execution` (wall time, host, threads).

## Failures raise

A failed calculation raises `qcdata.ProgramError`. Its `error` says what kind of failure
it was (`input`, `convergence`, `timeout`, `environment`, ...) and its `output` is the
failed `ProgramOutput`, with the partial results, the logs and the traceback:

```python
from qcdata import ProgramError

try:
    out = qccompute.compute(inp)
except ProgramError as e:
    print(e.error.kind, e.error.message)
    print(e.output.logs)
```

`raise_exc=False` returns the failed output instead. A program's bug (any other exception)
is a failure of kind `internal`, caused by the original exception.

## Many calculations

A list runs side by side and returns its outputs in input order. The inputs may mix
programs, methods and calculation types:

```python
outputs = qccompute.compute([inp1, inp2, inp3])
```

- Every input is planned first: an input its program refuses (an unknown keyword) fails at
  once, without running.
- Calculations are packed into the runner's share of the machine by what their plans ask
  for (threads, GPUs); a program that evaluates many inputs at once (a machine-learned
  potential) gets its inputs together.
- Everything runs, then the failures are raised together as `qcdata.ProgramErrors` (an
  `ExceptionGroup` of `ProgramError`s). Its `results` holds every output in input order,
  so no result is lost:

```python
import qcdata

try:
    outputs = qccompute.compute(inputs)
except qcdata.ProgramErrors as errors:
    outputs = errors.results
```

`Runner.as_completed(inputs)` yields outputs as they finish.

## Sessions: one molecule at many geometries

Optimizers, IRCs, dynamics and finite differences evaluate one molecule at many
geometries. A session pins the program, model, keywords, atoms, charge and multiplicity,
and lets a program keep what speeds the next step (orbitals for a warm start):

```python
with qccompute.session(inp) as s:
    frame = s.compute()                       # the input's own geometry and calctype
    frame = s.compute(new_positions)          # warm-started from the last success
    hessian = s.compute(new_positions, "hessian")
```

Only a program with sessions of its own (`capabilities().sessions`) warm-starts: it
hands back its own session object, so the numbers are exactly those of the program used
directly. Any other program (most external programs, tblite, a `FunctionProgram`) gets a
session over `compute`, which computes every request from scratch
(`record["warm_start"]` is false): a convenience, not a speed-up. A failed request raises
`ProgramError` and the session stays usable; `compute_many` runs every request, then
raises `qcdata.ProgramErrors`, whose `results` holds every frame in request order. A
session's threads are set when it opens, `qccompute.session(inp, threads=8)`, else what
its input asks for: a program's own session splits them across the lineages it runs at
once, and a session over `compute` runs `compute_many`'s requests side by side, as a
list.
Lineages name separate histories in one session (both branches of an IRC), `seed`
starts one from another, and `commit=False` reads a history without moving it; see
[Sessions, lineages and seeds](examples/sessions.md).

## A share of the machine: the Runner

`qccompute.compute` and `qccompute.session` are the methods of the default runner: every
core this process may use (its CPU affinity and cgroup quota) and every visible GPU. It is
made on first use and cannot be reconfigured.

Each calculation gets the threads it asks for: its `threads` keyword, else what its
program's plan says (1 unless the program states more; a program that scales, such as a
DFT code, may ask for all of the runner's), at most the runner's. A 3-atom gradient does
not get 128 threads because the machine has them; ask for more with
`keywords={"threads": 8}`. Sessions and subprograms get threads the same way.

A `Runner` of your own takes a share:

```python
with qccompute.Runner(threads=16, gpus=["0"], per_gpu=2, isolation="process") as runner:
    outputs = runner.compute(inputs, timeout=3600)
```

- `threads`, `gpus`: the share, and the most one calculation gets. Each calculation gets
  its grant in a `qccompute.RunContext`. In a list, a calculation's share is the larger
  of its share of the threads and its share of the GPUs; with `per_gpu=k`, k
  calculations share each GPU and each counts as 1/k of it. A calculation gets the GPUs
  its program's plan asks for, and `ctx.device_ordinals()` turns them into CUDA device
  numbers in the program's process.
- Memory is never scheduled. A calculation run alone gets all of the runner's memory; in
  a list each gets its share; each gets 1/k of a GPU's memory. A `memory_gib` or
  `device_memory_gib` keyword replaces these, at most what the runner (or, for a
  subprogram, its parent) has.
- `isolation`: `"auto"` runs programs in this process unless they cannot share it (they
  say `in_process=False`), which run in worker processes; `"process"` runs every program
  in a worker, so a crash ends only the worker.
- Workers stay alive between calls, so a program keeps its state; close the runner (or use
  it as a context manager) to stop them. A program's workers serve its calls whatever
  their threads, so a model loads once: on a GPU one worker per GPU (`per_gpu` of them),
  which its calculations and sessions share; on the CPU as many as run side by side.

The runner never changes a number by itself. A grant may change a program's own choices
(a storage scheme picked from the memory granted), and the program records each one in
`output.resolved`.

## A calculation store

```python
outputs = qccompute.compute(inputs, store=store, tags=["screen-1"], meta={"by": "me"})
```

With a store, every input is looked up first and only misses run; every run is stored.
A store files runs by the calculation's key (the input as given,
`qcdata.calculation_key`) and by the *identities* of the programs that made them:
`{program: numerics}` for the input's program and every subprogram it names (a program
named with several specs has its identities joined by `;`). Every output says which
code made it in `execution.identities`, with or without a store. A stored run
is reused only when its identities are those of the programs that would run now; a
stored failure only when the input determines it (kind `input`, `unsupported` or
`convergence`, or `program` with such a cause), never one of the machine or the moment
(`resource`, `timeout`, `environment`, ...). `rerun=True` always runs, past a stored
failure too; `tags` go onto every key, cache hits included. Every output returned with a
store names its run, computed or hit, in `execution.store_run` (`"<key>/<run>"`), so a
network or a report can cite the exact run and read it back with
`store.get(key, run=...)`. A store implements `key`, `get`, `put` and `tag`
(`qccompute.runner.Store`); qcstore is one.

## One calculation at a time, as futures

`qccompute.submit(inp)` (and `Runner.submit`) starts a calculation and returns a
`concurrent.futures.Future` of its output, for workflows whose next step depends on the
last: per reaction, the IRC starts as soon as its transition-state search ends, with no
stage barriers. Submitted calculations share the runner's threads as a list's do; a
stored result resolves at once. See [Dynamic workflows](examples/dynamic_workflows.md).

A program that is not thread-safe (qcopt) runs one calculation at a time per process.
The runner warns the first time one waits for its turn; `Runner(isolation="process")`
runs them side by side in worker processes.

## Progress

Programs can report what they are doing while they run (qcopt: one event per step).
Pass a callable to see it:

```python
def show(event):  # {"event": "step", "step": 12, "energy": -76.4, "path": ["qcopt"]}
    print(event["path"], event)

qccompute.compute(inp, progress=show)
```

Each event is a small JSON object of the program's own, with `path` added: the program
names from your input down to the one that emitted it, so a subprogram's events
(`["neb", "qcopt"]`) arrive too. The callable may be called from several threads
at once. Stored outputs emit nothing, and nothing is stored. A program in a worker
process streams its events back. `submit`, `as_completed` and `session` take
`progress=` too; to tell a list's inputs apart, submit each with its own callable.

## Programs

A program is an object implementing `qccompute.program.Program`: its `name`, `version()`,
`capabilities()`, the JSON Schema of its settings and of its record, `plan(inp)` (a dry
run that validates and resolves every `"auto"`) and `compute(inp, ctx=None)`. Programs
register an instance under the `qccompute.programs` entry points:

```toml
[project.entry-points."qccompute.programs"]
myprogram = "mypackage.qc:program"
```

The protocol is structural, so a program without subprograms imports only qcdata. A
program with subprograms runs them through `ctx.compute` and `ctx.session`; called
directly (`ctx is None`), it returns `qccompute.compute(inp)`. See
[Programs and sessions](api/program.md) and [the program interface](design/program-interface.md).

For models and scripts, `FunctionProgram` makes a program of plain functions:

```python
{!examples/function_program.py!}
```

`qccompute.testing.check_program(program, inputs)` checks a program against the contract
(schemas, plans, typed errors, the thread grant, sessions, batches);
`qccompute.testing.toy` is a complete example program.

## External programs

`qccompute[qccodec]` installs qccodec, which registers xtb, CREST, ORCA, Psi4, TeraChem,
DFTB+, geomeTRIC, Sella, tblite and machine-learned potentials as programs. Their keywords
are the programs' own; each [program page](https://qccodec.docs.afh.org/next/programs/)
explains the mapping. A program that is not on `PATH` is found with
`QCCODEC_<PROGRAM>_EXECUTABLE`. A `qcdata.FileInput` (a command and its files) runs
under qccodec's `files` program and returns every file the command wrote; see
[File Input](examples/file_input.md).

## Workers

`python -m qccompute.serve PROGRAM` hosts a program in its own process, over JSON lines on
stdin and stdout (or a Unix socket); `qccompute.worker.Worker` is its client, with the
same `compute`, `plan` and sessions as the program in process. The runner uses them for
programs that cannot share a process and for `isolation="process"`.

A program in a worker runs its subprograms back in your process: geomeTRIC's gradients
can come from a `FunctionProgram` you registered in your script, and they share your
runner's grant. Large arrays (Hessians, wavefunctions) cross between the processes as
`.npy` files rather than as text.
