# qccompute

qccompute runs programs on [qcdata](https://qcdata.docs.afh.org) inputs. Every program
runs the same way, whether it implements qccompute's protocol in Python or is an
external program that [qccodec](https://qccodec.docs.afh.org) brings in:

```sh
python -m pip install qccompute            # the runner
python -m pip install qccompute[qccodec]   # and external programs (xtb, ORCA, Psi4, ...)
```

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

A program with sessions of its own hands back its own session object, so the numbers are
exactly those of the program used directly; any other program gets a session over
`compute`. A failed request raises `ProgramError` and the session stays usable;
`compute_many` runs every request, then raises `qcdata.ProgramErrors`, whose `results`
holds every frame in request order.

## A share of the machine: the Runner

`qccompute.compute` and `qccompute.session` are the methods of the default runner: every
core this process may use (its CPU affinity and cgroup quota) and every visible GPU. It is
made on first use and cannot be reconfigured. A `Runner` of your own takes a share:

```python
with qccompute.Runner(threads=16, gpus=["0"], per_gpu=2, isolation="process") as runner:
    outputs = runner.compute(inputs, timeout=3600)
```

- `threads`, `gpus`: the share. Each calculation gets its own share of it in a
  `qccompute.RunContext`; host memory and GPU memory follow. A calculation's share is the
  larger of its share of the threads and its share of the GPUs; with `per_gpu=k`, k
  calculations share each GPU and each counts as 1/k of it.
- `isolation`: `"auto"` runs programs in this process unless they cannot share it (they
  say `in_process=False`), which run in worker processes; `"process"` runs every program
  in a worker, so a crash ends only the worker.
- Workers stay alive between calls, so a program keeps its state; close the runner (or use
  it as a context manager) to stop them.

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
`{program: numerics}` for the input's program and every subprogram it names. A stored run
is reused only when its identities are those of the programs that would run now; a
stored failure only when it is deterministic (kind `input` or `unsupported`).
`rerun=True` always runs; `tags` go onto every key, cache hits included. A store
implements `key`, `get`, `put` and `tag` (`qccompute.runner.Store`); qcstore is one.

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
are the programs' own; each [program page](https://qccodec.docs.afh.org/programs/)
explains the mapping. A program that is not on `PATH` is found with
`QCCODEC_<PROGRAM>_EXECUTABLE`. A `qcdata.FileInput` (a command and its files) runs
under qccodec's `files` program and returns every file the command wrote; see
[File Input](examples/file_input.md).

## Workers

`python -m qccompute.serve PROGRAM` hosts a program in its own process, over JSON lines on
stdin and stdout (or a Unix socket); `qccompute.worker.Worker` is its client, with the
same `compute`, `plan` and sessions as the program in process. The runner uses them for
programs that cannot share a process and for `isolation="process"`.
