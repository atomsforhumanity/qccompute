# The program interface

How qccompute runs programs: the protocol a program implements, the context a runner
gives each calculation, sessions, failures, the runner, the store, and worker processes.
This is the design as built; the API pages document each name.

## The layers

```
workflows, apps, analysis        read and write qcdata; call qccompute
─────────────────────────────────────────────────────────────────────────
qcdata     data only: inputs, outputs, Frame, Request, Accuracy, ProgramError,
           conventions, the calculation key, the save format
qccompute  the Program protocol (qccompute.program) and the Runner: discovery,
           planning, placement, resources, lists, sessions, workers, the store hook
─────────────────────────────────────────────────────────────────────────
programs   implement the protocol; a program without subprograms imports only qcdata
qccodec    external programs (xtb, ORCA, Psi4, CREST, ...) as programs: input files,
           subprocesses, output capture, file collection, parsers
```

qccompute depends on qcdata only. qccodec is the optional add-on for external programs
(`pip install qccompute[qccodec]`) and does not import qccompute to run them.

## Programs

There is one kind of program. A program maps one `ProgramInput` to one `ProgramOutput`
and may have subprograms (`ProgramSpec`s in its input) that it runs through the runner.
Running many programs, with logic between them, is a workflow: plain Python.

A program registers an instance under the `qccompute.programs` entry point group, or in a
process with `qccompute.register_program`:

```toml
[project.entry-points."qccompute.programs"]
myprogram = "mypackage.qc:program"
```

The object implements `qccompute.program.Program`, structurally:

| member | what it is |
|---|---|
| `name` | the program's name, as in `ProgramInput.program` |
| `version()` | the program's version |
| `capabilities()` | calctypes, methods with their calctypes, `sessions`, `batches`, `in_process`, `thread_safe`, `gpu`; a `Capabilities` or a plain mapping |
| `settings_schema(calctype)` | the JSON Schema of `keywords` and of `resolved.settings` |
| `record_schema(calctype)` | the JSON Schema of `ProgramOutput.record` |
| `plan(inp)` | a dry run: validates (kind `input` with a did-you-mean for unknown keys), resolves every `"auto"`, and says what the calculation needs (`Resources`: threads, GPUs) |
| `compute(inp, ctx=None)` | runs the calculation; returns the successful output or raises `ProgramError` |
| `numerics(spec)` (optional) | a string that changes whenever results for an unchanged input may change; the version by default |
| `open_session(inp, ctx=None, calctypes=None)` (optional) | a session, usually the program's own session class |
| `compute_batch(inputs, ctx=None)` (optional) | many inputs at once; one output per input, in order, failures in place |

What a program returns for `capabilities` and `plan` may be a plain mapping, which
qccompute validates (`capabilities_of`, `plan_of`): it is the same document a program's
command line prints and its worker sends. `numerics` takes the spec it would run (the
input, or a subprogram's `ProgramSpec`), so a program of several methods can give each
its own identity: a numerics epoch, an executable's sha256, a model's checksum.

`qccompute.program.FunctionProgram(name, energy_gradient, hessian=None)` is the smallest
complete program: functions of the positions, no settings, in process and thread-safe.

## The context

`compute`'s second argument is a `RunContext`, which only a runner passes, and a runner
always passes one. It carries the calculation's share of the machine and the way to run
subprograms:

- `threads`, `gpus`: the grant. The threads are what the input asks for (its `threads`
  keyword, else `plan.resources.threads`, which is 1 unless the program says more), at
  most the runner's (or, for a subprogram, the parent's grant). A program uses the
  smaller of its own request and the grant, or the grant when it requested nothing; a
  capped request is a `resource_capped` warning.
- `memory_gib`, `device_memory_gib`: host and GPU memory, following the calculation's
  share (below).
- `timeout`: the seconds left. A program that starts processes kills them when it runs
  out and fails with kind `timeout`.
- `plan`: the runner's plan of the input, so the program need not plan again.
- `scratch`: a directory, made on first use and removed when the call returns.
- `compute(inputs, raise_exc=True)` and `session(spec, structure, threads=, calctypes=)`:
  subprograms, within the grant.

`ctx is None` means a direct call: the program uses its own settings. A program with
subprograms has no route of its own to programs, so called directly it returns
`qccompute.compute(inp)`, and the default runner runs it with a context. The runner finds
programs by name: one that is neither installed nor registered fails with kind
`environment`, and the message says how to register it.

The runner never changes a number by itself. A grant may change a program's resolved
choices (a storage scheme picked from the memory granted, a precision from the device);
the program records each one in `resolved`.

## Failures

A failed calculation raises `qcdata.ProgramError`, from `compute` as from sessions. Its
`error` is an `ErrorInfo` (`kind`, `message`, an optional `code`, and for a failed
subprogram the subprogram's error as `cause`), and its `output` is the whole failed
output: partial results, logs, record, traceback. A program whose result is one output
raises `ProgramError(out.error, output=out)`; when a program raises without one, the
runner attaches the output it built. Any other exception from a program is a bug: the
runner reports it as kind `internal`, caused by the original exception.

The runner turns failures into failed outputs only where it needs data:

- a list runs every calculation, then raises `qcdata.ProgramErrors`, an `ExceptionGroup`
  of the `ProgramError`s whose `results` holds every output in input order;
- a session's `compute_many` runs every request, then raises the same group, whose
  `results` holds every frame in request order (`ProgramErrors.check(frames)`);
- `compute_batch` returns failures in place, a refused input included;
- `raise_exc=False` returns failed outputs in place; the store and workers carry failed
  outputs as data.

A subprogram's failure that a program lets escape fails it with kind `program`, the
subprogram's name and its error as `cause`; a program that catches it sees the
subprogram's own error. The runner sets the kinds `timeout` (a deadline) and
`environment` (a program not installed, a worker that died: code `worker_died`).

## Sessions

The same program and molecule at many geometries, warm-started: what optimizers, IRCs,
dynamics, path methods and finite differences need. `prog.Session(inp)` is both the
program's own API and the protocol object; `open_session` returns it.

```python
with program.Session(inp) as s:            # every calctype the method supports
    f = s.compute()                         # the input's geometry and calctype
    f = s.compute(x2)                       # warm start from this lineage's history
    h = s.compute(x2, calctype="hessian")   # energy and gradient too
    g = s.compute(y, lineage=("irc", "forward"), seed=["ts"])
    fd = s.compute_many([Request(positions=p, commit=False) for p in displaced])
```

- The input pins atoms, charge, multiplicity, model and keywords; its geometry and
  calctype are a request's defaults. `calctypes=` narrows what may run; a setting that
  applies to none of them is an input error (`not_applicable`).
- Results are `Frame`s, with the shared `warm_start` record key.
- A lineage is any JSON value, keyed by its canonical text (`qcdata.lineage_key`).
  `seed` lists lineages to start from when the request's own has no history; `commit`
  (default true) makes a success the lineage's history.
- `compute_many` runs requests of one lineage in order, each reading its lineage's
  history as it stands at its place; requests of different lineages may run concurrently.
- A failed request raises `ProgramError` with its failed frame; the session and its other
  lineages go on, and the lineage's history is untouched.
- Sessions may be called from several threads; the runner serializes those of programs
  that are not thread-safe.

`qccompute.session(inp)` hands back the program's own session in process, so the numbers
are exactly those of the program used directly. A program without sessions gets a
`ComputeSession`: one stateless calculation per request (`warm_start` false). A program
in a worker gets a `WorkerSession` over the pipe, with the same methods.

## The runner

`Runner(threads=None, gpus="auto", backend="local", isolation="auto", per_gpu=1)` with
`compute` (one input or a list), `as_completed` and `session`. Every call takes the same
options: `store`, `timeout`, `rerun`, `tags`, `meta`, `raise_exc`. `qccompute.compute`
and `qccompute.session` are the methods of the default runner: this machine (every core
this process may use, every visible GPU), made on first use, never reconfigured.

Every input goes one path: find the program, plan it (an input its program refuses
fails without running), place it (this process, or a worker process for programs with
`in_process=False` and for `isolation="process"`), call `compute` or `compute_batch`
with a context, record the execution.

**Resources.** qc* never allocates machines, queues work or enforces limits: a
dispatcher (a batch system, Kubernetes) gives the process its cores, memory and GPUs. A
runner shares them out. Each calculation gets the threads it asks for (its `threads`
keyword, else its plan's: 1 unless the program says more), at most the runner's; a
session likewise, and a subprogram at most its parent's grant (or the share the parent
gives a session with `threads=`). Asking for nothing never means "the whole machine": a
3-atom tblite gradient ran in 1.2 ms on one thread and 221 ms (10 CPU-seconds) with 46.
A list's calculations are admitted in input order while their shares sum to at most one; a calculation's share is
the larger of its share of the threads and its share of the GPUs, where each of k
calculations on one GPU counts as 1/k of it. Host memory and GPU memory follow the share.
There are no memory estimates and no memory scheduling.

**Placement.** A program that may share the process runs in its threads; one that is not
thread-safe takes turns on a lock per program. Worker processes are started for a
program, its GPUs and its threads, and kept for the next call until the runner closes.

## The store

A store files runs by two identities, kept apart:

- *which calculation*: the key, `qcdata.calculation_key(inp)`, the input as given;
- *which code produced the numbers*: the run's identities, `{program: numerics}` for the
  input's program and every subprogram it names, recursively. A program named with
  several specs (g-xTB gradients with GFN2 Hessians, one program) has its distinct
  identities sorted and joined by `;` (`"gfn2:5;gxtb-2.0.1:7"`). `given` results add no
  identities: they are data in the input, so already in the key.

With `compute(..., store=store)` the runner computes the input's identities from the
programs it would run now, looks the key up with them, and returns a stored run only if
it was made with exactly those identities: a success always, a failure only when the
input determines it (kind `input`, `unsupported` or `convergence`, or `program` with such
a cause; never `resource`, `backend`, `environment`, `timeout`, `numerical` or
`internal`). Misses run, and every run is stored with
its identities. `rerun=True` always runs; `tags` go onto every key, cache hits included;
`meta` goes with each run. An output whose record says `warm_start` true depends on
history, not only on its input, and is never stored.

The runner needs four methods of a store (`qccompute.runner.Store`): `key(inp)`,
`get(key, identities)`, `put(output, identities, tags=, meta=)` and `tag(key, *tags)`.

## Workers

`python -m qccompute.serve PROGRAM` hosts one program's calculations and sessions in its
own process, so a crash ends the worker and not its caller, and a program keeps its state
between calls. `qccompute.worker.Worker` is its client.

- **Transport:** stdin and stdout, one JSON message per line, one client per connection;
  `--socket PATH` serves a Unix socket on the same framing.
- **Hello:** the first line, `{"protocol": 1, "program", "version", "capabilities",
  "extras"}`.
- **Envelope:** every message has an `id`; replies are `{"id", "ok": true, "result"}` or
  `{"id", "ok": false, "error"}`. `ok: false` is for the message itself; a failed
  calculation is a result (a failed output or frame).
- **Ops:** `open`, `compute` (a session's request, or a one-off `input`), `compute_many`,
  `compute_batch`, `reset`, `plan`, `close`, `lineages`. End of input closes every session.
- **Context on the wire:** `{threads, memory_gib, device_memory_gib, timeout,
  scratch_dir}`; the GPUs are fixed when the worker starts.
- **Numbers:** shortest round-trip decimals, so positions and results cross bit for bit;
  no NaN or infinity on the wire.
- **Hygiene:** stdout carries protocol lines only; the worker points file descriptor 1
  at stderr, because native libraries print.
- **Crashes:** a dead worker is kind `environment`, code `worker_died`, with the exit
  status and the end of its stderr. After a GPU fault that poisons the device context
  (code `device_lost`) the worker answers, then exits 3. A runner kills a worker whose
  calculation exceeds its timeout.

## External programs

qccodec registers each external program as a program (`qccodec.CodecProgram`). An adapter
encodes the input, runs the command in the context's scratch directory with its output
captured and the context's deadline, decodes what it wrote, and raises `ProgramError`
with the failed output and the logs when it fails.

- An input that requests threads (ORCA `pal`, Psi4 `-n`, CREST `threads`) runs with its
  request; one that requests none runs with the grant, through `OMP_NUM_THREADS` and the
  other numerical libraries' variables. A GPU program sees its granted GPUs in
  `CUDA_VISIBLE_DEVICES`.
- Codecs that call a Python library take over the process's output streams and working
  directory while they run, so they are `in_process=False` and run in workers.
- geomeTRIC and Sella run their gradient subprograms through `ctx.compute`.
- An external program's `numerics` is the sha256 of the executable that would run; a
  library's, its version.
- A `qcdata.FileInput` runs under the `files` program and returns every file the command
  wrote.

## Cost

On one core, `qccompute.compute` adds about 4 µs to a direct call of a `FunctionProgram`
and about 9 µs for the toy program (finding it, planning it, its context, recording the
execution); the runner hands its plan on in `ctx.plan`, so a program need not plan twice. Through qccodec, a trivial external
command takes about 0.26 ms more than launching it in a fresh directory by hand. A worker
round trip for the toy program costs about 0.14 ms (JSON on a pipe), and starting a
worker about 0.15 s plus the program's own imports, once.

## Conformance

`qccompute.testing.check_program(program, inputs)` runs a program's own small inputs and
checks the contract: schemas that validate, plans that reproduce themselves from
`resolved.settings`, unknown keys refused with kind `input` by `plan` and `compute`, the
thread grant honoured, standard fields in atomic-unit shapes, energy components that add
up, and for sessions and batches, agreement with plain computes, refusal of other
calctypes, and a session that survives a failed request.
