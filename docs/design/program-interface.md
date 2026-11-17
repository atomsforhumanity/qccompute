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
| `capabilities()` | calctypes, methods with their calctypes, `sessions`, `batches`, `in_process`, `thread_safe`; a `Capabilities` or a plain mapping |
| `settings_schema(calctype)` | the JSON Schema of `keywords` and of `resolved.settings` |
| `record_schema(calctype)` | the JSON Schema of `ProgramOutput.record` |
| `plan(inp)` | a dry run: validates (kind `input` with a did-you-mean for unknown keys), resolves every `"auto"`, and says what the calculation needs (`Resources`: threads, GPUs) |
| `compute(inp, ctx=None)` | runs the calculation; returns the successful output or raises `ProgramError` |
| `numerics(spec)` (optional) | a string that changes whenever results for an unchanged input may change; the version by default |
| `open_session(inp, ctx=None, calctypes=None)` (optional) | a session, usually the program's own session class |
| `compute_batch(inputs, ctx=None)` (optional) | many inputs at once; one output per input, in order, failures in place |

A `qcdata.FileInput` that is not a `ProgramInput` (a command with its files, whose
`program` is the command) runs under the program registered as `files`, the one name the
protocol reserves; qccodec registers it, and its numerics identity is the run's identity
(`{"files": ...}`).

What a program returns for `capabilities` and `plan` may be a plain mapping, which
qccompute validates (`capabilities_of`, `plan_of`): it is the same document a program's
command line prints and its worker sends. `numerics` takes the spec it would run (the
input, or a subprogram's `ProgramSpec`), so a program of several methods can give each
its own identity: a numerics epoch, an executable's sha256, a model's checksum.

`qccompute.program.FunctionProgram(name, energy_gradient, hessian=None, *, version=None,
takes_input=False, settings=None)` is the smallest complete program: functions of the
positions, in process and thread-safe, taking the shared `threads` and `accuracy`. With
`takes_input=True` the functions also get the input (structure, model, keywords), and
`settings=` declares the JSON Schema of the keywords they read, so one program stands
in for an engine on any molecule. Its numerics identity is a digest of what the
functions compute with (code, constants, defaults, closures, the globals they read),
unless `version` names it.

## The context

`compute`'s second argument is a `RunContext`, which only a runner passes, and a runner
always passes one. It carries the calculation's share of the machine and the way to run
subprograms:

- `threads`, `gpus`: the grant. The threads are what the input asks for (its `threads`
  keyword, else `plan.resources.threads`: 1 unless the program says more, and None for
  all there is), at most the runner's (or, for a subprogram, the parent's grant). A program uses the
  smaller of its own request and the grant, or the grant when it requested nothing; a
  capped request is a `resource_capped` warning. The GPUs are as many as its plan asks
  for (`plan.resources.gpus`), of the runner's or the parent's: a program whose
  subprograms use GPUs plans theirs and passes them on.
- `device_ordinals()`: the granted GPUs as this process's CUDA device ordinals, so every
  program maps a grant alike. `gpus` names them as `CUDA_VISIBLE_DEVICES` does (indices
  or `GPU-`/`MIG-` UUIDs). When the process's `CUDA_VISIBLE_DEVICES` is set, a GPU's
  ordinal is its position there: a worker is started with `CUDA_VISIBLE_DEVICES` set to
  its grant, so its GPUs are 0, 1, ...; in the runner's process GPU `"2"` of
  `CUDA_VISIBLE_DEVICES=1,2` is ordinal 1. When it is unset, an index is its own ordinal
  and a UUID is the index `nvidia-smi` gives it. `nvidia-smi` numbers GPUs in PCI bus
  order and CUDA by default may not, so a runner that finds its GPUs with `nvidia-smi`
  sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` (unless it is set) when it is made, for its
  process and its workers; a process that starts CUDA before it makes a runner sets it
  itself. A dispatcher's `CUDA_VISIBLE_DEVICES` is never changed. A granted GPU the
  process cannot see fails with kind `environment`.
- `memory_gib`, `device_memory_gib`: host and GPU memory (GiB), as the runner grants it
  (below, **Memory**).
- `timeout`: the seconds left. A program that starts processes kills them when it runs
  out and fails with kind `timeout`; one that loops checks it. The runner kills a
  program in a worker process that runs out, but cannot stop one in its own process
  (Python cannot stop a thread): there the limit is the program's to honour, and the
  runner starts none of its subprograms once it has run out (`ctx.compute`,
  `ctx.session` and its child sessions' requests fail with kind `timeout`), so a driver
  stops at its next step. For a hard limit on any program, `isolation="process"`.
- `plan`: the runner's plan of the input, so the program need not plan again.
- `progress`: where the program reports what it is doing while it runs, when the
  caller listens (below, **Progress**); None when nobody does, so a program checks it
  before building an event.
- `scratch`: a directory, made on first use and removed when the call returns.
- `compute(inputs, raise_exc=True)` and `session(spec, structure, threads=, calctypes=)`:
  subprograms, within the grant. They run only programs the calculation's input names,
  its own and its subprograms' recursively, and every program a subprogram's input
  names must be among them; another is refused (kind `input`, code
  `unnamed_subprogram`), so a run's identities cover every program it used. A program
  that decides at run time which programs to run (a path method that minimizes its
  ends) names each in its input's subprograms.

`ctx is None` means a direct call: the program uses its own settings. A program with
subprograms has no route of its own to programs, so called directly it returns
`qccompute.compute(inp)`, and the default runner runs it with a context. The runner finds
programs by name: one that is neither installed nor registered fails with kind
`environment`, and the message says how to register it.

The runner never changes a number by itself. A grant may change a program's resolved
choices (a storage scheme picked from the memory granted, a precision from the device);
the program records each one in `resolved`.

## Progress

`submit`, `compute`, `as_completed` and `session` take `progress=`, a callable that
receives small JSON events while calculations run. Programs choose what to emit (qcopt:
one event per step); the runner only delivers them, and nothing is stored.

- A program calls `ctx.progress(event)` with a JSON object of its own: plain values,
  qcdata's units, small enough to send every step (a step number, an energy, the largest
  gradient component; positions where a live view needs them). A shared key: `event`,
  a snake_case name of what happened (`step`).
- The caller receives the event with `path` added: the program names from the
  caller's input down to the program that emitted it (`["mepd-neb", "qcopt"]`). A
  subprogram's events reach the same callable through its parent's context, so a
  workflow sees every level of a nested run.
- The callable may be called from several threads at once (a list's calculations, a
  program's own threads), and should return quickly. An exception it raises is a
  warning; the calculation goes on.
- Store hits emit nothing. A list's events do not say which input they are from:
  `submit` each input with a callable of its own (`functools.partial(show, i)`) to
  tell them apart.
- A program in a worker streams its events to the client while a message is in flight
  (`ctx.progress` on the wire, below); a program run without a listener has
  `ctx.progress` None, so an unused progress channel costs nothing.

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
  history as it stands at its place; requests of different lineages may run
  concurrently, within the session's grant, which is set when it opens
  (`session(inp, threads=)`, else what its input or plan asks for): the program splits
  it across lineages. A path method that evaluates 13 images at once opens its session
  with `threads=13`.
- A failed request raises `ProgramError` with its failed frame; the session and its other
  lineages go on, and the lineage's history is untouched.
- Sessions may be called from several threads; the runner serializes those of programs
  that are not thread-safe.

`qccompute.session(inp)` hands back the program's own session in process, so the numbers
are exactly those of the program used directly. A program in a worker gets a
`WorkerSession` over the pipe, with the same methods. A program without sessions gets a
`ComputeSession`: one stateless calculation per request (`warm_start` false), placed
like any other calculation (in this process, or in the program's workers). Its
`compute_many` runs as a list does: each request gets the threads it asks for, and they
run side by side within the sum of their threads, at most the session's `threads=` or
else the runner's (or the parent's grant). Thirteen one-thread path images on eight
threads run eight at a time.

## The runner

`Runner(threads=None, gpus="auto", backend="local", isolation="auto", per_gpu=1)` with
`submit` (one input; returns a `concurrent.futures.Future` of its output), `compute`
(one input or a list), `as_completed` and `session`. Every call takes the same options:
`store`, `timeout`, `rerun`, `tags`, `meta`, `raise_exc` (not `submit`: its future
raises) and `progress`. `qccompute.compute`, `qccompute.submit` and `qccompute.session`
are the methods of the default runner: this machine (every core this process may use,
every visible GPU), made on first use, never reconfigured.

`submit` is for workflows whose next calculation depends on the last (per reaction, a
transition-state search, the IRC from its saddle, then both ends), many at once with no
stage barriers. A stored output, or an input its program refuses, resolves the future
before `submit` returns; the others are planned at once and run in the runner's threads.
A runner's submitted calculations are packed into its threads and GPUs as a list's are,
admitted in the order submitted; `compute` calls take their own share, as before.
Cancelling a future before it starts skips its calculation.

Every input goes one path: find the program, plan it (an input its program refuses
fails without running), place it (this process, or a worker process for programs with
`in_process=False` and for `isolation="process"`), call `compute` or `compute_batch`
with a context, record the execution.

**Execution.** The runner sets each output's `execution`: `wall_time` (the call's),
`identities`, `threads` when the program left it unset (the threads granted), and the
environment: `hostname`, `host_cpu`, `host_mem_gib`, `cpu_model` (as the operating
system names the CPU), `libc` (`glibc 2.39`) and `devices` (the granted GPUs' names, as
`nvidia-smi` gives them), each read once per process and filled only where the program
left it unset, so a program that knows better (one that ran elsewhere) sets its own.
The outputs of a batch ran together: each keeps a `wall_time` its program set (a
lockstep batch knows when each task finished), else gets the batch's, and the batch's
wall time is always `timings["batch"]`. One whose program left `threads` unset gets its
own input's part of the batch's grant: the threads it asked for, scaled down when the
batch was capped (`max(1, asked × granted // total asked)`). So `wall_time × threads`
adds up across a batch's outputs, and a store's runs, to what the batch cost, to
rounding.

**Resources.** qc* never allocates machines, queues work or enforces limits: a
dispatcher (a batch system, Kubernetes) gives the process its cores, memory and GPUs. A
runner shares them out. Each calculation gets the threads it asks for (its `threads`
keyword, else its plan's: 1 unless the program says more), at most the runner's; a
session likewise, and a subprogram at most its parent's grant (or the share the parent
gives a session with `threads=`). A program that scales with whatever it is given (a DFT
code) plans `threads: None`: all of the runner's threads (all of the parent's grant for a
subprogram), and in a list a full share, so it runs alone. The input's `threads` keyword
still wins. Asking for nothing never means "the whole machine": a 3-atom tblite gradient
ran in 1.2 ms on one thread and 221 ms (10 CPU-seconds) with 46. A list's calculations
are admitted in input order while their shares sum to at most one; a calculation's share is
the larger of its share of the threads and its share of the GPUs, where each of k
calculations on one GPU counts as 1/k of it. The inputs of a list that go to a program
with batches are one calculation of the list: it asks for the sum of their threads, at
most the runner's (or the parent's grant), and for the most GPUs any of them asks for;
the program divides its grant among them.

**Memory.** There are no memory estimates and no memory scheduling: a plan's
`memory_gib` and `device_memory_gib` are informational, recorded and never scheduled.
What the runner grants, in `ctx`:

- *Host memory.* A calculation run alone (one input, a session) gets all of the
  runner's memory; a subprogram run alone, all of its parent's grant (a session opened
  with `threads=` gets the part of it those threads are of the parent's). In a list,
  each calculation, and each batch, gets its share of the list's memory: the runner's,
  or the parent's grant.
- *GPU memory.* 1/k of each granted GPU with `per_gpu=k`, at most a subprogram's
  parent's GPU grant.
- *Keywords.* The input's `memory_gib` or `device_memory_gib` replaces either: at most
  the runner's memory or the GPU's at the top level, at most the parent's grant for a
  subprogram, since a parent's grant holds its children. A batch's single `ctx` gets the
  shares, not its inputs' keywords. Like `threads`, these resource keywords are what a
  calculation asks for, and they are left out of its key.

**Placement.** A program that may share the process runs in its threads; one that is not
thread-safe takes turns on a lock per program. The first time one of its calculations
waits for its turn (threads calling `compute`, futures, a list), the runner warns once
(`RuntimeWarning`) that they run one at a time and that `isolation="process"` runs them
side by side. Worker processes are kept for the next
call until the runner closes, in pools. A worker sees exactly the GPUs granted to it
(`CUDA_VISIBLE_DEVICES`, empty for none). Workers are pooled by program and GPUs,
whatever the threads of the calls they serve, because a program's state (a model
loaded, a GPU context) is costly to repeat: a list's batches of different sizes, and
calls with different `threads`, share the program's workers rather than starting one
each. Each call's threads reach the program in `ctx.threads`, which it applies to that
call (a subprocess's `OMP_NUM_THREADS`, `torch.set_num_threads`); a worker's numerical
libraries start with the threads of the call that started it (`OMP_NUM_THREADS` and the
others, unless set), and a library that reads them only once keeps that count. On GPUs
at most `per_gpu` workers serve every calculation and session of a program on those
GPUs, whose work queues on the device anyway; without GPUs, at most as many as fit side
by side in the runner's threads at a call's threads. A call or session takes an idle
worker of its pool, else starts one while the pool has room, else shares the worker with
the fewest users, taking turns on it. A worker that dies or is killed for a timeout ends
the sessions it hosts.

## The store

A store files runs by two identities, kept apart:

- *which calculation*: the key, `qcdata.calculation_key(inp)`, the input as given;
- *which code produced the numbers*: the run's identities, `{program: numerics}` for the
  input's program and every subprogram it names, recursively. A calculation runs no
  program its input does not name (`ctx` refuses it), so these are every program
  that made the numbers. A program named with
  several specs (g-xTB gradients with GFN2 Hessians, one program) has its distinct
  identities sorted and joined by `;` (`"gfn2:5;gxtb-2.0.1:7"`). `given` results add no
  identities: they are data in the input, so already in the key.

With `compute(..., store=store)` the runner computes the input's identities from the
programs it would run now, looks the key up with them, and returns a stored run only if
it was made with exactly those identities: a success always, a failure only when the
input determines it (kind `input`, `unsupported` or `convergence`, or `program` with such
a cause; never `resource`, `backend`, `environment`, `timeout`, `numerical` or
`internal`). Misses run, and every run is stored with
its identities. Every output the runner returns carries the identities of the programs
that made it in `execution.identities`, with or without a store, so an output read back
says which code made it; a stored run keeps the ones it was made with. `rerun=True`
always runs; `tags` go onto every key, cache hits included;
`meta` goes with each run. An output whose record says `warm_start` true depends on
history, not only on its input, and is never stored. Every output a runner returns with
a store names its run, `execution.store_run = "<key>/<run>"`: a hit as the store marks
it, a fresh run once it is stored. So a result is cited exactly, and read back later
with `qcstore.Store.get(key, run=...)` even after a newer run became the preferred one.
Concurrent misses of one key each compute, and the store keeps every run.

The runner needs four methods of a store (`qccompute.runner.Store`): `key(inp)`,
`get(key, identities)`, `put(output, tags=, meta=)` (which files the run under
`output.execution.identities`) and `tag(key, *tags)`.

## Workers

`python -m qccompute.serve PROGRAM` hosts one program's calculations and sessions in its
own process, so a crash ends the worker and not its caller, and a program keeps its state
between calls. `qccompute.worker.Worker` is its client.

- **Transport:** stdin and stdout, one JSON message per line, one client per connection;
  `--socket PATH` serves a Unix socket on the same framing.
- **Hello:** the first line, `{"protocol": 1, "program", "version", "capabilities",
  "array_files": true, "calls": true}`.
- **Envelope:** every message has an `id`; replies are `{"id", "ok": true, "result"}` or
  `{"id", "ok": false, "error"}`. `ok: false` is for the message itself; a failed
  calculation is a result (a failed output or frame).
- **Ops:** `open`, `compute` (a session's request, or a one-off `input`), `compute_many`,
  `compute_batch`, `reset`, `plan`, `close`, `lineages`, `options`. End of input closes
  every session.
- **Options:** a client turns on what the hello offers with `options {array_dir,
  calls}`, per connection; a plain JSON-lines client sends nothing and gets neither.
- **Large arrays:** with `array_dir` (workers are local), an array field over 1 MiB, in
  a message either way, crosses as a `.npy` file in that directory, with qcdata's pointer
  `{path, shape, dtype, sha256}` in its place, as in a saved model; the reader checks it
  and deletes the file. A 1200 x 1200 Hessian's output crosses in 20 ms instead of
  0.85 s and a 28 MB line.
- **Subprograms run in the client:** with `calls: true`, the worker's program's
  `ctx.compute` and `ctx.session` come back to the client as call lines,
  `{"call", "op", ...}` with the ops above, which the client answers in any order
  (`{"call", "ok", "result"}`) while the message is in flight. So a subprogram of a
  program in a worker (geomeTRIC's gradients) is any program the client finds,
  registered ones included, run within the client's grant, and a nested worker is
  started by the client, never by a worker. Without it, subprograms run in the
  worker's own default runner.
- **Context on the wire:** `{threads, memory_gib, device_memory_gib, timeout,
  scratch_dir, progress}`; the GPUs are fixed when the worker starts. With
  `progress: true` (the client's caller listens) the program's progress events stream
  back as lines `{"id", "progress": event}` naming the message in flight (for a
  session, its request's message); the client hands each to the caller's callable.
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
- geomeTRIC and Sella run their gradient subprograms through `ctx.compute`, which from a
  worker comes back to the runner's process.
- An external program's `numerics` is the sha256 of the executable that would run; a
  library's, its version.
- A `qcdata.FileInput` runs under the `files` program and returns every file the command
  wrote.

## Cost

On one core, `qccompute.compute` adds about 7 µs to a direct call of a `FunctionProgram`
with a `version` and about 10 µs for the toy program (finding it, planning it, its
context, recording the execution and the identities); a `FunctionProgram` without a
`version` adds about 6 µs more, the digest of its functions that is its identity. The
runner hands its plan on in `ctx.plan`, so a program need not plan twice. Through qccodec, a trivial external
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
