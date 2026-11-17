# Migrating from qccompute 0.22

qccompute 0.23 runs programs through one protocol. It no longer depends on qccodec: the
external programs qccodec supports are programs that qccodec registers, installed with
`pip install qccompute[qccodec]`. There are no compatibility shims; this page lists every
change.

## Installing

| 0.22 | 0.23 |
|---|---|
| `pip install qccompute` (installed qccodec) | `pip install qccompute[qccodec]` for external programs |
| `pip install qccompute[geometric]`, `[tcpb]`, `[all]` | `pip install qccompute[qccodec] qccodec[geometric]` (or `[tcpb]`, `[sella]`) |

## Running one calculation

```python
# 0.22
out = qccompute.compute(inp, collect_files=True, print_logs=True, raise_exc=True)

# 0.23
out = qccompute.compute(inp)
```

`compute` takes `store`, `timeout`, `rerun`, `tags`, `meta` and `raise_exc`. The
removed options:

| 0.22 option | 0.23 |
|---|---|
| `codec=` | register a program: `qccompute.register_program(qccodec.CodecProgram(codec))` |
| `executables={"orca": path}`, `QCCOMPUTE_ORCA_EXECUTABLE` | `QCCODEC_ORCA_EXECUTABLE`, or `CodecProgram(codec, executable=path)` registered |
| `environment=` | `CodecProgram(codec, environment={...})` registered, or the process's environment |
| `scratch_dir=`, `rm_scratch_dir=` | gone: each calculation's scratch directory is made on first use and removed |
| `collect_logs=` | gone: external programs' logs are always in `output.logs` |
| `collect_files=` | gone for `ProgramInput`; a `FileInput` returns every file the command wrote |
| `collect_wfn=`, `propagate_wfn=` | gone; a program's own session keeps its warm start between steps |
| `options={"collect_rotamers": True}` (CREST) | the keyword `{"output": {"rotamers": True}}` |
| `print_logs=`, `update_func=`, `update_interval=` | gone: logs arrive with the output |
| `max_workers=` | `qccompute.Runner(threads=...)` |
| `compute_args(...)` | build a `ProgramInput` |

## Failures

0.22 raised its own exception classes (`InputError`, `ExternalProgramError`,
`ProgramNotFoundError`, `ProgramTimeoutError`, ... from `qccompute.exceptions`) with the
output as `prog_output`. 0.23 raises `qcdata.ProgramError` with the output as `output`,
and the kind of failure in `error.kind`:

| 0.22 | 0.23: `ProgramError` with `error.kind` |
|---|---|
| `InputError` | `input` (or `unsupported` for a calctype the program does not compute) |
| `ProgramNotFoundError`, `CodecNotFoundError` | `environment` |
| `ProgramTimeoutError` | `timeout` |
| `ExternalProgramError` | `internal`, or the program's own kind |
| `e.prog_output` | `e.output` |
| `ComputeErrors` from `qccompute.exceptions`, `e.input_index` | `qccompute.ComputeErrors`; `.outputs` in input order |

```python
# 0.22
try:
    out = compute(inp)
except ExternalProgramError as e:
    logs = e.prog_output.logs

# 0.23
try:
    out = qccompute.compute(inp)
except qcdata.ProgramError as e:
    logs = e.output.logs
```

## Lists

`compute(list)` packs calculations by what each program's plan asks for, within the
runner's threads and GPUs, instead of within `max_workers`. A program that evaluates
many inputs at once gets them together; there is no `batch_size` or `batch_index` in
`execution.extras` beyond what a program records. `Runner.as_completed` yields outputs
as they finish.

## Sessions

`qccompute.session(spec, structure, threads=..., **compute_options)` is
`qccompute.session(inp, calctypes=None)`: build the input with
`ProgramInput.from_spec(spec, structure)`. `compute_many` now runs every request and then
raises `qccompute.program.SessionErrors` when any failed, with every frame in `frames`.

## Programs

| 0.22 | 0.23 |
|---|---|
| `compute(self, inp, ctx)` returned a failed output | `compute(self, inp, ctx=None)` raises `ProgramError` (with `output=`) |
| `local_context(...)` for direct calls of a program with subprograms | with `ctx=None` it returns `qccompute.compute(inp)` |
| `Capabilities.driver` | gone: a program runs subprograms through `ctx.compute` and `ctx.session` |
| `RunContext(compute=..., open_session=..., make_scratch=..., scratch_dir=..., progress=...)` | runners make contexts; `ctx.compute()`, `ctx.session()`, `ctx.scratch`, `ctx.timeout` |
| `compute_batch` | returns failures in place, refused inputs included |
| `qccompute.get_program` | `qccompute.program.find_program` |
| `qccompute.utils.available_programs` | `qccompute.program.available_programs` |
| `check_program(..., children=...)` | register the subprograms; `check_program` also checks the thread grant |

A program may declare `numerics(spec)`, which a store uses to tell runs of different code
apart; without it the version stands in.
