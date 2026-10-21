# Many calculations at once

`compute` takes a single input or a list of them. A list returns a list of outputs in
input order, and every other argument applies to each input.

```python
from qccompute import compute

outputs = compute([inp1, inp2, inp3])
```

## What runs in parallel

- **Programs** (ORCA, xtb, Psi4, TeraChem, DFTB+, CREST, ...) run side by side, up to
  `max_workers` at once: by default, one per CPU core this process may use. Each runs
  in its own directory with its own captured output, exactly as a single calculation
  does.
- **Python calculations** (machine-learned potentials, tblite, geomeTRIC) take over
  the process's output streams, working directory, and environment while they run,
  so they run one at a time, after the programs.
- A list computed from inside a Python calculation (for example in a geomeTRIC child)
  runs one input at a time.

Most programs use one core per calculation unless you ask for more (ORCA's `pal`,
Psi4's `-n`). If your inputs use several cores each, lower `max_workers` to match.

## Failures

Every calculation runs first. Then, with `raise_exc=True` (the default), one
`ComputeErrors` is raised if any failed. It is an `ExceptionGroup`: each sub-exception is
the error that calculation would have raised on its own, with its `prog_output` and its
position in the list as `input_index`, and `outputs` holds every output.

```python
from qccompute.exceptions import ComputeErrors, ProgramTimeoutError

try:
    outputs = compute(inputs, timeout=600)
except* ProgramTimeoutError as timeouts:
    print(f"{len(timeouts.exceptions)} calculations timed out")
```

With `raise_exc=False` the list is returned as is; check each `output.success`.

## Following progress

`print_logs=True` prints every line of output prefixed with its input's index, such as
`[3]`. An `update_func` receives `(index, all_logs, new_logs)` and is called from the
threads running the calculations, so it must be thread-safe.

## Interrupting

Ctrl-C stops the list: queued calculations are dropped and running programs are killed
together with every process they started.
