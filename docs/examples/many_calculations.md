# Many calculations at once

`compute` takes a single input or a list of them. A list returns a list of outputs in
input order, and every other argument applies to each input.

```python
from qccompute import compute

outputs = compute([inp1, inp2, inp3])
```

## What runs in parallel

- **Programs** (ORCA, xtb, Psi4, TeraChem, DFTB+, CREST, ...) run side by side. Each
  runs in its own directory with its own captured output, exactly as a single
  calculation does.
- **Python calculations** (machine-learned potentials, tblite, geomeTRIC) take over a
  process's output streams, working directory, and environment while they run. When a
  list holds two or more, they run side by side in worker processes: qccompute starts
  them with `python -m qccompute._worker`, so your script is never re-run and needs no
  `if __name__ == "__main__":` guard. A worker starts with `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` set to its share of the cores (unless
  you set them), so the workers' numerical libraries do not each start a thread per
  core. A worker that needs a GPU is started on its own device.
- Python calculations whose codec exists only in your process and cannot be pickled
  (for example one built with lambdas and registered with `qccodec.register`), single
  Python inputs, and lists computed inside a calculation run in this process, one at a
  time.

## Cores and GPUs

`max_workers` is a budget of CPU threads: by default, the cores this process may use.
Each calculation draws the threads its codec declares from what the input asks for
(ORCA's `pal`, Psi4's `-n`, CREST's `threads`, xtb's `parallel`), and one otherwise. So
`pal 8` ORCA runs on a 32-core machine run four at a time, with no extra setting.
Calculations start in input order; one that asks for more threads than the budget runs
alone.

GPU programs (TeraChem) get their own devices: each calculation runs with
`CUDA_VISIBLE_DEVICES` set to its GPUs, as many at once as there are GPUs. The GPUs are
those in `CUDA_VISIBLE_DEVICES` if it is set (in `environment` or the process), or else
those `nvidia-smi` lists. If none are found, GPU calculations run one at a time without
pinning.

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

Four geomeTRIC optimizations of a six-water cluster with xtb gradients (117 steps each)
took 49.7 s one after another and 7.3 s in four workers on a 16-core laptop. Much of
that is the thread limit: numpy's BLAS starts a thread per core by default, which slows
geomeTRIC's small matrices even in a single process, where `OMP_NUM_THREADS=1` alone
brought the four optimizations from 49.7 s to 18.5 s.

## Following progress

`print_logs=True` prints every line of output prefixed with its input's index, such as
`[3]`. An `update_func` receives `(index, all_logs, new_logs)` and is called from the
threads running the calculations, so it must be thread-safe.

## Interrupting

Ctrl-C stops the list: queued calculations are dropped and running programs are killed
together with every process they started.
