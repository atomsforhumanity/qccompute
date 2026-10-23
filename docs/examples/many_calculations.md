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
- **Machine-learned potentials** (AIMNet2, TorchANI, MACE, Orb, FAIR-Chem) evaluate
  many structures in one pass. Inputs their codec can batch run together in this
  process; see [Batches](#batches).
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

## Batches

A codec that can batch (`Codec(encode_batch=..., batch_key=...)`, qccodec 0.19.0)
says which inputs may share a pass: for the machine-learned potentials, energy and
gradient inputs with the same program, model, and keywords. Such inputs in a list are
evaluated in batches of up to `codec.max_batch` (32 by default), each one call of the
model, and each value is decoded as if its input had run alone.

```python
from qcdata import ProgramInput

inputs = [
    ProgramInput(
        structure=s, calctype="gradient", model={"method": "aimnet2"}, program="aimnet2"
    )
    for s in structures
]
outputs = compute(inputs)  # one model call for up to 32 structures
```

On a laptop GPU (RTX 3050 Ti), 40 water gradients took 0.41 s one at a time and
0.05 s as a list with AIMNet2, 0.58 s and 0.04 s with TorchANI, and 1.22 s and
0.24 s with MACE.

- Batches run in this process, after the programs of the list, so a loaded model stays
  cached for the next `compute` call, as in iterative methods such as NEB.
- Each output's `execution.extras` records `batch_size` and `batch_index`; its
  `wall_time` and `logs` are the batch's.
- A batch that runs out of GPU memory is split in half until it fits. A batch that
  fails otherwise is computed again one input at a time, so each failure is reported
  with its own input. A value that fails to decode fails only its input.
- Hessians, inputs whose key is unique, and lists with `collect_files` or
  `collect_wfn` (a batch shares one directory) run as usual.
- Results match single calculations to the model's precision; models computing in
  float32 or TF32 on a GPU (Orb's default `float32-high`) differ in the last digits.

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
