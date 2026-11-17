# Many calculations at once

Pass `compute` a list of inputs to get a list of outputs in the same order. The inputs may
mix programs, methods and calculation types.

```python
import qccompute
import qcdata
from qcdata import CalcType, Model, ProgramInput, Structure

inputs = [
    ProgramInput(
        program="xtb",
        calctype=CalcType.gradient,
        structure=structure,
        model=Model(method="gfn2"),
        keywords={"parallel": 2},  # two threads each
    )
    for structure in structures
]
outputs = qccompute.compute(inputs)
```

## How a list runs

1. **Lookup.** With a `store`, every input is looked up first; reusable runs come back
   without running.
2. **Planning.** Each program plans its inputs (`program.plan(inp)`): an input it refuses
   fails at once, without running, and the plan says what the calculation needs (threads,
   GPUs). The plan is handed to the program, so nothing is planned twice.
3. **Packing.** Calculations are admitted in input order while their shares of the runner
   sum to at most one. A calculation's share is the larger of its share of the threads
   and its share of the GPUs; its host memory is its share of the runner's (unless its
   `memory_gib` keyword asks for an amount). On 16 threads, eight 2-thread xtb
   calculations run at once.
4. **Placement.** Programs that may share this process run in its threads (one at a time
   for programs that are not thread-safe); programs that cannot (they capture the
   process's output streams, as Python libraries run by qccodec do) run in worker
   processes, which stay alive for the next call. A program that evaluates many inputs at
   once (`compute_batch`, machine-learned potentials, qcopt) gets all of its inputs
   together, with the threads they ask for together (at most the runner's), and divides
   them among its inputs: 100 one-thread searches as one batch get 100 threads.

## Failures

Every calculation runs; then, if any failed, `qcdata.ProgramErrors` is raised. It is an
`ExceptionGroup` of `qcdata.ProgramError`s, each with its failed `output`, and its
`results` holds every output in input order:

```python
try:
    outputs = qccompute.compute(inputs)
except qcdata.ProgramErrors as errors:
    outputs = errors.results
    failed = [o for o in outputs if not o.success]

# or handle failures by kind
try:
    outputs = qccompute.compute(inputs)
except* qcdata.ProgramError as group:
    ...
```

`raise_exc=False` returns failed outputs in place.

## As they finish

```python
with qccompute.Runner(threads=32) as runner:
    for out in runner.as_completed(inputs, timeout=600):
        print(out.input_data.program, out.results.energy)
```

Stored outputs come first, then each as it finishes. With `raise_exc=True` (the default)
failures are held back and raised together after the last success.
