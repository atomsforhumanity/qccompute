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
        keywords={"threads": 2},  # two threads each
    )
    for structure in structures
]
outputs = qccompute.compute(inputs)
```

## How a list runs

1. **Lookup.** With a `store`, every input is looked up first; reusable runs come back
   without running.
2. **Planning.** Each program plans its inputs (`program.plan(inp)`): an input it refuses
   fails at once, without running, and the plan says which GPUs the calculation needs.
   The plan is handed to the program, so nothing is planned twice.
3. **Sharing.** The inputs that run share the runner's thread count: each gets the count
   divided by how many run, at least 1, unless it sets `threads`; they start in input
   order, as threads free up ([Threads](../threads.md)). On 16 threads, eight 2-thread
   xtb calculations run at once.
4. **Placement.** Programs that may share this process run in its threads (one at a time
   for programs that are not thread-safe); programs that cannot (they capture the
   process's output streams, as Python libraries run by qccodec do) run in worker
   processes, which stay alive for the next call. A program that evaluates many inputs at
   once (`compute_batch`, machine-learned potentials, qcopt) gets all of its inputs
   together in one call, with their threads together, and divides them among its
   inputs.

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

Submit each input and take the futures as they complete. Give each `threads` (usually
1), or they run one at a time:

```python
from concurrent.futures import as_completed

with qccompute.Runner(threads=32) as runner:
    one = [i.model_copy(update={"keywords": {**i.keywords, "threads": 1}}) for i in inputs]
    for future in as_completed([runner.submit(i, timeout=600) for i in one]):
        out = future.result()  # raises qcdata.ProgramError for a failure
        print(out.input_data.program, out.results.energy)
```

When the next calculations depend on earlier ones, submit each as its inputs become
known and get a future of its output: see [Dynamic workflows](dynamic_workflows.md).
