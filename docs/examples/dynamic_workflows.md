# Dynamic workflows: one future per calculation

Some workflows decide their next calculation from the last one. Per reaction: search
for the transition state, run the IRC from the saddle it found, then minimize both ends
of the path. With many reactions at once, `compute(list)` stage by stage waits for the
slowest member of every stage (the longest IRC holds up every reaction's minimizations).
`submit` starts one calculation and returns a `concurrent.futures.Future` of its output,
so each reaction moves to its next step as soon as its last one finishes:

```python
from concurrent.futures import FIRST_COMPLETED, wait

import qccompute
from qcdata import ProgramError, ProgramInput

gxtb = {"program": "axtb", "model": {"method": "gxtb-v2.0.1"}}
children = [{**gxtb, "calctype": "gradient"}, {**gxtb, "calctype": "hessian"}]


def qcopt(calctype, structure, **fields):
    return ProgramInput(
        program="qcopt", calctype=calctype, structure=structure,
        subprograms=children, **fields,
    )


with qccompute.Runner(threads=8, isolation="process") as runner:
    futures = {
        runner.submit(qcopt("transition_state", guess, keywords={"freq": {"check": True}}),
                      store=store): ("ts", name)
        for name, guess in guesses.items()
    }
    results = {name: {} for name in guesses}
    while futures:
        done, _ = wait(futures, return_when=FIRST_COMPLETED)
        for future in done:
            step, name = futures.pop(future)
            try:
                out = future.result()
            except ProgramError as e:   # a failed step: keep its output, stop the reaction
                results[name][step] = e.output
                continue
            results[name][step] = out
            if step == "ts":            # the IRC from the saddle, with its Hessian
                last = out.results.frames[-1]
                irc = qcopt("irc", last.structure, given=last.results)
                futures[runner.submit(irc, store=store)] = ("irc", name)
            elif step == "irc":         # both ends, minimized
                for side, frame in (("reactant", out.results.frames[0]),
                                    ("product", out.results.frames[-1])):
                    end = qcopt("optimization", frame.structure)
                    futures[runner.submit(end, store=store)] = (side, name)
```

- **The runner packs submitted calculations** into its threads and GPUs as it packs a
  list, admitted in the order submitted: on 8 threads, eight one-thread searches run at
  once and the ninth starts when one ends. A stored result resolves its future before
  `submit` returns, so a rerun of this workflow from a store computes nothing: it asks
  for the same inputs again.
- **`future.result()`** is the output, or raises `qcdata.ProgramError` with the failed
  output as `e.output`, as `compute` does. `concurrent.futures.as_completed` and
  `future.add_done_callback` work as for any future.
- **Priorities and cancelling.** `runner.submit(inp, priority=2)` puts a calculation
  ahead of every waiting one of lower priority (default 0); within a priority, the first
  submitted starts first, and running calculations are never displaced. So a workflow
  can keep submitting what it finds and still have its most promising work start
  first. `future.cancel()` works while the calculation runs, too: a worker running it
  is killed with every process it started (an external program's included), and one
  running in this process stops at its next look at `ctx.timeout`, which reads 0, as
  for a timeout. A cancelled calculation is never stored; `future.result()` raises
  `CancelledError`.
- **Programs that are not thread-safe**, qcopt among them, take turns in one process:
  submitted from threads or as futures, they would run one at a time. The runner warns
  the first time a calculation waits for its turn; `isolation="process"`, as above,
  runs each in a worker process, side by side. On ten reactions of 12 to 30 atoms
  (g-xTB, 8 cores), threads each chaining `compute` took 55.8 s, the three stage lists
  28.0 s, and the same chains on `Runner(threads=8, isolation="process")` 14.8 s.
- **Rerunning is not resuming.** A rerun is all cache hits only when the workflow, given
  the same results, asks for exactly the same inputs: the key is the input as given, so
  a geometry, seed or keyword that differs is a miss. A workflow whose choices depend on
  timing (which future finished first, a reference structure replaced when a lower one
  arrives, priorities that read what has been found so far) takes other branches when
  rerun. The store still serves every calculation it asks for again, but it does not
  hold the workflow's decisions: what was searched, what was set aside, where it was
  heading. Such a workflow keeps that decision state itself and resumes from it.
- **Live progress:** `submit(inp, progress=functools.partial(show, name))` hands each
  calculation's events to a callable that knows which reaction it is (see
  [Progress](../index.md#progress)).
