# Threads

Every calculation runs with one thread count. This page is the one place that says
where the count comes from, how lists and sessions share it, and what to set when you
want many calculations at once.

## The count

- A calculation's thread count is its `threads` keyword. Without one, it is its
  caller's whole count:
  - at top level, the runner's: `Runner(threads=...)`, else `OMP_NUM_THREADS` if it is
    set, else the CPUs this process may use (its affinity, capped by any cgroup quota);
  - for a subprogram or a session request, its parent's or its session's count. A
    `threads` keyword never takes more than that.
- A calculation given GPUs and no `threads` gets 1.
- The program uses at most that many threads, and fewer where more don't help: that
  cap is the program's own (qccodec's xtb runs on one thread unless asked). Programs
  don't suggest thread counts to the runner.
- The program reads its count in `ctx.threads`; the output records it in
  `execution.threads`.

## Lists and sessions share it

A list gives each calculation that runs the count divided by how many run, at least 1,
unless it sets `threads`. Store hits don't run, so they don't count. The same split
holds for a session's `compute_many`, and a program that evaluates many inputs at once
(a batch program) gets a list's inputs together in one call, with their threads
together.

```python
outputs = qccompute.compute(inputs)  # 32 inputs on 8 threads: 1 each, 8 at a time
```

## Never more than the count

Whatever holds a count (the runner, a running calculation, a session) never has more
than that count running under it at once. Work beyond it waits its turn, first come
first served.

- **The runner.** Calls from several threads, `submit` and lists all share the runner's
  count. The runner warns once when calculations that each take its whole count start
  waiting for each other: that's the sign to set `threads`.
- **A calculation.** Its subprograms never use more than its count together, however many
  of its own threads call them.
- **A session.** An open session holds nothing. Each request takes the session's count
  while it runs, so requests from several threads take turns.
- **Waiting.** A waiting calculation can be cancelled (`future.cancel()`) or stopped
  (`runner.stop(future)`). Its timeout counts from when it starts, not while it waits.
- **No waiting on yourself.** A thread that is running a calculation or a session
  request and makes a top-level call (`compute`, `submit`, a request of a session the
  caller opened) fails at once with `RuntimeError`: it would wait for threads it holds
  itself. A program runs its subprograms through `ctx.compute` and `ctx.session`,
  which share its count.

## Math libraries

While a calculation runs, the runner holds the process's BLAS, OpenMP and PyTorch
thread pools at its count. In one process, OpenBLAS's count is process-wide, so
calculations running at once with different counts share it.

A calculation whose input names subprograms holds its own pools at one thread: its
count belongs to its children. An idle OpenBLAS thread spins for about 0.1 s after
each call, so a parent's pools at its full count would keep cores busy beside its
children. A parent with heavy linear algebra of its own may raise its pools up to its
count while none of its children runs, and must lower them again before the next one
starts.

## Memory and GPUs

- **Memory.** A top-level calculation gets its share of the runner's memory: its threads'
  share or its GPUs' share, whichever is larger. A subprogram gets its share of its
  parent's. A `memory_gib` keyword replaces the share, at most what the caller has. qc*
  never schedules memory.
- **GPUs** are whole: each GPU goes to one calculation at a time, and a calculation gets
  as many as its plan asks for. Machine-learned potentials get their throughput from
  batches, many inputs in one call on one GPU.

## Recipes

**One calculation.** Set nothing: it gets the runner's whole count.

**Many calculations at once.** Run them as a list, or set `threads` on each input,
usually 1, and submit them:

```python
one = {**inp.keywords, "threads": 1}
futures = [runner.submit(i.model_copy(update={"keywords": one})) for i in inputs]
```

Submitted calculations without `threads` each take the runner's whole count, so they
run one at a time. The runner warns the first time that happens.

**Under a dispatcher.** A job runner or batch system that reserves CPUs for each task
and sets `OMP_NUM_THREADS` to them (a job runner's `threads_per_task`, or a batch
script's `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`) sets the count of a runner made
without `threads`. There is nothing else to set.

**A program that runs its children side by side.** Split your count. Give each of k
children `threads=ctx.threads // k` (in its keywords, or as a session's `threads=`), or
hand them over as one list, `ctx.compute(children)`, which splits it for you.
Children you don't split take your whole count each, and wait for each other.

**The same bits on every machine.** A program's numbers can depend on its thread count:
multithreaded linear algebra adds partial sums in a different order. Without `threads`,
the count depends on the machine and on the list a calculation came in. Set `threads`
where results must match bit for bit across runs.
