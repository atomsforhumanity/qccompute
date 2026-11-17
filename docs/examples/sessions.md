# Sessions: lineages, seeds and commits

A session evaluates one molecule at many geometries: the steps of an optimization, the
branches of an IRC, the displaced points of a finite-difference Hessian. Open one with
`qccompute.session(inp)` (or `runner.session(inp)`); the input pins the program, model,
keywords, atoms, charge and multiplicity, and its geometry and calctype are the defaults
of a request.

## Who gains from a session

Only a program with sessions of its own (`capabilities().sessions`, like the toy program
here) keeps anything between requests: orbitals for a warm start, a
pinned grid. Its frames say `record["warm_start"]: true` when a request continued a
history. Any other program (most external programs, tblite, a `FunctionProgram`) gets a
session over `compute`: every request is computed from scratch and its frames say
`warm_start: false`. Such a session is a convenience (one object, the same methods for
every program), not a speed-up.

```python
from qccompute.program import capabilities_of, find_program

capabilities_of(find_program("tblite")).sessions  # False: no warm starts
```

## Lineages, seeds and commits

- **Lineage:** any JSON value naming one history (`"ts"`, `["irc", "forward"]`); `None`
  is the default. A request continues its lineage's last committed success.
- **Seed:** lineages to start from when the request's own lineage has no history yet,
  in order: both IRC branches start from the transition state's history.
- **Commit:** `True` (the default) makes a success the lineage's new history; `False`
  reads the history without moving it (displaced points of a finite difference, a
  trial step).
- **Reset:** `s.reset(lineage)` forgets one lineage's history (after a large jump),
  `s.reset()` all of them.

A failed request raises `qcdata.ProgramError`; the lineage's history is untouched and
the session stays usable, so a caller may retry a smaller step. `compute_many` runs every
request (requests of one lineage in order, different lineages possibly side by side),
then raises `qcdata.ProgramErrors` if any failed, with every frame in `results`.

```python
{!examples/sessions.py!}
```
