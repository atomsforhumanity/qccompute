"""A session: one molecule at many geometries, with lineages, seeds and commits.

The toy program (Lennard-Jones argon, with sessions of its own) stands in for any
program that warm-starts. Its frames say whether they were warm-started and, in its own
record, which lineage's history they continued (``from_lineage``).
"""

import numpy as np
from qcdata import CalcType, Model, ProgramInput, Request, Structure

import qccompute
from qccompute.program import capabilities_of, find_program
from qccompute.testing import toy

qccompute.register_program(toy.program)

argon = Structure(
    symbols=["Ar", "Ar", "Ar"],
    geometry=np.array([[0.0, 0.0, 0.0], [7.2, 0.0, 0.0], [3.0, 6.1, 0.4]]),  # bohr
)
inp = ProgramInput(
    program="toy", calctype=CalcType.gradient, structure=argon, model=Model(method="lj")
)
x = np.asarray(argon.geometry)

# Only a program with sessions of its own warm-starts; any other program's session
# computes every request from scratch (warm_start is always False).
program = find_program("toy")
assert program is not None
print("warm-starts:", capabilities_of(program).sessions)

with qccompute.session(inp) as s:
    first = s.compute()  # the input's own geometry: nothing to continue from yet
    second = s.compute(x * 1.01)  # continues the default lineage's last success
    print(first.record["warm_start"], second.record["warm_start"])  # False True

    # A lineage is any JSON value naming one history. Optimize towards a transition
    # state in lineage "ts", then start both IRC branches from it: `seed` names the
    # lineage to start from when the request's own lineage has no history yet.
    s.compute(x * 1.005, lineage="ts")
    forward = s.compute(x * 1.02, lineage=["irc", "forward"], seed=["ts"])
    backward = s.compute(x * 0.99, lineage=["irc", "backward"], seed=["ts"])
    print(forward.record["from_lineage"], backward.record["from_lineage"])  # ts ts
    later = s.compute(x * 1.03, lineage=["irc", "forward"])  # now its own history
    print(later.record["from_lineage"])  # ['irc', 'forward']

    # Finite differences around the TS: each displaced point starts from the TS's
    # history but, with commit=False, leaves it as it was for the next real step.
    step = 1e-3
    displaced = [
        Request(
            positions=x * 1.005 + sign * step * e.reshape(3, 3),
            lineage="ts",
            commit=False,
        )
        for e in np.eye(9)
        for sign in (1, -1)
    ]
    energies = np.array([f.results.energy for f in s.compute_many(displaced)])
    gradient = (energies[0::2] - energies[1::2]) / (2 * step)
    print(np.round(gradient.reshape(3, 3), 6))

    s.reset("ts")  # forget one lineage's history (after a large jump); reset(): all
