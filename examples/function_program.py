"""Make a program of two Python functions and run it like any other program.

A Morse potential between two atoms: the smallest complete program. The runner gives
it lists, sessions and the store, exactly as for linus or an external program.
"""

import numpy as np
from qcdata import CalcType, ProgramInput, Structure

import qccompute
from qccompute.program import FunctionProgram

D, A, R0 = 0.17, 1.03, 1.40  # well depth (Eh), width (1/bohr), bond length (bohr)


def morse(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Energy (Eh) and gradient (Eh/bohr) of a Morse bond at positions x (bohr)."""
    d = x[1] - x[0]
    r = float(np.linalg.norm(d))
    e = np.exp(-A * (r - R0))
    de_dr = 2 * D * A * e * (1 - e)
    return D * (1 - e) ** 2, np.array([-de_dr * d / r, de_dr * d / r])


qccompute.register_program(FunctionProgram("morse", morse))

h2 = Structure(
    symbols=["H", "H"], geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]])
)
inp = ProgramInput(program="morse", calctype=CalcType.gradient, structure=h2)

out = qccompute.compute(inp)
print(out.results.energy, out.results.gradient[1])

# A scan along the bond, side by side
scan = [
    inp.model_copy(update={"structure": h2.model_copy(update={"geometry": g})})
    for g in (np.array([[0, 0, 0], [0, 0, r]]) for r in np.linspace(1.0, 3.0, 9))
]
energies = [o.results.energy for o in qccompute.compute(scan)]
print(np.round(energies, 4))
