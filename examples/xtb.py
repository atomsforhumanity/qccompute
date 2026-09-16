"""Must run script like this: python -m examples.xtb"""

from qcdata import CalcType, ProgramInput, Structure

from qccompute import compute

# Create the structure
# Can also open a structure from a file
# structure = Structure.open("path/to/h2o.xyz")
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[  # type: ignore
        [0.0, 0.0, 0.0],
        [0.52421003, 1.68733646, 0.48074633],
        [1.14668581, -0.45032174, -1.35474466],
    ],
)

# Define the program input
prog_input = ProgramInput(
    program="xtb",
    structure=structure,
    calctype=CalcType.energy,
    model={"method": "GFN2xTB"},  # type: ignore
    keywords={"max_iterations": 150},
)


prog_output = compute(prog_input)
print(prog_output)
