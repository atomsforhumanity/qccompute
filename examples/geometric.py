"""Example of using geometric subprogram to optimize H2 bond length.

Constraints docs: https://geometric.readthedocs.io/en/latest/constraints.html
"""

from qcdata import ProgramInput, ProgramSpec, Structure

from qccompute import compute, exceptions

# Create Structure
h2 = Structure(
    symbols=["H", "H"],
    geometry=[[0, 0.0, 0.0], [0, 0, 1.4]],  # type: ignore
)

# Define the program input
prog_input = ProgramInput(
    program="geometric",
    calctype="optimization",  # type: ignore
    structure=h2,
    subprograms=[
        ProgramSpec.model_validate(
            {
                "program": "terachem",
                "calctype": "gradient",
                "model": {"method": "HF", "basis": "6-31g"},
                "keywords": {"purify": "no"},
            }
        ),
    ],
    keywords={
        "check": 3,
        # This is obviously a stupid constraint, but it's just an example to show how
        # to use them
        "constraints": {
            "freeze": [
                {"type": "distance", "indices": [0, 1], "value": 1.4},
            ],
        },
    },
)

# Run calculation
try:
    prog_output = compute(prog_input, propagate_wfn=True, rm_scratch_dir=False)
except exceptions.QCComputeBaseError as e:
    # Calculation failed
    prog_output = e.prog_output
    print(prog_output.logs)
    # Input data used to generate the calculation
    print(prog_output.input_data)
    # Provenance of generated calculation
    print(prog_output.results.provenance)
    print(prog_output.traceback)
    raise

else:
    # Check results
    print("Energies:", prog_output.results.energies)
    print("Structures:", prog_output.results.structures)
    print("Trajectory:", prog_output.results.trajectory)
    # Stdout from the program
    print(prog_output.logs)
    # Input data used to generate the calculation
    print(prog_output.input_data)
    # Provenance of generated calculation
    print(prog_output.results.provenance)
