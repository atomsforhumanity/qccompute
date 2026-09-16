from qcdata import ProgramInput, Structure

from qccompute import compute, exceptions

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
    program="terachem",
    structure=structure,
    # Can be "energy", "gradient", "hessian"
    calctype="energy",  # type: ignore
    model={"method": "hf", "basis": "sto-3g"},  # type: ignore
    keywords={"purify": "no"},
)

# Run the calculation
try:
    # prog_output is a ProgramOutput instance
    prog_output = compute(prog_input, collect_files=True)
except exceptions.QCComputeBaseError as e:
    prog_output = e.prog_output
    print(prog_output.logs)
    print(f"Success: {prog_output.success}")  # False
    print(prog_output.input_data)  # Input data used to generate the calculation
    print(prog_output.results.provenance)  # Provenance of generated calculation
    print(prog_output.traceback)  # or output.ptraceback for short
    raise

else:
    # Check results
    print(prog_output.logs)
    print(f"Success: {prog_output.success}")  # True
    print("output.results: ", prog_output.results)
    print("output.results.energy:", prog_output.results.energy)
    print("output.results.gradient:", prog_output.results.gradient)
    print("output.results.hessian:", prog_output.results.hessian)
    print(prog_output.input_data)  # Input data used to generate the calculation
    print(prog_output.results.provenance)  # Provenance of generated calculation
