from qcdata import CalcType, ProgramInput, Structure

from qccompute import compute, exceptions

# Create the structure
# Can also open a structure from a file
# structure = Structure.open("path/to/h2o.xyz")
structure = Structure(
    symbols=["O", "H", "H"],
    geometry=[  # type: ignore
        [0.0253397, 0.01939466, -0.00696322],
        [0.22889176, 1.84438441, 0.16251426],
        [1.41760224, -0.62610794, -1.02954938],
    ],
)

# Define the program input
prog_input = ProgramInput(
    program="terachem",
    structure=structure,
    calctype=CalcType.hessian,
    model={"method": "b3lyp", "basis": "6-31g"},  # type: ignore
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
    print("output.results.hessian:", prog_output.results.hessian)
    print(prog_output.input_data)  # Input data used to generate the calculation
    print(prog_output.results.provenance)  # Provenance of generated calculation
