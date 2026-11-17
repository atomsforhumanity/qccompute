"""Run a command with its files and get every file it wrote back (needs qccodec).

A FileInput is a command line and the files it needs. qccompute runs it under the
``files`` program that ``qccompute[qccodec]`` provides, in a scratch directory, and
returns the files the command wrote in ``results.files``.
"""

from pathlib import Path

from qcdata import FileInput, Structure

from qccompute import compute

# Input files for the program
inp_file = Path("path/to/tc.in").read_text()  # Or your own function to create tc.in

# Structure object to XYZ file
structure = Structure.open("path/to/my/mol.xyz")
xyz_str = structure.to_xyz()  # type: ignore

# Run "terachem tc.in" in a scratch directory holding these files.
file_input = FileInput(
    program="terachem",
    files={"tc.in": inp_file, "coords.xyz": xyz_str},
    cmdline_args=["tc.in"],
)
prog_output = compute(file_input)

print(prog_output.logs)
print(prog_output.results.files.keys())  # every file terachem wrote
prog_output.results.save_files("to/this/directory")
