"""An example of how to create a FileInput object for a QC Program."""

from pathlib import Path

from qcdata import FileInput, Structure

from qccompute import compute

# Input files for QC Program
inp_file = Path("path/to/tc.in").read_text()  # Or your own function to create tc.in

# Structure object to XYZ file
structure = Structure.open("path/to/my/mol.xyz")
xyz_str = structure.to_xyz()  # type: ignore

# Create a FileInput object for TeraChem
file_input = FileInput(
    program="terachem",
    files={"tc.in": inp_file, "coords.xyz": xyz_str},
    cmdline_args=["tc.in"],
)

# This will write the files to disk in a temporary directory and then run
# "terachem tc.in" in that directory.
prog_output = compute(file_input, print_logs=True)

# Data
prog_output.logs
prog_output.input_data
prog_output.results.files  # Has all the files terachem creates
prog_output.results.files.keys()  # Print out file names
# Saves all outputs with the exact structure produced by the QC program
prog_output.results.save_files("to/this/directory")
