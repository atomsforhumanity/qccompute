"""Define a codec for a program qccodec does not know about, and run it.

The "program" here is a Python one-liner so the example runs anywhere; swap the
command and regex for the real thing.
"""

import sys

import qccodec
from qccodec import Codec, SubprocessInput, re_search
from qcdata import CalcType, ProgramInput, SinglePointData, Structure

from qccompute import compute

codec = Codec("myprog", calctypes={CalcType.energy})


@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    """Write the program's input file and build its command line."""
    assert inp.model is not None
    run_inp = f"method {inp.model.method}\n{inp.structure.to_xyz()}"
    # Stand-in for ("myprog", "run.inp")
    command = (sys.executable, "-c", "print('Total energy: -1.5')")
    return SubprocessInput(command, files={"run.inp": run_inp})


@codec.parser(target="energy", calctypes=[CalcType.energy])
def parse_energy(stdout: str) -> float:
    """Pull the final energy out of the program's stdout."""
    return float(re_search(r"Total energy:\s+(-?\d+\.\d+)", stdout).group(1))


@codec.parser(target=("provenance", "program_version"), required=False)
def parse_version(stdout: str) -> str:
    """Record the version when the program prints one."""
    return re_search(r"myprog version (\S+)", stdout).group(1)


prog_input = ProgramInput(
    program="myprog",
    calctype="energy",  # type: ignore
    structure=Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 1.4]]),  # type: ignore
    model={"method": "hf"},  # type: ignore
)

# Pass the codec explicitly...
output = compute(prog_input, codec=codec)
assert isinstance(output.results, SinglePointData)
print("energy:", output.results.energy)

# ...or register it once and run by program name, e.g. as a geomeTRIC gradient child.
qccodec.register(codec)
output = compute(prog_input)
print("provenance:", output.results.provenance)
