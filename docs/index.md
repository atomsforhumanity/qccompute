# qccompute

A package for running Quantum Chemistry programs using [qcdata](https://github.com/atomsforhumanity/qcdata) standardized data structures. Compatible with `TeraChem`, `psi4`, `Crest`, `xTB`, `QChem`, `NWChem`, `ORCA`, `Molpro`, `geomeTRIC` and many more.

[![image](https://img.shields.io/pypi/v/qccompute.svg)](https://pypi.python.org/pypi/qccompute)
[![image](https://img.shields.io/pypi/l/qccompute.svg)](https://pypi.python.org/pypi/qccompute)
[![image](https://img.shields.io/pypi/pyversions/qccompute.svg)](https://pypi.python.org/pypi/qccompute)
[![Actions status](https://github.com/atomsforhumanity/qccompute/workflows/Tests/badge.svg)](https://github.com/atomsforhumanity/qccompute/actions)
[![Actions status](https://github.com/atomsforhumanity/qccompute/workflows/Basic%20Code%20Quality/badge.svg)](https://github.com/atomsforhumanity/qccompute/actions)

`qccompute` works in harmony with a suite of other quantum chemistry tools for fast, structured, and interoperable quantum chemistry.

## The QC Suite of Programs

The QC Suite works in harmony to provide fast, structured, and interoperable quantum chemistry tools.

- [qcconst](https://github.com/atomsforhumanity/qcconst) - Physical constants, conversion factors, and a periodic table with clear source information for every value.
- [qcdata](https://github.com/atomsforhumanity/qcdata) - Elegant and intuitive data structures for quantum chemistry, featuring seamless Jupyter Notebook visualizations. [Documentation](https://qcdata.docs.atomsforhumanity.org)
- [qcinf](https://github.com/atomsforhumanity/qcinf) - Cheminformatics algorithms and structure utilities using standardized [qcdata](https://qcdata.docs.atomsforhumanity.org/) data structures.
- [qccodec](https://github.com/atomsforhumanity/qccodec) - A package for translating between standardized [qcdata](https://github.com/atomsforhumanity/qcdata) data structures and native QC program inputs and outputs.
- [qccompute](https://github.com/atomsforhumanity/qccompute) - A package for operating quantum chemistry programs using standardized [qcdata](https://qcdata.docs.atomsforhumanity.org/) data structures. Compatible with `TeraChem`, `psi4`, `QChem`, `NWChem`, `ORCA`, `Molpro`, `geomeTRIC` and many more.
- [BigChem](https://github.com/mtzgroup/bigchem) - A distributed application for running quantum chemistry calculations at scale across clusters of computers or the cloud. Bring multi-node scaling to your favorite quantum chemistry program.
- `ChemCloud` - A [web application](https://github.com/mtzgroup/chemcloud-server) and associated [Python client](https://github.com/mtzgroup/chemcloud-client) for exposing a BigChem cluster securely over the internet.

## Installation

```sh
python -m pip install qccompute
```

## Quickstart

`qccompute` uses the `qcdata` data structures to drive quantum chemistry programs in a standardized way. This allows for a simple and consistent interface to a wide variety of quantum chemistry programs. See the [qcdata](https://github.com/atomsforhumanity/qcdata) library for documentation on the input and output data structures.

The `compute` function is the main entry point for the library and is used to run a calculation.

```python
from qcdata import Structure, ProgramInput
from qccompute import compute
from qccompute.exceptions import ExternalProgramError

# Create the Structure
h2o = Structure.open("h2o.xyz")

# Define the program input
prog_input = ProgramInput(
    program="terachem",
    structure=h2o,
    calctype="energy",
    model={"method": "hf", "basis": "sto-3g"},
    keywords={"purify": "no", "restricted": False},
)

# Run the calculation; will return a ProgramOutput or raise an exception
try:
    prog_output = compute(prog_input, collect_files=True)
except ExternalProgramError as e:
    # External QQ program failed in some way
    prog_output = e.prog_output
    prog_output.input_data # Input data used by the QC program
    prog_output.success # Will be False
    prog_output.results # Any half-computed results before the failure
    prog_output.logs # Logs from the calculation
    prog_output.plogs # Shortcut to print out the logs in human readable format
    prog_output.traceback # Stack trace from the calculation
    prog_output.ptraceback # Shortcut to print out the traceback in human readable format
    raise e
else:
    # Calculation succeeded
    prog_output.input_data # Input data used by the QC program
    prog_output.success # Will be True
    prog_output.results # All structured data and files from the calculation
    prog_output.results.files # Any files returned by the calculation
    prog_output.logs # Logs from the calculation
    prog_output.plogs # Shortcut to print out the logs in human readable format
    prog_output.results.provenance # Provenance information about the calculation
    prog_output.extras # Any extra information not in the schema

```

One may also call `compute(..., raise_exc=False)` to return a `ProgramOutput` object rather than raising an exception when a calculation fails. This may allow easier handling of failures in some cases.

```python
from qcdata import Structure, ProgramInput
from qccompute import compute
from qccompute.exceptions import ExternalProgramError
# Create the Structure
h2o = Structure.open("h2o.xyz")

# Define the program input
prog_input = ProgramInput(
    program="terachem",
    structure=h2o,
    calctype="energy",
    model={"method": "hf", "basis": "sto-3g"},
    keywords={"purify": "no", "restricted": False},
)

# Run the calculation; will return a ProgramOutput object
prog_output = compute(prog_input, collect_files=True, raise_exc=False)
if not prog_output.success:
    # Same as except block above

else:
    # Same as else block above

```

Alternatively, the `compute_args` function can be used to run a calculation with the input data structures passed in as arguments rather than as a single `ProgramInput` object.

```python
from qcdata import Structure
from qccompute import compute_args
# Create the Structure
h2o = Structure.open("h2o.xyz")

# Run the calculation
prog_output = compute_args(
    "terachem",
    h2o,
    calctype="energy",
    model={"method": "hf", "basis": "sto-3g"},
    keywords={"purify": "no", "restricted": False},
    files={...},
    collect_files=True
)
```

The behavior of `compute()` and `compute_args()` can be tuned by passing in keyword arguments like `collect_files` shown above. Arguments can modify which scratch directory location to use, whether to delete or keep the scratch files after a calculation completes, what files to collect from a calculation, whether to stream the program logs in real time as the program executes, and whether to propagate a wavefunction through a series of calculations. Arguments also include hooks for passing in update functions that can be called as a program executes in real time. See the `qccompute.compute` docstring for details.

See the [/examples](https://github.com/atomsforhumanity/qccompute/tree/master/examples) directory for more examples.

## ✨ Visualization ✨

Visualize all your results with a single line of code!

First install the visualization module:

```sh
python -m pip install qcdata[view]
```

or if your shell requires `''` around arguments with brackets:

```sh
python -m pip install 'qcdata[view]'
```

Then in a Jupyter notebook import the `qcdata` view module and call `view.view(...)` passing it one or any number of `qcdata` objects you want to visualize, including `Structure` objects or any `ProgramOutput` object. You may also pass arrays of `titles` and/or `subtitles` to add additional information to the molecular structure display. If no titles are passed, `qcdata` will look for `Structure` identifiers such as a name or SMILES to label the `Structure`.

![Structure Viewer](https://public.coltonhicks.com/assets/qcdata/structure_viewer.png)

Seamless visualizations for `ProgramOutput` objects make results analysis easy!

![Optimization Viewer](assets/optimization_viewer.png)

Single point calculations display their results in a table.

![Single Point Viewer](assets/single_point_viewer.png)

If you want to use the HTML generated by the viewer to build your own dashboards, use the functions inside `qcdata.view.py` that begin with `generate_` to create HTML you can insert into any dashboard.

## Support

If you have any issues with `qccompute` or would like to request a feature, please open an [issue](https://github.com/atomsforhumanity/qccompute/issues).

## Custom programs

qccompute only executes; everything program-specific comes from a
[qccodec](https://github.com/atomsforhumanity/qccodec) `Codec`. When a program is not
built in, define one in your script and pass it to `compute`, or register it so it is
found by name (including as a geomeTRIC gradient child):

```python
from qcdata import CalcType, ProgramInput
from qccodec import Codec, SubprocessInput, re_search
import qccodec, qccompute

codec = Codec("myprog", calctypes={CalcType.energy})

@codec.encoder
def encode(inp: ProgramInput) -> SubprocessInput:
    return SubprocessInput(("myprog", "run.inp"), files={"run.inp": inp.structure.to_xyz()})

@codec.parser(target="energy", calctypes=[CalcType.energy])
def parse_energy(stdout: str) -> float:
    return float(re_search(r"Total energy:\s+(-?\d+\.\d+)", stdout).group(1))

output = qccompute.compute(prog_input, codec=codec)   # explicit
qccodec.register(codec)
output = qccompute.compute(prog_input)                # by program name
```

Program-specific decode options are passed with `options=`; each codec declares the
options it accepts in `codec.options` and unknown names are rejected:

```python
compute(prog_input, options={"collect_rotamers": True})   # CREST
```

## Development

qccompute requires `qcdata>=0.19.0` and `qccodec>=0.13.0`. For coordinated development
of unreleased changes keep the companion qccodec checkout at `../qccodec` and run
`uv sync --all-groups --locked`; the editable source override in `pyproject.toml` is
temporary and should be removed once qccodec 0.13 is published.

### How a calculation runs

1. `compute()` resolves a codec: the `codec=` argument, a codec registered with
   `qccodec.register`, a built-in, or QCEngine when `qccompute[qcengine]` is installed.
2. `codec.encode(input_data)` returns a `SubprocessInput` (command + files) or a
   `PythonInput` (callable + arguments). Encoding never executes anything.
3. qccompute creates the scratch directory, stages the files, and runs
   `run_subprocess` or `run_python`. Both merge stdout and stderr into one log;
   Python capture includes native writes to the stdout/stderr file descriptors and the
   loggers the codec names.
4. `codec.decode(input_data, native_output)` is always called, even after an execution
   failure, so partial results are recovered. Exceptions carrying `.data` keep it.
5. Requested files and wavefunctions are collected, the directory is removed if
   `rm_scratch_dir` is True, and the `ProgramOutput` is built. On failure, the
   exception's `.prog_output` holds the complete failed output.

Environment: invocation defaults from the codec override inherited values, and
`environment=` overrides both. Subprocess overrides are local to the child. Python
execution temporarily changes the process environment, working directory, and output
streams under a reentrant lock, so use separate processes for independent concurrent
Python calculations. xTB defaults to `OMP_NUM_THREADS=1` before its library import,
and CREST receives the same setting in its child environment.

Logs: subprocess callbacks stream `(all_logs, new_logs)` with a final callback for
remaining output; `update_interval` is a minimum spacing. Python callbacks receive the
captured log once after invocation. Logs remain attached on failure even with
`collect_logs=False`.

Child calculations: geomeTRIC drives gradients through an injected callback, with a
separate workspace and `ProgramOutput` for each child. Child codecs are resolved by
program name. `propagate_wfn=True` keeps each child's wavefunction files and calls the
child codec's `propagate_wfn(output, input_data)` so the next step starts from them.

Exceptions: `CodecNotFoundError`, `InputError`, `ProgramNotFoundError`, and
`ExternalProgramError` all derive from `QCComputeBaseError` and carry `.prog_output`.
The older `AdapterNotFoundError`/`AdapterInputError` names remain as aliases for one
release.

### TeraChem FE/PBS dependency

The optional FE/PBS codecs require a separate `tcpb` migration. Published
`tcpb==0.16.0` constructs the old qcdata models internally, so requests report an
explicit dependency error (or return a failed `ProgramOutput` when `raise_exc=False`).
Standard TeraChem subprocess execution is independent of this limitation.
