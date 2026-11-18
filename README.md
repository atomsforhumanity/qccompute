# qccompute

[![image](https://img.shields.io/pypi/v/qccompute.svg)](https://pypi.python.org/pypi/qccompute)
[![image](https://img.shields.io/pypi/l/qccompute.svg)](https://pypi.python.org/pypi/qccompute)
[![image](https://img.shields.io/pypi/pyversions/qccompute.svg)](https://pypi.python.org/pypi/qccompute)
[![CI](https://github.com/atomsforhumanity/qccompute/actions/workflows/ci.yaml/badge.svg)](https://github.com/atomsforhumanity/qccompute/actions/workflows/ci.yaml)

Runs quantum chemistry programs on [qcdata](https://github.com/atomsforhumanity/qcdata) inputs: one input or many, in this process or in worker processes, with an optional calculation store. Every program, whether written in Python for the protocol or an external program brought in by [qccodec](https://github.com/atomsforhumanity/qccodec), runs the same way.

**Documentation of the alpha: [qccompute.docs.afh.org/next](https://qccompute.docs.afh.org/next/)** (the site root documents the stable release)

## Install

The new runner is in alpha. Install the alpha line with minimum versions of its matching packages (`--prerelease=allow` lets the resolver take pre-releases; a plain `pip install qccompute` resolves the previous stable release):

```sh
uv pip install --prerelease=allow "qcdata>=0.24.0a8" "qccompute[qccodec]>=0.23.0rc1" "qccodec>=0.22.0rc1" "qcstore>=0.1.0a4"
```

`qccompute` alone is the runner, for programs that implement its protocol (`uv pip install --prerelease=allow "qcdata>=0.24.0a8" "qccompute>=0.23.0rc1"`); the `qccodec` extra adds external programs (xtb, ORCA, Psi4, CREST, ...) and `qcstore` is the calculation store. Coming from 0.22? See [the migration guide](https://qccompute.docs.afh.org/next/migration/).

## One way to run things

```python
import qccompute
from qcdata import CalcType, Model, ProgramInput, Structure

water = Structure(
    symbols=["O", "H", "H"],
    geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],  # bohr
)
inp = ProgramInput(
    program="xtb", calctype=CalcType.gradient, structure=water, model=Model(method="gfn2")
)

out = qccompute.compute(inp)                 # one input -> ProgramOutput
outs = qccompute.compute([inp] * 8)          # a list runs side by side -> list
future = qccompute.submit(inp)               # start one -> Future of its output
with qccompute.session(inp) as s:            # one molecule at many geometries
    frame = s.compute(water.geometry * 1.01)
```

- **Failures raise.** A failed calculation raises `qcdata.ProgramError`; its `output` is the failed `ProgramOutput`, with partial results and logs. A list runs everything first, then raises one `qcdata.ProgramErrors` group whose `results` holds every output in order. `raise_exc=False` returns failures in place.
- **A store, if you want one.** `compute(inputs, store=store, tags=[...])` looks each input up first and stores what it ran, with the identities of the programs that made it; a run is reused only when those are the programs that would run now.
- **One thread count per calculation.** Its `threads` keyword, else the runner's whole count; a list shares it, and nothing ever runs more threads than its runner has. For many calculations at once, set `threads` (usually 1) on each. See [Threads](https://qccompute.docs.afh.org/next/threads/).
- **A share of the machine.** `qccompute.compute` and `qccompute.session` belong to the default runner: every core and GPU this process may use (or `OMP_NUM_THREADS` cores). A `Runner` of your own takes a share and closes its worker processes when done:

```python
with qccompute.Runner(threads=16, gpus=["0"], isolation="process") as runner:
    outputs = runner.compute(inputs, timeout=3600)
```

## Programs

A program is any object with `name`, `version()`, `capabilities()`, `settings_schema()`, `record_schema()`, `plan(inp)` and `compute(inp, ctx=None)`, registered under the `qccompute.programs` entry points (or with `qccompute.register_program` in a script). The protocol is structural, so a program without subprograms imports only qcdata. `qccompute.program.FunctionProgram` turns Python functions of the positions into a program, and `qccompute.testing.check_program` checks one against the contract.

```python
from qccompute.program import FunctionProgram

def harmonic(x):  # positions (bohr) -> energy (Eh), gradient (Eh/bohr)
    return float((x**2).sum()), 2 * x

qccompute.register_program(FunctionProgram("harmonic", harmonic))
```

## The QC Suite

- [qcref](https://github.com/atomsforhumanity/qcref) - Reference data: physical constants, conversion factors, a periodic table, solvents, each value with its source.
- [qcdata](https://github.com/atomsforhumanity/qcdata) - Data structures for quantum chemistry inputs and results. [Documentation](https://qcdata.docs.afh.org/next/)
- [qcinf](https://github.com/atomsforhumanity/qcinf) - Cheminformatics algorithms and structure utilities for qcdata structures.
- [qccodec](https://github.com/atomsforhumanity/qccodec) - External programs as programs qccompute runs: their input files, output parsers, and execution. [Documentation](https://qccodec.docs.afh.org/next/)
- [qccompute](https://github.com/atomsforhumanity/qccompute) - Runs programs on qcdata inputs. [Documentation](https://qccompute.docs.afh.org/next/)

## Support

Questions and bug reports: [GitHub issues](https://github.com/atomsforhumanity/qccompute/issues).
