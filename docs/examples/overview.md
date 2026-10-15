Calculations are run by calling the `qccompute.compute()` function with the relevant arguments and keywords like this:

```python
from qcdata import Structure, ProgramInput
from qccompute import compute

# Create the Structure
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
    calctype="energy",
    model={"method": "hf", "basis": "sto-3g"},
    keywords={"purify": "no", "restricted": False},
)

# Run the calculation; will return a ProgramOutput or raise an exception
result = compute(prog_input, collect_files=True)
```

The `compute` function resolves a codec and executes its native invocation. It also accepts an explicit local `codec=`. The available arguments are:

::: qccompute.main.compute
