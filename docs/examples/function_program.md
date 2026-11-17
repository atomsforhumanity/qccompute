# A program from functions

`qccompute.program.FunctionProgram(name, energy_gradient, hessian=None)` makes a program
of Python functions of the positions (bohr): a model surface, an ML potential in a
script, a teaching example. It runs in this process and is thread-safe, so lists run
side by side and sessions work as for any other program. It takes the shared keywords
`threads` (what it asks the runner for, 1 by default) and `accuracy` (recorded as met:
nothing converges). An exception from a function is a failed calculation of kind
`internal`, with the function's traceback in the failed output; a `ProgramError` it
raises is the calculation's own failure.

With `takes_input=True` the functions get the input after the positions, so one program
stands in for a chemistry engine on any molecule and setting: the function reads the
symbols, charge, multiplicity, model and keywords. `settings=` declares the JSON Schema
of the keywords it reads; the program refuses any other key (`unknown_key`), and the
function checks their values:

```python
def engine(x, inp):  # positions (bohr), the input
    calc = Calculator(inp.structure.symbols, inp.structure.charge,
                      method=inp.model.method, accuracy=inp.keywords.get("acc", 1.0))
    return calc.energy_gradient(x)

settings = {"type": "object", "properties": {
    "acc": {"type": "number", "exclusiveMinimum": 0, "x-concern": "convergence"}}}
qccompute.register_program(
    FunctionProgram("engine", engine, takes_input=True, settings=settings)
)
```

Its version and numerics identity is a digest of the functions (`fn-...`): their code,
constants, defaults, closures and the module globals they read (`D`, `A` and `R0` below),
recursing into the Python functions they call. Edit the function or one of those values
and a store no longer serves the old results. Give `version="..."` to name the numbers
yourself, for example a model's checksum when the functions call an ML model.

```python
{!examples/function_program.py!}
```
