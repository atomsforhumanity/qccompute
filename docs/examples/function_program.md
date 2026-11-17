# A program from functions

`qccompute.program.FunctionProgram(name, energy_gradient, hessian=None)` makes a program
of Python functions of the positions (bohr): a model surface, an ML potential in a
script, a teaching example. It has no settings, runs in this process and is thread-safe,
so lists run side by side and sessions work as for any other program. An exception from
a function is a failed calculation of kind `internal`, with the function's traceback in
the failed output.

Its version and numerics identity is a digest of the functions (`fn-...`): their code,
constants, defaults, closures and the module globals they read (`D`, `A` and `R0` below),
recursing into the Python functions they call. Edit the function or one of those values
and a store no longer serves the old results. Give `version="..."` to name the numbers
yourself, for example a model's checksum when the functions call an ML model.

```python
{!examples/function_program.py!}
```
