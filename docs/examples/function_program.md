# A program from functions

`qccompute.program.FunctionProgram(name, energy_gradient, hessian=None)` makes a program
of Python functions of the positions (bohr): a model surface, an ML potential in a
script, a teaching example. It has no settings, runs in this process and is thread-safe,
so lists run side by side and sessions work as for any other program. An exception from
a function is a failed calculation of kind `internal`, with the function's traceback in
the failed output.

```python
{!examples/function_program.py!}
```
