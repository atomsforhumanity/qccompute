Program-specific translation and the native binding live in `qccodec.programs.xtb`.
Use `qccompute.compute(input_data)` to execute a calculation.

Install `qccompute[xtb]` in an environment supported by xtb-python. xTB runs with
`OMP_NUM_THREADS=1` unless overridden with `compute(..., environment=...)`; OpenMP
settings must be applied before xTB is first imported in the process.
