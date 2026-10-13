Program-specific translation and Python bindings live in `qccodec.codecs.geometric`.
Use `qccompute.compute(input_data)` to execute a calculation.

geomeTRIC requires a gradient child in `ProgramInput.subprograms`. Its native engine calls qccompute through an injected callback; failed children remain in the trajectory.
