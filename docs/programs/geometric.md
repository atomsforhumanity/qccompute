Program-specific translation and the native engine bridge live in `qccodec.programs.geometric`.
Use `qccompute.compute(input_data)` to execute a calculation.

geomeTRIC requires a gradient child in `ProgramInput.subprograms`. Its native engine
calls qccompute through an injected callback; each gradient is its own `ProgramOutput`
in the trajectory, and failed children remain there. The child codec is resolved by the
child's program name, so a codec defined in your script works as the gradient engine
once you call `qccodec.register(codec)`.
