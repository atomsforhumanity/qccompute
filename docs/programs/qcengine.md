Program-specific translation lives in `qccodec.programs.qcengine`.
Use `qccompute.compute(input_data)` to execute a calculation.

QCEngine is used automatically for programs without a built-in codec when
`qccompute[qcengine]` is installed. Pass QCEngine settings through its codec factory:
`compute(inp, codec=qccodec.programs.qcengine.codec_for("psi4", task_config={...}))`.
