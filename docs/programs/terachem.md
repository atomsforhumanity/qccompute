Program-specific translation lives in `qccodec.programs.terachem`.
Use `qccompute.compute(input_data)` to execute a calculation. Optimizations require
`keywords={"new_minimizer": "yes"}`. `propagate_wfn=True` on a geomeTRIC run reuses the
`c0`/`ca0`/`cb0` orbital files between gradient steps.

The `terachem-fe` and `terachem-pbs` integrations (`qccodec.programs.tcpb`) require a
tcpb release compatible with current qcdata; tcpb 0.16.0 is rejected with an explicit
error.
