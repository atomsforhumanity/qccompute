Program-specific translation and Python bindings live in `qccodec.codecs.terachem`.
Use `qccompute.compute(input_data)` to execute a calculation.

The `terachem-fe` and `terachem-pbs` integrations require a tcpb release compatible with current qcdata; tcpb 0.16.0 is rejected with an explicit error.
