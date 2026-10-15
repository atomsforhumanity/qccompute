qccodec supplies program integrations as `Codec` objects; qccompute executes their
prepared invocations. Built-ins include TeraChem, ORCA, CREST, xTB, and geomeTRIC, plus
the optional tcpb clients. Scientific keywords and supported calculation types are defined by each codec.

Program-specific decode options are passed as `compute(input_data, options={...})`;
each codec declares what it accepts in `codec.options`. Custom programs are defined in a
script as a `qccodec.Codec` and passed as `codec=` or made resolvable by name with
`qccodec.register(codec)`.
