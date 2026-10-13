qccodec supplies program integrations; qccompute executes their prepared invocations.
Built-ins include TeraChem, ORCA, CREST, xTB, and geomeTRIC, plus optional tcpb
clients. QCEngine remains an optional fallback for other programs.
Scientific keywords and supported calculation types are defined by each codec.
External integrations can be passed as `compute(input_data, codec=my_codec)`.
