# Execution boundaries

Program-specific encoding, decoding, validation, native API bindings, and restart translation belong to qccodec. qccompute owns workspace lifetime, invocation, logging, environment policy, output construction, and failures. There is no Adapter or Backend hierarchy.

The callable in a PythonInput can perform several native API calls; no setup/teardown hook hierarchy is needed. geomeTRIC additionally requests a synchronous compute_child callback. The native Engine subclass lives in qccodec because it is part of geomeTRIC's own interface.

Failures retain correctly typed partial results and attach ProgramOutput to exceptions. Producer provenance is never replaced with wrapper identity. Data needed by decoders remains available until decoding finishes, independently of retention settings.
