# Execution boundaries

Program-specific encoding, decoding, validation, native API bindings, restart
translation, and options belong to qccodec, as one `Codec` per program. qccompute owns
workspace lifetime, invocation, logging, environment policy, output construction, and
failures. qccompute never branches on a program name; it only calls `codec.encode`,
`codec.decode`, and `codec.propagate_wfn`, and forwards `options=` untouched.

`compute(codec=...)` accepts only `qccodec.Codec` instances. Codecs for programs that are
not built in are either passed explicitly or made resolvable with `qccodec.register`,
which is also how optimizers find user-defined gradient codecs.

The callable in a `PythonInput` can perform several native API calls; no setup/teardown
hook hierarchy is needed. geomeTRIC additionally requests a synchronous `compute_child`
callback. The native Engine subclass lives in qccodec because it is part of geomeTRIC's
own interface.

Failures retain correctly typed partial results and attach `ProgramOutput` to
exceptions. Producer provenance is never replaced with wrapper identity. Data needed by
decoders remains available until decoding finishes, independently of retention settings.
