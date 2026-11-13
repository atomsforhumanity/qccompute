# Programs and sessions

The protocol an AFH program implements so qccompute can discover, validate, schedule and run
it, and the sessions drivers use to evaluate one molecule at many geometries. A program
registers one object (an instance) under the `qccompute.programs` entry point group and
never imports qccompute: the protocols are structural, and what it returns for
`capabilities` and `plan` may be a plain mapping. What programs and drivers share as data
(`Request`, `Accuracy`, `ProgramError`, `Frame`) is in qcdata. Design: [the program
interface](../design/program-interface.md).

`qccompute.testing.check_program(program, inputs)` checks the contract (install
`qccompute[testing]` to validate keywords against the program's JSON Schema), and
`qccompute.testing.toy` is a complete example program (Lennard-Jones).

::: qccompute.program.Program

::: qccompute.program.SessionProgram

::: qccompute.program.BatchProgram

::: qccompute.program.Session

::: qccompute.program.Capabilities

::: qccompute.program.Plan

::: qccompute.program.Resources

::: qccompute.program.RunContext

::: qccompute.program.ComputeSession

::: qccompute.program.find_program

::: qccompute.program.register_program

::: qccompute.program.unregister_program

::: qccompute.program.available_programs

::: qccompute.testing.check_program

::: qccompute.program.capabilities_of

::: qccompute.program.plan_of

::: qccompute.program.local_context
