# Programs and sessions

The protocol a program implements so qccompute can find, plan, place and run it, and the
sessions callers use to evaluate one molecule at many geometries. A program registers one
object (an instance) under the `qccompute.programs` entry point group. The protocols are
structural and what a program returns for `capabilities` and `plan` may be a plain
mapping, so a program without subprograms never imports qccompute. What programs share as
data (`Request`, `Accuracy`, `ProgramError`, `Frame`) is in qcdata. Design: [the program
interface](../design/program-interface.md).

`qccompute.testing.check_program(program, inputs)` checks the contract (install
`qccompute[testing]` to validate keywords against the program's JSON Schema), and
`qccompute.testing.toy` is a complete example program (Lennard-Jones).

::: qccompute.program.Program

::: qccompute.program.SessionProgram

::: qccompute.program.BatchProgram

::: qccompute.program.Session

::: qccompute.program.RunContext

::: qccompute.program.Capabilities

::: qccompute.program.Plan

::: qccompute.program.Resources

::: qccompute.program.FunctionProgram

::: qccompute.program.ComputeSession

::: qccompute.program.register_program

::: qccompute.program.unregister_program

::: qccompute.program.find_program

::: qccompute.program.available_programs

::: qccompute.program.numerics_of

::: qccompute.program.capabilities_of

::: qccompute.program.plan_of

::: qccompute.testing.check_program
