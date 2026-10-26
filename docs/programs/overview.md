# Programs

qccompute runs programs; everything specific to a program lives in its
[qccodec](https://qccodec.docs.afh.org) codec. The qccodec documentation
has a page for each supported program, explaining how a `ProgramInput` maps onto it
(keywords, solvent, parallelism, quirks), with a complete, runnable example script for
every calculation type:

- [CREST](https://qccodec.docs.afh.org/programs/crest/)
- [geomeTRIC](https://qccodec.docs.afh.org/programs/geometric/)
- [ORCA](https://qccodec.docs.afh.org/programs/orca/)
- [Psi4](https://qccodec.docs.afh.org/programs/psi4/)
- [TeraChem](https://qccodec.docs.afh.org/programs/terachem/) and its
  [servers](https://qccodec.docs.afh.org/programs/terachem-servers/)
- [xtb](https://qccodec.docs.afh.org/programs/xtb/)

Program-specific decode options are passed as `compute(input_data, options={...})`;
each codec declares what it accepts in `codec.options`. A program not on `PATH`, or
shadowed by another command of the same name, is selected with
`compute(..., executables={"orca": "/path/to/orca"})` or the environment variable
`QCCOMPUTE_<PROGRAM>_EXECUTABLE`.

Programs qccodec doesn't support yet can be run by writing a codec in your own script;
see [Custom codec](../examples/custom_codec.md).
