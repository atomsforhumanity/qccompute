# Programs

qccompute runs any program that implements its [protocol](../api/program.md), found by
name among the installed programs (the `qccompute.programs` entry points) and those
registered in this process with `qccompute.register_program`.

External programs come with `pip install qccompute[qccodec]`: qccodec registers each as a
program. Its documentation has a page for each, explaining how a `ProgramInput` maps
onto it (keywords, solvent, parallelism, quirks), with a runnable example script for every
calculation type:

- [CREST](https://qccodec.docs.afh.org/next/programs/crest/)
- [DFTB+](https://qccodec.docs.afh.org/next/programs/dftbplus/)
- [geomeTRIC](https://qccodec.docs.afh.org/next/programs/geometric/) and
  [Sella](https://qccodec.docs.afh.org/next/programs/sella/)
- [ORCA](https://qccodec.docs.afh.org/next/programs/orca/)
- [Psi4](https://qccodec.docs.afh.org/next/programs/psi4/)
- [TeraChem](https://qccodec.docs.afh.org/next/programs/terachem/) and its
  [servers](https://qccodec.docs.afh.org/next/programs/terachem-servers/)
- [xtb](https://qccodec.docs.afh.org/next/programs/xtb/) and
  [tblite](https://qccodec.docs.afh.org/next/programs/tblite/)
- machine-learned potentials: [AIMNet2](https://qccodec.docs.afh.org/next/programs/aimnet2/),
  [FAIR-Chem](https://qccodec.docs.afh.org/next/programs/fairchem/),
  [MACE](https://qccodec.docs.afh.org/next/programs/mace/),
  [Orb](https://qccodec.docs.afh.org/next/programs/orb/),
  [TorchANI](https://qccodec.docs.afh.org/next/programs/torchani/)

An external program not on `PATH`, or shadowed by another command of the same name, is
found with the environment variable `QCCODEC_<PROGRAM>_EXECUTABLE`.
