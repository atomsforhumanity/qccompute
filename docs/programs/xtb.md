Program-specific translation lives in `qccodec.programs.xtb`, which runs the `xtb`
program on the command line. Use `qccompute.compute(input_data)` to execute a
calculation.

Install xtb from its [releases](https://github.com/grimme-lab/xtb/releases) and put it
on `PATH`. `model.method` selects `gfn0`, `gfn1`, `gfn2`, `gfnff`, or `gxtb`, and
keywords pass through as xtb's command-line flags (`{"alpb": "water"}` gives
`--alpb water`). GFN0 needs `XTBPATH` set to xtb's parameter directory. xtb runs with
`OMP_NUM_THREADS=1` unless overridden with `compute(..., environment=...)`.

g-xTB ships as its own build of xtb ([grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb)).
That build also runs GFN0/1/2-xTB and GFN-FF, so it can be your only xtb: put it on
`PATH`, or point qccompute at it:

```python
compute(input_data, executables={"xtb": "/opt/g-xtb/bin/xtb"})
```

or set `QCCOMPUTE_XTB_EXECUTABLE`. Requesting `gxtb` from an xtb build without g-xTB
fails with a message saying so, because xtb itself only warns about the unknown
`--gxtb` option. CREST (`program="crest"`) also runs GFN-xTB and
GFN-FF, and is the program for conformer searches.
