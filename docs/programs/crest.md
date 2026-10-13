Program-specific translation lives in `qccodec.programs.crest`.
Use `qccompute.compute(input_data)` to execute a calculation.

Conformer searches return conformers by default. Rotamer sets can be very large, so
they are opt-in: `compute(input_data, options={"collect_rotamers": True})`.
