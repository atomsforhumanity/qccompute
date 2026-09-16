::: qccompute.adapters.terachem.TeraChemAdapter

::: qccompute.adapters.terachem_fe.TeraChemFEAdapter

::: qccompute.adapters.terachem_pbs.TeraChemPBSAdapter

### TeraChem FE/PBS dependency

The optional FE/PBS adapters require a separate `tcpb` migration. Published
`tcpb==0.16.0` constructs the old qcdata models internally, so requests now report
an explicit dependency error (or return a failed `ProgramOutput` when
`raise_exc=False`). Standard TeraChem subprocess execution uses the migrated
qccodec parsers and is independent of this limitation.
