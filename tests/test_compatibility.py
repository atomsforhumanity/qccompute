"""Check optional integration compatibility and discovery errors."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from qcdata import (
    ProgramInput,
)

from qccompute import compute


def test_tcpb_guard(
    prog_input_factory: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tcpb guard."""
    import sys
    from types import SimpleNamespace

    from qccodec.programs import tcpb

    monkeypatch.setitem(sys.modules, "tcpb", SimpleNamespace())
    monkeypatch.setattr(tcpb, "version", lambda _: "0.16.0")
    output = compute(
        prog_input_factory("energy", program="terachem-fe"), raise_exc=False
    )
    assert output.traceback is not None
    assert not output.success and "tcpb 0.16.0" in output.traceback
    assert output.results.provenance.program == "terachem"
