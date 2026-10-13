"""Check optional integration compatibility and discovery errors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from qccodec.programs.qcengine import codec_for
from qcdata import (
    ProgramInput,
    SinglePointData,
)
from qcdata.qcel import to_qcel_input

from qccompute import compute
from qccompute.exceptions import (
    CodecNotFoundError,
    ProgramNotFoundError,
    QCComputeBaseError,
)


def test_qcengine_fallback_retains_producer_and_logs(
    prog_input_factory: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qcengine fallback retains producer and logs."""
    qcengine = pytest.importorskip("qcengine")
    inp = prog_input_factory("energy", program="psi4")
    native = dict(
        **to_qcel_input(inp),
        success=True,
        return_result=-1.0,
        properties={"return_energy": -1.0},
        stdout="native logs",
        provenance={"creator": "Psi4", "version": "1.9", "routine": "energy"},
    )
    monkeypatch.setattr(qcengine, "get_program", lambda program: object())
    calls = []

    def invoke(data: dict[str, Any], program: str, **options: Any) -> dict[str, Any]:
        """Exercise the callable boundary with deterministic native behavior."""
        calls.append(options)
        return native

    monkeypatch.setattr(qcengine, "compute", invoke)
    output = compute(inp, codec=codec_for("psi4", task_config={"ncores": 2}))
    assert isinstance(output.results, SinglePointData)
    assert output.success and output.results.energy == -1
    assert output.results.provenance.program == "Psi4"
    assert output.results.provenance.program_version == "1.9"
    assert output.logs == "native logs"
    assert calls[0]["task_config"]["ncores"] == 2
    assert calls[0]["task_config"]["scratch_directory"] == str(
        output.execution.scratch_dir
    )


@pytest.mark.parametrize(
    "exception,expected",
    [("InputError", CodecNotFoundError), ("ResourceError", ProgramNotFoundError)],
)
def test_qcengine_discovery_errors(
    prog_input_factory: Callable[..., ProgramInput],
    monkeypatch: pytest.MonkeyPatch,
    exception: str,
    expected: type[QCComputeBaseError],
) -> None:
    """Qcengine discovery errors."""
    qcengine = pytest.importorskip("qcengine")

    def fail(program: str) -> None:
        """Simulate the optional engine's discovery failure."""
        raise getattr(qcengine.exceptions, exception)("not available")

    monkeypatch.setattr(qcengine, "get_program", fail)
    with pytest.raises(expected) as raised:
        compute(prog_input_factory("energy", program="unknown"))
    assert raised.value.prog_output is not None
    assert not raised.value.prog_output.success


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
