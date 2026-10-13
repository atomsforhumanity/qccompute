"""Exercise callback-driven optimization with an external local codec."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import qccodec
from numpy.typing import NDArray
from qccodec import Codec, NativeOutput, PythonInput
from qcdata import (
    CalcType,
    OptimizationData,
    ProgramInput,
    ProgramSpec,
    SinglePointData,
    Structure,
)

from qccompute import compute


@pytest.mark.parametrize("fail_child", [False, True])
def test_geometric_with_analytic_child(tmp_path: Path, fail_child: bool) -> None:
    """Geometric with analytic child."""
    pytest.importorskip("geometric")

    def potential(geometry: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Evaluate a harmonic bond potential with its analytic gradient."""
        delta = geometry[1] - geometry[0]
        distance = np.linalg.norm(delta)
        displacement = distance - 1.4
        gradient = displacement * delta / distance
        return (float(0.5 * displacement**2), np.array([-gradient, gradient]))

    def decode(inp: ProgramInput, native: NativeOutput) -> SinglePointData:
        """Interpret the fake native result or deliberately exercise a decoding failure."""
        if fail_child:
            raise RuntimeError("intentional child failure")
        return SinglePointData.model_validate(
            {
                "provenance": {"program": "analytic"},
                "energy": native.value[0],
                "gradient": native.value[1],
            }
        )

    child = Codec(
        "analytic",
        calctypes={CalcType.gradient},
        encode=lambda inp: PythonInput(potential, (inp.structure.geometry,)),
        decode=decode,
    )
    inp = ProgramInput.model_validate(
        {
            "program": "geometric",
            "calctype": "optimization",
            "structure": Structure.model_validate(
                {"symbols": ["H", "H"], "geometry": [[0, 0, 0], [0, 0, 1.8]]}
            ),
            "keywords": {"coordsys": "cart", "maxiter": 30},
            "subprograms": [
                ProgramSpec.model_validate(
                    {"program": "analytic", "calctype": "gradient"}
                )
            ],
        }
    )
    before = inp.model_copy(deep=True)
    qccodec.register(child)
    try:
        output = compute(inp, scratch_dir=tmp_path / "outer", raise_exc=False)
    finally:
        qccodec.unregister("analytic")
    assert inp == before
    assert isinstance(output.results, OptimizationData)
    assert output.results.trajectory
    assert output.success is not fail_child, output.traceback
    if fail_child:
        assert not output.results.trajectory[-1].success
        assert output.results.trajectory[-1].traceback is not None
        assert "intentional child failure" in output.results.trajectory[-1].traceback
    else:
        assert output.results.final_energy is not None
        assert output.results.final_energy < 1e-08
    assert output.execution.scratch_dir is not None
    assert not output.execution.scratch_dir.exists()
    for child_output in output.results.trajectory:
        assert child_output.execution.scratch_dir is not None
        assert not child_output.execution.scratch_dir.exists()
