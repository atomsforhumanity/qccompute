"""Tests and fixtures for test xtb."""

from importlib.util import find_spec

import numpy as np
import pytest
from qcdata import CalcType, Model, ProgramInput, SinglePointData, Structure

from qccompute import compute


@pytest.mark.integration
@pytest.mark.skipif(find_spec("xtb") is None, reason="xtb-python is not installed")
def test_xtb() -> None:
    """Test xtb."""
    input_data = ProgramInput(
        program="xtb",
        structure=Structure(
            symbols=["O", "H", "H"],
            geometry=np.array(
                [[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]]
            ),
        ),
        calctype=CalcType.gradient,
        model=Model(method="GFN2xTB"),
        keywords={},
    )

    output = compute(input_data, environment={"OMP_NUM_THREADS": "1"})
    assert output.logs is not None
    assert isinstance(output.results, SinglePointData)
    assert output.results.energy is not None
    assert output.results.gradient is not None
    assert "total charge" in output.logs
    assert np.isclose(output.results.energy, -5.070218272184619, atol=1e-6)
    assert np.allclose(
        output.results.gradient,
        np.array(
            [
                [-0.01079716, -0.0081492, 0.00556273],
                [0.00481361, 0.00628781, -0.00093842],
                [0.00598355, 0.00186139, -0.00462431],
            ]
        ),
        atol=1e-6,
    )
