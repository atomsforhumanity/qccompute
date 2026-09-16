import numpy as np
import pytest
from qcdata import ProgramInput, Structure

from qccompute import compute
from tests.conftest import skipif_program_not_available


@pytest.mark.integration
@skipif_program_not_available("xtb")
def test_xtb():
    input_data = ProgramInput(
        program="xtb",
        structure=Structure(
            symbols=["O", "H", "H"],
            geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
        ),
        calctype="gradient",
        model={"method": "GFN2xTB"},
        keywords={},
    )

    output = compute(input_data)
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
