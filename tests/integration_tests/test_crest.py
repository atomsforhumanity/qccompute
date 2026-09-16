import numpy as np
import pytest
from qcdata import ProgramInput, Structure

from qccompute import compute
from tests.conftest import skipif_program_not_available


@pytest.mark.integration
@skipif_program_not_available("crest")
def test_crest():
    input_data = ProgramInput(
        program="crest",
        structure=Structure(
            symbols=["O", "H", "H"],
            geometry=[[0.0, 0.0, 0.0], [0.524, 1.687, 0.48], [1.146, -0.45, -1.354]],
        ),
        calctype="conformer_search",
        model={"method": "gfnff"},
        keywords={"calculation": {"level": [{"alpb": "acetonitrile"}]}},
    )

    prog_output = compute(input_data)

    assert len(prog_output.results.conformers) == 1
    assert np.isclose(prog_output.results.conformer_energies[0], -0.33568982, atol=1e-4)
    # Solvent getting passed correctly
    assert "alpb" in prog_output.logs
    assert "acetonitrile" in prog_output.logs
