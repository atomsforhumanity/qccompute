import pytest
from qcdata import CalcType, OptimizationData, ProgramInput, ProgramOutput

from qccompute.adapters import GeometricAdapter
from tests.conftest import skipif_program_not_available


@pytest.mark.integration
@skipif_program_not_available("terachem")
def test_full_optimization(nested_input_factory):
    prog_input = nested_input_factory(CalcType.optimization)
    prog_input_dict = prog_input.model_dump()
    prog_input_dict["subprograms"][0]["program"] = "terachem"
    prog_input = ProgramInput(**prog_input_dict)

    adapter = GeometricAdapter()
    output = adapter.compute(prog_input, propagate_wfn=True)
    assert isinstance(output, ProgramOutput)
    assert isinstance(output.input_data, ProgramInput)
    assert isinstance(output.results, OptimizationData)
    # Ensure wavefunction was propagated
    assert "Initial guess will be loaded from c0" in output.results.trajectory[-1].logs
    # Ensure energy went downhill
    assert output.results.energies[0] > output.results.energies[-1]


@pytest.mark.integration
@skipif_program_not_available("terachem")
def test_full_transition_state(nested_input_factory, water):
    """This test lacks any test of correctness, but it does ensure that the
    transition state search does not fail.
    """
    # Must use water or else the transition state search will fail
    prog_input = nested_input_factory(CalcType.transition_state)
    prog_input_dict = prog_input.model_dump()
    prog_input_dict["subprograms"][0]["program"] = "terachem"
    prog_input_dict["structure"] = water
    prog_input = ProgramInput(**prog_input_dict)

    adapter = GeometricAdapter()
    # Ensure output was produced
    output = adapter.compute(prog_input, propagate_wfn=True)
    assert isinstance(output, ProgramOutput)
    assert isinstance(output.input_data, ProgramInput)
    assert isinstance(output.results, OptimizationData)
    # Ensure wavefunction was propagated
    assert "Initial guess will be loaded from c0" in output.results.trajectory[-1].logs
