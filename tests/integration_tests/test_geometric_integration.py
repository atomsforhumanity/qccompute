"""Tests and fixtures for test geometric integration."""

from collections.abc import Callable

import pytest
from qcdata import CalcType, OptimizationData, ProgramInput, ProgramOutput, Structure

from qccompute import compute
from tests.conftest import skipif_program_not_available


@pytest.mark.integration
@skipif_program_not_available("terachem")
def test_full_optimization(
    nested_input_factory: Callable[[CalcType], ProgramInput],
) -> None:
    """Test full optimization."""
    prog_input = nested_input_factory(CalcType.optimization)
    prog_input_dict = prog_input.model_dump()
    prog_input_dict["subprograms"][0]["program"] = "terachem"
    prog_input = ProgramInput(**prog_input_dict)

    output = compute(prog_input, propagate_wfn=True)
    assert isinstance(output, ProgramOutput)
    assert isinstance(output.input_data, ProgramInput)
    assert isinstance(output.results, OptimizationData)
    # Ensure wavefunction was propagated
    assert output.results.trajectory[-1].logs is not None
    assert "Initial guess will be loaded from c0" in output.results.trajectory[-1].logs
    # Ensure energy went downhill
    assert output.results.energies[0] > output.results.energies[-1]


@pytest.mark.integration
@skipif_program_not_available("terachem")
def test_full_transition_state(
    nested_input_factory: Callable[[CalcType], ProgramInput], water: Structure
) -> None:
    """Check that transition-state search produces a complete trajectory."""
    # Must use water or else the transition state search will fail
    prog_input = nested_input_factory(CalcType.transition_state)
    prog_input_dict = prog_input.model_dump()
    prog_input_dict["subprograms"][0]["program"] = "terachem"
    prog_input_dict["structure"] = water
    prog_input = ProgramInput(**prog_input_dict)

    # Ensure output was produced
    output = compute(prog_input, propagate_wfn=True)
    assert isinstance(output, ProgramOutput)
    assert isinstance(output.input_data, ProgramInput)
    assert isinstance(output.results, OptimizationData)
    # Ensure wavefunction was propagated
    assert output.results.trajectory[-1].logs is not None
    assert "Initial guess will be loaded from c0" in output.results.trajectory[-1].logs
