import pytest
from qcdata import CalcType, ProgramOutput, SinglePointData

from qccompute.adapters import base, registry
from qccompute.exceptions import AdapterInputError, ExternalProgramError


def test_adapter_subclasses_must_define_program():
    """Test that subclasses of QCComputeProgramAdapter must define a nonempty
    program list."""
    with pytest.raises(NotImplementedError):

        class TestAdapter(base.ProgramAdapter):
            # supported_calctypes defined but program not defined
            supported_calctypes = [CalcType.energy]

            def compute_data(self, *args, **kwargs):
                pass


def test_adapter_subclasses_must_define_supported_calctypes():
    """Test that subclasses of QCComputeSinglePointAdapter must define a nonempty
    supported_calctypes list."""
    with pytest.raises(NotImplementedError):

        class TestAdapter(base.ProgramAdapter):
            # program defined but supported_calctypes not defined
            program = "test"

            def compute_data(self, *args, **kwargs):
                pass


def test_adapter_subclasses_must_define_compute_method():
    """Test that subclasses of QCComputeProgramAdapter defining a nonempty program
    list and supported_calctypes list can be instantiated."""

    class TestAdapter(base.ProgramAdapter):
        # Both program and supported_calctypes defined
        program = "test"
        supported_calctypes = [CalcType.energy]

    with pytest.raises(TypeError):
        TestAdapter()


def test_adapter_subclasses_defining_program_and_supported_calctypes():
    """Test that subclasses of QCComputeProgramAdapter defining a nonempty program
    list and supported_calctypes list can be instantiated."""

    class TestAdapter(base.ProgramAdapter):
        # Both program and supported_calctypes defined
        program = "test"
        supported_calctypes = [CalcType.energy]

        def compute_data(self, *args, **kwargs):
            pass

    assert registry.get("test") == TestAdapter


def test_adapters_raise_error_if_calctype_not_supported(prog_input_factory):
    """Test that adapters raise an error if the calctype is not supported."""

    class TestAdapter(base.ProgramAdapter):
        # Both program and supported_calctypes defined
        program = "test"
        supported_calctypes = [CalcType.energy]

        def program_version(self, *args, **kwargs) -> str:
            return "v1.0.0"

        def compute_data(self, *args, **kwargs):
            pass

    gradient_input = prog_input_factory("gradient", program="test")
    with pytest.raises(AdapterInputError):
        TestAdapter().compute(gradient_input)


def test_data_added_to_results_object_if_exception_contains_it(
    prog_input_factory, mocker, results, test_adapter
):
    """Test that results are added to the ProgramOutput object if the exception
    contains them."""
    test_adapter = registry["test"]()

    def raise_error(*args, **kwargs):
        raise ExternalProgramError(
            program="terachem", logs="some stdout", data=results.results
        )

    mocker.patch.object(
        test_adapter,
        "compute_data",
        side_effect=raise_error,
    )
    energy_input = prog_input_factory("energy", program="test")

    # Check that the exception object contains the results
    with pytest.raises(ExternalProgramError) as excinfo:
        test_adapter.compute(energy_input, raise_exc=True)
    assert excinfo.value.data == results.results

    # If no raise_exc=False, the results are added to the ProgramOutput
    computed_results = test_adapter.compute(energy_input, raise_exc=False)
    assert isinstance(computed_results, ProgramOutput)
    assert computed_results.results == results.results


def test_results_object_added_to_exception(
    prog_input_factory, mocker, results, test_adapter
):
    """Test that exceptions contain the ProgramOutput object."""
    test_adapter = registry["test"]()

    def raise_error(*args, **kwargs):
        raise ExternalProgramError(
            program="terachem",
            logs="some stdout",
            data=results.results,
        )

    mocker.patch.object(
        test_adapter,
        "compute_data",
        side_effect=raise_error,
    )
    energy_input = prog_input_factory("energy", program="test")

    # Check that the exception object contains the results
    with pytest.raises(ExternalProgramError) as excinfo:
        test_adapter.compute(energy_input, raise_exc=True)

    assert isinstance(excinfo.value.prog_output, ProgramOutput)
    assert excinfo.value.prog_output.success is False
    # NOTE: CHECK WITH BIGCHEM
    assert isinstance(excinfo.value.args[-1], SinglePointData)


def test_stdout_collected_with_failed_execution(
    prog_input_factory, mocker, results, test_adapter
):
    """Test that stdout is collected even if the execution fails."""
    test_adapter = registry["test"]()

    def raise_error(*args, **kwargs):
        raise ExternalProgramError(
            program="terachem",
            logs="some stdout",
            data=results.results,
        )

    mocker.patch.object(
        test_adapter,
        "compute_data",
        side_effect=raise_error,
    )
    energy_input = prog_input_factory("energy", program="test")

    # Check that the exception object contains the results
    with pytest.raises(ExternalProgramError) as excinfo:
        test_adapter.compute(energy_input, raise_exc=True)

    # Added to exception
    assert excinfo.value.logs == "some stdout"
    # Added to ProgramOutput
    assert excinfo.value.prog_output.logs == "some stdout"
    # Added to exception
    assert excinfo.value.data == results.results


def test_collect_wfn_raises_adapter_input_error_if_not_implemented(test_adapter):
    """Test that collect_wfn raises an AdapterInputError if not implemented."""
    with pytest.raises(AdapterInputError):
        test_adapter.collect_wfn()


def test_parsed_producer_is_preserved(test_adapter, prog_input_factory, mocker):
    from qcdata import SinglePointData

    data = SinglePointData.model_validate(
        {
            "provenance": {"program": "actual-engine", "program_version": "2.0"},
            "energy": -1.0,
        }
    )
    mocker.patch.object(test_adapter, "compute_data", return_value=(data, "logs"))
    version_lookup = mocker.spy(test_adapter, "program_version")
    output = test_adapter.compute(prog_input_factory("energy"))
    assert output.input_data.program == "test"
    assert output.results.provenance == data.provenance
    assert output.execution.wall_time is not None
    version_lookup.assert_not_called()
