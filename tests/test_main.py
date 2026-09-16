"""Many of these tests are really testing the BaseAdapter.compute() method because I
refactored the compute() method to be in the BaseAdapter class. This works for now.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from qcdata import (
    ConformerSearchData,
    FileData,
    FileInput,
    Files,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ScanData,
    SinglePointData,
)

from qccompute.adapters import registry
from qccompute.exceptions import (
    AdapterInputError,
    AdapterNotFoundError,
    QCComputeBaseError,
)
from qccompute.main import compute, compute_args


def test_file_adapter_works_inside_top_level_compute_function():
    # Create FileInput
    file_inp = FileInput(program="python", cmdline_args=["hello_world.py"])
    file_inp.files["hello_world.py"] = "print('hello world')"
    prog_out = compute(file_inp)
    assert isinstance(prog_out, ProgramOutput)
    assert prog_out.logs == "hello world\n"
    assert isinstance(prog_out.results, FileData)
    assert prog_out.results.provenance.program == "python"
    assert prog_out.execution.wall_time is not None


def test_compute_raises_adapter_not_found_error(prog_input_factory):
    """Test that compute raises an AdapterNotFoundError if the adapter is not
    found."""
    energy_inp = prog_input_factory("energy", program="not-a-real-program")
    with pytest.raises(AdapterNotFoundError):
        # Will check qccompute and qcng
        compute(energy_inp)


def test_compute_raises_adapter_not_found_error_no_qcng_fallback(prog_input_factory):
    """Test that compute raises an AdapterNotFoundError if the adapter is not
    found."""
    energy_inp = prog_input_factory("energy", program="not-a-real-program")
    with pytest.raises(AdapterNotFoundError):
        compute(energy_inp, qcng_fallback=False)


def test_print_logs(test_adapter, prog_input_factory, mocker):
    energy_inp = prog_input_factory("energy", program="test")

    spy = mocker.spy(type(test_adapter), "compute_data")

    compute(energy_inp)
    assert spy.call_args.args[2] is None  # update_func
    compute(energy_inp, print_logs=True)
    assert isinstance(spy.call_args.args[2], Callable)  # update_func passed
    assert isinstance(spy.call_args.args[3], float)  # update_interval passed


def test_update_func_preferred_over_print_logs(
    test_adapter, prog_input_factory, mocker
):
    energy_inp = prog_input_factory("energy", program="test")

    spy = mocker.spy(type(test_adapter), "compute_data")

    def update_func(stdout, stderr):
        pass

    compute(energy_inp, update_func=update_func, print_logs=True)
    assert spy.call_args.args[2] == update_func  # update_func passed
    assert spy.call_args.args[3] is None


def test_compute_uses_files_if_adaptor_uses_files_set(prog_input_factory):
    """Test that compute writes files if the adaptor has uses_files set to True."""

    # Set adapter.uses_files is True by default
    energy_inp = prog_input_factory("energy", program="test")
    filename, contents = "hello_world.py", "print('hello world')"
    energy_inp.files[filename] = contents

    result = compute(energy_inp, rm_scratch_dir=False)
    with open(Path(result.execution.scratch_dir) / filename) as f:
        assert f.read() == contents


def test_compute_does_not_uses_files_if_adaptor_uses_files_set(prog_input_factory):
    """Test that compute writes files if the adaptor has uses_files set to True."""

    # Set adapter.uses_files to False
    adapter = registry["test"]
    adapter.uses_files = False

    energy_inp = prog_input_factory("energy", program="test")
    filename, contents = "hello_world.py", "print('hello world')"
    energy_inp.files[filename] = contents
    with pytest.raises(AdapterInputError):
        compute(energy_inp, rm_scratch_dir=False)


def test_compute_raises_exception_if_program_fails_raise_exec_true(prog_input_factory):
    """Test that compute raises an exception if the program fails."""
    opt_input = prog_input_factory("optimization", program="test")

    with pytest.raises(AdapterInputError):
        compute(opt_input, raise_exc=True)


def test_compute_does_not_raise_exception_if_raise_exec_false(
    prog_input_factory, mocker
):
    """Test that compute does not raise an exception if the program fails."""
    grad_input = prog_input_factory("energy", program="test")
    adapter = registry["test"]
    mocker.patch.object(
        adapter,
        "compute_data",
        side_effect=QCComputeBaseError("Something failed!"),
    )
    po = compute(grad_input, raise_exc=False)
    assert po.success is False


def test_qcengine_import_error(mocker, prog_input_factory):
    """Test that an ImportError is raised when qcengine is not installed."""
    # Mock sys.modules to simulate qcengine not being installed
    mocker.patch.dict("sys.modules", {"qcengine": None})

    with pytest.raises(ModuleNotFoundError):
        # The code that attempts to import qcengine goes here
        energy_inp = prog_input_factory("energy", program="no-adaptor-for-program")
        compute(energy_inp, qcng_fallback=True)


def test_compute_args(hydrogen, mocker):
    """Test that compute_args correctly constructs input object and calls compute."""
    # Spy on top level compute function
    compute_spy = mocker.patch("qccompute.main.compute")
    values_dict = {
        "structure": hydrogen,
        "calctype": "energy",
        "model": {"method": "HF", "basis": "sto-3g"},
        "keywords": {"fake": "things"},
        "files": {"fake.py": "print('hello world')"},
        "extras": {"fake": "things"},
    }
    compute_args("test", extra_thing=123, **values_dict)
    compute_spy.assert_called_once_with(
        ProgramInput(program="test", **values_dict), extra_thing=123
    )


def test_compute_args_file_object_passed(hydrogen, mocker):
    """Test that compute_args correctly constructs input object and calls compute."""
    # Spy on top level compute function
    compute_spy = mocker.patch("qccompute.main.compute")
    values_dict = {
        "structure": hydrogen,
        "calctype": "energy",
        "model": {"method": "HF", "basis": "sto-3g"},
        "keywords": {"fake": "things"},
        "files": Files(files={"fake.py": "print('hello world')"}),
        "extras": {"fake": "things"},
    }
    compute_args("test", extra_thing=123, **values_dict)
    # Convert back for ProgramInput instantiation
    values_dict["files"] = {"fake.py": "print('hello world')"}
    compute_spy.assert_called_once_with(
        ProgramInput(program="test", **values_dict), extra_thing=123
    )


@pytest.mark.parametrize(
    "calctype,data_type",
    [
        ("energy", SinglePointData),
        ("gradient", SinglePointData),
        ("hessian", SinglePointData),
        ("optimization", OptimizationData),
        ("transition_state", OptimizationData),
        ("conformer_search", ConformerSearchData),
        ("scan", ScanData),
    ],
)
def test_lookup_failure_retains_requested_data_type(
    prog_input_factory, calctype, data_type
):
    input_data = prog_input_factory(calctype, program="not-a-program")
    output = compute(input_data, qcng_fallback=False, raise_exc=False)
    assert not output.success
    assert isinstance(output.results, data_type)
    assert output.results.provenance.program == input_data.program
    assert output.results.provenance.program_version is None
    assert output.execution.wall_time is not None
    assert ProgramOutput.model_validate_json(output.model_dump_json()) == output


def test_leaf_adapter_requires_model(prog_input_factory):
    input_data = ProgramInput.model_validate(
        {**prog_input_factory("energy").model_dump(), "model": None}
    )
    output = compute(input_data, raise_exc=False)
    assert not output.success
    assert isinstance(output.results, SinglePointData)
    assert "requires a scientific model" in output.traceback


def test_incompatible_tcpb_returns_actionable_failure(prog_input_factory, mocker):
    mocker.patch("qccompute.adapters.terachem_fe.importlib.import_module")
    mocker.patch("qccompute.adapters.terachem_fe.version", return_value="0.16.0")
    output = compute(
        prog_input_factory("energy", program="terachem-fe"), raise_exc=False
    )
    assert not output.success
    assert isinstance(output.results, SinglePointData)
    assert "tcpb 0.16.0 constructs legacy qcdata outputs" in output.traceback
