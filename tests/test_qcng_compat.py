import pytest

from qccompute import compute
from qccompute.adapters import registry
from qccompute.exceptions import (
    AdapterNotFoundError,
    ExternalProgramError,
    ProgramNotFoundError,
)
from qccompute.utils import check_qcng_support


def test_compute_raises_adapter_not_found_if_no_adapter_in_qcng():
    with pytest.raises(AdapterNotFoundError):
        check_qcng_support("not-a-real-program")


def test_compute_raises_program_not_found_if_adapter_but_no_program_in_qcng():
    with pytest.raises(ProgramNotFoundError):
        check_qcng_support("mrchem")  # Assumes mrchem is not installed


def test_qcng_fallback_tried_if_adapter_not_in_qccompute(mocker, prog_input_factory):
    # WOW: Just a crazy note. This test takes ~1.2 to execute, this is the startup
    # overhead cost of qcng.compute(). Absolutely crazy! It comes from the very slow
    # get_program() call, which takes >1s to execute.

    # So qcng thinks this program is installed and has a harness
    qcng_spy = mocker.patch("qcengine.get_program")

    qcng_adapter_cls = registry["qcengine"]
    compute_spy = mocker.spy(qcng_adapter_cls, "compute")
    # Mock the "program_version" method and set its return value
    mocker.patch.object(
        qcng_adapter_cls, "program_version", return_value="fake-version"
    )

    energy_inp = prog_input_factory("energy", program="mrchem")
    # Program not in qccompute, but in qcn
    # Will raise qcengine.exceptions.ResourceError: Program mrchem is registered with
    # QCEngine, but cannot be found. Test still demonstrates that qcng.compute() is
    # called.
    compute(energy_inp, raise_exc=False)

    assert qcng_spy.call_count == 1
    assert compute_spy.call_count == 1


def test_qcng_compute_not_called_if_adapter_in_qccompute(
    mocker, prog_input_factory, test_adapter
):
    qcng_spy = mocker.patch("qcengine.compute")

    test_adapter = registry["test"]
    compute_spy = mocker.spy(test_adapter, "compute")

    energy_inp = prog_input_factory("energy", program="test")
    compute(energy_inp)

    assert qcng_spy.call_count == 0  # qcng.compute() not called
    assert compute_spy.call_count == 1  # adaptor.compute() called


def test_qcng_exception_wrapping_raise_exc_true(mocker, prog_input_factory):
    # So system check passes for harness and program installation
    mocker.patch("qccompute.utils.check_qcng_support")
    # qcng_spy = mocker.patch("qcengine.compute")
    # qcng_spy.side_effect = QCEngineException("QCEngine Failed!")

    energy_inp = prog_input_factory("energy", program="mrchem")
    # Program not in qccompute, but in qcng
    with pytest.raises(ExternalProgramError):
        compute(energy_inp, raise_exc=True)


def test_qcng_exception_wrapping_raise_exc_false(mocker, prog_input_factory):
    # So system check passes for harness and program installation
    mocker.patch("qccompute.utils.check_qcng_support")
    qcng_adapter = registry["qcengine"]
    mocker.spy(qcng_adapter, "compute")
    # Mock the "program_version" method and set its return value
    mocker.patch.object(qcng_adapter, "program_version", return_value="fake-version")

    energy_inp = prog_input_factory("energy", program="mrchem")
    # Program not in qccompute, but in qcng
    output = compute(energy_inp, raise_exc=False)
    assert output.success is False
    assert isinstance(output.traceback, str)
    assert output.input_data == energy_inp


def test_qcengine_success_retains_external_producer(mocker, prog_input_factory):
    from qcdata.qcel import to_qcel_input

    input_data = prog_input_factory("energy", program="psi4")
    atomic_result = dict(
        **to_qcel_input(input_data),
        success=True,
        return_result=-1.0,
        properties={"return_energy": -1.0},
        provenance={"creator": "Psi4", "version": "1.9", "routine": "energy"},
        stdout="computed energy",
    )
    mocker.patch("qccompute.utils.check_qcng_support")
    run = mocker.patch("qcengine.compute", return_value=atomic_result)
    output = compute(input_data)
    assert output.input_data.program == "psi4"
    assert output.results.provenance.program == "Psi4"
    assert output.results.provenance.program_version == "1.9"
    assert output.results.energy == -1.0
    assert output.execution.wall_time is not None
    assert run.call_args.args[1] == "psi4"
