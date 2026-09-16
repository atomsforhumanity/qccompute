from pathlib import Path

import pytest

from qccompute.adapters import TeraChemAdapter
from qccompute.adapters.utils import tmpdir
from qccompute.exceptions import AdapterError, AdapterInputError


def test_get_version_no_stdout(monkeypatch):
    """Test get_version method."""
    adapter = TeraChemAdapter()
    # Monkey patch execute_subprocess to return a version string
    monkeypatch.setattr(
        "qccompute.adapters.terachem.execute_subprocess",
        lambda *args: (
            "TeraChem version 1.9-2021.12-dev "
            "[cc068f92bd92d3e9009c54834f0d63e9cfec3273]"
        ),
    )
    assert (
        adapter.program_version()
        == "1.9-2021.12-dev [cc068f92bd92d3e9009c54834f0d63e9cfec3273]"
    )


def test_get_version_stdout(mocker):
    """Test get_version method."""
    adapter = TeraChemAdapter()
    # Create a mock for parse_version_string
    mock_parse_version = mocker.patch("qccompute.adapters.terachem.parse_version")

    adapter.program_version("some stdout data")

    # Assert that parse_version_string was called once
    mock_parse_version.assert_called_once()


def test_propagate_wfn(prog_input_factory, results):
    """Test propagate_wavefunction method."""
    prog_input_factory_inst = prog_input_factory("energy", program="terachem")
    adapter = TeraChemAdapter()

    # Raises error if output does not contain wavefunction data
    with pytest.raises(AdapterInputError):
        adapter.propagate_wfn(results, prog_input_factory_inst)

    # TeraChem Output conventions
    scr_postfix = "geometry".split(".")[0]
    scr_dir = f"scr.{scr_postfix}"

    # Add restricted wavefunction data to output
    results.results.files[f"{scr_dir}/c0"] = "some file"
    updated_input = adapter.propagate_wfn(results, prog_input_factory_inst)
    assert not prog_input_factory_inst.files
    assert updated_input.files["c0"] == "some file"
    assert updated_input.keywords["guess"] == "c0"
    results.results.files.pop(f"{scr_dir}/c0")  # Remove c0 from output

    # Add unrestricted wavefunction data to output
    prog_input_factory_inst = prog_input_factory("energy", program="terachem")
    results.results.files[f"{scr_dir}/ca0"] = "some alpha file"
    results.results.files[f"{scr_dir}/cb0"] = "some beta file"
    updated_input = adapter.propagate_wfn(results, prog_input_factory_inst)
    assert not prog_input_factory_inst.files
    assert updated_input.files["ca0"] == "some alpha file"
    assert updated_input.files["cb0"] == "some beta file"
    assert updated_input.keywords["guess"] == "ca0 cb0"

    # Assert error raised if only alpha or beta wavefunction data is present
    results.results.files.pop(f"{scr_dir}/cb0")
    with pytest.raises(AdapterInputError):
        adapter.propagate_wfn(results, prog_input_factory_inst)

    results.results.files.pop(f"{scr_dir}/ca0")
    results.results.files[f"{scr_dir}/cb0"] = "some beta file"
    with pytest.raises(AdapterInputError):
        adapter.propagate_wfn(results, prog_input_factory_inst)


def test_collect_wfn(results):
    # Raises AdapterError if not wavefunction data in output
    adapter = TeraChemAdapter()
    with pytest.raises(AdapterError):
        adapter.collect_wfn()

    # Check collection of c0
    scr_dir_str = "scr.geometry"
    with tmpdir():
        scr_dir = Path(scr_dir_str)
        scr_dir.mkdir()
        (scr_dir / "c0").write_bytes(b"some data")
        wfns = adapter.collect_wfn()
        assert wfns[f"{scr_dir_str}/c0"] == b"some data"

    with tmpdir():
        scr_dir = Path(scr_dir_str)
        scr_dir.mkdir()
        (scr_dir / "ca0").write_bytes(b"some alpha data")
        (scr_dir / "cb0").write_bytes(b"some beta data")
        wfns = adapter.collect_wfn()
        assert wfns[f"{scr_dir_str}/ca0"] == b"some alpha data"
        assert wfns[f"{scr_dir_str}/cb0"] == b"some beta data"
