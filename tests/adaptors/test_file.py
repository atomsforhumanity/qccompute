import pytest
from qcdata import FileInput

from qccompute.adapters import FileAdapter
from qccompute.adapters.utils import tmpdir
from qccompute.exceptions import ProgramNotFoundError


def test_file_adapter_compute(tmp_path):
    # Create FileInput
    file_inp = FileInput(program="python", cmdline_args=["hello_world.py"])
    file_inp.files["hello_world.py"] = "print('hello world')"
    with tmpdir() as path:
        file_inp.save_files(path)
        _, stdout = FileAdapter("python").compute_data(file_inp)
    assert stdout == "hello world\n"


def test_file_adapter_raises_exception_if_program_fails():
    # Create FileInput
    file_inp = FileInput(program="python")
    with pytest.raises(ProgramNotFoundError):
        FileAdapter("does_not_exist").compute_data(file_inp)
