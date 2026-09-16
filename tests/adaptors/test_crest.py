import pytest
import qccodec
from qccodec.models import NativeInput
from qcdata import SinglePointData

from qccompute.adapters.crest import CRESTAdapter
from qccompute.exceptions import ExternalProgramError


def test_crest_parser_error_preserves_partial_data(
    monkeypatch, prog_input_factory, tmp_path
):
    monkeypatch.chdir(tmp_path)
    partial_data = SinglePointData(
        provenance={"program": "crest", "program_version": "3.0.2"}
    )

    def raise_parser_error(*args, **kwargs):
        raise qccodec.exceptions.ParserError(
            "Missing required output artifact(s): crest.engrad",
            data=partial_data,
        )

    monkeypatch.setattr(
        qccodec,
        "encode",
        lambda *args, **kwargs: NativeInput(
            input_file="",
            geometry_file="",
            geometry_filename="structure.xyz",
        ),
    )
    monkeypatch.setattr(
        "qccompute.adapters.crest.execute_subprocess",
        lambda *args, **kwargs: "Version 3.0.2, test stdout",
    )
    monkeypatch.setattr(qccodec, "decode", raise_parser_error)

    with pytest.raises(ExternalProgramError) as exc_info:
        CRESTAdapter().compute_data(prog_input_factory("gradient", program="crest"))

    assert exc_info.value.data is partial_data


@pytest.mark.parametrize(
    "program,nonzero_exit",
    [("crest", False), ("crest", True), ("terachem", True), ("orca", True)],
)
def test_failed_execution_decodes_partial_data(
    program, nonzero_exit, monkeypatch, prog_input_factory, tmp_path
):
    from pathlib import Path

    from qccompute.adapters import registry

    monkeypatch.chdir(tmp_path)
    module = __import__(
        f"qccompute.adapters.{program}", fromlist=["execute_subprocess"]
    )
    monkeypatch.setattr(
        qccodec, "encode", lambda *args: NativeInput("", "", "geometry.xyz")
    )
    if program == "orca":
        monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/bin/orca")

    def execute(*args):
        if program == "crest":
            Path("crest.engrad").write_text("# Energy ( Eh )\n#\n -1.0\n")
        logs = "Version 3.0.2, FAILED" if program == "crest" else "Execution failed"
        if nonzero_exit:
            raise ExternalProgramError(program, logs=logs)
        return logs

    monkeypatch.setattr(module, "execute_subprocess", execute)
    output = registry[program]().compute(
        prog_input_factory("gradient", program=program), raise_exc=False
    )
    assert not output.success
    assert isinstance(output.results, SinglePointData)
    assert output.results.provenance.program == program
    assert output.results.gradient is None
    assert output.logs
    if program == "crest":
        assert output.results.energy == -1.0
        assert output.results.provenance.program_version == "3.0.2"
