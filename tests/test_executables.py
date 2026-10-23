"""Choosing which executable runs: overrides, environment variables, and PATH."""

from __future__ import annotations

import shutil
import stat
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import qccodec
from qccodec import Codec, NativeOutput, SubprocessInput
from qcdata import (
    CalcType,
    OptimizationData,
    ProgramInput,
    ProgramSpec,
    SinglePointData,
    Structure,
)

from qccompute import compute
from qccompute.exceptions import ProgramNotFoundError
from qccompute.execution import executable_variable


def script(path: Path, marker: str) -> Path:
    """Write an executable shell script that prints ``marker``."""
    path.write_text(f"#!/bin/sh\necho {marker}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def codec_running(command: str, program: str = "test") -> Codec:
    """A codec that runs ``command`` and reports a constant energy."""
    return Codec(
        program,
        calctypes=list(CalcType),
        encode=lambda inp: SubprocessInput((command,)),
        decode=lambda inp, out: SinglePointData.model_validate(
            {"provenance": {"program": program}, "energy": -1.0}
        ),
    )


def test_variable_name() -> None:
    """Program names become upper-case environment variable names."""
    assert executable_variable("orca") == "QCCOMPUTE_ORCA_EXECUTABLE"
    assert executable_variable("g-xtb") == "QCCOMPUTE_G_XTB_EXECUTABLE"


def test_environment_variable_selects_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """QCCOMPUTE_<PROGRAM>_EXECUTABLE replaces the codec's command."""
    program = script(tmp_path / "from-variable", "variable-ran")
    monkeypatch.setenv("QCCOMPUTE_TEST_EXECUTABLE", str(program))
    output = compute(prog_input_factory("energy"), codec=codec_running("missing-cmd"))
    assert output.logs is not None and "variable-ran" in output.logs
    assert output.execution.extras["executable"] == str(program)


def test_mapping_wins_over_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """An explicit executables entry takes precedence over the variable."""
    monkeypatch.setenv(
        "QCCOMPUTE_TEST_EXECUTABLE", str(script(tmp_path / "variable", "variable"))
    )
    chosen = script(tmp_path / "mapping", "mapping-ran")
    output = compute(
        prog_input_factory("energy"),
        codec=codec_running("missing-cmd"),
        executables={"test": str(chosen), "unused": "/nowhere"},
    )
    assert output.logs is not None and "mapping-ran" in output.logs
    assert output.execution.extras["executable"] == str(chosen)


def test_program_on_path_runs_by_absolute_path(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Without overrides the PATH match runs, recorded as an absolute path."""
    output = compute(prog_input_factory("energy"), codec=codec_running("true"))
    assert output.execution.extras["executable"] == shutil.which("true")


def test_relative_override_is_relative_to_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """A relative override is resolved before changing into the scratch directory."""
    script(tmp_path / "local-program", "relative-ran")
    monkeypatch.chdir(tmp_path)
    output = compute(
        prog_input_factory("energy"),
        codec=codec_running("missing-cmd"),
        executables={"test": "./local-program"},
    )
    assert output.logs is not None and "relative-ran" in output.logs
    assert output.execution.extras["executable"] == str(tmp_path / "local-program")


@pytest.mark.parametrize("via", ["mapping", "variable"])
def test_bad_override_names_its_source(
    via: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """A missing or non-executable override fails before launch, naming its source."""
    not_executable = tmp_path / "plain-file"
    not_executable.write_text("")
    executables: dict[str, str] | None = None
    if via == "mapping":
        executables = {"test": str(not_executable)}
        source = r"executables\['test'\]"
    else:
        monkeypatch.setenv("QCCOMPUTE_TEST_EXECUTABLE", str(not_executable))
        source = "QCCOMPUTE_TEST_EXECUTABLE"
    with pytest.raises(ProgramNotFoundError, match=source) as raised:
        compute(
            prog_input_factory("energy"),
            codec=codec_running("true"),
            executables=executables,
        )
    assert raised.value.prog_output is not None
    assert not raised.value.prog_output.success


def test_executables_must_be_strings(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Non-string paths are rejected as input errors."""
    from qccompute.exceptions import InputError

    with pytest.raises(InputError, match="executables"):
        compute(
            prog_input_factory("energy"),
            codec=codec_running("true"),
            executables={"test": Path("/bin/true")},  # type: ignore[dict-item]
        )


def test_children_inherit_executables(tmp_path: Path) -> None:
    """geomeTRIC's child calculations run the overridden executable."""
    pytest.importorskip("geometric")
    program = script(tmp_path / "child-program", "child-ran")

    def decode(inp: ProgramInput, native: NativeOutput) -> SinglePointData:
        """Return a harmonic bond energy and gradient once the script has run."""
        assert "child-ran" in (native.stdout or "")
        geometry = inp.require_structure().geometry
        delta = geometry[1] - geometry[0]
        distance = float(np.linalg.norm(delta))
        gradient = (distance - 1.4) * delta / distance
        return SinglePointData.model_validate(
            {
                "provenance": {"program": "scripted"},
                "energy": 0.5 * (distance - 1.4) ** 2,
                "gradient": np.array([-gradient, gradient]),
            }
        )

    child = Codec(
        "scripted",
        calctypes={CalcType.gradient},
        encode=lambda inp: SubprocessInput(("missing-child-cmd",)),
        decode=decode,
    )
    inp = ProgramInput(
        program="geometric",
        calctype=CalcType.optimization,
        structure=Structure(
            symbols=["H", "H"], geometry=np.array([[0, 0, 0], [0, 0, 1.8]])
        ),
        keywords={"coordsys": "cart", "maxiter": 30},
        subprograms=[ProgramSpec(program="scripted", calctype=CalcType.gradient)],
    )
    qccodec.register(child)
    try:
        output = compute(inp, executables={"scripted": str(program)})
    finally:
        qccodec.unregister("scripted")
    assert isinstance(output.results, OptimizationData)
    assert output.results.frames
    for step in output.results.frames:
        assert step.execution.extras["executable"] == str(program)
