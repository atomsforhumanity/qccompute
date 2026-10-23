"""Verify the complete lifecycle independently of scientific executables."""

from __future__ import annotations

import pickle
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import qccodec
from qccodec import (
    Codec,
    NativeInput,
    NativeOutput,
    PythonInput,
    SubprocessInput,
)
from qccodec.exceptions import CalculationError, EncoderError, ParserError
from qcdata import (
    CalcType,
    FileInput,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
    StructuredData,
    get_data_type,
)

from qccompute import compute, compute_args
from qccompute.exceptions import (
    CodecNotFoundError,
    ExternalProgramError,
    InputError,
    ProgramNotFoundError,
    ProgramTimeoutError,
    QCComputeBaseError,
)


def local_codec(
    request: NativeInput,
    decode: Callable[[ProgramInput, NativeOutput], StructuredData] | None = None,
) -> Codec:
    """Build a standalone codec without discovery or registration."""
    return Codec(
        "test",
        calctypes=list(CalcType),
        encode=lambda inp: request,
        decode=decode
        or (
            lambda inp, out: SinglePointData.model_validate(
                {"provenance": {"program": "actual-producer"}, "energy": -1}
            )
        ),
    )


def test_external_python_codec(
    prog_input_factory: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """External python codec."""

    def invoke(x: int, *, scale: int) -> int:
        """Exercise the callable boundary with deterministic native behavior."""
        print("native logs")
        return x * scale

    codec = local_codec(
        PythonInput(invoke, (2,), {"scale": 3}),
        lambda inp, out: SinglePointData.model_validate(
            {"provenance": {"program": "external"}, "energy": out.value}
        ),
    )
    monkeypatch.setattr(
        "qccodec.get_codec",
        lambda *a, **k: pytest.fail("External codec triggered discovery"),
    )
    output = compute(prog_input_factory("energy"), codec=codec)
    assert isinstance(output.results, SinglePointData)
    assert output.results.energy == 6
    assert output.results.provenance.program == "external"
    assert output.logs == "native logs\n"
    assert output.execution.scratch_dir is not None
    assert not output.execution.scratch_dir.exists()
    assert pickle.loads(pickle.dumps(output)) == output


def test_script_codec_end_to_end(
    prog_input_factory: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codec defined in a script runs explicitly and, once registered, by name."""
    codec = Codec("myprog", calctypes={CalcType.energy})

    @codec.encoder
    def encode(inp: ProgramInput) -> SubprocessInput:
        """Fake a program that prints its result."""
        assert inp.model is not None
        return SubprocessInput(
            (sys.executable, "-c", "print('Total energy: -2.5')"),
            files={
                "run.inp": f"{inp.model.method}\n{inp.require_structure().to_xyz()}"
            },
        )

    @codec.parser(target="energy", calctypes=[CalcType.energy])
    def energy(stdout: str) -> float:
        """Parse a scalar directly from the supplied native contents."""
        return float(
            qccodec.re_search(r"Total energy:\s+(-?\d+\.\d+)", stdout).group(1)
        )

    inp = prog_input_factory("energy", program="myprog")
    output = compute(inp, codec=codec)
    assert isinstance(output.results, SinglePointData)
    assert output.results.energy == -2.5
    assert output.results.provenance.program == "myprog"

    with pytest.raises(CodecNotFoundError):
        compute(inp)
    qccodec.register(codec)
    try:
        by_name = compute(inp).results
        assert isinstance(by_name, SinglePointData) and by_name.energy == -2.5
    finally:
        qccodec.unregister("myprog")


def test_options_reach_the_codec(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Decode options flow to the codec's parsers and unknown keys are input errors."""
    codec = Codec(
        "test",
        calctypes={CalcType.energy},
        options={"natoms": None},
        encode=lambda inp: SubprocessInput((sys.executable, "-c", "print(-1.0)")),
    )
    codec.parser(target="energy")(lambda stdout: float(stdout))
    codec.parser(file="directory")(
        lambda ctx: (
            {"calcinfo_natoms": ctx.options["natoms"]} if ctx.options["natoms"] else {}
        )
    )
    inp = prog_input_factory("energy")
    default = compute(inp, codec=codec).results
    assert isinstance(default, SinglePointData) and default.calcinfo_natoms is None
    chosen = compute(inp, codec=codec, options={"natoms": 2}).results
    assert isinstance(chosen, SinglePointData) and chosen.calcinfo_natoms == 2
    with pytest.raises(InputError, match="does not accept options"):
        compute(inp, codec=codec, options={"typo": 1})


@pytest.mark.parametrize("calctype", list(CalcType))
def test_lookup_failure_preserves_result_contract(
    prog_input_factory: Callable[..., ProgramInput],
    calctype: CalcType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown program fails cleanly with the right data type."""
    with pytest.raises(CodecNotFoundError) as raised:
        compute(prog_input_factory(calctype, program="missing"))
    output = raised.value.prog_output
    assert output is not None and not output.success
    assert type(output.results) is get_data_type(calctype)
    assert ProgramOutput.model_validate_json(output.model_dump_json()) == output


def test_failure_recovers_data_logs_and_artifacts(
    prog_input_factory: Callable[..., ProgramInput], tmp_path: Path
) -> None:
    """Failure recovers data logs and artifacts."""
    request = SubprocessInput(
        (
            sys.executable,
            "-c",
            "from pathlib import Path; print('partial'); Path('answer').write_text('-3'); raise RuntimeError('failed')",
        )
    )

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        """Interpret the fake native result or deliberately exercise a decoding failure."""
        assert out.failed
        assert out.directory is not None
        return SinglePointData.model_validate(
            {
                "provenance": {"program": "test"},
                "energy": float((out.directory / "answer").read_text()),
            }
        )

    output = compute(
        prog_input_factory("gradient"),
        codec=local_codec(request, decode),
        collect_logs=False,
        collect_files=True,
        scratch_dir=tmp_path / "job",
        raise_exc=False,
    )
    assert isinstance(output.results, SinglePointData)
    assert not output.success and output.results.energy == -3
    assert output.results.files["answer"] == "-3"
    assert output.logs is not None
    assert "partial" in output.logs and "failed" in output.logs
    assert output.execution.scratch_dir is not None
    assert not output.execution.scratch_dir.exists()


def test_parser_error_attaches_partial_output(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Parser error attaches partial output."""
    partial = SinglePointData.model_validate(
        {"provenance": {"program": "test"}, "energy": -1}
    )

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        """Interpret the fake native result or deliberately exercise a decoding failure."""
        raise ParserError("missing gradient", data=partial)

    with pytest.raises(QCComputeBaseError) as raised:
        compute(
            prog_input_factory("gradient"),
            codec=local_codec(PythonInput(lambda: None), decode),
        )
    assert raised.value.prog_output is not None
    assert raised.value.prog_output.results == partial
    assert raised.value.prog_output is not None
    assert not raised.value.prog_output.success
    assert (
        pickle.loads(pickle.dumps(raised.value)).prog_output == raised.value.prog_output
    )


def test_invalid_success_becomes_failed_output(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Invalid success becomes failed output."""
    output = compute(
        prog_input_factory("gradient"),
        codec=local_codec(PythonInput(lambda: None)),
        raise_exc=False,
    )
    assert isinstance(output.results, SinglePointData)
    assert not output.success and output.results.energy == -1
    assert output.traceback is not None
    assert "gradient" in output.traceback


def test_encoding_and_staging_errors_are_wrapped(
    prog_input_factory: Callable[..., ProgramInput], tmp_path: Path
) -> None:
    """Encoding and staging errors are wrapped."""

    def encode(inp: ProgramInput) -> NativeInput:
        """Exercise translation failure before invoking any scientific program."""
        raise EncoderError("bad input")

    output = compute(
        prog_input_factory("energy"),
        codec=Codec("test", calctypes=[CalcType.energy], encode=encode),
        raise_exc=False,
    )
    assert output.traceback is not None
    assert not output.success and "bad input" in output.traceback
    request = PythonInput(lambda: None, files={"../escape": "bad"})
    output = compute(
        prog_input_factory("energy"),
        codec=local_codec(request),
        scratch_dir=tmp_path / "job",
        raise_exc=False,
    )
    assert not output.success and (not (tmp_path / "escape").exists())
    assert not (tmp_path / "job").exists()


def test_file_input_and_compute_args(
    prog_input_factory: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """File input and compute args."""
    inp = FileInput(
        program=sys.executable,
        cmdline_args=["hello.py"],
        files={"hello.py": "print('hello')"},
    )
    output = compute(inp)
    assert output.success and output.logs == "hello\n"
    captured = []
    monkeypatch.setattr(
        "qccompute.main.compute", lambda inp, **kw: captured.append((inp, kw))
    )
    source = prog_input_factory("energy")
    compute_args(
        "test",
        source.require_structure(),
        calctype="energy",
        model=source.model,
        environment={"X": "1"},
    )
    assert captured[0][0].program == "test"
    assert captured[0][1] == {"environment": {"X": "1"}}


def test_input_files_remain_available_to_decoder(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Input files remain available to decoder."""

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        """Interpret the fake native result or deliberately exercise a decoding failure."""
        assert out.directory is not None
        assert (out.directory / "nested/file").read_bytes() == b"bytes"
        return SinglePointData.model_validate(
            {"provenance": {"program": "test"}, "energy": 1}
        )

    output = compute(
        prog_input_factory("energy"),
        codec=local_codec(
            PythonInput(lambda: None, files={"nested/file": b"bytes"}), decode
        ),
    )
    assert output.success and (not output.results.files)


def test_print_logs_and_retention(
    prog_input_factory: Callable[..., ProgramInput], capsys: pytest.CaptureFixture[str]
) -> None:
    """Print logs and retention."""
    codec = local_codec(SubprocessInput((sys.executable, "-c", "print('hello')")))
    output = compute(
        prog_input_factory("energy"), codec=codec, print_logs=True, collect_logs=False
    )
    assert capsys.readouterr().out == "hello\n"
    assert output.logs is None


def test_execution_failure_remains_primary_when_decode_fails(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Execution failure remains primary when decode fails."""
    request = SubprocessInput(
        (sys.executable, "-c", "print('before failure'); exit(9)")
    )

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        """Interpret the fake native result or deliberately exercise a decoding failure."""
        raise ParserError(
            "truncated artifact",
            data=SinglePointData.model_validate(
                {"provenance": {"program": "test"}, "energy": -2}
            ),
        )

    output = compute(
        prog_input_factory("gradient"),
        codec=local_codec(request, decode),
        raise_exc=False,
    )
    assert isinstance(output.results, SinglePointData)
    assert not output.success and output.results.energy == -2
    assert output.traceback is not None
    assert (
        "return code 9" in output.traceback and "truncated artifact" in output.traceback
    )


def test_codec_explanation_of_nonzero_exit_is_primary(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """When the codec explains a nonzero exit, its reason leads; the status follows."""
    request = SubprocessInput((sys.executable, "-c", "exit(1)"))

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        """Explain the failure the way a codec's failure hook does."""
        raise CalculationError(
            "SCF did not converge",
            data=SinglePointData.model_validate(
                {"provenance": {"program": "test"}, "energy": -3}
            ),
        )

    with pytest.raises(ExternalProgramError, match="SCF did not converge") as raised:
        compute(prog_input_factory("gradient"), codec=local_codec(request, decode))
    output = raised.value.prog_output
    assert output is not None and output.results.energy == -3
    assert output.traceback is not None and "return code 1" in output.traceback


def test_timeout_stays_primary_over_codec_explanation(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """A timeout is the real cause even if the codec then finds the output incomplete."""
    request = SubprocessInput((sys.executable, "-c", "import time; time.sleep(5)"))

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        """Report the truncated output a killed program leaves."""
        raise CalculationError("output ends early")

    with pytest.raises(ProgramTimeoutError):
        compute(
            prog_input_factory("gradient"),
            codec=local_codec(request, decode),
            timeout=0.5,
        )


def test_wavefunction_retention_uses_codec_patterns(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Wavefunction retention uses codec patterns."""
    from pathlib import Path

    def invoke() -> None:
        """Exercise the callable boundary with deterministic native behavior."""
        Path("native").mkdir()
        Path("native/restart").write_bytes(b"restart")

    request = PythonInput(invoke, wavefunction_files=("native/restart",))
    output = compute(
        prog_input_factory("energy"), codec=local_codec(request), collect_wfn=True
    )
    assert output.results.files == {"native/restart": b"restart"}


def test_missing_match_exception_roundtrip(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Missing match exception roundtrip."""
    from qccodec.exceptions import MatchNotFoundError

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        """Interpret the fake native result or deliberately exercise a decoding failure."""
        raise MatchNotFoundError("energy", "truncated")

    with pytest.raises(QCComputeBaseError) as raised:
        compute(
            prog_input_factory("energy"),
            codec=local_codec(PythonInput(lambda: None), decode),
        )
    restored = pickle.loads(pickle.dumps(raised.value))
    assert str(restored) == str(raised.value)
    assert restored.prog_output == raised.value.prog_output
    assert restored.original_exception.regex == "energy"


@pytest.mark.parametrize("during_decode", [False, True])
@pytest.mark.parametrize("error_type", [FileNotFoundError, ModuleNotFoundError])
def test_missing_python_file_or_module_preserves_cause(
    prog_input_factory: Callable[..., ProgramInput],
    during_decode: bool,
    error_type: type[Exception],
) -> None:
    """Do not infer a missing scientific program from arbitrary Python errors."""
    original = error_type("missing auxiliary input or parser dependency")

    def fail(*args: object) -> SinglePointData:
        """Raise the same failure at either side of the decoding boundary."""
        raise original

    codec = local_codec(
        PythonInput(lambda: None) if during_decode else PythonInput(fail),
        fail if during_decode else None,
    )
    with pytest.raises(ExternalProgramError) as raised:
        compute(prog_input_factory("energy"), codec=codec)
    assert not isinstance(raised.value, ProgramNotFoundError)
    assert raised.value.__cause__ is original
    assert raised.value.prog_output is not None
    assert not raised.value.prog_output.success


def test_missing_executable_attaches_failed_output(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """A program missing from PATH fails before launch and says how to point to it."""
    with pytest.raises(
        ProgramNotFoundError, match="QCCOMPUTE_TEST_EXECUTABLE"
    ) as raised:
        compute(
            prog_input_factory("energy"),
            codec=local_codec(SubprocessInput(("nonexistent-qcc-program",))),
        )
    assert raised.value.prog_output is not None
    assert not raised.value.prog_output.success


def test_invalid_environment_is_input_error(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """Reject invalid options before invoking the program and attach diagnostics."""
    with pytest.raises(InputError, match="strings") as raised:
        compute(
            prog_input_factory("energy"),
            codec=local_codec(PythonInput(lambda: pytest.fail("Program executed"))),
            environment={"OMP_NUM_THREADS": 1},  # type: ignore[dict-item]
        )
    assert raised.value.prog_output is not None
    assert not raised.value.prog_output.success


def test_absent_wavefunction_artifact_is_execution_error(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """A supported artifact missing after execution is not an encoding error."""
    with pytest.raises(
        ExternalProgramError, match="No requested wavefunction"
    ) as raised:
        compute(
            prog_input_factory("energy"),
            codec=local_codec(
                PythonInput(lambda: None, wavefunction_files=("missing.wfn",))
            ),
            collect_wfn=True,
        )
    assert raised.value.prog_output is not None
    assert raised.value.prog_output.results.energy == -1


@pytest.mark.parametrize("raise_exc", [False, True])
def test_cleanup_failure_preserves_primary_error(
    prog_input_factory: Callable[..., ProgramInput],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raise_exc: bool,
) -> None:
    """Cleanup diagnostics must not replace the scientific failure or disagree with output."""

    def invoke() -> None:
        """Fail after producing useful diagnostics."""
        print("partial log")
        raise RuntimeError("primary calculation failure")

    def fail_cleanup(path: Path) -> None:
        """Simulate a scratch removal failure."""
        raise OSError("cleanup failure")

    monkeypatch.setattr("qccompute.execution.shutil.rmtree", fail_cleanup)
    codec = local_codec(PythonInput(invoke))
    if raise_exc:
        with pytest.raises(
            QCComputeBaseError, match="primary calculation failure"
        ) as raised:
            compute(
                prog_input_factory("energy"),
                codec=codec,
                scratch_dir=tmp_path / "job",
                collect_logs=False,
            )
        output = raised.value.prog_output
        assert output is not None
    else:
        output = compute(
            prog_input_factory("energy"),
            codec=codec,
            scratch_dir=tmp_path / "job",
            collect_logs=False,
            raise_exc=False,
        )
    assert not output.success and output.logs == "partial log\n"
    assert output.traceback is not None
    assert "primary calculation failure" in output.traceback
    assert "cleanup failure" in output.traceback


def test_timeout_raises_with_partial_output(
    prog_input_factory: Callable[..., ProgramInput],
) -> None:
    """compute(timeout=...) kills a slow program and still decodes what it wrote."""
    request = SubprocessInput(
        (
            sys.executable,
            "-c",
            "import time; print('partial', flush=True); time.sleep(60)",
        )
    )

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        """Recover what the program printed before it was killed."""
        assert out.failed and "partial" in out.stdout
        return SinglePointData.model_validate(
            {"provenance": {"program": "test"}, "energy": -1}
        )

    with pytest.raises(ProgramTimeoutError, match="timeout") as raised:
        compute(
            prog_input_factory("gradient"),
            codec=local_codec(request, decode),
            timeout=1,
        )
    output = raised.value.prog_output
    assert output is not None and not output.success
    assert output.logs is not None and "partial" in output.logs
    assert isinstance(output.results, SinglePointData) and output.results.energy == -1
    with pytest.raises(InputError, match="timeout"):
        compute(prog_input_factory("energy"), codec=local_codec(request), timeout=0)
