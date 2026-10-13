"""Scalar calculation lifecycle using independently testable codecs and runners."""

import traceback
from collections.abc import Callable, Mapping
from functools import partial
from time import monotonic
from typing import Any

import qccodec
from pydantic import ValidationError
from qccodec.exceptions import CodecNotFoundError, DependencyNotFoundError, EncoderError
from qcdata import (
    CalcType,
    Data,
    ExecutionInfo,
    FileInput,
    Files,
    InputType,
    Model,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Structure,
)
from qcdata.helper_types import StrOrPath

from .exceptions import (
    AdapterInputError,
    AdapterNotFoundError,
    ExternalProgramError,
    ProgramNotFoundError,
    QCComputeBaseError,
)
from .execution import collect_files as read_files
from .execution import construct_execution, empty_results, stage_files, workspace
from .runners import run_python, run_subprocess


def compute(
    input_data: InputType,
    *,
    codec: qccodec.Codec | None = None,
    environment: Mapping[str, str] | None = None,
    scratch_dir: StrOrPath | None = None,
    rm_scratch_dir: bool = True,
    collect_logs: bool = True,
    collect_files: bool = False,
    collect_wfn: bool = False,
    update_func: Callable[[str, str], None] | None = None,
    update_interval: float | None = None,
    print_logs: bool = False,
    raise_exc: bool = True,
    propagate_wfn: bool = False,
    qcng_fallback: bool = True,
    **codec_options: Any,
) -> ProgramOutput[InputType, Data]:
    """Compute the given input using a codec and a local runner.

    Args:
        input_data: A qcdata input object, such as ProgramInput or FileInput.
        codec: An object or module implementing encode(input_data) and
            decode(input_data, native_output). If None, resolve a built-in codec
            using input_data.program, or the native-file codec for FileInput.
        environment: Process environment overrides mapping strings to strings.
            Applied after the inherited environment and codec defaults. Subprocess
            overrides affect only the child. Python calls temporarily change the
            current process's environment, working directory, and output streams;
            use separate processes for parallel Python calculations. Environment
            settings must be applied before the relevant native library initializes.
        scratch_dir: The calculation directory, not its parent. If None, create a
            directory in the system's temporary directory. Each concurrent call
            needs its own directory. This directory, including any existing
            contents, is deleted when rm_scratch_dir is True.
        rm_scratch_dir: Delete the calculation directory after execution, decoding,
            and file collection finish, including on failure.
        collect_logs: Retain merged stdout/stderr in ProgramOutput.logs. Available
            logs are always retained for failed calculations.
        collect_files: Collect files from the calculation directory, excluding files
            supplied in input_data.files. FileInput always collects these files.
            Decoders can access generated files regardless of this setting.
        collect_wfn: Collect wavefunction files identified by the codec. Not every
            codec supports this. Ignored when collect_files is True, which retains
            all generated files including wavefunction files.
        update_func: Callback accepting (all_logs, new_logs), both strings.
            Subprocess updates arrive during execution; Python output is delivered
            after invocation. No callback is made for empty output.
        update_interval: Minimum seconds between subprocess log updates; defaults
            to 0.5. Updates are line-based, and final output is delivered without
            waiting for this interval. Has no effect on Python calls.
        print_logs: Print merged stdout/stderr using the update callback. Ignored
            when update_func is supplied. Python logs print after invocation.
        raise_exc: Raise a qccompute exception on calculation failure, with the
            completed ProgramOutput attached as .prog_output. If False, return
            the failed output instead. Partial results and diagnostics are retained
            when available.
        propagate_wfn: For integrations that request child calculations, such as
            geometry optimization, use the child codec's restart support to prepare
            each input from the previous result. Requires a child-calculation
            integration and support for collecting and restarting from wavefunctions.
        qcng_fallback: Use the QCEngine compatibility codec when no built-in codec
            exists for the requested program. Also applies to child calculations.
        **codec_options: Additional options configuring the selected built-in codec,
            including supported QCEngine options. Configure an explicitly supplied
            codec before passing it; these options are not accepted with codec or
            FileInput.

    Returns:
        A ProgramOutput containing scientific results, execution metadata, and the
        requested logs and files. Failed calculations may contain partial results.

    Raises:
        AdapterNotFoundError: No codec is available for the requested program.
        AdapterInputError: Inputs or compute options are unsupported or invalid.
        ProgramNotFoundError: The executable cannot be found, or a codec explicitly
            reports an unavailable dependency.
        ExternalProgramError: Execution, decoding, artifact collection, or cleanup
            fails. Unexpected codec exceptions also use this category.

    Exception names containing "Adapter" are retained for API compatibility.
    Calculation exceptions are raised only when raise_exc is True.
    """
    start = monotonic()
    directory = None
    result = empty_results(input_data)
    logs = None
    failures: list[tuple[Exception, str]] = []
    if print_logs and update_func is None:
        update_func = lambda _, new: print(new, end="")

    def record_failure(exc: Exception, trace: str | None = None) -> None:
        """Keep the first failure primary and retain subsequent diagnostics."""
        failures.append((exc, trace or "".join(traceback.format_exception(exc))))

    try:
        if environment is not None and any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in environment.items()
        ):
            raise AdapterInputError(
                input_data.program, "Environment overrides must map strings to strings."
            )
        if codec is None:
            if type(input_data) is FileInput:
                from qccodec.codecs import file as file_codec

                codec = file_codec
                if codec_options:
                    raise AdapterInputError(
                        input_data.program, "FileInput does not accept codec options."
                    )
            else:
                codec = qccodec.get_codec(
                    input_data.program, qcng_fallback=qcng_fallback, **codec_options
                )
        elif codec_options:
            raise AdapterInputError(
                input_data.program,
                "Configure an external codec before passing it to compute()."
            )
        result = empty_results(input_data, getattr(codec, "producer_program", None))
        request = codec.encode(input_data)
        if not isinstance(request, (qccodec.SubprocessInput, qccodec.PythonInput)):
            raise AdapterInputError(
                input_data.program,
                "codec.encode() must return SubprocessInput or PythonInput."
            )
        if collect_wfn and not collect_files and not request.wavefunction_files:
            raise AdapterInputError(
                input_data.program,
                "This codec does not support wavefunction file collection."
            )
        if propagate_wfn and not (
            isinstance(request, qccodec.PythonInput) and request.child_callback
        ):
            raise AdapterInputError(
                input_data.program,
                "Wavefunction propagation requires a child-calculation integration."
            )
        with workspace(scratch_dir, rm_scratch_dir) as directory:
            try:
                stage_files(directory, request.files)
                if isinstance(request, qccodec.SubprocessInput):
                    native = run_subprocess(
                        request,
                        directory,
                        environment=environment,
                        update_func=update_func,
                        update_interval=update_interval,
                    )
                else:
                    native = run_python(
                        request,
                        directory,
                        environment=environment,
                        compute_child=partial(
                            _compute_child,
                            propagate_wfn=propagate_wfn,
                            environment=environment,
                            update_func=update_func,
                            update_interval=update_interval,
                            qcng_fallback=qcng_fallback,
                        ),
                        update_func=update_func,
                    )
                logs = native.stdout
                if native.failed:
                    execution_error = native.error
                    if execution_error is None:
                        assert isinstance(native, qccodec.SubprocessOutput)
                        execution_error = ExternalProgramError(
                            input_data.program,
                            f"External program failed with return code {native.returncode}.",
                        )
                    record_failure(execution_error, native.traceback)
                # Always attempt scientific recovery, including after execution failure.
                try:
                    result = codec.decode(input_data, native)
                except Exception as exc:
                    recovered = getattr(exc, "data", None)
                    if isinstance(recovered, type(result)):
                        result = recovered
                    record_failure(exc)
            except Exception as exc:
                record_failure(exc)
            # Artifacts must remain available through decoding and retention.
            try:
                retained = {}
                if collect_files or type(input_data) is FileInput:
                    retained = read_files(directory, exclude=input_data.files)
                elif collect_wfn:
                    retained = read_files(
                        directory, patterns=request.wavefunction_files, binary=True
                    )
                    if not retained:
                        raise ExternalProgramError(
                            input_data.program,
                            "No requested wavefunction files were produced."
                        )
                if retained:
                    result = type(result).model_validate(
                        {**result.model_dump(), "files": {**result.files, **retained}}
                    )
            except Exception as exc:
                record_failure(exc)
    except Exception as exc:
        record_failure(exc)

    # Finalize only after cleanup, so raised errors and attached outputs agree.
    execution = construct_execution(directory, monotonic() - start)
    output = _finalize_output(
        input_data, result, logs, execution, failures, collect_logs=collect_logs
    )
    if failures and raise_exc:
        primary = failures[0][0]
        error = _calculation_error(primary, input_data.program)
        error.prog_output = output
        error.data = output.results
        if hasattr(error, "logs"):
            error.logs = output.logs
        if error is primary:
            raise error
        raise error from primary
    return output


def _compute_child(
    child_input: ProgramInput,
    *,
    propagate_wfn: bool,
    environment: Mapping[str, str] | None,
    update_func: Callable[[str, str], None] | None,
    update_interval: float | None,
    qcng_fallback: bool,
    previous: ProgramOutput[ProgramInput, Data] | None = None,
    codec: qccodec.Codec | None = None,
) -> ProgramOutput[ProgramInput, Data]:
    """Run a child with optional restart input, retaining failed steps for its parent.

    Children inherit environment and log callbacks, but use their own temporary
    workspaces and default retention settings except for restart wavefunctions.

    Args:
        child_input: Standardized input for the child calculation.
        propagate_wfn: Collect restart files and apply the codec's restart hook.
        environment: Environment overrides inherited from the parent.
        update_func: Log callback inherited from the parent.
        update_interval: Subprocess callback interval inherited from the parent.
        qcng_fallback: Allow QCEngine fallback when resolving the child codec.
        previous: Previous child output supplying restart data, if available.
        codec: Explicit child codec; otherwise resolve it from the input.

    Returns:
        The child output, including a failed output if restart preparation fails.
    """
    child_start = monotonic()
    try:
        if propagate_wfn:
            if codec is None:
                codec = qccodec.get_codec(
                    child_input.program, qcng_fallback=qcng_fallback
                )
            if previous is not None:
                restart = getattr(codec, "restart", None)
                if restart is None:
                    raise AdapterInputError(
                        child_input.program,
                        f"Codec {child_input.program} does not support restart propagation."
                    )
                child_input = restart(child_input, previous.results)
    except Exception:
        return ProgramOutput(
            input_data=child_input,
            results=empty_results(child_input),
            success=False,
            traceback=traceback.format_exc(),
            execution=construct_execution(None, monotonic() - child_start),
        )
    return compute(
        child_input,
        codec=codec,
        environment=environment,
        collect_wfn=propagate_wfn,
        raise_exc=False,
        update_func=update_func,
        update_interval=update_interval,
        qcng_fallback=qcng_fallback,
    )


def _finalize_output(
    input_data: InputType,
    result: Data,
    logs: str | None,
    execution: ExecutionInfo,
    failures: list[tuple[Exception, str]],
    *,
    collect_logs: bool,
) -> ProgramOutput[InputType, Data]:
    """Validate the output, retaining partial results if success validation fails.

    A validation failure is appended to failures so the caller can raise it with
    the completed failed output. Results of the wrong type are replaced with the
    empty result type required by the input contract.

    Args:
        input_data: Original standardized input.
        result: Decoded results, possibly incomplete after a failure.
        logs: Captured merged output.
        execution: Completed execution metadata, including cleanup time.
        failures: Ordered failures; validation errors are appended in place.
        collect_logs: Retain logs on success as well as on failure.

    Returns:
        A validated output with the available diagnostics and partial results.

    Raises:
        ValidationError: The fallback failed output also violates the data contract.
    """
    def build_output() -> ProgramOutput[InputType, Data]:
        """Construct the output from the current results and failure diagnostics."""
        return ProgramOutput(
            input_data=input_data,
            results=result,
            success=not failures,
            logs=logs if collect_logs or failures else None,
            traceback="\nAdditional failure:\n".join(detail for _, detail in failures)
            or None,
            execution=execution,
        )

    try:
        return build_output()
    except ValidationError as exc:
        failures.append((exc, "".join(traceback.format_exception(exc))))
        if type(result) is not type(empty_results(input_data)):
            result = empty_results(input_data)
        return build_output()


def _calculation_error(
    exc: Exception, program: str
) -> QCComputeBaseError:
    """Translate the primary failure into the existing public exception categories."""
    if isinstance(exc, QCComputeBaseError):
        return exc
    if isinstance(exc, CodecNotFoundError):
        return AdapterNotFoundError(program, str(exc))
    if isinstance(exc, EncoderError):
        return AdapterInputError(program, str(exc))
    if isinstance(exc, DependencyNotFoundError):
        return ProgramNotFoundError(program, str(exc))
    return ExternalProgramError(program, str(exc), original_exception=exc)


def compute_args(
    program: str,
    structure: Structure,
    *,
    calctype: str | CalcType,
    model: dict[str, str] | Model | None = None,
    keywords: dict[str, Any] | None = None,
    files: dict[str, str | bytes] | Files | None = None,
    extras: dict[str, Any] | None = None,
    subprograms: list[ProgramSpec] | None = None,
    structures: dict[str, Structure] | None = None,
    cmdline_args: list[str] | None = None,
    **kwargs: Any,
) -> ProgramOutput[ProgramInput, Data]:
    """Compute function that accepts independent argument for a ProgramInput.

    Args:
        program: The program to run.
        structure: The structure to use.
        calctype: The type of calculation to run.
        model: The model to use for the calculation.
        keywords: The keywords to use for the calculation.
        files: The files to use for the calculation. Either a qcdata.Files object or a
            dict mapping file names to file contents (bytes or str).
        extras: User metadata saved with the input.
        subprograms: Recursive specifications for child calculations.
        structures: Additional complete structures identified by role.
        cmdline_args: Additional command-line arguments for the program.
        **kwargs: Extra arguments to pass to the compute function.

    Returns:
        The output of the computation.

    Raises:
        See compute function for details.
    """
    if isinstance(files, Files):  # Check in case Files object is passed instead of dict
        files = files.files

    program_input = ProgramInput(
        program=program,
        calctype=CalcType(calctype),
        structure=structure,
        model=model,  # type: ignore
        keywords=keywords or {},
        files=files or {},
        extras=extras or {},
        subprograms=subprograms or [],
        structures=structures or {},
        cmdline_args=cmdline_args or [],
    )

    return compute(program_input, **kwargs)
