"""Run calculations: encode with a codec, execute, decode, and build the output."""

import dataclasses
import os
import traceback
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from time import monotonic
from typing import Any, overload

import qccodec
from pydantic import ValidationError
from qccodec.exceptions import CalculationError as CodecCalculationError
from qccodec.exceptions import CodecNotFoundError as CodecMissing
from qccodec.exceptions import DependencyNotFoundError, EncoderError
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
    CodecNotFoundError,
    ExternalProgramError,
    InputError,
    ProgramNotFoundError,
    QCComputeBaseError,
)
from .execution import collect_files as read_files
from .execution import (
    construct_execution,
    empty_results,
    resolve_executable,
    stage_files,
    workspace,
)
from .parallel import BatchError, compute_list
from .runners import run_python, run_subprocess


@overload
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
    update_func: Callable[..., None] | None = None,
    update_interval: float | None = None,
    print_logs: bool = False,
    raise_exc: bool = True,
    propagate_wfn: bool = False,
    options: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    executables: Mapping[str, str] | None = None,
    max_workers: int | None = None,
) -> ProgramOutput[InputType, Data]: ...


@overload
def compute(
    input_data: Sequence[InputType],
    *,
    codec: qccodec.Codec | None = None,
    environment: Mapping[str, str] | None = None,
    scratch_dir: StrOrPath | None = None,
    rm_scratch_dir: bool = True,
    collect_logs: bool = True,
    collect_files: bool = False,
    collect_wfn: bool = False,
    update_func: Callable[..., None] | None = None,
    update_interval: float | None = None,
    print_logs: bool = False,
    raise_exc: bool = True,
    propagate_wfn: bool = False,
    options: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    executables: Mapping[str, str] | None = None,
    max_workers: int | None = None,
) -> list[ProgramOutput[InputType, Data]]: ...


def compute(
    input_data: InputType | Sequence[InputType],
    *,
    codec: qccodec.Codec | None = None,
    environment: Mapping[str, str] | None = None,
    scratch_dir: StrOrPath | None = None,
    rm_scratch_dir: bool = True,
    collect_logs: bool = True,
    collect_files: bool = False,
    collect_wfn: bool = False,
    update_func: Callable[..., None] | None = None,
    update_interval: float | None = None,
    print_logs: bool = False,
    raise_exc: bool = True,
    propagate_wfn: bool = False,
    options: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    executables: Mapping[str, str] | None = None,
    max_workers: int | None = None,
) -> ProgramOutput[InputType, Data] | list[ProgramOutput[InputType, Data]]:
    """Compute one input, or a list of inputs in parallel, with a local runner.

    A single input returns a single ``ProgramOutput``. A list (or tuple) returns a
    list of outputs in input order; its inputs may be for different programs, and
    every other argument applies to each of them. Calculations that run as programs
    (ORCA, xtb, Psi4, ...) run side by side in threads. Calculations that run in
    Python (geomeTRIC, tblite) take over a process's output streams while they run,
    so two or more of them run side by side in worker processes, which start with the
    numerical libraries' thread counts set to their share. All of them draw on the
    ``max_workers`` budget. Inputs a codec can evaluate together (machine-learned
    potentials sharing a model) run in batches in this process, one call per batch
    of up to ``codec.max_batch``; ``execution.extras`` records ``batch_size`` and
    ``batch_index``.

    Args:
        input_data: A qcdata input object, such as ProgramInput or FileInput, or a
            list of them.
        codec: The ``qccodec.Codec`` translating this program's inputs and outputs.
            If None, resolve one with ``qccodec.get_codec(input_data)``: a codec
            registered with ``qccodec.register``, a built-in, or one installed by
            another package.
        environment: Process environment overrides mapping strings to strings.
            Applied after the inherited environment and codec defaults. Subprocess
            overrides affect only the child. Python calls temporarily change the
            current process's environment, working directory, and output streams;
            use separate processes for parallel Python calculations. Environment
            settings must be applied before the relevant native library initializes.
        scratch_dir: The calculation directory, not its parent; not accepted with a
            list of inputs. If None, create a
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
            after invocation. No callback is made for empty output. For a list of
            inputs it receives (index, all_logs, new_logs), where index is the
            input's position, and is called from the threads running the
            calculations, so it must be thread-safe.
        update_interval: Minimum seconds between subprocess log updates; defaults
            to 0.5. Updates are line-based, and final output is delivered without
            waiting for this interval. Has no effect on Python calls.
        print_logs: Print merged stdout/stderr using the update callback. Ignored
            when update_func is supplied. Python logs print after invocation. For a
            list, each line is prefixed with its input's index, e.g. ``[3]``.
        raise_exc: Raise a qccompute exception on calculation failure, with the
            completed ProgramOutput attached as .prog_output. If False, return
            the failed output instead. Partial results and diagnostics are retained
            when available. For a list, every calculation runs first; then
            ``ComputeErrors`` (an ``ExceptionGroup``) is raised if any failed, holding
            each failure's exception and every output in ``.outputs``.
        propagate_wfn: For integrations that request child calculations, such as
            geometry optimization, use the child codec's restart support to prepare
            each input from the previous result. Requires a child-calculation
            integration and support for collecting and restarting from wavefunctions.
        options: Program-specific decode options passed to the codec, such as
            ``{"collect_rotamers": True}`` for CREST. Each codec declares the
            options it accepts in ``codec.options``; unknown keys are rejected.
        timeout: Seconds each external program may run before it and the processes
            it started are killed, raising ``ProgramTimeoutError`` with any partial
            results attached. Applies to every subprocess, including child
            calculations of Python integrations such as geomeTRIC; a Python
            callable itself is not interrupted.
        executables: Executables to run, keyed by program name, such as
            ``{"orca": "/opt/orca/orca"}``. Child calculations of Python
            integrations such as geomeTRIC use the same mapping; entries for programs
            that do not run are ignored. Without an entry, the environment variable
            ``QCCOMPUTE_<PROGRAM>_EXECUTABLE`` (for example
            ``QCCOMPUTE_ORCA_EXECUTABLE``) is used, and otherwise the program is
            found on ``PATH``. Bare names are looked up on ``PATH``; the program
            always runs by its absolute path, recorded in
            ``execution.extras["executable"]``.
        max_workers: For a list of inputs, the CPU threads program calculations may
            use at once. Defaults to the number of CPU cores this process may use.
            Each calculation counts the threads its codec declares (ORCA ``pal``,
            Psi4 ``-n``, CREST ``threads``; one otherwise), and each GPU calculation
            gets its own devices through ``CUDA_VISIBLE_DEVICES``. Ignored for a
            single input.

    Returns:
        A ProgramOutput containing scientific results, execution metadata, and the
        requested logs and files, or a list of them in input order for a list of
        inputs. Failed calculations may contain partial results.

    Raises:
        CodecNotFoundError: No codec is available for the requested program.
        InputError: Inputs or compute options are unsupported or invalid.
        ProgramNotFoundError: The executable cannot be found, or a codec explicitly
            reports an unavailable dependency.
        ProgramTimeoutError: A program exceeded ``timeout``.
        ExternalProgramError: Execution, decoding, artifact collection, or cleanup
            fails. Unexpected codec exceptions also use this category.

        ComputeErrors: For a list of inputs, some calculations failed.

    Calculation exceptions are raised only when raise_exc is True.
    """
    if isinstance(input_data, Sequence) and not isinstance(input_data, str | bytes):
        return compute_list(
            _compute_one,
            input_data,
            compute_batch=_compute_batch,
            max_workers=max_workers,
            raise_exc=raise_exc,
            update_func=update_func,
            print_logs=print_logs,
            options={
                "codec": codec,
                "environment": environment,
                "scratch_dir": scratch_dir,
                "rm_scratch_dir": rm_scratch_dir,
                "collect_logs": collect_logs,
                "collect_files": collect_files,
                "collect_wfn": collect_wfn,
                "update_interval": update_interval,
                "propagate_wfn": propagate_wfn,
                "options": options,
                "timeout": timeout,
                "executables": executables,
            },
        )
    return _compute_one(
        input_data,
        codec=codec,
        environment=environment,
        scratch_dir=scratch_dir,
        rm_scratch_dir=rm_scratch_dir,
        collect_logs=collect_logs,
        collect_files=collect_files,
        collect_wfn=collect_wfn,
        update_func=update_func,
        update_interval=update_interval,
        print_logs=print_logs,
        raise_exc=raise_exc,
        propagate_wfn=propagate_wfn,
        options=options,
        timeout=timeout,
        executables=executables,
    )


def _compute_one(
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
    options: Mapping[str, Any] | None = None,
    timeout: float | None = None,
    executables: Mapping[str, str] | None = None,
) -> ProgramOutput[InputType, Data]:
    """Compute one input; ``compute`` documents every argument."""
    start = monotonic()
    directory = None
    result = empty_results(input_data)
    logs = None
    executable = None
    failures: list[tuple[Exception, str]] = []
    if print_logs and update_func is None:

        def update_func(all_logs: str, new_logs: str) -> None:
            """Print each new chunk of program output as it arrives."""
            print(new_logs, end="")

    def record_failure(exc: Exception, trace: str | None = None) -> None:
        """Keep the first failure primary and retain subsequent diagnostics."""
        failures.append((exc, trace or "".join(traceback.format_exception(exc))))

    try:
        if environment is not None and any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in environment.items()
        ):
            raise InputError(
                input_data.program, "Environment overrides must map strings to strings."
            )
        if timeout is not None and not timeout > 0:
            raise InputError(
                input_data.program, "timeout must be a positive number of seconds."
            )
        if executables is not None and any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in executables.items()
        ):
            raise InputError(
                input_data.program, "executables must map program names to paths."
            )
        if codec is None:
            codec = qccodec.get_codec(input_data)
        request = codec.encode(input_data)
        if collect_wfn and not collect_files and not request.wavefunction_files:
            raise InputError(
                input_data.program,
                "This codec does not support wavefunction file collection.",
            )
        if propagate_wfn and not (
            isinstance(request, qccodec.PythonInput) and request.child_callback
        ):
            raise InputError(
                input_data.program,
                "Wavefunction propagation requires a child-calculation integration.",
            )
        if isinstance(request, qccodec.SubprocessInput):
            executable = resolve_executable(
                input_data.program,
                request.command[0],
                {**os.environ, **request.environment, **(environment or {})},
                executables,
            )
            request = dataclasses.replace(
                request, command=(executable, *request.command[1:])
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
                        timeout=timeout,
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
                            timeout=timeout,
                            executables=executables,
                        ),
                        update_func=update_func,
                    )
                logs = native.stdout
                exit_status_only = native.failed and native.error is None
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
                    result = codec.decode(input_data, native, options)
                except Exception as exc:
                    recovered = getattr(exc, "data", None)
                    if isinstance(recovered, type(result)):
                        result = recovered
                    record_failure(exc)
                    if exit_status_only and isinstance(exc, CodecCalculationError):
                        # The codec explains a nonzero exit; lead with its reason and
                        # keep the bare exit status as an additional failure.
                        failures.insert(0, failures.pop())
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
                            "No requested wavefunction files were produced.",
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
    execution = construct_execution(directory, monotonic() - start, executable)
    output = _finalize_output(
        input_data, result, logs, execution, failures, collect_logs=collect_logs
    )
    if failures and raise_exc:
        raise _output_error(output, failures)
    return output


def _output_error(
    output: ProgramOutput[InputType, Data], failures: list[tuple[Exception, str]]
) -> QCComputeBaseError:
    """Return the exception for a failed output, caused by its primary failure."""
    primary = failures[0][0]
    error = _calculation_error(primary, output.input_data.program)
    error.prog_output = output
    error.data = output.results
    if hasattr(error, "logs"):
        error.logs = output.logs
    if error is not primary:
        error.__cause__ = primary
    return error


def _compute_batch(
    inputs: Sequence[ProgramInput],
    codec: qccodec.Codec,
    update_funcs: Sequence[Callable[[str, str], None] | None],
    *,
    environment: Mapping[str, str] | None = None,
    rm_scratch_dir: bool = True,
    collect_logs: bool = True,
    options: Mapping[str, Any] | None = None,
    **_: Any,
) -> list[tuple[ProgramOutput[ProgramInput, Data], QCComputeBaseError | None]]:
    """Compute inputs sharing a batch key in one call of the codec's batch function.

    Each value the call returns is decoded on its own, as if its input had run
    alone. Every output keeps the batch's logs and wall time, its directory, and
    ``execution.extras`` ``batch_size`` and ``batch_index``.

    Args:
        inputs: Inputs sharing one ``codec.batch_key``, at most ``codec.max_batch``.
        codec: The codec batching them.
        update_funcs: Each input's log callback, given the batch's logs afterward.
        environment: Environment overrides for the call.
        rm_scratch_dir: Delete the shared directory afterward.
        collect_logs: Retain the logs on successful outputs.
        options: The codec's decode options.

    Returns:
        Each input's output and, if it failed, its exception.

    Raises:
        BatchError: The batch could not be encoded, the call failed, or it returned
            the wrong number of values; nothing was decoded.
    """
    start = monotonic()
    try:
        request = codec.encode_batch(inputs)
    except Exception as exc:
        raise BatchError(f"The batch could not be encoded: {exc}") from exc
    decoded: list[tuple[Data, list[tuple[Exception, str]]]] = []
    with workspace(None, rm_scratch_dir) as directory:
        stage_files(directory, request.files)
        native = run_python(request, directory, environment=environment)
        if native.error is not None:
            # torch.OutOfMemoryError and torch.cuda.OutOfMemoryError
            raise BatchError(
                f"The batch failed: {native.error!r}",
                out_of_memory="OutOfMemory" in type(native.error).__name__,
            ) from native.error
        values = native.value
        if not isinstance(values, Sequence) or len(values) != len(inputs):
            raise BatchError(
                f"The batch function must return {len(inputs)} values, one per input."
            )
        for inp, value in zip(inputs, values, strict=True):
            failures: list[tuple[Exception, str]] = []
            result = empty_results(inp)
            try:
                result = codec.decode(
                    inp,
                    qccodec.PythonOutput(
                        value=value, stdout=native.stdout, directory=directory
                    ),
                    options,
                )
            except Exception as exc:
                recovered = getattr(exc, "data", None)
                if isinstance(recovered, type(result)):
                    result = recovered
                failures.append((exc, "".join(traceback.format_exception(exc))))
            decoded.append((result, failures))
    wall_time = monotonic() - start
    outputs = []
    for index, (inp, (result, failures), update) in enumerate(
        zip(inputs, decoded, update_funcs, strict=True)
    ):
        if update is not None and native.stdout:
            update(native.stdout, native.stdout)
        execution = construct_execution(directory, wall_time)
        execution.extras.update(batch_size=len(inputs), batch_index=index)
        output = _finalize_output(
            inp, result, native.stdout, execution, failures, collect_logs=collect_logs
        )
        outputs.append((output, _output_error(output, failures) if failures else None))
    return outputs


def _compute_child(
    child_input: ProgramInput,
    *,
    propagate_wfn: bool,
    environment: Mapping[str, str] | None,
    update_func: Callable[[str, str], None] | None,
    update_interval: float | None,
    timeout: float | None = None,
    executables: Mapping[str, str] | None = None,
    previous: ProgramOutput[ProgramInput, Data] | None = None,
) -> ProgramOutput[ProgramInput, Data]:
    """Run a child with optional restart input, retaining failed steps for its parent.

    Children inherit environment and log callbacks, but use their own temporary
    workspaces and default retention settings except for restart wavefunctions.
    The child codec is resolved by program name, so codecs registered with
    ``qccodec.register`` are available to optimizers.

    Args:
        child_input: Standardized input for the child calculation.
        propagate_wfn: Collect restart files and apply the codec's restart hook.
        environment: Environment overrides inherited from the parent.
        update_func: Log callback inherited from the parent.
        update_interval: Subprocess callback interval inherited from the parent.
        timeout: Per-program timeout inherited from the parent.
        executables: Executable overrides inherited from the parent.
        previous: Previous child output supplying restart data, if available.

    Returns:
        The child output, including a failed output if restart preparation fails.
    """
    child_start = monotonic()
    codec = None
    try:
        codec = qccodec.get_codec(child_input)
        if propagate_wfn and previous is not None:
            if codec.propagate_wfn is None:
                raise InputError(
                    child_input.program,
                    f"Codec {child_input.program!r} does not support wavefunction "
                    "propagation.",
                )
            child_input = codec.propagate_wfn(previous, child_input)
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
        timeout=timeout,
        executables=executables,
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


def _calculation_error(exc: Exception, program: str) -> QCComputeBaseError:
    """Translate the primary failure into the public exception categories."""
    if isinstance(exc, QCComputeBaseError):
        return exc
    if isinstance(exc, CodecMissing):
        return CodecNotFoundError(program, str(exc))
    if isinstance(exc, EncoderError):
        return InputError(program, str(exc))
    if isinstance(exc, DependencyNotFoundError):
        return ProgramNotFoundError(program, str(exc))
    return ExternalProgramError(program, str(exc), original_exception=exc)


def compute_args(
    program: str,
    structure: Structure | None,
    *,
    calctype: str | CalcType,
    model: dict[str, str] | Model | None = None,
    keywords: dict[str, Any] | None = None,
    files: dict[str, str | bytes] | Files | None = None,
    extras: dict[str, Any] | None = None,
    subprograms: list[ProgramSpec] | None = None,
    structures: dict[str, Structure | list[Structure]] | None = None,
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
