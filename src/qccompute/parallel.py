"""Run a list of calculations, in parallel where that is safe.

Subprocess calculations isolate themselves: each child has its own output pipe,
working directory, and environment. They run in a thread pool sized to the cores
this process may use. Python calculations redirect the process's file descriptors,
working directory, and environment while they run, so two can never overlap. They
run one at a time in the calling thread, after the pool has finished, so no thread
writes into another calculation's captured output.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import qccodec
from qcdata import Data, InputType, ProgramOutput

from .exceptions import ComputeErrors, InputError, QCComputeBaseError
from .runners import in_python_call, kill_running

ComputeOne = Callable[..., ProgramOutput[InputType, Data]]
ListUpdate = Callable[[int, str, str], None]


def usable_cores() -> int:
    """Return the number of CPU cores this process may run on."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS or Windows
        return os.cpu_count() or 1


def _runs_as_subprocess(input_data: InputType, codec: qccodec.Codec | None) -> bool:
    """Whether an input encodes to a subprocess, which may run alongside others.

    Inputs that cannot be encoded run serially; computing them reports the error.
    """
    try:
        resolved = codec if codec is not None else qccodec.get_codec(input_data)
        return isinstance(resolved.encode(input_data), qccodec.SubprocessInput)
    except Exception:  # noqa: BLE001 - the real computation reports it
        return False


def compute_list(
    compute_one: ComputeOne,
    inputs: Sequence[InputType],
    *,
    max_workers: int | None,
    raise_exc: bool,
    update_func: ListUpdate | None,
    print_logs: bool,
    options: Mapping[str, Any],
) -> list[ProgramOutput[InputType, Data]]:
    """Compute every input and return the outputs in input order.

    Args:
        compute_one: Computes one input, raising on failure (``compute`` itself).
        inputs: The inputs, of any programs.
        max_workers: Most calculations to run at once; the usable cores when None.
        raise_exc: After every calculation has run, raise ``ComputeErrors`` if any
            failed; otherwise return the failed outputs.
        update_func: Receives ``(index, all_logs, new_logs)`` for each input's
            output, from the thread running it, so it must be thread-safe.
        print_logs: Print each line of output prefixed with its input's index.
        options: The remaining ``compute`` arguments, applied to every input.

    Raises:
        InputError: ``max_workers`` is not a positive integer, or an option cannot
            apply to several calculations.
        ComputeErrors: Some calculations failed and ``raise_exc`` is True.
    """
    if max_workers is not None and (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise InputError("compute", "max_workers must be a positive integer.")
    if options.get("scratch_dir") is not None:
        raise InputError(
            "compute",
            "scratch_dir names one calculation's directory; with a list of inputs, "
            "leave it out (with rm_scratch_dir=False each output's "
            "execution.scratch_dir records its directory).",
        )
    workers = max_workers or usable_cores()
    outputs: list[ProgramOutput[InputType, Data] | None] = [None] * len(inputs)
    errors: dict[int, QCComputeBaseError] = {}
    printing = threading.Lock()

    def run(index: int) -> None:
        """Compute one input, keeping its output and any error by position."""
        on_update: Callable[[str, str], None] | None = None
        if update_func is not None:

            def on_update(all_logs: str, new_logs: str) -> None:
                update_func(index, all_logs, new_logs)

        elif print_logs:

            def on_update(all_logs: str, new_logs: str) -> None:
                prefixed = "".join(
                    f"[{index}] {line}" for line in new_logs.splitlines(keepends=True)
                )
                with printing:
                    print(prefixed, end="" if prefixed.endswith("\n") else "\n")

        try:
            outputs[index] = compute_one(
                inputs[index], raise_exc=True, update_func=on_update, **options
            )
        except QCComputeBaseError as error:
            error.input_index = index  # type: ignore[attr-defined]
            errors[index] = error
            outputs[index] = error.prog_output

    codec = options.get("codec")
    if in_python_call() or workers == 1:
        parallel: list[int] = []
    else:
        parallel = [
            i for i, inp in enumerate(inputs) if _runs_as_subprocess(inp, codec)
        ]
    serial = [i for i in range(len(inputs)) if i not in set(parallel)]

    if parallel:
        pool = ThreadPoolExecutor(max_workers=min(workers, len(parallel)))
        futures: list[Future[None]] = [pool.submit(run, i) for i in parallel]
        try:
            for future in futures:
                future.result()
        except BaseException:
            # Interrupted (Ctrl-C) or a bug: stop queued work and kill running
            # programs, which run in their own sessions and so miss the signal.
            pool.shutdown(wait=False, cancel_futures=True)
            kill_running()
            raise
        pool.shutdown()
    for index in serial:
        run(index)

    completed = [output for output in outputs if output is not None]
    if len(completed) != len(inputs):  # pragma: no cover - compute always attaches one
        raise RuntimeError("A calculation finished without an output.")
    if errors and raise_exc:
        failed = [errors[i] for i in sorted(errors)]
        raise ComputeErrors(
            f"{len(failed)} of {len(inputs)} calculations failed", failed, completed
        )
    return completed
