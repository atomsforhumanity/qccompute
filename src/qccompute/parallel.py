"""Run a list of calculations, in parallel where that is safe.

Subprocess calculations isolate themselves: each child has its own output pipe,
working directory, and environment. They run in a thread pool, within a budget of
CPU threads (the cores this process may use) that each calculation draws on as its
codec declares, and each GPU calculation gets its own devices. Python calculations
redirect the process's file descriptors, working directory, and environment while
they run, so two can never overlap. They run one at a time in the calling thread,
after the pool has finished, so no thread writes into another calculation's
captured output. With two or more of them in a list, they run side by side in
worker processes instead (``qccompute.workers``), drawing on the same budget.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import qccodec
from qcdata import Data, InputType, ProgramOutput

from .exceptions import (
    ComputeErrors,
    ExternalProgramError,
    InputError,
    QCComputeBaseError,
)
from .runners import in_python_call, kill_running
from .workers import WorkerDied, WorkerPool, supported

ComputeOne = Callable[..., ProgramOutput[InputType, Data]]
ListUpdate = Callable[[int, str, str], None]


def usable_cores() -> int:
    """Return the number of CPU cores this process may run on."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS or Windows
        return os.cpu_count() or 1


def visible_gpus(environment: Mapping[str, str] | None = None) -> list[str]:
    """Return the GPU device IDs calculations may use.

    ``CUDA_VISIBLE_DEVICES`` (from ``environment``, else this process) wins when set;
    otherwise ``nvidia-smi`` lists the NVIDIA GPUs. Returns ``[]`` when there are none
    or they cannot be found.
    """
    listed = (environment or {}).get("CUDA_VISIBLE_DEVICES")
    if listed is None:
        listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if listed is not None:
        return [d.strip() for d in listed.split(",") if d.strip() and d.strip() != "-1"]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _native_request(
    input_data: InputType, codec: qccodec.Codec | None
) -> qccodec.NativeInput | None:
    """Return an input's native invocation, or None if it cannot be encoded.

    Inputs that cannot be encoded run serially; computing them reports the error.
    """
    try:
        resolved = codec if codec is not None else qccodec.get_codec(input_data)
        return resolved.encode(input_data)
    except Exception:  # noqa: BLE001 - the real computation reports it
        return None


def _codec_for_worker(
    input_data: InputType, codec: qccodec.Codec | None
) -> tuple[bool, bytes | None]:
    """Return whether a worker can compute this input, and the codec to send it.

    A worker finds built-in codecs and those installed as entry points by name
    (``None``). A codec passed to ``compute`` or registered in this process exists
    only here, so it is sent pickled; if it cannot be pickled, the input runs in this
    process instead.
    """
    registered = codec is not None or input_data.program in getattr(
        qccodec.registry, "_REGISTERED", {}
    )
    if not registered:
        return True, None
    resolved = codec if codec is not None else qccodec.get_codec(input_data)
    try:
        return True, pickle.dumps(resolved)
    except Exception:  # noqa: BLE001 - lambdas and closures do not pickle
        return False, None


def _in_worker() -> bool:
    """Whether this process is a worker; lists inside one run serially."""
    return os.environ.get("QCCOMPUTE_WORKER") == "1"


def _died_output(
    input_data: InputType, message: str, collect_logs: bool
) -> tuple[ProgramOutput[InputType, Data], QCComputeBaseError]:
    """Build the failed output and error for a worker that died mid-calculation."""
    from .execution import construct_execution, empty_results
    from .main import _finalize_output

    error = ExternalProgramError(input_data.program, message)
    output = _finalize_output(
        input_data,
        empty_results(input_data),
        message,
        construct_execution(None, 0.0),
        [(error, message)],
        collect_logs=collect_logs,
    )
    error.prog_output = output
    return output, error


class _Resources:
    """CPU threads and GPU devices, handed to calculations in input order.

    Granting in order keeps a calculation that needs many threads from waiting
    forever behind a stream of small ones. A calculation asking for more threads
    than the budget holds runs alone; one asking for GPUs when none were found runs
    one at a time without pinning.
    """

    def __init__(self, threads: int, gpus: list[str]) -> None:
        self._total_threads = threads
        self._free_threads = threads
        self._devices = list(gpus)
        self._free_devices = list(gpus)
        # Without known devices, GPU calculations share one unpinned token.
        self._free_tokens = 1
        self._serving = 0
        self._ready = threading.Condition()

    def acquire(
        self, turn: int, threads: int, gpus: int
    ) -> tuple[int, list[str], bool]:
        """Wait for this calculation's turn and resources; return what it holds."""
        threads = max(1, min(threads, self._total_threads))
        pinned = bool(self._devices)
        gpus = min(gpus, len(self._devices)) if pinned else min(gpus, 1)
        with self._ready:
            self._ready.wait_for(
                lambda: (
                    self._serving == turn
                    and self._free_threads >= threads
                    and (
                        len(self._free_devices) >= gpus
                        if pinned
                        else self._free_tokens >= gpus
                    )
                )
            )
            self._serving += 1
            self._free_threads -= threads
            devices: list[str] = []
            token = False
            if pinned:
                devices = self._free_devices[:gpus]
                del self._free_devices[:gpus]
            elif gpus:
                self._free_tokens -= 1
                token = True
            self._ready.notify_all()
        return threads, devices, token

    def release(self, threads: int, devices: list[str], token: bool) -> None:
        """Return a finished calculation's resources."""
        with self._ready:
            self._free_threads += threads
            self._free_devices.extend(devices)
            if token:
                self._free_tokens += 1
            self._ready.notify_all()


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
        max_workers: The CPU threads calculations may use at once, the usable cores
            when None; a calculation counts the threads its codec declares.
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

        run_options = dict(options)
        held: tuple[int, list[str], bool] | None = None
        request = requests.get(index)
        if request is not None:
            held = resources.acquire(turns[index], request.threads, request.gpus)
            if held[1] and index not in worker_codecs:
                run_options["environment"] = {
                    **(options.get("environment") or {}),
                    "CUDA_VISIBLE_DEVICES": ",".join(held[1]),
                }
        try:
            if index in worker_codecs:
                assert held is not None
                outputs[index] = _compute_in_worker(
                    index, tuple(held[1]), held[0], on_update
                )
            else:
                outputs[index] = compute_one(
                    inputs[index], raise_exc=True, update_func=on_update, **run_options
                )
        except QCComputeBaseError as error:
            error.input_index = index  # type: ignore[attr-defined]
            errors[index] = error
            outputs[index] = error.prog_output
        finally:
            if held is not None:
                resources.release(*held)

    def _compute_in_worker(
        index: int,
        devices: tuple[str, ...],
        threads: int,
        on_update: Callable[[str, str], None] | None,
    ) -> ProgramOutput[InputType, Data]:
        """Compute a Python input in a worker started for its devices and threads."""
        assert pool is not None
        task_options = {k: v for k, v in options.items() if k != "codec"}
        task = (
            inputs[index],
            worker_codecs[index],
            task_options,
            on_update is not None,
        )
        worker = pool.acquire(devices, threads)
        try:
            kind, payload = worker.compute(task, on_update)
        except WorkerDied as died:
            output, error = _died_output(
                inputs[index], str(died), bool(options.get("collect_logs", True))
            )
            raise error from died
        finally:
            pool.release(worker)
        if kind == "error":
            raise payload
        return payload

    codec = options.get("codec")
    requests: dict[int, qccodec.NativeInput] = {}
    worker_codecs: dict[int, bytes | None] = {}
    if not in_python_call() and not _in_worker() and len(inputs) > 1:
        python: dict[int, qccodec.NativeInput] = {}
        for i, inp in enumerate(inputs):
            request = _native_request(inp, codec)
            if isinstance(request, qccodec.SubprocessInput):
                requests[i] = request
            elif isinstance(request, qccodec.PythonInput):
                python[i] = request
        # Starting a worker costs about a second; it pays only with company.
        if supported() and len(python) > 1 and workers > 1:
            for i, request in python.items():
                reachable, payload = _codec_for_worker(inputs[i], codec)
                if reachable:
                    requests[i] = request
                    worker_codecs[i] = payload
    pool = WorkerPool() if worker_codecs else None
    parallel = sorted(requests)
    serial = [i for i in range(len(inputs)) if i not in requests]
    turns = {index: turn for turn, index in enumerate(parallel)}
    gpus = (
        visible_gpus(options.get("environment"))
        if any(r.gpus for r in requests.values())
        else []
    )
    resources = _Resources(workers, gpus)

    if parallel:
        threads = ThreadPoolExecutor(max_workers=min(workers, len(parallel)))
        futures: list[Future[None]] = [threads.submit(run, i) for i in parallel]
        try:
            for future in futures:
                future.result()
        except BaseException:
            # Interrupted (Ctrl-C) or a bug: stop queued work and kill running
            # programs and workers, which run in their own sessions and so miss
            # the signal.
            threads.shutdown(wait=False, cancel_futures=True)
            kill_running()
            if pool is not None:
                pool.terminate()
            raise
        threads.shutdown()
        if pool is not None:
            pool.close()
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
