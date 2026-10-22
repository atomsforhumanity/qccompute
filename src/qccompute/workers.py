"""Worker processes for running Python calculations of a list side by side.

A Python calculation redirects its process's file descriptors, working directory,
and environment, so only one can run per process. To run several at once, each goes
to a worker process (``qccompute._worker``), started with ``python -m`` so it never
imports the caller's script. Parent and worker exchange length-prefixed pickles over
two pipes. A worker is started for the GPUs and CPU threads its tasks were granted:
``CUDA_VISIBLE_DEVICES`` and the thread counts of the numerical libraries are read once
per process, so they are set before it starts, and it only receives tasks granted the
same. Without the thread limits, every worker's BLAS would start a thread per core.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import IO, Any

# Returned by receive() when the other side closed its pipe.
EOF = object()
# Thread counts of numerical libraries (OpenMP, OpenBLAS, MKL, Accelerate, numexpr),
# set for a worker unless the caller's environment already sets them.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
# Every worker process started here, to check that none outlive their list.
_started: list[subprocess.Popen[bytes]] = []


def running_workers() -> list[int]:
    """Return the process IDs of workers started by this process that still run."""
    return [proc.pid for proc in _started if proc.poll() is None]


def send(stream: IO[bytes], message: Any) -> None:
    """Write one length-prefixed pickled message."""
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(len(data).to_bytes(8, "big"))
    stream.write(data)
    stream.flush()


def receive(stream: IO[bytes]) -> Any:
    """Read one length-prefixed pickled message, or ``EOF`` if the pipe closed."""
    header = stream.read(8)
    if len(header) < 8:
        return EOF
    size = int.from_bytes(header, "big")
    data = stream.read(size)
    if len(data) < size:
        return EOF
    return pickle.loads(data)


def supported() -> bool:
    """Whether workers can run here: they pass pipe descriptors (POSIX only)."""
    return os.name == "posix"


class WorkerDied(RuntimeError):
    """The worker process ended before returning a result."""


class Worker:
    """One worker process, started for given GPU devices and CPU threads."""

    def __init__(self, devices: tuple[str, ...] = (), threads: int = 1) -> None:
        """Start the worker process."""
        task_read, task_write = os.pipe()
        result_read, result_write = os.pipe()
        environment = dict(os.environ)
        if devices:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        for variable in THREAD_VARIABLES:
            environment.setdefault(variable, str(threads))
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "qccompute._worker",
                    str(task_read),
                    str(result_write),
                ],
                pass_fds=(task_read, result_write),
                env=environment,
                # Its own session: a terminal Ctrl-C reaches only the parent, which
                # then stops the worker in order.
                start_new_session=True,
            )
        finally:
            os.close(task_read)
            os.close(result_write)
        _started.append(self.process)
        self.key = (devices, threads)
        self._tasks = os.fdopen(task_write, "wb")
        self._results = os.fdopen(result_read, "rb")

    def compute(
        self, task: tuple[Any, ...], on_update: Callable[[str, str], None] | None
    ) -> tuple[str, Any]:
        """Send a task; relay its updates; return ``("ok", output)`` or ``("error", e)``.

        Raises:
            WorkerDied: The worker ended without a result.
        """
        try:
            send(self._tasks, task)
        except (BrokenPipeError, OSError) as exc:
            raise WorkerDied(
                f"The worker process could not take the task: {exc}"
            ) from exc
        while True:
            message = receive(self._results)
            if message is EOF:
                code = self.process.wait()
                raise WorkerDied(f"The worker process exited with status {code}.")
            kind = message[0]
            if kind == "update":
                if on_update is not None:
                    on_update(message[1], message[2])
            elif kind == "crash":
                raise WorkerDied(f"The worker process failed:\n{message[1]}")
            else:
                return kind, message[1]

    def alive(self) -> bool:
        """Whether the process is still running."""
        return self.process.poll() is None

    def stop(self) -> None:
        """Close the task pipe so the worker exits; kill it if it does not."""
        try:
            self._tasks.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.terminate()
        self._results.close()

    def terminate(self) -> None:
        """Stop the worker now; it kills the programs it started first."""
        if self.alive():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class WorkerPool:
    """Idle workers, kept by the GPU devices and CPU threads they were started with."""

    def __init__(self) -> None:
        """Start empty; workers start on first use."""
        self._idle: dict[tuple[tuple[str, ...], int], list[Worker]] = defaultdict(list)
        self._all: list[Worker] = []
        self._lock = threading.Lock()

    def acquire(self, devices: tuple[str, ...] = (), threads: int = 1) -> Worker:
        """Return an idle worker for these resources, starting one if none is idle."""
        with self._lock:
            idle = self._idle[(devices, threads)]
            while idle:
                worker = idle.pop()
                if worker.alive():
                    return worker
            worker = Worker(devices, threads)
            self._all.append(worker)
            return worker

    def release(self, worker: Worker) -> None:
        """Return a worker after its task, unless it has died."""
        with self._lock:
            if worker.alive():
                self._idle[worker.key].append(worker)

    def close(self) -> None:
        """Stop every worker, letting each finish cleanly."""
        for worker in self._all:
            worker.stop()

    def terminate(self) -> None:
        """Stop every worker now (Ctrl-C)."""
        for worker in self._all:
            worker.terminate()
