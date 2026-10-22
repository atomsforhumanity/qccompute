"""A worker process that computes Python calculations for a list in another process.

Started by ``qccompute.workers`` as ``python -m qccompute._worker <read-fd> <write-fd>``
rather than through ``multiprocessing``, so it never imports the user's script. It
reads tasks from one pipe and writes progress updates and results to the other; the
file descriptors are separate from stdout and stderr, which Python calculations
redirect while they run. Each task is one ``compute`` call; errors come back with
their completed output attached.
"""

from __future__ import annotations

import os
import pickle
import signal
import sys
import traceback
from types import FrameType
from typing import Any

from .workers import EOF, receive, send

# Lists computed inside this process (for example by a calculation of the task)
# run serially instead of starting workers of their own.
os.environ["QCCOMPUTE_WORKER"] = "1"


def _terminate(signum: int, frame: FrameType | None) -> None:
    """Kill the programs this worker started (they run in their own sessions)."""
    from .runners import kill_running

    kill_running()
    os._exit(128 + signum)


def _run(task: tuple[Any, ...], writer: Any) -> None:
    """Compute one task and send its result."""
    from .exceptions import ExternalProgramError, QCComputeBaseError
    from .main import _compute_one

    input_data, codec_bytes, options, updates = task
    codec = pickle.loads(codec_bytes) if codec_bytes is not None else None

    def update(all_logs: str, new_logs: str) -> None:
        send(writer, ("update", all_logs, new_logs))

    try:
        output = _compute_one(
            input_data,
            codec=codec,
            raise_exc=True,
            update_func=update if updates else None,
            **options,
        )
    except QCComputeBaseError as error:
        try:
            send(writer, ("error", error))
        except Exception:  # noqa: BLE001 - the error holds something unpicklable
            plain = ExternalProgramError(input_data.program, str(error))
            plain.prog_output = error.prog_output
            send(writer, ("error", plain))
        return
    send(writer, ("ok", output))


def main() -> None:
    """Serve tasks until the parent closes the task pipe."""
    signal.signal(signal.SIGTERM, _terminate)
    reader = os.fdopen(int(sys.argv[1]), "rb")
    writer = os.fdopen(int(sys.argv[2]), "wb")
    while True:
        task = receive(reader)
        if task is EOF:
            return
        try:
            _run(task, writer)
        except BaseException:  # noqa: BLE001 - report it rather than die silently
            send(writer, ("crash", traceback.format_exc()))


if __name__ == "__main__":
    main()
