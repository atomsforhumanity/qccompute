"""Thread counts: whatever holds one never runs more than it at once.

A runner, a running calculation and a session each hold a thread count (`Count`). Work
runs under one: a top-level calculation under its runner's, a subprogram under its
parent's, a session request under its session's, which in turn runs under whatever opened
the session. Work takes its threads (and any GPUs, whole) before it starts and gives them
back when it ends; work that does not fit waits its turn, first come first served. Only
running work counts, so an open session that computes nothing holds nothing.

A thread running work holds its count (`holding`). Were it to make a top-level call, the
call could wait for threads the thread itself holds, so a top-level call from it fails at
once (`check_free`).
"""

from __future__ import annotations

import threading
import warnings
from collections import deque
from collections.abc import Iterable

from qcdata import ProgramError

from .program import _Control

__all__ = ["Count", "Closed", "check_free", "holding"]

_local = threading.local()


class Closed(RuntimeError):
    """The runner closed while work waited for its threads."""


def check_free() -> None:
    """Refuse a top-level call from a thread that holds a thread count.

    Raises:
        RuntimeError: This thread runs a calculation or a session request.
    """
    if getattr(_local, "held", 0):
        raise RuntimeError(
            "This thread is running a calculation or a session request, which holds "
            "threads; a top-level call (compute, submit, a top-level session's "
            "request) from it could wait for those very threads. A program runs its "
            "subprograms through ctx.compute and ctx.session, which share its threads."
        )


class holding:  # noqa: N801 - used as a function: ``with holding():``
    """Mark this thread as holding a thread count for a block."""

    __slots__ = ()

    def __enter__(self) -> None:
        """Mark the thread."""
        _local.held = getattr(_local, "held", 0) + 1

    def __exit__(self, *exc: object) -> None:
        """Unmark it."""
        _local.held -= 1


class Count:
    """A thread count (and GPUs), and the work running under it.

    Args:
        threads: The count: the most threads the work under it runs at once.
        gpus: The GPUs the work under it may take, each by one piece of work at a time.
        memory: The holder's host memory (GiB), which work gets its share of; None:
            unknown.
        outer: The count this one runs under (a session's holder): work taken here is
            also taken there.
        warn: Warn once when work asking for the whole count first waits (a runner's).
    """

    __slots__ = (
        "threads",
        "memory",
        "outer",
        "_free",
        "_gpus",
        "_queue",
        "_lock",
        "_ready",
        "_closed",
        "_warn",
    )

    def __init__(
        self,
        threads: int,
        gpus: Iterable[str] = (),
        memory: float | None = None,
        outer: Count | None = None,
        warn: bool = False,
    ) -> None:
        """Create a count with nothing running under it."""
        self.threads = threads
        self.memory = memory
        self.outer = outer
        self._free = threads
        self._gpus = dict.fromkeys(gpus, True)  # True: free
        self._queue: deque[object] = deque()
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._closed = False
        self._warn = warn

    @property
    def gpus(self) -> int:
        """How many GPUs the holder has."""
        return len(self._gpus)

    def add_gpus(self, gpus: Iterable[str]) -> None:
        """Give a holder made without GPUs these, when work first asks for one."""
        with self._lock:
            if not self._gpus:
                self._gpus = dict.fromkeys(gpus, True)

    def share(self, threads: int, devices: tuple[str, ...]) -> float:
        """The share of the holder that work with these threads and GPUs takes."""
        gpu = len(devices) / len(self._gpus) if self._gpus else 0.0
        return max(threads / self.threads, gpu)

    def take(
        self, threads: int, gpus: int = 0, control: _Control | None = None
    ) -> tuple[str, ...]:
        """Wait for the threads and GPUs, in turn, and take them; give them back after.

        Args:
            threads: Threads to take (at most the count).
            gpus: GPUs to take (at most the holder's).
            control: The caller's cancel and stop: either ends the wait.

        Returns:
            The GPUs taken.

        Raises:
            ProgramError: Kind ``timeout`` (cancelled) or ``stopped`` while waiting.
            Closed: The runner closed while waiting.
        """
        threads = min(threads, self.threads)
        devices: tuple[str, ...] = ()
        with self._lock:
            if self._queue or self._free < threads or gpus and not self._fits(gpus):
                self._wait(threads, gpus, control)
            self._free -= threads
            if gpus:
                devices = tuple(g for g, free in self._gpus.items() if free)[:gpus]
                for g in devices:
                    self._gpus[g] = False
        if self.outer is not None:
            try:
                self.outer.take(threads, 0, control)
            except BaseException:
                self._give(threads, devices)
                raise
        return devices

    def give(self, threads: int, devices: tuple[str, ...] = ()) -> None:
        """Give back what `take` took."""
        threads = min(threads, self.threads)
        if self.outer is not None:
            self.outer.give(threads)
        self._give(threads, devices)

    def wake(self) -> None:
        """Have waiting work look again (a cancel or stop of its caller)."""
        with self._ready:
            self._ready.notify_all()

    def close(self) -> None:
        """End every wait: waiting work raises `Closed`."""
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    def _fits(self, gpus: int) -> bool:
        return sum(self._gpus.values()) >= min(gpus, len(self._gpus))

    def _wait(self, threads: int, gpus: int, control: _Control | None) -> None:
        """Wait (holding the lock) until it is this work's turn and it fits."""
        if self._warn and threads == self.threads:
            self._warn = False
            warnings.warn(
                f"Calculations are waiting for this runner's {self.threads} threads: "
                "each one without a threads keyword takes all of them, so they run one "
                "at a time. Set threads (usually 1) on each input to run several at "
                "once, or run them as a list.",
                RuntimeWarning,
                stacklevel=5,
            )
        turn = object()
        self._queue.append(turn)
        if control is not None:
            control.waiting.append(self)
        try:
            while (
                self._queue[0] is not turn
                or self._free < threads
                or not self._fits(gpus)
            ):
                if self._closed:
                    raise Closed("This runner is closed.")
                if control is not None:
                    if control.cancelled:
                        raise ProgramError.of(
                            "timeout", "Cancelled while it waited for threads."
                        )
                    if control.stop_requested():
                        raise ProgramError.of(
                            "stopped", "Stopped while it waited for threads."
                        )
                # A cancel or stop wakes this (`wake`); a worker's stop file can't.
                polled = control is not None and control.stop_file is not None
                self._ready.wait(0.1 if polled else None)
        finally:
            self._queue.remove(turn)
            if control is not None:
                control.waiting.remove(self)
            self._ready.notify_all()

    def _give(self, threads: int, devices: tuple[str, ...]) -> None:
        with self._lock:
            self._free += threads
            for g in devices:
                self._gpus[g] = True
            if self._queue:
                self._ready.notify_all()
