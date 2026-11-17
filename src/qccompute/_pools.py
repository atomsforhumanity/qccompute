"""A calculation's BLAS and OpenMP thread pools, held to its threads while it runs.

The runner holds the numerical libraries loaded in a process (OpenBLAS, MKL, BLIS,
FlexiBLAS and the OpenMP runtimes, as ``threadpoolctl`` finds them) at a calculation's
threads around each call of its program, in its own process and in workers. So a
calculation's numbers do not depend on where it ran, on the caller's environment, or on
the threads of the call that started a worker: multithreaded BLAS is not bitwise
reproducible across thread counts.

- Some counts belong to the calling thread (OpenMP's, MKL's, and OpenBLAS's when it
  runs on OpenMP): a call sets them on its own thread and restores them after, so calls
  in different threads never see each other's.
- Others belong to the process (OpenBLAS on its own threads, BLIS, FlexiBLAS). A call
  sets them and, on return, restores what it found, so a calculation run inside another
  (a subprogram) gets its own threads and gives its parent back the parent's.
  Calculations with different threads running at once in one process share them: the
  one started last holds them. Once none runs, every count is back where it was.

A program may hold its own small linear algebra lower while it works, if it restores
the counts around its subprograms' calls (qcopt pins its steps to one thread).

The libraries are found once, and again after the process imported modules (finding
them takes about 0.4 ms). Each library's own functions are called directly: a call
reads each count and sets only those that differ.
"""

from __future__ import annotations

import itertools
import sys
import threading
import warnings
from collections.abc import Callable
from typing import Any

__all__ = ["hold_threads"]

Get = Callable[[], int]
Set = Callable[[int], Any]

# Each library's get and set functions, by threadpoolctl's internal_api, and whether
# its count is the calling thread's own.
_FUNCTIONS: dict[str, tuple[str, str, bool]] = {
    "openmp": ("omp_get_max_threads", "omp_set_num_threads", True),
    "mkl": ("MKL_Get_Max_Threads", "MKL_Set_Num_Threads_Local", True),
    "openblas": ("openblas_get_num_threads", "openblas_set_num_threads", False),
    "blis": ("bli_thread_get_num_threads", "bli_thread_set_num_threads", False),
    "flexiblas": ("flexiblas_get_num_threads", "flexiblas_set_num_threads", False),
}


def _functions(lib: Any) -> tuple[Get, Set, bool] | None:
    """A threadpoolctl library controller's get and set functions, and if per thread.

    None for a library this module does not know. OpenBLAS's symbols may carry an
    affix (``scipy_openblas_get_num_threads64_`` in numpy's wheels); OpenBLAS built on
    OpenMP takes its count from OpenMP, per thread.
    """
    known = _FUNCTIONS.get(lib.internal_api)
    if known is None:
        return None
    get_name, set_name, per_thread = known
    dynlib = lib.dynlib
    if lib.internal_api == "openblas":
        if getattr(lib, "threading_layer", None) == "openmp":
            get_name, set_name, per_thread = _FUNCTIONS["openmp"]
        else:
            for prefix, suffix in itertools.product(("", "scipy_"), ("", "64_", "_64")):
                if hasattr(dynlib, f"{prefix}{get_name}{suffix}"):
                    get_name = f"{prefix}{get_name}{suffix}"
                    set_name = f"{prefix}{set_name}{suffix}"
                    break
    get = getattr(dynlib, get_name, None)
    set_ = getattr(dynlib, set_name, None)
    if get is None or set_ is None:
        return None
    return get, set_, per_thread


class _Pools:
    """The libraries of this process and the calls holding them (one per process)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.modules = -1  # len(sys.modules) when the libraries were last found
        self.shared: list[tuple[Get, Set]] = []  # process-wide counts
        self.local: list[tuple[Get, Set]] = []  # the calling thread's counts
        self.outside: list[int] = []  # the shared counts before the first call ran
        self.running: list[object] = []  # running calls, in the order they started

    def find(self) -> None:
        """Find the loaded libraries again (under the lock, while no call runs)."""
        self.modules = len(sys.modules)
        try:
            from threadpoolctl import ThreadpoolController

            found = [_functions(lib) for lib in ThreadpoolController().lib_controllers]
        except Exception as e:  # noqa: BLE001 - a calculation never fails on this
            warnings.warn(
                f"qccompute cannot find this process's BLAS and OpenMP libraries "
                f"({type(e).__name__}: {e}); their thread counts are left as they are.",
                RuntimeWarning,
                stacklevel=4,
            )
            found = []
        self.shared = [(f[0], f[1]) for f in found if f is not None and not f[2]]
        self.local = [(f[0], f[1]) for f in found if f is not None and f[2]]


_POOLS = _Pools()


class hold_threads:  # noqa: N801 - used as a function: ``with hold_threads(n):``
    """Hold the process's BLAS and OpenMP pools at ``threads`` for a block.

    Args:
        threads: The calculation's threads; None holds nothing (no grant).
    """

    __slots__ = ("threads", "found", "local")

    def __init__(self, threads: int | None) -> None:
        """Hold ``threads`` when the block starts."""
        self.threads = threads

    def __enter__(self) -> None:
        """Set each count that differs to the threads, noting what it was."""
        threads = self.threads
        if threads is None:
            return
        pools = _POOLS
        found: list[int] = []
        with pools.lock:
            if not pools.running:
                if len(sys.modules) != pools.modules:
                    pools.find()
                pools.outside = found
            for get, set_ in pools.shared:
                n = get()
                found.append(n)
                if n != threads:
                    set_(threads)
            pools.running.append(self)
        self.found = found
        local: list[tuple[Set, int]] = []
        for get, set_ in pools.local:  # this thread's own counts: no lock
            n = get()
            if n != threads:
                set_(threads)
                local.append((set_, n))
        self.local = local

    def __exit__(self, *exc: object) -> None:
        """Give back what the block found; the last call out restores the outside."""
        threads = self.threads
        if threads is None:
            return
        for set_, n in self.local:
            set_(n)
        pools = _POOLS
        with pools.lock:
            running = pools.running
            latest = running[-1] is self
            running.remove(self)
            if not running:  # the last out: the counts from before, exactly
                for (get, set_), n in zip(pools.shared, pools.outside, strict=True):
                    # Equal to ours, it may still differ: calls that overlapped
                    # this one may have left their own.
                    if n != threads or get() != n:
                        set_(n)
            elif latest:  # give back what it found (a parent's threads)
                for (_, set_), n in zip(pools.shared, self.found, strict=True):
                    if n != threads:
                        set_(n)
            # else a later call still runs, and holds the counts
