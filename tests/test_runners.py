"""Exercise generic native execution and process-state restoration."""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from qccodec import (
    PythonInput,
    PythonOutput,
    SubprocessInput,
)

from qccompute.exceptions import ProgramNotFoundError, ProgramTimeoutError
from qccompute.runners import run_python, run_subprocess


def test_subprocess_merged_output_and_nonzero(tmp_path: Path) -> None:
    """Preserve stream ordering, large output, and the final callback on failure."""
    request = SubprocessInput(
        (
            sys.executable,
            "-c",
            "import os; os.write(1, b'first\\n'); os.write(2, b'e'*200000); "
            "os.write(1, b'last'); raise SystemExit(7)",
        ),
    )
    updates = []
    output = run_subprocess(
        request,
        tmp_path,
        update_func=lambda all, new: updates.append((all, new)),
        update_interval=3600,
    )
    expected = "first\n" + "e" * 200000 + "last"
    assert output.returncode == 7 and output.failed
    assert output.stdout == expected
    assert updates == [(expected, expected)]


def test_subprocess_callback_failure_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain logs and close the child when a live update raises."""
    popen = subprocess.Popen
    children: list[subprocess.Popen[str]] = []

    def start(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        """Record the real child so cleanup can be checked after failure."""
        child = popen(*args, **kwargs)
        children.append(child)
        return child

    def fail(all_logs: str, new_logs: str) -> None:
        """Abort execution from a live callback."""
        raise RuntimeError("callback failed")

    monkeypatch.setattr(subprocess, "Popen", start)
    output = run_subprocess(
        SubprocessInput(
            (
                sys.executable,
                "-c",
                "import time; print('ready', flush=True); time.sleep(60)",
            )
        ),
        tmp_path,
        update_func=fail,
        update_interval=0,
    )
    assert isinstance(output.error, RuntimeError)
    assert output.stdout == "ready\n"
    assert children[0].poll() is not None
    assert children[0].stdout is not None and children[0].stdout.closed


def test_subprocess_environment_and_cwd_are_child_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subprocess environment and cwd are child local."""
    monkeypatch.setenv("QCC_TEST_ENV", "parent")
    cwd = Path.cwd()
    request = SubprocessInput(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ['QCC_TEST_ENV']); print(os.getcwd())",
        )
    )
    paths = [tmp_path / str(i) for i in range(2)]
    for path in paths:
        path.mkdir()
    with ThreadPoolExecutor(2) as pool:
        outputs = list(
            pool.map(
                lambda path: run_subprocess(
                    request, path, environment={"QCC_TEST_ENV": path.name}
                ),
                paths,
            )
        )
    for path, output in zip(paths, outputs, strict=True):
        assert output.stdout == f"{path.name}\n{path}\n"
    assert os.environ["QCC_TEST_ENV"] == "parent" and Path.cwd() == cwd


def test_missing_executable_is_native_failure(tmp_path: Path) -> None:
    """Missing executable is native failure."""
    output = run_subprocess(SubprocessInput(("nonexistent-qcc-program",)), tmp_path)
    assert isinstance(output.error, ProgramNotFoundError) and output.failed
    assert isinstance(output.error.__cause__, FileNotFoundError)


def test_missing_working_directory_is_not_missing_program(tmp_path: Path) -> None:
    """Distinguish a launch directory failure from an unavailable executable."""
    output = run_subprocess(
        SubprocessInput((sys.executable, "-c", "pass")), tmp_path / "missing"
    )
    assert isinstance(output.error, FileNotFoundError)
    assert output.failed


def test_executable_symlink_preserves_virtual_environment(tmp_path: Path) -> None:
    """Executable symlink preserves virtual environment."""
    output = run_subprocess(
        SubprocessInput((sys.executable, "-c", "import sys; print(sys.prefix)")),
        tmp_path,
    )
    assert output.stdout.strip() == sys.prefix


def test_invocation_environment_defaults_and_user_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invocation environment defaults and user overrides."""
    monkeypatch.setenv("QCC_TEST_ENV", "parent")
    request = PythonInput(
        lambda: os.environ["QCC_TEST_ENV"], environment={"QCC_TEST_ENV": "codec"}
    )
    assert run_python(request, tmp_path).value == "codec"
    assert (
        run_python(request, tmp_path, environment={"QCC_TEST_ENV": "user"}).value
        == "user"
    )
    assert os.environ["QCC_TEST_ENV"] == "parent"


def test_python_captures_large_native_output_and_restores_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python captures large native output and restores globals."""
    monkeypatch.setenv("QCC_TEST_ENV", "")
    before = Path.cwd()
    logger = logging.getLogger("fake-native")
    state = (logger.handlers[:], logger.level, logger.propagate)

    def invoke() -> None:
        """Exercise the callable boundary with deterministic native behavior."""
        assert Path.cwd() == tmp_path
        assert os.environ["QCC_TEST_ENV"] == "child"
        print("python")
        os.write(1, b"x" * 200000)
        os.write(2, b"native stderr")
        ctypes.CDLL(None).printf(b"buffered C output")
        logger.info("logger")
        raise RuntimeError("native failed")

    output = run_python(
        PythonInput(invoke, loggers=("fake-native",)),
        tmp_path,
        environment={"QCC_TEST_ENV": "child"},
    )
    assert output.failed and "native failed" in output.traceback
    assert "python" in output.stdout and "logger" in output.stdout
    assert "buffered C output" in output.stdout
    assert output.stdout.count("x") == 200000
    assert "native stderr" in output.stdout
    assert Path.cwd() == before and os.environ["QCC_TEST_ENV"] == ""
    assert (logger.handlers, logger.level, logger.propagate) == state


def test_nested_python_invocations_restore_parent(tmp_path: Path) -> None:
    """Nested python invocations restore parent."""
    inner = tmp_path / "inner"
    inner.mkdir()

    def outer(*, child: Callable[[], PythonOutput]) -> Any:
        """Exercise a synchronous child call while preserving the parent workspace."""
        print("before")
        result = child()
        assert Path.cwd() == tmp_path
        print("after")
        return result.value

    def callback() -> PythonOutput:
        """Run a nested Python invocation in the inner workspace."""
        return run_python(PythonInput(lambda: 42), inner)

    output = run_python(
        PythonInput(outer, child_callback="child"), tmp_path, compute_child=callback
    )
    assert output.value == 42 and output.stdout == "before\nafter\n"


def test_python_merged_order_and_reporting_failure(tmp_path: Path) -> None:
    """Preserve merged write order and the invocation error if reporting also fails."""
    original = ValueError("calculation failed")
    updates = []

    def invoke() -> None:
        """Write through Python and native streams before failing."""
        print("python", end="|", flush=True)
        os.write(2, b"native stderr|")
        os.write(1, b"native stdout|")
        print("python stderr", file=sys.stderr, flush=True)
        raise original

    def report(all_logs: str, new_logs: str) -> None:
        """Check restored state and simulate a reporting error."""
        assert Path.cwd() != tmp_path
        updates.append((all_logs, new_logs))
        raise RuntimeError("reporting failed")

    output = run_python(PythonInput(invoke), tmp_path, update_func=report)
    expected = "python|native stderr|native stdout|python stderr\n"
    assert output.stdout == expected
    assert updates == [(expected, expected)]
    assert output.error is original
    assert output.traceback is not None and "calculation failed" in output.traceback


def test_python_environment_applies_during_the_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A codec's environment is in place while its Python function runs, then undone.

    Native libraries (OpenMP, BLAS) read settings such as OMP_NUM_THREADS when they
    are first imported inside the function, so the environment must already be set.
    """
    observed: list[str] = []

    def function() -> None:
        """Record the thread setting a native import would see."""
        observed.append(os.environ["OMP_NUM_THREADS"])

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    request = PythonInput(function=function, environment={"OMP_NUM_THREADS": "1"})
    output = run_python(request, tmp_path)
    assert not output.failed and observed == ["1"]
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_crest_child_openmp_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CREST's executable starts with one OpenMP thread without changing its parent."""
    from dataclasses import replace

    from qccodec import get_codec
    from qcdata import ProgramInput
    from qcdata.utils import water

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    request = get_codec("crest").encode(
        ProgramInput.model_validate(
            {
                "program": "crest",
                "calctype": "energy",
                "structure": water,
                "model": {"method": "gfn2"},
            }
        )
    )
    assert isinstance(request, SubprocessInput)
    request = replace(
        request,
        command=(
            sys.executable,
            "-c",
            "import os; print(os.environ['OMP_NUM_THREADS'])",
        ),
    )
    output = run_subprocess(request, tmp_path)
    assert output.stdout == "1\n" and not output.failed
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_subprocess_timeout_kills_program_tree(tmp_path: Path) -> None:
    """A timeout kills the program and its children and keeps earlier output."""
    script = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "open('child.pid', 'w').write(str(child.pid)); "
        "print('started', flush=True); time.sleep(60)"
    )
    start = time.monotonic()
    output = run_subprocess(
        SubprocessInput((sys.executable, "-c", script)), tmp_path, timeout=1.0
    )
    assert time.monotonic() - start < 30
    assert isinstance(output.error, ProgramTimeoutError) and output.failed
    assert "1 s timeout" in str(output.error)
    assert output.stdout == "started\n"
    if hasattr(os, "killpg"):
        child = int((tmp_path / "child.pid").read_text())
        for _ in range(50):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail("child process survived the timeout")


def test_subprocess_within_timeout_is_unaffected(tmp_path: Path) -> None:
    """A program that finishes in time behaves exactly as without a timeout."""
    output = run_subprocess(
        SubprocessInput((sys.executable, "-c", "print('ok')")), tmp_path, timeout=30
    )
    assert output.error is None and output.returncode == 0 and output.stdout == "ok\n"
