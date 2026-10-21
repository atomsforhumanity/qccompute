"""Lists of inputs: order, parallel programs, serial Python, failures, and interrupts."""

from __future__ import annotations

import pickle
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
from qccodec import Codec, NativeOutput, PythonInput, SubprocessInput
from qcdata import CalcType, ProgramInput, ProgramOutput, SinglePointData

from qccompute import compute, runners
from qccompute.exceptions import ComputeErrors, ExternalProgramError, InputError


def _program(inp: ProgramInput) -> SubprocessInput:
    """A stand-in program that sleeps, prints its value, and exits with a code."""
    k = inp.keywords
    script = (
        f"import sys, time; time.sleep({k.get('sleep', 0)}); "
        f"print({k.get('value', 0)}); sys.exit({k.get('exit', 0)})"
    )
    return SubprocessInput((sys.executable, "-c", script))


def _decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
    value = float(out.stdout.split()[-1]) if out.stdout.strip() else 0.0
    return SinglePointData.model_validate(
        {"provenance": {"program": inp.program}, "energy": value}
    )


def energy(output: ProgramOutput) -> float | None:
    """The energy of a stand-in single point."""
    assert isinstance(output.results, SinglePointData)
    return output.results.energy


PROGRAM = Codec("program", calctypes=list(CalcType), encode=_program, decode=_decode)


def _python(inp: ProgramInput) -> PythonInput:
    """A stand-in library call that returns its value."""
    value = inp.keywords.get("value", 0)

    def invoke() -> float:
        print(f"python {value}")
        return float(value)

    return PythonInput(invoke)


PYTHON = Codec(
    "library",
    calctypes=list(CalcType),
    encode=_python,
    decode=lambda inp, out: SinglePointData.model_validate(
        {"provenance": {"program": inp.program}, "energy": out.value}
    ),
)


@pytest.fixture(autouse=True)
def _codecs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the stand-in codecs by program name."""
    codecs = {"program": PROGRAM, "library": PYTHON}
    monkeypatch.setattr("qccodec.get_codec", lambda inp: codecs[inp.program])


@pytest.fixture
def make(
    prog_input_factory: Callable[..., ProgramInput],
) -> Callable[..., ProgramInput]:
    """Build an input for a stand-in program with keywords."""

    def build(program: str = "program", **keywords: Any) -> ProgramInput:
        inp = prog_input_factory("energy")
        return inp.model_copy(update={"program": program, "keywords": keywords})

    return build


def test_a_single_input_returns_a_single_output(
    make: Callable[..., ProgramInput],
) -> None:
    """The single-input path is unchanged."""
    output = compute(make(value=1.5))
    assert isinstance(output, ProgramOutput) and energy(output) == 1.5


def test_lists_return_outputs_in_input_order(make: Callable[..., ProgramInput]) -> None:
    """Programs and Python calls mix; outputs come back in input order."""
    inputs = [
        make(value=1, sleep=0.3),
        make("library", value=2),
        make(value=3),
        make("library", value=4),
    ]
    outputs = compute(inputs)
    assert [energy(o) for o in outputs] == [1, 2, 3, 4]
    assert all(o.success for o in outputs)
    assert energy(compute(tuple(inputs[:2]))[1]) == 2
    assert compute([]) == []


def test_programs_run_in_parallel(make: Callable[..., ProgramInput]) -> None:
    """Four half-second programs take about half a second with four workers."""
    inputs = [make(value=i, sleep=0.5) for i in range(4)]
    start = time.monotonic()
    compute(inputs, max_workers=4)
    parallel = time.monotonic() - start
    start = time.monotonic()
    compute(inputs, max_workers=1)
    serial = time.monotonic() - start
    assert parallel < 1.5 < 2.0 <= serial


def test_python_calls_never_overlap(
    make: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python calculations run one at a time, after the programs."""
    active, peak = 0, 0
    lock = threading.Lock()
    original = runners.run_python

    def counting(*args: Any, **kwargs: Any) -> Any:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr("qccompute.main.run_python", counting)
    outputs = compute([make("library", value=i) for i in range(6)], max_workers=6)
    assert [energy(o) for o in outputs] == list(range(6)) and peak == 1


def test_failures_are_raised_together_after_everything_ran(
    make: Callable[..., ProgramInput],
) -> None:
    """One ComputeErrors holds each failure and every output."""
    inputs = [make(value=1), make(value=2, exit=3), make(value=3), make(exit=1)]
    with pytest.raises(ComputeErrors) as caught:
        compute(inputs)
    group = caught.value
    assert str(group).startswith("2 of 4 calculations failed")
    assert [getattr(e, "input_index", None) for e in group.exceptions] == [1, 3]
    assert all(isinstance(e, ExternalProgramError) for e in group.exceptions)
    assert [o.success for o in group.outputs] == [True, False, True, False]
    assert energy(group.outputs[2]) == 3
    restored = pickle.loads(pickle.dumps(group))
    assert len(restored.outputs) == 4 and len(restored.exceptions) == 2


def test_except_star_and_returning_failures(make: Callable[..., ProgramInput]) -> None:
    """Failures can be handled by type, or returned with raise_exc=False."""
    inputs = [make(value=1), make(exit=2)]
    try:
        compute(inputs)
    except* ExternalProgramError as group:
        assert len(group.exceptions) == 1
        assert len(group.outputs) == 2  # type: ignore[attr-defined]
    outputs = compute(inputs, raise_exc=False)
    assert [o.success for o in outputs] == [True, False]


def test_update_func_gets_the_index_and_print_logs_prefixes_it(
    make: Callable[..., ProgramInput], capsys: pytest.CaptureFixture[str]
) -> None:
    """Callbacks know which input they report on; printed lines say so too."""
    seen: dict[int, str] = {}
    lock = threading.Lock()

    def update(index: int, all_logs: str, new_logs: str) -> None:
        with lock:
            seen[index] = all_logs

    compute([make(value=7), make("library", value=8)], update_func=update)
    assert "7" in seen[0] and "python 8" in seen[1]
    compute([make(value=5), make("library", value=6)], print_logs=True)
    printed = capsys.readouterr().out
    assert "[0] 5" in printed and "[1] python 6" in printed


def test_invalid_list_options(make: Callable[..., ProgramInput], tmp_path: Any) -> None:
    """One scratch directory cannot hold many calculations; workers must be positive."""
    with pytest.raises(InputError, match="scratch_dir"):
        compute([make(), make()], scratch_dir=tmp_path)
    with pytest.raises(InputError, match="max_workers"):
        compute([make()], max_workers=0)


def test_nested_lists_run_serially(
    make: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A list computed inside a Python calculation does not start a pool."""
    nested = [make(value=1), make(value=2)]

    def outer(inp: ProgramInput) -> PythonInput:
        return PythonInput(lambda: [energy(o) for o in compute(nested)])

    codec = Codec(
        "outer",
        calctypes=list(CalcType),
        encode=outer,
        decode=lambda inp, out: SinglePointData.model_validate(
            {"provenance": {"program": "outer"}, "energy": sum(out.value)}
        ),
    )
    monkeypatch.setattr(
        "qccompute.parallel.ThreadPoolExecutor",
        lambda *a, **k: pytest.fail("a nested list started a thread pool"),
    )
    assert energy(compute(make("outer"), codec=codec)) == 3


def test_kill_running_stops_every_program(make: Callable[..., ProgramInput]) -> None:
    """Interrupting a list kills the programs it started (they ignore Ctrl-C)."""
    results: list[Any] = []
    worker = threading.Thread(
        target=lambda: results.append(
            compute([make(sleep=30), make(sleep=30)], raise_exc=False)
        )
    )
    start = time.monotonic()
    worker.start()
    while len(runners._running) < 2 and time.monotonic() - start < 10:
        time.sleep(0.05)
    runners.kill_running()
    worker.join(timeout=10)
    assert not worker.is_alive() and time.monotonic() - start < 10
    assert [o.success for o in results[0]] == [False, False]


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX timers")
def test_ctrl_c_kills_the_programs_of_a_list(make: Callable[..., ProgramInput]) -> None:
    """KeyboardInterrupt while a list runs stops it and leaves no programs running."""

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            compute([make(sleep=30), make(sleep=30), make(sleep=30)], max_workers=2)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    while runners._running and time.monotonic() - start < 10:
        time.sleep(0.05)
    assert not runners._running and time.monotonic() - start < 10
