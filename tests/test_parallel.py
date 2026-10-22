"""Lists of inputs: order, parallel programs, serial Python, failures, and interrupts."""

from __future__ import annotations

import os
import pickle
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import qccodec
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
    """A stand-in library call: sleeps, reports its process and GPUs, returns its value.

    ``die`` ends the process mid-calculation, as a crashing native library would.
    """
    k = dict(inp.keywords)

    def invoke() -> dict[str, Any]:
        print(f"python {k.get('value', 0)}")
        start = time.time()
        time.sleep(k.get("sleep", 0))
        if k.get("die"):
            os._exit(3)
        return {
            "value": float(k.get("value", 0)),
            "start": start,
            "end": time.time(),
            "pid": os.getpid(),
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", "-"),
            "omp": os.environ.get("OMP_NUM_THREADS", "-"),
        }

    return PythonInput(
        invoke, gpus=int(k.get("gpus", 0)), threads=int(k.get("threads", 1))
    )


def _python_decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
    value = out.value
    return SinglePointData.model_validate(
        {
            "provenance": {"program": inp.program},
            "energy": value["value"],
            "extras": {k: value.get(k) for k in ("start", "end", "pid", "gpus", "omp")},
        }
    )


PYTHON = Codec(
    "library", calctypes=list(CalcType), encode=_python, decode=_python_decode
)


@pytest.fixture(autouse=True)
def _codecs() -> Iterator[None]:
    """Register the stand-in codecs, as users register their own."""
    for codec in (PROGRAM, PYTHON):
        qccodec.register(codec)
    yield
    for codec in (PROGRAM, PYTHON):
        qccodec.unregister(codec.program)


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


def test_python_calls_run_side_by_side_in_workers(
    make: Callable[..., ProgramInput],
) -> None:
    """Python calculations of a list run in worker processes, in parallel."""
    inputs = [make("library", value=i, sleep=0.5) for i in range(4)]
    start = time.monotonic()
    outputs = compute(inputs, max_workers=4)
    assert time.monotonic() - start < 3.0  # four 0.5 s calls plus worker startup
    assert [energy(o) for o in outputs] == [0, 1, 2, 3]
    assert _peak(outputs) == 4
    assert os.getpid() not in {o.results.extras["pid"] for o in outputs}


def _unpicklable() -> Codec:
    return Codec(
        "local",
        calctypes=list(CalcType),
        encode=_python,
        decode=lambda inp, out: _python_decode(inp, out),
    )


def test_unpicklable_codecs_run_in_process_one_at_a_time(
    make: Callable[..., ProgramInput],
) -> None:
    """A registered codec a worker cannot receive runs here, serially."""
    qccodec.register(_unpicklable())
    try:
        outputs = compute([make("local", value=i, sleep=0.2) for i in range(3)])
    finally:
        qccodec.unregister("local")
    assert [energy(o) for o in outputs] == [0, 1, 2] and _peak(outputs) == 1
    assert {o.results.extras["pid"] for o in outputs} == {os.getpid()}


def test_worker_updates_reach_update_func(make: Callable[..., ProgramInput]) -> None:
    """Logs of calculations in workers arrive with their index."""
    seen: dict[int, str] = {}
    lock = threading.Lock()

    def update(index: int, all_logs: str, new_logs: str) -> None:
        with lock:
            seen[index] = all_logs

    compute([make("library", value=7), make("library", value=8)], update_func=update)
    assert "python 7" in seen[0] and "python 8" in seen[1]


def test_a_dying_worker_fails_only_its_calculation(
    make: Callable[..., ProgramInput],
) -> None:
    """A worker that exits mid-calculation gives a failed output, not a hang."""
    outputs = compute(
        [make("library", value=1), make("library", die=True), make("library", value=3)],
        raise_exc=False,
    )
    assert [o.success for o in outputs] == [True, False, True]
    assert "exited with status 3" in (outputs[1].traceback or "")
    assert energy(outputs[2]) == 3


def test_gpu_workers_are_pinned(
    make: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each GPU calculation runs in a worker started on its own device."""
    monkeypatch.setattr("qccompute.parallel.visible_gpus", lambda env=None: ["0", "1"])
    outputs = compute([make("library", sleep=0.4, gpus=1) for _ in range(4)])
    assert _peak(outputs) == 2
    assert {o.results.extras["gpus"] for o in outputs} == {"0", "1"}


def _nested(inp: ProgramInput) -> PythonInput:
    """A library call that computes a list of programs itself."""

    def invoke() -> dict[str, Any]:
        nested = [
            inp.model_copy(update={"program": "program", "keywords": {"value": v}})
            for v in (1, 2)
        ]
        outputs = compute(nested, codec=PROGRAM)
        value = sum(energy(o) or 0 for o in outputs)
        now = time.time()
        return {
            "value": value,
            "start": now,
            "end": now,
            "pid": os.getpid(),
            "gpus": "-",
        }

    return PythonInput(invoke)


NESTED = Codec(
    "nested", calctypes=list(CalcType), encode=_nested, decode=_python_decode
)


def test_lists_inside_workers_run_serially(make: Callable[..., ProgramInput]) -> None:
    """A list computed by a calculation in a worker does not start workers itself."""
    qccodec.register(NESTED)
    try:
        outputs = compute([make("nested"), make("nested")])
    finally:
        qccodec.unregister("nested")
    assert [energy(o) for o in outputs] == [3, 3]


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


# ------------------------------------------------------------------ resources


def _timed(threads: int = 1, gpus: int = 0) -> Codec:
    """A stand-in program declaring resources; prints start, end, and its GPUs."""

    def encode(inp: ProgramInput) -> SubprocessInput:
        script = (
            "import os, time; s = time.time(); time.sleep(0.4); "
            "print(s, time.time(), os.environ.get('CUDA_VISIBLE_DEVICES', '-'))"
        )
        return SubprocessInput(
            (sys.executable, "-c", script), threads=threads, gpus=gpus
        )

    def decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
        start, end, devices = out.stdout.split()
        return SinglePointData.model_validate(
            {
                "provenance": {"program": inp.program},
                "energy": 0,
                "extras": {"start": float(start), "end": float(end), "gpus": devices},
            }
        )

    return Codec("timed", calctypes=list(CalcType), encode=encode, decode=decode)


def _peak(outputs: list[ProgramOutput]) -> int:
    """The most calculations that ran at the same moment."""
    events = sorted(
        (t, step)
        for o in outputs
        for t, step in ((o.results.extras["start"], 1), (o.results.extras["end"], -1))
    )
    running = peak = 0
    for _, step in events:
        running += step
        peak = max(peak, running)
    return peak


def test_threads_share_the_budget(make: Callable[..., ProgramInput]) -> None:
    """Two-thread calculations run two at a time within four slots."""
    outputs = compute(
        [make("timed") for _ in range(4)], codec=_timed(threads=2), max_workers=4
    )
    assert _peak(outputs) == 2


def test_oversized_calculations_run_alone(make: Callable[..., ProgramInput]) -> None:
    """A calculation asking for more threads than the budget still runs, alone."""
    outputs = compute(
        [make("timed") for _ in range(3)], codec=_timed(threads=8), max_workers=2
    )
    assert all(o.success for o in outputs) and _peak(outputs) == 1


def test_gpu_calculations_get_their_own_devices(
    make: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each GPU calculation sees one device; no two share it at once."""
    monkeypatch.setattr("qccompute.parallel.visible_gpus", lambda env=None: ["0", "1"])
    outputs = compute(
        [make("timed") for _ in range(4)],
        codec=_timed(gpus=1),
        max_workers=8,
        environment={"OTHER": "kept"},
    )
    assert _peak(outputs) == 2
    assert {o.results.extras["gpus"] for o in outputs} == {"0", "1"}
    for a in outputs:
        for b in outputs:
            overlap = (
                a is not b
                and a.results.extras["start"] < b.results.extras["end"]
                and b.results.extras["start"] < a.results.extras["end"]
            )
            assert not overlap or a.results.extras["gpus"] != b.results.extras["gpus"]


def test_gpu_calculations_without_known_devices_run_one_at_a_time(
    make: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no GPUs found, GPU calculations are not pinned and do not overlap."""
    monkeypatch.setattr("qccompute.parallel.visible_gpus", lambda env=None: [])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    outputs = compute(
        [make("timed") for _ in range(3)], codec=_timed(gpus=1), max_workers=8
    )
    assert _peak(outputs) == 1
    assert {o.results.extras["gpus"] for o in outputs} == {"-"}


def test_visible_gpus_reads_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA_VISIBLE_DEVICES, from the environment argument or the process, wins."""
    from qccompute.parallel import visible_gpus

    assert visible_gpus({"CUDA_VISIBLE_DEVICES": "2, 3"}) == ["2", "3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    assert visible_gpus() == ["GPU-abc"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert visible_gpus() == []


def _workers_running() -> list[int]:
    """Worker processes started by this process that still run."""
    from qccompute.workers import running_workers

    return running_workers()


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs POSIX timers")
def test_ctrl_c_stops_the_workers(make: Callable[..., ProgramInput]) -> None:
    """KeyboardInterrupt while workers compute leaves no worker processes behind."""

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    start = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            compute([make("library", sleep=30) for _ in range(3)])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    while _workers_running() and time.monotonic() - start < 15:
        time.sleep(0.1)
    assert not _workers_running() and time.monotonic() - start < 15


def test_workers_limit_numerical_library_threads(
    make: Callable[..., ProgramInput], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers start with BLAS/OpenMP threads set to what their task was granted."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    outputs = compute([make("library", threads=2) for _ in range(2)], max_workers=4)
    assert {o.results.extras["omp"] for o in outputs} == {"2"}
    monkeypatch.setenv("OMP_NUM_THREADS", "5")  # the caller's own setting wins
    outputs = compute([make("library") for _ in range(2)], max_workers=4)
    assert {o.results.extras["omp"] for o in outputs} == {"5"}
