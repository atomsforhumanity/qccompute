"""Batches: inputs a codec can evaluate together run in one call of its batch function."""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Iterator, Sequence
from typing import Any

import pytest
import qccodec
from qccodec import Codec, NativeOutput, PythonInput
from qcdata import CalcType, ProgramInput, ProgramOutput, SinglePointData

from qccompute import compute
from qccompute.exceptions import ComputeErrors

# The size of every call of the batch function, and each single call as 1.
calls: list[int] = []


class OutOfMemoryError(RuntimeError):
    """Named like torch.OutOfMemoryError."""


def _value(inp: ProgramInput) -> dict[str, Any]:
    if inp.keywords.get("poison"):
        raise ValueError("poisoned input")
    return {"value": float(inp.keywords.get("value", 0)), "pid": os.getpid()}


def _encode(inp: ProgramInput) -> PythonInput:
    def invoke() -> dict[str, Any]:
        calls.append(1)
        print("alone")
        return _value(inp)

    return PythonInput(invoke)


def _encode_batch(inputs: Sequence[ProgramInput]) -> PythonInput:
    def invoke() -> list[dict[str, Any]]:
        calls.append(len(inputs))
        print(f"batch of {len(inputs)}")
        if len(inputs) > inputs[0].keywords.get("fits", 1000):
            raise OutOfMemoryError("CUDA out of memory")
        values = [_value(inp) for inp in inputs]
        return values[:-1] if inputs[0].keywords.get("short") else values

    return PythonInput(invoke)


def _batch_key(inp: ProgramInput) -> Hashable | None:
    return inp.keywords.get("group", "default")


def _decode(inp: ProgramInput, out: NativeOutput) -> SinglePointData:
    if inp.keywords.get("bad"):
        raise ValueError("cannot decode")
    return SinglePointData.model_validate(
        {
            "provenance": {"program": inp.program},
            "energy": out.value["value"],
            "gradient": [[0.0, 0.0, 0.0]] * len(inp.require_structure().symbols),
            "extras": {"pid": out.value["pid"]},
        }
    )


BATCHER = Codec(
    "batcher",
    calctypes=list(CalcType),
    encode=_encode,
    decode=_decode,
    encode_batch=_encode_batch,
    batch_key=_batch_key,
    max_batch=4,
)


@pytest.fixture(autouse=True)
def _codec() -> Iterator[None]:
    """Register the stand-in codec and forget earlier calls."""
    qccodec.register(BATCHER)
    calls.clear()
    yield
    qccodec.unregister(BATCHER.program)


@pytest.fixture
def make(
    prog_input_factory: Callable[..., ProgramInput],
) -> Callable[..., ProgramInput]:
    """Build an input for the stand-in codec with keywords."""

    def build(**keywords: Any) -> ProgramInput:
        inp = prog_input_factory("gradient")
        return inp.model_copy(update={"program": "batcher", "keywords": keywords})

    return build


def energies(outputs: list[ProgramOutput]) -> list[float | None]:
    """The energies of stand-in outputs."""
    return [o.results.energy for o in outputs]


def test_inputs_sharing_a_key_run_as_one_batch_here(
    make: Callable[..., ProgramInput],
) -> None:
    """One call computes them in this process; each output records its place."""
    outputs = compute([make(value=i) for i in range(3)])
    assert calls == [3]
    assert energies(outputs) == [0, 1, 2] and all(o.success for o in outputs)
    assert {o.results.extras["pid"] for o in outputs} == {os.getpid()}
    assert [o.execution.extras for o in outputs] == [
        {"batch_size": 3, "batch_index": i} for i in range(3)
    ]
    assert all("batch of 3" in (o.logs or "") for o in outputs)


def test_batches_hold_at_most_max_batch_in_even_chunks(
    make: Callable[..., ProgramInput],
) -> None:
    """Nine inputs with max_batch 4 run as three batches of three."""
    outputs = compute([make(value=i) for i in range(9)])
    assert calls == [3, 3, 3] and energies(outputs) == list(range(9))


def test_keys_separate_batches_and_none_runs_alone(
    make: Callable[..., ProgramInput],
) -> None:
    """Only inputs with equal keys share a batch; a key of None never batches."""
    inputs = [
        make(value=0, group="a"),
        make(value=1, group="b"),
        make(value=2, group="a"),
        make(value=3, group=None),
        make(value=4, group="b"),
        make(value=5, group="c"),
    ]
    outputs = compute(inputs, max_workers=1)
    assert sorted(calls) == [1, 1, 2, 2]
    assert energies(outputs) == list(range(6))
    assert "batch_size" not in outputs[3].execution.extras
    assert "batch_size" not in outputs[5].execution.extras


def test_out_of_memory_halves_the_batch(make: Callable[..., ProgramInput]) -> None:
    """A batch too big for the device is split in half until it fits."""
    outputs = compute([make(value=i, fits=1) for i in range(4)])
    assert calls == [4, 2, 1, 1, 2, 1, 1]
    assert energies(outputs) == [0, 1, 2, 3] and all(o.success for o in outputs)


def test_a_failed_batch_reruns_its_inputs_alone(
    make: Callable[..., ProgramInput],
) -> None:
    """Only the input that breaks the batch fails, with its own error."""
    inputs = [make(value=0), make(value=1, poison=True), make(value=2)]
    with pytest.raises(ComputeErrors) as info:
        compute(inputs, max_workers=1)
    assert calls == [3, 1, 1, 1]
    (error,) = info.value.exceptions
    assert error.input_index == 1  # type: ignore[union-attr]
    assert [o.success for o in info.value.outputs] == [True, False, True]


def test_a_batch_returning_the_wrong_count_reruns_alone(
    make: Callable[..., ProgramInput],
) -> None:
    """Values that cannot be matched to inputs are not decoded."""
    outputs = compute([make(value=i, short=True) for i in range(2)], max_workers=1)
    assert calls == [2, 1, 1] and energies(outputs) == [0, 1]


def test_a_decoding_failure_fails_only_its_input(
    make: Callable[..., ProgramInput],
) -> None:
    """Each value decodes on its own; the batch is not rerun."""
    inputs = [make(value=0), make(value=1, bad=True)]
    outputs = compute(inputs, raise_exc=False)
    assert calls == [2]
    assert [o.success for o in outputs] == [True, False]
    assert "cannot decode" in (outputs[1].traceback or "")
    with pytest.raises(ComputeErrors) as info:
        compute(inputs)
    assert [e.input_index for e in info.value.exceptions] == [1]  # type: ignore[union-attr]


def test_update_func_gets_the_batch_logs_for_each_input(
    make: Callable[..., ProgramInput],
) -> None:
    """Every input of a batch receives the batch's logs with its index."""
    seen: dict[int, str] = {}

    def update(index: int, all_logs: str, new_logs: str) -> None:
        seen[index] = all_logs

    compute([make(value=0), make(value=1)], update_func=update)
    assert seen == {0: "batch of 2\n", 1: "batch of 2\n"}


def test_file_collection_and_single_inputs_do_not_batch(
    make: Callable[..., ProgramInput],
) -> None:
    """Batches share a directory, so collecting files computes each alone."""
    compute([make(value=0), make(value=1)], collect_files=True, max_workers=1)
    compute(make(value=2))
    assert calls == [1, 1, 1]
    # Batching would hide that these inputs cannot propagate wavefunctions.
    with pytest.raises(ComputeErrors):
        compute([make(value=0), make(value=1)], propagate_wfn=True, max_workers=1)
    assert calls == [1, 1, 1]


def test_print_logs_prints_a_batch_once(
    make: Callable[..., ProgramInput], capsys: pytest.CaptureFixture[str]
) -> None:
    """The shared log is printed once, prefixed with every index of the batch."""
    compute([make(value=0), make(value=1)], print_logs=True)
    assert capsys.readouterr().out == "[0,1] batch of 2\n"
