"""Tests and fixtures for test exceptions."""

import pickle

import pytest
from qcdata import ProgramInput, ProgramOutput, SinglePointData

from qccompute import exceptions
from qccompute.exceptions import (
    CodecNotFoundError,
    ExternalProgramError,
    InputError,
    ProgramNotFoundError,
    QCComputeBaseError,
)

test_data = [
    (CodecNotFoundError, ("psi4",)),
    (InputError, ("psi4",)),
    (ExternalProgramError, ("psi4",)),
    (ProgramNotFoundError, ("psi4",)),
]


def test_adapter_names_remain_available() -> None:
    """Deprecated Adapter-era names still catch the new exceptions."""
    assert exceptions.AdapterNotFoundError is CodecNotFoundError
    assert exceptions.AdapterInputError is InputError
    assert issubclass(CodecNotFoundError, exceptions.AdapterError)


@pytest.mark.parametrize("exc_class,args", test_data)
def test_exception_pickle(
    results: ProgramOutput[ProgramInput, SinglePointData],
    exc_class: type[QCComputeBaseError],
    args: tuple[str],
) -> None:
    """Preserve exception attributes across process boundaries."""
    exc_instance = exc_class(*args)

    # Append ProgramOutput
    exc_instance.prog_output = results

    # Perform a full pickle round-trip.
    pickled = pickle.dumps(exc_instance)
    unpickled = pickle.loads(pickled)

    # Assert that the type is preserved.
    assert type(unpickled) is exc_class, (
        f"{exc_class.__name__} type changed after unpickling."
    )

    for attr in exc_instance.__dict__:
        assert getattr(unpickled, attr) == getattr(exc_instance, attr), (
            f"Attribute {attr} changed after unpickling."
        )
