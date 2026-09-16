import pickle

import pytest

from qccompute.exceptions import (
    AdapterInputError,
    AdapterNotFoundError,
    ExternalProgramError,
    ProgramNotFoundError,
)

test_data = [
    (AdapterNotFoundError, ("psi4",)),
    (AdapterInputError, ("psi4",)),
    (ExternalProgramError, ("psi4",)),
    (ProgramNotFoundError, ("psi4",)),
]


@pytest.mark.parametrize("exc_class,args", test_data)
def test_exception_pickle(results, exc_class, args):
    # Instantiate the exception.
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
