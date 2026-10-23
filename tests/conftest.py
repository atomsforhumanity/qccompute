"""Tests and fixtures for conftest."""

from collections.abc import Callable

import numpy as np
import pytest
from qcdata import (
    CalcType,
    Model,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
    Structure,
)


@pytest.fixture(scope="session")
def hydrogen() -> Structure:
    """Create a Hydrogen structure object."""
    return Structure(
        symbols=["H", "H"],
        # Integration test depend upon this geometry; do not change
        geometry=np.array([[0, 0, 0], [0, 0, 1.4]]),
    )


@pytest.fixture(scope="session")
def water() -> Structure:
    """Create a water structure object."""
    return Structure(
        symbols=["O", "H", "H"],
        # Integration test depend upon this geometry; do not change
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.524, 1.687, 0.480],
                [1.146, -0.450, -1.354],
            ]
        ),
    )


@pytest.fixture(scope="session")
def prog_input_factory(hydrogen: Structure) -> Callable[..., ProgramInput]:
    """Return a factory that creates ProgramInput instances with specified calctypes."""

    def create_program_input(
        calctype: str | CalcType, program: str = "test"
    ) -> ProgramInput:
        """Create program input; a path optimization gets a two-structure chain."""
        chain = CalcType(calctype) == CalcType.path_optimization
        return ProgramInput(
            program=program,
            structure=None if chain else hydrogen,
            structures={"initial_chain": [hydrogen, hydrogen]} if chain else {},
            calctype=CalcType(calctype),
            model=Model(method="hf", basis="sto-3g"),
            keywords={"purify": "no", "some-bool": False},
        )

    return create_program_input


@pytest.fixture
def results(
    prog_input_factory: Callable[..., ProgramInput],
) -> ProgramOutput[ProgramInput, SinglePointData]:
    """Create a ProgramOutput object."""
    sp_inp_energy = prog_input_factory("energy", program="test")
    energy = 1.0
    n_atoms = len(sp_inp_energy.require_structure().symbols)
    gradient = np.arange(n_atoms * 3).reshape(n_atoms, 3)
    hessian = np.arange(n_atoms**2 * 3**2).reshape(n_atoms * 3, n_atoms * 3)

    return ProgramOutput[ProgramInput, SinglePointData].model_validate(
        {
            "input_data": sp_inp_energy,
            "success": True,
            "logs": "program standard out...",
            "results": {
                "provenance": {"program": "qcdata-test-suite"},
                "energy": energy,
                "gradient": gradient,
                "hessian": hessian,
            },
            "execution": {"scratch_dir": "/tmp/qcdata"},
            "extras": {"some_extra": 1},
        }
    )
