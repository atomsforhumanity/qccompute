import numpy as np
import pytest
from qcdata import (
    CalcType,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    SinglePointData,
    Structure,
)

from qccompute.adapters.base import ProgramAdapter
from qccompute.utils import prog_available


@pytest.fixture(scope="session")
def hydrogen():
    """Create a Hydrogen structure object."""
    return Structure(
        symbols=["H", "H"],
        # Integration test depend upon this geometry; do not change
        geometry=[[0, 0, 0], [0, 0, 1.4]],
    )


@pytest.fixture(scope="session")
def water():
    """Create a water structure object."""
    return Structure(
        symbols=["O", "H", "H"],
        # Integration test depend upon this geometry; do not change
        geometry=[
            [0.0, 0.0, 0.0],
            [0.524, 1.687, 0.480],
            [1.146, -0.450, -1.354],
        ],
    )


@pytest.fixture(scope="session")
def prog_input_factory(hydrogen):
    """Return a factory that creates ProgramInput instances with specified calctypes."""

    def create_program_input(calctype, program="test"):
        return ProgramInput(
            program=program,
            structure=hydrogen,
            calctype=calctype,
            model={"method": "hf", "basis": "sto-3g"},
            keywords={"purify": "no", "some-bool": False},
        )

    return create_program_input


@pytest.fixture(scope="function")
def nested_input_factory(hydrogen):
    def create_nested_input(calctype):
        return ProgramInput(
            program="geometric",
            calctype=calctype,
            structure=hydrogen,
            subprograms=[
                ProgramSpec(
                    calctype="gradient",
                    model={"method": "hf", "basis": "sto-3g"},
                    program="test",
                )
            ],
        )

    return create_nested_input


@pytest.fixture
def results(prog_input_factory):
    """Create ProgramOutput object"""
    sp_inp_energy = prog_input_factory("energy", program="test")
    energy = 1.0
    n_atoms = len(sp_inp_energy.structure.symbols)
    gradient = np.arange(n_atoms * 3).reshape(n_atoms, 3)
    hessian = np.arange(n_atoms**2 * 3**2).reshape(n_atoms * 3, n_atoms * 3)

    return ProgramOutput[ProgramInput, SinglePointData](
        input_data=sp_inp_energy,
        success=True,
        logs="program standard out...",
        results={
            "provenance": {"program": "qcdata-test-suite"},
            "energy": energy,
            "gradient": gradient,
            "hessian": hessian,
        },
        execution={"scratch_dir": "/tmp/qcdata"},
        extras={"some_extra": 1},
    )


@pytest.fixture(scope="session")
def test_adapter():
    class TestAdapter(ProgramAdapter):
        # Both program and supported_driver defined
        program = "test"
        supported_calctypes = [CalcType.energy, CalcType.gradient]

        def compute_data(
            self, input_data, update_func=None, update_interval=None, **kwargs
        ):
            return SinglePointData(
                provenance={"program": "test"}, energy=0.0
            ), "Some stdout."

        def program_version(self, stdout: str | None = None) -> str:
            return "v1.0.0"

    return TestAdapter()


def skipif_program_not_available(program_name: str):
    """Skip a test if the given program is not available."""
    return pytest.mark.skipif(
        not prog_available(program_name), reason=f"{program_name} is not installed."
    )
