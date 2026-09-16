"""Adaptor for xtb-python package.

Great overview of how to implement this adaptor:
https://github.com/grimme-lab/xtb-python/blob/main/xtb/qcschema/harness.py
"""

import importlib
import os
from collections.abc import Callable

import numpy as np
from qcdata import CalcType, ProgramInput, Provenance, SinglePointData, Wavefunction

from qccompute.exceptions import (
    AdapterInputError,
    ExternalProgramError,
    ProgramNotFoundError,
)

from .base import ProgramAdapter
from .utils import capture_sys_stdout, set_env_variable

# NOTE: Calls to importlib.metadata.version("xtb") are slow due to underlying calls to
# os.listdir(), so we cache the version number here. This is necessary because the
# __version__ string in xtb.__init__.py is wrong so we have to look this up dynamically
# in the program_version method.
CACHED_XTB_VERSION = None


class XTBAdapter(ProgramAdapter[ProgramInput, SinglePointData]):
    """Adapter for xtb-python."""

    supported_calctypes = [CalcType.energy, CalcType.gradient]
    """Supported calculation types."""
    program = "xtb"
    uses_files = False  # xtb-python does not use files written to disk

    def __init__(self):
        super().__init__()
        # Check that xtb-python is installed.
        self.xtb = self._ensure_xtb()

    def validate_input(self, input_data: ProgramInput) -> None:
        """Validate the input for xtb-python."""
        super().validate_input(input_data)
        assert input_data.model is not None
        # Check that xtb supports the method.
        supported_methods = self.xtb.interface.Param.__members__.keys()
        if input_data.model.method not in supported_methods:
            raise AdapterInputError(
                program=self.program,
                message=f"Unsupported method '{input_data.model.method}'. "
                f"Supported methods include: {supported_methods}",
            )

    def program_version(self, stdout: str | None = None) -> str:
        """Get the program version.

        Args:
            stdout: The stdout from the program.

        Returns:
            The program version.
        """
        global CACHED_XTB_VERSION
        if not CACHED_XTB_VERSION:
            CACHED_XTB_VERSION = importlib.metadata.version(self.program)
        return CACHED_XTB_VERSION

    @staticmethod
    def _ensure_xtb():
        # NOTE: xtb-python is 10x slower with the default number of OpenMP threads,
        # which I think is the number of CPUs on the machine. The xtb library reads in
        # this env variable upon import so I have to set it before importing xtb. If you
        # import any other xtb module in your code, e.g.,
        # `from xtb.utils import Solvent` you have to set this env variable before
        # importing that module or xtb will use the default number of threads which will
        # be a 10x slower calculation despite the context manager used here. Strangely,
        # setting this to one still results in spawning of a thread per logical core on
        # your machine. And if you set this to the number of cores on your machine (in
        # my case 16), xtb spawns 47!!! threads. So they are spawning threads within
        # threads or some other insanity.
        with set_env_variable("OMP_NUM_THREADS", "1"):
            try:
                xtb = importlib.import_module("xtb")
                # xtb import structure is screwed up so I have to do this too
                importlib.import_module("xtb.interface")
                importlib.import_module("xtb.libxtb")

                return xtb
            except ModuleNotFoundError:
                raise ProgramNotFoundError(
                    "xtb",
                    install_msg=(
                        "Program not found: 'xtb'. To use xtb please install it with "
                        "pip install qccompute[xtb] or add '' if your shell requires it. "
                        "e.g., pip install 'qccompute[xtb]'."
                    ),
                )

    def compute_data(
        self,
        input_data: ProgramInput,
        update_func: Callable | None = None,
        update_interval: float | None = None,
        **kwargs,
    ) -> tuple[SinglePointData, str]:
        """Execute xtb on the given input.

        Args:
            input_data: The qcdata ProgramInput object for a computation.
            update_func: A callback function to call as the program executes.
            update_interval: The minimum time in seconds between calls to the
                update_func.

        Returns:
            A tuple of SinglePointComputedProps and the stdout str.
        """
        if input_data.model is None:
            raise AdapterInputError(self.program, "xtb requires a scientific model.")
        try:
            # Create Calculator
            calc = self.xtb.interface.Calculator(
                getattr(self.xtb.interface.Param, input_data.model.method),
                np.array(input_data.structure.atomic_numbers),
                input_data.structure.geometry,
                input_data.structure.charge,
                # From https://github.com/grimme-lab/xtb-python/blob/a32309a43e5a6572b033814eacf396328a2a36ed/xtb/qcschema/harness.py#L126 # noqa: E501
                input_data.structure.multiplicity - 1,
            )
            calc.set_verbosity(self.xtb.libxtb.VERBOSITY_FULL)  # all logs

            # Set Keywords
            for key, value in input_data.keywords.items():
                # TODO: Need to handle external_charges and solvent
                getattr(calc, f"set_{key}")(value)

            # Capture logs
            with capture_sys_stdout() as r_pipe:
                res = calc.singlepoint()
                # Not sure what this does but it's in the xtb-python docs
                calc.release_output()
                stdout = os.read(r_pipe, 100000).decode()

        except self.xtb.interface.XTBException as e:
            raise ExternalProgramError(program="xtb-python") from e

        # Collect results
        # TODO: Collect other results xtb produces
        results = SinglePointData(
            provenance=Provenance(
                program="xtb", program_version=self.program_version()
            ),
            energy=res.get_energy(),
            gradient=res.get_gradient(),
            scf_dipole_moment=res.get_dipole(),
            wavefunction=Wavefunction(
                scf_eigenvalues_a=res.get_orbital_eigenvalues(),
            ),
        )

        return results, stdout
