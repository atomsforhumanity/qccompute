import importlib
from collections.abc import Callable
from importlib.metadata import version

from qcdata import CalcType, ProgramInput, SinglePointData

from qccompute.exceptions import (
    AdapterError,
    ExternalProgramError,
    ProgramNotFoundError,
)

from .base import ProgramAdapter


class TeraChemFEAdapter(ProgramAdapter[ProgramInput, SinglePointData]):
    """Adapter for TeraChem's Protocol Buffer Server and Frontend file server."""

    supported_calctypes = [CalcType.energy, CalcType.gradient]
    """Supported calculation types."""
    program = "terachem-fe"

    def __init__(self):
        super().__init__()
        # Check that a compatible tcpb is installed.
        self.tcpb = self._ensure_tcpb()
        self.client = self.tcpb.TCFrontEndClient

    @staticmethod
    def _ensure_tcpb():
        try:
            tcpb = importlib.import_module("tcpb")
            if version("tcpb") == "0.16.0":
                raise AdapterError(
                    "tcpb 0.16.0 constructs legacy qcdata outputs. TeraChem FE/PBS "
                    "requires a tcpb migration to ProgramOutput.results, data provenance, "
                    "and ExecutionInfo before it can use this qcdata release."
                )
            return tcpb
        except ModuleNotFoundError:
            raise ProgramNotFoundError(
                "tcpb",
                install_msg=(
                    "Program not found: 'tcpb'. To use tcpb please install it with "
                    "pip install qccompute[tcpb] or add '' if your shell requires it. "
                    "e.g., pip install 'qccompute[tcpb]'."
                ),
            )

    @property
    def producer_program(self) -> str:
        return "terachem"

    def program_version(self, stdout: str | None = None) -> str | None:
        """Program version is not available via the PB server."""
        return None

    def compute_data(
        self,
        input_data: ProgramInput,
        update_func: Callable | None = None,
        update_interval: float | None = None,
        **kwargs,
    ) -> tuple[SinglePointData, str]:
        """Execute TeraChem on the given input.

        Args:
            input_data: The qcdata ProgramInput object for a computation.
            update_func: A callback function to call as the program executes.
            update_interval: The minimum time in seconds between calls to the
                update_func.

        Returns:
            A tuple of SinglePointData and the stdout str.
        """
        try:
            with self.client() as client:
                prog_output = client.compute(input_data)
        except self.tcpb.exceptions.TCPBError as e:
            exc = ExternalProgramError(
                program=self.program,
                # Pass logs to .compute() via the exception
                # Will only exist for TeraChemFrontendAdapter
                logs=e.program_output.logs if e.program_output is not None else None,
                data=e.program_output.results if e.program_output is not None else None,
            )

            raise exc

        else:
            # Write files to disk to be collected by BaseAdapter.compute()
            # Used only for TeraChemFrontendAdapter
            prog_output.results.save_files()

        return prog_output.results, prog_output.logs
