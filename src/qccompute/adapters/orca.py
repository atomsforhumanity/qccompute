import shutil
from collections.abc import Callable
from pathlib import Path

import qccodec
from qccodec import exceptions as qccodec_exceptions
from qccodec.parsers.orca import parse_version
from qcdata import CalcType, OptimizationData, ProgramInput, SinglePointData

from qccompute.exceptions import (
    AdapterInputError,
    ExternalProgramError,
    ProgramNotFoundError,
)

from .base import ProgramAdapter
from .utils import execute_subprocess


class OrcaAdapter(ProgramAdapter[ProgramInput, SinglePointData | OptimizationData]):
    """Adapter for Orca."""

    supported_calctypes = [
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
        CalcType.transition_state,
    ]
    program = "orca"

    def program_version(self, stdout: str | None = None) -> str | None:
        """Get the program version.

        Args:
            stdout: The stdout from the program.

        Returns:
            The program version.
        """
        if not stdout:
            return None
        try:
            return parse_version(stdout)
        except qccodec_exceptions.ParserError:
            return None

    def compute_data(
        self,
        input_data: ProgramInput,
        update_func: Callable | None = None,
        update_interval: float | None = None,
        **kwargs,
    ) -> tuple[SinglePointData | OptimizationData, str]:
        """Execute Orca on the given input.

        Args:
            input_data: The qcdata ProgramInput object for a computation.
            update_func: A callback function to call as the program executes.
            update_interval: The minimum time in seconds between calls to the
                update_func.

        Returns:
            A tuple of SinglePointData and the stdout str.
        """
        # Construct TeraChem native input files
        try:
            native_input = qccodec.encode(input_data)
        except qccodec.exceptions.EncoderError as e:
            raise AdapterInputError(program=self.program) from e

        # Write the input files to disk
        input_filename = "orca.inp"
        Path(input_filename).write_text(native_input.input_file)
        Path(native_input.geometry_filename).write_text(native_input.geometry_file)

        # Execute Orca
        # (A quirk of Orca: One must use the full executable path when running
        # in parallel. See:
        # https://www.faccts.de/docs/orca/6.1/tutorials/first_steps/parallel.html)
        full_orca_path = shutil.which(self.program)
        if full_orca_path is None:
            raise ProgramNotFoundError(program=self.program)
        execution_error: ExternalProgramError | None = None
        try:
            stdout = execute_subprocess(
                full_orca_path,
                [input_filename, *input_data.cmdline_args],
                update_func,
                update_interval,
            )
        except ProgramNotFoundError:
            raise  # Nothing ran, so there are no program artifacts to decode.
        except ExternalProgramError as exc:
            execution_error = exc
            stdout = exc.logs or ""

        # Parse output
        try:
            results = qccodec.decode(
                self.program,
                input_data.calctype,
                stdout=stdout,
                directory=Path.cwd(),
                input_data=input_data,
                failed=execution_error is not None,
            )
        except qccodec_exceptions.ParserError as e:
            if execution_error is not None:
                execution_error.data = e.data
                raise execution_error from e
            raise ExternalProgramError(
                program="qccodec",
                message="Failed to parse Orca output.",
                data=e.data,
                logs=stdout,
                original_exception=e,
            ) from e
        if execution_error is not None:
            execution_error.data = results
            raise execution_error
        return results, stdout
