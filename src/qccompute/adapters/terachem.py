from collections.abc import Callable
from pathlib import Path

import qccodec
from qccodec import exceptions as qccodec_exceptions
from qccodec.parsers.terachem import parse_version
from qcdata import (
    CalcType,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    SinglePointData,
)

from qccompute.exceptions import (
    AdapterError,
    AdapterInputError,
    ExternalProgramError,
    ProgramNotFoundError,
)

from .base import ProgramAdapter
from .utils import execute_subprocess


class TeraChemAdapter(ProgramAdapter[ProgramInput, SinglePointData | OptimizationData]):
    """Adapter for TeraChem."""

    supported_calctypes = [
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
    ]
    """Supported calculation types."""
    program = "terachem"

    def program_version(self, stdout: str | None = None) -> str | None:
        """Get the program version.

        Args:
            stdout: The stdout from the program.

        Returns:
            The program version.
        """
        if stdout:
            try:
                return parse_version(stdout)
            except qccodec_exceptions.ParserError:
                # If the version string is not found. Happens when libcuda.so is not
                # found and TeraChem fails to start. terachem --version will fail too.
                return None
        else:
            try:
                return execute_subprocess(self.program, ["--version"])[17:]
            except ExternalProgramError:
                return None

    def compute_data(
        self,
        input_data: ProgramInput,
        update_func: Callable | None = None,
        update_interval: float | None = None,
        **kwargs,
    ) -> tuple[SinglePointData | OptimizationData, str]:
        """Execute TeraChem on the given input.

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
        input_filename = "tc.in"
        Path(input_filename).write_text(native_input.input_file)
        Path(native_input.geometry_filename).write_text(native_input.geometry_file)

        # Execute TeraChem
        execution_error: ExternalProgramError | None = None
        try:
            stdout = execute_subprocess(
                self.program,
                [input_filename, *input_data.cmdline_args],
                update_func,
                update_interval,
            )
        except ProgramNotFoundError:
            raise  # Nothing ran, so there are no program artifacts to decode.
        except ExternalProgramError as exc:
            execution_error = exc
            stdout = exc.logs or ""

        # Get the scratch output directory
        parent = Path.cwd()
        # TeraChem creates a directory named scr<xyz_filename> in the current working
        scr_dir = next(parent.glob("scr*"), None)
        if scr_dir is None and execution_error is None:
            execution_error = ExternalProgramError(
                self.program,
                f"TeraChem did not create a 'scr' directory in {parent}.",
                logs=stdout,
            )

        # Parse output
        try:
            results = qccodec.decode(
                self.program,
                input_data.calctype,
                stdout=stdout,
                directory=scr_dir,
                input_data=input_data,
                failed=execution_error is not None,
            )
        except qccodec_exceptions.ParserError as e:
            if execution_error is not None:
                execution_error.data = e.data
                raise execution_error from e
            raise ExternalProgramError(
                program="qccodec",
                message="Failed to parse TeraChem output.",
                data=e.data,
                logs=stdout,
                original_exception=e,
            ) from e
        if execution_error is not None:
            execution_error.data = results
            raise execution_error
        return results, stdout

    def collect_wfn(self) -> dict[str, str | bytes]:
        """Append wavefunction data to the output."""

        # Naming conventions from TeraChem uses xyz filename as scratch dir postfix
        scr_dir = next(Path().glob("scr*"), None)
        if not scr_dir:
            raise AdapterError(
                "No scratch directory found for wavefunction collection."
            )
        # Wavefunction filenames
        wfn_filenames = ("c0", "ca0", "cb0")
        wfn_paths = [scr_dir / fn for fn in wfn_filenames]
        if not any(wfn_path.exists() for wfn_path in wfn_paths):
            raise AdapterError(f"No wavefunction files found in {Path.cwd()}")

        wfns: dict[str, str | bytes] = {}
        for wfn_path in wfn_paths:
            if wfn_path.exists():
                wfns[str(wfn_path)] = wfn_path.read_bytes()
        return wfns

    def propagate_wfn(
        self,
        output: ProgramOutput[ProgramInput, SinglePointData],
        program_input: ProgramInput,
    ) -> ProgramInput:
        """Propagate the wavefunction from the previous calculation.

        Args:
            output: The output from a previous calculation containing wavefunction data.
            program_input: The ProgramInput object on which to place the wavefunction data.

        Returns:
            A new input containing the propagated wavefunction and guess keyword.
        """

        values = program_input.model_dump()

        # Wavefunction filenames
        suffixes = ("c0", "ca0", "cb0")
        matches = {}
        for k, v in output.results.files.items():
            for suffix in suffixes:
                if k.endswith(suffix):
                    matches[suffix] = v

        if not "c0" in matches and not ("ca0" in matches and "cb0" in matches):
            raise AdapterInputError(
                program=self.program,
                message="Could not find c0 or ca/b0 files in output.",
            )

        # Load wavefunction data onto ProgramInput object

        if "c0" in matches:
            values["files"]["c0"] = matches["c0"]
            values["keywords"]["guess"] = "c0"

        else:  # ca0_bytes and cb0_bytes
            assert "ca0" in matches and "cb0" in matches  # for mypy
            values["files"]["ca0"] = matches["ca0"]
            values["files"]["cb0"] = matches["cb0"]
            values["keywords"]["guess"] = "ca0 cb0"

        return ProgramInput.model_validate(values)
