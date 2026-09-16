"""Adapter for CREST package. https://crest-lab.github.io/crest-docs/"""

from collections.abc import Callable
from pathlib import Path

import qccodec
from qccodec.parsers.crest import parse_version
from qcdata import (
    CalcType,
    ConformerSearchData,
    OptimizationData,
    ProgramInput,
    SinglePointData,
)

from qccompute.exceptions import (
    AdapterInputError,
    ExternalProgramError,
    ProgramNotFoundError,
)

from .base import ProgramAdapter
from .utils import execute_subprocess


class CRESTAdapter(
    ProgramAdapter[
        ProgramInput,
        SinglePointData | OptimizationData | ConformerSearchData,
    ]
):
    """Adapter for CREST.

    Note:
        The `ProgramInput.keywords` attribute is used to create the input file for
        CREST. This means that the structure of the `keywords` attribute should match
        that of [CREST's input specification](https://crest-lab.github.io/crest-docs/page/documentation/inputfiles.html).
        Keywords such as method, charge, and uhf (which are stored on the `Model` and
        `Structure`; uhf is `multiplicity - 1`) will be added to the input file
        automatically.

    Warning:
        CREST does not exit with a non-zero exit code on failure. Instead, it prints
        "FAILED" in the stdout. This adapter will raise an ExternalProgramError if
        "FAILED" is found in the stdout.

    Warning:
        CREST automatically translates the input geometry to the origin. This means
        that the input geometry printed to CREST's stdout will not match the input
        structure; however, all computed values (such as energies, gradients, etc.) are
        still valid because they are translationally invariant.
    """

    supported_calctypes = [
        CalcType.energy,
        CalcType.gradient,
        CalcType.hessian,
        CalcType.optimization,
        CalcType.conformer_search,
    ]
    """Supported calculation types."""
    program = "crest"

    def program_version(self, stdout: str | None = None) -> str | None:
        """Get the program version.

        Args:
            stdout: The stdout from the program.

        Returns:
            The program version.
        """
        if not stdout:
            stdout = execute_subprocess(self.program, ["--version"])
        try:
            return parse_version(stdout)
        except qccodec.exceptions.ParserError:
            return None

    def compute_data(
        self,
        input_data: ProgramInput,
        update_func: Callable | None = None,
        update_interval: float | None = None,
        collect_rotamers: bool = False,
        **kwargs,
    ) -> tuple[SinglePointData | OptimizationData | ConformerSearchData, str]:
        """Execute CREST on the given input.

        Args:
            input_data: The qcdata ProgramInput object for a computation.
            update_func: A function to call with the stdout at regular intervals.
            update_interval: The interval at which to call the update function.
            collect_rotamers: Collect rotamers if doing a conformer_search. Defaults to
                False since rotamers are usually not of interest and there will be many.

        Returns:
            A tuple of ConformerSearchData and the stdout str.
        """
        # Create CREST native input files
        try:
            native_inp = qccodec.encode(input_data)
        except qccodec.exceptions.EncoderError as e:
            raise AdapterInputError(program=self.program) from e

        # Write the input files to disk
        inp_file, struct_file = Path("input.toml"), Path(native_inp.geometry_filename)
        inp_file.write_text(native_inp.input_file)
        struct_file.write_text(native_inp.geometry_file)

        # Execute CREST
        execution_error: ExternalProgramError | None = None
        try:
            stdout = execute_subprocess(
                self.program,
                [inp_file.name, *input_data.cmdline_args],
                update_func,
                update_interval,
            )
        except ProgramNotFoundError:
            raise  # Nothing ran, so there are no program artifacts to decode.
        except ExternalProgramError as exc:
            execution_error = exc
            stdout = exc.logs or ""

        # CREST does not exit with a non-zero exit code on failure
        if "FAILED" in stdout and execution_error is None:
            execution_error = ExternalProgramError(
                program=self.program,
                message="CREST calculation failed. See the stdout for more information.",
                logs=stdout,
            )

        # Parse the output
        try:
            results = qccodec.decode(
                self.program,
                input_data.calctype,
                stdout=stdout,
                directory=".",
                input_data=input_data,
                failed=execution_error is not None,
            )
        except qccodec.exceptions.ParserError as e:
            if execution_error is not None:
                execution_error.data = e.data
                raise execution_error from e
            raise ExternalProgramError(
                program="qccodec",
                message="Failed to parse CREST output.",
                data=getattr(e, "data", None),
                logs=stdout,
                original_exception=e,
            ) from e

        if execution_error is not None:
            execution_error.data = results
            raise execution_error
        return results, stdout
