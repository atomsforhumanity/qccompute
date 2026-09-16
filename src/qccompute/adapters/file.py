from collections.abc import Callable

from qcdata import FileData, FileInput, Provenance

from qccompute.adapters.base import BaseAdapter

from .utils import execute_subprocess


class FileAdapter(BaseAdapter[FileInput, FileData]):
    """adapter for running a program on files."""

    def __init__(self, program: str) -> None:
        super().__init__()
        self.program = program

    def validate_input(self, input_data: FileInput) -> None:
        """No validation checks performed for FileAdapter"""
        pass

    def compute_data(
        self,
        input_data: FileInput,
        update_func: Callable | None = None,
        update_interval: float | None = None,
        **kwargs,
    ) -> tuple[FileData, str]:
        """Compute the given program on the given files.

        Args:
            input_data: The qcdata FileInput object for a computation.
            update_func: A callback function to call as the program executes.
            update_interval: The minimum time in seconds between calls to the
            update_func.

        Returns:
            Tuple of a `FileData` object and the program output string for a
            computation. The returned `FileData` instance is initially empty and
            will be populated with file data by the :meth:`.compute` method.

        Raises:
            ProgramNotFoundException: If the program is not found.

        """
        stdout = execute_subprocess(
            self.program, input_data.cmdline_args, update_func, update_interval
        )
        # FileData will be added to this object by the .compute() method
        return FileData(provenance=Provenance(program=self.program)), stdout
