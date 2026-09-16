"""Experimental exception hierarchy. This may be too complex and unhelpful for now"""

from qcdata import Data, ProgramOutput


class QCComputeBaseError(Exception):
    """
    Base class for all qccompute exceptions.

    All QCCompute exceptions must eventually have a non-None prog_output attribute.
    Lower-level code may leave prog_output as None; the top-level compute() method
    should attach the final ProgramOutput before propagating the error. If some results
    were computed before the error occurred, they should be attached to the exception
    as well.
    """

    def __init__(
        self,
        message: str,
        prog_output: ProgramOutput | None = None,
        data: Data | None = None,
    ):
        # Pass everything as positional arguments so they are captured in .args
        # Required for pickling and other serialization methods including celery.
        super().__init__(message, prog_output, data)
        self.prog_output = prog_output
        self.data = data

    def __str__(self):
        # Only the message is shown in the string representation.
        return self.args[0]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(message={self.args[0]!r}, "
            f"prog_output={self.prog_output!r}, data={self.data!r})"
        )


# ===================== Adapter-Related Errors =====================


class AdapterError(QCComputeBaseError):
    """Exceptions due to misconfiguration or invalid inputs for adapters."""

    pass


class AdapterNotFoundError(AdapterError):
    """Raised when no adapter can be found for the requested program."""

    def __init__(
        self,
        program: str,
        message: str | None = None,
        prog_output: ProgramOutput | None = None,
    ):
        if message is None:
            message = f"No adapter found for program '{program}'."
        super().__init__(message, prog_output, None)
        self.program = program

    def __repr__(self):
        return f"{super().__repr__()}, program={self.program!r}"


class AdapterInputError(AdapterError):
    """Raised when the inputs provided to an adapter are invalid."""

    def __init__(
        self,
        program: str,
        message: str | None = None,
        prog_output: ProgramOutput | None = None,
    ):
        if message is None:
            message = f"Invalid inputs for program '{program}'."
        super().__init__(message, prog_output, None)
        self.program = program

    def __repr__(self):
        return f"{super().__repr__()}, program={self.program!r}"


# ===================== External Program Errors =====================


class ExternalProgramError(QCComputeBaseError):
    """
    Raised when an external program or package fails to complete successfully.

    This covers failures from external subprocesses, Python packages (like qccodec or QCEngine),
    or any other external libraries.
    """

    def __init__(
        self,
        program: str,
        message: str | None = None,
        prog_output: ProgramOutput | None = None,
        data: Data | None = None,
        original_exception: Exception | None = None,
        logs: str | None = None,
    ):
        if message is None:
            message = f"External program '{program}' failed."
        super().__init__(message, prog_output, data)
        self.program = program
        self.original_exception = original_exception
        self.logs = logs

    def __repr__(self):
        return (
            f"{super().__repr__()}, program={self.program!r}, "
            f"original_exception={self.original_exception!r}, logs={self.logs!r}"
        )


class ProgramNotFoundError(ExternalProgramError):
    """Raised when the external program is not found on the host system."""

    def __init__(
        self,
        program: str,
        message: str | None = None,
        prog_output: ProgramOutput | None = None,
        install_msg: str | None = None,
    ):
        if message is None:
            message = (
                install_msg
                or f"Program not found: '{program}'. Please install it and ensure it is on your PATH."
            )
        # Call ExternalProgramError with prog_output, original_exception, and logs defaulting to None.
        super().__init__(
            program,
            message,
            prog_output,
            None,
            None,
            None,
        )
        self.program = program
        self.install_msg = install_msg

    def __repr__(self):
        return f"{super().__repr__()}, install_msg={self.install_msg!r}"
