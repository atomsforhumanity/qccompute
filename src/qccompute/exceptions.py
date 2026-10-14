"""Exceptions that carry the completed ProgramOutput of a failed calculation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qcdata import Data, ProgramOutput


def _restore_error(
    cls: type[Exception], args: tuple[Any, ...], attributes: dict[str, Any]
) -> Exception:
    """Restore without calling program-specific exception constructors."""
    error = cls.__new__(cls)
    Exception.__init__(error, *args)
    error.__dict__.update(attributes)
    return error


class QCComputeBaseError(Exception):
    """Base class for all qccompute exceptions.

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
    ) -> None:
        """Retain the original failure and available calculation diagnostics."""
        super().__init__(message, prog_output, data)
        self.prog_output: ProgramOutput | None = prog_output
        self.data: Data | None = data

    def __str__(self) -> str:
        """Display the original failure message without serialized output data."""
        return self.args[0]

    def __reduce__(self) -> tuple[Callable[..., Exception], tuple[object, ...]]:
        """Preserve error attributes and constructor-independent pickling."""
        return _restore_error, (type(self), self.args, self.__dict__)

    def __repr__(self) -> str:
        """Describe the failure together with its attached diagnostic state."""
        return (
            f"{self.__class__.__name__}(message={self.args[0]!r}, "
            f"prog_output={self.prog_output!r}, data={self.data!r})"
        )


# ===================== Codec and Input Errors =====================


class CodecError(QCComputeBaseError):
    """Misconfiguration or invalid inputs detected before the program runs."""


class CodecNotFoundError(CodecError):
    """Raised when no codec can be found for the requested program."""

    def __init__(
        self,
        program: str,
        message: str | None = None,
        prog_output: ProgramOutput | None = None,
    ) -> None:
        """Retain the original failure and available calculation diagnostics."""
        if message is None:
            message = f"No codec found for program '{program}'."
        super().__init__(message, prog_output, None)
        self.program: str = program

    def __repr__(self) -> str:
        """Describe the failure together with its attached diagnostic state."""
        return f"{super().__repr__()}, program={self.program!r}"


class InputError(CodecError):
    """Raised when the inputs or compute options are invalid for the codec."""

    def __init__(
        self,
        program: str,
        message: str | None = None,
        prog_output: ProgramOutput | None = None,
    ) -> None:
        """Retain the original failure and available calculation diagnostics."""
        if message is None:
            message = f"Invalid inputs for program '{program}'."
        super().__init__(message, prog_output, None)
        self.program: str = program

    def __repr__(self) -> str:
        """Describe the failure together with its attached diagnostic state."""
        return f"{super().__repr__()}, program={self.program!r}"


# Deprecated names from the Adapter era, kept for one release so existing
# ``except`` clauses keep working. Raised exceptions use the new classes above.
AdapterError = CodecError
AdapterNotFoundError = CodecNotFoundError
AdapterInputError = InputError


# ===================== External Program Errors =====================


class ExternalProgramError(QCComputeBaseError):
    """Raised when an external program or package fails to complete successfully.

    This covers failures from external subprocesses, Python packages (like qccodec),
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
    ) -> None:
        """Retain the original failure and available calculation diagnostics."""
        if message is None:
            message = f"External program '{program}' failed."
        super().__init__(message, prog_output, data)
        self.program: str = program
        self.original_exception: Exception | None = original_exception
        self.logs: str | None = logs

    def __repr__(self) -> str:
        """Describe the failure together with its attached diagnostic state."""
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
    ) -> None:
        """Retain the original failure and available calculation diagnostics."""
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
        self.program: str = program
        self.install_msg: str | None = install_msg

    def __repr__(self) -> str:
        """Describe the failure together with its attached diagnostic state."""
        return f"{super().__repr__()}, install_msg={self.install_msg!r}"


class ProgramTimeoutError(ExternalProgramError):
    """Raised when an external program exceeds the ``timeout`` given to ``compute``.

    The program and any processes it started are killed. Output produced before the
    deadline is retained and decoded, so partial results remain available.
    """
