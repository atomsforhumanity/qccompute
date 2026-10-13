"""Discover executable programs available to the current process."""

import shutil

from qccodec import supported_programs


def prog_available(program: str) -> bool:
    """Check if a program is available on the system.

    Args:
        program: The program to check.

    Returns:
        True if the program is available, False otherwise.
    """
    return shutil.which(program) is not None


def available_programs() -> list[str]:
    """Built-in programs with an executable on PATH (not a Python API probe)."""
    return [program for program in supported_programs() if prog_available(program)]
