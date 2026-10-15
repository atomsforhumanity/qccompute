"""Workspace and output mechanics, independent of scientific integrations."""

from __future__ import annotations

import os
import platform
import re
import shutil
import tempfile
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from qcdata import (
    Data,
    ExecutionInfo,
    FileData,
    Inputs,
    ProgramInput,
    Provenance,
    get_data_type,
)

from .exceptions import ProgramNotFoundError


def empty_results(input_data: Inputs) -> Data:
    """Create the result type required by an input before any values are available."""
    data_type = (
        get_data_type(input_data.calctype)
        if isinstance(input_data, ProgramInput)
        else FileData
    )
    return data_type(provenance=Provenance(program=input_data.program))


def construct_execution(
    directory: Path | None, wall_time: float, executable: str | None = None
) -> ExecutionInfo:
    """Record runtime observations independently of scientific producer identity.

    Args:
        directory: The calculation directory, if one was created.
        wall_time: Seconds from the start of the call.
        executable: Absolute path of the program that ran, stored in
            ``extras["executable"]``; None for Python calculations.
    """
    return ExecutionInfo(
        scratch_dir=directory,
        wall_time=round(wall_time, 6),
        hostname=platform.node(),
        host_cpu=os.cpu_count(),
        extras={"executable": executable} if executable else {},
    )


def executable_variable(program: str) -> str:
    """Name the environment variable that overrides a program's executable.

    ``orca`` becomes ``QCCOMPUTE_ORCA_EXECUTABLE``; characters other than letters
    and digits become underscores.
    """
    return f"QCCOMPUTE_{re.sub(r'[^A-Z0-9]', '_', program.upper())}_EXECUTABLE"


def resolve_executable(
    program: str,
    command: str,
    environment: Mapping[str, str],
    executables: Mapping[str, str] | None = None,
) -> str:
    """Return the absolute path of the executable to run for ``program``.

    The first of these wins: ``executables[program]``, the environment variable
    named by ``executable_variable(program)``, then ``command`` (the codec's first
    argument). A value without a path separator is looked up on the ``PATH`` in
    ``environment``; a relative path is taken relative to the current directory,
    not the calculation directory.

    Args:
        program: The program name from the input, which keys the overrides.
        command: The executable the codec asked for.
        environment: The environment the program will run with.
        executables: Explicit overrides mapping program names to executables.

    Raises:
        ProgramNotFoundError: The chosen executable does not exist or is not
            executable.
    """
    variable = executable_variable(program)
    if executables and program in executables:
        value, source = executables[program], f"executables[{program!r}]"
    elif environment.get(variable):
        value, source = environment[variable], variable
    else:
        value, source = command, None
    found = shutil.which(os.path.expanduser(value), path=environment.get("PATH"))
    if found is not None:
        return os.path.abspath(found)
    if source is not None:
        message = f"{source} is {value!r}, which is not an executable file."
    else:
        message = (
            f"Program not found: {value!r} is not on PATH. Install it, or set "
            f"{variable} or compute(..., executables={{{program!r}: path}}) to its "
            "full path."
        )
    raise ProgramNotFoundError(program, message)


@contextmanager
def workspace(
    directory: str | Path | None = None, remove: bool = True
) -> Iterator[Path]:
    """Keep calculation files alive through decoding and optionally remove them afterward."""
    path = (
        Path(directory).resolve()
        if directory is not None
        else Path(tempfile.mkdtemp()).resolve()
    )
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        if remove:
            shutil.rmtree(path)


def safe_path(directory: Path, name: str) -> Path:
    """Reject file names that would escape the calculation workspace."""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"File name must stay within the calculation directory: {name!r}"
        )
    path = directory / relative
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"File path escapes the calculation directory: {name!r}")
    return path


def stage_files(directory: Path, files: Mapping[str, str | bytes]) -> None:
    """Write native inputs while keeping relative paths within the workspace."""
    for name, value in files.items():
        path = safe_path(directory, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(value)


def collect_files(
    directory: Path,
    *,
    exclude: Collection[str] = (),
    patterns: tuple[str, ...] | None = None,
    binary: bool = False,
) -> dict[str, str | bytes]:
    """Read selected regular artifacts before workspace cleanup."""
    files: dict[str, str | bytes] = {}
    paths = (
        directory.rglob("*")
        if patterns is None
        else (path for pattern in patterns for path in directory.glob(pattern))
    )
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        name = path.relative_to(directory).as_posix()
        if name in exclude or not path.resolve().is_relative_to(directory.resolve()):
            continue
        contents = path.read_bytes()
        if binary:
            files[name] = contents
            continue
        try:
            files[name] = contents.decode("utf-8")
        except UnicodeDecodeError:
            files[name] = contents
    return files
