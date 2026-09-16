"""
Utility functions for adapters. Avoids circular imports when these functions placed in
top-level utils.py module.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from time import time

from qcdata import (
    Data,
    ExecutionInfo,
    FileData,
    Inputs,
    ProgramInput,
    Provenance,
    get_data_type,
)
from qcdata.helper_types import StrOrPath

from qccompute.exceptions import ExternalProgramError, ProgramNotFoundError


def execute_subprocess(
    program: str,
    cmdline_args: list[str] | None = None,
    update_func: Callable | None = None,
    update_interval: float | None = 0.5,
) -> str:
    """Execute a subprocess and monitor its stdout/stderr using a callback function.

    Args:
        program: The program to run.
        cmdline_args: The command line arguments to pass to the program.
        update_func: A function to call as the program executes to monitor stdout. The
            function must accept two string arguments, 1) The entire stdout/stderr
            output from the process, and 2) The output since the last call to
            update_func. The update_func is called at least once, and then every
            update_interval seconds. This is useful when executing code on remote
            servers and you want to pass a callback to monitor the output in real time.
            If you want to simply monitor stdout on your local machine in real time
            pass a simple update_func like this:
                execute_subprocess(
                    program,
                    update_func=lambda stdout, new_stdout: print(new_stdout
                    update_interval=0.5,
                )
        update_interval: The minimum time in seconds between calls to the update_func.
            Defaults to 0.5 seconds if update_func is not None. Default is set inside
            the function to avoid issues with mypy and default arguments.

    Returns:
        The stdout of the program as a string.

    Raises:
        ProgramNotFoundException: If the program is not found.
        ExternalProgramExecutionError: If the program fails during execution and
            returns a non-zero exit code.
    """
    cmd = [program] + (cmdline_args or [])
    # Execute subprocess asynchronously so we can monitor stdout and stderr.
    try:
        proc = subprocess.Popen(
            cmd,
            # redirects stdout to a pipe, ensures proc.stdout is not None
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr is redirected to stdout
            universal_newlines=True,  # returns stdout as string
        )
    except FileNotFoundError as e:
        raise ProgramNotFoundError(program=program) from e

    # Setup variables for monitoring stdout
    stdout_lines: list[str] = []
    prev_update_time = time()
    prev_line_count = len(stdout_lines)

    # Read stdout line by line as it is available until proc terminates
    while proc.poll() is None:
        # This blocks until it receives a newline.
        # ignore mypy because stdout=subprocess.PIPE, not None
        line = proc.stdout.readline()  # type: ignore
        if line:  # readline may return '' at EOF and we don't want to append that
            stdout_lines.append(line)

        if (
            update_func
            and time() - prev_update_time > (update_interval or 0.5)
            and len(stdout_lines) > prev_line_count
        ):
            # Call update_func with the current stdout
            stdout = "".join(stdout_lines)
            stdout_since_last_update = "".join(stdout_lines[prev_line_count:])
            update_func(stdout, stdout_since_last_update)
            prev_update_time = time()
            prev_line_count = len(stdout_lines)

    # Unconsumed output may still need to be processed
    # ignore mypy because stdout=subprocess.PIPE, not None
    for line in proc.stdout.readlines():  # type: ignore
        stdout_lines.append(line)
    stdout = "".join(stdout_lines)

    # Check if program executed successfully
    if proc.returncode != 0:
        raise ExternalProgramError(
            program=program,
            message=f"External program failed with return code {proc.returncode}. Command: '{cmd}'",
            logs=stdout,
        )

    return stdout


class DualOutputHandler(logging.Handler):
    """A logging handler that writes to both a string buffer and the console.

    To enable this handler to use arbitrary update_func like utils.execute_subprocess,
    I should factor out the update_func logic in execute_subprocess into a
    separate function and use it generically inside execute_subprocess and
    DualOutputHandler. For now printing to stdout is fine.
    """

    def __init__(
        self,
        buffer,
        update_func: Callable | None = None,
        update_interval: float | None = None,
    ):
        super().__init__()
        self.buffer = buffer
        self.update_func = update_func
        self.update_interval = update_interval

    def emit(self, record):
        # Write to the string buffer
        msg = self.format(record)
        self.buffer.write(msg + "\n")

        # Run update func
        if self.update_func:
            self.update_func(None, msg)


@contextmanager
def capture_logs(
    logger_name: str,
    update_func: Callable | None = None,
    update_interval: float | None = None,
):
    """Capture logs from a program during execution and print them to the console."""

    # Create a logger
    logger = logging.getLogger(logger_name)

    # Set the level to capture all logs
    logger.setLevel(logging.DEBUG)

    # Create a string buffer and add a DualOutputHandler using the buffer
    logs_string = StringIO()
    if update_func and update_interval:
        handler: DualOutputHandler | logging.StreamHandler = DualOutputHandler(
            logs_string, update_func, update_interval
        )
    else:
        handler = logging.StreamHandler(logs_string)

    # Optional: set a format for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    try:
        yield (logger, logs_string)
    finally:
        # Remove the handler when done
        logger.removeHandler(handler)


lock = threading.Lock()


@contextmanager
def capture_sys_stdout():
    """Capture stdout from a program that bypasses the sys.stdout object.

    Useful for capturing logs written by C/C++ libraries such as xtb.

    The lock is necessary since we are modifying a global resource, the stdout file
    descriptor, which is shared between threads. Without this lock race conditions
    cause random output to be printed to the console and may freeze the program.
    """
    with lock:
        # Create a pipe to capture output
        r, w = os.pipe()

        # Save the original stdout and stderr file descriptors
        try:
            stdout_fd = sys.stdout.fileno()
        except AttributeError:  # Handles case where celery LoggingProxy is sys.stdout
            stdout_fd = sys.__stdout__.fileno()
        old_stdout = os.dup(stdout_fd)

        # Redirect stdout and stderr to the write end of the pipe
        os.dup2(w, stdout_fd)
        try:
            yield r  # Allow code to be executed within the context manager block
        finally:
            # Restore stdout and stderr
            os.dup2(old_stdout, stdout_fd)

            # Close the duplicated fds
            os.close(old_stdout)
            os.close(w)

            # Close the read ends of the pipe
            os.close(r)


def construct_execution(scratch_dir: Path | None, wall_time: float) -> ExecutionInfo:
    """Record runtime information independently of scientific producer identity."""
    return ExecutionInfo(
        scratch_dir=scratch_dir,
        wall_time=round(wall_time, 6),
        hostname=platform.node(),
        host_cpu=os.cpu_count(),
    )


def empty_results(input_data: Inputs, program: str | None = None) -> Data:
    """Create correctly typed empty data when a calculation cannot produce values."""
    data_type: type[Data] = (
        get_data_type(input_data.calctype)
        if isinstance(input_data, ProgramInput)
        else FileData
    )
    return data_type(provenance=Provenance(program=program or input_data.program))


@contextmanager
def tmpdir(mkdir: bool = True, directory: StrOrPath | None = None, rmdir: bool = True):
    """Context manager for a temporary directory.

    Args:
        mkdir: Whether to create the temporary directory. If False, the current working
            directory is used.
        directory: Where to create the temporary directory. If None, a new directory is
            created in the system default temporary directory.
        rmdir: Whether to remove the temporary directory when the context manager exits.
    """
    if not mkdir:
        yield Path.cwd()
    else:
        cwd = Path.cwd()  # Save current working directory
        temp_dir = Path(directory or tempfile.mkdtemp())  # Set path to directory
        temp_dir.mkdir(parents=True, exist_ok=True)  # Create directory
        try:
            os.chdir(temp_dir)  # Change to temporary directory
            yield temp_dir  # Execute code in context manager
        finally:
            if rmdir:  # After exiting context manager
                shutil.rmtree(temp_dir)
            os.chdir(cwd)


@contextmanager
def set_env_variable(var_name, value):
    """Context manager to set an environment variable temporarily.

    Args:
        var_name: The name of the environment variable.
        value: The value to set the environment variable to.
    """
    original_value = os.environ.get(var_name)
    try:
        os.environ[var_name] = str(value)
        yield
    finally:
        if original_value:
            os.environ[var_name] = original_value
        else:
            del os.environ[var_name]
