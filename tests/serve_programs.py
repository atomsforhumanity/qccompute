"""Programs for the worker's tests, loaded in a worker process as module:attribute."""

import os

from qcdata import ErrorInfo, ErrorKind, ProgramOutput

from qccompute.testing.toy import ToyProgram


class DeviceLost(ToyProgram):
    """Every one-off compute reports a poisoned GPU context."""

    name = "device-lost"

    def compute(self, inp, ctx):
        error = ErrorInfo(
            kind=ErrorKind.backend, code="device_lost", message="context poisoned"
        )
        return ProgramOutput.failure(inp, error)


class Noisy(ToyProgram):
    """Prints to stdout from Python and from the file descriptor, as libraries do."""

    name = "noisy"

    def compute(self, inp, ctx):
        print("python print on stdout")
        os.write(1, b"fd 1 write\n")
        return super().compute(inp.model_copy(update={"program": "toy"}), ctx)


device_lost = DeviceLost()
noisy = Noisy()
