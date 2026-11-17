"""Run AFH programs: one input or many, in this process or in workers, with a store.

``qccompute.compute(inp)`` runs a calculation on this machine, ``qccompute.submit(inp)``
starts one and returns a future of its output, and ``qccompute.session(inp)`` opens a
session; all are the methods of the default `Runner`
(``qccompute.default_runner()``). Programs implement the
protocol in `qccompute.program`; external programs come with ``qccompute[qccodec]``.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    # Source tree / build hook / CI checkout
    __version__ = "0.0.0+local"

from .program import RunContext, register_program
from .runner import Runner, compute, default_runner, session, submit

__all__ = [
    "compute",
    "submit",
    "session",
    "Runner",
    "default_runner",
    "RunContext",
    "register_program",
    "__version__",
]
