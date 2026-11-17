"""Run AFH programs: one input or many, in this process or in workers, with a store.

``qccompute.compute(inp)`` runs a calculation on this machine and ``qccompute.session(inp)``
opens a session; both are the methods of the default `Runner`. Programs implement the
protocol in `qccompute.program`; external programs come with ``qccompute[qccodec]``.
"""

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    # Source tree / build hook / CI checkout
    __version__ = "0.0.0+local"

from .program import RunContext, register_program
from .runner import ComputeErrors, Runner, compute, session

__all__ = [
    "compute",
    "session",
    "Runner",
    "ComputeErrors",
    "RunContext",
    "register_program",
    "__version__",
]
