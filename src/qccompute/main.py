"""Top level compute functions for qccompute."""

import traceback
from collections.abc import Callable
from time import time
from typing import Any

from qcdata import (
    CalcType,
    Data,
    Files,
    InputType,
    Model,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Structure,
)
from qcdata.helper_types import StrOrPath

from .adapters import BaseAdapter
from .adapters.utils import construct_execution, empty_results
from .exceptions import AdapterError, ProgramNotFoundError
from .utils import get_adapter, inherit_docstring_from


@inherit_docstring_from(BaseAdapter.compute)
def compute(
    input_data: InputType,
    *,
    scratch_dir: StrOrPath | None = None,
    rm_scratch_dir: bool = True,
    collect_logs: bool = True,
    collect_files: bool = False,
    collect_wfn: bool = False,
    update_func: Callable | None = None,
    update_interval: float | None = None,
    print_logs: bool = False,
    raise_exc: bool = True,
    propagate_wfn: bool = False,
    qcng_fallback: bool = True,
    **adapter_kwargs,
) -> ProgramOutput[InputType, Data]:
    """Use the given program to compute on the given input.

    See BaseAdapter.compute for more details.
    """
    start = time()
    try:
        adapter = get_adapter(input_data.program, input_data, qcng_fallback)
    except (AdapterError, ProgramNotFoundError) as exc:
        output = ProgramOutput(
            input_data=input_data,
            results=empty_results(input_data),
            success=False,
            execution=construct_execution(None, time() - start),
            traceback=traceback.format_exc(),
        )
        exc.prog_output = output
        if raise_exc:
            raise
        return output

    return adapter.compute(
        input_data,
        scratch_dir=scratch_dir,
        rm_scratch_dir=rm_scratch_dir,
        collect_logs=collect_logs,
        collect_files=collect_files,
        collect_wfn=collect_wfn,
        update_func=update_func,
        update_interval=update_interval,
        print_logs=print_logs,
        raise_exc=raise_exc,
        propagate_wfn=propagate_wfn,
        **adapter_kwargs,
    )


def compute_args(
    program: str,
    structure: Structure,
    *,
    calctype: str | CalcType,
    model: dict[str, str] | Model | None = None,
    keywords: dict[str, Any] | None = None,
    files: dict[str, str | bytes] | Files | None = None,
    extras: dict[str, Any] | None = None,
    subprograms: list[ProgramSpec] | None = None,
    structures: dict[str, Structure] | None = None,
    cmdline_args: list[str] | None = None,
    **kwargs,
) -> ProgramOutput[ProgramInput, Data]:
    """Compute function that accepts independent argument for a ProgramInput.

    Args:
        program: The program to run.
        structure: The structure to use.
        calctype: The type of calculation to run.
        model: The model to use for the calculation.
        keywords: The keywords to use for the calculation.
        files: The files to use for the calculation. Either a qcdata.Files object or a
            dict mapping file names to file contents (bytes or str).
        extras: User metadata saved with the input.
        subprograms: Recursive specifications for child calculations.
        structures: Additional complete structures identified by role.
        cmdline_args: Additional command-line arguments for the program.
        **kwargs: Extra arguments to pass to the compute function.

    Returns:
        The output of the computation.

    Raises:
        See compute function for details.
    """
    if isinstance(files, Files):  # Check in case Files object is passed instead of dict
        files = files.files

    program_input = ProgramInput(
        program=program,
        calctype=CalcType(calctype),
        structure=structure,
        model=model,  # type: ignore
        keywords=keywords or {},
        files=files or {},
        extras=extras or {},
        subprograms=subprograms or [],
        structures=structures or {},
        cmdline_args=cmdline_args or [],
    )

    return compute(program_input, **kwargs)
