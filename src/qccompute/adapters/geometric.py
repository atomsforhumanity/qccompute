"""Adapter for geomeTRIC program."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from qcdata import (
    CalcType,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    SinglePointData,
    Structure,
)

from qccompute.exceptions import (
    AdapterInputError,
    ExternalProgramError,
    ProgramNotFoundError,
    QCComputeBaseError,
)
from qccompute.utils import get_adapter

from .base import ProgramAdapter
from .utils import capture_logs


class GeometricAdapter(ProgramAdapter[ProgramInput, OptimizationData]):
    """Adapter for geomeTRIC."""

    program = "geometric"
    requires_model = False
    supported_calctypes = [CalcType.optimization, CalcType.transition_state]
    """Supported calculation types."""

    def __init__(self):
        super().__init__()
        # Check that geomeTRIC is installed
        self.geometric = self._ensure_geometric_installed()
        # Dictionary from geometric.run_json; copying QCSchema workflow
        self.coordsys_params = {
            "cart": (self.geometric.internal.CartesianCoordinates, False, False),
            "prim": (self.geometric.internal.PrimitiveInternalCoordinates, True, False),
            "dlc": (
                self.geometric.internal.DelocalizedInternalCoordinates,
                True,
                False,
            ),
            "hdlc": (
                self.geometric.internal.DelocalizedInternalCoordinates,
                False,
                True,
            ),
            "tric": (
                self.geometric.internal.DelocalizedInternalCoordinates,
                False,
                False,
            ),
        }

    @staticmethod
    def _ensure_geometric_installed():
        try:
            import geometric
        except ModuleNotFoundError as e:
            raise ProgramNotFoundError("geometric") from e
        else:
            return geometric

    def program_version(self, *args) -> str:
        """Get the program version."""
        return self.geometric.__version__

    def compute_data(
        self,
        input_data: ProgramInput,
        update_func: Callable | None = None,
        update_interval: float | None = None,
        propagate_wfn: bool = True,
        **kwargs,
    ) -> tuple[OptimizationData, str]:
        """Compute the requested calculation.

        Args:
            input_data: The qcdata ProgramInput object for a computation.
            propagate_wfn: Whether to propagate the wavefunction between steps of the
                optimization.
        """
        # geomeTRIC setup mutates keywords (for example by removing constraints), so
        # operate on an internal copy and preserve the caller's original input object.
        input_data = input_data.model_copy(deep=True)

        try:
            gradient_spec = input_data.get_subprogram(CalcType.gradient)
        except ValueError as exc:
            raise AdapterInputError(self.program, str(exc)) from exc

        # Update the input object based on its calctype
        self._update_input_data(input_data)
        geometric_molecule = self._create_geometric_molecule(input_data.structure)
        internal_coords_sys = self._setup_coords(input_data, geometric_molecule)
        qcdata_adapter = get_adapter(gradient_spec.program, qcng_fallback=True)
        optimizer = self._construct_optimizer(
            input_data,
            geometric_molecule,
            internal_coords_sys,
            qcdata_adapter,
            propagate_wfn=propagate_wfn,
            update_func=update_func,
            update_interval=update_interval,
            **kwargs,
        )

        # Haven't update DualOutputHandler to send logs using arbitrary update funcs.
        with capture_logs("geometric", update_func, update_interval) as (_, log_string):
            try:
                optimizer.optimizeGeometry()
            except QCComputeBaseError as exc:
                raise ExternalProgramError(
                    program=self.program,
                    message="A child calculation failed during geomeTRIC optimization.",
                    logs=log_string.getvalue(),
                    data=exc.data,
                    original_exception=exc,
                ) from exc
            except self.geometric.errors.Error as e:
                raise ExternalProgramError(
                    program=self.program,
                    message="geomeTRIC optimization failed. See the traceback above for details.",
                    logs=log_string.getvalue(),
                    data=OptimizationData(
                        provenance=Provenance(
                            program=self.program, program_version=self.program_version()
                        ),
                        trajectory=optimizer.engine.qcdata_trajectory,
                    ),
                ) from e

        return (
            OptimizationData(
                provenance=Provenance(
                    program=self.program, program_version=self.program_version()
                ),
                trajectory=optimizer.engine.qcdata_trajectory,
            ),
            log_string.getvalue(),
        )

    def _update_input_data(self, input_data: ProgramInput) -> None:
        """Update the input_data based on its calctype

        Args:
            input_data: The qcdata ProgramInput object for a computation.

        Returns:
            None. The input_data is updated in place.
        """
        if input_data.calctype == CalcType.transition_state:
            input_data.keywords["transition"] = True
        else:
            input_data.keywords["transition"] = False

    def _create_geometric_molecule(self, structure: Structure):
        """Create a geomeTRIC Structure from the input object.

        Args:
            structure: The qcdata Structure object for a computation.

        Returns:
            The geomeTRIC Structure object.
        """
        xyz_path = "initial_geometry.xyz"
        structure.save(xyz_path)
        geometric_structure = self.geometric.molecule.Molecule(fnm=xyz_path)
        # Delete file
        Path(xyz_path).unlink()
        return geometric_structure

    def _construct_optimizer(
        self,
        input_data,
        geometric_molecule,
        internal_coords_sys,
        qcdata_adapter,
        propagate_wfn: bool = False,
        update_func: Callable | None = None,
        update_interval: float | None = None,
        **kwargs,
    ):
        """Construct the geomeTRIC optimizer object

        Args:
            input_data: The qcdata ProgramInput object for a computation.
            geometric_molecule: The geomeTRIC Molecule object.
            internal_coords_sys: The geomeTRIC internal coordinate system.
            qcdata_adapter: The qccompute adapter for the subprogram.

        Returns:
            The geomeTRIC optimizer object.
        """
        return self.geometric.optimize.Optimizer(
            input_data.structure.geometry.flatten(),
            geometric_molecule,
            internal_coords_sys,
            engine=self._geometric_engine()(
                qcdata_adapter,
                input_data.get_subprogram(CalcType.gradient),
                input_data.structure,
                geometric_molecule,
                propagate_wfn,
                update_func,
                update_interval,
                structures=input_data.structures,
            ),
            dirname=Path.cwd(),
            # Must declare xyzout here to avoid a bug in geomeTRIC
            # This file will contain every geometry in the optimization as a
            # multi-structure .xyz file with the energy of each step as a comment.
            params=self.geometric.params.OptParams(
                xyzout="qcdata_optim.xyz", **input_data.keywords
            ),
        )

    def _setup_coords(self, input_data, geometric_structure):
        """Setup the internal coordinate system.

        Args:
            input_data: The qcdata ProgramInput object for a computation.
            geometric_structure: The geomeTRIC Structure object.

        Returns:
            The geomeTRIC internal coordinate system with constraints applied.
        """

        # Default to internal coords
        coords_sys = input_data.keywords.get("coordsys", "tric")
        coords_cls, connect, addcart = self.coordsys_params[coords_sys]

        # Handle constraints; see geometric.run_json, it's a mess!
        # NOTE: From documentation says constraints may be 1-indexed instead of
        # 0-indexed; however, in practice they are 0-indexed.
        # https://geometric.readthedocs.io/en/latest/constraints.html
        constraints, constraint_vals = None, None
        constraints_dict = input_data.keywords.pop("constraints", None)
        if constraints_dict:
            if "scan" in constraints_dict:
                raise AdapterInputError(
                    "The constraint 'scan' keyword is not yet supported by the JSON "
                    "interface, reports geometric!",
                    program=self.program,
                )
            constraints_string = self.geometric.run_json.make_constraints_string(
                constraints_dict
            )

            constraints, constraint_vals = self.geometric.prepare.parse_constraints(
                geometric_structure, constraints_string
            )

        return coords_cls(
            geometric_structure,
            build=True,
            connect=connect,
            addcart=addcart,
            constraints=constraints,
            # No idea why we truncate the list here...
            cvals=constraint_vals[0] if constraint_vals is not None else None,
        )

    def _geometric_engine(self):
        """Return the qccompute geomeTRIC engine class."""

        provenance = Provenance(
            program=self.program, program_version=self.program_version()
        )

        class QCComputeGeometricEngine(self.geometric.engine.Engine):
            """QCCompute Engine for Geometric"""

            def __init__(
                self,
                qcdata_adapter: ProgramAdapter,
                qcdata_program_spec: ProgramSpec,
                qcdata_structure: Structure,
                geometric_structure,
                propagate_wfn: bool = False,
                update_func: Callable | None = None,
                update_interval: float | None = None,
                structures: dict[str, Structure] | None = None,
            ):
                super().__init__(geometric_structure)
                self.qcdata_adapter = qcdata_adapter
                self.qcdata_program_spec = qcdata_program_spec
                self.qcdata_structure = qcdata_structure
                self.qcdata_structures = structures or {}
                self.propagate_wfn = propagate_wfn
                self.qcdata_trajectory: list[ProgramOutput] = []
                self.update_func = update_func
                self.update_interval = update_interval

            def calc_new(self, coords, *args) -> dict[str, float | np.ndarray]:
                """Calculate the energy and gradient for a given geometry.

                Args:
                    coords: The geometry to evaluate in bohr.

                Returns:
                    A dictionary of {"energy": float, "gradient": ndarray}.
                """
                # Merge new coordinates into structure
                structure = Structure(
                    **{**self.qcdata_structure.model_dump(), "geometry": coords}
                )
                program_input = ProgramInput.from_spec(
                    self.qcdata_program_spec,
                    structure,
                    structures=self.qcdata_structures,
                )

                # Propagate wavefunction
                if (
                    self.qcdata_trajectory  # Not the first step
                    and self.propagate_wfn
                    and hasattr(self.qcdata_adapter, "propagate_wfn")
                ):
                    program_input = self.qcdata_adapter.propagate_wfn(
                        self.qcdata_trajectory[-1], program_input
                    )

                # Calculate energy and gradient
                try:
                    results: ProgramOutput[ProgramInput, SinglePointData] = (
                        self.qcdata_adapter.compute(
                            program_input,
                            raise_exc=True,
                            collect_wfn=self.propagate_wfn,
                            update_func=self.update_func,
                            update_interval=self.update_interval,
                        )
                    )
                except QCComputeBaseError as e:
                    if e.prog_output:  # For mypy
                        # Append error output
                        self.qcdata_trajectory.append(e.prog_output)
                    data = OptimizationData(
                        provenance=provenance, trajectory=self.qcdata_trajectory
                    )
                    e.data = data
                    # TODO: Add args/kwargs update for Celery serialization?
                    # Maybe not because .data is folded into e.prog_output in
                    # BaseAdapter.compute()?
                    raise e

                else:
                    self.qcdata_trajectory.append(results)

                assert (  # for mypy
                    results.results.energy is not None
                    and results.results.gradient is not None
                    and isinstance(results.results.gradient, np.ndarray)
                )
                return {
                    "energy": results.results.energy,
                    # geomeTRIC requires 1D array
                    "gradient": results.results.gradient.flatten(),
                }

        return QCComputeGeometricEngine
