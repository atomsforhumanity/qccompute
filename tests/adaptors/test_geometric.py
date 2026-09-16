import pytest
from qcdata import (
    CalcType,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    SinglePointData,
)

from qccompute.adapters import GeometricAdapter
from qccompute.adapters.utils import tmpdir
from qccompute.exceptions import (
    ExternalProgramError,
    ProgramNotFoundError,
)


def test_ensure_geometric():
    # Test successful import
    try:
        import geometric

        is_geometric_available = True
    except ImportError:
        is_geometric_available = False

    if is_geometric_available:
        result = GeometricAdapter()._ensure_geometric_installed()
        assert result == geometric
    else:
        # Test when geometric is not available
        with pytest.raises(ProgramNotFoundError) as excinfo:
            GeometricAdapter()._ensure_geometric_installed()
        assert "geometric" in str(excinfo.value)


@pytest.mark.parametrize(
    "calctype, expected",
    [
        (CalcType.transition_state, True),
        (CalcType.optimization, False),
    ],
)
def test_update_input_data(calctype, expected, nested_input_factory):
    adapter = GeometricAdapter()
    prog_input = nested_input_factory(calctype)
    adapter._update_input_data(prog_input)
    assert prog_input.keywords["transition"] is expected


def test_create_geometric_molecule(hydrogen):
    adapter = GeometricAdapter()

    with tmpdir() as scratch_dir:
        geometric_molecule = adapter._create_geometric_molecule(hydrogen)

        # assertions
        assert isinstance(geometric_molecule, adapter.geometric.molecule.Molecule)
        assert not any(scratch_dir.iterdir())  # check that the file was deleted


def test_qcdata_geometric_engine_exception_handling(
    test_adapter, hydrogen, results, mocker
):
    adapter = GeometricAdapter()  # Just for using helper methods below
    QCComputeGeometricEngine = adapter._geometric_engine()
    geometric_hydrogen = adapter._create_geometric_molecule(hydrogen)

    engine = QCComputeGeometricEngine(
        test_adapter,
        ProgramSpec(
            calctype="gradient",
            program="test",
            **{"model": {"method": "hf", "basis": "sto-3g"}},
        ),
        hydrogen,
        geometric_hydrogen,
    )

    # Set trajectory
    engine.qcdata_trajectory = [results]

    # Create a failed ProgramOutput object
    po_dict = results.model_dump()
    po_dict.update({"success": False, "traceback": "fake traceback"})
    po_failure = ProgramOutput[ProgramInput, SinglePointData](**po_dict)

    # Mock adapter to raise ExternalProgramExecutionError
    mocker.patch.object(
        test_adapter,
        "compute",
        side_effect=ExternalProgramError(
            program="terachem",
            logs="some stdout",
            prog_output=po_failure,
        ),
    )

    with pytest.raises(ExternalProgramError) as excinfo:
        # Random coordinates for testing
        coords = hydrogen.geometry
        engine.calc_new(coords)

    assert excinfo.value.data == OptimizationData(
        provenance={
            "program": "geometric",
            "program_version": adapter.program_version(),
        },
        trajectory=[results, po_failure],
    )


def test_geometric_exceptions_converted_to_qccompute_exceptions(
    mocker, nested_input_factory
):
    adapter = GeometricAdapter()

    # cause .optimizeGeometry to raise a geomeTRIC exception
    mocker.patch(
        "geometric.optimize.Optimizer.optimizeGeometry",
        side_effect=adapter.geometric.errors.Error("Some geomeTRIC exception."),
    )

    prog_input = nested_input_factory(CalcType.optimization)
    with pytest.raises(ExternalProgramError):
        adapter.compute_data(prog_input, propagate_wfn=False)


def test_compute_data_does_not_mutate_input_keywords(mocker, hydrogen):
    adapter = GeometricAdapter()
    prog_input = ProgramInput(
        program="geometric",
        calctype=CalcType.optimization,
        structure=hydrogen,
        keywords={
            "check": 3,
            "constraints": {
                "freeze": [{"type": "distance", "indices": [0, 1], "value": 1.4}]
            },
        },
        subprograms=[
            ProgramSpec(
                calctype="gradient",
                model={"method": "hf", "basis": "sto-3g"},
                program="test",
            )
        ],
    )
    original = prog_input.model_copy(deep=True)

    mocker.patch.object(
        adapter,
        "_create_geometric_molecule",
        return_value=object(),
    )
    mocker.patch.object(
        adapter,
        "_setup_coords",
        return_value=object(),
    )

    class DummyOptimizer:
        def __init__(self):
            self.engine = type("Engine", (), {"qcdata_trajectory": []})()

        def optimizeGeometry(self):
            return None

    mocker.patch.object(
        adapter,
        "_construct_optimizer",
        return_value=DummyOptimizer(),
    )

    data, logs = adapter.compute_data(prog_input, propagate_wfn=False)

    assert data == OptimizationData(
        provenance={
            "program": "geometric",
            "program_version": adapter.program_version(),
        },
        trajectory=[],
    )
    assert logs == ""
    assert prog_input == original


def test_gradient_child_spec_is_bound_without_losing_fields(
    test_adapter, hydrogen, mocker, tmp_path, monkeypatch
):
    import numpy as np

    monkeypatch.chdir(tmp_path)
    adapter = GeometricAdapter()
    child = ProgramSpec.model_validate(
        {
            "program": "test",
            "calctype": "gradient",
            "model": {"method": "hf", "basis": "sto-3g"},
            "keywords": {"threshold": 1e-6},
            "files": {"native.in": "input"},
            "cmdline_args": ["--threads", "2"],
            "extras": {"label": "child"},
            "subprograms": [{"program": "other", "calctype": "energy"}],
        }
    )

    def compute_child(input_data, **kwargs):
        return ProgramOutput(
            input_data=input_data,
            success=True,
            results=SinglePointData(
                provenance={"program": "test"}, energy=-1.0, gradient=np.zeros((2, 3))
            ),
        )

    compute_mock = mocker.patch.object(
        test_adapter, "compute", side_effect=compute_child
    )
    engine = adapter._geometric_engine()(
        test_adapter,
        child,
        hydrogen,
        adapter._create_geometric_molecule(hydrogen),
        structures={"reference": hydrogen},
    )
    result = engine.calc_new(hydrogen.geometry.flatten())
    bound = compute_mock.call_args.args[0]
    for field in (
        "program",
        "calctype",
        "model",
        "keywords",
        "files",
        "cmdline_args",
        "extras",
        "subprograms",
    ):
        assert getattr(bound, field) == getattr(child, field)
    assert bound.structures == {"reference": hydrogen}
    assert result["energy"] == -1.0
    assert len(engine.qcdata_trajectory) == 1


def test_missing_gradient_child_is_an_input_error(hydrogen):
    input_data = ProgramInput.model_validate(
        {"program": "geometric", "calctype": "optimization", "structure": hydrogen}
    )
    output = GeometricAdapter().compute(input_data, raise_exc=False)
    assert not output.success
    assert isinstance(output.results, OptimizationData)
    assert output.results.trajectory == []
    assert "gradient" in output.traceback
