import numpy as np
import pytest
from qcdata import CalcType, ProgramInput, Structure

from qccompute.main import compute
from tests.conftest import skipif_program_not_available


@pytest.mark.integration
@skipif_program_not_available("terachem")
def test_terachem_energy(hydrogen):
    # Modify keywords
    energy_inp = ProgramInput(
        program="terachem",
        structure=hydrogen,
        calctype=CalcType.energy,
        model={"method": "hf", "basis": "sto-3g"},
        keywords={"purify": "no"},
    )
    program = "terachem"
    output = compute(energy_inp)
    assert output.input_data == energy_inp
    assert output.results.provenance.program == program
    assert np.isclose(output.results.energy, -1.1167143325, atol=1e-6)


@pytest.mark.integration
@skipif_program_not_available("terachem")
def test_terachem_gradient(hydrogen):
    # Modify keywords
    energy_inp = ProgramInput(
        program="terachem",
        structure=hydrogen,
        calctype=CalcType.gradient,
        model={"method": "hf", "basis": "sto-3g"},
        keywords={"purify": "no"},
    )

    program = "terachem"
    output = compute(energy_inp)
    assert output.input_data == energy_inp
    assert output.results.provenance.program == program
    assert np.isclose(output.results.energy, -1.1167143325, atol=1e-6)
    assert np.allclose(
        output.results.gradient,
        np.array([[0.0, 0.0, -0.02845402], [0.0, 0.0, 0.02845402]]),
        atol=1e-6,
    )


@pytest.mark.integration
@skipif_program_not_available("terachem")
def test_terachem_hessian():
    # Modify keywords
    # This H2O converges in TeraChem
    h2o = Structure(
        symbols=["O", "H", "H"],
        geometry=[
            [0.022531001997713594, 0.017650541231852297, -0.012249224534705398],
            [0.5009229288099285, 1.7183882034872506, 0.5180806028365682],
            [1.147442147965631, -0.49902392855765926, -1.3798296942644486],
        ],
    )
    energy_inp = ProgramInput(
        program="terachem",
        structure=h2o,
        calctype=CalcType.hessian,
        model={"method": "b3lyp", "basis": "6-31g"},
        keywords={"purify": "no"},
    )

    program = "terachem"
    result = compute(energy_inp)

    # General assertions
    assert result.input_data == energy_inp
    assert result.results.provenance.program == program

    # Energy assertion
    assert np.isclose(result.results.energy, -76.3861099088, atol=1e-6)

    # Gradient assertion
    assert np.allclose(
        result.results.gradient,
        np.array(
            [
                [-2.69528e-05, -3.88595e-05, 3.06421e-05],
                [1.15012e-05, 2.39264e-05, 4.2012e-06],
                [1.5448e-05, 1.49307e-05, -3.48414e-05],
            ]
        ),
        atol=1e-6,
    )

    # Hessian assertion
    assert np.allclose(
        result.results.hessian,
        np.array(
            [
                [
                    0.2319485298040269,
                    0.04240308383589297,
                    -0.195377613895862,
                    -0.05131768231265779,
                    -0.08413835236086714,
                    -0.0002594768906270112,
                    -0.1807328894945268,
                    0.04171577173879733,
                    0.1956319703780435,
                ],
                [
                    0.04240308383589297,
                    0.472135635387827,
                    0.2320401621187684,
                    -0.1326385068310412,
                    -0.3998520582766313,
                    -0.1050911243234316,
                    0.09024761885613602,
                    -0.07227039336594292,
                    -0.1267194190358992,
                ],
                [
                    -0.195377613895862,
                    0.2320401621187684,
                    0.3202530388681138,
                    -0.0283249654646614,
                    -0.1513488477912904,
                    -0.06079076055028922,
                    0.223814543990352,
                    -0.08067254057485228,
                    -0.2593141043188557,
                ],
                [
                    -0.05131768231265779,
                    -0.1326385068310412,
                    -0.0283249654646614,
                    0.04979830933957985,
                    0.1011937317374027,
                    0.01152779363959022,
                    0.001533661600949116,
                    0.03145103663504886,
                    0.01672681598697059,
                ],
                [
                    -0.08413835236086714,
                    -0.3998520582766313,
                    -0.1513488477912904,
                    0.1011937317374027,
                    0.4152544361704591,
                    0.1438971253769659,
                    -0.01704107167130042,
                    -0.01540626005193634,
                    0.007287536095455793,
                ],
                [
                    -0.0002594768906270112,
                    -0.1050911243234316,
                    -0.06079076055028922,
                    0.01152779363959022,
                    0.1438971253769659,
                    0.07247868855366213,
                    -0.01126857222636078,
                    -0.03881133668853058,
                    -0.01172710920857058,
                ],
                [
                    -0.1807328894945268,
                    0.09024761885613602,
                    0.223814543990352,
                    0.001533661600949116,
                    -0.01704107167130042,
                    -0.01126857222636078,
                    0.1792874103355352,
                    -0.07319313552580954,
                    -0.2124705394875498,
                ],
                [
                    0.04171577173879733,
                    -0.07227039336594292,
                    -0.08067254057485228,
                    0.03145103663504886,
                    -0.01540626005193634,
                    -0.03881133668853058,
                    -0.07319313552580954,
                    0.08766703119815339,
                    0.1194182938515254,
                ],
                [
                    0.1956319703780435,
                    -0.1267194190358992,
                    -0.2593141043188557,
                    0.01672681598697059,
                    0.007287536095455793,
                    -0.01172710920857058,
                    -0.2124705394875498,
                    0.1194182938515254,
                    0.2709321551738031,
                ],
            ]
        ),
        atol=1e-6,
    )


@pytest.mark.integration
@skipif_program_not_available("terachem")
def test_terachem_optimization(water):
    # Modify keywords
    energy_inp = ProgramInput(
        program="terachem",
        structure=water,
        calctype=CalcType.gradient,
        model={"method": "hf", "basis": "sto-3g"},
        keywords={"purify": "no", "new_minimizer": "yes"},
    )

    program = "terachem"
    output = compute(energy_inp)
    assert output.input_data == energy_inp
    assert output.results.provenance.program == program
