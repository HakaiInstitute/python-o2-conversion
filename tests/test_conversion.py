import o2conversion
import pytest


@pytest.mark.parametrize("O2conc,T,S,P,expected_value", [(2, 2, 2, 2, 0.98904)])
def test_O2ctoO2p_conversion(O2conc, T, S, P, expected_value):
    value = o2conversion.O2ctoO2p(O2conc, T, S, P)
    assert value == pytest.approx(expected_value, 1e-5)


@pytest.mark.parametrize(
    "O2conc,T,S,P,p_atm,expected_value", [(2, 2, 2, 2, 1000, 0.475537)]
)
def test_O2ctoO2s_conversion(O2conc, T, S, P, p_atm, expected_value):
    value = o2conversion.O2ctoO2s(O2conc, T, S, P, p_atm)
    assert value == pytest.approx(expected_value, 1e-5)


@pytest.mark.parametrize("pO2,T,S,P,expected_value", [(2, 2, 2, 2, 4.04431)])
def test_O2ptoO2c_conversion(pO2, T, S, P, expected_value):
    value = o2conversion.O2ptoO2c(pO2, T, S, P)
    assert value == pytest.approx(expected_value, 1e-5)


@pytest.mark.parametrize(
    "pO2,T,S,P,p_atm,expected_value", [(2, 2, 2, 2, 1000, 0.96161)]
)
def test_O2ptoO2s_conversion(pO2, T, S, P, p_atm, expected_value):
    value = o2conversion.O2ptoO2s(pO2, T, S, P, p_atm)
    assert value == pytest.approx(expected_value, 1e-5)


@pytest.mark.parametrize(
    "O2sat,T,S,P,p_atm,expected_value", [(2, 2, 2, 2, 1000, 8.41155)]
)
def test_O2stoO2c_conversion(O2sat, T, S, P, p_atm, expected_value):
    value = o2conversion.O2stoO2c(O2sat, T, S, P, p_atm)
    assert value == pytest.approx(expected_value, 1e-5)


@pytest.mark.parametrize(
    "O2sat,T,S,P,p_atm,expected_value", [(2, 2, 2, 2, 1000, 4.15969)]
)
def test_O2stoO2p_conversion(O2sat, T, S, P, p_atm, expected_value):
    value = o2conversion.O2stoO2p(O2sat, T, S, P, p_atm)
    assert value == pytest.approx(expected_value, 1e-5)
