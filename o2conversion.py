"""
Converstion of the matlab package SCOR WG 142 to python

SCOR WG 142: Quality Control Procedures for Oxygen and Other 
Biogeochemical Sensors on Floats and Gliders. Recommendations on 
the conversion between oxygen quantities for Bio-Argo floats 
and other autonomous sensor platforms.

https://archimer.ifremer.fr/doc/00348/45915/
DOI: 10.13155/45915
"""

import numpy as np

xO2 = 0.20946  # mole fraction of O2 in dry air (Glueckauf 1951)
Vm = 0.317  # molar volume of O2 in m3 mol-1 Pa dbar-1 (Enns et al. 1965)
R = 8.314  # universal gas constant in J mol-1 K-1
DEFAULT_P_ATM = 1013.25  # mbar


def O2ctoO2p(O2conc: float, T: float, S: float, P: float = 0) -> float:
    """Convert molar oxygen concentration to oxygen partial pressure
    according to recommendations by SCOR WG 142 "Quality Control Procedures
    for Oxygen and Other Biogeochemical Sensors on Floats and Gliders

    Args:
        O2conc (float): oxygen concentration in umol L-1
        T (float): temperature in °C
        S (float): salinity (PSS-78)
        P (float, optional): hydrostatic pressure in dbar (default: 0 dbar). 
            Defaults to 0.

    Returns:
        float: oxygen partial pressure in mbar
    """

    pH2Osat = 1013.25 * (
        np.exp(
            24.4543
            - (67.4509 * (100 / (T + 273.15)))
            - (4.8489 * np.log(((273.15 + T) / 100)))
            - 0.000544 * S
        )
    )  # saturated water vapor in mbar
    sca_T = np.log(
        (298.15 - T) / (273.15 + T)
    )  # scaled temperature for use in TCorr and SCorr
    TCorr = 44.6596 * np.exp(
        2.00907
        + 3.22014 * sca_T
        + 4.05010 * sca_T**2
        + 4.94457 * sca_T**3
        - 2.56847e-1 * sca_T**4
        + 3.88767 * sca_T**5
    )
    # temperature correction part from Garcia and Gordon (1992),
    # Benson and Krause (1984) refit mL(STP) L-1 and
    # conversion from mL(STP) L-1 to umol L-1
    Scorr = np.exp(
        S
        * (
            -6.24523e-3
            - 7.37614e-3 * sca_T
            - 1.03410e-2 * sca_T**2
            - 8.17083e-3 * sca_T**3
        )
        - 4.88682e-7 * S**2
    )
    # salinity correction part from
    # Garcia and Gordon (1992), Benson and Krause (1984) refit ml(STP) L-1

    return (
        O2conc
        * (xO2 * (1013.25 - pH2Osat))
        / (TCorr * Scorr)
        * np.exp(Vm * P / (R * (T + 273.15)))
    )


def O2ctoO2s(
    O2conc: float, T: float, S: float, P: float = 0, p_atm: float = DEFAULT_P_ATM
) -> float:
    """Convert molar oxygen concentration to oxygen saturation
    according to recommendations by SCOR WG 142 "Quality Control Procedures
    for Oxygen and Other Biogeochemical Sensors on Floats and Gliders

    Args:
        O2conc (float): oxygen concentration in umol L-1
        T (float): temperature in °C
        S (float): salinity (PSS-78)
        P (float, optional): hydrostatic pressure in dbar. Defaults to 0.
        p_atm (float, optional): atmospheric (air) pressure in mbar.
            Defaults to DEFAULT_P_ATM.

    Returns:
        float:  oxygen saturation in %
    """

    pH2Osat = 1013.25 * (
        np.exp(
            24.4543
            - (67.4509 * (100 / (T + 273.15)))
            - (4.8489 * np.log(((273.15 + T) / 100)))
            - 0.000544 * S
        )
    )  # saturated water vapor in mbar
    sca_T = np.log(
        (298.15 - T) / (273.15 + T)
    )  # scaled temperature for use in TCorr and SCorr
    TCorr = 44.6596 * np.exp(
        2.00907
        + 3.22014 * sca_T
        + 4.05010 * sca_T**2
        + 4.94457 * sca_T**3
        - 2.56847e-1 * sca_T**4
        + 3.88767 * sca_T**5
    )
    # temperature correction part from Garcia and Gordon (1992),
    # Benson and Krause (1984) refit mL(STP) L-1 and
    # conversion from mL(STP) L-1 to umol L-1
    Scorr = np.exp(
        S
        * (
            -6.24523e-3
            - 7.37614e-3 * sca_T
            - 1.03410e-2 * sca_T**2
            - 8.17083e-3 * sca_T**3
        )
        - 4.88682e-7 * S**2
    )
    # salinity correction part from Garcia and Gordon (1992),
    # Benson and Krause (1984) refit ml(STP) L-1

    return (
        O2conc
        * 100
        / (TCorr * Scorr)
        / (p_atm - pH2Osat)
        * (1013.25 - pH2Osat)
        * np.exp(Vm * P / (R * (T + 273.15)))
    )


def O2ptoO2c(pO2: float, T: float, S: float, P: float = 0) -> float:
    """Convert oxygen partial pressure to molar oxygen concentration
    according to recommendations by SCOR WG 142 "Quality Control Procedures
    for Oxygen and Other Biogeochemical Sensors on Floats and Gliders

    Args:
        pO2 (float): oxygen partial pressure in mbar
        T (float): temperature in °C
        S (float): salinity (PSS-78)
        P (float, optional): ydrostatic pressure in dbar. Defaults to 0.

    Returns:
        float: oxygen concentration in umol L-1
    """

    pH2Osat = 1013.25 * (
        np.exp(
            24.4543
            - (67.4509 * (100 / (T + 273.15)))
            - (4.8489 * np.log(((273.15 + T) / 100)))
            - 0.000544 * S
        )
    )  # saturated water vapor in mbar
    sca_T = np.log(
        (298.15 - T) / (273.15 + T)
    )  # scaled temperature for use in TCorr and SCorr
    TCorr = 44.6596 * np.exp(
        2.00907
        + 3.22014 * sca_T
        + 4.05010 * sca_T**2
        + 4.94457 * sca_T**3
        - 2.56847e-1 * sca_T**4
        + 3.88767 * sca_T**5
    )
    # temperature correction part from Garcia and Gordon (1992),
    # Benson and Krause (1984) refit mL(STP) L-1
    # and conversion from mL(STP) L-1 to umol L-1
    Scorr = np.exp(
        S
        * (
            -6.24523e-3
            - 7.37614e-3 * sca_T
            - 1.03410e-2 * sca_T**2
            - 8.17083e-3 * sca_T**3
        )
        - 4.88682e-7 * S**2
    )
    # salinity correction part from Garcia and Gordon (1992),
    # Benson and Krause (1984) refit ml(STP) L-1

    return (
        pO2
        / (xO2 * (1013.25 - pH2Osat))
        * (TCorr * Scorr)
        / np.exp(Vm * P / (R * (T + 273.15)))
    )


def O2ptoO2s(
    pO2: float, T: float, S: float, P: float = 0, p_atm: float = DEFAULT_P_ATM
) -> float:
    """Convert oxygen partial pressure to oxygen saturation
    according to recommendations by SCOR WG 142 "Quality Control Procedures
    for Oxygen and Other Biogeochemical Sensors on Floats and Gliders

    Args:
        pO2 (float): oxygen partial pressure in mbar
        T (float):  temperature in °C
        S (float): salinity (PSS-78)
        P (float, optional): hydrostatic pressure in dbar. Defaults to 0.
        p_atm (float, optional): atmospheric (air) pressure in mbar.
            Defaults to DEFAULT_P_ATM.

    Returns:
        float: oxygen saturation in percent
    """

    pH2Osat = 1013.25 * (
        np.exp(
            24.4543
            - (67.4509 * (100 / (T + 273.15)))
            - (4.8489 * np.log(((273.15 + T) / 100)))
            - 0.000544 * S
        )
    )  # saturated water vapor in mbar

    return pO2 * 100 / (xO2 * (p_atm - pH2Osat))


def O2stoO2c(
    O2sat: float, T: float, S: float, P: float = 0, p_atm: float = DEFAULT_P_ATM
) -> float:
    """Convert oxygen saturation to molar oxygen concentration
      according to recommendations by SCOR WG 142 "Quality Control Procedures
      for Oxygen and Other Biogeochemical Sensors on Floats and Gliders"

    Args:
        O2sat (float): oxygen saturation in %
        T (float): temperature in °C
        S (float): salinity (PSS-78)
        P (float, optional): hydrostatic pressure in dbar. Defaults to 0.
        p_atm (float, optional): atmospheric (air) pressure in mbar.
            Defaults to DEFAULT_P_ATM.

    Returns:
        float: oxygen concentration in umol L-1
    """

    pH2Osat = 1013.25 * (
        np.exp(
            24.4543
            - (67.4509 * (100 / (T + 273.15)))
            - (4.8489 * np.log(((273.15 + T) / 100)))
            - 0.000544 * S
        )
    )  # saturated water vapor in mbar
    sca_T = np.log(
        (298.15 - T) / (273.15 + T)
    )  # scaled temperature for use in TCorr and SCorr
    TCorr = 44.6596 * np.exp(
        2.00907
        + 3.22014 * sca_T
        + 4.05010 * sca_T**2
        + 4.94457 * sca_T**3
        - 2.56847e-1 * sca_T**4
        + 3.88767 * sca_T**5
    )
    # temperature correction part from Garcia and Gordon (1992),
    # Benson and Krause (1984) refit mL(STP) L-1
    # and conversion from mL(STP) L-1 to umol L-1
    Scorr = np.exp(
        S
        * (
            -6.24523e-3
            - 7.37614e-3 * sca_T
            - 1.03410e-2 * sca_T**2
            - 8.17083e-3 * sca_T**3
        )
        - 4.88682e-7 * S**2
    )
    # salinity correction part from Garcia and Gordon (1992),
    # Benson and Krause (1984) refit ml(STP) L-1

    return (
        O2sat
        / 100
        * (TCorr * Scorr)
        * (p_atm - pH2Osat)
        / (1013.25 - pH2Osat)
        / np.exp(Vm * P / (R * (T + 273.15)))
    )


def O2stoO2p(
    O2sat: float, T: float, S: float, P: float = 0, p_atm: float = DEFAULT_P_ATM
) -> float:
    """Convert oxygen saturation to oxygen partial pressure
    according to recommendations by SCOR WG 142 "Quality Control Procedures
    for Oxygen and Other Biogeochemical Sensors on Floats and Gliders

    Args:
        O2sat (float): oxygen saturation in %
        T (float): temperature in °C
        S (float): salinity (PSS-78)
        P (float, optional): hydrostatic pressure in dbar. Defaults to 0.
        p_atm (float, optional): atmospheric (air) pressure in mbar.
            Defaults to DEFAULT_P_ATM.

    Returns:
        float: oxygen partial pressure in mbar
    """

    pH2Osat = 1013.25 * (
        np.exp(
            24.4543
            - (67.4509 * (100 / (T + 273.15)))
            - (4.8489 * np.log(((273.15 + T) / 100)))
            - 0.000544 * S
        )
    )  # saturated water vapor in mbar

    return O2sat / 100 * (xO2 * (p_atm - pH2Osat))
