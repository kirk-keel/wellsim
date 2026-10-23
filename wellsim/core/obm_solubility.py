"""Oil-Based Mud (OBM) & Synthetic-Based Mud (SBM) gas solubility and flash vaporization.

Models methane dissolution, bubble point pressure, and explosive near-surface flash
exsolution during well control operations in non-aqueous drilling fluids (NADF).

References:
- Thomas, D.C., O'Leary, J.F., & Dobson, C.A. (1984): "Gas Solubility in Oil-Based
  Drilling Fluids: Effects on Kick Detection," SPE 13023.
- Vasquez, M. & Beggs, H.D. (1980): "Correlations for Fluid Physical Property
  Prediction," JPT.
- O'Bryan, P.L., Bourkoyne, A.T., Monger, T.G., & Koperna, G.J. (1988):
  "Measurements of Gas Solubility in Oil-Base Drilling Fluids," SPE 18442.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from wellsim.core.pvt.dry_gas import DryGasPVT

__all__ = [
    "OBMProperties",
    "gas_solubility_rs",
    "bubble_point_pressure",
    "flash_equilibrium",
]


@dataclass(frozen=True)
class OBMProperties:
    """Oil-Based / Synthetic-Based Mud characterization.

    Attributes:
        base_oil_sg: Specific gravity of base oil (e.g. diesel ~0.84, mineral oil ~0.81, synthetic ~0.79).
        oil_water_ratio: Fraction of liquid phase that is oil (e.g. 0.80 for an 80/20 OBM).
        gas_sg: Influx gas specific gravity (air = 1.0, typical methane ~0.65).
    """

    base_oil_sg: float = 0.82
    oil_water_ratio: float = 0.80
    gas_sg: float = 0.65

    @property
    def api_gravity(self) -> float:
        """API gravity of the base oil."""
        return (141.5 / max(0.6, self.base_oil_sg)) - 131.5


def gas_solubility_rs(
    p_psia: float,
    t_f: float,
    obm: OBMProperties,
) -> float:
    """Calculate dissolved gas-oil ratio Rs (scf / bbl of base oil) at in-situ P and T.

    Uses Vasquez-Beggs formulation calibrated for light paraffinic/mineral base oils:
        Rs = C1 * gamma_gs * P^C2 * exp(C3 * API / T_R)
    """
    if p_psia <= 14.7:
        return 0.0

    api = obm.api_gravity
    t_r = t_f + 459.67

    # Gas gravity normalized to 100 psig separator reference
    gamma_gs = obm.gas_sg

    # Coefficients for API > 30 (typical base oils have API 35 to 45)
    c1 = 0.0178
    c2 = 1.1870
    c3 = 23.931

    rs = c1 * gamma_gs * (p_psia**c2) * math.exp(c3 * api / t_r)
    return max(0.0, rs)


def bubble_point_pressure(
    rs_scf_bbl: float,
    t_f: float,
    obm: OBMProperties,
) -> float:
    """Calculate bubble point pressure Pb (psia) for a given dissolved gas concentration.

    Inverts the Vasquez-Beggs solubility equation:
        Pb = [ Rs / (C1 * gamma_gs * exp(C3 * API / T_R)) ]^(1 / C2)
    """
    if rs_scf_bbl <= 0.0:
        return 14.7

    api = obm.api_gravity
    t_r = t_f + 459.67
    gamma_gs = obm.gas_sg

    c1 = 0.0178
    c2 = 1.1870
    c3 = 23.931

    denom = c1 * gamma_gs * math.exp(c3 * api / t_r)
    if denom <= 0.0:
        return 14.7

    pb = (rs_scf_bbl / denom) ** (1.0 / c2)
    return max(14.7, pb)


def flash_equilibrium(
    p_psia: float,
    t_f: float,
    total_gas_scf: float,
    mud_volume_bbl: float,
    obm: OBMProperties,
) -> Tuple[float, float, float, float]:
    """Calculate instantaneous flash equilibrium of gas in OBM at in-situ P and T.

    Args:
        p_psia: Local fluid pressure in psia.
        t_f: Local fluid temperature in degF.
        total_gas_scf: Total standard cubic feet of kick gas present in this mud volume.
        mud_volume_bbl: Total bulk volume of mud containing the gas.
        obm: OBMProperties.

    Returns:
        Tuple of:
            - dissolved_gas_scf: Standard cubic feet of gas dissolved in base oil.
            - free_gas_scf: Standard cubic feet of free (evolved) gas.
            - free_gas_bbl: In-situ volume of free gas phase in barrels.
            - liquid_swelling_bbl: Volumetric expansion of oil phase due to dissolved gas (bbl).
    """
    if total_gas_scf <= 0.0 or mud_volume_bbl <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    # Volume of active base oil in the mud
    oil_volume_bbl = mud_volume_bbl * obm.oil_water_ratio

    # Maximum solubility at local (P, T)
    rs_max = gas_solubility_rs(p_psia, t_f, obm)
    max_dissolved_scf = rs_max * oil_volume_bbl

    if total_gas_scf <= max_dissolved_scf:
        # Undersaturated condition (P > Pb): 100% of gas is dissolved
        dissolved_scf = total_gas_scf
        free_gas_scf = 0.0
    else:
        # Saturated condition (P <= Pb): excess gas flashes out as free gas
        dissolved_scf = max_dissolved_scf
        free_gas_scf = total_gas_scf - max_dissolved_scf

    # Calculate free gas in-situ volume (bbl) using real-gas law
    if free_gas_scf > 0.0 and p_psia > 0.0:
        t_r = t_f + 459.67
        gas_pvt = DryGasPVT(gas_sg=obm.gas_sg)
        z = gas_pvt.z(p_psia, t_f)
        # V_ft3 = n * R * T * Z / P = (scf / 379.3) * 10.73 * T_r * Z / P
        # Conversion: 1 bbl = 5.61458 ft^3
        v_ft3 = free_gas_scf * (14.696 / p_psia) * (t_r / 519.67) * z
        free_gas_bbl = v_ft3 / 5.61458
    else:
        free_gas_bbl = 0.0

    # Liquid swelling factor B_o: empirical swelling of base oil due to dissolved methane
    # delta_V_oil = oil_volume_bbl * (B_o - 1)
    # Vasquez-Beggs Bo - 1 approximation: ~ 1.2e-4 * Rs + 1.0e-4 * (T - 60)
    rs_actual = dissolved_scf / max(1e-4, oil_volume_bbl)
    swelling_ratio = 1.2e-4 * rs_actual + 1.0e-4 * max(0.0, t_f - 60.0)
    swelling_ratio = max(0.0, min(0.40, swelling_ratio))  # Cap realistic oil swelling at 40%
    liquid_swelling_bbl = oil_volume_bbl * swelling_ratio

    return dissolved_scf, free_gas_scf, free_gas_bbl, liquid_swelling_bbl
