"""Deepwater Gas Hydrate Formation & Thermodynamic Inhibitor Dosing Engine.

Models natural gas hydrate formation equilibrium in cold deepwater environments
(subsea BOP and choke lines) and calculates required injection rates of thermodynamic
inhibitors (Methanol or Monoethylene Glycol / MEG) to prevent line plugging.

References:
- Sloan, E.D. & Koh, C.A. (2007): "Clathrate Hydrates of Natural Gases," 3rd Ed., CRC Press.
- Hammerschmidt, E.G. (1934): "Formation of Gas Hydrates in Natural Gas Transmission
  Lines," Ind. Eng. Chem. 26(8), 851-855.
- Towler, B.F. & Mokhatab, S. (2005): "New correlation estimates hydrate formation
  conditions for natural gas," Oil & Gas Journal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = [
    "InhibitorType",
    "HydrateRiskAssessment",
    "hydrate_formation_temperature",
    "calculate_inhibitor_depression",
    "calculate_required_inhibitor_rate",
]


class InhibitorType:
    METHANOL = "methanol"
    MEG = "monoethylene_glycol"  # Monoethylene Glycol


@dataclass(frozen=True)
class HydrateRiskAssessment:
    """Hydrate risk screening result at local temperature and pressure.

    Attributes:
        p_psia: Local pressure (psia).
        t_fluid_degf: Local in-situ fluid temperature (degF).
        t_hydrate_degf: Hydrate equilibrium formation temperature (degF).
        is_hydrate_risk: True if fluid temperature is at or below hydrate point (freezing risk).
        subcooling_degf: Extent of thermal subcooling below hydrate threshold (degF).
        methanol_wt_pct_required: Weight percent of methanol in water needed to inhibit hydrates.
        meg_wt_pct_required: Weight percent of MEG in water needed to inhibit hydrates.
        inhibitor_volume_rate_gpm: Recommended injection rate of pure inhibitor (gpm).
    """

    p_psia: float
    t_fluid_degf: float
    t_hydrate_degf: float
    is_hydrate_risk: bool
    subcooling_degf: float
    methanol_wt_pct_required: float
    meg_wt_pct_required: float
    inhibitor_volume_rate_gpm: float


def hydrate_formation_temperature(
    p_psia: float,
    gas_sg: float = 0.65,
) -> float:
    """Calculate natural gas hydrate dissociation equilibrium temperature (degF).

    Uses Towler & Mokhatab / Sloan correlation for Structure I & II natural gas hydrates:
    Valid for pressures 200 to 10,000 psia and gas specific gravities 0.55 to 0.85.

    Returns:
        Equilibrium hydrate formation temperature in degF.
    """
    if p_psia <= 14.7:
        return 32.0

    p = max(50.0, min(12000.0, p_psia))
    sg = max(0.55, min(0.90, gas_sg))

    # Empirical correlation for Structure II hydrate equilibrium:
    # ln(P) relationship modified by gas gravity
    # T_hyd = a * ln(P) + b * SG + c
    ln_p = math.log(p)
    t_f = 13.384 * ln_p + 28.5 * (sg - 0.60) - 34.2
    return max(32.0, min(85.0, t_f))


def calculate_inhibitor_depression(
    subcooling_degf: float,
    inhibitor_type: str = InhibitorType.METHANOL,
    safety_margin_degf: float = 5.0,
) -> float:
    """Calculate required inhibitor weight percentage in water using the Hammerschmidt equation.

    Hammerschmidt (1934):
        Delta_T = (K_h * W) / (M * (100 - W))
    Inverted for W (wt%):
        W = (100 * M * Delta_T) / (K_h + M * Delta_T)

    Args:
        subcooling_degf: Difference between hydrate temperature and fluid temperature (degF).
        inhibitor_type: 'methanol' or 'monoethylene_glycol'.
        safety_margin_degf: Added thermal safety buffer (default 5.0 degF).

    Returns:
        Required inhibitor concentration in water (wt%, 0 to 80%).
    """
    total_delta_t = max(0.0, subcooling_degf + safety_margin_degf)
    if total_delta_t <= 0.0:
        return 0.0

    if inhibitor_type == InhibitorType.METHANOL:
        k_h = 2335.0
        m = 32.04  # Molecular weight of CH3OH (g/mol)
    else:
        k_h = 2700.0
        m = 62.07  # Molecular weight of C2H6O2 (g/mol)

    w = (100.0 * m * total_delta_t) / (k_h + m * total_delta_t)
    return min(80.0, max(0.0, w))


def calculate_required_inhibitor_rate(
    p_psia: float,
    t_fluid_degf: float,
    water_flow_rate_gpm: float,
    gas_sg: float = 0.65,
    inhibitor_type: str = InhibitorType.METHANOL,
    safety_margin_degf: float = 5.0,
) -> HydrateRiskAssessment:
    """Screen for hydrate plugging risk and size inhibitor injection pumping rate.

    Args:
        p_psia: Pressure at subsea BOP or choke line (psia).
        t_fluid_degf: Measured or modeled fluid temperature (degF).
        water_flow_rate_gpm: Produced or condensed free water rate (gpm).
        gas_sg: Influx gas specific gravity.
        inhibitor_type: 'methanol' or 'monoethylene_glycol'.
        safety_margin_degf: Thermal safety margin below hydrate temperature.

    Returns:
        HydrateRiskAssessment.
    """
    t_hyd = hydrate_formation_temperature(p_psia, gas_sg)
    subcooling = max(0.0, t_hyd - t_fluid_degf)
    is_risk = subcooling > 0.0

    w_meth = calculate_inhibitor_depression(subcooling, InhibitorType.METHANOL, safety_margin_degf)
    w_meg = calculate_inhibitor_depression(subcooling, InhibitorType.MEG, safety_margin_degf)

    # Calculate volumetric injection rate of inhibitor (gpm):
    # W = (m_inhibitor) / (m_inhibitor + m_water) -> m_inhibitor = m_water * (W / (100 - W))
    w_target = w_meth if inhibitor_type == InhibitorType.METHANOL else w_meg
    if w_target > 0.0 and water_flow_rate_gpm > 0.0:
        rho_water_ppg = 8.34
        rho_inhibitor_ppg = 6.61 if inhibitor_type == InhibitorType.METHANOL else 9.28
        mass_water_lbm_min = water_flow_rate_gpm * rho_water_ppg
        mass_inhibitor_lbm_min = mass_water_lbm_min * (w_target / (100.0 - w_target))
        q_inhibitor_gpm = mass_inhibitor_lbm_min / rho_inhibitor_ppg
    else:
        q_inhibitor_gpm = 0.0

    return HydrateRiskAssessment(
        p_psia=round(p_psia, 1),
        t_fluid_degf=round(t_fluid_degf, 1),
        t_hydrate_degf=round(t_hyd, 1),
        is_hydrate_risk=is_risk,
        subcooling_degf=round(subcooling, 1),
        methanol_wt_pct_required=round(w_meth, 1),
        meg_wt_pct_required=round(w_meg, 1),
        inhibitor_volume_rate_gpm=round(q_inhibitor_gpm, 2),
    )
