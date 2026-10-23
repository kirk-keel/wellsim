"""Comparative Well Control Strategy Matrix.

Simulates and benchmarks the 4 major international well control methods side-by-side
for any given kick scenario (IADC WellSharp & IWCF standards):
1. Driller's Method (2 Circulations)
2. Wait & Weight / Engineer's Method (1 Circulation)
3. Volumetric Method (Lubricate & Bleed, when pumps or drillstring unavailable)
4. Dynamic Bullheading (Direct injection back into formation)

Compares:
- Peak surface casing pressure (P_cp,max)
- Maximum pressure and stress on casing shoe (P_shoe,max)
- Total fluid volume pumped (bbl)
- Total execution time to kill (hours)
- Risk level and operational complexity
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from wellsim.core.well_control import (
    KillSheetInputs,
    calculate_kill_sheet,
    hydrostatic_pressure,
)

__all__ = [
    "KillMethodStrategy",
    "KillMethodSummary",
    "compare_kill_methods",
]


class KillMethodStrategy:
    DRILLERS = "drillers_method"
    WAIT_AND_WEIGHT = "wait_and_weight"
    VOLUMETRIC = "volumetric_method"
    BULLHEADING = "bullheading"


@dataclass(frozen=True)
class KillMethodSummary:
    """Summary metrics of a specific well control method execution.

    Attributes:
        method_name: Display name of the well control method.
        circulations_required: Number of circulations (0, 1, or 2).
        peak_casing_pressure_psi: Maximum surface casing pressure encountered during kill.
        max_shoe_pressure_psi: Maximum dynamic pressure acting on casing shoe (psi).
        shoe_fractured: Whether shoe pressure exceeds formation breakdown / LOT pressure.
        total_kill_time_hr: Total operational time to complete the kill (hours).
        total_volume_pumped_bbl: Total barrels of kill fluid required.
        operational_complexity: Subjective execution rating ('Low', 'Medium', 'High', 'Extreme').
        key_advantage: Primary engineering advantage.
        key_risk: Primary operational risk.
    """

    method_name: str
    circulations_required: int
    peak_casing_pressure_psi: float
    max_shoe_pressure_psi: float
    shoe_fractured: bool
    total_kill_time_hr: float
    total_volume_pumped_bbl: float
    operational_complexity: str
    key_advantage: str
    key_risk: str


def compare_kill_methods(
    inputs: KillSheetInputs,
    scr_pump_rate_gpm: float = 120.0,
    gas_migration_rate_ft_hr: float = 1000.0,
    bullhead_rate_gpm: float = 400.0,
    formation_fracture_psi: float | None = None,
) -> Dict[str, KillMethodSummary]:
    """Compare all four well control methods for the same kick scenario.

    Args:
        inputs: IADC KillSheetInputs.
        scr_pump_rate_gpm: Slow circulating rate pump flow (gpm).
        gas_migration_rate_ft_hr: Free gas natural slip/migration velocity (ft/hr).
        bullhead_rate_gpm: Pump rate for bullheading (gpm).
        formation_fracture_psi: Formation breakdown pressure at casing shoe (psi).

    Returns:
        Dictionary mapping method ID to KillMethodSummary.
    """
    ks = calculate_kill_sheet(inputs)

    # Casing shoe breakdown threshold (psi)
    lot_shoe_psi = inputs.lot_emw_ppg * 0.052 * inputs.tvd_shoe_ft
    shoe_limit_psi = formation_fracture_psi if formation_fracture_psi is not None else lot_shoe_psi

    # Circulation volumes
    v_ds = inputs.drillstring_volume_bbl
    v_ann = inputs.annular_volume_bbl
    v_well = v_ds + v_ann
    q_bbl_hr = (scr_pump_rate_gpm * 60.0) / 42.0

    # Influx properties
    pit_gain = inputs.pit_gain_bbl
    sicp = inputs.sicp_psi
    sidpp = inputs.sidpp_psi
    omw = inputs.omw_ppg
    kmw = ks.kmw_rounded_ppg

    # 1. DRILLER'S METHOD (2 Circulations)
    # Gas expansion with OMW creates highest casing pressure peak
    # Peak casing pressure when gas reaches surface: P_cp = SICP * (1.2 to 1.4 expansion factor)
    expansion_factor_drillers = 1.30
    p_cp_max_drillers = sicp * expansion_factor_drillers
    # Maximum shoe pressure occurs when gas bubble reaches shoe depth
    p_hyd_shoe_omw = hydrostatic_pressure(omw, inputs.tvd_shoe_ft)
    p_shoe_max_drillers = p_hyd_shoe_omw + (sicp * 0.75)
    t_drillers_hr = (v_well * 2.0) / q_bbl_hr

    drillers_res = KillMethodSummary(
        method_name="Driller's Method (2 Circulations)",
        circulations_required=2,
        peak_casing_pressure_psi=round(p_cp_max_drillers, 1),
        max_shoe_pressure_psi=round(p_shoe_max_drillers, 1),
        shoe_fractured=p_shoe_max_drillers > shoe_limit_psi,
        total_kill_time_hr=round(t_drillers_hr, 1),
        total_volume_pumped_bbl=round(v_well * 2.0, 1),
        operational_complexity="Low",
        key_advantage="Can start pumping immediately without waiting to weight up mud in pits.",
        key_risk="Higher surface casing pressure and maximum stress on casing shoe.",
    )

    # 2. WAIT & WEIGHT (1 Circulation)
    # Gas is displaced ahead of the KMW front. The higher hydrostatic column of KMW
    # suppresses peak casing pressure and reduces shoe pressure by ~15-25%.
    p_cp_max_ww = sicp * 1.08
    p_shoe_max_ww = p_hyd_shoe_omw + (sicp * 0.55)
    # Time includes mud weighting wait time (~1.5 hours) + 1 circulation
    t_ww_hr = 1.5 + (v_well / q_bbl_hr)

    ww_res = KillMethodSummary(
        method_name="Wait & Weight (Engineer's Method)",
        circulations_required=1,
        peak_casing_pressure_psi=round(p_cp_max_ww, 1),
        max_shoe_pressure_psi=round(p_shoe_max_ww, 1),
        shoe_fractured=p_shoe_max_ww > shoe_limit_psi,
        total_kill_time_hr=round(t_ww_hr, 1),
        total_volume_pumped_bbl=round(v_well, 1),
        operational_complexity="Medium",
        key_advantage="Lowest peak casing pressure and lowest casing shoe stress; 1 circulation.",
        key_risk="Delay while weighting up mud in pits; potential for gas migration during wait.",
    )

    # 3. VOLUMETRIC METHOD (Lubricate & Bleed)
    # Used when pumps are inoperable or pipe is off-bottom/plugged.
    # Gas migrates up the well. Casing pressure is allowed to rise by safety margin,
    # then mud is bled off incrementally to hold constant bottomhole pressure.
    p_cp_max_vol = sicp + (inputs.tvd_well_ft * 0.052 * omw * 0.20)
    p_shoe_max_vol = p_hyd_shoe_omw + (sicp * 0.85)
    # Time depends on gas migration rate from TD to surface
    t_vol_hr = inputs.tvd_well_ft / max(100.0, gas_migration_rate_ft_hr)
    # Volume pumped is zero during migration, followed by lubricating KMW equal to annular gas volume
    v_lubricate = inputs.pit_gain_bbl * 2.5

    vol_res = KillMethodSummary(
        method_name="Volumetric Method (Lubricate & Bleed)",
        circulations_required=0,
        peak_casing_pressure_psi=round(p_cp_max_vol, 1),
        max_shoe_pressure_psi=round(p_shoe_max_vol, 1),
        shoe_fractured=p_shoe_max_vol > shoe_limit_psi,
        total_kill_time_hr=round(t_vol_hr, 1),
        total_volume_pumped_bbl=round(v_lubricate, 1),
        operational_complexity="High",
        key_advantage="Can control well without drillstring on bottom or working mud pumps.",
        key_risk="Lengthy execution; high surface casing pressures while bleeding gas.",
    )

    # 4. DYNAMIC BULLHEADING
    # Pumping heavy mud directly down the annulus to force influx back into the formation.
    # Surface pressure = P_inject - P_hyd + P_friction
    p_form_inject = inputs.formation_pressure_psi if hasattr(inputs, "formation_pressure_psi") else (inputs.tvd_well_ft * 0.052 * inputs.lot_emw_ppg * 0.95)
    p_hyd_kmw = hydrostatic_pressure(kmw, inputs.tvd_well_ft)
    p_friction_bullhead = 450.0  # Frictional backpressure at high pump rate
    p_surf_bullhead = max(200.0, p_form_inject - p_hyd_kmw + p_friction_bullhead)
    p_shoe_max_bullhead = hydrostatic_pressure(kmw, inputs.tvd_shoe_ft) + (p_surf_bullhead * 0.6)
    q_bull_bbl_hr = (bullhead_rate_gpm * 60.0) / 42.0
    t_bull_hr = (v_ann * 0.5) / q_bull_bbl_hr

    bullhead_res = KillMethodSummary(
        method_name="Dynamic Bullheading",
        circulations_required=0,
        peak_casing_pressure_psi=round(p_surf_bullhead, 1),
        max_shoe_pressure_psi=round(p_shoe_max_bullhead, 1),
        shoe_fractured=p_shoe_max_bullhead > shoe_limit_psi,
        total_kill_time_hr=round(t_bull_hr, 1),
        total_volume_pumped_bbl=round(v_ann * 0.6, 1),
        operational_complexity="Extreme",
        key_advantage="Fastest method; zero hydrocarbons brought to surface (ideal for H2S kicks).",
        key_risk="Intentional formation fracturing; risk of underground blowout if thief zone ruptures.",
    )

    return {
        KillMethodStrategy.DRILLERS: drillers_res,
        KillMethodStrategy.WAIT_AND_WEIGHT: ww_res,
        KillMethodStrategy.VOLUMETRIC: vol_res,
        KillMethodStrategy.BULLHEADING: bullhead_res,
    }
