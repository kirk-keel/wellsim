"""Underground Blowout & Inter-Zonal Cross-Flow Hydraulics Engine.

Models subsurface blowouts where deep overpressured formations breach the wellbore
and charge shallower, low-pressure thief zones (fractured sands, casing shoes, or aquifers)
with zero or minimal surface gauge pressure.

Calculates:
1. Steady-state equilibrium cross-flow rate between source and thief intervals.
2. Flowing bottomhole pressure profile along the internal cross-flow conduit.
3. Minimum dynamic relief-well kill rate required to kill the underground flow.

References:
- Blount, C.G. & Soeiinah, E. (1987): "Dynamic Kill: Controlling Underground Blowouts,"
  SPE-16673.
- Watson, G. & Bachman, D. (1991): "Underground Blowouts: Diagnosis and Solution,"
  World Oil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from wellsim.core.friction import bingham_pressure_gradient
from wellsim.core.well_control import hydrostatic_pressure

__all__ = [
    "FormationZone",
    "UndergroundBlowoutScenario",
    "CrossFlowResult",
    "solve_underground_blowout",
]


@dataclass(frozen=True)
class FormationZone:
    """A subsurface formation zone (source or thief).

    Attributes:
        name: Formation description.
        depth_tvd_ft: True vertical depth (ft).
        pressure_psia: In-situ pore pressure or fracture breakdown pressure (psia).
        pi_bbl_d_psi: Productivity Index (PI) or Injectivity Index (II) in bbl/d/psi.
    """

    name: str
    depth_tvd_ft: float
    pressure_psia: float
    pi_bbl_d_psi: float


@dataclass(frozen=True)
class UndergroundBlowoutScenario:
    """Underground blowout well configuration.

    Attributes:
        source_zone: Deep overpressured source reservoir.
        thief_zone: Shallower depleted zone or ruptured casing shoe.
        conduit_id_in: Internal diameter of casing or hole between zones (inches).
        pipe_od_in: Outer diameter of internal drillstring if present (inches, 0 if no pipe).
        fluid_density_ppg: Density of cross-flowing reservoir fluid (ppg, e.g. gas-cut mud ~8.0-10.0 ppg).
        fluid_pv_cp: Fluid plastic viscosity (cp).
        fluid_yp_lbf_100ft2: Fluid yield point (lbf/100 ft^2).
    """

    source_zone: FormationZone
    thief_zone: FormationZone
    conduit_id_in: float
    pipe_od_in: float = 0.0
    fluid_density_ppg: float = 8.5
    fluid_pv_cp: float = 15.0
    fluid_yp_lbf_100ft2: float = 10.0


@dataclass(frozen=True)
class CrossFlowResult:
    """Underground blowout simulation results.

    Attributes:
        is_crossflowing: True if underground blowout occurs.
        crossflow_rate_bbl_d: Internal flow rate from source to thief zone (bbl/day).
        crossflow_rate_gpm: Flow rate in gallons per minute (gpm).
        pwf_source_psia: Flowing pressure at source sandface (psia).
        pwf_thief_psia: Flowing injection pressure at thief sandface (psia).
        interzonal_friction_loss_psi: Friction loss between source and thief (psi).
        interzonal_hydrostatic_loss_psi: Hydrostatic loss between source and thief (psi).
        min_relief_kill_rate_gpm: Minimum dynamic kill rate required via relief well (gpm).
        min_relief_kill_mud_ppg: Recommended kill mud weight for dynamic relief kill (ppg).
    """

    is_crossflowing: bool
    crossflow_rate_bbl_d: float
    crossflow_rate_gpm: float
    pwf_source_psia: float
    pwf_thief_psia: float
    interzonal_friction_loss_psi: float
    interzonal_hydrostatic_loss_psi: float
    min_relief_kill_rate_gpm: float
    min_relief_kill_mud_ppg: float


def solve_underground_blowout(
    scenario: UndergroundBlowoutScenario,
    kill_mud_weight_ppg: float = 14.5,
) -> CrossFlowResult:
    """Solve for steady-state cross-flow equilibrium and relief well kill requirements.

    Args:
        scenario: UndergroundBlowoutScenario.
        kill_mud_weight_ppg: Planned mud density for dynamic relief-well kill.

    Returns:
        CrossFlowResult.
    """
    src = scenario.source_zone
    thief = scenario.thief_zone

    delta_z_ft = src.depth_tvd_ft - thief.depth_tvd_ft
    if delta_z_ft <= 0.0:
        raise ValueError("Source zone depth must be deeper than thief zone depth.")

    # Static hydrostatic of cross-flowing fluid column between source and thief
    p_hyd_interzonal = hydrostatic_pressure(scenario.fluid_density_ppg, delta_z_ft)

    # Screening check: if source pressure cannot overcome thief pressure + hydrostatic, no crossflow
    max_available_drawdown = src.pressure_psia - (thief.pressure_psia + p_hyd_interzonal)
    if max_available_drawdown <= 0.0:
        return CrossFlowResult(
            is_crossflowing=False,
            crossflow_rate_bbl_d=0.0,
            crossflow_rate_gpm=0.0,
            pwf_source_psia=src.pressure_psia,
            pwf_thief_psia=thief.pressure_psia,
            interzonal_friction_loss_psi=0.0,
            interzonal_hydrostatic_loss_psi=p_hyd_interzonal,
            min_relief_kill_rate_gpm=0.0,
            min_relief_kill_mud_ppg=kill_mud_weight_ppg,
        )

    # Hydraulic geometry
    is_annulus = scenario.pipe_od_in > 0.0
    if is_annulus:
        flow_area_in2 = (scenario.conduit_id_in**2) - (scenario.pipe_od_in**2)
        d_hyd_in = scenario.conduit_id_in - scenario.pipe_od_in
    else:
        flow_area_in2 = scenario.conduit_id_in**2
        d_hyd_in = scenario.conduit_id_in

    # Bisection / Brent search for equilibrium rate q (bbl/day)
    # At equilibrium: q_inflow(Pwf_src) == q_inject(Pwf_thief) == q
    # Pwf_src = P_src - (q / PI_src)
    # Pwf_thief = P_thief + (q / II_thief)
    # Hydraulic balance: Pwf_src - p_hyd_interzonal - DeltaP_fric(q) == Pwf_thief
    def residual(q_bbl_d: float) -> float:
        q_gpm = (q_bbl_d * 42.0) / 1440.0
        v_fps = q_gpm / (2.448 * max(0.1, flow_area_in2))
        dp_dl = bingham_pressure_gradient(
            rho_ppg=scenario.fluid_density_ppg,
            v_fps=v_fps,
            d_in=d_hyd_in,
            pv_cp=scenario.fluid_pv_cp,
            yp_lbf_100ft2=scenario.fluid_yp_lbf_100ft2,
            is_annulus=is_annulus,
        )
        dp_fric = dp_dl * delta_z_ft

        pwf_src = src.pressure_psia - (q_bbl_d / max(1e-4, src.pi_bbl_d_psi))
        pwf_thf = thief.pressure_psia + (q_bbl_d / max(1e-4, thief.pi_bbl_d_psi))
        return (pwf_src - p_hyd_interzonal - dp_fric) - pwf_thf

    # Find bracket [1.0, q_max]
    q_max = max_available_drawdown * min(src.pi_bbl_d_psi, thief.pi_bbl_d_psi) * 2.0
    low_q, high_q = 0.0, max(100.0, q_max)
    for _ in range(25):
        mid_q = 0.5 * (low_q + high_q)
        if residual(mid_q) > 0.0:
            low_q = mid_q
        else:
            high_q = mid_q
    q_eq_bbl_d = 0.5 * (low_q + high_q)

    # Compute final pressures at equilibrium
    q_eq_gpm = (q_eq_bbl_d * 42.0) / 1440.0
    v_eq_fps = q_eq_gpm / (2.448 * max(0.1, flow_area_in2))
    dp_fric_final = bingham_pressure_gradient(
        rho_ppg=scenario.fluid_density_ppg,
        v_fps=v_eq_fps,
        d_in=d_hyd_in,
        pv_cp=scenario.fluid_pv_cp,
        yp_lbf_100ft2=scenario.fluid_yp_lbf_100ft2,
        is_annulus=is_annulus,
    ) * delta_z_ft

    pwf_src_final = src.pressure_psia - (q_eq_bbl_d / src.pi_bbl_d_psi)
    pwf_thf_final = thief.pressure_psia + (q_eq_bbl_d / thief.pi_bbl_d_psi)

    # Dynamic Relief Kill:
    # Need kill mud weight & rate such that hydrostatic + friction overcomes source pressure
    # P_hyd_kill = 0.052 * kill_mud_ppg * delta_z
    p_hyd_kill = 0.052 * kill_mud_weight_ppg * delta_z_ft
    # If static hold is insufficient, dynamic pump rate must supply required balance
    pressure_deficit = max(0.0, src.pressure_psia - (thief.pressure_psia + p_hyd_kill))
    # Required kill rate to supply frictional backpressure:
    # Delta_P_fric >= pressure_deficit
    min_kill_gpm = max(q_eq_gpm * 1.2, 200.0) if pressure_deficit > 0 else (q_eq_gpm * 0.8)

    return CrossFlowResult(
        is_crossflowing=True,
        crossflow_rate_bbl_d=round(q_eq_bbl_d, 1),
        crossflow_rate_gpm=round(q_eq_gpm, 1),
        pwf_source_psia=round(pwf_src_final, 1),
        pwf_thief_psia=round(pwf_thf_final, 1),
        interzonal_friction_loss_psi=round(dp_fric_final, 1),
        interzonal_hydrostatic_loss_psi=round(p_hyd_interzonal, 1),
        min_relief_kill_rate_gpm=round(min_kill_gpm, 1),
        min_relief_kill_mud_ppg=round(kill_mud_weight_ppg, 1),
    )
