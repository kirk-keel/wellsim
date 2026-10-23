"""Dynamic Surge and Swab Tripping Engine.

Models transient pressure surges during running in hole (RIH) and swab suction
pressures during pulling out of hole (POOH) according to Burkhardt (1961),
Mitchell (1988), and API RP 13D standards.

Calculates safe tripping velocity envelopes to prevent formation fracturing
(losses / underground blowouts) and swab-induced reservoir kicks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from wellsim.core.friction import bingham_pressure_gradient
from wellsim.core.rheology import HerschelBulkleyParams, herschel_bulkley_pressure_gradient
from wellsim.core.well_control import hydrostatic_pressure

__all__ = [
    "PipeCondition",
    "SurgeSwabResult",
    "calculate_surge_swab_gradient",
    "calculate_trip_envelope",
]


class PipeCondition:
    CLOSED_ENDED = "closed"  # Float valve closed; 100% displacement to annulus
    OPEN_ENDED = "open"  # Float valve open / fill-up line; mud flows through bore


@dataclass(frozen=True)
class SurgeSwabResult:
    """Surge / Swab calculation summary at a given depth.

    Attributes:
        depth_ft: Measured / vertical depth (ft).
        trip_speed_ft_min: Tripping velocity (ft/min).
        surge_pressure_psi: Dynamic pressure surge during RIH (psi).
        swab_pressure_psi: Dynamic swab pressure reduction during POOH (psi).
        ecd_surge_ppg: Equivalent circulating density during RIH (ppg).
        ecd_swab_ppg: Equivalent density during POOH (ppg).
        margin_to_frac_psi: Remaining safety margin to fracture gradient during surge (psi).
        margin_to_pore_psi: Remaining safety margin to pore pressure during swab (psi).
    """

    depth_ft: float
    trip_speed_ft_min: float
    surge_pressure_psi: float
    swab_pressure_psi: float
    ecd_surge_ppg: float
    ecd_swab_ppg: float
    margin_to_frac_psi: float
    margin_to_pore_psi: float


def calculate_surge_swab_gradient(
    trip_speed_ft_min: float,
    hole_id_in: float,
    pipe_od_in: float,
    pipe_id_in: float,
    rho_ppg: float,
    pv_cp: float,
    yp_lbf_100ft2: float,
    hb_params: HerschelBulkleyParams | None = None,
    pipe_condition: str = PipeCondition.CLOSED_ENDED,
    cling_factor: float = 0.42,
) -> float:
    """Calculate the frictional surge/swab pressure gradient (psi/ft) induced by pipe motion.

    Args:
        trip_speed_ft_min: Absolute velocity of pipe movement (ft/min).
        hole_id_in: Casing or borehole inner diameter (inches).
        pipe_od_in: Drillpipe / casing outer diameter (inches).
        pipe_id_in: Drillpipe inner diameter (inches).
        rho_ppg: Mud density (ppg).
        pv_cp: Plastic Viscosity (cp).
        yp_lbf_100ft2: Yield Point (lbf/100 ft^2).
        hb_params: Optional Herschel-Bulkley rheology parameters.
        pipe_condition: 'closed' (float valve closed) or 'open' (open-ended).
        cling_factor: Burkhardt mud cling factor (typically 0.40 to 0.45).

    Returns:
        Frictional pressure gradient in psi/ft.
    """
    if trip_speed_ft_min <= 0.0 or hole_id_in <= pipe_od_in:
        return 0.0

    v_pipe_fps = trip_speed_ft_min / 60.0
    d_hyd_in = hole_id_in - pipe_od_in

    # Annular displacement area ratio
    area_annulus = (hole_id_in**2) - (pipe_od_in**2)
    if area_annulus <= 0.0:
        return 0.0

    if pipe_condition == PipeCondition.CLOSED_ENDED:
        # Closed end: entire pipe volume displaces annular fluid
        displacement_ratio = (pipe_od_in**2) / area_annulus
    else:
        # Open end: only the steel wall thickness displaces fluid
        area_steel = (pipe_od_in**2) - (pipe_id_in**2)
        area_total_flow = area_annulus + (pipe_id_in**2)
        displacement_ratio = area_steel / max(1e-4, area_total_flow)

    # Effective annular fluid velocity = displacement velocity + clinging fluid drag
    v_eff_fps = v_pipe_fps * (displacement_ratio + cling_factor)

    # Calculate frictional gradient using Herschel-Bulkley or Bingham Plastic
    if hb_params is not None:
        return herschel_bulkley_pressure_gradient(
            rho_ppg=rho_ppg,
            v_fps=v_eff_fps,
            d_in=d_hyd_in,
            params=hb_params,
            is_annulus=True,
        )
    else:
        return bingham_pressure_gradient(
            rho_ppg=rho_ppg,
            v_fps=v_eff_fps,
            d_in=d_hyd_in,
            pv_cp=pv_cp,
            yp_lbf_100ft2=yp_lbf_100ft2,
            is_annulus=True,
        )


def calculate_trip_envelope(
    tvd_well_ft: float,
    shoe_tvd_ft: float,
    hole_id_in: float,
    pipe_od_in: float,
    pipe_id_in: float,
    rho_ppg: float,
    pv_cp: float,
    yp_lbf_100ft2: float,
    pore_pressure_ppg: float,
    frac_gradient_ppg: float,
    hb_params: HerschelBulkleyParams | None = None,
    pipe_condition: str = PipeCondition.CLOSED_ENDED,
    safety_margin_psi: float = 100.0,
) -> Tuple[float, float, List[SurgeSwabResult]]:
    """Determine the maximum safe tripping speed envelope across depths.

    Args:
        tvd_well_ft: Total vertical depth of bottomhole (ft).
        shoe_tvd_ft: Casing shoe vertical depth (ft), critical for fracture.
        hole_id_in: Casing / borehole ID (inches).
        pipe_od_in: Pipe OD (inches).
        pipe_id_in: Pipe ID (inches).
        rho_ppg: Mud weight (ppg).
        pv_cp: Plastic Viscosity (cp).
        yp_lbf_100ft2: Yield Point (lbf/100 ft^2).
        pore_pressure_ppg: Formation pore pressure equivalent (ppg).
        frac_gradient_ppg: Formation fracture gradient equivalent (ppg).
        hb_params: Optional Herschel-Bulkley parameters.
        pipe_condition: 'closed' or 'open'.
        safety_margin_psi: Buffer above pore and below fracture (psi).

    Returns:
        Tuple of:
            - max_safe_rih_speed_ft_min: Maximum allowable running-in speed without breaking shoe.
            - max_safe_pooh_speed_ft_min: Maximum allowable pulling-out speed without inducing kick.
            - depth_profile: List of SurgeSwabResult at standard tripping speeds (90 ft/min = 1 stand/min).
    """
    # Allowable pressure surge at shoe
    p_hyd_shoe = hydrostatic_pressure(rho_ppg, shoe_tvd_ft)
    p_frac_shoe = hydrostatic_pressure(frac_gradient_ppg, shoe_tvd_ft)
    max_allowable_surge_psi = max(10.0, p_frac_shoe - p_hyd_shoe - safety_margin_psi)

    # Allowable pressure swab at bottomhole
    p_hyd_bottom = hydrostatic_pressure(rho_ppg, tvd_well_ft)
    p_pore_bottom = hydrostatic_pressure(pore_pressure_ppg, tvd_well_ft)
    max_allowable_swab_psi = max(10.0, p_hyd_bottom - p_pore_bottom - safety_margin_psi)

    # Brent or bisection search for max safe RIH speed (causing surge <= max_allowable_surge)
    def surge_at_speed(speed_ft_min: float) -> float:
        grad = calculate_surge_swab_gradient(
            speed_ft_min,
            hole_id_in,
            pipe_od_in,
            pipe_id_in,
            rho_ppg,
            pv_cp,
            yp_lbf_100ft2,
            hb_params,
            pipe_condition,
        )
        return grad * shoe_tvd_ft

    # Search RIH speed bracket [10, 600] ft/min
    low_rih, high_rih = 10.0, 600.0
    for _ in range(25):
        mid = 0.5 * (low_rih + high_rih)
        if surge_at_speed(mid) > max_allowable_surge_psi:
            high_rih = mid
        else:
            low_rih = mid
    max_rih_speed = low_rih

    # Search POOH speed bracket [10, 600] ft/min
    def swab_at_speed(speed_ft_min: float) -> float:
        grad = calculate_surge_swab_gradient(
            speed_ft_min,
            hole_id_in,
            pipe_od_in,
            pipe_id_in,
            rho_ppg,
            pv_cp,
            yp_lbf_100ft2,
            hb_params,
            pipe_condition,
        )
        return grad * tvd_well_ft

    low_pooh, high_pooh = 10.0, 600.0
    for _ in range(25):
        mid = 0.5 * (low_pooh + high_pooh)
        if swab_at_speed(mid) > max_allowable_swab_psi:
            high_pooh = mid
        else:
            low_pooh = mid
    max_pooh_speed = low_pooh

    # Generate profile at standard tripping speed (90 ft/min, typical 1 stand/min)
    nominal_speed = 90.0
    depth_stations = [shoe_tvd_ft * 0.5, shoe_tvd_ft, tvd_well_ft * 0.8, tvd_well_ft]
    profile: List[SurgeSwabResult] = []

    grad_nom = calculate_surge_swab_gradient(
        nominal_speed,
        hole_id_in,
        pipe_od_in,
        pipe_id_in,
        rho_ppg,
        pv_cp,
        yp_lbf_100ft2,
        hb_params,
        pipe_condition,
    )

    for depth in depth_stations:
        surge_psi = grad_nom * depth
        swab_psi = grad_nom * depth

        p_hyd = hydrostatic_pressure(rho_ppg, depth)
        p_frac = hydrostatic_pressure(frac_gradient_ppg, depth)
        p_pore = hydrostatic_pressure(pore_pressure_ppg, depth)

        ecd_surge = rho_ppg + (surge_psi / (0.052 * max(1.0, depth)))
        ecd_swab = rho_ppg - (swab_psi / (0.052 * max(1.0, depth)))

        margin_frac = p_frac - (p_hyd + surge_psi)
        margin_pore = (p_hyd - swab_psi) - p_pore

        profile.append(
            SurgeSwabResult(
                depth_ft=depth,
                trip_speed_ft_min=nominal_speed,
                surge_pressure_psi=surge_psi,
                swab_pressure_psi=swab_psi,
                ecd_surge_ppg=ecd_surge,
                ecd_swab_ppg=ecd_swab,
                margin_to_frac_psi=margin_frac,
                margin_to_pore_psi=margin_pore,
            )
        )

    return max_rih_speed, max_pooh_speed, profile
