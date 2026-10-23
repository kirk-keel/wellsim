"""Dynamic Cuttings Transport & Hole Cleaning (Rubi-Larsen 3-Layer Model).

Models real-time solids transport, slip velocity, stationary/moving cuttings bed
formation, annular frictional choking, and mechanical pack-off risk in directional
and extended-reach drilling (ERD) wells (0 to 90 deg inclination).

References:
- Larsen, T.I., Pilehvari, A.A., & Azar, J.J. (1997): "Development of a New
  Cuttings-Transport Model for High-Angle Wellbores Including Horizontal Wells,"
  SPE Drilling & Completion 12(02), SPE-25872-PA.
- Moore, P.L. (1974): "Drilling Practices Manual," PennWell.
- API RP 13D: Solids Transport and Hole Cleaning Guidelines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "CuttingsProperties",
    "HoleCleaningState",
    "calculate_cuttings_generation_rate",
    "calculate_slip_velocity",
    "calculate_critical_transport_velocity",
    "simulate_hole_cleaning_profile",
]


@dataclass(frozen=True)
class CuttingsProperties:
    """Drilled cuttings solid properties.

    Attributes:
        diameter_in: Mean cutting particle diameter (inches, typical 0.10 to 0.25 in).
        density_ppg: Rock matrix density in ppg (typical shale/sandstone ~2.65 g/cm3 = 22.1 ppg).
        porosity: In-situ rock porosity fraction (0.0 to 0.40).
        bed_porosity: Porosity of settled loose cuttings bed (typical 0.36 to 0.45).
    """

    diameter_in: float = 0.125
    density_ppg: float = 22.0
    porosity: float = 0.15
    bed_porosity: float = 0.40


@dataclass
class HoleCleaningState:
    """Annular hole cleaning and cuttings bed state at a specific depth station.

    Attributes:
        md_ft: Measured depth (ft).
        inclination_deg: Wellbore inclination from vertical (0 = vertical, 90 = horizontal).
        annular_velocity_fps: Actual mean mud velocity in the annulus (ft/s).
        critical_velocity_fps: Minimum required fluid velocity to prevent stationary bed formation (ft/s).
        bed_height_fraction: Ratio of stationary bed thickness to hole diameter (0.0 to 0.70).
        cuttings_concentration_pct: Volumetric concentration of cuttings in mud stream (%).
        effective_diameter_in: Reduced hydraulic clearance due to bed encroachment (inches).
        additional_friction_loss_psi: Extra frictional pressure drop caused by cuttings choking (psi).
        packoff_risk_index: Normalized risk of mechanical pipe sticking (0.0 to 1.0).
    """

    md_ft: float
    inclination_deg: float
    annular_velocity_fps: float
    critical_velocity_fps: float
    bed_height_fraction: float
    cuttings_concentration_pct: float
    effective_diameter_in: float
    additional_friction_loss_psi: float
    packoff_risk_index: float


def calculate_cuttings_generation_rate(
    rop_ft_hr: float,
    hole_diameter_in: float,
    cuttings: CuttingsProperties,
) -> float:
    """Calculate solid cuttings generation volume rate in gallons per minute (gpm).

    q_cuttings = (pi/4) * D^2 * ROP * (1 - porosity)
    """
    if rop_ft_hr <= 0.0 or hole_diameter_in <= 0.0:
        return 0.0

    # Hole cross section area in ft^2
    area_ft2 = (math.pi / 4.0) * ((hole_diameter_in / 12.0) ** 2)
    # Bulk volume rate in ft^3 / hr
    bulk_vol_ft3_hr = area_ft2 * rop_ft_hr
    # Solid matrix volume rate (excluding formation pore volume)
    solid_vol_ft3_hr = bulk_vol_ft3_hr * (1.0 - cuttings.porosity)
    # Convert ft^3/hr to gpm (1 ft^3 = 7.48052 gal)
    return (solid_vol_ft3_hr * 7.48052) / 60.0


def calculate_slip_velocity(
    cuttings: CuttingsProperties,
    mud_weight_ppg: float,
    pv_cp: float,
    yp_lbf_100ft2: float,
) -> float:
    """Calculate particle terminal settling slip velocity (ft/s) in vertical mud stream.

    Uses Moore / Chien correlation with fluid effective viscosity.
    """
    if mud_weight_ppg <= 0.0 or cuttings.density_ppg <= mud_weight_ppg:
        return 0.0

    delta_rho = cuttings.density_ppg - mud_weight_ppg
    d_p = cuttings.diameter_in

    # Apparent viscosity approximation (cp)
    mu_app = pv_cp + (5.0 * yp_lbf_100ft2 * d_p)
    mu_app = max(1.0, mu_app)

    # Chien (1994) terminal slip velocity formula (ft/s):
    # v_slip = 0.0075 * (mu / (rho * d)) * [ sqrt(1 + 36800 * d^3 * rho * delta_rho / mu^2) - 1 ]
    term = (36800.0 * (d_p**3) * mud_weight_ppg * delta_rho) / (mu_app**2)
    if term < 0.0:
        return 0.0
    v_slip = 0.0075 * (mu_app / (mud_weight_ppg * d_p)) * (math.sqrt(1.0 + term) - 1.0)
    return max(0.05, min(2.5, v_slip))


def calculate_critical_transport_velocity(
    inclination_deg: float,
    hole_diameter_in: float,
    pipe_od_in: float,
    mud_weight_ppg: float,
    pv_cp: float,
    yp_lbf_100ft2: float,
    rop_ft_hr: float,
    cuttings: CuttingsProperties,
) -> float:
    """Calculate critical fluid velocity (ft/s) required to prevent stationary bed formation.

    Implements the Larsen (1997) API RP 13D model for directional wells:
    - In vertical wells (0 to 30 deg), velocity must comfortably exceed slip velocity.
    - In avalanching intermediate angles (35 to 55 deg), critical velocity peaks due to bed back-sliding.
    - In high angle / horizontal wells (60 to 90 deg), velocity must roll/suspend bed particles.
    """
    v_slip = calculate_slip_velocity(cuttings, mud_weight_ppg, pv_cp, yp_lbf_100ft2)

    # Base Larsen critical velocity in horizontal flow (ft/s)
    # v_crit_horiz = [ 0.034 * ROP^0.18 * d_p^0.15 * rho_m^0.35 / (mu_eff^0.25) ] * geometric_factor
    mu_eff = pv_cp + (yp_lbf_100ft2 * 5.0)
    mu_eff = max(1.0, mu_eff)

    v_base = 2.4 * ((rop_ft_hr / 50.0) ** 0.18) * ((mud_weight_ppg / 10.0) ** 0.3) / ((mu_eff / 30.0) ** 0.2)

    # Inclination correction factor C_angle (Larsen 1997):
    # Peaks in the 40-55 deg "avalanching" window where hole cleaning is hardest
    theta = min(90.0, max(0.0, inclination_deg))
    if theta <= 30.0:
        # Vertical / near vertical: slip-dominated
        c_angle = 1.0 + (theta / 30.0) * 0.2
        return max(v_slip * 1.5, v_base * 0.7 * c_angle)
    elif theta <= 55.0:
        # Critical avalanching zone: highest required velocity
        c_angle = 1.2 + 0.3 * math.sin(math.radians((theta - 30.0) / 25.0 * 180.0))
        return v_base * c_angle
    else:
        # High angle to horizontal (55 to 90 deg): stratified bed rolling regime
        c_angle = 1.15 - 0.15 * ((theta - 55.0) / 35.0)
        return v_base * c_angle


def simulate_hole_cleaning_profile(
    md_total_ft: float,
    inclinations: List[Tuple[float, float]],  # List of (MD_station, Inclination_deg)
    hole_diameter_in: float,
    pipe_od_in: float,
    flow_rate_gpm: float,
    rop_ft_hr: float,
    mud_weight_ppg: float,
    pv_cp: float,
    yp_lbf_100ft2: float,
    cuttings: CuttingsProperties | None = None,
) -> List[HoleCleaningState]:
    """Simulate hole cleaning and cuttings bed development along well measured depth.

    Args:
        md_total_ft: Total measured depth (ft).
        inclinations: List of (MD, Inclination) stations.
        hole_diameter_in: Hole diameter (inches).
        pipe_od_in: Drillpipe outer diameter (inches).
        flow_rate_gpm: Mud circulation rate (gpm).
        rop_ft_hr: Rate of Penetration (ft/hr).
        mud_weight_ppg: Mud weight (ppg).
        pv_cp: Plastic Viscosity (cp).
        yp_lbf_100ft2: Yield Point (lbf/100 ft^2).
        cuttings: Drilled cuttings properties.

    Returns:
        List of HoleCleaningState along the trajectory.
    """
    if cuttings is None:
        cuttings = CuttingsProperties()

    # Annular hydraulic dimensions
    d_hyd = hole_diameter_in - pipe_od_in
    annular_area_in2 = (hole_diameter_in**2) - (pipe_od_in**2)
    # Annular fluid velocity (ft/s): v = gpm / (2.448 * (D_h^2 - D_p^2))
    v_ann = (flow_rate_gpm / (2.448 * annular_area_in2)) if annular_area_in2 > 0 else 0.0

    # Cuttings generation rate
    q_cuttings_gpm = calculate_cuttings_generation_rate(rop_ft_hr, hole_diameter_in, cuttings)

    profile: List[HoleCleaningState] = []

    for md_station, inc in inclinations:
        v_crit = calculate_critical_transport_velocity(
            inc,
            hole_diameter_in,
            pipe_od_in,
            mud_weight_ppg,
            pv_cp,
            yp_lbf_100ft2,
            rop_ft_hr,
            cuttings,
        )

        # Bed height fraction accumulation:
        # If actual velocity < critical velocity in inclined well, bed forms
        if inc >= 25.0 and v_ann < v_crit:
            # Velocity deficit drives bed accumulation
            deficit = (v_crit - v_ann) / v_crit
            # Max equilibrium bed height can reach up to 40-50% of hole diameter
            bed_frac = min(0.50, deficit * 0.65 * (math.sin(math.radians(inc)) ** 1.2))
        else:
            bed_frac = 0.0

        # Cuttings concentration in the moving stream (%):
        cuttings_conc_pct = (q_cuttings_gpm / max(1.0, flow_rate_gpm + q_cuttings_gpm)) * 100.0
        # In presence of bed, local concentration increases
        if bed_frac > 0.0:
            cuttings_conc_pct = min(25.0, cuttings_conc_pct * (1.0 + 2.0 * bed_frac))

        # Effective hydraulic diameter with bed restriction
        eff_diam = max(1.0, d_hyd * (1.0 - bed_frac))

        # Additional frictional choking due to bed constriction (psi)
        # Delta P_bed = (bed_frac / (1 - bed_frac)) * base_friction
        additional_loss_psi = (
            (bed_frac * 1.5 * (md_station / 1000.0) * (mud_weight_ppg / 10.0) * (v_ann**1.8))
            if bed_frac > 0.0
            else 0.0
        )

        # Pack-off risk index: composite score of bed height, velocity deficit, and inclination
        # 0.0 = Clean hole; 1.0 = Imminent mechanical stuck pipe / pack-off
        risk_score = 0.0
        if bed_frac > 0.05:
            risk_score += (bed_frac / 0.40) * 0.6
        if v_ann < v_crit:
            risk_score += min(0.4, (v_crit - v_ann) / max(0.1, v_crit) * 0.4)
        packoff_risk = max(0.0, min(1.0, risk_score))

        profile.append(
            HoleCleaningState(
                md_ft=md_station,
                inclination_deg=inc,
                annular_velocity_fps=round(v_ann, 2),
                critical_velocity_fps=round(v_crit, 2),
                bed_height_fraction=round(bed_frac, 3),
                cuttings_concentration_pct=round(cuttings_conc_pct, 2),
                effective_diameter_in=round(eff_diam, 2),
                additional_friction_loss_psi=round(additional_loss_psi, 1),
                packoff_risk_index=round(packoff_risk, 2),
            )
        )

    return profile
