"""Casing & Drillstring Structural Mechanical Integrity (API Spec 5CT Standards).

Models burst, collapse, and axial tensile ratings according to API Spec 5CT and
API TR 5C3 (Technical Report on Calculating Casing Performance Properties).

Calculates dynamic load envelopes and safety factors during kicks, gas expansion,
lost circulation, and casing evacuation.

References:
- API Specification 5CT: "Specification for Casing and Tubing," 10th Edition.
- API Technical Report 5C3: "Calculating Performance Properties of Pipe Used as
  Casing or Tubing," 7th Edition.
- Aadnoy, B.S. et al. (2010): "Advanced Drilling Engineering: Principles & Designs."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

__all__ = [
    "CasingGrade",
    "CasingSection",
    "IntegrityCheckPoint",
    "CasingIntegrityReport",
    "calculate_casing_integrity",
]


class CasingGrade:
    J55 = 55000.0   # Yield strength in psi
    K55 = 55000.0
    N80 = 80000.0
    L80 = 80000.0
    C90 = 90000.0
    T95 = 95000.0
    P110 = 110000.0
    Q125 = 125000.0


@dataclass(frozen=True)
class CasingSection:
    """Casing string interval specification.

    Attributes:
        top_md_ft: Top measured depth of casing section (ft).
        bottom_md_ft: Bottom measured depth of casing section (ft).
        od_in: Nominal outer diameter (inches).
        weight_lbf_ft: Nominal weight in air (lbf/ft).
        wall_thickness_in: Nominal steel wall thickness (inches).
        grade_yield_psi: Steel minimum yield strength (psi, e.g. L80 = 80,000 psi).
    """

    top_md_ft: float
    bottom_md_ft: float
    od_in: float
    weight_lbf_ft: float
    wall_thickness_in: float
    grade_yield_psi: float = CasingGrade.L80

    @property
    def id_in(self) -> float:
        """Internal diameter (inches): ID = OD - 2 * t."""
        return self.od_in - 2.0 * self.wall_thickness_in

    @property
    def d_to_t_ratio(self) -> float:
        """Diameter-to-thickness ratio D / t."""
        return self.od_in / max(1e-4, self.wall_thickness_in)

    @property
    def api_burst_pressure_psi(self) -> float:
        """API Spec 5CT Internal Yield Pressure (Burst) with 87.5% wall tolerance:
        P_burst = 0.875 * (2 * Y_p * t / D_o).
        """
        return 0.875 * (2.0 * self.grade_yield_psi * self.wall_thickness_in / self.od_in)

    @property
    def api_collapse_pressure_psi(self) -> float:
        """API TR 5C3 collapse pressure rating considering D/t failure regimes.
        Covers Yield, Plastic, Transition, and Elastic collapse.
        """
        dt = self.d_to_t_ratio
        yp = self.grade_yield_psi

        # Coefficients per API TR 5C3
        a_coeff = 2.8762 + 0.10679e-5 * yp + 0.21301e-10 * (yp**2)
        b_coeff = 0.026233 + 0.50609e-6 * yp
        c_coeff = -465.93 + 0.030867 * yp - 0.10483e-7 * (yp**2)

        # Yield collapse boundary
        dt_yp = (
            math.sqrt((a_coeff - 2.0) ** 2 + 8.0 * (b_coeff + c_coeff / yp)) + (a_coeff - 2.0)
        ) / (2.0 * (b_coeff + c_coeff / yp))

        # Transition and elastic factors
        ba = b_coeff / a_coeff
        term = 3.0 * ba / (2.0 + ba)
        f_coeff = (46.95e6 * (term**3)) / (yp * (term - ba) * ((1.0 - term) ** 2))
        g_coeff = f_coeff * ba

        dt_pt = (yp * (a_coeff - f_coeff)) / (c_coeff + yp * (b_coeff - g_coeff))
        dt_te = (2.0 + ba) / (3.0 * ba)

        if dt <= dt_yp:
            # Yield strength collapse
            return 2.0 * yp * (dt - 1.0) / (dt**2)
        elif dt <= dt_pt:
            # Plastic collapse
            return yp * (a_coeff / dt - b_coeff) - c_coeff
        elif dt <= dt_te:
            # Transition collapse
            return yp * (f_coeff / dt - g_coeff)
        else:
            # Elastic collapse
            return 46.95e6 / (dt * ((dt - 1.0) ** 2))


@dataclass(frozen=True)
class IntegrityCheckPoint:
    """Stress evaluation at a specific depth station."""

    depth_ft: float
    p_internal_psi: float
    p_external_psi: float
    net_burst_load_psi: float
    net_collapse_load_psi: float
    burst_rating_psi: float
    collapse_rating_psi: float
    burst_safety_factor: float
    collapse_safety_factor: float
    is_burst_safe: bool
    is_collapse_safe: bool


@dataclass
class CasingIntegrityReport:
    """Overall casing integrity screening summary."""

    min_burst_sf: float
    min_collapse_sf: float
    critical_burst_depth_ft: float
    critical_collapse_depth_ft: float
    is_structurally_sound: bool
    alarms: List[str]
    checkpoints: List[IntegrityCheckPoint]


def calculate_casing_integrity(
    casing: CasingSection,
    surface_casing_pressure_psi: float,
    internal_mud_ppg: float,
    external_pore_ppg: float,
    gas_void_depth_ft: float = 0.0,
    min_burst_design_sf: float = 1.10,
    min_collapse_design_sf: float = 1.00,
    num_stations: int = 11,
) -> CasingIntegrityReport:
    """Evaluate casing burst and collapse safety factors along the string.

    Args:
        casing: CasingSection geometry and steel grade.
        surface_casing_pressure_psi: Dynamic surface casing pressure during kick (psi).
        internal_mud_ppg: Mud weight inside casing (ppg).
        external_pore_ppg: External formation pore pressure equivalent (ppg).
        gas_void_depth_ft: Depth down to which casing is evacuated/voided with dry gas (for collapse check).
        min_burst_design_sf: Minimum allowable burst design safety factor (default 1.10).
        min_collapse_design_sf: Minimum allowable collapse design safety factor (default 1.00).
        num_stations: Number of depth stations to evaluate.

    Returns:
        CasingIntegrityReport.
    """
    length = casing.bottom_md_ft - casing.top_md_ft
    dz = length / max(1, num_stations - 1)

    burst_rating = casing.api_burst_pressure_psi
    collapse_rating = casing.api_collapse_pressure_psi

    checkpoints: List[IntegrityCheckPoint] = []
    alarms: List[str] = []

    min_burst_sf = 999.0
    min_collapse_sf = 999.0
    crit_burst_z = casing.top_md_ft
    crit_collapse_z = casing.top_md_ft

    for i in range(num_stations):
        z = casing.top_md_ft + i * dz

        # Internal Pressure: surface pressure + internal fluid hydrostatic
        # (Worst case burst occurs at surface where internal casing pressure peaks)
        p_int = surface_casing_pressure_psi + (0.052 * internal_mud_ppg * z)

        # External Pressure: backup pore pressure or seawater/formation hydrostatic
        p_ext = 0.052 * external_pore_ppg * z

        # Net burst load = P_internal - P_external
        net_burst = max(0.0, p_int - p_ext)
        burst_sf = (burst_rating / net_burst) if net_burst > 0.0 else 999.0

        # Collapse Load: worst case occurs if well is voided/evacuated with low-pressure gas
        # inside (P_int = P_gas ~ 14.7 psi or small gas gradient) while full external formation
        # pressure pushes outside:
        if z <= gas_void_depth_ft:
            p_int_collapse = 14.7 + 0.10 * z  # Gas-voided casing
        else:
            p_int_collapse = p_int

        net_collapse = max(0.0, p_ext - p_int_collapse)
        collapse_sf = (collapse_rating / net_collapse) if net_collapse > 0.0 else 999.0

        is_burst_ok = burst_sf >= min_burst_design_sf
        is_coll_ok = collapse_sf >= min_collapse_design_sf

        if burst_sf < min_burst_sf:
            min_burst_sf = burst_sf
            crit_burst_z = z
        if collapse_sf < min_collapse_sf:
            min_collapse_sf = collapse_sf
            crit_collapse_z = z

        checkpoints.append(
            IntegrityCheckPoint(
                depth_ft=round(z, 1),
                p_internal_psi=round(p_int, 1),
                p_external_psi=round(p_ext, 1),
                net_burst_load_psi=round(net_burst, 1),
                net_collapse_load_psi=round(net_collapse, 1),
                burst_rating_psi=round(burst_rating, 1),
                collapse_rating_psi=round(collapse_rating, 1),
                burst_safety_factor=round(burst_sf, 2),
                collapse_safety_factor=round(collapse_sf, 2),
                is_burst_safe=is_burst_ok,
                is_collapse_safe=is_coll_ok,
            )
        )

    # Compile Alarms
    if min_burst_sf < min_burst_design_sf:
        alarms.append(
            f"CASING BURST INTEGRITY BREACH: SF_burst = {min_burst_sf:.2f} < {min_burst_design_sf:.2f} "
            f"at depth {crit_burst_z:.0f} ft. Surface pressure ({surface_casing_pressure_psi:.0f} psi) "
            f"exceeds casing yield threshold!"
        )
    if min_collapse_sf < min_collapse_design_sf:
        alarms.append(
            f"CASING COLLAPSE INTEGRITY BREACH: SF_collapse = {min_collapse_sf:.2f} < {min_collapse_design_sf:.2f} "
            f"at depth {crit_collapse_z:.0f} ft. External formation pressure exceeds casing collapse resistance!"
        )

    is_sound = (min_burst_sf >= min_burst_design_sf) and (min_collapse_sf >= min_collapse_design_sf)

    return CasingIntegrityReport(
        min_burst_sf=round(min_burst_sf, 2),
        min_collapse_sf=round(min_collapse_sf, 2),
        critical_burst_depth_ft=crit_burst_z,
        critical_collapse_depth_ft=crit_collapse_z,
        is_structurally_sound=is_sound,
        alarms=alarms,
        checkpoints=checkpoints,
    )
