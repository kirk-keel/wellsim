"""3D Torque & Drag and Drillstring Buckling Limits Engine (Soft-String Model).

Implements the industry standard Johancsik (1984) / Dawson & Paslay (1984) /
Wu & Juvkam-Wold (1995) models for:
1. 3D Hookload predictions for Tripping Out (POOH), Tripping In (RIH), and Rotating.
2. Surface and downhole torque profiles along the 3D trajectory.
3. Critical sinusoidal and helical buckling thresholds.
4. Drillstring lock-up depth identification in extended-reach drilling (ERD).

References:
- Johancsik, C.A., Friesen, D.B., & Dawson, R. (1984): "Torque and Drag in
  Directional Wells - Prediction and Measurement," JPT 36(6), SPE-11380-PA.
- Dawson, R. & Paslay, P.R. (1984): "Drillpipe Buckling in Inclined Holes,"
  SPE 11167.
- Wu, J. & Juvkam-Wold, H.C. (1995): "Coiled Tubing Buckling in Inclined and
  Horizontal Wells," SPE 27666.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from wellsim.core.survey import SurveyStation, WellTrajectory, calculate_minimum_curvature

__all__ = [
    "PipeSection",
    "TorqueDragStation",
    "TorqueDragResult",
    "calculate_torque_and_drag",
]

E_STEEL_PSI = 30.0e6  # Young's modulus for steel (psi)
STEEL_DENSITY_PPG = 65.5  # Steel density in ppg


@dataclass(frozen=True)
class PipeSection:
    """Drillstring component (drillpipe, heavy weight, drill collar).

    Attributes:
        length_ft: Total length of this section (ft).
        od_in: Outer diameter (inches).
        id_in: Inner diameter (inches).
        weight_air_lbf_ft: Nominal linear weight in air (lbf/ft).
    """

    length_ft: float
    od_in: float
    id_in: float
    weight_air_lbf_ft: float

    @property
    def moment_of_inertia_in4(self) -> float:
        """Area moment of inertia I = (pi/64) * (OD^4 - ID^4) in in^4."""
        return (math.pi / 64.0) * ((self.od_in**4) - (self.id_in**4))


@dataclass(frozen=True)
class TorqueDragStation:
    """Calculated torque, drag, and buckling state at a specific depth station."""

    md_ft: float
    tvd_ft: float
    inclination_deg: float
    tension_pooh_lbf: float
    tension_rih_lbf: float
    tension_rotating_lbf: float
    torque_ft_lbf: float
    normal_force_lbf: float
    sinusoidal_buckling_limit_lbf: float
    helical_buckling_limit_lbf: float
    is_buckled: bool
    is_locked_up: bool


@dataclass
class TorqueDragResult:
    """Complete torque and drag analysis for the wellbore."""

    hookload_pooh_klbf: float
    hookload_rih_klbf: float
    hookload_rotating_klbf: float
    surface_torque_ft_lbf: float
    lockup_depth_ft: Optional[float]
    stations: List[TorqueDragStation]


def calculate_torque_and_drag(
    trajectory: WellTrajectory,
    drillstring: List[PipeSection],
    hole_diameter_in: float,
    mud_weight_ppg: float,
    friction_coefficient: float = 0.25,
    wob_lbf: float = 0.0,
    bit_torque_ft_lbf: float = 0.0,
) -> TorqueDragResult:
    """Calculate 3D torque, drag, hookload, and buckling limits (Johancsik soft-string model).

    Args:
        trajectory: Calculated WellTrajectory (3D stations).
        drillstring: List of PipeSection from bit to surface.
        hole_diameter_in: Casing / borehole ID (inches).
        mud_weight_ppg: Mud weight (ppg).
        friction_coefficient: Cased/open hole friction factor mu (typical 0.20 to 0.35).
        wob_lbf: Weight on Bit during drilling (lbf).
        bit_torque_ft_lbf: Reaction torque at the bit (ft-lbf).

    Returns:
        TorqueDragResult containing hookloads, torque, and station-by-station profile.
    """
    if not trajectory.stations:
        return TorqueDragResult(0.0, 0.0, 0.0, 0.0, None, [])

    # Buoyancy factor BF = 1 - (rho_mud / 65.5)
    bf = max(0.2, 1.0 - (mud_weight_ppg / STEEL_DENSITY_PPG))

    # Build pipe property lookup by measured depth from surface
    total_md = trajectory.total_md_ft
    # Default fallback pipe
    default_pipe = drillstring[0] if drillstring else PipeSection(total_md, 5.0, 4.276, 19.5)

    def get_pipe_at_depth(depth_from_surface_ft: float) -> PipeSection:
        curr = 0.0
        for sec in drillstring:
            curr += sec.length_ft
            if depth_from_surface_ft <= curr:
                return sec
        return drillstring[-1] if drillstring else default_pipe

    # March from bit (bottom) up to surface
    # Boundary conditions at bit (TD):
    # POOH: tension = 0 (no overpull at bit)
    # RIH: compression = WOB
    # Rotating: tension = -WOB, torque = bit_torque
    calc_stations = list(reversed(trajectory.stations))

    t_pooh = 0.0
    t_rih = -float(wob_lbf)
    t_rot = -float(wob_lbf)
    torque = float(bit_torque_ft_lbf)

    results_rev: List[TorqueDragStation] = []
    lockup_md: Optional[float] = None

    for i in range(len(calc_stations) - 1):
        s_curr = calc_stations[i]
        s_above = calc_stations[i + 1]

        d_md = s_curr.md_ft - s_above.md_ft
        if d_md <= 0.0:
            continue

        theta_rad = math.radians(0.5 * (s_curr.inclination_deg + s_above.inclination_deg))
        d_theta = math.radians(s_curr.inclination_deg - s_above.inclination_deg)
        d_azm = math.radians(s_curr.azimuth_deg - s_above.azimuth_deg)

        pipe = get_pipe_at_depth(s_curr.md_ft)
        w_buoyant = pipe.weight_air_lbf_ft * bf
        w_s = w_buoyant * d_md

        # 1. Normal contact force Fn (Johancsik 1984 3D formula):
        # Fn = sqrt( (T * d_azm * sin(theta))^2 + (T * d_theta + W_s * sin(theta))^2 )
        # Evaluated for rotating tension
        term_azm = t_rot * d_azm * math.sin(theta_rad)
        term_inc = t_rot * d_theta + w_s * math.sin(theta_rad)
        fn = math.sqrt(term_azm**2 + term_inc**2)

        # Normal forces for POOH and RIH
        fn_pooh = math.sqrt((t_pooh * d_azm * math.sin(theta_rad)) ** 2 + (t_pooh * d_theta + w_s * math.sin(theta_rad)) ** 2)
        fn_rih = math.sqrt((t_rih * d_azm * math.sin(theta_rad)) ** 2 + (t_rih * d_theta + w_s * math.sin(theta_rad)) ** 2)

        # 2. Tension and Torque Increments (Euler integration upward)
        # Upward axial step:
        # POOH (pulling up): tension increases by weight component + friction
        t_pooh += w_s * math.cos(theta_rad) + friction_coefficient * fn_pooh
        # RIH (running down): tension increases by weight component - friction
        t_rih += w_s * math.cos(theta_rad) - friction_coefficient * fn_rih
        # Rotating off-bottom: axial tension carries weight only; friction generates torque
        t_rot += w_s * math.cos(theta_rad)
        torque += friction_coefficient * fn * (pipe.od_in / 24.0)

        # 3. Buckling Limits (Dawson & Paslay / Wu & Juvkam-Wold)
        # Radial clearance r_c = (D_hole - OD_pipe) / 2 (inches -> ft)
        r_c_in = max(0.5, (hole_diameter_in - pipe.od_in) / 2.0)
        r_c_ft = r_c_in / 12.0
        i_in4 = pipe.moment_of_inertia_in4

        # Critical Sinusoidal Buckling Load (lbf):
        # F_sin = 2 * sqrt(E * I * w_b * sin(theta) / r_c)
        sin_theta = max(0.01, math.sin(theta_rad))
        # Convert E*I to lbf*ft^2: E (psi) * I (in^4) / 144
        ei_lbf_ft2 = (E_STEEL_PSI * i_in4) / 144.0
        f_sin = 2.0 * math.sqrt(ei_lbf_ft2 * w_buoyant * sin_theta / max(1e-4, r_c_ft))
        # Helical Buckling Load: F_hel = sqrt(2) * F_sin
        f_hel = math.sqrt(2.0) * f_sin

        # Compressive axial force during RIH is -t_rih when t_rih < 0
        comp_rih = max(0.0, -t_rih)
        is_buckled = comp_rih >= f_sin
        is_locked = comp_rih >= f_hel

        if is_locked and lockup_md is None:
            lockup_md = s_curr.md_ft

        results_rev.append(
            TorqueDragStation(
                md_ft=round(s_curr.md_ft, 1),
                tvd_ft=round(s_curr.tvd_ft, 1),
                inclination_deg=round(s_curr.inclination_deg, 1),
                tension_pooh_lbf=round(t_pooh, 1),
                tension_rih_lbf=round(t_rih, 1),
                tension_rotating_lbf=round(t_rot, 1),
                torque_ft_lbf=round(torque, 1),
                normal_force_lbf=round(fn, 1),
                sinusoidal_buckling_limit_lbf=round(f_sin, 1),
                helical_buckling_limit_lbf=round(f_hel, 1),
                is_buckled=is_buckled,
                is_locked_up=is_locked,
            )
        )

    stations = list(reversed(results_rev))
    surface = stations[0] if stations else None

    hookload_pooh = (surface.tension_pooh_lbf / 1000.0) if surface else 0.0
    hookload_rih = (surface.tension_rih_lbf / 1000.0) if surface else 0.0
    hookload_rot = (surface.tension_rotating_lbf / 1000.0) if surface else 0.0
    surf_torque = surface.torque_ft_lbf if surface else 0.0

    return TorqueDragResult(
        hookload_pooh_klbf=round(hookload_pooh, 1),
        hookload_rih_klbf=round(hookload_rih, 1),
        hookload_rotating_klbf=round(hookload_rot, 1),
        surface_torque_ft_lbf=round(surf_torque, 1),
        lockup_depth_ft=lockup_md,
        stations=stations,
    )
