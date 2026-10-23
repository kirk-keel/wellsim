"""Dynamic Wellbore Heat Transfer Engine (Hasan & Kabir / Ramey Formulation).

Models conjugate convective and conductive heat transfer between circulating wellbore
fluids (drillpipe and annulus) and the geothermal formation over elapsed circulating time.

Replaces static linear or purely adiabatic approximations with the rigorous
Hasan & Kabir (1994, 1996) analytical formulation.

References:
- Hasan, A.R. & Kabir, C.S. (1994): "Aspects of Wellbore Heat Transfer During
  Drilling Operations," SPE-28485.
- Hasan, A.R. & Kabir, C.S. (1996): "A New Model for Computing Fluid Temperature
  Profiles in Wellbores," SPE-35603.
- Ramey, H.J. Jr. (1962): "Wellbore Heat Transmission," JPT 14(4), SPE-96-PA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "ThermalProperties",
    "WellboreTemperatureProfile",
    "calculate_hasan_kabir_temperatures",
]


@dataclass(frozen=True)
class ThermalProperties:
    """Thermal properties of the fluid and formation.

    Attributes:
        surface_temp_degf: Mean ambient surface temperature (degF).
        geothermal_gradient_degf_100ft: Formation geothermal gradient (degF / 100 ft, typical 1.2 to 2.0).
        fluid_cp_btu_lbm_f: Mud specific heat capacity (Btu / (lbm degF), typical 0.80 to 0.92).
        rock_k_btu_hr_ft_f: Formation rock thermal conductivity (Btu / (hr ft degF), typical 1.2 to 1.8).
        rock_diffusivity_ft2_hr: Formation thermal diffusivity (ft^2 / hr, typical 0.04 to 0.06).
        overall_u_btu_hr_ft2_f: Overall heat transfer coefficient U (Btu / (hr ft^2 degF), typical 2.0 to 10.0).
    """

    surface_temp_degf: float = 70.0
    geothermal_gradient_degf_100ft: float = 1.50
    fluid_cp_btu_lbm_f: float = 0.85
    rock_k_btu_hr_ft_f: float = 1.40
    rock_diffusivity_ft2_hr: float = 0.045
    overall_u_btu_hr_ft2_f: float = 5.0


@dataclass(frozen=True)
class WellboreTemperatureProfile:
    """Calculated temperature profile along the wellbore.

    Attributes:
        depth_stations_ft: Measured / vertical depth stations (ft).
        pipe_temp_degf: Downward flowing drillstring fluid temperature (degF).
        annulus_temp_degf: Upward flowing annular return fluid temperature (degF).
        formation_temp_degf: Undisturbed in-situ formation rock temperature (degF).
        surface_return_temp_degf: Final return mud temperature at shaker / choke (degF).
        bottomhole_temp_degf: Flowing temperature at bit (degF).
    """

    depth_stations_ft: List[float]
    pipe_temp_degf: List[float]
    annulus_temp_degf: List[float]
    formation_temp_degf: List[float]
    surface_return_temp_degf: float
    bottomhole_temp_degf: float


def _hasan_kabir_time_function(t_d: float) -> float:
    """Dimensionless transient heat conduction time function f(t_D)."""
    if t_d <= 0.0:
        return 0.01
    if t_d <= 1.5:
        return 1.1281 * math.sqrt(t_d) * (1.0 - 0.3 * math.sqrt(t_d))
    return (0.4063 + 0.5 * math.log(t_d)) * (1.0 + 0.6 / t_d)


def calculate_hasan_kabir_temperatures(
    total_depth_ft: float,
    flow_rate_gpm: float,
    mud_density_ppg: float,
    inlet_temp_degf: float,
    circulating_time_hr: float = 4.0,
    wellbore_radius_ft: float = 0.354,  # 8.5" hole -> r_wb = 4.25 in = 0.354 ft
    thermal: ThermalProperties | None = None,
    num_stations: int = 21,
) -> WellboreTemperatureProfile:
    """Calculate conjugate drillpipe and annular temperature profiles (Hasan & Kabir 1994).

    Args:
        total_depth_ft: Total vertical depth of well (ft).
        flow_rate_gpm: Mud circulation rate (gpm).
        mud_density_ppg: Mud weight (ppg).
        inlet_temp_degf: Surface suction pit mud temperature entering string (degF).
        circulating_time_hr: Continuous pumping duration (hours).
        wellbore_radius_ft: Wellbore radius (ft).
        thermal: ThermalProperties.
        num_stations: Number of depth stations to evaluate.

    Returns:
        WellboreTemperatureProfile.
    """
    if thermal is None:
        thermal = ThermalProperties()

    if flow_rate_gpm <= 0.0 or total_depth_ft <= 0.0:
        # Static geothermal profile
        dz = total_depth_ft / max(1, num_stations - 1)
        depths = [i * dz for i in range(num_stations)]
        t_form = [thermal.surface_temp_degf + (thermal.geothermal_gradient_degf_100ft / 100.0) * z for z in depths]
        return WellboreTemperatureProfile(
            depth_stations_ft=depths,
            pipe_temp_degf=t_form,
            annulus_temp_degf=t_form,
            formation_temp_degf=t_form,
            surface_return_temp_degf=thermal.surface_temp_degf,
            bottomhole_temp_degf=t_form[-1],
        )

    # Mass flow rate w (lbm/hr): gpm * (8.34 * ppg / 8.34) * 60 * 8.34 = gpm * ppg * 500.4
    w_lbm_hr = flow_rate_gpm * (mud_density_ppg / 8.34) * 8.345 * 60.0
    c_p = thermal.fluid_cp_btu_lbm_f
    k_e = thermal.rock_k_btu_hr_ft_f
    u = thermal.overall_u_btu_hr_ft2_f
    r_to = wellbore_radius_ft
    g_g = thermal.geothermal_gradient_degf_100ft / 100.0  # degF / ft

    # Dimensionless time t_D = alpha * t / r_wb^2
    t_d = (thermal.rock_diffusivity_ft2_hr * max(0.1, circulating_time_hr)) / (r_to**2)
    f_td = _hasan_kabir_time_function(t_d)

    # Numerical shooting integrator for coupled 2-stream counter-current heat exchange
    u_p = 25.0  # Overall pipe-to-annulus heat transfer U (Btu/(hr ft^2 degF))
    r_po = 0.208  # 5" OD pipe radius (ft)
    a_len = max(100.0, (w_lbm_hr * c_p) / (2.0 * math.pi * r_po * u_p))

    numerator = w_lbm_hr * c_p * (k_e + r_to * u * f_td)
    denominator = 2.0 * math.pi * r_to * u * k_e
    b_len = max(500.0, numerator / max(1e-4, denominator))

    # Shooting method to find T_annulus(0) such that T_pipe(TD) == T_annulus(TD)
    n_steps = max(100, num_stations * 5)
    dz_step = total_depth_ft / n_steps

    def _march_profile(t_ann_0: float) -> Tuple[float, List[float], List[float], List[float], List[float]]:
        tp = float(inlet_temp_degf)
        ta = float(t_ann_0)
        z = 0.0

        all_depths: List[float] = [0.0]
        all_tp: List[float] = [tp]
        all_ta: List[float] = [ta]
        all_te: List[float] = [thermal.surface_temp_degf]

        for _ in range(n_steps):
            z_mid = z + 0.5 * dz_step
            te_mid = thermal.surface_temp_degf + g_g * z_mid
            dtp = ((ta - tp) / a_len) * dz_step
            dta = (((ta - te_mid) / b_len) + ((ta - tp) / a_len)) * dz_step
            tp += dtp
            ta += dta
            z += dz_step
            all_depths.append(z)
            all_tp.append(tp)
            all_ta.append(ta)
            all_te.append(thermal.surface_temp_degf + g_g * z)

        mismatch = tp - ta
        return mismatch, all_depths, all_tp, all_ta, all_te

    # Linear secant solve for T_ann(0)
    guess1 = float(inlet_temp_degf)
    diff1, _, _, _, _ = _march_profile(guess1)
    guess2 = float(inlet_temp_degf) + 40.0
    diff2, _, _, _, _ = _march_profile(guess2)

    if abs(diff2 - diff1) > 1e-6:
        optimal_t_ann_0 = guess1 - diff1 * (guess2 - guess1) / (diff2 - diff1)
    else:
        optimal_t_ann_0 = guess1

    _, all_z, all_tp, all_ta, all_te = _march_profile(optimal_t_ann_0)

    # Downsample to requested num_stations
    step_skip = max(1, len(all_z) // (num_stations - 1))
    depths: List[float] = []
    t_pipe: List[float] = []
    t_ann: List[float] = []
    t_form: List[float] = []

    for idx in range(0, len(all_z), step_skip):
        depths.append(round(all_z[idx], 1))
        t_pipe.append(round(all_tp[idx], 1))
        t_ann.append(round(all_ta[idx], 1))
        t_form.append(round(all_te[idx], 1))

    # Ensure TD is the final point
    if depths[-1] != round(total_depth_ft, 1):
        depths.append(round(total_depth_ft, 1))
        t_pipe.append(round(all_tp[-1], 1))
        t_ann.append(round(all_ta[-1], 1))
        t_form.append(round(all_te[-1], 1))

    t_bit = all_tp[-1]
    t_ann_surf = all_ta[0]

    t_bit = t_pipe[-1]
    t_ann_surf = t_ann[0]

    return WellboreTemperatureProfile(
        depth_stations_ft=depths,
        pipe_temp_degf=t_pipe,
        annulus_temp_degf=t_ann,
        formation_temp_degf=t_form,
        surface_return_temp_degf=round(t_ann_surf, 1),
        bottomhole_temp_degf=round(t_bit, 1),
    )
