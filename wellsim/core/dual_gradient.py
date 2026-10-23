"""Dual-Gradient Drilling (DGD) & Subsea MudLift Pump (MLP) Hydraulics Engine.

Models deepwater dual-gradient drilling systems (e.g. Chevron / Transocean SMD,
Ocean Riser Systems, AGR RMR) where the marine riser contains seawater/light fluid
and the subsea wellbore contains weighted drilling mud.

Physics:
- Decouples the water column hydrostatic from the geological formation stress.
- Subsea MudLift Pump (MLP) at the seabed mudline pumps annular returns to surface
  through a separate Mud Return Line (MRL), maintaining mudline pressure equal to
  seawater hydrostatic (P_mudline = 0.052 * rho_seawater * WD).
- Eliminates the narrow deepwater drilling margin (ECD vs. pore/frac limits),
  preventing ballooning, kicks, and differential sticking while reducing required
  casing strings.
- Solves drillpipe-to-annulus U-tube hydrostatic imbalance and sizes required
  Drillstring Valve (DSV) cracking pressure.

References:
- Schubert, J.J. et al. (2006): "Dual Gradient Drilling: Technology, Implementation,
  and Operational Aspects," SPE 99042.
- Smith, K.L. et al. (2001): "Subsea MudLift Drilling Joint Industry Project:
  Operational Verification," SPE 71357.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DGDSystemConfig:
    """Dual Gradient Drilling configuration parameters.

    Attributes:
        water_depth_ft: Depth from rig floor to seabed mudline (ft).
        total_depth_ft: Total well measured depth (ft).
        seawater_density_ppg: Seawater / riser fluid density (ppg, ~8.55 ppg).
        mud_density_ppg: Subsea wellbore heavy mud weight (ppg, e.g. 14.0 ppg).
        mrl_id_in: Internal diameter of separate Mud Return Line (inches, ~6.0").
        hole_id_in: Wellbore open hole diameter (inches, ~8.5").
        pipe_od_in: Drillpipe outer diameter (inches, ~5.0").
        pipe_id_in: Drillpipe inner diameter (inches, ~4.276").
        mud_pv_cp: Plastic viscosity of heavy mud (cP).
        mud_yp_lbf_100ft2: Yield point of heavy mud (lbf/100 ft2).
        mlp_max_discharge_psi: Maximum rated MLP discharge pressure (psi).
        mlp_max_rate_gpm: Maximum rated MLP pumping capacity (gpm).
    """

    water_depth_ft: float = 6000.0
    total_depth_ft: float = 18000.0
    seawater_density_ppg: float = 8.55
    mud_density_ppg: float = 14.5
    mrl_id_in: float = 6.0
    hole_id_in: float = 8.5
    pipe_od_in: float = 5.0
    pipe_id_in: float = 4.276
    mud_pv_cp: float = 28.0
    mud_yp_lbf_100ft2: float = 16.0
    mlp_max_discharge_psi: float = 6000.0
    mlp_max_rate_gpm: float = 1200.0

    @property
    def seawater_hydrostatic_at_mudline_psi(self) -> float:
        """Target mudline pressure setpoint: P_set = 0.052 * rho_sw * WD."""
        return 0.052 * self.seawater_density_ppg * self.water_depth_ft


@dataclass
class DGDProfileStation:
    """Hydraulic pressure and ECD comparison station along the wellbore."""

    depth_ft: float
    p_conventional_psi: float
    p_dgd_psi: float
    ecd_conventional_ppg: float
    ecd_dgd_ppg: float
    pore_pressure_ppg: float
    fracture_gradient_ppg: float
    conventional_in_window: bool
    dgd_in_window: bool


@dataclass
class MLPOperatingState:
    """Operating conditions for the Subsea MudLift Pump."""

    suction_pressure_psi: float
    discharge_pressure_psi: float
    pump_differential_pressure_psi: float
    flow_rate_gpm: float
    mrl_friction_loss_psi: float
    hydraulic_horsepower_hhp: float
    is_within_operating_limits: bool


def calculate_pipe_friction_loss_psi(
    length_ft: float,
    id_in: float,
    flow_rate_gpm: float,
    density_ppg: float,
    pv_cp: float,
    yp_lbf_100ft2: float,
) -> float:
    """Bingham plastic frictional pressure drop through a pipe (psi)."""
    if flow_rate_gpm <= 0.0 or length_ft <= 0.0:
        return 0.0

    area_in2 = (math.pi / 4.0) * (id_in**2)
    velocity_fps = (flow_rate_gpm * 0.3208) / (area_in2 * 60.0)

    # Effective viscosity (cP)
    mu_eff = pv_cp + (5.0 * yp_lbf_100ft2 * id_in) / max(0.01, velocity_fps)
    reynolds = (928.0 * density_ppg * velocity_fps * id_in) / max(1.0, mu_eff)

    if reynolds < 2100.0:
        f = 16.0 / max(1.0, reynolds)
    else:
        f = 0.0791 / (reynolds**0.25)

    delta_p = (f * length_ft * density_ppg * (velocity_fps**2)) / (25.8 * id_in)
    return delta_p


def calculate_annular_friction_loss_psi(
    length_ft: float,
    hole_id_in: float,
    pipe_od_in: float,
    flow_rate_gpm: float,
    density_ppg: float,
    pv_cp: float,
    yp_lbf_100ft2: float,
) -> float:
    """Bingham plastic frictional pressure drop in the annulus (psi)."""
    if flow_rate_gpm <= 0.0 or length_ft <= 0.0:
        return 0.0

    d_eff = hole_id_in - pipe_od_in
    area_in2 = (math.pi / 4.0) * (hole_id_in**2 - pipe_od_in**2)
    velocity_fps = (flow_rate_gpm * 0.3208) / (area_in2 * 60.0)

    mu_eff = pv_cp + (5.0 * yp_lbf_100ft2 * d_eff) / max(0.01, velocity_fps)
    reynolds = (757.0 * density_ppg * velocity_fps * d_eff) / max(1.0, mu_eff)

    if reynolds < 2100.0:
        f = 24.0 / max(1.0, reynolds)
    else:
        f = 0.0791 / (reynolds**0.25)

    delta_p = (f * length_ft * density_ppg * (velocity_fps**2)) / (21.1 * d_eff)
    return delta_p


def evaluate_mlp_operating_state(
    config: DGDSystemConfig,
    flow_rate_gpm: float,
) -> MLPOperatingState:
    """Calculate subsea MudLift pump suction, discharge, and HHP requirements.

    Args:
        config: DGD system configuration.
        flow_rate_gpm: Circulation rate through wellbore and MRL.

    Returns:
        MLPOperatingState.
    """
    p_suction = config.seawater_hydrostatic_at_mudline_psi

    # Discharge pressure required to lift heavy mud through MRL back to rig floor:
    # P_discharge = Hydrostatic_MRL + Friction_MRL
    p_hydro_mrl = 0.052 * config.mud_density_ppg * config.water_depth_ft
    p_fric_mrl = calculate_pipe_friction_loss_psi(
        length_ft=config.water_depth_ft,
        id_in=config.mrl_id_in,
        flow_rate_gpm=flow_rate_gpm,
        density_ppg=config.mud_density_ppg,
        pv_cp=config.mud_pv_cp,
        yp_lbf_100ft2=config.mud_yp_lbf_100ft2,
    )
    p_discharge = p_hydro_mrl + p_fric_mrl

    delta_p_pump = max(0.0, p_discharge - p_suction)
    hhp = (delta_p_pump * flow_rate_gpm) / 1714.0

    is_ok = (
        p_discharge <= config.mlp_max_discharge_psi
        and flow_rate_gpm <= config.mlp_max_rate_gpm
    )

    return MLPOperatingState(
        suction_pressure_psi=round(p_suction, 1),
        discharge_pressure_psi=round(p_discharge, 1),
        pump_differential_pressure_psi=round(delta_p_pump, 1),
        flow_rate_gpm=round(flow_rate_gpm, 1),
        mrl_friction_loss_psi=round(p_fric_mrl, 1),
        hydraulic_horsepower_hhp=round(hhp, 1),
        is_within_operating_limits=is_ok,
    )


def calculate_u_tube_imbalance(config: DGDSystemConfig) -> Dict[str, float]:
    """Calculate the drillpipe-to-annulus U-tube hydrostatic pressure imbalance.

    Because drillpipe contains heavy mud from surface while the annulus has seawater
    in the riser, an unconstrained U-tube will siphon fluid and drop fluid level in DP.

    Returns:
        Dict containing differential pressure and required Drillstring Valve (DSV) cracking pressure.
    """
    # DP hydrostatic at mudline: 0.052 * rho_mud * WD
    p_dp_mudline = 0.052 * config.mud_density_ppg * config.water_depth_ft
    # Annular hydrostatic at mudline: 0.052 * rho_sw * WD
    p_ann_mudline = 0.052 * config.seawater_density_ppg * config.water_depth_ft

    delta_p_utube = p_dp_mudline - p_ann_mudline
    # Fluid drop in drillpipe if DSV is not present (ft)
    unbalanced_head_drop_ft = delta_p_utube / (0.052 * config.mud_density_ppg)

    return {
        "u_tube_differential_psi": round(delta_p_utube, 1),
        "drillstring_head_drop_ft": round(unbalanced_head_drop_ft, 1),
        "recommended_dsv_cracking_pressure_psi": round(delta_p_utube + 150.0, 1),
    }


def compare_dgd_vs_conventional_profiles(
    config: DGDSystemConfig,
    pore_pressures: List[Tuple[float, float]],
    frac_gradients: List[Tuple[float, float]],
    flow_rate_gpm: float = 650.0,
) -> List[DGDProfileStation]:
    """Compare ECD and pressure profiles between conventional single gradient and DGD.

    Args:
        config: DGD system configuration.
        pore_pressures: List of (depth_ft, pore_ppg) pairs.
        frac_gradients: List of (depth_ft, frac_ppg) pairs.
        flow_rate_gpm: Active circulation rate.

    Returns:
        List of DGDProfileStation records.
    """
    stations: List[DGDProfileStation] = []
    # Combined depth list
    all_depths = sorted(list({z for z, _ in pore_pressures} | {z for z, _ in frac_gradients}))

    def _interp(curve: List[Tuple[float, float]], z: float) -> float:
        if z <= curve[0][0]:
            return curve[0][1]
        if z >= curve[-1][0]:
            return curve[-1][1]
        for i in range(len(curve) - 1):
            z0, v0 = curve[i]
            z1, v1 = curve[i + 1]
            if z0 <= z <= z1:
                return v0 + (v1 - v0) * (z - z0) / (z1 - z0)
        return curve[-1][1]

    for z in all_depths:
        p_pore_ppg = _interp(pore_pressures, z)
        p_frac_ppg = _interp(frac_gradients, z)

        # 1. Conventional single-gradient system: single mud weight from surface
        # Mud weight must be selected to hold bottomhole pore pressure
        # Under conventional, hydrostatic = 0.052 * rho_mud * z
        fric_ann_conv = calculate_annular_friction_loss_psi(
            length_ft=z,
            hole_id_in=config.hole_id_in,
            pipe_od_in=config.pipe_od_in,
            flow_rate_gpm=flow_rate_gpm,
            density_ppg=config.mud_density_ppg,
            pv_cp=config.mud_pv_cp,
            yp_lbf_100ft2=config.mud_yp_lbf_100ft2,
        )
        p_conv = 0.052 * config.mud_density_ppg * z + fric_ann_conv
        ecd_conv = p_conv / (0.052 * max(1.0, z))

        # 2. DGD Dual-gradient system:
        # Above mudline (z <= WD): P = 0.052 * rho_sw * z
        # Below mudline (z > WD): P = P_mudline_setpoint + 0.052 * rho_mud * (z - WD) + Fric_ann
        if z <= config.water_depth_ft:
            p_dgd = 0.052 * config.seawater_density_ppg * z
        else:
            subsea_interval_ft = z - config.water_depth_ft
            fric_ann_dgd = calculate_annular_friction_loss_psi(
                length_ft=subsea_interval_ft,
                hole_id_in=config.hole_id_in,
                pipe_od_in=config.pipe_od_in,
                flow_rate_gpm=flow_rate_gpm,
                density_ppg=config.mud_density_ppg,
                pv_cp=config.mud_pv_cp,
                yp_lbf_100ft2=config.mud_yp_lbf_100ft2,
            )
            p_dgd = (
                config.seawater_hydrostatic_at_mudline_psi
                + 0.052 * config.mud_density_ppg * subsea_interval_ft
                + fric_ann_dgd
            )

        ecd_dgd = p_dgd / (0.052 * max(1.0, z))

        conv_ok = (p_pore_ppg <= ecd_conv <= p_frac_ppg)
        dgd_ok = (p_pore_ppg <= ecd_dgd <= p_frac_ppg)

        stations.append(
            DGDProfileStation(
                depth_ft=round(z, 1),
                p_conventional_psi=round(p_conv, 1),
                p_dgd_psi=round(p_dgd, 1),
                ecd_conventional_ppg=round(ecd_conv, 2),
                ecd_dgd_ppg=round(ecd_dgd, 2),
                pore_pressure_ppg=round(p_pore_ppg, 2),
                fracture_gradient_ppg=round(p_frac_ppg, 2),
                conventional_in_window=conv_ok,
                dgd_in_window=dgd_ok,
            )
        )

    return stations
