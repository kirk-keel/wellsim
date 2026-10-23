"""Managed Pressure Cementing (MPC) & Hydration Gas Migration Engine (API RP 65-2).

Models:
1. Dynamic U-tubing ("free-fall" effect) during slurry displacement caused by heavy
   cement vs. light displacement mud density disparities, predicting casing vacuum,
   return flow surges, and annular ECD spikes against formation fracture breakdown.
2. Post-placement Static Gel Strength (SGS) development and hydrostatic pressure decay
   during the critical transition window (100 -> 500 lbf/100 ft2), evaluating annular
   gas channeling potential per API RP 65-2 standards.

References:
- API RP 65-2: "Isolating Potential Flow Zones During Well Construction," 2nd Edition.
- Carter, L.G. & Slagle, K.A. (1972): "A Study of Formation Gas Influx into a Gelling Cement Column."
- Beirute, R.M. (1984): "The Phenomenon of Free Fall During Primary Cementing."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GasMigrationRisk(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"


@dataclass
class CementSlurry:
    """Cement slurry rheological and hydration properties."""

    name: str
    density_ppg: float
    pv_cp: float = 40.0
    yp_lbf_100ft2: float = 18.0
    time_to_100_sgs_min: float = 120.0   # Time after placement to reach 100 lbf/100 ft2
    time_to_500_sgs_min: float = 140.0   # Time after placement to reach 500 lbf/100 ft2


@dataclass
class CementJobConfig:
    """Casing cementing design parameters.

    Attributes:
        shoe_md_ft: Total depth of casing shoe (ft).
        hole_id_in: Wellbore open hole ID (inches).
        casing_od_in: Casing outer diameter (inches).
        casing_id_in: Casing inner diameter (inches).
        mud_density_ppg: Original/displacement drilling fluid density (ppg).
        spacer_volume_bbl: Volume of weighted chemical wash/spacer (bbl).
        spacer_density_ppg: Spacer density (ppg).
        tail_slurry: Tail cement slurry specification.
        tail_volume_bbl: Volume of high-density tail cement (bbl).
        pore_pressure_ppg: Formation pore pressure equivalent at shoe (ppg).
        fracture_pressure_ppg: Formation fracture breakdown limit at shoe (ppg).
        pump_rate_bpm: Surface pump rate in barrels per minute (bpm).
    """

    shoe_md_ft: float = 10000.0
    hole_id_in: float = 12.25
    casing_od_in: float = 9.625
    casing_id_in: float = 8.681
    mud_density_ppg: float = 10.5
    spacer_volume_bbl: float = 50.0
    spacer_density_ppg: float = 12.5
    tail_slurry: CementSlurry = field(
        default_factory=lambda: CementSlurry(
            name="Tail Class G",
            density_ppg=15.8,
            pv_cp=45.0,
            yp_lbf_100ft2=20.0,
            time_to_100_sgs_min=110.0,
            time_to_500_sgs_min=125.0,
        )
    )
    tail_volume_bbl: float = 120.0
    pore_pressure_ppg: float = 11.2
    fracture_pressure_ppg: float = 14.8
    pump_rate_bpm: float = 5.0

    @property
    def casing_capacity_bbl_ft(self) -> float:
        return (self.casing_id_in**2) / 1029.4

    @property
    def annular_capacity_bbl_ft(self) -> float:
        return (self.hole_id_in**2 - self.casing_od_in**2) / 1029.4


@dataclass
class UTubeTransientStep:
    """Snapshot during dynamic cement placement."""

    volume_pumped_bbl: float
    surface_casing_pressure_psi: float
    return_flow_rate_bpm: float
    annular_ecd_at_shoe_ppg: float
    is_in_free_fall: bool
    fracture_exceeded: bool


@dataclass
class GasMigrationAssessment:
    """API RP 65-2 gas migration risk report."""

    transition_time_min: float
    initial_overbalance_psi: float
    pressure_loss_at_500_sgs_psi: float
    net_effective_pressure_at_500_sgs_psi: float
    underbalanced_duration_min: float
    risk_level: GasMigrationRisk
    recommendations: List[str]


def simulate_cementing_u_tube(
    job: CementJobConfig,
    dt_bbl: float = 5.0,
) -> List[UTubeTransientStep]:
    """Simulate transient U-tubing (free-fall) and annular ECD during cement placement.

    Tracks fluid fronts inside casing and in the annulus. When hydrostatic in casing
    exceeds annulus, surface casing pressure drops to negative (vacuum / free fall),
    accelerating return flow rate.

    Args:
        job: Cement job geometry and slurry program.
        dt_bbl: Volume increment per simulation station (bbl).

    Returns:
        List of UTubeTransientStep records.
    """
    total_casing_vol = job.shoe_md_ft * job.casing_capacity_bbl_ft
    # Total job volume = spacer + tail cement + displacement mud
    total_pump_vol = total_casing_vol + job.spacer_volume_bbl + job.tail_volume_bbl

    timeline: List[UTubeTransientStep] = []
    v_pumped = 0.0

    while v_pumped <= total_pump_vol:
        # Determine fluid column distribution inside casing and in annulus
        # 1. Inside casing:
        # What is in casing?
        # If v_pumped < spacer_vol: spacer is entering
        # If spacer_vol <= v_pumped < spacer_vol + tail_vol: tail is entering
        # If v_pumped >= spacer_vol + tail_vol: displacement mud is chasing
        v_mud_in_csg = max(0.0, v_pumped - (job.spacer_volume_bbl + job.tail_volume_bbl))
        v_tail_in_csg = max(
            0.0,
            min(job.tail_volume_bbl, v_pumped - job.spacer_volume_bbl)
            - max(0.0, v_pumped - total_casing_vol - job.spacer_volume_bbl),
        )
        v_spacer_in_csg = max(
            0.0,
            min(job.spacer_volume_bbl, v_pumped)
            - max(0.0, v_pumped - total_casing_vol),
        )
        # Remaining casing volume is original mud
        v_orig_mud_csg = max(0.0, total_casing_vol - (v_mud_in_csg + v_tail_in_csg + v_spacer_in_csg))

        # Hydrostatic in casing (psi)
        h_csg = (
            v_mud_in_csg * job.mud_density_ppg
            + v_tail_in_csg * job.tail_slurry.density_ppg
            + v_spacer_in_csg * job.spacer_density_ppg
            + v_orig_mud_csg * job.mud_density_ppg
        ) / total_casing_vol
        p_hydro_csg = 0.052 * h_csg * job.shoe_md_ft

        # 2. In annulus (fluids displaced out casing shoe):
        v_displaced_out_shoe = max(0.0, v_pumped - total_casing_vol)
        v_spacer_ann = min(job.spacer_volume_bbl, v_displaced_out_shoe)
        v_tail_ann = min(job.tail_volume_bbl, max(0.0, v_displaced_out_shoe - job.spacer_volume_bbl))
        total_ann_vol = job.shoe_md_ft * job.annular_capacity_bbl_ft
        v_mud_ann = max(0.0, total_ann_vol - (v_spacer_ann + v_tail_ann))

        h_ann = (
            v_spacer_ann * job.spacer_density_ppg
            + v_tail_ann * job.tail_slurry.density_ppg
            + v_mud_ann * job.mud_density_ppg
        ) / total_ann_vol
        p_hydro_ann = 0.052 * h_ann * job.shoe_md_ft

        # U-tube differential: delta_p = P_hydro_csg - P_hydro_ann
        delta_p_utube = p_hydro_csg - p_hydro_ann

        # If casing hydrostatic > annular hydrostatic: Free-fall occurs!
        # Surface casing pressure is pulled down into vacuum
        # Returns surge: q_out = q_pump * sqrt(1 + delta_p / p_friction)
        is_free_fall = delta_p_utube > 50.0
        if is_free_fall:
            p_surf_csg = max(-14.7, -delta_p_utube * 0.4)
            q_returns = job.pump_rate_bpm * (1.0 + min(2.5, delta_p_utube / 400.0))
        else:
            p_surf_csg = max(50.0, -delta_p_utube + 120.0)
            q_returns = job.pump_rate_bpm

        # Annular ECD at casing shoe: hydrostatic + annular friction
        # Annular velocity in ft/s
        ann_area_in2 = (math.pi / 4.0) * (job.hole_id_in**2 - job.casing_od_in**2)
        v_ann_fps = (q_returns * 42.0 * 0.3208) / (ann_area_in2 * 60.0)
        # Approximate annular friction drop (psi)
        fric_ann_psi = (0.02 * job.shoe_md_ft * h_ann * (v_ann_fps**2)) / (
            21.1 * (job.hole_id_in - job.casing_od_in)
        )

        p_shoe = p_hydro_ann + fric_ann_psi
        ecd_shoe = p_shoe / (0.052 * job.shoe_md_ft)
        frac_breach = ecd_shoe > job.fracture_pressure_ppg

        timeline.append(
            UTubeTransientStep(
                volume_pumped_bbl=round(v_pumped, 1),
                surface_casing_pressure_psi=round(p_surf_csg, 1),
                return_flow_rate_bpm=round(q_returns, 2),
                annular_ecd_at_shoe_ppg=round(ecd_shoe, 2),
                is_in_free_fall=is_free_fall,
                fracture_exceeded=frac_breach,
            )
        )
        v_pumped += dt_bbl

    return timeline


def evaluate_api_gas_migration_risk(
    job: CementJobConfig,
) -> GasMigrationAssessment:
    """Evaluate post-placement gas channeling risk according to API RP 65-2.

    Models static gel strength development and hydrostatic decay during cement hydration.
    When wall shear stress from SGS supports the cement column, hydrostatic pressure
    transferred to the shoe falls below formation pore pressure, creating an underbalanced
    window.

    Args:
        job: Cement configuration and slurry properties.

    Returns:
        GasMigrationAssessment with risk tier and engineering recommendations.
    """
    slurry = job.tail_slurry

    # 1. Critical Transition Time: time from 100 to 500 lbf/100 ft2
    t_trans = max(1.0, slurry.time_to_500_sgs_min - slurry.time_to_100_sgs_min)

    # 2. Initial hydrostatic overbalance at casing shoe (psi)
    # Length of tail slurry column in annulus (ft)
    tail_col_len_ft = min(job.shoe_md_ft, job.tail_volume_bbl / job.annular_capacity_bbl_ft)
    mud_col_len_ft = job.shoe_md_ft - tail_col_len_ft

    p_init_hydro = 0.052 * (tail_col_len_ft * slurry.density_ppg + mud_col_len_ft * job.mud_density_ppg)
    p_formation = 0.052 * job.pore_pressure_ppg * job.shoe_md_ft
    overbalance_psi = max(0.0, p_init_hydro - p_formation)

    # 3. Maximum pressure restriction from Static Gel Strength (Carter & Slagle):
    # Delta P_gel = (4 * SGS * L) / (D_h - D_csg)
    # Unit conversion: SGS in lbf/100 ft2, L in ft, D in inches
    # Delta P (psi) = (SGS * L) / (300 * (D_hole - D_csg))
    d_hyd = max(0.5, job.hole_id_in - job.casing_od_in)
    dp_gel_at_500 = (500.0 * tail_col_len_ft) / (300.0 * d_hyd)

    # Effective pressure at shoe once cement reaches 500 lbf/100 ft2
    p_eff_at_500 = max(0.0, p_init_hydro - dp_gel_at_500)

    # 4. Underbalanced duration: time cement sits below pore pressure before solidifying
    if p_eff_at_500 < p_formation:
        # Pressure decay rate during transition (psi/min)
        dp_dt = dp_gel_at_500 / t_trans
        t_underbalanced = (p_formation - p_eff_at_500) / max(0.1, dp_dt)
    else:
        t_underbalanced = 0.0

    # 5. Risk Matrix per API RP 65-2
    recs: List[str] = []
    if t_trans <= 15.0 and p_eff_at_500 >= p_formation:
        risk = GasMigrationRisk.LOW
        recs.append("Slurry exhibits rapid right-angle set (transition time <= 15 min). Low gas channeling potential.")
    elif t_trans <= 30.0 or t_underbalanced < 15.0:
        risk = GasMigrationRisk.MODERATE
        recs.append(
            f"Moderate gas migration vulnerability: Transition time is {t_trans:.0f} min. "
            f"Recommend applying 200-300 psi surface backpressure (Managed Pressure Cementing) during curing."
        )
    elif t_trans <= 45.0:
        risk = GasMigrationRisk.HIGH
        recs.append(
            f"HIGH GAS CHANNELING RISK: Static gel strength transition time is {t_trans:.0f} min (>30 min). "
            f"Underbalanced exposure window is {t_underbalanced:.1f} min. "
            f"Mandatory: Add gas-block/latex additives and apply Managed Pressure Cementing backpressure."
        )
    else:
        risk = GasMigrationRisk.SEVERE
        recs.append(
            f"SEVERE INTEGRITY HAZARD: Slurry exhibits prolonged gelation ({t_trans:.0f} min). "
            f"Hydrostatic will decay by {dp_gel_at_500:.0f} psi, exposing open hole to underbalanced gas influx. "
            f"Do not pump without redesigning slurry for right-angle set and applying Managed Pressure Cementing surface backpressure."
        )

    return GasMigrationAssessment(
        transition_time_min=round(t_trans, 1),
        initial_overbalance_psi=round(overbalance_psi, 1),
        pressure_loss_at_500_sgs_psi=round(dp_gel_at_500, 1),
        net_effective_pressure_at_500_sgs_psi=round(p_eff_at_500, 1),
        underbalanced_duration_min=round(t_underbalanced, 1),
        risk_level=risk,
        recommendations=recs,
    )
