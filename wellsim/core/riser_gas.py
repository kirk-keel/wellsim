"""Deepwater Riser Gas Handling & Diverter Routing Engine (API RP 16Q & 53).

Simulates gas influx migration, exponential expansion, and dynamic unloading in deepwater
marine risers above the subsea BOP stack, with automated Riser Gas Handling System (RGHS)
isolation, Mud-Gas Separator (MGS) capacity screening, and diverter overboard routing.

Physics:
- Gas crossing the subsea BOP into a large-bore marine riser (e.g., 19.5" ID, 5000+ ft WD)
  experiences catastrophic volumetric expansion as hydrostatic drops near surface (P -> 14.7 psi).
- Liquid column above gas is accelerated upward, leading to severe annular unloading.
- Riser base pressure drops exponentially as heavy mud is voided, threatening riser collapse.
- RGHS packer isolates riser below water line, routing gas through choke manifold to MGS.
- When gas volume exceeds MGS liquid-seal blow-through threshold, automated diverter pack-off
  discharges gas overboard via downwind diverter lines (12"-16" ID).

References:
- API RP 16Q: "Design, Selection, Operation and Maintenance of Marine Drilling Riser Systems."
- API Standard 53: "Blowout Prevention Equipment Systems for Drilling Wells."
- Santos, O.L.A. (1991): "A Dynamic Model of Diverter Operations for Handling Gas Kicks."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class DiverterRoute(str, Enum):
    SHUT = "SHUT"
    MGS_POOR_BOY = "MGS_POOR_BOY"          # Choke line to Mud-Gas Separator
    OVERBOARD_STARBOARD = "OVERBOARD_STARBOARD"  # Overboard line downwind
    OVERBOARD_PORT = "OVERBOARD_PORT"            # Overboard line downwind


@dataclass
class MarineRiserGeometry:
    """Deepwater marine drilling riser and auxiliary line configuration.

    Attributes:
        water_depth_ft: Water depth from rig floor/rotary table to seabed BOP (ft).
        riser_id_in: Internal diameter of marine riser (inches, typically 19.5").
        diverter_line_id_in: Diverter overboard line ID (inches, typically 12" to 16").
        diverter_line_len_ft: Overboard discharge line length to ship side (ft).
        rghs_packer_depth_ft: Depth of subsea riser gas handling annular packer (ft).
        mgs_gas_capacity_mmscfd: Maximum handling rate of Poor Boy degasser before seal blow (MMSCFD).
        mgs_liquid_seal_height_ft: Mud-gas separator U-tube liquid seal head (ft).
        diverter_housing_rating_psi: Maximum safe backpressure rating on diverter packer (psi, ~500 psi).
    """

    water_depth_ft: float = 5000.0
    riser_id_in: float = 19.5
    diverter_line_id_in: float = 14.0
    diverter_line_len_ft: float = 120.0
    rghs_packer_depth_ft: float = 150.0
    mgs_gas_capacity_mmscfd: float = 40.0
    mgs_liquid_seal_height_ft: float = 20.0
    diverter_housing_rating_psi: float = 500.0

    @property
    def riser_capacity_bbl_ft(self) -> float:
        """Riser annular capacity (bbl/ft)."""
        return (self.riser_id_in**2) / 1029.4


@dataclass
class RiserGasTimeStep:
    """Snapshot of marine riser unloading dynamics during a rising gas bubble."""

    time_s: float
    bubble_top_depth_ft: float
    bubble_bottom_depth_ft: float
    bubble_volume_bbl: float
    void_fraction: float
    liquid_unloading_rate_gpm: float
    gas_flow_rate_mmscfd: float
    riser_base_pressure_psi: float
    diverter_housing_backpressure_psi: float
    active_route: DiverterRoute
    mgs_liquid_seal_intact: bool
    is_diverter_safe: bool


@dataclass
class RiserGasSimulationResult:
    """Overall summary of deepwater riser gas event."""

    peak_liquid_unloading_gpm: float
    peak_gas_rate_mmscfd: float
    peak_diverter_backpressure_psi: float
    min_riser_base_pressure_psi: float
    total_mud_expelled_bbl: float
    mgs_blown_through: bool
    diverter_rating_exceeded: bool
    summary_messages: List[str]
    timeline: List[RiserGasTimeStep]


def calculate_diverter_friction_backpressure(
    gas_rate_mmscfd: float,
    liquid_rate_gpm: float,
    line_id_in: float,
    line_length_ft: float,
    mud_weight_ppg: float,
) -> float:
    """Calculate dynamic backpressure in diverter overboard piping during multiphase venting.

    Uses homogeneous multiphase Fanning friction model for high-rate venting.
    """
    if gas_rate_mmscfd <= 0.0 and liquid_rate_gpm <= 0.0:
        return 0.0

    area_ft2 = (math.pi / 4.0) * ((line_id_in / 12.0) ** 2)

    # Convert gas rate MMSCFD to volumetric CFS at mean atmospheric pressure (20 psia)
    q_gas_cfs = (gas_rate_mmscfd * 1e6 / 86400.0) * (14.7 / 20.0) * (520.0 / 520.0)
    q_liq_cfs = liquid_rate_gpm / 448.831

    q_total_cfs = q_gas_cfs + q_liq_cfs
    velocity_fps = q_total_cfs / max(0.01, area_ft2)

    # Average mixture density (lbm/ft3)
    rho_liq_lb_ft3 = mud_weight_ppg * 7.48052
    rho_gas_lb_ft3 = 0.0764 * (20.0 / 14.7)  # air/methane approx
    h_l = q_liq_cfs / max(1e-4, q_total_cfs)
    rho_mix = h_l * rho_liq_lb_ft3 + (1.0 - h_l) * rho_gas_lb_ft3

    # Darcy friction factor for rough steel pipe
    f_darcy = 0.022

    d_ft = line_id_in / 12.0
    delta_p_psf = f_darcy * (line_length_ft / d_ft) * (rho_mix * (velocity_fps**2)) / (2.0 * 32.174)
    # Add exit kinetic head loss
    delta_p_psf += 1.0 * (rho_mix * (velocity_fps**2)) / (2.0 * 32.174)

    return delta_p_psf / 144.0  # psi


def simulate_riser_gas_unloading(
    riser: MarineRiserGeometry,
    influx_volume_at_bop_bbl: float,
    mud_weight_ppg: float,
    initial_pump_rate_gpm: float = 0.0,
    rghs_isolated: bool = False,
    dt_s: float = 2.0,
) -> RiserGasSimulationResult:
    """Simulate gas bubble rising through a marine riser from seabed BOP to surface.

    Tracks slip velocity, exponential Boyle/DAK expansion, liquid ejection from top of riser,
    and automatic diverter actuation / MGS seal integrity.

    Args:
        riser: Marine riser dimensions and diverter ratings.
        influx_volume_at_bop_bbl: Gas bubble volume as it crosses seabed BOP into riser (bbl).
        mud_weight_ppg: Riser drilling fluid density (ppg).
        initial_pump_rate_gpm: Rig pump circulation rate through riser (gpm).
        rghs_isolated: Whether Subsea Riser Gas Handling annular packer is closed.
        dt_s: Time step in seconds.

    Returns:
        RiserGasSimulationResult with complete unloading trajectory.
    """
    time = 0.0
    current_depth = riser.water_depth_ft  # Bubble top starts at seabed

    # Hydrostatic pressure at seabed BOP (psia)
    p_seabed = 14.7 + 0.052 * mud_weight_ppg * riser.water_depth_ft
    z_gas_base = 0.85  # Z-factor at seabed

    # Conserved gas mass parameter: (P * V / Z) = invariant
    p_v_over_z = (p_seabed * influx_volume_at_bop_bbl) / z_gas_base

    v_cap = riser.riser_capacity_bbl_ft
    # Initial bubble length in riser (ft)
    h_bubble = max(5.0, influx_volume_at_bop_bbl / v_cap)

    timeline: List[RiserGasTimeStep] = []
    messages: List[str] = []

    mgs_blown = False
    diverter_breached = False

    peak_liquid_gpm = 0.0
    peak_gas_mmscfd = 0.0
    peak_diverter_p = 0.0
    min_riser_base_p = p_seabed
    total_mud_expelled = 0.0

    # Slip velocity of gas in large riser (Taylor bubble slip: 0.35 * sqrt(g * D) ~ 2.5 ft/s)
    d_riser_ft = riser.riser_id_in / 12.0
    slip_velocity_fps = 0.35 * math.sqrt(32.174 * d_riser_ft)

    while current_depth > 0.0 and time < 3600.0:
        # Liquid velocity in riser from pumps
        v_liq_fps = initial_pump_rate_gpm / (v_cap * 42.0 * 60.0)

        # Expansion front velocity: as gas expands, the top of the bubble accelerates upward
        prev_h = (timeline[-1].bubble_volume_bbl / v_cap) if timeline else h_bubble
        # Total rising velocity of gas bubble top front
        dh_dt = max(0.0, (h_bubble - prev_h) / dt_s)
        u_bubble_fps = v_liq_fps + slip_velocity_fps + dh_dt

        current_depth = max(0.0, current_depth - u_bubble_fps * dt_s)

        # Local pressure at bubble top
        p_bubble = 14.7 + 0.052 * mud_weight_ppg * current_depth
        z_bubble = max(0.70, min(1.0, 1.0 - (p_bubble / 10000.0) * 0.15))

        # Instantaneous expanded bubble volume (bbl)
        v_bubble = (p_v_over_z * z_bubble) / max(14.7, p_bubble)
        h_bubble = v_bubble / v_cap

        # Gas rate in MMSCFD: V_std = V_bbl * (5.615 ft3/bbl) * (P / 14.7) * (520 / T)
        # Volumetric expansion rate dV/dt drives liquid ejection
        prev_v = timeline[-1].bubble_volume_bbl if timeline else influx_volume_at_bop_bbl
        dv_dt = (v_bubble - prev_v) / dt_s  # bbl/s

        # Physical bound: cannot expel more liquid mud than what exists above the bubble in the riser
        mud_above_bbl = max(0.0, current_depth * v_cap)
        liquid_expelled_bbl = min(mud_above_bbl, max(0.0, dv_dt * dt_s))
        liquid_unloading_gpm = (
            (liquid_expelled_bbl / dt_s) * 42.0 * 60.0
            + (initial_pump_rate_gpm if mud_above_bbl > 0.0 else 0.0)
        )
        total_mud_expelled += liquid_expelled_bbl

        # Standard gas flow rate emerging at surface (MMSCFD)
        gas_rate_cfs = (dv_dt * 5.615) * (p_bubble / 14.7)
        gas_rate_mmscfd = max(0.0, (gas_rate_cfs * 86400.0) / 1e6)

        # Riser base pressure: hydrostatic of remaining mud + gas void reduction
        mud_height_in_riser = max(0.0, riser.water_depth_ft - h_bubble)
        p_riser_base = 14.7 + 0.052 * mud_weight_ppg * mud_height_in_riser

        # Route decision:
        # If RGHS isolated, gas goes through subsea choke to MGS
        # If near surface (current_depth < 200 ft) or gas rate high:
        if rghs_isolated:
            active_route = DiverterRoute.MGS_POOR_BOY
        elif gas_rate_mmscfd > riser.mgs_gas_capacity_mmscfd or current_depth < 300.0:
            active_route = DiverterRoute.OVERBOARD_STARBOARD
        elif gas_rate_mmscfd > 2.0:
            active_route = DiverterRoute.MGS_POOR_BOY
        else:
            active_route = DiverterRoute.SHUT

        # MGS Liquid Seal Check (U-tube seal pressure = 0.052 * rho * h_seal ~ 10-15 psi)
        p_seal_max = 0.052 * mud_weight_ppg * riser.mgs_liquid_seal_height_ft
        # Backpressure in MGS
        mgs_backpressure = (gas_rate_mmscfd / max(1.0, riser.mgs_gas_capacity_mmscfd)) ** 2 * 12.0
        mgs_seal_ok = mgs_backpressure < p_seal_max

        if active_route == DiverterRoute.MGS_POOR_BOY and not mgs_seal_ok:
            mgs_blown = True
            active_route = DiverterRoute.OVERBOARD_STARBOARD  # Trip to overboard

        # Diverter line backpressure check
        if active_route in (DiverterRoute.OVERBOARD_STARBOARD, DiverterRoute.OVERBOARD_PORT):
            diverter_p = calculate_diverter_friction_backpressure(
                gas_rate_mmscfd=gas_rate_mmscfd,
                liquid_rate_gpm=liquid_unloading_gpm,
                line_id_in=riser.diverter_line_id_in,
                line_length_ft=riser.diverter_line_len_ft,
                mud_weight_ppg=mud_weight_ppg,
            )
        else:
            diverter_p = 0.0

        is_diverter_ok = diverter_p < riser.diverter_housing_rating_psi
        if not is_diverter_ok:
            diverter_breached = True

        # Track extremes
        peak_liquid_gpm = max(peak_liquid_gpm, liquid_unloading_gpm)
        peak_gas_mmscfd = max(peak_gas_mmscfd, gas_rate_mmscfd)
        peak_diverter_p = max(peak_diverter_p, diverter_p)
        min_riser_base_p = min(min_riser_base_p, p_riser_base)

        timeline.append(
            RiserGasTimeStep(
                time_s=round(time, 1),
                bubble_top_depth_ft=round(current_depth, 1),
                bubble_bottom_depth_ft=round(current_depth + h_bubble, 1),
                bubble_volume_bbl=round(v_bubble, 1),
                void_fraction=round(min(1.0, h_bubble / max(1.0, riser.water_depth_ft)), 3),
                liquid_unloading_rate_gpm=round(liquid_unloading_gpm, 1),
                gas_flow_rate_mmscfd=round(gas_rate_mmscfd, 2),
                riser_base_pressure_psi=round(p_riser_base, 1),
                diverter_housing_backpressure_psi=round(diverter_p, 1),
                active_route=active_route,
                mgs_liquid_seal_intact=mgs_seal_ok,
                is_diverter_safe=is_diverter_ok,
            )
        )
        time += dt_s

    # Generate Summary Warnings & Protocol Messages
    if mgs_blown:
        messages.append(
            f"MGS LIQUID SEAL BLOW-THROUGH: Peak gas rate ({peak_gas_mmscfd:.1f} MMSCFD) "
            f"exceeded degasser seal limit ({p_seal_max:.1f} psi). Automatic switchover to overboard diverter line executed."
        )
    if diverter_breached:
        messages.append(
            f"DIVERTER HOUSING OVERPRESSURE: Backpressure peaked at {peak_diverter_p:.1f} psi, "
            f"exceeding the {riser.diverter_housing_rating_psi:.0f} psi packer rating! "
            f"Risk of rig floor gas discharge."
        )
    else:
        messages.append(
            f"Diverter system safely contained peak backpressure: {peak_diverter_p:.1f} psi < "
            f"{riser.diverter_housing_rating_psi:.0f} psi limit."
        )

    hydro_loss = p_seabed - min_riser_base_p
    messages.append(
        f"Riser hydrostatic collapse vulnerability: Seabed base pressure dropped by {hydro_loss:.0f} psi "
        f"({p_seabed:.0f} -> {min_riser_base_p:.0f} psi) as gas voided the riser."
    )

    return RiserGasSimulationResult(
        peak_liquid_unloading_gpm=round(peak_liquid_gpm, 1),
        peak_gas_rate_mmscfd=round(peak_gas_mmscfd, 2),
        peak_diverter_backpressure_psi=round(peak_diverter_p, 1),
        min_riser_base_pressure_psi=round(min_riser_base_p, 1),
        total_mud_expelled_bbl=round(total_mud_expelled, 1),
        mgs_blown_through=mgs_blown,
        diverter_rating_exceeded=diverter_breached,
        summary_messages=messages,
        timeline=timeline,
    )
