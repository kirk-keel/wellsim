"""Transient wellbore kick simulator and rig operations state machine.

Provides a discretized time-domain simulation loop for well control training:
- Influx entry when bottomhole pressure falls below formation pressure
- Gas bubble upward migration (slip velocity) and Boyle's/real-gas expansion
- Dynamic surface pressures: Standpipe Pressure (SPP/SIDPP) and Casing Pressure (CICP/SICP)
- Active pit volume totalizer (PVT) tracking and flow paddle response
- Remote choke valve modeling (orifice flow vs. hydraulic choke position)
- Rig operational state machine: DRILLING, FLOW_CHECK, SHUTTING_IN, SHUT_IN, CIRCULATING_KILL
- Casing shoe integrity monitoring against MAASP / fracture pressure
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from wellsim.core.well_control import (
    PSI_PER_PPG_FT,
    final_circulating_pressure,
    hydrostatic_pressure,
    initial_circulating_pressure,
    kill_mud_weight,
    maasp,
    pressure_gradient,
)

__all__ = [
    "RigState",
    "WellGeometry",
    "UserControls",
    "SimulationState",
    "TransientSimulator",
]


class RigState(str, Enum):
    DRILLING = "DRILLING"
    FLOW_CHECK = "FLOW_CHECK"
    SHUTTING_IN = "SHUTTING_IN"
    SHUT_IN = "SHUT_IN"
    CIRCULATING_KILL = "CIRCULATING_KILL"


@dataclass(frozen=True)
class WellGeometry:
    tvd_well_ft: float = 10000.0
    md_well_ft: float = 10000.0
    tvd_shoe_ft: float = 6000.0
    hole_id_in: float = 8.5
    pipe_od_in: float = 5.0
    pipe_id_in: float = 4.276
    drillstring_volume_bbl: float = 150.0
    annular_volume_bbl: float = 850.0
    pump_displacement_bbl_per_stk: float = 0.10

    @property
    def annular_capacity_bbl_ft(self) -> float:
        """Annular capacity in bbl/ft: (D_hole^2 - D_pipe^2) / 1029.4."""
        return (self.hole_id_in**2 - self.pipe_od_in**2) / 1029.4

    @property
    def pipe_capacity_bbl_ft(self) -> float:
        """Internal pipe capacity in bbl/ft: D_pipe_id^2 / 1029.4."""
        return (self.pipe_id_in**2) / 1029.4


@dataclass
class UserControls:
    pump_spm: float = 0.0  # Strokes Per Minute (0 - 150)
    choke_pct: float = 100.0  # Choke opening percentage (0% = fully closed, 100% = fully open)
    bop_closed: bool = False
    hcr_open: bool = False
    spaced_out: bool = False
    kill_mud_active: bool = False  # Pumping kill mud instead of original mud
    auto_choke: bool = False  # Automated hydraulic choke follower
    auto_choke_target_psi: float = 750.0  # Target pressure setpoint


@dataclass
class SimulationState:
    time_s: float = 0.0
    rig_state: RigState = RigState.DRILLING
    controls: UserControls = field(default_factory=UserControls)

    # Pressures (psi)
    spp_psi: float = 0.0
    casing_pressure_psi: float = 0.0
    bhp_psi: float = 0.0
    shoe_pressure_psi: float = 0.0
    maasp_psi: float = 0.0

    # Volumes & Rates
    pit_volume_bbl: float = 500.0
    pit_gain_bbl: float = 0.0
    flow_out_pct: float = 0.0
    pump_rate_gpm: float = 0.0
    total_pump_strokes: int = 0

    # Influx / Gas Tracking
    influx_volume_bbl: float = 0.0
    influx_top_md: float = 0.0
    influx_bottom_md: float = 0.0
    gas_at_surface: bool = False
    gas_cleared: bool = False
    shoe_fractured: bool = False

    # Multi-fluid displacement tracking
    kill_mud_in_drillpipe_bbl: float = 0.0
    kill_mud_in_annulus_bbl: float = 0.0
    kill_mud_front_md: float = 0.0
    circulation_phase: str = "DRILLING"

    # Alarms / Messages
    active_alarms: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 1),
            "rig_state": self.rig_state.value,
            "spp_psi": round(self.spp_psi, 1),
            "casing_pressure_psi": round(self.casing_pressure_psi, 1),
            "bhp_psi": round(self.bhp_psi, 1),
            "shoe_pressure_psi": round(self.shoe_pressure_psi, 1),
            "maasp_psi": round(self.maasp_psi, 1),
            "pit_volume_bbl": round(self.pit_volume_bbl, 1),
            "pit_gain_bbl": round(self.pit_gain_bbl, 2),
            "flow_out_pct": round(self.flow_out_pct, 1),
            "pump_spm": round(self.controls.pump_spm, 1),
            "pump_rate_gpm": round(self.pump_rate_gpm, 1),
            "choke_pct": round(self.controls.choke_pct, 1),
            "total_pump_strokes": self.total_pump_strokes,
            "influx_volume_bbl": round(self.influx_volume_bbl, 2),
            "influx_top_md": round(self.influx_top_md, 1),
            "gas_at_surface": self.gas_at_surface,
            "gas_cleared": self.gas_cleared,
            "kill_mud_in_drillpipe_bbl": round(self.kill_mud_in_drillpipe_bbl, 1),
            "kill_mud_in_annulus_bbl": round(self.kill_mud_in_annulus_bbl, 1),
            "circulation_phase": self.circulation_phase,
            "shoe_fractured": self.shoe_fractured,
            "active_alarms": list(self.active_alarms),
        }


class TransientSimulator:
    """Discretized transient simulator engine for well control scenarios."""

    def __init__(
        self,
        geom: WellGeometry | None = None,
        omw_ppg: float = 10.0,
        formation_pressure_psi: float = 5720.0,
        lot_emw_ppg: float = 14.0,
        scrp_psi: float = 600.0,
        scr_spm: float = 30.0,
        gas_gradient_psi_ft: float = 0.10,
        bubble_slip_velocity_fps: float = 0.28,  # ~1000 ft/hr natural gas migration
    ):
        self.geom = geom or WellGeometry()
        self.omw_ppg = omw_ppg
        self.kmw_ppg = omw_ppg + (formation_pressure_psi - hydrostatic_pressure(omw_ppg, self.geom.tvd_well_ft)) / (
            PSI_PER_PPG_FT * self.geom.tvd_well_ft
        )
        self.formation_pressure_psi = formation_pressure_psi
        self.lot_emw_ppg = lot_emw_ppg
        self.scrp_psi = scrp_psi
        self.scr_spm = scr_spm
        self.gas_gradient_psi_ft = gas_gradient_psi_ft
        self.bubble_slip_velocity_fps = bubble_slip_velocity_fps

        # Circulating friction constant at normal drilling rate (e.g. 100 SPM = ~2500 psi SPP)
        self.drillstring_friction_factor = 2500.0 / (100.0**1.8)
        self.annular_friction_factor = 200.0 / (100.0**1.8)

        # Baseline pit volume
        self.initial_pit_volume_bbl = 500.0
        self._choke_pid_integral = 0.0
        self._choke_pid_prev_error = 0.0

        # Current state
        self.state = SimulationState(
            time_s=0.0,
            rig_state=RigState.DRILLING,
            controls=UserControls(pump_spm=100.0, choke_pct=100.0, bop_closed=False, hcr_open=False),
            pit_volume_bbl=self.initial_pit_volume_bbl,
            maasp_psi=maasp(self.lot_emw_ppg, self.omw_ppg, self.geom.tvd_shoe_ft),
            influx_top_md=self.geom.md_well_ft,
            influx_bottom_md=self.geom.md_well_ft,
        )

    def step(self, dt: float = 1.0, controls: UserControls | None = None) -> SimulationState:
        """Advance simulation by dt seconds with user control inputs."""
        if controls is not None:
            self.state.controls = controls

        c = self.state.controls
        t = self.state.time_s + dt
        self.state.time_s = t
        self.state.active_alarms.clear()

        # 1. State Machine Transitions
        self._update_rig_state()

        # 2. Pump Hydraulics
        pump_spm = c.pump_spm
        pump_gpm = pump_spm * self.geom.pump_displacement_bbl_per_stk * 42.0  # bbl/min * 42 = gpm
        stk_inc = (pump_spm / 60.0) * dt
        self.state.total_pump_strokes += int(math.floor(stk_inc)) if (t % 1.0 < dt) else 0
        self.state.pump_rate_gpm = pump_gpm

        # Fluid velocity in annulus (ft/s)
        q_ann_bbl_s = (pump_spm / 60.0) * self.geom.pump_displacement_bbl_per_stk
        v_ann_fluid_fps = q_ann_bbl_s / self.geom.annular_capacity_bbl_ft if self.geom.annular_capacity_bbl_ft > 0 else 0.0

        # Circulating friction losses
        dp_friction_ds = self.drillstring_friction_factor * (pump_spm**1.8) if pump_spm > 0 else 0.0
        dp_friction_ann = self.annular_friction_factor * (pump_spm**1.8) if (pump_spm > 0 and not c.bop_closed) else 0.0

        # 3. Choke Restriction / Wellhead Backpressure
        # Automated Choke PID controller
        if c.auto_choke and (c.bop_closed or c.hcr_open):
            error = self.state.casing_pressure_psi - c.auto_choke_target_psi
            self._choke_pid_integral = max(-500.0, min(500.0, self._choke_pid_integral + error * dt))
            d_error = (error - self._choke_pid_prev_error) / max(0.01, dt)
            self._choke_pid_prev_error = error
            adj = 0.04 * error + 0.005 * self._choke_pid_integral + 0.01 * d_error
            c.choke_pct = max(0.0, min(100.0, c.choke_pct + adj))

        # Effective choke backpressure: when open (100%), dp=0; when 0% (closed), full shut-in pressure
        choke_frac = max(0.0, min(100.0, c.choke_pct)) / 100.0
        choke_loss_psi = 0.0
        if pump_gpm > 0 and choke_frac < 1.0:
            # Flowing choke backpressure
            choke_loss_psi = (pump_gpm / (100.0 * max(choke_frac, 0.01))) ** 2

        # 4. Inflow & Kick Generation at Bit
        current_mw = self.kmw_ppg if c.kill_mud_active else self.omw_ppg
        p_hyd_ann = hydrostatic_pressure(current_mw, self.geom.tvd_well_ft)

        # Influx height and hydrostatic reduction
        h_influx = max(0.0, self.state.influx_bottom_md - self.state.influx_top_md)
        if h_influx > 0:
            delta_hyd = (pressure_gradient(current_mw) - self.gas_gradient_psi_ft) * h_influx
            p_hyd_ann -= max(0.0, delta_hyd)

        # Bottomhole Pressure
        if c.bop_closed:
            # Well is shut in or flowing through choke
            if self.state.rig_state == RigState.SHUT_IN:
                # Stabilized to formation pressure
                bhp = self.formation_pressure_psi
                casing_p = max(0.0, bhp - p_hyd_ann)
                spp = max(0.0, bhp - hydrostatic_pressure(current_mw, self.geom.tvd_well_ft))
            else:
                # Circulating through choke
                casing_p = choke_loss_psi + (self.formation_pressure_psi - p_hyd_ann) * (1.0 - choke_frac)
                bhp = p_hyd_ann + casing_p + dp_friction_ann
                spp = (bhp - hydrostatic_pressure(current_mw, self.geom.tvd_well_ft)) + dp_friction_ds
        else:
            # Open wellbore to pits
            casing_p = choke_loss_psi
            bhp = p_hyd_ann + dp_friction_ann + casing_p
            spp = dp_friction_ds + dp_friction_ann

        # Formation Influx Check (when underbalanced)
        if bhp < self.formation_pressure_psi and not c.bop_closed:
            # Inflow rate proportional to drawdown: q = PI * (P_form - BHP)
            pi_bbl_min = 0.05  # productivity index
            drawdown = self.formation_pressure_psi - bhp
            influx_rate_bbl_s = (pi_bbl_min * drawdown) / 60.0
            new_influx_bbl = influx_rate_bbl_s * dt
            self.state.influx_volume_bbl += new_influx_bbl
            self.state.pit_gain_bbl += new_influx_bbl
            self.state.influx_bottom_md = self.geom.md_well_ft
            self.state.influx_top_md = max(
                0.0,
                self.geom.md_well_ft - (self.state.influx_volume_bbl / self.geom.annular_capacity_bbl_ft),
            )

        # 5. Kick Migration and Expansion
        if self.state.influx_volume_bbl > 0 and self.state.influx_top_md > 0:
            # Migration speed: fluid speed + natural bubble slip
            v_mig_fps = (v_ann_fluid_fps if not c.bop_closed else 0.0) + self.bubble_slip_velocity_fps
            dz_mig = v_mig_fps * dt
            self.state.influx_top_md = max(0.0, self.state.influx_top_md - dz_mig)
            self.state.influx_bottom_md = max(0.0, self.state.influx_bottom_md - dz_mig)

            # Gas expansion as bubble nears surface (Boyle's law: P1 V1 = P2 V2)
            if self.state.influx_top_md < 3000.0 and not c.bop_closed:
                expansion_factor = 1.0 + (3000.0 - self.state.influx_top_md) / 10000.0 * 0.05 * dt
                expanded_vol = self.state.influx_volume_bbl * expansion_factor
                dV = expanded_vol - self.state.influx_volume_bbl
                self.state.influx_volume_bbl = expanded_vol
                self.state.pit_gain_bbl += dV

            if self.state.influx_top_md <= 0.0:
                self.state.gas_at_surface = True
                if choke_frac > 0.0 and not self.state.gas_cleared:
                    vent_rate = (2.0 * choke_frac) * dt
                    self.state.influx_volume_bbl = max(0.0, self.state.influx_volume_bbl - vent_rate)
                    if self.state.influx_volume_bbl <= 0.05:
                        self.state.influx_volume_bbl = 0.0
                        self.state.gas_at_surface = False
                        self.state.gas_cleared = True
                        self.state.active_alarms.append("INFLUX CLEARED: All formation gas circulated out of wellbore.")
                else:
                    self.state.active_alarms.append("GAS AT SURFACE: High pressure gas arriving at surface choke!")

        # Multi-fluid displacement tracking
        if c.kill_mud_active and pump_spm > 0:
            pumped_bbl = q_ann_bbl_s * dt
            if self.state.kill_mud_in_drillpipe_bbl < self.geom.drillstring_volume_bbl:
                self.state.circulation_phase = "CIRCULATION_2_KILL_MUD (DISPLACING STRING)"
                self.state.kill_mud_in_drillpipe_bbl = min(
                    self.geom.drillstring_volume_bbl,
                    self.state.kill_mud_in_drillpipe_bbl + pumped_bbl,
                )
            else:
                self.state.kill_mud_in_annulus_bbl = min(
                    self.geom.annular_volume_bbl,
                    self.state.kill_mud_in_annulus_bbl + pumped_bbl,
                )
                if self.state.kill_mud_in_annulus_bbl >= self.geom.annular_volume_bbl:
                    self.state.circulation_phase = "WELL_KILLED"
                    self.state.active_alarms.append("WELL KILLED: Kill mud circulated completely to surface!")
                else:
                    self.state.circulation_phase = "CIRCULATION_2_KILL_MUD (DISPLACING ANNULUS)"
        elif self.state.rig_state == RigState.CIRCULATING_KILL and not self.state.gas_cleared:
            self.state.circulation_phase = "CIRCULATION_1_GAS_REMOVAL"

        # 6. Pit Volume and Flow Out
        self.state.pit_volume_bbl = self.initial_pit_volume_bbl + self.state.pit_gain_bbl
        if not c.bop_closed:
            # Flow paddle sensor
            base_flow = (pump_spm / 100.0) * 100.0  # 100% flow at 100 SPM
            inflow_flow = (self.state.pit_gain_bbl / max(1.0, dt)) * 10.0
            self.state.flow_out_pct = min(150.0, base_flow + inflow_flow)
        else:
            self.state.flow_out_pct = 0.0

        # Alarms
        if self.state.pit_gain_bbl >= 10.0 and self.state.rig_state == RigState.DRILLING:
            self.state.active_alarms.append(f"PIT GAIN ALARM: +{self.state.pit_gain_bbl:.1f} bbl gain detected!")

        # 7. Shoe Pressure & MAASP Monitoring
        shoe_tvd = self.geom.tvd_shoe_ft
        p_hyd_above_shoe = hydrostatic_pressure(current_mw, shoe_tvd)
        # If gas has passed above the shoe, shoe pressure increases
        shoe_p = casing_p + p_hyd_above_shoe
        self.state.shoe_pressure_psi = shoe_p
        self.state.maasp_psi = maasp(self.lot_emw_ppg, current_mw, shoe_tvd)

        if shoe_p > (self.state.maasp_psi + hydrostatic_pressure(self.omw_ppg, shoe_tvd)):
            self.state.shoe_fractured = True
            self.state.active_alarms.append("FORMATION FRACTURE: Casing shoe fracture pressure exceeded!")

        # Set output variables
        self.state.bhp_psi = bhp
        self.state.spp_psi = spp
        self.state.casing_pressure_psi = casing_p

        return self.state

    def _update_rig_state(self) -> None:
        """Evaluate procedural transitions between rig operational states."""
        c = self.state.controls
        curr = self.state.rig_state

        if curr == RigState.DRILLING:
            if c.pump_spm == 0.0 and not c.bop_closed:
                self.state.rig_state = RigState.FLOW_CHECK
            elif c.bop_closed:
                self.state.rig_state = RigState.SHUT_IN

        elif curr == RigState.FLOW_CHECK:
            if c.bop_closed:
                self.state.rig_state = RigState.SHUT_IN
            elif c.pump_spm > 0:
                self.state.rig_state = RigState.DRILLING

        elif curr == RigState.SHUT_IN:
            if not c.bop_closed:
                self.state.rig_state = RigState.FLOW_CHECK
            elif c.pump_spm > 0 and c.hcr_open and c.choke_pct > 0:
                self.state.rig_state = RigState.CIRCULATING_KILL

        elif curr == RigState.CIRCULATING_KILL:
            if c.pump_spm == 0:
                self.state.rig_state = RigState.SHUT_IN
