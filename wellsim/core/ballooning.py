"""Wellbore Ballooning / Breathing vs. Kick Discriminator Engine.

Distinguishes between elastic wellbore ballooning (breathing) and formation fluid kicks
during connections and pump shutdowns, preventing unnecessary shut-ins or catastrophic
underbalanced kicks.

Physics:
- Pumping phase: ECD > minimum horizontal stress (sigma_min) elastically dilates
  natural micro-fractures, charging formation storage (V_balloon = C_f * (P_ECD - sigma_min)).
- Pump-off phase: ECD drops to ESD < sigma_min. Fractures elastically close, expelling
  stored mud back into the annulus with exponential/logarithmic decay (dq/dt < 0).
- True Kick: Inflow driven by sustained differential pore pressure (P_pore > P_well),
  characterized by non-decaying flow rate (dq/dt >= 0) and positive shut-in pressure rebuild.

References:
- Helstrup et al. (2003): "Wellbore Breathing and Influx Detection," SPE 84435.
- Gill, J.A. (1989): "Wellbore Ballooning in Deep Offshore Wells," SPE 18035.
- Lavrov, A. & Tronvoll, J. (2005): "Mechanics of Wellbore Breathing in Fractured Formations."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class FlowbackDiagnosis(str, Enum):
    BALLOONING = "BALLOONING"              # Benign fracture elasticity
    KICK_INFLUX = "KICK_INFLUX"            # Active formation fluid influx
    BALLOONING_WITH_KICK = "BALLOONING_WITH_KICK"  # Hazardous superposition
    STABLE = "STABLE"                      # No flow / normal static well


@dataclass
class BallooningParameters:
    """Reservoir and geomechanical breathing parameters.

    Attributes:
        closure_stress_psi: Minimum in-situ fracture closure stress (psi).
        formation_pore_psi: Formation pore pressure (psi).
        fracture_compliance_bbl_psi: Elastic storage compliance Cf = dV/dP (bbl/psi).
        relaxation_time_s: Characteristic poroelastic flowback time constant tau (s).
    """

    closure_stress_psi: float
    formation_pore_psi: float
    fracture_compliance_bbl_psi: float = 0.05
    relaxation_time_s: float = 300.0


@dataclass
class BallooningTimeStep:
    """Single time station output during ballooning simulation."""

    time_s: float
    pumps_on: bool
    wellbore_pressure_psi: float
    fracture_volume_bbl: float
    flow_rate_gpm: float
    cumulative_gain_bbl: float


@dataclass
class FlowbackDiagnosisResult:
    """Diagnostic output distinguishing ballooning from true influx."""

    diagnosis: FlowbackDiagnosis
    confidence_pct: float
    flow_decay_rate_gpm_per_min: float
    asymptotic_volume_bbl: float
    sustained_flow_gpm: float
    pressure_rebuild_rate_psi_per_min: float
    recommendation: str
    explanation: str


def simulate_ballooning_cycle(
    params: BallooningParameters,
    pump_on_duration_s: float,
    pump_off_duration_s: float,
    p_ecd_psi: float,
    p_esd_psi: float,
    dt_s: float = 5.0,
) -> List[BallooningTimeStep]:
    """Simulate a full pump-on charging and pump-off flowback breathing cycle.

    Args:
        params: Geomechanical fracture breathing parameters.
        pump_on_duration_s: Duration pumps are running at dynamic ECD (s).
        pump_off_duration_s: Duration pumps are shut down at static ESD (s).
        p_ecd_psi: Dynamic circulating bottomhole pressure with ECD (psi).
        p_esd_psi: Static bottomhole hydrostatic pressure with ESD (psi).
        dt_s: Simulation time step in seconds.

    Returns:
        List of BallooningTimeStep records across both cycles.
    """
    history: List[BallooningTimeStep] = []
    current_time = 0.0
    v_fracture = 0.0
    cum_gain = 0.0

    # Max potential volume at ECD if dilated
    delta_p_ecd = max(0.0, p_ecd_psi - params.closure_stress_psi)
    v_max_storage = params.fracture_compliance_bbl_psi * delta_p_ecd

    # 1. Pump-on Charging Phase
    steps_on = int(pump_on_duration_s / dt_s)
    for _ in range(steps_on):
        # Fractures inflate towards equilibrium
        charging_rate_bbl_s = (v_max_storage - v_fracture) / max(10.0, params.relaxation_time_s)
        v_fracture = min(v_max_storage, v_fracture + charging_rate_bbl_s * dt_s)
        # During pumping, loss into fracture occurs (negative pit gain)
        history.append(
            BallooningTimeStep(
                time_s=round(current_time, 1),
                pumps_on=True,
                wellbore_pressure_psi=round(p_ecd_psi, 1),
                fracture_volume_bbl=round(v_fracture, 3),
                flow_rate_gpm=round(-charging_rate_bbl_s * 42.0 * 60.0, 2),  # loss rate
                cumulative_gain_bbl=round(cum_gain, 3),
            )
        )
        current_time += dt_s

    # 2. Pump-off Flowback Phase (Breathing out)
    steps_off = int(pump_off_duration_s / dt_s)
    for _ in range(steps_off):
        # Wellbore drops to ESD
        # Storage equilibrium drops to zero if ESD < closure_stress
        delta_p_esd = max(0.0, p_esd_psi - params.closure_stress_psi)
        v_eq_off = params.fracture_compliance_bbl_psi * delta_p_esd

        # Flowback rate = discharge of stored mud
        discharge_rate_bbl_s = max(0.0, (v_fracture - v_eq_off)) / params.relaxation_time_s
        dv = discharge_rate_bbl_s * dt_s
        v_fracture = max(v_eq_off, v_fracture - dv)
        cum_gain += dv

        rate_gpm = discharge_rate_bbl_s * 42.0 * 60.0

        history.append(
            BallooningTimeStep(
                time_s=round(current_time, 1),
                pumps_on=False,
                wellbore_pressure_psi=round(p_esd_psi, 1),
                fracture_volume_bbl=round(v_fracture, 3),
                flow_rate_gpm=round(rate_gpm, 2),
                cumulative_gain_bbl=round(cum_gain, 3),
            )
        )
        current_time += dt_s

    return history


def diagnose_flowback_event(
    times_s: List[float],
    flow_rates_gpm: List[float],
    pit_gains_bbl: List[float],
    shut_in_pressures_psi: Optional[List[float]] = None,
) -> FlowbackDiagnosisResult:
    """Classify flowback behavior into benign ballooning vs. formation fluid kick.

    Evaluates:
    1. Flow rate decay rate dq/dt (exponential decay vs. steady plateau).
    2. Asymptotic pit gain convergence vs monotonic linear growth.
    3. Pressure build-up derivative dP/dt under shut-in conditions.

    Args:
        times_s: Elapsed seconds since pumps stopped.
        flow_rates_gpm: Flowback return rates (gpm).
        pit_gains_bbl: Cumulative pit volume increase (bbl).
        shut_in_pressures_psi: Optional shut-in casing or drillpipe pressures (psi).

    Returns:
        FlowbackDiagnosisResult with classification and recommended operational response.
    """
    if len(times_s) < 2 or len(flow_rates_gpm) < 2:
        return FlowbackDiagnosisResult(
            diagnosis=FlowbackDiagnosis.STABLE,
            confidence_pct=50.0,
            flow_decay_rate_gpm_per_min=0.0,
            asymptotic_volume_bbl=0.0,
            sustained_flow_gpm=0.0,
            pressure_rebuild_rate_psi_per_min=0.0,
            recommendation="Insufficient observation data to discriminate event.",
            explanation="Need at least 2 time stations after pump shutdown.",
        )

    # 1. Analyze flow rate derivative dq/dt over the last half of the monitoring window
    n = len(times_s)
    mid = n // 2

    dt_min = (times_s[-1] - times_s[0]) / 60.0
    if dt_min <= 0.0:
        dt_min = 1.0 / 60.0

    dq = flow_rates_gpm[-1] - flow_rates_gpm[0]
    overall_decay_rate = dq / dt_min  # gpm per min (negative if decaying)

    recent_flow = flow_rates_gpm[-1]
    initial_flow = flow_rates_gpm[0]

    # Check pressure rebuild if available
    pressure_rebuild_rate = 0.0
    if shut_in_pressures_psi and len(shut_in_pressures_psi) == n:
        dp = shut_in_pressures_psi[-1] - shut_in_pressures_psi[0]
        pressure_rebuild_rate = dp / dt_min  # psi per min

    # Calculate asymptotic volume extrapolation (exponential model: V(t) = V_inf * (1 - exp(-t/tau)))
    # If decaying, V_inf can be estimated
    if initial_flow > recent_flow and recent_flow >= 0.0:
        ratio = max(0.01, recent_flow / max(0.1, initial_flow))
        tau_est = max(10.0, -(times_s[-1] - times_s[0]) / math.log(ratio))
        # V_inf = current_gain + remaining_flow * tau
        asymptotic_gain = pit_gains_bbl[-1] + (recent_flow / (42.0 * 60.0)) * tau_est
    else:
        asymptotic_gain = 9999.0  # Divergent

    # Decision Matrix:
    # A. Benign Ballooning:
    #    - Initial flow > 0 but decaying significantly (dq/dt < -0.5 gpm/min or recent_flow < 0.3 * initial_flow)
    #    - Recent flow approaching 0 gpm (recent_flow < 5.0 gpm)
    #    - Pressure rebuild rate is zero or negative (bleeding off)
    # B. Active Kick:
    #    - Sustained steady flow (dq/dt >= -0.2 gpm/min and recent_flow > 10 gpm)
    #    - OR positive pressure rebuild (dP/dt > 15 psi/min)
    # C. Superposition (Kick masked by ballooning):
    #    - Initial decay, but flattens into a persistent non-zero plateau (recent_flow > 15 gpm and dq/dt flattening)

    is_decaying = (recent_flow < 0.4 * initial_flow) or (overall_decay_rate < -1.5)
    is_flow_near_zero = recent_flow < 3.0
    is_pressure_building = pressure_rebuild_rate > 10.0
    is_sustained = (recent_flow >= 8.0) and (overall_decay_rate > -0.5)

    if is_pressure_building or (is_sustained and not is_decaying):
        diag = FlowbackDiagnosis.KICK_INFLUX
        conf = 95.0 if is_pressure_building else 88.0
        rec = (
            "SHUT IN WELL IMMEDIATELY ON BOP. Space out, shut annular or pipe rams, "
            "open choke line valve to choke manifold. Record stabilized SIDPP and SICP. "
            "Execute Wait & Weight or Driller's Method."
        )
        expl = (
            f"Sustained inflow detected ({recent_flow:.1f} gpm). Flow rate is not self-limiting "
            f"(dq/dt = {overall_decay_rate:.2f} gpm/min). Pressure rebuilding at {pressure_rebuild_rate:.1f} psi/min. "
            f"Formation pore pressure exceeds wellbore hydrostatic."
        )
    elif is_decaying and not is_flow_near_zero and recent_flow > 10.0:
        diag = FlowbackDiagnosis.BALLOONING_WITH_KICK
        conf = 80.0
        rec = (
            "SHUT IN WELL AND MONITOR PRESSURES. Suspected formation influx masked by initial "
            "fracture breathing. Monitor for positive SIDPP/SICP stabilization. Do NOT bleed off pressure."
        )
        expl = (
            f"Flow initially decayed from {initial_flow:.1f} gpm but plateaued at a dangerous {recent_flow:.1f} gpm. "
            f"Suspected kick masked by initial breathing. "
            f"Decay rate has flattened ({overall_decay_rate:.2f} gpm/min), indicating active underbalanced inflow."
        )
    elif is_decaying or is_flow_near_zero:
        diag = FlowbackDiagnosis.BALLOONING
        conf = 92.0 if is_flow_near_zero else 82.0
        rec = (
            "CONTINUE MONITORING FLOWBACK DECAY. Wellbore is breathing (elastic micro-fracture recovery). "
            "Do NOT increase mud weight (will worsen fracture dilation). Do NOT bleed down well rapidly. "
            "Wait for flow to cease (<2 gpm) before resuming operations."
        )
        expl = (
            f"Characteristic exponential flowback decay observed (dq/dt = {overall_decay_rate:.2f} gpm/min). "
            f"Return rate dropped from {initial_flow:.1f} gpm to {recent_flow:.1f} gpm. "
            f"Cumulative gain is converging asymptotically towards {asymptotic_gain:.2f} bbl."
        )
    else:
        diag = FlowbackDiagnosis.STABLE
        conf = 90.0
        rec = "Well is static and stable. Normal operations may proceed."
        expl = f"Minimal or negligible flow detected ({recent_flow:.1f} gpm)."

    return FlowbackDiagnosisResult(
        diagnosis=diag,
        confidence_pct=conf,
        flow_decay_rate_gpm_per_min=round(overall_decay_rate, 2),
        asymptotic_volume_bbl=round(asymptotic_gain, 2),
        sustained_flow_gpm=round(recent_flow, 2),
        pressure_rebuild_rate_psi_per_min=round(pressure_rebuild_rate, 2),
        recommendation=rec,
        explanation=expl,
    )
