"""Unit tests for Wellbore Ballooning / Breathing vs. Kick Discriminator Engine."""

import pytest
from wellsim.core.ballooning import (
    BallooningParameters,
    FlowbackDiagnosis,
    diagnose_flowback_event,
    simulate_ballooning_cycle,
)


def test_ballooning_cycle_simulation():
    # 10,000 ft well
    # Closure stress = 6500 psi (~12.5 ppg equivalent)
    # Pore pressure = 5800 psi (~11.15 ppg equivalent)
    # ESD = 6240 psi (12.0 ppg static mud) -> Below closure stress
    # ECD = 6760 psi (13.0 ppg circulating) -> Exceeds closure stress by 260 psi
    params = BallooningParameters(
        closure_stress_psi=6500.0,
        formation_pore_psi=5800.0,
        fracture_compliance_bbl_psi=0.03,  # 0.03 bbl/psi -> 0.03 * 260 = 7.8 bbl max storage
        relaxation_time_s=180.0,
    )

    history = simulate_ballooning_cycle(
        params=params,
        pump_on_duration_s=600.0,  # 10 min circulation
        pump_off_duration_s=900.0,  # 15 min connection
        p_ecd_psi=6760.0,
        p_esd_psi=6240.0,
        dt_s=10.0,
    )

    assert len(history) > 100

    # 1. Pumping phase checks
    pump_on_steps = [s for s in history if s.pumps_on]
    assert len(pump_on_steps) == 60
    assert pump_on_steps[-1].fracture_volume_bbl > 6.0  # Dilated and charged near 7.8 bbl

    # 2. Pump-off flowback checks
    pump_off_steps = [s for s in history if not s.pumps_on]
    assert len(pump_off_steps) == 90

    # Initial flowback right after pumps shut down
    initial_flowback = pump_off_steps[0].flow_rate_gpm
    assert initial_flowback > 20.0  # Significant initial rush of mud returning

    # Mid flowback
    mid_flowback = pump_off_steps[40].flow_rate_gpm
    # Late flowback
    late_flowback = pump_off_steps[-1].flow_rate_gpm

    # Exponential decay verification
    assert initial_flowback > mid_flowback > late_flowback
    assert late_flowback < 5.0  # Decaying near zero

    # Cumulative gain cannot exceed initial charged volume
    final_gain = pump_off_steps[-1].cumulative_gain_bbl
    assert 6.0 <= final_gain <= 8.5


def test_diagnose_benign_ballooning():
    # Decaying return rate from 40 gpm to 1.5 gpm over 10 minutes
    times = [0.0, 120.0, 240.0, 360.0, 480.0, 600.0]
    flows = [42.0, 22.0, 11.0, 5.5, 2.8, 1.2]
    gains = [0.0, 2.5, 4.1, 5.0, 5.4, 5.6]

    res = diagnose_flowback_event(times, flows, gains)
    assert res.diagnosis == FlowbackDiagnosis.BALLOONING
    assert res.confidence_pct >= 80.0
    assert res.flow_decay_rate_gpm_per_min < 0.0
    assert res.asymptotic_volume_bbl < 10.0
    assert "breathing" in res.recommendation.lower()


def test_diagnose_active_formation_kick():
    # Sustained or accelerating return rate: 25 gpm rising to 38 gpm with pressure rebuild
    times = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    flows = [25.0, 28.0, 31.0, 33.0, 36.0, 39.0]
    gains = [0.0, 1.2, 2.6, 4.1, 5.8, 7.6]
    pressures = [50.0, 95.0, 140.0, 185.0, 230.0, 280.0]  # ~46 psi/min rebuild

    res = diagnose_flowback_event(times, flows, gains, shut_in_pressures_psi=pressures)
    assert res.diagnosis == FlowbackDiagnosis.KICK_INFLUX
    assert res.confidence_pct >= 90.0
    assert "SHUT IN" in res.recommendation
    assert res.pressure_rebuild_rate_psi_per_min > 30.0


def test_diagnose_superposition_ballooning_and_kick():
    # Rapid initial drop from 60 gpm, but plateaus at 18 gpm (sustained influx)
    times = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    flows = [60.0, 35.0, 22.0, 19.0, 18.2, 18.0]
    gains = [0.0, 3.0, 5.2, 6.7, 8.2, 9.7]

    res = diagnose_flowback_event(times, flows, gains)
    assert res.diagnosis == FlowbackDiagnosis.BALLOONING_WITH_KICK
    assert res.sustained_flow_gpm > 15.0
    assert "masked" in res.explanation.lower()
