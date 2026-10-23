"""Unit tests for Managed Pressure Cementing & API RP 65 Gas Migration Engine."""

import pytest
from wellsim.core.cementing import (
    CementJobConfig,
    CementSlurry,
    GasMigrationRisk,
    evaluate_api_gas_migration_risk,
    simulate_cementing_u_tube,
)


def test_u_tube_free_fall_dynamics():
    # 10,000 ft 9-5/8" casing, 10.5 ppg mud displaced with 15.8 ppg tail cement
    job = CementJobConfig(
        shoe_md_ft=10000.0,
        mud_density_ppg=10.5,
        spacer_volume_bbl=40.0,
        spacer_density_ppg=12.0,
        tail_volume_bbl=100.0,
        pump_rate_bpm=5.0,
    )

    timeline = simulate_cementing_u_tube(job, dt_bbl=10.0)

    assert len(timeline) > 30

    # 1. Free fall verification: when heavy slurry is in casing, free-fall should trigger
    free_fall_steps = [s for s in timeline if s.is_in_free_fall]
    assert len(free_fall_steps) > 0

    # During free fall, return rate surges above 5.0 bpm surface pump rate
    peak_returns = max(s.return_flow_rate_bpm for s in free_fall_steps)
    assert peak_returns > 5.5

    # During free fall, surface casing pressure is in vacuum (negative or near zero)
    min_surf_p = min(s.surface_casing_pressure_psi for s in free_fall_steps)
    assert min_surf_p < 0.0

    # 2. Later stage: once cement has entered annulus, U-tube pressure recovers
    final_steps = timeline[-3:]
    for st in final_steps:
        assert st.surface_casing_pressure_psi > 100.0
        assert st.is_in_free_fall is False


def test_api_gas_migration_risk_evaluation():
    # 1. Slurry with rapid right-angle set: transition time = 12 minutes
    fast_slurry = CementSlurry(
        name="Rapid Set Class G",
        density_ppg=15.8,
        time_to_100_sgs_min=90.0,
        time_to_500_sgs_min=102.0,  # 12 min transition
    )
    job_fast = CementJobConfig(tail_slurry=fast_slurry, pore_pressure_ppg=11.0)
    rep_fast = evaluate_api_gas_migration_risk(job_fast)

    assert rep_fast.transition_time_min == 12.0
    assert rep_fast.risk_level in (GasMigrationRisk.LOW, GasMigrationRisk.MODERATE)
    assert rep_fast.underbalanced_duration_min < 15.0

    # 2. Slurry with prolonged sluggish gelation: transition time = 55 minutes
    slow_slurry = CementSlurry(
        name="Sluggish Conventional Slurry",
        density_ppg=15.8,
        time_to_100_sgs_min=90.0,
        time_to_500_sgs_min=145.0,  # 55 min transition
    )
    job_slow = CementJobConfig(tail_slurry=slow_slurry, pore_pressure_ppg=12.2)
    rep_slow = evaluate_api_gas_migration_risk(job_slow)

    assert rep_slow.transition_time_min == 55.0
    assert rep_slow.risk_level == GasMigrationRisk.SEVERE
    assert any("backpressure" in r.lower() for r in rep_slow.recommendations)
    assert rep_slow.pressure_loss_at_500_sgs_psi > 200.0
