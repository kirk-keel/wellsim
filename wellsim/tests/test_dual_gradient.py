"""Unit tests for Dual-Gradient Drilling (DGD) & Subsea MudLift Pump (MLP) Engine."""

import pytest
from wellsim.core.dual_gradient import (
    DGDSystemConfig,
    calculate_u_tube_imbalance,
    compare_dgd_vs_conventional_profiles,
    evaluate_mlp_operating_state,
)


def test_mlp_operating_state_and_hydraulics():
    # 6,000 ft water depth, 14.5 ppg heavy mud, 650 gpm circulation
    config = DGDSystemConfig(
        water_depth_ft=6000.0,
        seawater_density_ppg=8.55,
        mud_density_ppg=14.5,
        mrl_id_in=6.0,
        mlp_max_discharge_psi=6000.0,
        mlp_max_rate_gpm=1200.0,
    )

    # 1. Mudline suction pressure check: P_sw = 0.052 * 8.55 * 6000 = 2667.6 psi
    assert pytest.approx(config.seawater_hydrostatic_at_mudline_psi, 1.0) == 2667.6

    mlp_state = evaluate_mlp_operating_state(config, flow_rate_gpm=650.0)

    assert mlp_state.suction_pressure_psi == 2667.6
    # Discharge pressure must lift 14.5 ppg mud 6000 ft + friction:
    # 0.052 * 14.5 * 6000 = 4524 psi + friction (~20-100 psi)
    assert 4524.0 <= mlp_state.discharge_pressure_psi <= 4800.0

    # Pump differential pressure
    delta_p = mlp_state.discharge_pressure_psi - mlp_state.suction_pressure_psi
    assert pytest.approx(mlp_state.pump_differential_pressure_psi, 0.1) == delta_p
    assert 1800.0 <= delta_p <= 2100.0

    # Hydraulic horsepower: (delta_p * 650) / 1714 ~ 700 to 800 HHP
    assert 650.0 <= mlp_state.hydraulic_horsepower_hhp <= 850.0
    assert mlp_state.is_within_operating_limits is True


def test_u_tube_imbalance_and_dsv_sizing():
    # 6,000 ft WD, 14.5 ppg DP mud, 8.55 ppg seawater
    config = DGDSystemConfig(
        water_depth_ft=6000.0,
        seawater_density_ppg=8.55,
        mud_density_ppg=14.5,
    )

    imbalance = calculate_u_tube_imbalance(config)

    # Delta P = (14.5 - 8.55) * 0.052 * 6000 = 5.95 * 312 = 1856.4 psi
    assert pytest.approx(imbalance["u_tube_differential_psi"], 1.0) == 1856.4

    # Unbalanced fluid head drop in drillpipe: 1856.4 / (0.052 * 14.5) ~ 2462 ft
    assert 2400.0 <= imbalance["drillstring_head_drop_ft"] <= 2500.0

    # DSV cracking pressure should be set slightly above U-tube differential (~2000 psi)
    assert imbalance["recommended_dsv_cracking_pressure_psi"] > 1856.4


def test_dgd_expands_deepwater_drilling_margin():
    # Demonstrates how DGD solves the narrow deepwater drilling margin
    config = DGDSystemConfig(
        water_depth_ft=6000.0,
        total_depth_ft=16000.0,
        seawater_density_ppg=8.55,
        mud_density_ppg=14.0,
    )

    # Realistic deepwater pore and fracture curves originating from mudline
    pore_curve = [
        (6000.0, 8.55),   # At mudline: seawater pressure
        (8000.0, 9.8),
        (12000.0, 11.8),
        (16000.0, 13.5),  # At TD: 13.5 ppg pore pressure
    ]
    frac_curve = [
        (6000.0, 10.5),  # At mudline: low fracture gradient (10.5 ppg)
        (8000.0, 12.0),
        (12000.0, 13.8),
        (16000.0, 15.2), # At TD: 15.2 ppg fracture gradient
    ]

    profile = compare_dgd_vs_conventional_profiles(
        config=config,
        pore_pressures=pore_curve,
        frac_gradients=frac_curve,
        flow_rate_gpm=500.0,
    )

    # Station at 8,000 ft (shallow casing seat below mudline):
    st_8k = [s for s in profile if s.depth_ft == 8000.0][0]

    # Conventional: single 14.0 ppg mud weight from surface yields ECD ~14.1 ppg,
    # which exceeds the 12.0 ppg fracture gradient -> Severe Formation Breakdown!
    assert st_8k.ecd_conventional_ppg > st_8k.fracture_gradient_ppg
    assert st_8k.conventional_in_window is False

    # DGD: Seawater in riser + 14.0 ppg below mudline yields ECD ~10.0 ppg,
    # which sits comfortably between pore (9.8 ppg) and frac (12.0 ppg)!
    assert st_8k.pore_pressure_ppg <= st_8k.ecd_dgd_ppg <= st_8k.fracture_gradient_ppg
    assert st_8k.dgd_in_window is True
