"""Unit tests for Real-Time Unscented Kalman Filter (UKF) Digital Twin Engine."""

import math
import pytest
from wellsim.core.kalman import (
    DrillingControlInput,
    DrillingObservation,
    WellboreUKF,
    _cholesky,
    _invert_matrix,
)


def test_matrix_math_cholesky_and_inversion():
    # 3x3 Positive definite symmetric matrix
    a = [
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 3.0],
        [1.0, 3.0, 6.0],
    ]

    l_mat = _cholesky(a)
    # Reconstruct A = L * L^T
    a_rec = [[sum(l_mat[i][k] * l_mat[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            assert pytest.approx(a[i][j], 1e-4) == a_rec[i][j]

    # Inversion check: A * A^(-1) = I
    a_inv = _invert_matrix(a)
    ident = [[sum(a[i][k] * a_inv[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert pytest.approx(ident[i][j], 1e-4) == expected


def test_ukf_steady_drilling_convergence():
    # Normal drilling at 10,000 ft with 12 ppg mud (hydrostatic = 6240 psi)
    ukf = WellboreUKF(
        initial_pore_pressure_psi=6000.0,  # Slight initial offset
        initial_influx_volume_bbl=0.0,
        initial_friction_mult=1.0,
    )

    ctrl = DrillingControlInput(
        pump_spm=100.0,
        choke_pct=100.0,
        well_tvd_ft=10000.0,
        mud_weight_ppg=12.0,
    )

    # Steady normal drilling telemetry with zero influx
    obs = DrillingObservation(
        spp_psi=1200.0,
        casing_pressure_psi=0.0,
        pit_gain_bbl=0.0,
        flow_out_pct=100.0,
    )

    # Run 10 filter cycles
    initial_cov = ukf.p[0][0]
    for _ in range(10):
        est = ukf.filter_step(ctrl, obs, dt_s=1.0)

    # Influx volume should remain ~0
    assert est.influx_volume_bbl < 0.2
    assert est.is_kick_detected is False
    # Filter uncertainty (covariance) for friction and kick volume must decrease as telemetry is assimilated
    final_fric_cov = ukf.p[2][2]
    assert final_fric_cov < 0.2**2
    assert est.friction_std < 0.1


def test_ukf_dynamic_kick_detection_and_tracking():
    # Wellbore drilling in balanced condition
    ukf = WellboreUKF(
        initial_pore_pressure_psi=6240.0,
        initial_influx_volume_bbl=0.0,
        initial_friction_mult=1.0,
    )

    ctrl = DrillingControlInput(
        pump_spm=100.0,
        choke_pct=100.0,
        well_tvd_ft=10000.0,
        mud_weight_ppg=12.0,
    )

    # Simulated sudden kick: pit gain rises from 1 to 8 bbl, flow out surges to 125%
    noisy_telemetry = [
        DrillingObservation(spp_psi=1190.0, casing_pressure_psi=20.0, pit_gain_bbl=1.2, flow_out_pct=106.0),
        DrillingObservation(spp_psi=1175.0, casing_pressure_psi=45.0, pit_gain_bbl=2.5, flow_out_pct=112.0),
        DrillingObservation(spp_psi=1160.0, casing_pressure_psi=75.0, pit_gain_bbl=4.3, flow_out_pct=119.0),
        DrillingObservation(spp_psi=1140.0, casing_pressure_psi=110.0, pit_gain_bbl=6.4, flow_out_pct=126.0),
        DrillingObservation(spp_psi=1120.0, casing_pressure_psi=145.0, pit_gain_bbl=8.2, flow_out_pct=132.0),
    ]

    estimates = []
    for obs in noisy_telemetry:
        est = ukf.filter_step(ctrl, obs, dt_s=2.0)
        estimates.append(est)

    final_est = estimates[-1]

    # UKF should filter sensor noise and track downhole influx volume (~5-9 bbl)
    assert 4.0 <= final_est.influx_volume_bbl <= 10.0
    # Kick flag must be triggered
    assert final_est.is_kick_detected is True
    # Inferred pore pressure should be above wellbore hydrostatic (>6240 psi)
    assert final_est.pore_pressure_psi >= 6240.0
