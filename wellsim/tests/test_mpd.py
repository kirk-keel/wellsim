"""Unit tests for Managed Pressure Drilling (MPD) and Deepwater Subsea Choke Line Hydraulics."""

import pytest

from wellsim.core.mpd import (
    CoriolisFlowMeter,
    MPDController,
    SubseaChokeLineModel,
)


def test_coriolis_micro_flux_detection():
    meter = CoriolisFlowMeter(influx_threshold_bbl=1.0)

    # 1. Steady balanced drilling: 400 gpm in, 400 gpm out
    df, alarms = meter.update(flow_in_gpm=400.0, flow_out_gpm=400.0, dt_s=5.0)
    assert df == 0.0
    assert meter.cumulative_flux_bbl == 0.0
    assert len(alarms) == 0

    # 2. Influx starts: 400 gpm in, 450 gpm out (+50 gpm delta)
    # After 30 seconds: 50 gpm * (30/60) / 42 = 0.595 bbl
    df, alarms = meter.update(flow_in_gpm=400.0, flow_out_gpm=450.0, dt_s=30.0)
    assert df == 50.0
    assert abs(meter.cumulative_flux_bbl - 0.595) < 0.01
    assert len(alarms) == 0  # Under 1.0 bbl

    # Another 30 seconds (+0.595 bbl -> 1.19 bbl > 1.0 bbl threshold)
    df, alarms = meter.update(flow_in_gpm=400.0, flow_out_gpm=450.0, dt_s=30.0)
    assert meter.cumulative_flux_bbl > 1.0
    assert len(alarms) == 1
    assert "MPD MICRO-FLUX DETECTED" in alarms[0]


def test_mpd_cbhp_backpressure_regulation():
    # Target BHP: 6200 psi
    # Hydrostatic: 5800 psi
    # Full pump rate annular friction: 400 psi -> Required SBP = 0 psi
    # Connection / pump off (0 gpm): Annular friction: 0 psi -> Required SBP = 400 psi!
    controller = MPDController(target_bhp_psi=6200.0)

    # Calculate theoretical target SBP when pumps shut down
    target_sbp = controller.compute_sbp_target(
        hydrostatic_ann_psi=5800.0,
        annular_friction_loss_psi=0.0,
    )
    assert target_sbp == 400.0

    # Test PID loop convergence
    current_bhp = 5800.0  # Currently 400 psi under target
    for _ in range(30):
        sbp = controller.step(current_bhp_psi=current_bhp, dt_s=1.0)
        # In-situ BHP = Hydrostatic (5800) + SBP
        current_bhp = 5800.0 + sbp

    # Should converge to within 15 psi of target 6200 psi
    assert abs(current_bhp - 6200.0) < 15.0


def test_deepwater_subsea_choke_line_friction():
    subsea = SubseaChokeLineModel(
        water_depth_ft=6000.0,
        choke_line_id_in=3.0,
        dual_lines_parallel=False,
    )

    # Circulate 400 gpm of 12.0 ppg mud up the single 3.0" choke line
    loss_single = subsea.calculate_choke_line_friction_loss(
        pump_rate_gpm=400.0,
        mud_weight_ppg=12.0,
        pv_cp=25.0,
        yp_lbf_100ft2=15.0,
    )
    assert loss_single > 250.0  # Significant choke line backpressure!

    # Now open both choke and kill lines in parallel
    subsea_dual = SubseaChokeLineModel(
        water_depth_ft=6000.0,
        choke_line_id_in=3.0,
        dual_lines_parallel=True,
    )
    loss_dual = subsea_dual.calculate_choke_line_friction_loss(
        pump_rate_gpm=400.0,
        mud_weight_ppg=12.0,
        pv_cp=25.0,
        yp_lbf_100ft2=15.0,
    )
    # Splitting flow in half should drop friction loss by ~60-75%
    assert loss_dual < loss_single * 0.5

    # Check shoe pressure includes the choke line backpressure
    shoe_p = subsea.effective_shoe_pressure(
        shoe_tvd_ft=10000.0,
        mud_weight_ppg=12.0,
        annular_friction_psi=150.0,
        choke_line_friction_psi=loss_single,
        surface_choke_psi=50.0,
    )
    p_hyd_shoe = 12.0 * 0.052 * 10000.0  # 6240 psi
    assert shoe_p == p_hyd_shoe + 150.0 + loss_single + 50.0
