"""Unit tests for Dynamic Surge & Swab Tripping Engine."""

import pytest

from wellsim.core.surge_swab import (
    PipeCondition,
    calculate_surge_swab_gradient,
    calculate_trip_envelope,
)


def test_surge_gradient_closed_vs_open():
    hole_id = 8.5
    pipe_od = 5.0
    pipe_id = 4.276
    rho = 12.0
    pv = 25.0
    yp = 15.0
    speed = 90.0  # 90 ft/min = 1 stand/min

    # Closed-ended (float valve closed)
    grad_closed = calculate_surge_swab_gradient(
        trip_speed_ft_min=speed,
        hole_id_in=hole_id,
        pipe_od_in=pipe_od,
        pipe_id_in=pipe_id,
        rho_ppg=rho,
        pv_cp=pv,
        yp_lbf_100ft2=yp,
        pipe_condition=PipeCondition.CLOSED_ENDED,
    )

    # Open-ended (mud can flow through bore)
    grad_open = calculate_surge_swab_gradient(
        trip_speed_ft_min=speed,
        hole_id_in=hole_id,
        pipe_od_in=pipe_od,
        pipe_id_in=pipe_id,
        rho_ppg=rho,
        pv_cp=pv,
        yp_lbf_100ft2=yp,
        pipe_condition=PipeCondition.OPEN_ENDED,
    )

    assert grad_closed > 0.0
    assert grad_open > 0.0
    # Closed-ended must produce higher annular displacement and surge gradient than open-ended
    assert grad_closed > grad_open


def test_trip_envelope_safety_margins():
    tvd_well = 12000.0
    shoe_tvd = 9000.0
    hole_id = 8.5
    pipe_od = 5.0
    pipe_id = 4.276
    rho = 12.0
    pv = 28.0
    yp = 16.0
    # Narrow drilling margin (typical deepwater/HPHT pore-fracture window)
    pore_p = 11.2  # 0.8 ppg trip margin
    frac_p = 12.8  # 0.8 ppg fracture margin

    max_rih, max_pooh, profile = calculate_trip_envelope(
        tvd_well_ft=tvd_well,
        shoe_tvd_ft=shoe_tvd,
        hole_id_in=hole_id,
        pipe_od_in=pipe_od,
        pipe_id_in=pipe_id,
        rho_ppg=rho,
        pv_cp=pv,
        yp_lbf_100ft2=yp,
        pore_pressure_ppg=pore_p,
        frac_gradient_ppg=frac_p,
        pipe_condition=PipeCondition.CLOSED_ENDED,
        safety_margin_psi=30.0,
    )

    # In a narrow window, speed limits MUST be strictly throttled to safe operating speeds
    assert 20.0 <= max_rih <= 450.0
    assert 20.0 <= max_pooh <= 450.0
    assert len(profile) == 4

    # Check shoe station
    shoe_stat = profile[1]
    assert shoe_stat.depth_ft == shoe_tvd
    # Surge increases ECD, swab decreases ECD
    assert shoe_stat.ecd_surge_ppg > rho
    assert shoe_stat.ecd_swab_ppg < rho
    assert shoe_stat.margin_to_frac_psi > 0.0
    assert shoe_stat.margin_to_pore_psi > 0.0
