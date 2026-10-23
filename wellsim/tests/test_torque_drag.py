"""Unit tests for 3D Torque & Drag and Drillstring Buckling Engine."""

import pytest

from wellsim.core.survey import SurveyStation, calculate_minimum_curvature
from wellsim.core.torque_drag import (
    PipeSection,
    calculate_torque_and_drag,
)


def test_torque_drag_pooh_vs_rih_hookloads():
    # 10,000 ft directional well with 45 deg build
    stations = [
        SurveyStation(0.0, 0.0, 0.0),
        SurveyStation(3000.0, 0.0, 0.0),
        SurveyStation(5000.0, 45.0, 45.0),
        SurveyStation(10000.0, 45.0, 45.0),
    ]
    traj = calculate_minimum_curvature(stations)

    # Drillstring: 10,000 ft of 5.0" 19.5 lbf/ft drillpipe
    ds = [PipeSection(length_ft=10000.0, od_in=5.0, id_in=4.276, weight_air_lbf_ft=19.5)]

    res = calculate_torque_and_drag(
        trajectory=traj,
        drillstring=ds,
        hole_diameter_in=8.5,
        mud_weight_ppg=12.0,
        friction_coefficient=0.25,
    )

    # Fundamental physical law of drilling hookloads:
    # POOH (pulling string + friction) > Rotating (pure string weight) > RIH (string weight - friction)
    assert res.hookload_pooh_klbf > res.hookload_rotating_klbf
    assert res.hookload_rotating_klbf > res.hookload_rih_klbf

    # Positive surface torque from string rotation in inclined hole
    assert res.surface_torque_ft_lbf > 500.0
    assert len(res.stations) > 0


def test_buckling_limits_and_lockup():
    stations = [
        SurveyStation(0.0, 0.0, 0.0),
        SurveyStation(2000.0, 0.0, 0.0),
        SurveyStation(4000.0, 75.0, 0.0),   # High angle 75 deg
        SurveyStation(8000.0, 75.0, 0.0),
    ]
    traj = calculate_minimum_curvature(stations)
    ds = [PipeSection(length_ft=8000.0, od_in=5.0, id_in=4.276, weight_air_lbf_ft=19.5)]

    # 1. Normal drilling: 15,000 lbf WOB (no lock-up)
    res_normal = calculate_torque_and_drag(
        trajectory=traj,
        drillstring=ds,
        hole_diameter_in=8.5,
        mud_weight_ppg=12.0,
        friction_coefficient=0.25,
        wob_lbf=15000.0,
    )
    # Sinusoidal buckling limit should be positive in inclined section
    assert res_normal.stations[-1].sinusoidal_buckling_limit_lbf > 1000.0
    assert res_normal.stations[-1].helical_buckling_limit_lbf > res_normal.stations[-1].sinusoidal_buckling_limit_lbf
