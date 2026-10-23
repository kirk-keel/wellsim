"""Unit tests for Minimum Curvature 3D Directional Survey Engine."""

import math
import pytest

from wellsim.core.survey import (
    SurveyStation,
    calculate_minimum_curvature,
)


def test_vertical_well_trajectory():
    stations = [
        SurveyStation(md_ft=0.0, inclination_deg=0.0, azimuth_deg=0.0),
        SurveyStation(md_ft=5000.0, inclination_deg=0.0, azimuth_deg=0.0),
        SurveyStation(md_ft=10000.0, inclination_deg=0.0, azimuth_deg=0.0),
    ]

    traj = calculate_minimum_curvature(stations)

    assert traj.total_md_ft == 10000.0
    assert traj.total_tvd_ft == 10000.0
    assert traj.max_dls_deg_100ft == 0.0

    last = traj.stations[-1]
    assert last.north_ft == 0.0
    assert last.east_ft == 0.0
    assert last.closure_distance_ft == 0.0


def test_directional_build_and_hold():
    # Build curve from 0 deg to 30 deg inclination at 2.0 deg/100 ft build rate
    stations = [
        SurveyStation(md_ft=0.0, inclination_deg=0.0, azimuth_deg=45.0),
        SurveyStation(md_ft=2000.0, inclination_deg=0.0, azimuth_deg=45.0),    # Kickoff point (KOP)
        SurveyStation(md_ft=3500.0, inclination_deg=30.0, azimuth_deg=45.0),   # End of build (EOB)
        SurveyStation(md_ft=6000.0, inclination_deg=30.0, azimuth_deg=45.0),   # Tangent section
    ]

    traj = calculate_minimum_curvature(stations, target_azimuth_deg=45.0)

    assert len(traj.stations) == 4
    eob = traj.stations[2]
    # In 1500 ft of MD, built 30 deg -> DLS should be 30 / 15 * 100 = 2.0 deg/100 ft
    assert abs(eob.dls_deg_100ft - 2.0) < 0.05

    # Tangent section: TVD must be less than MD due to 30 deg inclination
    last = traj.stations[3]
    assert last.tvd_ft < last.md_ft
    # Both north and east must be positive for 45 deg azimuth
    assert last.north_ft > 0.0
    assert last.east_ft > 0.0
    assert abs(last.north_ft - last.east_ft) < 1.0  # Equal for 45 deg azimuth
    assert last.vertical_section_ft > 0.0

    # 3D coordinate export
    coords = traj.to_3d_coordinates()
    assert len(coords) == 4
    assert coords[3]["inc"] == 30.0
