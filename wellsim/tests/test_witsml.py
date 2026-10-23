"""Unit tests for WITSML 1.4.1.1 XML Telemetry Engine."""

import pytest
from wellsim.core.survey import CalculatedStation, SurveyStation, calculate_minimum_curvature
from wellsim.core.witsml import (
    generate_trajectory_witsml,
    generate_wbgeometry_witsml,
    parse_trajectory_witsml,
    parse_wbgeometry_witsml,
)


def test_trajectory_witsml_generation_and_roundtrip():
    # Setup test trajectory using minimum curvature
    raw_stations = [
        SurveyStation(md_ft=0.0, inclination_deg=0.0, azimuth_deg=0.0),
        SurveyStation(md_ft=2000.0, inclination_deg=0.0, azimuth_deg=0.0),
        SurveyStation(md_ft=4000.0, inclination_deg=30.0, azimuth_deg=45.0),
        SurveyStation(md_ft=6000.0, inclination_deg=60.0, azimuth_deg=45.0),
    ]
    computed_traj = calculate_minimum_curvature(raw_stations)

    # 1. Generate XML
    xml_out = generate_trajectory_witsml(
        computed_traj.stations,
        well_name="Well-A1",
        wellbore_name="WB-Main",
        trajectory_name="Definitive MWD",
    )

    assert "<?xml version=" in xml_out
    assert "<trajectorys" in xml_out
    assert "<nameWell>Well-A1</nameWell>" in xml_out
    assert "<trajectoryStation" in xml_out
    assert 'uom="ft"' in xml_out
    assert 'uom="dega"' in xml_out

    # 2. Parse XML back
    parsed_stations = parse_trajectory_witsml(xml_out)
    assert len(parsed_stations) == 4

    # Check station 0 (Surface)
    assert parsed_stations[0].md_ft == 0.0
    assert parsed_stations[0].tvd_ft == 0.0
    assert parsed_stations[0].inclination_deg == 0.0

    # Check station 3 (Deep kick-off)
    stn_orig = computed_traj.stations[3]
    stn_parsed = parsed_stations[3]

    assert pytest.approx(stn_parsed.md_ft, 0.1) == 6000.0
    assert pytest.approx(stn_parsed.inclination_deg, 0.1) == 60.0
    assert pytest.approx(stn_parsed.azimuth_deg, 0.1) == 45.0
    assert pytest.approx(stn_parsed.tvd_ft, 1.0) == stn_orig.tvd_ft
    assert pytest.approx(stn_parsed.north_ft, 1.0) == stn_orig.north_ft
    assert pytest.approx(stn_parsed.east_ft, 1.0) == stn_orig.east_ft


def test_wbgeometry_witsml_generation_and_roundtrip():
    casing_program = [
        {
            "type": "conductor",
            "top_md_ft": 0.0,
            "bottom_md_ft": 1000.0,
            "od_in": 20.0,
            "id_in": 19.0,
            "weight_lbf_ft": 94.0,
        },
        {
            "type": "surface casing",
            "top_md_ft": 0.0,
            "bottom_md_ft": 5000.0,
            "od_in": 13.375,
            "id_in": 12.415,
            "weight_lbf_ft": 68.0,
        },
        {
            "type": "production casing",
            "top_md_ft": 0.0,
            "bottom_md_ft": 10000.0,
            "od_in": 9.625,
            "id_in": 8.681,
            "weight_lbf_ft": 47.0,
        },
    ]

    xml_out = generate_wbgeometry_witsml(casing_program, well_name="Offshore-01", wellbore_name="Slot-1")

    assert "<wbGeometrys" in xml_out
    assert "<typeHoleCsg>surface casing</typeHoleCsg>" in xml_out
    assert 'uom="in"' in xml_out

    parsed_sections = parse_wbgeometry_witsml(xml_out)
    assert len(parsed_sections) == 3

    assert parsed_sections[0]["type"] == "conductor"
    assert parsed_sections[0]["od_in"] == 20.0
    assert parsed_sections[1]["type"] == "surface casing"
    assert parsed_sections[1]["bottom_md_ft"] == 5000.0
    assert parsed_sections[2]["type"] == "production casing"
    assert parsed_sections[2]["od_in"] == 9.625
    assert parsed_sections[2]["weight_lbf_ft"] == 47.0
