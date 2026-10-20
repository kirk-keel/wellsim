"""Flow-path geometry and the linear temperature profile. All COMPUTED."""

from __future__ import annotations

import math

import pytest

from wellsim.cases.loader import case_from_dict
from wellsim.core.geometry import circular_area, exit_area, flow_path_segments
from wellsim.core.temperature import make_temperature


def test_no_string_segments_split_at_hole_boundaries(gas_case_dict):
    gas_case_dict["well"]["hole"] = [
        {"top_md": 0.0, "bottom_md": 8000.0, "id": 8.681},
        {"top_md": 8000.0, "bottom_md": 10000.0, "id": 6.184, "kind": "liner"},
    ]
    case = case_from_dict(gas_case_dict)
    segs = flow_path_segments(case.well, "no_string", case.reservoir.depth_md)
    assert [(s.top_md, s.bottom_md) for s in segs] == [(0.0, 8000.0), (8000.0, 10000.0)]
    assert segs[0].d_hyd == 8.681 and segs[1].d_hyd == 6.184
    assert abs(segs[0].area - math.pi / 4 * (8.681 / 12) ** 2) < 1e-12
    assert all(s.sin_theta == 1.0 for s in segs)
    assert exit_area(segs, None) == segs[0].area
    assert abs(exit_area(segs, 4.0) - circular_area(4.0)) < 1e-12


def test_annulus_and_drillstring_segments(oil_case_dict):
    case = case_from_dict(oil_case_dict)  # string 0-9000 od 5 id 4.276, hole 8.535 to 9200
    ann = flow_path_segments(case.well, "annulus", case.reservoir.depth_md)
    assert len(ann) == 1
    assert abs(ann[0].d_hyd - (8.535 - 5.0)) < 1e-12
    assert abs(ann[0].area - (circular_area(8.535) - circular_area(5.0))) < 1e-12
    ds = flow_path_segments(case.well, "drillstring", case.reservoir.depth_md)
    assert ds[0].d_hyd == 4.276
    # below the string bottom the flow is in the hole
    ann2 = flow_path_segments(case.well, "annulus", 9200.0)
    assert [(s.top_md, s.bottom_md) for s in ann2] == [(0.0, 9000.0), (9000.0, 9200.0)]
    assert ann2[1].d_hyd == 8.535


def test_survey_gives_piecewise_inclination(gas_case_dict):
    gas_case_dict["well"]["survey"] = [
        {"md": 0.0, "tvd": 0.0},
        {"md": 4000.0, "tvd": 4000.0},
        {"md": 10000.0, "tvd": 8000.0},
    ]
    case = case_from_dict(gas_case_dict)
    segs = flow_path_segments(case.well, "no_string", 10000.0)
    assert [(s.top_md, s.bottom_md) for s in segs] == [(0.0, 4000.0), (4000.0, 10000.0)]
    assert segs[0].sin_theta == 1.0
    assert abs(segs[1].sin_theta - 4000.0 / 6000.0) < 1e-12


def test_bottom_md_bounds(gas_case_dict):
    case = case_from_dict(gas_case_dict)
    with pytest.raises(ValueError):
        flow_path_segments(case.well, "no_string", 0.0)
    with pytest.raises(ValueError):
        flow_path_segments(case.well, "no_string", 20000.0)


def test_linear_temperature_defaults(gas_case_dict):
    case = case_from_dict(gas_case_dict)  # T_res 200, depth 10000, default loss 10
    t = make_temperature(case)
    assert t.at(10000.0) == 200.0
    assert t.at(0.0) == 190.0
    assert abs(t.at(5000.0) - 195.0) < 1e-12
    assert t.at(12000.0) == 200.0
    assert abs(t.gradient() - 0.001) < 1e-12


def test_linear_temperature_explicit_exit(gas_case_dict):
    gas_case_dict["temperature"] = {"exit_temperature": "40 degC"}
    case = case_from_dict(gas_case_dict)
    t = make_temperature(case)
    assert abs(t.at(0.0) - 104.0) < 1e-9


def test_segments_carry_outer_and_inner_diameter(oil_case_dict, gas_case_dict):
    # M3-1: annulus segments expose d_o (hole ID) and d_i (string OD); pipes have d_i = 0.
    from wellsim.core.geometry import FlowSegment, circular_area

    case = case_from_dict(oil_case_dict)
    ann = flow_path_segments(case.well, "annulus", case.reservoir.depth_md)
    assert ann[0].d_o == 8.535 and ann[0].d_i == 5.0 and ann[0].is_annulus
    assert ann[0].d_hyd == pytest.approx(ann[0].d_o - ann[0].d_i)
    ds = flow_path_segments(case.well, "drillstring", case.reservoir.depth_md)
    assert ds[0].d_o == ds[0].d_hyd == 4.276 and ds[0].d_i == 0.0 and not ds[0].is_annulus
    gas = case_from_dict(gas_case_dict)
    seg = flow_path_segments(gas.well, "no_string", gas.reservoir.depth_md)[0]
    assert seg.d_o == seg.d_hyd and seg.d_i == 0.0
    # Positional construction without d_o/d_i (the tests' style) still works.
    s = FlowSegment(0.0, 100.0, 2.992, circular_area(2.992), 0.0018, 1.0, "tubing")
    assert s.d_o == 2.992 and s.d_i == 0.0
    with pytest.raises(ValueError):
        FlowSegment(0.0, 100.0, 3.0, circular_area(3.0), 0.0018, 1.0, "x", d_o=8.0, d_i=8.0)
    with pytest.raises(ValueError):
        FlowSegment(0.0, 100.0, 3.0, circular_area(3.0), 0.0018, 1.0, "x", d_o=8.0, d_i=-1.0)
