"""Nodal solver (spec 4.6) on synthetic curves with known intersections. All COMPUTED."""

from __future__ import annotations

import math

import pytest

from wellsim.core.nodal import find_operating_point


def test_single_stable_intersection_linear_curves():
    pr, qmax = 5000.0, 1000.0
    ipr = lambda q: pr * (1.0 - q / qmax) if q <= qmax else None
    vlp = lambda q: 500.0 + 2.0 * q
    # pr - pr q/qmax = 500 + 2 q -> q = 4500 / (pr/qmax + 2) = 4500/7
    r = find_operating_point(ipr, vlp, 1.0, qmax, n_scan=16, rate_tol=1e-6)
    assert r.status == "ok"
    assert len(r.intersections) == 1
    op = r.operating
    assert abs(op.q - 4500.0 / 7.0) < 1e-3
    assert abs(op.pwf - vlp(op.q)) < 1e-3
    assert op.stable


def test_two_intersections_reports_unstable_left_hand():
    # VLP with a minimum (liquid-loading shape): pwf = 3000/q^0.5 * 20 + 3 q ... pick a
    # cubic-free form: vlp(q) = 400 + 60000/q + 3 q, ipr(q) = 5000 (1 - q/1000).
    ipr = lambda q: 5000.0 * (1.0 - q / 1000.0)
    vlp = lambda q: 400.0 + 60000.0 / q + 3.0 * q
    r = find_operating_point(ipr, vlp, 1.0, 1000.0, n_scan=32, rate_tol=1e-9)
    assert len(r.intersections) == 2
    lo, hi = sorted(r.intersections, key=lambda i: i.q)
    assert not lo.stable and hi.stable
    assert r.operating is hi
    for i in (lo, hi):
        assert abs(ipr(i.q) - vlp(i.q)) < 1e-2
    assert any("2 intersections" in w for w in r.warnings)


def test_no_flow_when_vlp_above_ipr_everywhere():
    ipr = lambda q: 3000.0 * (1.0 - q / 1000.0)
    vlp = lambda q: 3500.0 + q
    r = find_operating_point(ipr, vlp, 1.0, 1000.0)
    assert r.status == "no_flow"
    assert r.operating is None
    assert r.intersections == []


def test_curve_failures_are_skipped():
    ipr = lambda q: 5000.0 * (1.0 - q / 1000.0)

    def vlp(q):
        return None if q < 5.0 else 500.0 + 2.0 * q

    r = find_operating_point(ipr, vlp, 1.0, 1000.0, n_scan=16)
    assert r.status == "ok"
    assert any(v is None for v in r.pwf_vlp)
    assert any("could not be evaluated" in w for w in r.warnings)


def test_rate_tolerance_respected():
    ipr = lambda q: 5000.0 * (1.0 - q / 1000.0)
    vlp = lambda q: 500.0 + 2.0 * q
    exact = 4500.0 / 7.0
    r = find_operating_point(ipr, vlp, 1.0, 1000.0, n_scan=8, rate_tol=1e-3)
    assert abs(r.operating.q - exact) / exact < 1e-3


def test_pressure_tolerance_tightens_beyond_rate_tolerance():
    # steep, non-polynomial curves (Brent's interpolation is exact on linear and quadratic
    # ones): with a coarse rate tolerance alone the residual can be many psi; pressure_tol
    # keeps the bracket shrinking until the residual is within it
    ipr = lambda q: 5000.0 - 40.0 * q
    vlp = lambda q: 500.0 + 40.0 * q + 3000.0 / q
    exact = (4500.0 + math.sqrt(4500.0**2 - 4.0 * 80.0 * 3000.0)) / 160.0  # 80 q^2 - 4500 q + 3000 = 0
    loose = find_operating_point(ipr, vlp, 1.0, 100.0, n_scan=6, rate_tol=5e-2, pressure_tol=1e9)
    tight = find_operating_point(ipr, vlp, 1.0, 100.0, n_scan=6, rate_tol=5e-2, pressure_tol=0.5)
    assert abs(loose.operating.q - exact) / exact < 5e-2  # rate criterion alone
    assert abs(ipr(tight.operating.q) - vlp(tight.operating.q)) <= 0.5
    assert abs(tight.operating.q - exact) / exact < 1e-4


def test_bad_bounds():
    with pytest.raises(ValueError):
        find_operating_point(lambda q: 1.0, lambda q: 0.0, 10.0, 1.0)
