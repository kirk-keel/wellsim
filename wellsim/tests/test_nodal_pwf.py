"""Nodal solve in pwf space (default since the M2.5-3b closeout) and the IPR flat cap.

All COMPUTED: monotonicity of f(pwf) = pwf_vlp(q_ipr(pwf)) - pwf on the examples,
agreement between the pwf-space and rate-space solvers, the capped IPR, and synthetic
cases whose operating point sits on the cap.
"""

from __future__ import annotations

import math
import warnings

import pytest

from wellsim.cases.loader import case_from_dict, case_to_dict, load_case
from wellsim.core.exit import exit_state_adiabatic, gas_mass_rate
from wellsim.core.geometry import exit_area, flow_path_segments
from wellsim.core.ipr.gas import GasIPR
from wellsim.core.nodal import NodalError, find_operating_point_pwf
from wellsim.core.pvt import DryGasPVT, make_pvt
from wellsim.core.pvt.base import M_AIR, RangeWarning
from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU, enthalpy
from wellsim.core.solver import solve
from wellsim.core.vlp.march import march
from wellsim.tests.conftest import EXAMPLES_DIR

EXAMPLES = ["dry_gas_vertical", "dry_gas_big_bore_liner", "dry_gas_parted_tubing"]


def rel(a, b):
    return abs(a - b) / abs(b)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        return fn()


def _case(name, **over):
    d = case_to_dict(load_case(EXAMPLES_DIR / f"{name}.yaml"))
    for k, v in over.items():
        cur = d
        *path, last = k.split(".")
        for p in path:
            cur = cur.setdefault(p, {})
        cur[last] = v
    return d


# --- f(pwf) is monotonic on the three examples -------------------------------------------

@pytest.mark.parametrize("name", EXAMPLES)
def test_f_is_monotonic_decreasing_on_examples(name):
    # COMPUTED: f(pwf) = pwf_vlp(q_ipr(pwf)) - pwf, adiabatic default, 12 pwf points between
    # ambient and reservoir pressure: strictly decreasing, positive at ambient, negative at
    # reservoir pressure (single root).
    def run():
        c = case_from_dict(_case(name))
        res = c.reservoir
        pvt = make_pvt(c.fluid, c.options.correlations)
        segs = flow_path_segments(c.well, c.flow_path, res.depth_md)
        a = exit_area(segs, c.exit.exit_id)
        ipr = GasIPR(pvt=pvt, p_res=res.pressure, t_res=res.temperature, k=res.permeability, h=res.net_pay,
                     r_e=res.drainage_radius, r_w=res.wellbore_radius, skin=res.skin, h_p=res.exposed_interval,
                     h_1=res.exposed_top_offset, kv_kh=res.kv_kh, beta=res.non_darcy_beta,
                     beta_method="geertsma" if res.porosity is not None else "jones", porosity=res.porosity)
        m = M_AIR * pvt.gas_sg
        ht = enthalpy(pvt, res.pressure, res.temperature) / m - c.well.md_to_tvd(res.depth_md) / J_FT_LBF_PER_BTU
        p_amb = c.exit.ambient_pressure

        def f(pwf):
            q = ipr.rate(pwf)
            ex = exit_state_adiabatic(pvt, q, p_amb, a, ht, res.temperature - 40.0)
            mr = march(pvt, segs, None, gas_mass_rate(q, pvt.gas_sg), ex.p, 0.0, res.depth_md,
                       adiabatic=True, t_start=ex.t)
            return (mr.p_end if mr.ok else math.inf) - pwf

        ps = [p_amb + (res.pressure - p_amb) * i / 11 for i in range(12)]
        return [f(p) for p in ps]

    fs = _quiet(run)
    assert fs[0] > 0.0 and fs[-1] < 0.0
    assert all(b < a for a, b in zip(fs, fs[1:])), fs


# --- pwf space and rate space agree ------------------------------------------------------

def _pinned_cases():
    yield "vertical_adiabatic", _case("dry_gas_vertical")
    yield "big_bore_adiabatic", _case("dry_gas_big_bore_liner")
    yield "parted_adiabatic", _case("dry_gas_parted_tubing")
    for n in EXAMPLES:
        yield f"{n}_profile", _case(n, **{"temperature.model": "profile"})
    yield "vertical_k_override", _case("dry_gas_vertical", **{"exit.k_override": 1.28})
    d = _case("dry_gas_vertical")
    d["reservoir"]["exposed_interval"] = 0.10 * d["reservoir"]["net_pay"]
    d["reservoir"]["kv_kh"] = 0.1
    yield "vertical_h_pd_0.10", d


@pytest.mark.parametrize("label,d", list(_pinned_cases()), ids=[k for k, _ in _pinned_cases()])
def test_pwf_space_agrees_with_rate_space(label, d):
    # COMPUTED: the two formulations locate the same crossing. The rate-space solver is
    # run with tight tolerances here (its defaults stop at |residual| <= 0.5 psi, which is
    # ~1e-5 in rate); agreement 1e-6 in rate and 1e-3 psi in pwf.
    d_pwf = {**d, "options": {"numerics": {"nodal_space": "pwf"}}}
    d_rate = {**d, "options": {"numerics": {"nodal_space": "rate", "rate_tol": 1e-9, "pressure_tol": 1e-6}}}
    r_p = _quiet(lambda: solve(case_from_dict(d_pwf)))
    r_r = _quiet(lambda: solve(case_from_dict(d_rate)))
    assert r_p.nodal.space == "pwf" and r_r.nodal.space == "rate"
    assert r_p.flowing and r_r.flowing
    assert rel(r_p.gas_rate, r_r.gas_rate) < 1e-6, (label, r_p.gas_rate, r_r.gas_rate)
    assert abs(r_p.pwf - r_r.pwf) < 1e-3, (label, r_p.pwf, r_r.pwf)
    assert r_p.nodal.residual_psi < 1e-5


# --- the capped IPR --------------------------------------------------------------------------

def test_ipr_cap_is_flat_below_p_aof():
    # COMPUTED: below the pressure of maximum rate the rate equals the maximum; above it the
    # raw curve is used and decreases with pwf; the maximum is located over the full range.
    pvt = DryGasPVT(0.65)
    ipr = _quiet(lambda: GasIPR(pvt=pvt, p_res=5000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0, r_w=0.354,
                                h_p=5.0, kv_kh=0.1))
    assert 14.7 < ipr.p_aof < 5000.0
    for p in (14.7, 0.5 * ipr.p_aof, 0.999 * ipr.p_aof):
        assert ipr.rate(p) == ipr.q_aof and ipr.cap_active(p)
    assert ipr.rate(ipr.p_aof) == pytest.approx(ipr.q_aof, rel=1e-12)
    ps = [ipr.p_aof + (5000.0 - ipr.p_aof) * i / 40 for i in range(41)]
    qs = [ipr.rate(p) for p in ps]
    assert all(b <= a for a, b in zip(qs, qs[1:]))
    assert all(ipr._rate_raw(p) <= ipr.q_aof * (1.0 + 1e-12) for p in (14.7, 100.0, 300.0, ipr.p_aof))
    assert not ipr.cap_active(ipr.p_aof + 1.0)


def test_ipr_maximum_search_matches_brute_force():
    # COMPUTED: the full-range golden-section search agrees with a brute-force 4000-point
    # scan of the raw curve; a thin interval with a large beta puts the turnover at several
    # hundred psi (its location is set by the mu(p) and m(p) shapes, not by beta's size:
    # in the fully non-Darcy limit q ~ sqrt(dm(p) mu(p)) and beta scales out, so a peak at
    # 2000 psia is not reachable with the real IPR; the 2000 psia case is synthetic below).
    pvt = DryGasPVT(0.65)
    for beta, h_p in ((1e7, 5.0), (1e9, 2.0), (1e11, 2.0)):
        ipr = _quiet(lambda: GasIPR(pvt=pvt, p_res=6000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0,
                                    r_w=0.354, beta=beta, h_p=h_p, kv_kh=0.1))
        ps = [14.7 + (6000.0 - 14.7) * i / 4000 for i in range(4001)]
        qs = [ipr._rate_raw(p) for p in ps]
        i = max(range(len(ps)), key=lambda j: qs[j])
        assert abs(ipr.p_aof - ps[i]) <= 2.0 * (ps[1] - ps[0]), (beta, ipr.p_aof, ps[i])
        assert ipr.q_aof >= qs[i] * (1.0 - 1e-12)
        assert ipr.rate(14.7) == ipr.q_aof and ipr.rate(0.5 * ipr.p_aof) == ipr.q_aof
        assert ipr.rate(ipr.p_aof + 500.0) < ipr.q_aof


def test_synthetic_operating_point_on_the_cap():
    # COMPUTED: synthetic curves with q_max reached at 2000 psia and an outflow that needs
    # only 1514.7 psia at that rate: the root lies on the cap, q = q_max, pwf = 1514.7.
    p_res, p_cap, q_max = 5000.0, 2000.0, 100.0

    def q_ipr(pwf):
        if pwf >= p_res:
            return 0.0
        if pwf < p_cap:
            return q_max
        return q_max * (p_res - pwf) / (p_res - p_cap)

    def pwf_vlp(q):
        return 14.7 + 0.15 * q * q

    n = find_operating_point_pwf(q_ipr, pwf_vlp, 14.7, p_res)
    assert n.status == "ok" and n.space == "pwf"
    assert n.operating.q == pytest.approx(q_max, rel=1e-12)
    assert n.operating.pwf == pytest.approx(1514.7, abs=1e-5)
    assert n.residual_psi < 1e-6
    # an outflow that jumps from not-evaluable (+inf) above q = 60 straight to a value below
    # the inflow: physically impossible (the outflow pressure must rise to the ceiling
    # first), so the solver raises naming the two pressures instead of guessing
    with pytest.raises(NodalError, match="jumps from not-evaluable"):
        find_operating_point_pwf(q_ipr, lambda q: math.inf if q > 60.0 else 14.7 + 0.5 * q * q, 14.7, p_res)
    # a legitimate ceiling: the outflow pressure climbs to p_res and beyond it is
    # not-evaluable; f stays positive there and the root is found where f crosses zero
    def vlp_ceiling(q):
        v = 14.7 + 0.15 * q * q
        return math.inf if v > p_res else v

    n4 = find_operating_point_pwf(q_ipr, vlp_ceiling, 14.7, p_res)
    assert n4.status == "ok" and abs(n4.operating.pwf - 1514.7) < 1e-3
    # no flow: static column above reservoir pressure everywhere
    n3 = find_operating_point_pwf(q_ipr, lambda q: p_res + 100.0, 14.7, p_res)
    assert n3.status == "no_flow" and n3.operating is None


@pytest.fixture(scope="module")
def exit_restriction_results():
    out = {}
    for exit_id in (1.0, 0.5, 0.25):
        d = _case("dry_gas_vertical", **{"exit.exit_id": exit_id})
        out[exit_id] = _quiet(lambda: solve(case_from_dict(d)))
    return out


def test_exit_restrictions_solve_to_choked_exit_with_monotonic_f(exit_restriction_results):
    # COMPUTED (step 0d): with a 1.0, 0.5 and 0.25 in exit restriction the outflow is
    # not-evaluable at high rates (exit pressure would exceed reservoir pressure, mapped
    # to +inf) and the solve still lands on a choked exit. Across the scan f is strictly
    # decreasing over its finite points, and every +inf point precedes them.
    for exit_id, r in exit_restriction_results.items():
        assert r.flowing and r.nodal.status == "ok", exit_id
        assert r.exit.choked and r.exit.not_evaluable is None
        f = [(math.inf if v is None else v - p) for p, v in zip(r.nodal.pwf_ipr, r.nodal.pwf_vlp)]
        finite = [x for x in f if math.isfinite(x)]
        assert all(b < a for a, b in zip(finite, finite[1:])), (exit_id, f)
        # not-evaluable (+inf) points form a prefix: the outflow is monotone in rate, so
        # once it becomes evaluable it stays evaluable at every lower rate
        first_finite = next(i for i, x in enumerate(f) if math.isfinite(x))
        assert all(not math.isfinite(x) for x in f[:first_finite])
        assert all(math.isfinite(x) for x in f[first_finite:]), (exit_id, f)
        assert f[-1] < 0.0
        assert any(n.startswith("outflow not evaluable") and "exceed reservoir pressure" in n for n in r.notes)


def test_choked_rate_scales_with_area_at_fixed_exit_pressure():
    # COMPUTED: at the same total enthalpy and the same choked exit pressure the mass flux
    # is the same, so the rate is proportional to area; the rate that chokes each orifice
    # at p* = 4000 psia is found by bisection on q and the ratios are 0.25 to 1e-6.
    d = _case("dry_gas_vertical")
    c = case_from_dict(d)
    res = c.reservoir
    pvt = make_pvt(c.fluid, c.options.correlations)
    segs = flow_path_segments(c.well, c.flow_path, res.depth_md)
    m = M_AIR * pvt.gas_sg
    ht = enthalpy(pvt, res.pressure, res.temperature) / m - c.well.md_to_tvd(res.depth_md) / J_FT_LBF_PER_BTU
    p_target = 4000.0

    def q_for(exit_id):
        area = exit_area(segs, exit_id)
        lo, hi = 1.0, 2.0e6
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            ex = _quiet(lambda: exit_state_adiabatic(pvt, mid, 14.7, area, ht, res.temperature - 40.0, p_max=res.pressure))
            if ex.not_evaluable is not None or ex.p > p_target:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    q1, q05, q025 = q_for(1.0), q_for(0.5), q_for(0.25)
    assert abs(q05 / q1 - 0.25) < 1e-6, (q1, q05)
    assert abs(q025 / q05 - 0.25) < 1e-6, (q05, q025)


def test_thin_interval_case_solves_in_pwf_space_and_reports():
    d = _case("dry_gas_vertical")
    d["reservoir"]["exposed_interval"] = 0.10 * d["reservoir"]["net_pay"]
    d["reservoir"]["kv_kh"] = 0.1
    r = _quiet(lambda: solve(case_from_dict(d)))
    assert r.flowing and r.nodal.status == "ok" and r.nodal.space == "pwf"
    assert any(n.startswith("reservoir-limited") for n in r.notes)
    assert r.nodal.residual_psi < 1e-5


def test_cap_note_when_operating_pwf_below_p_aof():
    # COMPUTED: a very high beta with a thin interval puts p_aof high enough that the
    # wellbore needs less pressure than that at the capped rate; the cap note must fire and
    # the operating rate equal the IPR maximum.
    d = _case("dry_gas_vertical")
    d["reservoir"]["exposed_interval"] = 2.0
    d["reservoir"]["kv_kh"] = 0.1
    d["reservoir"]["non_darcy_beta"] = 1e11
    r = _quiet(lambda: solve(case_from_dict(d)))
    assert r.flowing
    assert r.pwf < r.ipr["p_aof"]
    assert r.gas_rate == pytest.approx(r.ipr["q_aof"], rel=1e-9)
    assert any(n.startswith("IPR capped at its maximum") for n in r.notes)
