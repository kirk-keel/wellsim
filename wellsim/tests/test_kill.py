"""M3-4 dynamic kill with a Newtonian kill fluid. Expected values COMPUTED unless tagged.
The full rho_k sweep (5c, 5d) runs under the ``slow`` marker (about 7 minutes)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wellsim.cases.loader import load_case
from wellsim.cases.schema import Case, DynamicKill
from wellsim.core import kill as km
from wellsim.core.ipr.gas import GasIPR
from wellsim.core.pvt import make_pvt
from wellsim.core.solver import solve
from wellsim.core.vlp import two_phase as tp

KILL = {"mu_k": 1.5, "cp_k": 1.0, "sigma": 60.0, "c_k": 4900.0}


def with_kill(case, rho_k, z_inj=None, rate=None, find_min_rate=True):
    d = case.model_dump()
    d["dynamic_kill"] = {"rho_k": rho_k, "z_inj": z_inj, "rate_bpm": rate, "find_min_rate": find_min_rate, **KILL}
    return Case(**d)


def ipr_of(case, pvt):
    res = case.reservoir
    return GasIPR(pvt=pvt, p_res=res.pressure, t_res=res.temperature, k=res.permeability, h=res.net_pay,
                  r_e=res.drainage_radius, r_w=res.wellbore_radius, skin=res.skin, h_p=res.exposed_interval,
                  h_1=res.exposed_top_offset, kv_kh=res.kv_kh, beta=res.non_darcy_beta,
                  beta_method="geertsma" if res.porosity is not None else "jones", porosity=res.porosity)


@pytest.fixture
def wgr_case(examples_dir):
    return load_case(examples_dir / "gas_water_wgr.yaml")


# --- schema ------------------------------------------------------------------------------------------

def test_dynamic_kill_schema(wgr_case):
    dk = DynamicKill(rho_k=10.0, mu_k=1.5)
    assert dk.cp_k == 1.0 and dk.z_inj is None and dk.rate_bpm is None and dk.find_min_rate
    assert dk.sigma_or_default()[0] == 30.0 and "defaulted" in dk.sigma_or_default()[1]
    assert dk.c_or_default()[0] == 4900.0
    with pytest.raises(ValueError):
        DynamicKill(rho_k=62.3, mu_k=1.0)  # lbm/ft3 typed as ppg
    d = wgr_case.model_dump()
    d["dynamic_kill"] = {"rho_k": 10.0, "mu_k": 1.0, "z_inj": 20000.0}
    with pytest.raises(ValueError):  # below the sandface
        Case(**d)
    kill = km.KillFluid.from_schema(DynamicKill(rho_k=10.0, mu_k=1.5))
    assert kill.rho == pytest.approx(74.805, rel=1e-4) and len(kill.warnings) == 2


def test_mixed_liquid_is_mass_weighted():
    from wellsim.core.vlp.two_phase import LiquidProps
    form = LiquidProps(62.3, 1.0, 1.0, 60.0, 4900.0)
    kill = km.KillFluid(74.8, 3.0, 0.8, 40.0, 4500.0)
    mix = km.mixed_liquid(form, 1.0, kill, 1.0)  # equal mass rates
    assert mix.rho_l == pytest.approx(1.0 / (0.5 / 62.3 + 0.5 / 74.8))
    assert mix.mu_l == 2.0 and mix.cp_l == 0.9 and mix.sigma == 50.0 and mix.c_l == 4700.0
    assert km.mixed_liquid(form, 0.0, kill, 1.0).rho_l == kill.rho
    assert km.mixed_liquid(None, 1.0, kill, 1.0).mu_l == kill.mu


# --- 5a. q_k = 0 reproduces M3-3 ----------------------------------------------------------------------

def test_zero_kill_rate_reproduces_the_operating_point(wgr_case):
    r0 = solve(wgr_case)
    r1 = solve(with_kill(wgr_case, 10.0, rate=0.0, find_min_rate=False))
    assert r1.gas_rate == r0.gas_rate and r1.pwf == r0.pwf and r1.water_rate == r0.water_rate
    assert np.array_equal(r1.profile.p, r0.profile.p)
    at = r1.kill["at_rate"]
    assert at["q_k_bpm"] == 0.0 and not at["killed"]
    assert at["nodal"]["operating"]["q"] == r0.gas_rate
    # the static-hold line is always there
    # absolute basis (M3 closeout): 14.7 + 0.052 x 10 x 10,000 - 5,000 = 214.7 psi
    assert "static_hold" in r1.kill and r1.kill["static_hold"]["ok"] and r1.kill["static_hold"]["margin_psi"] == pytest.approx(214.7)
    assert r1.kill["blount_soeiinah"]["q_bpm"] == 0.0


# --- 5b. Blount-Soeiinah hand check --------------------------------------------------------------------

def test_blount_soeiinah_hand_check(examples_dir, capsys):
    # dry_gas_vertical at 10 ppg: 0.052 x 10 x 11,483 ft = 5,971 psi > p_r 5,511 psia, so the
    # static column holds and the Blount-Soeiinah rate is zero. dry_gas_big_bore_liner
    # (p_r 8,000 psia, 11,950 ft) needs friction: the rate and the per-segment Re, f,
    # hydrostatic and friction terms are printed for a hand check (COMPUTED).
    for name in ("dry_gas_vertical", "dry_gas_big_bore_liner"):
        case = with_kill(load_case(examples_dir / f"{name}.yaml"), 10.0, find_min_rate=False)
        kill = km.KillFluid.from_schema(case.dynamic_kill)
        st = km.static_hold(case, kill)
        bs = km.blount_soeiinah(case, kill, km.injection_md(case, None))
        print(f"\n{name}, 10 ppg: static p_exit {st['p_exit_psia']:.1f} + 0.052 x 10 x {st['tvd_ft']:.0f} ft = "
              f"{st['p_sandface_static_psia']:.0f} psia vs p_r {st['p_res_psia']:.0f} psia, margin {st['margin_psi']:+.0f} psi, "
              f"hold {'OK' if st['ok'] else 'FAILS'}; Blount-Soeiinah rate {bs['q_bpm']:.3f} bbl/min "
              f"(column at rest {bs['p_static_psia']:.1f} psia)")
        for s in bs["segments"]:
            print(f"   {s['conduit']:20s} L {s['length_ft']:8.1f} ft  TVD {s['tvd_ft']:8.1f} ft  v {s['v_ft_s']:7.3f} ft/s  "
                  f"Re {s['re']:10.0f}  f {s['f']:.5f}  dp_hyd {s['dp_hydrostatic_psi']:8.1f}  dp_fric {s['dp_friction_psi']:8.1f} psi")
        if name == "dry_gas_vertical":
            assert st["ok"] and bs["q_bpm"] == 0.0 and bs["static_exceeds_p_res"]
            assert st["p_hydrostatic_psi"] == pytest.approx(0.052 * 10.0 * 11482.94, rel=1e-6)
        else:
            assert not st["ok"] and bs["q_bpm"] > 0.0
            total = 14.7 + sum(s["dp_hydrostatic_psi"] + s["dp_friction_psi"] for s in bs["segments"])
            assert total == pytest.approx(8000.0, abs=1e-3)
            # hand values (COMPUTED, Colebrook at eps 0.0018 in casing / 0.05 in open hole):
            # 72.15 bbl/min, casing 8.681 in: v 16.43 ft/s, Re 882,000, f 0.0149, 448 psi;
            # open hole 6 in: v 34.38 ft/s, Re 1,276,000, f 0.0357, 1,330 psi
            assert bs["q_bpm"] == pytest.approx(72.15, abs=0.05)
            seg = bs["segments"]
            assert seg[0]["re"] == pytest.approx(881867, rel=1e-3) and seg[0]["f"] == pytest.approx(0.01487, abs=1e-4)
            assert seg[1]["dp_friction_psi"] == pytest.approx(1329.8, abs=1.0)


def test_static_hold_absolute_basis_agrees_with_blount_soeiinah(wgr_case, capsys):
    # M3 closeout: p_exit + 0.052 rho_k TVD against p_r. At 9.6 ppg on the WGR example the
    # gauge basis failed by 8 psi while the Blount-Soeiinah column at rest (which includes the
    # 14.7 psia exit) already held; on the absolute basis both say "holds" (margin +6.7 psi,
    # rate 0). A subsea exit at 680 psia passes at 9 ppg where the gauge basis failed by 320.
    case = with_kill(wgr_case, 9.6, find_min_rate=False)
    kill = km.KillFluid.from_schema(case.dynamic_kill)
    st = km.static_hold(case, kill)
    bs = km.blount_soeiinah(case, kill, km.injection_md(case, None))
    print(f"\n9.6 ppg: p_exit {st['p_exit_psia']} + {st['p_hydrostatic_psi']:.1f} = {st['p_sandface_static_psia']:.1f} psia vs "
          f"{st['p_res_psia']:.0f}: margin {st['margin_psi']:+.1f} psi, hold {st['ok']}; Blount-Soeiinah {bs['q_bpm']:.2f} bbl/min")
    assert st["margin_psi"] == pytest.approx(14.7 + 0.052 * 9.6 * 10000.0 - 5000.0)
    assert st["ok"] and bs["q_bpm"] == 0.0 and bs["static_exceeds_p_res"]
    d = wgr_case.model_dump()
    d["exit"]["ambient_pressure"] = 680.0
    d["dynamic_kill"] = {"rho_k": 9.0, "find_min_rate": False, **KILL}
    subsea = Case(**d)
    st2 = km.static_hold(subsea, km.KillFluid.from_schema(subsea.dynamic_kill))
    assert st2["p_exit_psia"] == 680.0 and st2["ok"] and st2["margin_psi"] == pytest.approx(680.0 + 4680.0 - 5000.0)
    assert 0.052 * 9.0 * 10000.0 < 5000.0  # the gauge basis would have failed
    r = solve(subsea)
    assert r.kill["static_hold"]["ok"] and not any("dynamically dead only" in w for w in r.warnings)


# --- 5e. static hold and the warning ------------------------------------------------------------------------

@pytest.mark.parametrize("rho_k, hold", [(9.0, False), (12.0, True)])
def test_static_hold_warning_and_finite_kill_rate(wgr_case, rho_k, hold):
    case = with_kill(wgr_case, rho_k)
    r = solve(case)
    k = r.kill
    assert k["static_hold"]["ok"] is hold
    assert k["static_hold"]["margin_psi"] == pytest.approx(14.7 + 0.052 * rho_k * 10000.0 - 5000.0)
    q = k["q_k_min"]["q_k_min_bpm"]
    assert q is not None and math.isfinite(q) and q > 0.0
    has_warning = any("dynamically dead only" in w for w in r.warnings)
    assert has_warning is (not hold)
    if not hold:
        assert k["blount_soeiinah"]["q_bpm"] > 0.0 and q >= k["blount_soeiinah"]["q_bpm"]
    else:
        assert k["blount_soeiinah"]["q_bpm"] == 0.0


# --- 5f. injection depth ------------------------------------------------------------------------------------

def test_half_depth_injection_needs_more_kill_fluid(wgr_case):
    pvt = make_pvt(wgr_case.fluid, wgr_case.options.correlations)
    sand = with_kill(wgr_case, 12.0)
    half = with_kill(wgr_case, 12.0, z_inj=5000.0)
    a_sand = km.kill_analysis(sand, pvt, ipr_of(sand, pvt))
    a_half = km.kill_analysis(half, pvt, ipr_of(half, pvt))
    q_sand, q_half = a_sand["q_k_min"]["q_k_min_bpm"], a_half["q_k_min"]["q_k_min_bpm"]
    assert a_half["z_inj_tvd_ft"] == pytest.approx(5000.0) and a_sand["z_inj_tvd_ft"] == pytest.approx(10000.0)
    assert q_half > q_sand  # COMPUTED: 61.3 vs 27.5 bbl/min
    assert a_half["blount_soeiinah"]["q_bpm"] > a_sand["blount_soeiinah"]["q_bpm"]


# --- perf: warm-started residual minimum ---------------------------------------------------------------------

def test_warm_start_matches_cold_minimum_with_fewer_calls(wgr_case, capsys):
    # COMPUTED: the warm start changes how the golden bracket is found, not the criterion. Started
    # from a pwf 4% of the span off the cold minimiser at 27.0 bbl/min on the 12 ppg WGR case, the
    # warm evaluation lands within the golden resolution (0.30 span x 0.618^6 = 1.7% of the span,
    # 83 psi) and the residual within 1 psi, with fewer outflow calls (11 against 16).
    case = with_kill(wgr_case, 12.0)
    pvt = make_pvt(case.fluid, case.options.correlations)
    ipr = ipr_of(case, pvt)
    kill = km.KillFluid.from_schema(case.dynamic_kill)
    md_inj = km.injection_md(case, None)
    liq = tp.LiquidProps.from_schema(case.liquid)
    span = case.reservoir.pressure - case.exit.ambient_pressure
    cold_n, warm_n = [0], [0]
    g_cold, p_cold = km.residual_minimum(case, pvt, ipr, kill, 27.0, md_inj, liq, counter=cold_n)
    g_warm, p_warm = km.residual_minimum(case, pvt, ipr, kill, 27.0, md_inj, liq, counter=warm_n,
                                         p_start=p_cold - 0.04 * span)
    print(f"\ncold: g {g_cold:.3f} psi at {p_cold:.1f} psia, {cold_n[0]} calls; "
          f"warm: g {g_warm:.3f} psi at {p_warm:.1f} psia, {warm_n[0]} calls")
    assert abs(p_warm - p_cold) < 0.017 * span
    assert abs(g_warm - g_cold) < 1.0
    assert warm_n[0] < cold_n[0]
    # a start with no interior minimum inside the whole span falls back to the cold scan exactly
    g_fb, p_fb = km.residual_minimum(case, pvt, ipr, kill, 27.0, md_inj, liq, p_start=case.reservoir.pressure - 1.0)
    assert g_fb == g_cold or abs(p_fb - p_cold) < 0.017 * span


# --- 5c, 5d. the rho_k sweep (slow) -------------------------------------------------------------------------

@pytest.mark.slow
def test_rho_k_sweep_monotone_and_above_blount_soeiinah(wgr_case, capsys):
    pvt = make_pvt(wgr_case.fluid, wgr_case.options.correlations)
    base = with_kill(wgr_case, 10.0)
    rows = km.kill_sweep(base, pvt, ipr_of(base, pvt), np.arange(8.6, 18.01, 0.5))
    print(f"\n{'rho_k':>6} {'hold':>5} {'margin':>8} {'BS bpm':>8} {'q_k_min':>8} {'evals':>5} {'s':>6}")
    for r in rows:
        print(f"{r['rho_k_ppg']:6.1f} {'ok' if r['static_ok'] else 'FAIL':>5} {r['margin_psi']:8.0f} {r['blount_soeiinah_bpm']:8.2f} "
              f"{r['q_k_min_bpm']:8.2f} {r['evaluations']:5d} {r['seconds']:6.1f}")
    q = [r["q_k_min_bpm"] for r in rows]
    assert all(v is not None and math.isfinite(v) for v in q)
    for r in rows:  # 5c
        assert r["q_k_min_bpm"] >= r["blount_soeiinah_bpm"] - 1e-9, r
    for a, b in zip(q, q[1:]):  # 5d, to the 1% search tolerance
        assert b <= a * 1.01, (a, b)


# --- ks-2: the search is bracketed to its own tolerance -------------------------------------


@pytest.mark.slow
def test_kill_rate_is_independent_of_the_warm_start(examples_dir, capsys):
    """ks-2 step 4 (COMPUTED). Before ks-2 the same search returned 25.8774 bbl/min from a
    start below the root and 26.6278 from one above, a 2.90% spread, because Brent stopped
    at ``xtol = rel_tol * bracket_top`` and the scan gate that was supposed to catch a
    too-low rate was blind to a 90 psi window (out/ks-diagnosis.md). Four starts must now
    agree inside rel_tol, and each must satisfy the proving check that defines the answer.

    Slow: four kill searches."""
    from dataclasses import replace

    from wellsim.core.pvt import collect_range_warnings, make_pvt
    from wellsim.core.solver import build_ipr
    from wellsim.core.vlp.two_phase import LiquidProps

    case = load_case(examples_dir / "rw_wgr.json")
    rel_tol = 0.01
    with collect_range_warnings():
        pvt = make_pvt(case.fluid, case.options.correlations)
        ipr, _ = build_ipr(case, pvt)
        kill = replace(km.KillFluid.from_schema(case.dynamic_kill), t_inject=60.0)
        md_inj = km.injection_md(case, case.dynamic_kill.z_inj)
        liq = LiquidProps.from_schema(case.liquid)

        rows = []
        for q_start in (5.0, 15.0, 30.0, 60.0):
            r = km.min_kill_rate(case, pvt, ipr, kill, md_inj, liq, q_start=q_start)
            q_k = r["q_k_min_bpm"]
            assert q_k is not None
            # the proving check already ran inside min_kill_rate; re-read both values so the
            # test states the property rather than trusting the call not to have raised
            g_at = km.residual_minimum(case, pvt, ipr, kill, q_k, md_inj, liq)[0]
            g_below = km.residual_minimum(case, pvt, ipr, kill, q_k * (1.0 - rel_tol),
                                          md_inj, liq)[0]
            assert g_at >= 0.0, f"q_start {q_start}: well still flows at the reported {q_k}"
            assert g_below < 0.0, f"q_start {q_start}: {q_k} is not the smallest killing rate"
            lo, hi = r["bracket_bpm"]
            assert lo >= hi * (1.0 - rel_tol) - 1e-12, "the returned bracket is wider than rel_tol"
            rows.append((q_start, lo, hi, q_k, g_at, g_below, r["evaluations"]))

        values = [r[3] for r in rows]
        spread = (max(values) - min(values)) / min(values)
        assert spread < rel_tol, f"warm start still moves the answer by {spread:.3%}"

        print("\nks-2 start independence, rw_wgr 12 ppg, kill fluid at 60 degF:")
        print("   q_start   bracket lo   bracket hi      q_k_min    g(q_k)  g(0.99 q_k)")
        print("  " + "-" * 70)
        for q_start, lo, hi, q_k, g_at, g_below, _ in rows:
            print(f"  {q_start:8.1f} {lo:12.4f} {hi:12.4f} {q_k:12.4f} {g_at:+9.3f} "
                  f"{g_below:+12.3f}")
        print(f"  spread {spread:.4%} against rel_tol {rel_tol:.0%}; before ks-2 it was 2.90%")


def test_proving_check_rejects_a_rate_the_well_survives():
    """ks-2 step 3 is a real gate, not decoration: hand it a rate whose residual minimum is
    negative and it must raise rather than report that rate as the kill rate."""
    calls = {}

    def g_flowing(q):
        calls[q] = -5.0  # the well flows at every rate
        return -5.0

    with pytest.raises(km.KillSearchError, match="still flows"):
        km._prove_kill_rate(g_flowing, 27.0, 0.01)

    def g_dead(q):
        return +5.0  # the well is dead at every rate, so 27.0 is not the SMALLEST

    with pytest.raises(km.KillSearchError, match="not the minimum kill rate"):
        km._prove_kill_rate(g_dead, 27.0, 0.01)

    def g_real(q):
        return 1.0 if q >= 27.0 else -1.0  # the genuine shape

    km._prove_kill_rate(g_real, 27.0, 0.01)  # must not raise
    km._prove_kill_rate(lambda q: -5.0, 0.0, 0.01)  # zero rate has nothing below it
