"""M3-3: liquid inflow and the gas-liquid nodal solve. Expected values COMPUTED unless
tagged; the Beggs-Brill closure swap is MEMORY and tests only."""

from __future__ import annotations

import numpy as np
import pytest

from wellsim.cases.loader import load_case
from wellsim.cases.schema import Case, LiquidInflow
from wellsim.core.pvt import make_pvt
from wellsim.core.solver import solve
from wellsim.core.vlp import two_phase as tp
from wellsim.core.vlp.holdup import Holdup
from wellsim.tests.test_holdup_crosscheck import beggs_brill_holdup

LIQUID = {"rho_l": 8.33, "mu_l": 1.0, "cp_l": 1.0, "sigma": 60.0, "c_l": 4900.0}

# Dry-gas example results recorded before M3-3 (M3-2 summary, 21 Sep 2026): the gas path
# must give these exactly after the M3-3 changes to the solver and the nodal routine.
PINNED_DRY_GAS = {
    "dry_gas_vertical": (601242.622, 4016.686577),
    "dry_gas_big_bore_liner": (None, 7677.189758),
}


def _print_op(label, r):
    op = r.nodal.operating
    pr = r.profile
    print(f"\n{label}: q_g {r.gas_rate:.1f} Mscf/d, q_l {r.water_rate:.1f} stb/d, pwf {r.pwf:.1f} psia, "
          f"status {r.nodal.status}; sandface {pr.regime[-1]} (alpha_g {pr.alpha_g[-1]:.3f}), "
          f"exit {pr.regime[0]} (alpha_g {pr.alpha_g[0]:.3f}), exit {'choked' if r.exit.choked else 'unchoked'} at "
          f"{r.exit.p:.1f} psia (Wallis Mach {r.exit.mach_wallis:.3f}); roots:")
    for i in r.nodal.intersections:
        print(f"   q_g {i.q:.1f} pwf {i.pwf:.1f} {'stable' if i.stable else 'UNSTABLE'} "
              f"d pwf_vlp/dq {i.dvlp_dq:.4g} d pwf_ipr/dq {i.dipr_dq:.4g}")
    return op


# --- schema -----------------------------------------------------------------------------------------

def test_liquid_inflow_schema_exactly_one_mode(gas_case_dict):
    assert LiquidInflow(mode="wgr", water_gas_ratio=20.0).liquid_rate(3000.0, 1000.0, 4000.0) == pytest.approx(20.0)
    pi = LiquidInflow(mode="pi", productivity_index=2.0)
    assert pi.liquid_rate(3000.0, 1000.0, 4000.0) == pytest.approx(2000.0)
    assert pi.liquid_rate(4500.0, 0.0, 4000.0) == 0.0  # floored
    for bad in (dict(mode="wgr"), dict(mode="wgr", water_gas_ratio=1.0, productivity_index=1.0),
                dict(mode="pi"), dict(mode="pi", productivity_index=1.0, water_gas_ratio=1.0)):
        with pytest.raises(ValueError):
            LiquidInflow(**bad)
    d = dict(gas_case_dict)
    d["liquid_inflow"] = {"mode": "wgr", "water_gas_ratio": 5.0}
    with pytest.raises(ValueError):  # inflow without the liquid block
        Case(**d)


# --- 2. dry-gas path unchanged -------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dry_gas_vertical", "dry_gas_big_bore_liner", "dry_gas_parted_tubing"])
def test_dry_gas_examples_unchanged(name, examples_dir):
    case = load_case(examples_dir / f"{name}.yaml")
    r = solve(case)
    assert r.water_rate == 0.0 and r.correlations["vlp"] == "single_phase_gas"
    # a liquid block without liquid_inflow does not switch the solve
    d = case.model_dump()
    d["liquid"] = dict(LIQUID)
    r2 = solve(Case(**d))
    assert r2.gas_rate == r.gas_rate and r2.pwf == r.pwf and r2.correlations["vlp"] == "single_phase_gas"
    assert np.array_equal(r2.profile.p, r.profile.p)
    pin = PINNED_DRY_GAS.get(name)
    if pin is not None:
        q_pin, pwf_pin = pin
        assert abs(r.pwf - pwf_pin) < 5e-6, (r.pwf, pwf_pin)
        if q_pin is not None:
            assert abs(r.gas_rate - q_pin) < 5e-3


# --- 4a. gas limit -------------------------------------------------------------------------------------

def test_gas_limit_wgr_reproduces_the_dry_gas_rate(examples_dir):
    case = load_case(examples_dir / "dry_gas_vertical.yaml")
    r0 = solve(case)
    d = case.model_dump()
    d["liquid"] = dict(LIQUID)
    d["liquid_inflow"] = {"mode": "wgr", "water_gas_ratio": 1e-6}
    r1 = solve(Case(**d))
    assert r1.correlations["vlp"] == tp.NAME and r1.flowing
    assert abs(r1.gas_rate / r0.gas_rate - 1.0) < 1e-3  # measured 3.4e-9
    assert abs(r1.pwf / r0.pwf - 1.0) < 1e-3
    assert r1.exit.choked == r0.exit.choked


# --- 4b, 4c. WGR and PI examples ------------------------------------------------------------------------

def test_wgr_example(examples_dir, capsys):
    r = solve(load_case(examples_dir / "gas_water_wgr.yaml"))
    op = _print_op("WGR 20 bbl/MMscf, p_r 5,000 psia", r)
    assert r.flowing and r.nodal.status == "ok" and op.stable
    assert r.water_rate == pytest.approx(20.0 * r.gas_rate / 1000.0, rel=1e-9)
    assert 200000.0 < r.gas_rate < 260000.0 and 3500.0 < r.pwf < 3900.0  # COMPUTED 232,605 Mscf/d, 3,688 psia
    assert r.exit.choked and r.profile.regime[0] == "annular"
    assert op.dvlp_dq is not None and op.dvlp_dq > op.dipr_dq
    assert not any("unstable" in w for w in r.warnings)


def test_pi_example(examples_dir, capsys):
    r = solve(load_case(examples_dir / "gas_water_pi.yaml"))
    op = _print_op("PI J = 2 stb/d/psi, p_r 4,000 psia", r)
    assert r.flowing and r.nodal.status == "ok" and op.stable
    assert r.water_rate == pytest.approx(2.0 * (4000.0 - r.pwf), rel=1e-6)
    assert 170000.0 < r.gas_rate < 210000.0 and 2700.0 < r.pwf < 3100.0  # COMPUTED 190,377 Mscf/d, 2,898 psia
    # the loaded-column crossing just below reservoir pressure is a second, unstable root
    assert len(r.nodal.intersections) == 2
    assert [i.stable for i in sorted(r.nodal.intersections, key=lambda i: i.q)] == [False, True]
    assert any("unstable root" in w for w in r.warnings)


# --- 4d. two-root case ------------------------------------------------------------------------------------

def test_two_roots_by_lowering_reservoir_pressure(examples_dir, capsys):
    base = load_case(examples_dir / "gas_water_wgr.yaml")
    d = base.model_dump()
    d["reservoir"]["pressure"] = 4000.0  # 5,000 gives one root; 4,000 puts the IPR below the loaded column
    case = Case(**d)
    r = solve(case)
    _print_op("WGR 20 bbl/MMscf, p_r lowered to 4,000 psia", r)
    roots = sorted(r.nodal.intersections, key=lambda i: i.q)
    assert len(roots) == 2
    assert roots[0].stable is False and roots[1].stable is True
    assert r.nodal.operating.q == roots[1].q  # highest-rate stable root
    assert any("2 intersections found" in w for w in r.warnings)
    assert any("unstable root" in w for w in r.warnings)
    # derivative test agrees with the crossing direction at both roots (no disagreement warning)
    assert not any("disagrees" in w for w in r.warnings)
    for i in roots:
        assert (i.dvlp_dq > i.dipr_dq) == i.stable
    # the outflow curve at 20 gas rates, WGR fixed: falls through a minimum then rises
    pvt = make_pvt(case.fluid, case.options.correlations)
    rho_l = 8.33 * 7.480519480
    print("\noutflow curve at WGR 20 bbl/MMscf (q_g Mscf/d, q_l stb/d, pwf_vlp psia):")
    curve = []
    for q in np.geomspace(50.0, 20000.0, 20):
        w_l = tp.liquid_mass_rate_stbd(20.0 * q / 1000.0, rho_l)
        out = tp.outflow_profile(case, pvt, float(q), w_l)
        curve.append((float(q), out.pwf))
        print(f"   {q:9.1f} {20.0 * q / 1000.0:7.1f} {out.pwf:9.1f}")
    p = np.array([c[1] for c in curve])
    i_min = int(np.argmin(p))
    assert 0 < i_min < len(p) - 1
    assert p[0] > 3500.0 and p[-1] > p[i_min]  # loaded at low rate, friction at high rate


# --- 4e. continuity in WGR ---------------------------------------------------------------------------------

def test_operating_rate_smooth_in_wgr(examples_dir, capsys):
    base = load_case(examples_dir / "gas_water_wgr.yaml")
    wgrs = np.linspace(0.0, 200.0, 11)
    q = []
    print("\nWGR sweep: wgr, q_g, q_l, pwf, roots")
    for wgr in wgrs:
        d = base.model_dump()
        d["liquid_inflow"]["water_gas_ratio"] = float(wgr)
        r = solve(Case(**d))
        assert r.flowing
        print(f"   {wgr:6.1f} {r.gas_rate:10.1f} {r.water_rate:8.1f} {r.pwf:9.1f} {len(r.nodal.intersections)}")
        q.append(r.gas_rate)
    q = np.array(q)
    slopes = np.diff(q) / np.diff(wgrs)
    assert np.all(slopes < 0.0)  # more water, less gas
    ratio = slopes[:-1] / slopes[1:]
    assert np.all(ratio < 3.0) and np.all(ratio > 1.0 / 3.0), ratio


# --- 4f. Beggs-Brill closure swap, tests only ---------------------------------------------------------------

def _beggs_brill_closure(v_sg, v_sl, rho_g, rho_l, mu_l, sigma, d_o, d_i=0.0, theta=90.0):
    """MEMORY: Beggs-Brill liquid holdup as a drift-flux-style closure (alpha_g = 1 - H_L)."""
    v_m = v_sg + v_sl
    if v_sg <= 0.0:
        return Holdup(0.0, "liquid", 1.0, 0.0, v_m, v_m, sigma, 1.0, 0, ())
    if v_sl <= 0.0:
        return Holdup(1.0, "gas", 1.0, 0.0, v_m, v_m, sigma, 1.0, 0, ())
    h_l, reg = beggs_brill_holdup(v_sg, v_sl, rho_l, sigma, d_o - d_i, theta)
    alpha = min(1.0, max(0.0, 1.0 - h_l))
    return Holdup(alpha, f"bb_{reg}", 1.0, 0.0, v_m, v_m, sigma, 1.0, 0, ())


def test_beggs_brill_closure_operating_point_within_band(examples_dir, monkeypatch, capsys):
    case = load_case(examples_dir / "gas_water_wgr.yaml")
    r_df = solve(case)
    monkeypatch.setattr(tp, "CLOSURE", _beggs_brill_closure)
    r_bb = solve(case)
    delta = r_bb.gas_rate / r_df.gas_rate - 1.0
    print(f"\nBeggs-Brill closure: q_g {r_bb.gas_rate:.1f} vs drift-flux {r_df.gas_rate:.1f} Mscf/d ({delta:+.1%}); "
          f"pwf {r_bb.pwf:.1f} vs {r_df.pwf:.1f} psia; BB exit regime {r_bb.profile.regime[0]}")
    assert r_bb.flowing and r_bb.profile.regime[0].startswith("bb_")
    assert abs(delta) < 0.20
