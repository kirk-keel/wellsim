"""Inverse mode (core/inverse.py): kh from an observed rate, p_r from SIWHP. All expected
values COMPUTED: round trips through the forward solve and the static column, monotone
sweeps, downstream consistency and the stated failure modes. Searched kh is pinned at
its own 0.5% stopping rule (CLAUDE.md pinning rule); the SIWHP round trip at the 0.1%
the owner set (the RK4 error is far below it)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from wellsim.cases.loader import load_case
from wellsim.cases.schema import Case
from wellsim.core import inverse as inv
from wellsim.core import kill as km
from wellsim.core.ipr.gas import GasIPR
from wellsim.core.pvt import make_pvt
from wellsim.core.solver import solve
from wellsim.core.vlp.two_phase import LiquidProps

EXAMPLES = Path(__file__).resolve().parents[1] / "cases" / "examples"
KH_REL = inv.REL_TOL_Q  # 0.5%: the search tolerance on q_g and on kh (ln-kh bracket 5e-3)
SIWHP_REL = 1e-3  # owner's round-trip tolerance on p_r
KILL_SEARCH_REL = 1e-2  # min_kill_rate's own stopping rule
T_SURFACE = 60.0  # degF, shut-in wellhead temperature used in the SIWHP round trip


def observed_rate_case(case: Case, q_g: float, low: float | None = None, high: float | None = None,
                       q_l: float | None = None) -> Case:
    d = case.model_dump()
    d["observed"] = {"rate": {"q_g": q_g, "q_l": q_l, "q_g_low": low, "q_g_high": high}}
    return Case(**d)


def observed_siwhp_case(case: Case, p_wh: float, t_surface: float = T_SURFACE, time_note: str = "") -> Case:
    d = case.model_dump()
    d["observed"] = {"siwhp": {"pressure": p_wh, "surface_temperature": t_surface, "time_note": time_note}}
    return Case(**d)


def kh_true(case: Case) -> float:
    return case.reservoir.permeability * case.reservoir.net_pay


@pytest.fixture(scope="module")
def dry_case():
    return load_case(EXAMPLES / "dry_gas_vertical.yaml")


@pytest.fixture(scope="module")
def wgr_case():
    return load_case(EXAMPLES / "gas_water_wgr.yaml")


@pytest.fixture(scope="module")
def forward_dry(dry_case):
    return solve(dry_case)


@pytest.fixture(scope="module")
def forward_wgr(wgr_case):
    return solve(wgr_case)


@pytest.fixture(scope="module")
def inv_dry(dry_case, forward_dry):
    q = forward_dry.gas_rate
    return inv.invert(observed_rate_case(dry_case, q, low=0.8 * q, high=1.2 * q))


@pytest.fixture(scope="module")
def inv_wgr(wgr_case, forward_wgr):
    return inv.invert(observed_rate_case(wgr_case, forward_wgr.gas_rate))


def _print_round_trip(name, case, r, forward_q):
    print(f"\n{name}: kh_true {kh_true(case):.4f} md-ft, q_fwd {forward_q:.2f} Mscf/d")
    print(f"  recovered kh {r.value:.4f} md-ft ({r.value / kh_true(case) - 1.0:+.3%}), "
          f"q at solution {r.observation['q_g_at_solution_mscfd']:.2f} Mscf/d")
    print(f"  {len(r.evaluations)} forward solves, {r.elapsed_s:.1f} s; bracket top {r.tolerance['kh_hi']:g}")
    s = r.sensitivity
    print(f"  dq_g/d(ln kh) {s['dq_g_dln_kh_mscfd']:.1f} Mscf/d, elasticity {s['elasticity_dlnq_dlnkh']:.4f}")
    if "kh_for_q_minus_20pct_linearised" in s:
        print(f"  kh for q -20% / +20% (linearised): {s['kh_for_q_minus_20pct_linearised']:.1f} / "
              f"{s['kh_for_q_plus_20pct_linearised']:.1f} md-ft")
    for n in r.notes:
        print("  note:", n[:76])


# --- schema ---------------------------------------------------------------------------------------------

def test_observed_schema(dry_case):
    c = observed_rate_case(dry_case, 1000.0)
    assert c.observed.rate.q_g == 1000.0 and c.observed.siwhp is None
    with pytest.raises(ValidationError, match="exactly one"):
        d = dry_case.model_dump()
        d["observed"] = {"rate": {"q_g": 1.0}, "siwhp": {"pressure": 100.0, "surface_temperature": 60.0}}
        Case(**d)
    with pytest.raises(ValidationError, match="both q_g_low and q_g_high"):
        observed_rate_case(dry_case, 1000.0, low=800.0)
    with pytest.raises(ValidationError, match="q_g_low < q_g < q_g_high"):
        observed_rate_case(dry_case, 1000.0, low=1100.0, high=1200.0)
    with pytest.raises(ValidationError, match="q_l needs liquid"):
        observed_rate_case(dry_case, 1000.0, q_l=10.0)
    # the forward solve ignores the block
    assert solve(c).gas_rate == pytest.approx(solve(dry_case).gas_rate)


# --- a. round trips on kh -----------------------------------------------------------------------------------

def test_a_round_trip_kh_dry_gas(dry_case, forward_dry, inv_dry, capsys):
    r = inv_dry
    _print_round_trip("dry_gas_vertical", dry_case, r, forward_dry.gas_rate)
    assert r.kind == "kh" and r.unit == "md-ft"
    assert abs(r.value / kh_true(dry_case) - 1.0) < KH_REL  # COMPUTED round trip, pinned at the search tolerance
    assert abs(r.observation["q_g_at_solution_mscfd"] / forward_dry.gas_rate - 1.0) <= KH_REL
    assert r.statement == "effective kh: absorbs skin, net pay and geometry error; not a reservoir property."
    for key in ("h_ft", "skin", "kv_kh", "p_res_psia"):
        assert key in r.fixed_inputs
    assert r.fixed_inputs["h_ft"] == dry_case.reservoir.net_pay and r.fixed_inputs["skin"] == 2.0
    # the case copy carries k = kh/h and solves to the observed rate
    assert r.case.reservoir.permeability == pytest.approx(r.value / dry_case.reservoir.net_pay)
    assert r.case.reservoir.net_pay == dry_case.reservoir.net_pay
    assert abs(solve(r.case).gas_rate / forward_dry.gas_rate - 1.0) <= KH_REL


def test_a_round_trip_kh_gas_water(wgr_case, forward_wgr, inv_wgr, capsys):
    r = inv_wgr
    _print_round_trip("gas_water_wgr", wgr_case, r, forward_wgr.gas_rate)
    assert abs(r.value / kh_true(wgr_case) - 1.0) < KH_REL  # COMPUTED
    assert r.tolerance["kh_hi"] < inv.KH_HI  # 1e6 md-ft is not evaluable on this case (gas denser than water)
    assert any("stepped down" in n for n in r.notes)
    assert r.fixed_inputs["liquid_inflow"]["mode"] == "wgr"


# --- b. round trip on SIWHP -----------------------------------------------------------------------------------

def test_b_round_trip_siwhp(dry_case, capsys):
    res = dry_case.reservoir
    pvt = make_pvt(dry_case.fluid, dry_case.options.correlations)
    t_of = inv.geothermal(dry_case, T_SURFACE)
    p_wh = inv.static_gas_column(pvt, res.pressure, res.datum_tvd, 0.0, t_of)  # forward: datum up to surface
    assert 0.0 < p_wh < res.pressure
    r = inv.invert(observed_siwhp_case(dry_case, p_wh, time_note="illustrative, 24 h shut-in"))
    print(f"\nSIWHP round trip: p_r {res.pressure:.3f} psia -> p_wh {p_wh:.3f} psia -> p_r {r.value:.3f} psia "
          f"({r.value / res.pressure - 1.0:+.2e}); datum {res.datum_tvd:.1f} ft TVD")
    assert r.kind == "p_res" and r.unit == "psia"
    assert abs(r.value / res.pressure - 1.0) < SIWHP_REL  # COMPUTED round trip at the owner's 0.1%
    assert r.case.reservoir.pressure == r.value and r.observation["siwhp_psia"] == p_wh
    assert r.fixed_inputs["gas_sg"] == dry_case.fluid.gas_sg and r.fixed_inputs["surface_temperature_degf"] == T_SURFACE
    assert not any("time_note" in w for w in r.warnings)
    # downstream solve on the copy runs unchanged and lands on the forward rate (p_r differs by < 0.1%)
    assert solve(r.case).flowing
    # empty time note is an assumption -> warning
    r2 = inv.invert(observed_siwhp_case(dry_case, p_wh))
    assert any("time_note is empty" in w for w in r2.warnings)


# --- c. monotone in kh ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, top_decade", [("dry_gas_vertical", 6), ("gas_water_wgr", 5)])
def test_c_rate_increases_with_kh(name, top_decade, capsys):
    case = load_case(EXAMPLES / f"{name}.yaml")
    khs = [10.0 ** e for e in range(-2, top_decade + 1)]
    qs = [inv.forward_rate(inv.with_kh(case, kh)) for kh in khs]
    print(f"\n{name}: forward rate across the bracket")
    print(f"{'kh md-ft':>10} {'q_g Mscf/d':>12}")
    for kh, q in zip(khs, qs):
        print(f"{kh:10.2e} {q:12.2f}")
    assert all(b > a for a, b in zip(qs, qs[1:])), qs  # COMPUTED trend


# --- d. downstream: the 12 ppg kill rate from the inverted case -----------------------------------------------------

def _kill_setup(case: Case):
    d = case.model_dump()
    d["dynamic_kill"] = {"rho_k": 12.0, "mu_k": 1.5, "cp_k": 1.0, "sigma": 60.0, "c_k": 4900.0}
    c = Case(**d)
    pvt = make_pvt(c.fluid, c.options.correlations)
    res = c.reservoir
    ipr = GasIPR(pvt=pvt, p_res=res.pressure, t_res=res.temperature, k=res.permeability, h=res.net_pay,
                 r_e=res.drainage_radius, r_w=res.wellbore_radius, skin=res.skin, beta_method="jones")
    return c, pvt, ipr, km.KillFluid.from_schema(c.dynamic_kill), km.injection_md(c, None), LiquidProps.from_schema(c.liquid)


def test_d_downstream_kill_rate_matches_forward(wgr_case, inv_wgr, capsys):
    fwd = km.min_kill_rate(*_kill_setup(wgr_case))["q_k_min_bpm"]
    got = km.min_kill_rate(*_kill_setup(inv_wgr.case))["q_k_min_bpm"]
    print(f"\n12 ppg kill rate: forward {fwd:.3f} bbl/min, from the inverted case {got:.3f} bbl/min "
          f"({got / fwd - 1.0:+.3%}); kh {inv_wgr.value:.2f} vs {kh_true(wgr_case):.2f} md-ft")
    assert abs(got / fwd - 1.0) < KILL_SEARCH_REL  # COMPUTED, at the kill search's 1% stopping rule


# --- e. bounds -----------------------------------------------------------------------------------------------------

def test_e_bounds_bracket_the_solution(inv_dry, forward_dry, capsys):
    b = inv_dry.bounds
    s = inv_dry.sensitivity
    q = forward_dry.gas_rate
    print(f"\nkh for q_g -20% / observed / +20% on dry_gas_vertical (q_obs {q:.1f} Mscf/d)")
    print(f"{'':>12} {'q_g Mscf/d':>12} {'kh md-ft':>12} {'kh/kh_obs':>10}")
    print(f"{'q_g -20%':>12} {b['q_g_low_mscfd']:12.1f} {b['kh_low_md_ft']:12.1f} {b['kh_low_md_ft'] / b['kh_md_ft']:10.4f}")
    print(f"{'observed':>12} {q:12.1f} {b['kh_md_ft']:12.1f} {1.0:10.4f}")
    print(f"{'q_g +20%':>12} {b['q_g_high_mscfd']:12.1f} {b['kh_high_md_ft']:12.1f} {b['kh_high_md_ft'] / b['kh_md_ft']:10.4f}")
    print(f"linearised from dq_g/d(ln kh) = {s['dq_g_dln_kh_mscfd']:.1f} Mscf/d: "
          f"{s['kh_for_q_minus_20pct_linearised']:.1f} / {s['kh_for_q_plus_20pct_linearised']:.1f} md-ft")
    assert b["kh_low_md_ft"] < b["kh_md_ft"] < b["kh_high_md_ft"]  # COMPUTED ordering
    assert abs(b["q_g_at_kh_low"] / b["q_g_low_mscfd"] - 1.0) <= KH_REL
    assert abs(b["q_g_at_kh_high"] / b["q_g_high_mscfd"] - 1.0) <= KH_REL
    assert s["kh_for_q_minus_20pct_linearised"] < b["kh_md_ft"] < s["kh_for_q_plus_20pct_linearised"]
    assert s["dq_g_dln_kh_mscfd"] > 0.0


# --- f. failure modes ------------------------------------------------------------------------------------------------

def test_f_observed_rate_above_exit_limited_maximum_raises(dry_case):
    with pytest.raises(inv.InverseError, match="exit-limited"):
        inv.invert(observed_rate_case(dry_case, 2.0e6))


def test_f_observed_rate_below_bracket_bottom_raises(dry_case):
    with pytest.raises(inv.InverseError, match="bottom of the bracket"):
        inv.invert(observed_rate_case(dry_case, 0.1))


def test_f_missing_inputs_raise(dry_case, wgr_case):
    d = dry_case.model_dump()
    d["observed"] = {"rate": {"q_g": 1000.0}}
    del d["reservoir"]["net_pay"]
    with pytest.raises(ValidationError, match="net_pay"):
        Case(**d)
    d = dry_case.model_dump()
    d["observed"] = {"rate": {"q_g": 1000.0}}
    del d["reservoir"]["skin"]  # would silently default to 0 in a forward case
    with pytest.raises(ValidationError, match="skin must be given explicitly"):
        Case(**d)
    d = dry_case.model_dump()
    d["observed"] = {"siwhp": {"pressure": 3000.0}}  # no surface temperature
    with pytest.raises(ValidationError, match="surface_temperature"):
        Case(**d)
    with pytest.raises(inv.InverseError, match="liquid leg"):
        inv.invert(observed_siwhp_case(wgr_case, 3000.0))
    with pytest.raises(inv.InverseError, match="no observed block"):
        inv.invert(dry_case)
