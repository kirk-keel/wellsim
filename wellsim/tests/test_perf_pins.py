"""Performance-pass pins (22 Sep 2026, before any optimisation): the four results every
perf commit must reproduce to 1e-6 relative. COMPUTED at HEAD 421d145."""

from __future__ import annotations

import pytest

from wellsim.cases.loader import load_case
from wellsim.cases.schema import Case
from wellsim.core import kill as km
from wellsim.core.ipr.gas import GasIPR
from wellsim.core.pvt import make_pvt
from wellsim.core.solver import solve
from wellsim.core.vlp.two_phase import LiquidProps

REL = 1e-6
DRY_GAS_VERTICAL = (601242.6221193078, 4016.686577131023)  # gas_rate Mscf/d, pwf psia
GAS_WATER_WGR = (232604.7089382972, 4652.094178765944, 3687.9283329215727)  # gas, water, pwf
TWO_ROOT = (74.06147895002204, 3999.7297699113624, 185438.65694978897, 2941.431654975545)  # q, pwf x 2
# Kill path (perf-2, re-pinned 22 Sep 2026 with the algebraic junction of step 1). The
# previous pin, the searched q_k_min 27.458671812362837 bbl/min taken with the re-marching
# junction at its 0.2 F / 0.5 psi stopping rule, was under-converged by 1.1e-3 relative
# (the same junction converged to 1e-7 F gives 27.428757) and is superseded. Rule
# (CLAUDE.md): evaluations pinned at 1e-6, searched results at the search tolerance.
KILL_RESIDUAL_27 = (-112.105173291764, 4871.564508407357)  # g(q_k = 27.0 bbl/min) psi, pwf of the minimum; 1e-6
# ks-2 (22 Sep 2026): re-pinned after the kill search was fixed. The previous pin,
# 27.426611518379403 bbl/min, is SUPERSEDED because the well was still flowing there: the
# residual minimum at that rate is -54.021 psi. The old search accepted it because its
# confirmation scan (16 points over a 4,985 psi span, 332 psi apart) could not see a
# still-flowing window 90 psi wide, and because Brent stopped at rel_tol x the bracket top
# rather than rel_tol x the root. Both are gone; see out/ks-diagnosis.md and PROVENANCE.
KILL_12PPG_SEARCHED = 27.912190551757812  # bbl/min, min_kill_rate at 12 ppg; pinned at its 1e-2 search tolerance
KILL_SEARCH_REL = 1e-2


def _close(a, b):
    return abs(a / b - 1.0) < REL


def test_pin_dry_gas_vertical(examples_dir):
    r = solve(load_case(examples_dir / "dry_gas_vertical.yaml"))
    assert _close(r.gas_rate, DRY_GAS_VERTICAL[0]) and _close(r.pwf, DRY_GAS_VERTICAL[1]), (r.gas_rate, r.pwf)


def test_pin_gas_water_wgr(examples_dir):
    r = solve(load_case(examples_dir / "gas_water_wgr.yaml"))
    assert _close(r.gas_rate, GAS_WATER_WGR[0]) and _close(r.water_rate, GAS_WATER_WGR[1]) and _close(r.pwf, GAS_WATER_WGR[2]), \
        (r.gas_rate, r.water_rate, r.pwf)


def test_pin_two_root_wgr(examples_dir):
    d = load_case(examples_dir / "gas_water_wgr.yaml").model_dump()
    d["reservoir"]["pressure"] = 4000.0
    r = solve(Case(**d))
    roots = sorted(r.nodal.intersections, key=lambda i: i.q)
    assert len(roots) == 2
    got = (roots[0].q, roots[0].pwf, roots[1].q, roots[1].pwf)
    assert all(_close(g, p) for g, p in zip(got, TWO_ROOT)), got


def _kill_setup(examples_dir):
    d = load_case(examples_dir / "gas_water_wgr.yaml").model_dump()
    d["dynamic_kill"] = {"rho_k": 12.0, "mu_k": 1.5, "cp_k": 1.0, "sigma": 60.0, "c_k": 4900.0}
    case = Case(**d)
    pvt = make_pvt(case.fluid, case.options.correlations)
    res = case.reservoir
    ipr = GasIPR(pvt=pvt, p_res=res.pressure, t_res=res.temperature, k=res.permeability, h=res.net_pay,
                 r_e=res.drainage_radius, r_w=res.wellbore_radius, skin=res.skin, beta_method="jones")
    return case, pvt, ipr, km.KillFluid.from_schema(case.dynamic_kill), km.injection_md(case, None), \
        LiquidProps.from_schema(case.liquid)


def test_pin_kill_residual_at_fixed_rate(examples_dir):
    # an evaluation: the 1e-6 gate for the kill path
    case, pvt, ipr, kill, md_inj, liq = _kill_setup(examples_dir)
    g, p_at = km.residual_minimum(case, pvt, ipr, kill, 27.0, md_inj, liq)
    assert _close(g, KILL_RESIDUAL_27[0]) and _close(p_at, KILL_RESIDUAL_27[1]), (g, p_at)


def test_pin_kill_rate_12ppg_at_search_tolerance(examples_dir):
    # a searched result: pinned at its own 1e-2 stopping rule
    case, pvt, ipr, kill, md_inj, liq = _kill_setup(examples_dir)
    mk = km.min_kill_rate(case, pvt, ipr, kill, md_inj, liq)
    assert abs(mk["q_k_min_bpm"] / KILL_12PPG_SEARCHED - 1.0) < KILL_SEARCH_REL, (mk["q_k_min_bpm"], KILL_12PPG_SEARCHED)
