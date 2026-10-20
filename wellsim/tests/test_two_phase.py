"""M3-2 two-phase march: hand-off, gas and liquid limits, energy, choke, continuity and a
Beggs-Brill cross-check on the total gradient (tests only). Expected values are COMPUTED
unless tagged; Beggs-Brill is MEMORY."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wellsim.cases.loader import load_case
from wellsim.cases.schema import DEFAULT_C_L_FT_S, Case, Liquid
from wellsim.core.exit import G_C
from wellsim.core.friction import darcy_friction_factor, reynolds_number
from wellsim.core.pvt import make_pvt
from wellsim.core.solver import solve
from wellsim.core.vlp.two_phase import (
    LiquidProps,
    liquid_mass_rate_gpm,
    liquid_mass_rate_stbd,
    outflow_profile,
)
from wellsim.tests.test_holdup_crosscheck import beggs_brill_holdup

LBM_FT3_PER_PPG = 7.480519480
RHO_W = 8.33 * LBM_FT3_PER_PPG  # 62.31 lbm/ft3
LIQUID = {"rho_l": 8.33, "mu_l": 1.0, "cp_l": 1.0, "sigma": 60.0, "c_l": 4900.0}


def _case(d_in=6.0, depth=10000.0, p_res=4000.0, t_res=180.0, roughness=None, liquid=LIQUID):
    hole = {"top_md": 0.0, "bottom_md": depth, "id": d_in}
    if roughness is not None:
        hole["roughness"] = roughness
    d = {
        "name": "two_phase_test",
        "reservoir": {"pressure": p_res, "temperature": t_res, "depth_md": depth, "net_pay": 50.0,
                      "permeability": 100.0, "wellbore_radius": 0.354, "drainage_radius": 1500.0},
        "fluid": {"type": "dry_gas", "gas_sg": 0.65},
        "well": {"total_depth_md": depth, "hole": [hole]},
        "flow_path": "no_string",
    }
    if liquid is not None:
        d["liquid"] = dict(liquid)
    return Case(**d)


def _pvt(case):
    return make_pvt(case.fluid, case.options.correlations)


# --- 2. hand-off ----------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dry_gas_vertical", "dry_gas_big_bore_liner", "dry_gas_parted_tubing"])
def test_gas_handoff_is_byte_identical_to_the_solver(name, examples_dir):
    # No case.liquid: outflow_profile takes the solver's gas path; pwf must be identical
    # (==, not approx) to the solver's profile at the operating rate.
    case = load_case(examples_dir / f"{name}.yaml")
    r = solve(case)
    out = outflow_profile(case, _pvt(case), r.gas_rate, 0.0)
    assert not out.two_phase and out.profile is not None and out.profile.ok
    assert out.pwf == r.pwf
    assert out.exit.p == r.exit.p and out.exit.choked == r.exit.choked
    assert np.array_equal(out.profile.p, r.profile.p) and np.array_equal(out.profile.md, r.profile.md)
    # with case.liquid present but zero liquid rate the hand-off still applies
    d = case.model_dump()
    d["liquid"] = dict(LIQUID)
    out2 = outflow_profile(Case(**d), _pvt(case), r.gas_rate, 0.0)
    assert not out2.two_phase and out2.pwf == r.pwf


# --- 6a. gas limit -------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dry_gas_vertical", "dry_gas_big_bore_liner"])
def test_gas_limit_reproduces_the_gas_march(name, examples_dir):
    # Liquid mass rate 1e-6 of the gas mass rate: pwf within 0.1% of the gas march (both
    # examples have a choked exit and a sudden expansion; measured 8e-7 relative).
    case = load_case(examples_dir / f"{name}.yaml")
    r = solve(case)
    pvt = _pvt(case)
    d = case.model_dump()
    d["liquid"] = dict(LIQUID)
    case2 = Case(**d)
    w_g = outflow_profile(case, pvt, r.gas_rate, 0.0).w_gas
    out = outflow_profile(case2, pvt, r.gas_rate, 1e-6 * w_g)
    assert out.two_phase and out.profile.ok
    assert out.exit.choked == r.exit.choked
    # owner's band 0.1%; M3-2b confirms pwf to 1e-6 with the slip momentum flux (measured
    # 7.7e-7 and 7.8e-7). The exit sonic pressure moves by 1.3e-6: the liquid mass fraction
    # of 1e-6 itself shifts the mixture density by that much, so 2e-6 there.
    assert abs(out.pwf / r.pwf - 1.0) < 1e-6, (out.pwf, r.pwf)
    assert abs(out.exit.p / r.exit.p - 1.0) < 2e-6
    assert out.exit.alpha_g > 0.9999 and out.exit.regime == "annular"
    assert len(out.profile.area_changes) == len(r.profile.area_changes)


# --- 6b. liquid limit ----------------------------------------------------------------------------

ROUGHNESS_IN = 0.0006  # the owner's hand calculation; the schema default for casing is 0.0018 in


def test_liquid_limit_hydrostatic_and_friction(capsys):
    # Gas rate zero, vertical 10,000 ft, 2.992 in, water 8.33 ppg, 1 cp, 200 gpm, roughness
    # 0.0006 in (M3-2b: the schema default 0.0018 in gave 432 psi; both paths read the
    # segment roughness and call the same Colebrook routine, so the test sets the
    # roughness the owner's hand value uses).
    # COMPUTED hand values: v = 200 gal/min / A = 9.126 ft/s, Re = rho v d / mu = 211,000,
    # eps/D = 0.0006/2.992, Swamee-Jain f = 0.0170, Colebrook 0.0170, friction
    # f rho v^2 L/(2 g_c D)/144 = 381 psi (owner: 382 with Swamee-Jain), hydrostatic
    # rho L/144 = 4,327 psi (the owner's 0.433 psi/ft gives 4,330). Exit temperature within
    # 2 F of the sandface (friction heating 144 dp_f/(rho J cp) = 1.1 F).
    case = _case(d_in=2.992, p_res=6000.0, t_res=150.0, roughness=ROUGHNESS_IN, liquid={**LIQUID, "sigma": 72.0})
    pvt = _pvt(case)
    w_l = liquid_mass_rate_gpm(200.0, RHO_W)
    out = outflow_profile(case, pvt, 0.0, w_l)
    pr = out.profile
    assert out.two_phase and pr.ok and set(pr.regime) == {"liquid"}
    assert np.all(pr.alpha_g == 0.0)
    area = math.pi / 4.0 * (2.992 / 12.0) ** 2
    v = 200.0 * 0.133680556 / 60.0 / area
    re = reynolds_number(RHO_W, v, 2.992 / 12.0, 1.0)
    rel = ROUGHNESS_IN / 2.992
    f = darcy_friction_factor(re, rel)
    f_sj = 0.25 / math.log10(rel / 3.7 + 5.74 / re**0.9) ** 2  # Swamee-Jain, MEMORY standard form
    dp_hyd = RHO_W * 10000.0 / 144.0
    dp_f_hand = f * RHO_W * v * v / (2.0 * G_C * 2.992 / 12.0) / 144.0 * 10000.0
    dp_f_sj = f_sj * RHO_W * v * v / (2.0 * G_C * 2.992 / 12.0) / 144.0 * 10000.0
    dp_total = pr.p_end - out.exit.p
    dp_f = dp_total - dp_hyd
    print(f"\nliquid limit (roughness {ROUGHNESS_IN} in): v {v:.4f} ft/s, Re {re:.0f}, Colebrook f {f:.5f} "
          f"(module f {pr.f[0]:.5f}, Swamee-Jain {f_sj:.5f}), hydrostatic {dp_hyd:.1f} psi, "
          f"friction {dp_f:.1f} psi (Colebrook hand {dp_f_hand:.1f}, Swamee-Jain hand {dp_f_sj:.1f}), "
          f"pwf {pr.p_end:.1f} psia, exit T {out.exit.t:.3f} F, sandface T {pr.t_end:.3f} F")
    assert abs(v - 9.126) < 0.01
    assert abs(re - 211000.0) / 211000.0 < 2e-3
    assert abs(f_sj - 0.0170) < 1e-4  # the owner's hand value
    assert abs(f - f_sj) / f_sj < 5e-3  # Colebrook against Swamee-Jain
    assert pr.f[0] == pytest.approx(f, rel=1e-9) and pr.re[0] == pytest.approx(re, rel=1e-6)
    assert abs(dp_hyd - 4330.0) < 5.0
    assert dp_f == pytest.approx(dp_f_hand, rel=1e-3)  # acceleration is nil for a liquid
    assert abs(dp_f_sj - 382.0) < 2.0  # the owner's hand value
    assert abs(out.exit.t - pr.t_end) < 2.0
    assert out.exit.t > pr.t_end  # friction warms the liquid on the way up


def test_no_slip_closure_reduces_slip_momentum_to_the_homogeneous_form():
    # M3-2b: with the closure forced to no slip (C0 = 1, v_d = 0: alpha_g = v_sg/v_m) the
    # drift-flux momentum-flux coefficient M equals the homogeneous rho_m v_m v_sg/(rho_g
    # 144 g_c) to 1e-10 (COMPUTED identity; the sensitivity dalpha/drho_g is a Richardson
    # central difference, so the identity is met to round-off).
    from wellsim.core.vlp import two_phase as tp
    from wellsim.core.vlp.holdup import Holdup

    def no_slip(v_sg, v_sl, rho_g, rho_l, mu_l, sigma, d_o, d_i=0.0, theta=90.0):
        # alpha = v_sg/v_m: dalpha/dv_sg = v_sl/v_m^2, no explicit rho_g dependence (analytic,
        # the closure's own field since perf step 2)
        v_m = v_sg + v_sl
        da_dv = v_sl / (v_m * v_m)
        return Holdup(v_sg / v_m, "homogeneous", 1.0, 0.0, v_m, v_m, sigma, 1.0, 0, (),
                      dalpha_dv_sg=da_dv, dalpha_drho_g=-da_dv * v_sg / rho_g)

    case = _case()
    pvt = _pvt(case)
    liq = LiquidProps.from_schema(case.liquid)
    geom = tp.Geometry(math.pi / 4.0 * 0.25, 6.0, 6.0, 0.0, 1.0, 0.0018 / 6.0)
    w_g = liquid_mass_rate_stbd(0.0, RHO_W) + outflow_profile(case, pvt, 2000.0, 0.0).w_gas
    for w_l in (liquid_mass_rate_stbd(500.0, RHO_W), liquid_mass_rate_stbd(4000.0, RHO_W)):
        for p, t in ((200.0, 150.0), (1500.0, 170.0), (3500.0, 180.0)):
            ms = tp.mixture_state(pvt, liq, p, t, w_g, w_l, geom, closure=no_slip)
            assert ms.alpha == pytest.approx(ms.v_sg / ms.v_m)
            assert ms.v_g == pytest.approx(ms.v_m) and ms.v_l == pytest.approx(ms.v_m)
            assert abs(ms.m_coef / ms.m_coef_homogeneous - 1.0) < 1e-10, (p, t, ms.m_coef, ms.m_coef_homogeneous)
    # with the real closure the two differ where the closure slips
    ms = tp.mixture_state(pvt, liq, 1500.0, 170.0, w_g, liquid_mass_rate_stbd(4000.0, RHO_W), geom)
    assert ms.regime in ("slug", "bubbly", "churn") and abs(ms.m_coef / ms.m_coef_homogeneous - 1.0) > 1e-3


# --- 6c. energy: the liquid pressure term ----------------------------------------------------------

def test_frictionless_liquid_column_is_isothermal():
    # Zero gas, smooth pipe, low rate (Re 5,300, friction 0.5 psi): with the liquid pressure
    # term 144 (p - p_ref)/(rho_l J) in h_l the hydrostatic pressure work cancels the
    # potential energy exactly, so exit T equals the sandface T to 0.01 F (COMPUTED).
    case = _case(d_in=2.992, p_res=6000.0, t_res=150.0, roughness=0.0, liquid={**LIQUID, "sigma": 72.0})
    out = outflow_profile(case, _pvt(case), 0.0, liquid_mass_rate_gpm(5.0, RHO_W))
    pr = out.profile
    assert pr.ok
    assert abs(out.exit.t - pr.t_end) < 0.01, (out.exit.t, pr.t_end)
    # the sandface T carries the isenthalpic throttle from p_res: 144 (p_res - pwf)/(rho J cp)
    dt_throttle = 144.0 * (6000.0 - pr.p_end) / (RHO_W * 778.169 * 1.0)
    assert pr.t_end == pytest.approx(150.0 + dt_throttle, abs=0.02)


# --- 6d. choke -------------------------------------------------------------------------------------

def test_two_phase_exit_chokes_at_atmospheric(capsys):
    # 6 in vertical, 20 MMscf/d + 5,000 stb/d water into 14.7 psia: choked. The choke is the
    # singular point of the 2x2 system (mach = 1); the Wallis frozen c_m, alpha_g and v_m
    # are printed with the Wallis Mach at that point (below 1, see the module docstring).
    case = _case()
    out = outflow_profile(case, _pvt(case), 20000.0, liquid_mass_rate_stbd(5000.0, RHO_W))
    ex = out.exit
    # M3-2 (homogeneous acceleration): p_exit 25.60 psia, alpha_g 0.9978, v_m 769.3 ft/s,
    # c_m 842.8 ft/s, mach_wallis 0.9128 at the singular point, pwf 636.4 psia (recorded).
    print(f"\nchoke (M3-2b slip momentum): p_exit {ex.p:.2f} psia, T {ex.t:.1f} F, c_m {ex.c:.1f} ft/s (Wallis), "
          f"alpha_g {ex.alpha_g:.4f}, v_m {ex.v:.1f} ft/s, mach {ex.mach:.4f} (singular point), "
          f"mach_wallis {ex.mach_wallis:.4f}, regime {ex.regime}, pwf {out.pwf:.1f}")
    assert ex.choked and ex.p > 14.7
    assert abs(ex.mach - 1.0) < 1e-6
    assert 0.8 < ex.mach_wallis < 1.0
    assert 0.99 < ex.alpha_g < 1.0 and ex.regime == "annular"
    assert 500.0 < ex.c < 1500.0 and 500.0 < ex.v < 1500.0
    assert out.profile.ok and out.pwf > ex.p
    assert any("Wallis" in w for w in out.warnings)
    # the march starts at the singular point and its Mach falls immediately below
    assert out.profile.mach[0] <= 1.0 + 1e-6 and out.profile.mach[-1] < 0.1


# --- 6e. continuity in liquid rate ------------------------------------------------------------------

def test_pwf_smooth_in_liquid_rate(capsys):
    # 6 in vertical, gas 2 MMscf/d, liquid 0 to 1,000 gpm: pwf rises monotonically and the
    # finite-difference slope changes by at most 3x between neighbouring intervals, across
    # the gas-only hand-off, the slug-to-bubbly change at mid-depth and the exit choke.
    case = _case()
    pvt = _pvt(case)
    q = np.linspace(0.0, 1000.0, 11)
    pwf = []
    print("\nsweep at 2 MMscf/d: gpm, exit p, choked, pwf, mid-depth alpha_g, regime")
    for qq in q:
        out = outflow_profile(case, pvt, 2000.0, liquid_mass_rate_gpm(float(qq), RHO_W))
        pr = out.profile
        assert pr.ok
        i = int(np.argmin(abs(pr.md - 5000.0)))
        alpha = pr.alpha_g[i] if pr.alpha_g is not None else 1.0
        reg = pr.regime[i] if pr.regime is not None else "gas"
        print(f"  {qq:7.1f} {out.exit.p:8.2f} {out.exit.choked!s:5} {pr.p_end:9.2f} {alpha:.3f} {reg}")
        pwf.append(pr.p_end)
    pwf = np.array(pwf)
    slopes = np.diff(pwf) / np.diff(q)
    assert np.all(slopes > 0.0)
    ratio = slopes[:-1] / slopes[1:]
    assert np.all(ratio < 3.0) and np.all(ratio > 1.0 / 3.0), ratio


# --- 6f. Beggs-Brill total gradient, cross-check only ---------------------------------------------------

def _beggs_brill_dp(pvt, liq: LiquidProps, w_g, w_l, d_in, depth, p0, t_of_md, n=200):
    """Surface-to-bottom dp by Beggs-Brill 1973 (MEMORY): RK4 downward in md with the gas
    density from the same PVT at the march's temperature. Original friction correlation
    f_tp = f_n exp(S), f_n from the smooth-pipe Moody form (Drew, Koo and McAdams)."""
    area = math.pi / 4.0 * (d_in / 12.0) ** 2
    d = d_in / 12.0

    def grad(md, p):
        t = t_of_md(md)
        st = pvt.state(p, t)
        rho_g, mu_g = st.rho_g, st.mu_g
        v_sg, v_sl = w_g / (rho_g * area), w_l / (liq.rho_l * area)
        v_m = v_sg + v_sl
        lam = v_sl / v_m
        h_l, _ = beggs_brill_holdup(v_sg, v_sl, liq.rho_l, liq.sigma, d_in)
        rho_s = liq.rho_l * h_l + rho_g * (1.0 - h_l)
        rho_n = liq.rho_l * lam + rho_g * (1.0 - lam)
        mu_n = liq.mu_l * lam + mu_g * (1.0 - lam)
        re_n = reynolds_number(rho_n, v_m, d, mu_n)
        f_n = (2.0 * math.log10(re_n / (4.5223 * math.log10(re_n) - 3.8215))) ** -2
        y = lam / (h_l * h_l)
        if 1.0 < y < 1.2:
            s = math.log(2.2 * y - 1.2)
        else:
            ly = math.log(y)
            s = ly / (-0.0523 + 3.182 * ly - 0.8725 * ly * ly + 0.01853 * ly**4)
        f_tp = f_n * math.exp(s)
        e_k = rho_s * v_m * v_sg / (G_C * p * 144.0)
        return (rho_s / 144.0 + f_tp * rho_n * v_m * v_m / (2.0 * G_C * d) / 144.0) / (1.0 - e_k)

    h = depth / n
    md, p = 0.0, p0
    for _ in range(n):
        k1 = grad(md, p)
        k2 = grad(md + 0.5 * h, p + 0.5 * h * k1)
        k3 = grad(md + 0.5 * h, p + 0.5 * h * k2)
        k4 = grad(md + h, p + h * k3)
        p += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        md += h
    return p - p0


def test_beggs_brill_total_gradient_cross_check(capsys):
    # 6 in vertical, 1.5 MMscf/d + 4,000 stb/d water (drift-flux alpha_g 0.35, slug, at
    # mid-depth): surface-to-bottom dp within 20% of Beggs-Brill. Tests only; MEMORY.
    case = _case()
    pvt = _pvt(case)
    liq = LiquidProps.from_schema(case.liquid)
    w_g_out = outflow_profile(case, pvt, 1500.0, liquid_mass_rate_stbd(4000.0, RHO_W))
    pr = w_g_out.profile
    assert pr.ok and not w_g_out.exit.choked
    i = int(np.argmin(abs(pr.md - 5000.0)))
    assert 0.25 < pr.alpha_g[i] < 0.45, pr.alpha_g[i]
    order = np.argsort(pr.md)
    t_of_md = lambda md: float(np.interp(md, pr.md[order], pr.t[order]))
    dp_df = pr.p_end - w_g_out.exit.p
    dp_bb = _beggs_brill_dp(pvt, liq, w_g_out.w_gas, w_g_out.w_liquid, 6.0, 10000.0, w_g_out.exit.p, t_of_md)
    print(f"\nBeggs-Brill cross-check: drift-flux dp {dp_df:.1f} psi, Beggs-Brill dp {dp_bb:.1f} psi, "
          f"delta {dp_bb / dp_df - 1.0:+.1%}; mid-depth alpha_g {pr.alpha_g[i]:.3f} {pr.regime[i]}")
    assert abs(dp_bb / dp_df - 1.0) < 0.20


# --- schema and warnings ------------------------------------------------------------------------------

def test_c_l_default_is_warned_and_liquid_props_carry_it():
    liq = Liquid(rho_l=8.33, mu_l=1.0, cp_l=1.0)
    c_l, warn = liq.c_l_or_default()
    assert c_l == DEFAULT_C_L_FT_S == 4900.0 and "4900 ft/s" in warn
    props = LiquidProps.from_schema(liq)
    assert props.c_l == 4900.0 and props.sigma == 30.0 and len(props.warnings) == 2
    case = _case(liquid={"rho_l": 8.33, "mu_l": 1.0, "cp_l": 1.0})
    out = outflow_profile(case, _pvt(case), 2000.0, liquid_mass_rate_gpm(300.0, RHO_W))
    assert sum("defaulted" in w for w in out.warnings) == 2
    assert outflow_profile(case, _pvt(case), 2000.0, liquid_mass_rate_gpm(300.0, RHO_W)).pwf == out.pwf


def test_two_phase_needs_the_adiabatic_model():
    d = _case().model_dump()
    d["temperature"] = {"model": "profile"}
    with pytest.raises(NotImplementedError):
        outflow_profile(Case(**d), _pvt(Case(**d)), 2000.0, 1.0)
