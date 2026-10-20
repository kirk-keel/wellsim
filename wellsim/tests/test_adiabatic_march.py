"""M2.5-3b: adiabatic energy equation in the single-phase gas march.

Tags: COMPUTED unless stated. The Fanno-flow reference formulas are MEMORY (textbook
compressible flow, e.g. Shapiro 1953 / Anderson): with f the Fanning friction factor,

    4 f L*/D = (1 - M^2)/(k M^2) + ((k+1)/(2k)) ln[(k+1) M^2 / (2 + (k-1) M^2)]
    T/T*     = (k+1) / (2 + (k-1) M^2)

for adiabatic constant-area flow of a perfect gas with constant cp from Mach M to the
sonic point. The march is driven through a test-only ideal-gas PVT hook (Z = 1,
constant cp) so the reference applies exactly.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from wellsim.cases.loader import case_from_dict, case_to_dict, load_case
from wellsim.core.exit import G_C, gas_mass_rate
from wellsim.core.geometry import FlowSegment
from wellsim.core.pvt import DryGasPVT
from wellsim.core.pvt.base import M_AIR, R_PSIA_FT3, PVTModel, PVTState, RangeWarning, rankine
from wellsim.core.pvt.heat_capacity import R_BTU_LBMOL_R
from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU, thermo_from_state
from wellsim.core.solver import solve
from wellsim.core.vlp.correlations.single_phase_gas import gas_gradient_adiabatic
from wellsim.core.vlp.march import march
from wellsim.tests.conftest import EXAMPLES_DIR


def rel(a, b):
    return abs(a - b) / abs(b)


EXAMPLES = ["dry_gas_vertical", "dry_gas_big_bore_liner", "dry_gas_parted_tubing"]


@pytest.fixture(scope="module")
def adiabatic_results():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        return {n: solve(load_case(EXAMPLES_DIR / f"{n}.yaml")) for n in EXAMPLES}


# --- a. total enthalpy conservation ------------------------------------------------------

def test_total_enthalpy_conserved_along_every_march(adiabatic_results):
    # COMPUTED: h/M + v^2/(2 g_c J) - TVD/J is constant along the adiabatic march; the drift
    # expressed as a temperature (drift / cp_mass) is below 0.01 K.
    for name, r in adiabatic_results.items():
        assert r.correlations["temperature"] == "adiabatic_total_enthalpy"
        h = r.profile.h_total
        assert np.all(np.isfinite(h))
        drift = float(np.max(h) - np.min(h))  # Btu/lbm
        pvt = DryGasPVT(r.inputs["fluid"]["gas_sg"], r.inputs["fluid"]["n2"], r.inputs["fluid"]["co2"],
                        r.inputs["fluid"]["h2s"])
        m = M_AIR * pvt.gas_sg
        cps = [thermo_from_state(pvt, pvt.state(p, t)).cp / m for p, t in ((r.exit.p, r.exit.t), (r.pwf, r.profile.t_end))]
        drift_k = drift / min(cps) / 1.8
        assert drift_k < 0.01, (name, drift, drift_k)
        # the march lands on the sandface at the inlet total enthalpy (throttle + kinetic term)
        assert abs(h[-1] - r.temperature["h_total_btu_lbm"]) / min(cps) / 1.8 < 0.01


# --- b. determinant vanishes at the adiabatic sound speed ---------------------------------

def _segment(d_in=4.0, sin_theta=1.0, rough_in=0.0018, length=100.0):
    return FlowSegment(top_md=0.0, bottom_md=length, d_hyd=d_in, area=math.pi * (d_in / 24.0) ** 2,
                       roughness=rough_in, sin_theta=sin_theta, conduit="test")


def test_determinant_zero_crossing_matches_adiabatic_sound_speed():
    # COMPUTED: det = 1 - v^2/c_s^2 identically, and the mass rate at which det crosses zero
    # gives v = c_s to 1e-6.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        pvt = DryGasPVT(0.65, co2=0.02)
        seg = _segment()
        for p, t in ((150.0, 40.0), (600.0, 100.0), (3000.0, 200.0)):
            st = pvt.state(p, t)
            th = thermo_from_state(pvt, st)
            for frac in (0.1, 0.5, 0.9, 0.999):
                w = frac * th.c_s * st.rho_g * seg.area
                g = gas_gradient_adiabatic(pvt, p, t, w, seg)
                assert abs(g.det - (1.0 - g.mach * g.mach)) < 1e-9, (p, t, frac)
                assert abs(g.det - (1.0 - frac * frac)) < 1e-9
            # zero crossing by bisection on w
            lo, hi = 0.5 * th.c_s * st.rho_g * seg.area, 1.5 * th.c_s * st.rho_g * seg.area
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if gas_gradient_adiabatic(pvt, p, t, mid, seg).det > 0.0:
                    lo = mid
                else:
                    hi = mid
            v_star = lo / (st.rho_g * seg.area)
            assert rel(v_star, th.c_s) < 1e-6


# --- c. Fanno flow ---------------------------------------------------------------------------

class IdealGasPVT(PVTModel):
    """Test hook: Z = 1, constant ideal-gas cp, constant viscosity; DAK-shaped state so the
    departure functions evaluate to zero through the same code path."""

    fluid_type = "ideal_test"

    def __init__(self, gas_sg=0.62, cp_molar=10.0, mu_cp=0.012):
        self.gas_sg = gas_sg
        self.cp = cp_molar
        self.mu = mu_cp

    def cp0_molar(self, t):
        return self.cp

    def delta_h0(self, t1, t2):
        return self.cp * (t2 - t1)

    def k_ratio(self, t):
        return self.cp / (self.cp - R_BTU_LBMOL_R)

    def z(self, p, t):
        return 1.0

    def bg(self, p, t):
        return 0.0

    def gas_density(self, p, t):
        return p * M_AIR * self.gas_sg / (R_PSIA_FT3 * rankine(t))

    def gas_viscosity(self, p, t):
        return self.mu

    def state(self, p, t):
        rho = self.gas_density(p, t)
        return PVTState(p=p, t=t, z=1.0, bg=0.0, rho_g=rho, mu_g=self.mu, gas_sg=self.gas_sg,
                        tpr=5.0, ppr=0.0, rho_r=0.0, dz_drho_r=0.0, dz_dtpr=0.0,
                        dp_drho_t=R_PSIA_FT3 * rankine(t) / (M_AIR * self.gas_sg))


def _fanno_length(m, k):
    return (1.0 - m * m) / (k * m * m) + (k + 1.0) / (2.0 * k) * math.log((k + 1.0) * m * m / (2.0 + (k - 1.0) * m * m))


@pytest.mark.parametrize("m_in", [0.2, 0.5])
def test_fanno_choking_length_and_temperature_profile(m_in):
    # MEMORY (Fanno formulas above); the march must reproduce the choking length within
    # 0.5% and the temperature profile within 0.1%. Horizontal, fully rough pipe so the
    # friction factor is constant (asserted), inlet at 500 psia and 100 degF.
    pvt = IdealGasPVT()
    k = pvt.k_ratio(100.0)
    d_in = 4.0
    seg_len = 400.0 if m_in < 0.3 else 60.0
    seg = _segment(d_in=d_in, sin_theta=0.0, rough_in=0.04, length=seg_len)  # eps/D = 0.01
    p_in, t_in = 500.0, 100.0
    st = pvt.state(p_in, t_in)
    c_in = thermo_from_state(pvt, st).c_s
    w = m_in * c_in * st.rho_g * seg.area
    r = march(pvt, [seg], None, w, p_in, seg_len, 0.0, max_step=2.0, p_tol=0.02, adiabatic=True, t_start=t_in,
              t_tol=0.005)
    assert not r.ok and r.choked_md is not None
    f_moody = r.f
    assert np.max(np.abs(f_moody - f_moody.mean())) / f_moody.mean() < 1e-3
    f_fanning = float(f_moody.mean()) / 4.0
    d_ft = d_in / 12.0
    l_star = _fanno_length(m_in, k) * d_ft / (4.0 * f_fanning)
    l_march = seg_len - r.choked_md
    assert rel(l_march, l_star) < 0.005, (m_in, l_march, l_star)
    # temperature profile: T/T_in = (2 + (k-1) M_in^2) / (2 + (k-1) M^2)
    t_in_r = rankine(t_in)
    for mach, t in zip(r.mach, r.t):
        expected = t_in_r * (2.0 + (k - 1.0) * m_in * m_in) / (2.0 + (k - 1.0) * mach * mach)
        assert rel(rankine(t), expected) < 1e-3, (m_in, mach, t)


# --- d. profile mode and k_override reproduce 0dc1bad -------------------------------------

def test_profile_mode_and_k_override_reproduce_0dc1bad():
    # COMPUTED regression: rates on the profile temperature path (ideal-gas k0; and
    # k_override 1.28), reproduced to 1e-9 by model 'profile' and by k_override alone.
    # Values with the pwf-space nodal solve (default since 0a10f1d's successor): the root
    # is converged to 1e-6 psi. History: 0dc1bad (rate space) 586648.8658039935,
    # 613182.922833526, 72351.44504419934; 0a10f1d (rate space, scan to the IPR maximum)
    # 586648.8656466319, 613181.6118758222, 72351.97522397226. The pwf-space values differ
    # from those by +2.0e-8, -1.2e-5, -7.3e-6 relative: the rate-space Brent stopped at a
    # 0.5 psi residual, which is ~1e-5 in rate.
    expected_profile = {
        "dry_gas_vertical": 586648.877532835,
        "dry_gas_big_bore_liner": 613174.0972666596,
        "dry_gas_parted_tubing": 72351.44439532896,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        for name, q in expected_profile.items():
            d = case_to_dict(load_case(EXAMPLES_DIR / f"{name}.yaml"))
            d["temperature"] = {"model": "profile"}
            r = solve(case_from_dict(d))
            assert r.correlations["temperature"] == "linear"
            assert rel(r.gas_rate, q) < 1e-9, (name, r.gas_rate)
            d["temperature"] = {"model": "linear"}  # alias
            assert rel(solve(case_from_dict(d)).gas_rate, q) < 1e-9
        d = case_to_dict(load_case(EXAMPLES_DIR / "dry_gas_vertical.yaml"))
        d["exit"]["k_override"] = 1.28  # adiabatic requested by default, override forces the old path
        r = solve(case_from_dict(d))
        assert rel(r.gas_rate, 586799.0457176144) < 1e-9  # 586799.032499237 at 0a10f1d (rate space)
        assert any(w.startswith("k_override = 1.28 forces") for w in r.warnings)
        assert r.temperature["model"] == "profile"


# --- e. nine-well validation unchanged ----------------------------------------------------

def test_nine_well_profile_march_unchanged():
    # COMPUTED: the prescribed-profile march (the nine Cullender-Smith wells) is the pre-3b
    # arithmetic; these pwf values were computed from a git archive of 0dc1bad and are
    # reproduced bit-for-bit (asserted at 1e-12).
    from wellsim.core.temperature import LinearTemperature
    from wellsim.tests.test_vlp_gas import ROUGHNESS_IN, WELLS, tubing

    expected = {
        "A13": 2645.2435377795387, "A27": 2671.0752217734976, "B37": 2940.784248604625,
        "B25": 2944.4470364963936, "B63": 2893.9233960299794, "C18": 3246.9432345182818,
        "C19": 3056.5806852266496, "D23": 3373.066811535264, "D40": 2909.9563437962925,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        for name, depth, gamma, length, q_mm, ptf, ttf, twf, d, pwf_cs, pwf_meas in WELLS:
            r = march(DryGasPVT(gamma), [tubing(length, d, ROUGHNESS_IN)],
                      LinearTemperature(t_res=twf, t_exit=ttf, md_res=length),
                      gas_mass_rate(1000.0 * q_mm, gamma), ptf, 0.0, length)
            assert not r.adiabatic and r.h_total is None
            assert rel(r.p_end, expected[name]) < 1e-12, (name, r.p_end)


# --- 8. result fields, warnings and notes -------------------------------------------------

def test_adiabatic_result_reports_temperatures(adiabatic_results):
    for name, r in adiabatic_results.items():
        ti = r.temperature
        assert ti["model"] == "adiabatic" and ti["sandface_jt"] is True
        assert ti["exit_t"] == r.exit.t and ti["exit_mach"] == r.exit.mach
        # JT throttle at the sandface: cooling, except that dry_gas_big_bore_liner (250 degF,
        # high pressure) sits at the DAK inversion where mu_JT is within noise of zero and
        # the throttle "warms" by 0.05 degF; allow that.
        assert ti["sandface_t"] < r.inputs["reservoir"]["temperature"] + 0.1
        assert abs(ti["sandface_t"] - ti["sandface_t_march"]) < 1.0  # kinetic term only
        assert ti["exit_t"] < ti["sandface_t"]
        assert 0.0 < ti["min_tpr"] < 3.0
        assert not any(w.startswith("exit temperature defaulted") for w in r.warnings)
        assert any(w.startswith("adiabatic temperature model") for w in r.warnings)
        assert any(n.startswith("adiabatic temperature: sandface") for n in r.notes)
        assert r.correlations["heat_capacity_ratio"] == "real_gas_cp_cv_dak"
        assert r.as_dict()["temperature"]["min_tpr"] == ti["min_tpr"]


def test_sandface_jt_off_enters_at_reservoir_temperature():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        d = case_to_dict(load_case(EXAMPLES_DIR / "dry_gas_vertical.yaml"))
        d["temperature"] = {"model": "adiabatic", "sandface_jt": False}
        r = solve(case_from_dict(d))
    assert r.temperature["sandface_jt"] is False
    assert r.temperature["sandface_t"] == r.inputs["reservoir"]["temperature"]
    assert abs(r.temperature["sandface_t_march"] - r.temperature["sandface_t"]) < 1.0
    assert "sandface_jt off" in " ".join(r.notes)


def test_ipr_peak_above_ambient_still_finds_operating_point():
    # COMPUTED (M2.5-3b closeout): dry_gas_vertical at h_pD 0.10 has an IPR whose rate
    # peaks near 775 psia (the non-Darcy D carries mu at pwf); the adiabatic column needs
    # less sandface pressure than that, so the crossing sits between rate(p_amb) and the
    # true maximum. Scanning to the IPR maximum finds it, and the result says the well is
    # reservoir-limited.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        d = case_to_dict(load_case(EXAMPLES_DIR / "dry_gas_vertical.yaml"))
        d["reservoir"]["exposed_interval"] = 0.10 * d["reservoir"]["net_pay"]
        d["reservoir"]["kv_kh"] = 0.1
        r = solve(case_from_dict(d))
    assert r.flowing and r.nodal.status == "ok"
    assert min(r.nodal.q_scan) <= r.gas_rate <= max(r.nodal.q_scan)
    assert r.gas_rate > 0.99 * max(r.nodal.q_scan)
    assert any(n.startswith("reservoir-limited") for n in r.notes)
    assert r.temperature["sandface_t"] < r.inputs["reservoir"]["temperature"] - 20.0
    assert any(n.startswith("sandface Joule-Thomson cooling") and "exceeds 20 degF" in n for n in r.notes)


def test_reservoir_limited_note_absent_at_full_penetration(adiabatic_results):
    for r in adiabatic_results.values():
        assert not any(n.startswith("reservoir-limited") for n in r.notes)
        assert not any(n.startswith("sandface Joule-Thomson cooling") for n in r.notes)


def test_profile_mode_keeps_exit_temperature_warnings(gas_case_dict):
    gas_case_dict["temperature"] = {"model": "profile"}
    r = solve(case_from_dict(gas_case_dict))
    assert any(w.startswith("exit temperature defaulted") for w in r.warnings)
    assert r.temperature["model"] == "profile" and r.temperature["sandface_t"] == 200.0
