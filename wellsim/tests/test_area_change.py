"""M2.5-4: sudden area changes in the adiabatic march.

Tags: COMPUTED unless stated. Borda-Carnot (sudden expansion momentum balance), the
Crane TP-410 contraction coefficient K_c = 0.5 (1 - beta^2) and the isentropic
area-Mach relation A/A* = (1/M) [(2/(k+1)) (1 + (k-1)/2 M^2)]^((k+1)/(2(k-1))) are
textbook forms (MEMORY).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from wellsim.cases.loader import case_from_dict, case_to_dict, load_case
from wellsim.core.exit import G_C, temperature_from_total_enthalpy
from wellsim.core.pvt import DryGasPVT
from wellsim.core.pvt.base import M_AIR, RangeWarning, rankine
from wellsim.core.pvt.dry_gas import _dak_coeffs, dak_partials
from wellsim.core.pvt.heat_capacity import R_BTU_LBMOL_R
from wellsim.core.pvt.real_gas_thermo import (
    J_FT_LBF_PER_BTU,
    R,
    dak_entropy_integral,
    entropy_from_state,
    s_departure_from_state,
    thermo_from_state,
)
from wellsim.core.solver import solve
from wellsim.core.vlp.area_change import (
    AreaChangeNotEvaluable,
    contraction_with_loss,
    crane_contraction_coefficient,
    expansion_borda_carnot,
    isentropic_step,
)
from wellsim.tests.conftest import EXAMPLES_DIR, _gas_case
from wellsim.tests.test_adiabatic_march import IdealGasPVT


def rel(a, b):
    return abs(a - b) / abs(b)


def area(d_in):
    return math.pi * (d_in / 24.0) ** 2


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        yield


def _setup(pvt, p_l, t_l, a_l, a_s, mach_s):
    """Mass rate giving Mach ``mach_s`` in the small pipe at the large-pipe state, and the
    local total enthalpy per lbm."""
    m = M_AIR * pvt.gas_sg
    st = pvt.state(p_l, t_l)
    th = thermo_from_state(pvt, st)
    w = mach_s * th.c_s * st.rho_g * a_s
    v_l = w / (st.rho_g * a_l)
    return w, th.h / m + v_l * v_l / (2.0 * G_C * J_FT_LBF_PER_BTU)


# --- a. expansion at low Mach vs incompressible Borda-Carnot --------------------------------

def test_expansion_low_mach_matches_incompressible_borda_carnot():
    # COMPUTED: at Mach 0.05 the pressure rise across a sudden expansion (flow from the
    # small to the large pipe) is rho v_s^2 sigma (1 - sigma), sigma = A_s/A_L, i.e. the
    # reversible recovery rho (v_s^2 - v_L^2)/2 net of the loss (1 - beta^2)^2 rho v_s^2/2
    # (MEMORY, Borda-Carnot); agreement within 1%.
    pvt = DryGasPVT(0.65)
    p_l, t_l, a_l, a_s = 2000.0, 150.0, area(8.0), area(4.0)
    sigma = a_s / a_l
    w, h_local = _setup(pvt, p_l, t_l, a_l, a_s, 0.05)
    p_s, t_s, mach_s, choked, resid = expansion_borda_carnot(pvt, w, a_l, a_s, p_l, t_l, h_local, 20000.0)
    assert not choked and abs(resid) < 1e-6 and abs(mach_s - 0.05) < 2e-3
    st_s = pvt.state(p_s, t_s)
    v_s = w / (st_s.rho_g * a_s)
    dp_bc = st_s.rho_g * v_s * v_s * sigma * (1.0 - sigma) / (144.0 * G_C)
    loss = (1.0 - sigma) ** 2 * st_s.rho_g * v_s * v_s / (2.0 * 144.0 * G_C)
    recovery = st_s.rho_g * v_s * v_s * (1.0 - sigma * sigma) / (2.0 * 144.0 * G_C)
    assert rel(p_l - p_s, dp_bc) < 0.01
    assert rel(p_l - p_s, recovery - loss) < 0.01


# --- b. contraction with K_c = 0 conserves total pressure -----------------------------------

def test_contraction_without_loss_conserves_total_pressure():
    # COMPUTED: K_c = 0 makes the step purely isentropic; at Mach 0.05 the incompressible
    # total pressure p + rho v^2/2 is conserved within 0.1%, and the entropy exactly.
    pvt = DryGasPVT(0.65)
    p_l, t_l, a_l, a_s = 2000.0, 150.0, area(8.0), area(4.0)
    w, h_local = _setup(pvt, p_l, t_l, a_l, a_s, 0.05)
    p_s, t_s, *_ = expansion_borda_carnot(pvt, w, a_l, a_s, p_l, t_l, h_local, 20000.0)
    p_l2, t_l2, mach_l2, loss, resid = contraction_with_loss(pvt, w, a_s, a_l, p_s, t_s, h_local, 20000.0, 0.0)
    assert loss == 0.0 and abs(resid) < 1e-9
    st_s, st_l = pvt.state(p_s, t_s), pvt.state(p_l2, t_l2)
    v_s, v_l = w / (st_s.rho_g * a_s), w / (st_l.rho_g * a_l)
    pt_s = p_s + st_s.rho_g * v_s * v_s / (2.0 * 144.0 * G_C)
    pt_l = p_l2 + st_l.rho_g * v_l * v_l / (2.0 * 144.0 * G_C)
    assert rel(pt_l, pt_s) < 1e-3
    assert abs(entropy_from_state(pvt, st_l) - entropy_from_state(pvt, st_s)) < 1e-9
    # with the Crane coefficient the large-pipe pressure is higher by about the loss
    p_l3, *_ , loss3, _ = contraction_with_loss(pvt, w, a_s, a_l, p_s, t_s, h_local, 20000.0,
                                                crane_contraction_coefficient(4.0, 8.0))
    assert loss3 == pytest.approx(0.375 * st_s.rho_g * v_s * v_s / (2.0 * 144.0 * G_C), rel=1e-12)
    assert abs((p_l3 - p_l2) - loss3) < 0.02 * loss3


def test_crane_coefficient():
    # MEMORY (Crane TP-410 sudden contraction): K_c = 0.5 (1 - beta^2), beta = d_s/d_L.
    assert crane_contraction_coefficient(4.0, 8.0) == pytest.approx(0.375)
    assert crane_contraction_coefficient(8.0, 8.0) == 0.0
    assert crane_contraction_coefficient(1.0, 100.0) == pytest.approx(0.49995)


# --- c. isentropic nozzle with the ideal-gas hook -------------------------------------------

class IdealGasEntropyPVT(IdealGasPVT):
    def s0_from_ref(self, t):
        return self.cp * math.log(rankine(t) / rankine(60.0))


def _area_ratio(m, k):
    return (1.0 / m) * ((2.0 / (k + 1.0)) * (1.0 + 0.5 * (k - 1.0) * m * m)) ** ((k + 1.0) / (2.0 * (k - 1.0)))


@pytest.mark.parametrize("ratio", [1.5, 2.0])
def test_isentropic_nozzle_matches_area_mach_relation(ratio):
    # MEMORY (isentropic A/A* relation, perfect gas). From M = 0.3 in the large pipe, a
    # contraction by A_L/A_s = ratio: the downstream Mach from the relation within 0.1%.
    pvt = IdealGasEntropyPVT()
    k = pvt.k_ratio(100.0)
    a_l = area(8.0)
    a_s = a_l / ratio
    p_l, t_l = 500.0, 100.0
    st = pvt.state(p_l, t_l)
    th = thermo_from_state(pvt, st)
    w = 0.3 * th.c_s * st.rho_g * a_l
    m = M_AIR * pvt.gas_sg
    h_local = th.h / m + (w / (st.rho_g * a_l)) ** 2 / (2.0 * G_C * J_FT_LBF_PER_BTU)
    p_s, t_s, mach_s, resid = isentropic_step(pvt, w, a_l, a_s, p_l, t_l, h_local, 1e6)
    assert abs(resid) < 1e-9
    target = _area_ratio(0.3, k) / ratio  # A_s / A*
    lo, hi = 0.3, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _area_ratio(mid, k) > target:
            lo = mid
        else:
            hi = mid
    m_expected = 0.5 * (lo + hi)
    assert rel(mach_s, m_expected) < 1e-3, (ratio, mach_s, m_expected)


def test_isentropic_nozzle_ratio_3_chokes():
    # COMPUTED: at M = 0.3 the perfect-gas A/A* is about 2.06 for this k (cp 10 Btu/lbmol-R),
    # so a contraction by 3 cannot pass the flow subsonically: no subsonic state reaches
    # the target entropy and the step reports it instead of interpolating.
    pvt = IdealGasEntropyPVT()
    k = pvt.k_ratio(100.0)
    assert 2.0 < _area_ratio(0.3, k) < 2.2
    a_l = area(8.0)
    p_l, t_l = 500.0, 100.0
    st = pvt.state(p_l, t_l)
    th = thermo_from_state(pvt, st)
    w = 0.3 * th.c_s * st.rho_g * a_l
    m = M_AIR * pvt.gas_sg
    h_local = th.h / m + (w / (st.rho_g * a_l)) ** 2 / (2.0 * G_C * J_FT_LBF_PER_BTU)
    with pytest.raises(AreaChangeNotEvaluable):
        isentropic_step(pvt, w, a_l, a_l / 3.0, p_l, t_l, h_local, 1e6)


# --- d. expansion root selection ------------------------------------------------------------

def test_expansion_picks_subsonic_root_and_reports_choke():
    # COMPUTED: with Mach 0.5 in the small pipe the momentum residual has a subsonic and a
    # supersonic root; the subsonic one (p_s above the sonic pressure, Mach < 1) is taken.
    # With the mass rate raised until no root exists the result is the sonic state,
    # Mach = 1 to 1e-6, flagged choked.
    pvt = DryGasPVT(0.65)
    p_l, t_l, a_l, a_s = 2000.0, 150.0, area(8.0), area(4.0)
    w, h_local = _setup(pvt, p_l, t_l, a_l, a_s, 0.5)
    p_s, t_s, mach_s, choked, resid = expansion_borda_carnot(pvt, w, a_l, a_s, p_l, t_l, h_local, 20000.0)
    assert not choked and mach_s < 1.0 and abs(resid) < 1e-6
    st_l = pvt.state(p_l, t_l)
    rhs = p_l * a_l * 144.0 + w * (w / (st_l.rho_g * a_l)) / G_C

    def momentum(p):
        t = temperature_from_total_enthalpy(pvt, p, w, a_s, h_local, t_l)
        st = pvt.state(p, t)
        return p * a_l * 144.0 + w * (w / (st.rho_g * a_s)) / G_C - rhs

    # the chosen root is on the subsonic branch: the residual vanishes there, and every
    # state below the sonic pressure of the small pipe is supersonic (the other branch)
    from wellsim.core.exit import sonic_pressure_adiabatic

    p_star, _ = sonic_pressure_adiabatic(pvt, w, a_s, 1.0, h_local, t_l, 20000.0)
    assert p_s > p_star
    assert abs(momentum(p_s)) < 1e-6
    for p in np.linspace(0.3 * p_star, 0.95 * p_star, 8):
        t = temperature_from_total_enthalpy(pvt, p, w, a_s, h_local, t_l)
        st = pvt.state(p, t)
        assert w / (st.rho_g * a_s) / thermo_from_state(pvt, st).c_s > 1.0
    # the momentum residual rises through zero with p_s on the subsonic branch
    assert momentum(p_star) < 0.0 < momentum(1.5 * p_s)
    # raise the rate until no subsonic root exists
    w2, h2 = _setup(pvt, p_l, t_l, a_l, a_s, 1.5)
    p2, t2, mach2, choked2, resid2 = expansion_borda_carnot(pvt, w2, a_l, a_s, p_l, t_l, h2, 20000.0)
    assert choked2 and abs(mach2 - 1.0) < 1e-6 and resid2 > 0.0


# --- e. full-bore cases unchanged ---------------------------------------------------------

def test_full_bore_case_unchanged():
    # COMPUTED: the fixture well has a single conduit, so the area-change code never runs;
    # values pinned from the tree before M2.5-4 (commit eb4ae66), 1e-12.
    r = solve(case_from_dict(_gas_case()))
    assert sorted(set(r.profile.conduit)) == ["casing 6.184 in"]
    assert r.profile.area_changes == []
    assert rel(r.gas_rate, 254333.426935711) < 1e-12
    assert rel(r.pwf, 3437.5716492426204) < 1e-12
    assert rel(r.exit.p, 143.52670428677206) < 1e-12


# --- f. entropy ---------------------------------------------------------------------------

@pytest.mark.parametrize("tpr", [1.05, 1.3, 1.8, 2.6])
@pytest.mark.parametrize("rho", [0.05, 0.4, 1.2, 2.5])
def test_entropy_departure_integral_matches_quadrature(tpr, rho):
    # COMPUTED: J = int_0^r (Z - 1) dr / r closed form vs Simpson (8000 intervals, the r -> 0
    # limit C1 taken analytically), 1e-8 relative; and s_dep/R = ln Z - J - Tr I.
    j = dak_entropy_integral(rho, tpr)
    n = 8000
    rs = np.linspace(0.0, rho, n + 1)
    f = np.empty(n + 1)
    f[0] = _dak_coeffs(tpr)[0]
    for i in range(1, n + 1):
        f[i] = (dak_partials(rs[i], tpr)[0] - 1.0) / rs[i]
    h = rho / n
    simpson = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
    assert abs(j - simpson) <= 1e-8 * max(1.0, abs(simpson))
    pvt = DryGasPVT(0.65)
    st = pvt.state(3000.0, 150.0)
    from wellsim.core.pvt.real_gas_thermo import dak_departure_integral

    i_val, _ = dak_departure_integral(st.rho_r, st.tpr)
    assert rel(s_departure_from_state(st), R * (math.log(st.z) - dak_entropy_integral(st.rho_r, st.tpr) - st.tpr * i_val)) < 1e-12


def test_entropy_ideal_limits_and_isentropic_step_has_zero_ds():
    # COMPUTED: at low pressure s(T2) - s(T1) at fixed p equals int cp0/T dT (departure
    # negligible), and s(p2) - s(p1) at fixed T equals -R ln(p2/p1); across the isentropic
    # step ds = 0 to 1e-9.
    pvt = DryGasPVT(0.65)
    s1 = entropy_from_state(pvt, pvt.state(1.0, 100.0))
    s2 = entropy_from_state(pvt, pvt.state(1.0, 200.0))
    assert rel(s2 - s1, pvt.s0_from_ref(200.0) - pvt.s0_from_ref(100.0)) < 2e-3
    s3 = entropy_from_state(pvt, pvt.state(2.0, 100.0))
    assert rel(s3 - s1, -R * math.log(2.0)) < 2e-3  # departure difference ~7e-4 at 1 to 2 psia
    p_l, t_l, a_l, a_s = 2000.0, 150.0, area(8.0), area(4.0)
    w, h_local = _setup(pvt, p_l, t_l, a_l, a_s, 0.3)
    p_o, t_o, _, resid = isentropic_step(pvt, w, a_l, a_s, p_l, t_l, h_local, 20000.0)
    assert abs(entropy_from_state(pvt, pvt.state(p_o, t_o)) - entropy_from_state(pvt, pvt.state(p_l, t_l))) < 1e-9
    assert abs(resid) < 1e-9


# --- march integration --------------------------------------------------------------------

def test_examples_carry_area_change_records():
    for name, kind in (("dry_gas_vertical", "expansion"), ("dry_gas_big_bore_liner", "expansion"),
                       ("dry_gas_parted_tubing", "expansion")):
        r = solve(load_case(EXAMPLES_DIR / f"{name}.yaml"))
        assert r.flowing
        acs = r.profile.area_changes
        assert len(acs) == 1 and acs[0].kind == kind
        assert any(n.startswith("sudden expansion") for n in r.notes)
        ac = acs[0]
        assert abs(ac.residual) < 1e-4 or ac.choked
        if name == "dry_gas_parted_tubing":
            assert ac.choked and len(r.profile.chokes) == 1 and abs(ac.mach_after - 1.0) < 1e-6
