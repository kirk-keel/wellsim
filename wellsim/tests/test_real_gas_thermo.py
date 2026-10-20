"""M2.5-3a: analytic DAK partials, single-solve state, and real-gas departure functions.

All COMPUTED: derivative checks against central finite differences, the closed-form
departure integral against numerical quadrature, cp against a finite difference of h at
constant p, mu_JT against -(dh/dp)_T / cp, plus the regression that the example rates
are unchanged by moving from a finite-difference to the analytic (dp/drho)_T. Validation
against NIST methane data is not pinned here (tolerances await the owner's approval).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from wellsim.cases.loader import case_from_dict, case_to_dict, load_case
from wellsim.core.exit import G_C, drho_dp_isothermal, sonic_velocity
from wellsim.core.pvt import DryGasPVT
from wellsim.core.pvt import dry_gas as dg
from wellsim.core.pvt.base import M_AIR, R_PSIA_FT3, RangeWarning, rankine
from wellsim.core.pvt.dry_gas import dak_partials, dak_reduced_density, z_dak, z_dak_full
from wellsim.core.pvt.real_gas_thermo import (
    R,
    cp_real,
    cv_real,
    dak_departure_integral,
    enthalpy,
    h_departure,
    isenthalpic_outlet_temperature,
    mu_jt,
    sound_speed,
)
from wellsim.core.solver import solve
from wellsim.core.vlp.correlations.single_phase_gas import gas_gradient
from wellsim.core.geometry import FlowSegment
from wellsim.tests.conftest import EXAMPLES_DIR


def rel(a, b):
    return abs(a - b) / abs(b)


TR_GRID = [1.05, 1.2, 1.5, 2.0, 2.5, 3.0]
R_GRID = [0.01, 0.05, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5]


# --- 2. analytic partials ---------------------------------------------------------------

@pytest.mark.parametrize("tpr", TR_GRID)
@pytest.mark.parametrize("rho", R_GRID)
def test_dak_partials_match_central_differences(tpr, rho):
    # COMPUTED: dZ/dr and dZ/dTr vs central differences, 1e-6 relative (or absolute where
    # the derivative passes through zero).
    z, dz_dr, dz_dt = dak_partials(rho, tpr)
    assert z == dg._dak_z_of_rho(rho, tpr)[0]
    h = 1e-4
    fd_r = (dak_partials(rho + h, tpr)[0] - dak_partials(rho - h, tpr)[0]) / (2.0 * h)
    fd_t = (dak_partials(rho, tpr + h)[0] - dak_partials(rho, tpr - h)[0]) / (2.0 * h)
    assert abs(dz_dr - fd_r) <= 1e-6 * max(1.0, abs(fd_r))
    assert abs(dz_dt - fd_t) <= 1e-6 * max(1.0, abs(fd_t))


def test_z_dak_full_is_one_solve_and_consistent():
    # COMPUTED: rho_r Z = 0.27 Ppr/Tpr holds and z_dak agrees with the full solve.
    for ppr, tpr in ((0.5, 1.3), (3.0, 1.6), (12.0, 2.2), (25.0, 1.1)):
        z, rho, dz_dr, dz_dt = z_dak_full(ppr, tpr)
        assert rel(rho * z, 0.27 * ppr / tpr) < 1e-9
        assert z == z_dak(ppr, tpr)
        assert dak_partials(rho, tpr) == (z, dz_dr, dz_dt)
    assert z_dak_full(0.0, 1.5)[0] == 1.0


# --- 3. single-solve state and analytic (dp/drho)_T ------------------------------------

def test_state_carries_analytic_dp_drho():
    # COMPUTED: (dp/drho)_T = (R T / M)(Z + r dZ/dr) equals a central difference of p(rho)
    # obtained by re-solving Z at p +/- dp, to 1e-6.
    pvt = DryGasPVT(0.65, co2=0.02)
    for p, t in ((200.0, 100.0), (2000.0, 180.0), (5000.0, 240.0)):
        st = pvt.state(p, t)
        t_r = rankine(t)
        assert rel(st.dp_drho_t, R_PSIA_FT3 * t_r / (M_AIR * 0.65) * (st.z + st.rho_r * st.dz_drho_r)) < 1e-12
        dp = 1e-3 * p
        drho = pvt.gas_density(p + dp, t) - pvt.gas_density(p - dp, t)
        assert rel(st.dp_drho_t, 2.0 * dp / drho) < 1e-6
        assert rel(drho_dp_isothermal(pvt, p, t), 1.0 / st.dp_drho_t) < 1e-12


def test_ideal_gas_limit_of_dp_drho():
    # COMPUTED: at low pressure (dp/drho)_T -> R T / M.
    pvt = DryGasPVT(0.65)
    st = pvt.state(5.0, 100.0)
    assert rel(st.dp_drho_t, R_PSIA_FT3 * rankine(100.0) / (M_AIR * 0.65)) < 2e-3


def test_gradient_evaluation_uses_one_z_solve(monkeypatch):
    # COMPUTED: gas_gradient triggers exactly one reduced-density solve.
    calls = []
    orig = dg._dak_solve

    def counted(ppr, tpr, rho0=None):
        calls.append(1)
        return orig(ppr, tpr, rho0)

    monkeypatch.setattr(dg, "_dak_solve", counted)
    pvt = DryGasPVT(0.65)
    seg = FlowSegment(top_md=0.0, bottom_md=100.0, d_hyd=4.0, area=math.pi * (2.0 / 12.0) ** 2,
                      roughness=0.0018, sin_theta=1.0, conduit="t")
    gas_gradient(pvt, 1500.0, 150.0, 5.0, seg)
    assert len(calls) == 1


def test_sonic_velocity_reuses_state():
    pvt = DryGasPVT(0.65)
    st = pvt.state(1500.0, 150.0)
    k = pvt.k_ratio(150.0)
    assert sonic_velocity(pvt, 1500.0, 150.0, state=st) == sonic_velocity(pvt, 1500.0, 150.0)
    assert rel(sonic_velocity(pvt, 1500.0, 150.0, state=st), math.sqrt(k * G_C * 144.0 * st.dp_drho_t)) < 1e-12


def test_example_rates_unchanged_by_analytic_derivative():
    # COMPUTED regression, profile path. Originally the rates at f72d4f6 (forward-difference
    # (dp/drho)_T, rate-space nodal): 586649.3960513026, 613185.9727507554, 72351.55918858296,
    # which the analytic derivative moved by -9e-7, -5.0e-6, -1.6e-6 (owner's 1e-5 target,
    # met at 0dc1bad). The pwf-space nodal solve then converged the root tighter than the
    # rate-space Brent had (that one stopped at a 0.5 psi residual), moving big_bore_liner
    # to 1.9e-5 from f72d4f6; the pinned values are now the pwf-space ones (values only,
    # tolerance unchanged at 1e-5).
    expected = {
        "dry_gas_vertical": 586648.877532835,
        "dry_gas_big_bore_liner": 613174.0972666596,
        "dry_gas_parted_tubing": 72351.44439532896,
    }
    # Since M2.5-3b the default temperature model is adiabatic; this regression is on the
    # pre-3b 'profile' path, which is what those values were computed with.
    for name, q in expected.items():
        d = case_to_dict(load_case(EXAMPLES_DIR / f"{name}.yaml"))
        d["temperature"] = {"model": "profile"}
        r = solve(case_from_dict(d))
        assert rel(r.gas_rate, q) < 1e-5, (name, r.gas_rate)


# --- 4. departure functions ---------------------------------------------------------------

@pytest.mark.parametrize("tpr", [1.05, 1.3, 1.8, 2.6])
@pytest.mark.parametrize("rho", [0.05, 0.4, 1.2, 2.5])
def test_departure_integral_matches_quadrature(tpr, rho):
    # COMPUTED: closed-form I = int_0^r (dZ/dTr) dr / r vs composite Simpson, 1e-8 relative;
    # dI/dTr vs a central difference of I.
    i, di = dak_departure_integral(rho, tpr)
    n = 4000
    rs = np.linspace(0.0, rho, n + 1)
    f = np.empty(n + 1)
    f[0] = dak_partials(1e-12, tpr)[2] / 1e-12  # limit dZ/dTr / r -> C1'
    for j in range(1, n + 1):
        f[j] = dak_partials(rs[j], tpr)[2] / rs[j]
    h = rho / n
    simpson = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
    assert abs(i - simpson) <= 1e-8 * max(1.0, abs(simpson))
    dt = 1e-4
    fd = (dak_departure_integral(rho, tpr + dt)[0] - dak_departure_integral(rho, tpr - dt)[0]) / (2.0 * dt)
    assert abs(di - fd) <= 1e-6 * max(1.0, abs(fd))


def _pvt_methane_like():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        return DryGasPVT(0.65)


@pytest.mark.parametrize("p,t", [(500.0, 100.0), (3000.0, 150.0), (6000.0, 250.0), (1500.0, 40.0)])
def test_cp_real_matches_finite_difference_of_enthalpy(p, t):
    # COMPUTED: analytic cp vs central difference of h(T, p) at constant p, step 0.05 degF.
    pvt = _pvt_methane_like()
    dt = 0.05
    fd = (enthalpy(pvt, p, t + dt) - enthalpy(pvt, p, t - dt)) / (2.0 * dt)
    assert rel(cp_real(pvt, p, t), fd) < 1e-6
    assert cp_real(pvt, p, t) > cv_real(pvt, p, t) > 0.0


@pytest.mark.parametrize("p,t", [(500.0, 100.0), (3000.0, 150.0), (6000.0, 250.0)])
def test_mu_jt_matches_isothermal_enthalpy_derivative(p, t):
    # COMPUTED: mu_JT = -(dh/dp)_T / cp with (dh/dp)_T by central difference, step 0.01% of p
    # (a 0.1% step leaves a 2.5e-6 truncation residual); tolerance 1e-5.
    pvt = _pvt_methane_like()
    dp = 1e-4 * p
    dhdp = (enthalpy(pvt, p + dp, t) - enthalpy(pvt, p - dp, t)) / (2.0 * dp)  # Btu/lbmol per psi
    assert rel(mu_jt(pvt, p, t), -dhdp / cp_real(pvt, p, t)) < 1e-5


def test_departures_vanish_at_low_pressure():
    pvt = _pvt_methane_like()
    t = 150.0
    # at 1 psia (Ppr 0.0015) h_dep/RT is second-virial scale, -3.4e-4 here
    assert abs(h_departure(pvt, 1.0, t)) < 1e-3 * R * rankine(t)
    assert rel(cp_real(pvt, 1.0, t), pvt.cp0_molar(t)) < 2e-3
    assert rel(sound_speed(pvt, 1.0, t), sonic_velocity(pvt, 1.0, t)) < 2e-3


def test_sound_speed_formula_and_positive():
    pvt = _pvt_methane_like()
    p, t = 4000.0, 200.0
    st = pvt.state(p, t)
    cp, cv = cp_real(pvt, p, t), cv_real(pvt, p, t)
    assert rel(sound_speed(pvt, p, t), math.sqrt(cp / cv * G_C * 144.0 * st.dp_drho_t)) < 1e-12


def test_isenthalpic_expansion_cools_and_conserves_enthalpy():
    pvt = _pvt_methane_like()
    t_out = isenthalpic_outlet_temperature(pvt, 170.0, 3000.0, 150.0)
    assert t_out < 170.0
    assert abs(enthalpy(pvt, 150.0, t_out) - enthalpy(pvt, 3000.0, 170.0)) < 1e-6


# --- 6. Tpr < 1.05 warning ---------------------------------------------------------------

def test_dak_warns_below_tpr_1_05():
    with pytest.warns(RangeWarning, match="Tpr"):
        z_dak(2.0, 1.02)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RangeWarning)
        z_dak(2.0, 1.06)


def test_pseudocritical_override():
    pvt = DryGasPVT(0.65, tpc=343.08, ppc=668.6)
    assert pvt.pseudocritical_source == "override" and pvt.tpc == 343.08
    with pytest.raises(ValueError):
        DryGasPVT(0.65, tpc=343.08)
