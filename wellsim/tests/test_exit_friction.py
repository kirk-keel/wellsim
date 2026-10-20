"""Exit choke check (spec 4.5), friction factor, and the Brent root finder."""

from __future__ import annotations

import math

import pytest

from wellsim.core.exit import G_C, drho_dp_isothermal, exit_state, gas_mass_rate, sonic_velocity
from wellsim.core.friction import darcy_friction_factor, reynolds_number
from wellsim.core.geometry import circular_area
from wellsim.core.numerics import bracket_upward, brent
from wellsim.core.pvt import DryGasPVT
from wellsim.core.pvt.base import M_AIR, P_SC, R_PSIA_FT3, T_SC_R, rankine


def rel(a, b):
    return abs(a - b) / abs(b)


# --- friction -------------------------------------------------------------------

def test_laminar_friction_factor():
    # COMPUTED: 64/Re
    assert rel(darcy_friction_factor(1000.0, 0.001), 0.064) < 1e-12


def test_colebrook_satisfies_its_own_equation():
    # COMPUTED: the returned f satisfies 1/sqrt(f) = -2 log10(e/3.7 + 2.51/(Re sqrt f)).
    for re, rr in ((1e4, 0.0), (1e5, 1e-3), (2e6, 9.4e-4), (1e8, 0.05)):
        f = darcy_friction_factor(re, rr)
        lhs = 1.0 / math.sqrt(f)
        rhs = -2.0 * math.log10(rr / 3.7 + 2.51 / (re * math.sqrt(f)))
        assert abs(lhs - rhs) < 1e-9


def test_colebrook_smooth_pipe_moody_value():
    # MEMORY: Moody chart, smooth pipe at Re = 1e5 reads f ~ 0.018 (Prandtl smooth-pipe law
    # gives 0.01799). 2% tolerance for the chart reading.
    assert rel(darcy_friction_factor(1e5, 0.0), 0.0180) < 0.02


def test_colebrook_fully_rough_limit():
    # COMPUTED: at very high Re the Colebrook value tends to the von Karman rough-pipe law
    # 1/sqrt(f) = -2 log10(e/3.7d).
    rr = 0.01
    f_rough = 1.0 / (-2.0 * math.log10(rr / 3.7)) ** 2
    assert rel(darcy_friction_factor(1e12, rr), f_rough) < 1e-4


def test_reynolds_number_units():
    # COMPUTED: Re = 1488 rho v d / mu (rho lbm/ft3, v ft/s, d ft, mu cp)
    assert rel(reynolds_number(1.0, 1.0, 1.0, 1.0), 1488.16) < 1e-3


# --- sonic velocity and exit state --------------------------------------------------

def test_sonic_velocity_ideal_gas_limit():
    # COMPUTED: at low pressure Z -> 1 and c -> sqrt(k Z R T g_c / M) with R in
    # ft lbf/(lbmol degR) = 144 * 10.7316.
    pvt = DryGasPVT(0.65)
    t = 100.0
    c = sonic_velocity(pvt, 20.0, t, k=1.28)
    m = M_AIR * 0.65
    c_ideal = math.sqrt(1.28 * 144.0 * R_PSIA_FT3 * rankine(t) * G_C / m)
    assert rel(c, c_ideal) < 0.01


def test_drho_dp_positive_and_near_ideal_at_low_pressure():
    pvt = DryGasPVT(0.65)
    d = drho_dp_isothermal(pvt, 30.0, 100.0)
    ideal = M_AIR * 0.65 / (R_PSIA_FT3 * rankine(100.0))
    assert rel(d, ideal) < 0.01


def test_gas_mass_rate():
    # COMPUTED: 1 MMscf/d of 0.65 gravity gas = 1e6 * 0.07632 * 0.65 / 86400 lbm/s
    rho_sc = P_SC * M_AIR / (R_PSIA_FT3 * T_SC_R)
    assert rel(rho_sc, 0.07632) < 1e-3
    assert rel(gas_mass_rate(1000.0, 0.65), 1e6 * rho_sc * 0.65 / 86400.0) < 1e-12


def test_exit_unchoked_at_low_rate():
    pvt = DryGasPVT(0.65)
    ex = exit_state(pvt, 2000.0, 14.7, 100.0, circular_area(2.441))
    assert not ex.choked
    assert ex.p == 14.7
    assert 0.0 < ex.mach < 1.0


def test_exit_choked_at_high_rate_has_mach_one_above_ambient():
    pvt = DryGasPVT(0.65)
    ex = exit_state(pvt, 200000.0, 14.7, 150.0, circular_area(6.184))
    assert ex.choked
    assert ex.p > 14.7
    assert abs(ex.mach - 1.0) < 1e-6
    assert abs(ex.v - ex.c) / ex.c < 1e-6
    # COMPUTED: for a near-ideal gas the choked pressure scales linearly with rate
    ex2 = exit_state(pvt, 400000.0, 14.7, 150.0, circular_area(6.184))
    assert rel(ex2.p, 2.0 * ex.p) < 0.03


def test_exit_choked_pressure_is_where_mach_hits_one():
    # COMPUTED: just below p* Mach > 1, just above Mach < 1.
    pvt = DryGasPVT(0.65)
    area = circular_area(6.184)
    q = 200000.0
    ex = exit_state(pvt, q, 14.7, 150.0, area)
    w = gas_mass_rate(q, 0.65)

    def mach(p):
        rho = pvt.gas_density(p, 150.0)
        return w / (rho * area) / sonic_velocity(pvt, p, 150.0, rho=rho)

    assert mach(0.99 * ex.p) > 1.0 > mach(1.01 * ex.p)


def test_exit_higher_ambient_can_unchoke():
    pvt = DryGasPVT(0.65)
    area = circular_area(6.184)
    ex = exit_state(pvt, 200000.0, 14.7, 150.0, area)
    ex2 = exit_state(pvt, 200000.0, 2.0 * ex.p, 150.0, area)
    assert not ex2.choked and ex2.p == 2.0 * ex.p


def test_zero_rate_exit():
    pvt = DryGasPVT(0.65)
    ex = exit_state(pvt, 0.0, 14.7, 150.0, circular_area(6.184))
    assert not ex.choked and ex.mach == 0.0


# --- Brent ---------------------------------------------------------------------------

def test_brent_finds_roots():
    assert abs(brent(lambda x: x * x - 2.0, 0.0, 2.0) - math.sqrt(2.0)) < 1e-10
    assert abs(brent(math.cos, 0.0, 2.0) - math.pi / 2.0) < 1e-10
    with pytest.raises(ValueError):
        brent(lambda x: x * x + 1.0, 0.0, 1.0)


def test_bracket_upward():
    f = lambda x: 100.0 - x
    a, fa, b, fb = bracket_upward(f, 1.0, f(1.0))
    assert fa > 0 > fb and a < 100.0 < b
