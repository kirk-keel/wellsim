"""Gas IPR (spec 4.2): pseudo-pressure with non-Darcy skin.

No published worked example with a printed rate was FETCHED during M2 (see
docs/PROVENANCE.md); the checks here derive the field-unit constants from SI and test
limits, monotonicity and the two directions of the equation against each other.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wellsim.core.ipr.gas import (
    D_CONSTANT,
    GEERTSMA_FIELD,
    IPR_CONSTANT,
    GasIPR,
    PseudoPressure,
    beta_firoozabadi_katz,
    beta_geertsma,
    beta_jones,
)
from wellsim.core.pvt import DryGasPVT
from wellsim.core.pvt.base import M_AIR, P_SC, R_PSIA_FT3, T_SC_R


def rel(a, b):
    return abs(a - b) / abs(b)


class ConstantMuZ:
    """Fake PVT with mu Z constant: m(p) = p^2 / (mu Z) exactly."""

    gas_sg = 0.65

    def __init__(self, mu=0.02, z=0.9):
        self.mu, self._z = mu, z

    def z(self, p, t):
        return self._z

    def gas_viscosity(self, p, t):
        return self.mu


def test_pseudo_pressure_ideal_limit():
    # COMPUTED: with mu and Z constant, m(p) = p^2/(mu Z). Nodes are exact (Simpson on a
    # quadratic); between nodes the linear interpolation error is h^2/(4 p^2) relative,
    # about 1e-5 at 2000 psia for the 15 psi node spacing used here.
    pvt = ConstantMuZ()
    m = PseudoPressure(pvt, 200.0, 6000.0)
    for p in (1000.0, 2345.6, 6000.0):
        assert rel(m(p), p * p / (pvt.mu * pvt._z)) < 1e-4
        assert abs(m.inverse(m(p)) - p) < 1e-4 * p
    for p in m.p[1:]:
        assert rel(m(p), p * p / (pvt.mu * pvt._z)) < 1e-9
    assert m(0.0) == 0.0


def test_pseudo_pressure_real_gas_monotonic_and_bounded():
    # COMPUTED: real gas m(p) is increasing; below ~2000 psia it tracks p^2/(mu Z) and at
    # high pressure it grows more slowly than p^2 (mu Z rises).
    pvt = DryGasPVT(0.65)
    m = PseudoPressure(pvt, 200.0, 8000.0)
    assert np.all(np.diff(m.m) > 0.0)
    p = 1000.0
    assert rel(m(p), p * p / (pvt.gas_viscosity(p, 200.0) * pvt.z(p, 200.0))) < 0.05
    assert m(8000.0) / m(4000.0) < 4.0


def test_ipr_constant_1422_derived_from_si():
    # COMPUTED: q_sc = pi k h T_sc dm / (p_sc T ln(re/rw)) in SI, converted to
    # q Mscf/d, k md, h ft, T degR, m psia^2/cp gives q = k h dm / (C T ...) with C = 1422.
    k_si = 9.869233e-16  # m2 per md
    h_si = 0.3048
    t_sc = T_SC_R / 1.8
    p_sc = P_SC * 6894.757
    dm_si = 6894.757**2 / 1e-3  # (psia^2/cp) -> Pa^2/(Pa s)
    q_to_mscfd = 86400.0 / 0.028316847 / 1000.0  # m3/s -> Mscf/d
    coef = q_to_mscfd * math.pi * k_si * h_si * t_sc * dm_si / p_sc * 1.8  # per (k h dm / T_R)
    assert rel(1.0 / coef, IPR_CONSTANT) < 0.002


def test_non_darcy_d_constant_derived_from_si():
    # COMPUTED: integrating Forchheimer radially with mass flux over h_p gives
    # D = beta rho_sc k h / (2 pi mu h_p^2 r_w); in field units the coefficient is 2.222e-15
    # (beta 1/ft, k md, h/hp/rw ft, mu cp, q Mscf/d, rho_sc = 0.07632 gamma lbm/ft3).
    rho_sc_si = P_SC * M_AIR / (R_PSIA_FT3 * T_SC_R) * 16.018463  # kg/m3 per unit gamma
    coef = (
        (1.0 / 0.3048)          # beta 1/ft -> 1/m
        * rho_sc_si
        * 9.869233e-16          # k md -> m2
        * 0.3048                # h ft -> m
        * (1000.0 * 0.028316847 / 86400.0)  # q Mscf/d -> m3/s
        / (2.0 * math.pi * 1e-3 * 0.3048**2 * 0.3048)  # mu cp -> Pa s, hp^2 rw ft -> m
    )
    assert rel(coef, D_CONSTANT) < 0.002


def test_geertsma_field_constant_from_cgs():
    # COMPUTED conversion of the VERIFIED cgs form beta = 0.005/(phi^5.5 sqrt(k)), beta 1/cm,
    # k cm^2 (Geertsma 1974 via Elsanoose et al. 2022 eq. 5) to 1/ft and md:
    # 1 md = 9.869233e-12 cm^2, 1 ft = 30.48 cm.
    c = 0.005 / math.sqrt(9.869233e-12) * 30.48
    assert rel(c, GEERTSMA_FIELD) < 1e-4
    assert rel(beta_geertsma(100.0, 0.2), c / (0.2**5.5 * 10.0)) < 1e-12


def test_beta_correlations_orders_of_magnitude():
    # COMPUTED: all three land within a decade of each other at 150 md, phi 0.2, and all
    # fall with permeability.
    b = [beta_jones(150.0), beta_geertsma(150.0, 0.2), beta_firoozabadi_katz(150.0)]
    assert max(b) / min(b) < 10.0
    for fn in (beta_jones, beta_firoozabadi_katz):
        assert fn(10.0) > fn(100.0) > fn(1000.0)
    with pytest.raises(ValueError):
        beta_geertsma(100.0, None)


@pytest.fixture(scope="module")
def ipr():
    return GasIPR(
        pvt=DryGasPVT(0.65), p_res=5000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0, r_w=0.354, skin=2.0
    )


def test_ipr_rate_and_pwf_are_inverses(ipr):
    # COMPUTED
    for pwf in (4500.0, 3000.0, 1000.0, 500.0):
        q = ipr.rate(pwf)
        assert q > 0.0
        assert abs(ipr.pwf(q) - pwf) < 0.05
    assert ipr.rate(5000.0) == 0.0
    assert ipr.pwf(0.0) == 5000.0
    assert ipr.pwf(10.0 * ipr.aof()) is None


def test_ipr_monotonic_and_non_darcy_reduces_rate(ipr):
    # COMPUTED: rate falls as pwf rises (above the ~300 psia turnover described in
    # GasIPR.pwf); dropping the non-Darcy term raises the rate, and the difference grows
    # with drawdown.
    pwfs = np.linspace(500.0, 4900.0, 23)
    qs = [ipr.rate(p) for p in pwfs]
    assert all(a > b for a, b in zip(qs, qs[1:]))
    darcy = GasIPR(
        pvt=ipr.pvt, p_res=5000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0, r_w=0.354, skin=2.0,
        include_non_darcy=False,
    )
    small = darcy.rate(4900.0) / ipr.rate(4900.0)
    large = darcy.rate(500.0) / ipr.rate(500.0)
    assert 1.0 < small < large
    assert large > 1.1  # non-Darcy matters at blowout drawdowns


def test_ipr_low_rate_limit_is_darcy(ipr):
    # COMPUTED: at small drawdown D q -> 0 and q -> k h dm / (1422 T geom).
    pwf = 4990.0
    dm = ipr.m(ipr.p_res) - ipr.m(pwf)
    q_darcy = ipr.k * ipr.h * dm / (IPR_CONSTANT * ipr.t_r * ipr.geom)
    assert rel(ipr.rate(pwf), q_darcy) < 0.01


def test_ipr_user_beta_and_partial_interval():
    # COMPUTED: a user beta is used as given; a shorter perforated interval raises D.
    pvt = DryGasPVT(0.65)
    a = GasIPR(pvt=pvt, p_res=5000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0, r_w=0.354, beta=1e7)
    assert a.beta == 1e7 and a.beta_source == "user"
    b = GasIPR(pvt=pvt, p_res=5000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0, r_w=0.354, beta=1e7, h_p=25.0)
    assert rel(b.d_coefficient(3000.0), 4.0 * a.d_coefficient(3000.0)) < 1e-12
    assert b.rate(3000.0) < a.rate(3000.0)


def test_ipr_geertsma_needs_porosity():
    pvt = DryGasPVT(0.65)
    with pytest.raises(ValueError):
        GasIPR(pvt=pvt, p_res=5000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0, r_w=0.354, beta_method="geertsma")
    g = GasIPR(
        pvt=pvt, p_res=5000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0, r_w=0.354,
        beta_method="geertsma", porosity=0.2,
    )
    assert g.beta_source == "geertsma"
