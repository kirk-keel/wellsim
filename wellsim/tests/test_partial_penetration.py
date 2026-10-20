"""Partial-penetration pseudo-skin (TODO-M3 item 1, M2.5-1).

Papatzacos (1987, SPE-13956-PA) as coded is VERIFIED against Lu et al., Mathematical
Problems in Engineering, vol. 2010, Article 907206, eqs. 2.4 to 2.8 (owner check). The
checks here are limits and trends (COMPUTED), one independent hand value supplied by the
owner (COMPUTED), and a test-only Brons-Marting cross-check (VERIFIED, same source) that
must agree within 10%.
"""

from __future__ import annotations

import math

import pytest

from wellsim.core.ipr.gas import D_CONSTANT, GasIPR
from wellsim.core.ipr.partial_penetration import papatzacos_skin
from wellsim.core.pvt import DryGasPVT


def rel(a, b):
    return abs(a - b) / abs(b)


def brons_marting_skin(h: float, h_p: float, r_w: float, kv_kh: float) -> float:
    # VERIFIED (Lu et al. 2010, eqs. 2.4 to 2.8; test-only cross-check): Brons and Marting (1961) pseudo-skin with the
    # polynomial fit G(b) = 2.948 - 7.363 b + 11.45 b^2 - 4.675 b^3 for an interval
    # at the top of the zone; h_D = (h / r_w) sqrt(kh / kv).
    b = h_p / h
    h_d = (h / r_w) * math.sqrt(1.0 / kv_kh)
    g = 2.948 - 7.363 * b + 11.45 * b**2 - 4.675 * b**3
    return (1.0 / b - 1.0) * (math.log(h_d) - g)


# --- a. full penetration --------------------------------------------------------

def test_full_penetration_gives_zero_skin():
    # COMPUTED: at h_p = h both terms vanish (term1 by the (1/h_pD - 1) factor, term2 by
    # (1/3) sqrt(3 / (1/3)) = 1) for any h_1 = 0, r_w, kv_kh.
    for h, r_w, kv_kh in ((100.0, 0.35, 1.0), (30.0, 0.354, 0.1), (250.0, 0.25, 0.01)):
        assert abs(papatzacos_skin(h, h, 0.0, r_w, kv_kh)) < 1e-10


# --- b. top / bottom symmetry ---------------------------------------------------

@pytest.mark.parametrize("h_p", [5.0, 20.0, 50.0, 80.0])
def test_top_and_bottom_placement_are_symmetric(h_p):
    # COMPUTED: (A - 1)/(B - 1) is the same for h_1D = 0 and h_1D = 1 - h_pD, so the
    # formula is mirror-symmetric about the mid-plane.
    h, r_w, kv_kh = 100.0, 0.35, 0.1
    top = papatzacos_skin(h, h_p, 0.0, r_w, kv_kh)
    bottom = papatzacos_skin(h, h_p, h - h_p, r_w, kv_kh)
    assert abs(top - bottom) < 1e-10


# --- c. centred interval --------------------------------------------------------

def test_centred_interval_has_lower_skin_than_top():
    # COMPUTED: a centred interval draws flow from both sides of the zone.
    h, h_p, r_w, kv_kh = 100.0, 20.0, 0.35, 1.0
    top = papatzacos_skin(h, h_p, 0.0, r_w, kv_kh)
    centred = papatzacos_skin(h, h_p, 0.5 * (h - h_p), r_w, kv_kh)
    assert centred < top
    assert centred > 0.0


# --- d. monotonic trends --------------------------------------------------------

def test_skin_rises_as_penetration_falls():
    # COMPUTED trend, top placement.
    h, r_w, kv_kh = 100.0, 0.35, 0.1
    values = [papatzacos_skin(h, h_pd * h, 0.0, r_w, kv_kh) for h_pd in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_skin_rises_as_kv_kh_falls():
    # COMPUTED trend: r_D shrinks with sqrt(kv/kh), so ln(pi / 2 r_D) grows.
    h, h_p, r_w = 100.0, 20.0, 0.35
    values = [papatzacos_skin(h, h_p, 0.0, r_w, kv_kh) for kv_kh in (1.0, 0.5, 0.1, 0.01, 0.001)]
    assert all(b > a for a, b in zip(values, values[1:]))


# --- e. hand value --------------------------------------------------------------

def test_papatzacos_hand_value():
    # COMPUTED (owner's independent hand calculation, 21 Sep 2026): h = 100, h_p = 20,
    # h_1 = 0, r_w = 0.35, kv/kh = 1 gives s_pp = 15.46. Tolerance 0.5%.
    # Hand trail: h_pD 0.2, r_D 0.0035, A 20, B 6.6667; term1 = 4 ln(448.8) = 24.43;
    # term2 = 5 ln(0.09091 sqrt(19 / 5.6667)) = -8.96.
    s = papatzacos_skin(100.0, 20.0, 0.0, 0.35, 1.0)
    assert rel(s, 15.46) < 0.005


# --- f. Brons-Marting cross-check ------------------------------------------------

def test_brons_marting_agrees_within_10_percent():
    # VERIFIED (Brons-Marting form and G(b) polynomial, Lu et al. 2010; test-only). Same case as the hand
    # value; the two correlations must agree within 10%.
    h, h_p, r_w, kv_kh = 100.0, 20.0, 0.35, 1.0
    s_pap = papatzacos_skin(h, h_p, 0.0, r_w, kv_kh)
    s_bm = brons_marting_skin(h, h_p, r_w, kv_kh)
    assert rel(s_pap, s_bm) < 0.10, (s_pap, s_bm)


# --- g. non-Darcy D and skin combination in the IPR -----------------------------

def _ipr(**kw):
    pvt = DryGasPVT(0.65)
    base = dict(pvt=pvt, p_res=5000.0, t_res=200.0, k=100.0, h=50.0, r_e=1500.0, r_w=0.354, beta=1e7, skin=2.0)
    base.update(kw)
    return GasIPR(**base)


def test_d_at_full_penetration_is_unchanged():
    # COMPUTED: D = 2.222e-15 beta gamma k h / (mu r_w h_p^2) reduces to
    # 2.222e-15 beta gamma k / (mu r_w h) at h_p = h; h_p=None and h_p=h are identical.
    a = _ipr()
    b = _ipr(h_p=50.0)
    p = 3000.0
    mu = a.pvt.gas_viscosity(p, a.t_res)
    expected = D_CONSTANT * a.beta * a.pvt.gas_sg * a.k / (mu * a.r_w * a.h)
    assert rel(a.d_coefficient(p), expected) < 1e-12
    assert a.d_coefficient(p) == b.d_coefficient(p)
    assert a.s_pp == 0.0 and b.s_pp == 0.0
    assert a.s_total == 2.0 and b.s_total == 2.0
    assert not a.kv_kh_defaulted and not b.kv_kh_defaulted


def test_d_scales_with_open_interval_squared():
    # COMPUTED: halving h_p quadruples D at the same beta, k, h and viscosity.
    a = _ipr(kv_kh=0.1)
    b = _ipr(h_p=25.0, kv_kh=0.1)
    assert rel(b.d_coefficient(3000.0), 4.0 * a.d_coefficient(3000.0)) < 1e-12


def test_skin_combination_and_kv_kh_default():
    # COMPUTED: s_total = s_pp + (h / h_p) s_d; kv_kh omitted with h_p < h gives 0.1 and
    # sets the flag the solver warns on.
    ipr = _ipr(h_p=25.0)
    assert ipr.kv_kh == 0.1 and ipr.kv_kh_defaulted
    assert rel(ipr.s_mech, 2.0 * 2.0) < 1e-12
    assert rel(ipr.s_pp, papatzacos_skin(50.0, 25.0, 0.0, 0.354, 0.1)) < 1e-12
    assert rel(ipr.s_total, ipr.s_pp + 4.0) < 1e-12
    assert rel(ipr.geom, math.log(1500.0 / 0.354) - 0.75 + ipr.s_total) < 1e-12
    explicit = _ipr(h_p=25.0, kv_kh=0.1)
    assert not explicit.kv_kh_defaulted and explicit.s_pp == ipr.s_pp
    assert ipr.rate(3000.0) < _ipr().rate(3000.0)


def test_open_interval_must_fit_in_zone():
    with pytest.raises(ValueError):
        papatzacos_skin(100.0, 20.0, 90.0, 0.35, 1.0)
    with pytest.raises(ValueError):
        papatzacos_skin(100.0, 20.0, -1.0, 0.35, 1.0)
