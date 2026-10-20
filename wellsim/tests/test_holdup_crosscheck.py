"""M3-1b/c: Beggs-Brill cross-check (tests only, never in the model path) and the limit,
continuity and hand-check tests for the blended drift-flux closure.

Beggs and Brill (1973) is coded here from MEMORY: regime map (L1..L4), the horizontal
holdup constants (a, b, c), the uphill inclination constants (d, e, f, g) and
psi = 1 + C [sin(1.8 theta) - sin^3(1.8 theta)/3]. It is a cross-check on liquid holdup
H_L with a 20% band at the bubbly and slug points (M3-1c: the earlier gas-fraction band
was the wrong metric and was deleted). At the annular point Beggs-Brill's H_L is pinned
without a band: Beggs-Brill overpredicts liquid holdup at high gas velocity (it has no
film model; its distributed-flow holdup stays near 5% where the drift-flux annular
closure gives 1%), and that known failure is what the cross-check exists to show.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wellsim.core.vlp.holdup import (
    C0_BUBBLY,
    C0_SLUG,
    REGIMES,
    dispersed_bubble_velocity,
    harmathy_rise_velocity,
    holdup,
    inclination_factor,
    slug_drift,
    taylor_bubble_velocity,
)

G_FT = 32.174  # ft/s2
RHO_L = 998.0 / 16.018463  # lbm/ft3
RHO_G = 1.2 / 16.018463  # lbm/ft3
MU_L, SIGMA, D_O = 1.0, 72.0, 6.0
BAND = 0.20


def _df(v_sg, v_sl, **kw):
    return holdup(v_sg, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O, **kw)


# --- Beggs-Brill 1973, MEMORY, cross-check only ------------------------------------------

_BB_HORIZONTAL = {  # a, b, c in H_L(0) = a lambda^b / N_FR^c            MEMORY
    "segregated": (0.98, 0.4846, 0.0868),
    "intermittent": (0.845, 0.5351, 0.0173),
    "distributed": (1.065, 0.5824, 0.0609),
}
_BB_UPHILL = {  # d, e, f, g in C = (1 - lambda) ln(d lambda^e N_LV^f N_FR^g)   MEMORY
    "segregated": (0.011, -3.768, 3.539, -1.614),
    "intermittent": (2.96, 0.305, -0.4473, 0.0978),
    "distributed": None,  # C = 0, no inclination correction uphill
}


def beggs_brill_holdup(v_sg, v_sl, rho_l, sigma, d_in, theta_deg=90.0):
    """Liquid holdup H_L and Beggs-Brill regime, uphill flow. Field units as holdup().
    MEMORY throughout (Beggs and Brill, JPT May 1973)."""
    v_m = v_sg + v_sl
    lam = v_sl / v_m
    nfr = v_m**2 / (G_FT * d_in / 12.0)
    nlv = 1.938 * v_sl * (rho_l / sigma) ** 0.25  # MEMORY: Duns-Ros liquid velocity number
    l1 = 316.0 * lam**0.302
    l2 = 0.0009252 * lam**-2.4684
    l3 = 0.10 * lam**-1.4516
    l4 = 0.5 * lam**-6.738
    if (lam < 0.01 and nfr < l1) or (lam >= 0.01 and nfr < l2):
        regime = "segregated"
    elif lam >= 0.01 and l2 <= nfr <= l3:
        regime = "transition"
    elif (0.01 <= lam < 0.4 and l3 < nfr <= l1) or (lam >= 0.4 and l3 < nfr <= l4):
        regime = "intermittent"
    elif (lam < 0.4 and nfr >= l1) or (lam >= 0.4 and nfr > l4):
        regime = "distributed"
    else:  # pragma: no cover - the map is a partition
        raise ValueError("Beggs-Brill regime map gap")
    s = math.sin(math.radians(1.8 * theta_deg))

    def h_l(reg):
        a, b, c = _BB_HORIZONTAL[reg]
        h0 = max(a * lam**b / nfr**c, lam)  # H_L(0) >= no-slip holdup
        if _BB_UPHILL[reg] is None:
            corr = 0.0
        else:
            dd, e, f, g = _BB_UPHILL[reg]
            corr = max(0.0, (1.0 - lam) * math.log(dd * lam**e * nlv**f * nfr**g))
        return min(1.0, h0 * (1.0 + corr * (s - s**3 / 3.0)))

    if regime == "transition":
        a_w = (l3 - nfr) / (l3 - l2)
        return a_w * h_l("segregated") + (1.0 - a_w) * h_l("intermittent"), regime
    return h_l(regime), regime


@pytest.mark.parametrize(
    "name, v_sg, v_sl, df_regime, bb_regime",
    [("bubbly", 0.2, 2.0, "bubbly", "intermittent"), ("slug", 1.0, 1.0, "slug", "transition")],
)
def test_beggs_brill_liquid_holdup_within_band(name, v_sg, v_sl, df_regime, bb_regime):
    # H_L within 20% of the drift-flux value (measured 21 Sep 2026: -2.6% bubbly, -8.7% slug).
    r = _df(v_sg, v_sl)
    h_bb, reg = beggs_brill_holdup(v_sg, v_sl, RHO_L, SIGMA, D_O)
    assert r.regime == df_regime and reg == bb_regime
    d_h = h_bb / r.h_l - 1.0
    assert abs(d_h) < BAND, f"{name}: H_L delta {d_h:.1%}"


def test_beggs_brill_annular_liquid_holdup_pinned():
    # Beggs-Brill overpredicts liquid holdup at high gas velocity: at v_sg = 100, v_sl = 1
    # ft/s its distributed-flow H_L is 0.049 where the drift-flux annular closure gives
    # 0.010 (the no-slip 0.0099). This is the known failure the cross-check exists to
    # show; no band. The value is pinned (COMPUTED from the MEMORY form).
    r = _df(100.0, 1.0)
    h_bb, reg = beggs_brill_holdup(100.0, 1.0, RHO_L, SIGMA, D_O)
    assert r.regime == "annular" and reg == "distributed"
    assert h_bb == pytest.approx(0.0489, abs=5e-4)
    assert r.h_l == pytest.approx(1.0 / 101.0, rel=1e-3)
    assert h_bb > 4.0 * r.h_l


def test_beggs_brill_internal_consistency():
    # COMPUTED sanity on the MEMORY implementation: H_L >= no-slip, H_L <= 1, no
    # inclination correction at horizontal, distributed flow has psi = 1 uphill.
    for v_sg, v_sl in [(0.2, 2.0), (1.0, 1.0), (5.0, 1.0), (100.0, 1.0), (5.0, 30.0)]:
        lam = v_sl / (v_sg + v_sl)
        h, reg = beggs_brill_holdup(v_sg, v_sl, RHO_L, SIGMA, D_O)
        h0, _ = beggs_brill_holdup(v_sg, v_sl, RHO_L, SIGMA, D_O, theta_deg=0.0)
        assert lam - 1e-12 <= h0 <= h <= 1.0, (v_sg, v_sl, reg, h0, h)
        if reg == "distributed":
            assert h == pytest.approx(h0)


# --- limits -------------------------------------------------------------------------------

def test_alpha_goes_to_zero_as_v_sg_goes_to_zero():
    # COMPUTED: in bubbly flow alpha_g / v_sg -> 1 / (C0 v_sl + v_d) as v_sg -> 0.
    v_sl = 2.0
    slope = 1.0 / (C0_BUBBLY * v_sl + harmathy_rise_velocity(RHO_G, RHO_L, SIGMA))
    prev = None
    for v_sg in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        r = _df(v_sg, v_sl)
        assert r.regime == "bubbly" and r.regime_weight > 0.999
        if prev is not None:
            assert r.alpha_g < prev
        prev = r.alpha_g
        assert r.alpha_g / v_sg == pytest.approx(slope, rel=0.1 * v_sg / 1e-1 + 1e-9)
    assert prev < 1e-6
    assert _df(0.0, v_sl).alpha_g == 0.0


def test_v_sl_zero_is_the_single_phase_gas_guard():
    # M3-1c: v_sl = 0 returns alpha_g = 1, regime "gas", C0 = 1, v_d = 0, whatever v_sg;
    # the march hands off to the single-phase gas solver there (TODO-M3).
    for v_sg in (0.01, 2.0, 200.0):
        r = _df(v_sg, 0.0)
        assert r.regime == "gas" and r.alpha_g == 1.0 and r.c0 == 1.0 and r.v_d == 0.0
        assert r.v_g == v_sg and r.regime_weight == 1.0


def test_horizontal_gives_zero_drift_in_every_regime():
    # theta = 0: sqrt(sin 0) = 0 on the slug/churn drift, sin 0 = 0 on the bubbly drift
    # (bubbly v_d carries sin theta, M3-1b, NEEDS HAND CHECK), annular has none.
    seen = set()
    for v_sg, v_sl in [(0.2, 2.0), (2.2, 3.0), (25.0, 1.0), (100.0, 1.0), (5.0, 30.0), (0.0, 1.0), (1.0, 0.0)]:
        r = _df(v_sg, v_sl, theta=0.0)
        assert r.v_d == 0.0, (r.regime, r.v_d)
        seen.add(r.regime)
        if v_sg > 0 and v_sl > 0:
            assert r.alpha_g == pytest.approx(v_sg / (r.c0 * r.v_m))
    assert seen == set(REGIMES)


def test_d_i_zero_reproduces_pipe_form_exactly():
    # Also covered in test_holdup.py; kept here as the step-5 checklist item.
    for d_o in (2.992, 6.0, 8.681, 12.0):
        pipe = taylor_bubble_velocity(RHO_G, RHO_L, d_o)
        assert taylor_bubble_velocity(RHO_G, RHO_L, d_o, d_i=0.0) == pipe
        assert _df(2.2, 3.0, d_i=0.0).v_d == _df(2.2, 3.0).v_d
        hand = 0.35 * math.sqrt(G_FT * d_o / 12.0 * (RHO_L - RHO_G) / RHO_L)
        assert pipe == pytest.approx(hand, rel=1e-6)  # g in ft/s2 to 1e-6 of 9.80665


# --- continuity (M3-1c/d) -------------------------------------------------------------------

SWEEP = np.geomspace(0.01, 200.0, 2000)
DIAMETERS = (3.0, 6.0, 12.0, 17.5)


@pytest.mark.parametrize("theta", [90.0, 30.0], ids=["vertical", "30deg"])
@pytest.mark.parametrize("d_o", DIAMETERS, ids=[f"{d:g}in" for d in DIAMETERS])
@pytest.mark.parametrize("v_sl", [0.2, 1.0, 5.0])
def test_alpha_monotone_non_decreasing_and_derivative_bounded_across_all_transitions(v_sl, d_o, theta):
    # Owner's criterion (M3-1c/d): alpha_g is non-decreasing in v_sg at fixed v_sl over
    # 0.01 to 200 ft/s, the finite-difference derivative d alpha_g / d ln v_sg is bounded
    # (below 1: alpha (1 - C0 alpha v_m / v_g) < 1 for the pure closures), and alpha_g <= 1
    # at every point; on all four diameters, vertical and 30 degrees.
    rs = [holdup(float(v), v_sl, RHO_G, RHO_L, MU_L, SIGMA, d_o, theta=theta) for v in SWEEP]
    a = np.array([r.alpha_g for r in rs])
    assert np.all(a <= 1.0) and np.all(a > 0.0)
    d = np.diff(a) / np.diff(np.log(SWEEP))
    assert d.max() < 1.0, d.max()
    bad = np.where(np.diff(a) < 0.0)[0]
    assert len(bad) == 0, (
        f"v_sl {v_sl}, d_o {d_o}, theta {theta}: alpha_g decreases between v_sg {SWEEP[bad[0]]:.4f} and "
        f"{SWEEP[bad[-1] + 1]:.4f} ({rs[bad[0]].regime} -> {rs[bad[-1] + 1].regime}): "
        f"{a[bad[0]]:.4f} -> {a[bad[-1] + 1]:.4f}"
    )
    assert max(r.iterations for r in rs) <= 50


@pytest.mark.parametrize("d_i_frac", [0.0, 0.5], ids=["pipe", "annulus"])
@pytest.mark.parametrize("d_o", DIAMETERS, ids=[f"{d:g}in" for d in DIAMETERS])
def test_c0_and_v_d_continuous_at_the_bubbly_slug_switch(d_o, d_i_frac):
    # M3-1d: at alpha_g = 0.25 (v_sg = (0.3 v_sl + 0.25 v_H)/0.7) slug and bubbly share C0
    # and v_d; the closure is continuous there to 1e-9 with no logistic.
    v_sl = 3.0
    d_i = d_i_frac * d_o
    v_h = harmathy_rise_velocity(RHO_G, RHO_L, SIGMA)
    v_tr = (0.3 * v_sl + 0.25 * v_h) / 0.7
    lo = holdup(v_tr * (1 - 1e-10), v_sl, RHO_G, RHO_L, MU_L, SIGMA, d_o, d_i=d_i)
    hi = holdup(v_tr * (1 + 1e-10), v_sl, RHO_G, RHO_L, MU_L, SIGMA, d_o, d_i=d_i)
    assert lo.regime == "bubbly" and hi.regime == "slug"
    assert abs(hi.c0 - lo.c0) < 1e-9
    assert abs(hi.v_d - lo.v_d) < 1e-9
    assert abs(hi.alpha_g - lo.alpha_g) < 1e-9
    # The switch sits at the pure-bubbly alpha 0.25; the blended alpha there is 2.5e-7
    # higher because the slug-churn C0 logistic (10 widths away) leaks 1.8e-6 into C0.
    assert lo.alpha_g == pytest.approx(0.25, abs=1e-6)
    assert lo.v_d == pytest.approx(v_h, abs=1e-5)
    # Further into slug the drift follows the ramp with this conduit's v_TB on top.
    far = holdup(1.3 * v_tr, v_sl, RHO_G, RHO_L, MU_L, SIGMA, d_o, d_i=d_i)
    v_tb = taylor_bubble_velocity(RHO_G, RHO_L, d_o, d_i)
    assert far.regime == "slug" and far.v_d == pytest.approx(slug_drift(far.alpha_g, v_h, v_tb), rel=1e-6)


def test_blend_is_pure_far_from_switches_and_halfway_at_the_churn_switch():
    # COMPUTED: the slug-churn switch keeps its logistic, centred on the binding condition
    # (M3-1e: the dispersed line v_ms - v_sl at v_sl = 3): C0 is the mean of 1.2 and 1.15
    # there and within 1e-6 of the pure values at 0.2 and 100 ft/s.
    v_sl = 3.0
    at = holdup(dispersed_bubble_velocity(RHO_G, RHO_L, MU_L, SIGMA, D_O) - v_sl, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    assert at.c0 == pytest.approx(1.175, abs=1e-3)
    assert _df(0.2, 2.0).regime_weight > 1 - 1e-6 and _df(100.0, 1.0).regime_weight > 1 - 1e-6
    assert abs(_df(0.2, 2.0).c0 - C0_BUBBLY) < 1e-6 and abs(_df(100.0, 1.0).c0 - 1.0) < 1e-6


def test_slug_spans_at_least_a_decade_at_v_sl_one():
    # M3-1e: with churn requiring v_m > v_ms, slug at v_sl = 1 ft/s in a 6 in vertical pipe
    # runs from the 0.25 switch (0.72 ft/s) to the dispersed line (17.7 ft/s): 24x.
    slug = [float(v) for v in SWEEP if holdup(float(v), 1.0, RHO_G, RHO_L, MU_L, SIGMA, D_O).regime == "slug"]
    assert slug and max(slug) / min(slug) >= 10.0
    assert min(slug) < 0.8 and max(slug) > 15.0


# --- perf step 2: analytic holdup sensitivity against Richardson (tests only) ---------------

def richardson_alpha_sensitivity(closure, w_g, v_sl, rho_g, rho_l, mu_l, sigma, d_o, d_i, theta, area,
                                 h_rel=1e-4):
    """dalpha_g/drho_g at fixed mass rates by central differences at h and h/2 with
    Richardson extrapolation (error O(h^4)); the routine the model used before the
    analytic derivative (perf pass step 2). Kept in the tests only."""

    def alpha_at(r):
        return closure(w_g / (r * area), v_sl, r, rho_l, mu_l, sigma, d_o, d_i, theta).alpha_g

    h = h_rel * rho_g
    d1 = (alpha_at(rho_g + h) - alpha_at(rho_g - h)) / (2.0 * h)
    d2 = (alpha_at(rho_g + 0.5 * h) - alpha_at(rho_g - 0.5 * h)) / h
    return (4.0 * d2 - d1) / 3.0


@pytest.mark.parametrize("theta", [90.0, 30.0], ids=["vertical", "30deg"])
@pytest.mark.parametrize("d_o", DIAMETERS, ids=[f"{d:g}in" for d in DIAMETERS])
@pytest.mark.parametrize("v_sl", [0.2, 1.0, 5.0])
def test_analytic_sensitivity_matches_richardson_on_the_sweeps(v_sl, d_o, theta):
    # COMPUTED (perf step 2): the closure's implicit-differentiation derivative
    # dalpha_g/drho_g at fixed mass rates equals the Richardson central difference to 1e-6
    # relative on the 24 sweeps (400 points each, three gas densities), away from the ramp
    # kinks at alpha_g 0.25 and 0.52 where a finite difference straddling the kink is not
    # the one-sided derivative.
    area = math.pi / 4.0 * (d_o / 12.0) ** 2
    n = 0
    for rho_g in (0.075, 1.0, 5.0):
        for v in np.geomspace(0.01, 200.0, 400):
            hd = holdup(float(v), v_sl, rho_g, RHO_L, MU_L, SIGMA, d_o, 0.0, theta)
            if min(abs(hd.alpha_g - 0.25), abs(hd.alpha_g - 0.52)) < 1e-5:
                continue
            r = richardson_alpha_sensitivity(holdup, float(v) * rho_g * area, v_sl, rho_g, RHO_L, MU_L, SIGMA,
                                             d_o, 0.0, theta, area)
            assert abs(hd.dalpha_drho_g - r) <= 1e-6 * max(abs(r), 1e-9), (v, rho_g, hd.regime, hd.dalpha_drho_g, r)
            n += 1
    assert n > 1150


def test_analytic_sensitivity_chain_rule():
    # dalpha/drho_g = dalpha/dv_sg (-v_sg/rho_g) + explicit part; in annular flow (C0 = 1,
    # v_d = 0, far from switches) alpha = v_sg/v_m exactly, so dalpha/dv_sg = v_sl/v_m^2 and
    # the explicit part vanishes (COMPUTED).
    hd = _df(100.0, 1.0)
    assert hd.regime == "annular"
    assert hd.dalpha_dv_sg == pytest.approx(1.0 / 101.0**2, rel=1e-4)
    assert hd.dalpha_drho_g == pytest.approx(hd.dalpha_dv_sg * (-100.0 / RHO_G), rel=1e-4)


# --- perf step 3: closed-form closure solve against the fixed point ------------------------

@pytest.mark.parametrize("theta", [90.0, 30.0], ids=["vertical", "30deg"])
@pytest.mark.parametrize("d_o", DIAMETERS, ids=[f"{d:g}in" for d in DIAMETERS])
@pytest.mark.parametrize("v_sl", [0.2, 1.0, 5.0])
def test_closed_form_matches_fixed_point_on_the_sweeps(v_sl, d_o, theta, capsys):
    # COMPUTED (perf step 3): identical alpha_g to 1e-10 on the 24 sweeps; the fallback
    # count over the sweep is printed (expected 0: the pieces partition the ramp).
    import wellsim.core.vlp.holdup as hm

    before = hm.FALLBACKS["count"]
    worst = 0.0
    for v in SWEEP:
        a = holdup(float(v), v_sl, RHO_G, RHO_L, MU_L, SIGMA, d_o, 0.0, theta).alpha_g
        b = holdup(float(v), v_sl, RHO_G, RHO_L, MU_L, SIGMA, d_o, 0.0, theta, solver="fixed_point").alpha_g
        worst = max(worst, abs(a - b))
    fallbacks = hm.FALLBACKS["count"] - before
    print(f"\nclosed form vs fixed point, v_sl {v_sl} d_o {d_o} theta {theta}: worst |dalpha| {worst:.2e}, fallbacks {fallbacks}")
    assert worst < 1e-10
    assert fallbacks == 0


def test_closed_form_matches_fixed_point_on_the_hand_points():
    for name, v_sg, v_sl, regime, v_d, alpha in HAND_POINTS:
        a = _df(v_sg, v_sl)
        b = holdup(v_sg, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O, solver="fixed_point")
        assert abs(a.alpha_g - b.alpha_g) < 1e-10 and a.regime == b.regime == regime
        assert a.iterations == 0 and b.iterations > 0


# --- hand-check points -------------------------------------------------------------------

HAND_POINTS = [
    # name, v_sg, v_sl, regime, v_d ft/s, alpha_g   (COMPUTED by hand, 21 Sep 2026)
    # bubbly: v_d = 1.53 (9.80665 * 0.072 * 996.8 / 998^2)^0.25 = 0.24945 m/s = 0.81842 ft/s
    #         alpha = 0.2 / (1.2 * 2.2 + 0.81842) = 0.057830; blend weights below 1e-6 here
    ("bubbly", 0.2, 2.0, "bubbly", 0.81842, 0.057830),
    # slug:   v_TB = 0.35 sqrt(9.80665 * 0.1524 * 996.8 / 998) = 0.42762 m/s = 1.40296 ft/s.
    #         M3-1d ramp: alpha (1.2 v_m + v_H + k (alpha - 0.25)) = v_sg with
    #         k = (v_TB - v_H)/0.27 = 2.16496, a quadratic k a^2 + (6.24 + v_H - 0.25 k) a - 2.2 = 0
    #         -> alpha = 0.306385, v_d = v_H + k (alpha - 0.25) = 0.940495 ft/s (COMPUTED by
    #         hand, closed form). Unchanged by M3-1e; the churn switch is now far away.
    ("slug", 2.2, 3.0, "slug", 0.940495, 0.306385),
]


@pytest.mark.parametrize("pt", HAND_POINTS, ids=[p[0] for p in HAND_POINTS])
def test_hand_check_points(pt):
    name, v_sg, v_sl, regime, v_d, alpha = pt
    r = _df(v_sg, v_sl)
    print(
        f"\n{name}: v_sg {v_sg} ft/s, v_sl {v_sl} ft/s, rho_g {RHO_G:.5f} lbm/ft3, rho_l {RHO_L:.4f} lbm/ft3, "
        f"mu_l {MU_L} cp, sigma {SIGMA} dyne/cm, d_o {D_O} in, d_i 0, theta 90 -> "
        f"regime {r.regime} (weight {r.regime_weight:.4f}), C0 {r.c0:.5f}, v_d {r.v_d:.5f} ft/s, "
        f"alpha_g {r.alpha_g:.6f} (expected {alpha:.6f}, v_d {v_d:.5f})"
    )
    assert r.regime == regime
    assert r.c0 == pytest.approx({"bubbly": C0_BUBBLY, "slug": C0_SLUG}[regime], rel=1e-4)
    assert r.v_d == pytest.approx(v_d, rel=5e-4)
    assert r.alpha_g == pytest.approx(alpha, rel=5e-4)
