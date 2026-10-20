"""M3-1 drift-flux holdup closure. Expected values are COMPUTED from the stated forms
unless tagged otherwise; the forms themselves are MEMORY (Hasan-Kabir 1988/1992,
Harmathy 1960, Taitel et al. 1980) with the FETCHED confirmations listed in
docs/PROVENANCE.md (M3-1). The slug drift ramp and the logistic blends are our own
(COMPUTED, M3-1c/d)."""

from __future__ import annotations

import math

import pytest

from wellsim.cases.schema import DEFAULT_SIGMA_DYNE_CM, Case, Liquid
from wellsim.core.vlp import holdup as hm
from wellsim.core.vlp.holdup import (
    C0_ANNULAR,
    C0_BUBBLY,
    C0_CHURN,
    C0_SLUG,
    DISPERSED_CEILING_NO_SLIP,
    G_SI,
    annular_transition_velocity,
    dispersed_bubble_velocity,
    harmathy_rise_velocity,
    holdup,
    inclination_factor,
    slug_drift,
    taylor_bubble_velocity,
)

# Air-water at ambient, in the module's field units.
RHO_L = 998.0 / 16.018463  # lbm/ft3 (998 kg/m3)
RHO_G = 1.2 / 16.018463  # lbm/ft3 (1.2 kg/m3)
MU_L = 1.0  # cp
SIGMA = 72.0  # dyne/cm
D_O = 6.0  # in
FT_PER_M = 1.0 / 0.3048


def _si_harmathy() -> float:
    return 1.53 * (G_SI * 0.072 * (998.0 - 1.2) / 998.0**2) ** 0.25


def test_harmathy_air_water_hand_value():
    # COMPUTED: 1.53 (9.80665 * 0.072 * 996.8 / 998^2)^0.25 = 0.2494 m/s; the textbook
    # "about 0.25 m/s" for air bubbles in water is MEMORY.
    v = harmathy_rise_velocity(RHO_G, RHO_L, SIGMA)
    assert abs(v - _si_harmathy() * FT_PER_M) < 1e-9
    assert abs(v / FT_PER_M - 0.25) < 0.005


def test_taylor_bubble_pipe_hand_value_and_inclination():
    # COMPUTED: 0.35 sqrt(9.80665 * 0.1524 * 996.8/998) = 0.4275 m/s for a 6 in pipe.
    v = taylor_bubble_velocity(RHO_G, RHO_L, D_O)
    assert abs(v / FT_PER_M - 0.35 * math.sqrt(G_SI * 0.1524 * 996.8 / 998.0)) < 1e-9
    assert abs(v / FT_PER_M - 0.4275) < 5e-4
    assert inclination_factor(90.0) == pytest.approx(1.0)
    t = math.radians(45.0)
    assert inclination_factor(45.0) == pytest.approx(math.sqrt(math.sin(t)) * (1.0 + math.cos(t)) ** 1.2)
    assert inclination_factor(0.0) == 0.0


def test_taylor_bubble_annulus_uses_outer_diameter_and_ratio():
    # FETCHED: Hasan & Kabir 2nd ed. Eq. 4.75, [0.35 + 0.1 d_i/d_o]; at d_i/d_o = 0.5 that
    # is 0.40, against the pipe 0.35 on the same sqrt(g d_o (rho_l - rho_g)/rho_l).
    pipe = taylor_bubble_velocity(RHO_G, RHO_L, D_O)
    ann = taylor_bubble_velocity(RHO_G, RHO_L, D_O, d_i=3.0)
    assert ann / pipe == pytest.approx(0.40 / 0.35)
    # and it scales linearly in d_i/d_o, which is the whole content of the correction
    assert taylor_bubble_velocity(RHO_G, RHO_L, D_O, d_i=6.0) / pipe == pytest.approx(0.45 / 0.35)


def test_taylor_bubble_annulus_meets_the_pipe_form_at_d_i_zero():
    """hk-1. The two branches MUST agree as d_i -> 0, and now do, exactly.

    Until hk-1 the annulus pair was (0.30, 0.22), taken from a secondary source, and this
    test asserted the opposite: that the limit was 0.30/0.35 and that the two correlations
    "do not meet at d_i -> 0, which is a property of the published forms". It is not a
    property of the published forms. Hasan & Kabir Eq. 4.75 is [0.35 + 0.1 d_i/d_o], and
    sec. 4.4.1 says so in as many words: "for a circular channel, when there is no inner
    tubing, d_i is zero, and Eq. 4.75 collapses to Eq. 4.32". The old pair put a 14% step
    at the origin of a continuous physical quantity.
    """
    for d_o in (2.992, 6.0, 8.681, 12.0, 9.625):
        pipe = taylor_bubble_velocity(RHO_G, RHO_L, d_o)
        # the branch is selected by d_i <= 0, so check the limit is approached, not just
        # that the other branch is taken
        assert taylor_bubble_velocity(RHO_G, RHO_L, d_o, d_i=0.0) == pipe
        for eps in (1e-6, 1e-9, 1e-12):
            assert taylor_bubble_velocity(RHO_G, RHO_L, d_o, d_i=eps * d_o) == pytest.approx(
                pipe, rel=1e-5 * max(eps / 1e-6, 1.0))
        # and the coefficient itself is exactly the pipe one in the limit
        assert taylor_bubble_velocity(RHO_G, RHO_L, d_o, d_i=1e-12) / pipe == pytest.approx(
            1.0, rel=1e-9)


def test_slug_drift_ramp_shape():
    # COMPUTED (M3-1d): v_H at and below alpha 0.25, v_TB at and above 0.52, linear between.
    v_h, v_tb = 0.8, 1.4
    assert slug_drift(0.10, v_h, v_tb) == v_h
    assert slug_drift(0.25, v_h, v_tb) == v_h
    assert slug_drift(0.385, v_h, v_tb) == pytest.approx(0.5 * (v_h + v_tb))
    assert slug_drift(0.52, v_h, v_tb) == pytest.approx(v_tb)
    assert slug_drift(0.80, v_h, v_tb) == pytest.approx(v_tb)


@pytest.mark.parametrize(
    "v_sg, v_sl, expect",
    [
        (0.2, 2.0, "bubbly"),
        (1.0, 1.0, "slug"),
        (5.0, 1.0, "slug"),
        (25.0, 1.0, "churn"),
        (100.0, 1.0, "annular"),
        (5.0, 30.0, "dispersed_bubbly"),
        (0.0, 1.0, "liquid"),
        (1.0, 0.0, "gas"),
    ],
)
def test_regimes_and_zuber_findlay_identity(v_sg, v_sl, expect):
    r = holdup(v_sg, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    assert r.regime == expect
    assert r.v_m == pytest.approx(v_sg + v_sl)
    assert 0.0 <= r.alpha_g <= 1.0
    assert r.h_l == pytest.approx(1.0 - r.alpha_g)
    if expect not in ("liquid", "gas"):
        # COMPUTED: alpha_g (C0 v_m + v_d) = v_sg by construction (to the 1e-8 fixed-point
        # tolerance where v_d depends on alpha_g).
        assert r.alpha_g * (r.c0 * r.v_m + r.v_d) == pytest.approx(v_sg, rel=1e-7)
        assert r.v_g == pytest.approx(r.c0 * r.v_m + r.v_d)
        assert 0 <= r.iterations <= 50  # 0 with the closed-form solve (perf step 3)
    expected_c0 = {
        "bubbly": C0_BUBBLY, "dispersed_bubbly": C0_BUBBLY, "slug": C0_SLUG,
        "churn": C0_CHURN, "annular": C0_ANNULAR, "liquid": 1.0, "gas": 1.0,
    }[expect]
    assert r.c0 == pytest.approx(expected_c0, abs=0.02)  # blend leak near a switch
    assert r.warnings == ()


def test_bubbly_to_slug_boundary_reproduces_hasan_kabir_coefficients():
    # COMPUTED: alpha_bubbly = 0.25 with C0 = 1.2 gives v_sg = (0.3 v_sl + 0.25 v_d)/0.7
    # = 0.4286 v_sl + 0.3571 v_d; Hasan-Kabir print 0.429 and 0.357 (MEMORY). M3-1d: the
    # switch is a hard step in the name only; C0 and v_d are continuous through it.
    v_sl = 3.0
    v_d = harmathy_rise_velocity(RHO_G, RHO_L, SIGMA)
    v_crit = (0.3 * v_sl + 0.25 * v_d) / 0.7
    assert v_crit == pytest.approx(0.429 * v_sl + 0.357 * v_d, rel=2e-3)
    below = holdup(v_crit * (1 - 1e-10), v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    above = holdup(v_crit * (1 + 1e-10), v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    assert below.regime == "bubbly" and above.regime == "slug"
    assert below.regime_weight == 1.0 and above.regime_weight > 0.9999
    assert below.alpha_g == pytest.approx(0.25, abs=1e-6)
    assert abs(above.c0 - below.c0) < 1e-9 and abs(above.v_d - below.v_d) < 1e-9
    assert above.v_d == pytest.approx(v_d, abs=1e-5)  # the ramp starts at v_H
    far = holdup(1.2 * v_crit, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    assert far.regime == "slug" and far.alpha_g > 0.25
    assert far.v_d == pytest.approx(slug_drift(far.alpha_g, v_d, taylor_bubble_velocity(RHO_G, RHO_L, D_O)))


def test_slug_churn_boundary_and_annular_alpha():
    # M3-1e: churn needs v_m > v_ms and v_sg > 1.08 v_sl; at v_sl = 3 the dispersed line
    # binds (v_ms - v_sl = 15.7 ft/s > 3.24), so the logistic is centred there.
    v_sl = 3.0
    v_ch = dispersed_bubble_velocity(RHO_G, RHO_L, MU_L, SIGMA, D_O) - v_sl
    assert v_ch > DISPERSED_CEILING_NO_SLIP * v_sl
    v_h, v_tb = harmathy_rise_velocity(RHO_G, RHO_L, SIGMA), taylor_bubble_velocity(RHO_G, RHO_L, D_O)
    slug = holdup(0.8 * v_ch, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    churn = holdup(1.2 * v_ch, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    at = holdup(v_ch, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    assert slug.regime == "slug" and churn.regime == "churn"
    assert holdup(1.2 * DISPERSED_CEILING_NO_SLIP * v_sl, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O).regime == "slug"
    for r in (slug, churn, at):  # churn uses the slug drift ramp (annular logistic leaks 6e-6 at 19 ft/s)
        assert r.v_d == pytest.approx(slug_drift(r.alpha_g, v_h, v_tb), rel=1e-4)
    assert at.c0 == pytest.approx(0.5 * (C0_SLUG + C0_CHURN), abs=1e-3)
    v_ann = annular_transition_velocity(RHO_G, RHO_L, SIGMA)
    # COMPUTED: 3.1 (9.80665 * 0.072 * 996.8 / 1.2^2)^0.25 = 14.6 m/s for air-water.
    assert v_ann / FT_PER_M == pytest.approx(3.1 * (G_SI * 0.072 * 996.8 / 1.44) ** 0.25)
    assert abs(v_ann / FT_PER_M - 14.6) < 0.1
    r = holdup(v_ann * 1.5, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    assert r.regime == "annular" and r.v_d == pytest.approx(0.0, abs=1e-3)
    assert r.alpha_g == pytest.approx(v_ann * 1.5 / (v_ann * 1.5 + v_sl), rel=1e-4)


def test_low_v_sl_switch_is_bubbly_to_slug_and_continuous():
    # M3-1e: at v_sl = 0.2 the no-slip ceiling 1.08 v_sl lies below the bubbly-slug switch
    # but the dispersed line does not (v_ms - v_sl = 13.7 ft/s), so slug follows bubbly and
    # churn starts only at the dispersed line.
    v_sl = 0.2
    v_h = harmathy_rise_velocity(RHO_G, RHO_L, SIGMA)
    v_tr = (0.3 * v_sl + 0.25 * v_h) / 0.7
    assert DISPERSED_CEILING_NO_SLIP * v_sl < v_tr < dispersed_bubble_velocity(RHO_G, RHO_L, MU_L, SIGMA, D_O) - v_sl
    lo = holdup(v_tr * (1 - 1e-10), v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    hi = holdup(v_tr * (1 + 1e-10), v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    assert lo.regime == "bubbly" and hi.regime == "slug"
    assert abs(hi.c0 - lo.c0) < 1e-9 and abs(hi.v_d - lo.v_d) < 1e-9 and abs(hi.alpha_g - lo.alpha_g) < 1e-9
    assert holdup(10.0, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O).regime == "slug"
    assert holdup(20.0, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O).regime == "churn"


def test_single_phase_gas_guard():
    # M3-1c: v_sl = 0 is the single-phase gas guard whatever v_sg.
    for v_sg in (200.0, 2.0):
        r = holdup(v_sg, 0.0, RHO_G, RHO_L, MU_L, SIGMA, D_O)
        assert r.regime == "gas" and r.alpha_g == 1.0 and r.c0 == 1.0 and r.v_d == 0.0


def test_dispersed_bubble_line_value_and_cap():
    # COMPUTED from the Taitel et al. form in SI (d = 0.1524 m, nu = 1.002e-6 m2/s).
    nu = 1.0e-3 / 998.0
    v_si = 4.0 * 0.1524**0.429 * (0.072 / 998.0) ** 0.089 * nu ** (-0.072) * (G_SI * 996.8 / 998.0) ** 0.446
    assert dispersed_bubble_velocity(RHO_G, RHO_L, MU_L, SIGMA, D_O) / FT_PER_M == pytest.approx(v_si)
    assert 5.0 < v_si < 7.0  # order of magnitude sanity, COMPUTED
    # Above the line but with alpha_bubbly >= 0.52 the flow is not dispersed bubbly.
    r = holdup(60.0, 30.0, RHO_G, RHO_L, MU_L, SIGMA, D_O)
    assert r.regime != "dispersed_bubbly"


def test_sigma_default_is_warned_never_silent():
    r = holdup(1.0, 1.0, RHO_G, RHO_L, MU_L, None, D_O)
    assert r.sigma == DEFAULT_SIGMA_DYNE_CM == 30.0
    assert len(r.warnings) == 1 and "30 dyne/cm" in r.warnings[0]
    r2 = holdup(1.0, 1.0, RHO_G, RHO_L, MU_L, 30.0, D_O)
    assert r2.warnings == () and r2.alpha_g == r.alpha_g


def test_annulus_changes_slug_holdup_only_through_v_tb():
    # (2.2, 3.0) sits between the bubbly-slug and slug-churn switches with blend
    # residuals below 1e-3; the drift follows the ramp with the annulus v_TB on top.
    v_h = harmathy_rise_velocity(RHO_G, RHO_L, SIGMA)
    pipe = holdup(2.2, 3.0, RHO_G, RHO_L, MU_L, SIGMA, 8.0)
    ann = holdup(2.2, 3.0, RHO_G, RHO_L, MU_L, SIGMA, 8.0, d_i=4.0)
    assert pipe.regime == ann.regime == "slug"
    assert pipe.v_d == pytest.approx(slug_drift(pipe.alpha_g, v_h, taylor_bubble_velocity(RHO_G, RHO_L, 8.0)), rel=1e-3)
    assert ann.v_d == pytest.approx(slug_drift(ann.alpha_g, v_h, taylor_bubble_velocity(RHO_G, RHO_L, 8.0, 4.0)), rel=1e-3)
    assert ann.v_d > pipe.v_d and ann.alpha_g < pipe.alpha_g  # faster Taylor bubble, less gas held up


def test_inclination_scales_slug_drift():
    # The Hasan-Kabir factor sqrt(sin theta) (1 + cos theta)^1.2 exceeds 1 at intermediate
    # inclinations (1.49 at 30 deg): Taylor bubbles rise faster in inclined pipes than in
    # vertical ones (Zukoski 1966, MEMORY), and the factor goes to 0 at horizontal. The
    # bubbly drift carries sin theta (M3-1b, NEEDS HAND CHECK).
    v_h, v_tb = harmathy_rise_velocity(RHO_G, RHO_L, SIGMA), taylor_bubble_velocity(RHO_G, RHO_L, D_O)
    vert = holdup(2.2, 3.0, RHO_G, RHO_L, MU_L, SIGMA, D_O, theta=90.0)
    inc = holdup(2.2, 3.0, RHO_G, RHO_L, MU_L, SIGMA, D_O, theta=30.0)
    assert vert.regime == inc.regime == "slug"
    assert vert.v_d == pytest.approx(slug_drift(vert.alpha_g, v_h, v_tb), rel=1e-3)
    f, s = inclination_factor(30.0), math.sin(math.radians(30.0))
    assert inc.v_d == pytest.approx(slug_drift(inc.alpha_g, v_h * s, v_tb * f), rel=1e-3)
    assert f > 1.0 > inclination_factor(5.0)


def test_alpha_monotone_in_v_sg_within_bubbly_and_bounded_on_a_grid():
    prev = 0.0
    for i in range(1, 40):
        v_sg = 0.005 * i
        r = holdup(v_sg, 2.0, RHO_G, RHO_L, MU_L, SIGMA, D_O)
        assert r.regime == "bubbly"
        assert r.alpha_g > prev
        prev = r.alpha_g
    for v_sg in (0.01, 0.1, 1.0, 10.0, 100.0):
        for v_sl in (0.0, 0.1, 1.0, 10.0):
            for d_i in (0.0, 2.0):
                r = holdup(v_sg, v_sl, RHO_G, RHO_L, MU_L, SIGMA, D_O, d_i=d_i, theta=60.0)
                assert 0.0 < r.alpha_g <= 1.0 and r.regime in hm.REGIMES


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(v_sg=-1.0),
        dict(v_sl=-1.0),
        dict(v_sg=0.0, v_sl=0.0),
        dict(rho_l=RHO_G),
        dict(rho_g=0.0),
        dict(mu_l=0.0),
        dict(sigma=0.0),
        dict(d_i=6.0),
        dict(d_i=-1.0),
        dict(d_o=0.0),
        dict(theta=95.0),
        dict(theta=-5.0),
    ],
)
def test_invalid_inputs_raise(kwargs):
    base = dict(v_sg=1.0, v_sl=1.0, rho_g=RHO_G, rho_l=RHO_L, mu_l=MU_L, sigma=SIGMA, d_o=D_O)
    base.update(kwargs)
    with pytest.raises(ValueError):
        holdup(**base)


# --- schema ---------------------------------------------------------------------------

def test_liquid_schema_sigma_default_and_case_attachment(gas_case_dict):
    # rho_l is stored in ppg (the schema's one density kind, test_schema_review); the core
    # reads rho_l_lbm_ft3. COMPUTED: 998 kg/m3 = 62.30 lbm/ft3 = 8.328 ppg.
    liq = Liquid(rho_l=8.33, mu_l=MU_L, cp_l=1.0)
    sigma, warn = liq.sigma_or_default()
    assert sigma == 30.0 and warn is not None and "30 dyne/cm" in warn
    liq2 = Liquid(rho_l="998 kg/m3", mu_l=MU_L, cp_l=1.0, sigma=72.0)
    assert liq2.rho_l == pytest.approx(RHO_L / 7.480519480, rel=1e-6)
    assert liq2.rho_l_lbm_ft3 == pytest.approx(RHO_L, rel=1e-6)
    assert Liquid(rho_l="62.3 lbm/ft3", mu_l=MU_L, cp_l=1.0).rho_l_lbm_ft3 == pytest.approx(62.3)
    assert liq2.sigma_or_default() == (72.0, None)
    with pytest.raises(ValueError):
        Liquid(rho_l=8.33, mu_l=MU_L, cp_l=1.0, sigma=0.0)
    with pytest.raises(ValueError):  # a bare lbm/ft3 number is refused as a units mistake
        Liquid(rho_l=62.3, mu_l=MU_L, cp_l=1.0)
    d = dict(gas_case_dict)
    d["liquid"] = {"rho_l": 7.0, "mu_l": 0.8, "cp_l": 0.5}
    case = Case(**d)
    assert case.liquid is not None and case.liquid.sigma is None
    assert Case(**gas_case_dict).liquid is None
