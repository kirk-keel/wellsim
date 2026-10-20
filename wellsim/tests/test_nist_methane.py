"""NIST methane gates for the real-gas departures (M2.5-3a closeout).

FETCHED 21 Sep 2026 from the NIST WebBook fluid-property pages for methane
(Setzmann and Wagner 1991 EOS): fluid.cgi?ID=C74828&Type=IsoTherm at 250, 300, 350 K
and Type=IsoBar at 1 and 0.1 MPa; critical point from cbook.cgi?ID=C74828&Mask=4
(Tc 190.6 K, Pc 46.1 bar, NIST averages). Model: gravity 0.5539, no inerts.

Two runs, both pinned (owner's rule: tolerance at least the observed maximum and no more
than 1.25x it, rounded up):
    true critical point as override -> DAK accuracy gate
    Sutton pseudo-criticals         -> regression gate
Observed maxima (21 Sep 2026): true Tc,Pc: departure 5.09%, mu_JT 4.30%, sound speed
1.96%, throttle 1.69 K; Sutton: 9.49%, 8.41%, 2.21%, 3.81 K.
Bias: DAK departures run a few percent small, so predicted temperatures run warm
(about 2 to 4 K per throttle here).
"""

from __future__ import annotations

import warnings

import pytest

from wellsim.core.pvt import DryGasPVT
from wellsim.core.pvt.base import RangeWarning
from wellsim.core.pvt.real_gas_thermo import enthalpy, isenthalpic_outlet_temperature, mu_jt, sound_speed

MPA = 145.037738  # psia per MPa
KJ_PER_MOL_PER_BTU_LBMOL = 2.326 / 1000.0
TC_R, PC_PSIA = 190.6 * 1.8, 46.1 * 14.503774  # FETCHED NIST critical point

# FETCHED: (T K, p MPa) -> (H kJ/mol, Cp J/(mol K), mu_JT K/MPa, sound speed m/s)
NIST = {
    (250, 0.01): (12.930, 34.282, 6.1936, 413.55), (250, 2): (12.490, 37.689, 6.1300, 401.07),
    (250, 5): (11.748, 45.528, 5.8011, 385.97), (250, 10): (10.339, 66.874, 4.1754, 392.22),
    (250, 20): (8.7031, 66.246, 1.0400, 580.44), (250, 30): (8.3554, 57.835, 0.22956, 760.36),
    (300, 0.01): (14.679, 35.787, 4.3265, 450.03), (300, 0.1): (14.665, 35.869, 4.3216, 449.74),
    (300, 2): (14.367, 37.736, 4.2041, 444.38), (300, 5): (13.884, 41.234, 3.9428, 439.25),
    (300, 10): (13.077, 48.165, 3.2606, 444.53), (300, 20): (11.802, 56.917, 1.5878, 524.86),
    (300, 30): (11.213, 55.836, 0.61064, 652.65),
    (350, 0.01): (16.520, 37.969, 3.0905, 481.93), (350, 2): (16.287, 39.243, 2.9805, 479.96),
    (350, 5): (15.938, 41.313, 2.7849, 479.57), (350, 10): (15.381, 44.954, 2.3779, 487.29),
    (350, 20): (14.467, 50.698, 1.4443, 538.57), (350, 30): (13.919, 52.574, 0.72263, 624.29),
}
# FETCHED isobar rows bracketing the throttle outlets (T K -> H kJ/mol)
ISO_1MPA = {295: 14.341, 300: 14.524}
ISO_01MPA = {250: 12.911, 255: 13.083}

TOL = {
    # departure %, mu_JT %, sound speed %, throttle K
    "true_critical": (6.0, 5.0, 2.4, 2.0),
    "sutton": (11.0, 10.0, 2.5, 4.5),
}
# FETCHED (M2.5-4, 21 Sep 2026, same isotherm pages): entropy S, J/(mol K)
NIST_S = {
    250: {0.01: 120.22, 2: 74.928, 5: 65.152, 10: 55.107, 20: 44.771, 30: 40.799},
    300: {0.01: 126.59, 2: 81.772, 5: 72.954, 10: 65.137, 20: 56.099, 30: 51.227},
    350: {0.01: 132.26, 2: 87.690, 5: 79.288, 10: 72.247, 20: 64.326, 30: 59.575},
}
# entropy departure difference s(T, p) - s(T, 0.01 MPa), % of the NIST value; observed
# maxima 0.697% (true critical point) and 0.964% (Sutton); rule: at least the maximum, at
# most 1.25x, rounded up
TOL_S = {"true_critical": 0.8, "sutton": 1.2}


def degf(t_k):
    return t_k * 1.8 - 459.67


def kelvin(t_f):
    return (t_f + 459.67) / 1.8


def _interp_t(iso, h):
    (t1, h1), (t2, h2) = sorted(iso.items())
    return t1 + (t2 - t1) * (h - h1) / (h2 - h1)


@pytest.fixture(params=["true_critical", "sutton"])
def run(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        warnings.simplefilter("ignore", UserWarning)
        if request.param == "true_critical":
            pvt = DryGasPVT(0.5539, tpc=TC_R, ppc=PC_PSIA)
        else:
            pvt = DryGasPVT(0.5539)
    return request.param, pvt


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        yield


def test_enthalpy_departure(run):
    name, pvt = run
    tol = TOL[name][0] / 100.0
    worst = 0.0
    for tk in (250, 300, 350):
        t = degf(tk)
        h0_m = enthalpy(pvt, 0.01 * MPA, t) * KJ_PER_MOL_PER_BTU_LBMOL
        h0_n = NIST[(tk, 0.01)][0]
        for pm in (2, 5, 10, 20, 30):
            hm = enthalpy(pvt, pm * MPA, t) * KJ_PER_MOL_PER_BTU_LBMOL - h0_m
            hn = NIST[(tk, pm)][0] - h0_n
            worst = max(worst, abs(hm - hn) / abs(hn))
            assert abs(hm - hn) <= tol * abs(hn), (name, tk, pm, hm, hn)
    assert worst > 0.0


def test_entropy_departure(run):
    # FETCHED NIST entropy vs the closed-form DAK entropy departure (M2.5-4): isothermal
    # s(T, p) - s(T, 0.01 MPa), which includes -R ln(p/0.01) and the departure difference.
    from wellsim.core.pvt.real_gas_thermo import entropy

    name, pvt = run
    tol = TOL_S[name] / 100.0
    for tk, rows in NIST_S.items():
        t = degf(tk)
        s0 = entropy(pvt, 0.01 * MPA, t) * 4.1868
        for pm in (2, 5, 10, 20, 30):
            dm = entropy(pvt, pm * MPA, t) * 4.1868 - s0
            dn = rows[pm] - rows[0.01]
            assert abs(dm - dn) <= tol * abs(dn), (name, tk, pm, dm, dn)


def test_joule_thomson_and_sound_speed_at_300k(run):
    name, pvt = run
    tol_jt, tol_w = TOL[name][1] / 100.0, TOL[name][2] / 100.0
    t = degf(300)
    for pm in (0.1, 5, 10, 20):
        jt_m = mu_jt(pvt, pm * MPA, t) / 1.8 * MPA
        w_m = sound_speed(pvt, pm * MPA, t) * 0.3048
        _, _, jt_n, w_n = NIST[(300, pm)]
        assert abs(jt_m - jt_n) <= tol_jt * jt_n, (name, pm, jt_m, jt_n)
        assert abs(w_m - w_n) <= tol_w * w_n, (name, pm, w_m, w_n)


def test_isenthalpic_throttle_outlet_temperature(run):
    name, pvt = run
    tol_k = TOL[name][3]
    for tk, p_in, p_out, iso in ((350, 20, 1, ISO_1MPA), (300, 10, 0.1, ISO_01MPA)):
        t_out_m = kelvin(isenthalpic_outlet_temperature(pvt, degf(tk), p_in * MPA, p_out * MPA))
        t_out_n = _interp_t(iso, NIST[(tk, p_in)][0])
        assert abs(t_out_m - t_out_n) <= tol_k, (name, tk, p_in, p_out, t_out_m, t_out_n)
        assert t_out_m >= t_out_n - 0.5  # the documented warm bias: the model never runs colder
