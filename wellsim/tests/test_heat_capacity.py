"""Ideal-gas heat capacity of natural gas (M2.5-2): pure-component cp from NIST WebBook
(FETCHED), quartic fits to the hydrocarbon tables (COMPUTED), gravity pseudo-composition
(COMPUTED), and the k0 wiring into the sonic velocity with a k_override for reproducibility.
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pytest

from wellsim.cases.loader import case_from_dict, case_to_dict, load_case
from wellsim.core.exit import sonic_velocity
from wellsim.core.pvt import DryGasPVT
from wellsim.core.pvt.base import RangeWarning
from wellsim.core.pvt.heat_capacity import (
    CP_TABLES,
    MOLAR_MASS,
    R_BTU_LBMOL_R,
    SHOMATE,
    T_VALID_K,
    _POLY,
    component_cp_j,
    component_h_j,
    cp0_mass,
    cp0_molar,
    delta_h0,
    k0,
    pseudo_composition,
)
from wellsim.core.solver import solve
from wellsim.tests.conftest import EXAMPLES_DIR


def rel(a, b):
    return abs(a - b) / abs(b)


def degf(t_k: float) -> float:
    return t_k * 1.8 - 459.67


# --- fetched data and fits --------------------------------------------------------------

def test_table_fits_reproduce_nist_tables():
    # COMPUTED fit over FETCHED rows (NIST WebBook: CH4 and C2H6 Gurvich, Veyts et al. 1989;
    # C3H8 Chao 1973; CO2 Span-Wagner fluid data at 0.01 MPa plus JANAF; H2S Lemmon-Span
    # fluid data at 0.01 MPa plus JANAF). Max error against every table row in 200 to
    # 600 K below 0.3%.
    assert set(CP_TABLES) == {"CH4", "C2H6", "C3H8", "CO2", "H2S"}
    for name, table in CP_TABLES.items():
        rows = [(t, c) for t, c in table if T_VALID_K[0] <= t <= T_VALID_K[1]]
        assert len(rows) >= 5
        err = max(rel(component_cp_j(t, name), c) for t, c in rows)
        assert err < 0.003, (name, err)


def test_inert_fluid_data_range_check_below_298k():
    # FETCHED (NIST WebBook fluid.cgi isobars at 0.01 MPa, vapor): CO2 (Span and Wagner 1996)
    # 33.282 @216.59, 34.488 @241.59, 35.696 @266.59, 36.870 @291.59 K; H2S (Lemmon and Span
    # 2006) 33.591 @200, 33.617 @225, 33.734 @250, 33.923 @275, 34.172 @300 K, J/(mol K).
    # The NIST Shomate sets (298 K upward) extrapolated to these points miss by -2.24%
    # (CO2, 216.59 K) and +1.60% (H2S, 200 K), so the table fits are used; they must
    # reproduce the fluid data within 0.3%.
    fluid = [("CO2", 216.59, 33.282), ("CO2", 241.59, 34.488), ("CO2", 266.59, 35.696), ("CO2", 291.59, 36.870),
             ("H2S", 200.0, 33.591), ("H2S", 225.0, 33.617), ("H2S", 250.0, 33.734), ("H2S", 275.0, 33.923),
             ("H2S", 300.0, 34.172)]
    for name, t, cp in fluid:
        assert rel(component_cp_j(t, name), cp) < 0.003, (name, t, component_cp_j(t, name))

    def shomate(t_k, c):
        a, b, cc, d, e = c
        x = t_k / 1000.0
        return a + b * x + cc * x * x + d * x**3 + e / (x * x)

    co2 = (24.99735, 55.18696, -33.69137, 7.948387, -0.136638)
    h2s = (26.88412, 18.67809, 3.434203, -3.378702, 0.135882)
    assert shomate(216.59, co2) / 33.282 - 1.0 < -0.02
    assert shomate(200.0, h2s) / 33.591 - 1.0 > 0.015


def test_shomate_matches_nist_table_values():
    # FETCHED (NIST-JANAF table values, Chase 1998, printed on the WebBook table page):
    # N2 29.11 @200, 29.12 @300, 29.25 @400, 29.58 @500, 30.10 @600 J/(mol K).
    # The Shomate fit and its own table agree to the printed precision (0.05%).
    assert set(SHOMATE) == {"N2"}
    checks = [("N2", 200, 29.11), ("N2", 300, 29.12), ("N2", 400, 29.25), ("N2", 500, 29.58), ("N2", 600, 30.10)]
    for name, t, cp in checks:
        assert rel(component_cp_j(t, name), cp) < 0.0005, (name, t)


def test_nitrogen_shomate_ranges_join_continuously():
    # COMPUTED: the 100-500 and 500-2000 K coefficient sets meet at 500 K.
    lo = SHOMATE["N2"][0][1]
    hi = SHOMATE["N2"][1][1]
    t = 0.5

    def sh(c):
        a, b, cc, d, e = c
        return a + b * t + cc * t * t + d * t**3 + e / (t * t)

    assert rel(sh(lo), sh(hi)) < 0.002


def test_memory_cross_check_2_percent():
    # MEMORY (owner's recollection, J/(mol K)); FETCHED/fit values must agree within 2%.
    mem = {("CH4", 300): 35.8, ("CH4", 400): 40.6, ("CH4", 500): 46.5,
           ("C2H6", 300): 52.7, ("C2H6", 400): 65.5, ("C3H8", 300): 73.9,
           ("N2", 300): 29.1, ("CO2", 300): 37.2, ("CO2", 400): 41.3, ("H2S", 300): 34.2}
    for (name, t), v in mem.items():
        assert rel(component_cp_j(t, name), v) < 0.02, (name, t, component_cp_j(t, name), v)


# --- pseudo-composition ----------------------------------------------------------------

def test_pure_methane_gravity_gives_k0_about_1_30():
    # COMPUTED: gamma = 16.0425 / 28.9647 = 0.5539 is all methane; k0 at 300 K about 1.30.
    g = pseudo_composition(0.5539)
    assert g.fractions["CH4"] > 0.999 and g.hc_split == ("CH4", "C2H6")
    assert abs(k0(degf(300.0), g) - 1.30) < 0.01


def test_hand_value_gamma_065_at_300k():
    # COMPUTED (owner hand value): gamma 0.65, no inerts, 300 K gives x_C2H6 = 0.199 and
    # k0 = 1.270, tolerance 0.5%. M_mix = 28.9647 x 0.65 = 18.827 (the code's M_air; the
    # owner's 28.97 moves x by 0.1%, inside tolerance).
    g = pseudo_composition(0.65)
    assert rel(g.fractions["C2H6"], 0.199) < 0.005
    assert rel(k0(degf(300.0), g), 1.270) < 0.005


def test_composition_reproduces_molar_mass_with_inerts():
    # COMPUTED: mole fractions sum to 1 and the composition returns M_air x gamma.
    for sg, inerts in ((0.65, {}), (0.7, dict(n2=0.02, co2=0.03, h2s=0.01)), (1.2, dict(co2=0.1))):
        g = pseudo_composition(sg, **inerts)
        assert abs(sum(g.fractions.values()) - 1.0) < 1e-12
        m = sum(y * MOLAR_MASS[n] for n, y in g.fractions.items())
        assert rel(m, g.molar_mass) < 1e-12


def test_heavy_gas_uses_ethane_propane_and_warns_above_090():
    g = pseudo_composition(1.2)
    assert g.hc_split == ("C2H6", "C3H8") and g.fractions["CH4"] == 0.0
    with pytest.warns(RangeWarning):
        pseudo_composition(0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pseudo_composition(0.85)


def test_gravity_below_methane_or_above_propane_raises():
    with pytest.raises(ValueError):
        pseudo_composition(0.55)  # below methane, no clamping
    with pytest.raises(ValueError):
        pseudo_composition(0.65, co2=0.5)  # inerts heavier than the mixture: M_HC < methane
    with pytest.raises(ValueError):
        pseudo_composition(1.6)  # above propane


def test_below_methane_error_states_minimum_feasible_gravity():
    # COMPUTED: 0.65 with 5% CO2 and 10% H2S; the minimum feasible gravity is
    # (16.0425 x 0.85 + 0.05 x 44.0095 + 0.10 x 34.081) / 28.9647 = 0.6644, and that gravity
    # with the same inerts must construct (all-methane hydrocarbons).
    sg_min = (MOLAR_MASS["CH4"] * 0.85 + 0.05 * MOLAR_MASS["CO2"] + 0.10 * MOLAR_MASS["H2S"]) / 28.9647
    with pytest.raises(ValueError) as e:
        pseudo_composition(0.65, co2=0.05, h2s=0.10)
    msg = str(e.value)
    assert "below methane" in msg and "minimum feasible mixture gravity" in msg
    assert f"{sg_min:.4f}" in msg and "co2 0.0500" in msg and "h2s 0.1000" in msg
    g = pseudo_composition(sg_min * (1.0 + 1e-6), co2=0.05, h2s=0.10)
    assert g.fractions["CH4"] > 0.8499 and g.fractions["C2H6"] < 1e-4


# --- mixture API ----------------------------------------------------------------------

def test_k0_decreases_with_temperature_and_gravity():
    # COMPUTED trends.
    g = pseudo_composition(0.65)
    ks = [k0(degf(t), g) for t in (200.0, 250.0, 300.0, 400.0, 500.0, 600.0)]
    assert all(b < a for a, b in zip(ks, ks[1:]))
    ks = [k0(degf(300.0), pseudo_composition(sg)) for sg in (0.56, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.2)]
    assert all(b < a for a, b in zip(ks, ks[1:]))


def test_cp0_mass_and_units():
    # COMPUTED: cp0_mass = cp0_molar / M; k0 = cp/(cp - R) with R = 1.98588 Btu/(lbmol degR).
    g = pseudo_composition(0.65)
    t = 100.0
    cp = cp0_molar(t, g)
    assert rel(cp0_mass(t, g), cp / g.molar_mass) < 1e-12
    assert rel(k0(t, g), cp / (cp - R_BTU_LBMOL_R)) < 1e-12
    assert rel(R_BTU_LBMOL_R, 1.98588) < 1e-5


def test_delta_h0_matches_quadrature():
    # COMPUTED: analytic integral vs fine-grid Simpson on cp0, 1e-6 relative; both signs;
    # includes the N2 range split at 500 K.
    g = pseudo_composition(0.7, n2=0.05, co2=0.03, h2s=0.01)
    for t1, t2 in ((60.0, 250.0), (250.0, 60.0), (-60.0, 620.0)):  # 620 degF = 600 K
        n = 4000
        ts = np.linspace(t1, t2, n + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RangeWarning)
            f = np.array([cp0_molar(t, g) for t in ts])
            h = (t2 - t1) / n
            simpson = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
            assert rel(delta_h0(t1, t2, g), simpson) < 1e-6


def test_component_h_matches_quadrature_across_n2_split():
    # COMPUTED: piecewise Shomate integral for N2 across 500 K.
    ts = np.linspace(400.0, 600.0, 2001)
    f = np.array([component_cp_j(t, "N2") for t in ts])
    h = ts[1] - ts[0]
    simpson = h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum())
    assert rel(component_h_j(400.0, 600.0, "N2"), simpson) < 1e-7


def test_temperature_outside_range_warns_and_extrapolates():
    g = pseudo_composition(0.65)
    with pytest.warns(RangeWarning):
        v = cp0_molar(degf(150.0), g)
    assert math.isfinite(v) and v > 0.0
    with pytest.warns(RangeWarning):
        k0(degf(700.0), g)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        k0(degf(300.0), g)


# --- wiring -----------------------------------------------------------------------------

def test_pvt_k_ratio_and_sonic_velocity_use_k0():
    # COMPUTED: sonic_velocity with k=None equals sonic_velocity with k = pvt.k_ratio(t).
    pvt = DryGasPVT(0.65, co2=0.02, n2=0.01)
    t = 238.0
    k = pvt.k_ratio(t)
    assert rel(k, k0(t, pvt.composition)) < 1e-12
    assert 1.2 < k < 1.3
    assert sonic_velocity(pvt, 200.0, t) == sonic_velocity(pvt, 200.0, t, k=k)
    assert rel(sonic_velocity(pvt, 200.0, t, k=1.28), sonic_velocity(pvt, 200.0, t) * math.sqrt(1.28 / k)) < 1e-12


def test_k_override_reproduces_pre_change_rate():
    # COMPUTED: the dry_gas_vertical rate with the pre-M2.5-2 constant k = 1.28, recorded at
    # commit 96c9a61 before this change: 586799.5424418742 Mscf/d (pwf 4068.2932 psia, exit
    # 212.2462 psia). M2.5-3a replaced the forward-difference (dp/drho)_T with the analytic
    # form, moving it by -8.7e-7 relative to 586799.0326747787 (pwf 4068.2950, exit 212.2394).
    # The M2.5-3b closeout scan to the IPR maximum moved it by -3.0e-10 to 586799.032499237
    # (pwf 4068.295046609809, exit 212.2394269301541); the pwf-space nodal solve (root
    # converged to 1e-6 psi) moved it by +2.2e-8 to 586799.0457176144 (pwf 4068.2951397330726,
    # exit 212.2394317088271). k_override = 1.28 reproduces that value exactly and the
    # 96c9a61 value to 1e-5.
    case = load_case(EXAMPLES_DIR / "dry_gas_vertical.yaml")
    d = case_to_dict(case)
    d["exit"]["k_override"] = 1.28
    r = solve(case_from_dict(d))
    assert r.gas_rate == pytest.approx(586799.0457176144, rel=1e-12)
    assert r.gas_rate == pytest.approx(586799.5424418742, rel=1e-5)
    assert r.pwf == pytest.approx(4068.2951397330726, rel=1e-9)
    assert r.exit.p == pytest.approx(212.2394317088271, rel=1e-9)
    assert r.correlations["heat_capacity_ratio"] == "k_override"
    assert any("k_override = 1.28" in w for w in r.warnings)


def test_default_uses_ideal_gas_k0_and_reports_it():
    # COMPUTED: without the override the result documents k0 at the exit and the composition.
    # 'profile' mode (M2.5-3b): the adiabatic default uses real-gas cp/cv instead of k0.
    d = case_to_dict(load_case(EXAMPLES_DIR / "dry_gas_vertical.yaml"))
    d["temperature"] = {"model": "profile"}
    r = solve(case_from_dict(d))
    assert r.correlations["heat_capacity_ratio"] == "ideal_gas_cp0_pseudo_composition"
    comp = r.correlations["cp0_composition"]
    assert abs(sum(comp["mole_fractions"].values()) - 1.0) < 1e-12
    assert any(n.startswith("heat capacity ratio: ideal-gas") for n in r.notes)
    assert any("ideal-gas cp0/cv0" in w for w in r.warnings if w.startswith("exit is choked"))
    assert r.gas_rate != pytest.approx(586799.5424418742, rel=1e-6)
