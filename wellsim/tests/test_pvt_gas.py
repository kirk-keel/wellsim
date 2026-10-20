"""Gate G1 for the gas-phase correlations: each against published worked examples or
chart values, target within 2% (spec section 7).

Every expected value carries a provenance comment using the tags in docs/PROVENANCE.md:
FETCHED  read from the named web page or PDF during the M1 session (secondary source
         unless the original paper is named)
MEMORY   recalled from a publication, not verified against it
COMPUTED derived from the equation in the test or from this implementation
"chart" means a value read from a published chart, so the tolerance includes the
reading error of the source.
"""

from __future__ import annotations

import math
import warnings

import pytest

from wellsim.core.pvt import DryGasPVT, RangeWarning, collect_range_warnings, make_pvt
from wellsim.core.pvt.dry_gas import (
    gas_density,
    gas_fvf,
    lee_gonzalez_eakin,
    pseudocriticals,
    sutton_pseudocriticals,
    wichert_aziz,
    z_dak,
    z_hall_yarborough,
)

G1 = 0.02  # gate G1: within 2%


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


# --- Sutton -------------------------------------------------------------------

def test_sutton_gamma_0p7():
    # FETCHED: GasCompressibility-py documentation (aegis4048.github.io/GasCompressibility-py/
    # sutton.html), example output of that package for sg = 0.7: Tpc = 377.59 degR,
    # Ppc = 663.287 psia. Secondary source; not checked against Sutton (1985) itself.
    tpc, ppc = sutton_pseudocriticals(0.7)
    assert rel(tpc, 377.59) < 1e-3
    assert rel(ppc, 663.287) < 1e-3


# --- Wichert-Aziz -------------------------------------------------------------

def test_wichert_aziz_epsilon_docs():
    # FETCHED: GasCompressibility-py documentation (same page as above), example output for
    # H2S = 0.05, CO2 = 0.10: epsilon = 19.348 degR. Secondary source.
    _, _, eps = wichert_aziz(377.59, 663.287, co2=0.10, h2s=0.05)
    assert rel(eps, 19.348) < 1e-3


def test_wichert_aziz_ahmed_sour_gas_example():
    # FETCHED: KSU PGE 362 lecture 6 PDF (Khamis, 2015; faculty.ksu.edu.sa), text extracted
    # during the M1 session. Worked example "A sour gas (sp. gr. 0.7) contains 5% CO2 and
    # 10% H2S; calculate the density at 3500 psia and 160 degF". Printed values:
    # Tpc = 389.38 degR, Ppc = 669.1 psia (Standing's gravity equations, used here only as
    # the input to the correction), epsilon = 20.735 degR, Tpc' = 368.64 degR,
    # Ppc' = 630.44 psia, Ppr = 5.55, Tpr = 1.68, Z = 0.89 (Standing-Katz chart),
    # rho = 11.98 lb/ft3. The lecture reproduces an example from Ahmed, Reservoir
    # Engineering Handbook ch. 2 (MEMORY: example number not confirmed against the book).
    tpc_c, ppc_c, eps = wichert_aziz(389.38, 669.1, co2=0.05, h2s=0.10)
    assert rel(eps, 20.735) < 1e-3
    assert rel(tpc_c, 368.64) < 1e-3
    assert rel(ppc_c, 630.44) < 1e-3
    z = z_dak(3500.0 / ppc_c, (160.0 + 459.67) / tpc_c)
    assert rel(z, 0.89) < G1  # chart
    assert rel(gas_density(3500.0, 160.0, 0.7, z), 11.98) < G1  # chart-derived


def test_wichert_aziz_no_acid_gas_is_identity():
    assert wichert_aziz(380.0, 660.0) == (380.0, 660.0, 0.0)


# --- Z factor: Dranchuk-Abou-Kassem and Hall-Yarborough ------------------------

# FETCHED: zFactor R package vignette "Dranchuk-AbouKassem" (Reyes, A.R.,
# f0nzie.github.io/zFactor), getStandingKatzMatrix output: that package author's
# digitisation of the Standing and Katz (1942) chart. Not checked against the chart
# itself. Columns are Ppr = 0.5, 1.5, 2.5, 3.5, 4.5, 5.5.
_SK_PPR = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
_SK_CHART = {
    1.2: (0.893, 0.657, 0.519, 0.565, 0.650, 0.741),
    1.3: (0.916, 0.756, 0.638, 0.633, 0.684, 0.759),
    1.5: (0.948, 0.859, 0.794, 0.770, 0.790, 0.836),
    1.7: (0.968, 0.914, 0.876, 0.857, 0.864, 0.897),
    2.0: (0.982, 0.956, 0.941, 0.937, 0.945, 0.969),
    3.0: (1.002, 1.009, 1.018, 1.029, 1.041, 1.056),
}
_SK_POINTS = [(tpr, ppr, z) for tpr, row in _SK_CHART.items() for ppr, z in zip(_SK_PPR, row)]


@pytest.mark.parametrize("tpr,ppr,z_chart", _SK_POINTS)
def test_dak_vs_standing_katz_chart(tpr, ppr, z_chart):
    assert rel(z_dak(ppr, tpr), z_chart) < G1


@pytest.mark.parametrize("tpr,ppr,z_chart", _SK_POINTS)
def test_hall_yarborough_vs_standing_katz_chart(tpr, ppr, z_chart):
    assert rel(z_hall_yarborough(ppr, tpr), z_chart) < G1


@pytest.mark.parametrize(
    "tpr,ppr,z_ref",
    [
        # FETCHED: zFactor package vignette (same page as above), z.calc values produced by
        # that package's own DAK implementation: (Tpr, Ppr) = (2.0, 1.5) -> 0.9551087,
        # (1.1, 1.5) -> 0.4463987, and the Tpr = 1.05 isotherm: 0.8300683 (0.5),
        # 0.5866751 (1.0), 0.2837318 (1.5), 0.3284040 (2.0), 0.3868282 (2.5), 0.4466387 (3.0).
        # Cross-check between two implementations of the same equation, not a chart or
        # paper value. Agreement to 1e-7 is what pins the A1..A11 coefficients (MEMORY).
        (2.0, 1.5, 0.9551087),
        (1.1, 1.5, 0.4463987),
        (1.05, 0.5, 0.8300683),
        (1.05, 1.0, 0.5866751),
        (1.05, 1.5, 0.2837318),
        (1.05, 2.0, 0.3284040),
        (1.05, 2.5, 0.3868282),
        (1.05, 3.0, 0.4466387),
    ],
)
def test_dak_matches_independent_implementation(tpr, ppr, z_ref):
    assert rel(z_dak(ppr, tpr), z_ref) < 1e-5


def test_dak_ahmed_example_2_2():
    # FETCHED: TIU Reservoir Engineering I lecture "Natural Gases Part II" PDF (N. Ali,
    # 2022; lecture-notes.tiu.edu.iq), text extracted during the M1 session. Example "gas
    # reservoir at 3000 psia and 180 degF" (composition given, Kay's rule): Ppr = 4.50,
    # Tpr = 1.67, Z = 0.85 from the Standing-Katz chart. The lecture reproduces an Ahmed
    # ch. 2 example; the book was not consulted.
    assert rel(z_dak(4.50, 1.67), 0.85) < G1


def test_z_low_pressure_limit_is_unity():
    # COMPUTED: physical limit, Z -> 1 as p -> 0.
    assert abs(z_dak(0.2, 2.0) - 1.0) < 0.01
    assert abs(z_hall_yarborough(0.1, 2.0) - 1.0) < 0.01
    assert z_dak(0.0, 1.5) == 1.0


def test_dak_and_hall_yarborough_agree_in_common_range():
    # COMPUTED: two fits of the same chart should agree; 1.5% is my choice of tolerance.
    for tpr in (1.3, 1.6, 2.0, 2.5):
        for ppr in (1.0, 3.0, 6.0, 10.0, 15.0):
            assert rel(z_dak(ppr, tpr), z_hall_yarborough(ppr, tpr)) < 0.015


# --- Lee-Gonzalez-Eakin -------------------------------------------------------

def test_lge_gpsa_chart_point():
    # FETCHED: PetroSkills blog "Variation of Natural Gas Viscosity with Temperature,
    # Pressure, and Relative Density" (Dec 2022 tip of the month), which reports a reading
    # of GPSA Engineering Data Book Fig. 23-23: relative density 0.7, 100 degC (212 degF),
    # 5000 kPa (725 psia) -> 0.015 cP. Two-significant-figure chart reading by a third
    # party, so it cannot resolve better than about +/-3%. Sanity check only; LGE does not
    # yet have a G1-grade published value (see docs/PROVENANCE.md).
    tpc, ppc = sutton_pseudocriticals(0.7)
    z = z_dak(725.0 / ppc, (212.0 + 459.67) / tpc)
    mu = lee_gonzalez_eakin(212.0, 0.7, gas_density(725.0, 212.0, 0.7, z), 725.0)
    assert rel(mu, 0.015) < 0.06


def test_lge_reference_form():
    # COMPUTED: the expected value is the LGE equation evaluated in this test for M = 20.0
    # (gas_sg = 0.69048), T = 200 degF (659.67 degR), rho = 5.0 lbm/ft3. It checks that
    # the module codes the same equation, nothing more. The equation form
    # mu = 1e-4 K exp(X (rho/62.4)^Y), K = (9.4 + 0.02 M) T^1.5 / (209 + 19 M + T),
    # X = 3.5 + 986/T + 0.01 M, Y = 2.4 - 0.2 X (T degR, rho lbm/ft3) was FETCHED from
    # PSU PNG 520 module 19 (eqs. 19.25a-d) and Engineering LibreTexts 19.3 during the
    # M1 session; the original Lee, Gonzalez and Eakin (1966) paper was not consulted.
    # The 0.015-0.030 cp band is my own order-of-magnitude expectation.
    m = 20.0
    t = 659.67
    k = (9.4 + 0.02 * m) * t**1.5 / (209.0 + 19.0 * m + t)
    x = 3.5 + 986.0 / t + 0.01 * m
    y = 2.4 - 0.2 * x
    expected = 1e-4 * k * math.exp(x * (5.0 / 62.428) ** y)
    got = lee_gonzalez_eakin(200.0, m / 28.9647, 5.0, 2000.0)
    assert rel(got, expected) < 1e-9
    assert 0.015 < got < 0.030  # order of magnitude for a dense gas at 200 degF


def test_lge_increases_with_density_and_temperature():
    # COMPUTED: qualitative trends only.
    mu_lo = lee_gonzalez_eakin(200.0, 0.7, 2.0, 1000.0)
    mu_hi = lee_gonzalez_eakin(200.0, 0.7, 12.0, 5000.0)
    assert mu_hi > mu_lo
    assert lee_gonzalez_eakin(300.0, 0.7, 2.0, 1000.0) > lee_gonzalez_eakin(150.0, 0.7, 2.0, 1000.0)


# --- range warnings (spec section 6: never silent) --------------------------------
# The range limits coded in dry_gas.py: Sutton 0.57-1.68 and LGE 100-340 degF /
# 100-8000 psia were FETCHED (search snippets citing Sutton 1985; IHS Harmony and pengtools
# pages for LGE). Wichert-Aziz, DAK and Hall-Yarborough limits are MEMORY of the papers.
# The out-of-range inputs below are chosen by me to sit clearly outside those limits.

@pytest.mark.parametrize(
    "call,variable",
    [
        (lambda: sutton_pseudocriticals(0.5), "gas_sg"),
        (lambda: sutton_pseudocriticals(1.8), "gas_sg"),
        (lambda: wichert_aziz(380.0, 660.0, co2=0.6), "co2"),
        (lambda: wichert_aziz(380.0, 660.0, h2s=0.8), "h2s"),
        (lambda: z_dak(2.0, 3.5), "Tpr"),
        (lambda: z_dak(31.0, 2.0), "Ppr"),
        (lambda: z_dak(1.5, 0.95), "Ppr (Tpr <= 1)"),
        (lambda: z_hall_yarborough(2.0, 1.1), "Tpr"),
        (lambda: lee_gonzalez_eakin(50.0, 0.7, 2.0, 1000.0), "T"),
        (lambda: lee_gonzalez_eakin(200.0, 0.7, 2.0, 9000.0), "p"),
        (lambda: lee_gonzalez_eakin(200.0, 1.2, 2.0, 1000.0), "gas_sg"),
    ],
)
def test_out_of_range_warns(call, variable):
    with pytest.warns(RangeWarning) as rec:
        call()
    assert any(w.message.variable == variable for w in rec)


def test_in_range_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sutton_pseudocriticals(0.7)
        wichert_aziz(380.0, 660.0, co2=0.05, h2s=0.1)
        z_dak(3.0, 1.5)
        z_hall_yarborough(3.0, 1.5)
        lee_gonzalez_eakin(200.0, 0.7, 5.0, 2000.0)


def test_n2_above_5_percent_warns():
    with pytest.warns(UserWarning, match="N2"):
        pseudocriticals(0.7, n2=0.10)


def test_collect_range_warnings_dedups_and_reemits_others():
    with collect_range_warnings() as caught:
        for _ in range(3):
            sutton_pseudocriticals(0.5)
        z_dak(2.0, 3.5)
    assert [(w.correlation, w.variable) for w in caught] == [("Sutton", "gas_sg"), ("Dranchuk-Abou-Kassem", "Tpr")]
    with pytest.warns(UserWarning, match="N2"):
        with collect_range_warnings() as caught:
            pseudocriticals(0.7, n2=0.2)
    assert caught == []


# --- DryGasPVT model (all COMPUTED: internal consistency and wiring) ---------------

def test_dry_gas_state_is_self_consistent():
    pvt = DryGasPVT(0.65)
    st = pvt.state(3000.0, 200.0)
    assert st.z == pytest.approx(pvt.z(3000.0, 200.0))
    assert st.rho_g == pytest.approx(gas_density(3000.0, 200.0, 0.65, st.z))
    assert st.bg == pytest.approx(gas_fvf(3000.0, 200.0, st.z))
    # mass balance: standard density * (1/Bg) = in-situ density
    rho_sc = gas_density(14.696, 60.0, 0.65, 1.0)
    assert rho_sc / st.bg == pytest.approx(st.rho_g, rel=1e-9)
    assert st.mu_g == pytest.approx(pvt.gas_viscosity(3000.0, 200.0))
    assert st.rs is None and st.bo is None and st.bw is None and st.saturated is None


def test_dry_gas_z_method_alternate_and_bad_name():
    a = DryGasPVT(0.65).z(3000.0, 200.0)
    b = DryGasPVT(0.65, z_method="hall_yarborough").z(3000.0, 200.0)
    assert rel(a, b) < 0.015
    with pytest.raises(ValueError, match="gas_z"):
        DryGasPVT(0.65, z_method="nope")


def test_make_pvt_dry_gas_from_schema(gas_case_dict):
    from wellsim.cases.schema import Case

    # Gravity 0.70 (M2.5-2): the fixture's 0.65 with 15% acid gas implies a hydrocarbon
    # molar mass of 15.55, below methane, which the heat-capacity pseudo-composition now
    # rejects instead of clamping. 0.70 with the same inerts is consistent (M_HC 17.25).
    gas_case_dict["fluid"].update({"gas_sg": 0.70, "co2": 0.05, "h2s": 0.10})
    gas_case_dict["options"] = {"correlations": {"gas_z": "hall_yarborough"}}
    case = Case.model_validate(gas_case_dict)
    pvt = make_pvt(case.fluid, case.options.correlations)
    assert isinstance(pvt, DryGasPVT)
    assert pvt.z_method == "hall_yarborough"
    tpc, ppc = sutton_pseudocriticals(0.70)
    tpc_c, ppc_c, _ = wichert_aziz(tpc, ppc, 0.05, 0.10)
    assert pvt.tpc == pytest.approx(tpc_c) and pvt.ppc == pytest.approx(ppc_c)
    assert 0.7 < pvt.state(5000.0, 200.0).z < 1.1


def test_make_pvt_rejects_gravity_inconsistent_with_inerts(gas_case_dict):
    # COMPUTED (M2.5-2): 0.65 gravity with 5% CO2 and 10% H2S leaves the hydrocarbons
    # lighter than methane; the solver raises rather than clamps.
    from wellsim.cases.schema import Case

    gas_case_dict["fluid"].update({"co2": 0.05, "h2s": 0.10})
    case = Case.model_validate(gas_case_dict)
    with pytest.raises(ValueError, match="below methane"):
        make_pvt(case.fluid, case.options.correlations)
