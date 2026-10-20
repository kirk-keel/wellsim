"""Gate G1 for McCain formation-water properties and Baker-Swerdloff surface tension.

Each expected value carries a provenance comment tagged FETCHED / MEMORY / COMPUTED as
defined in docs/PROVENANCE.md.
"""

from __future__ import annotations

import warnings

import pytest

from wellsim.core.pvt import RangeWarning
from wellsim.core.pvt.black_oil import (
    baker_swerdloff_oil,
    baker_swerdloff_water,
    mccain_bw,
    mccain_cw,
    mccain_rsw,
    mccain_water_viscosity,
    water_density_sc,
)

G1 = 0.02


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


# --- McCain water ------------------------------------------------------------------

def test_mccain_bw_pengtools_example():
    # FETCHED: pengtools wiki, "Water formation volume factor" (McCain correlation), worked
    # example: 3176 psia, 165 degF -> dV_wP = -0.0047241, dV_wT = 0.027, Bw = 1.022 rb/stb.
    # The coefficients coded in black_oil.py were also read from that page. McCain (1990)
    # not consulted.
    assert rel(mccain_bw(3176.0, 165.0), 1.022) < G1
    assert rel(mccain_bw(3176.0, 165.0), (1 + 0.027) * (1 - 0.0047241)) < 1e-3


def test_mccain_cw_pengtools_example():
    # FETCHED: pengtools wiki, "Water compressibility" (McCain), worked example:
    # salinity 74,289.3 mg/L, 3500 psia, 165 degF -> cw = 2.6348e-6 1/psia. Coefficients
    # and the 200-270 degF / 1000-20000 psia / 200,000 mg/L range also from that page.
    # 165 degF is below the 200 degF lower limit, so the correlation must warn.
    with pytest.warns(RangeWarning) as rec:
        cw = mccain_cw(3500.0, 165.0, 74289.3)
    assert any(w.message.variable == "T" for w in rec)
    assert rel(cw, 2.6348e-6) < G1


def test_mccain_water_viscosity_pengtools_example():
    # FETCHED: pengtools wiki, "Water viscosity" (McCain 1991), worked example:
    # 74,289.3 mg/L, water SG 1.0476 -> C_w = 7.1 wt%; 3176 psia, 165 degF ->
    # A = 68.83262779, B = 0.984227021, mu_w1 = 0.45 cP, mu_w = 0.52 cP. Coefficients also
    # from that page (their B is written with the opposite sign convention; same result).
    # 7.1 wt% = 71,000 ppm by weight is my conversion.
    assert rel(mccain_water_viscosity(3176.0, 165.0, 71000.0), 0.52) < G1
    assert rel(mccain_water_viscosity(14.696, 165.0, 71000.0), 0.45) < G1


def test_water_density_sc():
    # 62.368: MEMORY, and tautological. It is the constant term of the density equation as
    # coded, so asserting it only checks that salinity 0 returns the constant. The equation
    # (62.368 + 0.438603 S + 1.60074e-3 S^2) is MEMORY of McCain (1990); not verified.
    # 1.0476: FETCHED from the pengtools water-viscosity example, which states that SG for
    # the 7.1 wt% brine without saying how it was obtained. 0.35% apart from the equation.
    assert water_density_sc(0.0) == pytest.approx(62.368)
    assert rel(water_density_sc(71000.0), 1.0476 * 62.368) < 0.01


def test_mccain_rsw_pure_water_vs_culberson_mcketta():
    # MEMORY, and the weakest value in the suite: my recollection that the Culberson and
    # McKetta (1951) methane-in-water chart reads about 15 scf/stb at 200 degF and 3000
    # psia. Neither the chart nor McCain (1990) was consulted. The coded polynomial
    # coefficients are also MEMORY. What was FETCHED (search snippets) is only the claim
    # that McCain's fit is within 5% of that data for 1000-10000 psia and 100-340 degF,
    # and the salinity-factor form 10^(-0.0840655 S T^-0.285854). A 5% check, not a 2% gate.
    rsw = mccain_rsw(3000.0, 200.0, 0.0)
    assert rel(rsw, 15.0) < 0.05
    # COMPUTED: salinity lowers solubility; the 0.4-0.8 band is my choice
    assert mccain_rsw(3000.0, 200.0, 100000.0) < 0.8 * rsw
    assert mccain_rsw(3000.0, 200.0, 100000.0) > 0.4 * rsw


@pytest.mark.parametrize(
    "call,variable",
    [
        (lambda: mccain_bw(6000.0, 200.0), "p"),
        (lambda: mccain_bw(3000.0, 300.0), "T"),
        (lambda: mccain_cw(500.0, 220.0, 0.0), "p"),
        (lambda: mccain_cw(3000.0, 220.0, 250000.0), "salinity"),
        (lambda: mccain_water_viscosity(3000.0, 450.0, 0.0), "T"),
        (lambda: mccain_water_viscosity(3000.0, 200.0, 280000.0), "salinity"),
        (lambda: mccain_rsw(500.0, 200.0, 0.0), "p"),
        (lambda: mccain_rsw(3000.0, 50.0, 0.0), "T"),
    ],
)
def test_mccain_out_of_range_warns(call, variable):
    # Bw, cw and Rsw limits FETCHED (pengtools pages, search snippets); viscosity limits
    # (100-400 degF, 26 wt%, 14,000 psia) are MEMORY. Inputs here are mine.
    with pytest.warns(RangeWarning) as rec:
        call()
    assert any(w.message.variable == variable for w in rec)


def test_mccain_in_range_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mccain_bw(3000.0, 200.0)
        mccain_cw(3000.0, 220.0, 50000.0)
        mccain_water_viscosity(3000.0, 150.0, 50000.0)
        mccain_rsw(3000.0, 200.0, 50000.0)
        water_density_sc(50000.0)


# --- Baker-Swerdloff ----------------------------------------------------------------

def test_baker_swerdloff_water_vs_crc_pure_water():
    # MEMORY: 71.99 mN/m for water against air at 25 degC (77 degF), as I recall it from
    # the CRC Handbook of Chemistry and Physics. Widely quoted (IAPWS gives 71.97), not
    # verified this session. Baker-Swerdloff gas-water at 74 degF and 1 atm is
    # 75 - 1.108 * 14.7^0.349 = 72.2 dyne/cm. A physical check of the low-temperature end
    # of the correlation, independent of any PVT source.
    assert rel(baker_swerdloff_water(14.696, 74.0), 71.99) < G1


def test_baker_swerdloff_water_equation_form_and_trends():
    # COMPUTED: expected values are the equation evaluated in this test. The equation form
    # sigma_74 = 75 - 1.108 p^0.349, sigma_280 = 53 - 0.1048 p^0.637, linear between, was
    # FETCHED from the pengtools "Gas/Water Interfacial Tension" page. My attribution of
    # that form to Beggs (2003) and Brill and Mukherjee (1999) is MEMORY; neither book was
    # consulted. No published numeric example located.
    p = 1000.0
    s74 = 75.0 - 1.108 * p**0.349
    s280 = 53.0 - 0.1048 * p**0.637
    assert baker_swerdloff_water(p, 74.0) == pytest.approx(s74)
    assert baker_swerdloff_water(p, 280.0) == pytest.approx(s280)
    assert baker_swerdloff_water(p, 177.0) == pytest.approx(0.5 * (s74 + s280))
    assert baker_swerdloff_water(3000.0, 150.0) < baker_swerdloff_water(500.0, 150.0)


def test_baker_swerdloff_oil_equation_form_and_trends():
    # COMPUTED: expected values are the equation evaluated in this test. The equation form
    # sigma_68 = 39 - 0.2571 API, sigma_100 = 37.5 - 0.2571 API, live-oil factor
    # (1 - 0.024 p^0.45), floor 1 dyne/cm, was FETCHED from the pengtools "Gas/Oil
    # Interfacial Tension" page; attributing it to Beggs (2003) is MEMORY. The original
    # Baker and Swerdloff (1956) is a chart and was not consulted. No published numeric
    # example located. The 25-35 dyne/cm band for dead crude at room temperature is a
    # general recollection (MEMORY), not a specific table.
    dead68 = 39.0 - 0.2571 * 30.0
    dead100 = 37.5 - 0.2571 * 30.0
    factor = lambda p: 1.0 - 0.024 * p**0.45  # noqa: E731
    assert baker_swerdloff_oil(14.696, 68.0, 30.0) == pytest.approx(dead68 * factor(14.696))
    assert baker_swerdloff_oil(14.696, 100.0, 30.0) == pytest.approx(dead100 * factor(14.696))
    assert baker_swerdloff_oil(14.696, 84.0, 30.0) == pytest.approx(0.5 * (dead68 + dead100) * factor(14.696))
    assert 25.0 < dead68 < 35.0
    assert baker_swerdloff_oil(2000.0, 100.0, 30.0) < baker_swerdloff_oil(500.0, 100.0, 30.0)
    # beyond the chart's pressure range: warns, and the floor keeps it positive
    with pytest.warns(RangeWarning) as rec:
        assert baker_swerdloff_oil(6000.0, 100.0, 30.0) == 1.0
    assert any(w.message.variable == "p" for w in rec)
    # outside the chart's temperature band the dead-oil value is held (published procedure,
    # so no warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert baker_swerdloff_oil(14.696, 200.0, 30.0) == pytest.approx(dead100 * factor(14.696))
        assert baker_swerdloff_oil(14.696, 40.0, 30.0) == pytest.approx(dead68 * factor(14.696))
        assert baker_swerdloff_water(14.696, 300.0) == pytest.approx(53.0 - 0.1048 * 14.696**0.637)
