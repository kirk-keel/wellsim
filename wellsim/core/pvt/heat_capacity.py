"""Ideal-gas heat capacity of natural gas from a gravity-based pseudo-composition
(TODO-M3 item "cp/cv", M2.5-2).

Pure-component ideal-gas molar cp, 200 to 600 K, all FETCHED from the NIST Chemistry
WebBook gas-phase thermochemistry pages on 21 Sep 2026 (see docs/PROVENANCE.md):

    CH4   table, Gurvich, Veyts et al. 1989   webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Mask=1
    C2H6  table, Gurvich, Veyts et al. 1989   webbook.nist.gov/cgi/cbook.cgi?ID=C74840&Mask=1
    C3H8  table, Chao 1973                    webbook.nist.gov/cgi/cbook.cgi?ID=C74986&Mask=1
    N2    Shomate 100-500 K and 500-2000 K    webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=1
    CO2   table: NIST fluid data (Span-Wagner 1996) at 0.01 MPa, 216.59 to 291.59 K,
          webbook.nist.gov/cgi/fluid.cgi?ID=C124389&Type=IsoBar&P=0.01 ..., plus the
          JANAF (Chase 1998) rows 300 to 700 K from cbook.cgi?ID=C124389&Mask=1&Type=JANAFG
    H2S   table: NIST fluid data (Lemmon-Span 2006) at 0.01 MPa, 200 to 300 K,
          fluid.cgi?ID=C7783064&Type=IsoBar&P=0.01 ..., plus the JANAF rows 298 to 700 K
          from cbook.cgi?ID=C7783064&Mask=1&Type=JANAFG

Every component except N2 uses a quartic in T (K) fitted at import to the fetched table
rows from 200 to 700 K (COMPUTED fit; max error against the table between 200 and 600 K
is tested below 0.3%). NIST gives Shomate fits for CH4, CO2 and H2S only from 298 K and
they extrapolate poorly below it (methane 8% high at 200 K, CO2 2.2% low at 216.6 K,
H2S 1.6% high at 200 K against the fluid data), so the table fits are used. The fluid
data at 0.01 MPa are real-gas values; the pressure departure at 0.01 MPa is far below
the 0.3% fit tolerance. CO2 fluid data start at the triple point (216.59 K); 200 to
216.59 K is fit extrapolation. Molecular weights are the values printed on the NIST
compound pages.

Pseudo-composition from gravity (modelling decision, COMPUTED):
    M_mix = M_air gamma; inerts removed by mole fraction give M_HC; the hydrocarbon
    fraction is split CH4 + C2H6 to match M_HC, or C2H6 + C3H8 when M_HC exceeds
    ethane. M_HC below methane or above propane raises; never clamped. A RangeWarning
    is issued when the hydrocarbon gravity exceeds 0.90 (dry-gas assumption doubtful).

API in oilfield units: T in degF, cp in Btu/(lbmol degR) or Btu/(lbm degR),
1 Btu/(lbmol degR) = 4.1868 J/(mol K). Temperatures outside 200 to 600 K warn
(RangeWarning) and extrapolate, per the project convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from wellsim.core.pvt.base import M_AIR, check_range

__all__ = [
    "R_J_PER_MOL_K",
    "BTU_LBMOL_R_PER_J_MOL_K",
    "R_BTU_LBMOL_R",
    "MOLAR_MASS",
    "CP_TABLES",
    "SHOMATE",
    "T_VALID_K",
    "component_cp_j",
    "component_h_j",
    "GasComposition",
    "pseudo_composition",
    "cp0_molar",
    "cp0_mass",
    "k0",
    "delta_h0",
]

R_J_PER_MOL_K = 8.314462618  # CODATA 2018 exact (MEMORY of the SI redefinition value)
BTU_LBMOL_R_PER_J_MOL_K = 1.0 / 4.1868  # 1 Btu/(lbmol degR) = 4.1868 J/(mol K) (owner-stated)
R_BTU_LBMOL_R = R_J_PER_MOL_K * BTU_LBMOL_R_PER_J_MOL_K  # 1.98588
J_PER_MOL_PER_BTU_PER_LBMOL = 4.1868 / 1.8  # 2.326: enthalpy conversion, from the cp conversion and 1 K = 1.8 degR
T_VALID_K = (200.0, 600.0)

# FETCHED: molecular weights as printed on the NIST WebBook pages above.
MOLAR_MASS = {
    "CH4": 16.0425,
    "C2H6": 30.0690,
    "C3H8": 44.0956,
    "N2": 28.0134,
    "CO2": 44.0095,
    "H2S": 34.081,
}

# FETCHED: cp,gas in J/(mol K) vs T in K, rows as printed.
CP_TABLES = {
    "CH4": ((100.0, 33.28), (200.0, 33.51), (298.15, 35.69), (300.0, 35.76), (400.0, 40.63),
            (500.0, 46.63), (600.0, 52.74), (700.0, 58.60)),
    "C2H6": ((100.0, 35.70), (200.0, 42.30), (298.15, 52.49), (300.0, 52.71), (400.0, 65.46),
             (500.0, 77.94), (600.0, 89.19), (700.0, 99.14)),
    "C3H8": ((150.0, 48.79), (200.0, 56.07), (273.15, 68.74), (298.15, 73.60), (300.0, 73.93),
             (400.0, 94.01), (500.0, 112.59), (600.0, 128.70), (700.0, 142.67)),
    # fluid.cgi isobar rows (Span-Wagner 1996, vapor, 0.01 MPa) then JANAF (Chase 1998) rows
    "CO2": ((216.59, 33.282), (241.59, 34.488), (266.59, 35.696), (291.59, 36.870),
            (300.0, 37.22), (400.0, 41.34), (500.0, 44.61), (600.0, 47.32), (700.0, 49.57)),
    # fluid.cgi isobar rows (Lemmon-Span 2006, vapor, 0.01 MPa) then JANAF (Chase 1998) rows
    "H2S": ((200.0, 33.591), (225.0, 33.617), (250.0, 33.734), (275.0, 33.923), (300.0, 34.172),
            (298.0, 34.20), (300.0, 34.22), (400.0, 35.54), (500.0, 37.20), (600.0, 38.97), (700.0, 40.76)),
}

# FETCHED: Shomate coefficients (A, B, C, D, E) with their stated ranges; F, G, H not needed
# for cp or for cp integrals between two temperatures.
#   cp = A + B t + C t^2 + D t^3 + E / t^2,  t = T / 1000
# Only N2 keeps a Shomate form (its 100 to 500 K set covers the range). The fetched CO2
# (298 to 1200 K: 24.99735, 55.18696, -33.69137, 7.948387, -0.136638) and H2S (298 to
# 1400 K: 26.88412, 18.67809, 3.434203, -3.378702, 0.135882) sets are recorded in
# PROVENANCE.md and not used, because they miss the fluid data below 298 K by more than 1%.
SHOMATE = {
    "N2": (((100.0, 500.0), (28.98641, 1.853978, -9.647459, 16.63537, 0.000117)),
           ((500.0, 2000.0), (19.50583, 19.88705, -8.598535, 1.369784, 0.527601))),
}

_FIT_DEGREE = 4
_FIT_RANGE_K = (200.0, 700.0)


def _fit(table: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Quartic in T (K) through the table rows in 200 to 700 K; the 700 K row is
    down-weighted so the fit is anchored on the validity range. COMPUTED."""
    rows = [(t, c) for t, c in table if _FIT_RANGE_K[0] <= t <= _FIT_RANGE_K[1]]
    t = np.array([r[0] for r in rows])
    c = np.array([r[1] for r in rows])
    w = np.where(t <= T_VALID_K[1], 1.0, 0.3)
    return np.polyfit(t, c, _FIT_DEGREE, w=w)


_POLY = {name: _fit(table) for name, table in CP_TABLES.items()}
_POLY_LIST = {name: [float(c) for c in p] for name, p in _POLY.items()}  # Horner without numpy
_POLY_INT = {name: np.polyint(p) for name, p in _POLY.items()}  # antiderivative in T


def _shomate_range(name: str, t_k: float):
    ranges = SHOMATE[name]
    for (lo, hi), coef in ranges:
        if t_k <= hi:
            return coef
    return ranges[-1][1]


def component_cp_j(t_k: float, name: str) -> float:
    """Ideal-gas cp of one pure component, J/(mol K), T in K. No range warning here
    (the mixture functions warn once)."""
    if name in _POLY:
        acc = 0.0
        for c in _POLY_LIST[name]:
            acc = acc * t_k + c
        return acc
    a, b, c, d, e = _shomate_range(name, t_k)
    t = t_k / 1000.0
    return a + b * t + c * t * t + d * t**3 + e / (t * t)


def component_h_j(t1_k: float, t2_k: float, name: str) -> float:
    """Analytic integral of component_cp_j from t1 to t2, J/mol."""
    if name in _POLY:
        p = _POLY_INT[name]
        return float(np.polyval(p, t2_k) - np.polyval(p, t1_k))
    # Shomate: integral of cp dT = 1000 [A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t], t = T/1000.
    # For N2 the two ranges meet at 500 K; integrate piecewise so the split is exact.
    ranges = SHOMATE[name]
    if len(ranges) > 1:
        bounds = sorted({t1_k, t2_k, *[hi for (lo, hi), _ in ranges[:-1]]})
        lo, hi = min(t1_k, t2_k), max(t1_k, t2_k)
        pieces = [b for b in bounds if lo <= b <= hi]
        total = 0.0
        for a_, b_ in zip(pieces, pieces[1:]):
            total += _shomate_integral(_shomate_range(name, 0.5 * (a_ + b_)), a_, b_)
        return total if t2_k >= t1_k else -total
    return _shomate_integral(ranges[0][1], t1_k, t2_k)


def _shomate_h_from_200(name: str, t_k: float) -> float:
    """Integral of the Shomate cp from 200 K to t_k, J/mol, range by range."""
    total = 0.0
    t_lo = 200.0
    ranges = SHOMATE[name]
    for (lo, hi), coef in ranges:
        if t_k <= hi:
            return total + _shomate_integral(coef, t_lo, t_k)
        total += _shomate_integral(coef, t_lo, hi)
        t_lo = hi
    return total + _shomate_integral(ranges[-1][1], t_lo, t_k)


def _shomate_s_from_200(name: str, t_k: float) -> float:
    """Integral of the Shomate cp / T from 200 K to t_k, J/(mol K), range by range."""
    total = 0.0
    t_lo = 200.0
    ranges = SHOMATE[name]
    for (lo, hi), coef in ranges:
        if t_k <= hi:
            return total + _shomate_s_integral(coef, t_lo, t_k)
        total += _shomate_s_integral(coef, t_lo, hi)
        t_lo = hi
    return total + _shomate_s_integral(ranges[-1][1], t_lo, t_k)


def _shomate_s_integral(coef, t1_k: float, t2_k: float) -> float:
    # integral of cp / T dT with t = T/1000: A ln t + B t + C t^2/2 + D t^3/3 - E/(2 t^2)
    a, b, c, d, e = coef

    def f(t_k: float) -> float:
        t = t_k / 1000.0
        return a * math.log(t) + b * t + c * t * t / 2.0 + d * t**3 / 3.0 - e / (2.0 * t * t)

    return f(t2_k) - f(t1_k)


def _shomate_integral(coef, t1_k: float, t2_k: float) -> float:
    a, b, c, d, e = coef

    def f(t_k: float) -> float:
        t = t_k / 1000.0
        return 1000.0 * (a * t + b * t * t / 2.0 + c * t**3 / 3.0 + d * t**4 / 4.0 - e / t)

    return f(t2_k) - f(t1_k)


# --- pseudo-composition -----------------------------------------------------------

@dataclass(frozen=True)
class GasComposition:
    """Mole fractions of the six components and the mixture molar mass, lbm/lbmol.

    ``table_poly`` is the mole-fraction-weighted sum of the table-fitted quartics (highest
    power first) and ``shomate_parts`` the (fraction, name) pairs with y > 0 for the
    Shomate components, both precomputed so the per-call evaluation in the march is plain
    Python arithmetic."""

    fractions: dict[str, float]
    molar_mass: float
    hc_gravity: float
    hc_split: tuple[str, str]
    table_poly: tuple[float, ...] = field(default=(), repr=False, compare=False)
    shomate_parts: tuple[tuple[float, str], ...] = field(default=(), repr=False, compare=False)
    table_poly_int: tuple[float, ...] = field(default=(), repr=False, compare=False)  # antiderivative in T

    def cp_j(self, t_k: float) -> float:
        """Mixture ideal-gas cp, J/(mol K), T in K; no range check."""
        acc = 0.0
        for c in self.table_poly:
            acc = acc * t_k + c
        for y, name in self.shomate_parts:
            acc += y * component_cp_j(t_k, name)
        return acc

    def h_ideal_j(self, t_k: float) -> float:
        """Integral of cp_j from 200 K to ``t_k``, J/mol; plain-Python Horner on the mixture
        antiderivative plus the Shomate parts (piecewise, continuous at the range joins)."""
        acc = 0.0
        for c in self.table_poly_int:
            acc = acc * t_k + c
        for y, name in self.shomate_parts:
            acc += y * _shomate_h_from_200(name, t_k)
        return acc

    def h_j(self, t1_k: float, t2_k: float) -> float:
        """Integral of cp_j from t1 to t2, J/mol."""
        return self.h_ideal_j(t2_k) - self.h_ideal_j(t1_k)

    def s_ideal_j(self, t_k: float) -> float:
        """Integral of cp_j / T from 200 K to ``t_k``, J/(mol K): the ideal-gas entropy's
        temperature part. Quartic c4 T^4 + ... + c0 gives c4 T^4/4 + c3 T^3/3 + c2 T^2/2 +
        c1 T + c0 ln T; the Shomate parts are integrated range by range."""
        c4, c3, c2, c1, c0 = self.table_poly
        f = lambda t: c4 * t**4 / 4.0 + c3 * t**3 / 3.0 + c2 * t * t / 2.0 + c1 * t + c0 * math.log(t)  # noqa: E731
        acc = f(t_k) - f(200.0)
        for y, name in self.shomate_parts:
            acc += y * _shomate_s_from_200(name, t_k)
        return acc

    def as_dict(self) -> dict:
        return {
            "mole_fractions": dict(self.fractions),
            "molar_mass": self.molar_mass,
            "hc_gravity": self.hc_gravity,
            "hc_split": list(self.hc_split),
        }


def pseudo_composition(gas_sg: float, n2: float = 0.0, co2: float = 0.0, h2s: float = 0.0) -> GasComposition:
    """Split a gas of gravity ``gas_sg`` (air = 1) with the given inert mole fractions into
    CH4 + C2H6 (or C2H6 + C3H8) that reproduces the molar mass. Raises ValueError when the
    hydrocarbon molar mass falls below methane or above propane; warns (RangeWarning) when
    the hydrocarbon gravity exceeds 0.90."""
    y_inert = n2 + co2 + h2s
    if y_inert >= 1.0 or min(n2, co2, h2s) < 0.0:
        raise ValueError("inert mole fractions must be >= 0 and sum to less than 1")
    m_mix = M_AIR * gas_sg
    m_inert = n2 * MOLAR_MASS["N2"] + co2 * MOLAR_MASS["CO2"] + h2s * MOLAR_MASS["H2S"]
    y_hc = 1.0 - y_inert
    m_hc = (m_mix - m_inert) / y_hc
    m1, m2, m3 = MOLAR_MASS["CH4"], MOLAR_MASS["C2H6"], MOLAR_MASS["C3H8"]
    if m_hc < m1 * (1.0 - 1e-9):
        sg_min = (m1 * y_hc + m_inert) / M_AIR  # all-methane hydrocarbons with the same inerts
        raise ValueError(
            f"hydrocarbon molar mass {m_hc:.3f} from gravity {gas_sg:.4f} with inerts "
            f"n2 {n2:.4f}, co2 {co2:.4f}, h2s {h2s:.4f} is below methane ({m1}); the minimum feasible "
            f"mixture gravity for these inert fractions is {sg_min:.4f} (pure methane hydrocarbons). "
            "The gravity or the inert fractions are inconsistent; nothing is clamped"
        )
    if m_hc > m3 * (1.0 + 1e-9):
        raise ValueError(
            f"hydrocarbon molar mass {m_hc:.3f} from gravity {gas_sg:.4f} is above propane ({m3}); "
            "not a dry gas the pseudo-composition can represent"
        )
    hc_gravity = m_hc / M_AIR
    check_range("ideal-gas cp pseudo-composition", "hydrocarbon gravity", hc_gravity, 0.0, 0.90, "air = 1")
    if m_hc <= m2:
        x = (m_hc - m1) / (m2 - m1)
        split = ("CH4", "C2H6")
        hc = {"CH4": (1.0 - x) * y_hc, "C2H6": x * y_hc, "C3H8": 0.0}
    else:
        x = (m_hc - m2) / (m3 - m2)
        split = ("C2H6", "C3H8")
        hc = {"CH4": 0.0, "C2H6": (1.0 - x) * y_hc, "C3H8": x * y_hc}
    fractions = {**hc, "N2": n2, "CO2": co2, "H2S": h2s}
    fractions = {k: max(v, 0.0) for k, v in fractions.items()}
    table_poly = tuple(float(v) for v in sum(fractions[n] * _POLY[n] for n in CP_TABLES))
    table_poly_int = tuple(float(v) for v in np.polyint(np.array(table_poly)))
    shomate_parts = tuple((fractions[n], n) for n in SHOMATE if fractions[n] > 0.0)
    return GasComposition(fractions, m_mix, hc_gravity, split, table_poly, shomate_parts, table_poly_int)


# --- mixture API, oilfield units ------------------------------------------------------

T_REF_DEGF = 60.0  # reference state for h(T, p): ideal gas at 60 degF, h = 0 (real_gas_thermo.T_REF)


def kelvin(t_degf: float) -> float:
    return (t_degf + 459.67) / 1.8


_kelvin = kelvin


def check_t(t_k: float) -> None:
    check_range("ideal-gas cp0", "T", t_k, T_VALID_K[0], T_VALID_K[1], "K")


_check_t = check_t


def cp0_molar(t_degf: float, gas: GasComposition) -> float:
    """Ideal-gas mixture cp, Btu/(lbmol degR)."""
    t_k = _kelvin(t_degf)
    _check_t(t_k)
    return gas.cp_j(t_k) * BTU_LBMOL_R_PER_J_MOL_K


def cp0_mass(t_degf: float, gas: GasComposition) -> float:
    """Ideal-gas mixture cp, Btu/(lbm degR)."""
    return cp0_molar(t_degf, gas) / gas.molar_mass


def k0(t_degf: float, gas: GasComposition) -> float:
    """Ideal-gas heat capacity ratio cp0 / (cp0 - R)."""
    cp = cp0_molar(t_degf, gas)
    return cp / (cp - R_BTU_LBMOL_R)


def delta_h0(t1_degf: float, t2_degf: float, gas: GasComposition) -> float:
    """Ideal-gas enthalpy change from t1 to t2, Btu/lbmol, analytic integral of cp0."""
    t1, t2 = _kelvin(t1_degf), _kelvin(t2_degf)
    _check_t(min(t1, t2))
    _check_t(max(t1, t2))
    # cp in J/(mol K) integrated over T in K gives J/mol. 1 Btu/(lbmol degR) = 4.1868 J/(mol K)
    # and 1 K = 1.8 degR, so 1 Btu/lbmol = 4.1868 / 1.8 = 2.326 J/mol.
    return gas.h_j(t1, t2) / J_PER_MOL_PER_BTU_PER_LBMOL
