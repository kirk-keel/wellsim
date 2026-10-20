"""Gas-phase correlations and the dry-gas PVT model (spec 4.1).

Defaults: Sutton pseudo-criticals with the Wichert-Aziz sour correction,
Dranchuk-Abou-Kassem Z, Lee-Gonzalez-Eakin viscosity. Hall-Yarborough Z is the
schema alternate. Black oil reuses these functions for its free gas.

Sources
-------
Sutton, R.P. (1985) "Compressibility Factors for High-Molecular-Weight Reservoir
    Gases", SPE 14265. gamma_g 0.57-1.68.
Wichert, E. and Aziz, K. (1972) "Calculate Z's for Sour Gases", Hydrocarbon
    Processing 51(5) 119-122. CO2 0-54.4 mol%, H2S 0-73.8 mol%, p 154-7026 psia,
    T 40-300 degF.
Dranchuk, P.M. and Abou-Kassem, J.H. (1975) "Calculation of Z Factors for Natural
    Gases Using Equations of State", JCPT 14(3) 34-36. 0.2 <= Ppr < 30 for
    1.0 < Tpr <= 3.0, and Ppr < 1.0 for 0.7 < Tpr <= 1.0.
Hall, K.R. and Yarborough, L. (1973) "A New Equation of State for Z-Factor
    Calculations", Oil & Gas J. 71(25) 82-92. 1.2 <= Tpr <= 3.0, 0.1 <= Ppr <= 24.
Lee, A.L., Gonzalez, M.H. and Eakin, B.E. (1966) "The Viscosity of Natural Gases",
    JPT 18(8) 997-1000, SPE 1340. 100-340 degF, 100-8000 psia; less accurate for
    gamma_g > 1.0 and for sour gas.
"""

from __future__ import annotations

import math
import warnings
from typing import Callable

import wellsim.core.pvt.heat_capacity as heat_capacity
from wellsim.core.pvt.base import (
    M_AIR,
    P_SC,
    R_PSIA_FT3,
    T_SC_R,
    PVTModel,
    PVTState,
    check_range,
    rankine,
)

__all__ = [
    "sutton_pseudocriticals",
    "wichert_aziz",
    "pseudocriticals",
    "z_dak",
    "z_dak_full",
    "dak_partials",
    "dak_reduced_density",
    "z_hall_yarborough",
    "Z_METHODS",
    "gas_density",
    "gas_fvf",
    "lee_gonzalez_eakin",
    "DryGasPVT",
]


# --- pseudo-critical properties -------------------------------------------------

def sutton_pseudocriticals(gas_sg: float) -> tuple[float, float]:
    """Sutton (1985): (Tpc degR, Ppc psia) from gas specific gravity (air = 1)."""
    check_range("Sutton", "gas_sg", gas_sg, 0.57, 1.68)
    tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg**2
    ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg**2
    return tpc, ppc


def wichert_aziz(tpc: float, ppc: float, co2: float = 0.0, h2s: float = 0.0) -> tuple[float, float, float]:
    """Wichert-Aziz (1972) sour-gas correction. Returns (Tpc', Ppc', epsilon degR).

    ``co2`` and ``h2s`` are mole fractions. epsilon = 120 (A^0.9 - A^1.6) + 15 (B^0.5 - B^4),
    A = y_CO2 + y_H2S, B = y_H2S.
    """
    if co2 == 0.0 and h2s == 0.0:
        return tpc, ppc, 0.0
    check_range("Wichert-Aziz", "co2", co2, 0.0, 0.544, "mol frac")
    check_range("Wichert-Aziz", "h2s", h2s, 0.0, 0.738, "mol frac")
    a = co2 + h2s
    b = h2s
    eps = 120.0 * (a**0.9 - a**1.6) + 15.0 * (b**0.5 - b**4)
    tpc_c = tpc - eps
    ppc_c = ppc * tpc_c / (tpc + b * (1.0 - b) * eps)
    return tpc_c, ppc_c, eps


def pseudocriticals(gas_sg: float, n2: float = 0.0, co2: float = 0.0, h2s: float = 0.0) -> tuple[float, float]:
    """Default pseudo-criticals: Sutton corrected by Wichert-Aziz. N2 is not corrected."""
    if n2 > 0.05:
        warnings.warn(
            f"pseudocriticals: n2 = {n2:.3g} mol frac; Sutton + Wichert-Aziz do not correct "
            "for N2 and the Z error grows above ~5 mol%",
            UserWarning,
            stacklevel=2,
        )
    tpc, ppc = sutton_pseudocriticals(gas_sg)
    tpc, ppc, _ = wichert_aziz(tpc, ppc, co2, h2s)
    return tpc, ppc


# --- Z factor -------------------------------------------------------------------

def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-12) -> float:
    """Root of f on [lo, hi] by bisection. Fallback when Newton fails; no scipy in the hot path."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        raise RuntimeError(f"Z-factor root not bracketed on [{lo}, {hi}]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or hi - lo < tol:
            return mid
        if flo * fm < 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


_DAK_A =(0.3265, -1.0700, -0.5339, 0.01569, -0.05165, 0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210)


def _dak_coeffs(tpr: float) -> tuple[float, float, float, float]:
    """Temperature coefficients (C1, C2, C3, A10/Tr^3) of the DAK equation, computed once
    per solve (they do not depend on the reduced density)."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _DAK_A
    t = tpr
    c1 = a1 + a2 / t + a3 / t**3 + a4 / t**4 + a5 / t**5
    c2 = a6 + a7 / t + a8 / t**2
    c3 = a9 * (a7 / t + a8 / t**2)
    return c1, c2, c3, a10 / t**3


def _dak_z_of_rho_c(rho: float, c: tuple[float, float, float, float]) -> tuple[float, float]:
    """Z(rho_r) and dZ/drho_r for the DAK equation with precomputed coefficients."""
    c1, c2, c3, a10_t3 = c
    a11 = _DAK_A[10]
    r2 = rho * rho
    e = math.exp(-a11 * r2)
    z = 1.0 + c1 * rho + c2 * r2 - c3 * rho**5 + a10_t3 * (1.0 + a11 * r2) * r2 * e
    dz = (
        c1
        + 2.0 * c2 * rho
        - 5.0 * c3 * rho**4
        + (2.0 * a10_t3 * rho) * e * (1.0 + a11 * r2 - a11**2 * r2 * r2)
    )
    return z, dz


def _dak_z_of_rho(rho: float, tpr: float) -> tuple[float, float]:
    """Z(rho_r) and dZ/drho_r for the DAK equation."""
    return _dak_z_of_rho_c(rho, _dak_coeffs(tpr))


def dak_partials(rho: float, tpr: float) -> tuple[float, float, float]:
    """Z, (dZ/drho_r)_Tpr and (dZ/dTpr)_rho_r of the DAK equation at reduced density
    ``rho`` and pseudo-reduced temperature ``tpr`` (M2.5-3a).

    With Z = 1 + C1 r + C2 r^2 - C3 r^5 + C4 and C4 = A10 (1 + A11 r^2) (r^2/Tr^3) exp(-A11 r^2):
        dC1/dTr = -A2/Tr^2 - 3 A3/Tr^4 - 4 A4/Tr^5 - 5 A5/Tr^6
        dC2/dTr = -A7/Tr^2 - 2 A8/Tr^3
        dC3/dTr = A9 (-A7/Tr^2 - 2 A8/Tr^3)
        dC4/dTr = -3 C4 / Tr
        (dZ/dTr)_r = r dC1/dTr + r^2 dC2/dTr - r^5 dC3/dTr - 3 C4 / Tr
    COMPUTED: both partials are tested against central finite differences (1e-6 relative).
    """
    z, dz_dr = _dak_z_of_rho(rho, tpr)
    return z, dz_dr, _dak_dz_dtpr(rho, tpr)


def _dak_dz_dtpr(rho: float, tpr: float) -> float:
    """(dZ/dTpr)_rho_r alone (see dak_partials)."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _DAK_A
    t = tpr
    r2 = rho * rho
    e = math.exp(-a11 * r2)
    c4 = a10 * (1.0 + a11 * r2) * (r2 / t**3) * e
    dc1 = -a2 / t**2 - 3.0 * a3 / t**4 - 4.0 * a4 / t**5 - 5.0 * a5 / t**6
    dc2 = -a7 / t**2 - 2.0 * a8 / t**3
    dc3 = a9 * dc2
    return rho * dc1 + r2 * dc2 - rho**5 * dc3 - 3.0 * c4 / t


def _check_dak_range(tpr: float, ppr: float) -> None:
    # Tpr below 1.05 is the region where the DAK fit degrades toward the two-phase
    # boundary (M2.5-3a): warn there whatever the pressure. Above it, the published
    # ranges: Tpr to 3.0, Ppr 0.2 to 30 (Ppr below 1 only for Tpr <= 1).
    check_range("Dranchuk-Abou-Kassem", "Tpr", tpr, 1.05, 3.0)
    if tpr <= 1.0:
        check_range("Dranchuk-Abou-Kassem", "Ppr (Tpr <= 1)", ppr, 0.0, 1.0)
    else:
        check_range("Dranchuk-Abou-Kassem", "Ppr", ppr, 0.2, 30.0)


def dak_reduced_density(ppr: float, tpr: float, rho0: float | None = None) -> float:
    """Reduced density rho_r solving rho_r Z(rho_r, Tpr) = 0.27 Ppr / Tpr. Newton with a
    bisection fallback; the single Z solve everything else derives from. ``rho0`` is an
    optional warm start (the previous solution); the converged root is the same."""
    return _dak_solve(ppr, tpr, rho0)[0]


def _dak_solve(ppr: float, tpr: float, rho0: float | None = None) -> tuple[float, float, float]:
    """(rho_r, Z, dZ/drho_r) with Z and its derivative evaluated at the converged rho_r, so
    callers need no second evaluation. Same iteration sequence and stopping rule as the M2
    solver (start at Z = 1 unless warm-started), so the pre-3b path is unchanged."""
    _check_dak_range(tpr, ppr)
    k = 0.27 * ppr / tpr  # rho_r * Z = k
    c = _dak_coeffs(tpr)

    def f(rho: float) -> float:
        z, _ = _dak_z_of_rho_c(rho, c)
        return rho * z - k

    rho = k if (rho0 is None or not rho0 > 0.0) else rho0  # Z = 1 start unless warm-started
    z, dz = _dak_z_of_rho_c(rho, c)
    for _ in range(50):
        g = rho * z - k
        dg = z + rho * dz
        if dg == 0.0:
            break
        rho_new = rho - g / dg
        if rho_new <= 0.0:
            rho_new = 0.5 * rho
        converged = abs(rho_new - rho) < 1e-12
        rho = rho_new
        z, dz = _dak_z_of_rho_c(rho, c)
        if converged:
            break
    else:
        rho = math.nan
    if not (rho > 0.0 and abs(rho * z - k) < 1e-9):
        rho = _bisect(f, 1e-9, 3.5)
        z, dz = _dak_z_of_rho_c(rho, c)
    return rho, z, dz


def z_dak_full(ppr: float, tpr: float, rho0: float | None = None) -> tuple[float, float, float, float]:
    """(Z, rho_r, dZ/drho_r, dZ/dTpr) from one DAK solve. Ppr <= 0 gives the ideal gas."""
    if ppr <= 0.0:
        return 1.0, 0.0, dak_partials(0.0, tpr)[1], 0.0
    rho, z, dz_dr = _dak_solve(ppr, tpr, rho0)
    return z, rho, dz_dr, _dak_dz_dtpr(rho, tpr)


def z_dak(ppr: float, tpr: float) -> float:
    """Dranchuk-Abou-Kassem (1975) Z factor. Newton on reduced density, bisection fallback."""
    return z_dak_full(ppr, tpr)[0]


def z_hall_yarborough(ppr: float, tpr: float) -> float:
    """Hall-Yarborough (1973) Z factor. Newton on reduced density y, bisection fallback."""
    if ppr <= 0.0:
        return 1.0
    check_range("Hall-Yarborough", "Tpr", tpr, 1.2, 3.0)
    check_range("Hall-Yarborough", "Ppr", ppr, 0.1, 24.0)
    t = 1.0 / tpr
    lhs = 0.06125 * ppr * t * math.exp(-1.2 * (1.0 - t) ** 2)
    b1 = 14.76 * t - 9.76 * t**2 + 4.58 * t**3
    b2 = 90.7 * t - 242.2 * t**2 + 42.4 * t**3
    n = 2.18 + 2.82 * t

    def f(y: float) -> float:
        return -lhs + (y + y**2 + y**3 - y**4) / (1.0 - y) ** 3 - b1 * y**2 + b2 * y**n

    def df(y: float) -> float:
        return (
            (1.0 + 4.0 * y + 4.0 * y**2 - 4.0 * y**3 + y**4) / (1.0 - y) ** 4
            - 2.0 * b1 * y
            + n * b2 * y ** (n - 1.0)
        )

    y = lhs  # Z = 1 start
    for _ in range(50):
        g = f(y)
        d = df(y)
        if d == 0.0:
            break
        y_new = y - g / d
        if not (0.0 < y_new < 1.0):
            y_new = 0.5 * (y + (0.0 if y_new <= 0.0 else 1.0))
        if abs(y_new - y) < 1e-12:
            y = y_new
            break
        y = y_new
    else:
        y = math.nan
    if not (0.0 < y < 1.0 and abs(f(y)) < 1e-9):
        y = _bisect(f, 1e-9, 0.9)
    return lhs / y


Z_METHODS = {"dak": z_dak, "hall_yarborough": z_hall_yarborough}


# --- density, FVF, viscosity ----------------------------------------------------

def gas_density(p: float, t_f: float, gas_sg: float, z: float) -> float:
    """lbm/ft3 from the real gas law."""
    return p * M_AIR * gas_sg / (z * R_PSIA_FT3 * rankine(t_f))


def gas_fvf(p: float, t_f: float, z: float) -> float:
    """Bg in ft3/scf (divide by 5.6146 for rb/scf)."""
    return (P_SC / T_SC_R) * z * rankine(t_f) / p


def lee_gonzalez_eakin(t_f: float, gas_sg: float, rho_g: float, p: float | None = None) -> float:
    """Lee-Gonzalez-Eakin (1966) gas viscosity, cp.

    ``rho_g`` in lbm/ft3 at (p, T). ``p`` is only used for the range check.
    mu = 1e-4 K exp(X rho^Y) with rho in g/cm3, K = (9.4 + 0.02 M) T^1.5 / (209 + 19 M + T),
    X = 3.5 + 986/T + 0.01 M, Y = 2.4 - 0.2 X, T in degR.
    """
    check_range("Lee-Gonzalez-Eakin", "T", t_f, 100.0, 340.0, "degF")
    if p is not None:
        check_range("Lee-Gonzalez-Eakin", "p", p, 100.0, 8000.0, "psia")
    check_range("Lee-Gonzalez-Eakin", "gas_sg", gas_sg, 0.55, 1.0)
    t = rankine(t_f)
    m = M_AIR * gas_sg
    k = (9.4 + 0.02 * m) * t**1.5 / (209.0 + 19.0 * m + t)
    x = 3.5 + 986.0 / t + 0.01 * m
    y = 2.4 - 0.2 * x
    rho_gcc = rho_g / 62.428
    return 1e-4 * k * math.exp(x * rho_gcc**y)


# --- model ----------------------------------------------------------------------

class DryGasPVT(PVTModel):
    """Dry gas: no condensate, no water. Composition enters only through gravity and acid gas."""

    fluid_type = "dry_gas"

    def __init__(
        self,
        gas_sg: float,
        n2: float = 0.0,
        co2: float = 0.0,
        h2s: float = 0.0,
        *,
        z_method: str = "dak",
        tpc: float | None = None,
        ppc: float | None = None,
    ):
        """``tpc`` (degR) and ``ppc`` (psia) override the Sutton + Wichert-Aziz
        pseudo-criticals (e.g. a pure component's true critical point for validation)."""
        if z_method not in Z_METHODS:
            raise ValueError(f"unknown gas_z correlation {z_method!r}; choose from {list(Z_METHODS)}")
        self.gas_sg = float(gas_sg)
        self.n2, self.co2, self.h2s = float(n2), float(co2), float(h2s)
        self.z_method = z_method
        self._z = Z_METHODS[z_method]
        if (tpc is None) != (ppc is None):
            raise ValueError("tpc and ppc overrides must be given together")
        if tpc is not None:
            self.tpc, self.ppc = float(tpc), float(ppc)
            self.pseudocritical_source = "override"
        else:
            self.tpc, self.ppc = pseudocriticals(self.gas_sg, self.n2, self.co2, self.h2s)
            self.pseudocritical_source = "sutton_wichert_aziz"
        # gravity-based pseudo-composition for the ideal-gas heat capacity (heat_capacity.py);
        # raises if the gravity and inerts are inconsistent, warns above 0.90 HC gravity
        self.composition = heat_capacity.pseudo_composition(self.gas_sg, self.n2, self.co2, self.h2s)
        self._h_ref_j = self.composition.h_ideal_j(heat_capacity.kelvin(heat_capacity.T_REF_DEGF))
        self._s_ref_j = self.composition.s_ideal_j(heat_capacity.kelvin(heat_capacity.T_REF_DEGF))
        self._rho_r_last: float | None = None  # warm start for state(warm=True)

    def s0_from_ref(self, t: float) -> float:
        """Temperature part of the ideal-gas entropy above the reference state (ideal gas
        at T_REF_DEGF), Btu/(lbmol degR): integral of cp0/T dT; the pressure part
        -R ln(p/p_ref) is added by real_gas_thermo.entropy."""
        t_k = heat_capacity.kelvin(t)
        heat_capacity.check_t(t_k)
        return (self.composition.s_ideal_j(t_k) - self._s_ref_j) * heat_capacity.BTU_LBMOL_R_PER_J_MOL_K

    def h0_from_ref(self, t: float) -> float:
        """Ideal-gas enthalpy above the reference state (ideal gas at T_REF_DEGF), Btu/lbmol;
        the cached reference makes this one Horner evaluation per call."""
        t_k = heat_capacity.kelvin(t)
        heat_capacity.check_t(t_k)
        return (self.composition.h_ideal_j(t_k) - self._h_ref_j) / heat_capacity.J_PER_MOL_PER_BTU_PER_LBMOL

    # --- ideal-gas heat capacity ------------------------------------------------------

    def cp0_molar(self, t: float) -> float:
        """Ideal-gas cp, Btu/(lbmol degR), at ``t`` degF."""
        return heat_capacity.cp0_molar(t, self.composition)

    def cp0_mass(self, t: float) -> float:
        """Ideal-gas cp, Btu/(lbm degR), at ``t`` degF."""
        return heat_capacity.cp0_mass(t, self.composition)

    def k_ratio(self, t: float) -> float:
        """Ideal-gas cp0/cv0 at ``t`` degF (warns outside 200 to 600 K)."""
        return heat_capacity.k0(t, self.composition)

    def delta_h0(self, t1: float, t2: float) -> float:
        """Ideal-gas enthalpy change t1 -> t2 (degF), Btu/lbmol."""
        return heat_capacity.delta_h0(t1, t2, self.composition)

    def z(self, p: float, t: float) -> float:
        return self._z(p / self.ppc, rankine(t) / self.tpc)

    def bg(self, p: float, t: float) -> float:
        return gas_fvf(p, t, self.z(p, t))

    def gas_density(self, p: float, t: float) -> float:
        return gas_density(p, t, self.gas_sg, self.z(p, t))

    def gas_viscosity(self, p: float, t: float) -> float:
        return lee_gonzalez_eakin(t, self.gas_sg, self.gas_density(p, t), p)

    @property
    def molar_mass(self) -> float:
        """lbm/lbmol, M_air x gas_sg. Here the gravity IS the molar-mass route, so this is
        a rename with no arithmetic change; `gas_sg` stays because Sutton, Lee-Gonzalez-
        Eakin, Standing and the non-Darcy D all take a genuine specific gravity."""
        return M_AIR * self.gas_sg

    def thermo(self, p: float, t: float, state: PVTState | None = None) -> "ThermoState":
        """Real-gas properties through the DAK departures: today's
        ``thermo_from_state(self, state)``, moved behind the method so the march does not
        have to know how it is made. Verbatim; no behaviour change.

        ``state`` is the state at (p, t) when the caller already has it, which is the
        march's case: passing it keeps the gradient at ONE Z solve, as before g2-4."""
        from wellsim.core.pvt.real_gas_thermo import thermo_from_state

        return thermo_from_state(self, self.state(p, t, True) if state is None else state)

    def state(self, p: float, t: float, warm: bool = False) -> PVTState:
        """All gas properties from one Z solve (M2.5-3a): Z, reduced density, both DAK
        partials and (dp/drho)_T = (R T / M)(Z + rho_r dZ/drho_r). With a non-DAK Z method
        the partials are not available and (dp/drho)_T falls back to a forward difference
        on density (two extra Z solves). ``warm`` starts the DAK Newton from the previous
        call's reduced density (same root, fewer iterations); the adiabatic march uses it,
        the pre-3b profile path does not, so that path stays bit-for-bit."""
        t_r = rankine(t)
        tpr = t_r / self.tpc
        ppr = p / self.ppc
        if self.z_method == "dak":
            # the warm start is only safe where the DAK density root is unique (Tpr >= 1);
            # below that at Ppr > 1 the equation has several roots and a start-dependent
            # answer would make the march path-dependent
            hint = self._rho_r_last if (warm and tpr >= 1.0) else None
            z, rho_r, dz_dr, dz_dt = z_dak_full(ppr, tpr, hint)
            if warm:
                self._rho_r_last = rho_r
            dp_drho = R_PSIA_FT3 * t_r / (M_AIR * self.gas_sg) * (z + rho_r * dz_dr)
        else:
            z = self._z(ppr, tpr)
            rho_r = 0.27 * ppr / (z * tpr) if ppr > 0.0 else 0.0
            dz_dr = dz_dt = None
            dp = max(0.5, 0.005 * p)
            rho0 = gas_density(p, t, self.gas_sg, z)
            rho1 = gas_density(p + dp, t, self.gas_sg, self._z((p + dp) / self.ppc, tpr))
            dp_drho = dp / (rho1 - rho0)
        rho = gas_density(p, t, self.gas_sg, z)
        return PVTState(
            p=p,
            t=t,
            z=z,
            bg=gas_fvf(p, t, z),
            rho_g=rho,
            mu_g=lee_gonzalez_eakin(t, self.gas_sg, rho, p),
            gas_sg=self.gas_sg,
            tpr=tpr,
            ppr=ppr,
            rho_r=rho_r,
            dz_drho_r=dz_dr,
            dz_dtpr=dz_dt,
            dp_drho_t=dp_drho,
        )

    def __repr__(self) -> str:
        return (
            f"DryGasPVT(gas_sg={self.gas_sg}, n2={self.n2}, co2={self.co2}, h2s={self.h2s}, "
            f"z_method={self.z_method!r})"
        )
