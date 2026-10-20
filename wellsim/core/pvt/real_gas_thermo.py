"""Real-gas thermodynamics from the DAK Z equation (M2.5-3a, TODO-M3 item 3 groundwork).

Departure functions from a pressure-explicit Z(rho_r, Tr) equation, per mole:

    h_dep / (R T) = Z - 1 - Tr * I,        I = Integral_0^r (dZ/dTr)|_r dr / r
    u_dep / (R T) = - Tr * I
    cv_dep        = (d u_dep / dT)_rho = - R (2 Tr I + Tr^2 dI/dTr)
    cp - cv       = R (Z + Tr Z_T)^2 / (Z + r Z_r)          (Z_T = dZ/dTr, Z_r = dZ/dr)
    mu_JT         = (1 / rho_m) (Tr Z_T - r Z_r) / (Z + r Z_r) / cp
    c_s^2         = (cp / cv) (dp/drho)_T                     (adiabatic sound speed)

The integral I is closed-form for DAK, Z = 1 + C1 r + C2 r^2 - C3 r^5 + C4:

    I = r C1' + (r^2 / 2) C2' - (r^5 / 5) C3' - (3 A10 / (2 A11 Tr^4)) G(r)
    G = 2 - (2 + A11 r^2) exp(-A11 r^2)
    dI/dTr = r C1'' + (r^2 / 2) C2'' - (r^5 / 5) C3'' + (12 A10 / (2 A11 Tr^5)) G

with primes denoting d/dTr of the temperature coefficients:
    C1' = -A2/Tr^2 - 3 A3/Tr^4 - 4 A4/Tr^5 - 5 A5/Tr^6
    C1'' = 2 A2/Tr^3 + 12 A3/Tr^5 + 20 A4/Tr^6 + 30 A5/Tr^7
    C2' = -A7/Tr^2 - 2 A8/Tr^3;      C2'' = 2 A7/Tr^3 + 6 A8/Tr^4
    C3' = A9 C2';                     C3'' = A9 C2''

The C4 part follows from u = A11 r^2: Integral (1 + u) e^-u du / (2 A11) = [2 - (2 + u) e^-u] / (2 A11).
The owner's C4 integral and the three thermodynamic identities above are tagged COMPUTED:
the closed form is tested against numerical quadrature (1e-8), cp against a central
difference of h at constant p, and mu_JT against -(dh/dp)_T / cp by finite difference.

cp_real is analytic (cv from the u departure, then cp - cv from the pressure derivatives);
no finite-difference step is involved.

Reference state for h(T, p): ideal gas at ``T_REF`` (60 degF), h = 0. Only differences
are meaningful. Units: T degF in, p psia; h in Btu/lbmol, cp and cv in Btu/(lbmol degR),
mu_JT in degR/psi, sound speed in ft/s. DAK path only (a non-DAK Z method raises).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wellsim.core.numerics import brent
from wellsim.core.pvt.base import P_SC, R_PSIA_FT3, rankine
from wellsim.core.pvt.dry_gas import _DAK_A
from wellsim.core.pvt.heat_capacity import R_BTU_LBMOL_R

__all__ = [
    "T_REF",
    "BTU_PER_PSIA_FT3",
    "dak_departure_integral",
    "h_departure",
    "enthalpy",
    "cv_real",
    "cp_real",
    "mu_jt",
    "sound_speed",
    "isenthalpic_outlet_temperature",
    "ThermoState",
    "thermo_from_state",
    "J_FT_LBF_PER_BTU",
    "dak_entropy_integral",
    "s_departure",
    "s_departure_from_state",
    "entropy",
    "entropy_from_state",
]

J_FT_LBF_PER_BTU = 778.169  # mechanical equivalent of heat (MEMORY standard value)

from wellsim.core.pvt.heat_capacity import T_REF_DEGF as T_REF  # degF; ideal-gas enthalpy zero
BTU_PER_PSIA_FT3 = 144.0 / 778.169  # 0.185049 (778.169 ft lbf per Btu, MEMORY standard value)
# Gas constant in energy units, derived from the same R_PSIA_FT3 and J that the pressure
# derivatives use (1.985866 Btu/(lbmol degR)); the SI-derived R_BTU_LBMOL_R (1.985875) differs
# by 5e-6 relative, and mixing the two breaks the isentropic identity det = 1 - v^2/c_s^2 at
# the 1e-8 level (M2.5-3b). heat_capacity.k0 keeps the SI value so the pre-3b path is unchanged.
R = R_PSIA_FT3 * BTU_PER_PSIA_FT3
assert abs(R / R_BTU_LBMOL_R - 1.0) < 1e-5


def dak_departure_integral(rho_r: float, tpr: float) -> tuple[float, float]:
    """I = Integral_0^r (dZ/dTr)|_r dr / r and dI/dTr, closed form for DAK."""
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = _DAK_A
    t = tpr
    r = rho_r
    c1p = -a2 / t**2 - 3.0 * a3 / t**4 - 4.0 * a4 / t**5 - 5.0 * a5 / t**6
    c1pp = 2.0 * a2 / t**3 + 12.0 * a3 / t**5 + 20.0 * a4 / t**6 + 30.0 * a5 / t**7
    c2p = -a7 / t**2 - 2.0 * a8 / t**3
    c2pp = 2.0 * a7 / t**3 + 6.0 * a8 / t**4
    c3p = a9 * c2p
    c3pp = a9 * c2pp
    u = a11 * r * r
    g = 2.0 - (2.0 + u) * math.exp(-u)
    r2, r5 = r * r, r**5
    i = r * c1p + 0.5 * r2 * c2p - 0.2 * r5 * c3p - 3.0 * a10 * g / (2.0 * a11 * t**4)
    di = r * c1pp + 0.5 * r2 * c2pp - 0.2 * r5 * c3pp + 12.0 * a10 * g / (2.0 * a11 * t**5)
    return i, di


def dak_entropy_integral(rho_r: float, tpr: float) -> float:
    """J = Integral_0^r (Z - 1) dr / r, closed form for DAK:
    C1 r + C2 r^2/2 - C3 r^5/5 + (A10 / (2 A11 Tr^3)) [2 - (2 + A11 r^2) exp(-A11 r^2)]."""
    from wellsim.core.pvt.dry_gas import _dak_coeffs

    a10, a11 = _DAK_A[9], _DAK_A[10]
    c1, c2, c3, _ = _dak_coeffs(tpr)
    r = rho_r
    u = a11 * r * r
    g = 2.0 - (2.0 + u) * math.exp(-u)
    return c1 * r + 0.5 * c2 * r * r - 0.2 * c3 * r**5 + a10 * g / (2.0 * a11 * tpr**3)


def s_departure_from_state(st) -> float:
    """s - s_ideal at the same (T, p), Btu/(lbmol degR):
    s_dep / R = ln Z - Integral_0^r [(Z - 1) + Tr (dZ/dTr)|_r] dr / r = ln Z - J - Tr I."""
    i, _ = dak_departure_integral(st.rho_r, st.tpr)
    j = dak_entropy_integral(st.rho_r, st.tpr)
    return R * (math.log(st.z) - j - st.tpr * i)


def s_departure(pvt, p: float, t: float) -> float:
    return s_departure_from_state(_dak_state(pvt, p, t))


def entropy_from_state(pvt, st, t_ref: float = T_REF, p_ref: float = P_SC) -> float:
    """s(T, p) = s0(T) - R ln(p / p_ref) + s_dep, Btu/(lbmol degR), zero for the ideal gas
    at (t_ref, p_ref)."""
    if hasattr(pvt, "s0_from_ref"):
        s0 = pvt.s0_from_ref(st.t)
    else:
        raise NotImplementedError("entropy needs a PVT model with s0_from_ref")
    return s0 - R * math.log(st.p / p_ref) + s_departure_from_state(st)


def entropy(pvt, p: float, t: float, t_ref: float = T_REF, p_ref: float = P_SC) -> float:
    return entropy_from_state(pvt, _dak_state(pvt, p, t), t_ref, p_ref)


def _dak_state(pvt, p: float, t: float):
    st = pvt.state(p, t)
    if st.dz_drho_r is None or st.dz_dtpr is None:
        raise NotImplementedError("real-gas departures need the DAK Z method (analytic partials)")
    return st


@dataclass(frozen=True, slots=True)
class ThermoState:
    """Real-gas thermodynamic properties at one state, from one Z solve (M2.5-3b).

    Per lbmol unless noted: h_dep and h (Btu/lbmol, h from the ideal gas at T_REF), cp and
    cv (Btu/(lbmol degR)), mu_jt (degR/psi), c_s (ft/s), rho_t = (drho/dT)_p (lbm/ft3 per
    degR), rho_p = (drho/dp)_T (lbm/ft3 per psi)."""

    h_dep: float
    h: float
    cp: float
    cv: float
    mu_jt: float
    c_s: float
    rho_p: float
    rho_t: float


def thermo_from_state(pvt, st, t_ref: float = T_REF) -> "ThermoState":
    """All departure-based properties from an existing PVT state (no extra Z solve)."""
    from wellsim.core.exit import G_C

    if st.dz_drho_r is None or st.dz_dtpr is None:
        raise NotImplementedError("real-gas departures need the DAK Z method (analytic partials)")
    t_r = rankine(st.t)
    i, di = dak_departure_integral(st.rho_r, st.tpr)
    h_dep = R * t_r * (st.z - 1.0 - st.tpr * i)
    cp0 = pvt.cp0_molar(st.t)
    cv = cp0 - R - R * (2.0 * st.tpr * i + st.tpr * st.tpr * di)
    zr = st.z + st.rho_r * st.dz_drho_r
    zt = st.z + st.tpr * st.dz_dtpr
    cp = cv + R * zt * zt / zr
    rho_m = st.p / (st.z * R_PSIA_FT3 * t_r)
    mu = ((st.tpr * st.dz_dtpr - st.rho_r * st.dz_drho_r) / zr) / rho_m * BTU_PER_PSIA_FT3 / cp
    c_s = math.sqrt((cp / cv) * G_C * 144.0 * st.dp_drho_t)
    rho_p = 1.0 / st.dp_drho_t
    # (drho/dT)_p = -(dp/dT)_rho / (dp/drho)_T with (dp/dT)_rho = (p/T)(Z + Tr Z_T)/Z
    rho_t = -(st.p / t_r) * (zt / st.z) / st.dp_drho_t
    if t_ref == T_REF and hasattr(pvt, "h0_from_ref"):
        h = pvt.h0_from_ref(st.t) + h_dep  # cached reference: one Horner evaluation
    else:
        h = pvt.delta_h0(t_ref, st.t) + h_dep
    return ThermoState(h_dep, h, cp, cv, mu, c_s, rho_p, rho_t)


def h_departure(pvt, p: float, t: float) -> float:
    """h - h_ideal at (p, t), Btu/lbmol."""
    st = _dak_state(pvt, p, t)
    i, _ = dak_departure_integral(st.rho_r, st.tpr)
    return R * rankine(t) * (st.z - 1.0 - st.tpr * i)


def enthalpy(pvt, p: float, t: float, t_ref: float = T_REF) -> float:
    """h(T, p) = ideal-gas enthalpy from ``t_ref`` plus the departure, Btu/lbmol."""
    return pvt.delta_h0(t_ref, t) + h_departure(pvt, p, t)


def cv_real(pvt, p: float, t: float) -> float:
    """Real-gas cv, Btu/(lbmol degR): cv0 + cv_dep."""
    st = _dak_state(pvt, p, t)
    i, di = dak_departure_integral(st.rho_r, st.tpr)
    cv_dep = -R * (2.0 * st.tpr * i + st.tpr * st.tpr * di)
    return pvt.cp0_molar(t) - R + cv_dep


def _cp_minus_cv(st) -> float:
    num = st.z + st.tpr * st.dz_dtpr
    return R * num * num / (st.z + st.rho_r * st.dz_drho_r)


def cp_real(pvt, p: float, t: float) -> float:
    """Real-gas cp, Btu/(lbmol degR), analytic: cv_real + R (Z + Tr Z_T)^2 / (Z + r Z_r)."""
    st = _dak_state(pvt, p, t)
    return cv_real(pvt, p, t) + _cp_minus_cv(st)


def mu_jt(pvt, p: float, t: float) -> float:
    """Joule-Thomson coefficient (dT/dp)_h, degR per psi."""
    st = _dak_state(pvt, p, t)
    t_r = rankine(t)
    rho_m = p / (st.z * R_PSIA_FT3 * t_r)  # lbmol/ft3
    num = st.tpr * st.dz_dtpr - st.rho_r * st.dz_drho_r
    den = st.z + st.rho_r * st.dz_drho_r
    return (num / den) / rho_m * BTU_PER_PSIA_FT3 / cp_real(pvt, p, t)


def sound_speed(pvt, p: float, t: float) -> float:
    """Adiabatic sound speed, ft/s: c^2 = (cp/cv) (dp/drho)_T with real-gas cp and cv."""
    from wellsim.core.exit import G_C

    st = _dak_state(pvt, p, t)
    cv = cv_real(pvt, p, t)
    cp = cv + _cp_minus_cv(st)
    return math.sqrt((cp / cv) * G_C * 144.0 * st.dp_drho_t)


def isenthalpic_outlet_temperature(pvt, t_in: float, p_in: float, p_out: float) -> float:
    """Outlet temperature (degF) of an isenthalpic expansion from (t_in, p_in) to p_out,
    by Brent on h(T, p_out) = h(t_in, p_in)."""
    h_in = enthalpy(pvt, p_in, t_in)

    def g(t: float) -> float:
        return enthalpy(pvt, p_out, t) - h_in

    lo, hi = t_in - 400.0, t_in + 50.0
    return brent(g, lo, hi, xtol=1e-8, rtol=1e-12)
