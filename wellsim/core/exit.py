"""Exit (critical flow) condition for single-phase gas (spec 4.5).

The exit is choked when the gas would have to leave faster than its sonic velocity
to discharge at ambient pressure. Then the exit pressure floats above ambient to the
value that makes Mach = 1 at the exit area.

Sonic velocity of a real gas, isentropic:
    c^2 = k * g_c * (dp/drho)_T      with k = cp/cv
which reduces to the textbook c = sqrt(k Z R T g_c / M) when Z is constant. The
isothermal derivative (drho/dp)_T is taken from the PVT model by finite difference,
so any PVT model that gives gas density works. k is the ideal-gas cp0/cv0 of the gas at
the local temperature from the PVT model (``pvt.k_ratio(t)``, heat_capacity.py, M2.5-2)
unless a fixed value is passed (``Exit.k_override`` in the case schema reproduces the
pre-M2.5-2 constant 1.28). The real-gas isentropic exponent at high pressure is not
modelled (docs/TODO-M3.md).

The same Mach number is used in the marching integrator's acceleration term so that
the integrator's singularity and the choke criterion coincide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wellsim.core.numerics import bracket_upward, brent
from wellsim.core.pvt.base import M_AIR, P_SC, R_PSIA_FT3, T_SC_R

if TYPE_CHECKING:
    from wellsim.core.pvt.base import PVTModel, PVTState

__all__ = [
    "G_C",
    "gas_mass_rate",
    "drho_dp_isothermal",
    "sonic_velocity",
    "sonic_pressure",
    "ExitState",
    "exit_state",
    "temperature_from_total_enthalpy",
    "sonic_pressure_adiabatic",
    "exit_state_adiabatic",
    "NOT_EVALUABLE_CEILING",
    "is_gas_state",
]

G_C = 32.174  # lbm ft / (lbf s2)


def gas_mass_rate(q_mscfd: float, gas_sg: float) -> float:
    """lbm/s from a standard gas rate in Mscf/d (Z = 1 at standard conditions)."""
    rho_sc = P_SC * M_AIR * gas_sg / (R_PSIA_FT3 * T_SC_R)  # lbm/scf
    return q_mscfd * 1000.0 * rho_sc / 86400.0


def drho_dp_isothermal(pvt: "PVTModel", p: float, t: float, rho: float | None = None) -> float:
    """(drho/dp)_T in lbm/ft3 per psi, the inverse of the analytic (dp/drho)_T carried on
    the PVT state (M2.5-3a: (R T / M)(Z + rho_r dZ/drho_r) for DAK). ``rho`` is accepted for
    backward compatibility and unused."""
    return 1.0 / pvt.state(p, t).dp_drho_t


def sonic_velocity(
    pvt: "PVTModel",
    p: float,
    t: float,
    k: float | None = None,
    rho: float | None = None,
    state: "PVTState | None" = None,
) -> float:
    """Isentropic real-gas sonic velocity, ft/s: c^2 = k g_c (dp/drho)_T. ``k`` None means
    the PVT model's ideal-gas cp0/cv0 at ``t``; a number is used as given (override).
    Pass ``state`` (from ``pvt.state(p, t)``) to reuse an existing Z solve; ``rho`` is
    accepted for backward compatibility and unused."""
    if k is None:
        k = pvt.k_ratio(t)
    st = state if state is not None else pvt.state(p, t)
    d = st.dp_drho_t
    if d is None or d <= 0.0:
        raise ValueError(f"non-positive (dp/drho)_T at p = {p} psia, T = {t} degF")
    return math.sqrt(k * G_C * 144.0 * d)


@dataclass(frozen=True)
class ExitState:
    p: float  # psia at the exit
    t: float  # degF
    p_ambient: float  # psia
    rho: float  # lbm/ft3
    v: float  # ft/s
    c: float  # sonic velocity, ft/s
    mach: float
    choked: bool
    area: float  # ft2, effective (cd x geometric)
    not_evaluable: str | None = None  # set when no exit state exists (e.g. exit pressure would exceed reservoir pressure)
    cd: float = 1.0  # discharge coefficient applied to the geometric exit area
    area_geometric: float | None = None  # ft2, the exit_id (or top conduit) area before cd

    def as_dict(self) -> dict:
        return {
            "p": self.p, "t": self.t, "p_ambient": self.p_ambient, "rho": self.rho, "v": self.v,
            "sonic_velocity": self.c, "mach": self.mach, "choked": self.choked, "area_ft2": self.area,
            "not_evaluable": self.not_evaluable, "cd": self.cd,
            "area_geometric_ft2": self.area if self.area_geometric is None else self.area_geometric,
        }


def _mach(pvt, p, t, w, area, k):
    st = pvt.state(p, t)  # one Z solve for density and sonic velocity
    rho = st.rho_g
    v = w / (rho * area)
    c = sonic_velocity(pvt, p, t, k, state=st)
    return v / c, rho, v, c


def sonic_pressure(pvt: "PVTModel", w: float, t: float, area: float, p_lo: float, k: float | None = None) -> float:
    """Pressure at which mass rate ``w`` (lbm/s) through ``area`` (ft2) at ``t`` is exactly
    Mach 1. ``p_lo`` must be a pressure at which Mach >= 1 (the bracket grows upward from it).
    Used for a choked exit and for a choked internal restriction."""
    ma0 = _mach(pvt, p_lo, t, w, area, k)[0]
    if ma0 < 1.0:
        raise ValueError(f"sonic_pressure: Mach {ma0:.4f} < 1 at p_lo {p_lo:.2f} psia; nothing to solve")
    if ma0 == 1.0:
        return p_lo

    def g(p: float) -> float:
        return _mach(pvt, p, t, w, area, k)[0] - 1.0

    a, fa, b, fb = bracket_upward(g, p_lo, ma0 - 1.0)
    return brent(g, a, b, fa=fa, fb=fb, xtol=1e-6, rtol=1e-10)


def exit_state(
    pvt: "PVTModel",
    q_mscfd: float,
    p_ambient: float,
    t_exit: float,
    area: float,
    k: float | None = None,
) -> ExitState:
    """Exit pressure and Mach number for a gas rate ``q_mscfd`` discharging into ``p_ambient``.

    If Mach at ambient pressure is <= 1 the exit is unchoked at ambient pressure.
    Otherwise the exit pressure is raised until Mach = 1 (Mach falls monotonically with
    pressure because density rises faster than sonic velocity).
    """
    w = gas_mass_rate(q_mscfd, pvt.gas_sg)
    if w <= 0.0:
        rho = pvt.gas_density(p_ambient, t_exit)
        return ExitState(p_ambient, t_exit, p_ambient, rho, 0.0, sonic_velocity(pvt, p_ambient, t_exit, k, rho),
                         0.0, False, area)
    ma, rho, v, c = _mach(pvt, p_ambient, t_exit, w, area, k)
    if ma <= 1.0:
        return ExitState(p_ambient, t_exit, p_ambient, rho, v, c, ma, False, area)
    p_star = sonic_pressure(pvt, w, t_exit, area, p_ambient, k)
    ma, rho, v, c = _mach(pvt, p_star, t_exit, w, area, k)
    return ExitState(p_star, t_exit, p_ambient, rho, v, c, ma, True, area)


# --- adiabatic (energy-equation) exit, M2.5-3b -------------------------------------------
#
# With the energy equation the exit temperature is not an input: the total enthalpy per
# unit mass H_t = h/M + v^2/(2 g_c J) + g z/(g_c J) is conserved along the adiabatic
# wellbore, so at the exit (z = 0) T follows from p and the mass rate. The choke
# condition is v = c_s, the adiabatic sound speed from the real-gas departures.


def temperature_from_total_enthalpy(pvt, p: float, w: float, area: float, h_total: float, t_guess: float) -> float:
    """T (degF) at pressure ``p`` such that h(p, T)/M + v^2/(2 g_c J) = ``h_total`` (Btu/lbm,
    potential term already removed). Newton on T; the function is monotonic in T
    (cp/M - v^2 rho_T/(rho g_c J) > 0), with a Brent fallback."""
    from wellsim.core.numerics import brent
    from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU as J, thermo_from_state

    m = M_AIR * pvt.gas_sg

    def g_and_dg(t: float):
        st = pvt.state(p, t)
        th = thermo_from_state(pvt, st)
        v = w / (st.rho_g * area)
        g = th.h / m + v * v / (2.0 * G_C * J) - h_total
        dg = th.cp / m - v * v * th.rho_t / (st.rho_g * G_C * J)
        return g, dg

    t = t_guess
    for _ in range(30):
        g, dg = g_and_dg(t)
        dt = -g / dg
        if abs(dt) > 200.0:
            dt = math.copysign(200.0, dt)
        t += dt
        if not (t > _T_FLOOR_DEGF):
            break  # Newton left the physical range: decide by the bracket below, not by the path
        if abs(dt) < 1e-9:
            return t
    # Bracketed fallback. g is monotonic increasing in T, so a solution exists iff
    # g(T_floor) < 0 < g(T_high); the Newton start is irrelevant here.
    lo, hi = _T_FLOOR_DEGF, max(t_guess, 0.0) + 3000.0
    g_lo = g_and_dg(lo)[0]
    g_hi = g_and_dg(hi)[0]
    if g_lo >= 0.0 or g_hi <= 0.0:
        raise ValueError(f"no physical temperature at p = {p:.2f} psia for this total enthalpy and mass rate")
    return brent(lambda tt: g_and_dg(tt)[0], lo, hi, fa=g_lo, fb=g_hi, xtol=1e-9, rtol=1e-12)


_T_FLOOR_DEGF = -300.0  # 160 degR: absolute floor for the temperature solve
_TPR_GAS_FLOOR = 0.7  # DAK's published lower Tpr limit: below it the state is not a gas the model represents


def _mach_adiabatic(pvt, p, w, area, h_total, t_guess):
    """Mach number on the adiabatic sound speed at pressure ``p`` with T from the total
    enthalpy.

    The energy balance h(p, T)/M + v(T)^2/2 = H_t always has a solution in T (both terms
    fall with T), so "no physical state" means the solution lies outside the gas region
    the model represents: the kinetic energy of the expansion to this pressure can only be
    paid for by cooling the fluid to Tpr < 1 at Ppr > 1 (dense, liquid-like; outside DAK's
    published range and where its density equation has several roots) or below Tpr 0.7
    anywhere. Such a state is on the supersonic side of the choke, so the Mach number is
    reported as +inf and the choke search moves the pressure up. Never raises."""
    from wellsim.core.pvt.real_gas_thermo import thermo_from_state

    try:
        t = temperature_from_total_enthalpy(pvt, p, w, area, h_total, t_guess)
        if not (t > _T_FLOOR_DEGF):
            return math.inf, math.nan, math.nan, math.nan, t_guess
        st = pvt.state(p, t)
        if not is_gas_state(st):
            return math.inf, math.nan, math.nan, math.nan, t_guess
        th = thermo_from_state(pvt, st)
    except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
        return math.inf, math.nan, math.nan, math.nan, t_guess
    v = w / (st.rho_g * area)
    return v / th.c_s, st.rho_g, v, th.c_s, t


_Z_GAS_FLOOR = 0.3  # DAK vapour roots at Tpr 0.7 to 1, Ppr < 1 have Z ~ 0.6 to 0.9; liquid-like roots sit below 0.2
_TPR_DENSE_FLOOR = 1.05  # at Ppr > 1 the DAK fit degrades below this (the same limit the range warning uses)


def is_gas_state(st) -> bool:
    """True when the PVT state is one the dry-gas model represents.

    At Ppr > 1: Tpr >= 1.05 (below it DAK is outside its stated range; between 1.0 and
    1.05 at high Ppr the near-critical dense fluid makes the choked mass flux
    non-monotonic in rate, and below 1.0 the density equation has several roots).
    At Ppr <= 1: Tpr >= 0.7 with a gas-like Z (>= 0.3; liquid-like roots sit below 0.2).
    Everything else is the dense / liquid-like region the model does not describe.

    A state with NO reduced coordinates is accepted, deliberately (g2-4). This refusal
    exists to keep a hydrocarbon gas out of the dense phase, where DAK is unfitted and
    multi-rooted. It is written entirely in Tpr, Ppr and Z, which only a model with
    pseudo-criticals possesses. An ideal-gas model such as
    :class:`~wellsim.core.pvt.air.AirIdealGas` has none: air has no Sutton
    pseudo-criticals, so the rule is not failed by such a state, it is undefined for it
    and out of scope. Inventing pseudo-criticals to force a verdict would be worse than
    having none. Models outside the rule police their own range instead; AirIdealGas
    warns above 150 psia."""
    if st.tpr is None or st.ppr is None:
        return True  # no reduced coordinates: this rule does not apply to that model
    if st.ppr > 1.0:
        return st.tpr >= _TPR_DENSE_FLOOR
    if st.tpr < _TPR_GAS_FLOOR:
        return False
    return st.z >= _Z_GAS_FLOOR


NOT_EVALUABLE_CEILING = "exit pressure would exceed reservoir pressure"


def sonic_pressure_adiabatic(
    pvt, w: float, area: float, p_lo: float, h_total: float, t_guess: float, p_max: float = math.inf
) -> tuple[float, float] | None:
    """(p*, T*) at which mass rate ``w`` through ``area`` is exactly Mach 1 on the adiabatic
    sound speed, with T tied to p by the total enthalpy ``h_total``. ``p_lo`` must give
    Mach >= 1 (the bracket grows upward from it, by doubling, capped at ``p_max``).
    Returns None when Mach is still above 1 at ``p_max``: no sonic state exists below the
    ceiling (the outflow is not evaluable at this rate). Never raises for an unattainable
    state: a pressure with no physical temperature counts as Mach > 1."""
    ma0, _, _, _, t0 = _mach_adiabatic(pvt, p_lo, w, area, h_total, t_guess)
    if ma0 < 1.0:
        raise ValueError(f"sonic_pressure_adiabatic: Mach {ma0:.4f} < 1 at p_lo {p_lo:.2f} psia; nothing to solve")
    if ma0 == 1.0:
        return p_lo, t0
    t_last = [t0]

    def g(p: float) -> float:
        ma, _, _, _, t = _mach_adiabatic(pvt, p, w, area, h_total, t_last[0])
        t_last[0] = t
        return ma - 1.0

    # grow the bracket upward by doubling, capped at the ceiling
    a, fa = p_lo, ma0 - 1.0
    b = min(2.0 * p_lo, p_max)
    while True:
        fb = g(b)
        if fb <= 0.0:
            break
        if b >= p_max:
            return None
        a, fa = b, fb
        b = min(2.0 * b, p_max)
    # the low end may be an unattainable state (Mach +inf): bisect until it is finite
    for _ in range(80):
        if math.isfinite(fa):
            break
        mid = 0.5 * (a + b)
        fm = g(mid)
        if fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    p_star = brent(g, a, b, fa=fa, fb=fb, xtol=1e-6, rtol=1e-10)
    _, _, _, _, t_star = _mach_adiabatic(pvt, p_star, w, area, h_total, t_last[0])
    return p_star, t_star


def exit_state_adiabatic(
    pvt, q_mscfd: float, p_ambient: float, area: float, h_total: float, t_guess: float, p_max: float = math.inf
) -> ExitState:
    """Exit pressure, temperature and Mach number for rate ``q_mscfd`` discharging into
    ``p_ambient`` with total enthalpy ``h_total`` (Btu/lbm, at z = 0). Unchoked at ambient
    pressure when Mach <= 1 there; otherwise the pressure rises to the adiabatic Mach 1.
    When no sonic state exists below ``p_max`` (reservoir pressure) the returned state
    carries ``not_evaluable`` with the reason and nan pressures."""
    w = gas_mass_rate(q_mscfd, pvt.gas_sg)
    if w <= 0.0:
        st = pvt.state(p_ambient, t_guess)
        return ExitState(p_ambient, t_guess, p_ambient, st.rho_g, 0.0, sonic_velocity(pvt, p_ambient, t_guess, state=st),
                         0.0, False, area)
    ma, rho, v, c, t = _mach_adiabatic(pvt, p_ambient, w, area, h_total, t_guess)
    if ma <= 1.0:
        return ExitState(p_ambient, t, p_ambient, rho, v, c, ma, False, area)
    sonic = sonic_pressure_adiabatic(pvt, w, area, p_ambient, h_total, t, p_max)
    if sonic is None:
        return ExitState(math.nan, math.nan, p_ambient, math.nan, math.nan, math.nan, math.inf, True, area,
                         not_evaluable=NOT_EVALUABLE_CEILING)
    p_star, t_star = sonic
    ma, rho, v, c, t = _mach_adiabatic(pvt, p_star, w, area, h_total, t_star)
    return ExitState(p_star, t, p_ambient, rho, v, c, ma, True, area)
