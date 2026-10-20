"""Sudden area changes in the adiabatic march (M2.5-4).

Every change is treated as sudden (tapers are out of scope, docs/TODO-M3.md). The
march runs downward against the flow, so the state above a boundary is known and the
state below is solved for. With mass rate w (lbm/s), G = w/A the mass flux, h per lbm,
v = w/(rho A):

Flow-direction EXPANSION (the conduit below is narrower; the flow leaves the small pipe
below into the large pipe above). Known: large-pipe state (p_L, T_L) above. Unknown: the
small-pipe state (p_s, T_s) at the top of the small pipe. Borda-Carnot control volume
between the small-pipe exit plane and the large pipe:

    mass       G_s = G_L A_L / A_s
    momentum   p_s A_L + w v_s / g_c = p_L A_L + w v_L / g_c
    energy     h_s + v_s^2 / 2 = h_L + v_L^2 / 2

The energy equation gives T_s(p_s) (bracketed solve, gas-region rule of step 0a); the
momentum residual F(p_s) then has two roots, subsonic (p_s above the sonic pressure of
the small pipe) and supersonic. The subsonic root is taken. If F has no root the
restriction is choked: M_s = 1 on the adiabatic sound speed sets (p_s, T_s); the
pressures above are then a continuity estimate only (recorded as an InternalChoke).

Flow-direction CONTRACTION (the conduit below is wider; the flow enters the small pipe
above from the large pipe below). Known: small-pipe state (p_s, T_s) above. Unknown:
(p_L, T_L) at the top of the large pipe below.
    a. undo the irreversible loss at constant total enthalpy:
       p_s' = p_s + K_c rho_s v_s^2 / 2,  K_c = 0.5 (1 - beta^2), beta = d_s / d_L
       (Crane TP-410 sudden contraction, on the small-pipe velocity head; MEMORY)
    b. isentropic deceleration: s(p_L, T_L) = s(p_s', T_s') and h_L + v_L^2/2 = H

Both are solved as a bracketed Brent in pressure with the temperature from the energy
balance inside (Newton with the bracketed fallback), the same rule everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wellsim.core.exit import (
    G_C,
    _mach_adiabatic,
    is_gas_state,
    sonic_pressure_adiabatic,
    temperature_from_total_enthalpy,
)
from wellsim.core.numerics import brent
from wellsim.core.pvt.base import M_AIR
from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU as J, entropy_from_state, thermo_from_state

__all__ = ["AreaChange", "crane_contraction_coefficient", "expansion_borda_carnot", "isentropic_step",
           "contraction_with_loss"]


@dataclass(frozen=True)
class AreaChange:
    md: float
    kind: str  # "expansion" (flow-direction) or "contraction"
    conduit_from: str
    conduit_to: str
    p_before: float  # state above the boundary (known), psia
    t_before: float
    p_after: float  # state below the boundary (solved), psia
    t_after: float
    mach_after: float
    choked: bool
    loss_psi: float = 0.0  # contraction: K_c rho v^2/2 removed before the isentropic step
    residual: float = 0.0  # momentum (lbf) or entropy (Btu/lbmol-R) residual at the solution

    def as_dict(self) -> dict:
        return {
            "md": self.md, "kind": self.kind, "conduit_from": self.conduit_from, "conduit_to": self.conduit_to,
            "p_before": self.p_before, "t_before": self.t_before, "p_after": self.p_after, "t_after": self.t_after,
            "mach_after": self.mach_after, "choked": self.choked, "loss_psi": self.loss_psi,
            "residual": self.residual,
        }


class AreaChangeNotEvaluable(Exception):
    """No state below the reservoir-pressure ceiling satisfies the boundary conditions."""


def crane_contraction_coefficient(d_small: float, d_large: float) -> float:
    """K_c = 0.5 (1 - beta^2), beta = d_s/d_L (Crane TP-410 sudden contraction; MEMORY)."""
    beta = d_small / d_large
    return 0.5 * (1.0 - beta * beta)


def _t_energy(pvt, p, w, area, h_local, t_guess):
    """Temperature from the energy balance at (p, area); raises ValueError when the state
    is outside the gas region (treated by callers as the supersonic side)."""
    t = temperature_from_total_enthalpy(pvt, p, w, area, h_local, t_guess)
    st = pvt.state(p, t)
    if not is_gas_state(st):
        raise ValueError("state outside the gas region")
    return t, st


def _velocity(st, w, area):
    return w / (st.rho_g * area)


P_FLOOR = 1.0  # psia: the lowest pressure any branch is searched from


def _sonic_or_floor(pvt, w, area, h_local, t0, p_max):
    """(p*, T*) of ``area`` at this mass rate, or (P_FLOOR, None) when the flow is already
    subsonic at the floor (tiny rates: the sonic pressure lies below 1 psia). Raises
    AreaChangeNotEvaluable when no sonic state exists below the ceiling."""
    ma0 = _mach_adiabatic(pvt, P_FLOOR, w, area, h_local, t0)[0]
    if ma0 < 1.0:
        return P_FLOOR, None
    sonic = sonic_pressure_adiabatic(pvt, w, area, P_FLOOR, h_local, t0, p_max)
    if sonic is None:
        raise AreaChangeNotEvaluable("no subsonic state in the narrower conduit below the reservoir-pressure ceiling")
    return sonic


def expansion_borda_carnot(pvt, w, area_large, area_small, p_large, t_large, h_local, p_max, t_guess=None):
    """Small-pipe state (p_s, T_s, Mach, choked, residual) at the top of the narrower conduit
    below, from the large-pipe state above (flow-direction sudden expansion)."""
    st_l = pvt.state(p_large, t_large)
    v_l = _velocity(st_l, w, area_large)
    rhs = p_large * area_large * 144.0 + w * v_l / G_C  # lbf
    t0 = t_large if t_guess is None else t_guess

    def residual(p_s):
        t_s, st_s = _t_energy(pvt, p_s, w, area_small, h_local, t0)
        v_s = _velocity(st_s, w, area_small)
        return p_s * area_large * 144.0 + w * v_s / G_C - rhs, t_s, st_s, v_s

    # sonic state of the small pipe (Mach 1 on the adiabatic sound speed), or the floor when
    # the flow is subsonic all the way down
    p_star, t_star = _sonic_or_floor(pvt, w, area_small, h_local, t0, p_max)
    f_star = residual(p_star)[0]
    if f_star > 0.0:
        if t_star is None:
            raise AreaChangeNotEvaluable("momentum balance across the expansion has no root above 1 psia")
        # no root: the momentum balance cannot be met with a subsonic small-pipe state -> choked
        st_s = pvt.state(p_star, t_star)
        th = thermo_from_state(pvt, st_s)
        v_s = _velocity(st_s, w, area_small)
        return p_star, t_star, v_s / th.c_s, True, f_star
    # subsonic root: F rises with p_s above p*; grow the bracket upward, capped at p_max
    a, fa = p_star, f_star
    b = min(2.0 * p_star, p_max)
    while True:
        fb = residual(b)[0]
        if fb >= 0.0:
            break
        if b >= p_max:
            raise AreaChangeNotEvaluable("momentum balance across the expansion not met below the reservoir-pressure ceiling")
        a, fa = b, fb
        b = min(2.0 * b, p_max)
    p_s = brent(lambda p: residual(p)[0], a, b, fa=fa, fb=fb, xtol=1e-7, rtol=1e-12)
    f_s, t_s, st_s, v_s = residual(p_s)
    th = thermo_from_state(pvt, st_s)
    return p_s, t_s, v_s / th.c_s, False, f_s


def isentropic_step(pvt, w, area_in, area_out, p_in, t_in, h_local, p_max, t_guess=None):
    """State (p_out, T_out, Mach_out, residual) in ``area_out`` on the same streamline as
    (p_in, T_in) in ``area_in``: same entropy, same total enthalpy, subsonic branch.
    Deceleration (area_out > area_in) brackets p_out in [p_in, p_max]; acceleration
    brackets in [p*, p_in] with p* the sonic pressure of the smaller conduit."""
    st_in = pvt.state(p_in, t_in)
    s_target = entropy_from_state(pvt, st_in)
    t0 = t_in if t_guess is None else t_guess

    def residual(p_out):
        t_out, st_out = _t_energy(pvt, p_out, w, area_out, h_local, t0)
        return entropy_from_state(pvt, st_out) - s_target, t_out, st_out

    if area_out > area_in:
        a, fa = p_in, residual(p_in)[0]
        b = min(2.0 * p_in, p_max)
        while True:
            fb = residual(b)[0]
            if fa * fb <= 0.0:
                break
            if b >= p_max:
                raise AreaChangeNotEvaluable("isentropic deceleration has no solution below the reservoir-pressure ceiling")
            a, fa = b, fb
            b = min(2.0 * b, p_max)
    else:
        p_star, _ = _sonic_or_floor(pvt, w, area_out, h_local, t0, p_max)
        a, b = p_star, p_in
        fa, fb = residual(a)[0], residual(b)[0]
        if fa * fb > 0.0:
            raise AreaChangeNotEvaluable("isentropic acceleration: no subsonic state reaches the target entropy")
    p_out = brent(lambda p: residual(p)[0], a, b, fa=fa, fb=fb, xtol=1e-7, rtol=1e-12)
    f_out, t_out, st_out = residual(p_out)
    th = thermo_from_state(pvt, st_out)
    return p_out, t_out, _velocity(st_out, w, area_out) / th.c_s, f_out


def contraction_with_loss(pvt, w, area_small, area_large, p_small, t_small, h_local, p_max, k_c, t_guess=None):
    """Large-pipe state (p_L, T_L, Mach, loss_psi, residual) at the top of the wider conduit
    below, from the small-pipe state above (flow-direction sudden contraction): undo the
    loss K_c rho_s v_s^2/2 at constant total enthalpy, then decelerate isentropically."""
    st_s = pvt.state(p_small, t_small)
    v_s = _velocity(st_s, w, area_small)
    loss = k_c * st_s.rho_g * v_s * v_s / (2.0 * G_C * 144.0)  # psi
    p_s2 = p_small + loss
    t_s2, _ = _t_energy(pvt, p_s2, w, area_small, h_local, t_small if t_guess is None else t_guess)
    p_l, t_l, mach_l, resid = isentropic_step(pvt, w, area_small, area_large, p_s2, t_s2, h_local, p_max, t_s2)
    return p_l, t_l, mach_l, loss, resid
