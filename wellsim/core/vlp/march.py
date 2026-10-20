"""Marching integrator for the wellbore pressure profile (spec 4.3).

The gradient correlations give dp/dx = F(x, p) / (1 - Ma^2) with x measured depth.
Near a choked exit Ma -> 1 and dp/dx is singular, so the march is not done in x but
in a parameter s with

    dx/ds = +/- (1 - Ma^2)        dp/ds = +/- F(x, p)

(sign + marching downward, - marching upward). Both right-hand sides are bounded, the
sonic point becomes an ordinary point where x stalls while p keeps rising, and away
from it dx/ds ~ 1 so an s-step is a depth step. Steps are taken with the embedded
Bogacki-Shampine 3(2) pair with first-same-as-last, so an accepted step costs three
gradient evaluations. Step size is controlled on the pressure error (spec: 0.1 psi
per segment, ``p_tol``) and on depth error (``x_tol``), capped at ``max_step`` ft and
stretched or shrunk to land exactly on the next conduit boundary, so geometry changes
are always landed on. Adaptive refinement where the gradient changes fast follows
from the error control.

Choking
-------
Marching upward from a bottomhole pressure, reaching Ma = 1 before the exit means the
flow would choke inside the pipe: the march stops and reports ``choked_md``.

Marching downward, Mach can only rise where the conduit narrows (area drops). At each
such boundary the Mach number in the narrower conduit is checked at the continuity
pressure. If it is 1 or more, the restriction is choked: the pressure just below it is
the sonic pressure of the narrower conduit for this mass rate, not the value carried
down from above. The march resets the pressure to that sonic value, records the event
in ``chokes`` (depth, pressure before and after, conduit) and continues; the restriction,
not the exit, then controls the rate, and the pressures above it are only a continuity
estimate. Any other Mach >= 1 on a downward march is a logic error and raises
:class:`SupersonicError` rather than being clamped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from wellsim.core.exit import _mach_adiabatic, sonic_pressure
from wellsim.core.vlp.area_change import (
    AreaChange,
    AreaChangeNotEvaluable,
    contraction_with_loss,
    crane_contraction_coefficient,
    expansion_borda_carnot,
)
from wellsim.core.vlp.correlations import single_phase_gas

if TYPE_CHECKING:
    from wellsim.core.geometry import FlowSegment
    from wellsim.core.pvt.base import PVTModel
    from wellsim.core.temperature import LinearTemperature

__all__ = ["MarchResult", "InternalChoke", "march", "ChokedInside", "SupersonicError", "pressure_at"]

# A sonic start (choked exit or reset restriction) comes from a Brent solve on pressure and
# lands within ~1e-7 of Mach 1; that band is regular. Anything beyond it is an error.
_MACH_TOL = 1e-6


class _BelowFloor(Exception):
    pass


class ChokedInside(Exception):
    """Upward march reached Mach 1 before the exit."""

    def __init__(self, md: float, p: float):
        self.md, self.p = md, p
        super().__init__(f"flow reaches Mach 1 inside the conduit at md {md:.1f} ft, p {p:.1f} psia")


class SupersonicError(RuntimeError):
    """Mach > 1 met on a downward march where no choke logic applies."""


@dataclass(frozen=True)
class InternalChoke:
    md: float
    p_before: float  # continuity pressure carried down from above, psia
    p_after: float  # sonic pressure of the narrower conduit, psia
    mach_before: float
    conduit: str

    def as_dict(self) -> dict:
        return {"md": self.md, "p_before": self.p_before, "p_after": self.p_after,
                "mach_before": self.mach_before, "conduit": self.conduit}


@dataclass
class MarchResult:
    md: np.ndarray
    p: np.ndarray
    t: np.ndarray
    rho: np.ndarray
    v: np.ndarray
    mach: np.ndarray
    re: np.ndarray
    f: np.ndarray
    conduit: list[str]
    p_end: float
    md_end: float
    ok: bool
    message: str = ""
    steps: int = 0
    evaluations: int = 0
    correlation: str = single_phase_gas.NAME
    choked_md: float | None = None  # upward march: where Mach 1 was reached
    chokes: list[InternalChoke] = field(default_factory=list)  # downward march: choked restrictions
    rejected: int = 0
    # M2.5-3b
    adiabatic: bool = False
    t_end: float | None = None  # temperature at md_end (adiabatic: marched; profile: prescribed)
    tpr: np.ndarray | None = None  # pseudo-reduced temperature along the profile
    h_total: np.ndarray | None = None  # adiabatic: h/M + v^2/(2 g_c J) + g z/(g_c J), Btu/lbm (z = -TVD along md)
    area_changes: list[AreaChange] = field(default_factory=list)  # adiabatic: sudden expansions / contractions
    # M3-2 (two-phase model hook)
    alpha_g: np.ndarray | None = None  # in-situ gas fraction along the profile
    regime: list[str] | None = None  # flow regime along the profile

    @property
    def min_tpr(self) -> float | None:
        if self.tpr is None or len(self.tpr) == 0:
            return None
        return float(np.min(self.tpr))

    def as_dict(self) -> dict:
        return {
            "correlation": self.correlation,
            "md": self.md.tolist(), "p": self.p.tolist(), "t": self.t.tolist(), "rho": self.rho.tolist(),
            "v": self.v.tolist(), "mach": self.mach.tolist(), "re": self.re.tolist(), "f": self.f.tolist(),
            "conduit": list(self.conduit), "p_end": self.p_end, "md_end": self.md_end, "ok": self.ok,
            "message": self.message, "steps": self.steps, "evaluations": self.evaluations,
            "choked_md": self.choked_md, "chokes": [c.as_dict() for c in self.chokes],
            "adiabatic": self.adiabatic, "t_end": self.t_end,
            "tpr": None if self.tpr is None else self.tpr.tolist(),
            "h_total": None if self.h_total is None else self.h_total.tolist(),
            "area_changes": [a.as_dict() for a in self.area_changes],
            "alpha_g": None if self.alpha_g is None else self.alpha_g.tolist(),
            "regime": None if self.regime is None else list(self.regime),
        }


@dataclass
class _Trace:
    md: list = field(default_factory=list)
    p: list = field(default_factory=list)
    t: list = field(default_factory=list)
    rho: list = field(default_factory=list)
    v: list = field(default_factory=list)
    mach: list = field(default_factory=list)
    re: list = field(default_factory=list)
    f: list = field(default_factory=list)
    conduit: list = field(default_factory=list)
    tpr: list = field(default_factory=list)
    h_total: list = field(default_factory=list)
    alpha: list = field(default_factory=list)
    regime: list = field(default_factory=list)

    def add(self, x: float, g, conduit: str, z_term: float = 0.0) -> None:
        self.md.append(x)
        self.p.append(g.p)
        self.t.append(g.t)
        self.rho.append(g.rho)
        self.v.append(g.v)
        self.mach.append(g.mach)
        self.re.append(g.re)
        self.f.append(g.f)
        self.conduit.append(conduit)
        self.tpr.append(g.tpr)
        h = getattr(g, "h_total", None)
        self.h_total.append(math.nan if h is None else h + z_term)
        self.alpha.append(getattr(g, "alpha_g", math.nan))
        self.regime.append(getattr(g, "regime", ""))


def march(
    pvt: "PVTModel",
    segments: list["FlowSegment"],
    temperature: "LinearTemperature",
    w: float,
    p_start: float,
    md_start: float,
    md_end: float,
    *,
    max_step: float = 100.0,
    p_tol: float = 0.1,
    x_tol: float = 0.05,
    min_step: float = 1e-4,
    k_ratio: float | None = None,
    p_floor: float = 1.0,
    correlation: str = single_phase_gas.NAME,
    adiabatic: bool = False,
    t_start: float | None = None,
    t_tol: float = 0.005,
    p_max: float = math.inf,
    k_c: float | None = None,
    model=None,
) -> MarchResult:
    """Integrate the pressure from ``md_start`` to ``md_end`` for mass rate ``w`` (lbm/s).

    ``segments`` must cover the interval. Returns the profile (one row per accepted
    step, plus a row on each side of a choked restriction) and ``p_end``. ``ok`` is False
    if the flow chokes inside on an upward march or the pressure falls below
    ``p_floor``; ``message`` says why. Choked restrictions on a downward march are not a
    failure: they are listed in ``chokes``.

    ``adiabatic`` (M2.5-3b) marches the state [x, p, T] with the coupled momentum and
    energy equations (``single_phase_gas.gas_gradient_adiabatic``) from ``t_start``;
    ``temperature`` is then unused and the choke criterion is the adiabatic sound speed.
    Area changes (M2.5-4, downward marches) are sudden: a narrower conduit below is a
    flow-direction expansion solved with the Borda-Carnot momentum balance (choked when
    no subsonic root exists), a wider conduit below is a flow-direction contraction solved
    as a Crane loss (``k_c``, None = 0.5 (1 - beta^2)) plus an isentropic deceleration;
    see ``area_change.py``. Otherwise the temperature comes from the prescribed
    ``temperature`` profile and the arithmetic is the pre-3b path unchanged.

    ``model`` (M3-2) replaces the gradient and the area-change solves with an object
    exposing ``gradient(seg, x, p, t)`` (returning det, dp_ds, dt_ds, mach, h_total ...)
    and ``area_change(prev_seg, seg, x, p, t, h_local, p_max, k_c)`` (returning
    p_new, t_new, AreaChange, InternalChoke or None); the integrator, the step control
    and the choke logic are shared. With ``model`` None every operation of the gas path
    is unchanged.
    """
    if model is not None:
        if not adiabatic:
            raise ValueError("a march model needs adiabatic=True")
        correlation = model.name
    elif correlation != single_phase_gas.NAME:
        raise NotImplementedError(f"VLP correlation {correlation!r} lands in M3")
    if md_end == md_start:
        raise ValueError("md_start and md_end are equal")
    if adiabatic and t_start is None:
        raise ValueError("adiabatic march needs t_start")
    down = md_end > md_start
    sign = 1.0 if down else -1.0
    lo, hi = min(md_start, md_end), max(md_start, md_end)
    path = [s for s in segments if s.bottom_md > lo + 1e-9 and s.top_md < hi - 1e-9]
    if not path:
        raise ValueError("no segments cover the march interval")
    if not down:
        path = path[::-1]

    trace = _Trace()
    chokes: list[InternalChoke] = []
    area_changes: list[AreaChange] = []
    n_eval = 0
    n_steps = 0
    n_rej = 0
    x, p = float(md_start), float(p_start)
    t = float(t_start) if adiabatic else temperature.at(x)
    # potential-energy bookkeeping for the h_total trace (adiabatic): z_term = -TVD/J relative
    # to md_start, accumulated from sin(theta) segment by segment
    z_term = 0.0
    z_x0 = x

    def rhs(seg, xx, pp, tt):
        nonlocal n_eval
        if pp < p_floor:
            raise _BelowFloor
        n_eval += 1
        if model is not None:
            g = model.gradient(seg, xx, pp, tt)
        elif adiabatic:
            g = single_phase_gas.gas_gradient_adiabatic(pvt, pp, tt, w, seg)
        else:
            g = single_phase_gas.gas_gradient(pvt, pp, temperature.at(xx), w, seg, k_ratio)
        if not down and g.mach >= 1.0:
            raise ChokedInside(xx, pp)
        if down and g.mach > 1.0 + _MACH_TOL:
            raise SupersonicError(
                f"Mach {g.mach:.4f} at md {xx:.1f} ft, p {pp:.1f} psia in {seg.conduit} on a downward march; "
                "not at a conduit narrowing, so no choke treatment applies"
            )
        if adiabatic:
            return sign * max(g.det, 0.0), sign * g.dp_ds, sign * g.dt_ds, g
        ek = min(g.ek, 1.0)
        return sign * (1.0 - ek), sign * (g.grav + g.fric), 0.0, g

    def zt(seg, xx):
        # -TVD(x)/J relative to the march start, Btu/lbm
        from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU as J

        return z_term - (xx - z_x0) * seg.sin_theta / J

    def finish(ok, msg, choked_md=None):
        return MarchResult(
            md=np.array(trace.md), p=np.array(trace.p), t=np.array(trace.t), rho=np.array(trace.rho),
            v=np.array(trace.v), mach=np.array(trace.mach), re=np.array(trace.re), f=np.array(trace.f),
            conduit=trace.conduit, p_end=p, md_end=x, ok=ok, message=msg, steps=n_steps,
            evaluations=n_eval, correlation=correlation, choked_md=choked_md, chokes=chokes, rejected=n_rej,
            adiabatic=adiabatic, t_end=t if adiabatic else temperature.at(x),
            tpr=np.array(trace.tpr), h_total=np.array(trace.h_total) if adiabatic else None,
            area_changes=area_changes,
            alpha_g=np.array(trace.alpha) if model is not None else None,
            regime=list(trace.regime) if model is not None else None,
        )

    try:
        prev_area = None
        prev_seg = None
        for seg in path:
            x_target = min(seg.bottom_md, hi) if down else max(seg.top_md, lo)
            h = max_step  # fresh step size in every conduit
            if adiabatic and prev_area is not None and seg.area != prev_area:
                if not down:
                    raise NotImplementedError("adiabatic upward march across an area change is not supported")
                # sudden area change (M2.5-4): the state above is known, solve the state below
                h_local = k1[3].h_total  # h/M + v^2/2 at the boundary, previous conduit
                try:
                    if model is not None:
                        p_new, t_new, ac, choke = model.area_change(prev_seg, seg, x, p, t, h_local, p_max, k_c)
                        area_changes.append(ac)
                        if choke is not None:
                            chokes.append(choke)
                    elif seg.area < prev_area:
                        # flow-direction expansion (narrower below): Borda-Carnot, subsonic root
                        p_new, t_new, mach_new, choked_here, resid = expansion_borda_carnot(
                            pvt, w, prev_area, seg.area, p, t, h_local, p_max)
                        if choked_here:
                            mach_before = _mach_adiabatic(pvt, p, w, seg.area, h_local, t)[0]
                            chokes.append(InternalChoke(x, p, p_new, mach_before, seg.conduit))
                        area_changes.append(AreaChange(x, "expansion", prev_seg.conduit, seg.conduit, p, t,
                                                       p_new, t_new, mach_new, choked_here, 0.0, resid))
                    else:
                        # flow-direction contraction (wider below): Crane loss + isentropic deceleration
                        kc = crane_contraction_coefficient(prev_seg.d_hyd, seg.d_hyd) if k_c is None else k_c
                        p_new, t_new, mach_new, loss, resid = contraction_with_loss(
                            pvt, w, prev_area, seg.area, p, t, h_local, p_max, kc)
                        area_changes.append(AreaChange(x, "contraction", prev_seg.conduit, seg.conduit, p, t,
                                                       p_new, t_new, mach_new, False, loss, resid))
                    p, t = p_new, t_new
                except AreaChangeNotEvaluable as e:
                    return finish(False, f"area change at md {x:.1f} ft entering {seg.conduit}: {e}")
            elif (not adiabatic) and down and prev_area is not None and seg.area < prev_area:
                # profile mode (pre-3b path, unchanged): narrowing on the way down, Mach rises
                # here and only here; if choked, reset to the k-based sonic pressure
                g_in = single_phase_gas.gas_gradient(pvt, p, temperature.at(x), w, seg, k_ratio)
                n_eval += 1
                if g_in.mach >= 1.0:
                    p_star = sonic_pressure(pvt, w, temperature.at(x), seg.area, p, k_ratio)
                    chokes.append(InternalChoke(x, p, p_star, g_in.mach, seg.conduit))
                    p = p_star
            prev_area = seg.area
            prev_seg = seg
            k1 = rhs(seg, x, p, t)
            trace.add(x, k1[3], seg.conduit, zt(seg, x))
            while True:
                remaining = abs(x_target - x)
                if remaining <= 1e-7:
                    break
                # stretch the last step so it lands on the boundary: dx/ds = 1 - Ma^2 <= 1
                h_land = remaining / max(abs(k1[0]), 1e-3)
                h = min(h, max_step, h_land)
                while True:
                    try:
                        k2 = rhs(seg, x + 0.5 * h * k1[0], p + 0.5 * h * k1[1], t + 0.5 * h * k1[2])
                        k3 = rhs(seg, x + 0.75 * h * k2[0], p + 0.75 * h * k2[1], t + 0.75 * h * k2[2])
                        x3 = x + h * (2.0 / 9.0 * k1[0] + 1.0 / 3.0 * k2[0] + 4.0 / 9.0 * k3[0])
                        p3 = p + h * (2.0 / 9.0 * k1[1] + 1.0 / 3.0 * k2[1] + 4.0 / 9.0 * k3[1])
                        t3 = t + h * (2.0 / 9.0 * k1[2] + 1.0 / 3.0 * k2[2] + 4.0 / 9.0 * k3[2])
                        if p3 < p_floor:
                            raise _BelowFloor
                        k4 = rhs(seg, x3, p3, t3)
                    except _BelowFloor:
                        if h <= min_step:
                            return finish(False, f"pressure fell below {p_floor} psia near md {x:.1f} ft")
                        n_rej += 1
                        h = max(min_step, 0.25 * h)
                        continue
                    x2 = x + h * (7.0 / 24.0 * k1[0] + 0.25 * k2[0] + 1.0 / 3.0 * k3[0] + 0.125 * k4[0])
                    p2 = p + h * (7.0 / 24.0 * k1[1] + 0.25 * k2[1] + 1.0 / 3.0 * k3[1] + 0.125 * k4[1])
                    t2 = t + h * (7.0 / 24.0 * k1[2] + 0.25 * k2[2] + 1.0 / 3.0 * k3[2] + 0.125 * k4[2])
                    err = max(abs(p3 - p2) / p_tol, abs(x3 - x2) / x_tol, abs(t3 - t2) / t_tol)
                    overshoot = abs(x3 - x) - remaining
                    if overshoot > 1e-7 and h > min_step:
                        n_rej += 1
                        h = max(min_step, h * remaining / abs(x3 - x) * 0.999)
                        continue
                    if err <= 1.0 or h <= min_step:
                        break
                    n_rej += 1
                    h = max(min_step, h * max(0.2, 0.9 * err ** (-1.0 / 3.0)))
                # accept
                if abs(abs(x3 - x) - remaining) <= 1e-7:
                    x3 = x_target
                x, p, t = x3, p3, t3
                if not adiabatic:
                    t = temperature.at(x)
                k1 = k4
                n_steps += 1
                trace.add(x, k1[3], seg.conduit, zt(seg, x))
                if err > 0.0:
                    h = h * min(4.0, max(0.2, 0.9 * err ** (-1.0 / 3.0)))
                else:
                    h = h * 4.0
            # carry the potential term across the conduit boundary
            z_term = zt(seg, x)
            z_x0 = x
    except ChokedInside as e:
        return finish(False, str(e), choked_md=e.md)
    return finish(True, "")


def pressure_at(result: MarchResult, md: float) -> float:
    """Interpolate the profile pressure at a measured depth."""
    order = np.argsort(result.md, kind="stable")
    return float(np.interp(md, result.md[order], result.p[order]))
