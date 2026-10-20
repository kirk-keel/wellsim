"""Dynamic kill with a Newtonian kill fluid (M3-4). Blowing well only: the kill fluid is
injected at z_inj (TVD, default the sandface) at a given rate; relief-well hydraulics and
pump pressure are a later item (docs/TODO-M3.md).

Column model (COMPUTED, our choice): above z_inj the liquid is the formation liquid plus
the kill fluid with mass-weighted properties (specific volume, viscosity, heat capacity,
surface tension and speed of sound all mass-weighted); below z_inj the formation stream
alone. The march cannot change the rate mid-conduit, so the outflow is two legs: the
exit down to z_inj with the mixed stream, then z_inj down to the sandface with the
formation stream. The kill fluid enters at the stream's own temperature at z_inj (no
injection-temperature model yet), so the mixed stream's total enthalpy at the surface
datum is

    H_above = [(w_g + w_l) H_form + w_k (h_k(p_inj, T_inj) - TVD_inj/J)] / (w_g + w_l + w_k)

with H_form the formation stream's value (reservoir throttle) and (p_inj, T_inj) the leg-1
arrival state. The junction closes algebraically (perf pass, step 2): one march from a
guess gives p_inj; a temperature-only iteration at that pressure solves the balance (the
mixed stream's total-enthalpy inversion T(p_inj, H) against the mixing rule, one inversion
per pass, converged to 1e-6 F); one final march with the converged H_above gives the
profile. The final arrival differs from the balance temperature only through the pressure
shift between the two marches; that residual is recorded (``junction_residual_degf``,
0.002 to 0.15 F on the WGR example at 12 ppg) and not iterated. With w_k = 0 the outflow
is exactly ``outflow_profile`` (M3-3 path).

Kill criterion (owner's rule): the well is dynamically killed at q_k when the nodal solve
with q_k injected returns no roots (the outflow curve lies above the inflow curve at
every sandface pressure between ambient and reservoir pressure). q_k_min is the smallest
such rate, found by bracketing (doubling or halving from the Blount-Soeiinah rate) and
bisection to 1% relative on q_k. The search uses a coarser sandface-pressure scan than
the reported solve; the final q_k_min is confirmed with the full scan and the bisection
continues upward if the full scan finds a root. Within a search each residual-minimum
evaluation after the first is warm-started from the previous minimiser (perf pass,
step 4): a +/- 15% bracket around it, doubled while the minimum is not interior, then
the same golden section; the first evaluation runs the coarse scan as before.

Reported with every kill rate:

    Blount-Soeiinah rate (MEMORY attribution: Blount and Soeiinah, World Oil 1981, the
    dynamic-kill rate at which the single-phase kill column from z_inj to the exit,
    hydrostatic plus Moody friction, matches the reservoir pressure at the sandface).
    Static hold on an absolute basis: p_exit + 0.052 rho_k TVD_res against p_r (psia),
    margin in psi; when it fails the result carries the warning "dynamically dead only".
    The Blount-Soeiinah column at rest is the same quantity (the exit pressure plus the
    exact hydrostatic), so the two verdicts agree in sign.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from wellsim.core.exit import G_C, gas_mass_rate
from wellsim.core.friction import darcy_friction_factor, reynolds_number
from wellsim.core.geometry import flow_path_segments
from wellsim.core.nodal import NodalResult, find_operating_point_pwf
from wellsim.core.numerics import brent
from wellsim.core.pvt.base import M_AIR, P_SC
from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU as J, T_REF, enthalpy
from wellsim.core.vlp import two_phase as tp
from wellsim.core.vlp.march import MarchResult

if TYPE_CHECKING:
    from wellsim.cases.schema import Case, DynamicKill

__all__ = ["KillFluid", "kill_mass_rate", "mixed_liquid", "injection_md", "static_hold", "blount_soeiinah",
           "outflow_with_kill", "kill_nodal", "min_kill_rate", "kill_analysis", "kill_sweep", "PSI_PER_PPG_FT",
           "z_inj_state", "residual_minimum", "geothermal_at", "KillSearchError"]

class KillSearchError(RuntimeError):
    """The kill search returned a rate that is not the answer by its own definition
    (ks-2 proving check). The message carries both residuals."""


PSI_PER_PPG_FT = 0.052  # owner's hydrostatic constant (0.0519 exact from 7.4805 lbm/ft3 per ppg / 144)
_FT3_PER_BBL = 5.614583
LBM_PER_FT3_PER_PPG = 7.480519480


@dataclass(frozen=True)
class KillFluid:
    rho: float  # lbm/ft3
    mu: float  # cp
    cp: float  # Btu/(lbm degF)
    sigma: float  # dyne/cm
    c: float  # ft/s
    warnings: tuple[str, ...] = ()
    t_inject: float | None = None
    """degF the kill fluid enters at, when it is imposed (rw-1). None is the M3-4 behaviour:
    the fluid enters at the blowing stream's own temperature at z_inj. Setting it is how the
    injection-temperature bounds are run, and it changes only the enthalpy the kill fluid
    carries into the mixing balance, not the column or the friction."""

    @classmethod
    def from_schema(cls, dk: "DynamicKill") -> "KillFluid":
        sigma, w1 = dk.sigma_or_default()
        c, w2 = dk.c_or_default()
        return cls(dk.rho_k * LBM_PER_FT3_PER_PPG, dk.mu_k, dk.cp_k, sigma, c, tuple(w for w in (w1, w2) if w))

    @property
    def rho_ppg(self) -> float:
        return self.rho / LBM_PER_FT3_PER_PPG

    def h(self, p: float, t: float) -> float:
        """Btu/lbm, same reference as the formation liquid (COMPUTED)."""
        return self.cp * (t - T_REF) + 144.0 * (p - P_SC) / (self.rho * J)

    def as_liquid(self) -> tp.LiquidProps:
        return tp.LiquidProps(self.rho, self.mu, self.cp, self.sigma, self.c)


def kill_mass_rate(q_bpm: float, rho: float) -> float:
    """lbm/s from bbl/min at the kill-fluid density (lbm/ft3)."""
    return q_bpm * _FT3_PER_BBL / 60.0 * rho


def mixed_liquid(liq_form: tp.LiquidProps | None, w_l: float, kill: KillFluid, w_k: float) -> tp.LiquidProps:
    """Liquid above z_inj: formation liquid plus kill fluid, mass-weighted (COMPUTED)."""
    if w_l <= 0.0 or liq_form is None:
        return kill.as_liquid()
    x_k = w_k / (w_l + w_k)
    x_l = 1.0 - x_k
    rho = 1.0 / (x_l / liq_form.rho_l + x_k / kill.rho)
    return tp.LiquidProps(
        rho, x_l * liq_form.mu_l + x_k * kill.mu, x_l * liq_form.cp_l + x_k * kill.cp,
        x_l * liq_form.sigma + x_k * kill.sigma, x_l * liq_form.c_l + x_k * kill.c,
    )


def injection_md(case: "Case", z_inj_tvd: float | None) -> float:
    """Measured depth of the injection point from its TVD (bisection on the survey);
    None means the sandface."""
    res = case.reservoir
    if z_inj_tvd is None:
        return float(res.depth_md)
    tvd_res = case.well.md_to_tvd(res.depth_md)
    if z_inj_tvd >= tvd_res:
        return float(res.depth_md)
    if z_inj_tvd <= 0.0:
        raise ValueError("z_inj must be above the sandface and below surface")
    lo, hi = 0.0, float(res.depth_md)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if case.well.md_to_tvd(mid) < z_inj_tvd:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def static_hold(case: "Case", kill: KillFluid) -> dict[str, Any]:
    """Absolute basis (M3 closeout): p_exit + 0.052 rho_k TVD_res against p_r (psia), with
    p_exit the case exit pressure, so a subsea exit counts. Margin in psi."""
    tvd = case.well.md_to_tvd(case.reservoir.depth_md)
    p_exit = case.exit.ambient_pressure
    p_hyd = PSI_PER_PPG_FT * kill.rho_ppg * tvd
    p_sandface = p_exit + p_hyd
    margin = p_sandface - case.reservoir.pressure
    return {"rho_k_ppg": kill.rho_ppg, "tvd_ft": tvd, "p_exit_psia": p_exit, "p_hydrostatic_psi": p_hyd,
            "p_sandface_static_psia": p_sandface, "p_res_psia": case.reservoir.pressure,
            "margin_psi": margin, "ok": margin >= 0.0}


def _column_pressure(case: "Case", kill: KillFluid, md_inj: float, q_bpm: float) -> tuple[float, list[dict]]:
    """Sandface pressure of the single-phase kill column from the exit to md_inj at rate
    q_bpm: ambient + hydrostatic + Moody friction, segment by segment."""
    res = case.reservoir
    segments = flow_path_segments(case.well, case.flow_path, res.depth_md)
    q_ft3 = q_bpm * _FT3_PER_BBL / 60.0
    p = case.exit.ambient_pressure
    rows = []
    for seg in segments:
        top, bottom = seg.top_md, min(seg.bottom_md, md_inj)
        if bottom <= top:
            continue
        length = bottom - top
        tvd = case.well.md_to_tvd(bottom) - case.well.md_to_tvd(top)
        v = q_ft3 / seg.area
        re = reynolds_number(kill.rho, v, seg.d_ft, kill.mu) if v > 0.0 else 0.0
        f = darcy_friction_factor(re, seg.rel_rough) if re > 0.0 else 0.0
        dp_hyd = kill.rho * tvd / 144.0
        dp_fric = f * kill.rho * v * v / (2.0 * G_C * seg.d_ft) / 144.0 * length
        p += dp_hyd + dp_fric
        rows.append({"conduit": seg.conduit, "length_ft": length, "tvd_ft": tvd, "v_ft_s": v, "re": re, "f": f,
                     "dp_hydrostatic_psi": dp_hyd, "dp_friction_psi": dp_fric})
    return p, rows


def blount_soeiinah(case: "Case", kill: KillFluid, md_inj: float, q_hi: float = 1000.0) -> dict[str, Any]:
    """Kill rate (bbl/min) at which the single-phase kill column from z_inj to the exit
    gives p_r at the sandface (MEMORY attribution: Blount and Soeiinah 1981). Zero when
    the static column already exceeds p_r. The column below z_inj is the formation liquid
    (or static gas) and is not part of the criterion, which is the owner's definition."""
    p_r = case.reservoir.pressure
    p0, rows0 = _column_pressure(case, kill, md_inj, 0.0)
    if p0 >= p_r:
        return {"q_bpm": 0.0, "p_static_psia": p0, "segments": rows0, "static_exceeds_p_res": True}

    def g(q):
        return _column_pressure(case, kill, md_inj, q)[0] - p_r

    a, fa = 0.0, p0 - p_r
    b = 1.0
    fb = g(b)
    while fb < 0.0:
        a, fa = b, fb
        b *= 2.0
        if b > q_hi:
            return {"q_bpm": math.inf, "p_static_psia": p0, "segments": rows0, "static_exceeds_p_res": False}
        fb = g(b)
    q = brent(g, a, b, fa=fa, fb=fb, xtol=1e-6, rtol=1e-10)
    p, rows = _column_pressure(case, kill, md_inj, q)
    return {"q_bpm": q, "p_static_psia": p0, "p_sandface_psia": p, "segments": rows, "static_exceeds_p_res": False}


# --- outflow with injection --------------------------------------------------------------------

def _concat_profiles(a: MarchResult, b: MarchResult) -> MarchResult:
    def cat(x, y):
        return None if x is None or y is None else np.concatenate([x, y])

    return MarchResult(
        md=cat(a.md, b.md), p=cat(a.p, b.p), t=cat(a.t, b.t), rho=cat(a.rho, b.rho), v=cat(a.v, b.v),
        mach=cat(a.mach, b.mach), re=cat(a.re, b.re), f=cat(a.f, b.f), conduit=list(a.conduit) + list(b.conduit),
        p_end=b.p_end, md_end=b.md_end, ok=a.ok and b.ok, message=b.message or a.message, steps=a.steps + b.steps,
        evaluations=a.evaluations + b.evaluations, correlation=a.correlation, choked_md=b.choked_md or a.choked_md,
        chokes=list(a.chokes) + list(b.chokes), rejected=a.rejected + b.rejected, adiabatic=True, t_end=b.t_end,
        tpr=cat(a.tpr, b.tpr), h_total=cat(a.h_total, b.h_total), area_changes=list(a.area_changes) + list(b.area_changes),
        alpha_g=cat(a.alpha_g, b.alpha_g), regime=(list(a.regime) + list(b.regime)) if a.regime is not None else None,
    )


def _static_below(case: "Case", pvt, liq_form: tp.LiquidProps | None, p_inj: float, t_inj: float,
                  md_inj: float) -> float:
    """Sandface pressure below z_inj with no formation flow: a static column of the
    formation liquid if the case has one, else a static gas column at reservoir
    temperature (RK4 in TVD, COMPUTED)."""
    res = case.reservoir
    dz = case.well.md_to_tvd(res.depth_md) - case.well.md_to_tvd(md_inj)
    if liq_form is not None:
        return p_inj + liq_form.rho_l * dz / 144.0

    def grad(p):
        return pvt.state(p, res.temperature).rho_g / 144.0

    p, h = p_inj, dz / 20.0
    for _ in range(20):
        k1 = grad(p)
        k2 = grad(p + 0.5 * h * k1)
        k3 = grad(p + 0.5 * h * k2)
        k4 = grad(p + h * k3)
        p += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return p


def outflow_with_kill(case: "Case", pvt, q_gas_mscfd: float, w_liquid: float, kill: KillFluid, w_k: float,
                      md_inj: float, liq_form: tp.LiquidProps | None, t_tol: float = 0.2,
                      p_inj_guess: float | None = None) -> tp.Outflow:
    """Exit state and sandface pressure with w_k (lbm/s) of kill fluid injected at md_inj.
    w_k = 0 is exactly outflow_profile (M3-3). ``p_inj_guess`` seeds the enthalpy iteration
    (the scanned sandface pressure is a good seed when z_inj is the sandface)."""
    if w_k <= 0.0:
        return tp.outflow_profile(case, pvt, q_gas_mscfd, w_liquid)
    res, tm = case.reservoir, case.temperature
    md_res = float(res.depth_md)
    tvd_res = case.well.md_to_tvd(md_res)
    tvd_inj = case.well.md_to_tvd(md_inj)
    w_g = gas_mass_rate(q_gas_mscfd, pvt.gas_sg)
    w_f = w_g + w_liquid
    w_total = w_f + w_k
    liq_mix = mixed_liquid(liq_form, w_liquid, kill, w_k)
    m_gas = M_AIR * pvt.gas_sg
    h_form = 0.0
    if w_f > 0.0:
        x_g = w_g / w_f
        h_in = x_g * (enthalpy(pvt, res.pressure, res.temperature) / m_gas if w_g > 0.0 else 0.0)
        if w_liquid > 0.0:
            h_in += (1.0 - x_g) * liq_form.h_l(res.pressure, res.temperature)
        h_form = h_in - tvd_res / J
    p_inj, t_inj = (res.pressure if p_inj_guess is None else p_inj_guess), res.temperature

    def h_above_of(p, t):
        # rw-1: with kill.t_inject set the kill fluid enters at that temperature instead of
        # the blowing stream's temperature at z_inj; everything else is unchanged
        h_k = kill.h(p, t if kill.t_inject is None else kill.t_inject) - tvd_inj / J
        return (w_f * h_form + w_k * h_k) / w_total if w_f > 0.0 else h_k

    def leg1_march(h_above):
        return tp.outflow_profile(case, pvt, q_gas_mscfd, w_liquid + w_k, liquid=liq_mix, h_total=h_above, md_end=md_inj)

    # Perf step 2 (junction without re-marching): one march from the guess, then close the
    # mixing balance at the arrival pressure with a temperature-only iteration (each pass
    # one enthalpy inversion of the mixed stream at (p_inj, H), no march) to 1e-6 F, then
    # one final march with the converged H_above. The final arrival differs from the
    # balance temperature only through the pressure shift between the two marches; that
    # residual is kept on the result (``junction_residual_degf``, 0.002 to 0.15 F on the
    # WGR example at 12 ppg, 1e-5 to 3e-5 relative on pwf) and is not iterated further.
    leg1 = leg1_march(h_above_of(p_inj, t_inj))
    if leg1.profile is None or not leg1.profile.ok:
        leg1.w_liquid = w_liquid
        leg1.w_kill = w_k
        return leg1
    p_inj = leg1.profile.p_end
    t_inj = leg1.profile.t_end
    segments = flow_path_segments(case.well, case.flow_path, md_res)
    seg_inj = next((s for s in segments if s.top_md - 1e-9 <= md_inj <= s.bottom_md + 1e-9), segments[-1])
    geom_inj = tp.geometry_of(seg_inj)
    h_above = h_above_of(p_inj, t_inj)
    for _ in range(50):
        t_new = tp.temperature_from_total_enthalpy_mix(pvt, liq_mix, p_inj, w_g, w_liquid + w_k, geom_inj,
                                                        h_above + tvd_inj / J, t_inj)
        h_new = h_above_of(p_inj, t_new)
        converged = abs(t_new - t_inj) < 1e-6
        t_inj, h_above = t_new, h_new
        if converged:
            break
    leg1 = leg1_march(h_above)
    if leg1.profile is None or not leg1.profile.ok:
        leg1.w_liquid = w_liquid
        leg1.w_kill = w_k
        return leg1
    p_inj = leg1.profile.p_end
    t_arrival = leg1.profile.t_end
    leg1.junction_residual_degf = t_arrival - t_inj
    t_inj = t_arrival
    out = leg1
    out.w_liquid = w_liquid
    out.w_kill = w_k
    out.md_inj = md_inj
    out.p_inj, out.t_inj = p_inj, t_inj
    if md_inj < md_res - 1e-6:
        if w_f > 0.0:
            leg2 = tp.march_leg(case, pvt, liq_form if liq_form is not None else kill.as_liquid(), w_g, w_liquid,
                                p_inj, t_inj, md_inj, md_res)
            out.profile = _concat_profiles(leg1.profile, leg2)
        else:
            p_sf = _static_below(case, pvt, liq_form, p_inj, t_inj, md_inj)
            out.profile.p_end = p_sf
            out.profile.md_end = md_res
    return out


# --- nodal with injection, kill rate search ---------------------------------------------------------

def _formation_liquid_rate(case: "Case", pwf: float, q_g: float) -> float:
    li = case.liquid_inflow
    return 0.0 if li is None else li.liquid_rate(pwf, q_g, case.reservoir.pressure)


def kill_nodal(case: "Case", pvt, ipr, kill: KillFluid, q_k_bpm: float, md_inj: float,
               liq_form: tp.LiquidProps | None, *, n_scan: int | None = None, derivatives: bool = False,
               outflows: dict | None = None, scan_only: bool = False) -> NodalResult:
    """Nodal solve with q_k_bpm of kill fluid injected at md_inj. ``scan_only`` skips the
    Brent refinement (the kill search only needs to know whether any root exists)."""
    res, num = case.reservoir, case.options.numerics
    p_amb = case.exit.ambient_pressure
    w_k = kill_mass_rate(q_k_bpm, kill.rho)
    rho_l = liq_form.rho_l if liq_form is not None else kill.rho
    tvd_res = case.well.md_to_tvd(res.depth_md)

    def outflow_pwf(pwf: float) -> float:
        q_g = ipr.rate(pwf)
        q_l = _formation_liquid_rate(case, pwf, q_g)
        if q_g <= 0.0 and q_l <= 0.0 and w_k <= 0.0:
            return p_amb + (rho_l * tvd_res / 144.0)  # M3-3 static-liquid endpoint
        out = outflow_with_kill(case, pvt, q_g, tp.liquid_mass_rate_stbd(q_l, rho_l), kill, w_k, md_inj, liq_form,
                                p_inj_guess=pwf if md_inj >= res.depth_md - 1e-6 else None)
        if outflows is not None:
            outflows[pwf] = out
        return math.inf if out.pwf is None else out.pwf

    return find_operating_point_pwf(
        ipr.rate, None, p_amb, res.pressure, n_scan=n_scan or num.nodal_scan_points,
        pressure_tol=num.pressure_tol, rate_tol=num.rate_tol, outflow_pwf=outflow_pwf, derivatives=derivatives,
        scan_only=scan_only,
    )


def killed(nodal: NodalResult) -> bool:
    return not nodal.intersections


def _residual_fn(case: "Case", pvt, ipr, kill: KillFluid, w_k: float, md_inj: float,
                 liq_form: tp.LiquidProps | None, counter: list):
    """f(pwf) = pwf_vlp(q_g(pwf), q_l(pwf); w_k) - pwf, +inf where not evaluable."""
    res = case.reservoir
    p_amb = case.exit.ambient_pressure
    rho_l = liq_form.rho_l if liq_form is not None else kill.rho
    tvd_res = case.well.md_to_tvd(res.depth_md)
    cache: dict[float, float] = {}

    def f(pwf: float) -> float:
        if pwf in cache:
            return cache[pwf]
        q_g = ipr.rate(pwf)
        q_l = _formation_liquid_rate(case, pwf, q_g)
        if q_g <= 0.0 and q_l <= 0.0 and w_k <= 0.0:
            v = p_amb + rho_l * tvd_res / 144.0
        else:
            counter[0] += 1
            out = outflow_with_kill(case, pvt, q_g, tp.liquid_mass_rate_stbd(q_l, rho_l), kill, w_k, md_inj, liq_form,
                                    p_inj_guess=pwf if md_inj >= res.depth_md - 1e-6 else None)
            v = math.inf if out.pwf is None else out.pwf
        cache[pwf] = v - pwf
        return cache[pwf]

    return f


WARM_START_HALF_WIDTH = 0.15  # of the pwf span, either side of the previous minimiser


def _warm_bracket(f, p_start: float, p_amb: float, p_res: float, half_frac: float) -> tuple | None:
    """Bracket [a, b] around ``p_start`` with the residual minimum interior (f(p_start) at or
    below both ends), starting at +/- half_frac of the span and doubling the half-width while
    the minimum sits on an end. None when the bracket has grown to the whole span without an
    interior minimum, or f(p_start) is not evaluable: the caller falls back to the cold scan."""
    span = p_res - p_amb
    fm = f(p_start)
    if not math.isfinite(fm):
        return None
    half = half_frac * span
    while True:
        a, b = max(p_amb, p_start - half), min(p_res, p_start + half)
        fa, fb = f(a), f(b)
        if fm <= fa and fm <= fb:
            return a, b, fm
        if a <= p_amb and b >= p_res:
            return None
        half *= 2.0


def residual_minimum(case: "Case", pvt, ipr, kill: KillFluid, q_k_bpm: float, md_inj: float,
                     liq_form: tp.LiquidProps | None, *, n_scan: int = 8, counter: list | None = None,
                     golden_iters: int = 6, p_start: float | None = None) -> tuple[float, float]:
    """g(q_k) = min over pwf of the outflow-minus-inflow residual, and the pwf where it sits:
    a coarse scan then golden-section refinement in the bracket around the scan minimum.
    g > 0 means no root (killed); g < 0 means the well still flows (COMPUTED).

    ``p_start`` (perf, warm start): the pwf that minimised the previous evaluation. The
    golden section then starts from a bracket of +/- 15% of the pwf span around it, doubled
    while the minimum is not interior; the coarse scan is skipped. Falls back to the scan
    when no interior bracket is found. The criterion (the continuous minimum) is unchanged;
    the golden bracket is the same size as the scan's (0.30 against 0.29 of the span)."""
    res = case.reservoir
    p_amb = case.exit.ambient_pressure
    counter = counter if counter is not None else [0]
    f = _residual_fn(case, pvt, ipr, kill, kill_mass_rate(q_k_bpm, kill.rho), md_inj, liq_form, counter)
    warm = None
    if p_start is not None and p_amb < p_start < res.pressure:
        warm = _warm_bracket(f, p_start, p_amb, res.pressure, WARM_START_HALF_WIDTH)
    if warm is not None:
        a, b, best = warm
        best_p = p_start
    else:
        grid = [p_amb + (res.pressure - p_amb) * i / (n_scan - 1) for i in range(n_scan)]
        vals = [f(p) for p in grid]
        i = min(range(n_scan), key=lambda k: vals[k])
        a = grid[max(i - 1, 0)]
        b = grid[min(i + 1, n_scan - 1)]
        best_p, best = grid[i], vals[i]
        if not math.isfinite(best):
            return best, best_p
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(golden_iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    for p, v in ((c, fc), (d, fd)):
        if v < best:
            best, best_p = v, p
    return best, best_p


def min_kill_rate(case: "Case", pvt, ipr, kill: KillFluid, md_inj: float, liq_form: tp.LiquidProps | None, *,
                  q_start: float | None = None, q_max: float = 200.0, rel_tol: float = 0.01,
                  n_scan_search: int = 8, want_z_inj_state: bool = False) -> dict[str, Any]:
    """Smallest injection rate (bbl/min) at which the well is dead: the root in q_k of
    g(q_k) = min over pwf of [pwf_vlp - pwf], which is continuous and rises with q_k.

    The continuous minimum is the ONLY killed / not-killed verdict (the CLAUDE.md rule).
    ks-2 deleted the scan gate that used to confirm the rate: on rw_wgr it certified
    25.8774 bbl/min as killed while the continuous minimum there was -37.52 psi, because
    the still-flowing window was 90.4 psi wide against a 332.4 psi scan step, 0.27 of one
    step (out/ks-diagnosis.md). The 1% bump that went with it is gone too.

    Bracket by doubling / halving from ``q_start`` (the Blount-Soeiinah rate, or 5 bbl/min
    when the static column already holds), then Brent with the tolerance set RELATIVE TO
    THE ROOT: ``brent`` takes an absolute ``xtol``, so it is solved, ``xtol`` is re-set to
    ``rel_tol * |root|``, and it is solved again until the tolerance used and the root agree
    to a factor of two. The old ``xtol = rel_tol * hi`` was 1% of the bracket TOP, so the
    answer moved with whatever bracket the warm start happened to produce: 25.6230 from
    [20, 40] and 26.6198 from [15, 30] against a true root of 26.1555.

    Brent is then finished by bisection on the bracket it landed in, so the returned rate
    is the upper end of a bracket narrower than ``rel_tol``: g at the answer is at or above
    zero and g just below it is negative, by construction rather than by hope.

    Proving check, always on and not a test (ks-2 step 3): at the returned q_k the residual
    minimum must be >= 0 and at q_k (1 - rel_tol) it must be < 0. Either failure raises
    :class:`KillSearchError` with both values. Two evaluations, one of them cached.
    """
    counter = [0]
    t0 = time.perf_counter()
    gcache: dict[float, float] = {}
    p_prev: list[float | None] = [None]  # minimiser of the previous evaluation: warm start of the next

    def g(q):
        if q not in gcache:
            gcache[q], p_min = residual_minimum(case, pvt, ipr, kill, q, md_inj, liq_form, n_scan=n_scan_search,
                                                counter=counter, p_start=p_prev[0])
            if math.isfinite(gcache[q]):
                p_prev[0] = p_min
        return gcache[q]

    def result(q, lo, hi, note=None):
        d = {"q_k_min_bpm": q, "bracket_bpm": (lo, hi), "evaluations": counter[0], "seconds": time.perf_counter() - t0}
        if note:
            d["note"] = note
        return d

    if q_start is None or not math.isfinite(q_start) or q_start <= 0.0:
        q_start = 5.0
    q0 = max(q_start, 0.05)
    g0 = g(q0)
    if g0 > 0.0:
        hi, ghi = q0, g0
        lo = q0 / 2.0
        glo = g(lo)
        while glo > 0.0 and lo > 0.02:
            hi, ghi = lo, glo
            lo /= 2.0
            glo = g(lo)
        if glo > 0.0:
            if g(0.0) > 0.0:
                return result(0.0, 0.0, hi, "no roots even without injection")
            lo, glo = 0.0, g(0.0)
    else:
        lo, glo = q0, g0
        hi = 2.0 * q0
        ghi = g(hi)
        while ghi <= 0.0:
            lo, glo = hi, ghi
            hi *= 2.0
            if hi > q_max:
                return result(None, lo, None, f"not killed at {q_max:g} bbl/min")
            ghi = g(hi)
    # Brent with the tolerance relative to the root, not to the bracket top. brent takes
    # an absolute xtol, so iterate: solve, re-set xtol from the root, solve again, until the
    # tolerance used and the root agree to a factor of two. rtol is 0 so the criterion is
    # exactly xtol and not xtol + rtol |x|.
    xtol = rel_tol * abs(hi)
    passes = []
    for _ in range(8):
        q = brent(g, lo, hi, fa=glo, fb=ghi, xtol=xtol, rtol=0.0)
        passes.append((xtol, q))
        want = rel_tol * abs(q)
        if want > 0.0 and 0.5 <= xtol / want <= 2.0:
            break
        xtol = want
    # finish on the bracket Brent landed in, so the answer is the upper end of a bracket
    # narrower than rel_tol: that is what makes the proving check below hold by construction
    if g(q) >= 0.0:
        hi, ghi = q, g(q)
    else:
        lo, glo = q, g(q)
    while lo < hi * (1.0 - rel_tol):
        mid = 0.5 * (lo + hi)
        gm = g(mid)
        if gm >= 0.0:
            hi, ghi = mid, gm
        else:
            lo, glo = mid, gm
    q_k = hi
    _prove_kill_rate(g, q_k, rel_tol)
    out = result(q_k, lo, hi)
    out["xtol_passes"] = [list(t) for t in passes]
    out["xtol_used"] = xtol
    out["xtol_over_root"] = xtol / (rel_tol * abs(q_k)) if q_k else None
    if want_z_inj_state:
        # rw-1 only: a warm-started residual minimum plus one outflow call at the converged
        # rate, about 10 to 15% on top of the search. Off by default so a kill search with
        # no relief well behind it does not pay for a state nothing reads.
        out["z_inj_state"] = z_inj_state(case, pvt, ipr, kill, q_k, md_inj, liq_form, p_start=p_prev[0],
                                         counter=counter)
    return out


def _prove_kill_rate(g, q_k: float, rel_tol: float) -> None:
    """ks-2 step 3, always on: the returned rate is the answer by its own definition, or the
    search raises. g(q_k) >= 0 says the well is dead there; g(q_k (1 - rel_tol)) < 0 says it
    is the SMALLEST such rate to within the tolerance. Two evaluations, one cached."""
    if q_k is None or q_k <= 0.0:
        return  # zero rate (the column alone kills) has nothing below it to check
    g_at = g(q_k)
    q_below = q_k * (1.0 - rel_tol)
    g_below = g(q_below)
    if not (g_at >= 0.0):
        raise KillSearchError(
            f"kill search returned q_k = {q_k:.6g} bbl/min but the residual minimum there is "
            f"{g_at:.6g} psi, which is negative: the well still flows at the rate reported as "
            f"killing it (residual at {q_below:.6g} bbl/min is {g_below:.6g} psi)"
        )
    if not (g_below < 0.0):
        raise KillSearchError(
            f"kill search returned q_k = {q_k:.6g} bbl/min with residual minimum {g_at:.6g} psi, "
            f"but the residual at {q_below:.6g} bbl/min is already {g_below:.6g} psi, at or above "
            f"zero: a smaller rate kills the well, so this is not the minimum kill rate"
        )


def z_inj_state(case: "Case", pvt, ipr, kill: KillFluid, q_k_bpm: float, md_inj: float,
                liq_form: tp.LiquidProps | None, *, p_start: float | None = None,
                counter: list | None = None) -> dict[str, Any] | None:
    """The blowing well's own state (p, T) at z_inj at the kill condition (rw-1).

    At q_k_min the well has no roots, so there is no operating point; the state that means
    something is the one at the RESIDUAL MINIMISER, the sandface pressure where the outflow
    curve touches the inflow curve. ``residual_minimum`` already returns that pwf and
    ``outflow_with_kill`` already records p_inj / t_inj on every evaluation, so this is one
    outflow call at the converged (q_k_min, minimiser), not a second search and not a
    separate hydraulic model of the blowing well. About 0.06 s.

    Returns None when the state is not evaluable there."""
    counter = counter if counter is not None else [0]
    if q_k_bpm is None or not math.isfinite(q_k_bpm):
        return None
    _, p_min = residual_minimum(case, pvt, ipr, kill, q_k_bpm, md_inj, liq_form, counter=counter,
                                p_start=p_start)
    q_g = ipr.rate(p_min)
    q_l = _formation_liquid_rate(case, p_min, q_g)
    rho_l = liq_form.rho_l if liq_form is not None else kill.rho
    counter[0] += 1
    out = outflow_with_kill(case, pvt, q_g, tp.liquid_mass_rate_stbd(q_l, rho_l), kill,
                            kill_mass_rate(q_k_bpm, kill.rho), md_inj, liq_form,
                            p_inj_guess=p_min if md_inj >= case.reservoir.depth_md - 1e-6 else None)
    if out.p_inj is None or out.profile is None or not out.profile.ok:
        return None
    return {
        "q_k_bpm": q_k_bpm, "pwf_psia": p_min, "md_inj_ft": md_inj,
        "tvd_inj_ft": case.well.md_to_tvd(md_inj),
        "p_psia": float(out.p_inj), "t_degf": float(out.t_inj),
        "q_g_mscfd": float(q_g), "q_l_stbd": float(q_l),
        "basis": "residual minimiser at the converged kill rate (one outflow evaluation, not a new search)",
    }


def kill_analysis(case: "Case", pvt, ipr, *, search: bool = True, q_start: float | None = None) -> dict[str, Any]:
    """Static hold, Blount-Soeiinah rate, q_k_min and (if dynamic_kill.rate_bpm is given)
    the nodal solve at that rate. Never reports a kill rate without the static-hold line."""
    dk = case.dynamic_kill
    kill = KillFluid.from_schema(dk)
    md_inj = injection_md(case, dk.z_inj)
    liq_form = tp.LiquidProps.from_schema(case.liquid) if case.liquid is not None else None
    warnings = list(kill.warnings)
    st = static_hold(case, kill)
    if not st["ok"]:
        warnings.append(
            f"static hold fails: p_exit {st['p_exit_psia']:.1f} + 0.052 x {kill.rho_ppg:g} ppg x {st['tvd_ft']:.0f} ft = "
            f"{st['p_sandface_static_psia']:.0f} psia against p_r {st['p_res_psia']:.0f} psia "
            f"(margin {st['margin_psi']:.0f} psi): dynamically dead only"
        )
    bs = blount_soeiinah(case, kill, md_inj)
    out: dict[str, Any] = {
        "rho_k_ppg": kill.rho_ppg, "mu_k_cp": kill.mu, "z_inj_tvd_ft": case.well.md_to_tvd(md_inj), "md_inj_ft": md_inj,
        "static_hold": st, "blount_soeiinah": bs, "warnings": warnings,
    }
    temperature: dict[str, Any] | None = None
    if search and dk.find_min_rate:
        q_bs = bs["q_bpm"] if math.isfinite(bs["q_bpm"]) else None
        q_seed = q_start if q_start is not None else (q_bs if q_bs else None)
        rw_block = case.relief_well
        if rw_block is not None and rw_block.temperature_runs:
            # rw-2: the headline kill rate is the one at the Ramey arrival temperature (the
            # upper bound on how warm the kill fluid can reach z_inj, and so the higher,
            # conservative kill rate). Getting there costs one search at the surface
            # temperature first, because the arrival temperature needs a mass rate.
            temperature, kill, q_seed = _arrival_temperature_runs(
                case, pvt, ipr, kill, md_inj, liq_form, q_seed, warnings
            )
        out["q_k_min"] = min_kill_rate(case, pvt, ipr, kill, md_inj, liq_form,
                                       q_start=q_seed,
                                       want_z_inj_state=case.relief_well is not None)
        if out["q_k_min"]["q_k_min_bpm"] is None:
            warnings.append(out["q_k_min"]["note"])
        if temperature is not None:
            _close_arrival_runs(case, pvt, ipr, kill, md_inj, liq_form, out, temperature, warnings)
    if dk.rate_bpm is not None:
        nodal = kill_nodal(case, pvt, ipr, kill, dk.rate_bpm, md_inj, liq_form, derivatives=True)
        out["at_rate"] = {"q_k_bpm": dk.rate_bpm, "killed": killed(nodal), "nodal": nodal.as_dict()}
    if case.relief_well is not None:
        from wellsim.core.relief_well import relief_well_analysis

        q_k = out.get("q_k_min", {}).get("q_k_min_bpm")
        state = out.get("q_k_min", {}).get("z_inj_state")
        rw = relief_well_analysis(case, kill, q_k, state)
        if temperature is not None:
            rw["temperature"] = temperature
        elif q_k is not None:
            # the Monte Carlo path: one search at the M3-4 default injection temperature,
            # and the arrival temperature reported from that rate without a re-run. rw-3
            # makes that the finite-U estimate rather than the U -> infinity bound.
            from wellsim.core.relief_well import ramey_arrival, ramey_arrival_finite_u

            arrival = (ramey_arrival_finite_u(case, kill, q_k) if case.relief_well.finite_u
                       else ramey_arrival(case, kill, q_k))
            rw["arrival"] = arrival.as_dict()
            rw["notes"].append(
                f"arrival temperature {arrival.t_arrival_degf:.2f} degF is the "
                + ("finite-U estimate" if case.relief_well.finite_u else "Ramey upper bound")
                + f" at this "
                f"kill rate, but the rate itself was found with the kill fluid entering at the blowing "
                f"stream temperature at z_inj (the M3-4 default), not at the arrival temperature: "
                f"relief_well.temperature_runs is off"
            )
            warnings.extend(w for w in arrival.warnings if w not in warnings)
        out["relief_well"] = rw
        warnings.extend(w for w in rw["warnings"] if w not in warnings)
    return out


def _arrival_temperature_runs(case: "Case", pvt, ipr, kill: KillFluid, md_inj: float,
                              liq_form: tp.LiquidProps | None, q_seed: float | None,
                              warnings: list[str]):
    """rw-2 first half: the surface-temperature (lower-bound) kill run, then the Ramey
    arrival temperature it implies. Returns (report, the KillFluid to search with, q_seed).

    The arrival temperature needs a mass rate and the kill rate needs an injection
    temperature, so the two are coupled. rw-2 does not iterate: it takes the arrival
    temperature at the lower-bound rate, searches once at that temperature, and reports how
    far the arrival temperature moves when recomputed at the answer (``_close_arrival_runs``).
    """
    from dataclasses import replace as _replace

    from wellsim.core.relief_well import ramey_arrival

    rw = case.relief_well
    t_surface = rw.surface_temperature
    if t_surface is None:
        from wellsim.core.relief_well import DEFAULT_SURFACE_T_DEGF

        t_surface = DEFAULT_SURFACE_T_DEGF
    cold = min_kill_rate(case, pvt, ipr, _replace(kill, t_inject=t_surface), md_inj, liq_form,
                         q_start=q_seed)
    q_cold = cold["q_k_min_bpm"]
    if q_cold is None:
        warnings.append(
            "the surface-temperature kill run returned no rate, so the Ramey arrival temperature has no "
            "mass rate to work from; the search falls back to the M3-4 injection temperature"
        )
        return None, kill, q_seed
    arrival = ramey_arrival(case, kill, q_cold)  # rw-2: the U -> infinity bound
    warnings.extend(w for w in arrival.warnings if w not in warnings)
    report: dict[str, Any] = {
        "lower_bound": {"label": "kill fluid enters z_inj at the surface temperature (no heating)",
                        "t_inject_degf": t_surface, "q_k_min_bpm": q_cold,
                        "evaluations": cold["evaluations"], "seconds": cold["seconds"]},
        "arrival": arrival.as_dict(),
    }
    if rw.finite_u:
        # rw-3: the estimate. A finite U adds the string's own resistance to the formation's,
        # so the fluid arrives cooler than the bound and the kill rate is lower. This is the
        # temperature the headline run uses; the bound becomes a third run in the closeout.
        from wellsim.core.relief_well import ramey_arrival_finite_u

        estimate = ramey_arrival_finite_u(case, kill, q_cold)
        warnings.extend(w for w in estimate.warnings if w not in warnings)
        report["estimate"] = estimate.as_dict()
        return report, _replace(kill, t_inject=estimate.t_arrival_degf), q_cold
    return report, _replace(kill, t_inject=arrival.t_arrival_degf), q_cold


def _close_arrival_runs(case: "Case", pvt, ipr, kill: KillFluid, md_inj: float,
                        liq_form: tp.LiquidProps | None, out: dict[str, Any],
                        temperature: dict[str, Any], warnings: list[str]) -> None:
    """rw-2 second half: label the headline run, recompute the arrival temperature at the
    rate that came out, and add the optional geothermal comparison run."""
    from dataclasses import replace as _replace

    from wellsim.core.relief_well import ramey_arrival

    from wellsim.core.relief_well import ramey_arrival_finite_u

    rw = case.relief_well
    q_reported = out["q_k_min"]["q_k_min_bpm"]
    t_bound = temperature["arrival"]["t_arrival_degf"]
    q_cold = temperature["lower_bound"]["q_k_min_bpm"]
    estimating = "estimate" in temperature
    t_reported = temperature["estimate"]["t_arrival_degf"] if estimating else t_bound

    if estimating:
        # rw-3: the headline run is the finite-U estimate; the U -> infinity bound is a
        # third search so the three rates can be reported together.
        temperature["estimate_run"] = {
            "label": "kill fluid enters z_inj at the finite-U arrival temperature (the ESTIMATE)",
            "t_inject_degf": t_reported, "q_k_min_bpm": q_reported,
            "evaluations": out["q_k_min"]["evaluations"], "seconds": out["q_k_min"]["seconds"],
            "reported": True,
        }
        if rw.bound_run:
            bound = min_kill_rate(case, pvt, ipr, _replace(kill, t_inject=t_bound), md_inj,
                                  liq_form, q_start=q_reported)
            temperature["upper_bound"] = {
                "label": "kill fluid enters z_inj at the U -> infinity Ramey bound (maximum heating)",
                "t_inject_degf": t_bound, "q_k_min_bpm": bound["q_k_min_bpm"],
                "evaluations": bound["evaluations"], "seconds": bound["seconds"],
                "reported": False,
            }
        else:
            # rw-3b: the Monte Carlo runs the surface and finite-U searches only, two per
            # sample instead of three; the U -> infinity rate is not searched for.
            temperature["upper_bound"] = {
                "label": "kill fluid enters z_inj at the U -> infinity Ramey bound (not searched)",
                "t_inject_degf": t_bound, "q_k_min_bpm": None, "evaluations": 0, "seconds": 0.0,
                "reported": False, "skipped": True,
            }
    else:
        temperature["upper_bound"] = {
            "label": "kill fluid enters z_inj at the Ramey arrival temperature (maximum heating)",
            "t_inject_degf": t_bound, "q_k_min_bpm": q_reported,
            "evaluations": out["q_k_min"]["evaluations"], "seconds": out["q_k_min"]["seconds"],
            "reported": True,
        }
    q_bound = temperature["upper_bound"]["q_k_min_bpm"]

    if q_reported is not None and q_cold:
        # convention: relative differences divide by the smaller value
        temperature["gap_relative"] = abs(q_bound - q_cold) / min(q_bound, q_cold) \
            if q_bound else abs(q_reported - q_cold) / min(q_reported, q_cold)
        temperature["gap_bpm"] = (q_bound if q_bound else q_reported) - q_cold
        if estimating and q_bound:
            temperature["gap_surface_to_estimate"] = (q_reported - q_cold) / q_cold
            temperature["gap_estimate_to_bound"] = (q_bound - q_reported) / q_reported
            temperature["note"] = (
                f"three injection temperatures: {temperature['lower_bound']['t_inject_degf']:.1f} degF "
                f"at surface (no heating) gives {q_cold:.4f} bbl/min, the finite-U arrival "
                f"{t_reported:.1f} degF gives {q_reported:.4f}, and the U -> infinity bound "
                f"{t_bound:.1f} degF gives {q_bound:.4f}. THE MIDDLE ONE IS THE ESTIMATE AND IS WHAT "
                f"IS REPORTED; the outer two bracket it, {temperature['gap_surface_to_estimate']:.2%} "
                f"below and {temperature['gap_estimate_to_bound']:.2%} above, "
                f"{temperature['gap_relative']:.2%} end to end"
            )
        else:
            temperature["note"] = (
                f"the kill rate sits between {q_cold:.4f} bbl/min (kill fluid arriving at the surface "
                f"temperature {temperature['lower_bound']['t_inject_degf']:.1f} degF, no heating) and "
                f"{q_reported:.4f} bbl/min (arriving at the Ramey upper bound {t_bound:.1f} degF, maximum "
                f"heating), a {temperature['gap_relative']:.2%} gap. The reported rate is the upper-bound "
                "one, which is the conservative end"
            )
        # self-consistency: A goes as the mass rate, so the arrival temperature moves when
        # recomputed at the answer. Reported, not iterated (the same treatment as the
        # injection junction residual in M3-4).
        again = (ramey_arrival_finite_u(case, kill, q_reported) if estimating
                 else ramey_arrival(case, kill, q_reported))
        temperature["arrival_at_reported_rate_degf"] = again.t_arrival_degf
        temperature["arrival_residual_degf"] = again.t_arrival_degf - t_reported
        if abs(again.t_arrival_degf - t_reported) > 1.0:
            warnings.append(
                f"the arrival temperature moves {again.t_arrival_degf - t_reported:+.2f} degF when "
                f"recomputed at the reported kill rate ({t_reported:.2f} to {again.t_arrival_degf:.2f} "
                "degF): the arrival temperature and the kill rate are coupled and are not iterated"
            )
    if case.relief_well.geothermal_bound:
        t_geo = geothermal_at(case, temperature["lower_bound"]["t_inject_degf"],
                              case.well.md_to_tvd(md_inj))
        geo = min_kill_rate(case, pvt, ipr, _replace(kill, t_inject=t_geo), md_inj, liq_form,
                            q_start=q_hot)
        temperature["geothermal_bound"] = {
            "label": "kill fluid fully equilibrated with the formation at z_inj (the rw-1 upper bound)",
            "t_inject_degf": t_geo, "q_k_min_bpm": geo["q_k_min_bpm"],
            "evaluations": geo["evaluations"], "seconds": geo["seconds"],
        }


def geothermal_at(case: "Case", t_surface: float, tvd: float) -> float:
    """Geothermal temperature at ``tvd``, linear from ``t_surface`` at surface to the
    reservoir temperature at the datum: the same profile `core/inverse.geothermal` uses."""
    datum = case.reservoir.datum_tvd
    return t_surface + (case.reservoir.temperature - t_surface) * tvd / datum


def kill_sweep(case: "Case", pvt, ipr, rho_values, **kw) -> list[dict[str, Any]]:
    """q_k_min and the Blount-Soeiinah rate over kill-fluid densities (ppg); each row starts
    its bracket from the previous row's q_k_min (q_k_min falls with rho_k)."""
    from wellsim.cases.schema import Case

    rows = []
    q_prev = None
    for rho in rho_values:
        d = case.model_dump()
        d["dynamic_kill"]["rho_k"] = float(rho)
        d["dynamic_kill"]["rate_bpm"] = None
        c = Case(**d)
        a = kill_analysis(c, pvt, ipr, q_start=q_prev, **kw)
        qm = a["q_k_min"]["q_k_min_bpm"]
        rows.append({"rho_k_ppg": float(rho), "static_ok": a["static_hold"]["ok"], "margin_psi": a["static_hold"]["margin_psi"],
                     "blount_soeiinah_bpm": a["blount_soeiinah"]["q_bpm"], "q_k_min_bpm": qm,
                     "evaluations": a["q_k_min"]["evaluations"], "seconds": a["q_k_min"]["seconds"]})
        q_prev = qm if qm else q_prev
    return rows
