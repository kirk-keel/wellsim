"""rw-1 and rw-2: relief-well hydraulics for the dynamic kill. One relief well.

The balance
    At a kill rate q the pumps must supply

        p_pump = p_zinj - hydrostatic(string, rho_k) + friction(string)
                 + friction(surface lines) + restriction loss

    p_zinj is the blowing well's own pressure at z_inj at the kill condition, taken from
    ``kill.z_inj_state`` (one outflow evaluation at the converged rate and the residual
    minimiser), never recomputed from a separate model of the blowing well. Everything
    else is single-phase Newtonian kill fluid through the existing Colebrook routine.

    Hydrostatic uses the exact rho_k TVD / 144, the same basis as the Blount-Soeiinah
    column in ``kill._column_pressure``, not the 0.052 rule of thumb that
    ``kill.static_hold`` uses; the two differ by 0.1% and the hand-check test prints both
    so there is no ambiguity about which one a hand calculation should match. Surface
    lines are horizontal: they carry friction and no hydrostatic.

Restriction
    Exactly one, no default (``schema.ReliefRestriction``):
    - bit nozzles, dp = rho q^2 / (12031 TFA^2), rho ppg, q gpm, TFA in^2 (MEMORY);
    - an open-ended string into the blowing wellbore, dp = K rho v^2 / (2 g_c) at the
      string ID (COMPUTED form; K is the user's).

Free fall
    p_pump rises monotonically with q (hydrostatic does not depend on rate, friction and
    the restriction both do), so there is one rate at which the balance reaches
    atmospheric. Below it the string is on vacuum: the kill fluid free-falls and the
    surface pressure does not control the rate; the pumps are filling the string, not
    pushing it. A negative pump pressure is never reported as a number to size a pump
    against: the balance value is kept as a diagnostic and the reported pump pressure is
    zero gauge with that sentence attached.

Arrival temperature (rw-2)
    The kill fluid warms on the way down the relief string, and how warm it arrives at
    z_inj moves the kill rate by several percent. rw-2 computes it instead of bracketing
    it: Ramey (1962) for downward single-phase flow,

        dT/dL = (T_g - T) / A,   A = w c_p f(t) / (2 pi k_e)

    integrated along the string in MEASURED depth (heat transfer is per unit of pipe) with
    the formation temperature T_g = T_s + a TVD taken from TVD, so a deviated string works.
    Each segment is solved in closed form, so there is no integration error.

    It is a BOUND, NOT AN ESTIMATE. That A is the U -> infinity limit, which drops the
    pipe, annulus and cement resistance; the full A is w c_p [1/(2 pi r_to U) + f(t)/(2 pi
    k_e)], so dropping the first term makes A as small as it can be and the fluid relax
    toward the formation as fast as it can. For downward flow into hotter rock that is the
    most heating available, so the arrival temperature is the highest the fluid can reach.
    The surface temperature is the matching lower bound. Both kill rates are reported with
    the gap between them; the reported one is the upper-bound (hotter, higher-rate,
    conservative) run. Tag MEMORY for Ramey and for the Hasan-Kabir f(t).

    The arrival temperature and the kill rate are coupled (A goes as the mass rate). rw-2
    does not iterate them: it takes the arrival temperature at the lower-bound rate,
    searches once there, and reports how far the arrival temperature moves when recomputed
    at the answer, the same treatment the M3-4 injection junction gets.

    The blowing well stays adiabatic; rw-2 touched the relief string only.

Deferred: two or more relief wells (shared manifold, split rate, sequential
intersection); the kill search injects one mass rate at one z_inj. Mud rheology: the kill
fluid is Newtonian here as everywhere else in the kill path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from wellsim.core.exit import G_C
from wellsim.core.friction import darcy_friction_factor, reynolds_number
from wellsim.core.kill import LBM_PER_FT3_PER_PPG, PSI_PER_PPG_FT, KillFluid
from wellsim.core.numerics import brent

if TYPE_CHECKING:
    from wellsim.cases.schema import Case, ReliefWell

__all__ = [
    "Leg", "PumpPoint", "bit_nozzle_loss", "intersection_loss", "pump_pressure",
    "free_fall_rate", "pump_schedule", "relief_well_analysis",
    "GAL_PER_BBL", "BIT_LOSS_CONSTANT", "HHP_CONSTANT", "DEFAULT_SURFACE_T_DEGF",
    "ArrivalProfile", "dimensionless_time", "time_function", "relaxation_length",
    "kill_mass_rate_lbm_hr", "arrival_temperature", "ramey_arrival",
    "RAMEY_K_E_DEFAULT", "RAMEY_ALPHA_DEFAULT",
    "prandtl", "dittus_boelter", "overall_u", "relaxation_length_finite_u",
    "ramey_arrival_finite_u", "SegmentThermal",
    "RW3_K_PIPE_DEFAULT", "RW3_K_CEM_DEFAULT", "RW3_K_ANNULUS_DEFAULT", "RW3_K_FLUID_DEFAULT",
]

GAL_PER_BBL = 42.0
_FT3_PER_BBL = 5.614583
BIT_LOSS_CONSTANT = 12031.0  # dp = rho q^2 / (12031 TFA^2), rho ppg, q gpm, TFA in^2 (MEMORY)
HHP_CONSTANT = 1714.0  # HHP = p psi x q gpm / 1714
DEFAULT_SURFACE_T_DEGF = 60.0
P_ATM_PSIA = 14.7  # the pump side discharges to atmosphere; the free-fall threshold


def _area_ft2(d_in: float) -> float:
    return math.pi * 0.25 * (d_in / 12.0) ** 2


@dataclass
class Leg:
    """One friction leg of the injection path, for the report and the hand check."""
    name: str
    length_ft: float
    tvd_ft: float  # vertical drop over the leg; 0 for surface lines
    id_in: float
    v_ft_s: float
    re: float
    f: float
    dp_friction_psi: float
    dp_hydrostatic_psi: float

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "length_ft": self.length_ft, "tvd_ft": self.tvd_ft,
                "id_in": self.id_in, "v_ft_s": self.v_ft_s, "re": self.re, "f": self.f,
                "dp_friction_psi": self.dp_friction_psi, "dp_hydrostatic_psi": self.dp_hydrostatic_psi}


@dataclass
class PumpPoint:
    """The pump balance at one rate."""
    q_bpm: float
    q_gpm: float
    p_zinj_psia: float
    dp_hydrostatic_psi: float  # what the column gives back, positive
    dp_friction_string_psi: float
    dp_friction_surface_psi: float
    dp_restriction_psi: float
    restriction_kind: str
    p_pump_balance_psia: float  # the raw balance, which may be below atmospheric
    p_pump_psig: float  # reported: zero on vacuum, never negative
    hhp: float
    on_vacuum: bool
    legs: list[Leg] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def p_pump_psia(self) -> float:
        return self.p_pump_psig + P_ATM_PSIA

    @property
    def over_any_limit(self) -> bool:
        return any(v is True for v in self.limits.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "q_bpm": self.q_bpm, "q_gpm": self.q_gpm, "p_zinj_psia": self.p_zinj_psia,
            "dp_hydrostatic_psi": self.dp_hydrostatic_psi,
            "dp_friction_string_psi": self.dp_friction_string_psi,
            "dp_friction_surface_psi": self.dp_friction_surface_psi,
            "dp_restriction_psi": self.dp_restriction_psi, "restriction_kind": self.restriction_kind,
            "p_pump_balance_psia": self.p_pump_balance_psia, "p_pump_psig": self.p_pump_psig,
            "p_pump_psia": self.p_pump_psia, "hhp": self.hhp, "on_vacuum": self.on_vacuum,
            "legs": [leg.as_dict() for leg in self.legs], "limits": dict(self.limits),
            "warnings": list(self.warnings),
        }


# --- the two restriction forms --------------------------------------------------------------------


def bit_nozzle_loss(rho_ppg: float, q_gpm: float, tfa_in2: float) -> float:
    """Bit-nozzle pressure drop, psi: dp = rho q^2 / (12031 TFA^2).

    rho ppg, q gpm, TFA in^2. MEMORY: the standard oilfield form (the constant absorbs a
    nozzle discharge coefficient near 0.95); no primary source consulted in-session.
    """
    return rho_ppg * q_gpm * q_gpm / (BIT_LOSS_CONSTANT * tfa_in2 * tfa_in2)


def intersection_loss(k: float, rho_lbm_ft3: float, v_ft_s: float) -> float:
    """Loss at an open-ended string discharging into the blowing wellbore, psi:
    dp = K rho v^2 / (2 g_c) / 144, with v at the string ID. COMPUTED form; K is the
    user's number."""
    return k * rho_lbm_ft3 * v_ft_s * v_ft_s / (2.0 * G_C) / 144.0


# --- the balance ------------------------------------------------------------------------------------


def _legs(rw: "ReliefWell", kill: KillFluid, q_bpm: float) -> tuple[list[Leg], float, float, float]:
    """(legs, string friction psi, surface friction psi, hydrostatic psi)."""
    q_ft3_s = q_bpm * _FT3_PER_BBL / 60.0
    legs: list[Leg] = []
    fric_string = fric_surface = hydrostatic = 0.0
    for i, line in enumerate(rw.surface_lines):
        d_ft = line.id / 12.0
        area = _area_ft2(line.id)
        v = q_ft3_s / area
        re = reynolds_number(kill.rho, v, d_ft, kill.mu) if v > 0.0 else 0.0
        f = darcy_friction_factor(re, line.roughness / line.id) if re > 0.0 else 0.0
        dp = f * kill.rho * v * v / (2.0 * G_C * d_ft) / 144.0 * line.length
        fric_surface += dp
        legs.append(Leg(f"surface line {i + 1}", line.length, 0.0, line.id, v, re, f, dp, 0.0))
    prev_md = prev_tvd = 0.0
    for i, sec in enumerate(rw.string):
        length = sec.md - prev_md
        drop = sec.tvd - prev_tvd
        d_ft = sec.id / 12.0
        area = _area_ft2(sec.id)
        v = q_ft3_s / area
        re = reynolds_number(kill.rho, v, d_ft, kill.mu) if v > 0.0 else 0.0
        f = darcy_friction_factor(re, sec.roughness / sec.id) if re > 0.0 else 0.0
        dp = f * kill.rho * v * v / (2.0 * G_C * d_ft) / 144.0 * length
        dp_hyd = kill.rho * drop / 144.0
        fric_string += dp
        hydrostatic += dp_hyd
        legs.append(Leg(f"string {i + 1}", length, drop, sec.id, v, re, f, dp, dp_hyd))
        prev_md, prev_tvd = sec.md, sec.tvd
    return legs, fric_string, fric_surface, hydrostatic


def pump_pressure(rw: "ReliefWell", kill: KillFluid, q_bpm: float, p_zinj: float) -> PumpPoint:
    """The pump balance at ``q_bpm`` bbl/min with the blowing well at ``p_zinj`` psia at
    z_inj. Pure: nothing here touches the blowing-well model."""
    q_gpm = q_bpm * GAL_PER_BBL
    legs, fric_string, fric_surface, hydrostatic = _legs(rw, kill, q_bpm)
    r = rw.restriction
    if r.bit_tfa is not None:
        dp_restriction = bit_nozzle_loss(kill.rho_ppg, q_gpm, r.bit_tfa)
    else:
        d_in = rw.string[-1].id
        v = (q_bpm * _FT3_PER_BBL / 60.0) / _area_ft2(d_in)
        dp_restriction = intersection_loss(r.intersection_k, kill.rho, v)
    balance = p_zinj - hydrostatic + fric_string + fric_surface + dp_restriction
    on_vacuum = balance < P_ATM_PSIA
    p_psig = 0.0 if on_vacuum else balance - P_ATM_PSIA
    hhp = p_psig * q_gpm / HHP_CONSTANT
    warnings: list[str] = []
    if on_vacuum:
        warnings.append(
            f"string on vacuum: kill fluid free-falls; surface pressure does not control rate "
            f"(the balance at {q_bpm:.3f} bbl/min is {balance:.1f} psia, below atmospheric "
            f"{P_ATM_PSIA:g} psia, because the {hydrostatic:,.0f} psi kill column outruns the "
            f"{fric_string + fric_surface + dp_restriction:,.0f} psi of friction and restriction)"
        )
    limits: dict[str, Any] = {}
    if rw.pumps.max_pressure is not None:
        limits["over_pump_pressure"] = p_psig > rw.pumps.max_pressure
        limits["pump_pressure_limit_psi"] = rw.pumps.max_pressure
    if rw.pumps.total_hhp is not None:
        limits["over_hhp"] = hhp > rw.pumps.total_hhp
        limits["hhp_limit"] = rw.pumps.total_hhp
    if rw.ratings.surface_iron is not None:
        limits["over_surface_iron"] = p_psig > rw.ratings.surface_iron
        limits["surface_iron_psi"] = rw.ratings.surface_iron
    if rw.ratings.string_burst is not None:
        limits["over_string_burst"] = p_psig > rw.ratings.string_burst
        limits["string_burst_psi"] = rw.ratings.string_burst
    return PumpPoint(
        q_bpm=q_bpm, q_gpm=q_gpm, p_zinj_psia=p_zinj, dp_hydrostatic_psi=hydrostatic,
        dp_friction_string_psi=fric_string, dp_friction_surface_psi=fric_surface,
        dp_restriction_psi=dp_restriction, restriction_kind=r.kind,
        p_pump_balance_psia=balance, p_pump_psig=p_psig, hhp=hhp, on_vacuum=on_vacuum,
        legs=legs, limits=limits, warnings=warnings,
    )


def free_fall_rate(rw: "ReliefWell", kill: KillFluid, p_zinj: float, *,
                   q_hi: float = 1000.0, xtol: float = 1e-6) -> float | None:
    """The rate at which the balance reaches atmospheric: below it the string is on
    vacuum, above it the pumps control. None when the string is never on vacuum (the
    balance is already at or above atmospheric at zero rate), and None when no rate up to
    ``q_hi`` bbl/min gets there.

    p_pump rises monotonically with q, so Brent on p_pump(q) - 14.7 has one root.
    """
    def f(q: float) -> float:
        return pump_pressure(rw, kill, q, p_zinj).p_pump_balance_psia - P_ATM_PSIA

    f0 = f(0.0)
    if f0 >= 0.0:
        return None  # never on vacuum: the column alone does not reach atmospheric
    a, fa = 0.0, f0
    b = 1.0
    fb = f(b)
    while fb < 0.0:
        a, fa = b, fb
        b *= 2.0
        if b > q_hi:
            return None
        fb = f(b)
    return brent(f, a, b, fa=fa, fb=fb, xtol=xtol, rtol=0.0)


def pump_schedule(rw: "ReliefWell", kill: KillFluid, q_k_min: float, p_zinj: float,
                  *, steps: int = 10, top_factor: float = 1.5) -> list[PumpPoint]:
    """p_pump and HHP from q_k_min to ``top_factor`` q_k_min in ``steps`` rows.

    p_zinj is held at its kill-condition value across the schedule: above q_k_min the well
    is already dead and the blowing-well state that set p_zinj no longer applies, so the
    rows above the first are the hydraulics of pumping into a dead well at that pressure,
    not a re-solved kill. The report says so.
    """
    if steps < 2:
        raise ValueError("pump_schedule needs at least two steps")
    lo, hi = q_k_min, top_factor * q_k_min
    return [pump_pressure(rw, kill, lo + (hi - lo) * i / (steps - 1), p_zinj) for i in range(steps)]


# --- the analysis attached to a kill result ------------------------------------------------------


def relief_well_analysis(case: "Case", kill: KillFluid, q_k_min: float | None,
                         z_inj_state: dict[str, Any] | None) -> dict[str, Any]:
    """Pump pressure, HHP, free-fall check and the pump schedule at the kill condition."""
    rw = case.relief_well
    warnings: list[str] = []
    notes: list[str] = []
    out: dict[str, Any] = {
        "rho_k_ppg": kill.rho_ppg,
        "restriction": rw.restriction.kind,
        "surface_line_count": len(rw.surface_lines),
        "string_sections": len(rw.string),
        "string_bottom_md_ft": rw.bottom_md,
        "string_bottom_tvd_ft": rw.bottom_tvd,
        "warnings": warnings,
        "notes": notes,
    }
    if q_k_min is None or z_inj_state is None:
        warnings.append(
            "no kill rate or no evaluable state at z_inj: the pump balance needs the blowing well's "
            "pressure at z_inj at the kill condition, so no pump pressure is reported"
        )
        return out
    p_zinj = z_inj_state["p_psia"]
    at_kill = pump_pressure(rw, kill, q_k_min, p_zinj)
    out["p_zinj_psia"] = p_zinj
    out["t_zinj_degf"] = z_inj_state["t_degf"]
    out["z_inj_basis"] = z_inj_state["basis"]
    out["at_kill"] = at_kill.as_dict()
    warnings.extend(at_kill.warnings)
    q_ff = free_fall_rate(rw, kill, p_zinj)
    out["free_fall"] = {"on_vacuum_at_kill_rate": at_kill.on_vacuum, "q_free_fall_bpm": q_ff}
    if q_ff is not None:
        notes.append(
            f"free-fall rate {q_ff:.4f} bbl/min: the balance reaches atmospheric there, so below it the "
            f"string is on vacuum and above it the pumps control"
        )
    elif at_kill.on_vacuum:
        warnings.append("on vacuum at the kill rate and no rate up to 1,000 bbl/min brings the balance "
                        "back to atmospheric")
    else:
        notes.append("never on vacuum: the balance is at or above atmospheric even at zero rate")
    out["schedule"] = [p.as_dict() for p in
                       pump_schedule(rw, kill, q_k_min, p_zinj, steps=rw.schedule_steps)]
    notes.append(
        f"pump schedule from {q_k_min:.3f} to {1.5 * q_k_min:.3f} bbl/min in {rw.schedule_steps} rows at a "
        f"fixed p_zinj {p_zinj:,.1f} psia: above the kill rate the well is already dead, so the rows are the "
        "hydraulics of pumping into a well held at that pressure, not a re-solved kill"
    )
    for name, flag in (("pump pressure", "over_pump_pressure"), ("hydraulic horsepower", "over_hhp"),
                       ("surface iron rating", "over_surface_iron"), ("string burst rating", "over_string_burst")):
        if at_kill.limits.get(flag):
            warnings.append(f"{name} exceeded at the kill rate")
    over_rows = [i for i, p in enumerate(out["schedule"]) if any(v is True for v in p["limits"].values())]
    if over_rows and not at_kill.over_any_limit:
        warnings.append(
            f"the kill rate is inside every limit but {len(over_rows)} of {len(out['schedule'])} schedule "
            f"rows are not: the first breach is at {out['schedule'][over_rows[0]]['q_bpm']:.3f} bbl/min"
        )
    if rw.ratings.string_burst is not None:
        notes.append(
            "string burst is checked against the surface pressure only: the deepest joint's differential "
            "needs the external pressure profile on the relief well, which is not modelled"
        )
    if not rw.surface_lines:
        notes.append("no surface lines given: the pumps are taken to be at the wellhead")
    hyd_thumb = PSI_PER_PPG_FT * kill.rho_ppg * rw.bottom_tvd
    notes.append(
        f"hydrostatic {at_kill.dp_hydrostatic_psi:,.1f} psi on the exact rho TVD / 144 basis (the "
        f"Blount-Soeiinah column's basis); the 0.052 rule of thumb gives {hyd_thumb:,.1f} psi, "
        f"{abs(hyd_thumb - at_kill.dp_hydrostatic_psi) / at_kill.dp_hydrostatic_psi:.2%} different"
        if at_kill.dp_hydrostatic_psi > 0.0 else
        "zero-length string: no hydrostatic and no friction, so the balance is p_zinj plus the restriction loss"
    )
    return out


# --- rw-2: Ramey arrival temperature on the relief string -------------------------------------------

RAMEY_K_E_DEFAULT = 1.4  # Btu/(hr ft degF), formation thermal conductivity
RAMEY_ALPHA_DEFAULT = 0.04  # ft2/hr, formation thermal diffusivity
_MIN_PER_HR = 60.0
_SERIES_X = 1e-4  # below this, x + expm1(-x) is evaluated as a series


def dimensionless_time(alpha_ft2_hr: float, t_hr: float, r_w_ft: float) -> float:
    """t_D = alpha t / r_w^2, with r_w the RELIEF hole radius, not the blowing well's."""
    return alpha_ft2_hr * t_hr / (r_w_ft * r_w_ft)


def time_function(t_d: float) -> float:
    """Hasan-Kabir transient heat-conduction time function f(t_D).

        t_D <= 1.5:  f = 1.1281 sqrt(t_D) (1 - 0.3 sqrt(t_D))
        t_D >  1.5:  f = (0.4063 + 0.5 ln t_D) (1 + 0.6 / t_D)

    MEMORY, NEEDS HAND CHECK (docs/PROVENANCE.md). Both branches are written exactly as
    published, and they do NOT meet: at t_D = 1.5 the short-time branch gives 0.873990 and
    the long-time branch 0.852646, a jump of 0.021344, 2.50% of the smaller. That is pinned
    by the tests rather than closed by tuning a coefficient. The jump does not make the
    kill search jagged: t_D does not depend on rate, so f is a constant for a given case
    and the relaxation length varies only through the mass rate.
    """
    if t_d <= 0.0:
        return 0.0
    if t_d <= 1.5:
        s = math.sqrt(t_d)
        return 1.1281 * s * (1.0 - 0.3 * s)
    return (0.4063 + 0.5 * math.log(t_d)) * (1.0 + 0.6 / t_d)


def relaxation_length(w_lbm_hr: float, cp: float, f_t: float, k_e: float) -> float:
    """Ramey A = w c_p f(t) / (2 pi k_e), ft: the length over which the fluid relaxes toward
    the formation temperature. Large A is a fast, poorly coupled string that arrives near
    its inlet temperature; small A is a slow, well coupled one that arrives near the
    formation temperature."""
    return w_lbm_hr * cp * f_t / (2.0 * math.pi * k_e)


def kill_mass_rate_lbm_hr(q_bpm: float, rho_lbm_ft3: float) -> float:
    return q_bpm * _FT3_PER_BBL * rho_lbm_ft3 * _MIN_PER_HR


def _segment_outlet(t_in: float, t_g_in: float, b: float, length: float, a_relax: float) -> float:
    """Exact solution of dT/dL = (T_g - T)/A over one segment with T_g linear in L:

        T(u) = T0 + b u + (T0 - Tg0 + b A) (exp(-u/A) - 1)
             = T0 + (T0 - Tg0) expm1(-x) + b A (x + expm1(-x)),   x = u / A

    The second form is the one used, because x + expm1(-x) is x^2/2 - x^3/6 + ... and
    cancels to nothing for a long, weakly coupled string (A much larger than the length);
    below x = 1e-4 it is summed as that series instead of by subtraction.
    """
    if a_relax <= 0.0:
        return t_g_in + b * length  # perfectly coupled: the fluid sits at the formation temperature
    if not math.isfinite(a_relax):
        # rw-3: U -> 0, a perfectly insulated string. The limit is the inlet temperature;
        # taken explicitly because b * inf * 0 in the series form below is a nan.
        return t_in
    x = length / a_relax
    if x < _SERIES_X:
        corr = 0.5 * x * x * (1.0 - x / 3.0 + x * x / 12.0)
    else:
        corr = x + math.expm1(-x)
    return t_in + (t_in - t_g_in) * math.expm1(-x) + b * a_relax * corr


@dataclass
class ArrivalProfile:
    """The kill fluid temperature at the bottom of the relief string, and how it got there."""
    t_arrival_degf: float
    t_inlet_degf: float
    t_formation_at_bottom_degf: float
    a_relax_ft: float
    t_d: float
    f_t: float
    w_lbm_hr: float
    cp: float
    k_e: float
    alpha: float
    r_w_ft: float
    t_hr: float
    gradient_degf_per_ft: float
    t_surface_degf: float
    q_k_bpm: float = 0.0
    bound: str = "upper"
    rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # rw-3, only set on the finite-U estimate
    segments: list[dict[str, Any]] = field(default_factory=list)
    k_pipe: float | None = None
    k_cem: float | None = None
    k_fluid: float | None = None
    pr: float | None = None

    @property
    def heating_degf(self) -> float:
        return self.t_arrival_degf - self.t_inlet_degf

    @property
    def fraction_of_available_heating(self) -> float:
        """How far the fluid got from its inlet toward the formation temperature at z_inj."""
        span = self.t_formation_at_bottom_degf - self.t_inlet_degf
        return self.heating_degf / span if span != 0.0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "t_arrival_degf": self.t_arrival_degf, "t_inlet_degf": self.t_inlet_degf,
            "t_formation_at_bottom_degf": self.t_formation_at_bottom_degf,
            "heating_degf": self.heating_degf,
            "fraction_of_available_heating": self.fraction_of_available_heating,
            "a_relax_ft": self.a_relax_ft, "t_d": self.t_d, "f_t": self.f_t,
            "w_lbm_hr": self.w_lbm_hr, "cp_btu_lbm_degf": self.cp, "k_e": self.k_e,
            "alpha_ft2_hr": self.alpha, "r_w_ft": self.r_w_ft, "t_hr": self.t_hr,
            "gradient_degf_per_ft": self.gradient_degf_per_ft,
            "t_surface_degf": self.t_surface_degf, "q_k_bpm": self.q_k_bpm, "bound": self.bound,
            "rows": list(self.rows), "warnings": list(self.warnings),
            "segments": list(self.segments), "k_pipe": self.k_pipe, "k_cem": self.k_cem,
            "k_fluid": self.k_fluid, "pr": self.pr,
        }


def arrival_temperature(stations: Sequence[tuple[float, float]], t_inlet: float, t_surface: float,
                        gradient: float,
                        a_relax: float | Sequence[float]) -> tuple[float, list[dict[str, Any]]]:
    """Arrival temperature degF and the per-segment rows, for downward flow down the string.

    ``stations`` are the string (md, tvd) bottom stations from (0, 0), the same pair the
    well survey uses, so a deviated string works: the ODE is integrated in MEASURED depth,
    because the heat transfer is per unit of pipe, while the formation temperature comes
    from TVD, T_g = t_surface + gradient x TVD. Each segment is solved in closed form, so
    the result carries no integration error.
    """
    t = t_inlet
    md_prev = tvd_prev = 0.0
    rows: list[dict[str, Any]] = []
    # rw-3: A may differ from section to section (a cemented interval and an open-hole one
    # have different annular resistance), so a sequence is accepted alongside the scalar
    per_segment = not isinstance(a_relax, (int, float))
    if per_segment and len(a_relax) != len(stations):
        raise ValueError(
            f"arrival_temperature: {len(a_relax)} relaxation lengths for {len(stations)} stations"
        )
    for i, (md, tvd) in enumerate(stations):
        length = md - md_prev
        if length <= 0.0:
            continue
        a_seg = a_relax[i] if per_segment else a_relax
        t_g_in = t_surface + gradient * tvd_prev
        t_g_out = t_surface + gradient * tvd
        b = (t_g_out - t_g_in) / length  # degF per ft of MD along this segment
        t_out = _segment_outlet(t, t_g_in, b, length, a_seg)
        rows.append({
            "md_top_ft": md_prev, "md_bottom_ft": md, "tvd_top_ft": tvd_prev, "tvd_bottom_ft": tvd,
            "length_ft": length, "t_in_degf": t, "t_out_degf": t_out, "a_relax_ft": a_seg,
            "t_formation_top_degf": t_g_in, "t_formation_bottom_degf": t_g_out,
        })
        t = t_out
        md_prev, tvd_prev = md, tvd
    return t, rows


def ramey_arrival(case: "Case", kill: KillFluid, q_k_bpm: float) -> ArrivalProfile:
    """The Ramey upper bound on the kill fluid arrival temperature at z_inj, at ``q_k_bpm``.

    A BOUND, NOT AN ESTIMATE. A = w c_p f(t) / (2 pi k_e) is the U -> infinity limit of the
    Ramey relaxation length, which drops the pipe, annulus and cement resistance. The full
    form is A = w c_p [1 / (2 pi r_to U) + f(t) / (2 pi k_e)], so dropping the first term
    makes A as small as it can be, and a smaller A means the fluid relaxes toward the
    formation faster. For downward flow into hotter rock that is the most heating physically
    available, so the arrival temperature is the highest the fluid can reach. The surface
    temperature is the matching lower bound, the fluid arriving with no heating at all.
    """
    rw = case.relief_well
    warnings: list[str] = []
    k_e = rw.k_e
    if k_e is None:
        k_e = RAMEY_K_E_DEFAULT
        warnings.append(
            f"relief_well.k_e not supplied; formation conductivity defaulted to {RAMEY_K_E_DEFAULT:g} "
            "Btu/(hr ft degF). A goes as 1/k_e, so this moves the arrival temperature"
        )
    alpha = rw.alpha
    if alpha is None:
        alpha = RAMEY_ALPHA_DEFAULT
        warnings.append(
            f"relief_well.alpha not supplied; formation diffusivity defaulted to {RAMEY_ALPHA_DEFAULT:g} "
            "ft2/hr; it enters only through t_D = alpha t / r_w^2, and so through f(t)"
        )
    t_surface = rw.surface_temperature
    if t_surface is None:
        t_surface = DEFAULT_SURFACE_T_DEGF
        warnings.append(
            f"relief_well.surface_temperature not supplied; the kill fluid is taken to enter the string "
            f"at {DEFAULT_SURFACE_T_DEGF:g} degF, which is also the lower bound on arrival"
        )
    t_hr = rw.pumping_time_min / _MIN_PER_HR
    t_d = dimensionless_time(alpha, t_hr, rw.hole_radius)
    f_t = time_function(t_d)
    cp = case.dynamic_kill.cp_k
    w = kill_mass_rate_lbm_hr(q_k_bpm, kill.rho)
    a_relax = relaxation_length(w, cp, f_t, k_e)
    gradient = (case.reservoir.temperature - t_surface) / case.reservoir.datum_tvd
    stations = [(s.md, s.tvd) for s in rw.string]
    t_arr, rows = arrival_temperature(stations, t_surface, t_surface, gradient, a_relax)
    t_form = t_surface + gradient * rw.bottom_tvd
    if abs(t_d - 1.5) < 0.15:
        warnings.append(
            f"t_D = {t_d:.4f} is within 10% of the t_D = 1.5 switch in f(t), where the two published "
            f"Hasan-Kabir branches disagree by 2.50%; f = {f_t:.6f} here"
        )
    return ArrivalProfile(
        t_arrival_degf=t_arr, t_inlet_degf=t_surface, t_formation_at_bottom_degf=t_form,
        a_relax_ft=a_relax, t_d=t_d, f_t=f_t, w_lbm_hr=w, cp=cp, k_e=k_e, alpha=alpha,
        r_w_ft=rw.hole_radius, t_hr=t_hr, gradient_degf_per_ft=gradient, t_surface_degf=t_surface,
        q_k_bpm=q_k_bpm, rows=rows, warnings=warnings,
    )


# --- rw-3: finite overall heat transfer coefficient ---------------------------------------------

RW3_K_PIPE_DEFAULT = 25.0  # Btu/(hr ft degF), steel
RW3_K_ANNULUS_DEFAULT = {"mud": 0.7, "cement": 1.0}  # Btu/(hr ft degF), by annulus_fill
RW3_K_CEM_DEFAULT = RW3_K_ANNULUS_DEFAULT["mud"]  # rw-3 name, kept for the deprecated alias
RW3_K_FLUID_DEFAULT = 0.4  # Btu/(hr ft degF), water-based kill fluid
_LBM_FT_HR_PER_CP = 2.4190883  # 1 cp = 6.719689e-4 lbm/(ft s) x 3600


def prandtl(cp: float, mu_cp: float, k_fluid: float) -> float:
    """Pr = c_p mu / k, dimensionless. c_p Btu/(lbm degF), mu cp, k Btu/(hr ft degF);
    the viscosity is converted to lbm/(ft hr) so the group is dimensionless."""
    return cp * mu_cp * _LBM_FT_HR_PER_CP / k_fluid


def dittus_boelter(re: float, pr: float, d_i_ft: float, k_fluid: float) -> float:
    """Inside film coefficient h_i, Btu/(hr ft2 degF), from Nu = 0.023 Re^0.8 Pr^0.4.

    The 0.4 exponent is the HEATING form, which is the right one here: the kill fluid is
    colder than the pipe. MEMORY, NEEDS HAND CHECK. Dittus-Boelter is a turbulent
    correlation (Re above about 10,000); a kill string at kill rates is far above that,
    and the result warns if it is not.
    """
    nu = 0.023 * re ** 0.8 * pr ** 0.4
    return nu * k_fluid / d_i_ft


def overall_u(r_ti: float, r_to: float, r_w: float, h_i: float, k_pipe: float,
              k_annulus: float) -> tuple[float, dict[str, float]]:
    """Overall heat transfer coefficient U, Btu/(hr ft2 degF), referred to the string OD,
    from three resistances in series per unit length:

        1/U = r_to/(r_ti h_i) + r_to ln(r_to/r_ti)/k_pipe + r_to ln(r_w/r_to)/k_annulus

    the inside film, the pipe wall and the annulus. The annulus is whatever stands in it:
    on a dynamic kill that is STATIC DRILLING FLUID, not cement and not circulating mud,
    because the kill fluid goes down the string and out at the intersection and nothing
    returns. An uncemented interval uses the formation conductivity over the same annulus.
    MEMORY, NEEDS HAND CHECK. Returns U and the three resistances, so the hand check can
    see which one dominates.
    """
    r_film = r_to / (r_ti * h_i) if h_i > 0.0 else math.inf
    r_pipe = r_to * math.log(r_to / r_ti) / k_pipe if r_to > r_ti else 0.0
    r_annulus = r_to * math.log(r_w / r_to) / k_annulus if r_w > r_to else 0.0
    total = r_film + r_pipe + r_annulus
    u = 1.0 / total if total > 0.0 else math.inf
    return u, {"film": r_film, "pipe": r_pipe, "annulus": r_annulus, "total": total}


def relaxation_length_finite_u(w_lbm_hr: float, cp: float, f_t: float, k_e: float,
                               r_to: float, u: float) -> float:
    """Ramey A with a finite U, ft:

        A = w c_p (k_e + r_to U f(t)) / (2 pi r_to U k_e)
          = w c_p [ 1/(2 pi r_to U) + f(t)/(2 pi k_e) ]

    As U -> infinity the first term vanishes and this is exactly
    :func:`relaxation_length`, the rw-2 bound. As U -> 0 it diverges, so the fluid arrives
    at its inlet temperature: no coupling at all. MEMORY (the Ramey form).
    """
    if not math.isfinite(u):
        return relaxation_length(w_lbm_hr, cp, f_t, k_e)
    if u <= 0.0:
        return math.inf
    return w_lbm_hr * cp * (k_e + r_to * u * f_t) / (2.0 * math.pi * r_to * u * k_e)


@dataclass
class SegmentThermal:
    """The finite-U picture over one string section, for the report and the hand check."""
    index: int
    md_top_ft: float
    md_bottom_ft: float
    cemented: bool
    d_i_in: float
    d_o_in: float
    r_ti_ft: float
    r_to_ft: float
    r_w_ft: float
    v_ft_s: float
    re: float
    pr: float
    h_i: float
    k_annulus: float
    annulus_is_formation: bool
    resistances: dict[str, float]
    u: float
    a_relax_ft: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "md_top_ft": self.md_top_ft, "md_bottom_ft": self.md_bottom_ft,
            "cemented": self.cemented, "d_i_in": self.d_i_in, "d_o_in": self.d_o_in,
            "r_ti_ft": self.r_ti_ft, "r_to_ft": self.r_to_ft, "r_w_ft": self.r_w_ft,
            "v_ft_s": self.v_ft_s, "re": self.re, "pr": self.pr, "h_i": self.h_i,
            "k_annulus": self.k_annulus, "annulus_is_formation": self.annulus_is_formation,
            "resistances": dict(self.resistances), "u": self.u, "a_relax_ft": self.a_relax_ft,
        }


def ramey_arrival_finite_u(case: "Case", kill: KillFluid, q_k_bpm: float) -> ArrivalProfile:
    """The kill fluid arrival temperature at z_inj with a FINITE U: an estimate, not a bound.

    Same Ramey integration as :func:`ramey_arrival`, but A carries the string's own
    resistance as well as the formation's. U is built per section from the inside film
    (Dittus-Boelter), the pipe wall and the annulus, so a section that is open hole rather
    than cemented gets the formation conductivity over the same annulus and the result
    says so. A therefore varies along the string and is passed per segment.

    The estimate is bracketed by the surface temperature below (no heating) and the rw-2
    U -> infinity bound above (all the heating the formation can supply).
    """
    rw = case.relief_well
    warnings: list[str] = []
    k_e = rw.k_e
    if k_e is None:
        k_e = RAMEY_K_E_DEFAULT
        warnings.append(f"relief_well.k_e not supplied; formation conductivity defaulted to "
                        f"{RAMEY_K_E_DEFAULT:g} Btu/(hr ft degF)")
    alpha = rw.alpha
    if alpha is None:
        alpha = RAMEY_ALPHA_DEFAULT
        warnings.append(f"relief_well.alpha not supplied; formation diffusivity defaulted to "
                        f"{RAMEY_ALPHA_DEFAULT:g} ft2/hr")
    k_pipe = rw.k_pipe
    if k_pipe is None:
        k_pipe = RW3_K_PIPE_DEFAULT
        warnings.append(f"relief_well.k_pipe not supplied; pipe conductivity defaulted to "
                        f"{RW3_K_PIPE_DEFAULT:g} Btu/(hr ft degF) (steel)")
    # rw-3b: the annulus fill names the default. On a dynamic kill the relief annulus holds
    # STATIC drilling fluid, not cement or circulating mud: the kill fluid goes down the
    # string and out at the intersection, so nothing returns up the annulus.
    k_cem = rw.k_annulus
    if k_cem is None and rw.k_cem is not None:
        k_cem = rw.k_cem
        warnings.append(
            "relief_well.k_cem is a deprecated alias for k_annulus (rw-3b) and was used for the "
            f"annular conductivity ({k_cem:g} Btu/(hr ft degF)); give k_annulus instead"
        )
    if k_cem is None:
        k_cem = RW3_K_ANNULUS_DEFAULT[rw.annulus_fill]
        warnings.append(
            f"relief_well.k_annulus not supplied; annulus_fill '{rw.annulus_fill}' defaults it to "
            f"{k_cem:g} Btu/(hr ft degF)"
        )
    k_fluid = case.dynamic_kill.k_f
    if k_fluid is None:
        k_fluid = RW3_K_FLUID_DEFAULT
        warnings.append(f"dynamic_kill.k_f not supplied; kill-fluid conductivity defaulted to "
                        f"{RW3_K_FLUID_DEFAULT:g} Btu/(hr ft degF) (water-based)")
    t_surface = rw.surface_temperature
    if t_surface is None:
        t_surface = DEFAULT_SURFACE_T_DEGF
        warnings.append(f"relief_well.surface_temperature not supplied; the kill fluid enters at "
                        f"{DEFAULT_SURFACE_T_DEGF:g} degF")

    t_hr = rw.pumping_time_min / _MIN_PER_HR
    t_d = dimensionless_time(alpha, t_hr, rw.hole_radius)
    f_t = time_function(t_d)
    cp = case.dynamic_kill.cp_k
    w = kill_mass_rate_lbm_hr(q_k_bpm, kill.rho)
    q_ft3_s = q_k_bpm * _FT3_PER_BBL / 60.0
    pr = prandtl(cp, kill.mu, k_fluid)

    segments: list[SegmentThermal] = []
    a_per_segment: list[float] = []
    md_prev = 0.0
    no_wall = False
    open_hole_md = 0.0
    for i, sec in enumerate(rw.string):
        d_o = sec.od if sec.od is not None else rw.string_od
        if d_o is None:
            d_o = sec.id
            no_wall = True
        r_ti = sec.id / 24.0  # inches diameter -> feet radius
        r_to = d_o / 24.0
        d_i_ft = sec.id / 12.0
        v = q_ft3_s / _area_ft2(sec.id)
        re = reynolds_number(kill.rho, v, d_i_ft, kill.mu) if v > 0.0 else 0.0
        h_i = dittus_boelter(re, pr, d_i_ft, k_fluid) if re > 0.0 else 0.0
        k_annulus = k_cem if sec.cemented else k_e
        if not sec.cemented:
            open_hole_md += sec.md - md_prev
        u, res = overall_u(r_ti, r_to, rw.hole_radius, h_i, k_pipe, k_annulus)
        a_seg = relaxation_length_finite_u(w, cp, f_t, k_e, r_to, u)
        segments.append(SegmentThermal(
            index=i, md_top_ft=md_prev, md_bottom_ft=sec.md, cemented=sec.cemented,
            d_i_in=sec.id, d_o_in=d_o, r_ti_ft=r_ti, r_to_ft=r_to, r_w_ft=rw.hole_radius,
            v_ft_s=v, re=re, pr=pr, h_i=h_i, k_annulus=k_annulus,
            annulus_is_formation=not sec.cemented, resistances=res, u=u, a_relax_ft=a_seg,
        ))
        a_per_segment.append(a_seg)
        md_prev = sec.md
        if 0.0 < re < 10000.0:
            warnings.append(
                f"relief string section {i + 1}: Re = {re:,.0f} is below the 10,000 Dittus-Boelter "
                "is stated for; the inside film coefficient is an extrapolation"
            )
    if no_wall:
        warnings.append(
            "no relief-string OD given (neither section od nor relief_well.string_od): the OD is "
            "taken equal to the ID, so the pipe-wall resistance is zero and the finite-U estimate "
            "is warmer than it should be"
        )
    if open_hole_md > 0.0:
        warnings.append(
            f"{open_hole_md:,.0f} ft of the relief string is uncemented: over that interval the "
            f"annular resistance uses the formation conductivity k_e = {k_e:g} instead of "
            f"k_annulus = {k_cem:g} Btu/(hr ft degF)"
        )

    gradient = (case.reservoir.temperature - t_surface) / case.reservoir.datum_tvd
    stations = [(s.md, s.tvd) for s in rw.string]
    t_arr, rows = arrival_temperature(stations, t_surface, t_surface, gradient, a_per_segment)
    t_form = t_surface + gradient * rw.bottom_tvd
    a_mean = sum(a_per_segment) / len(a_per_segment) if a_per_segment else math.inf
    prof = ArrivalProfile(
        t_arrival_degf=t_arr, t_inlet_degf=t_surface, t_formation_at_bottom_degf=t_form,
        a_relax_ft=a_mean, t_d=t_d, f_t=f_t, w_lbm_hr=w, cp=cp, k_e=k_e, alpha=alpha,
        r_w_ft=rw.hole_radius, t_hr=t_hr, gradient_degf_per_ft=gradient, t_surface_degf=t_surface,
        q_k_bpm=q_k_bpm, bound="estimate", rows=rows, warnings=warnings,
    )
    prof.segments = [s.as_dict() for s in segments]
    prof.k_pipe = k_pipe
    prof.k_cem = k_cem
    prof.k_fluid = k_fluid
    prof.pr = pr
    return prof
