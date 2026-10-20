"""Inverse mode: effective kh from an observed rate, p_r from a shut-in wellhead pressure.

Pulled ahead of M4 (docs/TODO-M4.md, "Now"): M4 is gated on hand checks and inverse
mode is a prerequisite for Monte Carlo on blowout rate.

Rate inverse (``invert_kh``)
    Find kh such that the forward nodal operating point (``solver.solve``) equals the
    observed gas rate. h is fixed from the case; k = kh/h is written into the case copy,
    so beta and the non-Darcy D read it exactly as the forward solve does (Jones beta
    goes as k^-1.55, D as k^-0.55 h/h_p^2). The search is Brent in ln(kh) on
    q_g(kh)/q_obs - 1 over [1e-2, 1e6] md-ft, stopping when the residual is within 0.5%
    of q_obs and the ln(kh) bracket is within 5e-3 (kh within 0.5% of the root). When
    the forward solve is not evaluable at the top of the bracket the top steps down a
    decade at a time and the result says so. An observed rate above the forward rate at
    the top of the bracket is an exit-limited well (no kh reproduces it); below the
    forward rate at the bottom, the bracket does not reach it. Both raise InverseError
    with that sentence. With bounds q_g_low / q_g_high all three are inverted.

    Sensitivity at the solution: dq_g/d(ln kh) by central difference with a 0.1 step in
    ln kh, and the linearised kh range for q_g +/- 20%,
    ln kh +/- 0.2 q_g / (dq_g/d ln kh). This is the input Monte Carlo on kh needs.

    The result carries the sentence "effective kh: absorbs skin, net pay and geometry
    error; not a reservoir property." An inverted kh is a lumped deliverability
    parameter of this model with these fixed inputs.

SIWHP inverse (``invert_p_res``)
    p_r at the datum TVD from the shut-in wellhead pressure through a static real-gas
    column: dp/dz = rho_g(p, T(z))/144 psi/ft integrated in TVD with RK4 (50 ft steps),
    the existing DAK PVT, and a geothermal temperature linear in TVD from the surface
    temperature in the observation to the reservoir temperature at the datum. The
    column is assumed gas-filled and in thermal equilibrium; a case with a liquid block
    is refused (a liquid leg in the shut-in column is not modelled).

Every result is an :class:`InverseResult` with the value, the observation it came
from, the fixed inputs it depends on, the statement above and a case copy with the
inverted value set, so downstream solves (kill, sweeps) run unchanged. All COMPUTED:
the tests are round trips, monotone sweeps and internal consistency; no published
inverse case is wired in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from wellsim.cases.schema import Case
from wellsim.core.numerics import brent
from wellsim.core.pvt import collect_range_warnings, make_pvt

__all__ = [
    "InverseError", "InverseResult", "invert", "invert_kh", "invert_p_res",
    "static_gas_column", "forward_rate", "with_kh",
    "KH_LO", "KH_HI", "REL_TOL_Q", "XTOL_LN_KH", "SENSITIVITY_STEP_LN_KH",
]

KH_LO = 1e-2  # md-ft, bottom of the search bracket
KH_HI = 1e6  # md-ft, top of the search bracket (stepped down a decade at a time if not evaluable)
REL_TOL_Q = 5e-3  # stop when |q_g(kh)/q_obs - 1| <= this
XTOL_LN_KH = 1.25e-3  # Brent returns within 4 xtol of the root in ln kh: kh within 0.5%
SENSITIVITY_STEP_LN_KH = 0.1  # central-difference step for dq_g/d(ln kh)
STATIC_COLUMN_STEP_FT = 50.0

KH_STATEMENT = "effective kh: absorbs skin, net pay and geometry error; not a reservoir property."
P_RES_STATEMENT = (
    "p_r from a static real-gas column: assumes a gas-filled wellbore in thermal equilibrium with the "
    "geothermal profile at the time of the reading; afterflow, wellbore storage and liquid legs are not "
    "modelled; not a stabilised reservoir pressure unless the shut-in was."
)


class InverseError(ValueError):
    """The observation cannot be inverted with this case (message says why)."""


@dataclass
class InverseResult:
    kind: str  # "kh" or "p_res"
    value: float
    unit: str
    observation: dict[str, Any]
    fixed_inputs: dict[str, Any]
    statement: str
    case: Case  # copy of the input case with the inverted value set
    tolerance: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, Any] | None = None
    bounds: dict[str, Any] | None = None
    evaluations: list[tuple[float, float]] = field(default_factory=list)  # (kh md-ft, q_g Mscf/d), search order
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    range_warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "value": self.value, "unit": self.unit, "observation": dict(self.observation),
            "fixed_inputs": dict(self.fixed_inputs), "statement": self.statement, "tolerance": dict(self.tolerance),
            "sensitivity": None if self.sensitivity is None else dict(self.sensitivity),
            "bounds": None if self.bounds is None else dict(self.bounds),
            "evaluations": [list(e) for e in self.evaluations], "notes": list(self.notes),
            "warnings": list(self.warnings), "range_warnings": list(self.range_warnings), "elapsed_s": self.elapsed_s,
        }


# --- helpers ------------------------------------------------------------------------------------------

def with_kh(case: Case, kh: float) -> Case:
    """Copy of ``case`` with reservoir.permeability = kh / net_pay (h fixed)."""
    d = case.model_dump()
    d["reservoir"]["permeability"] = kh / case.reservoir.net_pay
    return Case(**d)


def forward_rate(case: Case) -> float:
    """Operating gas rate of the forward solve, Mscf/d; 0 when the well does not flow."""
    from wellsim.core.solver import solve

    r = solve(case)
    return float(r.gas_rate) if r.flowing else 0.0


def _rate_case(case: Case, notes: list[str]) -> Case:
    """The case the rate inverse searches on: q_l attributed to the WGR when given."""
    obs = case.observed.rate
    if obs.q_l is None:
        return case
    li = case.liquid_inflow
    if li.mode != "wgr":
        raise InverseError(
            "observed.rate.q_l with liquid_inflow mode 'pi': the productivity index is coupled to the unknown "
            "pwf and is not inverted; give the liquid as a WGR or drop q_l"
        )
    wgr = 1000.0 * obs.q_l / obs.q_g
    d = case.model_dump()
    d["liquid_inflow"]["water_gas_ratio"] = wgr
    notes.append(
        f"observed q_l {obs.q_l:g} stb/d with q_g {obs.q_g:g} Mscf/d sets the WGR to {wgr:.4g} bbl/MMscf "
        f"(case value {li.water_gas_ratio:g} replaced)"
    )
    return Case(**d)


def _fixed_inputs_rate(case: Case) -> dict[str, Any]:
    r = case.reservoir
    li = case.liquid_inflow
    return {
        "h_ft": r.net_pay,
        "skin": r.skin,
        "h_p_ft": r.exposed_interval if r.exposed_interval is not None else r.net_pay,
        "h_1_ft": r.exposed_top_offset,
        "kv_kh": r.kv_kh if r.kv_kh is not None else "n/a (full penetration)" if r.exposed_interval is None else "0.1 defaulted (warned by the forward solve)",
        "p_res_psia": r.pressure,
        "t_res_degf": r.temperature,
        "r_w_ft": r.wellbore_radius,
        "r_e_ft": r.drainage_radius,
        "non_darcy_beta": r.non_darcy_beta if r.non_darcy_beta is not None else case.options.correlations.non_darcy_beta,
        "porosity": r.porosity,
        "gas_sg": case.fluid.gas_sg,
        "ambient_pressure_psia": case.exit.ambient_pressure,
        "exit_id_in": case.exit.exit_id,
        "flow_path": case.flow_path,
        "liquid_inflow": None if li is None else li.model_dump(),
        "prior_permeability_md": r.permeability,
    }


# --- rate inverse ---------------------------------------------------------------------------------------

def invert_kh(case: Case, *, kh_lo: float = KH_LO, kh_hi: float = KH_HI, rel_tol: float = REL_TOL_Q,
              sensitivity: bool = True) -> InverseResult:
    """kh (md-ft) such that the forward operating rate equals ``case.observed.rate.q_g``."""
    t0 = time.perf_counter()
    if case.observed is None or case.observed.rate is None:
        raise InverseError("invert_kh needs case.observed.rate")
    obs = case.observed.rate
    notes: list[str] = []
    warnings: list[str] = []
    base = _rate_case(case, notes)
    cache: dict[float, float] = {}
    evaluations: list[tuple[float, float]] = []

    def q_of(kh: float) -> float:
        if kh not in cache:
            cache[kh] = forward_rate(with_kh(base, kh))
            evaluations.append((kh, cache[kh]))
        return cache[kh]

    with collect_range_warnings() as range_warnings:
        # top of the bracket: step down a decade at a time while the forward solve is not evaluable
        top = kh_hi
        while True:
            try:
                q_top = q_of(top)
                break
            except Exception as e:  # the forward solve refused this state; say so and step down
                if top / 10.0 <= kh_lo:
                    raise InverseError(f"forward solve not evaluable anywhere in the bracket top range: {e}") from e
                notes.append(f"bracket top {top:g} md-ft not evaluable ({type(e).__name__}: {str(e)[:120]}); stepped down")
                top /= 10.0
        q_bot = q_of(kh_lo)
        if top != kh_hi:
            notes.append(f"bracket top used: {top:g} md-ft (forward rate {q_top:.6g} Mscf/d)")

        def search(q_target: float, label: str) -> tuple[float, float]:
            if q_target > q_top:
                raise InverseError(
                    f"{label} {q_target:.6g} Mscf/d exceeds the forward rate {q_top:.6g} Mscf/d at the top of the "
                    f"bracket (kh = {top:g} md-ft): the well is exit-limited (the wellbore, not the reservoir, sets "
                    "the rate), so no kh reproduces the observation"
                )
            if q_target < q_bot:
                raise InverseError(
                    f"{label} {q_target:.6g} Mscf/d falls below the forward rate {q_bot:.6g} Mscf/d at the bottom of "
                    f"the bracket (kh = {kh_lo:g} md-ft): the bracket does not reach it"
                )

            def f(x: float) -> float:
                return q_of(math.exp(x)) / q_target - 1.0

            a, b = math.log(kh_lo), math.log(top)
            fa, fb = q_bot / q_target - 1.0, q_top / q_target - 1.0
            x = brent(f, a, b, fa=fa, fb=fb, xtol=XTOL_LN_KH, rtol=0.0, ftol=rel_tol)
            kh = math.exp(x)
            q_at = q_of(kh)
            if abs(q_at / q_target - 1.0) > rel_tol:
                raise InverseError(
                    f"{label}: the search stopped at kh {kh:.6g} md-ft with q_g {q_at:.6g} Mscf/d, "
                    f"{abs(q_at / q_target - 1.0):.3%} from the target (tolerance {rel_tol:.2%})"
                )
            return kh, q_at

        kh, q_at = search(obs.q_g, "observed rate")
        bounds = None
        if obs.q_g_low is not None:
            kh_low, q_low_at = search(obs.q_g_low, "observed rate lower bound")
            kh_high, q_high_at = search(obs.q_g_high, "observed rate upper bound")
            bounds = {
                "kh_low_md_ft": kh_low, "kh_md_ft": kh, "kh_high_md_ft": kh_high,
                "q_g_low_mscfd": obs.q_g_low, "q_g_high_mscfd": obs.q_g_high,
                "q_g_at_kh_low": q_low_at, "q_g_at_kh_high": q_high_at,
            }
        sens = None
        if sensitivity:
            hx = SENSITIVITY_STEP_LN_KH
            q_plus = q_of(kh * math.exp(hx))
            q_minus = q_of(kh * math.exp(-hx))
            dq_dlnkh = (q_plus - q_minus) / (2.0 * hx)
            sens = {
                "dq_g_dln_kh_mscfd": dq_dlnkh,
                "elasticity_dlnq_dlnkh": dq_dlnkh / q_at if q_at > 0.0 else None,
                "step_ln_kh": hx,
                "q_g_at_kh_times_e_plus": q_plus,
                "q_g_at_kh_times_e_minus": q_minus,
            }
            if dq_dlnkh > 0.0:
                dx = 0.2 * q_at / dq_dlnkh
                sens.update({
                    "kh_for_q_minus_20pct_linearised": kh * math.exp(-dx),
                    "kh_for_q_plus_20pct_linearised": kh * math.exp(dx),
                })
            else:
                warnings.append(
                    f"dq_g/d(ln kh) = {dq_dlnkh:.4g} Mscf/d is not positive at the solution: the rate is insensitive "
                    "to kh here (exit- or wellbore-limited); the linearised kh range for q_g +/- 20% is not reported"
                )

    fixed = _fixed_inputs_rate(base)
    result_case = with_kh(base, kh)
    k = kh / base.reservoir.net_pay
    notes.append(f"kh {kh:.6g} md-ft with h {base.reservoir.net_pay:g} ft fixed: k = {k:.6g} md written to the case copy "
                 f"(prior {base.reservoir.permeability:g} md replaced); beta and D follow from k as in the forward solve")
    return InverseResult(
        kind="kh", value=kh, unit="md-ft",
        observation={"q_g_mscfd": obs.q_g, "q_l_stbd": obs.q_l, "q_g_low_mscfd": obs.q_g_low,
                     "q_g_high_mscfd": obs.q_g_high, "q_g_at_solution_mscfd": q_at},
        fixed_inputs=fixed, statement=KH_STATEMENT, case=result_case,
        tolerance={"rel_q": rel_tol, "ln_kh_bracket": 4.0 * XTOL_LN_KH, "kh_lo": kh_lo, "kh_hi": top,
                   "rule": "searched result: pin at 0.5% (the stopping rule), never tighter"},
        sensitivity=sens, bounds=bounds, evaluations=evaluations, notes=notes, warnings=warnings,
        range_warnings=[str(w) for w in range_warnings], elapsed_s=time.perf_counter() - t0,
    )


# --- static gas column and the SIWHP inverse -------------------------------------------------------------

def static_gas_column(pvt, p_start: float, tvd_start: float, tvd_end: float, t_of_tvd: Callable[[float], float],
                      *, max_step: float = STATIC_COLUMN_STEP_FT) -> float:
    """Pressure at ``tvd_end`` of a static gas column starting at ``p_start`` at ``tvd_start``:
    dp/dz = rho_g(p, T(z))/144 (z positive downward, so an upward integration has a negative
    step). Classical RK4 on steps of at most ``max_step`` ft; COMPUTED."""
    dz = tvd_end - tvd_start
    if dz == 0.0:
        return p_start
    n = max(1, int(math.ceil(abs(dz) / max_step)))
    h = dz / n

    def grad(p: float, z: float) -> float:
        return pvt.gas_density(p, t_of_tvd(z)) / 144.0

    p, z = p_start, tvd_start
    for _ in range(n):
        k1 = grad(p, z)
        k2 = grad(p + 0.5 * h * k1, z + 0.5 * h)
        k3 = grad(p + 0.5 * h * k2, z + 0.5 * h)
        k4 = grad(p + h * k3, z + h)
        p += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        z += h
    return p


def geothermal(case: Case, t_surface: float) -> Callable[[float], float]:
    """Temperature (degF) linear in TVD from ``t_surface`` at 0 to the reservoir temperature at the datum."""
    tvd = case.reservoir.datum_tvd
    t_res = case.reservoir.temperature

    def t_of(z: float) -> float:
        return t_surface + (t_res - t_surface) * z / tvd

    return t_of


def invert_p_res(case: Case) -> InverseResult:
    """p_r (psia, at the datum TVD) from ``case.observed.siwhp`` through a static real-gas column."""
    t0 = time.perf_counter()
    if case.observed is None or case.observed.siwhp is None:
        raise InverseError("invert_p_res needs case.observed.siwhp")
    if case.liquid is not None or case.liquid_inflow is not None:
        raise InverseError(
            "the case has a liquid block: a liquid leg in the shut-in column is not modelled, so the shut-in "
            "wellhead pressure cannot be carried to the datum; the SIWHP inverse stops here"
        )
    if case.fluid.type != "dry_gas":
        raise InverseError("the SIWHP inverse needs a dry_gas fluid (the static column uses the gas PVT)")
    obs = case.observed.siwhp
    res = case.reservoir
    warnings: list[str] = []
    notes: list[str] = []
    if not obs.time_note:
        warnings.append("observed.siwhp.time_note is empty: no shut-in time given, so the reading may not be stabilised")
    tvd = res.datum_tvd
    t_of = geothermal(case, obs.surface_temperature)
    with collect_range_warnings() as range_warnings:
        pvt = make_pvt(case.fluid, case.options.correlations)
        p_res = static_gas_column(pvt, obs.pressure, 0.0, tvd, t_of)
    d = case.model_dump()
    d["reservoir"]["pressure"] = p_res
    result_case = Case(**d)
    grad_100 = 100.0 * (res.temperature - obs.surface_temperature) / tvd
    notes.append(
        f"static gas column from {obs.pressure:.2f} psia at surface to {p_res:.2f} psia at the datum {tvd:.1f} ft TVD; "
        f"geothermal {obs.surface_temperature:.1f} to {res.temperature:.1f} degF ({grad_100:.2f} degF/100 ft), "
        f"RK4 in {STATIC_COLUMN_STEP_FT:g} ft steps; prior reservoir pressure {res.pressure:.2f} psia replaced"
    )
    return InverseResult(
        kind="p_res", value=p_res, unit="psia",
        observation={"siwhp_psia": obs.pressure, "surface_temperature_degf": obs.surface_temperature,
                     "time_note": obs.time_note},
        fixed_inputs={
            "gas_sg": case.fluid.gas_sg, "n2": case.fluid.n2, "co2": case.fluid.co2, "h2s": case.fluid.h2s,
            "datum_tvd_ft": tvd, "depth_md_ft": res.depth_md, "t_res_degf": res.temperature,
            "surface_temperature_degf": obs.surface_temperature, "geothermal_degf_per_100ft": grad_100,
            "gas_z": case.options.correlations.gas_z, "prior_pressure_psia": res.pressure,
        },
        statement=P_RES_STATEMENT, case=result_case,
        tolerance={"integrator": f"RK4, {STATIC_COLUMN_STEP_FT:g} ft steps in TVD", "rule": "evaluation, not a search"},
        notes=notes, warnings=warnings, range_warnings=[str(w) for w in range_warnings],
        elapsed_s=time.perf_counter() - t0,
    )


def invert(case: Case) -> InverseResult:
    """Dispatch on the observation: rate -> kh, siwhp -> p_r."""
    if case.observed is None:
        raise InverseError("the case has no observed block; nothing to invert")
    if case.observed.rate is not None:
        return invert_kh(case)
    return invert_p_res(case)
