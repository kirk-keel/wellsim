"""``solve(case) -> Result``: the pure end-to-end blowout-rate calculation (spec 1, 5).

No global state, no UI, nothing mutated on the case. M2 scope: dry gas only. Black
oil raises NotImplementedError until M4.

Pipeline
    PVT (make_pvt) -> temperature profile -> flow-path segments -> gas IPR
    -> outflow curve pwf_vlp(q) = [exit state for q] then march down to the inflow node
    -> nodal scan + Brent -> operating point -> final profile
Every RangeWarning raised anywhere inside is collected onto the result.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

import wellsim
from wellsim.cases.loader import case_to_dict
from wellsim.cases.schema import Case
from wellsim.core.exit import ExitState, exit_state, exit_state_adiabatic, gas_mass_rate
from wellsim.core.geometry import FlowSegment, exit_area, flow_path_segments
from wellsim.core.ipr.gas import GasIPR
from wellsim.core.nodal import NodalResult, find_operating_point, find_operating_point_pwf
from wellsim.core.pvt import collect_range_warnings, make_pvt
from wellsim.core.pvt.base import M_AIR
from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU, enthalpy, isenthalpic_outlet_temperature
from wellsim.core.temperature import make_temperature
from wellsim.core.vlp.area_change import AreaChangeNotEvaluable, contraction_with_loss, crane_contraction_coefficient
from wellsim.core.vlp.correlations import single_phase_gas
from wellsim.core.vlp.march import MarchResult, march

__all__ = ["Result", "solve", "build_ipr"]


@dataclass
class Result:
    case_name: str
    inputs: dict[str, Any]
    code_version: str
    fluid_type: str
    correlations: dict[str, Any]
    gas_rate: float  # Mscf/d at standard conditions
    oil_rate: float  # stb/d
    water_rate: float  # stb/d
    pwf: float | None  # psia
    reservoir_pressure: float
    drawdown: float | None
    flowing: bool
    exit: ExitState | None
    profile: MarchResult | None
    choke_md: list[float]  # depths of choked internal restrictions (empty if none); exit choke is in `exit`
    nodal: NodalResult
    ipr: dict[str, Any]
    temperature: dict[str, Any] = field(default_factory=dict)  # model, sandface/exit T, exit Mach, min Tpr
    exit_entry: dict[str, Any] | None = None  # adiabatic restricted exit: contraction step throat -> conduit
    warnings: list[str] = field(default_factory=list)
    range_warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # informational, not assumptions
    elapsed_s: float = 0.0
    kill: dict[str, Any] | None = None  # M3-4: dynamic-kill analysis when case.dynamic_kill is given

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "inputs": self.inputs,
            "code_version": self.code_version,
            "fluid_type": self.fluid_type,
            "correlations": self.correlations,
            "gas_rate_mscfd": self.gas_rate,
            "oil_rate_stbd": self.oil_rate,
            "water_rate_stbd": self.water_rate,
            "pwf_psia": self.pwf,
            "reservoir_pressure_psia": self.reservoir_pressure,
            "drawdown_psi": self.drawdown,
            "flowing": self.flowing,
            "exit": self.exit.as_dict() if self.exit else None,
            "profile": self.profile.as_dict() if self.profile else None,
            "choke_md": list(self.choke_md),
            "nodal": self.nodal.as_dict(),
            "ipr": self.ipr,
            "temperature": dict(self.temperature),
            "exit_entry": None if self.exit_entry is None else dict(self.exit_entry),
            "warnings": list(self.warnings),
            "range_warnings": list(self.range_warnings),
            "notes": list(self.notes),
            "elapsed_s": self.elapsed_s,
            "kill": None if self.kill is None else dict(self.kill),
        }


def _static_warnings(case: Case) -> list[str]:
    w: list[str] = []
    r = case.reservoir
    if case.flow_path == "annulus":
        w.append("annular flow uses the hydraulic diameter (ID - OD); known weak point (spec 4.3)")
    tm = case.temperature
    t_res = r.temperature
    profile_mode = tm.model == "profile" or case.exit.k_override is not None
    if profile_mode:
        t_exit = tm.exit_temperature if tm.exit_temperature is not None else t_res - tm.default_exit_loss
        if tm.exit_temperature is None:
            w.append(
                f"exit temperature defaulted to reservoir temperature minus {tm.default_exit_loss:g} degF "
                f"({t_exit:.1f} degF): the warm end of the plausible range; expansion (Joule-Thomson and "
                "kinetic) cooling is not modelled, so the rate may be understated by roughly 2 to 6%"
            )
        if t_exit > t_res:
            w.append(
                f"exit temperature {t_exit:.1f} degF exceeds reservoir temperature {t_res:.1f} degF: "
                "the flowing profile is inverted; check the inputs"
            )
    else:
        w.append(
            "adiabatic temperature model: wellbore heat transfer to the formation is not modelled "
            "(Hasan-Kabir later), so cooling is overstated at low rates"
        )
    if r.time_since_start is not None:
        w.append("transient IPR (time_since_start) not implemented; pseudo-steady state used")
    if case.exit.exit_id is not None:
        if profile_mode:
            w.append("exit_id restriction: Mach 1 is checked at the restriction area; no contraction loss "
                     "is applied on the profile path")
        else:
            w.append(
                "exit_id restriction: Mach 1 is checked at cd x the restriction area and the entry is a "
                "sudden contraction (Crane K_c when cd = 1.0, isentropic only when cd < 1) from the top conduit"
            )
        if "cd" not in case.exit.model_fields_set:
            w.append(
                "restricted exit with cd = 1.0; sharp orifice typically 0.6 to 0.65, rounded 0.95 to 0.98 "
                "(Crane TP-410, MEMORY)"
            )
    if case.options.correlations.vlp not in (None, [single_phase_gas.NAME]):
        w.append(f"only {single_phase_gas.NAME!r} is available in M2; requested {case.options.correlations.vlp}")
    return w


def build_ipr(case: Case, pvt) -> tuple[GasIPR, str]:
    """The gas IPR ``solve`` uses, and the sentence saying where beta came from.

    Split out of ``solve`` so anything that needs the same IPR without a full solve
    (the Monte Carlo kill pass, core/montecarlo.py) builds it the one way, including the
    'auto' non-Darcy beta rule (M2.5-5): Geertsma when porosity is supplied, else Jones.
    """
    res = case.reservoir
    beta_choice = case.options.correlations.non_darcy_beta
    if beta_choice == "auto":
        beta_method = "geertsma" if res.porosity is not None else "jones"
        beta_reason = (
            "porosity supplied, so Geertsma (phi^-5.5 k^-0.5)" if res.porosity is not None
            else "only permeability supplied, so Jones (k^-1.55)"
        )
    else:
        beta_method = beta_choice
        beta_reason = f"{beta_choice} requested in options.correlations.non_darcy_beta"
    ipr = GasIPR(
        pvt=pvt, p_res=res.pressure, t_res=res.temperature, k=res.permeability, h=res.net_pay,
        r_e=res.drainage_radius, r_w=res.wellbore_radius, skin=res.skin, h_p=res.exposed_interval,
        h_1=res.exposed_top_offset, kv_kh=res.kv_kh,
        beta=res.non_darcy_beta, beta_method=beta_method, porosity=res.porosity,
    )
    return ipr, beta_reason


def solve(case: Case) -> Result:
    t0 = time.perf_counter()
    if case.fluid.type != "dry_gas":
        raise NotImplementedError("black-oil blowout rate lands in M4; M2 solves dry gas only")
    opts = case.options
    num = opts.numerics
    res = case.reservoir
    p_amb = case.exit.ambient_pressure
    k_ratio = case.exit.k_override  # None: ideal-gas cp0/cv0 at local temperature
    warnings = _static_warnings(case)

    with collect_range_warnings() as range_warnings:
        pvt = make_pvt(case.fluid, opts.correlations)
        tm = case.temperature
        adiabatic = tm.model == "adiabatic" and k_ratio is None
        if tm.model == "adiabatic" and k_ratio is not None:
            warnings.append(
                f"k_override = {k_ratio} forces the pre-3b 'profile' temperature path (constant heat capacity "
                "ratio, prescribed linear temperature); the adiabatic energy equation is not used"
            )
        temperature = None if adiabatic else make_temperature(case)
        segments: list[FlowSegment] = flow_path_segments(case.well, case.flow_path, res.depth_md)
        a_exit_geom = exit_area(segments, case.exit.exit_id)
        cd = case.exit.cd
        a_exit = cd * a_exit_geom  # effective exit area
        restricted_exit = case.exit.exit_id is not None and a_exit < segments[0].area
        d_exit_in = case.exit.exit_id if case.exit.exit_id is not None else segments[0].d_hyd
        # adiabatic: the exit temperature is an outcome; res.temperature is only the Newton start
        t_exit = res.temperature if adiabatic else temperature.at(0.0)
        m_gas = M_AIR * pvt.gas_sg
        tvd_res = case.well.md_to_tvd(res.depth_md)
        h_res = enthalpy(pvt, res.pressure, res.temperature) / m_gas if adiabatic else None  # Btu/lbm

        # default non-Darcy beta (M2.5-5): Geertsma when porosity is supplied, else Jones
        ipr, beta_reason = build_ipr(case, pvt)
        notes: list[str] = []
        if res.non_darcy_beta is not None:
            notes.append(f"non-Darcy beta: user value {res.non_darcy_beta:.4g} 1/ft (correlations not used)")
        else:
            notes.append(f"non-Darcy beta: {ipr.beta_source} = {ipr.beta:.4g} 1/ft ({beta_reason})")
        comp = pvt.composition
        hc = ", ".join(f"{n} {y:.3f}" for n, y in comp.fractions.items() if y > 0.0)
        if adiabatic:
            notes.append(
                "heat capacity ratio: real-gas cp/cv from the DAK departures along the adiabatic march "
                f"(ideal-gas k0 not used); gravity pseudo-composition {hc} (M = {comp.molar_mass:.3f}, "
                f"HC gravity {comp.hc_gravity:.3f})"
            )
        elif k_ratio is None:
            notes.append(
                f"heat capacity ratio: ideal-gas cp0/cv0 = {pvt.k_ratio(t_exit):.4f} at the exit "
                f"({t_exit:.1f} degF) and {pvt.k_ratio(res.temperature):.4f} at reservoir temperature, from the "
                f"gravity pseudo-composition {hc} (M = {comp.molar_mass:.3f}, HC gravity {comp.hc_gravity:.3f})"
            )
        else:
            notes.append(f"heat capacity ratio: fixed k_override = {k_ratio} (ideal-gas cp0/cv0 not used)")
        if ipr.h_pd < 1.0:
            notes.append(
                f"partial penetration: Papatzacos pseudo-skin s_pp = {ipr.s_pp:.2f} for h_pD = {ipr.h_pd:.3f} "
                f"(open {ipr.h_p:.1f} of {ipr.h:.1f} ft, top offset {ipr.h_1:.1f} ft) with kv/kh = {ipr.kv_kh:g}; "
                f"mechanical skin {res.skin:g} scaled by h/h_p to {ipr.s_mech:.2f}; total skin {ipr.s_total:.2f}"
            )
            if ipr.kv_kh_defaulted:
                warnings.append(
                    "kv_kh not given for a partially penetrating well: 0.1 assumed for the Papatzacos "
                    f"pseudo-skin (s_pp = {ipr.s_pp:.2f}); supply reservoir.kv_kh to remove this assumption"
                )
            if ipr.h_pd < 0.05:
                warnings.append(
                    f"penetration ratio h_pD = {ipr.h_pd:.3f} is below 0.05: the Papatzacos pseudo-skin "
                    f"({ipr.s_pp:.1f}) and the h_p^2 non-Darcy term dominate the inflow; the correlation is "
                    "being used at the thin end of its range, check the inputs"
                )

        if case.liquid is not None and case.liquid_inflow is not None:
            # M3-3: gas-liquid nodal solve; the dry-gas path below is untouched
            from wellsim.core.solver_two_phase import solve_two_phase

            result = solve_two_phase(case, pvt, ipr, notes, warnings, t0)
            if case.dynamic_kill is not None:
                from wellsim.core.kill import kill_analysis

                result.kill = kill_analysis(case, pvt, ipr)
                result.warnings.extend(w for w in result.kill["warnings"] if w not in result.warnings)
                result.elapsed_s = time.perf_counter() - t0
            result.range_warnings = [str(w) for w in range_warnings]
            return result

        def h_total_for(q: float) -> float:
            """Total enthalpy per lbm at the surface datum (z = 0): inlet enthalpy less the
            potential energy gained from the inflow node (TVD) to the exit."""
            if tm.sandface_jt:
                h_in = h_res  # isenthalpic throttle from (p_res, T_res): h conserved
            else:
                p_in = ipr.pwf(q)
                if p_in is None:
                    p_in = p_amb
                h_in = enthalpy(pvt, p_in, res.temperature) / m_gas  # enters at T_res and pwf
            return h_in - tvd_res / J_FT_LBF_PER_BTU

        not_evaluable: dict[str, float] = {}  # reason -> lowest rate it was seen at
        entry_log: dict[str, Any] = {}  # exit-restriction entry step at the operating point (adiabatic)

        def _record(reason: str, q: float) -> None:
            not_evaluable[reason] = min(q, not_evaluable.get(reason, math.inf))

        def pwf_vlp(q: float, keep: bool = False):
            w = gas_mass_rate(q, pvt.gas_sg)
            if adiabatic:
                ht = h_total_for(q)
                ex = exit_state_adiabatic(pvt, q, p_amb, a_exit, ht, res.temperature - 40.0, p_max=res.pressure)
                ex = replace(ex, cd=cd, area_geometric=a_exit_geom)
                if ex.not_evaluable is not None:
                    _record(ex.not_evaluable, q)
                    return (ex, None) if keep else None
                p0, t0 = ex.p, ex.t
                if restricted_exit:
                    # entry to the restriction is a flow-direction contraction from the top
                    # conduit: Crane loss when cd = 1 (cd < 1 already carries it), then the
                    # isentropic deceleration back to the conduit state below the exit
                    kc = crane_contraction_coefficient(d_exit_in, segments[0].d_hyd) if cd == 1.0 else 0.0
                    try:
                        p0, t0, mach0, loss, resid = contraction_with_loss(
                            pvt, w, a_exit, segments[0].area, ex.p, ex.t, ht, res.pressure, kc)
                    except AreaChangeNotEvaluable as e:
                        _record(f"exit restriction entry: {e}", q)
                        return (ex, None) if keep else None
                    if keep:
                        entry_log.update({
                            "k_c": kc, "loss_psi": loss, "p_throat": ex.p, "t_throat": ex.t,
                            "p_conduit": p0, "t_conduit": t0, "mach_conduit": mach0, "residual": resid,
                        })
                m = march(
                    pvt, segments, None, w, p0, 0.0, res.depth_md,
                    max_step=num.vlp_max_segment, p_tol=num.vlp_pressure_tol,
                    adiabatic=True, t_start=t0, t_tol=num.vlp_temperature_tol, p_max=res.pressure,
                )
            else:
                ex = replace(exit_state(pvt, q, p_amb, t_exit, a_exit, k_ratio), cd=cd, area_geometric=a_exit_geom)
                m = march(
                    pvt, segments, temperature, w, ex.p, 0.0, res.depth_md,
                    max_step=num.vlp_max_segment, p_tol=num.vlp_pressure_tol, k_ratio=k_ratio,
                )
            if not m.ok:
                _record(m.message, q)
            if keep:
                return ex, m
            return m.p_end if m.ok else None

        # scan to the true IPR maximum (M2.5-3b closeout): rate(p) peaks a little above ambient
        # because D carries mu(p_wf), and an operating point can sit between rate(p_amb)
        # and that peak
        q_max = ipr.aof()
        if q_max <= 0.0 or res.pressure <= p_amb:
            nodal = NodalResult([], [], [], [], None, "no_flow", ["reservoir pressure is at or below ambient"])
        elif num.nodal_space == "pwf":

            def pwf_vlp_inf(q: float) -> float:
                v = pwf_vlp(q)
                return math.inf if v is None else v

            nodal = find_operating_point_pwf(
                ipr.rate, pwf_vlp_inf, p_amb, res.pressure, n_scan=num.nodal_scan_points,
                pressure_tol=num.pressure_tol, rate_tol=num.rate_tol,
            )
        else:
            nodal = find_operating_point(
                ipr.pwf, pwf_vlp, 1e-3 * q_max, q_max, n_scan=num.nodal_scan_points, rate_tol=num.rate_tol,
                pressure_tol=num.pressure_tol,
            )
        warnings.extend(nodal.warnings)
        for reason, q_low in sorted(not_evaluable.items(), key=lambda kv: kv[1]):
            notes.append(f"outflow not evaluable from {q_low:.0f} Mscf/d upward: {reason}")

        exit_st: ExitState | None = None
        profile: MarchResult | None = None
        pwf = None
        q = 0.0
        if nodal.operating is not None:
            q = nodal.operating.q
            exit_st, profile = pwf_vlp(q, keep=True)
            pwf = profile.p_end
            mismatch = abs(pwf - nodal.operating.pwf)
            if mismatch > num.pressure_tol:
                warnings.append(
                    f"IPR and VLP pwf differ by {mismatch:.2f} psi at the operating point "
                    f"(pressure_tol {num.pressure_tol} psi)"
                )
            if exit_st.choked:
                warnings.append(
                    f"exit is choked: exit pressure {exit_st.p:.1f} psia above ambient {p_amb:.1f} psia "
                    + (
                        f"(adiabatic sound speed {exit_st.c:.0f} ft/s from real-gas cp/cv at the exit "
                        f"temperature {exit_st.t:.1f} degF)"
                        if adiabatic
                        else f"(sonic velocity uses the fixed k_override = {k_ratio})"
                        if k_ratio is not None
                        else f"(sonic velocity uses ideal-gas cp0/cv0 = {pvt.k_ratio(t_exit):.4f} at "
                        f"{t_exit:.1f} degF; the real-gas isentropic exponent is not modelled)"
                    )
                )
            if entry_log:
                notes.append(
                    f"exit restriction entry (cd = {cd:g}, effective area {a_exit:.5f} ft2 of {a_exit_geom:.5f}): "
                    f"K_c = {entry_log['k_c']:.3f}, loss {entry_log['loss_psi']:.2f} psi, throat "
                    f"{entry_log['p_throat']:.1f} psia / {entry_log['t_throat']:.1f} degF to conduit "
                    f"{entry_log['p_conduit']:.1f} psia / {entry_log['t_conduit']:.1f} degF (Mach {entry_log['mach_conduit']:.3f})"
                )
            elif cd != 1.0:
                notes.append(f"exit discharge coefficient cd = {cd:g}: effective exit area {a_exit:.5f} ft2 of {a_exit_geom:.5f}")
            for ac in profile.area_changes:
                if ac.kind == "expansion":
                    notes.append(
                        f"sudden expansion (flow direction) at md {ac.md:.1f} ft, {ac.conduit_to} into "
                        f"{ac.conduit_from}: Borda-Carnot momentum balance, p {ac.p_after:.1f} -> {ac.p_before:.1f} psia, "
                        f"T {ac.t_after:.1f} -> {ac.t_before:.1f} degF, Mach {ac.mach_after:.3f} in the narrow conduit"
                        + (" (choked: sonic state, no subsonic root)" if ac.choked else "")
                    )
                else:
                    notes.append(
                        f"sudden contraction (flow direction) at md {ac.md:.1f} ft, {ac.conduit_to} into "
                        f"{ac.conduit_from}: Crane loss {ac.loss_psi:.2f} psi then isentropic deceleration, "
                        f"p {ac.p_after:.1f} -> {ac.p_before:.1f} psia, T {ac.t_after:.1f} -> {ac.t_before:.1f} degF, "
                        f"Mach {ac.mach_after:.3f} in the wide conduit"
                    )
            for c in profile.chokes:
                warnings.append(
                    f"flow is choked at the restriction at md {c.md:.1f} ft entering {c.conduit}: "
                    f"pressure below it reset from {c.p_before:.1f} to the sonic value {c.p_after:.1f} psia "
                    f"(Mach {c.mach_before:.2f} at the continuity pressure). The restriction, not the exit, "
                    "controls the rate; pressures above it are a continuity estimate only"
                    + (" and the exit is choked as well" if exit_st.choked else "")
                )
        else:
            # report the exit state at a token rate so the result still describes the well
            exit_st = exit_state(pvt, 0.0, p_amb, t_exit, a_exit, k_ratio)

        temp_info: dict[str, Any] = {
            "model": "adiabatic" if adiabatic else "profile",
            "sandface_jt": tm.sandface_jt if adiabatic else None,
            "exit_t": exit_st.t,
            "exit_mach": exit_st.mach,
            "min_tpr": profile.min_tpr if profile is not None else None,
        }
        if nodal.operating is not None:
            if adiabatic:
                t_sf = (
                    isenthalpic_outlet_temperature(pvt, res.temperature, res.pressure, pwf)
                    if tm.sandface_jt else res.temperature
                )
                temp_info.update({
                    "sandface_t": t_sf,
                    "sandface_t_march": profile.t_end,
                    "h_total_btu_lbm": h_total_for(q),
                })
                notes.append(
                    f"adiabatic temperature: sandface {t_sf:.1f} degF "
                    + (f"after an isenthalpic throttle from {res.temperature:.1f} degF at {res.pressure:.0f} psia "
                       f"to pwf {pwf:.0f} psia (Joule-Thomson cooling {res.temperature - t_sf:.1f} degF)"
                       if tm.sandface_jt else "(sandface_jt off: enters at reservoir temperature)")
                    + f"; march reaches the sandface at {profile.t_end:.2f} degF (difference is the kinetic "
                    f"term); exit static temperature {exit_st.t:.1f} degF, exit Mach {exit_st.mach:.3f}; "
                    f"minimum Tpr along the path {profile.min_tpr:.3f}"
                )
            else:
                temp_info["sandface_t"] = res.temperature
            if ipr.cap_active(pwf):
                notes.append(
                    f"IPR capped at its maximum: pwf {pwf:.0f} psia is below the pressure of maximum rate "
                    f"{ipr.p_aof:.0f} psia, so the inflow is held at {ipr.q_aof:.0f} Mscf/d (the raw curve's "
                    "turnover below that pressure is an artefact of D carrying mu at pwf)"
                )
            if q >= 0.99 * ipr.q_aof:
                notes.append(
                    f"reservoir-limited, wellbore backpressure negligible: the operating rate {q:.0f} Mscf/d "
                    f"is within 1% of the IPR maximum {ipr.q_aof:.0f} Mscf/d (reached at pwf {ipr.p_aof:.0f} psia)"
                )
            if adiabatic and tm.sandface_jt and res.temperature - temp_info["sandface_t"] > 20.0:
                notes.append(
                    f"sandface Joule-Thomson cooling of {res.temperature - temp_info['sandface_t']:.1f} degF exceeds "
                    "20 degF: the steady-state isenthalpic assumption applies once the near-wellbore rock has "
                    "reached thermal equilibrium (minutes to days at blowout rates); early-time exit temperatures "
                    "are warmer than reported"
                )

    kill_info = None
    if case.dynamic_kill is not None:
        from wellsim.core.kill import kill_analysis

        with collect_range_warnings() as kill_range:
            kill_info = kill_analysis(case, pvt, ipr)
        range_warnings = list(range_warnings) + list(kill_range)
        warnings.extend(w for w in kill_info["warnings"] if w not in warnings)
    range_strs = [str(w) for w in range_warnings]
    correlations = {
        "vlp": single_phase_gas.NAME,
        "ipr": "gas_pseudo_pressure_pss",
        "temperature": "adiabatic_total_enthalpy" if adiabatic else temperature.model,
        "pvt": repr(pvt),
        "gas_z": opts.correlations.gas_z,
        "non_darcy_beta": ipr.beta_source,
        "exit": "real_gas_adiabatic_mach_1" if adiabatic else "real_gas_isentropic_mach_1",
        "heat_capacity_ratio": (
            "real_gas_cp_cv_dak" if adiabatic
            else "k_override" if k_ratio is not None
            else "ideal_gas_cp0_pseudo_composition"
        ),
        "cp0_composition": pvt.composition.as_dict(),
    }
    return Result(
        case_name=case.name,
        inputs=case_to_dict(case),
        code_version=wellsim.__version__,
        fluid_type=case.fluid.type,
        correlations=correlations,
        gas_rate=q,
        oil_rate=0.0,
        water_rate=0.0,
        pwf=pwf,
        reservoir_pressure=res.pressure,
        drawdown=None if pwf is None else res.pressure - pwf,
        flowing=nodal.operating is not None,
        exit=exit_st,
        profile=profile,
        choke_md=[c.md for c in profile.chokes] if profile else [],
        nodal=nodal,
        ipr=ipr.describe(),
        temperature=temp_info,
        exit_entry=entry_log or None,
        warnings=warnings,
        range_warnings=range_strs,
        notes=notes,
        elapsed_s=time.perf_counter() - t0,
        kill=kill_info,
    )
