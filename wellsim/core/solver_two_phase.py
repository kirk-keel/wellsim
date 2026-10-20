"""Gas-liquid nodal solve (M3-3): gas inflow from the pseudo-pressure IPR, liquid inflow
from ``case.liquid_inflow`` (wgr or pi), outflow from the two-phase march
(``two_phase.outflow_profile``), root of pwf - pwf_vlp in pwf space.

Every root the scan finds is returned and tagged stable or unstable by the central-
difference test d pwf_vlp/dq_g > d pwf_ipr/dq_g (the crossing direction gives the same
answer and is checked). The highest-rate stable root is the operating point; a warning
names every unstable root. Nothing is picked silently.

The scan runs from ambient pressure to reservoir pressure. Below the static column the
residual is positive anyway, so the exit-limited minimum sandface pressure needs no
special start. A state with no gas and no liquid rate (pwf at or above reservoir
pressure) is reported not evaluable (+inf): in wgr mode the liquid vanishes with the gas
and the q -> 0+ limit is a loaded liquid column, not the static gas column, so the
endpoint is excluded rather than letting a spurious root sit at the discontinuity.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import wellsim
from wellsim.cases.loader import case_to_dict
from wellsim.core.nodal import find_operating_point_pwf
from wellsim.core.vlp import holdup as holdup_mod
from wellsim.core.vlp import two_phase as tp

if TYPE_CHECKING:
    from wellsim.cases.schema import Case
    from wellsim.core.ipr.gas import GasIPR

__all__ = ["solve_two_phase"]


def solve_two_phase(case: "Case", pvt, ipr: "GasIPR", notes: list[str], warnings: list[str], t0: float):
    from wellsim.core.solver import Result

    res, num, tm = case.reservoir, case.options.numerics, case.temperature
    p_amb = case.exit.ambient_pressure
    if tm.model != "adiabatic" or case.exit.k_override is not None:
        raise NotImplementedError("the gas-liquid nodal solve (M3-3) needs temperature.model 'adiabatic' without k_override")
    liq = tp.LiquidProps.from_schema(case.liquid)
    li = case.liquid_inflow
    outflows: dict[float, tp.Outflow] = {}

    def q_liquid(pwf: float, q_g: float) -> float:
        return li.liquid_rate(pwf, q_g, res.pressure)

    tvd_res = case.well.md_to_tvd(res.depth_md)
    p_static_liquid = p_amb + liq.rho_l * tvd_res / 144.0  # the q -> 0+ limit of the loaded column

    def outflow_at(pwf: float) -> tp.Outflow | None:
        q_g = ipr.rate(pwf)
        q_l = q_liquid(pwf, q_g)
        if q_g <= 0.0 and q_l <= 0.0:
            return None
        out = tp.outflow_profile(case, pvt, q_g, tp.liquid_mass_rate_stbd(q_l, liq.rho_l))
        outflows[pwf] = out
        return out

    def pwf_vlp_of_pwf(pwf: float) -> float:
        # At zero gas and zero liquid rate (pwf at or above reservoir pressure) the outflow
        # is the q -> 0+ limit of the two-phase column: the liquid vanishes with the gas in
        # both modes, but the holdup goes to 1 first, so the limit is the static liquid
        # column, not the static gas column of the hand-off (COMPUTED; keeps f continuous
        # at the top of the scan and lets a loaded well show its rising crossing there).
        q_g = ipr.rate(pwf)
        if q_g <= 0.0 and q_liquid(pwf, q_g) <= 0.0:
            return p_static_liquid
        out = outflow_at(pwf)
        return math.inf if out is None or out.pwf is None else out.pwf

    notes.append(
        f"liquid inflow: mode '{li.mode}', "
        + (f"WGR {li.water_gas_ratio:g} bbl/MMscf" if li.mode == "wgr" else f"J {li.productivity_index:g} stb/d/psi")
        + f"; liquid {case.liquid.rho_l:g} ppg, {case.liquid.mu_l:g} cp, sigma {liq.sigma:g} dyne/cm, c_l {liq.c_l:g} ft/s"
    )
    q_max = ipr.aof()
    if q_max <= 0.0 or res.pressure <= p_amb:
        from wellsim.core.nodal import NodalResult

        nodal = NodalResult([], [], [], [], None, "no_flow", ["reservoir pressure is at or below ambient"], space="pwf")
    else:
        nodal = find_operating_point_pwf(
            ipr.rate, None, p_amb, res.pressure, n_scan=num.nodal_scan_points,
            pressure_tol=num.pressure_tol, rate_tol=num.rate_tol, outflow_pwf=pwf_vlp_of_pwf, derivatives=True,
        )
    warnings.extend(nodal.warnings)

    exit_st = None
    profile = None
    pwf = None
    q_g = 0.0
    q_l = 0.0
    out = None
    temp_info: dict[str, Any] = {"model": "adiabatic", "sandface_jt": tm.sandface_jt}
    if nodal.operating is not None:
        pwf = nodal.operating.pwf
        q_g = nodal.operating.q
        q_l = q_liquid(pwf, q_g)
        out = outflows.get(pwf) or outflow_at(pwf)
        exit_st, profile = out.exit, out.profile
        pwf_march = profile.p_end
        mismatch = abs(pwf_march - pwf)
        if mismatch > num.pressure_tol:
            warnings.append(f"IPR and VLP pwf differ by {mismatch:.2f} psi at the operating point (pressure_tol {num.pressure_tol} psi)")
        pwf = pwf_march
        for w in out.warnings:
            if w not in warnings:
                warnings.append(w)
        for i in nodal.intersections:
            notes.append(
                f"root: q_g {i.q:.1f} Mscf/d, q_l {q_liquid(i.pwf, i.q):.1f} stb/d, pwf {i.pwf:.1f} psia, "
                f"{'stable' if i.stable else 'unstable'}"
                + (f" (d pwf_vlp/dq {i.dvlp_dq:.4g}, d pwf_ipr/dq {i.dipr_dq:.4g} psi per Mscf/d)"
                   if i.dvlp_dq is not None else "")
            )
        if profile.regime is not None:
            reg_exit, reg_sf = profile.regime[0], profile.regime[-1]
            al_exit, al_sf = profile.alpha_g[0], profile.alpha_g[-1]
        else:  # zero liquid rate at the operating point: the gas hand-off ran
            reg_exit = reg_sf = "gas"
            al_exit = al_sf = 1.0
        mach_w = getattr(exit_st, "mach_wallis", exit_st.mach)
        notes.append(
            f"two-phase operating point: q_g {q_g:.1f} Mscf/d, q_l {q_l:.1f} stb/d, pwf {pwf:.1f} psia; regime "
            f"{reg_sf} (alpha_g {al_sf:.3f}) at the sandface, {reg_exit} (alpha_g {al_exit:.3f}) at the exit; "
            f"exit {'choked' if exit_st.choked else 'unchoked'} at {exit_st.p:.1f} psia, Wallis Mach {mach_w:.3f}"
        )
        if exit_st.choked:
            warnings.append(
                f"exit is choked: exit pressure {exit_st.p:.1f} psia above ambient {p_amb:.1f} psia (two-phase singular "
                f"point; Wallis frozen sound speed {exit_st.c:.0f} ft/s gives Mach {mach_w:.3f} there)"
            )
        for c in profile.chokes:
            warnings.append(
                f"flow is choked at the restriction at md {c.md:.1f} ft entering {c.conduit}: pressure below it reset "
                f"from {c.p_before:.1f} to {c.p_after:.1f} psia"
            )
        temp_info.update({
            "exit_t": exit_st.t, "exit_mach": exit_st.mach, "exit_mach_wallis": mach_w,
            "sandface_t_march": profile.t_end, "min_tpr": profile.min_tpr, "h_total_btu_lbm": out.h_total,
        })
        if ipr.cap_active(pwf):
            notes.append(f"IPR capped at its maximum: pwf {pwf:.0f} psia is below the pressure of maximum rate {ipr.p_aof:.0f} psia")
    else:
        # describe the exit at a token liquid-free state so the result still says something
        out0 = tp.outflow_profile(case, pvt, 0.0, 0.0)
        exit_st = out0.exit

    correlations = {
        "vlp": tp.NAME,
        "holdup": holdup_mod.NAME,
        "ipr": "gas_pseudo_pressure_pss",
        "liquid_inflow": li.mode,
        "temperature": "adiabatic_total_enthalpy",
        "pvt": repr(pvt),
        "gas_z": case.options.correlations.gas_z,
        "non_darcy_beta": ipr.beta_source,
        "exit": "two_phase_singular_point (Wallis frozen c_m reported)",
        "heat_capacity_ratio": "real_gas_cp_cv_dak",
        "cp0_composition": pvt.composition.as_dict(),
    }
    return Result(
        case_name=case.name,
        inputs=case_to_dict(case),
        code_version=wellsim.__version__,
        fluid_type=case.fluid.type,
        correlations=correlations,
        gas_rate=q_g,
        oil_rate=0.0,
        water_rate=q_l,
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
        exit_entry=(out.entry or None) if out is not None else None,
        warnings=warnings,
        range_warnings=[],
        notes=notes,
        elapsed_s=time.perf_counter() - t0,
    )
