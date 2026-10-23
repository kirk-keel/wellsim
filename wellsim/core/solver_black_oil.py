"""Black-oil nodal solve (M4): oil inflow from OilIPR (Darcy + Vogel composite),
outflow from the two-phase adiabatic march (two_phase.outflow_profile),
root of pwf - pwf_vlp in pwf space.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import wellsim
from wellsim.cases.loader import case_to_dict
from wellsim.core.ipr.oil import OilIPR
from wellsim.core.nodal import NodalResult, find_operating_point_pwf
from wellsim.core.pvt.black_oil import (
    BlackOilPVT,
    baker_swerdloff_oil,
    mccain_water_viscosity,
    water_density_sc,
)
from wellsim.core.vlp import two_phase as tp

if TYPE_CHECKING:
    from wellsim.cases.schema import Case

__all__ = ["solve_black_oil"]


def solve_black_oil(
    case: "Case",
    pvt: BlackOilPVT,
    ipr: OilIPR,
    notes: list[str],
    warnings: list[str],
    t0: float,
):
    from wellsim.core.solver import Result

    res, num, tm = case.reservoir, case.options.numerics, case.temperature
    p_amb = case.exit.ambient_pressure

    outflows: dict[float, tp.Outflow] = {}
    rho_w = water_density_sc(pvt.water_salinity) if pvt.water_cut > 0 else 62.368
    wc = pvt.water_cut

    def liquid_props_at(pwf: float) -> tp.LiquidProps:
        p_eval = max(14.7, min(pwf, res.pressure))
        rho_o = pvt.oil_density(p_eval, res.temperature)
        rho_l = (1.0 - wc) * rho_o + wc * rho_w

        mu_o = pvt.oil_viscosity(p_eval, res.temperature)
        mu_w = mccain_water_viscosity(p_eval, res.temperature, pvt.water_salinity) if wc > 0 else 1.0
        mu_l = (1.0 - wc) * mu_o + wc * mu_w

        sigma = baker_swerdloff_oil(p_eval, res.temperature, pvt.oil_api)
        cp_l = (1.0 - wc) * 0.50 + wc * 1.0
        return tp.LiquidProps(rho_l=rho_l, mu_l=mu_l, cp_l=cp_l, sigma=sigma, c_l=4500.0)

    tvd_res = case.well.md_to_tvd(res.depth_md)
    # Static column at rest
    liq_res = liquid_props_at(res.pressure)
    p_static_liquid = p_amb + liq_res.rho_l * tvd_res / 144.0

    def outflow_at(pwf: float) -> tp.Outflow | None:
        q_o, q_w, q_g = ipr.rates(pwf)
        q_l = q_o + q_w
        if q_o <= 0.0 and q_l <= 0.0 and q_g <= 0.0:
            return None
        liq = liquid_props_at(pwf)
        w_l = tp.liquid_mass_rate_stbd(q_l, liq.rho_l)
        out = tp.outflow_profile(case, pvt, q_g, w_l, liquid=liq)
        outflows[pwf] = out
        return out

    def pwf_vlp_of_pwf(pwf: float) -> float:
        q_o, q_w, q_g = ipr.rates(pwf)
        if q_o <= 0.0 and (q_o + q_w) <= 0.0:
            return p_static_liquid
        out = outflow_at(pwf)
        return math.inf if out is None or out.pwf is None else out.pwf

    notes.append(
        f"fluid: black oil ({pvt.oil_api:.1f} API, GOR {pvt.gor:.0f} scf/stb, "
        f"water cut {pvt.water_cut * 100:.1f}%), Pb {ipr.pb:.1f} psia"
    )

    q_max = ipr.aof()
    if q_max <= 0.0 or res.pressure <= p_amb:
        nodal = NodalResult([], [], [], [], None, "no_flow", ["reservoir pressure is at or below ambient"], space="pwf")
    else:
        nodal = find_operating_point_pwf(
            ipr.oil_rate,
            None,
            p_amb,
            res.pressure,
            n_scan=num.nodal_scan_points,
            pressure_tol=num.pressure_tol,
            rate_tol=num.rate_tol,
            outflow_pwf=pwf_vlp_of_pwf,
            derivatives=True,
        )
    warnings.extend(nodal.warnings)

    exit_st = None
    profile = None
    pwf = None
    q_o = 0.0
    q_w = 0.0
    q_g = 0.0
    out = None
    temp_info: dict[str, Any] = {"model": tm.model, "sandface_jt": tm.sandface_jt}

    if nodal.operating is not None:
        pwf = nodal.operating.pwf
        q_o, q_w, q_g = ipr.rates(pwf)
        out = outflows.get(pwf) or outflow_at(pwf)
        if out is not None and out.exit is not None:
            exit_st = out.exit
            profile = out.profile
            warnings.extend(out.warnings)
            mach_w = getattr(exit_st, "mach_wallis", exit_st.mach)
            temp_info.update({
                "exit_t": exit_st.t,
                "exit_mach": exit_st.mach,
                "exit_mach_wallis": mach_w,
                "sandface_t_march": profile.t_end if profile else None,
                "min_tpr": profile.min_tpr if profile else None,
                "h_total_btu_lbm": out.h_total,
            })
    else:
        out0 = tp.outflow_profile(case, pvt, 0.0, 0.0)
        exit_st = out0.exit

    correlations = {
        "vlp": tp.NAME,
        "ipr": "oil_darcy_vogel_composite",
        "temperature": tm.model,
        "pvt": repr(pvt),
        "rs_pb": case.options.correlations.rs_pb,
        "bo": case.options.correlations.bo,
    }

    # Static hold mud weight check
    kill_info: dict[str, Any] = {}
    rho_static = (res.pressure - p_amb) / (0.052 * tvd_res)
    kill_info["static_hold_mud_weight_ppg"] = round(rho_static, 2)

    return Result(
        case_name=case.name,
        inputs=case_to_dict(case),
        code_version=wellsim.__version__,
        fluid_type=case.fluid.type,
        correlations=correlations,
        gas_rate=q_g,
        oil_rate=q_o,
        water_rate=q_w,
        pwf=pwf,
        reservoir_pressure=res.pressure,
        drawdown=None if pwf is None else res.pressure - pwf,
        flowing=nodal.operating is not None,
        exit=exit_st,
        profile=profile,
        choke_md=[c.md for c in profile.chokes] if profile else [],
        nodal=nodal,
        ipr=ipr.as_dict(),
        temperature=temp_info,
        exit_entry=(out.entry or None) if out is not None else None,
        warnings=warnings,
        range_warnings=[],
        notes=notes,
        elapsed_s=time.perf_counter() - t0,
        kill=kill_info or None,
    )
