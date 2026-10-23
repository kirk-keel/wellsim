"""Pipe friction factor (Darcy-Weisbach / Moody form).

``darcy_friction_factor(re, rel_rough)`` returns the Moody friction factor f such
that the frictional pressure gradient is f rho v^2 / (2 g_c d). The Fanning factor
used in the Cullender-Smith equations is f / 4.

Laminar: f = 64 / Re for Re < 2300.
Turbulent: Colebrook-White, 1/sqrt(f) = -2 log10(eps/(3.7 d) + 2.51/(Re sqrt(f))),
solved by Newton iteration from the Swamee-Jain explicit estimate. Between 2300
and 4000 the Colebrook value is used as-is (the usual engineering practice; the
transition region is not modelled).
"""

from __future__ import annotations

import math

__all__ = [
    "darcy_friction_factor",
    "reynolds_number",
    "RE_LAMINAR",
    "bingham_reynolds_number",
    "hedstrom_number",
    "critical_reynolds_number",
    "bingham_pressure_gradient",
]

RE_LAMINAR = 2300.0
_LBM_FT_S_PER_CP = 6.719689e-4  # 1 cp = 6.7197e-4 lbm/(ft s)


def bingham_reynolds_number(rho_ppg: float, v_fps: float, d_in: float, pv_cp: float) -> float:
    """Bingham Reynolds number: Re_B = 928 * rho * v * D / PV."""
    if pv_cp <= 0.0 or d_in <= 0.0:
        return 0.0
    return 928.0 * rho_ppg * abs(v_fps) * d_in / pv_cp


def hedstrom_number(rho_ppg: float, yp_lbf_100ft2: float, d_in: float, pv_cp: float) -> float:
    """Hedstrom number: He = 37100 * rho * YP * D^2 / PV^2."""
    if pv_cp <= 0.0 or d_in <= 0.0:
        return 0.0
    return 37100.0 * rho_ppg * max(0.0, yp_lbf_100ft2) * (d_in**2) / (pv_cp**2)


def critical_reynolds_number(hedstrom: float) -> float:
    """Critical Reynolds number for transition from laminar to turbulent Bingham plastic flow."""
    if hedstrom <= 0.0:
        return RE_LAMINAR
    # Hanks (1963) / API approximation
    return max(RE_LAMINAR, 1.35 * (hedstrom**0.45))


def bingham_pressure_gradient(
    rho_ppg: float,
    v_fps: float,
    d_in: float,
    pv_cp: float,
    yp_lbf_100ft2: float,
    is_annulus: bool = False,
    rel_rough: float = 0.00018,
) -> float:
    """Frictional pressure gradient (psi/ft) for a Bingham plastic drilling mud.

    Args:
        rho_ppg: Fluid density (ppg).
        v_fps: Flow velocity (ft/s).
        d_in: Inner diameter (pipe) or hydraulic clearance (ID_hole - OD_pipe) (inches).
        pv_cp: Plastic Viscosity (cp).
        yp_lbf_100ft2: Yield Point (lbf/100 ft2).
        is_annulus: True for annular flow, False for pipe flow.
        rel_rough: Pipe relative roughness eps/d.

    Returns:
        Frictional pressure gradient in psi/ft.
    """
    if v_fps <= 0.0 or d_in <= 0.0:
        return 0.0

    re_b = bingham_reynolds_number(rho_ppg, v_fps, d_in, pv_cp)
    he = hedstrom_number(rho_ppg, yp_lbf_100ft2, d_in, pv_cp)
    re_crit = critical_reynolds_number(he)

    if re_b < re_crit:
        # Laminar flow (API RP 13D)
        if is_annulus:
            return (pv_cp * v_fps / (1000.0 * (d_in**2))) + (yp_lbf_100ft2 / (200.0 * d_in))
        else:
            return (pv_cp * v_fps / (1500.0 * (d_in**2))) + (yp_lbf_100ft2 / (225.0 * d_in))
    else:
        # Turbulent flow (effective viscosity approach)
        if is_annulus:
            mu_eff = pv_cp + (6.66 * yp_lbf_100ft2 * d_in / v_fps)
        else:
            mu_eff = pv_cp + (5.0 * yp_lbf_100ft2 * d_in / v_fps)

        re_eff = 928.0 * rho_ppg * v_fps * d_in / max(1.0, mu_eff)
        f = darcy_friction_factor(re_eff, rel_rough)
        rho_lbm_ft3 = rho_ppg * 7.480519
        d_ft = d_in / 12.0
        # Darcy-Weisbach: dp/dL = f * rho * v^2 / (2 * g_c * D * 144)
        return (f * rho_lbm_ft3 * (v_fps**2)) / (2.0 * 32.174 * d_ft * 144.0)


def reynolds_number(rho: float, v: float, d_ft: float, mu_cp: float) -> float:
    """Re = rho v d / mu with rho lbm/ft3, v ft/s, d ft, mu cp."""
    return rho * abs(v) * d_ft / (mu_cp * _LBM_FT_S_PER_CP)


def darcy_friction_factor(re: float, rel_rough: float) -> float:
    """Moody (Darcy) friction factor for Reynolds number ``re`` and relative roughness eps/d."""
    if re <= 0.0:
        return 0.0
    if re < RE_LAMINAR:
        return 64.0 / re
    a = rel_rough / 3.7
    b = 2.51 / re
    # Swamee-Jain starting guess, then Newton on x = 1/sqrt(f)
    f0 = 0.25 / math.log10(a + 5.74 / re**0.9) ** 2
    x = 1.0 / math.sqrt(f0)
    for _ in range(30):
        arg = a + b * x
        g = x + 2.0 * math.log10(arg)
        dg = 1.0 + 2.0 * b / (arg * math.log(10.0))
        dx = g / dg
        x -= dx
        if abs(dx) < 1e-12 * max(1.0, abs(x)):
            break
    return 1.0 / (x * x)
