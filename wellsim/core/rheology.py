"""Herschel-Bulkley (Yield-Power-Law / YPL) non-Newtonian rheology and HPHT corrections.

Follows API RP 13D (Recommended Practice on the Rheology and Hydraulics of
Oil-well Drilling Fluids) standards for 3-parameter fluid characterization
and downhole pressure/temperature viscosity corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from wellsim.core.friction import RE_LAMINAR, darcy_friction_factor

__all__ = [
    "HerschelBulkleyParams",
    "fit_herschel_bulkley",
    "herschel_bulkley_pressure_gradient",
    "hpht_correction",
]


@dataclass(frozen=True)
class HerschelBulkleyParams:
    """Herschel-Bulkley (Yield-Power-Law) rheological parameters: tau = tau_0 + K * gamma_dot^n.

    Attributes:
        tau_0: Yield stress (lbf / 100 ft^2).
        k: Consistency index (lbf * s^n / 100 ft^2).
        n: Flow behavior index (dimensionless, 0 < n <= 1 for pseudoplastic/shear-thinning).
    """

    tau_0: float
    k: float
    n: float

    def wall_shear_stress(self, shear_rate_sec: float) -> float:
        """Shear stress in lbf/100 ft^2 at a given shear rate (1/s)."""
        gamma = max(0.0, shear_rate_sec)
        return self.tau_0 + self.k * (gamma**self.n)


def fit_herschel_bulkley(
    theta_600: float,
    theta_300: float,
    theta_200: Optional[float] = None,
    theta_100: Optional[float] = None,
    theta_6: Optional[float] = None,
    theta_3: Optional[float] = None,
) -> HerschelBulkleyParams:
    """Fit Herschel-Bulkley 3-parameter model from standard Fann 35 rotational viscometer dial readings.

    API RP 13D method:
    - tau_0 = 2 * theta_3 - theta_6 (or theta_3 if theta_6 unavailable)
    - n = 3.322 * log10((theta_600 - tau_0) / (theta_300 - tau_0))
    - K = (theta_300 - tau_0) / (511.0^n)

    Args:
        theta_600: 600 RPM dial reading.
        theta_300: 300 RPM dial reading.
        theta_200: Optional 200 RPM dial reading.
        theta_100: Optional 100 RPM dial reading.
        theta_6: Optional 6 RPM dial reading.
        theta_3: Optional 3 RPM dial reading.

    Returns:
        HerschelBulkleyParams instance.
    """
    if theta_300 <= 0.0 or theta_600 <= theta_300:
        # Fallback to Newtonian or mild Bingham
        return HerschelBulkleyParams(tau_0=0.0, k=theta_300 / 511.0, n=1.0)

    # 1. Estimate yield stress tau_0 from low-shear readings
    if theta_3 is not None and theta_6 is not None:
        tau_0 = 2.0 * theta_3 - theta_6
        if tau_0 < 0.0:
            tau_0 = theta_3 / 2.0
    elif theta_3 is not None:
        tau_0 = float(theta_3)
    elif theta_6 is not None:
        tau_0 = theta_6 / 2.0
    else:
        # Estimate from 600 & 300: standard Bingham YP = theta_300 - PV
        pv = theta_600 - theta_300
        tau_0 = max(0.0, theta_300 - pv)

    # Guard: tau_0 cannot exceed theta_300
    tau_0 = min(tau_0, 0.9 * theta_300)

    # 2. Flow behavior index n
    numerator = max(1e-4, theta_600 - tau_0)
    denominator = max(1e-4, theta_300 - tau_0)
    n = 3.321928 * math.log10(numerator / denominator)

    # Clamp n to physical shear-thinning drilling fluid range [0.2, 1.0]
    n = max(0.2, min(1.0, n))

    # 3. Consistency index K (lbf s^n / 100 ft^2)
    # At 300 RPM, shear rate gamma_dot = 511 s^-1
    k = (theta_300 - tau_0) / (511.0**n)
    k = max(1e-5, k)

    return HerschelBulkleyParams(tau_0=tau_0, k=k, n=n)


def hpht_correction(
    params: HerschelBulkleyParams,
    t_f: float,
    p_psia: float,
    t_ref_f: float = 120.0,
    p_ref_psia: float = 14.7,
    alpha_thermal: float = 0.007,
    beta_pressure: float = 0.000015,
) -> HerschelBulkleyParams:
    """Apply High-Pressure High-Temperature (HPHT) correction to mud rheology.

    Models thermal thinning and piezo-thickening based on empirical Fann 75/77 scaling:
        factor = exp(-alpha * (T - T_ref) + beta * (P - P_ref))

    Args:
        params: Surface ambient HerschelBulkleyParams.
        t_f: Downhole fluid temperature (degF).
        p_psia: Downhole hydrostatic/circulating pressure (psia).
        t_ref_f: Reference calibration temperature (default 120 degF).
        p_ref_psia: Reference calibration pressure (default 14.7 psia).
        alpha_thermal: Thermal thinning coefficient (1/degF, typically 0.005 to 0.012).
        beta_pressure: Piezo-thickening coefficient (1/psia, typically 1.0e-5 to 2.5e-5).

    Returns:
        HerschelBulkleyParams adjusted for downhole in-situ conditions.
    """
    exponent = -alpha_thermal * (t_f - t_ref_f) + beta_pressure * (p_psia - p_ref_psia)
    # Clamp exponent to prevent numerical blowup
    exponent = max(-2.0, min(2.0, exponent))
    factor = math.exp(exponent)

    # Consistency index K and yield stress tau_0 scale with viscosity factor
    tau_0_adj = max(0.0, params.tau_0 * factor)
    k_adj = max(1e-6, params.k * factor)
    # n remains largely constant with moderate P/T
    return HerschelBulkleyParams(tau_0=tau_0_adj, k=k_adj, n=params.n)


def herschel_bulkley_pressure_gradient(
    rho_ppg: float,
    v_fps: float,
    d_in: float,
    params: HerschelBulkleyParams,
    is_annulus: bool = False,
    rel_rough: float = 0.00018,
    t_f: Optional[float] = None,
    p_psia: Optional[float] = None,
) -> float:
    """Calculate frictional pressure gradient (psi/ft) for a Herschel-Bulkley fluid (API RP 13D).

    Args:
        rho_ppg: Mud density in ppg.
        v_fps: Mean fluid velocity in ft/s.
        d_in: Conduit diameter (pipe ID or hydraulic clearance ID - OD) in inches.
        params: Herschel-Bulkley parameters (tau_0, K, n).
        is_annulus: True for annular flow, False for pipe flow.
        rel_rough: Relative pipe roughness (default 0.00018 for commercial steel).
        t_f: Optional temperature (degF) for HPHT adjustment.
        p_psia: Optional pressure (psia) for HPHT adjustment.

    Returns:
        Frictional pressure gradient in psi/ft.
    """
    if v_fps <= 0.0 or d_in <= 0.0 or rho_ppg <= 0.0:
        return 0.0

    # Adjust for HPHT downhole conditions if provided
    hb = params
    if t_f is not None and p_psia is not None:
        hb = hpht_correction(params, t_f, p_psia)

    n = hb.n
    k = hb.k
    tau_0 = hb.tau_0

    # 1. Wall shear rate gamma_w (1/s) according to API RP 13D
    # Field formula:
    # Pipe: gamma_w = ((3n + 1) / (4n)) * (96 * v_fps / d_in)
    # Annulus: gamma_w = ((2n + 1) / (3n)) * (144 * v_fps / d_in)
    if is_annulus:
        geom_factor = (2.0 * n + 1.0) / (3.0 * n)
        gamma_w = geom_factor * (144.0 * v_fps / d_in)
    else:
        geom_factor = (3.0 * n + 1.0) / (4.0 * n)
        gamma_w = geom_factor * (96.0 * v_fps / d_in)

    # 2. Wall shear stress tau_w (lbf / 100 ft^2)
    tau_w = hb.wall_shear_stress(gamma_w)

    # 3. Effective viscosity mu_eff (cp)
    # 1 lbf*s/100 ft^2 = 47880.26 cp -> 100 lbf/100 ft^2 / (1/s) = 478.8 cp
    mu_eff = (tau_w / max(1e-3, gamma_w)) * 478.8

    # 4. Generalized Reynolds number
    # Re = 928 * rho * v * D / mu_eff
    re_gen = 928.0 * rho_ppg * v_fps * d_in / max(0.1, mu_eff)

    # 5. Critical Reynolds number transition (API RP 13D / Hanks)
    # For shear-thinning fluid, transition typically occurs around:
    re_crit = 3470.0 - 1370.0 * n

    if re_gen < re_crit:
        # Laminar regime: direct integration from wall shear stress
        # dp/dL = 4 * tau_w / (d * 144) for pipe -> psi/ft
        # In field units:
        if is_annulus:
            # Annular laminar loss: dp/dL = tau_w / (200 * d_in)
            return tau_w / (200.0 * d_in)
        else:
            # Pipe laminar loss: dp/dL = tau_w / (300 * d_in)
            return tau_w / (300.0 * d_in)
    else:
        # Turbulent regime: Dodge & Metzner / Colebrook-White using effective viscosity
        f = darcy_friction_factor(re_gen, rel_rough)
        rho_lbm_ft3 = rho_ppg * 7.480519
        d_ft = d_in / 12.0
        # Darcy-Weisbach: dp/dL = f * rho * v^2 / (2 * g_c * D * 144)
        return (f * rho_lbm_ft3 * (v_fps**2)) / (2.0 * 32.174 * d_ft * 144.0)
