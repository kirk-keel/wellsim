"""Single-phase gas pressure gradient (spec 4.3, correlation 1).

Steady one-dimensional momentum balance for a gas of constant mass flux G = w / A
along measured depth x (positive downward, flow upward):

    dp/dx = (grav + fric) / (1 - Ma^2)

    grav = rho sin(theta) / 144                        psi/ft
    fric = f rho v^2 / (2 g_c d) / 144                 psi/ft
    Ma^2 = v^2 / c^2,  c^2 = k g_c (dp/drho)_T * 144   (see wellsim.core.exit)

The 1/(1 - Ma^2) factor is the acceleration (kinetic energy) term: rho v dv/dx with
G constant becomes -(G^2/rho^2) drho/dx, and drho/dx = (drho/dp) dp/dx along the
flow. The density derivative is taken along an isentropic path (k times the
isothermal derivative) so that the singularity of the integrator coincides with the
Mach 1 choke criterion at the exit. The contribution of the prescribed temperature
gradient to drho/dx is dropped: with dT/dx of order 0.01 degF/ft it is below
1e-3 psi/ft even at Mach 0.9.

Cullender and Smith (1956) drop the acceleration term entirely; at ordinary gas-well
velocities Ma^2 is below 1e-3 and the two agree, which is the basis of the G1 check
in tests/test_vlp_gas.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wellsim.core.exit import G_C, sonic_velocity
from wellsim.core.friction import darcy_friction_factor, reynolds_number
from wellsim.core.pvt.base import M_AIR
from wellsim.core.pvt.dry_gas import DryGasPVT
from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU as J, thermo_from_state

if TYPE_CHECKING:
    from wellsim.core.geometry import FlowSegment
    from wellsim.core.pvt.base import PVTModel

__all__ = ["NAME", "GasGradient", "gas_gradient", "AdiabaticGradient", "gas_gradient_adiabatic"]

NAME = "single_phase_gas"


@dataclass(frozen=True, slots=True)
class GasGradient:
    p: float
    t: float
    rho: float  # lbm/ft3
    v: float  # ft/s
    mu: float  # cp
    re: float
    f: float  # Moody
    c: float  # sonic velocity, ft/s
    mach: float
    grav: float  # psi/ft
    fric: float  # psi/ft
    tpr: float = 0.0  # pseudo-reduced temperature (for the min-Tpr diagnostic)

    @property
    def ek(self) -> float:
        """Acceleration (kinetic energy) factor Ma^2; dp/dx = (grav + fric) / (1 - ek)."""
        return self.mach * self.mach

    @property
    def dpdx(self) -> float:
        """Total gradient along MD, psi/ft, positive downward. Infinite at Mach 1."""
        return (self.grav + self.fric) / (1.0 - self.ek)


@dataclass(frozen=True, slots=True)
class AdiabaticGradient:
    """Coupled momentum and energy gradient (M2.5-3b), state [p, T], x measured depth
    positive downward, flow upward (L = -x is the flow direction).

    With G the mass flux, rho_p = (drho/dp)_T, rho_T = (drho/dT)_p, cp per lbm, mu the
    Joule-Thomson coefficient, J = 778.169 ft lbf/Btu and primes d/dL:

        momentum:  (1 - a) p' - b T' = -(grav + fric)
        energy:   -(cp mu + 144 a/(rho J)) p' + (cp - 144 b/(rho J)) T' = -sin(theta)/J
        a = v^2 rho_p / (144 g_c),  b = v^2 rho_T / (144 g_c)

    The determinant is D = cp (1 - v^2/c_s^2) with c_s the adiabatic sound speed from
    the departures (COMPUTED identity, tested); Cramer's rule gives p' = N_p/D and
    T' = N_T/D. Along x: dp/dx = dp_ds / det, dT/dx = dt_ds / det with det = D/cp,
    dp_ds = -N_p/cp and dt_ds = -N_T/cp, all regular through the sonic point."""

    p: float
    t: float
    rho: float
    v: float
    mu: float
    re: float
    f: float
    c: float  # adiabatic sound speed c_s, ft/s
    mach: float  # v / c_s
    grav: float
    fric: float
    tpr: float
    det: float  # 1 - v^2/c_s^2 (D / cp)
    dp_ds: float  # psi/ft numerator: dp/dx = dp_ds / det
    dt_ds: float  # degF/ft numerator: dT/dx = dt_ds / det
    h_total: float  # h/M + v^2/(2 g_c J), Btu/lbm (potential term excluded)
    cp_mass: float  # Btu/(lbm degR)

    @property
    def ek(self) -> float:
        return 1.0 - self.det


def gas_gradient_adiabatic(pvt: "PVTModel", p: float, t: float, w: float, seg: "FlowSegment") -> AdiabaticGradient:
    """Adiabatic momentum-energy gradient terms at (p, t) for mass rate ``w`` (lbm/s)."""
    st = pvt.state(p, t, True) if isinstance(pvt, DryGasPVT) else pvt.state(p, t)  # warm DAK start
    th = thermo_from_state(pvt, st)
    rho, mu = st.rho_g, st.mu_g
    v = w / (rho * seg.area)
    d = seg.d_ft
    re = reynolds_number(rho, v, d, mu)
    f = darcy_friction_factor(re, seg.rel_rough)
    grav = rho * seg.sin_theta / 144.0
    fric = f * rho * v * v / (2.0 * G_C * d) / 144.0
    m = M_AIR * pvt.gas_sg
    cp = th.cp / m
    a = v * v * th.rho_p / (144.0 * G_C)
    b = v * v * th.rho_t / (144.0 * G_C)
    k144 = 144.0 / (rho * J)
    gf = grav + fric
    pot = seg.sin_theta / J
    e_p = cp * th.mu_jt + k144 * a  # energy-equation coefficient of p' (negated)
    e_t = cp - k144 * b  # energy-equation coefficient of T'
    det = ((1.0 - a) * e_t - b * e_p) / cp
    n_p = -gf * e_t - b * pot
    n_t = -(1.0 - a) * pot - gf * e_p
    mach = v / th.c_s
    h_total = th.h / m + v * v / (2.0 * G_C * J)
    return AdiabaticGradient(
        p=p, t=t, rho=rho, v=v, mu=mu, re=re, f=f, c=th.c_s, mach=mach, grav=grav, fric=fric, tpr=st.tpr,
        det=det, dp_ds=-n_p / cp, dt_ds=-n_t / cp, h_total=h_total, cp_mass=cp,
    )

    @property
    def dpdx(self) -> float:
        """Total pressure gradient along MD, psi/ft, positive downward. Infinite at Mach 1."""
        return self.dp_ds / self.det

    @property
    def dtdx(self) -> float:
        """Temperature gradient along MD, degF/ft. Infinite at Mach 1."""
        return self.dt_ds / self.det


def gas_gradient(
    pvt: "PVTModel", p: float, t: float, w: float, seg: "FlowSegment", k: float | None = None
) -> GasGradient:
    """Gradient terms at (p, t) for mass rate ``w`` (lbm/s) in segment ``seg``. ``k`` None
    means the PVT model's ideal-gas cp0/cv0 at ``t``."""
    st = pvt.state(p, t)
    rho, mu = st.rho_g, st.mu_g
    v = w / (rho * seg.area)
    d = seg.d_ft
    re = reynolds_number(rho, v, d, mu)
    f = darcy_friction_factor(re, seg.rel_rough)
    c = sonic_velocity(pvt, p, t, k, state=st)  # reuses the Z solve in st
    grav = rho * seg.sin_theta / 144.0
    fric = f * rho * v * v / (2.0 * G_C * d) / 144.0
    return GasGradient(p=p, t=t, rho=rho, v=v, mu=mu, re=re, f=f, c=c, mach=v / c, grav=grav, fric=fric,
                       tpr=st.tpr if st.tpr is not None else 0.0)
