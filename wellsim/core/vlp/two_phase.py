"""Two-phase (gas-liquid) adiabatic march, M3-2: steady state, given gas and liquid
mass rates, no inflow and no nodal solve (M3-3).

Plugs into the same call sites as the gas march: an exit state (ambient or choked), the
optional exit-restriction entry, then ``march()`` downward from the exit with a ``model``
hook that supplies the gradient and the sudden area changes. When ``case.liquid`` is
absent or the liquid rate is zero, ``outflow_profile`` calls the single-phase gas path
unchanged (hand-off).

Per step (COMPUTED, our choice unless noted), with alpha_g from the M3-1 drift-flux
closure at the local (v_sg, v_sl, rho_g, rho_l, mu_l, sigma, d_o, d_i, theta), rho_g from
DAK, rho_l constant:

    rho_m  = alpha_g rho_g + (1 - alpha_g) rho_l
    mu_m   = alpha_g mu_g + (1 - alpha_g) mu_l           (in-situ weighted)
    v_m    = v_sg + v_sl,  v_g = v_sg / alpha_g,  v_l = v_sl / (1 - alpha_g)
    hydrostatic  rho_m sin(theta) / 144                                    psi/ft
    friction     f rho_m v_m^2 / (2 g_c D_h) / 144, Moody f at Re_m = rho_m v_m D_h / mu_m
                 (Hasan and Kabir also use the mixture density in friction; MEMORY)
    acceleration (M3-2b) the drift-flux momentum flux with the same phase velocities as the
                 energy equation,
                     (1/(144 g_c)) d/dL [alpha_g rho_g v_g^2 + (1 - alpha_g) rho_l v_l^2]
                     = (1/(144 g_c)) [G_g dv_g/dL + G_l dv_l/dL],   G_g = w_g/A, G_l = w_l/A
                 with dv_g/dL = -v_g (1/rho_g + alpha'/alpha_g) drho_g/dL and
                 dv_l/dL = v_l alpha'/(1 - alpha_g) drho_g/dL (rho_l constant, alpha' =
                 dalpha_g/drho_g). Forced to no slip (alpha_g = v_sg/v_m) this is exactly the
                 homogeneous rho_m v_m dv_m/dL (tested to 1e-10).

Energy: adiabatic total-enthalpy balance per lbm of mixture, mass-flow weighted,

    H = x_g h_g(p, T) + x_l h_l(p, T) + [x_g v_g^2 + x_l v_l^2] / (2 g_c J) + g z / (g_c J)
    h_l = cp_l (T - T_ref) + 144 (p - p_ref) / (rho_l J)      (incompressible liquid)

with h_g from the DAK departures (real_gas_thermo). The march keeps the 2x2 system in
(p', T') along the flow direction L: the kinetic and momentum terms are linearised with
drho_g/dL = rho_p p' + rho_T T' and dalpha_g/dL = (dalpha_g/drho_g) drho_g/dL, the holdup
sensitivity being the closure's own analytic derivative at fixed mass rates (implicit
differentiation, perf step 2; the closure has no p or T dependence other than through
rho_g and v_sg). No third unknown.

    momentum  (1 - a) p' - b T' = -(grav + fric),
              a = M rho_p, b = M rho_T,
              M = [G_g v_g (1/rho_g + alpha'/alpha_g) - G_l v_l alpha'/(1 - alpha_g)] / (144 g_c)
    energy    -e_p p' + e_t T' = -sin(theta) / J
              e_p = x_g cp_g mu_JT - x_l 144 / (rho_l J) - K rho_p
              e_t = x_g cp_g + x_l cp_l + K rho_T
              K   = [ -x_g v_g^2 (1/rho_g + alpha'/alpha_g) + x_l v_l^2 alpha' / (1 - alpha_g) ] / (g_c J)

Cramer's rule as in the gas gradient, normalised by cp_mix = x_g cp_g + x_l cp_l, so the
gas limit (x_l -> 0, alpha_g -> 1) reproduces ``gas_gradient_adiabatic`` term by term.

Choking (MEMORY, Wallis 1969, homogeneous frozen sound speed):

    1 / (rho_m c_m^2) = alpha_g / (rho_g c_g^2) + (1 - alpha_g) / (rho_l c_l^2)

with c_g the adiabatic real-gas sound speed and c_l from the schema (default 4,900 ft/s,
warned). c_m and Mach_wallis = v_m / c_m are computed and reported at every step and at
the exit.

The choke itself is applied exactly as the gas march does it: at the point where the
2x2 system above becomes singular (det = 0), which for the gas is the identity
det = 1 - v^2/c_s^2. For the mixture the singular speed lies below the Wallis frozen
speed (COMPUTED; at the 6 in choke check Mach_wallis is 0.91 at the singular point).
The main reason is thermal equilibrium: the 2x2 system carries one temperature, so
when the liquid heat capacity dominates the gas expands nearly isothermally and the
mixture's characteristic speed tends to the isothermal-gas value, whereas Wallis
assumes a frozen, adiabatic gas (1/sqrt(k) = 0.88 for this gas; the 6d exit is
annular with no slip in the closure, and the M3-2b slip momentum flux left its
numbers unchanged to four figures). Slip adds a smaller shift in the slug and bubbly
regimes. A march started at Mach_wallis = 1 would begin inside its own supersonic
region, so the exit and restriction solves find det = 0 instead, the reported ``mach``
is sqrt(1 - det) (1 at the choke), and Mach_wallis is carried alongside. The march's
Mach > 1 check on a downward step raises SupersonicError as for the gas; nothing is
clamped.

Sudden area changes use the mixture forms of the gas treatment (COMPUTED): flow-direction
expansion by the Borda-Carnot momentum balance with the same slip momentum flux
w_g v_g + w_l v_l on both sides of the step (M3-2b) and the mixture total enthalpy,
choked at the singular point when no subsonic root exists; flow-
direction contraction by the Crane loss K_c rho_m v_m^2 / 2 on the mixture velocity head
followed by an isentropic deceleration on the mixture entropy
s = x_g s_g(p, T) + x_l cp_l ln(T / T_ref).

Sandface: both phases enter after an isenthalpic throttle from (p_res, T_res) (the gas
code's ``sandface_jt``); for an incompressible liquid that is a temperature rise of
144 (p_res - p_wf) / (rho_l J cp_l).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wellsim.core.exit import (
    G_C,
    NOT_EVALUABLE_CEILING,
    ExitState,
    _T_FLOOR_DEGF,
    gas_mass_rate,
    is_gas_state,
)
from wellsim.core.friction import darcy_friction_factor, reynolds_number
from wellsim.core.numerics import brent
from wellsim.core.pvt.base import M_AIR, P_SC, rankine
from wellsim.core.pvt.dry_gas import DryGasPVT
from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU as J, T_REF, entropy_from_state, thermo_from_state
from wellsim.core.vlp.area_change import AreaChange, AreaChangeNotEvaluable, P_FLOOR, crane_contraction_coefficient
from wellsim.core.vlp.holdup import holdup
from wellsim.core.vlp.march import InternalChoke, MarchResult, SupersonicError, march

if TYPE_CHECKING:
    from wellsim.cases.schema import Case, Liquid
    from wellsim.core.geometry import FlowSegment

__all__ = [
    "NAME", "DEFAULT_C_L_FT_S", "LiquidProps", "Geometry", "MixtureState", "TwoPhaseGradient",
    "TwoPhaseExitState", "TwoPhaseModel", "Outflow", "mixture_state", "two_phase_gradient",
    "temperature_from_total_enthalpy_mix", "mach_mix", "sonic_pressure_mix", "exit_state_mix",
    "expansion_mix", "contraction_mix", "outflow_profile", "liquid_mass_rate_gpm", "liquid_mass_rate_stbd",
]

NAME = "two_phase_drift_flux"
DEFAULT_C_L_FT_S = 4900.0  # owner's default; water at ambient is about 4,860 ft/s (MEMORY)
CLOSURE = holdup  # the holdup closure; tests swap in a no-slip closure to check the momentum identity
_FT3_PER_GAL = 0.133680556
_FT3_PER_BBL = 5.614583


def liquid_mass_rate_gpm(q_gpm: float, rho_l: float) -> float:
    """lbm/s from gal/min at the in-situ liquid density (lbm/ft3)."""
    return q_gpm * _FT3_PER_GAL / 60.0 * rho_l


def liquid_mass_rate_stbd(q_stbd: float, rho_l: float) -> float:
    """lbm/s from stb/d at the liquid density (lbm/ft3); no shrinkage (rho_l is constant)."""
    return q_stbd * _FT3_PER_BBL / 86400.0 * rho_l


@dataclass(frozen=True)
class LiquidProps:
    rho_l: float  # lbm/ft3
    mu_l: float  # cp
    cp_l: float  # Btu/(lbm degF)
    sigma: float  # dyne/cm
    c_l: float  # ft/s
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_schema(cls, liquid: "Liquid") -> "LiquidProps":
        sigma, w1 = liquid.sigma_or_default()
        c_l, w2 = liquid.c_l_or_default()
        return cls(liquid.rho_l_lbm_ft3, liquid.mu_l, liquid.cp_l, sigma, c_l, tuple(w for w in (w1, w2) if w))

    def h_l(self, p: float, t: float) -> float:
        """Liquid enthalpy, Btu/lbm: cp_l (T - T_ref) + 144 (p - p_ref)/(rho_l J) (COMPUTED)."""
        return self.cp_l * (t - T_REF) + 144.0 * (p - P_SC) / (self.rho_l * J)

    def s_l(self, t: float) -> float:
        """Liquid entropy, Btu/(lbm degR): cp_l ln(T/T_ref), incompressible (COMPUTED)."""
        return self.cp_l * math.log(rankine(t) / rankine(T_REF))


@dataclass(frozen=True)
class Geometry:
    """What the closure and the gradient need from a conduit."""
    area: float  # ft2 (effective: cd x geometric at a restricted exit)
    d_hyd: float  # in
    d_o: float  # in
    d_i: float  # in, 0 = pipe
    sin_theta: float
    rel_rough: float
    conduit: str = ""

    @property
    def d_ft(self) -> float:
        return self.d_hyd / 12.0

    @property
    def theta_deg(self) -> float:
        return math.degrees(math.asin(min(1.0, max(0.0, self.sin_theta))))


def geometry_of(seg: "FlowSegment") -> Geometry:
    return Geometry(seg.area, seg.d_hyd, seg.d_o, seg.d_i, seg.sin_theta, seg.rel_rough, seg.conduit)


@dataclass(frozen=True, slots=True)
class MixtureState:
    p: float
    t: float
    st: object  # gas PVTState
    th: object  # ThermoState
    rho_g: float
    mu_g: float
    v_sg: float
    v_sl: float
    v_m: float
    alpha: float
    regime: str
    rho_m: float
    mu_m: float
    v_g: float
    v_l: float
    c_g: float
    c_m: float
    mach: float
    x_g: float
    cp_g: float  # Btu/(lbm degR), gas
    cp_mix: float
    h_mix: float  # Btu/lbm of mixture
    ke: float  # Btu/lbm
    h_total: float  # h_mix + ke (potential term excluded)
    alpha_rho: float  # dalpha_g/drho_g at fixed mass rates, ft3/lbm
    k_kin: float  # K in the energy equation, Btu/lbm per lbm/ft3
    iterations: int
    m_coef: float  # M: slip momentum-flux derivative coefficient, ft3 psi/lbm per (lbm/ft3 per ft)... see docstring
    m_coef_homogeneous: float  # rho_m v_m v_sg/(rho_g 144 g_c): the pre-M3-2b homogeneous form (diagnostic)
    a: float  # momentum acceleration coefficient of p' (M rho_p)
    b: float  # momentum acceleration coefficient of T' (M rho_T)
    e_p: float  # energy coefficient of p' (negated)
    e_t: float  # energy coefficient of T'
    det: float  # 2x2 determinant / cp_mix: the march's singular point is det = 0
    mach_ode: float  # sqrt(1 - det): 1 where the system is singular (the choke criterion)


def _gas_state(pvt, p, t):
    # g2-4: `warm` is in the base signature, so no isinstance check is needed. Models that
    # do not iterate on Z ignore it. The march asks for it exactly as it did before.
    return pvt.state(p, t, True)


def mixture_state(pvt, liq: LiquidProps, p: float, t: float, w_g: float, w_l: float, geom: Geometry,
                  sensitivity: bool = True, closure=None) -> MixtureState:
    """All mixture properties at (p, t) for the mass rates in ``geom``. ``closure`` defaults
    to the module's CLOSURE (the drift-flux holdup)."""
    closure = CLOSURE if closure is None else closure
    st = _gas_state(pvt, p, t)
    th = pvt.thermo(p, t, st)   # st is reused: one Z solve per gradient, as before
    rho_g, mu_g = st.rho_g, st.mu_g
    area = geom.area
    v_sg = w_g / (rho_g * area)
    v_sl = w_l / (liq.rho_l * area)
    theta = geom.theta_deg
    hd = closure(v_sg, v_sl, rho_g, liq.rho_l, liq.mu_l, liq.sigma, geom.d_o, geom.d_i, theta)
    alpha = hd.alpha_g
    rho_m = alpha * rho_g + (1.0 - alpha) * liq.rho_l
    mu_m = alpha * mu_g + (1.0 - alpha) * liq.mu_l
    v_m = v_sg + v_sl
    v_g = v_sg / alpha if alpha > 0.0 else 0.0
    v_l = v_sl / (1.0 - alpha) if alpha < 1.0 else 0.0
    c_g = th.c_s
    inv = alpha / (rho_g * c_g * c_g) + (1.0 - alpha) / (liq.rho_l * liq.c_l * liq.c_l)
    c_m = math.sqrt(1.0 / (rho_m * inv))
    w = w_g + w_l
    x_g = w_g / w
    m = pvt.molar_mass
    cp_g = th.cp / m
    cp_mix = x_g * cp_g + (1.0 - x_g) * liq.cp_l
    h_mix = x_g * th.h / m + (1.0 - x_g) * liq.h_l(p, t)
    ke = (x_g * v_g * v_g + (1.0 - x_g) * v_l * v_l) / (2.0 * G_C * J)
    # perf step 2: the closure's analytic dalpha_g/drho_g at fixed mass rates (implicit
    # differentiation, COMPUTED); the Richardson difference it replaces lives in the tests
    alpha_rho = hd.dalpha_drho_g if (sensitivity and w_g > 0.0 and w_l > 0.0) else 0.0
    # phase-velocity derivatives per unit drho_g: dv_g = -v_g (1/rho_g + alpha'/alpha) drho_g,
    # dv_l = v_l alpha'/(1 - alpha) drho_g (rho_l constant)
    if 0.0 < alpha < 1.0:
        dvg = -v_g * (1.0 / rho_g + alpha_rho / alpha)
        dvl = v_l * alpha_rho / (1.0 - alpha)
    elif alpha >= 1.0:
        dvg, dvl = -v_g / rho_g, 0.0
    else:
        dvg, dvl = 0.0, 0.0
    k_kin = (x_g * v_g * dvg + (1.0 - x_g) * v_l * dvl) / (G_C * J)
    # the 2x2 momentum-energy coefficients and the determinant (no gravity or friction in it);
    # M3-2b: slip momentum flux G_g v_g + G_l v_l, derivative per unit drho_g
    g_g, g_l = w_g / area, w_l / area
    m_coef = -(g_g * dvg + g_l * dvl) / (144.0 * G_C)
    m_coef_homogeneous = rho_m * v_m * v_sg / (rho_g * 144.0 * G_C)
    a = m_coef * th.rho_p
    b = m_coef * th.rho_t
    x_l = 1.0 - x_g
    e_p = x_g * cp_g * th.mu_jt - x_l * 144.0 / (liq.rho_l * J) - k_kin * th.rho_p
    e_t = x_g * cp_g + x_l * liq.cp_l + k_kin * th.rho_t
    det = ((1.0 - a) * e_t - b * e_p) / cp_mix
    mach_ode = math.sqrt(1.0 - det) if det < 1.0 else 0.0
    return MixtureState(
        p=p, t=t, st=st, th=th, rho_g=rho_g, mu_g=mu_g, v_sg=v_sg, v_sl=v_sl, v_m=v_m, alpha=alpha,
        regime=hd.regime, rho_m=rho_m, mu_m=mu_m, v_g=v_g, v_l=v_l, c_g=c_g, c_m=c_m, mach=v_m / c_m,
        x_g=x_g, cp_g=cp_g, cp_mix=cp_mix, h_mix=h_mix, ke=ke, h_total=h_mix + ke, alpha_rho=alpha_rho,
        k_kin=k_kin, iterations=hd.iterations, m_coef=m_coef, m_coef_homogeneous=m_coef_homogeneous,
        a=a, b=b, e_p=e_p, e_t=e_t, det=det, mach_ode=mach_ode,
    )


@dataclass(frozen=True, slots=True)
class TwoPhaseGradient:
    """Same interface as ``AdiabaticGradient`` (used by the march and its trace) plus the
    two-phase diagnostics."""
    p: float
    t: float
    rho: float  # rho_m
    v: float  # v_m
    mu: float  # mu_m
    re: float
    f: float
    c: float  # c_m, the Wallis frozen sound speed (diagnostic)
    mach: float  # sqrt(1 - det): the march's own Mach number, 1 at the singular point
    grav: float
    fric: float
    tpr: float
    det: float
    dp_ds: float
    dt_ds: float
    h_total: float
    cp_mass: float
    alpha_g: float
    regime: str
    rho_g: float
    v_sg: float
    v_sl: float
    v_g: float
    v_l: float
    c_g: float
    mach_wallis: float  # v_m / c_m (diagnostic)
    holdup_iterations: int

    @property
    def ek(self) -> float:
        return 1.0 - self.det

    @property
    def dp_dx(self) -> float:
        """psi/ft, the physical gradient. ``dp_ds`` is the NUMERATOR of the 2x2 solve, as
        in `single_phase_gas`: the march integrates (det, dp_ds, dt_ds) in arclength so it
        stays regular through the singular point, and dp/dx is dp_ds/det. Anything wanting
        a point value rather than a march step wants this (g2-4)."""
        return self.dp_ds / self.det

    @property
    def dt_dx(self) -> float:
        """degF/ft, the physical temperature gradient. See :attr:`dp_dx`."""
        return self.dt_ds / self.det


def two_phase_gradient(pvt, liq: LiquidProps, p: float, t: float, w_g: float, w_l: float,
                       seg: "FlowSegment", closure=None) -> TwoPhaseGradient:
    """``closure`` is forwarded to :func:`mixture_state`, which has always taken one; it is
    exposed here (g2-4) so the G2 gate can run its Beggs-Brill cross-check through this
    function instead of reimplementing the gradient around a different closure."""
    geom = geometry_of(seg)
    ms = mixture_state(pvt, liq, p, t, w_g, w_l, geom, closure=closure)
    d = geom.d_ft
    re = reynolds_number(ms.rho_m, ms.v_m, d, ms.mu_m)
    f = darcy_friction_factor(re, geom.rel_rough)
    grav = ms.rho_m * geom.sin_theta / 144.0
    fric = f * ms.rho_m * ms.v_m * ms.v_m / (2.0 * G_C * d) / 144.0
    gf = grav + fric
    pot = geom.sin_theta / J
    cp = ms.cp_mix
    n_p = -gf * ms.e_t - ms.b * pot
    n_t = -(1.0 - ms.a) * pot - gf * ms.e_p
    return TwoPhaseGradient(
        p=p, t=t, rho=ms.rho_m, v=ms.v_m, mu=ms.mu_m, re=re, f=f, c=ms.c_m, mach=ms.mach_ode, grav=grav, fric=fric,
        tpr=ms.st.tpr if ms.st.tpr is not None else 0.0, det=ms.det, dp_ds=-n_p / cp, dt_ds=-n_t / cp,
        h_total=ms.h_total, cp_mass=cp, alpha_g=ms.alpha, regime=ms.regime, rho_g=ms.rho_g, v_sg=ms.v_sg,
        v_sl=ms.v_sl, v_g=ms.v_g, v_l=ms.v_l, c_g=ms.c_g, mach_wallis=ms.mach, holdup_iterations=ms.iterations,
    )


# --- total enthalpy inversion, Mach, sonic pressure, exit ------------------------------------

def temperature_from_total_enthalpy_mix(pvt, liq, p, w_g, w_l, geom, h_total, t_guess):
    """T (degF) at ``p`` such that the mixture h + v^2/2 equals ``h_total`` (Btu/lbm,
    potential term removed). Newton with the analytic d/dT (cp_mix + K rho_T, the energy
    coefficient e_t), Brent fallback on a bracket."""

    def g_and_dg(t):
        ms = mixture_state(pvt, liq, p, t, w_g, w_l, geom)
        return ms.h_total - h_total, ms.x_g * ms.cp_g + (1.0 - ms.x_g) * liq.cp_l + ms.k_kin * ms.th.rho_t

    t = t_guess
    for _ in range(30):
        g, dg = g_and_dg(t)
        if dg <= 0.0:
            break
        dt = -g / dg
        if abs(dt) > 200.0:
            dt = math.copysign(200.0, dt)
        t += dt
        if not (t > _T_FLOOR_DEGF):
            break
        if abs(dt) < 1e-9:
            return t
    lo, hi = _T_FLOOR_DEGF, max(t_guess, 0.0) + 3000.0
    g_lo, g_hi = g_and_dg(lo)[0], g_and_dg(hi)[0]
    if g_lo >= 0.0 or g_hi <= 0.0:
        raise ValueError(f"no physical temperature at p = {p:.2f} psia for this total enthalpy and mass rates")
    return brent(lambda tt: g_and_dg(tt)[0], lo, hi, fa=g_lo, fb=g_hi, xtol=1e-9, rtol=1e-12)


def mach_mix(pvt, liq, p, w_g, w_l, geom, h_total, t_guess):
    """(Mach_m, MixtureState or None, T) at ``p`` with T from the total enthalpy. A state
    outside the gas region (for the gas phase) counts as Mach +inf, as in the gas code."""
    try:
        t = temperature_from_total_enthalpy_mix(pvt, liq, p, w_g, w_l, geom, h_total, t_guess)
        if not (t > _T_FLOOR_DEGF):
            return math.inf, None, t_guess
        ms = mixture_state(pvt, liq, p, t, w_g, w_l, geom)
        if w_g > 0.0 and not is_gas_state(ms.st):
            return math.inf, None, t_guess
    except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
        return math.inf, None, t_guess
    return ms.mach_ode, ms, t


def sonic_pressure_mix(pvt, liq, w_g, w_l, geom, p_lo, h_total, t_guess, p_max=math.inf):
    """(p*, T*) at which the mixture is exactly Mach_m = 1 in ``geom``; ``p_lo`` must give
    Mach >= 1. None when Mach is still above 1 at ``p_max``. Mirrors sonic_pressure_adiabatic."""
    ma0, _, t0 = mach_mix(pvt, liq, p_lo, w_g, w_l, geom, h_total, t_guess)
    if ma0 < 1.0:
        raise ValueError(f"sonic_pressure_mix: Mach {ma0:.4f} < 1 at p_lo {p_lo:.2f} psia; nothing to solve")
    if ma0 == 1.0:
        return p_lo, t0
    t_last = [t0]

    def g(p):
        ma, _, t = mach_mix(pvt, liq, p, w_g, w_l, geom, h_total, t_last[0])
        t_last[0] = t
        return ma - 1.0

    a, fa = p_lo, ma0 - 1.0
    b = min(2.0 * p_lo, p_max)
    while True:
        fb = g(b)
        if fb <= 0.0:
            break
        if b >= p_max:
            return None
        a, fa = b, fb
        b = min(2.0 * b, p_max)
    for _ in range(80):
        if math.isfinite(fa):
            break
        mid = 0.5 * (a + b)
        fm = g(mid)
        if fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    p_star = brent(g, a, b, fa=fa, fb=fb, xtol=1e-6, rtol=1e-10)
    _, _, t_star = mach_mix(pvt, liq, p_star, w_g, w_l, geom, h_total, t_last[0])
    return p_star, t_star


@dataclass(frozen=True)
class TwoPhaseExitState(ExitState):
    """``mach`` is the march's own Mach number sqrt(1 - det) (1 when choked); ``c`` is the
    Wallis frozen sound speed c_m and ``mach_wallis`` = v_m / c_m, both diagnostics."""
    alpha_g: float = math.nan
    regime: str = ""
    c_g: float = math.nan
    c_l: float = math.nan
    v_sg: float = math.nan
    v_sl: float = math.nan
    mach_wallis: float = math.nan

    def as_dict(self) -> dict:
        d = super().as_dict()
        d.update({"alpha_g": self.alpha_g, "regime": self.regime, "c_g": self.c_g, "c_l": self.c_l,
                  "v_sg": self.v_sg, "v_sl": self.v_sl, "mach_wallis": self.mach_wallis, "two_phase": True})
        return d


def _exit_from(ms: MixtureState, p_amb, area, choked, liq) -> TwoPhaseExitState:
    return TwoPhaseExitState(ms.p, ms.t, p_amb, ms.rho_m, ms.v_m, ms.c_m, ms.mach_ode, choked, area,
                             alpha_g=ms.alpha, regime=ms.regime, c_g=ms.c_g, c_l=liq.c_l, v_sg=ms.v_sg,
                             v_sl=ms.v_sl, mach_wallis=ms.mach)


def exit_state_mix(pvt, liq, w_g, w_l, p_amb, geom, h_total, t_guess, p_max=math.inf) -> TwoPhaseExitState:
    """Exit state for the mixture discharging into ``p_amb``: ambient when Mach_m <= 1
    there, else the pressure at Mach_m = 1; ``not_evaluable`` when no sonic state exists
    below ``p_max``."""
    ma, ms, t = mach_mix(pvt, liq, p_amb, w_g, w_l, geom, h_total, t_guess)
    if ma <= 1.0:
        return _exit_from(ms, p_amb, geom.area, False, liq)
    sonic = sonic_pressure_mix(pvt, liq, w_g, w_l, geom, p_amb, h_total, t, p_max)
    if sonic is None:
        return TwoPhaseExitState(math.nan, math.nan, p_amb, math.nan, math.nan, math.nan, math.inf, True, geom.area,
                                 not_evaluable=NOT_EVALUABLE_CEILING, c_l=liq.c_l)
    p_star, t_star = sonic
    _, ms, _ = mach_mix(pvt, liq, p_star, w_g, w_l, geom, h_total, t_star)
    return _exit_from(ms, p_amb, geom.area, True, liq)


# --- sudden area changes on the mixture ---------------------------------------------------------

def _sonic_or_floor_mix(pvt, liq, w_g, w_l, geom, h_total, t0, p_max):
    ma0 = mach_mix(pvt, liq, P_FLOOR, w_g, w_l, geom, h_total, t0)[0]
    if ma0 < 1.0:
        return P_FLOOR, None
    sonic = sonic_pressure_mix(pvt, liq, w_g, w_l, geom, P_FLOOR, h_total, t0, p_max)
    if sonic is None:
        raise AreaChangeNotEvaluable("no subsonic mixture state in the narrower conduit below the reservoir-pressure ceiling")
    return sonic


def _state_energy(pvt, liq, p, w_g, w_l, geom, h_total, t0):
    t = temperature_from_total_enthalpy_mix(pvt, liq, p, w_g, w_l, geom, h_total, t0)
    ms = mixture_state(pvt, liq, p, t, w_g, w_l, geom)
    if w_g > 0.0 and not is_gas_state(ms.st):
        raise ValueError("gas phase outside the gas region")
    return ms


def expansion_mix(pvt, liq, w_g, w_l, geom_large, geom_small, p_large, t_large, h_total, p_max):
    """Small-pipe mixture state (p_s, T_s, Mach_s, choked, residual) below a flow-direction
    expansion: Borda-Carnot with the slip momentum flux w_g v_g + w_l v_l on both sides
    (COMPUTED, M3-2b)."""
    ms_l = mixture_state(pvt, liq, p_large, t_large, w_g, w_l, geom_large, sensitivity=False)
    rhs = p_large * geom_large.area * 144.0 + (w_g * ms_l.v_g + w_l * ms_l.v_l) / G_C

    def residual(p_s):
        try:
            ms = _state_energy(pvt, liq, p_s, w_g, w_l, geom_small, h_total, t_large)
        except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
            raise ValueError("state outside the gas region")
        return p_s * geom_large.area * 144.0 + (w_g * ms.v_g + w_l * ms.v_l) / G_C - rhs, ms

    p_star, t_star = _sonic_or_floor_mix(pvt, liq, w_g, w_l, geom_small, h_total, t_large, p_max)
    f_star, ms_star = residual(p_star)
    if f_star > 0.0:
        if t_star is None:
            raise AreaChangeNotEvaluable("momentum balance across the expansion has no root above 1 psia")
        return p_star, ms_star.t, ms_star.mach, True, f_star
    a, fa = p_star, f_star
    b = min(2.0 * p_star, p_max)
    while True:
        fb = residual(b)[0]
        if fb >= 0.0:
            break
        if b >= p_max:
            raise AreaChangeNotEvaluable("momentum balance across the expansion not met below the reservoir-pressure ceiling")
        a, fa = b, fb
        b = min(2.0 * b, p_max)
    p_s = brent(lambda pp: residual(pp)[0], a, b, fa=fa, fb=fb, xtol=1e-7, rtol=1e-12)
    f_s, ms_s = residual(p_s)
    return p_s, ms_s.t, ms_s.mach, False, f_s


def _entropy_mix(pvt, liq, ms: MixtureState) -> float:
    m = M_AIR * pvt.gas_sg
    return ms.x_g * entropy_from_state(pvt, ms.st) / m + (1.0 - ms.x_g) * liq.s_l(ms.t)


def contraction_mix(pvt, liq, w_g, w_l, geom_small, geom_large, p_small, t_small, h_total, p_max, k_c):
    """Large-pipe mixture state (p_L, T_L, Mach_L, loss_psi, residual) below a flow-direction
    contraction: undo K_c rho_m v_m^2/2 at constant total enthalpy, then decelerate on the
    mixture entropy (COMPUTED)."""
    ms_s = mixture_state(pvt, liq, p_small, t_small, w_g, w_l, geom_small, sensitivity=False)
    loss = k_c * ms_s.rho_m * ms_s.v_m * ms_s.v_m / (2.0 * G_C * 144.0)
    p_s2 = p_small + loss
    ms_s2 = _state_energy(pvt, liq, p_s2, w_g, w_l, geom_small, h_total, t_small)
    s_target = _entropy_mix(pvt, liq, ms_s2)

    def residual(p_l):
        ms = _state_energy(pvt, liq, p_l, w_g, w_l, geom_large, h_total, ms_s2.t)
        return _entropy_mix(pvt, liq, ms) - s_target, ms

    a, fa = p_s2, residual(p_s2)[0]
    b = min(2.0 * p_s2, p_max)
    while True:
        fb = residual(b)[0]
        if fa * fb <= 0.0:
            break
        if b >= p_max:
            raise AreaChangeNotEvaluable("isentropic deceleration of the mixture has no solution below the reservoir-pressure ceiling")
        a, fa = b, fb
        b = min(2.0 * b, p_max)
    p_l = brent(lambda pp: residual(pp)[0], a, b, fa=fa, fb=fb, xtol=1e-7, rtol=1e-12)
    f_l, ms_l = residual(p_l)
    return p_l, ms_l.t, ms_l.mach, loss, f_l


# --- march model hook ----------------------------------------------------------------------------

@dataclass
class TwoPhaseModel:
    """The ``model`` hook for ``march()``: gradient and sudden area changes on the mixture."""
    pvt: object
    liq: LiquidProps
    w_g: float
    w_l: float
    name: str = NAME

    def gradient(self, seg, x, p, t) -> TwoPhaseGradient:
        # g.mach = sqrt(1 - det): the march's own Mach > 1 check catches a singular system
        return two_phase_gradient(self.pvt, self.liq, p, t, self.w_g, self.w_l, seg)

    def area_change(self, prev_seg, seg, x, p, t, h_local, p_max, k_c):
        g_prev, g_new = geometry_of(prev_seg), geometry_of(seg)
        if seg.area < prev_seg.area:
            p_new, t_new, mach_new, choked, resid = expansion_mix(
                self.pvt, self.liq, self.w_g, self.w_l, g_prev, g_new, p, t, h_local, p_max)
            choke = None
            if choked:
                mach_before = mach_mix(self.pvt, self.liq, p, self.w_g, self.w_l, g_new, h_local, t)[0]
                choke = InternalChoke(x, p, p_new, mach_before, seg.conduit)
            ac = AreaChange(x, "expansion", prev_seg.conduit, seg.conduit, p, t, p_new, t_new, mach_new, choked, 0.0, resid)
            return p_new, t_new, ac, choke
        kc = crane_contraction_coefficient(prev_seg.d_hyd, seg.d_hyd) if k_c is None else k_c
        p_new, t_new, mach_new, loss, resid = contraction_mix(
            self.pvt, self.liq, self.w_g, self.w_l, g_prev, g_new, p, t, h_local, p_max, kc)
        ac = AreaChange(x, "contraction", prev_seg.conduit, seg.conduit, p, t, p_new, t_new, mach_new, False, loss, resid)
        return p_new, t_new, ac, None


# --- driver: exit state, restriction entry, march; gas hand-off -----------------------------------

@dataclass
class Outflow:
    exit: ExitState
    profile: MarchResult | None
    two_phase: bool
    h_total: float | None = None
    entry: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    w_gas: float = 0.0
    w_liquid: float = 0.0
    # M3-4 injection bookkeeping
    w_kill: float = 0.0
    md_inj: float | None = None
    p_inj: float | None = None
    t_inj: float | None = None
    junction_residual_degf: float | None = None  # perf step 2: arrival T minus the balance's T at z_inj

    @property
    def pwf(self) -> float | None:
        return self.profile.p_end if self.profile is not None and self.profile.ok else None


def _gas_outflow(case: "Case", pvt, q_gas, segments, a_exit, a_exit_geom, cd, restricted_exit, d_exit_in,
                 p_in) -> Outflow:
    """The solver's gas branch, unchanged (M3-2 hand-off)."""
    from dataclasses import replace

    from wellsim.core.exit import exit_state, exit_state_adiabatic
    from wellsim.core.pvt.real_gas_thermo import enthalpy
    from wellsim.core.temperature import make_temperature
    from wellsim.core.vlp.area_change import contraction_with_loss

    res, num, tm = case.reservoir, case.options.numerics, case.temperature
    p_amb = case.exit.ambient_pressure
    k_ratio = case.exit.k_override
    adiabatic = tm.model == "adiabatic" and k_ratio is None
    w = gas_mass_rate(q_gas, pvt.gas_sg)
    m_gas = M_AIR * pvt.gas_sg
    tvd_res = case.well.md_to_tvd(res.depth_md)
    out = Outflow(exit=None, profile=None, two_phase=False, w_gas=w)
    if adiabatic:
        p_h = res.pressure if (tm.sandface_jt or p_in is None) else p_in
        ht = enthalpy(pvt, p_h, res.temperature) / m_gas - tvd_res / J
        out.h_total = ht
        ex = exit_state_adiabatic(pvt, q_gas, p_amb, a_exit, ht, res.temperature - 40.0, p_max=res.pressure)
        ex = replace(ex, cd=cd, area_geometric=a_exit_geom)
        out.exit = ex
        if ex.not_evaluable is not None:
            return out
        p0, t0 = ex.p, ex.t
        if restricted_exit:
            kc = crane_contraction_coefficient(d_exit_in, segments[0].d_hyd) if cd == 1.0 else 0.0
            try:
                p0, t0, mach0, loss, resid = contraction_with_loss(
                    pvt, w, a_exit, segments[0].area, ex.p, ex.t, ht, res.pressure, kc)
            except AreaChangeNotEvaluable as e:
                out.warnings.append(f"exit restriction entry: {e}")
                return out
            out.entry.update({"k_c": kc, "loss_psi": loss, "p_throat": ex.p, "t_throat": ex.t,
                              "p_conduit": p0, "t_conduit": t0, "mach_conduit": mach0, "residual": resid})
        out.profile = march(
            pvt, segments, None, w, p0, 0.0, res.depth_md,
            max_step=num.vlp_max_segment, p_tol=num.vlp_pressure_tol,
            adiabatic=True, t_start=t0, t_tol=num.vlp_temperature_tol, p_max=res.pressure,
        )
    else:
        temperature = make_temperature(case)
        t_exit = temperature.at(0.0)
        ex = replace(exit_state(pvt, q_gas, p_amb, t_exit, a_exit, k_ratio), cd=cd, area_geometric=a_exit_geom)
        out.exit = ex
        out.profile = march(
            pvt, segments, temperature, w, ex.p, 0.0, res.depth_md,
            max_step=num.vlp_max_segment, p_tol=num.vlp_pressure_tol, k_ratio=k_ratio,
        )
    return out


def march_leg(case: "Case", pvt, liq: LiquidProps, w_g: float, w_l: float, p_start: float, t_start: float,
              md_start: float, md_end: float) -> MarchResult:
    """A two-phase march from a known state (M3-4: the leg below an injection point)."""
    from wellsim.core.geometry import flow_path_segments

    res, num = case.reservoir, case.options.numerics
    segments = flow_path_segments(case.well, case.flow_path, res.depth_md)
    model = TwoPhaseModel(pvt, liq, w_g, w_l)
    return march(
        pvt, segments, None, w_g + w_l, p_start, md_start, md_end,
        max_step=num.vlp_max_segment, p_tol=num.vlp_pressure_tol,
        adiabatic=True, t_start=t_start, t_tol=num.vlp_temperature_tol, p_max=res.pressure, model=model,
    )


def outflow_profile(case: "Case", pvt, q_gas_mscfd: float, w_liquid: float, *, p_in: float | None = None,
                    liquid: LiquidProps | None = None, h_total: float | None = None,
                    md_end: float | None = None) -> Outflow:
    """Exit state and downward march for a gas rate (Mscf/d) and a liquid mass rate (lbm/s).
    No inflow: the reservoir block supplies only the depth, the temperature and the
    pressure ceiling / throttle inlet state. ``p_in`` (psia) overrides the inlet pressure
    of the enthalpy datum when ``temperature.sandface_jt`` is False.

    Hand-off: with ``case.liquid`` absent (and no ``liquid`` override) or ``w_liquid`` <= 0
    this is the single-phase gas path of the solver, unchanged.

    M3-4 overrides: ``liquid`` replaces the schema liquid properties (a mixed stream above an
    injection point), ``h_total`` the total enthalpy at the surface datum, ``md_end`` the
    depth the march stops at. Omitted, the M3-3 arithmetic is unchanged."""
    from dataclasses import replace

    from wellsim.core.geometry import exit_area, flow_path_segments
    from wellsim.core.pvt.real_gas_thermo import enthalpy

    res, num, tm = case.reservoir, case.options.numerics, case.temperature
    p_amb = case.exit.ambient_pressure
    segments = flow_path_segments(case.well, case.flow_path, res.depth_md)
    a_exit_geom = exit_area(segments, case.exit.exit_id)
    cd = case.exit.cd
    a_exit = cd * a_exit_geom
    restricted_exit = case.exit.exit_id is not None and a_exit < segments[0].area
    d_exit_in = case.exit.exit_id if case.exit.exit_id is not None else segments[0].d_hyd
    if (case.liquid is None and liquid is None) or w_liquid <= 0.0:
        return _gas_outflow(case, pvt, q_gas_mscfd, segments, a_exit, a_exit_geom, cd, restricted_exit, d_exit_in, p_in)
    if tm.model != "adiabatic" or case.exit.k_override is not None:
        raise NotImplementedError("the two-phase march (M3-2) needs temperature.model 'adiabatic' without k_override")

    liq = liquid if liquid is not None else LiquidProps.from_schema(case.liquid)
    w_g = gas_mass_rate(q_gas_mscfd, pvt.gas_sg)
    w = w_g + w_liquid
    m_gas = M_AIR * pvt.gas_sg
    tvd_res = case.well.md_to_tvd(res.depth_md)
    p_h = res.pressure if (tm.sandface_jt or p_in is None) else p_in
    x_g = w_g / w
    h_in = x_g * (enthalpy(pvt, p_h, res.temperature) / m_gas if w_g > 0.0 else 0.0) \
        + (1.0 - x_g) * liq.h_l(p_h, res.temperature)
    ht = h_in - tvd_res / J if h_total is None else h_total
    out = Outflow(exit=None, profile=None, two_phase=True, h_total=ht, warnings=list(liq.warnings),
                  w_gas=w_g, w_liquid=w_liquid)
    top = segments[0]
    if restricted_exit:
        geom_exit = Geometry(a_exit, d_exit_in, d_exit_in, 0.0, top.sin_theta, top.rel_rough, f"exit {d_exit_in:g} in")
    else:
        geom_exit = replace(geometry_of(top), area=a_exit)
    ex = exit_state_mix(pvt, liq, w_g, w_liquid, p_amb, geom_exit, ht, res.temperature - 40.0, p_max=res.pressure)
    ex = replace(ex, cd=cd, area_geometric=a_exit_geom)
    out.exit = ex
    if ex.not_evaluable is not None:
        return out
    p0, t0 = ex.p, ex.t
    if restricted_exit:
        kc = crane_contraction_coefficient(d_exit_in, top.d_hyd) if cd == 1.0 else 0.0
        try:
            p0, t0, mach0, loss, resid = contraction_mix(
                pvt, liq, w_g, w_liquid, geom_exit, geometry_of(top), ex.p, ex.t, ht, res.pressure, kc)
        except AreaChangeNotEvaluable as e:
            out.warnings.append(f"exit restriction entry: {e}")
            return out
        out.entry.update({"k_c": kc, "loss_psi": loss, "p_throat": ex.p, "t_throat": ex.t,
                          "p_conduit": p0, "t_conduit": t0, "mach_conduit": mach0, "residual": resid})
    model = TwoPhaseModel(pvt, liq, w_g, w_liquid)
    out.profile = march(
        pvt, segments, None, w, p0, 0.0, res.depth_md if md_end is None else md_end,
        max_step=num.vlp_max_segment, p_tol=num.vlp_pressure_tol,
        adiabatic=True, t_start=t0, t_tol=num.vlp_temperature_tol, p_max=res.pressure, model=model,
    )
    if out.profile.ok:
        out.warnings.append(
            "two-phase critical flow uses the homogeneous frozen (Wallis) sound speed; "
            "Henry-Fauske or an equivalent is not implemented"
        ) if ex.choked or out.profile.chokes else None
    return out
