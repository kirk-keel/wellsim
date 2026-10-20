"""Drift-flux holdup closure (M3-1): gas fraction from a Zuber-Findlay slip model
with Hasan-Kabir regime coefficients, continuous across regime switches (M3-1c/d).

Standalone module: nothing in the march calls it yet (M3-2 integrates it into the
multiphase gradient). Inputs and outputs are in the march's oilfield units (ft/s,
lbm/ft3, cp, in); the surface tension is in dyne/cm. The correlations themselves are
evaluated in SI inside this module because every published constant (1.53, 0.35,
3.1, 4.0, ...) is stated in SI.

Zuber-Findlay (1965):

    v_g = C0 v_m + v_d          v_m = v_sg + v_sl
    alpha_g = v_sg / v_g        H_l = 1 - alpha_g

Regime coefficients (Hasan and Kabir 1988, SPEPE May 1988 p. 263, and the 1992
annulus paper; every line MEMORY unless noted, see docs/PROVENANCE.md, M3-1):

    bubbly     C0 = 1.2   v_d = v_H = 1.53 [g sigma (rho_l - rho_g) / rho_l^2]^0.25 sin theta
                          (Harmathy 1960; the sin theta is the axial component of the
                          rise velocity, Hasan-Kabir deviated wells 1988, MEMORY, NEEDS
                          HAND CHECK; added in M3-1b so that horizontal flow has v_d = 0
                          in every regime). Coded as the slug ramp below, which is v_H
                          wherever alpha_g <= 0.25, so v_d is one continuous function of
                          alpha_g from bubbly through churn.
    dispersed bubbly      C0 = 1.2, v_d = v_H
    slug       C0 = 1.2   v_d = v_H + (v_TB - v_H) min(1, max(0, (alpha_g - 0.25) / 0.27))
                          v_TB = v_TB,vertical sqrt(sin theta) (1 + cos theta)^1.2
                          pipe:    v_TB,vertical = 0.35 sqrt(g d_o (rho_l - rho_g) / rho_l)
                          annulus: v_TB,vertical = (0.35 + 0.10 d_i / d_o) sqrt(g d_o (rho_l - rho_g) / rho_l)
                                   (FETCHED: Hasan & Kabir 2nd ed. Eq. 4.75; hk-1)
    churn      C0 = 1.15  v_d as slug
    annular    C0 = 1.0   v_d = 0

theta is the inclination from horizontal in degrees (90 = vertical), so sin theta is the
march's sin_theta. Since hk-1 every coefficient above except the slug ramp is FETCHED
from Hasan & Kabir, *Fluid Flow and Heat Transfer in Wellbores*, 2nd ed. directly: the
bubbly C0 and Harmathy (sec. 3.3.1, Eq. 3.6), the pipe Taylor 0.35 (Eqs. 3.41/3.46), the
ANNULUS pair (Eq. 4.75, which corrected 0.30, 0.22 to 0.35, 0.10), the deviated factor
(Eq. 4.33), the churn C0 = 1.15 (sec. 3.3.3) and the annular no-slip pair (sec. 4.4.1,
"we recommend that liquid holdup in this regime be calculated on the basis of a no-slip
approach"). The sin theta on the bubbly v_d is the one item the book contradicts: it puts
inclination on the transition criterion instead (Eq. 4.7) and holds the small-bubble rise
velocity independent of angle. Left as-is pending the owner's decision; see
docs/TODO-M4.md and out/hk-locations.md.

Slug drift ramp (M3-1d, COMPUTED, our choice). Hasan-Kabir step the slug drift from the
Harmathy bubble velocity to the Taylor-bubble velocity at the bubbly-slug switch, which
steps alpha_g down (v_TB > v_H for any pipe wider than about 2 in). Here the slug and
churn drift ramps linearly in alpha_g from v_H at alpha_g = 0.25 (the bubbly-slug
switch) to v_TB at alpha_g = 0.52 (the dispersed-bubble cap), so slug and bubbly share
both C0 and v_d at the switch and the closure is continuous there by construction. The
ramp makes alpha_g implicit, alpha_g = v_sg / (C0 v_m + v_d(alpha_g)); it is solved by
fixed-point iteration from the bubbly value with the analytic-slope damping
omega = 1 / (1 + |dg/dalpha|), g(alpha) = v_sg / (C0 v_m + v_d(alpha)) (the undamped
map is not a contraction where v_TB / v_H is large: 17.5 in at v_sl = 0.2 ft/s has
|dg/dalpha| above 1), tolerance 1e-8 on the alpha_g step, capped at 50 iterations;
non-convergence raises HoldupConvergenceError. The iteration count is on the result.

Regime rule at a given (v_sg, v_sl), in the order tested:

    0. v_sl = 0 -> "gas" (alpha_g = 1, C0 = 1, v_d = 0); v_sg = 0 -> "liquid" (alpha_g = 0)
    1. v_sg > 3.1 [g sigma (rho_l - rho_g) / rho_g^2]^0.25   -> annular
       (Taitel, Barnea and Dukler 1980, adopted by Hasan-Kabir; MEMORY)
    2. v_m > v_ms = 4.0 D^0.429 (sigma/rho_l)^0.089 nu_l^-0.072 [g (rho_l - rho_g)/rho_l]^0.446
       (SI; Taitel, Barnea and Dukler 1980 dispersed-bubble mixture velocity, MEMORY,
       NEEDS HAND CHECK on the constants; D is the hydraulic diameter in annuli, COMPUTED)
       and alpha_bubbly < 0.52                      -> dispersed_bubbly (bubbly coefficients)
    3. v_m > v_ms and v_sg > 1.08 v_sl               -> churn
       (M3-1e: both conditions, the two-condition structure FETCHED by the owner from Hasan,
       Kabir and Sayarpour 2010, J. Pet. Sci. Eng., appendix A, eqs A-17 and A-18; 1.08 is
       the 0.52 dispersed ceiling in no-slip form, 0.52/0.48, `DISPERSED_CEILING_NO_SLIP`)
    4. alpha_bubbly <= 0.25                          -> bubbly
       (owner-specified rule; at vertical it is Hasan-Kabir's v_sg = 0.429 v_sl + 0.357 v_d;
       inclined, the sin theta sits on v_d only, where Hasan-Kabir's deviated-well form
       multiplies the whole right-hand side)
    5. otherwise                                     -> slug
       (so slug spans from the 0.25 switch up to the dispersed line: 0.72 to 17.7 ft/s at
       v_sl = 1 ft/s in a 6 in pipe)

Blended switches (M3-1c, COMPUTED, our choice). At fixed v_sl the rule above is
piecewise constant in v_sg with breakpoints v_tr,k, each solved from the same rule
(alpha_bubbly = 0.25 gives v_tr = (0.3 v_sl + 0.25 v_H)/0.7, i.e. 0.429 v_sl + 0.357 v_H;
alpha_bubbly = 0.52 the dispersed cap; v_m = v_ms the dispersed line; v_sg = 1.08 v_sl
the no-slip ceiling; v_sg = v_annular). The slug-churn logistic is centred on whichever
of the two churn conditions is binding at the current v_sl (the breakpoint where the
rule flips). Every switch except the bubbly-slug one (continuous
by the ramp; a hard step in the regime name) is blended with a logistic in v_sg,

    w_k = 1 / (1 + exp(-(v_sg - v_tr,k) / (0.05 v_tr,k)))
    C0  = C0_0 + sum_k w_k (C0_k - C0_(k-1)),   v_d likewise,

so a single switch is C0 = (1 - w) C0_lower + w C0_upper. The slug-churn and
churn-annular steps are small and upward in alpha_g. With the breakpoints sorted the
regime weights s_k = w_k - w_(k+1) (w_0 = 1, w_(K+1) = 0) are non-negative and sum to
1; the reported regime is the one with the largest weight and ``regime_weight`` is
that weight. Far from every switch the coefficients are the pure regime values to
better than 1e-6. The blend width is one constant, not diameter-aware.

The result carries the regime, the blended C0 and v_d, and a warning when the surface
tension was defaulted. No input is clamped; out-of-range inputs raise ValueError.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Holdup", "HoldupConvergenceError", "holdup", "NAME", "DEFAULT_SIGMA_DYNE_CM",
    "DISPERSED_CEILING_NO_SLIP", "REGIMES", "slug_drift",
]

NAME = "hasan_kabir_drift_flux"

# --- constants ---------------------------------------------------------------------
G_SI = 9.80665  # m/s2
_M_PER_FT = 0.3048
_KG_M3_PER_LBM_FT3 = 16.018463
_PA_S_PER_CP = 1.0e-3
_N_M_PER_DYNE_CM = 1.0e-3
_M_PER_IN = 0.0254

DEFAULT_SIGMA_DYNE_CM = 30.0  # used when sigma is None; always warned (owner rule, M3-1)

C0_BUBBLY = 1.2  # MEMORY: Hasan-Kabir 1988
C0_SLUG = 1.2  # MEMORY: Hasan-Kabir 1988 (FETCHED via arXiv physics/0504083 eq. 6)
C0_CHURN = 1.15  # MEMORY: Hasan-Kabir 1988
C0_ANNULAR = 1.0  # MEMORY: Hasan-Kabir 1988 (no slip in the core)
HARMATHY_COEF = 1.53  # MEMORY: Harmathy 1960 (FETCHED via arXiv physics/0504083 eq. 4)
TB_PIPE_COEF = 0.35  # MEMORY: Davies-Taylor / Nicklin (FETCHED via arXiv physics/0504083 eq. 6)
TB_ANNULUS_A, TB_ANNULUS_B = 0.35, 0.10  # FETCHED: Hasan & Kabir, Fluid Flow and Heat
# Transfer in Wellbores, 2nd ed., Eq. 4.75 (sec. 4.4.1). Was 0.30, 0.22 (MEMORY, via an
# arXiv secondary) until hk-1; see out/hk-locations.md. The book's form collapses to
# TB_PIPE_COEF at d_i = 0, which the old pair did not: 0.30 against 0.35.
ALPHA_BUBBLY_TO_SLUG = 0.25  # owner rule; Hasan et al. 1988 (FETCHED via arXiv)
ALPHA_DISPERSED_MAX = 0.52  # MEMORY: Taitel et al. 1980
DISPERSED_COEF = 4.0  # MEMORY: Taitel et al. 1980 eq. for the dispersed-bubble line
ANNULAR_COEF = 3.1  # MEMORY: Taitel et al. 1980 (Kutateladze-type criterion)
DISPERSED_CEILING_NO_SLIP = 1.08  # the 0.52 dispersed ceiling in homogeneous form, 0.52/0.48 = 1.083, as
# Hasan-Kabir print it: v_sg > 1.08 v_sl is the second of the two slug-to-churn conditions (FETCHED via the
# owner from Hasan, Kabir and Sayarpour 2010, J. Pet. Sci. Eng., appendix A, eqs A-17 and A-18)
BLEND_WIDTH = 0.05  # COMPUTED (owner's choice, M3-1c): logistic width as a fraction of v_tr
RAMP_ALPHA_LO, RAMP_ALPHA_HI = ALPHA_BUBBLY_TO_SLUG, ALPHA_DISPERSED_MAX  # M3-1d slug drift ramp
FIXED_POINT_TOL = 1.0e-8  # owner rule, M3-1d
FIXED_POINT_MAX_ITER = 50  # owner rule, M3-1d

REGIMES = ("liquid", "gas", "bubbly", "dispersed_bubbly", "slug", "churn", "annular")


class HoldupConvergenceError(RuntimeError):
    """The implicit alpha_g fixed-point iteration did not converge."""


@dataclass(frozen=True)
class Holdup:
    alpha_g: float  # in-situ gas volume fraction
    regime: str  # one of REGIMES: the regime with the largest blend weight
    c0: float  # distribution parameter used (blended)
    v_d: float  # drift velocity used (blended, ramped), ft/s
    v_m: float  # mixture superficial velocity, ft/s
    v_g: float  # in-situ gas velocity C0 v_m + v_d, ft/s
    sigma: float  # surface tension used, dyne/cm
    regime_weight: float = 1.0  # blend weight of the reported regime (1 = pure)
    iterations: int = 0  # fixed-point iterations used for the implicit alpha_g
    warnings: tuple[str, ...] = ()
    # perf pass step 2 (COMPUTED): analytic sensitivities at fixed v_sl
    dalpha_dv_sg: float = 0.0  # d alpha_g / d v_sg at fixed rho_g, s/ft
    dalpha_drho_g: float = 0.0  # d alpha_g / d rho_g at fixed mass rates (v_sg = w_g/(rho_g A)), ft3/lbm

    @property
    def h_l(self) -> float:
        return 1.0 - self.alpha_g


# --- pieces ------------------------------------------------------------------------

def harmathy_rise_velocity(rho_g: float, rho_l: float, sigma: float) -> float:
    """Small-bubble terminal rise velocity (vertical), ft/s:
    1.53 [g sigma (rho_l - rho_g)/rho_l^2]^0.25. rho in lbm/ft3, sigma in dyne/cm
    (MEMORY: Harmathy 1960)."""
    rl, rg = rho_l * _KG_M3_PER_LBM_FT3, rho_g * _KG_M3_PER_LBM_FT3
    s = sigma * _N_M_PER_DYNE_CM
    return HARMATHY_COEF * (G_SI * s * (rl - rg) / rl**2) ** 0.25 / _M_PER_FT


def taylor_bubble_velocity(rho_g: float, rho_l: float, d_o: float, d_i: float = 0.0) -> float:
    """Vertical Taylor-bubble rise velocity, ft/s.

        pipe (d_i = 0):  0.35                sqrt(g d_o (rho_l - rho_g)/rho_l)
        annulus:         0.35 + 0.10 d_i/d_o  x same

    ``d_o`` is the outer (casing ID) and ``d_i`` the inner (string OD) diameter, in inches.
    The length scale is d_o, not the hydraulic diameter, as the source writes it.

    FETCHED: Hasan & Kabir, *Fluid Flow and Heat Transfer in Wellbores*, 2nd ed., Eq. 4.75
    (sec. 4.4.1) for the annulus, Eq. 3.46 for the pipe. The annulus pair was 0.30, 0.22
    until hk-1, taken from a secondary source; the book's own pair is the one that makes
    the two branches agree in the limit they must, since 0.35 + 0.10 x 0 is exactly the
    pipe coefficient. The old pair gave 0.30 there, a 14% step at d_i -> 0."""
    rl, rg = rho_l * _KG_M3_PER_LBM_FT3, rho_g * _KG_M3_PER_LBM_FT3
    base = math.sqrt(G_SI * d_o * _M_PER_IN * (rl - rg) / rl)
    coef = TB_PIPE_COEF if d_i <= 0.0 else TB_ANNULUS_A + TB_ANNULUS_B * d_i / d_o
    return coef * base / _M_PER_FT


def inclination_factor(theta_deg: float) -> float:
    """Hasan-Kabir slug drift correction sqrt(sin theta) (1 + cos theta)^1.2, theta from
    horizontal in degrees (MEMORY; = 1 at vertical, above 1 at intermediate angles)."""
    t = math.radians(theta_deg)
    return math.sqrt(math.sin(t)) * (1.0 + math.cos(t)) ** 1.2


def dispersed_bubble_velocity(rho_g: float, rho_l: float, mu_l: float, sigma: float, d_hyd: float) -> float:
    """Mixture velocity above which bubbles stay dispersed (Taitel et al. 1980, MEMORY), ft/s:
    4.0 d^0.429 (sigma/rho_l)^0.089 nu_l^-0.072 [g (rho_l - rho_g)/rho_l]^0.446 in SI."""
    rl, rg = rho_l * _KG_M3_PER_LBM_FT3, rho_g * _KG_M3_PER_LBM_FT3
    s = sigma * _N_M_PER_DYNE_CM
    nu = mu_l * _PA_S_PER_CP / rl
    d = d_hyd * _M_PER_IN
    v = DISPERSED_COEF * d**0.429 * (s / rl) ** 0.089 * nu ** (-0.072) * (G_SI * (rl - rg) / rl) ** 0.446
    return v / _M_PER_FT


def annular_transition_velocity(rho_g: float, rho_l: float, sigma: float) -> float:
    """Superficial gas velocity above which the flow is annular (Taitel et al. 1980,
    MEMORY), ft/s: 3.1 [g sigma (rho_l - rho_g)/rho_g^2]^0.25 in SI."""
    rl, rg = rho_l * _KG_M3_PER_LBM_FT3, rho_g * _KG_M3_PER_LBM_FT3
    s = sigma * _N_M_PER_DYNE_CM
    return ANNULAR_COEF * (G_SI * s * (rl - rg) / rg**2) ** 0.25 / _M_PER_FT


def slug_drift(alpha_g: float, v_h: float, v_tb: float) -> float:
    """M3-1d slug/churn drift, ft/s: v_H + (v_TB - v_H) min(1, max(0, (alpha_g - 0.25)/0.27))
    (COMPUTED, our choice). v_h and v_tb already carry the inclination factors."""
    s = (alpha_g - RAMP_ALPHA_LO) / (RAMP_ALPHA_HI - RAMP_ALPHA_LO)
    return v_h + (v_tb - v_h) * min(1.0, max(0.0, s))


def _slug_drift_slope(alpha_g: float, v_h: float, v_tb: float) -> float:
    if RAMP_ALPHA_LO < alpha_g < RAMP_ALPHA_HI:
        return (v_tb - v_h) / (RAMP_ALPHA_HI - RAMP_ALPHA_LO)
    return 0.0


def logistic(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(x, 700.0)))
    e = math.exp(max(x, -700.0))
    return e / (1.0 + e)


@dataclass(frozen=True)
class _Ctx:
    """Everything the rule needs that does not depend on v_sg, with the rho_g derivatives
    of the velocity scales (perf step 2): v_H ~ (rho_l - rho_g)^0.25, v_TB ~ (rho_l - rho_g)^0.5,
    v_ms ~ (rho_l - rho_g)^0.446, v_annular ~ ((rho_l - rho_g)/rho_g^2)^0.25 (COMPUTED)."""
    v_sl: float
    v_d_bubbly: float  # Harmathy x sin theta
    v_tb: float  # Taylor bubble x inclination factor (top of the slug ramp)
    v_annular: float
    v_dispersed: float  # mixture velocity on the dispersed-bubble line
    d_v_d_bubbly: float = 0.0  # d/drho_g of the above, ft3/lbm x ft/s
    d_v_tb: float = 0.0
    d_v_annular: float = 0.0
    d_v_dispersed: float = 0.0

    def _ramp(self, alpha: float) -> tuple[float, float, float]:
        """(v_d, dv_d/dalpha, dv_d/drho_g) of the slug ramp at alpha."""
        s = (alpha - RAMP_ALPHA_LO) / (RAMP_ALPHA_HI - RAMP_ALPHA_LO)
        s = min(1.0, max(0.0, s))
        v_d = self.v_d_bubbly + (self.v_tb - self.v_d_bubbly) * s
        slope = _slug_drift_slope(alpha, self.v_d_bubbly, self.v_tb)
        d_rho = self.d_v_d_bubbly * (1.0 - s) + self.d_v_tb * s
        return v_d, slope, d_rho

    def coefficients(self, regime: str, alpha: float) -> tuple[float, float, float, float]:
        """(C0, v_d, dv_d/dalpha, dv_d/drho_g) of a pure regime at gas fraction alpha."""
        if regime in ("bubbly", "slug", "churn"):
            # The ramp is v_H wherever alpha <= 0.25, which the bubbly rule guarantees up to
            # the C0 blend leak; using it in bubbly keeps v_d continuous in alpha through
            # the bubbly-slug and bubbly-churn switches whatever C0 does there.
            c0 = C0_BUBBLY if regime == "bubbly" else C0_SLUG if regime == "slug" else C0_CHURN
            v_d, slope, d_rho = self._ramp(alpha)
            return c0, v_d, slope, d_rho
        if regime == "dispersed_bubbly":
            return C0_BUBBLY, self.v_d_bubbly, 0.0, self.d_v_d_bubbly
        if regime == "annular":
            return C0_ANNULAR, 0.0, 0.0, 0.0
        raise ValueError(regime)  # pragma: no cover

    def rule(self, v_sg: float) -> str:
        """The unblended regime at v_sg (v_sg > 0, v_sl > 0)."""
        if v_sg > self.v_annular:
            return "annular"
        v_m = v_sg + self.v_sl
        alpha_b = v_sg / (C0_BUBBLY * v_m + self.v_d_bubbly)
        if v_m > self.v_dispersed:
            if alpha_b < ALPHA_DISPERSED_MAX:
                return "dispersed_bubbly"
            if v_sg > DISPERSED_CEILING_NO_SLIP * self.v_sl:
                return "churn"
        if alpha_b <= ALPHA_BUBBLY_TO_SLUG:
            return "bubbly"
        return "slug"

    def v_sg_at_alpha_bubbly(self, alpha: float) -> float:
        # alpha = v_sg / (C0 (v_sg + v_sl) + v_d)  ->  v_sg = alpha (C0 v_sl + v_d) / (1 - alpha C0)
        return alpha * (C0_BUBBLY * self.v_sl + self.v_d_bubbly) / (1.0 - alpha * C0_BUBBLY)

    def d_v_sg_at_alpha_bubbly(self, alpha: float) -> float:
        return alpha * self.d_v_d_bubbly / (1.0 - alpha * C0_BUBBLY)

    def breakpoints(self) -> tuple[str, list[tuple[float, str, bool, float]]]:
        """(regime at v_sg -> 0, sorted [(v_tr, regime above v_tr, hard, dv_tr/drho_g), ...])
        from the rule. ``hard`` marks the bubbly-slug switch (alpha_bubbly = 0.25): no
        logistic there, the ramp makes it continuous."""
        v_bs = self.v_sg_at_alpha_bubbly(ALPHA_BUBBLY_TO_SLUG)
        cands = {
            v_bs: self.d_v_sg_at_alpha_bubbly(ALPHA_BUBBLY_TO_SLUG),
            self.v_sg_at_alpha_bubbly(ALPHA_DISPERSED_MAX): self.d_v_sg_at_alpha_bubbly(ALPHA_DISPERSED_MAX),
            self.v_dispersed - self.v_sl: self.d_v_dispersed,
            DISPERSED_CEILING_NO_SLIP * self.v_sl: 0.0,
            self.v_annular: self.d_v_annular,
        }
        keys = sorted(c for c in cands if c > 0.0)
        first = self.rule(0.5 * keys[0]) if keys else self.rule(1.0)
        out: list[tuple[float, str, bool, float]] = []
        prev = first
        for i, b in enumerate(keys):
            probe = 0.5 * (b + keys[i + 1]) if i + 1 < len(keys) else 2.0 * b
            reg = self.rule(probe)
            if reg != prev:
                out.append((b, reg, b == v_bs and prev == "bubbly", cands[b]))
                prev = reg
        return first, out


# --- the closure ---------------------------------------------------------------------

FALLBACKS = {"count": 0}  # perf step 3: how often the closed form failed and the fixed point ran
_RAMP_REGIMES = ("bubbly", "slug", "churn")


def _closed_form_alpha(ctx: "_Ctx", v_sg: float, v_m: float, w_logistic: list, regimes: list) -> float | None:
    """alpha_g on the ramp piece that holds it, or None (perf step 3, COMPUTED)."""
    bounds = [1.0] + list(w_logistic) + [0.0]
    w_ramp = w_disp = w_ann = c0 = 0.0
    for j, reg in enumerate(regimes):
        s = bounds[j] - bounds[j + 1]
        if reg in _RAMP_REGIMES:
            w_ramp += s
            c0 += s * (C0_BUBBLY if reg == "bubbly" else C0_SLUG if reg == "slug" else C0_CHURN)
        elif reg == "dispersed_bubbly":
            w_disp += s
            c0 += s * C0_BUBBLY
        else:
            w_ann += s
            c0 += s * C0_ANNULAR
    v_h, v_tb = ctx.v_d_bubbly, ctx.v_tb
    cv = c0 * v_m
    # piece 1: alpha <= 0.25, v_d = (w_ramp + w_disp) v_H
    a1 = v_sg / (cv + (w_ramp + w_disp) * v_h)
    if a1 <= RAMP_ALPHA_LO:
        return a1
    # piece 2: 0.25 < alpha < 0.52, v_d = A + B alpha
    k = (v_tb - v_h) / (RAMP_ALPHA_HI - RAMP_ALPHA_LO)
    b_lin = w_ramp * k
    a_lin = (w_ramp + w_disp) * v_h - b_lin * RAMP_ALPHA_LO
    b = cv + a_lin
    if abs(b_lin) < 1e-14 * max(1.0, abs(b)):
        a2 = v_sg / b
    else:
        disc = b * b + 4.0 * b_lin * v_sg
        if disc < 0.0:
            a2 = None
        else:
            a2 = 2.0 * v_sg / (b + math.sqrt(disc))  # the root continuous with piece 1, either sign of B
    if a2 is not None and RAMP_ALPHA_LO < a2 < RAMP_ALPHA_HI:
        return a2
    # piece 3: alpha >= 0.52, v_d = w_ramp v_TB + w_disp v_H
    a3 = v_sg / (cv + w_ramp * v_tb + w_disp * v_h)
    if a3 >= RAMP_ALPHA_HI:
        return a3
    return None


def _fixed_point_alpha(ctx: "_Ctx", v_sg: float, v_sl: float, v_m: float, blended) -> tuple[float, int]:
    """The M3-1d damped, bracketed fixed point with the Newton polish (now the fallback).
    The residual alpha - g(alpha) is increasing in alpha (v_d is non-decreasing in alpha),
    so its sign brackets the root; a damped step that leaves the bracket is replaced by
    the bracket midpoint (the ramp kinks at 0.25 and 0.52 can make the plain damped map
    cycle). One polish step after the stopping rule takes the error to round-off."""
    alpha = v_sg / (C0_BUBBLY * v_m + ctx.v_d_bubbly)
    lo, hi = 0.0, 1.0
    iterations = 0
    converged = False
    step = 0.0
    for iterations in range(1, FIXED_POINT_MAX_ITER + 1):
        c0, v_d, dvd = blended(alpha)
        den = c0 * v_m + v_d
        g = v_sg / den
        if abs(g - alpha) < FIXED_POINT_TOL:
            converged = True
            break
        if g < alpha:
            hi = alpha
        else:
            lo = alpha
        g_slope = -v_sg * dvd / (den * den)
        omega = 1.0 / (1.0 + abs(g_slope))
        new = alpha + omega * (g - alpha)
        if new < lo or new > hi:
            new = 0.5 * (lo + hi)
        step = abs(new - alpha)
        alpha = new
        if step < FIXED_POINT_TOL:
            converged = True
            break
    if not converged:
        raise HoldupConvergenceError(
            f"alpha_g fixed point did not converge in {FIXED_POINT_MAX_ITER} iterations at "
            f"v_sg {v_sg}, v_sl {v_sl} ft/s (last step {step:.2e})"
        )
    c0, v_d, dvd = blended(alpha)
    den = c0 * v_m + v_d
    alpha += (v_sg / den - alpha) / (1.0 + v_sg * dvd / (den * den))
    return alpha, iterations + 1


def holdup(
    v_sg: float,
    v_sl: float,
    rho_g: float,
    rho_l: float,
    mu_l: float,
    sigma: float | None,
    d_o: float,
    d_i: float = 0.0,
    theta: float = 90.0,
    solver: str = "closed_form",
) -> Holdup:
    """Gas fraction and regime from the continuous drift-flux closure.

    v_sg, v_sl  superficial gas and liquid velocities, ft/s (>= 0, not both 0)
    rho_g, rho_l  phase densities, lbm/ft3 (rho_l > rho_g > 0)
    mu_l  liquid viscosity, cp (> 0; enters only the dispersed-bubble line)
    sigma  gas-liquid surface tension, dyne/cm; None -> 30 with a warning
    d_o, d_i  outer and inner conduit diameters, in (d_i = 0 for a pipe; d_i < d_o)
    theta  inclination from horizontal, degrees, 0 <= theta <= 90
    solver  "closed_form" (default, perf step 3) or "fixed_point" (the M3-1d iteration, kept
            as the fallback and for the tests)
    """
    if solver not in ("closed_form", "fixed_point"):
        raise ValueError(f"solver must be 'closed_form' or 'fixed_point', got {solver!r}")
    if v_sg < 0.0 or v_sl < 0.0:
        raise ValueError(f"superficial velocities must be >= 0, got v_sg {v_sg}, v_sl {v_sl}")
    if v_sg == 0.0 and v_sl == 0.0:
        raise ValueError("v_sg and v_sl cannot both be zero")
    if rho_g <= 0.0 or rho_l <= rho_g:
        raise ValueError(f"need rho_l > rho_g > 0, got rho_l {rho_l}, rho_g {rho_g}")
    if mu_l <= 0.0:
        raise ValueError(f"mu_l must be > 0, got {mu_l}")
    if d_o <= 0.0 or d_i < 0.0 or d_i >= d_o:
        raise ValueError(f"need 0 <= d_i < d_o, got d_o {d_o}, d_i {d_i}")
    if not 0.0 <= theta <= 90.0:
        raise ValueError(f"theta must be between 0 and 90 degrees from horizontal, got {theta}")

    warnings: list[str] = []
    if sigma is None:
        sigma = DEFAULT_SIGMA_DYNE_CM
        warnings.append(
            f"gas-liquid surface tension not supplied; defaulted to {DEFAULT_SIGMA_DYNE_CM:g} dyne/cm"
        )
    elif sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    v_m = v_sg + v_sl
    if v_sg == 0.0:
        return Holdup(0.0, "liquid", 1.0, 0.0, v_m, v_m, sigma, 1.0, 0, tuple(warnings))
    if v_sl == 0.0:
        # Single-phase gas guard (M3-1c): the march hands off to the gas solver here.
        return Holdup(1.0, "gas", 1.0, 0.0, v_m, v_m, sigma, 1.0, 0, tuple(warnings))

    v_h = harmathy_rise_velocity(rho_g, rho_l, sigma) * math.sin(math.radians(theta))
    v_tb = taylor_bubble_velocity(rho_g, rho_l, d_o, d_i) * inclination_factor(theta)
    v_ann = annular_transition_velocity(rho_g, rho_l, sigma)
    v_ms = dispersed_bubble_velocity(rho_g, rho_l, mu_l, sigma, d_o - d_i)
    dr = rho_l - rho_g
    ctx = _Ctx(
        v_sl=v_sl, v_d_bubbly=v_h, v_tb=v_tb, v_annular=v_ann, v_dispersed=v_ms,
        d_v_d_bubbly=-0.25 * v_h / dr, d_v_tb=-0.5 * v_tb / dr,
        d_v_annular=v_ann * (-0.25 / dr - 0.5 / rho_g), d_v_dispersed=-0.446 * v_ms / dr,
    )
    first, switches = ctx.breakpoints()
    regimes = [first] + [r for _, r, _, _ in switches]
    # Logistic weights for the coefficient blend at every switch. At the bubbly-slug switch
    # bubbly and slug share C0 and the ramped v_d, so the logistic there is the identity;
    # the regime name takes a hard step at that switch instead.
    w_logistic = [logistic((v_sg - v_tr) / (BLEND_WIDTH * v_tr)) for v_tr, _, _, _ in switches]
    weights = [(1.0 if v_sg > v_tr else 0.0) if hard else w for w, (v_tr, _, hard, _) in zip(w_logistic, switches)]
    # weight derivatives: x = (v_sg - v_tr)/(BLEND_WIDTH v_tr); dw/dv_sg and dw/drho_g (through v_tr)
    dw_dv = [w * (1.0 - w) / (BLEND_WIDTH * v_tr) for w, (v_tr, _, _, _) in zip(w_logistic, switches)]
    dw_drho = [-w * (1.0 - w) * v_sg / (BLEND_WIDTH * v_tr * v_tr) * d_tr
               for w, (v_tr, _, _, d_tr) in zip(w_logistic, switches)]

    def blended(alpha: float) -> tuple[float, float, float]:
        c0, v_d, dvd, _ = ctx.coefficients(first, alpha)
        prev = (c0, v_d, dvd)
        for w, reg in zip(w_logistic, regimes[1:]):
            nxt = ctx.coefficients(reg, alpha)
            c0 += w * (nxt[0] - prev[0])
            v_d += w * (nxt[1] - prev[1])
            dvd += w * (nxt[2] - prev[2])
            prev = nxt[:3]
        return c0, v_d, dvd

    def sensitivities(alpha: float) -> tuple[float, float]:
        """(dalpha/dv_sg at fixed rho_g, dalpha/drho_g at fixed mass rates) by implicit
        differentiation of alpha (C0 v_m + v_d) = v_sg (COMPUTED, perf step 2)."""
        c0, v_d, dvd_a, dvd_r = ctx.coefficients(first, alpha)
        dc0_dv = dc0_dr = dvd_dv = 0.0
        prev = (c0, v_d, dvd_a, dvd_r)
        for w, wv, wr, reg in zip(w_logistic, dw_dv, dw_drho, regimes[1:]):
            nxt = ctx.coefficients(reg, alpha)
            dc, dv = nxt[0] - prev[0], nxt[1] - prev[1]
            c0 += w * dc
            v_d += w * dv
            dvd_a += w * (nxt[2] - prev[2])
            dvd_r += w * (nxt[3] - prev[3]) + dv * wr
            dc0_dv += dc * wv
            dc0_dr += dc * wr
            dvd_dv += dv * wv
            prev = nxt
        d = c0 * v_m + v_d + alpha * dvd_a
        da_dv = (1.0 - alpha * c0 - alpha * (dc0_dv * v_m + dvd_dv)) / d
        da_dr_explicit = -alpha * (dc0_dr * v_m + dvd_r) / d
        return da_dv, da_dv * (-v_sg / rho_g) + da_dr_explicit

    # Closed-form solve (perf step 3, COMPUTED). The blend is a convex combination of the
    # pure regimes with segment weights s_j = w_j - w_(j+1); the ramp regimes (bubbly,
    # slug, churn) share v_d = ramp(alpha), dispersed has v_H, annular 0, so on each ramp
    # piece the blended v_d is constant or linear in alpha and alpha (C0 v_m + v_d) = v_sg
    # is explicit or a quadratic. Exactly one piece holds the root (the residual is
    # increasing in alpha); if none does (round-off at a piece boundary) the bracketed
    # fixed point below runs instead and FALLBACKS is incremented.
    alpha = None
    iterations = 0
    if solver == "closed_form":
        alpha = _closed_form_alpha(ctx, v_sg, v_m, w_logistic, regimes)
        if alpha is None:
            FALLBACKS["count"] += 1
    if alpha is None:
        alpha, iterations = _fixed_point_alpha(ctx, v_sg, v_sl, v_m, blended)
    c0, v_d, _ = blended(alpha)
    v_g = c0 * v_m + v_d

    bounds = [1.0] + weights + [0.0]
    seg_weights = [bounds[k] - bounds[k + 1] for k in range(len(regimes))]
    k_best = max(range(len(regimes)), key=lambda k: seg_weights[k])
    da_dv, da_dr = sensitivities(alpha)
    return Holdup(
        v_sg / v_g, regimes[k_best], c0, v_d, v_m, v_g, sigma, seg_weights[k_best], iterations, tuple(warnings),
        dalpha_dv_sg=da_dv, dalpha_drho_g=da_dr,
    )
