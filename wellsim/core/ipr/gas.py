"""Gas inflow: pseudo-steady-state pseudo-pressure IPR with non-Darcy skin (spec 4.2).

    m(p_r) - m(p_wf) = 1422 T q / (k h) * [ln(r_e/r_w) - 0.75 + s + D q]

with m(p) = 2 * integral_0^p p'/(mu Z) dp' in psia^2/cp, q in Mscf/d, T in degR, k in md,
h in ft. The constant 1422 is VERIFIED by independent hand derivation from Darcy radial
flow in field units at standard conditions of 14.7 psia and 60 degF (see
docs/PROVENANCE.md); the SI derivation in tests/test_ipr_gas.py gives 1422.4. Usual
citations: Lee, "Well Testing" (1982); Ahmed, "Reservoir Engineering Handbook" ch. 8.

Non-Darcy coefficient, from the Forchheimer equation integrated radially with the
mass flux evaluated over the perforated interval h_p (derived in the tests):

    D = 2.222e-15 * gamma_g * k * h * beta / (mu_wf * r_w * h_p^2)     [d/Mscf]

with beta in 1/ft and mu_wf the gas viscosity at the flowing bottomhole pressure.
VERIFIED by hand from the general form 2.715e-15 beta k M p_sc / (h mu r_w T_sc) with
M = 28.97 gamma at 14.7 psia and 60 degF. Form as in Ahmed ch. 8.

Beta correlations (spec open question 2 on the default; `Reservoir.non_darcy_beta`
overrides them):
    jones (1987, SPE 16949):        beta = 6.15e10 / k^1.55        k md, beta 1/ft
        VERIFIED: Elsanoose et al., Energies 2022, 15, 7616, eq. 7.
    geertsma (1974, SPEJ, SPE-4706-PA): beta = 0.005 / (phi^5.5 k^0.5)   beta 1/cm, k cm^2
                                       = 48511 / (phi^5.5 k^0.5)       beta 1/ft, k md
        VERIFIED: same paper, eq. 5; field constant by hand conversion.
    firoozabadi_katz:               beta = 2.6e10 / k^1.2          k md, beta 1/ft
        MEMORY, NEEDS HAND CHECK: the 1979 JPT paper gives a chart; this is a later
        curve fit whose author has not been identified. Placeholder only.
Default (M2.5-5, `correlations.non_darcy_beta = "auto"`): Jones when only permeability
is given, Geertsma when porosity is also supplied; the solver resolves "auto" and notes
which one was used.

Partial penetration (M2.5-1). The open interval h_p enters in three places:
    1. D above carries h_p^2 (mass flux over the open interval, not the net pay). The
       h_p^2 form is MEMORY (Ahmed ch. 8); the constant is VERIFIED. Reduces to the
       full-pay form exactly when h_p = h.
    2. The Papatzacos (1987, SPE-13956-PA) pseudo-skin s_pp, computed in
       partial_penetration.py from h_p, the top offset h_1 and kv/kh (formula VERIFIED
       against Lu et al. 2010, Article 907206, eqs. 2.4 to 2.8).
    3. The mechanical skin is scaled by h/h_p (MEMORY: damage acts over the open
       interval only):  s_total = s_pp + (h / h_p) s_d.
When h_p < h and kv/kh is not given, 0.1 is used and ``kv_kh_defaulted`` is set so the
solver can warn; the physics layer never defaults silently on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from wellsim.core.ipr.partial_penetration import papatzacos_skin
from wellsim.core.numerics import brent

if TYPE_CHECKING:
    from wellsim.core.pvt.base import PVTModel

__all__ = [
    "IPR_CONSTANT",
    "D_CONSTANT",
    "beta_jones",
    "beta_geertsma",
    "beta_firoozabadi_katz",
    "BETA_METHODS",
    "PseudoPressure",
    "GasIPR",
]

IPR_CONSTANT = 1422.0  # psia^2/cp per (Mscf/d degR / (md ft))
D_CONSTANT = 2.222e-15
GEERTSMA_FIELD = 0.005 / math.sqrt(9.869233e-12) * 30.48  # = 48511: beta 1/ft, k md (1 md = 9.869e-12 cm^2)


def beta_jones(k_md: float, porosity: float | None = None) -> float:
    return 6.15e10 / k_md**1.55


def beta_geertsma(k_md: float, porosity: float | None = None) -> float:
    if porosity is None or porosity <= 0.0:
        raise ValueError("Geertsma beta correlation needs reservoir porosity")
    return GEERTSMA_FIELD / (porosity**5.5 * math.sqrt(k_md))


def beta_firoozabadi_katz(k_md: float, porosity: float | None = None) -> float:
    return 2.6e10 / k_md**1.2


BETA_METHODS = {
    "jones": beta_jones,
    "geertsma": beta_geertsma,
    "firoozabadi_katz": beta_firoozabadi_katz,
}


class PseudoPressure:
    """Tabulated m(p) = 2 * int_0^p p/(mu Z) dp for one PVT model at one temperature.

    Composite Simpson on a uniform grid to ``p_max``; linear interpolation between nodes.
    The integrand is 0 at p = 0 and smooth, so 400 intervals give better than 1e-5
    relative accuracy for the differences the IPR uses.
    """

    def __init__(self, pvt: "PVTModel", t: float, p_max: float, n: int = 400):
        if n % 2:
            n += 1
        self.t = t
        self.p_max = p_max
        self.p = np.linspace(0.0, p_max, n + 1)
        f = np.empty_like(self.p)
        f[0] = 0.0
        for i in range(1, n + 1):
            pi = self.p[i]
            f[i] = pi / (pvt.gas_viscosity(pi, t) * pvt.z(pi, t))
        h = p_max / n
        # cumulative Simpson at even nodes, trapezoid-corrected at odd nodes
        m = np.empty_like(self.p)
        m[0] = 0.0
        for i in range(2, n + 1, 2):
            m[i] = m[i - 2] + h / 3.0 * (f[i - 2] + 4.0 * f[i - 1] + f[i])
        for i in range(1, n, 2):
            # quadratic through i-1, i, i+1 integrated over [i-1, i]
            m[i] = m[i - 1] + h / 12.0 * (5.0 * f[i - 1] + 8.0 * f[i] - f[i + 1])
        self.m = 2.0 * m

    def __call__(self, p: float) -> float:
        if p <= 0.0:
            return 0.0
        if p > self.p_max:
            raise ValueError(f"pseudo-pressure table only covers 0 to {self.p_max:.0f} psia (asked {p:.0f})")
        return float(np.interp(p, self.p, self.m))

    def inverse(self, m: float) -> float:
        """p such that m(p) = m. m is monotonic increasing so interp on the swapped table is exact."""
        if m <= 0.0:
            return 0.0
        if m > self.m[-1]:
            raise ValueError("pseudo-pressure outside table")
        return float(np.interp(m, self.m, self.p))


@dataclass
class GasIPR:
    """Pseudo-steady-state gas IPR. Immutable after construction; pure functions of rate."""

    pvt: "PVTModel"
    p_res: float  # psia
    t_res: float  # degF
    k: float  # md
    h: float  # ft, net pay
    r_e: float  # ft
    r_w: float  # ft
    skin: float = 0.0
    h_p: float | None = None  # perforated / exposed interval, ft; None = h
    h_1: float = 0.0  # top of the open interval below the top of the zone, ft
    kv_kh: float | None = None  # vertical/horizontal permeability ratio; None with h_p < h -> 0.1, flagged
    beta: float | None = None  # 1/ft; None -> correlation
    beta_method: str = "jones"
    porosity: float | None = None
    include_non_darcy: bool = True

    def __post_init__(self) -> None:
        if self.h_p is None:
            self.h_p = self.h
        if self.beta is None:
            if self.beta_method not in BETA_METHODS:
                raise ValueError(f"unknown beta correlation {self.beta_method!r}; choose from {list(BETA_METHODS)}")
            self.beta = BETA_METHODS[self.beta_method](self.k, self.porosity)
            self.beta_source = self.beta_method
        else:
            self.beta_source = "user"
        self.t_r = self.t_res + 459.67
        # partial penetration: Papatzacos pseudo-skin plus mechanical skin scaled by h/h_p
        self.h_pd = self.h_p / self.h
        self.kv_kh_defaulted = False
        if self.h_p < self.h:
            if self.kv_kh is None:
                self.kv_kh = 0.1
                self.kv_kh_defaulted = True
            self.s_pp = papatzacos_skin(self.h, self.h_p, self.h_1, self.r_w, self.kv_kh)
        else:
            self.s_pp = 0.0
        self.s_mech = (self.h / self.h_p) * self.skin
        self.s_total = self.s_pp + self.s_mech
        self.geom = math.log(self.r_e / self.r_w) - 0.75 + self.s_total
        self.c = IPR_CONSTANT * self.t_r / (self.k * self.h)
        self.m = PseudoPressure(self.pvt, self.t_res, 1.05 * self.p_res)
        self.m_res = self.m(self.p_res)
        # Absolute open flow: the maximum of the raw rate(p). With D carrying mu at p_wf the
        # curve turns over above atmospheric (mu falls, D rises, while m(p) has nothing left
        # to give); for a thin interval the turnover can sit at hundreds of psi. The maximum
        # is located by a scan over the whole [14.7, p_res] range refined by golden section,
        # and rate(p) is capped at it below p_aof (flat cap; see rate()).
        self.p_aof, self.q_aof = self._locate_maximum()

    def _locate_maximum(self) -> tuple[float, float]:
        p_lo, p_hi = 14.7, self.p_res
        n = 60
        ps = [p_lo + (p_hi - p_lo) * i / n for i in range(n + 1)]
        qs = [self._rate_raw(p) for p in ps]
        i = max(range(n + 1), key=lambda j: qs[j])
        a = ps[max(i - 1, 0)]
        b = ps[min(i + 1, n)]
        if a == b:
            return ps[i], qs[i]
        # golden-section maximisation on [a, b]
        g = (math.sqrt(5.0) - 1.0) / 2.0
        x1 = b - g * (b - a)
        x2 = a + g * (b - a)
        f1, f2 = self._rate_raw(x1), self._rate_raw(x2)
        for _ in range(60):
            if f1 < f2:
                a, x1, f1 = x1, x2, f2
                x2 = a + g * (b - a)
                f2 = self._rate_raw(x2)
            else:
                b, x2, f2 = x2, x1, f1
                x1 = b - g * (b - a)
                f1 = self._rate_raw(x1)
            if b - a < 1e-6:
                break
        p_best = x1 if f1 >= f2 else x2
        q_best = max(f1, f2)
        if qs[i] > q_best:
            p_best, q_best = ps[i], qs[i]
        return p_best, q_best

    # --- non-Darcy ---------------------------------------------------------------

    def d_coefficient(self, p_wf: float) -> float:
        """D in d/Mscf with gas viscosity at p_wf."""
        if not self.include_non_darcy:
            return 0.0
        mu = self.pvt.gas_viscosity(max(p_wf, 14.7), self.t_res)
        return D_CONSTANT * self.pvt.gas_sg * self.k * self.h * self.beta / (mu * self.r_w * self.h_p**2)

    # --- the two directions ------------------------------------------------------

    def rate(self, p_wf: float) -> float:
        """Gas rate, Mscf/d, for a flowing bottomhole pressure. 0 if p_wf >= p_res. Below
        the pressure of maximum rate (``p_aof``) the rate is capped at that maximum
        (``q_aof``): the raw curve's turnover there is an artefact of D carrying mu(p_wf),
        not a physical decline (owner decision, nodal solve in pwf space)."""
        if p_wf < self.p_aof:
            return self.q_aof
        return self._rate_raw(p_wf)

    def cap_active(self, p_wf: float) -> bool:
        return p_wf < self.p_aof

    def _rate_raw(self, p_wf: float) -> float:
        """Uncapped rate(p_wf) from the deliverability equation."""
        if p_wf >= self.p_res:
            return 0.0
        dm = self.m_res - self.m(max(p_wf, 0.0))
        d = self.d_coefficient(p_wf)
        a = self.c * d
        b = self.c * self.geom
        if a == 0.0:
            return dm / b
        return (-b + math.sqrt(b * b + 4.0 * a * dm)) / (2.0 * a)

    def pwf(self, q: float) -> float | None:
        """Flowing bottomhole pressure for a gas rate, by Brent on rate(p) = q over
        [p_aof, p_res] where rate is monotonic. None if the rate exceeds the AOF."""
        if q <= 0.0:
            return self.p_res
        f_lo = self.q_aof - q
        if f_lo < 0.0:
            return None
        if f_lo == 0.0:
            return self.p_aof
        return brent(lambda p: self.rate(p) - q, self.p_aof, self.p_res, fa=f_lo, fb=-q, xtol=1e-6, rtol=1e-10)

    def aof(self) -> float:
        """Absolute open flow: the maximum rate the IPR gives (see __post_init__)."""
        return self.q_aof

    def describe(self) -> dict:
        return {
            "model": "gas_pseudo_pressure_pss",
            "beta_1_per_ft": self.beta,
            "beta_source": self.beta_source,
            "skin": self.skin,
            "skin_mechanical_scaled": self.s_mech,
            "skin_partial_penetration": self.s_pp,
            "skin_total": self.s_total,
            "h_pd": self.h_pd,
            "h_1_ft": self.h_1,
            "kv_kh": self.kv_kh,
            "kv_kh_defaulted": self.kv_kh_defaulted,
            "geometry_term": self.geom,
            "non_darcy": self.include_non_darcy,
            "q_aof": self.q_aof,
            "p_aof": self.p_aof,
        }
