"""Oil inflow: Darcy pseudo-steady-state above bubble point, Vogel composite below.

Darcy PSS above Pb:
    q_o = J * (p_r - p_wf)
    J = k * h / (141.2 * Bo * mu_o * (ln(re/rw) - 0.75 + s_total))

Vogel composite below Pb (Vogel 1968, SPE 1476):
    If p_r > p_b:
        q_b = J * (p_r - p_b)
        q_v = J * p_b / 1.8
        q_o = q_b + q_v * (1 - 0.2 * (p_wf / p_b) - 0.8 * (p_wf / p_b)^2)
    If p_r <= p_b:
        q_max = J * p_r / 1.8
        q_o = q_max * (1 - 0.2 * (p_wf / p_r) - 0.8 * (p_wf / p_r)^2)

Associated fluids:
    q_w = q_o * (water_cut / (1 - water_cut))
    q_g = q_o * max(0, GOR - Rs(p_wf))   [scf/d, converted to Mscf/d]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wellsim.core.ipr.partial_penetration import papatzacos_skin

if TYPE_CHECKING:
    from wellsim.core.pvt.black_oil import BlackOilPVT

__all__ = ["DARCY_OIL_CONSTANT", "OilIPR"]

# Standard field constant: 141.2 from 2 * pi * 0.001127 in field units
DARCY_OIL_CONSTANT = 141.2


@dataclass
class OilIPR:
    pvt: "BlackOilPVT"
    p_res: float  # psia
    t_res: float  # degF
    k: float  # md
    h: float  # ft (net pay)
    r_w: float  # ft
    r_e: float  # ft
    skin: float = 0.0
    h_p: float | None = None  # open interval
    h_1: float = 0.0  # top offset
    kv_kh: float | None = None

    def __post_init__(self) -> None:
        self.h_p = self.h if self.h_p is None else float(self.h_p)
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
        if self.geom <= 0.1:
            self.geom = 0.1  # prevent singularity on high negative skin

        # Fluid properties at reservoir conditions
        self.pb = self.pvt.bubble_point(self.t_res)
        bo_res = self.pvt.bo(self.p_res, self.t_res)
        mu_res = self.pvt.oil_viscosity(self.p_res, self.t_res)

        # Darcy productivity index J (stb/d/psi)
        self.j = (self.k * self.h) / (DARCY_OIL_CONSTANT * bo_res * mu_res * self.geom)

        # Precompute Vogel components
        if self.p_res > self.pb:
            self.q_b = self.j * (self.p_res - self.pb)
            self.q_vogel_max = self.j * self.pb / 1.8
            self.q_aof = self.q_b + self.q_vogel_max
        else:
            self.q_b = 0.0
            self.q_vogel_max = self.j * self.p_res / 1.8
            self.q_aof = self.q_vogel_max

    def oil_rate(self, p_wf: float) -> float:
        """Oil rate in stb/d for a flowing bottomhole pressure."""
        if p_wf >= self.p_res:
            return 0.0
        p_wf = max(0.0, p_wf)

        if self.p_res > self.pb:
            if p_wf >= self.pb:
                return self.j * (self.p_res - p_wf)
            # Below Pb: Darcy down to Pb + Vogel below Pb
            ratio = p_wf / self.pb
            vogel_term = 1.0 - 0.2 * ratio - 0.8 * (ratio**2)
            return self.q_b + self.q_vogel_max * vogel_term
        else:
            # Saturated reservoir (pure Vogel)
            ratio = p_wf / self.p_res
            vogel_term = 1.0 - 0.2 * ratio - 0.8 * (ratio**2)
            return self.q_vogel_max * vogel_term

    def rates(self, p_wf: float) -> tuple[float, float, float]:
        """Return (q_oil_stbd, q_water_stbd, q_gas_mscfd) at flowing bottomhole pressure."""
        q_o = self.oil_rate(p_wf)
        if q_o <= 0.0:
            return 0.0, 0.0, 0.0

        # Water rate from water cut
        wc = self.pvt.water_cut
        if wc >= 1.0:
            q_w = 0.0
        else:
            q_w = q_o * (wc / (1.0 - wc))

        # Gas rate from GOR and solution gas Rs(p_wf)
        p_eval = max(14.7, min(p_wf, self.p_res))
        rs_wf = self.pvt.rs(p_eval, self.t_res)
        # Total produced gas is q_o * GOR (scf/d)
        q_g_total_scfd = q_o * self.pvt.gor
        q_g_mscfd = q_g_total_scfd / 1000.0

        return q_o, q_w, q_g_mscfd

    def rate(self, p_wf: float) -> float:
        """Default rate is oil rate in stb/d (for compatibility with nodal solvers)."""
        return self.oil_rate(p_wf)

    def aof(self) -> float:
        """Absolute Open Flow (stb/d) at pwf = 0."""
        return self.q_aof

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "oil_darcy_vogel",
            "j_stbd_psi": self.j,
            "bubble_point_psia": self.pb,
            "aof_stbd": self.q_aof,
            "s_pp": self.s_pp,
            "s_total": self.s_total,
        }
