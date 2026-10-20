"""Partial-penetration pseudo-skin (spec 4.2, TODO-M3 item 1).

Papatzacos (1987), "Approximate Partial-Penetration Pseudoskin for Infinite-
Conductivity Wells", SPE Reservoir Engineering 2(2), SPE-13956-PA. Formula as coded
(VERIFIED by the owner against Lu et al., Mathematical Problems in Engineering, vol. 2010,
Article 907206, eqs. 2.4 to 2.8, which write ln(pi h_D / 2) with h_D = (h/r_w) sqrt(kh/kv),
identical to ln(pi/(2 r_D)) below; see docs/PROVENANCE.md):

    h_pD = h_p / h                  penetration ratio
    h_1D = h_1 / h                  top of the open interval below the top of the zone
    r_D  = (r_w / h) sqrt(kv/kh)    dimensionless wellbore radius
    A    = 1 / (h_1D + h_pD / 4)
    B    = 1 / (h_1D + 3 h_pD / 4)
    s_pp = (1/h_pD - 1) ln(pi / (2 r_D))
           + (1/h_pD) ln[ (h_pD / (2 + h_pD)) sqrt((A - 1) / (B - 1)) ]

Properties relied on (all tested): s_pp = 0 at full penetration; the same interval at
the top and at the bottom of the zone gives the same skin; a centred interval gives
less skin than one against a boundary; s_pp rises as h_pD or kv/kh falls.

Deviated (slanted) penetration is out of scope (docs/TODO-M3.md).
"""

from __future__ import annotations

import math

__all__ = ["papatzacos_skin"]


def papatzacos_skin(h: float, h_p: float, h_1: float, r_w: float, kv_kh: float) -> float:
    """Papatzacos pseudo-skin, dimensionless. Lengths in any consistent unit.

    h     net pay
    h_p   open (perforated) interval length, 0 < h_p <= h
    h_1   distance from the top of the zone to the top of the open interval, >= 0
    r_w   wellbore radius
    kv_kh vertical-to-horizontal permeability ratio, > 0
    """
    if h <= 0.0 or h_p <= 0.0 or r_w <= 0.0 or kv_kh <= 0.0:
        raise ValueError("papatzacos_skin needs h, h_p, r_w and kv_kh > 0")
    if h_1 < 0.0:
        raise ValueError("h_1 (top offset of the open interval) cannot be negative")
    if h_1 + h_p > h * (1.0 + 1e-9):
        raise ValueError("open interval extends below the zone: h_1 + h_p > h")
    h_pd = min(h_p / h, 1.0)
    if h_pd >= 1.0:
        return 0.0
    h_1d = h_1 / h
    r_d = (r_w / h) * math.sqrt(kv_kh)
    a = 1.0 / (h_1d + h_pd / 4.0)
    b = 1.0 / (h_1d + 3.0 * h_pd / 4.0)
    term1 = (1.0 / h_pd - 1.0) * math.log(math.pi / (2.0 * r_d))
    term2 = (1.0 / h_pd) * math.log((h_pd / (2.0 + h_pd)) * math.sqrt((a - 1.0) / (b - 1.0)))
    return term1 + term2
