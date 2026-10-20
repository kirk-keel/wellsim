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

__all__ = ["darcy_friction_factor", "reynolds_number", "RE_LAMINAR"]

RE_LAMINAR = 2300.0
_LBM_FT_S_PER_CP = 6.719689e-4  # 1 cp = 6.7197e-4 lbm/(ft s)


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
