"""Small root finders used by the solver. No scipy: the nodal solve and the choke
solve run once per case, but keeping them here avoids a scipy import in the hot path
(see the note on the scipy DLL block in the project memory) and keeps Monte Carlo cheap.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = ["brent", "bracket_upward"]


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    fa: float | None = None,
    fb: float | None = None,
    xtol: float = 1e-12,
    rtol: float = 1e-9,
    ftol: float = math.inf,
    maxiter: int = 100,
) -> float:
    """Brent-Dekker root of ``f`` on the bracket ``[a, b]`` (``f(a)`` and ``f(b)`` of opposite sign).

    Converges when the bracket is narrower than ``xtol + rtol * |x|`` AND ``|f(x)| <= ftol``
    (dual criterion; ``ftol`` defaults to no residual requirement). Pass ``fa``/``fb`` if
    already evaluated. Raises ValueError if the bracket is invalid.
    """
    fa = f(a) if fa is None else fa
    fb = f(b) if fb is None else fb
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError(f"brent: root not bracketed, f({a}) = {fa}, f({b}) = {fb}")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0.0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        # dual criterion: the loose x tolerance applies only once the residual is within ftol;
        # until then a much tighter x tolerance keeps the ordinary Brent steps (and the
        # bracket invariant) going instead of forcing tol1-sized steps out of the bracket
        if abs(fb) <= ftol:
            tol1 = 2.0 * (xtol + rtol * abs(b))
        else:
            tol1 = 2.0 * (1e-15 + 1e-12 * abs(b))
        xm = 0.5 * (c - b)
        if fb == 0.0 or abs(xm) <= tol1:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a, fa = b, fb
        if abs(d) > tol1:
            b += d
        else:
            b += math.copysign(tol1, xm)
        fb = f(b)
    return b


def bracket_upward(
    f: Callable[[float], float], x0: float, f0: float, *, factor: float = 2.0, max_doublings: int = 60
) -> tuple[float, float, float, float]:
    """Grow ``x`` geometrically from ``x0`` until ``f`` changes sign. Returns (a, fa, b, fb)."""
    a, fa = x0, f0
    b = x0 * factor if x0 > 0 else 1.0
    for _ in range(max_doublings):
        fb = f(b)
        if fa * fb <= 0.0:
            return a, fa, b, fb
        a, fa = b, fb
        b *= factor
    raise RuntimeError(f"bracket_upward: no sign change found up to x = {b}")
