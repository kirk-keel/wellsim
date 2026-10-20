"""Operating point: where inflow pwf(q) meets outflow pwf(q) (spec 4.6).

A scan over a log-spaced rate grid finds every sign change of

    g(q) = pwf_ipr(q) - pwf_vlp(q)

then Brent's method refines each one. An intersection is stable when g goes from
positive to negative with increasing rate (a small rate increase is choked back by
the outflow). Left-hand intersections (negative to positive) are unstable and are
reported, not silently chosen. If g < 0 everywhere the outflow needs more pressure
than the reservoir can give at any rate: the well does not flow.

Either curve may return None where it cannot be evaluated (rate above AOF, march
failure); those scan points are skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from wellsim.core.numerics import brent

__all__ = ["Intersection", "NodalResult", "NodalError", "find_operating_point", "find_operating_point_pwf"]

Curve = Callable[[float], "float | None"]


@dataclass(frozen=True)
class Intersection:
    q: float
    pwf: float
    stable: bool  # falling crossing of f(pwf) (equivalently d pwf_vlp/dq > d pwf_ipr/dq)
    dvlp_dq: float | None = None  # M3-3: d pwf_vlp / d q_g at the root by central difference, psi per Mscf/d
    dipr_dq: float | None = None  # M3-3: d pwf_ipr / d q_g at the root by central difference

    def as_dict(self) -> dict:
        return {"q": self.q, "pwf": self.pwf, "stable": self.stable, "dvlp_dq": self.dvlp_dq, "dipr_dq": self.dipr_dq}


@dataclass
class NodalResult:
    q_scan: list[float]
    pwf_ipr: list[float | None]
    pwf_vlp: list[float | None]
    intersections: list[Intersection]
    operating: Intersection | None
    status: str  # "ok", "no_flow", "no_intersection", "unstable_only"
    warnings: list[str] = field(default_factory=list)
    space: str = "rate"  # "rate": scan/Brent in q; "pwf": scan/Brent in pwf (q_scan holds q_ipr(pwf))
    residual_psi: float | None = None  # |pwf_vlp - pwf_ipr| at the operating point

    def as_dict(self) -> dict:
        return {
            "q_scan": self.q_scan, "pwf_ipr": self.pwf_ipr, "pwf_vlp": self.pwf_vlp,
            "intersections": [i.as_dict() for i in self.intersections],
            "operating": self.operating.as_dict() if self.operating else None,
            "status": self.status, "warnings": list(self.warnings),
            "space": self.space, "residual_psi": self.residual_psi,
        }


def find_operating_point(
    pwf_ipr: Curve,
    pwf_vlp: Curve,
    q_min: float,
    q_max: float,
    *,
    n_scan: int = 16,
    rate_tol: float = 1e-3,
    pressure_tol: float = 0.5,
) -> NodalResult:
    """Dual convergence at each intersection: the rate bracket is within ``rate_tol``
    (relative) AND the IPR-minus-VLP pressure residual is within ``pressure_tol`` (psi)."""
    if not (0.0 < q_min < q_max):
        raise ValueError("need 0 < q_min < q_max")
    n_scan = max(4, int(n_scan))
    ratio = (q_max / q_min) ** (1.0 / (n_scan - 1))
    q_scan = [q_min * ratio**i for i in range(n_scan)]
    q_scan[-1] = q_max
    ipr_vals: list[float | None] = []
    vlp_vals: list[float | None] = []
    warnings: list[str] = []
    cache: dict[float, float | None] = {}

    def g(q: float) -> float | None:
        if q in cache:
            return cache[q]
        a = pwf_ipr(q)
        b = pwf_vlp(q)
        val = None if (a is None or b is None) else a - b
        cache[q] = val
        return val

    for q in q_scan:
        a = pwf_ipr(q)
        b = pwf_vlp(q)
        ipr_vals.append(a)
        vlp_vals.append(b)
        cache[q] = None if (a is None or b is None) else a - b

    valid = [(q, cache[q]) for q in q_scan if cache[q] is not None]
    n_invalid = n_scan - len(valid)
    if n_invalid:
        warnings.append(f"{n_invalid} of {n_scan} scan rates could not be evaluated on one of the curves")
    if len(valid) < 2:
        return NodalResult(q_scan, ipr_vals, vlp_vals, [], None, "no_intersection", warnings +
                           ["fewer than two valid scan points"])

    intersections: list[Intersection] = []
    for (qa, ga), (qb, gb) in zip(valid, valid[1:]):
        if ga == 0.0:
            intersections.append(Intersection(qa, pwf_ipr(qa), gb < 0.0))
            continue
        if ga * gb > 0.0:
            continue

        def gg(q: float) -> float:
            v = g(q)
            if v is None:
                raise RuntimeError(f"curve not evaluable at q = {q}")
            return v

        try:
            q_root = brent(gg, qa, qb, fa=ga, fb=gb, xtol=1e-9 * q_max, rtol=rate_tol, ftol=pressure_tol)
        except RuntimeError as e:
            warnings.append(str(e))
            continue
        pwf = pwf_ipr(q_root)
        if pwf is None:
            continue
        intersections.append(Intersection(q_root, pwf, stable=ga > 0.0 > gb))

    if not intersections:
        if all(gv < 0.0 for _, gv in valid):
            status = "no_flow"
            warnings.append("outflow pressure exceeds inflow pressure at every scanned rate: the well does not flow")
        else:
            status = "no_intersection"
            warnings.append("no intersection found on the scanned rate range")
        return NodalResult(q_scan, ipr_vals, vlp_vals, [], None, status, warnings)

    stable = [i for i in intersections if i.stable]
    if len(intersections) > 1:
        warnings.append(
            f"{len(intersections)} intersections found; {len(stable)} stable, "
            f"{len(intersections) - len(stable)} unstable (left-hand)"
        )
    if stable:
        op = max(stable, key=lambda i: i.q)
        status = "ok"
    else:
        op = max(intersections, key=lambda i: i.q)
        status = "unstable_only"
        warnings.append("only unstable intersections found; reported operating point is not a steady state")
    return NodalResult(q_scan, ipr_vals, vlp_vals, intersections, op, status, warnings)


# --- pwf space (default since the M2.5-3b closeout) ---------------------------------------
#
#     f(pwf) = pwf_vlp(q_ipr(pwf)) - pwf          on [p_amb, p_res]
#
# q_ipr(pwf) is a plain function (the IPR is capped at its maximum below p_aof), so f is
# single-valued whatever the IPR's shape and no IPR inversion is needed. f(p_amb) >= 0
# always (the outflow needs at least ambient pressure at the sandface; +inf when the
# march cannot pass the rate), and f(p_res) = p_static - p_res < 0 whenever the well can
# flow, so the ordinary case is one root. Stability: f falling through zero with
# increasing pwf is the stable (right-hand) intersection of the rate-space picture.


class NodalError(RuntimeError):
    """The outflow curve is discontinuous where it must be continuous."""


def _root_pwf(f: Callable[[float], float], a: float, fa: float, b: float, fb: float) -> float:
    """Root of f on [a, b] where f(a) > 0 (possibly +inf: no outflow state exists below the
    reservoir-pressure ceiling at that rate) and f(b) < 0. Bisects the +inf end until it
    is finite, then Brent. By continuity f cannot jump from +inf straight to a negative
    value (as the ceiling is approached the outflow pressure rises to it, so f passes
    through positive finite values); if the bisection ever finds such a jump, raise."""
    for _ in range(200):
        if math.isfinite(fa):
            return brent(f, a, b, fa=fa, fb=fb, xtol=1e-6, rtol=1e-12, ftol=1e-6)
        if b - a <= 1e-9 * max(1.0, abs(b)):
            raise NodalError(
                f"outflow curve jumps from not-evaluable (+inf) at pwf {a:.6f} psia to f = {fb:.3f} psi at "
                f"pwf {b:.6f} psia with no finite positive value between: the outflow is discontinuous, "
                "which the physics forbids"
            )
        mid = 0.5 * (a + b)
        fm = f(mid)
        if fm <= 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
    raise NodalError("nodal: could not find a finite bracket end after 200 bisections")


def find_operating_point_pwf(
    q_ipr: Callable[[float], float],
    pwf_vlp: Callable[[float], float] | None,
    p_amb: float,
    p_res: float,
    *,
    n_scan: int = 16,
    pressure_tol: float = 0.5,
    rate_tol: float = 1e-3,
    outflow_pwf: Callable[[float], float] | None = None,
    derivatives: bool = False,
    scan_only: bool = False,
) -> NodalResult:
    """Root of f(pwf) = pwf_vlp(q_ipr(pwf)) - pwf by scan plus Brent in pwf.

    ``pwf_vlp`` must return +inf where the wellbore cannot pass the rate. Brent converges
    to 1e-6 psi in pwf and 1e-6 psi in residual, far inside ``pressure_tol``; the result's
    ``residual_psi`` reports the final |f| and a warning is issued if the rate bracket at the
    root is wider than ``rate_tol`` (it never is for a smooth IPR).

    M3-3: ``outflow_pwf`` (a function of the sandface pressure, for outflows whose liquid
    rate depends on pwf) replaces ``pwf_vlp(q_ipr(pwf))`` when given. ``derivatives`` adds
    d pwf_vlp/dq and d pwf_ipr/dq at every root by central difference (two extra outflow
    evaluations per root) and warns if the derivative test disagrees with the crossing
    direction. Every root is returned; the highest-rate stable one is the operating point.
    ``scan_only`` (M3-4 kill search) records each sign change at its bracket midpoint
    without the Brent refinement: it answers "is there any root" at scan cost.
    """
    if not (0.0 < p_amb < p_res):
        raise ValueError("need 0 < p_amb < p_res")
    if pwf_vlp is None and outflow_pwf is None:
        raise ValueError("need pwf_vlp or outflow_pwf")

    def vlp_at(pwf: float):
        return outflow_pwf(pwf) if outflow_pwf is not None else pwf_vlp(q_ipr(pwf))
    n_scan = max(4, int(n_scan))
    p_scan = [p_amb + (p_res - p_amb) * i / (n_scan - 1) for i in range(n_scan)]
    p_scan[-1] = p_res
    q_vals: list[float] = []
    vlp_vals: list[float | None] = []
    f_vals: list[float] = []
    warnings: list[str] = []
    cache: dict[float, float] = {}

    def f(pwf: float) -> float:
        if pwf in cache:
            return cache[pwf]
        v = vlp_at(pwf)
        val = math.inf if v is None else v - pwf
        cache[pwf] = val
        return val

    for pwf in p_scan:
        q = q_ipr(pwf)
        v = vlp_at(pwf)
        q_vals.append(q)
        vlp_vals.append(None if (v is None or not math.isfinite(v)) else v)
        cache[pwf] = math.inf if (v is None or not math.isfinite(v)) else v - pwf
        f_vals.append(cache[pwf])

    n_inf = sum(1 for v in f_vals if not math.isfinite(v))
    if n_inf:
        warnings.append(f"{n_inf} of {n_scan} scanned sandface pressures need a rate the wellbore cannot pass")

    intersections: list[Intersection] = []
    for (pa, fa), (pb, fb) in zip(zip(p_scan, f_vals), zip(p_scan[1:], f_vals[1:])):
        if fa == 0.0:
            intersections.append(Intersection(q_ipr(pa), pa, fb < 0.0))
            continue
        if fa * fb > 0.0 or (not math.isfinite(fa) and not math.isfinite(fb)):
            continue
        stable = fa > 0.0 > fb  # f falls through zero with increasing pwf
        if scan_only:
            root = 0.5 * (pa + pb)
        elif stable:
            root = _root_pwf(f, pa, fa, pb, fb)  # NodalError propagates: no guessing
        else:
            # rising crossing (unstable): f(pa) < 0 < f(pb); a +inf end here would mean the
            # outflow becomes unattainable as the rate *falls*, which the physics forbids
            if not math.isfinite(fb):
                raise NodalError(
                    f"outflow curve rises from f = {fa:.3f} psi at pwf {pa:.3f} psia to not-evaluable (+inf) "
                    f"at pwf {pb:.3f} psia: unattainable at a lower rate than an attainable one"
                )
            root = brent(f, pa, pb, fa=fa, fb=fb, xtol=1e-6, rtol=1e-12, ftol=1e-6)
        intersections.append(Intersection(q_ipr(root), root, stable))

    if derivatives and intersections:
        tagged: list[Intersection] = []
        dp = max(1e-3 * (p_res - p_amb), 1e-3)
        for i in intersections:
            pa, pb = max(p_amb, i.pwf - dp), min(p_res, i.pwf + dp)
            qa, qb = q_ipr(pa), q_ipr(pb)
            va, vb = vlp_at(pa), vlp_at(pb)
            ok = (va is not None and vb is not None and math.isfinite(va) and math.isfinite(vb) and qa != qb)
            if not ok:
                warnings.append(f"derivatives at the root pwf {i.pwf:.1f} psia not evaluable (outflow not finite on both sides)")
                tagged.append(i)
                continue
            dvlp = (vb - va) / (qb - qa)
            dipr = (pb - pa) / (qb - qa)
            stable_d = dvlp > dipr
            if stable_d != i.stable:
                warnings.append(
                    f"stability by derivatives (d pwf_vlp/dq {dvlp:.4g} vs d pwf_ipr/dq {dipr:.4g} psi per Mscf/d) "
                    f"disagrees with the crossing direction at pwf {i.pwf:.1f} psia; the derivative test is used"
                )
            tagged.append(Intersection(i.q, i.pwf, stable_d, dvlp, dipr))
        intersections = tagged

    if not intersections:
        if f_vals[-1] >= 0.0:
            status = "no_flow"
            warnings.append("static outflow column exceeds reservoir pressure: the well does not flow")
        else:
            status = "no_intersection"
            warnings.append("no intersection found on the sandface pressure range")
        return NodalResult(q_vals, p_scan, vlp_vals, [], None, status, warnings, space="pwf")

    stable_ones = [i for i in intersections if i.stable]
    if len(intersections) > 1:
        warnings.append(
            f"{len(intersections)} intersections found; {len(stable_ones)} stable, "
            f"{len(intersections) - len(stable_ones)} unstable"
        )
    for i in intersections:
        if not i.stable and stable_ones:
            warnings.append(
                f"unstable root at q {i.q:.1f} Mscf/d, pwf {i.pwf:.1f} psia (below it the well loads and dies); "
                "the operating point reported is the highest-rate stable root"
            )
    if stable_ones:
        op = max(stable_ones, key=lambda i: i.q)
        status = "ok"
    else:
        op = max(intersections, key=lambda i: i.q)
        status = "unstable_only"
        warnings.append("only unstable intersections found; reported operating point is not a steady state")
    residual = abs(f(op.pwf))
    if residual > pressure_tol:
        warnings.append(f"nodal residual {residual:.3f} psi exceeds pressure_tol {pressure_tol}")
    dq = abs(q_ipr(op.pwf - 1e-3) - q_ipr(op.pwf + 1e-3))
    if op.q > 0.0 and dq / op.q > rate_tol:
        warnings.append(f"rate changes by {dq / op.q:.2e} relative across 2e-3 psi at the operating point")
    return NodalResult(q_vals, p_scan, vlp_vals, intersections, op, status, warnings, space="pwf",
                       residual_psi=residual)
