"""mc-1: forward Monte Carlo on blowout rate, static-hold mud weight and kill rate.

What is sampled and why
    Reservoir pressure, net pay, the open interval and the exit condition are sampled as
    triangulars; kh is log-uniform. Those are the inputs a well-control job can defend:
    they come from offset wells, logs, the casing tally and the wellhead. A blowout rate
    read off a flare or a plume is not one of them, so it is never an input here. When
    there is a rate estimate it enters as ``mc.rate_band``, a FILTER on forward samples:
    it says which combinations of the defensible inputs are consistent with what was
    seen, and it never propagates its own error into the answer (contrast core/inverse.py,
    where inverting an observed rate to kh is explicit and the elasticity dlnq/dlnkh of
    about 0.3 means a 20% rate error is a factor 1.9 on kh).

    Permeability is derived, k = kh / net_pay, exactly as ``core/inverse.with_kh`` does,
    so beta and the non-Darcy D follow from k as in the forward solve. h_1
    (``exposed_top_offset``) is not sampled; it is held at the case value and enters the
    h_1 + h_p <= net_pay check.

Sampling
    Latin hypercube on the uniform scores (one stratified, permuted column per sampled
    dimension, from the master seed), then the inverse CDF of each distribution. A sample
    whose geometry violates h_1 + h_p <= net_pay has its geometry dimensions redrawn from
    their marginals with a seed derived from the sample index alone, so the draw is the
    same for any worker count; the resamples are counted and reported, and they do break
    the Latin hypercube stratification for those samples.

Rate pass
    One forward ``solver.solve`` per sample, each in its own process when workers > 1.
    Process, never thread: ``pvt.base.collect_range_warnings`` swaps a module-level global
    and installs a process-wide warnings filter, so the solver path is safe across
    processes and is not safe across threads. A sample that raises is recorded with its
    reason; a sample that solves but does not flow is recorded as a failure with that
    reason. Neither is ever dropped silently, and percentiles are computed from the
    survivors only. On Windows the pool spawns rather than forks, so a script that calls
    ``run`` with workers > 1 needs the usual ``if __name__ == "__main__":`` guard.

Static-hold mud weight
    rho_static = (p_r - p_exit) / (0.052 TVD) ppg for every survivor: the mud weight whose
    column exactly balances that sample's reservoir pressure on the absolute basis
    ``core/kill.static_hold`` uses. No solve needed.

Kill pass
    The survivors are stratified into ``n_kill`` equal-probability bins by q_g and one
    sample (the middle of the bin) is taken from each, so the kill rates reported span the
    rate distribution rather than clustering where the samples are dense. The kill samples
    are dealt round-robin, in q_g order, into ``chains``; each chain runs its searches in
    order and warm-starts every search from the previous q_k_min (the lever
    docs/TODO-M4.md left open one level up from ``kill_sweep``). A chain, not a worker, is
    the unit of warm-starting, so the answer does not depend on the worker count; changing
    ``chains`` does change the warm-start path, and the q_k_min values then move within
    the search's own 1% tolerance.

Reporting
    q_g, rho_static and q_k_min at the 10th, 50th and 90th percentiles, each labeled in
    words, because "P90" means the opposite thing to a reservoir engineer and to a
    driller. Spearman rank correlation of every sampled input against q_g and against
    q_k_min. Convergence as the percentiles at n/2 against n. Failures by reason,
    resample count, survivor count.

    rw-1 and rw-2: when the case carries a ``relief_well`` block the kill samples also
    report the pump pressure and hydraulic horsepower at their own kill rate and their own
    blowing-well pressure at z_inj, with the fraction of samples on vacuum or over a pump
    or rating limit, and the Ramey arrival temperature at that rate. No extra kill
    searches: all of it is arithmetic on a state the search already produced. The rw-2
    surface and arrival kill runs (a second search per sample) are forced off inside the
    Monte Carlo, so the reported kill rates are at the M3-4 injection temperature while the
    arrival temperature beside them is the Ramey bound at that rate; the report says so.

All COMPUTED: the tests are a degenerate reproduction of the deterministic solve,
reproducibility across worker counts, a one-input ranking check, a filter check and
failure accounting. No published Monte Carlo case is wired in.
"""

from __future__ import annotations

import math
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from wellsim.cases.schema import Case
from wellsim.core.kill import PSI_PER_PPG_FT

__all__ = [
    "MonteCarloError", "Sample", "SampleOutcome", "KillOutcome", "MonteCarloResult",
    "sample_inputs", "apply_sample", "run", "write_summary", "spearman",
    "DIMENSIONS", "PERCENTILES", "PERCENTILE_LABELS", "MAX_RESAMPLES", "MIN_SURVIVORS",
]

# Fixed order, so the Latin hypercube columns are stable for a given configuration.
DIMENSIONS = ("p_res", "net_pay", "h_p", "kh", "cd", "exit_id", "exit_scenario")
GEOMETRY_DIMENSIONS = ("net_pay", "h_p")
PERCENTILES = (10, 50, 90)
PERCENTILE_LABELS = {
    10: "exceeded with 90% probability",
    50: "exceeded with 50% probability (the median)",
    90: "exceeded with 10% probability",
}
MAX_RESAMPLES = 100  # per sample, before the geometry constraint is called impossible
MIN_SURVIVORS = 100  # below this the percentiles are warned about
_DEPTH_TOL_FT = 0.01

UNITS = {
    "p_res": "psia", "net_pay": "ft", "h_p": "ft", "kh": "md-ft", "cd": "-", "exit_id": "in",
    "q_g": "Mscf/d", "rho_static": "ppg", "q_k_min": "bbl/min",
    "p_pump": "psig", "hhp": "hhp", "t_arrival": "degF",
}
QUANTITY_WORDS = {
    "q_g": "blowout gas rate",
    "rho_static": "static-hold mud weight",
    "q_k_min": "minimum dynamic kill rate",
    "p_pump": "relief-well pump pressure at the kill rate",
    "hhp": "relief-well hydraulic horsepower at the kill rate",
    "t_arrival": "kill-fluid arrival temperature at z_inj",
}


class MonteCarloError(ValueError):
    """The Monte Carlo cannot run on this case (message says why)."""


# --- samples ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One draw. ``values`` holds only the sampled dimensions, in canonical units."""
    index: int
    values: dict[str, float]
    scenario: str | None = None
    resamples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, **self.values,
                **({"scenario": self.scenario} if self.scenario is not None else {}),
                "resamples": self.resamples}


@dataclass
class SampleOutcome:
    index: int
    ok: bool
    reason: str | None = None
    q_g: float | None = None
    pwf: float | None = None
    choked: bool | None = None
    regime_exit: str | None = None
    rho_static: float | None = None
    in_band: bool = True
    seconds: float = 0.0


@dataclass
class KillOutcome:
    index: int
    ok: bool
    reason: str | None = None
    q_g: float | None = None
    q_k_min: float | None = None
    static_ok: bool | None = None
    static_margin_psi: float | None = None
    blount_soeiinah_bpm: float | None = None
    evaluations: int = 0
    seconds: float = 0.0
    # rw-1: the relief-well pump balance at this kill rate and this z_inj state
    p_zinj_psia: float | None = None
    p_pump_psig: float | None = None
    hhp: float | None = None
    on_vacuum: bool | None = None
    over_limit: bool | None = None
    # rw-2: the Ramey arrival temperature at this sample own kill rate
    t_arrival_degf: float | None = None
    a_relax_ft: float | None = None


# --- inverse CDFs -------------------------------------------------------------------------------


def _triangular_icdf(u: float, a: float, m: float, b: float) -> float:
    """Inverse CDF of the triangular(a, m, b). a == b returns a exactly (the degenerate
    case must reproduce the deterministic input bitwise, so it never goes near a sqrt)."""
    if b == a:
        return a
    fc = (m - a) / (b - a)
    if u <= fc:
        return a + math.sqrt(u * (b - a) * (m - a))
    return b - math.sqrt((1.0 - u) * (b - a) * (b - m))


def _loguniform_icdf(u: float, lo: float, hi: float) -> float:
    """Inverse CDF of the log-uniform on [lo, hi]. lo == hi returns lo exactly."""
    if hi == lo:
        return lo
    return lo * (hi / lo) ** u


def _discrete_index(u: float, weights: Sequence[float]) -> int:
    cum = np.cumsum(np.asarray(weights, float))
    cum /= cum[-1]
    return int(min(np.searchsorted(cum, u, side="left"), len(weights) - 1))


def _h_p_is_sampled(mc) -> bool:
    """h_p has its own column only when it is a triangular; 'net_pay' ties it to net pay
    and needs no column of its own."""
    return mc.exposed_interval is not None and not isinstance(mc.exposed_interval, str)


def _configured_dimensions(mc) -> list[str]:
    """The sampled dimensions, in DIMENSIONS order so the hypercube columns are stable."""
    present = {
        "p_res": mc.p_res is not None,
        "net_pay": mc.net_pay is not None,
        "h_p": _h_p_is_sampled(mc),
        "kh": mc.kh is not None,
        "cd": mc.cd is not None,
        "exit_id": mc.exit_id is not None,
        "exit_scenario": mc.exit_scenarios is not None,
    }
    return [d for d in DIMENSIONS if present[d]]


def _lhs_scores(n: int, d: int, seed: int) -> np.ndarray:
    """Latin hypercube scores in (0, 1): one stratified, independently permuted column
    per dimension."""
    rng = np.random.default_rng(seed)
    scores = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        u = rng.random(n)
        scores[:, j] = (perm + u) / n
    return scores


def _draw(mc, dim: str, u: float) -> float | int:
    u = float(u)
    if dim == "p_res":
        return float(_triangular_icdf(u, mc.p_res.min, mc.p_res.mode, mc.p_res.max))
    if dim == "net_pay":
        return float(_triangular_icdf(u, mc.net_pay.min, mc.net_pay.mode, mc.net_pay.max))
    if dim == "h_p":
        t = mc.exposed_interval
        return float(_triangular_icdf(u, t.min, t.mode, t.max))
    if dim == "kh":
        return float(_loguniform_icdf(u, mc.kh.low, mc.kh.high))
    if dim == "cd":
        return float(_triangular_icdf(u, mc.cd.min, mc.cd.mode, mc.cd.max))
    if dim == "exit_id":
        return float(_triangular_icdf(u, mc.exit_id.min, mc.exit_id.mode, mc.exit_id.max))
    if dim == "exit_scenario":
        return _discrete_index(u, [s.weight for s in mc.exit_scenarios])
    raise MonteCarloError(f"unknown Monte Carlo dimension {dim!r}")


def _geometry(case: Case, values: dict[str, float]) -> tuple[float, float, float]:
    """(net_pay, h_p, h_1) for a draw: the sampled values where given, the case values
    otherwise. h_p follows net pay when the case is fully penetrated or mc ties it."""
    res = case.reservoir
    mc = case.mc
    net_pay = values.get("net_pay", res.net_pay)
    h_1 = res.exposed_top_offset
    if "h_p" in values:
        h_p = values["h_p"]
    elif isinstance(mc.exposed_interval, str) or res.exposed_interval is None:
        h_p = net_pay  # 'net_pay', or a fully penetrated case whose pay is being sampled
    else:
        h_p = res.exposed_interval
    return net_pay, h_p, h_1


def _geometry_ok(case: Case, values: dict[str, float]) -> bool:
    net_pay, h_p, h_1 = _geometry(case, values)
    return h_p <= net_pay + _DEPTH_TOL_FT and h_1 + h_p <= net_pay + _DEPTH_TOL_FT


def sample_inputs(case: Case) -> list[Sample]:
    """The Latin hypercube draw for ``case.mc``. Deterministic in ``mc.seed`` alone."""
    mc = case.mc
    if mc is None:
        raise MonteCarloError("the case has no mc block; nothing to sample")
    dims = _configured_dimensions(mc)
    scores = _lhs_scores(mc.n_samples, len(dims), mc.seed)
    resamplable = [d for d in dims if d in GEOMETRY_DIMENSIONS]
    samples: list[Sample] = []
    for i in range(mc.n_samples):
        values = {d: _draw(mc, d, scores[i, j]) for j, d in enumerate(dims)}
        tries = 0
        if not _geometry_ok(case, values):
            if not resamplable:
                net_pay, h_p, h_1 = _geometry(case, values)
                raise MonteCarloError(
                    f"sample {i}: h_1 {h_1:g} + h_p {h_p:g} exceeds net pay {net_pay:g} ft and neither net_pay "
                    "nor h_p is sampled, so no redraw can fix it; the fixed inputs contradict each other"
                )
            rng = np.random.default_rng([mc.seed, i])  # index-derived: independent of the worker count
            while tries < MAX_RESAMPLES and not _geometry_ok(case, values):
                tries += 1
                for d in resamplable:
                    values[d] = _draw(mc, d, float(rng.random()))
            if not _geometry_ok(case, values):
                net_pay, h_p, h_1 = _geometry(case, values)
                raise MonteCarloError(
                    f"sample {i}: the geometry constraint h_1 + h_p <= net_pay was still violated after "
                    f"{MAX_RESAMPLES} redraws (h_1 {h_1:g}, h_p {h_p:g}, net pay {net_pay:g} ft); the ranges "
                    "given cannot satisfy it"
                )
        scenario = None
        if "exit_scenario" in values:
            scenario = mc.exit_scenarios[int(values.pop("exit_scenario"))]
            if scenario.cd is not None:
                values["cd"] = scenario.cd
            if scenario.exit_id is not None:
                values["exit_id"] = scenario.exit_id
            scenario = scenario.name
        samples.append(Sample(index=i, values=values, scenario=scenario, resamples=tries))
    return samples


def apply_sample(case: Case, s: Sample, *, for_kill: bool = False) -> Case:
    """A plain forward case with this sample's values written in.

    The rate pass always clears ``dynamic_kill`` so a stray kill block on the case cannot
    put a 4 s kill search inside every one of n samples; the kill pass writes the block
    from ``mc.kill``.
    """
    mc = case.mc
    d = case.model_dump()
    net_pay, h_p, _ = _geometry(case, s.values)
    d["reservoir"]["net_pay"] = net_pay
    # None is full penetration: keep it None so a fully penetrated sample is exactly the
    # deterministic case, with no partial-penetration pseudo-skin path taken
    d["reservoir"]["exposed_interval"] = None if h_p >= net_pay - _DEPTH_TOL_FT else h_p
    if "p_res" in s.values:
        d["reservoir"]["pressure"] = s.values["p_res"]
    if "kh" in s.values:
        d["reservoir"]["permeability"] = s.values["kh"] / net_pay
    if "cd" in s.values:
        d["exit"]["cd"] = s.values["cd"]
    if "exit_id" in s.values:
        d["exit"]["exit_id"] = s.values["exit_id"]
    d["mc"] = None
    if d.get("relief_well") is not None:
        if for_kill:
            # rw-3b: each kill sample now runs the surface search and then the finite-U
            # arrival search, so the reported kill rate is at the arrival temperature the
            # string actually delivers rather than the M3-4 stream temperature. Two
            # searches per sample; the U -> infinity bound run and the geothermal run are
            # both off, so it is two rather than four.
            d["relief_well"]["temperature_runs"] = True
            d["relief_well"]["finite_u"] = True
            d["relief_well"]["bound_run"] = False
            d["relief_well"]["geothermal_bound"] = False
        else:
            # the rate pass has no kill and so no pump balance; the relief-well block goes
            # with the dynamic_kill block it depends on (the schema requires them together)
            d["relief_well"] = None
    if for_kill and mc.kill is not None:
        k = mc.kill
        d["dynamic_kill"] = {
            "rho_k": k.rho_k, "mu_k": k.mu_k, "cp_k": k.cp_k, "z_inj": k.z_inj,
            "rate_bpm": None, "find_min_rate": True,
        }
    else:
        d["dynamic_kill"] = None
    return Case(**d)


def static_hold_mud_weight(case: Case, p_res: float) -> float:
    """rho_static = (p_r - p_exit) / (0.052 TVD), ppg: the mud weight whose column balances
    p_r on the absolute basis of ``core/kill.static_hold``. No solve needed."""
    tvd = case.well.md_to_tvd(case.reservoir.depth_md)
    return (p_res - case.exit.ambient_pressure) / (PSI_PER_PPG_FT * tvd)


# --- workers ------------------------------------------------------------------------------------

_BASE: dict[str, Any] | None = None  # per-process base case dict, set by the pool initializer


def _init_worker(base_dict: dict[str, Any]) -> None:
    global _BASE
    _BASE = base_dict


def _base_case() -> Case:
    if _BASE is None:
        raise MonteCarloError("worker base case not initialised")
    return Case(**_BASE)


def _rate_one(payload: dict[str, Any]) -> SampleOutcome:
    """One forward solve. Never raises: a refusal comes back as ok=False with its reason."""
    from wellsim.core.solver import solve

    s = Sample(index=payload["index"], values=payload["values"], scenario=payload.get("scenario"),
               resamples=payload.get("resamples", 0))
    t0 = time.perf_counter()
    base = _base_case()
    try:
        case = apply_sample(base, s)
        r = solve(case)
    except Exception as e:  # solver refusal (dense-phase, supersonic, area change), or a bad case
        return SampleOutcome(index=s.index, ok=False, reason=f"{type(e).__name__}: {str(e)[:200]}",
                             seconds=time.perf_counter() - t0)
    if not r.flowing:
        return SampleOutcome(index=s.index, ok=False,
                             reason="no operating point: the well does not flow at this sample",
                             seconds=time.perf_counter() - t0)
    regime = "gas"
    prof = r.profile
    if prof is not None and getattr(prof, "regime", None) is not None:
        regime = str(prof.regime[0])  # the march runs exit -> sandface, so index 0 is the exit
    return SampleOutcome(
        index=s.index, ok=True, q_g=float(r.gas_rate), pwf=float(r.pwf),
        choked=bool(r.exit.choked) if r.exit is not None else None, regime_exit=regime,
        rho_static=static_hold_mud_weight(case, case.reservoir.pressure),
        seconds=time.perf_counter() - t0,
    )


def _arrival_of(rw: dict[str, Any]) -> dict[str, Any]:
    """The arrival-temperature block of a relief-well analysis, wherever it sits: under
    ``temperature.estimate`` when the kill search ran at it (rw-3b, the Monte Carlo path),
    under ``arrival`` when it was only reported beside a search at another temperature."""
    temp = rw.get("temperature") or {}
    return temp.get("estimate") or temp.get("arrival") or rw.get("arrival") or {}


def _kill_chain(chain: list[dict[str, Any]]) -> list[KillOutcome]:
    """One warm-start chain: the searches run in q_g order, each starting its bracket from
    the previous q_k_min."""
    from wellsim.core.kill import kill_analysis
    from wellsim.core.pvt import collect_range_warnings, make_pvt
    from wellsim.core.solver import build_ipr

    base = _base_case()
    out: list[KillOutcome] = []
    q_prev: float | None = None
    for payload in chain:
        s = Sample(index=payload["index"], values=payload["values"], scenario=payload.get("scenario"))
        t0 = time.perf_counter()
        try:
            case = apply_sample(base, s, for_kill=True)
            with collect_range_warnings():
                pvt = make_pvt(case.fluid, case.options.correlations)
                ipr, _ = build_ipr(case, pvt)
                a = kill_analysis(case, pvt, ipr, q_start=q_prev)
        except Exception as e:
            out.append(KillOutcome(index=s.index, ok=False, q_g=payload.get("q_g"),
                                   reason=f"{type(e).__name__}: {str(e)[:200]}",
                                   seconds=time.perf_counter() - t0))
            continue
        q_k = a.get("q_k_min", {}).get("q_k_min_bpm")
        st = a["static_hold"]
        rw = a.get("relief_well") or {}
        at_kill = rw.get("at_kill") or {}
        out.append(KillOutcome(
            index=s.index, ok=q_k is not None, q_g=payload.get("q_g"), q_k_min=q_k,
            reason=None if q_k is not None else a.get("q_k_min", {}).get("note", "no kill rate returned"),
            static_ok=bool(st["ok"]), static_margin_psi=float(st["margin_psi"]),
            blount_soeiinah_bpm=float(a["blount_soeiinah"]["q_bpm"]),
            evaluations=int(a.get("q_k_min", {}).get("evaluations", 0)),
            seconds=time.perf_counter() - t0,
            p_zinj_psia=rw.get("p_zinj_psia"),
            p_pump_psig=at_kill.get("p_pump_psig"),
            hhp=at_kill.get("hhp"),
            on_vacuum=at_kill.get("on_vacuum"),
            over_limit=(any(v is True for v in at_kill["limits"].values()) if at_kill else None),
            t_arrival_degf=_arrival_of(rw).get("t_arrival_degf"),
            a_relax_ft=_arrival_of(rw).get("a_relax_ft"),
        ))
        if q_k is not None and math.isfinite(q_k) and q_k > 0.0:
            q_prev = q_k
    return out


def _map(fn: Callable, items: list, workers: int, base_dict: dict[str, Any]) -> list:
    """``fn`` over ``items``, in this process when workers == 1 and in a process pool
    otherwise. Processes, never threads: collect_range_warnings swaps a module global and
    the process-wide warnings filter."""
    if workers <= 1 or len(items) <= 1:
        _init_worker(base_dict)
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(base_dict,)) as ex:
        return list(ex.map(fn, items))


# --- statistics ---------------------------------------------------------------------------------


def _ranks(a: Sequence[float]) -> np.ndarray:
    """Ranks with ties averaged (the Spearman convention)."""
    x = np.asarray(a, float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1.0, len(x) + 1.0)
    # average the ranks within each run of equal values
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i + 1
        while j < len(xs) and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation. A constant input (a degenerate range, or an exit
    scenario that never varies) has no rank order and reads exactly zero."""
    if len(x) < 2:
        return 0.0
    rx, ry = _ranks(x), _ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _percentiles(values: Sequence[float]) -> dict[str, Any] | None:
    if not len(values):
        return None
    v = np.asarray(values, float)
    out: dict[str, Any] = {"n": int(v.size), "mean": float(v.mean()),
                           "min": float(v.min()), "max": float(v.max())}
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(v, p))
        out[f"p{p}_label"] = PERCENTILE_LABELS[p]
    return out


def _convergence(values: Sequence[float]) -> dict[str, Any] | None:
    """The three percentiles on the first half of the values against all of them, and the
    largest relative change. Values are in sample order, so the half is a real prefix of
    the run, not a reordering."""
    v = np.asarray(values, float)
    if v.size < 4:
        return None
    half = v[: v.size // 2]
    rows = {}
    worst = 0.0
    for p in PERCENTILES:
        a, b = float(np.percentile(half, p)), float(np.percentile(v, p))
        rel = abs(b - a) / abs(b) if b != 0.0 else (0.0 if a == 0.0 else math.inf)
        rows[f"p{p}"] = {"half": a, "full": b, "rel_change": rel}
        worst = max(worst, rel)
    return {"n_half": int(half.size), "n_full": int(v.size), "percentiles": rows,
            "largest_rel_change": worst}


# --- result -------------------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    case_name: str
    seed: int
    n_samples: int
    workers: int
    chains: int
    samples: list[dict[str, Any]] = field(default_factory=list)
    kill_rows: list[dict[str, Any]] = field(default_factory=list)
    solved: int = 0
    survivors: int = 0
    filtered_out: int = 0
    resamples: int = 0
    failures: dict[str, int] = field(default_factory=dict)
    kill_failures: dict[str, int] = field(default_factory=dict)
    percentiles: dict[str, Any] = field(default_factory=dict)
    ranking: dict[str, list] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=dict)
    static_fail_fraction: float | None = None
    vacuum_fraction: float | None = None
    over_limit_fraction: float | None = None
    rho_k_ppg: float | None = None
    elapsed_rate_s: float = 0.0
    elapsed_kill_s: float = 0.0
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def value(self, quantity: str, p: int) -> float | None:
        d = self.percentiles.get(quantity)
        return None if d is None else d[f"p{p}"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name, "seed": self.seed, "n_samples": self.n_samples,
            "workers": self.workers, "chains": self.chains, "solved": self.solved,
            "survivors": self.survivors, "filtered_out": self.filtered_out, "resamples": self.resamples,
            "failures": dict(self.failures), "kill_failures": dict(self.kill_failures),
            "percentiles": self.percentiles, "ranking": {k: [list(t) for t in v] for k, v in self.ranking.items()},
            "convergence": self.convergence, "static_fail_fraction": self.static_fail_fraction,
            "vacuum_fraction": self.vacuum_fraction, "over_limit_fraction": self.over_limit_fraction,
            "rho_k_ppg": self.rho_k_ppg, "elapsed_rate_s": self.elapsed_rate_s,
            "elapsed_kill_s": self.elapsed_kill_s, "elapsed_s": self.elapsed_s,
            "samples": self.samples, "kill_rows": self.kill_rows,
            "warnings": list(self.warnings), "notes": list(self.notes),
        }


def _bin_representatives(survivors: list[dict[str, Any]], n_kill: int) -> list[dict[str, Any]]:
    """One survivor per equal-probability bin of q_g: sort by q_g, split into n_kill
    contiguous groups of as near equal count as possible, take the middle of each."""
    ordered = sorted(survivors, key=lambda r: (r["q_g"], r["index"]))
    if len(ordered) <= n_kill:
        return ordered
    picks = []
    for group in np.array_split(np.arange(len(ordered)), n_kill):
        if group.size:
            picks.append(ordered[int(group[group.size // 2])])
    return picks


def run(case: Case, *, workers: int | None = None, kill: bool = True,
        progress: Callable[[str], None] | None = None) -> MonteCarloResult:
    """The whole Monte Carlo: sample, rate pass, static-hold mud weight, kill pass, stats."""
    t_all = time.perf_counter()
    mc = case.mc
    if mc is None:
        raise MonteCarloError("the case has no mc block; nothing to run")
    n_workers = int(workers if workers is not None else mc.workers)
    chains = mc.kill.chains if mc.kill is not None else 1
    result = MonteCarloResult(case_name=case.name, seed=mc.seed, n_samples=mc.n_samples,
                              workers=n_workers, chains=chains)
    samples = sample_inputs(case)
    result.resamples = sum(s.resamples for s in samples)
    dims = sorted({d for s in samples for d in s.values})
    result.notes.append("sampled inputs: " + ", ".join(f"{d} ({UNITS.get(d, '')})" for d in dims))
    if result.resamples:
        result.notes.append(
            f"{result.resamples} geometry redraws for the h_1 + h_p <= net pay constraint; those samples "
            "are drawn from the marginals, so the Latin hypercube stratification does not hold for them"
        )
    base_dict = case.model_dump()

    # --- rate pass
    t0 = time.perf_counter()
    payloads = [{"index": s.index, "values": s.values, "scenario": s.scenario, "resamples": s.resamples}
                for s in samples]
    outcomes = _map(_rate_one, payloads, n_workers, base_dict)
    result.elapsed_rate_s = time.perf_counter() - t0
    if progress:
        progress(f"rate pass: {len(outcomes)} samples in {result.elapsed_rate_s:.1f} s")

    band = mc.rate_band
    rows: list[dict[str, Any]] = []
    for s, o in zip(samples, outcomes):
        row = s.as_dict()
        row.update({"ok": o.ok, "reason": o.reason, "q_g": o.q_g, "pwf": o.pwf, "choked": o.choked,
                    "regime_exit": o.regime_exit, "rho_static": o.rho_static, "seconds": o.seconds})
        if o.ok and band is not None:
            row["in_band"] = bool(band.q_low <= o.q_g <= band.q_high)
        else:
            row["in_band"] = o.ok
        rows.append(row)
    result.samples = rows
    result.solved = sum(1 for r in rows if r["ok"])
    for r in rows:
        if not r["ok"]:
            result.failures[r["reason"]] = result.failures.get(r["reason"], 0) + 1
    survivors = [r for r in rows if r["ok"] and r["in_band"]]
    result.survivors = len(survivors)
    result.filtered_out = result.solved - result.survivors
    if band is not None:
        result.notes.append(
            f"rate band {band.q_low:,.0f} to {band.q_high:,.0f} Mscf/d applied as a filter on forward samples "
            f"(never an input): {result.filtered_out} of {result.solved} solved samples dropped"
        )
    if result.survivors == 0:
        result.warnings.append("no surviving samples: no percentiles can be reported")
        result.elapsed_s = time.perf_counter() - t_all
        return result
    if result.survivors < MIN_SURVIVORS:
        result.warnings.append(
            f"only {result.survivors} surviving samples (below {MIN_SURVIVORS}): the percentiles are thin, "
            "and the 10th and 90th rest on a handful of samples each"
        )

    # --- statistics on the rate pass
    result.percentiles["q_g"] = _percentiles([r["q_g"] for r in survivors])
    result.percentiles["rho_static"] = _percentiles([r["rho_static"] for r in survivors])
    result.convergence["q_g"] = _convergence([r["q_g"] for r in survivors])
    result.convergence["rho_static"] = _convergence([r["rho_static"] for r in survivors])
    result.ranking["q_g"] = _rank_inputs(survivors, dims, "q_g")

    # --- kill pass
    if kill and mc.kill is not None:
        result.rho_k_ppg = mc.kill.rho_k
        picks = _bin_representatives(survivors, mc.kill.n_kill)
        ordered = sorted(picks, key=lambda r: (r["q_g"], r["index"]))
        chain_lists: list[list[dict[str, Any]]] = [[] for _ in range(chains)]
        for i, r in enumerate(ordered):  # dealt round-robin: each chain stays q_g-ordered
            chain_lists[i % chains].append({"index": r["index"], "values": {k: r[k] for k in dims if k in r},
                                            "scenario": r.get("scenario"), "q_g": r["q_g"]})
        chain_lists = [c for c in chain_lists if c]
        t0 = time.perf_counter()
        chain_out = _map(_kill_chain, chain_lists, min(n_workers, len(chain_lists)), base_dict)
        result.elapsed_kill_s = time.perf_counter() - t0
        if progress:
            progress(f"kill pass: {len(ordered)} samples in {result.elapsed_kill_s:.1f} s")
        krows: list[dict[str, Any]] = []
        for chain in chain_out:
            for k in chain:
                krows.append({
                    "index": k.index, "ok": k.ok, "reason": k.reason, "q_g": k.q_g, "q_k_min": k.q_k_min,
                    "static_ok": k.static_ok, "static_margin_psi": k.static_margin_psi,
                    "blount_soeiinah_bpm": k.blount_soeiinah_bpm, "evaluations": k.evaluations,
                    "seconds": k.seconds, "p_zinj_psia": k.p_zinj_psia, "p_pump_psig": k.p_pump_psig,
                    "hhp": k.hhp, "on_vacuum": k.on_vacuum, "over_limit": k.over_limit,
                    "t_arrival_degf": k.t_arrival_degf, "a_relax_ft": k.a_relax_ft,
                })
        krows.sort(key=lambda r: r["index"])
        result.kill_rows = krows
        for r in krows:
            if not r["ok"]:
                result.kill_failures[r["reason"]] = result.kill_failures.get(r["reason"], 0) + 1
        good = [r for r in krows if r["ok"]]
        if good:
            result.percentiles["q_k_min"] = _percentiles([r["q_k_min"] for r in good])
            result.convergence["q_k_min"] = _convergence([r["q_k_min"] for r in good])
            by_index = {r["index"]: r for r in survivors}
            kill_samples = [by_index[r["index"]] for r in good]
            result.ranking["q_k_min"] = _rank_inputs(
                kill_samples, dims, None, y=[r["q_k_min"] for r in good]
            )
        # rw-1: the relief-well pump balance rides along on the kill samples; no new searches
        pumped = [r for r in good if r["p_pump_psig"] is not None]
        if pumped:
            result.percentiles["p_pump"] = _percentiles([r["p_pump_psig"] for r in pumped])
            result.percentiles["hhp"] = _percentiles([r["hhp"] for r in pumped])
            result.convergence["p_pump"] = _convergence([r["p_pump_psig"] for r in pumped])
            n_vac = sum(1 for r in pumped if r["on_vacuum"])
            n_over = sum(1 for r in pumped if r["over_limit"])
            result.vacuum_fraction = n_vac / len(pumped)
            result.over_limit_fraction = n_over / len(pumped)
            result.notes.append(
                f"relief-well pump balance on {len(pumped)} kill samples, each at its own q_k_min and its "
                "own blowing-well pressure at z_inj; no extra kill searches were run"
            )
            if n_vac:
                result.warnings.append(
                    f"{n_vac} of {len(pumped)} kill samples ({result.vacuum_fraction:.1%}) put the relief-well "
                    "string on vacuum at the kill rate: the kill fluid free-falls there and surface pressure "
                    "does not control the rate"
                )
                result.notes.append(
                    f"the {n_vac} samples on vacuum enter the pump-pressure and horsepower percentiles as "
                    "zero, which is what they need from the pumps; the low end of those distributions is "
                    "therefore a mix of low pump pressures and free-fall samples, not a continuum"
                )
            if n_over:
                result.warnings.append(
                    f"{n_over} of {len(pumped)} kill samples ({result.over_limit_fraction:.1%}) exceed a pump "
                    "or rating limit at the kill rate"
                )
        # rw-2: the Ramey arrival temperature at each sample own kill rate, no new search
        warmed = [r for r in good if r["t_arrival_degf"] is not None]
        if warmed:
            result.percentiles["t_arrival"] = _percentiles([r["t_arrival_degf"] for r in warmed])
            result.notes.append(
                f"kill-fluid arrival temperature at z_inj on {len(warmed)} kill samples, each at its own "
                "kill rate (A goes as the mass rate, so every sample gets its own relaxation length). "
                "rw-3b: the kill searches RUN at this temperature, two searches per sample, the first at "
                "the surface temperature to give the arrival temperature a mass rate and the second at "
                "the arrival temperature itself. The U -> infinity bound is not searched for inside the "
                "Monte Carlo"
            )
        decided = [r for r in krows if r["static_ok"] is not None]
        if decided:
            fails = sum(1 for r in decided if not r["static_ok"])
            result.static_fail_fraction = fails / len(decided)
            if fails:
                result.warnings.append(
                    f"static hold fails at {mc.kill.rho_k:g} ppg on {fails} of {len(decided)} kill samples "
                    f"({result.static_fail_fraction:.1%}): those wells are dynamically dead only"
                )
        binned = (f"one from each of {mc.kill.n_kill} equal-probability q_g bins"
                  if result.survivors > mc.kill.n_kill
                  else f"every survivor ({result.survivors} at or below n_kill = {mc.kill.n_kill})")
        result.notes.append(
            f"kill pass at {mc.kill.rho_k:g} ppg on {len(krows)} samples, {binned}, run in "
            f"{len(chain_lists)} warm-start chain(s): each search starts its bracket from the previous "
            "q_k_min in its chain. The chain, not the worker, is the unit of warm-starting, so the answer "
            "does not depend on the worker count; changing mc.kill.chains does change the warm-start path "
            "and moves q_k_min within the search's own 1% tolerance"
        )
    result.elapsed_s = time.perf_counter() - t_all
    return result


def _rank_inputs(rows: list[dict[str, Any]], dims: list[str], key: str | None,
                 y: Sequence[float] | None = None) -> list[tuple[str, float]]:
    """Spearman of every sampled input against the output, sorted by magnitude."""
    out = y if y is not None else [r[key] for r in rows]
    ranked = [(d, spearman([r[d] for r in rows], out)) for d in dims if all(d in r for r in rows)]
    ranked.sort(key=lambda t: -abs(t[1]))
    return ranked


# --- report -------------------------------------------------------------------------------------


def _bullet(text: str, width: int = 78) -> list[str]:
    """One bullet, wrapped so the report stays inside 80 columns."""
    return textwrap.wrap(text, width=width, initial_indent="- ", subsequent_indent="  ") or ["-"]


def _para(text: str, width: int = 80) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def _fmt(x: float | None, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    if not math.isfinite(x):
        return "inf"
    return f"{x:,.{digits}f}"


def summary_markdown(result: MonteCarloResult) -> str:
    """The report, every table inside 80 columns."""
    L: list[str] = []
    L.append(f"# Monte Carlo (mc-1): {result.case_name}")
    L.append("")
    L.append(f"seed {result.seed}, {result.n_samples} samples, {result.workers} worker(s), "
             f"{result.chains} kill chain(s)")
    L.append(f"solved {result.solved}, survivors {result.survivors}, filtered out "
             f"{result.filtered_out}, geometry redraws {result.resamples}")
    L.append(f"wall time {result.elapsed_s:.1f} s (rate pass {result.elapsed_rate_s:.1f} s, "
             f"kill pass {result.elapsed_kill_s:.1f} s)")
    L.append("")
    L.append("## Percentiles")
    L.append("")
    for q in ("q_g", "rho_static", "q_k_min", "p_pump", "hhp", "t_arrival"):
        d = result.percentiles.get(q)
        if d is None:
            continue
        digits = 0 if q in ("q_g", "p_pump", "hhp") else 2
        if q == "t_arrival":
            digits = 1
        L.append(f"{QUANTITY_WORDS[q]}, {UNITS[q]} (n = {d['n']})")
        L.append("")
        L.append("  value          meaning")
        L.append("  " + "-" * 62)
        for p in PERCENTILES:
            L.append(f"  {_fmt(d[f'p{p}'], digits):>13s}  {PERCENTILE_LABELS[p]}")
        L.append(f"  {'':>13s}  (range {_fmt(d['min'], digits)} to {_fmt(d['max'], digits)}, "
                 f"mean {_fmt(d['mean'], digits)})")
        L.append("")
    if result.rho_k_ppg is not None:
        if result.static_fail_fraction is None:
            L.extend(_para(f"static hold at {result.rho_k_ppg:g} ppg: not evaluated"))
        else:
            L.extend(_para(f"static hold at {result.rho_k_ppg:g} ppg fails on "
                           f"{result.static_fail_fraction:.1%} of the kill samples "
                           "(dynamically dead only there)"))
        L.append("")
    if result.vacuum_fraction is not None:
        L.extend(_para(f"relief-well string on vacuum at the kill rate on "
                       f"{result.vacuum_fraction:.1%} of the kill samples; over a pump or rating "
                       f"limit on {result.over_limit_fraction:.1%}"))
        L.append("")
    L.append("## Input ranking (Spearman rank correlation)")
    L.append("")
    for out_key, rows in result.ranking.items():
        L.append(f"against {QUANTITY_WORDS[out_key]}:")
        L.append("")
        L.append("  input        rho")
        L.append("  " + "-" * 24)
        for name, rho in rows:
            L.append(f"  {name:11s}  {rho:+.3f}")
        L.append("")
    L.append("## Convergence (percentiles at n/2 against n)")
    L.append("")
    for q, c in result.convergence.items():
        if c is None:
            continue
        digits = 0 if q == "q_g" else 2
        L.append(f"{QUANTITY_WORDS[q]}, {UNITS[q]}: n/2 = {c['n_half']}, n = {c['n_full']}")
        L.append("")
        L.append("  pct        half          full     rel change")
        L.append("  " + "-" * 44)
        for p in PERCENTILES:
            r = c["percentiles"][f"p{p}"]
            L.append(f"  p{p:<3d}  {_fmt(r['half'], digits):>10s}  {_fmt(r['full'], digits):>12s}"
                     f"  {r['rel_change']:>12.2%}")
        L.append(f"  largest relative change {c['largest_rel_change']:.2%}")
        L.append("")
    if result.failures or result.kill_failures:
        L.append("## Failures")
        L.append("")
        for title, d in (("rate pass", result.failures), ("kill pass", result.kill_failures)):
            if not d:
                continue
            L.append(f"{title}:")
            for reason, n in sorted(d.items(), key=lambda kv: -kv[1]):
                L.append(f"  {n:5d}  {reason[:70]}")
            L.append("")
    if result.warnings:
        L.append("## Warnings")
        L.append("")
        for w in result.warnings:
            L.extend(_bullet(w))
        L.append("")
    if result.notes:
        L.append("## Notes")
        L.append("")
        for n in result.notes:
            L.extend(_bullet(n))
        L.append("")
    return "\n".join(L) + "\n"


def write_summary(result: MonteCarloResult, path: str | Path = "out/mc_summary.md") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(summary_markdown(result), encoding="utf-8")
    return p
