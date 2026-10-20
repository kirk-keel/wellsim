"""mc-1: Monte Carlo on blowout rate and kill rate.

All COMPUTED. There is no published Monte Carlo case to gate against, so every expected
value here is either an identity (a degenerate run must reproduce the deterministic solve
and the deterministic kill rate bitwise), a reproducibility claim (the same seed, and any
worker count, give the same numbers), or a sign/monotonicity claim (a single varied input
ranks first and a constant input reads exactly zero).

The dense-phase failure test uses a deliberately extreme range: a 50 degF reservoir at
9,000 to 19,000 psia is not a real well, it is a range chosen to cross the gas-region
boundary the model refuses (Tpr < 1.05 at Ppr > 1, docs/PROVENANCE.md and core/exit.py),
so the failure accounting has something to account for.
"""

from __future__ import annotations

import pytest

from wellsim.cases.loader import load_case
from wellsim.cases.schema import Case
from wellsim.core import montecarlo as mc
from wellsim.core.kill import kill_analysis
from wellsim.core.pvt import collect_range_warnings, make_pvt
from wellsim.core.solver import build_ipr, solve

RHO_K_PPG = 12.0


def with_mc(case: Case, **mc_block) -> Case:
    d = case.model_dump()
    d["mc"] = mc_block
    return Case(**d)


def tri(x: float) -> dict[str, float]:
    """A zero-width triangular: min = mode = max."""
    return {"min": x, "mode": x, "max": x}


def degenerate_mc(case: Case, **overrides) -> dict:
    """Every distribution pinned to the case's own value, so the draw is the case itself."""
    res = case.reservoir
    block = {
        "seed": 1,
        "n_samples": 2,
        "workers": 1,
        "p_res": tri(res.pressure),
        "net_pay": tri(res.net_pay),
        "cd": tri(case.exit.cd),
        "kh": {"low": res.permeability * res.net_pay, "high": res.permeability * res.net_pay},
    }
    block.update(overrides)
    return block


def deterministic_kill(case: Case, rho_k: float = RHO_K_PPG, q_start=None) -> dict:
    """``kill_analysis`` on ``case`` the way the Monte Carlo kill pass calls it."""
    d = case.model_dump()
    d["dynamic_kill"] = {"rho_k": rho_k, "mu_k": 1.0, "cp_k": 1.0, "z_inj": None,
                         "rate_bpm": None, "find_min_rate": True}
    d["mc"] = None
    c = Case(**d)
    with collect_range_warnings():
        pvt = make_pvt(c.fluid, c.options.correlations)
        ipr, _ = build_ipr(c, pvt)
        return kill_analysis(c, pvt, ipr, q_start=q_start)


# --- a: degenerate ranges reproduce the deterministic solve and kill rate ------------------


def test_degenerate_ranges_reproduce_the_deterministic_solve_and_kill(examples_dir, capsys):
    """COMPUTED identity: with min = mode = max on every input and a zero-width kh range,
    the sampler draws the case itself, so every sample must equal the deterministic solve
    bitwise, and the single kill sample must equal the deterministic kill rate bitwise."""
    base = load_case(examples_dir / "gas_water_wgr.yaml")
    case = with_mc(base, **degenerate_mc(
        base, kill={"rho_k": RHO_K_PPG, "mu_k": 1.0, "cp_k": 1.0, "n_kill": 1, "chains": 1}
    ))

    samples = mc.sample_inputs(case)
    assert len(samples) == 2
    # the draw is the case: every sample identical, and equal to the case's own values
    assert samples[0].values == samples[1].values
    assert samples[0].values["p_res"] == base.reservoir.pressure
    assert samples[0].values["net_pay"] == base.reservoir.net_pay
    assert samples[0].values["kh"] == base.reservoir.permeability * base.reservoir.net_pay
    assert samples[0].resamples == 0

    # the case the sampler builds is the deterministic case (k = kh / h round-trips here)
    applied = mc.apply_sample(case, samples[0])
    assert applied.reservoir.permeability == base.reservoir.permeability
    assert applied.reservoir.pressure == base.reservoir.pressure
    assert applied.reservoir.net_pay == base.reservoir.net_pay

    deterministic = solve(applied)
    det_kill = deterministic_kill(applied)

    result = mc.run(case)
    assert result.solved == 2 and result.survivors == 2
    assert result.failures == {}
    for row in result.samples:
        assert row["ok"], row["reason"]
        assert row["q_g"] == deterministic.gas_rate  # bitwise: nothing stochastic in the path
        assert row["pwf"] == deterministic.pwf
    assert result.value("q_g", 50) == deterministic.gas_rate

    # rho_static is an evaluation, not a search: pin it at 1e-6 relative
    tvd = base.well.md_to_tvd(base.reservoir.depth_md)
    expected_rho = (base.reservoir.pressure - base.exit.ambient_pressure) / (0.052 * tvd)
    assert result.value("rho_static", 50) == pytest.approx(expected_rho, rel=1e-6)

    assert len(result.kill_rows) == 1
    row = result.kill_rows[0]
    assert row["ok"], row["reason"]
    assert row["q_k_min"] == det_kill["q_k_min"]["q_k_min_bpm"]  # bitwise: same q_start (None)
    assert row["static_ok"] == det_kill["static_hold"]["ok"]
    assert row["blount_soeiinah_bpm"] == det_kill["blount_soeiinah"]["q_bpm"]

    print(f"\ndegenerate mc against the deterministic solve ({base.name}):")
    print(f"  q_g          {deterministic.gas_rate:12.4f} Mscf/d  (bitwise equal)")
    print(f"  pwf          {deterministic.pwf:12.4f} psia    (bitwise equal)")
    print(f"  rho_static   {expected_rho:12.4f} ppg")
    print(f"  q_k_min      {row['q_k_min']:12.4f} bbl/min (bitwise equal)")


# --- b: reproducible in the seed and in the worker count -----------------------------------


def _reproducibility_case(examples_dir, workers: int, *, kill: bool) -> Case:
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    res = base.reservoir
    block = dict(
        seed=7, n_samples=8, workers=workers,
        p_res={"min": 0.9 * res.pressure, "mode": res.pressure, "max": 1.1 * res.pressure},
        net_pay={"min": 0.5 * res.net_pay, "mode": res.net_pay, "max": 1.5 * res.net_pay},
        cd={"min": 0.7, "mode": 0.9, "max": 1.0},
        kh={"low": 1e3, "high": 3e4},
    )
    if kill:
        block["kill"] = {"rho_k": RHO_K_PPG, "mu_k": 1.0, "cp_k": 1.0, "n_kill": 3, "chains": 2}
    return with_mc(base, **block)


def _rate_key(r):
    return [(row["index"], row["p_res"], row["net_pay"], row["kh"], row["cd"],
             row["ok"], row["q_g"], row["pwf"], row["rho_static"]) for row in r.samples]


def _kill_key(r):
    return [(row["index"], row["ok"], row["q_k_min"], row["static_ok"],
             row["blount_soeiinah_bpm"]) for row in r.kill_rows]


def test_same_seed_and_any_worker_count_give_identical_rates(examples_dir, capsys):
    """COMPUTED: the sampler is a pure function of the seed and every forward solve is
    independent, so the same seed at one worker and at four must agree on every number,
    not merely within a tolerance."""
    r1 = mc.run(_reproducibility_case(examples_dir, 1, kill=False), kill=False)
    r1b = mc.run(_reproducibility_case(examples_dir, 1, kill=False), kill=False)
    r4 = mc.run(_reproducibility_case(examples_dir, 4, kill=False), kill=False)

    assert _rate_key(r1) == _rate_key(r1b), "same seed, same worker count: not reproducible"
    assert _rate_key(r1) == _rate_key(r4), "the rate pass depends on the worker count"
    assert r1.percentiles["q_g"] == r4.percentiles["q_g"]
    assert r1.percentiles["rho_static"] == r4.percentiles["rho_static"]
    assert r1.ranking["q_g"] == r4.ranking["q_g"]

    print("\nworkers 1 against workers 4, seed 7 (rate pass):")
    print("  index        q_g Mscf/d       pwf psia")
    for row in r1.samples:
        print(f"  {row['index']:5d}  {row['q_g']:16,.4f}  {row['pwf']:12.4f}")


@pytest.mark.slow
def test_kill_pass_is_identical_at_any_worker_count(examples_dir, capsys):
    """COMPUTED: the kill warm start is chained per chain, not per worker, so the kill
    rates must be bitwise identical at one worker and at four. Slow: each kill search is a
    few seconds and this runs two full passes."""
    r1 = mc.run(_reproducibility_case(examples_dir, 1, kill=True))
    r4 = mc.run(_reproducibility_case(examples_dir, 4, kill=True))

    assert _rate_key(r1) == _rate_key(r4)
    assert _kill_key(r1) == _kill_key(r4), "the kill pass depends on the worker count"
    assert r1.percentiles["q_k_min"] == r4.percentiles["q_k_min"]
    assert len(r1.kill_rows) == 3

    print("\nworkers 1 against workers 4, seed 7 (kill pass at 12 ppg):")
    print("  index      q_g Mscf/d   q_k_min bbl/min")
    for row in r1.kill_rows:
        print(f"  {row['index']:5d}  {row['q_g']:14,.0f}  {row['q_k_min']:16.4f}")


# --- c: ranking sanity ----------------------------------------------------------------------


def test_ranking_puts_the_only_varied_input_first_and_reads_constants_as_zero(examples_dir, capsys):
    """COMPUTED: with p_r the only input that moves, its rank correlation with q_g must be
    above 0.95 (the gas IPR rises monotonically with p_r at fixed kh) and every constant
    input must read exactly zero, because a constant has no rank order at all."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    res = base.reservoir
    case = with_mc(base, **degenerate_mc(
        base, seed=11, n_samples=50, workers=6,
        p_res={"min": 0.6 * res.pressure, "mode": res.pressure, "max": 1.4 * res.pressure},
    ))
    result = mc.run(case, kill=False)
    assert result.survivors == 50, result.failures
    ranking = dict(result.ranking["q_g"])

    assert ranking["p_res"] > 0.95, ranking
    for name in ("net_pay", "cd", "kh"):
        assert ranking[name] == 0.0, f"{name} is constant but ranks {ranking[name]}"
    assert result.ranking["q_g"][0][0] == "p_res"  # sorted by magnitude

    print("\nranking with p_r the only varied input (50 samples):")
    for name, rho in result.ranking["q_g"]:
        print(f"  {name:10s} {rho:+.4f}")


# --- d: the rate band is a filter -----------------------------------------------------------


def test_rate_band_filters_forward_samples_and_narrows_the_kh_spread(examples_dir, capsys):
    """COMPUTED: a band around the deterministic rate must keep the samples whose kh is
    near the deterministic kh and drop the rest, so the surviving kh range is strictly
    inside the unfiltered one. The band never enters the inputs: the sampled kh column is
    the same in both runs, and only which samples survive changes."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    deterministic = solve(base)
    q0 = deterministic.gas_rate
    common = dict(seed=3, n_samples=50, workers=6,
                  p_res=tri(base.reservoir.pressure), net_pay=tri(base.reservoir.net_pay),
                  cd=tri(base.exit.cd), kh={"low": 1e2, "high": 1e6})

    free = mc.run(with_mc(base, **common), kill=False)
    banded = mc.run(with_mc(base, **common, rate_band={"q_low": 0.8 * q0, "q_high": 1.25 * q0}),
                    kill=False)

    # the draw is untouched by the band: same kh column, sample for sample
    assert [r["kh"] for r in free.samples] == [r["kh"] for r in banded.samples]
    assert free.filtered_out == 0
    assert banded.filtered_out > 0
    assert 0 < banded.survivors < free.survivors

    free_kh = [r["kh"] for r in free.samples if r["ok"] and r["in_band"]]
    band_kh = [r["kh"] for r in banded.samples if r["ok"] and r["in_band"]]
    assert min(band_kh) > min(free_kh) and max(band_kh) < max(free_kh)
    free_span = max(free_kh) / min(free_kh)
    band_span = max(band_kh) / min(band_kh)
    assert band_span < free_span
    for row in banded.samples:
        if row["ok"] and row["in_band"]:
            assert 0.8 * q0 <= row["q_g"] <= 1.25 * q0

    print(f"\nrate band {0.8 * q0:,.0f} to {1.25 * q0:,.0f} Mscf/d around the deterministic "
          f"{q0:,.0f} Mscf/d:")
    print(f"  no band: {free.survivors:3d} survivors, kh {min(free_kh):12,.1f} to "
          f"{max(free_kh):12,.1f} md-ft (factor {free_span:,.0f})")
    print(f"  banded : {banded.survivors:3d} survivors, kh {min(band_kh):12,.1f} to "
          f"{max(band_kh):12,.1f} md-ft (factor {band_span:,.0f})")


# --- e: failure accounting --------------------------------------------------------------------


def test_failures_are_recorded_with_a_reason_and_percentiles_use_successes_only(
        examples_dir, capsys):
    """COMPUTED: a range that crosses the gas-region boundary must record its refusals with
    the reason, count them, and compute the percentiles from the successes alone. The range
    is deliberately unphysical (a 50 degF reservoir at 9,000 to 19,000 psia); it is chosen
    to cross the boundary, not to describe a well."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    d = base.model_dump()
    d["reservoir"]["temperature"] = 50.0
    cold = Case(**d)
    case = with_mc(
        cold,
        seed=5, n_samples=24, workers=6,
        p_res={"min": 9000.0, "mode": 14000.0, "max": 19000.0},
        kh={"low": 1e5, "high": 1e6},
    )
    result = mc.run(case, kill=False)

    assert result.failures, "the range did not cross the refusal; widen it"
    assert result.solved > 0, "every sample failed; narrow the range"
    assert result.solved + sum(result.failures.values()) == case.mc.n_samples
    assert all(reason for reason in result.failures), "a failure was recorded with no reason"
    assert any("gas region" in reason for reason in result.failures), result.failures
    assert all(row["reason"] is None for row in result.samples if row["ok"])
    assert all(row["q_g"] is None for row in result.samples if not row["ok"])

    # the percentiles come from the successes only
    assert result.percentiles["q_g"]["n"] == result.survivors == result.solved
    ok_rates = sorted(r["q_g"] for r in result.samples if r["ok"])
    assert result.value("q_g", 50) >= ok_rates[0]
    assert result.value("q_g", 50) <= ok_rates[-1]

    print(f"\nfailure accounting over {case.mc.n_samples} samples "
          f"(50 degF, p_r 9,000 to 19,000 psia, kh 1e5 to 1e6 md-ft):")
    print(f"  {result.solved} solved, {sum(result.failures.values())} refused")
    for reason, n in sorted(result.failures.items(), key=lambda kv: -kv[1]):
        print(f"  {n:3d}  {reason[:66]}")


# --- the report --------------------------------------------------------------------------------


def test_summary_tables_fit_in_80_columns(examples_dir, tmp_path):
    """Working convention: every table printed for the owner fits in 80 columns."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    case = with_mc(base, **degenerate_mc(base, seed=2, n_samples=4))
    result = mc.run(case, kill=False)
    path = mc.write_summary(result, tmp_path / "mc_summary.md")
    text = path.read_text(encoding="utf-8")
    wide = [line for line in text.splitlines() if len(line) > 80]
    assert not wide, f"lines over 80 columns: {wide[:3]}"
    assert "exceeded with 90% probability" in text
    assert "exceeded with 10% probability" in text


def test_percentile_labels_say_what_they_mean():
    """The labels are the point of the report: P90 means the opposite thing to a reservoir
    engineer and to a driller, so nothing is reported as a P-number alone."""
    assert mc.PERCENTILE_LABELS[10] == "exceeded with 90% probability"
    assert mc.PERCENTILE_LABELS[90] == "exceeded with 10% probability"
    assert "median" in mc.PERCENTILE_LABELS[50]


# --- sampler unit checks (no solves) ------------------------------------------------------------


def test_triangular_and_loguniform_inverse_cdfs():
    """COMPUTED: the inverse CDFs at their known points. The triangular CDF at the mode is
    (m - a)/(b - a); the log-uniform midpoint is the geometric mean."""
    a, m, b = 1.0, 3.0, 5.0
    assert mc._triangular_icdf(0.0, a, m, b) == pytest.approx(a)
    assert mc._triangular_icdf(1.0, a, m, b) == pytest.approx(b)
    assert mc._triangular_icdf((m - a) / (b - a), a, m, b) == pytest.approx(m)
    assert mc._triangular_icdf(0.4, a, a, a) == a  # degenerate returns the value itself
    assert mc._triangular_icdf(0.0, a, a, a) == a
    assert mc._loguniform_icdf(0.5, 100.0, 10000.0) == pytest.approx(1000.0)  # geometric mean
    assert mc._loguniform_icdf(0.7, 50.0, 50.0) == 50.0


def test_latin_hypercube_stratifies_every_dimension(examples_dir):
    """COMPUTED: one sample per stratum in every dimension is what makes it a Latin
    hypercube, so each column must hit all n strata exactly once."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    n = 64
    case = with_mc(base, seed=42, n_samples=n,
                   p_res={"min": 1000.0, "mode": 5000.0, "max": 9000.0},
                   kh={"low": 10.0, "high": 1e5})
    samples = mc.sample_inputs(case)
    for dim, lo, hi in (("p_res", 1000.0, 9000.0), ("kh", 10.0, 1e5)):
        values = sorted(s.values[dim] for s in samples)
        assert len(values) == n
        assert lo <= values[0] and values[-1] <= hi
    scores = mc._lhs_scores(n, 2, 42)
    for j in range(2):
        strata = sorted(int(u * n) for u in scores[:, j])
        assert strata == list(range(n)), "a stratum was hit twice or missed"


def test_geometry_resamples_are_counted_and_seeded_by_index(examples_dir):
    """COMPUTED: h_1 + h_p <= net pay is enforced on every sample. With h_1 fixed at 20 ft
    and net pay able to fall below h_p, some draws must be redrawn; the redraw is seeded
    from the sample index alone, so it is the same for any worker count."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    d = base.model_dump()
    d["reservoir"]["exposed_interval"] = 40.0
    d["reservoir"]["exposed_top_offset"] = 20.0
    case = with_mc(Case(**d), seed=13, n_samples=60,
                   net_pay={"min": 30.0, "mode": 90.0, "max": 150.0},
                   exposed_interval={"min": 20.0, "mode": 40.0, "max": 100.0})
    samples = mc.sample_inputs(case)
    assert sum(s.resamples for s in samples) > 0, "the ranges never violated the constraint"
    for s in samples:
        assert s.values["h_p"] + 20.0 <= s.values["net_pay"] + 0.01
    # every applied case validates: the schema enforces the same constraint
    for s in samples[:10]:
        applied = mc.apply_sample(case, s)
        assert applied.reservoir.exposed_top_offset == 20.0
    # the redraw is seeded from the sample index, so a second call reproduces it exactly
    again = mc.sample_inputs(case)
    assert [x.values for x in again] == [x.values for x in samples]
    assert [x.resamples for x in again] == [x.resamples for x in samples]


def test_impossible_geometry_is_refused_not_looped(examples_dir):
    """The constraint cannot be met by redrawing when neither geometry input is sampled:
    that is a contradiction in the fixed inputs and must say so, not spin."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    d = base.model_dump()
    d["reservoir"]["exposed_interval"] = 90.0
    d["reservoir"]["exposed_top_offset"] = 8.0
    case = with_mc(Case(**d), seed=1, n_samples=4, p_res=tri(base.reservoir.pressure),
                   net_pay={"min": 30.0, "mode": 40.0, "max": 50.0})
    with pytest.raises(mc.MonteCarloError, match="net pay"):
        mc.sample_inputs(case)


def test_spearman_handles_ties_and_constants():
    """COMPUTED: a constant has no rank order and must read exactly zero, never a nan that
    would sort into the ranking as if it meant something."""
    assert mc.spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
    assert mc.spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    assert mc.spearman([5.0, 5.0, 5.0, 5.0], [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert mc.spearman([1.0, 1.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]) == pytest.approx(1.0)


def test_exit_scenarios_are_weighted_and_exclusive(examples_dir):
    """A discrete exit list replaces the continuous exit range, and the weights are
    respected: a 9:1 split must land near 9:1 over 200 samples."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    case = with_mc(base, seed=4, n_samples=200,
                   p_res=tri(base.reservoir.pressure),
                   exit_scenarios=[{"name": "open", "weight": 9.0, "cd": 1.0},
                                   {"name": "restricted", "weight": 1.0, "cd": 0.6, "exit_id": 3.0}])
    samples = mc.sample_inputs(case)
    names = [s.scenario for s in samples]
    assert names.count("open") == 180 and names.count("restricted") == 20  # LHS is exact here
    for s in samples:
        assert s.values["cd"] in (1.0, 0.6)
        if s.scenario == "restricted":
            assert s.values["exit_id"] == 3.0
    with pytest.raises(ValueError, match="exit_scenarios replaces"):
        with_mc(base, seed=1, n_samples=4, cd=tri(1.0),
                exit_scenarios=[{"name": "a", "cd": 0.6}])


def test_mc_block_must_sample_something(examples_dir):
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    with pytest.raises(ValueError, match="samples nothing"):
        with_mc(base, seed=1, n_samples=10)
    with pytest.raises(ValueError, match="min <= mode <= max"):
        with_mc(base, seed=1, n_samples=10, p_res={"min": 5000.0, "mode": 4000.0, "max": 6000.0})


def test_the_forward_solve_ignores_the_mc_block(examples_dir):
    """The mc block is inert on the forward path, like `observed`: adding it must not move
    a single number."""
    base = load_case(examples_dir / "dry_gas_vertical.yaml")
    plain = solve(base)
    with_block = solve(with_mc(base, **degenerate_mc(base)))
    assert with_block.gas_rate == plain.gas_rate
    assert with_block.pwf == plain.pwf


# --- the example, full size ---------------------------------------------------------------------


@pytest.mark.slow
def test_mc_wgr_example_full_size(examples_dir, capsys):
    """The shipped example at its stated size: 2,000 samples and 100 kill searches. Slow
    (minutes); the numbers it prints are the ones in out/mc_summary.md."""
    case = load_case(examples_dir / "mc_wgr.json")
    result = mc.run(case, progress=lambda m: print("  ..", m))
    assert result.survivors > 1900, result.failures
    assert result.percentiles["q_g"]["p10"] < result.percentiles["q_g"]["p50"] \
        < result.percentiles["q_g"]["p90"]
    assert len(result.kill_rows) == case.mc.kill.n_kill
    assert result.convergence["q_g"]["largest_rel_change"] < 0.10
    ranking = dict(result.ranking["q_g"])
    assert ranking["kh"] > 0.8, ranking  # kh spans a factor 40; it must dominate
    # rw-1 rides along on the kill samples: the example carries a relief_well block
    assert result.percentiles["p_pump"]["n"] == case.mc.kill.n_kill
    assert result.percentiles["hhp"]["n"] == case.mc.kill.n_kill
    assert result.vacuum_fraction is not None and result.over_limit_fraction is not None
    assert all(r["p_pump_psig"] is not None for r in result.kill_rows if r["ok"])
    assert not [line for line in mc.summary_markdown(result).splitlines() if len(line) > 80]
    print(mc.summary_markdown(result))
