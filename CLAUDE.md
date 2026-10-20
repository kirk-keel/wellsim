# wellsim: briefing for a fresh session

## Purpose

Personal steady-state blowout-rate and dynamic-kill simulator (Stage 1 spec in
`docs/wellsim-stage1-spec.md`). Every physics number it reports must be traceable
to a public source or a derivation; the validation library in
`docs/wellsim-validation-library.md` defines the gates.

## Repo layout

```
wellsim/core/          physics, oilfield units, pure functions, no UI
  units.py             unit tables; conversion only at I/O boundaries
  pvt/                 base.py (PVTModel, PVTState, RangeWarning), dry_gas.py, black_oil.py,
                       heat_capacity.py (ideal-gas cp0, k0, delta_h0 from a gravity pseudo-composition)
  ipr/gas.py           pseudo-pressure PSS gas IPR with non-Darcy D (oil.py lands in M4)
  pvt/air.py           G2: air as an ideal gas, k 1.4, Sutherland mu; refuses DAK
  geometry.py          flow-path segments (hydraulic diameter, area, roughness, sin theta, d_o, d_i)
  temperature.py       linear flowing profile
  friction.py          Colebrook-White
  exit.py              real-gas sonic velocity, Mach-1 exit check, sonic_pressure()
  vlp/march.py         marching integrator, regular through Mach 1, internal-choke reset, model hook
  vlp/area_change.py   sudden expansion (Borda-Carnot) and contraction (Crane loss + isentropic) steps
  vlp/correlations/    single_phase_gas.py
  vlp/holdup.py        drift-flux holdup closure (M3-1); vlp/two_phase.py the two-phase march (M3-2)
  nodal.py             pwf-space root of pwf_vlp(q_ipr(pwf)) - pwf; every root returned and tagged
  solver.py            solve(case) -> Result, pure; solver_two_phase.py the gas-liquid nodal (M3-3)
  kill.py              dynamic kill with a Newtonian kill fluid (M3-4); z_inj_state for rw-1
  relief_well.py       rw-1 pump pressure, HHP, free fall, schedule; rw-2 Ramey arrival T
  inverse.py           inverse mode: kh from an observed rate, p_r from SIWHP (observed block)
  montecarlo.py        mc-1: forward Monte Carlo on rate, static-hold mud weight, kill rate (mc block)
  numerics.py          own brent()/bracket (no scipy in the solver path)
wellsim/cases/         schema.py (pydantic, canonical units), loader.py, examples/*.yaml and mc_wgr.json
wellsim/validation/    g2_lsu.py: the LSU PERTT gate, point values against Table A.1
wellsim/tests/         pytest; every expected value carries a provenance tag
docs/                  spec, validation library, PROVENANCE.md, TODO-M4.md, validation/G2-lsu-boem.md
```

## Running things

```
.venv\Scripts\python -m pytest            # pyproject adds -q and -m "not perf and not slow"; do not add another -q
.venv\Scripts\python -m pytest -m perf    # the wall-time budget test (1 s per solve); only meaningful on a quiet machine
.venv\Scripts\python -m pytest -m slow    # rho_k kill sweep, mc-1 worker invariance, mc_wgr full
                                          #   size, rw-2 arrival temperature and rw-1 in mc-1
.venv\Scripts\python -c "from wellsim.cases.loader import load_case; from wellsim.core.solver import solve; r = solve(load_case('wellsim/cases/examples/dry_gas_vertical.yaml')); print(r.gas_rate, r.pwf, r.exit.p, r.exit.choked, r.choke_md, r.warnings)"
```

Examples: `dry_gas_vertical` (choked exit), `dry_gas_big_bore_liner`, `dry_gas_parted_tubing`
(internal choke), `gas_water_wgr` and `gas_water_pi` (gas-liquid nodal), `black_oil_annulus_kill`
(raises until M4), `mc_wgr.json` (Monte Carlo on the WGR case, with a relief well),
`rw_wgr.json` (relief-well hydraulics). Diagnostic scripts run with
`PYTHONPATH=.` from the scratchpad, never the repo; anything calling `montecarlo.run` with
workers > 1 needs an `if __name__ == "__main__"` guard, because the pool spawns on Windows.

## Provenance rules

`docs/PROVENANCE.md` lists every value a test compares against, with one tag:

- FETCHED: read from a named page or PDF in-session (secondary unless the paper is named)
- COMPUTED: derived in the test, or an internal-consistency / trend check
- MEMORY: recalled from a publication, not verified against it
- NEEDS HAND CHECK: MEMORY item that a gate depends on; listed at the end of the doc
- VERIFIED: independently checked by the owner against a named source or hand derivation

Every test value carries its tag in a comment, and the doc gets a row when a test is
added. Never cite a source as consulted when it was not. When a value cannot be confirmed,
keep the test on the stated assumption and record the gap; never tune inputs to pass.

## Working conventions

- Diagnostics are printed and reported, not committed. Commit only what the owner asks
  for, with the message they give.
- Tolerances are never loosened without the owner's explicit approval; an outlier is
  documented and tested as an outlier (see D40 in PROVENANCE.md).
- Assumptions become warnings on the result, never silent defaults (defaulted exit
  temperature, partial penetration, choked exit or restriction, extrapolated
  correlations via RangeWarning).
- The solver raises rather than clamps: Mach > 1 outside the choke logic raises
  SupersonicError; correlations warn but never clamp inputs.
- Report numbers faithfully; state what was and was not verified.
- The kill rate is found from the continuous minimum of the outflow-minus-inflow
  residual, never from a scan; a scan under-reports the kill rate.
- Pin evaluations at 1e-6 relative; pin searched or optimized results at the search
  tolerance. Never pin a search result tighter than its own stopping rule.
- Every search returns a proven bracket: the answer satisfies the criterion and one
  tolerance step the other way does not. Tolerances are relative to the root, never to
  the bracket.
- Relative differences divide by the smaller value; a gap reads as how much higher the
  upper value is.
- Every table printed for the owner fits in 80 columns. Split a wide table into two
  narrower ones, or print one case per block; never let a row wrap.
- Every end-of-step summary is also written to `out/last_summary.md` (`out/` is
  gitignored), so the last report survives the terminal.
- Commit messages end with a `Co-Authored-By:` trailer that NAMES THE MODEL ACTUALLY
  USED FOR THE COMMIT, e.g. `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
  or `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. The trailer is
  a record of who did the work, not a fixed string: do not copy the previous commit's.

## Environment traps (Windows 11, project .venv, Python 3.13)

- Git Bash heredocs containing apostrophes or triple quotes fail to parse. Write patch
  scripts and long files with the Write tool, then run them; keep heredocs quote-free.
- No poppler: read PDFs with `pypdf`; `PYTHONIOENCODING=utf-8` when printing the text.
- Importing `scipy.optimize` was once blocked by an application-control policy; the
  solver path uses its own Brent for that reason. Keep scipy out of hot loops.
- Many textbook aggregator sites (GlobalSpec, ScienceDirect topics, ResearchGate,
  scribd, petrowiki, PSU e-education) refuse WebFetch; university eprint PDFs work.
- Git warns about LF to CRLF on every commit; harmless.

## Git

Remote `origin` is `greybeard00007@100.94.72.122:git/well_control_suite.git` (homelab,
SSH, works without prompts). Branch `main` only. Push after each commit the owner asks
to push; confirm `git rev-parse main` equals `git rev-parse origin/main`.

## Current state

HEAD: `b53e56e` "docs: M4 black-oil design record", on top of `9204250` (hk-2),
`ecf0d53` (hk-1) and `7d3ab7a`/`27e2cec` (G2 on the march path). Tests: **783 in the default
run, 11 deselected** -- 1 behind `-m perf` (the wall-time budget) and 10 behind
`-m slow` (33.5 min: the rho_k kill
sweep, the mc-1 worker-invariance kill pass, the mc_wgr example at full size, the rw-2
arrival temperature, rw-1 through mc-1, the ks-2 start-independence check, the rw-3b
k_annulus plausible range, the G2 full run and its pressure sensitivity). Perf pins in
`tests/test_perf_pins.py`. Default-run wall time 252 s (26 Sep 2026, ONE run,
so not a median and not to be quoted as one): same code as the 230 s run at
`9204250` -- b53e56e differs from it only by `docs/M4-design.md` -- so this is
single-run box variance, not a regression. All 59,670 warnings in the default
run are `RangeWarning`;
no other warning class appears. By count of summary lines the largest are DAK Ppr,
Lee-Gonzalez-Eakin p and T, the ideal-gas cp range, and DAK Tpr, with single hits from
McCain cw and Baker-Swerdloff oil. M4-1 moves these onto a once-per-solve collector.
`docs/M4-design.md` is the authoritative M4 plan.
Gate G1 (gas VLP) holds against the Cullender-Smith example and the nine-well table.

**Timing note (22 Sep 2026), read before chasing a performance regression:** this box ran
about 4x slow for the rw-1 and rw-2 sessions and recovered during ks-2. Anything timed in
rw-1 or rw-2 is on the slow box and is not comparable with anything before or after; the
tell was that `git stash` on HEAD reproduced the slow figure exactly, so it was the machine.
Quiet-box figures, measured back to back during ks-2: the 12 ppg WGR kill search 4.23 s
(71 outflow calls, 60 ms each) before ks-2 and 7.09 s (148 calls, 48 ms each) after; the
default suite 210 s for 765 tests.
**Quote runtime as the median of three quiet runs.** A single run is not a
measurement: report it as one, and never read a regression out of it.

rw-1 (22 Sep 2026, `core/relief_well.py`, `relief_well` case block): one relief well.
p_pump = p_zinj - hydrostatic(string) + friction(string) + friction(surface lines) +
restriction loss, Newtonian kill fluid through the existing Colebrook routine. p_zinj is
the blowing well's own pressure at z_inj at the kill condition from `kill.z_inj_state`:
at q_k_min there are no roots, so the state taken is the one at the RESIDUAL MINIMISER
(one warm-started minimum plus one outflow call, about 10 to 15% on top of the search,
off unless a relief_well block is present). String sections are (md, tvd) stations, the
survey's own pair, so a deviated string gets hydrostatic from the drop and friction from
the measured length; surface lines are horizontal. Restriction is exactly one of bit TFA
(dp = rho q^2 / (12031 TFA^2), MEMORY) or an intersection K (K rho v^2 / (2 g_c) at the
string ID); no default. Hydrostatic uses the exact rho TVD / 144, the Blount-Soeiinah
basis, not 0.052 (0.10% apart; the hand check prints both). p_pump rises monotonically
with q, so one rate puts the balance at atmospheric: below it the string is on vacuum and
a negative pump pressure is never reported as a number, only as a diagnostic beside the
free-fall rate. rw_wgr (on corrected rates, ks-2): q_k_min 26.7735 bbl/min, p_zinj
4,875.9 psia at 90.8 degF, hydrostatic 6,233.8, string friction 1,842.9, surface 120.3,
bit 1,261.2, p_pump 1,851.8 psig, HHP 1,215, free-fall rate 17.378 bbl/min.
rw-2 (22 Sep 2026, same module, `relief_well` block): the kill-fluid arrival temperature
at z_inj, replacing the rw-1 two-bound report. Ramey (1962) downward, dT/dL = (T_g - T)/A
with A = w c_p f(t)/(2 pi k_e), integrated along the string in MD (heat transfer is per
unit of pipe) with T_g from TVD so a deviated string works, closed form per segment. A
BOUND, not an estimate: the U -> infinity limit drops the pipe, annulus and cement
resistance, so it is the most heating available and the highest arrival temperature
reachable; the surface temperature is the lower bound. Both kill rates reported with the
gap; the reported one is the hotter, higher-rate end. Schema adds hole_radius and
pumping_time_min (required) and k_e / alpha (defaults 1.4 and 0.04, warned).
rw_wgr at 30 min: t_D 0.080, f 0.2920, A about 26,000 ft against an 11,500 ft string,
arrival 82.83 degF (19% of the available heating). Hasan-Kabir is DEMOTED back to deferred:
the gap it was promoted for was the width of a bracket, not a heat-transfer result.
NOTE: the gaps rw-2 first reported (0.99% arrival, 5.27% geothermal) were measured with
the pre-ks-2 search and are SUPERSEDED. On corrected rates: surface 60 degF 26.1604
bbl/min, arrival 82.83 degF 26.7735, geothermal 180 degF 27.9771, so 2.34% and 6.94%
under the convention above (divide by the smaller). `gap_relative` on the result object
reports the same 2.34%.

rw-3 (22 Sep 2026, `core/relief_well.py`, `relief_well` block): finite overall heat
transfer coefficient, turning the rw-2 bound into an estimate. Relief string only; the
blowing well stays adiabatic. Three resistances in series per unit length, referred to the
string OD, 1/U = r_to/(r_ti h_i) + r_to ln(r_to/r_ti)/k_pipe + r_to ln(r_w/r_to)/k_cem,
with h_i from Dittus-Boelter (Nu = 0.023 Re^0.8 Pr^0.4, the heating form) and the formation
f(t) term where rw-2 has it, so A = w c_p (k_e + r_to U f(t))/(2 pi r_to U k_e), which is
exactly rw-2 as U -> infinity and the inlet temperature as U -> 0. U is per section, so an
uncemented interval gets k_e over the same annulus and the result says how much of the
string that was. Schema: string_od / per-section od, k_pipe (25), k_cem (0.7), a per-section
`cemented` flag, dynamic_kill.k_f (0.4); the three conductivities warn on default.
THREE temperatures are reported and the MIDDLE one is the answer: rw_wgr gives surface
60.00 degF -> 26.1604 bbl/min, finite-U 64.03 degF -> 26.3647 (REPORTED), bound 82.83 degF
-> 26.7767; 0.78% below and 1.56% above, 2.36% end to end.
The result that matters: the annulus is 99.35% of the resistance (R_film 0.000371,
R_pipe 0.001290, R_annulus 0.255409), so U is 3.89 and A is 167,497 ft against a bound of
26,260 ft. The fluid gets 4.03 degF of the 120 available, 3.4%, against the bound's 22.83.
The true arrival temperature is near the COLD end of the rw-2 bracket, not the middle.

rw-3b (22 Sep 2026): three closing items.
- `k_cem` is now `k_annulus`, with `annulus_fill` naming the default: 'mud' 0.7 (the
  default, because on a dynamic kill the relief annulus holds STATIC drilling fluid, the
  kill fluid going down the string and out at the intersection with nothing returning) or
  'cement' 1.0; both warn when defaulted. `k_cem` still loads as a deprecated alias with a
  warning, and setting both is refused.
- k_annulus came OFF the hand-check list. Over its plausible range 0.4 to 1.2 the arrival
  temperature moves 3.62 degF (62.49 to 66.11) but the converged kill rate moves 0.2768%,
  inside the 1% search tolerance. At the shipped tolerance all three read 26.3648 bbl/min,
  which is the search quantising to its bracket, not true insensitivity; the verdict rests
  on the 1e-4 spread. rw-3 had swept 0.1 to 100, which no annulus occupies.
- The Monte Carlo kill pass now RUNS at the finite-U arrival temperature, two searches per
  sample (surface, then arrival) with `bound_run` off, instead of searching at the M3-4
  stream temperature and reporting the arrival temperature beside it.

ks-2 (22 Sep 2026, `core/kill.py`): the kill search bracketed to its own tolerance.
Diagnosis first, read-only, in `out/ks-diagnosis.md`. Two faults, both gone:
- the confirmation scan certified a still-flowing rate as killed. On rw_wgr it accepted
  25.8774 bbl/min where the continuous residual minimum is -37.52 psi, because the
  still-flowing window is 90.4 psi wide against a 332.4 psi scan step, 0.27 of one step.
  The scan gate and the 1% bump are DELETED; the continuous minimum is the only verdict,
  which was already the CLAUDE.md rule.
- Brent stopped at `xtol = rel_tol * bracket_top`, so the answer moved with the warm
  start: 25.6230 from [20, 40], 26.6198 from [15, 30], true root 26.1555. The tolerance is
  now set relative to the root by iteration, then finished by bisection until
  `lo >= hi (1 - rel_tol)`, and `hi` is returned: the smallest rate PROVEN to kill.
An always-on proving check (`_prove_kill_rate`, not a test) raises KillSearchError unless
the residual minimum is >= 0 at q_k and < 0 at q_k (1 - rel_tol). Four starts now agree to
0.44% inside the 1% tolerance, against 2.90% before. The 1e-6 evaluation pin g(27.0) is
UNCHANGED (1.1e-12), which is what shows ks-2 moved the search and not the physics. Every
re-pin is in docs/PROVENANCE.md; the old 12 ppg pins sat about -54 psi, still flowing.
Cost: 71 to 148 outflow calls, 4.3 to 7.1 s on this (slow) box.

Two defects rw-2 recorded rather than hid (both in docs/TODO-M4.md):
- the published Hasan-Kabir f(t_D) is NOT continuous at t_D = 1.5: 0.873990 against
  0.852646, a 2.50% jump (divide by the smaller, per the convention above; it was
  written as 2.44% until hk-1, which divided by the larger). The step asked for 1e-3.
  hk-1 confirmed all five constants against the book, so no constant is wrong: the
  discontinuity is the source's own and the book says so. Both branches are left as
  published and the jump is pinned by a test; it is on the hand-check list. It does not
  make the search jagged, because t_D does not depend on rate.
- `min_kill_rate` was NOT start-independent to its 1% tolerance. FIXED by ks-2, below.

mc-1 (22 Sep 2026, `core/montecarlo.py`, `mc` case block): `run(case)` returns a
MonteCarloResult and `write_summary` puts the report in `out/mc_summary.md`. Latin
hypercube on p_r, net pay, h_p and the exit condition (triangular) with kh log-uniform;
k = kh/net_pay derived as in `inverse.with_kh`; h_1 held and the h_1 + h_p <= net pay
constraint enforced by an index-seeded redraw. Rate pass one `solve` per sample in
processes (never threads: `collect_range_warnings` swaps a module global and the
process-wide warnings filter); refusals and non-flowing samples recorded with a reason,
percentiles from survivors only. `rate_band` is a filter on forward samples, never an
input. Kill pass stratifies survivors into equal-probability q_g bins and runs the search
in a fixed number of warm-start chains, so the answer is the same at any worker count.
Proof: the degenerate run reproduces the deterministic solve and kill rate bitwise, and
workers 1 and 4 agree bitwise. mc_wgr at 2,000 samples: q_g 76,655 / 200,340 / 285,051
Mscf/d, rho_static 9.06 / 9.59 / 10.12 ppg, q_k_min 23.2 / 27.2 / 29.8 bbl/min at the
10th / 50th / 90th; kh dominates the rate (+0.989), kh and p_r share the kill rate
(+0.72, +0.66); 499 s on 8 cores.

Inverse mode (22 Sep 2026, `core/inverse.py`, `observed` case block): `invert(case)` returns
an InverseResult (value, observation, fixed inputs, statement, a case copy with the value
set). kh from an observed rate is Brent in ln kh on the forward solve over 1e-2 to 1e6 md-ft
(top stepped down a decade at a time when not evaluable), 0.5% on q_g and on kh, with
dq_g/d(ln kh) and the kh range for q_g +/- 20% (exact when bounds are given, linearised
always). p_r from SIWHP is a static real-gas column (RK4 in TVD, geothermal linear in TVD);
liquid cases are refused. Proof: the two kh round trips (0.03% and 0.06%), the SIWHP round
trip (machine precision), the 12 ppg kill rate from the inverted case (+0.002%).

M3 (multiphase), completed 21 to 22 Sep 2026, each with its proving test:
- M3-1 drift-flux holdup closure (`core/vlp/holdup.py`): Zuber-Findlay with Hasan-Kabir
  coefficients, slug drift ramped Harmathy to Taylor-bubble over alpha_g 0.25 to 0.52,
  logistic blends elsewhere, churn needs v_m > v_ms; closed-form solve, analytic dalpha/drho_g.
  Proof: the bubbly and slug hand points and the 24 monotone sweeps (`test_holdup_crosscheck.py`).
- M3-2 two-phase adiabatic march (`core/vlp/two_phase.py`, a `model` hook on `march()`):
  mixture density and friction, slip momentum flux, mass-weighted energy with the liquid
  pressure term, mixture area changes, choke at the 2x2 singular point. Proof: the 200 gpm
  liquid limit (Re 211,011, f 0.0170, 381 psi) and the 1e-6 gas limit (`test_two_phase.py`).
- M3-3 liquid inflow and gas-liquid nodal (`core/solver_two_phase.py`, `liquid_inflow`
  wgr/pi): every root returned and tagged by the derivative test, unstable roots warned.
  Proof: the two-root WGR case at p_r 4,000 psia (`test_nodal_two_phase.py`).
- M3-4 dynamic kill (`core/kill.py`, `dynamic_kill` block): q_k_min from the continuous
  minimum of the residual, Blount-Soeiinah rate and static hold (absolute basis,
  p_exit + 0.052 rho_k TVD) always reported, two-leg march with the junction closed
  algebraically at z_inj. Proof: the big-bore Blount-Soeiinah hand check (72.15 bbl/min)
  and the rho_k sweep, monotone and above Blount-Soeiinah (`test_kill.py`, `-m slow`).

Known behaviours:
- The two-phase choke (singular point of the 2x2 momentum-energy system) sits below the
  Wallis frozen sound speed because the liquid heat capacity makes the gas expansion
  near-isothermal; the ratio tends to 1/sqrt(k). Wallis c_m and Mach are diagnostics.
- Beggs-Brill overpredicts liquid holdup in annular flow (about 5x at 100 ft/s); it is a
  cross-check in the tests only, never in the model path.
- **G2 now scores the march's OWN gradient (g2-4), so these figures grade the shipped
  model path and not a second implementation.** `g2_lsu.evaluate` builds a one-segment
  FlowSegment and calls `two_phase_gradient`; the gate's homogeneous 1/(1-E_k) form is
  deleted. On that basis the closure is **validated on gradient in bubble, bubbly and
  slug/churn (mean absolute error 3 to 5%) and NOT validated in churn (18%) or annular
  (61%)**, with the pass criterion, low-slip rows, at 3.84% against LSU's own 10% bar
  (BOEM M15PC00007 p.59).
  That matters more than it sounds: a blowout is annular essentially everywhere, and the
  shipped gas_water_wgr example is annular at all 125 march points, alpha_g 0.9751 at the
  sandface and 0.9986 at the choked exit. The gate's worst regime is the tool's main use
  case. The annular figures also BOUND rather than estimate the downhole error: LSU ran
  air and water at 832:1 density ratio against 5.6:1 at the WGR sandface, and slip is
  hardest at the extreme LSU occupies. Do not quote the overall 14.42% as if it applied
  to a blowout calculation. g2-4 widened this gap rather than closing it: pointing the
  gate at the march gradient took the overall figure from 20.41% to 14.42% and left the
  annular figure at 61.08%, so the annular error is the closure and nothing else.
- At zero gas and zero liquid rate the two-phase outflow is the static liquid column,
  the q -> 0+ limit of the loaded column, so a loaded well shows its rising crossing.
- The kill fluid enters at the stream temperature at z_inj unless rw-2 imposes an
  arrival temperature; static hold and the Blount-Soeiinah column both include the exit
  pressure and agree in sign.
- The kill search converges FROM ABOVE: it returns the upper end of a bracket narrower
  than rel_tol, the smallest rate proven to kill, so its 1% is one-sided in practice. On
  rw_wgr four warm starts land 26.1604 to 26.2761 bbl/min against a true root of 26.1555,
  all above it. Read a reported q_k_min as at or just above the answer, never below.
- DAK departures run a few percent small (marched temperatures 2 to 4 K warm per throttle);
  the sandface throttle is steady-state; no wellbore heat transfer; the gas-region rule
  (Tpr < 1.05 at Ppr > 1, or Tpr < 0.7 or Z < 0.3 at Ppr <= 1) is refused, never interpolated.

hk-1 (25 Sep 2026, `core/vlp/holdup.py`): the annulus Taylor-bubble coefficient pair,
after reading Hasan & Kabir, *Fluid Flow and Heat Transfer in Wellbores*, 2nd ed. directly
(located in `out/hk-locations.md`). `TB_ANNULUS_A, TB_ANNULUS_B` 0.30, 0.22 -> **0.35,
0.10**, per Eq. 4.75. The old pair came from an arXiv secondary and did not collapse to
the pipe coefficient at d_i = 0, giving 0.30 against `TB_PIPE_COEF = 0.35`, a 14% step at
the origin that a test had wrongly pinned as "a property of the published forms". The
book's pair collapses exactly, and sec. 4.4.1 says it must.
**No shipped number moved, and that is expected, not luck:** every runnable example is
pipe (`d_i = 0`) - gas_water_wgr, dry_gas_vertical, dry_gas_big_bore_liner, all of G2 -
and `black_oil_annulus_kill` is the only annular case and raises until M4. The exposure is
future annular work and the unit tests.
Also FETCHED at hk-1, off MEMORY: churn C0 = 1.15, annular no-slip C0 = 1/v_d = 0, the
0.25 bubbly-slug rule and its 0.429/0.357 form, the 0.52 dispersed cap, the 1.08
slug-churn rule, the 3.1 annular transition, Harmathy 1.53, pipe Taylor 0.35, the deviated
sqrt(sin)(1+cos)^1.2, and the f(t_D) branch constants. Still open: the explicit Taitel
v_ms constants (the book has only an implicit form) and the slug rise-velocity blend (the
book's Eq. 3.28 disagrees with its own two worked examples).
Recorded, NOT acted on (all in docs/TODO-M4.md): the book prescribes HOMOGENEOUS friction
and kinetic terms everywhere (Eq. 2.19), so there is no general non-homogeneous fix to
make and that TODO item was rewritten; annular flow is the one exception and its film
model is sec. 3.3.4 Eqs. 3.50-3.62, now logged as a hypothesis to test against G2 and the
ONLY live Hasan-Kabir lead; the book rejects sin(theta) on the bubbly v_d; and its
continuous f(t_D) Eq. 6.10 is a different correlation, not a smoothing.

hk-2 (25 Sep 2026, docs only, `out/hk-c0-diameter.md`): the large-diameter C0 caveat that
hk-1 logged as "the next G2 hypothesis" was read in full and **REJECTED, not left
pending**. Sec. 3.3.1's elevated values (2.0, 1.95, and an uncited 1.6) attach to STAGNANT
or near-stagnant columns - the quantified threshold is v_sL < 3.0 mm/s - and to BUBBLY flow
only. LSU's slowest row is 23 mm/s, 7.7x that threshold; the three elevated sources are an
annular-geometry MS thesis, bubble-column reactors and pumping-well testing; and the book's
own Chapter 5 model table (Table 5.1) carries no diameter term at all, C0 = 1.2 upward
cocurrent at every diameter. It is also the wrong regime: G2's per-diameter spread is churn
and annular, and LSU has 7 bubbly/bubble rows out of 131. **Kept as a LIMITATION: fixed
C0 = 1.2 is unproven in large-diameter stagnant or near-stagnant liquid** - which bites a
loaded column or a static large annulus, not any shipped example. Logged separately and
unscheduled: the Hughmark C0 = 3.04 - 0.151 ln(Re) of sec. 4.3, which is Reynolds-based
rather than diameter-based, runs the OPPOSITE way (larger pipe, higher Re, lower C0), is
for horizontal intermittent flow, and is outside its validity window on 105 of 131 G2 rows.

Open items (`docs/TODO-M4.md`, hand-check list in `docs/PROVENANCE.md`):
- Inverse mode (kh from an observed rate, p_r from a shut-in wellhead pressure) was
  pulled ahead of M4 because M4 is gated on hand checks; inverse mode is a prerequisite
  for Monte Carlo. Done 22 Sep 2026.
- rw-1: relief-well hydraulics for the dynamic kill. Done 22 Sep 2026. Two relief wells
  are deferred (docs/TODO-M4.md).
- rw-2: kill-fluid arrival temperature on the relief string. Done 22 Sep 2026.
- ks-2: the kill search bracketed to its own tolerance. Done 22 Sep 2026.
- rw-3: finite U on the relief string, turning the rw-2 bound into an estimate. Done
  22 Sep 2026. The annulus is 99% of the resistance, so the estimate lands near the cold
  end of the rw-2 bracket.
- rw-3b: annulus fill named, finite-U arrival wired into the Monte Carlo kill pass. Done
  22 Sep 2026.
- G2 (23 Sep 2026): the LSU PERTT large-diameter air-water gate is IN, as point values
  rather than a march. 131 runs from the LSU Final Report Table A.1 (PDF pp. 155-158) in
  `docs/validation/data/lsu_table_a1.csv`, gate in `wellsim/validation/g2_lsu.py`,
  results in `docs/validation/G2-results.md`. PASS: low-slip gradient MAE 2.97% against a
  10% bar taken from LSU's own finding (BOEM M15PC00007 p.59). Only that is asserted;
  everything else is reported until the owner sets standing bars.
  A consistency screen runs before scoring (docs/validation/G2-consistency.md): Table
  A.1's gradient and holdup columns contradict each other on 35 of the 93 rows with a
  holdup, by more than each run's own uncertainty. They agree below about 5 ft/s of
  mixture velocity and diverge above it, with the sign FLIPPING between the 3.83 in
  section (+0.132 over 15 ft/s) and the two larger ones (-0.063, -0.078). Row 1's
  reported holdup of 0.99 needs 23x the drift-flux in-situ gas velocity and is simply
  wrong. Gradient is therefore scored on all 131 rows, holdup on the 58 that pass; per
  diameter the screened holdup error improves everywhere (46.95 -> 42.77, 12.59 -> 8.25,
  16.81 -> 13.80%), while the OVERALL figure rises 28.59 -> 28.73% only because the mix
  changed. Read per diameter, not overall. (All re-measured at g2-4: the screen strips
  the march's acceleration term now, so it fails 35 rows instead of 39. The verdict and
  the sign flip are unchanged.)
  G2-2 (23 Sep 2026) tried the obvious annular fix and REVERTED it. C0_ANNULAR = 1.0
  with v_d = 0 is no-slip by construction (alpha = v_sg/v_m asserts the wall film moves
  at gas velocity) and the annular holdup is badly wrong because of it. Raising C0 to
  1.12 fixed the holdup almost exactly (run 8: 0.113 against 0.11 reported, was 0.010)
  and destroyed the gradient: annular MAE 61% -> 20,778%, overall 20.4% -> 340.8%,
  because rho_m rose tenfold, E_k reached 0.9911 and the homogeneous 1/(1-E_k) factor
  hit 112 at 14.7 psia. It also breaks the single-phase gas limit: alpha tends to 1/C0,
  not 1, so the M3-2 gas-limit test fails by 41% on pwf.
  **Those two gradient figures are SUPERSEDED by g2-4 and must not be quoted.** They were
  measured inside the homogeneous factor, which is now deleted. Re-measured through the
  march gradient, C0 = 1.12 costs overall 14.42% -> 17.61% and annular 61.08% -> 231.23%:
  a real cost, not a blow-up. The gas-limit break is unaffected and is now the whole case
  against a constant C0 > 1. See docs/TODO-M4.md, which was rewritten for this.
  Also recorded: the 0.20 m block's 25.04% gradient MAE is concentrated, not spread - 9
  of 28 rows carry 69.9% of the error, all at v_m above 19.8 ft/s, and gravity alone at
  the reported holdup is 94% of the whole measured gradient on them. The over-prediction
  is in the mixture FRICTION, not the closure and not acceleration: g2-4 showed the
  march's acceleration term is 0.3% to 2% of the gradient on exactly those rows, and on 7
  of the 9 the measured total is at or below hydrostatic at the row's own reported
  holdup, leaving no room for a friction term at all. The closure predicts annular on 0
  of those 28 rows.
  Three things to carry: (1) the closure degrades into churn and annular, gradient MAE
  18.17 / 61.08 / 54.35% against 4 to 5% in bubbly and slug, and it predicts holdup 0.01
  against a reported 0.11 on the highest gas-rate row; (2) Table A.1's printed header has
  u_sg and u_sl the wrong way round relative to its own data, confirmed three ways, so
  anyone reading the source table naively swaps them; (3) no tap spacing and no per-run
  pressure exist in any of the three reports, so a marched G2 is not possible from them. k_annulus came off the hand-check list: over its plausible range it moves
  the converged kill rate 0.28%, inside the search tolerance. The relief-well thread
  closes here.
  g2-4 (23 Sep 2026), two commits. (1) `pvt: thermodynamic interface for the march
  gradient`: `molar_mass`, `state(p, t, warm=False)` and `thermo(p, t, state=None)` on
  PVTModel, so the gradient no longer reaches into DAK-specific PVTState fields and any
  PVT that can answer those three can be marched. `AirIdealGas` implements all three in
  closed form, with h_dep and mu_jt identically zero BY DEFINITION and no reduced
  coordinates at all (tpr, ppr and both DAK partials are None, not invented). Refactor
  only: all 22 pins bit-identical, timings within 2%. `gas_sg` was deliberately NOT
  renamed, because it is a genuine specific gravity that Sutton, Lee-Gonzalez-Eakin,
  Standing, Vasquez-Beggs and the non-Darcy D consume; only the `M_AIR * gas_sg`
  molar-mass derivation in the gradient changed. (2) `G2: gate now runs the march
  gradient`: `g2_lsu.evaluate` builds a one-segment FlowSegment and calls
  `two_phase_gradient`; the gate's own gradient, including the homogeneous 1/(1-E_k)
  factor, is deleted. Gradient MAE 20.41% -> 14.42% overall, 43.89% -> 25.04% on the
  0.20 m block, churn 28.99% -> 18.17%, churn/annular 88.82% -> 54.35%, and **annular
  61.07% -> 61.08%, unmoved**. The low-slip pass criterion went 2.97% -> 3.84%, still
  inside its 10% bar. The consistency screen keeps 58 rows instead of 54 because the
  acceleration it strips from the reported gradient is now the march's.
  NOTE for anyone reading a point value out of `two_phase_gradient`: `dp_ds` is the
  NUMERATOR of the 2x2 solve, as in `single_phase_gas`. The physical gradient is
  `dp_ds/det`, now exposed as the `dp_dx` property. The march integrates (det, dp_ds,
  dt_ds) in arclength so it stays regular through the singular point.
  Recorded from mc-1 before rw-1 started: across kh 500 to 20,000 md-ft the blowout rate
  is reservoir-limited (kh ranks +0.989 against q_g, p_r +0.118, cd -0.040, net pay
  +0.006), so the reported rate spread is very nearly a restatement of the kh prior, not
  an emergent result. Per job kh needs an offset-well or log-based range, or a rate band
  to pin it; a generic factor-40 kh range gives a factor-4 rate range that says more
  about the analyst than about the well.
  Test budget: the default suite is at 170 s, so rw-1's tests stay fast; only the
  injection-temperature bounds (two kill searches) go under `-m slow`.
- mc-1: Monte Carlo sampling p_r, net pay, open interval and exit condition (triangular),
  kh log-uniform; an observed rate band is a filter on forward samples, never an input to
  invert. Outputs: blowout rate, static-hold mud weight and kill rate at the 10/50/90th
  percentiles, labeled in words, with an input ranking. Done 22 Sep 2026.
  The design changed from "Monte Carlo on kh" because a rate estimate from a flare or a
  plume is not a defensible input on most jobs, while pressure, pay and exit condition are.
  Inverting an observed rate to kh and then sampling around it would launder that
  indefensible number into the answer; filtering forward samples by a rate band keeps it
  where it belongs, as a check on the ensemble.
  Open on mc-1: correlated inputs are not modelled (p_r, net pay and kh are drawn
  independently); the rate pass dominates the wall time (391 s of 499 s at 2,000 samples),
  so the remaining lever is the forward solve itself, not the kill search.
- G1/G2 gate tests for the multiphase march, pending the LSU/BOEM data
  (`docs/validation/G2-lsu-boem.md` lists the fields needed).
- Black-oil hand checks gate M4. **Vasquez-Beggs is the primary oil correlation and
  Standing is the toggle** (owner decision, `docs/M4-design.md`, which is the
  authoritative M4 plan); VB needs separator-corrected gas gravity, so separator
  conditions are an input and a defaulted one is flagged in the summary.
- Performance pass done (22 Sep 2026): dry gas 0.30 s, WGR nodal 0.55 s, kill search
  4.3 s (warm-started residual minimum, 71 outflow calls), 19-row sweep 60 s. Remaining
  lever: carry the minimiser across sweep rows or Monte Carlo samples (each row's first
  evaluation still runs the cold scan).
- Hasan-Kabir heat transfer; gradual tapers; relief-well hydraulics and injection
  temperature; two-phase critical flow beyond HEM (Henry-Fauske); mud rheology in the
  liquid friction.
- Hand checks: Taitel v_ms constants, bubbly sin(theta), Firoozabadi-Katz attribution.
