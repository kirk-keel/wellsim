# Carried into M4 and beyond

M3 (multiphase: holdup closure, two-phase march, gas-liquid nodal, dynamic kill) closed
22 Sep 2026. Items below are open; each names where the placeholder or gap lives.

## Earlier: inverse mode (done 22 Sep 2026)

Pulled ahead of M4 because M4 is gated on hand checks; inverse mode is a prerequisite
for Monte Carlo.

- Done: effective kh from an observed rate (the forward nodal operating point equals the
  observation; h, skin and p_r fixed; Brent in ln kh to 0.5% on q_g) and p_r from a
  shut-in wellhead pressure through a static real-gas column (`core/inverse.py`, the
  `observed` case block, `tests/test_inverse.py`). Each result carries dq_g/d(ln kh) and
  the kh range for q_g +/- 20% (elasticity about 0.3 on both examples: a 20% rate error
  is a factor 1.9 on kh).
- Open: no published inverse case (a field rate with a known kh) is wired in; q_l in
  'pi' mode is refused (J is coupled to the unknown pwf); the SIWHP inverse has no
  liquid leg and no afterflow model; inversion cost is 10 to 26 forward solves.

## Now: rw-1, relief-well hydraulics for the dynamic kill

One relief well: surface lines, a string on its own trajectory, an exit restriction (bit
TFA or an intersection loss coefficient), pump pressure and hydraulic horsepower at the
kill rate, a free-fall check, injection-temperature bounds, a pump schedule, and the pump
pressure carried through the mc-1 kill samples. Two relief wells (shared manifold, split
rate, or sequential intersection) are NOT in rw-1; they go on the deferred list below.

Test budget: the default suite is at 170 s (752 tests) after mc-1. rw-1's tests stay
fast -- the pump balance is closed-form arithmetic on top of the existing Colebrook
routine, so everything but the two kill searches (the injection-temperature bounds) is
sub-second, and those go under `-m slow`.

### rw-2 (22 Sep 2026): the relief-string arrival temperature

Done. The kill fluid's arrival temperature at z_inj now comes from Ramey (1962) for
downward single-phase flow, integrated along the relief string's own survey, instead of
being bracketed between the surface and geothermal temperatures. The U -> infinity limit
(no pipe, annulus or cement resistance) makes it an upper bound on arrival temperature,
not an estimate; the surface temperature is the lower bound, and both kill rates are
reported with the gap between them.

Open on rw-2, carried forward:

- The rw-2 temperature brackets, on ks-2 corrected rates: surface 60 degF gives 26.1604
  bbl/min, the Ramey arrival 82.83 degF gives 26.7735, and the geothermal 180 degF gives
  27.9771. So the rw-2 bracket is **2.34%** (surface to arrival) and the rw-1 bracket was
  **6.94%** (surface to geothermal), both under the project convention of dividing by the
  smaller value, which `gap_relative` on the result object now follows. The figures rw-2
  first reported, 0.99% and 5.27%, were measured with the pre-ks-2 search and are
  superseded. rw-2 still closes the bracket about threefold, but 2.34% is above the 1%
  search tolerance rather than at it, so rw-3 below is worth doing.
- The Hasan-Kabir time function f(t_D) as published is NOT continuous at t_D = 1.5: the
  short-time branch gives 0.87399 and the long-time branch 0.85265, a jump of 0.02134
  (2.50%; was written as 2.44%, which divided by the larger value against our own
  convention, corrected at hk-1). rw-2 implements both branches exactly as published and
  pins the jump rather
  than tuning a coefficient to close it. It is on the hand-check list below. The jump does
  not make the kill search jagged: t_D = alpha t / r_w^2 does not depend on rate, so f is a
  constant for a given case and A varies only through the mass rate.
- A finite U (pipe, annulus, cement) would turn the bound into an estimate.
- The blowing well stays adiabatic; rw-2 touched the relief string only.
- FIXED by ks-2 (22 Sep 2026), kept here for the record. `min_kill_rate` was not
  start-independent to its stated 1% tolerance. On rw_wgr at 12 ppg
  the same search at the same injection temperature returns 25.8774 bbl/min from a warm
  start below the root (q_start of none, 0 or 20 bbl/min, three runs agreeing to 0.005%)
  and 26.6278 bbl/min from a start above it (q_start 30), a 2.90% spread. The pattern
  repeats at the other injection temperature (2.67%). So the search converges cleanly from
  below and lands about 3% high from above, which is larger than the whole 0.99% arrival
  temperature bracket rw-2 just closed, and larger than the 1% tolerance the result is
  pinned at. Two consequences: the rw-2 bound gap is only meaningful because both runs
  approach from below (the arrival run warm-starts from the surface run), and the mc-1 kill
  chains warm-start from the previous sample q_k_min, which can be above or below the next
  sample answer. The likely cause is the confirmation loop at the end of `min_kill_rate`,
  which steps q_k up by 1% until the full scan finds no root and so inherits wherever the
  bracket left it. ks-2 confirmed that diagnosis (out/ks-diagnosis.md), deleted the
  scan gate and the 1% bump, made the Brent tolerance relative to the root, and added an
  always-on proving check. Four starts now agree to 0.44% inside a 1% tolerance. The
  re-pins are in docs/PROVENANCE.md; at both old 12 ppg pins the residual minimum was about
  -54 psi, so the old answers were not merely imprecise, the well was still flowing.

### Recorded from mc-1 (22 Sep 2026), before rw-1 starts

Across kh 500 to 20,000 md-ft on the mc_wgr example the blowout rate is reservoir-limited:
kh ranks +0.989 against q_g and nothing else comes near (p_r +0.118, cd -0.040, net pay
+0.006). That means the reported rate spread is very nearly a restatement of the kh prior,
not an emergent result: widen the kh range and the rate band widens with it, almost one
for one. Per job, kh must come from an offset-well or log-based range, or be pinned by a
rate band; a generic factor-40 kh range produces a factor-4 rate range that says more
about the analyst than about the well. The elasticity dlnq/dlnkh near 0.3 (core/inverse.py)
is the same statement from the other side.

## Earlier: mc-1, Monte Carlo on blowout rate and kill rate

mc-1: Monte Carlo sampling p_r, net pay, open interval and exit condition (triangular),
kh log-uniform; an observed rate band is a filter on forward samples, never an input to
invert. Outputs: blowout rate, static-hold mud weight and kill rate at the 10/50/90th
percentiles, labeled in words, with an input ranking.

The design changed from the earlier "Monte Carlo on kh" (sample around an inverted kh)
because a rate estimate from a flare or a plume is not a defensible input on most jobs,
while reservoir pressure, net pay and the exit condition are: they come from offset wells,
logs and the wellhead, not from watching a fire. Inverting an observed rate to kh and
sampling around it would launder that indefensible number into the answer. A rate band,
when there is one, is applied the other way round: as a filter on forward samples, which
says which combinations of the defensible inputs are consistent with what was seen.

## Done: rw-3, finite U on the relief string (22 Sep 2026)

rw-3: finite U on the relief string (pipe wall, annulus, cement and f(t) in series) to turn
the Ramey bound into an estimate. Ahead of full Hasan-Kabir wellbore heat transfer, and a
much smaller job: the relaxation length already has the right shape,

    A = w c_p [ 1 / (2 pi r_to U) + f(t) / (2 pi k_e) ]

and rw-2 implements only the second term, the U -> infinity limit. rw-3 adds the first:
the overall coefficient U from the fluid film, the pipe wall, the annular fluid and the
cement sheath in series, referred to r_to. That makes A larger, the fluid arrive cooler,
and the answer an estimate bracketed by rw-2 above and the surface temperature below,
instead of a 2.34% band.

What it needs that rw-2 does not have: pipe OD and wall, the annulus geometry and fluid,
the cement sheath thickness and conductivity, and a convection correlation for the inside
film. Those are inputs a relief-well plan has. The blowing well stays adiabatic in rw-3;
full Hasan-Kabir (the blowing well march, transient and two-phase) stays deferred below,
and rw-3 does not depend on it.

Done. `1/U = r_to/(r_ti h_i) + r_to ln(r_to/r_ti)/k_pipe + r_to ln(r_w/r_to)/k_cem`, with
h_i from Dittus-Boelter and the formation f(t) term kept where rw-2 has it. Three
injection temperatures are now reported and the MIDDLE one is the answer: on rw_wgr,
surface 60.00 degF gives 26.1604 bbl/min, the finite-U estimate 64.03 degF gives 26.3647,
and the U -> infinity bound 82.83 degF gives 26.7767, so 0.78% below and 1.56% above the
estimate, 2.36% end to end.

What rw-3 showed, and it is the useful result: **the cement annulus is 99.35% of the
resistance** on this string (R_film 0.000371, R_pipe 0.001290, R_annulus 0.255409 hr ft2
degF/Btu), so U is only 3.89 Btu/(hr ft2 degF) and A is 167,497 ft against a bound of
26,260 ft and a string of 11,500 ft. The kill fluid gets 4.03 degF of the 120 degF
available, 3.4%, where the rw-2 bound said 22.83 degF. So the true arrival temperature
sits very near the COLD end of the rw-2 bracket, not in the middle of it, and the rw-2
bound was loose by about a factor of five in heating.

Open on rw-3, in the order they matter:

- k_annulus (rw-3b renamed it from k_cem) carries almost all of the RESISTANCE, but not
  the answer: over its plausible range, 0.4 to 1.2 Btu/(hr ft degF), the arrival
  temperature moves 3.62 degF and the converged kill rate moves 0.2768%, inside the 1%
  search tolerance. The 0.1 to 100 sweep rw-3 ran is what made it look decisive, and that
  is not a range any annulus occupies. It is off the hand-check list.
  What the fill IS matters for the default: on a dynamic kill the relief annulus holds
  STATIC drilling fluid, because the kill fluid goes down the string and out at the
  intersection and nothing returns. `annulus_fill` is 'mud' by default (0.7) with 'cement'
  (1.0) the alternative; both warn when the conductivity is defaulted.
- The annulus resistance assumes a solid conductive sheath from r_to to r_w with no
  convection in it. An uncemented interval is handled (k_e over the same annulus, flagged
  in the output) but a fluid-filled annulus with natural convection is not.
- Dittus-Boelter and the three-resistance series form are both MEMORY.

## M4: black-oil PVT and oil IPR (gate G3)

**`docs/M4-design.md` is the authoritative M4 plan** (owner design session, 25 Sep 2026;
committed b53e56e). It carries the owner decisions -- water always its own phase,
Vasquez-Beggs primary with Standing as the toggle, the A/B/C fluid input tiers, the tight
emulsion default and its oil-water no-slip consequence, the required `pvt_sensitivity`
run and its 25% shoe-margin criterion -- and the M4-0 to M4-5 sequence. Do not plan M4
from this section; it records only what M4 owes the rest of this file.

- Black-oil hand checks gate M4 and come first (M4-0, `docs/validation/M4-pins.md`):
  Vasquez-Beggs, Standing, Beggs-Robinson, McCain water and the Woelflin emulsion curve,
  each against its source (`docs/PROVENANCE.md`, hand-check list).
- Oil IPR (`core/ipr/oil.py`) is M4-3; the `black_oil_annulus_kill` example is unblocked
  by M4-1/M4-2. Note the design supersedes the old "`Case.liquid` filled from the
  black-oil PVT" framing: water does not fold into a single liquid, it becomes a typed
  phase of its own, and the WGR example's generic `Liquid` is migrated and repinned.

### G3, the oil gate

- **G3 as defined, and it stays the primary gate: published blowout-rate cases (Oudeman,
  the Norwegian studies) reproduced within 15%.**
- **Macondo is an extrapolation CHECK, not pass/fail.** Compare the model's rate against
  the FRTG (Flow Rate Technical Group) estimate range and report where it lands; do not
  assert a tolerance on it. Its GOR near 3,000 scf/stb sits above Vasquez-Beggs' fit
  range (roughly 2,200 scf/stb; the exact range is confirmed in M4-0), so a miss here is
  evidence about the correlation's range, not about the model.

Per case, whether it is an oil well and whether it sits inside the Vasquez-Beggs fit
range, **if the source says so**. UNREAD means the source has not been read in-session
and nothing is being asserted about it:

| Case | Oil well? | Inside VB fit range? |
| --- | --- | --- |
| Oudeman | UNREAD | UNREAD |
| Norwegian studies | UNREAD | UNREAD |
| Macondo | yes (light crude) | **no** -- GOR near 3,000 scf/stb against a VB range near 2,200 |

The Macondo row restates `docs/M4-design.md` §Gates, which is where that GOR claim is
recorded; the exact VB range is an M4-0 deliverable. The two UNREAD rows must be filled
from the papers before G3 is run -- G3's 15% bar is only meaningful on an oil well whose
fluid is inside the correlation being gated, and if neither Oudeman nor the Norwegian
studies turns out to be an in-range oil well, the design doc's "second oil case, dataset
TBD" is the item that closes that gap.

## After M4: the ANNULAR FILM FRICTION MODEL, as a hypothesis to test

**GATED (26 Sep 2026, per `docs/M4-design.md` §Deferred): this comes AFTER M4, and
it is gated on finding an annular dataset with more than G2's two annular rows.**
Air-water at 14.7 psia is a poor proxy for a blowout: LSU's liquid-to-gas density
ratio is about 832:1 against 5.6:1 at the `gas_water_wgr` sandface, a factor of 150,
and slip is hardest at the extreme LSU occupies. Two rows at the wrong density ratio
cannot decide between the homogeneous form and a core-film model, so testing against
G2 alone would be a measurement of the dataset, not of the model. Until such a
dataset is read this item is a standing hypothesis, not scheduled work; the section
below is kept in full because the evidence assembled for it is still current.

**Hasan & Kabir sec. 3.3.4, Eqs. 3.50-3.62: gas core over a liquid film, with Wallis
entrainment, core density and core velocity on (d - 2 delta), and the interfacial friction
factor f_c = f (1 + 75 f_L) (Eq. 3.58). Not a planned change: a hypothesis with a
measurable outcome, and no dataset yet adequate to measure it on.**

**Order of the Hasan-Kabir leads, settled at hk-2, re-gated 26 Sep 2026.** This one is
first among the six and the rest are not queued behind it, they are closed or parked --
but first among them no longer means next: M4 is ahead of all six.

    1. annular film friction, sec. 3.3.4     <- the only live lead, AFTER M4,
                                                gated on a new annular dataset
    2. large-diameter C0                     REJECTED hk-2, kept as a limitation
    3. Hughmark C0(Re), sec. 4.3             future, unscheduled, wrong regime
    4. annular C0 > 1                        CLOSED hk-1, against changing it
    5. sin(theta) on bubbly v_d              owner's decision, not a lead
    6. f(t_D) Eq. 6.10, Cicchitti mu_m       owner's decision, not leads

The film model is first among the six because it is the only one that is a complete,
citable model aimed at the term G2 actually indicts. What it lacks is data to be judged
on: "the rows where G2 actually fails" are two annular rows, which is the gate above.

REWRITTEN THREE TIMES. 23 Sep 2026 after g2-4; **25 Sep 2026 after hk-1**, which read
Hasan & Kabir directly (`out/hk-locations.md`); and again the same day after **hk-2**,
which read sec. 3.3.1 in full (`out/hk-c0-diameter.md`) and struck the large-diameter C0
lead hk-1 had just added. Every rewrite was forced by evidence changing under the item. The
hk-1 pass was the largest - two of the three things this item asserted turned out to be
wrong - and the hk-2 pass is the reminder that a lead logged from one paragraph can fail on
the next four.

### What hk-1 changed about this item

**1. There is no general non-homogeneous friction fix to make. That half of the item is
deleted, not deferred.** Hasan & Kabir Eq. 2.19 is

    -dp/dz =   f_m v_m^2 rho_m / (2 g_c d)      <- homogeneous friction
             + (g sin alpha / g_c) rho_m
             + (rho_m v_m / g_c)(dv_m/dz)       <- homogeneous kinetic

and the book restates it per regime in prose: bubbly (sec. 3.3.1) "using the homogeneous
model for estimating frictional pressure drop ... without introducing any significant
inaccuracy"; slug (sec. 3.3.2) "the homogeneous model may be used to estimate these two
components"; churn (sec. 3.3.3) uses the slug equations. **Our homogeneous friction is
what Hasan-Kabir prescribe**, not a shortcut we took, and "gated on Hasan-Kabir" was
gating on a fix the source does not contain.

**2. The annular C0 item is CLOSED, and closed against changing it.** See below.

**3. What survives is narrower and better founded**: annular flow is the one regime where
the book abandons the homogeneous form, and it supplies a complete film-and-core model to
replace it. That is now the item.

### The replacement item: the annular film friction model

Sec. 3.3.4, "Frictional loss constitutes a very large fraction of the total pressure
gradient in annular two-phase flow; consequently, an accurate estimation of the friction
factor becomes highly important."

    Eq. 3.50/3.51   total gradient written on the gas core
    Eq. 3.52        Wallis (1969, p.396) entrainment fraction E
    Eq. 3.53        critical gas velocity v_sgc
    Eq. 3.54/3.55   superficial core velocity and core density (gas + entrained drops)
    Eq. 3.56        in-situ core velocity on the reduced area pi(d - 2 delta)^2/4
    Eq. 3.57/3.58   Wallis interfacial friction factor, f_c = f (1 + 75 f_L),
                    f = 0.32/Re_c^0.25 smooth, and 4 delta/d = f_L from geometry
    Eq. 3.59        Whalley-Hewitt (1978) f_c, preferred when E <= 0.9
    Eq. 3.60-3.62   Ansari film thickness, core and film gradients equated

Eq. 3.58 has the right magnitude for what G2 measures: at a liquid holdup of only 0.05 the
interfacial friction factor is already 4.75x the smooth-pipe value.

**Test it, do not assume it.** The same section records a fallback that cuts the other
way: "at the high phase velocities involved in annular flow, the simple homogeneous model
performs quite well in estimating pressure loss and liquid holdup" (Fig. 3.14, Hasan et
al. 2010b). So the book itself does not claim the core model is always needed. The
measurable outcome is the G2 annular MAE (61.08% on 2 rows) and the 0.20 m block (25.04%,
9 rows carrying 69.9% of it). Note the closure predicts annular on 0 of those 28 rows, so
a model that only engages in annular flow may not touch the 0.20 m block at all - which
would itself be informative.

### CLOSED: the annular C0

**Annular C0 = 1.0 with v_d = 0 is Hasan & Kabir's own recommendation.** Sec. 4.4.1:

> "as in the case of annular two-phase flow in vertical and deviated wells, we recommend
> that liquid holdup in this regime be calculated on the basis of a no-slip approach;
> that is, f_L = v_sL/v_m."

`f_L = v_sL/v_m` is identical to `alpha_g = v_sg/v_m`, which is exactly what C0 = 1.0 and
v_d = 0 produce. Sec. 3.3.4 agrees: in annular flow "we do not need to estimate holdup,
given that we are modeling a true single-phase flow".

**So the 61% annular gradient error is the cost of the prescribed model, not a defect in
our implementation of it.** G2-2 recorded that "C0_ANNULAR = 1.0 with v_d = 0 is no-slip
by construction ... and the annular holdup is badly wrong because of it". The first half
is confirmed and is deliberate. Raising C0 above 1 in annular flow **departs from
Hasan-Kabir**; it does not restore it. Together with the gas-limit break that survived
g2-4 (alpha tends to 1/C0 = 0.892857, the M3-2 gas-limit test fails by 41% on pwf), there
is now no argument left for a constant annular C0 > 1 and one clear argument against.

The alpha-dependent C0 idea is not resurrected by this. If annular holdup is to improve it
must come from somewhere other than the distribution parameter, and sec. 3.3.4's core
model is where the book puts it.

### REJECTED: large-diameter C0 (hk-2, 25 Sep 2026)

**Investigated and rejected. Not pending, not deferred.** hk-1 logged this as "the next G2
hypothesis to test"; hk-2 read sec. 3.3.1 in full (`out/hk-c0-diameter.md`) and the lead
does not survive the reading. Four reasons, any one of which would be enough:

1. **The elevated values attach to STAGNANT liquid, not to diameter.** The trigger sec.
   3.3.1 states is a conjunction: "bubbly flow in large-diameter pipes (>100 mm) **with
   stagnant-liquid columns**", and the mechanism it gives - liquid recirculating back down
   the wall because net liquid movement is zero - requires stagnancy to exist at all. The
   only quantified threshold in the passage is a velocity one, v_sL < 3.0 mm/s.
2. **LSU is nowhere near it.** The slowest of the 131 rows is v_sL = 23 mm/s, already
   **7.7x** the 3 mm/s threshold; the fastest is 4.35 m/s, 1,450x it. Not one G2 row sits
   in the regime the elevated C0 values were measured in.
3. **The three elevated sources are not large-diameter pipe data.** From the book's own
   reference list: Rahman (1984) is an MS thesis on *annular geometry*, not the "127-mm
   vertical pipe" the prose calls it; Zahradnik and Kast Nek (1979) is *bubble column
   reactors*, which have zero net liquid throughput by construction; Kabir and Hasan (1994)
   is *pumping-well testing*, i.e. the near-stagnant annulus of a rod-pumped well. Only the
   1.6 at 140 mm attaches to moving liquid, and it carries **no citation at all**.
4. **The book does not carry the effect into its own model.** Chapter 5 is "Overall
   Modeling Approach"; Table 5.1 is its unified parameter set, and it has no diameter term:
   bubbly C0 is 1.2 upward cocurrent at every diameter, and the only 2.0 in the table is
   for *countercurrent* flow. Sec. 5.1.4 restates it without mentioning diameter.

It is also the wrong regime for G2's error. This is a **bubbly-flow** caveat, and it does
not carry to slug, churn (1.15, on the opposite argument - churning flattens the profile)
or annular. G2's per-diameter spread is driven by churn and annular rows; the 0.20 m
block's error is 9 rows above 19.8 ft/s that g2-4 attributed to friction. LSU has 4 Bubbly
and 3 Bubble rows out of 131. The hypothesis predicts error where G2's error is not.

**KEPT AS A LIMITATION, not as work.** Fixed C0 = 1.2 is **unproven in large-diameter
stagnant or near-stagnant liquid**, and the book says so itself: "the precise pipe size at
which the change in value for C_o occurs has not been investigated, nor do we know if the
transition is gradual or abrupt", and "the dependence of C_o on liquid velocity in
near-stagnant, large-diameter columns needs investigation". Chapter 4 adds the same unease
for flowing wells: "Further data, especially from large-diameter pipes, are needed to fully
establish the value of the flow parameter", the validating data being "from 1- to 2-in.-ID
pipes". That limitation matters where our own liquid is near-stagnant in a large annulus -
a loaded column, or the static relief annulus of rw-3b - and not on any shipped example
today. It is recorded in PROVENANCE.md; nothing is planned against it.

### FUTURE, separate item: the Hughmark C0(Re) correlation

Sec. 4.3 ("Flow in Horizontal Pipes and Wells", Intermittent Flow) carries the only C0
correlation in the book, in text:

    laminar,   Re < 1,000      C_o = 2.0
    intermediate               C_o = 3.04 - 0.151 ln(Re)
    turbulent, Re > 200,000    C_o between 1.2 and 1.35

with Re = d v_m rho_L / mu_L, attributed to Hughmark (1965), the turbulent bound also to
Gregory and Scott (1969). It is self-consistent with its own endpoints, giving 1.997 at
Re = 1,000 and 1.197 at Re = 200,000.

**Logged as a separate item, deliberately not merged with the rejected one above**, and
not scheduled. Three things to carry with it:

- it is **Reynolds-based, not diameter-based**: two pipes of different size at the same Re
  get the same C0, so it is not a diameter dependence even though d sits inside Re;
- it **runs opposite to the sec. 3.3.1 caveat**. At fixed v_m a larger pipe has a higher
  Re and therefore a *lower* C0 under Hughmark, while sec. 3.3.1 says larger pipes take a
  *higher* C0. The two cannot be describing the same effect, which is itself evidence that
  sec. 3.3.1 is about stagnancy rather than size;
- it is **outside its own validity window on 105 of the 131 G2 rows**. Only 26 fall inside
  1,000 < Re < 200,000 (6 of 40 at 0.10 m, 1 of 28 at 0.20 m, 19 of 63 at 0.30 m), and the
  book gives no formula above the window, only "between 1.2 and 1.35".

It is also for **horizontal and near-horizontal intermittent flow**, which is neither the
orientation nor the regime of a blowout. Numbers at our diameters are tabulated in
`out/hk-c0-diameter.md`. Nothing to do unless horizontal or near-horizontal flow becomes a
use case.

### Logged for the owner's decision, NOT acted on

- **sin(theta) on the bubbly drift velocity.** The book rejects it (sec. 4.2.2): "the
  terminal rise velocity of small bubbles is not appreciably affected by the inclination
  angle", and it names and declines Barnea et al. (1985)'s `v_inf sin alpha` as "doubtful".
  Inclination enters through the area projection instead, putting sin alpha on the whole
  transition criterion (Eq. 4.7), not on v_d. **Our term exists only to satisfy an M3-1b
  step-5 test criterion - that horizontal flow must give v_d = 0 in every regime - which
  is now known to be the wrong criterion.** Vertical results are unaffected (sin 90 = 1),
  so nothing shipped moves either way. Needs a decision, not a patch.
- **f(t_D): the book's continuous Eq. 6.10.** Sec. 6.2.2 acknowledges the discontinuity
  ("Eqs. 6.8 and 6.9 are discontinuous at t_D = 1.5") and supplies
  `T_D = ln[e^(-0.2 t_D) + (1.5 - 0.3719 e^(-t_D)) sqrt(t_D)]`, saying "For computing T_D,
  we consistently use Eq. 6.10". **It is not a smoothing of the two branches.** At
  t_D = 1.5 it gives 0.9068 against 0.8740 and 0.8526 - above both - converging on the
  long-time branch only at large t_D (0.33% apart at t_D = 200). Adopting it would move
  every rw-2/rw-3 arrival temperature, not merely close the seam. A judgement call about
  which correlation to ship, not a bug fix.
- **Mixture viscosity.** Sec. 2.4 recommends Cicchitti et al. (1960), mass-fraction
  weighted, `mu_m = x mu_g + (1 - x) mu_L`, and both worked examples use it. Ours is
  volume-fraction weighted, `alpha mu_g + (1 - alpha) mu_L`. It reaches the answer only
  through Re and hence the friction factor. Cheap to change, not obviously right to
  change; logged.

### What g2-4 changed about this item

The gate used to carry its own gradient with the homogeneous 1/(1-E_k) acceleration
factor, which the march superseded in M3-2b. g2-4 deleted it and pointed the gate at
`two_phase_gradient`. So every number G2-2 measured about acceleration was measured in a
term that is not in the model path.

**Which regimes that changed, and which it did not.** The churn and churn/annular errors
were largely the old gate's, not the closure's: those rows sit where the homogeneous E_k
is biggest, so the deleted factor was inflating the prediction there. Annular did not
move at all.

    regime                  old gate    march      whose error
    ------------------------------------------------------------------
    Churn (n=25)              28.99%     18.17%    largely the gate's
    Churn/Annular (n=4)       88.82%     54.35%    largely the gate's
    Annular (n=2)             61.07%     61.08%    the CLOSURE's, entire
    overall (131)             20.41%     14.42%
    0.20 m block (28)         43.89%     25.04%    largely the gate's

So the annular case against the closure is exactly as strong as it was, on exactly the
same two rows, and the surrounding numbers it used to sit inside were partly an artefact.
The fix order below is unchanged by that: non-homogeneous friction first, then an
alpha-dependent annular C0.

**1. The acceleration half of this item is closed, with nothing to do.** The march's slip
momentum flux already is the non-homogeneous form. On the 0.20 m rows that drove this item
it contributes 0.3% to 2% of the gradient, against a homogeneous factor of 1.09 to 1.44 on
the same rows. There is no acceleration work left here.

**2. The 340x argument against C0 = 1.12 is mostly gone.** Re-measured through the march
gradient (diagnostic only, nothing changed):

    C0_ANNULAR      overall MAE    annular MAE   run 8 H_l (reported 0.11)
    ------------------------------------------------------------------------
    1.0 (shipped)        14.42%         61.08%   0.0061
    1.12                 17.61%        231.23%   0.1126
    old gate, 1.0        20.41%         61.07%   0.0061
    old gate, 1.12      340.76%     20,777.70%   0.1126

Overall it costs 14.42% to 17.61%, a 22% relative worsening, not a 17-fold one. The
20,777% annular figure was almost entirely the deleted factor: E_k reached 0.9911 and
1/(1-E_k) reached 112, and the march's det on that row goes 0.937 to 0.988 instead.

**3. The disqualifying argument is now the gas limit alone, and it still disqualifies.**
With v_d = 0, alpha tends to 1/C0 rather than 1 as the liquid vanishes: at C0 = 1.12 a
pure-gas stream reports alpha_g 0.892857 and the M3-2 gas-limit test fails by 41% on pwf.
A constant C0 > 1 cannot be shipped for that reason and no other. The annular gradient
still gets 3.8x worse (61.08% to 231.23%) on two rows, which is a real cost but a thin
basis and no longer a blow-up.

So the order stands but for a different reason. It is no longer "C0 would explode the
gate"; it is "C0 must be alpha-dependent to survive the gas limit, and while the annular
gradient is 61% wrong on a friction term we cannot yet tell a C0 improvement from a
friction improvement".

### The evidence, from the 0.20 m block, re-measured through the march

The block MAE is 25.04% (was 43.89% under the gate's own gradient). The error is still
concentrated: 9 of 28 rows carry 69.9% of it, all over-predicting, all above 19.8 ft/s of
mixture velocity. The closure predicts annular on 0 of the 28, so no annular coefficient
could have touched them.

**The over-prediction is now unambiguously FRICTION.** Acceleration is ruled out by the
numbers above. On 7 of those 9 rows the measured TOTAL gradient is at or below the
hydrostatic gradient computed at the row's own REPORTED holdup - run 48 measures 0.05685
psi/ft against 0.05624 of gravity alone, 99% of it, leaving essentially no room for a
friction term, and the model adds 0.04004 on top. Over the 9, gravity at the reported
holdup averages 94% of the measured total. Two of the 9 over-predict in total while
UNDER-predicting holdup, which only friction can do.

Corroboration, and the only place Beggs-Brill beats the model path: through the same march
gradient, Beggs-Brill scores 9.85% on this block against the closure's 25.04%, and 9.60%
in churn/annular against 54.35%, while losing overall (18.46% against 14.42%) and badly in
the 0.30 m block. Its holdup is no better here, so the win is not a holdup win.

That is the whole point, and it is why this item is ordered friction-first. The ONE thing
Beggs-Brill has that the model path does not is a two-phase friction factor: it computes
friction from the no-slip mixture with its own flow-regime-dependent multiplier, rather
than a Moody f at the mixture Reynolds number applied to rho_m v_m^2. It beats the closure
precisely and only where friction dominates the error, and loses everywhere else. A
pointer at the term, not a proposal to adopt the correlation, which stays off the model
path.

The physical reading is unchanged: in churn and annular flow a fast gas core shears a slow
wall film, and f rho_m v_m^2 / (2 g_c D) with a Moody f at Re_m assumes a homogeneous
mixture moving at v_m, which is the wrong velocity and the wrong density for the shear
that is actually happening.

**No longer gated on Hasan-Kabir: hk-1 read it.** The annular film and core treatment is
sec. 3.3.4, Eqs. 3.50-3.62, and is the item at the top of this section. The "C0 form that
returns to 1 at the gas limit" is not in the book and is not wanted: the book's annular
recommendation is no-slip, C0 = 1. What the book does not settle is the Taitel v_ms
constants and the slug rise-velocity blend, both still on the hand-check list.

## Performance pass (done 22 Sep 2026; before Monte Carlo in M7)

Done (perf-2, commits ec80c9e, 08cccd9, 6d1e0e9): injection junction without
re-marching, analytic holdup sensitivity, closed-form closure solve. After: dry gas
0.30 s, WGR nodal 0.55 s, kill search 5.6 s, 19-row sweep 75 s (out/perf.md).
Done (perf step 4, 22 Sep 2026): warm-started kill search, each residual-minimum
evaluation after the first bracketed +/- 15% of the pwf span around the previous
minimiser; 12 ppg search 96 -> 71 outflow calls (4.3 s), 19-row sweep 1,184 -> 923
calls (60.5 s). Monte Carlo on blowout rate and on kill rate are both unblocked (mc-1
above). Remaining lever: carry the minimiser across sweep rows or Monte Carlo samples
(each row's first evaluation still runs the cold scan, 16 of its 49 calls). mc-1 carries
q_k_min across samples within a chain, which is the same lever one level up.

Timings before the pass, for the record:

- Dry-gas solves 0.3 s; the parted-tubing case 0.7 s quiet, over 1 s loaded.
- Two-phase nodal solves 1.1 to 1.5 s (each outflow evaluation is a march with about 300
  gradient calls, each with five holdup evaluations for the sensitivity).
- Kill-rate search 10 to 30 s per density (60 to 100 outflow evaluations); the rho_k
  sweep test runs under `-m slow` (about 6 min).
- Candidates: cache the DAK state between the gradient and the closure, a cheaper
  holdup sensitivity away from the ramp, a coarser march tolerance inside the kill
  search with a final full-tolerance confirmation.

## Gates

- **G2 is in** (23 Sep 2026). 131 LSU PERTT air-water runs from the Final Report Table
  A.1, compared as point values because the reports give neither tap spacing nor per-run
  pressure. Both of the open questions the gate page logged are answered: air-water (so a
  separate ideal-gas air path, DAK refused) and total gradients. PASS on the stated
  criterion, low-slip gradient MAE 2.97% against a 10% bar taken from LSU's own finding.
  See `docs/validation/G2-results.md`.
  A data consistency screen runs before scoring (`docs/validation/G2-consistency.md`).
  Table A.1's two reported columns contradict each other on 39 of the 93 rows that report
  a holdup, by more than each run's own stated uncertainty. They agree at low mixture
  velocity in every section and diverge above about 5 ft/s, with the sign of the
  divergence FLIPPING between the 3.83 in section and the two larger ones, which no single
  physical effect would do. Row 1 is separately wrong: its reported holdup of 0.99 needs
  23 times the drift-flux in-situ gas velocity. So gradient is scored on all 131 rows and
  holdup only on the 58 that pass (54 before g2-4 moved the acceleration term the
  screen strips).
  Open on G2, in the order they matter:
  - The closure degrades badly into churn and annular: gradient MAE 18.17% (churn),
    61.08% (annular), 54.35% (churn/annular), against 4 to 5% in bubbly and slug (all
    re-measured through the march gradient at g2-4; annular did not move). On the
    highest gas-rate row it predicts a liquid holdup of 0.01 against a reported 0.11. That
    is the one real physics finding from this gate and it wants investigating before any
    standing bar is set on those regimes.
  - The holdup column of Table A.1 is only partly usable, and the pattern of its failure
    (velocity-dependent, sign-flipping with pipe size) points at the shut-in-and-weigh
    measurement rather than at the model. Worth raising with LSU if the contact exists;
    worth not over-fitting a closure to in the meantime.
  - Standing bars are deliberately NOT set yet. Only the low-slip criterion is asserted;
    the holdup errors, the per-diameter and per-regime splits, the Beggs-Brill cross-check
    and the regime confusion are reported for the owner to read first.
  - The regime map disagrees more than the numbers do: 16 exact name matches out of 42
    labelled rows, with the closure calling slug where LSU observed churn on 15 of them.
  - Tap spacing and per-run pressure remain unavailable, so a marched version of G2 is not
    possible from these documents.
- G1 for the multiphase march is still open; no equivalent dataset has been read.

## Physics deferred

- Hasan-Kabir wellbore heat transfer. DEMOTED back to deferred by rw-2 (22 Sep 2026),
  having been promoted by rw-1 the same day. The 5.27% swing that promoted it was not a
  heat-transfer result: it was the width of a bracket, the gap between assuming the kill
  fluid arrives at z_inj at the surface temperature and assuming it arrives fully
  equilibrated with the formation. rw-2 computes the relief-string arrival temperature
  directly (Ramey 1962 in the U -> infinity limit, `core/relief_well.py`), which closes
  most of that gap from above, so the bracket is no longer the dominant uncertainty in the
  kill path. What is still deferred is the real thing: the blowing well's march is still
  adiabatic, and the relief-string arrival temperature is an upper bound rather than an
  estimate because the U -> infinity limit drops the pipe, annulus and cement resistance.
  Adding those resistances (the full Ramey A with a finite U, or Hasan-Kabir directly)
  turns the bound into an estimate; nothing in rw-2 depends on it first.
- Gradual tapers: every area change is sudden (`core/vlp/area_change.py`).
- Two or more relief wells: a shared manifold and a split rate, or sequential
  intersection at two depths. rw-1 does one relief well only; the kill search injects a
  single mass rate at a single z_inj, so more than one well needs the injection point (or
  the rate split) to become a list before the hydraulics can follow.
- Relief-well trajectory beyond a segment list: minimum curvature, dogleg severity and
  the string-in-hole annulus if the relief well circulates rather than injecting down the
  open string.
- Two-phase critical flow beyond HEM: Henry-Fauske or an equivalent (`core/vlp/two_phase.py`
  reports the Wallis frozen speed as a diagnostic; the choke is the 2x2 singular point).
- Mud rheology in the liquid friction: the kill fluid and the formation liquid are
  Newtonian (`core/kill.py`, `core/vlp/two_phase.py`); the M5 `kill` block already
  carries a rheology model that the friction does not use yet.
- Dense-phase gas region (Tpr 1.0 to 1.05 at high Ppr) refused, not computed.

## Hand checks still open

- Taitel, Barnea and Dukler 1980 dispersed-bubble constants (v_ms), first on the list
  since M3-1e. **hk-1 did not close this**: the book carries only the implicit Eq. 3.10
  balance, not our explicit 4.0/0.429/0.089/0.072/0.446 form, so AIChE J. 26, 345 is still
  needed. (The 0.52 cap and the annular 3.1 came off at hk-1, FETCHED.)
- The slug-flow average rise-velocity blend. Our linear ramp in alpha_g is ours; the book's
  Eq. 3.28 blends exponentially in a velocity ratio and its own two worked examples
  contradict both it and each other. Needs Hasan and Kabir (1988b) directly.
- ~~Hasan-Kabir deviated wells: sin(theta) on the bubbly drift; churn C0 = 1.15.~~
  **CLOSED at hk-1.** Churn C0 = 1.15 is FETCHED (sec. 3.3.3). The sin(theta) question is
  answered and answered against us - the book rejects it (sec. 4.2.2) - so it is no longer
  a hand check but an open decision, logged above.
- ~~Hasan-Kabir 1992 annulus Taylor-bubble coefficients (0.30 + 0.22 d_i/d_o).~~
  **CLOSED at hk-1**, and corrected: Eq. 4.75 gives 0.35 + 0.10 d_i/d_o.
- Firoozabadi-Katz curve-fit attribution.
- Cullender-Smith primary source and the SPE paper behind the UTP nine-well table.
- rw-3: the series-resistance form 1/U = r_to/(r_ti h_i) + r_to ln(r_to/r_ti)/k_pipe +
  r_to ln(r_w/r_to)/k_annulus, and the Dittus-Boelter Nu = 0.023 Re^0.8 Pr^0.4 heating
  form. Both MEMORY.
  (k_annulus itself came OFF this list in rw-3b. rw-3 put it on after sweeping 0.1 to 100,
  which is not a range any annulus occupies. Over the plausible 0.4 to 1.2 the arrival
  temperature moves 3.62 degF but the converged kill rate moves 0.2768%, inside the 1%
  search tolerance, so the model cannot resolve it and hand-checking it would buy nothing.
  At the shipped tolerance all three give the same 26.3648 bbl/min, which is the search
  quantising to its bracket rather than true insensitivity; the verdict rests on the
  converged 1e-4 spread.)
- ~~Hasan-Kabir transient time function f(t_D) (rw-2)~~ **CLOSED by hk-1.** All five
  constants are confirmed FETCHED against Hasan & Kabir 2nd ed. Eqs. 6.8 and 6.9, exactly
  as rw-2 has them: `T_D = 1.1281 sqrt(t_D)(1 - 0.3 sqrt(t_D))` for t_D <= 1.5 and
  `T_D = (0.4063 + 0.5 ln t_D)(1 + 0.6/t_D)` for t_D > 1.5. **The discontinuity is the
  book's own and the book says so**: "Eqs. 6.8 and 6.9 are discontinuous at t_D = 1.5". So
  no constant is wrong and the suspicion that one of the five was mis-remembered is
  retired. The jump is 2.50% (not 2.44%; see above). The book's own answer is its
  continuous Eq. 6.10, logged for decision in the section above - it is a different
  correlation, not a smoothing.
- Ramey (1962) relaxation length A = w c_p f(t) / (2 pi k_e) and the claim that the
  U -> infinity limit is an upper bound on arrival temperature for downward flow.
