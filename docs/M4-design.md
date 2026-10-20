# M4 — Black oil: design record

Decided Sep 25, 2026 (chat design session). Baseline: HEAD 9204250 (hk-2), 783 default / 1 perf / 10 slow.
This file records decisions and their reasons. Milestone prompts live in `docs/prompts/M4-*.md`.

## Owner decisions

1. **Water is always its own phase.** Never folded into oil. Water carries its own density (with salinity), Bw,
   viscosity, gas solubility (Rsw), mass balance and cp through the whole march. Applies to gas wells too:
   the M3 generic `Liquid` in the WGR example becomes a typed water phase; the resulting pin moves are
   recorded in PROVENANCE as a repin, not a regression.
2. **Vasquez-Beggs is the primary oil correlation; Standing is the toggle.** Reason: VB is the most robust
   baseline across a wide global range of crudes. VB requires separator-corrected gas gravity, so separator
   conditions are an input; a defaulted value is flagged in the summary.
3. **Fluid input has three tiers**, chosen by what the job provides:
   - **A — PVT lab report:** correlations calibrated to measured points (measured pb shifts the Rs curve;
     measured Bo and viscosity at pb become multipliers).
   - **B — offset API / GOR / gas gravity:** correlations uncalibrated.
   - **C — wildcat/exploratory:** API, GOR, gas gravity and water cut as distributions into the Monte Carlo,
     from regional basin averages or geological analogs. API and GOR must be sampled jointly
     (rank correlation from analogs, or GOR conditioned on API) — independent sampling creates fluids that
     don't exist and widens P10/P90 dishonestly.
4. **Produced oil and water are a tight, sheared emulsion by default** (owner field experience: blowouts almost
   universally return a highly stable emulsion). Emulsion viscosity = Woelflin tight curve below the inversion
   point, water-continuous above it. Water remains a distinct phase for mass, density and energy; the emulsion
   sets only the liquid transport viscosity.
   - Consequence: oil-water **no-slip** is the default (droplets locked in the emulsion). An oil-water slip
     closure is deferred to shut-in / low-rate work (check the Hasan & Kabir book for an oil-water section).
   - Caution: tight emulsion lowers both the predicted blowout rate and the predicted kill rate, i.e. it is
     non-conservative on both headline outputs if wrong. Surface observation does not prove the flow is
     emulsified below the restrictions. Hence the emulsion toggle in the sensitivity run (item 5).
5. **VB-vs-Standing sensitivity run is required** (`pvt_sensitivity`): solve the case with both correlations,
   and with emulsion tight vs none, and report side by side: blowout rate, kill rate, kill mud weight,
   shoe pressure during the kill vs fracture pressure, pump pressure.
   - Pass criterion (owner confirmed 25%, Sep 25, 2026): the VB-Standing shoe-pressure difference is under 25% of
     the smaller correlation's margin to fracture. Beyond that, flag it.
   - Summary wording: agreement means the answer does not hinge on correlation choice, **not** that the model
     is right (both are regressions of the same functional form). Disagreement means the answer rides on the
     fluid: get lab data (tier A) or widen the fluid ranges.
   - Kick tolerance and max casing pressure during a conventional (driller's / W&W) kill need the transient
     model and are **not** produced by M4. They get the same VB/Standing toggle as a named Stage 2 requirement.

## Sequence

- **M4-0 — hand-check pins, no code (done in chat).** Each value triangulated against three independent
  reproductions; fit ranges recorded. Output: `docs/validation/M4-pins.md`.
  - Vasquez-Beggs: Rs, pb (inverted Rs), Bo saturated, co, Bo undersaturated, separator gas gravity
    correction, undersaturated viscosity; both API branches (split at 30 API).
  - Standing: Rs, pb, Bo.
  - Beggs-Robinson: dead-oil and live-oil viscosity.
  - McCain water: Bw, density with salinity, viscosity, Rsw.
  - Emulsion: Woelflin tight curve; inversion-point correlation.
- **M4-1 — PVT module.** Typed gas / oil / water phases behind the g2-3 pvt interface; solution-gas gravity
  feeds DAK. Tiers A/B (C wiring comes with the Monte Carlo item). Calibration hook for tier A.
  Range-violation collector: each out-of-range correlation reported once per solve in the summary;
  Lee-Gonzalez-Eakin moved onto it (the current ~59,670 per-run warnings would bury real oil-range warnings).
  First commit also brings CLAUDE.md "Current state" up to HEAD 9204250 / 783 tests.
- **M4-2 — three-phase march with mass transfer.** Free gas = q_o(R_p − R_s)B_g plus water-phase gas;
  oil-water no-slip; emulsion viscosity; separate oil/water cp in the energy equation; heat of solution
  neglected and flagged. WGR example migrated to typed water and repinned.
- **M4-3 — oil IPR.** Straight-line PI above pb, Vogel below; water via water cut / WOR; reuses the
  all-roots nodal (highest-rate stable root = operating point).
- **M4-4 — kill on an oil well.** Liquid on a kill = oil + water + kill fluid, each tracked. Reports shoe
  pressure during the kill against fracture pressure.
  **Schema confirmed against the code, 26 Sep 2026: both inputs are ABSENT and both are new.**
  Nothing in `wellsim/cases/schema.py` carries a casing shoe, a leak-off/formation-integrity test or a
  fracture gradient; the only adjacent fields are `HoleSection.top_md/bottom_md/kind` (`casing` /
  `liner` / `open_hole`) and the equipment ratings in `ReliefRatings` / `Pumps`, which are pipe and
  iron strength, not formation strength. Decisions:
  - **Shoe is DERIVED, not entered: the `bottom_md` of the deepest section with `kind == "casing"`,
    with its TVD taken from the survey** (the same MD-to-TVD path the relief string already uses, so a
    deviated well needs no extra input). A case with no casing section has no shoe and the shoe-pressure
    report is omitted with a warning rather than defaulted.
  - **Formation strength is a NEW input, accepting either form: LOT/FIT as an equivalent mud weight
    (ppg) or as a pressure.** Exactly one of the two, no default, converted to a fracture pressure at
    the shoe TVD at load; supplying both is refused, supplying neither omits the comparison with a
    warning. Follows the `bit_tfa` / `intersection_k` precedent in `ReliefRestriction`: one of two
    forms, never a silent default.
  - EMW-to-pressure uses the project's exact `rho * TVD / 144` basis, not 0.052, per the Blount-Soeiinah
    convention already in `core/relief_well.py`; the 0.10% difference is stated where it is printed.
- **M4-5 — `pvt_sensitivity`** per decision 5.

## Gates

- **G3 is the primary gate, as `docs/TODO-M4.md` defines it: published blowout-rate cases
  (Oudeman, the Norwegian studies) reproduced within 15%.** This design session did not
  change G3 and does not replace it; the entries below sit under it.
- **Macondo — an extrapolation CHECK, not pass/fail.** Compare the model's rate against the
  FRTG (Flow Rate Technical Group) estimate range and report where it lands; assert no
  tolerance on it. It is an oil well, and its GOR (near 3,000 scf/stb) sits above VB's fit
  range (roughly 2,200 scf/stb; exact range confirmed in M4-0), so a miss is evidence about
  the correlation's range rather than about the model.
- **Second oil case, within VB's fit range — dataset TBD. KEPT (26 Sep 2026).** The
  condition for dropping this line was that one of the G3 cases turn out to be an in-range
  oil well. Neither Oudeman nor the Norwegian studies has been read in-session, so neither
  can be claimed as one: both are UNREAD in the G3 table in `docs/TODO-M4.md`. Fill that
  table from the papers first; if a G3 case is then an in-range oil well, this line goes.

## Deferred (recorded, not in M4)

- Formation-pressure-test fluid gradient as an optional filter on Monte Carlo forward samples
  (same pattern as the observed-rate band); tier C wiring into the Monte Carlo.
- Oil-water slip closure (shut-in / low rate).
- Kick tolerance and max casing pressure (Stage 2 transient), with the VB/Standing toggle.
- Hasan & Kabir sec. 3.3.4 annular film friction model: next physics item after M4, gated on finding an
  annular dataset with more than G2's two rows (air-water at 14.7 psia is a poor proxy for a blowout).
