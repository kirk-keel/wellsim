# Carried into M3

**M3 complete (22 Sep 2026): M3-1 holdup closure, M3-2 two-phase march, M3-3 liquid inflow
and gas-liquid nodal, M3-4 dynamic kill. Open items carried into `docs/TODO-M4.md`.**

Physics deferred during M2, in priority order. Each item names the warning the solver
emits today so the placeholder can be retired when the item lands.

## 1. Partial-penetration skin (done, M2.5-1)

Papatzacos (1987, SPE-13956-PA) pseudo-skin in `core/ipr/partial_penetration.py` from
`exposed_interval` (h_p), `exposed_top_offset` (h_1) and `kv_kh`; mechanical skin scaled
by h/h_p; D keeps h_p^2. The "skin is not applied" warning is retired; the result carries
a note with s_pp, h_pD and kv/kh, a warning when kv_kh is defaulted to 0.1, and a warning
when h_pD < 0.05. Formula VERIFIED against Lu et al. 2010 (see PROVENANCE.md). Still out of scope: deviated
(slanted) penetration through the pay; the pseudo-skin assumes a vertical well and a
horizontal open interval measured along the zone thickness.

## 2. Temperature: adiabatic total-enthalpy balance along the march (done, M2.5-3a/3b)

3a: analytic DAK partials, a single Z solve per gradient with
(dp/drho)_T = (RT/M)(Z + r dZ/dr) on the PVT state, and `core/pvt/real_gas_thermo.py`
with h_dep, h(T,p), cp_real, cv_real, mu_JT, adiabatic sound speed and an isenthalpic
throttle; DAK warns below Tpr 1.05; NIST methane gates pinned.
3b: `temperature.model = "adiabatic"` (default) marches [p, T] with the coupled momentum
and energy equations (`gas_gradient_adiabatic`), chokes where v = c_s (determinant
zero), conserves h + v^2/2 across area changes, and enters the wellbore after an
isenthalpic throttle from (p_res, T_res) to pwf (`sandface_jt`). "profile" keeps the
prescribed linear temperature bit-for-bit; `k_override` forces it. The result reports
sandface T, exit static T, exit Mach and the minimum Tpr. Real-gas cp/cv replaces k0 in
the adiabatic path; k0 survives only in "profile" mode.

Still open here:
- Wellbore heat transfer (Hasan-Kabir) is not modelled; adiabatic overstates cooling at
  low rates.
- DAK departures run a few percent small (NIST methane gates), so marched temperatures run
  warm by a few K per throttle-equivalent.

The linear profile (spec 4.4) fixes the exit temperature by hand, defaults it to reservoir
temperature minus 10 degF, and ignores expansion cooling. At blowout rates the gas
expands through thousands of psi in seconds, so the first real temperature model should be
an energy balance marched with the pressure:

    d(h + v^2 / 2 g_c + g z / g_c) = 0 per unit mass (adiabatic),

with real-gas enthalpy h(p, T) from the PVT model (ideal-gas cp plus the departure from
the same Z equation), the kinetic term from the march velocity and the potential term
from sin(theta). This gives Joule-Thomson cooling and the kinetic-energy temperature drop
at the exit without any heat-transfer input. Formation heat exchange is second-order at
blowout rates; Hasan-Kabir steady-state heat transfer is a later addition on top of this
balance, not a prerequisite. Retire the "default exit temperature is the warm end" warning
when the balance is the default.

Sensitivity that motivates it (dry_gas_vertical, exit temperature 250 to 60 degF):
rate +6%, flowing BHP -3%, choked exit pressure -9%.

## Also open

- Dry-gas model boundary for energy-balance inversions (exit and restrictions): states
  at Ppr > 1 with Tpr < 1.05, or at Ppr <= 1 with Tpr < 0.7 or Z < 0.3, are "not
  evaluable" (treated as the supersonic side). Cold dense exits (Tpr 1.0 to 1.05 at high
  pressure) are therefore refused rather than computed; a dense-phase EOS would be needed
  to go there.
- Nodal solve is in pwf space (default) with the IPR capped at its maximum; the
  rate-space solver stays behind `numerics.nodal_space = "rate"` for one release and
  then goes.
- Performance pass before Monte Carlo: the parted-tubing case exceeds the 1 s budget
  under machine load (1.14 s adiabatic, 0.72 s profile, with the vertical example at
  0.9 s); unloaded the same solves take 0.28, 0.18 and 0.22 s. Timing on this box
  varies 4x with load, so budget checks need a quiet machine or a relative measure.

- Done (M2.5-5): default non-Darcy beta is `auto`: Jones when only permeability is
  given, Geertsma when porosity is also supplied; the result notes which was used and
  why. Spec open question 2 is closed.
- Inverse mode: back out effective kh from an observed rate or shut-in pressure; typical
  inputs are a schematic, survey and DDRs only.
- Done (M2.5-2): cp/cv for the sonic velocity is the ideal-gas cp0(T)/cv0(T) of a
  gravity pseudo-composition (`core/pvt/heat_capacity.py`, NIST WebBook data), at local
  temperature; `Exit.k_override` reproduces the old constant 1.28. `delta_h0` (analytic
  integral of cp0) is ready for the enthalpy balance above.
- Real-gas isentropic exponent: retired for the adiabatic path in M2.5-3b (the choke
  criterion is the adiabatic sound speed from real-gas cp/cv); "profile" mode still pairs
  the ideal-gas k0 with the isothermal (dp/drho)_T.
- Done (M2.5-4): sudden area changes in the adiabatic march. A narrower conduit below
  (flow-direction expansion) is solved with the Borda-Carnot momentum balance, subsonic
  root, choked at the adiabatic sonic state when no root exists (this replaced the k0
  sonic reset); a wider conduit below (flow-direction contraction) undoes the Crane
  TP-410 loss K_c = 0.5 (1 - beta^2) on the small-pipe velocity head and decelerates
  isentropically (`core/vlp/area_change.py`, entropy in `real_gas_thermo.py`). Gradual
  tapers are not modelled: every change is treated as sudden. The `exit_id` restriction
  is entered the same way (Crane K_c when `exit.cd` = 1.0, isentropic only when cd < 1,
  Mach 1 checked at cd x the restriction area); the old treatment had used the throat
  static pressure as the conduit pressure. Profile mode stays pressure-continuous.
- Hand checks listed at the end of `docs/PROVENANCE.md`.

## M3 progress

- M3-1 (done, 21 Sep 2026): drift-flux holdup closure as a standalone module
  (`core/vlp/holdup.py`): Zuber-Findlay with Hasan-Kabir coefficients for bubbly,
  dispersed bubbly, slug, churn and annular flow, pipe and annulus Taylor-bubble
  velocities (the segment now carries d_o and d_i), inclination through the slug drift
  term, surface tension defaulted to 30 dyne/cm with a warning. Not in the march. The
  slug-churn line is a placeholder constant pending the owner's transition rules (see
  PROVENANCE.md, M3-1). `Case.liquid` (rho_l in ppg, the schema's one density kind; mu_l, cp_l, sigma) is the M3 liquid input
  until black-oil PVT fills it in M4.
- M3-2 (next): multiphase pressure gradient (gravity from the mixture density, friction
  on the mixture) and its march integration; G1 worked example; G2 LSU/BOEM comparison.
- M3-1b (done, 21 Sep 2026): Beggs-Brill 1973 cross-check in the tests (MEMORY, never in
  the model path) and the step-5 limit tests. Open from it, owner's call: (a) the closure
  is discontinuous at every regime switch (up to 21% in alpha_g) and non-monotone at
  bubbly to slug, so a blending scheme or a flow-pattern-independent closure (Bhagwat and
  Ghajar 2014, validation library 1.6) is needed before the march; (b) with C0 > 1 the
  slug and churn closures cannot reach alpha_g = 1 at v_sl = 0, so the march needs a
  single-phase-gas hand-off; (c) Beggs-Brill sits 25 to 42% above drift-flux on gas
  fraction at the bubbly and slug points; (d) the bubbly drift now carries sin theta
  (hand check pending).
- M3-1c (21 Sep 2026): logistic blend of C0 and v_d across every regime switch (width
  0.05 v_tr, COMPUTED); Beggs-Brill checked on H_L (20% band at bubbly and slug, annular
  pinned); v_sl = 0 guard returns regime "gas". The dip across the bubbly-to-slug
  switch at low v_sl was the slug closure stepping v_d, fixed in M3-1d.
- M3-1d (21 Sep 2026): slug and churn drift ramps linearly in alpha_g from the Harmathy
  velocity at 0.25 to the Taylor-bubble velocity at 0.52; implicit alpha_g by damped
  fixed point (1e-8, cap 50). Monotone on 3 to 17.5 in, vertical and 30 degrees, with
  the single 0.05 blend width.
- M3-1e (21 Sep 2026): slug-to-churn requires v_m > v_ms (Taitel dispersed line) and
  v_sg > 1.08 v_sl (the 0.52 ceiling in no-slip form), per Hasan, Kabir and Sayarpour
  2010 appendix A (FETCHED by the owner). Slug now spans 0.72 to 17.7 ft/s at v_sl = 1,
  6 in. Top hand check: the Taitel v_ms constants.
- M3-2 (done, 21 Sep 2026): two-phase adiabatic march (`core/vlp/two_phase.py`) for given
  gas and liquid mass rates through a ``model`` hook on ``march()``; hands off to the
  single-phase gas path (byte-identical) when case.liquid is absent or the liquid rate is
  zero. Choke at the singular point of the 2x2 system; the Wallis frozen sound speed is
  reported alongside (see PROVENANCE.md, M3-2, for why the two differ).
- M3-2b (21 Sep 2026): acceleration is the drift-flux momentum flux with the phase
  velocities of the energy equation (no-slip limit reproduces the homogeneous form to
  1e-10); area changes use the same flux; the 432 vs 382 psi liquid-limit friction gap
  was the schema default roughness (0.0018 in) against the hand value's 0.0006 in.
- Two-phase critical flow: HEM only; Henry-Fauske or an equivalent later.
- M3-3 (done, 21 Sep 2026): liquid inflow (wgr, pi) and the gas-liquid nodal solve in pwf
  space (`core/solver_two_phase.py`); every root returned and tagged by the derivative
  test; dry-gas path unchanged. Examples gas_water_wgr and gas_water_pi. Two-phase solves
  run 1.1 to 1.5 s: the performance pass must cover them before Monte Carlo.
- Gate G2: docs/validation/G2-lsu-boem.md lists the fields each LSU/BOEM case needs; no
  gate tests until the reports are read.
- M3-4 (done, 22 Sep 2026): dynamic kill with a Newtonian kill fluid (`core/kill.py`,
  `dynamic_kill` block): q_k_min from the no-roots criterion (continuous minimum of the
  residual), Blount-Soeiinah rate and static hold always reported; two-leg march with
  injection at z_inj. The rho_k sweep test runs under the slow marker (about 7 min).
- Relief-well hydraulics and pump pressure for the dynamic kill (injection temperature,
  the relief-well or drillstring path, pump limits: the M5 `kill` block) are not modelled;
  the kill fluid enters at the stream temperature at z_inj.
