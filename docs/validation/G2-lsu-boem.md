# Gate G2: LSU/BOEM large-diameter two-phase data

Target (spec section 7): match LSU's reported error for the same correlation. No gate
test exists yet; the data are not in hand. This page lists what a test needs from each
case so that the extraction can be done in one pass when the reports are read.

Sources (validation library 1.1, 1.2, 1.4):

- Waltrich et al., LSU for BOEM (2016), *Experimental Investigation of Two-Phase Flows in
  Large-Diameter Pipes and Evaluation of Flow Models Applied to Worst-Case-Discharge
  Calculations*, contract M15PC00007 (boem.gov).
- Teles, Waltrich et al., LSU for BOEM (2018), *Development and Improvement of Flow
  Models Applied to Multiphase Flows in Large-Diameter Pipes and High-Velocity Flows*,
  award M17PX00030 (boem.gov).
- LSU dissertation, drift-flux model for upward two-phase high-velocity flow in
  large-diameter pipes (repository.lsu.edu, gradschool_dissertations/5848).

## Fields a test needs from each case

Geometry (one row per test section):

| field | unit | notes |
|---|---|---|
| pipe inside diameter | in | large-diameter sections (6, 8, 12 in as reported) |
| test-section length | ft | between the pressure taps used for the gradient |
| inclination from horizontal | deg | vertical sections expected; confirm |
| roughness | in | as stated, or the material so a standard value can be tagged |
| pressure-tap positions | ft | needed if more than two taps |

Fluids:

| field | unit | notes |
|---|---|---|
| gas identity | - | air or natural gas; if air, the DAK PVT is not applicable and an air EOS row is needed |
| gas specific gravity or composition | - | |
| liquid identity | - | water, oil or a test liquid |
| liquid density | lbm/ft3 or ppg | at test temperature |
| liquid viscosity | cp | at test temperature |
| gas-liquid surface tension | dyne/cm | at test temperature |
| test temperature | degF | |

Rates and conditions (one row per run):

| field | unit | notes |
|---|---|---|
| gas rate | scf/min or Mscf/d | state the standard conditions used by LSU |
| liquid rate | gpm or stb/d | |
| superficial velocities | ft/s | as reported, to cross-check the rate conversion |
| inlet or reference pressure | psia | where the gradient starts |
| measured pressure gradient | psi/ft | or the pressures at the taps |
| measured liquid holdup | - | if available (quick-closing valves or tomography) |
| observed flow regime | - | as reported |

LSU's model comparison (to score against, per the G2 target):

| field | notes |
|---|---|
| correlations evaluated | Hagedorn-Brown, Duns-Ros, Beggs-Brill, Ansari, Hasan-Kabir, drift-flux variants, as listed |
| error statistic used | mean, absolute mean, standard deviation of relative error |
| reported error per correlation | the band the implementation must reproduce |
| any runs excluded by LSU | and why |

## How the test would run

For each run: build a case with the section geometry, the liquid block (rho_l, mu_l,
sigma, c_l), the reported gas and liquid rates and the reference pressure; call
`outflow_profile` (no inflow) over the tap spacing; compare the marched pressure
difference with the measured one; aggregate the relative errors and compare with LSU's
statistic for the drift-flux or Hasan-Kabir closure. Every extracted number gets a
PROVENANCE.md row tagged FETCHED with the page or table.

## BOTH OPEN QUESTIONS ANSWERED, 23 Sep 2026

The reports were read (`out/g2-sources.md`) and the gate built
(`wellsim/validation/g2_lsu.py`, results in `docs/validation/G2-results.md`).

- **Air-water, so the DAK PVT does not apply.** BOEM M15PC00007 p.13 and Interim Report
  Table 1 p.18: air and water at about 14.7 psia. `wellsim/core/pvt/air.py` is a separate
  ideal-gas path (Z = 1, k = 1.4, Sutherland viscosity) that does NOT subclass the
  gas-gravity model, and whose `dak_z` raises rather than returning a number for a fluid
  the correlation was never fitted to.
- **The reported gradients are TOTAL, hydrostatic included.** BOEM M15PC00007 p.44 defines
  dp/dz as gravitational + frictional + acceleration. Cross-checked independently in the
  test suite by magnitude and trend.

## What the data turned out to be

One usable set: **Table A.1 of the LSU Final Report, PDF pp. 155-158**, 131 runs. It is
the only per-run tabulation in the three reports; everything else is a summary of ranges
or a figure. The field data (Fancher and Brown, Asheim, Espanol, Reinicke) are NOT
reproduced per run anywhere, only as ranges, so they cannot be gated against from these
documents.

Two things the field list above asked for are simply not in the sources, and the gate is
built around their absence:

- **pressure-tap positions and test-section length per diameter.** BOEM M15PC00007 p.30
  says the sections are 20 to 36 ft and refers the detail to a figure. Not in the text
  layer. Without tap spacing there is no length to march over.
- **reference pressure per run.** Only "approximately 14.7 psi" campaign-wide.

So the gate compares POINT VALUES rather than marching: at each row's stated superficial
velocities it computes alpha_g from the closure and the total gradient from it, and
compares both against what was reported. The 14.7 psia assumption is stated and its
sensitivity computed once (16.0 psia moves the gradient -0.99% on average).

A third gap affects scoring rather than running: **LSU's per-correlation errors are in
figures, not tables**, so "match LSU's reported error for the same correlation" cannot be
done from the text. The pass criterion instead cites LSU's own sentence at BOEM
M15PC00007 p.59, "for usg/usl < 1, all models predict pressure gradient well", as a 10%
bar on the low-slip rows. Everything else is reported and not asserted until standing
bars are set.

## Column-order error in the source, carried forward

Table A.1's printed header reads `u_sg (m/s)  u_sl (m/s)` but its data columns are u_sl
then u_sg. Anyone reading the printed table naively will swap the two velocities. The
evidence and the three independent confirmations are in `docs/PROVENANCE.md` and in the
header comment of `docs/validation/data/lsu_table_a1.csv`.
