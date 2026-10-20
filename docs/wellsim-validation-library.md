# Wellsim Validation Library

Reference data and benchmark cases for validating a personal blowout / dynamic kill / kick simulator.

**Legend**
- ✅ = citation confirmed by search on 21 Sep 2026
- † = cited from memory; confirm the paper number on OnePetro before relying on it
- 🆓 = free public download. Everything else is OnePetro (SPE membership gets you discounted downloads; UWyo library access would make most of these free)

---

## Tier 1: Steady-state multiphase flow (VLP engine)

This is the core of Stage 1. If the pressure-drop engine is wrong, blowout rate and kill rate are both wrong.

| # | Source | Why it matters |
|---|---|---|
| 1.1 | ✅🆓 Waltrich et al., LSU for BOEM (2016), *Experimental Investigation of Two-Phase Flows in Large-Diameter Pipes and Evaluation of Flow Models Applied to Worst-Case-Discharge Calculations*, contract M15PC00007. On boem.gov. | The single most useful document here. Lab data in large-diameter vertical pipe at high rates, plus a head-to-head scoring of the standard correlations for WCD use. Tells you which correlations fail and where. |
| 1.2 | ✅🆓 Teles, Waltrich et al., LSU for BOEM (2018), *Development and Improvement of Flow Models Applied to Multiphase Flows in Large-Diameter Pipes and High-Velocity Flows*, award M17PX00030. On boem.gov. | Follow-up: an improved model (Teles & Waltrich, building on Pagan et al. 2017 and Duns & Ros) aimed at the large-diameter, high-velocity regime. Candidate closure for your engine. |
| 1.3 | ✅ Waltrich et al. (2017), *Experimental Evaluation of Wellbore Flow Models Applied to Worst-Case-Discharge Calculations*, SPE-184444-MS. | Paper version of 1.1. Key finding: small-diameter correlations can be off by more than 100% in large pipe, but agree within about 10% at low GLR. |
| 1.4 | ✅🆓 LSU dissertation, drift-flux model for upward two-phase high-velocity flow in large-diameter pipes (repository.lsu.edu, gradschool_dissertations/5848). | Drift-flux parameters fitted for WCD conditions. Bridges Tier 1 and Tier 3. |
| 1.5 | † Shi et al. (2005), *Drift-Flux Modeling of Two-Phase Flow in Wellbores*, SPE-84228-PA, SPE Journal. | Stanford/SLB 6-inch inclinable pipe data. Publishes optimized drift-flux parameters you can code directly. |
| 1.6 | † Bhagwat & Ghajar (2014), flow-pattern-independent drift-flux void fraction correlation, *Int. J. Multiphase Flow*. | Good general-purpose void fraction closure across inclinations and diameters. |
| 1.7 | † Classic correlations: Hagedorn-Brown (1965), Duns-Ros (1963), Orkiszewski (1967), Beggs-Brill (1973), Gray (1974), Ansari et al. (1994), Hasan-Kabir. | Implement at least three. Each original paper includes worked examples: use those as unit tests. |

**Acceptance target (proposed):** reproduce each correlation's published worked example within 2%. Against 1.1 lab data, match the error bands LSU reports for the same correlation. If you match their reported error, your implementation is right even where the correlation is wrong.

---

## Tier 2: Blowout rate (nodal: IPR + VLP + exit condition)

| # | Source | Why it matters |
|---|---|---|
| 2.1 | ✅ Oudeman (2010), *Validation of Blowout-Rate Calculations for Subsea Wells*, SPE-115019-PA, SPE Drilling & Completion 25(3): 282–289. | A Shell blowout-rate simulator validated against field cases. Covers the critical-flow-at-surface vs. subsea backpressure distinction. |
| 2.2 | ✅ Oudeman (1996/1998), *Analysis of Surface and Wellbore Hydraulics Provides Key to Efficient Blowout Control*, SPE-36485-MS; journal version in SPE D&C 13: 163–173. | Two field blowouts where rate was back-calculated from surface observations. Usable as test cases. |
| 2.3 | ✅ Kikani, Schwarz & Graute (1996), *A Method for Blowout-Rate Prediction for Sour-Gas Wells, Oldenburg Fields, Germany*, SPE-35582-PA. | Gas well case with sonic exit flow and transient early-time rates. |
| 2.4 | † SPE Technical Report (2015), *Calculation of Worst-Case Discharge (WCD)*. 🆓 on spe.org. | The industry's agreed method and input conventions. Follow this so your cases are structured the way BOEM submittals are. |
| 2.5 | ✅🆓 Norwegian blowout-rate study reports (e.g., Ranold reports for wells 7018/5-1 Spissa and others, published in Miljødirektoratet hearing documents). | Full commercial blowout-rate studies in the public record: inputs, scenarios (open hole / annulus / drillstring, surface / seabed), and resulting rates. Closest thing to free Drillbench/OLGA output you'll find. |
| 2.6 | ✅🆓 Offshore Norge guidance on calculating blowout rates and duration (and its amendment). | Scenario matrix and probability weighting conventions for a risked, multi-case rate study. Template for your Monte Carlo layer. |

**Acceptance target (proposed):** within 15% of published rates on 2.1–2.3 and 2.5 when using the same inputs. Differences beyond that need an explanation (PVT, correlation choice, exit condition).

---

## Tier 3: Macondo (complete blowout + kill case)

| # | Source | Why it matters |
|---|---|---|
| 3.1 | ✅🆓 McNutt et al. (2012), *Review of Flow Rate Estimates of the Deepwater Horizon Oil Spill*, PNAS 109(50): 20260–20267. Free on PMC (PMC3528583). | Consensus range of roughly 50,000–70,000 bopd, declining modestly over time, from independent methods. This is your target band. |
| 3.2 | ✅🆓 Hickman, Hsieh et al. (2012), *Scientific Basis for Safely Shutting in the Macondo Well*, PNAS, doi 10.1073/pnas.1115847109. | Shut-in pressure data from the well integrity test and the reservoir model behind it. Gives reservoir depletion and buildup behavior. |
| 3.3 | ✅🆓 Oldenburg et al. (2011), *Numerical Simulations of the Macondo Well Blowout Reveal Strong Control of Oil Flow by Reservoir Permeability and Exsolution of Gas*, PNAS. | LBNL coupled wellbore-reservoir simulation with stated inputs. A direct model-to-model comparison, and it proves the point that permeability uncertainty dominates. |
| 3.4 | ✅🆓 National Incident Command / Flow Rate Technical Group, *Assessment of Flow Rate Estimates for the Deepwater Horizon / Macondo Well Oil Spill* (DOI, 2011). | The Nodal Analysis Team appendix is the relevant part: several national labs each ran nodal models with stated assumptions. |
| 3.5 | †🆓 BP *Deepwater Horizon Accident Investigation Report* (Bly report, 2010), Appendix W: dynamic (OLGA) modeling of the blowout by Add Energy. | OLGA results for the initial unloading and flow paths. Actual OLGA curves to compare against. |
| 3.6 | †🆓 MDL 2179 Phase 2 (quantification) expert reports. | PVT (Zick), reservoir properties, well geometry, and competing flow models from both sides. Most detailed public input set for any blowout. |
| 3.7 | † Top kill and static kill pump data (rates, mud weights, pressures) from government and BP technical briefings, summer 2010. | Dynamic kill validation: top kill failed at the rates pumped, static kill succeeded after capping. Your model should reproduce both outcomes. |

**Acceptance target (proposed):** with public inputs, land inside the 50–70k bopd band, and show that top kill fails while static kill succeeds.

---

## Tier 4: Dynamic kill

| # | Source | Why it matters |
|---|---|---|
| 4.1 | † Blount & Soeiinah (1981), *Dynamic Kill: Controlling Wild Wells a New Way*, World Oil. | The original method, from Mobil's Arun C-II-2 kill. Steady-state approach: good Stage 1 benchmark. |
| 4.2 | † Kouba, MacDougall & Schumacher (1993), *Advancements in Dynamic Kill Calculations for Blowout Wells*, SPE-22559, SPE D&C. | Multiphase steady-state kill method with worked examples. |
| 4.3 | † Rygg & Gilhuus (1990), SPE-20433; Rygg, Smestad & Wright (1992), *Dynamic Two-Phase Flow Simulator: A Powerful Tool for Blowout and Relief Well Kill Analysis*, SPE-24578. | The first OLGA-based kill studies. Published OLGA kill curves to compare against. |
| 4.4 | ✅ Upchurch et al. (2017), *Blowout Prevention and Relief Well Planning for the Wheatstone Big-Bore Gas Well Project*, SPE-174890, SPE D&C. 🆓 slide deck on SPE streaming site. | Contains a table of historical relief-well dynamic kills (Arun, Ixtoc, Fateh L-3, El Isba, Enchova, West Cameron 165, Sheep Mountain, others) with pump rates and fluid weights. A ready-made list of field cases, plus a modern big-bore gas design case. |
| 4.5 | ✅ Lugo & de Leon (1981), *Ixtoc No. 1, Blowout and Control Operation*, SPE-9697. | Field data for a major offshore relief well kill. |
| 4.6 | ✅ Oudeman et al. (1994), *Bull Heading to Kill Live Gas Wells*, SPE-28896-MS. | Bullheading efficiency criteria with field validation. Needed once you add bullhead kills. |
| 4.7 | † Noynaert & Schubert (2005), *Modeling Ultra-Deepwater Blowouts and Dynamic Kills and the Resulting Blowout Control Best Practices Recommendations*, SPE-92626. Related Texas A&M theses 🆓 on OAKTrust. | Academic dynamic kill simulator (COMASim) with documented method and cases. The theses include equations and sample runs. |

**Acceptance target (proposed):** predicted minimum kill rate within 20% of what was actually pumped to succeed, and correct prediction of kill / no-kill for each field case in 4.4.

---

## Tier 5: Transient kicks and gas migration (Stage 2+)

| # | Source | Why it matters |
|---|---|---|
| 5.1 | ✅ Udegbunam, Fjelde, Evje & Nygaard (2015), *On the AUSMV Hybrid Scheme: A Simple Transient-Flow Model for MPD and UBD Applications*, SPE-168960-PA, SPE D&C 30: 98–109. | The numerical scheme to build on. Written for teaching, so it's complete enough to code from, with test cases to reproduce. |
| 5.2 | ✅ Evje & Fjelde (2002), *Hybrid Flux-Splitting Schemes for a Two-Phase Flow Model*, J. Computational Physics. | The math behind 5.1, including standard shock-tube style test problems for verifying the solver. |
| 5.3 | ✅🆓 Rommetveit & Blyberg (1989), *Simulation of Gas Kicks During Oil Well Drilling*, Modeling, Identification and Control 10(4): 213–225. Free PDF at mic-journal.no. | The Rogaland Research kick simulator (Drillbench's ancestor) described openly, with WBM vs. OBM example runs. |
| 5.4 | ✅ Nickens (1985), *A Dynamic Computer Model of a Kicking Well*, SPE-14183. ✅ Ekrann & Rommetveit (1985), *A Simulator for Gas Kicks in Oil-Based Drilling Muds*, SPE-14182. | The two foundational kick simulator papers. Example cases in both. |
| 5.5 | † Rommetveit & Olsen (1989), *Gas Kick Experiments in Oil-Based Drilling Muds in a Full-Scale, Inclined Research Well*, SPE-19561. | The Ullrigg full-scale experiments. Measured pressures and pit gain vs. time. |
| 5.6 | † Hovland & Rommetveit (1992), *Analysis of Gas-Rise Velocities From Full-Scale Kick Experiments*, SPE-24580. † Johnson & White (1991), *Gas-Rise Velocities During Kicks*, SPE-20431. | Measured gas rise velocities: direct check on your slip closure. |
| 5.7 | ✅ Rader, Bourgoyne & Ward (1975), *Factors Affecting Bubble-Rise Velocity of Gas Kicks*, JPT. ✅ Matthews & Bourgoyne (1983). | LSU bubble-rise experiments. Showed the single-slug assumption is invalid. |
| 5.8 | ✅ O'Bryan, Bourgoyne, Monger & Kopcso (1988), *An Experimental Study of Gas Solubility in Oil-Based Drilling Fluids*, SPE Drilling Engineering. | Solubility data for the OBM stage. |
| 5.9 | ✅🆓 LSU PERTT Lab instrumented 5,800 ft test well project (National Academies Gulf Research Program grant, Almeida/Williams). Search LSU repository for resulting theses and papers on riser gas and fiber-optic-monitored gas migration. | Modern full-scale data with distributed pressure/temperature sensing: far better resolution than the 1980s experiments. |
| 5.10 | 🆓 OpenLab Drilling (NORCE), openlab.app. | Free transient simulator with a Python API. Run identical cases in both and compare curves. Model-to-model, not model-to-data, but fast. |

**Acceptance target (proposed):** solver reproduces 5.2 test problems; gas arrival time within 10% and peak casing pressure within 10% on the Ullrigg and LSU WBM cases.

---

## Tier 6: What you're being compared to

| # | Source | Why it matters |
|---|---|---|
| 6.1 | † Bendiksen, Malnes, Moe & Nuland (1991), *The Dynamic Two-Fluid Model OLGA: Theory and Application*, SPE-19451-PA, SPE Production Engineering. | The OLGA model structure, published. Shows exactly which closures they use empirical fits for. |
| 6.2 | ✅🆓 SLB Drillbench product sheets (Dynamic Hydraulics, Dynamic Well Control, Blowout Control). | Feature list to benchmark scope against. |
| 6.3 | ✅ Karlsen & Ford (2014), *BlowFlow: Next Generation Software for Calculating Blowout Rates*, SPE-169226-MS. | A risk-based blowout rate tool built outside SLB. Proof the "many fast runs over uncertain inputs" approach is accepted practice, which is your competitive angle. |

---

## Suggested repo layout

```
wellsim/
  validation/
    cases/
      tier1_vlp/        hagedorn_brown_example.yaml, lsu_boem_run_XX.yaml ...
      tier2_rate/       oudeman_case_A.yaml, spissa_case1.yaml ...
      tier3_macondo/    inputs.yaml, top_kill.yaml, static_kill.yaml
      tier4_kill/       arun_c_ii_2.yaml, ixtoc.yaml, wheatstone.yaml ...
      tier5_transient/  ausmv_shocktube.yaml, ullrigg_wbm_01.yaml ...
    sources/            PDFs (gitignored), plus sources.bib
    run_validation.py   runs every case, writes a pass/fail + error table
    REPORT.md           auto-generated scorecard
```

Each case YAML holds: source citation, inputs with units, the published answer, tolerance, and a notes field for anything you had to assume because the paper didn't state it. The assumed inputs are where most "validation failures" actually come from, so log them.

## Build order

1. Tier 1 unit tests first. Nothing else is trustworthy until the VLP engine passes.
2. Tier 2 and Tier 4 steady-state cases. This completes Stage 1.
3. Tier 3 Macondo as the Stage 1 capstone.
4. Tier 5 when you start the transient solver.

## Cautions

- Keep this library built only from public sources. Don't add anything from Halliburton files or a Halliburton-licensed Drillbench seat.
- † items need their paper numbers confirmed before you cite them anywhere.
- Digitizing curves from papers for your own validation is normal practice. Redistributing the PDFs is not: keep `sources/` out of any public repo.
