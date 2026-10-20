# Wellsim Stage 1 Spec: Blowout Rate + Dynamic Kill (Steady-State)

Draft v0.1, 21 Sep 2026. Personal-use tool. Public sources only.

## 1. Scope

**Stage 1a: blowout rate.** Steady-state nodal analysis for a well flowing uncontrolled to surface. Black oil and dry gas.

**Stage 1b: dynamic kill.** Steady-state minimum kill rate, kill fluid density trade-off, pump pressure and horsepower, for kill fluid injected at a depth node.

**Stage 1c: batch and Monte Carlo.** After 1a and 1b pass single-case validation.

**Not in Stage 1:** transients of any kind (unloading, kill volume vs. time, kick migration), compositional PVT, gas solubility in OBM, subsea release, multi-layer reservoirs, crossflow / underground blowout, restrictions at BOP or wellhead (placeholder only).

### Design decisions that keep later stages cheap

- **Exit pressure is an input, default 14.7 psia.** Subsea release later becomes a one-line change (seawater hydrostatic at mudline), not a rewrite.
- **Kill is modeled as an injection node at a depth, not as "relief well" or "drillstring."** On the blowout side the hydraulics are identical: kill fluid enters at depth X and mixes with the flow above it. Relief well vs. drillstring only changes the injection path used to compute pump pressure. One kill engine, swappable injection paths.
- **PVT sits behind an interface.** Black oil now, NeqSim compositional later, no changes to VLP code.
- **The solver is a pure function:** `solve(case) -> result`, no global state, no UI dependencies. That's what makes Monte Carlo trivial in 1c.

## 2. Architecture

```
wellsim/
  core/
    units.py        oilfield units internally; convert only at I/O boundary
    pvt/            base.py (interface), black_oil.py, dry_gas.py
    ipr/            oil.py, gas.py
    geometry.py     well path, flow path segments, hydraulic diameters
    temperature.py  flowing temperature profile
    vlp/            march.py (integrator), correlations/*.py
    exit.py         critical flow check
    nodal.py        operating point solver
    kill.py         dynamic kill engine
    injection.py    injection path friction (relief well, drillstring)
  cases/            schema.py (pydantic), loader for YAML
  batch/            sweep.py, montecarlo.py            (Stage 1c)
  validation/       per validation library layout
  ui/               Streamlit app (thin layer, calls core only)
  tests/            pytest
```

Stack: Python 3.11+, numpy, scipy, pydantic, PyYAML, pytest, matplotlib/plotly, Streamlit. No unit-handling library inside the solver loops (too slow for Monte Carlo); units handled at the schema boundary.

## 3. Case inputs

**Reservoir:** pressure, temperature, net pay, permeability, skin, drainage radius, wellbore radius, porosity, total compressibility, exposed interval (partial penetration), time since blowout start (for transient IPR option).

**Fluids**
- Black oil: oil API, gas SG, GOR, water cut, water salinity. Bubble point either given or calculated.
- Dry gas: gas SG, N2/CO2/H2S fractions, condensate yield = 0 (wet gas is a later add).

**Well:** survey (MD/TVD or vertical), casing/liner/open hole IDs by depth, string in hole (ODs/IDs, depth) or none, roughness per section.

**Flow path (pick one per case):** open hole/casing with no string, annulus, inside drillstring, tubing. Combined annulus + string flow is deferred.

**Exit:** ambient pressure (default 14.7 psia), exit ID.

**Kill (1b):** injection depth, kill fluid density, rheology (Newtonian first; power law and Bingham next), injection path geometry, pump pressure limit, available HHP.

## 4. Physics

### 4.1 PVT

| Property | Default | Alternate |
|---|---|---|
| Rs, Pb | Standing | Vasquez-Beggs |
| Bo | Standing | Vasquez-Beggs |
| Oil viscosity | Beggs-Robinson (dead + live), Vasquez-Beggs above Pb | |
| Gas Z | Dranchuk-Abou-Kassem | Hall-Yarborough |
| Pseudo-criticals | Sutton, with Wichert-Aziz sour correction | |
| Gas viscosity | Lee-Gonzalez-Eakin | |
| Water | McCain | |
| Surface tension | Baker-Swerdloff | |

Unit tests: each correlation against its published example or chart values.

### 4.2 IPR

- **Oil:** Darcy pseudo-steady-state above Pb, Vogel below (composite IPR). Optional infinite-acting transient rate vs. time.
- **Gas:** pseudo-pressure formulation with **non-Darcy (rate-dependent) skin.** At blowout rates turbulence in the near-wellbore region is large and ignoring it overpredicts rate badly. Beta from a published correlation, user-overridable.
- Partial penetration skin for the top-of-reservoir scenario.

### 4.3 VLP

- Marching integrator, bottom-up or top-down, adaptive segment length (default max 100 ft, refine where gradient changes fast), pressure and temperature dependent properties each segment.
- **Acceleration term included.** Most production-engineering implementations drop it. At blowout velocities near the exit it matters.
- Annular flow via hydraulic diameter to start. Flag as a known weak point.
- Correlations, in build order:
  1. Single-phase gas (general marching, check against Cullender-Smith)
  2. Hagedorn-Brown with Griffith bubble correction
  3. Beggs-Brill
  4. Gray
  5. Duns-Ros
  6. Large-diameter model from the LSU/BOEM 2018 report (Teles-Waltrich)
- Results always report which correlation was used, and the UI can overlay several. Spread between correlations is a free uncertainty indicator.

### 4.4 Temperature

Start simple: linear profile between reservoir temperature and a user-set flowing exit temperature (default: reservoir temperature minus a small loss, since high-rate flow loses little heat). Upgrade path: Hasan-Kabir steady-state heat transfer. Joule-Thomson cooling for gas near the exit is a Stage 1 stretch item.

### 4.5 Exit condition

Compute mixture sonic velocity at the exit (gas: real-gas sonic; two-phase: homogeneous/Wood's equation). If flow would exceed Mach 1 at ambient pressure, the exit is choked: exit pressure floats above ambient until Mach = 1. This is what governs most surface gas blowouts. Two-phase sonic velocity is a known soft spot; log it as an uncertainty.

### 4.6 Nodal solver

Find q where IPR Pwf equals VLP Pwf. Brent's method on a bracketed interval, with a scan first to detect multiple intersections. Report unstable (left-hand) intersections rather than silently picking one.

### 4.7 Dynamic kill (1b)

Steady-state criterion (Blount-Soeiinah / Kouba approach): for a given kill fluid and injection rate, build the system intake curve with kill fluid mixed in above the injection node. **The well is dynamically killed when that curve sits above the IPR at every formation rate greater than zero**, meaning no operating point with inflow exists.

Outputs from a sweep: minimum kill rate vs. kill fluid density, pump pressure and HHP for each (from injection path friction plus hydrostatics), and a check of whether the kill fluid is statically overbalanced once flow stops. If not, the result says so: dynamic kill holds only while pumping and a weighted follow-up is required.

Kill fluid treated as an added liquid phase, mixture liquid properties by volume fraction. Newtonian friction first; power law and Bingham after the water cases validate.

**Known limit:** steady-state can't give kill volume or time. It gives the rate and pressure needed, which sizes the spread. Volume comes with the transient solver in Stage 2.

## 5. Outputs

- Blowout rate: oil, gas, water at standard conditions; flowing BHP; drawdown
- Profiles vs. depth: pressure, temperature, holdup, mixture velocity, flow regime, Mach number
- Exit state: choked or not, exit pressure, exit velocity
- Nodal plot: IPR plus VLP curves for each correlation
- Kill: kill-rate vs. density curve, pump pressure, HHP, static overbalance check, kill/no-kill against the available spread
- Every result carries its inputs, correlation choices, code version, and warnings. Reproducibility from day one.

## 6. Numerical requirements

- VLP pressure convergence 0.1 psi per segment; nodal rate convergence 0.1%
- Single case under 1 second on a laptop (needed for Monte Carlo). Vectorize or use numba only if profiling says so.
- No silent extrapolation: correlations used outside their published range raise a warning in the result.

## 7. Validation gates

Tied to the validation library. No milestone closes without its gate.

| Gate | Test | Target |
|---|---|---|
| G1 | PVT and correlation worked examples | within 2% |
| G2 | LSU/BOEM large-diameter data | match LSU's reported error for the same correlation |
| G3 | Published blowout rate cases (Oudeman, Norwegian studies) | within 15% |
| G4 | Historical dynamic kills (Upchurch table, Arun, Ixtoc) | min kill rate within 20%; correct kill/no-kill call |
| G5 | Macondo | inside 50–70k bopd band (needs subsea exit pressure, so this lands just after Stage 1) |

Targets are proposals. Revise once we see what the data supports.

## 8. Milestones

- **M0:** repo, case schema, units, test harness, CI on the homelab
- **M1:** PVT complete, G1 for PVT
- **M2:** single-phase gas VLP + gas IPR + exit choke → dry gas blowout rate end to end
- **M3:** multiphase correlations, G1 and G2
- **M4:** oil IPR + nodal solver → black oil blowout rate, G3
- **M5:** kill engine + injection paths, G4
- **M6:** Streamlit UI
- **M7 (1c):** sweeps, tornado charts, Monte Carlo with P10/P50/P90 rates and kill requirements

M2 gets a working end-to-end result early with the simplest physics, which shakes out the architecture before the hard correlations go in.

## 9. Open questions

1. Annulus flow: hydraulic diameter is crude for multiphase. Accept for Stage 1, or find an annulus-specific model (Caetano)?
2. Which non-Darcy beta correlation as default?
3. Is wet gas / condensate needed sooner than compositional? A lot of real gas blowouts aren't dry.
4. Restrictions (partly closed BOP, parted string) as a choke model: Stage 1 stretch or Stage 2?
5. Employment agreement IP clause: still unchecked.
