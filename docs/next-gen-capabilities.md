# wellsim Next-Gen Drilling & Wild Well Capabilities

This document provides a comprehensive reference for the advanced physics modules, petroleum engineering standards, real-time telemetry systems, and digital twin engines implemented in `wellsim`.

---

## 1. Non-Newtonian Rheology & HPHT Scaling (`wellsim.core.rheology`)
- **Standard**: API RP 13D (Recommended Practice on Rheology and Hydraulics of Oil-Well Drilling Fluids).
- **Formulation**: Herschel-Bulkley (Yield-Power-Law / YPL) 3-parameter fluid model:
  $$\tau = \tau_0 + K \dot{\gamma}^n$$
- **Parameter Fitting**: Least-squares fitting from 6-speed Fann rotational viscometer dial readings (600, 300, 200, 100, 6, 3 RPM).
- **HPHT Downhole Viscosity Correction**: Coupled temperature-thinning and piezo-thickening scaling based on Fann 75/77 laboratory data:
  $$\mu(P, T) = \mu_0 \exp\left[\alpha_p (P - P_0) - \beta_t (T - T_0)\right]$$

---

## 2. Non-Aqueous OBM/SBM Gas Solubility & Flash Exsolution (`wellsim.core.obm_solubility`)
- **Physics**: In oil-based mud (OBM) and synthetic-based mud (SBM), methane dissolves completely into the base oil downhole when pressure exceeds the bubble point ($P > P_b$), yielding zero free gas and no pit gain during early kick circulation.
- **Formulation**: Vasquez-Beggs & Thomas-O'Leary correlation for solution gas-oil ratio ($R_s$) and bubble point ($P_b$).
- **Near-Surface Flash**: As fluid rises above $P_b$ near surface, methane rapidly desorbs and flashes into free gas, causing exponential annular expansion and potential riser unloading.

---

## 3. Burkhardt & Mitchell Dynamic Surge and Swab Tripping (`wellsim.core.surge_swab`)
- **Formulation**: Burkhardt (1961) & Mitchell (1988) laminar/turbulent pipe tripping displacement hydraulics.
- **Clinging Factor**: $K_{cling}$ computed for open-ended pipe (fluid displacement inside and outside) and closed-ended float valve pipe (full fluid displacement).
- **Operational Envelope**: Automated bisection root-finding determining maximum safe running-in (RIH) and pulling-out (POOH) velocity profiles (ft/min) along the entire wellbore to prevent casing shoe fracture breakdown or swab-induced reservoir kicks.

---

## 4. Managed Pressure Drilling (MPD) Suite (`wellsim.core.mpd`)
- **Micro-Flux Detection**: Coriolis mass-flow meter differential mass-balance tracking flow-in vs. flow-out with micro-threshold detection (0.5 to 1.5 bbl).
- **Constant Bottomhole Pressure (CBHP)**: Automated PID choke feedback loop dynamically regulating surface backpressure (SBP) to maintain target downhole pressure during pump ramp-ups, connections, and shutdowns.
- **Deepwater Subsea Choke Line Friction**: Accurate hydrodynamic modeling of narrow subsea choke lines (3" ID over 3,000–10,000 ft water depth) creating substantial backpressure on the wellbore during kick circulation.

---

## 5. Cuttings Transport & Hole Cleaning (`wellsim.core.cuttings_transport`)
- **Standard**: Larsen et al. (1997) & API RP 13D 3-layer annular flow model.
- **Inclination Dependency**: Models particle slip velocity and critical transport velocity across the full $0^\circ \text{--} 90^\circ$ window, specifically capturing the hazardous avalanching dunes regime between $40^\circ \text{ and } 55^\circ$.
- **Pack-Off Screening**: Evaluates stationary cuttings bed height, annular choking index, and differential pack-off risk index.

---

## 6. Directional Survey & 3D Spatial Trajectory (`wellsim.core.survey`)
- **Standard**: API RP 78 / SPE Minimum Curvature Method.
- **Spatial Coordinates**: Maps directional survey stations $(MD, Inc, Azm)$ into 3D Cartesian coordinates $(TVD, North, East, DLS)$ with ratio factor $RF = \frac{2}{\beta} \tan\frac{\beta}{2}$.
- **Dogleg Severity**: Exact 3D dogleg calculation ($\beta$) scaled to degrees per 100 feet.

---

## 7. 3D Torque & Drag and Drillstring Buckling (`wellsim.core.torque_drag`)
- **Formulation**: Johancsik (1984) soft-string spatial integration of normal contact forces, axial tension/compression, and torsional friction.
- **Operating Modes**: Rotary drilling, slack-off (RIH), and pick-up (POOH) hookload schedules.
- **Buckling Limits**: Dawson-Paslay sinusoidal buckling threshold and Wu/Cheatham helical buckling and lock-up limits for extended-reach drilling (ERD).

---

## 8. Dynamic Wellbore Heat Transfer (`wellsim.core.hasan_kabir`)
- **Formulation**: Hasan & Kabir (1994) / Ramey conjugate convection-conduction engine.
- **Counter-Current Hydraulics**: Solves the coupled differential temperature equations for fluid flowing down the drillpipe and ascending the annulus with relaxation length parameter $B$:
  $$\frac{dT_p}{dz} = \frac{1}{B} (T_p - T_a)$$
  $$\frac{dT_a}{dz} = \frac{1}{B} (T_a - T_p) + \frac{1}{A} (T_a - T_{ei})$$

---

## 9. API Spec 5CT & TR 5C3 Casing Structural Integrity (`wellsim.core.casing_integrity`)
- **Standard**: API Spec 5CT 10th Edition & API TR 5C3 7th Edition.
- **Burst Strength**: Barlow equation with 87.5% manufacturing wall tolerance:
  $$P_{burst} = 0.875 \left(\frac{2 Y_p t}{D}\right)$$
- **Collapse Strength**: Complete multi-regime empirical envelope:
  - Yield collapse ($D/t \le (D/t)_{yp}$)
  - Plastic collapse ($(D/t)_{yp} < D/t \le (D/t)_{pt}$)
  - Transition collapse ($(D/t)_{pt} < D/t \le (D/t)_{te}$)
  - Euler elastic collapse ($D/t > (D/t)_{te}$)
- **Dynamic Screening**: Depth-stepping safety factor evaluation ($SF_{burst} \ge 1.10$, $SF_{collapse} \ge 1.00$) under gas kicks and complete casing evacuation.

---

## 10. Wellbore Ballooning / Breathing vs. Kick Discriminator (`wellsim.core.ballooning`)
- **Physics**: When pumps are running at dynamic ECD, bottomhole pressure exceeds minimum horizontal stress ($\sigma_{min}$), elastically dilating micro-fractures. Upon pump shutdown ($ECD \to ESD$), stored fluid returns to the wellbore ($q > 0$).
- **Discriminator Engine**: Analyzes flowback rate deceleration ($dq/dt < 0$ vs. steady $q_{kick}$) and asymptotic gain convergence ($V_{cum} \to V_{inf}$) to definitively prevent false well shut-ins or hazardous underbalanced influxes.

---

## 11. Deepwater Riser Gas Handling & Diverter Dynamics (`wellsim.core.riser_gas`)
- **Standard**: API RP 16Q & API Standard 53.
- **Riser Dynamics**: Models Taylor bubble slip velocity ($0.35 \sqrt{g D}$), catastrophic Boyle real-gas expansion near surface, liquid mud ejection limits, and seabed hydrostatic collapse risk.
- **Automated Routing**: Mud-Gas Separator (Poor Boy) liquid-seal blow-through screening and automated overboard diverter line backpressure evaluation.

---

## 12. Dual-Gradient Drilling (DGD) & Subsea MudLift Pump (`wellsim.core.dual_gradient`)
- **Physics**: Decouples water column hydrostatic by maintaining seawater (8.55 ppg) in the marine riser and heavy mud in the wellbore below mudline.
- **Subsea MudLift Pump (MLP)**: Regulates mudline pressure at seawater hydrostatic ($P_{MLP} = 0.052 \cdot \rho_{sw} \cdot WD$), sizes Mud Return Line (MRL) lift pressure and HHP, and resolves drillpipe-to-annulus U-tube imbalance and Drillstring Valve (DSV) cracking pressure.

---

## 13. Managed Pressure Cementing & API RP 65-2 Gas Migration (`wellsim.core.cementing`)
- **Standard**: API RP 65-2 (Isolating Potential Flow Zones During Well Construction).
- **Transient Free Fall**: Models U-tubing during slurry displacement, casing vacuum, return rate surges, and annular ECD vs. formation breakdown.
- **Static Gel Strength (SGS) Transition**: Evaluates hydration decay from $100 \text{ to } 500\text{ lbf/100 ft}^2$, identifies underbalanced gas migration windows, and sizes Managed Pressure Cementing surface backpressure.

---

## 14. Real-Time Unscented Kalman Filter (UKF) Digital Twin (`wellsim.core.kalman`)
- **Formulation**: Pure Python deterministic Sigma-Point propagation (Julier & Uhlmann, 2004) with analytic Cholesky factorization and Gauss-Jordan matrix inversion.
- **State Estimation**: Assimilates noisy surface rig telemetry (SPP, Casing Pressure, Pit Level, Flow Paddle) to dynamically infer downhole formation pore pressure ($P_{pore}$), kick volume ($V_{kick}$), and friction factor multipliers with 95% confidence bounds.

---

## 15. Telemetry & Web Infrastructure (`wellsim.core.wits`, `wellsim.core.witsml`, `wellsim.ui`)
- **WITS Level 0**: ASCII telemetry stream encoder/decoder for real-time rig cyberbase consoles.
- **WITSML 1.4.1.1 XML**: Schema-compliant XML serialization/deserialization for `<trajectorys>` and `<wbGeometrys>`.
- **Zero-Dependency Web Console**: HTML5 Canvas multi-fluid wellbore tracking, 3D orbiting trajectory viewer, real-time strip charts, PID choke control, and automated IADC WellSharp kill sheet grading.
