"""Gate G2: the LSU PERTT large-diameter air-water runs.

Provenance: every measured value comes from LSU Final Report Table A.1 (PDF pp. 155-158),
extracted to docs/validation/data/lsu_table_a1.csv, tagged FETCHED. The air ideal-gas
forms and the Beggs-Brill cross-check are MEMORY. The pass criterion cites LSU's own
finding (BOEM M15PC00007 p.59) and is the only assertion on the numbers; everything else
is reported to docs/validation/G2-results.md, not asserted, until standing bars are set.
"""

from __future__ import annotations

import math

import pytest

from wellsim.core.pvt.air import AirIdealGas, K_AIR, R_AIR
from wellsim.validation import g2_lsu as g2
from wellsim.tests.test_holdup_crosscheck import beggs_brill_holdup

PASS_BAR_SLIP_BELOW_1 = 0.10


# --- the air path ---------------------------------------------------------------------


def test_air_ideal_gas_properties_and_the_dak_refusal(capsys):
    """MEMORY forms, checked against their own definitions and a hand value."""
    air = AirIdealGas()
    # 14.7 psia, 70 degF: rho = 144 p / (R T)
    rho = air.gas_density(14.7, 70.0)
    assert rho == pytest.approx(144.0 * 14.7 / (R_AIR * 529.67), rel=1e-12)
    assert rho == pytest.approx(0.0749, abs=0.0005)  # standard air density
    assert air.z(14.7, 70.0) == 1.0
    assert air.k_ratio() == K_AIR == 1.4
    # sound speed c = sqrt(k g_c R T): about 1,128 ft/s at 70 degF
    assert air.sound_speed(70.0) == pytest.approx(1128.0, abs=3.0)
    # Sutherland: about 0.0181 cp at 70 degF, rising with temperature
    mu = air.gas_viscosity(14.7, 70.0)
    assert mu == pytest.approx(0.0181, abs=0.0004)
    assert air.gas_viscosity(14.7, 300.0) > mu
    # ideal gas: density doubles when pressure doubles at fixed temperature
    assert air.gas_density(29.4, 70.0) == pytest.approx(2.0 * rho, rel=1e-12)

    # DAK is refused outright, not quietly approximated
    with pytest.raises(NotImplementedError, match="refused for air"):
        air.dak_z(14.7, 70.0)

    print(f"\nair at 14.7 psia, 70 degF: rho {rho:.5f} lbm/ft3, mu {mu:.5f} cp, "
          f"c {air.sound_speed(70.0):.1f} ft/s, k {K_AIR}")


def test_air_warns_above_its_stated_pressure_ceiling():
    """The ideal-gas assumption is good here but is not silent about where it stops."""
    from wellsim.core.pvt.base import RangeWarning

    air = AirIdealGas()
    with pytest.warns(RangeWarning):
        air.gas_density(500.0, 70.0)
    # and not below it
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        air.gas_density(14.7, 70.0)



def test_air_thermo_interface_is_closed_form_and_self_consistent(capsys):
    """g2-4: the march interface on AirIdealGas, checked against thermodynamics, not
    against a fit. Every claim here is COMPUTED from the ideal-gas relations; no
    correlation is involved, so there is nothing to tag MEMORY except the measured cp
    comparison, which is reported and not asserted."""
    from wellsim.core.pvt.real_gas_thermo import R

    air = AirIdealGas()
    p, t = 14.7, 70.0
    t_r = t + 459.67
    st = air.state(p, t)
    th = air.thermo(p, t, st)

    # a. molar mass, and the identity that made gas_sg = 1.0 numerically right before
    assert air.molar_mass == pytest.approx(28.9647, rel=1e-12)

    # b. Z is exactly 1 and the reduced coordinates are ABSENT, not guessed
    assert st.z == 1.0
    assert st.tpr is None and st.ppr is None
    assert st.dz_drho_r is None and st.dz_dtpr is None

    # c. cp/cv = K_AIR exactly, so the thermo and the sound speed cannot disagree
    assert th.cp / th.cv == pytest.approx(K_AIR, rel=1e-12)
    assert th.cp == pytest.approx(3.5 * R, rel=1e-12)

    # d. the departures are identically zero, by definition
    assert th.h_dep == 0.0
    assert th.mu_jt == 0.0

    # e. the EOS derivatives against central differences on gas_density. rho is linear in
    # p, so that one is exact; it goes as 1/T, so the T difference carries a truncation
    # error of order (h/T)^2 and is checked to 1e-7 with h/T = 1e-4, not tighter.
    h_p, h_t = 1e-4 * p, 1e-4 * t_r
    rho_p_fd = (air.gas_density(p + h_p, t) - air.gas_density(p - h_p, t)) / (2 * h_p)
    rho_t_fd = (air.gas_density(t=t + h_t, p=p) - air.gas_density(t=t - h_t, p=p)) / (2 * h_t)
    assert th.rho_p == pytest.approx(rho_p_fd, rel=1e-12)
    assert th.rho_t == pytest.approx(rho_t_fd, rel=1e-7)
    assert st.dp_drho_t == pytest.approx(1.0 / rho_p_fd, rel=1e-12)

    # f. c_s from the state derivative reproduces sqrt(k g_c R_air T)
    c_from_state = math.sqrt(K_AIR * 32.174 * 144.0 * st.dp_drho_t)
    assert th.c_s == pytest.approx(c_from_state, rel=1e-12)

    # g. enthalpy is linear in T with slope cp, the ideal-gas statement
    th2 = air.thermo(p, t + 100.0)
    assert (th2.h - th.h) / 100.0 == pytest.approx(th.cp, rel=1e-12)

    print(f"air at {p} psia, {t} degF: Z {st.z}, rho {st.rho_g:.5f} lbm/ft3, "
          f"mu {st.mu_g:.5f} cp")
    print(f"  cp {th.cp:.4f} cv {th.cv:.4f} Btu/(lbmol degR), cp/cv {th.cp/th.cv:.6f}")
    print(f"  measured air cp near 6.95 Btu/(lbmol degR) at 70 degF (MEMORY, reported "
          f"only): 7/2 R gives {th.cp:.4f}")
    print(f"  c_s {th.c_s:.1f} ft/s, mu_jt {th.mu_jt} degF/psi, h_dep {th.h_dep} Btu/lbmol")


def test_air_marches_the_real_gradient_that_dak_cannot_reach(capsys):
    """The point of the interface: two_phase_gradient runs on air, which it could not do
    before, and what comes back is the march gradient, slip momentum flux and all.

    The point is LSU row 1 (0.10 m, usg 0.140, usl 0.134 m/s), so the number printed can
    be read against the gate. COMPUTED; nothing here is asserted against Table A.1."""
    from wellsim.core.geometry import FlowSegment
    from wellsim.core.vlp.two_phase import LiquidProps, two_phase_gradient

    air = AirIdealGas()
    water = LiquidProps(rho_l=62.4, mu_l=1.0, cp_l=1.0, sigma=72.0, c_l=4900.0)
    d_in = 0.10 / 0.0254
    area = math.pi * (d_in / 12.0) ** 2 / 4.0
    seg = FlowSegment(top_md=0.0, bottom_md=10.0, d_hyd=d_in, area=area,
                      roughness=6e-4, sin_theta=1.0, conduit="LSU 0.10 m test section")

    p_psia, t_degf = 14.7, 70.0
    v_sg, v_sl = 0.140 / 0.3048, 0.134 / 0.3048       # m/s to ft/s
    w_g = air.gas_density(p_psia, t_degf) * v_sg * area
    w_l = water.rho_l * v_sl * area
    g = two_phase_gradient(air, water, p_psia, t_degf, w_g, w_l, seg)

    # a real gradient with a real determinant, not a homogeneous stand-in
    assert 0.0 < g.alpha_g < 1.0
    assert g.det > 0.0
    assert math.isfinite(g.dp_ds) and math.isfinite(g.dt_ds)
    # the mixture density must sit between the phases, and near the liquid at this holdup
    assert air.gas_density(p_psia, t_degf) < g.rho < water.rho_l

    # and DAK is still refused for air, so this cannot have gone through the DAK path
    with pytest.raises(NotImplementedError):
        air.dak_z(14.7, 70.0)

    print(f"LSU row 1 through the march gradient: alpha_g {g.alpha_g:.4f}, "
          f"rho_m {g.rho:.3f} lbm/ft3")
    print(f"  dp/ds {g.dp_ds:.5f} psi/ft = {g.dp_ds * 22.6206:.3f} kPa/m, "
          f"det {g.det:.6f}, measured 7.759 kPa/m")


# --- the extracted data ---------------------------------------------------------------


def test_table_a1_loads_with_the_column_order_and_true_diameters(capsys):
    """The CSV is the gate's only input. Two things must hold or every number below is
    wrong: the velocity columns are u_sl then u_sg (the printed header has them the other
    way round, confirmed against the rendered page), and the diameters are the TRUE IDs
    from BOEM M15PC00007 p.30, not Table A.1's nominal 0.10/0.20/0.30 m."""
    rows = g2.load_rows()
    assert len(rows) == 131

    # the facility limits, LSU Final Report p.113: liquid to 4.35 m/s, gas to 30 m/s.
    # This is what pins the column order independently of the header.
    assert max(r.u_sl_m_s for r in rows) == pytest.approx(4.350, abs=0.001)
    assert max(r.u_sg_m_s for r in rows) == pytest.approx(29.669, abs=0.001)

    # true IDs, not nominal
    ids = sorted({r.id_in for r in rows})
    assert ids == pytest.approx([3.83, 7.825, 11.7])
    assert {r.nominal_id_m for r in rows} == {"0.10", "0.20", "0.30"}
    # and no 2 in rows: BOEM M15PC00007 p.36 reports that section as unreliable
    assert all(r.id_in > 3.0 for r in rows)

    assert sum(1 for r in rows if r.h_l is not None) == 93
    assert sum(1 for r in rows if r.obs_regime) == 42

    # unit conversion happens at load, not in the file
    r1 = rows[0]
    assert r1.v_sg == pytest.approx(r1.u_sg_m_s / 0.3048, rel=1e-12)
    assert r1.dpdz_psi_ft == pytest.approx(r1.dpdz_kpa_m * 0.1450377 * 0.3048, rel=1e-12)

    print(f"\nTable A.1: {len(rows)} runs, u_sl max {max(r.u_sl_m_s for r in rows):.3f} m/s "
          f"(facility limit 4.35), u_sg max {max(r.u_sg_m_s for r in rows):.3f} (limit 30)")


def test_reported_gradients_are_total_not_frictional():
    """A cross-check on the basis question G2-lsu-boem.md logged. A total gradient in
    near-vertical water-rich flow must approach the water hydrostatic as holdup goes to 1
    and must FALL as gas rate rises; a frictional-only gradient would do neither."""
    rows = g2.load_rows()
    water_hydrostatic_psi_ft = g2.WATER_RHO_LBM_FT3 / 144.0

    high_holdup = [r for r in rows if r.h_l is not None and r.h_l > 0.95]
    assert high_holdup
    for r in high_holdup:
        assert 0.6 * water_hydrostatic_psi_ft < r.dpdz_psi_ft <= 1.05 * water_hydrostatic_psi_ft

    # within one liquid-rate block at 0.10 m, the gradient falls monotonically as gas rises
    block = [r for r in rows if r.nominal_id_m == "0.10" and 0.13 < r.u_sl_m_s < 0.19]
    block.sort(key=lambda r: r.u_sg_m_s)
    grads = [r.dpdz_psi_ft for r in block]
    assert grads == sorted(grads, reverse=True), grads


# --- the gate, fast subset ------------------------------------------------------------


def test_g2_fast_subset_spans_diameters_and_regimes(capsys):
    """The default-suite subset must actually span what it claims to, or it is not a
    regression net."""
    rows = {r.run: r for r in g2.load_rows()}
    subset = [rows[n] for n in g2.FAST_SUBSET_RUNS]
    assert len({r.nominal_id_m for r in subset}) == 3
    regimes = {r.obs_regime for r in subset if r.obs_regime}
    assert regimes == {"Bubbly", "Bubble", "Slug", "Slug/Churn", "Churn",
                       "Churn/Annular", "Annular"}, regimes


def test_g2_fast_subset_evaluates_and_meets_the_pass_criterion(capsys):
    """The pass criterion on the fast subset: on rows with v_sg/v_sl < 1 the mean absolute
    error on the gradient is under 10%, which is LSU's own reported finding (BOEM
    M15PC00007 p.59: "for usg/usl < 1, all models predict pressure gradient well")."""
    rows = {r.run: r for r in g2.load_rows()}
    subset = [rows[n] for n in g2.FAST_SUBSET_RUNS]
    points = g2.evaluate_all(subset)

    assert len(points) == len(g2.FAST_SUBSET_RUNS)
    for p in points:
        assert 0.0 <= p.alpha_g <= 1.0
        assert p.evaluable, p.note
        assert p.dpdz_pred_psi_ft > 0.0
        assert p.grav_psi_ft > 0.0 and p.fric_psi_ft >= 0.0

    low = [p for p in points if p.row.slip_ratio < 1.0]
    assert low, "the subset has no low-slip rows to test the criterion on"
    mae = sum(abs(p.gradient_error) for p in low) / len(low)
    assert mae < PASS_BAR_SLIP_BELOW_1, f"low-slip gradient MAE {mae:.2%}"

    print("\nG2 fast subset (12 runs):")
    print("   run   ID   u_sl   u_sg   H_l rep/pred   dp/dz rep/pred psi/ft   err")
    print("  " + "-" * 70)
    for p in points:
        r = p.row
        hl = f"{r.h_l:.2f}/{p.h_l_pred:.2f}" if r.h_l is not None else f"  - /{p.h_l_pred:.2f}"
        print(f"  {r.run:4d} {r.nominal_id_m} {r.u_sl_m_s:6.3f} {r.u_sg_m_s:7.3f}  {hl:>11s}"
              f"   {r.dpdz_psi_ft:.4f}/{p.dpdz_pred_psi_ft:.4f}   {p.gradient_error:+7.1%}")
    print(f"  low-slip (v_sg/v_sl < 1) rows {len(low)}, gradient MAE {mae:.2%}, "
          f"bar {PASS_BAR_SLIP_BELOW_1:.0%}")


# --- the gate, full run ---------------------------------------------------------------


@pytest.mark.slow
def test_g2_full_run_meets_the_pass_criterion_and_reports_the_rest(capsys):
    """All 131 rows. The ONLY assertion is the low-slip gradient criterion; the holdup
    errors, the per-diameter and per-regime splits, the Beggs-Brill cross-check and the
    regime confusion are printed and written to docs/validation/G2-results.md, to be read
    before any standing bar is set."""
    rows = g2.load_rows()
    points = g2.evaluate_all(rows)
    s = g2.summarise(points)

    low = s["slip_below_1"]
    assert low["n_gradient"] > 0
    assert low["gradient_mae"] < PASS_BAR_SLIP_BELOW_1, (
        f"low-slip gradient MAE {low['gradient_mae']:.2%} against the {PASS_BAR_SLIP_BELOW_1:.0%} "
        "bar from BOEM M15PC00007 p.59"
    )

    # structural, not a bar on accuracy: every row must produce a number or say why not
    for p in points:
        assert 0.0 <= p.alpha_g <= 1.0
        assert p.evaluable or "det" in p.note

    def bb_fn(v_sg, v_sl, rho_g, rho_l, mu_l, sigma, d_o, d_i=0.0, theta=90.0):
        # g2-4: the march gradient also wants dalpha_g/drho_g, which it uses to build the
        # SLIP momentum flux. Beggs-Brill is a correlation with no analytic sensitivity, so
        # it is differenced numerically here rather than passed as zero: passing zero would
        # silently give Beggs-Brill a homogeneous acceleration term and make the
        # cross-check compare two different gradients.
        h_l, regime = beggs_brill_holdup(v_sg, v_sl, rho_l, sigma, d_o, theta)
        d = 1e-4 * rho_g
        v_lo, v_hi = v_sg * rho_g / (rho_g - d), v_sg * rho_g / (rho_g + d)
        a_lo = 1.0 - beggs_brill_holdup(v_lo, v_sl, rho_l, sigma, d_o, theta)[0]
        a_hi = 1.0 - beggs_brill_holdup(v_hi, v_sl, rho_l, sigma, d_o, theta)[0]
        return type("BB", (), {"alpha_g": 1.0 - h_l, "regime": regime,
                               "dalpha_drho_g": (a_hi - a_lo) / (2.0 * d),
                               "iterations": 0})()

    s_bb = g2.summarise(g2.evaluate_all(rows, holdup_fn=bb_fn))
    sens = g2.pressure_sensitivity(rows)

    def fmt(x):
        return "n/a" if x is None else f"{x:.2%}"

    a = s["all"]
    print(f"\nG2 full run, {len(rows)} rows, {a['n_holdup']} with holdup")
    print(f"  drift flux   holdup MAE {fmt(a['holdup_mae'])}, within 20% "
          f"{fmt(a['holdup_within_20pct'])}; gradient MAE {fmt(a['gradient_mae'])}")
    print(f"  Beggs-Brill  holdup MAE {fmt(s_bb['all']['holdup_mae'])}, within 20% "
          f"{fmt(s_bb['all']['holdup_within_20pct'])}; gradient MAE "
          f"{fmt(s_bb['all']['gradient_mae'])}   (cross-check only)")
    print("  by diameter (nominal m):    n  holdup MAE  grad MAE")
    for k in sorted(s["by_diameter"]):
        b = s["by_diameter"][k]
        print(f"      {k}  {b['n']:4d}   {fmt(b['holdup_mae']):>9s}  {fmt(b['gradient_mae']):>9s}")
    print("  by reported regime:")
    for k in sorted(s["by_observed_regime"]):
        b = s["by_observed_regime"][k]
        print(f"      {k:14s} {b['n']:3d}   {fmt(b['holdup_mae']):>9s}  {fmt(b['gradient_mae']):>9s}")
    conf = g2.regime_confusion(points)
    exact = sum(v for (o, p), v in conf.items() if o == p)
    print(f"  regime: {sum(conf.values())} labelled rows, {exact} exact name matches")
    print(f"  pressure sensitivity 14.7 -> 16.0 psia: gradient moves "
          f"{fmt(sens['mean_relative_move'])} on average, {fmt(sens['max_relative_move'])} at most")
    print(f"  PASS: low-slip gradient MAE {fmt(low['gradient_mae'])} over {low['n_gradient']} "
          f"rows, bar {PASS_BAR_SLIP_BELOW_1:.0%}")


@pytest.mark.slow
def test_pressure_assumption_sensitivity_is_bounded(capsys):
    """The 14.7 psia assumption is the gate's biggest free choice. Recorded once: if it
    were 16.0 psia instead, how much does the answer move?"""
    rows = g2.load_rows()
    sens = g2.pressure_sensitivity(rows, 14.7, 16.0)
    assert sens["n_compared"] > 100
    assert abs(sens["mean_relative_move"]) < 0.05, sens
    print(f"\n14.7 -> 16.0 psia: mean gradient move {sens['mean_relative_move']:+.2%}, "
          f"max {sens['max_relative_move']:+.2%}; gradient MAE "
          f"{sens['gradient_mae_at_p_lo']:.2%} -> {sens['gradient_mae_at_p_hi']:.2%}")


# --- step 2b: the data consistency screen ---------------------------------------------


def test_row_1_holdup_is_impossible_not_merely_odd(capsys):
    """Row 1 is where the screen starts. Its reported H_l of 0.99 contradicts its own
    reported gradient, and an independent kinematic check settles which column is wrong:
    0.99 would need the gas to move through the pipe 23 times faster than drift flux
    allows. The gradient is ordinary."""
    rows = g2.load_rows()
    r1 = next(r for r in rows if r.run == 1)
    assert (r1.nominal_id_m, r1.u_sl_m_s, r1.u_sg_m_s, r1.dpdz_kpa_m, r1.h_l, r1.obs_regime) \
        == ("0.10", 0.140, 0.134, 7.759, 0.99, "Bubbly")

    c1 = next(c for c in g2.consistency(rows) if c.row.run == 1)
    # the reported holdup implies a hydrostatic gradient the row does not report
    implied_grad_kpa_m = 0.99 * g2.WATER_RHO_LBM_FT3 / 144.0 / (0.1450377 * 0.3048)
    assert implied_grad_kpa_m == pytest.approx(9.69, abs=0.02)
    assert r1.dpdz_kpa_m == 7.759
    assert c1.h_l_implied == pytest.approx(0.792, abs=0.005)
    assert c1.difference == pytest.approx(-0.198, abs=0.005)
    # and it passes the screen ONLY because that section's tolerance is +/-0.19
    assert c1.within_uncertainty
    assert c1.tol_from_h_l == 0.19

    # the kinematic check, which uses neither the gradient nor any pressure assumption
    ratio = g2.implied_gas_velocity_ratio(r1)
    assert ratio > 20.0, ratio
    ratios = [g2.implied_gas_velocity_ratio(r) for r in rows if r.h_l is not None]
    assert sum(1 for x in ratios if x and x > 2.0) == 2  # only run 1 and run 126

    print(f"\nrow 1: reported H_l 0.99 implies {implied_grad_kpa_m:.2f} kPa/m against the "
          f"{r1.dpdz_kpa_m} kPa/m it reports")
    print(f"  gradient implies H_l {c1.h_l_implied:.3f}; difference {c1.difference:+.3f} "
          f"against a tolerance of {c1.tol_h_l:.3f}")
    print(f"  the reported holdup needs {ratio:.1f}x the drift-flux in-situ gas velocity")


def test_the_two_columns_agree_at_low_mixture_velocity_and_diverge_at_high(capsys):
    """The verdict is (iii), not (ii). Below 5 ft/s the two reported columns agree about
    as well as the instruments allow, in every section. The divergence appears at HIGH
    mixture velocity and changes sign with pipe size, which no single physical effect
    would do."""
    cons = g2.consistency(g2.load_rows())

    def mean_diff(nid, lo, hi):
        sel = [c for c in cons if c.row.nominal_id_m == nid and lo <= c.v_m < hi]
        return (sum(c.difference for c in sel) / len(sel), len(sel)) if sel else (None, 0)

    low = {nid: mean_diff(nid, 2.0, 5.0) for nid in ("0.10", "0.20", "0.30")}
    high = {nid: mean_diff(nid, 15.0, 1e9) for nid in ("0.10", "0.20", "0.30")}

    # low velocity: all three sections agree to within 0.05 of holdup
    for nid, (m, n) in low.items():
        assert n > 0 and abs(m) < 0.05, (nid, m, n)
    # high velocity: all three are further out than they were at low velocity
    for nid in ("0.10", "0.20", "0.30"):
        assert abs(high[nid][0]) > abs(low[nid][0]), (nid, low[nid], high[nid])
    # and the sign flips between the small PVC section and the two larger ones
    assert high["0.10"][0] > 0.0 and high["0.20"][0] < 0.0 and high["0.30"][0] < 0.0

    print("\nmean (implied - reported) holdup, by section and mixture velocity:")
    print("    ID     2-5 ft/s        >15 ft/s")
    for nid in ("0.10", "0.20", "0.30"):
        print(f"    {nid}  {low[nid][0]:+7.3f} (n={low[nid][1]:2d})  "
              f"{high[nid][0]:+7.3f} (n={high[nid][1]:2d})")


def test_the_screen_excludes_the_contradictory_rows_and_gradient_is_untouched(capsys):
    """Gradient is scored on everything; holdup only on the rows whose two reported
    columns agree inside the run's own uncertainty."""
    rows = g2.load_rows()
    points = g2.evaluate_all(rows)
    keep = g2.screened_rows(g2.consistency(rows))
    n_holdup_rows = sum(1 for r in rows if r.h_l is not None)

    # 58 since g2-4, was 54: the screen subtracts an acceleration term from the reported
    # gradient before implying a holdup from it, and that term is now the march one
    # (dp/dx x (1 - det)) rather than the homogeneous 1/(1 - E_k) one. The screen is
    # therefore slightly less aggressive at high gas rate. 49 runs are kept by both.
    assert len(keep) == 58 and n_holdup_rows == 93

    screened = g2.summarise(points, holdup_runs=keep)
    unscreened = g2.summarise(points)

    # the gradient statistics must be identical: the screen touches holdup only
    assert screened["all"]["n_gradient"] == unscreened["all"]["n_gradient"] == 131
    assert screened["all"]["gradient_mae"] == unscreened["all"]["gradient_mae"]
    # holdup is restricted
    assert screened["all"]["n_holdup"] == 58
    # and per diameter the screened error is no worse than the unscreened one
    for k in ("0.10", "0.20", "0.30"):
        a = unscreened["by_diameter"][k]["holdup_mae"]
        b = screened["by_diameter"][k]["holdup_mae"]
        assert b <= a + 1e-12, (k, a, b)

    print(f"\nscreen keeps {len(keep)} of {n_holdup_rows} holdup rows; gradient stays on all "
          f"{screened['all']['n_gradient']}")
    print("    holdup MAE     unscreened   screened")
    for k in ("0.10", "0.20", "0.30"):
        a = unscreened["by_diameter"][k]
        b = screened["by_diameter"][k]
        print(f"    {k} m         {a['holdup_mae']:7.2%} (n={a['n_holdup']:2d})  "
              f"{b['holdup_mae']:7.2%} (n={b['n_holdup']:2d})")
    print(f"    overall        {unscreened['all']['holdup_mae']:7.2%} (n=93)  "
          f"{screened['all']['holdup_mae']:7.2%} (n={screened['all']['n_holdup']})")
    print("    the overall number rises only because the mix changed: the 3.83 in section")
    print("    has the loosest tolerance, keeps 37 of 40 rows, and is where the closure is")
    print("    worst. Per diameter every number improves.")
