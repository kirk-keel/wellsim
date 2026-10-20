"""rw-1 and rw-2: relief-well hydraulics and the kill-fluid arrival temperature.

Provenance: the bit-nozzle constant 12031 and the HHP constant 1714 are MEMORY (the
standard oilfield forms; no primary source consulted in-session). Everything else is
COMPUTED: the friction is the project's own Colebrook routine, the hydrostatic is
rho TVD / 144, and the intersection loss is K rho v^2 / (2 g_c) written out. The hand
check below prints v, Re, f and every term so the whole balance can be redone on paper.

rw-2 adds Ramey (1962), MEMORY, and the Hasan-Kabir time function f(t_D), MEMORY and on
the hand-check list. Note what test 3c records: the two published f(t_D) branches do NOT
meet at t_D = 1.5, they differ by 2.50%, so the continuity the step asked for at 1e-3 does
not hold. The coefficients are left exactly as published and the jump is pinned.
"""

from __future__ import annotations

import math

import pytest

from wellsim.cases.loader import load_case
from wellsim.cases.schema import Case
from wellsim.core import relief_well as rwm
from wellsim.core.friction import darcy_friction_factor, reynolds_number
from wellsim.core.kill import LBM_PER_FT3_PER_PPG, PSI_PER_PPG_FT, KillFluid
from wellsim.core.solver import solve

WATER_PPG = 8.33
STEEL_ROUGHNESS_IN = 0.0018


def kill_fluid(rho_ppg: float, mu_cp: float = 1.0) -> KillFluid:
    return KillFluid(rho_ppg * LBM_PER_FT3_PER_PPG, mu_cp, 1.0, 30.0, 4900.0)


def relief_well(**kw) -> "rwm.ReliefWell":
    """A ReliefWell built straight from the schema, with no case around it."""
    from wellsim.cases.schema import ReliefWell

    block = {"surface_lines": [], "string": [], "restriction": {"bit_tfa": 1.0},
             "hole_radius": 0.5, "pumping_time_min": 30.0}
    block.update(kw)
    return ReliefWell(**block)


# --- a: the hand check -------------------------------------------------------------------------


def test_hand_check_vertical_string_open_ended(capsys):
    """Hand check (COMPUTED): 10,000 ft of 4.778 in ID vertical string, water at 8.33 ppg,
    500 gpm, open-ended with K = 1.0, p_zinj given directly as 5,000 psia. Every term is
    printed so the balance can be redone on paper."""
    q_gpm = 500.0
    q_bpm = q_gpm / rwm.GAL_PER_BBL
    p_zinj = 5000.0
    rw = relief_well(
        string=[{"md": 10000.0, "tvd": 10000.0, "id": 4.778}],
        restriction={"intersection_k": 1.0},
    )
    kill = kill_fluid(WATER_PPG)
    point = rwm.pump_pressure(rw, kill, q_bpm, p_zinj)

    # independent arithmetic, written out here rather than called from the module
    d_ft = 4.778 / 12.0
    area = math.pi * 0.25 * d_ft ** 2
    q_ft3_s = q_bpm * 5.614583 / 60.0
    v = q_ft3_s / area
    rho = WATER_PPG * LBM_PER_FT3_PER_PPG
    re = reynolds_number(rho, v, d_ft, 1.0)
    f = darcy_friction_factor(re, STEEL_ROUGHNESS_IN / 4.778)
    dp_fric = f * rho * v * v / (2.0 * 32.174 * d_ft) / 144.0 * 10000.0
    dp_hyd = rho * 10000.0 / 144.0
    dp_k = 1.0 * rho * v * v / (2.0 * 32.174) / 144.0
    expected = p_zinj - dp_hyd + dp_fric + dp_k

    assert point.legs[0].v_ft_s == pytest.approx(v, rel=1e-9)
    assert point.legs[0].re == pytest.approx(re, rel=1e-9)
    assert point.legs[0].f == pytest.approx(f, rel=1e-9)
    assert point.dp_friction_string_psi == pytest.approx(dp_fric, rel=1e-9)
    assert point.dp_hydrostatic_psi == pytest.approx(dp_hyd, rel=1e-9)
    assert point.dp_restriction_psi == pytest.approx(dp_k, rel=1e-9)
    assert point.p_pump_balance_psia == pytest.approx(expected, rel=1e-9)
    assert point.dp_friction_surface_psi == 0.0

    hhp = point.p_pump_psig * q_gpm / 1714.0
    assert point.hhp == pytest.approx(hhp, rel=1e-12)

    print("\n8a hand check: 10,000 ft x 4.778 in ID vertical, water 8.33 ppg, 500 gpm,")
    print("               open-ended K = 1.0, p_zinj 5,000.0 psia given directly")
    print(f"  flow rate           {q_gpm:12.4f} gpm = {q_bpm:.6f} bbl/min = {q_ft3_s:.6f} ft3/s")
    print(f"  ID                  {4.778:12.4f} in = {d_ft:.6f} ft")
    print(f"  area                {area:12.6f} ft2")
    print(f"  velocity v          {v:12.4f} ft/s")
    print(f"  density rho         {rho:12.4f} lbm/ft3  (8.33 ppg x 7.480519480)")
    print(f"  Reynolds Re         {re:12,.0f}   (mu 1.0 cp)")
    print(f"  relative roughness  {STEEL_ROUGHNESS_IN / 4.778:12.3e}")
    print(f"  Moody f             {f:12.6f}")
    print(f"  friction dp         {dp_fric:12.2f} psi  = f rho v2 / (2 gc d) / 144 x 10,000 ft")
    print(f"  hydrostatic dp      {dp_hyd:12.2f} psi  = rho x 10,000 / 144 (exact basis)")
    print(f"     0.052 rule       {PSI_PER_PPG_FT * WATER_PPG * 10000.0:12.2f} psi  "
          f"(rule of thumb, not used)")
    print(f"  restriction dp      {dp_k:12.2f} psi  = K rho v2 / (2 gc) / 144, K = 1.0")
    print(f"  p_pump balance      {point.p_pump_balance_psia:12.2f} psia")
    print(f"  p_pump reported     {point.p_pump_psig:12.2f} psig   on vacuum: {point.on_vacuum}")
    print(f"  HHP                 {point.hhp:12.2f}       = p_psig x gpm / 1714")


# --- b: the bit-nozzle hand value -----------------------------------------------------------------


def test_bit_nozzle_loss_hand_value(capsys):
    """MEMORY form dp = rho q^2 / (12031 TFA^2): 12 ppg, 800 gpm, TFA 1.0 in^2."""
    dp = rwm.bit_nozzle_loss(12.0, 800.0, 1.0)
    expected = 12.0 * 800.0 ** 2 / (12031.0 * 1.0 ** 2)
    assert dp == pytest.approx(expected, rel=1e-12)
    assert dp == pytest.approx(638.35, abs=0.01)  # 12 x 640,000 / 12,031
    # the square laws, both ways
    assert rwm.bit_nozzle_loss(12.0, 1600.0, 1.0) == pytest.approx(4.0 * dp, rel=1e-12)
    assert rwm.bit_nozzle_loss(12.0, 800.0, 2.0) == pytest.approx(dp / 4.0, rel=1e-12)
    assert rwm.bit_nozzle_loss(24.0, 800.0, 1.0) == pytest.approx(2.0 * dp, rel=1e-12)
    print(f"\n8b bit nozzle: 12 ppg, 800 gpm, TFA 1.0 in2 -> {dp:.2f} psi "
          f"(12 x 800^2 / (12031 x 1^2))")


# --- c: the zero-length string ---------------------------------------------------------------------


def test_zero_length_string_is_p_zinj_plus_the_restriction(capsys):
    """Degenerate case: with no string and no surface lines there is no friction and no
    hydrostatic, so the balance is p_zinj plus the restriction loss, exactly."""
    rw = relief_well(restriction={"bit_tfa": 1.0})
    kill = kill_fluid(12.0)
    q_bpm = 800.0 / rwm.GAL_PER_BBL
    p_zinj = 4000.0
    point = rwm.pump_pressure(rw, kill, q_bpm, p_zinj)

    assert point.dp_hydrostatic_psi == 0.0
    assert point.dp_friction_string_psi == 0.0
    assert point.dp_friction_surface_psi == 0.0
    assert point.legs == []
    # exact: the balance contributes nothing but p_zinj and the restriction
    assert point.p_pump_balance_psia == p_zinj + point.dp_restriction_psi
    # and that restriction is the bit-nozzle hand value (to 1e-12: rho and q each make a
    # ppg -> lbm/ft3 -> ppg and a bbl/min -> gpm round trip through KillFluid on the way in)
    assert point.dp_restriction_psi == pytest.approx(rwm.bit_nozzle_loss(12.0, 800.0, 1.0), rel=1e-12)
    assert not point.on_vacuum
    print(f"\n8c zero-length string: p_pump {point.p_pump_balance_psia:.6f} psia = p_zinj "
          f"{p_zinj:.1f} + bit loss {point.dp_restriction_psi:.6f} psi, exactly")


# --- d: free fall -----------------------------------------------------------------------------------


def test_free_fall_is_reported_with_the_rate_where_the_balance_reaches_atmospheric(capsys):
    """A heavy column with little friction goes on vacuum: the balance is below atmospheric
    at the kill rate, the message says so, no negative pump pressure is reported, and the
    free-fall rate solves p_pump = 14.7 psia to 1e-3."""
    # 12 ppg down 10,000 ft is 6,234 psi of column against p_zinj 3,000 psia: on vacuum
    # at any rate the friction cannot make up, through a wide 6 in string with a big TFA
    rw = relief_well(
        string=[{"md": 10000.0, "tvd": 10000.0, "id": 6.0}],
        restriction={"bit_tfa": 3.0},
    )
    kill = kill_fluid(12.0)
    p_zinj = 3000.0
    at_kill = rwm.pump_pressure(rw, kill, 20.0, p_zinj)
    assert at_kill.on_vacuum
    assert at_kill.p_pump_balance_psia < rwm.P_ATM_PSIA
    assert at_kill.p_pump_psig == 0.0  # never a negative pump pressure
    assert at_kill.hhp == 0.0
    assert any("free-falls" in w for w in at_kill.warnings)
    assert any("does not control rate" in w for w in at_kill.warnings)

    q_ff = rwm.free_fall_rate(rw, kill, p_zinj)
    assert q_ff is not None and q_ff > 20.0
    at_ff = rwm.pump_pressure(rw, kill, q_ff, p_zinj)
    assert at_ff.p_pump_balance_psia == pytest.approx(rwm.P_ATM_PSIA, abs=1e-3)
    # and it is a boundary: just below is vacuum, just above is not
    assert rwm.pump_pressure(rw, kill, q_ff * 0.99, p_zinj).on_vacuum
    assert not rwm.pump_pressure(rw, kill, q_ff * 1.01, p_zinj).on_vacuum

    # a case that is never on vacuum returns None rather than a rate
    shallow = relief_well(string=[{"md": 500.0, "tvd": 500.0, "id": 4.778}],
                          restriction={"bit_tfa": 1.0}, hole_radius=0.5, pumping_time_min=30.0)
    assert rwm.free_fall_rate(shallow, kill, 4000.0) is None

    print(f"\n8d free fall: 12 ppg, 10,000 ft of 6 in, TFA 3.0 in2, p_zinj {p_zinj:.0f} psia")
    print(f"  at 20 bbl/min the balance is {at_kill.p_pump_balance_psia:.1f} psia "
          f"(hydrostatic {at_kill.dp_hydrostatic_psi:,.0f} psi against friction + restriction "
          f"{at_kill.dp_friction_string_psi + at_kill.dp_restriction_psi:,.0f} psi): on vacuum")
    print(f"  free-fall rate {q_ff:.6f} bbl/min, balance there "
          f"{at_ff.p_pump_balance_psia:.6f} psia")


# --- e: monotone ---------------------------------------------------------------------------------


def test_pump_pressure_increases_with_rate_across_the_schedule(capsys):
    """Hydrostatic does not depend on rate and both friction and the restriction rise with
    it, so p_pump must rise monotonically wherever the string is not on vacuum."""
    rw = relief_well(
        surface_lines=[{"length": 300.0, "id": 4.0}],
        string=[{"md": 11500.0, "tvd": 10000.0, "id": 4.778}],
        restriction={"bit_tfa": 1.0},
    )
    kill = kill_fluid(12.0)
    schedule = rwm.pump_schedule(rw, kill, 27.5, 4800.0, steps=10)
    assert len(schedule) == 10
    assert schedule[0].q_bpm == pytest.approx(27.5)
    assert schedule[-1].q_bpm == pytest.approx(1.5 * 27.5)
    assert not any(p.on_vacuum for p in schedule)
    for a, b in zip(schedule, schedule[1:]):
        assert b.q_bpm > a.q_bpm
        assert b.p_pump_psig > a.p_pump_psig, f"p_pump fell from {a.q_bpm} to {b.q_bpm} bbl/min"
        assert b.hhp > a.hhp
        assert b.dp_hydrostatic_psi == pytest.approx(a.dp_hydrostatic_psi)  # rate-independent

    print("\n8e pump schedule, 12 ppg, 27.5 to 41.25 bbl/min:")
    print("     q bbl/min    q gpm   p_pump psig        HHP")
    for p in schedule:
        print(f"    {p.q_bpm:10.3f} {p.q_gpm:8.1f} {p.p_pump_psig:13.1f} {p.hhp:10.1f}")


# --- the schema refuses a missing or doubled loss spec ---------------------------------------------


def test_restriction_must_be_given_exactly_once():
    """Missing loss spec is an error, not a default: a relief-well job either has nozzles
    or it does not, and guessing puts hundreds of psi in or out of the pump pressure."""
    from wellsim.cases.schema import ReliefRestriction, ReliefWell

    with pytest.raises(ValueError, match="exactly one of bit_tfa"):
        ReliefRestriction()
    with pytest.raises(ValueError, match="exactly one of bit_tfa"):
        ReliefRestriction(bit_tfa=1.0, intersection_k=1.0)
    with pytest.raises(ValueError, match="needs a string"):
        ReliefWell(string=[], restriction={"intersection_k": 1.0}, hole_radius=0.5,
                   pumping_time_min=30.0)
    assert ReliefRestriction(bit_tfa=1.0).kind == "bit_tfa"
    assert ReliefRestriction(intersection_k=0.0).kind == "intersection_k"


def test_string_trajectory_rules_match_the_survey_rules():
    """The string reuses the survey's (md, tvd) station pair and its monotonicity rules:
    md increasing, tvd non-decreasing, and no tvd increment larger than its md increment."""
    from wellsim.cases.schema import ReliefWell

    ok = ReliefWell(string=[{"md": 5000.0, "tvd": 5000.0, "id": 4.778},
                            {"md": 11500.0, "tvd": 10000.0, "id": 4.0}],
                    restriction={"bit_tfa": 1.0}, hole_radius=0.5, pumping_time_min=30.0)
    assert ok.bottom_md == 11500.0 and ok.bottom_tvd == 10000.0
    with pytest.raises(ValueError, match="md must increase"):
        ReliefWell(string=[{"md": 5000.0, "tvd": 4000.0, "id": 4.778},
                           {"md": 5000.0, "tvd": 4500.0, "id": 4.778}],
                   restriction={"bit_tfa": 1.0}, hole_radius=0.5, pumping_time_min=30.0)
    with pytest.raises(ValueError, match="tvd must not decrease"):
        ReliefWell(string=[{"md": 5000.0, "tvd": 5000.0, "id": 4.778},
                           {"md": 9000.0, "tvd": 4000.0, "id": 4.778}],
                   restriction={"bit_tfa": 1.0}, hole_radius=0.5, pumping_time_min=30.0)
    with pytest.raises(ValueError, match="exceeds the md increment"):
        ReliefWell(string=[{"md": 5000.0, "tvd": 6000.0, "id": 4.778}],
                   restriction={"bit_tfa": 1.0}, hole_radius=0.5, pumping_time_min=30.0)


def test_relief_well_needs_the_kill_block_and_must_intersect_at_z_inj(examples_dir):
    """The pump balance starts from the blowing well's pressure at z_inj, so the case must
    carry dynamic_kill and the string must actually reach z_inj."""
    base = load_case(examples_dir / "rw_wgr.json")
    d = base.model_dump()
    d["dynamic_kill"] = None
    with pytest.raises(ValueError, match="relief_well needs the dynamic_kill block"):
        Case(**d)
    d = base.model_dump()
    d["relief_well"]["string"][0]["tvd"] = 9000.0
    with pytest.raises(ValueError, match="must intersect the blowing well at z_inj"):
        Case(**d)


# --- the example end to end --------------------------------------------------------------------


def test_rw_wgr_example_reports_a_pump_pressure(examples_dir, capsys):
    """The shipped example: the kill search runs, the pump balance is attached to the kill
    result, and the schedule and free-fall rate come with it."""
    result = solve(load_case(examples_dir / "rw_wgr.json"))
    rw = result.kill["relief_well"]
    at_kill = rw["at_kill"]
    q_k = result.kill["q_k_min"]["q_k_min_bpm"]

    assert rw["restriction"] == "bit_tfa"
    assert at_kill["q_bpm"] == pytest.approx(q_k)
    assert not at_kill["on_vacuum"]
    assert at_kill["p_pump_psig"] > 0.0
    assert at_kill["hhp"] == pytest.approx(at_kill["p_pump_psig"] * at_kill["q_gpm"] / 1714.0, rel=1e-12)
    assert len(at_kill["legs"]) == 2  # one surface line, one string section
    assert len(rw["schedule"]) == 10
    # p_zinj came from the kill search, not from a second model of the blowing well
    assert "residual minimiser" in rw["z_inj_basis"]
    assert 0.0 < rw["p_zinj_psia"] < result.reservoir_pressure
    # the free-fall rate is below the kill rate here, so the pumps control at the kill rate
    assert rw["free_fall"]["q_free_fall_bpm"] < q_k
    # inside every limit at the kill rate
    assert not any(v is True for v in at_kill["limits"].values())

    print(f"\nrw_wgr example: q_k_min {q_k:.4f} bbl/min, p_zinj {rw['p_zinj_psia']:,.1f} psia, "
          f"T {rw['t_zinj_degf']:.1f} degF")
    print(f"  p_pump {at_kill['p_pump_psig']:,.1f} psig, HHP {at_kill['hhp']:,.0f}, "
          f"free-fall rate {rw['free_fall']['q_free_fall_bpm']:.3f} bbl/min")


@pytest.mark.slow
def test_monte_carlo_carries_the_pump_balance_through_the_kill_samples(examples_dir, capsys):
    """rw-1 in mc-1: every kill sample reports the pump pressure and horsepower at its own
    kill rate and its own blowing-well pressure at z_inj, with no extra kill searches, and
    the injection-temperature bounds are forced off inside the Monte Carlo. Slow: two kill
    searches."""
    from wellsim.core import montecarlo as mc

    base = load_case(examples_dir / "rw_wgr.json")
    d = base.model_dump()
    d["relief_well"]["geothermal_bound"] = True  # on for the case, must be off per sample
    d["mc"] = {
        "seed": 5, "n_samples": 4, "workers": 1,
        "p_res": {"min": 4800.0, "mode": 5000.0, "max": 5200.0},
        "kh": {"low": 3000.0, "high": 8000.0},
        "kill": {"rho_k": 12.0, "mu_k": 1.0, "cp_k": 1.0, "n_kill": 2, "chains": 1},
    }
    case = Case(**d)
    # rw-3b: the sample case runs the surface and finite-U searches (two per sample) and
    # skips the U -> infinity bound run and the geothermal run, so the reported kill rate
    # is at the arrival temperature rather than the M3-4 stream temperature
    sample_case = mc.apply_sample(case, mc.sample_inputs(case)[0], for_kill=True)
    assert sample_case.relief_well.temperature_runs is True
    assert sample_case.relief_well.finite_u is True
    assert sample_case.relief_well.bound_run is False
    assert sample_case.relief_well.geothermal_bound is False

    result = mc.run(case)
    assert len(result.kill_rows) == 2
    # the arrival temperature the searches ran at must be the finite-U estimate, which on
    # this string is far below the U -> infinity bound
    for row in result.kill_rows:
        assert 60.0 < row["t_arrival_degf"] < 75.0, row["t_arrival_degf"]
    for row in result.kill_rows:
        assert row["p_pump_psig"] is not None and row["p_pump_psig"] > 0.0
        assert row["hhp"] > 0.0
        assert row["p_zinj_psia"] > 0.0
        assert row["on_vacuum"] is False
        assert row["over_limit"] is False
    assert result.percentiles["p_pump"]["n"] == 2
    assert result.percentiles["hhp"]["n"] == 2
    assert result.vacuum_fraction == 0.0
    assert result.over_limit_fraction == 0.0
    text = mc.summary_markdown(result)
    assert "relief-well pump pressure at the kill rate" in text
    assert "on vacuum at the kill rate on 0.0%" in text
    assert not [line for line in text.splitlines() if len(line) > 80]

    print("\nrw-1 through mc-1, 4 samples, 2 kill samples:")
    print("   index   q_g Mscf/d  q_k bbl/min  p_zinj psia  p_pump psig     HHP")
    for row in result.kill_rows:
        print(f"  {row['index']:6d} {row['q_g']:12,.0f} {row['q_k_min']:12.3f} "
              f"{row['p_zinj_psia']:12,.1f} {row['p_pump_psig']:12,.1f} {row['hhp']:7,.0f}")


# --- rw-2: the Ramey arrival temperature ------------------------------------------------------


def test_ramey_hand_check_against_the_closed_form(capsys):
    """3a hand check (COMPUTED): vertical 10,000 ft, T_s 60 degF, a 0.012 degF/ft, w c_p and
    f given directly. The segment solver must reproduce the Ramey closed form

        T = T_s + a (z - A (1 - exp(-z/A)))

    which holds when the fluid enters at the surface geothermal temperature. An independent
    RK4 integration of dT/dz = (T_g - T)/A is run alongside, so the closed form is checked
    against the differential equation and not only against itself."""
    t_s, a_grad, z = 60.0, 0.012, 10000.0
    w_cp = 800000.0  # w c_p, Btu/(hr degF), given directly
    f_t = 0.30       # fixed f, given directly
    k_e = 1.4
    a_relax = rwm.relaxation_length(w_cp, 1.0, f_t, k_e)

    t_arr, rows = rwm.arrival_temperature([(z, z)], t_s, t_s, a_grad, a_relax)
    closed = t_s + a_grad * (z - a_relax * (1.0 - math.exp(-z / a_relax)))
    assert t_arr == pytest.approx(closed, rel=1e-12, abs=1e-12)

    # independent RK4 on dT/dz = (T_g(z) - T)/A, 20,000 steps
    n = 20000
    h = z / n
    t = t_s

    def slope(tt, zz):
        return (t_s + a_grad * zz - tt) / a_relax

    for i in range(n):
        zz = i * h
        k1 = slope(t, zz)
        k2 = slope(t + 0.5 * h * k1, zz + 0.5 * h)
        k3 = slope(t + 0.5 * h * k2, zz + 0.5 * h)
        k4 = slope(t + h * k3, zz + h)
        t += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    assert t_arr == pytest.approx(t, abs=1e-6)

    assert len(rows) == 1
    assert rows[0]["t_in_degf"] == t_s
    assert rows[0]["t_formation_bottom_degf"] == pytest.approx(t_s + a_grad * z)

    print("\n3a Ramey hand check: vertical 10,000 ft, T_s 60 degF, a 0.012 degF/ft,")
    print(f"                     w c_p {w_cp:,.0f} Btu/(hr degF), f(t) {f_t:g}, k_e {k_e:g}")
    print(f"  A = w c_p f / (2 pi k_e)   {a_relax:14,.4f} ft")
    print(f"  z / A                      {z / a_relax:14.6f}")
    print(f"  formation T at 10,000 ft   {t_s + a_grad * z:14.6f} degF")
    print(f"  closed form T_arrival      {closed:14.6f} degF")
    print(f"  segment solver T_arrival   {t_arr:14.6f} degF  (difference "
          f"{abs(t_arr - closed):.2e})")
    print(f"  RK4, 20,000 steps          {t:14.6f} degF  (difference "
          f"{abs(t_arr - t):.2e})")
    print(f"  heating                    {t_arr - t_s:14.6f} degF of the "
          f"{a_grad * z:.1f} degF available")


def test_ramey_limits_are_the_inlet_and_the_formation(capsys):
    """3b limits (COMPUTED): A -> infinity is a string so fast and so poorly coupled that
    nothing heats, arrival T = T_s; A -> 0 is a string so slow and so well coupled that the
    fluid sits on the geothermal profile, arrival T = T_g at the bottom."""
    t_s, a_grad, z = 60.0, 0.012, 10000.0
    t_g_bottom = t_s + a_grad * z

    hot, _ = rwm.arrival_temperature([(z, z)], t_s, t_s, a_grad, 1e12)
    assert hot == pytest.approx(t_s, abs=1e-4)
    # and the approach is monotone from above in A
    prev = None
    for a_relax in (1e12, 1e9, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e-1):
        t_arr, _ = rwm.arrival_temperature([(z, z)], t_s, t_s, a_grad, a_relax)
        assert t_s - 1e-6 <= t_arr <= t_g_bottom + 1e-6
        if prev is not None:
            assert t_arr > prev, f"arrival T did not rise as A fell to {a_relax:g}"
        prev = t_arr
    cold, _ = rwm.arrival_temperature([(z, z)], t_s, t_s, a_grad, 1e-6)
    assert cold == pytest.approx(t_g_bottom, abs=1e-6)
    assert rwm.arrival_temperature([(z, z)], t_s, t_s, a_grad, 0.0)[0] == pytest.approx(t_g_bottom)

    print(f"\n3b limits, T_s {t_s:.0f} degF, formation at bottom {t_g_bottom:.0f} degF:")
    print(f"  A = 1e12 ft (no coupling)   arrival {hot:10.6f} degF")
    print(f"  A = 1e-6 ft (full coupling) arrival {cold:10.6f} degF")


def test_hasan_kabir_time_function_is_not_continuous_at_td_1p5(capsys):
    """3c. FETCHED at hk-1: Hasan & Kabir 2nd ed. Eqs. 6.9 (short) and 6.8 (long), sec.
    6.2.2. All five constants are exactly as implemented, so the jump is NOT a
    mis-remembered coefficient - the book prints the two branches and then says of them,
    in as many words, "Eqs. 6.8 and 6.9 are discontinuous at t_D = 1.5".

    The step asked for continuity at t_D = 1.5 to 1e-3. It does not hold: the two published
    branches disagree by 0.021344 absolute, 2.50% of the smaller. The coefficients are
    written exactly as published and the jump is PINNED here rather than closed by tuning
    one of them. The book's own remedy is its continuous Eq. 6.10, which is a DIFFERENT
    correlation rather than a smoothing (0.9068 at t_D = 1.5, above both branches);
    adopting it would move arrival temperatures, so it is logged in docs/TODO-M4.md for the
    owner rather than taken here.

    The jump does not make the kill search jagged: t_D = alpha t / r_w^2 does not depend on
    rate, so f is a constant for a given case."""
    short = rwm.time_function(1.5)
    long_ = rwm.time_function(1.5 + 1e-12)
    jump = long_ - short

    assert short == pytest.approx(0.8739896894, abs=1e-9)
    assert long_ == pytest.approx(0.8526455757, abs=1e-9)
    assert jump == pytest.approx(-0.0213441137, abs=1e-9)
    assert abs(jump) > 1e-3, "the published branches now agree; the pin needs revisiting"
    # the line-source solution the long-time branch approximates, for the hand check
    line_source = 0.5 * (math.log(1.5) + 0.80907) * (1.0 + 0.6 / 1.5)
    assert line_source == pytest.approx(0.8501745757, abs=1e-9)

    # each branch is well behaved on its own side
    assert rwm.time_function(0.0) == 0.0
    for t_d in (0.01, 0.1, 0.5, 1.0, 1.4):
        assert 0.0 < rwm.time_function(t_d) < 1.0
    for a, b in zip((2.0, 5.0, 20.0, 100.0), (5.0, 20.0, 100.0, 1000.0)):
        assert rwm.time_function(b) > rwm.time_function(a)

    print("\n3c Hasan-Kabir f(t_D) at the t_D = 1.5 switch:")
    print(f"  short-time branch 1.1281 sqrt(tD) (1 - 0.3 sqrt(tD))   {short:.10f}")
    print(f"  long-time  branch (0.4063 + 0.5 ln tD)(1 + 0.6/tD)     {long_:.10f}")
    print(f"  jump                                                   {jump:+.10f}"
          f"  ({jump / short:+.4%})")
    print(f"  line source 0.5(ln tD + 0.80907) x (1 + 0.6/tD)        {line_source:.10f}")
    print("  NOT continuous to 1e-3. Both branches are as published; at least one of the")
    print("  five constants is wrong. On the hand-check list in docs/PROVENANCE.md.")


@pytest.mark.slow
def test_rw_wgr_arrival_temperature_and_both_kill_rates(examples_dir, capsys):
    """3d on the shipped example with 30 minutes of pumping: the arrival temperature, the
    relaxation length, and the kill rate at the Ramey upper bound and at the surface
    temperature.

    Pinned to the rw-2 path (``finite_u`` off), so it stays a regression test of the
    bound-only behaviour after rw-3 made the finite-U estimate the reported run. The rw-3
    path is covered by test_rw_wgr_three_kill_rates_and_two_gaps.

    Slow: two kill searches."""
    d = load_case(examples_dir / "rw_wgr.json").model_dump()
    d["relief_well"]["finite_u"] = False
    result = solve(Case(**d))
    rw = result.kill["relief_well"]
    temp = rw["temperature"]
    arrival = temp["arrival"]
    lo, hi = temp["lower_bound"], temp["upper_bound"]

    assert arrival["bound"] == "upper"
    assert arrival["t_inlet_degf"] == pytest.approx(60.0)
    assert arrival["t_formation_at_bottom_degf"] == pytest.approx(180.0, abs=0.01)
    # the bound must sit between the two it brackets, and well below the geothermal one
    assert 60.0 < arrival["t_arrival_degf"] < 180.0
    assert hi["q_k_min_bpm"] > lo["q_k_min_bpm"]  # hotter kill fluid needs a higher rate
    assert hi["reported"] is True
    assert result.kill["q_k_min"]["q_k_min_bpm"] == hi["q_k_min_bpm"]
    # the gap divides by the smaller rate (CLAUDE.md convention), so it reads as how much
    # higher the arrival-bound rate is than the surface one. It must be well inside the
    # surface-to-geothermal bracket, which is 6.94% on the same convention and corrected rates
    assert temp["gap_relative"] < 0.0694
    assert temp["gap_relative"] == pytest.approx(
        (hi["q_k_min_bpm"] - lo["q_k_min_bpm"]) / lo["q_k_min_bpm"], rel=1e-12
    )
    assert "estimate" not in temp  # the rw-2 path computes no finite-U estimate

    print(f"\n3d rw_wgr with {arrival['t_hr'] * 60:.0f} min of pumping:")
    print(f"  t_D = alpha t / r_w^2      {arrival['t_d']:14.6f} "
          f"(alpha {arrival['alpha_ft2_hr']:g} ft2/hr, r_w {arrival['r_w_ft']:g} ft)")
    print(f"  f(t_D)                     {arrival['f_t']:14.6f}")
    print(f"  mass rate w                {arrival['w_lbm_hr']:14,.0f} lbm/hr")
    print(f"  A = w cp f / (2 pi k_e)    {arrival['a_relax_ft']:14,.1f} ft "
          f"(k_e {arrival['k_e']:g})")
    print(f"  string bottom              {11500.0:14,.1f} ft MD, "
          f"{10000.0:,.1f} ft TVD")
    print(f"  inlet T                    {arrival['t_inlet_degf']:14.2f} degF")
    print(f"  ARRIVAL T (upper bound)    {arrival['t_arrival_degf']:14.2f} degF")
    print(f"  formation T at z_inj       {arrival['t_formation_at_bottom_degf']:14.2f} degF")
    print(f"  heating achieved           {arrival['heating_degf']:14.2f} degF of "
          f"{arrival['t_formation_at_bottom_degf'] - arrival['t_inlet_degf']:.1f} available "
          f"({arrival['fraction_of_available_heating']:.1%})")
    print()
    print(f"  kill rate at surface T {lo['t_inject_degf']:.1f} degF   "
          f"{lo['q_k_min_bpm']:10.4f} bbl/min  (lower bound)")
    print(f"  kill rate at arrival T {hi['t_inject_degf']:.1f} degF   "
          f"{hi['q_k_min_bpm']:10.4f} bbl/min  (REPORTED, upper bound)")
    print(f"  gap                                    {temp['gap_relative']:10.2%}  "
          f"({temp['gap_bpm']:+.4f} bbl/min)")
    print(f"  arrival T recomputed at the reported rate "
          f"{temp['arrival_at_reported_rate_degf']:.2f} degF "
          f"({temp['arrival_residual_degf']:+.2f} degF, not iterated)")
    print(f"  rw-1 reported this gap as 5.27% between 60 and 180 degF; rw-2 closes it to "
          f"{temp['gap_relative']:.2%}")


# --- rw-3: finite overall heat transfer coefficient --------------------------------------------


def _rw3_relief_well(**kw):
    from wellsim.cases.schema import ReliefWell

    block = {
        "surface_lines": [], "string": [{"md": 11500.0, "tvd": 10000.0, "id": 4.778, "od": 5.5}],
        "restriction": {"bit_tfa": 1.0}, "hole_radius": 0.5, "pumping_time_min": 30.0,
        "surface_temperature": 60.0, "k_e": 1.4, "alpha": 0.04, "k_pipe": 25.0,
        "annulus_fill": "mud", "k_annulus": 0.7,
    }
    block.update(kw)
    return ReliefWell(**block)


def test_finite_u_reduces_to_the_rw2_bound_as_u_goes_to_infinity(capsys):
    """4a (COMPUTED). A = w c_p (k_e + r_to U f(t)) / (2 pi r_to U k_e) must become
    w c_p f(t) / (2 pi k_e) as U grows without bound, so the rw-3 estimate becomes the rw-2
    bound. Checked at 1e-6 on both A and the arrival temperature, and exactly at U = inf."""
    w, cp, f_t, k_e, r_to = 791089.0, 1.0, 0.292, 1.4, 5.5 / 24.0
    a_bound = rwm.relaxation_length(w, cp, f_t, k_e)

    assert rwm.relaxation_length_finite_u(w, cp, f_t, k_e, r_to, math.inf) == a_bound  # exact
    for u in (1e6, 1e8, 1e10):
        a_u = rwm.relaxation_length_finite_u(w, cp, f_t, k_e, r_to, u)
        assert a_u > a_bound  # a finite U always makes the string cooler
    a_big = rwm.relaxation_length_finite_u(w, cp, f_t, k_e, r_to, 1e10)
    assert a_big == pytest.approx(a_bound, rel=1e-6)

    stations = [(11500.0, 10000.0)]
    t_bound, _ = rwm.arrival_temperature(stations, 60.0, 60.0, 0.012, a_bound)
    t_big, _ = rwm.arrival_temperature(stations, 60.0, 60.0, 0.012, a_big)
    assert t_big == pytest.approx(t_bound, rel=1e-6)

    print(f"\n4a U -> infinity: A {a_bound:,.4f} ft (bound) against {a_big:,.4f} ft at "
          f"U = 1e10, arrival {t_bound:.6f} against {t_big:.6f} degF")


def test_finite_u_at_zero_gives_the_surface_temperature(capsys):
    """4b (COMPUTED). U -> 0 is a perfectly insulated string: A diverges, nothing heats, and
    the fluid arrives at the temperature it entered at."""
    w, cp, f_t, k_e, r_to = 791089.0, 1.0, 0.292, 1.4, 5.5 / 24.0
    stations = [(11500.0, 10000.0)]

    assert rwm.relaxation_length_finite_u(w, cp, f_t, k_e, r_to, 0.0) == math.inf
    t_zero, _ = rwm.arrival_temperature(stations, 60.0, 60.0, 0.012, math.inf)
    assert t_zero == pytest.approx(60.0, abs=1e-9)

    prev = None
    for u in (1e-6, 1e-4, 1e-2, 1.0, 10.0, 1e4):
        a_u = rwm.relaxation_length_finite_u(w, cp, f_t, k_e, r_to, u)
        t_u, _ = rwm.arrival_temperature(stations, 60.0, 60.0, 0.012, a_u)
        assert 60.0 - 1e-9 <= t_u
        if prev is not None:
            assert t_u > prev, f"arrival T did not rise with U at U = {u:g}"
        prev = t_u
    print(f"\n4b U -> 0: arrival {t_zero:.9f} degF, the inlet exactly; rising monotonically "
          f"with U to {prev:.4f} degF at U = 1e4")


def test_rw3_hand_check_on_rw_wgr(examples_dir, capsys):
    """4c hand check (COMPUTED, with MEMORY forms): every term of U and A printed for
    rw_wgr, so the series resistance can be redone on paper. No kill search: the arrival
    temperature is evaluated at a fixed rate."""
    from dataclasses import replace

    case = load_case(examples_dir / "rw_wgr.json")
    kill = KillFluid.from_schema(case.dynamic_kill)
    q_k = 26.1604  # the surface-bound kill rate, the rate the estimate is built at
    prof = rwm.ramey_arrival_finite_u(case, kill, q_k)
    seg = prof.segments[0]
    r = seg["resistances"]

    # re-derive each term here rather than trusting the module
    d_i, d_o = 4.778, 5.5
    r_ti, r_to, r_w = d_i / 24.0, d_o / 24.0, 0.5
    pr = 1.0 * 1.0 * 2.4190883 / 0.4
    assert seg["pr"] == pytest.approx(pr, rel=1e-12)
    h_i = 0.023 * seg["re"] ** 0.8 * pr ** 0.4 * 0.4 / (d_i / 12.0)
    assert seg["h_i"] == pytest.approx(h_i, rel=1e-12)
    r_film = r_to / (r_ti * h_i)
    r_pipe = r_to * math.log(r_to / r_ti) / 25.0
    r_ann = r_to * math.log(r_w / r_to) / 0.7
    assert r["film"] == pytest.approx(r_film, rel=1e-12)
    assert r["pipe"] == pytest.approx(r_pipe, rel=1e-12)
    assert r["annulus"] == pytest.approx(r_ann, rel=1e-12)
    u = 1.0 / (r_film + r_pipe + r_ann)
    assert seg["u"] == pytest.approx(u, rel=1e-12)
    a = prof.w_lbm_hr * prof.cp * (prof.k_e + r_to * u * prof.f_t) / (2.0 * math.pi * r_to * u * prof.k_e)
    assert seg["a_relax_ft"] == pytest.approx(a, rel=1e-12)

    print(f"\n4c rw-3 hand check on rw_wgr at q_k {q_k:.4f} bbl/min:")
    print(f"  string ID / OD          {d_i:10.4f} / {d_o:.4f} in")
    print(f"  r_ti                    {r_ti:10.6f} ft")
    print(f"  r_to                    {r_to:10.6f} ft")
    print(f"  r_w (relief hole)       {r_w:10.6f} ft")
    print(f"  velocity                {seg['v_ft_s']:10.4f} ft/s")
    print(f"  Reynolds Re             {seg['re']:10,.0f}")
    print(f"  Prandtl Pr              {pr:10.4f}   = cp mu / k, "
          f"1.0 x 1.0 cp x 2.4190883 / 0.4")
    print(f"  h_i (Dittus-Boelter)    {h_i:10.1f} Btu/(hr ft2 degF)")
    print(f"        Nu = 0.023 Re^0.8 Pr^0.4 = {0.023 * seg['re'] ** 0.8 * pr ** 0.4:,.1f}")
    print(f"  R_film  r_to/(r_ti h_i)      {r_film:12.6f} hr ft2 degF/Btu "
          f"({r_film / r['total']:6.2%})")
    print(f"  R_pipe  r_to ln(r_to/r_ti)/k {r_pipe:12.6f}                  "
          f"({r_pipe / r['total']:6.2%})")
    print(f"  R_ann   r_to ln(r_w/r_to)/k  {r_ann:12.6f}                  "
          f"({r_ann / r['total']:6.2%})")
    print(f"  total                        {r['total']:12.6f}")
    print(f"  U = 1/total             {u:10.4f} Btu/(hr ft2 degF)")
    print(f"  f(t_D)                  {prof.f_t:10.6f}   (t_D {prof.t_d:.6f})")
    print(f"  w                       {prof.w_lbm_hr:10,.0f} lbm/hr")
    print(f"  A (finite U)            {a:10,.1f} ft")
    print(f"  A (U -> infinity)       {rwm.relaxation_length(prof.w_lbm_hr, prof.cp, prof.f_t, prof.k_e):10,.1f} ft")
    print(f"  arrival T (estimate)    {prof.t_arrival_degf:10.2f} degF "
          f"({prof.heating_degf:.2f} degF of {prof.t_formation_at_bottom_degf - 60.0:.0f} available, "
          f"{prof.fraction_of_available_heating:.1%})")


def test_arrival_temperature_rises_with_cement_conductivity(examples_dir, capsys):
    """4d (COMPUTED). A more conductive annulus is a smaller annular resistance, a larger U
    and more heating, so the arrival temperature must rise monotonically with k_annulus,
    and stay under the U -> infinity bound at every value. The range here is deliberately
    absurd (0.1 to 100) to exercise the monotonicity; the PLAUSIBLE range is 0.4 to 1.2 and
    is checked separately by test_k_annulus_over_its_plausible_range."""
    from wellsim.cases.schema import Case

    base = load_case(examples_dir / "rw_wgr.json")
    kill = KillFluid.from_schema(base.dynamic_kill)
    q_k = 26.1604
    bound = rwm.ramey_arrival(base, kill, q_k).t_arrival_degf

    rows = []
    prev_t = prev_u = None
    for k_cem in (0.1, 0.3, 0.7, 1.5, 4.0, 20.0, 100.0):
        d = base.model_dump()
        d["relief_well"]["k_annulus"] = k_cem
        prof = rwm.ramey_arrival_finite_u(Case(**d), kill, q_k)
        u = prof.segments[0]["u"]
        rows.append((k_cem, u, prof.segments[0]["a_relax_ft"], prof.t_arrival_degf))
        assert 60.0 < prof.t_arrival_degf < bound, f"k_cem {k_cem}: outside the bracket"
        if prev_t is not None:
            assert u > prev_u and prof.t_arrival_degf > prev_t, f"not monotone at k_cem {k_cem}"
        prev_t, prev_u = prof.t_arrival_degf, u

    print(f"\n4d arrival temperature against cement conductivity "
          f"(U -> infinity bound {bound:.2f} degF):")
    print(" k_annulus          U            A ft   arrival degF")
    print("  " + "-" * 50)
    for k_cem, u, a, t in rows:
        print(f"  {k_cem:8.1f} {u:10.4f} {a:15,.0f} {t:14.2f}")


@pytest.mark.slow
def test_rw_wgr_three_kill_rates_and_two_gaps(examples_dir, capsys):
    """4e (COMPUTED). The three injection temperatures and the three kill rates they give,
    with both gaps under the project convention of dividing by the smaller value. Slow:
    three kill searches."""
    result = solve(load_case(examples_dir / "rw_wgr.json"))
    t = result.kill["relief_well"]["temperature"]
    lo, est, hi = t["lower_bound"], t["estimate_run"], t["upper_bound"]

    assert est["reported"] is True and hi["reported"] is False
    assert result.kill["q_k_min"]["q_k_min_bpm"] == est["q_k_min_bpm"]
    # the estimate must sit strictly inside the bracket, on both temperature and rate
    assert lo["t_inject_degf"] < est["t_inject_degf"] < hi["t_inject_degf"]
    assert lo["q_k_min_bpm"] < est["q_k_min_bpm"] < hi["q_k_min_bpm"]
    assert t["gap_surface_to_estimate"] == pytest.approx(
        (est["q_k_min_bpm"] - lo["q_k_min_bpm"]) / lo["q_k_min_bpm"], rel=1e-12)
    assert t["gap_estimate_to_bound"] == pytest.approx(
        (hi["q_k_min_bpm"] - est["q_k_min_bpm"]) / est["q_k_min_bpm"], rel=1e-12)
    assert "THE MIDDLE ONE IS THE ESTIMATE" in t["note"]

    print("\n4e rw_wgr, three injection temperatures on corrected kill rates:")
    print("     bound                     T degF     q_k bbl/min")
    print("  " + "-" * 52)
    print(f"     surface, no heating   {lo['t_inject_degf']:10.2f} {lo['q_k_min_bpm']:15.4f}")
    print(f"     finite U, ESTIMATE    {est['t_inject_degf']:10.2f} {est['q_k_min_bpm']:15.4f}")
    print(f"     U -> infinity, bound  {hi['t_inject_degf']:10.2f} {hi['q_k_min_bpm']:15.4f}")
    print()
    print(f"     gap surface -> estimate  {t['gap_surface_to_estimate']:8.2%}")
    print(f"     gap estimate -> bound    {t['gap_estimate_to_bound']:8.2%}")
    print(f"     gap end to end           {t['gap_relative']:8.2%}")
    print("     (all divided by the smaller value, the project convention)")


def test_open_hole_interval_uses_the_formation_conductivity(examples_dir, capsys):
    """An uncemented interval has no cement: the annular resistance uses k_e over the same
    r_w/r_to, and the result says how much of the string was treated that way."""
    from wellsim.cases.schema import Case

    base = load_case(examples_dir / "rw_wgr.json")
    kill = KillFluid.from_schema(base.dynamic_kill)
    d = base.model_dump()
    d["relief_well"]["string"] = [
        {"md": 8000.0, "tvd": 7000.0, "id": 4.778, "od": 5.5, "cemented": True},
        {"md": 11500.0, "tvd": 10000.0, "id": 4.778, "od": 5.5, "cemented": False},
    ]
    prof = rwm.ramey_arrival_finite_u(Case(**d), kill, 26.1604)

    cemented, open_hole = prof.segments
    assert cemented["k_annulus"] == 0.7 and not cemented["annulus_is_formation"]
    assert open_hole["k_annulus"] == prof.k_e and open_hole["annulus_is_formation"]
    # k_e (1.4) beats k_cem (0.7), so the open interval is better coupled: smaller A
    assert open_hole["a_relax_ft"] < cemented["a_relax_ft"]
    assert any("uncemented" in w and "3,500 ft" in w for w in prof.warnings), prof.warnings
    assert len(prof.rows) == 2 and prof.rows[0]["a_relax_ft"] != prof.rows[1]["a_relax_ft"]

    print("\nopen-hole interval: 3,500 ft of the 11,500 ft string uncemented")
    print(f"  cemented   k_annulus {cemented['k_annulus']:.2f}  U {cemented['u']:.4f}  "
          f"A {cemented['a_relax_ft']:,.0f} ft")
    print(f"  open hole  k_annulus {open_hole['k_annulus']:.2f}  U {open_hole['u']:.4f}  "
          f"A {open_hole['a_relax_ft']:,.0f} ft")
    print(f"  arrival {prof.t_arrival_degf:.2f} degF")
    for w in prof.warnings:
        if "uncemented" in w:
            print(f"  warning: {w}")


# --- rw-3b: the annulus fill, and whether k_annulus is resolvable -------------------------------


def test_annulus_fill_names_the_default_and_k_cem_is_a_deprecated_alias(examples_dir, capsys):
    """rw-3b. 'mud' is the default because on a dynamic kill the relief annulus holds static
    drilling fluid: the kill fluid goes down the string and out at the intersection, so
    nothing returns and there is no circulation. 'cement' is the other choice. The rw-3
    name k_cem still loads, with a warning."""
    from wellsim.cases.schema import Case, ReliefWell

    base = load_case(examples_dir / "rw_wgr.json")
    kill = KillFluid.from_schema(base.dynamic_kill)
    q_k = 26.1604

    assert rwm.RW3_K_ANNULUS_DEFAULT == {"mud": 0.7, "cement": 1.0}

    def k_used(**over):
        d = base.model_dump()
        d["relief_well"].update(over)
        prof = rwm.ramey_arrival_finite_u(Case(**d), kill, q_k)
        return prof.segments[0]["k_annulus"], prof.warnings

    k, w = k_used(k_annulus=None, annulus_fill="mud")
    assert k == 0.7 and any("annulus_fill 'mud' defaults it to 0.7" in x for x in w)
    k, w = k_used(k_annulus=None, annulus_fill="cement")
    assert k == 1.0 and any("annulus_fill 'cement' defaults it to 1" in x for x in w)
    k, w = k_used(k_annulus=0.55)
    assert k == 0.55 and not any("defaults it to" in x for x in w)

    # the deprecated alias still loads and is used, with a warning
    k, w = k_used(k_annulus=None, k_cem=0.9)
    assert k == 0.9 and any("k_cem is a deprecated alias" in x for x in w)
    # and setting both is refused outright rather than guessed
    d = base.model_dump()
    d["relief_well"]["k_cem"] = 0.9
    with pytest.raises(ValueError, match="k_annulus and the deprecated alias k_cem"):
        Case(**d)

    print("\nrw-3b annulus fill: mud -> 0.7, cement -> 1.0 Btu/(hr ft degF), both warned;")
    print("  k_cem still accepted as a deprecated alias, both together refused")


@pytest.mark.slow
def test_k_annulus_over_its_plausible_range(examples_dir, capsys):
    """rw-3b step 2 (COMPUTED). rw-3 put k_annulus on the hand-check list after a sweep over
    0.1 to 100, which is not a range any annulus occupies. Over the PLAUSIBLE range, 0.4 to
    1.2, the kill rate moves less than the search tolerance, so the model cannot resolve
    k_annulus and hand-checking it would buy nothing.

    The kill rate at the shipped 1% tolerance is IDENTICAL at all three values, which is the
    search quantising to its own bracket rather than true zero sensitivity; the converged
    spread at 1e-4 is what the verdict rests on. Slow: three kill searches at each of three
    tolerances."""
    from dataclasses import replace

    from wellsim.cases.schema import Case
    from wellsim.core import kill as km
    from wellsim.core.pvt import collect_range_warnings, make_pvt
    from wellsim.core.solver import build_ipr
    from wellsim.core.vlp.two_phase import LiquidProps

    rows = []
    with collect_range_warnings():
        for k_ann in (0.4, 0.7, 1.2):
            d = load_case(examples_dir / "rw_wgr.json").model_dump()
            d["relief_well"]["k_annulus"] = k_ann
            case = Case(**d)
            pvt = make_pvt(case.fluid, case.options.correlations)
            ipr, _ = build_ipr(case, pvt)
            kill = KillFluid.from_schema(case.dynamic_kill)
            md_inj = km.injection_md(case, None)
            liq = LiquidProps.from_schema(case.liquid)
            prof = rwm.ramey_arrival_finite_u(case, kill, 26.1604)
            k2 = replace(kill, t_inject=prof.t_arrival_degf)
            q_shipped = km.min_kill_rate(case, pvt, ipr, k2, md_inj, liq, q_start=26.1604,
                                         rel_tol=1e-2)["q_k_min_bpm"]
            q_tight = km.min_kill_rate(case, pvt, ipr, k2, md_inj, liq, q_start=26.1604,
                                       rel_tol=1e-4)["q_k_min_bpm"]
            rows.append((k_ann, prof.segments[0]["u"], prof.t_arrival_degf, q_shipped, q_tight))

    t_lo, t_hi = min(r[2] for r in rows), max(r[2] for r in rows)
    q_lo, q_hi = min(r[4] for r in rows), max(r[4] for r in rows)
    spread = (q_hi - q_lo) / q_lo
    assert spread < 0.01, f"k_annulus moves the kill rate {spread:.3%}, outside the tolerance"
    assert t_hi - t_lo > 1.0  # the arrival temperature does move; the kill rate does not care

    print("\nrw-3b step 2: k_annulus over the plausible range only")
    print("   k_annulus         U   arrival degF   q_k @1e-2   q_k @1e-4")
    print("  " + "-" * 62)
    for k_ann, u, t_arr, q_s, q_t in rows:
        print(f"  {k_ann:10.1f} {u:9.4f} {t_arr:14.2f} {q_s:11.4f} {q_t:11.5f}")
    print(f"  arrival T spread {t_hi - t_lo:.2f} degF; converged q_k spread {spread:.4%} "
          f"against a 1% tolerance")
    print("  VERDICT: k_annulus is not resolvable by this model over its plausible range,")
    print("  so it comes OFF the hand-check list.")
