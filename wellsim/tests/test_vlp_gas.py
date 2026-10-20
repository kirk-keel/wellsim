"""Gate G1 for the single-phase gas VLP: the marching integrator against a published
Cullender-Smith worked example (target within 2%, spec section 7), plus internal
consistency checks on the integrator.

Provenance tags as in docs/PROVENANCE.md:
FETCHED  read from the named page or PDF during the M2 session (21 Sep 2026)
MEMORY   recalled, not verified against the publication
COMPUTED derived in the test or from this implementation
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wellsim.core.exit import G_C, gas_mass_rate
from wellsim.core.friction import darcy_friction_factor, reynolds_number
from wellsim.core.geometry import FlowSegment, circular_area
from wellsim.core.pvt import DryGasPVT
from wellsim.core.pvt.base import M_AIR, P_SC, R_PSIA_FT3, T_SC_R
from wellsim.core.temperature import LinearTemperature
from wellsim.core.vlp.march import march

G1 = 0.02


def rel(a, b):
    return abs(a - b) / abs(b)


def tubing(length_ft, id_in, rough_in):
    return FlowSegment(0.0, length_ft, id_in, circular_area(id_in), rough_in, 1.0, f"tubing {id_in} in")


# --- Cullender-Smith worked example ------------------------------------------------
#
# FETCHED: N. Abdul Rahman, "Developing a Mathematical Model for Predicting the Bottom-hole
# Flowing Pressure of a Gas Well", Universiti Teknologi PETRONAS final-year project report
# (utpedia.utp.edu.my/id/eprint/10676), text extracted with pypdf. Section 4.2 "Manual
# calculation", Cullender and Smith method:
#   gamma_g 0.65, Ppc 667 psia, Tpc 366 degR (Standing sweet-gas pseudo-criticals),
#   Ptf 2175 psia, Ttf 118 degF, Twf 216 degF, L 6818 ft (vertical), d 2.441 in,
#   e 0.0023 (used as 0.0023/2.441 in the friction factor, so 0.0023 in absolute),
#   q 6300 Mscf/d, mu_g 0.0162 cp, f 0.0195 (Moody, Jain explicit form).
#   Hand result: Pmp 2419 psia, Pwf 2658 psia (two steps plus Simpson correction).
#   Section 4.3, the same case in the report's VBA spreadsheet: f 0.019133, Pwf 2662 psia,
#   profile (depth ft, T degR, p psia, Z): (0, 578, 2175, 0.7934), (3409, 627, 2421, 0.8429),
#   (6818, 676, 2662, 0.8866).
# Secondary source: the report reproduces the method of Cullender and Smith (1956), which
# was not consulted. The report's Z is a Brill-Beggs style explicit fit and its friction
# factor is the Jain explicit form; this implementation uses Sutton + DAK and Colebrook,
# so agreement tests the whole chain, not only the integrator.

CS_GAMMA, CS_L, CS_D, CS_ROUGH = 0.65, 6818.0, 2.441, 0.0023
CS_Q, CS_PTF, CS_TTF, CS_TWF = 6300.0, 2175.0, 118.0, 216.0
CS_PWF_HAND, CS_PWF_SHEET = 2658.0, 2662.0


@pytest.fixture(scope="module")
def cs_case():
    pvt = DryGasPVT(CS_GAMMA)
    seg = tubing(CS_L, CS_D, CS_ROUGH)
    temp = LinearTemperature(t_res=CS_TWF, t_exit=CS_TTF, md_res=CS_L)
    w = gas_mass_rate(CS_Q, CS_GAMMA)
    return pvt, seg, temp, w


def test_cullender_smith_utp_example_top_down(cs_case):
    pvt, seg, temp, w = cs_case
    r = march(pvt, [seg], temp, w, CS_PTF, 0.0, CS_L)
    assert r.ok
    assert rel(r.p_end, CS_PWF_HAND) < G1
    assert rel(r.p_end, CS_PWF_SHEET) < G1
    # the report's spreadsheet midpoint (3409 ft, 2421 psia) and friction factor
    order = np.argsort(r.md)
    p_mid = float(np.interp(0.5 * CS_L, r.md[order], r.p[order]))
    assert rel(p_mid, 2421.0) < G1
    assert rel(r.f[0], 0.019133) < 0.05  # Colebrook vs Jain explicit, same roughness


def test_cullender_smith_utp_example_bottom_up_round_trip(cs_case):
    # COMPUTED: marching back up from the computed pwf must return the wellhead pressure.
    pvt, seg, temp, w = cs_case
    down = march(pvt, [seg], temp, w, CS_PTF, 0.0, CS_L)
    up = march(pvt, [seg], temp, w, down.p_end, CS_L, 0.0)
    assert up.ok
    assert abs(up.p_end - CS_PTF) < 0.05


def _cullender_smith_two_step(pvt, seg, temp, q_mscfd, p_top, length, n_iter=30):
    """Cullender-Smith (1956) two-interval integration with Simpson correction, with this
    package's PVT and Colebrook friction factor. Form (COMPUTED, derived in the docstring
    of test_cullender_smith_constant_derivation):
        18.75 gamma L = int_{ptf}^{pwf} I dp,  I = (p/TZ) / (0.001 (p/TZ)^2 + F^2),
        F^2 = 0.6716 f_Moody q^2 / d^5,  q in MMscf/d, d in inches.
    Friction factor evaluated at each node's own conditions."""
    q_mm = q_mscfd / 1000.0
    w = gas_mass_rate(q_mscfd, pvt.gas_sg)

    def I(p, t_f):
        st = pvt.state(p, t_f)
        v = w / (st.rho_g * seg.area)
        f = darcy_friction_factor(reynolds_number(st.rho_g, v, seg.d_ft, st.mu_g), seg.rel_rough)
        f2 = 0.6716 * f * q_mm**2 / seg.d_hyd**5
        x = p / ((t_f + 459.67) * st.z)
        return x / (0.001 * x * x + f2)

    rhs = 18.75 * pvt.gas_sg * length
    t_top, t_mid, t_bot = temp.at(0.0), temp.at(0.5 * length), temp.at(length)
    i_top = I(p_top, t_top)
    p_mid = p_top + 0.5 * rhs / i_top
    for _ in range(n_iter):
        i_mid = I(p_mid, t_mid)
        p_new = p_top + rhs / (i_top + i_mid)
        if abs(p_new - p_mid) < 1e-6:
            break
        p_mid = p_new
    p_bot = p_mid + 0.5 * rhs / i_mid
    for _ in range(n_iter):
        i_bot = I(p_bot, t_bot)
        p_new = p_mid + rhs / (i_mid + i_bot)
        if abs(p_new - p_bot) < 1e-6:
            break
        p_bot = p_new
    return p_top + 6.0 * rhs / (i_top + 4.0 * i_mid + i_bot)


def test_march_matches_own_cullender_smith(cs_case):
    # COMPUTED: same PVT and friction factor, so this isolates the integrator (and the
    # dropped acceleration term, Ma ~ 0.01 here) from the correlation chain.
    pvt, seg, temp, w = cs_case
    r = march(pvt, [seg], temp, w, CS_PTF, 0.0, CS_L)
    p_cs = _cullender_smith_two_step(pvt, seg, temp, CS_Q, CS_PTF, CS_L)
    assert rel(r.p_end, p_cs) < 0.003


def test_cullender_smith_constant_derivation():
    # COMPUTED: the 18.75 and the F^2 coefficient used above follow from the momentum
    # balance in this package's units. rho = M_air gamma p / (Z R T); dividing dp/dx by
    # rho gives (p/TZ) dp / [(p/TZ)^2 + f w^2 / (2 g_c d A^2 c^2)] = c dx / 144 with
    # c = M_air / R. The 0.001 factor rescales that to the textbook 18.75 gamma L form.
    c = M_AIR / R_PSIA_FT3
    assert rel(1000.0 * c / 144.0, 18.75) < 0.002
    # F^2 = 0.001 f w^2 / (2 g_c d A^2 c^2) with w for 1 MMscf/d, d = 1 in, gamma = 1
    rho_sc = P_SC * M_AIR / (R_PSIA_FT3 * T_SC_R)
    w = 1e6 / 86400.0 * rho_sc
    d_ft = 1.0 / 12.0
    a = math.pi / 4.0 * d_ft**2
    f2_per_f = 0.001 * w * w / (2.0 * G_C * d_ft * a * a * c * c)
    assert rel(f2_per_f, 0.6716) < 0.002
    # the textbook constant 2.6665 (MEMORY, Lee and Wattenbarger / Ikoku) applies to the
    # Fanning factor: 2.6665 / 4 = 0.6666, within 1% of the derived Moody-form value.
    assert rel(2.6665 / 4.0, f2_per_f) < 0.01


# --- nine field wells from the same report -----------------------------------------
#
# FETCHED: same report, Table 2 "Well data" (stated to come from an SPE paper, not named)
# and Table 3 "Bottom-hole flowing pressure": measured ("Actual") and the report's
# Cullender-Smith spreadsheet value. Columns: depth ft, gas gravity, tubing length ft,
# rate MMscf/d, wellhead pressure psia, wellhead T degF, bottomhole T degF, tubing ID in.
# Roughness is not given for these wells. The Cullender and Smith (1956) standard
# assumption of an absolute roughness of 0.0006 in (MEMORY) is used. The tubing length is
# used as the vertical flow length. The report's actual roughness and Z-factor method for
# these runs are unknown; with 0.0006 in this implementation sits 0.7% to 1.6% below the
# report on all nine wells, the gap growing with pressure, which points to a different
# Z-factor method in the report's spreadsheet. (With the worked example's relative
# roughness of 0.0009 it sat 0.4% to 2.4% above.)
# D40 is excluded from the measured-value summary: see test_utp_well_d40_is_a_data_outlier.
ROUGHNESS_IN = 0.0006
WELLS = [
    # name, depth, gamma, L, q_MM, ptf, ttf, twf, d, pwf_cs, pwf_measured
    ("A13", 5790, 0.60, 5700, 5.153, 2222, 89, 180, 1.9956, 2665, 2659.10),
    ("A27", 5840, 0.63, 5750, 5.181, 2204, 85, 184, 1.9551, 2694, 2693.35),
    ("B37", 6000, 0.70, 5910, 5.954, 2206, 87, 190, 1.7586, 2978, 2955.08),
    ("B25", 5900, 0.69, 5810, 5.966, 2171, 93, 196, 1.7002, 2981, 2969.99),
    ("B63", 6800, 0.71, 6710, 5.450, 2156, 87, 205, 1.8076, 2932, 2909.30),
    ("C18", 7400, 0.73, 7320, 5.838, 2159, 90, 220, 1.6516, 3295, 3273.15),
    ("C19", 8000, 0.74, 7930, 5.835, 2196, 87, 224, 1.9197, 3107, 3074.28),
    ("D23", 8400, 0.68, 8300, 5.354, 2127, 93, 190, 1.5507, 3421, 3407.78),
    ("D40", 6200, 0.67, 6110, 6.094, 2176, 88, 187, 1.7653, 2947, 3023.65),
]


@pytest.mark.parametrize("well", WELLS, ids=[w[0] for w in WELLS])
def test_utp_field_wells_against_report_cullender_smith(well):
    name, depth, gamma, length, q_mm, ptf, ttf, twf, d, pwf_cs, pwf_meas = well
    pvt = DryGasPVT(gamma)
    seg = tubing(length, d, ROUGHNESS_IN)
    temp = LinearTemperature(t_res=twf, t_exit=ttf, md_res=length)
    r = march(pvt, [seg], temp, gas_mass_rate(1000.0 * q_mm, gamma), ptf, 0.0, length)
    assert r.ok
    assert rel(r.p_end, pwf_cs) < G1, f"{name}: {r.p_end:.0f} vs report {pwf_cs}"


def _well_pwf(well):
    name, depth, gamma, length, q_mm, ptf, ttf, twf, d, pwf_cs, pwf_meas = well
    pvt = DryGasPVT(gamma)
    seg = tubing(length, d, ROUGHNESS_IN)
    temp = LinearTemperature(t_res=twf, t_exit=ttf, md_res=length)
    r = march(pvt, [seg], temp, gas_mass_rate(1000.0 * q_mm, gamma), ptf, 0.0, length)
    assert r.ok
    return r.p_end


def test_utp_field_wells_against_measured_summary():
    # FETCHED measured values; the report's own C-S errors run 0.02% to 2.5%. This checks
    # the implementation on the eight consistent wells as a set (D40 excluded, see below).
    errs = [rel(_well_pwf(w), w[10]) for w in WELLS if w[0] != "D40"]
    assert len(errs) == 8
    assert sum(errs) / len(errs) < 0.010
    assert max(errs) < 0.015


def test_utp_well_d40_is_a_data_outlier():
    # D40's measured pressure (3023.65 psia) is inconsistent with its stated inputs: two
    # independent calculations, the report's own Cullender-Smith spreadsheet (2947) and
    # this implementation, both undershoot it by more than 2%, while on the other eight
    # wells the report matches or exceeds the measured value. A liquid-loaded column or a
    # data error in the source table is the likely cause. The well stays in the report
    # comparison above and is excluded from the measured-value summary.
    well = next(w for w in WELLS if w[0] == "D40")
    pwf_cs, pwf_meas = well[9], well[10]
    ours = _well_pwf(well)
    assert (pwf_meas - pwf_cs) / pwf_meas > 0.02
    assert (pwf_meas - ours) / pwf_meas > 0.02
    for w in WELLS:
        if w[0] != "D40":
            assert w[9] >= w[10] * 0.999, f"{w[0]}: report C-S below measured"


# --- integrator behaviour ------------------------------------------------------------

def test_static_column_matches_fine_euler(cs_case):
    # COMPUTED: q = 0 -> dp/dx = rho/144 only; compare with a 1-ft explicit Euler march.
    pvt, seg, temp, _ = cs_case
    r = march(pvt, [seg], temp, 0.0, CS_PTF, 0.0, CS_L)
    p = CS_PTF
    n = int(CS_L)
    for i in range(n):
        x = i + 0.5
        p += pvt.gas_density(p, temp.at(x)) / 144.0
    assert abs(r.p_end - p) < 0.5
    assert r.p_end > CS_PTF


def test_step_size_convergence(cs_case):
    # COMPUTED: halving the maximum step and tightening the tolerance moves the answer by
    # less than the spec's 0.1 psi per-segment convergence target.
    pvt, seg, temp, w = cs_case
    coarse = march(pvt, [seg], temp, w, CS_PTF, 0.0, CS_L, max_step=200.0, p_tol=0.5)
    fine = march(pvt, [seg], temp, w, CS_PTF, 0.0, CS_L, max_step=25.0, p_tol=0.01)
    assert abs(coarse.p_end - fine.p_end) < 0.1
    assert fine.steps > coarse.steps


def test_march_lands_on_conduit_boundaries():
    # COMPUTED: two conduits; the profile must contain the boundary depth exactly and the
    # conduit label must switch there.
    pvt = DryGasPVT(0.65)
    segs = [
        FlowSegment(0.0, 4000.0, 2.441, circular_area(2.441), 0.0018, 1.0, "tubing"),
        FlowSegment(4000.0, 9000.0, 6.184, circular_area(6.184), 0.0018, 1.0, "casing"),
    ]
    temp = LinearTemperature(200.0, 120.0, 9000.0)
    r = march(pvt, segs, temp, gas_mass_rate(20000.0, 0.65), 1000.0, 0.0, 9000.0)
    assert r.ok
    assert any(abs(m - 4000.0) < 1e-6 for m in r.md)
    # one row on each side of the boundary, same md and pressure, different conduit
    rows = [i for i, m in enumerate(r.md) if abs(m - 4000.0) < 1e-6]
    assert len(rows) == 2
    assert r.conduit[rows[0]] == "tubing" and r.conduit[rows[1]] == "casing"
    assert r.p[rows[0]] == r.p[rows[1]]
    dmd, dp = np.diff(r.md), np.diff(r.p)
    assert np.all(dmd >= 0)
    assert np.all(dp[dmd > 0] > 0)


def test_acceleration_term_negligible_at_low_mach_and_large_near_sonic(cs_case):
    # COMPUTED: at the C-S conditions Ma^2 ~ 1e-4, so the acceleration factor changes the
    # gradient by < 0.1%. Near the sonic point the factor 1/(1 - Ma^2) dominates.
    from wellsim.core.vlp.correlations.single_phase_gas import gas_gradient

    pvt, seg, temp, w = cs_case
    g = gas_gradient(pvt, CS_PTF, CS_TTF, w, seg)
    assert g.ek < 1e-3
    assert rel(g.dpdx, g.grav + g.fric) < 1e-3
    # 2.5x the rate at 100 psia in the same tubing: Mach well above 0.5
    g2 = gas_gradient(pvt, 100.0, CS_TTF, 2.5 * w, seg)
    assert 0.5 < g2.mach < 1.0
    assert g2.dpdx > 1.3 * (g2.grav + g2.fric)


def test_upward_march_reports_choke_inside():
    # COMPUTED: marching up from a low bottomhole pressure at a very high rate, the flow
    # reaches Mach 1 before the surface; the march must stop and say where.
    pvt = DryGasPVT(0.65)
    seg = tubing(10000.0, 2.441, 0.0018)
    temp = LinearTemperature(200.0, 150.0, 10000.0)
    r = march(pvt, [seg], temp, gas_mass_rate(60000.0, 0.65), 800.0, 10000.0, 0.0)
    assert not r.ok
    assert r.choked_md is not None and 0.0 < r.choked_md < 10000.0
    assert "Mach 1" in r.message


def test_top_down_march_from_choked_exit_is_regular():
    # COMPUTED: starting exactly at Mach 1 (choked exit) the parametrised march must
    # proceed with Mach decreasing monotonically with depth and finite pressures.
    from wellsim.core.exit import exit_state

    pvt = DryGasPVT(0.65)
    seg = tubing(5000.0, 6.184, 0.0018)
    temp = LinearTemperature(200.0, 180.0, 5000.0)
    q = 300000.0
    ex = exit_state(pvt, q, 14.7, temp.at(0.0), seg.area)
    assert ex.choked and abs(ex.mach - 1.0) < 1e-6
    r = march(pvt, [seg], temp, gas_mass_rate(q, 0.65), ex.p, 0.0, 5000.0)
    assert r.ok
    assert abs(r.mach[0] - 1.0) < 1e-6
    assert np.all(np.diff(r.mach) <= 1e-9)
    assert np.all(np.diff(r.p) > 0.0)
    assert r.p_end < 10000.0
