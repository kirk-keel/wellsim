"""End-to-end ``solve(case)`` on the dry-gas example: a pure function returning a result
that carries its inputs, correlation choices, warnings and profiles (spec 1, 5).
All COMPUTED: no published blowout-rate case is wired in yet (gate G3 is M4)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from wellsim.cases.loader import case_from_dict, case_to_dict, load_case
from wellsim.core.solver import Result, solve


EXAMPLES = Path(__file__).resolve().parents[1] / "cases" / "examples"


@pytest.fixture(scope="module")
def example_result():
    case = load_case(EXAMPLES / "dry_gas_vertical.yaml")
    return case, solve(case)


def test_example_flows_and_is_choked(example_result):
    case, r = example_result
    assert isinstance(r, Result)
    assert r.flowing and r.nodal.status == "ok"
    assert r.gas_rate > 0.0
    assert 14.7 < r.pwf < case.reservoir.pressure
    assert abs(r.drawdown - (case.reservoir.pressure - r.pwf)) < 1e-9
    assert r.exit.choked
    assert r.exit.p > case.exit.ambient_pressure
    assert abs(r.exit.mach - 1.0) < 1e-6
    assert any("choked" in w for w in r.warnings)


def test_example_profile_is_consistent(example_result):
    case, r = example_result
    p = r.profile
    assert p.ok
    assert p.md[0] == 0.0 and abs(p.md[-1] - case.reservoir.depth_md) < 1e-6
    assert abs(p.p[0] - r.exit.p) < 1e-9
    assert abs(p.p[-1] - r.pwf) < 1e-9
    dmd, dp = np.diff(p.md), np.diff(p.p)
    assert np.all(dmd >= 0.0)  # one row each side of the casing shoe shares its md
    assert np.all(dp[dmd > 0.0] > 0.0)
    # M2.5-4: the unchoked narrowing (a flow-direction expansion) carries a Borda-Carnot
    # pressure step, recorded on the profile; the rows sharing that md differ by exactly it
    acs = r.profile.area_changes
    assert len(acs) == 1 and not acs[0].choked
    jumps = dp[dmd == 0.0]
    assert len(jumps) == 1 and abs(jumps[0] - (acs[0].p_after - acs[0].p_before)) < 1e-9
    assert abs(jumps[0]) < 5.0  # 8.681 to 8.5 in: a small recovery
    assert r.choke_md == []
    # Mach falls with depth within each conduit; it steps up where the hole narrows from
    # the 8.681 in casing to the 8.5 in open hole
    for name in set(p.conduit):
        idx = [i for i, c in enumerate(p.conduit) if c == name]
        assert np.all(np.diff(p.mach[idx]) <= 1e-9), name
    i_switch = p.conduit.index(p.conduit[-1])
    assert p.mach[i_switch] > p.mach[i_switch - 1]
    # adiabatic default (M2.5-3b): the exit temperature is an outcome, well below reservoir
    assert p.t[0] == r.exit.t and p.t[0] < case.reservoir.temperature - 10.0
    assert p.t[-1] > p.t[0] and abs(p.t[-1] - r.temperature["sandface_t_march"]) < 1e-9
    assert p.conduit[0].startswith("casing") and p.conduit[-1].startswith("open_hole")


def test_operating_point_lies_on_both_curves(example_result):
    case, r = example_result
    op = r.nodal.operating
    assert op.stable
    assert abs(op.q - r.gas_rate) < 1e-9
    # the IPR pwf at the operating rate and the VLP pwf from the final march agree
    assert abs(op.pwf - r.pwf) < 10.0 * case.options.numerics.vlp_pressure_tol
    # scan curves bracket the operating point (pwf-space scan lists rates in decreasing order)
    qs = np.array(r.nodal.q_scan)
    assert qs.min() < op.q < qs.max()


def test_result_is_pure_and_carries_inputs(example_result):
    case, r = example_result
    before = case_to_dict(case)
    r2 = solve(case)
    assert case_to_dict(case) == before
    assert r2.gas_rate == r.gas_rate and r2.pwf == r.pwf
    assert r.inputs == before
    assert r.code_version
    assert r.correlations["vlp"] == "single_phase_gas"
    assert r.correlations["ipr"] == "gas_pseudo_pressure_pss"
    assert r.ipr["beta_source"] == "jones"


def test_result_serialises(example_result):
    _, r = example_result
    d = r.as_dict()
    s = json.dumps(d)
    assert "gas_rate_mscfd" in d and "profile" in d and "nodal" in d
    assert len(s) > 1000


def test_range_warnings_collected(example_result):
    # the choked exit is far below the DAK and LGE pressure ranges; the extrapolation
    # must be reported, not silent (spec section 6)
    _, r = example_result
    assert any("Dranchuk" in w or "Lee-Gonzalez" in w for w in r.range_warnings)


@pytest.mark.perf
def test_example_runs_fast(example_result):
    _, r = example_result
    assert r.elapsed_s < 1.0  # spec section 6: single case under 1 s


def test_unchoked_case(gas_case_dict):
    # a tight reservoir in small tubing: rate too low to choke at 14.7 psia
    gas_case_dict["reservoir"]["permeability"] = 0.5
    gas_case_dict["reservoir"]["pressure"] = 3000.0
    gas_case_dict["well"]["hole"][0]["id"] = 6.184
    r = solve(case_from_dict(gas_case_dict))
    assert r.flowing
    assert not r.exit.choked
    assert r.exit.p == 14.7
    assert r.profile.p[0] == 14.7


def test_dead_well_when_ambient_exceeds_reservoir(gas_case_dict):
    gas_case_dict["exit"] = {"ambient_pressure": 6000.0}
    r = solve(case_from_dict(gas_case_dict))
    assert not r.flowing
    assert r.gas_rate == 0.0 and r.pwf is None
    assert r.nodal.status == "no_flow"


def test_non_darcy_override_and_beta_choice(gas_case_dict):
    base = solve(case_from_dict(gas_case_dict))
    d = copy.deepcopy(gas_case_dict)
    d["reservoir"]["non_darcy_beta"] = 1e12  # enormous turbulence: rate must drop
    r = solve(case_from_dict(d))
    assert r.ipr["beta_source"] == "user"
    assert r.gas_rate < base.gas_rate
    d2 = copy.deepcopy(gas_case_dict)
    d2["reservoir"]["porosity"] = 0.2
    d2["options"] = {"correlations": {"non_darcy_beta": "geertsma"}}
    r2 = solve(case_from_dict(d2))
    assert r2.ipr["beta_source"] == "geertsma" and r2.flowing


def test_static_warnings(gas_case_dict):
    gas_case_dict["exit"] = {"exit_id": 3.0}
    r = solve(case_from_dict(gas_case_dict))
    assert any("exit_id" in w for w in r.warnings)


def test_partial_penetration_note_and_kv_kh_warning(gas_case_dict):
    # COMPUTED: a partial interval without kv_kh reports the skin as a note and warns
    # about the 0.1 assumption; the retired "overstated" warning is gone.
    gas_case_dict["reservoir"]["exposed_interval"] = 20.0  # net_pay is 50
    r = solve(case_from_dict(gas_case_dict))
    notes = [n for n in r.notes if n.startswith("partial penetration")]
    assert len(notes) == 1
    assert "h_pD = 0.400" in notes[0] and "kv/kh = 0.1" in notes[0]
    assert f"s_pp = {r.ipr['skin_partial_penetration']:.2f}" in notes[0]
    assert r.ipr["kv_kh_defaulted"] is True
    kv = [w for w in r.warnings if w.startswith("kv_kh not given")]
    assert len(kv) == 1
    assert not any("rate is overstated" in w for w in r.warnings)  # the retired M2 warning
    assert not any("h_pD" in w and "below 0.05" in w for w in r.warnings)
    assert r.as_dict()["notes"] == r.notes


def test_partial_penetration_explicit_kv_kh_and_offset(gas_case_dict):
    # COMPUTED: kv_kh given -> no assumption warning; the note reports the offset; a
    # bottom-placed interval gives the same skin as the top-placed one (symmetry).
    gas_case_dict["reservoir"]["exposed_interval"] = 20.0
    gas_case_dict["reservoir"]["kv_kh"] = 0.25
    top = solve(case_from_dict(gas_case_dict))
    gas_case_dict["reservoir"]["exposed_top_offset"] = 30.0
    bottom = solve(case_from_dict(gas_case_dict))
    for r in (top, bottom):
        assert not any(w.startswith("kv_kh not given") for w in r.warnings)
        assert r.ipr["kv_kh"] == 0.25 and r.ipr["kv_kh_defaulted"] is False
    pp = [n for n in bottom.notes if n.startswith("partial penetration")]
    assert len(pp) == 1 and "top offset 30.0 ft" in pp[0]
    assert abs(top.ipr["skin_partial_penetration"] - bottom.ipr["skin_partial_penetration"]) < 1e-10
    assert abs(top.gas_rate - bottom.gas_rate) / top.gas_rate < 1e-9


def test_partial_penetration_thin_interval_warns(gas_case_dict):
    # COMPUTED: h_pD = 2 / 50 = 0.04 < 0.05 fires the thin-interval warning.
    gas_case_dict["reservoir"]["exposed_interval"] = 2.0
    gas_case_dict["reservoir"]["kv_kh"] = 0.1
    r = solve(case_from_dict(gas_case_dict))
    hits = [w for w in r.warnings if "below 0.05" in w]
    assert len(hits) == 1 and "h_pD = 0.040" in hits[0]


def test_partial_penetration_silent_at_full_pay(gas_case_dict):
    for value in (None, 50.0):
        d = copy.deepcopy(gas_case_dict)
        if value is not None:
            d["reservoir"]["exposed_interval"] = value
        r = solve(case_from_dict(d))
        assert not any(w.startswith("partial penetration") for w in r.warnings)
        assert not any(w.startswith("kv_kh not given") for w in r.warnings)
        assert not any(n.startswith("partial penetration") for n in r.notes)
        assert r.ipr["skin_partial_penetration"] == 0.0
        assert r.ipr["skin_total"] == r.ipr["skin"]


def test_open_interval_must_fit_in_net_pay(gas_case_dict):
    gas_case_dict["reservoir"]["exposed_interval"] = 20.0
    gas_case_dict["reservoir"]["exposed_top_offset"] = 31.0  # 31 + 20 > 50
    with pytest.raises(ValueError):
        case_from_dict(gas_case_dict)


def test_default_exit_temperature_warning(gas_case_dict):
    # 'profile' mode (M2.5-3b): the adiabatic default has no exit temperature to default
    gas_case_dict["temperature"] = {"model": "profile"}
    r = solve(case_from_dict(gas_case_dict))  # no exit_temperature: defaulted
    hits = [w for w in r.warnings if w.startswith("exit temperature defaulted")]
    assert len(hits) == 1
    assert "190.0 degF" in hits[0] and "2 to 6%" in hits[0]
    d = copy.deepcopy(gas_case_dict)
    d["temperature"] = {"exit_temperature": 150.0}
    r2 = solve(case_from_dict(d))
    assert not any(w.startswith("exit temperature defaulted") for w in r2.warnings)
    assert not any("exceeds reservoir temperature" in w for w in r2.warnings)


def test_exit_temperature_above_reservoir_warns(gas_case_dict):
    gas_case_dict["temperature"] = {"model": "profile", "exit_temperature": 210.0}  # reservoir is 200
    r = solve(case_from_dict(gas_case_dict))
    assert any("exceeds reservoir temperature 200.0 degF" in w for w in r.warnings)


def test_dual_nodal_convergence_pressure_residual(gas_case_dict):
    # the IPR and final-march VLP pressures at the operating point agree within pressure_tol
    gas_case_dict["reservoir"]["permeability"] = 2000.0  # steep IPR: rate_tol alone leaves psi-level residuals
    for ptol in (0.5, 0.05):
        d = copy.deepcopy(gas_case_dict)
        d["options"] = {"numerics": {"pressure_tol": ptol}}
        r = solve(case_from_dict(d))
        assert abs(r.pwf - r.nodal.operating.pwf) <= ptol
        assert not any(w.startswith("IPR and VLP pwf differ") for w in r.warnings)


def test_black_oil_not_implemented(examples_dir):
    case = load_case(examples_dir / "black_oil_annulus_kill.yaml")
    with pytest.raises(NotImplementedError):
        solve(case)
