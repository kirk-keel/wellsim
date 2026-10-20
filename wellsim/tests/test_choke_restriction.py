"""Internal choke at a conduit narrowing (spec 4.5 extended to restrictions) on the
parted-tubing example, plus the march safety net. All COMPUTED."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from wellsim.cases.loader import case_from_dict, case_to_dict, load_case
from wellsim.core.exit import gas_mass_rate, sonic_pressure
from wellsim.core.geometry import FlowSegment, circular_area
from wellsim.core.pvt import DryGasPVT
from wellsim.core.solver import solve
from wellsim.core.temperature import LinearTemperature
from wellsim.core.vlp.march import SupersonicError, march

EXAMPLES = Path(__file__).resolve().parents[1] / "cases" / "examples"


@pytest.fixture(scope="module")
def parted():
    case = load_case(EXAMPLES / "dry_gas_parted_tubing.yaml")
    return case, solve(case)


def test_parted_tubing_chokes_at_500_ft(parted):
    case, r = parted
    p = r.profile
    assert r.flowing and p.ok
    # Mach never exceeds 1 anywhere (round-off at the two sonic points only)
    assert np.all(p.mach <= 1.0 + 1e-9)
    # choke location reported at the tubing top
    assert r.choke_md == [500.0]
    c = p.chokes[0]
    assert c.md == 500.0 and c.conduit.startswith("casing 2.992")
    assert c.mach_before > 1.0 and c.p_after > c.p_before
    assert any("restriction at md 500.0 ft" in w for w in r.warnings)
    # the first tubing point is exactly sonic and at the recorded sonic pressure
    i = p.conduit.index(c.conduit)
    assert p.md[i] == 500.0
    assert abs(p.mach[i] - 1.0) < 1e-6
    assert abs(p.p[i] - c.p_after) < 1e-9
    assert abs(p.p[i - 1] - c.p_before) < 1e-9


def test_parted_tubing_no_silent_pressure_jump(parted):
    # every pressure discontinuity between consecutive profile points is a recorded choke
    _, r = parted
    p = r.profile
    recorded = {(round(c.md, 6), round(c.p_before, 6), round(c.p_after, 6)) for c in p.chokes}
    for i in range(1, len(p.md)):
        if p.md[i] == p.md[i - 1] and abs(p.p[i] - p.p[i - 1]) > 1e-9:
            key = (round(p.md[i], 6), round(p.p[i - 1], 6), round(p.p[i], 6))
            assert key in recorded, f"unrecorded pressure jump at md {p.md[i]}"
    # and elsewhere the pressure rises smoothly with depth
    dp = np.diff(p.p)
    dmd = np.diff(p.md)
    assert np.all(dp[dmd > 0] > 0.0)


def test_parted_tubing_exit_and_restriction_both_choked(parted):
    _, r = parted
    assert r.exit.choked and abs(r.exit.mach - 1.0) < 1e-6
    assert any("exit is choked as well" in w for w in r.warnings)
    assert r.profile.mach[0] == pytest.approx(1.0, abs=1e-9)


def test_rate_invariant_to_geometry_above_the_choke(parted):
    # once the flow chokes at 500 ft, the casing above cannot affect the rate
    case, r = parted
    d = case_to_dict(case)
    d["well"]["hole"][0]["id"] = 12.415
    r2 = solve(case_from_dict(d))
    assert r2.choke_md == [500.0]
    assert abs(r2.gas_rate - r.gas_rate) / r.gas_rate < 1e-3
    assert abs(r2.pwf - r.pwf) < 1.0
    # the casing pressures above the choke do differ
    assert r2.exit.p < r.exit.p


def test_no_choke_when_narrowing_is_mild(parted):
    # the big-bore liner example narrows from 8.681 to 6 in and stays subsonic
    r = solve(load_case(EXAMPLES / "dry_gas_big_bore_liner.yaml"))
    assert r.choke_md == []
    assert r.profile.chokes == []
    assert not any("restriction" in w for w in r.warnings)


def test_march_sonic_reset_matches_sonic_pressure():
    # direct march check: p just below the narrowing equals sonic_pressure of the tubing
    pvt = DryGasPVT(0.65)
    segs = [
        FlowSegment(0.0, 500.0, 8.681, circular_area(8.681), 0.0018, 1.0, "casing"),
        FlowSegment(500.0, 6000.0, 2.992, circular_area(2.992), 0.0018, 1.0, "tubing"),
    ]
    temp = LinearTemperature(250.0, 240.0, 6000.0)
    w = gas_mass_rate(100000.0, 0.65)
    r = march(pvt, segs, temp, w, 100.0, 0.0, 6000.0)  # Mach ~0.3 in the casing, ~2.5 in the tubing
    assert r.ok and len(r.chokes) == 1
    c = r.chokes[0]
    assert c.p_after == pytest.approx(sonic_pressure(pvt, w, temp.at(500.0), segs[1].area, c.p_before), rel=1e-9)
    # the profile rows are ordered and there is exactly one row pair at 500 ft
    assert list(r.md) == sorted(r.md)
    assert int(np.sum(r.md == 500.0)) == 2


def test_step_size_resets_at_conduit_boundary():
    # no cluster of near-duplicate points after a boundary (the old artefact)
    r = solve(load_case(EXAMPLES / "dry_gas_big_bore_liner.yaml"))
    md = r.profile.md
    assert int(np.sum(np.abs(md - 10000.0) < 1e-3)) == 2  # last casing row, first open-hole row
    assert np.all(np.diff(md) >= 0.0)


def test_supersonic_safety_net():
    # a downward march started above Mach 1 with no narrowing to blame must raise
    pvt = DryGasPVT(0.65)
    seg = FlowSegment(0.0, 1000.0, 2.992, circular_area(2.992), 0.0018, 1.0, "tubing")
    temp = LinearTemperature(200.0, 150.0, 1000.0)
    w = gas_mass_rate(100000.0, 0.65)
    with pytest.raises(SupersonicError):
        march(pvt, [seg], temp, w, 50.0, 0.0, 1000.0)
