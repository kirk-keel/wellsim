"""Exit discharge coefficient (cd) and the entry loss into an exit restriction.

COMPUTED throughout; the Crane TP-410 orifice guidance quoted in the warning is MEMORY.
"""

from __future__ import annotations

import math
import warnings

import pytest

from wellsim.cases.loader import case_from_dict, case_to_dict, load_case
from wellsim.core.exit import exit_state_adiabatic
from wellsim.core.geometry import exit_area, flow_path_segments
from wellsim.core.pvt import make_pvt
from wellsim.core.pvt.base import M_AIR, RangeWarning
from wellsim.core.pvt.real_gas_thermo import J_FT_LBF_PER_BTU, enthalpy
from wellsim.core.solver import solve
from wellsim.tests.conftest import EXAMPLES_DIR


def rel(a, b):
    return abs(a - b) / abs(b)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        yield


def _case(name, **exit_over):
    d = case_to_dict(load_case(EXAMPLES_DIR / f"{name}.yaml"))
    d["exit"].update(exit_over)
    return d


def _raw_case(name, **exit_over):
    """The YAML mapping itself (so fields the file does not set stay unset, as in real use)."""
    import yaml

    with open(EXAMPLES_DIR / f"{name}.yaml", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    d.setdefault("exit", {}).update(exit_over)
    return d


def test_cd_default_reproduces_3ac3565():
    # COMPUTED: cd = 1.0 with an open-pipe exit changes nothing; values pinned from 3ac3565.
    pinned = {
        "dry_gas_vertical": (601242.6221193078, 4016.686577131023, 183.1802659730422),
        "dry_gas_big_bore_liner": (623584.0561871971, 7677.189757996649, 190.24852768562477),
        "dry_gas_parted_tubing": (73845.97817383775, 7969.317836790716, 22.53751140117531),
    }
    for name, (q, pwf, pex) in pinned.items():
        r = solve(case_from_dict(_raw_case(name)))
        assert rel(r.gas_rate, q) < 1e-12 and rel(r.pwf, pwf) < 1e-12 and rel(r.exit.p, pex) < 1e-12
        assert r.exit.cd == 1.0 and r.exit.area == r.exit.area_geometric and r.exit_entry is None
        assert not any(w.startswith("restricted exit with cd") for w in r.warnings)  # open pipe: no warning


def test_choked_rate_scales_linearly_with_cd_at_fixed_stagnation_state():
    # COMPUTED: at a fixed total enthalpy and a fixed choked exit pressure the mass flux
    # through the effective area is fixed, so the rate is proportional to cd (1e-6).
    c = case_from_dict(_case("dry_gas_vertical"))
    res = c.reservoir
    pvt = make_pvt(c.fluid, c.options.correlations)
    segs = flow_path_segments(c.well, c.flow_path, res.depth_md)
    a_geom = exit_area(segs, 1.0)
    m = M_AIR * pvt.gas_sg
    ht = enthalpy(pvt, res.pressure, res.temperature) / m - c.well.md_to_tvd(res.depth_md) / J_FT_LBF_PER_BTU
    p_target = 4000.0

    def q_for(cd):
        lo, hi = 1.0, 2.0e6
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            ex = exit_state_adiabatic(pvt, mid, 14.7, cd * a_geom, ht, res.temperature - 40.0, p_max=res.pressure)
            if ex.not_evaluable is not None or ex.p > p_target:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    q1 = q_for(1.0)
    for cd in (0.8, 0.6, 0.3):
        assert rel(q_for(cd) / q1, cd) < 1e-6, cd


def test_cd_sweep_on_1in_exit_and_warning():
    # COMPUTED: dry_gas_vertical with exit_id 1.0 at cd 1.0 (Crane entry loss applied),
    # 0.8 and 0.6 (loss carried by cd): rate falls with cd, the exit is choked, the warning
    # fires only when cd is not supplied, and the entry step is reported.
    out = {}
    for cd, supply in ((1.0, False), (1.0, True), (0.8, True), (0.6, True)):
        over = {"exit_id": 1.0}
        if supply:
            over["cd"] = cd
        r = solve(case_from_dict(_raw_case("dry_gas_vertical", **over)))
        assert r.flowing and r.exit.choked
        assert r.exit.cd == cd and rel(r.exit.area, cd * r.exit.area_geometric) < 1e-12
        warned = any(w.startswith("restricted exit with cd = 1.0") for w in r.warnings)
        assert warned == (not supply)
        assert r.exit_entry is not None
        if cd == 1.0:
            assert r.exit_entry["k_c"] == pytest.approx(0.5 * (1.0 - (1.0 / 8.681) ** 2))
            assert r.exit_entry["loss_psi"] > 0.0
        else:
            assert r.exit_entry["k_c"] == 0.0 and r.exit_entry["loss_psi"] == 0.0
        assert abs(r.exit_entry["residual"]) < 1e-6
        assert r.exit_entry["p_conduit"] > r.exit_entry["p_throat"]  # deceleration recovers pressure
        assert any(n.startswith("exit restriction entry") for n in r.notes)
        out[(cd, supply)] = r.gas_rate
    assert out[(1.0, False)] == out[(1.0, True)]
    assert out[(1.0, True)] > out[(0.8, True)] > out[(0.6, True)]


def test_cd_below_range_rejected():
    with pytest.raises(ValueError):
        case_from_dict(_case("dry_gas_vertical", exit_id=1.0, cd=0.2))
    with pytest.raises(ValueError):
        case_from_dict(_case("dry_gas_vertical", cd=1.2))
