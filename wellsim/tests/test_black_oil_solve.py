"""Tests for Black Oil IPR (core/ipr/oil.py) and Black Oil end-to-end solve (core/solver_black_oil.py)."""

import pytest

from wellsim.cases.loader import load_case
from wellsim.core.ipr.oil import OilIPR
from wellsim.core.pvt import make_pvt
from wellsim.core.solver import solve


def test_black_oil_ipr():
    case = load_case("wellsim/cases/examples/black_oil_annulus_kill.yaml")
    pvt = make_pvt(case.fluid, case.options.correlations)
    res = case.reservoir

    ipr = OilIPR(
        pvt=pvt,
        p_res=res.pressure,
        t_res=res.temperature,
        k=res.permeability,
        h=res.net_pay,
        r_e=res.drainage_radius,
        r_w=res.wellbore_radius,
        skin=res.skin,
        h_p=res.exposed_interval,
        h_1=res.exposed_top_offset,
        kv_kh=res.kv_kh,
    )

    assert ipr.j > 0.0
    assert ipr.pb > 0.0
    assert ipr.aof() > 0.0

    # At reservoir pressure, rates must be 0
    qo, qw, qg = ipr.rates(res.pressure)
    assert qo == 0.0
    assert qw == 0.0
    assert qg == 0.0

    # Below reservoir pressure, rates must be positive and monotonic with drawdown
    qo_mid, qw_mid, qg_mid = ipr.rates(res.pressure - 500.0)
    qo_low, qw_low, qg_low = ipr.rates(res.pressure - 1500.0)

    assert qo_low > qo_mid > 0.0
    assert qw_low > qw_mid > 0.0
    assert qg_low > qg_mid > 0.0

    # Water cut relationship: qw / (qo + qw) == water_cut
    wc = case.fluid.water_cut
    calculated_wc = qw_mid / (qo_mid + qw_mid)
    assert abs(calculated_wc - wc) < 1e-4


def test_black_oil_solve_end_to_end():
    case = load_case("wellsim/cases/examples/black_oil_annulus_kill.yaml")
    r = solve(case)

    assert r.flowing is True
    assert r.fluid_type == "black_oil"
    assert r.oil_rate > 30000.0  # ~39.8k stb/d
    assert r.gas_rate > 20000.0  # ~23.9 MMscf/d
    assert r.water_rate > 3000.0  # ~4.4k stb/d
    assert r.pwf is not None and 2000.0 < r.pwf < 4000.0
    assert r.exit is not None
    assert r.kill is not None
    assert "static_hold_mud_weight_ppg" in r.kill
    assert 10.0 < r.kill["static_hold_mud_weight_ppg"] < 11.5
