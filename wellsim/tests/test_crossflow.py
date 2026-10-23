"""Unit tests for Underground Blowout & Inter-Zonal Cross-Flow Engine."""

import pytest

from wellsim.core.crossflow import (
    FormationZone,
    UndergroundBlowoutScenario,
    solve_underground_blowout,
)


def test_underground_blowout_equilibrium():
    # Source: 12,000 ft TVD, 8,500 psia, PI = 15 bbl/d/psi
    source = FormationZone(
        name="Deep Overpressured Gas/Condensate",
        depth_tvd_ft=12000.0,
        pressure_psia=8500.0,
        pi_bbl_d_psi=15.0,
    )
    # Thief: 7,000 ft TVD, 3,800 psia (depleted sand / broken shoe), II = 20 bbl/d/psi
    thief = FormationZone(
        name="Shallow Thief Sand",
        depth_tvd_ft=7000.0,
        pressure_psia=3800.0,
        pi_bbl_d_psi=20.0,
    )
    # 8.5" casing inter-zonal conduit
    scenario = UndergroundBlowoutScenario(
        source_zone=source,
        thief_zone=thief,
        conduit_id_in=8.5,
        fluid_density_ppg=9.0,
        fluid_pv_cp=18.0,
        fluid_yp_lbf_100ft2=12.0,
    )

    res = solve_underground_blowout(scenario, kill_mud_weight_ppg=15.0)

    # Cross-flow must be active
    assert res.is_crossflowing is True
    assert res.crossflow_rate_bbl_d > 5000.0
    assert res.crossflow_rate_gpm > 100.0

    # Hydraulic balance: Pwf_source - Hyd - Fric should equal Pwf_thief
    p_downhole_balance = res.pwf_source_psia - res.interzonal_hydrostatic_loss_psi - res.interzonal_friction_loss_psi
    assert abs(p_downhole_balance - res.pwf_thief_psia) < 1.0

    # Relief well kill rate must be sized
    assert res.min_relief_kill_rate_gpm > 0.0


def test_no_crossflow_when_balanced():
    # Source has insufficient pressure to overcome thief pressure + hydrostatic
    source = FormationZone("Normal Source", depth_tvd_ft=10000.0, pressure_psia=4800.0, pi_bbl_d_psi=10.0)
    thief = FormationZone("Charged Thief", depth_tvd_ft=6000.0, pressure_psia=4000.0, pi_bbl_d_psi=10.0)

    scenario = UndergroundBlowoutScenario(
        source_zone=source,
        thief_zone=thief,
        conduit_id_in=8.5,
        fluid_density_ppg=10.0,
    )

    # 4000 ft of 10 ppg mud = 2080 psi. 4800 - 2080 = 2720 < 4000 (No cross-flow!)
    res = solve_underground_blowout(scenario)
    assert res.is_crossflowing is False
    assert res.crossflow_rate_bbl_d == 0.0
