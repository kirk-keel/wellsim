"""Unit tests for IADC WellSharp well control calculations (core/well_control.py).

Every test value is verified against standard petroleum engineering formulas
and worked examples from IADC WellSharp and IWCF well control handbooks.
"""

import pytest

from wellsim.core.well_control import (
    KillSheetInputs,
    calculate_kill_sheet,
    equivalent_mud_weight,
    final_circulating_pressure,
    formation_pressure,
    hydrostatic_pressure,
    identify_influx_type,
    influx_gradient,
    influx_height,
    initial_circulating_pressure,
    kick_tolerance_volume,
    kill_mud_weight,
    maasp,
    pressure_gradient,
)


def test_hydrostatic_and_gradients():
    # 10.0 ppg at 10,000 ft TVD -> 5200.0 psi
    assert hydrostatic_pressure(10.0, 10000.0) == 5200.0

    # 12.5 ppg gradient = 0.052 * 12.5 = 0.65 psi/ft
    assert abs(pressure_gradient(12.5) - 0.65) < 1e-6

    # EMW for 5200 psi at 10,000 ft = 10.0 ppg
    assert abs(equivalent_mud_weight(5200.0, 10000.0) - 10.0) < 1e-6


def test_formation_pressure():
    # TVD 8,000 ft, OMW 9.6 ppg, SIDPP 450 psi
    # P_hyd = 0.052 * 9.6 * 8000 = 3993.6 psi
    # P_form = 3993.6 + 450 = 4443.6 psi
    assert abs(formation_pressure(450.0, 9.6, 8000.0) - 4443.6) < 1e-6


def test_kill_mud_weight():
    # TVD 10,000 ft, OMW 10.0 ppg, SIDPP 520 psi
    # KMW = 10.0 + 520 / (0.052 * 10000) = 11.0 ppg
    kmw = kill_mud_weight(10.0, 520.0, 10000.0)
    assert abs(kmw - 11.0) < 1e-6

    # With safety margin 0.2 ppg
    kmw_safe = kill_mud_weight(10.0, 520.0, 10000.0, safety_margin_ppg=0.2)
    assert abs(kmw_safe - 11.2) < 1e-6

    # Rounding up to 0.1 ppg: 10.0 + 350 / (0.052 * 9200) = 10.0 + 0.7316 = 10.7316 -> 10.8 ppg
    kmw_rounded = kill_mud_weight(10.0, 350.0, 9200.0, round_to=0.1)
    assert abs(kmw_rounded - 10.8) < 1e-6


def test_maasp():
    # LOT 14.0 ppg, Current MW 10.0 ppg, Shoe TVD 6,000 ft
    # MAASP = (14.0 - 10.0) * 0.052 * 6000 = 1248.0 psi
    assert abs(maasp(14.0, 10.0, 6000.0) - 1248.0) < 1e-6

    # With KMW 11.0 ppg
    assert abs(maasp(14.0, 11.0, 6000.0) - 936.0) < 1e-6


def test_circulating_pressures():
    # SIDPP 520, SCRP 600 -> ICP = 1120 psi
    assert initial_circulating_pressure(520.0, 600.0) == 1120.0

    # SCRP 600, KMW 11.0, OMW 10.0 -> FCP = 660 psi
    assert abs(final_circulating_pressure(600.0, 11.0, 10.0) - 660.0) < 1e-6


def test_influx_analysis():
    # 20 bbl pit gain in 0.040 bbl/ft annular capacity -> 500 ft height
    h = influx_height(20.0, 0.040)
    assert abs(h - 500.0) < 1e-6

    # OMW 10.0 ppg (grad 0.52), SIDPP 520, SICP 750 (delta 230), h 500 ft
    # G_inf = 0.52 - 230/500 = 0.52 - 0.46 = 0.06 psi/ft -> Gas
    grad = influx_gradient(10.0, 520.0, 750.0, h)
    assert abs(grad - 0.06) < 1e-6
    assert identify_influx_type(grad) == "gas"

    # Oil kick test: delta P = 80 psi across 500 ft -> G = 0.52 - 0.16 = 0.36 psi/ft -> Oil
    grad_oil = influx_gradient(10.0, 520.0, 600.0, h)
    assert abs(grad_oil - 0.36) < 1e-6
    assert identify_influx_type(grad_oil) == "oil"

    # Water kick test: delta P = 30 psi across 500 ft -> G = 0.52 - 0.06 = 0.46 psi/ft -> Water
    grad_water = influx_gradient(10.0, 520.0, 550.0, h)
    assert abs(grad_water - 0.46) < 1e-6
    assert identify_influx_type(grad_water) == "water"


def test_kick_tolerance():
    # LOT 14 ppg, OMW 10 ppg, Shoe 6000 ft -> MAASP 1248 psi
    # SICP 750 psi -> margin 498 psi
    # gas grad 0.10 psi/ft, mud grad 0.52 psi/ft -> grad diff 0.42 psi/ft
    # max height = 498 / 0.42 = 1185.71 ft
    # max vol = 1185.71 * 0.05 bbl/ft = 59.28 bbl
    v_max = kick_tolerance_volume(14.0, 10.0, 6000.0, 750.0, gas_gradient_psi_ft=0.10, annular_capacity_bbl_ft=0.05)
    assert abs(v_max - 59.2857) < 0.01


def test_complete_kill_sheet():
    inputs = KillSheetInputs(
        omw_ppg=10.0,
        tvd_well_ft=10000.0,
        tvd_shoe_ft=6000.0,
        lot_emw_ppg=14.0,
        sidpp_psi=520.0,
        sicp_psi=750.0,
        pit_gain_bbl=20.0,
        scrp_psi=600.0,
        drillstring_volume_bbl=150.0,
        annular_volume_bbl=850.0,
        annular_capacity_around_bha_bbl_ft=0.040,
        pump_displacement_bbl_per_stk=0.10,
        step_down_interval_strokes=500,
    )

    res = calculate_kill_sheet(inputs)

    assert res.formation_pressure_psi == 5720.0
    assert res.kmw_ppg == 11.0
    assert res.kmw_rounded_ppg == 11.0
    assert res.icp_psi == 1120.0
    assert res.fcp_psi == 660.0
    assert res.maasp_initial_psi == 1248.0
    assert res.maasp_with_kmw_psi == 936.0
    assert res.influx_height_ft == 500.0
    assert res.influx_gradient_psi_ft == 0.06
    assert res.influx_type == "gas"
    assert res.strokes_surface_to_bit == 1500
    assert res.strokes_bit_to_surface == 8500
    assert res.total_kill_strokes == 10000

    # Step-down schedule check
    assert res.step_down.icp == 1120.0
    assert res.step_down.fcp == 660.0
    assert res.step_down.dpp_at_stroke(0) == 1120.0
    assert res.step_down.dpp_at_stroke(750) == 890.0
    assert res.step_down.dpp_at_stroke(1500) == 660.0
    assert res.step_down.dpp_at_stroke(2000) == 660.0  # past bit holds FCP

    d = res.as_dict()
    assert d["kmw_ppg"] == 11.0
    assert d["influx_type"] == "gas"


def test_casing_pressure_exceeds_maasp_warning():
    inputs = KillSheetInputs(
        omw_ppg=10.0,
        tvd_well_ft=10000.0,
        tvd_shoe_ft=6000.0,
        lot_emw_ppg=11.0,  # low LOT -> initial MAASP = (11-10)*0.052*6000 = 312 psi
        sidpp_psi=200.0,
        sicp_psi=500.0,   # exceeds 312 psi
        pit_gain_bbl=20.0,
        scrp_psi=400.0,
        drillstring_volume_bbl=100.0,
        annular_volume_bbl=500.0,
        annular_capacity_around_bha_bbl_ft=0.040,
        pump_displacement_bbl_per_stk=0.10,
    )
    res = calculate_kill_sheet(inputs)
    assert len(res.warnings) == 1
    assert "exceeds initial MAASP" in res.warnings[0]
