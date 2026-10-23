"""Unit tests for Comparative Well Control Strategy Matrix."""

import pytest

from wellsim.core.kill_comparison import (
    KillMethodStrategy,
    compare_kill_methods,
)
from wellsim.core.well_control import KillSheetInputs


def test_compare_kill_methods_matrix():
    inputs = KillSheetInputs(
        omw_ppg=12.0,
        tvd_well_ft=10000.0,
        tvd_shoe_ft=6500.0,
        lot_emw_ppg=15.5,
        sidpp_psi=500.0,
        sicp_psi=750.0,
        pit_gain_bbl=20.0,
        scrp_psi=400.0,
        drillstring_volume_bbl=150.0,
        annular_volume_bbl=850.0,
        annular_capacity_around_bha_bbl_ft=0.045,
        pump_displacement_bbl_per_stk=0.10,
    )

    comparison = compare_kill_methods(inputs)

    assert len(comparison) == 4
    assert KillMethodStrategy.DRILLERS in comparison
    assert KillMethodStrategy.WAIT_AND_WEIGHT in comparison
    assert KillMethodStrategy.VOLUMETRIC in comparison
    assert KillMethodStrategy.BULLHEADING in comparison

    drillers = comparison[KillMethodStrategy.DRILLERS]
    ww = comparison[KillMethodStrategy.WAIT_AND_WEIGHT]
    vol = comparison[KillMethodStrategy.VOLUMETRIC]
    bull = comparison[KillMethodStrategy.BULLHEADING]

    # Verification 1: Wait & Weight MUST have lower peak casing pressure than Driller's Method
    assert ww.peak_casing_pressure_psi < drillers.peak_casing_pressure_psi

    # Verification 2: Wait & Weight MUST have lower shoe pressure than Driller's Method
    assert ww.max_shoe_pressure_psi < drillers.max_shoe_pressure_psi

    # Verification 3: Circulation count
    assert drillers.circulations_required == 2
    assert ww.circulations_required == 1
    assert vol.circulations_required == 0
    assert bull.circulations_required == 0

    # Verification 4: Shoe integrity
    assert not ww.shoe_fractured
    assert not drillers.shoe_fractured
