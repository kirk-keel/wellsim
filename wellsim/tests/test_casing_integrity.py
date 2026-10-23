"""Unit tests for Casing & Drillstring Structural Mechanical Integrity (API Spec 5CT)."""

import pytest

from wellsim.core.casing_integrity import (
    CasingGrade,
    CasingSection,
    calculate_casing_integrity,
)


def test_api_burst_and_collapse_ratings():
    # 9-5/8" 47# L-80 Casing: OD = 9.625", wall thickness = 0.472"
    csg_l80 = CasingSection(
        top_md_ft=0.0,
        bottom_md_ft=10000.0,
        od_in=9.625,
        weight_lbf_ft=47.0,
        wall_thickness_in=0.472,
        grade_yield_psi=CasingGrade.L80,
    )

    burst_l80 = csg_l80.api_burst_pressure_psi
    collapse_l80 = csg_l80.api_collapse_pressure_psi

    # API Spec 5CT published book value for 9-5/8" 47# L-80:
    # Burst = 6870 psi, Collapse = 4750 psi
    assert 6500.0 <= burst_l80 <= 7200.0
    assert 4000.0 <= collapse_l80 <= 5500.0

    # P-110 steel must have higher burst than L-80
    csg_p110 = CasingSection(
        top_md_ft=0.0,
        bottom_md_ft=10000.0,
        od_in=9.625,
        weight_lbf_ft=47.0,
        wall_thickness_in=0.472,
        grade_yield_psi=CasingGrade.P110,
    )
    assert csg_p110.api_burst_pressure_psi > burst_l80


def test_casing_safety_factor_screening():
    csg = CasingSection(
        top_md_ft=0.0,
        bottom_md_ft=8000.0,
        od_in=9.625,
        weight_lbf_ft=47.0,
        wall_thickness_in=0.472,
        grade_yield_psi=CasingGrade.L80,
    )

    # 1. Normal safe shut-in: 1200 psi casing pressure
    rep_safe = calculate_casing_integrity(
        casing=csg,
        surface_casing_pressure_psi=1200.0,
        internal_mud_ppg=12.0,
        external_pore_ppg=9.0,
    )
    assert rep_safe.is_structurally_sound is True
    assert rep_safe.min_burst_sf >= 1.10
    assert len(rep_safe.alarms) == 0

    # 2. Catastrophic kick: 7200 psi surface pressure (exceeds L80 burst rating ~6870 psi)
    rep_burst = calculate_casing_integrity(
        casing=csg,
        surface_casing_pressure_psi=7200.0,
        internal_mud_ppg=12.0,
        external_pore_ppg=9.0,
    )
    assert rep_burst.is_structurally_sound is False
    assert rep_burst.min_burst_sf < 1.00
    assert any("BURST INTEGRITY BREACH" in a for a in rep_burst.alarms)

    # 3. Evacuation / Gas-voided casing check: casing evacuated down to 8000 ft in 14 ppg pore pressure
    rep_collapse = calculate_casing_integrity(
        casing=csg,
        surface_casing_pressure_psi=0.0,
        internal_mud_ppg=12.0,
        external_pore_ppg=14.0,
        gas_void_depth_ft=8000.0,
    )
    assert any("COLLAPSE INTEGRITY BREACH" in a for a in rep_collapse.alarms)
