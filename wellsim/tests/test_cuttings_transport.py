"""Unit tests for Dynamic Cuttings Transport & Hole Cleaning (3-Layer Model)."""

import pytest

from wellsim.core.cuttings_transport import (
    CuttingsProperties,
    calculate_critical_transport_velocity,
    calculate_cuttings_generation_rate,
    calculate_slip_velocity,
    simulate_hole_cleaning_profile,
)


def test_cuttings_generation_rate():
    props = CuttingsProperties(porosity=0.20)
    # 8.5" hole at 60 ft/hr ROP
    q_gpm = calculate_cuttings_generation_rate(rop_ft_hr=60.0, hole_diameter_in=8.5, cuttings=props)
    assert q_gpm > 0.0
    # Doubling ROP must double generation rate
    q_double = calculate_cuttings_generation_rate(rop_ft_hr=120.0, hole_diameter_in=8.5, cuttings=props)
    assert abs(q_double - 2.0 * q_gpm) < 1e-4


def test_slip_velocity():
    props = CuttingsProperties(diameter_in=0.15, density_ppg=22.0)
    v_slip = calculate_slip_velocity(props, mud_weight_ppg=12.0, pv_cp=20.0, yp_lbf_100ft2=15.0)
    # In standard 12 ppg drilling mud, 0.15" cuttings slip at ~0.2 to 1.0 ft/s
    assert 0.1 <= v_slip <= 1.2


def test_critical_velocity_peaks_in_avalanching_window():
    props = CuttingsProperties()
    # Compare vertical (10 deg), critical avalanching (48 deg), and horizontal (85 deg)
    v_vert = calculate_critical_transport_velocity(
        10.0, 8.5, 5.0, 11.5, 22.0, 14.0, 50.0, props
    )
    v_aval = calculate_critical_transport_velocity(
        48.0, 8.5, 5.0, 11.5, 22.0, 14.0, 50.0, props
    )
    v_horiz = calculate_critical_transport_velocity(
        85.0, 8.5, 5.0, 11.5, 22.0, 14.0, 50.0, props
    )

    # API RP 13D / Larsen: critical velocity is highest in the 40-55 deg avalanching window
    assert v_aval > v_vert
    assert v_aval > v_horiz


def test_hole_cleaning_profile_clean_vs_packoff():
    # Inclined directional well trajectory
    trajectory = [
        (2000.0, 10.0),  # Vertical section
        (5000.0, 45.0),  # Avalanching buildup
        (8000.0, 65.0),  # High angle tangent
        (12000.0, 90.0), # Horizontal lateral
    ]

    # 1. High circulation rate: 550 gpm -> Good cleaning, 0% bed
    clean_profile = simulate_hole_cleaning_profile(
        md_total_ft=12000.0,
        inclinations=trajectory,
        hole_diameter_in=8.5,
        pipe_od_in=5.0,
        flow_rate_gpm=550.0,
        rop_ft_hr=40.0,
        mud_weight_ppg=12.0,
        pv_cp=24.0,
        yp_lbf_100ft2=16.0,
    )
    assert len(clean_profile) == 4
    for st in clean_profile:
        assert st.bed_height_fraction == 0.0
        assert st.packoff_risk_index <= 0.35

    # 2. Inadequate circulation rate: 180 gpm -> Cuttings bed builds up in lateral and avalanching sections
    dirty_profile = simulate_hole_cleaning_profile(
        md_total_ft=12000.0,
        inclinations=trajectory,
        hole_diameter_in=8.5,
        pipe_od_in=5.0,
        flow_rate_gpm=180.0,
        rop_ft_hr=80.0,
        mud_weight_ppg=12.0,
        pv_cp=24.0,
        yp_lbf_100ft2=16.0,
    )
    # Horizontal lateral (station 3) should have formed a bed
    lat_stat = dirty_profile[3]
    assert lat_stat.bed_height_fraction > 0.15
    assert lat_stat.effective_diameter_in < (8.5 - 5.0)
    assert lat_stat.additional_friction_loss_psi > 0.0
    assert lat_stat.packoff_risk_index > 0.35
