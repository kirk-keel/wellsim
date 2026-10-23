"""Unit tests for Dynamic Wellbore Heat Transfer Engine (Hasan & Kabir 1994)."""

import pytest

from wellsim.core.hasan_kabir import (
    ThermalProperties,
    calculate_hasan_kabir_temperatures,
)


def test_hasan_kabir_circulating_temperature_profile():
    # 10,000 ft well, geothermal gradient = 1.5 degF/100 ft -> Formation T at 10,000 ft = 70 + 150 = 220 degF
    # Inlet pit temp = 80 degF, circulation rate = 120 gpm
    profile = calculate_hasan_kabir_temperatures(
        total_depth_ft=10000.0,
        flow_rate_gpm=120.0,
        mud_density_ppg=12.0,
        inlet_temp_degf=80.0,
        circulating_time_hr=4.0,
    )

    assert len(profile.depth_stations_ft) >= 20
    # Static formation temperature at 10,000 ft
    t_form_bottom = profile.formation_temp_degf[-1]
    assert abs(t_form_bottom - 220.0) < 1.0

    # Physical Law of Drilling Wellbore Heat Transfer:
    # 1. Flowing bit temperature is cooler than undisturbed formation rock (cooling effect of mud)
    assert profile.bottomhole_temp_degf < t_form_bottom
    assert profile.bottomhole_temp_degf > 100.0

    # 2. Return mud temperature at surface is warmer than inlet suction temperature
    assert profile.surface_return_temp_degf > 80.0


def test_hasan_kabir_time_warming_effect():
    # Lower flow rate of 80 gpm to test warming over time
    p_1hr = calculate_hasan_kabir_temperatures(
        total_depth_ft=10000.0,
        flow_rate_gpm=80.0,
        mud_density_ppg=12.0,
        inlet_temp_degf=80.0,
        circulating_time_hr=1.0,
    )
    p_24hr = calculate_hasan_kabir_temperatures(
        total_depth_ft=10000.0,
        flow_rate_gpm=80.0,
        mud_density_ppg=12.0,
        inlet_temp_degf=80.0,
        circulating_time_hr=24.0,
    )

    # Surface return mud warms up over time as heat is carried to surface
    assert p_24hr.surface_return_temp_degf >= p_1hr.surface_return_temp_degf
    # Bottomhole temperature stabilizes within circulating dynamic range
    assert 100.0 <= p_24hr.bottomhole_temp_degf <= 200.0
    assert 100.0 <= p_1hr.bottomhole_temp_degf <= 200.0
