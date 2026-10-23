"""Unit tests for Deepwater Riser Gas Handling & Diverter Routing Engine."""

import pytest
from wellsim.core.riser_gas import (
    DiverterRoute,
    MarineRiserGeometry,
    calculate_diverter_friction_backpressure,
    simulate_riser_gas_unloading,
)


def test_diverter_friction_backpressure():
    # Zero flow check
    assert calculate_diverter_friction_backpressure(0.0, 0.0, 14.0, 120.0, 12.0) == 0.0

    # High rate multiphase venting: 60 MMSCFD gas + 2500 gpm mud through 14" line (120 ft long)
    p_14in = calculate_diverter_friction_backpressure(
        gas_rate_mmscfd=60.0,
        liquid_rate_gpm=2500.0,
        line_id_in=14.0,
        line_length_ft=120.0,
        mud_weight_ppg=12.0,
    )
    # Realistic backpressure for 14" overboard line at 60 MMSCFD is ~30 to 120 psi
    assert 20.0 <= p_14in <= 150.0

    # Restricted line size (10" line) should have significantly higher backpressure (roughly (14/10)^5)
    p_10in = calculate_diverter_friction_backpressure(
        gas_rate_mmscfd=60.0,
        liquid_rate_gpm=2500.0,
        line_id_in=10.0,
        line_length_ft=120.0,
        mud_weight_ppg=12.0,
    )
    assert p_10in > 2.5 * p_14in


def test_riser_gas_unloading_and_expansion():
    # 5,000 ft water depth, 19.5" ID riser, 12 ppg mud
    # 15 bbl gas kick crossing the subsea BOP
    riser = MarineRiserGeometry(
        water_depth_ft=5000.0,
        riser_id_in=19.5,
        diverter_line_id_in=14.0,
        mgs_gas_capacity_mmscfd=35.0,
    )

    res = simulate_riser_gas_unloading(
        riser=riser,
        influx_volume_at_bop_bbl=15.0,
        mud_weight_ppg=12.0,
        initial_pump_rate_gpm=0.0,
        rghs_isolated=False,
        dt_s=5.0,
    )

    assert len(res.timeline) > 50

    # 1. Volumetric expansion check: 15 bbl at 3135 psia should expand to >800 bbl near surface
    max_volume = max(step.bubble_volume_bbl for step in res.timeline)
    assert max_volume > 600.0  # Massive >40x expansion

    # 2. Riser base pressure drop check: seabed pressure drops as gas voids the riser
    p_initial = res.timeline[0].riser_base_pressure_psi
    assert p_initial > 3000.0
    assert res.min_riser_base_pressure_psi < p_initial - 500.0  # Loss of at least 500 psi hydrostatic head

    # 3. Peak unloading rate check
    assert res.peak_liquid_unloading_gpm > 1000.0  # High surge of expelled mud

    # 4. Overboard routing check: when gas exceeds 35 MMSCFD, diverter routes overboard
    overboard_steps = [s for s in res.timeline if s.active_route == DiverterRoute.OVERBOARD_STARBOARD]
    assert len(overboard_steps) > 0

    # 5. Diverter housing rating check (14" line should stay below 500 psi rating)
    assert res.diverter_rating_exceeded is False
    assert res.peak_diverter_backpressure_psi < 500.0


def test_riser_gas_handling_system_isolated():
    # When RGHS subsea packer is closed, gas is diverted early to MGS
    riser = MarineRiserGeometry(water_depth_ft=3000.0, riser_id_in=19.5)

    res = simulate_riser_gas_unloading(
        riser=riser,
        influx_volume_at_bop_bbl=5.0,
        mud_weight_ppg=10.0,
        rghs_isolated=True,
        dt_s=5.0,
    )

    # All active steps should route through MGS
    routes = {s.active_route for s in res.timeline}
    assert DiverterRoute.MGS_POOR_BOY in routes
