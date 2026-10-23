"""Unit tests for Deepwater Gas Hydrate Formation and Inhibitor Dosing Engine."""

import pytest

from wellsim.core.hydrates import (
    InhibitorType,
    calculate_inhibitor_depression,
    calculate_required_inhibitor_rate,
    hydrate_formation_temperature,
)


def test_hydrate_equilibrium_temperature_trend():
    # As pressure rises from 500 psia to 5000 psia, hydrate freezing temp rises
    t_hyd_500 = hydrate_formation_temperature(500.0, gas_sg=0.65)
    t_hyd_2000 = hydrate_formation_temperature(2000.0, gas_sg=0.65)
    t_hyd_5000 = hydrate_formation_temperature(5000.0, gas_sg=0.65)

    assert 40.0 <= t_hyd_500 <= 55.0
    assert t_hyd_500 < t_hyd_2000 < t_hyd_5000
    assert 60.0 <= t_hyd_5000 <= 85.0


def test_deepwater_hydrate_risk_and_inhibitor_rate():
    # Cold deepwater seabed condition: 4500 psia subsea choke, cold seawater 38 degF
    # Free water production: 5.0 gpm
    risk = calculate_required_inhibitor_rate(
        p_psia=4500.0,
        t_fluid_degf=38.0,
        water_flow_rate_gpm=5.0,
        gas_sg=0.65,
        inhibitor_type=InhibitorType.METHANOL,
        safety_margin_degf=5.0,
    )

    # 38 degF is well below ~68 degF hydrate temperature
    assert risk.is_hydrate_risk is True
    assert risk.subcooling_degf > 20.0
    # Methanol requirement should be substantial (typical 20% to 45% wt%)
    assert 20.0 <= risk.methanol_wt_pct_required <= 50.0
    # Inhibitor injection rate must be positive
    assert risk.inhibitor_volume_rate_gpm > 1.0


def test_hammerschmidt_methanol_vs_meg():
    subcooling = 15.0  # 15 degF thermal subcooling
    w_meth = calculate_inhibitor_depression(subcooling, InhibitorType.METHANOL)
    w_meg = calculate_inhibitor_depression(subcooling, InhibitorType.MEG)

    # Methanol (lower molecular weight 32 vs 62) requires fewer weight % than MEG for same depression
    assert w_meth < w_meg
    assert w_meth > 0.0
    assert w_meg > 0.0
