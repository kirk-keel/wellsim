"""Unit tests for Bingham plastic non-Newtonian drilling mud hydraulics and ECD."""

import pytest

from wellsim.core.friction import (
    bingham_pressure_gradient,
    bingham_reynolds_number,
    critical_reynolds_number,
    hedstrom_number,
)
from wellsim.core.well_control import equivalent_circulating_density


def test_bingham_numbers_and_transitions():
    # 10.0 ppg, 3.0 fps, 8.5 in hole, PV 20 cp, YP 15 lbf/100ft2
    re_b = bingham_reynolds_number(10.0, 3.0, 8.5, 20.0)
    assert re_b > 0.0

    he = hedstrom_number(10.0, 15.0, 8.5, 20.0)
    assert he > 0.0

    re_crit = critical_reynolds_number(he)
    assert re_crit >= 2100.0


def test_bingham_laminar_and_turbulent_gradients():
    # Annular laminar flow test
    dp_ann = bingham_pressure_gradient(
        rho_ppg=10.0,
        v_fps=1.5,
        d_in=3.5,  # 8.5 in hole - 5.0 in pipe
        pv_cp=20.0,
        yp_lbf_100ft2=12.0,
        is_annulus=True,
    )
    # Check that gradient is positive and within expected oilfield range (0.01 - 0.1 psi/ft)
    assert 0.01 < dp_ann < 0.15

    # Pipe flow test
    dp_pipe = bingham_pressure_gradient(
        rho_ppg=10.0,
        v_fps=8.0,
        d_in=4.276,
        pv_cp=20.0,
        yp_lbf_100ft2=12.0,
        is_annulus=False,
    )
    assert dp_pipe > dp_ann


def test_equivalent_circulating_density():
    # MW = 10.0 ppg, Annular Pressure Loss = 260 psi at 10,000 ft TVD
    # ECD = 10.0 + 260 / (0.052 * 10000) = 10.0 + 0.50 = 10.50 ppg
    ecd = equivalent_circulating_density(10.0, 260.0, 10000.0)
    assert abs(ecd - 10.50) < 1e-4
