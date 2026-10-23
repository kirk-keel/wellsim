"""Unit tests for Herschel-Bulkley (YPL) rheology, HPHT corrections, and OBM gas solubility."""

import pytest

from wellsim.core.obm_solubility import (
    OBMProperties,
    bubble_point_pressure,
    flash_equilibrium,
    gas_solubility_rs,
)
from wellsim.core.rheology import (
    HerschelBulkleyParams,
    fit_herschel_bulkley,
    herschel_bulkley_pressure_gradient,
    hpht_correction,
)


def test_fit_herschel_bulkley_from_fann_readings():
    # Typical 12.0 ppg water/oil mud readings:
    # 600 RPM: 70, 300 RPM: 45, 6 RPM: 10, 3 RPM: 8
    hb = fit_herschel_bulkley(
        theta_600=70.0,
        theta_300=45.0,
        theta_6=10.0,
        theta_3=8.0,
    )

    # API RP 13D: tau_0 = 2*8 - 10 = 6.0
    assert abs(hb.tau_0 - 6.0) < 1e-4
    assert 0.4 <= hb.n <= 0.8  # Shear thinning
    assert hb.k > 0.0

    # Wall shear stress at 511 s^-1 (300 RPM equivalent) should match ~45 lbf/100 ft^2
    tau_300 = hb.wall_shear_stress(511.0)
    assert abs(tau_300 - 45.0) < 1.0


def test_herschel_bulkley_pressure_gradient_and_hpht():
    hb = HerschelBulkleyParams(tau_0=8.0, k=0.35, n=0.65)
    rho_ppg = 12.5
    d_in = 6.0

    # Flow at 3.0 ft/s
    dp_dl_base = herschel_bulkley_pressure_gradient(
        rho_ppg=rho_ppg,
        v_fps=3.0,
        d_in=d_in,
        params=hb,
        is_annulus=True,
    )
    assert dp_dl_base > 0.0

    # Higher velocity must produce higher pressure gradient
    dp_dl_fast = herschel_bulkley_pressure_gradient(
        rho_ppg=rho_ppg,
        v_fps=6.0,
        d_in=d_in,
        params=hb,
        is_annulus=True,
    )
    assert dp_dl_fast > dp_dl_base

    # HPHT thermal thinning: at 250 degF (hot wellbore), viscosity and gradient drop
    dp_dl_hot = herschel_bulkley_pressure_gradient(
        rho_ppg=rho_ppg,
        v_fps=3.0,
        d_in=d_in,
        params=hb,
        is_annulus=True,
        t_f=250.0,
        p_psia=3000.0,
    )
    assert dp_dl_hot < dp_dl_base


def test_obm_gas_solubility_and_bubble_point():
    obm = OBMProperties(base_oil_sg=0.82, oil_water_ratio=0.80, gas_sg=0.65)

    p_deep = 6000.0
    t_deep = 180.0
    rs = gas_solubility_rs(p_deep, t_deep, obm)
    assert rs > 500.0  # Substantial dissolved gas capacity in OBM

    # Inverting via bubble_point_pressure should recover p_deep within 1%
    pb_recovered = bubble_point_pressure(rs, t_deep, obm)
    assert abs(pb_recovered - p_deep) / p_deep < 0.01


def test_obm_flash_equilibrium_downhole_vs_surface():
    obm = OBMProperties(base_oil_sg=0.82, oil_water_ratio=0.80, gas_sg=0.65)

    # 10 bbl influx containing 15,000 scf of methane in 50 bbl of mud
    total_gas_scf = 15000.0
    mud_vol_bbl = 50.0

    # 1. Downhole condition: 7000 psia, 200 degF (P >> Pb)
    diss_dh, free_scf_dh, free_bbl_dh, swell_dh = flash_equilibrium(
        p_psia=7000.0,
        t_f=200.0,
        total_gas_scf=total_gas_scf,
        mud_volume_bbl=mud_vol_bbl,
        obm=obm,
    )

    # Downhole: all gas is dissolved, free gas volume is strictly ZERO!
    assert diss_dh == total_gas_scf
    assert free_scf_dh == 0.0
    assert free_bbl_dh == 0.0
    # Oil swelling is modest (few barrels)
    assert 0.0 < swell_dh < 5.0

    # 2. Near surface / shallow casing condition: 300 psia, 80 degF (P << Pb)
    diss_surf, free_scf_surf, free_bbl_surf, swell_surf = flash_equilibrium(
        p_psia=300.0,
        t_f=80.0,
        total_gas_scf=total_gas_scf,
        mud_volume_bbl=mud_vol_bbl,
        obm=obm,
    )

    # Near surface: oil can hold very little gas, vast majority flashes into free gas
    assert free_scf_surf > 10000.0
    # Free gas expands enormously at low pressure
    assert free_bbl_surf > 50.0  # Explosive volumetric unloading!
