"""Unit tests for the transient wellbore kick simulator (core/transient.py)."""

import pytest

from wellsim.core.transient import RigState, TransientSimulator, UserControls, WellGeometry


def test_initial_drilling_and_kick_influx():
    sim = TransientSimulator(
        geom=WellGeometry(tvd_well_ft=10000.0, tvd_shoe_ft=6000.0),
        omw_ppg=10.0,
        formation_pressure_psi=5720.0,  # ~11.0 ppg pore pressure
    )

    # Initial step in DRILLING
    st = sim.step(dt=1.0)
    assert st.rig_state == RigState.DRILLING
    assert st.pump_rate_gpm > 0.0

    # Over several seconds, underbalanced condition causes kick influx and pit gain
    for _ in range(30):
        st = sim.step(dt=1.0)

    assert st.influx_volume_bbl > 0.0
    assert st.pit_gain_bbl > 0.0
    assert st.pit_volume_bbl > 500.0


def test_flow_check_procedure():
    sim = TransientSimulator(
        omw_ppg=10.0,
        formation_pressure_psi=5720.0,
    )

    # Induce a kick
    for _ in range(20):
        sim.step(dt=1.0)

    # Driller initiates Flow Check: stop pumps (0 SPM) while well remains open
    ctrls = UserControls(pump_spm=0.0, choke_pct=100.0, bop_closed=False, spaced_out=True)
    st = sim.step(dt=1.0, controls=ctrls)

    assert st.rig_state == RigState.FLOW_CHECK
    # Influx continues to flow in -> flow out > 0 on flow check confirms positive kick
    assert st.flow_out_pct > 0.0


def test_shut_in_and_pressure_stabilization():
    sim = TransientSimulator(
        omw_ppg=10.0,
        formation_pressure_psi=5720.0,
    )

    # Induce kick
    for _ in range(20):
        sim.step(dt=1.0)

    # Driller shuts in well: close BOP, close choke
    shut_in_ctrls = UserControls(pump_spm=0.0, choke_pct=0.0, bop_closed=True, hcr_open=False, spaced_out=True)
    st = sim.step(dt=1.0, controls=shut_in_ctrls)

    assert st.rig_state == RigState.SHUT_IN
    assert st.flow_out_pct == 0.0

    # Let pressures stabilize
    for _ in range(5):
        st = sim.step(dt=1.0, controls=shut_in_ctrls)

    # SIDPP: Formation (5720) - Hydrostatic (5200) = 520 psi
    assert abs(st.spp_psi - 520.0) < 5.0
    # SICP must be higher than SIDPP because low-density gas influx is in the annulus
    assert st.casing_pressure_psi > st.spp_psi
    # Bottomhole pressure balances formation pressure
    assert abs(st.bhp_psi - 5720.0) < 5.0


def test_circulating_kill_transition():
    sim = TransientSimulator(
        omw_ppg=10.0,
        formation_pressure_psi=5720.0,
    )

    # Shut in the well
    shut_in_ctrls = UserControls(pump_spm=0.0, choke_pct=0.0, bop_closed=True)
    sim.step(dt=1.0, controls=shut_in_ctrls)

    # Start circulating kill: start pump at 30 SPM, open HCR, open choke to 15%
    kill_ctrls = UserControls(pump_spm=30.0, choke_pct=15.0, bop_closed=True, hcr_open=True, kill_mud_active=True)
    st = sim.step(dt=1.0, controls=kill_ctrls)

    assert st.rig_state == RigState.CIRCULATING_KILL
    assert st.pump_rate_gpm > 0.0


def test_auto_choke_pid_and_displacement():
    sim = TransientSimulator(
        omw_ppg=10.0,
        formation_pressure_psi=5720.0,
    )

    # Shut in well first
    sim.step(dt=1.0, controls=UserControls(pump_spm=0.0, choke_pct=0.0, bop_closed=True))

    # Engage Auto-Choke targeting 650 psi while circulating kill mud
    ctrls = UserControls(
        pump_spm=30.0,
        choke_pct=10.0,
        bop_closed=True,
        hcr_open=True,
        kill_mud_active=True,
        auto_choke=True,
        auto_choke_target_psi=650.0,
    )

    for _ in range(25):
        st = sim.step(dt=1.0, controls=ctrls)

    # Verify auto-choke modulated the choke opening percentage
    assert st.controls.choke_pct > 0.0
    # Verify kill mud displaced into drillpipe
    assert st.kill_mud_in_drillpipe_bbl > 0.0
    assert "CIRCULATION_2" in st.circulation_phase
