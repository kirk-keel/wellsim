"""Unit tests for Rig Telemetry Anomaly Detector."""

import pytest

from wellsim.core.anomaly_detection import (
    DrillingAnomalyType,
    RigAnomalyDetector,
    RigTelemetrySnapshot,
)


def test_drilling_break_detection():
    detector = RigAnomalyDetector()

    # Steady baseline ROP = 25 ft/hr for 6 steps
    for t in range(6):
        sn = RigTelemetrySnapshot(
            time_s=t * 10.0,
            rop_ft_hr=25.0,
            spp_psi=3000.0,
            pump_spm=80.0,
            torque_ft_lbf=4000.0,
            hookload_klbf=180.0,
            pit_gain_bbl=0.0,
            flow_out_pct=100.0,
        )
        alerts = detector.process_snapshot(sn)
        assert len(alerts) == 0

    # Sudden surge to 85 ft/hr (drilling break into pay zone)
    break_snap = RigTelemetrySnapshot(
        time_s=70.0,
        rop_ft_hr=85.0,
        spp_psi=3000.0,
        pump_spm=80.0,
        torque_ft_lbf=4200.0,
        hookload_klbf=180.0,
        pit_gain_bbl=0.0,
        flow_out_pct=100.0,
    )
    alerts = detector.process_snapshot(break_snap)
    assert len(alerts) == 1
    assert alerts[0].anomaly_type == DrillingAnomalyType.DRILLING_BREAK
    assert "conduct mandatory flow check" in alerts[0].recommended_action


def test_bit_nozzle_plugging_and_washout():
    detector = RigAnomalyDetector()

    # Establish baseline at 80 spm, 3200 psi (impedance = 3200 / 6400 = 0.50)
    for t in range(5):
        sn = RigTelemetrySnapshot(
            time_s=t * 5.0,
            rop_ft_hr=30.0,
            spp_psi=3200.0,
            pump_spm=80.0,
            torque_ft_lbf=4000.0,
            hookload_klbf=180.0,
            pit_gain_bbl=0.0,
            flow_out_pct=100.0,
        )
        detector.process_snapshot(sn)

    # 1. Nozzle plugging: SPP jumps to 3900 psi (+700 psi) at same 80 spm
    plug_snap = RigTelemetrySnapshot(
        time_s=30.0,
        rop_ft_hr=30.0,
        spp_psi=3900.0,
        pump_spm=80.0,
        torque_ft_lbf=4000.0,
        hookload_klbf=180.0,
        pit_gain_bbl=0.0,
        flow_out_pct=100.0,
    )
    alerts = detector.process_snapshot(plug_snap)
    assert any(a.anomaly_type == DrillingAnomalyType.NOZZLE_PLUGGING for a in alerts)

    # Reset and test washout
    detector_washout = RigAnomalyDetector()
    for t in range(5):
        detector_washout.process_snapshot(
            RigTelemetrySnapshot(
                time_s=t * 5.0,
                rop_ft_hr=30.0,
                spp_psi=3200.0,
                pump_spm=80.0,
                torque_ft_lbf=4000.0,
                hookload_klbf=180.0,
                pit_gain_bbl=0.0,
                flow_out_pct=100.0,
            )
        )

    # Washout: SPP drops to 2400 psi (-800 psi) at same 80 spm
    wash_snap = RigTelemetrySnapshot(
        time_s=30.0,
        rop_ft_hr=30.0,
        spp_psi=2400.0,
        pump_spm=80.0,
        torque_ft_lbf=4000.0,
        hookload_klbf=180.0,
        pit_gain_bbl=0.0,
        flow_out_pct=100.0,
    )
    alerts_wash = detector_washout.process_snapshot(wash_snap)
    assert any(a.anomaly_type == DrillingAnomalyType.STRING_WASHOUT for a in alerts_wash)


def test_packoff_vs_differential_sticking():
    # Differential sticking: torque spikes, SPP completely steady
    det_diff = RigAnomalyDetector()
    for t in range(5):
        det_diff.process_snapshot(
            RigTelemetrySnapshot(
                time_s=t * 5.0,
                rop_ft_hr=0.0,
                spp_psi=3000.0,
                pump_spm=80.0,
                torque_ft_lbf=4000.0,
                hookload_klbf=180.0,
                pit_gain_bbl=0.0,
                flow_out_pct=100.0,
            )
        )

    diff_alert = det_diff.process_snapshot(
        RigTelemetrySnapshot(
            time_s=30.0,
            rop_ft_hr=0.0,
            spp_psi=3000.0,  # Unchanged pressure
            pump_spm=80.0,
            torque_ft_lbf=9500.0,  # Torque spike
            hookload_klbf=180.0,
            pit_gain_bbl=0.0,
            flow_out_pct=100.0,
        )
    )
    assert any(a.anomaly_type == DrillingAnomalyType.DIFFERENTIAL_STICKING for a in diff_alert)

    # Mechanical pack-off: BOTH torque and SPP spike simultaneously
    det_pack = RigAnomalyDetector()
    for t in range(5):
        det_pack.process_snapshot(
            RigTelemetrySnapshot(
                time_s=t * 5.0,
                rop_ft_hr=40.0,
                spp_psi=3000.0,
                pump_spm=80.0,
                torque_ft_lbf=4000.0,
                hookload_klbf=180.0,
                pit_gain_bbl=0.0,
                flow_out_pct=100.0,
            )
        )

    pack_alert = det_pack.process_snapshot(
        RigTelemetrySnapshot(
            time_s=30.0,
            rop_ft_hr=40.0,
            spp_psi=3450.0,  # Pressure surge
            pump_spm=80.0,
            torque_ft_lbf=9800.0,  # Torque spike
            hookload_klbf=180.0,
            pit_gain_bbl=0.0,
            flow_out_pct=100.0,
        )
    )
    assert any(a.anomaly_type == DrillingAnomalyType.MECHANICAL_PACKOFF for a in pack_alert)
