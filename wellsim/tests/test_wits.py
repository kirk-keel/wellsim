"""Unit tests for WITS Level 0 telemetry encoding and decoding."""

import pytest

from wellsim.core.wits import WITSRecord01, decode_wits0_packet, encode_wits0_packet


def test_wits0_encode_and_decode_roundtrip():
    orig = WITSRecord01(
        spp_psi=3250.5,
        pump_spm=85.0,
        flow_out_pct=104.2,
        pit_gain_bbl=12.45,
        casing_pressure_psi=480.0,
        mw_in_ppg=12.2,
        mw_out_ppg=12.15,
        ecd_ppg=12.65,
        bhp_psi=6820.0,
    )

    encoded = encode_wits0_packet(orig)
    assert encoded.startswith("&&\r\n")
    assert encoded.endswith("!!\r\n")
    assert "01083250.5" in encoded
    assert "011312.45" in encoded

    decoded = decode_wits0_packet(encoded)
    assert decoded.spp_psi == 3250.5
    assert decoded.pump_spm == 85.0
    assert decoded.flow_out_pct == 104.2
    assert decoded.pit_gain_bbl == 12.45
    assert decoded.casing_pressure_psi == 480.0
    assert decoded.mw_in_ppg == 12.2
    assert decoded.ecd_ppg == 12.65
    assert decoded.bhp_psi == 6820.0
