"""Wellsite Information Transfer Specification (WITS Level 0) parser and emitter.

Enables wellsim to act as a real-time digital twin connected to live rig data streams,
or as a hardware-in-the-loop (HIL) simulator feeding drilling cyberbase consoles.

Standard WITS Record 01 (General Drilling Data Items):
- 0108: Standpipe Pressure (psi)
- 0110: Pump Rate / SPM
- 0112: Flow Out Percentage (%)
- 0113: Total Pit Volume / Gain (bbl)
- 0120: Casing / Choke Pressure (psi)
- 0121: Mud Weight In (ppg)
- 0122: Mud Weight Out (ppg)
- 0123: Downhole ECD (ppg)
- 0124: Bottomhole Pressure (psi)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["WITSRecord01", "decode_wits0_packet", "encode_wits0_packet"]


@dataclass
class WITSRecord01:
    """Standard WITS Record 01 (General Drilling Data)."""

    spp_psi: Optional[float] = None  # Item 0108
    pump_spm: Optional[float] = None  # Item 0110
    flow_out_pct: Optional[float] = None  # Item 0112
    pit_gain_bbl: Optional[float] = None  # Item 0113
    casing_pressure_psi: Optional[float] = None  # Item 0120
    mw_in_ppg: Optional[float] = None  # Item 0121
    mw_out_ppg: Optional[float] = None  # Item 0122
    ecd_ppg: Optional[float] = None  # Item 0123
    bhp_psi: Optional[float] = None  # Item 0124


def encode_wits0_packet(data: WITSRecord01) -> str:
    """Encode a WITSRecord01 into standard WITS Level 0 ASCII packet stream."""
    lines = ["&&"]  # Record start delimiter

    if data.spp_psi is not None:
        lines.append(f"0108{data.spp_psi:.1f}")
    if data.pump_spm is not None:
        lines.append(f"0110{data.pump_spm:.1f}")
    if data.flow_out_pct is not None:
        lines.append(f"0112{data.flow_out_pct:.1f}")
    if data.pit_gain_bbl is not None:
        lines.append(f"0113{data.pit_gain_bbl:.2f}")
    if data.casing_pressure_psi is not None:
        lines.append(f"0120{data.casing_pressure_psi:.1f}")
    if data.mw_in_ppg is not None:
        lines.append(f"0121{data.mw_in_ppg:.2f}")
    if data.mw_out_ppg is not None:
        lines.append(f"0122{data.mw_out_ppg:.2f}")
    if data.ecd_ppg is not None:
        lines.append(f"0123{data.ecd_ppg:.2f}")
    if data.bhp_psi is not None:
        lines.append(f"0124{data.bhp_psi:.1f}")

    lines.append("!!")  # Record end delimiter
    return "\r\n".join(lines) + "\r\n"


def decode_wits0_packet(packet: str) -> WITSRecord01:
    """Decode a raw WITS Level 0 ASCII packet stream into a WITSRecord01."""
    record = WITSRecord01()
    lines = [line.strip() for line in packet.strip().splitlines() if line.strip()]

    for line in lines:
        if line.startswith("&&") or line.startswith("!!"):
            continue
        if len(line) < 5:
            continue

        item_id = line[:4]
        val_str = line[4:].strip()
        try:
            val = float(val_str)
        except ValueError:
            continue

        if item_id == "0108":
            record.spp_psi = val
        elif item_id == "0110":
            record.pump_spm = val
        elif item_id == "0112":
            record.flow_out_pct = val
        elif item_id == "0113":
            record.pit_gain_bbl = val
        elif item_id == "0120":
            record.casing_pressure_psi = val
        elif item_id == "0121":
            record.mw_in_ppg = val
        elif item_id == "0122":
            record.mw_out_ppg = val
        elif item_id == "0123":
            record.ecd_ppg = val
        elif item_id == "0124":
            record.bhp_psi = val

    return record
