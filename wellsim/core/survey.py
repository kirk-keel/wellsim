"""Directional Survey & 3D Wellbore Trajectory Engine (Minimum Curvature Method).

Implements the industry standard Minimum Curvature Method (API RP 78 / SPE standards)
for converting directional survey stations (MD, Inclination, Azimuth) into full 3D
spatial coordinates: (TVD, Northing, Easting, Vertical Section, Dogleg Severity).

Enables 3D spatial wellbore visualization, dogleg severity screening, and inclination-
dependent hole cleaning / torque & drag modeling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "SurveyStation",
    "CalculatedStation",
    "WellTrajectory",
    "calculate_minimum_curvature",
]


@dataclass(frozen=True)
class SurveyStation:
    """Raw directional survey input station.

    Attributes:
        md_ft: Measured depth along wellbore (ft).
        inclination_deg: Wellbore inclination from vertical (0 to 180 deg).
        azimuth_deg: Wellbore azimuth clockwise from True North (0 to 360 deg).
    """

    md_ft: float
    inclination_deg: float
    azimuth_deg: float


@dataclass(frozen=True)
class CalculatedStation:
    """Full 3D spatial station computed via Minimum Curvature.

    Attributes:
        md_ft: Measured depth (ft).
        tvd_ft: True vertical depth (ft).
        north_ft: Northing displacement from surface location (ft).
        east_ft: Easting displacement from surface location (ft).
        inclination_deg: Inclination (deg).
        azimuth_deg: Azimuth (deg).
        dls_deg_100ft: Dogleg Severity (deg / 100 ft).
        vertical_section_ft: Horizontal displacement projected onto target azimuth (ft).
        closure_distance_ft: Straight-line horizontal distance from wellhead (ft).
        closure_azimuth_deg: Direction from wellhead to station (deg).
    """

    md_ft: float
    tvd_ft: float
    north_ft: float
    east_ft: float
    inclination_deg: float
    azimuth_deg: float
    dls_deg_100ft: float
    vertical_section_ft: float
    closure_distance_ft: float
    closure_azimuth_deg: float


@dataclass
class WellTrajectory:
    """Complete 3D wellbore trajectory profile."""

    stations: List[CalculatedStation]
    target_azimuth_deg: float = 0.0

    @property
    def total_md_ft(self) -> float:
        return self.stations[-1].md_ft if self.stations else 0.0

    @property
    def total_tvd_ft(self) -> float:
        return self.stations[-1].tvd_ft if self.stations else 0.0

    @property
    def max_dls_deg_100ft(self) -> float:
        return max((s.dls_deg_100ft for s in self.stations), default=0.0)

    def to_3d_coordinates(self) -> List[Dict[str, float]]:
        """Return 3D coordinates suitable for WebGL / Plotly 3D visualizers."""
        return [
            {
                "md": s.md_ft,
                "x_east": s.east_ft,
                "y_north": s.north_ft,
                "z_tvd": s.tvd_ft,
                "inc": s.inclination_deg,
                "azm": s.azimuth_deg,
                "dls": s.dls_deg_100ft,
            }
            for s in self.stations
        ]


def calculate_minimum_curvature(
    raw_stations: List[SurveyStation],
    target_azimuth_deg: float = 0.0,
) -> WellTrajectory:
    """Calculate 3D well path coordinates using the standard Minimum Curvature Method.

    Args:
        raw_stations: List of SurveyStation instances in ascending MD order.
        target_azimuth_deg: Azimuth for Vertical Section projection.

    Returns:
        WellTrajectory containing calculated spatial coordinates.
    """
    if not raw_stations:
        return WellTrajectory(stations=[])

    sorted_stations = sorted(raw_stations, key=lambda s: s.md_ft)

    # Initialize at wellhead (MD=0, TVD=0, North=0, East=0)
    calculated: List[CalculatedStation] = []

    # First station
    first = sorted_stations[0]
    tvd = first.md_ft if first.inclination_deg == 0.0 else first.md_ft * math.cos(math.radians(first.inclination_deg))
    north = 0.0
    east = 0.0
    closure_dist = math.hypot(north, east)
    closure_azm = math.degrees(math.atan2(east, north)) % 360.0
    vs = north * math.cos(math.radians(target_azimuth_deg)) + east * math.sin(math.radians(target_azimuth_deg))

    calculated.append(
        CalculatedStation(
            md_ft=first.md_ft,
            tvd_ft=round(tvd, 2),
            north_ft=round(north, 2),
            east_ft=round(east, 2),
            inclination_deg=first.inclination_deg,
            azimuth_deg=first.azimuth_deg,
            dls_deg_100ft=0.0,
            vertical_section_ft=round(vs, 2),
            closure_distance_ft=round(closure_dist, 2),
            closure_azimuth_deg=round(closure_azm, 2),
        )
    )

    # March through subsequent survey stations
    for i in range(1, len(sorted_stations)):
        s1 = sorted_stations[i - 1]
        s2 = sorted_stations[i]

        d_md = s2.md_ft - s1.md_ft
        if d_md <= 0.0:
            continue

        i1 = math.radians(s1.inclination_deg)
        i2 = math.radians(s2.inclination_deg)
        a1 = math.radians(s1.azimuth_deg)
        a2 = math.radians(s2.azimuth_deg)

        # Dogleg angle beta: cos(beta) = cos(I2 - I1) - sin(I1)*sin(I2)*(1 - cos(A2 - A1))
        cos_beta = math.cos(i2 - i1) - math.sin(i1) * math.sin(i2) * (1.0 - math.cos(a2 - a1))
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)

        # Ratio factor F
        if beta < 1e-6:
            rf = 1.0 + (beta**2) / 12.0
        else:
            rf = (2.0 / beta) * math.tan(beta / 2.0)

        # Coordinate increments
        d_tvd = (d_md / 2.0) * (math.cos(i1) + math.cos(i2)) * rf
        d_north = (d_md / 2.0) * (math.sin(i1) * math.cos(a1) + math.sin(i2) * math.cos(a2)) * rf
        d_east = (d_md / 2.0) * (math.sin(i1) * math.sin(a1) + math.sin(i2) * math.sin(a2)) * rf

        # Accumulate
        tvd += d_tvd
        north += d_north
        east += d_east

        # Dogleg severity (deg/100 ft)
        dls = math.degrees(beta) * (100.0 / d_md)

        # Closure and Vertical Section
        closure_dist = math.hypot(north, east)
        closure_azm = math.degrees(math.atan2(east, north)) % 360.0
        vs = north * math.cos(math.radians(target_azimuth_deg)) + east * math.sin(math.radians(target_azimuth_deg))

        calculated.append(
            CalculatedStation(
                md_ft=s2.md_ft,
                tvd_ft=round(tvd, 2),
                north_ft=round(north, 2),
                east_ft=round(east, 2),
                inclination_deg=s2.inclination_deg,
                azimuth_deg=s2.azimuth_deg,
                dls_deg_100ft=round(dls, 2),
                vertical_section_ft=round(vs, 2),
                closure_distance_ft=round(closure_dist, 2),
                closure_azimuth_deg=round(closure_azm, 2),
            )
        )

    return WellTrajectory(stations=calculated, target_azimuth_deg=target_azimuth_deg)
