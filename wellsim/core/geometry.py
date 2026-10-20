"""Flow-path geometry: the conduit the fluid sees at each measured depth.

``flow_path_segments(well, flow_path, bottom_md)`` cuts the well from surface to
``bottom_md`` (normally the inflow node) into segments of constant conduit and
constant inclination. Split points are every hole and string boundary and every
survey station. Within a segment the hydraulic diameter, flow area, roughness and
sin(theta) = dTVD/dMD are constant, so the marching integrator only needs to evaluate
fluid properties.

Annular flow uses the hydraulic diameter (ID_hole - OD_string) for friction; the
segment also carries d_o and d_i (M3-1) so the multiphase closure can use the
annulus-specific Taylor-bubble velocity. Spec 4.3 flags the hydraulic diameter as
as a known weak point for multiphase flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wellsim.cases.schema import Well

__all__ = ["FlowSegment", "flow_path_segments", "circular_area", "exit_area"]

_MD_TOL = 1e-6


@dataclass(frozen=True)
class FlowSegment:
    top_md: float  # ft
    bottom_md: float  # ft
    d_hyd: float  # hydraulic diameter, in
    area: float  # flow area, ft2
    roughness: float  # absolute roughness, in
    sin_theta: float  # dTVD/dMD (1 = vertical, 0 = horizontal)
    conduit: str  # human-readable description
    d_o: float = 0.0  # outer (casing / hole) diameter, in; 0 means "same as d_hyd" (pipe)
    d_i: float = 0.0  # inner (string OD) diameter, in; 0 means a pipe, > 0 an annulus

    def __post_init__(self) -> None:
        if self.d_o <= 0.0:
            object.__setattr__(self, "d_o", float(self.d_hyd))
        if self.d_i < 0.0:
            raise ValueError(f"d_i must be >= 0 (0 = pipe), got {self.d_i}")
        if self.d_i >= self.d_o:
            raise ValueError(f"d_i {self.d_i} in must be smaller than d_o {self.d_o} in")

    @property
    def is_annulus(self) -> bool:
        return self.d_i > 0.0

    @property
    def length(self) -> float:
        return self.bottom_md - self.top_md

    @property
    def d_ft(self) -> float:
        return self.d_hyd / 12.0

    @property
    def rel_rough(self) -> float:
        return self.roughness / self.d_hyd

    def contains(self, md: float) -> bool:
        return self.top_md - _MD_TOL <= md <= self.bottom_md + _MD_TOL


def circular_area(d_in: float) -> float:
    """ft2 from a diameter in inches."""
    return math.pi / 4.0 * (d_in / 12.0) ** 2


def _section_at(sections, md: float):
    for s in sections:
        if s.top_md - _MD_TOL <= md <= s.bottom_md + _MD_TOL:
            return s
    return None


def flow_path_segments(well: "Well", flow_path: str, bottom_md: float) -> list[FlowSegment]:
    """Segments from surface (0) down to ``bottom_md``, ordered top to bottom."""
    if bottom_md <= 0.0 or bottom_md > well.total_depth_md + _MD_TOL:
        raise ValueError(f"bottom_md {bottom_md} must be within (0, total_depth_md]")
    cuts = {0.0, bottom_md}
    for h in well.hole:
        cuts.update((h.top_md, h.bottom_md))
    if well.string:
        for s in well.string:
            cuts.update((s.top_md, s.bottom_md))
    if well.survey:
        cuts.update(p.md for p in well.survey)
    mds = sorted(c for c in cuts if 0.0 <= c <= bottom_md + _MD_TOL)
    segments: list[FlowSegment] = []
    for a, b in zip(mds, mds[1:]):
        if b - a <= _MD_TOL:
            continue
        mid = 0.5 * (a + b)
        hole = _section_at(well.hole, mid)
        if hole is None:
            raise ValueError(f"no hole section covers md {mid} ft")
        string = _section_at(well.string, mid) if well.string else None
        d_o, d_i = 0.0, 0.0
        if flow_path == "no_string" or string is None:
            d_hyd, area, rough = hole.id, circular_area(hole.id), hole.roughness
            conduit = f"{hole.kind} {hole.id:g} in"
        elif flow_path == "annulus":
            d_o, d_i = hole.id, string.od
            d_hyd = hole.id - string.od
            area = circular_area(hole.id) - circular_area(string.od)
            rough = max(hole.roughness, string.roughness)
            conduit = f"annulus {hole.id:g} x {string.od:g} in"
        elif flow_path in ("drillstring", "tubing"):
            d_hyd, area, rough = string.id, circular_area(string.id), string.roughness
            conduit = f"{flow_path} {string.id:g} in"
        else:
            raise ValueError(f"unknown flow_path {flow_path!r}")
        sin_theta = (well.md_to_tvd(b) - well.md_to_tvd(a)) / (b - a)
        sin_theta = min(1.0, max(0.0, sin_theta))
        segments.append(
            FlowSegment(
                top_md=a, bottom_md=b, d_hyd=d_hyd, area=area, roughness=rough, sin_theta=sin_theta, conduit=conduit,
                d_o=d_o, d_i=d_i,
            )
        )
    return segments


def exit_area(segments: list[FlowSegment], exit_id: float | None) -> float:
    """Flow area at the exit, ft2: the exit restriction if given, else the top segment's area."""
    if exit_id is not None:
        return circular_area(exit_id)
    return segments[0].area
