"""WITSML 1.4.1.1 XML Telemetry Engine (Energistics Standards).

Provides robust XML serialization and parsing for directional trajectory and wellbore
geometry objects complying with Energistics WITSML 1.4.1.1 schemas.

Allows seamless bi-directional integration with real-time drilling data hubs
(e.g., Kongsberg SiteCom, SLB InterACT, Halliburton Insite, Baker Hughes ADVANTAGE).

References:
- Energistics WITSML Standards Version 1.4.1.1:
  http://w3.energistics.org/schema/WITSML_v1.4.1.1_Data_Schema/
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from wellsim.core.survey import CalculatedStation, SurveyStation

__all__ = [
    "WITSMLTrajectoryStation",
    "generate_trajectory_witsml",
    "parse_trajectory_witsml",
    "generate_wbgeometry_witsml",
    "parse_wbgeometry_witsml",
]

WITSML_NS = "http://www.witsml.org/schemas/1series"
SCHEMA_VERSION = "1.4.1.1"


@dataclass
class WITSMLTrajectoryStation:
    """WITSML trajectory station representation."""

    uid: str
    md_ft: float
    tvd_ft: float
    inclination_deg: float
    azimuth_deg: float
    disp_ns_ft: float
    disp_ew_ft: float
    dls_deg_per_100ft: float = 0.0


def generate_trajectory_witsml(
    stations: List[Union[SurveyStation, CalculatedStation]],
    well_name: str = "Well-01",
    wellbore_name: str = "Wellbore-01",
    trajectory_name: str = "Planned Trajectory",
) -> str:
    """Generate WITSML 1.4.1.1 XML document for a directional wellbore trajectory.

    Args:
        stations: List of SurveyStation or CalculatedStation objects.
        well_name: Name of the well.
        wellbore_name: Name of the wellbore.
        trajectory_name: Name of the trajectory object.

    Returns:
        Formatted XML string conforming to WITSML 1.4.1.1 schema.
    """
    root = ET.Element(
        "trajectorys",
        xmlns=WITSML_NS,
        version=SCHEMA_VERSION,
    )

    traj = ET.SubElement(root, "trajectory", uidWell="W-01", uidWellbore="WB-01", uid="TR-01")
    ET.SubElement(traj, "nameWell").text = well_name
    ET.SubElement(traj, "nameWellbore").text = wellbore_name
    ET.SubElement(traj, "name").text = trajectory_name
    ET.SubElement(traj, "aziRef").text = "true north"

    for i, s in enumerate(stations):
        t_stat = ET.SubElement(traj, "trajectoryStation", uid=f"stn-{i + 1}")
        ET.SubElement(t_stat, "typeTrajStation").text = "tie in" if i == 0 else "survey"

        md_elem = ET.SubElement(t_stat, "md", uom="ft")
        md_elem.text = f"{s.md_ft:.2f}"

        tvd_val = getattr(s, "tvd_ft", s.md_ft)
        tvd_elem = ET.SubElement(t_stat, "tvd", uom="ft")
        tvd_elem.text = f"{tvd_val:.2f}"

        incl_elem = ET.SubElement(t_stat, "incl", uom="dega")
        incl_elem.text = f"{s.inclination_deg:.2f}"

        azi_elem = ET.SubElement(t_stat, "azi", uom="dega")
        azi_elem.text = f"{s.azimuth_deg:.2f}"

        disp_ns = getattr(s, "north_ft", 0.0)
        ns_elem = ET.SubElement(t_stat, "dispNs", uom="ft")
        ns_elem.text = f"{disp_ns:.2f}"

        disp_ew = getattr(s, "east_ft", 0.0)
        ew_elem = ET.SubElement(t_stat, "dispEw", uom="ft")
        ew_elem.text = f"{disp_ew:.2f}"

        dls = getattr(s, "dls_deg_per_100ft", 0.0)
        dls_elem = ET.SubElement(t_stat, "dls", uom="dega/30m")
        dls_elem.text = f"{dls:.3f}"

    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
    return f'<?xml version="1.0" encoding="utf-8"?>\n{xml_str}'


def parse_trajectory_witsml(xml_str: str) -> List[CalculatedStation]:
    """Parse WITSML 1.4.1.1 XML string and return list of CalculatedStation objects.

    Handles XML with or without namespace declarations.

    Args:
        xml_str: WITSML trajectory XML string.

    Returns:
        List of CalculatedStation objects.
    """
    root = ET.fromstring(xml_str)

    # Strip XML namespace if present
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    stations: List[CalculatedStation] = []

    for stn_elem in root.iter(f"{ns}trajectoryStation"):
        def _get_val(tag_name: str, default: float = 0.0) -> float:
            elem = stn_elem.find(f"{ns}{tag_name}")
            if elem is not None and elem.text:
                try:
                    return float(elem.text.strip())
                except ValueError:
                    pass
            return default

        md = _get_val("md", 0.0)
        tvd = _get_val("tvd", md)
        incl = _get_val("incl", 0.0)
        azi = _get_val("azi", 0.0)
        disp_ns = _get_val("dispNs", 0.0)
        disp_ew = _get_val("dispEw", 0.0)
        dls = _get_val("dls", 0.0)

        closure_dist = (disp_ns**2 + disp_ew**2) ** 0.5
        closure_azi = (math.degrees(math.atan2(disp_ew, disp_ns)) + 360.0) % 360.0 if closure_dist > 1e-4 else 0.0

        stations.append(
            CalculatedStation(
                md_ft=md,
                tvd_ft=tvd,
                north_ft=disp_ns,
                east_ft=disp_ew,
                inclination_deg=incl,
                azimuth_deg=azi,
                dls_deg_100ft=dls,
                vertical_section_ft=closure_dist,
                closure_distance_ft=closure_dist,
                closure_azimuth_deg=closure_azi,
            )
        )

    return stations


def generate_wbgeometry_witsml(
    sections: List[Dict[str, Any]],
    well_name: str = "Well-01",
    wellbore_name: str = "Wellbore-01",
) -> str:
    """Generate WITSML 1.4.1.1 XML document for wellbore tubular geometry (wbGeometry).

    Args:
        sections: List of dicts representing casing/hole sections:
                  [{"type": "casing", "od_in": 9.625, "id_in": 8.681, "top_md_ft": 0.0, "bottom_md_ft": 8000.0}, ...]
        well_name: Name of the well.
        wellbore_name: Name of the wellbore.

    Returns:
        Formatted WITSML wbGeometrys XML string.
    """
    root = ET.Element(
        "wbGeometrys",
        xmlns=WITSML_NS,
        version=SCHEMA_VERSION,
    )

    wb_geom = ET.SubElement(root, "wbGeometry", uidWell="W-01", uidWellbore="WB-01", uid="WBG-01")
    ET.SubElement(wb_geom, "nameWell").text = well_name
    ET.SubElement(wb_geom, "nameWellbore").text = wellbore_name
    ET.SubElement(wb_geom, "name").text = f"{wellbore_name} Casing Scheme"

    for i, sec in enumerate(sections):
        wb_sec = ET.SubElement(wb_geom, "wbGeometrySection", uid=f"sec-{i + 1}")
        ET.SubElement(wb_sec, "typeHoleCsg").text = sec.get("type", "casing")

        md_top = ET.SubElement(wb_sec, "mdTop", uom="ft")
        md_top.text = f"{float(sec.get('top_md_ft', 0.0)):.2f}"

        md_bot = ET.SubElement(wb_sec, "mdBottom", uom="ft")
        md_bot.text = f"{float(sec.get('bottom_md_ft', 0.0)):.2f}"

        id_elem = ET.SubElement(wb_sec, "idSection", uom="in")
        id_elem.text = f"{float(sec.get('id_in', 0.0)):.3f}"

        od_elem = ET.SubElement(wb_sec, "odSection", uom="in")
        od_elem.text = f"{float(sec.get('od_in', 0.0)):.3f}"

        if "weight_lbf_ft" in sec:
            wt_elem = ET.SubElement(wb_sec, "wtPerLen", uom="lbm/ft")
            wt_elem.text = f"{float(sec['weight_lbf_ft']):.2f}"

    xml_str = ET.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
    return f'<?xml version="1.0" encoding="utf-8"?>\n{xml_str}'


def parse_wbgeometry_witsml(xml_str: str) -> List[Dict[str, Any]]:
    """Parse WITSML wbGeometrys XML string and return list of section dicts."""
    root = ET.fromstring(xml_str)
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    sections: List[Dict[str, Any]] = []

    for sec in root.iter(f"{ns}wbGeometrySection"):
        sec_type = "casing"
        type_elem = sec.find(f"{ns}typeHoleCsg")
        if type_elem is not None and type_elem.text:
            sec_type = type_elem.text.strip()

        def _val(tag: str, default: float = 0.0) -> float:
            el = sec.find(f"{ns}{tag}")
            if el is not None and el.text:
                try:
                    return float(el.text.strip())
                except ValueError:
                    pass
            return default

        sections.append({
            "type": sec_type,
            "top_md_ft": _val("mdTop"),
            "bottom_md_ft": _val("mdBottom"),
            "id_in": _val("idSection"),
            "od_in": _val("odSection"),
            "weight_lbf_ft": _val("wtPerLen"),
        })

    return sections
