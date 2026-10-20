"""Units for wellsim.

Internal (canonical) units are oilfield units. Every function in ``wellsim.core``
takes and returns canonical units, so the solver loops never touch this module.
Conversion happens only at the I/O boundary: ``wellsim.cases.quantities`` on
the way in, reporting and the UI on the way out.

Each *kind* of quantity has one canonical unit and a table of accepted input
units. A unit converts to canonical as ``canonical = value * scale + offset``.

Canonical units by kind
-----------------------
pressure         psia        absolute pressure; a bare number is psia, "psig" adds 14.7
dpressure        psi         pressure difference (drawdown, friction, pump limit)
temperature      degF
length           ft          depths, radii, pay thickness, segment length
diameter         in          pipe IDs/ODs, roughness
liquid_rate      stb/d
gas_rate         Mscf/d
gor              scf/stb
density          lbm/ft3     PVT densities
mud_weight       ppg         kill / drilling fluid density
viscosity        cp
permeability     md
compressibility  1/psi
salinity         ppm         total dissolved solids by weight
time             day
power            hp
velocity         ft/s
volume           bbl
fraction         -           dimensionless 0..1 ("%" accepted)

Standard conditions: metric standard volumes (Sm3 at 15 degC, 1 atm) are
converted to scf (60 degF, 14.696 psia) by pure volume ratio, ignoring the
~0.2% difference in reference temperature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "UnitError",
    "Unit",
    "Kind",
    "KINDS",
    "canonical_unit",
    "units_for",
    "normalize_unit",
    "convert",
    "to_internal",
    "from_internal",
    "parse_quantity",
    "coerce",
]


class UnitError(ValueError):
    """Unknown unit, unknown kind, or unit not valid for the kind."""


@dataclass(frozen=True)
class Unit:
    scale: float
    offset: float = 0.0


@dataclass(frozen=True)
class Kind:
    name: str
    canonical: str
    units: dict[str, Unit]
    description: str = ""

    def unit(self, name: str) -> Unit:
        key = normalize_unit(name)
        try:
            return self.units[key]
        except KeyError:
            raise UnitError(
                f"unit {name!r} is not valid for {self.name} "
                f"(accepted: {', '.join(sorted(self.units))})"
            ) from None


# --- unit-string handling ------------------------------------------------------

_NORMALIZE_TABLE = str.maketrans(
    {"°": "deg", "º": "deg", "³": "3", "²": "2", "·": ".", " ": "", "\t": ""}
)


def normalize_unit(unit: str | None) -> str:
    """Canonical spelling for lookups: strip spaces, fold case, degree sign -> deg, ^3 -> 3."""
    if unit is None:
        return ""
    s = str(unit).translate(_NORMALIZE_TABLE).replace("^", "").replace("**", "")
    return s.lower()


# --- conversion constants (exact or NIST) -------------------------------------

_PSI_PER_ATM = 14.695950
_PSIG_OFFSET = 14.7  # gauge -> absolute; matches the 14.7 psia default exit pressure
_PSI_PER_BAR = 14.503773773
_PSI_PER_KPA = 0.14503773773
_FT_PER_M = 1.0 / 0.3048
_IN_PER_MM = 1.0 / 25.4
_M3_PER_BBL = 0.158987294928
_BBL_PER_M3 = 1.0 / _M3_PER_BBL
_FT3_PER_M3 = _FT_PER_M**3
_FT3_PER_BBL = 5.614583333
_LBM_PER_FT3_PER_PPG = 7.480519480
_LBM_PER_FT3_PER_KG_PER_M3 = 0.062427960576
_HP_PER_KW = 1.341022090
_RANKINE_OFFSET = 459.67


def _kind(
    name: str,
    canonical: str,
    description: str,
    table: dict[str, Unit | float],
    aliases: dict[str, str] | None = None,
) -> Kind:
    units: dict[str, Unit] = {}
    for key, val in table.items():
        units[normalize_unit(key)] = val if isinstance(val, Unit) else Unit(float(val))
    for alias, target in (aliases or {}).items():
        units[normalize_unit(alias)] = units[normalize_unit(target)]
    canon = normalize_unit(canonical)
    if units.get(canon) != Unit(1.0, 0.0):
        raise ValueError(f"canonical unit {canonical!r} of kind {name!r} must have scale 1, offset 0")
    return Kind(name=name, canonical=canon, units=units, description=description)


_KIND_LIST = [
    _kind(
        "pressure", "psia", "absolute pressure",
        {
            "psia": 1.0,
            "psi": 1.0,
            "psig": Unit(1.0, _PSIG_OFFSET),
            "bar": _PSI_PER_BAR,
            "bara": _PSI_PER_BAR,
            "barg": Unit(_PSI_PER_BAR, _PSIG_OFFSET),
            "kPa": _PSI_PER_KPA,
            "kPaa": _PSI_PER_KPA,
            "MPa": _PSI_PER_KPA * 1e3,
            "Pa": _PSI_PER_KPA * 1e-3,
            "atm": _PSI_PER_ATM,
        },
    ),
    _kind(
        "dpressure", "psi", "pressure difference",
        {
            "psi": 1.0,
            "bar": _PSI_PER_BAR,
            "kPa": _PSI_PER_KPA,
            "MPa": _PSI_PER_KPA * 1e3,
            "Pa": _PSI_PER_KPA * 1e-3,
            "atm": _PSI_PER_ATM,
        },
    ),
    _kind(
        "temperature", "degF", "temperature",
        {
            "degF": 1.0,
            "degC": Unit(1.8, 32.0),
            "K": Unit(1.8, -_RANKINE_OFFSET),
            "degR": Unit(1.0, -_RANKINE_OFFSET),
        },
        aliases={"F": "degF", "C": "degC", "R": "degR", "kelvin": "K"},
    ),
    _kind(
        "length", "ft", "length, depth, radius",
        {"ft": 1.0, "feet": 1.0, "m": _FT_PER_M, "km": _FT_PER_M * 1e3, "in": 1.0 / 12.0},
    ),
    _kind(
        "diameter", "in", "pipe diameter, roughness",
        {"in": 1.0, "inch": 1.0, "mm": _IN_PER_MM, "cm": _IN_PER_MM * 10.0, "m": _IN_PER_MM * 1e3, "ft": 12.0},
    ),
    _kind(
        "liquid_rate", "stb/d", "liquid rate at standard conditions",
        {
            "stb/d": 1.0,
            "stb/day": 1.0,
            "bbl/d": 1.0,
            "bpd": 1.0,
            "bopd": 1.0,
            "bwpd": 1.0,
            "bpm": 1440.0,
            "bbl/min": 1440.0,
            "gpm": 1440.0 / 42.0,
            "gal/min": 1440.0 / 42.0,
            "m3/d": _BBL_PER_M3,
            "Sm3/d": _BBL_PER_M3,
            "m3/h": _BBL_PER_M3 * 24.0,
            "L/min": _BBL_PER_M3 * 1e-3 * 1440.0,
        },
    ),
    _kind(
        "gas_rate", "Mscf/d", "gas rate at standard conditions",
        {
            "Mscf/d": 1.0,
            "Mcf/d": 1.0,
            "mscfd": 1.0,
            "scf/d": 1e-3,
            "scfd": 1e-3,
            "MMscf/d": 1e3,
            "MMcf/d": 1e3,
            "mmscfd": 1e3,
            "Sm3/d": _FT3_PER_M3 * 1e-3,
            "m3/d": _FT3_PER_M3 * 1e-3,
            "kSm3/d": _FT3_PER_M3,
            "MSm3/d": _FT3_PER_M3 * 1e3,
            "1e6Sm3/d": _FT3_PER_M3 * 1e3,
        },
    ),
    _kind(
        "gor", "scf/stb", "gas-oil ratio",
        {
            "scf/stb": 1.0,
            "scf/bbl": 1.0,
            "Sm3/Sm3": _FT3_PER_BBL,
            "m3/m3": _FT3_PER_BBL,
            "Mscf/stb": 1e3,
        },
    ),
    _kind(
        "density", "lbm/ft3", "mass density",
        {
            "lbm/ft3": 1.0,
            "lb/ft3": 1.0,
            "ppg": _LBM_PER_FT3_PER_PPG,
            "lbm/gal": _LBM_PER_FT3_PER_PPG,
            "lb/gal": _LBM_PER_FT3_PER_PPG,
            "kg/m3": _LBM_PER_FT3_PER_KG_PER_M3,
            "g/cm3": _LBM_PER_FT3_PER_KG_PER_M3 * 1e3,
            "g/cc": _LBM_PER_FT3_PER_KG_PER_M3 * 1e3,
            "sg": _LBM_PER_FT3_PER_KG_PER_M3 * 1e3,
        },
    ),
    _kind(
        "mud_weight", "ppg", "kill / drilling fluid density",
        {
            "ppg": 1.0,
            "lbm/gal": 1.0,
            "lb/gal": 1.0,
            "lbm/ft3": 1.0 / _LBM_PER_FT3_PER_PPG,
            "lb/ft3": 1.0 / _LBM_PER_FT3_PER_PPG,
            "kg/m3": _LBM_PER_FT3_PER_KG_PER_M3 / _LBM_PER_FT3_PER_PPG,
            "g/cm3": _LBM_PER_FT3_PER_KG_PER_M3 * 1e3 / _LBM_PER_FT3_PER_PPG,
            "g/cc": _LBM_PER_FT3_PER_KG_PER_M3 * 1e3 / _LBM_PER_FT3_PER_PPG,
            "sg": _LBM_PER_FT3_PER_KG_PER_M3 * 1e3 / _LBM_PER_FT3_PER_PPG,
        },
    ),
    _kind(
        "viscosity", "cp", "dynamic viscosity",
        {"cp": 1.0, "cP": 1.0, "mPa.s": 1.0, "mPa*s": 1.0, "Pa.s": 1e3, "Pa*s": 1e3},
    ),
    _kind(
        "permeability", "md", "permeability",
        {"md": 1.0, "mD": 1.0, "d": 1e3, "darcy": 1e3, "ud": 1e-3, "microdarcy": 1e-3},
    ),
    _kind(
        "compressibility", "1/psi", "compressibility",
        {
            "1/psi": 1.0,
            "psi-1": 1.0,
            "1/bar": 1.0 / _PSI_PER_BAR,
            "1/kPa": 1.0 / _PSI_PER_KPA,
            "1/Pa": 1e3 / _PSI_PER_KPA,
        },
    ),
    _kind(
        "salinity", "ppm", "total dissolved solids by weight",
        {"ppm": 1.0, "mg/kg": 1.0, "mg/L": 1.0, "g/L": 1e3, "wt%": 1e4, "%": 1e4},
    ),
    _kind(
        "time", "day", "time",
        {
            "day": 1.0, "d": 1.0, "days": 1.0,
            "hr": 1.0 / 24.0, "h": 1.0 / 24.0, "hour": 1.0 / 24.0,
            "min": 1.0 / 1440.0, "s": 1.0 / 86400.0, "sec": 1.0 / 86400.0,
            "yr": 365.25, "year": 365.25,
        },
    ),
    _kind(
        "power", "hp", "hydraulic horsepower",
        {"hp": 1.0, "hhp": 1.0, "kW": _HP_PER_KW, "W": _HP_PER_KW * 1e-3, "MW": _HP_PER_KW * 1e3},
    ),
    _kind(
        "velocity", "ft/s", "velocity",
        {"ft/s": 1.0, "fps": 1.0, "m/s": _FT_PER_M, "ft/min": 1.0 / 60.0},
    ),
    _kind(
        "volume", "bbl", "volume",
        {"bbl": 1.0, "stb": 1.0, "m3": _BBL_PER_M3, "ft3": 1.0 / _FT3_PER_BBL, "gal": 1.0 / 42.0, "L": _BBL_PER_M3 * 1e-3},
    ),
    _kind(
        "fraction", "frac", "dimensionless fraction 0..1",
        {"frac": 1.0, "fraction": 1.0, "-": 1.0, "": 1.0, "%": 0.01, "pct": 0.01, "percent": 0.01, "mol%": 0.01, "vol%": 0.01},
    ),
]

KINDS: dict[str, Kind] = {k.name: k for k in _KIND_LIST}


def _get_kind(kind: str | Kind) -> Kind:
    if isinstance(kind, Kind):
        return kind
    try:
        return KINDS[kind]
    except KeyError:
        raise UnitError(f"unknown quantity kind {kind!r} (known: {', '.join(KINDS)})") from None


def canonical_unit(kind: str | Kind) -> str:
    return _get_kind(kind).canonical


def units_for(kind: str | Kind) -> list[str]:
    return sorted(_get_kind(kind).units)


def _infer_kind(from_unit: str, to_unit: str) -> Kind:
    f, t = normalize_unit(from_unit), normalize_unit(to_unit)
    matches = [k for k in _KIND_LIST if f in k.units and t in k.units]
    if not matches:
        raise UnitError(f"no quantity kind accepts both {from_unit!r} and {to_unit!r}")
    if len(matches) > 1:
        raise UnitError(
            f"units {from_unit!r} -> {to_unit!r} are ambiguous between kinds "
            f"{[k.name for k in matches]}; pass kind= explicitly"
        )
    return matches[0]


# --- conversion ---------------------------------------------------------------

def convert(value: Any, from_unit: str, to_unit: str, kind: str | Kind | None = None) -> Any:
    """Convert ``value`` (scalar or numpy array) from one unit to another.

    ``kind`` may be omitted when the pair of units belongs to exactly one kind.
    """
    k = _get_kind(kind) if kind is not None else _infer_kind(from_unit, to_unit)
    src, dst = k.unit(from_unit), k.unit(to_unit)
    if src == dst:
        return value
    canonical = value * src.scale + src.offset
    return (canonical - dst.offset) / dst.scale


def to_internal(value: Any, unit: str, kind: str | Kind) -> Any:
    """Convert from ``unit`` to the canonical unit of ``kind``."""
    u = _get_kind(kind).unit(unit)
    if u.scale == 1.0 and u.offset == 0.0:
        return value
    return value * u.scale + u.offset


def from_internal(value: Any, unit: str, kind: str | Kind) -> Any:
    """Convert from the canonical unit of ``kind`` to ``unit``."""
    u = _get_kind(kind).unit(unit)
    if u.scale == 1.0 and u.offset == 0.0:
        return value
    return (value - u.offset) / u.scale


_QUANTITY_RE = re.compile(
    r"^\s*(?P<num>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(?P<unit>\S.*?)?\s*$"
)


def parse_quantity(text: str) -> tuple[float, str | None]:
    """Split ``"3500 psig"`` into ``(3500.0, "psig")``. Unit is None if absent."""
    m = _QUANTITY_RE.match(text)
    if not m:
        raise UnitError(f"cannot parse quantity {text!r}; expected '<number> [unit]'")
    return float(m.group("num")), m.group("unit")


def coerce(value: Any, kind: str | Kind) -> float:
    """Turn any accepted input form into a float in the canonical unit of ``kind``.

    Accepted forms:
      - number: already in canonical units
      - "3500 psig", "150 degC", "12 %": number with unit
      - {"value": 3500, "unit": "psig"} or (3500, "psig")
    """
    k = _get_kind(kind)
    if isinstance(value, bool):
        raise UnitError(f"boolean is not a valid {k.name}")
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):  # numpy scalar
        return float(value.item())
    if isinstance(value, str):
        num, unit = parse_quantity(value)
        return float(to_internal(num, unit, k)) if unit else num
    if isinstance(value, dict):
        try:
            num, unit = value["value"], value.get("unit")
        except KeyError:
            raise UnitError(f"quantity dict needs a 'value' key: {value!r}") from None
        if isinstance(num, str):
            num, inner_unit = parse_quantity(num)
            unit = unit or inner_unit
        num = float(num)
        return float(to_internal(num, unit, k)) if unit else num
    if isinstance(value, (tuple, list)) and len(value) == 2:
        num, unit = value
        return float(to_internal(float(num), unit, k))
    raise UnitError(f"cannot interpret {value!r} as a {k.name}")
