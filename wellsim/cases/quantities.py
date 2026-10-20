"""Unit-aware pydantic field types.

Each alias is a ``float`` in the canonical oilfield unit of its kind. Inputs may
be a bare number (taken as canonical), a string such as ``"250 bar"``, or a
``{"value": .., "unit": ..}`` mapping. Conversion happens here, at the schema
boundary, so everything downstream sees canonical floats.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, Field

from wellsim.core import units


def quantity(kind: str):
    """Annotated float type that coerces any accepted input form to canonical units."""
    canonical = units.canonical_unit(kind)

    def _coerce(value: Any) -> float:
        return units.coerce(value, kind)

    _coerce.__name__ = f"coerce_{kind}"
    return Annotated[
        float,
        BeforeValidator(_coerce),
        Field(json_schema_extra={"kind": kind, "unit": canonical}),
    ]


Pressure = quantity("pressure")                # psia
DPressure = quantity("dpressure")              # psi
Temperature = quantity("temperature")          # degF
Length = quantity("length")                    # ft
Diameter = quantity("diameter")                # in
LiquidRate = quantity("liquid_rate")           # stb/d
GasRate = quantity("gas_rate")                 # Mscf/d
GasOilRatio = quantity("gor")                  # scf/stb
Density = quantity("density")                  # lbm/ft3
MudWeight = quantity("mud_weight")             # ppg
Viscosity = quantity("viscosity")              # cp
Permeability = quantity("permeability")        # md
Compressibility = quantity("compressibility")  # 1/psi
Salinity = quantity("salinity")                # ppm
Time = quantity("time")                        # day
Power = quantity("power")                      # hp
Velocity = quantity("velocity")                # ft/s
Volume = quantity("volume")                    # bbl
Fraction = Annotated[quantity("fraction"), Field(ge=0.0, le=1.0)]

__all__ = [
    "quantity", "Pressure", "DPressure", "Temperature", "Length", "Diameter",
    "LiquidRate", "GasRate", "GasOilRatio", "Density", "MudWeight", "Viscosity",
    "Permeability", "Compressibility", "Salinity", "Time", "Power", "Velocity",
    "Volume", "Fraction",
]
