"""PVT behind an interface (spec 4.1).

``make_pvt(fluid, correlations)`` builds the right :class:`PVTModel` from a case's
fluid block and correlation options. VLP, IPR and the kill engine only see
:class:`PVTModel` and :class:`PVTState`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wellsim.core.pvt.base import (
    PVTModel,
    PVTState,
    RangeWarning,
    check_range,
    collect_range_warnings,
)
from wellsim.core.pvt.black_oil import BlackOilPVT
from wellsim.core.pvt.dry_gas import DryGasPVT

if TYPE_CHECKING:
    from wellsim.cases.schema import BlackOil, Correlations, DryGas

__all__ = [
    "PVTModel",
    "PVTState",
    "RangeWarning",
    "check_range",
    "collect_range_warnings",
    "BlackOilPVT",
    "DryGasPVT",
    "make_pvt",
]


def make_pvt(fluid: "BlackOil | DryGas", correlations: "Correlations | None" = None) -> PVTModel:
    """Build a PVT model from a case fluid block and (optional) correlation choices."""
    from wellsim.cases.schema import Correlations

    c = correlations if correlations is not None else Correlations()
    if fluid.type == "black_oil":
        return BlackOilPVT(
            oil_api=fluid.oil_api,
            gas_sg=fluid.gas_sg,
            gor=fluid.gor,
            water_cut=fluid.water_cut,
            water_salinity=fluid.water_salinity,
            bubble_point=fluid.bubble_point,
            rs_pb_method=c.rs_pb,
            bo_method=c.bo,
            z_method=c.gas_z,
        )
    if fluid.type == "dry_gas":
        return DryGasPVT(fluid.gas_sg, fluid.n2, fluid.co2, fluid.h2s, z_method=c.gas_z)
    raise ValueError(f"unknown fluid type {fluid.type!r}")
