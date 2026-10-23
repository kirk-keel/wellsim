"""Inflow performance: gas.py (pseudo-pressure, non-Darcy skin; M2) and
oil.py (Darcy + Vogel composite; M4)."""

from wellsim.core.ipr.gas import GasIPR, PseudoPressure
from wellsim.core.ipr.oil import OilIPR
from wellsim.core.ipr.partial_penetration import papatzacos_skin

__all__ = ["GasIPR", "OilIPR", "PseudoPressure", "papatzacos_skin"]

