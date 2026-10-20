"""Pressure-gradient correlations, in build order: single-phase gas (M2), Hagedorn-Brown,
Beggs-Brill, Gray, Duns-Ros, Teles-Waltrich (LSU/BOEM 2018) (M3)."""

from wellsim.core.vlp.correlations import single_phase_gas

CORRELATIONS = {single_phase_gas.NAME: single_phase_gas}

__all__ = ["CORRELATIONS", "single_phase_gas"]
