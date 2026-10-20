"""Flowing temperature profile (spec 4.4).

Stage 1: linear in measured depth between the reservoir temperature at the inflow
node and the flowing exit temperature at surface. Below the inflow node the
temperature is held at reservoir temperature. Upgrade path: Hasan-Kabir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wellsim.cases.schema import Case

__all__ = ["LinearTemperature", "make_temperature"]


@dataclass(frozen=True)
class LinearTemperature:
    t_res: float  # degF at md_res
    t_exit: float  # degF at md = 0
    md_res: float  # ft

    def at(self, md: float) -> float:
        if md >= self.md_res:
            return self.t_res
        if md <= 0.0:
            return self.t_exit
        return self.t_exit + (self.t_res - self.t_exit) * md / self.md_res

    def gradient(self) -> float:
        """dT/dMD, degF/ft, positive when temperature rises with depth."""
        return (self.t_res - self.t_exit) / self.md_res

    @property
    def model(self) -> str:
        return "linear"


def make_temperature(case: "Case") -> LinearTemperature:
    # The prescribed linear profile is what 'profile' (alias 'linear') uses; the solver also
    # builds it when k_override forces the pre-3b path on an 'adiabatic' case.
    tm = case.temperature
    t_res = case.reservoir.temperature
    t_exit = tm.exit_temperature if tm.exit_temperature is not None else t_res - tm.default_exit_loss
    return LinearTemperature(t_res=t_res, t_exit=t_exit, md_res=case.reservoir.depth_md)
