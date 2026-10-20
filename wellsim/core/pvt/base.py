"""PVT interface (spec 4.1). Everything above this layer talks to :class:`PVTModel`.

Canonical units throughout (see ``wellsim.core.units``): pressure psia,
temperature degF, density lbm/ft3, viscosity cp, Rs scf/stb, Bo and Bw rb/stb,
Bg ft3/scf, surface tension dyne/cm, compressibility 1/psi.

Range policy (spec section 6, "no silent extrapolation"): every correlation
checks its inputs against the range of the data it was fitted to and emits a
:class:`RangeWarning` when it extrapolates. It never raises and never clamps.
The solver collects these with :func:`collect_range_warnings` and attaches
them to the result.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ClassVar, Iterator

__all__ = [
    "RangeWarning",
    "check_range",
    "collect_range_warnings",
    "PVTState",
    "PVTModel",
    "R_PSIA_FT3",
    "M_AIR",
    "T_SC_R",
    "P_SC",
    "RHO_WATER_SC",
    "rankine",
]

R_PSIA_FT3 = 10.7316  # psia ft3 / (lbmol degR)
M_AIR = 28.9647  # lbm/lbmol
T_SC_R = 519.67  # 60 degF
P_SC = 14.696  # psia
RHO_WATER_SC = 62.368  # lbm/ft3, pure water at standard conditions (McCain)


def rankine(t_f: float) -> float:
    return t_f + 459.67


class RangeWarning(UserWarning):
    """A correlation was evaluated outside the range of its source data."""

    def __init__(self, correlation: str, variable: str, value: float, lo: float, hi: float, unit: str = ""):
        self.correlation = correlation
        self.variable = variable
        self.value = float(value)
        self.lo = lo
        self.hi = hi
        self.unit = unit
        u = f" {unit}" if unit else ""
        super().__init__(
            f"{correlation}: {variable} = {value:.6g}{u} is outside the published range "
            f"{lo:.6g} to {hi:.6g}{u}; result is an extrapolation"
        )

    @property
    def key(self) -> tuple[str, str]:
        return (self.correlation, self.variable)


_ACTIVE_SEEN: set[tuple[str, str]] | None = None  # set while a collect_range_warnings block runs


def check_range(correlation: str, variable: str, value: float, lo: float, hi: float, unit: str = "") -> bool:
    """Warn (never raise) when ``value`` is outside ``[lo, hi]``. Returns True if in range.

    Inside a ``collect_range_warnings`` block only the first occurrence of each
    (correlation, variable) is issued: the collector keeps one per key anyway, and the
    adiabatic march can hit the same range thousands of times per solve (M2.5-3b)."""
    if math.isnan(value) or value < lo or value > hi:
        seen = _ACTIVE_SEEN
        if seen is not None:
            key = (correlation, variable)
            if key in seen:
                return False
            seen.add(key)
        warnings.warn(RangeWarning(correlation, variable, value, lo, hi, unit), stacklevel=3)
        return False
    return True


@contextmanager
def collect_range_warnings() -> Iterator[list[RangeWarning]]:
    """Record every distinct RangeWarning raised inside the block into the yielded list.

    Warnings of other categories are re-emitted after the block so nothing is lost.
    Distinct means distinct (correlation, variable); the first occurrence is kept.
    """
    global _ACTIVE_SEEN
    collected: list[RangeWarning] = []
    outer = _ACTIVE_SEEN
    _ACTIVE_SEEN = set()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", RangeWarning)
        try:
            yield collected
        finally:
            _ACTIVE_SEEN = outer
            seen: set[tuple[str, str]] = set()
            others = []
            for w in record:
                if isinstance(w.message, RangeWarning):
                    if w.message.key not in seen:
                        seen.add(w.message.key)
                        collected.append(w.message)
                else:
                    others.append(w)
    for w in others:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


@dataclass(slots=True)
class PVTState:
    """All fluid properties at one (p, T). Fields that do not apply to the fluid are None."""

    p: float  # psia
    t: float  # degF
    # gas phase (always present; for black oil this is the free gas)
    z: float
    bg: float  # ft3/scf
    rho_g: float  # lbm/ft3
    mu_g: float  # cp
    gas_sg: float
    # oil phase
    pb: float | None = None  # psia
    rs: float | None = None  # scf/stb
    bo: float | None = None  # rb/stb
    rho_o: float | None = None  # lbm/ft3
    mu_o: float | None = None  # cp
    co: float | None = None  # 1/psi
    sigma_og: float | None = None  # dyne/cm
    # water phase
    rsw: float | None = None  # scf/stb
    bw: float | None = None  # rb/stb
    rho_w: float | None = None  # lbm/ft3
    mu_w: float | None = None  # cp
    cw: float | None = None  # 1/psi
    sigma_gw: float | None = None  # dyne/cm
    # real-gas derivatives from the same Z solve (M2.5-3a; gas models only)
    tpr: float | None = None  # pseudo-reduced temperature
    ppr: float | None = None  # pseudo-reduced pressure
    rho_r: float | None = None  # reduced density of the Z equation
    dz_drho_r: float | None = None  # (dZ/drho_r)_Tpr
    dz_dtpr: float | None = None  # (dZ/dTpr)_rho_r
    dp_drho_t: float | None = None  # (dp/drho)_T, psia per lbm/ft3 = (R T / M) (Z + rho_r dZ/drho_r)
    # bookkeeping
    saturated: bool | None = None  # p < pb (black oil only)
    notes: list[str] = field(default_factory=list)


class PVTModel(ABC):
    """Fluid property model. Subclasses are immutable value objects: no state besides inputs.

    The VLP march calls :meth:`state` once per segment; everything it needs is in the
    returned :class:`PVTState`. Individual property methods exist for tests and IPR.
    """

    fluid_type: ClassVar[str]

    @property
    def molar_mass(self) -> float:
        """Molar mass, lbm/lbmol. The march needs this and only this to go from molar
        thermodynamic properties to per-lbm ones; it does NOT need a gas gravity. For a
        gas-gravity model it is M_air x gas_sg; for air it is the air molar mass directly.
        Split out from ``gas_sg`` (g2-4) because a model can have a molar mass without
        having a meaningful specific gravity, which is exactly the air case."""
        return M_AIR * self.gas_sg

    def thermo(self, p: float, t: float, state: "PVTState | None" = None):
        """:class:`~wellsim.core.pvt.real_gas_thermo.ThermoState` at (p, t): h, cp, cv,
        mu_jt, c_s, rho_p, rho_t.

        This is the whole thermodynamic interface the two-phase march needs. Each model
        builds it however it likes: the gas-gravity model through the DAK departures, an
        ideal gas in closed form. Before g2-4 the march reached into DAK-specific fields
        of :class:`PVTState` (tpr, rho_r, dZ/drho_r, dZ/dTpr) to do this itself, which
        meant only a DAK model could be marched.

        ``state`` lets a caller that already has the :class:`PVTState` at (p, t) hand it
        over, so the march keeps doing ONE Z solve per gradient evaluation rather than
        two. Models that need no state ignore it."""
        raise NotImplementedError(f"{type(self).__name__} has no thermodynamic model")

    @abstractmethod
    def state(self, p: float, t: float, warm: bool = False) -> PVTState:
        """All properties at pressure ``p`` (psia) and temperature ``t`` (degF).

        ``warm`` is a solver hint, not a request for extra outputs: it lets an iterative
        Z method start from the previous call's root. Models that do not iterate ignore
        it. It is in the base signature so callers need no isinstance check to pass it."""

    @abstractmethod
    def z(self, p: float, t: float) -> float: ...

    @abstractmethod
    def gas_density(self, p: float, t: float) -> float: ...

    @abstractmethod
    def gas_viscosity(self, p: float, t: float) -> float: ...

    @abstractmethod
    def bg(self, p: float, t: float) -> float: ...

    def k_ratio(self, t: float) -> float:
        """Ideal-gas heat capacity ratio cp0/cv0 at temperature ``t`` (degF), used for the
        sonic velocity. Gas models implement it; others raise."""
        raise NotImplementedError(f"{type(self).__name__} has no heat-capacity model")
