"""Air as an ideal gas, for the G2 LSU/PERTT gate. Its own path, not the gas-gravity one.

Why a separate module
    The LSU PERTT Lab runs are AIR and water at about 14.7 psia (docs/validation/
    G2-lsu-boem.md, answered in out/g2-sources.md). Air is not a hydrocarbon gas: the
    Sutton pseudo-criticals and the Dranchuk-Abou-Kassem Z correlation have no meaning
    for it, and routing air through ``dry_gas.DryGasPVT`` by picking a specific gravity
    of 1.0 would silently hand DAK a fluid it was never fitted to. This module therefore
    does NOT subclass the gas-gravity model and does not reuse the pseudo-composition
    heat-capacity path; :func:`z` returns exactly 1 and :meth:`AirIdealGas.dak_z` refuses.

    At 14.7 psia and ambient temperature the reduced pressure of air is about 0.04, where
    the ideal-gas law is good to far better than 0.1%, so nothing is lost by it. The model
    is deliberately restricted: it warns above 150 psia, where the departure starts to
    matter and a real EOS would be wanted.

Forms, all MEMORY (standard references, none consulted in-session)
    density        rho = 144 p / (R_air T),  R_air = 53.3533 ft lbf/(lbm degR)
    viscosity      Sutherland, mu = mu_0 (T_0 + S)/(T + S) (T/T_0)^1.5, with
                   mu_0 = 1.716e-5 Pa s at T_0 = 273.15 K and S = 110.4 K
    heat capacity  k = cp/cv = 1.4, constant (diatomic, and within 0.5% of the real
                   value over the whole range these experiments cover)
    sound speed    c = sqrt(k g_c R_air T)

Every one of those is on the hand-check list in docs/PROVENANCE.md.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

from wellsim.core.pvt.base import RangeWarning

__all__ = [
    "AirIdealGas", "R_AIR", "M_AIR_LBM_LBMOL", "K_AIR", "SUTHERLAND_MU0_CP",
    "SUTHERLAND_T0_R", "SUTHERLAND_S_R", "MAX_PRESSURE_PSIA",
]

NAME = "air_ideal_gas"

M_AIR_LBM_LBMOL = 28.9647           # lbm/lbmol, standard dry air (MEMORY)
R_UNIVERSAL = 1545.3489             # ft lbf/(lbmol degR) (MEMORY)
R_AIR = R_UNIVERSAL / M_AIR_LBM_LBMOL   # 53.3533 ft lbf/(lbm degR)
K_AIR = 1.4                          # cp/cv, diatomic ideal gas (MEMORY)
G_C = 32.174                         # lbm ft/(lbf s2)

SUTHERLAND_MU0_CP = 1.716e-5 * 1000.0   # 1 Pa s = 1000 cp -> 0.01716 cp
SUTHERLAND_T0_R = 273.15 * 1.8          # 491.67 degR
SUTHERLAND_S_R = 110.4 * 1.8            # 198.72 degR
MAX_PRESSURE_PSIA = 150.0               # above this the ideal-gas assumption is warned


def _rankine(t_degf: float) -> float:
    return t_degf + 459.67


@dataclass(frozen=True)
class AirIdealGas:
    """Air as an ideal gas. Immutable; no state beyond the flag below.

    ``strict`` (default True) makes the pressure ceiling a warning rather than silence.
    """

    strict: bool = True
    fluid_type: str = "air"

    # --- the refusal ------------------------------------------------------------------
    @staticmethod
    def dak_z(*_args, **_kwargs) -> float:
        """Refused on purpose. DAK is a hydrocarbon-gas correlation fitted to
        Standing-Katz; air has no meaningful pseudo-criticals in that framework, and a
        gas gravity of 1.0 would let it run and return a number that means nothing."""
        raise NotImplementedError(
            "Dranchuk-Abou-Kassem is a hydrocarbon-gas Z correlation and is refused for air: "
            "air has no Sutton pseudo-criticals, and forcing it through the gas-gravity path "
            "(gas_sg = 1.0) would return a Z that is not a property of air. Use z() = 1, which "
            "is accurate to better than 0.1% at the pressures this model allows"
        )

    def _check(self, p: float, t: float) -> None:
        if self.strict and p > MAX_PRESSURE_PSIA:
            warnings.warn(
                RangeWarning("air_ideal_gas", "pressure", p, 0.0, MAX_PRESSURE_PSIA, "psia"),
                stacklevel=3,
            )

    # --- properties -------------------------------------------------------------------
    def z(self, p: float = 14.7, t: float = 60.0) -> float:
        """Exactly 1. Air at the pressures this model allows departs by under 0.1%."""
        self._check(p, t)
        return 1.0

    def gas_density(self, p: float, t: float) -> float:
        """lbm/ft3 from rho = 144 p / (R_air T), p psia, T degF."""
        self._check(p, t)
        return 144.0 * p / (R_AIR * _rankine(t))

    def gas_viscosity(self, p: float, t: float) -> float:
        """cp, Sutherland. Pressure-independent, which is right to well under 1% here."""
        self._check(p, t)
        tr = _rankine(t)
        return (SUTHERLAND_MU0_CP * (SUTHERLAND_T0_R + SUTHERLAND_S_R) / (tr + SUTHERLAND_S_R)
                * (tr / SUTHERLAND_T0_R) ** 1.5)

    def k_ratio(self, t: float = 60.0) -> float:
        """cp/cv = 1.4, constant."""
        return K_AIR

    def sound_speed(self, t: float) -> float:
        """ft/s, c = sqrt(k g_c R_air T)."""
        return math.sqrt(K_AIR * G_C * R_AIR * _rankine(t))

    def bg(self, p: float, t: float) -> float:
        """ft3 at (p, T) per ft3 at standard conditions (14.696 psia, 60 degF)."""
        return self.gas_density(14.696, 60.0) / self.gas_density(p, t)

    def __repr__(self) -> str:
        return f"AirIdealGas(R={R_AIR:.4f} ft lbf/(lbm degR), k={K_AIR})"

    # --- the march interface (g2-4) ---------------------------------------------------
    #
    # Everything below is CLOSED FORM. The real-gas departures are IDENTICALLY ZERO for an
    # ideal gas by definition, not by approximation: h - h_ideal = 0 and the
    # Joule-Thomson coefficient is 0 because (dh/dp)_T vanishes when Z is constant. The
    # DAK path computes the same quantities through a departure integral that evaluates to
    # zero here; it cannot be run for air only because it is written in reduced
    # coordinates (Tpr, rho_r) that air does not possess.

    @property
    def molar_mass(self) -> float:
        """lbm/lbmol. Air has a molar mass but no meaningful specific gravity, which is
        why the march asks for this rather than for gas_sg (g2-4)."""
        return M_AIR_LBM_LBMOL

    @property
    def gas_sg(self) -> float:
        """1.0 by definition: specific gravity is relative to air. Present so anything
        that still reads gas_sg keeps working; nothing about air should be derived from
        it, and the hydrocarbon correlations that take a gravity must not be handed this."""
        return 1.0

    def state(self, p: float, t: float, warm: bool = False) -> "PVTState":
        """A :class:`PVTState` with Z = 1 exactly and NO reduced coordinates.

        ``tpr``, ``ppr``, ``rho_r`` and both DAK partials are left None on purpose: air has
        no Sutton pseudo-criticals, so they are undefined rather than unknown, and
        `exit.is_gas_state` treats a state without them as outside the scope of its
        dense-phase refusal (which is the correct verdict, not a loophole). ``warm`` is
        accepted and ignored: there is no iteration to warm-start.
        """
        from wellsim.core.pvt.base import PVTState

        self._check(p, t)
        rho = self.gas_density(p, t)
        return PVTState(
            p=p, t=t, z=1.0, bg=self.bg(p, t), rho_g=rho, mu_g=self.gas_viscosity(p, t),
            gas_sg=1.0,
            # (dp/drho)_T = R_air T / M = p / rho for an ideal gas, psia per lbm/ft3
            dp_drho_t=p / rho,
        )

    def thermo(self, p: float, t: float, state: "PVTState | None" = None) -> "ThermoState":
        """:class:`ThermoState` in closed form, molar units to match the DAK path.

            cp    = 7/2 R          diatomic ideal gas, Btu/(lbmol degR)
            cv    = cp - R = 5/2 R so cp/cv = 1.4 exactly, consistent with K_AIR
            h     = cp (T - T_ref) measured from the same reference as the DAK path
            mu_jt = 0              EXACTLY: an ideal gas has no Joule-Thomson effect
            c_s   = sqrt(k g_c R_air T)   the same value as :meth:`sound_speed`
            rho_p = (drho/dp)_T = rho / p
            rho_t = (drho/dT)_p = -rho / T

        The 7/2 R is the diatomic ideal-gas value, not a correlation: it gives
        6.9506 Btu/(lbmol degR), against a measured air cp of about 6.95 at 70 degF, and
        stays within about 1% to 600 degF, far outside anything these experiments reach.
        MEMORY for the measured comparison only.
        """
        from wellsim.core.pvt.real_gas_thermo import R, T_REF, ThermoState

        self._check(p, t)
        t_r = _rankine(t)
        rho = self.gas_density(p, t)
        cp = 3.5 * R
        cv = cp - R
        return ThermoState(
            h_dep=0.0,                      # zero by definition for an ideal gas
            h=cp * (t_r - _rankine(T_REF)),
            cp=cp,
            cv=cv,
            mu_jt=0.0,                      # zero by definition, not by approximation
            c_s=self.sound_speed(t),
            rho_p=rho / p,
            rho_t=-rho / t_r,
        )
