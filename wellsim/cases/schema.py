"""Case schema (spec section 3). All stored values are in canonical oilfield units.

Structural validation only lives here (depth ordering, geometry consistency,
flow-path / string agreement). Physics-range warnings belong to the solver.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wellsim.cases.quantities import (
    Compressibility,
    Diameter,
    DPressure,
    Fraction,
    GasOilRatio,
    GasRate,
    Length,
    LiquidRate,
    MudWeight,
    Permeability,
    Power,
    Pressure,
    Salinity,
    Temperature,
    Time,
    Viscosity,
)

STEEL_ROUGHNESS_IN = 0.0018  # commercial steel, inches (0.00015 ft)
OPEN_HOLE_ROUGHNESS_IN = 0.05
MIN_KILL_FLUID_PPG = 6.0   # below diesel; anything lower is a units mistake
MAX_KILL_FLUID_PPG = 25.0  # above the heaviest practical mud; lbm/ft3 values start at ~50
_DEPTH_TOL_FT = 0.01
LBM_PER_FT3_PER_PPG = 7.480519480  # 231 in3/gal x 1728 in3/ft3 inverse; matches core/units.py


class WellsimModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --- reservoir ------------------------------------------------------------------

class Reservoir(WellsimModel):
    pressure: Annotated[Pressure, Field(
        gt=0, description="Reservoir pressure at datum_tvd. Bare number is psia; '5000 psig' adds 14.7"
    )]
    temperature: Annotated[Temperature, Field(gt=-100)]
    depth_md: Annotated[Length, Field(gt=0, description="MD of the inflow node (middle of exposed interval)")]
    datum_tvd: Optional[Annotated[Length, Field(gt=0)]] = Field(
        default=None,
        description="TVD the reservoir pressure is referenced to. None means the TVD of depth_md from the survey",
    )
    net_pay: Annotated[Length, Field(gt=0)]
    permeability: Annotated[Permeability, Field(gt=0)]
    wellbore_radius: Annotated[Length, Field(gt=0)]
    drainage_radius: Annotated[Length, Field(gt=0)]
    skin: float = 0.0
    exposed_interval: Optional[Annotated[Length, Field(gt=0)]] = Field(
        default=None, description="h_p: open (perforated) interval length; None means full net pay"
    )
    exposed_top_offset: Annotated[Length, Field(ge=0)] = Field(
        default=0.0,
        description="h_1: distance from the top of the zone to the top of the open interval; 0 = drilled in from the top",
    )
    kv_kh: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="Vertical/horizontal permeability ratio for the partial-penetration skin; None with a partial interval uses 0.1 and warns",
    )
    porosity: Optional[Fraction] = Field(default=None, description="Needed only for transient IPR")
    total_compressibility: Optional[Annotated[Compressibility, Field(gt=0)]] = Field(
        default=None, description="Needed only for transient IPR"
    )
    time_since_start: Optional[Annotated[Time, Field(gt=0)]] = Field(
        default=None, description="Time since blowout start; enables transient IPR"
    )
    non_darcy_beta: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None, description="Override for gas non-Darcy beta, 1/ft; None uses the correlation"
    )

    @model_validator(mode="after")
    def _check(self) -> "Reservoir":
        if self.wellbore_radius >= self.drainage_radius:
            raise ValueError("wellbore_radius must be smaller than drainage_radius")
        if self.exposed_interval is not None and self.exposed_interval > self.net_pay + _DEPTH_TOL_FT:
            raise ValueError("exposed_interval cannot exceed net_pay")
        h_p = self.net_pay if self.exposed_interval is None else self.exposed_interval
        if self.exposed_top_offset + h_p > self.net_pay + _DEPTH_TOL_FT:
            raise ValueError("exposed_top_offset + exposed_interval cannot exceed net_pay")
        if self.time_since_start is not None and (self.porosity is None or self.total_compressibility is None):
            raise ValueError("transient IPR (time_since_start) needs porosity and total_compressibility")
        return self


# --- fluids ---------------------------------------------------------------------

class BlackOil(WellsimModel):
    type: Literal["black_oil"] = "black_oil"
    oil_api: Annotated[float, Field(gt=0, lt=100)]
    gas_sg: Annotated[float, Field(gt=0.5, lt=2.0, description="Gas specific gravity, air = 1")]
    gor: Annotated[GasOilRatio, Field(ge=0, description="Producing gas-oil ratio")]
    water_cut: Fraction = 0.0
    water_salinity: Salinity = Field(default=0.0, description="TDS; 0 = fresh water")
    bubble_point: Optional[Annotated[Pressure, Field(gt=0)]] = Field(
        default=None, description="None means calculated from the Rs correlation"
    )


class DryGas(WellsimModel):
    type: Literal["dry_gas"] = "dry_gas"
    gas_sg: Annotated[float, Field(
        gt=0.5, lt=2.0,
        description="Well-stream gas specific gravity (air = 1): all produced gas plus the gas equivalent "
                    "of any liquid, not the separator gas gravity",
    )]
    n2: Fraction = 0.0
    co2: Fraction = 0.0
    h2s: Fraction = 0.0
    condensate_yield: Annotated[float, Field(ge=0, description="stb/MMscf; must be 0 in Stage 1")] = 0.0

    @model_validator(mode="after")
    def _check(self) -> "DryGas":
        if self.n2 + self.co2 + self.h2s >= 1.0:
            raise ValueError("n2 + co2 + h2s must be less than 1")
        if self.condensate_yield > 0:
            raise ValueError("wet gas (condensate_yield > 0) is not supported in Stage 1")
        return self


Fluid = Annotated[Union[BlackOil, DryGas], Field(discriminator="type")]

DEFAULT_SIGMA_DYNE_CM = 30.0  # gas-liquid surface tension used when none is given; always warned (M3-1)
DEFAULT_C_L_FT_S = 4900.0  # liquid speed of sound used when none is given; always warned (M3-2)


class Liquid(WellsimModel):
    """Explicit liquid-phase properties for the multiphase closure (M3). Canonical units:
    rho_l ppg (the schema's single density kind; '998 kg/m3' or '62.3 lbm/ft3' convert),
    mu_l cp, cp_l Btu/(lbm degF), sigma dyne/cm. Black-oil PVT will fill these from the
    fluid model in M4; until then they are entered directly. The core reads rho_l_lbm_ft3."""
    rho_l: Annotated[MudWeight, Field(
        ge=MIN_KILL_FLUID_PPG, le=MAX_KILL_FLUID_PPG,
        description="Liquid density, ppg (bare numbers outside 6 to 25 ppg are a units mistake)",
    )]
    mu_l: Annotated[Viscosity, Field(gt=0, description="Liquid viscosity, cp")]
    cp_l: Annotated[float, Field(gt=0, description="Liquid heat capacity, Btu/(lbm degF)")]
    sigma: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="Gas-liquid surface tension, dyne/cm. None means DEFAULT_SIGMA_DYNE_CM (30) with a "
        "warning on the result; never a silent default",
    )
    c_l: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="Liquid speed of sound, ft/s, for the homogeneous frozen (Wallis) mixture sound speed. "
        "None means DEFAULT_C_L_FT_S (4,900) with a warning on the result",
    )

    @property
    def rho_l_lbm_ft3(self) -> float:
        """Liquid density in the core's lbm/ft3 (1 ppg = 7.480519480 lbm/ft3, exact)."""
        return self.rho_l * LBM_PER_FT3_PER_PPG

    def c_l_or_default(self) -> tuple[float, Optional[str]]:
        """(c_l in ft/s, warning text or None)."""
        if self.c_l is not None:
            return float(self.c_l), None
        return DEFAULT_C_L_FT_S, (
            f"liquid.c_l not supplied; liquid speed of sound defaulted to {DEFAULT_C_L_FT_S:g} ft/s"
        )

    def sigma_or_default(self) -> tuple[float, Optional[str]]:
        """(sigma in dyne/cm, warning text or None). The warning must be attached to the result."""
        if self.sigma is not None:
            return float(self.sigma), None
        return DEFAULT_SIGMA_DYNE_CM, (
            f"liquid.sigma not supplied; gas-liquid surface tension defaulted to {DEFAULT_SIGMA_DYNE_CM:g} dyne/cm"
        )


class LiquidInflow(WellsimModel):
    """Liquid inflow for the gas-liquid nodal solve (M3-3). Exactly one mode:
    'wgr': q_l = water_gas_ratio (bbl/MMscf) x q_g(pwf); 'pi': q_l = J (p_r - pwf), floored at 0."""
    mode: Literal["wgr", "pi"]
    water_gas_ratio: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None, description="bbl/MMscf, mode 'wgr' only"
    )
    productivity_index: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None, description="J, stb/d per psi of drawdown, mode 'pi' only"
    )

    @model_validator(mode="after")
    def _check(self) -> "LiquidInflow":
        if self.mode == "wgr":
            if self.water_gas_ratio is None or self.productivity_index is not None:
                raise ValueError("liquid_inflow mode 'wgr' needs water_gas_ratio and no productivity_index")
        else:
            if self.productivity_index is None or self.water_gas_ratio is not None:
                raise ValueError("liquid_inflow mode 'pi' needs productivity_index and no water_gas_ratio")
        return self

    def liquid_rate(self, pwf: float, q_gas_mscfd: float, p_res: float) -> float:
        """stb/d at (pwf, q_g)."""
        if self.mode == "wgr":
            return self.water_gas_ratio * q_gas_mscfd / 1000.0
        return max(0.0, self.productivity_index * (p_res - pwf))


class DynamicKill(WellsimModel):
    """Dynamic kill with a Newtonian kill fluid injected into the blowing well (M3-4).
    Relief-well hydraulics and pump pressure are a later item (the M5 `kill` block)."""
    rho_k: Annotated[MudWeight, Field(
        ge=MIN_KILL_FLUID_PPG, le=MAX_KILL_FLUID_PPG,
        description="Kill-fluid density, ppg (bare numbers outside 6 to 25 ppg are a units mistake)",
    )]
    mu_k: Annotated[Viscosity, Field(gt=0, description="Kill-fluid viscosity, cp (Newtonian)")]
    cp_k: Annotated[float, Field(gt=0, description="Kill-fluid heat capacity, Btu/(lbm degF)")] = 1.0
    z_inj: Optional[Annotated[Length, Field(gt=0)]] = Field(
        default=None, description="Injection depth, ft TVD; None means the sandface"
    )
    rate_bpm: Optional[Annotated[float, Field(ge=0)]] = Field(
        default=None, description="Injection rate, bbl/min, to evaluate the well at (None: only q_k_min is searched)"
    )
    find_min_rate: bool = Field(
        default=True, description="Search q_k_min (10 to 30 s per solve); False reports only the static hold, "
        "the Blount-Soeiinah rate and, if rate_bpm is given, the solve at that rate"
    )
    sigma: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None, description="Kill-fluid gas-liquid surface tension, dyne/cm; None means 30 with a warning"
    )
    c_k: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None, description="Kill-fluid speed of sound, ft/s; None means 4,900 with a warning"
    )
    k_f: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="rw-3: kill-fluid thermal conductivity, Btu/(hr ft degF), for the Dittus-Boelter inside "
        "film coefficient. None means 0.4 (water-based) with a warning",
    )

    def sigma_or_default(self) -> tuple[float, Optional[str]]:
        if self.sigma is not None:
            return float(self.sigma), None
        return DEFAULT_SIGMA_DYNE_CM, f"dynamic_kill.sigma not supplied; defaulted to {DEFAULT_SIGMA_DYNE_CM:g} dyne/cm"

    def c_or_default(self) -> tuple[float, Optional[str]]:
        if self.c_k is not None:
            return float(self.c_k), None
        return DEFAULT_C_L_FT_S, f"dynamic_kill.c_k not supplied; defaulted to {DEFAULT_C_L_FT_S:g} ft/s"


# --- well -----------------------------------------------------------------------

class SurveyPoint(WellsimModel):
    md: Annotated[Length, Field(ge=0)]
    tvd: Annotated[Length, Field(ge=0)]


class HoleSection(WellsimModel):
    """Casing, liner, or open hole. Sections must tile 0..total_depth_md."""
    top_md: Annotated[Length, Field(ge=0)]
    bottom_md: Annotated[Length, Field(gt=0)]
    id: Annotated[Diameter, Field(gt=0)]
    kind: Literal["casing", "liner", "open_hole"] = "casing"
    roughness: Optional[Annotated[Diameter, Field(ge=0)]] = Field(
        default=None, description="None -> steel default, or open-hole default for kind=open_hole"
    )

    @model_validator(mode="after")
    def _check(self) -> "HoleSection":
        if self.bottom_md <= self.top_md:
            raise ValueError("bottom_md must be greater than top_md")
        if self.roughness is None:
            self.roughness = OPEN_HOLE_ROUGHNESS_IN if self.kind == "open_hole" else STEEL_ROUGHNESS_IN
        return self


class StringSection(WellsimModel):
    """Drillstring or tubing section. Sections must tile 0..string bottom."""
    top_md: Annotated[Length, Field(ge=0)]
    bottom_md: Annotated[Length, Field(gt=0)]
    od: Annotated[Diameter, Field(gt=0)]
    id: Annotated[Diameter, Field(gt=0)]
    roughness: Annotated[Diameter, Field(ge=0)] = STEEL_ROUGHNESS_IN

    @model_validator(mode="after")
    def _check(self) -> "StringSection":
        if self.bottom_md <= self.top_md:
            raise ValueError("bottom_md must be greater than top_md")
        if self.id >= self.od:
            raise ValueError("string id must be smaller than od")
        return self


def _check_tiling(sections, name: str, start: float = 0.0) -> None:
    expected = start
    for i, s in enumerate(sections):
        if abs(s.top_md - expected) > _DEPTH_TOL_FT:
            raise ValueError(
                f"{name} section {i} starts at {s.top_md} ft, expected {expected} ft "
                "(sections must be contiguous from surface)"
            )
        expected = s.bottom_md


class Well(WellsimModel):
    total_depth_md: Annotated[Length, Field(gt=0)]
    survey: Optional[list[SurveyPoint]] = Field(default=None, description="None means vertical (tvd = md)")
    hole: list[HoleSection] = Field(min_length=1)
    string: Optional[list[StringSection]] = Field(default=None, description="None means no string in hole")

    @property
    def string_bottom_md(self) -> Optional[float]:
        return self.string[-1].bottom_md if self.string else None

    def md_to_tvd(self, md: float) -> float:
        """TVD at a measured depth. Vertical well: identity. Otherwise linear interpolation
        between survey stations (minimum curvature lands with geometry.py in M2)."""
        if self.survey is None:
            return float(md)
        mds = [p.md for p in self.survey]
        tvds = [p.tvd for p in self.survey]
        return float(np.interp(md, mds, tvds))

    @model_validator(mode="after")
    def _check(self) -> "Well":
        td = self.total_depth_md
        if self.survey is not None:
            if len(self.survey) < 2:
                raise ValueError("survey needs at least two points")
            if self.survey[0].md > _DEPTH_TOL_FT or self.survey[0].tvd > _DEPTH_TOL_FT:
                raise ValueError("survey must start at md = tvd = 0")
            for a, b in zip(self.survey, self.survey[1:]):
                if b.md <= a.md:
                    raise ValueError("survey md must increase monotonically")
                if b.tvd < a.tvd - _DEPTH_TOL_FT:
                    raise ValueError("survey tvd must not decrease (horizontal is allowed, up-dip is not)")
                if (b.tvd - a.tvd) > (b.md - a.md) + _DEPTH_TOL_FT:
                    raise ValueError("survey tvd increment cannot exceed md increment")
            if self.survey[-1].md < td - _DEPTH_TOL_FT:
                raise ValueError("survey must extend to total_depth_md")
        _check_tiling(self.hole, "hole")
        if abs(self.hole[-1].bottom_md - td) > _DEPTH_TOL_FT:
            raise ValueError("hole sections must end at total_depth_md")
        if self.string is not None:
            if not self.string:
                raise ValueError("string list is empty; use null for no string")
            _check_tiling(self.string, "string")
            if self.string[-1].bottom_md > td + _DEPTH_TOL_FT:
                raise ValueError("string bottom cannot be below total_depth_md")
            for s in self.string:
                for h in self.hole:
                    overlap = min(s.bottom_md, h.bottom_md) - max(s.top_md, h.top_md)
                    if overlap > _DEPTH_TOL_FT and s.od >= h.id:
                        raise ValueError(
                            f"string od {s.od} in is not smaller than hole id {h.id} in between "
                            f"{max(s.top_md, h.top_md)} and {min(s.bottom_md, h.bottom_md)} ft md"
                        )
        return self


FlowPath = Literal["no_string", "annulus", "drillstring", "tubing"]
"""no_string: open hole / casing with nothing in it; annulus: between string and hole;
drillstring / tubing: inside the string. Combined annulus + string flow is deferred."""


# --- exit and temperature -------------------------------------------------------

class Exit(WellsimModel):
    ambient_pressure: Annotated[Pressure, Field(gt=0)] = Field(
        default=14.7, description="Pressure the flow discharges into; subsea becomes mudline hydrostatic later"
    )
    exit_id: Optional[Annotated[Diameter, Field(gt=0)]] = Field(
        default=None, description="Exit restriction ID; None means the flow-path ID at surface"
    )
    cd: Annotated[float, Field(ge=0.3, le=1.0)] = Field(
        default=1.0,
        description="Exit discharge coefficient: the effective exit area is cd x the exit_id area in the "
        "choked and subsonic exit solves. With cd = 1.0 and a restricted exit the adiabatic march applies "
        "the Crane sudden-contraction loss K_c = 0.5 (1 - beta^2) on entry to the restriction, the same "
        "treatment as an internal area change; with cd < 1 that loss is skipped because the discharge "
        "coefficient already carries the entry loss. Sharp orifice typically 0.6 to 0.65, rounded 0.95 to "
        "0.98 (Crane TP-410, MEMORY)",
    )
    k_override: Optional[Annotated[float, Field(gt=1.0, lt=2.0)]] = Field(
        default=None,
        description="Fixed heat capacity ratio for the sonic velocity; None uses the ideal-gas cp0(T)/cv0(T) "
        "of the gas at local temperature. 1.28 reproduces pre-M2.5-2 results",
    )


class TemperatureModel(WellsimModel):
    model: Literal["adiabatic", "profile", "linear"] = Field(
        default="adiabatic",
        description="'adiabatic': energy equation marched with the pressure (M2.5-3b, default for blowouts); "
        "'profile': prescribed linear flowing profile between reservoir and exit temperature "
        "(pre-3b behaviour; 'linear' is accepted as an alias)",
    )
    exit_temperature: Optional[Temperature] = Field(
        default=None,
        description="'profile' only: flowing temperature at exit; None means reservoir temperature minus default_exit_loss",
    )
    default_exit_loss: Annotated[float, Field(ge=0, description="degF, 'profile' only")] = 10.0
    sandface_jt: bool = Field(
        default=True,
        description="'adiabatic' only: gas enters the wellbore after an isenthalpic throttle from reservoir "
        "pressure to pwf (Joule-Thomson cooling at the sandface); False enters at reservoir temperature. "
        "Steady-state assumption: the near-wellbore rock reaches thermal equilibrium with the flowing gas "
        "in minutes to days at blowout rates, so early-time exit temperatures are warmer than reported",
    )

    @model_validator(mode="after")
    def _alias(self) -> "TemperatureModel":
        if self.model == "linear":
            object.__setattr__(self, "model", "profile")
        return self


# --- kill -----------------------------------------------------------------------

class NewtonianRheology(WellsimModel):
    type: Literal["newtonian"] = "newtonian"
    viscosity: Annotated[Viscosity, Field(gt=0)] = 1.0


Rheology = NewtonianRheology  # power_law and bingham join this after the water cases validate


class InjectionSection(WellsimModel):
    """One leg of the injection path. Annular if od_inner is set."""
    length: Annotated[Length, Field(gt=0)]
    id: Annotated[Diameter, Field(gt=0)]
    od_inner: Optional[Annotated[Diameter, Field(gt=0)]] = None
    roughness: Annotated[Diameter, Field(ge=0)] = STEEL_ROUGHNESS_IN

    @model_validator(mode="after")
    def _check(self) -> "InjectionSection":
        if self.od_inner is not None and self.od_inner >= self.id:
            raise ValueError("od_inner must be smaller than id")
        return self


class InjectionPath(WellsimModel):
    type: Literal["relief_well", "drillstring"]
    sections: list[InjectionSection] = Field(
        default_factory=list,
        description="Explicit path geometry; for type=drillstring an empty list means use well.string",
    )

    @model_validator(mode="after")
    def _check(self) -> "InjectionPath":
        if self.type == "relief_well" and not self.sections:
            raise ValueError("relief_well injection path needs at least one section")
        return self


class Kill(WellsimModel):
    injection_depth_md: Annotated[Length, Field(gt=0)]
    kill_fluid_density: Annotated[MudWeight, Field(
        ge=MIN_KILL_FLUID_PPG, le=MAX_KILL_FLUID_PPG,
        description="Bare number is ppg. Bounds catch lbm/ft3 or SG typed without a unit",
    )]
    rheology: Rheology = Field(default_factory=NewtonianRheology)
    injection_path: InjectionPath
    pump_pressure_limit: Annotated[DPressure, Field(gt=0)]
    available_hhp: Annotated[Power, Field(gt=0)]


# --- observations (inverse mode) --------------------------------------------------

class ObservedRate(WellsimModel):
    """An observed blowout rate, for the kh inverse (`core/inverse.py`). q_g is the gas
    rate the forward nodal solve must reproduce; q_g_low / q_g_high are optional bounds
    on the observation (both or neither) and are inverted alongside it."""
    q_g: Annotated[GasRate, Field(gt=0, description="Observed gas rate, Mscf/d")]
    q_l: Optional[Annotated[LiquidRate, Field(ge=0)]] = Field(
        default=None, description="Observed liquid rate, stb/d; with liquid_inflow mode 'wgr' it sets the WGR"
    )
    q_g_low: Optional[Annotated[GasRate, Field(gt=0)]] = Field(default=None, description="Lower bound on q_g, Mscf/d")
    q_g_high: Optional[Annotated[GasRate, Field(gt=0)]] = Field(default=None, description="Upper bound on q_g, Mscf/d")

    @model_validator(mode="after")
    def _check(self) -> "ObservedRate":
        if (self.q_g_low is None) != (self.q_g_high is None):
            raise ValueError("observed.rate needs both q_g_low and q_g_high, or neither")
        if self.q_g_low is not None and not (self.q_g_low < self.q_g < self.q_g_high):
            raise ValueError("observed.rate needs q_g_low < q_g < q_g_high")
        return self


class ObservedSIWHP(WellsimModel):
    """A shut-in wellhead pressure, for the p_r inverse through a static real-gas column."""
    pressure: Annotated[Pressure, Field(gt=0, description="Shut-in wellhead pressure, psia")]
    surface_temperature: Annotated[Temperature, Field(
        gt=-100, description="Wellhead temperature of the shut-in column, degF; the geothermal profile is linear "
        "in TVD from here to the reservoir temperature at the datum"
    )]
    time_note: str = Field(default="", description="Shut-in time or stabilisation note; empty is warned")


class Observed(WellsimModel):
    """Exactly one observation: a rate (kh inverse) or a shut-in wellhead pressure (p_r inverse)."""
    rate: Optional[ObservedRate] = None
    siwhp: Optional[ObservedSIWHP] = None

    @model_validator(mode="after")
    def _check(self) -> "Observed":
        if (self.rate is None) == (self.siwhp is None):
            raise ValueError("observed needs exactly one of rate or siwhp")
        return self


# --- relief well (rw-1) -----------------------------------------------------------

class SurfaceLine(WellsimModel):
    """One run of surface iron between the pumps and the wellhead. Horizontal: length is
    measured and contributes friction only, never hydrostatic."""
    length: Annotated[Length, Field(gt=0)]
    id: Annotated[Diameter, Field(gt=0)]
    roughness: Annotated[Diameter, Field(ge=0)] = STEEL_ROUGHNESS_IN


class ReliefStringSection(WellsimModel):
    """One section of the relief-well injection string, given by its BOTTOM station.

    ``md`` and ``tvd`` are the same pair as :class:`SurveyPoint`, which is the project's
    only trajectory schema; the string starts at md = tvd = 0 and each section runs from
    the previous section's bottom to this one. The vertical drop comes from tvd, so a
    deviated relief well gets its hydrostatic right; the friction uses md.
    """
    md: Annotated[Length, Field(gt=0, description="Bottom MD of this section, ft")]
    tvd: Annotated[Length, Field(gt=0, description="Bottom TVD of this section, ft")]
    id: Annotated[Diameter, Field(gt=0, description="Inside diameter of the injection string, in")]
    od: Optional[Annotated[Diameter, Field(gt=0)]] = Field(
        default=None,
        description="rw-3: outside diameter of the injection string, in. None means the relief_well "
        "string_od, and if that is absent too the OD is taken equal to the ID with a warning, which "
        "models no pipe-wall resistance at all",
    )
    cemented: bool = Field(
        default=True,
        description="rw-3: whether this interval is cemented. False is open hole, where the annular "
        "resistance uses the formation conductivity k_e over r_w/r_to instead of k_cem, and the result "
        "says which intervals were treated that way",
    )
    roughness: Annotated[Diameter, Field(ge=0)] = STEEL_ROUGHNESS_IN

    @model_validator(mode="after")
    def _check_od(self) -> "ReliefStringSection":
        if self.od is not None and self.od <= self.id:
            raise ValueError(f"relief string od {self.od} in must exceed id {self.id} in")
        return self


class ReliefRestriction(WellsimModel):
    """The loss at the far end of the injection string. Exactly one of:

    ``bit_tfa``: total flow area of the bit nozzles, in^2, giving the standard bit-nozzle
    pressure drop dp = rho q^2 / (12031 TFA^2) with rho in ppg and q in gpm.

    ``intersection_k``: a dimensionless loss coefficient applied as K rho v^2 / (2 g_c) at
    the string ID, for an open-ended string discharging into the blowing wellbore.

    There is no default: a relief-well job either has nozzles or it does not, and guessing
    silently would put a few hundred psi of pump pressure in or out of the answer.
    """
    bit_tfa: Optional[Annotated[float, Field(gt=0, description="Bit total flow area, in^2")]] = None
    intersection_k: Optional[Annotated[float, Field(
        ge=0, description="Intersection loss coefficient K, dimensionless, at the string ID"
    )]] = None

    @model_validator(mode="after")
    def _check(self) -> "ReliefRestriction":
        if (self.bit_tfa is None) == (self.intersection_k is None):
            raise ValueError(
                "relief_well.restriction needs exactly one of bit_tfa (in^2) or intersection_k; "
                "there is no default loss"
            )
        return self

    @property
    def kind(self) -> str:
        return "bit_tfa" if self.bit_tfa is not None else "intersection_k"


class ReliefRatings(WellsimModel):
    """Pressure ratings the pump pressure is checked against, gauge psi. Optional: with
    none given nothing is flagged and the result says so."""
    surface_iron: Optional[Annotated[DPressure, Field(gt=0, description="Surface iron working pressure, psi")]] = None
    string_burst: Optional[Annotated[DPressure, Field(gt=0, description="Injection string burst, psi")]] = None


class Pumps(WellsimModel):
    """What is on location. Optional, like the ratings."""
    max_pressure: Optional[Annotated[DPressure, Field(gt=0, description="Maximum pump pressure, psi gauge")]] = None
    total_hhp: Optional[Annotated[Power, Field(gt=0, description="Total hydraulic horsepower available")]] = None


class ReliefWell(WellsimModel):
    """rw-1: one relief well injecting kill fluid into the blowing well at z_inj.

    The pump balance at a rate q is

        p_pump = p_zinj - hydrostatic(string, rho_k) + friction(string)
                 + friction(surface lines) + restriction loss

    with p_zinj the blowing well's own pressure at z_inj at the kill condition, taken from
    the kill search rather than recomputed. Two or more relief wells are deferred
    (docs/TODO-M4.md): the kill search injects one mass rate at one z_inj.
    """
    surface_lines: list[SurfaceLine] = Field(
        default_factory=list, description="Surface iron runs; empty means the pumps are at the wellhead"
    )
    string: list[ReliefStringSection] = Field(
        default_factory=list,
        description="Injection string sections, bottom station each, tiling from surface to the intersection. "
        "An empty list is the degenerate zero-length string: no friction and no hydrostatic, so the balance "
        "reduces to p_zinj plus the restriction loss",
    )
    restriction: ReliefRestriction
    surface_temperature: Optional[Temperature] = Field(
        default=None,
        description="Kill-fluid temperature at the pumps, degF. None defaults to 60 degF with a warning on "
        "the result; it sets the cold end of the injection-temperature bounds",
    )
    hole_radius: Annotated[Length, Field(
        gt=0, description="rw-2: RELIEF hole radius, ft (not the blowing well). Sets t_D = alpha t / r_w^2"
    )]
    pumping_time_min: Annotated[float, Field(
        gt=0,
        description="rw-2: time the relief well has been pumping when the arrival temperature is wanted, "
        "MINUTES (named in the unit because the schema Time kind is days and a 1,440x slip here is silent). "
        "It enters only through t_D",
    )]
    k_e: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="rw-2: formation thermal conductivity, Btu/(hr ft degF). None means 1.4 with a warning; "
        "the Ramey relaxation length goes as 1/k_e",
    )
    string_od: Optional[Annotated[Diameter, Field(gt=0)]] = Field(
        default=None,
        description="rw-3: injection-string outside diameter, in, used for every section that does not give "
        "its own od. None with no section od takes the OD equal to the ID and warns: no pipe-wall resistance",
    )
    k_pipe: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="rw-3: pipe thermal conductivity, Btu/(hr ft degF). None means 25 (steel) with a warning",
    )
    annulus_fill: Literal["mud", "cement"] = Field(
        default="mud",
        description="rw-3b: what stands in the relief-well annulus over the cemented interval. 'mud' "
        "defaults k_annulus to 0.7 Btu/(hr ft degF), 'cement' to 1.0; both warn when defaulted. Mud is "
        "the default because on a dynamic kill the relief annulus holds STATIC drilling fluid: the kill "
        "fluid goes down the string and out at the intersection, so nothing returns up the annulus and "
        "there is no circulation to carry heat",
    )
    k_annulus: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="rw-3b: thermal conductivity of whatever fills the annulus, Btu/(hr ft degF). None "
        "takes the annulus_fill default with a warning. Uncemented sections use k_e over the same annulus "
        "instead, whatever this says",
    )
    k_cem: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="DEPRECATED alias for k_annulus (rw-3 called it k_cem). Accepted so existing cases "
        "load, with a warning on the result; give k_annulus instead",
    )
    bound_run: bool = Field(
        default=True,
        description="rw-3b: with temperature_runs and finite_u on, also run the kill search at the "
        "U -> infinity arrival temperature so the estimate is reported between both bounds. False runs "
        "the surface and finite-U searches only, two instead of three; the Monte Carlo sets it False",
    )
    finite_u: bool = Field(
        default=True,
        description="rw-3: compute the arrival temperature with a finite overall heat transfer coefficient "
        "(an estimate) as well as the U -> infinity bound. The reported kill rate is the finite-U estimate, "
        "with the surface and infinite-U rates bracketing it. False keeps the rw-2 bound-only behaviour",
    )
    alpha: Optional[Annotated[float, Field(gt=0)]] = Field(
        default=None,
        description="rw-2: formation thermal diffusivity, ft2/hr. None means 0.04 with a warning; it enters "
        "only through t_D",
    )
    temperature_runs: bool = Field(
        default=True,
        description="rw-2: run the kill search at the surface temperature (the lower bound on arrival) and "
        "at the Ramey arrival temperature (the upper bound), and report both rates with the gap. Two kill "
        "searches; the Monte Carlo forces this off and reports the arrival temperature from its single search",
    )
    geothermal_bound: bool = Field(
        default=False,
        description="rw-2: also run the kill search with the kill fluid fully equilibrated with the formation "
        "at z_inj. This was the rw-1 upper bound; the Ramey arrival temperature replaced it and is far tighter, "
        "so it is off by default and kept only for comparison. A third kill search",
    )
    ratings: ReliefRatings = Field(default_factory=ReliefRatings)
    pumps: Pumps = Field(default_factory=Pumps)
    schedule_steps: Annotated[int, Field(ge=2, le=100)] = Field(
        default=10, description="Rows in the pump schedule from q_k_min to 1.5 q_k_min"
    )

    @model_validator(mode="after")
    def _check(self) -> "ReliefWell":
        prev_md = prev_tvd = 0.0
        for i, s in enumerate(self.string):
            if s.md <= prev_md:
                raise ValueError(f"relief_well.string section {i}: md must increase (got {s.md} after {prev_md})")
            if s.tvd < prev_tvd - _DEPTH_TOL_FT:
                raise ValueError(f"relief_well.string section {i}: tvd must not decrease")
            if (s.tvd - prev_tvd) > (s.md - prev_md) + _DEPTH_TOL_FT:
                raise ValueError(
                    f"relief_well.string section {i}: tvd increment {s.tvd - prev_tvd:g} ft exceeds the md "
                    f"increment {s.md - prev_md:g} ft"
                )
            prev_md, prev_tvd = s.md, s.tvd
        if self.k_annulus is not None and self.k_cem is not None:
            raise ValueError(
                "relief_well: k_annulus and the deprecated alias k_cem are both set; give k_annulus only"
            )
        if self.restriction.intersection_k is not None and not self.string:
            raise ValueError(
                "relief_well.restriction.intersection_k needs a string: the loss K rho v^2 / (2 g_c) is "
                "evaluated at the string ID, and a zero-length string has none. Use bit_tfa instead"
            )
        return self

    @property
    def bottom_tvd(self) -> float:
        return self.string[-1].tvd if self.string else 0.0

    @property
    def bottom_md(self) -> float:
        return self.string[-1].md if self.string else 0.0


# --- Monte Carlo (mc-1) -----------------------------------------------------------

class _Triangular(WellsimModel):
    """min <= mode <= max. A zero-width triangular (all three equal) is the degenerate
    case and is allowed: it reproduces the deterministic input exactly."""

    @model_validator(mode="after")
    def _check(self) -> "_Triangular":
        if not (self.min <= self.mode <= self.max):
            raise ValueError(f"triangular needs min <= mode <= max, got {self.min}, {self.mode}, {self.max}")
        return self

    @property
    def degenerate(self) -> bool:
        return self.min == self.max


class TriangularPressure(_Triangular):
    min: Annotated[Pressure, Field(gt=0)]
    mode: Annotated[Pressure, Field(gt=0)]
    max: Annotated[Pressure, Field(gt=0)]


class TriangularLength(_Triangular):
    min: Annotated[Length, Field(gt=0)]
    mode: Annotated[Length, Field(gt=0)]
    max: Annotated[Length, Field(gt=0)]


class TriangularDiameter(_Triangular):
    min: Annotated[Diameter, Field(gt=0)]
    mode: Annotated[Diameter, Field(gt=0)]
    max: Annotated[Diameter, Field(gt=0)]


class TriangularCd(_Triangular):
    min: Annotated[float, Field(ge=0.3, le=1.0)]
    mode: Annotated[float, Field(ge=0.3, le=1.0)]
    max: Annotated[float, Field(ge=0.3, le=1.0)]


class LogUniform(WellsimModel):
    """Log-uniform over [low, high]. low == high is the degenerate case."""
    low: Annotated[float, Field(gt=0)]
    high: Annotated[float, Field(gt=0)]

    @model_validator(mode="after")
    def _check(self) -> "LogUniform":
        if self.low > self.high:
            raise ValueError(f"log-uniform needs low <= high, got {self.low}, {self.high}")
        return self

    @property
    def degenerate(self) -> bool:
        return self.low == self.high


class ExitScenario(WellsimModel):
    """One discrete exit condition, sampled against the other scenarios by weight.
    An unset field keeps the case value; cd and exit_id are the only exit fields the
    Monte Carlo varies."""
    name: str
    weight: Annotated[float, Field(gt=0)] = 1.0
    cd: Optional[Annotated[float, Field(ge=0.3, le=1.0)]] = None
    exit_id: Optional[Annotated[Diameter, Field(gt=0)]] = None

    @model_validator(mode="after")
    def _check(self) -> "ExitScenario":
        if self.cd is None and self.exit_id is None:
            raise ValueError(f"exit scenario '{self.name}' sets neither cd nor exit_id")
        return self


class RateBand(WellsimModel):
    """An observed blowout-rate band, Mscf/d. A FILTER on forward samples, never an input
    to invert: samples whose forward rate falls outside the band are dropped from the
    reported statistics and counted. A rate read off a flare or a plume is not a
    defensible input, so it never enters the inputs; it only says which combinations of
    the defensible inputs are consistent with what was seen."""
    q_low: Annotated[GasRate, Field(gt=0)]
    q_high: Annotated[GasRate, Field(gt=0)]

    @model_validator(mode="after")
    def _check(self) -> "RateBand":
        if self.q_low >= self.q_high:
            raise ValueError("rate_band needs q_low < q_high")
        return self


class MonteCarloKill(WellsimModel):
    """The planned kill fluid, and how many survivors the kill search runs on. Replaces
    any dynamic_kill block on the case for the Monte Carlo kill pass."""
    rho_k: Annotated[MudWeight, Field(
        ge=MIN_KILL_FLUID_PPG, le=MAX_KILL_FLUID_PPG, description="Planned kill-fluid density, ppg"
    )]
    mu_k: Annotated[Viscosity, Field(gt=0, description="Kill-fluid viscosity, cp (Newtonian)")] = 1.0
    cp_k: Annotated[float, Field(gt=0)] = 1.0
    z_inj: Optional[Annotated[Length, Field(gt=0)]] = Field(
        default=None, description="Injection depth, ft TVD; None means the sandface"
    )
    n_kill: Annotated[int, Field(ge=1)] = Field(
        default=100, description="Survivors the kill search runs on, stratified into equal-probability bins by q_g"
    )
    chains: Annotated[int, Field(ge=1)] = Field(
        default=8,
        description="Warm-start chains the kill samples are split into. Each chain is a q_g-ordered sequence "
        "whose searches warm-start from the previous q_k_min; chains run in parallel. Fixed here rather than "
        "taken from `workers` so the result does not depend on the worker count. Changing it changes the "
        "warm-start path, so results move within the 1% search tolerance",
    )


class MonteCarlo(WellsimModel):
    """mc-1: forward Monte Carlo on blowout rate, static-hold mud weight and kill rate.

    Sampled inputs: p_r, net pay, open interval h_p and the exit condition (cd or the
    restriction diameter) as triangulars, kh log-uniform. h_p may be the string 'net_pay',
    which ties the open interval to the sampled net pay (full penetration). Permeability
    is derived, k = kh / net_pay, exactly as core/inverse.with_kh does, so beta and the
    non-Darcy D follow from k as in the forward solve. h_1 (exposed_top_offset) is not
    sampled; it is held at the case value and enters the h_1 + h_p <= net_pay check.
    """
    seed: int = Field(description="Required: the sampler is deterministic given the seed")
    n_samples: Annotated[int, Field(ge=2)] = 2000
    workers: Annotated[int, Field(ge=1)] = 1
    p_res: Optional[TriangularPressure] = None
    net_pay: Optional[TriangularLength] = None
    exposed_interval: Optional[Union[TriangularLength, Literal["net_pay"]]] = Field(
        default=None,
        description="h_p: a triangular in ft, or 'net_pay' to tie it to the sampled net pay (full penetration). "
        "None keeps the case value",
    )
    cd: Optional[TriangularCd] = None
    exit_id: Optional[TriangularDiameter] = None
    kh: Optional[LogUniform] = Field(default=None, description="md-ft, log-uniform over [low, high]")
    exit_scenarios: Optional[list[ExitScenario]] = Field(
        default=None,
        description="Discrete exit conditions with weights, instead of the continuous cd / exit_id triangulars",
    )
    rate_band: Optional[RateBand] = None
    kill: Optional[MonteCarloKill] = None

    @model_validator(mode="after")
    def _check(self) -> "MonteCarlo":
        if self.exit_scenarios is not None:
            if not self.exit_scenarios:
                raise ValueError("exit_scenarios is empty; use null for no scenarios")
            if self.cd is not None or self.exit_id is not None:
                raise ValueError(
                    "exit_scenarios replaces the continuous exit range; give scenarios or cd / exit_id, not both"
                )
            names = [s.name for s in self.exit_scenarios]
            if len(set(names)) != len(names):
                raise ValueError("exit scenario names must be unique")
        if (self.p_res is None and self.net_pay is None and self.exposed_interval is None
                and self.cd is None and self.exit_id is None and self.kh is None
                and self.exit_scenarios is None):
            raise ValueError("the mc block samples nothing; give at least one distribution")
        return self


# --- options --------------------------------------------------------------------

VlpCorrelation = Literal[
    "single_phase_gas", "hagedorn_brown", "beggs_brill", "gray", "duns_ros", "teles_waltrich"
]


class Correlations(WellsimModel):
    rs_pb: Literal["standing", "vasquez_beggs"] = "standing"
    bo: Literal["standing", "vasquez_beggs"] = "standing"
    oil_viscosity: Literal["beggs_robinson"] = "beggs_robinson"
    gas_z: Literal["dak", "hall_yarborough"] = "dak"
    pseudocritical: Literal["sutton"] = "sutton"
    gas_viscosity: Literal["lee_gonzalez_eakin"] = "lee_gonzalez_eakin"
    water: Literal["mccain"] = "mccain"
    surface_tension: Literal["baker_swerdloff"] = "baker_swerdloff"
    non_darcy_beta: Literal["auto", "jones", "geertsma", "firoozabadi_katz"] = Field(
        default="auto",
        description="Turbulence factor correlation for the gas IPR. 'auto' (default, M2.5-5): Jones when only "
        "permeability is given, Geertsma when reservoir.porosity is also supplied; the result notes which "
        "was used. geertsma needs porosity; firoozabadi_katz is an unattributed placeholder",
    )
    vlp: Optional[list[VlpCorrelation]] = Field(
        default=None, description="Correlations to run; None means the default for the fluid type"
    )


class Numerics(WellsimModel):
    vlp_max_segment: Annotated[Length, Field(gt=0)] = 100.0
    vlp_pressure_tol: Annotated[DPressure, Field(gt=0)] = 0.1
    vlp_temperature_tol: Annotated[float, Field(
        gt=0, description="degF per step, adiabatic march; 0.005 keeps the total-enthalpy drift below 0.01 K"
    )] = 0.005
    rate_tol: Annotated[float, Field(gt=0, description="relative")] = 1e-3
    pressure_tol: Annotated[DPressure, Field(
        gt=0, description="Nodal convergence also requires |pwf_ipr - pwf_vlp| within this at the operating point"
    )] = 0.5
    nodal_scan_points: Annotated[int, Field(ge=4, le=200)] = Field(
        default=16, description="Grid points scanned for intersections before Brent refinement"
    )
    nodal_space: Literal["pwf", "rate"] = Field(
        default="pwf",
        description="'pwf': root of pwf_vlp(q_ipr(pwf)) - pwf between ambient and reservoir pressure "
        "(default; single-valued for any IPR). 'rate': the pre-closeout scan and Brent in rate, kept "
        "for one release",
    )


class Options(WellsimModel):
    correlations: Correlations = Field(default_factory=Correlations)
    numerics: Numerics = Field(default_factory=Numerics)


# --- case -----------------------------------------------------------------------

class Case(WellsimModel):
    name: str
    description: str = ""
    source: str = Field(default="", description="Citation, for validation cases")
    notes: str = Field(default="", description="Anything assumed because the source did not state it")
    reservoir: Reservoir
    fluid: Fluid
    liquid: Optional[Liquid] = Field(
        default=None,
        description="M3: explicit liquid-phase properties for the multiphase closure; None means "
        "single-phase gas (dry gas) or, from M4, values from the black-oil PVT",
    )
    liquid_inflow: Optional[LiquidInflow] = Field(
        default=None,
        description="M3-3: liquid inflow (wgr or pi). With liquid properties present this switches the solve "
        "to the gas-liquid nodal; absent, the dry-gas solve runs unchanged",
    )
    well: Well
    flow_path: FlowPath
    exit: Exit = Field(default_factory=Exit)
    temperature: TemperatureModel = Field(default_factory=TemperatureModel)
    kill: Optional[Kill] = None
    dynamic_kill: Optional[DynamicKill] = Field(
        default=None, description="M3-4: dynamic kill of the blowing well with a Newtonian kill fluid"
    )
    observed: Optional[Observed] = Field(
        default=None,
        description="Inverse mode: an observed rate (kh from the forward nodal solve) or a shut-in wellhead "
        "pressure (p_r from a static gas column). The forward solve ignores it; core/inverse.py returns a "
        "case copy with the inverted value set",
    )
    relief_well: Optional[ReliefWell] = Field(
        default=None,
        description="rw-1: relief-well hydraulics for the dynamic kill (surface lines, injection string, "
        "restriction, ratings, pumps). Needs dynamic_kill; core/relief_well.py runs it",
    )
    mc: Optional[MonteCarlo] = Field(
        default=None,
        description="mc-1: forward Monte Carlo on blowout rate, static-hold mud weight and kill rate. The "
        "forward solve ignores it; core/montecarlo.py runs it",
    )
    options: Options = Field(default_factory=Options)

    def md_to_tvd_safe(self, md: float) -> float:
        return self.well.md_to_tvd(md)

    @model_validator(mode="after")
    def _check(self) -> "Case":
        td = self.well.total_depth_md
        if self.reservoir.depth_md > td + _DEPTH_TOL_FT:
            raise ValueError("reservoir depth_md cannot be below well total_depth_md")
        if self.reservoir.datum_tvd is None:
            self.reservoir.datum_tvd = self.well.md_to_tvd(self.reservoir.depth_md)
        elif self.reservoir.datum_tvd > self.well.md_to_tvd(td) + _DEPTH_TOL_FT:
            raise ValueError("reservoir datum_tvd cannot be below the TVD of total_depth_md")
        if self.dynamic_kill is not None and self.dynamic_kill.z_inj is not None:
            if self.dynamic_kill.z_inj > self.md_to_tvd_safe(self.reservoir.depth_md) + _DEPTH_TOL_FT:
                raise ValueError("dynamic_kill.z_inj (TVD) cannot be below the sandface")
        if self.liquid_inflow is not None and self.liquid is None:
            raise ValueError("liquid_inflow needs the liquid block (rho_l, mu_l, cp_l, sigma, c_l)")
        if self.relief_well is not None:
            if self.dynamic_kill is None:
                raise ValueError(
                    "relief_well needs the dynamic_kill block: the pump balance starts from the blowing "
                    "well's pressure at z_inj at the kill condition, which only the kill search knows"
                )
            if self.relief_well.string:
                tvd_inj = (self.dynamic_kill.z_inj if self.dynamic_kill.z_inj is not None
                           else self.md_to_tvd_safe(self.reservoir.depth_md))
                bottom = self.relief_well.bottom_tvd
                if abs(bottom - tvd_inj) > 1.0:
                    raise ValueError(
                        f"relief_well string reaches {bottom:g} ft TVD but the kill fluid is injected at "
                        f"{tvd_inj:g} ft TVD: the string must intersect the blowing well at z_inj "
                        "(tolerance 1 ft)"
                    )
        if self.observed is not None and self.observed.rate is not None:
            # the rate inverse holds h, skin and p_r fixed: each must be an explicit input,
            # never the schema default (skin defaults to 0 for forward cases only)
            if "skin" not in self.reservoir.model_fields_set:
                raise ValueError("observed.rate: reservoir.skin must be given explicitly (the kh inverse holds it fixed)")
            if self.observed.rate.q_l is not None and self.liquid_inflow is None:
                raise ValueError("observed.rate.q_l needs liquid and liquid_inflow blocks (no liquid model to attribute it to)")
        has_string = self.well.string is not None
        if self.flow_path == "no_string" and has_string:
            raise ValueError(
                "flow_path 'no_string' but well.string is set; remove the string or choose annulus/drillstring/tubing"
            )
        if self.flow_path in ("annulus", "drillstring", "tubing") and not has_string:
            raise ValueError(f"flow_path '{self.flow_path}' needs well.string")
        if self.kill is not None:
            if self.kill.injection_depth_md > td + _DEPTH_TOL_FT:
                raise ValueError("kill injection_depth_md cannot be below well total_depth_md")
            if self.kill.injection_path.type == "drillstring" and not self.kill.injection_path.sections:
                if not has_string:
                    raise ValueError("drillstring injection path with no sections needs well.string")
                # equal to the string bottom is the normal case: fluid exits at the bit
                if self.kill.injection_depth_md > self.well.string_bottom_md + _DEPTH_TOL_FT:
                    raise ValueError("drillstring injection depth cannot be below the string bottom")
        return self
