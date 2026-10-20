"""Black-oil correlations, formation water, surface tension, and the black-oil PVT model.

Defaults per spec 4.1: Standing for Rs / Pb / Bo, Beggs-Robinson dead and live oil
viscosity with Vasquez-Beggs above the bubble point, Vasquez-Beggs oil
compressibility (needed for Bo and density above Pb), McCain water, Baker-Swerdloff
surface tension. Vasquez-Beggs Rs / Pb / Bo are the schema alternates.

Stage 1 simplifications, stated so they are not silent:
- The producing GOR is the solution GOR at the bubble point (Rsb). When the case
  gives a bubble point, Rsb is the correlation's Rs at that pressure capped at GOR.
- Free gas has the same gravity as solution gas (one gas_sg per case).
- Vasquez-Beggs gas gravity is used as given, not normalised to 100 psig separator.
- Water compressibility uses McCain's undersaturated form at every pressure.

Sources
-------
Standing, M.B. (1947) "A Pressure-Volume-Temperature Correlation for Mixtures of
    California Oils and Gases", API Drilling and Production Practice, 275-287;
    equation forms from Standing (1977/1981) "Volumetric and Phase Behavior of Oil
    Field Hydrocarbon Systems", SPE. Data: Pb 130-7000 psia, T 100-258 degF,
    Rs 20-1425 scf/stb, API 16.5-63.8, gamma_g 0.59-0.95, Bo 1.024-2.15.
Vasquez, M. and Beggs, H.D. (1980) "Correlations for Fluid Physical Property
    Prediction", JPT 32(6) 968-970, SPE 6719. Data: p 141-9515 psia, Rs 9.3-2199,
    API 15.3-59.5, gamma_g 0.511-1.351, T 75-294 degF.
Beggs, H.D. and Robinson, J.R. (1975) "Estimating the Viscosity of Crude Oil
    Systems", JPT 27(9) 1140-1141, SPE 5434. Dead oil: API 16-58, T 70-295 degF.
    Live oil: Rs 20-2070 scf/stb, p 0-5250 psia.
McCain, W.D. Jr. (1990) "The Properties of Petroleum Fluids", 2nd ed., PennWell,
    ch. 16; and McCain (1991) "Reservoir-Fluid Property Correlations - State of the
    Art", SPERE 6(2) 266-272. Bw: T <= 260 degF, p <= 5000 psia, any salinity.
    cw: T 200-270 degF, p 1000-20000 psia, S <= 200000 mg/L. Viscosity at 1 atm:
    T 100-400 degF, S <= 26 wt%; pressure factor T 86.5-167 degF, p <= 14000 psia.
    Rsw: p 1000-10000 psia, T 100-340 degF.
Baker, O. and Swerdloff, W. (1956) "Finding Surface Tension of Hydrocarbon
    Liquids", Oil & Gas J. 54 (Jan 2) 125; equation forms as given in Beggs, H.D.
    (2003) "Production Optimization Using NODAL Analysis", 2nd ed., OGCI, ch. 2.
    Dead-oil chart covers 68-100 degF; live-oil correction to ~3500 psia.
"""

from __future__ import annotations

import math
import warnings

from wellsim.core.pvt.base import (
    P_SC,
    RHO_WATER_SC,
    PVTModel,
    PVTState,
    check_range,
)
from wellsim.core.pvt.dry_gas import (
    Z_METHODS,
    gas_density,
    gas_fvf,
    lee_gonzalez_eakin,
    pseudocriticals,
)

__all__ = [
    "oil_sg_from_api",
    "standing_rs",
    "standing_pb",
    "standing_bo",
    "vasquez_beggs_rs",
    "vasquez_beggs_pb",
    "vasquez_beggs_bo",
    "vasquez_beggs_co",
    "undersaturated_bo",
    "oil_density",
    "beggs_robinson_dead",
    "beggs_robinson_live",
    "vasquez_beggs_undersaturated_viscosity",
    "water_density_sc",
    "mccain_bw",
    "mccain_cw",
    "mccain_water_viscosity",
    "mccain_rsw",
    "baker_swerdloff_oil",
    "baker_swerdloff_water",
    "RS_PB_METHODS",
    "BO_METHODS",
    "BlackOilPVT",
]

def oil_sg_from_api(api: float) -> float:
    return 141.5 / (131.5 + api)


# --- Standing ------------------------------------------------------------------

def _standing_check(api: float, gas_sg: float, t_f: float) -> None:
    check_range("Standing", "API", api, 16.5, 63.8, "degAPI")
    check_range("Standing", "gas_sg", gas_sg, 0.59, 0.95)
    check_range("Standing", "T", t_f, 100.0, 258.0, "degF")


def standing_rs(p: float, t_f: float, api: float, gas_sg: float) -> float:
    """Standing solution GOR, scf/stb, valid at and below the bubble point.

    Rs = gamma_g [ (p/18.2 + 1.4) 10^(0.0125 API - 0.00091 T) ]^1.2048
    """
    _standing_check(api, gas_sg, t_f)
    check_range("Standing", "p", p, 130.0, 7000.0, "psia")
    x = 0.0125 * api - 0.00091 * t_f
    return gas_sg * ((p / 18.2 + 1.4) * 10.0**x) ** 1.2048


def standing_pb(rs: float, t_f: float, api: float, gas_sg: float) -> float:
    """Standing bubble-point pressure, psia.

    Pb = 18.2 [ (Rs/gamma_g)^0.83 10^(0.00091 T - 0.0125 API) - 1.4 ]
    """
    _standing_check(api, gas_sg, t_f)
    check_range("Standing", "Rs", rs, 20.0, 1425.0, "scf/stb")
    x = 0.00091 * t_f - 0.0125 * api
    pb = 18.2 * ((rs / gas_sg) ** 0.83 * 10.0**x - 1.4)
    return max(pb, 14.696)


def standing_bo(rs: float, t_f: float, api: float, gas_sg: float) -> float:
    """Standing saturated oil FVF, rb/stb.

    Bo = 0.9759 + 0.00012 [ Rs (gamma_g/gamma_o)^0.5 + 1.25 T ]^1.2
    """
    _standing_check(api, gas_sg, t_f)
    check_range("Standing", "Rs", rs, 20.0, 1425.0, "scf/stb")
    oil_sg = oil_sg_from_api(api)
    return 0.9759 + 0.00012 * (rs * math.sqrt(gas_sg / oil_sg) + 1.25 * t_f) ** 1.2


# --- Vasquez-Beggs -------------------------------------------------------------

def _vb_coeffs(api: float) -> tuple[float, float, float]:
    if api <= 30.0:
        return 0.0362, 1.0937, 25.7240
    return 0.0178, 1.1870, 23.9310


def _vb_check(api: float, gas_sg: float, t_f: float) -> None:
    check_range("Vasquez-Beggs", "API", api, 15.3, 59.5, "degAPI")
    check_range("Vasquez-Beggs", "gas_sg", gas_sg, 0.511, 1.351)
    check_range("Vasquez-Beggs", "T", t_f, 75.0, 294.0, "degF")


def vasquez_beggs_rs(p: float, t_f: float, api: float, gas_sg: float) -> float:
    """Vasquez-Beggs solution GOR, scf/stb: Rs = C1 gamma_g p^C2 exp(C3 API / T_R)."""
    _vb_check(api, gas_sg, t_f)
    check_range("Vasquez-Beggs", "p", p, 141.0, 9515.0, "psia")
    c1, c2, c3 = _vb_coeffs(api)
    return c1 * gas_sg * p**c2 * math.exp(c3 * api / (t_f + 459.67))


def vasquez_beggs_pb(rs: float, t_f: float, api: float, gas_sg: float) -> float:
    """Vasquez-Beggs bubble-point pressure, psia (inverse of the Rs equation)."""
    _vb_check(api, gas_sg, t_f)
    check_range("Vasquez-Beggs", "Rs", rs, 9.3, 2199.0, "scf/stb")
    c1, c2, c3 = _vb_coeffs(api)
    pb = (rs / (c1 * gas_sg * math.exp(c3 * api / (t_f + 459.67)))) ** (1.0 / c2)
    return max(pb, 14.696)


def vasquez_beggs_bo(rs: float, t_f: float, api: float, gas_sg: float) -> float:
    """Vasquez-Beggs saturated oil FVF, rb/stb."""
    _vb_check(api, gas_sg, t_f)
    check_range("Vasquez-Beggs", "Rs", rs, 9.3, 2199.0, "scf/stb")
    if api <= 30.0:
        c1, c2, c3 = 4.677e-4, 1.751e-5, -1.811e-8
    else:
        c1, c2, c3 = 4.670e-4, 1.100e-5, 1.337e-9
    f = (t_f - 60.0) * api / gas_sg
    return 1.0 + c1 * rs + c2 * f + c3 * rs * f


def vasquez_beggs_co(p: float, t_f: float, api: float, gas_sg: float, rs: float) -> float:
    """Vasquez-Beggs undersaturated oil compressibility, 1/psi.

    co = (-1433 + 5 Rs + 17.2 T - 1180 gamma_g + 12.61 API) / (1e5 p)
    """
    _vb_check(api, gas_sg, t_f)
    check_range("Vasquez-Beggs", "p", p, 141.0, 9515.0, "psia")
    co = (-1433.0 + 5.0 * rs + 17.2 * t_f - 1180.0 * gas_sg + 12.61 * api) / (1e5 * p)
    return max(co, 1e-7)


def undersaturated_bo(bob: float, co: float, p: float, pb: float) -> float:
    """Bo above the bubble point: Bob exp(-co (p - pb))."""
    return bob * math.exp(-co * (p - pb))


def oil_density(rs: float, bo: float, api: float, gas_sg: float) -> float:
    """Saturated oil density, lbm/ft3: (62.4 gamma_o + 0.0136 Rs gamma_g) / Bo."""
    return (62.4 * oil_sg_from_api(api) + 0.0136 * rs * gas_sg) / bo


# --- oil viscosity ---------------------------------------------------------------

def beggs_robinson_dead(t_f: float, api: float) -> float:
    """Beggs-Robinson dead-oil viscosity, cp: 10^x - 1, x = T^-1.163 10^(3.0324 - 0.02023 API)."""
    check_range("Beggs-Robinson dead oil", "API", api, 16.0, 58.0, "degAPI")
    check_range("Beggs-Robinson dead oil", "T", t_f, 70.0, 295.0, "degF")
    x = t_f ** (-1.163) * 10.0 ** (3.0324 - 0.02023 * api)
    return 10.0**x - 1.0


def beggs_robinson_live(mu_dead: float, rs: float, p: float | None = None) -> float:
    """Beggs-Robinson saturated-oil viscosity, cp: A mu_od^B.

    A = 10.715 (Rs + 100)^-0.515, B = 5.44 (Rs + 150)^-0.338. ``p`` only feeds the range check.
    """
    check_range("Beggs-Robinson live oil", "Rs", rs, 20.0, 2070.0, "scf/stb")
    if p is not None:
        check_range("Beggs-Robinson live oil", "p", p, 0.0, 5250.0, "psia")
    a = 10.715 * (rs + 100.0) ** (-0.515)
    b = 5.44 * (rs + 150.0) ** (-0.338)
    return a * mu_dead**b


def vasquez_beggs_undersaturated_viscosity(mu_ob: float, p: float, pb: float) -> float:
    """Vasquez-Beggs oil viscosity above the bubble point, cp.

    mu = mu_ob (p/pb)^m, m = 2.6 p^1.187 10^(-3.9e-5 p - 5).
    """
    check_range("Vasquez-Beggs viscosity", "p", p, 141.0, 9515.0, "psia")
    m = 2.6 * p**1.187 * 10.0 ** (-3.9e-5 * p - 5.0)
    return mu_ob * (p / pb) ** m


# --- formation water (McCain) ----------------------------------------------------

def _salinity_wt_pct(salinity_ppm: float) -> float:
    return salinity_ppm * 1e-4


def water_density_sc(salinity_ppm: float) -> float:
    """Brine density at standard conditions, lbm/ft3 (McCain 1990 eq. 16.1); S in wt%."""
    s = _salinity_wt_pct(salinity_ppm)
    check_range("McCain water density", "salinity", s, 0.0, 30.0, "wt%")
    return RHO_WATER_SC + 0.438603 * s + 1.60074e-3 * s**2


def mccain_bw(p: float, t_f: float) -> float:
    """McCain water FVF, rb/stb: (1 + dV_T)(1 + dV_p). Salinity effect is negligible (McCain)."""
    check_range("McCain Bw", "T", t_f, 32.0, 260.0, "degF")
    check_range("McCain Bw", "p", p, P_SC, 5000.0, "psia")
    dv_t = -1.0001e-2 + 1.33391e-4 * t_f + 5.50654e-7 * t_f**2
    dv_p = -1.95301e-9 * p * t_f - 1.72834e-13 * p**2 * t_f - 3.58922e-7 * p - 2.25341e-10 * p**2
    return (1.0 + dv_t) * (1.0 + dv_p)


def mccain_cw(p: float, t_f: float, salinity_ppm: float) -> float:
    """McCain water compressibility above the bubble point, 1/psi.

    cw = 1 / (7.033 p + 0.5415 S_mg/L - 537 T + 403300). Salinity in mg/L is taken
    equal to ppm by weight (the difference is within the correlation's accuracy).
    """
    check_range("McCain cw", "T", t_f, 200.0, 270.0, "degF")
    check_range("McCain cw", "p", p, 1000.0, 20000.0, "psia")
    check_range("McCain cw", "salinity", salinity_ppm, 0.0, 200000.0, "mg/L")
    return 1.0 / (7.033 * p + 0.5415 * salinity_ppm - 537.0 * t_f + 403300.0)


def mccain_water_viscosity(p: float, t_f: float, salinity_ppm: float) -> float:
    """McCain brine viscosity, cp. mu_1atm = A T^B, then a pressure factor."""
    s = _salinity_wt_pct(salinity_ppm)
    check_range("McCain water viscosity", "T", t_f, 100.0, 400.0, "degF")
    check_range("McCain water viscosity", "salinity", s, 0.0, 26.0, "wt%")
    check_range("McCain water viscosity", "p", p, P_SC, 14000.0, "psia")
    a = 109.574 - 8.40564 * s + 0.313314 * s**2 + 8.72213e-3 * s**3
    b = -1.12166 + 2.63951e-2 * s - 6.79461e-4 * s**2 - 5.47119e-5 * s**3 + 1.55586e-6 * s**4
    mu1 = a * t_f**b
    return mu1 * (0.9994 + 4.0295e-5 * p + 3.1062e-9 * p**2)


def mccain_rsw(p: float, t_f: float, salinity_ppm: float) -> float:
    """McCain gas solubility in brine, scf/stb. Pure-water polynomial times a salinity factor."""
    s = _salinity_wt_pct(salinity_ppm)
    check_range("McCain Rsw", "T", t_f, 100.0, 340.0, "degF")
    check_range("McCain Rsw", "p", p, 1000.0, 10000.0, "psia")
    check_range("McCain Rsw", "salinity", s, 0.0, 30.0, "wt%")
    a = 8.15839 - 6.12265e-2 * t_f + 1.91663e-4 * t_f**2 - 2.1654e-7 * t_f**3
    b = 1.01021e-2 - 7.44241e-5 * t_f + 3.05553e-7 * t_f**2 - 2.94883e-10 * t_f**3
    c = (-9.02505 + 0.130237 * t_f - 8.53425e-4 * t_f**2 + 2.34122e-6 * t_f**3 - 2.37049e-9 * t_f**4) * 1e-7
    rsw_pure = a + b * p + c * p**2
    factor = 10.0 ** (-0.0840655 * s * t_f ** (-0.285854))
    return max(rsw_pure * factor, 0.0)


# --- surface tension (Baker-Swerdloff) -----------------------------------------

def baker_swerdloff_oil(p: float, t_f: float, api: float) -> float:
    """Gas-oil surface tension, dyne/cm.

    Dead oil: 39 - 0.2571 API at 68 degF, 37.5 - 0.2571 API at 100 degF, linear between,
    held constant outside. Live oil: times (1 - 0.024 p^0.45), floored at 1. Holding the
    dead-oil value outside 68-100 degF is the published procedure (Beggs 2003), so it is
    not flagged; pressure beyond the chart is.
    """
    check_range("Baker-Swerdloff oil", "p", p, P_SC, 3500.0, "psia")
    s68 = 39.0 - 0.2571 * api
    s100 = 37.5 - 0.2571 * api
    if t_f <= 68.0:
        dead = s68
    elif t_f >= 100.0:
        dead = s100
    else:
        dead = s68 + (t_f - 68.0) * (s100 - s68) / 32.0
    live = dead * (1.0 - 0.024 * p**0.45)
    return max(live, 1.0)


def baker_swerdloff_water(p: float, t_f: float) -> float:
    """Gas-water surface tension, dyne/cm.

    75 - 1.108 p^0.349 at 74 degF, 53 - 0.1048 p^0.637 at 280 degF, linear between,
    held constant outside (published procedure, not flagged).
    """
    s74 = 75.0 - 1.108 * p**0.349
    s280 = 53.0 - 0.1048 * p**0.637
    if t_f <= 74.0:
        return max(s74, 1.0)
    if t_f >= 280.0:
        return max(s280, 1.0)
    return max(s74 + (t_f - 74.0) * (s280 - s74) / 206.0, 1.0)


RS_PB_METHODS = {
    "standing": (standing_rs, standing_pb),
    "vasquez_beggs": (vasquez_beggs_rs, vasquez_beggs_pb),
}
BO_METHODS = {"standing": standing_bo, "vasquez_beggs": vasquez_beggs_bo}


# --- model ----------------------------------------------------------------------

class BlackOilPVT(PVTModel):
    """Black oil with associated gas and (optionally) water.

    ``gor`` is the producing GOR in scf/stb and doubles as Rsb unless ``bubble_point``
    is given. Water enters through ``water_cut`` and ``water_salinity`` (ppm TDS).
    """

    fluid_type = "black_oil"

    def __init__(
        self,
        oil_api: float,
        gas_sg: float,
        gor: float,
        water_cut: float = 0.0,
        water_salinity: float = 0.0,
        bubble_point: float | None = None,
        *,
        rs_pb_method: str = "standing",
        bo_method: str = "standing",
        z_method: str = "dak",
    ):
        for name, table, chosen in (
            ("rs_pb", RS_PB_METHODS, rs_pb_method),
            ("bo", BO_METHODS, bo_method),
            ("gas_z", Z_METHODS, z_method),
        ):
            if chosen not in table:
                raise ValueError(f"unknown {name} correlation {chosen!r}; choose from {list(table)}")
        self.oil_api = float(oil_api)
        self.oil_sg = oil_sg_from_api(self.oil_api)
        self.gas_sg = float(gas_sg)
        self.gor = float(gor)
        self.water_cut = float(water_cut)
        self.water_salinity = float(water_salinity)
        self.bubble_point_given = None if bubble_point is None else float(bubble_point)
        self._rsb_warned = False
        self.rs_pb_method, self.bo_method, self.z_method = rs_pb_method, bo_method, z_method
        self._rs, self._pb = RS_PB_METHODS[rs_pb_method]
        self._bo = BO_METHODS[bo_method]
        self._z = Z_METHODS[z_method]
        self.tpc, self.ppc = pseudocriticals(self.gas_sg)
        self.rho_w_sc = water_density_sc(self.water_salinity) if self.water_cut > 0 else None

    # -- oil ---------------------------------------------------------------------

    def bubble_point(self, t: float) -> float:
        """Pb at temperature t. Given value wins; else the correlation at Rs = GOR."""
        if self.bubble_point_given is not None:
            return self.bubble_point_given
        if self.gor <= 0.0:
            return 14.696
        return self._pb(self.gor, t, self.oil_api, self.gas_sg)

    def rsb(self, t: float) -> float:
        """Solution GOR at the bubble point."""
        if self.bubble_point_given is None or self.gor <= 0.0:
            return self.gor
        rs_corr = self._rs(self.bubble_point_given, t, self.oil_api, self.gas_sg)
        if rs_corr > 1.05 * self.gor and not self._rsb_warned:
            self._rsb_warned = True  # once per model, not once per VLP segment
            warnings.warn(
                f"BlackOilPVT: {self.rs_pb_method} gives Rs = {rs_corr:.0f} scf/stb at the given "
                f"bubble point {self.bubble_point_given:.0f} psia but GOR is {self.gor:.0f}; "
                "Rsb capped at GOR",
                UserWarning,
                stacklevel=3,
            )
        return min(rs_corr, self.gor)

    def rs(self, p: float, t: float) -> float:
        pb = self.bubble_point(t)
        rsb = self.rsb(t)
        if p >= pb or self.gor <= 0.0:
            return rsb
        return min(self._rs(p, t, self.oil_api, self.gas_sg), rsb)

    def oil_compressibility(self, p: float, t: float) -> float:
        return vasquez_beggs_co(p, t, self.oil_api, self.gas_sg, self.rs(p, t))

    def bo(self, p: float, t: float) -> float:
        pb = self.bubble_point(t)
        if p <= pb:
            return self._bo(self.rs(p, t), t, self.oil_api, self.gas_sg)
        bob = self._bo(self.rsb(t), t, self.oil_api, self.gas_sg)
        return undersaturated_bo(bob, self.oil_compressibility(p, t), p, pb)

    def oil_density(self, p: float, t: float) -> float:
        return oil_density(self.rs(p, t), self.bo(p, t), self.oil_api, self.gas_sg)

    def oil_viscosity(self, p: float, t: float) -> float:
        mu_dead = beggs_robinson_dead(t, self.oil_api)
        pb = self.bubble_point(t)
        if p <= pb:
            return beggs_robinson_live(mu_dead, self.rs(p, t), p)
        mu_ob = beggs_robinson_live(mu_dead, self.rsb(t), pb)
        return vasquez_beggs_undersaturated_viscosity(mu_ob, p, pb)

    # -- gas ---------------------------------------------------------------------

    def z(self, p: float, t: float) -> float:
        return self._z(p / self.ppc, (t + 459.67) / self.tpc)

    def bg(self, p: float, t: float) -> float:
        return gas_fvf(p, t, self.z(p, t))

    def gas_density(self, p: float, t: float) -> float:
        return gas_density(p, t, self.gas_sg, self.z(p, t))

    def gas_viscosity(self, p: float, t: float) -> float:
        return lee_gonzalez_eakin(t, self.gas_sg, self.gas_density(p, t), p)

    # -- water -------------------------------------------------------------------

    def bw(self, p: float, t: float) -> float:
        return mccain_bw(p, t)

    def water_density(self, p: float, t: float) -> float:
        return water_density_sc(self.water_salinity) / mccain_bw(p, t)

    def water_viscosity(self, p: float, t: float) -> float:
        return mccain_water_viscosity(p, t, self.water_salinity)

    # -- everything --------------------------------------------------------------

    def state(self, p: float, t: float, warm: bool = False) -> PVTState:
        pb = self.bubble_point(t)
        rs = self.rs(p, t)
        bo = self.bo(p, t)
        z = self.z(p, t)
        rho_g = gas_density(p, t, self.gas_sg, z)
        st = PVTState(
            p=p,
            t=t,
            z=z,
            bg=gas_fvf(p, t, z),
            rho_g=rho_g,
            mu_g=lee_gonzalez_eakin(t, self.gas_sg, rho_g, p),
            gas_sg=self.gas_sg,
            pb=pb,
            rs=rs,
            bo=bo,
            rho_o=oil_density(rs, bo, self.oil_api, self.gas_sg),
            mu_o=self.oil_viscosity(p, t),
            co=self.oil_compressibility(p, t),
            sigma_og=baker_swerdloff_oil(p, t, self.oil_api),
            saturated=p < pb,
        )
        if self.water_cut > 0.0:
            bw = mccain_bw(p, t)
            st.bw = bw
            st.rho_w = water_density_sc(self.water_salinity) / bw
            st.mu_w = mccain_water_viscosity(p, t, self.water_salinity)
            st.cw = mccain_cw(p, t, self.water_salinity)
            st.rsw = mccain_rsw(p, t, self.water_salinity)
            st.sigma_gw = baker_swerdloff_water(p, t)
        return st

    def __repr__(self) -> str:
        return (
            f"BlackOilPVT(oil_api={self.oil_api}, gas_sg={self.gas_sg}, gor={self.gor}, "
            f"water_cut={self.water_cut}, water_salinity={self.water_salinity}, "
            f"bubble_point={self.bubble_point_given}, rs_pb_method={self.rs_pb_method!r}, "
            f"bo_method={self.bo_method!r}, z_method={self.z_method!r})"
        )
