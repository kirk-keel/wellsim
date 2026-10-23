"""IADC WellSharp / IWCF standard well control calculations and kill sheet engine.

Implements standard oilfield well control formulations:
- Hydrostatic pressure and equivalent mud weights (EMW)
- Formation pressure from Shut-In Drillpipe Pressure (SIDPP)
- Influx height, influx gradient, and influx fluid identification (gas, oil, water)
- Kill Mud Weight (KMW) with optional safety margin and rounding
- Initial Circulating Pressure (ICP) and Final Circulating Pressure (FCP)
- Maximum Allowable Annular Surface Pressure (MAASP) at casing shoe
- Drillpipe pressure step-down schedule (strokes or pumped volume vs. DPP)
- Kick tolerance and maximum allowable influx volume

All internal units are canonical oilfield units (ppg, ft, psi, bbl, bbl/ft, bbl/stk).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "PSI_PER_PPG_FT",
    "hydrostatic_pressure",
    "pressure_gradient",
    "equivalent_mud_weight",
    "equivalent_circulating_density",
    "formation_pressure",
    "kill_mud_weight",
    "maasp",
    "initial_circulating_pressure",
    "final_circulating_pressure",
    "influx_height",
    "influx_gradient",
    "identify_influx_type",
    "kick_tolerance_volume",
    "StepDownPoint",
    "StepDownSchedule",
    "KillSheetInputs",
    "KillSheetResult",
    "calculate_kill_sheet",
]

# Standard petroleum engineering constant: 7.4805 gal/ft3 / 144 in2/ft2 = 0.051948 ~ 0.052
PSI_PER_PPG_FT: float = 0.052


def hydrostatic_pressure(mud_weight_ppg: float, tvd_ft: float) -> float:
    """Hydrostatic pressure exerted by a fluid column (psi).

    P_hyd = 0.052 * MW * TVD
    """
    if mud_weight_ppg < 0 or tvd_ft < 0:
        raise ValueError(f"Mud weight ({mud_weight_ppg}) and TVD ({tvd_ft}) must be non-negative.")
    return PSI_PER_PPG_FT * mud_weight_ppg * tvd_ft


def pressure_gradient(mud_weight_ppg: float) -> float:
    """Pressure gradient of a fluid in psi/ft.

    G = 0.052 * MW
    """
    if mud_weight_ppg < 0:
        raise ValueError(f"Mud weight ({mud_weight_ppg}) must be non-negative.")
    return PSI_PER_PPG_FT * mud_weight_ppg


def equivalent_mud_weight(pressure_psi: float, tvd_ft: float) -> float:
    """Equivalent Mud Weight (EMW) in ppg from a given pressure and TVD.

    EMW = P / (0.052 * TVD)
    """
    if tvd_ft <= 0:
        raise ValueError(f"TVD must be strictly positive, got {tvd_ft} ft.")
    if pressure_psi < 0:
        raise ValueError(f"Pressure must be non-negative, got {pressure_psi} psi.")
    return pressure_psi / (PSI_PER_PPG_FT * tvd_ft)


def equivalent_circulating_density(
    mud_weight_ppg: float,
    annular_pressure_loss_psi: float,
    tvd_ft: float,
) -> float:
    """Equivalent Circulating Density (ECD) in ppg.

    ECD = MW + (APL / (0.052 * TVD))
    """
    if tvd_ft <= 0:
        raise ValueError(f"TVD must be strictly positive, got {tvd_ft} ft.")
    return mud_weight_ppg + (annular_pressure_loss_psi / (PSI_PER_PPG_FT * tvd_ft))


def formation_pressure(sidpp_psi: float, omw_ppg: float, tvd_ft: float) -> float:
    """Pore / formation pressure determined from shut-in drillpipe pressure (psia).

    P_form = P_hyd_drillstring + SIDPP
    """
    if sidpp_psi < 0:
        raise ValueError(f"SIDPP must be non-negative, got {sidpp_psi} psi.")
    return hydrostatic_pressure(omw_ppg, tvd_ft) + sidpp_psi


def kill_mud_weight(
    omw_ppg: float,
    sidpp_psi: float,
    tvd_ft: float,
    safety_margin_ppg: float = 0.0,
    round_to: float | None = None,
) -> float:
    """Calculate required Kill Mud Weight (KMW) in ppg.

    KMW = OMW + (SIDPP / (0.052 * TVD)) + safety_margin

    Args:
        omw_ppg: Original mud weight (ppg).
        sidpp_psi: Shut-in drill pipe pressure (psi).
        tvd_ft: True vertical depth of the well (ft).
        safety_margin_ppg: Optional trip/safety margin to add (ppg, default 0.0).
        round_to: Optional rounding step, e.g. 0.1 for 0.1 ppg precision ceiling.
    """
    if tvd_ft <= 0:
        raise ValueError(f"TVD must be strictly positive, got {tvd_ft} ft.")
    if sidpp_psi < 0:
        raise ValueError(f"SIDPP must be non-negative, got {sidpp_psi} psi.")

    exact_kmw = omw_ppg + (sidpp_psi / (PSI_PER_PPG_FT * tvd_ft)) + safety_margin_ppg
    if round_to is not None and round_to > 0:
        # Standard field practice: round UP to next 0.1 ppg
        return math.ceil(exact_kmw / round_to) * round_to
    return exact_kmw


def maasp(lot_emw_ppg: float, current_mw_ppg: float, tvd_shoe_ft: float) -> float:
    """Maximum Allowable Annular Surface Pressure (MAASP) in psi at casing shoe.

    MAASP = (LOT_EMW - Current_MW) * 0.052 * TVD_shoe
    """
    if tvd_shoe_ft <= 0:
        raise ValueError(f"Casing shoe TVD must be strictly positive, got {tvd_shoe_ft} ft.")
    if lot_emw_ppg < current_mw_ppg:
        # Mud weight already exceeds shoe fracture gradient
        return (lot_emw_ppg - current_mw_ppg) * PSI_PER_PPG_FT * tvd_shoe_ft
    return (lot_emw_ppg - current_mw_ppg) * PSI_PER_PPG_FT * tvd_shoe_ft


def initial_circulating_pressure(sidpp_psi: float, scrp_psi: float) -> float:
    """Initial Circulating Pressure (ICP) in psi for Driller's and Wait & Weight methods.

    ICP = SIDPP + SCRP
    """
    if sidpp_psi < 0 or scrp_psi < 0:
        raise ValueError(f"SIDPP ({sidpp_psi}) and SCRP ({scrp_psi}) must be non-negative.")
    return sidpp_psi + scrp_psi


def final_circulating_pressure(scrp_psi: float, kmw_ppg: float, omw_ppg: float) -> float:
    """Final Circulating Pressure (FCP) in psi once kill mud fills the drillstring.

    FCP = SCRP * (KMW / OMW)
    """
    if omw_ppg <= 0:
        raise ValueError(f"Original mud weight must be positive, got {omw_ppg} ppg.")
    if scrp_psi < 0 or kmw_ppg < 0:
        raise ValueError("SCRP and KMW must be non-negative.")
    return scrp_psi * (kmw_ppg / omw_ppg)


def influx_height(pit_gain_bbl: float, annular_capacity_bbl_ft: float) -> float:
    """Calculate the height of the influx in the annulus (ft).

    h_influx = PitGain / AnnularCapacity
    """
    if annular_capacity_bbl_ft <= 0:
        raise ValueError(f"Annular capacity must be strictly positive, got {annular_capacity_bbl_ft} bbl/ft.")
    if pit_gain_bbl < 0:
        raise ValueError(f"Pit gain cannot be negative, got {pit_gain_bbl} bbl.")
    return pit_gain_bbl / annular_capacity_bbl_ft


def influx_gradient(
    omw_ppg: float,
    sidpp_psi: float,
    sicp_psi: float,
    influx_height_ft: float,
) -> float:
    """Estimate the density/pressure gradient of the influx fluid (psi/ft).

    From hydrostatic balance:
        P_form = P_hyd_dp + SIDPP = P_hyd_annulus_above + P_influx + SICP
        SICP - SIDPP = (0.052 * OMW - G_influx) * h_influx
        G_influx = 0.052 * OMW - ((SICP - SIDPP) / h_influx)
    """
    if influx_height_ft <= 0:
        raise ValueError(f"Influx height must be strictly positive, got {influx_height_ft} ft.")
    omw_grad = pressure_gradient(omw_ppg)
    delta_p = sicp_psi - sidpp_psi
    return omw_grad - (delta_p / influx_height_ft)


def identify_influx_type(gradient_psi_ft: float) -> Literal["gas", "oil", "water", "unknown"]:
    """Classify influx type from estimated pressure gradient (psi/ft).

    - Gas: gradient <= 0.15 psi/ft (~0.1 psi/ft typical methane)
    - Oil: 0.15 < gradient <= 0.40 psi/ft
    - Water / Saltwater: > 0.40 psi/ft (~0.433 - 0.465 psi/ft)
    """
    if gradient_psi_ft < 0:
        return "gas"  # Very light gas / vacuum effect
    if gradient_psi_ft <= 0.15:
        return "gas"
    if gradient_psi_ft <= 0.40:
        return "oil"
    if gradient_psi_ft <= 0.55:
        return "water"
    return "unknown"


def kick_tolerance_volume(
    lot_emw_ppg: float,
    omw_ppg: float,
    tvd_shoe_ft: float,
    sicp_psi: float,
    gas_gradient_psi_ft: float = 0.10,
    annular_capacity_bbl_ft: float = 0.05,
) -> float:
    """Calculate the maximum kick volume (bbl) allowable at the casing shoe before fracturing.

    V_kick_max = (MAASP - SICP) / ( (0.052 * OMW - G_gas) / AnnularCapacity )
    """
    current_maasp = maasp(lot_emw_ppg, omw_ppg, tvd_shoe_ft)
    grad_diff = pressure_gradient(omw_ppg) - gas_gradient_psi_ft
    if grad_diff <= 0:
        return 0.0
    pressure_margin = current_maasp - sicp_psi
    if pressure_margin <= 0:
        return 0.0
    return (pressure_margin / grad_diff) * annular_capacity_bbl_ft


@dataclass(frozen=True)
class StepDownPoint:
    strokes: int
    volume_bbl: float
    dpp_psi: float


@dataclass(frozen=True)
class StepDownSchedule:
    icp: float
    fcp: float
    strokes_to_bit: int
    volume_to_bit_bbl: float
    points: list[StepDownPoint]

    def dpp_at_stroke(self, stroke: int) -> float:
        """Interpolate drillpipe pressure at any stroke count from 0 to strokes_to_bit."""
        if stroke <= 0:
            return self.icp
        if stroke >= self.strokes_to_bit:
            return self.fcp
        fraction = stroke / self.strokes_to_bit
        return self.icp - fraction * (self.icp - self.fcp)

    def as_dict(self) -> dict[str, Any]:
        return {
            "icp_psi": self.icp,
            "fcp_psi": self.fcp,
            "strokes_to_bit": self.strokes_to_bit,
            "volume_to_bit_bbl": self.volume_to_bit_bbl,
            "points": [
                {"strokes": p.strokes, "volume_bbl": p.volume_bbl, "dpp_psi": p.dpp_psi}
                for p in self.points
            ],
        }


@dataclass(frozen=True)
class KillSheetInputs:
    omw_ppg: float
    tvd_well_ft: float
    tvd_shoe_ft: float
    lot_emw_ppg: float
    sidpp_psi: float
    sicp_psi: float
    pit_gain_bbl: float
    scrp_psi: float
    drillstring_volume_bbl: float
    annular_volume_bbl: float
    annular_capacity_around_bha_bbl_ft: float
    pump_displacement_bbl_per_stk: float
    safety_margin_ppg: float = 0.0
    step_down_interval_strokes: int = 100


@dataclass(frozen=True)
class KillSheetResult:
    inputs: KillSheetInputs
    formation_pressure_psi: float
    kmw_ppg: float
    kmw_rounded_ppg: float
    icp_psi: float
    fcp_psi: float
    maasp_initial_psi: float
    maasp_with_kmw_psi: float
    influx_height_ft: float
    influx_gradient_psi_ft: float
    influx_type: Literal["gas", "oil", "water", "unknown"]
    strokes_surface_to_bit: int
    strokes_bit_to_surface: int
    total_kill_strokes: int
    step_down: StepDownSchedule
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "formation_pressure_psi": self.formation_pressure_psi,
            "kmw_ppg": self.kmw_ppg,
            "kmw_rounded_ppg": self.kmw_rounded_ppg,
            "icp_psi": self.icp_psi,
            "fcp_psi": self.fcp_psi,
            "maasp_initial_psi": self.maasp_initial_psi,
            "maasp_with_kmw_psi": self.maasp_with_kmw_psi,
            "influx_height_ft": self.influx_height_ft,
            "influx_gradient_psi_ft": self.influx_gradient_psi_ft,
            "influx_type": self.influx_type,
            "strokes_surface_to_bit": self.strokes_surface_to_bit,
            "strokes_bit_to_surface": self.strokes_bit_to_surface,
            "total_kill_strokes": self.total_kill_strokes,
            "step_down": self.step_down.as_dict(),
            "warnings": list(self.warnings),
        }


def calculate_kill_sheet(inputs: KillSheetInputs) -> KillSheetResult:
    """Execute a complete IADC WellSharp / IWCF well control kill sheet calculation."""
    warnings: list[str] = []

    # 1. Formation pressure
    p_form = formation_pressure(inputs.sidpp_psi, inputs.omw_ppg, inputs.tvd_well_ft)

    # 2. Kill mud weight (exact and standard field rounded up to 0.1 ppg)
    kmw_exact = kill_mud_weight(
        inputs.omw_ppg, inputs.sidpp_psi, inputs.tvd_well_ft, inputs.safety_margin_ppg
    )
    kmw_round = kill_mud_weight(
        inputs.omw_ppg, inputs.sidpp_psi, inputs.tvd_well_ft, inputs.safety_margin_ppg, round_to=0.1
    )

    # 3. Circulating pressures
    icp = initial_circulating_pressure(inputs.sidpp_psi, inputs.scrp_psi)
    fcp = final_circulating_pressure(inputs.scrp_psi, kmw_exact, inputs.omw_ppg)

    # 4. MAASP (with original mud weight and reduced MAASP once kill mud enters the annulus)
    maasp_init = maasp(inputs.lot_emw_ppg, inputs.omw_ppg, inputs.tvd_shoe_ft)
    maasp_kmw = maasp(inputs.lot_emw_ppg, kmw_round, inputs.tvd_shoe_ft)

    if inputs.sicp_psi > maasp_init:
        warnings.append(
            f"Shut-in casing pressure ({inputs.sicp_psi:.1f} psi) exceeds initial MAASP "
            f"({maasp_init:.1f} psi) by {inputs.sicp_psi - maasp_init:.1f} psi! Shoe integrity at risk."
        )

    # 5. Influx analysis
    h_inf = influx_height(inputs.pit_gain_bbl, inputs.annular_capacity_around_bha_bbl_ft)
    grad_inf = influx_gradient(inputs.omw_ppg, inputs.sidpp_psi, inputs.sicp_psi, h_inf)
    inf_type = identify_influx_type(grad_inf)

    # 6. Strokes and circulation volumes
    if inputs.pump_displacement_bbl_per_stk <= 0:
        raise ValueError("Pump displacement must be positive.")

    stk_to_bit = int(math.ceil(inputs.drillstring_volume_bbl / inputs.pump_displacement_bbl_per_stk))
    stk_to_surf = int(math.ceil(inputs.annular_volume_bbl / inputs.pump_displacement_bbl_per_stk))
    total_stk = stk_to_bit + stk_to_surf

    # 7. Drillpipe step-down schedule
    points: list[StepDownPoint] = []
    interval = max(10, inputs.step_down_interval_strokes)
    curr_stk = 0
    while curr_stk < stk_to_bit:
        vol = curr_stk * inputs.pump_displacement_bbl_per_stk
        dpp = icp - (curr_stk / stk_to_bit) * (icp - fcp)
        points.append(StepDownPoint(strokes=curr_stk, volume_bbl=round(vol, 2), dpp_psi=round(dpp, 1)))
        curr_stk += interval

    # Final point at bit
    points.append(
        StepDownPoint(
            strokes=stk_to_bit,
            volume_bbl=round(inputs.drillstring_volume_bbl, 2),
            dpp_psi=round(fcp, 1),
        )
    )

    step_down = StepDownSchedule(
        icp=icp,
        fcp=fcp,
        strokes_to_bit=stk_to_bit,
        volume_to_bit_bbl=inputs.drillstring_volume_bbl,
        points=points,
    )

    return KillSheetResult(
        inputs=inputs,
        formation_pressure_psi=round(p_form, 1),
        kmw_ppg=round(kmw_exact, 2),
        kmw_rounded_ppg=round(kmw_round, 1),
        icp_psi=round(icp, 1),
        fcp_psi=round(fcp, 1),
        maasp_initial_psi=round(maasp_init, 1),
        maasp_with_kmw_psi=round(maasp_kmw, 1),
        influx_height_ft=round(h_inf, 1),
        influx_gradient_psi_ft=round(grad_inf, 4),
        influx_type=inf_type,
        strokes_surface_to_bit=stk_to_bit,
        strokes_bit_to_surface=stk_to_surf,
        total_kill_strokes=total_stk,
        step_down=step_down,
        warnings=warnings,
    )
