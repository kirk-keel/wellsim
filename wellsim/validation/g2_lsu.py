"""Gate G2: the LSU PERTT Lab large-diameter air-water runs, compared POINT BY POINT.

What this is
    131 runs from the LSU Final Report (award M17PX00030) Table A.1, PDF pages 155-158,
    extracted to ``docs/validation/data/lsu_table_a1.csv``. Air and water, vertical
    upward, nominal 4, 8 and 12 in pipe, superficial liquid velocities to 4.35 m/s and gas
    to 29.7 m/s, at about atmospheric pressure. Each run reports a TOTAL pressure gradient
    and, on 93 of the 131, a liquid holdup.

    This is NOT a march. The reports give no tap spacing (BOEM M15PC00007 refers it to a
    figure) and no per-run pressure, so there is nothing to integrate over and no inlet
    condition to integrate from. What they do give is a gradient and a holdup AT a stated
    condition, so the gate compares point values: at each row's superficial velocities,
    the closure's alpha_g against the reported H_l, and the total gradient built from that
    alpha_g against the reported dp/dz.

The gradient (the reported value is total; BOEM M15PC00007 p.44)

    THE GATE DOES NOT IMPLEMENT ONE. It builds a one-segment FlowSegment at the row
    diameter, roughness and 90 degrees, converts the row superficial velocities to the
    mass rates the march takes, and calls :func:`wellsim.core.vlp.two_phase.two_phase_gradient`.
    The number scored is that function own ``dp_dx``. Nothing here duplicates a core
    formula, so the gate cannot drift from the thing it is meant to be grading.

    Until g2-4 it did implement one, and the difference is not cosmetic. The gate form was

        dp/dz = (grav + fric) / (1 - E_k),   E_k = rho_m v_m v_sg / (g_c p 144)

    the HOMOGENEOUS acceleration term, which assumes both phases accelerate together at
    the mixture velocity. M3-2b superseded it: the march differentiates the SLIP momentum
    flux G_g v_g + G_l v_l, so each phase is accelerated at its own velocity through the
    closure own dalpha/drho_g, and the singular point is det = 0 of the 2x2
    momentum-energy system rather than E_k = 1. The march keeps the homogeneous form only
    as a labelled diagnostic, ``MixtureState.m_coef_homogeneous``.

    At 14.7 psia the two disagree strongly, because the homogeneous form divides by
    1 - E_k with E_k proportional to 1/p. That is the term G2-2 measured the annular C0
    blow-up inside, and it is why the annular result had to be re-measured through the
    march before anything is concluded from it.

    A row is not evaluable when det <= 0, at or past the march singular point.

Assumptions, each stated because none is in the source per run
    pressure     14.7 psia (Interim Report Table 1 gives the campaign as "approximately
                 14.7 psi"; BOEM M15PC00007 p.13 says "close to atmospheric"). The
                 sensitivity to this is computed ONCE, not per row, by
                 :func:`pressure_sensitivity`.
    temperature  70 degF, ambient. Nothing per run is reported.
    water        rho 62.3 lbm/ft3, mu 1.0 cp, sigma 72 dyne/cm at those conditions.
    roughness    PVC 6e-5 in (the 4 and 12 in sections), commercial steel 0.0018 in (the
                 8 in section). Materials from BOEM M15PC00007 pp. 30, 36.

Air goes through :mod:`wellsim.core.pvt.air`, an ideal-gas path that refuses DAK.
"""

from __future__ import annotations

import csv
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from wellsim.core.geometry import FlowSegment
from wellsim.core.pvt.air import AirIdealGas
from wellsim.core.vlp.two_phase import LiquidProps, two_phase_gradient

__all__ = [
    "Row", "Point", "load_rows", "evaluate", "evaluate_all", "summarise",
    "pressure_sensitivity", "regime_confusion", "normalise_observed_regime",
    "DEFAULT_CSV", "REF_PRESSURE_PSIA", "REF_TEMPERATURE_DEGF",
    "WATER_RHO_LBM_FT3", "WATER_MU_CP", "WATER_SIGMA_DYNE_CM", "FAST_SUBSET_RUNS",
    "Consistency", "consistency", "consistency_summary", "screened_rows", "UNCERTAINTY",
    "implied_gas_velocity_ratio", "WATER", "segment_of", "WATER_CP_BTU_LBM_DEGF",
    "WATER_C_FT_S",
]

DEFAULT_CSV = pathlib.Path(__file__).resolve().parents[2] / "docs" / "validation" / "data" / "lsu_table_a1.csv"

REF_PRESSURE_PSIA = 14.7
REF_TEMPERATURE_DEGF = 70.0
WATER_RHO_LBM_FT3 = 62.3
WATER_MU_CP = 1.0
WATER_SIGMA_DYNE_CM = 72.0
WATER_CP_BTU_LBM_DEGF = 1.0    # MEMORY; enters only the energy row of the 2x2 system
WATER_C_FT_S = 4900.0          # MEMORY; enters only the Wallis c_m diagnostic, not dp/dx
ROUGHNESS_IN = {"PVC": 6.0e-5, "steel": 0.0018}
G_C = 32.174
M_S_TO_FT_S = 1.0 / 0.3048
KPA_M_TO_PSI_FT = 0.1450377 * 0.3048  # kPa -> psi, then per m -> per ft

# 12 runs for the default suite, chosen to span all three diameters AND every one of the
# seven reported flow regimes, so a regression in any regime shows up without the slow run:
#   1 Bubbly, 2 Slug/Churn, 3 Churn, 8 Annular (all 0.10 m)
#   15 Churn/Annular (0.10), 44 Slug (0.20), 46 Churn (0.20), 50 Bubble (0.20)
#   100 Churn (0.30), 122 and 131 unlabelled high-rate (0.30), 123 Bubbly (0.30)
FAST_SUBSET_RUNS = (1, 2, 3, 8, 15, 44, 46, 50, 100, 122, 123, 131)


@dataclass(frozen=True)
class Row:
    """One reported run, in the units the source uses plus the march units."""
    run: int
    pdf_page: int
    nominal_id_m: str
    id_in: float
    material: str
    u_sl_m_s: float
    u_sg_m_s: float
    dpdz_kpa_m: float
    h_l: float | None
    obs_regime: str

    @property
    def v_sl(self) -> float:
        """ft/s"""
        return self.u_sl_m_s * M_S_TO_FT_S

    @property
    def v_sg(self) -> float:
        """ft/s"""
        return self.u_sg_m_s * M_S_TO_FT_S

    @property
    def dpdz_psi_ft(self) -> float:
        return self.dpdz_kpa_m * KPA_M_TO_PSI_FT

    @property
    def roughness_in(self) -> float:
        return ROUGHNESS_IN[self.material]

    @property
    def slip_ratio(self) -> float:
        return self.v_sg / self.v_sl if self.v_sl > 0.0 else math.inf


@dataclass
class Point:
    """The march gradient at one row conditions, beside what was reported."""
    row: Row
    alpha_g: float
    h_l_pred: float
    regime_pred: str
    rho_g: float
    rho_m: float
    mu_m: float
    v_m: float
    re: float
    f: float
    grav_psi_ft: float
    fric_psi_ft: float
    e_k: float  # 1 - det, the march own acceleration measure, NOT the homogeneous E_k
    det: float
    e_k_homogeneous: float  # the pre-g2-4 rho_m v_m v_sg/(g_c p 144), diagnostic only
    dpdz_pred_psi_ft: float | None  # None where det <= 0, the march singular point
    evaluable: bool = True
    note: str = ""

    @property
    def holdup_error(self) -> float | None:
        """Relative error on H_l, model minus reported over reported."""
        if self.row.h_l is None or self.row.h_l == 0.0:
            return None
        return (self.h_l_pred - self.row.h_l) / self.row.h_l

    @property
    def gradient_error(self) -> float | None:
        if not self.evaluable or self.dpdz_pred_psi_ft is None or self.row.dpdz_psi_ft == 0.0:
            return None
        return (self.dpdz_pred_psi_ft - self.row.dpdz_psi_ft) / self.row.dpdz_psi_ft


def load_rows(path: str | pathlib.Path = DEFAULT_CSV) -> list[Row]:
    """Read the CSV. Units convert HERE, never in the file."""
    rows: list[Row] = []
    with pathlib.Path(path).open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    for d in csv.DictReader(lines):
        rows.append(Row(
            run=int(d["run"]), pdf_page=int(d["pdf_page"]), nominal_id_m=d["nominal_id_m"],
            id_in=float(d["id_in"]), material=d["material"],
            u_sl_m_s=float(d["u_sl_m_s"]), u_sg_m_s=float(d["u_sg_m_s"]),
            dpdz_kpa_m=float(d["dpdz_kpa_m"]),
            h_l=float(d["h_l"]) if d["h_l"] else None,
            obs_regime=d["obs_regime"].strip(),
        ))
    return rows


WATER = LiquidProps(rho_l=WATER_RHO_LBM_FT3, mu_l=WATER_MU_CP, cp_l=WATER_CP_BTU_LBM_DEGF,
                    sigma=WATER_SIGMA_DYNE_CM, c_l=WATER_C_FT_S)


def segment_of(row: Row) -> FlowSegment:
    """A one-segment vertical pipe at the row diameter, material roughness and 90 degrees.

    The length is nominal: the gate compares a POINT value, so nothing is integrated over
    it. The march takes mass rates, so the row superficial velocities are converted at the
    area this returns and converted straight back inside `mixture_state`.
    """
    return FlowSegment(
        top_md=0.0, bottom_md=1.0, d_hyd=row.id_in,
        area=math.pi * (row.id_in / 12.0) ** 2 / 4.0,
        roughness=row.roughness_in, sin_theta=1.0,
        conduit=f"LSU {row.nominal_id_m} m section, {row.material}",
    )


def evaluate(row: Row, p_psia: float = REF_PRESSURE_PSIA, t_degf: float = REF_TEMPERATURE_DEGF,
             holdup_fn: Callable[..., Any] | None = None) -> Point:
    """The MARCH gradient at one row conditions. This function implements no physics.

    ``holdup_fn`` defaults to the march own closure (None is passed straight through). A
    different callable is how the Beggs-Brill cross-check is injected from the tests
    without putting it on the model path; it must return an object with ``alpha_g``,
    ``regime``, ``iterations`` and ``dalpha_drho_g``.
    """
    air = AirIdealGas()
    seg = segment_of(row)
    rho_g = air.gas_density(p_psia, t_degf)
    w_g = rho_g * row.v_sg * seg.area
    w_l = WATER.rho_l * row.v_sl * seg.area
    g = two_phase_gradient(air, WATER, p_psia, t_degf, w_g, w_l, seg, closure=holdup_fn)

    # the pre-g2-4 homogeneous acceleration measure, kept beside the march one so the two
    # can be reported against each other; it is not used to build anything
    e_k_hom = g.rho * g.v * g.v_sg / (G_C * p_psia * 144.0)

    if g.det > 0.0:
        dpdz, evaluable, note = g.dp_dx, True, ""
    else:
        dpdz, evaluable = None, False
        note = (f"det = {g.det:.4f} is at or below 0 at {p_psia:g} psia: the row sits at or "
                "past the march singular point, where a point value does not exist")
    return Point(row=row, alpha_g=g.alpha_g, h_l_pred=1.0 - g.alpha_g, regime_pred=g.regime,
                 rho_g=g.rho_g, rho_m=g.rho, mu_m=g.mu, v_m=g.v, re=g.re, f=g.f,
                 grav_psi_ft=g.grav, fric_psi_ft=g.fric, e_k=g.ek, det=g.det,
                 e_k_homogeneous=e_k_hom, dpdz_pred_psi_ft=dpdz,
                 evaluable=evaluable, note=note)


def evaluate_all(rows: Sequence[Row], **kw) -> list[Point]:
    return [evaluate(r, **kw) for r in rows]


def _mae(values: Sequence[float]) -> float | None:
    vals = [abs(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _within(values: Sequence[float], band: float) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(1 for v in vals if abs(v) <= band) / len(vals)


def summarise(points: Sequence[Point], holdup_runs: set[int] | None = None) -> dict[str, Any]:
    """Mean absolute error and the within-20% fraction, overall and split.

    ``holdup_runs`` restricts the HOLDUP statistics to those run numbers, leaving the
    gradient statistics over everything. That is how the consistency screen is applied
    (docs/validation/G2-consistency.md): 39 of the 93 reported holdups contradict the
    gradient reported for the same run by more than the run's own stated uncertainty, and
    the model cannot be graded against a number its own source contradicts. Nothing is
    tuned to either column; the failing rows are simply not scored on holdup.
    """
    out: dict[str, Any] = {}

    def block(sel: Sequence[Point]) -> dict[str, Any]:
        h = [p.holdup_error for p in sel
             if p.holdup_error is not None
             and (holdup_runs is None or p.row.run in holdup_runs)]
        g = [p.gradient_error for p in sel if p.gradient_error is not None]
        return {
            "n": len(sel),
            "n_holdup": len(h), "holdup_mae": _mae(h), "holdup_within_20pct": _within(h, 0.20),
            "n_gradient": len(g), "gradient_mae": _mae(g), "gradient_within_20pct": _within(g, 0.20),
            "n_not_evaluable": sum(1 for p in sel if not p.evaluable),
        }

    out["all"] = block(points)
    out["by_diameter"] = {
        nid: block([p for p in points if p.row.nominal_id_m == nid])
        for nid in sorted({p.row.nominal_id_m for p in points})
    }
    out["by_observed_regime"] = {
        reg: block([p for p in points if p.row.obs_regime == reg])
        for reg in sorted({p.row.obs_regime for p in points if p.row.obs_regime})
    }
    low = [p for p in points if p.row.slip_ratio < 1.0]
    out["slip_below_1"] = block(low)
    return out


def pressure_sensitivity(rows: Sequence[Row], p_lo: float = REF_PRESSURE_PSIA,
                         p_hi: float = 16.0) -> dict[str, Any]:
    """How much the predicted gradient moves if the assumed pressure were ``p_hi``
    instead of ``p_lo``. Computed ONCE over the whole set, not per row."""
    a = evaluate_all(rows, p_psia=p_lo)
    b = evaluate_all(rows, p_psia=p_hi)
    moves = []
    for pa, pb in zip(a, b):
        if pa.dpdz_pred_psi_ft and pb.dpdz_pred_psi_ft:
            moves.append((pb.dpdz_pred_psi_ft - pa.dpdz_pred_psi_ft) / pa.dpdz_pred_psi_ft)
    sa, sb = summarise(a), summarise(b)
    return {
        "p_lo_psia": p_lo, "p_hi_psia": p_hi,
        "n_compared": len(moves),
        "mean_relative_move": sum(moves) / len(moves) if moves else None,
        "max_relative_move": max(moves, key=abs) if moves else None,
        "gradient_mae_at_p_lo": sa["all"]["gradient_mae"],
        "gradient_mae_at_p_hi": sb["all"]["gradient_mae"],
        "not_evaluable_at_p_lo": sa["all"]["n_not_evaluable"],
        "not_evaluable_at_p_hi": sb["all"]["n_not_evaluable"],
    }


# --- flow regime ---------------------------------------------------------------------

def normalise_observed_regime(label: str) -> str:
    """LSU writes Bubble and Bubbly for the same thing, and uses slashes for transitions.
    Mapped onto the closure's own vocabulary; a transition keeps both names."""
    s = label.strip().lower().replace("bubble", "bubbly")
    return {"": "", "slug/churn": "slug/churn", "churn/annular": "churn/annular"}.get(s, s)


def regime_confusion(points: Sequence[Point]) -> dict[tuple[str, str], int]:
    """(observed, predicted) -> count, over the rows that report a regime."""
    table: dict[tuple[str, str], int] = {}
    for p in points:
        obs = normalise_observed_regime(p.row.obs_regime)
        if not obs:
            continue
        key = (obs, p.regime_pred)
        table[key] = table.get(key, 0) + 1
    return table


# --- data consistency screen (step 2b) ------------------------------------------------

# BOEM M15PC00007 Table 4, p.34: reported measurement uncertainty per test section.
UNCERTAINTY = {          # nominal ID (m) -> (dp/dz psi/ft, H_l absolute)
    "0.10": (0.040, 0.19),   # the 3.83 in section
    "0.20": (0.006, 0.03),   # the 7.83 in section
    "0.30": (0.020, 0.05),   # the 11.7 in section
}


@dataclass
class Consistency:
    """One row's two reported columns checked against each other.

    The reported gradient is total. Strip the friction and acceleration terms from it,
    using the closure's own mixture velocity, and what is left is hydrostatic, which fixes
    the liquid holdup. That implied holdup is compared with the holdup the same row
    reports. Neither column is adjusted; this only asks whether they agree.
    """
    row: Row
    h_l_reported: float
    h_l_implied: float
    dpdz_reported_psi_ft: float
    fric_psi_ft: float
    accel_psi_ft: float
    hydrostatic_psi_ft: float
    v_m: float
    tol_h_l: float           # combined uncertainty on the difference
    tol_from_h_l: float
    tol_from_dpdz: float

    @property
    def difference(self) -> float:
        """Implied minus reported. Negative means the gradient is too small for the
        holdup the row claims."""
        return self.h_l_implied - self.h_l_reported

    @property
    def within_uncertainty(self) -> bool:
        return abs(self.difference) <= self.tol_h_l


def consistency(rows: Sequence[Row], p_psia: float = REF_PRESSURE_PSIA,
                t_degf: float = REF_TEMPERATURE_DEGF) -> list[Consistency]:
    """The screen, over the rows that report a holdup.

    H_l_implied = (dp/dz - dp/dz_friction - dp/dz_accel) / (rho_l g)

    with rho_l g in field units = rho_l / 144 psi/ft per unit holdup. The gas contributes
    alpha rho_g to the real mixture density, which is under 0.15% of rho_l here, and is
    dropped as the formula above drops it.

    The friction and acceleration terms are evaluated at the closure's mixture state, which
    is the only self-consistent choice: the reported holdup is what is being tested, so it
    cannot also be the input that sets those terms.
    """
    out: list[Consistency] = []
    rho_l_grad = WATER_RHO_LBM_FT3 / 144.0     # psi/ft per unit liquid holdup
    for row in rows:
        if row.h_l is None:
            continue
        pt = evaluate(row, p_psia=p_psia, t_degf=t_degf)
        # The acceleration term as a gradient, taken from the march gradient the gate now
        # scores (g2-4). The expression is unchanged from the homogeneous version because
        # it is an identity in either: dp/dx = (grav + fric)/det, so the accelerational
        # part is dp/dx - (grav + fric) = dp/dx (1 - det) = dp/dx E_k. Only the VALUE of
        # E_k changed, from the homogeneous rho_m v_m v_sg/(g_c p 144) to the march 1 - det.
        accel = pt.e_k * (pt.dpdz_pred_psi_ft if pt.dpdz_pred_psi_ft is not None else 0.0)
        hydro = row.dpdz_psi_ft - pt.fric_psi_ft - accel
        d_dpdz, d_hl = UNCERTAINTY[row.nominal_id_m]
        tol_from_dpdz = d_dpdz / rho_l_grad
        tol = math.sqrt(d_hl * d_hl + tol_from_dpdz * tol_from_dpdz)
        out.append(Consistency(
            row=row, h_l_reported=row.h_l, h_l_implied=hydro / rho_l_grad,
            dpdz_reported_psi_ft=row.dpdz_psi_ft, fric_psi_ft=pt.fric_psi_ft,
            accel_psi_ft=accel, hydrostatic_psi_ft=hydro, v_m=pt.v_m,
            tol_h_l=tol, tol_from_h_l=d_hl, tol_from_dpdz=tol_from_dpdz,
        ))
    return out


def consistency_summary(cons: Sequence[Consistency]) -> dict[str, Any]:
    """Distribution of the difference, and the failure counts, split two ways."""
    def block(sel: Sequence[Consistency]) -> dict[str, Any]:
        if not sel:
            return {"n": 0}
        d = sorted(c.difference for c in sel)
        n = len(d)
        return {
            "n": n,
            "mean": sum(d) / n,
            "median": d[n // 2] if n % 2 else 0.5 * (d[n // 2 - 1] + d[n // 2]),
            "min": d[0], "max": d[-1],
            "n_fail": sum(1 for c in sel if not c.within_uncertainty),
            "frac_fail": sum(1 for c in sel if not c.within_uncertainty) / n,
            "n_negative": sum(1 for c in sel if c.difference < 0.0),
        }

    vm_bins = [(0.0, 2.0), (2.0, 5.0), (5.0, 15.0), (15.0, 1e9)]
    return {
        "all": block(cons),
        "by_diameter": {k: block([c for c in cons if c.row.nominal_id_m == k])
                        for k in sorted({c.row.nominal_id_m for c in cons})},
        "by_mixture_velocity": {
            f"{lo:g}-{hi:g} ft/s" if hi < 1e8 else f">{lo:g} ft/s":
                block([c for c in cons if lo <= c.v_m < hi])
            for lo, hi in vm_bins
        },
    }


def screened_rows(cons: Sequence[Consistency]) -> set[int]:
    """Run numbers whose two reported columns agree inside the measurement uncertainty."""
    return {c.row.run for c in cons if c.within_uncertainty}


def implied_gas_velocity_ratio(row: Row) -> float | None:
    """How fast the gas would have to move in situ for the REPORTED holdup to be true,
    against what drift flux allows: (v_sg / alpha_reported) / (C0 v_m + v_d).

    A purely kinematic check on the holdup column, independent of the gradient column and
    of any pressure assumption. C0 = 1.2 and v_d = 0.8 ft/s are the bubbly-flow values;
    the test only asks whether a row is off by a factor, not by a few per cent, so the
    exact coefficients do not matter. Ratios near 1 are normal. A ratio of 20 means the
    reported holdup cannot be a measurement of that run.
    """
    if row.h_l is None:
        return None
    alpha = 1.0 - row.h_l
    if alpha <= 0.0:
        return None
    return (row.v_sg / alpha) / (1.2 * (row.v_sg + row.v_sl) + 0.8)
