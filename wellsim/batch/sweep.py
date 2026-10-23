"""Parametric sweeps and sensitivity analysis engine (Milestone M7).

Enables multi-case parameter sweeps, sensitivity tornado generation, and tabular reporting:
- Single-parameter sweeps (e.g. sweep choke diameter, permeability, skin)
- Kill mud weight vs. pump pressure and dynamic kill rate curves
- Tornado sensitivity rankings across defensible well control inputs
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from wellsim.cases.schema import Case
from wellsim.core.solver import Result, solve

__all__ = [
    "SweepRow",
    "SweepResult",
    "TornadoItem",
    "TornadoResult",
    "sweep_parameter",
    "tornado_sensitivity",
]


@dataclass(frozen=True)
class SweepRow:
    parameter_value: Any
    flowing: bool
    gas_rate_mscfd: float
    oil_rate_stbd: float
    water_rate_stbd: float
    pwf_psia: float | None
    exit_p_psia: float | None
    exit_choked: bool
    elapsed_s: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class SweepResult:
    case_name: str
    parameter_path: str
    rows: list[SweepRow]

    def to_markdown(self) -> str:
        lines = [
            f"### Parameter Sweep: `{self.parameter_path}` ({self.case_name})",
            "",
            "| Value | Flowing | Gas Rate (Mscf/d) | Oil Rate (stb/d) | Pwf (psia) | Choked | Runtime (s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in self.rows:
            pwf_str = f"{r.pwf_psia:.1f}" if r.pwf_psia is not None else "-"
            lines.append(
                f"| {r.parameter_value} | {r.flowing} | {r.gas_rate_mscfd:.1f} | "
                f"{r.oil_rate_stbd:.1f} | {pwf_str} | {r.exit_choked} | {r.elapsed_s:.3f} |"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "parameter_path": self.parameter_path,
            "rows": [
                {
                    "value": r.parameter_value,
                    "flowing": r.flowing,
                    "gas_rate_mscfd": r.gas_rate_mscfd,
                    "oil_rate_stbd": r.oil_rate_stbd,
                    "water_rate_stbd": r.water_rate_stbd,
                    "pwf_psia": r.pwf_psia,
                    "exit_p_psia": r.exit_p_psia,
                    "exit_choked": r.exit_choked,
                    "elapsed_s": r.elapsed_s,
                }
                for r in self.rows
            ],
        }


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def sweep_parameter(case: Case, param_path: str, values: Sequence[Any]) -> SweepResult:
    """Run a parametric sweep varying `param_path` across `values`."""
    rows: list[SweepRow] = []

    for val in values:
        case_copy = copy.deepcopy(case)
        _set_nested_attr(case_copy, param_path, val)
        res: Result = solve(case_copy)

        rows.append(
            SweepRow(
                parameter_value=val,
                flowing=res.flowing,
                gas_rate_mscfd=res.gas_rate,
                oil_rate_stbd=res.oil_rate,
                water_rate_stbd=res.water_rate,
                pwf_psia=res.pwf,
                exit_p_psia=res.exit.p if res.exit else None,
                exit_choked=res.exit.choked if res.exit else False,
                elapsed_s=res.elapsed_s,
                warnings=list(res.warnings),
            )
        )

    return SweepResult(case_name=case.name, parameter_path=param_path, rows=rows)


@dataclass(frozen=True)
class TornadoItem:
    parameter_path: str
    base_value: Any
    low_value: Any
    high_value: Any
    low_gas_rate: float
    high_gas_rate: float
    rate_swing: float  # abs(high - low)


@dataclass
class TornadoResult:
    case_name: str
    base_gas_rate: float
    items: list[TornadoItem]

    def to_markdown(self) -> str:
        lines = [
            f"### Tornado Sensitivity Analysis ({self.case_name})",
            f"**Baseline Gas Rate:** {self.base_gas_rate:.1f} Mscf/d\n",
            "| Parameter | Low Value | High Value | Rate @ Low | Rate @ High | Swing (Mscf/d) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for it in sorted(self.items, key=lambda x: x.rate_swing, reverse=True):
            lines.append(
                f"| `{it.parameter_path}` | {it.low_value} | {it.high_value} | "
                f"{it.low_gas_rate:.1f} | {it.high_gas_rate:.1f} | {it.rate_swing:.1f} |"
            )
        return "\n".join(lines)


def tornado_sensitivity(
    case: Case,
    param_ranges: dict[str, tuple[Any, Any]],
) -> TornadoResult:
    """Generate a sensitivity tornado across multiple parameter ranges."""
    base_res = solve(case)
    base_rate = base_res.gas_rate if base_res.gas_rate > 0 else base_res.oil_rate
    items: list[TornadoItem] = []

    for path, (low_val, high_val) in param_ranges.items():
        # Low solve
        c_low = copy.deepcopy(case)
        _set_nested_attr(c_low, path, low_val)
        r_low = solve(c_low)
        rate_low = r_low.gas_rate if r_low.gas_rate > 0 else r_low.oil_rate

        # High solve
        c_high = copy.deepcopy(case)
        _set_nested_attr(c_high, path, high_val)
        r_high = solve(c_high)
        rate_high = r_high.gas_rate if r_high.gas_rate > 0 else r_high.oil_rate

        swing = abs(rate_high - rate_low)
        items.append(
            TornadoItem(
                parameter_path=path,
                base_value="base",
                low_value=low_val,
                high_value=high_val,
                low_gas_rate=rate_low,
                high_gas_rate=rate_high,
                rate_swing=swing,
            )
        )

    # Sort descending by impact
    items.sort(key=lambda it: it.rate_swing, reverse=True)
    return TornadoResult(case_name=case.name, base_gas_rate=base_rate, items=items)
