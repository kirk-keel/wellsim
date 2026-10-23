"""Command Line Interface (CLI) for wellsim.

Enables headless execution for engineering pipelines, automated batch studies,
case syntax linting, and web simulator launcher.

Usage:
    wellsim run case.yaml [--json]
    wellsim check case.yaml
    wellsim sweep case.yaml parameter val1 val2 ...
    wellsim ui [--port 8080]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List

from wellsim.batch.sweep import sweep_parameter
from wellsim.cases.loader import load_case
from wellsim.core.solver import solve

__all__ = ["main"]


def cmd_run(args: argparse.Namespace) -> int:
    """Solve steady-state blowout rate and print engineering report."""
    try:
        case = load_case(args.case_path)
    except Exception as e:
        print(f"Error loading case {args.case_path}: {e}", file=sys.stderr)
        return 1

    try:
        res = solve(case)
    except Exception as e:
        print(f"Solver failed: {e}", file=sys.stderr)
        return 1

    if args.json:
        data = {
            "name": case.name,
            "flowing": res.flowing,
            "gas_rate_mscfd": res.gas_rate,
            "oil_rate_bopd": res.oil_rate if hasattr(res, "oil_rate") else 0.0,
            "pwf_psia": res.pwf,
            "exit_pressure_psia": res.exit.p if res.exit else None,
            "exit_choked": res.exit.choked if res.exit else False,
            "choke_md_ft": res.choke_md,
            "warnings": res.warnings,
        }
        print(json.dumps(data, indent=2))
        return 0

    print("=" * 70)
    print(f" WELLSIM STEADY-STATE BLOWOUT SOLVER // CASE: {case.name}")
    print("=" * 70)
    print(f" Flowing:              {'YES' if res.flowing else 'NO'}")
    if res.fluid_type == "black_oil":
        print(f" Oil Rate:             {res.oil_rate:,.1f} bopd")
        print(f" Associated Gas Rate:  {res.gas_rate:,.1f} Mscf/d")
    else:
        print(f" Gas Rate:             {res.gas_rate:,.1f} Mscf/d")
    print(f" Flowing BHP (Pwf):    {res.pwf:,.1f} psia")
    if res.exit:
        print(f" Exit Pressure:        {res.exit.p:,.1f} psia ({'CHOKED MACH 1.0' if res.exit.choked else 'Subsonic'})")
        print(f" Exit Temperature:     {res.exit.t:,.1f} degF")
    if res.choke_md:
        chokes_str = ", ".join(f"{d:,.0f} ft" for d in res.choke_md)
        print(f" Internal Choke Depth: {chokes_str} MD")
    print("-" * 70)
    if res.warnings:
        print(f" Engineering Warnings ({len(res.warnings)}):")
        for w in res.warnings[:5]:
            print(f"   • {w}")
    print("=" * 70)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Validate case schema and well geometry connectivity."""
    try:
        case = load_case(args.case_path)
        print(f"✓ Case '{case.name}' is valid.")
        print(f"  Fluid: {case.fluid.type}")
        print(f"  Reservoir: {case.reservoir.pressure} psia at {case.reservoir.depth_md} ft MD")
        print(f"  Well Total Depth: {case.well.total_depth_md} ft MD ({len(case.well.hole)} conduit sections)")
        return 0
    except Exception as e:
        print(f"✗ Validation error in {args.case_path}:\n  {e}", file=sys.stderr)
        return 1


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run parametric sweep across values and print Markdown table."""
    try:
        case = load_case(args.case_path)
        values = [float(v) for v in args.values]
    except Exception as e:
        print(f"Input error: {e}", file=sys.stderr)
        return 1

    try:
        res = sweep_parameter(case, args.param_path, values)
        print(res.to_markdown())
        return 0
    except Exception as e:
        print(f"Sweep execution failed: {e}", file=sys.stderr)
        return 1


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch interactive browser simulator."""
    from wellsim.ui.server import run_server
    run_server(port=args.port)
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wellsim", description="Wild Well Engineering & Drilling Simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="Solve steady-state blowout rate")
    p_run.add_argument("case_path", help="Path to YAML or JSON case file")
    p_run.add_argument("--json", action="store_true", help="Output JSON instead of table")

    # check
    p_check = subparsers.add_parser("check", help="Validate case schema and parameters")
    p_check.add_argument("case_path", help="Path to YAML or JSON case file")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Run batch parameter sweep")
    p_sweep.add_argument("case_path", help="Path to YAML or JSON case file")
    p_sweep.add_argument("param_path", help="Dotted field path (e.g. reservoir.permeability)")
    p_sweep.add_argument("values", nargs="+", help="Numeric values to sweep")

    # ui
    p_ui = subparsers.add_parser("ui", help="Launch interactive web console")
    p_ui.add_argument("--port", type=int, default=8080, help="Web server port (default 8080)")

    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "check":
        return cmd_check(args)
    elif args.command == "sweep":
        return cmd_sweep(args)
    elif args.command == "ui":
        return cmd_ui(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
