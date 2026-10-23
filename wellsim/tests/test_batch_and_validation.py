"""Unit tests for Macondo validation case and batch parameter sweeps (Phase D)."""

import pytest

from wellsim.batch.sweep import sweep_parameter, tornado_sensitivity
from wellsim.cases.loader import load_case
from wellsim.core.solver import solve


def test_macondo_blowout_validation_case():
    case = load_case("wellsim/validation/cases/tier3_macondo/inputs.yaml")
    res = solve(case)

    assert res.flowing is True
    assert res.fluid_type == "black_oil"
    # Macondo consensus FRTG flow rate range: 50,000 to 70,000 bopd
    # Check that the model predicts within the historical ballpark
    assert 45000.0 <= res.oil_rate <= 80000.0
    assert res.exit is not None
    assert res.exit.p >= 2230.0  # At or above 5000 ft subsea mudline pressure


def test_batch_parameter_sweep():
    case = load_case("wellsim/cases/examples/dry_gas_vertical.yaml")

    # Sweep permeability across 3 values
    k_values = [50.0, 100.0, 200.0]
    sweep_res = sweep_parameter(case, "reservoir.permeability", k_values)

    assert len(sweep_res.rows) == 3
    # Gas rate must increase with higher permeability
    rates = [r.gas_rate_mscfd for r in sweep_res.rows]
    assert rates[0] < rates[1] < rates[2]

    # Markdown export test
    md = sweep_res.to_markdown()
    assert "Parameter Sweep" in md
    assert "reservoir.permeability" in md


def test_tornado_sensitivity():
    case = load_case("wellsim/cases/examples/dry_gas_vertical.yaml")

    tornado_res = tornado_sensitivity(
        case,
        {
            "reservoir.permeability": (50.0, 200.0),
            "reservoir.skin": (0.0, 5.0),
        },
    )

    assert len(tornado_res.items) == 2
    # Verify items are sorted descending by rate swing
    assert tornado_res.items[0].rate_swing >= tornado_res.items[1].rate_swing
    md = tornado_res.to_markdown()
    assert "Tornado Sensitivity Analysis" in md
