"""Unit tests for the wellsim command-line interface (CLI)."""

import pytest

from wellsim.cli import main


def test_cli_check_command():
    ret = main(["check", "wellsim/cases/examples/aliso_canyon_ss25.yaml"])
    assert ret == 0


def test_cli_run_command_table():
    ret = main(["run", "wellsim/cases/examples/aliso_canyon_ss25.yaml"])
    assert ret == 0


def test_cli_run_command_json(capsys):
    ret = main(["run", "wellsim/cases/examples/aliso_canyon_ss25.yaml", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    assert '"flowing": true' in captured.out
    assert '"gas_rate_mscfd"' in captured.out


def test_cli_sweep_command():
    ret = main([
        "sweep",
        "wellsim/cases/examples/dry_gas_vertical.yaml",
        "reservoir.permeability",
        "50.0",
        "100.0",
    ])
    assert ret == 0
