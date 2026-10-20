"""Shared fixtures: minimal valid case dicts that tests mutate."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "cases" / "examples"


def _gas_case() -> dict:
    return {
        "name": "gas_min",
        "reservoir": {
            "pressure": 5000.0,
            "temperature": 200.0,
            "depth_md": 10000.0,
            "net_pay": 50.0,
            "permeability": 100.0,
            "wellbore_radius": 0.354,
            "drainage_radius": 1500.0,
        },
        "fluid": {"type": "dry_gas", "gas_sg": 0.65},
        "well": {
            "total_depth_md": 10000.0,
            "hole": [{"top_md": 0.0, "bottom_md": 10000.0, "id": 6.184}],
        },
        "flow_path": "no_string",
    }


def _oil_case() -> dict:
    return {
        "name": "oil_min",
        "reservoir": {
            "pressure": 4000.0,
            "temperature": 180.0,
            "depth_md": 9000.0,
            "net_pay": 60.0,
            "permeability": 300.0,
            "wellbore_radius": 0.354,
            "drainage_radius": 1500.0,
        },
        "fluid": {"type": "black_oil", "oil_api": 35.0, "gas_sg": 0.7, "gor": 500.0},
        "well": {
            "total_depth_md": 9200.0,
            "hole": [{"top_md": 0.0, "bottom_md": 9200.0, "id": 8.535}],
            "string": [{"top_md": 0.0, "bottom_md": 9000.0, "od": 5.0, "id": 4.276}],
        },
        "flow_path": "annulus",
        "kill": {
            "injection_depth_md": 9000.0,
            "kill_fluid_density": 12.0,
            "injection_path": {"type": "drillstring"},
            "pump_pressure_limit": 5000.0,
            "available_hhp": 3000.0,
        },
    }


@pytest.fixture
def gas_case_dict() -> dict:
    return copy.deepcopy(_gas_case())


@pytest.fixture
def oil_case_dict() -> dict:
    return copy.deepcopy(_oil_case())


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES_DIR
