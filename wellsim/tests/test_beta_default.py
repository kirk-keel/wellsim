"""M2.5-5: default non-Darcy beta rule (owner decision, 21 Sep 2026). COMPUTED."""

from __future__ import annotations

import copy
import warnings

from wellsim.cases.loader import case_from_dict
from wellsim.core.ipr.gas import beta_geertsma, beta_jones
from wellsim.core.pvt.base import RangeWarning
from wellsim.core.solver import solve


def _solve(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RangeWarning)
        return solve(case_from_dict(d))


def test_auto_uses_jones_without_porosity(gas_case_dict):
    assert gas_case_dict["reservoir"].get("porosity") is None
    r = _solve(gas_case_dict)
    assert r.correlations["non_darcy_beta"] == "jones"
    assert r.ipr["beta_1_per_ft"] == beta_jones(gas_case_dict["reservoir"]["permeability"])
    note = [n for n in r.notes if n.startswith("non-Darcy beta")]
    assert len(note) == 1 and "jones" in note[0] and "only permeability supplied" in note[0]


def test_auto_uses_geertsma_with_porosity(gas_case_dict):
    d = copy.deepcopy(gas_case_dict)
    d["reservoir"]["porosity"] = 0.18
    r = _solve(d)
    assert r.correlations["non_darcy_beta"] == "geertsma"
    assert r.ipr["beta_1_per_ft"] == beta_geertsma(d["reservoir"]["permeability"], 0.18)
    note = [n for n in r.notes if n.startswith("non-Darcy beta")]
    assert len(note) == 1 and "geertsma" in note[0] and "porosity supplied" in note[0]


def test_explicit_choice_and_user_beta_override_auto(gas_case_dict):
    d = copy.deepcopy(gas_case_dict)
    d["reservoir"]["porosity"] = 0.18
    d["options"] = {"correlations": {"non_darcy_beta": "jones"}}
    r = _solve(d)
    assert r.correlations["non_darcy_beta"] == "jones"
    assert any("requested in options" in n for n in r.notes if n.startswith("non-Darcy beta"))
    d2 = copy.deepcopy(gas_case_dict)
    d2["reservoir"]["non_darcy_beta"] = 5e8
    r2 = _solve(d2)
    assert r2.correlations["non_darcy_beta"] == "user"
    assert any(n.startswith("non-Darcy beta: user value") for n in r2.notes)
