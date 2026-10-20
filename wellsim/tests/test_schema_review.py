"""Tests for the six schema review points (datum, psig, density kinds, bit injection,
spec section 3 inputs, extra=forbid)."""

import inspect

import pytest
from pydantic import BaseModel, ValidationError

import wellsim.cases.schema as schema_mod
from wellsim.cases.schema import MAX_KILL_FLUID_PPG, MIN_KILL_FLUID_PPG, Case, Exit, Kill, Reservoir
from wellsim.core import units


# 1. reservoir pressure datum ----------------------------------------------------

def test_datum_tvd_defaults_to_depth_md_for_vertical_well(gas_case_dict):
    case = Case.model_validate(gas_case_dict)
    assert case.reservoir.datum_tvd == pytest.approx(10000.0)


def test_datum_tvd_comes_from_survey_when_deviated(oil_case_dict):
    oil_case_dict["well"]["survey"] = [
        {"md": 0, "tvd": 0},
        {"md": 4000, "tvd": 4000},
        {"md": 9200, "tvd": 8000},
    ]
    case = Case.model_validate(oil_case_dict)
    # depth_md 9000 sits 5000/5200 of the way through the 4000 ft tvd step
    assert case.reservoir.datum_tvd == pytest.approx(4000 + 4000 * 5000 / 5200)
    assert case.well.md_to_tvd(9200) == pytest.approx(8000)


def test_explicit_datum_tvd_is_kept(oil_case_dict):
    oil_case_dict["reservoir"]["datum_tvd"] = "2700 m"
    case = Case.model_validate(oil_case_dict)
    assert case.reservoir.datum_tvd == pytest.approx(2700 / 0.3048)


def test_datum_tvd_below_well_rejected(oil_case_dict):
    oil_case_dict["reservoir"]["datum_tvd"] = 9500
    with pytest.raises(ValidationError, match="datum_tvd"):
        Case.model_validate(oil_case_dict)


def test_datum_tvd_survives_dump_roundtrip(oil_case_dict):
    case = Case.model_validate(oil_case_dict)
    again = Case.model_validate(case.model_dump(mode="json"))
    assert again.reservoir.datum_tvd == case.reservoir.datum_tvd


# 2. psig input ------------------------------------------------------------------

def test_psig_converts_with_14_7_offset(gas_case_dict):
    gas_case_dict["reservoir"]["pressure"] = "5000 psig"
    case = Case.model_validate(gas_case_dict)
    assert case.reservoir.pressure == pytest.approx(5014.7)
    assert Case.model_validate({**gas_case_dict, "exit": {"ambient_pressure": "0 psig"}}).exit.ambient_pressure == 14.7


def test_bare_pressure_is_documented_as_psia():
    assert units.canonical_unit("pressure") == "psia"
    field = Reservoir.model_fields["pressure"]
    assert "psia" in field.description
    props = Case.model_json_schema()["$defs"]["Reservoir"]["properties"]
    assert props["pressure"]["unit"] == "psia"
    assert Case.model_json_schema()["$defs"]["Exit"]["properties"]["ambient_pressure"]["unit"] == "psia"


# 3. density vs mud weight -------------------------------------------------------

def test_density_and_mud_weight_are_distinct_kinds():
    assert units.canonical_unit("density") == "lbm/ft3"
    assert units.canonical_unit("mud_weight") == "ppg"
    with pytest.raises(units.UnitError, match="ambiguous"):
        units.convert(12.5, "ppg", "lbm/ft3")


def test_kill_fluid_density_with_units_converts(oil_case_dict):
    for text in ("12.5 ppg", "93.5065 lbm/ft3", "1.4977 sg", "1497.7 kg/m3"):
        oil_case_dict["kill"]["kill_fluid_density"] = text
        case = Case.model_validate(oil_case_dict)
        assert case.kill.kill_fluid_density == pytest.approx(12.5, rel=1e-4), text


@pytest.mark.parametrize("bad", [93.5, 1.5, 0.0, MIN_KILL_FLUID_PPG - 0.01, MAX_KILL_FLUID_PPG + 0.01])
def test_bare_kill_fluid_density_outside_ppg_range_rejected(oil_case_dict, bad):
    oil_case_dict["kill"]["kill_fluid_density"] = bad
    with pytest.raises(ValidationError, match="kill_fluid_density"):
        Case.model_validate(oil_case_dict)


def test_schema_has_no_lbm_ft3_density_fields():
    """The case schema carries exactly one density input and it is ppg."""
    for name, obj in inspect.getmembers(schema_mod, inspect.isclass):
        if not (issubclass(obj, BaseModel) and obj.__module__ == schema_mod.__name__):
            continue
        for fname, f in obj.model_fields.items():
            extra = getattr(f, "json_schema_extra", None) or {}
            kind = extra.get("kind") if isinstance(extra, dict) else None
            if kind is None:
                for meta in f.metadata:
                    ex = getattr(meta, "json_schema_extra", None)
                    if isinstance(ex, dict) and "kind" in ex:
                        kind = ex["kind"]
            assert kind != "density", f"{name}.{fname} uses lbm/ft3 density in the case schema"


# 4. drillstring injection at the bit -------------------------------------------

def test_drillstring_injection_may_equal_string_bottom(oil_case_dict):
    bottom = oil_case_dict["well"]["string"][-1]["bottom_md"]
    oil_case_dict["kill"]["injection_depth_md"] = bottom
    assert Case.model_validate(oil_case_dict).kill.injection_depth_md == bottom
    oil_case_dict["kill"]["injection_depth_md"] = bottom - 500
    Case.model_validate(oil_case_dict)
    oil_case_dict["kill"]["injection_depth_md"] = bottom + 1
    with pytest.raises(ValidationError, match="below the string bottom"):
        Case.model_validate(oil_case_dict)


# 5. spec section 3 inputs -------------------------------------------------------

def test_section_3_inputs_exist():
    assert Exit.model_fields["ambient_pressure"].default == 14.7
    assert "exit_id" in Exit.model_fields
    assert "time_since_start" in Reservoir.model_fields
    assert "exposed_interval" in Reservoir.model_fields
    for cls in (schema_mod.HoleSection, schema_mod.StringSection, schema_mod.InjectionSection):
        assert "roughness" in cls.model_fields, cls.__name__
    assert "pump_pressure_limit" in Kill.model_fields
    assert "available_hhp" in Kill.model_fields


# 6. unknown fields fail loudly ---------------------------------------------------

def _schema_models():
    return [
        obj for _, obj in inspect.getmembers(schema_mod, inspect.isclass)
        if issubclass(obj, BaseModel) and obj.__module__ == schema_mod.__name__
    ]


def test_every_model_forbids_extra_fields():
    models = _schema_models()
    assert len(models) >= 17
    for m in models:
        assert m.model_config.get("extra") == "forbid", m.__name__


@pytest.mark.parametrize(
    "path, key",
    [
        ((), "flowpath"),
        (("reservoir",), "presure"),
        (("fluid",), "api"),
        (("well",), "casing"),
        (("well", "hole", 0), "ID"),
        (("well", "string", 0), "roughnes"),
        (("exit",), "pressure"),
        (("kill",), "density"),
        (("kill", "injection_path"), "sectons"),
        (("kill", "rheology"), "pv"),
        (("options", "correlations"), "vlp_correlation"),
        (("options", "numerics"), "tolerance"),
    ],
)
def test_typo_anywhere_in_yaml_is_rejected(oil_case_dict, path, key):
    oil_case_dict["exit"] = {}
    oil_case_dict["kill"]["rheology"] = {"type": "newtonian"}
    oil_case_dict["options"] = {"correlations": {}, "numerics": {}}
    node = oil_case_dict
    for p in path:
        node = node[p]
    node[key] = 1
    with pytest.raises(ValidationError, match="[Ee]xtra"):
        Case.model_validate(oil_case_dict)
