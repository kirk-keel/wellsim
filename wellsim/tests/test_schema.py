import pytest
from pydantic import ValidationError

from wellsim.cases.schema import (
    OPEN_HOLE_ROUGHNESS_IN,
    STEEL_ROUGHNESS_IN,
    BlackOil,
    Case,
    DryGas,
    Reservoir,
    Well,
)


def test_minimal_gas_case_validates(gas_case_dict):
    case = Case.model_validate(gas_case_dict)
    assert isinstance(case.fluid, DryGas)
    assert case.exit.ambient_pressure == 14.7
    assert case.exit.exit_id is None
    assert case.kill is None
    assert case.options.correlations.gas_z == "dak"
    assert case.well.hole[0].roughness == STEEL_ROUGHNESS_IN


def test_minimal_oil_case_with_kill(oil_case_dict):
    case = Case.model_validate(oil_case_dict)
    assert isinstance(case.fluid, BlackOil)
    assert case.fluid.bubble_point is None
    assert case.kill.rheology.type == "newtonian"
    assert case.kill.injection_path.sections == []
    assert case.well.string_bottom_md == 9000.0


def test_units_convert_at_boundary(gas_case_dict):
    gas_case_dict["reservoir"]["pressure"] = "300 bar"
    gas_case_dict["reservoir"]["temperature"] = {"value": 100, "unit": "degC"}
    gas_case_dict["reservoir"]["depth_md"] = "3048 m"
    gas_case_dict["well"]["total_depth_md"] = "3048 m"
    gas_case_dict["well"]["hole"][0]["bottom_md"] = "3048 m"
    gas_case_dict["well"]["hole"][0]["id"] = "157.07 mm"
    gas_case_dict["fluid"]["co2"] = "2 %"
    gas_case_dict["exit"] = {"ambient_pressure": "1 atm"}
    case = Case.model_validate(gas_case_dict)
    assert case.reservoir.pressure == pytest.approx(4351.13, rel=1e-5)
    assert case.reservoir.temperature == pytest.approx(212.0)
    assert case.reservoir.depth_md == pytest.approx(10000.0)
    assert case.well.hole[0].id == pytest.approx(6.184, rel=1e-4)
    assert case.fluid.co2 == pytest.approx(0.02)
    assert case.exit.ambient_pressure == pytest.approx(14.69595)


def test_wrong_kind_unit_rejected(gas_case_dict):
    gas_case_dict["reservoir"]["pressure"] = "300 degC"
    with pytest.raises(ValidationError, match="not valid for pressure"):
        Case.model_validate(gas_case_dict)


def test_extra_fields_forbidden(gas_case_dict):
    gas_case_dict["reservoir"]["presure"] = 5000
    with pytest.raises(ValidationError, match="extra"):
        Case.model_validate(gas_case_dict)


def test_fluid_discriminator(gas_case_dict):
    gas_case_dict["fluid"] = {"type": "black_oil", "oil_api": 30, "gas_sg": 0.7, "gor": 400}
    assert isinstance(Case.model_validate(gas_case_dict).fluid, BlackOil)
    gas_case_dict["fluid"] = {"type": "compositional"}
    with pytest.raises(ValidationError):
        Case.model_validate(gas_case_dict)


def test_wet_gas_rejected():
    with pytest.raises(ValidationError, match="wet gas"):
        DryGas(gas_sg=0.7, condensate_yield=5.0)
    with pytest.raises(ValidationError, match="less than 1"):
        DryGas(gas_sg=0.7, n2=0.5, co2=0.5)


def test_fraction_bounds(gas_case_dict):
    gas_case_dict["fluid"]["h2s"] = 1.5
    with pytest.raises(ValidationError):
        Case.model_validate(gas_case_dict)


def test_reservoir_checks():
    base = dict(pressure=5000, temperature=200, depth_md=10000, net_pay=50, permeability=100,
                wellbore_radius=0.354, drainage_radius=1500)
    Reservoir(**base)
    with pytest.raises(ValidationError, match="wellbore_radius"):
        Reservoir(**{**base, "wellbore_radius": 2000})
    with pytest.raises(ValidationError, match="exposed_interval"):
        Reservoir(**{**base, "exposed_interval": 80})
    with pytest.raises(ValidationError, match="transient"):
        Reservoir(**{**base, "time_since_start": 2.0})
    r = Reservoir(**{**base, "time_since_start": "48 hr", "porosity": 0.2, "total_compressibility": 1e-5})
    assert r.time_since_start == pytest.approx(2.0)


def test_flow_path_needs_matching_string(gas_case_dict, oil_case_dict):
    gas_case_dict["flow_path"] = "annulus"
    with pytest.raises(ValidationError, match="needs well.string"):
        Case.model_validate(gas_case_dict)
    oil_case_dict["flow_path"] = "no_string"
    with pytest.raises(ValidationError, match="well.string is set"):
        Case.model_validate(oil_case_dict)
    oil_case_dict["flow_path"] = "tubing"
    Case.model_validate(oil_case_dict)


def test_hole_sections_must_tile_to_td():
    well = dict(total_depth_md=10000, hole=[{"top_md": 0, "bottom_md": 9000, "id": 8.5}])
    with pytest.raises(ValidationError, match="end at total_depth_md"):
        Well(**well)
    well["hole"] = [{"top_md": 0, "bottom_md": 5000, "id": 8.5}, {"top_md": 5500, "bottom_md": 10000, "id": 6.0}]
    with pytest.raises(ValidationError, match="contiguous"):
        Well(**well)
    well["hole"] = [{"top_md": 0, "bottom_md": 5000, "id": 8.5},
                    {"top_md": 5000, "bottom_md": 10000, "id": 6.0, "kind": "open_hole"}]
    w = Well(**well)
    assert w.hole[1].roughness == OPEN_HOLE_ROUGHNESS_IN


def test_string_must_fit_inside_hole():
    well = dict(total_depth_md=10000, hole=[{"top_md": 0, "bottom_md": 10000, "id": 6.184}],
                string=[{"top_md": 0, "bottom_md": 9000, "od": 6.5, "id": 4.0}])
    with pytest.raises(ValidationError, match="not smaller than hole id"):
        Well(**well)
    well["string"] = [{"top_md": 0, "bottom_md": 10500, "od": 5.0, "id": 4.0}]
    with pytest.raises(ValidationError, match="below total_depth_md"):
        Well(**well)
    well["string"] = []
    with pytest.raises(ValidationError, match="empty"):
        Well(**well)


def test_survey_checks():
    hole = [{"top_md": 0, "bottom_md": 10000, "id": 8.5}]
    with pytest.raises(ValidationError, match="start at md = tvd = 0"):
        Well(total_depth_md=10000, hole=hole, survey=[{"md": 100, "tvd": 100}, {"md": 10000, "tvd": 9000}])
    with pytest.raises(ValidationError, match="increase monotonically"):
        Well(total_depth_md=10000, hole=hole,
             survey=[{"md": 0, "tvd": 0}, {"md": 5000, "tvd": 5000}, {"md": 4000, "tvd": 4000}])
    with pytest.raises(ValidationError, match="cannot exceed md increment"):
        Well(total_depth_md=10000, hole=hole, survey=[{"md": 0, "tvd": 0}, {"md": 10000, "tvd": 10001}])
    with pytest.raises(ValidationError, match="extend to total_depth_md"):
        Well(total_depth_md=10000, hole=hole, survey=[{"md": 0, "tvd": 0}, {"md": 9000, "tvd": 9000}])
    Well(total_depth_md=10000, hole=hole,
         survey=[{"md": 0, "tvd": 0}, {"md": 6000, "tvd": 5500}, {"md": 10000, "tvd": 5500}])


def test_reservoir_and_kill_depths_within_well(oil_case_dict):
    oil_case_dict["reservoir"]["depth_md"] = 9500
    with pytest.raises(ValidationError, match="reservoir depth_md"):
        Case.model_validate(oil_case_dict)
    oil_case_dict["reservoir"]["depth_md"] = 9000
    oil_case_dict["kill"]["injection_depth_md"] = 9100
    with pytest.raises(ValidationError, match="below the string bottom"):
        Case.model_validate(oil_case_dict)


def test_relief_well_path_needs_sections(oil_case_dict):
    oil_case_dict["kill"]["injection_path"] = {"type": "relief_well"}
    with pytest.raises(ValidationError, match="at least one section"):
        Case.model_validate(oil_case_dict)
    oil_case_dict["kill"]["injection_path"] = {
        "type": "relief_well",
        "sections": [{"length": "2800 m", "id": "4.276 in"}, {"length": 300, "id": 8.535, "od_inner": 5.0}],
    }
    case = Case.model_validate(oil_case_dict)
    assert case.kill.injection_path.sections[0].length == pytest.approx(9186.35, rel=1e-5)
    assert case.kill.kill_fluid_density == 12.0


def test_dump_roundtrip_is_stable(oil_case_dict):
    case = Case.model_validate(oil_case_dict)
    dumped = case.model_dump(mode="json")
    again = Case.model_validate(dumped)
    assert again == case
    assert again.model_dump(mode="json") == dumped


def test_json_schema_carries_units():
    schema = Case.model_json_schema()
    res = schema["$defs"]["Reservoir"]["properties"]
    assert res["pressure"]["unit"] == "psia"
    assert res["depth_md"]["unit"] == "ft"
    assert schema["$defs"]["Kill"]["properties"]["kill_fluid_density"]["unit"] == "ppg"
