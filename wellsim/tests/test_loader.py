import pytest

from wellsim.cases import Case, case_from_dict, load_case, save_case
from wellsim.cases.schema import BlackOil, DryGas


def test_load_dry_gas_example(examples_dir):
    case = load_case(examples_dir / "dry_gas_vertical.yaml")
    assert isinstance(case.fluid, DryGas)
    assert case.flow_path == "no_string"
    assert case.reservoir.pressure == pytest.approx(380 * 14.503773773)
    assert case.reservoir.temperature == pytest.approx(248.0)
    assert case.reservoir.depth_md == pytest.approx(3500 / 0.3048)
    assert case.fluid.co2 == pytest.approx(0.02)
    assert case.well.hole[1].kind == "open_hole"


def test_load_black_oil_example(examples_dir):
    case = load_case(examples_dir / "black_oil_annulus_kill.yaml")
    assert isinstance(case.fluid, BlackOil)
    assert case.flow_path == "annulus"
    assert case.kill is not None
    assert case.kill.kill_fluid_density == 12.5
    assert case.kill.injection_path.type == "relief_well"
    assert case.temperature.exit_temperature == 180.0
    assert case.options.correlations.vlp == ["hagedorn_brown", "beggs_brill"]
    assert case.well.survey[-1].tvd == 9600.0


def test_save_and_reload(tmp_path, examples_dir):
    case = load_case(examples_dir / "black_oil_annulus_kill.yaml")
    out = tmp_path / "case.yaml"
    save_case(case, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# wellsim case, canonical")
    assert "12.5" in text
    reloaded = load_case(out)
    assert reloaded == case


def test_case_from_dict_matches_model_validate(gas_case_dict):
    assert case_from_dict(gas_case_dict) == Case.model_validate(gas_case_dict)


def test_load_rejects_non_mapping(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_case(p)
