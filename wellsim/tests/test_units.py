import numpy as np
import pytest

from wellsim.core import units
from wellsim.core.units import (
    UnitError,
    canonical_unit,
    coerce,
    convert,
    from_internal,
    parse_quantity,
    to_internal,
)


@pytest.mark.parametrize(
    "value, unit, kind, expected",
    [
        (1.0, "bar", "pressure", 14.503773773),
        (0.0, "psig", "pressure", 14.7),
        (5000.0, "psig", "pressure", 5014.7),
        (0.0, "barg", "pressure", 14.7),
        (1.0, "atm", "pressure", 14.695950),
        (100.0, "degC", "temperature", 212.0),
        (273.15, "K", "temperature", 32.0),
        (671.67, "degR", "temperature", 212.0),
        (1.0, "m", "length", 3.280839895),
        (25.4, "mm", "diameter", 1.0),
        (1.0, "bpm", "liquid_rate", 1440.0),
        (42.0, "gpm", "liquid_rate", 1440.0),
        (1.0, "m3/d", "liquid_rate", 6.289810770),
        (1.0, "MMscf/d", "gas_rate", 1000.0),
        (1000.0, "Sm3/d", "gas_rate", 35.31466672),
        (1.0, "Sm3/Sm3", "gor", 5.614583333),
        (10.0, "ppg", "density", 74.80519480),
        (1000.0, "kg/m3", "density", 62.427960576),
        (1.0, "sg", "mud_weight", 8.345404),
        (1.0, "Pa.s", "viscosity", 1000.0),
        (1.0, "darcy", "permeability", 1000.0),
        (1.0, "1/bar", "compressibility", 1.0 / 14.503773773),
        (3.5, "wt%", "salinity", 35000.0),
        (24.0, "hr", "time", 1.0),
        (1.0, "kW", "power", 1.341022090),
        (50.0, "%", "fraction", 0.5),
    ],
)
def test_to_internal(value, unit, kind, expected):
    assert to_internal(value, unit, kind) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("kind", list(units.KINDS))
def test_canonical_is_identity(kind):
    assert to_internal(123.4, canonical_unit(kind), kind) == 123.4
    assert from_internal(123.4, canonical_unit(kind), kind) == 123.4


@pytest.mark.parametrize("kind", list(units.KINDS))
def test_roundtrip_every_unit(kind):
    for u in units.units_for(kind):
        x = 37.25
        assert from_internal(to_internal(x, u, kind), u, kind) == pytest.approx(x, rel=1e-12, abs=1e-9)


def test_convert_infers_kind():
    assert convert(1.0, "bar", "psia") == pytest.approx(14.503773773)
    assert convert(212.0, "degF", "degC") == pytest.approx(100.0)
    assert convert(1000.0, "psig", "psia") == pytest.approx(1014.7)


def test_convert_ambiguous_needs_kind():
    with pytest.raises(UnitError, match="ambiguous"):
        convert(1.0, "ppg", "kg/m3")
    assert convert(1.0, "ppg", "kg/m3", kind="mud_weight") == pytest.approx(119.826, rel=1e-4)


def test_convert_unknown_pair():
    with pytest.raises(UnitError):
        convert(1.0, "psia", "ft")
    with pytest.raises(UnitError):
        to_internal(1.0, "furlong", "length")
    with pytest.raises(UnitError):
        to_internal(1.0, "ft", "no_such_kind")


def test_unit_spelling_is_forgiving():
    assert to_internal(100.0, "°C", "temperature") == pytest.approx(212.0)
    assert to_internal(100.0, " Deg C ", "temperature") == pytest.approx(212.0)
    assert to_internal(1.0, "m^3/d", "liquid_rate") == pytest.approx(6.28981077)
    assert to_internal(1.0, "Sm³/d", "gas_rate") == pytest.approx(0.03531466672)


def test_numpy_arrays():
    p = np.array([1.0, 2.0, 3.0])
    out = to_internal(p, "bar", "pressure")
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, p * 14.503773773)
    np.testing.assert_allclose(convert(np.array([0.0, 100.0]), "degC", "degF"), [32.0, 212.0])


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3500 psig", (3500.0, "psig")),
        ("3500psig", (3500.0, "psig")),
        ("  -12.5e2 degC ", (-1250.0, "degC")),
        ("42", (42.0, None)),
        ("1e3 Sm3/d", (1000.0, "Sm3/d")),
        (".5 %", (0.5, "%")),
    ],
)
def test_parse_quantity(text, expected):
    assert parse_quantity(text) == expected


def test_parse_quantity_rejects_garbage():
    with pytest.raises(UnitError):
        parse_quantity("psia 3500")
    with pytest.raises(UnitError):
        parse_quantity("")


def test_coerce_forms():
    assert coerce(3500, "pressure") == 3500.0
    assert coerce("3500 psig", "pressure") == pytest.approx(3514.7)
    assert coerce("3500", "pressure") == 3500.0
    assert coerce({"value": 250, "unit": "bar"}, "pressure") == pytest.approx(3625.943443)
    assert coerce({"value": "250 bar"}, "pressure") == pytest.approx(3625.943443)
    assert coerce((250, "bar"), "pressure") == pytest.approx(3625.943443)
    assert coerce(np.float64(7.0), "length") == 7.0


def test_coerce_rejects():
    with pytest.raises(UnitError):
        coerce(True, "pressure")
    with pytest.raises(UnitError):
        coerce("100 degC", "pressure")
    with pytest.raises(UnitError):
        coerce({"unit": "bar"}, "pressure")
    with pytest.raises(UnitError):
        coerce(object(), "pressure")


def test_canonicals_match_docstring():
    expected = {
        "pressure": "psia", "dpressure": "psi", "temperature": "degf", "length": "ft",
        "diameter": "in", "liquid_rate": "stb/d", "gas_rate": "mscf/d", "gor": "scf/stb",
        "density": "lbm/ft3", "mud_weight": "ppg", "viscosity": "cp", "permeability": "md",
        "compressibility": "1/psi", "salinity": "ppm", "time": "day", "power": "hp",
        "velocity": "ft/s", "volume": "bbl", "fraction": "frac",
    }
    assert {k: canonical_unit(k) for k in units.KINDS} == expected
