"""YAML I/O for cases. Load converts units at the boundary; save writes canonical units."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from wellsim.cases.schema import Case


def case_from_dict(data: Mapping[str, Any]) -> Case:
    return Case.model_validate(dict(data))


def load_case(path: str | Path) -> Case:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top level must be a mapping")
    return case_from_dict(data)


def case_to_dict(case: Case) -> dict[str, Any]:
    """Plain dict, canonical units, JSON-compatible types."""
    return case.model_dump(mode="json")


def save_case(case: Case, path: str | Path) -> None:
    path = Path(path)
    header = "# wellsim case, canonical oilfield units (see wellsim.core.units)\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(case_to_dict(case), f, sort_keys=False, default_flow_style=False)
