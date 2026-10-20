"""Case definition: pydantic schema, unit-aware fields, YAML loader."""

from wellsim.cases.loader import case_from_dict, load_case, save_case
from wellsim.cases.schema import Case

__all__ = ["Case", "case_from_dict", "load_case", "save_case"]
