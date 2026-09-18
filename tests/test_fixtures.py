"""Validate IR nodes against external JSON fixtures.

Fixtures live under tests/fixtures/<node>/{valid,invalid}/*.json.
Each file is auto-discovered and turned into a parametrized test case,
so adding a new JSON file is enough to extend coverage.

Valid fixtures must parse and round-trip through model_dump_json.
Invalid fixtures must raise ValidationError.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from edge_ir.ir.base import IRBase
from edge_ir.ir.tensor import (
    BuiltinDataType,
    RankDeclaration,
    TensorDeclaration,
    UserDefinedDataType,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"

NODE_TYPES: dict[str, type[IRBase]] = {
    "builtin_data_type": BuiltinDataType,
    "user_defined_data_type": UserDefinedDataType,
    "rank_declaration": RankDeclaration,
    "tensor_declaration": TensorDeclaration,
}


def _collect(category: str) -> list[tuple[type[IRBase], Path]]:
    cases: list[tuple[type[IRBase], Path]] = []
    for dir_name, cls in NODE_TYPES.items():
        for path in sorted((FIXTURES_DIR / dir_name / category).glob("*.json")):
            cases.append((cls, path))
    return cases


def _case_id(case: tuple[type[IRBase], Path]) -> str:
    cls, path = case
    return f"{cls.__name__}-{path.stem}"


@pytest.mark.parametrize("case", _collect("valid"), ids=_case_id)
def test_valid_fixture_parses_and_round_trips(
    case: tuple[type[IRBase], Path],
) -> None:
    cls, path = case
    raw = path.read_text()
    instance = cls.model_validate_json(raw)
    # round-trip: serialize and re-parse, must compare equal
    re_parsed = cls.model_validate_json(instance.model_dump_json())
    assert re_parsed == instance


@pytest.mark.parametrize("case", _collect("invalid"), ids=_case_id)
def test_invalid_fixture_raises(case: tuple[type[IRBase], Path]) -> None:
    cls, path = case
    raw = path.read_text()
    with pytest.raises(ValidationError):
        cls.model_validate_json(raw)
