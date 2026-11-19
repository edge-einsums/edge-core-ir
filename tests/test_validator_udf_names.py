"""Layer 2: every operator a program names must be one something implements.

The IR holds operator names as bare strings, checked by
``REGEX_USER_DEFINED_NAME`` and nothing else. That is correct for Layer 1 --
the IR should not know what is implemented -- but until this pass existed
nothing checked it afterwards either, so a program naming an operator no
registry provides passed Layer 1 AND Layer 2 and failed for the first time
inside the evaluator, at a coordinate, partway through a cascade.

Two real mistakes came from that, a week apart, and both are pinned below:

  * ``greater_then`` -- a misspelling of a name that does exist.
  * ``any_of`` -- the Python function name of an operator REGISTERED as
    ``ANY``. A consumer keyed on the function name resolved ``AND`` and
    ``OR``, then failed on ``ANY``.

The second is why suggestions fold case: ``any_of`` and ``ANY`` are far enough
apart for a case-sensitive comparison to suggest nothing, which is precisely
the report that would have saved the time.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from edge_ir.ir.program import Program
from edge_ir.udf.builtins import default_registry
from edge_ir.udf.categories import UdfCategory
from edge_ir.validator import check_udf_names, udf_uses

ARTIFACTS = Path(__file__).resolve().parent.parent / "examples" / "algorithms"


def _program(name: str) -> Program:
    path = next((ARTIFACTS / name).glob("*_program.json"))
    return Program.model_validate(json.loads(path.read_text()))


def _raw(name: str) -> dict:
    path = next((ARTIFACTS / name).glob("*_program.json"))
    return json.loads(path.read_text())


def _rename_first_udf(raw: dict, new: str) -> dict:
    """Point the first user-defined operator at a different name."""
    out = copy.deepcopy(raw)

    def walk(node) -> bool:
        if isinstance(node, dict):
            if node.get("kind") == "user_defined" and "name" in node:
                node["name"] = new
                return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    assert walk(out["main_edge"]), "no user-defined operator to rename"
    return out


################################################################################
# Shipped programs resolve
################################################################################


@pytest.mark.parametrize(
    "name", ["bfs", "dfs", "bellman-ford", "label-propagation"]
)
def test_shipped_programs_name_only_builtins(name: str) -> None:
    """These use nothing beyond the default registry, so they must come clean.

    A failure here means either a program gained an operator the builtins do
    not have, or a builtin was renamed out from under one.
    """
    assert check_udf_names(_program(name), default_registry()) == []


def test_max_flow_needs_its_own_registry() -> None:
    """And the report says which names, rather than just failing.

    max-flow's populate names two operators that live in its own
    ``max_flow_functions.py``, not in the builtins. Against the default
    registry that is a finding, and against its own registry it is not --
    which is the whole reason this pass takes the registry as an argument
    instead of assuming one.
    """
    findings = check_udf_names(_program("max-flow"), default_registry())
    missing = {f.name for f in findings}
    assert missing == {"pick-admissible-edge"}
    assert {f.category for f in findings} == {UdfCategory.COORDINATE}


def test_max_flow_resolves_against_its_own_registry() -> None:
    pytest.importorskip("max_flow_functions")
    from max_flow_functions import max_flow_registry

    assert check_udf_names(_program("max-flow"), max_flow_registry()) == []


################################################################################
# The mistakes this exists to catch
################################################################################


def test_a_misspelled_operator_is_reported_with_the_right_spelling() -> None:
    findings = check_udf_names(
        Program.model_validate(_rename_first_udf(_raw("bfs"), "greater_then")),
        default_registry(),
    )
    assert len(findings) == 1
    assert findings[0].name == "greater_then"
    assert "greater_than" in findings[0].suggestions
    assert "greater_than" in str(findings[0])


def test_a_python_function_name_suggests_the_registry_name() -> None:
    """``any_of`` is registered as ``ANY``; the report has to bridge that.

    This is the case a case-sensitive comparison misses, and the one that
    cost real time. Folding case is what makes the suggestion appear.
    """
    findings = check_udf_names(
        Program.model_validate(_rename_first_udf(_raw("bfs"), "any_of")),
        default_registry(),
    )
    assert len(findings) == 1
    assert findings[0].suggestions[:1] == ("ANY",), findings[0].suggestions


def test_a_name_that_differs_only_in_case_leads_the_suggestions() -> None:
    findings = check_udf_names(
        Program.model_validate(_rename_first_udf(_raw("bfs"), "any")),
        default_registry(),
    )
    assert findings[0].suggestions[0] == "ANY"


def test_an_unrecognisable_name_suggests_nothing_rather_than_guessing() -> None:
    findings = check_udf_names(
        Program.model_validate(_rename_first_udf(_raw("bfs"), "qqqzzz")),
        default_registry(),
    )
    assert len(findings) == 1
    assert findings[0].suggestions == ()
    assert "Did you mean" not in str(findings[0])


def test_a_user_defined_merge_operator_is_reported_as_unsupportable() -> None:
    """No registry can provide one, so this is not a registry question.

    Merge operators are the 16 builtin truth tables and the evaluator refuses
    a user-defined one outright. Catching it at validation beats catching it
    at the first point evaluated.
    """
    raw = _raw("bfs")
    found = False

    def walk(node) -> bool:
        nonlocal found
        if isinstance(node, dict):
            if "merge_op" in node and isinstance(node["merge_op"], dict):
                node["merge_op"] = {"kind": "user_defined", "name": "my-merge"}
                found = True
                return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    walk(raw["main_edge"])
    if not found:
        pytest.skip("no merge op to replace in this artifact")

    findings = check_udf_names(
        Program.model_validate(raw), default_registry()
    )
    merge = [f for f in findings if f.name == "my-merge"]
    assert len(merge) == 1
    assert merge[0].category is None
    assert "built-in only" in merge[0].reason


################################################################################
# Coverage of the walk
################################################################################


def test_every_operator_site_is_visited() -> None:
    """The walk must reach specs AND expressions, or it reports too little.

    A pass that silently looks in fewer places than the program uses is worse
    than no pass, because it reads as a clean bill of health.
    """
    uses = udf_uses(_program("max-flow"))
    cats = {u.category for u in uses}
    assert UdfCategory.MAP_COMPUTE in cats, "specs not visited"
    assert UdfCategory.REDUCE_COMPUTE in cats, "reduce specs not visited"
    assert UdfCategory.COORDINATE in cats, "populate coord_op not visited"
    assert UdfCategory.POPULATE_COMPUTE in cats, "populate compute not visited"
    assert all(u.where.startswith(("initialization[", "cascade[")) for u in uses)


def test_locations_match_the_single_assignment_pass() -> None:
    """Both passes name a site the same way, so findings can be read together."""
    from edge_ir.validator import write_sites

    program = _program("max-flow")
    sites = {s.path for s in write_sites(program)}
    for use in udf_uses(program):
        assert use.where in sites, f"{use.where} is not a known write site"
