"""The user-defined operators max-flow names, registered for the evaluator.

The IR carries operator names only; a FunctionRegistry gives them meaning
(``edge_ir/udf/__init__.py``). The default registry already defines every
operator this cascade uses except the one on E10::

    PushCand_{i,u,v*} = Adm_{i,u,v} <<<_{v*} 1(pick-admissible-edge)

``1(name)`` puts the 1 glyph in the populate's COMPUTE slot -- ``identity``
in the registry -- and the parenthesised name in the COORDINATE slot, the one
that decides which ``v`` survives. dfs lowers ``1(select-max-val)`` the same
way. So there is exactly one name to register, and it is the choice function:

- ``pick-admissible-edge`` (coordinate) is ``select-any``. The specification
  is the authority here, and it says ANY admissible edge -- ``einsum.md``
  writes ``1(pick-admissible-edge)`` with no ranking, and ``pseudocode.md``
  says "choose one such v using pick_admissible_edge". ``select-any`` is the
  operator for exactly that: it ranks nothing and keeps whichever candidate
  the schedule offers first.

  It was previously ``select-min-coord``, a lowest-coordinate rule, chosen so
  that the result would not depend on the order Populate visits candidates
  in. That bought a property the specification does not promise, at the cost
  of making the IR claim a tie-break the algorithm does not have -- and of
  sampling the legal executions with a bias, so a bug that only shows up on
  some choices would never show up at all.

  The order-independence that DOES hold is about the ANSWER, not the route:
  every traversal order reaches the same maximum flow. That is what
  ``test_max_flow_reaches_the_same_answer_under_any_order`` asserts.

This file previously registered a second name, ``select-one-admissible-v``,
for the coordinate slot -- invented to fill it because ``pick-admissible-edge``
had been put in the compute slot instead. It appeared in no specification.

Import note: ``edge_ir.evaluator`` must be imported before ``edge_ir.udf``.
The reverse order hits a circular import.
"""

from __future__ import annotations

import edge_ir.evaluator  # noqa: F401  (must precede edge_ir.udf)
from edge_ir.udf import FunctionRegistry, UdfCategory, UdfDecl, default_registry

PICK_ADMISSIBLE_EDGE = "pick-admissible-edge"


def _alias(
    registry: FunctionRegistry,
    category: UdfCategory,
    existing: str,
    name: str,
    doc: str,
) -> None:
    """Register ``name`` with the implementation already bound to ``existing``."""
    decl = registry.decl(category, existing)
    registry.register(
        UdfDecl(
            name=name,
            category=category,
            arity=decl.arity,
            wants_context=decl.wants_context,
            doc=doc,
        ),
        registry.impl(category, existing),
    )


def max_flow_registry() -> FunctionRegistry:
    """The default registry plus E10's two populate operators."""
    registry = default_registry()
    _alias(
        registry,
        UdfCategory.COORDINATE,
        "select-any",
        PICK_ADMISSIBLE_EDGE,
        "E10 coordinate op: keep one admissible v per (i, u) fiber, ANY one. "
        "Alias of select-any, because einsum.md writes 1(pick-admissible-edge) "
        "with no ranking. The route therefore varies with traversal order; "
        "the answer does not.",
    )
    return registry
