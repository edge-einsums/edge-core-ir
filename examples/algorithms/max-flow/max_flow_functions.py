"""The user-defined operators max-flow names, registered for the evaluator.

The IR carries operator names only; a FunctionRegistry gives them meaning
(``edge_ir/udf/__init__.py``). The default registry already defines every
compute operator this cascade uses except the two on E10::

    PushCand_{i,u,v*} = Adm_{i,u,v} <<<_{v*} 1(pick-admissible-edge)

which the builder encodes as a PopulateSpec with compute op
``pick-admissible-edge`` and coordinate op ``select-one-admissible-v``.

Both are registered here as aliases of existing default operators, so the
max-flow run adds no new operator semantics to the evaluator:

- ``pick-admissible-edge`` (populate compute) is ``identity``, the ASCII name
  of the paper's 1 glyph. It writes the RedTmp value at the chosen
  coordinate. E10's map carries Adm through unchanged, so that value is
  ``True``, which is ``PushCand[i,u,v] <- true`` in ``pseudocode.md``.
- ``select-one-admissible-v`` (coordinate) is ``select-min-coord``. Per fiber
  ``(i, u)`` it keeps the admissible ``v`` with the smallest coordinate and
  deletes any other. ``pseudocode.md`` says "choose one such v using
  pick_admissible_edge" and ``einsum.md`` leaves the choice free, so any one
  rule is valid. Smallest-coordinate makes the result independent of the
  order Populate visits the candidates in.

Import note: ``edge_ir.evaluator`` must be imported before ``edge_ir.udf``.
The reverse order hits a circular import.
"""

from __future__ import annotations

import edge_ir.evaluator  # noqa: F401  (must precede edge_ir.udf)
from edge_ir.udf import FunctionRegistry, UdfCategory, UdfDecl, default_registry

PICK_ADMISSIBLE_EDGE = "pick-admissible-edge"
SELECT_ONE_ADMISSIBLE_V = "select-one-admissible-v"


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
        UdfCategory.POPULATE_COMPUTE,
        "identity",
        PICK_ADMISSIBLE_EDGE,
        "E10 populate compute: write the RedTmp value (Adm's True) at the "
        "chosen coordinate; alias of identity",
    )
    _alias(
        registry,
        UdfCategory.COORDINATE,
        "select-min-coord",
        SELECT_ONE_ADMISSIBLE_V,
        "E10 coordinate op: keep the admissible v with the smallest "
        "coordinate in each (i, u) fiber; alias of select-min-coord",
    )
    return registry
