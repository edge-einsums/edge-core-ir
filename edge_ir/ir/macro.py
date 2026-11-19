"""Macros: named, parameterized Einsum templates, and their expansion.

LOCAL PATCH (edge-workbench issue 9), for review.

Per the paper's appendix on macros: "A macro is a named, parameterized
Einsum template. Invoking a macro expands it into one or more Einsums via
lexical substitution of its tensor and rank variable expression parameters."

    MACRO NAME(TA_{a,...}, TB_{b,...}, ...) -> TZ_{c,...}, TY_{d,...}
        (body: cascade of Einsums using the parameters)

"The left-hand side of -> lists the input tensor parameters; the right-hand
side declares the output tensors produced by the expansion."

Because a macro is a lexical substitution, it has no meaning at run time:
the cascade in a ``Program`` holds the EXPANDED Einsums, and the evaluator
never sees a macro. What lives here is

- :class:`MacroDefinition`, the template as written, and
- :func:`expand`, the substitution, so every lowering expands the same way.

A ``Program`` does not carry its macro definitions: adding a field to
``Program`` would change every serialized program, and nothing downstream of
the lowering needs them. A conditional arm records the NAME of the macro it
was expanded from (``ConditionalArm.macro``).

Two invocation forms are covered, and they expand identically:

- assignment form, ``Z_{m,n} = GEMM(A_{m,k}, B_{k,n})``, one output;
- cascade form, ``Z, Y, X = 3-GEMM(...)``, several outputs bound
  positionally to the declared ones.

Not covered: the inline form, where a single-output macro is used as an
anonymous tensor inside a larger expression. That needs the body spliced
into the enclosing expression tree, which nothing has asked for yet.

Substitution rules:

- a formal tensor name becomes the actual tensor's name, wherever the body
  reads or writes it;
- a formal rank variable becomes the actual's rank expression at the same
  subscript position. "Since the arguments are rank variable expressions
  (not just variables), substitutions such as a -> m+1 are permitted." A
  formal subscript written as a shift (``TF_{i+1,d}``) binds ``i`` so that
  the shift lands on the actual subscript: against ``F_{i+1,d}`` that is
  ``i -> i``, against ``F_{g,d}`` it is ``i -> g-1``;
- a rank variable several parameters share must be given consistent
  subscripts by the caller ("the caller is responsible for providing
  consistent subscripts across those outputs"); an inconsistent call raises;
- a tensor the body names that is not a parameter is left alone: after
  expansion it is an ordinary named tensor of the enclosing cascade.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from edge_ir.ir import patterns
from edge_ir.ir.base import IRBase
from edge_ir.ir.einsum import Einsum
from edge_ir.ir.expr import TensorProjection

__all__ = ["MacroDefinition", "MacroExpansionError", "expand"]


class MacroExpansionError(ValueError):
    """An invocation that cannot be substituted into its template."""


class MacroDefinition(IRBase):
    """A macro template: its parameters, its declared outputs, and its body.

    ``inputs`` and ``outputs`` are projections of FORMAL tensors: the names
    are placeholders, and the subscripts name the rank variables the body
    uses. ``body`` is written entirely in those formal names.
    """

    name: str = Field(pattern=patterns.REGEX_USER_DEFINED_NAME)
    inputs: list[TensorProjection] = []
    outputs: list[TensorProjection] = Field(min_length=1)
    body: list[Einsum] = Field(min_length=1)


def _shift(rank: dict[str, Any]) -> tuple[str, int] | None:
    """(variable, integer offset) when a subscript is ``v``, ``v+k`` or ``v-k``."""
    if rank.get("kind") == "var":
        return rank["name"], 0
    if (
        rank.get("kind") == "arith"
        and rank["op"] in ("+", "-")
        and rank["lhs"].get("kind") == "var"
        and rank["rhs"].get("kind") == "literal"
        and isinstance(rank["rhs"]["value"], int)
    ):
        k = rank["rhs"]["value"]
        return rank["lhs"]["name"], k if rank["op"] == "+" else -k
    return None


def _minus(rank: dict[str, Any], k: int) -> dict[str, Any]:
    """``rank - k`` as a rank expression, folding an existing integer shift."""
    if k == 0:
        return rank
    shifted = _shift(rank)
    if shifted is not None:
        var, have = shifted
        net = have - k
        base = {"kind": "var", "name": var}
        if net == 0:
            return base
        return {
            "kind": "arith",
            "op": "+" if net > 0 else "-",
            "lhs": base,
            "rhs": {"kind": "literal", "value": abs(net)},
        }
    return {
        "kind": "arith",
        "op": "-" if k > 0 else "+",
        "lhs": rank,
        "rhs": {"kind": "literal", "value": abs(k)},
    }


def _bind(
    macro: MacroDefinition,
    formals: list[TensorProjection],
    actuals: list[TensorProjection],
    what: str,
    tensors: dict[str, str],
    variables: dict[str, dict[str, Any]],
) -> None:
    if len(formals) != len(actuals):
        raise MacroExpansionError(
            f"macro {macro.name!r} declares {len(formals)} {what}(s) and was "
            f"invoked with {len(actuals)}."
        )
    for formal, actual in zip(formals, actuals, strict=True):
        bound = tensors.setdefault(formal.tensor, actual.tensor)
        if bound != actual.tensor:
            raise MacroExpansionError(
                f"macro {macro.name!r}: the parameter {formal.tensor!r} is "
                f"bound to both {bound!r} and {actual.tensor!r}."
            )
        if len(formal.ranks) != len(actual.ranks):
            raise MacroExpansionError(
                f"macro {macro.name!r}: {formal.tensor!r} is written with "
                f"{len(formal.ranks)} subscript(s), but {actual.tensor!r} was "
                f"passed with {len(actual.ranks)}."
            )
        for f_rank, a_rank in zip(formal.ranks, actual.ranks, strict=True):
            shifted = _shift(f_rank.model_dump())
            if shifted is None:
                raise MacroExpansionError(
                    f"macro {macro.name!r}: the subscript of {formal.tensor!r} "
                    f"is not a rank variable or an integer shift of one, so "
                    f"nothing says which variable the call binds."
                )
            var, offset = shifted
            value = _minus(a_rank.model_dump(), offset)
            if variables.setdefault(var, value) != value:
                raise MacroExpansionError(
                    f"macro {macro.name!r}: the rank variable {var!r} is bound "
                    f"inconsistently across the call's subscripts. The caller "
                    f"must give consistent subscripts wherever parameters "
                    f"share a rank variable."
                )


def _substitute(
    node: Any, tensors: dict[str, str], variables: dict[str, dict[str, Any]]
) -> Any:
    if isinstance(node, list):
        return [_substitute(item, tensors, variables) for item in node]
    if not isinstance(node, dict):
        return node
    if node.get("kind") in ("var", "arith"):
        # A subscript that is a formal variable, or an integer shift of one:
        # substitute, folding the shift into the actual's own (`i+1` under
        # i -> g-1 is `g`, not `(g-1)+1`).
        shifted = _shift(node)
        if shifted is not None and shifted[0] in variables:
            return _minus(variables[shifted[0]], -shifted[1])
        if node.get("kind") == "var":
            return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("output_tensor", "tensor") and isinstance(value, str):
            out[key] = tensors.get(value, value)
        elif key == "rank_list" and isinstance(value, list):
            # Specs name rank variables as plain strings; follow a variable
            # that was renamed to another variable.
            renamed = []
            for name in value:
                target = variables.get(name)
                if target is not None and target.get("kind") == "var":
                    renamed.append(target["name"])
                else:
                    renamed.append(name)
            out[key] = renamed
        else:
            out[key] = _substitute(value, tensors, variables)
    return out


def expand(
    macro: MacroDefinition,
    inputs: list[TensorProjection],
    outputs: list[TensorProjection],
) -> list[Einsum]:
    """The Einsums one invocation of ``macro`` stands for.

    ``inputs`` are the call's arguments and ``outputs`` the tensors on its
    left-hand side, each bound positionally to the declared ones.
    """
    tensors: dict[str, str] = {}
    variables: dict[str, dict[str, Any]] = {}
    _bind(macro, macro.inputs, inputs, "input", tensors, variables)
    _bind(macro, macro.outputs, outputs, "output", tensors, variables)
    return [
        Einsum.model_validate(
            _substitute(einsum.model_dump(), tensors, variables), strict=False
        )
        for einsum in macro.body
    ]
