"""Shared builders for the catalog of EDGE stopping conditions.

These were previously copy-pasted across ``test_ir_stopping.py``,
``test_ir_einsum.py``, and ``test_ir_program.py``. They build structurally
valid :class:`StoppingCondition` instances for well-known EDGE programs;
full structural coverage of the stopping-condition tree lives in
``test_ir_stopping.py``.

Scalars in stopping-condition comparisons are represented as
:class:`TensorProjectionValue` references to a synthesized zero-rank
:class:`~edge_ir.ir.tensor.TensorDeclaration` (named ``Lit_<n>`` by
convention). The declaration itself lives in the surrounding ``Program``;
the helpers here only construct the reference site.
"""

from __future__ import annotations

from edge_ir.ir.expr import (
    RankArith,
    RankConstantLiteral,
    RankConstantShapeSym,
    RankVariable,
    TensorProjection,
)
from edge_ir.ir.stopping import (
    Comparison,
    PropertyApp,
    RankExpressionValue,
    StoppingCondition,
    TensorProjectionValue,
)

__all__ = [
    "bfs_stopping_condition",
    "sssp_stopping_condition",
    "fusemax_stopping_condition",
    "skip_by_three_stopping_condition",
]


def _zero_rank_literal_ref(name: str = "Lit_0") -> TensorProjectionValue:
    """Reference to a synthesized zero-rank tensor literal.

    The actual TensorDeclaration (carrying ``value=0`` and the contextually
    inherited data_type / empty_value=None) lives in the surrounding
    Program's declarations list.

    Note: these catalog helpers default the literal name to ``Lit_0`` because
    each helper is constructed in isolation (no surrounding Program) and the
    tests only need a structurally valid TensorProjectionValue reference.
    They do NOT model the D21 "fresh name per literal occurrence" rule —
    that rule is a parser-side contract, not a test-helper contract. Tests
    that need to exercise the fresh-name invariant should construct
    TensorProjectionValue references directly with distinct names.
    """
    return TensorProjectionValue(proj=TensorProjection(tensor=name, ranks=[]))


def bfs_stopping_condition(
    rank_variable: str = "i", tensor: str = "F"
) -> StoppingCondition:
    """BFS: ``<>_i : occupancy(F_{i+1}) == 0``."""
    return StoppingCondition(
        rank_variable=rank_variable,
        predicate=Comparison(
            lhs=PropertyApp(
                name="occupancy",
                pinned_rank_variables=[rank_variable],
                operands=[
                    TensorProjection(
                        tensor=tensor,
                        ranks=[
                            RankArith(
                                op="+",
                                lhs=RankVariable(name=rank_variable),
                                rhs=RankConstantLiteral(value=1),
                            )
                        ],
                    )
                ],
            ),
            op="==",
            rhs=_zero_rank_literal_ref(),
        ),
    )


def sssp_stopping_condition() -> StoppingCondition:
    """SSSP: ``<> : D_{i+1} == D_i`` (convergence on the distance vector)."""
    return StoppingCondition(
        rank_variable="i",
        predicate=Comparison(
            lhs=TensorProjectionValue(
                proj=TensorProjection(
                    tensor="D",
                    ranks=[
                        RankArith(
                            op="+",
                            lhs=RankVariable(name="i"),
                            rhs=RankConstantLiteral(value=1),
                        ),
                        RankVariable(name="s"),
                    ],
                )
            ),
            op="==",
            rhs=TensorProjectionValue(
                proj=TensorProjection(
                    tensor="D",
                    ranks=[RankVariable(name="i"), RankVariable(name="s")],
                )
            ),
        ),
    )


def fusemax_stopping_condition() -> StoppingCondition:
    """FuseMax-ish: ``<> : i >= K``."""
    return StoppingCondition(
        rank_variable="i",
        predicate=Comparison(
            lhs=RankExpressionValue(expr=RankVariable(name="i")),
            op=">=",
            rhs=RankExpressionValue(expr=RankConstantShapeSym(name="K")),
        ),
    )


def skip_by_three_stopping_condition() -> StoppingCondition:
    """skip-by-3: ``<>_i : F_{i+3} == F_i``."""
    return StoppingCondition(
        rank_variable="i",
        predicate=Comparison(
            lhs=TensorProjectionValue(
                proj=TensorProjection(
                    tensor="F",
                    ranks=[
                        RankArith(
                            op="+",
                            lhs=RankVariable(name="i"),
                            rhs=RankConstantLiteral(value=3),
                        )
                    ],
                )
            ),
            op="==",
            rhs=TensorProjectionValue(
                proj=TensorProjection(tensor="F", ranks=[RankVariable(name="i")])
            ),
        ),
    )
