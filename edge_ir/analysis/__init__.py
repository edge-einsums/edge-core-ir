"""Analysis passes that derive facts about the core IR.

This package imports from ``edge_ir.ir`` but never the reverse. Each
pass returns a *side table* (a dict keyed by ``id(node)``) or a plain
dataclass so the core IR can stay frozen and purely syntactic. See
``docs/design_decisions.md`` for the rationale.

The fusion-analysis pass (dependency graph, taxonomy classification,
treatment eligibility, DAG rendering) lives in a separate
package that depends on this
one. Only the truly general, IR-derived passes live here.
"""

from edge_ir.analysis.affine import Affinity, affinity_table, analyze_affinity
from edge_ir.analysis.iteration_space import (
    CoordImage,
    IterationSpace,
    coord_image,
    iteration_space,
    operand_subscripts,
)

__all__ = [
    "Affinity",
    "CoordImage",
    "IterationSpace",
    "affinity_table",
    "analyze_affinity",
    "coord_image",
    "iteration_space",
    "operand_subscripts",
]
