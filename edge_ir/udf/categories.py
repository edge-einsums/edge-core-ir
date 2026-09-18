"""The user-defined function categories.

Per ``docs/udf_registry_plan.md`` section 2. Six categories are
user-definable. **Merge operators are deliberately NOT here**: they are
built-in only (the 16 truth tables in ``edge_ir/runtime/op_properties.py``),
settled in the review of that plan.

The lookup key into the registry is ``(category, name)``, not the bare
name. That is what lets a user-defined ``take_left`` *compute* op coexist
with the built-in ``take_left`` *merge* op, and what keeps a coordinate
operator named ``foo`` from colliding with a unary operator named ``foo``.

Compute operators occupy three categories rather than one because their
contracts differ: a reduce compute op is folded (so it wants an identity
and, for a well-defined result, associativity), while a populate compute
op's result is handed to a coordinate operator. The IR has a single
``UserDefinedComputeOp`` node; which category applies is decided
structurally by the spec the operator sits in (``MapSpec.compute_op`` vs
``ReduceSpec.compute_op`` vs ``PopulateSpec.compute_op``).
"""

from __future__ import annotations

from enum import Enum


class UdfCategory(str, Enum):
    """A user-defined function's category; half of its registry key."""

    MAP_COMPUTE = "map_compute"
    REDUCE_COMPUTE = "reduce_compute"
    POPULATE_COMPUTE = "populate_compute"
    COORDINATE = "coordinate"
    UNARY = "unary"
    RANK_MAPPING = "rank_mapping"
    BOOLEAN = "boolean"


COMPUTE_CATEGORIES = frozenset(
    {
        UdfCategory.MAP_COMPUTE,
        UdfCategory.REDUCE_COMPUTE,
        UdfCategory.POPULATE_COMPUTE,
    }
)
