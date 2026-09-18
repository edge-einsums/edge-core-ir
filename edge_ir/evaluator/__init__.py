"""The reference evaluator: the component that executes an EDGE program.

Module layout mirrors the EDGE semantics, so a reader who
knows the EDGE semantics can open this package and recognise it:

===========================  ==============================================
``spaces.py``                rank coordinate sets, tensor
                             coordinate spaces, data spaces, empty values
``tensors.py``               the tensor as a partial
                             function, existence, fibers
``projection.py``            rank variable expressions,
                             the Einsum projection
``iteration.py``             iteration-space construction and predicates
``actions/map.py``           the Map action
``actions/reduce.py``        the Reduce action
``actions/populate.py``      the Populate action
``einsum.py``                one Einsum end to end
``cascade.py``               the cascade, its iterative rank, its stopping
``stopping.py``              the diamond predicate
``order.py``                 the traversal-order seam
===========================  ==============================================

Merge decides existence and compute decides value, and those two stay
separate in this code the way they are separate in the EDGE semantics.

The evaluator is the reference semantics: if the EDGE semantics and this code
disagree, this code is wrong. It fails loudly rather than guessing -- an
action with no operator label, an unbounded rank variable set, an
un-registered function, a reduce operator with no identity and a cascade
that advances a rank without saying when to stop all raise, with a message
naming the construct.
"""

from edge_ir.evaluator.cascade import (
    DEFAULT_MAX_ITERATIONS,
    CascadeTrace,
    detect_iterative_rank,
    run_cascade,
)
from edge_ir.evaluator.datatype_registry import TypeImpl, TypeRegistry
from edge_ir.evaluator.einsum import EinsumResult, evaluate_einsum
from edge_ir.evaluator.environment import Environment, build_environment
from edge_ir.evaluator.errors import (
    EvaluationError,
    IterationLimitExceeded,
    PopulateConstraintViolation,
    UnboundNameError,
    UnderSpecifiedProgramError,
    UnsupportedConstructError,
)
from edge_ir.evaluator.order import (
    Phase,
    Schedule,
    SequentialSchedule,
    ShuffledSchedule,
)
from edge_ir.evaluator.program import (
    EvaluationResult,
    check_order_independence,
    evaluate,
)
from edge_ir.evaluator.spaces import CoordinateSpace, RankCoordinateSet
from edge_ir.evaluator.tensors import FiberEntry, Tensor

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "CascadeTrace",
    "CoordinateSpace",
    "EinsumResult",
    "Environment",
    "EvaluationError",
    "EvaluationResult",
    "FiberEntry",
    "IterationLimitExceeded",
    "Phase",
    "PopulateConstraintViolation",
    "RankCoordinateSet",
    "Schedule",
    "SequentialSchedule",
    "ShuffledSchedule",
    "Tensor",
    "TypeImpl",
    "TypeRegistry",
    "UnboundNameError",
    "UnderSpecifiedProgramError",
    "UnsupportedConstructError",
    "build_environment",
    "check_order_independence",
    "detect_iterative_rank",
    "evaluate",
    "evaluate_einsum",
    "run_cascade",
]
