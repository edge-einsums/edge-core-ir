"""The three EDGE actions, one module each.

Mirrors the EDGE semantics of Map, Reduce and
Populate. Map decides whether a computation happens at a point and
what it computes; Reduce gathers Map's results by output coordinate; Populate
places the result into the output tensor. Merge decides *existence* and
compute decides *value*, and the two stay separate here the way they are
separate in the EDGE semantics.
"""

from edge_ir.evaluator.actions.map import MapResult, map_action
from edge_ir.evaluator.actions.populate import populate_action
from edge_ir.evaluator.actions.reduce import reduce_action

__all__ = ["MapResult", "map_action", "populate_action", "reduce_action"]
