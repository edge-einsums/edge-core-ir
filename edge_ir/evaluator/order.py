"""Traversal order: the seam for the randomized iteration-order mode.

The paper is explicit that an implementation may execute the actions in any
strategy it likes "as long as dependency constraints are maintained"
(the Map semantics): Map for every point first, or Map-then-Reduce per point, or
everything asynchronously in flight. An Einsum's result must therefore not
depend on the order its iteration-space points are visited in.

The randomized iteration-order mode makes that testable: visit the points in
a seeded shuffle and assert the output is unchanged. If it changes, either
the evaluator is secretly relying on a traversal order or the program is not
order-independent.

The dependency model that makes a scramble safe is **phase barriers**, and
they come straight from the semantics rather than being bolted on:

- Map is pointwise. It reads only input tensors, never its own output, so
  every point is independent and any order is legal.
- Reduce reads *the whole* MapTmp (``Exists_MapTmp(is)`` is a query against
  a completed tensor) and writes one reduction state per output coordinate.
  So Reduce may not begin at a point until Map has finished everywhere --
  that is the first barrier. Within Reduce the order across points sharing
  an output coordinate is a genuine semantic degree of freedom: the EDGE semantics
  says an associative-commutative operator makes the result order-free, and
  otherwise "EDGE guarantees only that the result is equivalent to applying
  the reduction operator to some valid ordering".
- Populate reads RedTmp and the *current* Z, and writes Z. It is a fold, so
  it may not begin until Reduce has finished -- the second barrier.

That is the whole dependency graph at Einsum granularity, and it is what a
worklist scheduler would enforce. :class:`Schedule` is the object a future
out-of-order or parallel backend would replace: it hands out the points for
one phase, and the phases are ordered by the barriers above. Nothing in the
action modules reaches for ``points`` directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeVar

T = TypeVar("T")


class Phase(str, Enum):
    """The three phases of one Einsum, separated by dependency barriers."""

    MAP = "map"
    REDUCE = "reduce"
    POPULATE = "populate"


class Schedule(Protocol):
    """Decides the order in which one phase's work items are handed out."""

    def order(self, items: list[T], phase: Phase) -> list[T]:
        """Return ``items`` in the order this phase should process them."""


@dataclass(frozen=True)
class SequentialSchedule:
    """Visit work items in their natural (deterministic, sorted) order."""

    def order(self, items: list[T], phase: Phase) -> list[T]:
        del phase
        return items


@dataclass(frozen=True)
class ShuffledSchedule:
    """Visit work items in a seeded shuffle.

    Seeded so a failing run is reproducible. Running the same program under
    several seeds and asserting the outputs agree is the sharp
    order-independence test; :func:`edge_ir.evaluator.check_order_independence`
    packages it.
    """

    seed: int
    phases: frozenset[Phase] = frozenset({Phase.MAP, Phase.REDUCE, Phase.POPULATE})

    def order(self, items: list[T], phase: Phase) -> list[T]:
        if phase not in self.phases:
            return items
        shuffled = list(items)
        # A fresh Random per call keyed by the seed, the phase and the item
        # count, so the permutation depends only on those -- not on how many
        # times the schedule has been consulted before. Keying on a string
        # keeps it reproducible across runs (unlike hash() of a tuple, which
        # is salted per process).
        rng = random.Random(f"{self.seed}:{phase.value}:{len(shuffled)}")
        rng.shuffle(shuffled)
        return shuffled
