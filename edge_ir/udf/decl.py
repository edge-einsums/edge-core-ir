"""Declaration and implementation halves of a user-defined function.

Per ``docs/udf_registry_plan.md`` section 6, every registered UDF has two
halves:

- a **declaration** (:class:`UdfDecl`) -- pure data: name, category, arity,
  and the category-specific extras the validator and future compiler
  backends need (a reduce identity; associativity; a rank-mapping
  function's affine flag). The Layer 2 validator can do its whole job from
  declarations alone, without executing anything.
- an **implementation** (:class:`UdfImpl`) -- however the function actually
  runs. :class:`PythonUdfImpl` wraps a Python callable; the ABC is the seam
  for a future subprocess/Wasm/ctypes implementation in another language.

The evaluator only ever reaches an implementation through the registry; it
never imports user code directly.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from edge_ir.udf.categories import UdfCategory


class _EmptyIdentity:
    """Sentinel: the reduction identity is the output tensor's empty value.

    Some reduce compute operators have no identity in the data space of
    their own -- ``ANY`` is the canonical case: it selects one of the
    contributions, so its "identity" is simply the absence of any
    contribution. Declaring ``identity=EMPTY_IDENTITY`` tells the evaluator
    to seed the reduction state with ``e^Z``, which makes the merge
    operator's left presence flag ``b_s`` False on first touch -- exactly
    the Reduce semantics describe for a freshly created state.
    """

    _instance: _EmptyIdentity | None = None

    def __new__(cls) -> _EmptyIdentity:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "EMPTY_IDENTITY"


EMPTY_IDENTITY = _EmptyIdentity()

# Sentinel distinct from "no identity declared".
_NO_IDENTITY = object()


@dataclass(frozen=True)
class UdfDecl:
    """The pure-data half of a registered user-defined function."""

    name: str
    category: UdfCategory
    arity: int | None = None
    """Number of operands, or ``None`` for variadic."""

    identity: Any = _NO_IDENTITY
    """Reduction identity. Only meaningful for ``REDUCE_COMPUTE``."""

    associative: bool | None = None
    commutative: bool | None = None
    is_affine: bool | None = None
    """User-declared affineness. Only meaningful for ``RANK_MAPPING``."""

    maps_empty_to_empty: bool = True
    """The operator sends the empty value to the empty value.

    Only meaningful for ``UNARY``. It is what decides the merge operator of
    the degenerate Map a unary sugars into
    (EDGE's decomposition of unary functions): an operator that maps
    empty to empty visits only the non-empty points (``take_left``); one
    that maps empty to NON-empty -- logical complement is the example the
    EDGE semantics give -- must instead visit the points where the operand is
    empty (``not_left``).
    """

    wants_context: bool = False
    """The implementation takes the action context as a ``ctx`` keyword."""

    doc: str = ""

    @property
    def has_identity(self) -> bool:
        return self.identity is not _NO_IDENTITY


class UdfImpl(ABC):
    """How a user-defined function actually runs."""

    @abstractmethod
    def call(self, *args: Any, ctx: Any = None) -> Any:
        """Invoke the function."""


@dataclass(frozen=True)
class PythonUdfImpl(UdfImpl):
    """A user-defined function implemented as a Python callable."""

    fn: Callable[..., Any]
    wants_context: bool = False

    def call(self, *args: Any, ctx: Any = None) -> Any:
        if self.wants_context:
            return self.fn(*args, ctx=ctx)
        return self.fn(*args)


def wants_context(fn: Callable[..., Any]) -> bool:
    """True when ``fn`` declares a ``ctx`` parameter.

    Action context (the iteration-space point, the operand coordinates, the
    right operand's presence flag, the reduction identity) is passed only to
    functions that ask for it, so the common two-argument compute operator
    stays a plain two-argument Python function.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
    return "ctx" in params


@dataclass(frozen=True)
class DataTypeDecl:
    """The pure-data half of a registered user-defined data type."""

    name: str
    doc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
