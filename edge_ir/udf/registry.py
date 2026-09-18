"""The user-defined function registry.

Sibling of ``edge_ir/evaluator/datatype_registry.py`` and deliberately the
same shape: ``register`` refuses a duplicate, ``get`` raises on a name it
does not know.

It lives in its own package rather than under ``evaluator/`` because three
consumers need it and they must not import each other: the Layer 2
validator (declarations only), the reference evaluator (implementations),
and any future compiler backend (declarations, for the algebraic-property
metadata that licenses a rewrite). See ``docs/udf_registry_plan.md``
section 1.

Extension contract: adding a compute, coordinate, unary, rank-mapping or
boolean function means registering it here. No evaluator module changes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from edge_ir.evaluator.errors import UnboundNameError
from edge_ir.udf.categories import UdfCategory
from edge_ir.udf.decl import (
    DataTypeDecl,
    PythonUdfImpl,
    UdfDecl,
    UdfImpl,
    wants_context,
)

_Key = tuple[UdfCategory, str]


class FunctionRegistry:
    """Maps ``(category, name)`` to a declaration and an implementation."""

    def __init__(self) -> None:
        self._entries: dict[_Key, tuple[UdfDecl, UdfImpl]] = {}
        self._datatypes: dict[str, DataTypeDecl] = {}

    # -- registration ------------------------------------------------------

    def register(
        self,
        decl: UdfDecl,
        impl: UdfImpl,
        *,
        allow_override: bool = False,
    ) -> None:
        """Register one function. Refuses a duplicate unless told otherwise.

        Overriding is off by default so that a ``Program`` referencing
        ``min`` means the same thing everywhere (plan section 10.6).
        """
        key = (decl.category, decl.name)
        if key in self._entries and not allow_override:
            raise ValueError(
                f"{decl.category.value} function {decl.name!r} is already "
                f"registered; pass allow_override=True to replace it."
            )
        self._entries[key] = (decl, impl)

    def register_python(
        self,
        name: str,
        category: UdfCategory,
        fn: Callable[..., Any],
        **decl_kwargs: Any,
    ) -> None:
        """Convenience: register a Python callable in one call.

        ``wants_context`` is inferred from the callable's signature (a
        ``ctx`` parameter) unless given explicitly.
        """
        ctx_wanted = bool(decl_kwargs.pop("wants_context", wants_context(fn)))
        allow_override = bool(decl_kwargs.pop("allow_override", False))
        decl = UdfDecl(
            name=name,
            category=category,
            wants_context=ctx_wanted,
            **decl_kwargs,
        )
        self.register(
            decl,
            PythonUdfImpl(fn=fn, wants_context=ctx_wanted),
            allow_override=allow_override,
        )

    def register_datatype(self, decl: DataTypeDecl) -> None:
        if decl.name in self._datatypes:
            raise ValueError(f"data type {decl.name!r} is already registered")
        self._datatypes[decl.name] = decl

    # -- lookup ------------------------------------------------------------

    def has(self, category: UdfCategory, name: str) -> bool:
        return (category, name) in self._entries

    def decl(self, category: UdfCategory, name: str) -> UdfDecl:
        """The declaration half. Raises :class:`UnboundNameError` if absent."""
        try:
            return self._entries[(category, name)][0]
        except KeyError:
            raise UnboundNameError(self._missing_message(category, name)) from None

    def impl(self, category: UdfCategory, name: str) -> UdfImpl:
        """The implementation half. Raises :class:`UnboundNameError` if absent."""
        try:
            return self._entries[(category, name)][1]
        except KeyError:
            raise UnboundNameError(self._missing_message(category, name)) from None

    def datatype_decl(self, name: str) -> DataTypeDecl:
        try:
            return self._datatypes[name]
        except KeyError:
            raise UnboundNameError(f"data type {name!r} is not registered.") from None

    def names(self, category: UdfCategory) -> list[str]:
        return sorted(n for (c, n) in self._entries if c is category)

    def __iter__(self) -> Iterator[_Key]:
        return iter(sorted(self._entries, key=lambda k: (k[0].value, k[1])))

    def __len__(self) -> int:
        return len(self._entries)

    # -- composition -------------------------------------------------------

    def extended(self) -> FunctionRegistry:
        """A copy that can be added to without touching this registry."""
        clone = FunctionRegistry()
        clone._entries = dict(self._entries)
        clone._datatypes = dict(self._datatypes)
        return clone

    def _missing_message(self, category: UdfCategory, name: str) -> str:
        known = self.names(category)
        hint = f" Registered {category.value} names: {known}." if known else ""
        return (
            f"no {category.value} function named {name!r} is registered."
            f"{hint} Register one with "
            f"registry.register_python({name!r}, UdfCategory."
            f"{category.name}, fn)."
        )
