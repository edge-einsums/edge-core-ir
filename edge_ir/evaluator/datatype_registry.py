"""User-defined type registry.

Maps user-defined type names to Python implementations. Each
implementation knows how to:
- validate that a value is a valid instance of the type
- compare two values for equality
- provide an empty value if asked
"""

from typing import Protocol


class TypeImpl(Protocol):
    """Protocol that user-defined type implementations must satisfy."""

    def validate(self, value: object) -> bool: ...
    def equal(self, a: object, b: object) -> bool: ...


class TypeRegistry:
    def __init__(self) -> None:
        self._types: dict[str, TypeImpl] = {}

    def register(self, name: str, impl: TypeImpl) -> None:
        if name in self._types:
            raise ValueError(f"type {name!r} already registered")
        self._types[name] = impl

    def get(self, name: str) -> TypeImpl:
        if name not in self._types:
            raise KeyError(f"unknown type {name!r}")
        return self._types[name]
