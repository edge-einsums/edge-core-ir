"""Load user-defined function implementations from a Python file.

A ``.edge`` program's ``.udf`` section declares functions by name, category,
and metadata (arity, associativity, ...) but carries no implementation --
the IR must not hold one (``IRBase`` is frozen and JSON-round-trippable, see
``edge_ir/udf/decl.py``). :func:`register_from_module` is the connection
between a declared name and an actual Python callable, so the evaluator has
something to call.

This module is frontend-agnostic: it takes an already-parsed module path and
declaration list, so it has no opinion on ``.edge`` syntax. A parser is
responsible for producing ``decls`` from a ``.udf`` section; this is the
piece downstream of that, per ``docs/udf_registry_plan.md``'s "directory-
loader/decorator ergonomics" gap.

Executes user-supplied code by design -- callers should not point this at
untrusted ``.edge`` files without knowing that.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from edge_ir.udf.categories import UdfCategory
from edge_ir.udf.registry import FunctionRegistry

__all__ = ["UdfModuleDecl", "register_from_module"]


class UdfModuleDecl:
    """One declared function: its registry key plus registration metadata.

    Mirrors the shape a ``.udf`` section parser would produce -- a name, the
    category it's registered under, and whatever extra keyword arguments
    :meth:`FunctionRegistry.register_python` accepts (``arity``, ``identity``,
    ``associative``, ``commutative``, ``wants_context``, ``allow_override``,
    ``doc``, ...).
    """

    __slots__ = ("name", "category", "kwargs")

    def __init__(self, name: str, category: UdfCategory, **kwargs: Any) -> None:
        self.name = name
        self.category = category
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"UdfModuleDecl({self.name!r}, {self.category!r}, {self.kwargs!r})"


def _load_module(module_path: str | Path) -> Any:
    """Import a Python file at an arbitrary path as a module.

    Not a normal ``import`` statement -- the path comes from a ``.edge``
    file's ``impl "..."`` line, which is not necessarily importable via the
    normal package machinery (it may sit next to the ``.edge`` file rather
    than on ``sys.path``).
    """
    path = Path(module_path)
    if not path.is_file():
        raise FileNotFoundError(f"UDF implementation file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load a module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_arity(name: str, fn: Any, declared_arity: int | None) -> None:
    """Check fn's parameter count against a declared arity, if one was given.

    ``ctx`` is excluded from the count -- it is how a function opts into
    receiving evaluation context (see :func:`edge_ir.udf.decl.wants_context`),
    not a value the .edge program passes.
    """
    if declared_arity is None:
        return

    sig = inspect.signature(fn)
    n_params = sum(1 for p in sig.parameters.values() if p.name != "ctx")
    if n_params != declared_arity:
        raise ValueError(
            f"'{name}': .udf declares arity={declared_arity}, but the "
            f"Python function takes {n_params} argument(s) "
            f"(excluding 'ctx' if present)."
        )


def register_from_module(
    reg: FunctionRegistry,
    module_path: str | Path,
    decls: Sequence[UdfModuleDecl],
) -> None:
    """Import ``module_path`` and register each declared function in ``reg``.

    For each :class:`UdfModuleDecl`:

    1. Look up an attribute of the same name in the imported module.
    2. If a declared ``arity`` is present in ``kwargs``, check it against
       the function's real signature.
    3. Register it via ``reg.register_python(name, category, fn, **kwargs)``
       -- unchanged from manual registration.

    Raises :class:`AttributeError` if a declared name has no matching
    function in the module, and :class:`ValueError` on an arity mismatch or
    a duplicate registration (per
    :meth:`FunctionRegistry.register` -- pass ``allow_override=True`` in a
    decl's kwargs to replace an existing entry instead).
    """
    module = _load_module(module_path)

    for decl in decls:
        fn = getattr(module, decl.name, None)
        if fn is None:
            raise AttributeError(
                f"'{decl.name}' was declared in .udf but no function by "
                f"that name exists in {module_path}"
            )
        _check_arity(decl.name, fn, decl.kwargs.get("arity"))
        reg.register_python(decl.name, decl.category, fn, **decl.kwargs)
