"""User-defined functions: declarations, implementations, and the registry.

An EDGE program references compute, coordinate, unary, rank-mapping and
boolean functions **by name only**; the IR never carries an implementation
and must not (``IRBase`` is frozen, strict, and JSON-round-trippable). This
package is where those names acquire meaning.

Three consumers share it and none of them imports the others: the Layer 2
validator (declarations), the reference evaluator (implementations), and
any future compiler backend (declarations, for algebraic metadata).

Merge operators are **not** here -- they are built-in only, the 16 truth
tables in ``edge_ir/runtime/op_properties.py``.
"""

from edge_ir.udf.builtins import default_registry
from edge_ir.udf.categories import COMPUTE_CATEGORIES, UdfCategory
from edge_ir.udf.decl import (
    EMPTY_IDENTITY,
    DataTypeDecl,
    PythonUdfImpl,
    UdfDecl,
    UdfImpl,
)
from edge_ir.udf.registry import FunctionRegistry

__all__ = [
    "COMPUTE_CATEGORIES",
    "EMPTY_IDENTITY",
    "DataTypeDecl",
    "FunctionRegistry",
    "PythonUdfImpl",
    "UdfCategory",
    "UdfDecl",
    "UdfImpl",
    "default_registry",
]
