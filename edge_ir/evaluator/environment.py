"""The runtime environment an Einsum is evaluated against.

Everything in D22's buckets B and C lives here: the shape symbols resolved
to integers, the host-supplied named coordinate sets, the actual tensor
data, and the two registries that give user-defined names meaning. The IR
itself (bucket A) is passed alongside, never merged in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from edge_ir.evaluator.datatype_registry import TypeRegistry
from edge_ir.evaluator.errors import UnboundNameError
from edge_ir.evaluator.spaces import (
    Coordinate,
    CoordinateSpace,
    CoordSetEnv,
    ShapeEnv,
    coordinate_space,
)
from edge_ir.evaluator.tensors import Tensor
from edge_ir.ir.program import Program
from edge_ir.ir.tensor import TensorDeclaration
from edge_ir.udf.registry import FunctionRegistry


@dataclass
class Environment:
    """Resolved declarations, live tensor data, and the registries."""

    declarations: dict[str, TensorDeclaration]
    spaces: dict[str, CoordinateSpace]
    tensors: dict[str, Tensor]
    shape_env: ShapeEnv
    coord_sets: CoordSetEnv
    functions: FunctionRegistry
    types: TypeRegistry = field(default_factory=TypeRegistry)

    def tensor(self, name: str) -> Tensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise UnboundNameError(
                f"tensor {name!r} has no data in this environment. Declared "
                f"tensors: {sorted(self.declarations)}; bound: "
                f"{sorted(self.tensors)}."
            ) from None

    def empty_value(self, name: str) -> Any:
        try:
            return self.declarations[name].empty_value
        except KeyError:
            raise UnboundNameError(f"tensor {name!r} is not declared.") from None

    def rank_positions(self, tensor: str, rank_names: list[str]) -> tuple[int, ...]:
        """Positions of the named ranks within a tensor's declaration."""
        decl = self.declarations.get(tensor)
        if decl is None:
            raise UnboundNameError(f"tensor {tensor!r} is not declared.")
        declared = [r.name for r in decl.ranks]
        positions = []
        for name in rank_names:
            if name not in declared:
                raise UnboundNameError(
                    f"tensor {tensor!r} has no rank named {name!r}; its ranks "
                    f"are {declared}."
                )
            positions.append(declared.index(name))
        return tuple(positions)


def build_environment(
    program: Program,
    *,
    inputs: Mapping[str, Mapping[tuple[Coordinate, ...], Any]] | None = None,
    shape_env: ShapeEnv | None = None,
    coord_sets: CoordSetEnv | None = None,
    functions: FunctionRegistry | None = None,
    types: TypeRegistry | None = None,
) -> Environment:
    """Resolve a program's declarations and bind the supplied input data.

    Every declared tensor gets a live :class:`Tensor`, empty unless the
    caller supplied data for it. A zero-rank declaration carrying a literal
    ``value`` (design decision D21's scalar encoding) is seeded with that
    value automatically -- the program pins it, so the host must not have to.
    """
    decls = {d.name: d for d in program.declarations}
    shapes: ShapeEnv = dict(shape_env or {})
    sets: CoordSetEnv = {k: tuple(v) for k, v in (coord_sets or {}).items()}

    spaces: dict[str, CoordinateSpace] = {}
    for name, decl in decls.items():
        spaces[name] = coordinate_space(
            decl, declarations=decls, shape_env=shapes, coord_sets=sets
        )

    tensors: dict[str, Tensor] = {}
    supplied = {k: dict(v) for k, v in (inputs or {}).items()}
    undeclared = sorted(n for n in supplied if n not in decls)
    if undeclared:
        raise UnboundNameError(
            f"input data was supplied for {undeclared!r}, which the program "
            f"does not declare. Declared tensors: {sorted(decls)}."
        )
    for name, decl in decls.items():
        graph = supplied.get(name, {})
        tensor = Tensor(name, spaces[name], decl.empty_value, graph)
        if not decl.ranks and decl.value is not None and name not in supplied:
            tensor.write((), decl.value)
        tensors[name] = tensor

    # Imported here, not at module level: edge_ir.udf.builtins imports
    # edge_ir.evaluator.context, so a module-level import makes
    # `import edge_ir.udf` fail when it runs before `import edge_ir.evaluator`.
    from edge_ir.udf.builtins import default_registry

    return Environment(
        declarations=decls,
        spaces=spaces,
        tensors=tensors,
        shape_env=shapes,
        coord_sets=sets,
        functions=functions if functions is not None else default_registry(),
        types=types if types is not None else TypeRegistry(),
    )
