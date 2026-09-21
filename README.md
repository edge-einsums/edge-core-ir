# EDGE IR

EDGE IR is a typed intermediate representation and a reference evaluator for EDGE, a notation that expresses graph algorithms, linear algebra and deep learning as extended Einsums. A front end lowers an EDGE program into this IR. A back end, or the reference evaluator in this repository, consumes it.

The notation is described in [The EDGE Language: Extended General Einsums for Graph Algorithms](https://arxiv.org/abs/2404.11591) (Odemuyiwa et al., arXiv:2404.11591).

## Install

EDGE IR needs Python 3.10 or later. Its only runtime dependency is Pydantic 2.

```bash
pip install -e ".[dev]"
```

## Run a program

Each folder in `examples/algorithms/` has a builder script that constructs a program in the IR. This runs breadth-first search from vertex 0 on a four-vertex graph:

```python
import sys

sys.path.insert(0, "examples/algorithms/bfs")

from build_bfs_program import build_bfs

from edge_ir.evaluator import evaluate

edges = [(0, 1), (1, 2), (0, 3)]
result = evaluate(
    build_bfs(),
    inputs={"G": {edge: 1 for edge in edges}},
    shape_env={"|V|": 4},
    coord_sets={"id": [0]},
)
depths = {}
for (_generation, vertex), depth in result["F"].items():
    depths.setdefault(vertex, depth)
print(depths)
```

This prints `{0: 0, 1: 1, 3: 1, 2: 2}`. Vertex 0 is the root, vertices 1 and 3 are one hop away, and vertex 2 is two hops away.

A serialized program is plain JSON or a pydantic Python file. `examples/algorithms/bfs/bfs_program.json` is the same BFS program, and `schemas/program.schema.json` is the JSON Schema for any program. All EDGE expressions must validate/meet the contract expressed by the program.schema.json file. Please file issues and make pull requests if you run into any issues or have requests. 

## Repository layout

| Path | Contents |
| --- | --- |
| `edge_ir/ir/` | The IR: Pydantic models for tensors, Einsums, expressions, operators, predicates, stopping conditions and programs. |
| `edge_ir/evaluator/` | The reference evaluator. `evaluate(program, inputs=...)` executes a program and returns its tensors. |
| `edge_ir/analysis/` | Analysis passes over the IR, such as iteration spaces and affine rank expressions. |
| `edge_ir/udf/` | Declarations, implementations and the registry for user-defined functions. |
| `edge_ir/runtime/` | Algebraic properties of the built-in operators. |
| `edge_ir/validator/` | Notes on the planned semantic validator. |
| `schemas/` | The JSON Schema for a serialized program, generated from the IR models. |
| `examples/algorithms/` | Hand-built programs: BFS, DFS, Bellman-Ford, label propagation, max-flow, Dijkstra, A\* and widest path. Each folder has the cascade (`einsum.md`), a builder script, the serialized program and notes. |
| `tests/` | The test suite. |
| `docs/` | Design notes and plans. `docs/README.md` lists them. |

`edge_ir/ast/` and `edge_ir/verify/` are empty placeholders.

## Checks

```bash
make check        # ruff, mypy and pytest
python -m pytest  # tests only
```

## Contributors

- Toluwanimi Odemuyiwa ([@Malloc26](https://github.com/Malloc26))
- Sanjana Mali ([@sanjanamali8](https://github.com/sanjanamali8))
- Serban Porumbescu ([@porumbes](https://github.com/porumbes))

## Attribution of the documents

- *Created with the help of Claude Code.* means Claude Code produced the document.
- *<Author Name>, combined with the help of Claude Code.* means the author authored it, but also incorporated some AI validation into the flow.

## License

Apache-2.0. See `LICENSE`.
