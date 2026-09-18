Validate the IR semantically. I want to make sure a downstream user (interpreter, compiler) can be confident that the program is well-formed.
For example:

```Python
def validate_program(prog: Program, registry: TypeRegistry) -> None:
    """Layer 2 validation: cross-reference consistency.
    Run after Pydantic validation, before the program is executed
    or compiled. Raises a structured error if anything is wrong.
    """
    _check_tensor_names_unique(prog)
    _check_all_rank_refs_resolve(prog)
    _check_all_udf_names_registered(prog, registry)
    _check_empty_values_match_data_types(prog, registry)
```


Some rules I need to make sure we have:
1. At the validation stage, the Einsum list CAN ONLY CONTAIN output tensors that appear in the cascade. 
2. I need to do a pass on which rank variable expressions are affine
3. I need to do a pass that user-defined functions are actually defined.
4. Stopping condition MUST produce a Boolean.
5. Stopping condition should have same data types as the original tensors it uses.
6. Resolve RankVariableExpressions such that their values match the coordinate set of the rank name they refer to. This also means I need to make sure RankConstantLiterals can take in other data types (user-defined value types beyond int / char-label) — see D20 future-work in design_decisions.md.

## Analysis results go in side tables, not on the IR
Some passes here compute derived facts about the IR (like rule 2 -- which rank variable expressions are affine). Those results do NOT get written onto the core IR nodes. The core IR is frozen and I want it to stay pure syntax. So each analysis pass returns a side table instead: a dict from node -> result, living in edge_ir/analysis/. The IR doesn't know it exists.

Why not just put is_affine on the node? Three reasons: (a) the IR is frozen, so I'd have to rebuild the whole tree to fill the field in, (b) it'd ship as is_affine: null in every saved program before analysis even runs, and (c) it's derived data, so it goes stale if the IR ever gets rewritten. The side table avoids all three, and I just recompute it when I need it.

One exception: RankFunction.is_affine stays on the node, because that one is USER-declared (the user tells us, since the function is a black box). Since it is an input, not something I derive. Only the derived ones (RankArith) move out.