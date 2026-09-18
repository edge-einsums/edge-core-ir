# Documentation

> *Created with the help of Claude Code.*

These notes record how the IR was designed. Several plans were written before the reference evaluator existed and describe it as future work. The code in `edge_ir/` is the current behavior.

## IR design

| File | Contents |
| --- | --- |
| `design_decisions.md` | The IR's design decisions, each with its rationale and the alternatives considered. |
| `core_ir_decisions_pass.md` | Open decisions for the IR, the UDF registry and the validator. |
| `design-notes.md` | Early design notes on rank variable expressions, validation and user hints. |
| `parsing.md` | Parser and lowering rules. |
| `cases.md` | How case statements lower into the IR. |
| `usecase.md` | Predicate use cases, with sketches of the lowered IR. |
| `keywords.md` | Reserved keywords. |
| `populate.md` | Terminology for the populate action. |
| `stopping_condition_open_questions.md` | Open questions about stopping conditions. |

## Plans

| File | Contents |
| --- | --- |
| `interpreter_future_plan.md` | Roadmap for the interpreter. |
| `validator_plan.md` | Implementation plan for the semantic (Layer 2) validator. |
| `validator_plan_predicates.md` | The validator's predicate checks. |
| `udf_registry_plan.md` | Design of the user-defined function registry, now implemented in `edge_ir/udf/`. |
| `braindump.md` | A short list of future directions. |
| `TODO.md` | Task list. |

## Reference

| File | Contents |
| --- | --- |
| `ir_file_flow.md` | Map of the modules and how a program moves through them. |
| `debug.md` | Findings from an audit of the IR, and the fixes. |
| `populate-exploration.ipynb` | Notebook that explores populate semantics with fibertree. |
