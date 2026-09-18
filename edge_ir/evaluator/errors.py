"""Errors raised by the reference evaluator.

The evaluator FAILS LOUDLY. When the IR under-specifies something the
evaluator needs -- an action with no operator label, an empty value the
Einsum never pins down, a rank variable whose coordinate set cannot be
derived -- it raises rather than guessing a default and proceeding.

Every error carries enough context to point at the offending construct:
the Einsum's output tensor, the rank variable, the operator name, or the
tensor name, as applicable.
"""

from __future__ import annotations


class EvaluationError(Exception):
    """Base class for every error the evaluator raises."""


class UnderSpecifiedProgramError(EvaluationError):
    """The IR does not pin down something the evaluator must know.

    This is the "honest" failure: the program is structurally valid
    Pydantic, but a value the semantics require is simply absent. Do not
    catch this to substitute a default -- fix the program (or the IR).
    """


class UnboundNameError(EvaluationError):
    """A name the IR carries does not resolve in the runtime environment.

    Bucket-C resolution failure per ``docs/design_decisions.md`` D22: a
    tensor name with no data, a shape symbol with no integer, a named
    coordinate set the host did not supply, or a user-defined function
    absent from the registry.
    """


class IterationLimitExceeded(EvaluationError):
    """The cascade ran past its configured maximum iteration count.

    A buggy stopping condition cannot hang the evaluator; it trips this
    instead.
    """


class PopulateConstraintViolation(EvaluationError):
    """A Populate coordinate operator broke a semantic constraint.

    Per the Populate semantics, a coordinate may be marked ``Write`` only
    if it maps to the empty value of Z at this iteration-space point, and
    a coordinate deleted in this round may not also be written in it. The
    EDGE semantics say an implementation "should raise an error (at compile
    or runtime) if a violation is detected"; this is that error.
    """


class UnsupportedConstructError(EvaluationError):
    """A well-formed IR construct the reference evaluator does not implement.

    Distinct from :class:`UnderSpecifiedProgramError`: the program is
    complete and meaningful, but this evaluator has not implemented that
    corner yet. Always names what is unsupported so the gap is visible
    rather than silently mis-evaluated.
    """
