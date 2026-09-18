"""Top-level Program structure.

Per the grammar:
    <EDGE> := <declarations> <initializations> <main-edge>

A Program is the complete IR of an EDGE source file: tensor
declarations, initialization einsums, and the main extended Einsum
(a nested cascade).

Generational rank initialization (which used to be a separate
production in the cascade) lives in tensor declarations as part of the
coord_set machinery. There is no separate
GenerationalRankInit class in the IR.
"""

from __future__ import annotations

from pydantic import Field

from edge_ir.ir.base import IRBase
from edge_ir.ir.einsum import Einsum, NestedCascade
from edge_ir.ir.tensor import TensorDeclaration

################################################################################
# Initialization
################################################################################


class Initialization(IRBase):
    """The initialization section: a list of Einsums that initialize
    input tensors.

    Per the grammar, the initialization list is a sequence of Einsums.
    Any valid Einsum may appear here (per paper Section 7.2). The
    initialization list may be empty for programs that don't need
    explicit tensor initialization.

    At the validation stage, the Einsum list CAN ONLY CONTAIN
    output tensors that appear in the cascade.
    """

    einsums: list[Einsum] = []


################################################################################
# MainEdge
################################################################################


class MainEdge(IRBase):
    """The main extended Einsum section: a nested cascade.

    This is the workhorse of an EDGE program. Per paper Section 7.3,
    a fully specified extended Einsum is a cascade of Einsum
    expressions, possibly nested.
    """

    cascade: NestedCascade


################################################################################
# Program
################################################################################


class Program(IRBase):
    """A complete EDGE program in IR form.

    Per the grammar:
        <EDGE> := <declarations> <initializations> <main-edge>

    The three sections together specify a complete EDGE computation:
    declarations name the tensors and their shapes, initialization sets
    up input tensor values, and the main edge specifies the computation.
    """

    declarations: list[TensorDeclaration] = Field(min_length=1)
    initialization: Initialization
    main_edge: MainEdge
