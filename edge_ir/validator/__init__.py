"""Layer 2 validation: checks that run on a constructed Program.

Pydantic validates a program's structure as it is built. These passes check
what structure alone cannot: rules that hold across Einsums.

Passes report; they do not raise. A caller decides what a violation means.
"""

from __future__ import annotations

from edge_ir.validator.single_assignment import (
    SingleAssignmentViolation,
    WriteSite,
    check_single_assignment,
    write_sites,
)

__all__ = [
    "SingleAssignmentViolation",
    "WriteSite",
    "check_single_assignment",
    "write_sites",
]
