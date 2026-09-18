"""Base class for all IR nodes.

Every IR node inherits from IRBase and gets the same configuration:
- frozen: instances are immutable after construction
- strict: no silent type coercion (e.g., "5" is not auto-converted to 5)
- extra=forbid: unknown fields raise an error rather than being silently
  dropped or stored
# - use_enum_values: when serializing to dict, enum fields will be represented
#   by their values
"""

# evaluate type annotations lazily.
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class IRBase(BaseModel):
    """
    Parent class for all IR nodes.
    All IR nodes will inherit from this class.
    Also, we are haven it be frozen and strict because we want
    this to return a schema that is strictly data. Later IRs
    can do cool stuff with this.
    """

    model_config = ConfigDict(
        frozen=True,
        strict=True,
        extra="forbid",
    )
