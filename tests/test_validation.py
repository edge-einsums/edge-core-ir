import pytest


@pytest.mark.skip(reason="Layer 2 validator not implemented yet")
def test_layer2_rejects_string_empty_for_int_tensor() -> None:
    """The Layer 2 validator should reject string empty_value for int tensor."""
    # When Layer 2 exists, this would look like:
    # decl = TensorDeclaration(...)
    # with pytest.raises(SomeValidationError):
    #     validate_program(prog_containing_decl, registry)
    pass
