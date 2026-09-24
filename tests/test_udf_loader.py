"""Tests for edge_ir.udf.loader: register_from_module and its checks."""

from __future__ import annotations

import textwrap

import pytest

from edge_ir.udf import UdfCategory, UdfModuleDecl, default_registry, register_from_module


@pytest.fixture
def module_file(tmp_path):
    path = tmp_path / "ops.py"
    path.write_text(
        textwrap.dedent(
            """
            import math

            def sub_then_exp(left, right):
                return math.exp(left - right)

            def wants_ctx(left, right, ctx):
                return left
            """
        )
    )
    return path


def test_registers_a_matching_function(module_file):
    reg = default_registry()
    register_from_module(
        reg,
        module_file,
        [UdfModuleDecl("sub_then_exp", UdfCategory.MAP_COMPUTE, arity=2)],
    )
    impl = reg.impl(UdfCategory.MAP_COMPUTE, "sub_then_exp")
    assert impl.call(5.0, 3.0) == pytest.approx(7.389056, rel=1e-5)


def test_wants_context_inferred_from_signature(module_file):
    reg = default_registry()
    register_from_module(
        reg,
        module_file,
        [UdfModuleDecl("wants_ctx", UdfCategory.MAP_COMPUTE, arity=2)],
    )
    decl = reg.decl(UdfCategory.MAP_COMPUTE, "wants_ctx")
    assert decl.wants_context is True


def test_missing_function_raises(module_file):
    reg = default_registry()
    with pytest.raises(AttributeError, match="no_such_fn"):
        register_from_module(
            reg,
            module_file,
            [UdfModuleDecl("no_such_fn", UdfCategory.MAP_COMPUTE, arity=2)],
        )


def test_arity_mismatch_raises(module_file):
    reg = default_registry()
    with pytest.raises(ValueError, match="arity"):
        register_from_module(
            reg,
            module_file,
            [UdfModuleDecl("sub_then_exp", UdfCategory.MAP_COMPUTE, arity=3)],
        )


def test_missing_file_raises(tmp_path):
    reg = default_registry()
    with pytest.raises(FileNotFoundError):
        register_from_module(
            reg,
            tmp_path / "does_not_exist.py",
            [UdfModuleDecl("anything", UdfCategory.MAP_COMPUTE)],
        )


def test_duplicate_registration_without_override_raises(module_file):
    reg = default_registry()
    decls = [UdfModuleDecl("sub_then_exp", UdfCategory.MAP_COMPUTE, arity=2)]
    register_from_module(reg, module_file, decls)
    with pytest.raises(ValueError, match="already registered"):
        register_from_module(reg, module_file, decls)
