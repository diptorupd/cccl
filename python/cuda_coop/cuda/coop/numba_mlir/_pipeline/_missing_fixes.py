# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Refuse a kernel that needs a Numba-CUDA-MLIR fix the compiler lacks.

The typed pipeline has no patch standing in for these fixes. On a compiler
without one, ``register`` adds a typing rule that refuses a kernel needing
it with an error naming the fix and its pull request, where the compiler
would otherwise report a generic typing failure.

Scope: error to error. A rule acts only on a kernel the compiler refuses.
"""

from __future__ import annotations

from numba_cuda_mlir.descriptor import mlir_target
from numba_cuda_mlir.typing import cuda as cuda_typing

from . import _bridges as nb
from . import _ncm_fixes as fixes
from . import _typing_rules as rules
from ._errors import TypedPipelineError


def argument_sized_local_array(fnty, arg_types, constraint, released):
    """A cuda.local.array sized by a non-literal integer needs the fix."""

    if not (
        isinstance(fnty, nb.types.Function)
        and cuda_typing.LocalArrayTemplate in fnty.templates
        and arg_types
    ):
        return
    shape, *rest = arg_types
    if not isinstance(shape, nb.types.Integer) or isinstance(
        shape, nb.types.IntegerLiteral
    ):
        return
    typer = cuda_typing.LocalArrayTemplate(mlir_target.typing_context).generic()
    try:
        literal = typer(nb.types.IntegerLiteral(1), *rest)
    except TypeError:
        return
    if literal is None:
        return
    raise TypedPipelineError(
        fixes.missing(
            fixes.LOCAL_ARRAY_SHAPE,
            f"A cuda.local.array whose shape is a non-literal {shape}",
        )
        + " Size it with a literal or a global constant.",
        loc=constraint.loc,
    )


_registered = False


def register() -> None:
    """Register a rule for each fix the installed compiler lacks, once."""

    global _registered
    if _registered:
        return
    if not fixes.present(fixes.LOCAL_ARRAY_SHAPE):
        rules.register(
            nb.types.Function, rules.GATE, argument_sized_local_array
        )
    _registered = True


__all__ = [
    "argument_sized_local_array",
    "register",
]
