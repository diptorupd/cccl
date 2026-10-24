# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A cooperative primitive the typed pipeline lacks is a typing error.

It never falls back to the planner, and it never surfaces as Numba's generic
"untyped global name" message. Every kernel-callable primitive now has a
template, so the refusal is tested on its own.
"""

from __future__ import annotations

from ._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline


def test_an_unsupported_primitive_names_itself_and_the_switch():
    from numba_cuda_mlir.numba_cuda.core.errors import TypingError

    from cuda.coop.numba_mlir._pipeline._errors import unsupported_primitive

    error = unsupported_primitive("cuda.coop.numba_mlir.example")
    assert isinstance(error, TypingError)
    message = str(error)
    assert (
        "cuda.coop.numba_mlir.example is not supported on the typed pipeline"
        in message
    )
    assert "Unset CUDA_COOP_NUMBA_MLIR_PIPELINE" in message


def test_every_kernel_callable_primitive_has_a_template():
    from numba_cuda_mlir.extending import typing_registry

    from cuda.coop.numba_mlir._pipeline._templates import kernel_primitives

    primitives = kernel_primitives()
    names = set(primitives.values())
    assert {"cuda.coop.load", "cuda.coop.numba_mlir.store"} <= names
    assert "cuda.coop.register" not in names
    registered = {key for key, _ in typing_registry.globals}
    missing = [n for obj, n in primitives.items() if obj not in registered]
    assert not missing
