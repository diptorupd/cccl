# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The memory operand of load and store must be one-dimensional and contiguous.

A provider receives only the base address of a ``load`` source or a ``store``
destination, so coop's planner types the operand as a one-dimensional
C-contiguous array and rejects any other. The typed pipeline rejects a
strided view, an argument of any layout and a two-dimensional array after
typing, naming the operand and the call line.
"""

from __future__ import annotations

import inspect

import pytest
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as common_coop
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import BLOCK, CHIP, requires_typed_pipeline

pytestmark = requires_typed_pipeline

ITEMS = 4
CONTIGUOUS = types.float32[::1]

# The operand's type as the kernel receives it, whether the kernel takes a
# strided view of it, and the type the operand has at the call.
FORMS = {
    "strided_view": (
        CONTIGUOUS,
        True,
        "array(float32, 1d, A) (1-D, layout 'A')",
    ),
    "any_layout_argument": (
        types.float32[:],
        False,
        "array(float32, 1d, A) (1-D, layout 'A')",
    ),
    "two_dimensional": (
        types.float32[:, ::1],
        False,
        "array(float32, 2d, C) (2-D, layout 'C')",
    ),
}


def _kernels(coop):
    def load_strided(source, output):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS, dtype=types.float32)
        coop.load(block, source[::2], items)
        coop.store(block, output, items)

    def load_whole(source, output):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS, dtype=types.float32)
        coop.load(block, source, items)
        coop.store(block, output, items)

    def store_strided(source, output):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS, dtype=types.float32)
        coop.load(block, source, items)
        coop.store(block, output[::2], items)

    return {
        ("load", True): load_strided,
        ("load", False): load_whole,
        ("store", True): store_strided,
        ("store", False): load_whole,
    }


def _call_line(kernel, operation):
    lines, first = inspect.getsourcelines(kernel)
    return first + next(
        i for i, text in enumerate(lines) if f"coop.{operation}(" in text
    )


@pytest.mark.parametrize(
    "coop", [common_coop, qualified_coop], ids=["common", "qualified"]
)
@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("form", list(FORMS))
def test_a_memory_operand_must_be_one_dimensional_and_contiguous(
    coop, operation, form
):
    memory, strided, got = FORMS[form]
    kernel = _kernels(coop)[operation, strided]
    if operation == "load":
        operand, args = "source", (memory, CONTIGUOUS)
    else:
        operand, args = "destination", (CONTIGUOUS, memory)

    with pytest.raises(TypedPipelineError) as info:
        testing.compile_kernel(
            kernel, types.void(*args), block=BLOCK, chip=CHIP
        )

    assert isinstance(info.value, TypingError)
    line = _call_line(kernel, operation)
    assert (
        f"coop.{operation} at line {line}: {operand} must be a "
        f"one-dimensional C-contiguous array, got {got}."
    ) in str(info.value)


@pytest.mark.parametrize(
    "coop", [common_coop, qualified_coop], ids=["common", "qualified"]
)
def test_a_contiguous_slice_is_accepted(coop):
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS, dtype=types.float32)
        coop.load(block, source[128:], items)
        coop.store(block, output[128:], items)

    testing.compile_kernel(
        kernel, types.void(CONTIGUOUS, CONTIGUOUS), block=BLOCK, chip=CHIP
    )
