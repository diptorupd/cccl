# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A kernel's own shared arrays next to cooperative scratch.

Numba-CUDA-MLIR does not keep the launch's dynamic shared-memory window
apart from static allocations, so a ``cuda.shared.array`` of shape zero or
of a runtime shape could overlap the static scratch backing. As on coop's
planner, the typed
pipeline refuses such an array when a call takes scratch, with the
planner's ``CoopSinglePhaseRewriteError``, and accepts it when no call
does. Statically sized arrays are accepted.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as common_coop
from cuda.coop.numba_mlir._compiler._rewrite_support import (
    CoopSinglePhaseRewriteError,
)
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import (
    SharedMemoryAliasError,
    TypedPipelineError,
)

from .._support import BLOCK, CHIP, requires_typed_pipeline

pytestmark = requires_typed_pipeline

I32 = types.int32[::1]


def _compile(kernel):
    return testing.compile_kernel(
        kernel, types.void(I32, I32), block=BLOCK, chip=CHIP
    )


def _dynamic_kernels(coop):
    def zero(source, output):
        thread = cuda.threadIdx.x
        mine = cuda.shared.array(0, types.int32)
        mine[thread] = source[thread]
        cuda.syncthreads()
        items = coop.ThreadData(4, dtype=types.int32)
        coop.load(coop.this_block(), source, items, algorithm="transpose")
        items[0] += mine[thread]
        coop.store(coop.this_block(), output, items)

    def keyword_zero(source, output):
        thread = cuda.threadIdx.x
        mine = cuda.shared.array(shape=0, dtype=types.int32)
        mine[thread] = source[thread]
        cuda.syncthreads()
        items = coop.ThreadData(4, dtype=types.int32)
        coop.load(coop.this_block(), source, items)
        items[0] += mine[thread]
        coop.store(coop.this_block(), output, items, algorithm="transpose")

    def rebound_zero(source, output):
        thread = cuda.threadIdx.x
        size = 0
        mine = cuda.shared.array(size, types.int32)
        size = 32
        other = cuda.shared.array(size, types.int32)
        mine[thread] = source[thread]
        other[thread] = source[thread]
        cuda.syncthreads()
        items = coop.ThreadData(4, dtype=types.int32)
        coop.load(coop.this_block(), source, items, algorithm="transpose")
        items[0] += mine[thread] + other[thread]
        coop.store(coop.this_block(), output, items)

    return {f.__name__: f for f in (zero, keyword_zero, rebound_zero)}


@pytest.mark.parametrize(
    "coop", [common_coop, qualified_coop], ids=["common", "qualified"]
)
@pytest.mark.parametrize("shape", ["zero", "keyword_zero", "rebound_zero"])
def test_a_dynamic_shared_array_is_refused_with_scratch(coop, shape):
    kernel = _dynamic_kernels(coop)[shape]

    with pytest.raises(SharedMemoryAliasError) as info:
        _compile(kernel)

    error = info.value
    assert isinstance(error, CoopSinglePhaseRewriteError)
    assert isinstance(error, TypedPipelineError)
    assert isinstance(error, TypingError)
    lines, first = inspect.getsourcelines(kernel)
    line = first + next(
        i for i, text in enumerate(lines) if "cuda.shared.array" in text
    )
    message = str(error)
    assert (
        "-byte static shared-memory backing, but this kernel also declares "
        "dynamic/runtime-sized cuda.shared.array(...) at "
    ) in message
    assert f"test_shared_memory.py ({line}). " in message
    assert "they would alias" in message


@pytest.mark.parametrize(
    "count", [types.int32, types.IntegerLiteral(64)], ids=["runtime", "literal"]
)
def test_a_shared_array_sized_by_an_argument(count):
    coop = common_coop

    def kernel(source, output, count):
        thread = cuda.threadIdx.x
        mine = cuda.shared.array(count, types.int32)
        mine[thread] = source[thread]
        items = coop.ThreadData(4, dtype=types.int32)
        coop.load(coop.this_block(), source, items, algorithm="transpose")
        items[0] += mine[thread]
        coop.store(coop.this_block(), output, items)

    def compile():
        return testing.compile_kernel(
            kernel, types.void(I32, I32, count), block=BLOCK, chip=CHIP
        )

    if isinstance(count, types.IntegerLiteral):
        compile()
        return
    with pytest.raises(
        SharedMemoryAliasError,
        match=r"dynamic/runtime-sized cuda\.shared\.array\(\.\.\.\) at .*"
        r"would alias",
    ):
        compile()


@pytest.mark.parametrize("algorithm", ["direct", "striped", "vectorize"])
def test_a_dynamic_shared_array_is_accepted_without_scratch(algorithm):
    coop = common_coop

    def kernel(source, output):
        thread = cuda.threadIdx.x
        mine = cuda.shared.array(0, types.int32)
        mine[thread] = source[thread]
        items = coop.ThreadData(4, dtype=types.int32)
        coop.load(coop.this_block(), source, items, algorithm=algorithm)
        items[0] += mine[thread]
        coop.store(coop.this_block(), output, items, algorithm=algorithm)

    _compile(kernel)


@pytest.mark.parametrize(
    "shape",
    [128, np.int32(128), np.int64(128), (np.int32(128),)],
    ids=["literal", "int32", "int64", "tuple"],
)
def test_a_static_shared_array_is_accepted_with_scratch(shape):
    coop = common_coop

    def kernel(source, output):
        thread = cuda.threadIdx.x
        mine = cuda.shared.array(shape, types.int32)
        mine[thread] = source[thread]
        items = coop.ThreadData(4, dtype=types.int32)
        coop.load(coop.this_block(), source, items, algorithm="transpose")
        items[0] += mine[thread]
        coop.store(coop.this_block(), output, items)

    _compile(kernel)
