# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block load and store through contiguous slices, run on the GPU.

A slice without a step stays one-dimensional and C-contiguous, so it is a
valid memory operand; its base address is the slice's first element.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba_cuda_mlir import cuda

import cuda.coop as coop_common
import cuda.coop.numba_mlir as coop_backend

from .._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline

THREADS = 128
ITEMS = 4
START = 128


@pytest.mark.parametrize(
    "coop", [coop_common, coop_backend], ids=["common", "backend"]
)
@pytest.mark.parametrize("algorithm", ["direct", "transpose"])
def test_a_contiguous_slice_loads_and_stores_from_its_start(coop, algorithm):
    @cuda.jit
    def sliced(source, output):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS)
        coop.load(block, source[START:], items, algorithm=algorithm)
        coop.store(block, output[START:], items, algorithm=algorithm)

    source = np.arange(START + THREADS * ITEMS, dtype=np.int32) * 3 + 1
    d_out = cuda.to_device(np.zeros_like(source))
    sliced[1, THREADS](cuda.to_device(source), d_out)

    expected = source.copy()
    expected[:START] = 0
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)
