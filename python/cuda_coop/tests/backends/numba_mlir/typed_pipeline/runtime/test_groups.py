# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A ``group_by`` count that is a kernel argument specializes the kernel.

As on coop's planner, the dispatcher compiles the kernel again with the
argument as a literal, so each count is its own specialization, and a count
coop refuses is refused with its class at that launch.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba_cuda_mlir import cuda

from cuda import coop
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline


def test_a_kernel_argument_count_is_typed_as_a_literal():
    @cuda.jit
    def partition(count, output):
        coop.this_warp().group_by(count)
        output[cuda.threadIdx.x] = count

    output = cuda.to_device(np.zeros(32, np.int64))
    partition[1, 32](8, output)
    partition[1, 32](16, output)
    np.testing.assert_array_equal(output.copy_to_host(), np.full(32, 16))
    assert len(partition.overloads) == 2
    assert all("Literal[int]" in str(s) for s in partition.overloads)

    with pytest.raises(
        ValueError, match="requires the count to divide"
    ) as info:
        partition[1, 32](3, output)
    assert isinstance(info.value, TypedPipelineError)
