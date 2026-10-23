# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block load and store into explicit ``TempStorage``, run on the GPU.

A transposing load followed by a transposing store returns the source, so
each kernel checks that its calls' scratch regions do not overlap where
they must not and are reused where they may.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba_cuda_mlir import cuda

import cuda.coop as coop_common
import cuda.coop.numba_mlir as coop_backend
from cuda.coop.numba_mlir._pipeline._errors import TempStorageError

from .._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline

THREADS = 128
ITEMS = 4
DYNAMIC = 64 * 1024


def _run(kernel, sharedmem=0):
    source = np.arange(THREADS * ITEMS, dtype=np.float32) * 3 + 1
    d_out = cuda.device_array_like(source)
    kernel[1, THREADS, 0, sharedmem](cuda.to_device(source), d_out)
    np.testing.assert_array_equal(d_out.copy_to_host(), source)


@pytest.mark.parametrize(
    "coop", [coop_common, coop_backend], ids=["common", "backend"]
)
@pytest.mark.parametrize("sharing", ["shared", "exclusive"])
@pytest.mark.parametrize("auto_sync", [False, True])
def test_one_descriptor_for_load_and_store(coop, sharing, auto_sync):
    @cuda.jit
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS)
        storage = coop.TempStorage(sharing=sharing, auto_sync=auto_sync)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        if not auto_sync and sharing == "shared":
            cuda.syncthreads()
        coop.store(
            block, output, items, algorithm="transpose", temp_storage=storage
        )

    _run(kernel)


def test_explicit_and_implicit_regions_side_by_side():
    coop = coop_common

    @cuda.jit
    def kernel(source, output):
        block = coop.this_block()
        staged = coop.ThreadData(ITEMS)
        items = coop.ThreadData(ITEMS)
        storage = coop.TempStorage(4096, alignment=32)
        coop.load(
            block, source, staged, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, staged, algorithm="warp_transpose")
        cuda.syncthreads()
        coop.load(block, output, items, algorithm="warp_transpose")
        coop.store(
            block, output, items, algorithm="transpose", temp_storage=storage
        )

    _run(kernel)


def test_scratch_above_the_static_limit_runs_in_dynamic_shared_memory():
    coop = coop_common

    @cuda.jit
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS)
        storage = coop.TempStorage(DYNAMIC, auto_sync=True)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(
            block, output, items, algorithm="transpose", temp_storage=storage
        )

    _run(kernel, sharedmem=DYNAMIC)
    _run(kernel, sharedmem=DYNAMIC)


def test_a_dynamic_backing_the_launch_does_not_request_is_refused():
    coop = coop_common

    @cuda.jit
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(ITEMS)
        storage = coop.TempStorage(DYNAMIC)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    with pytest.raises(TempStorageError, match="launch requests 0"):
        _run(kernel)
