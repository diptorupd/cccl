# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Payload shapes and selectors from ``literal_unroll``, and a rebound
``dtype=`` name, run on the GPU and checked against NumPy.

``literal_unroll`` types its body once per tuple element, so each iteration's
value is a literal of its own: it can be a payload's shape or a call's
``algorithm=``. A name rebound to another dtype is a new value, so each
``ThreadData`` gets the dtype bound when it is built. The planner rejects
these kernels, because it resolves arguments before typing and sees several
definitions of one name; the typed pipeline compiles them.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.misc.special import literal_unroll

import cuda.coop.numba_mlir as coop

from .._support import ALGORITHMS, requires_typed_pipeline

pytestmark = requires_typed_pipeline

THREADS = 128
ITEMS = 4
COUNTS = (1, 2, 4)


def _thread_data_shapes(source, output):
    offset = 0
    for count in literal_unroll(COUNTS):
        payload = coop.ThreadData(items_per_thread=count, dtype=types.int32)
        coop.load(coop.this_block(), source, payload)
        base = offset + cuda.threadIdx.x * count
        for i in range(count):
            output[base + i] = payload[i]
        offset += THREADS * count


def _local_array_shapes(source, output):
    offset = 0
    for count in literal_unroll(COUNTS):
        payload = cuda.local.array(count, types.int32)
        coop.load(coop.this_block(), source, payload)
        base = offset + cuda.threadIdx.x * count
        for i in range(count):
            output[base + i] = payload[i]
        offset += THREADS * count


def _local_array_keyword_shapes(source, output):
    offset = 0
    for count in literal_unroll(COUNTS):
        payload = cuda.local.array(shape=count, dtype=types.int32)
        coop.load(coop.this_block(), source, payload)
        base = offset + cuda.threadIdx.x * count
        for i in range(count):
            output[base + i] = payload[i]
        offset += THREADS * count


# One kernel per constructor: NCM's literal_unroll pass raises KeyError on a
# dynamic index into a name assigned in two branches of the unrolled body.
@pytest.mark.parametrize(
    "kernel",
    [_thread_data_shapes, _local_array_shapes, _local_array_keyword_shapes],
    ids=["thread_data", "local_array", "local_array_keyword"],
)
def test_literal_unroll_values_shape_each_payload(kernel):
    shapes = cuda.jit(kernel)
    source = np.arange(THREADS * max(COUNTS), dtype=np.int32) * 3 + 1
    d_out = cuda.to_device(np.full(THREADS * sum(COUNTS), -1, dtype=np.int32))
    shapes[1, THREADS](cuda.to_device(source), d_out)
    expected = np.concatenate([source[: THREADS * count] for count in COUNTS])
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def _stored(values, algorithm):
    if algorithm == "striped":
        return values.reshape(THREADS, ITEMS).T.ravel()
    return values


def _loaded(values, algorithm):
    if algorithm == "striped":
        return values.reshape(ITEMS, THREADS).T.ravel()
    return values


def test_literal_unroll_values_select_each_store_algorithm():
    @cuda.jit
    def stores(scratch, output):
        thread = cuda.threadIdx.x
        offset = 0
        for algorithm in literal_unroll(ALGORITHMS):
            payload = coop.ThreadData(items_per_thread=ITEMS, dtype=types.int32)
            for i in range(ITEMS):
                payload[i] = thread * ITEMS + i
            coop.store(coop.this_block(), scratch, payload, algorithm=algorithm)
            cuda.syncthreads()
            for i in range(ITEMS):
                output[offset + i * THREADS + thread] = scratch[
                    i * THREADS + thread
                ]
            cuda.syncthreads()
            offset += THREADS * ITEMS

    tile = THREADS * ITEMS
    d_out = cuda.to_device(np.full(tile * len(ALGORITHMS), -1, np.int32))
    stores[1, THREADS](cuda.device_array(tile, np.int32), d_out)
    values = np.arange(tile, dtype=np.int32)
    expected = np.concatenate(
        [_stored(values, algorithm) for algorithm in ALGORITHMS]
    )
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_literal_unroll_values_select_each_load_algorithm():
    @cuda.jit
    def loads(source, output):
        offset = 0
        for algorithm in literal_unroll(ALGORITHMS):
            payload = coop.ThreadData(items_per_thread=ITEMS, dtype=types.int32)
            coop.load(coop.this_block(), source, payload, algorithm=algorithm)
            base = offset + cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                output[base + i] = payload[i]
            offset += THREADS * ITEMS

    tile = THREADS * ITEMS
    source = np.arange(tile, dtype=np.int32) * 3 + 1
    d_out = cuda.to_device(np.full(tile * len(ALGORITHMS), -1, np.int32))
    loads[1, THREADS](cuda.to_device(source), d_out)
    expected = np.concatenate(
        [_loaded(source, algorithm) for algorithm in ALGORITHMS]
    )
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_a_rebound_dtype_name_types_each_payload_where_it_is_built():
    @cuda.jit
    def rebound(flag, as_int, as_float):
        if flag[0] > 0:
            dtype = types.int32
            first = coop.ThreadData(items_per_thread=2, dtype=dtype)
            first[0] = 16777217
            as_int[0] = first[0]
            dtype = types.float32
            second = coop.ThreadData(items_per_thread=2, dtype=dtype)
            second[0] = 1.5
            as_float[0] = second[0]

    d_int = cuda.to_device(np.zeros(1, np.int64))
    d_float = cuda.to_device(np.zeros(1, np.float64))
    rebound[1, 1](cuda.to_device(np.ones(1, np.int32)), d_int, d_float)
    # 16777217 is not a float32, and 1.5 is not an int32.
    assert d_int.copy_to_host()[0] == 16777217
    assert d_float.copy_to_host()[0] == 1.5
