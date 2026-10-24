# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Warp and logical-warp load and store through the typed pipeline, on the GPU.

Each warp group moves its own tile of ``width * items`` elements; the group
of the thread with x-major rank ``r`` starts at ``(r // width) * width *
items``. Results are checked against NumPy's view of that arrangement:
blocked for ``direct``, ``vectorize`` and ``transpose``, striped within the
group for ``striped``. A ``transpose`` exchanges through its group
instance's slot of the scratch, so the last tests run several in one kernel,
next to other groups' and the block's.
"""

from __future__ import annotations

import numpy as np
import pytest
from numba_cuda_mlir import cuda

import cuda.coop as coop_common
import cuda.coop.numba_mlir as coop_backend

from .._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline

THREADS = 64
ITEMS = 4
ALGORITHMS = ("direct", "striped", "vectorize", "transpose")
PHYSICAL = None
WIDTHS = [PHYSICAL, 1, 2, 4, 8, 16, 32]

_APIS = [
    pytest.param(coop_common, id="common"),
    pytest.param(coop_backend, id="backend"),
]


def _width_id(width):
    return "this_warp" if width is PHYSICAL else f"group_by-{width}"


def _positions(width, algorithm, threads=THREADS, items=ITEMS, offset=0):
    """The element each thread's item ``i`` holds, as a (thread, item) array."""

    w = 32 if width is PHYSICAL else width
    rank = np.arange(threads)[:, None]
    item = np.arange(items)[None, :]
    origin = offset + (rank // w) * w * items
    lane = rank % w
    if algorithm == "striped":
        return origin + item * w + lane
    return origin + lane * items + item


def _load_kernel(coop, width, algorithm):
    if width is PHYSICAL:

        @cuda.jit
        def kernel(source, output):
            items = coop.ThreadData(ITEMS)
            coop.load(coop.this_warp(), source, items, algorithm=algorithm)
            base = cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                output[base + i] = items[i]

    else:

        @cuda.jit
        def kernel(source, output):
            items = coop.ThreadData(ITEMS)
            group = coop.this_warp().group_by(width)
            coop.load(group, source, items, algorithm=algorithm)
            base = cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                output[base + i] = items[i]

    return kernel


def _store_kernel(coop, width, algorithm):
    if width is PHYSICAL:

        @cuda.jit
        def kernel(output):
            items = coop.ThreadData(ITEMS, dtype=output.dtype)
            base = cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                items[i] = base + i + 1
            coop.store(coop.this_warp(), output, items, algorithm=algorithm)

    else:

        @cuda.jit
        def kernel(output):
            items = coop.ThreadData(ITEMS, dtype=output.dtype)
            base = cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                items[i] = base + i + 1
            group = coop.this_warp().group_by(width)
            coop.store(group, output, items, algorithm=algorithm)

    return kernel


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("width", WIDTHS, ids=_width_id)
@pytest.mark.parametrize("coop", _APIS)
def test_each_group_loads_its_own_tile(coop, width, algorithm):
    source = np.arange(THREADS * ITEMS, dtype=np.float32) * 3 + 1
    d_out = cuda.device_array_like(source)
    _load_kernel(coop, width, algorithm)[1, THREADS](
        cuda.to_device(source), d_out
    )
    expected = source[_positions(width, algorithm)].reshape(-1)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("width", WIDTHS, ids=_width_id)
@pytest.mark.parametrize("coop", _APIS)
def test_each_group_stores_its_own_tile(coop, width, algorithm):
    d_out = cuda.to_device(np.zeros(THREADS * ITEMS, np.int32))
    _store_kernel(coop, width, algorithm)[1, THREADS](d_out)
    expected = np.zeros(THREADS * ITEMS, np.int32)
    expected[_positions(width, algorithm)] = (
        np.arange(THREADS * ITEMS).reshape(THREADS, ITEMS) + 1
    )
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def _partial_kernel(width, algorithm, static):
    coop = coop_common
    if width is PHYSICAL and static:

        @cuda.jit
        def kernel(source, output, loaded, valid, fill):
            items = coop.ThreadData(ITEMS, dtype=source.dtype)
            group = coop.this_warp()
            coop.load(
                group,
                source,
                items,
                algorithm=algorithm,
                valid_items=37,
                oob_default=-1.0,
            )
            base = cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                loaded[base + i] = items[i]
            coop.store(
                group, output, items, algorithm=algorithm, valid_items=37
            )

    elif width is PHYSICAL:

        @cuda.jit
        def kernel(source, output, loaded, valid, fill):
            items = coop.ThreadData(ITEMS, dtype=source.dtype)
            group = coop.this_warp()
            count = valid[0]
            coop.load(
                group,
                source,
                items,
                algorithm=algorithm,
                valid_items=count,
                oob_default=fill[0],
            )
            base = cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                loaded[base + i] = items[i]
            coop.store(
                group, output, items, algorithm=algorithm, valid_items=count
            )

    elif static:

        @cuda.jit
        def kernel(source, output, loaded, valid, fill):
            items = coop.ThreadData(ITEMS, dtype=source.dtype)
            group = coop.this_warp().group_by(width)
            coop.load(
                group,
                source,
                items,
                algorithm=algorithm,
                valid_items=5,
                oob_default=-1.0,
            )
            base = cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                loaded[base + i] = items[i]
            coop.store(group, output, items, algorithm=algorithm, valid_items=5)

    else:

        @cuda.jit
        def kernel(source, output, loaded, valid, fill):
            items = coop.ThreadData(ITEMS, dtype=source.dtype)
            group = coop.this_warp().group_by(width)
            count = valid[0]
            coop.load(
                group,
                source,
                items,
                algorithm=algorithm,
                valid_items=count,
                oob_default=fill[0],
            )
            base = cuda.threadIdx.x * ITEMS
            for i in range(ITEMS):
                loaded[base + i] = items[i]
            coop.store(
                group, output, items, algorithm=algorithm, valid_items=count
            )

    return kernel


@pytest.mark.parametrize("static", [True, False], ids=["static", "runtime"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("width", [PHYSICAL, 8], ids=_width_id)
def test_valid_items_and_oob_default_apply_to_each_group_tile(
    width, algorithm, static
):
    valid = 37 if width is PHYSICAL else 5
    w = 32 if width is PHYSICAL else width
    source = np.arange(THREADS * ITEMS, dtype=np.float32) + 1
    sentinel = np.float32(-7)
    d_out = cuda.to_device(np.full_like(source, sentinel))
    d_loaded = cuda.device_array_like(source)
    _partial_kernel(width, algorithm, static)[1, THREADS](
        cuda.to_device(source),
        d_out,
        d_loaded,
        cuda.to_device(np.array([valid], np.int32)),
        cuda.to_device(np.array([-3], np.float32)),
    )

    positions = _positions(width, algorithm)
    in_tile = positions % (w * ITEMS)
    fill = np.float32(-1 if static else -3)
    loaded = np.where(in_tile < valid, source[positions], fill)
    np.testing.assert_array_equal(d_loaded.copy_to_host(), loaded.reshape(-1))
    stored = np.full_like(source, sentinel)
    kept = in_tile < valid
    stored[positions[kept]] = source[positions[kept]]
    np.testing.assert_array_equal(d_out.copy_to_host(), stored)


@pytest.mark.parametrize("static", [True, False], ids=["static", "runtime"])
@pytest.mark.parametrize("width", [PHYSICAL, 4], ids=_width_id)
def test_an_offset_is_added_to_each_group_tile_origin(width, static):
    coop = coop_common
    offset = 3
    if width is PHYSICAL:

        @cuda.jit
        def kernel(source, output, shift):
            items = coop.ThreadData(ITEMS, dtype=source.dtype)
            group = coop.this_warp()
            at = offset if static else shift[0]
            coop.load(group, source, items, offset=at)
            coop.store(group, output, items, offset=at)

    else:

        @cuda.jit
        def kernel(source, output, shift):
            items = coop.ThreadData(ITEMS, dtype=source.dtype)
            group = coop.this_warp().group_by(width)
            at = offset if static else shift[0]
            coop.load(group, source, items, offset=at)
            coop.store(group, output, items, offset=at)

    size = THREADS * ITEMS + offset
    source = np.arange(size, dtype=np.float32) + 1
    d_out = cuda.to_device(np.zeros(size, np.float32))
    kernel[1, THREADS](
        cuda.to_device(source),
        d_out,
        cuda.to_device(np.array([0 if static else offset], np.int64)),
    )
    expected = source.copy()
    expected[:offset] = 0
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


@pytest.mark.parametrize("width", [PHYSICAL, 8], ids=_width_id)
def test_a_multidimensional_block_ranks_threads_x_major(width):
    coop = coop_common
    block = (16, 2, 2)
    if width is PHYSICAL:

        @cuda.jit
        def kernel(source, output):
            items = coop.ThreadData(ITEMS)
            coop.load(coop.this_warp(), source, items)
            rank = cuda.threadIdx.x + 16 * (
                cuda.threadIdx.y + 2 * cuda.threadIdx.z
            )
            for i in range(ITEMS):
                output[rank * ITEMS + i] = items[i]

    else:

        @cuda.jit
        def kernel(source, output):
            items = coop.ThreadData(ITEMS)
            coop.load(coop.this_warp().group_by(width), source, items)
            rank = cuda.threadIdx.x + 16 * (
                cuda.threadIdx.y + 2 * cuda.threadIdx.z
            )
            for i in range(ITEMS):
                output[rank * ITEMS + i] = items[i]

    source = np.arange(THREADS * ITEMS, dtype=np.float32) * 2
    d_out = cuda.device_array_like(source)
    kernel[1, block](cuda.to_device(source), d_out)
    expected = source[_positions(width, "direct")].reshape(-1)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_a_phi_of_two_warp_groups_loads_and_stores_through_either():
    @cuda.jit
    def kernel(source, output):
        if source[0] > 0:
            group = coop_common.this_warp().group_by(8)
        else:
            group = coop_backend.this_warp().group_by(8)
        pair = (group, 1)
        items = coop_common.ThreadData(ITEMS)
        coop_common.load(pair[0], source, items, algorithm="striped")
        for i in range(ITEMS):
            items[i] = items[i] * 2
        coop_common.store(group, output, items, algorithm="striped")

    source = np.arange(THREADS * ITEMS, dtype=np.float32) + 1
    d_out = cuda.device_array_like(source)
    kernel[1, THREADS](cuda.to_device(source), d_out)
    np.testing.assert_array_equal(d_out.copy_to_host(), source * 2)


def test_logical_warps_of_different_widths_in_one_kernel():
    coop = coop_common

    @cuda.jit
    def kernel(source, output):
        items = coop.ThreadData(ITEMS)
        coop.load(coop.this_warp().group_by(4), source, items)
        coop.store(
            coop.this_warp().group_by(16), output, items, algorithm="striped"
        )

    source = np.arange(THREADS * ITEMS, dtype=np.float32) + 1
    d_out = cuda.device_array_like(source)
    kernel[1, THREADS](cuda.to_device(source), d_out)
    expected = np.zeros_like(source)
    expected[_positions(16, "striped")] = source[_positions(4, "direct")]
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


@pytest.mark.parametrize("width", [PHYSICAL, 8], ids=_width_id)
def test_a_transpose_reuses_its_scratch_across_iterations(width):
    coop = coop_common
    rounds = 3
    tile = THREADS * ITEMS
    if width is PHYSICAL:

        @cuda.jit
        def kernel(source, output):
            items = coop.ThreadData(ITEMS, dtype=source.dtype)
            group = coop.this_warp()
            for r in range(rounds):
                coop.load(
                    group, source, items, algorithm="transpose", offset=r * tile
                )
                for i in range(ITEMS):
                    items[i] = items[i] + r
                coop.store(
                    group, output, items, algorithm="transpose", offset=r * tile
                )

    else:

        @cuda.jit
        def kernel(source, output):
            items = coop.ThreadData(ITEMS, dtype=source.dtype)
            group = coop.this_warp().group_by(width)
            for r in range(rounds):
                coop.load(
                    group, source, items, algorithm="transpose", offset=r * tile
                )
                for i in range(ITEMS):
                    items[i] = items[i] + r
                coop.store(
                    group, output, items, algorithm="transpose", offset=r * tile
                )

    source = np.arange(rounds * tile, dtype=np.float32) * 5 + 1
    d_out = cuda.device_array_like(source)
    kernel[1, THREADS](cuda.to_device(source), d_out)
    expected = source + np.repeat(np.arange(rounds, dtype=np.float32), tile)
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_transposes_on_warps_of_different_widths_and_the_block_in_one_kernel():
    coop = coop_common
    tile = THREADS * ITEMS

    @cuda.jit
    def kernel(source, output):
        items = coop.ThreadData(ITEMS, dtype=source.dtype)
        coop.load(coop.this_block(), source, items, algorithm="transpose")
        coop.store(
            coop.this_warp().group_by(16), output, items, algorithm="transpose"
        )
        coop.load(
            coop.this_warp().group_by(4),
            source,
            items,
            algorithm="transpose",
            offset=tile,
        )
        coop.store(
            coop.this_warp(), output, items, algorithm="transpose", offset=tile
        )

    source = np.arange(2 * tile, dtype=np.float32) + 1
    d_out = cuda.device_array_like(source)
    kernel[1, THREADS](cuda.to_device(source), d_out)
    expected = np.zeros_like(source)
    expected[_positions(16, "transpose")] = source[:tile].reshape(
        THREADS, ITEMS
    )
    expected[tile + _positions(PHYSICAL, "transpose")] = source[
        tile + _positions(4, "transpose")
    ]
    np.testing.assert_array_equal(d_out.copy_to_host(), expected)


def test_a_transpose_in_a_multidimensional_block_uses_its_group_slot():
    coop = coop_common
    block = (16, 2, 2)
    offset = 5

    @cuda.jit
    def kernel(source, output):
        items = coop.ThreadData(ITEMS, dtype=source.dtype)
        group = coop.this_warp().group_by(8)
        coop.load(group, source, items, algorithm="transpose", offset=offset)
        coop.store(group, output, items, algorithm="transpose")

    size = THREADS * ITEMS + offset
    source = np.arange(size, dtype=np.float32) * 2 + 1
    d_out = cuda.to_device(np.zeros(THREADS * ITEMS, np.float32))
    kernel[1, block](cuda.to_device(source), d_out)
    np.testing.assert_array_equal(d_out.copy_to_host(), source[offset:])
