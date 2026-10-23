# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A refusal coop's planner also makes raises the planner's exception class.

Each case is a kernel the planner refuses, with the class it raises for it.
The typed pipeline refuses the same kernel with an error of that class that
is also a ``TypedPipelineError``, so it keeps its source location. The
message is the typed pipeline's own; each case checks only the part that
names the cause.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda, types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop
from cuda.coop.numba_mlir._compiler._group_errors import (
    CommonLoadPayloadError,
    CommonStorePayloadError,
    DefaultDtypeMismatchError,
    InconsistentLoopPayloadExtentError,
    InvalidGroupSelectorError,
    InvalidLoadStoreAlgorithmError,
    MemoryDtypeMismatchError,
    NonConstantThreadGroupError,
    UnknownLoadStoreExtentError,
    UnsupportedLoadStoreGroupError,
)
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import BLOCK, CHIP, requires_typed_pipeline

pytestmark = requires_typed_pipeline

f32 = types.float32[::1]
i32 = types.int32[::1]


def common_local_array_load(source, output):
    items = cuda.local.array(4, types.float32)
    coop.load(coop.this_block(), source, items)
    output[0] = items[0]


def common_local_array_store(source, output):
    items = cuda.local.array(4, types.float32)
    for i in range(4):
        items[i] = source[i]
    coop.store(coop.this_block(), output, items)


def common_scalar_load_output(source, output):
    value = source[0]
    coop.load(coop.this_block(), source, value)
    output[0] = value


def qualified_scalar_load_output(source, output):
    value = source[0]
    qualified_coop.load(qualified_coop.this_block(), source, value)
    output[0] = value


def payload_and_memory_dtypes_differ(source, output):
    items = coop.ThreadData(4, dtype=types.int32)
    coop.load(coop.this_block(), source, items)
    coop.store(coop.this_block(), output, items)


def explicit_payload_stored_to_another_dtype(source, output):
    items = coop.ThreadData(4, dtype=types.float32)
    for i in range(4):
        items[i] = 1
    coop.store(coop.this_block(), output, items)


def runtime_store_value_of_another_dtype(source, output):
    coop.store(coop.this_block(), output, source[cuda.threadIdx.x])


_SCALAR_SOURCE_REFUSAL = (
    r"source must be an array, got float64, whose dtype also conflicts with "
    r"payload dtype float32"
)


def source_is_a_scalar(source, output):
    items = coop.ThreadData(4, dtype=types.float32)
    coop.load(coop.this_block(), 3.0, items)
    coop.store(coop.this_block(), output, items)


def common_unknown_algorithm(source, output):
    items = coop.ThreadData(4)
    coop.load(coop.this_block(), source, items, algorithm="diagonal")
    coop.store(coop.this_block(), output, items)


def qualified_unknown_algorithm(source, output):
    items = qualified_coop.ThreadData(4)
    block = qualified_coop.this_block()
    qualified_coop.load(block, source, items, algorithm="diagonal")
    qualified_coop.store(block, output, items)


def group_is_an_integer(source, output):
    items = coop.ThreadData(4)
    coop.load(1, source, items)
    coop.store(coop.this_block(), output, items)


def float16_payload(source, output):
    items = coop.ThreadData(4)
    coop.load(coop.this_block(), source, items)
    coop.store(coop.this_block(), output, items)


def warp_transpose_in_partial_warps(source, output):
    items = coop.ThreadData(4)
    coop.load(coop.this_block(), source, items, algorithm="warp_transpose")
    coop.store(coop.this_block(), output, items)


def static_valid_items_float(source, output):
    items = coop.ThreadData(4)
    coop.load(coop.this_block(), source, items, valid_items=1.5)
    coop.store(coop.this_block(), output, items)


def static_valid_items_beyond_the_tile(source, output):
    items = coop.ThreadData(4)
    coop.load(coop.this_block(), source, items, valid_items=513)
    coop.store(coop.this_block(), output, items)


def negative_static_offset(source, output):
    items = coop.ThreadData(4)
    coop.load(coop.this_block(), source, items)
    coop.store(coop.this_block(), output, items, offset=-4)


def runtime_oob_default_of_another_dtype(source, output, pad):
    items = coop.ThreadData(4)
    coop.load(coop.this_block(), source, items, valid_items=7, oob_default=pad)
    coop.store(coop.this_block(), output, items)


def bool_store_value(source, output):
    coop.store(coop.this_block(), output, True)


def float_store_value_into_integers(source, output):
    coop.store(coop.this_block(), output, 2.5)


def store_value_out_of_range(source, output):
    coop.store(coop.this_block(), output, 1 << 40)


def qualified_payload_without_an_extent(source, output):
    qualified_coop.store(qualified_coop.this_block(), output, source)


def qualified_payloads_of_two_extents(source, output, flag):
    if flag[0]:
        items = cuda.local.array(2, types.float32)
    else:
        items = cuda.local.array(4, types.float32)
    qualified_coop.load(qualified_coop.this_block(), source, items)
    output[0] = items[0]


CASES = [
    (common_local_array_load, (f32, f32), CommonLoadPayloadError, None),
    (common_local_array_store, (f32, f32), CommonStorePayloadError, None),
    (common_scalar_load_output, (f32, f32), CommonLoadPayloadError, None),
    (
        qualified_scalar_load_output,
        (f32, f32),
        TypeError,
        r"output must be a coop\.ThreadData or a fixed-size local array",
    ),
    (
        payload_and_memory_dtypes_differ,
        (f32, f32),
        MemoryDtypeMismatchError,
        r"payload dtype int32 conflicts with source dtype float32",
    ),
    (
        explicit_payload_stored_to_another_dtype,
        (f32, i32),
        MemoryDtypeMismatchError,
        r"payload dtype float32 conflicts with destination dtype int32",
    ),
    (
        runtime_store_value_of_another_dtype,
        (f32, i32),
        MemoryDtypeMismatchError,
        r"value dtype float32 does not match destination dtype int32",
    ),
    (
        source_is_a_scalar,
        (f32, f32),
        MemoryDtypeMismatchError,
        _SCALAR_SOURCE_REFUSAL,
    ),
    (common_unknown_algorithm, (f32, f32), InvalidGroupSelectorError, None),
    (
        qualified_unknown_algorithm,
        (f32, f32),
        InvalidLoadStoreAlgorithmError,
        r"algorithm for block groups must be one of: direct, striped",
    ),
    (group_is_an_integer, (f32, f32), NonConstantThreadGroupError, None),
    (
        float16_payload,
        (types.float16[::1], types.float16[::1]),
        TypeError,
        r"cuda\.coop\.load supports dtypes int8, .* through the common API",
    ),
    (
        warp_transpose_in_partial_warps,
        (f32, f32),
        UnsupportedLoadStoreGroupError,
        r"requires a block size that is a multiple of 32",
    ),
    (
        static_valid_items_float,
        (f32, f32),
        TypeError,
        r"valid_items must be an integer",
    ),
    (
        static_valid_items_beyond_the_tile,
        (f32, f32),
        ValueError,
        r"static valid_items must be between zero and the group tile size",
    ),
    (
        negative_static_offset,
        (f32, f32),
        ValueError,
        r"static offset must be nonnegative",
    ),
    (
        runtime_oob_default_of_another_dtype,
        (f32, f32, types.float64),
        DefaultDtypeMismatchError,
        r"runtime oob_default dtype float64 does not match payload dtype",
    ),
    (bool_store_value, (f32, i32), TypeError, r"value must not be bool"),
    (
        float_store_value_into_integers,
        (f32, i32),
        TypeError,
        r"does not permit float-to-integer conversion",
    ),
    (
        store_value_out_of_range,
        (f32, i32),
        ValueError,
        r"is outside the range of int32",
    ),
    (
        qualified_payload_without_an_extent,
        (f32, f32),
        UnknownLoadStoreExtentError,
        r"requires a static items_per_thread extent",
    ),
    (
        qualified_payloads_of_two_extents,
        (f32, f32, i32),
        InconsistentLoopPayloadExtentError,
        r"inconsistent items_per_thread extents",
    ),
]

_BLOCKS = {warp_transpose_in_partial_warps: (48, 1, 1)}


@pytest.mark.parametrize(
    "kernel, arg_types, planner_class, message",
    CASES,
    ids=[case[0].__name__ for case in CASES],
)
def test_a_refusal_has_the_planner_class(
    kernel, arg_types, planner_class, message
):
    with pytest.raises(planner_class, match=message) as info:
        testing.compile_kernel(
            kernel,
            types.void(*arg_types),
            block=_BLOCKS.get(kernel, BLOCK),
            chip=CHIP,
        )
    assert isinstance(info.value, TypedPipelineError)


_OTHER_GROUPS = ["this_thread", "this_cluster", "this_grid"]
_APIS = {"common": coop, "qualified": qualified_coop}


@pytest.mark.parametrize("constructor", _OTHER_GROUPS)
@pytest.mark.parametrize("api", _APIS)
def test_a_kernel_may_construct_a_group_load_and_store_refuse(api, constructor):
    make_group = getattr(_APIS[api], constructor)

    def unused(source, output):
        make_group()
        output[0] = source[0]

    testing.compile_kernel(unused, types.void(f32, f32), block=BLOCK, chip=CHIP)


_COMMON_GROUP_REFUSAL = (
    r"cuda\.coop\.load does not support group kind '{kind}' in the common API"
)
_QUALIFIED_GROUP_REFUSAL = (
    r"cuda\.coop\.numba_mlir\.load cannot lower this group: cuda\.coop "
    r"Load/Store supports this_block\(\)"
)


@pytest.mark.parametrize("constructor", _OTHER_GROUPS)
@pytest.mark.parametrize(
    "api, planner_class, message",
    [
        ("common", NotImplementedError, _COMMON_GROUP_REFUSAL),
        ("qualified", UnsupportedLoadStoreGroupError, _QUALIFIED_GROUP_REFUSAL),
    ],
    ids=["common", "qualified"],
)
def test_a_load_on_a_group_other_than_the_block(
    api, planner_class, message, constructor
):
    module = _APIS[api]
    make_group = getattr(module, constructor)

    def kernel(source, output):
        items = module.ThreadData(4)
        module.load(make_group(), source, items)
        module.store(module.this_block(), output, items)

    kind = constructor.removeprefix("this_")
    with pytest.raises(planner_class, match=message.format(kind=kind)) as info:
        testing.compile_kernel(
            kernel, types.void(f32, f32), block=BLOCK, chip=CHIP
        )
    assert isinstance(info.value, TypedPipelineError)
