# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Payload dtypes that disagree, refused after typing with the planner's class.

A variable that joins ``ThreadData`` payloads must have the dtype of every
payload it joins. Typing gives the join the dtype of its strongest evidence
and leaves each input with its own, so the check runs on the typed IR, before
lowering; the error is the planner's ``CoopSinglePhaseRewriteError``.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from cuda import coop
from cuda.coop.numba_mlir._compiler._rewrite_support import (
    CoopSinglePhaseRewriteError,
)
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import BLOCK, CHIP, requires_typed_pipeline

pytestmark = requires_typed_pipeline

ITEMS = 4
F32 = types.float32[::1]
I32 = types.int32[::1]


def _compile(kernel, *args):
    return testing.compile_kernel(
        kernel, types.void(*args), block=BLOCK, chip=CHIP
    )


def store_i32_a_load_f32_b(src_f32, dst_i32, out_f32, flag):
    block = coop.this_block()
    a = coop.ThreadData(ITEMS)
    b = coop.ThreadData(ITEMS)
    coop.store(block, dst_i32, a)
    coop.load(block, src_f32, b)
    if flag[0]:
        x = a
    else:
        x = b
    coop.store(block, out_f32, x)


def write_i32_a_load_f32_b(src_i32, src_f32, out_f32, flag):
    block = coop.this_block()
    a = coop.ThreadData(ITEMS)
    b = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        a[i] = src_i32[base + i]
    coop.load(block, src_f32, b)
    if flag[0]:
        x = a
    else:
        x = b
    coop.store(block, out_f32, x)


def load_f32_b_write_i32_a(src_i32, src_f32, out_f32, flag):
    block = coop.this_block()
    a = coop.ThreadData(ITEMS)
    b = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    coop.load(block, src_f32, b)
    for i in range(ITEMS):
        a[i] = src_i32[base + i]
    if flag[0]:
        x = a
    else:
        x = b
    coop.store(block, out_f32, x)


def write_range_a_load_f32_b(src_f32, out_f32, flag):
    block = coop.this_block()
    a = coop.ThreadData(ITEMS)
    b = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        a[i] = i
    coop.load(block, src_f32, b)
    if flag[0]:
        x = a
    else:
        x = b
    coop.store(block, out_f32, x)


def write_f64_a_load_i32_b(src_i32, out_i32, flag):
    block = coop.this_block()
    a = coop.ThreadData(ITEMS)
    b = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        a[i] = 2.5
    coop.load(block, src_i32, b)
    if flag[0]:
        x = a
    else:
        x = b
    coop.store(block, out_i32, x)


JOINS = {
    "store_i32_a_load_f32_b": (
        store_i32_a_load_f32_b,
        (F32, I32, F32, I32),
        "int32 from 'a'",
        "float32 from 'b'",
    ),
    "write_i32_a_load_f32_b": (
        write_i32_a_load_f32_b,
        (I32, F32, F32, I32),
        "int32 from 'a'",
        "float32 from 'b'",
    ),
    "load_f32_b_write_i32_a": (
        load_f32_b_write_i32_a,
        (I32, F32, F32, I32),
        "int32 from 'a'",
        "float32 from 'b'",
    ),
    "write_range_a_load_f32_b": (
        write_range_a_load_f32_b,
        (F32, F32, I32),
        "int64 from 'a'",
        "float32 from 'b'",
    ),
    "write_f64_a_load_i32_b": (
        write_f64_a_load_i32_b,
        (I32, I32, I32),
        "float64 from 'a'",
        "int32 from 'b'",
    ),
}


@pytest.mark.parametrize("name", JOINS)
def test_a_join_of_payloads_of_different_dtypes(name):
    kernel, args, first, second = JOINS[name]
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            rf"coop\.ThreadData 'x' has dtype \w+, but it joins payloads of "
            rf"different dtypes: {first} \(line \d+\), {second} \(line \d+\)"
        ),
    ) as raised:
        _compile(kernel, *args)
    assert isinstance(raised.value, TypedPipelineError)


def test_a_join_of_payloads_of_one_dtype_and_different_evidence():
    def kernel(src_i32, src_i32b, out_i32, flag):
        block = coop.this_block()
        a = coop.ThreadData(ITEMS)
        b = coop.ThreadData(ITEMS)
        base = cuda.threadIdx.x * ITEMS
        for i in range(ITEMS):
            a[i] = src_i32[base + i]
        coop.load(block, src_i32b, b)
        if flag[0]:
            x = a
        else:
            x = b
        coop.store(block, out_i32, x)

    _compile(kernel, I32, I32, I32, I32)


def test_a_join_loaded_after_one_input_was_stored():
    def kernel(src_f32, dst_i32, out_f32, flag):
        block = coop.this_block()
        a = coop.ThreadData(ITEMS)
        b = coop.ThreadData(ITEMS)
        coop.store(block, dst_i32, a)
        if flag[0]:
            x = a
        else:
            x = b
        coop.load(block, src_f32, x)
        coop.store(block, out_f32, x)

    # Whether the join's load retypes 'a' before its store is typed depends
    # on what the process has already compiled; either way it is refused.
    with pytest.raises(
        TypingError,
        match=(
            r"payload dtype float32 conflicts with destination dtype int32"
            r"|int32 from 'a' \(line \d+\), float32 from 'b'"
        ),
    ):
        _compile(kernel, F32, I32, F32, I32)


def test_a_join_loaded_after_one_input_was_written():
    """Refused although the load overwrites the writes, which coop's planner
    accepts: typing does not retype a written input from its join."""

    def kernel(src_i32, src_f32, out_f32, flag):
        block = coop.this_block()
        a = coop.ThreadData(ITEMS)
        b = coop.ThreadData(ITEMS)
        base = cuda.threadIdx.x * ITEMS
        for i in range(ITEMS):
            a[i] = src_i32[base + i]
        if flag[0]:
            x = a
        else:
            x = b
        coop.load(block, src_f32, x)
        coop.store(block, out_f32, x)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=r"int32 from 'a' \(line \d+\), float32 from 'b'",
    ):
        _compile(kernel, I32, F32, F32, I32)
