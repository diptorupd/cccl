# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Payload dtypes that disagree, refused after typing with the planner's class.

A variable that joins ``ThreadData`` payloads must have the dtype of every
payload it joins. Typing gives the join the dtype of its strongest evidence
and leaves each input with its own, so the check runs on the typed IR, before
lowering; the error is the planner's ``CoopSinglePhaseRewriteError``.

A payload whose dtype comes from the destination of a store must have been
written with values of that dtype; the error is the planner's
``MemoryDtypeMismatchError``. Every write counts, including writes of a
``range`` loop variable and writes through a tuple projection, which the
planner cannot type before inference and lets the destination convert. A
``load`` or ``dtype=`` converts the writes, as on the planner.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from cuda import coop
from cuda.coop.numba_mlir._compiler._group_errors import (
    MemoryDtypeMismatchError,
)
from cuda.coop.numba_mlir._compiler._rewrite_support import (
    CoopSinglePhaseRewriteError,
)
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import BLOCK, CHIP, requires_fix, requires_typed_pipeline

pytestmark = requires_typed_pipeline

ITEMS = 4
F32 = types.float32[::1]
F64 = types.float64[::1]
I32 = types.int32[::1]
I64 = types.int64[::1]


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


@requires_fix("PHI_REFINEMENT")
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


# Writes and a store destination ----------------------------------------------


def w_i32_then_store_f32(src_i32, dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src_i32[base + i]
    coop.store(block, dst_f32, td)


def w_f32_then_store_i32(src_f32, dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src_f32[base + i]
    coop.store(block, dst_i32, td)


def store_f32_before_w_i32(src_i32, dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for it in range(2):
        if it == 1:
            coop.store(block, dst_f32, td)
        else:
            for i in range(ITEMS):
                td[i] = src_i32[base + i]


def lit_2p5_then_store_i32(dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        td[i] = 2.5
    coop.store(block, dst_i32, td)


def lit_2p5_then_store_f32(dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        td[i] = 2.5
    coop.store(block, dst_f32, td)


def lit_7_then_store_f32(dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        td[i] = 7
    coop.store(block, dst_f32, td)


def alias_w_i32_then_store_f32(src_i32, dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    alias = td
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        alias[i] = src_i32[base + i]
    coop.store(block, dst_f32, td)


def w_i32_then_store_f32_via_alias(src_i32, dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src_i32[base + i]
    alias = td
    coop.store(block, dst_f32, alias)


def alias_lit_2p5_then_store_i32(dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    alias = td
    for i in range(ITEMS):
        alias[i] = 2.5
    coop.store(block, dst_i32, td)


def phi_w_i32_a_store_f32_a_load_f32_b(
    src_i32, src_f32, dst_f32, out_f32, flag
):
    block = coop.this_block()
    a = coop.ThreadData(ITEMS)
    b = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        a[i] = src_i32[base + i]
    coop.store(block, dst_f32, a)
    coop.load(block, src_f32, b)
    if flag[0]:
        x = a
    else:
        x = b
    coop.store(block, out_f32, x)


def phi_w_i32_a_store_f32_join(src_i32, dst_f32, flag):
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
    coop.store(block, dst_f32, x)


def w_i32_expr_then_store_f32(src_i32, dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src_i32[base + i] + 1
    coop.store(block, dst_f32, td)


def w_f32_expr_then_store_i32(src_f32, dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src_f32[base + i] * 2
    coop.store(block, dst_i32, td)


def range_i_then_store_f32(dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        td[i] = i
    coop.store(block, dst_f32, td)


def w_i32_and_range_then_store_i32(src_i32, dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    td[0] = src_i32[base]
    for i in range(1, ITEMS):
        td[i] = i
    coop.store(block, dst_i32, td)


def tuple_w_i32_then_store_f32(src_i32, dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    alias = (td,)[0]
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        alias[i] = src_i32[base + i]
    coop.store(block, dst_f32, td)


def w_i32_then_store_f32_via_tuple(src_i32, dst_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src_i32[base + i]
    coop.store(block, dst_f32, (td,)[0])


def tuple_lit_2p5_then_store_i32(dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    alias = (td,)[0]
    for i in range(ITEMS):
        alias[i] = 2.5
    coop.store(block, dst_i32, td)


def tuple_w_f32_then_store_i32(src_f32, dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    alias = (td,)[0]
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        alias[i] = src_f32[base + i]
    coop.store(block, dst_i32, td)


# Without the fix a store of a tuple projection does not compile, and a load
# through one does not reach the payload, so the store destination types it.
NEEDS_TUPLE_REFINEMENT = {
    "w_i32_then_store_f32_via_tuple",
    "tuple_load_f32_then_w_i32",
}


def _params(table):
    return [
        pytest.param(
            name,
            marks=(
                (requires_fix("TUPLE_REFINEMENT"),)
                if name in NEEDS_TUPLE_REFINEMENT
                else ()
            ),
        )
        for name in table
    ]


def _mismatch(memory, payload, written):
    message = (
        rf"coop\.store memory dtype {memory} does not match the dtype of the "
        rf"writes to coop\.ThreadData '{payload}' \({written} at line \d+\)"
    )
    return MemoryDtypeMismatchError, message


# Kernels coop's planner refuses too, then kernels it accepts only because it
# cannot type the writes before inference.
STORES = {
    "w_i32_then_store_f32": (
        w_i32_then_store_f32,
        (I32, F32),
        _mismatch("float32", "td", "int32"),
    ),
    "w_f32_then_store_i32": (
        w_f32_then_store_i32,
        (F32, I32),
        _mismatch("int32", "td", "float32"),
    ),
    "store_f32_before_w_i32": (
        store_f32_before_w_i32,
        (I32, F32),
        _mismatch("float32", "td", "int32"),
    ),
    "lit_2p5_then_store_i32": (
        lit_2p5_then_store_i32,
        (I32,),
        _mismatch("int32", "td", "float64"),
    ),
    "lit_2p5_then_store_f32": (
        lit_2p5_then_store_f32,
        (F32,),
        _mismatch("float32", "td", "float64"),
    ),
    "lit_7_then_store_f32": (
        lit_7_then_store_f32,
        (F32,),
        _mismatch("float32", "td", "int64"),
    ),
    "alias_w_i32_then_store_f32": (
        alias_w_i32_then_store_f32,
        (I32, F32),
        _mismatch("float32", "alias", "int32"),
    ),
    "w_i32_then_store_f32_via_alias": (
        w_i32_then_store_f32_via_alias,
        (I32, F32),
        _mismatch("float32", "td", "int32"),
    ),
    "alias_lit_2p5_then_store_i32": (
        alias_lit_2p5_then_store_i32,
        (I32,),
        _mismatch("int32", "alias", "float64"),
    ),
    "phi_w_i32_a_store_f32_a_load_f32_b": (
        phi_w_i32_a_store_f32_a_load_f32_b,
        (I32, F32, F32, F32, I32),
        _mismatch("float32", "a", "int32"),
    ),
    # The join takes the store's dtype; 'a' keeps its writes' dtype.
    "phi_w_i32_a_store_f32_join": (
        phi_w_i32_a_store_f32_join,
        (I32, F32, I32),
        (CoopSinglePhaseRewriteError, r"int32 from 'a' \(line \d+\)"),
    ),
    "w_i32_expr_then_store_f32": (
        w_i32_expr_then_store_f32,
        (I32, F32),
        _mismatch("float32", "td", "int64"),
    ),
    "w_f32_expr_then_store_i32": (
        w_f32_expr_then_store_i32,
        (F32, I32),
        _mismatch("int32", "td", "float64"),
    ),
    "range_i_then_store_f32": (
        range_i_then_store_f32,
        (F32,),
        _mismatch("float32", "td", "int64"),
    ),
    "w_i32_and_range_then_store_i32": (
        w_i32_and_range_then_store_i32,
        (I32, I32),
        _mismatch("int32", "td", "int64"),
    ),
    "tuple_w_i32_then_store_f32": (
        tuple_w_i32_then_store_f32,
        (I32, F32),
        _mismatch("float32", "alias", "int32"),
    ),
    "w_i32_then_store_f32_via_tuple": (
        w_i32_then_store_f32_via_tuple,
        (I32, F32),
        _mismatch("float32", "td", "int32"),
    ),
    "tuple_lit_2p5_then_store_i32": (
        tuple_lit_2p5_then_store_i32,
        (I32,),
        _mismatch("int32", "alias", "float64"),
    ),
    "tuple_w_f32_then_store_i32": (
        tuple_w_f32_then_store_i32,
        (F32, I32),
        _mismatch("int32", "alias", "float32"),
    ),
}


@pytest.mark.parametrize("name", _params(STORES))
def test_writes_that_disagree_with_the_store_destination(name):
    kernel, args, (error, message) = STORES[name]
    with pytest.raises(error, match=message) as raised:
        _compile(kernel, *args)
    assert isinstance(raised.value, TypedPipelineError)


def ctl_w_i32_then_store_i32(src_i32, dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src_i32[base + i]
    coop.store(block, dst_i32, td)


def ctl_lit_2p5_then_store_f64(dst_f64):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        td[i] = 2.5
    coop.store(block, dst_f64, td)


def ctl_range_i_then_store_i64(dst_i64):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        td[i] = i
    coop.store(block, dst_i64, td)


def ctl_tuple_w_i32_then_store_i32(src_i32, dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    alias = (td,)[0]
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        alias[i] = src_i32[base + i]
    coop.store(block, dst_i32, td)


def load_i32_then_lit_2p5(src_i32, out_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src_i32, td)
    for i in range(1, ITEMS):
        td[i] = 2.5
    coop.store(block, out_i32, td)


def load_f32_then_range_i(src_f32, out_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src_f32, td)
    for i in range(1, ITEMS):
        td[i] = i
    coop.store(block, out_f32, td)


def alias_load_f32_then_alias_w_i32(src_i32, src_f32, out_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    alias = td
    base = cuda.threadIdx.x * ITEMS
    coop.load(block, src_f32, td)
    for i in range(1, ITEMS):
        alias[i] = src_i32[base + i]
    coop.store(block, out_f32, td)


def tuple_load_f32_then_w_i32(src_i32, src_f32, out_f32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    coop.load(block, src_f32, (td,)[0])
    for i in range(1, ITEMS):
        td[i] = src_i32[base + i]
    coop.store(block, out_f32, td)


def explicit_i32_w_f32_store_i32(src_f32, dst_i32):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS, dtype=types.int32)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src_f32[base + i]
    coop.store(block, dst_i32, td)


ACCEPTED = {
    "ctl_w_i32_then_store_i32": (ctl_w_i32_then_store_i32, (I32, I32)),
    "ctl_lit_2p5_then_store_f64": (ctl_lit_2p5_then_store_f64, (F64,)),
    "ctl_range_i_then_store_i64": (ctl_range_i_then_store_i64, (I64,)),
    "ctl_tuple_w_i32_then_store_i32": (
        ctl_tuple_w_i32_then_store_i32,
        (I32, I32),
    ),
    "load_i32_then_lit_2p5": (load_i32_then_lit_2p5, (I32, I32)),
    "load_f32_then_range_i": (load_f32_then_range_i, (F32, F32)),
    "alias_load_f32_then_alias_w_i32": (
        alias_load_f32_then_alias_w_i32,
        (I32, F32, F32),
    ),
    "tuple_load_f32_then_w_i32": (tuple_load_f32_then_w_i32, (I32, F32, F32)),
    "explicit_i32_w_f32_store_i32": (explicit_i32_w_f32_store_i32, (F32, I32)),
}


@pytest.mark.parametrize("name", _params(ACCEPTED))
def test_writes_a_store_destination_or_a_load_accepts(name):
    kernel, args = ACCEPTED[name]
    _compile(kernel, *args)
