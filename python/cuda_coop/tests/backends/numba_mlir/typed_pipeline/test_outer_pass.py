# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The outer typing pass: payload types decided around Numba's inference."""

from __future__ import annotations

import contextlib
import re

import numpy as np
from numba_cuda_mlir import cuda, types

from cuda import coop

from ._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


@contextlib.contextmanager
def _counted_solves():
    from numba_cuda_mlir import mlir_compiler

    stage = mlir_compiler.type_inference_stage
    solves = []

    def counting(typingctx, targetctx, interp, *args, **kwargs):
        solves.append(
            (interp.func_id.func_name, kwargs.get("raise_errors", True))
        )
        return stage(typingctx, targetctx, interp, *args, **kwargs)

    mlir_compiler.type_inference_stage = counting
    try:
        yield solves
    finally:
        mlir_compiler.type_inference_stage = stage


def _compile(kernel):
    from cuda.coop.numba_mlir._pipeline import testing

    return testing.compile_kernel(kernel, f32_in_out(), block=BLOCK, chip=CHIP)


def test_a_read_before_the_load_types_to_the_load_dtype():
    def kernel(source, destination):
        payload = coop.ThreadData(items_per_thread=1)
        payload[0] = np.float64(1.25)
        observed = payload[0]
        coop.load(coop.this_block(), source, payload)
        destination[0] = observed + payload[0]

    with _counted_solves() as solves:
        compiled = _compile(kernel)
    assert re.search(r"\bobserved = .*:: float32", compiled.typed_ir)
    assert solves == [("kernel", True), ("kernel", True)]


def test_a_failed_first_typing_is_decided_from_a_partial_typing():
    def kernel(source, out):
        items = coop.ThreadData(4)
        items[0] = 2.5
        acc = 0
        for v in items:
            acc += v
        coop.load(coop.this_block(), source, items)
        out[cuda.threadIdx.x] = acc

    with _counted_solves() as solves:
        compiled = _compile(kernel)
    assert re.search(r"\bv = .*:: float32", compiled.typed_ir)
    assert solves == [("kernel", True), ("kernel", False), ("kernel", True)]


def test_a_payload_whose_reads_agree_is_typed_once():
    def kernel(source, out):
        items = coop.ThreadData(4, dtype=np.float32)
        items[0] = 1.0
        first = items[0]
        coop.load(coop.this_block(), source, items)
        out[0] = first + items[1]

    with _counted_solves() as solves:
        _compile(kernel)
    assert solves == [("kernel", True)]


def test_a_plain_kernel_is_typed_once_by_the_stock_pass():
    from numba_cuda_mlir import compiler, mlir_compiler

    from cuda.coop.numba_mlir._pipeline import _ncm_patches

    def saxpy(x, y, out, a):
        i = cuda.grid(1)
        if i < out.size:
            out[i] = a * x[i] + y[i]

    f32 = types.float32[::1]
    sig = types.void(f32, f32, f32, types.float32)

    def mlir():
        return str(compiler.compile_mlir(cuda.jit(chip=CHIP)(saxpy), sig))

    with _counted_solves() as solves:
        patched = mlir()
    stock_run_pass = next(
        old for _, name, old in _ncm_patches._replaced if name == "run_pass"
    )
    patched_run_pass = mlir_compiler.MLIRTypeInference.run_pass
    mlir_compiler.MLIRTypeInference.run_pass = stock_run_pass
    try:
        stock = mlir()
    finally:
        mlir_compiler.MLIRTypeInference.run_pass = patched_run_pass
    assert solves == [("saxpy", True)]
    assert patched == stock


def test_the_reviewers_kernel_reads_the_loaded_value():
    @cuda.jit
    def kernel(source, out):
        payload = coop.ThreadData(items_per_thread=1)
        payload[0] = np.float64(1.25)
        observed = payload[0]
        coop.load(coop.this_block(), source, payload)
        if cuda.threadIdx.x == 0:
            out[0] = observed + payload[0]

    out = np.zeros(1, np.float32)
    kernel[1, 32](np.full(32, 1.0, np.float32), out)
    assert out[0] == 2.25
