# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What the typed pipeline refuses for ``TempStorage``, and how it says so.

Each refusal is the planner's: its message and its class, which the typed
error also is (``GroupRewriteError`` for a ``temp_storage=`` that is not
bound to one descriptor, ``CoopSinglePhaseRewriteError`` otherwise).
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from cuda import coop
from cuda.coop.numba_mlir._compiler import _rewrite_storage
from cuda.coop.numba_mlir._compiler._group_planner_support import (
    GroupRewriteError,
)
from cuda.coop.numba_mlir._compiler._rewrite_support import (
    CoopSinglePhaseRewriteError,
)
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import (
    SharedMemoryAliasError,
    TempStorageBindingError,
    TempStorageError,
)

from .._support import (
    BLOCK,
    CHIP,
    f32_in_out,
    native,
    requires_typed_pipeline,
)

pytestmark = requires_typed_pipeline

STATIC_LIMIT = 48 * 1024


def _compile(kernel, sharedmem=0):
    return testing.compile_kernel(
        kernel, f32_in_out(), block=BLOCK, sharedmem=sharedmem, chip=CHIP
    )


@pytest.fixture
def device_limits(monkeypatch):
    monkeypatch.setattr(
        _rewrite_storage,
        "_query_device_shared_memory_limits",
        lambda: {
            "max_default_shared_memory_per_block": STATIC_LIMIT,
            "max_optin_shared_memory_per_block": 96 * 1024,
        },
    )


def test_typed_errors_are_the_planners_classes():
    assert issubclass(TempStorageBindingError, GroupRewriteError)
    assert issubclass(TempStorageError, CoopSinglePhaseRewriteError)


def test_a_path_that_binds_none_is_refused():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = None
        if source.shape[0] > 0:
            storage = coop.TempStorage()
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    with pytest.raises(
        TempStorageBindingError,
        match=r"must be bound to a TempStorage descriptor on every path; .* "
        r"is also bound to a non-descriptor value such as None",
    ):
        _compile(kernel)


def test_joined_descriptors_must_have_one_contract():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        if source.shape[0] > 0:
            storage = coop.TempStorage(4096)
        else:
            storage = coop.TempStorage(8192)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    with pytest.raises(
        TempStorageBindingError,
        match="TempStorage aliases have inconsistent contracts",
    ):
        _compile(kernel)


def test_manual_sync_descriptor_has_one_constructor():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        if source.shape[0] > 0:
            storage = coop.TempStorage(auto_sync=False)
        else:
            storage = coop.TempStorage(auto_sync=False)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    with pytest.raises(
        TempStorageBindingError,
        match=r"auto_sync=False must be constructed at exactly one site; "
        r"'storage' reaches 2 constructor sites",
    ):
        _compile(kernel)


def test_a_descriptor_no_call_consumes_is_refused():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        spare = coop.TempStorage()  # noqa: F841
        coop.load(block, source, items)
        coop.store(block, output, items)

    with pytest.raises(
        TempStorageError,
        match=r"constructor\(s\) 'spare' have no primitive consumer",
    ):
        _compile(kernel)


def test_a_descriptor_passed_to_a_separate_device_function_is_refused():
    @cuda.jit(device=True, inline="never")
    def helper(source, items, storage):
        coop.load(
            coop.this_block(),
            source,
            items,
            algorithm="transpose",
            temp_storage=storage,
        )

    def kernel(source, output):
        items = coop.ThreadData(4)
        storage = coop.TempStorage()
        helper(source, items, storage)
        coop.store(coop.this_block(), output, items)

    # The refusal is raised while typing the helper, so it reaches the
    # kernel as Numba's TypingError around the planner's message.
    with pytest.raises(
        TypingError,
        match=r"TempStorage descriptor 'storage' is passed to a device "
        r"function that was not inlined into this kernel",
    ):
        _compile(kernel)


def test_options_must_be_compile_time_literals():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage(source.shape[0])
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    with pytest.raises(
        TempStorageError,
        match="TempStorage size_in_bytes must be a compile-time literal",
    ):
        _compile(kernel)


def test_an_explicit_size_must_cover_its_calls():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage(32)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    with pytest.raises(
        TempStorageError,
        match=r"TempStorage size_in_bytes is smaller than required by "
        r"primitive uses \(32 < \d+\)",
    ):
        _compile(kernel)


def _dynamic_kernel(alignment=None):
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage(64 * 1024, alignment=alignment)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    return kernel


def test_a_dynamic_backing_the_launch_omits_compiles_exactly_with_the_fix(
    device_limits,
):
    if native("FIRST_LAUNCH_SHARED_MEMORY"):
        compiled = _compile(_dynamic_kernel())
        assert "backing dynamic size=65536 align=16" in compiled.plan
        return
    with pytest.raises(
        TempStorageError,
        match=r"cooperative scratch needs 65536 bytes of dynamic shared "
        r"memory, .* but the launch requests 0\. Applying a compile's dynamic "
        r"shared-memory requirement to the launch that compiles the kernel "
        r"needs Numba-CUDA-MLIR's first-launch dynamic shared memory fix "
        r"\(not yet proposed upstream\)",
    ):
        _compile(_dynamic_kernel())


def test_a_dynamic_backing_refuses_wider_alignment(device_limits):
    with pytest.raises(TempStorageError, match="only 16-byte alignment"):
        _compile(_dynamic_kernel(alignment=32), sharedmem=64 * 1024)


def test_a_dynamic_backing_refuses_user_shared_arrays(device_limits):
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage(64 * 1024)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        mine = cuda.shared.array(32, types.float32)
        mine[0] = items[0]
        coop.store(block, output, items)

    with pytest.raises(
        SharedMemoryAliasError,
        match=r"65536-byte dynamic shared-memory backing, but this kernel "
        r"also declares static cuda\.shared\.array",
    ):
        _compile(kernel, sharedmem=64 * 1024)


def test_a_backing_above_the_optin_limit_is_refused(device_limits):
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage(128 * 1024)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    with pytest.raises(
        TempStorageError,
        match="requires 131072 bytes dynamic shared memory, but device max "
        "opt-in is 98304 bytes",
    ):
        _compile(kernel, sharedmem=128 * 1024)
