# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Where explicit ``TempStorage`` puts each call's scratch, read from dumps.

As on coop's planner, each descriptor with a call that takes scratch is a
region, in constructor order and before the implicit one; a shared
descriptor's calls reuse one slice, an exclusive one's get their own; and
only ``auto_sync=True`` asks for barriers. Scratch above the static limit is
a dynamic backing.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop
from cuda.coop.numba_mlir._compiler import _rewrite_storage
from cuda.coop.numba_mlir._pipeline import testing

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def test_descriptors_are_regions_in_constructor_order_before_implicit():
    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        first = coop.TempStorage()
        second = qualified_coop.TempStorage(4096, alignment=32)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=second
        )
        coop.store(
            block, output, items, algorithm="transpose", temp_storage=first
        )
        coop.load(block, source, items, algorithm="transpose")

    testing.check(
        k1,
        f32_in_out(),
        """
        CHECK-LABEL: coop-plan k1
        CHECK:         backing static size=8224 align=32
        CHECK-NEXT:    region explicit.0 offset=0 size=2064 align=16
        CHECK-NEXT:    region explicit.1 offset=2080 size=4096 align=32
        CHECK-NEXT:    region implicit.0 offset=6176 size=2048 align=16
        CHECK-NEXT:    site #0 load {{.*}} region=explicit.1{{$}}
        CHECK-NEXT:    site #1 store {{.*}} region=explicit.0{{$}}
        CHECK-NEXT:    site #2 load {{.*}} region=implicit.0{{$}}
        CHECK-NEXT:    barrier after #2
        CHECK-NOT:     barrier
        """,
        block=BLOCK,
        chip=CHIP,
    )


@pytest.mark.parametrize(
    "sharing, store_offset", [("shared", ""), ("exclusive", "+2048")]
)
def test_sharing_decides_whether_calls_reuse_a_slice(sharing, store_offset):
    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage(sharing=sharing)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(
            block, output, items, algorithm="transpose", temp_storage=storage
        )

    size = 2064 if sharing == "shared" else 4112
    checks = """
        CHECK-LABEL: coop-plan k1
        CHECK:         region explicit.0 offset=0 size=SIZE align=16
        CHECK-NEXT:    site #0 load {{.*}} region=explicit.0{{$}}
        CHECK-NEXT:    site #1 store {{.*}} region=explicit.0OFFSET{{$}}
        CHECK-NOT:     barrier
        """
    testing.check(
        k1,
        f32_in_out(),
        checks.replace("SIZE", str(size)).replace("OFFSET", store_offset),
        block=BLOCK,
        chip=CHIP,
    )


def test_auto_sync_asks_for_a_barrier_after_each_call():
    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage(auto_sync=True)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(
            block, output, items, algorithm="transpose", temp_storage=storage
        )

    testing.check(
        k1,
        f32_in_out(),
        """
        CHECK-LABEL: coop-plan k1
        CHECK:         site #0 load {{.*}} region=explicit.0{{$}}
        CHECK-NEXT:    barrier after #0
        CHECK-NEXT:    site #1 store {{.*}} region=explicit.0{{$}}
        CHECK-NEXT:    barrier after #1
        """,
        block=BLOCK,
        chip=CHIP,
    )


def test_storage_given_only_to_direct_calls_takes_no_scratch():
    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage()
        coop.load(block, source, items, temp_storage=storage)
        coop.store(block, output, items, temp_storage=storage)

    testing.check(
        k1,
        f32_in_out(),
        """
        CHECK-LABEL: coop-plan k1
        CHECK-NOT:     backing
        CHECK-NOT:     region
        CHECK:         site #0 load {{.*}} algorithm=direct
        CHECK-NOT:     region
        """,
        block=BLOCK,
        chip=CHIP,
    )


def test_scratch_above_the_static_limit_is_dynamic_shared_memory(monkeypatch):
    monkeypatch.setattr(
        _rewrite_storage,
        "_query_device_shared_memory_limits",
        lambda: {
            "max_default_shared_memory_per_block": 48 * 1024,
            "max_optin_shared_memory_per_block": 96 * 1024,
        },
    )

    @cuda.jit
    def k1(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        storage = coop.TempStorage(64 * 1024)
        coop.load(
            block, source, items, algorithm="transpose", temp_storage=storage
        )
        coop.store(block, output, items)

    compiled = testing.check(
        k1,
        f32_in_out(),
        """
        CHECK-LABEL: coop-plan k1
        CHECK:         backing dynamic size=65536 align=16
        CHECK-NEXT:    region explicit.0 offset=0 size=65536 align=16
        CHECK-NEXT:    site #0 load {{.*}} region=explicit.0{{$}}
        """,
        block=BLOCK,
        sharedmem=64 * 1024,
        chip=CHIP,
    )
    testing.filecheck(
        """
        CHECK:     gpu.dynamic_shared_memory
        CHECK-NOT: memref.global {{.*}}workgroup
        """,
        compiled.mlir,
    )
