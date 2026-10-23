# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The group types typing gives warps, logical warps and block partitions.

A kernel that only builds groups has no cooperative call to resolve, so it
compiles; its typed IR shows each group's type, a logical warp's width
included, and its MLIR shows that a group carries no run-time data.
"""

# ruff: noqa: F841

from __future__ import annotations

from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline

WARP_BLOCK = (64, 1, 1)


def test_each_group_has_its_own_type():
    @cuda.jit
    def groups(source, output):
        warp = coop.this_warp()
        qualified = qualified_coop.this_warp()
        eight = warp.group_by(8)
        sixteen = qualified.group_by(count=16, exhaustive=False)
        warps = coop.this_block().group_by(2)
        output[0] = source[0]

    testing.check(
        groups,
        f32_in_out(),
        """
        CHECK-LABEL: typed-ir groups
        CHECK:         warp = call {{.*}} :: coop.group[warp]
        CHECK:         qualified = call {{.*}} :: coop.group[warp]
        CHECK:         eight = call {{.*}} :: coop.group[threads_within_warp, 8]
        CHECK:         sixteen = call
        CHECK-SAME:      :: coop.group[threads_within_warp, 16, partial]
        CHECK:         warps = call
        CHECK-SAME:      :: coop.group[warps_within_block, 2]
        """,
        dump="typed_ir",
        block=WARP_BLOCK,
        chip=CHIP,
    )


def test_a_phi_and_a_tuple_projection_keep_the_warp_type():
    @cuda.jit
    def joined(source, output):
        if source[0] > 0:
            group = coop.this_warp().group_by(8)
        else:
            group = qualified_coop.this_warp().group_by(8)
        pair = (group, 1)
        again = pair[0]
        output[0] = source[0]

    testing.check(
        joined,
        f32_in_out(),
        r"""
        CHECK-LABEL: typed-ir joined
        CHECK:         [[JOIN:group\.[0-9]+]] = group ::
        CHECK-SAME:      coop.group[threads_within_warp, 8]
        CHECK:         [[JOIN]] = group.1 ::
        CHECK-SAME:      coop.group[threads_within_warp, 8]
        CHECK:         again = static_getitem
        CHECK-SAME:      :: coop.group[threads_within_warp, 8]
        """,
        dump="typed_ir",
        block=WARP_BLOCK,
        chip=CHIP,
    )


def test_a_group_lowers_to_a_placeholder():
    @cuda.jit
    def placeholder(source, output):
        group = coop.this_warp().group_by(4)
        output[0] = source[0]

    testing.check(
        placeholder,
        f32_in_out(),
        """
        CHECK: arith.constant 0 : i8
        CHECK: arith.constant 0 : i8
        CHECK-NOT: func.call
        """,
        dump="mlir",
        block=WARP_BLOCK,
        chip=CHIP,
    )
