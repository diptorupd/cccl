# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""What the typed pipeline refuses for warp groups and ``group_by``.

Each refusal is raised with the class coop's planner raises, which is also
a ``TypedPipelineError``. A ``group_by`` refusal points at the call; a load
or store on a warp group is refused until the typed pipeline lowers warp
providers.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import cuda

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop
from cuda.coop.numba_mlir._compiler._group_errors import (
    NonConstantGroupArgumentError,
    UnsupportedLoadStoreGroupError,
)
from cuda.coop.numba_mlir._compiler._group_planner_support import (
    GroupRewriteError,
)
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError
from cuda.coop.numba_mlir._pipeline._groups import (
    WarpLoadStoreUnsupportedError,
)

from .._support import CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline

WARP_BLOCK = (64, 1, 1)
NOT_YET = "warp load/store is not supported on the typed pipeline yet"


def _compile(kernel, block=WARP_BLOCK):
    return testing.compile_kernel(kernel, f32_in_out(), block=block, chip=CHIP)


def _raises(error, match):
    return pytest.raises((error, TypedPipelineError), match=match)


def _check_class(info, *classes):
    for cls in (*classes, TypedPipelineError):
        assert isinstance(info.value, cls), type(info.value).__mro__


def test_a_nested_partition_is_refused_at_its_call():
    def kernel(source, output):
        coop.this_warp().group_by(8).group_by(2)
        output[0] = source[0]

    with _raises(NotImplementedError, r"nested ThreadGroup\.group_by") as info:
        _compile(kernel)
    _check_class(info, NotImplementedError)
    assert "coop.this_warp().group_by(8).group_by(2)" in str(info.value)


def test_a_runtime_count_is_refused_with_the_planner_class():
    def kernel(source, output):
        group = coop.this_warp().group_by(cuda.threadIdx.x)
        items = coop.ThreadData(4)
        coop.load(group, source, items)
        coop.store(group, output, items)

    with _raises(NonConstantGroupArgumentError, "got 'count'") as info:
        _compile(kernel)
    _check_class(info, NonConstantGroupArgumentError)


@pytest.mark.parametrize(
    "count, match",
    [(64, "cannot exceed 32 threads"), (3, "requires the count to divide")],
)
def test_a_count_that_does_not_fit_the_warp_is_refused(count, match):
    def kernel(source, output):
        coop.this_warp().group_by(count)
        output[0] = source[0]

    with _raises(ValueError, match) as info:
        _compile(kernel)
    _check_class(info, ValueError)


def test_group_by_call_syntax_is_checked_as_the_planner_does():
    def kernel(source, output):
        qualified_coop.this_warp().group_by(8, width=8)
        output[0] = source[0]

    with _raises(GroupRewriteError, r"unexpected keyword\(s\): width") as info:
        _compile(kernel)
    _check_class(info, GroupRewriteError)


def test_a_partition_load_store_does_not_support_is_refused_by_planning():
    def kernel(source, output):
        group = coop.this_warp().group_by(3, exhaustive=False)
        items = coop.ThreadData(4)
        coop.load(group, source, items)
        coop.store(group, output, items)

    with _raises(
        UnsupportedLoadStoreGroupError, "power-of-two group width"
    ) as info:
        _compile(kernel)
    _check_class(info, UnsupportedLoadStoreGroupError)


def test_a_block_partition_is_not_a_load_store_group():
    def kernel(source, output):
        group = coop.this_block().group_by(2)
        items = coop.ThreadData(4)
        coop.load(group, source, items)
        coop.store(coop.this_block(), output, items)

    with _raises(UnsupportedLoadStoreGroupError, "supports this_block") as info:
        _compile(kernel)
    _check_class(info, UnsupportedLoadStoreGroupError)


def _warp_kernel(api, width, kind):
    """A kernel with one ``kind`` call on ``api.this_warp()`` or a partition."""

    if kind == "load" and width is None:

        def kernel(source, output):
            items = coop.ThreadData(4)
            api.load(api.this_warp(), source, items)
            output[0] = items[0]

    elif kind == "load":

        def kernel(source, output):
            items = coop.ThreadData(4)
            api.load(api.this_warp().group_by(width), source, items)
            output[0] = items[0]

    elif width is None:

        def kernel(source, output):
            items = coop.ThreadData(4, dtype=source.dtype)
            items[0] = source[0]
            api.store(api.this_warp(), output, items)

    else:

        def kernel(source, output):
            items = coop.ThreadData(4, dtype=source.dtype)
            items[0] = source[0]
            api.store(api.this_warp().group_by(width), output, items)

    return kernel


_APIS = [
    pytest.param(coop, id="common"),
    pytest.param(qualified_coop, id="qualified"),
]


@pytest.mark.parametrize("width", [None, 8])
def test_a_warp_group_needs_a_block_of_complete_warps(width):
    with _raises(
        UnsupportedLoadStoreGroupError, "complete 32-thread warps"
    ) as info:
        _compile(_warp_kernel(coop, width, "load"), block=(48, 1, 1))
    _check_class(info, UnsupportedLoadStoreGroupError)


@pytest.mark.parametrize("api", _APIS)
@pytest.mark.parametrize("width", [None, 8, 16])
@pytest.mark.parametrize("kind", ["load", "store"])
def test_load_and_store_on_a_warp_group_are_refused_by_name(api, width, kind):
    group = (
        r"coop\.group\[warp\]"
        if width is None
        else rf"coop\.group\[threads_within_warp, {width}\]"
    )
    with _raises(
        WarpLoadStoreUnsupportedError, rf"coop\.{kind} at line \d+ on {group}"
    ) as info:
        _compile(_warp_kernel(api, width, kind))
    _check_class(info, UnsupportedLoadStoreGroupError, NotImplementedError)
    assert NOT_YET in str(info.value)


def test_a_phi_of_two_warp_groups_types_and_reaches_the_warp_refusal():
    def kernel(source, output):
        if source[0] > 0:
            group = coop.this_warp()
        else:
            group = qualified_coop.this_warp()
        pair = (group, 1)
        items = coop.ThreadData(4)
        coop.load(pair[0], source, items)
        coop.store(group, output, items)

    with _raises(WarpLoadStoreUnsupportedError, NOT_YET):
        _compile(kernel)


def test_this_warp_takes_no_arguments():
    def kernel(source, output):
        coop.this_warp(32)
        output[0] = source[0]

    with pytest.raises(
        Exception, match=r"this_warp\(\) too many positional arguments"
    ):
        _compile(kernel)
