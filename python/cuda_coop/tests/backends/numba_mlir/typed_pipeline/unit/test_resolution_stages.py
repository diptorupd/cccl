# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specialization planning and the scratch layout, on stub inputs.

Neither needs NVRTC, a GPU or a compiled kernel: planning reads only each
provider's storage ABI, and the layout takes plain sizes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuda.coop.numba_mlir._compiler._operations import StorageABI
from cuda.coop.numba_mlir._pipeline import _bridges as nb
from cuda.coop.numba_mlir._pipeline import _link, _specialize, _storage
from cuda.coop.numba_mlir._pipeline._descriptors import Descriptor
from cuda.coop.numba_mlir._pipeline._errors import (
    SharedMemoryAliasError,
    TempStorageError,
    TypedPipelineError,
)
from cuda.coop.numba_mlir._pipeline._plan import Backing, Region
from cuda.coop.numba_mlir._pipeline._sites import UserSharedArray
from cuda.coop.numba_mlir._pipeline._specialize import LaunchBounds

BLOCK = (128, 1, 1)
LIMIT = _storage.STATIC_SHARED_LIMIT


def _provider(storage_abi, digest="digest"):
    return SimpleNamespace(
        storage_abi=storage_abi, c_name="stub", _private_symbol_digest=digest
    )


SCRATCH = _provider(StorageABI.LEADING_POINTER, "scratch")
FREE = _provider(StorageABI.NONE, "free")


def test_barriers_follow_scratch_use():
    plan = _specialize.plan(
        [FREE, SCRATCH], [FREE, SCRATCH, FREE, SCRATCH], BLOCK, {}
    )
    assert plan.scratch == (False, True, False, True)
    assert plan.barrier_after == plan.scratch


def test_a_site_whose_plan_asks_for_no_barrier_gets_none():
    plan = _specialize.plan(
        [SCRATCH], [SCRATCH, SCRATCH], BLOCK, {}, barriers=[False, True]
    )
    assert plan.barrier_after == (False, True)


def test_launch_bounds_default_to_the_block_thread_count():
    assert _specialize.plan_launch_bounds((32, 4, 1), {}) == LaunchBounds(
        128, rebind=True
    )


def test_explicit_launch_bounds_are_kept_and_checked_against_the_block():
    bounds = {"launch_bounds": (256, 2)}
    assert _specialize.plan_launch_bounds(BLOCK, bounds) == LaunchBounds(256)
    with pytest.raises(TypedPipelineError, match="more than launch_bounds"):
        _specialize.plan_launch_bounds((512, 1, 1), bounds)


def test_max_registers_leaves_launch_bounds_unset():
    options = {"max_registers": 32}
    assert _specialize.plan_launch_bounds(BLOCK, options) == LaunchBounds(None)


def test_one_region_as_large_as_the_largest_need_not_their_sum():
    layout = _storage.layout([(1024, 16), None, (512, 8)])
    assert layout.regions == (Region("implicit.0", 0, 1024, 16),)
    assert layout.backing == Backing("static", 1024, 16)
    assert layout.scratch == (True, False, True)


def test_backing_rounds_up_to_its_alignment_of_at_least_16():
    assert _storage.layout([(100, 4)]).backing == Backing("static", 112, 16)
    wide = _storage.layout([(100, 32)])
    assert wide.backing == Backing("static", 128, 32)
    assert wide.regions == (Region("implicit.0", 0, 100, 32),)


def test_no_scratch_needs_no_backing():
    layout = _storage.layout([None, None], user_shared=(_user(4096),))
    assert (layout.backing, layout.regions) == (None, ())
    assert layout.scratch == (False, False)


EIGHTS = ("threads_within_warp", 8, 8)
WARPS = ("warp", 32, 2)


def test_a_warp_group_gets_an_aligned_slot_per_instance():
    layout = _storage.layout([(132, 4)], groups=[EIGHTS])
    assert layout.regions == (Region("implicit.0", 0, 8 * 136, 8),)
    assert layout.slices[0] == _storage.Slice("implicit.0", 0, 132, 0, 136)


def test_group_domains_follow_each_other_in_first_use_order():
    layout = _storage.layout(
        [(1024, 16), (132, 4), (516, 4), (100, 4), (64, 8)],
        groups=[None, EIGHTS, WARPS, EIGHTS, None],
    )
    assert layout.regions == (Region("implicit.0", 0, 2112 + 2 * 520, 16),)
    assert [(s.region_offset, s.stride) for s in layout.slices] == [
        (0, 0),
        (1024, 136),
        (2112, 520),
        (1024, 136),
        (0, 0),
    ]


def test_a_group_with_one_instance_takes_no_stride():
    layout = _storage.layout([(100, 4)], groups=[("warp", 32, 1)])
    assert layout.regions == (Region("implicit.0", 0, 100, 8),)
    assert layout.slices[0].stride == 0


def test_explicit_storage_keeps_its_layout_next_to_a_warp_domain():
    storage = _storage_descriptor()
    layout = _storage.layout(
        [(64, 16), (132, 4)], [storage, None], groups=[None, EIGHTS]
    )
    assert layout.regions == (
        Region("explicit.0", 0, 64, 16),
        Region("implicit.0", 64, 8 * 136, 8),
    )
    assert layout.slices[1].offset == 64


def _user(size, static=True, line=3):
    return UserSharedArray(static, size, nb.ir.Loc("kernel.py", line))


def _storage_descriptor(
    index=0, size=None, alignment=None, auto_sync=True, sharing="shared"
):
    return Descriptor(
        f"storage{index}",
        index,
        size,
        alignment,
        auto_sync,
        sharing,
        nb.ir.Loc("kernel.py", 10 + index),
    )


def test_shared_descriptor_sites_reuse_one_slice():
    storage = _storage_descriptor()
    layout = _storage.layout([(96, 16), (64, 4)], [storage, storage])
    assert layout.regions == (Region("explicit.0", 0, 96, 16),)
    assert [(s.offset, s.size) for s in layout.slices] == [(0, 96), (0, 64)]


def test_exclusive_descriptor_sites_get_their_own_slices():
    storage = _storage_descriptor(sharing="exclusive")
    layout = _storage.layout([(100, 4), (64, 16)], [storage, storage])
    assert layout.regions == (Region("explicit.0", 0, 176, 16),)
    assert [s.region_offset for s in layout.slices] == [0, 112]


def test_explicit_regions_come_in_constructor_order_before_implicit():
    first, second = _storage_descriptor(0), _storage_descriptor(1, size=256)
    layout = _storage.layout(
        [(40, 16), (64, 8), (32, 8), None], [None, second, first, second]
    )
    assert layout.regions == (
        Region("explicit.0", 0, 32, 8),
        Region("explicit.1", 32, 256, 8),
        Region("implicit.0", 288, 40, 16),
    )
    assert [s and s.region for s in layout.slices] == [
        "implicit.0",
        "explicit.1",
        "explicit.0",
        None,
    ]
    assert layout.backing == Backing("static", 336, 16)


def test_descriptor_without_storage_bearing_sites_has_no_region():
    storage = _storage_descriptor(size=8, sharing="exclusive")
    layout = _storage.layout([None, (64, 16)], [storage, None])
    assert [r.name for r in layout.regions] == ["implicit.0"]


def test_descriptor_size_must_cover_its_sites():
    storage = _storage_descriptor(size=32)
    with pytest.raises(
        TempStorageError,
        match=r"size_in_bytes is smaller than required by primitive uses "
        r"\(32 < 64\)",
    ):
        _storage.layout([(64, 16)], [storage])


def test_descriptor_alignment_is_raised_to_the_pointer_minimum():
    for requested, expected in ((1, 16), (64, 64)):
        storage = _storage_descriptor(alignment=requested)
        layout = _storage.layout([(64, 16)], [storage])
        assert layout.regions[0].alignment == expected


def test_static_limit_includes_the_kernels_own_shared_bytes():
    _storage.layout([(LIMIT - 1024, 16)], user_shared=(_user(1024),))
    with pytest.raises(
        TypedPipelineError,
        match=f"needs {LIMIT - 1024} bytes of static shared memory, which "
        "with the kernel's own 1040 bytes of static cuda.shared.array "
        "exceeds",
    ):
        _storage.layout([(LIMIT - 1024, 16)], user_shared=(_user(1040),))


def test_scratch_refuses_dynamic_user_shared_arrays():
    user = (_user(0, static=False),)
    _storage.layout([None], user_shared=user)
    with pytest.raises(
        SharedMemoryAliasError,
        match=r"requires a 112-byte static shared-memory backing, but this "
        r"kernel also declares dynamic/runtime-sized "
        r"cuda\.shared\.array\(\.\.\.\) at kernel\.py \(3\)\. .*would alias",
    ):
        _storage.layout([(100, 4)], user_shared=user)


def _limits(default, optin):
    return lambda required: (default, optin)


def test_scratch_above_the_static_limit_is_a_dynamic_backing():
    storage = _storage_descriptor(size=64 * 1024, alignment=16)
    layout = _storage.layout(
        [(256, 16)], [storage], limits=_limits(LIMIT, 96 * 1024)
    )
    assert layout.backing == Backing("dynamic", 64 * 1024, 16)
    assert layout.dynamic_shared_bytes == 64 * 1024


def test_dynamic_backing_refuses_every_user_shared_array():
    with pytest.raises(
        SharedMemoryAliasError,
        match=r"65536-byte dynamic shared-memory backing, but this kernel "
        r"also declares static cuda\.shared\.array",
    ):
        _storage.layout(
            [(256, 16)],
            [_storage_descriptor(size=64 * 1024)],
            [_user(128)],
            limits=_limits(LIMIT, 96 * 1024),
        )


def test_dynamic_backing_is_limited_to_16_byte_alignment_and_the_optin():
    with pytest.raises(TempStorageError, match="only 16-byte alignment"):
        _storage.layout(
            [(256, 16)],
            [_storage_descriptor(size=64 * 1024, alignment=32)],
            limits=_limits(LIMIT, 96 * 1024),
        )
    with pytest.raises(
        TempStorageError,
        match="requires 131072 bytes dynamic shared memory, but device max "
        "opt-in is 98304 bytes",
    ):
        _storage.layout(
            [(256, 16)],
            [_storage_descriptor(size=128 * 1024)],
            limits=_limits(LIMIT, 96 * 1024),
        )


def test_device_limits_query_only_above_the_static_limit(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _rewrite_storage

    def query():
        raise RuntimeError("no device")

    monkeypatch.setattr(
        _rewrite_storage, "_query_device_shared_memory_limits", query
    )
    assert _storage.device_limits(LIMIT) == (LIMIT, LIMIT)
    with pytest.raises(TempStorageError, match="exact current-device"):
        _storage.device_limits(LIMIT + 1)


def test_a_dynamic_backing_needs_the_launch_to_request_it():
    layout = _storage.layout(
        [(256, 16)],
        [_storage_descriptor(size=64 * 1024)],
        limits=_limits(LIMIT, 96 * 1024),
    )
    _storage.check_launch(layout, 64 * 1024)
    _storage.check_launch(_storage.layout([(256, 16)]), 0)
    with pytest.raises(TempStorageError, match="launch requests 0"):
        _storage.check_launch(layout, 0)


def test_compiled_layouts_must_agree_with_the_planned_scratch(monkeypatch):
    bundle = SimpleNamespace(layouts={"free": (64, 16)})
    monkeypatch.setattr(_link, "compile_bundle", lambda *args: bundle)
    plan = _specialize.plan([FREE], [FREE], BLOCK, {})
    sites = SimpleNamespace(sites=[SimpleNamespace(storage=None)])
    with pytest.raises(AssertionError, match="disagrees"):
        _storage.analyze(plan, sites, identity=None, context=None)
