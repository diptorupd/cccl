# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group types, ``group_by`` typing and warp requests, on typed inputs.

None of this needs a compiled kernel: the group types and the ``group_by``
rule take Numba types, and a warp request is completed and specialized the
way call resolution does it, without NVRTC.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop
from cuda.coop._core import BindingKind
from cuda.coop.numba_mlir._compiler._group_errors import (
    InvalidLoadStoreAlgorithmError,
    NonConstantGroupArgumentError,
    UnsupportedLoadStoreGroupError,
)
from cuda.coop.numba_mlir._compiler._group_planner_support import (
    GroupRewriteError,
)
from cuda.coop.numba_mlir._pipeline import _groups, _specialize, _templates
from cuda.coop.numba_mlir._pipeline._bridges import types as numba_types
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError
from cuda.coop.numba_mlir._pipeline._types import (
    PRODUCER,
    CoopGroupType,
    RefusedGroupType,
    ThreadDataType,
)

BLOCK = (64, 1, 1)
WIDTHS = (1, 2, 4, 8, 16, 32)
literal = numba_types.literal


def _warp(width=None, *, exhaustive=True):
    group = coop.this_warp()
    if width is not None:
        group = group.group_by(width, exhaustive=exhaustive)
    return CoopGroupType(group)


def _request(group, algorithm="direct", kind="load", *, qualified=False):
    return _templates.request_for(
        kind,
        [
            group,
            types.float32[::1],
            ThreadDataType(types.float32, 4, PRODUCER),
        ],
        {"algorithm": literal(algorithm)},
        qualified=qualified,
    )


def test_the_width_of_a_logical_warp_is_part_of_its_type():
    groups = [_warp(width) for width in WIDTHS]
    assert [g.width for g in groups] == list(WIDTHS)
    assert len({g.key for g in groups}) == len(WIDTHS)
    assert [str(g) for g in groups[3:]] == [
        "coop.group[threads_within_warp, 8]",
        "coop.group[threads_within_warp, 16]",
        "coop.group[threads_within_warp, 32]",
    ]
    assert all(g.is_warp and not g.is_block for g in groups)


def test_both_apis_give_one_group_type():
    assert CoopGroupType(coop.this_warp()) is CoopGroupType(
        qualified_coop.this_warp()
    )
    assert CoopGroupType(coop.this_block()) is CoopGroupType(
        qualified_coop.this_block()
    )


def test_physical_and_logical_warps_and_blocks_are_distinct():
    physical, block = _warp(), CoopGroupType(coop.this_block())
    assert (physical.kind, physical.width, str(physical)) == (
        "warp",
        32,
        "coop.group[warp]",
    )
    assert (block.kind, block.width, block.is_warp) == ("block", None, False)
    assert len({physical, _warp(32), _warp(8, exhaustive=False), block}) == 4
    assert str(_warp(8, exhaustive=False)) == (
        "coop.group[threads_within_warp, 8, partial]"
    )


@pytest.mark.parametrize("width", [8, 16])
def test_group_by_types_coop_s_own_partition(width):
    result = _groups.partition(_warp(), (literal(width),), {})
    assert result is _warp(width)
    kept = _groups.partition(
        _warp(), (literal(width),), {"exhaustive": literal(False)}
    )
    assert kept is _warp(width, exhaustive=False)


def test_a_block_partition_is_a_group_of_warps():
    block = CoopGroupType(coop.this_block())
    result = _groups.partition(block, (), {"count": literal(2)})
    assert (result.kind, result.group.static_size) == ("warps_within_block", 64)


_REFUSALS = [
    pytest.param(
        _warp(8),
        (literal(2),),
        {},
        NotImplementedError,
        "nested ThreadGroup.group_by",
        id="nested",
    ),
    pytest.param(
        _warp(),
        (literal(64),),
        {},
        ValueError,
        "cannot exceed 32 threads",
        id="wider-than-a-warp",
    ),
    pytest.param(
        _warp(),
        (literal(3),),
        {},
        ValueError,
        "requires the count to divide",
        id="not-dividing",
    ),
    pytest.param(
        _warp(),
        (literal(0),),
        {},
        ValueError,
        "count must be positive",
        id="zero",
    ),
    pytest.param(
        _warp(),
        (types.float64,),
        {},
        TypeError,
        "count must be a static integer",
        id="float-count",
    ),
    pytest.param(
        _warp(),
        (types.int32,),
        {},
        NonConstantGroupArgumentError,
        r"must be\scompile-time constants; got 'count'",
        id="runtime-count",
    ),
    pytest.param(
        _warp(),
        (literal(8),),
        {"exhaustive": types.boolean},
        NonConstantGroupArgumentError,
        "got 'exhaustive'",
        id="runtime-exhaustive",
    ),
    pytest.param(
        _warp(),
        (literal(8),),
        {"exhaustive": literal(1)},
        TypeError,
        "exhaustive must be a bool",
        id="integer-exhaustive",
    ),
    pytest.param(
        _warp(),
        (),
        {},
        GroupRewriteError,
        "requires count",
        id="no-count",
    ),
    pytest.param(
        _warp(),
        (literal(8), literal(True)),
        {},
        GroupRewriteError,
        "one positional count argument",
        id="positional-exhaustive",
    ),
    pytest.param(
        _warp(),
        (literal(8),),
        {"count": literal(8)},
        GroupRewriteError,
        "count more than once",
        id="count-twice",
    ),
    pytest.param(
        _warp(),
        (literal(8),),
        {"width": literal(8)},
        GroupRewriteError,
        "unexpected keyword\\(s\\): width",
        id="unknown-keyword",
    ),
]


@pytest.mark.parametrize("parent, args, kws, error, message", _REFUSALS)
def test_a_refused_partition_types_as_its_error(
    parent, args, kws, error, message
):
    result = _groups.partition(parent, args, kws)
    assert isinstance(result, RefusedGroupType)
    assert issubclass(result.error, error)
    assert issubclass(result.error, TypedPipelineError)
    with pytest.raises(error, match=message):
        raise result.error(result.message)


def test_a_partition_of_a_refused_group_keeps_the_first_refusal():
    refused = _groups.partition(_warp(), (types.int64,), {})
    assert _groups.partition(refused, (literal(8),), {}) is refused


def test_requests_on_different_groups_are_distinct():
    requests = [_request(_warp(w)) for w in WIDTHS]
    requests.append(_request(CoopGroupType(coop.this_block())))
    assert len({r.key for r in requests}) == len(requests)
    assert _request(_warp(8)) is _request(_warp(8))


@pytest.mark.parametrize(
    "algorithm, factory, scratch",
    [
        ("direct", "warp_load", False),
        ("transpose", "_warp_load_with_storage", True),
    ],
)
def test_a_warp_request_completes_to_a_warp_provider_of_its_width(
    algorithm, factory, scratch
):
    for width in (8, 16, 32):
        completed = _specialize._complete(
            _request(_warp(width), algorithm), BLOCK
        )
        assert completed.factory.__name__ == factory
        assert completed.kwargs["threads_in_warp"] == width
        assert completed.kwargs["threads_per_block"] == BLOCK
        assert completed.kwargs["offset"].kind is BindingKind.RUNTIME
        assert completed.tile == _specialize.GroupTile(width, width * 4)
        assert completed.scratch is scratch


def test_a_block_request_has_no_group_tile():
    completed = _specialize._complete(
        _request(CoopGroupType(coop.this_block())), BLOCK
    )
    assert completed.factory.__name__ == "load"
    assert completed.tile is None
    assert not completed.scratch


@pytest.mark.parametrize("width", [None, 8])
def test_a_warp_request_takes_only_the_warp_algorithms(width):
    with pytest.raises(
        InvalidLoadStoreAlgorithmError, match=r"direct,\s+striped,\s+transpose"
    ) as info:
        _specialize._complete(_request(_warp(width), "warp_transpose"), BLOCK)
    assert isinstance(info.value, TypedPipelineError)


def test_logical_warp_widths_have_distinct_specializations_and_cache_keys():
    from cuda.coop.numba_mlir._compiler import _nvrtc

    target = _nvrtc.CompileTarget.from_targetoptions({"chip": "sm_120"})
    requests = [_request(_warp(w), "transpose") for w in (8, 16)]
    with _nvrtc.compile_context_scope():
        algorithms = _specialize._instantiate(requests, BLOCK, target)
    assert [a.logical_warp_threads for a in algorithms] == [8, 16]
    assert len({a._make_lto_ir_cache_key() for a in algorithms}) == 2
    for width, algorithm in zip((8, 16), algorithms):
        source = algorithm._source_code()[0]
        assert f", {width}>" in source
        assert f"temp_storages[{BLOCK[0] // width}]" in source


@pytest.mark.parametrize("width", [None, 8])
def test_coop_planning_refuses_a_warp_group_in_a_partial_warp_block(width):
    with pytest.raises(
        UnsupportedLoadStoreGroupError, match="complete 32-thread warps"
    ) as info:
        _specialize._complete(_request(_warp(width)), (48, 1, 1))
    assert isinstance(info.value, TypedPipelineError)


def test_coop_planning_refuses_a_partition_load_store_does_not_support():
    block_of_warps = CoopGroupType(coop.this_block().group_by(2))
    with pytest.raises(NotImplementedError, match="in the common API"):
        _request(block_of_warps)
    with pytest.raises(
        UnsupportedLoadStoreGroupError, match="supports this_block"
    ):
        _specialize._complete(_request(block_of_warps, qualified=True), BLOCK)
    with pytest.raises(
        UnsupportedLoadStoreGroupError, match="power-of-two group width"
    ):
        _specialize._complete(_request(_warp(3, exhaustive=False)), BLOCK)
