# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Choose a provider specialization for each site, once the block is known.

Specialization planning is the second stage of ``CoopCallResolution``. From
the checked sites and the launch's block it:

* completes each distinct request with coop's ``plan_group_primitive``,
  memoized per request and block, and specializes its provider at the
  kernel's target, with one compile context for all of them;
* decides ``launch_bounds`` from the block's thread count;
* decides which sites take scratch, from each provider's storage ABI, and
  places a barrier after each one whose plan asks for a storage-reuse
  barrier: every site with implicit scratch, and the sites of a
  ``TempStorage`` with ``auto_sync=True``, as coop's planner does;
* records, for each site on a warp group, where its group's tile starts,
  which coop's planner leaves to the backend. A warp site's scratch and
  barrier are its group's: one slot per warp of the block, and a
  ``syncwarp`` over the group's lanes.

Specializing generates provider source and nothing more: this stage does
not call NVRTC, so scratch use is decided before any size is known. It does
not change the IR or the compile state; the driver applies its decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from ._errors import (
    InvalidLoadStoreAlgorithmError,
    TypedPipelineError,
    TypedPipelineValueError,
    planner_text,
    typed_error,
)
from ._groups import UnsupportedGroupError
from ._sites import Sites
from ._templates import LoadStoreRequest


@dataclass(frozen=True)
class LaunchBounds:
    """The kernel's ``launch_bounds``; ``rebind`` when the stage chose it."""

    value: int | None
    rebind: bool = False


@dataclass(frozen=True)
class GroupTile:
    """Where a group's tile starts in memory, for one call.

    A tile holds ``items`` elements (``width * items_per_thread``). The
    group of the thread with x-major linear rank ``r`` moves the tile at
    ``(r // width) * items``, plus the call's ``offset``.
    """

    width: int
    items: int


@dataclass(frozen=True)
class Specialization:
    """What the kernel's sites specialized to, one entry per site.

    ``algorithms`` holds each distinct provider once, in first-use order;
    ``site_algorithms`` repeats them per site. ``tiles`` holds each site's
    ``GroupTile``, or ``None`` when the provider addresses the block's
    tile itself.
    """

    block: tuple[int, int, int]
    algorithms: tuple
    site_algorithms: tuple
    scratch: tuple[bool, ...]
    barrier_after: tuple[bool, ...]
    launch_bounds: LaunchBounds
    tiles: tuple[GroupTile | None, ...] = ()


# Request completion ----------------------------------------------------------


class Completion(NamedTuple):
    """A request completed against a block: the provider and its inputs.

    ``barrier`` is whether the plan asks for a reuse barrier after the call.
    ``tile`` is set when the plan leaves the group's tile origin to the
    caller (warp groups); the factory then takes a runtime ``offset``.
    ``scratch`` is whether the planned provider needs scratch storage.
    """

    factory: Any
    kwargs: dict
    barrier: bool
    tile: GroupTile | None
    scratch: bool


_COMPLETIONS: dict[tuple, Completion] = {}


def _complete(request: LoadStoreRequest, block: tuple) -> Completion:
    """Return the provider factory and its arguments for ``request``.

    ``plan_group_primitive`` hashes the request's semantic key, which is
    costly, so its result is memoized per request and block. A warp group
    plans to a warp provider, which takes the group's width as
    ``threads_in_warp`` and, as an offset, its tile's origin.
    """

    key = (request.key, block)
    completed = _COMPLETIONS.get(key)
    if completed is not None:
        return completed

    from ..._core import ArgumentBinding
    from ..._core.group._dispatch import plan_group_primitive
    from ..._core.group._model import (
        GroupLoweringTarget,
        StorageOwnership,
        SynchronizationScope,
    )
    from ..._core.launch import LaunchFacts
    from .._compiler import _group_errors as planner
    from .._compiler._load_store_algorithms import (
        _WARP_LOAD_STORE_ALGORITHMS,
    )
    from .._lowering import _load_store
    from .._lowering._core import NumbaMlirCoreAdapter

    if (
        request.group.is_warp
        and request.algorithm not in _WARP_LOAD_STORE_ALGORITHMS
    ):
        reason = planner.InvalidLoadStoreAlgorithmError(
            request.kind,
            ", ".join(sorted(_WARP_LOAD_STORE_ALGORITHMS)),
            request.group.kind,
        )
        raise InvalidLoadStoreAlgorithmError(planner_text(reason))
    if request.group.is_warp and request.storage is not None:
        raise TypedPipelineValueError(
            f"cuda.coop.numba_mlir.{request.kind} temp_storage is not "
            "supported for Warp groups; omit it so the implementation can "
            "provide per-group storage"
        )
    try:
        plan = plan_group_primitive(
            request.call, LaunchFacts(exact_block_dim=block)
        ).require_supported()
    except NotImplementedError as exc:
        reason = planner.UnsupportedLoadStoreGroupError(request.kind, str(exc))
        raise UnsupportedGroupError(planner_text(reason)) from None
    except (TypeError, ValueError) as exc:
        raise typed_error(exc, f"coop.{request.kind}: ") from None
    storage_free = plan.temp_storage.ownership is StorageOwnership.NONE
    if plan.target is GroupLoweringTarget.CUB_BLOCK:
        prefix, group_kwargs = "", {}
    elif plan.target is GroupLoweringTarget.CUB_WARP:
        prefix = "warp_"
        group_kwargs = {"threads_in_warp": plan.topology.logical_width}
    else:
        raise TypedPipelineError(
            f"coop.{request.kind}: planned for {plan.target.value}, which the "
            "typed pipeline does not lower"
        )
    implementation = plan.implementation
    factory_name = (
        f"{prefix}{request.kind}"
        if storage_free
        else f"_{prefix}{request.kind}_with_storage"
    )
    tile = None
    if implementation.metadata.get("requires_runtime_effective_offset"):
        tile = GroupTile(
            plan.topology.logical_width,
            implementation.metadata["effective_offset_stride"],
        )
    omitted = ArgumentBinding.omitted()
    kwargs = {
        **group_kwargs,
        "dtype": NumbaMlirCoreAdapter().normalize_dtype(
            implementation.template_arguments["T"]
        ),
        "threads_per_block": block,
        "items_per_thread": implementation.template_arguments[
            "ITEMS_PER_THREAD"
        ],
        "algorithm": implementation.metadata["algorithm"],
        "num_valid_items": request.binding("valid_items") or omitted,
        "offset": (
            ArgumentBinding.runtime()
            if tile is not None
            else request.binding("offset") or omitted
        ),
    }
    if request.binding("oob_default") is not None:
        kwargs["oob_default"] = request.binding("oob_default")
    barrier = (
        not storage_free
        and plan.synchronization.storage_reuse_barrier
        is not SynchronizationScope.NONE
    )
    completed = Completion(
        getattr(_load_store, factory_name),
        kwargs,
        barrier,
        tile,
        not storage_free,
    )
    _COMPLETIONS[key] = completed
    return completed


def _instantiate(
    requests: Sequence[LoadStoreRequest], block: tuple, target
) -> list:
    """Specialize each request's provider at ``target``.

    The providers resolve their compile context when they are qualified;
    inside the driver's ``compile_context_scope`` that is the context the
    driver resolved for the whole kernel.
    """

    from .._types import collect_specializations, kernel_compile_target

    with (
        kernel_compile_target(target),
        collect_specializations() as collected,
    ):
        for request in requests:
            completed = _complete(request, block)
            completed.factory(**completed.kwargs)
    if len(collected) != len(requests):
        raise AssertionError("a provider factory did not specialize once")
    return collected


# Decisions -------------------------------------------------------------------


def uses_scratch(algorithm: Any) -> bool:
    """Whether ``algorithm``'s provider takes the scratch pointer."""

    from .._compiler._operations import StorageABI

    return algorithm.storage_abi is StorageABI.LEADING_POINTER


def plan_launch_bounds(block: tuple, targetoptions: dict) -> LaunchBounds:
    threads = block[0] * block[1] * block[2]
    bounds = targetoptions.get("launch_bounds")
    if bounds is not None:
        maximum = bounds[0] if isinstance(bounds, tuple) else bounds
        if threads > maximum:
            raise TypedPipelineError(
                f"the launch's block {block} has {threads} threads, "
                f"more than launch_bounds={bounds!r}"
            )
        return LaunchBounds(maximum)
    if targetoptions.get("max_registers") is None:
        return LaunchBounds(threads, rebind=True)
    return LaunchBounds(None)


def plan(
    algorithms: Sequence[Any],
    site_algorithms: Sequence[Any],
    block: tuple,
    targetoptions: dict,
    barriers: Sequence[bool] | None = None,
    tiles: Sequence[GroupTile | None] | None = None,
) -> Specialization:
    """Decide scratch, barriers and ``launch_bounds`` for specialized sites.

    ``barriers`` holds whether each site's plan asks for a reuse barrier;
    by default every call that uses scratch does.
    """

    launch_bounds = plan_launch_bounds(block, targetoptions)
    scratch = tuple(uses_scratch(a) for a in site_algorithms)
    if barriers is None:
        barriers = scratch
    return Specialization(
        block=block,
        algorithms=tuple(algorithms),
        site_algorithms=tuple(site_algorithms),
        scratch=scratch,
        barrier_after=tuple(
            s and b for s, b in zip(scratch, barriers, strict=True)
        ),
        launch_bounds=launch_bounds,
        tiles=(
            (None,) * len(site_algorithms) if tiles is None else tuple(tiles)
        ),
    )


def specialize(
    sites: Sites,
    block: tuple,
    *,
    target: Any,
    targetoptions: dict,
) -> Specialization:
    """Specialize every site's provider for ``block`` and plan around them.

    Every request is completed first, so that coop's planning checks the
    groups against the block before any provider is specialized.
    """

    unique = list(dict.fromkeys(s.request for s in sites.sites))
    completions = {request: _complete(request, block) for request in unique}
    algorithms = _instantiate(unique, block, target)
    algorithm_for = dict(zip(unique, algorithms))
    return plan(
        algorithms,
        [algorithm_for[s.request] for s in sites.sites],
        block,
        targetoptions,
        [completions[s.request].barrier for s in sites.sites],
        tiles=[completions[s.request].tile for s in sites.sites],
    )


__all__ = [
    "Completion",
    "GroupTile",
    "LaunchBounds",
    "Specialization",
    "plan",
    "plan_launch_bounds",
    "specialize",
    "uses_scratch",
]
