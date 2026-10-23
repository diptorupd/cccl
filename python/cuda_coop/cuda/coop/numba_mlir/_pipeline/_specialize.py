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
  places a barrier after every one of them, as coop's planner does.

Specializing generates provider source and nothing more: this stage does
not call NVRTC, so scratch use is decided before any size is known. It does
not change the IR or the compile state; the driver applies its decisions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._errors import MODE_HINT, TypedPipelineError, planner_text, typed_error
from ._groups import UnsupportedGroupError, WarpLoadStoreUnsupportedError
from ._sites import Sites
from ._templates import LoadStoreRequest


@dataclass(frozen=True)
class LaunchBounds:
    """The kernel's ``launch_bounds``; ``rebind`` when the stage chose it."""

    value: int | None
    rebind: bool = False


@dataclass(frozen=True)
class Specialization:
    """What the kernel's sites specialized to, one entry per site.

    ``algorithms`` holds each distinct provider once, in first-use order;
    ``site_algorithms`` repeats them per site.
    """

    block: tuple[int, int, int]
    algorithms: tuple
    site_algorithms: tuple
    scratch: tuple[bool, ...]
    barrier_after: tuple[bool, ...]
    launch_bounds: LaunchBounds


# Request completion ----------------------------------------------------------

_COMPLETIONS: dict[tuple, tuple[Any, dict]] = {}


def _complete(request: LoadStoreRequest, block: tuple) -> tuple[Any, dict]:
    """Return the provider factory and its arguments for ``request``.

    ``plan_group_primitive`` hashes the request's semantic key, which is
    costly, so its result is memoized per request and block. A warp group
    plans to a warp provider, which takes the group's width as
    ``threads_in_warp``.
    """

    key = (request.key, block)
    completed = _COMPLETIONS.get(key)
    if completed is not None:
        return completed

    from ..._core import ArgumentBinding
    from ..._core.group._dispatch import plan_group_primitive
    from ..._core.group._model import GroupLoweringTarget, StorageOwnership
    from ..._core.launch import LaunchFacts
    from .._compiler import _group_errors as planner
    from .._lowering import _load_store
    from .._lowering._core import NumbaMlirCoreAdapter

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
        "offset": request.binding("offset") or omitted,
    }
    if request.binding("oob_default") is not None:
        kwargs["oob_default"] = request.binding("oob_default")
    completed = (getattr(_load_store, factory_name), kwargs)
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
            factory, kwargs = _complete(request, block)
            factory(**kwargs)
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
) -> Specialization:
    """Decide scratch, barriers and ``launch_bounds`` for specialized sites.

    A barrier follows every call that uses scratch.
    """

    launch_bounds = plan_launch_bounds(block, targetoptions)
    scratch = tuple(uses_scratch(a) for a in site_algorithms)
    return Specialization(
        block=block,
        algorithms=tuple(algorithms),
        site_algorithms=tuple(site_algorithms),
        scratch=scratch,
        barrier_after=scratch,
        launch_bounds=launch_bounds,
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
    groups against the block, before a warp site is refused.
    """

    unique = list(dict.fromkeys(s.request for s in sites.sites))
    for request in unique:
        _complete(request, block)
    for site in sites.sites:
        if site.request.group.is_warp:
            raise WarpLoadStoreUnsupportedError(
                f"coop.{site.kind} at line {site.line} on "
                f"{site.request.group}: warp load/store is not supported on "
                f"the typed pipeline yet. {MODE_HINT}"
            )
    algorithms = _instantiate(unique, block, target)
    algorithm_for = dict(zip(unique, algorithms))
    return plan(
        algorithms,
        [algorithm_for[s.request] for s in sites.sites],
        block,
        targetoptions,
    )


__all__ = [
    "LaunchBounds",
    "Specialization",
    "plan",
    "plan_launch_bounds",
    "specialize",
    "uses_scratch",
]
