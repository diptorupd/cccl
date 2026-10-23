# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Size each site's scratch and lay it out in shared memory.

Storage analysis is the third stage of ``CoopCallResolution``, and the only
place the typed pipeline calls NVRTC. It compiles the kernel's providers in
one bundle (``_link``), which also reports each provider's scratch size and
alignment, and lays scratch out as coop's planner does:

* one region per ``TempStorage`` descriptor whose sites take scratch, in
  constructor order. Sites of a ``sharing="shared"`` descriptor reuse its
  bytes; each site of an ``"exclusive"`` one gets its own slice. An omitted
  ``size_in_bytes`` is the sites' requirement; an explicit one must cover
  it;
* then one implicit region for the sites without ``temp_storage=``, sized
  to their largest need;
* all regions in one ``cuda.shared.array`` backing. Up to the device's
  static limit it is a static array; above it, the backing is the launch's
  dynamic shared memory, up to the device's opt-in limit.

Numba-CUDA-MLIR does not keep the dynamic shared-memory window apart from
static allocations: its dynamic window is an internal one-byte static
variable, so a static array would overlap the window. A dynamic backing is
therefore refused next to any user shared array, and a static one next to a
user array that is itself dynamic, as coop's planner refuses both. The
launch that compiles a kernel with a dynamic backing must also request the
bytes itself (``check_launch``).

Which sites take scratch was decided before the compile, from each
provider's storage ABI; the compiled layouts must agree. ``layout`` is pure
and takes plain sizes. This stage does not change the IR.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from . import _link
from ._errors import (
    SharedMemoryAliasError,
    TempStorageError,
    TypedPipelineError,
)
from ._plan import Backing, Region
from ._sites import Sites, UserSharedArray
from ._specialize import Specialization

STATIC_SHARED_LIMIT = 48 * 1024
IMPLICIT_REGION = "implicit.0"
_BACKING_ALIGNMENT = 16


@dataclass(frozen=True)
class Slice:
    """A site's scratch: ``size`` bytes at ``offset`` in the backing.

    ``region_offset`` is the same offset from the start of ``region``.
    """

    region: str
    offset: int
    size: int
    region_offset: int = 0


@dataclass(frozen=True)
class StorageLayout:
    """The kernel's scratch backing and regions, and each site's slice.

    ``backing`` is ``None`` and ``regions`` empty when no site takes
    scratch.
    """

    backing: Backing | None
    regions: tuple[Region, ...]
    slices: tuple[Slice | None, ...]

    @property
    def scratch(self) -> tuple[bool, ...]:
        return tuple(s is not None for s in self.slices)

    @property
    def dynamic_shared_bytes(self) -> int:
        if self.backing is None or self.backing.kind != "dynamic":
            return 0
        return self.backing.size


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def device_limits(required: int) -> tuple[int, int]:
    """Return the device's default and opt-in shared memory per block.

    As in coop's planner, a backing within the static limit needs no
    device, and the query is coop's own, so a test can substitute it.
    """

    from .._compiler import _rewrite_storage

    if required <= STATIC_SHARED_LIMIT:
        return STATIC_SHARED_LIMIT, STATIC_SHARED_LIMIT
    try:
        limits = _rewrite_storage._query_device_shared_memory_limits()
        default = int(limits["max_default_shared_memory_per_block"])
        optin = int(limits["max_optin_shared_memory_per_block"])
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise TempStorageError(
            f"TempStorage requirements above the conservative "
            f"{STATIC_SHARED_LIMIT}-byte static shared-memory limit require "
            "an exact current-device shared-memory query"
        ) from exc
    if default <= 0 or optin <= 0 or optin < default:
        raise TempStorageError(
            "The current device reported invalid shared-memory limits: "
            f"default={default}, opt-in={optin}."
        )
    return default, optin


def _explicit_region(
    descriptor: Any, uses: list[tuple[int, int]]
) -> tuple[int, int, list[int]]:
    """Size a descriptor's region; return it with each use's offset in it."""

    from .._compiler._rewrite_support import (
        _MIN_TEMP_STORAGE_ALIGNMENT,
        CoopSinglePhaseRewriteError,
        _default_temp_storage_alignment,
        _normalize_temp_storage_alignment,
        _validate_temp_storage_alignment,
    )

    required_alignment = max(_MIN_TEMP_STORAGE_ALIGNMENT, *(a for _, a in uses))
    if descriptor.sharing == "exclusive":
        offsets, required = [], 0
        for size, alignment in uses:
            required = _align_up(
                required, max(_MIN_TEMP_STORAGE_ALIGNMENT, alignment)
            )
            offsets.append(required)
            required += size
    else:
        offsets, required = [0] * len(uses), max(s for s, _ in uses)
    size = descriptor.size_in_bytes
    if size is None:
        size = required
    elif size < required:
        raise TempStorageError(
            "TempStorage size_in_bytes is smaller than required by "
            f"primitive uses ({size} < {required}).",
            loc=descriptor.loc,
        )
    alignment = _default_temp_storage_alignment(required_alignment)
    try:
        if descriptor.alignment is not None:
            alignment = max(
                alignment,
                _normalize_temp_storage_alignment(descriptor.alignment),
            )
        _validate_temp_storage_alignment(alignment)
    except CoopSinglePhaseRewriteError as exc:
        raise TempStorageError(str(exc), loc=descriptor.loc) from None
    return size, alignment, offsets


def _alias_error(backing: Backing, conflicts: Sequence[UserSharedArray]):
    where = ", ".join(
        f"{'static' if array.static else 'dynamic/runtime-sized'} "
        f"cuda.shared.array(...) at {array.loc}"
        for array in conflicts[:3]
    )
    return SharedMemoryAliasError(
        f"cuda.coop temporary storage requires a {backing.size}-byte "
        f"{backing.kind} shared-memory backing, but this kernel also declares "
        f"{where}. The supported numba-cuda-mlir compiler does not separate "
        "these allocations; they would alias. Use statically sized user "
        "shared arrays and keep the combined cooperative backing within the "
        "static shared-memory limit, or move the user data out of shared "
        "memory.",
        loc=conflicts[0].loc,
    )


def layout(
    needs: Sequence[tuple[int, int] | None],
    storage: Sequence[Any] | None = None,
    user_shared: Sequence[UserSharedArray] = (),
    limits: Callable[[int], tuple[int, int]] = device_limits,
) -> StorageLayout:
    """Lay out the sites' scratch ``needs``, one ``(size, alignment)`` each.

    A site without scratch needs ``None``. ``storage`` holds each site's
    ``TempStorage`` descriptor, or ``None`` for implicit scratch (the
    default for every site). ``user_shared`` describes the kernel's own
    ``cuda.shared.array`` calls; ``limits`` returns the device's default and
    opt-in shared memory for a backing of the given size.
    """

    if storage is None:
        storage = [None] * len(needs)
    slices: list[Slice | None] = [None] * len(needs)
    uses: dict[Any, list[int]] = {}
    for index, (need, descriptor) in enumerate(
        zip(needs, storage, strict=True)
    ):
        if need is not None:
            uses.setdefault(descriptor, []).append(index)
    if not uses:
        return StorageLayout(None, (), tuple(slices))

    regions, offset, max_alignment = [], 0, 1
    explicit = sorted((d for d in uses if d is not None), key=lambda d: d.index)
    for descriptor in [*explicit, None]:
        indexes = uses.get(descriptor)
        if not indexes:
            continue
        site_needs = [needs[i] for i in indexes]
        if descriptor is None:
            name = IMPLICIT_REGION
            size = max(s for s, _ in site_needs)
            alignment = max(a for _, a in site_needs)
            offsets = [0] * len(indexes)
        else:
            name = f"explicit.{len(regions)}"
            size, alignment, offsets = _explicit_region(descriptor, site_needs)
        offset = _align_up(offset, alignment)
        regions.append(Region(name, offset, size, alignment))
        for i, at, (need_size, _) in zip(indexes, offsets, site_needs):
            slices[i] = Slice(name, offset + at, need_size, at)
        offset += size
        max_alignment = max(max_alignment, alignment)

    total = _align_up(offset, max_alignment)
    static_alignment = max(_BACKING_ALIGNMENT, max_alignment)
    static_size = _align_up(total, static_alignment)
    default, optin = limits(static_size)
    user_static = sum(a.size_in_bytes for a in user_shared if a.static)
    user_dynamic = [a for a in user_shared if not a.static]
    if static_size <= default:
        backing = Backing("static", static_size, static_alignment)
        if user_dynamic:
            raise _alias_error(backing, user_dynamic)
        if backing.size + user_static > default:
            raise TypedPipelineError(
                f"cooperative scratch needs {backing.size} bytes of static "
                f"shared memory, which with the kernel's own {user_static} "
                f"bytes of static cuda.shared.array exceeds the {default}-"
                "byte static limit. Scratch moves to dynamic shared memory "
                "only when it alone exceeds that limit and the kernel has no "
                "shared arrays of its own; reduce the scratch (a smaller "
                "block, fewer items per thread, or a storage-free algorithm) "
                "or the kernel's own shared arrays."
            )
    else:
        backing = Backing("dynamic", total, _BACKING_ALIGNMENT)
        if max_alignment > _BACKING_ALIGNMENT:
            raise TempStorageError(
                f"TempStorage requires {max_alignment}-byte alignment, but "
                f"the {total}-byte backing exceeds the {default}-byte static "
                "shared-memory limit and dynamic shared memory guarantees "
                f"only {_BACKING_ALIGNMENT}-byte alignment; reduce the "
                "requested alignment or the storage size."
            )
        if total > optin:
            raise TempStorageError(
                f"TempStorage requires {total} bytes dynamic shared memory, "
                f"but device max opt-in is {optin} bytes."
            )
        if user_shared:
            raise _alias_error(backing, user_shared)
    return StorageLayout(backing, tuple(regions), tuple(slices))


def check_launch(storage: StorageLayout, sharedmem: int) -> None:
    """Require a launch that compiles a dynamic backing to request it.

    Numba-CUDA-MLIR applies a compile's required dynamic shared memory to
    later launches, but not to the launch that runs the compile unless a
    whole-function planner is registered, and the typed pipeline registers
    none. That launch would run with ``sharedmem`` bytes and fault.
    """

    required = storage.dynamic_shared_bytes
    if required <= sharedmem:
        return
    raise TempStorageError(
        f"cooperative scratch needs {required} bytes of dynamic shared "
        f"memory, above the {STATIC_SHARED_LIMIT}-byte static limit, but the "
        f"launch requests {sharedmem}. Numba-CUDA-MLIR does not apply a "
        "compile's dynamic shared-memory requirement to the launch that "
        "compiles the kernel, so on the typed pipeline the launch must "
        f"request it: kernel[grid, block, stream, {required}](...). Or "
        "reduce the scratch to the static limit."
    )


def analyze(
    specialization: Specialization,
    sites: Sites,
    identity: Any,
    context: Any,
) -> tuple[StorageLayout, _link.Bundle]:
    """Compile the specialization's bundle and lay out its scratch."""

    bundle = _link.compile_bundle(specialization.algorithms, identity, context)
    needs = []
    for algorithm, scratch in zip(
        specialization.site_algorithms, specialization.scratch, strict=True
    ):
        need = bundle.layouts[algorithm._private_symbol_digest]
        if (need is not None) != scratch:
            raise AssertionError(
                f"{algorithm.c_name}: its {algorithm.storage_abi} storage "
                f"ABI disagrees with its compiled scratch layout {need}"
            )
        needs.append(need)
    storage = [site.storage for site in sites.sites]
    return layout(needs, storage, sites.user_shared), bundle


__all__ = [
    "IMPLICIT_REGION",
    "STATIC_SHARED_LIMIT",
    "Slice",
    "StorageLayout",
    "analyze",
    "check_launch",
    "device_limits",
    "layout",
]
