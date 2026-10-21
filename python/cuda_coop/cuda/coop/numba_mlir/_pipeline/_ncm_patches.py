# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Patch Numba-CUDA-MLIR where the typed pipeline needs a compiler change.

Each patch replaces attributes of Numba-CUDA-MLIR classes with versions that
behave as an upstream ask would make them, and is listed with that ask in the
``_bridges`` table. A patch stands in for a fix that ``_ncm_fixes`` probes
for (``stand_in``): the typed pipeline's ``_activate`` applies the patches
for the fixes the compiler lacks, once, before it registers anything; the
default pipeline never imports this module, so it leaves the compiler
untouched.

A patch is pinned to what it was written against: Numba-CUDA-MLIR
``NCM_VERSION`` and a digest of the source of every attribute it replaces or
relies on. ``apply`` checks every pin before it changes anything, and raises
``TypedPipelineUnavailableError``, naming the patch and its ask, on the first
mismatch; ``stand_in`` adds the fix it stands in for.

A patch leaves kernels without cooperative types as they were in one of two
ways, which its docstring states:

* error to success: it acts only where the compiler would have refused the
  kernel;
* cooperative types only: it acts only on values whose type is, or is a
  tuple holding, a ``ThreadDataType``.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numba_cuda_mlir

from . import _bridges as nb
from ._ncm_fixes import Fix, present

NCM_VERSION = "0.5.4"

try:
    from numba_cuda_mlir.numba_cuda.core import typeinfer
except ImportError as exc:
    raise nb.TypedPipelineUnavailableError(
        "The cuda.coop typed pipeline patches Numba-CUDA-MLIR "
        f"{NCM_VERSION}; the installed compiler is missing {exc.name or exc}."
    ) from exc

from ._types import ThreadDataType


@dataclass(frozen=True)
class Pin:
    """An attribute a patch replaces or relies on, as it was written against.

    ``digest`` is the start of the SHA-256 of the attribute's source, or
    ``None`` when the class must not define the attribute itself.
    """

    owner: type
    name: str
    digest: str | None


@dataclass(frozen=True)
class Patch:
    """A compiler change, the ask that retires it, and its pins."""

    name: str
    ask: str
    pins: tuple[Pin, ...]
    build: Callable[[], tuple[tuple[type, str, Any], ...]]


def source_digest(obj: Any) -> str | None:
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _pinned(pin: Pin) -> bool:
    if pin.digest is None:
        return pin.name not in vars(pin.owner)
    return source_digest(vars(pin.owner).get(pin.name)) == pin.digest


def check(patch: Patch) -> None:
    """Raise unless ``patch`` matches the installed compiler."""

    installed = numba_cuda_mlir.__version__
    if installed != NCM_VERSION:
        changed = f"the installed compiler is {installed}"
    else:
        stale = [p for p in patch.pins if not _pinned(p)]
        if not stale:
            return
        owner = stale[0].owner
        owner = getattr(owner, "__qualname__", owner.__name__)
        changed = f"{owner}.{stale[0].name} has changed"
    raise nb.TypedPipelineUnavailableError(
        f"The cuda.coop typed pipeline's {patch.name!r} patch was written "
        f"against Numba-CUDA-MLIR {NCM_VERSION}, but {changed}. The patch is "
        f"retired by {patch.ask}; until then, unset "
        "CUDA_COOP_NUMBA_MLIR_PIPELINE to use the default pipeline."
    )


def holds_thread_data(ty: Any) -> bool:
    """Whether ``ty`` is a ``ThreadDataType`` or a tuple holding one."""

    if isinstance(ty, ThreadDataType):
        return True
    return isinstance(ty, nb.types.BaseTuple) and any(
        holds_thread_data(item) for item in ty.types
    )


# Payload through a tuple alias (U-H1-4) -------------------------------------


def _tuple_refinement() -> tuple[tuple[type, str, Any], ...]:
    """Refine a tuple's items from a refined projection of the tuple.

    Numba-CUDA-MLIR's ``static_getitem`` and ``build_tuple`` constraints
    have no ``refine`` hook, so refining ``x = (td,)[0]`` never reaches
    ``td``. With this patch a refined projection refines the tuple's item
    type, and a refined tuple refines the variables it was built from.

    Scope: cooperative types only. A constraint takes part only when the
    tuple holds a ``ThreadDataType``, and refines only items that do.

    Applied only on a compiler without tuple refinement of its own
    (``_ncm_fixes.TUPLE_REFINEMENT``).
    """

    static_getitem = typeinfer.StaticGetItemConstraint.__call__
    build_tuple = typeinfer.BuildTupleConstraint.__call__

    def static_getitem_call(self, typeinfer_):
        static_getitem(self, typeinfer_)
        tuple_var = typeinfer_.typevars[self.value.name]
        if (
            isinstance(self.index, int)
            and tuple_var.defined
            and holds_thread_data(tuple_var.getone())
        ):
            typeinfer_.refine_map.setdefault(self.target, self)

    def static_getitem_refine(self, typeinfer_, target_type):
        if not holds_thread_data(target_type):
            return
        items = list(typeinfer_.typevars[self.value.name].getone().types)
        items[self.index] = target_type
        typeinfer_.add_type(
            self.value.name,
            nb.types.BaseTuple.from_types(items),
            loc=self.loc,
        )

    def build_tuple_call(self, typeinfer_):
        build_tuple(self, typeinfer_)
        target = typeinfer_.typevars[self.target]
        if target.defined and holds_thread_data(target.getone()):
            typeinfer_.refine_map.setdefault(self.target, self)

    def build_tuple_refine(self, typeinfer_, target_type):
        if not isinstance(target_type, nb.types.BaseTuple):
            return
        for item, item_type in zip(self.items, target_type.types):
            if holds_thread_data(item_type) and item_type.is_precise():
                typeinfer_.add_type(
                    item.name, item_type, loc=self.loc, unless_locked=True
                )

    return (
        (typeinfer.StaticGetItemConstraint, "__call__", static_getitem_call),
        (typeinfer.StaticGetItemConstraint, "refine", static_getitem_refine),
        (typeinfer.BuildTupleConstraint, "__call__", build_tuple_call),
        (typeinfer.BuildTupleConstraint, "refine", build_tuple_refine),
    )


TUPLE_REFINEMENT = Patch(
    name="tuple refinement",
    ask="U-H1-4 (refine hooks for tuples)",
    pins=(
        Pin(typeinfer.StaticGetItemConstraint, "__call__", "8b86dc739bada065"),
        Pin(typeinfer.StaticGetItemConstraint, "refine", None),
        Pin(typeinfer.BuildTupleConstraint, "__call__", "c54497d13b566d89"),
        Pin(typeinfer.BuildTupleConstraint, "refine", None),
    ),
    build=_tuple_refinement,
)

_ABSENT = object()
_replaced: list[tuple[type, str, Any]] = []


def stand_in(fix: Fix, patch: Patch) -> tuple[Patch, ...]:
    """The patch to apply for ``fix``: none if the compiler has the fix.

    Otherwise ``patch``, which must match the installed compiler; if it
    does not, raise ``TypedPipelineUnavailableError`` naming the fix.
    """

    if present(fix):
        return ()
    try:
        check(patch)
    except nb.TypedPipelineUnavailableError as exc:
        raise nb.TypedPipelineUnavailableError(
            f"Numba-CUDA-MLIR {numba_cuda_mlir.__version__} does not have "
            f"the {fix.name} fix ({fix.reference}), and the patch that "
            f"stands in for it does not apply: {exc}"
        ) from exc
    return (patch,)


def apply(*patches: Patch) -> None:
    """Check every patch, then apply them all; later calls do nothing."""

    if _replaced:
        return
    for patch in patches:
        check(patch)
    for patch in patches:
        for owner, name, value in patch.build():
            _replaced.append((owner, name, vars(owner).get(name, _ABSENT)))
            setattr(owner, name, value)


def replaced() -> tuple[tuple[type, str], ...]:
    """The ``(class, attribute)`` pairs ``apply`` replaced."""

    return tuple((owner, name) for owner, name, _ in _replaced)


__all__ = [
    "NCM_VERSION",
    "TUPLE_REFINEMENT",
    "Patch",
    "Pin",
    "apply",
    "check",
    "holds_thread_data",
    "replaced",
    "source_digest",
    "stand_in",
]
