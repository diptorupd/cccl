# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Decide the final type of each payload around Numba's type inference.

A ``ThreadData`` without ``dtype=`` takes its dtype from its strongest
evidence, so a ``load`` into it outranks an earlier indexed write. Numba
types each use of the payload while the evidence known so far holds and
never revisits it, so a read, a call or an iteration can keep the write's
dtype after the payload has taken the load's.

``seeds`` reads one typing result of a function and returns, for each
payload a stale use was typed against, the payload's final type, keyed by
its constructor's target variable. The typing pass then types the function
again with Numba's own type inference, passing those types as locals: each
payload is locked at its final type from the start, so every use types
against it. The IR is not changed.

Only payloads whose type rests on a ``load`` or a ``store`` are seeded. A
payload fixed by its writes alone has no stronger evidence to wait for,
and leaving it free lets a later round still refine it.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from typing import Any

from . import _bridges as nb
from ._templates import OPERATIONS, kernel_primitives
from ._types import EXPLICIT, WRITE, ThreadDataType


@cache
def _constructors() -> frozenset:
    return frozenset(
        obj
        for obj, name in kernel_primitives().items()
        if name.rsplit(".", 1)[1] == "ThreadData"
    )


@cache
def _cooperative_objects() -> frozenset:
    return frozenset(kernel_primitives()) | frozenset(OPERATIONS)


def _thread_data_in(ty: Any) -> Iterator[ThreadDataType]:
    if isinstance(ty, ThreadDataType):
        yield ty
    elif isinstance(ty, nb.types.BaseTuple):
        for item in ty.types:
            yield from _thread_data_in(item)


def holds_thread_data(typemap: dict) -> bool:
    """Whether any variable of a typing result holds a ``ThreadDataType``."""

    return any(next(_thread_data_in(ty), None) for ty in typemap.values())


def mentions_coop(func_ir: Any, args: tuple) -> bool:
    """Whether a function refers to a cooperative object or takes a payload.

    This is read from the IR, before typing, to decide whether a function
    that failed to type is worth typing again partially.
    """

    if any(next(_thread_data_in(ty), None) for ty in args):
        return True
    cooperative = _cooperative_objects()
    for block in func_ir.blocks.values():
        for stmt in block.body:
            value = getattr(stmt, "value", None)
            if isinstance(value, (nb.ir.Global, nb.ir.FreeVar)):
                try:
                    if value.value in cooperative:
                        return True
                except TypeError:
                    continue
    return False


def payloads(func_ir: Any, typemap: dict) -> dict[str, ThreadDataType]:
    """Map each inferred constructor's target to its payload's type."""

    constructors = _constructors()
    found = {}
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if not (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, nb.ir.Expr)
                and stmt.value.op == "call"
            ):
                continue
            callee = getattr(
                typemap.get(stmt.value.func.name), "typing_key", None
            )
            payload = typemap.get(stmt.target.name)
            if (
                callee in constructors
                and isinstance(payload, ThreadDataType)
                and payload.dtype.is_precise()
                and payload.strength != EXPLICIT
            ):
                found[stmt.target.name] = payload
    return found


def stale_reads(func_ir: Any, typemap: dict) -> Iterator[ThreadDataType]:
    """Yield the payload type of each read typed to another dtype."""

    for block in func_ir.blocks.values():
        for stmt in block.body:
            if not (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, nb.ir.Expr)
                and stmt.value.op in ("getitem", "static_getitem")
            ):
                continue
            payload = typemap.get(stmt.value.value.name)
            if not isinstance(payload, ThreadDataType):
                continue
            result = typemap.get(stmt.target.name)
            if isinstance(result, nb.types.Array):
                result = result.dtype
            if nb.unliteral(result) != payload.dtype:
                yield payload


def stale_calls(typemap: dict, calltypes: dict) -> Iterator[ThreadDataType]:
    """Yield the payload type of each call typed with another payload dtype.

    A call's signature keeps the payload type it was typed with; each of the
    call's variables whose payload has since changed dtype is stale.
    """

    for inst, sig in (calltypes or {}).items():
        if sig is None:
            continue
        seen = [
            td
            for ty in (*sig.args, getattr(sig, "recvr", None))
            for td in _thread_data_in(ty)
            if td.dtype.is_precise()
        ]
        if not seen:
            continue
        for var in inst.list_vars():
            for now in _thread_data_in(typemap.get(var.name)):
                if any(
                    td.items == now.items and td.dtype != now.dtype
                    for td in seen
                ):
                    yield now


def seeds(
    func_ir: Any, typemap: dict, calltypes: dict, failed: bool
) -> dict[str, ThreadDataType]:
    """Return the payload types to lock before typing the function again.

    A payload is chosen when a read or a call was typed against another
    dtype than its final one. When typing failed and no stale use is
    visible, every payload with a ``load`` or ``store`` is chosen: the
    failing use has no type to compare.
    """

    found = {
        name: ty
        for name, ty in payloads(func_ir, typemap).items()
        if ty.strength > WRITE
    }
    stale = {*stale_reads(func_ir, typemap), *stale_calls(typemap, calltypes)}
    if not stale and failed:
        return found
    return {name: ty for name, ty in found.items() if ty in stale}


__all__ = [
    "holds_thread_data",
    "mentions_coop",
    "payloads",
    "seeds",
    "stale_calls",
    "stale_reads",
]
