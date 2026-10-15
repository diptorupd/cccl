# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Retype a kernel whose payload reads were typed before the payload's dtype.

A ``ThreadData`` without ``dtype=`` takes its dtype from its strongest use,
so a ``load`` into it outranks an earlier indexed write. Numba types a read
while only the write is known and never narrows the read afterwards, so the
read can keep the write's dtype after the payload has taken the load's. As
in NumPy, a read must have its array's dtype.

When that happens, this stage writes each such payload's final dtype into
its constructor call as ``dtype=``, as if the kernel's source had passed it,
and retypes the function with Numba's own type inference. A kernel whose
reads already agree is left as typing produced it.
"""

from __future__ import annotations

from typing import Any

from . import _bridges as nb
from ._templates import kernel_primitives
from ._types import ThreadDataType


def stale_reads(func_ir: Any, typemap: dict):
    """Yield ``(stmt, read type, payload type)`` for each read that disagrees."""

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
            result = nb.unliteral(result)
            if result != payload.dtype:
                yield stmt, result, payload


def _constructors() -> frozenset:
    return frozenset(
        obj
        for obj, name in kernel_primitives().items()
        if name.rsplit(".", 1)[1] == "ThreadData"
    )


def _pin_dtypes(func_ir: Any, typemap: dict) -> bool:
    """Add ``dtype=`` to every inferred constructor; return whether any was."""

    constructors = _constructors()
    pinned = False
    for block in func_ir.blocks.values():
        body = []
        for stmt in block.body:
            expr = stmt.value if isinstance(stmt, nb.ir.Assign) else None
            payload = typemap.get(stmt.target.name) if expr is not None else None
            if (
                isinstance(expr, nb.ir.Expr)
                and expr.op == "call"
                and isinstance(payload, ThreadDataType)
                and payload.dtype.is_precise()
                and len(expr.args) < 2
                and "dtype" not in dict(expr.kws)
                and getattr(typemap.get(expr.func.name), "typing_key", None)
                in constructors
            ):
                dtype = block.scope.redefine("$coop_dtype", stmt.loc)
                body.append(
                    nb.ir.Assign(
                        nb.ir.Global(str(payload.dtype), payload.dtype, stmt.loc),
                        dtype,
                        stmt.loc,
                    )
                )
                expr.kws = [*expr.kws, ("dtype", dtype)]
                pinned = True
            body.append(stmt)
        block.body = body
    return pinned


def settle(state: Any, func_ir: Any, typemap: dict, calltypes: Any) -> None:
    """Retype the function once if a payload read disagrees with its payload."""

    if next(stale_reads(func_ir, typemap), None) is None:
        return
    if not _pin_dtypes(func_ir, typemap):
        return
    typed = nb.type_inference_stage(
        state.typingctx,
        state.targetctx,
        func_ir,
        state.args,
        state.return_type,
        state.locals,
    )
    typemap.clear()
    typemap.update(typed.typemap)
    calltypes.clear()
    calltypes.update(typed.calltypes)


__all__ = ["settle", "stale_reads"]
