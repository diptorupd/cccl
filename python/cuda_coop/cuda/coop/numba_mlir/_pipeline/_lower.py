# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower typed cooperative calls to MLIR.

Lowering takes no decisions. Everything it needs per call site (the
provider symbol and ABI, the scratch operand, the barrier, the link items)
was decided by call resolution and stored in
``state.metadata["coop_plan"]``:

* a group (``this_block()``, ``this_warp()``, ``group_by()`` and the other
  ``this_*()`` factories) and a ``TempStorage(...)`` are a placeholder
  ``i8`` constant;
* ``ThreadData(...)`` is a ``memref.alloca`` of its type's extent, aligned as
  the constructor's own ``alignment=`` asks, and its ``items_per_thread``
  is that extent as a constant;
* a ``load`` or ``store`` is a ``func.call`` to the provider, with the
  scratch pointer first when the provider takes one, then the memory and
  payload pointers, then the runtime controls converted to their ABI
  types, followed by a ``gpu.barrier`` when the plan has one. A scalar
  ``store`` value is boxed into a one-item array first. On a warp group
  the offset is the group tile's origin and the scratch pointer the
  group instance's slot, both computed from the thread index as the plan
  describes them, and the barrier is a ``syncwarp`` over the group's
  lanes.
"""

from __future__ import annotations

import struct
from typing import Any

from . import _bridges as nb
from ._templates import (
    _CONSTRUCTOR_PARAMETERS,
    GROUP_FACTORIES,
    OPERATIONS,
    constructor_alignment,
    kernel_primitives,
)
from ._types import ThreadDataType

SCRATCH_KEYWORD = "__coop_scratch"


def lower_placeholder(builder, target, args, kwargs):
    builder.store_var(
        target, nb.arith.constant(nb.mlir_ir.IntegerType.get_signless(8), 0)
    )


def _lower_thread_data(builder, target, args, kwargs):
    payload = builder.get_numba_type(target.name)
    if not isinstance(payload, ThreadDataType) or not payload.is_precise():
        raise AssertionError(f"coop.ThreadData lowered as {payload}")
    bound = dict(
        zip(
            _CONSTRUCTOR_PARAMETERS,
            (builder.get_numba_type(a.name) for a in args),
        )
    )
    bound.update({k: builder.get_numba_type(v.name) for k, v in (kwargs or [])})
    alignment = constructor_alignment("coop.ThreadData", bound.get("alignment"))
    element = builder.get_storage_type(payload.dtype)
    memref_type = nb.mlir_ir.MemRefType.get([payload.items], element)
    with builder.alloca_insertion_point():
        if alignment is None:
            value = nb.memref.alloca(memref_type, [], [])
        else:
            value = nb.memref.alloca(
                memref_type,
                [],
                [],
                alignment=max(struct.calcsize("P"), alignment),
            )
    builder.store_var(target, value)


def _lower_items_per_thread(context, builder, target, value):
    extent = builder.get_numba_type(target.name)
    builder.store_var(
        target,
        nb.arith.constant(builder.get_mlir_type(extent), extent.literal_value),
    )


def _call_entry(builder, target) -> Any:
    plan = builder.metadata.get("coop_plan")
    entry = None if plan is None else plan.calls.get(target.name)
    if entry is None:
        raise AssertionError(
            "a cooperative call reached lowering without a storage plan"
        )
    return entry


def _box(builder, var, dtype, value):
    element = builder.get_storage_type(dtype)
    with builder.alloca_insertion_point():
        array = nb.memref.alloca(
            nb.mlir_ir.MemRefType.get([1], element), [], []
        )
    if value is None:
        item = nb.cast_value(
            builder,
            builder.load_var(var),
            builder.get_numba_type(var.name),
            dtype,
        )
    else:
        item = nb.arith.constant(element, value.item())
    zero = nb.arith.constant(nb.mlir_ir.IndexType.get(), 0)
    nb.memref.store(item, array, [zero])
    return array


def _keyword_value(builder, var, abi_type):
    return nb.cast_value(
        builder,
        builder.load_var(var),
        builder.get_numba_type(var.name),
        abi_type,
    )


def _index(value):
    return nb.arith.constant(nb.mlir_ir.IndexType.get(), value)


def _linear_rank(block):
    """The x-major rank of the thread in ``block``, as an ``index``."""

    def thread_id(axis):
        return nb.gpu.thread_id(getattr(nb.gpu.Dimension, axis))

    rank = thread_id("x")
    if block[1] > 1 or block[2] > 1:
        yz = nb.arith.addi(
            thread_id("y"), nb.arith.muli(_index(block[1]), thread_id("z"))
        )
        rank = nb.arith.addi(rank, nb.arith.muli(_index(block[0]), yz))
    return rank


def _instance(width, block):
    return nb.arith.divui(_linear_rank(block), _index(width))


def _tile_offset(builder, operand, abi_type, keywords):
    _, width, items, block, base = operand
    origin = nb.arith.muli(_instance(width, block), _index(items))
    origin = nb.arith.index_cast(builder.get_mlir_type(abi_type), origin)
    if base == 0:
        return origin
    if isinstance(base, int):
        base = nb.arith.constant(builder.get_mlir_type(abi_type), base)
    else:
        base = _keyword_value(builder, keywords[base[1]], abi_type)
    return nb.arith.addi(origin, base)


def _operand(builder, operand, abi_type, args, keywords):
    tag = operand[0]
    if tag == "tile_offset":
        return _tile_offset(builder, operand, abi_type, keywords)
    if tag == "scratch":
        pointer = nb.memref_data_pointer(
            builder.load_var(keywords[SCRATCH_KEYWORD])
        )
        if len(operand) == 1:
            return pointer
        _, width, stride, block = operand
        offset = nb.arith.muli(_instance(width, block), _index(stride))
        offset = nb.arith.index_cast(nb.mlir_types.i64(), offset)
        return nb.llvm_ptr_add_bytes(pointer, offset)
    if tag == "pointer":
        return nb.memref_data_pointer(builder.load_var(args[operand[1]]))
    if tag == "box":
        _, index, dtype, value = operand
        return nb.memref_data_pointer(_box(builder, args[index], dtype, value))
    return _keyword_value(builder, keywords[operand[1]], abi_type)


def _warp_mask(width, block):
    """The ``syncwarp`` mask of the thread's group: its lanes in the warp."""

    i32 = nb.mlir_types.i32()
    if width >= 32:
        return nb.arith.constant(i32, -1)
    lane = nb.arith.andi(_linear_rank(block), _index(31))
    shift = nb.arith.muli(nb.arith.divui(lane, _index(width)), _index(width))
    shift = nb.arith.index_cast(i32, shift)
    return nb.arith.shli(nb.arith.constant(i32, (1 << width) - 1), shift)


def _lower_operation(builder, target, args, kwargs):
    entry = _call_entry(builder, target)
    keywords = dict(kwargs or [])
    operands = [
        _operand(builder, operand, abi_type, args, keywords)
        for operand, abi_type in zip(entry.operands, entry.abi_arguments)
    ]
    function_type = nb.external_abi_function_type(
        builder, nb.signature(nb.types.void, *entry.abi_arguments)
    )
    with builder.alloca_insertion_point():
        result = nb.llvm.alloca(
            res=nb.llvm.PointerType.get(),
            array_size=nb.arith.constant(nb.mlir_types.i32(), 1),
            elem_type=nb.mlir_types.i8(),
        )
    callee = nb.get_or_insert_function(
        entry.symbol, function_type, builder.mlir_gpu_module
    )
    nb.func.call(
        result=list(function_type.results),
        callee=callee.name.value,
        operands_=[result, *operands],
    )
    if entry.barrier_after and entry.barrier_group is None:
        nb.gpu.barrier()
    elif entry.barrier_after:
        nb.nvvm.bar_warp_sync(_warp_mask(*entry.barrier_group))
    for item in entry.link_items:
        builder.link_external_item(item)


_registered = False


def register() -> None:
    """Register lowering for every typed cooperative call."""

    global _registered
    if _registered:
        return
    any_args = nb.types.VarArg(nb.types.Any)
    for obj, qualified_name in kernel_primitives().items():
        short = qualified_name.rsplit(".", 1)[1]
        if short in GROUP_FACTORIES:
            nb.lowering_registry.lower(obj)(lower_placeholder)
        elif short == "TempStorage":
            nb.lowering_registry.lower(obj, any_args)(lower_placeholder)
        elif short == "ThreadData":
            nb.lowering_registry.lower(obj, any_args)(_lower_thread_data)
    nb.lowering_registry.lower_getattr(ThreadDataType, "items_per_thread")(
        _lower_items_per_thread
    )
    for op in OPERATIONS:
        nb.lowering_registry.lower(op, any_args)(_lower_operation)
    _registered = True


__all__ = ["SCRATCH_KEYWORD", "lower_placeholder", "register"]
