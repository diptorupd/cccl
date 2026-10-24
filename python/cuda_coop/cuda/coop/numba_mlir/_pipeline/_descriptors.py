# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trace each ``temp_storage=`` operand to the ``TempStorage`` it names.

A ``TempStorage`` is a compile-time descriptor: it has no run-time value,
only options that decide where a call's scratch goes. Typing gives it a
``TempStorageType`` and lets it flow through aliases and joins; this part of
site analysis, after typing and inlining, finds the constructors each
cooperative call's ``temp_storage=`` reaches and checks the descriptors as
coop's planner does, with its classes:

* every path to the operand comes from a constructor, never ``None`` or
  another value (``TempStorageBindingError``);
* the constructors one operand reaches have the same options, and one with
  ``auto_sync=False`` is reached from a single constructor;
* a descriptor is only aliased, joined or passed as ``temp_storage=``:
  passing it to a device function Numba did not inline, or any other use,
  is refused (``TempStorageError``), as is a constructor no cooperative call
  consumes.

Options typing could not fold, such as ``64 * 1024`` or a NumPy integer, are
read from the IR as constants. Constructors that one operand reaches are one
descriptor, named after the first of them in program order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from ._errors import TempStorageBindingError, TempStorageError
from ._types import is_temp_storage


@dataclass(frozen=True, eq=False)
class Descriptor:
    """One ``TempStorage``, with its constructor's validated options.

    ``index`` orders descriptors by their first constructor in the
    function; ``loc`` is that constructor's location.
    """

    name: str
    index: int
    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: str
    loc: Any

    @property
    def options(self) -> tuple:
        """The options as a request carries them."""

        return (
            self.size_in_bytes,
            self.alignment,
            self.auto_sync,
            self.sharing,
        )


def _temp_storage_objects() -> frozenset:
    from ._templates import kernel_primitives

    return frozenset(
        obj
        for obj, name in kernel_primitives().items()
        if name.endswith(".TempStorage")
    )


def _callee(typemap: dict, expr: Any) -> Any:
    ty = typemap.get(expr.func.name)
    if isinstance(ty, nb.types.Function):
        return ty.typing_key
    return None


def _omitted(ty: Any) -> bool:
    return ty is None or isinstance(ty, (nb.types.NoneType, nb.types.Omitted))


def _user_name(name: str) -> str:
    return name.split(".", 1)[0]


class _Constructor:
    """A ``TempStorage(...)`` call and its options, read on first use."""

    def __init__(self, stmt: Any, index: int) -> None:
        self.stmt, self.index = stmt, index
        self._options = None

    @property
    def name(self) -> str:
        return _user_name(self.stmt.target.name)

    def options(self, func_ir: Any, typemap: dict) -> tuple:
        if self._options is None:
            self._options = _read_options(self.stmt, func_ir, typemap)
        return self._options


def _read_options(stmt: Any, func_ir: Any, typemap: dict) -> tuple:
    from .._compiler._descriptor_provenance import temp_storage_constructor
    from .._compiler._rewrite_support import _INFERENCE_EXCEPTIONS

    loc = stmt.loc

    def constant(value, *, name):
        try:
            return func_ir.infer_constant(value)
        except _INFERENCE_EXCEPTIONS:
            pass
        ty = typemap.get(value.name)
        if isinstance(ty, nb.types.Literal):
            return ty.literal_value
        if _omitted(ty):
            return None
        raise TempStorageError(
            f"TempStorage {name} must be a compile-time literal.", loc=loc
        )

    try:
        descriptor = temp_storage_constructor(
            stmt.value,
            constant,
            syntax_error=lambda message: TempStorageError(message, loc=loc),
        )
    except (TypeError, ValueError) as exc:
        raise TempStorageError(str(exc), loc=loc) from None
    return (
        descriptor.size_in_bytes,
        descriptor.alignment,
        descriptor.auto_sync,
        descriptor.sharing,
    )


def _check_uses(func_ir, typemap, constructors, consumers) -> None:
    """Refuse every use of a descriptor but an alias, a join or a consumer."""

    operands = {
        id(stmt): dict(stmt.value.kws).get("temp_storage")
        for stmt, _, _ in consumers
    }
    for label in sorted(func_ir.blocks):
        for stmt in func_ir.blocks[label].body:
            value = getattr(stmt, "value", None)
            if id(value) in constructors:
                continue
            target = getattr(stmt, "target", None)
            used = [
                var
                for var in stmt.list_vars()
                if is_temp_storage(typemap.get(var.name))
                and not (isinstance(stmt, nb.ir.Assign) and var is target)
            ]
            if not used:
                continue
            if isinstance(stmt, nb.ir.Assign) and (
                isinstance(value, nb.ir.Var)
                or (
                    isinstance(value, nb.ir.Expr)
                    and value.op in ("cast", "phi", "exhaust_iter")
                )
            ):
                continue
            operand = operands.get(id(stmt))
            if operand is not None and all(
                var.name == operand.name for var in used
            ):
                continue
            names = ", ".join(sorted({_user_name(var.name) for var in used}))
            callee = (
                typemap.get(value.func.name)
                if isinstance(value, nb.ir.Expr) and value.op == "call"
                else None
            )
            if isinstance(callee, nb.types.Dispatcher):
                helper = callee.dispatcher.py_func.__qualname__
                raise TempStorageError(
                    f"TempStorage descriptor {names!r} is passed to a device "
                    "function that was not inlined into this kernel "
                    f"({helper!r}); let Numba-CUDA-MLIR inline the collective "
                    "helper (inline='always') or move its cooperative calls "
                    "into the kernel.",
                    loc=stmt.loc,
                )
            raise TempStorageError(
                "TempStorage values are opaque compile-time descriptors and "
                "may only be passed as temp_storage= to a registered "
                f"cooperative primitive; a use involving {names!r} would "
                "escape to runtime.",
                loc=stmt.loc,
            )


def analyze(
    func_ir: Any,
    typemap: dict,
    consumers: Sequence[tuple[Any, str, Any]],
) -> tuple[Descriptor | None, ...]:
    """Return the descriptor each consumer's ``temp_storage=`` names.

    ``consumers`` holds each cooperative call's assignment, kind and kernel
    location. A call without ``temp_storage=``, or with ``None``, gets
    ``None``.
    """

    objects = _temp_storage_objects()
    constructors: dict[int, _Constructor] = {}
    for label in sorted(func_ir.blocks):
        for stmt in func_ir.blocks[label].body:
            if (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, nb.ir.Expr)
                and stmt.value.op == "call"
                and _callee(typemap, stmt.value) in objects
            ):
                constructors[id(stmt.value)] = _Constructor(
                    stmt, len(constructors)
                )
    _check_uses(func_ir, typemap, constructors, consumers)

    definitions = func_ir._definitions
    reached: list[list[_Constructor] | None] = []
    for stmt, kind, loc in consumers:
        operand = dict(stmt.value.kws).get("temp_storage")
        ty = None if operand is None else typemap.get(operand.name)
        if _omitted(ty):
            reached.append(None)
            continue
        if not is_temp_storage(ty):
            raise TempStorageBindingError(
                f"cuda.coop.numba_mlir.{kind} temp_storage must resolve to "
                "a compile-time TempStorage descriptor",
                loc=loc,
            )
        reached.append(_reach(operand, definitions, constructors, loc))

    for found in reached:
        if found is None:
            continue
        options = {c.options(func_ir, typemap) for c in found}
        if len(options) > 1:
            raise TempStorageBindingError(
                "cuda.coop.numba_mlir TempStorage aliases have inconsistent "
                "contracts",
                loc=found[0].stmt.loc,
            )
        if len(found) > 1 and options.pop()[2] is False:
            raise TempStorageBindingError(
                "cuda.coop.numba_mlir TempStorage with auto_sync=False must "
                f"be constructed at exactly one site; {found[0].name!r} "
                f"reaches {len(found)} constructor sites. The compiler cannot "
                "verify caller synchronization when it merges these regions. "
                "Construct the descriptor once or set auto_sync=True.",
                loc=found[0].stmt.loc,
            )

    consumed = {c.index for found in reached if found for c in found}
    unconsumed = [c for c in constructors.values() if c.index not in consumed]
    if unconsumed:
        names = ", ".join(sorted({c.name for c in unconsumed}))
        raise TempStorageError(
            "TempStorage values are opaque compile-time descriptors and must "
            "be passed as temp_storage= to a registered cooperative "
            f"primitive; constructor(s) {names!r} have no primitive consumer.",
            loc=unconsumed[0].stmt.loc,
        )
    return _merge(reached, func_ir, typemap)


def _reach(operand, definitions, constructors, loc) -> list[_Constructor]:
    from .._compiler._descriptor_provenance import descriptor_definitions

    found, others = {}, []
    for owner, leaf in descriptor_definitions(
        operand, lambda var: definitions.get(var.name, ())
    ):
        constructor = constructors.get(id(leaf))
        if constructor is not None:
            found[constructor.index] = constructor
        else:
            others.append(owner)
    if not found:
        raise TempStorageError(
            "cooperative group temp_storage= must originate from a "
            "TempStorage constructor in the compiled function.",
            loc=loc,
        )
    if others:
        name = _user_name(others[0] or operand.name)
        if name.startswith("$"):
            name = next(iter(found.values())).name
        raise TempStorageBindingError(
            "cuda.coop.numba_mlir TempStorage variables must be bound to a "
            f"TempStorage descriptor on every path; {name!r} is also bound "
            "to a non-descriptor value such as None. Remove the None "
            "initializer or construct the descriptor unconditionally.",
            loc=loc,
        )
    return [found[i] for i in sorted(found)]


def _merge(reached, func_ir, typemap) -> tuple[Descriptor | None, ...]:
    """Make constructors that share a consumer one descriptor."""

    parent: dict[int, int] = {}

    def root(index: int) -> int:
        while parent.setdefault(index, index) != index:
            index = parent[index]
        return index

    for found in reached:
        for constructor in found or ():
            a, b = root(found[0].index), root(constructor.index)
            parent[max(a, b)] = min(a, b)
    first = {c.index: c for found in reached if found for c in found}
    descriptors: dict[int, Descriptor] = {}
    result = []
    for found in reached:
        if found is None:
            result.append(None)
            continue
        index = root(found[0].index)
        descriptor = descriptors.get(index)
        if descriptor is None:
            constructor = first[index]
            size, alignment, auto_sync, sharing = constructor.options(
                func_ir, typemap
            )
            descriptor = descriptors[index] = Descriptor(
                constructor.name,
                index,
                size,
                alignment,
                auto_sync,
                sharing,
                constructor.stmt.loc,
            )
        result.append(descriptor)
    return tuple(result)


def check_device_function(args: Sequence[Any], func_id: Any) -> None:
    """Refuse a separately compiled device function given a descriptor."""

    names = [
        name for name, ty in zip(func_id.arg_names, args) if is_temp_storage(ty)
    ]
    if names:
        raise TempStorageError(
            f"TempStorage descriptor {', '.join(names)!r} is passed to a "
            "device function that was not inlined into this kernel "
            f"({func_id.func_qualname!r}); let Numba-CUDA-MLIR inline the "
            "collective helper (inline='always') or move its cooperative "
            "calls into the kernel."
        )


__all__ = ["Descriptor", "analyze", "check_device_function"]
