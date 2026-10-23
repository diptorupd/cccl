# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type ``group_by`` partitions of a group, and refuse bad ones after typing.

``group.group_by(count, exhaustive=True)`` on a ``this_block()`` or
``this_warp()`` value types as the ``CoopGroupType`` of the descriptor
coop's own ``ThreadGroup.group_by`` returns, so the partition rules (warp
or block parent, no nested partition, a count that fits the parent) are
coop's. ``this_warp().group_by(n)`` is a logical warp of ``n`` threads;
its width is part of its type.

A partition coop refuses does not fail its typing template: Numba-CUDA-MLIR
reports a template's exception as a plain ``TypingError``. It types as a
``RefusedGroupType`` instead, a load or store on it types too, and the
check ``refused_groups`` raises the refusal with coop's class at the last
fixed point of typing. The classes are the planner's (``GroupRewriteError``
for the call's syntax, ``NonConstantGroupArgumentError`` for a count or
``exhaustive`` that is not a compile-time constant, and the
``NotImplementedError``, ``TypeError`` or ``ValueError`` of coop's
descriptor), each also a ``TypedPipelineError``.

As on the planner, a count that is a kernel argument is not refused: typing
restarts with the argument as a literal.
"""

from __future__ import annotations

from typing import Any

from .._compiler._group_errors import (
    NonConstantGroupArgumentError,
    UnsupportedLoadStoreGroupError,
)
from .._compiler._group_planner_support import GroupRewriteError
from . import _bridges as nb
from . import _typing_rules as rules
from ._errors import TypedPipelineError
from ._types import CoopGroupType, RefusedGroupType

GROUP_BY_KEY = "coop.group.group_by"


def _init(self, msg, loc=None):
    # The planner classes' initializers format a message of their own and
    # would skip the one that records the location.
    TypedPipelineError.__init__(self, msg, loc=loc)


class GroupByError(GroupRewriteError, TypedPipelineError):
    """A ``group_by`` call coop's planner cannot read."""

    __init__ = _init


class NonConstantGroupByArgumentError(
    NonConstantGroupArgumentError, TypedPipelineError
):
    """A ``group_by`` count or ``exhaustive`` is not a compile-time constant."""

    __init__ = _init


class GroupByNotImplementedError(NotImplementedError, TypedPipelineError):
    """A partition coop does not support, such as a nested one."""

    __init__ = _init


class GroupByTypeError(TypeError, TypedPipelineError):
    """A ``group_by`` argument of the wrong type."""

    __init__ = _init


class GroupByValueError(ValueError, TypedPipelineError):
    """A ``group_by`` count that does not fit its parent."""

    __init__ = _init


class UnsupportedGroupError(UnsupportedLoadStoreGroupError, TypedPipelineError):
    """Coop's planning of a load or store refuses the call's group."""

    __init__ = _init


class WarpLoadStoreUnsupportedError(UnsupportedGroupError):
    """A load or store on a warp group, which the typed pipeline lacks."""


_DESCRIPTOR_ERRORS = (
    (NotImplementedError, GroupByNotImplementedError),
    (TypeError, GroupByTypeError),
    (ValueError, GroupByValueError),
)


def _constant(ty: Any, parameter: str) -> Any:
    """Return the value of a literal argument, or a ``RefusedGroupType``.

    Any other type is returned as is; coop's descriptor then rejects it
    with the message for the wrong type.
    """

    if isinstance(ty, nb.types.Literal):
        return ty.literal_value
    if isinstance(ty, (nb.types.Integer, nb.types.Boolean)):
        message = str(NonConstantGroupArgumentError(parameter))
        return RefusedGroupType(
            NonConstantGroupByArgumentError, message, parameter
        )
    return ty


def partition(parent: Any, args: tuple, kws: dict) -> Any:
    """Return the type of ``parent.group_by(*args, **kws)``."""

    if isinstance(parent, RefusedGroupType):
        return parent
    unknown = sorted(set(kws) - {"count", "exhaustive"})
    if unknown:
        return RefusedGroupType(
            GroupByError,
            "ThreadGroup.group_by got unexpected keyword(s): "
            + ", ".join(unknown),
        )
    if len(args) > 1:
        return RefusedGroupType(
            GroupByError,
            "ThreadGroup.group_by accepts one positional count argument",
        )
    if args and "count" in kws:
        return RefusedGroupType(
            GroupByError, "ThreadGroup.group_by received count more than once"
        )
    if not args and "count" not in kws:
        return RefusedGroupType(
            GroupByError, "ThreadGroup.group_by requires count"
        )
    count = _constant(args[0] if args else kws["count"], "count")
    exhaustive = True
    if "exhaustive" in kws:
        exhaustive = _constant(kws["exhaustive"], "exhaustive")
    for value in (count, exhaustive):
        if isinstance(value, RefusedGroupType):
            return value
    try:
        group = parent.group.group_by(count, exhaustive=exhaustive)
    except (NotImplementedError, TypeError, ValueError) as exc:
        error = next(
            c for base, c in _DESCRIPTOR_ERRORS if isinstance(exc, base)
        )
        return RefusedGroupType(error, str(exc))
    return CoopGroupType(group)


class _GroupBy(nb.AbstractTemplate):
    key = GROUP_BY_KEY
    prefer_literal = True

    def generic(self, args, kws):
        result = partition(self.this, args, kws)
        return nb.signature(result, *args, *kws.values(), recvr=self.this)


def _attributes(key: type) -> type:
    class GroupAttributes(nb.AttributeTemplate):
        def resolve_group_by(self, group):
            return nb.types.BoundFunction(_GroupBy, group)

    GroupAttributes.key = key
    GroupAttributes.__name__ = f"CoopGroupAttributes_{key.__name__}"
    return GroupAttributes


def _literal_retry(refused: RefusedGroupType, calls: list, func_ir: Any):
    """The restart of typing with a kernel-argument count as a literal."""

    for call in calls:
        bound = dict(zip(("count",), call.args))
        bound.update(call.kws)
        var = bound.get(refused.parameter)
        if var is None:
            continue
        try:
            definition = func_ir.get_definition(var)
        except KeyError:
            continue
        if isinstance(definition, nb.ir.Arg):
            return nb.errors.ForceLiteralArg({definition.index})
    return None


def refused_groups(typeinfer_, matching, released):
    """Check: a group coop refuses is an error, with coop's class."""

    func_ir = typeinfer_.func_ir
    calls: dict[RefusedGroupType, list] = {}
    for name, ty in matching.items():
        if not isinstance(ty, RefusedGroupType):
            continue
        try:
            definition = func_ir.get_definition(name)
        except KeyError:
            continue
        if isinstance(definition, nb.ir.Expr) and definition.op == "call":
            calls.setdefault(ty, []).append(definition)
    errors = []
    for refused, found in calls.items():
        found.sort(key=lambda call: call.loc.line)
        retry = None
        if refused.parameter is not None:
            retry = _literal_retry(refused, found, func_ir)
        errors.append(retry or refused.error(refused.message, loc=found[0].loc))
    return sorted(errors, key=lambda e: getattr(e.loc, "line", 0) or 0)


_registered = False


def register() -> None:
    """Register ``group_by`` typing, its check and its lowering once."""

    global _registered
    if _registered:
        return
    from ._lower import lower_group

    for key in (CoopGroupType, RefusedGroupType):
        nb.typing_registry.register_attr(_attributes(key))
    rules.register(RefusedGroupType, rules.CHECK, refused_groups)
    nb.lowering_registry.lower(GROUP_BY_KEY, nb.types.VarArg(nb.types.Any))(
        lower_group
    )
    _registered = True


__all__ = [
    "GROUP_BY_KEY",
    "GroupByError",
    "GroupByNotImplementedError",
    "GroupByTypeError",
    "GroupByValueError",
    "NonConstantGroupByArgumentError",
    "UnsupportedGroupError",
    "WarpLoadStoreUnsupportedError",
    "partition",
    "refused_groups",
    "register",
]
