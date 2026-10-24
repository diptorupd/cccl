# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Diagnostics raised by the typed pipeline.

Every error is a Numba ``TypingError``, so it is reported with the kernel's
source location like any other typing failure, whether it is raised by a
typing template or by call resolution after typing. An error that coop's
planner also reports is an instance of the planner's exception class too.

Numba-CUDA-MLIR reports an error raised inside a typing template as a plain
``TypingError`` that quotes it, so only an error raised after typing (by a
check rule or by call resolution) reaches the user with its class. A check
whose class matters therefore runs after typing where the template can type
the call without it.
"""

from __future__ import annotations

from .._compiler import _group_errors as planner
from .._compiler._group_planner_support import GroupRewriteError
from .._compiler._rewrite_support import CoopSinglePhaseRewriteError
from . import _bridges as nb

MODE_HINT = (
    "Unset CUDA_COOP_NUMBA_MLIR_PIPELINE to compile this kernel with the "
    "default pipeline."
)

SUPPORTED_SURFACE = (
    "this_block(), this_warp(), group_by(), ThreadData, TempStorage, and load "
    "and store on a block or warp group"
)


class TypedPipelineError(nb.errors.TypingError):
    """A cooperative construct the typed pipeline rejects."""


def _init(self, msg, loc=None):
    # The planner's class comes first in the bases, and its initializer (or
    # the builtin one) does not take a location.
    TypedPipelineError.__init__(self, msg, loc=loc)


class TypedPipelineTypeError(TypeError, TypedPipelineError):
    """An argument coop's planner refuses with a ``TypeError``."""

    __init__ = _init


class TypedPipelineValueError(ValueError, TypedPipelineError):
    """An argument coop's planner refuses with a ``ValueError``."""

    __init__ = _init


class TypedPipelineNotImplementedError(NotImplementedError, TypedPipelineError):
    """A use coop's planner refuses with a ``NotImplementedError``."""

    __init__ = _init


class RewriteArgumentError(CoopSinglePhaseRewriteError, TypedPipelineError):
    """A call argument coop's planner refuses while rewriting the call."""


class PayloadJoinDtypeError(CoopSinglePhaseRewriteError, TypedPipelineError):
    """Payloads of different dtypes meet in one variable."""


class StoreDtypeMismatchError(
    planner.MemoryDtypeMismatchError, TypedPipelineError
):
    """A payload's writes disagree with the destination it is stored to."""

    __init__ = _init


class PayloadMemoryDtypeError(
    planner.MemoryDtypeMismatchError, TypedPipelineError
):
    """A payload, or a scalar ``store`` value, of another dtype than memory."""

    __init__ = _init


class CommonLoadPayloadError(
    planner.CommonLoadPayloadError, TypedPipelineError
):
    """A common-API ``load`` output that is not a ``ThreadData``."""

    __init__ = _init


class CommonStorePayloadError(
    planner.CommonStorePayloadError, TypedPipelineError
):
    """A common-API ``store`` value that is not a scalar or ``ThreadData``."""

    __init__ = _init


class InvalidGroupSelectorError(
    planner.InvalidGroupSelectorError, TypedPipelineError
):
    """A common-API ``algorithm`` outside the supported choices."""

    __init__ = _init


class InvalidLoadStoreAlgorithmError(
    planner.InvalidLoadStoreAlgorithmError, TypedPipelineError
):
    """A qualified ``algorithm`` outside the choices for the group kind."""

    __init__ = _init


class DefaultDtypeMismatchError(
    planner.DefaultDtypeMismatchError, TypedPipelineError
):
    """A runtime ``oob_default`` of another dtype than the payload."""

    __init__ = _init


class NonConstantGroupArgumentError(
    planner.NonConstantGroupArgumentError, TypedPipelineError
):
    """A control that shapes the provider is not a compile-time constant."""

    __init__ = _init


class NonConstantThreadGroupError(
    planner.NonConstantThreadGroupError, TypedPipelineError
):
    """A group operand that is not a ``this_*()`` group."""

    __init__ = _init


class UnknownLoadStoreExtentError(
    planner.UnknownLoadStoreExtentError, TypedPipelineError
):
    """A payload whose extent is not known at compile time."""

    __init__ = _init


class InconsistentLoopPayloadExtentError(
    planner.InconsistentLoopPayloadExtentError, TypedPipelineError
):
    """Payload aliases with different extents."""

    __init__ = _init


class DeviceFunctionCallError(GroupRewriteError, TypedPipelineError):
    """A cooperative call in a separately compiled device function."""


# Checked in order: a planner class before the builtin class it derives from.
_TYPED_CLASSES = (
    (planner.MemoryDtypeMismatchError, PayloadMemoryDtypeError),
    (planner.DefaultDtypeMismatchError, DefaultDtypeMismatchError),
    (CoopSinglePhaseRewriteError, RewriteArgumentError),
    (NotImplementedError, TypedPipelineNotImplementedError),
    (TypeError, TypedPipelineTypeError),
    (ValueError, TypedPipelineValueError),
)


def planner_text(exc: Exception) -> str:
    """The message of a planner exception, on one line.

    The planner wraps its diagnostics at 80 columns; the typed pipeline
    reports them as one line, as its own.
    """

    return " ".join(str(exc).split())


def typed_error(
    exc: Exception, prefix: str = "", loc=None
) -> TypedPipelineError:
    """Return ``exc``, raised by coop's shared code, as a typed error.

    The result has the class coop's planner raises for ``exc``, and
    ``exc``'s message after ``prefix``.
    """

    msg = prefix + planner_text(exc)
    for planner_class, typed_class in _TYPED_CLASSES:
        if isinstance(exc, planner_class):
            return typed_class(msg, loc=loc)
    return TypedPipelineError(msg, loc=loc)


class SharedMemoryAliasError(CoopSinglePhaseRewriteError, TypedPipelineError):
    """A user shared array that cooperative scratch would overlap."""


class TempStorageBindingError(GroupRewriteError, TypedPipelineError):
    """A ``temp_storage=`` operand that does not bind one descriptor."""


class TempStorageError(CoopSinglePhaseRewriteError, TypedPipelineError):
    """A ``TempStorage`` descriptor, or its scratch, that coop refuses."""


def unsupported_primitive(qualified_name: str) -> TypedPipelineError:
    """The error for a cooperative primitive the typed pipeline lacks."""

    return TypedPipelineError(
        f"{qualified_name} is not supported on the typed pipeline "
        "(CUDA_COOP_NUMBA_MLIR_PIPELINE=typed). The typed pipeline "
        f"supports {SUPPORTED_SURFACE}. {MODE_HINT}"
    )


__all__ = [
    "MODE_HINT",
    "SUPPORTED_SURFACE",
    "CommonLoadPayloadError",
    "CommonStorePayloadError",
    "DefaultDtypeMismatchError",
    "DeviceFunctionCallError",
    "InconsistentLoopPayloadExtentError",
    "InvalidGroupSelectorError",
    "InvalidLoadStoreAlgorithmError",
    "NonConstantGroupArgumentError",
    "NonConstantThreadGroupError",
    "PayloadJoinDtypeError",
    "PayloadMemoryDtypeError",
    "RewriteArgumentError",
    "SharedMemoryAliasError",
    "StoreDtypeMismatchError",
    "TempStorageBindingError",
    "TempStorageError",
    "TypedPipelineError",
    "TypedPipelineNotImplementedError",
    "TypedPipelineTypeError",
    "TypedPipelineValueError",
    "UnknownLoadStoreExtentError",
    "planner_text",
    "typed_error",
    "unsupported_primitive",
]
