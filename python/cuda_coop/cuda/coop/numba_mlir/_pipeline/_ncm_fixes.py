# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR fixes the typed pipeline relies on, found by probing.

A fix is detected by what the installed compiler does, never by its
version: each probe runs Numba-CUDA-MLIR's own type inference, its
argument binding or its dispatcher's compile callback, on a small function
that behaves as expected only with the fix, and checks the result. A
compiler with the fix backported passes, and one that has the fix's names
but not its behaviour fails.

The probes need no GPU and change no registry: the imprecise value the
refinement probes refine comes from a function type passed as an argument.
The argument-binding probe initializes no compiler context, and the
launch probe compiles and launches nothing.
Each probe runs once per process, at the typed pipeline's first compile
(``_prepare``) or when ``present`` is first asked about its fix.

Where a fix has no patch to stand in for it, ``_missing_fixes`` refuses a
kernel that needs it with an error naming the fix.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numba_cuda_mlir
from numba_cuda_mlir import cuda, types

from . import _bridges as nb

try:
    from numba_cuda_mlir import descriptor
    from numba_cuda_mlir.descriptor import mlir_target
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numba_cuda.core import typeinfer
    from numba_cuda_mlir.numba_cuda.core.ssa import reconstruct_ssa
    from numba_cuda_mlir.numba_cuda.typing import templates
except ImportError as exc:
    raise nb.TypedPipelineUnavailableError(
        "The cuda.coop typed pipeline probes Numba-CUDA-MLIR's type "
        f"inference; the installed compiler is missing {exc.name or exc}."
    ) from exc

_PULL = "https://github.com/NVIDIA/numba-cuda-mlir/pull/"


@dataclass(frozen=True)
class Fix:
    """A Numba-CUDA-MLIR change the typed pipeline relies on.

    ``pull`` is the upstream pull request, ``None`` until there is one.
    """

    name: str
    ask: str
    pull: int | None
    probe: Callable[[], bool]

    @property
    def reference(self) -> str:
        if self.pull is None:
            return "not yet proposed upstream"
        return f"NVIDIA/numba-cuda-mlir#{self.pull}, {_PULL}{self.pull}"


def present(fix: Fix) -> bool:
    """Whether the installed compiler has ``fix``; probed once."""

    return _probe(fix.probe)


@cache
def _probe(probe: Callable[[], bool]) -> bool:
    return probe()


def missing(fix: Fix, what: str) -> str:
    """The message for a kernel that needs ``fix`` on a compiler without it."""

    return (
        f"{what} needs Numba-CUDA-MLIR's {fix.name} fix ({fix.reference}); "
        f"the installed Numba-CUDA-MLIR {numba_cuda_mlir.__version__} does "
        "not have it."
    )


# The probe harness ---------------------------------------------------------


def _undefined_array() -> nb.types.Function:
    """The type of a function returning a 1-D array of undefined dtype.

    Type inference completes the dtype from a later indexed write, as it
    does for a ``ThreadData`` payload.
    """

    class UndefinedArray(nb.AbstractTemplate):
        key = "cuda.coop.probe.undefined_array"

        def generic(self, args, kws):
            if not args and not kws:
                return nb.signature(
                    nb.types.Array(nb.undefined, 1, "C"),
                )
            return None

    return nb.types.Function(UndefinedArray)


def infer(function: Callable, *arg_types: Any) -> tuple[Any, list]:
    """Run type inference on ``function``'s SSA form.

    Returns the inferer and the errors propagation reported; unification,
    which refuses imprecise types, is not run.
    """

    mlir_target.ensure_initialized()
    func_ir = reconstruct_ssa(run_frontend(function))
    inferer = typeinfer.TypeInferer(
        mlir_target.typing_context,
        func_ir,
        nb.errors.WarningsFixer(nb.errors.NumbaWarning),
    )
    for index, (name, ty) in enumerate(zip(func_ir.arg_names, arg_types)):
        inferer.seed_argument(name, index, ty)
    inferer.build_constraint()
    return inferer, list(inferer.propagate(raise_errors=False) or ())


def all_precise(inferer) -> bool:
    return all(
        tv.type.is_precise()
        for tv in inferer.typevars.values()
        if tv.type is not None
    )


# Type-inference hooks (U-R1 to U-R3) ---------------------------------------


def _written(make):
    x = make()
    x[0] = 1.0


def _has_typeinfer_hooks() -> bool:
    """A registered extension defers a call, which is released, and runs at
    each fixed point and the final one."""

    extension = getattr(nb.extending, "TypeInferExtension", None)
    register = getattr(nb.extending, "register_typeinfer_extension", None)
    unregister = getattr(nb.extending, "unregister_typeinfer_extension", None)
    if extension is None or register is None or unregister is None:
        return False
    calls = []

    class Probe(extension):
        def defer_call(self, typeinfer_, constraint, fnty, args, kws, released):
            calls.append(("defer", released))
            return True

        def at_fixpoint(self, typeinfer_, final):
            calls.append(("fixpoint", final))

    probe = Probe()
    register(probe)
    try:
        inferer, errors = infer(_written, _undefined_array())
    finally:
        unregister(probe)
    return (
        not errors
        and all_precise(inferer)
        and {("defer", False), ("defer", True), ("fixpoint", False)}
        <= set(calls)
        and calls[-1] == ("fixpoint", True)
    )


TYPEINFER_HOOKS = Fix(
    name="type-inference hooks",
    ask="U-R1 to U-R3",
    pull=392,
    probe=_has_typeinfer_hooks,
)


# Tuple refinement (U-H1-4) ---------------------------------------------------


def _unpacked(make):
    pair = (make(), 1)
    x, _n = pair
    x[0] = 1.0


def _has_tuple_refinement() -> bool:
    """A write through a value unpacked from a tuple refines the tuple and
    the value it was built from."""

    inferer, errors = infer(_unpacked, _undefined_array())
    return not errors and all_precise(inferer)


TUPLE_REFINEMENT = Fix(
    name="tuple refinement",
    ask="U-H1-4",
    pull=396,
    probe=_has_tuple_refinement,
)


# Phi refinement (U-H1-3) ----------------------------------------------------


def _joined(make, flag):
    if flag:
        x = make()
    else:
        x = make()
    x[0] = 1.0


def _has_phi_refinement() -> bool:
    """A write through a phi of two imprecise values refines both."""

    inferer, errors = infer(_joined, _undefined_array(), nb.types.boolean)
    return not errors and all_precise(inferer)


PHI_REFINEMENT = Fix(
    name="phi refinement",
    ask="U-H1-3",
    pull=395,
    probe=_has_phi_refinement,
)


# Argument-sized cuda.local.array (U-P-1) ------------------------------------


def _sized(n):
    cuda.local.array(n, types.float32)


def _has_literal_local_array_shape() -> bool:
    """A ``cuda.local.array`` sized by an integer argument requests the
    argument as a literal."""

    _, errors = infer(_sized, nb.types.int64)
    return any(
        isinstance(e, typeinfer.ForceLiteralArg) and e.requested_args == {0}
        for e in errors
    )


LOCAL_ARRAY_SHAPE = Fix(
    name="argument-sized cuda.local.array",
    ask="U-P-1",
    pull=394,
    probe=_has_literal_local_array_shape,
)


# Argument binding (N1) ------------------------------------------------------


def _keyword_only(a, *, b=0):
    pass


def _binds_like_python() -> bool:
    """Keyword-only parameters bind by keyword, and a call that does not
    bind raises a ``TypeError`` naming the callee, as in Python.

    ``fold_arguments`` is the binding the inliner runs on every inlined
    call, given the callee's name.
    """

    pysig = inspect.signature(_keyword_only)

    def fold(args, kws, **callee):
        return templates.fold_arguments(
            pysig,
            args,
            kws,
            lambda index, param, value: value,
            lambda index, param, default: default,
            lambda index, param, values: values,
            **callee,
        )

    try:
        if tuple(fold((1,), {"b": 2})) != (1, 2):
            return False
        fold((1, 2), {}, name="f")
    except nb.errors.TypingError as exc:
        return isinstance(exc, TypeError) and (
            "f() too many positional arguments" in str(exc)
        )
    except (TypeError, IndexError):
        return False
    return False


ARGUMENT_BINDING = Fix(
    name="argument binding",
    ask="N1",
    pull=None,
    probe=_binds_like_python,
)


# Dynamic shared memory on the compiling launch (N2) -------------------------


def _unlaunched():
    pass


def _relaunches_for_dynamic_shared_memory() -> bool:
    """A launch whose on-demand compile is the first to require dynamic
    shared memory is aborted, to be relaunched with the bytes.

    The native launcher calls the dispatcher's ``_compile`` and launches
    what it returns with the launch's own ``sharedmem``. The probe stands
    in for the compile, which records the requirement on the dispatcher,
    on a throwaway dispatcher it never launches.
    """

    relaunch = getattr(descriptor, "_RelaunchForDynamicSharedMemory", None)
    dispatcher = cuda.jit(_unlaunched)
    if relaunch is None or (
        getattr(dispatcher, "_requires_dynamic_shared_memory", None)
        is not False
    ):
        return False

    def compile_requiring(args):
        dispatcher._requires_dynamic_shared_memory = True
        return b"", "_unlaunched", False

    dispatcher._compile_impl = compile_requiring
    try:
        dispatcher._compile(())
    except relaunch:
        return True
    except (AttributeError, TypeError, ValueError):
        return False
    return False


FIRST_LAUNCH_SHARED_MEMORY = Fix(
    name="first-launch dynamic shared memory",
    ask="N2",
    pull=None,
    probe=_relaunches_for_dynamic_shared_memory,
)


__all__ = [
    "ARGUMENT_BINDING",
    "FIRST_LAUNCH_SHARED_MEMORY",
    "LOCAL_ARRAY_SHAPE",
    "PHI_REFINEMENT",
    "TUPLE_REFINEMENT",
    "TYPEINFER_HOOKS",
    "Fix",
    "all_precise",
    "infer",
    "missing",
    "present",
]
