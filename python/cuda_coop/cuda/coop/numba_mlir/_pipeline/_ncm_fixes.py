# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR fixes the typed pipeline relies on, found by probing.

A fix is detected by what the installed compiler does, never by its
version: each probe runs Numba-CUDA-MLIR's own type inference on a small
function that types only with the fix, and checks the result. A compiler
with the fix backported passes, and one that has the fix's names but not
its behaviour fails.

The probes need no GPU and change no registry: the imprecise value the
refinement probes refine comes from a function type passed as an argument.
Each probe runs once per process, when the typed pipeline activates.

Where a fix has no patch to stand in for it, ``_missing_fixes`` refuses a
kernel that needs it with an error naming the fix.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import numba_cuda_mlir
from numba_cuda_mlir import cuda, types

from . import _bridges as nb

try:
    from numba_cuda_mlir.descriptor import mlir_target
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numba_cuda.core import typeinfer
    from numba_cuda_mlir.numba_cuda.core.ssa import reconstruct_ssa
except ImportError as exc:
    raise nb.TypedPipelineUnavailableError(
        "The cuda.coop typed pipeline probes Numba-CUDA-MLIR's type "
        f"inference; the installed compiler is missing {exc.name or exc}."
    ) from exc

_PULL = "https://github.com/NVIDIA/numba-cuda-mlir/pull/"


@dataclass(frozen=True)
class Fix:
    """A Numba-CUDA-MLIR change the typed pipeline relies on.

    ``pull`` is the upstream pull request, or ``None`` while there is none.
    """

    name: str
    ask: str
    pull: int | None
    probe: Callable[[], bool]

    @property
    def reference(self) -> str:
        if self.pull is None:
            return f"{self.ask}, upstream fix in progress"
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
    pull=None,
    probe=_has_tuple_refinement,
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


__all__ = [
    "LOCAL_ARRAY_SHAPE",
    "TUPLE_REFINEMENT",
    "TYPEINFER_HOOKS",
    "Fix",
    "all_precise",
    "infer",
    "missing",
    "present",
]
