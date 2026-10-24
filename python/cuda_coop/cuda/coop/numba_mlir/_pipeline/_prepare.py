# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Prepare Numba-CUDA-MLIR for the typed pipeline at its first compile.

Probing the compiler's fixes (``_ncm_fixes``) runs Numba-CUDA-MLIR's type
inference, which initializes its typing and target contexts. So activation
only registers, and ``prepare`` does the rest once per process, before the
first function is typed: it probes, applies the patches that stand in for
missing fixes (``_ncm_patches``), registers the rules that refuse a kernel
needing a fix with no patch (``_missing_fixes``), and refreshes the
compiler's registries. A failure, such as a patch that does not match the
installed compiler, is raised by that compile, and the next compile tries
again.

``CompilerPreparation`` calls ``prepare`` from Numba-CUDA-MLIR's
before-inference rewrites and rewrites nothing.
"""

from __future__ import annotations

from threading import RLock

from . import _bridges as nb
from . import _ncm_fixes, _ncm_patches, _typing_rules

_lock = RLock()
_prepared = False
_preparing = False


def prepare() -> None:
    """Make the compiler ready for typed kernels, once.

    A probe compiles its own small function, which calls ``prepare`` again
    on the same thread while it runs; that call returns at once.
    ``_missing_fixes`` is imported here, not at activation: it imports the
    compiler's CUDA typing, which registers ``typeof`` implementations.
    """

    global _prepared, _preparing
    if _prepared:
        return
    with _lock:
        if _prepared or _preparing:
            return
        _preparing = True
        try:
            from . import _missing_fixes

            hooks = _ncm_fixes.TYPEINFER_HOOKS
            _ncm_patches.apply(
                *_ncm_patches.stand_in(hooks, _typing_rules.ENGINE)
            )
            if _ncm_fixes.present(hooks):
                _typing_rules.install()
            _missing_fixes.register()
            nb.refresh_registries()
            _prepared = True
        finally:
            _preparing = False


def prepared() -> bool:
    """Whether ``prepare`` has completed in this process."""

    return _prepared


class CompilerPreparation(nb.Rewrite):
    """Run ``prepare`` before the first function is typed."""

    def __init__(self, state=None):
        super().__init__(state)
        prepare()

    def match(self, func_ir, block, typemap, calltypes):
        return False

    def apply(self):
        raise AssertionError("CompilerPreparation.match never selects a block")


def register() -> None:
    """Register ``CompilerPreparation`` with the before-inference rewrites."""

    nb.register_before_inference_rewrite(CompilerPreparation)


__all__ = ["CompilerPreparation", "prepare", "prepared", "register"]
