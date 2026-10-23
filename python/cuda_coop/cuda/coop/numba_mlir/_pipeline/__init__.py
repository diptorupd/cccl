# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile cooperative kernels through Numba typing instead of the planner.

Setting ``CUDA_COOP_NUMBA_MLIR_PIPELINE=typed`` before the backend activates
selects this pipeline in place of ``CoopWholeFunctionPlanner``. No planner is
registered, and the pipeline has four stages, all on Numba-CUDA-MLIR's own
extension points:

1. typing templates on cooperative Numba types (``_types``, ``_templates``),
   which give each cooperative call a request that knows nothing of the
   launch;
2. one after-inference call resolution per kernel (``_resolve``), which
   turns every request into a concrete provider call and whose decisions
   are plain data (``_plan``);
3. lowering of each resolved call to a provider call (``_lower``);
4. link items backed by the resolution's one compile (``_link``).

Nothing cooperative runs before typing. Every Numba-CUDA-MLIR interface that
is not public is reached through ``_bridges``. The compiler fixes the
pipeline relies on are detected by probing (``_ncm_fixes``); where one is
missing, a pinned patch in ``_ncm_patches`` stands in for it, applied only
by this pipeline, or a kernel that needs it is refused with an error naming
the fix (``_missing_fixes``). Probing initializes the compiler's typing
context, so activation only registers, and the probes and patches run at
the first compile, before it types anything (``_prepare``). The supported
surface is
``this_block()``, ``ThreadData``, and ``load`` and ``store`` on a block
group; any other cooperative call is a typing error, never a fallback.
"""

from __future__ import annotations

_activated = False


def _activate() -> None:
    """Register the pipeline's typing, rewrite and lowering hooks once.

    The backend's activation calls this under its activation lock, in place
    of registering the planner.
    """

    global _activated
    if _activated:
        return
    from . import (
        _lower,
        _payload_rules,
        _prepare,
        _resolve,
        _templates,
        _types,
    )

    _payload_rules.register()
    _types.register()
    _templates.register()
    _lower.register()
    _resolve.register()
    _prepare.register()
    _activated = True
