# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan each kernel's cooperative calls once, after Numba typing.

The pass runs as an after-inference rewrite, after typing and before
lowering. It sees the whole typed function at once, which is what the
planner had to reconstruct from untyped IR. For each kernel it:

* refuses cooperative calls inside functions compiled as separate device
  functions (``targetoptions["device"]``), which have no launch;
* checks that write-only payloads agree on their dtype, and that every read
  of a payload typed to its final dtype;
* collects the cooperative call sites in program order;
* reads the launch's block, only when a site needs it;
* completes each site's request with coop's ``plan_group_primitive``,
  memoized per request and block, and specializes the providers;
* compiles all of them in one NVRTC bundle (``_link``), with one compile
  context and at the kernel's target;
* sets ``launch_bounds`` to the block's thread count;
* plans one implicit scratch region for the providers that need scratch,
  backs it with a static ``cuda.shared.array`` at the kernel's entry, and
  places barriers by the barrier policy;
* stores what lowering needs in ``state.metadata["coop_plan"]`` and reports
  the ``KernelPlan`` to ``_debug.record``.

Barrier policies:

``after-each`` (default)
    A barrier after every call that uses scratch.
``reuse``
    A barrier after a scratch call only when another scratch call follows it
    in program order, or when it is inside a loop.

Select one with ``use_barrier_policy`` or with the environment variable
``CUDA_COOP_NUMBA_MLIR_PIPELINE_BARRIERS``, read at each compile.

The rewrite registry is process-wide, so the pass runs for every function
Numba-CUDA-MLIR compiles once the typed pipeline is active.
"""

from __future__ import annotations

import contextlib
import operator
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from . import _bridges as nb
from . import _debug, _link
from ._errors import TypedPipelineError
from ._lower import SCRATCH_KEYWORD
from ._plan import Backing, KernelPlan, Region, SitePlan
from ._templates import (
    OPERATIONS,
    LoadStoreRequest,
    is_wrapper_location,
    request_for,
)
from ._types import WRITE, ThreadDataType

BARRIER_POLICY_VARIABLE = "CUDA_COOP_NUMBA_MLIR_PIPELINE_BARRIERS"
BARRIER_POLICIES = ("after-each", "reuse")
STATIC_SHARED_LIMIT = 48 * 1024
IMPLICIT_REGION = "implicit.0"
_BACKING_ALIGNMENT = 16

_policy_override: str | None = None


def barrier_policy() -> str:
    """Return the barrier policy for the next compile."""

    policy = _policy_override or os.environ.get(
        BARRIER_POLICY_VARIABLE, "after-each"
    )
    policy = policy.strip().lower()
    if policy not in BARRIER_POLICIES:
        raise ValueError(
            f"{BARRIER_POLICY_VARIABLE}={policy!r} is not a barrier policy; "
            f"use one of {', '.join(BARRIER_POLICIES)}"
        )
    return policy


@contextlib.contextmanager
def use_barrier_policy(policy: str) -> Iterator[None]:
    """Compile kernels inside the block with barrier ``policy``."""

    global _policy_override
    if policy not in BARRIER_POLICIES:
        raise ValueError(f"unknown barrier policy {policy!r}")
    previous = _policy_override
    _policy_override = policy
    try:
        yield
    finally:
        _policy_override = previous


# Plan data for lowering ------------------------------------------------------


@dataclass(frozen=True)
class CallEntry:
    """What lowering emits for one call site."""

    symbol: str
    abi_arguments: tuple
    pointer_arguments: tuple[int, ...]
    scratch: bool
    barrier_after: bool
    link_items: tuple


@dataclass
class CoopPlan:
    """``state.metadata["coop_plan"]``: the plan and its lowering data.

    ``calls`` is keyed by the name of each call's target variable.
    """

    kernel: KernelPlan
    calls: dict[str, CallEntry] = field(default_factory=dict)


@dataclass
class _Site:
    """A call site; ``line`` is the kernel source line of the public call."""

    label: int
    stmt: Any
    kind: str
    line: int | None
    request: LoadStoreRequest | None = None
    algorithm: Any = None


# Request completion (C-T7-6) -------------------------------------------------

_COMPLETIONS: dict[tuple, tuple[Any, dict]] = {}


def _complete(request: LoadStoreRequest, block: tuple) -> tuple[Any, dict]:
    """Return the provider factory and its arguments for ``request``.

    ``plan_group_primitive`` hashes the request's semantic key, which is
    costly, so its result is memoized per request and block.
    """

    key = (request.key, block)
    completed = _COMPLETIONS.get(key)
    if completed is not None:
        return completed

    from ..._core import ArgumentBinding
    from ..._core.group._dispatch import plan_group_primitive
    from ..._core.group._model import GroupLoweringTarget, StorageOwnership
    from ..._core.launch import LaunchFacts
    from .._lowering import _load_store
    from .._lowering._core import NumbaMlirCoreAdapter

    plan = plan_group_primitive(
        request.call, LaunchFacts(exact_block_dim=block)
    )
    try:
        plan = plan.require_supported()
    except NotImplementedError as exc:
        raise TypedPipelineError(f"coop.{request.kind}: {exc}") from None
    if plan.target is not GroupLoweringTarget.CUB_BLOCK:
        raise TypedPipelineError(
            f"coop.{request.kind}: planned for {plan.target.value}, which the "
            "typed pipeline does not lower"
        )
    implementation = plan.implementation
    storage_free = plan.temp_storage.ownership is StorageOwnership.NONE
    factory_name = (
        request.kind if storage_free else f"_{request.kind}_with_storage"
    )
    completed = (
        getattr(_load_store, factory_name),
        {
            "dtype": NumbaMlirCoreAdapter().normalize_dtype(
                implementation.template_arguments["T"]
            ),
            "threads_per_block": block,
            "items_per_thread": implementation.template_arguments[
                "ITEMS_PER_THREAD"
            ],
            "algorithm": implementation.metadata["algorithm"],
            "num_valid_items": ArgumentBinding.omitted(),
            "offset": ArgumentBinding.omitted(),
        },
    )
    _COMPLETIONS[key] = completed
    return completed


@contextlib.contextmanager
def _one_compile_context(context: Any) -> Iterator[None]:
    """Make every provider built inside the block use ``context``.

    Coop's load/store factories resolve their own compile context and take
    none from the caller (C-T9-2). Compiles run under Numba's compiler lock,
    so replacing the resolver for the duration of a compile is not visible
    to another compile.
    """

    from .._compiler import _nvrtc

    resolve = _nvrtc.resolve_compile_context
    _nvrtc.resolve_compile_context = lambda: context
    try:
        yield
    finally:
        _nvrtc.resolve_compile_context = resolve


def _ignored_parameter(parameter: Any) -> bool:
    from .. import _types as coop_types

    if isinstance(
        parameter,
        (
            coop_types.CxxFunction,
            coop_types.StatelessOperator,
            coop_types.StatefulOperator,
        ),
    ):
        return True
    return (
        isinstance(parameter, coop_types.PointerOffset)
        and parameter.static_value is not None
    )


def _select_method(algorithm: Any, operands: int) -> Any:
    for method in algorithm.parameters:
        inputs = [
            p for p in method if not _ignored_parameter(p) and not p.is_output
        ]
        if len(inputs) == operands:
            return method
    raise AssertionError(f"{algorithm.c_name} has no {operands}-operand method")


def _abi_arguments(method: Any) -> tuple:
    from .. import _types as coop_types

    pointer_kinds = (
        coop_types.Pointer,
        coop_types.Array,
        coop_types.PointerReference,
    )
    arguments = []
    for parameter in method:
        if _ignored_parameter(parameter) or parameter.is_output:
            continue
        if isinstance(parameter, pointer_kinds):
            arguments.append(nb.types.CPointer(nb.types.none))
        else:
            arguments.append(parameter.dtype())
    return tuple(arguments)


# Checks -----------------------------------------------------------------------


def _check_write_agreement(func_ir: Any, typemap: dict) -> None:
    """Payloads typed only by indexed writes must agree on the dtype."""

    written: dict[str, set] = {}
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, (nb.ir.SetItem, nb.ir.StaticSetItem)):
                ty = typemap.get(stmt.target.name)
                if isinstance(ty, ThreadDataType) and ty.strength == WRITE:
                    written.setdefault(stmt.target.name, set()).add(
                        nb.unliteral(typemap[stmt.value.name])
                    )
    for name, dtypes in written.items():
        if dtypes != {typemap[name].dtype}:
            found = ", ".join(sorted(map(str, dtypes)))
            raise TypedPipelineError(
                f"coop.ThreadData {name!r}: its writes disagree on the dtype "
                f"({found}). Pass dtype= or make the writes agree."
            )


def _check_reads(func_ir: Any, typemap: dict) -> None:
    """Every read of a payload must type to the payload's final dtype.

    A read typed while weaker evidence held can keep a wider type after a
    stronger refinement; that is an error rather than a silent widening.
    """

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
                raise TypedPipelineError(
                    f"coop.ThreadData {stmt.value.value.name!r}: a read at "
                    f"line {stmt.loc.line} typed as {result}, but the "
                    f"payload's dtype is {payload.dtype}. Pass "
                    f"dtype={payload.dtype} to coop.ThreadData."
                )


def _callee(typemap: dict, expr: Any) -> Any:
    ty = typemap.get(expr.func.name)
    if isinstance(ty, nb.types.Function):
        return ty.typing_key
    return None


def _user_static_shared_bytes(func_ir: Any, typemap: dict, calltypes) -> int:
    total = 0
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if not (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, nb.ir.Expr)
                and stmt.value.op == "call"
                and _callee(typemap, stmt.value) is nb.cuda.shared.array
            ):
                continue
            shape = typemap[stmt.value.args[0].name]
            result = calltypes[stmt.value].return_type
            if isinstance(shape, nb.types.IntegerLiteral):
                total += shape.literal_value * result.dtype.bitwidth // 8
    return total


# The pass ---------------------------------------------------------------------


class CoopStoragePass(nb.Rewrite):
    """The after-inference rewrite that owns every whole-kernel decision.

    All of the work happens in the first ``match``, which rewrites the
    function in place and always returns ``False``.
    """

    def __init__(self, state=None):
        super().__init__(state)
        self.state = state
        self._planned = False

    def match(self, func_ir, block, typemap, calltypes):
        if self._planned or "coop_plan" in self.state.metadata:
            return False
        self._planned = True
        self.func_ir, self.typemap, self.calltypes = func_ir, typemap, calltypes
        self._run()
        return False

    def apply(self):
        raise AssertionError("CoopStoragePass.match never selects a block")

    # -- collect -----------------------------------------------------------

    def _collect(self) -> list[_Site]:
        sites = []
        line = None
        for label in sorted(self.func_ir.blocks):
            for stmt in self.func_ir.blocks[label].body:
                if not is_wrapper_location(stmt.loc):
                    line = stmt.loc.line
                if (
                    isinstance(stmt, nb.ir.Assign)
                    and isinstance(stmt.value, nb.ir.Expr)
                    and stmt.value.op == "call"
                ):
                    kind = OPERATIONS.get(_callee(self.typemap, stmt.value))
                    if kind is not None:
                        sites.append(_Site(label, stmt, kind, line))
        return sites

    # -- run -----------------------------------------------------------------

    def _run(self) -> None:
        state = self.state
        targetoptions = state.metadata.get("targetoptions") or {}
        name = state.func_id.func_name
        sites = self._collect()
        if targetoptions.get("device"):
            if sites:
                raise TypedPipelineError(
                    f"coop.{sites[0].kind} at line {sites[0].line} "
                    f"is inside {state.func_id.func_qualname!r}, a separately "
                    "compiled device function, which has no launch to plan "
                    "for. Let the function inline (the cuda.jit default) "
                    "instead of inline='never'."
                )
            return
        _check_write_agreement(self.func_ir, self.typemap)
        _check_reads(self.func_ir, self.typemap)
        plan = KernelPlan(name=name, barrier_policy=barrier_policy())
        coop_plan = CoopPlan(plan)
        if sites:
            self._plan_sites(sites, plan, coop_plan, targetoptions)
        state.metadata["coop_plan"] = coop_plan
        _debug.record(plan, self.func_ir, self.typemap)

    def _plan_sites(self, sites, plan, coop_plan, targetoptions) -> None:
        from .._compiler import _nvrtc
        from .._compiler._operations import StorageABI
        from .._types import collect_specializations, kernel_compile_target

        state = self.state
        launch = nb.require_launch_config(state)
        block = tuple(launch["block"]) + (1,) * (3 - len(launch["block"]))
        plan.block, plan.block_source = block, "launch"

        target = _nvrtc.CompileTarget.from_targetoptions(targetoptions)
        if target is None:
            target = _nvrtc.CompileTarget.from_targetoptions(
                {
                    **targetoptions,
                    "chip": nb.resolve_gpu_target(targetoptions)["chip"],
                }
            )
        context = _nvrtc.resolve_compile_context()
        identity = _nvrtc.compiler_identity(
            context=context, rdc=True, code="lto", **target.nvrtc_kwargs()
        )

        for site in sites:
            expr = site.stmt.value
            site.request = request_for(
                site.kind,
                [self.typemap[a.name] for a in expr.args],
                {k: self.typemap[v.name] for k, v in expr.kws},
            )
        unique = list(dict.fromkeys(s.request for s in sites))
        with (
            kernel_compile_target(target),
            _one_compile_context(context),
            collect_specializations() as collected,
        ):
            for request in unique:
                factory, kwargs = _complete(request, block)
                factory(**kwargs)
        if len(collected) != len(unique):
            raise AssertionError("a provider factory did not specialize once")
        algorithm_for = dict(zip(unique, collected))
        bundle = _link.compile_bundle(collected, identity, context)
        plan.bundle = bundle.specializations

        layouts = {}
        for site in sites:
            site.algorithm = algorithm_for[site.request]
            layout = bundle.layouts[site.algorithm._private_symbol_digest]
            if site.algorithm.storage_abi is not StorageABI.LEADING_POINTER:
                layout = None
            layouts[id(site)] = layout

        self._plan_launch_bounds(plan, block, targetoptions)
        region = self._plan_region(sites, layouts, plan)
        barriers = self._plan_barriers(sites, layouts, plan.barrier_policy)

        for index, site in enumerate(sites):
            scratch = layouts[id(site)] is not None
            method = _select_method(site.algorithm, 2 + int(scratch))
            request = site.request
            plan.sites.append(
                SitePlan(
                    index=index,
                    operation=site.kind,
                    line=site.line,
                    request=(
                        ("algorithm", request.algorithm),
                        ("dtype", str(request.dtype)),
                        ("items_per_thread", request.items_per_thread),
                        ("block", "x".join(map(str, block))),
                    ),
                    region=IMPLICIT_REGION if scratch else None,
                    barrier_after=barriers[index],
                )
            )
            coop_plan.calls[site.stmt.target.name] = CallEntry(
                symbol=site.algorithm.mangled_name(method) + "__abi",
                abi_arguments=_abi_arguments(method),
                pointer_arguments=(1, 2),
                scratch=scratch,
                barrier_after=barriers[index],
                link_items=(bundle.link_item,),
            )
            if scratch:
                site.stmt.value.kws = (
                    *site.stmt.value.kws,
                    (SCRATCH_KEYWORD, region),
                )

    # -- decisions -------------------------------------------------------------

    def _plan_launch_bounds(self, plan, block, targetoptions) -> None:
        threads = block[0] * block[1] * block[2]
        bounds = targetoptions.get("launch_bounds")
        if bounds is not None:
            maximum = bounds[0] if isinstance(bounds, tuple) else bounds
            if threads > maximum:
                raise TypedPipelineError(
                    f"the launch's block {block} has {threads} threads, "
                    f"more than launch_bounds={bounds!r}"
                )
            plan.launch_bounds = maximum
        elif targetoptions.get("max_registers") is None:
            nb.rebind_targetoptions(self.state, launch_bounds=threads)
            plan.launch_bounds = threads

    def _plan_region(self, sites, layouts, plan) -> Any:
        used = [layouts[id(s)] for s in sites if layouts[id(s)] is not None]
        if not used:
            return None
        size = max(s for s, _ in used)
        alignment = max(a for _, a in used)
        backing_alignment = max(_BACKING_ALIGNMENT, alignment)
        backing_size = -(-size // backing_alignment) * backing_alignment
        user = _user_static_shared_bytes(
            self.func_ir, self.typemap, self.calltypes
        )
        if backing_size + user > STATIC_SHARED_LIMIT:
            raise TypedPipelineError(
                f"cooperative scratch needs {backing_size} bytes of shared "
                f"memory, which with the kernel's own {user} bytes exceeds "
                f"the {STATIC_SHARED_LIMIT}-byte static limit. Scratch above "
                "48 KB is not supported on the typed pipeline; use a smaller "
                "block, fewer items per thread, or a storage-free algorithm "
                "(direct, striped, vectorize)."
            )
        plan.backing = Backing("static", backing_size, backing_alignment)
        plan.regions.append(Region(IMPLICIT_REGION, 0, size, alignment))
        return self._emit_backing(backing_size, backing_alignment, size)

    def _plan_barriers(self, sites, layouts, policy) -> list[bool]:
        in_loop = set()
        if policy == "reuse":
            loops = nb.compute_cfg_from_blocks(self.func_ir.blocks).loops()
            in_loop = {label for loop in loops.values() for label in loop.body}
        users = [i for i, s in enumerate(sites) if layouts[id(s)] is not None]
        barriers = [False] * len(sites)
        for position, index in enumerate(users):
            if policy == "after-each":
                barriers[index] = True
            else:
                later = position + 1 < len(users)
                barriers[index] = later or sites[index].label in in_loop
        return barriers

    # -- rewriting ---------------------------------------------------------

    def _new_var(self, scope, name, ty, loc):
        var = scope.redefine(name, loc)
        self.typemap[var.name] = ty
        return var

    def _emit_backing(self, size: int, alignment: int, region_size: int):
        """Prepend the backing and the region's slice to the entry block."""

        entry = self.func_ir.blocks[min(self.func_ir.blocks)]
        scope, loc = entry.scope, entry.loc
        typingctx = self.state.typingctx
        types = nb.types

        array_fn_ty = typingctx.resolve_getattr(
            types.Module(nb.cuda.shared), "array"
        )
        dtype_ty = typingctx.resolve_value_type(types.uint8)
        size_ty = types.IntegerLiteral(size)
        align_ty = types.IntegerLiteral(alignment)
        array_fn = self._new_var(scope, "$coop_shared_array", array_fn_ty, loc)
        size_var = self._new_var(scope, "$coop_backing_size", size_ty, loc)
        dtype_var = self._new_var(scope, "$coop_backing_dtype", dtype_ty, loc)
        align_var = self._new_var(scope, "$coop_backing_align", align_ty, loc)
        call = nb.ir.Expr.call(
            array_fn, [size_var, dtype_var], (("alignment", align_var),), loc
        )
        array_sig = typingctx.resolve_function_type(
            array_fn_ty, (size_ty, dtype_ty), {"alignment": align_ty}
        )
        self.calltypes[call] = array_sig
        backing = self._new_var(
            scope, "$coop_backing", array_sig.return_type, loc
        )

        slice_fn_ty = typingctx.resolve_value_type(slice)
        slice_sig = typingctx.resolve_function_type(
            slice_fn_ty, (types.intp, types.intp), {}
        )
        slice_fn = self._new_var(scope, "$coop_slice", slice_fn_ty, loc)
        lo = self._new_var(scope, "$coop_region_lo", types.intp, loc)
        hi = self._new_var(scope, "$coop_region_hi", types.intp, loc)
        bounds = self._new_var(
            scope, "$coop_region_bounds", slice_sig.return_type, loc
        )
        slice_call = nb.ir.Expr.call(slice_fn, [lo, hi], (), loc)
        self.calltypes[slice_call] = slice_sig
        getitem = nb.ir.Expr.getitem(backing, bounds, loc)
        getitem_sig = typingctx.resolve_function_type(
            operator.getitem, (array_sig.return_type, slice_sig.return_type), {}
        )
        self.calltypes[getitem] = getitem_sig
        region = self._new_var(
            scope, "$coop_region_implicit", getitem_sig.return_type, loc
        )

        entry.body[:0] = [
            nb.ir.Assign(
                nb.ir.Global("shared_array", nb.cuda.shared.array, loc),
                array_fn,
                loc,
            ),
            nb.ir.Assign(nb.ir.Const(size, loc), size_var, loc),
            nb.ir.Assign(
                nb.ir.Global("uint8", types.uint8, loc), dtype_var, loc
            ),
            nb.ir.Assign(nb.ir.Const(alignment, loc), align_var, loc),
            nb.ir.Assign(call, backing, loc),
            nb.ir.Assign(nb.ir.Global("slice", slice, loc), slice_fn, loc),
            nb.ir.Assign(nb.ir.Const(0, loc), lo, loc),
            nb.ir.Assign(nb.ir.Const(region_size, loc), hi, loc),
            nb.ir.Assign(slice_call, bounds, loc),
            nb.ir.Assign(getitem, region, loc),
        ]
        self.func_ir._definitions = nb.build_definitions(self.func_ir.blocks)
        return region


def register() -> None:
    """Register the pass with the after-inference rewrite registry."""

    nb.register_after_inference_rewrite(CoopStoragePass)


__all__ = [
    "BARRIER_POLICIES",
    "BARRIER_POLICY_VARIABLE",
    "STATIC_SHARED_LIMIT",
    "CallEntry",
    "CoopPlan",
    "CoopStoragePass",
    "barrier_policy",
    "register",
    "use_barrier_policy",
]
