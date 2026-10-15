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
* reads the extent of a ``cuda.coop.numba_mlir`` call's ``cuda.local.array``
  payload from its constructor's shape, as coop's planner does;
* binds each site's ``valid_items``, ``oob_default`` and ``offset`` as
  static or runtime by where the value comes from, as coop's planner does,
  and checks them and any scalar ``store`` value by coop's rules;
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
from numbers import Integral
from typing import Any

from . import _bridges as nb
from . import _debug, _link
from ._errors import TypedPipelineError
from ._lower import SCRATCH_KEYWORD
from ._plan import Backing, KernelPlan, Region, SitePlan
from ._templates import (
    CONTROLS,
    OPERATIONS,
    QUALIFIED_OPERATIONS,
    LoadStoreRequest,
    is_scalar_value,
    is_wrapper_location,
    request_for,
    with_controls,
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
    """What lowering emits for one call site.

    ``operands`` follows ``abi_arguments``, one entry per provider operand:

    ``("scratch",)``
        the scratch region's pointer;
    ``("pointer", index)``
        the data pointer of the call's argument ``index``;
    ``("box", index, dtype, value)``
        a one-item array of ``dtype`` holding the scalar argument ``index``,
        or the static ``value`` when it is not ``None``;
    ``("value", keyword)``
        the call's keyword argument, converted to the ABI type.
    """

    symbol: str
    abi_arguments: tuple
    operands: tuple
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
    scalar: Any = None
    qualified: bool = False


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

    try:
        plan = plan_group_primitive(
            request.call, LaunchFacts(exact_block_dim=block)
        ).require_supported()
    except (NotImplementedError, TypeError, ValueError) as exc:
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
    omitted = ArgumentBinding.omitted()
    kwargs = {
        "dtype": NumbaMlirCoreAdapter().normalize_dtype(
            implementation.template_arguments["T"]
        ),
        "threads_per_block": block,
        "items_per_thread": implementation.template_arguments[
            "ITEMS_PER_THREAD"
        ],
        "algorithm": implementation.metadata["algorithm"],
        "num_valid_items": request.binding("valid_items") or omitted,
        "offset": request.binding("offset") or omitted,
    }
    if request.binding("oob_default") is not None:
        kwargs["oob_default"] = request.binding("oob_default")
    completed = (getattr(_load_store, factory_name), kwargs)
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


def _inputs(method: Any) -> list:
    return [p for p in method if not _ignored_parameter(p) and not p.is_output]


def _select_method(algorithm: Any, operands: int) -> Any:
    for method in algorithm.parameters:
        if len(_inputs(method)) == operands:
            return method
    raise AssertionError(f"{algorithm.c_name} has no {operands}-operand method")


def _operands(method: Any, site: _Site, scratch: bool) -> tuple:
    """Pair each provider operand with where lowering finds its value.

    Coop's providers take the scratch pointer, the memory pointer, the
    payload, then the runtime controls in ``CONTROLS`` order.
    """

    from .. import _types as coop_types

    pointer_kinds = (
        coop_types.Pointer,
        coop_types.Array,
        coop_types.PointerReference,
    )
    payload = (
        ("pointer", 2)
        if site.scalar is None
        else ("box", 2, site.request.dtype, site.scalar.value)
    )
    pointers = [("scratch",)] * scratch + [("pointer", 1), payload]
    runtime = [
        ("value", name)
        for name, binding in site.request.controls
        if binding.kind.value == "runtime"
    ]
    abi_arguments, operands = [], []
    for parameter in _inputs(method):
        if isinstance(parameter, pointer_kinds):
            abi_arguments.append(nb.types.CPointer(nb.types.none))
            operands.append(pointers.pop(0))
        else:
            abi_arguments.append(parameter.dtype())
            operands.append(runtime.pop(0))
    if pointers or runtime:
        raise AssertionError(f"{site.algorithm.c_name}: operands left over")
    return tuple(abi_arguments), tuple(operands)


# Controls ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Scalar:
    """A scalar ``store`` value; ``value`` is ``None`` when it is runtime."""

    value: Any


_RUNTIME = object()


def _describe(binding: Any) -> str:
    if binding.kind.value == "runtime":
        return "runtime"
    value = binding.value
    return str(value.item() if hasattr(value, "item") else value)


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

    # -- controls ----------------------------------------------------------

    def _provenance(self, var: Any) -> Any:
        """Return ``var``'s static provenance, or ``_RUNTIME``.

        A value is static when coop's planner would call it static: it
        comes, through aliases, casts and phis, from a constant, a global, a
        free variable or a literal argument.
        """

        from .._compiler._scalar_provenance import (
            try_resolve_static_scalar_provenance,
        )

        definitions = self.func_ir._definitions
        arguments = tuple(self.state.args)
        resolved, scalar = try_resolve_static_scalar_provenance(
            var,
            definitions=lambda v: definitions.get(v.name, ()),
            argument_type=lambda index: (
                arguments[index] if 0 <= index < len(arguments) else None
            ),
        )
        return scalar if resolved else _RUNTIME

    def _controls(self, site: _Site) -> tuple:
        """Bind the site's controls as coop's planner binds them.

        Checks follow the planner's ``_rewrite_load_store``: static
        ``valid_items`` and ``offset`` first, then the integer types of
        runtime ones, then ``oob_default`` against the payload dtype.
        """

        from ..._core import ArgumentBinding
        from .._compiler._parameters import (
            _validate_common_numeric_dtype,
            _validate_runtime_integer_dtype,
            coerce_static_scalar,
        )

        kind, dtype = site.kind, site.request.dtype
        keywords = dict(site.stmt.value.kws)
        found = {}
        for name in CONTROLS[kind]:
            var = keywords.get(name)
            if var is None:
                continue
            scalar = self._provenance(var)
            if scalar is _RUNTIME:
                found[name] = (_RUNTIME, self.typemap[var.name])
            elif scalar.value is not None:
                found[name] = (scalar, None)

        def check(condition: bool, message: str) -> None:
            if condition:
                raise TypedPipelineError(message)

        for name in ("valid_items", "offset"):
            scalar, _ = found.get(name, (None, None))
            if scalar is None or scalar is _RUNTIME:
                continue
            value = scalar.value
            check(
                isinstance(value, bool) or not isinstance(value, Integral),
                f"coop {kind} {name} must be an integer",
            )
            if name == "offset":
                check(
                    int(value) < 0,
                    "coop load/store static offset must be nonnegative",
                )
                check(
                    int(value) > (1 << 63) - 1,
                    "coop load/store static offset must fit a signed 64-bit "
                    "integer",
                )
        for name in ("valid_items", "offset"):
            scalar, ty = found.get(name, (None, None))
            if scalar is _RUNTIME:
                try:
                    _validate_runtime_integer_dtype(
                        ty, operation=kind, parameter=name
                    )
                except TypeError as exc:
                    raise TypedPipelineError(str(exc)) from None

        controls = []
        for name in CONTROLS[kind]:
            if name not in found:
                continue
            scalar, ty = found[name]
            if scalar is _RUNTIME:
                if name == "oob_default":
                    try:
                        ty = _validate_common_numeric_dtype(
                            ty, operation="load", parameter="oob_default"
                        )
                    except (TypeError, ValueError) as exc:
                        raise TypedPipelineError(str(exc)) from None
                    check(
                        ty != dtype,
                        "cuda.coop.numba_mlir.load runtime oob_default dtype "
                        f"{ty} does not match payload dtype {dtype}",
                    )
                controls.append((name, ArgumentBinding.runtime()))
                continue
            value = scalar.value
            if name == "oob_default":
                try:
                    value = coerce_static_scalar(
                        value,
                        dtype,
                        operation="load",
                        parameter="oob_default",
                        source_dtype=scalar.dtype,
                    )
                except (TypeError, ValueError) as exc:
                    raise TypedPipelineError(str(exc)) from None
            controls.append((name, ArgumentBinding.static(value)))
        return tuple(controls)

    def _store_scalar(self, site: _Site) -> _Scalar | None:
        """Check a scalar ``store`` value as coop's planner does.

        A static value converts to the destination's dtype under coop's
        scalar rules; a runtime value must already have that dtype exactly.
        """

        from .._compiler._parameters import coerce_static_scalar

        if site.kind != "store":
            return None
        var = site.stmt.value.args[2]
        ty = self.typemap[var.name]
        if not is_scalar_value(ty):
            return None
        dtype = site.request.dtype
        scalar = self._provenance(var)
        if scalar is _RUNTIME or scalar.value is None:
            actual = getattr(ty, "literal_type", ty)
            if actual != dtype:
                raise TypedPipelineError(
                    f"cuda.coop.numba_mlir.store value dtype {actual} does "
                    f"not match destination dtype {dtype}"
                )
            return _Scalar(None)
        try:
            value = coerce_static_scalar(
                scalar.value,
                dtype,
                operation="store",
                parameter="value",
                source_dtype=scalar.dtype,
            )
        except (TypeError, ValueError) as exc:
            raise TypedPipelineError(str(exc)) from None
        return _Scalar(value)

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
                    callee = _callee(self.typemap, stmt.value)
                    kind = OPERATIONS.get(callee)
                    if kind is not None:
                        sites.append(
                            _Site(
                                label,
                                stmt,
                                kind,
                                line,
                                qualified=callee in QUALIFIED_OPERATIONS,
                            )
                        )
        return sites

    def _array_extent(self, site: _Site) -> int | None:
        """Return the extent of a plain-array payload, read from the IR.

        As in coop's planner, the payload must come, through aliases, from
        ``cuda.local.array`` calls with a constant integer shape, and they
        must agree. Return ``None`` for a payload that is not a plain array.
        """

        var = site.stmt.value.args[2]
        ty = self.typemap[var.name]
        if not isinstance(ty, nb.types.Array) or isinstance(ty, ThreadDataType):
            return None
        extents, pending, seen = set(), [var], set()
        while pending:
            var = pending.pop()
            if var.name in seen:
                continue
            seen.add(var.name)
            for definition in self.func_ir._definitions.get(var.name, ()):
                if isinstance(definition, nb.ir.Var):
                    pending.append(definition)
                elif isinstance(definition, nb.ir.Expr) and (
                    definition.op == "cast"
                ):
                    pending.append(definition.value)
                elif isinstance(definition, nb.ir.Expr) and (
                    definition.op == "phi"
                ):
                    pending.extend(definition.incoming_values)
                else:
                    extents.add(self._local_array_extent(definition))
        if None in extents or not extents:
            raise TypedPipelineError(
                f"cuda.coop.numba_mlir.{site.kind} requires a static "
                "items_per_thread extent before provider selection"
            )
        if len(extents) > 1:
            raise TypedPipelineError(
                "cuda.coop.numba_mlir loop-carried payloads have "
                "inconsistent items_per_thread extents"
            )
        return extents.pop()

    def _local_array_extent(self, definition: Any) -> int | None:
        if not (
            isinstance(definition, nb.ir.Expr)
            and definition.op == "call"
            and _callee(self.typemap, definition) is nb.cuda.local.array
        ):
            return None
        shape = (
            definition.args[0]
            if definition.args
            else dict(definition.kws).get("shape")
        )
        ty = None if shape is None else self.typemap.get(shape.name)
        if isinstance(ty, nb.types.IntegerLiteral):
            return ty.literal_value
        return None

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
                qualified=site.qualified,
                extent=self._array_extent(site) if site.qualified else None,
            )
            site.scalar = self._store_scalar(site)
            site.request = with_controls(site.request, self._controls(site))
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
            request = site.request
            runtime = sum(
                b.kind.value == "runtime" for _, b in request.controls
            )
            method = _select_method(site.algorithm, 2 + scratch + runtime)
            abi_arguments, operands = _operands(method, site, scratch)
            value = ()
            if site.scalar is not None:
                value = (
                    (
                        "value",
                        "runtime"
                        if site.scalar.value is None
                        else f"{site.scalar.value.item()!r}",
                    ),
                )
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
                        *value,
                        *(
                            (name, _describe(binding))
                            for name, binding in request.controls
                        ),
                    ),
                    region=IMPLICIT_REGION if scratch else None,
                    barrier_after=barriers[index],
                )
            )
            coop_plan.calls[site.stmt.target.name] = CallEntry(
                symbol=site.algorithm.mangled_name(method) + "__abi",
                abi_arguments=abi_arguments,
                operands=operands,
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
