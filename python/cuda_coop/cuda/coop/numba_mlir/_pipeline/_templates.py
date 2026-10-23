# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type cooperative calls with Numba typing templates.

Each kernel-callable object of both public APIs, ``cuda.coop`` and
``cuda.coop.numba_mlir``, gets one typing template. Objects the typed
pipeline supports are typed for real; every other one gets a template that
rejects the call with ``unsupported_primitive``, so a kernel never falls
back to the planner and never fails with Numba's generic "untyped global"
message.

``load`` and ``store`` are reached through inline wrappers. Numba typing
cannot refine a call argument from a template, and a payload's
dtype often comes from the ``load`` that fills it. Each public ``load`` and
``store`` object therefore gets a wrapper, generated at activation from the
object's own signature and attached to it, which Numba inlines before
typing::

    def _coop_load_wrapper(group, source, output, /, algorithm='direct', ...):
        if _never():
            output._coop_refine_load(source)
        _coop_load_op(group, source, output, algorithm=algorithm, ...)

The guarded method call is a witness: its template returns a refined
receiver, which Numba propagates to the caller's payload with the evidence
strength of a producer (or, for ``store``, of a store destination). The
private operation then types with a precise payload and builds coop's own
request, a dims-free ``GroupPrimitiveCall`` that call resolution completes
with the launch's block.

Each API has its own private operations, because their payload rules
differ: a ``cuda.coop.numba_mlir`` call also accepts a ``cuda.local.array``
payload, whose extent call resolution reads from the IR, and a
``cuda.coop`` call refuses one with coop's common-API diagnostic.

A ``ThreadData``'s ``items_per_thread`` attribute is its extent as an
integer literal, the constant coop's planner substitutes for it.
"""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import _bridges as nb
from ._errors import (
    CommonLoadPayloadError,
    CommonStorePayloadError,
    InvalidGroupSelectorError,
    InvalidLoadStoreAlgorithmError,
    NonConstantGroupArgumentError,
    NonConstantThreadGroupError,
    PayloadMemoryDtypeError,
    RewriteArgumentError,
    TypedPipelineError,
    TypedPipelineNotImplementedError,
    TypedPipelineTypeError,
    TypedPipelineValueError,
    planner_text,
    typed_error,
    unsupported_primitive,
)
from ._types import (
    EXPLICIT,
    PRODUCER,
    STORE_DST,
    CoopGroupType,
    RefusedGroupType,
    ThreadDataType,
)

# Public names that are never called inside a kernel: host-side descriptors
# and registration, typing protocols, and Numba-CUDA-MLIR's own namespaces
# re-exported for convenience.
_NOT_KERNEL_CALLABLE = frozenset(
    {
        "Hierarchy",
        "StatefulFunction",
        "TempStorageLike",
        "ThreadDataLike",
        "ThreadGroup",
        "ThreadHierarchy",
        "local",
        "register",
        "shared",
    }
)

# Optional scalar controls, in the order coop's providers take them at run
# time.
CONTROLS = {
    "load": ("valid_items", "oob_default", "offset"),
    "store": ("valid_items", "offset"),
}

_PAYLOAD = {"load": "output", "store": "value"}
_MEMORY = {"load": "source", "store": "destination"}


def kernel_primitives() -> dict[object, str]:
    """Map each kernel-callable public object to its qualified name."""

    from ... import numba_mlir as qualified
    from ..._core import api as common

    primitives: dict[object, str] = {}
    for prefix, module in (
        ("cuda.coop", common),
        ("cuda.coop.numba_mlir", qualified),
    ):
        for name in module.__all__:
            if name in _NOT_KERNEL_CALLABLE:
                continue
            obj = getattr(module, name)
            if callable(obj):
                primitives.setdefault(obj, f"{prefix}.{name}")
    return primitives


def group_operation(obj: object) -> str | None:
    """Return coop's registered group operation for a public object."""

    from ..._core.api._dispatch import _common_group_operation_name
    from .._compiler._operations import group_operation_name

    return _common_group_operation_name(obj) or group_operation_name(obj)


def register_template(
    key: object,
    generic: Callable,
    name: str,
    *,
    prefer_literal: bool = False,
) -> type:
    """Type calls to ``key`` with ``generic(template, args, kws)``."""

    template = type(
        f"CoopTyped_{name}",
        (nb.AbstractTemplate,),
        {"key": key, "generic": generic, "prefer_literal": prefer_literal},
    )
    nb.typing_registry.register(template)
    nb.typing_registry.register_global(key, nb.types.Function(template))
    return template


def _rejecting(qualified_name: str) -> Callable:
    def generic(self, args, kws):
        raise unsupported_primitive(qualified_name)

    return generic


def _not_inlined(qualified_name: str) -> Callable:
    def generic(self, args, kws):
        raise TypedPipelineError(
            f"{qualified_name} reached typing without its inline wrapper. "
            "Call it directly by name inside the kernel."
        )

    return generic


def _omitted(ty: Any) -> bool:
    return ty is None or isinstance(ty, (nb.types.NoneType, nb.types.Omitted))


def _payload_dtype(ty: Any) -> Any:
    """Return the dtype a ``dtype=`` argument's type names, or ``None``.

    Accepts the spellings coop's planner accepts: a Numba or NumPy scalar
    type, the Python classes ``bool``, ``int``, ``float`` and ``complex``,
    and a type name string.
    """

    from .._compiler._parameters import normalize_dtype_param

    if isinstance(ty, nb.types.NumberClass):
        return ty.instance_type
    if isinstance(ty, nb.types.DType):
        return ty.dtype
    if isinstance(ty, nb.types.Function) and ty.typing_key in _PYTHON_DTYPES:
        return normalize_dtype_param(ty.typing_key)
    if isinstance(ty, nb.types.StringLiteral):
        try:
            return normalize_dtype_param(ty.literal_value)
        except ValueError as exc:
            raise TypedPipelineValueError(str(exc)) from None
    return None


_PYTHON_DTYPES = (bool, int, float, complex)


# Groups --------------------------------------------------------------------

# The group factories a kernel may call. A load or store refuses a thread,
# cluster or grid group at the call, as coop's planner does.
GROUP_FACTORIES = frozenset(
    {"this_block", "this_warp", "this_thread", "this_cluster", "this_grid"}
)
_NON_BLOCK_CONSTRUCTOR_KINDS = frozenset({"thread", "cluster", "grid"})


def _bind(function: Callable, args, kws) -> dict:
    """Bind argument types to a public function's parameters, as Python does.

    A call Python would refuse raises coop's planner's ``TypeError``.
    """

    try:
        bound = inspect.signature(function).bind(*args, **kws)
    except TypeError as exc:
        raise TypedPipelineTypeError(f"{function.__name__}() {exc}") from None
    return bound.arguments


def _this_group(function: Callable) -> Callable:
    def generic(self, args, kws):
        _bind(function, args, kws)
        return nb.signature(CoopGroupType(function()))

    return generic


# ThreadData ----------------------------------------------------------------

_CONSTRUCTOR_PARAMETERS = ("items_per_thread", "dtype", "alignment")


def _fold_constructor_arguments(args, kws) -> tuple:
    bound = dict(zip(_CONSTRUCTOR_PARAMETERS, args))
    bound.update(dict(kws))
    return tuple(bound[n] for n in _CONSTRUCTOR_PARAMETERS if n in bound)


def constructor_alignment(qualified_name: str, ty: Any) -> int | None:
    """Return the normalized ``alignment=`` of a constructor call."""

    from ..._core.api._payload import _normalize_alignment

    if _omitted(ty):
        return None
    if not isinstance(ty, nb.types.IntegerLiteral):
        raise RewriteArgumentError(
            f"{qualified_name}: alignment must be a compile-time integer"
        )
    try:
        return _normalize_alignment(ty.literal_value)
    except (TypeError, ValueError) as exc:
        raise RewriteArgumentError(f"{qualified_name}: {exc}") from None


def _thread_data(function: Callable, qualified_name: str) -> Callable:
    def generic(self, args, kws):
        # Numba-CUDA-MLIR retries a failed literal attempt with unliteral
        # arguments on the same template, and reports a literal request
        # from any attempt in place of every other error.
        literal_error = getattr(self, "_coop_literal_error", None)
        try:
            return typed(args, kws)
        except TypedPipelineError as exc:
            self._coop_literal_error = exc
            raise
        except nb.errors.ForceLiteralArg:
            if literal_error is not None:
                raise literal_error from None
            raise

    def typed(args, kws):
        bound = _bind(function, args, kws)
        items = bound["items_per_thread"]
        if isinstance(items, nb.types.Integer) and not isinstance(
            items, nb.types.IntegerLiteral
        ):
            raise nb.force_literal_arg({0}, _fold_constructor_arguments)
        if not isinstance(items, nb.types.IntegerLiteral):
            raise TypedPipelineTypeError(
                f"{qualified_name}: items_per_thread must be a positive "
                f"compile-time integer, got {items}"
            )
        if items.literal_value <= 0:
            raise TypedPipelineValueError(
                f"{qualified_name}: items_per_thread must be a positive "
                f"integer, got {items.literal_value}"
            )
        alignment = constructor_alignment(
            qualified_name, bound.get("alignment")
        )
        dtype_ty = bound.get("dtype")
        if _omitted(dtype_ty):
            payload = ThreadDataType(
                nb.undefined, items.literal_value, alignment=alignment
            )
        else:
            dtype = _payload_dtype(dtype_ty)
            if dtype is None:
                raise TypedPipelineTypeError(
                    f"{qualified_name}: dtype must be a Numba or NumPy "
                    f"scalar type, got {dtype_ty}"
                )
            payload = ThreadDataType(
                dtype, items.literal_value, EXPLICIT, alignment
            )
        return nb.signature(payload, *args, *kws.values())

    return generic


# The method witness ----------------------------------------------------------


def _never() -> bool:
    """A condition that is false at run time and unknown to typing."""

    raise NotImplementedError("cuda.coop: kernel only")


def _witness(td, args, strength):
    if (
        len(args) == 1
        and isinstance(td, ThreadDataType)
        and isinstance(args[0], nb.types.Array)
        and args[0].dtype.is_precise()
    ):
        return nb.signature(
            nb.types.none, *args, recvr=td.refined(args[0].dtype, strength)
        )
    # Anything else leaves the receiver alone; the operation's template
    # then reports what is wrong with the call.
    return nb.signature(nb.types.none, *args, recvr=td)


def _witness_template(key: type) -> type:
    class Witness(nb.AttributeTemplate):
        @nb.bound_function("coop.thread_data.refine_load")
        def resolve__coop_refine_load(self, td, args, kws):
            return _witness(td, args, PRODUCER)

        @nb.bound_function("coop.thread_data.refine_store")
        def resolve__coop_refine_store(self, td, args, kws):
            return _witness(td, args, STORE_DST)

    Witness.key = key
    Witness.__name__ = f"CoopWitness_{key.__name__}"
    return Witness


class _ThreadDataAttributes(nb.AttributeTemplate):
    key = ThreadDataType

    def resolve_items_per_thread(self, td):
        return nb.types.IntegerLiteral(td.items)


# load and store --------------------------------------------------------------


def _coop_load_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _coop_store_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _coop_qualified_load_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _coop_qualified_store_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


OPERATIONS: dict[Callable, str] = {
    _coop_load_op: "load",
    _coop_store_op: "store",
    _coop_qualified_load_op: "load",
    _coop_qualified_store_op: "store",
}

# The private operations of ``cuda.coop.numba_mlir``'s load and store.
QUALIFIED_OPERATIONS = frozenset(
    {_coop_qualified_load_op, _coop_qualified_store_op}
)

_QUALIFIED_PREFIX = "cuda.coop.numba_mlir."


@dataclass(frozen=True, eq=False)
class LoadStoreRequest:
    """A typed ``load`` or ``store``, before the launch is known.

    ``controls`` holds coop's ``ArgumentBinding`` for each control the call
    supplies, in ``CONTROLS`` order. A static binding carries its value and
    a runtime one does not, so two calls that differ only in a runtime
    value have the same request, and two that differ in a static value do
    not. Requests are interned by ``key``, so identity is equality.

    ``call`` is coop's ``GroupPrimitiveCall``; call resolution completes
    it with ``plan_group_primitive`` once it knows the block.
    """

    kind: str
    dtype: Any
    items_per_thread: int
    algorithm: str
    group: Any
    controls: tuple[tuple[str, Any], ...]
    call: Any

    @property
    def key(self) -> tuple:
        return _request_key(
            self.kind,
            self.group,
            self.dtype,
            self.items_per_thread,
            self.algorithm,
            self.controls,
        )

    def binding(self, name: str) -> Any:
        """Return the binding of control ``name``, or ``None`` if omitted."""

        return dict(self.controls).get(name)


_REQUESTS: dict[tuple, LoadStoreRequest] = {}


def _request_key(kind, group, dtype, items, algorithm, controls) -> tuple:
    return (
        kind,
        group.key,
        dtype,
        items,
        algorithm,
        tuple((name, b.semantic_key) for name, b in controls),
    )


def _intern(kind, group, dtype, items, algorithm, controls=()):
    from ..._core import ArgumentBinding
    from ..._core.group._model import GroupPrimitiveCall
    from ..._core.group.load_store import (
        GroupLoadStoreAlgorithm,
        GroupLoadStoreKind,
        GroupLoadStoreSemantics,
    )
    from .._lowering._core import NumbaMlirCoreAdapter

    key = _request_key(kind, group, dtype, items, algorithm, controls)
    request = _REQUESTS.get(key)
    if request is not None:
        return request
    bindings = dict(controls)
    omitted = ArgumentBinding.omitted()
    try:
        semantics = GroupLoadStoreSemantics(
            kind=GroupLoadStoreKind(kind),
            dtype=NumbaMlirCoreAdapter().core_dtype(dtype),
            items_per_thread=items,
            algorithm=GroupLoadStoreAlgorithm(algorithm),
            valid_items=bindings.get("valid_items", omitted),
            oob_default=bindings.get("oob_default", omitted),
            offset=bindings.get("offset", omitted),
        )
    except (TypeError, ValueError) as exc:
        raise typed_error(exc, f"coop.{kind}: ") from None
    call = GroupPrimitiveCall(group=group.group, operation=semantics)
    request = _REQUESTS[key] = LoadStoreRequest(
        kind, dtype, items, algorithm, group, tuple(controls), call
    )
    return request


def with_controls(
    request: LoadStoreRequest, controls: tuple[tuple[str, Any], ...]
) -> LoadStoreRequest:
    """Return the request for ``request``'s transfer with ``controls``.

    Raises the coop diagnostic for a combination coop rejects, such as
    ``oob_default`` without ``valid_items``.
    """

    return _intern(
        request.kind,
        request.group,
        request.dtype,
        request.items_per_thread,
        request.algorithm,
        controls,
    )


def _api(kind: str, qualified: bool) -> str:
    return f"{_QUALIFIED_PREFIX if qualified else 'cuda.coop.'}{kind}"


def _algorithm(kind: str, ty: Any, qualified: bool) -> str:
    """Return the block algorithm a call's ``algorithm=`` names.

    Raises the error coop's planner raises for the common or the qualified
    API.
    """

    from ..._core.group.load_store import GroupLoadStoreAlgorithm
    from .._compiler import _group_errors as planner
    from .._compiler._load_store_algorithms import (
        _BLOCK_LOAD_STORE_ALGORITHMS,
        _resolve_algorithm,
    )

    if _omitted(ty):
        return "direct"
    if isinstance(ty, nb.types.UnicodeType):
        raise NonConstantGroupArgumentError(
            planner_text(planner.NonConstantGroupArgumentError("algorithm"))
        )
    if not isinstance(ty, nb.types.StringLiteral):
        raise TypedPipelineTypeError(
            f"{_api(kind, qualified)} algorithm must be a string, got {ty}"
        )
    try:
        token = _resolve_algorithm(
            ty.literal_value,
            _BLOCK_LOAD_STORE_ALGORITHMS,
            _api(kind, qualified),
        )
    except ValueError:
        choices = ", ".join(sorted(_BLOCK_LOAD_STORE_ALGORITHMS))
        if qualified:
            exc = planner.InvalidLoadStoreAlgorithmError(kind, choices, "block")
            raise InvalidLoadStoreAlgorithmError(planner_text(exc)) from None
        exc = planner.InvalidGroupSelectorError(kind, "algorithm", choices)
        raise InvalidGroupSelectorError(planner_text(exc)) from None
    return GroupLoadStoreAlgorithm(token).value


def _check_group(kind: str, group: Any, qualified: bool) -> None:
    """Refuse a group operand the API does not take.

    The common API refuses a thread, cluster or grid group as coop's
    common-API check does. Coop's shared planner refuses the qualified API's,
    and any block or warp group it cannot lower, when the call is
    specialized.
    """

    from ..._core.api._dispatch import _validate_common_operation_group
    from .._compiler import _group_errors as planner

    if not isinstance(group, CoopGroupType):
        raise NonConstantThreadGroupError(
            planner_text(planner.NonConstantThreadGroupError(kind))
        )
    if qualified or group.kind not in _NON_BLOCK_CONSTRUCTOR_KINDS:
        return
    try:
        _validate_common_operation_group(kind, group.group)
    except NotImplementedError as exc:
        raise TypedPipelineNotImplementedError(planner_text(exc)) from None


def _check_payload(kind: str, payload: Any, qualified: bool) -> None:
    from .._compiler import _group_errors as planner

    if isinstance(payload, ThreadDataType):
        return
    if isinstance(payload, nb.types.Array) and qualified:
        return
    if not qualified:
        if kind == "load":
            exc = CommonLoadPayloadError
            planner_exc = planner.CommonLoadPayloadError()
        else:
            exc = CommonStorePayloadError
            planner_exc = planner.CommonStorePayloadError()
        raise exc(planner_text(planner_exc))
    raise TypedPipelineTypeError(
        f"{_api(kind, qualified)} {_PAYLOAD[kind]} must be a coop.ThreadData "
        f"or a fixed-size local array, got {payload}"
    )


def _bind_operation(kind: str, args, kws) -> dict:
    bound = dict(zip(("group", _MEMORY[kind], _PAYLOAD[kind]), args))
    bound.update(kws)
    return bound


def is_scalar_value(ty: Any) -> bool:
    """Whether ``ty`` is a scalar ``store`` value rather than a payload."""

    return isinstance(nb.unliteral(ty), (nb.types.Number, nb.types.Boolean))


_PENDING_EXTENT = object()
_REFUSED_GROUP = object()


def _request(
    kind: str, args, kws, *, qualified: bool = False, extent: int | None = None
) -> Any:
    """Validate a typed call and return its interned request.

    The request has no controls: whether a control is static or runtime
    depends on where its value comes from, which site analysis reads
    from the IR. A scalar ``store`` value stores one item per thread of the
    destination's dtype; site analysis checks the value against it.

    Return ``None`` while the payload's dtype is still undefined, so that a
    later typing round retries with the refined payload. A qualified call
    with a plain-array payload has no extent in its types; it returns
    ``_PENDING_EXTENT`` until site analysis supplies ``extent``. A call on
    a refused group returns ``_REFUSED_GROUP``: it types, so that the
    group's own error is the one typing reports.
    """

    from ..._core.dtype_policy import validate_common_numeric_dtype_name

    bound = _bind_operation(kind, args, kws)
    if not _typable(kind, bound):
        return None
    group = bound.get("group")
    if isinstance(group, RefusedGroupType):
        return _REFUSED_GROUP
    _check_group(kind, group, qualified)
    memory = bound.get(_MEMORY[kind])
    payload = bound.get(_PAYLOAD[kind])
    if kind == "store" and is_scalar_value(payload):
        dtype, items = memory.dtype, 1
    else:
        _check_payload(kind, payload, qualified)
        if payload.dtype != memory.dtype:
            raise PayloadMemoryDtypeError(
                f"coop.{kind}: payload dtype {payload.dtype} conflicts with "
                f"{_MEMORY[kind]} dtype {memory.dtype}"
            )
        dtype = payload.dtype
        items = payload.items if isinstance(payload, ThreadDataType) else extent
    algorithm = _algorithm(kind, bound.get("algorithm"), qualified)
    try:
        validate_common_numeric_dtype_name(str(dtype), operation=kind)
    except TypeError as exc:
        raise typed_error(exc) from None
    if items is None:
        return _PENDING_EXTENT
    return _intern(kind, group, dtype, items, algorithm)


def _typable(kind: str, bound: dict) -> bool:
    """Whether typing can type a call; refuses what typing cannot type.

    A call types once its payload's dtype is known. Every other check runs
    in call resolution, after typing, where its error keeps the class coop's
    planner raises.
    """

    if not _omitted(bound.get("temp_storage")):
        raise TypedPipelineError(
            f"coop.{kind}: explicit temp_storage= is not supported on the "
            "typed pipeline; omit it and the pipeline provides the scratch."
        )
    payload = bound.get(_PAYLOAD[kind])
    return not isinstance(payload, nb.types.Array) or payload.dtype.is_precise()


def _check_array(kind: str, bound: dict, loc: Any) -> None:
    """Require an array memory operand.

    Coop's planner types the memory operand's dtype against the payload's,
    so a scalar of another dtype raises its dtype mismatch.
    """

    memory = bound.get(_MEMORY[kind])
    if isinstance(memory, nb.types.Array):
        return
    line = "" if loc is None else f" at line {loc.line}"
    msg = f"coop.{kind}{line}: {_MEMORY[kind]} must be an array, got {memory}"
    payload = bound.get(_PAYLOAD[kind])
    dtype = getattr(payload, "dtype", None)
    if (
        isinstance(dtype, nb.types.Type)
        and dtype.is_precise()
        and dtype != nb.unliteral(memory)
    ):
        raise PayloadMemoryDtypeError(
            f"{msg}, whose dtype also conflicts with payload dtype {dtype}",
            loc=loc,
        )
    raise TypedPipelineError(msg, loc=loc)


def _check_memory(kind: str, memory: Any, loc: Any) -> None:
    """Require the memory operand coop's providers take, as the planner does.

    A provider's memory parameter is a ``Pointer``, which receives only the
    array's base address, so its type is a one-dimensional C-contiguous
    array. Typing accepts any array; a strided view or a multi-dimensional
    array would be read as contiguous memory from its base.
    """

    from .._types import Pointer

    required = Pointer(memory.dtype).dtype()
    if (memory.ndim, memory.layout) == (required.ndim, required.layout):
        return
    line = "" if loc is None else f" at line {loc.line}"
    raise TypedPipelineError(
        f"coop.{kind}{line}: {_MEMORY[kind]} must be a one-dimensional "
        f"C-contiguous array, got {memory} ({memory.ndim}-D, layout "
        f"{memory.layout!r}). The provider receives only the array's base "
        "address and would read a strided or multi-dimensional array as "
        "contiguous memory. Pass a contiguous one-dimensional array or "
        "slice, such as a[start:stop], or copy the data into one.",
        loc=loc,
    )


def request_for(
    kind: str,
    args,
    kws,
    *,
    qualified: bool = False,
    extent: int | None = None,
    loc: Any = None,
) -> LoadStoreRequest:
    """Return the request a typed call built, from its argument types.

    ``extent`` is the extent of a plain-array payload, which its type does
    not carry. ``loc`` is the kernel location of the public call.
    """

    bound = _bind_operation(kind, args, kws)
    _check_array(kind, bound, loc)
    _check_memory(kind, bound[_MEMORY[kind]], loc)
    request = _request(kind, args, kws, qualified=qualified, extent=extent)
    if request is None or request in (_PENDING_EXTENT, _REFUSED_GROUP):
        raise AssertionError(f"coop.{kind} reached call resolution untyped")
    return request


def _operation(kind: str) -> Callable:
    def generic(self, args, kws):
        if not _typable(kind, _bind_operation(kind, args, kws)):
            return None
        return nb.signature(nb.types.none, *args, *kws.values())

    return generic


# Inline wrappers --------------------------------------------------------------

WRAPPER_SOURCE: dict[str, str] = {}
_WRAPPER_FILENAME = "<cuda.coop typed-pipeline wrapper for {}>"


def is_wrapper_location(loc: Any) -> bool:
    """Whether ``loc`` is inside an inlined wrapper, not the kernel."""

    prefix = _WRAPPER_FILENAME.split("{", 1)[0]
    return str(getattr(loc, "filename", "")).startswith(prefix)


def _parameter_text(p: inspect.Parameter) -> str:
    if p.default is p.empty:
        return p.name
    if eval(repr(p.default), {}) != p.default:
        raise TypeError(f"default of {p.name!r} has no literal spelling")
    return f"{p.name}={p.default!r}"


def make_wrapper(
    public: Callable, qualified_name: str, operation: str
) -> Callable:
    """Generate the inline wrapper of a public ``load`` or ``store`` object.

    The wrapper has the public object's parameters, with keyword-only ones
    relaxed to ordinary parameters: Numba's inliner cannot bind keyword-only
    parameters. The private operation still receives them by
    keyword.
    """

    params = list(inspect.signature(public).parameters.values())
    positional_only = [p for p in params if p.kind is p.POSITIONAL_ONLY]
    ordinary = [
        p for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    header = [_parameter_text(p) for p in positional_only]
    if positional_only:
        header.append("/")
    header += [_parameter_text(p) for p in ordinary]
    call = [p.name for p in positional_only]
    call += [
        p.name if p.kind is p.POSITIONAL_OR_KEYWORD else f"{p.name}={p.name}"
        for p in ordinary
    ]
    if qualified_name.startswith(_QUALIFIED_PREFIX):
        load, store = _coop_qualified_load_op, _coop_qualified_store_op
    else:
        load, store = _coop_load_op, _coop_store_op
    op = load if operation == "load" else store
    name = f"_coop_{operation}_wrapper"
    source = (
        f"def {name}({', '.join(header)}):\n"
        "    if _never():\n"
        f"        {_PAYLOAD[operation]}._coop_refine_{operation}"
        f"({_MEMORY[operation]})\n"
        f"    {op.__name__}({', '.join(call)})\n"
    )
    filename = _WRAPPER_FILENAME.format(qualified_name)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {"_never": _never, op.__name__: op}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    WRAPPER_SOURCE[qualified_name] = source
    return namespace[name]


# Registration ----------------------------------------------------------------


def _never_generic(self, args, kws):
    if not args and not kws:
        return nb.signature(nb.types.boolean)
    return None


def _lower_never(builder, target, args, kwargs):
    builder.store_var(
        target, nb.arith.constant(nb.mlir_ir.IntegerType.get_signless(1), 0)
    )


def _lower_witness(builder, target, args, kwargs):
    pass


_registered = False


def register() -> None:
    """Register a template for every kernel-callable public object.

    The group factories in ``GROUP_FACTORIES``, ``ThreadData``, ``load``
    and ``store`` of both APIs are typed; every other primitive is
    rejected.
    """

    global _registered
    if _registered:
        return
    for obj, qualified_name in kernel_primitives().items():
        short = qualified_name.rsplit(".", 1)[1]
        name = qualified_name.replace(".", "_")
        operation = group_operation(obj)
        if short in GROUP_FACTORIES:
            register_template(obj, _this_group(obj), name)
        elif short == "ThreadData":
            register_template(
                obj,
                _thread_data(obj, qualified_name),
                name,
                prefer_literal=True,
            )
        elif operation in OPERATIONS.values():
            nb.attach_inline_wrapper(
                obj, make_wrapper(obj, qualified_name, operation)
            )
            register_template(obj, _not_inlined(qualified_name), name)
        else:
            register_template(obj, _rejecting(qualified_name), name)

    for op, kind in OPERATIONS.items():
        qualified = op in QUALIFIED_OPERATIONS
        register_template(
            op,
            _operation(kind),
            f"private_{'qualified_' if qualified else ''}{kind}",
            prefer_literal=True,
        )
    for key in (
        ThreadDataType,
        nb.types.Array,
        nb.types.Number,
        nb.types.Boolean,
    ):
        nb.typing_registry.register_attr(_witness_template(key))
    nb.typing_registry.register_attr(_ThreadDataAttributes)
    for method in ("refine_load", "refine_store"):
        nb.lowering_registry.lower(
            f"coop.thread_data.{method}", nb.types.VarArg(nb.types.Any)
        )(_lower_witness)

    register_template(_never, _never_generic, "never")
    nb.lowering_registry.lower(_never)(_lower_never)
    _registered = True


__all__ = [
    "CONTROLS",
    "GROUP_FACTORIES",
    "OPERATIONS",
    "QUALIFIED_OPERATIONS",
    "WRAPPER_SOURCE",
    "LoadStoreRequest",
    "constructor_alignment",
    "group_operation",
    "is_scalar_value",
    "is_wrapper_location",
    "kernel_primitives",
    "make_wrapper",
    "register",
    "register_template",
    "request_for",
    "with_controls",
]
