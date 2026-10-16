# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative mutation contracts as typing rules on Numba's own inference.

Native constraints own ordinary expressions, calls and control-flow joins.
Allocation origins retain the distinction between inferred writes and typed
producers. Producer rules establish those dtypes before indexed writes
or their consumers are typed; Store only supplies a dtype when writes do not.

``register`` adds these rules to ``_typing_rules``; the compiler's type
inferer, constraints and propagation loop stay its own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

from . import _bridges as nb
from . import _typing_rules as rules
from ._types import EXPLICIT, PRODUCER, STORE_DST, WRITE, ThreadDataType


@dataclass(frozen=True)
class DtypeSource:
    """Derive a mutation dtype from one typed operation operand."""

    operand: int | str
    resolve: Callable[[Any], Any]


@dataclass(frozen=True)
class OperationInference:
    """Output/source operand pairs for producer and Store contracts."""

    producers: tuple[tuple[int | str, int | str | DtypeSource], ...] = ()
    checked_producers: tuple[
        tuple[int | str, int | str | DtypeSource], ...
    ] = ()
    stores: tuple[tuple[int | str, int | str | DtypeSource], ...] = ()


operation_inference_contracts: dict[Callable, OperationInference] = {}
constructor_origin_factories: dict[Callable, Callable[[Any, tuple], Any]] = {}
operation_static_arguments: dict[
    Callable, dict[int | str, Callable[[Any], Any]]
] = {}
operation_literal_arguments: dict[
    Any, dict[int | str, Callable[[Any], Any]]
] = {}


def _argument(call, operand):
    if isinstance(operand, int):
        return call.args[operand] if operand < len(call.args) else None
    return dict(call.kws).get(operand)


def numpy_integer_literal(value):
    """Expose a captured NumPy integer as a call-local static control."""

    import numpy as np

    if isinstance(value, np.integer):
        return nb.types.IntegerLiteral(int(value))
    return None


def static_call_arguments(func_ir, call, typing_key, args, kws):
    """Normalize registered constant controls only at their call site."""

    controls = operation_literal_arguments.get(typing_key)
    if not controls:
        return args, kws
    args, kws = list(args), dict(kws)
    for operand, factory in controls.items():
        variable = _argument(call, operand)
        if variable is None:
            continue
        try:
            value = func_ir.infer_constant(variable)
        except nb.errors.ConstantInferenceError:
            continue
        normalized = factory(value)
        if normalized is not None:
            if isinstance(operand, int):
                args[operand] = normalized
            else:
                kws[operand] = normalized
    return tuple(args), kws


def _contains_imprecise_payload(value: Any) -> bool:
    if isinstance(value, ThreadDataType):
        return not value.is_precise()
    if isinstance(value, nb.types.BaseTuple):
        return any(
            _contains_imprecise_payload(item)
            for item in cast(Iterable[Any], value)
        )
    return False


def _native_type(value: Any) -> Any:
    """Keep inference-local allocation policy out of helper signatures."""

    if isinstance(value, ThreadDataType):
        return ThreadDataType(value.dtype, value.items, EXPLICIT)
    if isinstance(value, nb.types.BaseTuple):
        return nb.types.BaseTuple.from_types(
            tuple(_native_type(item) for item in cast(Iterable[Any], value))
        )
    return value


class _PayloadWrite:
    """Apply write agreement to every possible write-inferred allocation."""

    def __init__(self, statement: Any) -> None:
        self.target = statement.target
        self.value = statement.value
        self.loc = statement.loc

    def __call__(self, infer: Any) -> None:
        payload = infer.variable_type(self.target)
        value = infer.variable_type(self.value)
        if not isinstance(payload, ThreadDataType) or value is None:
            return
        producers = infer.producer_origins()
        if producers is None:
            return
        # A separately compiled helper receives an already typed array.
        # Its ordinary writes cast to that dtype; only constructors owned
        # by this inference attempt participate in write-based inference.
        inferred = payload.inferred_origins & infer.payload_origins.keys()
        if not inferred - producers:
            return
        # Reads of inferred allocations depend on the context being solved.
        # They cannot fix another allocation before its independent writes
        # and Store destination have been considered. Once the target has a
        # dtype, these dependent writes use ordinary array casting.
        dependent = infer.inferred_read_origins() & infer.value_origins(
            self.value
        )
        if dependent and (
            payload.is_precise() or not infer.allow_dependent_writes
        ):
            return
        dtype = value.dtype if isinstance(value, nb.types.Array) else value
        dtype = nb.unliteral(dtype)
        if not dtype.is_precise():
            return
        if payload.is_precise() and payload.dtype != dtype:
            raise nb.errors.TypingError(
                "Failed to infer a consistent dtype from "
                "coop.ThreadData writes.",
                loc=self.loc,
            )
        if not payload.is_precise():
            infer.add_type(
                self.target.name, payload.with_dtype(dtype, WRITE), self.loc
            )


class _PayloadMutation:
    """Refine producer outputs, or validate/fill a Store payload dtype."""

    def __init__(
        self,
        payload: Any,
        source: Any,
        strength: int,
        loc: Any,
        resolver=None,
        checked=False,
    ):
        self.payload = payload
        self.source = source
        self.strength = strength
        self.loc = loc
        self.resolver = resolver
        self.checked = checked

    def __call__(self, infer: Any) -> None:
        payload = infer.variable_type(self.payload)
        source = infer.variable_type(self.source)
        if not isinstance(payload, ThreadDataType) or source is None:
            return
        dtype = (
            self.resolver(source)
            if self.resolver is not None
            else source.dtype
            if isinstance(source, nb.types.Array)
            else source
        )
        if dtype is None:
            return
        dtype = nb.unliteral(dtype)
        if not dtype.is_precise():
            return
        if self.strength == STORE_DST and not payload.is_precise():
            written = infer.write_origins()
            if (
                written is None or payload.origins & written
            ) and not infer.allow_fallback:
                return
        if payload.is_precise() and payload.dtype != dtype:
            raise nb.errors.TypingError(
                f"cooperative memory dtype {dtype} does not match "
                f"payload dtype {payload.dtype}",
                loc=self.loc,
            )
        infer.add_type(
            self.payload.name,
            payload.with_dtype(dtype, max(payload.strength, self.strength)),
            self.loc,
        )


def _mutations(call: Any, contract: OperationInference) -> list:
    """The producer and Store mutations a contract makes of one call."""

    mutations = []
    for pairs, strength, checked in (
        (contract.producers, PRODUCER, False),
        (contract.checked_producers, PRODUCER, True),
        (contract.stores, STORE_DST, False),
    ):
        for output, source in pairs:
            resolver = None
            if isinstance(source, DtypeSource):
                resolver = source.resolve
                source = source.operand
            target = _argument(call, output)
            value = _argument(call, source)
            if target is None or value is None:
                continue
            mutations.append(
                _PayloadMutation(
                    target, value, strength, call.loc, resolver, checked
                )
            )
    return mutations


def _typing_key(ty: Any) -> Any:
    key = getattr(ty, "typing_key", None)
    try:
        hash(key)
    except TypeError:
        return None
    return key


_payload_constructors: set = set()


class _Inference:
    """One type inference's allocation origins, contracts and strata.

    Constructor targets and contract calls are found from the callee types
    of the inference's own call constraints. Static callback arguments are
    found by constant inference on the IR, as their values have no type.
    """

    def __init__(self, typeinfer: Any) -> None:
        self.typeinfer = typeinfer
        self.func_ir = typeinfer.func_ir
        self.origin_scope = object()
        self.payload_origins: dict[tuple, ThreadDataType] = {}
        self.refining_origins: set[tuple] = set()
        self.unstamped: Any = None
        self.allow_fallback = False
        self.allow_dependent_writes = False
        self._static: tuple[dict, dict] | None = None
        self._targets: dict[str, Any] = {}
        self._indexed = 0
        self._mutations: dict[Any, list] = {}
        self._writes: dict[Any, _PayloadWrite] = {}

    def variable_type(self, variable):
        value = self.typeinfer.typevars[variable.name]
        return value.getone() if value.defined else None

    def add_type(self, var, tp, loc):
        self.typeinfer.add_type(var, tp, loc)

    def _call_constraints(self):
        for _, constraint in self.typeinfer.calls:
            if type(constraint) is nb.typeinfer.CallConstraint:
                yield constraint

    def constructor(self, var: str) -> Any:
        """The typing key of the call that assigns ``var``, if typed."""

        calls = self.typeinfer.calls
        if self._indexed != len(calls):
            self._targets = {c.target: c for c in self._call_constraints()}
            self._indexed = len(calls)
        call = self._targets.get(var)
        if call is None:
            return None
        return _typing_key(self.typeinfer.typevars[call.func].type)

    def mutations_of(self, call: Any) -> list:
        found = self._mutations.get(call)
        if found is None:
            callee = self.typeinfer.typevars[call.func].type
            contract = operation_inference_contracts.get(_typing_key(callee))
            if contract is None:
                return []
            found = self._mutations[call] = _mutations(call, contract)
        return found

    def payload_mutations(self):
        for call in self._call_constraints():
            yield from self.mutations_of(call)

    def write_of(self, constraint: Any) -> _PayloadWrite:
        write = self._writes.get(constraint)
        if write is None:
            write = self._writes[constraint] = _PayloadWrite(constraint)
        return write

    def payload_writes(self):
        for statement, constraint in self.typeinfer.calls:
            if isinstance(statement, (nb.ir.SetItem, nb.ir.StaticSetItem)):
                yield self.write_of(constraint)

    def producer_origins(self):
        origins = set()
        for mutation in self.payload_mutations():
            if mutation.strength != PRODUCER or mutation.checked:
                continue
            payload = self.variable_type(mutation.payload)
            if payload is None:
                return None
            if isinstance(payload, ThreadDataType):
                origins.update(payload.origins)
        return origins

    def inferred_read_origins(self):
        """Allocations whose reads depend on the dtype being inferred."""

        inferred = {
            origin
            for origin, declared in self.payload_origins.items()
            if origin in declared.inferred_origins
        }
        return inferred - (self.producer_origins() or set())

    def value_origins(self, value):
        """Read payload dependencies without evaluating ordinary dtypes.

        Native SSA definitions retain aliases, phis and expression inputs.
        A helper's payload arguments count conservatively as dependencies;
        its native call constraint still owns the return type.
        """

        origins, pending, seen = set(), [value], set()
        while pending:
            variable = pending.pop()
            if variable.name in seen:
                continue
            seen.add(variable.name)
            ty = self.variable_type(variable)
            if isinstance(ty, ThreadDataType):
                origins.update(ty.origins)
                continue
            for definition in self.func_ir._definitions.get(variable.name, ()):
                if isinstance(definition, nb.ir.Var):
                    pending.append(definition)
                elif isinstance(definition, nb.ir.Expr):
                    pending.extend(definition.list_vars())
        return origins

    def write_origins(self):
        origins = set()
        for write in self.payload_writes():
            payload = self.variable_type(write.target)
            if payload is None:
                return None
            if isinstance(payload, ThreadDataType):
                origins.update(payload.origins)
        return origins

    def static(self) -> tuple[dict, dict]:
        """Static types of untypeable callbacks, by value id and variable."""

        if self._static is not None:
            return self._static
        values: dict[int, Any] = {}
        arguments: dict[str, Any] = {}
        self._static = values, arguments
        func_ir, context = self.func_ir, self.typeinfer.context
        for block in func_ir.blocks.values():
            for statement in block.find_insts(nb.ir.Assign):
                call = statement.value
                if not isinstance(call, nb.ir.Expr) or call.op != "call":
                    continue
                try:
                    function = func_ir.infer_constant(call.func)
                except nb.errors.ConstantInferenceError:
                    continue
                for operand, factory in operation_static_arguments.get(
                    function, {}
                ).items():
                    variable = _argument(call, operand)
                    if variable is None:
                        continue
                    try:
                        value = func_ir.infer_constant(variable)
                    except nb.errors.ConstantInferenceError:
                        continue
                    try:
                        context.resolve_value_type(value)
                    except ValueError:
                        static_type = factory(value)
                        values[id(value)] = static_type
                        arguments[variable.name] = static_type
        return self._static


def _inference(typeinfer: Any) -> _Inference:
    return rules.local(typeinfer, _Inference, _Inference)


# Rules ------------------------------------------------------------------------


def _payload_gate(fnty, arg_types, constraint, released):
    """Writes wait for a payload dtype; helpers wait for precise payloads.

    An indexed write to a payload whose dtype is not yet known would fix it
    through Numba's setitem refinement; producers and write agreement
    establish it first. A separately compiled helper is typed against
    whatever payload it is given, so it waits until the dtype is known.
    """

    if fnty is None:
        target = arg_types[0]
        if isinstance(target, ThreadDataType) and not target.is_precise():
            return rules.HOLD
        return None
    if isinstance(fnty, nb.types.Dispatcher) and any(
        _contains_imprecise_payload(arg) for arg in arg_types
    ):
        return rules.HOLD
    return None


def _static_controls(typeinfer, constraint, fnty, args, kws, resolve):
    if type(constraint) is nb.typeinfer.CallConstraint:
        args, kws = static_call_arguments(
            typeinfer.func_ir, constraint, _typing_key(fnty), args, kws
        )
    return resolve(fnty, args, kws)


def _helper_arguments(typeinfer, constraint, fnty, args, kws, resolve):
    if isinstance(fnty, nb.types.Dispatcher):
        args = tuple(_native_type(arg) for arg in args)
        kws = {name: _native_type(arg) for name, arg in kws.items()}
    return resolve(fnty, args, kws)


def _iteration_waits(typeinfer, constraint, fnty, args, kws, resolve):
    result = resolve(fnty, args, kws)
    if (
        result is not None
        and isinstance(result.return_type, nb.types.IteratorType)
        and any(
            _contains_imprecise_payload(arg) for arg in (*args, *kws.values())
        )
    ):
        return None
    return result


def _constructor_signature(typeinfer, constraint, fnty, args, kws, resolve):
    """A constructor call's signature returns its target's current type."""

    result = resolve(fnty, args, kws)
    if result is None or type(constraint) is not nb.typeinfer.CallConstraint:
        return result
    typeinfer.add_type(constraint.target, result.return_type, constraint.loc)
    return result.replace(
        return_type=typeinfer.typevars[constraint.target].getone()
    )


def _origins(typeinfer, var, tp, loc):
    """Stamp a constructor's allocation origin; refine it from its aliases."""

    state = _inference(typeinfer)
    if state.unstamped is tp:
        state.unstamped = None
        return tp
    key = state.constructor(var)
    factory = constructor_origin_factories.get(key)
    if factory is not None:
        tp = factory(tp, (state.origin_scope, var))
    if not isinstance(tp, ThreadDataType):
        return tp
    if key in _payload_constructors:
        origin = (state.origin_scope, var)
        origins = frozenset({origin})
        inferred = (
            frozenset()
            if tp.is_precise() and tp.strength == EXPLICIT
            else origins
        )
        tp = tp.with_origins(origins, inferred)
        state.payload_origins.setdefault(origin, tp)
    if tp.is_precise():
        for origin in tp.origins:
            declared = state.payload_origins.get(origin)
            if declared is None or origin in state.refining_origins:
                continue
            refined = declared.with_dtype(
                tp.dtype, max(declared.strength, tp.strength)
            )
            state.refining_origins.add(origin)
            state.unstamped = refined
            try:
                typeinfer.add_type(origin[1], refined, loc)
            finally:
                state.unstamped = None
                state.refining_origins.remove(origin)
    return tp


def _static_value(typeinfer, inst, value):
    return _inference(typeinfer).static()[0].get(id(value))


def _static_argument(typeinfer, inst):
    return _inference(typeinfer).static()[1].get(inst.target.name)


def _apply_contract(typeinfer, call):
    state = _inference(typeinfer)
    raised = []
    for mutation in state.mutations_of(call):
        try:
            mutation(state)
        except nb.errors.TypingError as exc:
            raised.append(exc)
    return raised


def _apply_write(typeinfer, constraint):
    if isinstance(
        constraint,
        (nb.typeinfer.SetItemConstraint, nb.typeinfer.StaticSetItemConstraint),
    ):
        state = _inference(typeinfer)
        state.write_of(constraint)(state)


def _store_fallback(typeinfer, matching, waiting):
    """Let a Store destination type a payload whose writes all depend."""

    state = _inference(typeinfer)
    if state.allow_fallback or state.allow_dependent_writes:
        return None
    if any(
        mutation.strength == STORE_DST
        and isinstance(
            payload := state.variable_type(mutation.payload), ThreadDataType
        )
        and not payload.is_precise()
        for mutation in state.payload_mutations()
    ):
        state.allow_fallback = True
        return rules.PROGRESS
    return None


def _dependent_writes(typeinfer, matching, waiting):
    """Then let writes that read inferred allocations type their target."""

    state = _inference(typeinfer)
    if state.allow_dependent_writes:
        return None
    if state.payload_origins and any(state.payload_writes()):
        state.allow_dependent_writes = True
        return rules.PROGRESS
    return None


_registered = False


def register() -> None:
    """Register the cooperative typing rules, after every family's contracts."""

    global _registered
    if _registered:
        return
    from ._temp_storage import TempStorageType
    from ._templates import kernel_primitives

    _payload_constructors.update(
        function
        for function, name in kernel_primitives().items()
        if name.rsplit(".", 1)[-1] == "ThreadData"
    )
    rules.register(ThreadDataType, rules.GATE, _payload_gate)
    for key in operation_literal_arguments:
        rules.register(key, rules.CALL, _static_controls)
    rules.register(ThreadDataType, rules.CALL, _helper_arguments)
    rules.register(ThreadDataType, rules.CALL, _iteration_waits)
    for key in (*_payload_constructors, *constructor_origin_factories):
        rules.register(key, rules.CALL, _constructor_signature)
    rules.register(ThreadDataType, rules.ADD, _origins)
    rules.register(TempStorageType, rules.ADD, _origins)
    rules.register(None, rules.VALUE, _static_value)
    rules.register(None, rules.ASSIGN, _static_argument)
    for key in operation_inference_contracts:
        rules.register(key, rules.AFTER, _apply_contract)
    rules.register(ThreadDataType, rules.AFTER, _apply_write)
    rules.register(ThreadDataType, rules.SETTLE, _store_fallback, priority=0)
    rules.register(ThreadDataType, rules.SETTLE, _dependent_writes, priority=1)
    _registered = True


__all__ = [
    "DtypeSource",
    "OperationInference",
    "constructor_origin_factories",
    "numpy_integer_literal",
    "operation_inference_contracts",
    "operation_literal_arguments",
    "operation_static_arguments",
    "register",
    "static_call_arguments",
]
