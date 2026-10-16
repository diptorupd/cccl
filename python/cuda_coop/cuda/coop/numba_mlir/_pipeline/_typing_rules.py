# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing rules that a compiler extension registers with type inference.

A stand-in for a fourth registry bucket next to templates, attributes and
globals. A rule is keyed by a Numba type class, which matches an instance by
its MRO, a tuple holding one and a bound function whose receiver matches; or
by a typing key, which matches a function type whose ``typing_key`` it is,
as a template's key does. Type inference runs a rule only at the point its
kind names, with the types that matched. Nothing here knows which types or
functions an extension registers, and the compiler's classes stay its own:
each point wraps one method.

==========  ===================================================================
Kind        Called as, at, and what it may do
==========  ===================================================================
``gate``    ``fn(fnty, arg_types, constraint, released)`` when a call's
            arguments are typed (``CallConstraint.resolve``) or an indexed
            assignment's target, index and value are (``fnty`` is ``None``).
            ``WAIT`` leaves it untyped and retries it each round; once
            propagation is otherwise stuck it is released, with
            ``released`` true, and types. ``HOLD`` leaves it untyped as an
            imprecise argument does, and is never released.
``call``    ``fn(typeinfer, constraint, fnty, args, kws, resolve)`` around
            ``TypeInferer.resolve_call``; ``constraint`` is the call being
            resolved, or ``None``. It returns a signature or ``None``,
            usually ``resolve(fnty, args, kws)`` with rewritten arguments or
            a rewritten result.
``add``     ``fn(typeinfer, var, ty, loc)`` before ``ty`` is added to
            ``var`` (``TypeInferer.add_type``) and after a copy whose type
            matches (``copy_type``). It returns the type to add, which unifies
            as ``ty`` would, and may add types to other variables.
``after``   ``fn(typeinfer, constraint)`` after each pass over the
            constraints, for every call, keyed by its callee, and every
            indexed assignment, keyed by its target, in program order and
            rule by rule. It may add types; errors it raises or returns are
            collected as a constraint's are.
``value``   ``fn(typeinfer, inst, value)``, unkeyed, before a Python value
            is typed (``TypeInferer.resolve_value_type``). A type it returns
            is the value's type.
``assign``  ``fn(typeinfer, inst)``, unkeyed, before an assignment is
            constrained (``TypeInferer.typeof_assign``). A type it returns is
            added to the target in place of the assignment's constraint.
``settle``  ``fn(typeinfer, matching, waiting)`` at each fixed point, one
            priority at a time, with the matching type variables and the
            waiting constraints. It may only widen types and returns errors,
            ``PROGRESS`` (propagate again) or ``None``.
``check``   ``fn(typeinfer, matching, released)`` once, at the last fixed
            point. ``released`` maps each released constraint to its types
            when it typed, and now. It returns errors or ``None``.
==========  ===================================================================

Type inference runs in strata: propagate to a fixed point; settle, lowest
priority first, and propagate again after the first priority that made
progress; else release the waiting constraints and propagate again; else
check. With no rule registered, type inference is the compiler's own.
"""

from __future__ import annotations

import itertools
import operator
from collections.abc import Callable, Iterable
from functools import reduce
from typing import Any

from numba_cuda_mlir.numba_cuda import types, utils
from numba_cuda_mlir.numba_cuda.core import errors, typeinfer

GATE, CALL, ADD, AFTER = "gate", "call", "add", "after"
VALUE, ASSIGN, SETTLE, CHECK = "value", "assign", "settle", "check"
KINDS = (GATE, CALL, ADD, AFTER, VALUE, ASSIGN, SETTLE, CHECK)
UNKEYED = (VALUE, ASSIGN)


class _Marker:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


WAIT = _Marker("WAIT")
HOLD = _Marker("HOLD")
PROGRESS = _Marker("PROGRESS")


class RuleSet:
    """Typing rules by kind and key, in registration order.

    ``lookups`` counts how often type inference consulted the set, and
    ``calls`` how often a rule ran.
    """

    def __init__(self) -> None:
        self._rules: dict[str, dict[Any, list[Callable]]] = {
            kind: {} for kind in KINDS
        }
        self._order: dict[tuple[str, Callable], tuple[int, int]] = {}
        self._count = itertools.count()
        self.lookups = 0
        self.calls = 0

    def register(
        self, key: Any, kind: str, fn: Callable, *, priority: int = 0
    ) -> Callable:
        if kind not in KINDS:
            raise ValueError(f"unknown typing rule kind {kind!r}")
        if (key is None) != (kind in UNKEYED):
            raise TypeError(f"a {kind!r} rule key cannot be {key!r}")
        if key is not None:
            hash(key)
        self._rules[kind].setdefault(key, []).append(fn)
        self._order.setdefault((kind, fn), (priority, next(self._count)))
        return fn

    def __bool__(self) -> bool:
        return any(self._rules.values())

    def has(self, kind: str) -> bool:
        return bool(self._rules[kind])

    def unkeyed(self, kind: str) -> tuple[Callable, ...]:
        return tuple(self._rules[kind].get(None, ()))

    def priority(self, kind: str, fn: Callable) -> int:
        return self._order[kind, fn][0]

    def matching(self, kind: str, tys: Iterable[Any]) -> tuple[Callable, ...]:
        """The rules of ``kind`` whose key matches any of ``tys``."""

        self.lookups += 1
        table = self._rules[kind]
        found: dict[Callable, None] = {}
        for ty in tys:
            for key in _keys(table, ty):
                found.update(dict.fromkeys(table[key]))
        return tuple(sorted(found, key=lambda fn: self._order[kind, fn]))


def _keys(table: dict, ty: Any) -> Iterable[Any]:
    """The keys of ``table`` that ``ty`` matches."""

    if ty is None:
        return
    for cls in type(ty).__mro__:
        if cls in table:
            yield cls
    typing_key = getattr(ty, "typing_key", None)
    if typing_key is not None:
        try:
            if typing_key in table:
                yield typing_key
        except TypeError:
            pass
    if isinstance(ty, types.BoundFunction):
        yield from _keys(table, ty.this)
    elif isinstance(ty, types.BaseTuple):
        for item in ty.types:
            yield from _keys(table, item)


RULES = RuleSet()
register = RULES.register


class _State:
    """One type inference's waiting, released and active constraints.

    ``released`` maps a released constraint to the types it typed with,
    recorded the first time it types after its release. ``local`` holds
    an extension's own state for this inference, by key.
    """

    def __init__(self) -> None:
        self.waiting: dict[Any, None] = {}
        self.released: dict[Any, tuple | None] = {}
        self.active: list[Any] = []
        self.local: dict[Any, Any] = {}


def _state(typeinfer_) -> _State:
    state = vars(typeinfer_).get("_typing_rules")
    if state is None:
        state = typeinfer_._typing_rules = _State()
    return state


def local(typeinfer_, key: Any, factory: Callable[[Any], Any]) -> Any:
    """The state ``factory(typeinfer_)`` an extension keeps under ``key``."""

    found = _state(typeinfer_).local
    if key not in found:
        found[key] = factory(typeinfer_)
    return found[key]


def _assignment_vars(constraint) -> tuple:
    index = (
        constraint.index_var
        if isinstance(constraint, typeinfer.StaticSetItemConstraint)
        else constraint.index
    )
    return constraint.target, index, constraint.value


def _constraint_types(constraint, typevars, fnty=None) -> tuple | None:
    """A constraint's argument types and callee, or ``None`` if untyped."""

    if isinstance(constraint, typeinfer.CallConstraint):
        folded = typeinfer.fold_arg_vars(
            typevars, constraint.args, constraint.vararg, constraint.kws
        )
        if folded is None:
            return None
        pos_args, kw_args = folded
        return (*pos_args, *kw_args.values(), fnty)
    tvs = [typevars[var.name] for var in _assignment_vars(constraint)]
    if not all(tv.defined for tv in tvs):
        return None
    return (*(tv.getone() for tv in tvs), None)


def _gate(constraint, typeinfer_, typevars, fnty) -> bool:
    """Whether a gate rule keeps ``constraint`` from typing this round."""

    seen = _constraint_types(constraint, typevars, fnty)
    if seen is None:
        return False
    arg_types = seen[:-1]
    # As the compiler: a call waits for imprecise arguments but arrays.
    if fnty is not None and not all(
        a.is_precise() or isinstance(a, types.Array) for a in arg_types
    ):
        return False
    rules = RULES.matching(GATE, seen)
    if not rules:
        return False
    state = _state(typeinfer_)
    released = constraint in state.released
    for fn in rules:
        RULES.calls += 1
        verdict = fn(fnty, arg_types, constraint, released)
        if verdict is HOLD:
            state.waiting.pop(constraint, None)
            return True
        if verdict is WAIT and not released:
            state.waiting[constraint] = None
            return True
    state.waiting.pop(constraint, None)
    if released and state.released[constraint] is None:
        state.released[constraint] = seen
    return False


def _subject(constraint, typevars) -> Any:
    """What an ``after`` rule is keyed on: a call's callee or a target."""

    if isinstance(constraint, typeinfer.IntrinsicCallConstraint):
        return None
    if isinstance(constraint, typeinfer.CallConstraint):
        return typevars[constraint.func].type
    if isinstance(
        constraint,
        (typeinfer.SetItemConstraint, typeinfer.StaticSetItemConstraint),
    ):
        return typevars[constraint.target.name].type
    return None


def _after(typeinfer_) -> list:
    """Run ``after`` rules over the calls and assignments, rule by rule."""

    typevars = typeinfer_.typevars
    by_rule: dict[Callable, list] = {}
    for _, constraint in typeinfer_.calls:
        subject = _subject(constraint, typevars)
        if subject is None:
            continue
        for fn in RULES.matching(AFTER, (subject,)):
            by_rule.setdefault(fn, []).append(constraint)
    found = []
    for fn in sorted(by_rule, key=lambda fn: RULES._order[AFTER, fn]):
        for constraint in by_rule[fn]:
            loc = constraint.loc
            with typeinfer_.warnings.catch_warnings(
                filename=loc.filename, lineno=loc.line
            ):
                RULES.calls += 1
                try:
                    raised = list(fn(typeinfer_, constraint) or ())
                except (errors.ForceLiteralArg, errors.TypingError) as exc:
                    raised = [exc]
            for exc in raised:
                if isinstance(exc, errors.ForceLiteralArg):
                    found.append(exc)
                    continue
                new_exc = errors.TypingError(
                    str(exc), loc=loc, highlighting=False
                )
                found.append(utils.chain_exception(new_exc, exc))
    return found


def _matching_vars(kind: str, typeinfer_) -> dict[Callable, dict[str, Any]]:
    by_rule: dict[Callable, dict[str, Any]] = {}
    for name, tv in typeinfer_.typevars.items():
        if tv.type is None:
            continue
        for fn in RULES.matching(kind, (tv.type,)):
            by_rule.setdefault(fn, {})[name] = tv.type
    return by_rule


def _settle(typeinfer_, waiting: tuple) -> tuple[list, bool]:
    """Run settle rules one priority at a time: conflicts, and progress."""

    by_rule = _matching_vars(SETTLE, typeinfer_)
    ordered = sorted(by_rule, key=lambda fn: RULES._order[SETTLE, fn])
    for _, level in itertools.groupby(
        ordered, key=lambda fn: RULES.priority(SETTLE, fn)
    ):
        token = typeinfer_.get_state_token()
        conflicts, progress = [], False
        for fn in level:
            RULES.calls += 1
            result = fn(typeinfer_, by_rule[fn], waiting)
            if result is PROGRESS:
                progress = True
            else:
                conflicts.extend(result or ())
        if conflicts:
            return conflicts, False
        if progress or typeinfer_.get_state_token() != token:
            return [], True
    return [], False


def _released_now(typeinfer_, state: _State) -> dict[Any, tuple]:
    typevars = typeinfer_.typevars
    released = {}
    for constraint, seen in state.released.items():
        if seen is None:
            continue
        func = getattr(constraint, "func", None)
        fnty = typevars[func].type if isinstance(func, str) else seen[-1]
        now = _constraint_types(constraint, typevars, fnty)
        released[constraint] = (seen, now)
    return released


def _raise_or_return(found: list, raise_errors: bool) -> list | None:
    if not found:
        return None
    if not raise_errors:
        return found
    force_lit_args = [e for e in found if isinstance(e, errors.ForceLiteralArg)]
    if not force_lit_args:
        raise found[0]
    raise reduce(operator.or_, force_lit_args)


def points() -> tuple[tuple[Any, str, Any], ...]:
    """The compiler methods that run typing rules, wrapping the current ones.

    Scope: registered rules only. A point with no rule of its kind calls the
    compiler's method and nothing else; a kernel whose types match no key
    consults the set but runs no keyed rule.
    """

    TypeInferer = typeinfer.TypeInferer
    propagate = TypeInferer.propagate
    add_type = TypeInferer.add_type
    copy_type = TypeInferer.copy_type
    resolve_call = TypeInferer.resolve_call
    resolve_value_type = TypeInferer.resolve_value_type
    typeof_assign = TypeInferer.typeof_assign
    network_propagate = typeinfer.ConstraintNetwork.propagate
    resolve = typeinfer.CallConstraint.resolve
    setitem = typeinfer.SetItemConstraint.__call__
    static_setitem = typeinfer.StaticSetItemConstraint.__call__

    def resolve_with_rules(self, typeinfer_, typevars, fnty):
        if RULES.has(GATE) and _gate(self, typeinfer_, typevars, fnty):
            return None
        if not RULES.has(CALL):
            return resolve(self, typeinfer_, typevars, fnty)
        active = _state(typeinfer_).active
        active.append(self)
        try:
            return resolve(self, typeinfer_, typevars, fnty)
        finally:
            active.pop()

    def gated(call):
        def call_with_gates(self, typeinfer_):
            if RULES.has(GATE) and _gate(
                self, typeinfer_, typeinfer_.typevars, None
            ):
                return None
            return call(self, typeinfer_)

        call_with_gates.__qualname__ = call.__qualname__
        return call_with_gates

    def resolve_call_with_rules(self, fnty, pos_args, kw_args):
        if not RULES.has(CALL):
            return resolve_call(self, fnty, pos_args, kw_args)
        rules = RULES.matching(CALL, (*pos_args, *kw_args.values(), fnty))
        if not rules:
            return resolve_call(self, fnty, pos_args, kw_args)
        active = _state(self).active
        constraint = active[-1] if active else None

        def chain(index, fnty, args, kws):
            if index == len(rules):
                return resolve_call(self, fnty, args, kws)
            RULES.calls += 1
            return rules[index](
                self,
                constraint,
                fnty,
                args,
                kws,
                lambda f, a, k: chain(index + 1, f, a, k),
            )

        return chain(0, fnty, tuple(pos_args), dict(kw_args))

    def add_type_with_rules(self, var, tp, loc, unless_locked=False):
        if RULES.has(ADD):
            for fn in RULES.matching(ADD, (tp,)):
                RULES.calls += 1
                tp = fn(self, var, tp, loc)
        return add_type(self, var, tp, loc, unless_locked)

    def copy_type_with_rules(self, src_var, dest_var, loc):
        copy_type(self, src_var, dest_var, loc)
        if RULES.has(ADD):
            copied = self.typevars[dest_var].type
            if copied is not None and RULES.matching(ADD, (copied,)):
                self.add_type(dest_var, copied, loc)

    def resolve_value_type_with_rules(self, inst, val):
        for fn in RULES.unkeyed(VALUE):
            RULES.calls += 1
            ty = fn(self, inst, val)
            if ty is not None:
                return ty
        return resolve_value_type(self, inst, val)

    def typeof_assign_with_rules(self, inst):
        for fn in RULES.unkeyed(ASSIGN):
            RULES.calls += 1
            ty = fn(self, inst)
            if ty is not None:
                self.add_type(inst.target.name, ty, inst.loc)
                return
        typeof_assign(self, inst)

    def network_propagate_with_rules(self, typeinfer_):
        found = network_propagate(self, typeinfer_)
        if RULES.has(AFTER):
            found.extend(_after(typeinfer_))
        return found

    def propagate_in_strata(self, raise_errors=True):
        if not RULES:
            return propagate(self, raise_errors=raise_errors)
        state = _state(self)
        while True:
            found = propagate(self, raise_errors=False) or []
            conflicts, progress = _settle(self, tuple(state.waiting))
            if conflicts:
                found.extend(conflicts)
                break
            if progress:
                continue
            if state.waiting:
                state.released.update(dict.fromkeys(state.waiting))
                state.waiting.clear()
                continue
            break
        found.extend(_run_check(self, state))
        return _raise_or_return(found, raise_errors)

    return (
        (TypeInferer, "propagate", propagate_in_strata),
        (TypeInferer, "add_type", add_type_with_rules),
        (TypeInferer, "copy_type", copy_type_with_rules),
        (TypeInferer, "resolve_call", resolve_call_with_rules),
        (TypeInferer, "resolve_value_type", resolve_value_type_with_rules),
        (TypeInferer, "typeof_assign", typeof_assign_with_rules),
        (
            typeinfer.ConstraintNetwork,
            "propagate",
            network_propagate_with_rules,
        ),
        (typeinfer.CallConstraint, "resolve", resolve_with_rules),
        (typeinfer.SetItemConstraint, "__call__", gated(setitem)),
        (typeinfer.StaticSetItemConstraint, "__call__", gated(static_setitem)),
    )


def _run_check(typeinfer_, state: _State) -> list:
    found = []
    released = _released_now(typeinfer_, state)
    for fn, matching in _matching_vars(CHECK, typeinfer_).items():
        RULES.calls += 1
        found.extend(fn(typeinfer_, matching, released) or ())
    return found


__all__ = [
    "ADD",
    "AFTER",
    "ASSIGN",
    "CALL",
    "CHECK",
    "GATE",
    "HOLD",
    "KINDS",
    "PROGRESS",
    "RULES",
    "SETTLE",
    "VALUE",
    "WAIT",
    "RuleSet",
    "local",
    "points",
    "register",
]
