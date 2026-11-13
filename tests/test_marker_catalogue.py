"""Evidence markers of ``src/``, ``scripts/`` and ``web/src``, as a ratchet.

A marker is one line, ``[marker-name] {json}``, that is meant to carry counts, kinds and
categories only. Nothing owns that contract yet (ADR 0007), so this test measures what
exists and binds it by equality to the tables in ``tests/support/marker_catalogue.py``:

- ``EMIT_SITES``: every marker name, the files that write it and how many sites each has;
- ``UNPROVEN_VALUES``: every value at a write site that this scan cannot show to be a
  literal, a number, a flag or a member of a literal set;
- ``READERS``: every place that recognises a marker by hand;
- ``COMPUTED_PREFIXES``: every prefix assembled from parts;
- ``NEVER_WRITTEN``: every name that is read or declared and has no write site.

The test fails on a measurement that is not listed and on a listing that no longer
matches, so the last four tables can only shrink. ``uv run python -m
tests.test_marker_catalogue`` prints the tables as measured.

What shows a value to be content-free is stated in two places only: ``_judge`` for the
shape of an expression, and ``_guarded`` with ``_vocabulary`` for a conditional that
checks a value against a set. A disjunction shows nothing unless every alternative does,
and a named set counts only when it can be read from source as a collection of constants.

Limits. The measurement is static and syntactic. A prefix is seen only as a string or
bytes constant that starts with ``[name]`` followed by a space or the end of the constant,
or as one of the assembled forms in ``_computed``. A prefix built any other way (joined
from a list, read from data, added by a helper) is not seen. ``UNPROVEN_VALUES`` is a
superset of the free strings: a value it lists may be a number or a fixed label that the
scan could not follow, and a value it accepts is literal only as far as one module shows.

Two limits of the kind tags in ``UNPROVEN_VALUES``:

- A value is tagged by the expression the scan stopped at, not by where the value came
  from. One that reaches the payload through a tuple assignment, a loop or a ``with``
  target is tagged ``bound`` whatever it is: ``kind, tool = type(error).__name__, None``
  yields ``bound``, not ``class-name``. A ``bound`` row has to be read in source.
- A payload printed whole (the result of a call, an awaited value, a built dictionary) is
  one row with the field ``*`` or ``**``. The scan cannot see its members, and a change
  inside the function that builds it changes no row.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pytest

from tests.support import marker_catalogue as tables

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = ("src", "scripts")
WEB_ROOT = "web/src"

_NAME = r"[a-z][a-z0-9]*(?:-[a-z0-9]+)+"
# The whole prefix, at the start of a constant, followed by a space or nothing.
_PREFIX = re.compile(r"\[(" + _NAME + r")\](?: |\Z)")
# A file with none of these cannot hold anything the scan looks for, and is not parsed.
_CANDIDATE = re.compile(
    r"\[" + _NAME + r"\]|\[\{[^}\n]*\}\][ {]|[\"']\[[\"']\s*\+|\[%s\]|\\\[.*\\\]"
)
_ENTRY_WIDTH = 84
_GENERIC = "<any>"

_READ_METHODS = frozenset({
    "startswith", "endswith", "removeprefix", "removesuffix", "find", "rfind", "index",
    "count", "partition", "rpartition", "split",
})
_SINK_METHODS = frozenset({
    "write", "debug", "info", "warning", "error", "critical", "exception", "log",
})
_NUMBER_CALLS = frozenset({"len", "int", "bool", "float", "ord"})
_PASS_THROUGH_CALLS = frozenset({
    "dumps", "frozen", "dict", "sorted", "list", "tuple", "set", "frozenset", "min", "max",
    "sum", "abs", "round",
})
_STRING_CALLS = frozenset({"str", "repr", "format", "join"})
_NUMBER_TYPES = frozenset({"int", "bool", "float"})
_Function = ast.FunctionDef | ast.AsyncFunctionDef
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class Scan:
    writes: dict[str, dict[str, int]] = field(default_factory=dict)
    reads: dict[str, dict[str, int]] = field(default_factory=dict)
    declared: set[str] = field(default_factory=set)
    unproven: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    computed: dict[str, int] = field(default_factory=dict)

    def write(self, name: str, where: str) -> None:
        sites = self.writes.setdefault(name, {})
        sites[where] = sites.get(where, 0) + 1

    def read(self, name: str, where: str) -> None:
        sites = self.reads.setdefault(name, {})
        sites[where] = sites.get(where, 0) + 1

    def doubt(self, where: str, marker: str, entry: str) -> None:
        self.unproven.setdefault(where, {}).setdefault(marker, set()).add(entry)


def _text(node: ast.AST) -> str:
    return " ".join(ast.unparse(node).split())


def _entry(label: str, kind: str, node: ast.AST | str) -> str:
    text = node if type(node) is str else _text(node)
    head = f"{label}: {kind}: "
    room = max(_ENTRY_WIDTH - len(head), 12)
    return head + (text if len(text) <= room else text[: room - 1] + "~")


def _prefix_name(node: ast.AST) -> str | None:
    if type(node) is not ast.Constant or type(node.value) not in (str, bytes):
        return None
    value = node.value if type(node.value) is str else node.value.decode("latin-1")
    match = _PREFIX.match(value)
    return match.group(1) if match else None


def _called(call: ast.Call) -> str:
    return getattr(call.func, "id", None) or getattr(call.func, "attr", "") or ""


def _is_sink(call: ast.Call) -> bool:
    if type(call.func) is ast.Name:
        return call.func.id == "print"
    return type(call.func) is ast.Attribute and call.func.attr in _SINK_METHODS


class _Module:
    """One parsed file: classify each prefix it holds and judge what is written with it."""

    def __init__(self, where: str, tree: ast.Module, scan: Scan) -> None:
        self.where = where
        self.tree = tree
        self.scan = scan
        self.nodes = list(ast.walk(tree))
        self.parent: dict[ast.AST, ast.AST] = {}
        self.functions: dict[str, list[_Function]] = defaultdict(list)
        self.calls: dict[str, list[ast.Call]] = defaultdict(list)
        for node in self.nodes:
            for child in ast.iter_child_nodes(node):
                self.parent[child] = node
            if type(node) in _FUNCTIONS:
                self.functions[node.name].append(node)
            elif type(node) is ast.Call:
                self.calls[_called(node)].append(node)
        self.inside: dict[int, list[ast.AST]] = {id(tree): self.nodes}
        self.helpers: list[tuple[_Function, str]] = []
        self.judged: set[tuple[int, int, str]] = set()

    def _nodes(self, scope: ast.AST) -> list[ast.AST]:
        if id(scope) not in self.inside:
            self.inside[id(scope)] = list(ast.walk(scope))
        return self.inside[id(scope)]

    # -- structure ------------------------------------------------------------------

    def _function(self, node: ast.AST) -> _Function | None:
        current = self.parent.get(node)
        while current is not None and type(current) not in _FUNCTIONS:
            current = self.parent.get(current)
        return current

    def _call_holding(self, node: ast.AST) -> tuple[ast.Call | None, ast.AST]:
        """The nearest call that has ``node`` inside an argument, and that argument."""
        current = node
        while current in self.parent and not isinstance(current, ast.stmt):
            above = self.parent[current]
            if type(above) is ast.Call and current is not above.func:
                return above, current
            if type(above) is ast.keyword:
                current = above
                continue
            current = above
        return None, current

    # -- classification -------------------------------------------------------------

    def run(self) -> None:
        for node in self.nodes:
            name = _prefix_name(node)
            if name is not None:
                self._constant(node, name)
            self._computed(node)
        done: set[tuple[int, str]] = set()
        while self.helpers:
            function, parameter = self.helpers.pop()
            if (id(function), parameter) in done:
                continue
            done.add((id(function), parameter))
            for use in self._nodes(function):
                if type(use) is ast.Name and use.id == parameter and type(use.ctx) is ast.Load:
                    self._use(use, f"*{function.name}", owner=function)

    def _constant(self, node: ast.AST, name: str) -> None:
        above = self.parent.get(node)
        if type(above) is ast.Expr:
            return
        if type(above) in (ast.Tuple, ast.List, ast.Set, ast.Dict):
            self.scan.read(name, self.where)
            return
        bound = None
        if type(above) is ast.Assign and len(above.targets) == 1:
            bound = above.targets[0]
        elif type(above) is ast.AnnAssign and above.value is node:
            bound = above.target
        if type(bound) is not ast.Name:
            self._use(node, name, owner=None)
            return
        local = self._function(node)
        scope: ast.AST = local or self.tree
        # A local name may be reused above its assignment; only later uses hold this prefix.
        first = above.lineno if local is not None else 0
        uses = [
            use for use in self._nodes(scope)
            if type(use) is ast.Name and use.id == bound.id and type(use.ctx) is ast.Load
            and use.lineno >= first
        ]
        if not uses:
            self.scan.declared.add(name)
        for use in uses:
            self._use(use, name, owner=None)

    def _use(self, node: ast.AST, name: str, *, owner: _Function | None) -> None:
        above = self.parent.get(node)
        reads = type(above) is ast.Compare or (
            type(above) in (ast.Tuple, ast.List, ast.Set)
        ) or (
            type(above) is ast.Call
            and node in above.args
            and (
                (type(above.func) is ast.Attribute and above.func.attr in _READ_METHODS)
                or (type(above.func) is ast.Name and above.func.id == "len")
            )
        )
        if reads:
            if owner is None:
                self.scan.read(name, self.where)
            return
        if owner is None:
            self.scan.write(name, self.where)
        call, argument = self._call_holding(node)
        context = self._function(node)
        self._judge(argument, context, name, "*", owner, skip=node)
        if call is None:
            return
        others: list[ast.AST] = [item for item in call.args if item is not argument]
        if _called(call) != "print":
            # ``print`` keywords choose the stream; any other call's keywords are values.
            others += [item.value for item in call.keywords if item is not argument]
        for other in others:
            self._judge(other, context, name, "*", owner, skip=node)
        if not _is_sink(call) and node in call.args:
            position = call.args.index(node)
            for function in self.functions.get(_called(call), ()):
                parameters = self._positional(function, call)
                if position < len(parameters):
                    self.helpers.append((function, parameters[position].arg))

    def _computed(self, node: ast.AST) -> None:
        found = False
        if type(node) is ast.JoinedStr:
            parts = node.values
            found = any(
                type(first) is ast.Constant and str(first.value).endswith("[")
                and type(middle) is ast.FormattedValue
                and type(last) is ast.Constant and str(last.value).startswith(("] ", "]{"))
                for first, middle, last in zip(parts, parts[1:], parts[2:], strict=False)
            )
        elif type(node) is ast.BinOp and type(node.op) is ast.Add:
            if type(self.parent.get(node)) is ast.BinOp:
                return
            parts = list(self._operands(node))
            found = any(
                type(first) is ast.Constant and type(first.value) is str
                and first.value.endswith("[")
                and type(last) is ast.Constant and type(last.value) is str
                and last.value.startswith(("] ", "]{"))
                for first, last in zip(parts, parts[2:], strict=False)
            )
        elif type(node) is ast.Constant and type(node.value) is str:
            found = re.match(r"\[(?:%s|\{\w*\})\](?: |\Z)", node.value) is not None
        if found:
            self.scan.computed[self.where] = self.scan.computed.get(self.where, 0) + 1

    def _operands(self, node: ast.AST) -> Iterator[ast.AST]:
        if type(node) is ast.BinOp and type(node.op) is ast.Add:
            yield from self._operands(node.left)
            yield from self._operands(node.right)
        else:
            yield node

    # -- values ---------------------------------------------------------------------

    def _doubt(self, marker: str, label: str, kind: str, node: ast.AST | str) -> None:
        self.scan.doubt(self.where, marker, _entry(label, kind, node))

    def _judge(
        self,
        node: ast.AST | None,
        context: _Function | None,
        marker: str,
        label: str,
        owner: _Function | None,
        *,
        skip: ast.AST | None = None,
        depth: int = 0,
    ) -> None:
        """Record ``node`` as unproven unless it is literal, numeric or a listed label."""
        if node is None or node is skip or type(node) is ast.Constant:
            return

        def again(item: ast.AST | None, name: str = label) -> None:
            self._judge(item, context, marker, name, owner, skip=skip, depth=depth)

        kind = type(node)
        if kind is ast.JoinedStr:
            if skip is not None and any(part is skip for part in node.values):
                for part in node.values:
                    if type(part) is ast.FormattedValue:
                        again(part.value)
            else:
                self._doubt(marker, label, "text", node)
        elif kind is ast.FormattedValue:
            again(node.value)
        elif kind is ast.BinOp:
            again(node.left)
            again(node.right)
        elif kind is ast.BoolOp:
            for value in node.values:
                again(value)
        elif kind is ast.UnaryOp:
            if type(node.op) is not ast.Not:
                again(node.operand)
        elif kind is ast.Compare:
            return
        elif kind is ast.IfExp:
            if not self._guarded(node.test, node.body, context, when=True):
                again(node.body)
            if not self._guarded(node.test, node.orelse, context, when=False):
                again(node.orelse)
        elif kind is ast.Dict:
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    again(value, label if label != "*" else "**")
                elif type(key) is ast.Constant and type(key.value) is str:
                    again(value, key.value if label in ("*", "**") else f"{label}.{key.value}")
                else:
                    self._doubt(marker, label, "key", key)
                    again(value)
        elif kind in (ast.List, ast.Tuple, ast.Set):
            for item in node.elts:
                again(item)
        elif kind is ast.Starred:
            again(node.value)
        elif kind is ast.Call:
            self._call(node, context, marker, label, owner, skip, depth)
        elif kind is ast.Attribute:
            named = node.attr == "__name__"
            self._doubt(marker, label, "class-name" if named else "attribute", node)
        elif kind is ast.Subscript:
            self._doubt(marker, label, "subscript", node)
        elif kind is ast.Name:
            self._name(node, context, marker, label, owner, depth)
        else:
            self._doubt(marker, label, "other", node)

    def _call(
        self,
        node: ast.Call,
        context: _Function | None,
        marker: str,
        label: str,
        owner: _Function | None,
        skip: ast.AST | None,
        depth: int,
    ) -> None:
        called = _called(node)
        if called in _NUMBER_CALLS:
            return
        if called in _PASS_THROUGH_CALLS:
            values = (
                [item.value for item in node.keywords] if called == "dict" and not node.args
                else node.args if called in ("min", "max") else node.args[:1]
            )
            for value in values:
                self._judge(value, context, marker, label, owner, skip=skip, depth=depth)
            return
        self._doubt(marker, label, "text" if called in _STRING_CALLS else "call", node)

    # -- guards ---------------------------------------------------------------------

    def _guarded(
        self, test: ast.AST, value: ast.AST, context: _Function | None, *, when: bool,
    ) -> bool:
        """Whether ``test`` being ``when`` shows ``value`` to be a listed label or a number.

        This is the one rule for a conditional expression, ``value if test else other``
        (``when`` true) and ``other if test else value`` (``when`` false):

        - ``value in V`` shows it when ``V`` is a literal vocabulary (``_vocabulary``);
          ``value == "constant"`` and ``type(value) is int`` (or ``bool``, ``float``) show it.
          Their negations (``not in``, ``!=``, ``is not``) show it when the test is false.
        - ``not T`` shows it exactly when ``T`` being the opposite does.
        - A conjunction that is true has every part true, so one part that shows it is
          enough. A disjunction that is true has only some part true, so EVERY part must
          show it. For a test that is false the two swap.
        - Nothing else shows it, and ``value`` must be a plain name, attribute or subscript:
          a call is evaluated again and may differ.

        The rule is local to the expression. A value checked by an earlier statement
        (``if x not in V: raise``) is not followed and stays unproven.
        """
        if any(type(node) is ast.Call for node in ast.walk(value)):
            return False
        kind = type(test)
        if kind is ast.UnaryOp and type(test.op) is ast.Not:
            return self._guarded(test.operand, value, context, when=not when)
        if kind is ast.BoolOp:
            shown = [self._guarded(part, value, context, when=when) for part in test.values]
            every = type(test.op) is (ast.Or if when else ast.And)
            return all(shown) if every else any(shown)
        if kind is not ast.Compare or len(test.ops) != 1:
            return False
        left, operator, right = test.left, type(test.ops[0]), test.comparators[0]
        if _text(left) == _text(value):
            if operator is (ast.In if when else ast.NotIn):
                return self._vocabulary(right, context)
            if operator is (ast.Eq if when else ast.NotEq):
                return type(right) is ast.Constant
        return (
            operator is (ast.Is if when else ast.IsNot)
            and type(left) is ast.Call
            and _called(left) == "type"
            and len(left.args) == 1
            and _text(left.args[0]) == _text(value)
            and type(right) is ast.Name
            and right.id in _NUMBER_TYPES
        )

    def _vocabulary(self, node: ast.AST | None, context: _Function | None, depth: int = 0) -> bool:
        """Whether ``node`` is a collection of constants that can be read from source.

        Accepted: a tuple, list, set or dict literal of constants; ``frozenset`` and its
        like around one; a union of two; ``get_args`` of a ``Literal`` alias; an ``Enum``
        class; and a name bound once, in this function or module or through a ``from``
        import of a module in this repository, to one of those. Anything else, including a
        parameter, an attribute and a computed collection, is not a vocabulary.
        """
        if node is None or depth > 6:
            return False
        kind = type(node)
        if kind in (ast.Set, ast.Tuple, ast.List):
            return all(self._is_constant(item, depth + 1) for item in node.elts)
        if kind is ast.Dict:
            return all(self._is_constant(key, depth + 1) for key in node.keys)
        if kind is ast.BinOp:
            left = self._vocabulary(node.left, context, depth + 1)
            if type(node.op) is ast.BitOr:
                return left and self._vocabulary(node.right, context, depth + 1)
            return left and type(node.op) in (ast.Sub, ast.BitAnd)
        if kind is ast.Call:
            called, arguments = _called(node), node.args
            if node.keywords or len(arguments) > 1:
                return False
            if called in ("frozenset", "set", "tuple", "list", "sorted"):
                return not arguments or self._vocabulary(arguments[0], context, depth + 1)
            return called == "get_args" and len(arguments) == 1 and self._literal(arguments[0])
        if kind is not ast.Name:
            return False
        scope: ast.AST | None = context
        while scope is not None:
            bound = self._plainly_bound(scope, node.id)
            if bound is None:
                return False
            if bound:
                return all(self._vocabulary(value, scope, depth + 1) for value in bound)
            scope = self._function(scope)
        return self._module_vocabulary(node.id, depth + 1)

    def _is_constant(self, node: ast.AST | None, depth: int) -> bool:
        """A constant, a choice between constants, or a module-level name for one."""
        if node is None or depth > 6:
            return False
        if type(node) is ast.IfExp:
            return self._is_constant(node.body, depth + 1) and self._is_constant(
                node.orelse, depth + 1,
            )
        if type(node) is ast.Name:
            values = self._module_values(node.id)
            return bool(values) and all(self._is_constant(value, depth + 1) for value in values)
        return type(node) is ast.Constant

    def _plainly_bound(self, scope: ast.AST, name: str) -> list[ast.AST] | None:
        """Values assigned to ``name`` in a function; None if it is bound any other way."""
        assert type(scope) in _FUNCTIONS
        if self._parameter(scope, name) is not None:
            return None
        values: list[ast.AST] = []
        for node in self._nodes(scope):
            kind = type(node)
            if kind is ast.Assign and any(
                type(item) is ast.Name and item.id == name
                for target in node.targets for item in ast.walk(target)
            ):
                if not all(type(target) is ast.Name for target in node.targets):
                    return None
                values.append(node.value)
            elif kind is ast.AnnAssign and _text(node.target) == name:
                if node.value is None:
                    return None
                values.append(node.value)
            elif self._binds_otherwise(node, name):
                return None
        return values

    def _binds_otherwise(self, node: ast.AST, name: str) -> bool:
        """Whether ``node`` binds ``name`` in any way but a plain assignment to it."""
        kind = type(node)
        if kind is ast.Name and node.id == name and type(node.ctx) is not ast.Load:
            above = self.parent.get(node)
            plain = (type(above) is ast.Assign and node in above.targets) or (
                type(above) is ast.AnnAssign and above.target is node
            )
            return not plain
        if kind in (ast.Global, ast.Nonlocal):
            return name in node.names
        if kind is ast.ExceptHandler:
            return node.name == name
        return kind in (ast.Import, ast.ImportFrom) and any(
            (alias.asname or alias.name.partition(".")[0]) == name for alias in node.names
        )

    def _literal(self, node: ast.AST) -> bool:
        """Whether ``node`` names a module-level ``Literal[...]`` alias, or is one."""
        if type(node) is ast.Subscript:
            return _text(node.value) in ("Literal", "typing.Literal")
        return type(node) is ast.Name and any(
            self._literal(value) for value in self._module_values(node.id) or ()
        )

    def _module_values(self, name: str) -> list[ast.AST] | None:
        """Values assigned to ``name`` at module level; None if bound in any other way."""
        values: list[ast.AST] = []
        for statement in self.tree.body:
            kind = type(statement)
            if kind is ast.Assign and any(
                type(item) is ast.Name and item.id == name
                for target in statement.targets for item in ast.walk(target)
            ):
                if not all(type(target) is ast.Name for target in statement.targets):
                    return None
                values.append(statement.value)
            elif kind is ast.AnnAssign and _text(statement.target) == name:
                if statement.value is None:
                    return None
                values.append(statement.value)
            elif kind is ast.AugAssign and _text(statement.target) == name:
                return None
        return values

    def _module_vocabulary(self, name: str, depth: int) -> bool:
        if depth > 6:
            return False
        for statement in self.tree.body:
            if type(statement) is ast.ClassDef and statement.name == name:
                return any(_text(base).endswith("Enum") for base in statement.bases)
        values = self._module_values(name)
        if values is None:
            return False
        if values:
            return all(self._vocabulary(value, None, depth + 1) for value in values)
        for node in self.tree.body:
            if type(node) is not ast.ImportFrom:
                continue
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    other = _load(_module_file(self.where, node.module, node.level))
                    return other is not None and other._module_vocabulary(alias.name, depth + 1)
        return False

    # -- names ----------------------------------------------------------------------

    def _name(
        self,
        node: ast.Name,
        context: _Function | None,
        marker: str,
        label: str,
        owner: _Function | None,
        depth: int,
    ) -> None:
        scope: ast.AST | None = context
        while scope is not None:
            key = (id(scope), hash((marker, label)), node.id)
            if key in self.judged:
                return
            if self._bindings(scope, node.id, marker, label, owner, depth, key):
                return
            scope = self._function(scope)
        key = (id(self.tree), hash((marker, label)), node.id)
        if key in self.judged:
            return
        self.judged.add(key)
        found = False
        for statement in self.tree.body:
            targets = (
                statement.targets if type(statement) is ast.Assign
                else [statement.target] if type(statement) is ast.AnnAssign
                else []
            )
            if any(type(target) is ast.Name and target.id == node.id for target in targets):
                found = True
                self._judge(statement.value, None, marker, label, owner, depth=depth)
        if not found:
            self._doubt(marker, label, "name", node)

    def _bindings(
        self,
        scope: ast.AST,
        name: str,
        marker: str,
        label: str,
        owner: _Function | None,
        depth: int,
        key: tuple[int, int, str],
    ) -> bool:
        """Judge every value bound to ``name`` in ``scope``; False when it binds none."""
        assert type(scope) in _FUNCTIONS
        context: _Function = scope
        found = False

        def judge(value: ast.AST | None, part: str = label) -> None:
            self._judge(value, context, marker, part, owner, depth=depth)

        def names(target: ast.AST) -> bool:
            return any(type(item) is ast.Name and item.id == name for item in ast.walk(target))

        def member(part: ast.AST) -> str:
            if type(part) is ast.Constant and type(part.value) is str:
                return part.value if label in ("*", "**") else f"{label}.{part.value}"
            return label

        self.judged.add(key)
        for node in self._nodes(scope):
            kind = type(node)
            if kind is ast.Assign:
                for target in node.targets:
                    if type(target) is ast.Name and target.id == name:
                        found = True
                        judge(node.value)
                    elif type(target) is ast.Subscript and _text(target.value) == name:
                        found = True
                        judge(node.value, member(target.slice))
                    elif type(target) in (ast.Tuple, ast.List) and names(target):
                        found = True
                        self._doubt(marker, label, "bound", name)
            elif kind in (ast.AnnAssign, ast.AugAssign, ast.NamedExpr):
                if type(node.target) is ast.Name and node.target.id == name:
                    found = True
                    if kind is ast.AnnAssign and self._typed(node.annotation):
                        continue
                    judge(node.value)
            elif kind in (ast.For, ast.AsyncFor, ast.comprehension):
                if names(node.target):
                    found = True
                    self._doubt(marker, label, "bound", name)
            elif kind is ast.withitem:
                if node.optional_vars is not None and names(node.optional_vars):
                    found = True
                    self._doubt(marker, label, "bound", name)
            elif kind is ast.ExceptHandler:
                if node.name == name:
                    found = True
                    self._doubt(marker, label, "bound", name)
            elif (
                kind is ast.Call
                and type(node.func) is ast.Attribute
                and node.func.attr in ("update", "setdefault", "append", "extend", "add")
                and _text(node.func.value) == name
            ):
                found = True
                for item in node.args:
                    judge(item)
                for item in node.keywords:
                    judge(item.value, member(ast.Constant(item.arg)) if item.arg else label)
        parameter = self._parameter(context, name)
        if parameter is not None:
            found = True
            self._argument(context, parameter, marker, label, owner, depth)
        return found

    def _typed(self, annotation: ast.AST | None) -> bool:
        """True when an annotation names a number, a flag or a ``Literal`` vocabulary."""
        if annotation is None:
            return False
        if type(annotation) is ast.Constant and type(annotation.value) is str:
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return False
        if type(annotation) is ast.BinOp and type(annotation.op) is ast.BitOr:
            sides = [annotation.left, annotation.right]
            rest = [side for side in sides if _text(side) != "None"]
            return bool(rest) and all(self._typed(side) for side in rest)
        if type(annotation) is ast.Name:
            if annotation.id in _NUMBER_TYPES:
                return True
            for statement in self.tree.body:
                value = getattr(statement, "value", None)
                targets = getattr(statement, "targets", None) or [getattr(statement, "target", 0)]
                if any(type(item) is ast.Name and item.id == annotation.id for item in targets):
                    return self._typed(value)
            return False
        return type(annotation) is ast.Subscript and _text(annotation.value) in (
            "Literal", "typing.Literal",
        )

    @staticmethod
    def _parameter(function: _Function, name: str) -> ast.arg | None:
        every = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        return next((item for item in every if item.arg == name), None)

    @staticmethod
    def _positional(function: _Function, call: ast.Call) -> list[ast.arg]:
        parameters = [*function.args.posonlyargs, *function.args.args]
        if parameters and parameters[0].arg in ("self", "cls") and type(call.func) is ast.Attribute:
            return parameters[1:]
        return parameters

    def _argument(
        self,
        function: _Function,
        parameter: ast.arg,
        marker: str,
        label: str,
        owner: _Function | None,
        depth: int,
    ) -> None:
        if function is owner or self._typed(parameter.annotation):
            return
        calls = self.calls.get(function.name, [])
        if not calls or depth >= 3 or len(self.functions[function.name]) != 1:
            self._doubt(marker, label, "argument", parameter.arg)
            return
        for call in calls:
            if any(type(item) is ast.Starred for item in call.args) or any(
                item.arg is None for item in call.keywords
            ):
                self._doubt(marker, label, "argument", parameter.arg)
                continue
            positional = self._positional(function, call)
            value = next((item.value for item in call.keywords if item.arg == parameter.arg), None)
            if value is None and parameter in positional:
                position = positional.index(parameter)
                value = call.args[position] if position < len(call.args) else None
            if value is not None:
                self._judge(
                    value, self._function(call), marker, label, owner, depth=depth + 1,
                )


def _module_file(where: str, module: str | None, level: int) -> str | None:
    """The repository file a ``from`` import names, when it can be read from source."""
    parts = module.split(".") if module else []
    if level:
        base = Path(where).parent
        for _ in range(level - 1):
            base = base.parent
    elif parts[:1] == ["hermes_realtime"]:
        base = Path("src")
    elif parts[:1] == ["scripts"]:
        base = Path()
    else:
        return None
    target = base.joinpath(*parts)
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if parts and (ROOT / candidate).is_file():
            return candidate.as_posix()
    return None


@cache
def _load(where: str | None) -> _Module | None:
    """Another module of the repository, parsed only to resolve a name it defines."""
    if where is None:
        return None
    tree = ast.parse((ROOT / where).read_text(encoding="utf-8"), filename=where)
    return _Module(where, tree, Scan())


# -- the browser ------------------------------------------------------------------------

_WEB_PREFIX = re.compile(r"([`'\"])\[(" + _NAME + r")\](?= |\1)")
_WEB_LITERAL = re.compile(r"""(?:"[^"$`]*"|'[^'$`]*'|-?\d+(?:\.\d+)?|true|false|null)\Z""")


def _balanced(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return len(text) - 1


def _web_members(body: str) -> Iterator[str]:
    depth, start = 0, 0
    for index, character in enumerate(body + ","):
        if character in "({[":
            depth += 1
        elif character in ")}]":
            depth -= 1
        elif character == "," and depth == 0:
            if body[start:index].strip():
                yield " ".join(body[start:index].split())
            start = index + 1


def _scan_web(where: str, text: str, scan: Scan) -> None:
    for match in _WEB_PREFIX.finditer(text):
        quote, name = match.group(1), match.group(2)
        opening = text.rfind("(", 0, match.start())
        written = re.search(r"console\.\w+\s*\(\s*\Z", text[: match.start()]) is not None
        if not written or opening < 0:
            scan.read(name, where)
            continue
        scan.write(name, where)
        closing = text.find(quote, match.end())
        if quote != "`":
            if text[closing + 1 :].lstrip()[:1] != ")":
                scan.doubt(where, name, _entry("*", "text", "joined to another value"))
            continue
        template = text[match.end() : closing]
        cursor = 0
        while (at := template.find("${", cursor)) >= 0:
            end = _balanced(template, at + 1, "{", "}")
            inner = template[at + 2 : end].strip()
            cursor = end + 1
            argument = re.fullmatch(r"JSON\.stringify\((.*)\)", inner, re.DOTALL)
            value = argument.group(1).strip() if argument else None
            if value is None or not (value.startswith("{") and value.endswith("}")):
                scan.doubt(where, name, _entry("*", "value", " ".join(inner.split())))
                continue
            for member in _web_members(value[1:-1]):
                key, colon, item = member.partition(":")
                if not colon:
                    scan.doubt(where, name, _entry(key.strip(), "value", key.strip()))
                elif _WEB_LITERAL.match(item.strip()) is None:
                    scan.doubt(where, name, _entry(key.strip(), "value", item.strip()))


# -- measurement ------------------------------------------------------------------------


def _generic_readers(where: str, nodes: list[ast.AST], scan: Scan) -> None:
    """A regular expression that matches any ``[name] {json}`` line reads every marker."""
    for node in nodes:
        if (
            type(node) is ast.Constant
            and type(node.value) is str
            and re.search(r"\\\[.*\\\] ?\(?\\\{", node.value) is not None
        ):
            scan.read(_GENERIC, where)


def scan_python(where: str, source: str, scan: Scan) -> None:
    tree = ast.parse(source, filename=where)
    module = _Module(where, tree, scan)
    module.run()
    _generic_readers(where, module.nodes, scan)


@cache
def measured() -> Scan:
    scan = Scan()
    for root in PYTHON_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if _CANDIDATE.search(source) is not None:
                scan_python(path.relative_to(ROOT).as_posix(), source, scan)
    for path in sorted((ROOT / WEB_ROOT).rglob("*.ts")):
        _scan_web(path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8"), scan)
    return scan


def _unproven(scan: Scan) -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        where: {marker: tuple(sorted(entries)) for marker, entries in sorted(markers.items())}
        for where, markers in sorted(scan.unproven.items())
    }


def _never_written(scan: Scan) -> set[str]:
    return (set(scan.reads) | scan.declared) - set(scan.writes) - {_GENERIC}


def _difference(measured_table: dict, listed: dict) -> tuple[list, list]:
    new = sorted(
        (key, value) for key, value in measured_table.items() if listed.get(key) != value
    )
    stale = sorted(
        (key, value) for key, value in listed.items() if measured_table.get(key) != value
    )
    return new, stale


def _assert_equal(measured_table: dict, listed: dict, what: str) -> None:
    new, stale = _difference(measured_table, listed)
    assert not new, f"{what} measured in source and not listed as measured: {new}"
    assert not stale, f"{what} listed and not measured in source as listed: {stale}"


# -- the ratchet ------------------------------------------------------------------------


def test_every_marker_name_and_its_emit_sites_are_listed() -> None:
    _assert_equal(measured().writes, tables.EMIT_SITES, "marker emit sites")


def test_every_value_that_is_not_shown_to_be_content_free_is_listed() -> None:
    _assert_equal(_unproven(measured()), tables.UNPROVEN_VALUES, "unproven marker values")


def test_every_hand_written_reader_is_listed() -> None:
    _assert_equal(measured().reads, tables.READERS, "hand-written marker readers")


def test_every_prefix_assembled_from_parts_is_listed() -> None:
    _assert_equal(measured().computed, tables.COMPUTED_PREFIXES, "assembled marker prefixes")


def test_every_name_that_is_read_and_never_written_is_listed() -> None:
    assert _never_written(measured()) == tables.NEVER_WRITTEN


def test_the_known_free_strings_are_among_the_unproven_values() -> None:
    listed = {
        (where, marker, entry)
        for where, markers in tables.UNPROVEN_VALUES.items()
        for marker, entries in markers.items()
        for entry in entries
    }
    missing = sorted(set(tables.KNOWN_FREE_STRINGS) - listed)
    assert not missing, f"free strings recorded as known and no longer measured: {missing}"
    assert all(type(why) is str and why for why in tables.KNOWN_FREE_STRINGS.values())


# -- what the scan sees, on small sources -----------------------------------------------


def _scan(source: str, where: str = "m.py") -> Scan:
    scan = Scan()
    scan_python(where, source, scan)
    return scan


def _guard(test: str, *, preamble: str = "", where: str = "m.py") -> tuple[str, ...]:
    """The unproven entries of ``{'operation': <test>}`` printed from a small module."""
    scan = _scan(
        "import json\n"
        "_OPERATIONS = frozenset({'pull', 'start'})\n"
        "NPM = 'npm.cmd' if True else 'npm'\n"
        "_TOOLS = ('git', NPM)\n"
        f"{preamble}"
        "def f(first, arguments, other):\n"
        "    local = {'pull', 'start'}\n"
        "    rebuilt = set(arguments)\n"
        f"    operation = {test}\n"
        "    print('[some-marker] ' + json.dumps({'operation': operation}))\n",
        where,
    )
    return _unproven(scan).get(where, {}).get("some-marker", ())


_FIRST = ("operation: argument: first",)


@pytest.mark.parametrize(
    "test",
    [
        "first if first in _OPERATIONS else 'other'",
        "first if type(first) is str and first in _OPERATIONS else 'other'",
        "first if first in {'pull', 'start'} else 'other'",
        "first if first in ('pull', 'start') else 'other'",
        "first if first in local else 'other'",
        "first if first in _TOOLS else 'other'",
        "first if first in _OPERATIONS | {'rm'} else 'other'",
        "first if first in _OPERATIONS or first in local else 'other'",
        "first if first == 'pull' else 'other'",
        "first if type(first) is int else None",
        "first if not (first not in _OPERATIONS) else 'other'",
        "first if not (type(first) is not str or first not in _OPERATIONS) else 'other'",
        "'other' if first not in _OPERATIONS else first",
        "'other' if not (first in _OPERATIONS) else first",
        "'other' if type(first) is not str or first not in _OPERATIONS else first",
        "first if type(first) is str and (first in _OPERATIONS or first == 'rm') else 'other'",
    ],
)
def test_a_guard_that_bounds_the_value_is_accepted(test: str) -> None:
    assert _guard(test) == ()


@pytest.mark.parametrize(
    "test",
    [
        # A disjunction is true when only one part is: every part must bound the value.
        "first if type(first) is str or first in _OPERATIONS else 'other'",
        "first if first in _OPERATIONS or other else 'other'",
        "first if type(first) is str and (first in _OPERATIONS or first) else 'other'",
        # A negated membership is the complement of a vocabulary.
        "first if not (first in _OPERATIONS) else 'other'",
        "first if first not in _OPERATIONS else 'other'",
        "'other' if first in _OPERATIONS else first",
        "'other' if type(first) is not str and first not in _OPERATIONS else first",
        "first if first != 'pull' else 'other'",
        # A named collection must resolve to a literal vocabulary.
        "first if first in arguments else 'other'",
        "first if type(first) is str and first in arguments else 'other'",
        "first if first in rebuilt else 'other'",
        "first if first in set(arguments) else 'other'",
        "first if first in other.allowed else 'other'",
        "first if first in _UNDEFINED else 'other'",
        "first if first in _OPERATIONS | set(arguments) else 'other'",
        # The guard must be about the value that is written.
        "first if other in _OPERATIONS else 'other'",
        "first if type(first) is str else 'other'",
    ],
)
def test_a_guard_that_does_not_bound_the_value_leaves_it_unproven(test: str) -> None:
    assert _guard(test) == _FIRST


def test_a_guarded_call_is_unproven_because_it_is_evaluated_again() -> None:
    assert _guard("str(first) if str(first) in _OPERATIONS else 'other'") == (
        "operation: text: str(first)",
    )


def test_a_rebound_or_computed_named_collection_is_not_a_vocabulary() -> None:
    rebound = "_SEEN = {'pull'}\n_SEEN = set(open('x'))\n"
    assert _guard("first if first in _SEEN else 'other'", preamble=rebound) == _FIRST
    grown = "_SEEN = {'pull'}\n_SEEN |= {'start'}\n"
    assert _guard("first if first in _SEEN else 'other'", preamble=grown) == _FIRST
    unpacked = "_SEEN, _REST = {'pull'}, {'start'}\n"
    assert _guard("first if first in _SEEN else 'other'", preamble=unpacked) == _FIRST
    mixed = "_SEEN = ('pull', other_module.NAME)\n"
    assert _guard("first if first in _SEEN else 'other'", preamble=mixed) == _FIRST


def test_a_named_collection_is_followed_through_an_alias_an_enum_and_an_import() -> None:
    alias = "from typing import Literal, get_args\nKind = Literal['pull', 'start']\n"
    assert _guard("first if first in get_args(Kind) else 'other'", preamble=alias) == ()
    assert _guard("first if first in get_args(Missing) else 'other'", preamble=alias) == _FIRST
    enum = "from enum import Enum\nclass Kind(Enum):\n    PULL = 'pull'\n"
    assert _guard("first if first in Kind else 'other'", preamble=enum) == ()
    plain = "class Kind:\n    PULL = 'pull'\n"
    assert _guard("first if first in Kind else 'other'", preamble=plain) == _FIRST
    # A real module of this repository: a frozenset of constants, and a name that is not one.
    imported = "from hermes_realtime.protocol.events import VOICE_REFUSAL_CATEGORIES as LISTED\n"
    script = "scripts/_synthetic.py"
    assert _guard("first if first in LISTED else 'other'", preamble=imported, where=script) == ()
    function = "from hermes_realtime.protocol.events import parse_event as LISTED\n"
    assert _guard(
        "first if first in LISTED else 'other'", preamble=function, where=script,
    ) == _FIRST
    foreign = "from os import environ as LISTED\n"
    assert _guard(
        "first if first in LISTED else 'other'", preamble=foreign, where=script,
    ) == _FIRST


def test_a_literal_line_is_a_write_with_nothing_unproven() -> None:
    scan = _scan(
        "import json\n"
        "def f(ok):\n"
        "    category = 'a' if ok else 'b'\n"
        "    print('[some-marker] ' + json.dumps({'category': category, 'count': len(ok)}))\n"
        "    print('[other-marker] {\"category\":\"fixed\"}')\n"
    )
    assert scan.writes == {"some-marker": {"m.py": 1}, "other-marker": {"m.py": 1}}
    assert scan.unproven == {} and scan.reads == {}


def test_a_computed_string_is_unproven_by_field() -> None:
    scan = _scan(
        "import json\n"
        "def f(error, thing):\n"
        "    print('[some-marker] ' + json.dumps({'failure': type(error).__name__,\n"
        "        'category': str(error), 'model': thing.model}))\n"
    )
    assert _unproven(scan) == {"m.py": {"some-marker": (
        "category: text: str(error)",
        "failure: class-name: type(error).__name__",
        "model: attribute: thing.model",
    )}}


def test_a_value_is_followed_through_a_helper_and_its_callers() -> None:
    scan = _scan(
        "import json\n"
        "_MARKER = '[some-marker] '\n"
        "def _marker(evidence):\n"
        "    print(_MARKER + json.dumps(evidence))\n"
        "def good():\n"
        "    _marker({'refusal': 'capacity', 'version': 1})\n"
        "def bad(reply):\n"
        "    _marker({'refusal': reply.category})\n"
    )
    assert scan.writes == {"some-marker": {"m.py": 1}}
    assert _unproven(scan) == {
        "m.py": {"some-marker": ("refusal: attribute: reply.category",)}
    }


def test_a_helper_that_takes_the_prefix_is_judged_where_it_adds_values() -> None:
    scan = _scan(
        "import json\n"
        "def _marker(prefix, evidence):\n"
        "    print(prefix + json.dumps(evidence | {'cause': evidence.kind}))\n"
        "def f():\n"
        "    _marker('[some-marker] ', {'version': 1})\n"
    )
    assert scan.writes == {"some-marker": {"m.py": 1}}
    assert _unproven(scan) == {"m.py": {"*_marker": ("cause: attribute: evidence.kind",)}}


def test_a_typed_or_guarded_value_is_accepted() -> None:
    scan = _scan(
        "import json\n"
        "from typing import Literal\n"
        "Category = Literal['a', 'b']\n"
        "def f(category: Category, count: int, raw, stage):\n"
        "    print('[some-marker] ' + json.dumps({'category': category, 'count': count,\n"
        "        'bytes': raw if type(raw) is int else None,\n"
        "        'stage': stage if stage in {'x', 'y'} else 'unknown'}))\n"
    )
    assert scan.unproven == {}


def test_readers_tables_and_declarations_are_not_writes() -> None:
    scan = _scan(
        "_UNUSED = '[unused-marker] '\n"
        "_TABLE = {'[table-marker] ': 'kind'}\n"
        "def f(line):\n"
        "    if line.startswith('[read-marker] '):\n"
        "        return line[len('[read-marker] '):]\n"
    )
    assert scan.writes == {}
    assert scan.reads == {"table-marker": {"m.py": 1}, "read-marker": {"m.py": 2}}
    assert scan.declared == {"unused-marker"}
    assert _never_written(scan) == {"unused-marker", "table-marker", "read-marker"}


def test_assembled_prefixes_are_counted_and_a_joined_one_is_not_seen() -> None:
    seen = _scan(
        "def f(name, value):\n"
        "    a = f'[{name}] ' + value\n"
        "    b = '[' + name + '] ' + value\n"
        "    c = '[%s] ' % name\n"
        "    return a, b, c\n"
    )
    assert seen.computed == {"m.py": 3} and seen.writes == {}
    unseen = _scan("def f(name):\n    return ''.join(['[', name, ']', ' '])\n")
    assert unseen.computed == {} and unseen.writes == {}


def test_a_generic_marker_expression_is_a_reader_of_every_marker() -> None:
    scan = _scan("import re\n_LINE = re.compile(r'\\[[a-z][a-z0-9-]{0,47}\\] \\{.*\\}')\n")
    assert scan.reads == {_GENERIC: {"m.py": 1}}


def test_browser_lines_are_measured() -> None:
    scan = Scan()
    _scan_web(
        "w.ts",
        "console.info('[fixed-line] {\"category\":\"a\"}');\n"
        "console.info(`[made-line] ${JSON.stringify({ count: 1, category: \"b\" })}`);\n"
        "console.info(`[open-line] ${JSON.stringify({ category, stage: where.stage })}`);\n"
        "console.info(`[whole-line] ${JSON.stringify(refused)}`);\n",
        scan,
    )
    assert set(scan.writes) == {"fixed-line", "made-line", "open-line", "whole-line"}
    assert _unproven(scan) == {"w.ts": {
        "open-line": ("category: value: category", "stage: value: where.stage"),
        "whole-line": ("*: value: JSON.stringify(refused)",),
    }}


# -- printing the tables ----------------------------------------------------------------


def _render() -> str:
    scan = measured()
    lines = []
    quoted = json.dumps

    def table(name: str, kind: str, rows: dict) -> None:
        lines.append(f"{name}: {kind} = {{")
        for key, value in sorted(rows.items()):
            if type(value) is dict:
                lines.append(f"    {quoted(key)}: {{")
                for inner, item in sorted(value.items()):
                    if type(item) is tuple:
                        lines.append(f"        {quoted(inner)}: (")
                        lines.extend(f"            {quoted(entry)}," for entry in item)
                        lines.append("        ),")
                    else:
                        lines.append(f"        {quoted(inner)}: {item!r},")
                lines.append("    },")
            else:
                lines.append(f"    {quoted(key)}: {value!r},")
        lines.append("}\n")

    table("EMIT_SITES", "dict[str, dict[str, int]]", scan.writes)
    table("UNPROVEN_VALUES", "dict[str, dict[str, tuple[str, ...]]]", _unproven(scan))
    table("READERS", "dict[str, dict[str, int]]", scan.reads)
    table("COMPUTED_PREFIXES", "dict[str, int]", scan.computed)
    lines.append("NEVER_WRITTEN: set[str] = {")
    lines.extend(f"    {quoted(name)}," for name in sorted(_never_written(scan)))
    lines.append("}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.stdout.write(_render())
