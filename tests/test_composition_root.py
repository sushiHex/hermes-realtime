"""What the composition roots contain beyond construction, as a ratchet.

The rule for a composition root, and its only statement: a composition root constructs
objects and hands them to each other. It may call constructors and factories, pass
objects and bound methods as arguments, choose what to construct from an option, and
read options. It may not define a nested function or class, declare ``nonlocal``,
``await``, use ``try``, or hold a lambda that does more than forward its arguments.
Whatever needs a branch of its own, a loop, a second statement or a field is a
component: it gets a name, a home in a tier and a test.

``BUILDERS`` freezes how far each root is from that rule today. The test fails when a
number grows and when it falls without being lowered here, so the numbers can only
shrink and always describe the source.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "hermes_realtime"

# What is counted, per root. Every count covers the whole body of the root, nested
# scopes included, except the two marked "own": those look only at code the root itself
# executes when it is called.
#
#   functions          nested ``def`` and ``async def``, at any depth
#   classes            nested ``class``, at any depth
#   lambdas            every ``lambda``, at any depth. A lambda that only forwards its
#                      arguments is allowed by the rule; telling the two apart is a
#                      reviewer's judgement, so all are counted and none may be added
#   nonlocal_declarations  names listed in ``nonlocal`` statements, once per declaring
#                      scope: each is one writer of state shared between closures
#   own_awaits         ``await`` expressions outside every nested scope
#   own_tries          ``try`` statements outside every nested scope
#   nested_statements  statements inside nested functions and classes: how much
#                      behaviour the closures hold
#
# Lines are deliberately not counted. A line count moves on a comment, a blank line, a
# reflowed call or one more keyword argument to a constructor: edits the rule permits.
# A number that fails on harmless edits is bumped without being read. Statements inside
# nested scopes measure the same thing the line count was meant to (behaviour living
# in a root) and move only when that behaviour changes. Statements the root executes
# itself are not counted either: constructing one more collaborator is what a root is for.
BUILDERS: dict[tuple[str, str], dict[str, int]] = {
    ("host_launcher.py", "build_local_host_launcher"): {
        "functions": 35,
        "classes": 1,
        "lambdas": 17,
        "nonlocal_declarations": 11,
        "own_awaits": 0,
        "own_tries": 7,
        "nested_statements": 257,
    },
    ("launcher.py", "build_local_conversation_launcher"): {
        "functions": 4,
        "classes": 0,
        "lambdas": 1,
        "nonlocal_declarations": 0,
        "own_awaits": 0,
        "own_tries": 0,
        "nested_statements": 9,
    },
}

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _walk(node: ast.AST, scope: tuple[int, ...]) -> Iterator[tuple[ast.AST, tuple[int, ...]]]:
    """Yield every node under ``node`` with the nested scopes that enclose it.

    ``scope`` is empty for code the root executes itself. A nested function, class or
    lambda is yielded in the scope that defines it; its body is one scope deeper.
    """
    for child in ast.iter_child_nodes(node):
        yield child, scope
        yield from _walk(child, (*scope, id(child)) if type(child) in _SCOPES else scope)


def measure(root: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, int]:
    """Count what ``root`` holds that the rule for a composition root forbids."""
    nodes = list(_walk(root, ()))
    return {
        "functions": sum(type(node) in _FUNCTIONS for node, _ in nodes),
        "classes": sum(type(node) is ast.ClassDef for node, _ in nodes),
        "lambdas": sum(type(node) is ast.Lambda for node, _ in nodes),
        "nonlocal_declarations": sum(
            len(node.names) for node, _ in nodes if type(node) is ast.Nonlocal
        ),
        "own_awaits": sum(type(node) is ast.Await and not scope for node, scope in nodes),
        "own_tries": sum(
            type(node) in (ast.Try, ast.TryStar) and not scope for node, scope in nodes
        ),
        "nested_statements": sum(
            isinstance(node, ast.stmt) and bool(scope) for node, scope in nodes
        ),
    }


def _root(file: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    path = SOURCE / file
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = [node for node in tree.body if type(node) in _FUNCTIONS and node.name == name]
    assert len(found) == 1, f"{file} must define exactly one module-level {name}"
    return found[0]


def test_measurement_counts_each_construct_the_rule_forbids() -> None:
    sample = ast.parse(
        "async def root(option):\n"
        "    held = {}\n"
        "    flag = False\n"
        "    def closure(value):\n"
        "        nonlocal flag\n"
        "        flag = True\n"
        "        async def inner():\n"
        "            await value\n"
        "            try:\n"
        "                pass\n"
        "            finally:\n"
        "                pass\n"
        "        return inner\n"
        "    class Owner:\n"
        "        close = lambda self: held.clear()\n"
        "    try:\n"
        "        await option\n"
        "    except* ValueError:\n"
        "        pass\n"
        "    return closure, Owner, lambda: flag\n"
    ).body[0]
    assert type(sample) is ast.AsyncFunctionDef

    assert measure(sample) == {
        "functions": 2,
        "classes": 1,
        "lambdas": 2,
        "nonlocal_declarations": 1,
        "own_awaits": 1,
        "own_tries": 1,
        # nonlocal, flag = True, async def inner, return inner; inner's Expr, try and two
        # passes; the class's one assignment.
        "nested_statements": 9,
    }


def test_composition_roots_hold_no_more_than_recorded() -> None:
    measured = {key: measure(_root(*key)) for key in BUILDERS}
    grown: list[str] = []
    shrunk: list[str] = []
    for (file, name), recorded in BUILDERS.items():
        for what, limit in recorded.items():
            found = measured[file, name][what]
            entry = f"{file} {name} {what}: recorded {limit}, found {found}"
            if found > limit:
                grown.append(entry)
            elif found < limit:
                shrunk.append(entry)

    assert not grown, (
        "a composition root gained what the rule at the top of this file forbids. Move "
        "the new behaviour into a named component in the package that owns it and have "
        f"the root construct it; do not raise the number: {grown}"
    )
    assert not shrunk, (
        "a composition root now holds less than recorded. Lower these numbers in BUILDERS "
        f"in the same change, so the next addition is caught: {shrunk}"
    )
