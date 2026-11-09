"""Every reviewed launch on the Windows qualification caller chains stays windowless.

``test_qualification_windowless`` observes that the shared flags leave the real
virtual-environment redirector child without a console. This inventory proves
each launch site on those chains actually selects such flags: dropping
``CREATE_NO_WINDOW``, adding ``DETACHED_PROCESS``/``CREATE_NEW_CONSOLE``, or
adding an unreviewed launch fails here. Flags are resolved for the Windows branch
only; an expression this resolver cannot evaluate is a failure, not a pass.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CREATE_NO_WINDOW = 0x08000000
_CONSOLE_FLAGS = 0x00000008 | 0x00000010  # DETACHED_PROCESS | CREATE_NEW_CONSOLE
_SUBPROCESS_FLAGS = {
    "CREATE_NO_WINDOW": _CREATE_NO_WINDOW,
    "CREATE_NEW_PROCESS_GROUP": 0x00000200,
    "CREATE_BREAKAWAY_FROM_JOB": 0x01000000,
    "DETACHED_PROCESS": 0x00000008,
    "CREATE_NEW_CONSOLE": 0x00000010,
}
# Reviewed launches: guardian -> controller (#262/#266), the controllers it runs,
# and the preview owner that reuses the same shared flags.
_CHAIN = {
    "scripts/qualification_launch.py": {"run_owned_controller": 1, "_owned_controller": 1},
    "scripts/evaluate_natural_work_routing.py": {"_codex_version": 1, "_git_value": 1},
    "scripts/qualification_cpu_readiness.py": {"_controller": 1},
    "src/hermes_realtime/providers/codex_app_server.py": {
        "SubprocessCodexJsonLineTransport.create": 1
    },
    "scripts/preview_owner_launch.py": {"_probe_interpreter": 1, "launch_preview_owner": 1},
}
_SUBPROCESS_CALLS = {"run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}
_ANY_RECEIVER = {"Popen", "create_subprocess_exec", "create_subprocess_shell"}
_OS_CALLS = {"system", "startfile", "popen", "posix_spawn", "posix_spawnp"}
_LAUNCH_NAMES = _SUBPROCESS_CALLS | _ANY_RECEIVER | _OS_CALLS


def _module(relative: str) -> ast.Module:
    return ast.parse((_ROOT / relative).read_bytes(), relative)


def _is_launch(call: ast.Call) -> bool:
    function = call.func
    if type(function) is not ast.Attribute:
        return False
    receiver = function.value.id if type(function.value) is ast.Name else None
    return (
        function.attr in _ANY_RECEIVER
        or function.attr == "create_process_suspended"
        or (receiver == "subprocess" and function.attr in _SUBPROCESS_CALLS)
        or (
            receiver == "os"
            and (function.attr in _OS_CALLS or function.attr.startswith(("spawn", "exec")))
        )
    )


def _launches(tree: ast.Module) -> list[tuple[str, ast.Call, list[ast.AST]]]:
    found: list[tuple[str, ast.Call, list[ast.AST]]] = []

    def visit(node: ast.AST, names: tuple[str, ...], scopes: list[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            if type(child) in (ast.FunctionDef, ast.AsyncFunctionDef):
                visit(child, (*names, child.name), [*scopes, child])
            elif type(child) is ast.ClassDef:
                visit(child, (*names, child.name), scopes)
            else:
                if type(child) is ast.Call and _is_launch(child):
                    found.append((".".join(names), child, scopes))
                visit(child, names, scopes)

    visit(tree, (), [tree])
    return found


def _flags_argument(call: ast.Call) -> ast.expr:
    function = call.func
    assert type(function) is ast.Attribute
    keyword = "flags" if function.attr == "create_process_suspended" else "creationflags"
    selected = [item.value for item in call.keywords if item.arg == keyword]
    if keyword == "flags" and len(call.args) >= 5:
        selected.append(call.args[4])
    assert len(selected) == 1, f"launch at line {call.lineno} has no explicit creation flags"
    return selected[0]


def _assignment(scope: ast.AST, name: str) -> ast.expr | None:
    nodes = scope.body if type(scope) is ast.Module else list(ast.walk(scope))
    values = [
        node.value
        for node in nodes
        if type(node) is ast.Assign
        and len(node.targets) == 1
        and type(node.targets[0]) is ast.Name
        and node.targets[0].id == name
    ]
    assert len(values) <= 1, f"{name} is assigned more than once"
    return values[0] if values else None


def _imported(tree: ast.Module, name: str) -> str | None:
    for node in tree.body:
        if (
            type(node) is ast.ImportFrom
            and node.module is not None
            and node.module.startswith("scripts.")
            and any(alias.name == name and alias.asname is None for alias in node.names)
        ):
            return "scripts/" + node.module.removeprefix("scripts.") + ".py"
    return None


def _windows_value(expression: ast.expr, scopes: list[ast.AST], tree: ast.Module) -> int:
    if type(expression) is ast.Constant and type(expression.value) is int:
        return expression.value
    if type(expression) is ast.BinOp and type(expression.op) is ast.BitOr:
        left = _windows_value(expression.left, scopes, tree)
        return left | _windows_value(expression.right, scopes, tree)
    if (
        type(expression) is ast.Attribute
        and type(expression.value) is ast.Name
        and expression.value.id == "subprocess"
        and expression.attr in _SUBPROCESS_FLAGS
    ):
        return _SUBPROCESS_FLAGS[expression.attr]
    if type(expression) is ast.IfExp and ast.unparse(expression.test) == "os.name == 'nt'":
        return _windows_value(expression.body, scopes, tree)
    if type(expression) is ast.Name:
        for scope in reversed(scopes):
            value = _assignment(scope, expression.id)
            if value is not None:
                return _windows_value(value, [tree], tree)
        source = _imported(tree, expression.id)
        if source is not None:
            other = _module(source)
            value = _assignment(other, expression.id)
            if value is not None:
                return _windows_value(value, [other], other)
    raise AssertionError(f"unreviewed creation flag expression: {ast.unparse(expression)}")


@pytest.mark.parametrize("relative", sorted(_CHAIN))
def test_every_reviewed_chain_launch_selects_windowless_flags(relative: str) -> None:
    tree = _module(relative)
    assert not [
        alias.name
        for node in ast.walk(tree)
        if type(node) is ast.ImportFrom and node.module in {"subprocess", "asyncio", "os"}
        for alias in node.names
        if alias.name in _LAUNCH_NAMES
    ], "launch functions must stay module-qualified so the inventory sees them"
    counts: dict[str, int] = {}
    for owner, call, scopes in _launches(tree):
        counts[owner] = counts.get(owner, 0) + 1
        flags = _windows_value(_flags_argument(call), scopes, tree)
        assert flags & _CREATE_NO_WINDOW, f"{relative}:{call.lineno} drops CREATE_NO_WINDOW"
        assert not flags & _CONSOLE_FLAGS, f"{relative}:{call.lineno} allows its own console"
    assert counts == _CHAIN[relative], "launch inventory changed; review it here"


def test_inventory_resolver_refuses_unreviewed_flag_expressions() -> None:
    tree = ast.parse(
        "import os, subprocess\n"
        "def hidden():\n    subprocess.run(['x'], creationflags=subprocess.CREATE_NO_WINDOW)\n"
        "def detached():\n"
        "    subprocess.Popen(['x'], creationflags=subprocess.CREATE_NO_WINDOW | 0x8)\n"
        "def opaque(flags):\n    subprocess.run(['x'], creationflags=flags)\n"
        "def implicit():\n    subprocess.run(['x'])\n"
    )
    results: dict[str, int | str] = {}
    for owner, call, scopes in _launches(tree):
        try:
            results[owner] = _windows_value(_flags_argument(call), scopes, tree)
        except AssertionError as error:
            results[owner] = str(error).splitlines()[0].split(":")[0]
    assert results == {
        "hidden": _CREATE_NO_WINDOW,
        "detached": _CREATE_NO_WINDOW | 0x8,
        "opaque": "unreviewed creation flag expression",
        "implicit": "launch at line 9 has no explicit creation flags",
    }
