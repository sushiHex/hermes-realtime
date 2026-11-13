"""Import direction of the ``hermes_realtime`` package, as a ratchet.

``TIERS`` is the rule and its only statement: a module imports package code only from
its own unit or from a tier strictly below it. ``KNOWN_VIOLATIONS`` freezes the imports
that broke the rule when it was first measured. The test fails on an import that is not
listed and on a listing that no longer matches an import, so the list can only shrink.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

PACKAGE = "hermes_realtime"
SOURCE = Path(__file__).resolve().parents[1] / "src" / PACKAGE

# Lowest tier first. A unit is a top-level package or module of ``hermes_realtime``;
# ``__init__`` is the root package module itself. Units that share a tier are independent
# of each other. Placing a unit here is a design decision: review it as one.
TIERS: tuple[frozenset[str], ...] = (
    # Contracts: depend on nothing else in the package.
    frozenset({"__init__", "markers", "protocol", "speech"}),
    # The deterministic core: depends only on the contracts.
    frozenset({"conversation"}),
    # Shared facts and instrumentation that the core must not reach.
    frozenset({
        "_qualification",
        "memory",
        "network_origin",
        "production_observation",
        "result_limits",
        "search_egress",
    }),
    frozenset({"evidence"}),
    # The outside: transports, providers, surfaces and the Hermes side.
    frozenset({"client", "companion", "integration", "livekit", "providers", "testing"}),
    # Composition roots: the only place the outside is wired together.
    frozenset({"hermes_plugin", "host_launcher", "launcher"}),
)

# How an import is reached. All three are dependencies of the design: a function-level
# import defers a cycle without removing it, and an annotation-only import still binds
# the importer to the imported shape. The kind is part of the listing so that turning
# one kind into another is a visible change.
RUNTIME, LAZY, TYPING = "import", "lazy", "typing"

_TO_PROTOCOL = "move the shared bounds into protocol"
_EVIDENCE_PORT = "give the core a narrow evidence port; evidence imports inward"
_OBSERVATION_PORT = "give the core a narrow observation port"
_SHARED_ENUMS = "move the shared turn enums below evidence"
_SPEECH_TEXT = "move the provider-neutral speech text helpers into speech"
_MODEL_PORT = "make model selection a port that a composition root binds"
_CLIENT_TRANSPORT = "undecided: bind the transport in a root, or place client above livekit"
_SAVED_STATE = "undecided: give the saved-conversation readers one owner below both"

# (importing file, imported module, kind) -> what removes it.
KNOWN_VIOLATIONS: dict[tuple[str, str, str], str] = {
    ("protocol/events.py", "memory", LAZY): _TO_PROTOCOL,
    ("conversation/context.py", "memory", RUNTIME): _TO_PROTOCOL,
    ("conversation/streaming.py", "memory", RUNTIME): _TO_PROTOCOL,
    ("conversation/result_projection.py", "result_limits", RUNTIME): _TO_PROTOCOL,
    ("conversation/tasks.py", "result_limits", RUNTIME): _TO_PROTOCOL,
    ("conversation/actions.py", "evidence", RUNTIME): _EVIDENCE_PORT,
    ("conversation/actions.py", "evidence.lifecycle", RUNTIME): _EVIDENCE_PORT,
    ("conversation/commands.py", "evidence", RUNTIME): _EVIDENCE_PORT,
    ("conversation/commands.py", "evidence.lifecycle", RUNTIME): _EVIDENCE_PORT,
    ("conversation/streaming.py", "evidence", RUNTIME): _EVIDENCE_PORT,
    ("conversation/worker.py", "evidence", RUNTIME): _EVIDENCE_PORT,
    ("conversation/worker.py", "evidence.lifecycle", RUNTIME): _EVIDENCE_PORT,
    ("conversation/actions.py", "production_observation", RUNTIME): _OBSERVATION_PORT,
    ("conversation/streaming.py", "production_observation", RUNTIME): _OBSERVATION_PORT,
    ("conversation/worker.py", "production_observation", RUNTIME): _OBSERVATION_PORT,
    ("conversation/streaming.py", "_qualification", RUNTIME): (
        "pass the qualification context in from a composition root"
    ),
    ("conversation/knowledge.py", "providers.current_facts", RUNTIME): (
        "move the lookup types and predicates into conversation; the provider keeps the HTTP"
    ),
    ("conversation/result_projection.py", "providers._speech_text", LAZY): _SPEECH_TEXT,
    ("integration/api.py", "providers._speech_text", RUNTIME): _SPEECH_TEXT,
    ("_qualification.py", "evidence.models", RUNTIME): _SHARED_ENUMS,
    ("production_observation.py", "evidence.models", RUNTIME): _SHARED_ENUMS,
    ("client/http.py", "providers.ollama_session", LAZY): _MODEL_PORT,
    ("client/http.py", "providers.ollama_session", TYPING): _MODEL_PORT,
    ("client/runtime.py", "providers.ollama_session", TYPING): _MODEL_PORT,
    ("client/session.py", "providers.ollama_models", RUNTIME): _MODEL_PORT,
    ("client/session.py", "providers.ollama_session", LAZY): _MODEL_PORT,
    ("client/session.py", "providers.ollama_session", TYPING): _MODEL_PORT,
    ("providers/ollama_session.py", "client.session", RUNTIME): _MODEL_PORT,
    ("client/bootstrap.py", "livekit", RUNTIME): _CLIENT_TRANSPORT,
    ("client/runtime.py", "livekit", RUNTIME): _CLIENT_TRANSPORT,
    ("client/runtime.py", "livekit.worker", RUNTIME): _CLIENT_TRANSPORT,
    ("companion/host.py", "integration.run_record", RUNTIME): _SAVED_STATE,
    ("integration/voice_tail.py", "companion.integrity", RUNTIME): _SAVED_STATE,
    ("integration/voice_tail.py", "companion.review", RUNTIME): _SAVED_STATE,
    ("providers/codex_app_server.py", "integration.tool_catalog", LAZY): (
        "hand the tool catalog to the model adapter from a composition root"
    ),
    ("host_launcher.py", "launcher", RUNTIME): (
        "give the two launchers one composition function and its shared types a home below"
    ),
}


def _name(path: Path) -> str:
    parts = path.relative_to(SOURCE).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts) or "__init__"


FILES = {_name(path): path for path in sorted(SOURCE.rglob("*.py"))}


def _unit(module: str) -> str:
    return module.partition(".")[0]


def _is_type_checking(test: ast.expr) -> bool:
    return (type(test) is ast.Name and test.id == "TYPE_CHECKING") or (
        type(test) is ast.Attribute and test.attr == "TYPE_CHECKING"
    )


def _statements(node: ast.AST, kind: str) -> Iterator[tuple[ast.AST, str]]:
    """Yield every import statement and import call under ``node`` with its kind."""
    if type(node) is ast.If and _is_type_checking(node.test):
        for statement in node.body:
            yield from _statements(statement, TYPING)
        for statement in node.orelse:
            yield from _statements(statement, kind)
        return
    if type(node) in (ast.Import, ast.ImportFrom, ast.Call):
        yield node, kind
    if kind == RUNTIME and type(node) in (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda):
        kind = LAZY
    for child in ast.iter_child_nodes(node):
        yield from _statements(child, kind)


def _in_package(name: str) -> bool:
    return name == PACKAGE or name.startswith(PACKAGE + ".")


def _relative(package: str, level: int, module: str | None) -> str:
    """Resolve a relative name as Python does: ``level`` 1 is ``package`` itself."""
    parts = package.split(".")
    if level > len(parts):
        raise AssertionError(f"relative import climbs above {package}")
    return ".".join([*parts[: len(parts) - (level - 1)], *([module] if module else [])])


def _members(base: str, names: list[str]) -> Iterator[str]:
    """``from base import name`` imports a submodule when one has that name."""
    for name in names:
        child = f"{base}.{name}"
        yield child if child.removeprefix(PACKAGE + ".") in FILES else base


def _is_text(node: ast.AST | None) -> bool:
    return type(node) is ast.Constant and type(node.value) is str


def _argument(call: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(call.args) > position:
        return call.args[position]
    return next((item.value for item in call.keywords if item.arg == keyword), None)


def _dynamic(call: ast.Call, where: str) -> Iterator[str]:
    """Yield what an ``import_module`` or ``__import__`` call with a literal name imports.

    A computed name is outside a static measurement. A literal name whose target still
    depends on a runtime value fails here instead: an import that cannot be resolved
    must not read as an allowed one.
    """
    called = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
    name = _argument(call, 0, "name")
    if called not in ("import_module", "__import__") or not _is_text(name):
        return
    text = name.value

    def unresolved(why: str) -> AssertionError:
        return AssertionError(
            f"{where} line {call.lineno}: the target of {called}({text!r}, ...) cannot be "
            f"established from source: {why}. Use an import statement or literal arguments."
        )

    if any(type(item) is ast.Starred for item in call.args) or any(
        item.arg is None for item in call.keywords
    ):
        raise unresolved("its arguments are unpacked")
    if called == "import_module":
        if not text.startswith("."):
            yield text
            return
        package, module = _argument(call, 1, "package"), text.lstrip(".")
        if not _is_text(package) or not module:
            raise unresolved("a relative name needs a literal package to resolve against")
        yield _relative(package.value, len(text) - len(module), module)
        return
    level, fromlist = _argument(call, 4, "level"), _argument(call, 3, "fromlist")
    absolute = level is None or (type(level) is ast.Constant and level.value == 0)
    if text.startswith(".") or not absolute:
        raise unresolved("a relative __import__ resolves against its globals")
    if fromlist is None:
        yield text
    elif type(fromlist) in (ast.Tuple, ast.List) and all(map(_is_text, fromlist.elts)):
        yield from _members(text, [item.value for item in fromlist.elts] or [""])
    elif _in_package(text):
        raise unresolved("its fromlist is not a literal")


def _targets(node: ast.AST, anchor: str, where: str) -> Iterator[str]:
    """Yield the absolute dotted names ``node`` imports. ``anchor`` is its package."""
    if type(node) is ast.Import:
        yield from (alias.name for alias in node.names)
    elif type(node) is ast.ImportFrom:
        base = _relative(anchor, node.level, node.module) if node.level else node.module or ""
        yield from _members(base, [alias.name for alias in node.names])
    elif type(node) is ast.Call:
        yield from _dynamic(node, where)


def _package_imports() -> Iterator[tuple[str, str, str]]:
    """Yield (importing module, imported module, kind) for imports inside the package."""
    for module, path in FILES.items():
        qualified = PACKAGE if module == "__init__" else f"{PACKAGE}.{module}"
        anchor = qualified if path.name == "__init__.py" else qualified.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node, kind in _statements(tree, RUNTIME):
            for target in _targets(node, anchor, module):
                if _in_package(target):
                    yield module, target.removeprefix(PACKAGE).lstrip(".") or "__init__", kind


def test_every_top_level_unit_is_placed_in_exactly_one_tier() -> None:
    placed = sorted(unit for tier in TIERS for unit in tier)
    assert placed == sorted({_unit(module) for module in FILES})


def test_imports_point_only_to_lower_tiers() -> None:
    tier = {unit: index for index, units in enumerate(TIERS) for unit in units}
    measured: set[tuple[str, str, str]] = set()
    for module, target, kind in _package_imports():
        assert target in FILES, f"{module} imports {target}, which is not a module here"
        source_unit, target_unit = _unit(module), _unit(target)
        if source_unit != target_unit and tier[target_unit] >= tier[source_unit]:
            measured.add((FILES[module].relative_to(SOURCE).as_posix(), target, kind))

    new = sorted(measured - KNOWN_VIOLATIONS.keys())
    stale = sorted(KNOWN_VIOLATIONS.keys() - measured)
    assert not new, f"imports that do not point to a lower tier: {new}"
    assert not stale, f"listed violations that no longer exist; delete them: {stale}"
    assert all(type(reason) is str and reason for reason in KNOWN_VIOLATIONS.values())
