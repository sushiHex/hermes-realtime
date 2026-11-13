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
    frozenset({"__init__", "protocol", "speech"}),
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


def _absolute(anchor: str, level: int, module: str | None) -> str:
    parts = [PACKAGE, *anchor.split(".")] if anchor else [PACKAGE]
    if level > len(parts):
        raise AssertionError(f"relative import climbs above {PACKAGE}")
    return ".".join([*parts[: len(parts) - (level - 1)], *([module] if module else [])])


def _targets(node: ast.AST, anchor: str) -> Iterator[str]:
    """Yield the absolute dotted names ``node`` imports. ``anchor`` is its package."""
    if type(node) is ast.Import:
        yield from (alias.name for alias in node.names)
    elif type(node) is ast.ImportFrom:
        base = _absolute(anchor, node.level, node.module) if node.level else node.module or ""
        for alias in node.names:
            # ``from package import name`` imports a submodule when one has that name.
            child = f"{base}.{alias.name}"
            yield child if child.removeprefix(PACKAGE + ".") in FILES else base
    elif type(node) is ast.Call:
        function = node.func
        called = getattr(function, "id", None) or getattr(function, "attr", None)
        literal = node.args[0] if node.args else None
        if called not in ("import_module", "__import__") or type(literal) is not ast.Constant:
            return
        if type(literal.value) is str:
            name = literal.value.lstrip(".")
            level = len(literal.value) - len(name)
            yield _absolute(anchor, level, name or None) if level else name


def _package_imports() -> Iterator[tuple[str, str, str]]:
    """Yield (importing module, imported module, kind) for imports inside the package."""
    for module, path in FILES.items():
        is_package = path.name == "__init__.py"
        anchor = module if is_package else module.rpartition(".")[0]
        anchor = "" if anchor == "__init__" else anchor
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node, kind in _statements(tree, RUNTIME):
            for target in _targets(node, anchor):
                if target == PACKAGE or target.startswith(PACKAGE + "."):
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
