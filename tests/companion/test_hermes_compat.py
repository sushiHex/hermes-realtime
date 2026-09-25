from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.companion import hermes_compat
from hermes_realtime.companion.hermes_compat import (
    HERMES_MODULE,
    SURFACE,
    CompatError,
    HermesArchivePort,
    check_surface,
    resolve,
)

_SOURCE = Path(hermes_compat.__file__).read_text(encoding="utf-8")


def test_importing_the_compat_module_does_not_import_hermes() -> None:
    assert HERMES_MODULE not in sys.modules
    top_level = [
        node
        for node in ast.parse(_SOURCE).body
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    imported = set()
    for node in top_level:
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
        else:
            imported.update(alias.name.split(".")[0] for alias in node.names)
    assert imported == {"__future__", "importlib", "inspect", "typing", "hermes_realtime"}


def test_the_surface_names_each_hermes_name_once() -> None:
    names = [entry.name for entry in SURFACE]
    assert len(names) == len(set(names))
    assert all(name.split(".")[0] in {"SessionDB", "SessionTurnLeaseLostError",
                                      "CompressionSessionClosedError"} for name in names)


def test_every_hermes_attribute_the_module_uses_is_on_the_surface() -> None:
    used = set()
    for node in ast.walk(ast.parse(_SOURCE)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"resolve", "_method"}
        ):
            argument = node.args[-1]
            assert isinstance(argument, ast.Constant), "surface names must be literals"
            used.add(argument.value)
    assert used == {entry.name for entry in SURFACE}


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, str]:
    owners: dict[ast.AST, str] = {}
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            for node in ast.walk(function):
                owners.setdefault(node, function.name)
    return owners


def hermes_access_violations(source: str) -> list[str]:
    """Every way ``hermes_compat`` reaches into Hermes other than ``_method`` or ``resolve``.

    Hermes objects are the ``SessionDB`` (named ``db``, or held as ``self._db``) and whatever
    ``_lookup`` resolves. Only ``_lookup`` and ``_method`` may use ``getattr`` or import;
    the only thing done to a Hermes-provided connection is ``conn.execute``.
    """

    tree = ast.parse(source)
    owners = _enclosing_functions(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            value = node.value
            if isinstance(value, ast.Name) and value.id == "db":
                violations.append(f"db.{node.attr}")
            if isinstance(value, ast.Attribute) and value.attr == "_db":
                violations.append(f"_db.{node.attr}")
            if isinstance(value, ast.Name) and value.id == "conn" and node.attr != "execute":
                violations.append(f"conn.{node.attr}")
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in {"_session_db", "resolve", "_lookup", "_method"}
            ):
                violations.append(f"{value.func.id}().{node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (
            node.func.id in {"getattr", "setattr", "hasattr", "vars", "__import__"}
        ) and owners.get(node) not in {"_lookup", "_method"}:
            violations.append(f"{node.func.id}()")
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "import_module"
            and owners.get(node) != "_lookup"
        ):
            violations.append("import_module")
    return violations


def test_hermes_is_reached_only_through_the_enumerated_surface() -> None:
    assert hermes_access_violations(_SOURCE) == []


@pytest.mark.parametrize(
    ("before", "after", "violation"),
    [
        ('_method(_session_db(db), "SessionDB.release_session_turn_lease")',
         "_session_db(db).release_session_turn_lease",
         "_session_db().release_session_turn_lease"),
        ("def _session_db(db: Any) -> Any:\n",
         "def _session_db(db: Any) -> Any:\n    db.anything\n", "db.anything"),
        ("return durability_level(self._db)", "return self._db.durability", "_db.durability"),
    ],
    ids=["call-result", "db", "self-db"],
)
def test_the_access_rule_catches_a_direct_reach(before: str, after: str, violation: str) -> None:
    assert before in _SOURCE
    assert hermes_access_violations(_SOURCE.replace(before, after, 1)) == [violation]


def _stand_in(**overrides: Any) -> types.ModuleType:
    """A module shaped like the pinned ``hermes_state`` surface, with nothing behind it."""

    module = types.ModuleType(HERMES_MODULE)

    class SessionDB:
        _TRANSCRIPT_WRITE_PATIENCE_S = 60.0

        def _execute_write(self, fn: Any, patience_s: Any = None) -> Any: ...

        def _check_transcript_write_guards(
            self,
            conn: Any,
            session_id: Any,
            compression_lock_holder: Any,
            turn_lease_holder: Any = None,
            turn_lease_ttl_seconds: float = 300.0,
            reject_active_turn_lease: bool = False,
            reject_active_compression_lock: bool = False,
        ) -> None: ...

        def _insert_message_rows(self, conn: Any, session_id: Any, messages: Any) -> Any: ...

        def create_session(self, session_id: Any, source: Any, **kwargs: Any) -> Any: ...

        def try_acquire_session_turn_lease(
            self, session_id: Any, holder: Any, *, ttl_seconds: float = 300.0,
            patience_s: Any = None,
        ) -> bool: ...

        def refresh_session_turn_lease(
            self, session_id: Any, holder: Any, *, ttl_seconds: float = 300.0
        ) -> bool: ...

        def release_session_turn_lease(self, session_id: Any, holder: Any) -> None: ...

    for name, value in overrides.items():
        if value is None:
            delattr(SessionDB, name)
        else:
            setattr(SessionDB, name, value)
    module.SessionDB = SessionDB  # type: ignore[attr-defined]
    module.SessionTurnLeaseLostError = type("SessionTurnLeaseLostError", (RuntimeError,), {})  # type: ignore[attr-defined]
    module.CompressionSessionClosedError = type(  # type: ignore[attr-defined]
        "CompressionSessionClosedError", (RuntimeError,), {}
    )
    return module


def test_a_surface_matching_the_pin_has_no_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, HERMES_MODULE, _stand_in())
    assert check_surface() == ()


def test_a_missing_name_fails_the_surface_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, HERMES_MODULE, _stand_in(_insert_message_rows=None))
    assert check_surface() == ("missing:SessionDB._insert_message_rows",)


def test_a_changed_signature_fails_the_surface_check(monkeypatch: pytest.MonkeyPatch) -> None:
    def refresh(self: Any, session_id: Any, holder: Any, *, ttl: float = 300.0) -> bool: ...

    monkeypatch.setitem(
        sys.modules, HERMES_MODULE, _stand_in(refresh_session_turn_lease=refresh)
    )
    assert check_surface() == ("signature:SessionDB.refresh_session_turn_lease",)


def test_an_unlisted_hermes_name_is_refused_before_any_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, HERMES_MODULE, raising=False)
    with pytest.raises(CompatError):
        resolve("SessionDB.append_message")
    assert HERMES_MODULE not in sys.modules


def test_the_port_binds_only_an_exact_session_db(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _stand_in()
    monkeypatch.setitem(sys.modules, HERMES_MODULE, module)
    HermesArchivePort(module.SessionDB())

    class Derived(module.SessionDB):  # type: ignore[name-defined,misc]
        pass

    with pytest.raises(TypeError):
        HermesArchivePort(Derived())
    with pytest.raises(TypeError):
        HermesArchivePort(object())
