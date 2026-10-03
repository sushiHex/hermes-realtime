from __future__ import annotations

import ast
import contextlib
import sys
import threading
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
from hermes_realtime.companion.integrity import ArchiveRefusal, Fingerprint, Identity
from hermes_realtime.companion.review import ReviewRequest
from hermes_realtime.companion.store import ConversationRecord, Progress

_SOURCE = Path(hermes_compat.__file__).read_text(encoding="utf-8")


def _rollback_port(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    fingerprint = Fingerprint(1, "0" * 64)
    record = ConversationRecord(
        "conv", "session", Progress(fingerprint, Identity(0, 0)),
        None, None, None, "lease",
    )
    request = ReviewRequest("conv", 0, 0, 0, True, True, False)

    class Parent:
        def __init__(self) -> None:
            self.active: object | None = None
            self.closed = False

        def close(self) -> None:
            assert self.active is None
            self.closed = True

    class Native:
        def __init__(self) -> None:
            self.commit_fails = True
            self.finish_failures = 0
            self.finish_calls = 0
            self.prepare_calls = 0
            self.block_finish_call = 0
            self.finish_entered = threading.Event()
            self.finish_release = threading.Event()

        def prepare(self, parent: Parent) -> object | None:
            self.prepare_calls += 1
            if parent.active is not None:
                return None
            token = object()
            parent.active = token
            return token

        def finish(self, parent: Parent, token: object) -> None:
            self.finish_calls += 1
            if self.finish_calls == self.block_finish_call:
                self.finish_entered.set()
                if not self.finish_release.wait(5):
                    raise RuntimeError("synthetic release did not arrive")
            if self.finish_failures:
                self.finish_failures -= 1
                raise RuntimeError("synthetic transient finish failure")
            assert parent.active is token
            parent.active = None

        def execute_write(self, callback: Any) -> Any:
            admitted = callback(Connection())
            if self.commit_fails:
                raise OSError("synthetic commit failure")
            return admitted

    class Connection:
        def execute(self, sql: str, params: Any) -> Any:
            assert "SELECT role, content" in sql
            return type("Rows", (), {"fetchall": lambda self: [
                ("user", "review me", "voice:conv:0:0")
            ]})()

    native = Native()

    def method(db: Any, name: str) -> Any:
        return {
            "SessionDB._execute_write": native.execute_write,
            "SessionDB._check_transcript_write_guards": lambda *args, **kwargs: None,
        }[name]

    def review_resolve(name: str) -> Any:
        return {
            "agent.background_review.prepare_background_review_run": native.prepare,
            "agent.background_review.finish_background_review_run": native.finish,
            "SessionTurnLeaseLostError": RuntimeError,
            "CompressionSessionClosedError": RuntimeError,
            "run_agent.AIAgent": Parent,
            "run_agent.AIAgent.close": Parent.close,
        }[name]

    monkeypatch.setattr(hermes_compat, "_method", method)
    monkeypatch.setattr(hermes_compat, "resolve", review_resolve)
    monkeypatch.setattr(
        hermes_compat, "_read_projection",
        lambda conn, session_id, cap: type(
            "Projection", (), {"fingerprint": lambda self: fingerprint}
        )(),
    )
    port = object.__new__(HermesArchivePort)
    port._db = object()
    port._review_bindings = {}
    port._failed_parent_closes = []
    port._rollback_finishes = {}
    port._rollback_lock = threading.Lock()
    port._review_scope = contextlib.nullcontext  # type: ignore[method-assign]
    return port, native, Parent(), record, request


def test_failed_transaction_rollback_finish_retries_before_next_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, native, parent, record, request = _rollback_port(monkeypatch)
    native.finish_failures = 2
    with pytest.raises(OSError, match="synthetic commit failure"):
        port.admit(parent, record, request, 4096, 30.0)
    assert native.prepare_calls == 1
    assert parent.active is not None

    native.commit_fails = False
    admitted = port.admit(parent, record, request, 4096, 30.0)
    assert admitted is not None
    assert native.prepare_calls == 2
    assert native.finish_calls == 3
    port.finish(parent, admitted[1])
    assert parent.active is None


@pytest.mark.parametrize("drain_first", [False, True])
def test_failed_rollback_finish_is_drained_before_parent_close(
    monkeypatch: pytest.MonkeyPatch, drain_first: bool,
) -> None:
    port, native, parent, record, request = _rollback_port(monkeypatch)
    native.finish_failures = 2
    with pytest.raises(OSError, match="synthetic commit failure"):
        port.admit(parent, record, request, 4096, 30.0)
    if drain_first:
        assert port.drain_failed_parents(2)
        assert parent.active is None
    port.close_parent(parent)
    assert parent.closed
    assert native.finish_calls == 3


def test_live_rollback_finish_is_not_duplicated_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, native, parent, record, request = _rollback_port(monkeypatch)
    native.finish_failures = 1
    native.block_finish_call = 2
    try:
        with pytest.raises(OSError, match="synthetic commit failure"):
            port.admit(parent, record, request, 4096, 30.0)
        assert native.finish_entered.wait(2)
        assert not port.drain_failed_parents(0.05)
        assert native.finish_calls == 2
        assert not parent.closed
        native.finish_release.set()
        port.close_parent(parent)
        assert parent.closed
        assert native.finish_calls == 2
    finally:
        native.finish_release.set()


def test_port_close_waits_for_live_rollback_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, native, parent, record, request = _rollback_port(monkeypatch)
    native.finish_failures = 1
    native.block_finish_call = 2
    closed: list[object] = []
    errors: list[BaseException] = []
    monkeypatch.setattr(hermes_compat, "close_session_db", closed.append)
    try:
        with pytest.raises(OSError, match="synthetic commit failure"):
            port.admit(parent, record, request, 4096, 30.0)
        assert native.finish_entered.wait(2)

        def close() -> None:
            try:
                port.close()
            except BaseException as error:
                errors.append(error)

        closer = threading.Thread(target=close)
        closer.start()
        closer.join(0.05)
        assert closer.is_alive()
        assert closed == []
        assert native.finish_calls == 2
        native.finish_release.set()
        closer.join(2)
        assert not closer.is_alive()
        assert errors == []
        assert closed == [port._db]
    finally:
        native.finish_release.set()


def test_rollback_finish_thread_start_failure_retains_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port, native, parent, record, request = _rollback_port(monkeypatch)
    native.finish_failures = 1
    original_start = threading.Thread.start
    failures = 0

    def fail_once(worker: threading.Thread) -> None:
        nonlocal failures
        if worker.name == "voice-review-rollback-finish" and failures == 0:
            failures += 1
            raise RuntimeError("synthetic thread start failure")
        original_start(worker)

    monkeypatch.setattr(threading.Thread, "start", fail_once)
    with pytest.raises(OSError, match="synthetic commit failure"):
        port.admit(parent, record, request, 4096, 30.0)
    assert failures == 1
    assert parent.active is not None
    assert port.drain_failed_parents(2)
    assert parent.active is None
    assert native.finish_calls == 2


def test_routed_credential_change_changes_parent_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    port = object.__new__(HermesArchivePort)
    routed = {"provider": "custom", "model": "other", "api_key": "first"}

    def review_resolve(name: str) -> Any:
        assert name == "hermes_cli.runtime_provider.resolve_runtime_provider"
        return lambda **kwargs: routed

    monkeypatch.setattr(hermes_compat, "resolve", review_resolve)
    config = {"auxiliary": {"background_review": {"provider": "custom", "model": "other"}}}
    main = {"provider": "custom", "model": "main", "api_key": "main-key"}
    baseline = port._binding(config, "main", main)
    routed["api_key"] = "second"
    assert port._binding(config, "main", main) != baseline


@pytest.mark.parametrize(("attribute", "value"), [("_session_db", object()),
                                              ("_session_json_enabled", True)])
def test_review_parent_shape_refuses_db_or_json_persistence(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: object
) -> None:
    class MemoryStore:
        pass

    class Parent:
        pass

    parent = Parent()
    parent.session_id = "voice"
    parent.enabled_toolsets = ["memory", "skills"]
    parent.disabled_toolsets = None
    parent.tools = []
    parent._memory_enabled = True
    parent._memory_store = MemoryStore()
    parent._memory_manager = None
    parent.background_review_callback = None
    parent._session_db = None
    parent._owns_session_db = False
    parent._end_session_on_close = False
    parent._persist_disabled = True
    parent._session_json_enabled = False

    def review_resolve(name: str) -> Any:
        return {
            "run_agent.AIAgent": Parent,
            "tools.memory_tool.MemoryStore": MemoryStore,
            "model_tools.get_tool_definitions": lambda **kwargs: [],
        }[name]

    monkeypatch.setattr(hermes_compat, "resolve", review_resolve)
    port = object.__new__(HermesArchivePort)
    port._verify_review_parent_shape(parent)
    setattr(parent, attribute, value)
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port._verify_review_parent_shape(parent)


def _partial_parent_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, close: Any
) -> HermesArchivePort:
    class MemoryStore:
        def load_from_disk(self) -> None:
            raise RuntimeError("synthetic load failure")

    class Parent:
        def __init__(self, **kwargs: Any) -> None:
            self.session_id = kwargs["session_id"]
            self.enabled_toolsets = ["memory", "skills"]
            self.disabled_toolsets = None
            self.tools = []
            self._memory_enabled = True
            self._memory_store = MemoryStore()
            self._memory_manager = None
            self.background_review_callback = None
            self._session_db = kwargs["session_db"]
            self._owns_session_db = False
            self._end_session_on_close = True
            self._persist_disabled = False
            self._session_json_enabled = True

        def close(self) -> None:
            close()

    config = {"model": {"default": "main"}}

    def review_resolve(name: str) -> Any:
        return {
            "hermes_constants.get_hermes_home": lambda: tmp_path,
            "hermes_cli.config.load_config_readonly": lambda: config,
            "hermes_cli.runtime_provider.resolve_runtime_provider":
                lambda **kwargs: {"provider": "custom", "api_key": "synthetic"},
            "run_agent.AIAgent": Parent,
            "run_agent.AIAgent.close": Parent.close,
            "tools.memory_tool.MemoryStore": MemoryStore,
            "tools.memory_tool.MemoryStore.load_from_disk": MemoryStore.load_from_disk,
            "model_tools.get_tool_definitions": lambda **kwargs: [],
        }[name]

    monkeypatch.setattr(hermes_compat, "resolve", review_resolve)
    port = object.__new__(HermesArchivePort)
    port._review_bindings = {}
    port._failed_parent_closes = []
    port._rollback_finishes = {}
    port._rollback_lock = threading.Lock()
    port._parent_creation_failed = False
    port._review_home = lambda: tmp_path  # type: ignore[method-assign]
    port._config_signature = lambda: (None, None)  # type: ignore[method-assign]
    port._review_scope = contextlib.nullcontext  # type: ignore[method-assign]
    return port


def test_failed_constructed_parent_is_closed_before_port_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed = threading.Event()
    port = _partial_parent_port(monkeypatch, tmp_path, closed.set)
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port._make_review_parent_scoped("voice")
    assert port._parent_creation_failed
    assert port.drain_failed_parents(2)
    assert closed.is_set()


def test_partial_parent_close_failure_retries_on_next_drain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempts: list[bool] = []

    def close() -> None:
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("synthetic transient close failure")

    port = _partial_parent_port(monkeypatch, tmp_path, close)
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port._make_review_parent_scoped("voice")
    assert port._parent_creation_failed
    assert not port.drain_failed_parents(2)
    assert port.drain_failed_parents(2)
    assert attempts == [True, True]


def test_partial_parent_live_close_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered = threading.Event()
    release = threading.Event()
    attempts: list[bool] = []

    def close() -> None:
        attempts.append(True)
        entered.set()
        if not release.wait(5):
            raise RuntimeError("synthetic close did not release")

    port = _partial_parent_port(monkeypatch, tmp_path, close)
    try:
        with pytest.raises(ArchiveRefusal, match="configuration"):
            port._make_review_parent_scoped("voice")
        assert entered.wait(2)
        assert not port.drain_failed_parents(0.05)
        assert not port.drain_failed_parents(0.05)
        assert attempts == [True]
        release.set()
        assert port.drain_failed_parents(2)
        assert attempts == [True]
    finally:
        release.set()


def test_partial_parent_close_start_failure_retains_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    closed = threading.Event()
    original_start = threading.Thread.start
    starts: list[bool] = []

    def fail_once(worker: threading.Thread) -> None:
        if worker.name == "voice-review-parent-cleanup" and not starts:
            starts.append(True)
            raise RuntimeError("synthetic thread start failure")
        original_start(worker)

    monkeypatch.setattr(threading.Thread, "start", fail_once)
    port = _partial_parent_port(monkeypatch, tmp_path, closed.set)
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port._make_review_parent_scoped("voice")
    assert starts == [True]
    assert port._parent_creation_failed
    assert not port.drain_failed_parents(2)
    assert port.drain_failed_parents(2)
    assert closed.is_set()


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
    assert imported == {
        "__future__", "contextlib", "copy", "hashlib", "importlib", "inspect",
        "json", "pathlib", "threading", "time", "typing",
        "hermes_realtime",
    }


def test_the_surface_names_each_hermes_name_once() -> None:
    names = [entry.name for entry in SURFACE]
    assert len(names) == len(set(names))
    assert all(name.split(".")[0] in {
        "SessionDB", "SessionTurnLeaseLostError", "CompressionSessionClosedError",
        "agent", "hermes_cli", "run_agent", "model_tools", "gateway",
        "hermes_constants",
        "tools",
    } for name in names)


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
    pinned_parent_methods = {
        "run_agent.AIAgent._safe_print", "run_agent.AIAgent._emit_auxiliary_failure"
    }
    assert used | hermes_compat.REVIEW_INSTANCE_FIELDS | pinned_parent_methods == {
        entry.name for entry in SURFACE
    }


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

        def close(self) -> None: ...

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


def test_the_close_the_companion_relies_on_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    def close(self: Any, wait: bool = True) -> None: ...

    monkeypatch.setitem(sys.modules, HERMES_MODULE, _stand_in(close=close))
    assert check_surface() == ("signature:SessionDB.close",)
    monkeypatch.setitem(sys.modules, HERMES_MODULE, _stand_in(close=None))
    assert check_surface() == ("missing:SessionDB.close",)


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
