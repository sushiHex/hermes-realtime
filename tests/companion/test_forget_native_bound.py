from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.companion import hermes_compat
from hermes_realtime.companion.hermes_compat import _bounded_delete_subtree, _compression_child
from hermes_realtime.companion.integrity import ArchiveRefusal
from hermes_realtime.companion.store import MAX_BOUND_CONVERSATIONS, DeleteTarget


def test_native_delegate_walk_is_bounded_before_it_starts() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            "CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, model_config TEXT)"
        )
        connection.execute("INSERT INTO sessions VALUES ('root', NULL, NULL)")
        connection.executemany(
            "INSERT INTO sessions VALUES (?, 'root', NULL)",
            ((f"child_{number}",) for number in range(MAX_BOUND_CONVERSATIONS)),
        )
        with pytest.raises(ArchiveRefusal, match="capacity"):
            _bounded_delete_subtree(connection, "root")
        connection.execute("DELETE FROM sessions WHERE id = 'child_0'")
        _bounded_delete_subtree(connection, "root")
    finally:
        connection.close()


def test_compression_child_metadata_is_bounded_before_parsing() -> None:
    with pytest.raises(ArchiveRefusal, match="lineage"):
        _compression_child(
            {"source": "voice", "model_config": "{}", "config_length": 4097},
            "parent",
        )


def test_native_delete_refuses_new_compression_child_inside_write_transaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, "
        "source TEXT, model_config TEXT, end_reason TEXT)"
    )
    connection.execute("INSERT INTO sessions VALUES ('voice', NULL, 'voice', NULL, 'compression')")
    connection.execute("INSERT INTO sessions VALUES ('late', 'voice', 'voice', '{}', NULL)")
    native_actions: list[str] = []

    class SessionDB:
        db_path = str(tmp_path / "state.db")

        def _execute_write(self, action: Any) -> Any:
            return action(connection)

    def native_delete(
        receiver: Any, session_id: str, *, sessions_dir: Path,
        expected_delete_ids: list[str],
    ) -> bool:
        assert session_id == "voice"
        assert expected_delete_ids == ["voice"]
        assert sessions_dir == tmp_path / "sessions"
        return bool(receiver._execute_write(lambda _conn: native_actions.append(session_id)))

    original_resolve = hermes_compat.resolve
    monkeypatch.setattr(
        hermes_compat, "resolve",
        lambda name: SessionDB if name == "SessionDB" else (
            native_delete if name == "SessionDB.delete_session" else original_resolve(name)
        ),
    )
    try:
        with pytest.raises(ArchiveRefusal, match="lineage"):
            hermes_compat.delete_target(SessionDB(), DeleteTarget("voice"))
        assert native_actions == []
    finally:
        connection.close()


def _native_db(connection: sqlite3.Connection, tmp_path: Path) -> Any:
    class SessionDB:
        db_path = str(tmp_path / "state.db")

        def _execute_write(self, action: Any) -> Any:
            return action(connection)

    return SessionDB


def test_a_branch_copy_is_found_even_after_hermes_orphaned_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, model_config TEXT)"
    )
    # Hermes's delete set the parent link to NULL; the /branch marker survives.
    connection.execute(
        "INSERT INTO sessions VALUES ('copy', NULL, '{\"_branched_from\": \"voice\"}')"
    )
    connection.execute("INSERT INTO sessions VALUES ('other', NULL, '{\"_branched_from\": \"x\"}')")
    session_db = _native_db(connection, tmp_path)
    monkeypatch.setattr(hermes_compat, "resolve", lambda name: session_db)
    try:
        assert hermes_compat.branch_copies_absent(session_db(), ("voice",)) is False
        connection.execute("DELETE FROM sessions WHERE id = 'copy'")
        assert hermes_compat.branch_copies_absent(session_db(), ("voice",)) is True
    finally:
        connection.close()


def test_a_retried_delete_still_removes_the_session_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, "
        "source TEXT, model_config TEXT, end_reason TEXT)"
    )
    removed: list[tuple[Path | None, str]] = []
    session_db = _native_db(connection, tmp_path)
    session_db._remove_session_files = lambda self, directory, sid: removed.append(
        (directory, sid)
    )

    def native_delete(receiver: Any, session_id: str, **_options: Any) -> bool:
        # A previous attempt committed the row deletion and was killed before its files.
        return False

    original_resolve = hermes_compat.resolve
    monkeypatch.setattr(
        hermes_compat, "resolve",
        lambda name: session_db if name == "SessionDB" else (
            native_delete if name == "SessionDB.delete_session" else original_resolve(name)
        ),
    )
    try:
        assert hermes_compat.delete_target(session_db(), DeleteTarget("voice")) is False
        assert removed == [(tmp_path / "sessions", "voice")]
    finally:
        connection.close()
