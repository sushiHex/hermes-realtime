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
