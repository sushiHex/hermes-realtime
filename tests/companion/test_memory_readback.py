from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.companion import hermes_compat
from hermes_realtime.companion.hermes_compat import HermesArchivePort
from hermes_realtime.companion.integrity import ArchiveRefusal
from hermes_realtime.memory import MEMORY_MAX_BLOCK_BYTES, BuiltinMemorySnapshot


def _reader_port(
    monkeypatch: pytest.MonkeyPatch, home: Path,
    *, memory_enabled: bool = True, user_enabled: bool = True,
) -> tuple[HermesArchivePort, list[str], dict[str, Any]]:
    calls: list[str] = []
    config: dict[str, Any] = {
        "memory": {
            "memory_char_limit": 2200,
            "user_char_limit": 1375,
            "memory_enabled": memory_enabled,
            "user_profile_enabled": user_enabled,
        }
    }

    class NativeStore:
        def __init__(
            self, memory_char_limit: int = 2200, user_char_limit: int = 1375,
            *, memory_enabled: bool = True, user_profile_enabled: bool = True,
        ) -> None:
            self.memory_char_limit = memory_char_limit
            self.user_char_limit = user_char_limit
            self.memory_enabled = memory_enabled
            self.user_profile_enabled = user_profile_enabled

        @staticmethod
        def _parse_entries(raw: str) -> list[str]:
            calls.append("parse")
            return [item.strip() for item in raw.split("\n§\n") if item.strip()]

        @staticmethod
        def _sanitize_entries_for_snapshot(
            entries: list[str], filename: str
        ) -> list[str]:
            calls.append("sanitize:" + filename)
            return ["[BLOCKED]" if "POISON" in entry else entry for entry in entries]

        def _render_block(self, target: str, entries: list[str]) -> str:
            calls.append("render:" + target)
            return f"{target}:" + "\n§\n".join(entries) if entries else ""

        def load_from_disk(self) -> None:
            raise AssertionError("reader must not call unbounded native loader")

    table: dict[str, Any] = {
        "hermes_constants.get_hermes_home": lambda: home,
        "hermes_cli.config.load_config_readonly": lambda: config,
        "tools.memory_tool.get_memory_dir": lambda: home / "memories",
        "tools.memory_tool.get_builtin_memory_store_flags":
            lambda value: (
                value["memory"]["memory_enabled"],
                value["memory"]["user_profile_enabled"],
            ),
        "tools.memory_tool.MemoryStore": NativeStore,
        "tools.memory_tool.MemoryStore._parse_entries": NativeStore._parse_entries,
        "tools.memory_tool.MemoryStore._sanitize_entries_for_snapshot":
            NativeStore._sanitize_entries_for_snapshot,
        "tools.memory_tool.MemoryStore._render_block": NativeStore._render_block,
        "tools.memory_tool.ENTRY_DELIMITER": "\n§\n",
    }

    def checked_resolve(name: str) -> Any:
        calls.append(name)
        return table[name]

    monkeypatch.setattr(hermes_compat, "resolve", checked_resolve)
    monkeypatch.setattr(hermes_compat, "check_review_surface", lambda: ())
    monkeypatch.setattr(hermes_compat, "check_memory_surface", lambda: ())
    port = object.__new__(HermesArchivePort)
    port._review_home = lambda: home  # type: ignore[method-assign]
    port._review_scope = contextlib.nullcontext  # type: ignore[method-assign]
    port._config_signature = lambda: (None, None)  # type: ignore[method-assign]
    return port, calls, table


def test_readback_is_fresh_hermes_parsed_and_model_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("first\n§\nPOISON", encoding="utf-8")
    (memory_dir / "USER.md").write_text("likes clarity", encoding="utf-8")
    port, calls, _ = _reader_port(monkeypatch, tmp_path)

    first = port.read_builtin_memory()
    assert first == BuiltinMemorySnapshot("memory:first\n§\n[BLOCKED]", "user:likes clarity")
    (memory_dir / "MEMORY.md").write_text("corrected", encoding="utf-8")
    second = port.read_builtin_memory()
    assert second == BuiltinMemorySnapshot("memory:corrected", "user:likes clarity")
    assert calls.count("parse") == 4
    assert calls.count("sanitize:MEMORY.md") == 2
    assert calls.count("render:memory") == 2
    assert all("AIAgent" not in call and "provider" not in call for call in calls)


def test_readback_omits_disabled_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("enabled", encoding="utf-8")
    (memory_dir / "USER.md").write_text("must not leak", encoding="utf-8")
    port, _, _ = _reader_port(monkeypatch, tmp_path, user_enabled=False)
    snapshot = port.read_builtin_memory()
    assert snapshot == BuiltinMemorySnapshot("memory:enabled", "")


def test_readback_truncates_oversize_valid_prefix_and_keeps_byte_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_bytes(b"A" * 20_000 + b"POISON beyond cap")
    port, _, _ = _reader_port(monkeypatch, tmp_path)
    snapshot = port.read_builtin_memory()
    assert snapshot.truncated
    assert snapshot.memory.startswith("memory:AAAA")
    assert "POISON" not in snapshot.memory
    assert "[truncated]" in snapshot.memory
    assert len(snapshot.memory.encode("utf-8")) <= MEMORY_MAX_BLOCK_BYTES


def test_readback_drops_only_incomplete_utf8_at_prefix_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_bytes(b"A" * (16_384 - 1) + "€".encode() + b"end")
    port, _, _ = _reader_port(monkeypatch, tmp_path)
    snapshot = port.read_builtin_memory()
    assert snapshot.truncated
    assert "�" not in snapshot.memory


def test_readback_refuses_interior_invalid_utf8(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_bytes(b"valid\xffinvalid")
    port, _, _ = _reader_port(monkeypatch, tmp_path)
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port.read_builtin_memory()


def test_readback_refuses_wrong_memory_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("valid", encoding="utf-8")
    port, _, table = _reader_port(monkeypatch, tmp_path)
    table["tools.memory_tool.get_memory_dir"] = lambda: tmp_path.parent / "other"
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port.read_builtin_memory()


def test_readback_refuses_native_home_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    port, _, table = _reader_port(monkeypatch, tmp_path)
    table["hermes_constants.get_hermes_home"] = lambda: tmp_path.parent
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port.read_builtin_memory()


def test_readback_refuses_config_change_during_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    port, _, _ = _reader_port(monkeypatch, tmp_path)
    signatures = iter(((None, None), (1, 1)))
    port._config_signature = lambda: next(signatures)  # type: ignore[method-assign]
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port.read_builtin_memory()


def test_readback_refuses_changed_native_surface_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    port, calls, _ = _reader_port(monkeypatch, tmp_path)
    monkeypatch.setattr(hermes_compat, "check_memory_surface", lambda: ("signature",))
    with pytest.raises(ArchiveRefusal, match="incompatible"):
        port.read_builtin_memory()
    assert calls == []


def test_readback_refuses_bad_flags_and_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    port, _, table = _reader_port(monkeypatch, tmp_path)
    table["tools.memory_tool.get_builtin_memory_store_flags"] = lambda _: (1, True)
    with pytest.raises(ArchiveRefusal, match="incompatible"):
        port.read_builtin_memory()

    table["tools.memory_tool.get_builtin_memory_store_flags"] = lambda _: (True, True)
    table["hermes_cli.config.load_config_readonly"] = lambda: {
        "memory": {"memory_char_limit": True, "user_char_limit": 1375}
    }
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port.read_builtin_memory()


def test_readback_refuses_changed_file_during_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    memory_file = memory_dir / "MEMORY.md"
    memory_file.write_bytes(b"before")
    port, _, _ = _reader_port(monkeypatch, tmp_path)
    original = port._bounded_memory_source
    observed = 0

    def changing_source(path: Path) -> Any:
        nonlocal observed
        result = original(path)
        observed += 1
        if observed == 1:
            memory_file.write_bytes(b"after")
        return result

    monkeypatch.setattr(port, "_bounded_memory_source", changing_source)
    with pytest.raises(ArchiveRefusal, match="configuration"):
        port.read_builtin_memory()


def test_readback_native_char_limit_and_hard_byte_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("€" * 1_500, encoding="utf-8")
    port, _, table = _reader_port(monkeypatch, tmp_path)
    table["hermes_cli.config.load_config_readonly"] = lambda: {
        "memory": {
            "memory_char_limit": 3_000,
            "user_char_limit": 1_375,
            "memory_enabled": True,
            "user_profile_enabled": False,
        }
    }
    snapshot = port.read_builtin_memory()
    assert snapshot.truncated
    assert snapshot.memory.endswith("[truncated]")
    assert len(snapshot.memory.encode("utf-8")) <= MEMORY_MAX_BLOCK_BYTES
    assert snapshot.memory.count("€") > 1_000


def test_readback_native_char_limit_even_when_byte_bound_fits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_bytes(b"A" * 200)
    port, _, table = _reader_port(monkeypatch, tmp_path)
    table["hermes_cli.config.load_config_readonly"] = lambda: {
        "memory": {
            "memory_char_limit": 50,
            "user_char_limit": 1_375,
            "memory_enabled": True,
            "user_profile_enabled": False,
        }
    }
    snapshot = port.read_builtin_memory()
    assert snapshot.truncated
    assert snapshot.memory.endswith("[truncated]")
    assert len(snapshot.memory.removeprefix("memory:")) <= 50


def test_readback_absent_files_are_empty_and_no_unbounded_native_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    port, calls, _ = _reader_port(monkeypatch, tmp_path)
    assert port.read_builtin_memory() == BuiltinMemorySnapshot("", "")
    assert "load_from_disk" not in calls
