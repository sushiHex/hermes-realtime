from __future__ import annotations

import pytest

from hermes_realtime.memory import MEMORY_MAX_BLOCK_BYTES, BuiltinMemorySnapshot


def test_builtin_memory_snapshot_is_exact_and_byte_bounded() -> None:
    assert MEMORY_MAX_BLOCK_BYTES == 4096
    assert BuiltinMemorySnapshot("note", "profile") == BuiltinMemorySnapshot(
        "note", "profile", False
    )
    assert len(BuiltinMemorySnapshot("é" * 2048, "").memory.encode("utf-8")) == 4096
    with pytest.raises(ValueError, match="bound"):
        BuiltinMemorySnapshot("é" * 2049, "")
    with pytest.raises(TypeError, match="exact"):
        BuiltinMemorySnapshot(True, "")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact"):
        BuiltinMemorySnapshot("", "", 1)  # type: ignore[arg-type]
