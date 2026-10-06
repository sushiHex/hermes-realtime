"""Bounded, model-free built-in memory snapshot passed to the foreground."""

from __future__ import annotations

from dataclasses import dataclass

MEMORY_MAX_BLOCK_BYTES = 4096


@dataclass(frozen=True, slots=True)
class BuiltinMemorySnapshot:
    memory: str
    user: str
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.memory) is not str or type(self.user) is not str:
            raise TypeError("memory blocks must be exact strings")
        if type(self.truncated) is not bool:
            raise TypeError("memory truncation flag must be an exact bool")
        if any(
            len(block.encode("utf-8")) > MEMORY_MAX_BLOCK_BYTES
            for block in (self.memory, self.user)
        ):
            raise ValueError("memory block exceeds byte bound")
