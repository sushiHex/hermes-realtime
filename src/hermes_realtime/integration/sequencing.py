"""Shared ordering for events emitted by independent integration components."""

from __future__ import annotations

import asyncio


class EventSequencer:
    """Allocate unique, monotonically increasing event sequence numbers."""

    def __init__(self) -> None:
        self._last_sequence = -1
        self._lock = asyncio.Lock()

    async def observe(self, sequence: int) -> None:
        """Advance to an externally emitted sequence without allocating one."""

        async with self._lock:
            self._last_sequence = max(self._last_sequence, sequence)

    async def next(self, observed_sequence: int = -1) -> int:
        """Return one sequence strictly after all observed or allocated values."""

        async with self._lock:
            sequence = max(self._last_sequence, observed_sequence) + 1
            self._last_sequence = sequence
            return sequence
