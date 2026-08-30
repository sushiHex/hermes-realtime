"""Bounded non-awaiting PCM admission for one media binding."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from hermes_realtime.speech import AudioFrame

DEFAULT_INGRESS_MAX_FRAMES = 256
DEFAULT_INGRESS_MAX_BYTES = 512 * 1024
DEFAULT_INGRESS_MAX_AGE_SECONDS = 2.0
MAX_INGRESS_FRAMES = 512
MAX_INGRESS_BYTES = 8 * 1024 * 1024
MAX_INGRESS_AGE_SECONDS = 10.0


class IngressOverloadError(RuntimeError):
    """The exact binding lost ordered PCM continuity and must be replaced."""


class IngressClosedError(RuntimeError):
    """The exact ingress generation is no longer readable or writable."""


class IngressOrderingError(RuntimeError):
    """PCM evidence was duplicated, skipped, or reordered."""


@dataclass(frozen=True, slots=True, eq=False)
class IngressRecord:
    """One immutable SDK-yielded PCM observation admitted in source order."""

    sequence: int
    participant_identity: str
    session_generation: int
    track_name: str | None
    frame: AudioFrame
    observed_at: float

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("ingress sequence must be a positive exact integer")
        if type(self.participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if type(self.session_generation) is not int:
            raise TypeError("session_generation must be an exact integer")
        if self.track_name is not None and type(self.track_name) is not str:
            raise TypeError("track_name must be an exact built-in string or None")
        if type(self.frame) is not AudioFrame:
            raise TypeError("frame must be an exact AudioFrame")
        if type(self.observed_at) is not float:
            raise TypeError("observed_at must be an exact float")
        if not math.isfinite(self.observed_at) or self.observed_at < 0:
            raise ValueError("observed_at must be a finite monotonic timestamp")
        object.__setattr__(
            self,
            "frame",
            AudioFrame(
                pcm=self.frame.pcm,
                sample_rate_hz=self.frame.sample_rate_hz,
                channels=self.frame.channels,
            ),
        )


class BoundedPcmIngress:
    """Single-loop, single-consumer sequencer with synchronous admission."""

    def __init__(
        self,
        *,
        max_frames: int = DEFAULT_INGRESS_MAX_FRAMES,
        max_bytes: int = DEFAULT_INGRESS_MAX_BYTES,
        max_age_seconds: float = DEFAULT_INGRESS_MAX_AGE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(max_frames) is not int:
            raise TypeError("max_frames must be an exact integer")
        if type(max_bytes) is not int:
            raise TypeError("max_bytes must be an exact integer")
        if not 1 <= max_frames <= MAX_INGRESS_FRAMES:
            raise ValueError("max_frames is outside the supported range")
        if not 1 <= max_bytes <= MAX_INGRESS_BYTES:
            raise ValueError("max_bytes is outside the supported range")
        if type(max_age_seconds) is not float:
            raise TypeError("max_age_seconds must be an exact float")
        if not math.isfinite(max_age_seconds) or not (
            0 < max_age_seconds <= MAX_INGRESS_AGE_SECONDS
        ):
            raise ValueError("max_age_seconds is outside the supported range")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._max_frames = max_frames
        self._max_bytes = max_bytes
        self._max_age_seconds = max_age_seconds
        self._clock = clock
        self._records: deque[IngressRecord] = deque()
        self._outstanding_frames = 0
        self._outstanding_bytes = 0
        self._active: IngressRecord | None = None
        self._next_sequence = 1
        self._changed = asyncio.Event()
        self._fault: BaseException | None = None
        self._closed = False

    @property
    def outstanding_frames(self) -> int:
        return self._outstanding_frames

    @property
    def outstanding_bytes(self) -> int:
        return self._outstanding_bytes

    @property
    def max_age_seconds(self) -> float:
        return self._max_age_seconds

    def check_health(self) -> None:
        """Synchronously surface sticky faults and queued-frame expiry."""

        self._raise_if_unavailable()
        self._fault_if_oldest_expired()

    def admit(self, record: IngressRecord) -> None:
        """Admit without suspension or fault the exact ingress generation."""

        if type(record) is not IngressRecord:
            raise TypeError("record must be an exact IngressRecord")
        self._raise_if_unavailable()
        self._fault_if_oldest_expired()
        if record.sequence != self._next_sequence:
            ordering_error = IngressOrderingError(
                "PCM ingress sequence is not exact; the media binding is desynchronized"
            )
            self._fault = ordering_error
            self._changed.set()
            raise ordering_error
        frame_bytes = len(record.frame.pcm)
        if (
            self._outstanding_frames >= self._max_frames
            or self._outstanding_bytes + frame_bytes > self._max_bytes
        ):
            capacity_error = IngressOverloadError(
                "PCM ingress capacity exhausted; the media binding is desynchronized"
            )
            self._fault = capacity_error
            self._changed.set()
            raise capacity_error
        self._records.append(record)
        self._next_sequence += 1
        self._outstanding_frames += 1
        self._outstanding_bytes += frame_bytes
        self._changed.set()

    async def receive(self) -> IngressRecord:
        """Claim the next ordered record for the sole consumer."""

        if self._active is not None:
            raise RuntimeError("the previous ingress record is still active")
        while True:
            self._raise_if_faulted()
            if self._records:
                self._fault_if_oldest_expired()
                record = self._records.popleft()
                self._active = record
                return record
            if self._closed:
                raise IngressClosedError("PCM ingress is closed")
            self._changed.clear()
            if self._records or self._fault is not None or self._closed:
                continue
            await self._changed.wait()

    def complete(self, record: IngressRecord) -> None:
        """Release accounting for the exact claimed record."""

        if self._active is not record:
            raise PermissionError("record does not own the active ingress claim")
        self._active = None
        self._outstanding_frames -= 1
        self._outstanding_bytes -= len(record.frame.pcm)
        self._changed.set()

    def close(self) -> None:
        self._closed = True
        self._changed.set()

    def _raise_if_faulted(self) -> None:
        if self._fault is not None:
            raise self._fault

    def _raise_if_unavailable(self) -> None:
        self._raise_if_faulted()
        if self._closed:
            raise IngressClosedError("PCM ingress is closed")

    def _fault_if_oldest_expired(self) -> None:
        if not self._records:
            return
        age_seconds = self._clock() - self._records[0].observed_at
        if age_seconds <= self._max_age_seconds:
            return
        error = IngressOverloadError(
            "PCM ingress queue age exhausted; the media binding is desynchronized"
        )
        self._fault = error
        self._changed.set()
        raise error
