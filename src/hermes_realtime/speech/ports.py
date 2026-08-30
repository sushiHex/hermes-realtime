"""Interfaces implemented by speech providers."""

from collections.abc import AsyncIterator, Callable
from typing import Protocol

from .types import AudioFrame, SpeechChunk, SpeechPresence, Transcript, VoiceActivity


class VoiceActivityDetector(Protocol):
    """Classify audio frames without owning conversation state."""

    def process(self, frame: AudioFrame) -> VoiceActivity: ...


class SpeechPresenceVerifier(Protocol):
    """Classify one bounded candidate independently of echo and transcript text.

    Implementations must retain no candidate state between calls and must be safe
    to invoke from a worker thread. ``CONFIRMED_NON_SPEECH`` is the only veto;
    callers admit both speech and uncertainty. Provider failures must raise rather
    than fabricate a decision so the conversation owner can report and fail closed.
    """

    def classify(self, frames: tuple[AudioFrame, ...]) -> SpeechPresence: ...


class StreamingTranscriber(Protocol):
    """Receive frames incrementally and finalize one utterance on demand."""

    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]: ...

    async def finish_utterance(self) -> Transcript | None: ...

    async def cancel(self) -> None: ...


class StreamingSynthesizer(Protocol):
    """Yield speech chunks that may be played before synthesis completes."""

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]: ...

    async def cancel(self, turn_id: str) -> None: ...


class SpeechPlayback(Protocol):
    """Deliver synthesized speech through one cancellable playback boundary."""

    async def play(
        self,
        chunk: SpeechChunk,
        *,
        is_valid: Callable[[], bool],
    ) -> None:
        """Play only while ``is_valid`` authorizes the foreground generation."""
        ...

    async def cancel(self, turn_id: str) -> None: ...
