import asyncio
from collections.abc import AsyncIterator

import pytest

from hermes_realtime.speech import (
    AudioFrame,
    DurationBoundedSynthesizer,
    SpeechChunk,
    WordTiming,
)


class WordCountDurationSynthesizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.cancelled: list[str] = []

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self.calls.append((text, turn_id))
        duration_seconds = 4 if len(text.split()) > 2 else 1
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"provider_{len(self.calls)}",
            text=text,
            audio=AudioFrame(
                pcm=b"\x01\x00" * (100 * duration_seconds),
                sample_rate_hz=100,
                channels=1,
            ),
        )

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def cancel(self, turn_id: str) -> None:
        self.cancelled.append(turn_id)


@pytest.mark.asyncio
async def test_overlong_timingless_output_is_resynthesized_at_text_boundaries() -> None:
    provider = WordCountDurationSynthesizer()
    synthesizer = DurationBoundedSynthesizer(
        provider,
        max_duration_seconds=3.0,
        initial_segment_chars=4096,
        max_split_depth=4,
        max_provider_attempts=16,
    )
    source = "Alpha beta gamma delta."

    chunks = [chunk async for chunk in synthesizer.synthesize(source, "turn_public")]

    assert "".join(chunk.text for chunk in chunks) == source
    assert all(chunk.turn_id == "turn_public" for chunk in chunks)
    assert all(
        len(chunk.audio.pcm) / (2 * chunk.audio.channels * chunk.audio.sample_rate_hz)
        <= 3.0
        for chunk in chunks
    )
    assert provider.calls[0][0] == source
    assert all(provider_turn != "turn_public" for _, provider_turn in provider.calls)


class ProviderTimedSynthesizer(WordCountDurationSynthesizer):
    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self.calls.append((text, turn_id))
        timings = tuple(
            WordTiming(
                word=word,
                start_sample=index * 100,
                end_sample=(index + 1) * 100,
                text_start=start,
                text_end=start + len(word),
            )
            for index, (word, start) in enumerate(
                (("One", 0), ("two", 4), ("three", 8), ("four", 14))
            )
        )
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id="qualified_provider_chunk",
            text=text,
            audio=AudioFrame(
                pcm=b"\x01\x00" * 400,
                sample_rate_hz=100,
                channels=1,
            ),
            word_timings=timings,
            timing_source="provider",
        )


@pytest.mark.asyncio
async def test_qualified_word_timings_split_pcm_without_resynthesis() -> None:
    provider = ProviderTimedSynthesizer()
    synthesizer = DurationBoundedSynthesizer(
        provider,
        max_duration_seconds=3.0,
        initial_segment_chars=4096,
    )
    source = "One two three four"

    chunks = [chunk async for chunk in synthesizer.synthesize(source, "turn_timed")]

    assert len(provider.calls) == 1
    assert "".join(chunk.text for chunk in chunks) == source
    assert [len(chunk.audio.pcm) // 2 for chunk in chunks] == [300, 100]
    assert all(chunk.timing_source == "provider" for chunk in chunks)


class BlockingSynthesizer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled: list[str] = []

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        del text, turn_id
        self.started.set()
        await asyncio.Event().wait()
        if False:
            yield

    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def cancel(self, turn_id: str) -> None:
        self.cancelled.append(turn_id)


@pytest.mark.asyncio
async def test_consumer_cancellation_cancels_exact_private_provider_turn() -> None:
    provider = BlockingSynthesizer()
    synthesizer = DurationBoundedSynthesizer(provider)
    pending = asyncio.ensure_future(
        anext(synthesizer.synthesize("Hello.", "turn_cancel"))
    )
    await provider.started.wait()

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert len(provider.cancelled) == 1
    assert provider.cancelled[0].startswith("bounded_")
