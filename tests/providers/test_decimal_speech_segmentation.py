"""Numeric streaming boundaries through the actual adapters and CPU TTS wrapper."""

import asyncio

import pytest

from hermes_realtime.conversation import (
    ConversationContextStore,
    ForegroundTurnCoordinator,
    StreamingSpeechLoop,
)
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from hermes_realtime.providers.kokoro import KokoroSynthesizer
from hermes_realtime.providers.ollama import OllamaStreamingInference
from hermes_realtime.speech import DeliveredSpeechLedger, Transcript

_ADAPTERS = (CodexAppServerStreamingInference, OllamaStreamingInference)


def _adapter(kind):
    # Only exercise the actual segmentation method; never resolve a real transport.
    adapter = object.__new__(kind)
    adapter._max_segment_chars = 4096
    return adapter


@pytest.mark.parametrize("kind", _ADAPTERS)
def test_numeric_terminal_period_waits_for_lookahead_but_final_flush_is_preserved(kind):
    adapter = _adapter(kind)
    text = "The synthetic counter reached 12."
    assert adapter._extract_segments(text, final=False) == ([], text)
    assert adapter._extract_segments(text, final=True) == ([text], "")
    assert adapter._extract_segments(text + " Next sample.", final=False) == (
        [text, "Next sample."], "",
    )
    assert adapter._extract_segments("The sample is ready.", final=False) == (
        ["The sample is ready."], "",
    )


class _SegmentedInference:
    def __init__(self, adapter, deltas):
        self.adapter, self.deltas = adapter, deltas

    async def stream(self, snapshot, *, turn_id):
        del snapshot, turn_id
        buffer = ""
        for delta in self.deltas:
            buffer += delta
            segments, buffer = self.adapter._extract_segments(buffer, final=False)
            for segment in segments:
                yield segment
            await asyncio.sleep(0)
        segments, buffer = self.adapter._extract_segments(buffer, final=True)
        assert not buffer
        for segment in segments:
            yield segment

    async def cancel(self, turn_id):
        pass

    async def close(self):
        pass


class _Playback:
    def __init__(self):
        self.texts = []

    async def play(self, chunk, *, is_valid):
        assert is_valid()
        self.texts.append(chunk.text)

    async def cancel(self, turn_id):
        pass


@pytest.mark.parametrize("kind", _ADAPTERS)
@pytest.mark.parametrize("prefix", [
    "It is ", "The synthetic instrument calibration reference measurement is ",
])
@pytest.mark.parametrize("value", ["12.5", "0.25", "-0.5", ".5", "-.5", "\u2212.5"])
@pytest.mark.asyncio
async def test_split_decimal_reaches_synthesis_and_playback_without_space_or_boundary(
    kind, prefix, value,
):
    integral, fractional = value.split(".")
    source = prefix + value + " degrees Celsius."
    deltas = (prefix + integral + ".", fractional + " degrees Celsius.")
    backend_texts = []

    def fake_pcm(text, voice, speed):
        del voice, speed
        backend_texts.append(text)
        return b"\x01\x00" * 240

    synthesizer = KokoroSynthesizer(stream_chunk_chars=128, synthesize_pcm=fake_pcm)
    playback = _Playback()
    loop = StreamingSpeechLoop(
        context=ConversationContextStore(), foreground=ForegroundTurnCoordinator(),
        inference=_SegmentedInference(_adapter(kind), deltas), synthesizer=synthesizer,
        playback=playback, ledger=DeliveredSpeechLedger(),
    )
    try:
        await loop.respond("turn_decimal", Transcript("Synthetic numeric request", final=True))
        assert backend_texts == [source]
        assert playback.texts == [source]
    finally:
        await loop.close()
