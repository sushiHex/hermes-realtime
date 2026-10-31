from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hermes_realtime.providers import _speech_text
from hermes_realtime.providers.edge_tts import EdgeTtsSynthesizer
from hermes_realtime.providers.kokoro import KokoroSynthesizer


@pytest.mark.parametrize("source, local, expected", [
    ("2026-10-31T02:15:00Z", (2026, 10, 30, 19, 15, -7, "Pacific Daylight Time"),
     "October 30, 2026 at 7:15 PM Pacific Daylight Time"),
    ("2026-03-08T09:30:00+00:00", (2026, 3, 8, 1, 30, -8, "Pacific Standard Time"),
     "March 8, 2026 at 1:30 AM Pacific Standard Time"),
    ("2026-03-08T10:30:00Z", (2026, 3, 8, 3, 30, -7, "Pacific Daylight Time"),
     "March 8, 2026 at 3:30 AM Pacific Daylight Time"),
])
@pytest.mark.asyncio
async def test_local_timestamp_projection_reaches_both_backends_preserving_source(
    monkeypatch, source, local, expected,
) -> None:
    """Fixture local zones witness per-instant conversion, not the machine's zone setting."""
    calls = []
    year, month, day, hour, minute, offset, name = local

    class LocalClock(datetime):
        def astimezone(self, tz=None):
            assert tz is None  # Ask the OS for this instant; never reuse today's offset.
            calls.append(self.isoformat())
            return datetime(year, month, day, hour, minute,
                            tzinfo=timezone(timedelta(hours=offset), name))

    monkeypatch.setattr(_speech_text, "datetime", LocalClock, raising=False)
    supplied = f"Observed {source}."
    spoken = f"Observed {expected}."
    kokoro_calls = []

    def pcm(text, _voice, _speed):
        kokoro_calls.append(text)
        return b"\x01\x00" * 240

    kokoro = KokoroSynthesizer(synthesize_pcm=pcm)
    edge_calls = []

    async def mp3(text, _voice):
        edge_calls.append(text)
        return b"synthetic-mp3"

    async def decode(_payload):
        return b"\x01\x00" * 480

    edge = EdgeTtsSynthesizer(synthesize_mp3=mp3, decode_mp3=decode)
    try:
        for provider in (kokoro, edge):
            chunk = await anext(provider.synthesize(supplied, "turn_local_time"))
            assert chunk.text == supplied
            assert provider.project_spoken_text(supplied) == spoken
        assert kokoro_calls == edge_calls == [spoken]
        assert calls and set(calls) == {datetime.fromisoformat(source).isoformat()}
    finally:
        await kokoro.close()
        await edge.close()


@pytest.mark.parametrize("source", [
    "2026-10-30T18:00:00", "Oct 30, 6 PM", "2026-02-30T18:00:00Z",
    "2026-10-30T18:00:00+24:00", "2026-10-30T18:00:00.123Z",
    "2026-10-30T18:00:00Zsuffix", "https://example.test/2026-10-30T18:00:00Z",
    "CA is ambiguous; CPU is a technical abbreviation.",
])
def test_ambiguous_invalid_and_embedded_timestamps_remain_exact(source) -> None:
    assert EdgeTtsSynthesizer.project_spoken_text(source) == source


def test_timestamp_speech_expansion_keeps_existing_text_bound() -> None:
    with pytest.raises(ValueError, match="projected speech text"):
        EdgeTtsSynthesizer.project_spoken_text("2026-10-30T18:00:00Z " * 200)
