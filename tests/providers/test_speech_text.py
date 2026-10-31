from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hermes_realtime.providers import _speech_text
from hermes_realtime.providers.edge_tts import EdgeTtsSynthesizer
from hermes_realtime.providers.kokoro import KokoroSynthesizer


@pytest.mark.parametrize("source, local, expected", [
    ("2026-10-31T02:15:00Z", (2026, 10, 30, 19, 15, -7, "Pacific Daylight Time"),
     "7:15 PM Pacific Daylight Time on October 30, 2026"),
    ("2026-03-08T09:30:00+00:00", (2026, 3, 8, 1, 30, -8, "Pacific Standard Time"),
     "1:30 AM Pacific Standard Time on March 8, 2026"),
    ("2026-03-08T10:30:00Z", (2026, 3, 8, 3, 30, -7, "Pacific Daylight Time"),
     "3:30 AM Pacific Daylight Time on March 8, 2026"),
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


@pytest.mark.parametrize("source, expected", [
    ("It is 80.1°F.", "It is 80.1 degrees Fahrenheit."),
    ("From -5°C to +20 °C.", "From -5 degrees Celsius to +20 degrees Celsius."),
    ("32 ° F equals 0°C.", "32 degrees Fahrenheit equals 0 degrees Celsius."),
    ("−5℃ and 32℉.", "−5 degrees Celsius and 32 degrees Fahrenheit."),
    pytest.param(
        "1°F, -1.00°C and .1°F.",
        "1 degree Fahrenheit, -1.00 degree Celsius and .1 degrees Fahrenheit.",
        id="singular_degrees",
    ),
    ("Grade F; vitamin C; f/8; 80F.", "Grade F; vitamin C; f/8; 80F."),
    ("90° is an angle; °F needs a value.", "90° is an angle; °F needs a value."),
])
@pytest.mark.asyncio
async def test_explicit_temperature_units_are_spoken_without_guessing(source, expected):
    calls = []

    def pcm(text, _voice, _speed):
        calls.append(text)
        return b"\x01\x00" * 240

    kokoro = KokoroSynthesizer(synthesize_pcm=pcm)
    try:
        chunk = await anext(kokoro.synthesize(source, "turn_temperature"))
        assert chunk.text == source
        assert calls == [expected]
        assert EdgeTtsSynthesizer.project_spoken_text(source) == expected
    finally:
        await kokoro.close()


def test_temperature_expansion_keeps_existing_speech_bound():
    with pytest.raises(ValueError, match="projected speech text"):
        EdgeTtsSynthesizer.project_spoken_text("80°F " * 500)
