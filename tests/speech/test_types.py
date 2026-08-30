from typing import cast

import pytest

from hermes_realtime.speech import AudioFrame, SpeechChunk, Transcript, WordTiming


@pytest.mark.parametrize(
    ("pcm", "channels"),
    [
        (b"\x00", 1),
        (b"\x00\x00", 2),
        (b"\x00\x00\x00\x00\x00\x00", 2),
    ],
)
def test_audio_frame_rejects_incomplete_interleaved_sample_groups(
    pcm: bytes, channels: int
) -> None:
    with pytest.raises(ValueError, match="complete interleaved"):
        AudioFrame(pcm=pcm, sample_rate_hz=16_000, channels=channels)


def test_audio_frame_accepts_complete_stereo_sample_group() -> None:
    frame = AudioFrame(
        pcm=b"\x00\x00\x01\x00", sample_rate_hz=16_000, channels=2
    )

    assert frame.channels == 2


def test_audio_frame_requires_exact_immutable_field_types() -> None:
    with pytest.raises(TypeError, match="exact bytes"):
        AudioFrame(
            pcm=cast(bytes, bytearray(b"\x00\x00")),
            sample_rate_hz=16_000,
            channels=1,
        )
    with pytest.raises(TypeError, match="exact integer"):
        AudioFrame(pcm=b"\x00\x00", sample_rate_hz=cast(int, True), channels=1)


def test_transcript_rejects_string_subclass_before_method_access() -> None:
    touched = False

    class SideEffectString(str):
        def strip(self, chars: str | None = None) -> str:
            nonlocal touched
            touched = True
            return super().strip(chars)

    with pytest.raises(TypeError):
        Transcript(text=SideEffectString("untrusted"), final=True)

    assert touched is False


def test_speech_chunk_rejects_string_subclass_before_method_access() -> None:
    touched = False

    class SideEffectString(str):
        def strip(self, chars: str | None = None) -> str:
            nonlocal touched
            touched = True
            return super().strip(chars)

    with pytest.raises(TypeError):
        SpeechChunk(
            turn_id=SideEffectString("turn_001"),
            chunk_id="chunk_001",
            text="hello",
            audio=AudioFrame(
                pcm=b"\x01\x00",
                sample_rate_hz=16_000,
                channels=1,
            ),
        )

    assert touched is False


def test_speech_chunk_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="text must not be blank"):
        SpeechChunk(
            turn_id="turn_001",
            chunk_id="chunk_001",
            text="   ",
            audio=AudioFrame(
                pcm=b"\x01\x00",
                sample_rate_hz=16_000,
                channels=1,
            ),
        )


def test_speech_chunk_canonicalizes_bounded_monotonic_word_timings() -> None:
    first = WordTiming(
        word="Hello",
        start_sample=0,
        end_sample=160,
        text_start=0,
        text_end=5,
    )
    second = WordTiming(
        word="world",
        start_sample=160,
        end_sample=320,
        text_start=6,
        text_end=11,
    )
    chunk = SpeechChunk(
        turn_id="turn_001",
        chunk_id="chunk_001",
        text="Hello world.",
        audio=AudioFrame(
            pcm=b"\x01\x00" * 320,
            sample_rate_hz=16_000,
            channels=1,
        ),
        word_timings=(first, second),
        timing_source="estimated",
    )

    object.__setattr__(first, "word", "mutated")

    assert chunk.word_timings == (
        WordTiming("Hello", 0, 160, 0, 5),
        WordTiming("world", 160, 320, 6, 11),
    )
    assert chunk.timing_source == "estimated"


def test_speech_chunk_rejects_invalid_or_unscoped_word_timings() -> None:
    frame = AudioFrame(pcm=b"\x01\x00" * 320, sample_rate_hz=16_000, channels=1)

    with pytest.raises(ValueError, match="timing source"):
        SpeechChunk(
            turn_id="turn_001",
            chunk_id="chunk_001",
            text="Hello",
            audio=frame,
            word_timings=(WordTiming("Hello", 0, 160, 0, 5),),
        )
    with pytest.raises(ValueError, match="monotonic"):
        SpeechChunk(
            turn_id="turn_001",
            chunk_id="chunk_001",
            text="Hello world",
            audio=frame,
            word_timings=(
                WordTiming("Hello", 100, 200, 0, 5),
                WordTiming("world", 150, 250, 6, 11),
            ),
            timing_source="provider",
        )
    with pytest.raises(ValueError, match="audio duration"):
        SpeechChunk(
            turn_id="turn_001",
            chunk_id="chunk_001",
            text="Hello",
            audio=frame,
            word_timings=(WordTiming("Hello", 0, 321, 0, 5),),
            timing_source="estimated",
        )
