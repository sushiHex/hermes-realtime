from __future__ import annotations

import asyncio
import io
import threading
import wave

import pytest

from hermes_realtime.providers import FasterWhisperTranscriber
from hermes_realtime.speech import AudioFrame, Transcript


def _frame(sample: int = 1000) -> AudioFrame:
    pcm = int(sample).to_bytes(2, "little", signed=True) * 480
    return AudioFrame(pcm=pcm, sample_rate_hz=48_000, channels=1)


@pytest.mark.asyncio
async def test_faster_whisper_buffers_exact_pcm_and_emits_one_final_transcript() -> None:
    wav_payloads: list[bytes] = []

    def transcribe_wav(payload: bytes) -> str:
        wav_payloads.append(payload)
        return "  exact final transcript  "

    stt = FasterWhisperTranscriber(
        transcribe_wav=transcribe_wav,
        sample_rate_hz=48_000,
        channels=1,
        max_utterance_bytes=4096,
    )

    assert await stt.push(_frame()) == ()
    assert await stt.push(_frame(2000)) == ()
    transcript = await stt.finish_utterance()

    assert transcript == Transcript(text="exact final transcript", final=True)
    assert len(wav_payloads) == 1
    with wave.open(io.BytesIO(wav_payloads[0]), "rb") as recording:
        assert recording.getframerate() == 48_000
        assert recording.getnchannels() == 1
        assert recording.getsampwidth() == 2
        assert recording.readframes(recording.getnframes()) == _frame().pcm + _frame(2000).pcm


@pytest.mark.asyncio
async def test_faster_whisper_returns_no_transcript_for_expected_blank_audio() -> None:
    stt = FasterWhisperTranscriber(transcribe_wav=lambda _payload: "   ")
    await stt.push(_frame())

    assert await stt.finish_utterance() is None


@pytest.mark.asyncio
async def test_faster_whisper_cancel_joins_owned_transcription_and_discards_result() -> None:
    started = threading.Event()
    release = threading.Event()

    def transcribe_wav(_payload: bytes) -> str:
        started.set()
        assert release.wait(timeout=5)
        return "stale transcript"

    stt = FasterWhisperTranscriber(transcribe_wav=transcribe_wav)
    await stt.push(_frame())
    finishing = asyncio.create_task(stt.finish_utterance())
    assert await asyncio.to_thread(started.wait, 1)
    cancelling = asyncio.create_task(stt.cancel())
    await asyncio.sleep(0)
    assert not cancelling.done()

    release.set()
    await cancelling
    with pytest.raises(RuntimeError, match="cancelled"):
        await finishing

    with pytest.raises(RuntimeError, match="no buffered audio"):
        await stt.finish_utterance()
