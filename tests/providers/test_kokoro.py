from __future__ import annotations

import asyncio
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

from hermes_realtime import __version__
from hermes_realtime.providers import KokoroSynthesizer
from hermes_realtime.providers import kokoro as kokoro_module
from hermes_realtime.providers.kokoro import _strip_markdown_emphasis_for_speech
from hermes_realtime.speech import AudioFrame, SpeechChunk, WordTiming


def test_kokoro_asset_request_identifies_runtime_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_user_agents: list[str | None] = []

    def reject_download(request: urllib.request.Request, timeout: int) -> None:
        assert timeout == 60
        observed_user_agents.append(request.get_header("User-agent"))
        raise RuntimeError("stop after request inspection")

    monkeypatch.setattr(kokoro_module.urllib.request, "urlopen", reject_download)
    synthesizer = KokoroSynthesizer(cache_dir=tmp_path)
    asset = kokoro_module._PinnedAsset(filename="asset.bin", size=1, sha256="0" * 64)

    with pytest.raises(RuntimeError, match="request inspection"):
        synthesizer._ensure_asset(asset)

    assert observed_user_agents == [f"hermes-realtime/{__version__}"]


@pytest.mark.asyncio
async def test_kokoro_cuda_worker_owns_warm_synthesis_and_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    model_path.write_bytes(b"model")
    voices_path.write_bytes(b"voices")
    calls: list[tuple[str, str, float]] = []

    class FakeWorkerClient:
        def __init__(self, **_kwargs: object) -> None:
            self.warmed = False
            self.closed = False

        def warm(self) -> None:
            self.warmed = True

        def synthesize_pcm(self, text: str, voice: str, speed: float) -> bytes:
            calls.append((text, voice, speed))
            return b"\x01\x00" * 240

        def close(self) -> None:
            self.closed = True

    worker = FakeWorkerClient()
    monkeypatch.setattr(kokoro_module, "KokoroWorkerClient", lambda **_kwargs: worker)
    monkeypatch.setattr(
        KokoroSynthesizer,
        "_ensure_asset",
        lambda _self, asset: model_path if asset.filename.endswith(".onnx") else voices_path,
    )
    tts = KokoroSynthesizer(worker_python=Path(sys.executable))

    await tts.warm()
    chunk = await anext(tts.synthesize("Worker speech.", "turn_worker"))
    await tts.close()

    assert worker.warmed
    assert worker.closed
    assert calls == [("Worker speech.", "bf_isabella", 1.0)]
    assert chunk.text == "Worker speech."
    assert chunk.audio.sample_rate_hz == 48_000


@pytest.mark.asyncio
async def test_kokoro_cuda_close_interrupts_native_synthesis_before_executor_wait(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    model_path.write_bytes(b"model")
    voices_path.write_bytes(b"voices")
    started = threading.Event()
    released = threading.Event()

    class BlockingWorker:
        def __init__(self, **_: object) -> None:
            pass

        def warm(self) -> None:
            pass

        def synthesize_pcm(self, text: str, voice: str, speed: float) -> bytes:
            del text, voice, speed
            started.set()
            released.wait(timeout=5.0)
            if not released.is_set():
                raise RuntimeError("worker close did not interrupt synthesis")
            raise RuntimeError("worker closed")

        def close(self) -> None:
            released.set()

    monkeypatch.setattr(kokoro_module, "KokoroWorkerClient", BlockingWorker)
    monkeypatch.setattr(
        KokoroSynthesizer,
        "_ensure_asset",
        lambda self, asset: model_path if asset.filename.endswith(".onnx") else voices_path,
    )
    synthesizer = KokoroSynthesizer(worker_python=Path(sys.executable))
    operation = synthesizer.synthesize("Hello", "cuda-close")
    next_chunk = asyncio.ensure_future(anext(operation))
    assert await asyncio.to_thread(started.wait, 1.0)

    await asyncio.wait_for(synthesizer.close(), timeout=1.0)

    with pytest.raises((asyncio.CancelledError, RuntimeError, StopAsyncIteration)):
        await next_chunk


@pytest.mark.asyncio
async def test_kokoro_cuda_close_interrupts_worker_warm_before_executor_wait(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    voices_path = tmp_path / "voices.bin"
    model_path.write_bytes(b"model")
    voices_path.write_bytes(b"voices")
    started = threading.Event()
    released = threading.Event()

    class BlockingWarmWorker:
        def __init__(self, **_: object) -> None:
            pass

        def warm(self) -> None:
            started.set()
            released.wait(timeout=5.0)
            raise RuntimeError("worker warm interrupted")

        def synthesize_pcm(self, text: str, voice: str, speed: float) -> bytes:
            del text, voice, speed
            raise AssertionError("synthesis must not start")

        def close(self) -> None:
            released.set()

    monkeypatch.setattr(kokoro_module, "KokoroWorkerClient", BlockingWarmWorker)
    monkeypatch.setattr(
        KokoroSynthesizer,
        "_ensure_asset",
        lambda self, asset: model_path if asset.filename.endswith(".onnx") else voices_path,
    )
    synthesizer = KokoroSynthesizer(worker_python=Path(sys.executable))
    warm_task = asyncio.create_task(synthesizer.warm())
    assert await asyncio.to_thread(started.wait, 1.0)

    await asyncio.wait_for(synthesizer.close(), timeout=1.0)

    with pytest.raises(RuntimeError, match="worker warm interrupted"):
        await warm_task


@pytest.mark.asyncio
async def test_kokoro_live_voice_selection_applies_to_subsequent_operations() -> None:
    calls: list[str] = []

    def synthesize_pcm(text: str, voice: str, speed: float) -> bytes:
        del text, speed
        calls.append(voice)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(synthesize_pcm=synthesize_pcm)
    assert "zf_xiaobei" in tts.available_voices

    await tts.select_voice("zf_xiaobei")
    chunks = [chunk async for chunk in tts.synthesize("Hello.", "turn_voice")]

    assert len(chunks) == 1
    assert tts.selected_voice == "zf_xiaobei"
    assert calls == ["zf_xiaobei"]
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_yields_one_isabella_pcm_chunk_with_exact_text_authority() -> None:
    calls: list[tuple[str, str, float]] = []

    def synthesize_pcm(text: str, voice: str, speed: float) -> bytes:
        calls.append((text, voice, speed))
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(synthesize_pcm=synthesize_pcm)
    chunk = await anext(tts.synthesize("Hello there.", "turn_1"))

    assert chunk == SpeechChunk(
        turn_id="turn_1",
        chunk_id=chunk.chunk_id,
        text="Hello there.",
        audio=AudioFrame(pcm=chunk.audio.pcm, sample_rate_hz=48_000, channels=1),
        word_timings=chunk.word_timings,
        timing_source=chunk.timing_source,
    )
    assert chunk.chunk_id.startswith("kokoro_")
    assert len(chunk.audio.pcm) == 480 * 2
    assert chunk.audio.pcm == b"\x01\x00" * 480
    assert calls == [("Hello there.", "bf_isabella", 1.0)]
    assert chunk.timing_source == "estimated"
    assert chunk.word_timings == (
        WordTiming("Hello", 0, chunk.word_timings[0].end_sample, 0, 5),
        WordTiming(
            "there",
            chunk.word_timings[0].end_sample,
            480,
            6,
            11,
        ),
    )
    assert chunk.word_timings[0].end_sample < 240
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_conditionally_streams_exact_deterministic_text_chunks() -> None:
    calls: list[str] = []
    source = "First sentence stays intact. Second sentence stays intact. Third."

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(
        stream_chunk_chars=48,
        synthesize_pcm=synthesize_pcm,
    )
    chunks = [chunk async for chunk in tts.synthesize(source, "turn_stream")]

    assert calls == [
        "First sentence stays intact. ",
        "Second sentence stays intact. Third.",
    ]
    assert [chunk.text for chunk in chunks] == calls
    assert "".join(chunk.text for chunk in chunks) == source
    assert len({chunk.chunk_id for chunk in chunks}) == 2
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_streaming_is_demand_driven_and_cancel_suppresses_suffix() -> None:
    calls: list[str] = []
    source = "First sentence stays intact. Second sentence stays intact. Third."

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(stream_chunk_chars=48, synthesize_pcm=synthesize_pcm)
    stream = tts.synthesize(source, "turn_stream_cancel")

    first = await anext(stream)
    await asyncio.sleep(0)
    assert first.text == "First sentence stays intact. "
    assert calls == [first.text]

    await tts.cancel("turn_stream_cancel")
    with pytest.raises(asyncio.CancelledError):
        await anext(stream)
    assert calls == [first.text]
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_revoked_completed_operation_cannot_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts = KokoroSynthesizer(synthesize_pcm=lambda _text, _voice, _speed: b"\x01\x00")

    async def build_then_revoke(
        _self: KokoroSynthesizer,
        text: str,
        turn_id: str,
        _voice: str,
    ) -> SpeechChunk:
        async with tts._lock:
            tts._operations.pop(turn_id)
        return SpeechChunk(
            turn_id=turn_id,
            chunk_id="kokoro_revoked",
            text=text,
            audio=AudioFrame(pcm=b"\x01\x00" * 480, sample_rate_hz=48_000, channels=1),
        )

    monkeypatch.setattr(KokoroSynthesizer, "_build_chunk", build_then_revoke)
    stream = tts.synthesize("This completed chunk has lost its publication lease.", "turn_revoked")

    with pytest.raises(asyncio.CancelledError):
        await anext(stream)
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_streaming_avoids_tiny_preferred_sentence_chunks() -> None:
    calls: list[str] = []
    source = "Tiny. This sentence contains enough words to create a useful first chunk. End."

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(stream_chunk_chars=40, synthesize_pcm=synthesize_pcm)
    chunks = [chunk async for chunk in tts.synthesize(source, "turn_stream_minimum")]

    assert calls[0] == "Tiny. This sentence contains enough "
    assert "".join(chunk.text for chunk in chunks) == source
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_streaming_never_emits_whitespace_only_source_chunks() -> None:
    calls: list[str] = []
    source = "x" + (" " * 64)

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(stream_chunk_chars=32, synthesize_pcm=synthesize_pcm)
    chunks = [chunk async for chunk in tts.synthesize(source, "turn_stream_whitespace")]

    assert calls == [source]
    assert [chunk.text for chunk in chunks] == [source]
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_does_not_split_across_markdown_speech_projection() -> None:
    calls: list[str] = []
    source = "Prefix *an emphasized phrase that crosses the streaming boundary safely* suffix."

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(stream_chunk_chars=40, synthesize_pcm=synthesize_pcm)
    chunks = [chunk async for chunk in tts.synthesize(source, "turn_markdown_stream")]

    assert calls == [
        "Prefix an emphasized phrase that crosses the streaming boundary safely suffix."
    ]
    assert [chunk.text for chunk in chunks] == [source]
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_does_not_split_across_multiword_pronunciation() -> None:
    calls: list[str] = []
    source = "Prefix words before Hermes Agent continues safely."

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(
        pronunciations={"Hermes Agent": "Her-mees Agent"},
        stream_chunk_chars=32,
        synthesize_pcm=synthesize_pcm,
    )
    chunks = [chunk async for chunk in tts.synthesize(source, "turn_lexicon_projection")]

    assert calls == ["Prefix words before Her-mees Agent continues safely."]
    assert [chunk.text for chunk in chunks] == [source]
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_does_not_split_a_maximum_length_unbroken_pronunciation() -> None:
    calls: list[str] = []
    source = "x" * 64

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(
        pronunciations={source: "spoken term"},
        stream_chunk_chars=32,
        synthesize_pcm=synthesize_pcm,
    )
    chunks = [chunk async for chunk in tts.synthesize(source, "turn_pronunciation_unbroken")]

    assert calls == ["spoken term"]
    assert [chunk.text for chunk in chunks] == [source]
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_strips_markdown_emphasis_only_from_backend_speech() -> None:
    calls: list[str] = []
    source = "My favorite is Peter Parker from *Into the Spider-Verse*."

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(synthesize_pcm=synthesize_pcm)
    chunk = await anext(tts.synthesize(source, "turn_markdown"))

    assert calls == ["My favorite is Peter Parker from Into the Spider-Verse."]
    assert chunk.text == source
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_applies_pronunciation_lexicon_only_to_backend_speech() -> None:
    calls: list[str] = []
    source = "Hermes uses LiveKit, but LiveKitchen is unrelated. HERMES remains exact."

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(
        pronunciations={"Hermes": "Her-mees", "LiveKit": "Live Kit"},
        synthesize_pcm=synthesize_pcm,
    )
    chunk = await anext(tts.synthesize(source, "turn_pronunciation"))

    assert calls == [
        "Her-mees uses Live Kit, but LiveKitchen is unrelated. Her-mees remains exact."
    ]
    assert chunk.text == source
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_pronunciation_substitutions_do_not_cascade() -> None:
    calls: list[str] = []

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(
        pronunciations={"LiveKit": "Hermes", "Hermes": "Her mees"},
        synthesize_pcm=synthesize_pcm,
    )
    await anext(tts.synthesize("LiveKit and Hermes.", "turn_pronunciation_once"))

    assert calls == ["Hermes and Her mees."]
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_pronunciation_ignores_unicode_regex_casefold_extras() -> None:
    calls: list[str] = []

    def synthesize_pcm(text: str, _voice: str, _speed: float) -> bytes:
        calls.append(text)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(pronunciations={"i": "eye"}, synthesize_pcm=synthesize_pcm)
    await anext(tts.synthesize("i İ ı", "turn_pronunciation_unicode"))

    assert calls == ["eye İ ı"]
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_rejects_aggregate_pronunciation_expansion() -> None:
    tts = KokoroSynthesizer(
        pronunciations={"a": "b" * 128},
        synthesize_pcm=lambda _text, _voice, _speed: b"\x01\x00" * 240,
    )

    with pytest.raises(ValueError, match="projected speech text"):
        await anext(tts.synthesize("a " * 2047, "turn_pronunciation_expansion"))
    await tts.close()


@pytest.mark.parametrize(
    "pronunciations",
    [
        {f"term-{index}": "spoken" for index in range(65)},
        {"": "spoken"},
        {"term": ""},
        {"term\n": "spoken"},
        {"term": "spoken\x00"},
        {"x" * 65: "spoken"},
        {"term": "x" * 129},
        {"Hermes": "one", "hermes": "two"},
    ],
)
def test_kokoro_rejects_unbounded_or_ambiguous_pronunciations(
    pronunciations: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="pronunciation"):
        KokoroSynthesizer(
            pronunciations=pronunciations,
            synthesize_pcm=lambda *_: b"pcm",
        )


@pytest.mark.parametrize(
    ("source", "spoken"),
    [
        ("***important***", "important"),
        ("**bold and _nested_**", "bold and nested"),
        ("snake_case stays", "snake_case stays"),
        (r"escaped \*literal\* stays", r"escaped \*literal\* stays"),
        ("an *unmatched marker", "an *unmatched marker"),
        ("_word_ beside punctuation.", "word beside punctuation."),
    ],
)
def test_markdown_speech_projection_handles_delimiter_edges(
    source: str,
    spoken: str,
) -> None:
    assert _strip_markdown_emphasis_for_speech(source) == spoken


def test_markdown_speech_projection_handles_maximum_unmatched_input_linearly() -> None:
    source = "*" + ("a" * 4095)

    assert _strip_markdown_emphasis_for_speech(source) == source


@pytest.mark.asyncio
async def test_kokoro_linearly_upsamples_native_pcm_for_livekit_transport() -> None:
    native_pcm = b"\x00\x00\xe8\x03\x18\xfc"
    tts = KokoroSynthesizer(synthesize_pcm=lambda *_: native_pcm)

    chunk = await anext(tts.synthesize("Transport.", "turn_transport"))

    assert chunk.audio.sample_rate_hz == 48_000
    assert chunk.audio.pcm[:12] == (
        b"\x00\x00"  # 0
        b"\xf4\x01"  # 500
        b"\xe8\x03"  # 1000
        b"\x00\x00"  # 0
        b"\x18\xfc"  # -1000
        b"\x18\xfc"  # terminal sample
    )
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_cancel_revokes_turn_without_publishing_late_audio() -> None:
    started = threading.Event()
    release = threading.Event()

    def synthesize_pcm(_text: str, _voice: str, _speed: float) -> bytes:
        started.set()
        release.wait(timeout=5)
        return b"\x01\x00" * 240

    tts = KokoroSynthesizer(synthesize_pcm=synthesize_pcm)
    pending = asyncio.create_task(anext(tts.synthesize("Cancelled.", "turn_cancel")))
    await asyncio.to_thread(started.wait, 2)

    await tts.cancel("turn_cancel")
    with pytest.raises(asyncio.CancelledError):
        await pending

    release.set()
    await tts.close()


@pytest.mark.asyncio
async def test_kokoro_rejects_pcm_beyond_livekit_chunk_duration() -> None:
    native_samples = (40 * 24_000) + 1
    tts = KokoroSynthesizer(
        synthesize_pcm=lambda *_: b"\x01\x00" * native_samples,
    )

    with pytest.raises(ValueError, match="duration"):
        await anext(tts.synthesize("A deliberately oversized segment.", "turn_too_long"))

    await tts.close()


def test_kokoro_defaults_to_measured_eight_thread_cpu_session() -> None:
    tts = KokoroSynthesizer(synthesize_pcm=lambda *_: b"pcm")
    assert tts.intra_op_threads == 8
    with pytest.raises(ValueError, match="intra_op_threads"):
        KokoroSynthesizer(intra_op_threads=0, synthesize_pcm=lambda *_: b"pcm")


def test_kokoro_rejects_unknown_voice() -> None:
    with pytest.raises(ValueError, match="voice"):
        KokoroSynthesizer(voice="unknown_voice", synthesize_pcm=lambda *_: b"pcm")
