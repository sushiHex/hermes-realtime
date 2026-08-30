"""Cancellable local Kokoro-ONNX TTS with pinned, verified model assets."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import re
import threading
import unicodedata
import urllib.request
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_realtime import __version__
from hermes_realtime.providers._speech_text import (
    strip_markdown_emphasis_for_speech as _strip_markdown_emphasis_for_speech,
)
from hermes_realtime.providers.kokoro_worker import KokoroWorkerClient
from hermes_realtime.speech import AudioFrame, SpeechChunk, WordTiming

_MODEL_SAMPLE_RATE_HZ = 24_000
_SAMPLE_RATE_HZ = 48_000
_CHANNELS = 1
_MAX_TEXT_CHARS = 4096
_MAX_PCM_BYTES = 64 * 1024 * 1024
_MAX_ACTIVE_TURNS = 4
_MAX_DOWNLOAD_BYTES = 384 * 1024 * 1024
_MAX_PRONUNCIATIONS = 64
_MAX_PRONUNCIATION_TERM_CHARS = 64
_MAX_PRONUNCIATION_SPOKEN_CHARS = 128
_MIN_STREAM_CHUNK_CHARS = 32
_MAX_STREAM_CHUNK_CHARS = 1024
_MODEL_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_VOICE_IDS = frozenset(
    {
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    }
)

_SynthesizePcm = Callable[[str, str, float], bytes]
_WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


@dataclass(frozen=True)
class _PinnedAsset:
    filename: str
    size: int
    sha256: str


_MODEL_ASSET = _PinnedAsset(
    filename="kokoro-v1.0.onnx",
    size=325_532_387,
    sha256="7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
)
_VOICES_ASSET = _PinnedAsset(
    filename="voices-v1.0.bin",
    size=28_214_398,
    sha256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
)


class KokoroSynthesizer:
    """Synthesize each inference segment locally as one cancellable PCM chunk."""

    def __init__(
        self,
        *,
        voice: str = "bf_isabella",
        speed: float = 1.0,
        intra_op_threads: int = 8,
        language: str = "en-gb",
        cache_dir: Path | None = None,
        max_pcm_bytes: int = 16 * 1024 * 1024,
        max_active_turns: int = 4,
        pronunciations: Mapping[str, str] | None = None,
        stream_chunk_chars: int = 400,
        worker_python: Path | None = None,
        synthesize_pcm: _SynthesizePcm | None = None,
    ) -> None:
        if type(voice) is not str or voice not in _VOICE_IDS:
            raise ValueError("voice must be a supported Kokoro voice id")
        if type(speed) is not float or not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be an exact float from 0.5 through 2.0")
        if type(intra_op_threads) is not int or not 1 <= intra_op_threads <= 32:
            raise ValueError("intra_op_threads must be an exact integer between 1 and 32")
        if type(language) is not str or language not in {"en-us", "en-gb"}:
            raise ValueError("language must be exactly 'en-us' or 'en-gb'")
        if cache_dir is not None and not isinstance(cache_dir, Path):
            raise TypeError("cache_dir must be a Path")
        for name, value, ceiling in (
            ("max_pcm_bytes", max_pcm_bytes, _MAX_PCM_BYTES),
            ("max_active_turns", max_active_turns, _MAX_ACTIVE_TURNS),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if not 1 <= value <= ceiling:
                raise ValueError(f"{name} exceeds supported bounds")
        if synthesize_pcm is not None and not callable(synthesize_pcm):
            raise TypeError("synthesize_pcm must be callable")
        if worker_python is not None:
            if not isinstance(worker_python, Path):
                raise TypeError("worker_python must be a Path")
            if not worker_python.is_absolute() or not worker_python.is_file():
                raise ValueError("worker_python must be an existing absolute file")
            if synthesize_pcm is not None:
                raise ValueError("worker_python cannot be combined with synthesize_pcm")
        if type(stream_chunk_chars) is not int:
            raise TypeError("stream_chunk_chars must be an exact integer")
        if not _MIN_STREAM_CHUNK_CHARS <= stream_chunk_chars <= _MAX_STREAM_CHUNK_CHARS:
            raise ValueError("stream_chunk_chars must be between 32 and 1024")
        if pronunciations is not None and not isinstance(pronunciations, Mapping):
            raise TypeError("pronunciations must be a mapping")
        pronunciation_items = tuple((pronunciations or {}).items())
        if len(pronunciation_items) > _MAX_PRONUNCIATIONS:
            raise ValueError("pronunciation lexicon exceeds 64 entries")
        normalized_terms: set[str] = set()
        for term, replacement in pronunciation_items:
            if type(term) is not str or type(replacement) is not str:
                raise TypeError("pronunciation terms and replacements must be strings")
            if (
                not term
                or term != term.strip()
                or len(term) > _MAX_PRONUNCIATION_TERM_CHARS
                or any(unicodedata.category(character).startswith("C") for character in term)
            ):
                raise ValueError("pronunciation term exceeds supported bounds")
            if (
                not replacement
                or replacement != replacement.strip()
                or len(replacement) > _MAX_PRONUNCIATION_SPOKEN_CHARS
                or any(unicodedata.category(character).startswith("C") for character in replacement)
            ):
                raise ValueError("pronunciation replacement exceeds supported bounds")
            normalized = term.casefold()
            if normalized in normalized_terms:
                raise ValueError("pronunciation terms must be case-insensitively unique")
            normalized_terms.add(normalized)

        self._voice = voice
        self._speed = speed
        self._intra_op_threads = intra_op_threads
        self._language = language
        self._cache_dir = cache_dir or Path.home() / ".cache" / "hermes-realtime" / "kokoro"
        self._max_pcm_bytes = max_pcm_bytes
        self._max_active_turns = max_active_turns
        self._stream_chunk_chars = stream_chunk_chars
        ordered_pronunciations = sorted(
            pronunciation_items,
            key=lambda item: (-len(item[0]), item[0].casefold()),
        )
        self._pronunciation_replacements = {
            term.casefold(): replacement for term, replacement in ordered_pronunciations
        }
        alternatives = "|".join(re.escape(term) for term, _ in ordered_pronunciations)
        self._pronunciation_pattern = (
            re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)
            if ordered_pronunciations
            else None
        )
        self._engine: object | None = None
        self._worker_python = worker_python
        self._worker_client: KokoroWorkerClient | None = None
        self._backend_lock = threading.Lock()
        self._uses_default_backend = synthesize_pcm is None
        self._synthesize_pcm = synthesize_pcm or self._default_synthesize_pcm
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro-tts")
        self._operations: dict[str, asyncio.Task[SpeechChunk]] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def available_voices(self) -> tuple[str, ...]:
        return tuple(sorted(_VOICE_IDS))

    @property
    def intra_op_threads(self) -> int:
        return self._intra_op_threads

    @property
    def selected_voice(self) -> str:
        return self._voice

    async def select_voice(self, voice: str) -> None:
        if type(voice) is not str or voice not in _VOICE_IDS:
            raise ValueError("voice must be a supported Kokoro voice id")
        async with self._lock:
            if self._closed:
                raise RuntimeError("Kokoro synthesizer is closed")
            self._voice = voice

    async def warm(self) -> None:
        if self._closed:
            raise RuntimeError("Kokoro synthesizer is closed")
        if not self._uses_default_backend:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._ensure_engine)

    async def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        self._validate_text(text)
        self._validate_turn_id(turn_id)
        text_chunks = self._split_streaming_text(text)
        async with self._lock:
            if self._closed:
                raise RuntimeError("Kokoro synthesizer is closed")
            if turn_id in self._operations:
                raise RuntimeError("turn already owns a Kokoro TTS operation")
            if len(self._operations) >= self._max_active_turns:
                raise RuntimeError("Kokoro TTS operation capacity exhausted")
            voice = self._voice
            operation = asyncio.create_task(
                self._build_chunk(text_chunks[0], turn_id, voice),
                name=f"kokoro-tts-synthesis:{turn_id}:0",
            )
            self._operations[turn_id] = operation
        try:
            for index in range(len(text_chunks)):
                chunk = await operation
                async with self._lock:
                    if self._operations.get(turn_id) is not operation:
                        raise asyncio.CancelledError
                yield chunk
                if index + 1 >= len(text_chunks):
                    break
                async with self._lock:
                    if self._operations.get(turn_id) is not operation:
                        raise asyncio.CancelledError
                    operation = asyncio.create_task(
                        self._build_chunk(text_chunks[index + 1], turn_id, voice),
                        name=f"kokoro-tts-synthesis:{turn_id}:{index + 1}",
                    )
                    self._operations[turn_id] = operation
        except asyncio.CancelledError:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            raise
        finally:
            async with self._lock:
                if self._operations.get(turn_id) is operation:
                    del self._operations[turn_id]

    def _split_streaming_text(self, text: str) -> tuple[str, ...]:
        if len(text) <= self._stream_chunk_chars:
            return (text,)
        sentence_boundaries = tuple(match.end() for match in re.finditer(r"(?<=[.!?;:])\s+", text))
        whitespace_boundaries = tuple(match.end() for match in re.finditer(r"\s+", text))
        chunks: list[str] = []
        start = 0
        while len(text) - start > self._stream_chunk_chars:
            limit = start + self._stream_chunk_chars
            minimum = start + (self._stream_chunk_chars // 2)
            boundary = max(
                (offset for offset in sentence_boundaries if minimum <= offset <= limit),
                default=0,
            )
            if not boundary:
                boundary = max(
                    (offset for offset in whitespace_boundaries if start < offset <= limit),
                    default=limit,
                )
            chunks.append(text[start:boundary])
            start = boundary
        chunks.append(text[start:])
        candidate = tuple(chunks)
        if any(not chunk.strip() for chunk in candidate):
            return (text,)
        if "".join(self._project_spoken_text(chunk) for chunk in candidate) != (
            self._project_spoken_text(text)
        ):
            return (text,)
        return candidate

    def _project_spoken_text(self, text: str) -> str:
        spoken_text = _strip_markdown_emphasis_for_speech(text)
        if self._pronunciation_pattern is None:
            return spoken_text

        def replace_pronunciation(match: re.Match[str]) -> str:
            return self._pronunciation_replacements.get(
                match.group(0).casefold(),
                match.group(0),
            )

        projected = self._pronunciation_pattern.sub(replace_pronunciation, spoken_text)
        if len(projected) > _MAX_TEXT_CHARS:
            raise ValueError("projected speech text exceeds supported bounds")
        return projected

    async def cancel(self, turn_id: str) -> None:
        self._validate_turn_id(turn_id)
        async with self._lock:
            operation = self._operations.pop(turn_id, None)
        if operation is not None:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            operations = tuple(self._operations.values())
            self._operations.clear()
        for operation in operations:
            operation.cancel()
        await asyncio.gather(*operations, return_exceptions=True)
        with self._backend_lock:
            worker_client = self._worker_client
            self._worker_client = None
        if worker_client is not None:
            await asyncio.to_thread(worker_client.close)
        await asyncio.to_thread(lambda: self._executor.shutdown(wait=True, cancel_futures=True))
        self._engine = None

    async def _build_chunk(self, text: str, turn_id: str, voice: str) -> SpeechChunk:
        loop = asyncio.get_running_loop()
        spoken_text = self._project_spoken_text(text)
        pcm = await loop.run_in_executor(
            self._executor,
            self._synthesize_pcm,
            spoken_text,
            voice,
            self._speed,
        )
        if type(pcm) is not bytes:
            raise TypeError("Kokoro backend must return exact PCM bytes")
        if not pcm or len(pcm) > self._max_pcm_bytes // 2:
            raise ValueError("Kokoro PCM output exceeds supported bounds")
        pcm = self._upsample_pcm_for_transport(pcm)
        frame_bytes = (_SAMPLE_RATE_HZ // 100) * _CHANNELS * 2
        remainder = len(pcm) % frame_bytes
        if remainder:
            padding = frame_bytes - remainder
            if len(pcm) + padding > self._max_pcm_bytes:
                raise ValueError("aligned Kokoro PCM output exceeds supported bounds")
            pcm += b"\x00" * padding
        total_samples = len(pcm) // (2 * _CHANNELS)
        timings = self._estimate_word_timings(text, total_samples)
        return SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"kokoro_{uuid.uuid4().hex}",
            text=text,
            audio=AudioFrame(pcm=pcm, sample_rate_hz=_SAMPLE_RATE_HZ, channels=_CHANNELS),
            word_timings=timings,
            timing_source="estimated" if timings else None,
        )

    @staticmethod
    def _estimate_word_timings(text: str, total_samples: int) -> tuple[WordTiming, ...]:
        matches = tuple(_WORD_RE.finditer(text))
        if not matches or len(matches) > 4096 or total_samples < len(matches):
            return ()
        if any(len(match.group(0)) > 256 for match in matches):
            return ()

        weights: list[float] = []
        for index, match in enumerate(matches):
            word = match.group(0)
            syllables = max(1, len(_VOWEL_GROUP_RE.findall(word)))
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            trailing = text[match.end() : next_start]
            pause = 0.0
            if any(mark in trailing for mark in ".!?"):
                pause = 1.2
            elif any(mark in trailing for mark in ",;:"):
                pause = 0.6
            weights.append(float(syllables) + 0.35 + pause)

        total_weight = sum(weights)
        elapsed_weight = 0.0
        previous_end = 0
        timings: list[WordTiming] = []
        for index, (match, weight) in enumerate(zip(matches, weights, strict=True)):
            elapsed_weight += weight
            remaining_words = len(matches) - index - 1
            if remaining_words == 0:
                end_sample = total_samples
            else:
                target = round(total_samples * elapsed_weight / total_weight)
                end_sample = min(
                    max(previous_end + 1, target),
                    total_samples - remaining_words,
                )
            timings.append(
                WordTiming(
                    word=match.group(0),
                    start_sample=previous_end,
                    end_sample=end_sample,
                    text_start=match.start(),
                    text_end=match.end(),
                )
            )
            previous_end = end_sample
        return tuple(timings)

    def _ensure_engine(self) -> object:
        if self._engine is not None:
            return self._engine
        model_path = self._ensure_asset(_MODEL_ASSET)
        voices_path = self._ensure_asset(_VOICES_ASSET)
        if self._worker_python is not None:
            worker = KokoroWorkerClient(
                python_executable=self._worker_python,
                worker_script=Path(__file__).with_name("kokoro_worker.py").resolve(),
                model_path=model_path,
                voices_path=voices_path,
                expected_model_sha256=_MODEL_ASSET.sha256,
                expected_voices_sha256=_VOICES_ASSET.sha256,
                language=self._language,
                max_pcm_bytes=self._max_pcm_bytes,
                attestation_voice=self._voice,
            )
            with self._backend_lock:
                if self._closed:
                    publish_worker = False
                else:
                    self._worker_client = worker
                    publish_worker = True
            if not publish_worker:
                worker.close()
                raise RuntimeError("Kokoro synthesizer is closed")
            try:
                worker.warm()
            except BaseException:
                with self._backend_lock:
                    if self._worker_client is worker:
                        self._worker_client = None
                worker.close()
                raise
            with self._backend_lock:
                if self._closed or self._worker_client is not worker:
                    publish_engine = False
                else:
                    self._engine = worker
                    publish_engine = True
            if not publish_engine:
                worker.close()
                raise RuntimeError("Kokoro synthesizer closed during worker startup")
            return worker
        try:
            module = importlib.import_module("kokoro_onnx")
            runtime = importlib.import_module("onnxruntime")
        except ImportError as error:
            raise RuntimeError("Kokoro requires the 'kokoro-onnx' local dependency") from error
        engine_type = getattr(module, "Kokoro", None)
        from_session = getattr(engine_type, "from_session", None)
        if not callable(engine_type) or not callable(from_session):
            raise RuntimeError("kokoro-onnx Kokoro engine is unavailable")
        session_options = runtime.SessionOptions()
        session_options.intra_op_num_threads = self._intra_op_threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = runtime.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = runtime.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        engine = from_session(session, str(voices_path))
        get_voices = getattr(engine, "get_voices", None)
        voices: Any = get_voices() if callable(get_voices) else None
        if type(voices) is not list or self._voice not in voices:
            raise RuntimeError(f"Kokoro voice is unavailable in pinned assets: {self._voice}")
        self._engine = engine
        return engine

    def _ensure_asset(self, asset: _PinnedAsset) -> Path:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_dir / asset.filename
        if path.exists():
            self._verify_asset(path, asset)
            return path
        partial = path.with_suffix(path.suffix + ".part")
        partial.unlink(missing_ok=True)
        request = urllib.request.Request(
            f"{_MODEL_BASE_URL}/{asset.filename}",
            headers={"User-Agent": f"hermes-realtime/{__version__}"},
        )
        digest = hashlib.sha256()
        total = 0
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > _MAX_DOWNLOAD_BYTES or total > asset.size:
                        raise ValueError("Kokoro asset download exceeds pinned bounds")
                    digest.update(block)
                    out.write(block)
            if total != asset.size or digest.hexdigest() != asset.sha256:
                raise ValueError("Kokoro asset download failed pinned integrity verification")
            partial.replace(path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def _verify_asset(path: Path, asset: _PinnedAsset) -> None:
        if path.stat().st_size != asset.size:
            raise ValueError(f"Kokoro asset has unexpected size: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != asset.sha256:
            raise ValueError(f"Kokoro asset failed integrity verification: {path}")

    def _default_synthesize_pcm(self, text: str, voice: str, speed: float) -> bytes:
        engine: Any = self._ensure_engine()
        if self._worker_client is not None:
            return self._worker_client.synthesize_pcm(text, voice, speed)
        samples, sample_rate = engine.create(
            text,
            voice=voice,
            speed=speed,
            lang=self._language,
        )
        if type(sample_rate) is not int or sample_rate != _MODEL_SAMPLE_RATE_HZ:
            raise ValueError("Kokoro returned an unsupported sample rate")
        numpy = importlib.import_module("numpy")
        array = numpy.asarray(samples, dtype=numpy.float32).reshape(-1)
        if not array.size:
            raise RuntimeError("Kokoro produced no speakable audio")
        clipped = numpy.clip(array, -1.0, 1.0)
        pcm_array = (clipped * 32767.0).astype("<i2", copy=False)
        return bytes(pcm_array.tobytes())

    @staticmethod
    def _upsample_pcm_for_transport(pcm: bytes) -> bytes:
        if len(pcm) % 2:
            raise ValueError("Kokoro PCM output must contain complete int16 samples")
        numpy = importlib.import_module("numpy")
        native = numpy.frombuffer(pcm, dtype="<i2").astype(numpy.int32)
        if not native.size:
            raise ValueError("Kokoro PCM output must not be empty")
        transport = numpy.empty(native.size * 2, dtype="<i2")
        transport[0::2] = native
        if native.size > 1:
            transport[1:-1:2] = ((native[:-1] + native[1:]) // 2).astype("<i2")
        transport[-1] = native[-1]
        return bytes(transport.tobytes())

    @staticmethod
    def _validate_text(text: str) -> None:
        if type(text) is not str:
            raise TypeError("text must be an exact built-in string")
        if not text.strip() or len(text) > _MAX_TEXT_CHARS:
            raise ValueError("text must contain 1 to 4096 characters")

    @staticmethod
    def _validate_turn_id(turn_id: str) -> None:
        if type(turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if not turn_id or len(turn_id) > 128 or not turn_id.isascii():
            raise ValueError("turn_id must contain 1 to 128 ASCII characters")


__all__ = ["KokoroSynthesizer"]
