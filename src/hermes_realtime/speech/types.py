"""Provider-neutral speech values."""

from dataclasses import dataclass
from enum import StrEnum

MAX_SPEECH_CHUNK_DURATION_SECONDS = 40.0


class VoiceActivity(StrEnum):
    """Turn-boundary observations emitted by a voice activity detector."""

    SILENCE = "silence"
    SPEECH_STARTED = "speech_started"
    SPEECH_CONTINUED = "speech_continued"
    SPEECH_ENDED = "speech_ended"


class SpeechPresence(StrEnum):
    """Independent speech evidence for one bounded acoustic candidate."""

    CONFIRMED_SPEECH = "confirmed_speech"
    CONFIRMED_NON_SPEECH = "confirmed_non_speech"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """One provider-neutral frame of interleaved signed 16-bit PCM audio."""

    pcm: bytes
    sample_rate_hz: int
    channels: int

    def __post_init__(self) -> None:
        if type(self.pcm) is not bytes:
            raise TypeError("pcm must be exact bytes")
        if type(self.sample_rate_hz) is not int:
            raise TypeError("sample_rate_hz must be an exact integer")
        if type(self.channels) is not int:
            raise TypeError("channels must be an exact integer")
        if not self.pcm:
            raise ValueError("pcm must not be empty")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        sample_group_bytes = 2 * self.channels
        if len(self.pcm) % sample_group_bytes:
            raise ValueError(
                "pcm must contain complete interleaved signed 16-bit sample groups"
            )


@dataclass(frozen=True, slots=True)
class ParticipantAudioFrame:
    """One decoded PCM frame bound to its authenticated participant identity."""

    participant_identity: str
    frame: AudioFrame
    track_name: str | None = None

    def __post_init__(self) -> None:
        if type(self.participant_identity) is not str:
            raise TypeError("participant_identity must be an exact built-in string")
        if not self.participant_identity.strip() or len(self.participant_identity) > 128:
            raise ValueError("participant_identity must contain 1 to 128 characters")
        if type(self.frame) is not AudioFrame:
            raise TypeError("frame must be an exact AudioFrame value")
        if self.track_name is not None and (
            type(self.track_name) is not str
            or not self.track_name.strip()
            or len(self.track_name) > 128
        ):
            raise ValueError("track_name must contain 1 to 128 characters or be None")
        object.__setattr__(
            self,
            "frame",
            AudioFrame(
                pcm=self.frame.pcm,
                sample_rate_hz=self.frame.sample_rate_hz,
                channels=self.frame.channels,
            ),
        )


@dataclass(frozen=True, slots=True)
class Transcript:
    """Partial or final speech recognition result."""

    text: str
    final: bool

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("transcript text must be an exact built-in string")
        if type(self.final) is not bool:
            raise TypeError("transcript final marker must be an exact boolean")
        if not self.text.strip():
            raise ValueError("transcript text must not be blank")


@dataclass(frozen=True, slots=True)
class WordTiming:
    """One bounded word range correlated to PCM samples and source text offsets."""

    word: str
    start_sample: int
    end_sample: int
    text_start: int
    text_end: int

    def __post_init__(self) -> None:
        if type(self.word) is not str:
            raise TypeError("word must be an exact built-in string")
        for name, value in (
            ("start_sample", self.start_sample),
            ("end_sample", self.end_sample),
            ("text_start", self.text_start),
            ("text_end", self.text_end),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if not self.word.strip() or len(self.word) > 256:
            raise ValueError("word must contain 1 to 256 characters")
        if not 0 <= self.start_sample < self.end_sample <= (1 << 31) - 1:
            raise ValueError("word sample range is invalid")
        if not 0 <= self.text_start < self.text_end <= 65_536:
            raise ValueError("word text range is invalid")


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    """An independently queueable and cancellable synthesized speech segment."""

    turn_id: str
    chunk_id: str
    text: str
    audio: AudioFrame
    word_timings: tuple[WordTiming, ...] = ()
    timing_source: str | None = None

    def __post_init__(self) -> None:
        if type(self.turn_id) is not str:
            raise TypeError("turn_id must be an exact built-in string")
        if type(self.chunk_id) is not str:
            raise TypeError("chunk_id must be an exact built-in string")
        if type(self.text) is not str:
            raise TypeError("speech chunk text must be an exact built-in string")
        if type(self.audio) is not AudioFrame:
            raise TypeError("audio must be an exact AudioFrame value")
        if type(self.word_timings) is not tuple:
            raise TypeError("word timings must be an exact tuple")
        if self.timing_source is not None and type(self.timing_source) is not str:
            raise TypeError("timing source must be an exact built-in string or None")
        if not self.turn_id.strip():
            raise ValueError("turn_id must not be blank")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be blank")
        if not self.text.strip():
            raise ValueError("speech chunk text must not be blank")
        if self.timing_source not in (None, "estimated", "provider"):
            raise ValueError("timing source is invalid")
        if bool(self.word_timings) != (self.timing_source is not None):
            raise ValueError("word timings and timing source must be present together")
        if len(self.word_timings) > 4096:
            raise ValueError("word timing capacity exceeded")

        canonical: list[WordTiming] = []
        previous_sample_end = 0
        previous_text_end = 0
        total_samples = len(self.audio.pcm) // (2 * self.audio.channels)
        if total_samples > int(
            MAX_SPEECH_CHUNK_DURATION_SECONDS * self.audio.sample_rate_hz
        ):
            raise ValueError("speech chunk audio exceeds maximum duration")
        for candidate in self.word_timings:
            if type(candidate) is not WordTiming:
                raise TypeError("word timings must contain exact WordTiming values")
            timing = WordTiming(
                word=candidate.word,
                start_sample=candidate.start_sample,
                end_sample=candidate.end_sample,
                text_start=candidate.text_start,
                text_end=candidate.text_end,
            )
            if timing.start_sample < previous_sample_end or timing.text_start < previous_text_end:
                raise ValueError("word timings must be monotonic and non-overlapping")
            if timing.end_sample > total_samples:
                raise ValueError("word timing exceeds audio duration")
            if timing.text_end > len(self.text):
                raise ValueError("word timing exceeds speech text")
            canonical.append(timing)
            previous_sample_end = timing.end_sample
            previous_text_end = timing.text_end
        object.__setattr__(self, "word_timings", tuple(canonical))
