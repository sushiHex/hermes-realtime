"""Public provider-neutral speech boundary."""

from .delivery import (
    DeliveredSpeechConfirmation,
    DeliveredSpeechLedger,
    PlaybackReceipt,
    SpeechDeliveryAdmission,
    SpeechDeliveryStage,
    SpeechLedgerCapacityError,
    SpeechTurnCounts,
)
from .ports import (
    SpeechPlayback,
    SpeechPresenceVerifier,
    StreamingSynthesizer,
    StreamingTranscriber,
    VoiceActivityDetector,
)
from .segmentation import DurationBoundedSynthesizer
from .types import (
    MAX_SPEECH_CHUNK_DURATION_SECONDS,
    AudioFrame,
    ParticipantAudioFrame,
    SpeechChunk,
    SpeechPresence,
    Transcript,
    VoiceActivity,
    WordTiming,
)

__all__ = [
    "AudioFrame",
    "DeliveredSpeechLedger",
    "DeliveredSpeechConfirmation",
    "DurationBoundedSynthesizer",
    "MAX_SPEECH_CHUNK_DURATION_SECONDS",
    "ParticipantAudioFrame",
    "PlaybackReceipt",
    "SpeechChunk",
    "SpeechDeliveryAdmission",
    "SpeechDeliveryStage",
    "SpeechLedgerCapacityError",
    "SpeechTurnCounts",
    "SpeechPlayback",
    "SpeechPresence",
    "SpeechPresenceVerifier",
    "StreamingSynthesizer",
    "StreamingTranscriber",
    "Transcript",
    "VoiceActivity",
    "VoiceActivityDetector",
    "WordTiming",
]
