"""Concrete provider adapters for explicit host-selected profiles."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .codex_app_server import CodexAppServerStreamingInference
from .ollama import OllamaStreamingInference

if TYPE_CHECKING:
    from .echo import PlaybackEchoGuard
    from .edge_tts import EdgeTtsSynthesizer
    from .faster_whisper import FasterWhisperTranscriber
    from .kokoro import KokoroSynthesizer
    from .moonshine import MoonshineStreamingTranscriber
    from .speech_presence import SileroSpeechPresenceVerifier
    from .vad import WebRtcVoiceActivityDetector

_LAZY_PROVIDERS = {
    "EdgeTtsSynthesizer": ("edge_tts", "EdgeTtsSynthesizer"),
    "FasterWhisperTranscriber": ("faster_whisper", "FasterWhisperTranscriber"),
    "KokoroSynthesizer": ("kokoro", "KokoroSynthesizer"),
    "MoonshineStreamingTranscriber": ("moonshine", "MoonshineStreamingTranscriber"),
    "PlaybackEchoGuard": ("echo", "PlaybackEchoGuard"),
    "SileroSpeechPresenceVerifier": ("speech_presence", "SileroSpeechPresenceVerifier"),
    "WebRtcVoiceActivityDetector": ("vad", "WebRtcVoiceActivityDetector"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_PROVIDERS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "CodexAppServerStreamingInference",
    "EdgeTtsSynthesizer",
    "FasterWhisperTranscriber",
    "KokoroSynthesizer",
    "MoonshineStreamingTranscriber",
    "OllamaStreamingInference",
    "PlaybackEchoGuard",
    "SileroSpeechPresenceVerifier",
    "WebRtcVoiceActivityDetector",
]
