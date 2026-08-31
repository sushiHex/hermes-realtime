from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "project.yml",
    "Sources/VPIOProbeApp.swift",
    "Sources/ProbeModel.swift",
    "Sources/ProbeView.swift",
    "README.md",
)

REQUIRED_SOURCE_MARKERS = (
    "LocalAudioTrack.createTrack(name:",
    "setAudioProcessingOptions",
    "AudioManager.shared.audioProcessingState",
    "AudioManager.shared.platformVoiceProcessingState",
    "AudioManager.shared.audioSession.isSpeakerOutputPreferred",
    "ConnectOptions(autoSubscribe: false",
    "localParticipant.publish(",
    "candidate.set(subscribed: true)",
    "AVAudioSession.routeChangeNotification",
    "AVAudioSession.interruptionNotification",
    "AVAudioSession.mediaServicesWereResetNotification",
    "AVAudioSession.mediaServicesWereLostNotification",
)

FORBIDDEN_SOURCE_PATTERNS = (
    r"setCategory\s*\(",
    r"setMode\s*\(",
    r"setActive\s*\(",
    r"AVAudioEngine\s*\(",
    r"customConfigureAudioSessionFunc",
    r"set\s*\(\s*engineObservers:",
    r"isAutomaticConfigurationEnabled\s*=\s*false",
)

SECRET_PATTERNS = (
    re.compile(r"wss://[A-Za-z0-9-]+\.[A-Za-z0-9.-]+", re.IGNORECASE),
    re.compile(r"https://[^\s\"']+\.ts\.net", re.IGNORECASE),
    re.compile(r"(?:api[_-]?secret|authorization|bearer)\s*[:=]\s*[\"'][^\"']+", re.IGNORECASE),
)


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> int:
    missing = [relative for relative in REQUIRED_FILES if not (ROOT / relative).is_file()]
    if missing:
        fail(f"missing required probe files: {missing}")

    project = (ROOT / "project.yml").read_text(encoding="utf-8")
    if 'exactVersion: "2.15.3"' not in project:
        fail("LiveKit Swift must be pinned exactly to 2.15.3")
    if "IPHONEOS_DEPLOYMENT_TARGET: 17.0" not in project:
        fail("probe must target iOS 17.0")
    if "SWIFT_VERSION: 6.0" not in project:
        fail("probe must request Swift 6 language mode")

    source = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in REQUIRED_FILES
        if relative.endswith((".swift", ".md", ".yml"))
    )
    for marker in REQUIRED_SOURCE_MARKERS:
        if marker not in source:
            fail(f"missing required source marker: {marker}")
    for pattern in FORBIDDEN_SOURCE_PATTERNS:
        if re.search(pattern, source):
            fail(f"forbidden parallel/session-owner API: {pattern}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(source):
            fail(f"possible committed private endpoint or credential: {pattern.pattern}")

    if source.count("LocalAudioTrack.createTrack(name:") != 1:
        fail("probe must create exactly one local microphone track")
    if "autoGainControl: false" not in source:
        fail("probe must disable AGC in every tested processing request")
    if "echoCancellationMode: .platform" not in source:
        fail("probe must request platform echo cancellation")
    if "guard case .applied = postPublishResult" not in source:
        fail("probe must reject a post-publish processing request that was merely stored")
    if "guard case .applied = result" not in source:
        fail("probe must reject a stored runtime processing request")
    if "AudioManager.shared.isVoiceProcessingAGCEnabled = false" not in source:
        fail("probe must explicitly disable the platform VPIO AGC flag")
    if "private func describe<Mode: Sendable>" not in source:
        fail("Swift 6 requires the SDK component-state generic to retain its Sendable constraint")
    if "@unknown default: \"unknown\"" not in source:
        fail("imported non-frozen result enums require an unknown case")

    print("VPIO probe structural gate passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"VPIO probe structural gate failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
