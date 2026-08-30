"""Opt-in real-browser self-acceptance through the production local host."""

from __future__ import annotations

import asyncio
import hashlib
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

import pytest

try:
    from playwright.async_api import Browser, async_playwright, expect
except ModuleNotFoundError:
    Browser = Any
    async_playwright = None
    expect = None

from hermes_realtime.conversation import ConversationContextSnapshot
from hermes_realtime.host_launcher import build_local_host_launcher
from hermes_realtime.speech import (
    AudioFrame,
    SpeechChunk,
    SpeechPresence,
    Transcript,
    VoiceActivity,
)
from tests.support.qualification import InProcessQualificationComposition

_PINNED_LIVEKIT_SHA256 = "4d60c4043c8c6ff34845727587c7a7f86946d92c390b879ea35ad3793fcbd916"

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        sys.platform != "win32", reason="system Chrome qualification is Windows-only"
    ),
    pytest.mark.skipif(
        os.getenv("HERMES_REALTIME_BROWSER_SELF_ACCEPTANCE") != "1",
        reason="set HERMES_REALTIME_BROWSER_SELF_ACCEPTANCE=1 for the opt-in browser gate",
    ),
]


class _Inference:
    def stream(self, snapshot: ConversationContextSnapshot, *, turn_id: str) -> AsyncIterator[str]:
        return self._stream(snapshot, turn_id)

    async def _stream(
        self, snapshot: ConversationContextSnapshot, turn_id: str
    ) -> AsyncIterator[str]:
        del snapshot, turn_id
        yield "Browser qualification response."

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return None


class _Synthesizer:
    def synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        return self._synthesize(text, turn_id)

    async def _synthesize(self, text: str, turn_id: str) -> AsyncIterator[SpeechChunk]:
        yield SpeechChunk(
            turn_id=turn_id,
            chunk_id=f"browser-{turn_id}",
            text=text,
            audio=AudioFrame(pcm=_pcm(), sample_rate_hz=48_000, channels=1),
        )

    async def cancel(self, turn_id: str) -> None:
        del turn_id

    async def close(self) -> None:
        return None


class _Transcriber:
    async def push(self, frame: AudioFrame) -> tuple[Transcript, ...]:
        del frame
        return ()

    async def finish_utterance(self) -> Transcript | None:
        return None

    async def cancel(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _Vad:
    required_pre_roll_frames = 0

    def process(self, frame: AudioFrame) -> VoiceActivity:
        del frame
        return VoiceActivity.SILENCE


class _Presence:
    def classify(self, frames: tuple[AudioFrame, ...]) -> SpeechPresence:
        del frames
        return SpeechPresence.CONFIRMED_SPEECH

    async def close(self) -> None:
        return None


def _pcm() -> bytes:
    return (8_000).to_bytes(2, "little", signed=True) * 480


def _available_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _system_chrome() -> Path:
    candidates = (
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        / "Google/Chrome/Application/chrome.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.fail("system Chrome missing")


def _wait_for_livekit(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail("test-owned LiveKit exited before readiness")
        try:
            with urllib.request.urlopen("http://127.0.0.1:7880/", timeout=1) as response:
                if response.read() == b"OK":
                    if _livekit_listener_pid() != process.pid:
                        pytest.fail("test-owned LiveKit does not own the signaling listener")
                    return
        except urllib.error.URLError:
            pass
        time.sleep(0.1)
    pytest.fail("test-owned LiveKit did not become ready")


def _livekit_listener_pid() -> int:
    """Return the sole exact IPv4 loopback listener PID, rejecting ambient listeners."""

    completed = subprocess.run(
        ("netstat", "-ano", "-p", "tcp"),
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail("could not inspect the test-owned LiveKit listener")
    listener_pids: list[int] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != "TCP":
            continue
        local_address, state, pid = fields[1], fields[3], fields[4]
        if not local_address.endswith(":7880") or state != "LISTENING":
            continue
        if local_address != "127.0.0.1:7880" or not pid.isdecimal():
            pytest.fail("LiveKit listener is not an exact IPv4 loopback listener")
        listener_pids.append(int(pid))
    if len(listener_pids) != 1:
        pytest.fail("expected exactly one test-owned LiveKit listener")
    return listener_pids[0]


def _livekit_responding() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:7880/", timeout=1) as response:
            return response.read() == b"OK"
    except urllib.error.URLError:
        return False


@contextmanager
def _owned_livekit() -> Iterator[None]:
    configured = os.environ.get("HERMES_REALTIME_BROWSER_LIVEKIT_SERVER")
    if configured is None:
        executable = Path(__file__).parents[2] / ".tools/livekit/livekit-server.exe"
        expected_sha256 = _PINNED_LIVEKIT_SHA256
    else:
        executable = Path(configured)
        if not executable.is_absolute():
            pytest.fail("LiveKit executable override must be absolute")
        expected_sha256 = os.environ.get("HERMES_REALTIME_BROWSER_LIVEKIT_SERVER_SHA256", "")
        if len(expected_sha256) != 64:
            pytest.fail("LiveKit executable override requires a SHA-256")
    if not executable.is_file():
        pytest.fail("pinned LiveKit server is missing")
    actual_sha256 = hashlib.sha256(executable.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        pytest.fail("LiveKit executable SHA-256 mismatch")
    if _livekit_responding():
        pytest.fail("refusing to use a pre-existing LiveKit listener")
    environment = os.environ | {"LIVEKIT_KEYS": "devkey: local-" + "x" * 32 + "\n"}
    process = subprocess.Popen(
        [str(executable), "--dev", "--bind", "127.0.0.1"],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_livekit(process)
        yield
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        assert not _livekit_responding()


async def _close_browser(browser: Browser | None) -> None:
    if browser is not None:
        await browser.close()


async def test_system_chrome_typed_turn_advances_remote_audio_and_stops() -> None:
    """Exercise real served DOM, production host, LiveKit playback, and shutdown."""
    if async_playwright is None or expect is None:
        pytest.fail("install the browser-acceptance extra to run the browser gate")
    port, suffix = _available_port(), uuid.uuid4().hex[:10]
    composition = InProcessQualificationComposition()
    registration = composition.compose_full_host(
        lambda: build_local_host_launcher(
            hermes_api_bearer=None,
            browser_port=port,
            room_name=f"browser-acceptance-{suffix}",
            worker_identity=f"worker_{suffix}",
        ),
        inference_factory=_Inference,
        speech_presence_factory=_Presence,
        synthesizer_factory=_Synthesizer,
        transcriber_factory=_Transcriber,
        vad_factory=_Vad,
        identity_factory=lambda: f"browser_verifier_{suffix}",
    )
    running = None
    browser: Browser | None = None
    console_errors: list[str] = []
    with _owned_livekit():
        try:
            running = await composition.start_host(registration)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=str(_system_chrome()),
                    headless=True,
                    args=[
                        "--autoplay-policy=no-user-gesture-required",
                        "--use-fake-device-for-media-stream",
                        "--use-fake-ui-for-media-stream",
                    ],
                )
                page = await browser.new_page()
                def record_console_error(message: object) -> None:
                    if message.type != "error":
                        return
                    location = message.location
                    url = location["url"]
                    # Chrome requests the absent favicon; deterministic composition has no catalog.
                    if url.endswith("/favicon.ico") or url.endswith("/api/v1/models"):
                        return
                    console_errors.append(f"{message.text} @ {url}")

                page.on("console", record_console_error)
                page.on("pageerror", lambda error: console_errors.append(str(error)))
                launch_origin, launch_fragment = urldefrag(running.url)
                assert launch_fragment
                navigated = asyncio.get_running_loop().create_future()

                def record_navigation(frame: object) -> None:
                    if (
                        not navigated.done()
                        and frame == page.main_frame
                        and frame.url.startswith(launch_origin)
                    ):
                        navigated.set_result(None)

                page.on("framenavigated", record_navigation)
                await page.evaluate(
                    "([origin, fragment]) => {"
                    " setTimeout(() => window.location.replace(`${origin}#${fragment}`), 0);"
                    "}",
                    [launch_origin, launch_fragment],
                )
                await asyncio.wait_for(navigated, timeout=30)
                await page.wait_for_load_state("networkidle")
                await page.get_by_role("button", name="Connect").click()
                await expect(page.locator("#typed-input")).to_be_enabled(timeout=30_000)
                await page.locator("#typed-input").fill("browser deterministic typed turn")
                await page.get_by_role("button", name="Send message").click()
                await page.get_by_role("listitem").filter(
                    has_text="Browser qualification response."
                ).wait_for(timeout=30_000)
                remote_audio = page.locator("#remote-audio")
                async with asyncio.timeout(30):
                    while await remote_audio.evaluate("element => element.currentTime") <= 0:
                        await asyncio.sleep(0.1)
                assert console_errors == []
                await page.get_by_role("button", name="Stop session").click()
                await expect(page.locator("#typed-input")).to_be_disabled(timeout=15_000)
                assert await remote_audio.evaluate("element => element.srcObject === null")
        finally:
            await _close_browser(browser)
            if running is not None:
                await composition.close_host(running)
    assert composition.trace.status().trace_complete
