"""Opt-in typed client for an already running isolated native preview.

Run as ``python -m scripts.native_typed_journey``. Without --execute this only
validates local inputs; it never bootstraps a session or contacts a provider.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import contextlib
import hashlib
import io
import json
import math
import re
import struct
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hermes_realtime.client.projection import _EVENT_KINDS, BrowserPublicEvent
from hermes_realtime.conversation.commands import _CANCEL_PREFIX, _START_PREFIX
from hermes_realtime.conversation.task_proposals import cancel_readback
from hermes_realtime.integration.api import _WEB_REFUSAL_NOTICES, _WEB_UNVERIFIED_NOTICE
from hermes_realtime.protocol.identifiers import TASK_ID as TASK
from scripts.native_run_witness import RUN_ID, NativeRunWitness, require, validate_origin

LABEL = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
TERMINAL = {"completed", "failed", "interrupted"}
# The exact fixed notices the host substitutes when it refuses a natively completed
# web run (integration/api.py _web_completion); never Hermes or model text.
_HOST_REFUSALS = {notice: reason for reason, notice in _WEB_REFUSAL_NOTICES.items()} | {
    _WEB_UNVERIFIED_NOTICE: "unverified",
}
# Retained collect() categories; None means not read (collect() did not run).
_NATIVE_DIAGNOSTIC_TERMINALS = {None, "completed", "failed", "cancelled"}


def _native_category(value: object) -> str | None:
    if value is None:
        return None
    return value if type(value) is str and value in _NATIVE_DIAGNOSTIC_TERMINALS else "other"


_PUBLICATION_TIMING_FIELDS = (
    "firstPublicationMs", "tokenToPublicationMs", "publicationToPlayableMs",
)


# Synthetic-microphone `speak` step (#237). It bypasses the physical microphone and AEC:
# it proves the controlled synthetic path, never source-attributed microphone qualification.
_SPEAK_FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/audio/sapi_hello_can_you_hear_me.wav"
)
_SPEAK_FIXTURE_SHA256 = "8960e6f7223c5d47b0f31f4588f8d327e1bf6398cc9953015a528d906deef1bc"
_SPEAK_WORDS = ("hello", "can", "you", "hear", "me")
_BARGE_TARGET_FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/audio/sapi_long_ocean_story.wav"
)
_BARGE_TARGET_SHA256 = "2d9ef11bd8f8ac01d62c585f70053f21b0de28aab4d2f7cbb5dc400bd17c48c5"
_BARGE_TARGET_WORDS = ("please", "tell", "me", "a", "long", "story", "about", "the", "ocean")
_BARGE_STOP_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/audio/sapi_stop.wav"
_BARGE_STOP_SHA256 = "3ef6a5a2465c5e4669e1998a3cbfdbdf7a15d39e81f6eb5ecd0ee47cdfc84233"
_BARGE_STOP_WORDS = ("stop",)
# Declared up front; never tuned to pass. Milliseconds of worker audio TARGET must have
# played before Stop, the budget after injection end within which audio must cease, and
# the whole quiet interval that must then be observed.
_BARGE_MIN_TARGET_MS, _BARGE_BUDGET_MS, _BARGE_QUIET_MS = 1500, 1500, 500
_BARGE_PRECONDITION_TIMEOUT, _BARGE_INTERRUPT_TIMEOUT = 30.0, 10.0
# Diagnostic post-injection counts of existing projected kinds (receipt key by kind). Counts
# are downstream of the host's echo and playback-authority filtering and never a verdict.
_BARGE_KIND_COUNTS = {
    "echo_suppressed": "echoSuppressedCount",
    "transcript_echo_suppressed": "transcriptEchoSuppressedCount",
    "speech_ended": "speechEndedCount",
    "voice_activity_ended": "voiceActivityEndedCount",
    "barge_in_non_speech_suppressed": "bargeInNonSpeechSuppressedCount",
    "barge_in_verifier_unavailable": "bargeInVerifierUnavailableCount",
    "echo_barge_in_confirmed": "echoBargeInConfirmedCount",
}
_BARGE_COUNT_KEYS = (
    *_BARGE_KIND_COUNTS.values(), "userPartialCount",
    "interruptTargetCount", "interruptEmptyCount", "interruptOtherCount",
)
_TASK_DISPATCH_FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/audio/sapi_task_dispatch.wav"
)
_TASK_DISPATCH_SHA256 = "1068ad9d4119dc34f04a6a8a725469bc067883d3e68403e62f02a37eccf4da77"
_TASK_DISPATCH_WORDS = ("research", "european", "public", "transit", "funding", "and",
                        "compare", "the", "major", "cities")
_TASK_CANCEL_FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests/fixtures/audio/sapi_task_cancel.wav"
)
_TASK_CANCEL_SHA256 = "f24ade6b163ccdafbbebf291cac8724811defecc328ebd260fd6f867c4208c69"
_TASK_CANCEL_WORDS = ("actually", "stop", "that", "research", "i", "don", "t", "need", "it",
                      "anymore")
# Bounds for the spoken-delegation step: dispatch attribution, the quiet wait with the task
# still active, and the observed cancellation of exactly TARGET.
_TASK_DISPATCH_TIMEOUT, _TASK_QUIET_TIMEOUT, _TASK_CANCEL_TIMEOUT = 90.0, 90.0, 30.0
# Confirmed natural spoken cancel with two tasks (`speak_confirm_cancel`, #581). Both tasks are
# started by typed explicit command, so their stored objectives are exactly these; the host's
# readback is computed from them by the package's own rule, never retyped here.
_CONFIRM_OBJECTIVE_A = "Compare rail ticket prices for three European capitals in detail."
_CONFIRM_OBJECTIVE_B = "Summarize recent ocean temperature research in detail."
_CONFIRM_READBACKS = {
    "a": cancel_readback(_CONFIRM_OBJECTIVE_A), "b": cancel_readback(_CONFIRM_OBJECTIVE_B),
}
_CONFIRM_FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/audio"
_CONFIRM_REQUEST_B_SHA256 = "1a62a02660dffe23d3d86986dafa4dcde14edee619a49d5d6f56192067867965"
_CONFIRM_NO_SHA256 = "88adf9fdaf8aa9cff0cb792fc965cedfb576c29caa68580709b68dc48d2ee31a"
_CONFIRM_REQUEST_A_SHA256 = "18ef57088232b525b75a22ecc8a792f195378e71db4404c17734791c61e693e8"
_CONFIRM_YES_SHA256 = "0477f0f8418f933a01340b5156753b44d57583bd04a0d43ad5868a3129b85dae"
_CONFIRM_REQUEST_B_WORDS = ("please", "stop", "the", "ocean", "temperature", "research", "task")
_CONFIRM_REQUEST_A_WORDS = ("please", "stop", "the", "rail", "ticket", "prices", "task")
# Receipt key prefixes, in injection order: request B, No, request A, Yes.
_CONFIRM_CLIPS = ("requestB", "no", "requestA", "yes")
# The bound on each request's reply (readback or not), and the window after No in which no
# cancellation may appear. Declared up front; never tuned to pass.
_CONFIRM_REPLY_TIMEOUT, _CONFIRM_NO_DRAIN_SECONDS = 60.0, 3.0
_CONFIRM_CANCELS = {"cancelling", "interrupted"}
_CONFIRM_LIVE_RUN = {"queued", "running", "waiting_for_approval"}
_SPEAK_RATE, _SPEAK_FRAME_SAMPLES = 48_000, 480  # 10 ms mono frames
_SPEAK_LEAD_FRAMES, _SPEAK_TRAIL_FRAMES = 30, 100  # bounded silence: 0.3 s and 1.0 s
_SPEAK_FRAME_SECONDS = 0.01  # real-time pace
_SPEAK_RMS_THRESHOLD = 300  # a received frame below this RMS is silence
_SPEAK_MIN_AUDIO_MS = 50  # non-silent agent PCM required inside the response window
_SPEAK_SETTLE_SECONDS, _SPEAK_SETTLE_TIMEOUT = 1.0, 20.0
_SPEAK_READY_TIMEOUT, _SPEAK_TURN_TIMEOUT, _SPEAK_DRAIN_SECONDS = 30.0, 60.0, 0.5
# Windows only advance, so a later window can never re-open an earlier one or overlap it.
_SPEAK_WINDOWS = ("settle", "input", "response", "stop", "quiesce", "after", "closed")
_SPEAK_IDLE_MAX_FRAMES, _SPEAK_IDLE_MAX_BYTES = 4_000, 4_000_000  # warm-up silence bounds
_SPEAK_STREAM_MARGIN = 10.0  # seconds beyond the clip's real-time duration
_SPEAK_QUEUE_MS = 200 # AudioSource queue: capture_frame blocks once this much is buffered
_SPEAK_MAX_FIXTURE_BYTES, _SPEAK_MAX_FIXTURE_SECONDS = 1_048_576, 5
_SPEAK_LIFECYCLE = {
    "first_foreground_token", "first_playable_audio", "assistant_text_generated",
    "assistant_turn_completed", "assistant_turn_interrupted",
}


def _speak_words(text: object) -> tuple[str, ...]:
    if type(text) is not str:
        return ()
    return tuple(re.sub(r"[^\w\s]", " ", text.lower()).split())


def load_speak_fixture() -> tuple[tuple[bytes, ...], tuple[str, ...]]:
    """Verify the pinned synthetic clip and return 10 ms 48 kHz PCM frames plus its words."""
    frames, _, _ = _load_clip(_SPEAK_FIXTURE, _SPEAK_FIXTURE_SHA256)
    return frames, _SPEAK_WORDS


def load_barge_fixtures() -> tuple[
    tuple[bytes, ...], tuple[str, ...], tuple[bytes, ...], tuple[int, int], tuple[str, ...]
]:
    """The pinned long-reply request and the pinned Stop clip: (target frames, target words,
    stop frames, stop speech span [first, end) in frames, stop words)."""
    target, _, _ = _load_clip(_BARGE_TARGET_FIXTURE, _BARGE_TARGET_SHA256)
    stop, first, end = _load_clip(_BARGE_STOP_FIXTURE, _BARGE_STOP_SHA256)
    return target, _BARGE_TARGET_WORDS, stop, (first, end), _BARGE_STOP_WORDS


def load_task_fixtures() -> tuple[
    tuple[bytes, ...], tuple[str, ...], tuple[bytes, ...], tuple[int, int], tuple[str, ...]
]:
    """The pinned spoken delegation and the pinned spoken cancellation: (dispatch frames,
    dispatch words, cancel frames, cancel speech span [first, end) in frames, cancel words)."""
    dispatch, _, _ = _load_clip(_TASK_DISPATCH_FIXTURE, _TASK_DISPATCH_SHA256)
    cancel, first, end = _load_clip(_TASK_CANCEL_FIXTURE, _TASK_CANCEL_SHA256)
    return dispatch, _TASK_DISPATCH_WORDS, cancel, (first, end), _TASK_CANCEL_WORDS


def load_confirm_fixtures() -> tuple[tuple[tuple[bytes, ...], tuple[str, ...]], ...]:
    """The four pinned clips in injection order (request B, No, request A, Yes), each as
    (frames, words). Each clip's first speech frame is _SPEAK_LEAD_FRAMES."""
    clips = (
        ("sapi_confirm_request_b.wav", _CONFIRM_REQUEST_B_SHA256, _CONFIRM_REQUEST_B_WORDS),
        ("sapi_confirm_no.wav", _CONFIRM_NO_SHA256, ("no",)),
        ("sapi_confirm_request_a.wav", _CONFIRM_REQUEST_A_SHA256, _CONFIRM_REQUEST_A_WORDS),
        ("sapi_confirm_yes.wav", _CONFIRM_YES_SHA256, ("yes",)),
    )
    loaded = []
    for name, sha256, words in clips:
        frames, first, _ = _load_clip(_CONFIRM_FIXTURES / name, sha256)
        require(first == _SPEAK_LEAD_FRAMES, "journey_speak_fixture")
        loaded.append((frames, words))
    return tuple(loaded)


def _load_clip(path: Path, sha256: str) -> tuple[tuple[bytes, ...], int, int]:
    """Verify one pinned clip; return 10 ms 48 kHz frames and its speech span [first, end)."""
    try:
        with path.open("rb") as stream:
            raw = stream.read(_SPEAK_MAX_FIXTURE_BYTES + 1)
    except OSError:
        raise ValueError("journey_speak_fixture") from None
    # The read is capped, so an oversized file is truncated and can never match the pin.
    require(hashlib.sha256(raw).hexdigest() == sha256, "journey_speak_fixture")
    try:
        with wave.open(io.BytesIO(raw), "rb") as clip:
            shape = (clip.getnchannels(), clip.getsampwidth(), clip.getframerate())
            count = clip.getnframes()
            # Bound the duration before decoding any PCM.
            require(0 < count <= _SPEAK_MAX_FIXTURE_SECONDS * 22_050, "journey_speak_fixture")
            original = struct.unpack(f"<{count}h", clip.readframes(count))
    except (wave.Error, struct.error, EOFError):
        raise ValueError("journey_speak_fixture") from None
    require(shape == (1, 2, 22_050), "journey_speak_fixture")
    upsampled = [
        original[index * 22_050 // _SPEAK_RATE]
        for index in range(len(original) * _SPEAK_RATE // 22_050)
    ]
    silence = bytes(2 * _SPEAK_FRAME_SAMPLES)
    frames = [silence] * _SPEAK_LEAD_FRAMES
    for offset in range(0, len(upsampled), _SPEAK_FRAME_SAMPLES):
        chunk = upsampled[offset : offset + _SPEAK_FRAME_SAMPLES]
        chunk += [0] * (_SPEAK_FRAME_SAMPLES - len(chunk))
        frames.append(struct.pack(f"<{_SPEAK_FRAME_SAMPLES}h", *chunk))
    speech_end = len(frames)
    frames += [silence] * _SPEAK_TRAIL_FRAMES
    return tuple(frames), _SPEAK_LEAD_FRAMES, speech_end


class SpeakTrace:
    """In-memory evidence of one synthetic-microphone step. Only booleans and counts leave it."""

    def __init__(self, prior: frozenset[tuple[str, int]], expected: tuple[str, ...],
                 incarnation: int) -> None:
        self.prior, self.expected, self.incarnation = prior, expected, incarnation
        self.ready = False
        self.started: int | None = None
        self.ended: int | None = None
        self.final_count = 0
        self.first_final: tuple[int, bool] | None = None
        self.unbound = 0
        self.pairs: dict[tuple[str, int], dict[str, Any]] = {}

    def observe(self, kind: str, data: dict, seq: int, pair: tuple[str, int] | None) -> None:
        if kind == "voice_input_ready":
            # Only the incarnation this step activated; a stale, foreign, missing or
            # non-int (bool) value is not readiness.
            value = data.get("mediaIncarnation")
            if type(value) is int and value == self.incarnation:
                self.ready = True
        elif kind == "voice_activity_started" and self.started is None:
            self.started = seq
        elif kind == "voice_activity_ended" and self.ended is None:
            self.ended = seq
        elif kind == "transcript_final" and data.get("role") == "user":
            self.final_count += 1
            if self.first_final is None:
                self.first_final = (seq, _speak_words(data.get("text")) == self.expected)
        elif kind in _SPEAK_LIFECYCLE:
            if pair is None:
                self.unbound += 1
                return
            require(pair in self.pairs or len(self.pairs) < 16, "journey_speak_concurrent")
            record = self.pairs.setdefault(
                pair, {"token": None, "playable": False, "completed": False,
                       "interrupted": False}
            )
            if kind == "first_playable_audio":
                record["playable"] = True
            elif kind == "first_foreground_token" and record["token"] is None:
                record["token"] = seq
            elif kind == "assistant_turn_completed":
                record["completed"] = True
            elif kind == "assistant_turn_interrupted":
                record["interrupted"] = True

    def terminal(self) -> bool:
        return any(row["completed"] or row["interrupted"] for row in self.pairs.values())


class BargeTrace(SpeakTrace):
    """SpeakTrace plus a Stop injected over TARGET, the first reply's exact pair.

    Events at or before the injection's browser-event cursor are the first utterance; later
    ones are attributed to the Stop and to TARGET's exact pair only. Booleans and counts only.
    """

    def __init__(self, prior: frozenset[tuple[str, int]], expected: tuple[str, ...],
                 incarnation: int, stop_words: tuple[str, ...]) -> None:
        super().__init__(prior, expected, incarnation)
        self.stop_words = stop_words
        self.target: tuple[str, int] | None = None
        self.inj_seq: int | None = None
        self.stop_started = 0
        self.stop_finals = 0
        self.stop_matches = False
        self.interrupt_requested = False
        self.foreign_interrupt = False
        self.silenced = False
        self.overlap = False
        self.fresh_token = False
        self.stop_final_seq: int | None = None
        self.replayed = False
        self.premature_fresh = False
        # Diagnostic counts of projected events after the injection cursor. They never
        # change a verdict and hold no text.
        self.counts = dict.fromkeys(_BARGE_COUNT_KEYS, 0)
        self.partial_alnum = False

    def mark_injection(self, cursor: int) -> None:
        self.inj_seq = cursor

    def target_row(self) -> dict[str, Any]:
        assert self.target is not None
        return self.pairs[self.target]

    def target_terminal(self) -> bool:
        return self.target is not None and bool(
            self.target_row()["completed"] or self.target_row()["interrupted"]
        )

    def fresh_pairs(self) -> int:
        return len(self.pairs) - (0 if self.target is None else 1)

    def observe(self, kind: str, data: dict, seq: int, pair: tuple[str, int] | None) -> None:
        if self.inj_seq is None or seq <= self.inj_seq or self.target is None:
            super().observe(kind, data, seq, pair)
            return
        target = self.target
        counted = _BARGE_KIND_COUNTS.get(kind)
        if counted is not None:
            self.counts[counted] += 1
        if kind == "voice_activity_started":
            self.stop_started += 1
        elif kind == "transcript_partial" and data.get("role") == "user":
            self.counts["userPartialCount"] += 1
            text = data.get("text")
            # Evaluated in memory; the text itself is never stored.
            if type(text) is str and sum(char.isalnum() for char in text) >= 3:
                self.partial_alnum = True
        elif kind == "transcript_final" and data.get("role") == "user":
            self.stop_finals += 1
            if self.stop_finals == 1:
                self.stop_final_seq = seq
                self.stop_matches = _speak_words(data.get("text")) == self.stop_words
        elif kind == "interrupt_requested":
            # The host publishes an anchorable interrupt_requested only with the full exact
            # authority {turnId, turnGeneration, chunkId, streamId}
            # (host_launcher.py _renderer_yield_event_data, lines 937-942) and otherwise an
            # empty payload (lines 933 and 936). So a generation is never legitimately absent
            # next to a turnId: require the strict pair (exact str turnId, exact int
            # generation) to equal TARGET's. An empty payload is unattributed, so it is not
            # TARGET evidence and not foreign either. Any other value (another turn, True,
            # 1.0, None, a string, an absent generation) is foreign.
            turn, generation = data.get("turnId"), data.get("turnGeneration")
            if not data:
                self.counts["interruptEmptyCount"] += 1
            elif (type(turn) is str and turn == target[0]
                    and type(generation) is int and generation == target[1]):
                self.counts["interruptTargetCount"] += 1
                self.interrupt_requested = True
            else:
                self.counts["interruptOtherCount"] += 1
                self.foreign_interrupt = True
        elif kind == "playback_silenced":
            self.silenced = True
        elif kind in _SPEAK_LIFECYCLE:
            if pair is not None and pair != target:
                # Any other pair active while TARGET is still non-terminal overlaps it.
                if not self.target_terminal():
                    self.overlap = True
                if pair in self.prior:
                    # A replayed earlier pair is never a fresh reply and cannot close the window.
                    self.replayed = True
                elif kind == "first_foreground_token":
                    # A fresh reply is a genuinely new pair whose first token follows the Stop
                    # final; a token before it cannot be the reply to Stop.
                    if self.stop_final_seq is not None and seq > self.stop_final_seq:
                        self.fresh_token = True
                    else:
                        self.premature_fresh = True
            super().observe(kind, data, seq, pair)


class TaskSpeakTrace(SpeakTrace):
    """SpeakTrace for a spoken delegation followed by a spoken cancellation of TARGET.

    Task events up to the cancellation's browser-event cursor go to ``task_log`` (the dispatch
    attribution); later ones are attributed to the cancellation. Microphone input has no
    typed admission, so attribution is by event order around the user finals. Identities stay
    in memory; only booleans and counts leave it.
    """

    def __init__(self, prior: frozenset[tuple[str, int]], expected: tuple[str, ...],
                 incarnation: int, cancel_words: tuple[str, ...]) -> None:
        super().__init__(prior, expected, incarnation)
        self.cancel_words = cancel_words
        self.cancel_cursor: int | None = None
        self.task_log: list[tuple[int, str, str, str]] = []  # (seq, kind, task, status)
        self.task_overflow = False
        self.target: str | None = None
        self.cancel_started = 0
        self.cancel_finals = 0
        self.cancel_final_seq: int | None = None
        self.cancel_matches = False
        self.target_cancel_seq: int | None = None
        self.cancel_stale = False
        self.foreign_cancel = 0
        self.competing_active = 0
        # From the dispatch freeze to the cancel injection: further voice starts and unbound
        # or replayed foreground events are unexpected input (user finals count in final_count).
        self.frozen_seq: int | None = None
        self.late_voice = 0
        self.unexpected_foreground = 0

    def mark_cancel(self, cursor: int) -> None:
        self.cancel_cursor = cursor

    def mark_frozen(self, cursor: int) -> None:
        self.frozen_seq = cursor

    def observe(self, kind: str, data: dict, seq: int, pair: tuple[str, int] | None) -> None:
        post = self.cancel_cursor is not None and seq > self.cancel_cursor
        if kind in {"task_state", "task_result"}:
            task, status = data.get("taskId"), data.get("status")
            if type(task) is str and type(status) is str:
                if post:
                    self._post_task(kind, task, status, seq)
                elif len(self.task_log) < 128:
                    self.task_log.append((seq, kind, task, status))
                else:
                    self.task_overflow = True
            return
        if not post:
            if self.frozen_seq is not None and seq > self.frozen_seq:
                if kind == "voice_activity_started":
                    self.late_voice += 1
                elif kind in _SPEAK_LIFECYCLE and (pair is None or pair in self.prior):
                    self.unexpected_foreground += 1
            super().observe(kind, data, seq, pair)
        elif kind == "voice_activity_started":
            self.cancel_started += 1
        elif kind == "transcript_final" and data.get("role") == "user":
            self.cancel_finals += 1
            if self.cancel_finals == 1:
                self.cancel_final_seq = seq
                self.cancel_matches = _speak_words(data.get("text")) == self.cancel_words

    def _post_task(self, kind: str, task: str, status: str, seq: int) -> None:
        cancelling = kind == "task_state" and status in {"cancelling", "interrupted"}
        if task == self.target:
            if cancelling and self.target_cancel_seq is None:
                # Only a cancellation after the cancel final is attributed to it.
                if self.cancel_final_seq is not None and seq > self.cancel_final_seq:
                    self.target_cancel_seq = seq
                else:
                    self.cancel_stale = True
        elif cancelling:
            self.foreign_cancel += 1
        elif status == "active":
            self.competing_active += 1


class ConfirmCancelTrace(SpeakTrace):
    """Evidence for two typed tasks and four spoken clips on one microphone track.

    Facts are logged in event order with their clip phase: -1 before the first clip's first
    speech frame, then the index of the latest injected clip. An assistant final is reduced in
    memory to which pinned readback it equals exactly, if any; no text is kept. Identities stay
    in memory; only booleans, counts and categories leave it.
    """

    def __init__(self, prior: frozenset[tuple[str, int]], incarnation: int,
                 typed: frozenset[str]) -> None:
        super().__init__(prior, (), incarnation)
        self.typed = typed
        self.marks: list[int] = []
        self.words: list[tuple[str, ...]] = []
        self.log: list[tuple[int, int, str, Any, Any]] = []  # (seq, phase, fact, key, value)
        self.overflow = False
        # The phase in which each pair was first seen; -2 for pairs seen before the step.
        self.first_phase: dict[tuple[str, int], int] = dict.fromkeys(prior, -2)
        # From the freeze to the first clip only the typed commands are expected: a voice
        # start, any other user final, or an unbound or replayed foreground event is not.
        self.frozen_seq: int | None = None
        self.unexpected = 0

    def mark(self, cursor: int, words: tuple[str, ...]) -> None:
        self.marks.append(cursor)
        self.words.append(words)

    def _log(self, seq: int, fact: str, key: Any, value: Any) -> None:
        if len(self.log) < 512:
            self.log.append((seq, len(self.marks) - 1, fact, key, value))
        else:
            self.overflow = True

    def observe(self, kind: str, data: dict[str, Any], seq: int,
                pair: tuple[str, int] | None) -> None:
        if kind == "voice_input_ready":
            super().observe(kind, data, seq, pair)
            return
        phase = len(self.marks) - 1
        if kind in {"task_state", "task_result"}:
            task, status = data.get("taskId"), data.get("status")
            if type(task) is str and type(status) is str:
                self._log(seq, kind, task, status)
            return
        foreground = kind in _SPEAK_LIFECYCLE or kind == "transcript_final"
        if foreground and pair is not None and pair not in self.first_phase:
            if len(self.first_phase) >= 128:
                self.overflow = True
                return
            self.first_phase[pair] = phase
        role, text = data.get("role"), data.get("text")
        if phase < 0:
            if self.frozen_seq is None or seq <= self.frozen_seq:
                return
            if (kind == "voice_activity_started"
                    or (kind == "transcript_final" and role == "user"
                        and (type(text) is not str or text not in self.typed))
                    or (kind in _SPEAK_LIFECYCLE
                        and (pair is None or self.first_phase.get(pair) == -2))):
                self.unexpected += 1
            return
        if kind == "voice_activity_started":
            self._log(seq, "voice", None, None)
        elif kind == "transcript_final" and role == "user":
            self._log(seq, "user", None, _speak_words(text) == self.words[phase])
        elif kind == "transcript_final" and role == "assistant":
            match = next((key for key, value in _CONFIRM_READBACKS.items() if text == value), None)
            self._log(seq, "assistant", pair, match)
        elif kind in _SPEAK_LIFECYCLE:
            self._log(seq, kind, pair, None)

    def cancels(self, task: str) -> list[int]:
        """Every cancelling or interrupted state or result of ``task`` (broad, for refusals)."""
        return [seq for seq, _, fact, key, value in self.log
                if fact in {"task_state", "task_result"} and key == task
                and value in _CONFIRM_CANCELS]

    def first_task_row(self, task: str) -> tuple[int, str, str] | None:
        for seq, _, fact, key, value in self.log:
            if fact in {"task_state", "task_result"} and key == task:
                return seq, fact, value
        return None

    def replayed(self, tasks: set[str], after: int) -> bool:
        return any(fact == "task_state" and value == "active" and key in tasks and seq > after
                   for seq, _, fact, key, value in self.log)

    def clip(self, phase: int) -> tuple[bool, int, bool, int | None]:
        """(voice start before the first final, finals, first final matches, its sequence)."""
        users = [row for row in self.log if row[1] == phase and row[2] == "user"]
        final = users[0][0] if users else None
        voice = any(row[1] == phase and row[2] == "voice" and (final is None or row[0] < final)
                    for row in self.log)
        return voice, len(users), bool(users) and users[0][4] is True, final

    def arming(self, task: str, phase: int) -> int | None:
        """The first cancelling or interrupted task_state of ``task`` after the phase's final."""
        final = self.clip(phase)[3]
        if final is None:
            return None
        return next((seq for seq, _, fact, key, value in self.log
                     if fact == "task_state" and key == task and value in _CONFIRM_CANCELS
                     and seq > final), None)

    def settled(self, phase: int) -> bool:
        return any(row[1] == phase and row[2] in {"assistant_turn_completed",
                                                  "assistant_turn_interrupted"}
                   for row in self.log)

    def reply(self, phase: int, final: int, want: str, other: str) -> str:
        """Classify the one reply turn to a request: target, other, clarified, interrupted,
        incomplete, none or ambiguous. A reply must be one pair first seen in this phase,
        wholly after the request's final."""
        rows = [row for row in self.log
                if row[1] == phase and (row[2] in _SPEAK_LIFECYCLE or row[2] == "assistant")]
        pairs: list[tuple[str, int]] = []
        for seq, _, _, pair, _ in rows:
            if pair is None or self.first_phase.get(pair) != phase or seq < final:
                return "ambiguous"
            if pair not in pairs:
                pairs.append(pair)
        if len(pairs) > 1:
            return "ambiguous"
        if not pairs:
            return "none"
        facts = {row[2] for row in rows}
        if "assistant_turn_interrupted" in facts:
            return "interrupted"
        if "assistant_turn_completed" not in facts:
            return "incomplete"
        # The host speaks its readback after the provider stream closes and refuses model text
        # after it, so a readback is the turn's last assistant final and appears once; model
        # prose may precede it. Any other shape around a readback is refused, never accepted
        # by membership.
        finals = [row[4] for row in rows if row[2] == "assistant"]
        readbacks = [match for match in finals if match is not None]
        if not readbacks:
            return "clarified"
        if len(readbacks) != 1 or finals[-1] is None:
            return "ambiguous"
        return "target" if readbacks[0] == want else "other"


class WorkerAudioMeter:
    """Count non-silent PCM from exactly one remote identity. No audio is retained or written."""

    def __init__(self, rtc: Any, room: Any, worker_identity: str,
                 clock: Any = time.monotonic) -> None:
        self.rtc, self.room, self.worker_identity, self._clock = rtc, room, worker_identity, clock
        self.window = "settle"
        self.windows: dict[str, list[float]] = {}
        self.last_audio_at: float | None = None
        self.ambiguous = False
        # Client-local times only: when each window was entered, and the last non-silent
        # frame observed inside each. Never mixed with server monotonic values.
        self.entered_at: dict[str, float] = {}
        self.last_by_window: dict[str, float] = {}
        self._track_sid: object = None
        self._tasks: list[asyncio.Task] = []

    def advance(self, name: str) -> None:
        require(
            name in _SPEAK_WINDOWS
            and _SPEAK_WINDOWS.index(name) >= _SPEAK_WINDOWS.index(self.window),
            "journey_speak_window",
        )
        self.window = name
        self.entered_at[name] = self._clock()

    def start(self) -> None:
        self.room.on("track_published", self._published)
        self.room.on("track_subscribed", self._subscribed)
        for participant in tuple(self.room.remote_participants.values()):
            for publication in tuple(participant.track_publications.values()):
                self._published(publication, participant)

    def _published(self, publication: Any, participant: Any) -> None:
        if (participant.identity == self.worker_identity
                and publication.kind == self.rtc.TrackKind.KIND_AUDIO):
            publication.set_subscribed(True)

    def _subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        if (participant.identity != self.worker_identity
                or track.kind != self.rtc.TrackKind.KIND_AUDIO):
            return
        # PCM is bound to one actual worker track; a second distinct one is ambiguity.
        sid = getattr(track, "sid", None)
        if self._tasks and sid != self._track_sid:
            self.ambiguous = True
            return
        if self._tasks:
            return
        self._track_sid = sid
        stream = self.rtc.AudioStream(track, sample_rate=_SPEAK_RATE, num_channels=1)
        self._tasks.append(asyncio.get_running_loop().create_task(self._consume(stream)))

    async def _consume(self, stream: Any) -> None:
        try:
            async for event in stream:
                self.observe(event.frame)
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    def observe(self, frame: Any) -> None:
        samples = array.array("h")
        samples.frombytes(bytes(frame.data))
        count = len(samples)
        if count == 0 or frame.sample_rate <= 0 or frame.samples_per_channel <= 0:
            return
        if math.sqrt(sum(sample * sample for sample in samples) / count) < _SPEAK_RMS_THRESHOLD:
            return
        row = self.windows.setdefault(self.window, [0, 0.0])
        row[0] += 1
        row[1] += frame.samples_per_channel * 1000 / frame.sample_rate
        self.last_audio_at = self.last_by_window[self.window] = self._clock()

    def window_totals(self, name: str) -> tuple[int, int]:
        row = self.windows.get(name, [0, 0.0])
        return int(row[0]), int(round(row[1]))

    def totals(self) -> tuple[int, int]:
        return (sum(int(row[0]) for row in self.windows.values()),
                int(round(sum(row[1] for row in self.windows.values()))))

    def quiet_for(self, seconds: float) -> bool:
        return self.last_audio_at is None or self._clock() - self.last_audio_at >= seconds

    def now(self) -> float:
        return float(self._clock())

    def quiet_since(self, entry: float, seconds: float) -> bool:
        """One whole quiet interval since the later of ``entry`` and the last worker audio.

        Unlike quiet_for, no audio so far is not quiet: the interval must still elapse.
        """
        start = entry if self.last_audio_at is None else max(entry, self.last_audio_at)
        return self._clock() - start >= seconds

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


class SyntheticVoice:
    """One microphone track fed by the pinned clip, plus the worker-only audio meter."""

    def __init__(self, rtc: Any, room: Any, worker_identity: str) -> None:
        self.rtc, self.room = rtc, room
        self.meter = WorkerAudioMeter(rtc, room, worker_identity)
        self.source: Any = None
        self.incarnation = 0
        self._idle_task: asyncio.Future | None = None
        self._streaming = False
        self.warmup_failed = False
        self.fresh_closed = False

    async def activate(self, post_media: Any) -> None:
        require(self.source is None, "journey_speak_once")
        rtc = self.rtc
        self.source = rtc.AudioSource(_SPEAK_RATE, 1, queue_size_ms=_SPEAK_QUEUE_MS)
        track = rtc.LocalAudioTrack.create_audio_track("microphone-1", self.source)
        await self.room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        self.incarnation += 1
        reply = await post_media({"mediaIncarnation": self.incarnation})
        require(reply == {"mediaIncarnation": self.incarnation, "version": 1},
                "journey_speak_media")

    async def stream(self, frames: tuple[bytes, ...],
                     marks: dict[int, Any] | None = None) -> None:
        """Feed frames paced in real time. ``marks[i]`` runs just before frame ``i``."""
        require(self.source is not None, "journey_speak_media")
        # Exactly one PCM producer at a time: the warm-up must have settled first.
        require(self._idle_task is None and not self._streaming, "journey_speak_producer")
        self._streaming = True
        try:
            begin = time.monotonic()
            for index, pcm in enumerate(frames):
                delay = begin + index * _SPEAK_FRAME_SECONDS - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                if marks is not None and index in marks:
                    marks[index]()
                await self.source.capture_frame(self.rtc.AudioFrame(
                    data=pcm, sample_rate=_SPEAK_RATE, num_channels=1,
                    samples_per_channel=_SPEAK_FRAME_SAMPLES,
                ))
            await self.source.wait_for_playout()
        finally:
            self._streaming = False

    async def _idle(self) -> None:
        """Warm-up silence on the published track, bounded by frames, bytes and time.

        It only stops being fed; it never closes or replaces the shared AudioSource.
        """
        require(self.source is not None, "journey_speak_media")
        silence = bytes(2 * _SPEAK_FRAME_SAMPLES)
        budget = _SPEAK_READY_TIMEOUT + _SPEAK_SETTLE_TIMEOUT
        limit = min(int(budget / 0.01), _SPEAK_IDLE_MAX_FRAMES,
                    _SPEAK_IDLE_MAX_BYTES // len(silence))
        begin = time.monotonic()
        for index in range(limit):
            if time.monotonic() - begin > budget:
                break
            delay = begin + index * _SPEAK_FRAME_SECONDS - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await self.source.capture_frame(self.rtc.AudioFrame(
                data=silence, sample_rate=_SPEAK_RATE, num_channels=1,
                samples_per_channel=_SPEAK_FRAME_SAMPLES,
            ))

    def start_idle(self) -> None:
        require(self._idle_task is None and not self._streaming, "journey_speak_producer")
        self._idle_task = asyncio.ensure_future(self._idle())

    async def stop_idle(self) -> None:
        task, self._idle_task = self._idle_task, None
        if task is not None:
            task.cancel()
            outcome, = await asyncio.gather(task, return_exceptions=True)
            # Our own cancellation is expected; any other error means the explicit warm-up
            # failed, even if readiness arrived through implicit transport silence. Record
            # it (cleanup must keep running); the caller fails the step.
            if isinstance(outcome, BaseException) and not isinstance(
                outcome, asyncio.CancelledError
            ):
                self.warmup_failed = True

    def sync(self, trace: SpeakTrace | None) -> None:
        # The response window opens when the user's final is first observed.
        if trace is not None and trace.final_count and self.meter.window == "input":
            self.meter.advance("response")
        # A fresh reply's first token closes the quiescence window: later audio belongs
        # to the fresh pair, never to TARGET.
        if isinstance(trace, BargeTrace) and trace.fresh_token and self.meter.window == "quiesce":
            self.meter.advance("after")
            self.fresh_closed = True

    async def aclose(self) -> None:
        await self.meter.aclose()


def read_json(path: Path, cap: int = 65_536) -> Any:
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, "journey_file_bound")
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        raise ValueError("journey_json_invalid") from None


def load_scenario(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    require(
        type(value) is dict
        and set(value) == {"version", "steps"}
        and type(value["version"]) is int
        and value["version"] == 1
        and type(value["steps"]) is list
        and 1 <= len(value["steps"]) <= 12,
        "journey_scenario_shape",
    )
    labels, tasks, speaking = set(), set(), False
    for step in value["steps"]:
        require(
            type(step) is dict
            and {"label", "kind"} <= set(step)
            and not set(step) - {"label", "kind", "text", "task", "wait", "tool"},
            "journey_step_shape",
        )
        label, kind = step["label"], step["kind"]
        require(
            type(label) is str
            and LABEL.fullmatch(label) is not None
            and label not in labels
            and kind in {"dispatch", "status", "recall", "cancel", "natural_cancel",
                         "conversation", "speak", "speak_barge", "speak_task",
                         "speak_confirm_cancel"},
            "journey_step_identity",
        )
        labels.add(label)
        if kind == "cancel":
            require(set(step) == {"label", "kind", "task"}, "journey_cancel_shape")
        elif kind in {"speak", "speak_barge", "speak_task", "speak_confirm_cancel"}:
            # One fixed pinned synthetic clip set, once: no text, task or repeat. A barge and a
            # plain speak share the one microphone track, so a scenario holds only one of them.
            require(set(step) == {"label", "kind"}, "journey_speak_shape")
            require(not speaking, "journey_speak_count")
            speaking = True
            if kind == "speak_barge":
                load_barge_fixtures()
            elif kind == "speak_task":
                load_task_fixtures()
            elif kind == "speak_confirm_cancel":
                load_confirm_fixtures()
            else:
                load_speak_fixture()
            continue
        else:
            require(
                type(step.get("text")) is str and 1 <= len(step["text"].strip()) <= 4096,
                "journey_text_bound",
            )
        if kind == "dispatch":
            require(
                "task" not in step and step.get("wait", "completed") in {"accepted", "completed"},
                "journey_dispatch_shape",
            )
            if "tool" in step:
                require(
                    step["tool"] in {"web_search", "web_extract", "mcp__vram__vram_status"},
                    "journey_expected_tool",
                )
            tasks.add(label)
        elif kind == "conversation":
            require(set(step) == {"label", "kind", "text"}, "journey_conversation_shape")
        elif kind == "natural_cancel":
            require(set(step) == {"label", "kind", "task", "text"}
                    and step["task"] in tasks, "journey_natural_cancel_shape")
            # Natural language only: the typed command grammar is the `cancel` kind.
            folded = step["text"].strip().casefold()
            require(not folded.startswith((_CANCEL_PREFIX, _START_PREFIX)),
                    "journey_natural_cancel_grammar")
        else:
            require(
                step.get("task") in tasks and "wait" not in step and "tool" not in step,
                "journey_task_reference",
            )
    return value["steps"]


def read_journal(path: Path) -> tuple[set[str], int]:
    try:
        value = read_json(path)
    except FileNotFoundError:
        # A fresh native adapter has not created its first run record yet.
        # This never supplies an admitted ID or qualifies a dispatch.
        return set(), 0
    require(
        type(value) is dict
        and set(value) == {"version", "pending", "admitted"}
        and type(value["version"]) is int
        and value["version"] in {1, 2, 3}
        and type(value["pending"]) is list
        and type(value["admitted"]) is list
        and len(value["pending"]) + len(value["admitted"]) <= 64,
        "journey_journal_shape",
    )
    ids = value["admitted"]
    require(
        all(type(item) is str and RUN_ID.fullmatch(item) for item in ids)
        and len(set(ids)) == len(ids),
        "journey_journal_identity",
    )
    return set(ids), len(value["pending"])


class EventTracker:
    """Keep identity only in memory; project receipt fields explicitly."""

    def __init__(self) -> None:
        self.cursor = 0
        self.current: dict[str, Any] | None = None
        self.turns: list[dict[str, Any]] = []
        self.pair: tuple[str, int] | None = None
        self.seen: set[tuple[str, int]] = set()
        self.markers: dict[tuple[str, int], dict[str, float]] = {}
        self.tasks: dict[str, str] = {}
        self.text = ""
        self.user_seen = False
        self.admitted = False
        self.admitted_input: tuple[int, int] | None = None
        self.prior_seen: set[tuple[str, int]] = set()
        # Unbound terminals of this step: pair -> (outcome, admitted-input correlation).
        self.terminals: dict[tuple[str, int], tuple[str, tuple[int, int] | None]] = {}
        # Browser sequence of this step's own input admission, and of the first
        # cancelling/interrupted task_state per task seen during this step.
        self.admission_event: int | None = None
        self.step_cancels: dict[str, int] = {}
        # Tasks whose failed result is exactly a fixed host refusal notice.
        self.host_refusals: dict[str, str] = {}
        # Per-step diagnosis inputs, in memory only; the receipt gets booleans/counts.
        self.records: list[dict[str, Any]] = []
        self.started = 0.0
        # Foreground turns with a first token and no terminal yet; a `speak` step starts only
        # from none. `speak` is the live trace of the synthetic-microphone step, if any.
        self.open_turns: set[tuple[str, int]] = set()
        self.speak: SpeakTrace | None = None

    def begin(self, label: str, sequence: int, text: str) -> None:
        require(len(self.turns) < 12, "journey_turn_bound")
        self.current = {
            "label": label,
            "inputSequence": sequence,
            "outcome": "pending",
            "assistantSegments": 0,
            "firstTokenMs": None,
            "firstPlayableAudioMs": None,
            "tokenToPlayableMs": None,
            "firstPublicationMs": None,
            "tokenToPublicationMs": None,
            "publicationToPlayableMs": None,
            "deliveredSegments": 0,
            "unboundTerminalEvents": 0,
            "terminalBinding": None,
        }
        self.turns.append(self.current)
        self.pair, self.text, self.user_seen, self.admitted = None, text, False, False
        self.admitted_input = None
        self.prior_seen = set(self.seen)
        self.terminals = {}
        self.admission_event = None
        self.step_cancels = {}
        self.records.append({
            "task": None, "admitted": False, "admittedInput": None, "admissionEvent": None,
            "cancels": self.step_cancels, "pair": None, "priorSeen": self.prior_seen,
            "kinds": {"completed": 0, "interrupted": 0}, "terminals": [], "omitted": 0,
        })
        self.started = time.monotonic() * 1000

    @staticmethod
    def _input_correlation(data: dict) -> tuple[int, int] | None:
        sequence, binding = data.get("inputSequence"), data.get("bindingGeneration")
        if (type(sequence) is int and type(binding) is int
                and 1 <= sequence < (1 << 53) and 1 <= binding < (1 << 53)):
            return sequence, binding
        return None

    def _reconcile_terminals(self) -> None:
        # A terminal with no first-token binding (pre-text interruption, or a
        # tokenless natural-cancel turn) binds only through this step's exact
        # admitted (inputSequence, bindingGeneration), never a prior step's pair.
        if (self.current is None or self.admitted_input is None
                or self.current["outcome"] != "pending"):
            return
        for pair, (outcome, correlation) in self.terminals.items():
            if (correlation != self.admitted_input or pair in self.prior_seen
                    or (self.pair is not None and pair != self.pair)):
                continue
            self.pair = pair
            self.current["outcome"] = outcome
            self.current["terminalBinding"] = "admittedInput"
            self.current["unboundTerminalEvents"] -= 1
            return

    def consume(self, batch: object) -> None:
        require(type(batch) is list and len(batch) <= 256, "journey_event_bound")
        for raw in batch:
            require(
                type(raw) is dict and set(raw) == {"sequence", "kind", "monotonicMs", "data"},
                "journey_event_shape",
            )
            seq, kind, clock, data = (raw[k] for k in ("sequence", "kind", "monotonicMs", "data"))
            require(
                type(seq) is int
                and seq == self.cursor + 1
                and kind in _EVENT_KINDS
                and type(clock) in {int, float}
                and math.isfinite(clock)
                and clock >= 0
                and type(data) is dict
                and len(json.dumps(data)) <= 32_768,
                "journey_event_identity",
            )
            event = BrowserPublicEvent(seq, kind, float(clock), data)
            self.cursor = seq
            pair = (data.get("turnId"), data.get("turnGeneration"))
            valid_pair = type(pair[0]) is str and type(pair[1]) is int and pair[1] > 0
            if kind in {"task_state", "task_result"}:
                identity, status = data.get("taskId"), data.get("status")
                if kind == "task_state" and status == "rejected":
                    require(identity is None or (type(identity) is str
                            and TASK.fullmatch(identity) is not None), "journey_task_identity")
                    # A rejected control is not a task admission or terminal outcome.
                    continue
                require(
                    type(identity) is str
                    and TASK.fullmatch(identity) is not None
                    and status in {"active", "cancelling", *TERMINAL},
                    "journey_task_identity",
                )
                require(identity in self.tasks or len(self.tasks) < 12, "journey_task_bound")
                self.tasks[identity] = status
                if (kind == "task_state" and status in {"cancelling", "interrupted"}
                        and self.current is not None):
                    # Measured phase only; cleanup runs with no current step.
                    self.step_cancels.setdefault(identity, seq)
                if kind == "task_result":
                    refusal = (_HOST_REFUSALS.get(data.get("text"))
                               if status == "failed" and type(data.get("text")) is str else None)
                    if refusal is None:
                        self.host_refusals.pop(identity, None)
                    else:
                        self.host_refusals[identity] = refusal
            if valid_pair and kind == "first_foreground_token":
                require(pair in self.open_turns or len(self.open_turns) < 128,
                        "journey_identity_bound")
                self.open_turns.add(pair)
            elif valid_pair and kind in {"assistant_turn_completed", "assistant_turn_interrupted"}:
                self.open_turns.discard(pair)
            if self.speak is not None:
                self.speak.observe(kind, data, seq, pair if valid_pair else None)
            if self.current is None:
                if valid_pair:
                    self.seen.add(pair)
                continue
            if kind == "transcript_final" and data.get("role") == "user":
                require(data.get("text") == self.text, "journey_input_source")
                self.user_seen = True
            if kind == "typed_input_admitted":
                require(
                    data.get("inputSequence") == self.current["inputSequence"],
                    "journey_input_sequence",
                )
                if not self.admitted:
                    self.admitted_input = self._input_correlation(data)
                    self.admission_event = seq
                    self.records[-1].update(admitted=True, admittedInput=self.admitted_input,
                                            admissionEvent=seq)
                self.admitted = True
            if kind in {"first_foreground_token", "first_playable_audio"} and valid_pair:
                require(pair in self.markers or len(self.markers) < 128, "journey_marker_bound")
                self.markers.setdefault(pair, {}).setdefault(kind, event.monotonic_ms)
            if (
                kind == "first_foreground_token"
                and self.pair is None
                and self.user_seen
                and self.admitted
                and valid_pair
                and pair not in self.seen
            ):
                self.pair = pair
            if valid_pair and kind in {
                "assistant_text_generated",
                "assistant_turn_completed",
                "assistant_turn_interrupted",
            }:
                self.seen.add(pair)
                require(len(self.seen) <= 128, "journey_identity_bound")
            if kind in {"assistant_turn_completed", "assistant_turn_interrupted"}:
                outcome = kind.removeprefix("assistant_turn_")
                record = self.records[-1]
                record["kinds"][outcome] += 1
                if len(record["terminals"]) < 8:
                    record["terminals"].append((
                        outcome, pair if valid_pair else None, type(pair[0]) is str,
                        "inputSequence" in data, "bindingGeneration" in data,
                        self._input_correlation(data),
                    ))
                else:
                    record["omitted"] += 1
                if valid_pair and pair == self.pair:
                    if self.current["outcome"] == "pending":
                        self.current["outcome"] = outcome
                        self.current["terminalBinding"] = "token"
                else:
                    require(self.current["unboundTerminalEvents"] < 128,
                            "journey_terminal_bound")
                    self.current["unboundTerminalEvents"] += 1
                    if valid_pair:
                        require(pair in self.terminals or len(self.terminals) < 128,
                                "journey_terminal_bound")
                        self.terminals.setdefault(pair, (outcome, self._input_correlation(data)))
            self._reconcile_terminals()
            self.records[-1]["pair"] = self.pair
            if not valid_pair or pair != self.pair:
                continue
            if kind == "assistant_text_generated":
                self.current["assistantSegments"] += 1
                if (self.current["firstPublicationMs"] is None
                        and event.monotonic_ms >= self.started):
                    self.current["firstPublicationMs"] = event.monotonic_ms - self.started
            elif kind == "transcript_final" and data.get("role") == "assistant":
                self.current["deliveredSegments"] += 1
            for marker, key in (
                ("first_foreground_token", "firstTokenMs"),
                ("first_playable_audio", "firstPlayableAudioMs"),
            ):
                stamp = self.markers.get(pair, {}).get(marker)
                if stamp is not None and stamp >= self.started:
                    self.current[key] = stamp - self.started
            marks = self.markers.get(pair, {})
            token, playable = marks.get("first_foreground_token"), marks.get("first_playable_audio")
            if token is not None and playable is not None and playable >= token:
                self.current["tokenToPlayableMs"] = playable - token
            publication = self.current["firstPublicationMs"]
            if publication is not None:
                relative_token = self.current["firstTokenMs"]
                relative_playable = self.current["firstPlayableAudioMs"]
                if relative_token is not None and publication >= relative_token:
                    self.current["tokenToPublicationMs"] = publication - relative_token
                if relative_playable is not None and relative_playable >= publication:
                    self.current["publicationToPlayableMs"] = relative_playable - publication

    def receipt(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.turns]


def _journey_diagnostics(
    tracker: EventTracker,
    bindings: dict[str, tuple[str, str]],
    pre_cleanup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Content-free per-step facts: booleans, counts and fixed categories only.

    No tuple, turn or task identity, or text leaves memory. Built from state captured
    before cleanup, so cleanup can neither supply nor erase a step's evidence.
    """
    steps = []
    for turn, record in zip(tracker.turns, tracker.records, strict=True):
        admitted = record["admittedInput"]
        cancels, admission = record["cancels"], record["admissionEvent"]
        bound_task = bindings.get(record["task"]) if record["task"] is not None else None
        task = None if bound_task is None else bound_task[0]
        terminals = [
            {
                "kind": outcome,
                "turnIdPresent": has_turn,
                "inputSequencePresent": has_sequence,
                "bindingGenerationPresent": has_binding,
                "inputMatchesAdmitted": None if admitted is None else correlation == admitted,
                "priorPair": pair is not None and pair in record["priorSeen"],
                "bound": pair is not None and pair == record["pair"],
            }
            for outcome, pair, has_turn, has_sequence, has_binding, correlation
            in record["terminals"]
        ]
        steps.append({
            "label": turn["label"],
            "admissionObserved": record["admitted"],
            "admittedInputValid": admitted is not None,
            "terminalKinds": dict(record["kinds"]),
            "terminals": terminals,
            "terminalsOmitted": record["omitted"],
            "targetCancelObserved": None if task is None else task in cancels,
            "targetCancelAfterAdmission": (
                None if task is None
                else task in cancels and admission is not None and cancels[task] > admission
            ),
            "nativeTerminal": (None if record["task"] is None else
                               pre_cleanup.get(record["task"], {}).get("nativeTerminal")),
        })
    return {"version": 1, "steps": steps, "preCleanupTasks": pre_cleanup}


def request_json(
    origin: str, route: str, *, token: str | None = None, payload: dict | None = None
) -> tuple[int, dict]:
    require(
        route in {"stable-bootstrap", "refresh", "events", "input", "media", "stop"},
        "journey_route",
    )
    body = b"" if payload is None else json.dumps(payload).encode()
    headers = {"Origin": validate_origin(origin), "Content-Length": str(len(body))}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    if body:
        headers["Content-Type"] = "application/json"
    if route == "stable-bootstrap":
        headers["Sec-Fetch-Site"] = "same-origin"

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    version = 2 if route == "events" else 1
    req = urllib.request.Request(
        origin.rstrip("/") + f"/api/v{version}/" + route, data=body, headers=headers, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        response = opener.open(req, timeout=8)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read(1_048_577)
        require(len(raw) <= 1_048_576, "journey_http_bound")
        value = json.loads(raw) if raw else {}
        require(type(value) is dict, "journey_http_shape")
        return response.status, value


class Journey:
    def __init__(self, options: argparse.Namespace, credential: dict, bearer: str) -> None:
        self.options, self.credential, self.bearer = options, credential, bearer
        self.tracker = EventTracker()
        self.journal = options.preview_dir / "state/hermes-runs-v1.json"
        self.expiry = time.monotonic() + credential["expiresInSeconds"] / 2
        self.admissions: set[str] = set()
        self.pending = 0
        self.bindings: dict[str, tuple[str, str]] = {}
        self.results: dict[str, dict] = {}
        self.failures: dict[str, dict[str, str]] = {}
        self.sequence = 0
        self.voice: SyntheticVoice | None = None
        self.speak_receipt: dict[str, Any] | None = None

    async def request(self, route: str, payload: dict | None = None) -> dict:
        code, value = await asyncio.to_thread(
            request_json,
            self.options.browser_origin,
            route,
            token=self.credential["token"],
            payload=payload,
        )
        require(code == (202 if route in {"input", "media"} else 200), "journey_http_refused")
        return value

    async def poll(self) -> None:
        ids, self.pending = read_journal(self.journal)
        self.admissions.update(ids)
        require(len(self.admissions) <= 12, "journey_admission_bound")
        if time.monotonic() >= self.expiry:
            value = await self.request("refresh")
            require(
                all(
                    value.get(k) == self.credential[k]
                    for k in ("participantIdentity", "roomName", "workerIdentity")
                ),
                "journey_refresh_binding",
            )
            require(
                type(value.get("token")) is str
                and type(value.get("expiresInSeconds")) is int
                and 1 <= value["expiresInSeconds"] <= 3600,
                "journey_refresh_shape",
            )
            self.credential = value
            self.expiry = time.monotonic() + value["expiresInSeconds"] / 2
        value = await self.request("events", {"after": self.tracker.cursor})
        ids, self.pending = read_journal(self.journal)
        self.admissions.update(ids)
        require(len(self.admissions) <= 12, "journey_admission_bound")
        require(value.get("version") == 2, "journey_event_version")
        self.tracker.consume(value.get("events"))
        if self.voice is not None:
            self.voice.sync(self.tracker.speak)

    async def wait(self, predicate, seconds: float = 60) -> None:
        async with asyncio.timeout(seconds):
            while not predicate():
                await self.poll()
                await asyncio.sleep(0.05)

    async def collect(self, label: str) -> None:
        task, run = self.bindings[label]
        await self.wait(lambda: self.tracker.tasks.get(task) in TERMINAL, 150)
        value = await asyncio.to_thread(
            NativeRunWitness(
                self.bearer,
                run,
                profile=self.options.native_profile,
                api_origin=self.options.api_origin,
            ).read
        )
        expected = {"completed": "completed", "failed": "failed", "interrupted": "cancelled"}
        status = self.tracker.tasks[task]
        if value["terminal"] != expected[status]:
            # The host withholds a natively completed web run as failed (api.py
            # _web_completion). Name that only on its exact fixed public notice.
            refusal = self.tracker.host_refusals.get(task)
            if value["terminal"] == "completed" and status == "failed" and refusal is not None:
                self.failures[label] = {"classification": "host_refused", "refusal": refusal}
                require(False, "journey_task_failed")
            require(False, "journey_terminal_binding")
        self.results[label] = value

    async def _post_media(self, payload: dict) -> dict:
        assert self.speak_receipt is not None
        self.speak_receipt["mediaRequests"] += 1
        return await self.request("media", payload)

    async def _mic_quiet_start(
        self, voice: SyntheticVoice, make_trace: Any
    ) -> tuple[Any, Any, Any]:
        """Require a quiet session (no active task, no open turn, no recent worker audio)."""
        tracker = self.tracker
        # A preceding typed step must be settled, never erased while active or ambiguous. Only
        # its completed per-step observation state is cleared, so its input and text cannot
        # also claim the microphone's user final (journey_input_source). Seen identities,
        # admissions, task baselines and sequence authority are kept.
        require(tracker.current is None or tracker.current["outcome"] != "pending",
                "journey_speak_previous_turn_active")
        tracker.current = None
        tracker.text = ""

        def quiet() -> bool:
            return (all(status in TERMINAL for status in tracker.tasks.values())
                    and not tracker.open_turns and voice.meter.quiet_for(_SPEAK_SETTLE_SECONDS))

        try:
            await self.poll()  # flush events from before this step
            await self.wait(quiet, _SPEAK_SETTLE_TIMEOUT)
        except TimeoutError:
            require(False, "journey_speak_not_quiet")
        before_tasks, before_runs = set(tracker.tasks), set(self.admissions)
        trace = tracker.speak = make_trace(frozenset(tracker.seen), voice.incarnation + 1)
        return trace, before_tasks, before_runs

    async def _mic_ready(self, voice: SyntheticVoice, trace: SpeakTrace,
                         receipt: dict[str, Any]) -> None:
        """Publish the one microphone track, activate media, await this incarnation's
        readiness, then require the post-ready quiet gate."""
        tracker = self.tracker
        try:
            await asyncio.wait_for(voice.activate(self._post_media), _SPEAK_READY_TIMEOUT)
        except TimeoutError:
            require(False, "journey_speak_publish_timeout")
        # The worker confirms readiness only once it receives PCM, so feed bounded
        # silence on the published track until the clip starts.
        voice.start_idle()
        try:
            await self.wait(lambda: trace.ready, _SPEAK_READY_TIMEOUT)
        except TimeoutError:
            require(False, "journey_speak_not_ready")
        # The host schedules the readiness cue after publishing readiness, and audio
        # input is suppressed while it plays. Require one whole local quiet interval,
        # measured from the later of gate entry and the last observed worker audio, so
        # a cue that has not started yet cannot pass for silence. Client clock only.
        entry = voice.meter.now()
        cue_frames = voice.meter.window_totals("settle")[0]
        try:
            await self.wait(
                lambda: not tracker.open_turns
                and voice.meter.quiet_since(entry, _SPEAK_SETTLE_SECONDS),
                _SPEAK_SETTLE_TIMEOUT,
            )
        except TimeoutError:
            gate_failed = True
        else:
            gate_failed = False
        receipt["quietGateMs"] = int((voice.meter.now() - entry) * 1000)
        receipt["cueAudioObserved"] = voice.meter.window_totals("settle")[0] > cue_frames
        require(not gate_failed, "journey_speak_not_quiet")
        await voice.stop_idle()  # before the clip: nothing else is fed from here on
        require(not voice.warmup_failed, "journey_speak_warmup_failed")
        voice.meter.advance("input")

    async def _stream_bounded(self, voice: SyntheticVoice, frames: tuple[bytes, ...],
                              marks: dict[int, Any] | None = None) -> None:
        streaming = asyncio.ensure_future(voice.stream(frames, marks))
        try:
            # Real-time pace plus a fixed margin: a capture that never returns cannot
            # hold the step (or its cleanup) open.
            async with asyncio.timeout(len(frames) * _SPEAK_FRAME_SECONDS
                                       + _SPEAK_STREAM_MARGIN):
                while not streaming.done():
                    await self.poll()
                    await asyncio.sleep(0.05)
                await streaming
        except TimeoutError:
            require(False, "journey_speak_stream_timeout")
        finally:
            if not streaming.done():
                streaming.cancel()
                await asyncio.gather(streaming, return_exceptions=True)

    @staticmethod
    def _require_first_utterance(
        trace: SpeakTrace, receipt: dict[str, Any]
    ) -> tuple[tuple[str, int], dict[str, Any]]:
        """Voice activity ordered, exactly one user final with the exact words, and exactly
        one new foreground pair bound by its first token after that final."""
        first = trace.first_final
        ordered = (trace.started is not None and trace.ended is not None
                   and trace.started < trace.ended
                   and (first is None or first[0] > trace.started))
        receipt.update(
            voiceActivityOrdered=ordered, finalsCount=trace.final_count,
            transcriptMatches=first is not None and first[1],
        )
        require(ordered, "journey_speak_voice_activity")
        require(trace.final_count == 1, "journey_speak_finals")
        require(receipt["transcriptMatches"], "journey_speak_transcript")
        require(trace.unbound == 0, "journey_speak_pair_unbound")
        require(trace.pairs, "journey_speak_no_pair")
        require(len(trace.pairs) == 1, "journey_speak_concurrent")
        (pair, row), = trace.pairs.items()
        assert first is not None
        receipt["pairBound"] = (pair not in trace.prior and row["token"] is not None
                                and row["token"] > first[0])
        require(receipt["pairBound"], "journey_speak_pair_unbound")
        return pair, row

    async def speak_barge(self, step: dict) -> None:
        """Ask for a long reply, then interrupt it with the pinned Stop clip on the same track.

        TARGET is the first reply's exact pair. Stop is injected only once TARGET has played
        at least _BARGE_MIN_TARGET_MS of worker audio and is still running; a TARGET that
        ends first is inconclusive, never an interruption. Pass needs: Stop recognized (a
        fresh voice start and one final with the exact words), TARGET's exact pair
        interrupted and never completed, and worker audio quiescent for a whole interval
        that starts within _BARGE_BUDGET_MS of injection end. Client clock only; windows are
        observed through polls and a bounded AudioSource queue, so the bounds are coarse.
        """
        voice, tracker = self.voice, self.tracker
        require(voice is not None, "journey_speak_unavailable")
        assert voice is not None
        target_frames, target_words, stop_frames, (stop_first, stop_end), stop_words = (
            load_barge_fixtures()
        )
        receipt = self.speak_receipt = {
            "label": step["label"], "kind": "speak_barge", "mediaRequests": 0,
            "voiceActivityOrdered": False, "finalsCount": 0, "transcriptMatches": False,
            "pairBound": False, "cueAudioObserved": False, "quietGateMs": 0,
            "targetAudioMsBeforeStop": 0, "stopInjected": False, "stopVadObserved": False,
            "stopFinalsCount": 0, "stopRecognized": False, "targetInterrupted": None,
            "targetCompleted": None, "interruptEventObserved": False,
            "playbackSilencedObserved": False, "postStopAudioMs": None,
            "quiescenceObserved": False, "quiescenceInconclusive": False,
            "freshReplyObserved": False, "stopLatencyMs": None, "quietTailMs": None,
            "userPartialAtLeast3Alnum": False, "snapshotError": False,
            **dict.fromkeys(_BARGE_COUNT_KEYS, 0),
        }
        trace, before_tasks, before_runs = await self._mic_quiet_start(
            voice,
            lambda prior, incarnation: BargeTrace(prior, target_words, incarnation, stop_words),
        )
        meter = voice.meter

        def snapshot() -> None:
            """Whatever evidence exists, as pure bounded metadata: no I/O, no await, and no
            assumption that a stage was reached. Unreached target flags stay null. It never
            raises, so it can neither change the verdict nor block cleanup."""
            try:
                row = None if trace.target is None else trace.pairs.get(trace.target)
                if row is not None:
                    receipt["targetInterrupted"] = bool(row["interrupted"])
                    receipt["targetCompleted"] = bool(row["completed"])
                if trace.inj_seq is None:
                    receipt["targetAudioMsBeforeStop"] = meter.window_totals("response")[1]
                receipt.update(
                    stopVadObserved=trace.stop_started >= 1,
                    stopFinalsCount=trace.stop_finals,
                    stopRecognized=trace.stop_started >= 1 and trace.stop_finals == 1
                    and trace.stop_matches,
                    interruptEventObserved=trace.interrupt_requested,
                    playbackSilencedObserved=trace.silenced,
                    freshReplyObserved=trace.fresh_token,
                    userPartialAtLeast3Alnum=trace.partial_alnum, **trace.counts,
                )
                if "quiesce" in meter.entered_at:
                    receipt["postStopAudioMs"] = meter.window_totals("quiesce")[1]
                    start = meter.entered_at["quiesce"]
                    last = meter.last_by_window.get("quiesce")
                    tail_start = start if last is None else max(start, last)
                    receipt["stopLatencyMs"] = int(round((tail_start - start) * 1000))
                    if "after" in meter.entered_at:
                        receipt["quietTailMs"] = max(
                            int(round((meter.entered_at["after"] - tail_start) * 1000)), 0
                        )
            except Exception:  # evidence is best effort; the original outcome stands
                receipt["snapshotError"] = True  # the diagnostic metadata is unavailable

        try:
            await self._mic_ready(voice, trace, receipt)
            await self._stream_bounded(voice, target_frames)
            with contextlib.suppress(TimeoutError):
                await self.wait(lambda: any(row["token"] is not None
                                            for row in trace.pairs.values()), _SPEAK_TURN_TIMEOUT)
            pair, row = self._require_first_utterance(trace, receipt)
            trace.target = pair

            def exclusive() -> None:
                # The first-utterance exclusivity, re-validated after every poll so far: still
                # one pair (TARGET), still one user final (its words were checked at binding),
                # and no unbound foreground event.
                require(trace.unbound == 0 and len(trace.pairs) == 1
                        and trace.final_count == 1, "journey_barge_ambiguous")

            def injectable() -> bool:
                return trace.target_terminal() or (
                    row["playable"] and meter.window_totals("response")[1] >= _BARGE_MIN_TARGET_MS
                )

            with contextlib.suppress(TimeoutError):
                await self.wait(injectable, _BARGE_PRECONDITION_TIMEOUT)
            audio_ms = meter.window_totals("response")[1]
            receipt["targetAudioMsBeforeStop"] = audio_ms
            # Audio from an ambiguous window never qualifies TARGET.
            exclusive()
            # A TARGET that ended first can never show an interruption: inconclusive.
            require(not trace.target_terminal(), "journey_barge_target_terminal")
            require(row["playable"] and audio_ms >= _BARGE_MIN_TARGET_MS,
                    "journey_barge_precondition")

            def inject_start() -> None:
                # Re-checked at the first speech frame: TARGET may have ended during the
                # clip's leading silence. Abort before any Stop speech is sent.
                require(not trace.target_terminal(), "journey_barge_target_terminal")
                exclusive()  # and still exactly the first utterance, after every lead-in poll
                trace.mark_injection(tracker.cursor)
                receipt["targetAudioMsBeforeStop"] = meter.window_totals("response")[1]
                receipt["stopInjected"] = True
                meter.advance("stop")

            def inject_end() -> None:
                meter.advance("quiesce")
                voice.sync(trace)

            await self._stream_bounded(
                voice, stop_frames, {stop_first: inject_start, stop_end: inject_end}
            )
            injection_end = meter.entered_at["quiesce"]
            horizon = (_BARGE_BUDGET_MS + _BARGE_QUIET_MS) / 1000
            try:
                await self.wait(
                    lambda: meter.window != "quiesce" or meter.now() - injection_end >= horizon,
                    horizon + _BARGE_INTERRUPT_TIMEOUT,
                )
            except TimeoutError:
                require(False, "journey_barge_quiescence_timeout")
            if meter.window == "quiesce":
                meter.advance("after")  # bounded drain elapsed with no fresh reply
            with contextlib.suppress(TimeoutError):
                await self.wait(trace.target_terminal, _BARGE_INTERRUPT_TIMEOUT)
            await asyncio.sleep(_SPEAK_DRAIN_SECONDS)
            await self.poll()
        finally:
            snapshot()  # first, synchronously: partial evidence survives any early refusal
            await voice.stop_idle()  # every exit path: cancel and settle the idle feeder
            meter.advance("closed")
            tracker.speak = None
        start, close = meter.entered_at["quiesce"], meter.entered_at["after"]
        last = meter.last_by_window.get("quiesce")
        tail_start = start if last is None else max(start, last)
        latency_ms = int(round((tail_start - start) * 1000))
        tail_ms = int(round((close - tail_start) * 1000))
        target = trace.target_row()
        require(trace.stop_started >= 1, "journey_barge_stop_vad")
        require(trace.stop_finals == 1, "journey_barge_stop_finals")
        require(trace.stop_matches, "journey_barge_stop_transcript")
        require(trace.unbound == 0 and not trace.overlap and not trace.foreign_interrupt
                and not trace.replayed and not trace.premature_fresh
                and trace.fresh_pairs() <= 1, "journey_barge_ambiguous")
        require(not target["completed"], "journey_barge_target_completed")
        require(target["interrupted"], "journey_barge_target_not_interrupted")
        # Acceptance reads the validated trace, never the best-effort receipt snapshot.
        require(trace.interrupt_requested, "journey_barge_no_interrupt_event")
        require(set(tracker.tasks) == before_tasks and self.admissions == before_runs,
                "journey_followup_dispatched")
        require(not meter.ambiguous, "journey_speak_audio_track")
        require(latency_ms <= _BARGE_BUDGET_MS, "journey_barge_audio_continued")
        if tail_ms < _BARGE_QUIET_MS:
            # A fresh reply closed the window before a whole quiet interval was observable.
            receipt["quiescenceInconclusive"] = True
            require(False, "journey_barge_quiescence_inconclusive")
        receipt["quiescenceObserved"] = True

    async def speak(self, step: dict) -> None:
        """Send the pinned synthetic clip as microphone audio through the real pipeline.

        Microphone input emits no typed admission, so attribution is strict instead: a quiet
        start, one user final with the exact words, one new foreground pair bound by its
        first token after that final, that completes, and worker audio inside its window.
        Anything ambiguous fails. The receipt keeps booleans and counts only.
        """
        voice, tracker = self.voice, self.tracker
        require(voice is not None, "journey_speak_unavailable")
        assert voice is not None
        frames, words = load_speak_fixture()
        receipt = self.speak_receipt = {
            "label": step["label"], "clipFrames": len(frames), "mediaRequests": 0,
            "voiceActivityOrdered": False, "finalsCount": 0, "transcriptMatches": False,
            "pairBound": False, "pairCompleted": False, "agentAudioFrames": 0,
            "agentAudioMs": 0, "inputWindowAudioFrames": 0, "inputWindowAudioMs": 0,
            "cueAudioObserved": False, "quietGateMs": 0,
        }

        trace, before_tasks, before_runs = await self._mic_quiet_start(
            voice, lambda prior, incarnation: SpeakTrace(prior, words, incarnation)
        )
        try:
            await self._mic_ready(voice, trace, receipt)
            await self._stream_bounded(voice, frames)
            with contextlib.suppress(TimeoutError):
                await self.wait(trace.terminal, _SPEAK_TURN_TIMEOUT)
            await asyncio.sleep(_SPEAK_DRAIN_SECONDS)
            await self.poll()
        finally:
            await voice.stop_idle()  # every exit path: cancel and settle the idle feeder
            voice.meter.advance("closed")
            tracker.speak = None
        agent_frames, agent_ms = voice.meter.window_totals("response")
        input_frames, input_ms = voice.meter.window_totals("input")
        receipt.update(
            agentAudioFrames=agent_frames, agentAudioMs=agent_ms,
            inputWindowAudioFrames=input_frames, inputWindowAudioMs=input_ms,
        )
        _, row = self._require_first_utterance(trace, receipt)
        receipt["pairCompleted"] = row["completed"] and not row["interrupted"]
        require(receipt["pairCompleted"], "journey_speak_turn_incomplete")
        require(set(tracker.tasks) == before_tasks and self.admissions == before_runs,
                "journey_followup_dispatched")
        require(not voice.meter.ambiguous, "journey_speak_audio_track")
        require(agent_ms >= _SPEAK_MIN_AUDIO_MS, "journey_speak_no_audio")

    def _require_task_dispatch(
        self, trace: TaskSpeakTrace, receipt: dict[str, Any],
        before_tasks: set[str], before_runs: set[str],
    ) -> tuple[str, str]:
        """The spoken delegation produced exactly one new task and one new admission, bound to
        this step's user final by event order, with nothing stale or replayed."""
        tracker = self.tracker
        first = trace.first_final
        ordered = (trace.started is not None and trace.ended is not None
                   and trace.started < trace.ended
                   and (first is None or first[0] > trace.started))
        receipt.update(
            voiceActivityOrdered=ordered, dispatchFinalsCount=trace.final_count,
            dispatchTranscriptMatches=first is not None and first[1],
        )
        require(ordered, "journey_speak_voice_activity")
        require(trace.final_count == 1, "journey_speak_finals")
        require(receipt["dispatchTranscriptMatches"], "journey_speak_transcript")
        assert first is not None
        new_tasks = set(tracker.tasks) - before_tasks
        new_runs = self.admissions - before_runs
        receipt.update(dispatchTasks=len(new_tasks), dispatchAdmissions=len(new_runs))
        require(not trace.task_overflow, "journey_task_competing")
        replayed = any(kind == "task_state" and status == "active" and task in before_tasks
                       for _, kind, task, status in trace.task_log)
        require(not replayed, "journey_task_replayed")
        require(bool(new_tasks) and bool(new_runs), "journey_task_not_dispatched")
        require(len(new_tasks) == 1 and len(new_runs) == 1, "journey_task_competing")
        target, run = next(iter(new_tasks)), next(iter(new_runs))
        rows = [row for row in trace.task_log if row[2] == target]
        receipt["taskBound"] = bool(rows) and (
            rows[0][1] == "task_state" and rows[0][3] == "active" and rows[0][0] > first[0]
        )
        require(receipt["taskBound"], "journey_task_dispatch_unbound")
        return target, run

    def _own_single_dispatch(self, label: str, before_tasks: set[str],
                             before_runs: set[str]) -> None:
        """Own the one new task and run as soon as they are unambiguous, whatever the verdict.

        Called on every exit so a later failure still lets cleanup cancel exactly that task by
        id. With zero or several candidates nothing is guessed: cleanup stays honestly false.
        """
        new_tasks = set(self.tracker.tasks) - before_tasks
        new_runs = self.admissions - before_runs
        if label not in self.bindings and len(new_tasks) == 1 and len(new_runs) == 1:
            self.bindings[label] = (next(iter(new_tasks)), next(iter(new_runs)))

    async def speak_task(self, step: dict) -> None:
        try:
            await self._speak_task(step)
        finally:
            # The trace stays live through witness collection and the final poll.
            self.tracker.speak = None

    async def _speak_task(self, step: dict) -> None:
        """Delegate by voice, then cancel that exact task by voice.

        Microphone input has no typed admission, so attribution is by event order: the
        dispatch is one new task and one new admission whose first active state follows the
        dispatch final; TARGET is that task and run, frozen. Once the session is quiet with
        TARGET still active, a natural spoken cancellation is injected. At its first speech
        frame TARGET must still be the sole live task with no competing admission, otherwise
        the result is inconclusive (journey_task_target_terminal); no replacement is chosen.
        The cancellation counts only if TARGET itself is observed cancelling or interrupted
        after that final and the native witness, read before cleanup, says cancelled.
        """
        voice, tracker = self.voice, self.tracker
        require(voice is not None, "journey_speak_unavailable")
        assert voice is not None
        dispatch, dispatch_words, cancel, (cancel_first, _), cancel_words = load_task_fixtures()
        label = step["label"]
        receipt = self.speak_receipt = {
            "label": label, "kind": "speak_task", "mediaRequests": 0,
            "voiceActivityOrdered": False, "dispatchFinalsCount": 0,
            "dispatchTranscriptMatches": False, "dispatchTasks": 0, "dispatchAdmissions": 0,
            "taskBound": False, "cueAudioObserved": False, "quietGateMs": 0,
            "preCancelQuietMs": 0, "targetActiveBeforeCancel": False, "cancelInjected": False,
            "cancelVadObserved": False, "cancelFinalsCount": 0,
            "cancelTranscriptMatches": False, "cancelObserved": False,
            "foreignCancelCount": 0, "competingAdmissionCount": 0,
            "nativeCancelledBeforeCleanup": False, "unexpectedInputCount": 0,
        }
        trace, before_tasks, before_runs = await self._mic_quiet_start(
            voice,
            lambda prior, incarnation: TaskSpeakTrace(
                prior, dispatch_words, incarnation, cancel_words
            ),
        )
        meter = voice.meter
        target = run = ""
        try:
            await self._mic_ready(voice, trace, receipt)
            await self._stream_bounded(voice, dispatch)
            with contextlib.suppress(TimeoutError):
                await self.wait(
                    lambda: trace.first_final is not None
                    and bool(set(tracker.tasks) - before_tasks)
                    and bool(self.admissions - before_runs),
                    _TASK_DISPATCH_TIMEOUT,
                )
            self._own_single_dispatch(label, before_tasks, before_runs)
            target, run = self._require_task_dispatch(trace, receipt, before_tasks, before_runs)
            trace.target = target
            trace.mark_frozen(tracker.cursor)

            def exclusive() -> None:
                # Exactly one matching dispatch final, and no further input or replayed
                # foreground, from the freeze through the cancel injection.
                receipt["dispatchFinalsCount"] = trace.final_count
                require(trace.final_count == 1, "journey_speak_finals")
                receipt["unexpectedInputCount"] = trace.late_voice + trace.unexpected_foreground
                require(trace.late_voice == 0 and trace.unexpected_foreground == 0,
                        "journey_task_unexpected_input")
            # Quiet session with TARGET still active: no open turn, no pending admission and
            # one whole quiet interval since the dispatch settled. Client clock only.
            entry = meter.now()
            try:
                await self.wait(
                    lambda: tracker.tasks.get(target) in TERMINAL
                    or (not tracker.open_turns and self.pending == 0
                        and meter.quiet_since(entry, _SPEAK_SETTLE_SECONDS)),
                    _TASK_QUIET_TIMEOUT,
                )
            except TimeoutError:
                require(False, "journey_task_not_quiet")
            receipt["preCancelQuietMs"] = int((meter.now() - entry) * 1000)
            require(tracker.tasks.get(target) not in TERMINAL, "journey_task_target_terminal")
            require(tracker.tasks.get(target) == "active", "journey_task_not_active")
            exclusive()
            receipt["targetActiveBeforeCancel"] = True

            def inject_start() -> None:
                # Re-checked at the first speech frame, after the clip's leading silence.
                status = tracker.tasks.get(target)
                require(status not in TERMINAL, "journey_task_target_terminal")
                require(status == "active", "journey_task_cancel_stale")
                exclusive()
                live = {task for task, state in tracker.tasks.items() if state not in TERMINAL}
                require(live == {target} and set(tracker.tasks) - before_tasks == {target}
                        and self.admissions - before_runs == {run} and self.pending == 0,
                        "journey_task_competing")
                trace.mark_cancel(tracker.cursor)
                receipt["cancelInjected"] = True

            await self._stream_bounded(voice, cancel, {cancel_first: inject_start})
            with contextlib.suppress(TimeoutError):
                await self.wait(
                    lambda: trace.target_cancel_seq is not None
                    or tracker.tasks.get(target) in TERMINAL,
                    _TASK_CANCEL_TIMEOUT,
                )
            await asyncio.sleep(_SPEAK_DRAIN_SECONDS)
            await self.poll()
        finally:
            self._own_single_dispatch(label, before_tasks, before_runs)  # before any cleanup
            await voice.stop_idle()  # every exit path: cancel and settle the idle feeder
            meter.advance("closed")

        def validate_cancellation() -> None:
            # One bounded validate-and-snapshot of the cancellation: its input (a voice start,
            # exactly one final with the exact words) and its scope (nothing foreign cancelled,
            # no competing task or admission). The counts are refreshed into the receipt first,
            # so a refusal keeps them. Run before, and again after, the witness read and the
            # final poll, with the trace still live.
            extra_admissions = len(self.admissions - before_runs - {run})
            receipt.update(
                cancelVadObserved=trace.cancel_started >= 1,
                cancelFinalsCount=trace.cancel_finals,
                cancelTranscriptMatches=trace.cancel_matches,
                cancelObserved=trace.target_cancel_seq is not None,
                foreignCancelCount=trace.foreign_cancel,
                competingAdmissionCount=trace.competing_active + extra_admissions,
            )
            require(trace.cancel_started >= 1, "journey_task_cancel_vad")
            require(trace.cancel_finals == 1, "journey_task_cancel_finals")
            require(trace.cancel_matches, "journey_task_cancel_transcript")
            require(trace.foreign_cancel == 0, "journey_task_foreign_cancel")
            require(trace.competing_active == 0 and extra_admissions == 0,
                    "journey_task_competing")
            require(not trace.cancel_stale, "journey_task_cancel_stale")

        validate_cancellation()
        require(trace.target_cancel_seq is not None, "journey_task_cancel_unobserved")
        # The native witness is read here, before cleanup: cleanup never supplies evidence.
        await self.collect(label)
        require(self.results[label]["terminal"] == "cancelled", "journey_cancel_unproven")
        receipt["nativeCancelledBeforeCleanup"] = True
        await self.poll()  # a late dispatch or foreign cancel would show here
        validate_cancellation()  # re-checked after the witness read and the final poll
        require(set(tracker.tasks) == before_tasks | {target}
                and self.admissions - before_runs == {run}, "journey_followup_dispatched")

    async def _confirm_dispatch(
        self, trace: ConfirmCancelTrace, label: str, text: str,
        before_tasks: set[str], before_runs: set[str], receipt: dict[str, Any],
    ) -> tuple[str, str]:
        """Start one task by typed explicit command and bind exactly one new task and run to
        it by event order: the task's first projected state is active, after the send."""
        tracker = self.tracker
        # Not a measured turn: a command starts none, and the microphone finals must never be
        # checked as this typed text.
        tracker.current, tracker.text = None, ""
        sent = tracker.cursor
        self.sequence += 1
        try:
            await self.request("input", {"sequence": self.sequence, "text": text})
            with contextlib.suppress(TimeoutError):
                await self.wait(lambda: bool(set(tracker.tasks) - before_tasks)
                                and bool(self.admissions - before_runs), _TASK_DISPATCH_TIMEOUT)
        finally:
            # Owned as soon as unambiguous, whatever the verdict, so cleanup can cancel it.
            self._own_single_dispatch(label, before_tasks, before_runs)
        new_tasks, new_runs = set(tracker.tasks) - before_tasks, self.admissions - before_runs
        receipt["dispatchTasks"] += len(new_tasks)
        receipt["dispatchAdmissions"] += len(new_runs)
        require(not trace.overflow, "journey_confirm_trace_bound")
        require(not trace.replayed(before_tasks, sent), "journey_confirm_competing")
        require(bool(new_tasks) and bool(new_runs), "journey_confirm_not_dispatched")
        require(len(new_tasks) == 1 and len(new_runs) == 1, "journey_confirm_competing")
        task, run = next(iter(new_tasks)), next(iter(new_runs))
        first = trace.first_task_row(task)
        require(first is not None and first[0] > sent and first[1:] == ("task_state", "active"),
                "journey_confirm_dispatch_unbound")
        return task, run

    def _require_confirm_live(self, trace: ConfirmCancelTrace, tasks: tuple[str, ...],
                              cancel_category: str,
                              failures: Callable[[], None] | None = None) -> None:
        """Each task still active: any cancel seen fails as ``cancel_category``; a natural end
        is inconclusive (journey_confirm_target_terminal), never replaced. Failure precedes
        inconclusive: every task's cancels, then every other failure-only check
        (``failures``: competing work, unexpected input, bounds), run before any task's
        natural end is named."""
        statuses = [self.tracker.tasks.get(task) for task in tasks]
        for task, status in zip(tasks, statuses, strict=True):
            require(status not in _CONFIRM_CANCELS and not trace.cancels(task), cancel_category)
        if failures is not None:
            failures()
        for status in statuses:
            require(status not in TERMINAL, "journey_confirm_target_terminal")
            require(status == "active", "journey_confirm_not_active")

    async def speak_confirm_cancel(self, step: dict[str, Any]) -> None:
        try:
            await self._speak_confirm_cancel(step)
        finally:
            self.tracker.speak = None

    async def _speak_confirm_cancel(self, step: dict[str, Any]) -> None:
        """Two typed tasks, then a confirmed natural spoken cancel of exactly one of them.

        A and B are started by typed `task:` commands so their stored objectives, and so the
        host's fixed readbacks, are exact. With both active and the session quiet: a request
        naming B must get B's exact readback, and No must cancel nothing; a request naming A
        must get A's exact readback, and only after Yes may A be observed cancelling, with B
        never cancelled and still live at the native witness read, which precedes all cleanup.
        A readback for the other task, a reply without a readback, or a task that ends on its
        own is inconclusive; no answer is sent after it and no replacement is chosen.
        """
        voice, tracker = self.voice, self.tracker
        require(voice is not None, "journey_speak_unavailable")
        assert voice is not None
        clips = load_confirm_fixtures()
        label = step["label"]
        labels = {"a": label + ".a", "b": label + ".b"}  # never a scenario label (no dot)
        commands = {"a": _START_PREFIX + " " + _CONFIRM_OBJECTIVE_A,
                    "b": _START_PREFIX + " " + _CONFIRM_OBJECTIVE_B}
        receipt = self.speak_receipt = {
            "label": label, "kind": "speak_confirm_cancel", "mediaRequests": 0,
            "cueAudioObserved": False, "quietGateMs": 0, "dispatchTasks": 0,
            "dispatchAdmissions": 0, "tasksBound": False, "bothActiveBeforeRequest": False,
            "quietBeforeRequestBMs": 0, "quietBeforeRequestAMs": 0, "unexpectedInputCount": 0,
            **{name + suffix: value for name in _CONFIRM_CLIPS
               for suffix, value in (("Injected", False), ("VadObserved", False),
                                     ("FinalsCount", 0), ("TranscriptMatches", False))},
            "readbackBOutcome": None, "readbackAOutcome": None, "bothActiveAfterNo": False,
            "targetCancelObserved": False, "bystanderCancelObserved": False,
            "bystanderActiveAtWitness": False, "nativeCancelledBeforeCleanup": False,
            "nativeBystanderLiveBeforeCleanup": False, "successPathCleanupCancels": 0,
        }
        trace, before_tasks, before_runs = await self._mic_quiet_start(
            voice, lambda prior, incarnation: ConfirmCancelTrace(
                prior, incarnation, frozenset(commands.values())
            ),
        )
        meter = voice.meter
        task_a = task_b = run_a = run_b = ""

        def disturbed() -> bool:
            return any(tracker.tasks.get(task) != "active" for task in (task_a, task_b))

        def scope() -> None:
            # Exactly A and B (and their runs) are new, nothing else is live, and nothing
            # unexpected arrived before the first clip.
            receipt["unexpectedInputCount"] = trace.unexpected
            require(not trace.overflow, "journey_confirm_trace_bound")
            require(trace.unexpected == 0, "journey_confirm_unexpected_input")
            live = {task for task, state in tracker.tasks.items() if state not in TERMINAL}
            require(live <= {task_a, task_b} and set(tracker.tasks) - before_tasks
                    == {task_a, task_b} and self.admissions - before_runs == {run_a, run_b},
                    "journey_confirm_competing")

        def require_clip(index: int) -> None:
            name = _CONFIRM_CLIPS[index]
            vad, finals, matches, _ = trace.clip(index)
            receipt.update({name + "VadObserved": vad, name + "FinalsCount": finals,
                            name + "TranscriptMatches": matches})
            require(vad, "journey_confirm_vad")
            require(finals == 1, "journey_confirm_finals")
            require(matches, "journey_confirm_transcript")

        async def quiet(key: str, cancel_category: str) -> None:
            # No open turn, no pending admission and one whole quiet interval, with both
            # tasks still active. Client clock only.
            entry = meter.now()
            try:
                await self.wait(lambda: disturbed() or (
                    not tracker.open_turns and self.pending == 0
                    and meter.quiet_since(entry, _SPEAK_SETTLE_SECONDS)), _TASK_QUIET_TIMEOUT)
            except TimeoutError:
                require(False, "journey_confirm_not_quiet")
            receipt[key] = int((meter.now() - entry) * 1000)
            self._require_confirm_live(trace, (task_a, task_b), cancel_category, scope)

        async def inject(index: int, cancel_category: str) -> None:
            frames, words = clips[index]

            def start() -> None:
                # Re-checked at the first speech frame, after the clip's leading silence: both
                # still active, nothing competing. Abort before any speech if not.
                def failures() -> None:
                    scope()
                    require(self.pending == 0, "journey_confirm_competing")

                self._require_confirm_live(trace, (task_a, task_b), cancel_category, failures)
                trace.mark(tracker.cursor, words)
                receipt[_CONFIRM_CLIPS[index] + "Injected"] = True

            await self._stream_bounded(voice, frames, {_SPEAK_LEAD_FRAMES: start})

        async def request(index: int, want: str, other: str) -> None:
            cancel_category = ("journey_confirm_premature_cancel" if index == 0
                               else "journey_confirm_no_cancelled")
            await inject(index, cancel_category)
            with contextlib.suppress(TimeoutError):
                await self.wait(lambda: disturbed() or trace.settled(index),
                                _CONFIRM_REPLY_TIMEOUT)
            await asyncio.sleep(_SPEAK_DRAIN_SECONDS)
            await self.poll()
            # A readback is only a question: nothing may be cancelling yet. Side effects are
            # proven absent before any reply is classified, so a mismatch or a clarification
            # can never mask one. From the No onward a cancel is a No side effect.
            self._require_confirm_live(trace, (task_a, task_b), cancel_category, scope)
            require_clip(index)
            final = trace.clip(index)[3]
            assert final is not None
            outcome = trace.reply(index, final, want, other)
            receipt["readback" + want.upper() + "Outcome"] = outcome
            require(outcome != "ambiguous", "journey_confirm_ambiguous")
            require(outcome in {"target", "other", "clarified"},
                    "journey_confirm_readback_incomplete")
            require(outcome != "other", "journey_confirm_nomination_mismatch")
            require(outcome == "target", "journey_confirm_clarified")

        try:
            await self._mic_ready(voice, trace, receipt)
            trace.frozen_seq = tracker.cursor
            task_a, run_a = await self._confirm_dispatch(
                trace, labels["a"], commands["a"], before_tasks, before_runs, receipt)
            self._require_confirm_live(trace, (task_a,), "journey_confirm_premature_cancel")
            task_b, run_b = await self._confirm_dispatch(
                trace, labels["b"], commands["b"], before_tasks | {task_a},
                before_runs | {run_a}, receipt)
            self._require_confirm_live(trace, (task_a, task_b), "journey_confirm_premature_cancel")
            receipt["tasksBound"] = True
            await quiet("quietBeforeRequestBMs", "journey_confirm_premature_cancel")
            receipt["bothActiveBeforeRequest"] = True
            await request(0, "b", "a")
            # No: nothing may be cancelled through a whole drain window after its final.
            await inject(1, "journey_confirm_premature_cancel")
            with contextlib.suppress(TimeoutError):
                await self.wait(lambda: disturbed() or (
                    trace.clip(1)[3] is not None and not tracker.open_turns),
                    _CONFIRM_REPLY_TIMEOUT)
            with contextlib.suppress(TimeoutError):
                await self.wait(disturbed, _CONFIRM_NO_DRAIN_SECONDS)
            await self.poll()
            self._require_confirm_live(trace, (task_a, task_b), "journey_confirm_no_cancelled",
                                       scope)
            require_clip(1)
            receipt["bothActiveAfterNo"] = True
            await quiet("quietBeforeRequestAMs", "journey_confirm_no_cancelled")
            await request(2, "a", "b")
            await inject(3, "journey_confirm_no_cancelled")
            with contextlib.suppress(TimeoutError):
                await self.wait(lambda: trace.arming(task_a, 3) is not None
                                or tracker.tasks.get(task_b) != "active"
                                or tracker.tasks.get(task_a) in {"completed", "failed"},
                                _TASK_CANCEL_TIMEOUT)
            await asyncio.sleep(_SPEAK_DRAIN_SECONDS)
            await self.poll()
        finally:
            await voice.stop_idle()  # every exit path: cancel and settle the idle feeder
            meter.advance("closed")

        def validate() -> None:
            # B never cancelled and not ended; nothing else new; Yes heard exactly once; and A
            # cancelling or interrupted only after that final. Run before and after each
            # native read, with the trace still live.
            status_b = tracker.tasks.get(task_b)
            receipt["bystanderCancelObserved"] = (bool(trace.cancels(task_b))
                                                  or status_b in _CONFIRM_CANCELS)
            require(not receipt["bystanderCancelObserved"], "journey_confirm_wrong_target")
            # Every failure-only check before the inconclusive ended-bystander outcome.
            scope()
            require_clip(3)
            final = trace.clip(3)[3]
            assert final is not None
            require(all(seq > final for seq in trace.cancels(task_a)),
                    "journey_confirm_premature_cancel")
            require(status_b == "active", "journey_confirm_bystander_ended")
            armed = trace.arming(task_a, 3)
            receipt["targetCancelObserved"] = armed is not None
            if armed is None:
                require(tracker.tasks.get(task_a) not in {"completed", "failed"},
                        "journey_confirm_target_terminal")
            require(armed is not None, "journey_confirm_not_cancelled")

        validate()
        # The native witness is read here, before any cleanup: cleanup never supplies evidence.
        await self.collect(labels["a"])
        require(self.results[labels["a"]]["terminal"] == "cancelled", "journey_cancel_unproven")
        receipt["nativeCancelledBeforeCleanup"] = True
        await self.poll()
        validate()
        receipt["bystanderActiveAtWitness"] = True
        live = await asyncio.to_thread(NativeRunWitness(
            self.bearer, run_b, profile=self.options.native_profile,
            api_origin=self.options.api_origin,
        ).live_status)
        require(live not in {"stopping", "cancelled", "interrupted"},
                "journey_confirm_wrong_target")
        require(live in _CONFIRM_LIVE_RUN, "journey_confirm_bystander_ended")
        receipt["nativeBystanderLiveBeforeCleanup"] = True
        await self.poll()
        validate()  # re-checked after the live read and the final poll
        # Cleanup, after all acceptance evidence: B is cancelled by exact id and counted
        # separately. Its events are not trace evidence.
        tracker.speak = None
        tracker.current, tracker.text = None, ""
        self.sequence += 1
        receipt["successPathCleanupCancels"] += 1
        await self.request("input", {"sequence": self.sequence,
                                     "text": _CANCEL_PREFIX + " " + task_b})
        await self.wait(lambda: tracker.tasks.get(task_b) in TERMINAL, 45)
        await self.collect(labels["b"])

    async def step(self, step: dict) -> None:
        kind, label = step["kind"], step["label"]
        if kind in {"speak", "speak_barge", "speak_task", "speak_confirm_cancel"}:
            if kind == "speak_barge":
                await self.speak_barge(step)
            elif kind == "speak_task":
                await self.speak_task(step)
            elif kind == "speak_confirm_cancel":
                await self.speak_confirm_cancel(step)
            else:
                await self.speak(step)
            return
        if kind == "recall":
            await self.collect(step["task"])
        if kind == "dispatch":
            require(
                all(self.tracker.tasks.get(task) in TERMINAL for task, _ in self.bindings.values()),
                "journey_previous_work_active",
            )
        before_tasks, before_runs = set(self.tracker.tasks), set(self.admissions)
        text = step.get("text")
        if kind in {"cancel", "natural_cancel"}:
            task, _ = self.bindings[step["task"]]
            require(self.tracker.tasks.get(task) == "active", "journey_cancel_not_active")
        if kind == "cancel":
            text = "cancel task: " + task
        self.sequence += 1
        self.tracker.begin(label, self.sequence, text)
        self.tracker.records[-1]["task"] = step.get("task")
        await self.request("input", {"sequence": self.sequence, "text": text})
        if kind == "dispatch":
            await self.wait(lambda: bool(set(self.tracker.tasks) - before_tasks)
                            or self.tracker.current["outcome"] == "interrupted")
            new_tasks, new_runs = (
                set(self.tracker.tasks) - before_tasks,
                self.admissions - before_runs,
            )
            if not new_tasks:
                require(self.tracker.current["outcome"] != "interrupted",
                        "journey_turn_interrupted")
            require(len(new_tasks) == len(new_runs) == 1, "journey_admission_binding")
            self.bindings[label] = (next(iter(new_tasks)), next(iter(new_runs)))
            self.tracker.records[-1]["task"] = label
            require(self.tracker.current["outcome"] != "interrupted", "journey_turn_interrupted")
            if step.get("wait", "completed") == "completed":
                await self.collect(label)
                require(self.results[label]["terminal"] == "completed", "journey_task_failed")
            if "tool" in step:
                await self.collect(label)
                require(
                    any(
                        row["tool"] == step["tool"] and row["successfulResults"] > 0
                        and (step["tool"] not in {"web_search", "web_extract"}
                             or row.get("usableResults", 0) > 0)
                        for row in self.results[label]["toolCalls"]
                    ),
                    "journey_tool_unproven",
                )
        elif kind == "cancel":
            await self.collect(step["task"])
            require(
                self.results[step["task"]]["terminal"] == "cancelled", "journey_cancel_unproven"
            )
        elif kind == "natural_cancel":
            # Turn completion (token or admitted-input lineage) is not task success.
            await self.wait(lambda: self.tracker.current["outcome"] != "pending")
            outcome = self.tracker.current["outcome"]
            require(outcome != "interrupted", "journey_turn_interrupted")
            require(outcome == "completed", "journey_turn_incomplete")
            # The task's cancelling projection precedes the turn's completion, and
            # only this step's post-admission projections count: never pre-step
            # state, and never the cleanup `cancel task:` after the step.
            def require_cancel_scope() -> None:
                cancels, admission = self.tracker.step_cancels, self.tracker.admission_event
                require(task in cancels, "journey_cancel_unobserved")
                require(admission is not None and cancels[task] > admission,
                        "journey_cancel_stale")
                require(set(cancels) == {task}, "journey_cancel_foreign")

            require_cancel_scope()
            await self.collect(step["task"])
            # collect() keeps consuming events while the task settles; a known
            # foreign task cancelled in that window must still fail the step.
            require_cancel_scope()
            require(
                self.results[step["task"]]["terminal"] == "cancelled", "journey_cancel_unproven"
            )
            require(
                set(self.tracker.tasks) == before_tasks and self.admissions == before_runs,
                "journey_followup_dispatched",
            )
        else:
            await self.wait(lambda: self.tracker.current["outcome"] != "pending")
            require(self.tracker.current["outcome"] == "completed", "journey_turn_incomplete")
            # A spoken step needs its own first token; lineage alone is silent.
            require(self.tracker.current["terminalBinding"] == "token", "journey_turn_tokenless")
            require(
                set(self.tracker.tasks) == before_tasks and self.admissions == before_runs,
                "journey_followup_dispatched",
            )

    def pre_cleanup_tasks(self) -> dict[str, dict[str, Any]]:
        """Synchronous task state before cleanup, for diagnosis only; never a verdict input.

        Nothing is awaited and no native witness is read: mandatory cleanup follows with
        no suspension point in between. The native category is the one collect() already
        retained, or null when collect() did not run for that task.
        """
        tasks: dict[str, dict[str, Any]] = {}
        for label, (task, _) in self.bindings.items():
            status = self.tracker.tasks.get(task)
            native = self.results[label]["terminal"] if label in self.results else None
            tasks[label] = {
                "trackerStatus": status if status in {"active", "cancelling", *TERMINAL}
                else None,
                "nativeTerminal": _native_category(native),
            }
        return tasks

    async def settle_owned(self) -> bool:
        # Cleanup commands are not another measured foreground turn.
        self.tracker.current = None
        self.tracker.text = ""
        for label, (task, _) in self.bindings.items():
            if self.tracker.tasks.get(task) not in TERMINAL:
                self.sequence += 1
                await self.request(
                    "input", {"sequence": self.sequence, "text": "cancel task: " + task}
                )
                await self.wait(lambda task=task: self.tracker.tasks.get(task) in TERMINAL, 45)
                await self.collect(label)
        return read_journal(self.journal) == (set(), 0)


async def execute(options, steps, receipt) -> None:
    from hermes_realtime.host_launcher import _load_api_bearer

    bearer = _load_api_bearer(options.native_profile / ".env")
    from livekit import rtc

    require(
        read_journal(options.preview_dir / "state/hermes-runs-v1.json") == (set(), 0),
        "journey_initial_work",
    )
    credential, journey, room, bootstrap_code = None, None, None, None
    try:
        bootstrap_code, credential = await asyncio.to_thread(
            request_json, options.browser_origin, "stable-bootstrap"
        )
        suffix = hashlib.sha256(
            str(options.preview_dir / "active.private.json").encode()
        ).hexdigest()[:16]
        require(
            bootstrap_code == 200
            and credential.get("workerIdentity") == "worker_preview_" + suffix
            and credential.get("roomName") == "hermes-preview-" + suffix
            and credential.get("url") == options.livekit_url
            and type(credential.get("token")) is str
            and type(credential.get("participantIdentity")) is str
            and type(credential.get("expiresInSeconds")) is int
            and 1 <= credential["expiresInSeconds"] <= 3600,
            "journey_bootstrap_binding",
        )
        journey = Journey(options, credential, bearer)
        room = rtc.Room()
        await asyncio.wait_for(
            room.connect(
                credential["url"],
                credential["token"],
                options=rtc.RoomOptions(auto_subscribe=False),
            ),
            15,
        )
        if any(step.get("kind") in {"speak", "speak_barge", "speak_task",
                                    "speak_confirm_cancel"}
               for step in steps):
            # Subscribe only to the worker's audio, by the identity validated above.
            journey.voice = SyntheticVoice(rtc, room, credential["workerIdentity"])
            journey.voice.meter.start()
        await journey.poll()
        require(not journey.admissions and journey.pending == 0
                and all(status in TERMINAL for status in journey.tracker.tasks.values()),
                "journey_existing_task")
        for step in steps:
            await journey.step(step)
        for label in journey.bindings:
            if label not in journey.results:
                await journey.collect(label)
        require(read_journal(journey.journal) == (set(), 0), "journey_unsettled")
        receipt["outcome"] = "completed"
    finally:
        receipt["cleanup"] = {
            "ownedTasksSettled": journey is None,
            "sessionStopped": False,
            "roomDisconnected": room is None,
        }
        diagnostics = None
        if journey is not None:
            # Diagnosis only, captured synchronously before cleanup acts: no await and no
            # native read may precede mandatory cleanup, and it never changes a verdict.
            with contextlib.suppress(Exception):
                diagnostics = _journey_diagnostics(
                    journey.tracker, journey.bindings, journey.pre_cleanup_tasks(),
                )
            with contextlib.suppress(Exception):
                receipt["cleanup"]["ownedTasksSettled"] = await journey.settle_owned()
            if journey.voice is not None:
                with contextlib.suppress(Exception):
                    await journey.voice.aclose()
            credential = journey.credential
        # Even a mismatched bootstrap may have provisioned a lease. Its returned
        # bearer is authority only to stop that lease at the same browser origin.
        token = credential.get("token") if type(credential) is dict else None
        if bootstrap_code == 200 and type(token) is str and 0 < len(token) <= 16_384:
            with contextlib.suppress(Exception):
                code, _ = await asyncio.to_thread(
                    request_json, options.browser_origin, "stop", token=token
                )
                receipt["cleanup"]["sessionStopped"] = code in {200, 202}
        if room is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(room.disconnect(), 10)
                receipt["cleanup"]["roomDisconnected"] = True
        if journey is not None:
            receipt.update(
                turns=journey.tracker.receipt(),
                tasks=journey.results,
                acceptedDispatches=len(journey.bindings),
            )
            if journey.failures:
                receipt["taskFailures"] = journey.failures
            if journey.speak_receipt is not None:
                receipt["speak"] = journey.speak_receipt
            if diagnostics is not None:
                receipt["diagnostics"] = diagnostics
        if receipt["outcome"] == "completed":
            require(all(receipt["cleanup"].values()), "journey_cleanup_unsettled")


def _os_error_evidence(error: OSError) -> dict[str, object]:
    """Nearest supported journey context, never an attribution of the OS cause."""
    facts: dict[str, object] = {
        "sourceFunction": "other", "sourceLine": None, "pathRole": "other",
        "exceptionKind": {
            OSError: "OSError", PermissionError: "PermissionError",
            FileNotFoundError: "FileNotFoundError", TimeoutError: "TimeoutError",
            NotADirectoryError: "NotADirectoryError", IsADirectoryError: "IsADirectoryError",
        }.get(type(error), "OSError"),
    }
    frame = error.__traceback__
    for _ in range(64):
        if frame is None:
            break
        context = _OS_ERROR_CONTEXTS.get(frame.tb_frame.f_code)
        if context is not None:
            function, role = context
            facts["sourceFunction"] = function
            line = frame.tb_lineno
            facts["sourceLine"] = line if type(line) is int and 1 <= line <= 1_000_000 else None
            if role is not None:
                facts["pathRole"] = role
        frame = frame.tb_next
    for name in ("errno", "winerror"):
        value = getattr(error, name, None)
        facts[name] = (value if type(value) is int and -(2**31) <= value <= 2**32 - 1 else None)
    return facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preview-dir", "native-profile", "scenario", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("browser-origin", "api-origin", "livekit-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args(argv)
    receipt: dict[str, Any] = {"version": 1, "outcome": "refused", "executed": False}
    reserved = None
    try:
        for name in ("preview_dir", "native_profile"):
            setattr(options, name, getattr(options, name).resolve(strict=True))
        validate_origin(options.browser_origin)
        validate_origin(options.api_origin)
        require(options.livekit_url.startswith("ws://"), "journey_livekit_scheme")
        validate_origin("http://" + options.livekit_url[5:])
        steps = load_scenario(options.scenario)
        reserved = options.receipt.open("x", encoding="utf-8")
        receipt["stepCount"] = len(steps)
        if options.execute:
            receipt["executed"] = True
            asyncio.run(execute(options, steps, receipt))
        else:
            receipt["outcome"] = "preflight_only"
    except Exception as error:
        category = (
            str(error)
            if isinstance(error, ValueError)
            and re.fullmatch(r"(?:journey|witness)_[a-z_]+", str(error))
            else type(error).__name__
        )
        receipt.update(outcome="refused", failure=category)
        if isinstance(error, OSError):
            with contextlib.suppress(Exception):
                receipt["osError"] = _os_error_evidence(error)
    finally:
        raw = json.dumps(receipt, sort_keys=True)
        if len(raw) > 65_536 and "diagnostics" in receipt:
            # Diagnosis is the most optional evidence: it goes first, never the verdict.
            receipt.pop("diagnostics")
            raw = json.dumps(receipt, sort_keys=True)
        if len(raw) > 65_536 and type(receipt.get("turns")) is list:
            # Passive timings cannot reject a previously fitting receipt or
            # displace its existing failure evidence. Keep the original bound.
            for turn in receipt["turns"]:
                if type(turn) is dict:
                    for key in _PUBLICATION_TIMING_FIELDS:
                        turn.pop(key, None)
            raw = json.dumps(receipt, sort_keys=True)
        if "osError" in receipt and len(raw) > 65_536:
            # Optional evidence cannot make a previously fitting receipt fail.
            receipt.pop("osError")
            raw = json.dumps(receipt, sort_keys=True)
        if reserved is not None:
            with reserved:
                require(len(raw) <= 65_536, "journey_receipt_bound")
                reserved.write(raw + "\n")
        print("[native-typed-journey] " + raw)
    return int(receipt["outcome"] == "refused")


# Capture code identities once: wrappers or arbitrary traceback names cannot
# supply private names/paths. No frame locals or exception messages are read.
_OS_ERROR_CONTEXTS = {
    read_json.__code__: ("read_json", None),
    load_scenario.__code__: ("load_scenario", "scenario"),
    read_journal.__code__: ("read_journal", "journal"),
    request_json.__code__: ("request_json", None),
    execute.__code__: ("execute", None),
    Journey.request.__code__: ("Journey.request", None),
    Journey.poll.__code__: ("Journey.poll", None),
    Journey.wait.__code__: ("Journey.wait", None),
    Journey.collect.__code__: ("Journey.collect", None),
    Journey.step.__code__: ("Journey.step", None),
    Journey.settle_owned.__code__: ("Journey.settle_owned", None),
    main.__code__: ("main", None),
}


if __name__ == "__main__":
    raise SystemExit(main())
