"""Bounded public browser state and latency-event projection."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from hermes_realtime.evidence.models import ProjectionReservation

_EVENT_KINDS = frozenset(
    {
        "approval_state",
        "assistant_text_generated",
        "assistant_turn_completed",
        "assistant_turn_interrupted",
        "barge_in_non_speech_suppressed",
        "barge_in_verifier_unavailable",
        "completion_received",
        "capture_status",
        "echo_barge_in_confirmed",
        "echo_suppressed",
        "transcript_echo_suppressed",
        "first_foreground_token",
        "first_playable_audio",
        "interrupt_requested",
        "knowledge_timing",
        "notification_queued",
        "playback_silenced",
        "session_model",
        "session_usage",
        "session_ready",
        "search_egress_status",
        "session_stopped",
        "speech_timing",
        "speech_ended",
        "task_state",
        "task_result",
        "transcript_final",
        "transcript_partial",
        "typed_input_admitted",
        "voice_activity_ended",
        "voice_activity_started",
        "voice_input_ready",
    }
)
_KEY = re.compile(r"[A-Za-z][A-Za-z0-9]{0,63}\Z")
_TASK_ID = re.compile(r"task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,122}\Z")
_TURN_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")
_METRIC_LABEL = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_PRIVATE_HANDLE = re.compile(
    r"(?:deleg_[A-Za-z0-9_-]*|run_[A-Za-z0-9_-]{8,}|subagent_[A-Za-z0-9_-]+)"
)

PublicValue = str | int | bool | None


@dataclass(frozen=True, slots=True)
class BrowserPublicEvent:
    """One immutable, sequenced browser-safe projection event."""

    sequence: int
    kind: str
    monotonic_ms: float
    data: Mapping[str, PublicValue]


class BrowserEventProjection:
    """Fail-closed bounded public event log for one browser lease."""

    def __init__(
        self,
        *,
        capacity: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(capacity) is not int:
            raise TypeError("capacity must be an exact integer")
        if not 1 <= capacity <= 4096:
            raise ValueError("capacity must be between 1 and 4096")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._capacity = capacity
        self._clock = clock
        self._events: list[BrowserPublicEvent] = []
        self._sequence = 0
        self._overflowed = False
        self._capture_status_reservations: set[ProjectionReservation] = set()

    def reset(self) -> None:
        """Begin a fresh browser lease with sequence authority restarting at one."""

        self._events.clear()
        self._sequence = 0
        self._overflowed = False
        self._capture_status_reservations.clear()

    def reserve_capture_status(self) -> ProjectionReservation:
        """Retain one nondroppable capture-status publication slot."""

        return self.reserve_capture_status_slots(1)[0]

    def reserve_capture_status_slots(
        self,
        slots: int,
    ) -> tuple[ProjectionReservation, ...]:
        """Atomically retain a bounded lifecycle set of nondroppable status slots."""

        if type(slots) is not int:
            raise TypeError("capture status slot count must be an exact integer")
        if not 1 <= slots <= 4:
            raise ValueError("capture status slot count must be between one and four")
        if (
            self._overflowed
            or len(self._events) + len(self._capture_status_reservations) + slots
            > self._capacity
        ):
            self._overflowed = True
            raise RuntimeError("public event projection capacity was exceeded")
        reservations = tuple(object.__new__(ProjectionReservation) for _ in range(slots))
        self._capture_status_reservations.update(reservations)
        return reservations

    def publish_capture_status(
        self,
        reservation: ProjectionReservation,
        data: dict[str, PublicValue],
    ) -> BrowserPublicEvent:
        """Consume one exact reservation to publish authoritative capture status."""

        if (
            type(reservation) is not ProjectionReservation
            or reservation not in self._capture_status_reservations
        ):
            raise RuntimeError("capture status reservation is stale or foreign")
        self._validate_capture_status(data)
        self._capture_status_reservations.remove(reservation)
        return self._publish_validated("capture_status", data, reserved=True)

    def validate_capture_status_reservation(
        self,
        reservation: ProjectionReservation,
    ) -> None:
        """Validate exact lease ownership without consuming the retained slot."""

        if (
            type(reservation) is not ProjectionReservation
            or reservation not in self._capture_status_reservations
        ):
            raise RuntimeError("capture status reservation is stale or foreign")

    def release_capture_status_reservations(
        self,
        reservations: tuple[ProjectionReservation, ...],
    ) -> None:
        """Atomically abandon exact retained slots before authoritative side effects."""

        if type(reservations) is not tuple or not reservations:
            raise TypeError("capture status reservations must be a nonempty exact tuple")
        if len(set(reservations)) != len(reservations):
            raise RuntimeError("capture status reservations contain duplicates")
        for reservation in reservations:
            self.validate_capture_status_reservation(reservation)
        self._capture_status_reservations.difference_update(reservations)

    def ensure_capacity(self, slots: int = 1) -> None:
        """Fail before an authoritative side effect when event slots are unavailable."""

        if type(slots) is not int:
            raise TypeError("projection slot count must be an exact integer")
        if not 1 <= slots <= self._capacity:
            raise ValueError("projection slot count is outside the capacity bound")
        if (
            self._overflowed
            or len(self._events) + len(self._capture_status_reservations) + slots
            > self._capacity
        ):
            self._overflowed = True
            raise RuntimeError("public event projection capacity was exceeded")

    def publish(self, kind: str, data: dict[str, PublicValue]) -> BrowserPublicEvent:
        """Publish one validated event or permanently fail closed on overflow."""

        if type(kind) is not str:
            raise TypeError("kind must be an exact built-in string")
        if kind not in _EVENT_KINDS:
            raise ValueError("event kind is not public")
        if kind == "capture_status":
            raise PermissionError("capture status requires a retained reservation")
        if type(data) is not dict:
            raise TypeError("event data must be an exact built-in dictionary")
        return self._publish_validated(kind, data, reserved=False)

    def _publish_validated(
        self,
        kind: str,
        data: dict[str, PublicValue],
        *,
        reserved: bool,
    ) -> BrowserPublicEvent:
        if kind == "session_ready":
            expected = {
                "conversationProfile",
                "mode",
                "sttModel",
                "sttProvider",
                "ttsModel",
                "ttsProvider",
            }
            if set(data) != expected:
                raise ValueError("session speech runtime data has an invalid shape")
            if data["conversationProfile"] not in {"legacy", "natural_v1"}:
                raise ValueError("session speech runtime profile is invalid")
            if data["mode"] != "microphone_or_typed":
                raise ValueError("session speech runtime mode is invalid")
            for key, maximum in (
                ("sttProvider", 64),
                ("sttModel", 256),
                ("ttsProvider", 64),
                ("ttsModel", 256),
            ):
                value = data[key]
                if type(value) is not str or not value.strip() or len(value) > maximum:
                    raise ValueError("session speech runtime identity is invalid")
        if kind == "voice_input_ready":
            if set(data) != {"generation", "mediaIncarnation"}:
                raise ValueError(
                    "voice readiness data must contain generation and mediaIncarnation"
                )
            generation = data["generation"]
            media_incarnation = data["mediaIncarnation"]
            if type(generation) is not int or type(media_incarnation) is not int:
                raise TypeError("voice readiness authority fields must be exact integers")
            if not 1 <= generation <= (1 << 53) - 1:
                raise ValueError("voice readiness generation is outside the browser-safe range")
            if not 1 <= media_incarnation <= (1 << 53) - 1:
                raise ValueError("voice readiness incarnation is outside the browser-safe range")
        if kind == "task_result":
            if set(data) != {"status", "taskId", "text"}:
                raise ValueError("task result data has an invalid shape")
            status = data["status"]
            task_id = data["taskId"]
            text = data["text"]
            if type(status) is not str or status not in {
                "completed",
                "failed",
                "interrupted",
            }:
                raise ValueError("task result status is invalid")
            if type(task_id) is not str or _TASK_ID.fullmatch(task_id) is None:
                raise ValueError("task result identity is invalid")
            if type(text) is not str or not text.strip() or len(text) > 1024:
                raise ValueError("task result text is invalid")
        if kind == "knowledge_timing":
            expected = {
                "backend",
                "lastMs",
                "lookupBlockingMs",
                "lookupElapsedMs",
                "lookupOverlapMs",
                "outcome",
                "p50Ms",
                "p95Ms",
                "recoveryUsed",
                "route",
                "sampleCount",
                "turnId",
            }
            health_fields = {
                "lookupClosed",
                "lookupDetachedCalls",
                "lookupDetachedCallsTotal",
                "lookupSaturationEvents",
            }
            if set(data) != expected and set(data) != expected | health_fields:
                raise ValueError("knowledge timing shape is invalid")
            turn_id = data["turnId"]
            if type(turn_id) is not str or _TURN_ID.fullmatch(turn_id) is None:
                raise ValueError("knowledge timing turn identity is invalid")
            for label in ("route", "backend"):
                value = data[label]
                if type(value) is not str or _METRIC_LABEL.fullmatch(value) is None:
                    raise ValueError("knowledge timing label is invalid")
            if data["outcome"] not in {"usable", "weak", "empty", "timeout", "failed"}:
                raise ValueError("knowledge timing outcome is invalid")
            sample_count = data["sampleCount"]
            if type(sample_count) is not int or not 1 <= sample_count <= 4096:
                raise ValueError("knowledge timing sample count is invalid")
            for key in (
                "lastMs",
                "p50Ms",
                "p95Ms",
                "lookupElapsedMs",
                "lookupBlockingMs",
                "lookupOverlapMs",
            ):
                value = data[key]
                if type(value) is not int or not 0 <= value <= 600_000:
                    raise ValueError("knowledge timing milliseconds are invalid")
            elapsed = cast(int, data["lookupElapsedMs"])
            blocking = cast(int, data["lookupBlockingMs"])
            overlap = cast(int, data["lookupOverlapMs"])
            if elapsed != blocking + overlap:
                raise ValueError("knowledge timing milliseconds do not sum")
            if type(data["recoveryUsed"]) is not bool:
                raise TypeError("knowledge timing recovery flag must be an exact bool")
            if health_fields <= set(data):
                if type(data["lookupClosed"]) is not bool:
                    raise TypeError("knowledge lookup closed flag must be an exact bool")
                for key in (
                    "lookupDetachedCalls",
                    "lookupDetachedCallsTotal",
                    "lookupSaturationEvents",
                ):
                    value = data[key]
                    if type(value) is not int or not 0 <= value <= (1 << 53) - 1:
                        raise ValueError("knowledge lookup health counter is invalid")
        if kind == "search_egress_status":
            expected = {
                "available",
                "consentVersion",
                "disclosureDigest",
                "searchEgressState",
            }
            if set(data) != expected:
                raise ValueError("search egress status data has an invalid shape")
            available = data["available"]
            state = data["searchEgressState"]
            version = data["consentVersion"]
            digest = data["disclosureDigest"]
            if type(available) is not bool:
                raise TypeError("search egress availability must be an exact bool")
            if state not in {"unavailable", "idle", "active"}:
                raise ValueError("search egress state is invalid")
            if available != (state != "unavailable"):
                raise ValueError("search egress availability contradicts its state")
            if version != "realtime-search-egress-consent-v1":
                raise ValueError("search egress consent version is invalid")
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("search egress disclosure digest is invalid")
        if len(data) > 16:
            raise ValueError("event data exceeds the field bound")
        projected: dict[str, PublicValue] = {}
        aggregate_string_chars = 0
        for key, value in data.items():
            if type(key) is not str or _KEY.fullmatch(key) is None:
                raise ValueError("event data key is invalid")
            if type(value) not in (str, int, bool) and value is not None:
                raise TypeError("event data value has an invalid primitive type")
            if type(value) is str:
                if len(value) > 4096:
                    raise ValueError("event string exceeds the character bound")
                aggregate_string_chars += len(value)
                if aggregate_string_chars > 8192:
                    raise ValueError("event aggregate string data exceeds the bound")
                if _PRIVATE_HANDLE.search(value) is not None:
                    raise ValueError("event data contains a private handle")
            if type(value) is int and not -(1 << 53) + 1 <= value <= (1 << 53) - 1:
                raise ValueError("event integer is outside the browser-safe range")
            projected[key] = value

        if not reserved:
            self.ensure_capacity()
        observed = self._clock()
        if type(observed) not in (int, float):
            raise TypeError("clock must return an exact number")
        if not math.isfinite(observed) or observed < 0:
            raise ValueError("clock must return a finite non-negative number")
        self._sequence += 1
        event = BrowserPublicEvent(
            sequence=self._sequence,
            kind=kind,
            monotonic_ms=float(observed) * 1000,
            data=MappingProxyType(projected),
        )
        self._events.append(event)
        return event

    @staticmethod
    def _validate_capture_status(data: dict[str, PublicValue]) -> None:
        expected = {
            "available",
            "captureState",
            "consentVersion",
            "disclosureDigest",
            "retentionHours",
        }
        if type(data) is not dict or set(data) != expected:
            raise ValueError("capture status data has an invalid shape")
        available = data["available"]
        state = data["captureState"]
        consent_version = data["consentVersion"]
        digest = data["disclosureDigest"]
        retention = data["retentionHours"]
        if type(available) is not bool:
            raise TypeError("capture status availability must be an exact bool")
        if state not in {
            "unavailable",
            "idle",
            "active",
            "revoked_purging",
            "purge_failed",
            "faulted",
        }:
            raise ValueError("capture status state is invalid")
        if available != (state != "unavailable"):
            raise ValueError("capture status availability contradicts its state")
        if consent_version != "realtime-evidence-consent-v1":
            raise ValueError("capture status consent version is invalid")
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("capture status disclosure digest is invalid")
        if type(retention) is not int or not 1 <= retention <= 168:
            raise ValueError("capture status retention is invalid")

    def publish_advisory(
        self,
        kind: str,
        data: dict[str, PublicValue],
    ) -> BrowserPublicEvent | None:
        """Publish a droppable display event without latching capacity overflow."""

        # The guard must mirror ensure_capacity() exactly, reservations included.
        # A narrower test would fall through to publish() and latch _overflowed,
        # permanently killing the projection this method promises never to fail.
        if (
            self._overflowed
            or len(self._events) + len(self._capture_status_reservations) + 1
            > self._capacity
        ):
            return None
        return self.publish(kind, data)

    def events_after(self, sequence: int) -> tuple[BrowserPublicEvent, ...]:
        if type(sequence) is not int:
            raise TypeError("sequence must be an exact integer")
        if sequence < 0:
            raise ValueError("sequence must not be negative")
        return tuple(event for event in self._events if event.sequence > sequence)[:32]

    def acknowledge_through(self, sequence: int) -> None:
        """Release acknowledged history without recovering an overflowed projection."""

        if type(sequence) is not int:
            raise TypeError("sequence must be an exact integer")
        if sequence < 0 or sequence > self._sequence:
            raise ValueError("acknowledgement sequence is outside the published range")
        self._events = [event for event in self._events if event.sequence > sequence]
