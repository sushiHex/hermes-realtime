from __future__ import annotations

import asyncio
import json

import pytest

from tests.integration import test_qualification_full_host_ingress as full_host_ingress
from tests.support.ingress_observation import (
    ingress_event_cursor,
    ingress_event_observation,
)


def _event(sequence: int, kind: str, data: dict[str, object]) -> dict[str, object]:
    return {"data": data, "kind": kind, "monotonicMs": 0, "sequence": sequence}


def test_ingress_event_observation_counts_only_the_microphone_window_without_content() -> None:
    secret = "microphone transcript must not be retained"

    observation = ingress_event_observation(
        [
            _event(11, "speech_ended", {}),
            _event(12, "transcript_final", {"role": "user", "text": secret}),
            _event(13, "transcript_final", {"role": "assistant", "text": secret}),
            _event(14, "first_foreground_token", {}),
            _event(15, "assistant_turn_completed", {}),
            _event(16, "echo_suppressed", {}),
            _event(17, "transcript_echo_suppressed", {}),
        ],
        after=10,
    )

    assert observation == {
        "available": True,
        "batch_limit_reached": False,
        "counts": {
            "assistant_turn_completed": 1,
            "barge_in_non_speech_suppressed": 0,
            "echo_barge_in_confirmed": 0,
            "echo_suppressed": 1,
            "first_foreground_token": 1,
            "speech_ended": 1,
            "transcript_echo_suppressed": 1,
            "user_final_emitted": 1,
        },
    }
    assert secret not in json.dumps(observation, sort_keys=True)


def test_ingress_event_observation_marks_a_full_public_event_batch_as_incomplete() -> None:
    observation = ingress_event_observation(
        [_event(sequence, "speech_ended", {}) for sequence in range(1, 33)], after=0
    )

    assert observation["available"] is True
    assert observation["batch_limit_reached"] is True
    assert observation["counts"] == {
        "assistant_turn_completed": 0,
        "barge_in_non_speech_suppressed": 0,
        "echo_barge_in_confirmed": 0,
        "echo_suppressed": 0,
        "first_foreground_token": 0,
        "speech_ended": 32,
        "transcript_echo_suppressed": 0,
        "user_final_emitted": 0,
    }


def test_ingress_event_observation_refuses_a_reset_or_nonmonotonic_cursor() -> None:
    assert ingress_event_observation([_event(10, "speech_ended", {})], after=10) == {
        "available": False
    }


@pytest.mark.parametrize(
    ("completed_sequence", "ready_sequence", "expected"),
    [(14, 9, 14), (9, 14, 14)],
)
def test_ingress_event_cursor_excludes_both_typed_baseline_events(
    completed_sequence: int, ready_sequence: int, expected: int
) -> None:
    assert ingress_event_cursor(completed_sequence, ready_sequence) == expected


@pytest.mark.asyncio
async def test_best_effort_event_fetch_summarises_the_public_window_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private microphone transcript"

    async def available_events(**_kwargs: object) -> tuple[int, list[dict[str, object]]]:
        return 200, [_event(11, "transcript_final", {"role": "user", "text": secret})]

    monkeypatch.setattr(full_host_ingress, "_events", available_events)

    observation = await full_host_ingress._best_effort_ingress_event_observation(
        port=1,
        origin="http://127.0.0.1:1",
        token="test-token",
        after=10,
    )

    assert observation["available"] is True
    assert observation["counts"] == {
        "assistant_turn_completed": 0,
        "barge_in_non_speech_suppressed": 0,
        "echo_barge_in_confirmed": 0,
        "echo_suppressed": 0,
        "first_foreground_token": 0,
        "speech_ended": 0,
        "transcript_echo_suppressed": 0,
        "user_final_emitted": 1,
    }
    assert secret not in json.dumps(observation, sort_keys=True)


@pytest.mark.asyncio
async def test_best_effort_event_fetch_does_not_mask_the_primary_acceptance_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private event fetch error"

    async def failed_events(**_kwargs: object) -> tuple[int, list[dict[str, object]]]:
        raise RuntimeError(secret)

    monkeypatch.setattr(full_host_ingress, "_events", failed_events)
    event_observation: dict[str, object] | None = None

    async def primary_acceptance_wait() -> None:
        nonlocal event_observation
        try:
            raise TimeoutError("evidence acceptance bound expired")
        finally:
            event_observation = await full_host_ingress._best_effort_ingress_event_observation(
                port=1,
                origin="http://127.0.0.1:1",
                token="test-token",
                after=10,
            )

    with pytest.raises(TimeoutError, match="acceptance bound expired"):
        await primary_acceptance_wait()
    assert event_observation == {"available": False}
    assert secret not in json.dumps(event_observation)


@pytest.mark.asyncio
async def test_best_effort_event_fetch_reports_unavailable_when_its_own_bound_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked_events(**_kwargs: object) -> tuple[int, list[dict[str, object]]]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(full_host_ingress, "_events", blocked_events)
    monkeypatch.setattr(full_host_ingress, "_INGRESS_EVENT_FETCH_TIMEOUT_SECONDS", 0.001)

    assert await full_host_ingress._best_effort_ingress_event_observation(
        port=1,
        origin="http://127.0.0.1:1",
        token="test-token",
        after=10,
    ) == {"available": False}


_CONSENT_ACTIVATED: tuple[int, dict[str, object]] = (
    200,
    {"captureState": "active", "result": "consent_activated", "sequence": 1},
)
_CONSENT_TIMED_OUT: tuple[int, dict[str, object]] = (
    503,
    {"captureState": "idle", "error": "control_timeout", "sequence": 1},
)
_CONSENT_SECRET = "private consent response detail"


def _capture_status(state: str) -> dict[str, object]:
    return {
        "available": True,
        "captureState": state,
        "consentVersion": "realtime-evidence-consent-v1",
        "disclosureDigest": "d" * 64,
        "retentionHours": 24,
    }


class _ConsentHost:
    """The two host routes ``_consent_capture`` uses, with scripted consent answers.

    A timed-out activation publishes its settled capture status only a few event
    polls later, as the host's retained settlement does once a stalled store
    create completes.
    """

    def __init__(
        self,
        answers: list[tuple[int, dict[str, object]]],
        *,
        settled_state: str | None = None,
    ) -> None:
        self.answers = list(answers)
        self.settled_state = settled_state
        self.consent_bodies: list[bytes] = []
        self.events = [_event(4, "capture_status", _capture_status("idle"))]
        self._polls_until_settled: int | None = None

    async def request(
        self, *, port: int, origin: str, path: str, bearer: str, body: bytes
    ) -> tuple[int, dict[str, object]]:
        if path == "/api/v1/events":
            if self._polls_until_settled == 0:
                assert self.settled_state is not None
                self.events.append(_event(5, "voice_ready", {}))
                self.events.append(
                    _event(6, "capture_status", _capture_status(self.settled_state))
                )
                self._polls_until_settled = None
            elif self._polls_until_settled is not None:
                self._polls_until_settled -= 1
            after = json.loads(body)["after"]
            return 200, {"events": [event for event in self.events if event["sequence"] > after]}
        assert path == "/api/v1/evidence-consent"
        if not self.answers:
            raise RuntimeError("unscripted consent request")
        self.consent_bodies.append(body)
        status, payload = self.answers.pop(0)
        if (
            len(self.consent_bodies) == 1
            and payload.get("error") == "control_timeout"
            and self.settled_state is not None
        ):
            self._polls_until_settled = 2
        return status, dict(payload)


async def _consent(host: _ConsentHost, monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(full_host_ingress, "_request", host.request)
    return await full_host_ingress._consent_capture(
        port=1, origin="http://127.0.0.1:1", token="test-token"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answers",
    [
        pytest.param([_CONSENT_ACTIVATED], id="within-control-bound"),
        pytest.param([_CONSENT_TIMED_OUT, _CONSENT_ACTIVATED], id="settled-after-timeout"),
        pytest.param(
            [_CONSENT_TIMED_OUT, _CONSENT_TIMED_OUT, _CONSENT_ACTIVATED],
            id="settled-before-session-records-it",
        ),
    ],
)
async def test_consent_capture_settles_a_control_timeout_to_the_same_active_capture(
    monkeypatch: pytest.MonkeyPatch, answers: list[tuple[int, dict[str, object]]]
) -> None:
    host = _ConsentHost(answers, settled_state="active")

    sent = await _consent(host, monkeypatch)

    assert sent == host.consent_bodies[0]
    assert host.answers == []
    assert len(host.consent_bodies) == len(answers)
    assert all(body == host.consent_bodies[0] for body in host.consent_bodies)
    assert json.loads(host.consent_bodies[0])["sequence"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answers", "settled_state", "diagnostic"),
    [
        pytest.param(
            [(500, {"error": "request_rejected", "detail": _CONSENT_SECRET})],
            None,
            "consent answered HTTP 500 with error request_rejected",
            id="rejected",
        ),
        pytest.param(
            [
                (
                    503,
                    {
                        "captureState": "faulted",
                        "detail": _CONSENT_SECRET,
                        "error": "writer_unavailable",
                        "sequence": 1,
                    },
                )
            ],
            None,
            "consent answered HTTP 503 with error writer_unavailable",
            id="faulted",
        ),
        pytest.param(
            [(503, {"captureState": "idle", "error": _CONSENT_SECRET, "sequence": 1})],
            None,
            "consent answered HTTP 503 with error unrecognized",
            id="unrecognized-error",
        ),
        pytest.param(
            [(200, {"captureState": "active", "result": "consent_activated", "sequence": 2})],
            None,
            "consent answered HTTP 200 with error absent",
            id="wrong-sequence",
        ),
        pytest.param(
            [(200, {"captureState": "active", "result": "consent_activated", "sequence": True})],
            None,
            "consent answered HTTP 200 with error absent",
            id="boolean-sequence",
        ),
        pytest.param(
            [_CONSENT_TIMED_OUT],
            "faulted",
            "consent settlement published capture state faulted",
            id="settled-faulted",
        ),
        pytest.param(
            [_CONSENT_TIMED_OUT, (500, {"error": "request_rejected", "detail": _CONSENT_SECRET})],
            "active",
            "consent retry answered HTTP 500 with error request_rejected",
            id="retry-rejected",
        ),
    ],
)
async def test_consent_capture_fails_any_other_answer_with_a_content_free_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[tuple[int, dict[str, object]]],
    settled_state: str | None,
    diagnostic: str,
) -> None:
    host = _ConsentHost(answers, settled_state=settled_state)

    with pytest.raises(AssertionError) as failure:
        await _consent(host, monkeypatch)

    assert str(failure.value) == diagnostic
    assert _CONSENT_SECRET not in str(failure.value)
    assert host.answers == []


@pytest.mark.asyncio
async def test_consent_capture_bounds_an_unsettled_control_timeout_by_its_hang_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _ConsentHost([_CONSENT_TIMED_OUT])
    monkeypatch.setattr(full_host_ingress, "_CONSENT_SETTLEMENT_HANG_BOUND_SECONDS", 0.2)

    with pytest.raises(TimeoutError):
        await _consent(host, monkeypatch)

    assert len(host.consent_bodies) == 1
