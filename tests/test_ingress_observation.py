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
