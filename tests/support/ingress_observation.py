"""Content-free summaries of the public ingress observation window."""

from __future__ import annotations

_EVENT_BATCH_LIMIT = 32
_SUPPRESSION_KINDS = (
    "echo_barge_in_confirmed",
    "echo_suppressed",
    "barge_in_non_speech_suppressed",
    "transcript_echo_suppressed",
)


def ingress_event_cursor(completed_sequence: int, ready_sequence: int) -> int:
    """Exclude both typed completion and the readiness event from a microphone window."""

    if type(completed_sequence) is not int or type(ready_sequence) is not int:
        raise TypeError("event sequences must be exact integers")
    if completed_sequence < 1 or ready_sequence < 1:
        raise ValueError("event sequences must be positive")
    return max(completed_sequence, ready_sequence)


def ingress_event_observation(
    events: list[dict[str, object]], *, after: int
) -> dict[str, object]:
    """Summarise one bounded public-event response without retaining its contents."""

    if type(after) is not int or after < 0 or len(events) > _EVENT_BATCH_LIMIT:
        return {"available": False}

    previous = after
    counts = {
        "assistant_turn_completed": 0,
        "first_foreground_token": 0,
        "speech_ended": 0,
        "user_final_emitted": 0,
        **{kind: 0 for kind in _SUPPRESSION_KINDS},
    }
    for event in events:
        if type(event) is not dict:
            return {"available": False}
        sequence = event.get("sequence")
        kind = event.get("kind")
        data = event.get("data")
        if (
            type(sequence) is not int
            or sequence <= previous
            or type(kind) is not str
            or type(data) is not dict
        ):
            return {"available": False}
        previous = sequence
        if kind in counts:
            counts[kind] += 1
        if kind == "transcript_final" and data.get("role") == "user":
            counts["user_final_emitted"] += 1
    return {
        "available": True,
        "batch_limit_reached": len(events) == _EVENT_BATCH_LIMIT,
        "counts": counts,
    }
