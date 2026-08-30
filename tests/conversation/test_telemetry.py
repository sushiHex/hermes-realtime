from __future__ import annotations

import pytest

from hermes_realtime.conversation.telemetry import (
    KnowledgeLookupTiming,
    KnowledgeTurnBudget,
    RollingRouteMetrics,
    UtteranceTicket,
    nearest_rank_percentile,
)


def test_utterance_ticket_rejects_ambiguous_or_unbounded_identity() -> None:
    ticket = UtteranceTicket(
        session_generation=7,
        media_incarnation=3,
        utterance_sequence=11,
    )
    assert ticket.public_turn_id == "session_7_media_3_utterance_11"

    with pytest.raises(TypeError, match="session_generation"):
        UtteranceTicket(session_generation=True, media_incarnation=3, utterance_sequence=11)
    with pytest.raises(ValueError, match="media_incarnation"):
        UtteranceTicket(session_generation=7, media_incarnation=0, utterance_sequence=11)


def test_knowledge_budget_uses_one_absolute_deadline() -> None:
    now = [10.0]
    budget = KnowledgeTurnBudget.start(
        ticket=UtteranceTicket(1, 2, 3),
        total_seconds=3.5,
        clock=lambda: now[0],
    )
    assert budget.admitted_at == 10.0
    assert budget.deadline == 13.5
    assert budget.remaining_seconds() == 3.5

    now[0] = 12.25
    assert budget.remaining_seconds() == 1.25
    now[0] = 20.0
    assert budget.remaining_seconds() == 0.0


def test_lookup_timing_separates_overlap_from_final_blocking() -> None:
    timing = KnowledgeLookupTiming.from_monotonic_seconds(
        lookup_started_at=8.0,
        final_admitted_at=10.0,
        lookup_completed_at=11.0,
    )
    assert timing.lookup_elapsed_ms == 3_000.0
    assert timing.lookup_blocking_ms == 1_000.0
    assert timing.lookup_overlap_ms == 2_000.0

    final_only = KnowledgeLookupTiming.from_monotonic_seconds(
        lookup_started_at=10.25,
        final_admitted_at=10.0,
        lookup_completed_at=11.0,
    )
    assert final_only.lookup_elapsed_ms == 750.0
    assert final_only.lookup_blocking_ms == 750.0
    assert final_only.lookup_overlap_ms == 0.0

    with pytest.raises(ValueError, match="ordering"):
        KnowledgeLookupTiming.from_monotonic_seconds(
            lookup_started_at=11.0,
            final_admitted_at=10.0,
            lookup_completed_at=9.0,
        )


def test_rolling_route_metrics_are_bounded_and_nearest_rank() -> None:
    metrics = RollingRouteMetrics(capacity=3)
    for elapsed in (10.0, 20.0, 30.0, 40.0):
        metrics.observe(
            route="current_fact",
            backend="ddgs",
            elapsed_ms=elapsed,
        )

    summary = metrics.summary(route="current_fact", backend="ddgs")
    assert summary == {
        "route": "current_fact",
        "backend": "ddgs",
        "sampleCount": 3,
        "lastMs": 40.0,
        "p50Ms": 30.0,
        "p95Ms": 40.0,
    }
    assert nearest_rank_percentile((20.0, 30.0, 40.0), 0.5) == 30.0


def test_rolling_route_metrics_do_not_mix_routes_or_backends() -> None:
    metrics = RollingRouteMetrics(capacity=8)
    metrics.observe(route="current_fact", backend="ddgs", elapsed_ms=50.0)
    metrics.observe(route="technical", backend="exa", elapsed_ms=5.0)

    assert metrics.summary(route="current_fact", backend="ddgs")["sampleCount"] == 1
    assert metrics.summary(route="technical", backend="exa")["lastMs"] == 5.0
    assert metrics.summary(route="technical", backend="ddgs") is None
