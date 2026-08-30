from __future__ import annotations

import asyncio

import pytest

from hermes_realtime.conversation.knowledge import KnowledgePrefetchCoordinator
from hermes_realtime.conversation.telemetry import KnowledgeTurnBudget, UtteranceTicket
from hermes_realtime.providers.current_facts import (
    CurrentFactEvidence,
    CurrentFactSource,
    EvidencePassage,
)


class RecordingLookup:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.closed = False

    async def lookup(self, query: str) -> CurrentFactEvidence:
        self.calls.append(query)
        await asyncio.sleep(0)
        return CurrentFactEvidence(
            query=query,
            retrieved_date="2026-08-02",
            sources=(
                CurrentFactSource(
                    title="Official release",
                    url="https://example.com/release",
                    snippet="Python 3.14 is the latest stable release.",
                ),
            ),
        )

    async def close(self) -> None:
        self.closed = True


class SequenceLookup(RecordingLookup):
    def __init__(self, evidence: list[CurrentFactEvidence]) -> None:
        super().__init__()
        self._evidence = evidence

    async def lookup(self, query: str) -> CurrentFactEvidence:
        self.calls.append(query)
        return self._evidence.pop(0)


class BlockingLookup(RecordingLookup):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def lookup(self, query: str) -> CurrentFactEvidence:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return await super().lookup(query)


def budget(ticket: UtteranceTicket) -> KnowledgeTurnBudget:
    return KnowledgeTurnBudget.start(ticket=ticket, total_seconds=1.0)


@pytest.mark.asyncio
async def test_disabled_speculation_performs_only_final_lookup() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(lookup=lookup, enabled=False)
    ticket = UtteranceTicket(1, 1, 1)

    await coordinator.observe_partial(ticket, "What is the latest stable Python release")
    await asyncio.sleep(0.02)
    assert lookup.calls == []

    evidence = await coordinator.consume(
        ticket,
        "What is the latest stable Python release?",
        budget(ticket),
    )

    assert evidence is not None and evidence.quality == "usable"
    assert lookup.calls == ["What is the latest stable Python release?"]
    await coordinator.close()
    assert lookup.closed is True


@pytest.mark.asyncio
async def test_exact_stable_partial_reuses_speculative_result() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=True,
        debounce_seconds=0.01,
        min_stable_prefix_chars=20,
        min_alphanumeric_chars=24,
    )
    ticket = UtteranceTicket(4, 2, 9)
    first = "What is the latest stable Python release"
    final = "What is the latest stable Python release right now?"

    await coordinator.observe_partial(ticket, first)
    await coordinator.observe_partial(ticket, final)
    await asyncio.sleep(0.03)
    evidence = await coordinator.consume(ticket, final, budget(ticket))

    assert evidence is not None
    assert lookup.calls == [final]
    assert coordinator.accepted_speculations == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_final_hash_mismatch_cannot_consume_stale_speculation() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=True,
        debounce_seconds=0.01,
        min_stable_prefix_chars=20,
        min_alphanumeric_chars=24,
    )
    ticket = UtteranceTicket(4, 2, 10)
    partial = "What is the latest stable Python release right now?"

    await coordinator.observe_partial(ticket, "What is the latest stable Python release")
    await coordinator.observe_partial(ticket, partial)
    for _ in range(100):
        if lookup.calls:
            break
        await asyncio.sleep(0.005)
    assert lookup.calls == [partial]
    final = "What is the latest stable Ruby release right now?"
    evidence = await coordinator.consume(ticket, final, budget(ticket))

    assert evidence is not None and evidence.query == final
    assert lookup.calls == [partial, final]
    assert coordinator.rejected_speculations == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_revoked_ticket_never_returns_evidence() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=True,
        debounce_seconds=0.01,
        min_stable_prefix_chars=20,
        min_alphanumeric_chars=24,
    )
    ticket = UtteranceTicket(8, 3, 1)
    final = "What is the latest stable Python release right now?"

    await coordinator.observe_partial(ticket, "What is the latest stable Python release")
    await coordinator.observe_partial(ticket, final)
    await coordinator.revoke(ticket, "echo_rejected")

    assert await coordinator.consume(ticket, final, budget(ticket)) is None
    assert coordinator.revocation_count == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_binding_close_revokes_only_exact_generation_and_incarnation() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(lookup=lookup, enabled=True)
    stale = UtteranceTicket(2, 7, 1)
    current = UtteranceTicket(2, 8, 1)

    await coordinator.observe_partial(stale, "What is the latest stable Python release")
    await coordinator.observe_partial(current, "What is the latest stable Python release")
    await coordinator.close_binding(2, 7)

    assert (
        await coordinator.consume(
            stale,
            "What is the latest stable Python release",
            budget(stale),
        )
        is None
    )
    assert (
        await coordinator.consume(
            current,
            "What is the latest stable Python release",
            budget(current),
        )
        is not None
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_explicit_turn_handoff_rejects_stale_evidence_then_looks_up_final() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(lookup=lookup, enabled=False)
    ticket = UtteranceTicket(3, 1, 4)
    final = "What is the latest stable Python release?"
    turn_budget = budget(ticket)

    coordinator.admit_final("session_3_turn_4", ticket, final, turn_budget)
    changed_final = "What is the latest stable Ruby release?"
    evidence = (await coordinator.consume_turn_result(
        "session_3_turn_4",
        changed_final,
    )).evidence
    assert evidence is not None and evidence.query == changed_final
    assert lookup.calls == [changed_final]
    await coordinator.close()


@pytest.mark.asyncio
async def test_expired_budget_returns_schema_valid_timeout_evidence() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(lookup=lookup, enabled=False)
    ticket = UtteranceTicket(3, 1, 5)
    now = [10.0]
    turn_budget = KnowledgeTurnBudget.start(
        ticket=ticket,
        total_seconds=1.0,
        clock=lambda: now[0],
    )
    now[0] = 12.0

    evidence = await coordinator.consume(
        ticket,
        "What is the latest stable Python release?",
        turn_budget,
    )

    assert evidence is not None and evidence.quality == "empty"
    assert len(evidence.retrieved_date) == 10
    assert lookup.calls == []
    await coordinator.close()


@pytest.mark.asyncio
async def test_long_transcript_is_inert_when_speculation_is_disabled() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(lookup=lookup, enabled=False)
    ticket = UtteranceTicket(3, 1, 6)
    long_text = "What is current? " + ("ordinary conversation " * 40)

    await coordinator.observe_partial(ticket, long_text)
    evidence = await coordinator.consume(ticket, long_text, budget(ticket))

    assert evidence is None
    assert lookup.calls == []
    await coordinator.close()


@pytest.mark.asyncio
async def test_cancelling_consumer_cancels_owned_speculative_lookup() -> None:
    lookup = BlockingLookup()
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=True,
        debounce_seconds=0.0,
        min_stable_prefix_chars=20,
        min_alphanumeric_chars=24,
    )
    ticket = UtteranceTicket(3, 1, 7)
    final = "What is the latest stable Python release right now?"
    await coordinator.observe_partial(ticket, "What is the latest stable Python release")
    await coordinator.observe_partial(ticket, final)
    await lookup.started.wait()

    consumer = asyncio.create_task(coordinator.consume(ticket, final, budget(ticket)))
    await asyncio.sleep(0)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.wait_for(lookup.cancelled.wait(), timeout=1.0)
    await coordinator.close()


@pytest.mark.asyncio
async def test_revocation_after_speculative_completion_cannot_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=True,
        debounce_seconds=0,
    )
    ticket = UtteranceTicket(7, 3, 4)
    final = "What is the latest stable Python release available now?"
    await coordinator.observe_partial(ticket, final)
    await coordinator.observe_partial(ticket, final)

    async def revoke_after_completion(
        task: asyncio.Task[CurrentFactEvidence],
        turn_budget: KnowledgeTurnBudget,
    ) -> CurrentFactEvidence:
        del turn_budget
        result = await task
        await coordinator.revoke(ticket, "race_probe")
        return result

    monkeypatch.setattr(coordinator, "_await_with_budget", revoke_after_completion)

    result = await coordinator.consume(ticket, final, budget(ticket))

    assert result is None
    assert lookup.calls == [final]
    await coordinator.close()


@pytest.mark.asyncio
async def test_consume_turn_result_reports_true_speculative_overlap() -> None:
    lookup = BlockingLookup()
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=True,
        debounce_seconds=0,
    )
    ticket = UtteranceTicket(5, 2, 9)
    final = "What is the latest stable Python release available now?"
    await coordinator.observe_partial(ticket, final)
    await coordinator.observe_partial(ticket, final)
    await asyncio.wait_for(lookup.started.wait(), timeout=1.0)
    await asyncio.sleep(0.03)
    coordinator.admit_final("session_5_turn_9", ticket, final, budget(ticket))
    consumer = asyncio.create_task(
        coordinator.consume_turn_result("session_5_turn_9", final)
    )
    await asyncio.sleep(0.03)
    lookup.release.set()

    result = await consumer

    assert result.evidence is not None
    assert result.speculative is True
    assert result.timing is not None
    assert result.timing.lookup_overlap_ms > 0
    assert result.timing.lookup_blocking_ms > 0
    assert result.timing.lookup_elapsed_ms == pytest.approx(
        result.timing.lookup_overlap_ms + result.timing.lookup_blocking_ms
    )
    await coordinator.close()


@pytest.mark.asyncio
async def test_budget_timeout_is_included_in_turn_timing() -> None:
    lookup = BlockingLookup()
    coordinator = KnowledgePrefetchCoordinator(lookup=lookup, enabled=False)
    ticket = UtteranceTicket(5, 3, 11)
    final = "What is the latest stable Python release available now?"
    turn_budget = KnowledgeTurnBudget.start(ticket=ticket, total_seconds=0.05)
    coordinator.admit_final("session_5_turn_11", ticket, final, turn_budget)

    result = await coordinator.consume_turn_result("session_5_turn_11", final)

    assert result.evidence is not None
    assert result.evidence.error == "Current-fact lookup timed out."
    assert result.timing is not None
    assert result.timing.lookup_blocking_ms > 0
    assert result.timing.lookup_overlap_ms == 0
    await coordinator.close()


@pytest.mark.asyncio
async def test_completed_prefetch_excludes_idle_time_before_final() -> None:
    lookup = BlockingLookup()
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=True,
        debounce_seconds=0,
    )
    ticket = UtteranceTicket(5, 2, 10)
    final = "What is the latest stable Python release available now?"
    await coordinator.observe_partial(ticket, final)
    await coordinator.observe_partial(ticket, final)
    await asyncio.wait_for(lookup.started.wait(), timeout=1.0)
    await asyncio.sleep(0.03)
    lookup.release.set()
    await asyncio.sleep(0.08)
    coordinator.admit_final("session_5_turn_10", ticket, final, budget(ticket))

    result = await coordinator.consume_turn_result("session_5_turn_10", final)

    assert result.evidence is not None
    assert result.speculative is True
    assert result.timing is not None
    assert result.timing.lookup_overlap_ms > 0
    assert result.timing.lookup_blocking_ms == 0
    assert result.timing.lookup_elapsed_ms < 70
    await coordinator.close()


@pytest.mark.asyncio
async def test_discarding_non_routed_turn_revokes_admitted_ticket() -> None:
    lookup = RecordingLookup()
    coordinator = KnowledgePrefetchCoordinator(lookup=lookup, enabled=True)
    ticket = UtteranceTicket(3, 1, 8)
    coordinator.admit_final(
        "session_3_turn_8",
        ticket,
        "Tell me a joke about Python.",
        budget(ticket),
    )

    await coordinator.discard_turn("session_3_turn_8", "not_source_sensitive")

    assert (
        await coordinator.consume_turn_result(
            "session_3_turn_8", "Tell me a joke about Python."
        )
    ).evidence is None
    assert coordinator.revocation_count == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_recovery_is_independent_single_attempt_with_url_dedupe() -> None:
    original = "What does LiveKit AudioStream capacity zero mean?"
    lookup = SequenceLookup(
        [
            CurrentFactEvidence(
                query=original,
                retrieved_date="2026-08-02",
                sources=(
                    CurrentFactSource(
                        title="Thin result",
                        url="https://docs.livekit.io/reference#old",
                        snippet="Thin.",
                    ),
                ),
            ),
            CurrentFactEvidence(
                query="rewritten",
                retrieved_date="2026-08-02",
                sources=(
                    CurrentFactSource(
                        title="Official reference",
                        url="https://docs.livekit.io/reference#capacity",
                        snippet=(
                            "A capacity of zero means the queue is unbounded and puts no limit "
                            "on stored frames."
                        ),
                    ),
                ),
            ),
        ]
    )
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=False,
        recovery_enabled=True,
    )
    ticket = UtteranceTicket(5, 1, 1)

    evidence = await coordinator.consume(ticket, original, budget(ticket))

    assert evidence is not None and evidence.recovery_used is True
    assert evidence.quality == "usable"
    assert len(evidence.sources) == 1
    assert len(lookup.calls) == 2
    assert lookup.calls[1] != original
    await coordinator.close()


@pytest.mark.asyncio
async def test_recovery_disabled_never_performs_second_lookup() -> None:
    original = "What does LiveKit AudioStream capacity zero mean?"
    lookup = SequenceLookup(
        [
            CurrentFactEvidence(
                query=original,
                retrieved_date="2026-08-02",
                sources=(),
            )
        ]
    )
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=False,
        recovery_enabled=False,
    )
    ticket = UtteranceTicket(5, 1, 2)

    evidence = await coordinator.consume(ticket, original, budget(ticket))

    assert evidence is not None and evidence.recovery_used is False
    assert lookup.calls == [original]
    await coordinator.close()


@pytest.mark.asyncio
async def test_recovery_preserves_and_marks_conflicting_evidence() -> None:
    original = "What does AudioStream capacity zero mean?"
    lookup = SequenceLookup(
        [
            CurrentFactEvidence(
                query=original,
                retrieved_date="2026-08-02",
                sources=(
                    CurrentFactSource(
                        title="Reference A",
                        url="https://a.example/reference",
                        snippet="Zero is unbounded.",
                        backend="primary",
                        passages=(
                            EvidencePassage(
                                source_id="source_1",
                                text="Zero is unbounded.",
                            ),
                        ),
                    ),
                ),
            ),
            CurrentFactEvidence(
                query="recovery",
                retrieved_date="2026-08-02",
                sources=(
                    CurrentFactSource(
                        title="Reference B",
                        url="https://b.example/reference",
                        snippet="Zero is bounded.",
                        backend="recovery",
                        passages=(
                            EvidencePassage(
                                source_id="source_1",
                                text="Zero is bounded.",
                            ),
                        ),
                    ),
                ),
            ),
        ]
    )
    coordinator = KnowledgePrefetchCoordinator(
        lookup=lookup,
        enabled=False,
        recovery_enabled=True,
    )
    ticket = UtteranceTicket(8, 2, 1)

    evidence = await coordinator.consume(ticket, original, budget(ticket))

    assert evidence is not None
    assert evidence.recovery_used is True
    assert evidence.conflict_detected is True
    assert evidence.backend == "mixed"
    assert len(evidence.sources) == 2
    context = evidence.model_context()
    assert "sources conflict" in context.casefold()
    assert context.count("[source_1]") == 1
    assert context.count("[source_2]") == 1
    rendered = evidence.tool_result()["sources"]
    assert type(rendered) is list
    assert [source["backend"] for source in rendered] == ["primary", "recovery"]
    await coordinator.close()
