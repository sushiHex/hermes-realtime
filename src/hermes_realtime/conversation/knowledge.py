"""Cancellation-safe foreground knowledge prefetch keyed by exact utterance identity."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from dataclasses import dataclass
from datetime import date
from urllib.parse import urldefrag

from hermes_realtime.conversation.telemetry import (
    KnowledgeLookupTiming,
    KnowledgeTurnBudget,
    UtteranceTicket,
)
from hermes_realtime.providers.current_facts import (
    CurrentFactEvidence,
    CurrentFactLookup,
    CurrentFactSource,
    contains_private_material,
    foreground_search_query,
)

_MAX_TRANSCRIPT_CHARS = 512
_TURN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_CONFLICT_TERMS = (
    ("allowed", "forbidden"),
    ("available", "unavailable"),
    ("bounded", "unbounded"),
    ("enabled", "disabled"),
    ("increase", "decrease"),
    ("required", "optional"),
    ("supported", "unsupported"),
    ("true", "false"),
)


def _evidence_words(sources: tuple[CurrentFactSource, ...]) -> frozenset[str]:
    words: set[str] = set()
    for source in sources:
        texts = [source.snippet, *(passage.text for passage in source.passages)]
        for text in texts:
            words.update(re.findall(r"[a-z]+", text.casefold()))
    return frozenset(words)


def _sources_conflict(
    primary: tuple[CurrentFactSource, ...],
    recovered: tuple[CurrentFactSource, ...],
) -> bool:
    primary_words = _evidence_words(primary)
    recovered_words = _evidence_words(recovered)
    return any(
        (left in primary_words and right in recovered_words)
        or (right in primary_words and left in recovered_words)
        for left, right in _CONFLICT_TERMS
    )


def _normalize_transcript(text: str) -> str:
    if type(text) is not str:
        raise TypeError("transcript must be an exact built-in string")
    normalized = " ".join(text.split())
    return normalized


def _transcript_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_prefix_chars(previous: str, current: str) -> int:
    limit = min(len(previous), len(current))
    index = 0
    while index < limit and previous[index] == current[index]:
        index += 1
    if index == 0:
        return 0
    if (
        index < len(previous)
        and index < len(current)
        and (previous[index].isalnum() or current[index].isalnum())
    ):
        index = previous.rfind(" ", 0, index)
        if index < 0:
            return 0
    return index


def _eligible_transcript(text: str) -> bool:
    return (
        bool(text)
        and len(text) <= _MAX_TRANSCRIPT_CHARS
        and not contains_private_material(text)
        and foreground_search_query(text) is not None
    )


@dataclass(slots=True)
class _Operation:
    transcript_hash: str
    task: asyncio.Task[CurrentFactEvidence]
    replacements: int
    started_at: list[float | None]
    completed_at: list[float | None]


@dataclass(frozen=True, slots=True)
class _AdmittedTurn:
    ticket: UtteranceTicket
    transcript_hash: str
    budget: KnowledgeTurnBudget


@dataclass(frozen=True, slots=True)
class KnowledgeConsumeResult:
    """Evidence plus exact content-free timing for one admitted turn."""

    evidence: CurrentFactEvidence | None
    timing: KnowledgeLookupTiming | None
    speculative: bool


class KnowledgePrefetchCoordinator:
    """Own speculative lookup authority for exact tickets and transcript hashes."""

    def __init__(
        self,
        *,
        lookup: CurrentFactLookup,
        enabled: bool = False,
        debounce_seconds: float = 0.12,
        min_stable_prefix_chars: int = 24,
        min_alphanumeric_chars: int = 32,
        max_replacements: int = 1,
        recovery_enabled: bool = False,
        owns_lookup: bool = True,
    ) -> None:
        if not callable(getattr(lookup, "lookup", None)):
            raise TypeError("lookup must provide lookup()")
        if any(type(value) is not bool for value in (enabled, recovery_enabled, owns_lookup)):
            raise TypeError("feature and ownership flags must be exact booleans")
        if (
            type(debounce_seconds) not in (int, float)
            or isinstance(debounce_seconds, bool)
            or not math.isfinite(float(debounce_seconds))
            or float(debounce_seconds) < 0.0
        ):
            raise ValueError("debounce_seconds must be finite and non-negative")
        for name, value, lower in (
            ("min_stable_prefix_chars", min_stable_prefix_chars, 1),
            ("min_alphanumeric_chars", min_alphanumeric_chars, 1),
            ("max_replacements", max_replacements, 0),
        ):
            if type(value) is not int or value < lower:
                raise ValueError(f"{name} must be an exact integer >= {lower}")
        self._lookup = lookup
        self._enabled = enabled
        self._debounce_seconds = float(debounce_seconds)
        self._min_stable_prefix_chars = min_stable_prefix_chars
        self._min_alphanumeric_chars = min_alphanumeric_chars
        self._max_replacements = max_replacements
        self._recovery_enabled = recovery_enabled
        self._owns_lookup = owns_lookup
        self._partials: dict[UtteranceTicket, str] = {}
        self._operations: dict[UtteranceTicket, _Operation] = {}
        self._admitted_turns: dict[str, _AdmittedTurn] = {}
        self._revoked: set[UtteranceTicket] = set()
        self._closed = False
        self.accepted_speculations = 0
        self.rejected_speculations = 0
        self.revocation_count = 0

    @property
    def lookup(self) -> CurrentFactLookup:
        """Return the coordinator-owned backend for ordinary final-turn fallback."""
        return self._lookup

    def admit_final(
        self,
        turn_id: str,
        ticket: UtteranceTicket,
        final_text: str,
        budget: KnowledgeTurnBudget,
    ) -> None:
        if type(turn_id) is not str or _TURN_ID.fullmatch(turn_id) is None:
            raise ValueError("turn_id must be a bounded public identifier")
        self._validate_ticket(ticket)
        if type(budget) is not KnowledgeTurnBudget or budget.ticket != ticket:
            raise ValueError("budget must belong to the exact utterance ticket")
        if self._closed or ticket in self._revoked:
            raise RuntimeError("cannot admit a closed or revoked utterance")
        if turn_id in self._admitted_turns:
            raise RuntimeError("turn_id is already admitted")
        normalized = _normalize_transcript(final_text)
        self._admitted_turns[turn_id] = _AdmittedTurn(
            ticket=ticket,
            transcript_hash=_transcript_hash(normalized),
            budget=budget,
        )

    async def consume_turn_result(
        self,
        turn_id: str,
        final_text: str,
    ) -> KnowledgeConsumeResult:
        if type(turn_id) is not str or _TURN_ID.fullmatch(turn_id) is None:
            raise ValueError("turn_id must be a bounded public identifier")
        admitted = self._admitted_turns.get(turn_id)
        if admitted is None:
            return KnowledgeConsumeResult(evidence=None, timing=None, speculative=False)
        normalized = _normalize_transcript(final_text)
        final_hash = _transcript_hash(normalized)
        if final_hash != admitted.transcript_hash:
            operation = self._operations.pop(admitted.ticket, None)
            if operation is not None:
                operation.task.cancel()
                self.rejected_speculations += 1
            self._partials.pop(admitted.ticket, None)
        accepted_before = self.accepted_speculations
        timing_phases: list[tuple[float, float]] = []
        try:
            evidence = await self.consume(
                admitted.ticket,
                normalized,
                admitted.budget,
                _timing_phases=timing_phases,
            )
            speculative = self.accepted_speculations > accepted_before
            elapsed_seconds = sum(end - start for start, end in timing_phases)
            overlap_seconds = sum(
                max(0.0, min(end, admitted.budget.admitted_at) - start)
                for start, end in timing_phases
            )
            overlap_seconds = min(elapsed_seconds, overlap_seconds)
            timing = None
            if evidence is not None and timing_phases:
                timing = KnowledgeLookupTiming(
                    lookup_elapsed_ms=elapsed_seconds * 1000,
                    lookup_blocking_ms=(elapsed_seconds - overlap_seconds) * 1000,
                    lookup_overlap_ms=overlap_seconds * 1000,
                )
            return KnowledgeConsumeResult(
                evidence=evidence,
                timing=timing,
                speculative=speculative,
            )
        finally:
            self._admitted_turns.pop(turn_id, None)

    async def discard_turn(self, turn_id: str, reason: str) -> None:
        """Revoke and remove an admitted turn that will not consume knowledge."""
        if type(turn_id) is not str or _TURN_ID.fullmatch(turn_id) is None:
            raise ValueError("turn_id must be a bounded public identifier")
        if type(reason) is not str or not reason.strip() or len(reason) > 64:
            raise ValueError("reason must be a non-empty bounded string")
        admitted = self._admitted_turns.pop(turn_id, None)
        if admitted is not None:
            await self.revoke(admitted.ticket, reason)

    async def observe_partial(self, ticket: UtteranceTicket, text: str) -> None:
        self._validate_ticket(ticket)
        normalized = _normalize_transcript(text)
        if self._closed or not self._enabled or ticket in self._revoked:
            return
        previous = self._partials.get(ticket)
        self._partials[ticket] = normalized
        if (
            previous is None
            or not _eligible_transcript(normalized)
            or sum(character.isalnum() for character in normalized)
            < self._min_alphanumeric_chars
            or _stable_prefix_chars(previous, normalized) < self._min_stable_prefix_chars
        ):
            return
        transcript_hash = _transcript_hash(normalized)
        existing = self._operations.get(ticket)
        replacements = 0
        if existing is not None:
            if existing.transcript_hash == transcript_hash:
                return
            if existing.replacements >= self._max_replacements:
                return
            replacements = existing.replacements + 1
            existing.task.cancel()
        started_at: list[float | None] = [None]
        completed_at: list[float | None] = [None]
        task = asyncio.create_task(
            self._debounced_lookup(ticket, normalized, started_at, completed_at),
            name=f"knowledge-prefetch-{ticket.utterance_sequence}",
        )
        self._operations[ticket] = _Operation(
            transcript_hash=transcript_hash,
            task=task,
            replacements=replacements,
            started_at=started_at,
            completed_at=completed_at,
        )

    async def consume(
        self,
        ticket: UtteranceTicket,
        final_text: str,
        budget: KnowledgeTurnBudget,
        *,
        _timing_phases: list[tuple[float, float]] | None = None,
    ) -> CurrentFactEvidence | None:
        self._validate_ticket(ticket)
        if type(budget) is not KnowledgeTurnBudget or budget.ticket != ticket:
            raise ValueError("budget must belong to the exact utterance ticket")
        normalized = _normalize_transcript(final_text)
        if self._closed or ticket in self._revoked:
            return None
        operation = self._operations.get(ticket)
        final_hash = _transcript_hash(normalized)
        if operation is not None and operation.transcript_hash == final_hash:
            result = await self._await_with_budget(operation.task, budget)
            if self._operations.get(ticket) is operation:
                self._operations.pop(ticket, None)
            if (
                _timing_phases is not None
                and operation.started_at[0] is not None
                and operation.completed_at[0] is not None
            ):
                _timing_phases.append(
                    (operation.started_at[0], operation.completed_at[0])
                )
            if self._closed or ticket in self._revoked:
                return None
            if result is not None and ticket not in self._revoked:
                self.accepted_speculations += 1
                self._partials.pop(ticket, None)
                return await self._recover_if_needed(
                    result,
                    normalized,
                    budget,
                    timing_phases=_timing_phases,
                )
        elif operation is not None:
            self.rejected_speculations += 1
            operation.task.cancel()
            self._operations.pop(ticket, None)
        self._partials.pop(ticket, None)
        if not _eligible_transcript(normalized):
            return None
        remaining = budget.remaining_seconds()
        if remaining <= 0.0:
            return self._timeout_evidence(normalized)
        lookup_started_at = time.monotonic()
        try:
            async with asyncio.timeout(remaining):
                result = await self._lookup.lookup(normalized)
            lookup_completed_at = time.monotonic()
            if _timing_phases is not None:
                _timing_phases.append((lookup_started_at, lookup_completed_at))
            if self._closed or ticket in self._revoked:
                return None
            return await self._recover_if_needed(
                result,
                normalized,
                budget,
                timing_phases=_timing_phases,
            )
        except TimeoutError:
            if _timing_phases is not None:
                _timing_phases.append((lookup_started_at, time.monotonic()))
            return self._timeout_evidence(normalized)

    async def _recover_if_needed(
        self,
        evidence: CurrentFactEvidence,
        original_query: str,
        budget: KnowledgeTurnBudget,
        *,
        timing_phases: list[tuple[float, float]] | None = None,
    ) -> CurrentFactEvidence | None:
        if self._closed or budget.ticket in self._revoked:
            return None
        if not self._recovery_enabled or evidence.quality == "usable":
            return evidence
        remaining = budget.remaining_seconds()
        if remaining <= 0.0:
            return evidence
        recovery_query = self._recovery_query(original_query)
        recovery_started_at = time.monotonic()
        try:
            async with asyncio.timeout(remaining):
                recovered = await self._lookup.lookup(recovery_query)
        except TimeoutError:
            if timing_phases is not None:
                timing_phases.append((recovery_started_at, time.monotonic()))
            if self._closed or budget.ticket in self._revoked:
                return None
            return evidence
        if timing_phases is not None:
            timing_phases.append((recovery_started_at, time.monotonic()))
        if self._closed or budget.ticket in self._revoked:
            return None
        return self._merge_recovery(evidence, recovered, original_query)

    @staticmethod
    def _recovery_query(original_query: str) -> str:
        stem = original_query.rstrip(" ?.!")
        suffix = " official documentation source"
        return f"{stem[: _MAX_TRANSCRIPT_CHARS - len(suffix)]}{suffix}"

    @staticmethod
    def _merge_recovery(
        primary: CurrentFactEvidence,
        recovered: CurrentFactEvidence,
        original_query: str,
    ) -> CurrentFactEvidence:
        by_url: dict[str, CurrentFactSource] = {}
        for source in (*primary.sources, *recovered.sources):
            url_key = urldefrag(source.url)[0].rstrip("/").casefold()
            existing = by_url.get(url_key)
            if existing is None:
                by_url[url_key] = source
                continue
            passages = []
            seen_passages: set[str] = set()
            for passage in (*existing.passages, *source.passages):
                text_key = " ".join(passage.text.split()).casefold()
                if text_key not in seen_passages:
                    seen_passages.add(text_key)
                    passages.append(passage)
            better = source if len(source.snippet) > len(existing.snippet) else existing
            by_url[url_key] = CurrentFactSource(
                title=better.title,
                url=better.url,
                snippet=better.snippet,
                backend=better.backend,
                passages=tuple(passages[:3]),
            )
        sources = tuple(by_url.values())[:8]
        source_backends = {source.backend for source in sources}
        if not source_backends:
            merged_backend = recovered.backend
        elif len(source_backends) == 1:
            merged_backend = next(iter(source_backends))
        else:
            merged_backend = "mixed"
        return CurrentFactEvidence(
            query=original_query,
            retrieved_date=recovered.retrieved_date,
            sources=sources,
            error=recovered.error if not sources else None,
            backend=merged_backend,
            recovery_used=True,
            conflict_detected=_sources_conflict(primary.sources, recovered.sources),
        )

    async def revoke(self, ticket: UtteranceTicket, reason: str) -> None:
        self._validate_ticket(ticket)
        if type(reason) is not str or not reason.strip() or len(reason) > 64:
            raise ValueError("reason must be a non-empty bounded string")
        if ticket in self._revoked:
            return
        self._revoked.add(ticket)
        self.revocation_count += 1
        self._partials.pop(ticket, None)
        operation = self._operations.pop(ticket, None)
        if operation is not None:
            operation.task.cancel()
        stale_turn_ids = [
            turn_id
            for turn_id, admitted in self._admitted_turns.items()
            if admitted.ticket == ticket
        ]
        for turn_id in stale_turn_ids:
            self._admitted_turns.pop(turn_id, None)

    async def close_binding(self, session_generation: int, media_incarnation: int) -> None:
        for value, name in (
            (session_generation, "session_generation"),
            (media_incarnation, "media_incarnation"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be an exact positive integer")
        tickets = (
            set(self._partials)
            | set(self._operations)
            | {admitted.ticket for admitted in self._admitted_turns.values()}
        )
        for ticket in tickets:
            if (
                ticket.session_generation == session_generation
                and ticket.media_incarnation == media_incarnation
            ):
                await self.revoke(ticket, "binding_closed")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for operation in self._operations.values():
            operation.task.cancel()
        tasks = [operation.task for operation in self._operations.values()]
        self._operations.clear()
        self._partials.clear()
        self._admitted_turns.clear()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._owns_lookup:
            await self._lookup.close()

    async def _debounced_lookup(
        self,
        ticket: UtteranceTicket,
        query: str,
        started_at: list[float | None],
        completed_at: list[float | None],
    ) -> CurrentFactEvidence:
        await asyncio.sleep(self._debounce_seconds)
        if self._closed or ticket in self._revoked:
            raise asyncio.CancelledError
        started_at[0] = time.monotonic()
        try:
            return await self._lookup.lookup(query)
        finally:
            completed_at[0] = time.monotonic()

    @staticmethod
    async def _await_with_budget(
        task: asyncio.Task[CurrentFactEvidence],
        budget: KnowledgeTurnBudget,
    ) -> CurrentFactEvidence | None:
        remaining = budget.remaining_seconds()
        if remaining <= 0.0:
            task.cancel()
            return None
        try:
            async with asyncio.timeout(remaining):
                return await asyncio.shield(task)
        except TimeoutError:
            task.cancel()
            return None
        except asyncio.CancelledError:
            task.cancel()
            raise

    @staticmethod
    def _timeout_evidence(query: str) -> CurrentFactEvidence:
        return CurrentFactEvidence(
            query=query,
            retrieved_date=date.today().isoformat(),
            sources=(),
            error="Current-fact lookup timed out.",
        )

    @staticmethod
    def _validate_ticket(ticket: UtteranceTicket) -> None:
        if type(ticket) is not UtteranceTicket:
            raise TypeError("ticket must be an exact UtteranceTicket")
