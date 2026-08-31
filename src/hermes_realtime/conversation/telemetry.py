"""Bounded, content-free telemetry for exact source-backed conversation turns."""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, cast

_MAX_BROWSER_INTEGER: Final = (1 << 53) - 1
_LABEL = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


def _finite_non_negative(name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be an exact number")
    normalized = float(cast(float, value))
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def nearest_rank_percentile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic nearest-rank percentile over finite timings."""

    if type(quantile) is not float:
        raise TypeError("quantile must be an exact float")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    if not values:
        raise ValueError("percentile requires at least one value")
    normalized = tuple(_finite_non_negative("percentile value", value) for value in values)
    ordered = sorted(normalized)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class UtteranceTicket:
    """Exact media-generation ownership for one admitted utterance."""

    session_generation: int
    media_incarnation: int
    utterance_sequence: int

    def __post_init__(self) -> None:
        for name, value in (
            ("session_generation", self.session_generation),
            ("media_incarnation", self.media_incarnation),
            ("utterance_sequence", self.utterance_sequence),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if not 1 <= value <= _MAX_BROWSER_INTEGER:
                raise ValueError(f"{name} is outside the browser-safe range")

    @property
    def public_turn_id(self) -> str:
        return (
            f"session_{self.session_generation}_media_{self.media_incarnation}"
            f"_utterance_{self.utterance_sequence}"
        )


@dataclass(frozen=True, slots=True)
class KnowledgeTurnBudget:
    """One absolute deadline shared by every lookup operation for a final turn."""

    ticket: UtteranceTicket
    admitted_at: float
    deadline: float
    _clock: Callable[[], float] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.ticket) is not UtteranceTicket:
            raise TypeError("ticket must be an exact UtteranceTicket")
        admitted = _finite_non_negative("admitted_at", self.admitted_at)
        deadline = _finite_non_negative("deadline", self.deadline)
        if deadline <= admitted:
            raise ValueError("knowledge deadline must follow final admission")
        if not callable(self._clock):
            raise TypeError("clock must be callable")
        object.__setattr__(self, "admitted_at", admitted)
        object.__setattr__(self, "deadline", deadline)

    @classmethod
    def start(
        cls,
        *,
        ticket: UtteranceTicket,
        total_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> KnowledgeTurnBudget:
        total = _finite_non_negative("total_seconds", total_seconds)
        if not 0 < total <= 10:
            raise ValueError("total_seconds must be between 0 and 10")
        observed = _finite_non_negative("clock", clock())
        return cls(
            ticket=ticket,
            admitted_at=observed,
            deadline=observed + total,
            _clock=clock,
        )

    def remaining_seconds(self) -> float:
        observed = _finite_non_negative("clock", self._clock())
        return max(0.0, self.deadline - observed)


@dataclass(frozen=True, slots=True)
class KnowledgeLookupTiming:
    """Exact content-free lookup elapsed, final blocking, and overlapped time."""

    lookup_elapsed_ms: float
    lookup_blocking_ms: float
    lookup_overlap_ms: float

    def __post_init__(self) -> None:
        elapsed = _finite_non_negative("lookup_elapsed_ms", self.lookup_elapsed_ms)
        blocking = _finite_non_negative("lookup_blocking_ms", self.lookup_blocking_ms)
        overlap = _finite_non_negative("lookup_overlap_ms", self.lookup_overlap_ms)
        if not math.isclose(elapsed, blocking + overlap, abs_tol=1e-6):
            raise ValueError("lookup timing components do not sum to elapsed time")
        object.__setattr__(self, "lookup_elapsed_ms", elapsed)
        object.__setattr__(self, "lookup_blocking_ms", blocking)
        object.__setattr__(self, "lookup_overlap_ms", overlap)

    @classmethod
    def from_monotonic_seconds(
        cls,
        *,
        lookup_started_at: float,
        final_admitted_at: float,
        lookup_completed_at: float,
    ) -> KnowledgeLookupTiming:
        started = _finite_non_negative("lookup_started_at", lookup_started_at)
        admitted = _finite_non_negative("final_admitted_at", final_admitted_at)
        completed = _finite_non_negative("lookup_completed_at", lookup_completed_at)
        if completed < started:
            raise ValueError("lookup timing ordering is invalid")
        elapsed = completed - started
        blocking = max(0.0, completed - max(started, admitted))
        blocking = min(elapsed, blocking)
        overlap = elapsed - blocking
        return cls(
            lookup_elapsed_ms=elapsed * 1000,
            lookup_blocking_ms=blocking * 1000,
            lookup_overlap_ms=overlap * 1000,
        )


class RollingRouteMetrics:
    """Bounded per-route/backend latency samples with deterministic summaries."""

    def __init__(self, *, capacity: int = 256) -> None:
        if type(capacity) is not int:
            raise TypeError("capacity must be an exact integer")
        if not 1 <= capacity <= 4096:
            raise ValueError("capacity must be between 1 and 4096")
        self._capacity = capacity
        self._samples: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=self._capacity)
        )

    @staticmethod
    def _label(name: str, value: object) -> str:
        if type(value) is not str or _LABEL.fullmatch(value) is None:
            raise ValueError(f"{name} is invalid")
        return value

    def observe(self, *, route: str, backend: str, elapsed_ms: float) -> None:
        key = (self._label("route", route), self._label("backend", backend))
        elapsed = _finite_non_negative("elapsed_ms", elapsed_ms)
        self._samples[key].append(elapsed)

    def summary(self, *, route: str, backend: str) -> dict[str, str | int | float] | None:
        normalized_route = self._label("route", route)
        normalized_backend = self._label("backend", backend)
        samples = self._samples.get((normalized_route, normalized_backend))
        if not samples:
            return None
        values = tuple(samples)
        return {
            "route": normalized_route,
            "backend": normalized_backend,
            "sampleCount": len(values),
            "lastMs": values[-1],
            "p50Ms": nearest_rank_percentile(values, 0.5),
            "p95Ms": nearest_rank_percentile(values, 0.95),
        }
