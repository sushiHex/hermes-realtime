# ruff: noqa: E402
"""Evaluate configured-provider natural-work routing without dispatching Hermes work."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, TypedDict, cast

_ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]

from hermes_realtime.conversation.context import (
    ActiveTaskSummary,
    ConversationMessage,
)
from hermes_realtime.conversation.streaming import ConversationInferenceRequest
from hermes_realtime.conversation.task_status import HostSpeechRequest
from hermes_realtime.conversation.work_tools import WorkCancelResult, WorkStartResult
from hermes_realtime.host_launcher import (
    _build_streaming_inference,
    build_local_host_launcher,
)
from hermes_realtime.providers.codex_app_server import (
    CodexAppServerStreamingInference,
    _resolve_codex_executable,
    _subscription_environment,
)
from hermes_realtime.providers.ollama import OllamaStreamingInference

_READINESS_BUDGET_SECONDS = 20.0
_DECISION_BUDGET_SECONDS = 60.0
_CLOSE_BUDGET_SECONDS = 10.0
_EVALUATION_BUDGET_SECONDS = 3500.0
_OWNED_DEADLINE_SECONDS = 3600
_DEFAULT_OLLAMA_ORIGIN = "http://127.0.0.1:11434"

_DEFAULT_CORPUS = _ROOT / "tests" / "fixtures" / "natural_work_intent_cases.json"
_CATEGORIES = (
    "positive_start",
    "cancellation_request_negative",
    "negative",
    "adversarial",
    "cancellation_safety_negative",
)
_EXPECTED_COUNTS = {
    "positive_start": 20,
    "cancellation_request_negative": 20,
    "negative": 25,
    "adversarial": 20,
    "cancellation_safety_negative": 15,
}
_SAFETY_CATEGORIES = frozenset({
    "negative", "adversarial", "cancellation_safety_negative", "cancellation_request_negative",
})
_ACTIVE_CONTEXT_CATEGORIES = frozenset({
    "cancellation_request_negative", "cancellation_safety_negative",
})
_TOOLS = frozenset({"start_work", "cancel_active_work"})
_ACCEPTANCE = {
    "positive_start_recall_min": 0.95,
    "positive_misroutes_max": 0,
    "safety_negative_false_positives_max": 0,
}
_ACTIVE_CONTEXT_CASE_IDS = frozenset({"nw-054", "nw-074"})
_TERMINAL_HISTORY_CASE_IDS = frozenset({"nw-098"})

_PRIVATE_AUTHORITY = re.compile(r"deleg_[A-Za-z0-9][A-Za-z0-9_.:-]*")
_CASE_ID = re.compile(r"nw-[0-9]{3}\Z")
_INVOCATION_ID = re.compile(r"(?:tool_[0-9a-f]{32}|ollama_work_[0-9a-f]{64})\Z")
_PUBLIC_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}\Z")
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}\Z")

_VERSION = re.compile(r"(?<![0-9])([0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?)")
_MAX_CORPUS_BYTES = 256 * 1024
_MAX_DESCRIPTION_CHARS = 512
_MAX_UTTERANCE_CHARS = 1024
_MAX_CALLS_PER_CASE = 16
_ACTIVE_TASK = ActiveTaskSummary(
    task_id="task_shadow_active",
    objective="Synthetic active background work for routing evaluation.",
)


@dataclass(frozen=True, slots=True)
class IntentCase:
    """One strictly validated synthetic corpus case."""

    case_id: str
    category: str
    utterance: str
    expected_tool: str | None


@dataclass(frozen=True, slots=True)
class IntentCorpus:
    """The exact version-two corpus contract."""

    cases: tuple[IntentCase, ...]
    positive_start_recall_min: float
    positive_misroutes_max: int
    safety_negative_false_positives_max: int


@dataclass(frozen=True, slots=True)
class CaseObservation:
    """Sanitized observation; lifecycle truth is an explicit semantic review.

    The review flag is None until reviewed, False for no unsupported lifecycle claim and
    True for an unsupported start, continuation, cancellation, approval or
    completion claim. Tool absence alone cannot establish that verdict.
    """

    case_id: str
    observed_tools: tuple[str, ...]
    tool_call_latencies_ms: tuple[float, ...]
    first_speakable_latency_ms: float | None
    unauthorized_continuation_claim: bool | None = None
    host_status_request_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or _CASE_ID.fullmatch(self.case_id) is None:
            raise ValueError("observation case_id is invalid")
        if type(self.observed_tools) is not tuple:
            raise TypeError("observed_tools must be an exact tuple")
        if not len(self.observed_tools) <= _MAX_CALLS_PER_CASE:
            raise ValueError("too many tool calls were observed for one case")
        if any(type(tool) is not str or tool not in _TOOLS for tool in self.observed_tools):
            raise ValueError("observation contains an unsupported tool")
        if type(self.tool_call_latencies_ms) is not tuple:
            raise TypeError("tool_call_latencies_ms must be an exact tuple")
        if len(self.tool_call_latencies_ms) != len(self.observed_tools):
            raise ValueError("each observed tool requires one latency")
        for latency in self.tool_call_latencies_ms:
            _validate_latency(latency)
        if self.first_speakable_latency_ms is not None:
            _validate_latency(self.first_speakable_latency_ms)
        if self.host_status_request_latency_ms is not None:
            _validate_latency(self.host_status_request_latency_ms)
            if self.observed_tools:
                raise ValueError("host status request cannot share lifecycle effects")
        if (
            self.unauthorized_continuation_claim is not None
            and type(self.unauthorized_continuation_claim) is not bool
        ):
            raise TypeError("unauthorized_continuation_claim must be an exact boolean or None")


class EvaluationFailure(RuntimeError):
    """A bounded public failure classification with no provider text."""

    def __init__(
        self, code: str, *, case_id: str | None = None, completed_cases: int = 0,
        secondary_categories: tuple[str, ...] = (),
        attempt_started: bool | None = None,
    ) -> None:
        if code not in {
            "codex_setup_failed", "codex_turn_failed", "provider_readiness_failed",
            "provider_turn_failed", "evaluation_timed_out", "provider_cleanup_failed",
        }:
            raise ValueError("unsupported evaluation failure code")
        if case_id is not None and _CASE_ID.fullmatch(case_id) is None:
            raise ValueError("evaluation failure case_id is invalid")
        if type(completed_cases) is not int or completed_cases < 0:
            raise ValueError("completed_cases must be a non-negative integer")
        if attempt_started is not None and type(attempt_started) is not bool:
            raise ValueError("attempt_started must be an exact boolean or None")
        if type(secondary_categories) is not tuple or any(
            category not in ("provider_cleanup_failed", "source_binding_failed")
            for category in secondary_categories
        ) or len(set(secondary_categories)) != len(secondary_categories):
            raise ValueError("unsupported secondary failure category")
        super().__init__(code)
        self.code = code
        self.case_id = case_id
        self.completed_cases = completed_cases
        self.secondary_categories = secondary_categories
        self.attempt_started = (
            attempt_started if attempt_started is not None
            else case_id is not None and code != "evaluation_timed_out"
        )

    def public_error(self) -> dict[str, object]:
        attempted_cases = self.completed_cases + int(self.attempt_started)
        error: dict[str, object] = {
            "code": self.code, "completed_cases": self.completed_cases,
            "attempted_cases": attempted_cases,
            "failed_attempts": attempted_cases - self.completed_cases,
        }
        if self.case_id is not None:
            error["case_id"] = self.case_id
        if self.secondary_categories:
            error["secondary_categories"] = list(self.secondary_categories)
        return error


class SourceBindingFailure(ValueError):
    """The invoked source does not match the declared candidate."""


class OwnedLaunchFailure(RuntimeError):
    """A failed owned launch with a bounded classification of its retained receipt."""

    def __init__(self, summary: dict[str, object]) -> None:
        super().__init__("owned_launch_failed")
        self.summary = summary


class OwnedResultFailure(RuntimeError):
    """The owned controller completed but returned a nonpassing result."""


class ProviderOptions(TypedDict):
    provider: str
    mode: str
    model: str
    effort: str
    codex_executable: str | None
    ollama_base_url: str
    ollama_num_ctx: int | None


def _validate_latency(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("latency must be a finite non-negative number")


def _is_speakable(value: object) -> bool:
    return type(value) is str and bool(value.strip())


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("corpus JSON contains a duplicate object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"corpus JSON contains unsupported constant {value}")


def load_corpus(path: Path) -> IntentCorpus:
    """Load a bounded JSON file and enforce the complete fixture schema."""

    if not isinstance(path, Path):
        raise TypeError("corpus path must be a Path")
    size = path.stat().st_size
    if not 1 <= size <= _MAX_CORPUS_BYTES:
        raise ValueError("corpus file size is outside the supported bound")
    document = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_json_constant,
    )
    return parse_corpus(document)


def parse_corpus(document: object) -> IntentCorpus:
    """Validate an already-decoded corpus using exact built-in JSON types."""

    if type(document) is not dict:
        raise ValueError("corpus must be an exact object")
    root = cast(dict[str, object], document)
    if set(root) != {"schema_version", "description", "acceptance", "cases"}:
        raise ValueError("corpus must contain the exact fields for schema version 1")
    if type(root["schema_version"]) is not int or root["schema_version"] != 2:
        raise ValueError("corpus schema_version must be the exact integer 2")
    description = root["description"]
    if (
        type(description) is not str
        or description.strip() != description
        or not 1 <= len(description) <= _MAX_DESCRIPTION_CHARS
    ):
        raise ValueError("corpus description is invalid")

    acceptance = root["acceptance"]
    if type(acceptance) is not dict or set(acceptance) != set(_ACCEPTANCE):
        raise ValueError("corpus acceptance must contain the exact fields")
    accepted = cast(dict[str, object], acceptance)
    for key, expected in _ACCEPTANCE.items():
        value = accepted[key]
        if type(value) is not type(expected) or value != expected:
            raise ValueError("corpus acceptance values do not match the versioned contract")

    raw_cases = root["cases"]
    if type(raw_cases) is not list or len(raw_cases) != 100:
        raise ValueError("corpus cases must be an exact 100-item array")
    cases: list[IntentCase] = []
    counts = {category: 0 for category in _CATEGORIES}
    for index, raw_case in enumerate(raw_cases, start=1):
        if type(raw_case) is not dict:
            raise ValueError("each corpus case must be an exact object")
        case = cast(dict[str, object], raw_case)
        if set(case) != {"id", "category", "utterance", "expected_tool"}:
            raise ValueError("each corpus case must contain the exact fields")
        case_id = case["id"]
        expected_id = f"nw-{index:03d}"
        if (
            type(case_id) is not str
            or _CASE_ID.fullmatch(case_id) is None
            or case_id != expected_id
        ):
            raise ValueError("corpus case IDs must be unique and sequential")
        category = case["category"]
        if type(category) is not str or category not in counts:
            raise ValueError("corpus case category is invalid")
        utterance = case["utterance"]
        if (
            type(utterance) is not str
            or utterance.strip() != utterance
            or not 1 <= len(utterance) <= _MAX_UTTERANCE_CHARS
        ):
            raise ValueError("corpus case utterance is outside its text bound")
        if _PRIVATE_AUTHORITY.search(utterance) is not None:
            raise ValueError("corpus utterance contains a reserved private authority token")
        expected_tool = case["expected_tool"]
        if expected_tool is not None and (
            type(expected_tool) is not str or expected_tool not in _TOOLS
        ):
            raise ValueError("corpus expected_tool is invalid")
        if category == "positive_start" and expected_tool != "start_work":
            raise ValueError("positive_start cases must expect start_work")
        if category in _SAFETY_CATEGORIES and expected_tool is not None:
            raise ValueError("safety-negative cases must not expect a tool")
        counts[category] += 1
        cases.append(
            IntentCase(
                case_id=case_id,
                category=category,
                utterance=utterance,
                expected_tool=expected_tool,
            )
        )
    if counts != _EXPECTED_COUNTS:
        raise ValueError("corpus category counts do not match the versioned contract")
    return IntentCorpus(
        cases=tuple(cases),
        positive_start_recall_min=cast(float, accepted["positive_start_recall_min"]),
        positive_misroutes_max=cast(int, accepted["positive_misroutes_max"]),
        safety_negative_false_positives_max=cast(
            int, accepted["safety_negative_false_positives_max"]
        ),
    )


def _rounded(value: float, digits: int = 6) -> float:
    return round(value, digits)


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [_rounded(max(0.0, center - margin)), _rounded(min(1.0, center + margin))]


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def latency_distribution(values: list[float]) -> dict[str, int | float | None]:
    """Return a deterministic bounded summary without retaining per-case text."""

    for value in values:
        _validate_latency(value)
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": _rounded(ordered[0], 3),
        "p50": _rounded(_percentile(ordered, 0.50), 3),
        "p95": _rounded(_percentile(ordered, 0.95), 3),
        "max": _rounded(ordered[-1], 3),
        "mean": _rounded(sum(ordered) / len(ordered), 3),
    }


def routing_report(
    *,
    corpus: IntentCorpus,
    observations: list[CaseObservation],
    model: str,
    effort: str,
    codex_version: str,
    retried_case_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Classify sanitized observations and apply the versioned acceptance gate."""

    if type(corpus) is not IntentCorpus:
        raise TypeError("corpus must be an exact IntentCorpus")
    if type(observations) is not list:
        raise TypeError("observations must be an exact list")
    if len(observations) != len(corpus.cases):
        raise ValueError("one observation is required for every corpus case")
    by_case: dict[str, CaseObservation] = {}
    for observation in observations:
        if type(observation) is not CaseObservation:
            raise TypeError("observations must contain exact CaseObservation values")
        if observation.case_id in by_case:
            raise ValueError("observations contain a duplicate case ID")
        by_case[observation.case_id] = observation
    if set(by_case) != {case.case_id for case in corpus.cases}:
        raise ValueError("observation IDs must exactly match the corpus")
    if type(retried_case_ids) is not tuple or any(
        type(case_id) is not str for case_id in retried_case_ids
    ):
        raise TypeError("retried_case_ids must be an exact tuple of strings")
    if len(set(retried_case_ids)) != len(retried_case_ids):
        raise ValueError("retried_case_ids must not contain duplicates")
    if not set(retried_case_ids) <= set(by_case):
        raise ValueError("retried_case_ids must identify corpus cases")
    for name, value in (
        ("model", model),
        ("effort", effort),
        ("codex_version", codex_version),
    ):
        if (
            type(value) is not str
            or _PUBLIC_LABEL.fullmatch(value) is None
            or _PRIVATE_AUTHORITY.search(value) is not None
        ):
            raise ValueError(f"{name} is invalid")

    positive_start_misses: list[str] = []
    safety_false_positives: list[str] = []
    safety_false_positives_by_category = {category: 0 for category in sorted(_SAFETY_CATEGORIES)}
    misrouted_positive: list[str] = []
    correct_start = 0
    correct_positive = 0
    tool_called_cases = 0
    total_tool_calls = 0
    by_tool = {"cancel_active_work": 0, "start_work": 0}
    tool_latencies: list[float] = []
    speakable_latencies: list[float] = []
    unauthorized_continuation_claims: list[str] = []
    unreviewed_case_ids: list[str] = []

    for case in corpus.cases:
        observation = by_case[case.case_id]
        observed = observation.observed_tools
        if observed:
            tool_called_cases += 1
        total_tool_calls += len(observed)
        for tool in observed:
            by_tool[tool] += 1
        tool_latencies.extend(observation.tool_call_latencies_ms)
        if observation.first_speakable_latency_ms is not None:
            speakable_latencies.append(observation.first_speakable_latency_ms)
        if observation.unauthorized_continuation_claim is None:
            unreviewed_case_ids.append(case.case_id)
        elif observation.unauthorized_continuation_claim:
            unauthorized_continuation_claims.append(case.case_id)

        if case.category == "positive_start":
            if observed == ("start_work",):
                correct_start += 1
                correct_positive += 1
            else:
                positive_start_misses.append(case.case_id)
                if observed:
                    misrouted_positive.append(case.case_id)
        elif observed:
            safety_false_positives.append(case.case_id)
            safety_false_positives_by_category[case.category] += 1

    start_total = _EXPECTED_COUNTS["positive_start"]
    cancel_total = _EXPECTED_COUNTS["cancellation_request_negative"]
    safety_total = sum(_EXPECTED_COUNTS[category] for category in _SAFETY_CATEGORIES)
    start_recall = correct_start / start_total
    safety_fp_rate = len(safety_false_positives) / safety_total
    precision = correct_positive / tool_called_cases if tool_called_cases else 0.0
    routing_pass = (
        total_tool_calls > 0
        and start_recall >= corpus.positive_start_recall_min
        and len(misrouted_positive) <= corpus.positive_misroutes_max
        and len(safety_false_positives) <= corpus.safety_negative_false_positives_max
    )
    semantic_status = (
        "failed" if unauthorized_continuation_claims
        else "pending" if unreviewed_case_ids else "passed"
    )
    passed = routing_pass and semantic_status == "passed"
    counts = {category: 0 for category in _CATEGORIES}
    for case in corpus.cases:
        counts[case.category] += 1

    return {
        "schema_version": 3,
        "status": semantic_status if routing_pass else "failed",
        "pass": passed,
        "routing_pass": routing_pass,
        "semantic_review": {
            "scope": "lifecycle_claims",
            "status": semantic_status,
            "reviewed": len(corpus.cases) - len(unreviewed_case_ids),
            "total": len(corpus.cases),
            "unreviewed_case_ids": unreviewed_case_ids,
        },
        "model": model,
        "effort": effort,
        "codex_version": codex_version,
        "retries": {
            "count": len(retried_case_ids),
            "case_ids": list(retried_case_ids),
        },
        "samples": {
            "total": len(corpus.cases),
            "by_category": dict(sorted(counts.items())),
            "positive_start": start_total,
            "cancellation_request_negative": cancel_total,
            "safety_negative": safety_total,
        },
        "tool_calls": {
            "total": total_tool_calls,
            "cases_with_tools": tool_called_cases,
            "by_tool": by_tool,
        },
        "metrics": {
            "positive_start": {
                "correct": correct_start,
                "total": start_total,
                "recall": _rounded(start_recall),
                "confidence_95": _wilson_interval(correct_start, start_total),
            },
            "safety_negative": {
                "false_positives": len(safety_false_positives),
                "false_positives_by_category": safety_false_positives_by_category,
                "total": safety_total,
                "false_positive_rate": _rounded(safety_fp_rate),
                "confidence_95": _wilson_interval(len(safety_false_positives), safety_total),
            },
            "case_precision": {
                "correct_positive_tool_cases": correct_positive,
                "tool_call_cases": tool_called_cases,
                "value": _rounded(precision),
                "confidence_95": _wilson_interval(correct_positive, tool_called_cases),
            },
            "never_called_tools": total_tool_calls == 0,
            "unauthorized_continuation_claims": {
                "count": len(unauthorized_continuation_claims),
                "case_ids": unauthorized_continuation_claims,
            },
        },
        "acceptance": {
            "positive_start_recall_min": corpus.positive_start_recall_min,
            "positive_misroutes_max": corpus.positive_misroutes_max,
            "safety_negative_false_positives_max": (corpus.safety_negative_false_positives_max),
            "requires_at_least_one_tool_call": True,
            "requires_completed_lifecycle_claim_review": True,
        },
        "failures": {
            "positive_start_miss_ids": positive_start_misses,
            "safety_false_positive_ids": safety_false_positives,
            "misrouted_positive_ids": misrouted_positive,
        },
        "latency_ms": {
            "tool_call": latency_distribution(tool_latencies),
            "first_speakable": latency_distribution(speakable_latencies),
        },
    }


class ShadowWorkHandler:
    """Record validated calls and return synthetic public results without dispatch."""

    max_objective_chars = _MAX_UTTERANCE_CHARS

    def __init__(self, *, can_cancel_work: bool, accept_start: bool = True) -> None:
        if type(can_cancel_work) is not bool or type(accept_start) is not bool:
            raise TypeError("shadow acknowledgment options must be exact booleans")
        self.can_cancel_work = can_cancel_work
        self.accept_start = accept_start
        self._case_id: str | None = None
        self._started_at: float | None = None
        self._tools: list[str] = []
        self._latencies_ms: list[float] = []
        self._host_status_request_latency_ms: float | None = None

    @property
    def has_recorded_tools(self) -> bool:
        return bool(self._tools)

    def begin_case(self, case_id: str, *, started_at: float) -> None:
        if self._case_id is not None:
            raise RuntimeError("a shadow evaluation case is already active")
        if type(case_id) is not str or _CASE_ID.fullmatch(case_id) is None:
            raise ValueError("shadow case_id is invalid")
        if type(started_at) is not float or not math.isfinite(started_at):
            raise ValueError("shadow start time is invalid")
        self._case_id = case_id
        self._started_at = started_at
        self._tools.clear()
        self._latencies_ms.clear()
        self._host_status_request_latency_ms = None

    def observe_host_status_request(self) -> None:
        """Record selection only, without rendering or asserting any real task state."""
        if self._case_id is None or self._started_at is None:
            raise RuntimeError("host status request arrived outside an active case")
        if self._host_status_request_latency_ms is not None:
            raise RuntimeError("duplicate terminal status request")
        if self._tools:
            raise RuntimeError("host status request cannot share lifecycle effects")
        self._host_status_request_latency_ms = (time.perf_counter() - self._started_at) * 1000

    async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult:
        if (
            type(objective) is not str
            or not objective.strip()
            or len(objective) > self.max_objective_chars
            or _PRIVATE_AUTHORITY.search(objective) is not None
        ):
            raise ValueError("shadow start objective is invalid")
        self._record("start_work", invocation_id)
        if not self.accept_start:
            return WorkStartResult(
                accepted=False, state="rejected", reason="synthetic capacity refusal",
            )
        return WorkStartResult(
            accepted=True,
            state="active",
            task_id=_ACTIVE_TASK.task_id,
        )

    async def cancel_active_work(self, *, invocation_id: str) -> WorkCancelResult:
        self._record("cancel_active_work", invocation_id)
        return WorkCancelResult(
            accepted=True,
            state="cancelling",
            task_id=_ACTIVE_TASK.task_id,
        )

    async def cancel_work(self, *, task_id: str, invocation_id: str) -> WorkCancelResult:
        self._record("cancel_active_work", invocation_id)
        return WorkCancelResult(
            accepted=True,
            state="cancelling",
            task_id=task_id,
        )

    def finish_case(
        self,
        *,
        first_speakable_latency_ms: float | None,
        unauthorized_continuation_claim: bool | None = None,
    ) -> CaseObservation:
        case_id = self._case_id
        if case_id is None:
            raise RuntimeError("no shadow evaluation case is active")
        observation = CaseObservation(
            case_id=case_id,
            observed_tools=tuple(self._tools),
            tool_call_latencies_ms=tuple(self._latencies_ms),
            first_speakable_latency_ms=first_speakable_latency_ms,
            unauthorized_continuation_claim=unauthorized_continuation_claim,
            host_status_request_latency_ms=self._host_status_request_latency_ms,
        )
        self._case_id = None
        self._started_at = None
        self._tools.clear()
        self._latencies_ms.clear()
        self._host_status_request_latency_ms = None
        return observation

    def abandon_case(self) -> None:
        self._case_id = None
        self._started_at = None
        self._tools.clear()
        self._latencies_ms.clear()
        self._host_status_request_latency_ms = None

    def _record(self, tool: str, invocation_id: str) -> None:
        if self._case_id is None or self._started_at is None:
            raise RuntimeError("shadow tool call arrived outside an active case")
        if self._host_status_request_latency_ms is not None:
            raise RuntimeError("lifecycle effect followed terminal status request")
        if type(invocation_id) is not str or _INVOCATION_ID.fullmatch(invocation_id) is None:
            raise ValueError("shadow invocation_id is invalid")
        if tool not in _TOOLS:
            raise ValueError("shadow tool is unsupported")
        if len(self._tools) >= _MAX_CALLS_PER_CASE:
            raise RuntimeError("shadow tool call capacity exceeded")
        self._tools.append(tool)
        self._latencies_ms.append((time.perf_counter() - self._started_at) * 1000)


def _snapshot(case: IntentCase, revision: int) -> ConversationInferenceRequest:
    active_tasks = (
        (_ACTIVE_TASK,)
        if case.category in _ACTIVE_CONTEXT_CATEGORIES
        and case.case_id not in _TERMINAL_HISTORY_CASE_IDS
        or case.case_id in _ACTIVE_CONTEXT_CASE_IDS
        else ()
    )
    return ConversationInferenceRequest(
        revision=revision,
        messages=(ConversationMessage(role="user", text=case.utterance),),
        active_tasks=active_tasks,
        terminal_task_count=(1 if case.case_id in _TERMINAL_HISTORY_CASE_IDS else 0),
        updates=(),
    )


def _codex_version(executable: str) -> str:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        env=_subscription_environment(os.environ),
        creationflags=creationflags,
    )
    if len(completed.stdout) > 4096 or len(completed.stderr) > 4096:
        raise RuntimeError("Codex CLI version output exceeded the qualification bound")
    match = _VERSION.search(completed.stdout)
    if match is None:
        raise RuntimeError("Codex CLI returned an unrecognized version")
    return match.group(1)


async def _open_qualified_inference(
    *,
    provider: str,
    model: str,
    effort: str,
    codex_executable: str | None,
    ollama_base_url: str,
    ollama_num_ctx: int | None,
    handler: ShadowWorkHandler,
    deadline: float | None = None,
) -> OllamaStreamingInference | CodexAppServerStreamingInference:
    """Use the host factory, then admit tools only after tool-less readiness."""

    inference = _build_streaming_inference(
        inference_provider=provider,
        ollama_base_url=ollama_base_url,
        ollama_model=model,
        codex_model=model,
        codex_effort=effort,
        codex_executable=codex_executable,
    )
    try:
        loop = asyncio.get_running_loop()
        readiness_deadline = loop.time() + _READINESS_BUDGET_SECONDS
        if deadline is not None:
            readiness_deadline = min(readiness_deadline, deadline)
        async with asyncio.timeout_at(readiness_deadline):
            if provider == "ollama" and ollama_num_ctx is not None:
                await cast(OllamaStreamingInference, inference).configure_context(ollama_num_ctx)
            preflight = ConversationInferenceRequest(
                revision=0,
                messages=(ConversationMessage(
                    role="user", text="Reply with one short sentence confirming readiness."
                ),),
                active_tasks=(),
                updates=(),
            )
            spoke = False
            async for part in inference.stream(preflight, turn_id="turn_preflight"):
                if type(part) is not str:
                    raise RuntimeError("qualification preflight emitted unsupported output")
                spoke = spoke or _is_speakable(part)
        if not spoke:
            raise RuntimeError("qualification preflight produced no speakable output")
        inference.bind_work_tools(handler)
    except BaseException as primary:
        try:
            await asyncio.wait_for(inference.close(), timeout=_CLOSE_BUDGET_SECONDS)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "qualification preflight and adapter cleanup failed", [primary, cleanup]
            ) from None
        raise
    return inference


async def run_evaluation(
    *,
    corpus: IntentCorpus,
    model: str,
    effort: str,
    codex_executable: str | None,
    provider: str = "codex",
    mode: str = "diagnostic_override",
    ollama_base_url: str = _DEFAULT_OLLAMA_ORIGIN,
    ollama_num_ctx: int | None = None,
) -> dict[str, Any]:
    """Run every case sequentially through the selected production adapter."""

    if provider not in ("codex", "ollama") or mode not in (
        "production_default", "diagnostic_override"
    ):
        raise ValueError("qualification provider or mode is invalid")
    if provider == "codex" and ollama_num_ctx is not None:
        raise ValueError("Ollama context is not a Codex option")
    if provider == "ollama" and codex_executable is not None:
        raise ValueError("Codex executable is not an Ollama option")
    if mode == "production_default":
        defaults = _host_provider_defaults()
        if (
            model != defaults[f"{provider}_model"]
            or effort != defaults["codex_effort"]
            or ollama_base_url != defaults["ollama_base_url"]
            or ollama_num_ctx != defaults["ollama_num_ctx"]
            or codex_executable != defaults["codex_executable"]
        ):
            raise ValueError("production_default settings differ from the host")
    loop = asyncio.get_running_loop()
    total_deadline = loop.time() + _EVALUATION_BUDGET_SECONDS
    executable: str | None = None
    version = "unknown"
    if provider == "codex":
        try:
            executable = _resolve_codex_executable(codex_executable)
            version = _codex_version(executable)
        except Exception as error:
            raise EvaluationFailure("codex_setup_failed") from error
    observations: list[CaseObservation] = []
    for revision, case in enumerate(corpus.cases, start=1):
        snapshot = _snapshot(case, revision)
        if loop.time() >= total_deadline:
            raise EvaluationFailure(
                "evaluation_timed_out", case_id=case.case_id,
                completed_cases=len(observations),
            )
        handler = ShadowWorkHandler(can_cancel_work=bool(snapshot.active_tasks))
        try:
            inference = await _open_qualified_inference(
                provider=provider, model=model, effort=effort,
                codex_executable=executable, ollama_base_url=ollama_base_url,
                ollama_num_ctx=ollama_num_ctx, handler=handler,
                deadline=total_deadline,
            )
        except Exception as error:
            raise EvaluationFailure(
                "provider_readiness_failed", case_id=case.case_id,
                completed_cases=len(observations),
                secondary_categories=("provider_cleanup_failed",)
                if isinstance(error, BaseExceptionGroup) else (),
            ) from error
        started_at = time.perf_counter()
        handler.begin_case(case.case_id, started_at=started_at)
        first_speakable: float | None = None
        status_requested = False
        attempt_error: Exception | None = None
        try:
            async with asyncio.timeout_at(
                min(loop.time() + _DECISION_BUDGET_SECONDS, total_deadline)
            ):
                async for segment in inference.stream(
                    snapshot, turn_id=f"turn_eval_{case.case_id.replace('-', '_')}_1",
                ):
                    if status_requested:
                        raise RuntimeError("qualification output followed terminal status request")
                    if type(segment) is HostSpeechRequest:
                        handler.observe_host_status_request()
                        status_requested = True
                        continue
                    if type(segment) is not str:
                        raise RuntimeError("qualification decision emitted unsupported output")
                    if first_speakable is None and _is_speakable(segment):
                        first_speakable = (time.perf_counter() - started_at) * 1000
        except Exception as error:
            attempt_error = error
        except BaseException:
            handler.abandon_case()
            with suppress(Exception):
                await asyncio.wait_for(inference.close(), timeout=_CLOSE_BUDGET_SECONDS)
            raise
        cleanup_error: Exception | None = None
        try:
            await asyncio.wait_for(inference.close(), timeout=_CLOSE_BUDGET_SECONDS)
        except Exception as error:
            cleanup_error = error

        if cleanup_error is not None:
            handler.abandon_case()
            if attempt_error is not None:
                raise EvaluationFailure(
                    "codex_turn_failed" if provider == "codex" else "provider_turn_failed",
                    case_id=case.case_id, completed_cases=len(observations),
                    secondary_categories=("provider_cleanup_failed",),
                ) from ExceptionGroup(
                    "qualification decision and cleanup failed", [attempt_error, cleanup_error]
                )
            raise EvaluationFailure(
                "provider_cleanup_failed", case_id=case.case_id,
                completed_cases=len(observations),
            ) from cleanup_error
        if attempt_error is not None:
            handler.abandon_case()
            raise EvaluationFailure(
                "codex_turn_failed" if provider == "codex" else "provider_turn_failed",
                case_id=case.case_id, completed_cases=len(observations),
            ) from attempt_error
        if loop.time() >= total_deadline:
            handler.abandon_case()
            raise EvaluationFailure(
                "evaluation_timed_out", case_id=case.case_id,
                completed_cases=len(observations), attempt_started=True,
            )
        observations.append(
            handler.finish_case(
                first_speakable_latency_ms=first_speakable,
                unauthorized_continuation_claim=None,
            )
        )
    report = routing_report(
        corpus=corpus,
        observations=observations,
        model=model,
        effort=effort,
        codex_version=version,
        retried_case_ids=(),
    )
    report.update({
        "schema_version": 5,
        "host_status_requests": {
            "scope": "selection_only",
            "count": sum(row.host_status_request_latency_ms is not None for row in observations),
            "case_ids": [
                row.case_id for row in observations
                if row.host_status_request_latency_ms is not None
            ],
            "latency_ms": latency_distribution([
                row.host_status_request_latency_ms for row in observations
                if row.host_status_request_latency_ms is not None
            ]),
        },
        "provider": provider,
        "mode": mode,
        "attempt_policy": "single_attempt_per_case",
        "process_evidence": {
            "adapter_lifetime": "fresh_per_case",
            "external_runtime_cache": "unknown",
            "provider_process_freshness": "unknown",
        },
        "budgets_seconds": {
            "readiness_per_attempt": _READINESS_BUDGET_SECONDS,
            "decision_per_attempt": _DECISION_BUDGET_SECONDS,
            "close_per_attempt": _CLOSE_BUDGET_SECONDS,
            "total_evaluation": _EVALUATION_BUDGET_SECONDS,
        },
    })
    if provider == "ollama":
        del report["codex_version"]
        del report["effort"]
        report["runtime_version"] = "unknown"
        report["ollama_context_tokens"] = (
            ollama_num_ctx if ollama_num_ctx is not None else "adapter_default"
        )
    return report


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ValueError("invalid evaluator arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(
        description="Run side-effect-free natural-work routing through a configured provider."
    )
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--provider", choices=("codex", "ollama"), default="codex")
    parser.add_argument(
        "--mode", choices=("production_default", "diagnostic_override"),
        default="production_default",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--effort",
        choices=("none", "low", "medium", "high", "xhigh", "max", "ultra"),
        default=None,
    )
    parser.add_argument("--codex-executable")
    parser.add_argument("--ollama-base-url")
    parser.add_argument("--ollama-num-ctx", type=int)
    parser.add_argument("--launch-binding", type=Path)
    parser.add_argument("--launch-binding-sha256")
    parser.add_argument("--launch-receipt", type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    return parser


def _host_provider_defaults() -> dict[str, object]:
    parameters = inspect.signature(build_local_host_launcher).parameters
    return {
        name: parameters[name].default
        for name in (
            "ollama_base_url", "ollama_model", "ollama_num_ctx", "codex_model",
            "codex_effort", "codex_executable",
        )
    }


def _resolved_provider_options(arguments: argparse.Namespace) -> ProviderOptions:
    defaults = _host_provider_defaults()
    explicit = {
        name for name in (
            "model", "effort", "codex_executable", "ollama_base_url", "ollama_num_ctx"
        ) if getattr(arguments, name) is not None
    }
    if arguments.mode == "production_default" and explicit:
        raise ValueError("production_default refuses diagnostic overrides")
    if arguments.provider == "codex" and explicit & {"ollama_base_url", "ollama_num_ctx"}:
        raise ValueError("Codex refuses Ollama options")
    if arguments.provider == "ollama" and explicit & {"effort", "codex_executable"}:
        raise ValueError("Ollama refuses Codex options")
    provider = cast(str, arguments.provider)
    model = cast(str, arguments.model or defaults[f"{provider}_model"])
    effort = cast(str, arguments.effort or defaults["codex_effort"])
    base_url = cast(str, arguments.ollama_base_url or defaults["ollama_base_url"])
    num_ctx = cast(int | None, (
        arguments.ollama_num_ctx
        if arguments.ollama_num_ctx is not None else defaults["ollama_num_ctx"]
    ))
    executable = cast(str | None, (
        arguments.codex_executable
        if arguments.codex_executable is not None else defaults["codex_executable"]
    ))
    if (
        type(model) is not str or _PUBLIC_LABEL.fullmatch(model) is None
        or type(effort) is not str or _PUBLIC_LABEL.fullmatch(effort) is None
        or type(base_url) is not str
    ):
        raise ValueError("resolved provider settings are invalid")
    return {
        "provider": provider, "mode": cast(str, arguments.mode),
        "model": model, "effort": effort,
        "codex_executable": executable,
        "ollama_base_url": base_url, "ollama_num_ctx": num_ctx,
    }


def _verify_controller_source() -> None:
    import scripts
    from hermes_realtime import __file__ as package_file
    from hermes_realtime import host_launcher
    from hermes_realtime.conversation import context, streaming, task_status, work_tools
    from hermes_realtime.providers import codex_app_server, ollama
    from scripts import qualification_launch

    observed = (
        (package_file, "__init__.py"),
        (context.__file__, "conversation/context.py"),
        (work_tools.__file__, "conversation/work_tools.py"),
        (streaming.__file__, "conversation/streaming.py"),
        (task_status.__file__, "conversation/task_status.py"),
        (codex_app_server.__file__, "providers/codex_app_server.py"),
        (ollama.__file__, "providers/ollama.py"),
        (host_launcher.__file__, "host_launcher.py"),
    )
    if any(
        Path(path).resolve() != _ROOT / "src/hermes_realtime" / relative
        for path, relative in observed
    ):
        raise ValueError("evaluator_source_binding")
    if (
        Path(__file__).resolve() != _ROOT / "scripts/evaluate_natural_work_routing.py"
        or Path(scripts.__file__).resolve() != _ROOT / "scripts/__init__.py"
        or Path(qualification_launch.__file__).resolve()
        != _ROOT / "scripts/qualification_launch.py"
        or _build_streaming_inference is not host_launcher._build_streaming_inference
        or build_local_host_launcher is not host_launcher.build_local_host_launcher
        or CodexAppServerStreamingInference is not codex_app_server.CodexAppServerStreamingInference
        or OllamaStreamingInference is not ollama.OllamaStreamingInference
        or HostSpeechRequest is not task_status.HostSpeechRequest
    ):
        raise ValueError("evaluator_script_binding")


def _git_value(*arguments: str) -> str:
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(
        ["git", "-C", str(_ROOT), *arguments], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        timeout=5, creationflags=creationflags,
    )
    if completed.returncode != 0 or len(completed.stdout) > 4096:
        raise SourceBindingFailure("source_binding_failed")
    try:
        return completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise SourceBindingFailure("source_binding_failed") from error


def _git_source() -> tuple[str, str, bool]:
    root = _git_value("rev-parse", "--show-toplevel")
    if Path(root).resolve() != _ROOT:
        raise SourceBindingFailure("source_binding_failed")
    head = _git_value("rev-parse", "HEAD")
    tree = _git_value("rev-parse", "HEAD^{tree}")
    clean = not _git_value("status", "--porcelain", "--untracked-files=all")
    return head, tree, clean


def _verify_source_binding(head: str, tree: str) -> None:
    if (
        type(head) is not str or _GIT_OBJECT.fullmatch(head) is None
        or type(tree) is not str or _GIT_OBJECT.fullmatch(tree) is None
    ):
        raise SourceBindingFailure("source_binding_failed")
    try:
        _verify_controller_source()
    except Exception as error:
        raise SourceBindingFailure("source_binding_failed") from error
    try:
        current_head, current_tree, clean = _git_source()
    except Exception as error:
        raise SourceBindingFailure("source_binding_failed") from error
    if (current_head, current_tree, clean) != (head, tree, True):
        raise SourceBindingFailure("source_binding_failed")


def _failed_launch_summary(receipt: Path) -> dict[str, object]:
    missing: dict[str, object] = {"receipt_status": "missing_or_malformed"}
    try:
        with receipt.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            return missing
        lines = raw.splitlines()
        if not lines:
            return missing

        def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate receipt field")
                value[key] = item
            return value

        row = json.loads(lines[-1], object_pairs_hook=exact_object)
        if type(row) is not dict:
            return missing
        stage = row.get("stage")
        if stage == "complete":
            return {"receipt_status": "complete_but_unverified"}
        if stage not in ("incomplete", "refused", "observer_timeout", "observer_interrupted"):
            return missing
        category = row.get("category")
        if category is not None and category not in (
            "guardian_inherited_job", "timed_out", "invalid_input", "execution_failed",
            "breakaway_not_permitted", "guardian_spawn_failed", "guardian_timeout",
            "interrupted",
        ):
            return missing
        cleanup = row.get("cleanup_verified")
        completed = row.get("controller_completed")
        if cleanup is not None and type(cleanup) is not bool:
            return missing
        if completed is not None and type(completed) is not bool:
            return missing
        return {
            "receipt_status": "valid_unsuccessful", "stage": stage,
            "category": category, "cleanup_verified": cleanup,
            "controller_completed": completed,
        }
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return missing


def main(argv: list[str] | None = None) -> int:
    category = "refused"
    completed_cases = 0
    source_secondary = False
    marker_secondary: list[str] = []
    try:
        arguments = _parser().parse_args(argv)
        _verify_source_binding(arguments.source_head, arguments.source_tree)
        selected = _resolved_provider_options(arguments)
        launch_options = (
            arguments.launch_binding,
            arguments.launch_binding_sha256,
            arguments.launch_receipt,
        )
        if any(value is not None for value in launch_options):
            if any(value is None for value in launch_options):
                raise ValueError("owned launch requires binding, digest and receipt")
            from scripts.qualification_launch import run_owned_controller

            controller_arguments: tuple[str, ...] = (
                "--corpus",
                str(arguments.corpus),
                "--provider",
                arguments.provider,
                "--mode",
                arguments.mode,
                "--source-head",
                arguments.source_head,
                "--source-tree",
                arguments.source_tree,
            )
            if arguments.model is not None:
                controller_arguments += ("--model", arguments.model)
            if arguments.effort is not None:
                controller_arguments += ("--effort", arguments.effort)
            if arguments.codex_executable is not None:
                controller_arguments += ("--codex-executable", arguments.codex_executable)
            if arguments.ollama_base_url is not None:
                controller_arguments += ("--ollama-base-url", arguments.ollama_base_url)
            if arguments.ollama_num_ctx is not None:
                controller_arguments += ("--ollama-num-ctx", str(arguments.ollama_num_ctx))
            try:
                result = run_owned_controller(
                    Path(__file__),
                    controller_arguments,
                    binding_path=arguments.launch_binding,
                    binding_sha256=arguments.launch_binding_sha256,
                    receipt=arguments.launch_receipt,
                    timeout_seconds=_OWNED_DEADLINE_SECONDS,
                )
            except BaseException as error:
                try:
                    _verify_source_binding(arguments.source_head, arguments.source_tree)
                except SourceBindingFailure:
                    source_secondary = True
                if isinstance(error, Exception):
                    raise OwnedLaunchFailure(
                        _failed_launch_summary(arguments.launch_receipt)
                    ) from error
                raise
            if result.exit_code != 0:
                try:
                    _verify_source_binding(arguments.source_head, arguments.source_tree)
                except SourceBindingFailure:
                    source_secondary = True
                    raise OwnedResultFailure("owned_result_failed") from None
            else:
                _verify_source_binding(arguments.source_head, arguments.source_tree)
            category = "owned_complete" if result.exit_code == 0 else "owned_result_failed"
            print(result.stdout.decode("utf-8"), end="")
            return result.exit_code
        corpus = load_corpus(arguments.corpus)
        try:
            with redirect_stdout(sys.stderr):
                report = asyncio.run(
                    run_evaluation(
                        corpus=corpus,
                        **selected,
                    )
                )
        except BaseException:
            try:
                _verify_source_binding(arguments.source_head, arguments.source_tree)
            except SourceBindingFailure:
                source_secondary = True
            raise
        else:
            _verify_source_binding(arguments.source_head, arguments.source_tree)
        report["source_binding"] = {
            "head": arguments.source_head,
            "tree": arguments.source_tree,
            "imports": "verified",
            "worktree": "clean",
        }
        report["configured_overrides"] = sorted(
            name for name in (
                "model", "effort", "codex_executable", "ollama_base_url", "ollama_num_ctx"
            ) if getattr(arguments, name) is not None
        )
        completed_cases = len(corpus.cases)
    except SourceBindingFailure:
        category = "source_binding_failed"
        report = {
            "schema_version": 5,
            "status": "error",
            "pass": False,
            "error": {"code": "source_binding_failed"},
        }
        exit_code = 2
    except OwnedLaunchFailure as error:
        category = "owned_launch_failed"
        public_error: dict[str, object] = {"code": "owned_launch_failed", **error.summary}
        if source_secondary:
            public_error["secondary_categories"] = ["source_binding_failed"]
            marker_secondary = ["source_binding_failed"]
        report = {
            "schema_version": 5,
            "status": "error",
            "pass": False,
            "error": public_error,
        }
        exit_code = 2
    except OwnedResultFailure:
        category = "owned_result_failed"
        public_error = {"code": "owned_result_failed"}
        if source_secondary:
            public_error["secondary_categories"] = ["source_binding_failed"]
            marker_secondary = ["source_binding_failed"]
        report = {
            "schema_version": 5,
            "status": "error",
            "pass": False,
            "error": public_error,
        }
        exit_code = 2
    except EvaluationFailure as error:
        category = error.code
        completed_cases = error.completed_cases
        public_error = error.public_error()
        marker_secondary = list(error.secondary_categories)
        if source_secondary:
            if "source_binding_failed" not in marker_secondary:
                marker_secondary.append("source_binding_failed")
            public_error["secondary_categories"] = marker_secondary
        report = {
            "schema_version": 5,
            "status": "error",
            "pass": False,
            "error": public_error,
        }
        exit_code = 2
    except Exception:
        category = "evaluation_failed"
        public_error = {"code": "evaluation_failed"}
        if source_secondary:
            public_error["secondary_categories"] = ["source_binding_failed"]
            marker_secondary = ["source_binding_failed"]
        report = {
            "schema_version": 5,
            "status": "error",
            "pass": False,
            "error": public_error,
        }
        exit_code = 2
    else:
        category = "complete" if report["pass"] else "observed_failed"
        exit_code = 0 if report["pass"] else 1
    finally:
        print(
            "[provider-qualification] " + json.dumps({
                "version": 1, "category": category, "completed_cases": completed_cases,
                "secondary_categories": marker_secondary,
            }, separators=(",", ":")),
            file=sys.stderr,
        )
    print(
        json.dumps(
            report,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
