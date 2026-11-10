"""Separately frozen cancellation target-resolution fixtures and their isolated scorer.

The expected outcome of every case is derived from visible text alone: the case names the
key phrase it hinges on, the phrase must occur in the utterance, and the number of visible
active-work descriptions containing it decides the outcome (one: target that description;
none: no_match; two or more: clarify). Internal task ids are validated only so that they
can never be visible and never decide an outcome. Scoring stays in its own matched,
mismatched and ambiguous buckets; cancellation effects and lifecycle claims are scored
elsewhere and are refused here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

TARGET_RESOLUTION_SHA256 = "39bb0bbf823f55df07c5e2e73ca84ceafc3ae6c0dc31701bf03db484d8558351"

_MAX_INPUT_BYTES = 64 * 1024
_MAX_CASES = 64
_MAX_ACTIVE_WORK = 8
_MAX_TEXT_CHARS = 512
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CASE_ID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
# Mirrors the model-visible task id contract in hermes_realtime.conversation.context.
_TASK_ID = re.compile(r"task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,122}\Z")
_OUTCOMES = ("target", "no_match", "clarify")
_BUCKETS = {"target": "matched", "no_match": "mismatched", "clarify": "ambiguous"}


@dataclass(frozen=True, slots=True)
class ActiveWork:
    """One visible active-work description and its internal, never-visible task id."""

    task_id: str
    description: str


@dataclass(frozen=True, slots=True)
class TargetCase:
    """One case whose expectation has been re-derived from visible text."""

    case_id: str
    utterance: str
    key_phrase: str
    active_work: tuple[ActiveWork, ...]
    outcome: str
    target_description: str | None

    @property
    def bucket(self) -> str:
        return _BUCKETS[self.outcome]


@dataclass(frozen=True, slots=True)
class TargetFixture:
    fixture_id: str
    sha256: str
    cases: tuple[TargetCase, ...]


def _refuse(kind: str) -> ValueError:
    return ValueError(f"target_resolution_{kind}_refused")


_REFUSAL = re.compile(r"target_resolution_[a-z]+_refused\Z")


@contextlib.contextmanager
def _refusal_evidence(stage: str) -> Iterator[None]:
    """Emit one bounded marker line when a stage stops: its category, never input text."""
    failure: str | None = None
    try:
        yield
    except BaseException as error:
        message = str(error) if type(error) is ValueError else ""
        failure = message if _REFUSAL.fullmatch(message) else type(error).__name__
        raise
    finally:
        if failure is not None:
            print(
                "[target-resolution] " + json.dumps({"stage": stage, "refusal": failure}),
                file=sys.stderr,
                flush=True,
            )


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _refuse("schema")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _refuse("schema")


def _text(value: object, kind: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > _MAX_TEXT_CHARS:
        raise _refuse(kind)
    return value


def _contains(text: str, pattern: re.Pattern[str]) -> bool:
    return pattern.search(" ".join(text.casefold().split())) is not None


def derive_outcome(
    utterance: str, key_phrase: str, descriptions: tuple[str, ...]
) -> tuple[str, str | None]:
    """Derive the expected outcome from the utterance and visible descriptions only."""
    phrase = re.escape(" ".join(key_phrase.casefold().split()))
    pattern = re.compile(r"(?<!\w)" + phrase + r"(?!\w)")
    if not _contains(utterance, pattern):
        raise _refuse("case")
    matches = tuple(text for text in descriptions if _contains(text, pattern))
    if not matches:
        return "no_match", None
    if len(matches) == 1:
        return "target", matches[0]
    return "clarify", None


def _parse_case(row: object) -> TargetCase:
    if type(row) is not dict or set(row) != {
        "id", "utterance", "key_phrase", "active_work", "expected",
    }:
        raise _refuse("case")
    case_id = row["id"]
    if type(case_id) is not str or _CASE_ID.fullmatch(case_id) is None:
        raise _refuse("case")
    utterance = _text(row["utterance"], "case")
    key_phrase = _text(row["key_phrase"], "case")
    items = row["active_work"]
    if type(items) is not list or not 1 <= len(items) <= _MAX_ACTIVE_WORK:
        raise _refuse("case")
    work: list[ActiveWork] = []
    for item in items:
        if type(item) is not dict or set(item) != {"task_id", "description"}:
            raise _refuse("case")
        task_id = item["task_id"]
        if type(task_id) is not str or _TASK_ID.fullmatch(task_id) is None:
            raise _refuse("case")
        work.append(ActiveWork(task_id, _text(item["description"], "case")))
    ids = tuple(item.task_id for item in work)
    descriptions = tuple(item.description for item in work)
    if len(set(ids)) != len(ids):
        raise _refuse("case")
    visible = (utterance, key_phrase, *descriptions)
    if any(task_id.casefold() in text.casefold() for task_id in ids for text in visible):
        raise _refuse("case")
    expected = row["expected"]
    if type(expected) is not dict or set(expected) != {"outcome", "target_description"}:
        raise _refuse("case")
    outcome, description = expected["outcome"], expected["target_description"]
    if outcome not in _OUTCOMES or (description is not None and type(description) is not str):
        raise _refuse("case")
    if (outcome, description) != derive_outcome(utterance, key_phrase, descriptions):
        raise _refuse("case")
    return TargetCase(case_id, utterance, key_phrase, tuple(work), outcome, description)


@_refusal_evidence("load")
def load_target_resolution(
    path: Path, expected_sha256: str = TARGET_RESOLUTION_SHA256
) -> TargetFixture:
    """Load the hash-pinned fixture and re-derive every expectation from visible text."""
    if type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None:
        raise _refuse("binding")
    with path.open("rb") as stream:
        raw = stream.read(_MAX_INPUT_BYTES + 1)
    observed = hashlib.sha256(raw).hexdigest()
    if not 1 <= len(raw) <= _MAX_INPUT_BYTES or observed != expected_sha256:
        raise _refuse("binding")
    document = json.loads(
        raw, object_pairs_hook=_object_without_duplicates, parse_constant=_reject_constant,
    )
    if (
        type(document) is not dict
        or set(document) != {"schema_version", "fixture_id", "derivation", "cases"}
        or type(document["schema_version"]) is not int
        or document["schema_version"] != 1
        or type(document["fixture_id"]) is not str
        or _PUBLIC_ID.fullmatch(document["fixture_id"]) is None
        or type(document["derivation"]) is not str
    ):
        raise _refuse("schema")
    rows = document["cases"]
    if type(rows) is not list or not 1 <= len(rows) <= _MAX_CASES:
        raise _refuse("schema")
    cases = tuple(_parse_case(row) for row in rows)
    if len({case.case_id for case in cases}) != len(cases) or {
        case.outcome for case in cases
    } != set(_OUTCOMES):
        raise _refuse("schema")
    return TargetFixture(document["fixture_id"], observed, cases)


@_refusal_evidence("score")
def score_target_resolution(
    fixture: TargetFixture, observations: list[dict[str, object]]
) -> dict[str, object]:
    """Score observed resolutions in matched, mismatched and ambiguous buckets only.

    An observation names its target by visible description. It must not carry
    cancellation-effect or lifecycle-claim fields; those are scored separately.
    """
    by_id = {case.case_id: case for case in fixture.cases}
    if type(observations) is not list or len(observations) != len(by_id):
        raise _refuse("observation")
    seen: dict[str, tuple[str, str | None]] = {}
    for row in observations:
        if type(row) is not dict or set(row) != {"id", "outcome", "target_description"}:
            raise _refuse("observation")
        case_id, outcome, description = row["id"], row["outcome"], row["target_description"]
        if type(case_id) is not str or case_id not in by_id or case_id in seen:
            raise _refuse("observation")
        if (
            type(outcome) is not str
            or outcome not in _OUTCOMES
            or (description is not None and type(description) is not str)
            or (outcome == "target") != (description is not None)
        ):
            raise _refuse("observation")
        seen[case_id] = (outcome, description)
    totals = {name: 0 for name in _BUCKETS.values()}
    incorrect: dict[str, list[str]] = {name: [] for name in _BUCKETS.values()}
    for case in fixture.cases:
        totals[case.bucket] += 1
        if seen[case.case_id] != (case.outcome, case.target_description):
            incorrect[case.bucket].append(case.case_id)
    buckets = {
        name: {"cases": total, "correct": total - len(incorrect[name]),
               "incorrect_case_ids": incorrect[name]}
        for name, total in totals.items()
    }
    return {"fixture_id": fixture.fixture_id, "fixture_sha256": fixture.sha256,
            "buckets": buckets}
