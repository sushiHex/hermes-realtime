from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CORPUS = Path(__file__).parent / "fixtures" / "natural_work_intent_cases.json"
_EXPECTED_COUNTS = {
    "positive_start": 20,
    "positive_cancel": 20,
    "negative": 25,
    "adversarial": 20,
    "cancellation_safety_negative": 15,
}
_SAFETY_CATEGORIES = {
    "negative",
    "adversarial",
    "cancellation_safety_negative",
}


def _load() -> dict[str, Any]:
    value = json.loads(_CORPUS.read_text(encoding="utf-8"))
    assert type(value) is dict
    return value


def test_natural_work_intent_corpus_is_bounded_and_complete() -> None:
    corpus = _load()
    assert set(corpus) == {"schema_version", "description", "acceptance", "cases"}
    assert corpus["schema_version"] == 1
    cases = corpus["cases"]
    assert type(cases) is list
    assert len(cases) == 100

    identifiers: set[str] = set()
    counts = {category: 0 for category in _EXPECTED_COUNTS}
    for case in cases:
        assert type(case) is dict
        assert set(case) == {"id", "category", "utterance", "expected_tool"}
        assert type(case["id"]) is str and case["id"] not in identifiers
        identifiers.add(case["id"])
        category = case["category"]
        assert category in counts
        counts[category] += 1
        assert type(case["utterance"]) is str
        assert case["utterance"].strip() == case["utterance"]
        assert 1 <= len(case["utterance"]) <= 1024
        expected = case["expected_tool"]
        assert expected in {None, "start_work", "cancel_active_work"}
        if category == "positive_start":
            assert expected == "start_work"
        elif category == "positive_cancel":
            assert expected == "cancel_active_work"
        else:
            assert expected is None

    assert counts == _EXPECTED_COUNTS


def test_natural_work_safety_denominator_and_thresholds_are_explicit() -> None:
    corpus = _load()
    acceptance = corpus["acceptance"]
    assert acceptance == {
        "positive_start_recall_min": 0.95,
        "positive_cancel_recall_min": 0.95,
        "positive_misroutes_max": 0,
        "safety_negative_false_positives_max": 0,
    }
    safety_cases = [case for case in corpus["cases"] if case["category"] in _SAFETY_CATEGORIES]
    assert len(safety_cases) == 60
    assert all(case["expected_tool"] is None for case in safety_cases)
