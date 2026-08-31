from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "evaluate_conversation_quality.py"
_FIXTURE = Path(__file__).with_name("fixtures") / "conversation_quality_cases.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_conversation_quality", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passing_observations(evaluator: ModuleType, corpus: object) -> tuple[object, ...]:
    observations = []
    for case in corpus.cases:
        response = " ".join(case.requirements.required_any[: case.requirements.min_matches]) + "."
        observations.append(
            evaluator.CaseObservation(
                case_id=case.case_id,
                response=response,
                segments=(response,),
                latency_ms=10.0,
            )
        )
    return tuple(observations)


def test_conversation_quality_corpus_preserves_multiturn_context_and_limits() -> None:
    evaluator = _load_script()

    corpus = evaluator.load_corpus(_FIXTURE)
    correction = next(case for case in corpus.cases if case.case_id == "correction")
    snapshot = evaluator.snapshot_for_case(correction, revision=7)

    assert len(corpus.cases) == 12
    assert snapshot.revision == 7
    assert [message.role for message in snapshot.messages] == ["user", "assistant", "user"]
    assert snapshot.messages[-1].text.endswith("after the first token.")
    assert correction.limits.max_words == 35
    assert correction.limits.allow_structured_list is False
    assert correction.requirements.min_matches == 1
    assert "latency" in correction.requirements.required_any


def test_conversation_quality_corpus_rejects_schema_drift(tmp_path: Path) -> None:
    evaluator = _load_script()
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    document["cases"][0]["limits"]["unexpected"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="exact fields"):
        evaluator.load_corpus(invalid)


def test_response_metrics_use_emitted_segment_boundaries_and_anchored_boilerplate() -> None:
    evaluator = _load_script()

    structured = evaluator.response_metrics(
        "Three easy ways.\n1. Salt the eggs early.\n2. Cook gently.\n3. Finish with butter."
    )
    natural = evaluator.response_metrics("There are two likely causes. Where are you seeing it?")

    assert structured["listMarkers"] == 3
    assert structured["sentences"] == 4
    assert natural["boilerplate"] == []


def test_conversation_quality_report_fails_chattiness_and_emitted_structure() -> None:
    evaluator = _load_script()
    corpus = evaluator.load_corpus(_FIXTURE)
    observations = list(_passing_observations(evaluator, corpus))
    tiny_index = next(
        index for index, case in enumerate(corpus.cases) if case.case_id == "tiny_fact"
    )
    segments = (
        "Absolutely.",
        "Here are several unnecessarily long thoughts that keep going well past the answer.",
        "- First item",
        "- Second item?",
    )
    observations[tiny_index] = evaluator.CaseObservation(
        case_id="tiny_fact",
        response=" ".join(segments),
        segments=segments,
        latency_ms=10.0,
    )

    report = evaluator.quality_report(
        corpus=corpus,
        observations=tuple(observations),
        shadow_tool_calls=(),
        shadow_knowledge_calls=(),
        model="test-model",
        effort="low",
        label="test",
    )

    assert report["gate"] == "failed"
    tiny = next(item for item in report["cases"] if item["id"] == "tiny_fact")
    assert set(tiny["violations"]) >= {
        "max_words",
        "max_questions",
        "unrequested_structure",
        "boilerplate",
        "relevance",
    }


def test_conversation_quality_report_rejects_generic_irrelevant_replies() -> None:
    evaluator = _load_script()
    corpus = evaluator.load_corpus(_FIXTURE)
    observations = tuple(
        evaluator.CaseObservation(
            case_id=case.case_id,
            response="Brief and direct.",
            segments=("Brief and direct.",),
            latency_ms=10.0,
        )
        for case in corpus.cases
    )

    report = evaluator.quality_report(
        corpus=corpus,
        observations=observations,
        shadow_tool_calls=(),
        shadow_knowledge_calls=(),
        model="test-model",
        effort="low",
        label="test",
    )

    assert report["gate"] == "failed"
    assert all("relevance" in case["violations"] for case in report["cases"])


def test_conversation_quality_report_passes_bounded_relevant_conversation() -> None:
    evaluator = _load_script()
    corpus = evaluator.load_corpus(_FIXTURE)

    report = evaluator.quality_report(
        corpus=corpus,
        observations=_passing_observations(evaluator, corpus),
        shadow_tool_calls=(),
        shadow_knowledge_calls=(),
        model="test-model",
        effort="low",
        label="test",
    )

    assert report["gate"] == "passed"
    assert report["summary"] == {
        "cases": 12,
        "violating_cases": 0,
        "shadow_tool_calls": 0,
        "shadow_knowledge_calls": 0,
    }

    searched = evaluator.quality_report(
        corpus=corpus,
        observations=_passing_observations(evaluator, corpus),
        shadow_tool_calls=(),
        shadow_knowledge_calls=("unexpected lookup",),
        model="test-model",
        effort="low",
        label="test",
    )
    assert searched["gate"] == "failed"
    assert searched["summary"]["shadow_knowledge_calls"] == 1
