from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, NamedTuple

from hermes_realtime.conversation.context import ConversationContextSnapshot, ConversationMessage
from hermes_realtime.conversation.work_tools import WorkCancelResult, WorkStartResult
from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference
from hermes_realtime.providers.current_facts import CurrentFactEvidence

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CORPUS = _ROOT / "tests" / "fixtures" / "conversation_quality_cases.json"
_WORD_RE = re.compile(r"\b[\w'-]+\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_LIST_RE = re.compile(r"(?m)^\s*(?:[-*]|\d+[.)])\s+")
_MARKDOWN_RE = re.compile(r"(?m)^\s*(?:#{1,6}\s|```)|\[[^\]]+\]\([^)]+\)")
_BOILERPLATE_PATTERNS = (
    ("great question", re.compile(r"^\s*(?:that's\s+)?a?\s*great question\b", re.IGNORECASE)),
    ("absolutely", re.compile(r"^\s*absolutely\b", re.IGNORECASE)),
    ("certainly", re.compile(r"^\s*certainly\b", re.IGNORECASE)),
    ("i can help", re.compile(r"^\s*i can help\b", re.IGNORECASE)),
    ("here are", re.compile(r"^\s*here are\b", re.IGNORECASE)),
    ("let me know", re.compile(r"\blet me know\b", re.IGNORECASE)),
)


class CaseLimits(NamedTuple):
    max_words: int
    max_sentences: int
    max_questions: int
    allow_structured_list: bool


class CaseRequirements(NamedTuple):
    required_any: tuple[str, ...]
    min_matches: int


class ConversationCase(NamedTuple):
    case_id: str
    messages: tuple[ConversationMessage, ...]
    limits: CaseLimits
    requirements: CaseRequirements


class ConversationCorpus(NamedTuple):
    schema_version: int
    cases: tuple[ConversationCase, ...]


class CaseObservation(NamedTuple):
    case_id: str
    response: str
    segments: tuple[str, ...]
    latency_ms: float


class ShadowWorkHandler:
    max_objective_chars = 1024
    can_cancel_work = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def start_work(self, *, objective: str, invocation_id: str) -> WorkStartResult:
        del objective, invocation_id
        self.calls.append("start_work")
        return WorkStartResult(accepted=False, state="rejected", reason="shadow evaluation")

    async def cancel_active_work(self, *, invocation_id: str) -> WorkCancelResult:
        del invocation_id
        self.calls.append("cancel_active_work")
        return WorkCancelResult(accepted=False, state="rejected", reason="shadow evaluation")

    async def cancel_work(self, *, task_id: str, invocation_id: str) -> WorkCancelResult:
        del task_id, invocation_id
        self.calls.append("cancel_work")
        return WorkCancelResult(accepted=False, state="rejected", reason="shadow evaluation")


class ShadowKnowledgeLookup:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def lookup(self, query: str) -> CurrentFactEvidence:
        self.calls.append(query)
        return CurrentFactEvidence(query=query, retrieved_date="1970-01-01", sources=())

    async def close(self) -> None:
        return None


def _exact_fields(value: dict[str, Any], expected: set[str], *, location: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{location} must contain exact fields {sorted(expected)}")


def _positive_int(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def load_corpus(path: Path) -> ConversationCorpus:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("corpus must be an object")
    _exact_fields(document, {"schemaVersion", "cases"}, location="corpus")
    if document["schemaVersion"] != 1:
        raise ValueError("schemaVersion must be 1")
    raw_cases = document["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty array")

    cases: list[ConversationCase] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        location = f"cases[{index}]"
        if not isinstance(raw_case, dict):
            raise ValueError(f"{location} must be an object")
        _exact_fields(
            raw_case,
            {"id", "messages", "limits", "requirements"},
            location=location,
        )
        case_id = raw_case["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"{location}.id must be a unique non-empty string")
        seen.add(case_id)

        raw_messages = raw_case["messages"]
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError(f"{location}.messages must be a non-empty array")
        messages: list[ConversationMessage] = []
        for message_index, raw_message in enumerate(raw_messages):
            message_location = f"{location}.messages[{message_index}]"
            if not isinstance(raw_message, dict):
                raise ValueError(f"{message_location} must be an object")
            _exact_fields(raw_message, {"role", "text"}, location=message_location)
            role = raw_message["role"]
            text = raw_message["text"]
            if role not in {"user", "assistant"} or not isinstance(text, str) or not text:
                raise ValueError(f"{message_location} has invalid role or text")
            messages.append(ConversationMessage(role=role, text=text))
        if messages[-1].role != "user":
            raise ValueError(f"{location}.messages must end with a user turn")

        raw_limits = raw_case["limits"]
        if not isinstance(raw_limits, dict):
            raise ValueError(f"{location}.limits must be an object")
        _exact_fields(
            raw_limits,
            {"maxWords", "maxSentences", "maxQuestions", "allowStructuredList"},
            location=f"{location}.limits",
        )
        allow_list = raw_limits["allowStructuredList"]
        if not isinstance(allow_list, bool):
            raise ValueError(f"{location}.limits.allowStructuredList must be boolean")
        raw_requirements = raw_case["requirements"]
        if not isinstance(raw_requirements, dict):
            raise ValueError(f"{location}.requirements must be an object")
        _exact_fields(
            raw_requirements,
            {"requiredAny", "minMatches"},
            location=f"{location}.requirements",
        )
        raw_terms = raw_requirements["requiredAny"]
        if (
            not isinstance(raw_terms, list)
            or not raw_terms
            or any(not isinstance(term, str) or not term.strip() for term in raw_terms)
        ):
            raise ValueError(f"{location}.requirements.requiredAny must contain strings")
        terms = tuple(dict.fromkeys(term.strip().casefold() for term in raw_terms))
        min_matches = _positive_int(
            raw_requirements["minMatches"],
            location=f"{location}.requirements.minMatches",
        )
        if min_matches < 1 or min_matches > len(terms):
            raise ValueError(f"{location}.requirements.minMatches is out of bounds")

        cases.append(
            ConversationCase(
                case_id=case_id,
                messages=tuple(messages),
                limits=CaseLimits(
                    max_words=_positive_int(raw_limits["maxWords"], location="maxWords"),
                    max_sentences=_positive_int(
                        raw_limits["maxSentences"], location="maxSentences"
                    ),
                    max_questions=_positive_int(
                        raw_limits["maxQuestions"], location="maxQuestions"
                    ),
                    allow_structured_list=allow_list,
                ),
                requirements=CaseRequirements(required_any=terms, min_matches=min_matches),
            )
        )
    return ConversationCorpus(schema_version=1, cases=tuple(cases))


def snapshot_for_case(case: ConversationCase, *, revision: int) -> ConversationContextSnapshot:
    return ConversationContextSnapshot(
        revision=revision,
        messages=case.messages,
        active_tasks=(),
    )


def response_metrics(text: str) -> dict[str, Any]:
    stripped = text.strip()
    sentence_text = _LIST_RE.sub("", stripped)
    sentences = [item for item in _SENTENCE_RE.split(sentence_text) if item]
    boilerplate = [
        label for label, pattern in _BOILERPLATE_PATTERNS if pattern.search(stripped)
    ]
    return {
        "characters": len(stripped),
        "words": len(_WORD_RE.findall(stripped)),
        "sentences": len(sentences) if stripped else 0,
        "questions": stripped.count("?"),
        "listMarkers": len(_LIST_RE.findall(stripped)),
        "markdownMarkers": len(_MARKDOWN_RE.findall(stripped)),
        "boilerplate": boilerplate,
    }


def _requirement_matches(text: str, requirements: CaseRequirements) -> list[str]:
    matches: list[str] = []
    for term in requirements.required_any:
        suffix = r"(?:\w*)?" if term[-1].isalpha() else ""
        if re.search(rf"(?<!\w){re.escape(term)}{suffix}(?!\w)", text, re.IGNORECASE):
            matches.append(term)
    return matches


def quality_report(
    *,
    corpus: ConversationCorpus,
    observations: tuple[CaseObservation, ...],
    shadow_tool_calls: tuple[str, ...],
    shadow_knowledge_calls: tuple[str, ...],
    model: str,
    effort: str,
    label: str,
) -> dict[str, Any]:
    by_id: dict[str, CaseObservation] = {}
    for observation in observations:
        if observation.case_id in by_id:
            raise ValueError(f"duplicate observation for {observation.case_id}")
        by_id[observation.case_id] = observation
    expected = {case.case_id for case in corpus.cases}
    if set(by_id) != expected:
        raise ValueError("observations must cover every corpus case exactly")

    case_reports: list[dict[str, Any]] = []
    for case in corpus.cases:
        observation = by_id[case.case_id]
        emitted_text = "\n".join(observation.segments)
        metrics = response_metrics(emitted_text)
        requirement_matches = _requirement_matches(
            observation.response,
            case.requirements,
        )
        violations: list[str] = []
        if not observation.response.strip():
            violations.append("empty_response")
        if metrics["words"] > case.limits.max_words:
            violations.append("max_words")
        if metrics["sentences"] > case.limits.max_sentences:
            violations.append("max_sentences")
        if metrics["questions"] > case.limits.max_questions:
            violations.append("max_questions")
        if (
            not case.limits.allow_structured_list
            and (metrics["listMarkers"] or metrics["markdownMarkers"])
        ):
            violations.append("unrequested_structure")
        if metrics["boilerplate"]:
            violations.append("boilerplate")
        if len(requirement_matches) < case.requirements.min_matches:
            violations.append("relevance")
        case_reports.append(
            {
                "id": case.case_id,
                "response": observation.response,
                "segments": list(observation.segments),
                "latencyMs": observation.latency_ms,
                "metrics": metrics,
                "requirementMatches": requirement_matches,
                "violations": violations,
            }
        )

    violating_cases = sum(bool(item["violations"]) for item in case_reports)
    gate = (
        "passed"
        if not violating_cases and not shadow_tool_calls and not shadow_knowledge_calls
        else "failed"
    )
    return {
        "schemaVersion": 1,
        "label": label,
        "model": model,
        "effort": effort,
        "gate": gate,
        "summary": {
            "cases": len(case_reports),
            "violating_cases": violating_cases,
            "shadow_tool_calls": len(shadow_tool_calls),
            "shadow_knowledge_calls": len(shadow_knowledge_calls),
        },
        "shadowToolCalls": list(shadow_tool_calls),
        "shadowKnowledgeCalls": list(shadow_knowledge_calls),
        "cases": case_reports,
    }


async def evaluate(
    corpus: ConversationCorpus,
    *,
    model: str,
    effort: str,
    label: str,
) -> dict[str, Any]:
    knowledge = ShadowKnowledgeLookup()
    inference = CodexAppServerStreamingInference(
        model=model,
        effort=effort,
        current_fact_lookup=knowledge,
        request_timeout_seconds=90.0,
    )
    shadow = ShadowWorkHandler()
    inference.bind_work_tools(shadow)
    observations: list[CaseObservation] = []
    try:
        for revision, case in enumerate(corpus.cases, start=1):
            started = time.perf_counter()
            segments = tuple(
                [
                    segment
                    async for segment in inference.stream(
                        snapshot_for_case(case, revision=revision),
                        turn_id=f"turn_conversation_quality_{revision:02d}",
                    )
                ]
            )
            observations.append(
                CaseObservation(
                    case_id=case.case_id,
                    response=" ".join(segments),
                    segments=segments,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                )
            )
    finally:
        await inference.close()
    return quality_report(
        corpus=corpus,
        observations=tuple(observations),
        shadow_tool_calls=tuple(shadow.calls),
        shadow_knowledge_calls=tuple(knowledge.calls),
        model=model,
        effort=effort,
        label=label,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize bounded realtime conversation quality"
    )
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--label", default="conversation-quality")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    report = await evaluate(
        load_corpus(args.corpus),
        model=args.model,
        effort=args.effort,
        label=args.label,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["gate"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
