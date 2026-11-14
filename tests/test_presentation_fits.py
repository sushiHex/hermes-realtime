"""Every presenter output the per-request schema admits renders, or is cut and refused (#550).

The presenter backend lands exactly on a schema bound instead of overflowing it, so each
schema bound is one past the bound the host accepts: an output that reaches a bound was cut
and is refused as ``too_long`` before any other content check. Every output within the
accepted bounds renders inside the unchanged 1800-character detail bound, because the
footer budget is reserved for the longest sources the presenter may cite, and a source
whose URL the host could not display is never offered to it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sys
import threading
import time
import types
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from hermes_realtime.integration import evidence_presentation as presenter
from hermes_realtime.integration.api import HermesApiTaskSession
from hermes_realtime.integration.web_evidence_contract import (
    REFUSED_REASONS,
    attestation_outcome,
)
from hermes_realtime.result_limits import (
    MAX_PRESENTATION_DETAIL_CHARS,
    MAX_PRESENTATION_VOICE_CHARS,
)
from tests.test_native_web_evidence import (
    GENERATION,
    RUN_ID,
    URL,
    _bind_run,
    _final_response,
    _final_turn,
    _install_fake_openai_models,
    _observer,
    _Presentation,
    _web_turn,
)
from tests.test_web_evidence_refusal_names import _answer, _Formatted, _markers

HEADING = "\n\nSources:\n"


def _passages(catalog: tuple[str, ...]) -> Any:
    return MappingProxyType({url: (("search", "Supplied evidence."),) for url in catalog})


def _chat(content: str, finish_reason: str = "stop") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content, refusal=None, tool_calls=None),
        )],
        usage=None,
    )


def _render(catalog: tuple[str, ...], content: str, finish_reason: str = "stop") -> str:
    return presenter.render_presentation_response(
        _chat(content, finish_reason), catalog, "chat_completions", {},
        passages=_passages(catalog))


def _footer(urls: list[str]) -> int:
    return len(HEADING) + sum(len(url) for url in urls) + len(urls) - 1 if urls else 0


# -- limits -----------------------------------------------------------------------------


def test_limits_reserve_the_footer_of_every_eligible_source_when_it_fits() -> None:
    catalog = ("https://example.com/a", "https://example.com/bb")
    limits = presenter.presentation_limits(catalog)

    assert dict(limits) == {
        "lead": MAX_PRESENTATION_VOICE_CHARS,
        "support": MAX_PRESENTATION_DETAIL_CHARS - MAX_PRESENTATION_VOICE_CHARS - 2
        - _footer(list(catalog)),
        "sources": 2,
    }


def test_limits_cap_the_source_count_at_the_longest_footer_that_fits() -> None:
    budget = presenter.MAX_PRESENTATION_FOOTER_CHARS
    assert budget == 600
    long_urls = ["https://example.com/" + letter * 160 for letter in "abcd"]
    limits = presenter.presentation_limits(tuple(sorted(long_urls)))

    # Three 180-character URLs fit 600 with the heading and separators; four do not.
    assert _footer(long_urls[:3]) <= budget < _footer(long_urls)
    assert limits["sources"] == 3
    assert limits["support"] == (MAX_PRESENTATION_DETAIL_CHARS - MAX_PRESENTATION_VOICE_CHARS
                                 - 2 - _footer(long_urls[:3]))


def test_the_detail_bound_and_instructions_stay_one_constant() -> None:
    assert MAX_PRESENTATION_DETAIL_CHARS == 1800
    instructions = presenter.PRESENTATION_INSTRUCTIONS
    assert type(instructions) is str
    assert "1800" not in instructions
    assert "top-level limits object" in instructions


def test_limits_are_packed_into_the_bounded_input() -> None:
    catalog = ("https://example.com/report",)
    payload = json.loads(presenter.presentation_input("objective", catalog, _passages(catalog)))

    assert payload["limits"] == dict(presenter.presentation_limits(catalog))
    assert set(payload) == {"objective", "limits", "sources", "evidence", "missing_source_ids"}


def test_limits_count_against_the_input_bound() -> None:
    catalog = tuple(f"https://example.com/{index:03d}" for index in range(64))
    passages = MappingProxyType({url: (("search", "x" * 4096),) for url in catalog})
    source_input = presenter.presentation_input("objective", catalog, passages)

    assert len(source_input.encode("utf-8")) <= 16384
    assert json.loads(source_input)["limits"]["sources"] >= 1


# -- schema -----------------------------------------------------------------------------


def test_every_schema_bound_is_one_past_the_accepted_bound() -> None:
    catalog = ("https://example.com/a", "https://example.com/b", "https://example.com/c")
    source_input = presenter.presentation_input("objective", catalog, _passages(catalog))
    limits = presenter.presentation_limits(catalog)
    schema = presenter.presentation_schema(source_input)

    answer = schema["properties"]["answer"]["properties"]
    ids = schema["properties"]["source_ids"]
    assert answer["lead"]["maxLength"] == limits["lead"] + 1
    assert answer["support"]["maxLength"] == limits["support"] + 1
    assert ids["maxItems"] == limits["sources"] + 1
    assert ids["items"] == {"type": "integer", "enum": [0, 1, 2]}
    # A per-request copy; the shared template is not mutated.
    assert presenter.PRESENTATION_SCHEMA["properties"]["source_ids"]["maxItems"] == 256


def test_schema_admits_the_accepted_maximum_and_the_cut_but_nothing_past_it() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    catalog = ("https://example.com/a", "https://example.com/b")
    limits = presenter.presentation_limits(catalog)
    schema = presenter.presentation_schema(
        presenter.presentation_input("objective", catalog, _passages(catalog)))

    def valid(lead: int, support: int, ids: list[int]) -> bool:
        value = {"answer": {"lead": "x" * lead, "support": "y" * support}, "source_ids": ids}
        return jsonschema.Draft202012Validator(schema).is_valid(value)

    assert valid(limits["lead"], limits["support"], [0, 1])
    assert valid(limits["lead"] + 1, limits["support"] + 1, [0, 1, 0])
    assert not valid(limits["lead"] + 2, 0, [])
    assert not valid(1, limits["support"] + 2, [])
    assert not valid(1, 0, [2])


@pytest.mark.parametrize("source_input", [
    "", "[]", json.dumps({"limits": {"lead": 220, "support": 1}, "sources": []}),
    json.dumps({"limits": {"lead": 220, "support": 1, "sources": True},
                "sources": [{"id": 0, "url": URL}]}),
    json.dumps({"limits": {"lead": 220, "support": 1, "sources": 1}, "sources": []}),
], ids=["empty", "list", "missing-key", "bool-count", "no-sources"])
def test_schema_refuses_malformed_limits(source_input: str) -> None:
    with pytest.raises(ValueError, match="^presentation-input-limits$"):
        presenter.presentation_schema(source_input)


# -- cut detection ----------------------------------------------------------------------


CATALOG = ("https://example.com/a", "https://example.com/b")


def test_accepted_maximum_renders() -> None:
    limits = presenter.presentation_limits(CATALOG)
    content = _render(CATALOG, _answer("x" * limits["lead"], "y" * limits["support"], [0, 1]))

    assert HermesApiTaskSession._completed_result_text(content) == content


@pytest.mark.parametrize("field", ["lead", "support", "sources"])
def test_a_field_at_its_schema_bound_is_cut_and_refused_before_other_checks(field: str) -> None:
    limits = presenter.presentation_limits(CATALOG)
    # Each cut also carries a fault a later check would name, so order is observable.
    lead, support, ids = "Lead.", "", [0]
    if field == "lead":
        lead = "two\nlines" + "x" * (limits["lead"] + 1 - len("two\nlines"))
    elif field == "support":
        support = "<detail>" + "y" * (limits["support"] + 1 - len("<detail>"))
        ids = [9]
    else:
        ids = [0, 0, 7][: limits["sources"] + 1]
        assert len(ids) == limits["sources"] + 1
    with pytest.raises(ValueError, match="^presentation-content-length$"):
        _render(CATALOG, _answer(lead, support, ids))


def test_a_duplicate_within_the_accepted_count_is_still_refused_as_a_duplicate() -> None:
    with pytest.raises(ValueError, match="^presentation-source-id-duplicate$"):
        _render(CATALOG, _answer("Lead.", "", [0, 0]))


def test_a_cut_list_is_refused_before_type_range_or_evidence_checks() -> None:
    catalog = ("https://example.com/a",)
    passages = MappingProxyType({catalog[0]: ()})
    response = _chat(_answer("Lead.", "", ["x", 5]))
    with pytest.raises(ValueError, match="^presentation-content-length$"):
        presenter.render_presentation_response(
            response, catalog, "chat_completions", {}, passages=passages)


def test_bounds_count_code_points_so_an_astral_character_at_the_bound_is_accepted() -> None:
    limits = presenter.presentation_limits(CATALOG)
    at_bound = "x" * (limits["lead"] - 1) + "\U0001F600"
    # The presenter emits it escaped as a surrogate pair; JSON decoding joins it.
    raw = json.dumps({"answer": {"lead": at_bound, "support": ""}, "source_ids": [0]})
    assert "\\ud83d\\ude00" in raw
    assert _render(CATALOG, raw).startswith("<voice>" + at_bound + "</voice>")

    cut = json.dumps({"answer": {"lead": at_bound + "y", "support": ""}, "source_ids": [0]})
    with pytest.raises(ValueError, match="^presentation-content-length$"):
        _render(CATALOG, cut)


@pytest.mark.parametrize("field", ["lead", "support"])
def test_a_lone_surrogate_is_a_cut(field: str) -> None:
    value = {"answer": {"lead": "Lead.", "support": ""}, "source_ids": [0]}
    value["answer"][field] = "A split character \ud83d"
    raw = json.dumps(value)
    assert "\\ud83d" in raw
    with pytest.raises(ValueError, match="^presentation-content-length$"):
        _render(CATALOG, raw)


def test_chat_length_finish_is_a_cut_even_when_the_json_is_incomplete() -> None:
    with pytest.raises(ValueError, match="^presentation-content-length$"):
        _render(CATALOG, '{"answer": {"lead": "Cut', finish_reason="length")


# -- end to end through the observer ----------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_a_cut_reaches_the_attestation_as_too_long(
    finish_reason: str, monkeypatch: Any, capsys: Any,
) -> None:
    _install_fake_openai_models(monkeypatch)
    content = _answer("x" * (MAX_PRESENTATION_VOICE_CHARS + 1), "", [0])

    class Cut(_Formatted):
        async def complete(self, *_args: Any, **_kwargs: Any) -> Any:
            return _chat(self.content, finish_reason)

    observer = _observer(Cut(content))
    task = asyncio.current_task()
    assert task is not None
    _bind_run(observer, task)
    _web_turn(observer)
    returned = await asyncio.to_thread(_final_turn, observer, _final_response())
    published = returned.choices[0].message.content

    assert attestation_outcome(observer.reports[RUN_ID], GENERATION, RUN_ID, published) == (
        "refused", "too_long")
    [marker] = _markers(capsys)
    stage = "content" if finish_reason == "stop" else "response-envelope"
    assert (marker["reason"], marker["cause"], marker["stage"]) == (
        "too_long", "presentation-content-length", stage)


def test_too_long_is_a_refusal_reason() -> None:
    assert "too_long" in REFUSED_REASONS


INELIGIBLE = (
    "https://example.com/run_0123456789abcdef",
    "https://example.com/deleg_handle",
    "https://example.com/\x07bell",
    "https://example.com/</detail>",
    "https://example.com/" + "a" * 600,
)


def _search(urls: tuple[str, ...]) -> str:
    return json.dumps({"success": True, "data": {"web": [
        {"url": url, "title": "Result", "description": "A supported result."} for url in urls]}})


@pytest.mark.asyncio
async def test_an_undisplayable_source_never_reaches_the_presenter(monkeypatch: Any) -> None:
    _install_fake_openai_models(monkeypatch)
    presentation = _Presentation()
    observer = _observer(presentation)
    task = asyncio.current_task()
    assert task is not None
    _bind_run(observer, task)
    _web_turn(observer, _search((URL, *INELIGIBLE)))
    returned = await asyncio.to_thread(_final_turn, observer, _final_response())

    assert presentation.source_input is not None
    payload = json.loads(presentation.source_input)
    assert payload["sources"] == [{"id": 0, "url": URL}]
    assert all(url not in presentation.source_input for url in INELIGIBLE[:2])
    schema = presenter.presentation_schema(presentation.source_input)
    assert schema["properties"]["source_ids"]["items"]["enum"] == [0]
    published = returned.choices[0].message.content
    assert published.endswith("\n\nSources:\n" + URL + "</detail>")
    report = json.loads(observer.reports[RUN_ID])
    assert attestation_outcome(observer.reports[RUN_ID], GENERATION, RUN_ID, published) == (
        "presented", None)
    # Every attested count and hash is on the shown basis.
    assert report["catalog_count"] == 1
    assert report["catalog_sha256"] == hashlib.sha256(
        presenter.frozen([{"id": 0, "url": URL}]).encode()).hexdigest()
    assert report["missing_count"] == 0 and report["passage_count"] == 1


@pytest.mark.asyncio
async def test_no_displayable_source_is_insufficient(monkeypatch: Any) -> None:
    _install_fake_openai_models(monkeypatch)
    presentation = _Presentation()
    observer = _observer(presentation)
    task = asyncio.current_task()
    assert task is not None
    _bind_run(observer, task)
    _web_turn(observer, _search(INELIGIBLE))
    returned = await asyncio.to_thread(_final_turn, observer, _final_response())

    published = returned.choices[0].message.content
    assert attestation_outcome(observer.reports[RUN_ID], GENERATION, RUN_ID, published) == (
        "refused", "insufficient")
    assert presentation.calls == 0


@pytest.mark.parametrize("url", INELIGIBLE)
def test_ineligible_sources_are_the_host_predicate_or_the_footer_budget(url: str) -> None:
    from hermes_realtime.integration.api import displayable_detail_text

    catalog = tuple(sorted({URL, url}))
    shown, shown_passages = presenter.displayable_sources(catalog, _passages(catalog))
    assert shown == (URL,) and set(shown_passages) == {URL}
    assert (not displayable_detail_text(url)) or len(HEADING) + len(url) > 600


_BREAKS = ("\t", "\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85",
           "\xa0", " ", " ", " ", "　")


@pytest.mark.parametrize("character", _BREAKS, ids=[f"U+{ord(c):04X}" for c in _BREAKS])
def test_a_url_with_whitespace_or_a_line_break_is_not_a_footer_line(character: str) -> None:
    # The footer is one URL per line; the host's detail check alone admits some of these.
    url = "https://example.com/split" + character + "url"
    assert url.split() != [url] or url.splitlines() != [url]
    catalog = tuple(sorted({URL, url}))
    shown, _ = presenter.displayable_sources(catalog, _passages(catalog))
    assert shown == (URL,)


def test_displayable_sources_keeps_the_catalog_identity_check() -> None:
    catalog = (URL,)
    with pytest.raises(ValueError, match="^presentation-evidence-catalog$"):
        presenter.displayable_sources(catalog, MappingProxyType({}))
    with pytest.raises(ValueError, match="^presentation-evidence-catalog$"):
        presenter.displayable_sources(catalog, {URL: ()})


# -- the property -----------------------------------------------------------------------


_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-./"


def test_every_admitted_selection_renders_and_passes_the_host_contract() -> None:
    rng = random.Random(550)
    checked = 0
    for _ in range(400):
        urls: set[str] = set()
        size = rng.randint(1, 24)
        while len(urls) < size:
            length = rng.choice((rng.randint(1, 40), rng.randint(40, 250),
                                 rng.randint(250, 640)))
            urls.add("https://e.example/" + "".join(rng.choice(_ALPHABET)
                                                    for _ in range(length)))
        catalog = tuple(sorted(urls))
        shown, shown_passages = presenter.displayable_sources(catalog, _passages(catalog))
        assert all(len(HEADING) + len(url) <= 600 for url in shown)
        if not shown:
            continue
        limits = presenter.presentation_limits(shown)
        count = limits["sources"]
        assert 1 <= count <= len(shown)
        longest = sorted(range(len(shown)), key=lambda index: -len(shown[index]))
        selections = [longest[:count], rng.sample(range(len(shown)), rng.randint(0, count)), []]
        for ids in selections:
            for support in ("y" * limits["support"], ""):
                content = presenter.render_presentation_response(
                    _chat(_answer("x" * limits["lead"], support, ids)), shown,
                    "chat_completions", {}, passages=shown_passages)
                assert HermesApiTaskSession._completed_result_text(content) == content
                checked += 1
        if count < len(shown):
            with pytest.raises(ValueError, match="^presentation-content-length$"):
                presenter.render_presentation_response(
                    _chat(_answer("Lead.", "", longest[: count + 1])), shown,
                    "chat_completions", {}, passages=shown_passages)
    assert checked > 1000


# -- the transport sends one instruction constant and a per-request schema --------------


class _Owner:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.row: dict[str, Any] = {"deadline": time.monotonic() + 60, "cancelled": False,
                                    "presentation_active": None}

    def scope(self) -> tuple[str, dict[str, Any] | None]:
        return "run", self.row


def _install_fake_transport(monkeypatch: Any, bodies: list[dict[str, Any]]) -> None:
    class Http:
        def __init__(self, **kwargs: Any) -> None:
            self.hooks = kwargs["event_hooks"]["request"]

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

    class Stream:
        def __init__(self, terminal: Any) -> None:
            self.events = [SimpleNamespace(type="response.completed", response=terminal)]

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            if not self.events:
                raise StopAsyncIteration
            return self.events.pop(0)

        async def close(self) -> None:
            return None

    class Client:
        def __init__(self, *, http_client: Any, **_kwargs: Any) -> None:
            self.http = http_client
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
            self.responses = SimpleNamespace(create=self.create_stream)

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def _send(self, body: dict[str, Any]) -> None:
            bodies.append(body)
            raw = json.dumps(body).encode()
            for hook in self.http.hooks:
                await hook(SimpleNamespace(content=raw))

        async def create(self, **body: Any) -> Any:
            await self._send(body)
            return "formatted"

        async def create_stream(self, **body: Any) -> Any:
            await self._send(body)
            return Stream(SimpleNamespace(status="completed"))

    httpx = types.ModuleType("httpx")
    httpx.AsyncClient = Http  # type: ignore[attr-defined]
    openai = types.ModuleType("openai")
    openai.AsyncOpenAI = Client  # type: ignore[attr-defined]
    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    headers = types.ModuleType("agent.codex_headers")
    headers.codex_cloudflare_headers = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", httpx)
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.codex_headers", headers)


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses"])
async def test_instructions_are_identical_and_limits_ride_only_in_the_input(
    api_mode: str, monkeypatch: Any,
) -> None:
    from hermes_realtime.integration.presentation_transport import Presentation

    bodies: list[dict[str, Any]] = []
    _install_fake_transport(monkeypatch, bodies)
    monkeypatch.setenv("PRESENTATION_FITS_TEST_KEY", "synthetic")
    presentation = Presentation({
        "provider": "test", "base_url": "https://model.example.invalid/v1", "model": "test",
        "api_key_env": "PRESENTATION_FITS_TEST_KEY", "api_mode": api_mode,
    })
    catalogs = [("https://example.com/a",),
                tuple("https://example.com/" + letter * 300 for letter in "abc")]
    inputs = [presenter.presentation_input("objective", catalog, _passages(catalog))
              for catalog in catalogs]
    wires = []
    for source_input in inputs:
        evidence: dict[str, Any] = {"wire_requests": 0}
        owner = _Owner()
        await presentation.complete(owner, "run", owner.row, source_input, evidence)
        wires.append(evidence["wire"])

    schemas = []
    for body, source_input in zip(bodies, inputs, strict=True):
        if api_mode == "chat_completions":
            instructions = body["messages"][0]["content"]
            sent_input = body["messages"][-1]["content"]
            schema = body["response_format"]["json_schema"]["schema"]
            assert body["max_tokens"] == 4096
        else:
            instructions = body["instructions"]
            sent_input = body["input"][0]["content"]
            schema = body["text"]["format"]["schema"]
        assert instructions == presenter.PRESENTATION_INSTRUCTIONS
        assert sent_input == source_input and "limits" in json.loads(sent_input)
        assert schema == presenter.presentation_schema(source_input)
        schemas.append(schema)
    assert schemas[0] != schemas[1]
    assert wires[0]["instruction_sha256"] == wires[1]["instruction_sha256"]
    assert all(wire["schema_matches"] for wire in wires)


# -- Codex reports its own truncation ---------------------------------------------------


def _incomplete(reason: str | None) -> Any:
    details = None if reason is None else SimpleNamespace(reason=reason)
    return SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
        status="incomplete", incomplete_details=details))


async def _events(*events: Any) -> Any:
    for event in events:
        yield event


@pytest.mark.asyncio
@pytest.mark.parametrize(("reason", "category"), [
    ("max_output_tokens", "presentation-content-length"),
    ("content_filter", "responses-terminal-failure"),
    (None, "responses-terminal-failure"),
])
async def test_codex_incomplete_is_too_long_only_for_max_output_tokens(
    reason: str | None, category: str,
) -> None:
    from hermes_realtime.integration.presentation_transport import _consume_codex_stream

    with pytest.raises(ValueError, match="^" + category + "$"):
        await _consume_codex_stream(_events(_incomplete(reason)))


def test_cut_categories_map_to_too_long() -> None:
    from hermes_realtime.integration.native_web_evidence import _reason

    for category in ("presentation-content-length", "responses-stream-text-bound"):
        assert _reason(ValueError(category), "internal") == "too_long"
    assert _reason(ValueError("responses-terminal-failure"), "presentation_failed") == (
        "presentation_failed")
