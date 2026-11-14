from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
import time
import types
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hermes_realtime.integration.native_web_evidence import (
    NativeWebEvidence,
    _request_binding_for,
    _RequestBinding,
)
from hermes_realtime.integration.web_evidence_contract import (
    CONTRACT,
    CONTRACT_HEADER,
    GENERATION_HEADER,
    REFUSED_REASONS,
    attestation_outcome,
    attestation_valid,
)

RUN_ID = "run_0123456789abcdef0123456789abcdef"
GENERATION = "0123456789abcdef0123456789abcdef"
URL = "https://example.com/report"


def _search_result() -> str:
    return json.dumps(
        {
            "success": True,
            "data": {
                "web": [{"url": URL, "title": "Report", "description": "A supported result."}]
            },
        }
    )


def _extract_result(url: str) -> str:
    return json.dumps({"results": [{"url": url, "content": "Retrieved page text."}]})


def _install_fake_openai_models(monkeypatch: Any) -> None:
    class ChatCompletion:
        @classmethod
        def model_validate(cls, value: dict[str, Any]) -> Any:
            response = cls()
            response.__dict__.update(value)
            response.choices = [
                SimpleNamespace(
                    finish_reason=choice["finish_reason"],
                    index=choice["index"],
                    message=SimpleNamespace(**choice["message"]),
                )
                for choice in value["choices"]
            ]
            return response

    openai = types.ModuleType("openai")
    openai.__path__ = []  # type: ignore[attr-defined]
    openai_types = types.ModuleType("openai.types")
    openai_types.__path__ = []  # type: ignore[attr-defined]
    chat_types = types.ModuleType("openai.types.chat")
    chat_types.ChatCompletion = ChatCompletion
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "openai.types", openai_types)
    monkeypatch.setitem(sys.modules, "openai.types.chat", chat_types)


def _final_response(text: str = "Native draft.") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=text, refusal=None, tool_calls=None),
            )
        ]
    )


class _Presentation:
    config = {
        "provider": "test",
        "base_url": "https://example.com",
        "model": "test",
        "api_mode": "chat_completions",
    }

    def __init__(self) -> None:
        self.calls = 0
        self.source_input: str | None = None
        self.started: threading.Event | None = None
        self.release: threading.Event | None = None

    def validate_route(self, _kw: dict[str, Any]) -> None:
        return None

    async def complete(
        self,
        owner: NativeWebEvidence,
        run_id: str,
        row: dict[str, Any],
        source_input: str,
        evidence: dict[str, Any],
    ) -> Any:
        self.calls += 1
        self.source_input = source_input
        assert owner.scope() == (run_id, row)
        assert json.loads(source_input)["objective"] == "Summarize the report."
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await asyncio.to_thread(self.release.wait)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "answer": {
                                    "lead": "The report found a supported result.",
                                    "support": "The retrieved report supports this result.",
                                },
                                "source_ids": [0],
                            }
                        ),
                        refusal=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        )


def _observer(presentation: _Presentation | None = None) -> NativeWebEvidence:
    observer = NativeWebEvidence(presentation or _Presentation(), generation=GENERATION)  # type: ignore[arg-type]
    return observer


def _bind_run(observer: NativeWebEvidence, task: asyncio.Task[Any]) -> None:
    observer.adapter = SimpleNamespace(_active_run_tasks={RUN_ID: task})
    observer._get_session_key = lambda: RUN_ID
    binding = _RequestBinding(
        CONTRACT,
        GENERATION,
        "Summarize the report.",
        deadline=time.monotonic() + 120,
    )
    assert observer._accepted_row(binding, RUN_ID, False) is not None


@pytest.mark.asyncio
async def test_first_tool_effect_binds_exact_run_task_before_native_call() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    calls = 0
    raw = _search_result()

    def native_effect(args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        row = observer.rows[RUN_ID]
        assert row["native_task"] is task
        assert observer.adapter._active_run_tasks[RUN_ID] is task
        assert args == {"query": "current report"}
        return raw

    returned = observer.tool_execution(
        tool_name="web_search",
        args={"query": "current report"},
        next_call=native_effect,
    )

    assert returned is raw
    assert calls == 1
    row = observer.rows[RUN_ID]
    assert row["web_seen"] is True
    assert row["catalog_count"] == 1
    assert row["passage_count"] == 1


@pytest.mark.asyncio
async def test_cancelled_run_refuses_observation_but_keeps_native_effect_once() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    observer.rows[RUN_ID]["cancelled"] = True
    calls = 0
    raw = _search_result()

    def native_effect(_args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return raw

    returned = observer.tool_execution(
        tool_name="web_search",
        args={"query": "current report"},
        next_call=native_effect,
    )

    assert returned is raw
    assert calls == 1
    assert observer.rows[RUN_ID]["disposition"] == "refused"
    assert observer.rows[RUN_ID]["passage_count"] == 0


@pytest.mark.asyncio
async def test_direct_native_extract_can_supply_requested_url_evidence() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    raw = _extract_result(URL)
    calls = 0

    def native_effect(args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        assert args == {"urls": [URL]}
        return raw

    returned = observer.tool_execution(
        tool_name="web_extract",
        args={"urls": [URL]},
        next_call=native_effect,
    )

    assert returned is raw
    assert calls == 1
    row = observer.rows[RUN_ID]
    assert row["urls"] == {URL}
    assert row["catalog_count"] == 1
    assert row["passage_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_native_mixed_extract_preserves_bound_source_without_false_gap(reverse: bool) -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    missing = "https://example.com/missing"
    items = [{"url": URL, "content": "Bound source text"},
             {"url": "https://example.com/foreign", "content": "UNBOUND_TEXT"}]
    raw = json.dumps({"results": list(reversed(items)) if reverse else items})
    calls = 0

    def native_effect(args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        assert args == {"urls": [URL, missing]}
        return raw

    returned = observer.tool_execution(
        tool_name="web_extract", args={"urls": [URL, missing]}, next_call=native_effect,
    )

    assert returned is raw and calls == 1
    row = observer.rows[RUN_ID]
    assert row["passages"][URL] == (("extract", "Bound source text"),)
    assert row["passages"][missing] == (("extract-unavailable", ""),)
    assert row["catalog_count"] == row["passage_count"] == 2
    assert "UNBOUND_TEXT" not in repr(dict(row["passages"]))


@pytest.mark.asyncio
async def test_native_extract_does_not_capture_unrequested_result_url() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    raw = _extract_result("https://example.com/other")
    calls = 0

    def native_effect(_args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return raw

    returned = observer.tool_execution(
        tool_name="web_extract",
        args={"urls": [URL]},
        next_call=native_effect,
    )

    assert returned is raw
    assert calls == 1
    row = observer.rows[RUN_ID]
    assert row["urls"] == {URL}
    assert set(row["passages"]) == {URL}
    assert row["passages"][URL] == (("extract-unavailable", ""),)


def test_selected_generation_mismatch_refuses_before_native_dispatch() -> None:
    with pytest.raises(ValueError, match="web-evidence-generation"):
        _request_binding_for(
            CONTRACT,
            "f" * 32,
            {"input": "Summarize the report."},
            expected_generation=GENERATION,
            admitted_at=1.0,
        )


def test_selected_objective_must_be_bound_before_native_dispatch() -> None:
    with pytest.raises(ValueError, match="web-evidence-objective"):
        _request_binding_for(
            CONTRACT,
            GENERATION,
            {"input": ""},
            expected_generation=GENERATION,
            admitted_at=1.0,
        )


@pytest.mark.asyncio
async def test_first_start_row_is_bound_before_native_task_can_execute() -> None:
    observer = _observer()
    observer.adapter = SimpleNamespace(_active_run_tasks={})
    observer._get_session_key = lambda: RUN_ID
    binding = _RequestBinding(
        CONTRACT,
        GENERATION,
        "Summarize the report.",
        deadline=time.monotonic() + 120,
    )
    raw = _search_result()
    calls = 0

    def native_tool_thread() -> str:
        nonlocal calls
        calls += 1
        return observer.tool_execution(
            tool_name="web_search",
            args={"query": "current report"},
            next_call=lambda _args: raw,
        )

    task = asyncio.create_task(asyncio.to_thread(native_tool_thread))
    observer.adapter._active_run_tasks[RUN_ID] = task
    row = observer._accepted_row(binding, RUN_ID, False)
    assert row is not None and row["native_task"] is task
    assert await task is raw
    assert calls == 1
    assert observer.rows[RUN_ID]["passage_count"] == 1


@pytest.mark.asyncio
async def test_wrong_native_task_binding_refuses_observation_but_keeps_effect_once() -> None:
    admitted_task = asyncio.current_task()
    assert admitted_task is not None
    observer = _observer()
    _bind_run(observer, admitted_task)
    foreign_task = asyncio.create_task(asyncio.sleep(60))
    observer.adapter._active_run_tasks[RUN_ID] = foreign_task
    observed = 0
    native_calls = 0
    raw = _search_result()

    def observe(*_args: Any) -> None:
        nonlocal observed
        observed += 1

    observer._observe_result = observe  # type: ignore[method-assign]

    def native_effect(_args: dict[str, Any]) -> str:
        nonlocal native_calls
        native_calls += 1
        return raw

    try:
        assert (
            observer.tool_execution(
                tool_name="web_search",
                args={"query": "current report"},
                next_call=native_effect,
            )
            is raw
        )
        assert native_calls == 1
        assert observed == 0
        assert observer.rows[RUN_ID]["disposition"] == "refused"
    finally:
        foreign_task.cancel()
        await asyncio.gather(foreign_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_native_effect_is_called_once_and_result_is_unchanged() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    calls = 0
    raw = _search_result()

    async def native_effect(_args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return raw

    returned = observer.tool_execution(
        tool_name="web_search",
        args={"query": "current report"},
        next_call=native_effect,
    )
    assert await returned is raw
    assert calls == 1
    assert observer.rows[RUN_ID]["passage_count"] == 1


@pytest.mark.asyncio
async def test_malformed_observation_preserves_native_result_and_refuses_grounding(
    capsys: Any,
) -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    raw = "private malformed body"
    calls = 0

    def native_effect(_args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return raw

    returned = observer.tool_execution(
        tool_name="web_search",
        args={"query": "current report"},
        next_call=native_effect,
    )
    assert returned is raw
    assert calls == 1
    assert observer.rows[RUN_ID]["disposition"] == "refused"
    assert capsys.readouterr().out == (
        '[web-evidence] {"category":"refused","cause":"native-result-json","count":1,'
        '"error_class":"ValueError","phase":"capture","reason":"capture",'
        '"rendered_detail_length":null,"selected_source_count":null,"stage":"none"}\n'
    )


@pytest.mark.asyncio
async def test_known_search_failure_preserves_prior_evidence_for_presentation() -> None:
    task = asyncio.current_task()
    assert task is not None
    presentation = _Presentation()
    observer = _observer(presentation)
    _bind_run(observer, task)
    good = _search_result()
    failed = json.dumps(
        {
            "success": False,
            "error": "temporary backend failure",
            "data": {
                "web": [
                    {
                        "url": "https://example.com/untrusted",
                        "description": "Must not be evidence.",
                    }
                ]
            },
        }
    )
    calls = 0

    def native_search(_args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return good if calls == 1 else failed

    assert (
        observer.tool_execution(
            tool_name="web_search", args={"query": "report"}, next_call=native_search
        )
        == good
    )
    assert (
        observer.tool_execution(
            tool_name="web_search", args={"query": "report"}, next_call=native_search
        )
        == failed
    )
    assert observer.rows[RUN_ID]["invalid"] is False
    native = _final_response()
    returned = await asyncio.to_thread(
        observer.llm_execution,
        request={"model": "test"},
        provider="test",
        base_url="https://example.com",
        model="test",
        api_mode="chat_completions",
        next_call=lambda _request: native,
    )

    assert returned is not native
    assert observer.rows[RUN_ID]["disposition"] == "presented"
    assert presentation.source_input is not None
    assert "A supported result." in presentation.source_input
    assert "Must not be evidence." not in presentation.source_input
    assert "https://example.com/untrusted" not in presentation.source_input


@pytest.mark.asyncio
async def test_failed_extract_seeds_requested_missing_catalog_without_error_data() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    failed = json.dumps(
        {
            "success": False,
            "error": "temporary backend failure",
            "results": [{"url": URL, "content": "Must not be evidence."}],
        }
    )
    calls = 0

    def native_extract(_args: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        return failed

    returned = observer.tool_execution(
        tool_name="web_extract",
        args={"urls": [URL]},
        next_call=native_extract,
    )

    assert returned == failed
    assert calls == 1
    row = observer.rows[RUN_ID]
    assert row["invalid"] is False
    assert row["urls"] == {URL}
    assert row["passages"][URL] == ()
    assert row["disposition"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"success": False},
        {"success": False, "error": 7},
        {"success": False, "error": ""},
    ],
)
@pytest.mark.asyncio
async def test_incomplete_operational_error_shape_invalidates_observation(
    payload: dict[str, Any],
) -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    raw = json.dumps(payload)
    returned = observer.tool_execution(
        tool_name="web_search",
        args={"query": "report"},
        next_call=lambda _args: raw,
    )

    assert returned == raw
    assert observer.rows[RUN_ID]["invalid"] is True
    assert observer.rows[RUN_ID]["disposition"] == "refused"


@pytest.mark.asyncio
async def test_all_known_search_failures_still_refuse_final_presentation(
    monkeypatch: Any,
) -> None:
    _install_fake_openai_models(monkeypatch)
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    failed = json.dumps({"success": False, "error": "temporary backend failure"})
    observer.tool_execution(
        tool_name="web_search",
        args={"query": "report"},
        next_call=lambda _args: failed,
    )
    native = _final_response()
    returned = await asyncio.to_thread(
        observer.llm_execution,
        request={"model": "test"},
        provider="test",
        base_url="https://example.com",
        model="test",
        api_mode="chat_completions",
        next_call=lambda _request: native,
    )

    assert returned is not native
    assert returned.choices[0].message.content == '{"error": "policy_refused"}'
    assert json.loads(observer.reports[RUN_ID])["disposition"] == "refused"


@pytest.mark.asyncio
async def test_mixed_tool_run_preserves_native_output_and_attests_ungrounded() -> None:
    task = asyncio.current_task()
    assert task is not None
    presentation = _Presentation()
    observer = _observer(presentation)
    _bind_run(observer, task)
    search = observer.tool_execution(
        tool_name="web_search",
        args={"query": "current report"},
        next_call=lambda _args: _search_result(),
    )
    assert search == _search_result()
    other = observer.tool_execution(
        tool_name="terminal",
        args={"command": "..."},
        next_call=lambda _args: "tool output",
    )
    assert other == "tool output"
    native = _final_response()
    returned = await asyncio.to_thread(
        observer.llm_execution,
        request={"model": "test"},
        provider="test",
        base_url="https://example.com",
        model="test",
        api_mode="chat_completions",
        next_call=lambda _request: native,
    )
    assert returned is native
    assert presentation.calls == 0
    assert observer.rows[RUN_ID]["disposition"] == "mixed_ungrounded"
    report = observer.reports[RUN_ID]
    assert json.loads(report)["disposition"] == "mixed_ungrounded"
    assert attestation_valid(report, GENERATION, RUN_ID, "Native draft.")


@pytest.mark.asyncio
async def test_web_result_is_formatted_and_attestation_binds_exact_output() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    observer.tool_execution(
        tool_name="web_search",
        args={"query": "current report"},
        next_call=lambda _args: _search_result(),
    )
    native = _final_response()
    published = await asyncio.to_thread(
        observer.llm_execution,
        request={"model": "test"},
        provider="test",
        base_url="https://example.com",
        model="test",
        api_mode="chat_completions",
        next_call=lambda _request: native,
    )
    assert published is not native
    assert published.choices[0].message.content.startswith("<voice>")
    assert observer.rows[RUN_ID]["disposition"] == "presented"
    report = observer.reports[RUN_ID]
    assert attestation_valid(
        report,
        GENERATION,
        RUN_ID,
        published.choices[0].message.content,
    )
    assert not attestation_valid(report, GENERATION, RUN_ID, "different terminal")


@pytest.mark.asyncio
async def test_stop_before_presentation_prevents_stale_publication(monkeypatch: Any) -> None:
    _install_fake_openai_models(monkeypatch)
    task = asyncio.current_task()
    assert task is not None
    presentation = _Presentation()
    presentation.started = threading.Event()
    presentation.release = threading.Event()
    observer = _observer(presentation)
    _bind_run(observer, task)
    observer.tool_execution(
        tool_name="web_search",
        args={"query": "current report"},
        next_call=lambda _args: _search_result(),
    )
    native = _final_response()
    pending = asyncio.create_task(
        asyncio.to_thread(
            observer.llm_execution,
            request={"model": "test"},
            provider="test",
            base_url="https://example.com",
            model="test",
            api_mode="chat_completions",
            next_call=lambda _request: native,
        )
    )

    async def stop_during_presentation() -> None:
        assert await asyncio.to_thread(presentation.started.wait, 5)
        observer.stop(RUN_ID)
        presentation.release.set()

    stopper = asyncio.create_task(stop_during_presentation())
    returned = await pending
    await stopper
    assert returned is not native
    assert json.loads(returned.choices[0].message.content) == {"error": "policy_refused"}
    assert observer.rows[RUN_ID]["disposition"] == "refused"
    assert json.loads(observer.reports[RUN_ID])["disposition"] == "refused"


@pytest.mark.asyncio
async def test_failed_presentation_returns_native_compatible_refusal(monkeypatch: Any) -> None:
    _install_fake_openai_models(monkeypatch)

    class FailedPresentation(_Presentation):
        async def complete(self, *_args: Any, **_kwargs: Any) -> Any:
            raise ValueError("fixed-test-provider-failure")

    task = asyncio.current_task()
    assert task is not None
    observer = _observer(FailedPresentation())
    _bind_run(observer, task)
    observer.tool_execution(
        tool_name="web_search",
        args={"query": "current report"},
        next_call=lambda _args: _search_result(),
    )
    native = _final_response()
    returned = await asyncio.to_thread(
        observer.llm_execution,
        request={"model": "test"},
        provider="test",
        base_url="https://example.com",
        model="test",
        api_mode="chat_completions",
        next_call=lambda _request: native,
    )
    assert returned is not native
    assert json.loads(returned.choices[0].message.content) == {"error": "policy_refused"}
    report = json.loads(observer.reports[RUN_ID])
    assert report["disposition"] == "refused"
    assert report["output_sha256"]


@pytest.mark.asyncio
async def test_malformed_native_choices_still_return_independent_refusal(monkeypatch: Any) -> None:
    _install_fake_openai_models(monkeypatch)

    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    malformed = SimpleNamespace(choices=[SimpleNamespace(), SimpleNamespace()])
    returned = await asyncio.to_thread(
        observer.llm_execution,
        request={"model": "test"},
        provider="test",
        base_url="https://example.com",
        model="test",
        api_mode="chat_completions",
        next_call=lambda _request: malformed,
    )

    assert type(returned).__name__ == "ChatCompletion"
    assert len(returned.choices) == 1
    assert returned.choices[0].finish_reason == "stop"
    assert returned.choices[0].message.content == '{"error": "policy_refused"}'


@pytest.mark.asyncio
async def test_metadata_discovery_does_not_count_as_mixed_tool_evidence() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    raw = {"name": "web_search"}
    calls = 0

    def native_effect(_args: dict[str, Any]) -> object:
        nonlocal calls
        calls += 1
        return raw

    returned = observer.tool_execution(
        tool_name="tool_search",
        args={"query": "web"},
        next_call=native_effect,
    )
    assert returned is raw
    assert calls == 1
    assert observer.rows[RUN_ID]["web_seen"] is False
    assert observer.rows[RUN_ID]["other_tool_seen"] is False


@pytest.mark.asyncio
async def test_ordinary_unselected_run_is_not_observed() -> None:
    observer = _observer()
    observer._get_session_key = lambda: RUN_ID
    calls = 0
    result = object()

    def native_effect(_args: dict[str, Any]) -> object:
        nonlocal calls
        calls += 1
        return result

    assert (
        observer.tool_execution(
            tool_name="web_search",
            args={"query": "ordinary"},
            next_call=native_effect,
        )
        is result
    )
    assert calls == 1
    assert observer.rows == {}


@pytest.mark.asyncio
async def test_expired_deadline_does_not_change_no_web_native_completion() -> None:
    task = asyncio.current_task()
    assert task is not None
    observer = _observer()
    _bind_run(observer, task)
    observer.rows[RUN_ID]["deadline"] = 0.0
    native = _final_response()
    returned = await asyncio.to_thread(
        observer.llm_execution,
        request={"model": "test"},
        provider="test",
        base_url="https://example.com",
        model="test",
        api_mode="chat_completions",
        next_call=lambda _request: native,
    )
    assert returned is native
    assert json.loads(observer.reports[RUN_ID])["disposition"] == "no_web"


@pytest.mark.asyncio
async def test_stale_generation_is_rejected_before_native_run_handler() -> None:
    observer = _observer()
    observer._get_session_key = lambda: RUN_ID
    adapter = SimpleNamespace(
        _active_run_tasks={},
        _check_auth=lambda _request: None,
    )
    app = web.Application()
    calls = 0

    async def native_runs(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.json_response({"run_id": RUN_ID, "status": "started"}, status=202)

    app.router.add_post("/v1/runs", native_runs)
    observer.install(app, adapter)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/runs",
            json={"input": "Summarize the report."},
            headers={"X-Web-Evidence-Contract": CONTRACT, "X-Web-Evidence-Generation": "f" * 32},
        )
        assert response.status == 409
        assert calls == 0
        assert await response.json() == {"error": "web-evidence-generation"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_closed_observer_rejects_selected_post_before_native_handler() -> None:
    observer = _observer()
    observer._get_session_key = lambda: RUN_ID
    adapter = SimpleNamespace(
        _active_run_tasks={},
        _check_auth=lambda _request: None,
    )
    app = web.Application()
    calls = 0

    async def native_runs(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.json_response({"run_id": RUN_ID, "status": "started"}, status=202)

    app.router.add_post("/v1/runs", native_runs)
    observer.install(app, adapter)
    observer.close()
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/runs",
            json={"input": "Summarize the report."},
            headers={CONTRACT_HEADER: CONTRACT, GENERATION_HEADER: GENERATION},
        )
        assert response.status == 503
        assert calls == 0
        assert await response.json() == {"error": "web_evidence_closed"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_custom_task_factory_rejects_selected_post_before_native_handler() -> None:
    observer = _observer()
    observer._get_session_key = lambda: RUN_ID
    adapter = SimpleNamespace(
        _active_run_tasks={},
        _check_auth=lambda _request: None,
    )
    app = web.Application()
    calls = 0

    async def native_runs(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.json_response({"run_id": RUN_ID, "status": "started"}, status=202)

    app.router.add_post("/v1/runs", native_runs)
    observer.install(app, adapter)
    client = TestClient(TestServer(app))
    await client.start_server()
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()

    def task_factory(
        current_loop: asyncio.AbstractEventLoop,
        coro: Any,
        context: Any = None,
    ) -> asyncio.Task[Any]:
        return asyncio.tasks.Task(coro, loop=current_loop, context=context)

    try:
        loop.set_task_factory(task_factory)
        response = await client.post(
            "/v1/runs",
            json={"input": "Summarize the report."},
            headers={CONTRACT_HEADER: CONTRACT, GENERATION_HEADER: GENERATION},
        )
        assert response.status == 503
        assert calls == 0
        assert await response.json() == {"error": "web_evidence_task_factory"}
    finally:
        loop.set_task_factory(previous_factory)
        await client.close()


@pytest.mark.asyncio
async def test_terminal_get_uses_cached_attestation_after_row_retirement() -> None:
    finish = asyncio.Event()

    async def finished() -> None:
        await finish.wait()

    task = asyncio.create_task(finished())
    await asyncio.sleep(0)
    observer = _observer()
    observer._get_session_key = lambda: RUN_ID
    observer.adapter = SimpleNamespace(_active_run_tasks={RUN_ID: task})
    binding = _RequestBinding(
        CONTRACT,
        GENERATION,
        "Summarize the report.",
        deadline=time.monotonic() + 120,
    )
    row = observer._accepted_row(binding, RUN_ID, False)
    assert row is not None
    row["closed"] = True
    row["disposition"] = "no_web"
    row["output_sha256"] = hashlib.sha256(b"Native draft.").hexdigest()
    cached = observer._store_report_locked(RUN_ID, row)
    finish.set()
    await task
    with observer.lock:
        observer._retire_locked()
    assert RUN_ID not in observer.rows

    adapter = SimpleNamespace(_check_auth=lambda _request: None)
    app = web.Application()

    async def native_status(_request: web.Request) -> web.Response:
        return web.json_response({"status": "completed"})

    app.router.add_get(f"/v1/runs/{RUN_ID}", native_status)
    observer.install(app, adapter)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get(f"/v1/runs/{RUN_ID}")
        assert response.status == 200
        assert response.headers["X-Web-Evidence-Attestation"] == cached
    finally:
        await client.close()


def _tool_call_response() -> Any:
    call = SimpleNamespace(id="call_1", type="function",
                           function=SimpleNamespace(name="web_search", arguments="{}"))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(content=None, refusal=None, tool_calls=[call]),
            )
        ]
    )


async def _terminal_status(observer: NativeWebEvidence, body: dict[str, Any]) -> Any:
    adapter = SimpleNamespace(_check_auth=lambda _request: None)
    app = web.Application()

    async def native_status(_request: web.Request) -> web.Response:
        return web.json_response(body)

    app.router.add_get(f"/v1/runs/{RUN_ID}", native_status)
    observer.install(app, adapter)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get(f"/v1/runs/{RUN_ID}")
        assert response.status == 200
        return response.headers.get("X-Web-Evidence-Attestation")
    finally:
        await client.close()


_SHAPE_TOOLS = {
    "web_only": ("web_search",),
    "no_web": ("terminal",),
    "mixed": ("web_search", "terminal"),
}


async def _budget_spent_run(observer: NativeWebEvidence, shape: str = "mixed") -> None:
    """Live shape: every observed model turn calls a tool; the final answer is unobserved."""

    async def native_run() -> None:
        def turns() -> None:
            for tool in _SHAPE_TOOLS[shape]:
                observer.tool_execution(
                    tool_name=tool,
                    args={"query": "current report"},
                    next_call=lambda _args, tool=tool: (
                        _search_result() if tool == "web_search" else "tool output"),
                )
            tool_turn = _tool_call_response()
            assert observer.llm_execution(
                request={"model": "test"}, provider="test", base_url="https://example.com",
                model="test", api_mode="chat_completions", next_call=lambda _request: tool_turn,
            ) is tool_turn
            # Hermes's turn-budget summary call does not pass through LLM middleware.

        await asyncio.to_thread(turns)

    task = asyncio.create_task(native_run())
    observer.adapter = SimpleNamespace(_active_run_tasks={RUN_ID: task})
    observer._get_session_key = lambda: RUN_ID
    binding = _RequestBinding(
        CONTRACT, GENERATION, "Summarize the report.", deadline=time.monotonic() + 120,
    )
    assert observer._accepted_row(binding, RUN_ID, False) is not None
    await task


UNOBSERVED_FINAL = "Here is what I found so far about the report."


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [False, True])
async def test_unobserved_web_only_final_is_refused_not_delivered(
    retired: bool, capsys: Any
) -> None:
    """A web-only run reaches the user only through the passage-backed presenter."""
    presentation = _Presentation()
    observer = _observer(presentation)
    await _budget_spent_run(observer, "web_only")
    if retired:
        with observer.lock:
            observer._retire_locked()
        assert RUN_ID not in observer.rows

    raw = await _terminal_status(
        observer, {"object": "hermes.run", "run_id": RUN_ID, "status": "completed",
                   "output": UNOBSERVED_FINAL},
    )

    assert presentation.calls == 0
    assert not attestation_valid(raw, GENERATION, RUN_ID, UNOBSERVED_FINAL)
    assert attestation_outcome(raw, GENERATION, RUN_ID, UNOBSERVED_FINAL) == (
        "refused", "unobserved")
    assert capsys.readouterr().out == (
        '[web-evidence] {"category":"refused","cause":"none","count":1,"error_class":"none",'
        '"phase":"status","reason":"unobserved","rendered_detail_length":null,'
        '"selected_source_count":null,"stage":"none"}\n'
    )


@pytest.mark.asyncio
async def test_retired_refused_row_still_reports_its_reason_at_the_terminal_read() -> None:
    observer = _observer()

    async def native_run() -> None:
        def turns() -> None:
            observer.tool_execution(
                tool_name="web_search", args={"query": "current report"},
                next_call=lambda _args: "private malformed body",
            )

        await asyncio.to_thread(turns)

    task = asyncio.create_task(native_run())
    observer.adapter = SimpleNamespace(_active_run_tasks={RUN_ID: task})
    observer._get_session_key = lambda: RUN_ID
    binding = _RequestBinding(
        CONTRACT, GENERATION, "Summarize the report.", deadline=time.monotonic() + 120,
    )
    assert observer._accepted_row(binding, RUN_ID, False) is not None
    await task
    assert observer.rows[RUN_ID]["disposition"] == "refused"
    # Another run's admission or tool effect retires the finished row before the read.
    with observer.lock:
        observer._retire_locked()
    assert RUN_ID not in observer.rows

    raw = await _terminal_status(
        observer, {"object": "hermes.run", "run_id": RUN_ID, "status": "completed",
                   "output": UNOBSERVED_FINAL},
    )

    assert attestation_outcome(raw, GENERATION, RUN_ID, UNOBSERVED_FINAL) == (
        "refused", "capture")


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [False, True])
@pytest.mark.parametrize("shape", ["no_web", "mixed"])
async def test_unobserved_terminal_answer_is_delivered_ungrounded_and_bound(
    shape: str, retired: bool, capsys: Any
) -> None:
    presentation = _Presentation()
    observer = _observer(presentation)
    await _budget_spent_run(observer, shape)
    if retired:
        with observer.lock:
            observer._retire_locked()
        assert RUN_ID not in observer.rows

    raw = await _terminal_status(
        observer, {"object": "hermes.run", "run_id": RUN_ID, "status": "completed",
                   "output": UNOBSERVED_FINAL},
    )

    assert presentation.calls == 0
    assert attestation_valid(raw, GENERATION, RUN_ID, UNOBSERVED_FINAL)
    report = json.loads(raw)
    assert report["disposition"] == "ungrounded"
    assert report["reason"] == "unobserved"
    assert report["catalog_count"] == report["passage_count"] == 0
    assert not attestation_valid(raw, GENERATION, RUN_ID, UNOBSERVED_FINAL + " ")
    assert capsys.readouterr().out == (
        '[web-evidence] {"category":"ungrounded","cause":"none","count":1,"error_class":"none",'
        '"phase":"status","reason":"unobserved","rendered_detail_length":null,'
        '"selected_source_count":null,"stage":"none"}\n'
    )


def _final_turn(observer: NativeWebEvidence, response: Any) -> Any:
    return observer.llm_execution(
        request={"model": "test"}, provider="test", base_url="https://example.com",
        model="test", api_mode="chat_completions", next_call=lambda _request: response,
    )


def _web_turn(observer: NativeWebEvidence, raw: str | None = None) -> None:
    observer.tool_execution(
        tool_name="web_search", args={"query": "current report"},
        next_call=lambda _args: _search_result() if raw is None else raw,
    )


class _RefusalScenario:
    """Drive one real refusal path; ``status`` is the completed terminal output."""

    def __init__(self, monkeypatch: Any) -> None:
        self.monkeypatch = monkeypatch
        self.presentation = _Presentation()
        self.observer = _observer(self.presentation)
        self.foreign: asyncio.Task[Any] | None = None

    async def run(self, reason: str) -> str | None:
        observer = self.observer
        task = asyncio.current_task()
        assert task is not None
        _bind_run(observer, task)
        row = observer.rows[RUN_ID]
        if reason == "capture":
            _web_turn(observer, "private malformed body")
        elif reason == "binding":
            self.foreign = asyncio.create_task(asyncio.sleep(60))
            observer.adapter._active_run_tasks[RUN_ID] = self.foreign
            _web_turn(observer)
            observer.adapter._active_run_tasks[RUN_ID] = task
        elif reason == "closed":
            row["cancelled"] = True
            _web_turn(observer)
        elif reason == "unobserved":
            _web_turn(observer)
            return None
        elif reason == "insufficient":
            # A known extract failure seeds the requested URL with no passage.
            observer.tool_execution(
                tool_name="web_extract", args={"urls": [URL]},
                next_call=lambda _args: json.dumps({"success": False, "error": "backend"}),
            )
            returned = await asyncio.to_thread(_final_turn, observer, _final_response())
            return cast(str, returned.choices[0].message.content)
        else:
            _web_turn(observer)
            response = _final_response()
            if reason == "deadline":
                row["deadline"] = 0.0
            elif reason == "final_response":
                response.choices[0].finish_reason = "length"
            elif reason == "input_bound":
                import hermes_realtime.integration.native_web_evidence as module

                self.monkeypatch.setattr(module, "_MAX_INPUT_BYTES", 8)
            elif reason == "internal":
                import hermes_realtime.integration.native_web_evidence as module

                def broken(*_args: Any) -> str:
                    raise RuntimeError("synthetic")

                self.monkeypatch.setattr(module, "presentation_input", broken)
            elif reason == "presentation_failed":
                async def failed(*_args: Any) -> Any:
                    raise ValueError("fixed-test-provider-failure")

                self.presentation.complete = failed  # type: ignore[method-assign]
            elif reason == "route":
                def wrong_route(_kw: dict[str, Any]) -> None:
                    raise ValueError("presentation-route")

                self.presentation.validate_route = wrong_route  # type: ignore[method-assign]
            elif reason == "source_changed":
                original = self.presentation.complete

                async def changing(*args: Any) -> Any:
                    row["passages"] = MappingProxyType(dict(row["passages"]))
                    return await original(*args)

                self.presentation.complete = changing  # type: ignore[method-assign]
            elif reason == "too_long":
                async def cut(*_args: Any) -> Any:
                    # The presenter reports its own cut; the JSON it cut is never parsed.
                    return SimpleNamespace(choices=[SimpleNamespace(
                        finish_reason="length", message=SimpleNamespace(
                            content='{"answer": {"lead": "Cut', refusal=None,
                            tool_calls=None))], usage=None)

                self.presentation.complete = cut  # type: ignore[method-assign]
            returned = await asyncio.to_thread(_final_turn, observer, response)
            return cast(str, returned.choices[0].message.content)
        return cast(str, (await asyncio.to_thread(_final_turn, observer, _final_response()))
                    .choices[0].message.content)


# reason -> (phase, cause, error_class[, stage, selected_source_count, rendered_detail_length])
_REFUSAL_CAUSES: dict[str, tuple[Any, ...]] = {
    "binding": ("tool-binding", "web-evidence-binding", "ValueError"),
    "capture": ("capture", "native-result-json", "ValueError"),
    "closed": ("tool-binding", "web-evidence-closed", "ValueError"),
    "deadline": ("final", "web-evidence-deadline", "ValueError"),
    "final_response": ("final", "native-final-response", "ValueError"),
    "input_bound": ("final", "web-evidence-input-bound", "ValueError"),
    "insufficient": ("final", "web-evidence-insufficient", "ValueError"),
    "internal": ("final", "none", "RuntimeError"),
    "presentation_failed": ("presentation", "undeclared", "ValueError"),
    "route": ("route", "presentation-route", "ValueError"),
    "source_changed": ("publish", "web-evidence-source-changed", "ValueError",
                       "terminal-contract", 1, 117),
    "too_long": ("render", "presentation-content-length", "ValueError",
                 "response-envelope", None, None),
    "unobserved": ("status", "none", "none"),
}


def test_refusal_cause_table_covers_every_reason() -> None:
    assert set(_REFUSAL_CAUSES) == REFUSED_REASONS


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", sorted(REFUSED_REASONS))
async def test_each_refusal_reason_reaches_attestation_and_marker(
    reason: str, monkeypatch: Any, capsys: Any
) -> None:
    _install_fake_openai_models(monkeypatch)
    scenario = _RefusalScenario(monkeypatch)
    try:
        published = await scenario.run(reason)
        if reason == "unobserved":
            # A completed terminal whose output the observer cannot bind.
            raw = await _terminal_status(
                scenario.observer,
                {"object": "hermes.run", "run_id": RUN_ID, "status": "completed",
                 "output": None},
            )
            assert raw is not None
            assert attestation_outcome(raw, GENERATION, RUN_ID, "anything") is None
            report = json.loads(raw)
        else:
            assert published == '{"error": "policy_refused"}'
            raw = scenario.observer.reports[RUN_ID]
            assert attestation_outcome(raw, GENERATION, RUN_ID, published) == ("refused", reason)
            report = json.loads(raw)
        assert report["disposition"] == "refused"
        assert report["reason"] == reason
        markers = [line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("[web-evidence] ")]
        phase, cause, error_class, *presenter = _REFUSAL_CAUSES[reason]
        stage, selected, length = presenter or ("none", None, None)
        assert markers == [
            "[web-evidence] " + json.dumps(
                {"category": "refused", "cause": cause, "count": 1, "error_class": error_class,
                 "phase": phase, "reason": reason, "rendered_detail_length": length,
                 "selected_source_count": selected, "stage": stage},
                separators=(",", ":"))
        ]
    finally:
        if scenario.foreign is not None:
            scenario.foreign.cancel()
            await asyncio.gather(scenario.foreign, return_exceptions=True)


@pytest.mark.asyncio
async def test_deterministic_render_failure_is_internal_not_retryable(
    monkeypatch: Any, capsys: Any
) -> None:
    _install_fake_openai_models(monkeypatch)

    class MalformedPresentation(_Presentation):
        async def complete(self, *_args: Any, **_kwargs: Any) -> Any:
            return _final_response("not a presentation envelope")

    task = asyncio.current_task()
    assert task is not None
    observer = _observer(MalformedPresentation())
    _bind_run(observer, task)
    _web_turn(observer)
    returned = await asyncio.to_thread(_final_turn, observer, _final_response())

    published = returned.choices[0].message.content
    assert published == '{"error": "policy_refused"}'
    assert attestation_outcome(observer.reports[RUN_ID], GENERATION, RUN_ID, published) == (
        "refused", "internal")
    assert capsys.readouterr().out == (
        '[web-evidence] {"category":"refused","cause":"presentation-json","count":1,'
        '"error_class":"ValueError","phase":"render","reason":"internal",'
        '"rendered_detail_length":null,"selected_source_count":null,"stage":"json"}\n'
    )


def test_refusal_reason_table_is_closed_over_the_contract() -> None:
    from hermes_realtime.integration.native_web_evidence import _REASONS

    assert set(_REASONS.values()) <= REFUSED_REASONS


@pytest.mark.asyncio
async def test_accepted_post_binds_row_and_returns_exact_run_headers() -> None:
    observer = _observer()
    observer._get_session_key = lambda: RUN_ID
    adapter = SimpleNamespace(
        _active_run_tasks={},
        _check_auth=lambda _request: None,
    )
    app = web.Application()
    native_task: asyncio.Task[Any] | None = None
    effect_saw_bound_row = False

    def first_native_effect(_args: dict[str, Any]) -> str:
        nonlocal effect_saw_bound_row
        row = observer.rows.get(RUN_ID)
        effect_saw_bound_row = row is not None and row["native_task"] is native_task
        return _search_result()

    async def execute_first_effect() -> str:
        return await asyncio.to_thread(
            observer.tool_execution,
            tool_name="web_search",
            args={"query": "current report"},
            next_call=first_native_effect,
        )

    async def native_runs(_request: web.Request) -> web.Response:
        nonlocal native_task
        native_task = asyncio.create_task(execute_first_effect())
        adapter._active_run_tasks[RUN_ID] = native_task
        return web.json_response({"run_id": RUN_ID, "status": "started"}, status=202)

    app.router.add_post("/v1/runs", native_runs)
    observer.install(app, adapter)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post(
            "/v1/runs",
            json={"input": "Summarize the report."},
            headers={"X-Web-Evidence-Contract": CONTRACT, "X-Web-Evidence-Generation": GENERATION},
        )
        assert response.status == 202
        assert response.headers["X-Web-Evidence-Contract"] == CONTRACT
        assert response.headers["X-Web-Evidence-Generation"] == GENERATION
        assert response.headers["X-Web-Evidence-Run"] == RUN_ID
        assert native_task is not None
        await native_task
        assert effect_saw_bound_row is True
        assert observer.rows[RUN_ID]["native_task"] is native_task
    finally:
        await client.close()
