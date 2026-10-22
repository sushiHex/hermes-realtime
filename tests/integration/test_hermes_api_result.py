from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from hermes_realtime.host_launcher import _completion_projections
from hermes_realtime.integration import HermesApiConfig, HermesApiTaskSession
from hermes_realtime.integration.api import _PendingApproval, _RunAuthority
from hermes_realtime.protocol import WorkTerminalStatus
from tests.integration.test_hermes_api_session import (
    _available_port,
    _cancel_request,
    _capabilities_response,
    _dispatch_request,
)

_REFUSAL = "The task finished, but its result could not be used."
_MARKER = "[hermes-result-refusal] "
_RUN_ID = "run_0123456789abcdef"


def _session() -> tuple[HermesApiTaskSession, _RunAuthority]:
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url="http://127.0.0.1:8642",
            bearer="result-test-bearer-value-32-characters",
        ),
        session_id="session_api_1",
    )
    authority = _RunAuthority("task_release_check", _RUN_ID, "deleg_result_private")
    session._started = True
    session._runs_by_task[authority.task_id] = authority
    session._runs_by_api[_RUN_ID] = authority
    session._record_admitted.append(_RUN_ID)

    async def unexpected_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("completed content must not trigger settlement requests")

    session._request_json = unexpected_request  # type: ignore[method-assign,assignment]
    return session, authority


@pytest.mark.asyncio
async def test_untagged_completed_result_is_refused_without_reclassifying_backend(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session, authority = _session()
    try:
        await session._consume_event(
            authority,
            {"event": "run.completed", "run_id": _RUN_ID, "output": "Synthetic raw result."},
        )
        terminal = await session.next_update()
        assert terminal.payload.status is WorkTerminalStatus.COMPLETED
        assert terminal.payload.summary == _REFUSAL
        assert terminal.payload.reason is None
        assert authority.terminal is True
        assert session._runs_by_task == session._runs_by_api == {}
        assert session._record_admitted == []
        evidence = capsys.readouterr().out
        assert "Synthetic raw result." not in evidence
        assert json.loads(evidence.removeprefix(_MARKER)) == {
            "category": "envelope",
            "characters": 21,
            "version": 1,
        }
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_valid_result_passes_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    session, authority = _session()
    output = (
        "<voice>The check passed.</voice><detail>The check passed with supporting detail.</detail>"
    )
    try:
        await session._consume_event(
            authority, {"event": "run.completed", "run_id": _RUN_ID, "output": output}
        )
        assert (await session.next_update()).payload.summary == output
        assert capsys.readouterr().out == ""
    finally:
        await session.close()


def test_legacy_plain_completion_projection_is_unchanged() -> None:
    assert _completion_projections("Legacy plain result.") == (
        "Legacy plain result.",
        "Legacy plain result.",
    )


def _envelope(voice: str = "Useful answer.", detail: str = "Supporting detail.") -> str:
    return f"<voice>{voice}</voice><detail>{detail}</detail>"


_INVALID_RESULTS = [
    pytest.param(None, "terminal_text", id="non_string"),
    pytest.param({"voice": "Synthetic"}, "terminal_text", id="object"),
    pytest.param("x" * 4097, "terminal_text", id="oversized"),
    pytest.param("x" * 5000, "terminal_text", id="oversized_saturated"),
    pytest.param(_envelope(detail="run_0123456789abcdef"), "terminal_text", id="private_authority"),
    pytest.param(
        _envelope(detail="Xrun_0123456789abcdefY"), "terminal_text", id="embedded_private_authority"
    ),
    pytest.param(
        _envelope(detail="Xrun_0123456789abcdegY"), "terminal_text", id="nonhex_private_authority"
    ),
    pytest.param(" " + _envelope(), "envelope", id="leading_text"),
    pytest.param(_envelope() + " ", "envelope", id="trailing_text"),
    pytest.param(
        _envelope().replace("</voice><detail>", "</voice>\n<detail>"),
        "envelope",
        id="between_elements",
    ),
    pytest.param(_envelope().replace("<voice>", "<Voice>"), "envelope", id="tag_case"),
    pytest.param(
        "<detail>Detail.</detail><voice>Voice.</voice>", "envelope", id="reversed_elements"
    ),
    pytest.param("<voice>Voice.</voice><detail>Detail.", "envelope", id="incomplete"),
    pytest.param(_envelope(voice=""), "voice", id="empty_voice"),
    pytest.param(_envelope(voice="  "), "voice", id="blank_voice"),
    pytest.param(_envelope(voice=" \u200b\u200d\ufeff "), "voice", id="format_only_voice"),
    pytest.param(_envelope(voice="x" * 221), "voice", id="long_voice"),
    pytest.param(_envelope(voice="Line one.\nLine two."), "voice", id="voice_newline"),
    pytest.param(_envelope(voice="Line one.\n"), "voice", id="voice_trailing_newline"),
    pytest.param(_envelope(voice="Line one.\u2028Line two."), "voice", id="voice_unicode_line"),
    pytest.param(_envelope(voice="<b>Voice.</b>"), "voice", id="voice_html"),
    pytest.param(_envelope(voice="**Voice.**"), "voice", id="voice_bold"),
    pytest.param(_envelope(voice="_Voice._"), "voice", id="voice_underscore_emphasis"),
    pytest.param(
        _envelope(voice="_release_candidate_ passed."),
        "voice",
        id="voice_outer_underscore_emphasis",
    ),
    pytest.param(_envelope(voice="__Voice.__"), "voice", id="voice_underscore_strong"),
    pytest.param(_envelope(voice="`Voice.`"), "voice", id="voice_code"),
    pytest.param(_envelope(voice="[Voice](https://example.com)."), "voice", id="voice_link"),
    pytest.param(_envelope(voice="  # The check passed."), "voice", id="voice_leading_heading"),
    pytest.param(_envelope(voice="  - The check passed."), "voice", id="voice_leading_list"),
    pytest.param(_envelope(voice="~~Voice.~~"), "voice", id="voice_strikethrough"),
    pytest.param(_envelope(voice="[Voice][source]."), "voice", id="voice_reference_link"),
    pytest.param(_envelope(detail=""), "detail", id="empty_detail"),
    pytest.param(_envelope(detail="  "), "detail", id="blank_detail"),
    pytest.param(_envelope(detail=" \u200b\u200d\ufeff "), "detail", id="format_only_detail"),
    pytest.param(_envelope(detail="x" * 1801), "detail", id="long_detail"),
    pytest.param(_envelope(detail="<voice>Nested.</voice>"), "detail", id="nested_voice"),
    pytest.param(_envelope(detail="<detail>Nested.</detail>"), "detail", id="nested_detail"),
    *[
        pytest.param(
            _envelope(**{field: "Text" + chr(codepoint) + "text."}),
            field,
            id=f"{field}_control_{codepoint:04x}",
        )
        for field in ("voice", "detail")
        for codepoint in [*range(32), *range(127, 160), 0xD800, 0xDFFF]
        if not (field == "detail" and codepoint in (9, 10, 13))
    ],
]


@pytest.mark.parametrize(("output", "category"), _INVALID_RESULTS)
@pytest.mark.asyncio
async def test_invalid_api_result_is_never_presented(
    output: object,
    category: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session, authority = _session()
    try:
        await session._consume_event(
            authority, {"event": "run.completed", "run_id": _RUN_ID, "output": output}
        )
        terminal = await session.next_update()
        assert terminal.payload.status is WorkTerminalStatus.COMPLETED
        assert terminal.payload.summary == _REFUSAL
        assert terminal.payload.reason is None
        assert _completion_projections(terminal.payload.summary) == (_REFUSAL, _REFUSAL)
        evidence = capsys.readouterr().out.splitlines()
        assert len(evidence) == 1 and evidence[0].startswith(_MARKER)
        assert json.loads(evidence[0][len(_MARKER) :]) == {
            "category": category,
            "characters": min(len(output), 4097) if type(output) is str else 0,
            "version": 1,
        }
        assert authority.terminal
        assert session._record_admitted == []
        assert session._runs_by_task == session._runs_by_api == {}
    finally:
        await session.close()


@pytest.mark.parametrize(
    "output",
    [
        _envelope(voice="x" * 220, detail="x" * 1800),
        _envelope(detail="**Detail**\n\nMore context."),
        _envelope(voice="  The check passed. "),
        _envelope(voice="release_candidate_build passed."),
        _envelope(voice="release__candidate__build passed."),
        _envelope(voice="release_candidate_ passed."),
        _envelope(voice="_release_candidate passed."),
        _envelope(voice="release__candidate__ passed."),
        _envelope(voice="__release__candidate passed."),
        _envelope(voice="The result is 2 * 3 = 6 * 1."),
        _envelope(voice=r"The \*literal\* marker is escaped."),
        _envelope(voice="Café 👩\u200d💻 passed.", detail="日本語 café 👩\u200d💻."),
        _envelope(detail="One\tcolumn.\r\nNext line.\nFinal line.\rDone."),
    ],
    ids=[
        "exact_limits",
        "detail_markup",
        "voice_whitespace",
        "intraword_underscore",
        "intraword_double_underscore",
        "underscore_opening_boundary",
        "underscore_closing_boundary",
        "double_underscore_opening_boundary",
        "double_underscore_closing_boundary",
        "literal_multiplication",
        "escaped_emphasis",
        "unicode_joiner",
        "detail_whitespace",
    ],
)
@pytest.mark.asyncio
async def test_valid_api_result_preserves_exact_text(
    output: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session, authority = _session()
    try:
        await session._consume_event(
            authority, {"event": "run.completed", "run_id": _RUN_ID, "output": output}
        )
        assert (await session.next_update()).payload.summary == output
        assert capsys.readouterr().out == ""
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_wrong_run_identity_is_refused_before_result_validation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session, authority = _session()
    try:
        with pytest.raises(RuntimeError, match="another run"):
            await session._consume_event(
                authority,
                {"event": "run.completed", "run_id": "run_fedcba9876543210", "output": None},
            )
        assert not authority.terminal
        assert session._updates.empty()
        assert capsys.readouterr().out == ""
    finally:
        session._runs_by_task.clear()
        session._runs_by_api.clear()
        session._record_admitted.clear()
        await session.close()


@pytest.mark.asyncio
async def test_refused_completed_result_retires_once_without_cancel_or_retry() -> None:
    session, authority = _session()
    approvals: list[dict[str, str | int | bool | None]] = []
    session._approval_observer = approvals.append
    session._pending_approvals.append(
        _PendingApproval("approval_synthetic", authority.task_id, _RUN_ID)
    )
    payload = {"event": "run.completed", "run_id": _RUN_ID, "output": None}
    try:
        await session._consume_event(authority, payload)
        terminal = await session.next_update()
        assert terminal.payload.summary == _REFUSAL
        assert terminal.payload.status is WorkTerminalStatus.COMPLETED
        assert list(session._pending_approvals) == []
        assert approvals == [
            {
                "actionable": False,
                "approvalId": "approval_synthetic",
                "state": "withdrawn",
                "taskId": authority.task_id,
            }
        ]
        with pytest.raises(RuntimeError, match="replayed terminal"):
            await session._consume_event(authority, payload)
        assert session._updates.empty()
        cancellation = await session.cancel(_cancel_request())
        assert cancellation.payload.accepted is False
        assert cancellation.payload.reason == "task is not active"
    finally:
        await session.close()


@asynccontextmanager
async def _sse_session(output: object) -> AsyncIterator[tuple[HermesApiTaskSession, list[str]]]:
    calls: list[str] = []
    port = _available_port()

    async def admit(_request: web.Request) -> web.Response:
        calls.append("admit")
        return web.json_response({"run_id": _RUN_ID, "status": "started"}, status=202)

    async def events(request: web.Request) -> web.StreamResponse:
        calls.append("events")
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        event = {"event": "run.completed", "run_id": _RUN_ID, "output": output}
        await response.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        await response.write_eof()
        return response

    async def unexpected(_request: web.Request) -> web.Response:
        calls.append("unexpected_settlement")
        return web.json_response({}, status=500)

    app = web.Application()
    app.router.add_get("/v1/capabilities", _capabilities_response)
    app.router.add_post("/v1/runs", admit)
    app.router.add_get("/v1/runs/{run_id}/events", events)
    app.router.add_get("/v1/runs/{run_id}", unexpected)
    app.router.add_post("/v1/runs/{run_id}/stop", unexpected)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = HermesApiTaskSession(
        config=HermesApiConfig(
            base_url=f"http://127.0.0.1:{port}", bearer="sse-result-test-bearer-32-characters"
        ),
        session_id="session_api_1",
    )
    try:
        await session.start()
        assert (await session.dispatch(_dispatch_request())).payload.accepted
        yield session, calls
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.parametrize(
    "output",
    [
        None,
        "x" * 4097,
        "Synthetic untagged result.",
        _envelope(detail="Xrun_0123456789abcdefY"),
        _envelope(detail="Xrun_0123456789abcdegY"),
    ],
)
@pytest.mark.asyncio
async def test_sse_unusable_completion_does_not_enter_settlement(output: object) -> None:
    async with _sse_session(output) as (session, calls):
        terminal = await asyncio.wait_for(session.next_update(), timeout=2)
        assert terminal.payload.status is WorkTerminalStatus.COMPLETED
        assert terminal.payload.summary == _REFUSAL
        assert calls == ["admit", "events"]
        assert session._runs_by_task == session._runs_by_api == {}


@pytest.mark.asyncio
async def test_status_readback_uses_the_same_result_acceptance_boundary() -> None:
    session, authority = _session()
    try:
        terminal = session._terminal_from_status(_RUN_ID, {"status": "completed", "output": None})
        assert terminal is not None
        await session._consume_event(authority, terminal)
        assert (await session.next_update()).payload.summary == _REFUSAL
        assert authority.terminal
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_diagnostic_sink_failure_does_not_reopen_completed_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, authority = _session()

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic diagnostic sink unavailable")

    monkeypatch.setattr("builtins.print", unavailable)
    try:
        await session._consume_event(
            authority, {"event": "run.completed", "run_id": _RUN_ID, "output": None}
        )
        terminal = await session.next_update()
        assert terminal.payload.status is WorkTerminalStatus.COMPLETED
        assert terminal.payload.summary == _REFUSAL
        assert authority.terminal
        assert session._runs_by_task == session._runs_by_api == {}
    finally:
        await session.close()
