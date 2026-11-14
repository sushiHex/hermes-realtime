"""Effect-free consumer contract checks; no provider/model calls."""

import asyncio
import hashlib
import json
import time

import pytest
from aiohttp import web

from hermes_realtime.integration import HermesApiConfig, run_record
from hermes_realtime.integration import api as api_module
from hermes_realtime.protocol import WorkTerminalStatus
from tests.integration.test_hermes_api_session import (
    _capabilities_response,
    _dispatch_request,
    _LookupHermes,
)

CONTRACT = "hermes.web-evidence/v1"
GENERATION = "a" * 32
RUN = "run_0123456789abcdef"
OUTPUT = "<voice>Synthetic.</voice><detail>Synthetic.</detail>"


def report(disposition="presented", reason=None):
    shown = disposition == "presented"
    return {
        "version": 2, "contract": CONTRACT, "generation": GENERATION, "run_id": RUN,
        "disposition": disposition, "reason": reason,
        "output_sha256": hashlib.sha256(OUTPUT.encode()).hexdigest(),
        "input_sha256": "b" * 64 if shown else None,
        "input_bytes": 1024 if shown else 0,
        "catalog_sha256": "c" * 64 if shown else None,
        "catalog_count": 1 if shown else 0,
        "passage_count": 1 if shown else 0,
        "truncated_count": 0, "missing_count": 0,
    }


def test_web_contract_is_explicit_and_separate_from_lookup():
    config = HermesApiConfig(base_url="http://127.0.0.1:8642", bearer="x" * 32,
                             web_evidence_contract=CONTRACT)
    assert config.web_evidence_contract == CONTRACT and config.lookup_policy is None
    with pytest.raises(ValueError):
        HermesApiConfig(base_url="http://127.0.0.1:8642", bearer="x" * 32,
                        web_evidence_contract=CONTRACT, lookup_policy="ordinary-lookup/v1")


@pytest.mark.parametrize("disposition,reason", [
    ("presented", None), ("no_web", None), ("mixed_ungrounded", None),
    ("ungrounded", "unobserved"),
])
def test_exact_output_and_explicit_disposition_are_required(disposition, reason):
    from hermes_realtime.integration.web_evidence_contract import (
        attestation_outcome,
        attestation_valid,
    )

    raw = json.dumps(report(disposition, reason))
    assert attestation_valid(raw, GENERATION, RUN, OUTPUT)
    assert attestation_outcome(raw, GENERATION, RUN, OUTPUT) == (disposition, reason)
    assert not attestation_valid(raw, GENERATION, RUN, OUTPUT + " changed")
    assert not attestation_valid(raw, GENERATION, RUN, OUTPUT + " ")


def test_every_refusal_reason_is_reported_but_never_publishes():
    from hermes_realtime.integration.web_evidence_contract import (
        REFUSED_REASONS,
        attestation_outcome,
        attestation_valid,
    )

    for reason in REFUSED_REASONS:
        raw = json.dumps(report("refused", reason))
        assert attestation_outcome(raw, GENERATION, RUN, OUTPUT) == ("refused", reason)
        assert not attestation_valid(raw, GENERATION, RUN, OUTPUT)
        assert attestation_outcome(raw, GENERATION, RUN, OUTPUT + " ") is None


@pytest.mark.parametrize("change", [
    {"version": True}, {"version": 1}, {"contract": "unknown"}, {"generation": "d" * 32},
    {"run_id": "run_ffffffffffffffff"}, {"disposition": "refused"},
    {"disposition": "mixed"}, {"output_sha256": "e" * 64},
    {"input_bytes": 16385}, {"input_bytes": True}, {"input_sha256": None},
    {"catalog_count": 257}, {"passage_count": 2049}, {"truncated_count": 2},
    {"missing_count": 2}, {"extra": None},
    {"disposition": []},
    {"disposition": {}}, {"catalog_count": 0}, {"passage_count": 0}, {"missing_count": 1},
    {"reason": "unobserved"}, {"disposition": "ungrounded"},
    {"disposition": "ungrounded", "reason": "insufficient"},
    {"disposition": "refused", "reason": "unknown"},
    {"disposition": "refused", "reason": ["closed"]},
])
def test_malformed_or_unbound_attestation_cannot_publish(change):
    from hermes_realtime.integration.web_evidence_contract import attestation_valid

    assert not attestation_valid(json.dumps(report() | change), GENERATION, RUN, OUTPUT)


@pytest.mark.parametrize("change", [
    {"disposition": "no_web", "reason": "unobserved"},
    {"disposition": "mixed_ungrounded", "reason": "closed"},
    {"disposition": "ungrounded"},
    {"disposition": "refused"},
    {"disposition": "refused", "reason": "attestation"},
    {"version": 1},
])
def test_reason_is_closed_set_per_disposition(change):
    from hermes_realtime.integration.web_evidence_contract import attestation_outcome

    base = report("no_web")
    assert attestation_outcome(json.dumps(base | change), GENERATION, RUN, OUTPUT) is None


def test_duplicate_or_oversized_report_and_nonweb_grounding_claim_are_refused():
    from hermes_realtime.integration.web_evidence_contract import attestation_valid

    raw = json.dumps(report())
    assert not attestation_valid(raw[:-1] + ',"version":1}', GENERATION, RUN, OUTPUT)
    assert not attestation_valid(raw + " " * 2048, GENERATION, RUN, OUTPUT)
    assert not attestation_valid(json.dumps(report("no_web") | {"input_bytes": 1}),
                                 GENERATION, RUN, OUTPUT)


def test_v3_pending_persists_contract_and_original_generation():
    entry = run_record.PendingRun(key="a" * 32, minted_at=1.0,
        body={"input": "Synthetic", "instructions": "Synthetic"},
        web_evidence_contract=CONTRACT, web_evidence_generation=GENERATION)
    raw = run_record.run_record_bytes([entry], [])
    assert json.loads(raw)["version"] == 3
    parsed = run_record.parse_run_record(raw, max_entries=8, max_request_bytes=65536,
                                         run_id=api_module._RUN_ID_EXACT)
    assert parsed is not None
    assert parsed[0][0].web_evidence_contract == CONTRACT
    assert parsed[0][0].web_evidence_generation == GENERATION


@pytest.mark.parametrize("field,value", [
    ("web_evidence_contract", None), ("web_evidence_generation", None),
    ("web_evidence_generation", "bad"), ("web_evidence_contract", True),
])
def test_v3_record_cannot_downgrade_or_relabel_incomplete_optin(field, value):
    entry = {"key": "a" * 32, "minted_at": 1.0,
        "request": {"input": "Synthetic", "instructions": "Synthetic"},
        "lookup_policy": None, "web_evidence_contract": CONTRACT,
        "web_evidence_generation": GENERATION}
    entry[field] = value
    raw = json.dumps({"version": 3, "pending": [entry], "admitted": []}).encode()
    assert run_record.parse_run_record(raw, max_entries=8, max_request_bytes=65536,
                                       run_id=api_module._RUN_ID_EXACT) is None


class WebHermes(_LookupHermes):
    """Actual local HTTP adapter path, no downstream provider execution."""

    def __init__(self, fault):
        super().__init__(fault)
        self.run_id = RUN

    async def ready(self, request):
        assert request.headers["Authorization"] == "Bearer " + "x" * 32
        self.ready_calls += 1
        value = {"ready": True, "contract": CONTRACT, "generation": GENERATION}
        if self.fault == "ready":
            value["generation"] = "bad"
        return web.json_response(value)

    async def submit(self, request):
        self.posts.append(dict(request.headers))
        headers = {"X-Web-Evidence-Contract": CONTRACT,
                   "X-Web-Evidence-Generation": GENERATION, "X-Web-Evidence-Run": RUN}
        if self.fault == "ack-missing":
            headers.clear()
        if self.fault == "ack-generation":
            headers["X-Web-Evidence-Generation"] = "d" * 32
        response = web.json_response({"run_id": RUN, "status": "started"},
                                     status=202, headers=headers)
        if self.fault == "ack-duplicate":
            response.headers.add("X-Web-Evidence-Generation", GENERATION)
        return response

    async def state(self, request):
        disposition = self.fault if self.fault in {"no_web", "mixed_ungrounded"} else "presented"
        value = report(disposition)
        if self.fault == "ungrounded":
            value = report("ungrounded", "unobserved")
        if self.fault == "hash":
            value["output_sha256"] = "0" * 64
        if self.fault == "status-generation":
            value["generation"] = "d" * 32
        if self.fault == "status-run":
            value["run_id"] = "run_ffffffffffffffff"
        if self.fault == "refused":
            value = report("refused", "insufficient")
        headers = {"X-Web-Evidence-Attestation": json.dumps(value)}
        if self.fault == "status-missing":
            headers.clear()
        response = web.json_response({"object": "hermes.run", "run_id": RUN,
            "status": self.status, "output": OUTPUT}, headers=headers)
        if self.fault == "status-duplicate":
            response.headers.add("X-Web-Evidence-Attestation", json.dumps(value))
        return response

    async def events(self, request):
        if self.fault == "fallback":
            return web.Response(text="", content_type="text/event-stream")
        return await super().events(request)

    async def start(self):
        app = web.Application()
        app.router.add_get("/v1/capabilities", _capabilities_response)
        app.router.add_get("/hermes-realtime/v1/web-evidence/ready", self.ready)
        app.router.add_post("/v1/runs", self.submit)
        app.router.add_get("/v1/runs/{run_id}", self.state)
        app.router.add_get("/v1/runs/{run_id}/events", self.events)
        app.router.add_post("/v1/runs/{run_id}/stop", self.stop)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        return site._server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    "healthy", "no_web", "mixed_ungrounded", "ungrounded", "ordinary", "fallback", "cancelled",
    "ready", "ack-missing", "ack-generation", "ack-duplicate",
    "hash", "status-generation", "status-run", "status-missing", "status-duplicate", "refused",
])
async def test_host_gates_completed_native_output_on_exact_attestation(fault, capsys):
    hermes = WebHermes(fault)
    port = await hermes.start()
    session = api_module.HermesApiTaskSession(config=HermesApiConfig(
        base_url=f"http://127.0.0.1:{port}", bearer="x" * 32,
        web_evidence_contract=None if fault == "ordinary" else CONTRACT,
    ), session_id="session_api_1")
    try:
        await session.start()
        if fault == "ready" or fault.startswith("ack-"):
            with pytest.raises(RuntimeError, match="web evidence"):
                await session.dispatch(_dispatch_request())
            assert len(hermes.posts) == (0 if fault == "ready" else 1)
            assert hermes.stops == (0 if fault == "ready" else 1)
            assert not session._runs_by_task and not session._unpublished_api_run_ids
        else:
            ack = await session.dispatch(_dispatch_request())
            assert ack.payload.accepted
            update = await asyncio.wait_for(session.next_update(), 3)
            if fault == "cancelled":
                assert update.payload.status is WorkTerminalStatus.INTERRUPTED
            elif fault in {"healthy", "no_web", "mixed_ungrounded", "ungrounded", "ordinary",
                           "fallback"}:
                assert update.payload.status is WorkTerminalStatus.COMPLETED
                expected = (
                    "<voice>Unverified stream result</voice>"
                    "<detail>Unverified stream result</detail>"
                ) if fault == "ordinary" else OUTPUT
                assert update.payload.summary == expected
            else:
                assert update.payload.status is WorkTerminalStatus.FAILED
                assert update.payload.summary is None
                assert update.payload.reason == (
                    "The web search did not find enough source text to support an answer."
                    if fault == "refused" else
                    "The task result could not be verified, so it was withheld."
                )
        assert hermes.ready_calls == int(fault != "ordinary")
        if hermes.posts:
            assert hermes.posts[0].get("X-Web-Evidence-Contract") == (
                None if fault == "ordinary" else CONTRACT
            )
            assert hermes.posts[0].get("X-Web-Evidence-Generation") == (
                None if fault == "ordinary" else GENERATION
            )
        output = capsys.readouterr().out
        assert OUTPUT not in output and GENERATION not in output and RUN not in output
    finally:
        await session.close()
        await hermes.runner.cleanup()


@pytest.mark.asyncio
async def test_pending_generation_is_persisted_before_first_post(tmp_path):
    hermes = WebHermes("healthy")
    port = await hermes.start()
    record = tmp_path / "runs.json"
    session = api_module.HermesApiTaskSession(config=HermesApiConfig(
        base_url=f"http://127.0.0.1:{port}", bearer="x" * 32, web_evidence_contract=CONTRACT,
    ), session_id="session_api_1", run_record_path=record)
    original = session._submit_run

    async def submit(pending, **kwargs):
        saved = json.loads(record.read_text())
        assert saved["version"] == 3
        assert saved["pending"][0]["web_evidence_generation"] == GENERATION
        assert saved["pending"][0]["web_evidence_contract"] == CONTRACT
        return await original(pending, **kwargs)

    session._submit_run = submit
    try:
        await session.start()
        assert (await session.dispatch(_dispatch_request())).payload.accepted
        update = await asyncio.wait_for(session.next_update(), 3)
        assert update.payload.status is WorkTerminalStatus.COMPLETED
    finally:
        await session.close()
        await hermes.runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("current", [None, CONTRACT])
async def test_restart_resend_uses_saved_web_generation_not_current_readiness(current):
    session = api_module.HermesApiTaskSession(config=HermesApiConfig(
        base_url="http://127.0.0.1:8642", bearer="x" * 32, web_evidence_contract=current,
    ), session_id="session_api_1")
    entry = run_record.PendingRun(key="a" * 32, minted_at=time.time(),
        body={"input": "Synthetic", "instructions": "Synthetic"},
        web_evidence_contract=CONTRACT, web_evidence_generation=GENERATION)
    observed = []

    async def request(method, path, **kwargs):
        observed.append((method, path, kwargs))
        if method == "POST":
            return 202, {"run_id": RUN}
        return 200, {"run_id": RUN, "status": "cancelled"}

    session._request_json = request
    assert await session._settle_recorded_dispatch(entry, 3600) == "stopped"
    assert observed[0][2]["headers"] == {"Idempotency-Key": entry.key,
        "X-Web-Evidence-Contract": CONTRACT, "X-Web-Evidence-Generation": GENERATION}
    assert len(observed) == 2


@pytest.mark.asyncio
async def test_lost_ack_resends_the_same_saved_generation():
    session = api_module.HermesApiTaskSession(config=HermesApiConfig(
        base_url="http://127.0.0.1:8642", bearer="x" * 32), session_id="session_api_1")
    session._resend_window = 3600
    entry = run_record.PendingRun(key="a" * 32, minted_at=time.time(),
        body={"input": "Synthetic", "instructions": "Synthetic"},
        web_evidence_contract=CONTRACT, web_evidence_generation=GENERATION)
    observed = []

    async def request(method, path, **kwargs):
        observed.append((kwargs["body"], dict(kwargs["headers"])))
        if len(observed) == 1:
            raise api_module._NoResponse("synthetic")
        return 202, {"run_id": RUN, "status": "running", "replayed": True}

    session._request_json = request
    await session._submit_run(entry, response_headers={})
    assert len(observed) == 2 and observed[0] == observed[1]
    assert observed[0][1]["X-Web-Evidence-Generation"] == GENERATION


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body", [
    (503, {"object": "hermes.run", "run_id": RUN, "status": "completed"}),
    (200, {"object": "hermes.run", "run_id": "run_foreign", "status": "completed"}),
    (200, {"object": "hermes.run", "run_id": RUN, "status": "running"}),
    (200, {"run_id": RUN, "status": "completed"}),
])
async def test_unknown_native_terminal_never_releases_host_authority(status, body):
    session = api_module.HermesApiTaskSession(config=HermesApiConfig(
        base_url="http://127.0.0.1:8642", bearer="x" * 32), session_id="session_api_1")

    async def request(*args, **kwargs):
        return status, body

    session._request_json = request
    authority = api_module._RunAuthority(task_id="task_synthetic", api_run_id=RUN,
        protocol_run_id="deleg_synthetic", web_evidence_generation=GENERATION)
    with pytest.raises(RuntimeError, match="authority"):
        await session._web_completion(authority)
    assert authority.terminal is False


async def _host_terminal(raw, body, run_id, generation=GENERATION):
    session = api_module.HermesApiTaskSession(config=HermesApiConfig(
        base_url="http://127.0.0.1:8642", bearer="x" * 32), session_id="session_api_1")

    async def request(method, path, *, response_headers=None, **kwargs):
        if raw is not None:
            response_headers["x-web-evidence-attestation"] = raw
        return 200, body

    session._request_json = request
    authority = api_module._RunAuthority(task_id="task_synthetic", api_run_id=run_id,
        protocol_run_id="deleg_synthetic", web_evidence_generation=generation)
    return await session._web_completion(authority)


@pytest.mark.asyncio
async def test_turn_budget_final_answer_reaches_the_user_from_the_real_companion(capsys):
    from tests.test_native_web_evidence import GENERATION as COMPANION_GENERATION
    from tests.test_native_web_evidence import (
        RUN_ID,
        UNOBSERVED_FINAL,
        _budget_spent_run,
        _observer,
        _terminal_status,
    )

    observer = _observer()
    await _budget_spent_run(observer)
    body = {"object": "hermes.run", "run_id": RUN_ID, "status": "completed",
            "output": UNOBSERVED_FINAL}
    raw = await _terminal_status(observer, body)

    terminal = await _host_terminal(raw, body, RUN_ID, COMPANION_GENERATION)

    assert terminal == {"event": "run.completed", "run_id": RUN_ID, "output": UNOBSERVED_FINAL}
    assert capsys.readouterr().out.splitlines()[-1] == (
        '[hermes-web-evidence] {"category":"ungrounded","reason":"unobserved","version":1}'
    )


@pytest.mark.asyncio
async def test_web_only_turn_budget_final_is_withheld_with_a_truthful_notice(capsys):
    from tests.test_native_web_evidence import GENERATION as COMPANION_GENERATION
    from tests.test_native_web_evidence import (
        RUN_ID,
        UNOBSERVED_FINAL,
        _budget_spent_run,
        _observer,
        _terminal_status,
    )

    observer = _observer()
    await _budget_spent_run(observer, "web_only")
    body = {"object": "hermes.run", "run_id": RUN_ID, "status": "completed",
            "output": UNOBSERVED_FINAL}
    raw = await _terminal_status(observer, body)

    terminal = await _host_terminal(raw, body, RUN_ID, COMPANION_GENERATION)

    assert terminal == {"event": "run.failed", "run_id": RUN_ID, "error": (
        "The task's answer could not be checked against its web sources, so it was withheld."
    )}
    assert "try again" not in terminal["error"].lower()
    assert capsys.readouterr().out.splitlines()[-1] == (
        '[hermes-web-evidence] {"category":"completion","reason":"unobserved","version":1}'
    )


_RETRYABLE = {"deadline", "presentation_failed", "source_changed"}


def test_every_refusal_reason_has_exactly_one_notice():
    from hermes_realtime.integration.web_evidence_contract import REFUSED_REASONS

    assert set(api_module._WEB_REFUSAL_NOTICES) == REFUSED_REASONS
    assert "try again" not in api_module._WEB_UNVERIFIED_NOTICE.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", sorted([
    "binding", "capture", "closed", "deadline", "final_response", "input_bound",
    "insufficient", "internal", "presentation_failed", "route", "source_changed",
    "too_long", "unobserved",
]))
async def test_host_speaks_the_attested_refusal_reason(reason, capsys):
    body = {"object": "hermes.run", "run_id": RUN, "status": "completed", "output": OUTPUT}
    terminal = await _host_terminal(json.dumps(report("refused", reason)), body, RUN)

    assert terminal["event"] == "run.failed"
    assert terminal["error"] == api_module._WEB_REFUSAL_NOTICES[reason]
    assert ("try again" in terminal["error"].lower()) is (reason in _RETRYABLE)
    assert capsys.readouterr().out == (
        "[hermes-web-evidence] " + json.dumps(
            {"category": "completion", "reason": reason, "version": 1}, separators=(",", ":"))
        + "\n"
    )


@pytest.mark.asyncio
async def test_unbound_attestation_is_withheld_without_a_retry_prompt(capsys):
    body = {"object": "hermes.run", "run_id": RUN, "status": "completed", "output": OUTPUT}
    terminal = await _host_terminal(None, body, RUN)

    assert terminal == {"event": "run.failed", "run_id": RUN,
                        "error": "The task result could not be verified, so it was withheld."}
    assert capsys.readouterr().out == (
        '[hermes-web-evidence] {"category":"completion","reason":"attestation","version":1}\n'
    )


@pytest.mark.parametrize("selection", ["", True, 1, "unknown"])
def test_web_evidence_selection_never_coerces_unknown_values(selection):
    with pytest.raises(ValueError, match="web_evidence_contract"):
        HermesApiConfig(base_url="http://127.0.0.1:8642", bearer="x" * 32,
                        web_evidence_contract=selection)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_records_do_not_acquire_current_web_contract(version):
    entry = {"key": "a" * 32, "minted_at": 1.0,
             "request": {"input": "Synthetic", "instructions": "Synthetic"}}
    if version == 2:
        entry["lookup_policy"] = None
    raw = json.dumps({"version": version, "pending": [entry], "admitted": []}).encode()
    parsed = run_record.parse_run_record(raw, max_entries=8, max_request_bytes=65536,
                                         run_id=api_module._RUN_ID_EXACT)
    assert parsed is not None
    assert parsed[0][0].web_evidence_contract is None
    assert parsed[0][0].web_evidence_generation is None
