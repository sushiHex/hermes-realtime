"""Authenticated proposal decisions remain distinct from downstream approvals."""

import asyncio
import json
from types import MethodType

import pytest

from hermes_realtime.client import (
    BrowserBootstrapApplication,
    BrowserClientRuntime,
    BrowserEventProjection,
    BrowserSessionDirector,
    BrowserTokenIssuer,
    BrowserTokenVerifier,
    LoopbackPeerAuthorizer,
)
from hermes_realtime.client.task_proposal_control import TaskProposalDecision
from hermes_realtime.conversation.task_proposals import (
    ConversationTaskProposals,
    HostTaskProposalRequest,
)
from hermes_realtime.conversation.work_tools import WorkCommandAdmission
from hermes_realtime.livekit import LiveKitConnection, LiveKitRoomPeer
from hermes_realtime.livekit.worker import LiveKitConversationWorker


def setup():
    projection = BrowserEventProjection()
    dispatched = []
    approvals = []

    class Surface:
        max_objective_chars = 1024

        async def submit_start_command(self, *, objective, invocation_id):
            dispatched.append((objective, invocation_id))
            return WorkCommandAdmission.ADMITTED

    proposals = ConversationTaskProposals(surface=Surface(), observer=projection.publish)
    connection = LiveKitConnection("wss://livekit.test", "key", "synthetic-secret" * 3)
    identities = iter(["browser_0123456789abcdef", "browser_fedcba9876543210"])
    generation = 0

    async def provision(identity):
        nonlocal generation
        generation += 1
        proposals.bind(identity, generation)
        return generation

    async def noop(*args):
        pass

    async def approval(*args):
        approvals.append(args)

    async def decide(participant, generation, proposal_id, objective, decision):
        return await proposals.decide(
            participant=participant,
            generation=generation,
            proposal_id=proposal_id,
            objective=objective,
            decision=decision,
        )

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection, room_name="room", identity_factory=lambda: next(identities)
        ),
        provision=provision,
        reprovision=provision,
        submit=noop,
        stop=noop,
        approval=approval,
        projection=projection,
        on_session_started=proposals.bind,
        task_proposal_decision=decide,
    )
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="room"),
        capability=None,
        loopback_authorizer=LoopbackPeerAuthorizer(),
        allowed_origin="http://127.0.0.1:8765",
        worker_identity="worker_hermes_browser",
    )
    return director, app, proposals, projection, dispatched, approvals


async def request(app, token, body, **headers):
    encoded = json.dumps(body).encode() if isinstance(body, dict) else body
    return await app.handle(
        method="POST",
        path="/api/v1/task-proposal-decision",
        headers={
            "origin": "http://127.0.0.1:8765",
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
            "content-length": str(len(encoded)),
            **headers,
        },
        body=encoded,
    )


def decision(proposal, **changes):
    return (
        dict(
            proposalId=proposal.proposal_id,
            objective=proposal.objective,
            generation=proposal.generation,
            decision="confirm",
            sequence=1,
        )
        | changes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["confirm", "decline"])
async def test_exact_sequenced_decision_consumes_once_without_downstream_approval(choice):
    director, app, proposals, projection, dispatched, approvals = setup()
    credential = await director.start()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect **exact** release"))
    assert projection.events_after(0)[-1].kind == "task_proposal"
    body = decision(proposal, decision=choice)
    first, duplicate = await asyncio.gather(
        request(app, credential.token, body), request(app, credential.token, body)
    )
    assert first.status == duplicate.status == 202
    assert json.loads(first.body) == {
        "version": 1,
        "proposalId": proposal.proposal_id,
        "generation": proposal.generation,
        "decision": choice,
        "sequence": 1,
    }
    assert len(dispatched) == (1 if choice == "confirm" else 0)
    assert approvals == []
    assert all(event.data.get("status") != "active" for event in projection.events_after(0))
    with pytest.raises(RuntimeError):
        await request(
            app,
            credential.token,
            decision(proposal, decision="decline" if choice == "confirm" else "confirm"),
        )
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"generation": 2},
        {"generation": True},
        {"sequence": 2},
        {"sequence": True},
        {"sequence": 0},
        {"sequence": 2**53},
        {"generation": 0},
        {"generation": 2**53},
        {"proposalId": "proposal_other_123"},
        {"proposalId": "invalid"},
        {"objective": "changed"},
        {"objective": " "},
        {"objective": "x" * 1025},
        {"objective": 1},
        {"decision": "approve"},
        {"decision": True},
        {"extra": True},
    ],
)
async def test_invalid_or_mismatched_decision_dispatches_nothing(change):
    director, app, proposals, _projection, dispatched, approvals = setup()
    credential = await director.start()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect exact release"))
    with pytest.raises((TypeError, ValueError, PermissionError, RuntimeError)):
        await request(app, credential.token, decision(proposal, **change))
    assert dispatched == approvals == []
    await request(app, credential.token, decision(proposal))
    assert len(dispatched) == 1
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"authorization": "Bearer invalid"},
        {"origin": "http://other.test"},
        {"content-type": "text/plain"},
    ],
)
async def test_proposal_http_authentication_and_origin_are_required(headers):
    director, app, proposals, _projection, dispatched, _approvals = setup()
    credential = await director.start()
    proposal = proposals.offer(HostTaskProposalRequest("Inspect release"))
    with pytest.raises((PermissionError, ValueError)):
        await request(app, credential.token, decision(proposal), **headers)
    assert dispatched == []
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body", [b"{", b"\xff", b"[]", b"", b" " * 8193, b'{"sequence":1,"sequence":1}']
)
async def test_proposal_http_requires_bounded_strict_json(body):
    director, app, _proposals, _projection, dispatched, _approvals = setup()
    credential = await director.start()
    with pytest.raises((TypeError, ValueError)):
        await request(app, credential.token, body)
    assert dispatched == []
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["oversized", "duplicate"])
async def test_http_rejects_invalid_envelopes_even_with_an_exact_valid_decision(kind):
    director, app, proposals, _projection, dispatched, _approvals = setup()
    credential = await director.start()
    proposal = proposals.offer(HostTaskProposalRequest("Exact objective"))
    encoded = json.dumps(decision(proposal)).encode()
    body = b" " * 8192 + encoded if kind == "oversized" else encoded[:-1] + b',"sequence":1}'
    with pytest.raises(ValueError):
        await request(app, credential.token, body)
    assert dispatched == []
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "objective", ["\U0001f9ea" * 1024, '"\\\n' * 341 + "X"], ids=["unicode", "escaped"]
)
async def test_full_unicode_and_escaped_boundary_objective_round_trips(objective):
    director, app, proposals, _projection, dispatched, _approvals = setup()
    credential = await director.start()
    proposal = proposals.offer(HostTaskProposalRequest(objective))
    # Same escaping as JSON.stringify is no larger than ensure_ascii=False here.
    encoded = json.dumps(decision(proposal), ensure_ascii=False).encode()
    assert len(encoded) <= 8192
    await request(app, credential.token, encoded)
    assert dispatched[0][0] == objective
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
async def test_old_generation_is_not_laundered_by_the_new_authenticated_binding():
    director, app, proposals, _projection, dispatched, _approvals = setup()
    first = await director.start()
    old = proposals.offer(HostTaskProposalRequest("Old objective"))
    replacement = await director.rebind(
        participant_identity=first.participant_identity, request_id="rebind_0123456789abcdef"
    )
    with pytest.raises(PermissionError):
        await request(app, replacement.token, decision(old))
    assert dispatched == []
    new = proposals.offer(HostTaskProposalRequest("New objective"))
    await request(app, replacement.token, decision(new))
    assert dispatched[0][0] == "New objective"
    await director.stop(participant_identity=replacement.participant_identity)


def test_proposal_projection_replays_pending_and_settlement_without_truncation():
    projection = BrowserEventProjection()
    data = dict(
        proposalId="proposal_12345678",
        objective="Review the exact objective",
        generation=1,
        status="pending",
        reason=None,
    )
    projection.publish("task_proposal", data)
    projection.acknowledge_through(1)
    projection.reset()
    projection.republish_open_state()
    assert dict(projection.events_after(0)[0].data) == data
    projection.publish("task_proposal", data | {"status": "retired"})
    projection.reset()
    projection.republish_open_state()
    assert [dict(event.data) for event in projection.events_after(0)] == [
        data | {"status": "retired"}
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"objective": "x" * 1025},
        {"generation": True},
        {"proposalId": "invalid"},
        {"status": "active"},
        {"extra": True},
        {"reason": "arbitrary"},
    ],
)
def test_projection_rejects_invalid_proposal_fields(change):
    projection = BrowserEventProjection()
    data = (
        dict(
            proposalId="proposal_12345678",
            objective="Review exact objective",
            generation=1,
            status="pending",
            reason=None,
        )
        | change
    )
    with pytest.raises((TypeError, ValueError)):
        projection.publish("task_proposal", data)
    assert projection.events_after(0) == ()


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "change",
    [
        {"proposal_id": StringSubclass("proposal_12345678")},
        {"proposal_id": "invalid"},
        {"objective": StringSubclass("Exact objective")},
        {"objective": " "},
        {"objective": "x" * 1025},
        {"generation": True},
        {"generation": 0},
        {"generation": 2**53},
        {"decision": StringSubclass("confirm")},
        {"decision": "approve"},
        {"sequence": True},
        {"sequence": 0},
        {"sequence": 2**53},
    ],
)
def test_decision_contract_refuses_invalid_fields_before_callback(change):
    values = dict(
        proposal_id="proposal_12345678",
        objective="Exact objective",
        generation=1,
        decision="confirm",
        sequence=1,
    )
    TaskProposalDecision(**values)
    with pytest.raises(ValueError):
        TaskProposalDecision(**(values | change))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        dict(participant_identity="browser_other"),
        dict(generation=2),
        dict(participant_identity=StringSubclass("browser_0123456789abcdef")),
    ],
)
async def test_session_binding_refusal_precedes_even_a_permissive_callback(change):
    director, _app, _proposals, _projection, _dispatched, _approvals = setup()
    credential = await director.start()
    calls = []

    async def accept(*args):
        calls.append(args)
        return WorkCommandAdmission.ADMITTED

    director._task_proposal_decision = accept
    values = dict(
        participant_identity=credential.participant_identity,
        generation=1,
        proposal_id="proposal_12345678",
        objective="Exact objective",
        sequence=1,
        decision="confirm",
    )
    with pytest.raises(PermissionError):
        await director.decide_task_proposal(**(values | change))
    assert calls == []
    await director.decide_task_proposal(**values)
    assert len(calls) == 1
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
async def test_rebound_binding_restarts_only_proposal_sequence():
    director, app, proposals, _projection, dispatched, _approvals = setup()
    first = await director.start()
    old = proposals.offer(HostTaskProposalRequest("Old objective"))
    await request(app, first.token, decision(old))
    replacement = await director.rebind(
        participant_identity=first.participant_identity, request_id="rebind_0123456789abcdef"
    )
    new = proposals.offer(HostTaskProposalRequest("New objective"))
    await request(app, replacement.token, decision(new))
    assert [row[0] for row in dispatched] == ["Old objective", "New objective"]
    await director.stop(participant_identity=replacement.participant_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["absent", "unavailable"])
async def test_proposal_control_requires_active_session_and_configured_authority(state):
    director, _app, _proposals, _projection, dispatched, _approvals = setup()
    if state == "unavailable":
        await director.start()
    director._task_proposal_decision = None
    with pytest.raises(RuntimeError):
        await director.decide_task_proposal(
            participant_identity="browser_0123456789abcdef",
            generation=1,
            proposal_id="proposal_12345678",
            objective="Exact objective",
            decision="confirm",
            sequence=1,
        )
    assert dispatched == []
    if state == "unavailable":
        await director.stop(participant_identity="browser_0123456789abcdef")


def test_optional_proposal_callback_requires_callable_when_configured():
    director, *_ = setup()
    values = dict(
        issuer=director._issuer,
        provision=director._provision,
        submit=director._submit,
        stop=director._stop,
        approval=director._approval,
        projection=director._projection,
    )
    BrowserSessionDirector(**values, task_proposal_decision=None)
    with pytest.raises(TypeError):
        BrowserSessionDirector(**values, task_proposal_decision=False)


@pytest.mark.asyncio
async def test_projection_capacity_refuses_before_calling_confirmation_authority(capsys):
    director, _app, _proposals, projection, _dispatched, _approvals = setup()
    credential = await director.start()
    calls = []

    async def accept(*args):
        calls.append(args)
        return WorkCommandAdmission.ADMITTED

    director._task_proposal_decision = accept
    for _ in range(256):
        try:
            projection.publish("first_foreground_token", {})
        except RuntimeError:
            break
    with pytest.raises(RuntimeError):
        await director.decide_task_proposal(
            participant_identity=credential.participant_identity,
            generation=1,
            proposal_id="proposal_12345678",
            objective="Exact objective",
            decision="confirm",
            sequence=1,
        )
    assert calls == []
    assert (
        capsys.readouterr().out
        == '[task-proposal-control] {"category":"projection_capacity","count":1}\n'
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result", [WorkCommandAdmission.REFUSED, WorkCommandAdmission.UNAVAILABLE, None, "admitted"]
)
async def test_confirmation_refusal_result_is_not_an_accepted_control_response(result):
    director, app, proposals, _projection, dispatched, _approvals = setup()
    credential = await director.start()
    proposal = proposals.offer(HostTaskProposalRequest("Exact objective"))

    async def refuse(*args):
        return result

    director._task_proposal_decision = refuse
    with pytest.raises(PermissionError):
        await request(app, credential.token, decision(proposal))
    assert dispatched == []
    await director.stop(participant_identity=credential.participant_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_real_runtime_constructor_routes_optional_proposal_control_without_media(
    enabled, tmp_path
):
    template, _app, proposals, projection, dispatched, approvals = setup()
    worker = object.__new__(LiveKitConversationWorker)

    async def connect(self, peer, **kwargs):
        return 7

    async def close(self):
        pass

    worker.connect = MethodType(connect, worker)
    worker.close = MethodType(close, worker)
    (tmp_path / "index.html").write_text("<!doctype html><title>Synthetic</title>", encoding="utf-8")
    runtime = BrowserClientRuntime(
        connection=LiveKitConnection("wss://livekit.test", "key", "synthetic-secret" * 3),
        room_name="room",
        worker_identity="worker_hermes_browser",
        worker=worker,
        approval=template._approval,
        static_root=tmp_path,
        projection=projection,
        peer_factory=lambda: object.__new__(LiveKitRoomPeer),
        task_proposal_decision=template._task_proposal_decision if enabled else None,
        on_session_started=proposals.bind,
    )
    credential = await runtime._sessions.start()
    proposal = proposals.offer(HostTaskProposalRequest("Exact runtime objective"))
    assert projection.events_after(0)[-1].data["objective"] == proposal.objective
    if enabled:
        reply = await request(runtime._server._application, credential.token, decision(proposal))
        assert reply.status == 202
        assert dispatched[0][0] == proposal.objective
    else:
        with pytest.raises(RuntimeError, match="unavailable"):
            await request(runtime._server._application, credential.token, decision(proposal))
        assert dispatched == []
    assert approvals == []
    await runtime._sessions.stop(participant_identity=credential.participant_identity)
