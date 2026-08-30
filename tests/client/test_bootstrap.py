"""Security boundary tests for browser LiveKit bootstrap credentials."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from livekit import api

from hermes_realtime.client import (
    BrowserAudioDiagnostic,
    BrowserBindingSnapshot,
    BrowserBootstrapApplication,
    BrowserBootstrapResponse,
    BrowserEventProjection,
    BrowserEvidenceConsentOperation,
    BrowserEvidenceControlResponse,
    BrowserEvidenceRevokeOperation,
    BrowserModelCatalog,
    BrowserModelConfiguration,
    BrowserSelectableModel,
    BrowserSessionDirector,
    BrowserSpeechRuntime,
    BrowserTokenIssuer,
    BrowserTokenVerifier,
    LoopbackPeerAddress,
    LoopbackPeerAuthorizer,
    OneTimeBootstrapCapability,
    TailnetPeerAddress,
    TailnetPeerAuthorizer,
)
from hermes_realtime.livekit import LiveKitConnection


async def _noop_submit(identity: str, generation: int, text: str) -> None:
    del identity, generation, text


async def _noop_stop(identity: str, generation: int) -> None:
    del identity, generation


async def _noop_approval(
    identity: str,
    generation: int,
    sequence: int,
    approval_id: str,
    decision: str,
) -> None:
    del identity, generation, sequence, approval_id, decision


@pytest.mark.asyncio
async def test_authenticated_evidence_consent_uses_server_binding_and_strict_model_parser() -> None:
    from hermes_realtime.evidence import models as m

    secret = "synthetic-browser-bootstrap-secret-32-bytes"
    connection = LiveKitConnection("wss://livekit.test", "test-key", secret)
    observed: list[tuple[object, object]] = []

    async def provision(_identity: str) -> int:
        return 17

    async def complete(request: object) -> BrowserEvidenceControlResponse:
        return BrowserEvidenceControlResponse(
            status=200,
            payload={
                "captureState": "active",
                "result": "consent_activated",
                "sequence": request.sequence,
            },
        )

    def consent(binding: object, request: object) -> BrowserEvidenceConsentOperation:
        assert director._start_lock.locked() is True
        observed.append((binding, request))
        return BrowserEvidenceConsentOperation(complete=lambda: complete(request))

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection,
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        evidence_consent=consent,
        evidence_status=lambda: m.CaptureStatusV1(
            available=True,
            capture_state=m.CaptureState.IDLE,
            retention_hours=24,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
        ),
    )
    credential = await director.start()
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "z" * 43),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )
    body = (
        b'{"accepted":true,"consentVersion":"realtime-evidence-consent-v1",'
        b'"disclosureDigest":"' + b"a" * 64 + b'","retentionHours":24,'
        b'"sequence":1,"sources":{"microphone":true,"typed":false}}'
    )

    response = await app.handle(
        method="POST",
        path="/api/v1/evidence-consent",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=body,
    )

    assert response.status == 200
    assert json.loads(response.body) == {
        "captureState": "active",
        "result": "consent_activated",
        "sequence": 1,
    }
    [(binding, request)] = observed
    assert binding.participant_identity == credential.participant_identity
    assert binding.binding_generation == 17
    assert type(request) is m.EvidenceConsentRequestV1


@pytest.mark.asyncio
async def test_evidence_consent_rejects_request_that_does_not_match_published_status() -> None:
    from hermes_realtime.evidence import models as m

    calls: list[object] = []

    async def provision(_identity: str) -> int:
        return 3

    def consent(_binding: object, request: object) -> BrowserEvidenceConsentOperation:
        calls.append(request)
        raise AssertionError("mismatched consent reached trusted issuer")

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(capacity=2),
        evidence_consent=consent,
        evidence_status=lambda: m.CaptureStatusV1(
            available=True,
            capture_state=m.CaptureState.IDLE,
            retention_hours=24,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
        ),
    )
    credential = await director.start()
    request = m.parse_evidence_consent_request(
        json.dumps(
            {
                "accepted": True,
                "consentVersion": m.CONSENT_VERSION,
                "disclosureDigest": "b" * 64,
                "retentionHours": 24,
                "sequence": 1,
                "sources": {"microphone": True, "typed": True},
            }
        ).encode()
    )

    with pytest.raises(RuntimeError, match="published capture status"):
        await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )

    assert calls == []


@pytest.mark.asyncio
async def test_authenticated_evidence_revoke_uses_independent_server_binding_gate() -> None:
    from hermes_realtime.evidence import models as m

    secret = "synthetic-browser-bootstrap-secret-32-bytes"
    connection = LiveKitConnection("wss://livekit.test", "test-key", secret)
    observed: list[tuple[object, object]] = []

    async def provision(_identity: str) -> int:
        return 19

    async def complete(request: object) -> BrowserEvidenceControlResponse:
        return BrowserEvidenceControlResponse(
            status=202,
            payload={
                "captureState": "revoked_purging",
                "result": "revoke_durably_scheduled",
                "sequence": request.sequence,
            },
        )

    def revoke(binding: object, request: object) -> BrowserEvidenceRevokeOperation:
        assert director._start_lock.locked() is False
        observed.append((binding, request))
        return BrowserEvidenceRevokeOperation(complete=lambda: complete(request))

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection,
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        evidence_revoke=revoke,
    )
    credential = await director.start()
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "z" * 43),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )
    body = b'{"sequence":1}'

    response = await app.handle(
        method="POST",
        path="/api/v1/evidence-revoke",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=body,
    )

    assert response.status == 202
    assert json.loads(response.body) == {
        "captureState": "revoked_purging",
        "result": "revoke_durably_scheduled",
        "sequence": 1,
    }
    [(binding, request)] = observed
    assert binding.participant_identity == credential.participant_identity
    assert binding.binding_generation == 19
    assert type(request) is m.EvidenceRevokeRequestV1


@pytest.mark.asyncio
async def test_evidence_consent_rejects_skipped_control_sequence_before_reservation() -> None:
    from hermes_realtime.evidence import models as m

    calls: list[object] = []

    async def provision(_identity: str) -> int:
        return 3

    def consent(_binding: object, request: object) -> BrowserEvidenceConsentOperation:
        calls.append(request)
        raise AssertionError("skipped sequence reached reservation")

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(capacity=2),
        evidence_consent=consent,
        evidence_status=lambda: m.CaptureStatusV1(
            available=True,
            capture_state=m.CaptureState.IDLE,
            retention_hours=24,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
        ),
    )
    credential = await director.start()
    request = m.parse_evidence_consent_request(
        json.dumps(
            {
                "accepted": True,
                "consentVersion": m.CONSENT_VERSION,
                "disclosureDigest": "a" * 64,
                "retentionHours": 24,
                "sequence": 2,
                "sources": {"microphone": True, "typed": True},
            }
        ).encode()
    )

    with pytest.raises(RuntimeError, match="control sequence"):
        await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )

    assert calls == []


@pytest.mark.asyncio
async def test_exact_consent_retry_shares_one_live_reserved_operation() -> None:
    from hermes_realtime.evidence import models as m

    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def provision(_identity: str) -> int:
        return 1

    async def complete() -> BrowserEvidenceControlResponse:
        entered.set()
        await release.wait()
        return BrowserEvidenceControlResponse(
            status=200,
            payload={
                "captureState": "active",
                "result": "consent_activated",
                "sequence": 1,
            },
        )

    def consent(*_args: object) -> BrowserEvidenceConsentOperation:
        nonlocal calls
        calls += 1
        return BrowserEvidenceConsentOperation(complete=complete)

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(capacity=2),
        evidence_consent=consent,
        evidence_status=lambda: m.CaptureStatusV1(
            available=True,
            capture_state=m.CaptureState.IDLE,
            retention_hours=24,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
        ),
    )
    credential = await director.start()
    request = m.parse_evidence_consent_request(
        json.dumps(
            {
                "accepted": True,
                "consentVersion": m.CONSENT_VERSION,
                "disclosureDigest": "a" * 64,
                "retentionHours": 24,
                "sequence": 1,
                "sources": {"microphone": True, "typed": True},
            }
        ).encode()
    )
    first = asyncio.create_task(
        director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
    )
    await entered.wait()
    retry = asyncio.create_task(
        director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
    )
    await asyncio.sleep(0)

    assert calls == 1
    release.set()
    assert await first == await retry


@pytest.mark.asyncio
async def test_exact_revoke_retries_share_one_reserved_operation_and_cache_result() -> None:
    from hermes_realtime.evidence import models as m

    entered = asyncio.Event()
    release = asyncio.Event()
    reservations = 0
    completed = BrowserEvidenceControlResponse(
        status=202,
        payload={
            "captureState": "revoked_purging",
            "result": "revoke_durably_scheduled",
            "sequence": 1,
        },
    )

    async def provision(_identity: str) -> int:
        return 1

    async def complete() -> BrowserEvidenceControlResponse:
        entered.set()
        await release.wait()
        return completed

    def revoke(*_args: object) -> BrowserEvidenceRevokeOperation:
        nonlocal reservations
        reservations += 1
        return BrowserEvidenceRevokeOperation(complete=complete)

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        evidence_revoke=revoke,
    )
    credential = await director.start()
    request = m.EvidenceRevokeRequestV1(sequence=1)
    first = asyncio.create_task(
        director.revoke_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
    )
    await entered.wait()
    retry = asyncio.create_task(
        director.revoke_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
    )
    await asyncio.sleep(0)

    assert reservations == 1
    release.set()
    first_result = await first
    assert await retry is first_result
    assert (
        await director.revoke_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
    ) is first_result
    assert reservations == 1


@pytest.mark.asyncio
async def test_consent_timeout_exact_retry_reobserves_one_reserved_operation() -> None:
    from hermes_realtime.evidence import models as m

    release = asyncio.Event()
    reservations = 0
    observations = 0

    async def provision(_identity: str) -> int:
        return 1

    async def complete() -> BrowserEvidenceControlResponse:
        nonlocal observations
        observations += 1
        return BrowserEvidenceControlResponse(
            status=503,
            payload={
                "captureState": "idle",
                "error": "control_timeout",
                "sequence": 1,
            },
        )

    async def settlement() -> BrowserEvidenceControlResponse:
        await release.wait()
        return BrowserEvidenceControlResponse(
            status=200,
            payload={
                "captureState": "active",
                "result": "consent_activated",
                "sequence": 1,
            },
        )

    def consent(*_args: object) -> BrowserEvidenceConsentOperation:
        nonlocal reservations
        reservations += 1
        return BrowserEvidenceConsentOperation(complete=complete, settlement=settlement)

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(capacity=2),
        evidence_consent=consent,
        evidence_status=lambda: m.CaptureStatusV1(
            available=True,
            capture_state=m.CaptureState.IDLE,
            retention_hours=24,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
        ),
    )
    credential = await director.start()
    request = m.parse_evidence_consent_request(
        json.dumps(
            {
                "accepted": True,
                "consentVersion": m.CONSENT_VERSION,
                "disclosureDigest": "a" * 64,
                "retentionHours": 24,
                "sequence": 1,
                "sources": {"microphone": True, "typed": True},
            }
        ).encode()
    )
    different_request = m.parse_evidence_consent_request(
        json.dumps(
            {
                "accepted": True,
                "consentVersion": m.CONSENT_VERSION,
                "disclosureDigest": "a" * 64,
                "retentionHours": 24,
                "sequence": 1,
                "sources": {"microphone": True, "typed": False},
            }
        ).encode()
    )

    first = await director.consent_to_evidence(
        participant_identity=credential.participant_identity,
        request=request,
    )
    assert first.payload["error"] == "control_timeout"
    with pytest.raises(RuntimeError, match="different evidence consent is pending"):
        await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=different_request,
        )

    release.set()
    activated = first
    for _ in range(100):
        activated = await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
        if activated.status == 200:
            break
        await asyncio.sleep(0.01)
    assert activated.payload["result"] == "consent_activated"
    assert (
        await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
    ) is activated
    assert reservations == 1
    assert observations == 1


@pytest.mark.asyncio
async def test_revoke_timeout_exact_retry_reobserves_one_reserved_operation() -> None:
    from hermes_realtime.evidence import models as m

    reservations = 0
    observations = 0

    async def provision(_identity: str) -> int:
        return 1

    async def complete() -> BrowserEvidenceControlResponse:
        nonlocal observations
        observations += 1
        if observations == 1:
            return BrowserEvidenceControlResponse(
                status=503,
                payload={
                    "captureState": "revoked_purging",
                    "error": "control_timeout",
                    "sequence": 1,
                },
            )
        return BrowserEvidenceControlResponse(
            status=202,
            payload={
                "captureState": "revoked_purging",
                "result": "revoke_durably_scheduled",
                "sequence": 1,
            },
        )

    def revoke(*_args: object) -> BrowserEvidenceRevokeOperation:
        nonlocal reservations
        reservations += 1
        return BrowserEvidenceRevokeOperation(complete=complete)

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        evidence_revoke=revoke,
    )
    credential = await director.start()
    request = m.EvidenceRevokeRequestV1(sequence=1)

    first = await director.revoke_evidence(
        participant_identity=credential.participant_identity,
        request=request,
    )
    assert first.payload == {
        "captureState": "revoked_purging",
        "error": "control_timeout",
        "sequence": 1,
    }

    completed = await director.revoke_evidence(
        participant_identity=credential.participant_identity,
        request=request,
    )
    assert completed.payload["result"] == "revoke_durably_scheduled"
    assert (
        await director.revoke_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
    ) is completed
    assert reservations == 1
    assert observations == 2


@pytest.mark.asyncio
async def test_evidence_revoke_binding_rotates_with_browser_rebind() -> None:
    from hermes_realtime.evidence import models as m

    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    observed: list[BrowserBindingSnapshot] = []

    async def provision(_identity: str) -> int:
        return 1

    async def reprovision(_identity: str) -> int:
        return 2

    async def complete() -> BrowserEvidenceControlResponse:
        return BrowserEvidenceControlResponse(
            status=202,
            payload={
                "captureState": "revoked_purging",
                "result": "revoke_durably_scheduled",
                "sequence": 1,
            },
        )

    def revoke(
        binding: BrowserBindingSnapshot,
        _request: object,
    ) -> BrowserEvidenceRevokeOperation:
        observed.append(binding)
        return BrowserEvidenceRevokeOperation(complete=complete)

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: next(identities),
        ),
        provision=provision,
        reprovision=reprovision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        evidence_revoke=revoke,
    )
    original = await director.start()
    replacement = await director.rebind(
        participant_identity=original.participant_identity,
        request_id="rebind_0123456789abcdef",
    )

    await director.revoke_evidence(
        participant_identity=replacement.participant_identity,
        request=m.EvidenceRevokeRequestV1(sequence=1),
    )

    assert observed == [
        BrowserBindingSnapshot(
            participant_identity=replacement.participant_identity,
            binding_generation=2,
        )
    ]


@pytest.mark.asyncio
async def test_browser_stop_clears_evidence_revoke_binding() -> None:
    from hermes_realtime.evidence import models as m

    async def provision(_identity: str) -> int:
        return 1

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        evidence_revoke=lambda *_args: pytest.fail("stopped binding reached revoke"),
    )
    credential = await director.start()

    await director.stop(participant_identity=credential.participant_identity)

    with pytest.raises(RuntimeError, match="no browser evidence binding"):
        await director.revoke_evidence(
            participant_identity=credential.participant_identity,
            request=m.EvidenceRevokeRequestV1(sequence=1),
        )


@pytest.mark.asyncio
async def test_revoke_closes_gate_while_typed_submit_holds_start_lock() -> None:
    from hermes_realtime.evidence import models as m

    entered = asyncio.Event()
    release = asyncio.Event()
    projection = BrowserEventProjection(capacity=8)
    submitted_sequences: list[int] = []

    async def provision(_identity: str) -> int:
        return 1

    async def submit(*args: object) -> None:
        sequence = args[2]
        assert type(sequence) is int
        submitted_sequences.append(sequence)
        if sequence != 1:
            return
        entered.set()
        await release.wait()

    async def complete() -> BrowserEvidenceControlResponse:
        return BrowserEvidenceControlResponse(
            status=202,
            payload={
                "captureState": "revoked_purging",
                "result": "revoke_durably_scheduled",
                "sequence": 1,
            },
        )

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=projection,
        evidence_revoke=lambda *_args: BrowserEvidenceRevokeOperation(complete=complete),
    )
    credential = await director.start()
    projection.events_after(0)
    typed = asyncio.create_task(
        director.submit_text(
            participant_identity=credential.participant_identity,
            sequence=1,
            text="do not persist after revoke",
        )
    )
    await entered.wait()

    revoke = await director.revoke_evidence(
        participant_identity=credential.participant_identity,
        request=m.EvidenceRevokeRequestV1(sequence=1),
    )
    assert revoke.status == 202
    assert not typed.done()

    release.set()
    await typed
    assert all(
        event.kind != "typed_input_admitted" for event in projection.events_after(0)
    )

    # The suppressed turn still reached production authority, so its sequence is
    # spent. If revocation rolled the commit back, the browser's next turn would
    # be refused forever and typed input would deadlock for the session.
    assert submitted_sequences == [1]
    await director.submit_text(
        participant_identity=credential.participant_identity,
        sequence=2,
        text="typed input survives evidence revocation",
    )
    assert submitted_sequences == [1, 2]


@pytest.mark.asyncio
async def test_session_start_reserves_and_publishes_initial_capture_status() -> None:
    from hermes_realtime.evidence import models as m

    projection = BrowserEventProjection(capacity=3)

    async def provision(_identity: str) -> int:
        projection.publish("notification_queued", {})
        return 1

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=projection,
        evidence_status=lambda: m.CaptureStatusV1(
            available=True,
            capture_state=m.CaptureState.IDLE,
            retention_hours=24,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
        ),
    )

    await director.start()

    assert [event.kind for event in projection.events_after(0)] == [
        "notification_queued",
        "session_ready",
        "capture_status",
    ]


def test_issuer_returns_short_lived_fixed_room_microphone_credential() -> None:
    secret = "synthetic-browser-bootstrap-secret-32-bytes"
    issuer = BrowserTokenIssuer(
        connection=LiveKitConnection("wss://livekit.test", "test-key", secret),
        room_name="hermes-local",
        ttl_seconds=60,
        identity_factory=lambda: "browser_0123456789abcdef",
    )

    credential = issuer.issue()
    claims = api.TokenVerifier("test-key", secret).verify(credential.token)
    raw_claims = jwt.decode(
        credential.token,
        secret,
        algorithms=["HS256"],
        issuer="test-key",
    )

    assert credential.url == "wss://livekit.test"
    assert credential.room_name == "hermes-local"
    assert credential.participant_identity == "browser_0123456789abcdef"
    assert credential.expires_in_seconds == 60
    assert claims.identity == credential.participant_identity
    assert claims.video is not None
    assert claims.video.room_join is True
    assert claims.video.room == "hermes-local"
    assert claims.video.can_subscribe is True
    assert claims.video.can_publish_data is False
    assert claims.video.can_publish_sources == ["microphone"]
    assert raw_claims["exp"] - raw_claims["nbf"] == 60
    assert credential.token not in repr(credential)
    assert secret not in repr(credential)


def test_verifier_accepts_only_matching_scoped_browser_token() -> None:
    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-browser-bootstrap-secret-32-bytes",
    )
    credential = BrowserTokenIssuer(
        connection=connection,
        room_name="hermes-local",
        identity_factory=lambda: "browser_0123456789abcdef",
    ).issue()
    verifier = BrowserTokenVerifier(connection=connection, room_name="hermes-local")

    assert verifier.verify(credential.token) == "browser_0123456789abcdef"
    header, payload, signature = credential.token.split(".")
    replacement = "A" if signature[0] != "A" else "B"
    tampered = f"{header}.{payload}.{replacement}{signature[1:]}"
    with pytest.raises(PermissionError, match="invalid"):
        verifier.verify(tampered)


def test_bootstrap_capability_is_consumed_exactly_once() -> None:
    capability = OneTimeBootstrapCapability(
        ttl_seconds=60,
        token_factory=lambda: "a" * 43,
        clock=lambda: 100.0,
    )

    capability.consume("a" * 43)

    try:
        capability.consume("a" * 43)
    except RuntimeError as error:
        assert str(error) == "bootstrap capability has already been consumed"
    else:
        raise AssertionError("replayed bootstrap capability was accepted")


def test_bootstrap_capability_supports_one_hour_local_diagnostic_window() -> None:
    now = [100.0]
    valid = OneTimeBootstrapCapability(
        ttl_seconds=3_600,
        token_factory=lambda: "v" * 43,
        clock=lambda: now[0],
    )
    expired = OneTimeBootstrapCapability(
        ttl_seconds=3_600,
        token_factory=lambda: "x" * 43,
        clock=lambda: now[0],
    )

    now[0] = 3_699.999
    valid.consume("v" * 43)
    now[0] = 3_700.0
    with pytest.raises(RuntimeError, match="expired"):
        expired.consume("x" * 43)


@pytest.mark.asyncio
async def test_session_start_projects_authoritative_model_configuration() -> None:
    projection = BrowserEventProjection()

    async def provision(_identity: str) -> int:
        return 1

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=projection,
        speech_runtime=BrowserSpeechRuntime(
            stt_provider="moonshine",
            stt_model="moonshine-v2-small",
            tts_provider="kokoro",
            tts_model="kokoro-v1.0.onnx",
        ),
        model_configuration=BrowserModelConfiguration(
            authentication="subscription",
            provider="openai-codex",
            transport="subscription-app-server",
            model="gpt-5.6-terra",
            effort="medium",
            context_window_tokens=None,
            reports_token_usage=False,
        ),
    )

    await director.start()

    events = projection.events_after(0)
    assert [event.kind for event in events] == ["session_ready", "session_model"]
    assert dict(events[0].data) == {
        "conversationProfile": "legacy",
        "mode": "microphone_or_typed",
        "sttModel": "moonshine-v2-small",
        "sttProvider": "moonshine",
        "ttsModel": "kokoro-v1.0.onnx",
        "ttsProvider": "kokoro",
    }
    assert dict(events[1].data) == {
        "authentication": "subscription",
        "contextWindowTokens": None,
        "effort": "medium",
        "model": "gpt-5.6-terra",
        "provider": "openai-codex",
        "reportsTokenUsage": False,
        "transport": "subscription-app-server",
    }


@pytest.mark.asyncio
async def test_session_yield_preserves_exact_stream_match_result() -> None:
    claims: list[tuple[str, int, str, int, str, str]] = []

    async def provision(_identity: str) -> int:
        return 3

    async def yield_speech(
        identity: str,
        generation: int,
        turn_id: str,
        turn_generation: int,
        chunk_id: str,
        stream_id: str,
    ) -> bool:
        claims.append((identity, generation, turn_id, turn_generation, chunk_id, stream_id))
        return stream_id == "stream_current"

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        yield_speech=yield_speech,
        conversation_profile="natural_v1",
    )
    credential = await director.start()

    assert await director.yield_speech(
        participant_identity=credential.participant_identity,
        turn_id="turn_001",
        turn_generation=7,
        chunk_id="chunk_001",
        stream_id="stream_current",
    )
    assert not await director.yield_speech(
        participant_identity=credential.participant_identity,
        turn_id="turn_001",
        turn_generation=7,
        chunk_id="chunk_001",
        stream_id="stream_stale",
    )
    assert claims == [
        (credential.participant_identity, 3, "turn_001", 7, "chunk_001", "stream_current"),
        (credential.participant_identity, 3, "turn_001", 7, "chunk_001", "stream_stale"),
    ]


@pytest.mark.asyncio
async def test_session_stop_does_not_wait_for_blocked_speech_yield_cleanup() -> None:
    yield_started = asyncio.Event()
    release_yield = asyncio.Event()

    async def provision(_identity: str) -> int:
        return 3

    async def yield_speech(
        identity: str,
        generation: int,
        turn_id: str,
        turn_generation: int,
        chunk_id: str,
        stream_id: str,
    ) -> bool:
        del identity, generation, turn_id, turn_generation, chunk_id, stream_id
        yield_started.set()
        await release_yield.wait()
        return True

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        yield_speech=yield_speech,
        conversation_profile="natural_v1",
    )
    credential = await director.start()
    yield_task = asyncio.create_task(
        director.yield_speech(
            participant_identity=credential.participant_identity,
            turn_id="turn_001",
            turn_generation=7,
            chunk_id="chunk_001",
            stream_id="stream_current",
        )
    )
    await yield_started.wait()

    await asyncio.wait_for(
        director.stop(participant_identity=credential.participant_identity),
        timeout=0.1,
    )
    release_yield.set()
    assert await yield_task is True


@pytest.mark.asyncio
async def test_session_rebind_rotates_identity_without_resetting_conversation_sequence() -> None:
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    provisioned: list[str] = []
    rebound: list[str] = []
    submitted: list[tuple[str, int, str]] = []

    async def provision(identity: str) -> int:
        provisioned.append(identity)
        return 1

    async def reprovision(identity: str) -> int:
        rebound.append(identity)
        return 2

    async def submit(identity: str, generation: int, sequence: int, text: str) -> None:
        del sequence
        submitted.append((identity, generation, text))

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: next(identities),
        ),
        provision=provision,
        reprovision=reprovision,
        submit=submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
    )

    initial = await director.start()
    await director.submit_text(
        participant_identity=initial.participant_identity,
        sequence=1,
        text="Before reconnect",
    )
    replacement = await director.rebind(
        participant_identity=initial.participant_identity,
    )
    await director.submit_text(
        participant_identity=replacement.participant_identity,
        sequence=2,
        text="After reconnect",
    )

    assert provisioned == ["browser_0123456789abcdef"]
    assert rebound == ["browser_fedcba9876543210"]
    assert replacement.participant_identity == "browser_fedcba9876543210"
    assert director.active_identity == replacement.participant_identity
    assert director.active_generation == 2
    assert submitted == [
        ("browser_0123456789abcdef", 1, "Before reconnect"),
        ("browser_fedcba9876543210", 2, "After reconnect"),
    ]
    with pytest.raises(PermissionError, match="does not own"):
        await director.refresh_credential(
            participant_identity=initial.participant_identity,
        )


@pytest.mark.asyncio
async def test_session_current_binding_snapshot_is_atomic_and_identity_bound() -> None:
    async def provision(_identity: str) -> int:
        return 17

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
    )
    credential = await director.start()

    snapshot = await director.current_binding_snapshot(
        participant_identity=credential.participant_identity,
    )

    assert snapshot.participant_identity == credential.participant_identity
    assert snapshot.binding_generation == 17
    assert snapshot.__dataclass_params__.frozen is True
    assert not hasattr(snapshot, "__dict__")
    with pytest.raises(PermissionError, match="does not own"):
        await director.current_binding_snapshot(
            participant_identity="browser_fedcba9876543210",
        )

    await director.stop(participant_identity=credential.participant_identity)
    with pytest.raises(RuntimeError, match="no browser session"):
        await director.current_binding_snapshot(
            participant_identity=credential.participant_identity,
        )


@pytest.mark.asyncio
async def test_session_rebind_replays_previous_identity_without_reprovisioning() -> None:
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    rebound: list[str] = []

    async def reprovision(identity: str) -> int:
        rebound.append(identity)
        return 2

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: next(identities),
        ),
        provision=lambda _identity: asyncio.sleep(0, result=1),
        reprovision=reprovision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
    )

    initial = await director.start()
    request_id = "rebind_0123456789abcdef"
    replacement = await director.rebind(
        participant_identity=initial.participant_identity,
        request_id=request_id,
    )
    replay = await director.rebind(
        participant_identity=initial.participant_identity,
        request_id=request_id,
    )
    repeated_replay = await director.rebind(
        participant_identity=initial.participant_identity,
        request_id=request_id,
    )

    assert rebound == [replacement.participant_identity]
    assert replay.participant_identity == replacement.participant_identity
    assert replay == replacement
    assert repeated_replay == replacement
    assert director.active_identity == replacement.participant_identity
    assert director.active_generation == 2
    with pytest.raises(PermissionError, match="does not own"):
        await director.stop(participant_identity=initial.participant_identity)
    await director.stop(participant_identity=replacement.participant_identity)
    assert director.active_identity is None


@pytest.mark.asyncio
async def test_session_stop_settles_a_lost_rebind_response_only_with_its_request_id() -> None:
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    stopped: list[tuple[str, int]] = []

    async def stop(identity: str, generation: int) -> None:
        stopped.append((identity, generation))

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: next(identities),
        ),
        provision=lambda _identity: asyncio.sleep(0, result=1),
        reprovision=lambda _identity: asyncio.sleep(0, result=2),
        submit=_noop_submit,
        stop=stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
    )
    initial = await director.start()
    request_id = "rebind_0123456789abcdef"
    await director.rebind(
        participant_identity=initial.participant_identity,
        request_id=request_id,
    )

    with pytest.raises(PermissionError, match="does not own"):
        await director.stop(participant_identity=initial.participant_identity)
    await director.stop(
        participant_identity=initial.participant_identity,
        request_id=request_id,
    )

    assert stopped == [("browser_fedcba9876543210", 2)]
    assert director.active_identity is None


@pytest.mark.asyncio
async def test_session_rebind_rolls_back_old_authority_after_reprovision_failure() -> None:
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    rebound: list[str] = []

    async def reprovision(identity: str) -> int:
        rebound.append(identity)
        if identity == "browser_fedcba9876543210":
            raise TimeoutError("replacement peer timed out")
        return 3

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: next(identities),
        ),
        provision=lambda _identity: asyncio.sleep(0, result=1),
        reprovision=reprovision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
    )

    initial = await director.start()
    with pytest.raises(TimeoutError, match="replacement peer timed out"):
        await director.rebind(participant_identity=initial.participant_identity)

    assert rebound == ["browser_fedcba9876543210", "browser_0123456789abcdef"]
    assert director.active_identity == initial.participant_identity
    assert director.active_generation == 3


@pytest.mark.asyncio
async def test_session_rebind_rollback_timeout_fails_closed_until_late_worker_is_stopped() -> None:
    identities = iter(
        (
            "browser_0123456789abcdef",
            "browser_fedcba9876543210",
            "browser_aabbccddeeff0011",
        )
    )
    rollback_started = asyncio.Event()
    release_rollback = asyncio.Event()
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()
    stopped: list[tuple[str, int]] = []

    async def reprovision(identity: str) -> int:
        if identity == "browser_fedcba9876543210":
            raise RuntimeError("replacement failed")
        rollback_started.set()
        await release_rollback.wait()
        return 3

    async def stop(identity: str, generation: int) -> None:
        stopped.append((identity, generation))
        stop_started.set()
        await release_stop.wait()

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: next(identities),
        ),
        provision=lambda _identity: asyncio.sleep(0, result=1),
        reprovision=reprovision,
        submit=_noop_submit,
        stop=stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        rollback_timeout_seconds=0.01,
    )

    initial = await director.start()
    started = asyncio.get_running_loop().time()
    with pytest.raises(ExceptionGroup, match="rebind and rollback failed") as raised:
        await director.rebind(participant_identity=initial.participant_identity)
    elapsed = asyncio.get_running_loop().time() - started

    assert any(
        isinstance(error, TimeoutError) and "rollback timed out" in str(error)
        for error in raised.value.exceptions
    )
    assert elapsed < 0.04
    assert director.active_identity is None
    assert director.active_generation is None
    await rollback_started.wait()
    with pytest.raises(RuntimeError, match="finalizing"):
        await director.start()
    finalizer = director._rebind_finalizer
    assert finalizer is not None
    release_rollback.set()
    await stop_started.wait()
    assert stopped == [(initial.participant_identity, 3)]
    with pytest.raises(RuntimeError, match="finalizing"):
        await director.start()
    release_stop.set()
    await finalizer

    fresh = await director.start()

    assert fresh.participant_identity == "browser_aabbccddeeff0011"


@pytest.mark.asyncio
async def test_session_rebind_finalizer_failure_keeps_new_starts_failed_closed() -> None:
    rollback_started = asyncio.Event()
    release_failure = asyncio.Event()

    async def reprovision(identity: str) -> int:
        if identity == "browser_fedcba9876543210":
            raise RuntimeError("replacement failed")
        rollback_started.set()
        await release_failure.wait()
        raise RuntimeError("rollback failed")

    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: next(identities),
        ),
        provision=lambda _identity: asyncio.sleep(0, result=1),
        reprovision=reprovision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        rollback_timeout_seconds=0.01,
    )
    initial = await director.start()

    with pytest.raises(ExceptionGroup, match="rebind and rollback failed"):
        await director.rebind(participant_identity=initial.participant_identity)
    await rollback_started.wait()
    finalizer = director._rebind_finalizer
    assert finalizer is not None
    release_failure.set()
    await finalizer

    with pytest.raises(RuntimeError, match="failed closed"):
        await director.start()


@pytest.mark.asyncio
async def test_session_start_callback_failure_rolls_back_worker_without_public_events() -> None:
    projection = BrowserEventProjection()
    stopped: list[tuple[str, int]] = []
    callback_error = RuntimeError("usage reset failed")
    callback_attempts = 0
    callback_active_state: list[tuple[str | None, int | None]] = []

    async def provision(_identity: str) -> int:
        return 7

    async def stop(identity: str, generation: int) -> None:
        stopped.append((identity, generation))

    def on_session_started(_identity: str, _generation: int) -> None:
        nonlocal callback_attempts
        callback_active_state.append((director.active_identity, director.active_generation))
        callback_attempts += 1
        if callback_attempts == 1:
            raise callback_error

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=stop,
        approval=_noop_approval,
        projection=projection,
        on_session_started=on_session_started,
    )

    with pytest.raises(RuntimeError) as raised:
        await director.start()

    assert raised.value is callback_error
    assert stopped == [("browser_0123456789abcdef", 7)]
    assert director.active_identity is None
    assert director.active_generation is None
    assert projection.events_after(0) == ()

    credential = await director.start()

    assert credential.participant_identity == "browser_0123456789abcdef"
    assert callback_active_state == [(None, None), (None, None)]
    assert director.active_generation == 7
    assert [event.kind for event in projection.events_after(0)] == ["session_ready"]


@pytest.mark.asyncio
async def test_session_start_callback_failure_reports_rollback_failure_and_cleans_state() -> None:
    projection = BrowserEventProjection()
    callback_error = RuntimeError("usage reset failed")
    rollback_error = RuntimeError("worker stop failed")

    async def provision(_identity: str) -> int:
        return 8

    async def stop(_identity: str, _generation: int) -> None:
        raise rollback_error

    def on_session_started(_identity: str, _generation: int) -> None:
        raise callback_error

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=stop,
        approval=_noop_approval,
        projection=projection,
        on_session_started=on_session_started,
    )

    with pytest.raises(ExceptionGroup) as raised:
        await director.start()

    assert raised.value.exceptions == (callback_error, rollback_error)
    assert director.active_identity is None
    assert director.active_generation is None
    assert projection.events_after(0) == ()


@pytest.mark.asyncio
async def test_session_start_callback_failure_cleans_state_when_rollback_is_cancelled() -> None:
    projection = BrowserEventProjection()
    callback_error = RuntimeError("usage reset failed")
    rollback_error = asyncio.CancelledError("worker stop cancelled")

    async def provision(_identity: str) -> int:
        return 9

    async def stop(_identity: str, _generation: int) -> None:
        raise rollback_error

    def on_session_started(_identity: str, _generation: int) -> None:
        raise callback_error

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=stop,
        approval=_noop_approval,
        projection=projection,
        on_session_started=on_session_started,
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await director.start()

    assert raised.value.exceptions == (callback_error, rollback_error)
    assert director.active_identity is None
    assert director.active_generation is None
    assert projection.events_after(0) == ()


@pytest.mark.asyncio
async def test_session_start_cancellation_waits_for_callback_rollback_settlement() -> None:
    projection = BrowserEventProjection()
    callback_error = RuntimeError("usage reset failed")
    rollback_started = asyncio.Event()
    release_rollback = asyncio.Event()
    rollback_completed = asyncio.Event()

    async def provision(_identity: str) -> int:
        return 9

    async def stop(_identity: str, _generation: int) -> None:
        rollback_started.set()
        await release_rollback.wait()
        rollback_completed.set()

    def on_session_started(_identity: str, _generation: int) -> None:
        raise callback_error

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=stop,
        approval=_noop_approval,
        projection=projection,
        on_session_started=on_session_started,
    )

    operation = asyncio.create_task(director.start())
    await rollback_started.wait()
    operation.cancel()
    await asyncio.sleep(0)
    assert operation.done() is False

    release_rollback.set()
    with pytest.raises(BaseExceptionGroup) as raised:
        await operation

    assert raised.value.exceptions[0] is callback_error
    assert isinstance(raised.value.exceptions[1], asyncio.CancelledError)
    assert rollback_completed.is_set()
    assert director.active_identity is None
    assert director.active_generation is None
    assert projection.events_after(0) == ()


@pytest.mark.asyncio
async def test_same_origin_bootstrap_exchanges_capability_without_exposing_secret() -> None:
    secret = "synthetic-browser-bootstrap-secret-32-bytes"
    provisioned: list[str] = []

    async def provision(identity: str) -> int:
        provisioned.append(identity)
        return 1

    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=LiveKitConnection("wss://livekit.test", "test-key", secret),
                room_name="hermes-local",
                identity_factory=lambda: "browser_0123456789abcdef",
            ),
            provision=provision,
            submit=_noop_submit,
            stop=_noop_stop,
            approval=_noop_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(
            connection=LiveKitConnection("wss://livekit.test", "test-key", secret),
            room_name="hermes-local",
        ),
        capability=OneTimeBootstrapCapability(
            token_factory=lambda: "b" * 43,
            clock=lambda: 100.0,
        ),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )

    response = await app.handle(
        method="POST",
        path="/api/v1/bootstrap",
        headers={
            "authorization": f"Bearer {'b' * 43}",
            "origin": "https://phone.test:8443",
            "content-length": "0",
        },
        body=b"",
    )
    payload = json.loads(response.body)

    assert response.status == 200
    assert provisioned == ["browser_0123456789abcdef"]
    assert response.headers == {
        "cache-control": "no-store",
        "content-type": "application/json; charset=utf-8",
        "referrer-policy": "no-referrer",
        "x-content-type-options": "nosniff",
    }
    assert payload["url"] == "wss://livekit.test"
    assert payload["roomName"] == "hermes-local"
    assert payload["participantIdentity"] == "browser_0123456789abcdef"
    assert payload["workerIdentity"] == "worker_hermes_browser"
    assert payload["expiresInSeconds"] == 60
    assert type(payload["token"]) is str
    assert secret not in response.body.decode("utf-8")


@pytest.mark.asyncio
async def test_authenticated_rebind_rotates_browser_join_credential() -> None:
    secret = "synthetic-browser-bootstrap-secret-32-bytes"
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    rebound: list[str] = []

    async def provision(_identity: str) -> int:
        return 1

    async def reprovision(identity: str) -> int:
        rebound.append(identity)
        return 2

    connection = LiveKitConnection("wss://livekit.test", "test-key", secret)
    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=connection,
                room_name="hermes-local",
                identity_factory=lambda: next(identities),
            ),
            provision=provision,
            reprovision=reprovision,
            submit=_noop_submit,
            stop=_noop_stop,
            approval=_noop_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "r" * 43),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )
    initial_response = await app.handle(
        method="POST",
        path="/api/v1/bootstrap",
        headers={
            "authorization": f"Bearer {'r' * 43}",
            "content-length": "0",
            "origin": "https://phone.test:8443",
        },
        body=b"",
    )
    initial = json.loads(initial_response.body)
    body = b'{"requestId":"rebind_0123456789abcdef"}'

    response = await app.handle(
        method="POST",
        path="/api/v1/rebind",
        headers={
            "authorization": f"Bearer {initial['token']}",
            "content-length": str(len(body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=body,
    )
    replacement = json.loads(response.body)
    replay_responses = [
        await app.handle(
            method="POST",
            path="/api/v1/rebind",
            headers={
                "authorization": f"Bearer {initial['token']}",
                "content-length": str(len(body)),
                "content-type": "application/json",
                "origin": "https://phone.test:8443",
            },
            body=body,
        )
        for _ in range(2)
    ]

    assert response.status == 200
    assert rebound == ["browser_fedcba9876543210"]
    assert replacement["participantIdentity"] == "browser_fedcba9876543210"
    assert replacement["participantIdentity"] != initial["participantIdentity"]
    assert replacement["workerIdentity"] == initial["workerIdentity"]
    assert replacement["roomName"] == initial["roomName"]
    assert replacement["url"] == initial["url"]
    assert all(replay.status == 200 for replay in replay_responses)
    assert all(replay.body == response.body for replay in replay_responses)

    stop_response = await app.handle(
        method="POST",
        path="/api/v1/stop",
        headers={
            "authorization": f"Bearer {initial['token']}",
            "content-length": str(len(body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=body,
    )

    assert stop_response.status == 200


@pytest.mark.asyncio
async def test_authenticated_projection_resync_rotates_once() -> None:
    """A projection gap gets one server-derived recovery, not a polling retry loop."""

    from hermes_realtime.evidence.models import BindingCloseReason

    secret = "synthetic-browser-bootstrap-secret-32-bytes"
    identities = iter(
        (
            "browser_0123456789abcdef",
            "browser_fedcba9876543210",
        )
    )
    invalidations: list[BindingCloseReason] = []
    rebound: list[str] = []

    async def provision(_identity: str) -> int:
        return 1

    async def reprovision(identity: str) -> int:
        rebound.append(identity)
        return 2

    async def invalidate(reason: BindingCloseReason) -> None:
        invalidations.append(reason)

    connection = LiveKitConnection("wss://livekit.test", "test-key", secret)
    projection = BrowserEventProjection()
    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=connection,
                room_name="hermes-local",
                identity_factory=lambda: next(identities),
            ),
            provision=provision,
            reprovision=reprovision,
            submit=_noop_submit,
            stop=_noop_stop,
            approval=_noop_approval,
            projection=projection,
            evidence_invalidate=invalidate,
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "p" * 43),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )
    initial = json.loads(
        (
            await app.handle(
                method="POST",
                path="/api/v1/bootstrap",
                headers={
                    "authorization": f"Bearer {'p' * 43}",
                    "content-length": "0",
                    "origin": "https://phone.test:8443",
                },
                body=b"",
            )
        ).body
    )
    projection.publish("session_stopped", {})

    response = await app.handle(
        method="POST",
        path="/api/v1/projection-resync",
        headers={
            "authorization": f"Bearer {initial['token']}",
            "content-length": "0",
            "origin": "https://phone.test:8443",
        },
        body=b"",
    )
    replacement = json.loads(response.body)

    assert response.status == 200
    assert replacement["participantIdentity"] == "browser_fedcba9876543210"
    assert replacement["participantIdentity"] != initial["participantIdentity"]
    assert rebound == ["browser_fedcba9876543210"]
    assert invalidations == [BindingCloseReason.PROJECTION_RESYNC]
    events = projection.events_after(0)
    assert [(event.sequence, event.kind) for event in events] == [(1, "session_ready")]
    assert all(event.kind != "session_stopped" for event in events)

    with pytest.raises(RuntimeError, match="projection resync was already attempted"):
        await app.handle(
            method="POST",
            path="/api/v1/projection-resync",
            headers={
                "authorization": f"Bearer {replacement['token']}",
                "content-length": "0",
                "origin": "https://phone.test:8443",
            },
            body=b"",
        )


@pytest.mark.asyncio
async def test_tailnet_rebind_needs_no_expiring_browser_bearer_and_is_replay_safe() -> None:
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    rebound: list[str] = []

    async def provision(_identity: str) -> int:
        return 1

    async def reprovision(identity: str) -> int:
        rebound.append(identity)
        return 2

    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        return b'{"Node":{"StableID":"allowed-node"}}'

    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-browser-bootstrap-secret-32-bytes",
    )
    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=connection,
                room_name="hermes-local",
                identity_factory=lambda: next(identities),
            ),
            provision=provision,
            reprovision=reprovision,
            submit=_noop_submit,
            stop=_noop_stop,
            approval=_noop_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=None,
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
        tailnet_authorizer=TailnetPeerAuthorizer(
            allowed_stable_id="allowed-node",
            resolver=resolve,
        ),
    )
    peer = TailnetPeerAddress.from_peername(("100.64.9.9", 54321))
    common_headers = {
        "origin": "https://phone.test:8443",
        "sec-fetch-site": "same-origin",
    }
    initial_response = await app.handle(
        method="POST",
        path="/api/v1/stable-bootstrap",
        headers={**common_headers, "content-length": "0"},
        body=b"",
        peer=peer,
    )
    initial = json.loads(initial_response.body)
    body = json.dumps(
        {
            "participantIdentity": initial["participantIdentity"],
            "requestId": "rebind_0123456789abcdef",
        },
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {
        **common_headers,
        "content-length": str(len(body)),
        "content-type": "application/json",
    }

    first = await app.handle(
        method="POST",
        path="/api/v1/stable-rebind",
        headers=headers,
        body=body,
        peer=peer,
    )
    replay = await app.handle(
        method="POST",
        path="/api/v1/tailnet-rebind",
        headers=headers,
        body=body,
        peer=peer,
    )
    repeated_replay = await app.handle(
        method="POST",
        path="/api/v1/tailnet-rebind",
        headers=headers,
        body=body,
        peer=peer,
    )
    first_payload = json.loads(first.body)
    replay_payload = json.loads(replay.body)

    assert first.status == replay.status == repeated_replay.status == 200
    assert rebound == ["browser_fedcba9876543210"]
    assert first_payload["participantIdentity"] == "browser_fedcba9876543210"
    assert replay_payload["participantIdentity"] == first_payload["participantIdentity"]
    assert replay.body == repeated_replay.body == first.body


@pytest.mark.asyncio
async def test_loopback_stable_rebind_enforces_peer_browser_and_session_authority() -> None:
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    rebound: list[str] = []

    async def provision(_identity: str) -> int:
        return 1

    async def reprovision(identity: str) -> int:
        rebound.append(identity)
        return 2

    connection = LiveKitConnection("wss://livekit.test", "key", "s" * 32)
    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=connection,
                room_name="room",
                identity_factory=lambda: next(identities),
            ),
            provision=provision,
            reprovision=reprovision,
            submit=_noop_submit,
            stop=_noop_stop,
            approval=_noop_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="room"),
        capability=None,
        allowed_origin="http://127.0.0.1:8765",
        worker_identity="worker_hermes_browser",
        loopback_authorizer=LoopbackPeerAuthorizer(),
    )
    peer = LoopbackPeerAddress.from_peername(("127.0.0.1", 54321))
    common_headers = {
        "origin": "http://127.0.0.1:8765",
        "sec-fetch-site": "same-origin",
    }
    initial_response = await app.handle(
        method="POST",
        path="/api/v1/stable-bootstrap",
        headers={**common_headers, "content-length": "0"},
        body=b"",
        peer=peer,
    )
    initial = json.loads(initial_response.body)

    def rebind_body(participant_identity: str) -> bytes:
        return json.dumps(
            {
                "participantIdentity": participant_identity,
                "requestId": "rebind_0123456789abcdef",
            },
            separators=(",", ":"),
        ).encode("utf-8")

    body = rebind_body(initial["participantIdentity"])
    headers = {
        **common_headers,
        "content-length": str(len(body)),
        "content-type": "application/json",
    }
    wrong_identity_body = rebind_body("browser_wrong_identity")
    for rejected_headers, rejected_body, rejected_peer in (
        ({key: value for key, value in headers.items() if key != "sec-fetch-site"}, body, peer),
        ({**headers, "authorization": "Bearer forbidden"}, body, peer),
        (headers, body, None),
        (
            {**headers, "content-length": str(len(wrong_identity_body))},
            wrong_identity_body,
            peer,
        ),
    ):
        with pytest.raises(PermissionError):
            await app.handle(
                method="POST",
                path="/api/v1/stable-rebind",
                headers=rejected_headers,
                body=rejected_body,
                peer=rejected_peer,
            )

    first = await app.handle(
        method="POST",
        path="/api/v1/stable-rebind",
        headers=headers,
        body=body,
        peer=peer,
    )
    replay = await app.handle(
        method="POST",
        path="/api/v1/stable-rebind",
        headers=headers,
        body=body,
        peer=peer,
    )

    assert first.status == replay.status == 200
    assert first.body == replay.body
    assert json.loads(first.body)["participantIdentity"] == "browser_fedcba9876543210"
    assert rebound == ["browser_fedcba9876543210"]


@pytest.mark.asyncio
async def test_stable_tailnet_bootstrap_uses_peer_without_bearer() -> None:
    provisioned: list[str] = []
    whois_payload = [b'{"Node":{"StableID":"other-node"}}']

    async def provision(identity: str) -> int:
        provisioned.append(identity)
        return 1

    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        return whois_payload[0]

    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-browser-bootstrap-secret-32-bytes",
    )
    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=connection,
                room_name="hermes-local",
                identity_factory=lambda: "browser_0123456789abcdef",
            ),
            provision=provision,
            submit=_noop_submit,
            stop=_noop_stop,
            approval=_noop_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=None,
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
        tailnet_authorizer=TailnetPeerAuthorizer(
            allowed_stable_id="allowed-node",
            resolver=resolve,
        ),
    )
    headers = {
        "content-length": "0",
        "origin": "https://phone.test:8443",
        "sec-fetch-site": "same-origin",
    }

    with pytest.raises(PermissionError):
        await app.handle(
            method="POST",
            path="/api/v1/tailnet-bootstrap",
            headers=headers,
            body=b"",
            peer=None,
        )
    with pytest.raises(PermissionError):
        await app.handle(
            method="POST",
            path="/api/v1/tailnet-bootstrap",
            headers=headers,
            body=b"",
            peer=TailnetPeerAddress.from_peername(("100.64.1.2", 54321)),
        )
    assert provisioned == []
    whois_payload[0] = b'{"Node":{"StableID":"allowed-node"}}'

    response = await app.handle(
        method="POST",
        path="/api/v1/stable-bootstrap",
        headers=headers,
        body=b"",
        peer=TailnetPeerAddress.from_peername(("100.64.1.2", 54321)),
    )

    assert response.status == 200
    assert provisioned == ["browser_0123456789abcdef"]
    assert "allowed-node" not in response.body.decode()
    with pytest.raises(PermissionError):
        await app.handle(
            method="POST",
            path="/api/v1/bootstrap",
            headers={
                "authorization": f"Bearer {'b' * 43}",
                "content-length": "0",
                "origin": "https://phone.test:8443",
            },
            body=b"",
        )


@pytest.mark.asyncio
async def test_concurrent_stable_bootstrap_has_exactly_one_session_winner() -> None:
    provisioned: list[str] = []

    async def provision(identity: str) -> int:
        provisioned.append(identity)
        await asyncio.sleep(0)
        return 1

    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        await asyncio.sleep(0)
        return b'{"Node":{"StableID":"allowed-node"}}'

    connection = LiveKitConnection("wss://livekit.test", "key", "s" * 32)
    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(connection=connection, room_name="room"),
            provision=provision,
            submit=_noop_submit,
            stop=_noop_stop,
            approval=_noop_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="room"),
        capability=None,
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
        tailnet_authorizer=TailnetPeerAuthorizer(
            allowed_stable_id="allowed-node", resolver=resolve
        ),
    )
    headers = {
        "content-length": "0",
        "origin": "https://phone.test:8443",
        "sec-fetch-site": "same-origin",
    }
    peers = (
        TailnetPeerAddress.from_peername(("100.64.1.2", 54321)),
        TailnetPeerAddress.from_peername(("100.64.1.2", 54322)),
    )

    outcomes = await asyncio.gather(
        *(
            app.handle(
                method="POST",
                path="/api/v1/tailnet-bootstrap",
                headers=headers,
                body=b"",
                peer=peer,
            )
            for peer in peers
        ),
        return_exceptions=True,
    )

    assert sum(isinstance(outcome, BrowserBootstrapResponse) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, RuntimeError) for outcome in outcomes) == 1
    assert len(provisioned) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers,body",
    [
        (
            {
                "content-length": "0",
                "origin": "https://phone.test:8443",
            },
            b"",
        ),
        (
            {
                "authorization": "Bearer forbidden",
                "content-length": "0",
                "origin": "https://phone.test:8443",
                "sec-fetch-site": "same-origin",
            },
            b"",
        ),
        (
            {
                "content-length": "1",
                "origin": "https://phone.test:8443",
                "sec-fetch-site": "same-origin",
            },
            b"x",
        ),
    ],
)
async def test_invalid_stable_bootstrap_provisions_nothing(
    headers: dict[str, str], body: bytes
) -> None:
    provisioned: list[str] = []

    async def provision(identity: str) -> int:
        provisioned.append(identity)
        return 1

    async def resolve(_peer: TailnetPeerAddress) -> bytes:
        return b'{"Node":{"StableID":"allowed-node"}}'

    connection = LiveKitConnection("wss://livekit.test", "key", "s" * 32)
    app = BrowserBootstrapApplication(
        sessions=BrowserSessionDirector(
            issuer=BrowserTokenIssuer(connection=connection, room_name="room"),
            provision=provision,
            submit=_noop_submit,
            stop=_noop_stop,
            approval=_noop_approval,
            projection=BrowserEventProjection(),
        ),
        verifier=BrowserTokenVerifier(connection=connection, room_name="room"),
        capability=None,
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
        tailnet_authorizer=TailnetPeerAuthorizer(
            allowed_stable_id="allowed-node", resolver=resolve
        ),
    )
    with pytest.raises((PermissionError, ValueError)):
        await app.handle(
            method="POST",
            path="/api/v1/tailnet-bootstrap",
            headers=headers,
            body=body,
            peer=TailnetPeerAddress.from_peername(("100.64.1.2", 54321)),
        )
    assert provisioned == []


@pytest.mark.asyncio
async def test_authenticated_input_submits_server_bound_sequenced_text() -> None:
    secret = "synthetic-browser-bootstrap-secret-32-bytes"
    connection = LiveKitConnection("wss://livekit.test", "test-key", secret)
    submitted: list[tuple[str, int, str]] = []
    stopped: list[tuple[str, int]] = []
    approvals: list[tuple[str, int, int, str, str]] = []
    media_activations: list[tuple[str, int, int]] = []
    audio_diagnostics: list[tuple[str, int, BrowserAudioDiagnostic]] = []
    yield_claims: list[tuple[str, int, str, int, str, str]] = []

    async def provision(identity: str) -> int:
        return 3

    async def submit(identity: str, generation: int, sequence: int, text: str) -> None:
        del sequence
        submitted.append((identity, generation, text))

    async def stop(identity: str, generation: int) -> None:
        stopped.append((identity, generation))

    async def activate_media(identity: str, generation: int, incarnation: int) -> None:
        media_activations.append((identity, generation, incarnation))

    async def yield_speech(
        identity: str,
        generation: int,
        turn_id: str,
        turn_generation: int,
        chunk_id: str,
        stream_id: str,
    ) -> bool:
        yield_claims.append((identity, generation, turn_id, turn_generation, chunk_id, stream_id))
        return True

    async def observe_audio_diagnostic(
        identity: str, generation: int, diagnostic: BrowserAudioDiagnostic
    ) -> None:
        await director.public_events_after(participant_identity=identity, sequence=0)
        audio_diagnostics.append((identity, generation, diagnostic))

    async def approval(
        identity: str,
        generation: int,
        sequence: int,
        approval_id: str,
        decision: str,
    ) -> None:
        approvals.append((identity, generation, sequence, approval_id, decision))

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection,
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=submit,
        stop=stop,
        approval=approval,
        activate_media=activate_media,
        observe_audio_diagnostic=observe_audio_diagnostic,
        yield_speech=yield_speech,
        conversation_profile="natural_v1",
        projection=BrowserEventProjection(),
    )
    credential = await director.start()
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "c" * 43),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )
    media_body = b'{"mediaIncarnation":4}'
    media_response = await app.handle(
        method="POST",
        path="/api/v1/media",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(media_body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=media_body,
    )
    assert media_response.status == 202
    assert json.loads(media_response.body) == {"mediaIncarnation": 4, "version": 1}
    assert media_activations == [("browser_0123456789abcdef", 3, 4)]

    incomplete_yield_body = (
        b'{"turnId":"turn_001","chunkId":"chunk_001","streamId":"hermes-speech-01234567"}'
    )
    with pytest.raises(ValueError, match="turnGeneration"):
        await app.handle(
            method="POST",
            path="/api/v1/yield",
            headers={
                "authorization": f"Bearer {credential.token}",
                "content-length": str(len(incomplete_yield_body)),
                "content-type": "application/json",
                "origin": "https://phone.test:8443",
            },
            body=incomplete_yield_body,
        )
    assert yield_claims == []

    yield_body = (
        b'{"turnId":"turn_001","turnGeneration":7,"chunkId":"chunk_001",'
        b'"streamId":"hermes-speech-01234567"}'
    )
    yield_response = await app.handle(
        method="POST",
        path="/api/v1/yield",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(yield_body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=yield_body,
    )
    assert yield_response.status == 202
    assert json.loads(yield_response.body) == {"matched": True, "version": 1}
    assert yield_claims == [
        (
            "browser_0123456789abcdef",
            3,
            "turn_001",
            7,
            "chunk_001",
            "hermes-speech-01234567",
        )
    ]

    diagnostic_body = (
        b'{"version":1,"streamId":"hermes-speech-01234567",'
        b'"supported":{"echoCancellation":true,"autoGainControl":false,'
        b'"noiseSuppression":true,"voiceIsolation":false},'
        b'"applied":{"echoCancellation":"enabled","autoGainControl":"unknown",'
        b'"noiseSuppression":"disabled","voiceIsolation":"unknown"},'
        b'"render":{"subscribedToAttachMs":4,"attachToPlayingMs":36,'
        b'"playingToAdvanceMs":15}}'
    )
    diagnostic_response = await app.handle(
        method="POST",
        path="/api/v1/audio-diagnostic",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(diagnostic_body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=diagnostic_body,
    )
    assert diagnostic_response.status == 202
    assert json.loads(diagnostic_response.body) == {
        "streamId": "hermes-speech-01234567",
        "version": 1,
    }
    assert len(audio_diagnostics) == 1
    assert audio_diagnostics[0][0:2] == ("browser_0123456789abcdef", 3)
    assert audio_diagnostics[0][2].stream_id == "hermes-speech-01234567"

    replay_response = await app.handle(
        method="POST",
        path="/api/v1/audio-diagnostic",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(diagnostic_body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=diagnostic_body,
    )
    assert replay_response.status == 202
    assert len(audio_diagnostics) == 1

    changed_diagnostic = diagnostic_body.replace(
        b'"playingToAdvanceMs":15', b'"playingToAdvanceMs":16'
    )
    with pytest.raises(RuntimeError, match="different data"):
        await app.handle(
            method="POST",
            path="/api/v1/audio-diagnostic",
            headers={
                "authorization": f"Bearer {credential.token}",
                "content-length": str(len(changed_diagnostic)),
                "content-type": "application/json",
                "origin": "https://phone.test:8443",
            },
            body=changed_diagnostic,
        )
    assert len(audio_diagnostics) == 1

    body = b'{"sequence":1,"text":"typed request"}'

    response = await app.handle(
        method="POST",
        path="/api/v1/input",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=body,
    )

    assert response.status == 202
    assert json.loads(response.body) == {"sequence": 1, "version": 1}
    assert submitted == [("browser_0123456789abcdef", 3, "typed request")]

    duplicate_body = b'{"sequence":2,"sequence":3,"text":"duplicate"}'
    with pytest.raises(ValueError, match="duplicate"):
        await app.handle(
            method="POST",
            path="/api/v1/input",
            headers={
                "authorization": f"Bearer {credential.token}",
                "content-length": str(len(duplicate_body)),
                "content-type": "application/json",
                "origin": "https://phone.test:8443",
            },
            body=duplicate_body,
        )
    assert len(submitted) == 1

    events_body = b'{"after":0}'
    events_response = await app.handle(
        method="POST",
        path="/api/v1/events",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(events_body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=events_body,
    )
    assert [item["kind"] for item in json.loads(events_response.body)["events"]] == [
        "session_ready",
        "typed_input_admitted",
    ]

    approval_body = b'{"approvalId":"approval_0123456789abcdef","decision":"approve","sequence":1}'
    approval_response = await app.handle(
        method="POST",
        path="/api/v1/approval",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": str(len(approval_body)),
            "content-type": "application/json",
            "origin": "https://phone.test:8443",
        },
        body=approval_body,
    )
    assert approval_response.status == 202
    assert approvals == [
        (
            "browser_0123456789abcdef",
            3,
            1,
            "approval_0123456789abcdef",
            "approve",
        )
    ]

    refresh_response = await app.handle(
        method="POST",
        path="/api/v1/refresh",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": "0",
            "origin": "https://phone.test:8443",
        },
        body=b"",
    )
    refresh_payload = json.loads(refresh_response.body)
    assert refresh_payload["participantIdentity"] == "browser_0123456789abcdef"
    assert refresh_payload["expiresInSeconds"] == credential.expires_in_seconds

    stop_response = await app.handle(
        method="POST",
        path="/api/v1/stop",
        headers={
            "authorization": f"Bearer {credential.token}",
            "content-length": "0",
            "origin": "https://phone.test:8443",
        },
        body=b"",
    )

    assert stop_response.status == 200
    assert json.loads(stop_response.body) == {"stopped": True, "version": 1}
    assert stopped == [("browser_0123456789abcdef", 3)]


@pytest.mark.asyncio
async def test_authenticated_codex_model_catalog_can_select_next_turn_configuration() -> None:
    secret = "synthetic-browser-model-secret-32-bytes"
    connection = LiveKitConnection("wss://livekit.test", "test-key", secret)
    selected = ["gpt-5.6-terra", "low"]
    models = (
        BrowserSelectableModel(
            model="gpt-5.6-terra",
            display_name="GPT-5.6 Terra",
            description="Fast coding model",
            supported_efforts=("low", "medium"),
            default_effort="medium",
        ),
        BrowserSelectableModel(
            model="gpt-5.6-sol",
            display_name="GPT-5.6 Sol",
            description="Deep coding model",
            supported_efforts=("medium", "high", "xhigh", "ultra"),
            default_effort="high",
        ),
    )

    async def catalog() -> BrowserModelCatalog:
        return BrowserModelCatalog(models, selected[0], selected[1])

    async def select_model(model: str, effort: str) -> BrowserModelCatalog:
        selected[:] = [model, effort]
        return BrowserModelCatalog(models, model, effort)

    projection = BrowserEventProjection()
    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection,
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=lambda _identity: asyncio.sleep(0, result=3),
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=projection,
        model_configuration=BrowserModelConfiguration(
            authentication="subscription",
            provider="openai-codex",
            transport="subscription-app-server",
            model="gpt-5.6-terra",
            effort="low",
            context_window_tokens=None,
            reports_token_usage=True,
        ),
        model_catalog=catalog,
        select_model=select_model,
    )
    credential = await director.start()
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "m" * 43),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )
    headers = {
        "authorization": f"Bearer {credential.token}",
        "content-length": "0",
        "origin": "https://phone.test:8443",
    }

    catalog_response = await app.handle(
        method="POST", path="/api/v1/models", headers=headers, body=b""
    )
    catalog_payload = json.loads(catalog_response.body)

    assert catalog_response.status == 200
    assert catalog_payload["selectedModel"] == "gpt-5.6-terra"
    assert catalog_payload["selectedEffort"] == "low"
    assert catalog_payload["models"][1] == {
        "defaultEffort": "high",
        "description": "Deep coding model",
        "displayName": "GPT-5.6 Sol",
        "model": "gpt-5.6-sol",
        "supportedEfforts": ["medium", "high", "xhigh", "ultra"],
    }

    body = b'{"effort":"ultra","model":"gpt-5.6-sol"}'
    selection_response = await app.handle(
        method="POST",
        path="/api/v1/model",
        headers={**headers, "content-length": str(len(body)), "content-type": "application/json"},
        body=body,
    )
    selection_payload = json.loads(selection_response.body)

    assert selection_response.status == 200
    assert selection_payload["selectedModel"] == "gpt-5.6-sol"
    assert selection_payload["selectedEffort"] == "ultra"
    assert selected == ["gpt-5.6-sol", "ultra"]
    assert projection.events_after(0)[-1].data["model"] == "gpt-5.6-sol"
    assert projection.events_after(0)[-1].data["effort"] == "ultra"


@pytest.mark.asyncio
async def test_blocked_model_selection_does_not_block_stop_or_publish_stale_result() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    provider_applied: list[tuple[str, str]] = []
    projection = BrowserEventProjection()
    models = (
        BrowserSelectableModel(
            model="gpt-5.6-terra",
            display_name="GPT-5.6 Terra",
            description="Fast coding model",
            supported_efforts=("low", "medium"),
            default_effort="low",
        ),
    )

    async def select_model(model: str, effort: str) -> BrowserModelCatalog:
        entered.set()
        await release.wait()
        provider_applied.append((model, effort))
        return BrowserModelCatalog(models, model, effort)

    async def catalog() -> BrowserModelCatalog:
        return BrowserModelCatalog(models, "gpt-5.6-terra", "low")

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test", "test-key", "synthetic-model-race-secret-32-bytes"
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=lambda _identity: asyncio.sleep(0, result=9),
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=projection,
        model_configuration=BrowserModelConfiguration(
            authentication="subscription",
            provider="openai-codex",
            transport="subscription-app-server",
            model="gpt-5.6-terra",
            effort="low",
            context_window_tokens=None,
            reports_token_usage=True,
        ),
        model_catalog=catalog,
        select_model=select_model,
    )
    credential = await director.start()
    selection = asyncio.create_task(
        director.change_model(
            participant_identity=credential.participant_identity,
            model="gpt-5.6-terra",
            effort="medium",
        )
    )
    await entered.wait()

    await asyncio.wait_for(director.stop(participant_identity=credential.participant_identity), 0.1)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            await asyncio.wait_for(asyncio.shield(selection), 0.1)
    finally:
        release.set()
        if not selection.done():
            with pytest.raises(RuntimeError, match="stale"):
                await selection

    assert provider_applied == []
    assert director.active_identity is None
    assert [event.kind for event in projection.events_after(0)] == [
        "session_ready",
        "session_model",
        "session_stopped",
    ]


@pytest.mark.asyncio
async def test_rebind_waits_for_admitted_model_selection_before_rotating_authority() -> None:
    selection_entered = asyncio.Event()
    release_selection = asyncio.Event()
    reprovision_entered = asyncio.Event()
    order: list[str] = []
    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    models = (
        BrowserSelectableModel(
            model="gpt-5.6-terra",
            display_name="GPT-5.6 Terra",
            description="Fast coding model",
            supported_efforts=("low", "medium"),
            default_effort="low",
        ),
    )

    async def select_model(model: str, effort: str) -> BrowserModelCatalog:
        selection_entered.set()
        await release_selection.wait()
        order.append("model")
        return BrowserModelCatalog(models, model, effort)

    async def reprovision(_identity: str) -> int:
        order.append("rebind")
        reprovision_entered.set()
        return 10

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test", "test-key", "synthetic-model-race-secret-32-bytes"
            ),
            room_name="hermes-local",
            identity_factory=lambda: next(identities),
        ),
        provision=lambda _identity: asyncio.sleep(0, result=9),
        reprovision=reprovision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        model_configuration=BrowserModelConfiguration(
            authentication="subscription",
            provider="openai-codex",
            transport="subscription-app-server",
            model="gpt-5.6-terra",
            effort="low",
            context_window_tokens=None,
            reports_token_usage=True,
        ),
        model_catalog=lambda: asyncio.sleep(
            0,
            result=BrowserModelCatalog(models, "gpt-5.6-terra", "low"),
        ),
        select_model=select_model,
    )
    credential = await director.start()
    selection = asyncio.create_task(
        director.change_model(
            participant_identity=credential.participant_identity,
            model="gpt-5.6-terra",
            effort="medium",
        )
    )
    await selection_entered.wait()
    rebind = asyncio.create_task(
        director.rebind(
            participant_identity=credential.participant_identity,
            request_id="rebind_0123456789abcdef",
        )
    )

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(reprovision_entered.wait(), timeout=0.01)
    release_selection.set()
    await selection
    replacement = await rebind

    assert order == ["model", "rebind"]
    assert replacement.participant_identity == "browser_fedcba9876543210"


def test_concurrent_bootstrap_capability_claim_has_one_winner() -> None:
    contenders = threading.Barrier(2)
    clock_calls = 0
    clock_lock = threading.Lock()

    def synchronized_clock() -> float:
        nonlocal clock_calls
        with clock_lock:
            clock_calls += 1
            call = clock_calls
        if call > 1:
            contenders.wait(timeout=2)
        return 100.0

    capability = OneTimeBootstrapCapability(
        token_factory=lambda: "c" * 43,
        clock=synchronized_clock,
    )

    def claim() -> str:
        try:
            capability.consume("c" * 43)
        except RuntimeError:
            return "rejected"
        return "accepted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(claim), pool.submit(claim)]
        outcomes = sorted(future.result() for future in futures)

    assert outcomes == ["accepted", "rejected"]


@pytest.mark.asyncio
async def test_session_director_provisions_exact_identity_before_returning_token() -> None:
    events: list[str] = []

    def identity() -> str:
        events.append("identity")
        return "browser_0123456789abcdef"

    async def provision(participant_identity: str) -> int:
        events.append(f"provision:{participant_identity}")
        return 7

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=identity,
        ),
        provision=provision,
        submit=_noop_submit,
        stop=_noop_stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
    )

    credential = await director.start()
    events.append(f"returned:{credential.participant_identity}")

    assert director.active_generation == 7
    assert events == [
        "identity",
        "provision:browser_0123456789abcdef",
        "returned:browser_0123456789abcdef",
    ]


@pytest.mark.asyncio
async def test_session_director_owns_typed_input_and_stop_authority() -> None:
    calls: list[tuple[object, ...]] = []

    async def provision(identity: str) -> int:
        return 9

    async def submit(identity: str, generation: int, sequence: int, text: str) -> None:
        calls.append(("submit", identity, generation, sequence, text))

    async def stop(identity: str, generation: int) -> None:
        calls.append(("stop", identity, generation))

    async def approval(
        identity: str,
        generation: int,
        sequence: int,
        approval_id: str,
        decision: str,
    ) -> None:
        calls.append(("approval", identity, generation, sequence, approval_id, decision))

    projection = BrowserEventProjection(clock=lambda: 5.0)
    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-bootstrap-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=submit,
        stop=stop,
        approval=approval,
        projection=projection,
    )
    await director.start()

    await director.submit_text(
        participant_identity="browser_0123456789abcdef",
        sequence=1,
        text="typed request",
    )
    await director.submit_text(
        participant_identity="browser_0123456789abcdef",
        sequence=1,
        text="typed request",
    )
    with pytest.raises(RuntimeError, match="sequence"):
        await director.submit_text(
            participant_identity="browser_0123456789abcdef",
            sequence=1,
            text="replay",
        )
    await director.decide_approval(
        participant_identity="browser_0123456789abcdef",
        approval_id="approval_0123456789abcdef",
        sequence=1,
        decision="approve",
    )
    await director.decide_approval(
        participant_identity="browser_0123456789abcdef",
        approval_id="approval_0123456789abcdef",
        sequence=1,
        decision="approve",
    )
    with pytest.raises(RuntimeError, match="sequence"):
        await director.decide_approval(
            participant_identity="browser_0123456789abcdef",
            approval_id="approval_0123456789abcdef",
            sequence=1,
            decision="reject",
        )
    await director.stop(participant_identity="browser_0123456789abcdef")

    assert calls == [
        ("submit", "browser_0123456789abcdef", 9, 1, "typed request"),
        (
            "approval",
            "browser_0123456789abcdef",
            9,
            1,
            "approval_0123456789abcdef",
            "approve",
        ),
        ("stop", "browser_0123456789abcdef", 9),
    ]
    assert [event.kind for event in projection.events_after(0)] == [
        "session_ready",
        "typed_input_admitted",
        "approval_state",
        "session_stopped",
    ]
    assert director.active_generation is None


@pytest.mark.asyncio
async def test_browser_session_lease_expires_only_after_authenticated_inactivity() -> None:
    now = [100.0]
    stopped: list[tuple[str, int]] = []

    async def provision(identity: str) -> int:
        del identity
        return 9

    async def stop(identity: str, generation: int) -> None:
        stopped.append((identity, generation))

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=LiveKitConnection(
                "wss://livekit.test",
                "test-key",
                "synthetic-browser-lease-secret-32-bytes",
            ),
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=_noop_submit,
        stop=stop,
        approval=_noop_approval,
        projection=BrowserEventProjection(),
        inactivity_timeout_seconds=30,
        clock=lambda: now[0],
    )
    credential = await director.start()

    now[0] = 129.0
    assert not await director.expire_if_inactive()
    await director.public_events_after(
        participant_identity=credential.participant_identity,
        sequence=0,
    )
    now[0] = 158.0
    assert not await director.expire_if_inactive()
    now[0] = 160.0
    assert await director.expire_if_inactive()
    assert stopped == [(credential.participant_identity, 9)]
    assert director.active_identity is None
