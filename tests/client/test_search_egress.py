"""Binding-scoped public-search egress authority tests."""

from __future__ import annotations

import json

import pytest

from hermes_realtime.client import (
    BrowserEventProjection,
    BrowserSessionDirector,
    BrowserTokenIssuer,
)
from hermes_realtime.livekit import LiveKitConnection


def test_search_egress_is_default_closed_and_binding_rotation_revokes_consent() -> None:
    try:
        from hermes_realtime.search_egress import (
            SEARCH_EGRESS_CONSENT_VERSION,
            SEARCH_EGRESS_DISCLOSURE_DIGEST,
            SearchEgressAuthority,
            SearchEgressBindingV1,
            SearchEgressConsentRequestV1,
        )
    except ImportError as error:
        pytest.fail(f"RED bootstrap: public-search egress authority is missing: {error}")

    authority = SearchEgressAuthority(operator_enabled=True)
    original = SearchEgressBindingV1(
        participant_identity="browser_0123456789abcdef",
        binding_generation=1,
    )
    replacement = SearchEgressBindingV1(
        participant_identity="browser_fedcba9876543210",
        binding_generation=2,
    )

    authority.bind(original)
    assert authority.status().state == "idle"
    assert authority.permits_egress() is False

    status = authority.consent(
        original,
        SearchEgressConsentRequestV1(
            sequence=1,
            accepted=True,
            consent_version=SEARCH_EGRESS_CONSENT_VERSION,
            disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
        ),
    )
    assert status.state == "active"
    assert authority.permits_egress() is True

    authority.bind(replacement)
    assert authority.status().state == "idle"
    assert authority.permits_egress() is False
    with pytest.raises(PermissionError, match="active search egress binding"):
        authority.consent(
            original,
            SearchEgressConsentRequestV1(
                sequence=2,
                accepted=True,
                consent_version=SEARCH_EGRESS_CONSENT_VERSION,
                disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
            ),
        )


def test_search_egress_controls_are_exactly_sequenced_replay_safe_and_revocable() -> None:
    from hermes_realtime.search_egress import (
        SEARCH_EGRESS_CONSENT_VERSION,
        SEARCH_EGRESS_DISCLOSURE_DIGEST,
        SearchEgressAuthority,
        SearchEgressBindingV1,
        SearchEgressConsentRequestV1,
        SearchEgressRevokeRequestV1,
    )

    authority = SearchEgressAuthority(operator_enabled=True)
    binding = SearchEgressBindingV1(
        participant_identity="browser_0123456789abcdef",
        binding_generation=1,
    )
    consent = SearchEgressConsentRequestV1(
        sequence=1,
        accepted=True,
        consent_version=SEARCH_EGRESS_CONSENT_VERSION,
        disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
    )
    revoke = SearchEgressRevokeRequestV1(sequence=2)
    authority.bind(binding)

    assert authority.consent(binding, consent).state == "active"
    assert authority.consent(binding, consent).state == "active"
    admission = authority.admit()
    assert admission is not None
    with pytest.raises(RuntimeError, match="control sequence"):
        authority.revoke(binding, SearchEgressRevokeRequestV1(sequence=1))

    assert authority.revoke(binding, revoke).state == "idle"
    assert authority.admit() is None
    admission.close()
    admission.close()
    assert authority.revoke(binding, revoke).state == "idle"
    assert authority.permits_egress() is False

    replacement = SearchEgressBindingV1(
        participant_identity="browser_fedcba9876543210",
        binding_generation=2,
    )
    authority.bind(replacement)
    with pytest.raises(PermissionError, match="active search egress binding"):
        authority.invalidate(binding)
    assert authority.status().state == "idle"
    authority.invalidate(replacement)
    assert authority.permits_egress() is False
    assert authority.status().available is False
    assert authority.status().state == "unavailable"


def test_search_egress_wire_parsers_are_strict_and_content_free() -> None:
    from hermes_realtime.search_egress import (
        SEARCH_EGRESS_CONSENT_VERSION,
        SEARCH_EGRESS_DISCLOSURE_DIGEST,
        SearchEgressConsentRequestV1,
        SearchEgressRevokeRequestV1,
        SearchEgressStatusV1,
        parse_search_egress_consent_request,
        parse_search_egress_revoke_request,
        search_egress_status_to_primitive,
    )

    consent = parse_search_egress_consent_request(
        (
            '{"accepted":true,"consentVersion":"'
            + SEARCH_EGRESS_CONSENT_VERSION
            + '","disclosureDigest":"'
            + SEARCH_EGRESS_DISCLOSURE_DIGEST
            + '","sequence":1}'
        ).encode()
    )
    assert type(consent) is SearchEgressConsentRequestV1
    assert parse_search_egress_revoke_request(b'{"sequence":2}') == SearchEgressRevokeRequestV1(
        sequence=2
    )
    assert search_egress_status_to_primitive(
        SearchEgressStatusV1(available=True, state="idle")
    ) == {
        "available": True,
        "consentVersion": SEARCH_EGRESS_CONSENT_VERSION,
        "disclosureDigest": SEARCH_EGRESS_DISCLOSURE_DIGEST,
        "searchEgressState": "idle",
    }

    with pytest.raises(ValueError, match="duplicate"):
        parse_search_egress_consent_request(
            (
                '{"accepted":true,"accepted":true,"consentVersion":"'
                + SEARCH_EGRESS_CONSENT_VERSION
                + '","disclosureDigest":"'
                + SEARCH_EGRESS_DISCLOSURE_DIGEST
                + '","sequence":1}'
            ).encode()
        )
    with pytest.raises(ValueError, match="exact fields"):
        parse_search_egress_revoke_request(b'{"sequence":2,"query":"forbidden"}')


@pytest.mark.asyncio
async def test_browser_session_owns_search_egress_binding_lifecycle() -> None:
    from hermes_realtime.search_egress import (
        SEARCH_EGRESS_CONSENT_VERSION,
        SEARCH_EGRESS_DISCLOSURE_DIGEST,
        SearchEgressAuthority,
        SearchEgressConsentRequestV1,
    )

    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    authority = SearchEgressAuthority(operator_enabled=True)
    projection = BrowserEventProjection(capacity=16)

    async def provision(_identity: str) -> int:
        return 1

    async def reprovision(identity: str) -> int:
        return 2 if identity == "browser_fedcba9876543210" else 3

    async def submit(*_args: object) -> None:
        return None

    async def stop(*_args: object) -> None:
        return None

    async def approval(*_args: object) -> None:
        return None

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
        stop=stop,
        approval=approval,
        projection=projection,
        search_egress_authority=authority,
    )

    original = await director.start()
    initial = [
        event for event in projection.events_after(0) if event.kind == "search_egress_status"
    ]
    assert initial[-1].data["searchEgressState"] == "idle"

    status = await director.consent_to_search_egress(
        participant_identity=original.participant_identity,
        request=SearchEgressConsentRequestV1(
            sequence=1,
            accepted=True,
            consent_version=SEARCH_EGRESS_CONSENT_VERSION,
            disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
        ),
    )
    assert status.state == "active"
    assert authority.permits_egress() is True

    replacement = await director.rebind(
        participant_identity=original.participant_identity,
        request_id="rebind_0123456789abcdef",
    )
    assert replacement.participant_identity == "browser_fedcba9876543210"
    assert authority.permits_egress() is False
    latest = [event for event in projection.events_after(0) if event.kind == "search_egress_status"]
    assert latest[-1].data["searchEgressState"] == "idle"
    with pytest.raises(PermissionError, match="active session"):
        await director.consent_to_search_egress(
            participant_identity=original.participant_identity,
            request=SearchEgressConsentRequestV1(
                sequence=1,
                accepted=True,
                consent_version=SEARCH_EGRESS_CONSENT_VERSION,
                disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
            ),
        )

    await director.stop(participant_identity=replacement.participant_identity)
    assert authority.permits_egress() is False


@pytest.mark.asyncio
async def test_failed_rebind_rollback_restores_a_fresh_closed_search_binding() -> None:
    from hermes_realtime.search_egress import (
        SEARCH_EGRESS_CONSENT_VERSION,
        SEARCH_EGRESS_DISCLOSURE_DIGEST,
        SearchEgressAuthority,
        SearchEgressConsentRequestV1,
    )

    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    authority = SearchEgressAuthority(operator_enabled=True)

    async def provision(_identity: str) -> int:
        return 1

    async def reprovision(identity: str) -> int:
        if identity == "browser_fedcba9876543210":
            raise RuntimeError("replacement failed")
        return 3

    async def noop(*_args: object) -> None:
        return None

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
        submit=noop,
        stop=noop,
        approval=noop,
        projection=BrowserEventProjection(capacity=16),
        search_egress_authority=authority,
    )
    credential = await director.start()
    consent = SearchEgressConsentRequestV1(
        sequence=1,
        accepted=True,
        consent_version=SEARCH_EGRESS_CONSENT_VERSION,
        disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
    )
    await director.consent_to_search_egress(
        participant_identity=credential.participant_identity,
        request=consent,
    )

    with pytest.raises(RuntimeError, match="replacement failed"):
        await director.rebind(participant_identity=credential.participant_identity)

    assert director.active_identity == credential.participant_identity
    assert director.active_generation == 3
    assert authority.permits_egress() is False
    restored = await director.consent_to_search_egress(
        participant_identity=credential.participant_identity,
        request=consent,
    )
    assert restored.state == "active"


@pytest.mark.asyncio
async def test_projection_resync_rotates_search_consent_and_republishes_idle() -> None:
    from hermes_realtime.search_egress import (
        SEARCH_EGRESS_CONSENT_VERSION,
        SEARCH_EGRESS_DISCLOSURE_DIGEST,
        SearchEgressAuthority,
        SearchEgressConsentRequestV1,
    )

    identities = iter(("browser_0123456789abcdef", "browser_fedcba9876543210"))
    authority = SearchEgressAuthority(operator_enabled=True)
    projection = BrowserEventProjection(capacity=16)

    async def provision(_identity: str) -> int:
        return 1

    async def reprovision(_identity: str) -> int:
        return 2

    async def noop(*_args: object) -> None:
        return None

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
        submit=noop,
        stop=noop,
        approval=noop,
        projection=projection,
        search_egress_authority=authority,
    )
    original = await director.start()
    consent = SearchEgressConsentRequestV1(
        sequence=1,
        accepted=True,
        consent_version=SEARCH_EGRESS_CONSENT_VERSION,
        disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
    )
    await director.consent_to_search_egress(
        participant_identity=original.participant_identity,
        request=consent,
    )

    replacement = await director.projection_resync(
        participant_identity=original.participant_identity
    )

    assert replacement.participant_identity == "browser_fedcba9876543210"
    assert authority.permits_egress() is False
    statuses = [
        event for event in projection.events_after(0) if event.kind == "search_egress_status"
    ]
    assert len(statuses) == 1
    assert statuses[0].data["searchEgressState"] == "idle"
    refreshed = await director.consent_to_search_egress(
        participant_identity=replacement.participant_identity,
        request=consent,
    )
    assert refreshed.state == "active"


@pytest.mark.asyncio
async def test_inactivity_expiry_closes_search_egress_before_session_teardown() -> None:
    from hermes_realtime.search_egress import (
        SEARCH_EGRESS_CONSENT_VERSION,
        SEARCH_EGRESS_DISCLOSURE_DIGEST,
        SearchEgressAuthority,
        SearchEgressConsentRequestV1,
    )

    now = [0.0]
    authority = SearchEgressAuthority(operator_enabled=True)

    async def provision(_identity: str) -> int:
        return 1

    async def noop(*_args: object) -> None:
        return None

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
        submit=noop,
        stop=noop,
        approval=noop,
        projection=BrowserEventProjection(capacity=16),
        search_egress_authority=authority,
        inactivity_timeout_seconds=30.0,
        clock=lambda: now[0],
    )
    credential = await director.start()
    await director.consent_to_search_egress(
        participant_identity=credential.participant_identity,
        request=SearchEgressConsentRequestV1(
            sequence=1,
            accepted=True,
            consent_version=SEARCH_EGRESS_CONSENT_VERSION,
            disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
        ),
    )
    assert authority.permits_egress() is True

    now[0] = 31.0
    assert await director.expire_if_inactive() is True
    assert authority.permits_egress() is False


@pytest.mark.asyncio
async def test_authenticated_http_search_consent_and_revoke_are_content_free() -> None:
    from hermes_realtime.client import (
        BrowserBootstrapApplication,
        BrowserTokenVerifier,
        OneTimeBootstrapCapability,
    )
    from hermes_realtime.search_egress import (
        SEARCH_EGRESS_CONSENT_VERSION,
        SEARCH_EGRESS_DISCLOSURE_DIGEST,
        SearchEgressAuthority,
    )

    connection = LiveKitConnection(
        "wss://livekit.test",
        "test-key",
        "synthetic-browser-bootstrap-secret-32-bytes",
    )
    authority = SearchEgressAuthority(operator_enabled=True)

    async def provision(_identity: str) -> int:
        return 1

    async def noop(*_args: object) -> None:
        return None

    director = BrowserSessionDirector(
        issuer=BrowserTokenIssuer(
            connection=connection,
            room_name="hermes-local",
            identity_factory=lambda: "browser_0123456789abcdef",
        ),
        provision=provision,
        submit=noop,
        stop=noop,
        approval=noop,
        projection=BrowserEventProjection(capacity=16),
        search_egress_authority=authority,
    )
    credential = await director.start()
    app = BrowserBootstrapApplication(
        sessions=director,
        verifier=BrowserTokenVerifier(connection=connection, room_name="hermes-local"),
        capability=OneTimeBootstrapCapability(token_factory=lambda: "z" * 43),
        allowed_origin="https://phone.test:8443",
        worker_identity="worker_hermes_browser",
    )

    async def post(path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload, separators=(",", ":")).encode()
        response = await app.handle(
            method="POST",
            path=path,
            headers={
                "authorization": f"Bearer {credential.token}",
                "content-length": str(len(body)),
                "content-type": "application/json",
                "origin": "https://phone.test:8443",
            },
            body=body,
        )
        assert response.status == 200
        decoded = json.loads(response.body)
        assert type(decoded) is dict
        return decoded

    active = await post(
        "/api/v1/search-egress-consent",
        {
            "accepted": True,
            "consentVersion": SEARCH_EGRESS_CONSENT_VERSION,
            "disclosureDigest": SEARCH_EGRESS_DISCLOSURE_DIGEST,
            "sequence": 1,
        },
    )
    assert active == {
        "available": True,
        "consentVersion": SEARCH_EGRESS_CONSENT_VERSION,
        "disclosureDigest": SEARCH_EGRESS_DISCLOSURE_DIGEST,
        "searchEgressState": "active",
    }
    assert authority.permits_egress() is True

    idle = await post("/api/v1/search-egress-revoke", {"sequence": 2})
    assert idle["searchEgressState"] == "idle"
    assert authority.permits_egress() is False
