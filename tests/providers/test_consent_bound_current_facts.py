"""Server-enforced public-search egress gate tests."""

from __future__ import annotations

import pytest

from hermes_realtime.providers.current_facts import CurrentFactEvidence
from hermes_realtime.search_egress import (
    SEARCH_EGRESS_CONSENT_VERSION,
    SEARCH_EGRESS_DISCLOSURE_DIGEST,
    SearchEgressAuthority,
    SearchEgressBindingV1,
    SearchEgressConsentRequestV1,
    SearchEgressRevokeRequestV1,
)


class _RecordingLookup:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.closed = False

    async def lookup(self, query: str) -> CurrentFactEvidence:
        self.queries.append(query)
        return CurrentFactEvidence(
            query=query,
            retrieved_date="2026-08-30",
            sources=(),
            backend="public-rss",
        )

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_consent_bound_lookup_never_calls_delegate_while_gate_is_closed() -> None:
    try:
        from hermes_realtime.providers.current_facts import ConsentBoundCurrentFactLookup
    except ImportError as error:
        pytest.fail(f"RED bootstrap: consent-bound lookup is missing: {error}")

    authority = SearchEgressAuthority(operator_enabled=True)
    binding = SearchEgressBindingV1(
        participant_identity="browser_0123456789abcdef",
        binding_generation=1,
    )
    authority.bind(binding)
    delegate = _RecordingLookup()
    lookup = ConsentBoundCurrentFactLookup(delegate=delegate, authority=authority)

    denied = await lookup.lookup("latest project release")
    assert denied.error == "search_egress_not_consented"
    assert denied.backend == "public-rss"
    assert delegate.queries == []

    authority.consent(
        binding,
        SearchEgressConsentRequestV1(
            sequence=1,
            accepted=True,
            consent_version=SEARCH_EGRESS_CONSENT_VERSION,
            disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
        ),
    )
    allowed = await lookup.lookup("latest project release")
    assert allowed.error is None
    assert delegate.queries == ["latest project release"]

    authority.revoke(binding, SearchEgressRevokeRequestV1(sequence=2))
    denied_again = await lookup.lookup("latest project release notes")
    assert denied_again.error == "search_egress_not_consented"
    assert delegate.queries == ["latest project release"]

    await lookup.close()
    assert delegate.closed is True


@pytest.mark.asyncio
async def test_lookup_uses_a_reserved_admission_not_a_boolean_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.providers.current_facts import ConsentBoundCurrentFactLookup

    authority = SearchEgressAuthority(operator_enabled=True)
    binding = SearchEgressBindingV1(
        participant_identity="browser_0123456789abcdef",
        binding_generation=1,
    )
    authority.bind(binding)
    authority.consent(
        binding,
        SearchEgressConsentRequestV1(
            sequence=1,
            accepted=True,
            consent_version=SEARCH_EGRESS_CONSENT_VERSION,
            disclosure_digest=SEARCH_EGRESS_DISCLOSURE_DIGEST,
        ),
    )
    original_admit = authority.admit

    def admit_then_revoke():  # type: ignore[no-untyped-def]
        admission = original_admit()
        if admission is not None:
            authority.revoke(binding, SearchEgressRevokeRequestV1(sequence=2))
        return admission

    def forbidden_snapshot() -> bool:
        raise AssertionError("lookup used non-reserving boolean authority")

    monkeypatch.setattr(authority, "admit", admit_then_revoke)
    monkeypatch.setattr(authority, "permits_egress", forbidden_snapshot)
    delegate = _RecordingLookup()
    lookup = ConsentBoundCurrentFactLookup(delegate=delegate, authority=authority)

    admitted = await lookup.lookup("latest project release")
    denied = await lookup.lookup("latest project release notes")

    assert admitted.error is None
    assert denied.error == "search_egress_not_consented"
    assert delegate.queries == ["latest project release"]
