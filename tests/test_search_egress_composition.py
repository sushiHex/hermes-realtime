"""Fail-closed full-host public-search composition tests."""

from __future__ import annotations

import inspect

import pytest


def test_codex_builder_never_implicitly_constructs_public_search(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import hermes_realtime.host_launcher as host

    captured: dict[str, object] = {}

    class FakeCodexInference:
        def __init__(self, **options: object) -> None:
            captured.update(options)

    def forbidden_lookup(**_options: object) -> object:
        raise AssertionError("default Codex composition constructed public-search egress")

    monkeypatch.setattr(host, "CodexAppServerStreamingInference", FakeCodexInference)
    monkeypatch.setattr(host, "PublicRssCurrentFactLookup", forbidden_lookup)

    result = host._build_streaming_inference(
        inference_provider="codex",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_model="unused",
        codex_model="gpt-5.6-codex",
        codex_effort="high",
        codex_executable=None,
        current_fact_lookup=None,
    )

    assert type(result) is FakeCodexInference
    assert captured["current_fact_lookup"] is None
    assert captured["knowledge_coordinator"] is None


def test_public_search_composition_is_default_off_and_consent_bound(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import hermes_realtime.host_launcher as host

    raw_lookups: list[object] = []
    raw_options: list[dict[str, object]] = []

    class FakePublicRssLookup:
        async def lookup(self, _query: str) -> object:
            raise AssertionError("composition test does not execute network lookup")

        async def close(self) -> None:
            return None

    def make_lookup(**_options: object) -> FakePublicRssLookup:
        raw_options.append(dict(_options))
        lookup = FakePublicRssLookup()
        raw_lookups.append(lookup)
        return lookup

    monkeypatch.setattr(host, "PublicRssCurrentFactLookup", make_lookup)

    disabled_authority, disabled_lookup, disabled_coordinator = host._compose_public_search(
        operator_enabled=False,
        knowledge_budget_seconds=3.5,
        knowledge_speculation=False,
        knowledge_recovery=False,
    )
    assert type(disabled_authority) is host.SearchEgressAuthority
    assert disabled_authority.status().available is False
    assert disabled_lookup is None
    assert disabled_coordinator is None
    assert raw_lookups == []

    enabled_authority, enabled_lookup, enabled_coordinator = host._compose_public_search(
        operator_enabled=True,
        knowledge_budget_seconds=2.0,
        knowledge_speculation=False,
        knowledge_recovery=False,
    )
    assert type(enabled_authority) is host.SearchEgressAuthority
    assert enabled_authority.status().state == "unavailable"
    assert type(enabled_lookup) is host.ConsentBoundCurrentFactLookup
    assert enabled_coordinator is None
    assert len(raw_lookups) == 1
    assert raw_options == [
        {"enrich_results": 0, "max_results": 8, "timeout_seconds": 2.0}
    ]


def test_full_host_public_search_is_explicit_and_cannot_be_bypassed() -> None:
    from hermes_realtime.host_launcher import build_local_host_launcher

    signature = inspect.signature(build_local_host_launcher)
    assert signature.parameters["public_search"].default is False

    with pytest.raises(ValueError, match="public search"):
        build_local_host_launcher(
            hermes_api_bearer="synthetic-bearer",
            allow_unsandboxed_tasks=True,
            inference_provider="codex",
            public_search=False,
            knowledge_speculation=True,
        )

    source = inspect.getsource(build_local_host_launcher)
    assert "_compose_public_search(" in source
    assert "search_egress_authority=search_egress_authority" in source

    import hermes_realtime.host_launcher as host

    module_source = inspect.getsource(host)
    assert '"--enable-public-search"' in module_source
    assert "public_search=args.enable_public_search" in module_source
