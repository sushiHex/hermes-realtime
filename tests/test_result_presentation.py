"""Source/time text is forwarded, never inferred or independently verified here."""
import pytest

from hermes_realtime.client import BrowserEventProjection
from hermes_realtime.conversation import ConversationContextStore, TaskTerminalOutcome
from hermes_realtime.conversation.updates import UpdatePolicyInput
from hermes_realtime.host_launcher import (
    ProactiveCompletionUpdatePolicy,
    _voice_friendly_completion_text,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("idle", [False, True])
async def test_result_preserves_visual_sources_and_observation_time_in_each_projection(idle):
    voice = "The monitor counted 12 sessions at 07:53 UTC, per https://monitor.example.invalid/status."
    detail = (
        "The monitor counted 12 sessions. Observed at 07:53 UTC; retrieved at 09:30 UTC. "
        "See [Monitor report](https://monitor.example.invalid/status) and "
        "https://archive.example.invalid/records."
    )
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection, floor_available=lambda: idle)
    directive = await policy.decide(UpdatePolicyInput(sequence=1, completion=TaskTerminalOutcome(
        task_id="task_source_result", status="completed",
        summary=f"<voice>{voice}</voice><detail>{detail}</detail>",
    ), context=ConversationContextStore().snapshot()))
    assert projection.events_after(0)[-1].data["text"] == detail
    assert directive.kind == ("interrupt" if idle else "mention_next")
    assert "07:53 UTC" in directive.text
    assert "the source link" not in directive.text
    if idle:
        assert directive.text == (
            "The monitor counted 12 sessions at 07:53 UTC, per monitor.example.invalid."
        )
        assert "archive.example.invalid" not in directive.text
    else:
        assert directive.text == (
            "The monitor counted 12 sessions. Observed at 07:53 UTC; retrieved at 09:30 UTC. "
            "See Monitor report and archive.example.invalid."
        )


def test_source_catalog_keeps_distinct_hosts_and_subdomains_in_first_seen_order():
    assert _voice_friendly_completion_text("completed", (
        "Findings.\n\nSources:\n"
        "https://blog.example.invalid/a\n"
        "https://www.example.invalid/a\n"
        "https://docs.example.invalid/b\n"
        "https://archive.example.invalid/a\n"
        "https://BLOG.example.invalid/b"
    )) == (
        "Findings. Sources: blog.example.invalid www.example.invalid docs.example.invalid "
        "archive.example.invalid."
    )


def test_named_references_are_not_deduplicated_by_label():
    assert _voice_friendly_completion_text("completed", (
        "Findings.\n\nSources:\n"
        "[Source](https://one.example.invalid/a)\n"
        "[Source](https://two.example.invalid/a)\n"
        "[Documentation](https://two.example.invalid/b)"
    )) == "Findings. Sources: Source Source Documentation."


def test_repeated_inline_source_references_are_not_deduplicated():
    assert _voice_friendly_completion_text("completed", (
        "Version one: https://docs.example.invalid/one. "
        "Version two: https://docs.example.invalid/two."
    )) == "Version one: docs.example.invalid. Version two: docs.example.invalid."


@pytest.mark.parametrize("catalog", [
    "https://www.example.invalid/one\nNot a source entry.\nhttps://www.example.invalid/two",
    "https://www.example.invalid/one https://www.example.invalid/two",
])
def test_non_catalog_source_text_is_not_deduplicated(catalog):
    spoken = _voice_friendly_completion_text("completed", "Findings.\n\nSources:\n" + catalog)
    assert spoken.count("www.example.invalid") == 2


def test_source_catalog_still_excludes_url_credentials_and_paths():
    spoken = _voice_friendly_completion_text("completed", (
        "Findings.\n\nSources:\n"
        "https://synthetic:fixture@docs.example.invalid/private?key=synthetic\n"
        "https://synthetic:fixture@docs.example.invalid/another#synthetic"
    ))
    assert spoken == "Findings. Sources: docs.example.invalid."


@pytest.mark.asyncio
async def test_short_legacy_source_hostname_survives_deferred_context_handoff():
    detail = "Observed 12 sessions at 07:53 UTC. Source: https://monitor.example.invalid/status"
    policy = ProactiveCompletionUpdatePolicy(projection=BrowserEventProjection())
    directive = await policy.decide(UpdatePolicyInput(sequence=1, completion=TaskTerminalOutcome(
        task_id="task_short_source", status="completed", summary=detail,
    ), context=ConversationContextStore().snapshot()))
    assert directive.text == "Observed 12 sessions at 07:53 UTC. Source: monitor.example.invalid"


@pytest.mark.parametrize("scheme", ["https", "HTTPS", "HtTpS"])
def test_spoken_source_labels_do_not_read_paths_queries_or_userinfo(scheme):
    result = _voice_friendly_completion_text("completed",
        f"See [The monitor]({scheme}://monitor.example.invalid/status) and "
        f"{scheme}://synthetic:fixture@archive.example.invalid/path?key=synthetic#fragment.")
    assert result == "See The monitor and archive.example.invalid."
    assert "synthetic" not in result


def test_malformed_spoken_url_keeps_a_bounded_reference_without_reading_raw_authority():
    assert _voice_friendly_completion_text(
        "completed", "See https://synthetic:fixture@[broken/",
    ) == "See link."


@pytest.mark.asyncio
async def test_idle_uses_supplied_voice_instead_of_rewriting_full_detail():
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection, floor_available=lambda: True)
    directive = await policy.decide(UpdatePolicyInput(sequence=1, completion=TaskTerminalOutcome(
        task_id="task_voice_projection", status="completed",
        summary="<voice>The monitor counted 12 sessions.</voice>"
        "<detail>The monitor counted 12 sessions at 07:53 UTC.</detail>",
    ), context=ConversationContextStore().snapshot()))
    assert directive.text == "The monitor counted 12 sessions."
    assert "07:53 UTC" in projection.events_after(0)[-1].data["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", ["https", "HTTPS", "HtTpS"])
async def test_deferred_context_excludes_url_credentials_paths_queries_and_fragments(scheme):
    detail = (
        "Observed 12 sessions at 07:53 UTC. Source: "
        f"{scheme}://synthetic:fixture@monitor.example.invalid/private-path?key=synthetic#fragment."
    )
    projection = BrowserEventProjection()
    policy = ProactiveCompletionUpdatePolicy(projection=projection)
    directive = await policy.decide(UpdatePolicyInput(sequence=1, completion=TaskTerminalOutcome(
        task_id="task_credential_shaped_reference", status="completed",
        summary=f"<voice>The monitor counted 12 sessions.</voice><detail>{detail}</detail>",
    ), context=ConversationContextStore().snapshot()))
    assert directive.text == "Observed 12 sessions at 07:53 UTC. Source: monitor.example.invalid."
    assert projection.events_after(0)[-1].data["text"] == detail


@pytest.mark.asyncio
@pytest.mark.parametrize("reference, expected", [
    ("https://synthetic:fixture@monitor.example.invalid/a(b)?key=synthetic#fragment",
     "monitor.example.invalid"),
    ("[Monitor](https://synthetic:fixture@monitor.example.invalid/a(b)?key=synthetic#fragment)",
     "Monitor"),
    ("[https://synthetic:fixture@label.example.invalid/path?key=synthetic]"
     "(https://monitor.example.invalid/a(b)?key=synthetic#fragment)", "label.example.invalid"),
])
async def test_parenthesized_url_paths_stay_local_in_deferred_context(reference, expected):
    detail = f"Observed 12 sessions at 07:53 UTC. Source: {reference}."
    projection = BrowserEventProjection()
    directive = await ProactiveCompletionUpdatePolicy(projection=projection).decide(
        UpdatePolicyInput(sequence=1, completion=TaskTerminalOutcome(
            task_id="task_parenthesized_source", status="completed",
            summary=f"<voice>The monitor counted 12 sessions.</voice><detail>{detail}</detail>",
        ), context=ConversationContextStore().snapshot()),
    )
    assert directive.text == f"Observed 12 sessions at 07:53 UTC. Source: {expected}."
    assert projection.events_after(0)[-1].data["text"] == detail


@pytest.mark.asyncio
async def test_deferred_source_catalog_speaks_each_label_once_without_losing_links():
    detail = (
        "The release adds two capabilities.\n\nSources:\n"
        "https://blog.example.invalid/release\n"
        "https://www.example.invalid/one\n"
        "https://www.example.invalid/two\n"
        "https://www.example.invalid/three"
    )
    projection = BrowserEventProjection()
    directive = await ProactiveCompletionUpdatePolicy(projection=projection).decide(
        UpdatePolicyInput(sequence=1, completion=TaskTerminalOutcome(
            task_id="task_source_catalog", status="completed",
            summary=f"<voice>The release adds two capabilities.</voice><detail>{detail}</detail>",
        ), context=ConversationContextStore().snapshot()),
    )
    assert directive.text == (
        "The release adds two capabilities. Sources: blog.example.invalid www.example.invalid"
    )
    result_event = projection.events_after(0)[-1]
    assert result_event.kind == "task_result"
    assert result_event.data["taskId"] == "task_source_catalog"
    assert result_event.data["text"] == detail
