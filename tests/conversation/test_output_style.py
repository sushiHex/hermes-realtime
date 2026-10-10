from __future__ import annotations

import pytest

from hermes_realtime.conversation.output_style import (
    OUTPUT_STYLES,
    OutputStyleSelection,
    communication_policy,
)


def test_output_styles_are_exact_bounded_communication_preferences() -> None:
    assert OUTPUT_STYLES == ("default", "proactive", "concise", "explanatory", "learning")
    selection = OutputStyleSelection()
    assert selection.get() == "default"
    for style in OUTPUT_STYLES:
        selection.select(style)
        assert selection.get() == style
        policy = communication_policy(style)
        assert policy.startswith(f"Output style: {style}.\n")
        assert "final user message" in policy
        assert "communication preference only" in policy
        assert "accepted dispatch" in policy
        assert "foreground tools" in policy
        assert "manageable spoken chunks" in policy


@pytest.mark.parametrize("value", ["Default", "", "unknown", True, None, 1])
def test_style_refusal_retains_previous_selection(value: object) -> None:
    selection = OutputStyleSelection()
    selection.select("learning")
    with pytest.raises((TypeError, ValueError)):
        selection.select(value)  # type: ignore[arg-type]
    assert selection.get() == "learning"


def test_style_rejects_string_subclasses() -> None:
    class Style(str):
        pass

    with pytest.raises(TypeError):
        OutputStyleSelection().select(Style("default"))


@pytest.mark.parametrize("provider", ["ollama", "codex"])
def test_host_passes_the_same_live_selection_getter_to_each_provider(provider) -> None:
    from hermes_realtime.host_launcher import _build_streaming_inference

    selection = OutputStyleSelection()
    inference = _build_streaming_inference(
        inference_provider=provider, ollama_base_url="http://127.0.0.1:11434",
        ollama_model="synthetic", codex_model="synthetic", codex_effort="low",
        codex_executable=None, output_style=selection.get,
    )
    selection.select("learning")
    assert inference._output_style() == "learning"
    selection.select("concise")
    assert inference._output_style() == "concise"
