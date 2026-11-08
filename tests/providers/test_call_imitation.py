"""Predicate and registry-drift tests for tool-call imitation in model output (#311)."""

import pytest

from hermes_realtime.providers import codex_app_server, ollama
from hermes_realtime.providers._call_imitation import (
    _CALL,
    _KEYED_NAME,
    _NAME_AS_KEY,
    _XML_FUNCTION,
    ADVERTISED_TOOL_NAMES,
    contains_tool_call_imitation,
)


def test_registry_matches_the_codex_dynamic_tools() -> None:
    assert ADVERTISED_TOOL_NAMES == codex_app_server._DYNAMIC_TOOL_NAMES
    assert sorted(ADVERTISED_TOOL_NAMES) == ["search_knowledge", "start_work", "task_status"]


def test_ollama_work_kinds_are_advertised_names() -> None:
    schema = ollama.OllamaStreamingInference._work_response_schema(256, can_start=True)
    kinds = {
        branch["properties"]["kind"]["const"]  # type: ignore[index]
        for branch in schema["oneOf"]  # type: ignore[attr-defined]
    }
    assert "reply" in kinds
    assert kinds - {"reply"} <= ADVERTISED_TOOL_NAMES


CALL = [
    'start_work(objective="check the logs")',
    "Start_Work(objective='x')",
    "hermes.task_status()",
    "functions.start_work(objective='x')",
    "mcp.hermes.search_knowledge(query='x')",
    "I will call `start_work(objective=x)` now",
    "```\ntask_status()\n```",
    "Sure. TASK_STATUS() and more words",
]
KEYED = [
    '{"name": "start_work", "arguments": {"objective": "x"}}',
    '{"name":"search_knowledge"}',
    "{'name': 'task_status'}",
    '[{"name": "search_knowledge", "arguments": {}}]',
    '{\\"name\\": \\"start_work\\"}',
    '{ "name" : "Start_Work" }',
    '{"kind": "start_work", "objective": "x"}',
    '{"tool": "task_status"}',
]
NAME_AS_KEY = [
    '{"start_work": {"objective": "x"}}',
    "{'task_status' : {}}",
    '{"Search_Knowledge":{"query":"x"}}',
]
XML = [
    "<function=start_work>",
    '<function="search_knowledge">',
    "< function = task_status >",
    "<FUNCTION=START_WORK",
]
NEGATIVE = [
    "I can start_work for that.",
    "I'll use start_work (the background tool).",
    "The task status is fine.",
    "task_status is a tool I have",
    "Use search_knowledge when you need facts.",
    "my_start_work(x) is unrelated",
    "restart_work(x) is unrelated",
    "start_works(x) is unrelated",
    "task_status_report(x) is unrelated",
    "start_work.",
    '{"name": "Alice", "role": "start_work"}',
    '{"name": "other_tool", "arguments": {}}',
    '{"kind": "reply", "text": "Okay, I can help."}',
    '"start_work" is the name of a tool',
    '{"start_work": true}',
    "<function=other_tool>",
    "<function=start_workers>",
    "Hello, how can I help you today?",
    "",
]


@pytest.mark.parametrize("text", CALL + KEYED + NAME_AS_KEY + XML)
def test_call_syntax_is_detected(text: str) -> None:
    assert contains_tool_call_imitation(text) is True


@pytest.mark.parametrize("text", NEGATIVE)
def test_ordinary_text_is_allowed(text: str) -> None:
    assert contains_tool_call_imitation(text) is False


@pytest.mark.parametrize(
    ("form", "texts"),
    [(_CALL, CALL), (_KEYED_NAME, KEYED), (_NAME_AS_KEY, NAME_AS_KEY), (_XML_FUNCTION, XML)],
    ids=["call", "keyed_name", "name_as_key", "xml_function"],
)
def test_each_form_matches_its_own_examples(form, texts) -> None:
    assert all(form.search(text) is not None for text in texts)


def test_detection_is_bounded_on_adversarial_input() -> None:
    big = " " * 100_000
    assert contains_tool_call_imitation("start_work" + big + "x") is False
    assert contains_tool_call_imitation('"name"' + big + ':"x"') is False
    assert contains_tool_call_imitation('"start_work"' + big + ":x") is False
    assert contains_tool_call_imitation("<" + big + "function=x") is False
    assert contains_tool_call_imitation("a" * 100_000) is False


def test_non_string_is_rejected() -> None:
    with pytest.raises(TypeError):
        contains_tool_call_imitation(b"start_work(")  # type: ignore[arg-type]
