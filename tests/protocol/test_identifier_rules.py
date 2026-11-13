"""Every site that enforces a shared identifier rule accepts and refuses the same inputs.

Each rule is an enumerated truth table. Each site is reached the way production reaches it:
through its validator where one exists, otherwise through the pattern it matches with.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from functools import reduce

import pytest
from pydantic import TypeAdapter, ValidationError

from hermes_realtime.protocol import Durability
from hermes_realtime.protocol.events import IdentifierString, VoiceConversationId

# rule -> (accepted, refused)
EXACT_RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "identifier": (
        ("a", "Z", "0", "evt_1", "a.b:c-d_e", "0._:-", "a" * 128, "A" + "-" * 127),
        (
            "", "a" * 129, "_a", ".a", ":a", "-a", " a", "a ", "a\n", "a b", "a/b", "a\x00",
            "é", "aé", "١",
        ),
    ),
    "task_id": (
        ("task_a", "task_0", "task_A", "task_a.b:c-d_e", "task_" + "a" * 123),
        (
            "", "tas", "task_", "task__a", "task_.a", "task_-a", "task_" + "a" * 124, "Task_a",
            "task-a", "xtask_a", " task_a", "task_a\n", "task_a b", "task_é", "task_a/b",
        ),
    ),
    "delegation_run_id": (
        ("deleg_a", "deleg_0.b:c-d_e", "deleg_" + "a" * 122),
        (
            "", "deleg_", "deleg__a", "deleg_-a", "deleg_" + "a" * 123, "run_a", "Deleg_a",
            "xdeleg_a", "deleg_a\n", "deleg_a b",
        ),
    ),
    "turn_id": (
        (
            "a", "Z", "session_1_turn_2", "resume_00000000000000ff", "mention_3", "update_7",
            "a.b:c-d_e", "a" * 128,
        ),
        ("", "1a", "_a", ".a", "a" * 129, "a b", " a", "a\n", "é", "a/b"),
    ),
    "metric_label": (
        ("a", "codex_app-server9", "a" * 64),
        ("", "A", "1a", "_a", "-a", "a" * 65, "a.b", "a b", "a\n", "aB"),
    ),
    "worker_identity": (
        ("worker_" + "a" * 8, "worker_A0_-A0_-", "worker_" + "a" * 64),
        (
            "", "worker_", "worker_" + "a" * 7, "worker_" + "a" * 65, "Worker_" + "a" * 8,
            "worker_aaaaaaa.", "worker_" + "a" * 8 + "\n", "xworker_" + "a" * 8,
        ),
    ),
    "browser_identity": (
        ("browser_" + "0" * 16, "browser_0123456789abcdef", "browser_" + "f" * 64),
        (
            "", "browser_", "browser_" + "0" * 15, "browser_" + "0" * 65,
            "browser_" + "A" * 16, "browser_" + "g" * 16, "Browser_" + "0" * 16,
            "browser_" + "0" * 16 + "\n", "xbrowser_" + "0" * 16,
        ),
    ),
    "voice_conversation_id": (
        ("a", "0", "_", "-", "A0_-", "a" * 64),
        ("", "a" * 65, "a.b", "a:b", "a b", "a\n", "é"),
    ),
}

# (rule, kind, module, attribute[, arguments after the value])
EXACT_SITES: tuple[tuple[str, str, str, str] | tuple[str, str, str, str, tuple[object, ...]], ...]
EXACT_SITES = (
    ("identifier", "adapter", "hermes_realtime.protocol.events", "IdentifierString"),
    ("identifier", "call", "hermes_realtime.conversation.tasks", "_validate_protocol_identifier",
     ("field",)),
    ("identifier", "call", "hermes_realtime.conversation.work_tools", "_validate_identifier",
     ("field",)),
    ("identifier", "call", "hermes_realtime.conversation.context",
     "ConversationContextStore._validate_identifier", ("field",)),
    # Knowledge validates its turn identity with the general identifier rule, not turn_id.
    ("identifier", "pattern", "hermes_realtime.conversation.knowledge", "_TURN_ID"),
    ("identifier", "pattern", "hermes_realtime.integration.bridge", "_IDENTIFIER_PATTERN"),
    ("identifier", "pattern", "hermes_realtime.integration.plugin", "_IDENTIFIER_PATTERN"),
    ("identifier", "pattern", "hermes_realtime.integration.api", "_SESSION_ID"),
    ("identifier", "dispatch", "hermes_realtime.integration.service", "HermesDispatchCommand"),
    ("task_id", "call", "hermes_realtime.conversation.tasks", "_validate_task_id"),
    ("task_id", "call", "hermes_realtime.conversation.work_tools", "_validate_task_id"),
    ("task_id", "call", "hermes_realtime.conversation.context",
     "ConversationContextStore._validate_task_identifier"),
    ("task_id", "call", "hermes_realtime.conversation.context", "ActiveTaskSummary",
     ("objective",)),
    ("task_id", "pattern", "hermes_realtime.conversation.commands", "_TASK_ID"),
    ("task_id", "pattern", "hermes_realtime.client.projection", "_TASK_ID"),
    ("task_id", "pattern", "scripts.native_typed_journey", "TASK"),
    ("task_id", "pattern", "scripts.natural_work_target_resolution", "_TASK_ID"),
    ("delegation_run_id", "call", "hermes_realtime.conversation.tasks", "_validate_run_id"),
    ("delegation_run_id", "call", "hermes_realtime.conversation.context",
     "ConversationContextStore._validate_run_identifier"),
    ("turn_id", "pattern", "hermes_realtime.client.projection", "_TURN_ID"),
    ("turn_id", "pattern", "hermes_realtime.conversation.work_tools", "_ANCHOR_TURN_ID"),
    ("metric_label", "pattern", "hermes_realtime.client.projection", "_METRIC_LABEL"),
    ("metric_label", "pattern", "hermes_realtime.conversation.telemetry", "_LABEL"),
    ("worker_identity", "pattern", "hermes_realtime.client.http",
     "BrowserBootstrapApplication._WORKER_IDENTITY"),
    ("worker_identity", "pattern", "hermes_realtime.client.runtime",
     "BrowserClientRuntime._WORKER_IDENTITY"),
    ("browser_identity", "pattern", "hermes_realtime.client.bootstrap", "_IDENTITY_PATTERN"),
    ("browser_identity", "pattern", "hermes_realtime.search_egress", "_IDENTITY"),
    ("voice_conversation_id", "adapter", "hermes_realtime.protocol.events",
     "VoiceConversationId"),
    ("voice_conversation_id", "pattern", "hermes_realtime.companion.integrity",
     "_CONVERSATION_ID"),
    ("voice_conversation_id", "pattern", "hermes_realtime.integration.voice_deletes",
     "_CONVERSATION_ID"),
    ("voice_conversation_id", "pattern", "hermes_realtime.integration.voice_tail",
     "_CONVERSATION_ID"),
)

# rule -> ((text, first match or None), ...)
SEARCH_RULES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "private_run_token": (
        ("deleg_a", "deleg_a"),
        ("see deleg_abc.def:g-h_i now", "deleg_abc.def:g-h_i"),
        ("deleg_a deleg_b", "deleg_a"),
        ("task_deleg_1", "deleg_1"),
        ("xdeleg_a", "deleg_a"),
        ("1deleg_a", "deleg_a"),
        ("deleg_" + "a" * 300, "deleg_" + "a" * 300),
        ("deleg_", None),
        ("deleg_-a", None),
        ("deleg__a", None),
        ("deleg_ a", None),
        ("DELEG_a", None),
        ("delegate a task", None),
        ("", None),
    ),
    # The task controller alone also requires that no letter or digit precede the token.
    "private_run_token_at_boundary": (
        ("deleg_a", "deleg_a"),
        ("see deleg_abc.def:g-h_i now", "deleg_abc.def:g-h_i"),
        ("task_deleg_1", "deleg_1"),
        ("_deleg_a", "deleg_a"),
        ("xdeleg_a", None),
        ("1deleg_a", None),
        ("deleg_", None),
        ("deleg_-a", None),
        ("DELEG_a", None),
        ("", None),
    ),
    "private_handle": (
        ("deleg_abc-1_x done", "deleg_abc-1_x"),
        ("deleg_", "deleg_"),
        ("xdeleg_a", "deleg_a"),
        # The tail stops at characters a delegation run id may contain.
        ("deleg_a.b:c", "deleg_a"),
        ("run_12345678", "run_12345678"),
        ("run_1234567", None),
        ("run_until_complete", "run_until_complete"),
        ("run_a.bcdefgh", None),
        ("subagent_x", "subagent_x"),
        ("subagent_", None),
        ("task_1", None),
        ("DELEG_a RUN_12345678 SUBAGENT_x", None),
        ("", None),
    ),
    "private_authority_gate": (
        ("deleg_a", "deleg_a"),
        ("xdeleg_a.b:c", "deleg_a.b:c"),
        ("deleg_", None),
        ("run_" + "a" * 16, "run_" + "a" * 16),
        ("run_" + "a" * 128, "run_" + "a" * 128),
        ("run_" + "a" * 15, None),
        ("run_" + "a" * 129, None),
        ("xrun_" + "a" * 16, None),
        ("_run_" + "a" * 16, None),
        ("run__" + "a" * 16, None),
        (" run_" + "a" * 16 + ".", "run_" + "a" * 16),
        ("", None),
    ),
}

SEARCH_SITES: tuple[tuple[str, str, str], ...] = (
    ("private_run_token", "hermes_realtime.conversation.commands", "_PRIVATE_AUTHORITY"),
    ("private_run_token", "hermes_realtime.conversation.context", "_PRIVATE_RUN_TOKEN_PATTERN"),
    ("private_run_token", "hermes_realtime.conversation.updates", "_PRIVATE_RUN_TOKEN_PATTERN"),
    ("private_run_token", "hermes_realtime.conversation.work_tools", "_PRIVATE_AUTHORITY"),
    ("private_run_token", "hermes_realtime.providers.codex_app_server", "_PRIVATE_AUTHORITY"),
    ("private_run_token", "hermes_realtime.providers.ollama", "_PRIVATE_AUTHORITY"),
    ("private_run_token", "scripts.evaluate_natural_work_routing", "_PRIVATE_AUTHORITY"),
    ("private_run_token_at_boundary", "hermes_realtime.conversation.tasks",
     "_PRIVATE_RUN_TOKEN_PATTERN"),
    ("private_handle", "hermes_realtime.client.projection", "_PRIVATE_HANDLE"),
    ("private_handle", "hermes_realtime.conversation.result_projection",
     "_PRIVATE_RESULT_HANDLE"),
    ("private_authority_gate", "scripts.real_natural_work_gate", "_PRIVATE_AUTHORITY"),
)


def _resolve(module: str, attribute: str) -> object:
    return reduce(getattr, attribute.split("."), importlib.import_module(module))


def _acceptor(site: tuple[object, ...]) -> Callable[[str], bool]:
    kind, module, attribute = site[1], site[2], site[3]
    arguments = site[4] if len(site) == 5 else ()
    assert type(kind) is str and type(module) is str and type(attribute) is str
    assert type(arguments) is tuple
    target = _resolve(module, attribute)

    if kind == "pattern":
        assert type(target) is re.Pattern
        return lambda value: target.fullmatch(value) is not None
    if kind == "adapter":
        adapter: TypeAdapter[str] = TypeAdapter(target)

        def validates(value: str) -> bool:
            try:
                return adapter.validate_python(value, strict=True) == value
            except ValidationError:
                return False

        return validates
    if kind == "dispatch":
        assert callable(target)

        def constructs(value: str) -> bool:
            for field in ("session_id", "task_id"):
                fields = {"session_id": "s", "task_id": "t", field: value}
                try:
                    target(objective="o", durability=Durability.EPHEMERAL, **fields)
                except ValueError:
                    return False
            return True

        return constructs
    assert kind == "call" and callable(target)

    def returns(value: str) -> bool:
        try:
            target(value, *arguments)
        except (TypeError, ValueError, RuntimeError):
            return False
        return True

    return returns


def _site_id(site: tuple[object, ...]) -> str:
    return f"{site[0]}@{site[2]}.{site[3]}"


def test_rule_tables_cover_every_site_and_every_wire_type() -> None:
    assert {site[0] for site in EXACT_SITES} == set(EXACT_RULES)
    assert {site[0] for site in SEARCH_SITES} == set(SEARCH_RULES)
    assert IdentifierString is not VoiceConversationId


@pytest.mark.parametrize("site", EXACT_SITES, ids=_site_id)
def test_site_accepts_and_refuses_exactly_its_rule(site: tuple[object, ...]) -> None:
    accepted, refused = EXACT_RULES[str(site[0])]
    accepts = _acceptor(site)
    assert [value for value in accepted if not accepts(value)] == []
    assert [value for value in refused if accepts(value)] == []


@pytest.mark.parametrize("site", SEARCH_SITES, ids=lambda site: f"{site[0]}@{site[1]}.{site[2]}")
def test_site_finds_exactly_the_private_tokens_of_its_rule(site: tuple[str, str, str]) -> None:
    pattern = _resolve(site[1], site[2])
    assert type(pattern) is re.Pattern
    observed = []
    for text, _ in SEARCH_RULES[site[0]]:
        match = pattern.search(text)
        observed.append((text, None if match is None else match.group(0)))
    assert tuple(observed) == SEARCH_RULES[site[0]]


def test_bridge_worker_bounds_identifiers_at_the_identifier_length() -> None:
    worker = importlib.import_module("hermes_realtime.integration.worker")
    bound = worker._MAX_IDENTIFIER_CHARS
    assert worker._require_exact_string("a" * 128, "event_id", max_chars=bound) == "a" * 128
    with pytest.raises(worker.BridgeProtocolError):
        worker._require_exact_string("a" * 129, "event_id", max_chars=bound)


def test_noncompleted_result_is_shortened_to_the_legacy_result_bound() -> None:
    projection = importlib.import_module("hermes_realtime.conversation.result_projection")
    bounded = projection._bounded_result

    assert bounded("x" * 1024) == "x" * 1024
    assert bounded("x" * 1025) == "x" * 1021 + "..."
    late_space = "x" * 512 + " " + "y" * 600
    assert bounded(late_space) == "x" * 512 + "..."
    early_space = "x" * 511 + " " + "y" * 600
    assert bounded(early_space) == early_space[:1021] + "..."
    assert len(bounded(early_space)) == 1024
