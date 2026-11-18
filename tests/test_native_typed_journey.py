"""Offline contracts for the optional typed native-journey client."""

import asyncio
import contextlib
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from scripts.native_run_witness import NativeRunWitness, validate_origin
from scripts.native_typed_journey import EventTracker, Journey, load_scenario, main, read_journal


def test_scenario_accepts_followup_without_requiring_duplicate_work(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "steps": [
                    {
                        "label": "lookup",
                        "kind": "dispatch",
                        "text": "Synthetic lookup",
                        "wait": "accepted",
                    },
                    {
                        "label": "status",
                        "kind": "status",
                        "task": "lookup",
                        "text": "How is it going?",
                    },
                    {
                        "label": "recall",
                        "kind": "recall",
                        "task": "lookup",
                        "text": "What did it find?",
                    },
                ],
            }
        )
    )
    assert len(load_scenario(path)) == 3


def test_scenario_accepts_standalone_conversation_and_rejects_work_fields(tmp_path):
    path = tmp_path / "scenario.json"
    step = {"label": "speech", "kind": "conversation", "text": "Say this exact sentence."}
    value = {"version": 1, "steps": [step]}
    path.write_text(json.dumps(value), encoding="utf-8")
    assert load_scenario(path) == [step]
    for field, extra in (("task", "lookup"), ("wait", "completed"), ("tool", "web_search")):
        value["steps"] = [step | {field: extra}]
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match="journey_conversation_shape"):
            load_scenario(path)


@pytest.mark.parametrize("unexpected", [None, "task", "run"])
def test_conversation_binds_timing_without_admitting_work(tmp_path, monkeypatch, unexpected):
    import scripts.native_typed_journey as client

    options = SimpleNamespace(preview_dir=tmp_path)
    journey = Journey(options, {"expiresInSeconds": 3600}, "secret")
    runs = {"run_" + "a" * 32} if unexpected == "run" else set()
    monkeypatch.setattr(client, "read_journal", lambda _: (runs, 0))
    monkeypatch.setattr(
        client, "NativeRunWitness", lambda *a, **kw: pytest.fail("ordinary turn read native run")
    )
    submitted = []
    text = 'Say exactly "A synthetic sentence."'

    async def request(route, payload=None):
        if route == "input":
            submitted.append(payload)
            return {}
        assert route == "events"
        batch = [
            event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
            event(2, "transcript_final", role="user", text=text),
            event(3, "first_foreground_token", turnId="opaque_A", turnGeneration=1),
            event(4, "assistant_text_generated", turnId="opaque_A", turnGeneration=1),
            event(5, "first_playable_audio", turnId="opaque_A", turnGeneration=1),
            event(6, "assistant_turn_completed", turnId="opaque_A", turnGeneration=1,
                  inputSequence=1, bindingGeneration=7),
        ]
        if unexpected == "task":
            batch.append(event(7, "task_state", taskId="task_unexpected", status="active"))
        for row in batch:
            row["monotonicMs"] = journey.tracker.started + row["sequence"]
        return {"version": 2, "events": batch}

    journey.request = request
    operation = journey.step({"label": "speech", "kind": "conversation", "text": text})
    if unexpected is not None:
        with pytest.raises(ValueError, match="journey_followup_dispatched"):
            asyncio.run(operation)
    else:
        asyncio.run(operation)
        assert journey.tracker.current["outcome"] == "completed"
        assert journey.tracker.current["tokenToPlayableMs"] == 2
        assert journey.bindings == {} and journey.results == {}
        assert text not in json.dumps(journey.tracker.receipt())
    assert submitted == [{"sequence": 1, "text": text}]


def event(sequence, kind, **data):
    return {"sequence": sequence, "kind": kind, "monotonicMs": 10.5 + sequence, "data": data}


@pytest.mark.parametrize("terminal_first", [False, True])
def test_pretext_interruption_requires_exact_admitted_input(terminal_first):
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic text")
    admitted = event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7)
    terminal = event(2, "assistant_turn_interrupted", turnId="actual_turn", turnGeneration=3,
                     inputSequence=1, bindingGeneration=7)
    rows = [terminal, admitted] if terminal_first else [admitted, terminal]
    for index, row in enumerate(rows, 1):
        row["sequence"] = index
    tracker.consume(rows)
    assert tracker.current["outcome"] == "interrupted"
    assert tracker.current["firstTokenMs"] is None
    assert tracker.pair == ("actual_turn", 3)
    assert "actual_turn" not in json.dumps(tracker.receipt())


@pytest.mark.parametrize("correlation", [
    {}, {"inputSequence": 2, "bindingGeneration": 7},
    {"inputSequence": 1, "bindingGeneration": 8},
    {"inputSequence": True, "bindingGeneration": 7},
    {"inputSequence": 1, "bindingGeneration": True},
])
def test_unbound_interruption_cannot_bind_current_input(correlation):
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic text")
    tracker.consume([
        event(1, "typed_input_admitted", inputSequence=1,
              bindingGeneration=1 if type(correlation.get("bindingGeneration")) is bool else 7),
        event(2, "assistant_turn_interrupted", turnId="foreign_turn", turnGeneration=1,
              **correlation),
    ])
    assert tracker.current["outcome"] == "pending"
    assert tracker.pair is None
    assert tracker.current["unboundTerminalEvents"] == 1


def test_prior_turn_pair_cannot_bind_a_later_matching_input():
    tracker = EventTracker()
    tracker.begin("first", 1, "First input")
    tracker.consume([event(1, "assistant_turn_interrupted", turnId="stale_turn", turnGeneration=1)])
    tracker.begin("second", 2, "Second input")
    tracker.consume([
        event(2, "typed_input_admitted", inputSequence=2, bindingGeneration=7),
        event(3, "assistant_turn_interrupted", turnId="stale_turn", turnGeneration=1,
              inputSequence=2, bindingGeneration=7),
    ])
    assert tracker.current["outcome"] == "pending"
    assert tracker.pair is None


def test_cross_turn_terminal_cannot_replace_current_token_binding():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.consume([
        event(1, "transcript_final", role="user", text="Synthetic input"),
        event(2, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
        event(3, "first_foreground_token", turnId="current_turn", turnGeneration=1),
        event(4, "assistant_turn_interrupted", turnId="foreign_turn", turnGeneration=1,
              inputSequence=1, bindingGeneration=7),
    ])
    assert tracker.current["outcome"] == "pending"
    assert tracker.pair == ("current_turn", 1)


def test_missing_admission_binding_keeps_legacy_terminal_unbound():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.consume([
        event(1, "typed_input_admitted", inputSequence=1),
        event(2, "assistant_turn_interrupted", turnId="legacy_turn", turnGeneration=1),
    ])
    assert tracker.current["outcome"] == "pending"
    assert tracker.current["unboundTerminalEvents"] == 1


def test_correlated_completed_terminal_cannot_certify_silent_success():
    # A tokenless completion binds by admitted input lineage, never as spoken (#237);
    # spoken steps refuse it (test_tokenless_completion_still_cannot_certify_a_spoken_turn).
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.consume([
        event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
        event(2, "assistant_turn_completed", turnId="actual_turn", turnGeneration=1,
              inputSequence=1, bindingGeneration=7),
    ])
    assert tracker.current["outcome"] == "completed"
    assert tracker.current["terminalBinding"] == "admittedInput"
    tracker.consume([event(3, "transcript_final", role="user", text="Synthetic input"),
                     event(4, "first_foreground_token", turnId="actual_turn", turnGeneration=1)])
    assert tracker.current["terminalBinding"] == "admittedInput"


def test_modern_admission_preserves_legacy_bound_completion():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.consume([
        event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
        event(2, "transcript_final", role="user", text="Synthetic input"),
        event(3, "first_foreground_token", turnId="actual_turn", turnGeneration=1),
        event(4, "first_playable_audio", turnId="actual_turn", turnGeneration=1),
        event(5, "assistant_turn_completed", turnId="actual_turn", turnGeneration=1),
    ])
    assert tracker.current["outcome"] == "completed"


@pytest.mark.asyncio
async def test_dispatch_wait_refuses_matching_pretext_interruption_without_timeout(tmp_path):
    journey = Journey(SimpleNamespace(preview_dir=tmp_path), {"expiresInSeconds": 3600}, "secret")

    async def request(route, payload=None):
        if route == "input":
            return {}
        return {"version": 2, "events": [
            event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
            event(2, "assistant_turn_interrupted", turnId="actual_turn", turnGeneration=1,
                  inputSequence=1, bindingGeneration=7),
        ]}

    journey.request = request
    with pytest.raises(ValueError, match="journey_turn_interrupted"):
        await asyncio.wait_for(journey.step({"kind": "dispatch", "label": "work",
                                            "text": "Synthetic task", "wait": "accepted"}), 0.5)


@pytest.mark.asyncio
async def test_interrupted_dispatch_keeps_real_accepted_task_bound_for_cleanup(
    tmp_path, monkeypatch,
):
    import scripts.native_typed_journey as client

    run = "run_" + "a" * 32
    monkeypatch.setattr(client, "read_journal", lambda _: ({run}, 0))
    journey = Journey(SimpleNamespace(preview_dir=tmp_path), {"expiresInSeconds": 3600}, "secret")

    async def request(route, payload=None):
        if route == "input":
            return {}
        return {"version": 2, "events": [
            event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
            event(2, "task_state", taskId="task_accepted", status="active"),
            event(3, "assistant_turn_interrupted", turnId="actual_turn", turnGeneration=1,
                  inputSequence=1, bindingGeneration=7),
        ]}

    journey.request = request
    with pytest.raises(ValueError, match="journey_turn_interrupted"):
        await asyncio.wait_for(journey.step({"kind": "dispatch", "label": "work",
                                            "text": "Synthetic task", "wait": "accepted"}), 0.5)
    assert journey.bindings == {"work": ("task_accepted", run)}


def test_public_lifecycle_events_and_float_clock_do_not_fail_or_claim_warm_audio():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic text")
    tracker.consume(
        [
            event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
            event(2, "transcript_final", role="user", text="Synthetic text"),
            event(
                3,
                "assistant_text_generated",
                turnId="session_1_turn_1",
                turnGeneration=1,
                text="Synthetic answer",
            ),
            event(4, "first_playable_audio"),
            event(5, "assistant_turn_completed", turnId="session_1_turn_1", turnGeneration=1),
        ]
    )
    assert tracker.current["outcome"] == "pending"  # Old marker lacks input/turn binding.
    assert tracker.current["firstPlayableAudioMs"] is None
    assert "Synthetic" not in json.dumps(tracker.receipt())


def test_old_generation_and_background_mention_cannot_complete_current_turn():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic text")
    tracker.consume(
        [
            event(1, "transcript_final", role="user", text="Synthetic text"),
            event(2, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
            event(3, "first_foreground_token", turnId="opaque_A", turnGeneration=2),
            event(
                4,
                "assistant_text_generated",
                turnId="opaque_A",
                turnGeneration=2,
                text="Synthetic answer",
            ),
            event(5, "first_playable_audio", turnId="opaque_A", turnGeneration=1),
            event(6, "assistant_turn_completed", turnId="mention_1", turnGeneration=2),
            event(7, "assistant_turn_completed", turnId="opaque_A", turnGeneration=1),
        ]
    )
    assert tracker.current["outcome"] == "pending"
    assert tracker.current["firstPlayableAudioMs"] is None


def test_identity_bound_warm_marker_can_precede_generated_text():
    tracker = EventTracker()
    tracker.begin("warm", 1, "Synthetic text")
    batch = [
        event(1, "transcript_final", role="user", text="Synthetic text"),
        event(2, "typed_input_admitted", inputSequence=1),
        event(3, "first_foreground_token", turnId="opaque_A", turnGeneration=1),
        event(4, "assistant_text_generated", turnId="opaque_A", turnGeneration=1),
        event(5, "first_playable_audio", turnId="opaque_A", turnGeneration=1),
        event(6, "assistant_turn_completed", turnId="opaque_A", turnGeneration=1),
    ]
    for row in batch:
        row["monotonicMs"] = time.monotonic() * 1000
    tracker.consume(batch)
    assert tracker.current["firstTokenMs"] is not None
    assert tracker.current["firstPlayableAudioMs"] is not None
    assert tracker.current["tokenToPlayableMs"] is not None
    assert tracker.current["outcome"] == "completed"


def test_foreign_user_input_cannot_bind_the_current_turn():
    tracker = EventTracker()
    tracker.begin("current", 1, "Synthetic text")
    with pytest.raises(ValueError):
        tracker.consume([event(1, "transcript_final", role="user", text="Other input")])


@pytest.mark.parametrize(
    "origin", ["https://example.test", "http://127.0.0.1:1@evil.test", "http://127.0.0.1:1/path"]
)
def test_only_exact_loopback_origins(origin):
    with pytest.raises(ValueError):
        validate_origin(origin)


def test_journal_rejects_duplicates_and_oversize(tmp_path):
    path = tmp_path / "journal.json"
    path.write_text(json.dumps({"version": 2, "pending": [], "admitted": ["run_" + "a" * 32] * 2}))
    with pytest.raises(ValueError):
        read_journal(path)
    path.write_bytes(b" " * 65_537)
    with pytest.raises(ValueError):
        read_journal(path)


def witness(tmp_path, *, source="api_server", result_id="call_one", tool="web_search"):
    run = "run_" + "a" * 32
    db = sqlite3.connect(tmp_path / "state.db")
    db.executescript(
        "CREATE TABLE sessions(id TEXT,source TEXT);"
        "CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,"
        "content TEXT,tool_calls TEXT,tool_call_id TEXT,tool_name TEXT,active INTEGER);"
    )
    db.execute("INSERT INTO sessions VALUES (?,?)", (run, source))
    arguments = {"cmdline_chars": 0} if tool.startswith("mcp__") else {"query": "private query"}
    if tool == "web_extract":
        arguments = {"urls": ["https://private.test"]}
    function = {"name": tool, "arguments": json.dumps(arguments)}
    if tool.startswith("mcp__"):
        function = {
            "name": "tool_call",
            "arguments": json.dumps({"name": tool, "arguments": arguments}),
        }
    calls = json.dumps([{"id": "call_one", "function": function}])
    db.execute(
        "INSERT INTO messages VALUES(1,?,?,?,?,?,?,1)", (run, "assistant", None, calls, None, None)
    )
    result = {"success": True, "data": {"web": [{"url": "https://private.test"}]}}
    if tool.startswith("mcp__"):
        result = {
            "structuredContent": {
                "gpus": [{}],
                "observations": {"gpu": {}, "ollama": {}},
                "other_processes": [],
            }
        }
    db.execute(
        "INSERT INTO messages VALUES(2,?,?,?,?,?,?,1)",
        (run, "tool", json.dumps(result), None, result_id, tool),
    )
    db.commit()
    db.close()
    subject = NativeRunWitness("secret", run, profile=tmp_path, api_origin="http://127.0.0.1:1")
    subject._status = lambda: {
        "object": "hermes.run",
        "run_id": run,
        "session_id": run,
        "status": "completed",
    }
    return subject


def test_durable_web_pair_is_noninterfering_and_metadata_only(tmp_path):
    receipt = witness(tmp_path).read()
    assert receipt["pairSuccess"] and receipt["toolRecordCount"] == 2
    assert receipt["toolEventCount"] == 0
    assert "private" not in json.dumps(receipt) and "secret" not in json.dumps(receipt)


def test_bridge_status_pair_proves_zero_commandline_request(tmp_path):
    receipt = witness(tmp_path, tool="mcp__vram__vram_status").read()
    assert receipt["toolCalls"][0]["zeroCommandlineRequest"] is True


def replace_web_result(tmp_path, value, *, urls=None):
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("UPDATE messages SET content=? WHERE role='tool'", (json.dumps(value),))
        if urls is not None:
            raw = db.execute("SELECT tool_calls FROM messages WHERE id=1").fetchone()[0]
            calls = json.loads(raw)
            calls[0]["function"]["arguments"] = json.dumps({"urls": urls})
            db.execute("UPDATE messages SET tool_calls=? WHERE id=1", (json.dumps(calls),))


def test_empty_search_placeholder_is_operational_not_usable_evidence(tmp_path):
    subject = witness(tmp_path)
    replace_web_result(tmp_path, {"success": True, "data": {"web": [
        {"url": "", "title": "", "description": "", "position": 1},
    ]}})
    receipt = subject.read()
    row = receipt["toolCalls"][0]
    assert receipt["pairSuccess"] and row["successfulResults"] == row["pairedResults"] == 1
    assert row["usableResults"] == row["usableRows"] == 0
    assert row["emptyRows"] == row["unboundRows"] == 1


@pytest.mark.parametrize("url,text,error", [
    ("https://example.test/source", "Synthetic snippet", None),
    ("", "Synthetic snippet", None),
    ("file:///synthetic", "Synthetic snippet", None),
    ("http://[invalid", "Synthetic snippet", None),
    ("https://example.test/source", "  ", None),
    ("https://example.test/source", None, None),
    ("https://example.test/source", "Synthetic snippet", "Synthetic error"),
])
def test_search_usable_evidence_requires_url_description_and_no_item_error(
    tmp_path, url, text, error,
):
    subject = witness(tmp_path)
    replace_web_result(tmp_path, {"success": True, "data": {"web": [
        {"url": url, "title": "Title alone is insufficient", "description": text, "error": error},
    ]}})
    row = subject.read()["toolCalls"][0]
    expected = int(url == "https://example.test/source"
                   and bool(text and text.strip()) and not error)
    assert row["successfulResults"] == 1  # Preserve legacy shape-success meaning.
    assert row["usableResults"] == row["usableRows"] == expected
    assert row["errorRows"] == int(bool(error))
    assert "Synthetic" not in json.dumps(row) and "example.test" not in json.dumps(row)


def test_call_error_prevents_usable_evidence_despite_nonempty_rows(tmp_path):
    subject = witness(tmp_path)
    replace_web_result(tmp_path, {"success": False, "error": "Synthetic call error", "data": {
        "web": [{"url": "https://example.test/source", "description": "Synthetic snippet"}],
    }})
    row = subject.read()["toolCalls"][0]
    assert row["successfulResults"] == row["usableResults"] == row["usableRows"] == 0
    assert row["pairedResults"] == row["errorPresent"] == 1


def test_partial_extract_reports_rows_without_changing_legacy_validity(tmp_path):
    subject = witness(tmp_path, tool="web_extract")
    first, second = "https://example.test/first", "https://example.test/second"
    replace_web_result(tmp_path, {"results": [
        {"url": first, "content": "Synthetic body", "error": None},
        {"url": second, "content": "", "error": "Synthetic item error"},
    ]}, urls=[first, second])
    receipt = subject.read()
    row = receipt["toolCalls"][0]
    assert not receipt["pairSuccess"] and row["successfulResults"] == 0
    assert row["pairedResults"] == row["errorPresent"] == 1
    assert row["usableResults"] == 0
    assert row["usableRows"] == row["errorRows"] == row["emptyRows"] == 1
    assert row["resultRows"] == 2 and row["unboundRows"] == 0


def test_extract_success_and_usable_rows_from_different_results_cannot_combine(tmp_path):
    subject = witness(tmp_path, tool="web_extract")
    replace_web_result(tmp_path, {"results": [
        {"url": "file:///synthetic", "content": "Synthetic body", "error": None},
    ]}, urls=["file:///synthetic"])
    urls = ["https://example.test/first", "https://example.test/second"]
    call = [{"id": "call_two", "function": {
        "name": "web_extract", "arguments": json.dumps({"urls": urls}),
    }}]
    result = {"results": [
        {"url": urls[0], "content": "Synthetic body", "error": None},
        {"url": urls[1], "content": "", "error": "Synthetic failure"},
    ]}
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("INSERT INTO messages VALUES(3,?,?,?,?,?,?,1)",
                   (subject.run_id, "assistant", None, json.dumps(call), None, None))
        db.execute("INSERT INTO messages VALUES(4,?,?,?,?,?,?,1)",
                   (subject.run_id, "tool", json.dumps(result), None, "call_two", "web_extract"))
    receipt = subject.read()
    row = receipt["toolCalls"][0]
    assert row["pairedResults"] == 2 and row["successfulResults"] == row["usableRows"] == 1
    assert row["usableResults"] == 0 and not receipt["pairSuccess"]
    assert row["resultRows"] == 3 and row["errorRows"] == row["emptyRows"] == 1


def test_extract_without_requested_urls_keeps_legacy_success_but_no_usable_binding(tmp_path):
    subject = witness(tmp_path, tool="web_extract")
    replace_web_result(tmp_path, {"results": [
        {"url": "https://example.test/source", "content": "Synthetic body", "error": None},
    ]})
    with sqlite3.connect(tmp_path / "state.db") as db:
        calls = json.loads(db.execute("SELECT tool_calls FROM messages WHERE id=1").fetchone()[0])
        calls[0]["function"]["arguments"] = "{}"
        db.execute("UPDATE messages SET tool_calls=? WHERE id=1", (json.dumps(calls),))
    row = subject.read()["toolCalls"][0]
    assert row["successfulResults"] == 1
    assert row["usableResults"] == row["usableRows"] == 0 and row["unboundRows"] == 1


@pytest.mark.parametrize("returned", [
    ["https://example.test/first"],
    ["https://example.test/first", "https://foreign.test/second"],
])
def test_extract_usable_evidence_keeps_exact_requested_url_set(tmp_path, returned):
    subject = witness(tmp_path, tool="web_extract")
    replace_web_result(tmp_path, {"results": [
        {"url": url, "content": "Synthetic body", "error": None} for url in returned
    ]}, urls=["https://example.test/first", "https://example.test/second"])
    row = subject.read()["toolCalls"][0]
    assert row["successfulResults"] == row["usableResults"] == row["usableRows"] == 0
    assert row["unboundRows"] == len(returned)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,successful,usable,accepted", [
    ("web_search", 1, 0, False), ("web_search", 1, 1, True),
    ("web_extract", 1, 0, False),
    ("web_extract", 0, 1, False), ("web_extract", 1, 1, True),
    ("mcp__vram__vram_status", 1, None, True),
])
async def test_web_tool_assertion_requires_both_legacy_success_and_usable_evidence(
    tmp_path, monkeypatch, tool, successful, usable, accepted,
):
    import scripts.native_typed_journey as client

    run = "run_" + "a" * 32
    monkeypatch.setattr(client, "read_journal", lambda _: ({run}, 0))
    journey = Journey(SimpleNamespace(preview_dir=tmp_path), {"expiresInSeconds": 3600}, "secret")

    async def request(route, payload=None):
        if route == "input":
            return {}
        return {"version": 2, "events": [
            event(1, "task_state", taskId="task_evidence", status="active"),
            event(2, "task_result", taskId="task_evidence", status="completed"),
        ]}

    async def collect(label):
        row = {"tool": tool, "successfulResults": successful}
        if usable is not None:
            row["usableResults"] = usable
        journey.results[label] = {"terminal": "completed", "toolCalls": [row]}

    journey.request, journey.collect = request, collect
    operation = journey.step({"kind": "dispatch", "label": "evidence",
                              "text": "Synthetic task", "tool": tool})
    if accepted:
        await operation
    else:
        with pytest.raises(ValueError, match="journey_tool_unproven"):
            await operation


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"results": [{"error": "failed", "content": ""}]},
        {"results": [{"url": "https://foreign.test", "content": "Text", "error": None}]},
    ],
)
def test_extract_missing_failed_or_unbound_content_is_not_success(tmp_path, result):
    subject = witness(tmp_path, tool="web_extract")
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("UPDATE messages SET content=? WHERE role='tool'", (json.dumps(result),))
    db.close()
    receipt = subject.read()
    assert receipt["toolCalls"][0]["successfulResults"] == 0
    assert not receipt["pairSuccess"]


def test_native_extract_inline_results_without_success_flag_are_accepted(tmp_path):
    subject = witness(tmp_path, tool="web_extract")
    result = {
        "results": [
            {
                "url": "https://private.test",
                "title": "Synthetic title",
                "content": "Synthetic extracted body",
                "error": None,
            }
        ]
    }
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("UPDATE messages SET content=? WHERE role='tool'", (json.dumps(result),))
    db.close()
    receipt = subject.read()
    assert receipt["toolCalls"][0]["successfulResults"] == 1
    assert receipt["toolCalls"][0]["usableResults"] == receipt["toolCalls"][0]["usableRows"] == 1
    assert receipt["pairSuccess"]


@pytest.mark.parametrize("drift", ["source", "session", "pair", "bound"])
def test_native_witness_refuses_unbound_or_oversized_evidence(tmp_path, drift):
    subject = witness(
        tmp_path,
        source="foreign" if drift == "source" else "api_server",
        result_id="foreign" if drift == "pair" else "call_one",
    )
    if drift == "session":
        subject._status = lambda: {
            "object": "hermes.run",
            "run_id": subject.run_id,
            "session_id": "foreign",
            "status": "completed",
        }
    if drift == "bound":
        with sqlite3.connect(tmp_path / "state.db") as db:
            oversized = json.dumps({"success": True, "data": {"web": []}, "padding": "x" * 262_145})
            db.execute("UPDATE messages SET content=? WHERE role='tool'", (oversized,))
        db.close()
    with pytest.raises(ValueError):
        subject.read()


def test_failed_result_tool_binding_cannot_be_reported_as_status_success(tmp_path):
    subject = witness(tmp_path, tool="mcp__vram__vram_status")
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("UPDATE messages SET tool_name='tool_call' WHERE role='tool'")
    db.close()
    with pytest.raises(ValueError, match="witness_result_tool_binding"):
        subject.read()


def test_fast_completed_work_then_status_does_not_require_a_second_dispatch(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    run, task = "run_" + "a" * 32, "task_synthetic"
    options = SimpleNamespace(
        preview_dir=tmp_path, native_profile=tmp_path, api_origin="http://127.0.0.1:1"
    )
    journey = Journey(options, {"expiresInSeconds": 3600}, "secret")
    monkeypatch.setattr(client, "read_journal", lambda _: ({run}, 0))
    monkeypatch.setattr(
        client,
        "NativeRunWitness",
        lambda *a, **kw: SimpleNamespace(
            read=lambda: {"terminal": "completed", "toolCalls": [], "pairSuccess": False}
        ),
    )
    batches = iter(
        [
            [
                event(1, "task_state", taskId=task, status="active"),
                event(2, "task_result", taskId=task, status="completed"),
            ],
            [
                event(3, "transcript_final", role="user", text="Status please"),
                event(4, "typed_input_admitted", inputSequence=2, bindingGeneration=7),
                event(5, "first_foreground_token", turnId="opaque_B", turnGeneration=2),
                event(6, "assistant_text_generated", turnId="opaque_B", turnGeneration=2),
                event(7, "assistant_turn_completed", turnId="opaque_B", turnGeneration=2,
                      inputSequence=2, bindingGeneration=7),
            ],
        ]
    )

    async def request(route, payload=None):
        return {"version": 2, "events": next(batches)} if route == "events" else {}

    journey.request = request

    async def scenario():
        await journey.step({"kind": "dispatch", "label": "lookup", "text": "Synthetic lookup"})
        await journey.step(
            {"kind": "status", "label": "status", "task": "lookup", "text": "Status please"}
        )

    asyncio.run(scenario())
    assert len(journey.bindings) == 1 and journey.tracker.current["outcome"] == "completed"


def test_preflight_never_contacts_runtime_and_existing_receipt_is_preserved(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    monkeypatch.setattr(
        client, "request_json", lambda *a, **kw: pytest.fail("network in preflight")
    )
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps(
            {
                "version": 1,
                "steps": [
                    {"kind": "dispatch", "label": "synthetic", "text": "Private synthetic input"}
                ],
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    argv = [
        "--preview-dir",
        str(tmp_path),
        "--native-profile",
        str(tmp_path),
        "--scenario",
        str(scenario),
        "--receipt",
        str(receipt),
        "--browser-origin",
        "http://127.0.0.1:1",
        "--api-origin",
        "http://127.0.0.1:2",
        "--livekit-url",
        "ws://127.0.0.1:3",
    ]
    assert main(argv) == 0
    saved = receipt.read_bytes()
    assert b"Private synthetic input" not in saved and b"preflight_only" in saved
    assert main(argv) == 1
    assert receipt.read_bytes() == saved


def test_bootstrap_binding_mismatch_stops_only_the_just_issued_lease(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client
    from hermes_realtime import host_launcher

    options = SimpleNamespace(
        preview_dir=tmp_path,
        native_profile=tmp_path,
        browser_origin="http://127.0.0.1:1",
        livekit_url="ws://127.0.0.1:2",
    )
    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))
    monkeypatch.setattr(host_launcher, "_load_api_bearer", lambda _: "private-native-bearer")
    requests = []

    def request(origin, route, **kwargs):
        requests.append((origin, route, kwargs.get("token")))
        if route == "stable-bootstrap":
            return 200, {"workerIdentity": "mismatched", "token": "private-issued-lease"}
        assert route == "stop"
        return 200, {}

    monkeypatch.setattr(client, "request_json", request)
    receipt = {"outcome": "refused"}
    with pytest.raises(ValueError, match="journey_bootstrap_binding"):
        asyncio.run(client.execute(options, [], receipt))
    assert requests == [
        (options.browser_origin, "stable-bootstrap", None),
        (options.browser_origin, "stop", "private-issued-lease"),
    ]
    assert receipt["cleanup"] == {
        "ownedTasksSettled": True,
        "sessionStopped": True,
        "roomDisconnected": True,
    }
    assert "private-" not in json.dumps(receipt)


def test_missing_native_env_fails_before_bootstrap_network(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    options = SimpleNamespace(
        preview_dir=tmp_path, native_profile=tmp_path, browser_origin="http://127.0.0.1:1"
    )
    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))
    monkeypatch.setattr(client, "request_json", lambda *a, **kw: pytest.fail("network before env"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(client.execute(options, [], {"outcome": "refused"}))


def test_fresh_preview_without_run_record_has_no_recorded_admissions(tmp_path):
    # The native adapter creates its journal only when its task session starts.
    assert read_journal(tmp_path / "hermes-runs-v1.json") == (set(), 0)


def test_client_reads_version_three_admission_journal_without_changing_authority(tmp_path):
    path = tmp_path / "hermes-runs-v1.json"
    run_id = "run_" + "a" * 32
    path.write_text(json.dumps({"version": 3, "pending": [], "admitted": [run_id]}))
    assert read_journal(path) == ({run_id}, 0)


@pytest.mark.asyncio
async def test_client_requires_v2_event_response_without_downgrade(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))
    journey = Journey(SimpleNamespace(preview_dir=tmp_path), {"expiresInSeconds": 3600}, "secret")

    async def request(route, payload=None):
        assert route == "events"
        return {"version": 1, "events": []}

    journey.request = request
    with pytest.raises(ValueError, match="journey_event_version"):
        await journey.poll()


@pytest.mark.parametrize("route,version", [("events", 2), ("input", 1), ("media", 1)])
def test_client_uses_versioned_events_without_changing_other_routes(monkeypatch, route, version):
    import scripts.native_typed_journey as client

    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return b"{}"

    def opened(request, **_kwargs):
        requests.append(request.full_url)
        return Response()

    monkeypatch.setattr(
        client.urllib.request, "build_opener", lambda *_: SimpleNamespace(open=opened),
    )
    assert client.request_json("http://127.0.0.1:1", route) == (200, {})
    assert requests == [f"http://127.0.0.1:1/api/v{version}/{route}"]


@pytest.mark.parametrize("name", [[], {}], ids=["list", "object"])
def test_unhashable_native_tool_name_retains_refusal_category(tmp_path, capsys, name):
    subject = witness(tmp_path)
    calls = [
        {
            "id": "call_one",
            "function": {"name": "tool_call", "arguments": json.dumps({"name": name})},
        }
    ]
    with sqlite3.connect(tmp_path / "state.db") as db:
        db.execute("UPDATE messages SET tool_calls=? WHERE id=1", (json.dumps(calls),))
    with pytest.raises(ValueError, match="witness_tool_identity"):
        subject.read()
    assert "witness_tool_identity" in capsys.readouterr().out


@pytest.mark.parametrize(
    "historical", ["completed", "failed", "interrupted", "active", "cancelling"],
)
@pytest.mark.parametrize("journal", ["empty", "admitted", "pending"])
def test_initial_baseline_allows_only_terminal_history_and_an_empty_native_journal(
    tmp_path, monkeypatch, historical, journal,
):
    import hashlib
    import sys

    import scripts.native_typed_journey as client
    from hermes_realtime import host_launcher
    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              browser_origin="http://127.0.0.1:1",
                              livekit_url="ws://127.0.0.1:2")
    suffix = hashlib.sha256(str(tmp_path / "active.private.json").encode()).hexdigest()[:16]
    credential = dict(
        workerIdentity="worker_preview_" + suffix, roomName="hermes-preview-" + suffix,
        url=options.livekit_url, token="synthetic-lease", participantIdentity="synthetic",
        expiresInSeconds=3600,
    )
    class Room:
        async def connect(self, *args, **kwargs):
            pass
        async def disconnect(self):
            pass
    monkeypatch.setitem(sys.modules, "livekit", SimpleNamespace(rtc=SimpleNamespace(
        Room=Room, RoomOptions=lambda **kwargs: kwargs)))
    monkeypatch.setattr(host_launcher, "_load_api_bearer", lambda _: "synthetic")
    reads = []
    def read(_):
        reads.append(True)
        if len(reads) == 1 or journal == "empty":
            return set(), 0
        return ({"run_" + "a" * 32}, 0) if journal == "admitted" else (set(), 1)
    monkeypatch.setattr(client, "read_journal", read)
    def request(origin, route, **kwargs):
        if route == "stable-bootstrap":
            return 200, credential
        if route == "events":
            return 200, {"version": 2, "events": [
                event(1, "task_state", taskId="task_history", status=historical),
            ]}
        assert route == "stop"
        return 200, {}
    monkeypatch.setattr(client, "request_json", request)
    started = []
    async def step(self, value):
        started.append(dict(self.tracker.tasks))
    monkeypatch.setattr(client.Journey, "step", step)
    receipt = {"outcome": "refused"}
    operation = client.execute(options, [{"label": "synthetic"}], receipt)
    if historical in client.TERMINAL and journal == "empty":
        asyncio.run(operation)
        assert started == [{"task_history": historical}]
        assert receipt["acceptedDispatches"] == 0
        assert receipt["outcome"] == "completed"
    else:
        with pytest.raises(ValueError, match="journey_existing_task"):
            asyncio.run(operation)
        assert started == []


@pytest.mark.parametrize("prior", [None, "active", "completed", "failed", "interrupted"])
@pytest.mark.parametrize("identity", [None, "task_history", "task_other"])
def test_rejected_task_control_never_changes_task_lifecycle_or_admission(prior, identity):
    tracker = EventTracker()
    if prior is not None:
        tracker.tasks["task_history"] = prior
    before = dict(tracker.tasks)
    tracker.consume([event(1, "task_state", taskId=identity, status="rejected")])
    assert tracker.tasks == before


@pytest.mark.parametrize("kind,status,identity", [
    ("task_result", "rejected", "task_valid"),
    ("task_state", "unknown", "task_valid"),
    ("task_state", "cancelled", "task_valid"),
    ("task_state", "rejected", "invalid"),
    ("task_state", "rejected", "task_"),
    ("task_state", "rejected", 1),
    ("task_state", "rejected", True),
    ("task_state", "rejected", "task_" + "a" * 124),
])
def test_task_control_compatibility_keeps_unknown_status_and_id_guards(kind, status, identity):
    tracker = EventTracker()
    with pytest.raises(ValueError, match="journey_task_identity"):
        tracker.consume([event(1, kind, taskId=identity, status=status)])
    assert tracker.tasks == {}


@pytest.mark.asyncio
async def test_historical_terminal_map_entry_never_counts_as_a_new_dispatch(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client
    journey = Journey(
        SimpleNamespace(preview_dir=tmp_path), {"expiresInSeconds": 3600}, "synthetic",
    )
    journey.tracker.tasks["task_history"] = "completed"
    run = "run_" + "b" * 32
    monkeypatch.setattr(client, "read_journal", lambda _: ({run}, 0))
    async def request(route, payload=None):
        if route == "input":
            return {}
        return {"version": 2, "events": [
            event(1, "task_state", taskId="task_new", status="active"),
        ]}
    journey.request = request
    await journey.step({"label": "new", "kind": "dispatch", "text": "Synthetic work",
                        "wait": "accepted"})
    assert journey.bindings == {"new": ("task_new", run)}
    assert journey.tracker.tasks["task_history"] == "completed"


def os_error_options(tmp_path):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"version": 1, "steps": [
        {"kind": "conversation", "label": "synthetic", "text": "Synthetic fixture"},
    ]}))
    receipt = tmp_path / "receipt.json"
    return scenario, receipt, [
        "--preview-dir", str(tmp_path), "--native-profile", str(tmp_path),
        "--scenario", str(scenario), "--receipt", str(receipt),
        "--browser-origin", "http://127.0.0.1:1", "--api-origin", "http://127.0.0.1:2",
        "--livekit-url", "ws://127.0.0.1:3",
    ]


@pytest.mark.parametrize("role", ["scenario", "journal"])
def test_main_os_error_receipt_identifies_trusted_read_context_without_private_data(
    tmp_path, monkeypatch, capsys, role,
):
    import inspect
    from pathlib import Path

    import scripts.native_typed_journey as client
    scenario, receipt, argv = os_error_options(tmp_path)
    journal = tmp_path / "state" / "hermes-runs-v1.json"
    target = scenario if role == "scenario" else journal
    error = PermissionError(13, "SYNTHETIC_PRIVATE_MESSAGE", "SYNTHETIC_PRIVATE_PATH")
    error.winerror = 5
    original = Path.open
    attempts = []
    def open_file(path, *args, **kwargs):
        if path == target:
            attempts.append(True)
            raise error
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    if role == "journal":
        async def execute(*args):
            client.read_journal(journal)
        monkeypatch.setattr(client, "execute", execute)
        argv.append("--execute")
    assert client.main(argv) == 1
    assert attempts == [True]
    output = capsys.readouterr()
    saved = json.loads(receipt.read_text()) if receipt.exists() else json.loads(
        output.out.removeprefix("[native-typed-journey] "))
    assert saved["failure"] == "PermissionError" and saved["outcome"] == "refused"
    metadata = saved["osError"]
    assert metadata == {
        "sourceFunction": "read_json", "sourceLine": metadata["sourceLine"],
        "pathRole": role, "exceptionKind": "PermissionError", "errno": 13, "winerror": 5,
    }
    lines, first = inspect.getsourcelines(client.read_json)
    assert metadata["sourceLine"] == first + next(
        i for i, line in enumerate(lines) if 'with path.open("rb")' in line)
    serialized = json.dumps(saved) + output.out + output.err
    assert "SYNTHETIC_PRIVATE_MESSAGE" not in serialized
    assert "SYNTHETIC_PRIVATE_PATH" not in serialized
    assert str(tmp_path) not in serialized


def test_passive_os_metadata_keeps_original_reader_exception(tmp_path, monkeypatch):
    from pathlib import Path

    import scripts.native_typed_journey as client
    error = PermissionError(13, "SYNTHETIC_PRIVATE_MESSAGE")
    def open_file(*args, **kwargs):
        raise error
    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(PermissionError) as caught:
        client.read_json(tmp_path / "synthetic")
    assert caught.value is error


def test_os_diagnostic_failure_cannot_mask_existing_outcome_or_cleanup(
    tmp_path, monkeypatch, capsys,
):
    import scripts.native_typed_journey as client
    _, receipt, argv = os_error_options(tmp_path)
    error = PermissionError(13, "SYNTHETIC_PRIVATE_MESSAGE")
    async def execute(options, steps, result):
        try:
            raise error
        finally:
            result["cleanup"] = {"syntheticSettled": True}
    def failed_metadata(_):
        raise RuntimeError("SYNTHETIC_DIAGNOSTIC_FAILURE")
    monkeypatch.setattr(client, "execute", execute)
    monkeypatch.setattr(client, "_os_error_evidence", failed_metadata)
    assert client.main([*argv, "--execute"]) == 1
    saved = json.loads(receipt.read_text())
    assert saved["failure"] == "PermissionError" and saved["outcome"] == "refused"
    assert saved["cleanup"] == {"syntheticSettled": True}
    assert "osError" not in saved
    assert "SYNTHETIC" not in capsys.readouterr().out


@pytest.mark.parametrize("code", [None, True, "SYNTHETIC_PRIVATE_VALUE", 1.5, -(2**31)-1, 2**32])
def test_os_diagnostic_codes_are_exact_bounded_integers(code):
    import scripts.native_typed_journey as client
    error = OSError("SYNTHETIC_PRIVATE_MESSAGE")
    error.errno = code
    error.winerror = code
    metadata = client._os_error_evidence(error)
    assert metadata == {"sourceFunction": "other", "sourceLine": None, "pathRole": "other",
                        "exceptionKind": "OSError", "errno": None, "winerror": None}
    assert "SYNTHETIC" not in json.dumps(metadata)


@pytest.mark.parametrize("code", [-(2**31), 0, 2**32-1])
def test_os_diagnostic_retains_supported_integer_scalars(code):
    import scripts.native_typed_journey as client
    error = OSError("SYNTHETIC_PRIVATE_MESSAGE")
    error.errno = error.winerror = code
    metadata = client._os_error_evidence(error)
    assert metadata["errno"] == metadata["winerror"] == code


@pytest.mark.parametrize("failure", [False, True])
def test_success_and_non_os_failure_have_no_os_metadata(tmp_path, monkeypatch, capsys, failure):
    import scripts.native_typed_journey as client
    _, receipt, argv = os_error_options(tmp_path)
    if failure:
        async def execute(*args):
            raise ValueError("journey_synthetic")
        monkeypatch.setattr(client, "execute", execute)
        argv.append("--execute")
    assert client.main(argv) == int(failure)
    saved = json.loads(receipt.read_text())
    assert "osError" not in saved
    assert saved["outcome"] == ("refused" if failure else "preflight_only")
    assert saved.get("failure") == ("journey_synthetic" if failure else None)


def test_unknown_os_subclass_and_wrapper_names_do_not_enter_metadata():
    import scripts.native_typed_journey as client
    class SyntheticPrivateException(OSError):
        pass
    def synthetic_private_wrapper():
        raise SyntheticPrivateException(13, "SYNTHETIC_PRIVATE_MESSAGE", "SYNTHETIC_PRIVATE_PATH")
    try:
        synthetic_private_wrapper()
    except OSError as error:
        metadata = client._os_error_evidence(error)
    assert metadata["sourceFunction"] == "other" and metadata["sourceLine"] is None
    assert metadata["exceptionKind"] == "OSError" and metadata["pathRole"] == "other"
    assert "synthetic" not in json.dumps(metadata).lower()


def test_optional_os_metadata_cannot_overflow_an_existing_fitting_receipt(
    tmp_path, monkeypatch, capsys,
):
    import scripts.native_typed_journey as client
    _, receipt, argv = os_error_options(tmp_path)
    baseline = {"version": 1, "outcome": "refused", "executed": True, "stepCount": 1,
                "failure": "PermissionError", "padding": ""}
    padding = "x" * (65536 - len(json.dumps(baseline, sort_keys=True)))
    async def execute(options, steps, result):
        result["padding"] = padding
        raise PermissionError(13, "SYNTHETIC_PRIVATE_MESSAGE")
    monkeypatch.setattr(client, "execute", execute)
    assert client.main([*argv, "--execute"]) == 1
    saved = json.loads(receipt.read_text())
    assert saved == baseline | {"padding": padding}
    assert "osError" not in saved
    assert len(json.dumps(saved, sort_keys=True)) == 65536
    assert "SYNTHETIC" not in capsys.readouterr().out


def test_optional_os_metadata_never_relaxes_existing_receipt_bound(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client
    _, _, argv = os_error_options(tmp_path)
    async def execute(options, steps, result):
        result["padding"] = "x" * 65536
        raise PermissionError(13, "SYNTHETIC_PRIVATE_MESSAGE")
    monkeypatch.setattr(client, "execute", execute)
    with pytest.raises(ValueError, match="journey_receipt_bound"):
        client.main([*argv, "--execute"])


def test_passive_publication_intervals_use_only_first_current_pair():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.started = 1000.0
    batch = [
        event(1, "transcript_final", role="user", text="Synthetic input"),
        event(2, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
        event(3, "first_foreground_token", turnId="current", turnGeneration=1),
        event(4, "assistant_text_generated", turnId="foreign", turnGeneration=1),
        event(5, "assistant_text_generated", turnId="current", turnGeneration=1),
        event(6, "assistant_text_generated", turnId="current", turnGeneration=1),
        event(7, "first_playable_audio", turnId="current", turnGeneration=1),
        event(8, "assistant_turn_completed", turnId="current", turnGeneration=1),
    ]
    for row, offset in zip(batch, [1, 2, 10, 20, 70, 80, 120, 130], strict=True):
        row["monotonicMs"] = tracker.started + offset
    tracker.consume(batch)
    assert tracker.current["firstPublicationMs"] == 70
    assert tracker.current["tokenToPublicationMs"] == 60
    assert tracker.current["publicationToPlayableMs"] == 50
    assert tracker.current["tokenToPlayableMs"] == 110
    assert tracker.current["assistantSegments"] == 2
    assert tracker.current["outcome"] == "completed"
    assert set(tracker.markers) == {("current", 1)}
    assert "Synthetic input" not in json.dumps(tracker.receipt())


def test_missing_publication_does_not_change_existing_completion():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.consume([
        event(1, "transcript_final", role="user", text="Synthetic input"),
        event(2, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
        event(3, "first_foreground_token", turnId="current", turnGeneration=1),
        event(4, "first_playable_audio", turnId="current", turnGeneration=1),
        event(5, "assistant_turn_completed", turnId="current", turnGeneration=1),
    ])
    assert tracker.current["outcome"] == "completed"
    for key in ("firstPublicationMs", "tokenToPublicationMs", "publicationToPlayableMs"):
        assert tracker.current[key] is None


def test_publication_observation_does_not_expand_existing_marker_capacity():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.consume([event(i + 1, "first_foreground_token", turnId=f"prior_{i}",
                           turnGeneration=1) for i in range(128)])
    tracker.consume([event(129, "assistant_text_generated", turnId="foreign", turnGeneration=1)])
    assert len(tracker.markers) == 128
    assert tracker.current["firstPublicationMs"] is None
    with pytest.raises(ValueError, match="journey_marker_bound"):
        tracker.consume([event(130, "first_playable_audio", turnId="foreign", turnGeneration=1)])


def test_optional_publication_metadata_cannot_overflow_existing_fitting_receipt(
    tmp_path, monkeypatch, capsys,
):
    import scripts.native_typed_journey as client
    _, receipt, argv = os_error_options(tmp_path)
    baseline = {"version": 1, "outcome": "completed", "executed": True,
                "stepCount": 1, "turns": [{"outcome": "completed"}], "padding": ""}
    padding = "x" * (65536 - len(json.dumps(baseline, sort_keys=True)))
    async def execute(options, steps, result):
        result.update(baseline | {"padding": padding})
        result["turns"] = [{"outcome": "completed", "firstPublicationMs": 10,
                            "tokenToPublicationMs": 2, "publicationToPlayableMs": 3}]
    monkeypatch.setattr(client, "execute", execute)
    assert client.main([*argv, "--execute"]) == 0
    saved = json.loads(receipt.read_text())
    assert saved == baseline | {"padding": padding}
    assert len(json.dumps(saved, sort_keys=True)) == 65536
    capsys.readouterr()


@pytest.mark.parametrize("publication,playable,expected", [
    (999, 1120, (None, None, None)),
    (1070, 1060, (70, 60, None)),
])
def test_publication_intervals_keep_unsupported_timestamps_unknown(
    publication, playable, expected,
):
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.started = 1000.0
    batch = [
        event(1, "transcript_final", role="user", text="Synthetic input"),
        event(2, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
        event(3, "first_foreground_token", turnId="current", turnGeneration=1),
        event(4, "assistant_text_generated", turnId="current", turnGeneration=1),
        event(5, "first_playable_audio", turnId="current", turnGeneration=1),
        event(6, "assistant_turn_completed", turnId="current", turnGeneration=1),
    ]
    for row, stamp in zip(batch, [1001, 1002, 1010, publication, playable, 1130], strict=True):
        row["monotonicMs"] = stamp
    tracker.consume(batch)
    assert tuple(tracker.current[key] for key in (
        "firstPublicationMs", "tokenToPublicationMs", "publicationToPlayableMs",
    )) == expected
    assert tracker.current["outcome"] == "completed"


def test_optional_publication_fields_do_not_displace_existing_failure_evidence(
    tmp_path, monkeypatch, capsys,
):
    import scripts.native_typed_journey as client
    _, receipt, argv = os_error_options(tmp_path)
    metadata = {"sourceFunction": "other", "sourceLine": None, "pathRole": "other",
                "exceptionKind": "PermissionError", "errno": 13, "winerror": None}
    baseline = {"version": 1, "outcome": "refused", "executed": True, "stepCount": 1,
                "failure": "PermissionError", "osError": metadata,
                "turns": [{"outcome": "pending"}], "padding": ""}
    padding = "x" * (65536 - len(json.dumps(baseline, sort_keys=True)))
    async def execute(options, steps, result):
        result.update(baseline | {"padding": padding})
        result["turns"] = [{"outcome": "pending", "firstPublicationMs": 10,
                            "tokenToPublicationMs": 2, "publicationToPlayableMs": 3}]
        raise PermissionError(13, "SYNTHETIC_PRIVATE_MESSAGE")
    monkeypatch.setattr(client, "execute", execute)
    monkeypatch.setattr(client, "_os_error_evidence", lambda error: metadata)
    assert client.main([*argv, "--execute"]) == 1
    saved = json.loads(receipt.read_text())
    assert saved == baseline | {"padding": padding}
    assert "SYNTHETIC" not in capsys.readouterr().out


def test_optional_publication_metadata_never_relaxes_original_receipt_bound(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client
    _, _, argv = os_error_options(tmp_path)
    async def execute(options, steps, result):
        result.update(outcome="completed", padding="x" * 65536, turns=[{
            "firstPublicationMs": 10, "tokenToPublicationMs": 2, "publicationToPlayableMs": 3,
        }])
    monkeypatch.setattr(client, "execute", execute)
    with pytest.raises(ValueError, match="journey_receipt_bound"):
        client.main([*argv, "--execute"])


# Natural cancellation and tokenless completion binding (#237).

CANCEL_TASK, CANCEL_RUN = "task_natural", "run_" + "c" * 32
CANCEL_TEXT = "Please stop that lookup."
CANCEL_STEP = {"label": "stop", "kind": "natural_cancel", "task": "work", "text": CANCEL_TEXT}


def numbered(rows, start):
    return [event(start + index, kind, **data) for index, (kind, data) in enumerate(rows)]


def admitted(sequence=1, binding=7):
    return ("typed_input_admitted", {"inputSequence": sequence, "bindingGeneration": binding})


def user_final(text=CANCEL_TEXT):
    return ("transcript_final", {"role": "user", "text": text})


def task_state(status, task=CANCEL_TASK):
    return ("task_state", {"taskId": task, "status": status})


def completed(turn="cancel_turn", generation=1, sequence=1, binding=7):
    data = {"turnId": turn, "turnGeneration": generation}
    if sequence is not None:
        data |= {"inputSequence": sequence, "bindingGeneration": binding}
    return ("assistant_turn_completed", data)


def tokenless_cancel_turn(task=CANCEL_TASK):
    return [
        admitted(), user_final(), task_state("cancelling", task),
        ("assistant_text_generated", {"turnId": "cancel_turn", "turnGeneration": 1}),
        completed(),
    ]


def settled(task=CANCEL_TASK):
    return [task_state("interrupted", task),
            ("task_result", {"taskId": task, "status": "interrupted", "text": "Stopped."})]


def natural_cancel_journey(tmp_path, monkeypatch, batches, *, prior=(), terminal="cancelled"):
    import scripts.native_typed_journey as client

    options = SimpleNamespace(
        preview_dir=tmp_path, native_profile=tmp_path, api_origin="http://127.0.0.1:1"
    )
    journey = Journey(options, {"expiresInSeconds": 3600}, "secret")
    monkeypatch.setattr(client, "read_journal", lambda _: ({CANCEL_RUN}, 0))
    monkeypatch.setattr(client, "NativeRunWitness", lambda *a, **kw: SimpleNamespace(
        read=lambda: {"terminal": terminal, "toolCalls": []}))
    history = [task_state("active"), *prior]
    journey.tracker.consume(numbered(history, 1))
    journey.bindings["work"] = (CANCEL_TASK, CANCEL_RUN)
    journey.admissions.add(CANCEL_RUN)
    cursor = len(history) + 1
    pending = []
    for rows in batches:
        pending.append(numbered(rows, cursor))
        cursor += len(rows)
    submitted = []

    async def request(route, payload=None):
        if route == "input":
            submitted.append(payload)
            return {}
        assert route == "events"
        return {"version": 2, "events": pending.pop(0) if pending else []}

    journey.request = request
    return journey, submitted


def run_step(journey, step=CANCEL_STEP, seconds=2):
    asyncio.run(asyncio.wait_for(journey.step(step), seconds))


def test_tokenless_natural_cancel_turn_completes_and_proves_cancellation(tmp_path, monkeypatch):
    journey, submitted = natural_cancel_journey(
        tmp_path, monkeypatch, [tokenless_cancel_turn() + settled()]
    )
    run_step(journey)
    assert submitted == [{"sequence": 1, "text": CANCEL_TEXT}]
    assert journey.tracker.current["outcome"] == "completed"
    assert journey.tracker.current["terminalBinding"] == "admittedInput"
    assert journey.tracker.current["firstTokenMs"] is None
    assert journey.results["work"]["terminal"] == "cancelled"
    assert "cancel_turn" not in json.dumps(journey.tracker.receipt())


def test_token_bound_natural_cancel_turn_is_also_trusted_lineage(tmp_path, monkeypatch):
    rows = [admitted(), user_final(),
            ("first_foreground_token", {"turnId": "cancel_turn", "turnGeneration": 1}),
            task_state("cancelling"), completed(sequence=None), *settled()]
    journey, _ = natural_cancel_journey(tmp_path, monkeypatch, [rows])
    run_step(journey)
    assert journey.tracker.current["terminalBinding"] == "token"


def test_cleanup_only_cancel_cannot_prove_a_natural_cancel(tmp_path, monkeypatch):
    rows = tokenless_cancel_turn()
    del rows[2]  # The turn completes, but nothing cancelled during this step.
    journey, submitted = natural_cancel_journey(tmp_path, monkeypatch, [rows, settled()])
    with pytest.raises(ValueError, match="journey_cancel_unobserved"):
        run_step(journey)
    asyncio.run(asyncio.wait_for(journey.settle_owned(), 2))
    # Cleanup cancelled the task with the typed grammar; that is not step evidence.
    assert submitted[-1] == {"sequence": 2, "text": "cancel task: " + CANCEL_TASK}
    assert journey.tracker.tasks[CANCEL_TASK] == "interrupted"
    assert journey.tracker.step_cancels == {}


def test_cancelling_a_different_task_cannot_prove_a_natural_cancel(tmp_path, monkeypatch):
    journey, _ = natural_cancel_journey(
        tmp_path, monkeypatch,
        [tokenless_cancel_turn(task="task_other") + settled(task="task_other")],
    )
    with pytest.raises(ValueError, match="journey_cancel_unobserved"):
        run_step(journey)


def test_an_additional_foreign_cancel_fails_the_natural_cancel(tmp_path, monkeypatch):
    rows = tokenless_cancel_turn()
    rows.insert(3, task_state("cancelling", "task_other"))
    journey, _ = natural_cancel_journey(tmp_path, monkeypatch, [rows + settled()])
    with pytest.raises(ValueError, match="journey_cancel_foreign"):
        run_step(journey)


@pytest.mark.parametrize("status", ["cancelling", "interrupted"])
def test_foreign_cancel_during_settlement_fails_the_natural_cancel(tmp_path, monkeypatch, status):
    # A known foreign task cancels in the batch collect() consumes, after the
    # pre-collect scope check passed; the identity-set check cannot see it.
    rows = tokenless_cancel_turn()
    journey, _ = natural_cancel_journey(
        tmp_path, monkeypatch, [rows, [task_state(status, "task_other"), *settled()]],
        prior=[task_state("active", "task_other")],
    )
    with pytest.raises(ValueError, match="journey_cancel_foreign"):
        run_step(journey)


@pytest.mark.parametrize("prior", ["cancelling", "interrupted", "completed"])
def test_task_not_active_at_step_start_cannot_be_naturally_cancelled(
    tmp_path, monkeypatch, prior,
):
    journey, submitted = natural_cancel_journey(
        tmp_path, monkeypatch, [tokenless_cancel_turn() + settled()],
        prior=[task_state(prior)],
    )
    with pytest.raises(ValueError, match="journey_cancel_not_active"):
        run_step(journey)
    assert submitted == []


@pytest.mark.parametrize("status", ["cancelling", "interrupted"])
@pytest.mark.parametrize("later", [False, True])
def test_task_state_before_this_steps_admission_never_counts(
    tmp_path, monkeypatch, status, later,
):
    rows = tokenless_cancel_turn()
    rows.remove(task_state("cancelling"))
    rows.insert(0, task_state(status))
    # Even a later post-admission terminal cannot launder a pre-admission cancel.
    journey, _ = natural_cancel_journey(
        tmp_path, monkeypatch, [rows + (settled() if later else [])]
    )
    with pytest.raises(ValueError, match="journey_cancel_stale"):
        run_step(journey)


def test_natural_cancel_requires_the_native_witness_to_end_cancelled(tmp_path, monkeypatch):
    rows = tokenless_cancel_turn() + [
        task_state("completed"),
        ("task_result", {"taskId": CANCEL_TASK, "status": "completed", "text": "Done."}),
    ]
    journey, _ = natural_cancel_journey(tmp_path, monkeypatch, [rows], terminal="completed")
    with pytest.raises(ValueError, match="journey_cancel_unproven"):
        run_step(journey)


def test_natural_cancel_that_dispatches_new_work_fails(tmp_path, monkeypatch):
    rows = tokenless_cancel_turn() + settled() + [task_state("active", "task_new")]
    journey, _ = natural_cancel_journey(tmp_path, monkeypatch, [rows])
    with pytest.raises(ValueError, match="journey_followup_dispatched"):
        run_step(journey)


def test_interrupted_natural_cancel_turn_fails(tmp_path, monkeypatch):
    rows = tokenless_cancel_turn()
    rows[-1] = ("assistant_turn_interrupted", rows[-1][1])
    journey, _ = natural_cancel_journey(tmp_path, monkeypatch, [rows + settled()])
    with pytest.raises(ValueError, match="journey_turn_interrupted"):
        run_step(journey)


@pytest.mark.parametrize("stale", ["prior_pair", "prior_input", "wrong_binding", "wrong_sequence",
                                   "uncorrelated"])
def test_natural_cancel_never_accepts_an_unbound_completion(tmp_path, monkeypatch, stale):
    prior = [("assistant_turn_completed", {"turnId": "old_turn", "turnGeneration": 1})]
    turn = {
        "prior_pair": completed(turn="old_turn"),
        "prior_input": completed(sequence=9),
        "wrong_binding": completed(binding=8),
        "wrong_sequence": completed(sequence=2),
        "uncorrelated": completed(sequence=None),
    }[stale]
    rows = tokenless_cancel_turn()
    rows[-1] = turn
    journey, _ = natural_cancel_journey(tmp_path, monkeypatch, [rows + settled()], prior=prior)
    with pytest.raises(TimeoutError):
        run_step(journey, seconds=0.5)
    assert journey.tracker.current["outcome"] == "pending"
    assert journey.tracker.pair is None
    assert "work" not in journey.results


@pytest.mark.parametrize("correlation", [
    {}, {"inputSequence": 2, "bindingGeneration": 7},
    {"inputSequence": 1, "bindingGeneration": 8},
    {"inputSequence": True, "bindingGeneration": 7},
    {"inputSequence": 1, "bindingGeneration": True},
])
def test_unbound_tokenless_completion_cannot_bind_current_input(correlation):
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic text")
    tracker.consume([
        event(1, "typed_input_admitted", inputSequence=1,
              bindingGeneration=1 if type(correlation.get("bindingGeneration")) is bool else 7),
        event(2, "assistant_turn_completed", turnId="foreign_turn", turnGeneration=1,
              **correlation),
    ])
    assert tracker.current["outcome"] == "pending"
    assert tracker.pair is None
    assert tracker.current["unboundTerminalEvents"] == 1


@pytest.mark.parametrize("terminal_first", [False, True])
def test_tokenless_completion_binds_only_its_own_admitted_input(terminal_first):
    tracker = EventTracker()
    tracker.begin("first", 1, "First input")
    tracker.consume([event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
                     event(2, "assistant_turn_completed", turnId="first_turn", turnGeneration=1,
                           inputSequence=1, bindingGeneration=7)])
    assert tracker.current["outcome"] == "completed"
    assert tracker.current["terminalBinding"] == "admittedInput"
    assert tracker.current["unboundTerminalEvents"] == 0
    tracker.begin("second", 2, "Second input")
    own = event(0, "assistant_turn_completed", turnId="second_turn", turnGeneration=1,
                inputSequence=2, bindingGeneration=7)
    rows = [
        event(0, "assistant_turn_completed", turnId="first_turn", turnGeneration=1,
              inputSequence=2, bindingGeneration=7),
        event(0, "assistant_turn_completed", turnId="other_turn", turnGeneration=1,
              inputSequence=1, bindingGeneration=7),
        event(0, "typed_input_admitted", inputSequence=2, bindingGeneration=7),
    ]
    rows.insert(0 if terminal_first else 3, own)
    for index, row in enumerate(rows, 3):
        row["sequence"] = index
    tracker.consume(rows)
    assert tracker.current["outcome"] == "completed"
    assert tracker.pair == ("second_turn", 1)
    assert tracker.current["terminalBinding"] == "admittedInput"
    assert tracker.current["unboundTerminalEvents"] == 2


@pytest.mark.parametrize("kind", ["conversation", "status"])
def test_tokenless_completion_still_cannot_certify_a_spoken_turn(tmp_path, monkeypatch, kind):
    step = {"label": "speech", "kind": kind, "text": CANCEL_TEXT}
    if kind == "status":
        step["task"] = "work"
    rows = [admitted(), user_final(), completed()]
    journey, _ = natural_cancel_journey(tmp_path, monkeypatch, [rows])
    with pytest.raises(ValueError, match="journey_turn_tokenless"):
        run_step(journey, step)


def natural_scenario(tmp_path, step):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({"version": 1, "steps": [
        {"label": "work", "kind": "dispatch", "text": "Synthetic lookup", "wait": "accepted"},
        step,
    ]}), encoding="utf-8")
    return load_scenario(path)


def test_scenario_accepts_natural_cancel_and_keeps_typed_cancel(tmp_path):
    assert natural_scenario(tmp_path, CANCEL_STEP)[1] == CANCEL_STEP
    typed = {"label": "stop", "kind": "cancel", "task": "work"}
    assert natural_scenario(tmp_path, typed)[1] == typed


@pytest.mark.parametrize("change", [
    {"text": None}, {"task": None}, {"task": "missing"}, {"wait": "completed"},
    {"tool": "web_search"},
])
def test_scenario_rejects_malformed_natural_cancel(tmp_path, change):
    step = dict(CANCEL_STEP)
    for key, value in change.items():
        if value is None:
            del step[key]
        else:
            step[key] = value
    expected = "journey_text_bound" if "text" not in step else "journey_natural_cancel_shape"
    with pytest.raises(ValueError, match=expected):
        natural_scenario(tmp_path, step)


@pytest.mark.parametrize("text", [
    "cancel task: task_natural", "  CANCEL TASK: task_natural", "Task: start something",
])
def test_natural_cancel_scenario_refuses_the_typed_command_grammar(tmp_path, text):
    with pytest.raises(ValueError, match="journey_natural_cancel_grammar"):
        natural_scenario(tmp_path, CANCEL_STEP | {"text": text})


# Host-refused web runs (#237): the native run completed, the tracker failed.

REFUSED_TASK, REFUSED_RUN = "task_refused", "run_" + "d" * 32


def refused_journey(tmp_path, monkeypatch, rows, *, terminal="completed"):
    import scripts.native_typed_journey as client

    options = SimpleNamespace(
        preview_dir=tmp_path, native_profile=tmp_path, api_origin="http://127.0.0.1:1"
    )
    journey = Journey(options, {"expiresInSeconds": 3600}, "secret")
    monkeypatch.setattr(client, "NativeRunWitness", lambda *a, **kw: SimpleNamespace(
        read=lambda: {"terminal": terminal, "toolCalls": []}))
    journey.tracker.consume(numbered(rows, 1))
    journey.bindings["web"] = (REFUSED_TASK, REFUSED_RUN)
    return journey


def failed_result(text, task=REFUSED_TASK, status="failed"):
    return [("task_state", {"taskId": task, "status": status}),
            ("task_result", {"taskId": task, "status": status, "text": text})]


@pytest.mark.parametrize("reason", ["insufficient", "too_long", "unverified"])
def test_host_refused_web_run_is_classified_task_failure(tmp_path, monkeypatch, reason):
    from hermes_realtime.integration.api import _WEB_REFUSAL_NOTICES, _WEB_UNVERIFIED_NOTICE

    notice = (_WEB_UNVERIFIED_NOTICE if reason == "unverified"
              else _WEB_REFUSAL_NOTICES[reason])
    rows = [("task_state", {"taskId": REFUSED_TASK, "status": "active"}),
            *failed_result(notice)]
    journey = refused_journey(tmp_path, monkeypatch, rows)
    with pytest.raises(ValueError, match="journey_task_failed"):
        asyncio.run(journey.collect("web"))
    assert journey.failures == {"web": {"classification": "host_refused", "refusal": reason}}
    assert "web" not in journey.results


@pytest.mark.parametrize("variant", [
    "untrusted_text", "other_task", "interrupted_status", "later_interrupted", "state_only",
    "superseded", "witness_cancelled",
])
def test_host_refused_classification_requires_trusted_exact_evidence(
    tmp_path, monkeypatch, variant,
):
    from hermes_realtime.integration.api import _WEB_REFUSAL_NOTICES

    notice = _WEB_REFUSAL_NOTICES["insufficient"]
    terminal = "completed"
    rows = [("task_state", {"taskId": REFUSED_TASK, "status": "active"})]
    if variant == "untrusted_text":
        rows += failed_result("Synthetic " + notice)
    elif variant == "other_task":
        rows += [("task_state", {"taskId": "task_other", "status": "active"}),
                 *failed_result(notice, task="task_other"),
                 ("task_state", {"taskId": REFUSED_TASK, "status": "failed"})]
    elif variant == "interrupted_status":
        rows += failed_result(notice, status="interrupted")
    elif variant == "later_interrupted":
        rows += [*failed_result(notice), ("task_state", {"taskId": REFUSED_TASK,
                                                          "status": "interrupted"})]
    elif variant == "state_only":
        rows += [("task_state", {"taskId": REFUSED_TASK, "status": "failed"})]
    elif variant == "superseded":
        rows += [*failed_result(notice), ("task_result", {
            "taskId": REFUSED_TASK, "status": "failed", "text": "Synthetic failure."})]
    else:
        rows += failed_result(notice)
        terminal = "cancelled"
    journey = refused_journey(tmp_path, monkeypatch, rows, terminal=terminal)
    if variant in {"interrupted_status", "superseded"}:
        assert journey.tracker.host_refusals == {}
    with pytest.raises(ValueError, match="journey_terminal_binding"):
        asyncio.run(journey.collect("web"))
    assert journey.failures == {}


def test_matching_failed_witness_keeps_unclassified_task_failure(tmp_path, monkeypatch):
    from hermes_realtime.integration.api import _WEB_REFUSAL_NOTICES

    rows = [("task_state", {"taskId": REFUSED_TASK, "status": "active"}),
            *failed_result(_WEB_REFUSAL_NOTICES["insufficient"])]
    journey = refused_journey(tmp_path, monkeypatch, rows, terminal="failed")
    asyncio.run(journey.collect("web"))
    assert journey.results["web"]["terminal"] == "failed"
    assert journey.failures == {}


def test_receipt_reports_host_refused_classification(tmp_path, monkeypatch):
    import hashlib
    import sys

    import scripts.native_typed_journey as client
    from hermes_realtime import host_launcher

    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              browser_origin="http://127.0.0.1:1",
                              livekit_url="ws://127.0.0.1:2")
    suffix = hashlib.sha256(str(tmp_path / "active.private.json").encode()).hexdigest()[:16]
    credential = dict(
        workerIdentity="worker_preview_" + suffix, roomName="hermes-preview-" + suffix,
        url=options.livekit_url, token="synthetic-lease", participantIdentity="synthetic",
        expiresInSeconds=3600,
    )

    class Room:
        async def connect(self, *args, **kwargs):
            pass

        async def disconnect(self):
            pass

    monkeypatch.setitem(sys.modules, "livekit", SimpleNamespace(rtc=SimpleNamespace(
        Room=Room, RoomOptions=lambda **kwargs: kwargs)))
    monkeypatch.setattr(host_launcher, "_load_api_bearer", lambda _: "synthetic")
    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))

    def request(origin, route, **kwargs):
        if route == "events":
            return 200, {"version": 2, "events": []}
        return (200, credential) if route == "stable-bootstrap" else (200, {})

    monkeypatch.setattr(client, "request_json", request)

    async def step(self, value):
        self.failures["web"] = {"classification": "host_refused", "refusal": "insufficient"}
        raise ValueError("journey_task_failed")

    monkeypatch.setattr(client.Journey, "step", step)
    receipt = {"outcome": "refused"}
    with pytest.raises(ValueError, match="journey_task_failed"):
        asyncio.run(client.execute(options, [{"label": "web"}], receipt))
    assert receipt["taskFailures"] == {
        "web": {"classification": "host_refused", "refusal": "insufficient"}
    }
    assert receipt["outcome"] == "refused"


# Content-free per-step diagnostics (#237). They explain a verdict; they never change it.

PRIVATE_MARKERS = ("cancel_turn", "old_turn", CANCEL_TASK, CANCEL_RUN, CANCEL_TEXT)


def diagnose_execute(tmp_path, monkeypatch, step_rows, *, witness="cancelled",
                     cleanup_rows=(), fail="journey_turn_incomplete", prior_rows=(),
                     collected=None, cleanup=None):
    """Run execute() with one synthetic step that consumes ``step_rows`` then fails.

    ``collected`` retains a collect() result for the step's task before it fails.
    ``cleanup`` collects mandatory cleanup calls in order, plus whether any other
    callback ran (a suspension) between the step's failure and settle_owned.
    Returns the receipt, cleanup inputs, and the native witnesses constructed.
    """
    import hashlib
    import sys

    import scripts.native_typed_journey as client
    from hermes_realtime import host_launcher

    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              browser_origin="http://127.0.0.1:1", api_origin="http://127.0.0.1:3",
                              livekit_url="ws://127.0.0.1:2")
    suffix = hashlib.sha256(str(tmp_path / "active.private.json").encode()).hexdigest()[:16]
    credential = dict(
        workerIdentity="worker_preview_" + suffix, roomName="hermes-preview-" + suffix,
        url=options.livekit_url, token="synthetic-lease", participantIdentity="synthetic",
        expiresInSeconds=3600,
    )

    class Room:
        async def connect(self, *args, **kwargs):
            pass

        async def disconnect(self):
            if cleanup is not None:
                cleanup.append("disconnect")

    monkeypatch.setitem(sys.modules, "livekit", SimpleNamespace(rtc=SimpleNamespace(
        Room=Room, RoomOptions=lambda **kwargs: kwargs)))
    monkeypatch.setattr(host_launcher, "_load_api_bearer", lambda _: "synthetic")
    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))
    constructed = []

    def witness_factory(*args, **kwargs):
        constructed.append(True)
        return SimpleNamespace(read=lambda: {"terminal": witness, "toolCalls": []})

    monkeypatch.setattr(client, "NativeRunWitness", witness_factory)
    suspended = []
    if cleanup is not None:
        real_settle = client.Journey.settle_owned

        async def settle_owned(self):
            cleanup.append("settle_owned_after_suspension" if suspended else "settle_owned")
            return await real_settle(self)

        monkeypatch.setattr(client.Journey, "settle_owned", settle_owned)
    inputs, pending = [], []

    def request(origin, route, **kwargs):
        if route == "stop" and cleanup is not None:
            cleanup.append("stop")
        if route == "input":
            inputs.append(kwargs["payload"])
            start = len(history) + 1
            pending.append(numbered(list(cleanup_rows), start))
            return 202, {}
        if route == "events":
            return 200, {"version": 2, "events": pending.pop(0) if pending else []}
        return (200, credential) if route == "stable-bootstrap" else (200, {})

    monkeypatch.setattr(client, "request_json", request)
    before = [task_state("active"), *prior_rows]
    history = [*before, *step_rows]

    async def step(self, value):
        self.tracker.consume(numbered(before, 1))
        self.bindings["work"] = (CANCEL_TASK, CANCEL_RUN)
        self.sequence += 1
        self.tracker.begin(value["label"], self.sequence, CANCEL_TEXT)
        self.tracker.records[-1]["task"] = "work"
        self.tracker.consume(numbered(step_rows, len(before) + 1))
        if collected is not None:
            self.results["work"] = {"terminal": collected, "toolCalls": []}
        # Runs only if execute() suspends before mandatory cleanup begins.
        asyncio.get_running_loop().call_soon(suspended.append, True)
        raise ValueError(fail)

    monkeypatch.setattr(client.Journey, "step", step)
    receipt = {"outcome": "refused"}
    with pytest.raises(ValueError, match=fail):
        asyncio.run(client.execute(options, [{"label": "stop"}], receipt))
    return receipt, inputs, constructed


def test_mandatory_cleanup_follows_the_failure_with_no_diagnostic_suspension(
    tmp_path, monkeypatch,
):
    cleanup = []
    rows = tokenless_cancel_turn() + settled()
    receipt, inputs, constructed = diagnose_execute(tmp_path, monkeypatch, rows, cleanup=cleanup)
    # No await precedes settle_owned, so nothing (not even a cancellation) can run between
    # the failure and mandatory cleanup; all of it completes and the failure is kept.
    assert cleanup == ["settle_owned", "stop", "disconnect"]
    assert receipt["cleanup"] == {
        "ownedTasksSettled": True, "sessionStopped": True, "roomDisconnected": True,
    }
    assert receipt["outcome"] == "refused" and inputs == [] and constructed == []
    assert receipt["diagnostics"]["preCleanupTasks"] == {
        "work": {"trackerStatus": "interrupted", "nativeTerminal": None,
                 "hostRefusalObserved": None},
    }


def test_diagnostics_never_construct_a_native_witness(tmp_path, monkeypatch):
    # Already terminal, never collected: the one case a diagnostic read could tempt.
    rows = tokenless_cancel_turn() + settled()
    receipt, inputs, constructed = diagnose_execute(tmp_path, monkeypatch, rows)
    assert constructed == [] and inputs == []
    assert receipt["diagnostics"]["steps"][0]["nativeTerminal"] is None


def test_diagnostics_explain_an_uncorrelated_cancel_turn_without_changing_the_verdict(
    tmp_path, monkeypatch,
):
    rows = tokenless_cancel_turn()
    rows[-1] = completed(sequence=None)  # The no-capture host omits the typed lineage.
    rows += [("assistant_turn_completed", {"turnId": "old_turn", "turnGeneration": 2}),
             *settled()]
    receipt, inputs, constructed = diagnose_execute(tmp_path, monkeypatch, rows)
    assert receipt["outcome"] == "refused" and receipt["turns"][0]["outcome"] == "pending"
    assert receipt["cleanup"] == {
        "ownedTasksSettled": True, "sessionStopped": True, "roomDisconnected": True,
    }
    assert inputs == [] and constructed == []  # Already terminal: no cleanup cancel.
    assert receipt["diagnostics"] == {
        "version": 1,
        "steps": [{
            "label": "stop",
            "admissionObserved": True,
            "admittedInputValid": True,
            "terminalKinds": {"completed": 2, "interrupted": 0},
            "terminals": [
                {"kind": "completed", "turnIdPresent": True, "inputSequencePresent": False,
                 "bindingGenerationPresent": False, "inputMatchesAdmitted": False,
                 "priorPair": False, "bound": False},
                {"kind": "completed", "turnIdPresent": True, "inputSequencePresent": False,
                 "bindingGenerationPresent": False, "inputMatchesAdmitted": False,
                 "priorPair": False, "bound": False},
            ],
            "terminalsOmitted": 0,
            "targetCancelObserved": True,
            "targetCancelAfterAdmission": True,
            "nativeTerminal": None,  # Not read: collect() did not run for this task.
        }],
        "preCleanupTasks": {"work": {"trackerStatus": "interrupted", "nativeTerminal": None,
                                     "hostRefusalObserved": None}},
    }
    assert not any(marker in json.dumps(receipt) for marker in PRIVATE_MARKERS)


def test_diagnostics_report_a_bound_correlated_terminal_and_prior_pairs(tmp_path, monkeypatch):
    rows = [admitted(), user_final(),
            ("assistant_turn_interrupted", {"turnId": "old_turn", "turnGeneration": 1,
                                            "inputSequence": 1, "bindingGeneration": 8}),
            completed(), *settled()]
    receipt, _, _ = diagnose_execute(tmp_path, monkeypatch, rows)
    diagnosis = receipt["diagnostics"]["steps"][0]
    assert diagnosis["terminalKinds"] == {"completed": 1, "interrupted": 1}
    assert [(row["kind"], row["inputMatchesAdmitted"], row["bound"])
            for row in diagnosis["terminals"]] == [
        ("interrupted", False, False), ("completed", True, True),
    ]
    assert receipt["turns"][0]["terminalBinding"] == "admittedInput"


def test_diagnostics_flag_a_terminal_whose_pair_predates_the_step(tmp_path, monkeypatch):
    prior = [("assistant_turn_completed", {"turnId": "old_turn", "turnGeneration": 1})]
    rows = [admitted(), user_final(), completed(turn="old_turn"), *settled()]
    receipt, _, _ = diagnose_execute(tmp_path, monkeypatch, rows, prior_rows=prior)
    terminal = receipt["diagnostics"]["steps"][0]["terminals"][0]
    assert (terminal["inputMatchesAdmitted"], terminal["priorPair"], terminal["bound"]) == (
        True, True, False,
    )


def test_diagnostics_count_only_the_steps_own_target_cancellation(tmp_path, monkeypatch):
    rows = [admitted(), user_final(), task_state("active", "task_other"),
            task_state("cancelling", "task_other"), completed()]
    receipt, _, _ = diagnose_execute(tmp_path, monkeypatch, rows, cleanup_rows=settled())
    diagnosis = receipt["diagnostics"]["steps"][0]
    assert (diagnosis["targetCancelObserved"], diagnosis["targetCancelAfterAdmission"]) == (
        False, False,
    )
    assert "task_other" not in json.dumps(receipt)


def test_diagnostics_without_admission_leave_input_equality_unknown(tmp_path, monkeypatch):
    rows = [user_final(), task_state("cancelling"), completed()]
    receipt, _, _ = diagnose_execute(
        tmp_path, monkeypatch, rows, cleanup_rows=settled(),
    )
    diagnosis = receipt["diagnostics"]["steps"][0]
    assert (diagnosis["admissionObserved"], diagnosis["admittedInputValid"]) == (False, False)
    assert diagnosis["terminals"][0]["inputMatchesAdmitted"] is None
    assert diagnosis["targetCancelObserved"] is True
    assert diagnosis["targetCancelAfterAdmission"] is False


def test_pre_cleanup_snapshot_survives_cleanup(tmp_path, monkeypatch):
    rows = [admitted(), user_final(), completed()]
    receipt, inputs, constructed = diagnose_execute(
        tmp_path, monkeypatch, rows, cleanup_rows=settled(),
    )
    # Cleanup still cancels the active task and collects it exactly as before.
    assert inputs == [{"sequence": 2, "text": "cancel task: " + CANCEL_TASK}]
    assert constructed == [True]  # cleanup's own collect(), never diagnostics.
    assert receipt["tasks"]["work"]["terminal"] == "cancelled"
    # The snapshot taken before cleanup is unchanged by what cleanup then did.
    diagnostics = receipt["diagnostics"]
    assert diagnostics["preCleanupTasks"] == {
        "work": {"trackerStatus": "active", "nativeTerminal": None,
                 "hostRefusalObserved": None},
    }
    assert diagnostics["steps"][0]["targetCancelObserved"] is False
    assert diagnostics["steps"][0]["nativeTerminal"] is None


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled", "surprising"])
def test_native_terminal_is_the_retained_collect_category(tmp_path, monkeypatch, terminal):
    rows = tokenless_cancel_turn() + settled()
    receipt, _, constructed = diagnose_execute(tmp_path, monkeypatch, rows, collected=terminal)
    expected = terminal if terminal != "surprising" else "other"
    assert constructed == []
    assert receipt["diagnostics"]["preCleanupTasks"]["work"]["nativeTerminal"] == expected
    assert receipt["diagnostics"]["steps"][0]["nativeTerminal"] == expected


def test_diagnostic_failure_never_changes_verdict_or_cleanup(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr(client, "_journey_diagnostics", explode)
    rows = tokenless_cancel_turn() + settled()
    receipt, _, _ = diagnose_execute(tmp_path, monkeypatch, rows)
    assert "diagnostics" not in receipt
    assert receipt["cleanup"] == {
        "ownedTasksSettled": True, "sessionStopped": True, "roomDisconnected": True,
    }


def test_diagnostics_never_overflow_an_otherwise_fitting_receipt(tmp_path, monkeypatch, capsys):
    import scripts.native_typed_journey as client

    _, receipt, argv = os_error_options(tmp_path)
    baseline = {"version": 1, "outcome": "refused", "executed": True, "stepCount": 1,
                "failure": "journey_turn_incomplete", "padding": ""}
    padding = "x" * (65536 - len(json.dumps(baseline, sort_keys=True)))

    async def execute(options, steps, result):
        result.update(baseline | {"padding": padding})
        result["diagnostics"] = {"version": 1, "steps": [{"label": "stop"}]}
        raise ValueError("journey_turn_incomplete")

    monkeypatch.setattr(client, "execute", execute)
    assert client.main([*argv, "--execute"]) == 1
    saved = json.loads(receipt.read_text())
    assert saved == baseline | {"padding": padding}
    capsys.readouterr()


def test_terminal_diagnostics_are_bounded_per_step():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.consume([event(index, "assistant_turn_completed", turnId=f"foreign_{index}",
                           turnGeneration=1) for index in range(1, 12)])
    record = tracker.records[-1]
    assert len(record["terminals"]) == 8 and record["omitted"] == 3
    assert record["kinds"] == {"completed": 11, "interrupted": 0}


# Synthetic microphone `speak` step (#237). A mechanism proof of the controlled synthetic
# path: it bypasses the physical microphone and AEC.

SPEAK_TEXT = "Hello, can you hear me?"
SPEAK_STEP = {"label": "greet", "kind": "speak"}
SPEAK_WORKER = "worker_preview_speak"


class FakeAudioSource:
    def __init__(self, rate, channels, queue_size_ms=1000):
        self.rate, self.channels, self.queue_size_ms = rate, channels, queue_size_ms
        self.frames, self.played_out, self.closed = [], False, False

    async def aclose(self):
        self.closed = True

    async def wait_for_playout(self):
        self.played_out = True

    async def capture_frame(self, frame):
        self.frames.append(frame)
        await asyncio.sleep(0)


def fake_frame(level, samples=480):
    import array

    pcm = array.array("h", [level, -level] * (samples // 2))
    return SimpleNamespace(
        data=pcm, sample_rate=48_000, num_channels=1, samples_per_channel=samples
    )


class FakeAudioStream:
    tracks: list = []

    def __init__(self, track, **kwargs):
        self.track = track
        FakeAudioStream.tracks.append(track)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        for _ in range(3):
            yield SimpleNamespace(frame=fake_frame(8000))

    async def aclose(self):
        return None


def fake_rtc():
    FakeAudioStream.tracks = []
    return SimpleNamespace(
        AudioSource=FakeAudioSource,
        AudioStream=FakeAudioStream,
        AudioFrame=lambda **kwargs: SimpleNamespace(**kwargs),
        LocalAudioTrack=SimpleNamespace(
            create_audio_track=lambda name, source: SimpleNamespace(name=name, source=source)
        ),
        TrackPublishOptions=lambda **kwargs: kwargs,
        TrackSource=SimpleNamespace(SOURCE_MICROPHONE="microphone"),
        TrackKind=SimpleNamespace(KIND_AUDIO="audio"),
    )


class FakeRoom:
    def __init__(self):
        self.published, self.handlers, self.remote_participants = [], {}, {}
        self.local_participant = SimpleNamespace(publish_track=self._publish)

    async def _publish(self, track, options):
        self.published.append((track, options))

    def on(self, event, callback=None):
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event, *args):
        for callback in self.handlers.get(event, []):
            callback(*args)


SPEAK_PAIR = {"turnId": "speak_turn", "turnGeneration": 1}
READY = {"generation": 1, "mediaIncarnation": 1}


def speak_flow(text=SPEAK_TEXT, *, finals=1, pairs=(SPEAK_PAIR,), complete=True, vad=True):
    """Event batches for one spoken step; the last batch holds the terminal."""
    final = ("transcript_final", {"role": "user", "text": text})
    voice = [("voice_activity_started", {}), ("voice_activity_ended", {})] if vad else []
    # The empty first batch is the step's flush poll, before its trace starts.
    batches = [[], [("voice_input_ready", READY)], [*voice, *[final] * finals]]
    for pair in pairs:
        batches.append([("first_foreground_token", pair), ("assistant_text_generated", pair)])
    tail = [("assistant_turn_completed", pair) for pair in pairs] if complete else []
    batches.append(tail)
    return batches


def speak_journey(tmp_path, monkeypatch, batches, *, audio_at=None, before=(), settle=None,
                  ready_needs_audio=False):
    """A Journey with a fake mic transport. ``audio_at[n]`` frames of worker audio arrive
    just before the n-th events response (1-based)."""
    import scripts.native_typed_journey as client

    for name, value in {
        "_SPEAK_FRAME_SECONDS": 0.0,
        "_SPEAK_DRAIN_SECONDS": 0.0,
        "_SPEAK_TURN_TIMEOUT": 0.5,
        "_SPEAK_SETTLE_TIMEOUT": 0.2,
        "_SPEAK_READY_TIMEOUT": 0.3,
        "_SPEAK_SETTLE_SECONDS": 0.0,
    }.items():
        monkeypatch.setattr(client, name, value)
    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              api_origin="http://127.0.0.1:1")
    journey = Journey(options, {"expiresInSeconds": 3600}, "secret")
    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))
    rtc, room = fake_rtc(), FakeRoom()
    journey.voice = client.SyntheticVoice(rtc, room, SPEAK_WORKER)
    if before:
        journey.tracker.consume(numbered(list(before), 1))
    cursor = len(before) + 1
    pending = []
    for rows in batches:
        pending.append(numbered(rows, cursor))
        cursor += len(rows)
    journey.media, journey.inputs, calls = [], [], [0]
    if settle is not None:
        journey.voice.meter.observe(fake_frame(8000))

    async def request(route, payload=None):
        if route == "media":
            journey.media.append(payload)
            return {"mediaIncarnation": payload["mediaIncarnation"], "version": 1}
        if route == "input":
            journey.inputs.append(payload)
            return {}
        assert route == "events"
        if (ready_needs_audio and pending
                and any(row["kind"] == "voice_input_ready" for row in pending[0])
                and not any(track.source.frames for track, _ in room.published)):
            return {"version": 2, "events": []}  # the worker has received no PCM yet
        calls[0] += 1
        for _ in range((audio_at or {}).get(calls[0], 0)):
            journey.voice.meter.observe(fake_frame(8000))
        return {"version": 2, "events": pending.pop(0) if pending else []}

    journey.request = request
    journey.rtc, journey.room = rtc, room
    return journey


def speak(journey, seconds=5):
    asyncio.run(asyncio.wait_for(journey.step(SPEAK_STEP), seconds))


def passing_speak(tmp_path, monkeypatch, **overrides):
    kwargs = {"audio_at": {4: 6}} | overrides.pop("kwargs", {})
    journey = speak_journey(tmp_path, monkeypatch, speak_flow(**overrides), **kwargs)
    return journey


def test_speak_passes_with_one_final_one_completed_pair_and_worker_audio(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch)
    speak(journey)
    receipt = journey.speak_receipt
    assert receipt["transcriptMatches"] is True
    assert receipt["finalsCount"] == 1
    assert receipt["pairBound"] is True and receipt["pairCompleted"] is True
    assert receipt["voiceActivityOrdered"] is True
    assert receipt["agentAudioFrames"] == 6 and receipt["agentAudioMs"] == 60
    assert journey.inputs == []  # the microphone path never fakes a typed admission
    assert journey.tracker.turns == []


def test_speak_makes_exactly_one_media_request_and_publishes_one_microphone_track(
    tmp_path, monkeypatch
):
    journey = passing_speak(tmp_path, monkeypatch)
    speak(journey)
    assert journey.media == [{"mediaIncarnation": 1}]
    assert len(journey.room.published) == 1
    track, options = journey.room.published[0]
    assert track.name == "microphone-1" and options == {"source": "microphone"}
    assert track.source.rate == 48_000 and track.source.channels == 1
    frames = track.source.frames
    assert {frame.samples_per_channel for frame in frames} == {480}
    assert frames[0].data == bytes(960) and frames[-1].data == bytes(960)
    assert any(any(frame.data) for frame in frames)


def test_speak_receipt_is_metadata_only(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch)
    speak(journey)
    raw = json.dumps(journey.speak_receipt)
    for private in ("hello", "speak_turn", "can you hear"):
        assert private not in raw.lower()
    assert all(type(value) in {bool, int, str} for value in journey.speak_receipt.values())


@pytest.mark.parametrize("text", ["Hello, can you hear you?", "Hello can you hear me please",
                                  "", "hello"])
def test_speak_refuses_wrong_words(tmp_path, monkeypatch, text):
    journey = passing_speak(tmp_path, monkeypatch, text=text)
    with pytest.raises(ValueError, match="journey_speak_transcript"):
        speak(journey)
    assert journey.speak_receipt["transcriptMatches"] is False


def test_speak_normalizes_case_and_punctuation_only(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, text="hello -- CAN you hear, me!")
    speak(journey)
    assert journey.speak_receipt["transcriptMatches"] is True


def test_speak_refuses_two_finals_even_when_both_match(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, finals=2)
    with pytest.raises(ValueError, match="journey_speak_finals"):
        speak(journey)
    assert journey.speak_receipt["finalsCount"] == 2


def test_speak_refuses_zero_finals(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, finals=0)
    with pytest.raises(ValueError, match="journey_speak_finals"):
        speak(journey)


def test_speak_refuses_a_missing_pair(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, pairs=())
    with pytest.raises(ValueError, match="journey_speak_no_pair"):
        speak(journey)
    assert journey.speak_receipt["pairBound"] is False


def test_speak_refuses_a_concurrent_turn(tmp_path, monkeypatch):
    other = {"turnId": "other_turn", "turnGeneration": 1}
    journey = passing_speak(tmp_path, monkeypatch, pairs=(SPEAK_PAIR, other))
    with pytest.raises(ValueError, match="journey_speak_concurrent"):
        speak(journey)


def test_speak_refuses_an_unbound_terminal_beside_the_pair(tmp_path, monkeypatch):
    flow = speak_flow()
    flow[-1].append(("assistant_turn_completed", {}))  # a terminal with no turn identity
    journey = speak_journey(tmp_path, monkeypatch, flow, audio_at={4: 6})
    with pytest.raises(ValueError, match="journey_speak_pair_unbound"):
        speak(journey)


def test_speak_refuses_a_turn_that_does_not_complete(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, complete=False)
    with pytest.raises(ValueError, match="journey_speak_turn_incomplete"):
        speak(journey)
    assert journey.speak_receipt["pairCompleted"] is False


def test_speak_refuses_a_pair_whose_token_precedes_the_final(tmp_path, monkeypatch):
    flow = speak_flow()
    flow[2], flow[3] = flow[3], flow[2]  # the token arrives before the user final
    journey = speak_journey(tmp_path, monkeypatch, flow, audio_at={4: 6})
    with pytest.raises(ValueError, match="journey_speak_pair_unbound"):
        speak(journey)


def test_speak_refuses_a_pair_already_seen_before_the_step(tmp_path, monkeypatch):
    before = [("assistant_turn_completed", SPEAK_PAIR)]
    journey = passing_speak(tmp_path, monkeypatch, kwargs={"before": before, "audio_at": {4: 6}})
    with pytest.raises(ValueError, match="journey_speak_pair_unbound"):
        speak(journey)


def test_speak_refuses_missing_voice_activity(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, vad=False)
    with pytest.raises(ValueError, match="journey_speak_voice_activity"):
        speak(journey)
    assert journey.speak_receipt["voiceActivityOrdered"] is False


def test_speak_refuses_ended_before_started(tmp_path, monkeypatch):
    flow = speak_flow()
    flow[2][0], flow[2][1] = flow[2][1], flow[2][0]
    journey = speak_journey(tmp_path, monkeypatch, flow, audio_at={4: 6})
    with pytest.raises(ValueError, match="journey_speak_voice_activity"):
        speak(journey)


def test_speak_refuses_a_turn_with_no_agent_audio(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, kwargs={"audio_at": {}})
    with pytest.raises(ValueError, match="journey_speak_no_audio"):
        speak(journey)
    assert journey.speak_receipt["agentAudioFrames"] == 0


def test_speak_refuses_audio_that_is_below_the_threshold(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, kwargs={"audio_at": {4: 1}})
    with pytest.raises(ValueError, match="journey_speak_no_audio"):
        speak(journey)


def test_speak_audio_before_the_final_cannot_satisfy_the_response_window(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, kwargs={"audio_at": {3: 40}})
    with pytest.raises(ValueError, match="journey_speak_no_audio"):
        speak(journey)
    assert journey.speak_receipt["inputWindowAudioFrames"] == 40


def test_speak_splits_worker_audio_between_the_input_and_response_windows(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch, kwargs={"audio_at": {3: 40, 4: 6}})
    speak(journey)
    receipt = journey.speak_receipt
    assert (receipt["inputWindowAudioFrames"], receipt["agentAudioFrames"]) == (40, 6)


def test_speak_never_speaks_while_worker_audio_keeps_arriving_after_ready(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    cue = {call: 1 for call in range(3, 60)}  # a cue that never stops
    journey = passing_speak(tmp_path, monkeypatch, kwargs={"audio_at": cue})
    monkeypatch.setattr(client, "_SPEAK_SETTLE_SECONDS", 0.1)  # after the helper resets it
    with pytest.raises(ValueError, match="journey_speak_not_quiet"):
        speak(journey)
    assert not any(any(f.data) for f in journey.room.published[0][0].source.frames)  # no clip
    assert journey.speak_receipt["cueAudioObserved"] is True


def test_the_post_ready_gate_needs_a_whole_quiet_interval_even_with_no_audio_yet(
    tmp_path, monkeypatch
):
    # The host schedules the cue AFTER publishing readiness, so "no audio so far" is not quiet.
    import scripts.native_typed_journey as client

    journey = passing_speak(tmp_path, monkeypatch, kwargs={"audio_at": {}})
    monkeypatch.setattr(client, "_SPEAK_SETTLE_SECONDS", 0.3)
    monkeypatch.setattr(client, "_SPEAK_SETTLE_TIMEOUT", 3.0)
    with contextlib.suppress(ValueError):  # the gate's polls consume the scripted events
        speak(journey)
    assert journey.speak_receipt["quietGateMs"] >= 280
    assert journey.speak_receipt["cueAudioObserved"] is False


def test_a_cue_arriving_inside_the_interval_restarts_it(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    # Gate polls start at call 3 and are >= 50 ms apart, so call 5 is >= 100 ms in.
    journey = passing_speak(tmp_path, monkeypatch, kwargs={"audio_at": {5: 5, 9: 6}})
    monkeypatch.setattr(client, "_SPEAK_SETTLE_SECONDS", 0.3)
    monkeypatch.setattr(client, "_SPEAK_SETTLE_TIMEOUT", 3.0)
    with contextlib.suppress(ValueError):
        speak(journey)
    receipt = journey.speak_receipt
    assert receipt["cueAudioObserved"] is True
    assert receipt["quietGateMs"] >= 380  # >= 100 ms to the cue plus a whole 300 ms interval


def test_the_gate_interval_starts_at_the_later_of_entry_and_the_last_audio(tmp_path):
    import scripts.native_typed_journey as client

    now = [100.0]
    meter = client.SyntheticVoice(fake_rtc(), FakeRoom(), SPEAK_WORKER).meter
    meter._clock = lambda: now[0]
    assert meter.quiet_since(100.0, 1.0) is False  # nothing observed is not yet quiet
    now[0] = 100.9
    assert meter.quiet_since(100.0, 1.0) is False
    now[0] = 101.0
    assert meter.quiet_since(100.0, 1.0) is True
    meter.last_audio_at = 100.8  # audio before entry would not matter; after entry restarts
    assert meter.quiet_since(100.0, 1.0) is False
    now[0] = 101.8
    assert meter.quiet_since(100.0, 1.0) is True
    meter.last_audio_at = 50.0  # stale audio older than entry never shortens the interval
    now[0] = 100.5
    assert meter.quiet_since(100.0, 1.0) is False


def test_a_drained_cue_never_counts_toward_the_response(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    journey = passing_speak(tmp_path, monkeypatch, kwargs={"audio_at": {2: 5}})
    monkeypatch.setattr(client, "_SPEAK_SETTLE_SECONDS", 0.0)
    with pytest.raises(ValueError, match="journey_speak_no_audio"):
        speak(journey)
    assert journey.speak_receipt["agentAudioFrames"] == 0
    assert journey.speak_receipt["inputWindowAudioFrames"] == 0


def test_the_audio_source_is_queue_bounded_paced_and_played_out(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    journey = passing_speak(tmp_path, monkeypatch)
    monkeypatch.setattr(client, "_SPEAK_FRAME_SECONDS", 0.001)
    started = time.monotonic()
    speak(journey)
    source = journey.room.published[0][0].source
    assert source.queue_size_ms <= 200 and source.played_out is True
    frames = len(source.frames)
    assert time.monotonic() - started >= (frames - 1) * 0.001  # not dumped at once


def test_a_second_distinct_worker_audio_track_is_ambiguous(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    room = FakeRoom()
    voice = client.SyntheticVoice(fake_rtc(), room, SPEAK_WORKER)
    worker = SimpleNamespace(identity=SPEAK_WORKER)

    async def run():
        voice.meter.start()
        for sid in ("TR_one", "TR_one", "TR_two"):
            room.emit("track_subscribed", SimpleNamespace(kind="audio", sid=sid, tag=sid),
                      None, worker)
        await asyncio.gather(*voice.meter._tasks)

    asyncio.run(run())
    assert [track.tag for track in FakeAudioStream.tracks] == ["TR_one"]
    assert voice.meter.ambiguous is True


def test_speak_refuses_when_the_worker_track_was_ambiguous(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch)
    journey.voice.meter.ambiguous = True
    with pytest.raises(ValueError, match="journey_speak_audio_track"):
        speak(journey)


def test_measurement_windows_only_advance(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    meter = client.SyntheticVoice(fake_rtc(), FakeRoom(), SPEAK_WORKER).meter
    meter.advance("input")
    meter.advance("response")
    with pytest.raises(ValueError, match="journey_speak_window"):
        meter.advance("input")
    meter.advance("closed")
    with pytest.raises(ValueError, match="journey_speak_window"):
        meter.advance("response")


def test_speak_never_reads_server_monotonic_values(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch)
    inner = journey.request

    async def request(route, payload=None):
        value = await inner(route, payload)
        for row in value.get("events", []):
            row["monotonicMs"] = 9e12 if row["sequence"] % 2 else 0.5
        return value

    journey.request = request
    speak(journey)
    timings = {"agentAudioMs", "inputWindowAudioMs", "quietGateMs"}
    assert not any("ms" in key.lower() and key not in timings
                   for key in journey.speak_receipt)


def fixture_wav(tmp_path, *, channels=1, rate=22_050, frames=2_205, width=2):
    import wave

    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(channels)
        clip.setsampwidth(width)
        clip.setframerate(rate)
        clip.writeframes(bytes(width * channels * frames))
    return path


@pytest.mark.parametrize("shape", [
    {"channels": 2}, {"rate": 16_000}, {"width": 1}, {"frames": 22_050 * 5 + 1},
])
def test_fixture_shape_and_duration_are_bounded_even_when_the_hash_matches(
    tmp_path, monkeypatch, shape
):
    import hashlib

    import scripts.native_typed_journey as client

    path = fixture_wav(tmp_path, **shape)
    monkeypatch.setattr(client, "_SPEAK_FIXTURE", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(client, "_SPEAK_FIXTURE_SHA256", digest)
    with pytest.raises(ValueError, match="journey_speak_fixture"):
        client.load_speak_fixture()


def test_fixture_bytes_are_bounded_before_decoding(tmp_path, monkeypatch):
    import hashlib

    import scripts.native_typed_journey as client

    path = fixture_wav(tmp_path)
    monkeypatch.setattr(client, "_SPEAK_FIXTURE", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(client, "_SPEAK_FIXTURE_SHA256", digest)
    assert client.load_speak_fixture()
    monkeypatch.setattr(client, "_SPEAK_MAX_FIXTURE_BYTES", 100)
    with pytest.raises(ValueError, match="journey_speak_fixture"):
        client.load_speak_fixture()


def test_a_missing_fixture_is_a_categorized_refusal(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    monkeypatch.setattr(client, "_SPEAK_FIXTURE", tmp_path / "absent.wav")
    with pytest.raises(ValueError, match="journey_speak_fixture"):
        client.load_speak_fixture()


def test_a_second_activation_never_publishes_another_track_or_media_request(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    room = FakeRoom()
    voice = client.SyntheticVoice(fake_rtc(), room, SPEAK_WORKER)
    posted = []

    async def post(payload):
        posted.append(payload)
        return {"mediaIncarnation": payload["mediaIncarnation"], "version": 1}

    async def twice():
        await voice.activate(post)
        await voice.activate(post)

    with pytest.raises(ValueError, match="journey_speak_once"):
        asyncio.run(twice())
    assert len(room.published) == 1 and posted == [{"mediaIncarnation": 1}]


def test_speak_refuses_new_work(tmp_path, monkeypatch):
    task = ("task_state", {"taskId": "task_new_speak", "status": "active"})
    flow = speak_flow()
    flow[2].append(task)
    journey = speak_journey(tmp_path, monkeypatch, flow, audio_at={4: 6})
    with pytest.raises(ValueError, match="journey_followup_dispatched"):
        speak(journey)


def test_speak_requires_a_quiet_start_with_no_active_task(tmp_path, monkeypatch):
    journey = passing_speak(
        tmp_path, monkeypatch,
        kwargs={"before": [task_state("active")], "audio_at": {4: 6}},
    )
    with pytest.raises(ValueError, match="journey_speak_not_quiet"):
        speak(journey)
    assert journey.media == []


def test_speak_requires_playback_to_settle_first(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    journey = passing_speak(tmp_path, monkeypatch, kwargs={"settle": True, "audio_at": {4: 6}})
    monkeypatch.setattr(client, "_SPEAK_SETTLE_SECONDS", 60.0)
    with pytest.raises(ValueError, match="journey_speak_not_quiet"):
        speak(journey)
    assert journey.media == [] and journey.room.published == []


def test_speak_requires_no_open_foreground_turn_at_start(tmp_path, monkeypatch):
    before = [("first_foreground_token", {"turnId": "old_turn", "turnGeneration": 1})]
    journey = passing_speak(tmp_path, monkeypatch, kwargs={"before": before, "audio_at": {4: 6}})
    with pytest.raises(ValueError, match="journey_speak_not_quiet"):
        speak(journey)
    assert journey.media == [] and journey.room.published == []  # refused before publishing


def test_only_the_worker_identity_audio_is_counted(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    rtc, room = fake_rtc(), FakeRoom()
    voice = client.SyntheticVoice(rtc, room, SPEAK_WORKER)

    async def run():
        voice.meter.start()
        audio = SimpleNamespace(kind="audio", name="audio")
        for identity in ("someone_else", SPEAK_WORKER, "worker_preview_speak_two"):
            participant = SimpleNamespace(identity=identity)
            room.emit("track_subscribed", SimpleNamespace(kind="audio", tag=identity),
                      audio, participant)
        await asyncio.gather(*voice.meter._tasks)

    asyncio.run(run())
    assert [track.tag for track in FakeAudioStream.tracks] == [SPEAK_WORKER]
    assert voice.meter.totals() == (3, 30)


def test_only_the_worker_audio_publication_is_subscribed(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    rtc, room = fake_rtc(), FakeRoom()
    voice = client.SyntheticVoice(rtc, room, SPEAK_WORKER)
    asked = []
    publication = lambda kind, name: SimpleNamespace(  # noqa: E731
        kind=kind, set_subscribed=lambda value: asked.append((name, value))
    )
    voice.meter.start()
    room.emit("track_published", publication("audio", "other"), SimpleNamespace(identity="x"))
    room.emit("track_published", publication("video", "video"),
              SimpleNamespace(identity=SPEAK_WORKER))
    room.emit("track_published", publication("audio", "worker"),
              SimpleNamespace(identity=SPEAK_WORKER))
    assert asked == [("worker", True)]


def test_audio_frames_are_never_retained_or_written(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    meter = client.SyntheticVoice(fake_rtc(), FakeRoom(), SPEAK_WORKER).meter
    meter.observe(fake_frame(8000))
    meter.observe(fake_frame(0))
    assert meter.totals() == (1, 10)
    assert not any(isinstance(v, (bytes, bytearray, memoryview)) for v in vars(meter).values())
    assert list(tmp_path.iterdir()) == []


def test_scenario_accepts_one_exact_speak_step_and_refuses_a_second_or_extra_fields(tmp_path):
    def scenario(*steps):
        path = tmp_path / "speak.json"
        path.write_text(json.dumps({"version": 1, "steps": list(steps)}), encoding="utf-8")
        return load_scenario(path)

    assert scenario({"label": "hello", "kind": "speak"})[0]["kind"] == "speak"
    for bad in (
        [{"label": "a", "kind": "speak"}, {"label": "b", "kind": "speak"}],
        [{"label": "a", "kind": "speak", "text": "Hi"}],
        [{"label": "a", "kind": "speak", "task": "x"}],
    ):
        with pytest.raises(ValueError, match="journey_(speak|step)"):
            scenario(*bad)


def test_the_speak_fixture_is_pinned_by_sha256(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    assert client.load_speak_fixture()[1] == ("hello", "can", "you", "hear", "me")
    monkeypatch.setattr(client, "_SPEAK_FIXTURE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="journey_speak_fixture"):
        client.load_speak_fixture()


def test_the_fixture_is_streamed_as_ten_millisecond_frames_with_bounded_silence():
    import scripts.native_typed_journey as client

    frames, _ = client.load_speak_fixture()
    assert {len(frame) for frame in frames} == {960}
    assert 100 <= len(frames) <= 600
    assert frames[0] == bytes(960) and frames[-1] == bytes(960)


def test_media_route_is_allowed_on_the_v1_endpoint(monkeypatch):
    import scripts.native_typed_journey as client

    seen = []

    class Response:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return b'{"mediaIncarnation":1,"version":1}'

    def opened(request, **_kwargs):
        seen.append((request.full_url, request.data))
        return Response()

    monkeypatch.setattr(
        client.urllib.request, "build_opener", lambda *_: SimpleNamespace(open=opened),
    )
    code, value = client.request_json(
        "http://127.0.0.1:1", "media", token="t", payload={"mediaIncarnation": 1}
    )
    assert (code, value) == (202, {"mediaIncarnation": 1, "version": 1})
    assert seen == [("http://127.0.0.1:1/api/v1/media", b'{"mediaIncarnation": 1}')]
    with pytest.raises(ValueError, match="journey_route"):
        client.request_json("http://127.0.0.1:1", "audio-diagnostic")


@pytest.mark.parametrize("scenario_step", [SPEAK_STEP, {"label": "stop", "kind": "speak_barge"},
                                           {"label": "delegate", "kind": "speak_task"}],
                         ids=["speak", "speak_barge", "speak_task"])
def test_execute_gives_a_speak_journey_exactly_the_worker_bound_voice(
    tmp_path, monkeypatch, scenario_step
):
    import hashlib
    import sys

    import scripts.native_typed_journey as client
    from hermes_realtime import host_launcher

    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              browser_origin="http://127.0.0.1:1", api_origin="http://127.0.0.1:3",
                              livekit_url="ws://127.0.0.1:2")
    suffix = hashlib.sha256(str(tmp_path / "active.private.json").encode()).hexdigest()[:16]
    credential = dict(
        workerIdentity="worker_preview_" + suffix, roomName="hermes-preview-" + suffix,
        url=options.livekit_url, token="synthetic-lease", participantIdentity="synthetic",
        expiresInSeconds=3600,
    )
    rooms = []

    class Room(FakeRoom):
        def __init__(self):
            super().__init__()
            rooms.append(self)

        async def connect(self, url, token, options=None):
            self.options = options

        async def disconnect(self):
            return None

    rtc = fake_rtc()
    rtc.Room, rtc.RoomOptions = Room, lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "livekit", SimpleNamespace(rtc=rtc))
    monkeypatch.setattr(host_launcher, "_load_api_bearer", lambda _: "synthetic")
    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))

    def request(origin, route, **kwargs):
        if route == "events":
            return 200, {"version": 2, "events": []}
        return (200, credential) if route == "stable-bootstrap" else (200, {})

    monkeypatch.setattr(client, "request_json", request)
    seen = []

    async def step(self, value):
        seen.append(self.voice)
        raise ValueError("journey_stop_here")

    monkeypatch.setattr(client.Journey, "step", step)
    receipt = {"outcome": "refused"}
    with pytest.raises(ValueError, match="journey_stop_here"):
        asyncio.run(client.execute(options, [scenario_step], receipt))
    voice = seen[0]
    assert voice.meter.worker_identity == credential["workerIdentity"]
    assert rooms[0].options == {"auto_subscribe": False}
    assert rooms[0].published == []  # the mic track is published inside the step only
    assert set(rooms[0].handlers) == {"track_published", "track_subscribed"}
    # A typed-only scenario gets no voice, no handlers and no subscription.
    seen.clear()
    rooms.clear()
    with pytest.raises(ValueError, match="journey_stop_here"):
        asyncio.run(client.execute(options, [{"label": "a", "kind": "conversation",
                                              "text": "Hi"}], {"outcome": "refused"}))
    assert seen == [None] and rooms[0].handlers == {}


@pytest.mark.parametrize("data", [
    {"generation": 1, "mediaIncarnation": 2},  # a foreign or stale incarnation
    {"generation": 1},  # no incarnation
    {"generation": 1, "mediaIncarnation": True},  # bool is not an exact int
    {"generation": 1, "mediaIncarnation": "1"},
], ids=["foreign", "missing", "bool", "string"])
def test_speak_ignores_readiness_for_any_other_media_incarnation(tmp_path, monkeypatch, data):
    flow = speak_flow()
    flow[1] = [("voice_input_ready", data)]
    journey = speak_journey(tmp_path, monkeypatch, flow, audio_at={4: 6})
    with pytest.raises(ValueError, match="journey_speak_not_ready"):
        speak(journey)
    assert not any(any(f.data) for f in journey.room.published[0][0].source.frames)  # no clip


def test_speak_accepts_readiness_for_the_activated_incarnation_among_stale_ones(
    tmp_path, monkeypatch
):
    flow = speak_flow()
    flow[1] = [("voice_input_ready", {"generation": 1, "mediaIncarnation": 9}),
               ("voice_input_ready", READY)]
    speak(speak_journey(tmp_path, monkeypatch, flow, audio_at={4: 6}))


class HangingSource(FakeAudioSource):
    cancelled = False

    async def capture_frame(self, frame):
        try:
            await asyncio.Event().wait()  # never returns
        except asyncio.CancelledError:
            HangingSource.cancelled = True
            raise


def test_a_capture_that_never_returns_fails_the_step_and_still_runs_cleanup(
    tmp_path, monkeypatch
):
    import hashlib
    import sys

    import scripts.native_typed_journey as client
    from hermes_realtime import host_launcher

    for name, value in {"_SPEAK_FRAME_SECONDS": 0.0, "_SPEAK_STREAM_MARGIN": 0.2,
                        "_SPEAK_DRAIN_SECONDS": 0.0}.items():
        monkeypatch.setattr(client, name, value)
    HangingSource.cancelled = False
    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              browser_origin="http://127.0.0.1:1", api_origin="http://127.0.0.1:3",
                              livekit_url="ws://127.0.0.1:2")
    suffix = hashlib.sha256(str(tmp_path / "active.private.json").encode()).hexdigest()[:16]
    credential = dict(
        workerIdentity="worker_preview_" + suffix, roomName="hermes-preview-" + suffix,
        url=options.livekit_url, token="synthetic-lease", participantIdentity="synthetic",
        expiresInSeconds=3600,
    )
    cleanup = []

    class Room(FakeRoom):
        async def connect(self, url, token, options=None):
            return None

        async def disconnect(self):
            cleanup.append("disconnect")

    rtc = fake_rtc()
    rtc.Room, rtc.RoomOptions, rtc.AudioSource = Room, (lambda **kw: kw), HangingSource
    monkeypatch.setitem(sys.modules, "livekit", SimpleNamespace(rtc=rtc))
    monkeypatch.setattr(host_launcher, "_load_api_bearer", lambda _: "synthetic")
    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))
    batches = [[], [], [("voice_input_ready", READY)]]  # connect poll, flush poll, ready
    cursor = [1]

    def request(origin, route, **kwargs):
        if route == "stop":
            cleanup.append("stop")
        if route == "media":
            return 202, {"mediaIncarnation": 1, "version": 1}
        if route == "events":
            rows = numbered(batches.pop(0), cursor[0]) if batches else []
            cursor[0] += len(rows)
            return 200, {"version": 2, "events": rows}
        return (200, credential) if route == "stable-bootstrap" else (200, {})

    monkeypatch.setattr(client, "request_json", request)
    real_settle = client.Journey.settle_owned

    async def settle_owned(self):
        cleanup.append("settle_owned")
        return await real_settle(self)

    monkeypatch.setattr(client.Journey, "settle_owned", settle_owned)
    receipt = {"outcome": "refused"}
    with pytest.raises(ValueError, match="journey_speak_stream_timeout"):
        asyncio.run(asyncio.wait_for(client.execute(options, [SPEAK_STEP], receipt), 10))
    assert HangingSource.cancelled is True
    assert cleanup == ["settle_owned", "stop", "disconnect"]
    assert receipt["cleanup"]["sessionStopped"] and receipt["cleanup"]["roomDisconnected"]


def test_a_publish_that_never_returns_is_bounded(tmp_path, monkeypatch):
    journey = passing_speak(tmp_path, monkeypatch)

    async def hang(track, options):
        await asyncio.Event().wait()

    journey.room.local_participant.publish_track = hang
    with pytest.raises(ValueError, match="journey_speak_publish_timeout"):
        speak(journey)


def test_speak_after_a_completed_typed_step_does_not_claim_the_microphone_final(
    tmp_path, monkeypatch
):
    typed = ("typed_turn", 1)
    pair = {"turnId": typed[0], "turnGeneration": typed[1]}
    typed_batch = [
        admitted(sequence=1), user_final("Hi there"), ("first_foreground_token", pair),
        ("assistant_text_generated", pair), completed(turn="typed_turn", sequence=1),
    ]
    journey = speak_journey(tmp_path, monkeypatch, [typed_batch, *speak_flow()],
                            audio_at={5: 6})

    async def both():
        await journey.step({"label": "greeting", "kind": "conversation", "text": "Hi there"})
        await journey.step(SPEAK_STEP)

    asyncio.run(asyncio.wait_for(both(), 10))
    assert journey.speak_receipt["transcriptMatches"] is True
    assert journey.tracker.current is None and journey.tracker.text == ""
    assert [turn["label"] for turn in journey.tracker.turns] == ["greeting"]


def test_readiness_that_needs_pcm_arrives_after_the_silence_feed_starts(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    journey = speak_journey(tmp_path, monkeypatch, speak_flow(), audio_at={4: 6},
                            ready_needs_audio=True)
    seen = []
    original = client.SyntheticVoice.stream

    async def stream(self, *args):
        # The warm-up has settled and only silence has been captured so far.
        seen.append((self._idle_task, [bytes(f.data) for f in self.source.frames]))
        await original(self, *args)

    monkeypatch.setattr(client.SyntheticVoice, "stream", stream)
    speak(journey)
    source = journey.room.published[0][0].source
    assert len(source.frames) > len(client.load_speak_fixture()[0])  # silence preceded the clip
    (idle, before_clip), = seen
    assert idle is None and before_clip and not any(any(pcm) for pcm in before_clip)
    assert journey.voice._idle_task is None
    # Stopping the warm-up never disposes or replaces the shared AudioSource.
    assert journey.voice.source is source and source.closed is False


def test_without_a_silence_feed_a_pcm_gated_readiness_never_arrives(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    monkeypatch.setattr(client.SyntheticVoice, "start_idle", lambda self: None)
    journey = speak_journey(tmp_path, monkeypatch, speak_flow(), ready_needs_audio=True)
    with pytest.raises(ValueError, match="journey_speak_not_ready"):
        speak(journey)


def test_the_silence_feed_is_bounded(tmp_path, monkeypatch):
    flow = speak_flow()
    flow[1] = []  # readiness never arrives
    journey = speak_journey(tmp_path, monkeypatch, flow)
    with pytest.raises(ValueError, match="journey_speak_not_ready"):
        speak(journey)
    # (ready timeout 0.3 s + settle timeout 0.2 s) / 10 ms
    assert len(journey.room.published[0][0].source.frames) == 50


def idle_state_after_refusal(journey, category):
    """Run the step; read the idle feeder state while the loop is still alive (loop teardown
    would otherwise cancel a leaked task and hide a missing cancel)."""

    async def run():
        with pytest.raises(ValueError, match=category):
            await asyncio.wait_for(journey.step(SPEAK_STEP), 5)
        return RecordingIdle.cancelled, journey.voice._idle_task

    return asyncio.run(run())


class RecordingIdle:
    """A stand-in idle feeder that records its cancellation."""

    cancelled = False

    @staticmethod
    async def run(self):
        RecordingIdle.cancelled = False
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            RecordingIdle.cancelled = True
            raise


def test_the_silence_feed_is_cancelled_and_settled_when_readiness_never_arrives(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    monkeypatch.setattr(client.SyntheticVoice, "_idle", RecordingIdle.run)
    flow = speak_flow()
    flow[1] = []
    journey = speak_journey(tmp_path, monkeypatch, flow)
    assert idle_state_after_refusal(journey, "journey_speak_not_ready") == (True, None)


def test_the_silence_feed_is_settled_when_the_clip_stream_times_out(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    monkeypatch.setattr(client.SyntheticVoice, "_idle", RecordingIdle.run)
    journey = speak_journey(tmp_path, monkeypatch, speak_flow(), audio_at={4: 6})
    monkeypatch.setattr(client, "_SPEAK_STREAM_MARGIN", 0.2)
    journey.voice.rtc.AudioSource = HangingSource
    assert idle_state_after_refusal(journey, "journey_speak_stream_timeout") == (True, None)


def test_the_silence_feed_is_bounded_by_frames_and_by_bytes(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    for name, value, expected in (("_SPEAK_IDLE_MAX_FRAMES", 7, 7),
                                  ("_SPEAK_IDLE_MAX_BYTES", 3 * 960, 3)):
        flow = speak_flow()
        flow[1] = []  # readiness never arrives, so the feed runs to its bound
        journey = speak_journey(tmp_path, monkeypatch, flow)
        monkeypatch.setattr(client, name, value)
        with pytest.raises(ValueError, match="journey_speak_not_ready"):
            speak(journey)
        assert len(journey.room.published[0][0].source.frames) == expected
        monkeypatch.undo()


def test_the_silence_feed_is_bounded_by_time_when_capture_backpressures(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    class SlowSource(FakeAudioSource):
        async def capture_frame(self, frame):
            self.frames.append(frame)
            await asyncio.sleep(0.05)

    journey = speak_journey(tmp_path, monkeypatch, speak_flow())
    monkeypatch.setattr(client, "_SPEAK_READY_TIMEOUT", 0.15)
    monkeypatch.setattr(client, "_SPEAK_SETTLE_TIMEOUT", 0.05)  # 20 frames by count, 0.2 s
    voice = journey.voice
    voice.source = SlowSource(48_000, 1)
    started = time.monotonic()
    asyncio.run(voice._idle())
    assert time.monotonic() - started < 0.5 and len(voice.source.frames) <= 6


def test_only_one_pcm_producer_runs_at_a_time(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    voice = client.SyntheticVoice(fake_rtc(), FakeRoom(), SPEAK_WORKER)
    voice.source = FakeAudioSource(48_000, 1)
    frames = (bytes(960),) * 3

    async def run():
        voice.start_idle()
        with pytest.raises(ValueError, match="journey_speak_producer"):
            await voice.stream(frames)  # the warm-up has not settled
        with pytest.raises(ValueError, match="journey_speak_producer"):
            voice.start_idle()  # nor can a second warm-up start
        await voice.stop_idle()
        await voice.stream(frames)

    asyncio.run(run())
    assert len(voice.source.frames) >= 3


def test_the_silence_feed_is_settled_when_the_quiet_gate_fails(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    monkeypatch.setattr(client.SyntheticVoice, "_idle", RecordingIdle.run)
    cue = {call: 1 for call in range(3, 60)}  # a cue that never stops: the gate times out
    journey = speak_journey(tmp_path, monkeypatch, speak_flow(), audio_at=cue)
    monkeypatch.setattr(client, "_SPEAK_SETTLE_SECONDS", 0.1)
    assert idle_state_after_refusal(journey, "journey_speak_not_quiet") == (True, None)


def typed_conversation_step(label="greeting", text="Hi there"):
    pair = {"turnId": "typed_turn", "turnGeneration": 1}
    rows = [admitted(sequence=1), user_final(text), ("first_foreground_token", pair),
            ("assistant_text_generated", pair), completed(turn="typed_turn", sequence=1)]
    return {"label": label, "kind": "conversation", "text": text}, rows, pair


def run_steps(journey, *steps):
    async def go():
        for step in steps:
            await journey.step(step)

    asyncio.run(asyncio.wait_for(go(), 10))


def test_clearing_the_typed_step_keeps_identities_admissions_baselines_and_sequence(
    tmp_path, monkeypatch
):
    step, rows, pair = typed_conversation_step()
    journey = speak_journey(tmp_path, monkeypatch, [rows, *speak_flow()], audio_at={5: 6},
                            before=[task_state("completed")])
    run_steps(journey, step, SPEAK_STEP)
    assert (pair["turnId"], pair["turnGeneration"]) in journey.tracker.seen
    assert journey.sequence == 1 and journey.admissions == set() and journey.bindings == {}
    assert journey.tracker.tasks == {CANCEL_TASK: "completed"}  # the owned-task baseline
    assert [turn["label"] for turn in journey.tracker.turns] == ["greeting"]


@pytest.mark.parametrize("replayed", [
    [("assistant_turn_completed", {"turnId": "typed_turn", "turnGeneration": 1})],
    [("first_foreground_token", {"turnId": "typed_turn", "turnGeneration": 1})],
])
def test_a_replayed_prior_pair_during_speak_is_rejected(tmp_path, monkeypatch, replayed):
    step, rows, _ = typed_conversation_step()
    flow = speak_flow()
    flow[2] = [*flow[2], *replayed]  # arrives with the microphone's user final
    journey = speak_journey(tmp_path, monkeypatch, [rows, *flow], audio_at={5: 6})
    with pytest.raises(ValueError, match="journey_speak_concurrent"):
        run_steps(journey, step, SPEAK_STEP)


def test_a_prior_pair_cannot_stand_in_for_the_microphone_turn(tmp_path, monkeypatch):
    step, rows, pair = typed_conversation_step()
    journey = speak_journey(tmp_path, monkeypatch,
                            [rows, *speak_flow(pairs=(pair,))], audio_at={5: 6})
    with pytest.raises(ValueError, match="journey_speak_pair_unbound"):
        run_steps(journey, step, SPEAK_STEP)


def test_speak_rejects_a_previous_typed_turn_that_is_still_active(tmp_path, monkeypatch):
    journey = speak_journey(tmp_path, monkeypatch, speak_flow())
    journey.tracker.begin("greeting", 1, "Hi there")  # admitted, never completed
    with pytest.raises(ValueError, match="journey_speak_previous_turn_active"):
        speak(journey)
    # It was rejected, not erased: the active turn is still the tracker's current step.
    assert journey.tracker.current is not None and journey.tracker.text == "Hi there"
    assert journey.media == []


def test_a_failed_warmup_fails_the_step_even_when_readiness_arrives_without_it(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    async def failing_idle(self):
        raise RuntimeError("capture failed")

    monkeypatch.setattr(client.SyntheticVoice, "_idle", failing_idle)
    # Readiness arrives anyway (implicit transport silence): the step must still fail.
    journey = speak_journey(tmp_path, monkeypatch, speak_flow(), audio_at={4: 6})

    async def run():
        with pytest.raises(ValueError, match="journey_speak_warmup_failed"):
            await asyncio.wait_for(journey.step(SPEAK_STEP), 5)
        # Cleanup still ran: nothing is left feeding, tracing or windowed.
        return (journey.voice._idle_task, journey.tracker.speak, journey.voice.meter.window,
                journey.room.published[0][0].source.frames)

    idle, trace, window, frames = asyncio.run(run())
    assert idle is None and trace is None and window == "closed"
    assert not any(any(frame.data) for frame in frames)  # the clip was never streamed

# Synthetic barge-in `speak_barge` step (#237): a long reply is requested, then the pinned
# Stop clip is injected over it on the same microphone track. Controlled synthetic path only.

BARGE_STEP = {"label": "stop", "kind": "speak_barge"}
TARGET_PAIR = {"turnId": "target_turn", "turnGeneration": 1}
FRESH_PAIR = {"turnId": "fresh_turn", "turnGeneration": 1}
OTHER_PAIR = {"turnId": "other_turn", "turnGeneration": 1}
LONG_TEXT = "Please tell me a long story about the ocean."
CLOCK_STEP = 0.25  # fake client seconds per events poll


def user_say(text):
    return ("transcript_final", {"role": "user", "text": text})


def barge_pre(*, playable=True, complete=False, text=LONG_TEXT):
    token = [("first_foreground_token", TARGET_PAIR), ("assistant_text_generated", TARGET_PAIR)]
    if playable:
        token.append(("first_playable_audio", TARGET_PAIR))
    if complete:
        token.append(("assistant_turn_completed", TARGET_PAIR))
    return [
        [], [("voice_input_ready", READY)],
        [("voice_activity_started", {}), ("voice_activity_ended", {}), user_say(text)],
        token,
    ]


def barge_post(*, vad=True, finals=("Stop.",), interrupt=True, silenced=True, interrupted=True,
               completed=False, extra=(), request_turn="target_turn", merge=False):
    first = [("voice_activity_started", {})] if vad else []
    second = [("voice_activity_ended", {}), *[user_say(text) for text in finals]]
    if interrupt:
        second.append(("interrupt_requested", {"turnId": request_turn, "turnGeneration": 1}))
    if silenced:
        second.append(("playback_silenced", {}))
    if interrupted:
        second.append(("assistant_turn_interrupted", TARGET_PAIR))
    if completed:
        second.append(("assistant_turn_completed", TARGET_PAIR))
    if merge:  # everything arrives on the first post-injection poll
        return [first + second, *extra]
    return [first, second, *extra]


def fresh_token(pair=FRESH_PAIR):
    return [("first_foreground_token", pair), ("assistant_text_generated", pair)]


def barge_journey(tmp_path, monkeypatch, *, pre=None, post=None, response_frames=3,
                  quiesce_audio_polls=0, after_frames=0, before=()):
    """A Journey with a fake mic transport and a state-driven fake host.

    Events before the Stop's injection start come from ``pre``; afterwards from ``post``, one
    batch per poll. Worker audio is injected per poll by window: ``response_frames`` while
    TARGET plays, ``quiesce_audio_polls`` polls of audio right after the injection ends, and
    ``after_frames`` once the quiescence window has closed. The client clock is fake and
    advances CLOCK_STEP per poll, so every timing assertion is exact.
    """
    import scripts.native_typed_journey as client

    for name, value in {
        "_SPEAK_FRAME_SECONDS": 0.0, "_SPEAK_DRAIN_SECONDS": 0.0, "_SPEAK_TURN_TIMEOUT": 1.0,
        "_SPEAK_SETTLE_TIMEOUT": 0.5, "_SPEAK_READY_TIMEOUT": 1.0, "_SPEAK_SETTLE_SECONDS": 0.0,
        "_BARGE_MIN_TARGET_MS": 50, "_BARGE_BUDGET_MS": 500, "_BARGE_QUIET_MS": 400,
        "_BARGE_PRECONDITION_TIMEOUT": 0.4, "_BARGE_INTERRUPT_TIMEOUT": 1.0,
    }.items():
        monkeypatch.setattr(client, name, value)
    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              api_origin="http://127.0.0.1:1")
    journey = Journey(options, {"expiresInSeconds": 3600}, "secret")
    monkeypatch.setattr(client, "read_journal", lambda _: (set(), 0))
    rtc, room = fake_rtc(), FakeRoom()
    journey.voice = client.SyntheticVoice(rtc, room, SPEAK_WORKER)
    meter = journey.voice.meter
    now, cursor, quiesce_polls = [1000.0], [len(before) + 1], [0]
    if before:
        journey.tracker.consume(numbered(list(before), 1))
    meter._clock = lambda: now[0]
    queues = {"pre": list(barge_pre() if pre is None else pre),
              "post": list(barge_post() if post is None else post)}
    journey.media, journey.rtc, journey.room = [], rtc, room

    async def request(route, payload=None):
        if route == "media":
            journey.media.append(payload)
            return {"mediaIncarnation": payload["mediaIncarnation"], "version": 1}
        if route == "input":
            return {}
        assert route == "events"
        now[0] += CLOCK_STEP
        window = meter.window
        frames = 0
        if window == "response":
            frames = response_frames
        elif window == "quiesce":
            quiesce_polls[0] += 1
            frames = 3 if quiesce_polls[0] <= quiesce_audio_polls else 0
        elif window == "after":
            frames = after_frames
        for _ in range(frames):
            meter.observe(fake_frame(8000))
        queue = queues["post" if window in {"stop", "quiesce", "after"} else "pre"]
        rows = numbered(queue.pop(0) if queue else [], cursor[0])
        cursor[0] += len(rows)
        return {"version": 2, "events": rows}

    journey.request = request
    return journey


def barge(journey, seconds=10):
    asyncio.run(asyncio.wait_for(journey.step(BARGE_STEP), seconds))


def refused(journey, category):
    with pytest.raises(ValueError, match=category):
        barge(journey)
    return journey.speak_receipt


def test_barge_passes_when_stop_is_recognized_target_is_interrupted_and_audio_goes_quiet(
    tmp_path, monkeypatch
):
    journey = barge_journey(tmp_path, monkeypatch)
    barge(journey)
    receipt = journey.speak_receipt
    assert receipt["stopInjected"] and receipt["stopRecognized"]
    assert receipt["targetInterrupted"] is True and receipt["targetCompleted"] is False
    assert receipt["interruptEventObserved"] is True
    assert receipt["quiescenceObserved"] is True and receipt["quiescenceInconclusive"] is False
    assert receipt["freshReplyObserved"] is False
    assert receipt["targetAudioMsBeforeStop"] >= 50
    assert receipt["postStopAudioMs"] == 0 and receipt["stopLatencyMs"] == 0
    assert receipt["quietTailMs"] >= 400


def test_barge_uses_one_track_one_media_request_and_both_pinned_clips(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    journey = barge_journey(tmp_path, monkeypatch)
    barge(journey)
    target, _, stop, _, _ = client.load_barge_fixtures()
    assert journey.media == [{"mediaIncarnation": 1}]
    assert len(journey.room.published) == 1
    pcm = [frame.data for frame in journey.room.published[0][0].source.frames]
    assert pcm[-(len(target) + len(stop)):] == list(target) + list(stop)  # after idle silence


def test_barge_receipt_is_metadata_only(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch)
    barge(journey)
    raw = json.dumps(journey.speak_receipt).lower()
    for private in ("ocean", "stop.", "target_turn", "story"):
        assert private not in raw
    assert all(type(value) in {bool, int, str} for value in journey.speak_receipt.values())


def test_barge_measures_stop_latency_from_injection_end_on_the_client_clock(
    tmp_path, monkeypatch
):
    journey = barge_journey(tmp_path, monkeypatch, quiesce_audio_polls=2)
    barge(journey)
    receipt = journey.speak_receipt
    assert receipt["stopLatencyMs"] == 500  # last audio two polls (2 x 250 ms) after the end
    assert receipt["postStopAudioMs"] == 60
    assert receipt["quiescenceObserved"] is True


def test_barge_declared_thresholds_are_the_reviewed_values():
    import scripts.native_typed_journey as client

    assert (client._BARGE_MIN_TARGET_MS, client._BARGE_BUDGET_MS, client._BARGE_QUIET_MS) == (
        1500, 1500, 500
    )


def test_a_target_that_ends_first_is_inconclusive_and_nothing_is_injected(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, pre=barge_pre(complete=True))
    receipt = refused(journey, "journey_barge_target_terminal")
    assert receipt["stopInjected"] is False and receipt["targetInterrupted"] is False
    import scripts.native_typed_journey as client

    # Only the first clip was ever streamed (after the idle silence, which is all zero).
    pcm = [frame.data for frame in journey.room.published[0][0].source.frames]
    assert pcm[-len(client.load_barge_fixtures()[0]):] == list(client.load_barge_fixtures()[0])
    assert not any(any(data) for data in pcm[:-len(client.load_barge_fixtures()[0])])


def test_stop_is_not_injected_before_target_has_enough_audio(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, response_frames=0)
    receipt = refused(journey, "journey_barge_precondition")
    assert receipt["stopInjected"] is False


def test_stop_is_not_injected_before_target_is_playable(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, pre=barge_pre(playable=False))
    receipt = refused(journey, "journey_barge_precondition")
    assert receipt["stopInjected"] is False


def test_stop_needs_a_fresh_voice_start_after_injection(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, post=barge_post(vad=False))
    receipt = refused(journey, "journey_barge_stop_vad")
    assert receipt["stopInjected"] is True and receipt["stopRecognized"] is False


@pytest.mark.parametrize("finals,category", [
    ((), "journey_barge_stop_finals"),
    (("Stop.", "Stop."), "journey_barge_stop_finals"),
    (("Go on.",), "journey_barge_stop_transcript"),
    (("Stop it now.",), "journey_barge_stop_transcript"),
])
def test_stop_needs_exactly_one_final_with_the_exact_words(tmp_path, monkeypatch, finals, category):
    journey = barge_journey(tmp_path, monkeypatch, post=barge_post(finals=finals))
    assert refused(journey, category)["stopRecognized"] is False


def test_stop_words_are_normalized_by_case_and_punctuation_only(tmp_path, monkeypatch):
    barge(barge_journey(tmp_path, monkeypatch, post=barge_post(finals=("STOP!",))))


def test_an_interruption_of_a_different_turn_is_ambiguous(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, post=barge_post(request_turn="other_turn"))
    refused(journey, "journey_barge_ambiguous")


def test_an_interrupt_request_for_another_generation_is_ambiguous(tmp_path, monkeypatch):
    post = barge_post(interrupt=False)
    post[1].append(("interrupt_requested", {"turnId": "target_turn", "turnGeneration": 2}))
    refused(barge_journey(tmp_path, monkeypatch, post=post), "journey_barge_ambiguous")


def test_another_pair_active_while_target_is_running_is_ambiguous(tmp_path, monkeypatch):
    post = barge_post(interrupted=False, completed=False)
    post[0].extend(fresh_token(OTHER_PAIR))  # a second pair starts before TARGET ends
    post[1].append(("assistant_turn_interrupted", TARGET_PAIR))
    refused(barge_journey(tmp_path, monkeypatch, post=post), "journey_barge_ambiguous")


def test_two_fresh_pairs_are_ambiguous(tmp_path, monkeypatch):
    post = barge_post(extra=(fresh_token(), fresh_token(OTHER_PAIR)))
    refused(barge_journey(tmp_path, monkeypatch, post=post), "journey_barge_ambiguous")


def test_an_unbound_assistant_terminal_after_injection_is_ambiguous(tmp_path, monkeypatch):
    post = barge_post()
    post[1].append(("assistant_turn_completed", {}))
    refused(barge_journey(tmp_path, monkeypatch, post=post), "journey_barge_ambiguous")


def test_a_target_that_completes_is_never_an_interruption(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, post=barge_post(interrupted=False,
                                                                   completed=True))
    receipt = refused(journey, "journey_barge_target_completed")
    assert receipt["targetCompleted"] is True and receipt["targetInterrupted"] is False


def test_a_target_that_both_interrupts_and_completes_fails(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, post=barge_post(completed=True))
    refused(journey, "journey_barge_target_completed")


def test_a_target_that_is_never_interrupted_fails(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, post=barge_post(interrupted=False))
    receipt = refused(journey, "journey_barge_target_not_interrupted")
    assert receipt["targetInterrupted"] is False


def test_the_exact_pair_interruption_needs_an_interrupt_request_for_its_turn(
    tmp_path, monkeypatch
):
    journey = barge_journey(tmp_path, monkeypatch,
                            post=barge_post(interrupt=False, silenced=False))
    receipt = refused(journey, "journey_barge_no_interrupt_event")
    assert receipt["interruptEventObserved"] is False
    assert receipt["playbackSilencedObserved"] is False


def test_an_unanchored_playback_silence_never_satisfies_the_interrupt_requirement(
    tmp_path, monkeypatch
):
    # playback_silenced has an empty payload: it cannot be tied to TARGET.
    journey = barge_journey(tmp_path, monkeypatch,
                            post=barge_post(interrupt=False, silenced=True))
    receipt = refused(journey, "journey_barge_no_interrupt_event")
    assert receipt["interruptEventObserved"] is False
    assert receipt["playbackSilencedObserved"] is True


def test_an_interrupt_request_for_target_suffices_and_silence_is_only_supporting(
    tmp_path, monkeypatch
):
    journey = barge_journey(tmp_path, monkeypatch,
                            post=barge_post(interrupt=True, silenced=False))
    barge(journey)
    assert journey.speak_receipt["interruptEventObserved"] is True
    assert journey.speak_receipt["playbackSilencedObserved"] is False


def test_audio_that_continues_past_the_budget_fails(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, quiesce_audio_polls=3)
    receipt = refused(journey, "journey_barge_audio_continued")
    assert receipt["stopLatencyMs"] == 750 and receipt["quiescenceObserved"] is False


def test_a_fresh_reply_before_a_whole_quiet_interval_is_inconclusive_not_a_pass(
    tmp_path, monkeypatch
):
    post = barge_post(merge=True)  # the fresh token arrives on the first quiesce poll
    post[0].extend(fresh_token())
    journey = barge_journey(tmp_path, monkeypatch, post=post, after_frames=3)
    receipt = refused(journey, "journey_barge_quiescence_inconclusive")
    assert receipt["quiescenceInconclusive"] is True and receipt["quiescenceObserved"] is False
    assert receipt["freshReplyObserved"] is True


def test_a_fresh_reply_after_a_whole_quiet_interval_passes_and_its_audio_is_not_counted(
    tmp_path, monkeypatch
):
    post = barge_post(extra=([], fresh_token()))  # fresh token on the second quiesce poll
    journey = barge_journey(tmp_path, monkeypatch, post=post, after_frames=3)
    barge(journey)
    receipt = journey.speak_receipt
    assert receipt["freshReplyObserved"] is True and receipt["quiescenceObserved"] is True
    assert receipt["postStopAudioMs"] == 0  # audio after the fresh token is the fresh pair's


def test_a_fresh_pair_that_precedes_its_interruption_in_order_is_ambiguous(tmp_path, monkeypatch):
    post = barge_post(interrupted=False, merge=True)
    post[0].extend(fresh_token())
    post[0].append(("assistant_turn_interrupted", TARGET_PAIR))  # after the fresh token
    refused(barge_journey(tmp_path, monkeypatch, post=post), "journey_barge_ambiguous")


def test_barge_refuses_new_work(tmp_path, monkeypatch):
    post = barge_post()
    post[1].append(("task_state", {"taskId": "task_new_barge", "status": "active"}))
    refused(barge_journey(tmp_path, monkeypatch, post=post), "journey_followup_dispatched")


def test_barge_refuses_an_ambiguous_worker_track(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch)
    journey.voice.meter.ambiguous = True
    refused(journey, "journey_speak_audio_track")


def test_barge_refuses_a_first_utterance_that_is_not_the_exact_request(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, pre=barge_pre(text="Hello there"))
    receipt = refused(journey, "journey_speak_transcript")
    assert receipt["stopInjected"] is False


def test_scenario_accepts_one_barge_step_and_never_a_second_microphone_step(tmp_path):
    def scenario(*steps):
        path = tmp_path / "barge.json"
        path.write_text(json.dumps({"version": 1, "steps": list(steps)}), encoding="utf-8")
        return load_scenario(path)

    assert scenario({"label": "stop", "kind": "speak_barge"})[0]["kind"] == "speak_barge"
    for bad in (
        [{"label": "a", "kind": "speak"}, {"label": "b", "kind": "speak_barge"}],
        [{"label": "a", "kind": "speak_barge"}, {"label": "b", "kind": "speak_barge"}],
        [{"label": "a", "kind": "speak_barge", "text": "Stop"}],
    ):
        with pytest.raises(ValueError, match="journey_(speak|step)"):
            scenario(*bad)


@pytest.mark.parametrize("name", ["_BARGE_TARGET_SHA256", "_BARGE_STOP_SHA256"])
def test_the_barge_fixtures_are_pinned_by_sha256(monkeypatch, name):
    import scripts.native_typed_journey as client

    assert client.load_barge_fixtures()[1] == (
        "please", "tell", "me", "a", "long", "story", "about", "the", "ocean"
    )
    monkeypatch.setattr(client, name, "0" * 64)
    with pytest.raises(ValueError, match="journey_speak_fixture"):
        client.load_barge_fixtures()


def test_the_barge_silence_feed_is_cancelled_and_settled_when_readiness_never_arrives(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    monkeypatch.setattr(client.SyntheticVoice, "_idle", RecordingIdle.run)
    pre = barge_pre()
    pre[1] = []  # readiness never arrives
    journey = barge_journey(tmp_path, monkeypatch, pre=pre)

    async def run():
        with pytest.raises(ValueError, match="journey_speak_not_ready"):
            await asyncio.wait_for(journey.step(BARGE_STEP), 5)
        return RecordingIdle.cancelled, journey.voice._idle_task

    assert asyncio.run(run()) == (True, None)


def test_a_barge_after_a_completed_typed_step_does_not_claim_the_microphone_final(
    tmp_path, monkeypatch
):
    typed_pair = {"turnId": "typed_turn", "turnGeneration": 1}
    typed = [admitted(sequence=1), user_final("Hi there"),
             ("first_foreground_token", typed_pair), ("assistant_text_generated", typed_pair),
             completed(turn="typed_turn", sequence=1)]
    journey = barge_journey(tmp_path, monkeypatch, pre=[typed, *barge_pre()])

    async def both():
        await journey.step({"label": "greeting", "kind": "conversation", "text": "Hi there"})
        await journey.step(BARGE_STEP)

    asyncio.run(asyncio.wait_for(both(), 10))
    assert journey.speak_receipt["quiescenceObserved"] is True


@pytest.mark.parametrize("generation", [True, 1.0, None, "1", 2, "absent"],
                         ids=["bool", "float", "none", "string", "other-int", "absent"])
def test_an_interrupt_request_needs_an_exact_int_generation_equal_to_targets(
    tmp_path, monkeypatch, generation
):
    data = {"turnId": "target_turn"}
    if generation != "absent":
        data["turnGeneration"] = generation
    post = barge_post(interrupt=False)
    post[1].append(("interrupt_requested", data))
    journey = barge_journey(tmp_path, monkeypatch, post=post)
    receipt = refused(journey, "journey_barge_ambiguous")
    assert receipt["interruptEventObserved"] is False


def test_an_interrupt_request_with_a_non_string_turn_is_foreign(tmp_path, monkeypatch):
    post = barge_post(interrupt=False)
    post[1].append(("interrupt_requested", {"turnId": ["target_turn"], "turnGeneration": 1}))
    refused(barge_journey(tmp_path, monkeypatch, post=post), "journey_barge_ambiguous")


def test_a_replayed_prior_pair_is_not_a_fresh_reply_and_cannot_close_the_window(
    tmp_path, monkeypatch
):
    before = [("assistant_turn_completed", OTHER_PAIR)]  # seen before the step
    post = barge_post(extra=([], [("first_foreground_token", OTHER_PAIR)]))
    journey = barge_journey(tmp_path, monkeypatch, post=post, before=before, after_frames=3)
    receipt = refused(journey, "journey_barge_ambiguous")
    assert receipt["freshReplyObserved"] is False and receipt["quiescenceObserved"] is False


def test_a_new_pair_whose_first_token_precedes_the_stop_final_is_not_a_fresh_reply(
    tmp_path, monkeypatch
):
    interrupt = ("interrupt_requested", {"turnId": "target_turn", "turnGeneration": 1})
    post = [
        [("voice_activity_started", {}), interrupt, ("assistant_turn_interrupted", TARGET_PAIR),
         *fresh_token()],
        [("voice_activity_ended", {}), user_say("Stop.")],
    ]
    receipt = refused(barge_journey(tmp_path, monkeypatch, post=post),
                      "journey_barge_ambiguous")
    assert receipt["freshReplyObserved"] is False


def test_a_target_that_ends_during_the_lead_in_aborts_before_any_stop_speech(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    # The completion is delivered at the first poll of the Stop stream, i.e. during its
    # leading silence, after the pre-injection precondition already passed.
    pre = [*barge_pre(), [], [("assistant_turn_completed", TARGET_PAIR)]]
    journey = barge_journey(tmp_path, monkeypatch, pre=pre)
    receipt = refused(journey, "journey_barge_target_terminal")
    assert receipt["stopInjected"] is False and receipt["targetInterrupted"] is False
    sent = [frame.data for frame in journey.room.published[0][0].source.frames]
    target, _, _, _, _ = client.load_barge_fixtures()
    assert sum(any(pcm) for pcm in sent) == sum(any(pcm) for pcm in target)  # no Stop speech


def test_an_empty_interrupt_request_is_not_target_evidence_and_not_foreign(tmp_path, monkeypatch):
    # The host projects {} when renderer authority does not match: unattributed.
    post = barge_post(interrupt=False)
    post[1].append(("interrupt_requested", {}))
    journey = barge_journey(tmp_path, monkeypatch, post=post)
    receipt = refused(journey, "journey_barge_no_interrupt_event")
    assert receipt["interruptEventObserved"] is False
    assert receipt["playbackSilencedObserved"] is True


def test_an_interrupt_request_turn_id_must_be_an_exact_string(tmp_path, monkeypatch):
    class LoudStr(str):
        pass

    post = barge_post(interrupt=False)
    post[1].append(("interrupt_requested",
                    {"turnId": LoudStr("target_turn"), "turnGeneration": 1}))
    refused(barge_journey(tmp_path, monkeypatch, post=post), "journey_barge_ambiguous")


SECRET_PARTIAL = "secret words here"
ALL_COUNT_KEYS = (
    "userPartialCount", "echoSuppressedCount", "transcriptEchoSuppressedCount",
    "speechEndedCount", "voiceActivityEndedCount", "bargeInNonSpeechSuppressedCount",
    "bargeInVerifierUnavailableCount", "echoBargeInConfirmedCount",
    "interruptTargetCount", "interruptEmptyCount", "interruptOtherCount",
)


def diagnostic_events():
    """One event of every counted kind, to be placed before or after the injection."""
    return [
        ("transcript_partial", {"role": "user", "text": SECRET_PARTIAL}),
        ("echo_suppressed", {}), ("transcript_echo_suppressed", {}), ("speech_ended", {}),
        ("voice_activity_ended", {}), ("barge_in_non_speech_suppressed", {}),
        ("barge_in_verifier_unavailable", {}), ("echo_barge_in_confirmed", {}),
        ("interrupt_requested", {}),
    ]


def counted_barge(tmp_path, monkeypatch, *, post_extra=(), pre_extra=()):
    pre = barge_pre()
    pre[3] = [*pre[3], *pre_extra]
    post = barge_post()
    post[1] = [*post[1], *post_extra]
    return barge_journey(tmp_path, monkeypatch, pre=pre, post=post)


def test_barge_counts_each_diagnostic_kind_only_after_the_injection(tmp_path, monkeypatch):
    # The same kinds before the injection must not be counted.
    journey = counted_barge(tmp_path, monkeypatch, pre_extra=diagnostic_events(),
                            post_extra=[*diagnostic_events(), ("echo_suppressed", {})])
    barge(journey)
    receipt = journey.speak_receipt
    assert receipt["userPartialCount"] == 1
    assert receipt["echoSuppressedCount"] == 2  # one more post-injection event of that kind
    assert receipt["transcriptEchoSuppressedCount"] == 1
    assert receipt["speechEndedCount"] == 1
    assert receipt["voiceActivityEndedCount"] == 2  # barge_post's own plus the extra
    assert receipt["bargeInNonSpeechSuppressedCount"] == 1
    assert receipt["bargeInVerifierUnavailableCount"] == 1
    assert receipt["echoBargeInConfirmedCount"] == 1
    assert (receipt["interruptTargetCount"], receipt["interruptEmptyCount"],
            receipt["interruptOtherCount"]) == (1, 1, 0)


def test_barge_counts_are_zero_when_nothing_follows_the_injection(tmp_path, monkeypatch):
    journey = counted_barge(tmp_path, monkeypatch, pre_extra=diagnostic_events())
    barge(journey)
    receipt = journey.speak_receipt
    # barge_post's own ended event and strict interrupt are the only post-injection ones.
    assert {key: receipt[key] for key in ALL_COUNT_KEYS if key not in {
        "voiceActivityEndedCount", "interruptTargetCount"}} == {
        key: 0 for key in ALL_COUNT_KEYS
        if key not in {"voiceActivityEndedCount", "interruptTargetCount"}}
    assert receipt["userPartialAtLeast3Alnum"] is False


def test_barge_splits_interrupt_requests_into_target_empty_and_other(tmp_path, monkeypatch):
    other = ("interrupt_requested", {"turnId": "other_turn", "turnGeneration": 1})
    journey = counted_barge(tmp_path, monkeypatch, post_extra=[("interrupt_requested", {}), other])
    receipt = refused(journey, "journey_barge_ambiguous")  # the verdict is unchanged
    assert (receipt["interruptTargetCount"], receipt["interruptEmptyCount"],
            receipt["interruptOtherCount"]) == (1, 1, 1)


def test_barge_user_partials_are_counted_by_role_and_alnum_without_storing_text(
    tmp_path, monkeypatch
):
    extra = [("transcript_partial", {"role": "assistant", "text": SECRET_PARTIAL}),
             ("transcript_partial", {"role": "user", "text": "a!"})]
    journey = counted_barge(tmp_path, monkeypatch, post_extra=extra)
    barge(journey)
    receipt = journey.speak_receipt
    assert receipt["userPartialCount"] == 1  # the assistant partial is not a user partial
    assert receipt["userPartialAtLeast3Alnum"] is False  # "a!" has one alphanumeric


def test_barge_flags_a_user_partial_with_three_alphanumerics_and_keeps_no_text(
    tmp_path, monkeypatch
):
    extra = [("transcript_partial", {"role": "user", "text": SECRET_PARTIAL})]
    journey = counted_barge(tmp_path, monkeypatch, post_extra=extra)
    barge(journey)
    receipt = journey.speak_receipt
    assert receipt["userPartialCount"] == 1 and receipt["userPartialAtLeast3Alnum"] is True
    assert "secret" not in json.dumps(receipt).lower()
    assert all(type(value) in {bool, int, str} for value in receipt.values())


def test_barge_diagnostics_never_change_the_verdict(tmp_path, monkeypatch):
    # A refused run keeps its category with every diagnostic kind present.
    post = barge_post(vad=False)
    post[1] = [*post[1], *diagnostic_events()]
    journey = barge_journey(tmp_path, monkeypatch, post=post)
    receipt = refused(journey, "journey_barge_stop_vad")
    assert receipt["userPartialCount"] == 1 and receipt["echoSuppressedCount"] == 1


def test_another_pair_that_starts_during_the_precondition_wait_is_ambiguous(tmp_path, monkeypatch):
    # TARGET accumulates worker audio while a second foreground pair starts: that audio
    # can no longer qualify TARGET.
    pre = [*barge_pre(), fresh_token(OTHER_PAIR)]
    journey = barge_journey(tmp_path, monkeypatch, pre=pre)
    receipt = refused(journey, "journey_barge_ambiguous")
    assert receipt["stopInjected"] is False
    assert receipt["targetAudioMsBeforeStop"] >= 50  # the audio was heard but not accepted
    import scripts.native_typed_journey as client

    pcm = [frame.data for frame in journey.room.published[0][0].source.frames]
    target = client.load_barge_fixtures()[0]
    assert pcm[-len(target):] == list(target)  # refused before any Stop lead-in frame


def test_another_pair_that_starts_during_the_stop_lead_in_is_ambiguous_and_nothing_is_sent(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    # Delivered at the first poll of the Stop stream, i.e. during its leading silence.
    pre = [*barge_pre(), [], fresh_token(OTHER_PAIR)]
    journey = barge_journey(tmp_path, monkeypatch, pre=pre)
    receipt = refused(journey, "journey_barge_ambiguous")
    assert receipt["stopInjected"] is False
    sent = [frame.data for frame in journey.room.published[0][0].source.frames]
    target = client.load_barge_fixtures()[0]
    assert sum(any(pcm) for pcm in sent) == sum(any(pcm) for pcm in target)  # no Stop speech


@pytest.mark.parametrize("extra", [
    [("assistant_turn_completed", {})],  # an unbound foreground terminal
    [("first_foreground_token", {})],
], ids=["unbound-terminal", "unbound-token"])
def test_an_unbound_foreground_event_after_binding_is_ambiguous(tmp_path, monkeypatch, extra):
    pre = [*barge_pre(), extra]
    receipt = refused(barge_journey(tmp_path, monkeypatch, pre=pre), "journey_barge_ambiguous")
    assert receipt["stopInjected"] is False  # refused before the Stop was sent


def test_a_second_user_final_after_binding_is_ambiguous(tmp_path, monkeypatch):
    pre = [*barge_pre(), [user_say("Please tell me something else")]]
    receipt = refused(barge_journey(tmp_path, monkeypatch, pre=pre), "journey_barge_ambiguous")
    assert receipt["stopInjected"] is False  # refused before the Stop was sent


def test_targets_own_events_after_binding_are_not_ambiguity(tmp_path, monkeypatch):
    own = [("assistant_text_generated", TARGET_PAIR), ("first_playable_audio", TARGET_PAIR)]
    barge(barge_journey(tmp_path, monkeypatch, pre=[*barge_pre(), own]))


def test_target_terminal_keeps_what_is_known_about_target(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, pre=barge_pre(complete=True))
    receipt = refused(journey, "journey_barge_target_terminal")
    assert receipt["targetCompleted"] is True and receipt["targetInterrupted"] is False
    assert receipt["stopInjected"] is False and receipt["targetAudioMsBeforeStop"] >= 30


def test_stage_flags_are_null_when_target_was_never_established(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch, pre=barge_pre(text="Hello there"))
    receipt = refused(journey, "journey_speak_transcript")
    assert receipt["targetCompleted"] is None and receipt["targetInterrupted"] is None
    assert receipt["postStopAudioMs"] is None and receipt["stopLatencyMs"] is None
    assert receipt["quietTailMs"] is None
    assert receipt["targetAudioMsBeforeStop"] == 30  # the audio heard before the refusal


def test_a_quiescence_timeout_keeps_the_stop_recognition_and_interrupt_evidence(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    journey = barge_journey(tmp_path, monkeypatch, quiesce_audio_polls=1)
    journey.voice.meter._clock = lambda: 1000.0  # time never advances: the drain never ends
    monkeypatch.setattr(client, "_BARGE_INTERRUPT_TIMEOUT", 0.05)
    receipt = refused(journey, "journey_barge_quiescence_timeout")
    assert receipt["stopInjected"] is True and receipt["stopRecognized"] is True
    assert receipt["stopFinalsCount"] == 1 and receipt["interruptEventObserved"] is True
    assert receipt["interruptTargetCount"] == 1 and receipt["targetInterrupted"] is True
    assert receipt["targetCompleted"] is False and receipt["postStopAudioMs"] == 30


class HangOnStopSource(FakeAudioSource):
    """Streams the first clip and the Stop's leading silence, then hangs on Stop speech."""

    speech_frames = 0
    limit = 0

    async def capture_frame(self, frame):
        if any(frame.data):
            HangOnStopSource.speech_frames += 1
            if HangOnStopSource.speech_frames > HangOnStopSource.limit:
                await asyncio.Event().wait()
        self.frames.append(frame)
        await asyncio.sleep(0)


def test_a_stop_stream_timeout_keeps_the_pre_injection_evidence(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    HangOnStopSource.speech_frames = 0
    HangOnStopSource.limit = sum(any(pcm) for pcm in client.load_barge_fixtures()[0])
    journey = barge_journey(tmp_path, monkeypatch)
    journey.voice.rtc.AudioSource = HangOnStopSource
    monkeypatch.setattr(client, "_SPEAK_STREAM_MARGIN", 0.2)
    receipt = refused(journey, "journey_speak_stream_timeout")
    assert receipt["stopInjected"] is True and receipt["targetAudioMsBeforeStop"] >= 50
    # What the host delivered during the Stop stream is kept, as bools, never null.
    assert receipt["targetInterrupted"] is True and receipt["targetCompleted"] is False


def test_a_failing_evidence_snapshot_neither_changes_the_verdict_nor_blocks_cleanup(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    original = client.BargeTrace.__init__

    def broken(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.counts = None  # the snapshot's receipt.update(**trace.counts) raises

    monkeypatch.setattr(client.BargeTrace, "__init__", broken)
    journey = barge_journey(tmp_path, monkeypatch, pre=barge_pre(complete=True))

    async def run():
        with pytest.raises(ValueError, match="journey_barge_target_terminal"):
            await asyncio.wait_for(journey.step(BARGE_STEP), 10)
        return (journey.voice._idle_task, journey.tracker.speak, journey.voice.meter.window)

    assert asyncio.run(run()) == (None, None, "closed")


def test_a_valid_barge_passes_even_when_the_evidence_snapshot_raises(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    class SnapshotOnlyFailure(dict):
        def get(self, *args, **kwargs):  # used only by the best-effort snapshot
            raise RuntimeError("snapshot failed")

    original = client.BargeTrace.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.pairs = SnapshotOnlyFailure()

    monkeypatch.setattr(client.BargeTrace, "__init__", init)
    journey = barge_journey(tmp_path, monkeypatch)

    async def run():
        await asyncio.wait_for(journey.step(BARGE_STEP), 10)  # still passes
        return (journey.voice._idle_task, journey.tracker.speak, journey.voice.meter.window)

    assert asyncio.run(run()) == (None, None, "closed")  # cleanup ran
    receipt = journey.speak_receipt
    assert receipt["snapshotError"] is True  # the diagnostic metadata is flagged unavailable
    assert receipt["quiescenceObserved"] is True and journey.media == [{"mediaIncarnation": 1}]
    assert receipt["interruptEventObserved"] is False  # unavailable, not the verdict's source


def test_the_snapshot_flag_is_false_when_the_snapshot_succeeds(tmp_path, monkeypatch):
    journey = barge_journey(tmp_path, monkeypatch)
    barge(journey)
    assert journey.speak_receipt["snapshotError"] is False
    assert journey.speak_receipt["interruptEventObserved"] is True


# Spoken delegation and natural spoken cancellation `speak_task` step (#159, cross-ref #237):
# controlled synthetic path only; it bypasses the physical microphone and AEC.

TASK_STEP = {"label": "delegate", "kind": "speak_task"}
T_TARGET, T_OTHER, T_OLD = "task_target", "task_other", "task_old"
RUN_TARGET, RUN_OTHER, RUN_OLD = "run_" + "t" * 32, "run_" + "o" * 32, "run_" + "l" * 32
DISPATCH_TEXT = "Research European public transit funding and compare the major cities."
CANCEL_SPOKEN = "Actually, stop that research, I don't need it anymore."
TASK_CLOCK_STEP = 0.25


def task_state_row(task, status):
    return ("task_state", {"taskId": task, "status": status})


def task_result_row(task, status):
    return ("task_result", {"taskId": task, "status": status, "text": "Stopped."})


def admit(run=RUN_TARGET):
    return ("__admit__", {"run": run})  # the run journal gains an admission, not an event


def task_pre(text=DISPATCH_TEXT, *, vad=True, finals=1, task=True, admission=True, extra=()):
    voice = [("voice_activity_started", {}), ("voice_activity_ended", {})] if vad else []
    dispatched = []
    if task:
        dispatched.append(task_state_row(T_TARGET, "active"))
    if admission:
        dispatched.append(admit())
    return [[], [("voice_input_ready", READY)], [*voice, *[user_final(text)] * finals],
            dispatched, *extra]


def task_post(text=CANCEL_SPOKEN, *, vad=True, finals=1, cancelling=True, terminal=True,
              before_final=()):
    first = [("voice_activity_started", {})] if vad else []
    second = [*before_final, ("voice_activity_ended", {}), *[user_final(text)] * finals]
    third = [task_state_row(T_TARGET, "cancelling")] if cancelling else []
    fourth = []
    if terminal:
        fourth = [task_state_row(T_TARGET, "interrupted") if cancelling
                  else task_result_row(T_TARGET, "interrupted"),
                  task_result_row(T_TARGET, "interrupted")]
    return [first, second, third, fourth]


def task_journey(tmp_path, monkeypatch, *, pre=None, post=None, before=(), settle=0.0,
                 witness="cancelled", journal=()):
    """A Journey with a fake mic transport and a state-driven fake host.

    Events before the cancellation's injection cursor come from ``pre``; afterwards from
    ``post``; one batch per poll. ``__admit__`` rows grow the fake run journal. The client clock
    is fake and advances TASK_CLOCK_STEP per poll.
    """
    import scripts.native_typed_journey as client

    for name, value in {
        "_SPEAK_FRAME_SECONDS": 0.0, "_SPEAK_DRAIN_SECONDS": 0.0, "_SPEAK_TURN_TIMEOUT": 1.0,
        "_SPEAK_SETTLE_TIMEOUT": 0.5, "_SPEAK_READY_TIMEOUT": 1.0,
        "_SPEAK_SETTLE_SECONDS": settle, "_TASK_DISPATCH_TIMEOUT": 0.6,
        "_TASK_QUIET_TIMEOUT": 0.6, "_TASK_CANCEL_TIMEOUT": 0.6,
    }.items():
        monkeypatch.setattr(client, name, value)
    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              api_origin="http://127.0.0.1:1")
    journey = Journey(options, {"expiresInSeconds": 3600}, "secret")
    run_journal = {"ids": set(journal), "pending": 0}
    monkeypatch.setattr(client, "read_journal",
                        lambda _: (set(run_journal["ids"]), run_journal["pending"]))
    journey.witness_reads = []

    def witness_factory(*args, **kwargs):
        journey.witness_reads.append(args[1])
        return SimpleNamespace(read=lambda: {"terminal": witness, "toolCalls": []})

    monkeypatch.setattr(client, "NativeRunWitness", witness_factory)
    rtc, room = fake_rtc(), FakeRoom()
    journey.voice = client.SyntheticVoice(rtc, room, SPEAK_WORKER)
    meter = journey.voice.meter
    now, cursor = [1000.0], [len(before) + 1]
    if before:
        journey.tracker.consume(numbered(list(before), 1))
    meter._clock = lambda: now[0]
    queues = {"pre": list(task_pre() if pre is None else pre),
              "post": list(task_post() if post is None else post)}
    journey.media, journey.rtc, journey.room, journey.journal = [], rtc, room, run_journal
    journey.queues = queues

    async def request(route, payload=None):
        if route == "media":
            journey.media.append(payload)
            return {"mediaIncarnation": payload["mediaIncarnation"], "version": 1}
        if route == "input":
            journey.media.append(("input", payload))
            return {}
        assert route == "events"
        now[0] += TASK_CLOCK_STEP
        trace = journey.tracker.speak
        if trace is not None and getattr(trace, "cancel_cursor", None) is not None:
            journey.journal["post"] = True  # the cancellation was injected: stay post afterwards
        phase = "post" if journey.journal.get("post") else "pre"
        rows = []
        for kind, data in queues[phase].pop(0) if queues[phase] else []:
            if kind == "__admit__":
                run_journal["ids"].add(data["run"])
            else:
                rows.append((kind, data))
        events = numbered(rows, cursor[0])
        cursor[0] += len(events)
        return {"version": 2, "events": events}

    journey.request = request
    return journey


def run_task(journey, seconds=10):
    asyncio.run(asyncio.wait_for(journey.step(TASK_STEP), seconds))


def task_refused(journey, category):
    with pytest.raises(ValueError, match=category):
        run_task(journey)
    return journey.speak_receipt


def test_speak_task_delegates_by_voice_then_cancels_exactly_that_task(tmp_path, monkeypatch):
    journey = task_journey(tmp_path, monkeypatch)
    run_task(journey)
    receipt = journey.speak_receipt
    assert receipt["dispatchTasks"] == 1 and receipt["dispatchAdmissions"] == 1
    assert receipt["taskBound"] and receipt["targetActiveBeforeCancel"]
    assert receipt["cancelInjected"] and receipt["cancelObserved"]
    assert receipt["cancelFinalsCount"] == 1 and receipt["cancelTranscriptMatches"]
    assert receipt["nativeCancelledBeforeCleanup"] is True
    assert receipt["foreignCancelCount"] == 0 and receipt["competingAdmissionCount"] == 0
    assert journey.bindings == {"delegate": (T_TARGET, RUN_TARGET)}
    assert journey.results["delegate"]["terminal"] == "cancelled"
    assert journey.witness_reads == [RUN_TARGET]  # read once, in the step, before any cleanup


def test_speak_task_uses_one_track_one_media_request_and_never_a_typed_admission(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    journey = task_journey(tmp_path, monkeypatch)
    run_task(journey)
    dispatch, _, cancel, _, _ = client.load_task_fixtures()
    assert journey.media == [{"mediaIncarnation": 1}]  # no typed "input" request either
    assert len(journey.room.published) == 1
    pcm = [frame.data for frame in journey.room.published[0][0].source.frames]
    assert pcm[-(len(dispatch) + len(cancel)):] == list(dispatch) + list(cancel)
    assert journey.tracker.turns == []


def test_speak_task_receipt_is_metadata_only(tmp_path, monkeypatch):
    journey = task_journey(tmp_path, monkeypatch)
    run_task(journey)
    raw = json.dumps(journey.speak_receipt).lower()
    for private in (T_TARGET, RUN_TARGET, "research", "european", "actually"):
        assert private not in raw
    assert all(type(value) in {bool, int, str} for value in journey.speak_receipt.values())


@pytest.mark.parametrize("text,category", [
    ("Research European public transit funding", "journey_speak_transcript"),
    ("", "journey_speak_transcript"),
])
def test_speak_task_refuses_wrong_dispatch_words(tmp_path, monkeypatch, text, category):
    journey = task_journey(tmp_path, monkeypatch, pre=task_pre(text))
    receipt = task_refused(journey, category)
    assert receipt["dispatchTranscriptMatches"] is False and receipt["cancelInjected"] is False


@pytest.mark.parametrize("finals,category", [(0, "journey_speak_finals"),
                                             (2, "journey_speak_finals")])
def test_speak_task_refuses_zero_or_two_dispatch_finals(tmp_path, monkeypatch, finals, category):
    journey = task_journey(tmp_path, monkeypatch, pre=task_pre(finals=finals))
    task_refused(journey, category)


def test_speak_task_refuses_a_dispatch_with_no_voice_activity(tmp_path, monkeypatch):
    journey = task_journey(tmp_path, monkeypatch, pre=task_pre(vad=False))
    task_refused(journey, "journey_speak_voice_activity")


def test_speak_task_refuses_when_no_task_or_no_admission_appears(tmp_path, monkeypatch):
    for pre in (task_pre(task=False), task_pre(admission=False)):
        journey = task_journey(tmp_path, monkeypatch, pre=pre)
        task_refused(journey, "journey_task_not_dispatched")
        monkeypatch.undo()


def test_speak_task_refuses_a_competing_admission_during_dispatch(tmp_path, monkeypatch):
    pre = task_pre()
    pre[3] = [*pre[3], task_state_row(T_OTHER, "active"), admit(RUN_OTHER)]
    receipt = task_refused(task_journey(tmp_path, monkeypatch, pre=pre), "journey_task_competing")
    assert receipt["dispatchTasks"] == 2 and receipt["dispatchAdmissions"] == 2
    assert receipt["targetActiveBeforeCancel"] is False  # refused before any quiet wait


def test_speak_task_refuses_a_replayed_prior_task_during_dispatch(tmp_path, monkeypatch):
    pre = task_pre()
    pre[3] = [task_state_row(T_OLD, "active"), *pre[3]]  # a replay of an earlier task
    journey = task_journey(tmp_path, monkeypatch, pre=pre,
                           before=[task_state_row(T_OLD, "completed")],
                           journal=[RUN_OLD])
    task_refused(journey, "journey_task_replayed")


def test_speak_task_refuses_a_task_whose_first_state_precedes_the_dispatch_final(
    tmp_path, monkeypatch
):
    pre = task_pre()
    pre[2] = [("voice_activity_started", {}), task_state_row(T_TARGET, "active"),
              ("voice_activity_ended", {}), user_final(DISPATCH_TEXT)]
    pre[3] = [admit()]
    receipt = task_refused(task_journey(tmp_path, monkeypatch, pre=pre),
                           "journey_task_dispatch_unbound")
    assert receipt["taskBound"] is False


def test_speak_task_never_selects_a_replacement_when_target_ends_before_the_cancel(
    tmp_path, monkeypatch
):
    pre = task_pre()
    pre[3] = [*pre[3], task_result_row(T_TARGET, "completed")]  # ends with the dispatch
    journey = task_journey(tmp_path, monkeypatch, pre=pre)
    receipt = task_refused(journey, "journey_task_target_terminal")
    assert receipt["cancelInjected"] is False and receipt["targetActiveBeforeCancel"] is False


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_a_target_that_ends_during_the_cancel_lead_in_is_inconclusive_and_nothing_is_sent(
    tmp_path, monkeypatch, status
):
    import scripts.native_typed_journey as client

    # Delivered at the first poll of the cancel stream, i.e. during its leading silence,
    # after the quiet session was established. A failed run covers a rate-limit or lookup refusal.
    pre = task_pre(extra=([task_result_row(T_TARGET, status)],))
    journey = task_journey(tmp_path, monkeypatch, pre=pre)
    receipt = task_refused(journey, "journey_task_target_terminal")
    assert receipt["targetActiveBeforeCancel"] is True and receipt["cancelInjected"] is False
    sent = [frame.data for frame in journey.room.published[0][0].source.frames]
    dispatch = client.load_task_fixtures()[0]
    assert sum(any(pcm) for pcm in sent) == sum(any(pcm) for pcm in dispatch)  # no cancel speech


def test_a_competing_admission_at_the_first_cancel_speech_frame_aborts_before_any_speech(
    tmp_path, monkeypatch
):
    import scripts.native_typed_journey as client

    pre = task_pre(extra=([admit(RUN_OTHER)],))
    journey = task_journey(tmp_path, monkeypatch, pre=pre)
    receipt = task_refused(journey, "journey_task_competing")
    assert receipt["cancelInjected"] is False
    sent = [frame.data for frame in journey.room.published[0][0].source.frames]
    dispatch = client.load_task_fixtures()[0]
    assert sum(any(pcm) for pcm in sent) == sum(any(pcm) for pcm in dispatch)


def test_a_competing_active_task_at_the_first_cancel_speech_frame_aborts(tmp_path, monkeypatch):
    pre = task_pre(extra=([task_state_row(T_OTHER, "active")],))
    task_refused(task_journey(tmp_path, monkeypatch, pre=pre), "journey_task_competing")


def test_a_pending_admission_blocks_the_quiet_session_and_the_cancel(tmp_path, monkeypatch):
    journey = task_journey(tmp_path, monkeypatch)
    journey.journal["pending"] = 1  # a dispatch that is not yet an accepted admission
    task_refused(journey, "journey_task_not_quiet")


def test_the_cancel_needs_a_whole_quiet_interval_after_the_dispatch(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    journey = task_journey(tmp_path, monkeypatch)
    monkeypatch.setattr(client, "_SPEAK_SETTLE_SECONDS", 0.5)
    monkeypatch.setattr(client, "_TASK_QUIET_TIMEOUT", 3.0)
    monkeypatch.setattr(client, "_SPEAK_SETTLE_TIMEOUT", 3.0)
    run_task(journey)
    assert journey.speak_receipt["preCancelQuietMs"] >= 500


def test_the_cancel_waits_for_an_open_foreground_turn_to_close(tmp_path, monkeypatch):
    ack = {"turnId": "ack_turn", "turnGeneration": 1}
    pre = task_pre(extra=([], [("assistant_turn_completed", ack)]))
    pre[3] = [*pre[3], ("first_foreground_token", ack)]  # the acknowledgment turn is open
    journey = task_journey(tmp_path, monkeypatch, pre=pre)
    run_task(journey)
    assert journey.speak_receipt["preCancelQuietMs"] >= 500  # two polls with the turn open


def test_an_open_foreground_turn_that_never_closes_is_not_quiet(tmp_path, monkeypatch):
    ack = {"turnId": "ack_turn", "turnGeneration": 1}
    pre = task_pre()
    pre[3] = [*pre[3], ("first_foreground_token", ack)]  # opened and never closed
    task_refused(task_journey(tmp_path, monkeypatch, pre=pre), "journey_task_not_quiet")


@pytest.mark.parametrize("text,category", [
    ("Never mind", "journey_task_cancel_transcript"),
    ("", "journey_task_cancel_transcript"),
])
def test_speak_task_refuses_wrong_cancel_words(tmp_path, monkeypatch, text, category):
    journey = task_journey(tmp_path, monkeypatch, post=task_post(text))
    receipt = task_refused(journey, category)
    assert receipt["cancelTranscriptMatches"] is False and receipt["cancelInjected"] is True
    assert journey.results == {}  # no witness read, so no acceptance evidence


@pytest.mark.parametrize("finals", [0, 2])
def test_speak_task_refuses_zero_or_two_cancel_finals(tmp_path, monkeypatch, finals):
    journey = task_journey(tmp_path, monkeypatch, post=task_post(finals=finals))
    task_refused(journey, "journey_task_cancel_finals")


def test_speak_task_needs_a_fresh_voice_start_for_the_cancel(tmp_path, monkeypatch):
    journey = task_journey(tmp_path, monkeypatch, post=task_post(vad=False))
    receipt = task_refused(journey, "journey_task_cancel_vad")
    assert receipt["cancelVadObserved"] is False


def test_a_foreign_cancellation_fails_the_step(tmp_path, monkeypatch):
    post = task_post()
    post[2] = [*post[2], task_state_row(T_OTHER, "cancelling")]
    journey = task_journey(tmp_path, monkeypatch, post=post)
    receipt = task_refused(journey, "journey_task_foreign_cancel")
    assert receipt["foreignCancelCount"] == 1
    assert journey.witness_reads == []  # refused before the native witness was read


def test_a_task_started_after_the_cancel_fails_the_step(tmp_path, monkeypatch):
    post = task_post()
    post[2] = [*post[2], task_state_row(T_OTHER, "active")]
    task_refused(task_journey(tmp_path, monkeypatch, post=post), "journey_task_competing")


def test_a_second_admission_after_the_cancel_fails_the_step(tmp_path, monkeypatch):
    post = task_post()
    post[2] = [*post[2], admit(RUN_OTHER)]
    receipt = task_refused(task_journey(tmp_path, monkeypatch, post=post),
                           "journey_task_competing")
    assert receipt["competingAdmissionCount"] == 1


def test_a_cancelling_state_before_the_cancel_final_is_stale_not_evidence(tmp_path, monkeypatch):
    post = task_post(before_final=[task_state_row(T_TARGET, "cancelling")], cancelling=False)
    task_refused(task_journey(tmp_path, monkeypatch, post=post), "journey_task_cancel_stale")


def test_a_cancel_visible_only_as_a_terminal_result_is_not_accepted(tmp_path, monkeypatch):
    # No cancelling or interrupted task_state for TARGET after the final: whatever settles it
    # later (cleanup included) cannot be acceptance evidence.
    post = task_post(cancelling=False)
    journey = task_journey(tmp_path, monkeypatch, post=post)
    receipt = task_refused(journey, "journey_task_cancel_unobserved")
    assert receipt["cancelObserved"] is False and receipt["nativeCancelledBeforeCleanup"] is False
    assert journey.results == {} and journey.witness_reads == []


def test_a_cancel_that_never_arrives_is_not_accepted(tmp_path, monkeypatch):
    post = task_post(cancelling=False, terminal=False)
    journey = task_journey(tmp_path, monkeypatch, post=post)
    task_refused(journey, "journey_task_cancel_unobserved")
    assert journey.witness_reads == []


def test_the_native_witness_must_agree_that_the_task_was_cancelled(tmp_path, monkeypatch):
    journey = task_journey(tmp_path, monkeypatch, witness="completed")
    task_refused(journey, "journey_terminal_binding")
    assert journey.speak_receipt["nativeCancelledBeforeCleanup"] is False


def test_speak_task_owns_the_dispatched_task_even_when_the_step_fails(tmp_path, monkeypatch):
    journey = task_journey(tmp_path, monkeypatch, post=task_post(vad=False))
    task_refused(journey, "journey_task_cancel_vad")
    assert journey.bindings == {"delegate": (T_TARGET, RUN_TARGET)}  # cleanup can cancel it


def test_scenario_accepts_one_speak_task_step_and_no_second_microphone_step(tmp_path):
    def scenario(*steps):
        path = tmp_path / "task.json"
        path.write_text(json.dumps({"version": 1, "steps": list(steps)}), encoding="utf-8")
        return load_scenario(path)

    assert scenario({"label": "delegate", "kind": "speak_task"})[0]["kind"] == "speak_task"
    for bad in (
        [{"label": "a", "kind": "speak"}, {"label": "b", "kind": "speak_task"}],
        [{"label": "a", "kind": "speak_task"}, {"label": "b", "kind": "speak_task"}],
        [{"label": "a", "kind": "speak_task", "text": "Research"}],
        [{"label": "a", "kind": "speak_task", "task": "x"}],
    ):
        with pytest.raises(ValueError, match="journey_(speak|step)"):
            scenario(*bad)


@pytest.mark.parametrize("name", ["_TASK_DISPATCH_SHA256", "_TASK_CANCEL_SHA256"])
def test_the_task_fixtures_are_pinned_by_sha256(monkeypatch, name):
    import scripts.native_typed_journey as client

    frames = client.load_task_fixtures()
    assert frames[1] == ("research", "european", "public", "transit", "funding", "and",
                         "compare", "the", "major", "cities")
    assert frames[4][:4] == ("actually", "stop", "that", "research")
    monkeypatch.setattr(client, name, "0" * 64)
    with pytest.raises(ValueError, match="journey_speak_fixture"):
        client.load_task_fixtures()


def test_a_task_dispatched_after_the_witness_read_still_fails_the_step(tmp_path, monkeypatch):
    post = [*task_post(), [task_result_row(T_OTHER, "completed")]]  # first seen as a result
    journey = task_journey(tmp_path, monkeypatch, post=post)
    receipt = task_refused(journey, "journey_followup_dispatched")
    assert receipt["nativeCancelledBeforeCleanup"] is True  # evidence was read, then refused


def test_a_previously_seen_task_reactivated_at_the_first_cancel_speech_frame_aborts(
    tmp_path, monkeypatch
):
    pre = task_pre(extra=([task_state_row(T_OLD, "active")],))
    journey = task_journey(tmp_path, monkeypatch, pre=pre,
                           before=[task_state_row(T_OLD, "completed")], journal=[RUN_OLD])
    receipt = task_refused(journey, "journey_task_competing")
    assert receipt["cancelInjected"] is False


class FailAfterSpeechSource(FakeAudioSource):
    """Captures the warm-up and the clip's leading silence, then fails on the first speech."""

    async def capture_frame(self, frame):
        await asyncio.sleep(0.01)  # real time passes, so the step polls between frames
        if any(frame.data):
            raise RuntimeError("capture failed")
        self.frames.append(frame)


def cleanup_cancels(journey):
    """Run settle_owned() with a host that settles a task once it is asked to cancel it."""
    requested = []
    inner = journey.request

    async def request(route, payload=None):
        if route == "input":
            requested.append(payload["text"])
            journey.queues["pre"].insert(0, [task_state_row(T_TARGET, "interrupted"),
                                             task_result_row(T_TARGET, "interrupted")])
        return await inner(route, payload)

    journey.request = request
    return requested


def test_an_accepted_task_is_owned_and_cancelled_when_the_dispatch_stream_later_fails(
    tmp_path, monkeypatch
):
    journey = task_journey(tmp_path, monkeypatch)
    journey.voice.rtc.AudioSource = FailAfterSpeechSource
    requested = cleanup_cancels(journey)

    async def run():
        with pytest.raises(RuntimeError, match="capture failed"):
            await asyncio.wait_for(journey.step(TASK_STEP), 10)
        owned = dict(journey.bindings)
        settled = await journey.settle_owned()
        return owned, settled

    owned, settled = asyncio.run(run())
    assert owned == {"delegate": (T_TARGET, RUN_TARGET)}
    assert requested == ["cancel task: " + T_TARGET]  # exactly that task, by exact id
    assert settled is False  # the journal still holds the admission: honest, not hidden


def test_ambiguous_dispatch_peers_are_never_guessed_into_ownership(tmp_path, monkeypatch):
    pre = task_pre()
    pre[3] = [*pre[3], task_state_row(T_OTHER, "active"), admit(RUN_OTHER)]
    journey = task_journey(tmp_path, monkeypatch, pre=pre)
    requested = cleanup_cancels(journey)

    async def run():
        with pytest.raises(ValueError, match="journey_task_competing"):
            await asyncio.wait_for(journey.step(TASK_STEP), 10)
        return dict(journey.bindings), await journey.settle_owned()

    owned, settled = asyncio.run(run())
    assert owned == {} and requested == [] and settled is False


def test_a_second_user_final_during_the_cancel_lead_in_fails_the_step(tmp_path, monkeypatch):
    pre = task_pre(extra=([user_final("unexpected extra input")],))
    journey = task_journey(tmp_path, monkeypatch, pre=pre)
    receipt = task_refused(journey, "journey_speak_finals")
    assert receipt["dispatchFinalsCount"] == 2 and receipt["cancelInjected"] is False


def test_a_second_user_final_during_the_quiet_wait_fails_the_step(tmp_path, monkeypatch):
    ack = {"turnId": "ack_turn", "turnGeneration": 1}
    pre = task_pre(extra=([user_final("unexpected extra input")],
                          [("assistant_turn_completed", ack)]))
    pre[3] = [*pre[3], ("first_foreground_token", ack)]  # an open turn keeps the wait polling
    journey = task_journey(tmp_path, monkeypatch, pre=pre)
    receipt = task_refused(journey, "journey_speak_finals")
    assert receipt["targetActiveBeforeCancel"] is False


@pytest.mark.parametrize("rows", [
    [("voice_activity_started", {})],
    [("assistant_turn_completed", {})],  # an unbound foreground terminal
    [("assistant_turn_completed", {"turnId": "old_turn", "turnGeneration": 1})],  # replayed
], ids=["voice-start", "unbound-foreground", "replayed-foreground"])
def test_unexpected_input_or_foreground_after_the_dispatch_fails_the_step(
    tmp_path, monkeypatch, rows
):
    old = ("assistant_turn_completed", {"turnId": "old_turn", "turnGeneration": 1})
    pre = task_pre(extra=(rows,))
    journey = task_journey(tmp_path, monkeypatch, pre=pre, before=[old])
    receipt = task_refused(journey, "journey_task_unexpected_input")
    assert receipt["unexpectedInputCount"] == 1 and receipt["cancelInjected"] is False


def test_the_expected_acknowledgment_turn_is_not_unexpected_input(tmp_path, monkeypatch):
    ack = {"turnId": "ack_turn", "turnGeneration": 1}
    pre = task_pre(extra=([], [("assistant_turn_completed", ack)]))
    pre[3] = [*pre[3], ("first_foreground_token", ack), ("assistant_text_generated", ack)]
    run_task(task_journey(tmp_path, monkeypatch, pre=pre))


def test_a_foreign_cancel_in_the_final_poll_fails_after_the_witness_read(tmp_path, monkeypatch):
    # A prior completed task goes cancelling only in the poll after the witness read.
    post = [*task_post(), [task_state_row(T_OLD, "cancelling")]]
    journey = task_journey(tmp_path, monkeypatch, post=post,
                           before=[task_state_row(T_OLD, "completed")], journal=[RUN_OLD])
    receipt = task_refused(journey, "journey_task_foreign_cancel")
    assert receipt["foreignCancelCount"] == 1
    # The pre-cleanup evidence is preserved in the receipt even though the verdict is refused.
    assert receipt["nativeCancelledBeforeCleanup"] is True and journey.witness_reads == [RUN_TARGET]


def test_a_foreign_cancel_during_witness_collection_fails_the_step(tmp_path, monkeypatch):
    # Delivered while collect() polls for TARGET's terminal state.
    post = task_post(terminal=False)  # [vad], [ended, final], [cancelling TARGET], []
    post.append([task_state_row(T_OTHER, "cancelling")])  # delivered by collect()'s polling
    post.append([task_state_row(T_TARGET, "interrupted"), task_result_row(T_TARGET, "interrupted")])
    journey = task_journey(tmp_path, monkeypatch, post=post)
    receipt = task_refused(journey, "journey_task_foreign_cancel")
    assert receipt["foreignCancelCount"] == 1 and journey.witness_reads == [RUN_TARGET]


def test_an_admission_in_the_final_poll_fails_after_the_witness_read(tmp_path, monkeypatch):
    post = [*task_post(), [admit(RUN_OTHER)]]
    journey = task_journey(tmp_path, monkeypatch, post=post)
    receipt = task_refused(journey, "journey_task_competing")
    assert receipt["competingAdmissionCount"] == 1
    assert receipt["nativeCancelledBeforeCleanup"] is True


def test_a_second_cancel_final_in_the_final_poll_fails_after_the_witness_read(
    tmp_path, monkeypatch
):
    post = [*task_post(), [user_final(CANCEL_SPOKEN)]]
    journey = task_journey(tmp_path, monkeypatch, post=post)
    receipt = task_refused(journey, "journey_task_cancel_finals")
    assert receipt["cancelFinalsCount"] == 2  # the refusal keeps the updated count
    assert receipt["nativeCancelledBeforeCleanup"] is True and journey.witness_reads == [RUN_TARGET]


def test_a_second_cancel_final_during_witness_collection_fails_the_step(tmp_path, monkeypatch):
    post = task_post(terminal=False)  # [vad], [ended, final], [cancelling TARGET], []
    post.append([user_final("Please stop it")])  # delivered by collect()'s polling
    post.append([task_state_row(T_TARGET, "interrupted"), task_result_row(T_TARGET, "interrupted")])
    journey = task_journey(tmp_path, monkeypatch, post=post)
    receipt = task_refused(journey, "journey_task_cancel_finals")
    assert receipt["cancelFinalsCount"] == 2 and receipt["cancelTranscriptMatches"] is True


# Confirmed natural spoken cancel with two tasks `speak_confirm_cancel` step (#237, #581):
# controlled synthetic path only; it bypasses the physical microphone and AEC.

CONFIRM_STEP = {"label": "confirm", "kind": "speak_confirm_cancel"}
C_A, C_B, C_OTHER = "task_alpha", "task_bravo", "task_other"
RUN_A, RUN_B, RUN_X = "run_" + "a" * 32, "run_" + "b" * 32, "run_" + "c" * 32
REQ_B_SPOKEN = "Please stop the ocean temperature research task."
REQ_A_SPOKEN = "Please stop the rail ticket prices task."
P_B, P_NO, P_A, P_YES = ({"turnId": "confirm_" + name, "turnGeneration": 1}
                         for name in ("b", "no", "a", "yes"))


def confirm_objectives():
    import scripts.native_typed_journey as client

    return client._CONFIRM_OBJECTIVE_A, client._CONFIRM_OBJECTIVE_B


def readback_of(objective):
    from hermes_realtime.conversation.task_proposals import cancel_readback

    return cancel_readback(objective)


def assistant_final(text, pair):
    return ("transcript_final", {"role": "assistant", "text": text, **pair})


def confirm_reply(spoken, pair, *, said=None, complete=True, interrupted=False, extra=()):
    """One spoken clip's phase: voice start, its final, a reply turn, and that turn's end."""
    reply = [("first_foreground_token", pair)]
    if said is not None:
        reply.append(assistant_final(said, pair))
    end = []
    if interrupted:
        end = [("assistant_turn_interrupted", pair)]
    elif complete:
        end = [("assistant_turn_completed", pair)]
    return [[("voice_activity_started", {})],
            [("voice_activity_ended", {}), user_final(spoken)],
            [*reply, *extra], end]


def confirm_yes(*, cancel=None, before_final=(), extra=()):
    if cancel is None:
        cancel = (task_state_row(C_A, "cancelling"),)
    return [[("voice_activity_started", {})],
            [*before_final, ("voice_activity_ended", {}), user_final("Yes.")],
            [*cancel, *extra],
            [task_state_row(C_A, "interrupted"), task_result_row(C_A, "interrupted")]]


def confirm_phases(**overrides):
    obj_a, obj_b = confirm_objectives()
    phases = {
        "b": confirm_reply(REQ_B_SPOKEN, P_B, said=readback_of(obj_b)),
        "no": confirm_reply("No.", P_NO, said="Okay."),
        "a": confirm_reply(REQ_A_SPOKEN, P_A, said=readback_of(obj_a)),
        "yes": confirm_yes(),
    }
    phases.update(overrides)
    return [phases["b"], phases["no"], phases["a"], phases["yes"]]


def confirm_dispatches(a=(), b=()):
    obj_a, obj_b = confirm_objectives()
    return {
        "task: " + obj_a: [task_state_row(C_A, "active"), admit(RUN_A), *a],
        "task: " + obj_b: [task_state_row(C_B, "active"), admit(RUN_B), *b],
    }


def confirm_journey(tmp_path, monkeypatch, *, phases=None, pre=(), dispatches=None,
                    witness=None, live=None, before=(), journal=()):
    """A Journey with a fake mic transport and a state-driven fake host.

    Typed inputs answer with their dispatch (or a cleanup cancellation) at the next poll.
    Otherwise each poll returns one batch of the current clip phase (the number of clips
    whose first speech frame has been sent), or of ``pre`` before any clip.
    """
    import scripts.native_typed_journey as client

    for name, value in {
        "_SPEAK_FRAME_SECONDS": 0.0, "_SPEAK_DRAIN_SECONDS": 0.0, "_SPEAK_TURN_TIMEOUT": 1.0,
        "_SPEAK_SETTLE_TIMEOUT": 0.5, "_SPEAK_READY_TIMEOUT": 1.0,
        "_SPEAK_SETTLE_SECONDS": 0.0, "_TASK_DISPATCH_TIMEOUT": 0.6,
        "_TASK_QUIET_TIMEOUT": 0.6, "_TASK_CANCEL_TIMEOUT": 0.6,
        "_CONFIRM_REPLY_TIMEOUT": 0.6, "_CONFIRM_NO_DRAIN_SECONDS": 0.3,
    }.items():
        monkeypatch.setattr(client, name, value)
    options = SimpleNamespace(preview_dir=tmp_path, native_profile=tmp_path,
                              api_origin="http://127.0.0.1:1")
    journey = Journey(options, {"expiresInSeconds": 3600}, "secret")
    run_journal = {"ids": set(journal), "pending": 0}
    monkeypatch.setattr(client, "read_journal",
                        lambda _: (set(run_journal["ids"]), run_journal["pending"]))
    journey.witness_reads = []

    def witness_factory(bearer, run, **kwargs):
        def read():
            journey.witness_reads.append(("read", run))
            return {"terminal": (witness or {}).get(run, "cancelled"), "toolCalls": []}

        def live_status():
            journey.witness_reads.append(("live", run))
            return (live or {}).get(run, "running")

        return SimpleNamespace(read=read, live_status=live_status)

    monkeypatch.setattr(client, "NativeRunWitness", witness_factory)
    rtc, room = fake_rtc(), FakeRoom()
    journey.voice = client.SyntheticVoice(rtc, room, SPEAK_WORKER)
    meter = journey.voice.meter
    now, cursor = [1000.0], [len(before) + 1]
    if before:
        journey.tracker.consume(numbered(list(before), 1))
    meter._clock = lambda: now[0]
    queues = {"pre": [[], [("voice_input_ready", READY)], *pre]}
    for index, batches in enumerate(confirm_phases() if phases is None else phases):
        queues[index] = list(batches)
    answers = confirm_dispatches() if dispatches is None else dispatches
    immediate, phase = [], ["pre"]
    journey.inputs, journey.media, journey.rtc, journey.room = [], [], rtc, room
    journey.journal, journey.queues = run_journal, queues

    async def request(route, payload=None):
        if route == "media":
            journey.media.append(payload)
            return {"mediaIncarnation": payload["mediaIncarnation"], "version": 1}
        if route == "input":
            text = payload["text"]
            journey.inputs.append(text)
            if text.startswith("cancel task: "):
                task = text.removeprefix("cancel task: ")
                immediate.append([task_state_row(task, "interrupted"),
                                  task_result_row(task, "interrupted")])
            elif text in answers:
                immediate.append(list(answers[text]))
            return {}
        assert route == "events"
        now[0] += TASK_CLOCK_STEP
        trace = journey.tracker.speak
        marks = getattr(trace, "marks", None)
        if marks:
            phase[0] = len(marks) - 1
        if immediate:
            batch = immediate.pop(0)
        else:
            batch = queues[phase[0]].pop(0) if queues.get(phase[0]) else []
        rows = []
        for kind, data in batch:
            if kind == "__admit__":
                run_journal["ids"].add(data["run"])
            elif kind == "__pending__":
                run_journal["pending"] = data["count"]
            else:
                rows.append((kind, data))
        events = numbered(rows, cursor[0])
        cursor[0] += len(events)
        return {"version": 2, "events": events}

    journey.request = request
    return journey


def run_confirm(journey, seconds=15):
    asyncio.run(asyncio.wait_for(journey.step(CONFIRM_STEP), seconds))


def confirm_refused(journey, category):
    with pytest.raises(ValueError, match=category):
        run_confirm(journey)
    return journey.speak_receipt


def sent_speech_frames(journey):
    return sum(any(frame.data) for frame in journey.room.published[0][0].source.frames)


def clip_speech_frames(*indexes):
    import scripts.native_typed_journey as client

    clips = client.load_confirm_fixtures()
    return sum(sum(any(pcm) for pcm in clips[index][0]) for index in indexes)


def test_speak_confirm_cancel_cancels_exactly_the_confirmed_task(tmp_path, monkeypatch):
    journey = confirm_journey(tmp_path, monkeypatch)
    run_confirm(journey)
    receipt = journey.speak_receipt
    obj_a, obj_b = confirm_objectives()
    assert journey.inputs == ["task: " + obj_a, "task: " + obj_b, "cancel task: " + C_B]
    assert journey.bindings == {"confirm.a": (C_A, RUN_A), "confirm.b": (C_B, RUN_B)}
    assert receipt["dispatchTasks"] == 2 and receipt["dispatchAdmissions"] == 2
    assert receipt["tasksBound"] and receipt["bothActiveBeforeRequest"]
    for clip in ("requestB", "no", "requestA", "yes"):
        assert receipt[clip + "Injected"] and receipt[clip + "VadObserved"]
        assert receipt[clip + "FinalsCount"] == 1 and receipt[clip + "TranscriptMatches"]
    assert receipt["readbackBOutcome"] == "target" and receipt["readbackAOutcome"] == "target"
    assert receipt["bothActiveAfterNo"] is True
    assert receipt["targetCancelObserved"] is True and receipt["bystanderCancelObserved"] is False
    assert receipt["bystanderActiveAtWitness"] is True
    assert receipt["nativeCancelledBeforeCleanup"] is True
    assert receipt["nativeBystanderLiveBeforeCleanup"] is True
    assert receipt["successPathCleanupCancels"] == 1  # B, after the witness: never acceptance
    # A's witness, then B's live status, both before any cleanup; B's read is cleanup's.
    assert journey.witness_reads == [("read", RUN_A), ("live", RUN_B), ("read", RUN_B)]
    assert journey.results["confirm.a"]["terminal"] == "cancelled"
    assert journey.tracker.speak is None and journey.tracker.turns == []
    assert journey.media == [{"mediaIncarnation": 1}] and len(journey.room.published) == 1
    assert sent_speech_frames(journey) == clip_speech_frames(0, 1, 2, 3)


def test_speak_confirm_cancel_receipt_is_metadata_only(tmp_path, monkeypatch):
    journey = confirm_journey(tmp_path, monkeypatch)
    run_confirm(journey)
    raw = json.dumps(journey.speak_receipt).lower()
    for private in (C_A, C_B, RUN_A, RUN_B, "ocean", "rail", "ticket", "temperature",
                    "background task", "okay"):
        assert private not in raw
    assert all(value is None or type(value) in {bool, int, str}
               for value in journey.speak_receipt.values())


def test_the_confirm_readbacks_are_the_hosts_exact_readbacks():
    import scripts.native_typed_journey as client

    obj_a, obj_b = confirm_objectives()
    assert {"a": readback_of(obj_a), "b": readback_of(obj_b)} == client._CONFIRM_READBACKS
    assert readback_of(obj_a) != readback_of(obj_b)


def test_scenario_accepts_one_speak_confirm_cancel_step_and_no_second_microphone_step(tmp_path):
    def scenario(*steps):
        path = tmp_path / "confirm.json"
        path.write_text(json.dumps({"version": 1, "steps": list(steps)}), encoding="utf-8")
        return load_scenario(path)

    assert scenario(CONFIRM_STEP)[0]["kind"] == "speak_confirm_cancel"
    for bad in (
        [{"label": "a", "kind": "speak_task"}, {"label": "b", "kind": "speak_confirm_cancel"}],
        [{"label": "a", "kind": "speak_confirm_cancel"}, {"label": "b", "kind": "speak"}],
        [{"label": "a", "kind": "speak_confirm_cancel", "text": "Stop"}],
        [{"label": "a", "kind": "speak_confirm_cancel", "task": "x"}],
    ):
        with pytest.raises(ValueError, match="journey_(speak|step)"):
            scenario(*bad)


@pytest.mark.parametrize("name", ["_CONFIRM_REQUEST_B_SHA256", "_CONFIRM_NO_SHA256",
                                  "_CONFIRM_REQUEST_A_SHA256", "_CONFIRM_YES_SHA256"])
def test_the_confirm_fixtures_are_pinned_by_sha256(monkeypatch, name):
    import scripts.native_typed_journey as client

    clips = client.load_confirm_fixtures()
    assert [words for _, words in clips] == [
        ("please", "stop", "the", "ocean", "temperature", "research", "task"),
        ("no",), ("please", "stop", "the", "rail", "ticket", "prices", "task"), ("yes",),
    ]
    monkeypatch.setattr(client, name, "0" * 64)
    with pytest.raises(ValueError, match="journey_speak_fixture"):
        client.load_confirm_fixtures()


# Inconclusive: the precondition did not hold, so nothing is claimed either way.

def test_a_task_that_ends_with_its_dispatch_is_inconclusive(tmp_path, monkeypatch):
    dispatches = confirm_dispatches(b=[task_result_row(C_B, "completed")])
    journey = confirm_journey(tmp_path, monkeypatch, dispatches=dispatches)
    receipt = confirm_refused(journey, "journey_confirm_target_terminal")
    assert receipt["requestBInjected"] is False and receipt["bothActiveBeforeRequest"] is False
    assert journey.bindings == {"confirm.a": (C_A, RUN_A), "confirm.b": (C_B, RUN_B)}


@pytest.mark.parametrize("task", [C_A, C_B])
def test_a_task_that_ends_during_the_request_lead_in_is_inconclusive_and_nothing_is_sent(
    tmp_path, monkeypatch, task
):
    journey = confirm_journey(tmp_path, monkeypatch, pre=[[task_result_row(task, "failed")]])
    receipt = confirm_refused(journey, "journey_confirm_target_terminal")
    assert receipt["bothActiveBeforeRequest"] is True and receipt["requestBInjected"] is False
    assert sent_speech_frames(journey) == 0


def _failed_with_notice(task, text):
    return ("task_result", {"taskId": task, "status": "failed", "text": text})


@pytest.mark.parametrize("reason", ["insufficient", "deadline", "unverified"])
def test_an_early_host_refusal_is_retained_as_an_observed_notice(tmp_path, monkeypatch, reason):
    """Diagnosis only: A's fixed host refusal category survives into the pre-cleanup
    snapshot even though its native terminal is never read. The verdict is unchanged."""
    import scripts.native_typed_journey as client

    notice = {value: key for key, value in client._HOST_REFUSALS.items()}[reason]
    journey = confirm_journey(tmp_path, monkeypatch, pre=[[_failed_with_notice(C_A, notice)]])
    receipt = confirm_refused(journey, "journey_confirm_target_terminal")
    assert receipt["requestBInjected"] is False
    snapshot = journey.pre_cleanup_tasks()
    assert snapshot["confirm.a"] == {"trackerStatus": "failed", "nativeTerminal": None,
                                     "hostRefusalObserved": reason}
    assert snapshot["confirm.b"]["hostRefusalObserved"] is None
    assert notice not in json.dumps(snapshot)


@pytest.mark.parametrize("text", ["Some raw model or Hermes error text.", None])
def test_an_unknown_failure_text_is_never_copied_as_a_notice(tmp_path, monkeypatch, text):
    row = (("task_result", {"taskId": C_A, "status": "failed", "text": text})
           if text is not None else ("task_state", {"taskId": C_A, "status": "failed"}))
    journey = confirm_journey(tmp_path, monkeypatch, pre=[[row]])
    confirm_refused(journey, "journey_confirm_target_terminal")
    snapshot = journey.pre_cleanup_tasks()
    assert snapshot["confirm.a"] == {"trackerStatus": "failed", "nativeTerminal": None,
                                     "hostRefusalObserved": None}
    if text is not None:
        assert text not in json.dumps(snapshot)


def test_a_task_that_ends_before_the_yes_is_inconclusive_and_no_yes_is_sent(
    tmp_path, monkeypatch
):
    phases = confirm_phases(a=[*confirm_reply(REQ_A_SPOKEN, P_A,
                                              said=readback_of(confirm_objectives()[0])),
                               [task_result_row(C_A, "completed")]])
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_target_terminal")
    assert receipt["requestAInjected"] is True and receipt["yesInjected"] is False
    assert sent_speech_frames(journey) == clip_speech_frames(0, 1, 2)


def test_a_task_that_ends_during_the_no_drain_is_inconclusive(tmp_path, monkeypatch):
    phases = confirm_phases(no=[*confirm_reply("No.", P_NO, said="Okay."),
                                [task_result_row(C_B, "completed")]])
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_target_terminal")
    assert receipt["noInjected"] is True and receipt["requestAInjected"] is False


@pytest.mark.parametrize("phase", ["b", "a"])
def test_a_readback_naming_the_other_task_is_inconclusive_and_no_answer_is_sent(
    tmp_path, monkeypatch, phase
):
    obj_a, obj_b = confirm_objectives()
    spoken, pair, other = ((REQ_B_SPOKEN, P_B, obj_a) if phase == "b"
                           else (REQ_A_SPOKEN, P_A, obj_b))
    phases = confirm_phases(**{phase: confirm_reply(spoken, pair, said=readback_of(other))})
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_nomination_mismatch")
    key = "readbackBOutcome" if phase == "b" else "readbackAOutcome"
    assert receipt[key] == "other"
    assert receipt["yesInjected"] is False
    sent = clip_speech_frames(0) if phase == "b" else clip_speech_frames(0, 1, 2)
    assert sent_speech_frames(journey) == sent


@pytest.mark.parametrize("phase", ["b", "a"])
def test_a_reply_without_a_readback_is_clarified_and_inconclusive(tmp_path, monkeypatch, phase):
    spoken, pair = (REQ_B_SPOKEN, P_B) if phase == "b" else (REQ_A_SPOKEN, P_A)
    phases = confirm_phases(**{phase: confirm_reply(spoken, pair, said="Which task?")})
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_clarified")
    assert receipt["readbackBOutcome" if phase == "b" else "readbackAOutcome"] == "clarified"
    assert receipt["yesInjected"] is False


def test_a_readback_with_extra_words_is_not_the_readback(tmp_path, monkeypatch):
    said = readback_of(confirm_objectives()[1]) + " Really?"
    phases = confirm_phases(b=confirm_reply(REQ_B_SPOKEN, P_B, said=said))
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                    "journey_confirm_clarified")


# Failures.

def test_an_interrupted_readback_turn_fails(tmp_path, monkeypatch):
    phases = confirm_phases(b=confirm_reply(REQ_B_SPOKEN, P_B, interrupted=True,
                                            said=readback_of(confirm_objectives()[1])))
    receipt = confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                              "journey_confirm_readback_incomplete")
    assert receipt["readbackBOutcome"] == "interrupted"


def test_a_readback_turn_that_never_ends_fails(tmp_path, monkeypatch):
    phases = confirm_phases(b=confirm_reply(REQ_B_SPOKEN, P_B, complete=False,
                                            said=readback_of(confirm_objectives()[1])))
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                    "journey_confirm_readback_incomplete")


@pytest.mark.parametrize("task", [C_A, C_B])
@pytest.mark.parametrize("status", ["cancelling", "interrupted"])
def test_a_cancel_after_the_request_and_before_any_answer_is_premature(
    tmp_path, monkeypatch, task, status
):
    phases = confirm_phases(b=confirm_reply(REQ_B_SPOKEN, P_B, extra=[task_state_row(task, status)],
                                            said=readback_of(confirm_objectives()[1])))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_premature_cancel")
    assert receipt["noInjected"] is False and journey.witness_reads == []


def test_a_cancel_of_a_after_its_readback_and_before_the_yes_is_a_no_side_effect(
    tmp_path, monkeypatch
):
    phases = confirm_phases(a=confirm_reply(REQ_A_SPOKEN, P_A,
                                            extra=[task_state_row(C_A, "cancelling")],
                                            said=readback_of(confirm_objectives()[0])))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_no_cancelled")
    assert receipt["yesInjected"] is False


# Precedence: a failure is never masked by a later or simultaneous inconclusive outcome.

def test_a_mismatched_readback_with_a_premature_cancel_fails_premature(tmp_path, monkeypatch):
    obj_a, _ = confirm_objectives()
    phases = confirm_phases(b=confirm_reply(REQ_B_SPOKEN, P_B, said=readback_of(obj_a),
                                            extra=[task_state_row(C_A, "cancelling")]))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_premature_cancel")
    assert receipt["readbackBOutcome"] is None  # never classified past the side effect


def test_a_clarification_with_a_cancel_after_no_fails_no_cancelled(tmp_path, monkeypatch):
    phases = confirm_phases(a=confirm_reply(REQ_A_SPOKEN, P_A, said="Which task?",
                                            extra=[task_state_row(C_B, "cancelling")]))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_no_cancelled")
    assert receipt["readbackAOutcome"] is None


@pytest.mark.parametrize("phase", ["b", "a"])
def test_an_early_end_beside_a_cancel_fails_and_is_not_inconclusive(tmp_path, monkeypatch, phase):
    obj_a, obj_b = confirm_objectives()
    spoken, pair, said = ((REQ_B_SPOKEN, P_B, readback_of(obj_b)) if phase == "b"
                          else (REQ_A_SPOKEN, P_A, readback_of(obj_a)))
    extra = [task_result_row(C_A, "completed"), task_state_row(C_B, "cancelling")]
    phases = confirm_phases(**{phase: confirm_reply(spoken, pair, said=said, extra=extra)})
    category = ("journey_confirm_premature_cancel" if phase == "b"
                else "journey_confirm_no_cancelled")
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases), category)


@pytest.mark.parametrize("phase", ["b", "a"])
@pytest.mark.parametrize("other", ["active", "cancelling"])
def test_competing_work_beside_an_early_end_fails_and_is_not_inconclusive(
    tmp_path, monkeypatch, phase, other,
):
    """Astra 9263: a third task or admission next to an early end is competing work, a failure;
    the early end must never be what is reported."""
    obj_a, obj_b = confirm_objectives()
    spoken, pair, said = ((REQ_B_SPOKEN, P_B, readback_of(obj_b)) if phase == "b"
                          else (REQ_A_SPOKEN, P_A, readback_of(obj_a)))
    extra = [task_result_row(C_A, "completed"), task_state_row(C_OTHER, other), admit(RUN_X)]
    phases = confirm_phases(**{phase: confirm_reply(spoken, pair, said=said, extra=extra)})
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                    "journey_confirm_competing")


def _reply_with_finals(spoken, pair, finals):
    """A request phase whose reply turn publishes these assistant finals, in order."""
    phase = confirm_reply(spoken, pair)
    phase[2] = [("first_foreground_token", pair), *(assistant_final(t, pair) for t in finals)]
    return phase


@pytest.mark.parametrize("phase", ["b", "a"])
@pytest.mark.parametrize("shape", ["readback_then_text", "readback_twice", "both_readbacks"])
def test_a_reply_that_is_not_exactly_one_final_readback_is_ambiguous(
    tmp_path, monkeypatch, phase, shape,
):
    """PR #583 review 4609479174: the readback must be the reply's last assistant final and
    appear exactly once; extra or duplicate publications around it are refused."""
    obj_a, obj_b = confirm_objectives()
    want, other = ((readback_of(obj_b), readback_of(obj_a)) if phase == "b"
                   else (readback_of(obj_a), readback_of(obj_b)))
    finals = {"readback_then_text": [want, "Which one did you mean?"],
              "readback_twice": [want, want],
              "both_readbacks": [other, want]}[shape]
    spoken, pair = (REQ_B_SPOKEN, P_B) if phase == "b" else (REQ_A_SPOKEN, P_A)
    phases = confirm_phases(**{phase: _reply_with_finals(spoken, pair, finals)})
    receipt = confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                              "journey_confirm_ambiguous")
    assert receipt["readback" + phase.upper() + "Outcome"] == "ambiguous"


def test_prose_before_the_readback_still_reaches_the_target(tmp_path, monkeypatch):
    """Model prose spoken before the nomination is allowed: the host's readback is last."""
    obj_b = confirm_objectives()[1]
    b = _reply_with_finals(REQ_B_SPOKEN, P_B, ["Sure.", readback_of(obj_b)])
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(b=b))
    run_confirm(journey)
    assert journey.speak_receipt["readbackBOutcome"] == "target"


def _early_end_with_competing_work():
    return [task_result_row(C_A, "completed"), task_state_row(C_OTHER, "active"), admit(RUN_X)]


def test_competing_work_first_seen_at_the_quiet_gate_fails(tmp_path, monkeypatch):
    """Site-specific (Astra 9272): the quiet gate before request B sees the early end first.
    B's dispatch leaves an admission pending, so the gate polls; its next batch clears the
    pending admission together with the early end and the competing work."""
    journey = confirm_journey(
        tmp_path, monkeypatch,
        dispatches=confirm_dispatches(b=[("__pending__", {"count": 1})]),
        pre=[[("__pending__", {"count": 0}), *_early_end_with_competing_work()]],
    )
    receipt = confirm_refused(journey, "journey_confirm_competing")
    assert receipt["requestBInjected"] is False


def test_competing_work_first_seen_at_a_clip_lead_in_fails(tmp_path, monkeypatch):
    """Site-specific: request B's reply settles cleanly; the No clip's first-frame check is the
    first to see the early end and the competing work."""
    obj_b = confirm_objectives()[1]
    b = [*confirm_reply(REQ_B_SPOKEN, P_B, said=readback_of(obj_b)), [],
         _early_end_with_competing_work()]
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(b=b))
    receipt = confirm_refused(journey, "journey_confirm_competing")
    assert receipt["readbackBOutcome"] == "target" and receipt["noInjected"] is False


def test_competing_work_first_seen_after_no_fails(tmp_path, monkeypatch):
    """Site-specific: the post-No check is the first to see the early end."""
    no = confirm_reply("No.", P_NO, said="Okay.", extra=_early_end_with_competing_work())
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(no=no))
    receipt = confirm_refused(journey, "journey_confirm_competing")
    assert receipt["noInjected"] is True and receipt["requestAInjected"] is False


@pytest.mark.parametrize("other", ["active", "cancelling"])
def test_competing_work_beside_an_ended_bystander_after_yes_fails(tmp_path, monkeypatch, other):
    yes = confirm_yes(extra=[task_result_row(C_B, "completed"), task_state_row(C_OTHER, other),
                             admit(RUN_X)])
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(yes=yes))
    confirm_refused(journey, "journey_confirm_competing")


def test_a_premature_a_cancel_beside_an_ended_bystander_fails_premature(tmp_path, monkeypatch):
    yes = confirm_yes(before_final=[task_state_row(C_A, "cancelling")],
                      cancel=[task_result_row(C_B, "completed")])
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(yes=yes))
    confirm_refused(journey, "journey_confirm_premature_cancel")


def test_a_ends_early_and_b_cancelled_after_yes_is_the_wrong_target(tmp_path, monkeypatch):
    yes = confirm_yes(cancel=[task_result_row(C_A, "completed"),
                              task_state_row(C_B, "cancelling")])
    yes[3] = []
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(yes=yes))
    receipt = confirm_refused(journey, "journey_confirm_wrong_target")
    assert receipt["bystanderCancelObserved"] is True and journey.witness_reads == []


def test_a_b_cancel_during_the_yes_lead_in_is_a_no_side_effect(tmp_path, monkeypatch):
    obj_a, _ = confirm_objectives()
    phases = confirm_phases(a=[*confirm_reply(REQ_A_SPOKEN, P_A, said=readback_of(obj_a)),
                               [], [task_state_row(C_B, "cancelling")]])
    receipt = confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                              "journey_confirm_no_cancelled")
    assert receipt["yesInjected"] is False


def test_a_cancel_of_a_before_the_yes_final_is_premature(tmp_path, monkeypatch):
    phases = confirm_phases(yes=confirm_yes(cancel=(),
                                            before_final=[task_state_row(C_A, "cancelling")]))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_premature_cancel")
    assert receipt["targetCancelObserved"] is False and journey.witness_reads == []


@pytest.mark.parametrize("task", [C_A, C_B])
def test_a_cancel_after_no_fails(tmp_path, monkeypatch, task):
    phases = confirm_phases(no=confirm_reply("No.", P_NO, said="Okay.",
                                             extra=[task_state_row(task, "cancelling")]))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_no_cancelled")
    assert receipt["bothActiveAfterNo"] is False and receipt["requestAInjected"] is False


def test_a_cancel_late_in_the_no_drain_still_fails(tmp_path, monkeypatch):
    phases = confirm_phases(no=[*confirm_reply("No.", P_NO, said="Okay."),
                                [task_state_row(C_B, "cancelling")]])  # after its reply ended
    receipt = confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                              "journey_confirm_no_cancelled")
    assert receipt["bothActiveAfterNo"] is False  # caught inside the drain window itself


def test_a_dispatch_whose_active_state_precedes_its_send_is_unbound(tmp_path, monkeypatch):
    obj_a, _ = confirm_objectives()
    dispatches = confirm_dispatches()
    dispatches["task: " + obj_a] = [admit(RUN_A)]
    journey = confirm_journey(tmp_path, monkeypatch, dispatches=dispatches)
    journey.queues["pre"][1].append(task_state_row(C_A, "active"))  # before the typed send
    receipt = confirm_refused(journey, "journey_confirm_dispatch_unbound")
    assert receipt["tasksBound"] is False


def test_a_pending_admission_at_the_first_request_frame_aborts(tmp_path, monkeypatch):
    journey = confirm_journey(tmp_path, monkeypatch, pre=[[("__pending__", {"count": 1})]])
    receipt = confirm_refused(journey, "journey_confirm_competing")
    assert receipt["requestBInjected"] is False and sent_speech_frames(journey) == 0


def test_a_target_that_completes_after_cancelling_is_not_a_proven_cancel(tmp_path, monkeypatch):
    yes = confirm_yes()
    yes[3] = [task_result_row(C_A, "completed")]  # cancelling, then a natural completion
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(yes=yes),
                              witness={RUN_A: "completed"})
    receipt = confirm_refused(journey, "journey_cancel_unproven")
    assert receipt["nativeCancelledBeforeCleanup"] is False
    assert ("live", RUN_B) not in journey.witness_reads


def test_cleanup_cancellation_is_never_recorded_as_trace_evidence(tmp_path, monkeypatch):
    import scripts.native_typed_journey as client

    traces = []

    class Recording(client.ConfirmCancelTrace):
        def __init__(self, *args):
            super().__init__(*args)
            traces.append(self)

    monkeypatch.setattr(client, "ConfirmCancelTrace", Recording)
    journey = confirm_journey(tmp_path, monkeypatch)
    run_confirm(journey)
    assert journey.speak_receipt["successPathCleanupCancels"] == 1
    assert journey.tracker.tasks[C_B] == "interrupted"  # cleanup did cancel B
    (trace,) = traces
    assert trace.cancels(C_B) == []  # but its events never reached the step's trace


def test_cancelling_the_other_task_after_yes_is_the_wrong_target(tmp_path, monkeypatch):
    phases = confirm_phases(yes=confirm_yes(extra=[task_state_row(C_B, "cancelling")]))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_wrong_target")
    assert receipt["bystanderCancelObserved"] is True
    assert receipt["nativeCancelledBeforeCleanup"] is False and journey.witness_reads == []


def test_cancelling_only_the_other_task_after_yes_is_the_wrong_target(tmp_path, monkeypatch):
    phases = confirm_phases(yes=confirm_yes(cancel=[task_state_row(C_B, "cancelling")]))
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                    "journey_confirm_wrong_target")


def test_a_cancel_of_the_other_task_during_witness_collection_is_the_wrong_target(
    tmp_path, monkeypatch
):
    yes = confirm_yes()
    yes[3] = []  # A settles only after B is cancelled, while collect() polls
    yes += [[task_state_row(C_B, "cancelling")],
            [task_state_row(C_A, "interrupted"), task_result_row(C_A, "interrupted")]]
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(yes=yes))
    receipt = confirm_refused(journey, "journey_confirm_wrong_target")
    assert receipt["bystanderActiveAtWitness"] is False
    assert ("live", RUN_B) not in journey.witness_reads


def test_a_bystander_that_ends_before_the_witness_is_inconclusive(tmp_path, monkeypatch):
    phases = confirm_phases(yes=confirm_yes(extra=[task_result_row(C_B, "completed")]))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_bystander_ended")
    assert receipt["bystanderActiveAtWitness"] is False
    assert ("live", RUN_B) not in journey.witness_reads


def test_a_yes_that_cancels_nothing_is_not_accepted(tmp_path, monkeypatch):
    phases = confirm_phases(yes=confirm_yes(cancel=())[:2])
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    receipt = confirm_refused(journey, "journey_confirm_not_cancelled")
    assert receipt["targetCancelObserved"] is False
    assert journey.witness_reads == [] and journey.results == {}


def test_a_cancel_visible_only_as_a_terminal_result_is_not_arming(tmp_path, monkeypatch):
    yes = confirm_yes(cancel=())
    yes[3] = [task_result_row(C_A, "interrupted")]
    journey = confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(yes=yes))
    confirm_refused(journey, "journey_confirm_not_cancelled")
    assert journey.witness_reads == []


def test_a_target_cancelled_only_by_cleanup_is_never_accepted(tmp_path, monkeypatch):
    phases = confirm_phases(yes=confirm_yes(cancel=())[:2])
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)

    async def run():
        with pytest.raises(ValueError, match="journey_confirm_not_cancelled"):
            await asyncio.wait_for(journey.step(CONFIRM_STEP), 15)
        return await journey.settle_owned()

    asyncio.run(run())
    receipt = journey.speak_receipt
    # Cleanup cancelled both tasks by exact id; the refused verdict and its evidence stand.
    assert journey.inputs[-2:] == ["cancel task: " + C_A, "cancel task: " + C_B]
    assert receipt["targetCancelObserved"] is False
    assert receipt["nativeCancelledBeforeCleanup"] is False
    assert receipt["successPathCleanupCancels"] == 0


def test_the_native_witness_must_say_the_target_was_cancelled(tmp_path, monkeypatch):
    journey = confirm_journey(tmp_path, monkeypatch, witness={RUN_A: "completed"})
    receipt = confirm_refused(journey, "journey_terminal_binding")
    assert receipt["nativeCancelledBeforeCleanup"] is False
    assert journey.witness_reads == [("read", RUN_A)]


@pytest.mark.parametrize("status,category", [
    ("cancelled", "journey_confirm_wrong_target"),
    ("interrupted", "journey_confirm_wrong_target"),
    ("stopping", "journey_confirm_wrong_target"),
    ("completed", "journey_confirm_bystander_ended"),
    ("failed", "journey_confirm_bystander_ended"),
])
def test_the_native_witness_must_say_the_bystander_is_still_live(
    tmp_path, monkeypatch, status, category
):
    journey = confirm_journey(tmp_path, monkeypatch, live={RUN_B: status})
    receipt = confirm_refused(journey, category)
    assert receipt["nativeCancelledBeforeCleanup"] is True
    assert receipt["nativeBystanderLiveBeforeCleanup"] is False
    assert receipt["successPathCleanupCancels"] == 0


def test_a_bystander_cancelled_after_its_live_witness_read_still_fails(tmp_path, monkeypatch):
    journey = confirm_journey(tmp_path, monkeypatch)
    real = journey.poll
    polls_after_live = []

    async def poll():
        if ("live", RUN_B) in journey.witness_reads and not polls_after_live:
            polls_after_live.append(1)
            # Delivered at the first poll after the live read, before any cleanup.
            journey.queues[3].insert(0, [task_state_row(C_B, "cancelling")])
        await real()

    journey.poll = poll
    receipt = confirm_refused(journey, "journey_confirm_wrong_target")
    assert polls_after_live and receipt["successPathCleanupCancels"] == 0


# Input and attribution.

def test_a_typed_dispatch_that_starts_nothing_fails(tmp_path, monkeypatch):
    obj_a, _ = confirm_objectives()
    dispatches = confirm_dispatches()
    dispatches["task: " + obj_a] = []
    receipt = confirm_refused(confirm_journey(tmp_path, monkeypatch, dispatches=dispatches),
                              "journey_confirm_not_dispatched")
    assert receipt["tasksBound"] is False


def test_a_competing_task_during_a_typed_dispatch_fails(tmp_path, monkeypatch):
    dispatches = confirm_dispatches(a=[task_state_row(C_OTHER, "active"), admit(RUN_X)])
    journey = confirm_journey(tmp_path, monkeypatch, dispatches=dispatches)
    confirm_refused(journey, "journey_confirm_competing")
    assert journey.bindings == {}  # never guessed into ownership


def test_a_dispatch_whose_first_state_is_not_active_is_unbound(tmp_path, monkeypatch):
    obj_a, _ = confirm_objectives()
    dispatches = confirm_dispatches()
    dispatches["task: " + obj_a] = [task_result_row(C_A, "completed"),
                                    task_state_row(C_A, "active"), admit(RUN_A)]
    confirm_refused(confirm_journey(tmp_path, monkeypatch, dispatches=dispatches),
                    "journey_confirm_dispatch_unbound")


def test_a_replayed_prior_task_during_dispatch_fails(tmp_path, monkeypatch):
    dispatches = confirm_dispatches(a=[task_state_row(T_OLD, "active")])
    journey = confirm_journey(tmp_path, monkeypatch, dispatches=dispatches,
                              before=[task_state_row(T_OLD, "completed")], journal=[RUN_OLD])
    confirm_refused(journey, "journey_confirm_competing")


@pytest.mark.parametrize("rows", [
    [("voice_activity_started", {})],
    [user_final("unexpected extra input")],
    [("assistant_turn_completed", {})],
], ids=["voice-start", "user-final", "unbound-foreground"])
def test_unexpected_input_before_the_first_request_fails(tmp_path, monkeypatch, rows):
    journey = confirm_journey(tmp_path, monkeypatch, pre=[rows])
    receipt = confirm_refused(journey, "journey_confirm_unexpected_input")
    assert receipt["unexpectedInputCount"] == 1 and receipt["requestBInjected"] is False


def test_a_competing_task_at_the_first_request_frame_aborts(tmp_path, monkeypatch):
    journey = confirm_journey(tmp_path, monkeypatch,
                              pre=[[task_state_row(C_OTHER, "active"), admit(RUN_X)]])
    receipt = confirm_refused(journey, "journey_confirm_competing")
    assert receipt["requestBInjected"] is False and sent_speech_frames(journey) == 0


def test_a_competing_task_at_the_yes_frame_aborts_before_the_yes(tmp_path, monkeypatch):
    phases = confirm_phases(a=[*confirm_reply(REQ_A_SPOKEN, P_A,
                                              said=readback_of(confirm_objectives()[0])),
                               [admit(RUN_X)]])
    receipt = confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                              "journey_confirm_competing")
    assert receipt["yesInjected"] is False


@pytest.mark.parametrize("phase,spoken,key", [
    ("b", "Please stop the task.", "requestB"),
    ("no", "Know.", "no"),
    ("a", "Please stop the rail task.", "requestA"),
])
def test_wrong_words_for_a_clip_fail(tmp_path, monkeypatch, phase, spoken, key):
    obj_a, obj_b = confirm_objectives()
    pair = {"b": P_B, "no": P_NO, "a": P_A}[phase]
    said = {"b": readback_of(obj_b), "no": "Okay.", "a": readback_of(obj_a)}[phase]
    phases = confirm_phases(**{phase: confirm_reply(spoken, pair, said=said)})
    receipt = confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                              "journey_confirm_transcript")
    assert receipt[key + "TranscriptMatches"] is False and receipt["yesInjected"] is False


def test_a_yes_with_wrong_words_cancels_nothing_it_can_claim(tmp_path, monkeypatch):
    yes = confirm_yes()
    yes[1] = [("voice_activity_ended", {}), user_final("Yes, please.")]
    receipt = confirm_refused(confirm_journey(tmp_path, monkeypatch,
                                              phases=confirm_phases(yes=yes)),
                              "journey_confirm_transcript")
    assert receipt["yesTranscriptMatches"] is False
    assert receipt["nativeCancelledBeforeCleanup"] is False


@pytest.mark.parametrize("phase", ["b", "no", "a"])
def test_two_finals_for_a_clip_fail(tmp_path, monkeypatch, phase):
    obj_a, obj_b = confirm_objectives()
    spoken = {"b": REQ_B_SPOKEN, "no": "No.", "a": REQ_A_SPOKEN}[phase]
    pair = {"b": P_B, "no": P_NO, "a": P_A}[phase]
    said = {"b": readback_of(obj_b), "no": "Okay.", "a": readback_of(obj_a)}[phase]
    batches = confirm_reply(spoken, pair, said=said)
    batches[1].append(user_final(spoken))
    confirm_refused(confirm_journey(tmp_path, monkeypatch,
                                    phases=confirm_phases(**{phase: batches})),
                    "journey_confirm_finals")


def test_a_clip_with_no_voice_start_fails(tmp_path, monkeypatch):
    batches = confirm_reply(REQ_B_SPOKEN, P_B, said=readback_of(confirm_objectives()[1]))
    batches[0] = []
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(b=batches)),
                    "journey_confirm_vad")


def test_a_second_reply_turn_for_a_request_is_ambiguous(tmp_path, monkeypatch):
    other = {"turnId": "confirm_other", "turnGeneration": 1}
    batches = confirm_reply(REQ_B_SPOKEN, P_B, said=readback_of(confirm_objectives()[1]))
    batches[3] += [("first_foreground_token", other), ("assistant_turn_completed", other)]
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(b=batches)),
                    "journey_confirm_ambiguous")


def test_a_readback_from_a_reply_that_starts_before_the_final_is_not_bound(tmp_path, monkeypatch):
    batches = confirm_reply(REQ_B_SPOKEN, P_B, said=readback_of(confirm_objectives()[1]))
    batches[0] = [("voice_activity_started", {}), ("first_foreground_token", P_B)]
    batches[2] = batches[2][1:]
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=confirm_phases(b=batches)),
                    "journey_confirm_ambiguous")


def test_a_task_dispatched_after_the_yes_fails(tmp_path, monkeypatch):
    phases = confirm_phases(yes=confirm_yes(extra=[task_state_row(C_OTHER, "active")]))
    confirm_refused(confirm_journey(tmp_path, monkeypatch, phases=phases),
                    "journey_confirm_competing")


def test_both_tasks_are_owned_for_cleanup_even_when_the_step_fails(tmp_path, monkeypatch):
    phases = confirm_phases(b=confirm_reply(REQ_B_SPOKEN, P_B, said="Which task?"))
    journey = confirm_journey(tmp_path, monkeypatch, phases=phases)
    confirm_refused(journey, "journey_confirm_clarified")
    assert journey.bindings == {"confirm.a": (C_A, RUN_A), "confirm.b": (C_B, RUN_B)}
    assert journey.tracker.speak is None and journey.voice.meter.window == "closed"
