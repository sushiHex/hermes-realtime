"""Offline contracts for the optional typed native-journey client."""

import asyncio
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
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic input")
    tracker.consume([
        event(1, "typed_input_admitted", inputSequence=1, bindingGeneration=7),
        event(2, "assistant_turn_completed", turnId="actual_turn", turnGeneration=1,
              inputSequence=1, bindingGeneration=7),
    ])
    assert tracker.current["outcome"] == "pending"
    assert tracker.pair is None
    tracker.consume([event(3, "transcript_final", role="user", text="Synthetic input"),
                     event(4, "first_foreground_token", turnId="actual_turn", turnGeneration=1)])
    assert tracker.current["outcome"] == "pending"


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


@pytest.mark.parametrize("route,version", [("events", 2), ("input", 1)])
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
