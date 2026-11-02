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


def event(sequence, kind, **data):
    return {"sequence": sequence, "kind": kind, "monotonicMs": 10.5 + sequence, "data": data}


def test_public_lifecycle_events_and_float_clock_do_not_fail_or_claim_warm_audio():
    tracker = EventTracker()
    tracker.begin("synthetic", 1, "Synthetic text")
    tracker.consume(
        [
            event(1, "typed_input_admitted", inputSequence=1),
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
            event(2, "typed_input_admitted", inputSequence=1),
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
                event(4, "typed_input_admitted", inputSequence=2),
                event(5, "first_foreground_token", turnId="opaque_B", turnGeneration=2),
                event(6, "assistant_text_generated", turnId="opaque_B", turnGeneration=2),
                event(7, "assistant_turn_completed", turnId="opaque_B", turnGeneration=2),
            ],
        ]
    )

    async def request(route, payload=None):
        return {"version": 1, "events": next(batches)} if route == "events" else {}

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
