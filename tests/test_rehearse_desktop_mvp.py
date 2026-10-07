from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rehearse_desktop_mvp.py"
sys.path.insert(0, str(_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("rehearse_desktop_mvp", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
rehearsal = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rehearsal
_SPEC.loader.exec_module(rehearsal)

_TERMINAL = [{"type": "function", "function": {"name": "terminal", "parameters": {}}}]
_MEMORY = [{"type": "function", "function": {"name": "memory", "parameters": {}}}]


def _body(objective: object, *, tools: object = _TERMINAL, after: list[object] | None = None):
    return {
        "messages": [
            {"role": "system", "content": "about 99 seconds, until you are stopped"},
            {"role": "user", "content": objective},
            *(after or []),
        ],
        "tools": tools,
    }


@pytest.mark.parametrize(
    ("objective", "expected"),
    [
        ("Reply with one sentence confirming the rehearsal delegation.", ("text", "")),
        ("Wait about 60 seconds, then reply with one sentence.", ("timed", "60")),
        (
            "Keep working on the rehearsal restart task until you are stopped.",
            ("endless", "restart"),
        ),
        (
            "Without using tools, think carefully for several minutes before replying.",
            ("endless", "other"),
        ),
        (
            "run exactly this command: chmod 777 C:/tmp/hermes-approval-probe-ab12. If ...",
            ("tool", "chmod 777 C:/tmp/hermes-approval-probe-ab12"),
        ),
    ],
)
def test_the_stand_in_routes_dispatched_work_by_its_objective(objective, expected) -> None:
    assert rehearsal.route(_body(objective)) == expected


def test_the_stand_in_reads_text_parts_of_the_last_user_turn() -> None:
    parts = [{"type": "text", "text": "Wait about 7 seconds"}, {"type": "image_url"}]
    assert rehearsal.route(_body(parts)) == ("timed", "7")


def test_work_without_the_terminal_tool_is_never_held() -> None:
    # A review or summary quotes the conversation, task commands included; it must end at
    # once, or a running review would keep a deletion pending for ever.
    body = _body("rehearsal restart task until you are stopped", tools=_MEMORY)
    assert rehearsal.route(body) == ("text", "")
    assert rehearsal.route(_body("about 60 seconds", tools=None)) == ("text", "")


def test_a_tool_result_ends_the_run() -> None:
    tool_turn = [
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "content": "denied"},
    ]
    assert rehearsal.route(_body("chmod 777 /x", after=tool_turn)) == ("text", "")


def test_a_malformed_request_gets_a_sentence() -> None:
    for body in (None, {}, {"messages": []}, {"messages": "x"}, {"messages": [{"role": "x"}]}):
        assert rehearsal.route(body) == ("text", "")


def test_markers_are_kept_by_name_and_rendered_from_their_values() -> None:
    lines = [
        '[voice-tail] {"version": 1, "restored": 3}',
        "plain output",
        '[voice-tail] not json {',
        '[voice-tail] {not json}',
        '[Bad-Name] {"x":1}',
        '[some-upstream-tool] {"x":1}',
        '[hermes-restart-settlement] {"stopped":1}',
        '[voice-archive-send] {"path":"C:/Users/someone/home"}',
        '[voice-archive-send] {"note":"two words"}',
        '[voice-archive-send] {"Bad Key":1}',
        "[voice-review] {" + '"x":"' + "a" * 600 + '"}',
        '  [voice-tail] {"x":1}',
        '[real-hermes-gate] {"category":"health","components":{"hermes-identity":'
        '{"refusal":"modified"}},"log":{"aiohttp.client":{"WARNING":1}},"version":1}',
    ]

    found, dropped = rehearsal.markers(lines)

    assert found == [
        '[voice-tail] {"restored":3,"version":1}',
        '[hermes-restart-settlement] {"stopped":1}',
        '[real-hermes-gate] {"category":"health","components":{"hermes-identity":'
        '{"refusal":"modified"}},"log":{"aiohttp.client":{"WARNING":1}},"version":1}',
    ]
    # Unknown names, text-bearing values and malformed bodies are counted, never recorded.
    assert dropped == 6
    assert "someone" not in json.dumps(found)


def test_markers_are_capped_and_the_rest_counted() -> None:
    lines = [f'[voice-archive-send] {{"n":{index}}}' for index in range(40)]

    found, dropped = rehearsal.markers(lines)

    assert found == lines[:32]
    assert dropped == 8


def test_tracebacks_are_classified_by_allowlist_and_unknown_ones_named_by_type() -> None:
    lines = [
        "Exception ignored in: <function FfiHandle.__del__ at 0x0>",
        "Traceback (most recent call last):",
        '  File "C:\\x\\livekit\\rtc\\_ffi_client.py", line 99, in dispose',
        "    assert dropped",
        "AssertionError: ",
        "Traceback (most recent call last):",
        '  File "C:\\x\\gateway\\shutdown_watchdog.py", line 572, in loop',
        "    tick_server = await asyncio.start_unix_server(",
        "AttributeError: module 'asyncio' has no attribute 'start_unix_server'",
        "Traceback (most recent call last):",
        '  File "C:\\Users\\someone\\private.py", line 1, in f',
        "RuntimeError: secret detail C:\\Users\\someone",
        "Traceback (most recent call last):",
        '  File "C:\\x\\_ffi_client.py", line 1, in f',
        "ValueError: not the known one",
        "Exception ignored in: <function FfiHandle.__del__ at 0x0>",
        "Traceback (most recent call last):",
        '  File "C:\\x\\livekit\\rtc\\_ffi_client.py", line 99, in dispose',
        "TypeError: the known place, another failure",
    ]

    known, unexpected = rehearsal.tracebacks(lines)

    assert known == {"livekit_ffi_handle_dispose": 1, "hermes_watchdog_unix_server": 1}
    assert unexpected == {"RuntimeError": 1, "ValueError": 1, "TypeError": 1}


def test_a_step_applies_each_finding_at_its_level() -> None:
    step = rehearsal.Step("6", "cancel_task", [])
    step.apply([("differ", "state_sequence")])
    assert (step.outcome, step.category) == ("different", "state_sequence")
    step.apply([("fail", "work_left_running")])
    assert (step.outcome, step.category) == ("failed", "work_left_running")
    assert step.findings == ["state_sequence", "work_left_running"]


class _Page:
    def __init__(self, rows: list[str]) -> None:
        self.rows = rows

    async def evaluate(self, script: str, *arguments: object) -> object:
        del script, arguments
        return self.rows


@pytest.mark.asyncio
async def test_a_reply_says_the_phrase_if_any_of_its_rows_does() -> None:
    probe = object.__new__(rehearsal.Rehearsal)
    probe.phrase = "hoopoe"
    # One row per sentence: the phrase is in the first, not the last.
    probe.page = _Page(["Your favorite bird is the Hoopoe.", "It has a crest."])
    assert await probe.turn_has_phrase()
    probe.page = _Page(["I do not know.", "Tell me more."])
    assert not await probe.turn_has_phrase()


def test_a_step_with_an_unexpected_traceback_is_different(tmp_path: Path) -> None:
    log = tmp_path / "host.log"
    log.write_text("", encoding="utf-8")
    tail = rehearsal.LogTail("host", log)
    step = rehearsal.Step("2", "start_host_open_browser_talk", [tail])
    log.write_text(
        "Traceback (most recent call last):\n  File \"x.py\", line 1\nKeyError: 'k'\n",
        encoding="utf-8",
    )

    record = step.record()

    assert record["outcome"] == "different"
    assert record["category"] == "unexpected_traceback"
    assert record["tracebacks"] == {"known": {}, "unexpected": {"host:KeyError": 1}}


def test_a_known_traceback_alone_keeps_the_step_as_expected(tmp_path: Path) -> None:
    log = tmp_path / "host.log"
    log.write_text("", encoding="utf-8")
    tail = rehearsal.LogTail("host", log)
    step = rehearsal.Step("9", "cleanup", [tail])
    log.write_text(
        "Exception ignored in: <function FfiHandle.__del__ at 0x0>\n"
        "Traceback (most recent call last):\n"
        '  File "_ffi_client.py", line 99, in dispose\n'
        "AssertionError: \n",
        encoding="utf-8",
    )

    assert step.record()["outcome"] == "as_expected"


def test_the_transcript_must_say_what_was_spoken() -> None:
    expected = "What is the capital of France?"
    assert rehearsal.transcript_matches(expected, "What is the capital of France?")
    assert rehearsal.transcript_matches(expected, "what's... what is the capital of france")
    assert not rehearsal.transcript_matches(expected, "What is the capital of Spain?")
    assert not rehearsal.transcript_matches(expected, "")


_SEEDED = {"page": 1, "voice_tail": 2, "hermes_database": 3, "sessions": 0, "memories": 0}
_GONE = {"page": 0, "voice_tail": 0, "hermes_database": 0, "sessions": 0, "memories": 0}


def test_deletion_passes_only_when_a_seeded_phrase_is_gone_everywhere() -> None:
    assert rehearsal.deletion_verdict(_SEEDED, _GONE, _GONE, False) == []
    assert rehearsal.deletion_verdict(_SEEDED, _GONE | {"sessions": 1}, _GONE, False) == [
        ("fail", "phrase_retained")
    ]
    assert rehearsal.deletion_verdict(_SEEDED, _GONE, _GONE | {"voice_tail": 1}, False) == [
        ("fail", "phrase_returned")
    ]
    assert rehearsal.deletion_verdict(_SEEDED, _GONE, _GONE, True) == [
        ("fail", "phrase_in_next_reply")
    ]
    # A phrase never seen before deletion proves nothing about deletion.
    for source in ("voice_tail", "hermes_database"):
        assert ("fail", "phrase_not_seeded") in rehearsal.deletion_verdict(
            _SEEDED | {source: 0}, _GONE, _GONE, False
        )
    # Built-in memory keeps what was learned: the documented limit, not a failure.
    assert rehearsal.deletion_verdict(_SEEDED, _GONE | {"memories": 1}, _GONE, False) == [
        ("differ", "memory_retains_phrase")
    ]


def test_context_and_work_verdicts() -> None:
    assert rehearsal.context_verdict(True, True) == []
    assert rehearsal.context_verdict(False, True) == [("differ", "context_lost")]
    assert rehearsal.context_verdict(True, False) == [("differ", "context_not_answered")]
    assert rehearsal.cancel_work_verdict(0) == []
    assert rehearsal.cancel_work_verdict(1) == [("fail", "work_left_running")]
    assert rehearsal.restart_work_verdict(1, 1, 0) == []
    assert rehearsal.restart_work_verdict(0, 0, 0) == [("differ", "no_work_left_by_crash")]
    assert rehearsal.restart_work_verdict(1, 0, 0) == [("differ", "no_work_left_by_crash")]
    assert rehearsal.restart_work_verdict(1, 1, 1) == [("fail", "work_left_running")]


def _frames(start: float, seconds: float, energy: float) -> list[tuple[float, float]]:
    return [(start + 50 * index, energy) for index in range(int(seconds * 20))]


def test_only_the_identified_readiness_cue_is_excused() -> None:
    quiet = _frames(0, 7, 0.0)
    cue = _frames(1000, 1.0, 5e-4)
    late = _frames(5000, 1.0, 5e-4)
    timeline = sorted(quiet + cue)

    assert rehearsal.audio_without_speech(timeline, 900.0, 0)[0] == []
    # Without an identified cue, the same burst is stale.
    assert rehearsal.audio_without_speech(timeline, None, 0)[0] == [("differ", "stale_audio")]
    # A burst away from the confirmation is stale.
    assert rehearsal.audio_without_speech(sorted(quiet + cue + late), 900.0, 0)[0] == [
        ("differ", "stale_audio")
    ]
    # A burst longer than one cue is stale, even inside the window around the confirmation.
    long = _frames(1000, 2.5, 5e-4)
    assert rehearsal.audio_without_speech(sorted(quiet + long), 900.0, 0)[0] == [
        ("differ", "stale_audio")
    ]
    # Speech rendered as an assistant row is stale too: nothing was asked.
    assert rehearsal.audio_without_speech(quiet, None, 1)[0] == [("differ", "stale_audio")]


def test_the_database_scan_reads_every_table_and_full_text_index(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    with contextlib.closing(sqlite3.connect(database)) as db:
        db.execute("CREATE TABLE messages (id INTEGER, content TEXT)")
        db.execute("CREATE TABLE sessions (id INTEGER, title TEXT, extra BLOB)")
        db.execute("CREATE VIRTUAL TABLE search USING fts5(body, content='')")
        db.execute("INSERT INTO messages VALUES (1, 'nothing here')")
        db.commit()
    assert rehearsal.phrase_in_database(database, "hoopoe") == 0

    with contextlib.closing(sqlite3.connect(database)) as db:
        db.execute("INSERT INTO sessions VALUES (1, 'about a Hoopoe', NULL)")
        db.commit()
    assert rehearsal.phrase_in_database(database, "hoopoe") == 1

    with contextlib.closing(sqlite3.connect(database)) as db:
        db.execute("DELETE FROM sessions")
        # A contentless index holds no readable text: only MATCH finds it.
        db.execute("INSERT INTO search(rowid, body) VALUES (1, 'the hoopoe again')")
        db.commit()
    assert rehearsal.phrase_in_database(database, "hoopoe") == 1

    with pytest.raises(FileNotFoundError):
        rehearsal.phrase_in_database(tmp_path / "missing.db", "hoopoe")


def test_the_file_scan_reads_every_file_and_refuses_a_missing_directory(tmp_path: Path) -> None:
    (tmp_path / "sessions" / "nested").mkdir(parents=True)
    (tmp_path / "sessions" / "a.json").write_text('{"t":"x"}', encoding="utf-8")
    assert rehearsal.phrase_in_files(tmp_path / "sessions", "hoopoe") == 0
    (tmp_path / "sessions" / "nested" / "b.jsonl").write_text("HOOPOE", encoding="utf-8")
    assert rehearsal.phrase_in_files(tmp_path / "sessions", "hoopoe") == 1
    with pytest.raises(FileNotFoundError):
        rehearsal.phrase_in_files(tmp_path / "memories", "hoopoe")


def test_every_verdict_is_applied_where_it_is_observed() -> None:
    source = {
        name: inspect.getsource(getattr(rehearsal.Rehearsal, name))
        for name in (
            "start_and_talk",
            "delegate",
            "cancel",
            "reconnect",
            "_after_reconnect",
            "_context_check",
            "restart",
            "delete",
            "cleanup",
            "quiet",
            "_ask_phrase",
        )
    }
    assert "self.quiet(step" in source["start_and_talk"]
    assert "transcript_matches(" in source["start_and_talk"]
    assert 'step.differ("result_not_reported")' in source["delegate"]
    assert "step.apply(cancel_work_verdict(" in source["cancel"]
    assert "self.quiet(step" in source["_after_reconnect"]
    assert source["reconnect"].count("self._after_reconnect(") == 2
    assert "self._context_check(" in source["reconnect"]
    assert "step.apply(context_verdict(" in source["_context_check"]
    assert "restart_work_verdict(" in source["restart"]
    assert "self._context_check(" in source["restart"]
    assert "step.apply(deletion_verdict(" in source["delete"]
    assert "step.apply(retained_verdict(" in source["cleanup"]
    assert "step.apply(findings)" in source["quiet"]
    assert "self.turn_has_phrase()" in source["_ask_phrase"]
    # Nothing restates the fact a context check depends on.
    assert not hasattr(rehearsal.Rehearsal, "_restate")
    assert "favorite bird is" not in source["reconnect"] + source["restart"]


@pytest.mark.skipif(sys.platform != "win32", reason="Job Objects are Windows containment")
def test_a_hard_killed_rehearsal_leaves_no_orphan(tmp_path: Path) -> None:
    import subprocess

    probe = tmp_path / "contained.py"
    probe.write_text(
        "import os, subprocess, sys\n"
        f"sys.path.insert(0, {str(_PATH.parent)!r})\n"
        "import rehearse_desktop_mvp as r\n"
        "containment = r.Containment()\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print(child.pid, len(containment.members() - {os.getpid()}), flush=True)\n"
        "os._exit(9)\n",
        encoding="utf-8",
    )
    # The base interpreter: a venv launcher would contain its child in a job of its own.
    base = getattr(sys, "_base_executable", sys.executable)
    completed = subprocess.run(
        (base, str(probe)), capture_output=True, text=True, timeout=60, check=False
    )
    pid, members = (int(value) for value in completed.stdout.split())

    assert completed.returncode == 9
    assert members == 1
    deadline = time.monotonic() + 10
    while rehearsal.started_at(pid) is not None and time.monotonic() < deadline:
        time.sleep(0.1)
    assert rehearsal.started_at(pid) is None


def test_notice_reads_only_the_counts() -> None:
    line = (
        "NOTICE: a previous session left Hermes background work behind: 2 run(s) were "
        "stopped or had already ended, and 1 dispatch(es) have an unknown outcome. "
        "Tasks are not resumed."
    )
    assert rehearsal.notice(["x", line]) == {"stopped": 2, "unknown": 1}
    assert rehearsal.notice(["NOTICE: something else"]) is None


def test_latencies_take_the_latest_named_value_and_refuse_other_shapes() -> None:
    entries = [
        "transcript_to_first_token: 41.2 ms",
        "first_token_to_audio: 300 ms",
        "transcript_to_first_token: 39.0 ms",
        "typed_input_admitted: server monotonic 1234.5 ms",
        "heard words here: 3.0 ms",
    ]
    assert rehearsal.latencies(entries) == {
        "transcript_to_first_token": 39.0,
        "first_token_to_audio": 300.0,
    }


def test_descendants_follow_the_whole_tree_from_live_roots_only() -> None:
    table = {
        10: (1, "uv.exe"),
        11: (10, "python.exe"),
        12: (11, "python.exe"),
        20: (1, "other.exe"),
        21: (99, "orphan.exe"),
    }
    assert rehearsal.descendants(table, {10, 30}) == {10, 11, 12}


def test_page_categories_carry_no_text_the_page_holds() -> None:
    snapshot = {
        "state": "connected",
        "label": "heard: my secret words",
        "toggle": "Stop session",
        "typed": True,
        "users": 2,
        "assistants": 3,
        "live": 0,
        "tasks": [["task_private_1", "completed"], ["task_private_2", "active"]],
        "markers": [
            "first_token_to_audio: 812.4 ms",
            "typed_input_admitted: server monotonic 1234.5 ms",
            "heard my secret words: 3.0 ms",
            "room browser-acceptance-0a1b: x ms",
        ],
    }

    categories = rehearsal._page_categories(snapshot)

    assert categories == {
        "connection": "connected",
        "label": "other",
        "toggle": "Stop session",
        "typed_enabled": True,
        "users": 2,
        "assistants": 3,
        "live": 0,
        "tasks": ["active", "completed"],
        "marker_names": ["first_token_to_audio", "typed_input_admitted"],
    }
    assert "secret" not in json.dumps(categories)
    assert "task_private" not in json.dumps(categories)


@pytest.mark.skipif(sys.platform != "win32", reason="process start times are read on Windows")
def test_a_process_that_exited_is_not_counted_as_left_running() -> None:
    import subprocess

    child = subprocess.Popen((sys.executable, "--version"), stdout=subprocess.DEVNULL)
    running = rehearsal.started_at(child.pid)
    child.wait()

    assert type(rehearsal.started_at(os.getpid())) is int
    # The handle Popen still holds keeps the exited process object alive; it is not running.
    assert rehearsal.started_at(child.pid) is None
    assert running is None or type(running) is int


def test_a_step_record_keeps_the_first_finding_as_its_category() -> None:
    step = rehearsal.Step("4", "delegate_task", [])
    step.differ("state_sequence")
    step.differ("result_not_reported")
    assert step.record()["outcome"] == "different"
    assert step.record()["category"] == "state_sequence"

    step.fail("private_id_visible")
    record = step.record()
    assert record["outcome"] == "failed"
    assert record["category"] == "private_id_visible"
    assert record["findings"] == ["state_sequence", "result_not_reported", "private_id_visible"]


def test_a_step_records_only_markers_that_appeared_during_it(tmp_path: Path) -> None:
    log = tmp_path / "host.log"
    log.write_text('[voice-tail] {"restored":1}\n', encoding="utf-8")
    tail = rehearsal.LogTail("host", log)
    tail.poll()
    step = rehearsal.Step("8", "restart_host", [tail])
    with log.open("a", encoding="utf-8") as stream:
        stream.write('[hermes-restart-settlement] {"stopped":1}\r\nTraceback (most recent call')
    first = step.record()
    with log.open("a", encoding="utf-8") as stream:
        stream.write(" last):\n")

    record = step.record()

    assert first["markers"] == ['host: [hermes-restart-settlement] {"stopped":1}']
    assert "tracebacks" not in first  # A partial line is not yet a line.
    assert record["markers"] == first["markers"]
    assert record["tracebacks"] == {"known": {}, "unexpected": {"host:unparsed": 1}}
    assert json.loads(json.dumps(record)) == record
