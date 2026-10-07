from __future__ import annotations

import importlib.util
import json
import os
import sys
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


def test_markers_keep_only_bounded_json_objects_in_order() -> None:
    lines = [
        '[voice-tail] {"restored":3,"version":1}',
        "plain output",
        '[voice-tail] not json {',
        '[voice-tail] {not json}',
        '[Bad-Name] {"x":1}',
        '[hermes-restart-settlement] {"stopped":1}',
        '[voice-archive-send] ["list"]',
        "[voice-review] {" + '"x":"' + "a" * 600 + '"}',
        '  [indented] {"x":1}',
    ]

    found, dropped = rehearsal.markers(lines)

    assert found == [
        '[voice-tail] {"restored":3,"version":1}',
        '[hermes-restart-settlement] {"stopped":1}',
    ]
    assert dropped == 0


def test_markers_are_capped_and_the_rest_counted() -> None:
    lines = [f'[voice-archive-send] {{"n":{index}}}' for index in range(40)]

    found, dropped = rehearsal.markers(lines)

    assert found == lines[:32]
    assert dropped == 8


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
    assert record["tracebacks"] == {"host": 1}
    assert json.loads(json.dumps(record)) == record
