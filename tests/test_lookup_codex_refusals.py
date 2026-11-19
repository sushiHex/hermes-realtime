"""The existing Codex refusal marker survives bounded, privacy-safe normal stop."""

import json
from types import SimpleNamespace

import pytest

from scripts import lookup_backend_logs
from scripts.lookup_backend_logs import CappedCapture, CappedProcesses


def marker(**changes):
    value = {"version": 1, "refusal": "decision_schema"}
    value.update(changes)
    return b"[codex-output-refusal] " + json.dumps(value).encode() + b"\n"


def test_codex_refusal_reader_categories_equal_the_authoritative_contract():
    from hermes_realtime.conversation.streaming import MODEL_OUTPUT_REFUSAL_CATEGORIES

    assert lookup_backend_logs._CODEX_OUTPUT_REFUSALS == MODEL_OUTPUT_REFUSAL_CATEGORIES


def test_codex_refusal_survives_fragmented_normal_stop_and_raw_output_removal(tmp_path):
    stopped = []
    owner = SimpleNamespace(roots={}, stop_all=lambda containment: stopped.append(containment))
    collector = CappedProcesses(owner, tmp_path)
    path = tmp_path / "host.private.log"
    capture = CappedCapture(path)
    collector.captures["host"] = capture
    raw = marker()
    capture.feed(raw[:17])
    capture.feed(raw[17:])
    collector.stop_all("synthetic-containment")
    facts = collector.finish()["host"]
    path.unlink()
    assert stopped == ["synthetic-containment"]
    assert set(facts["categories"]) == {"codex_output_refusal_decision_schema"}
    item = facts["categories"]["codex_output_refusal_decision_schema"]
    assert item["count"] == 1 and item["first_ns"] <= item["last_ns"]
    assert not path.exists()


def test_codex_refusal_category_survives_raw_storage_cap_and_count_saturates(tmp_path):
    capture = CappedCapture(tmp_path / "child.private.log", limit=1)
    capture.feed(b"x\n")
    capture.categories["codex_output_refusal_decision_schema"] = {
        "count": 999999, "first_ns": 1, "last_ns": 1,
    }
    capture.feed(marker() + marker())
    facts = capture.finish()
    assert facts["stored_bytes"] == 1 and facts["overflow"]
    assert set(facts["categories"]) == {"codex_output_refusal_decision_schema"}
    assert facts["categories"]["codex_output_refusal_decision_schema"]["count"] == 1000000


@pytest.mark.parametrize("changes", [
    {"refusal": "PRIVATE https://private.invalid/?token=SECRET out of memory"},
    {"refusal": None}, {"refusal": []}, {"refusal": {}}, {"refusal": True}, {"refusal": 1},
    {"version": True}, {"version": 2}, {"version": "1"},
    {"text": "PRIVATE transcript"}, {"stage": "PRIVATE"},
])
def test_codex_refusal_rejects_extra_fields_versions_and_private_or_nonstring_values(
    tmp_path, changes,
):
    capture = CappedCapture(tmp_path / "child.private.log")
    capture.feed(marker(**changes))
    facts = capture.finish()
    assert set(facts["categories"]) == {"codex_output_refusal_invalid"}
    assert facts["categories"]["codex_output_refusal_invalid"]["count"] == 1
    metadata = json.dumps(facts)
    assert "PRIVATE" not in metadata and "SECRET" not in metadata
    assert "private.invalid" not in metadata


@pytest.mark.parametrize("raw", [
    b'{"version":1,"refusal":"decision_schema","refusal":"PRIVATE"}',
    b'{"version":1,"version":1,"refusal":"decision_schema"}',
    b'{"version":1}', b'{"refusal":"decision_schema"}', b'[]', b'"PRIVATE"', b'{',
])
def test_codex_refusal_rejects_duplicate_missing_fields_and_malformed_objects(tmp_path, raw):
    capture = CappedCapture(tmp_path / "child.private.log")
    capture.feed(b"[codex-output-refusal] " + raw + b"\n")
    facts = capture.finish()
    assert set(facts["categories"]) == {"codex_output_refusal_invalid"}
    assert "PRIVATE" not in json.dumps(facts)


def test_codex_refusal_final_line_is_counted_once_and_oversized_line_stays_bounded(tmp_path):
    capture = CappedCapture(tmp_path / "child.private.log")
    capture.feed(b"[codex-output-refusal] " + b"PRIVATE" * 2000 + b"\n")
    capture.feed(marker().rstrip(b"\n"))
    first = capture.finish()
    assert first["line_overflow"]
    assert set(first["categories"]) == {"codex_output_refusal_decision_schema"}
    assert first["categories"]["codex_output_refusal_decision_schema"]["count"] == 1
    assert capture.finish()["categories"] == first["categories"]
    assert "PRIVATE" not in json.dumps(first)
