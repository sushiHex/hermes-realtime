import json
from types import SimpleNamespace

import pytest

from scripts.lookup_backend_logs import CappedCapture, CappedProcesses


def marker(**changes):
    value = {
        "version": 1, "stage": "producer", "failure_kind": "runtime",
        "text_phase": "before_text",
    }
    value.update(changes)
    return b"[foreground-failure] " + json.dumps(value).encode() + b"\n"


def test_fixed_foreground_failure_survives_split_drain_and_finish(tmp_path):
    capture = CappedCapture(tmp_path / "child.private.log")
    raw = marker()
    capture.feed(raw[:19])
    capture.feed(raw[19:])
    facts = capture.finish()
    event = facts["categories"]["foreground_producer_runtime_before_text"]
    assert event["count"] == 1
    assert event["first_ns"] <= event["last_ns"]
    assert "foreground-failure" not in json.dumps(facts)


def test_refused_foreground_failure_is_counted_not_invalid(tmp_path):
    capture = CappedCapture(tmp_path / "child.private.log")
    capture.feed(marker(failure_kind="refused", text_phase="after_text"))
    facts = capture.finish()
    assert set(facts["categories"]) == {"foreground_producer_refused_after_text"}
    assert facts["categories"]["foreground_producer_refused_after_text"]["count"] == 1


def test_unterminated_final_marker_is_counted_exactly_once(tmp_path):
    capture = CappedCapture(tmp_path / "child.private.log")
    capture.feed(marker().rstrip(b"\n"))
    first = capture.finish()
    assert first["categories"]["foreground_producer_runtime_before_text"]["count"] == 1
    assert capture.finish()["categories"] == first["categories"]


@pytest.mark.parametrize("changes", [
    {"failure_kind": "SYNTHETIC_PRIVATE_EXCEPTION https://private.invalid/token"},
    {"stage": "private-stage"}, {"text_phase": "private-phase"},
    {"version": True}, {"version": 2},
    {"message": "SYNTHETIC_PRIVATE_EXCEPTION"},
])
def test_foreground_failure_metadata_rejects_arbitrary_fields_and_values(tmp_path, changes):
    capture = CappedCapture(tmp_path / "child.private.log")
    capture.feed(marker(**changes))
    facts = capture.finish()
    assert set(facts["categories"]) == {"foreground_failure_invalid"}
    assert facts["categories"]["foreground_failure_invalid"]["count"] == 1
    assert "SYNTHETIC_PRIVATE_EXCEPTION" not in json.dumps(facts)
    assert "private.invalid" not in json.dumps(facts)


def test_foreground_failure_duplicate_keys_are_not_reinterpreted(tmp_path):
    capture = CappedCapture(tmp_path / "child.private.log")
    raw = marker().replace(b'"version": 1', b'"version": 1, "version": 1')
    capture.feed(raw)
    assert set(capture.finish()["categories"]) == {"foreground_failure_invalid"}


def test_normal_owned_stop_keeps_counts_after_raw_output_is_removed(tmp_path):
    stopped = []
    owner = SimpleNamespace(roots={}, stop_all=lambda containment: stopped.append(containment))
    collector = CappedProcesses(owner, tmp_path)
    capture = CappedCapture(tmp_path / "host.private.log")
    collector.captures["host"] = capture
    capture.feed(marker())
    collector.stop_all("synthetic-containment")
    metadata = collector.finish()
    (tmp_path / "host.private.log").unlink()
    assert stopped == ["synthetic-containment"]
    assert metadata["host"]["categories"]["foreground_producer_runtime_before_text"]["count"] == 1
    assert not (tmp_path / "host.private.log").exists()


def test_oversized_or_unmarked_private_output_is_not_retained_in_metadata(tmp_path):
    capture = CappedCapture(tmp_path / "child.private.log")
    capture.feed(b"SYNTHETIC_PRIVATE_EXCEPTION https://private.invalid/token\n")
    capture.feed(b"[foreground-failure] " + b"x" * 8192 + b"\n")
    capture.feed(b"shutting down\n")
    facts = capture.finish()
    assert facts["line_overflow"]
    assert set(facts["categories"]) == {"shutdown"}
    assert "SYNTHETIC_PRIVATE_EXCEPTION" not in json.dumps(facts)
    assert "private.invalid" not in json.dumps(facts)


def test_foreground_category_is_observed_after_raw_storage_cap(tmp_path):
    capture = CappedCapture(tmp_path / "child.private.log", limit=1)
    capture.feed(b"x\n")
    capture.feed(marker())
    facts = capture.finish()
    assert facts["stored_bytes"] == 1 and facts["overflow"]
    assert facts["categories"]["foreground_producer_runtime_before_text"]["count"] == 1
