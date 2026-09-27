from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qualify_voice_archive_path.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))  # The script imports its sibling gate support.
_SPEC = importlib.util.spec_from_file_location("qualify_voice_archive_path", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SCRIPT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SCRIPT
_SPEC.loader.exec_module(_SCRIPT)

_BASELINE: dict[str, object] = {"version": "0.21.0", "commit": "29112bef", "baseline": True}

# Closed rows 0-6: 0-1 discarded (gap on user row 2), 3-4 archived, 5-6 a trailing gap.
_TRUTH: list[list[Any]] = [
    ["user", "q0", False],
    ["assistant", "a1", False],
    ["user", "q2", False],
    ["assistant", "a3", True],
    ["user", "q4", False],
    ["user", "q5", False],
    ["assistant", "a6", False],
]


def _row(seq: int, gap: list[int] | None = None) -> dict[str, Any]:
    role, text, interrupted = _TRUTH[seq]
    return {"seq": seq, "role": role, "text": text, "interrupted": interrupted,
            "ts": 100.0 + seq, "gap_before": gap}


def _event(*rows: dict[str, Any]) -> dict[str, Any]:
    start = rows[0]["gap_before"][0] if rows[0]["gap_before"] else rows[0]["seq"]
    return {"protocol_version": "0.2", "type": "voice_archive", "conversation_id": "qualify",
            "generation": 0, "seq_from": start, "seq_through": rows[-1]["seq"],
            "rows": list(rows)}


def _scenario() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    first = _event(_row(2, [0, 1]), _row(3))
    second = _event(_row(4))
    sent = [
        {"event": first, "outcome": "lost"},
        {"event": copy.deepcopy(first), "outcome": "voice_archive_ack"},
        {"event": second, "outcome": "voice_archive_ack"},
    ]
    archived = [_row(2, [0, 1]), _row(3), _row(4)]
    tail = {"gap": [5, 6], "outbox": []}
    return sent, archived, tail


def test_a_faithful_archive_reports_no_fault() -> None:
    sent, archived, tail = _scenario()

    assert _SCRIPT._fidelity(_TRUTH, sent, archived, tail) == {
        "archived": 3, "archived_differs_from_frozen": 0, "discarded": 4,
        "double_gapped": 0, "gaps": 1, "lost": 0, "mismatched": 0, "outbox_left": 0,
        "overlapping": 0, "phantom": 0, "trailing": 2,
    }
    assert _SCRIPT._sends(sent) == {
        "resends": 1, "resends_changed": 0, "resends_not_after_unknown": 0
    }


@pytest.mark.parametrize(
    ("mutate", "fault"),
    [
        (lambda s, a, t: a[1].update(text="other"), "mismatched"),
        (lambda s, a, t: a[1].update(interrupted=False), "mismatched"),
        (lambda s, a, t: a[1].update(ts=0.5), "archived_differs_from_frozen"),
        (lambda s, a, t: a.pop(), "lost"),
        (lambda s, a, t: a[0].update(gap_before=[1, 1]), "lost"),
        (lambda s, a, t: t.update(gap=None), "lost"),
        (lambda s, a, t: a[2].update(gap_before=[0, 3]), "double_gapped"),
        (lambda s, a, t: a[2].update(gap_before=[3, 3]), "overlapping"),
        (lambda s, a, t: t.update(outbox=[{}]), "outbox_left"),
        (lambda s, a, t: t.update(gap=[5, 7]), "phantom"),
    ],
)
def test_every_fidelity_fault_is_counted(mutate: Any, fault: str) -> None:
    sent, archived, tail = _scenario()
    mutate(sent, archived, tail)

    assert _SCRIPT._fidelity(_TRUTH, sent, archived, tail)[fault] > 0


def test_a_changed_or_unprompted_resend_is_counted() -> None:
    sent, _, _ = _scenario()
    changed = copy.deepcopy(sent)
    changed[1]["event"]["rows"][0]["ts"] = 1.0
    after_ack = copy.deepcopy(sent) + [copy.deepcopy(sent[2])]

    assert _SCRIPT._sends(changed)["resends_changed"] == 1
    assert _SCRIPT._sends(after_ack)["resends_not_after_unknown"] == 1


def _evidence() -> dict[str, Any]:
    sent, archived, tail = _scenario()
    fidelity = _SCRIPT._fidelity(_TRUTH, sent, archived, tail) | {"gaps": 2}
    return {
        "fidelity": fidelity,
        "sends": _SCRIPT._sends(sent),
        "frozen_resent_unchanged": True,
        "partition": {"category": "partition", "mutations": 0},
        "reads": {"bridge_connections": 3, "foreign_connections": 0, "http_requests": 0,
                  "messages_route_reads": 0},
        "steps": dict(_SCRIPT._EXPECTED_STEPS),
    }


def test_the_expected_evidence_passes_only_on_the_baseline() -> None:
    assert _SCRIPT._passed(_evidence(), _BASELINE) is True
    assert _SCRIPT._passed(_evidence(), _BASELINE | {"baseline": False}) is False
    assert _SCRIPT._passed(_evidence(), {}) is False


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("fidelity", "lost", 1),
        ("fidelity", "mismatched", 1),
        ("fidelity", "double_gapped", 1),
        ("fidelity", "overlapping", 1),
        ("fidelity", "archived_differs_from_frozen", 1),
        ("fidelity", "outbox_left", 1),
        ("fidelity", "phantom", 1),
        ("fidelity", "gaps", 1),
        ("fidelity", "trailing", 0),
        ("fidelity", "archived", 0),
        ("sends", "resends", 0),
        ("sends", "resends_changed", 1),
        ("sends", "resends_not_after_unknown", 1),
        ("reads", "bridge_connections", 0),
        ("reads", "foreign_connections", 1),
        ("reads", "http_requests", 1),
        ("reads", "messages_route_reads", 1),
        ("partition", "mutations", 1),
        ("partition", "category", "accepted"),
        ("steps", "drain", 1),
    ],
)
def test_every_guard_can_fail_the_qualification(section: str, key: str, value: object) -> None:
    evidence = _evidence()
    evidence[section][key] = value

    assert _SCRIPT._passed(evidence, _BASELINE) is False


def test_an_unfrozen_first_send_fails_the_qualification() -> None:
    evidence = _evidence() | {"frozen_resent_unchanged": False}

    assert _SCRIPT._passed(evidence, _BASELINE) is False
