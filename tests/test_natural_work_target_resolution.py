from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from hermes_realtime.conversation.context import ActiveTaskSummary
from scripts import natural_work_target_resolution as target

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/natural_work_target_resolution_cases.json"
DEVELOPMENT = ROOT / "tests/fixtures/natural_work_development_cases.json"
# The original development evidence (PR #248) is bound to these exact bytes.
DEVELOPMENT_SHA256 = "92bd45d83ac643fa2d154995eb81a2c3d0ada11fc41c311bb3872293e5365070"


def _document() -> dict[str, Any]:
    value: dict[str, Any] = json.loads(FIXTURE.read_bytes())
    return value


def _write(tmp_path: Path, document: object, name: str = "cases.json") -> tuple[Path, str]:
    raw = json.dumps(document, indent=2).encode()
    path = tmp_path / name
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _expectations(fixture: target.TargetFixture) -> list[tuple[str, str, str | None]]:
    return [(c.case_id, c.outcome, c.target_description) for c in fixture.cases]


def _correct(fixture: target.TargetFixture) -> list[dict[str, object]]:
    return [
        {"id": case_id, "outcome": outcome, "target_description": description}
        for case_id, outcome, description in _expectations(fixture)
    ]


def test_frozen_fixture_is_hash_pinned_and_covers_each_bucket() -> None:
    fixture = target.load_target_resolution(FIXTURE)
    assert fixture.sha256 == target.TARGET_RESOLUTION_SHA256
    assert fixture.sha256 == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert _expectations(fixture) == [
        ("clear_match", "target", "Flight price comparison for the Lisbon trip."),
        ("semantic_mismatch", "no_match", None),
        ("multiple_plausible", "clarify", None),
    ]
    assert [case.bucket for case in fixture.cases] == ["matched", "mismatched", "ambiguous"]


def _markers(err: str) -> list[dict[str, object]]:
    prefix = "[target-resolution] "
    return [json.loads(line[len(prefix):]) for line in err.splitlines() if line.startswith(prefix)]


def test_refusals_emit_one_bounded_marker_and_success_emits_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = target.load_target_resolution(FIXTURE)
    target.score_target_resolution(fixture, _correct(fixture))
    assert capsys.readouterr().err == ""

    with pytest.raises(ValueError, match="target_resolution_binding_refused"):
        target.load_target_resolution(FIXTURE, "0" * 64)
    with pytest.raises(ValueError, match="target_resolution_observation_refused"):
        target.score_target_resolution(fixture, _correct(fixture)[:-1])
    missing = tmp_path / "secret-name-absent.json"
    with pytest.raises(FileNotFoundError):
        target.load_target_resolution(missing, "0" * 64)
    err = capsys.readouterr().err
    assert _markers(err) == [
        {"stage": "load", "refusal": "target_resolution_binding_refused"},
        {"stage": "score", "refusal": "target_resolution_observation_refused"},
        {"stage": "load", "refusal": "FileNotFoundError"},
    ]
    assert len(err.splitlines()) == 3
    assert "secret-name-absent" not in err and str(tmp_path) not in err


def test_visible_active_work_fits_the_production_task_context() -> None:
    for case in target.load_target_resolution(FIXTURE).cases:
        for item in case.active_work:
            assert ActiveTaskSummary(item.task_id, item.description).objective == item.description


def test_tampered_fixture_bytes_fail_the_hash_pin(tmp_path: Path) -> None:
    raw = FIXTURE.read_bytes()
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(raw.replace(b"Please stop the flight", b"Please stop the hotel", 1))
    with pytest.raises(ValueError, match="target_resolution_binding_refused"):
        target.load_target_resolution(tampered)
    trailing = tmp_path / "trailing.json"
    trailing.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="target_resolution_binding_refused"):
        target.load_target_resolution(trailing)
    with pytest.raises(ValueError, match="target_resolution_binding_refused"):
        target.load_target_resolution(FIXTURE, "0" * 64)


@pytest.mark.parametrize("seed", range(8))
def test_renaming_internal_task_ids_changes_no_expectation_or_score(
    tmp_path: Path, seed: int,
) -> None:
    frozen = target.load_target_resolution(FIXTURE)
    rng = random.Random(seed)
    document = _document()
    rows = [item for case in document["cases"] for item in case["active_work"]]
    names = rng.sample(range(10**6), len(rows))
    for item, number in zip(rows, names, strict=True):
        item["task_id"] = f"task_{number:06d}_{rng.choice('abcxyz')}"
    for case in document["cases"]:
        rng.shuffle(case["active_work"])
    path, sha = _write(tmp_path, document)
    renamed = target.load_target_resolution(path, sha)
    assert _expectations(renamed) == _expectations(frozen)
    wrong = _correct(frozen)
    wrong[2] = {"id": "multiple_plausible", "outcome": "target",
                "target_description": frozen.cases[2].active_work[0].description}
    for observations in (_correct(frozen), wrong):
        before = target.score_target_resolution(frozen, observations)["buckets"]
        after = target.score_target_resolution(renamed, observations)["buckets"]
        assert after == before


def test_observation_naming_an_internal_task_id_is_never_a_correct_target() -> None:
    fixture = target.load_target_resolution(FIXTURE)
    observations = _correct(fixture)
    observations[0]["target_description"] = fixture.cases[0].active_work[0].task_id
    report = target.score_target_resolution(fixture, observations)
    assert report["buckets"]["matched"] == {
        "cases": 1, "correct": 0, "incorrect_case_ids": ["clear_match"],
    }


def _case(document: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(row for row in document["cases"] if row["id"] == case_id)


def _refused(tmp_path: Path, document: dict[str, Any]) -> None:
    path, sha = _write(tmp_path, document)
    with pytest.raises(ValueError, match="target_resolution_case_refused"):
        target.load_target_resolution(path, sha)


# Nominal author expectations the visible descriptions do not support: (case, outcome,
# index of the active-work description named as target, or None).
@pytest.mark.parametrize(("case_id", "outcome", "index"), [
    pytest.param("clear_match", "no_match", None, id="clear_relabelled_no_match"),
    pytest.param("semantic_mismatch", "target", 0, id="mismatch_claims_target"),
    pytest.param("multiple_plausible", "target", 0, id="ambiguous_claims_target"),
    pytest.param("clear_match", "target", 1, id="match_picks_wrong_description"),
])
def test_declared_expectation_not_following_from_visible_text_is_rejected(
    tmp_path: Path, case_id: str, outcome: str, index: int | None,
) -> None:
    document = _document()
    case = _case(document, case_id)
    case["expected"] = {"outcome": outcome, "target_description": (
        None if index is None else case["active_work"][index]["description"]
    )}
    _refused(tmp_path, document)


@pytest.mark.parametrize("change", [
    pytest.param(lambda case: case["active_work"][1].update(
        description=case["active_work"][0]["description"]), id="target_only_distinguishable_by_id"),
    pytest.param(lambda case: case.update(utterance="Please stop that comparison."),
                 id="hinge_absent_from_utterance"),
    pytest.param(lambda case: case.update(utterance=(
        f"Please stop {case['active_work'][0]['task_id']}, the flight price comparison.")),
        id="task_id_visible_in_utterance"),
    pytest.param(lambda case: case["active_work"][0].update(label="unrelated"),
                 id="author_label_field"),
])
def test_visible_content_that_cannot_decide_the_case_is_rejected(
    tmp_path: Path, change: Any,
) -> None:
    document = _document()
    change(_case(document, "clear_match"))
    _refused(tmp_path, document)


def test_unchanged_copy_with_its_own_pin_is_admitted(tmp_path: Path) -> None:
    path, sha = _write(tmp_path, _document())
    assert _expectations(target.load_target_resolution(path, sha)) == _expectations(
        target.load_target_resolution(FIXTURE)
    )


def test_buckets_are_separate_and_carry_no_cancellation_or_lifecycle_scoring() -> None:
    fixture = target.load_target_resolution(FIXTURE)
    report = target.score_target_resolution(fixture, _correct(fixture))
    assert set(report) == {"fixture_id", "fixture_sha256", "buckets"}
    assert report["fixture_sha256"] == target.TARGET_RESOLUTION_SHA256
    assert report["buckets"] == {
        "matched": {"cases": 1, "correct": 1, "incorrect_case_ids": []},
        "mismatched": {"cases": 1, "correct": 1, "incorrect_case_ids": []},
        "ambiguous": {"cases": 1, "correct": 1, "incorrect_case_ids": []},
    }
    wrong = _correct(fixture)
    wrong[2] = {"id": "multiple_plausible", "outcome": "no_match", "target_description": None}
    buckets = target.score_target_resolution(fixture, wrong)["buckets"]
    assert buckets["ambiguous"] == {
        "cases": 1, "correct": 0, "incorrect_case_ids": ["multiple_plausible"],
    }
    assert buckets["matched"]["correct"] == buckets["mismatched"]["correct"] == 1


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(cancellation_effect=False),
    lambda rows: rows[0].update(unsupported_lifecycle_claim=False),
    lambda rows: rows[0].update(outcome="cancelled"),
    lambda rows: rows[1].update(target_description="Weekly grocery budget spreadsheet update."),
])
def test_scorer_refuses_incomplete_or_foreign_observations(change: Any) -> None:
    fixture = target.load_target_resolution(FIXTURE)
    observations = _correct(fixture)
    change(observations)
    with pytest.raises(ValueError, match="target_resolution_observation_refused"):
        target.score_target_resolution(fixture, observations)


def test_development_fixture_and_results_stay_unchanged(tmp_path: Path) -> None:
    from scripts import evaluate_natural_work_scenario as scenario

    before = DEVELOPMENT.read_bytes()
    assert hashlib.sha256(before).hexdigest() == DEVELOPMENT_SHA256
    assert scenario.DEVELOPMENT_SHA256 == DEVELOPMENT_SHA256
    development = scenario.load_development(DEVELOPMENT)
    assert {
        case.case_id: development.expected_decision(case) for case in development.cases
    } == {
        "research": "start_work", "greeting": "reply", "explanation": "reply",
        "quotation": "reply", "negation": "reply", "cancel_model_authority": "task_status",
        "approval_model_authority": "reply", "ambiguity": "reply", "host_status": "task_status",
    }
    # The target-resolution set is separately frozen: neither loader admits the other.
    with pytest.raises(ValueError, match="target_resolution_binding_refused"):
        target.load_target_resolution(DEVELOPMENT)
    with pytest.raises(ValueError, match="input_binding"):
        scenario.load_development(FIXTURE)
    assert DEVELOPMENT.read_bytes() == before
