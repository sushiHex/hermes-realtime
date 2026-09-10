"""A packaged producer must derive revocation evidence and reject supplied success."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import zipfile

import pytest


def _wheel_bytes(change: str = "") -> tuple[bytes, dict[str, tuple[int, str]]]:
    package = {"hermes_realtime/__init__.py": b"", "hermes_realtime/runtime.py": b"candidate = 1\n"}
    expected = {name: (len(raw), hashlib.sha256(raw).hexdigest()) for name, raw in package.items()}
    info = "hermes_realtime-0.0.3.dist-info/"
    members = package | {
        info + "METADATA": b"Metadata-Version: 2.4\nName: hermes-realtime\nVersion: 0.0.3\n",
        info + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        info + "entry_points.txt": (
            b"[console_scripts]\nhermes-realtime-host = hermes_realtime.host_launcher:main\n"
            b"hermes-realtime-local = hermes_realtime.launcher:main\n"
            b"[hermes_agent.plugins]\nhermes-realtime = hermes_realtime.hermes_plugin\n"
        ),
        info + "licenses/LICENSE": b"synthetic license\n",
    }
    if change == "entrypoint":
        members[info + "entry_points.txt"] = b"[console_scripts]\nhost = unexpected:main\n"
    elif change == "entrypoint_case":
        members[info + "entry_points.txt"] = members[info + "entry_points.txt"].replace(
            b"hermes-realtime-host =", b"Hermes-realtime-host ="
        )
    elif change == "source":
        members["hermes_realtime/runtime.py"] = b"candidate = 2\n"
    elif change == "extra":
        members["startup.pth"] = b"import private_configuration\n"
    elif change == "missing":
        del members["hermes_realtime/runtime.py"]
    elif change == "path":
        members["../outside"] = b"escape"
    elif change == "native":
        members[info + "WHEEL"] = (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp311-win_amd64\n"
        )
    elif change == "identity":
        members[info + "METADATA"] = b"Name: other-project\nVersion: 0.0.3\n"
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, raw in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(raw)])
    writer.writerow([info + "RECORD", "", ""])
    members[info + "RECORD"] = record.getvalue().encode()
    if change == "record":
        members[info + "RECORD"] = b"invalid record\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as wheel:
        for name, raw in members.items():
            wheel.writestr(name, raw)
        if change == "duplicate":
            with pytest.warns(UserWarning):
                wheel.writestr("hermes_realtime/runtime.py", members["hermes_realtime/runtime.py"])
    return output.getvalue(), expected


def test_wheel_inspection_binds_every_runtime_blob_and_closed_metadata() -> None:
    from scripts.candidate_wheel import _inspect_wheel

    raw, expected = _wheel_bytes()
    members = _inspect_wheel(raw, expected)
    assert {name for name in members if name.startswith("hermes_realtime/")} == set(expected)


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "extra",
        "missing",
        "path",
        "native",
        "identity",
        "record",
        "duplicate",
        "entrypoint",
        "entrypoint_case",
    ],
)
def test_wheel_inspection_refuses_foreign_incomplete_or_executable_extra_members(
    change: str,
) -> None:
    from scripts.candidate_wheel import _inspect_wheel

    raw, expected = _wheel_bytes(change)
    with pytest.raises(ValueError):
        _inspect_wheel(raw, expected)


def test_revoke_receipts_cannot_be_constructed_from_caller_assertions() -> None:
    from scripts.revoke_race import ObservedRevokeRaceV1, validate_revoke_race_v1

    with pytest.raises(TypeError):
        ObservedRevokeRaceV1()
    with pytest.raises((TypeError, ValueError)):
        validate_revoke_race_v1({"passed": True})
    with pytest.raises((TypeError, ValueError)):
        validate_revoke_race_v1(object.__new__(ObservedRevokeRaceV1))


def _observations() -> dict:
    from scripts.deterministic_equivalence import _expected_close

    pending = dict(
        events=8,
        sessions=1,
        epochs=1,
        requests=1,
        pending_revocations=1,
        revoked_epochs=1,
        tombstones=0,
        erased_events=0,
        erased_sessions=0,
        last_ordinal=12,
        final_ordinal=0,
        integrity_ok=True,
        scope_matches=True,
        authority="a" * 64,
    )
    purged = dict(
        events=0,
        sessions=0,
        epochs=0,
        requests=0,
        pending_revocations=0,
        revoked_epochs=0,
        tombstones=1,
        erased_events=8,
        erased_sessions=1,
        last_ordinal=12,
        final_ordinal=12,
        integrity_ok=True,
        scope_matches=True,
        authority="a" * 64,
    )
    return dict(
        arm="revoke_race",
        snapshots=[pending, dict(pending), purged],
        acknowledgment="durable",
        completed_turns=2,
        capture_terminals=1,
        admitted_before=4,
        admitted_after=4,
        revoke_accepted=1,
        revoke_terminal=1,
        all_capacity_released=True,
        trace_complete=True,
        host_return="returned",
        close=_expected_close("consented"),
    )


def test_revoke_validator_derives_durability_admission_closure_and_verified_purge() -> None:
    from scripts.revoke_race import _validate_observations

    _validate_observations(_observations())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_snapshot",
        "reordered",
        "empty_before",
        "not_durable",
        "new_admission",
        "new_record",
        "purge_incomplete",
        "purge_count",
        "false_integrity",
        "bool_count",
        "missing_close",
        "failed_close",
        "incomplete",
        "extra",
        "conversation_stopped",
        "wrong_authority",
        "wrong_scope",
    ],
)
def test_revoke_validator_rejects_missing_or_contradictory_evidence(mutation: str) -> None:
    from scripts.revoke_race import _validate_observations

    row = _observations()
    if mutation == "missing_snapshot":
        row["snapshots"].pop()
    elif mutation == "reordered":
        row["snapshots"].reverse()
    elif mutation == "empty_before":
        row["snapshots"][0]["events"] = 0
    elif mutation == "not_durable":
        row["snapshots"][0]["pending_revocations"] = 0
    elif mutation == "new_admission":
        row["admitted_after"] += 1
    elif mutation == "new_record":
        row["snapshots"][1]["events"] += 1
    elif mutation == "purge_incomplete":
        row["snapshots"][2]["events"] = 1
    elif mutation == "purge_count":
        row["snapshots"][2]["erased_events"] -= 1
    elif mutation == "false_integrity":
        row["snapshots"][2]["integrity_ok"] = False
    elif mutation == "bool_count":
        row["snapshots"][0]["requests"] = True
    elif mutation == "missing_close":
        row["close"].pop()
    elif mutation == "failed_close":
        row["close"][0]["result"] = "failed"
    elif mutation == "incomplete":
        row["trace_complete"] = False
    elif mutation == "extra":
        row["private_output"] = "synthetic marker"
    elif mutation == "conversation_stopped":
        row["completed_turns"] = 1
    elif mutation == "wrong_authority":
        row["snapshots"][2]["authority"] = "b" * 64
    elif mutation == "wrong_scope":
        row["snapshots"][2]["scope_matches"] = False
    with pytest.raises(ValueError):
        _validate_observations(row)


def test_only_the_canonical_revoke_registration_can_invoke_the_packaged_producer() -> None:
    from pathlib import Path

    from scripts import qualify_evidence_slice_zero as core

    registry = core.SCENARIO_REGISTRY_V1
    assert tuple(item.scenario_id for item in registry) == tuple(core.ScenarioIdV1)
    assert type(registry[10]) is core.RevokeRaceRegistrationV1
    assert len(core.UNAVAILABLE_SCENARIO_REGISTRY_V1) == 18
    copied = core.RevokeRaceRegistrationV1()
    with pytest.raises(ValueError, match="not canonical"):
        copied.produce(
            object(),
            object(),
            object(),
            livekit_executable=Path("unusable.exe"),
            livekit_sha256="a" * 64,
        )


def test_native_gate_uses_the_existing_candidate_wheel_for_revoke_qualification() -> None:
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/release-gates.yml"
    ).read_text(encoding="utf-8")
    native = workflow.split("  native-livekit:", 1)[1]
    assert "needs: candidate-wheel" in native
    assert "name: hermes-realtime-pure-candidate-wheel" in native
    assert "scripts.qualify_revoke_race" in native
    assert "--candidate-wheel" in native and "--wheel-sha256" in native
    assert native.index("Qualify packaged revocation race") < native.index("Run real-browser")
