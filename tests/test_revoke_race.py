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
    elif change == "metadata_missing_version":
        members[info + "METADATA"] = b"Name: hermes-realtime\nVersion: 0.0.3\n"
    elif change == "metadata_unsupported_version":
        members[info + "METADATA"] = members[info + "METADATA"].replace(b"2.4", b"99.0")
    elif change == "metadata_duplicate_version":
        members[info + "METADATA"] += b"Metadata-Version: 2.4\n"
    elif change in {"metadata_defect", "wheel_defect"}:
        name = "METADATA" if change == "metadata_defect" else "WHEEL"
        members[info + name] += b"malformed header without a separator\n"
    elif change in {"metadata_encoding", "wheel_encoding"}:
        name = "METADATA" if change == "metadata_encoding" else "WHEEL"
        members[info + name] += b"\n\n\xff"
    elif change in {"metadata_unixfrom", "wheel_unixfrom"}:
        name = "METADATA" if change == "metadata_unixfrom" else "WHEEL"
        members[info + name] = b"From synthetic envelope\n" + members[info + name]
    elif change in {"metadata_multipart", "wheel_multipart"}:
        name = "METADATA" if change == "metadata_multipart" else "WHEEL"
        members[info + name] += (
            b'Content-Type: multipart/mixed; boundary="synthetic"\n'
            b'\n--synthetic\nContent-Type: text/plain\n\nbody\n--synthetic--\n'
        )
    elif change == "wheel_whitespace_body":
        members[info + "WHEEL"] += b"\n \t\r\n"
    elif change == "wheel_body":
        members[info + "WHEEL"] += b"\nunsupported wheel body\n"
    elif change in {"metadata_header_name", "wheel_header_name"}:
        name = "METADATA" if change == "metadata_header_name" else "WHEEL"
        members[info + name] += b"Invalid\x00Header: value\n"
    elif change == "metadata_requirement":
        members[info + "METADATA"] += b"Requires-Dist: !!!invalid!!!\n"
    elif change == "metadata_python":
        members[info + "METADATA"] += b"Requires-Python: unsupported-version-syntax\n"
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
        "metadata_missing_version",
        "metadata_unsupported_version",
        "metadata_duplicate_version",
        "metadata_defect",
        "wheel_defect",
        "metadata_encoding",
        "wheel_encoding",
        "metadata_unixfrom",
        "wheel_unixfrom",
        "metadata_multipart",
        "wheel_multipart",
        "wheel_body",
        "wheel_whitespace_body",
        "metadata_header_name",
        "wheel_header_name",
        "metadata_requirement",
        "metadata_python",
    ],
)
def test_wheel_inspection_refuses_foreign_incomplete_or_executable_extra_members(
    change: str,
) -> None:
    from scripts.candidate_wheel import _inspect_wheel

    raw, expected = _wheel_bytes(change)
    with pytest.raises(ValueError):
        _inspect_wheel(raw, expected)


_SOURCE_PROJECT = b'''[build-system]
requires = ["hatchling==1.27.0"]
build-backend = "hatchling.build"
[project]
name = "hermes-realtime"
version = "0.0.3"
description = "Synthetic package"
readme = "README.md"
requires-python = ">=3.11,<3.12"
authors = [{name = "Synthetic author"}]
license = "MIT"
license-files = ["LICENSE"]
keywords = ["voice", "realtime"]
classifiers = ["Typing :: Typed", "Programming Language :: Python :: 3"]
dependencies = ["aiohttp>=3.11,<4", "pydantic>=2.11,<3"]
[project.optional-dependencies]
local = ["numpy>=1.26,<3; sys_platform == 'win32'"]
[project.urls]
Homepage = "https://example.org/"
[project.scripts]
hermes-realtime-host = "hermes_realtime.host_launcher:main"
hermes-realtime-local = "hermes_realtime.launcher:main"
[project.entry-points."hermes_agent.plugins"]
hermes-realtime = "hermes_realtime.hermes_plugin"
'''
_SOURCE_CORE = b'''Metadata-Version: 2.4
Name: hermes-realtime
Version: 0.0.3
Summary: Synthetic package
Description-Content-Type: text/markdown
Requires-Python: <3.12,>=3.11
Author: Synthetic author
License-Expression: MIT
License-File: LICENSE
Keywords: realtime,voice
Classifier: Programming Language :: Python :: 3
Classifier: Typing :: Typed
Project-URL: Homepage, https://example.org/
Provides-Extra: local
Requires-Dist: pydantic<3,>=2.11
Requires-Dist: aiohttp<4,>=3.11
Requires-Dist: numpy<3,>=1.26; sys_platform == "win32" and extra == "local"

# Synthetic description
'''


def _source_members() -> dict[str, bytes]:
    from scripts.candidate_wheel import _inspect_wheel

    members = _inspect_wheel(*_wheel_bytes())
    info = "hermes_realtime-0.0.3.dist-info/"
    members[info + "METADATA"] = _SOURCE_CORE
    members[info + "WHEEL"] += b"Generator: hatchling 1.27.0\n"
    return members


def test_wheel_metadata_matches_source_despite_header_and_requirement_order() -> None:
    from scripts.candidate_wheel import _bind_source_metadata

    members = _source_members()
    info = "hermes_realtime-0.0.3.dist-info/"
    for reorder in (False, True):
        if reorder:
            headers, body = _SOURCE_CORE.split(b"\n\n", 1)
            members[info + "METADATA"] = b"\n".join(reversed(headers.splitlines())) + b"\n\n" + body
            members[info + "METADATA"] = members[info + "METADATA"].replace(
                b"aiohttp<4,>=3.11", b"AioHTTP>=3.11,<4"
            )
        _bind_source_metadata(members, _SOURCE_PROJECT, b"# Synthetic description\n")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b"Requires-Dist: pydantic<3,>=2.11\n", b""),
        (b"Requires-Dist: pydantic<3,>=2.11", b"Requires-Dist: unexpected>=1"),
        (
            b"Requires-Dist: pydantic<3,>=2.11",
            b"Requires-Dist: pydantic<3,>=2.11\nRequires-Dist: unexpected>=1",
        ),
        (
            b"Requires-Dist: pydantic<3,>=2.11",
            b"Requires-Dist: pydantic<3,>=2.11\nRequires-Dist: pydantic<3,>=2.11",
        ),
        (b"sys_platform == \"win32\" and extra == \"local\"", b"extra == \"local\""),
        (b"Provides-Extra: local", b"Provides-Extra: other"),
        (b"Requires-Python: <3.12,>=3.11", b"Requires-Python: >=3.11"),
        (b"Summary: Synthetic package", b"Summary: Other package"),
        (b"Author: Synthetic author", b"Author: Other author"),
        (b"License-Expression: MIT", b"License-Expression: BSD-3-Clause"),
        (b"License-File: LICENSE", b"License-File: OTHER"),
        (b"Keywords: realtime,voice", b"Keywords: realtime"),
        (b"Classifier: Typing :: Typed\n", b""),
        (b"https://example.org/", b"https://example.net/"),
        (b"# Synthetic description", b"# Other description"),
        (b"Description-Content-Type: text/markdown", b"Description-Content-Type: text/plain"),
        (b"Name: hermes-realtime", b"Name: hermes-realtime\nPlatform: any"),
    ],
)
def test_valid_but_foreign_core_metadata_is_rejected(old: bytes, new: bytes) -> None:
    from packaging.metadata import Metadata

    from scripts.candidate_wheel import _bind_source_metadata

    members = _source_members()
    name = "hermes_realtime-0.0.3.dist-info/METADATA"
    assert old in members[name]
    members[name] = members[name].replace(old, new)
    # Each mutant is valid packaging metadata; provenance, not syntax, must reject it.
    Metadata.from_email(members[name], validate=True)
    with pytest.raises(ValueError, match="static source profile"):
        _bind_source_metadata(members, _SOURCE_PROJECT, b"# Synthetic description\n")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b"aiohttp>=3.11,<4", b"aiohttp>=3.12,<4"),
        (b"sys_platform == 'win32'", b"sys_platform == 'linux'"),
        (b"host_launcher:main", b"host_launcher:other"),
        (b"[project]", b'[project]\ndynamic = ["version"]'),
        (b"hatchling==1.27.0", b"hatchling==1.28.0"),
        (b'readme = "README.md"', b'readme = "OTHER.md"'),
        (b'name = "Synthetic author"', b'name = "Synthetic author", email = "author@example.org"'),
    ],
)
def test_source_changes_require_matching_wheel_metadata(old: bytes, new: bytes) -> None:
    from scripts.candidate_wheel import _bind_source_metadata

    assert old in _SOURCE_PROJECT
    with pytest.raises(ValueError, match="static source profile"):
        _bind_source_metadata(
            _source_members(), _SOURCE_PROJECT.replace(old, new), b"# Synthetic description\n"
        )


@pytest.mark.parametrize("addition", [b"Build: 1\n", b"Generator: hatchling 1.27.0\n"])
def test_wheel_build_metadata_is_closed_and_matches_the_source_pin(addition: bytes) -> None:
    from scripts.candidate_wheel import _bind_source_metadata

    members = _source_members()
    members["hermes_realtime-0.0.3.dist-info/WHEEL"] += addition
    with pytest.raises(ValueError, match="static source profile"):
        _bind_source_metadata(members, _SOURCE_PROJECT, b"# Synthetic description\n")


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
    assert len(core.UNAVAILABLE_SCENARIO_REGISTRY_V1) == 17
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
