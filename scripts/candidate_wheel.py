"""Bind a bounded pure wheel's runtime bytes to a verified source archive."""

from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from typing import cast
from weakref import WeakKeyDictionary

from packaging.metadata import Metadata

from scripts import candidate_source_archive_oracle as archives
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_INFO = "hermes_realtime-0.0.3.dist-info/"
_METADATA = frozenset(
    _INFO + name
    for name in (
        "METADATA",
        "WHEEL",
        "entry_points.txt",
        "licenses/LICENSE",
        "RECORD",
    )
)
_MAX_WHEEL = 15 * 1024**2
_MAX_EXPANDED = 32 * 1024**2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class _EntryPointParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _inspect_wheel(raw: bytes, expected: dict[str, tuple[int, str]]) -> dict[str, bytes]:
    _require(0 < len(raw) <= _MAX_WHEEL, "candidate wheel size is outside its bound")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
            entries = wheel.infolist()
            names = [entry.filename for entry in entries]
            _require(len(names) == len(set(names)), "candidate wheel contains duplicate members")
            _require(set(names) == set(expected) | _METADATA, "candidate wheel member set differs")
            _require(
                all(not entry.is_dir() and entry.file_size >= 0 for entry in entries)
                and sum(entry.file_size for entry in entries) <= _MAX_EXPANDED,
                "candidate wheel expansion exceeds its bound",
            )
            members = {entry.filename: wheel.read(entry) for entry in entries}
    except (zipfile.BadZipFile, RuntimeError) as error:
        raise ValueError("candidate wheel is unreadable") from error
    for name, (size, digest) in expected.items():
        value = members[name]
        _require(
            (len(value), hashlib.sha256(value).hexdigest()) == (size, digest),
            "candidate wheel runtime blob differs from source",
        )
    for name in ("METADATA", "WHEEL"):
        try:
            members[_INFO + name].decode("utf-8")
        except UnicodeError as error:
            raise ValueError("candidate wheel metadata is not UTF-8") from error
    metadata, wheel_metadata = (
        BytesParser().parsebytes(members[_INFO + name]) for name in ("METADATA", "WHEEL")
    )
    for document in (metadata, wheel_metadata):
        _require(
            not document.defects
            and document.get_unixfrom() is None
            and not document.is_multipart()
            and type(document.get_payload()) is str
            and all(re.fullmatch(r"[!-9;-~]+", name) for name in document),
            "candidate wheel metadata is not a plain header document",
        )
    _require(
        metadata.get_all("Metadata-Version") == ["2.4"]
        and metadata.get_all("Name") == ["hermes-realtime"]
        and metadata.get_all("Version") == ["0.0.3"],
        "candidate wheel core metadata differs or is malformed",
    )
    try:
        Metadata.from_email(members[_INFO + "METADATA"], validate=True)
    except (ValueError, ExceptionGroup):
        raise ValueError("candidate wheel core metadata is invalid") from None
    _require(
        wheel_metadata.get_payload() == ""
        and wheel_metadata.get_all("Wheel-Version") == ["1.0"]
        and wheel_metadata.get_all("Root-Is-Purelib") == ["true"]
        and wheel_metadata.get_all("Tag") == ["py3-none-any"],
        "candidate wheel is not pure",
    )
    entry_points = _EntryPointParser(interpolation=None)
    try:
        entry_points.read_string(members[_INFO + "entry_points.txt"].decode("utf-8"))
    except (UnicodeError, configparser.Error) as error:
        raise ValueError("candidate wheel entry points are unreadable") from error
    _require(
        not entry_points.defaults()
        and set(entry_points.sections()) == {"console_scripts", "hermes_agent.plugins"}
        and dict(entry_points["console_scripts"])
        == {
            "hermes-realtime-host": "hermes_realtime.host_launcher:main",
            "hermes-realtime-local": "hermes_realtime.launcher:main",
        }
        and dict(entry_points["hermes_agent.plugins"])
        == {
            "hermes-realtime": "hermes_realtime.hermes_plugin",
        },
        "candidate wheel entry points differ",
    )
    try:
        records = list(csv.reader(io.StringIO(members[_INFO + "RECORD"].decode("utf-8"))))
    except (UnicodeError, csv.Error) as error:
        raise ValueError("candidate wheel RECORD is unreadable") from error
    _require(all(len(row) == 3 for row in records), "candidate wheel RECORD fields differ")
    _require(
        len(records) == len(members) and {row[0] for row in records} == set(members),
        "candidate wheel RECORD member set differs",
    )
    for name, digest, record_size in records:
        value = members[name]
        expected_digest = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(value).digest()
        ).rstrip(b"=").decode("ascii")
        _require(
            (digest, record_size)
            == (("", "") if name == _INFO + "RECORD" else (expected_digest, str(len(value)))),
            "candidate wheel RECORD does not bind its bytes",
        )
    return members


class VerifiedCandidateWheelV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("wheel capabilities are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _VerifiedWheel:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    wheel_sha256: str
    members: dict[str, bytes]


_WHEELS: WeakKeyDictionary[VerifiedCandidateWheelV1, _VerifiedWheel] = WeakKeyDictionary()


def verify_candidate_wheel_v1(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    path: Path,
    sha256: str,
) -> VerifiedCandidateWheelV1:
    metadata = archives.verified_candidate_source_archive_metadata(archive)
    # Recheck the capability's candidate identity before reading the supplied wheel.
    archives._archive_bytes_for_consumer(archive, identity)
    with path.open("rb") as stream:
        raw = stream.read(_MAX_WHEEL + 1)
    _require(
        type(sha256) is str and hashlib.sha256(raw).hexdigest() == sha256,
        "candidate wheel digest differs",
    )
    expected = {
        member.path.removeprefix("src/"): (member.size, cast(str, member.sha256))
        for member in metadata.manifest
        if member.kind == "file" and member.path.startswith("src/hermes_realtime/")
    }
    _require(bool(expected), "source archive contains no runtime package")
    members = _inspect_wheel(raw, expected)
    licenses = [member for member in metadata.manifest if member.path == "LICENSE"]
    _require(
        len(licenses) == 1
        and hashlib.sha256(members[_INFO + "licenses/LICENSE"]).hexdigest() == licenses[0].sha256,
        "candidate wheel license differs from source",
    )
    token = object.__new__(VerifiedCandidateWheelV1)
    _WHEELS[token] = _VerifiedWheel(
        metadata.candidate_head_oid,
        metadata.candidate_tree_oid,
        metadata.archive_sha256,
        sha256,
        members,
    )
    return token


def _wheel_for_consumer(
    token: VerifiedCandidateWheelV1,
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
) -> _VerifiedWheel:
    _require(
        type(token) is VerifiedCandidateWheelV1 and token in _WHEELS,
        "candidate wheel has no verified authority",
    )
    archives._archive_bytes_for_consumer(archive, identity)
    metadata = archives.verified_candidate_source_archive_metadata(archive)
    value = _WHEELS[token]
    _require(
        (value.source_commit, value.source_tree, value.source_archive_sha256)
        == (
            metadata.candidate_head_oid,
            metadata.candidate_tree_oid,
            metadata.archive_sha256,
        ),
        "candidate wheel is bound to another source archive",
    )
    return value
