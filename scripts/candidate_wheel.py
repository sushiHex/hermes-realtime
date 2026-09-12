"""Bind a bounded pure wheel's runtime and metadata to a verified source archive."""

from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import io
import re
import tarfile
import tomllib
import zipfile
from collections import Counter
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from typing import cast
from weakref import WeakKeyDictionary

from packaging.markers import Marker
from packaging.metadata import Metadata, RawMetadata, parse_email
from packaging.requirements import Requirement

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


def _normalized_metadata(raw: RawMetadata) -> dict[str, object]:
    metadata = Metadata.from_raw(raw, validate=True)
    # Typed requirements/specifiers compare semantically; repeated fields retain
    # their multiplicity, without imposing an RFC header serialization order.
    result: dict[str, object] = {}
    for field in raw:
        value = getattr(metadata, field)
        result[field] = Counter(value) if isinstance(value, list) else value
    return result


def _bind_source_metadata(members: dict[str, bytes], project_bytes: bytes, readme: bytes) -> None:
    """Bind the repository's static PEP 621 profile, without executing a builder."""
    try:
        document = tomllib.loads(project_bytes.decode("utf-8"))
        project = document["project"]
        _require(
            set(project)
            == {
                "name",
                "version",
                "description",
                "readme",
                "requires-python",
                "authors",
                "license",
                "license-files",
                "keywords",
                "classifiers",
                "dependencies",
                "entry-points",
                "urls",
                "scripts",
                "optional-dependencies",
            }
            and project["readme"] == "README.md"
            and project["license-files"] == ["LICENSE"]
            and document["build-system"]
            == {
                "requires": ["hatchling==1.27.0"],
                "build-backend": "hatchling.build",
            },
            "source metadata profile is unsupported",
        )
        authors = project["authors"]
        _require(
            type(authors) is list
            and bool(authors)
            and all(
                type(author) is dict and set(author) == {"name"} and type(author["name"]) is str
                for author in authors
            ),
            "source author profile is unsupported",
        )
        dependencies = project["dependencies"]
        extras = project["optional-dependencies"]
        _require(
            type(dependencies) is list
            and type(extras) is dict
            and all(type(value) is list for value in extras.values()),
            "source dependency profile is unsupported",
        )
        requirements = [str(Requirement(value)) for value in dependencies]
        for extra, values in extras.items():
            for value in values:
                requirement = Requirement(value)
                marker = f"extra == {extra!r}"
                if requirement.marker is not None:
                    marker = f"({requirement.marker}) and {marker}"
                requirement.marker = Marker(marker)
                requirements.append(str(requirement))
        expected: RawMetadata = {
            "metadata_version": "2.4",
            "name": project["name"],
            "version": project["version"],
            "summary": project["description"],
            "description": readme.decode("utf-8"),
            "description_content_type": "text/markdown",
            "requires_python": project["requires-python"],
            "author": ", ".join(author["name"] for author in authors),
            "license_expression": project["license"],
            "license_files": project["license-files"],
            "keywords": project["keywords"],
            "classifiers": project["classifiers"],
            "project_urls": project["urls"],
            "provides_extra": list(extras),
            "requires_dist": requirements,
        }
        actual, unparsed = parse_email(members[_INFO + "METADATA"])
        _require(
            not unparsed and _normalized_metadata(actual) == _normalized_metadata(expected),
            "candidate wheel core metadata differs from source",
        )
        entry_points = _EntryPointParser(interpolation=None)
        entry_points.read_string(members[_INFO + "entry_points.txt"].decode("utf-8"))
        expected_entries = dict(project["entry-points"])
        _require("console_scripts" not in expected_entries, "source entry points are ambiguous")
        expected_entries["console_scripts"] = project["scripts"]
        _require(
            not entry_points.defaults()
            and {section: dict(entry_points[section]) for section in entry_points.sections()}
            == expected_entries,
            "candidate wheel entry points differ from source",
        )
        wheel = BytesParser().parsebytes(members[_INFO + "WHEEL"])
        _require(
            set(wheel) == {"Wheel-Version", "Generator", "Root-Is-Purelib", "Tag"}
            and wheel.get_all("Generator") == ["hatchling 1.27.0"],
            "candidate wheel build metadata differs from source",
        )
    except (KeyError, TypeError, ValueError, AttributeError, configparser.Error, ExceptionGroup):
        # Metadata is untrusted; exceptions must never echo its field values.
        raise ValueError(
            "candidate wheel metadata does not match the static source profile"
        ) from None


def _metadata_source_files(
    payload: bytes,
    metadata: archives.CandidateSourceArchiveMetadataV1,
) -> tuple[bytes, bytes]:
    values = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as source:
        for name in ("pyproject.toml", "README.md"):
            manifest = [member for member in metadata.manifest if member.path == name]
            _require(
                len(manifest) == 1
                and manifest[0].kind == "file"
                and 0 < manifest[0].size <= 1024**2,
                "source metadata file is missing or outside its bound",
            )
            stream = source.extractfile(f"{metadata.prefix}/{name}")
            _require(stream is not None, "source metadata file is unreadable")
            assert stream is not None
            with stream:
                raw = stream.read(1024**2 + 1)
            _require(
                (len(raw), hashlib.sha256(raw).hexdigest())
                == (manifest[0].size, manifest[0].sha256),
                "source metadata file differs from its manifest",
            )
            values.append(raw)
    return values[0], values[1]


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


def _verify_candidate_wheel_bytes_v1(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    raw: bytes,
    sha256: str,
) -> VerifiedCandidateWheelV1:
    metadata = archives.verified_candidate_source_archive_metadata(archive)
    # Recheck the capability's candidate identity before reading the supplied wheel.
    payload = archives._archive_bytes_for_consumer(archive, identity)
    _require(
        type(raw) is bytes and 0 < len(raw) <= _MAX_WHEEL,
        "candidate wheel size is outside its bound",
    )
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
    _bind_source_metadata(members, *_metadata_source_files(payload, metadata))
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


def verify_candidate_wheel_v1(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    path: Path,
    sha256: str,
) -> VerifiedCandidateWheelV1:
    # Refuse a confused source before opening any caller-selected wheel path.
    archives.verified_candidate_source_archive_metadata(archive)
    archives._archive_bytes_for_consumer(archive, identity)
    with path.open("rb") as stream:
        raw = stream.read(_MAX_WHEEL + 1)
    return _verify_candidate_wheel_bytes_v1(archive, identity, raw, sha256)


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
