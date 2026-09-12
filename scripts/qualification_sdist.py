"""Compare bounded sdist bytes with genuine source and wheel authorities.

This inspects files without extracting or executing them. Equal outputs do not
prove that independent builds occurred; invocation receipts remain separate.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
import tomllib
from dataclasses import dataclass

from packaging.metadata import parse_email

from scripts import candidate_source_archive_oracle as archives
from scripts import candidate_wheel as wheels
from scripts.qualify_evidence_slice_zero import _safe_posix_path
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_PREFIX = "hermes_realtime-0.0.3/"
_MAX_COMPRESSED = 16 * 1024**2
_MAX_EXPANDED = 32 * 1024**2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class CandidateSdistMetadataV1:
    """File comparison only; never a build or installed execution capability."""

    sha256: str
    source_commit: str
    source_tree: str
    files: int


def _selector(value: object) -> tuple[str, bool]:
    _require(type(value) is str and value.startswith("/"), "sdist selector is unsupported")
    text = str(value)[1:]
    recursive = text.endswith("/**")
    path = text[:-3] if recursive else text
    _safe_posix_path(path, label="sdist selector")
    _require(not any(character in path for character in "*?[]!"), "sdist glob is unsupported")
    return path, recursive


def _matches(path: str, selection: tuple[str, bool]) -> bool:
    name, recursive = selection
    return path.startswith(name + "/") if recursive else path == name


def inspect_candidate_sdist(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: wheels.VerifiedCandidateWheelV1,
    raw: bytes,
) -> CandidateSdistMetadataV1:
    return _candidate_sdist_files(archive, identity, wheel, raw)[0]


def _candidate_sdist_files(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: wheels.VerifiedCandidateWheelV1,
    raw: bytes,
) -> tuple[CandidateSdistMetadataV1, dict[str, bytes]]:
    """Source/wheel capabilities precede all supplied archive parsing."""
    source = archives.verified_candidate_source_archive_metadata(archive)
    payload = archives._archive_bytes_for_consumer(archive, identity)
    distribution = wheels._wheel_for_consumer(wheel, archive, identity)
    project, _ = wheels._metadata_source_files(payload, source)
    try:
        profile = tomllib.loads(project.decode("utf-8"))["tool"]["hatch"]["build"]["targets"][
            "sdist"
        ]
        _require(set(profile) == {"include", "exclude"}, "sdist selection profile differs")
        _require(
            type(profile["include"]) is list and type(profile["exclude"]) is list,
            "sdist selectors must be lists",
        )
        include = tuple(_selector(item) for item in profile["include"])
        exclude = tuple(_selector(item) for item in profile["exclude"])
        _require(0 < len(include) <= 64 and len(exclude) <= 64, "sdist selectors exceed bounds")
        expected = {
            item.path: (item.size, item.sha256)
            for item in source.manifest
            if item.kind == "file"
            and any(_matches(item.path, part) for part in include)
            and not any(_matches(item.path, part) for part in exclude)
        }
        # Hatchling 1.27 force-includes the root VCS exclusion file. Require its
        # candidate blob; an ambient exclusion file or arbitrary extra is not admitted.
        expected.update(
            {
                item.path: (item.size, item.sha256)
                for item in source.manifest
                if item.kind == "file" and item.path == ".gitignore"
            }
        )
        _require(
            {"pyproject.toml", "README.md", "LICENSE"} <= expected.keys(),
            "sdist source metadata is not selected",
        )
        _require(
            type(raw) is bytes and 0 < len(raw) <= _MAX_COMPRESSED,
            "sdist compressed bytes exceed bounds",
        )
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
            expanded = compressed.read(_MAX_EXPANDED + 1)
        _require(len(expanded) <= _MAX_EXPANDED, "sdist expanded bytes exceed bounds")
        actual: dict[str, tuple[int, str]] = {}
        files: dict[str, bytes] = {}
        names: set[str] = set()
        package_info: bytes | None = None
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as contents:
            for member in contents:
                _require(
                    len(names) < 8192 and member.name.startswith(_PREFIX),
                    "sdist prefix or member bound differs",
                )
                name = member.name.removeprefix(_PREFIX).rstrip("/")
                if not name and member.isdir():
                    _require("" not in names, "sdist root is duplicated")
                    names.add("")
                    continue
                _safe_posix_path(name, label="sdist member")
                _require(name.casefold() not in names, "sdist member is duplicated")
                names.add(name.casefold())
                _require(member.isdir() or member.isfile(), "sdist contains an indirect member")
                _require(
                    not member.sparse and member.mode in (0o644, 0o755),
                    "sdist member mode or sparse representation differs",
                )
                if member.isdir():
                    _require(
                        any(path.startswith(name + "/") for path in expected),
                        "sdist directory is not source-selected",
                    )
                    continue
                _require(0 <= member.size <= 4 * 1024**2, "sdist member exceeds its bound")
                stream = contents.extractfile(member)
                _require(stream is not None, "sdist file is unreadable")
                assert stream is not None
                value = stream.read(member.size + 1)
                _require(len(value) == member.size, "sdist file size differs")
                files[name] = value
                if name == "PKG-INFO":
                    package_info = value
                else:
                    _require(name in expected, "sdist file is not source-selected")
                    actual[name] = len(value), hashlib.sha256(value).hexdigest()
        _require(actual == expected and package_info is not None, "sdist source closure differs")
        actual_info, unparsed = parse_email(package_info)
        expected_info, expected_unparsed = parse_email(
            distribution.members[wheels._INFO + "METADATA"]
        )
        _require(
            not unparsed
            and not expected_unparsed
            and wheels._normalized_metadata(actual_info)
            == wheels._normalized_metadata(expected_info),
            "sdist metadata differs from its source-bound wheel",
        )
    except (OSError, EOFError, tarfile.TarError, KeyError, TypeError, ValueError, ExceptionGroup):
        raise ValueError("sdist does not match its bounded source profile") from None
    return CandidateSdistMetadataV1(
        hashlib.sha256(raw).hexdigest(),
        source.candidate_head_oid,
        source.candidate_tree_oid,
        len(actual) + 1,
    ), files
