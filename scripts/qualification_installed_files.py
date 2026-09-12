"""Inspect a Windows uv target's bytes without claiming installation execution.

The caller must separately authenticate wheels, own the actual installer, retain
the complete runtime/package namespace and observe import origins and cleanup.
This ordinary metadata result cannot substitute for any of those capabilities.
"""

from __future__ import annotations

import base64
import configparser
import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from typing import Any

from scripts.qualification_file_seals import _relative_windows_member
from scripts.qualification_wheelhouse import WheelDistributionV1, inspect_wheelhouse_files


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class InstalledFileMetadataV1:
    file_count: int
    distribution_count: int
    inventory_sha256: str


def _members(distribution: WheelDistributionV1) -> tuple[str, dict[str, tuple[str, int]]]:
    info = next(
        path.rsplit("/", 1)[0]
        for path, _, _ in distribution.members
        if path.count("/") == 1 and path.endswith(".dist-info/METADATA")
    )
    data = info.removesuffix(".dist-info") + ".data/"
    mapped = {}
    for path, digest, size in distribution.members:
        if path.startswith(data):
            scheme, path = path[len(data) :].split("/", 1)
            _require(scheme in {"purelib", "platlib"}, "installed wheel scheme is unavailable")
        mapped[path] = (digest, size)
    return info, mapped


class _EntryPointConfig(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _entry_points(raw: bytes, info: str) -> set[str]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        name = info + "/entry_points.txt"
        if name not in archive.namelist():
            return set()
        _require(
            archive.getinfo(name).file_size <= 4 * 1024**2, "entry-point metadata is unbounded"
        )
        text = archive.read(name).decode("utf-8")
    config = _EntryPointConfig(interpolation=None, strict=True)
    try:
        config.read_string(text)
    except configparser.Error:
        raise ValueError("installed entry-point metadata is invalid") from None
    result: set[str] = set()
    for section in ("console_scripts", "gui_scripts"):
        for name in config[section] if config.has_section(section) else ():
            _require(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", name) is not None,
                "installed entry-point name is unsafe",
            )
            path = "bin/" + name + ".exe"
            _require(path not in result, "installed entry-point ownership is ambiguous")
            result.add(path)
    return result


def _cache(raw: bytes) -> None:
    _require(len(raw) <= 4096, "installed cache metadata is unbounded")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        _require(len(dict(items)) == len(items), "installed cache metadata is ambiguous")
        return dict(items)

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("installed cache metadata is invalid") from None
    _require(type(value) is dict, "installed cache metadata is not an object")
    timestamp = value.get("timestamp")
    _require(
        type(timestamp) is dict
        and set(timestamp) == {"secs_since_epoch", "nanos_since_epoch"}
        and type(timestamp["secs_since_epoch"]) is int
        and 0 <= timestamp["secs_since_epoch"] < 2**63
        and type(timestamp["nanos_since_epoch"]) is int
        and 0 <= timestamp["nanos_since_epoch"] < 10**9,
        "installed cache timestamp differs",
    )
    _require(
        value
        == {"timestamp": timestamp, "commit": None, "tags": None, "env": {}, "directories": {}},
        "installed cache metadata exceeds the offline wheel profile",
    )


def _record(raw: bytes, record: str, expected: set[str], installed: dict[str, bytes]) -> None:
    _require(len(raw) <= 4 * 1024**2, "installed RECORD is unbounded")
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8")), strict=True))
    except (UnicodeError, csv.Error):
        raise ValueError("installed RECORD is invalid") from None
    _require(
        len(rows) == len(expected)
        and all(len(row) == 3 for row in rows)
        and len({row[0] for row in rows}) == len(rows)
        and {row[0] for row in rows} == expected,
        "installed RECORD ownership differs",
    )
    for name, digest, size in rows:
        payload = installed[name]
        encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
        actual = ("", "") if name == record else ("sha256=" + encoded, str(len(payload)))
        _require((digest, size) == actual, "installed RECORD content differs")


def inspect_installed_wheels(
    *,
    requirements: bytes,
    constraints: bytes,
    wheels: dict[str, bytes],
    python_version: str,
    platform: str,
    installed: dict[str, bytes],
) -> InstalledFileMetadataV1:
    """Compare original wheel members and the closed, pinned uv target profile."""
    _require(platform == "windows_amd64", "installed inspection requires the Windows uv profile")
    inventory = inspect_wheelhouse_files(
        requirements=requirements,
        constraints=constraints,
        wheels=wheels,
        python_version=python_version,
        platform=platform,
        # Qualification consumers require -I -S -B. Preserve path-configuration
        # files as inert original bytes; this inventory does not execute them.
        site_processing=False,
    )
    _require(
        type(installed) is dict and 0 < len(installed) <= 16384, "installed inventory is unbounded"
    )
    installed = dict(installed)
    for name, raw in installed.items():
        _relative_windows_member(name)
        _require(type(raw) is bytes and len(raw) <= 512 * 1024**2, "installed file is unbounded")
    parents = {
        "/".join(name.split("/")[:depth])
        for name in installed
        for depth in range(1, len(name.split("/")))
    }
    folded_files = {name.casefold() for name in installed}
    folded_parents = {name.casefold() for name in parents}
    _require(
        len(folded_files) == len(installed)
        and len(folded_parents) == len(parents)
        and not folded_files.intersection(folded_parents)
        and sum(map(len, installed.values())) <= 4 * 1024**3,
        "installed namespace is ambiguous or unbounded",
    )
    _require(installed.get(".lock") == b"", "installed target lock differs")
    owners = {".lock"}
    by_digest = {hashlib.sha256(raw).hexdigest(): raw for raw in wheels.values()}
    for distribution in inventory:
        info, original = _members(distribution)
        record = info + "/RECORD"
        generated = {info + "/" + name for name in ("INSTALLER", "REQUESTED", "uv_cache.json")}
        generated.update(_entry_points(by_digest[distribution.sha256], info))
        expected = set(original) | generated
        _require(
            not set(original).intersection(generated)
            and not owners.intersection(expected)
            and expected <= set(installed),
            "installed distribution ownership is missing or ambiguous",
        )
        for name, binding in original.items():
            if name != record:
                raw = installed[name]
                _require(
                    (hashlib.sha256(raw).hexdigest(), len(raw)) == binding,
                    "installed source bytes differ from their wheel",
                )
        _require(
            installed[info + "/INSTALLER"] == b"uv" and installed[info + "/REQUESTED"] == b"",
            "installed origin metadata differs",
        )
        _cache(installed[info + "/uv_cache.json"])
        _record(installed[record], record, expected, installed)
        owners.update(expected)
    _require(owners == set(installed), "installed inventory has unowned files")
    rows = [
        (name, hashlib.sha256(raw).hexdigest(), len(raw)) for name, raw in sorted(installed.items())
    ]
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return InstalledFileMetadataV1(len(rows), len(inventory), digest)
