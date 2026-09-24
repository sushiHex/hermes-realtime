"""Durable, bounded file format for the Hermes runs one host may have left running."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_VERSION = 1
_KEY = re.compile(r"[0-9a-f]{32}\Z")
_REQUEST_FIELDS = frozenset({"input", "instructions", "provider", "model", "model_options"})


@dataclass(frozen=True, slots=True, eq=False)
class PendingRun:
    """One dispatch as recorded before its first POST: its key, when it was minted, its body."""

    key: str | None
    minted_at: float
    body: dict[str, object]


def max_run_record_bytes(max_entries: int, max_request_bytes: int) -> int:
    """One entry per run slot, and no recorded request is larger than one Hermes body."""
    return (max_entries + 1) * max_request_bytes


def write_run_record(path: Path, data: bytes) -> None:
    """Replace the record atomically: a crash leaves the old record or the new one, never part."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def lock_run_record(path: Path) -> int | None:
    """Hold the record's sibling lock file exclusively; None when another holder has it."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(descriptor)
        return None
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def unlock_run_record(descriptor: int) -> None:
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def read_run_record(path: Path, max_bytes: int) -> bytes | None:
    """Read at most one byte past the bound, so an oversized record is seen as oversized."""
    try:
        with path.open("rb") as stream:
            return stream.read(max_bytes + 1)
    except FileNotFoundError:
        return None


def run_record_bytes(pending: list[PendingRun], admitted: list[str]) -> bytes:
    document = {
        "admitted": admitted,
        "pending": [
            {"key": entry.key, "minted_at": entry.minted_at, "request": entry.body}
            for entry in pending
        ],
        "version": _VERSION,
    }
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8")


def parse_run_record(
    raw: bytes,
    *,
    max_entries: int,
    max_request_bytes: int,
    run_id: re.Pattern[str],
) -> tuple[list[PendingRun], list[str]] | None:
    """Return the record's pending and admitted entries, or None when it is malformed."""
    if len(raw) > max_run_record_bytes(max_entries, max_request_bytes):
        return None
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeDecodeError, ValueError):
        return None
    if type(document) is not dict or set(document) != {"admitted", "pending", "version"}:
        return None
    version, pending, admitted = document["version"], document["pending"], document["admitted"]
    if type(version) is not int or version != _VERSION:
        return None
    if type(pending) is not list or type(admitted) is not list:
        return None
    if len(pending) + len(admitted) > max_entries:
        return None
    run_ids: list[str] = []
    for api_run_id in admitted:
        if type(api_run_id) is not str or run_id.fullmatch(api_run_id) is None:
            return None
        if api_run_id in run_ids:
            return None
        run_ids.append(api_run_id)
    entries: list[PendingRun] = []
    keys: set[str] = set()
    for entry in pending:
        if type(entry) is not dict or set(entry) != {"key", "minted_at", "request"}:
            return None
        key, minted_at, request = entry["key"], entry["minted_at"], entry["request"]
        if key is not None:
            if type(key) is not str or _KEY.fullmatch(key) is None or key in keys:
                return None
            keys.add(key)
        if type(minted_at) is not float or not math.isfinite(minted_at) or minted_at <= 0:
            return None
        if not _is_run_request(request, max_request_bytes):
            return None
        entries.append(PendingRun(key=key, minted_at=minted_at, body=request))
    return entries, run_ids


def _is_run_request(value: object, max_request_bytes: int) -> bool:
    if type(value) is not dict or not {"input", "instructions"} <= set(value) <= _REQUEST_FIELDS:
        return False
    for name in ("input", "instructions", "provider", "model"):
        if name in value and type(value[name]) is not str:
            return False
    if "model_options" in value:
        options = value["model_options"]
        if type(options) is not dict or any(
            type(name) is not str or type(option) is not str for name, option in options.items()
        ):
            return False
    return len(json.dumps(value).encode("utf-8")) <= max_request_bytes


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, rejecting a duplicated field instead of keeping the last one."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result
