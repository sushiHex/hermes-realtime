"""Private durable facts for a future qualification-run recovery owner.

This module borrows an already owned journal handle.  It does not choose or
create a recovery location, launch or adopt a process, remove a file, perform
recovery, or mint qualification authority.  Recorded exit/removal events are
observations which a future recovery owner must independently validate.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Protocol, TypeVar, cast

_MAGIC = b"HRQJ1\x00\r\n"
_ZERO_DIGEST = bytes(32)
_MAX_FRAME_BYTES = 64 * 1024
_MAX_PAYLOAD_BYTES = _MAX_FRAME_BYTES - 4 - 32
_MAX_RECORDS = 2048
_MAX_JOURNAL_BYTES = 8 * 1024**2
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OID = re.compile(r"[0-9a-f]{40}")
_LABEL = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_int(value: object, *, positive: bool = False, maximum: int = 2**64 - 1) -> int:
    if type(value) is not int:
        raise ValueError("journal integer fact differs")
    result = value
    _require(
        (result > 0 if positive else result >= 0) and result <= maximum,
        "journal integer fact differs",
    )
    return result


def _path(value: object) -> str:
    if type(value) is not str:
        raise ValueError("journal path fact differs")
    result = value
    _require(3 <= len(result) <= 32767, "journal path fact differs")
    path = PureWindowsPath(result)
    _require(
        path.is_absolute()
        and not result.startswith("\\\\")
        and ".." not in path.parts
        and path.as_posix() == result,
        "journal path fact differs",
    )
    return result


def _relative(value: str) -> str:
    _require(type(value) is str and 1 <= len(value) <= 512, "journal artifact fact differs")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and path.as_posix() == value
        and value not in {".", ".."}
        and ".." not in path.parts
        and "\\" not in value,
        "journal artifact fact differs",
    )
    return value


@dataclass(frozen=True, slots=True)
class RunJournalBindingV1:
    run_id: str
    candidate_head: str
    candidate_tree: str
    qualification_input_sha256: str

    def __post_init__(self) -> None:
        _require(
            type(self.run_id) is str and _RUN.fullmatch(self.run_id) is not None,
            "journal run binding differs",
        )
        _require(
            type(self.candidate_head) is str
            and _OID.fullmatch(self.candidate_head) is not None
            and type(self.candidate_tree) is str
            and _OID.fullmatch(self.candidate_tree) is not None
            and type(self.qualification_input_sha256) is str
            and _SHA256.fullmatch(self.qualification_input_sha256) is not None,
            "journal candidate binding differs",
        )


@dataclass(frozen=True, slots=True)
class RecoveryLocationFactsV1:
    journal_path: str
    parent_path: str
    volume_serial: int
    journal_file_id: int
    parent_file_id: int
    marker_path: str
    marker_file_id: int
    marker_sha256: str

    def __post_init__(self) -> None:
        journal, parent, marker = map(
            _path, (self.journal_path, self.parent_path, self.marker_path)
        )
        _require(
            PureWindowsPath(journal).parent == PureWindowsPath(parent)
            and PureWindowsPath(marker).parent == PureWindowsPath(parent)
            and PureWindowsPath(marker) != PureWindowsPath(journal),
            "journal recovery location differs",
        )
        _exact_int(self.volume_serial, positive=True, maximum=2**32 - 1)
        for value in (self.journal_file_id, self.parent_file_id, self.marker_file_id):
            _exact_int(value, positive=True)
        _require(
            self.marker_file_id != self.journal_file_id,
            "journal recovery location differs",
        )
        _require(
            type(self.marker_sha256) is str and _SHA256.fullmatch(self.marker_sha256) is not None,
            "journal marker fact differs",
        )


@dataclass(frozen=True, slots=True)
class FilesystemIntentV1:
    role: str
    path: str
    directory: bool
    cleanup_scope: str
    allowed_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(
            type(self.role) is str and _LABEL.fullmatch(self.role) is not None,
            "filesystem intent differs",
        )
        _path(self.path)
        _require(type(self.directory) is bool, "filesystem intent differs")
        _require(
            self.cleanup_scope in {"exact", "owned_subtree"}
            and type(self.cleanup_scope) is str
            and (self.cleanup_scope == "owned_subtree" or not self.directory),
            "filesystem intent differs",
        )
        _require(
            type(self.allowed_artifacts) is tuple
            and len(self.allowed_artifacts) <= 256
            and all(type(item) is str for item in self.allowed_artifacts),
            "filesystem intent differs",
        )
        for item in self.allowed_artifacts:
            _relative(item)
        _require(
            tuple(sorted(self.allowed_artifacts)) == self.allowed_artifacts
            and len(set(self.allowed_artifacts)) == len(self.allowed_artifacts),
            "filesystem intent differs",
        )


@dataclass(frozen=True, slots=True)
class FilesystemIdentityV1:
    path: str
    parent_path: str
    volume_serial: int
    file_id: int
    parent_file_id: int
    marker_path: str
    marker_file_id: int
    marker_sha256: str
    directory: bool

    def __post_init__(self) -> None:
        path, parent, marker = map(_path, (self.path, self.parent_path, self.marker_path))
        _require(
            PureWindowsPath(path).parent == PureWindowsPath(parent)
            and PureWindowsPath(marker).parent == PureWindowsPath(parent)
            and PureWindowsPath(marker) != PureWindowsPath(path),
            "filesystem identity differs",
        )
        _exact_int(self.volume_serial, positive=True, maximum=2**32 - 1)
        for value in (self.file_id, self.parent_file_id, self.marker_file_id):
            _exact_int(value, positive=True)
        _require(self.marker_file_id != self.file_id, "filesystem identity differs")
        _require(
            type(self.marker_sha256) is str
            and _SHA256.fullmatch(self.marker_sha256) is not None
            and type(self.directory) is bool,
            "filesystem identity differs",
        )


@dataclass(frozen=True, slots=True)
class ProcessIntentV1:
    scenario_id: str
    role: str
    image_basename: str
    image_sha256: str

    def __post_init__(self) -> None:
        _require(
            type(self.scenario_id) is str
            and _LABEL.fullmatch(self.scenario_id) is not None
            and type(self.role) is str
            and _LABEL.fullmatch(self.role) is not None,
            "process intent differs",
        )
        _require(
            type(self.image_basename) is str
            and re.fullmatch(r'[^\\/:*?"<>|\x00]{1,255}', self.image_basename) is not None
            and type(self.image_sha256) is str
            and _SHA256.fullmatch(self.image_sha256) is not None,
            "process intent differs",
        )


@dataclass(frozen=True, slots=True)
class ProcessIdentityFactsV1:
    scenario_id: str
    role: str
    pid: int
    parent_pid: int
    parent_creation_filetime: int
    creation_filetime: int
    image_basename: str
    image_sha256: str

    def __post_init__(self) -> None:
        ProcessIntentV1(self.scenario_id, self.role, self.image_basename, self.image_sha256)
        for value in (
            self.pid,
            self.parent_pid,
            self.parent_creation_filetime,
            self.creation_filetime,
        ):
            _exact_int(value, positive=True)


@dataclass(frozen=True, slots=True)
class RunJournalFactsV1:
    """Structural private facts only; never cleanup, recovery, or acceptance authority."""

    frame_count: int
    byte_count: int
    integrity_complete: bool
    partial_tail: bool
    recorded_complete: bool
    execution_uncertain: bool
    pending_filesystems: tuple[int, ...]
    pending_processes: tuple[int, ...]
    reason: str | None


@dataclass(frozen=True, slots=True)
class _JournalHandleFactV1:
    volume_serial: int
    file_id: int
    size: int
    reparse: bool
    links: int
    directory: bool = False
    inheritable: bool = False


class _JournalIoV1(Protocol):
    def handle_fact(self, handle: int) -> _JournalHandleFactV1: ...

    def final_path(self, handle: int) -> str: ...

    def read(self, handle: int, maximum: int) -> bytes: ...

    def write(self, handle: int, offset: int, raw: bytes) -> int: ...

    def flush(self, handle: int) -> None: ...


@dataclass(slots=True)
class _State:
    binding: RunJournalBindingV1 | None
    location: RecoveryLocationFactsV1 | None
    filesystems: dict[int, tuple[FilesystemIntentV1, str, FilesystemIdentityV1 | None]]
    processes: dict[int, tuple[ProcessIntentV1, str, ProcessIdentityFactsV1 | None]]
    uncertain: set[int]
    complete: bool


@dataclass(frozen=True, slots=True)
class _Parsed:
    facts: RunJournalFactsV1
    state: _State
    digest: bytes


class RunJournalDurabilityError(RuntimeError):
    pass


def _new_state() -> _State:
    return _State(None, None, {}, {}, set(), False)


_Fact = (
    RunJournalBindingV1
    | RecoveryLocationFactsV1
    | FilesystemIntentV1
    | FilesystemIdentityV1
    | ProcessIntentV1
    | ProcessIdentityFactsV1
)


def _dict(value: _Fact) -> dict[str, Any]:
    return asdict(value)


def _strict_object(raw: bytes) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if type(key) is not str or key in result:
                raise ValueError("journal record fields differ")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    _require(type(value) is dict, "journal record differs")
    _require(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode() == raw,
        "journal record is noncanonical",
    )
    return cast(dict[str, Any], value)


T = TypeVar("T")


def _typed(mapping: object, expected: set[str], constructor: type[T]) -> T:
    if type(mapping) is not dict:
        raise ValueError("journal record facts differ")
    values = cast(dict[str, Any], mapping)
    _require(set(values) == expected, "journal record facts differ")
    return constructor(**values)


def _binding_from(value: object) -> RunJournalBindingV1:
    return _typed(
        value,
        {"run_id", "candidate_head", "candidate_tree", "qualification_input_sha256"},
        RunJournalBindingV1,
    )


def _location_from(value: object) -> RecoveryLocationFactsV1:
    return _typed(
        value,
        {
            "journal_path",
            "parent_path",
            "volume_serial",
            "journal_file_id",
            "parent_file_id",
            "marker_path",
            "marker_file_id",
            "marker_sha256",
        },
        RecoveryLocationFactsV1,
    )


def _filesystem_intent_from(value: object) -> FilesystemIntentV1:
    if type(value) is not dict:
        raise ValueError("journal filesystem intent differs")
    copied = dict(cast(dict[str, Any], value))
    artifacts = copied.get("allowed_artifacts")
    if type(artifacts) not in {list, tuple}:
        raise ValueError("journal filesystem intent differs")
    assert isinstance(artifacts, (list, tuple))
    copied["allowed_artifacts"] = tuple(artifacts)
    return _typed(
        copied,
        {"role", "path", "directory", "cleanup_scope", "allowed_artifacts"},
        FilesystemIntentV1,
    )


def _filesystem_identity_from(value: object) -> FilesystemIdentityV1:
    return _typed(
        value,
        {
            "path",
            "parent_path",
            "volume_serial",
            "file_id",
            "parent_file_id",
            "marker_path",
            "marker_file_id",
            "marker_sha256",
            "directory",
        },
        FilesystemIdentityV1,
    )


def _process_intent_from(value: object) -> ProcessIntentV1:
    return _typed(value, {"scenario_id", "role", "image_basename", "image_sha256"}, ProcessIntentV1)


def _process_identity_from(value: object) -> ProcessIdentityFactsV1:
    return _typed(
        value,
        {
            "scenario_id",
            "role",
            "pid",
            "parent_pid",
            "parent_creation_filetime",
            "creation_filetime",
            "image_basename",
            "image_sha256",
        },
        ProcessIdentityFactsV1,
    )


def _ordinal(value: object) -> int:
    if type(value) is not int:
        raise TypeError("journal ordinal type differs")
    _require(0 < value < _MAX_RECORDS, "journal ordinal differs")
    return value


def _record(sequence: int, previous: bytes, kind: str, facts: dict[str, Any]) -> bytes:
    value = {
        "facts": facts,
        "kind": kind,
        "previousSha256": previous.hex(),
        "schemaVersion": 1,
        "sequence": sequence,
    }
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    _require(0 < len(raw) <= _MAX_PAYLOAD_BYTES, "journal frame exceeds its bound")
    length = struct.pack(">I", len(raw))
    digest = hashlib.sha256(previous + length + raw).digest()
    return length + raw + digest


def _same_filesystem(intent: FilesystemIntentV1, identity: FilesystemIdentityV1) -> bool:
    return intent.path == identity.path and intent.directory is identity.directory


def _same_process(intent: ProcessIntentV1, identity: ProcessIdentityFactsV1) -> bool:
    return (
        intent.scenario_id,
        intent.role,
        intent.image_basename,
        intent.image_sha256,
    ) == (
        identity.scenario_id,
        identity.role,
        identity.image_basename,
        identity.image_sha256,
    )


def _apply(state: _State, record: dict[str, Any], expected_sequence: int) -> None:
    _require(
        set(record) == {"facts", "kind", "previousSha256", "schemaVersion", "sequence"}
        and type(record["schemaVersion"]) is int
        and record["schemaVersion"] == 1
        and type(record["sequence"]) is int
        and record["sequence"] == expected_sequence
        and type(record["kind"]) is str
        and type(record["facts"]) is dict,
        "journal record fields differ",
    )
    kind, facts = record["kind"], record["facts"]
    _require(not state.complete, "journal sequence is already complete")
    if expected_sequence == 0:
        _require(
            kind == "opened" and set(facts) == {"binding", "location"},
            "journal opening record differs",
        )
        state.binding = _binding_from(facts["binding"])
        state.location = _location_from(facts["location"])
        return
    _require(
        state.binding is not None and state.location is not None, "journal opening record is absent"
    )
    if kind == "filesystem_intent":
        intent = _filesystem_intent_from(facts)
        state.filesystems[expected_sequence] = (intent, "intended", None)
        return
    if kind in {"filesystem_bound", "filesystem_removed"}:
        _require(set(facts) == {"ordinal", "identity"}, "journal filesystem transition differs")
        ordinal = _ordinal(facts["ordinal"])
        file_identity = _filesystem_identity_from(facts["identity"])
        file_prior = state.filesystems.get(ordinal)
        if file_prior is None:
            raise ValueError("filesystem identity differs")
        _require(
            _same_filesystem(file_prior[0], file_identity)
            and (kind == "filesystem_bound" or file_prior[2] == file_identity),
            "filesystem identity differs",
        )
        required = "intended" if kind == "filesystem_bound" else "bound"
        _require(file_prior[1] == required, "filesystem transition differs")
        state.filesystems[ordinal] = (
            file_prior[0],
            "bound" if kind == "filesystem_bound" else "removed",
            file_identity,
        )
        return
    if kind == "filesystem_absent":
        _require(set(facts) == {"ordinal"}, "journal filesystem transition differs")
        ordinal = _ordinal(facts["ordinal"])
        file_prior = state.filesystems.get(ordinal)
        if file_prior is None:
            raise ValueError("filesystem transition differs")
        _require(file_prior[1] == "intended", "filesystem transition differs")
        state.filesystems[ordinal] = (file_prior[0], "absent", None)
        return
    if kind == "process_intent":
        process_intent = _process_intent_from(facts)
        state.processes[expected_sequence] = (process_intent, "intended", None)
        return
    if kind == "process_bound":
        _require(set(facts) == {"ordinal", "identity"}, "journal process transition differs")
        ordinal = _ordinal(facts["ordinal"])
        process_identity = _process_identity_from(facts["identity"])
        process_prior = state.processes.get(ordinal)
        if process_prior is None:
            raise ValueError("process identity differs")
        _require(_same_process(process_prior[0], process_identity), "process identity differs")
        _require(process_prior[1] == "intended", "process transition differs")
        state.processes[ordinal] = (process_prior[0], "bound", process_identity)
        return
    if kind == "resume_intent":
        _require(set(facts) == {"ordinal"}, "journal process transition differs")
        ordinal = _ordinal(facts["ordinal"])
        process_prior = state.processes.get(ordinal)
        if process_prior is None:
            raise ValueError("process transition differs")
        _require(process_prior[1] == "bound", "process transition differs")
        state.processes[ordinal] = (
            process_prior[0],
            "resume_intended",
            process_prior[2],
        )
        state.uncertain.add(ordinal)
        return
    if kind == "resumed":
        _require(set(facts) == {"ordinal"}, "journal process transition differs")
        ordinal = _ordinal(facts["ordinal"])
        process_prior = state.processes.get(ordinal)
        if process_prior is None:
            raise ValueError("process transition differs")
        _require(process_prior[1] == "resume_intended", "process transition differs")
        state.processes[ordinal] = (process_prior[0], "resumed", process_prior[2])
        state.uncertain.discard(ordinal)
        return
    if kind == "process_exited":
        _require(set(facts) == {"ordinal", "identity"}, "journal process transition differs")
        ordinal = _ordinal(facts["ordinal"])
        process_identity = _process_identity_from(facts["identity"])
        process_prior = state.processes.get(ordinal)
        _require(
            process_prior is not None
            and process_prior[1] in {"bound", "resume_intended", "resumed"}
            and process_prior[2] == process_identity,
            "process identity differs",
        )
        assert process_prior is not None
        state.processes[ordinal] = (process_prior[0], "exited", process_identity)
        return
    if kind == "process_absent":
        _require(set(facts) == {"ordinal"}, "journal process transition differs")
        ordinal = _ordinal(facts["ordinal"])
        process_prior = state.processes.get(ordinal)
        if process_prior is None:
            raise ValueError("process transition differs")
        _require(process_prior[1] == "intended", "process transition differs")
        state.processes[ordinal] = (process_prior[0], "absent", None)
        return
    if kind == "sequence_complete":
        _require(not facts, "journal completion record differs")
        pending_files = [
            row for row in state.filesystems.values() if row[1] not in {"removed", "absent"}
        ]
        pending_processes = [
            row for row in state.processes.values() if row[1] not in {"exited", "absent"}
        ]
        _require(not pending_files and not pending_processes, "sequence has unresolved obligations")
        _require(not state.uncertain, "sequence has unresolved execution")
        state.complete = True
        return
    raise ValueError("journal record kind differs")


def _facts(
    state: _State, count: int, size: int, *, integrity: bool, partial: bool, reason: str | None
) -> RunJournalFactsV1:
    pending_files = tuple(
        sorted(key for key, row in state.filesystems.items() if row[1] not in {"removed", "absent"})
    )
    pending_processes = tuple(
        sorted(key for key, row in state.processes.items() if row[1] not in {"exited", "absent"})
    )
    return RunJournalFactsV1(
        count,
        size,
        integrity,
        partial,
        state.complete and integrity,
        bool(state.uncertain),
        pending_files,
        pending_processes,
        reason,
    )


class _RunJournalWriterV1:
    __slots__ = ("_handle", "_io", "_location", "_state", "_digest", "_count", "_size", "_poisoned")

    def __init__(
        self,
        handle: int,
        io: Any,
        location: RecoveryLocationFactsV1,
        state: _State,
        digest: bytes,
        count: int,
        size: int,
    ) -> None:
        self._handle = handle
        self._io = io
        self._location = location
        self._state = state
        self._digest = digest
        self._count = count
        self._size = size
        self._poisoned = False

    def _append(self, kind: str, facts: dict[str, Any]) -> int:
        if self._poisoned:
            raise RunJournalDurabilityError("journal writer is poisoned")
        sequence = self._count
        _require(sequence < _MAX_RECORDS, "journal record count exceeds its bound")
        record = {
            "facts": facts,
            "kind": kind,
            "previousSha256": self._digest.hex(),
            "schemaVersion": 1,
            "sequence": sequence,
        }
        candidate = copy.deepcopy(self._state)
        _apply(candidate, record, sequence)
        frame = _record(sequence, self._digest, kind, facts)
        raw = (_MAGIC if sequence == 0 else b"") + frame
        _require(self._size + len(raw) <= _MAX_JOURNAL_BYTES, "journal bytes exceed their bound")
        try:
            before = self._io.handle_fact(self._handle)
            _validate_handle(before, self._io.final_path(self._handle), self._location, self._size)
            count = self._io.write(self._handle, self._size, raw)
            if type(count) is not int or count != len(raw):
                raise OSError("incomplete")
            self._io.flush(self._handle)
            after = self._io.handle_fact(self._handle)
            _validate_handle(
                after, self._io.final_path(self._handle), self._location, self._size + len(raw)
            )
        except BaseException as error:
            self._poisoned = True
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise RunJournalDurabilityError("journal append is not durable") from None
        self._state = candidate
        self._digest = hashlib.sha256(self._digest + frame[:4] + frame[4:-32]).digest()
        self._count += 1
        self._size += len(raw)
        return sequence

    def intend_filesystem(self, intent: FilesystemIntentV1) -> int:
        if type(intent) is not FilesystemIntentV1:
            raise TypeError("filesystem intent type differs")
        return self._append("filesystem_intent", _dict(intent))

    def bind_filesystem(self, ordinal: int, identity: FilesystemIdentityV1) -> None:
        if type(identity) is not FilesystemIdentityV1:
            raise TypeError("filesystem identity type differs")
        self._append(
            "filesystem_bound", {"identity": _dict(identity), "ordinal": _ordinal(ordinal)}
        )

    def record_filesystem_absent(self, ordinal: int) -> None:
        self._append("filesystem_absent", {"ordinal": _ordinal(ordinal)})

    def record_filesystem_removed(self, ordinal: int, identity: FilesystemIdentityV1) -> None:
        if type(identity) is not FilesystemIdentityV1:
            raise TypeError("filesystem identity type differs")
        self._append(
            "filesystem_removed", {"identity": _dict(identity), "ordinal": _ordinal(ordinal)}
        )

    def intend_process(self, intent: ProcessIntentV1) -> int:
        if type(intent) is not ProcessIntentV1:
            raise TypeError("process intent type differs")
        return self._append("process_intent", _dict(intent))

    def bind_suspended_process(self, ordinal: int, identity: ProcessIdentityFactsV1) -> None:
        if type(identity) is not ProcessIdentityFactsV1:
            raise TypeError("process identity type differs")
        self._append("process_bound", {"identity": _dict(identity), "ordinal": _ordinal(ordinal)})

    def intend_resume(self, ordinal: int) -> None:
        self._append("resume_intent", {"ordinal": _ordinal(ordinal)})

    def record_resumed(self, ordinal: int) -> None:
        self._append("resumed", {"ordinal": _ordinal(ordinal)})

    def record_process_exited(self, ordinal: int, identity: ProcessIdentityFactsV1) -> None:
        if type(identity) is not ProcessIdentityFactsV1:
            raise TypeError("process identity type differs")
        self._append("process_exited", {"identity": _dict(identity), "ordinal": _ordinal(ordinal)})

    def record_process_absent(self, ordinal: int) -> None:
        self._append("process_absent", {"ordinal": _ordinal(ordinal)})

    def record_sequence_complete(self) -> None:
        self._append("sequence_complete", {})


def _validate_handle(
    fact: _JournalHandleFactV1, final_path: str, location: RecoveryLocationFactsV1, size: int
) -> None:
    _require(
        type(fact) is _JournalHandleFactV1
        and fact.volume_serial == location.volume_serial
        and fact.file_id == location.journal_file_id
        and fact.size == size
        and not fact.reparse
        and not fact.directory
        and not fact.inheritable
        and fact.links == 1
        and PureWindowsPath(final_path) == PureWindowsPath(location.journal_path),
        "journal handle identity differs",
    )


def _parse(
    handle: int, binding: RunJournalBindingV1, location: RecoveryLocationFactsV1, io: Any
) -> _Parsed:
    before = io.handle_fact(handle)
    _validate_handle(before, io.final_path(handle), location, before.size)
    if before.size > _MAX_JOURNAL_BYTES:
        state = _new_state()
        return _Parsed(
            _facts(state, 0, before.size, integrity=False, partial=False, reason="size-bound"),
            state,
            _ZERO_DIGEST,
        )
    try:
        raw = io.read(handle, _MAX_JOURNAL_BYTES)
        after = io.handle_fact(handle)
        after_path = io.final_path(handle)
    except Exception:
        state = _new_state()
        return _Parsed(
            _facts(state, 0, before.size, integrity=False, partial=False, reason="read"),
            state,
            _ZERO_DIGEST,
        )
    try:
        _validate_handle(after, after_path, location, before.size)
        handle_changed = False
    except Exception:
        handle_changed = True
    if (
        type(raw) is not bytes
        or len(raw) > _MAX_JOURNAL_BYTES
        or handle_changed
        or len(raw) != before.size
    ):
        state = _new_state()
        return _Parsed(
            _facts(
                state,
                0,
                len(raw) if type(raw) is bytes else before.size,
                integrity=False,
                partial=False,
                reason="changed-while-read",
            ),
            state,
            _ZERO_DIGEST,
        )
    state, digest, count, offset = _new_state(), _ZERO_DIGEST, 0, 0
    if not raw.startswith(_MAGIC):
        return _Parsed(
            _facts(
                state,
                0,
                len(raw),
                integrity=False,
                partial=len(raw) < len(_MAGIC),
                reason="partial-tail" if len(raw) < len(_MAGIC) else "magic",
            ),
            state,
            digest,
        )
    offset = len(_MAGIC)
    reason: str | None = None
    partial = False
    while offset < len(raw):
        if count >= _MAX_RECORDS:
            reason = "record-bound"
            break
        if len(raw) - offset < 4:
            reason, partial = "partial-tail", True
            break
        size = struct.unpack(">I", raw[offset : offset + 4])[0]
        if not 0 < size <= _MAX_PAYLOAD_BYTES:
            reason = "frame-bound"
            break
        end = offset + 4 + size + 32
        if end > len(raw):
            reason, partial = "partial-tail", True
            break
        payload = raw[offset + 4 : offset + 4 + size]
        observed = raw[offset + 4 + size : end]
        expected = hashlib.sha256(digest + raw[offset : offset + 4] + payload).digest()
        if observed != expected:
            reason = "digest"
            break
        try:
            record = _strict_object(payload)
            _require(record.get("previousSha256") == digest.hex(), "journal digest chain differs")
            _apply(state, record, count)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            reason = "state"
            break
        digest, count, offset = observed, count + 1, end
    integrity = reason is None and offset == len(raw) and count > 0
    if integrity or state.binding is not None:
        _require(state.binding == binding and state.location == location, "journal binding differs")
    return _Parsed(
        _facts(state, count, len(raw), integrity=integrity, partial=partial, reason=reason),
        state,
        digest,
    )


def _create_run_journal(
    handle: int, binding: RunJournalBindingV1, location: RecoveryLocationFactsV1
) -> _RunJournalWriterV1:
    _require(type(handle) is int and handle > 0, "journal handle differs")
    _require(
        type(binding) is RunJournalBindingV1 and type(location) is RecoveryLocationFactsV1,
        "journal opening facts differ",
    )
    io = _journal_io()
    fact = io.handle_fact(handle)
    _validate_handle(fact, io.final_path(handle), location, 0)
    writer = _RunJournalWriterV1(handle, io, location, _new_state(), _ZERO_DIGEST, 0, 0)
    writer._append("opened", {"binding": _dict(binding), "location": _dict(location)})
    return writer


def _inspect_run_journal(
    handle: int, binding: RunJournalBindingV1, location: RecoveryLocationFactsV1
) -> RunJournalFactsV1:
    _require(type(handle) is int and handle > 0, "journal handle differs")
    return _parse(handle, binding, location, _journal_io()).facts


def _resume_run_journal(
    handle: int, binding: RunJournalBindingV1, location: RecoveryLocationFactsV1
) -> _RunJournalWriterV1:
    _require(type(handle) is int and handle > 0, "journal handle differs")
    io = _journal_io()
    parsed = _parse(handle, binding, location, io)
    _require(
        parsed.facts.integrity_complete and not parsed.facts.recorded_complete,
        "journal cannot be resumed",
    )
    return _RunJournalWriterV1(
        handle,
        io,
        location,
        parsed.state,
        parsed.digest,
        parsed.facts.frame_count,
        parsed.facts.byte_count,
    )


class _FileInfo(ctypes.Structure):
    _fields_ = (
        ("attributes", ctypes.c_uint32),
        ("times", ctypes.c_uint32 * 6),
        ("volume", ctypes.c_uint32),
        ("size_high", ctypes.c_uint32),
        ("size_low", ctypes.c_uint32),
        ("links", ctypes.c_uint32),
        ("index_high", ctypes.c_uint32),
        ("index_low", ctypes.c_uint32),
    )


class _WindowsJournalIo:
    def __init__(self) -> None:
        _require(
            os.name == "nt" and ctypes.sizeof(ctypes.c_void_p) == 8,
            "run journal requires 64-bit Windows",
        )
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        pointer, word = ctypes.c_void_p, ctypes.c_uint32
        signatures = (
            ("GetFileType", word, [pointer]),
            ("GetFileInformationByHandle", ctypes.c_int, [pointer, pointer]),
            ("GetHandleInformation", ctypes.c_int, [pointer, pointer]),
            ("GetFinalPathNameByHandleW", word, [pointer, ctypes.c_wchar_p, word, word]),
            ("SetFilePointerEx", ctypes.c_int, [pointer, ctypes.c_int64, pointer, word]),
            ("ReadFile", ctypes.c_int, [pointer, pointer, word, pointer, pointer]),
            ("WriteFile", ctypes.c_int, [pointer, pointer, word, pointer, pointer]),
            ("FlushFileBuffers", ctypes.c_int, [pointer]),
        )
        for name, result, arguments in signatures:
            function = getattr(self.kernel, name)
            function.restype = result
            function.argtypes = arguments

    def handle_fact(self, handle: int) -> _JournalHandleFactV1:
        info, flags = _FileInfo(), ctypes.c_uint32()
        _require(
            self.kernel.GetFileType(ctypes.c_void_p(handle)) == 1
            and bool(
                self.kernel.GetFileInformationByHandle(ctypes.c_void_p(handle), ctypes.byref(info))
            )
            and bool(
                self.kernel.GetHandleInformation(ctypes.c_void_p(handle), ctypes.byref(flags))
            ),
            "journal handle fact is unavailable",
        )
        return _JournalHandleFactV1(
            int(info.volume),
            (int(info.index_high) << 32) | int(info.index_low),
            (int(info.size_high) << 32) | int(info.size_low),
            bool(info.attributes & 0x400),
            int(info.links),
            bool(info.attributes & 0x10),
            bool(flags.value & 1),
        )

    def final_path(self, handle: int) -> str:
        value = ctypes.create_unicode_buffer(32768)
        count = self.kernel.GetFinalPathNameByHandleW(ctypes.c_void_p(handle), value, len(value), 0)
        _require(0 < count < len(value), "journal handle path is unavailable")
        raw = value.value.removeprefix("\\\\?\\")
        return PureWindowsPath(raw).as_posix()

    def read(self, handle: int, maximum: int) -> bytes:
        _require(
            bool(self.kernel.SetFilePointerEx(ctypes.c_void_p(handle), 0, None, 0)),
            "journal seek failed",
        )
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            size = min(remaining, 64 * 1024)
            buffer, count = ctypes.create_string_buffer(size), ctypes.c_uint32()
            _require(
                bool(
                    self.kernel.ReadFile(
                        ctypes.c_void_p(handle), buffer, size, ctypes.byref(count), None
                    )
                ),
                "journal read failed",
            )
            if count.value == 0:
                break
            _require(count.value <= size, "journal read count differs")
            chunks.append(buffer.raw[: count.value])
            remaining -= count.value
        return b"".join(chunks)

    def write(self, handle: int, offset: int, raw: bytes) -> int:
        _require(
            bool(self.kernel.SetFilePointerEx(ctypes.c_void_p(handle), offset, None, 0)),
            "journal seek failed",
        )
        buffer, count = ctypes.create_string_buffer(raw), ctypes.c_uint32()
        _require(
            bool(
                self.kernel.WriteFile(
                    ctypes.c_void_p(handle), buffer, len(raw), ctypes.byref(count), None
                )
            ),
            "journal write failed",
        )
        return int(count.value)

    def flush(self, handle: int) -> None:
        _require(
            bool(self.kernel.FlushFileBuffers(ctypes.c_void_p(handle))), "journal flush failed"
        )


def _journal_io() -> _WindowsJournalIo:
    return _WindowsJournalIo()
