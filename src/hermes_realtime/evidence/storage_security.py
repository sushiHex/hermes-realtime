"""Fail-closed artifact, marker, sentinel, and path primitives for the V1 store.

Nothing here opens a SQLite database; these are the ownership and encoding
primitives the writer transport must satisfy before any database byte exists.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import struct
import sys
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PureWindowsPath
from typing import Protocol
from uuid import UUID

from .models import (
    EvidenceModelError,
    SentinelState,
    WriterFault,
    parse_strict_json_object,
    validate_canonical_uuid4,
)

__all__ = [
    "ACTIVATION_MOVE_FLAGS",
    "ActivationTemporaryIdentityV1",
    "EVIDENCE_PHYSICAL_MAINTENANCE_CEILING_BYTES",
    "FILE_ATTRIBUTE_DIRECTORY",
    "FILE_ATTRIBUTE_REPARSE_POINT",
    "FILE_FLAG_BACKUP_SEMANTICS",
    "FILE_FLAG_OPEN_REPARSE_POINT",
    "FILE_SHARE_DELETE",
    "FILE_SHARE_READ",
    "FILE_SHARE_WRITE",
    "MANIFEST_V1",
    "MOVEFILE_REPLACE_EXISTING",
    "MOVEFILE_WRITE_THROUGH",
    "ROOT_HANDLE_ACCESS",
    "ROOT_HANDLE_FLAGS",
    "ROOT_HANDLE_SHARE_MODE",
    "EvidenceRootHandleV1",
    "ResolvedEvidenceManifestV1",
    "RootIdentityV1",
    "WindowsEvidenceRootHandleV1",
    "MAX_SENTINEL_GENERATION",
    "ROOT_MARKER_MANIFEST_VERSION",
    "ROOT_MARKER_MAX_BYTES",
    "REPOSITORY_MARKERS",
    "ROOT_MARKER_OWNER",
    "SENTINEL_MAGIC",
    "SENTINEL_SIZE",
    "SENTINEL_SLOT_OFFSETS",
    "SENTINEL_SLOT_SIZE",
    "SENTINEL_STATE_BYTES",
    "SENTINEL_VERSION",
    "EvidenceArtifactManifestV1",
    "EvidenceStorageError",
    "EvidenceStorageProbeV1",
    "FixedRepositoryBoundaryV1",
    "MarkerRepositoryBoundaryV1",
    "RepositoryBoundaryV1",
    "SentinelHandleV1",
    "SentinelImageV1",
    "SentinelSlotV1",
    "SentinelState",
    "ValidatedEvidenceRootV1",
    "WindowsStorageProbeV1",
    "WriterFault",
    "audit_root_occupancy",
    "check_maintenance_headroom",
    "decode_sentinel_image",
    "encode_root_marker",
    "encode_sentinel_slot",
    "initial_sentinel_image",
    "identify_activation_temporary",
    "next_sentinel_image",
    "next_sentinel_slot",
    "owned_allocated_bytes",
    "parse_root_marker",
    "path_is_at_or_below",
    "required_free_bytes",
    "resolve_for_containment",
    "validate_evidence_database_path",
]

EVIDENCE_PHYSICAL_MAINTENANCE_CEILING_BYTES = 320 * 1024 * 1024

ROOT_MARKER_OWNER = "hermes-realtime-evidence"
ROOT_MARKER_MANIFEST_VERSION = 1
ROOT_MARKER_MAX_BYTES = 4_096

SENTINEL_MAGIC = b"HRELOCK1"
SENTINEL_VERSION = 1
SENTINEL_SIZE = 512
SENTINEL_SLOT_SIZE = 248
MAX_SENTINEL_GENERATION = 2**64 - 1

_SLOT_BODY_SIZE = 216
_SENTINEL_PREFIX = SENTINEL_MAGIC + struct.pack(">I", SENTINEL_VERSION)
SENTINEL_STATE_BYTES: dict[SentinelState, int] = {
    SentinelState.CLEAR: 0,
    SentinelState.FULL_PURGE_PENDING: 1,
    SentinelState.CLOCK_ROLLBACK_PURGE_PENDING: 2,
    SentinelState.FIRST_CREATE_PENDING: 3,
}
_SENTINEL_STATES_BY_BYTE = {value: key for key, value in SENTINEL_STATE_BYTES.items()}


class EvidenceStorageError(Exception):
    """A fail-closed storage rejection carrying its exact public writer fault."""

    __slots__ = ("fault",)

    def __init__(self, fault: WriterFault, message: str) -> None:
        super().__init__(message)
        self.fault = fault


@dataclass(frozen=True, slots=True)
class EvidenceArtifactManifestV1:
    """The exact versioned relative names the transport may ever resolve."""

    version: int = 1
    root_marker: str = ".hermes-realtime-evidence-root-v1"
    root_marker_init: str = ".hermes-realtime-evidence-root-v1.init"
    sentinel: str = "capture-v1.owner"
    sentinel_init: str = "capture-v1.owner.init"
    database: str = "capture-v1.sqlite3"
    database_journal: str = "capture-v1.sqlite3-journal"
    database_wal: str = "capture-v1.sqlite3-wal"
    database_shm: str = "capture-v1.sqlite3-shm"
    database_vacuum: str = "capture-v1.sqlite3-vacuum"
    database_tmp: str = "capture-v1.sqlite3-tmp"

    @property
    def deletable_names(self) -> tuple[str, ...]:
        """The six database artifacts a full purge deletes and verifies absent."""

        return (
            self.database,
            self.database_journal,
            self.database_wal,
            self.database_shm,
            self.database_vacuum,
            self.database_tmp,
        )

    @property
    def retained_names(self) -> tuple[str, ...]:
        """The two names a verified full purge keeps."""

        return (self.root_marker, self.sentinel)

    @property
    def activation_temp_names(self) -> tuple[str, ...]:
        """The two pre-activation temporaries."""

        return (self.root_marker_init, self.sentinel_init)

    @property
    def all_names(self) -> tuple[str, ...]:
        return (
            self.root_marker,
            self.root_marker_init,
            self.sentinel,
            self.sentinel_init,
            *self.deletable_names,
        )

    def child(self, root: Path, name: str) -> Path:
        """Resolve one manifest name below ``root``; never a glob or recomputed basename."""

        if type(name) is not str or name not in self.all_names:
            raise EvidenceStorageError(
                WriterFault.PATH_INVALID,
                "only exact artifact-manifest names may be resolved",
            )
        return root / name

    def resolve(self, root: Path) -> ResolvedEvidenceManifestV1:
        """Resolve every relative name once into a retained capability."""

        return ResolvedEvidenceManifestV1(
            version=self.version,
            root_marker=self.child(root, self.root_marker),
            root_marker_init=self.child(root, self.root_marker_init),
            sentinel=self.child(root, self.sentinel),
            sentinel_init=self.child(root, self.sentinel_init),
            database=self.child(root, self.database),
            database_journal=self.child(root, self.database_journal),
            database_wal=self.child(root, self.database_wal),
            database_shm=self.child(root, self.database_shm),
            database_vacuum=self.child(root, self.database_vacuum),
            database_tmp=self.child(root, self.database_tmp),
        )


@dataclass(frozen=True, slots=True)
class ResolvedEvidenceManifestV1:
    """The manifest resolved once against a validated parent authority.

    The transport uses only these retained paths; it never recomputes a basename
    or enumerates the root with a glob.
    """

    version: int
    root_marker: Path
    root_marker_init: Path
    sentinel: Path
    sentinel_init: Path
    database: Path
    database_journal: Path
    database_wal: Path
    database_shm: Path
    database_vacuum: Path
    database_tmp: Path

    @property
    def deletable(self) -> tuple[Path, ...]:
        return (
            self.database,
            self.database_journal,
            self.database_wal,
            self.database_shm,
            self.database_vacuum,
            self.database_tmp,
        )

    @property
    def retained(self) -> tuple[Path, ...]:
        return (self.root_marker, self.sentinel)

    @property
    def activation_temps(self) -> tuple[Path, ...]:
        return (self.root_marker_init, self.sentinel_init)

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return (*self.retained, *self.activation_temps, *self.deletable)


MANIFEST_V1 = EvidenceArtifactManifestV1()


def _path_invalid(message: str) -> EvidenceStorageError:
    return EvidenceStorageError(WriterFault.PATH_INVALID, message)


def _canonical_root_id(value: object) -> str:
    try:
        return validate_canonical_uuid4(value, field_name="rootId")
    except EvidenceModelError as exc:
        raise _path_invalid("root marker rootId is not a canonical UUIDv4") from exc


def encode_root_marker(root_id: str) -> bytes:
    """Return the fixed canonical root-marker record plus its one terminal LF."""

    document = {
        "manifestVersion": ROOT_MARKER_MANIFEST_VERSION,
        "owner": ROOT_MARKER_OWNER,
        "rootId": _canonical_root_id(root_id),
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{encoded}\n".encode()


def parse_root_marker(raw: bytes) -> str:
    """Return the owned ``rootId``, or fail closed on any byte-level deviation."""

    if type(raw) is not bytes:
        raise _path_invalid("root marker must be exact built-in bytes")
    if len(raw) > ROOT_MARKER_MAX_BYTES or not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise _path_invalid("root marker must be one LF-terminated line")
    try:
        document = parse_strict_json_object(raw[:-1], max_bytes=ROOT_MARKER_MAX_BYTES)
    except EvidenceModelError as exc:
        raise _path_invalid("root marker is not a strict canonical JSON object") from exc
    if set(document) != {"manifestVersion", "owner", "rootId"}:
        raise _path_invalid("root marker has a missing or unknown key")
    if (
        type(document["manifestVersion"]) is not int
        or document["manifestVersion"] != ROOT_MARKER_MANIFEST_VERSION
    ):
        raise _path_invalid("root marker manifestVersion must be the exact integer 1")
    if type(document["owner"]) is not str or document["owner"] != ROOT_MARKER_OWNER:
        raise _path_invalid("root marker owner is not this package")
    root_id = _canonical_root_id(document["rootId"])
    if encode_root_marker(root_id) != raw:
        raise _path_invalid("root marker bytes are not the canonical spelling")
    return root_id


def _store_corrupt(message: str) -> EvidenceStorageError:
    return EvidenceStorageError(WriterFault.STORE_CORRUPT, message)


@dataclass(frozen=True, slots=True)
class SentinelSlotV1:
    """One decoded 248-byte sentinel slot."""

    generation: int
    state: SentinelState
    state_generation_id: str | None


@dataclass(frozen=True, slots=True)
class SentinelImageV1:
    """Both slots of a validated sentinel plus the winning slot."""

    slot_a: SentinelSlotV1 | None
    slot_b: SentinelSlotV1 | None
    active_index: int
    active: SentinelSlotV1


def _validated_slot(slot: object) -> SentinelSlotV1:
    if type(slot) is not SentinelSlotV1:
        raise _store_corrupt("sentinel slot must be an exact SentinelSlotV1")
    if type(slot.generation) is not int or not 0 <= slot.generation <= MAX_SENTINEL_GENERATION:
        raise _store_corrupt("sentinel generation is outside its unsigned 64-bit range")
    if type(slot.state) is not SentinelState:
        raise _store_corrupt("sentinel state must be an exact SentinelState")
    if slot.state is SentinelState.CLEAR:
        if slot.state_generation_id is not None:
            raise _store_corrupt("a clear sentinel slot carries no state UUID")
    else:
        if slot.generation == 0:
            raise _store_corrupt("generation zero is reserved for the unused clear slot")
        try:
            validate_canonical_uuid4(slot.state_generation_id, field_name="state_generation_id")
        except EvidenceModelError as exc:
            raise _store_corrupt("a pending sentinel slot requires a state UUID") from exc
    return slot


def encode_sentinel_slot(slot: SentinelSlotV1) -> bytes:
    """Encode one exact 248-byte slot including its trailing raw SHA-256."""

    validated = _validated_slot(slot)
    identifier = validated.state_generation_id
    body = (
        struct.pack(">Q", validated.generation)
        + bytes([SENTINEL_STATE_BYTES[validated.state]])
        + b"\x00" * 7
        + (b"\x00" * 16 if identifier is None else UUID(identifier).bytes)
        + b"\x00" * 184
    )
    return body + hashlib.sha256(_SENTINEL_PREFIX + body).digest()


def _decode_sentinel_slot(raw: bytes) -> SentinelSlotV1 | None:
    body, digest = raw[:_SLOT_BODY_SIZE], raw[_SLOT_BODY_SIZE:]
    if hashlib.sha256(_SENTINEL_PREFIX + body).digest() != digest:
        return None
    if body[9:16] != b"\x00" * 7 or body[32:] != b"\x00" * 184:
        return None
    state = _SENTINEL_STATES_BY_BYTE.get(body[8])
    if state is None:
        return None
    generation = struct.unpack(">Q", body[:8])[0]
    identifier = body[16:32]
    if state is SentinelState.CLEAR:
        return None if identifier != b"\x00" * 16 else SentinelSlotV1(generation, state, None)
    if identifier == b"\x00" * 16 or generation == 0:
        return None
    return SentinelSlotV1(generation, state, str(UUID(bytes=identifier)))


def decode_sentinel_image(raw: bytes) -> SentinelImageV1:
    """Decode both slots and select the highest valid generation, or fail closed."""

    if type(raw) is not bytes or len(raw) != SENTINEL_SIZE:
        raise _store_corrupt("sentinel must be exactly 512 bytes")
    if raw[:8] != SENTINEL_MAGIC or raw[8:12] != struct.pack(">I", SENTINEL_VERSION):
        raise _store_corrupt("sentinel magic or version is not HRELOCK1 v1")
    if raw[12:16] != b"\x00" * 4:
        raise _store_corrupt("sentinel reserved header bytes are not zero")
    slot_a = _decode_sentinel_slot(raw[16 : 16 + SENTINEL_SLOT_SIZE])
    slot_b = _decode_sentinel_slot(raw[16 + SENTINEL_SLOT_SIZE : SENTINEL_SIZE])
    if slot_a is None and slot_b is None:
        raise _store_corrupt("neither sentinel slot is valid")
    if (
        slot_a is not None
        and slot_b is not None
        and slot_a.generation == slot_b.generation
        and slot_a.generation != 0
    ):
        raise _store_corrupt("sentinel slots share an equal nonzero generation")
    if slot_b is None or (slot_a is not None and slot_a.generation > slot_b.generation):
        index, active = 0, slot_a
    else:
        index, active = 1, slot_b
    assert active is not None
    return SentinelImageV1(slot_a=slot_a, slot_b=slot_b, active_index=index, active=active)


def _sentinel_image_bytes(slot_a: SentinelSlotV1, slot_b: SentinelSlotV1) -> bytes:
    return (
        _SENTINEL_PREFIX
        + b"\x00" * 4
        + encode_sentinel_slot(slot_a)
        + encode_sentinel_slot(slot_b)
    )


def initial_sentinel_image(state_generation_id: str) -> bytes:
    """Build the first image: slot A generation 1 ``first_create_pending``, slot B clear."""

    return _sentinel_image_bytes(
        SentinelSlotV1(1, SentinelState.FIRST_CREATE_PENDING, state_generation_id),
        SentinelSlotV1(0, SentinelState.CLEAR, None),
    )


SENTINEL_SLOT_OFFSETS = (16, 16 + SENTINEL_SLOT_SIZE)


def next_sentinel_slot(
    raw: bytes,
    state: SentinelState,
    *,
    state_generation_id: str | None = None,
) -> tuple[int, bytes]:
    """Return the exact byte offset and 248-byte slot one transition must write.

    The returned offset always addresses the *inactive* slot, so an interrupted
    transition can never damage the durable predecessor the reader would select.
    """

    image = decode_sentinel_image(raw)
    if image.active.generation >= MAX_SENTINEL_GENERATION:
        raise _store_corrupt("sentinel generation would wrap around")
    successor = _validated_slot(
        SentinelSlotV1(image.active.generation + 1, state, state_generation_id)
    )
    return SENTINEL_SLOT_OFFSETS[1 - image.active_index], encode_sentinel_slot(successor)


def next_sentinel_image(
    raw: bytes,
    state: SentinelState,
    *,
    state_generation_id: str | None = None,
) -> bytes:
    """Return the whole image after applying one inactive-slot transition."""

    offset, slot = next_sentinel_slot(raw, state, state_generation_id=state_generation_id)
    return raw[:offset] + slot + raw[offset + SENTINEL_SLOT_SIZE :]


class SentinelHandleV1:
    """A retained, exclusively leased handle on the final sentinel.

    Initialization writes the complete 512-byte image to the pre-activation
    temporary; every later transition writes exactly one 248-byte slot in place and
    flushes that same handle, never replacing the locked inode.
    """

    __slots__ = ("_handle",)

    def __init__(self, path: Path) -> None:
        import msvcrt

        handle = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
        try:
            os.lseek(handle, 0, os.SEEK_SET)
            msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            os.close(handle)
            raise EvidenceStorageError(
                WriterFault.OWNERSHIP_UNAVAILABLE,
                "another owner holds the evidence sentinel lease",
            ) from exc
        self._handle = handle

    def read_image(self) -> bytes:
        os.lseek(self._handle, 0, os.SEEK_SET)
        return os.read(self._handle, SENTINEL_SIZE)

    def write_slot(self, offset: int, slot: bytes) -> None:
        if offset not in SENTINEL_SLOT_OFFSETS or len(slot) != SENTINEL_SLOT_SIZE:
            raise _store_corrupt("a sentinel transition must write one exact slot")
        os.lseek(self._handle, offset, os.SEEK_SET)
        written = 0
        while written < len(slot):
            written += os.write(self._handle, slot[written:])
        os.fsync(self._handle)

    def close(self) -> None:
        with suppress(OSError, ImportError):  # unlocking is best effort
            import msvcrt

            os.lseek(self._handle, 0, os.SEEK_SET)
            msvcrt.locking(self._handle, msvcrt.LK_UNLCK, 1)
        with suppress(OSError):  # closing is best effort
            os.close(self._handle)


class EvidenceStorageProbeV1(Protocol):
    """The physical Windows attributes path validation cannot derive from a string."""

    def platform_is_supported(self) -> bool: ...
    def volume_is_fixed_local(self, root: Path) -> bool: ...
    def path_has_reparse_point(self, path: Path) -> bool: ...
    def path_has_alternate_data_streams(self, path: Path) -> bool: ...
    def path_grants_only_owner(self, path: Path) -> bool: ...
    def allocated_bytes(self, path: Path) -> int: ...
    def volume_free_bytes(self, root: Path) -> int: ...


@dataclass(frozen=True, slots=True)
class ValidatedEvidenceRootV1:
    """An unbound producer-storage capability; never a profile capability.

    When a root-handle opener is supplied the validated root also retains the parent
    authority for the exact evidence root and the manifest resolved once against it.
    Path-shape validation alone leaves both unset, and :meth:`require_authority`
    then fails closed so no transport can operate without the retained authority.
    """

    root: Path
    database: Path
    manifest: EvidenceArtifactManifestV1
    authority: EvidenceRootHandleV1 | None = None
    resolved: ResolvedEvidenceManifestV1 | None = None

    def require_authority(self) -> tuple[EvidenceRootHandleV1, ResolvedEvidenceManifestV1]:
        if self.authority is None or self.resolved is None:
            raise _path_invalid("the evidence root retains no validated parent authority")
        return self.authority, self.resolved


REPOSITORY_MARKERS: tuple[str, ...] = (".git", ".hg", ".svn", "pyproject.toml", "uv.lock")


class RepositoryBoundaryV1(Protocol):
    """Answers whether a candidate path lies inside a source checkout."""

    def repository_root_for(self, path: Path) -> Path | None: ...


def _containment_parts(path: Path) -> tuple[str, ...]:
    """Windows-semantics case-insensitive components for containment tests."""

    return tuple(part.casefold() for part in PureWindowsPath(str(path)).parts)


def path_is_at_or_below(ancestor: Path, candidate: Path) -> bool:
    """Component-wise containment; never a raw string prefix."""

    above = _containment_parts(ancestor)
    below = _containment_parts(candidate)
    return len(below) >= len(above) and below[: len(above)] == above


class MarkerRepositoryBoundaryV1:
    """Detects a checkout by walking up for repository markers; no subprocess."""

    __slots__ = ("_markers",)

    def __init__(self, markers: tuple[str, ...] = REPOSITORY_MARKERS) -> None:
        self._markers = markers

    def repository_root_for(self, path: Path) -> Path | None:
        for candidate in (path, *path.parents):
            for marker in self._markers:
                try:
                    if (candidate / marker).exists():
                        return candidate
                except OSError:  # pragma: no cover - an unreadable ancestor is not a root
                    continue
        return None


class FixedRepositoryBoundaryV1:
    """An explicit deterministic set of repository roots supplied by the host."""

    __slots__ = ("_roots",)

    def __init__(self, roots: Iterable[Path]) -> None:
        self._roots = tuple(roots)

    def repository_root_for(self, path: Path) -> Path | None:
        for root in self._roots:
            if path_is_at_or_below(root, path):
                return root
        return None


def resolve_for_containment(path: Path) -> Path:
    """Resolve a candidate for containment checks, failing closed on any error."""

    try:
        return Path(os.path.realpath(path))
    except (OSError, ValueError) as exc:
        raise _path_invalid("the evidence path cannot be resolved") from exc


def _reject_repository_containment(
    root: Path,
    *,
    boundary: RepositoryBoundaryV1,
    resolver: Callable[[Path], Path],
) -> None:
    """Reject a repository-contained target lexically and after resolution (§4.3)."""

    try:
        resolved = resolver(root)
    except EvidenceStorageError:
        raise
    except (OSError, ValueError) as exc:
        raise _path_invalid("the evidence path cannot be resolved") from exc
    for candidate in (root, resolved):
        if boundary.repository_root_for(candidate) is not None:
            raise _path_invalid("the evidence root is contained in a source repository")


_DRIVE_LETTER = re.compile(r"\A[A-Za-z]:\Z")
_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{digit}" for digit in range(1, 10)}
    | {f"LPT{digit}" for digit in range(1, 10)}
)
_FORBIDDEN_COMPONENT_CHARACTERS = frozenset('<>:"|?*/\\')


def _reject_component(component: str) -> None:
    if not component:
        raise _path_invalid("evidence paths contain no empty component")
    if component in {".", ".."}:
        raise _path_invalid("evidence paths contain no relative component")
    if component[-1] in {" ", "."}:
        raise _path_invalid("evidence path components end in no space or dot")
    if any(character in _FORBIDDEN_COMPONENT_CHARACTERS for character in component):
        raise _path_invalid("evidence path components contain no reserved character")
    if any(ord(character) < 32 for character in component):
        raise _path_invalid("evidence path components contain no control character")
    if component.split(".", 1)[0].upper() in _RESERVED_DEVICE_NAMES:
        raise _path_invalid("evidence path components are never DOS device names")


def validate_evidence_database_path(
    path: Path,
    *,
    probe: EvidenceStorageProbeV1,
    manifest: EvidenceArtifactManifestV1 = MANIFEST_V1,
    boundary: RepositoryBoundaryV1 | None = None,
    resolver: Callable[[Path], Path] = resolve_for_containment,
    root_handle_opener: Callable[[Path], EvidenceRootHandleV1] | None = None,
) -> ValidatedEvidenceRootV1:
    """Validate the operator-selected database file and return its owned root."""

    if not probe.platform_is_supported():
        raise EvidenceStorageError(
            WriterFault.UNSUPPORTED_PLATFORM,
            "Windows is the only Slice 0 evidence-storage platform",
        )
    pure = PureWindowsPath(str(path))
    if _DRIVE_LETTER.match(pure.drive) is None or pure.root != "\\":
        raise _path_invalid("evidence paths are absolute local drive-letter paths only")
    parts = pure.parts[1:]
    if len(parts) < 2:
        raise _path_invalid("the evidence root is never a volume root")
    for component in parts:
        _reject_component(component)
    if parts[-1] != manifest.database:
        raise _path_invalid("the evidence database basename is exactly capture-v1.sqlite3")

    root = Path(str(pure.parent))
    _reject_repository_containment(
        root,
        boundary=MarkerRepositoryBoundaryV1() if boundary is None else boundary,
        resolver=resolver,
    )
    if not probe.volume_is_fixed_local(root):
        raise _path_invalid("the evidence root must live on a fixed local volume")
    for candidate in (root, *root.parents):
        if probe.path_has_reparse_point(candidate):
            raise _path_invalid("no evidence path component may be a reparse point")
    if probe.path_has_alternate_data_streams(root):
        raise _path_invalid("the evidence root carries no alternate data stream")
    if not probe.path_grants_only_owner(root):
        raise _path_invalid("the evidence root grants access beyond its owner")

    # The final database leaf is validated last and never opened: an existing
    # capture-v1.sqlite3 that is itself a reparse point, carries an alternate
    # stream, or grants access beyond its owner is refused. The probe reports the
    # benign answer for a leaf that does not exist yet.
    database = Path(str(pure))
    if probe.path_has_reparse_point(database):
        raise _path_invalid("the evidence database may not be a reparse point")
    if probe.path_has_alternate_data_streams(database):
        raise _path_invalid("the evidence database carries no alternate data stream")
    if not probe.path_grants_only_owner(database):
        raise _path_invalid("the evidence database grants access beyond its owner")
    if root_handle_opener is None:
        return ValidatedEvidenceRootV1(root=root, database=database, manifest=manifest)

    # Retain the parent authority and resolve every manifest name exactly once while
    # that authority is known valid.
    authority = root_handle_opener(root)
    try:
        resolved = manifest.resolve(root)
        if resolved.database != database:  # pragma: no cover - the manifest is fixed
            raise _path_invalid("the resolved manifest disagrees with the validated database")
    except BaseException:
        authority.close()
        raise
    return ValidatedEvidenceRootV1(
        root=root,
        database=database,
        manifest=manifest,
        authority=authority,
        resolved=resolved,
    )


def audit_root_occupancy(
    root: Path,
    *,
    manifest: EvidenceArtifactManifestV1 = MANIFEST_V1,
) -> None:
    """Fail closed unless every leaf occupant is authorised by §§3.7 and 5.1.

    The gate applies only while ownership is still being established. Once a valid
    final root marker *and* the final sentinel both exist the store is owned, later
    occupants are never re-litigated, and nothing outside the six deletable names is
    ever removed -- so runner-seeded decoys survive untouched.
    """

    try:
        entries = {entry.name for entry in root.iterdir()}
    except FileNotFoundError:
        return
    except OSError as exc:
        raise _path_invalid("the evidence root cannot be enumerated") from exc

    marker = root / manifest.root_marker
    marker_valid = False
    if marker.is_file():
        try:
            parse_root_marker(marker.read_bytes())
        except EvidenceStorageError:
            marker_valid = False
        else:
            marker_valid = True

    if not marker_valid:
        # An absent or invalid marker may only ever be accompanied by its own
        # pre-activation temporary; package code never takes ownership otherwise.
        allowed = {manifest.root_marker_init}
    elif manifest.sentinel in entries:
        return
    else:
        allowed = {
            manifest.root_marker,
            manifest.root_marker_init,
            manifest.sentinel_init,
            *manifest.deletable_names,
        }
    if entries - allowed:
        raise _path_invalid("the evidence root holds an unauthorised occupant")


def required_free_bytes(owned_allocated: int) -> int:
    """The exact ``max(0, ceiling - owned_allocated)`` preflight requirement."""

    if type(owned_allocated) is not int or owned_allocated < 0:
        raise _path_invalid("owned allocation must be a nonnegative exact integer")
    return max(0, EVIDENCE_PHYSICAL_MAINTENANCE_CEILING_BYTES - owned_allocated)


def owned_allocated_bytes(
    root: Path,
    *,
    probe: EvidenceStorageProbeV1,
    manifest: EvidenceArtifactManifestV1 = MANIFEST_V1,
) -> int:
    """Sum the filesystem allocation of exactly the six deletable names, once each."""

    return sum(
        probe.allocated_bytes(manifest.child(root, name)) for name in manifest.deletable_names
    )


def check_maintenance_headroom(
    root: Path,
    *,
    probe: EvidenceStorageProbeV1,
    manifest: EvidenceArtifactManifestV1 = MANIFEST_V1,
) -> None:
    """Fail closed when the volume cannot hold one more full maintenance instant."""

    owned = owned_allocated_bytes(root, probe=probe, manifest=manifest)
    if owned > EVIDENCE_PHYSICAL_MAINTENANCE_CEILING_BYTES:
        raise EvidenceStorageError(
            WriterFault.QUOTA_UNAVAILABLE,
            "the owned evidence artifacts exceed the physical maintenance ceiling",
        )
    if probe.volume_free_bytes(root) < required_free_bytes(owned):
        raise EvidenceStorageError(
            WriterFault.QUOTA_UNAVAILABLE,
            "the volume lacks the required maintenance free space",
        )


FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_DRIVE_FIXED = 3
_INVALID_HANDLES = frozenset({0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF})
_INVALID_FILE_SIZE = 0xFFFFFFFF
# ERROR_HANDLE_EOF means "this object has no streams to enumerate"; a file-not-found
# race means the artifact is already gone. Any other failure must fail closed.
_NO_MORE_STREAM_ERRORS = frozenset({2, 3, 38})
_OWNER_SECURITY_INFORMATION = 0x00000001
_DACL_SECURITY_INFORMATION = 0x00000004
_SE_FILE_OBJECT = 1
_ACCESS_ALLOWED_ACE_TYPE = 0x00
_ACCESS_ALLOWED_OBJECT_ACE_TYPE = 0x05
_INHERIT_ONLY_ACE = 0x08
# ACCESS_ALLOWED_ACE is header(4) + mask(4) + SidStart. ACCESS_ALLOWED_OBJECT_ACE
# inserts Flags(4) and up to two conditional 16-byte GUIDs before its SidStart.
_ACCESS_ALLOWED_ACE_SID_OFFSET = 8
_ACCESS_ALLOWED_OBJECT_ACE_SID_OFFSET = 12
_ACE_OBJECT_TYPE_PRESENT = 0x00000001
_ACE_INHERITED_OBJECT_TYPE_PRESENT = 0x00000002
_GUID_BYTES = 16
_ACL_SIZE_INFORMATION = 2
_ALWAYS_PERMITTED_SIDS = frozenset({"S-1-5-18", "S-1-5-32-544", "S-1-3-0", "S-1-3-4"})
_MAX_STREAM_NAME = 260 + 36


class _StreamData(ctypes.Structure):
    _fields_ = (
        ("StreamSize", ctypes.c_longlong),
        ("cStreamName", ctypes.c_wchar * _MAX_STREAM_NAME),
    )


class _AclSizeInformation(ctypes.Structure):
    _fields_ = (
        ("AceCount", ctypes.c_uint32),
        ("AclBytesInUse", ctypes.c_uint32),
        ("AclBytesFree", ctypes.c_uint32),
    )


class _AceHeader(ctypes.Structure):
    _fields_ = (
        ("AceType", ctypes.c_ubyte),
        ("AceFlags", ctypes.c_ubyte),
        ("AceSize", ctypes.c_uint16),
    )


@cache
def _kernel32() -> ctypes.WinDLL:
    library = ctypes.WinDLL("kernel32", use_last_error=True)
    library.GetDriveTypeW.argtypes = (ctypes.c_wchar_p,)
    library.GetDriveTypeW.restype = ctypes.c_uint32
    library.FindFirstStreamW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    library.FindFirstStreamW.restype = ctypes.c_void_p
    library.FindNextStreamW.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    library.FindNextStreamW.restype = ctypes.c_int
    library.FindClose.argtypes = (ctypes.c_void_p,)
    library.FindClose.restype = ctypes.c_int
    library.GetCompressedFileSizeW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_uint32),
    )
    library.GetCompressedFileSizeW.restype = ctypes.c_uint32
    library.GetDiskFreeSpaceExW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    library.GetDiskFreeSpaceExW.restype = ctypes.c_int
    library.LocalFree.argtypes = (ctypes.c_void_p,)
    library.LocalFree.restype = ctypes.c_void_p
    library.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    library.CreateFileW.restype = ctypes.c_void_p
    library.CloseHandle.argtypes = (ctypes.c_void_p,)
    library.CloseHandle.restype = ctypes.c_int
    library.GetFileInformationByHandle.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    library.GetFileInformationByHandle.restype = ctypes.c_int
    library.GetFileInformationByHandleEx.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    library.GetFileInformationByHandleEx.restype = ctypes.c_int
    library.GetFinalPathNameByHandleW.argtypes = (
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )
    library.GetFinalPathNameByHandleW.restype = ctypes.c_uint32
    library.MoveFileExW.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    library.MoveFileExW.restype = ctypes.c_int
    library.SetFileInformationByHandle.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    library.SetFileInformationByHandle.restype = ctypes.c_int
    library.FlushFileBuffers.argtypes = (ctypes.c_void_p,)
    library.FlushFileBuffers.restype = ctypes.c_int
    return library


@cache
def _advapi32() -> ctypes.WinDLL:
    library = ctypes.WinDLL("advapi32", use_last_error=True)
    library.GetNamedSecurityInfoW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    )
    library.GetNamedSecurityInfoW.restype = ctypes.c_uint32
    library.GetAclInformation.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
    )
    library.GetAclInformation.restype = ctypes.c_int
    library.GetAce.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
    library.GetAce.restype = ctypes.c_int
    library.ConvertSidToStringSidW.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_wchar_p),
    )
    library.ConvertSidToStringSidW.restype = ctypes.c_int
    return library


def _sid_to_string(sid: ctypes.c_void_p) -> str:
    text = ctypes.c_wchar_p()
    if not _advapi32().ConvertSidToStringSidW(sid, ctypes.byref(text)):
        return ""
    try:
        return text.value or ""
    finally:
        _kernel32().LocalFree(text)


def _granted_sids(dacl: ctypes.c_void_p) -> tuple[str, ...]:
    advapi = _advapi32()
    information = _AclSizeInformation()
    if not advapi.GetAclInformation(
        dacl,
        ctypes.byref(information),
        ctypes.sizeof(information),
        _ACL_SIZE_INFORMATION,
    ):
        return ("",)
    granted: list[str] = []
    for index in range(int(information.AceCount)):
        ace = ctypes.c_void_p()
        if not advapi.GetAce(dacl, index, ctypes.byref(ace)):
            return ("",)
        header = ctypes.cast(ace, ctypes.POINTER(_AceHeader)).contents
        if header.AceType not in (_ACCESS_ALLOWED_ACE_TYPE, _ACCESS_ALLOWED_OBJECT_ACE_TYPE):
            continue
        if header.AceFlags & _INHERIT_ONLY_ACE:
            continue
        offset = _ace_sid_offset(ace, header.AceType)
        if offset is None:
            return ("",)
        granted.append(_sid_to_string(ctypes.c_void_p((ace.value or 0) + offset)))
    return tuple(granted)


def _ace_sid_offset(ace: ctypes.c_void_p, ace_type: int) -> int | None:
    """Byte offset of SidStart within this ACE, or None when it cannot be read."""

    if ace_type == _ACCESS_ALLOWED_ACE_TYPE:
        return _ACCESS_ALLOWED_ACE_SID_OFFSET
    if ace_type != _ACCESS_ALLOWED_OBJECT_ACE_TYPE:
        return None
    address = ace.value or 0
    if not address:
        return None
    flags = ctypes.cast(
        ctypes.c_void_p(address + _ACCESS_ALLOWED_ACE_SID_OFFSET),
        ctypes.POINTER(ctypes.c_uint32),
    ).contents.value
    offset = _ACCESS_ALLOWED_OBJECT_ACE_SID_OFFSET
    if flags & _ACE_OBJECT_TYPE_PRESENT:
        offset += _GUID_BYTES
    if flags & _ACE_INHERITED_OBJECT_TYPE_PRESENT:
        offset += _GUID_BYTES
    return offset


def _path_grants_only_owner(path: Path) -> bool:
    """Whether this path's DACL grants nothing beyond its owner and the system SIDs."""

    if not os.path.lexists(path):
        return True
    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    status = _advapi32().GetNamedSecurityInfoW(
        str(path),
        _SE_FILE_OBJECT,
        _OWNER_SECURITY_INFORMATION | _DACL_SECURITY_INFORMATION,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if status != 0:
        # GetNamedSecurityInfoW allocates the descriptor only on success, so
        # there is nothing to release on this path.
        return False
    try:
        # A NULL DACL grants everyone access; the descriptor was still allocated
        # and must be released before reporting that.
        if not dacl:
            return False
        permitted = _ALWAYS_PERMITTED_SIDS | {_sid_to_string(owner)}
        return all(granted in permitted for granted in _granted_sids(dacl))
    finally:
        _kernel32().LocalFree(descriptor)


def _require_owner_only_dacl(path: Path) -> None:
    """Fail closed unless this exact path still grants nothing beyond its owner."""

    if not _path_grants_only_owner(path):
        raise _path_invalid("the evidence root grants access beyond its owner")


GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
_DELETE_ACCESS = 0x00010000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
FILE_SHARE_DELETE = 0x00000004
OPEN_EXISTING = 3
CREATE_NEW = 1
FILE_ATTRIBUTE_NORMAL = 0x00000080
FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
FILE_ATTRIBUTE_DIRECTORY = 0x00000010
MOVEFILE_REPLACE_EXISTING = 0x00000001
MOVEFILE_WRITE_THROUGH = 0x00000008
_ERROR_FILE_EXISTS = 80
_ERROR_ALREADY_EXISTS = 183
_FILE_RENAME_INFORMATION_CLASS = 3
_FILE_DISPOSITION_INFORMATION_CLASS = 4
_FILE_STREAM_INFORMATION_CLASS = 7
_FILE_STREAM_INFORMATION_BUFFER_BYTES = 65_536

#: Deliberately excludes FILE_SHARE_DELETE so the validated root cannot be renamed
#: or deleted out from under this owner while the authority is retained.
ROOT_HANDLE_ACCESS = GENERIC_READ
ROOT_HANDLE_SHARE_MODE = FILE_SHARE_READ | FILE_SHARE_WRITE
ROOT_HANDLE_FLAGS = FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT
#: Activation never replaces an existing final name and is flushed write-through.
ACTIVATION_MOVE_FLAGS = MOVEFILE_WRITE_THROUGH

_FILE_NAME_NORMALIZED = 0x0
_MAX_FINAL_PATH = 32_768


@dataclass(frozen=True, slots=True)
class RootIdentityV1:
    """The stable Windows identity of a retained evidence-root handle."""

    volume_serial_number: int
    file_index: int
    attributes: int
    final_path: str


class EvidenceRootHandleV1(Protocol):
    """A retained parent authority for the exact validated evidence root."""

    @property
    def identity(self) -> RootIdentityV1: ...
    def revalidate(self) -> RootIdentityV1: ...
    def compare_fresh(self) -> None: ...
    def move_no_replace_write_through(self, source: Path, destination: Path) -> None: ...
    def activate_exact_temporary(
        self,
        temporary: Path,
        final: Path,
        retained: ActivationTemporaryV1,
        expected: bytes,
    ) -> None: ...
    def close(self) -> None: ...


class _FileTime(ctypes.Structure):
    """FILETIME is two DWORDs; modelling it as a 64-bit field would misalign the rest."""

    _fields_ = (("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32))


class _ByHandleFileInformation(ctypes.Structure):
    _fields_ = (
        ("dwFileAttributes", ctypes.c_uint32),
        ("ftCreationTime", _FileTime),
        ("ftLastAccessTime", _FileTime),
        ("ftLastWriteTime", _FileTime),
        ("dwVolumeSerialNumber", ctypes.c_uint32),
        ("nFileSizeHigh", ctypes.c_uint32),
        ("nFileSizeLow", ctypes.c_uint32),
        ("nNumberOfLinks", ctypes.c_uint32),
        ("nFileIndexHigh", ctypes.c_uint32),
        ("nFileIndexLow", ctypes.c_uint32),
    )


@dataclass(frozen=True, slots=True)
class ActivationTemporaryIdentityV1:
    """Stable identity captured while an activation temporary is still open."""

    device: int
    inode: int


@dataclass(slots=True)
class ActivationTemporaryV1:
    """The one exclusive-create descriptor retained until activation terminates."""

    identity: ActivationTemporaryIdentityV1
    attributes: int
    _descriptor: int | None

    def require_descriptor(self) -> int:
        if self._descriptor is None:
            raise _path_invalid("the activation temporary descriptor is no longer retained")
        return self._descriptor

    def close(self) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            os.close(descriptor)


def identify_activation_temporary(file_descriptor: int) -> ActivationTemporaryIdentityV1:
    """Capture the exact temporary identity before its exclusive-create handle closes."""

    identity = os.fstat(file_descriptor)
    return ActivationTemporaryIdentityV1(device=int(identity.st_dev), inode=int(identity.st_ino))


def create_activation_temporary(path: Path) -> ActivationTemporaryV1:
    """Create one no-replace leaf whose restrictive handle stays owned by the caller."""

    if sys.platform != "win32":
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as exc:
            raise EvidenceStorageError(
                WriterFault.OWNERSHIP_UNAVAILABLE,
                "another contender already owns this activation temporary",
            ) from exc
        except OSError as exc:
            raise _path_invalid("the evidence root does not accept an exclusive create") from exc
        return ActivationTemporaryV1(
            identity=identify_activation_temporary(descriptor),
            attributes=0,
            _descriptor=descriptor,
        )

    import msvcrt

    kernel = _kernel32()
    handle = kernel.CreateFileW(
        str(path),
        GENERIC_READ | GENERIC_WRITE | _DELETE_ACCESS,
        FILE_SHARE_READ,
        None,
        CREATE_NEW,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle is None or handle in _INVALID_HANDLES:
        error = ctypes.get_last_error()
        if error in (_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS):
            raise EvidenceStorageError(
                WriterFault.OWNERSHIP_UNAVAILABLE,
                "another contender already owns this activation temporary",
            ) from ctypes.WinError(error)
        raise _path_invalid(
            "the evidence root does not accept an exclusive create"
        ) from ctypes.WinError(error)
    try:
        descriptor = msvcrt.open_osfhandle(
            int(handle),
            os.O_RDWR | getattr(os, "O_BINARY", 0),
        )
    except OSError as exc:
        with suppress(OSError):
            kernel.CloseHandle(handle)
        raise _path_invalid("the activation temporary descriptor cannot be retained") from exc
    try:
        information = _ByHandleFileInformation()
        raw_handle = msvcrt.get_osfhandle(descriptor)
        if not kernel.GetFileInformationByHandle(raw_handle, ctypes.byref(information)):
            raise _path_invalid("the activation temporary identity cannot be captured")
        return ActivationTemporaryV1(
            identity=identify_activation_temporary(descriptor),
            attributes=int(information.dwFileAttributes),
            _descriptor=descriptor,
        )
    except BaseException:
        os.close(descriptor)
        raise


class _FileDispositionInformation(ctypes.Structure):
    _fields_ = (("DeleteFile", ctypes.c_int),)


class _FileStreamInformation(ctypes.Structure):
    _fields_ = (
        ("NextEntryOffset", ctypes.c_uint32),
        ("StreamNameLength", ctypes.c_uint32),
        ("StreamSize", ctypes.c_int64),
        ("StreamAllocationSize", ctypes.c_int64),
        ("StreamName", ctypes.c_wchar * 1),
    )


class _FileRenameOptions(ctypes.Union):
    _fields_ = (("ReplaceIfExists", ctypes.c_ubyte), ("Flags", ctypes.c_uint32))


class _FileRenameInformation(ctypes.Structure):
    _fields_ = (
        ("Options", _FileRenameOptions),
        ("RootDirectory", ctypes.c_void_p),
        ("FileNameLength", ctypes.c_uint32),
        ("FileName", ctypes.c_wchar * 1),
    )


def _rename_information(destination: Path) -> tuple[ctypes.Array[ctypes.c_char], int]:
    """Build a zero-padded full-path FILE_RENAME_INFO buffer."""

    encoded = str(destination).encode("utf-16-le")
    offset = _FileRenameInformation.FileName.offset
    size = max(ctypes.sizeof(_FileRenameInformation), offset + len(encoded) + 2)
    buffer = ctypes.create_string_buffer(size)
    information = _FileRenameInformation.from_buffer(buffer)
    information.Options.Flags = 0
    information.RootDirectory = None
    information.FileNameLength = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + offset, encoded, len(encoded))
    return buffer, size


def _normalized_final_path(text: str) -> str:
    for prefix in ("\\\\?\\UNC\\", "\\\\?\\"):
        if text.startswith(prefix):
            return text[len(prefix) :] if prefix == "\\\\?\\" else f"\\\\{text[len(prefix):]}"
    return text


def _open_root_authority_handle(root: Path) -> int:
    kernel = _kernel32()
    handle = kernel.CreateFileW(
        str(root),
        ROOT_HANDLE_ACCESS,
        ROOT_HANDLE_SHARE_MODE,
        None,
        OPEN_EXISTING,
        ROOT_HANDLE_FLAGS,
        None,
    )
    if handle is None or handle in _INVALID_HANDLES:
        raise _path_invalid("the evidence root cannot be opened as a parent authority")
    return int(handle)


def _read_root_identity(handle: int, *, expected: Path) -> RootIdentityV1:
    kernel = _kernel32()
    information = _ByHandleFileInformation()
    if not kernel.GetFileInformationByHandle(handle, ctypes.byref(information)):
        raise _path_invalid("the evidence root identity cannot be read")
    buffer = ctypes.create_unicode_buffer(_MAX_FINAL_PATH)
    length = int(
        kernel.GetFinalPathNameByHandleW(handle, buffer, _MAX_FINAL_PATH, _FILE_NAME_NORMALIZED)
    )
    if length == 0 or length >= _MAX_FINAL_PATH:
        raise _path_invalid("the evidence root final path cannot be read")
    final_path = _normalized_final_path(buffer.value)
    attributes = int(information.dwFileAttributes)
    if not attributes & FILE_ATTRIBUTE_DIRECTORY:
        raise _path_invalid("the evidence root authority is not a directory")
    if attributes & FILE_ATTRIBUTE_REPARSE_POINT:
        raise _path_invalid("the evidence root authority is a reparse point")
    if _containment_parts(Path(final_path)) != _containment_parts(expected):
        raise _path_invalid("the evidence root authority escaped its configured path")
    return RootIdentityV1(
        volume_serial_number=int(information.dwVolumeSerialNumber),
        file_index=(int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow),
        attributes=attributes,
        final_path=final_path,
    )


class WindowsEvidenceRootHandleV1:
    """The production parent authority; real Win32, closed safely and idempotently."""

    __slots__ = ("_root", "_handle", "_identity")

    def __init__(self, root: Path) -> None:
        self._root = root
        self._handle: int | None = _open_root_authority_handle(root)
        try:
            self._identity = _read_root_identity(self._handle, expected=root)
            _require_owner_only_dacl(Path(self._identity.final_path))
        except BaseException:
            self.close()
            raise

    @property
    def identity(self) -> RootIdentityV1:
        return self._identity

    def _require_handle(self) -> int:
        if self._handle is None:
            raise _path_invalid("the evidence root authority is closed")
        return self._handle

    def revalidate(self) -> RootIdentityV1:
        """Re-read the retained handle: identity and DACL must both still hold.

        The DACL is re-queried from this handle's current final path on every call,
        so a grant broadened after the handle was opened is refused before any
        activation rather than trusted from construction time.
        """

        current = _read_root_identity(self._require_handle(), expected=self._root)
        if current != self._identity:
            raise _path_invalid("the retained evidence root authority changed identity")
        _require_owner_only_dacl(Path(current.final_path))
        return current

    def compare_fresh(self) -> None:
        """Open a second no-delete-sharing handle; identity and DACL must match."""

        handle = _open_root_authority_handle(self._root)
        try:
            fresh = _read_root_identity(handle, expected=self._root)
        finally:
            with suppress(OSError):
                _kernel32().CloseHandle(handle)
        if (
            fresh.volume_serial_number != self._identity.volume_serial_number
            or fresh.file_index != self._identity.file_index
            or fresh.final_path != self._identity.final_path
        ):
            raise _path_invalid("the configured evidence root no longer names the same object")
        _require_owner_only_dacl(self._root)
        _require_owner_only_dacl(Path(fresh.final_path))

    def move_no_replace_write_through(self, source: Path, destination: Path) -> None:
        """Activate a temporary with Windows no-replace write-through semantics."""

        self._require_handle()
        if not _kernel32().MoveFileExW(str(source), str(destination), ACTIVATION_MOVE_FLAGS):
            error = ctypes.get_last_error()
            if error in (_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS):
                raise EvidenceStorageError(
                    WriterFault.OWNERSHIP_UNAVAILABLE,
                    "another contender already activated this evidence artifact",
                ) from ctypes.WinError(error)
            raise _path_invalid(
                "the evidence artifact could not be activated without replacing"
            ) from ctypes.WinError(error)

    def activate_exact_temporary(
        self,
        temporary: Path,
        final: Path,
        retained: ActivationTemporaryV1,
        expected: bytes,
    ) -> None:
        """Validate and activate the exclusive-create handle, or dispose its loser link."""

        import msvcrt

        self.revalidate()
        retained_root = _containment_parts(Path(self._identity.final_path))
        if (
            _containment_parts(temporary.parent) != retained_root
            or _containment_parts(final.parent) != retained_root
            or temporary.name == final.name
        ):
            raise _path_invalid("the activation paths escaped the retained evidence root")
        descriptor = retained.require_descriptor()
        raw_handle = msvcrt.get_osfhandle(descriptor)
        kernel = _kernel32()

        def validate_retained_handle() -> None:
            information = _ByHandleFileInformation()
            if not kernel.GetFileInformationByHandle(
                raw_handle,
                ctypes.byref(information),
            ):
                raise _path_invalid("the activation temporary identity cannot be read")
            attributes = int(information.dwFileAttributes)
            if (
                attributes != retained.attributes
                or attributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)
            ):
                raise _path_invalid("the activation temporary attributes changed")
            if int(information.nNumberOfLinks) != 1:
                raise _path_invalid("the activation temporary gained another hard link")
            actual_identity = identify_activation_temporary(descriptor)
            identity = retained.identity
            if (
                identity.device <= 0
                or identity.inode <= 0
                or actual_identity.device <= 0
                or actual_identity.inode <= 0
                or actual_identity != identity
            ):
                raise _path_invalid("the activation temporary changed identity")

            streams = ctypes.create_string_buffer(_FILE_STREAM_INFORMATION_BUFFER_BYTES)
            ctypes.set_last_error(0)
            if not kernel.GetFileInformationByHandleEx(
                raw_handle,
                _FILE_STREAM_INFORMATION_CLASS,
                streams,
                len(streams),
            ):
                raise _path_invalid("the activation temporary streams cannot be verified")
            offset = 0
            while True:
                if offset + _FileStreamInformation.StreamName.offset > len(streams):
                    raise _path_invalid("the activation temporary stream data is malformed")
                stream = _FileStreamInformation.from_buffer(streams, offset)
                name_length = int(stream.StreamNameLength)
                name_start = offset + _FileStreamInformation.StreamName.offset
                name_end = name_start + name_length
                if name_length % 2 or name_end > len(streams):
                    raise _path_invalid("the activation temporary stream name is malformed")
                try:
                    stream_name = ctypes.string_at(
                        ctypes.addressof(streams) + name_start,
                        name_length,
                    ).decode("utf-16-le")
                except UnicodeDecodeError as exc:
                    raise _path_invalid(
                        "the activation temporary stream name is malformed"
                    ) from exc
                if stream_name != "::$DATA":
                    raise _path_invalid("the activation temporary has an alternate data stream")
                next_offset = int(stream.NextEntryOffset)
                if next_offset == 0:
                    break
                if next_offset < _FileStreamInformation.StreamName.offset:
                    raise _path_invalid("the activation temporary stream chain is malformed")
                offset += next_offset
                if offset >= len(streams):
                    raise _path_invalid("the activation temporary stream chain escaped its buffer")

            os.lseek(descriptor, 0, os.SEEK_SET)
            content = bytearray()
            while len(content) <= len(expected):
                chunk = os.read(descriptor, len(expected) + 1 - len(content))
                if not chunk:
                    break
                content.extend(chunk)
            if bytes(content) != expected:
                raise _path_invalid("the activation temporary changed before activation")

        collision_error: int | None = None
        try:
            validate_retained_handle()
            if not kernel.FlushFileBuffers(raw_handle):
                raise _path_invalid("the activation temporary cannot be flushed before rename")
            rename, rename_size = _rename_information(final)
            if not kernel.SetFileInformationByHandle(
                raw_handle,
                _FILE_RENAME_INFORMATION_CLASS,
                rename,
                rename_size,
            ):
                error = ctypes.get_last_error()
                if error not in (_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS):
                    raise _path_invalid(
                        "the evidence artifact could not be activated without replacing"
                    ) from ctypes.WinError(error)
                validate_retained_handle()
                disposition = _FileDispositionInformation(DeleteFile=1)
                if not kernel.SetFileInformationByHandle(
                    raw_handle,
                    _FILE_DISPOSITION_INFORMATION_CLASS,
                    ctypes.byref(disposition),
                    ctypes.sizeof(disposition),
                ):
                    raise _path_invalid(
                        "the competing activation temporary could not be deleted"
                    )
                collision_error = error
            else:
                try:
                    validate_retained_handle()
                except EvidenceStorageError as exc:
                    disposition = _FileDispositionInformation(DeleteFile=1)
                    if not kernel.SetFileInformationByHandle(
                        raw_handle,
                        _FILE_DISPOSITION_INFORMATION_CLASS,
                        ctypes.byref(disposition),
                        ctypes.sizeof(disposition),
                    ):
                        raise _path_invalid(
                            "the invalid activated artifact could not be removed"
                        ) from exc
                    raise
                if not kernel.FlushFileBuffers(raw_handle):
                    raise _path_invalid("the activated evidence artifact could not be flushed")
        except OSError as exc:
            raise _path_invalid("the activation handle operation failed") from exc
        finally:
            try:
                retained.close()
            except OSError as exc:
                raise _path_invalid("the activation temporary handle could not be closed") from exc
        if collision_error is not None:
            raise EvidenceStorageError(
                WriterFault.OWNERSHIP_UNAVAILABLE,
                "another contender already activated this evidence artifact",
            ) from ctypes.WinError(collision_error)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            with suppress(OSError):
                _kernel32().CloseHandle(handle)


class WindowsStorageProbeV1:
    """The production probe; it reads Windows attributes and never mutates them."""

    def platform_is_supported(self) -> bool:
        return sys.platform == "win32"

    def volume_is_fixed_local(self, root: Path) -> bool:
        anchor = PureWindowsPath(str(root)).anchor
        if not anchor:
            return False
        return int(_kernel32().GetDriveTypeW(anchor)) == _DRIVE_FIXED

    def path_has_reparse_point(self, path: Path) -> bool:
        try:
            attributes = os.lstat(path).st_file_attributes
        except (FileNotFoundError, NotADirectoryError):
            # Nothing exists at this component, so nothing can redirect through it.
            return False
        except (OSError, ValueError, AttributeError):
            # Any other failure leaves redirection unproven; report it as unsafe.
            return True
        return bool(attributes & FILE_ATTRIBUTE_REPARSE_POINT)

    def path_has_alternate_data_streams(self, path: Path) -> bool:
        if not os.path.lexists(path):
            return False
        default = "::$INDEX_ALLOCATION" if os.path.isdir(path) else "::$DATA"
        kernel = _kernel32()
        data = _StreamData()
        ctypes.set_last_error(0)
        handle = kernel.FindFirstStreamW(str(path), 0, ctypes.byref(data), 0)
        if handle is None or handle in _INVALID_HANDLES:
            return ctypes.get_last_error() not in _NO_MORE_STREAM_ERRORS
        try:
            while True:
                if data.cStreamName != default:
                    return True
                ctypes.set_last_error(0)
                if not kernel.FindNextStreamW(handle, ctypes.byref(data)):
                    return ctypes.get_last_error() not in _NO_MORE_STREAM_ERRORS
        finally:
            kernel.FindClose(handle)

    def path_grants_only_owner(self, path: Path) -> bool:
        return _path_grants_only_owner(path)

    def allocated_bytes(self, path: Path) -> int:
        if not os.path.lexists(path):
            return 0
        high = ctypes.c_uint32(0)
        ctypes.set_last_error(0)
        low = int(_kernel32().GetCompressedFileSizeW(str(path), ctypes.byref(high)))
        if low == _INVALID_FILE_SIZE and ctypes.get_last_error() != 0:
            # The artifact exists but its allocation is unproven. Reporting 0
            # would under-count owned bytes and let the maintenance headroom
            # gate pass on a volume that cannot hold another maintenance
            # instant, so fail closed with the fault that gate exists to raise.
            raise EvidenceStorageError(
                WriterFault.QUOTA_UNAVAILABLE,
                "the evidence artifact allocation could not be measured",
            )
        return (int(high.value) << 32) | low

    def volume_free_bytes(self, root: Path) -> int:
        existing = root
        while not os.path.isdir(existing) and existing != existing.parent:
            existing = existing.parent
        free = ctypes.c_ulonglong(0)
        if not _kernel32().GetDiskFreeSpaceExW(str(existing), ctypes.byref(free), None, None):
            return 0
        return int(free.value)
