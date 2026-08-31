"""Task 4 contracts: the private V1 writer transport and its SQLite spool.

Every import is performed inside a test body so that a missing production
symbol fails the exact contract under test rather than collection.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import os
import queue
import re
import sqlite3
import struct
import sys
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, cast, get_type_hints
from uuid import UUID

import pytest

EXPECTED_MANIFEST_NAMES: dict[str, str] = {
    "root_marker": ".hermes-realtime-evidence-root-v1",
    "root_marker_init": ".hermes-realtime-evidence-root-v1.init",
    "sentinel": "capture-v1.owner",
    "sentinel_init": "capture-v1.owner.init",
    "database": "capture-v1.sqlite3",
    "database_journal": "capture-v1.sqlite3-journal",
    "database_wal": "capture-v1.sqlite3-wal",
    "database_shm": "capture-v1.sqlite3-shm",
    "database_vacuum": "capture-v1.sqlite3-vacuum",
    "database_tmp": "capture-v1.sqlite3-tmp",
}

EXPECTED_TRANSPORT_SIGNATURES: tuple[tuple[str, tuple[str, ...], str, str | None], ...] = (
    ("recover_existing", (), "RecoveryDisposition", None),
    ("create_epoch", ("command",), "StoreDisposition", "CreateEpochV1"),
    ("append_record", ("item",), "StoreDisposition", "QueuedEvidenceRecordV1"),
    ("append_binding_close", ("command",), "StoreDisposition", "BindingCloseV1"),
    ("rollover_session", ("command",), "StoreDisposition", "RolloverSessionV1"),
    ("expire_session", ("command",), "PurgeDisposition", "ExpireSessionV1"),
    ("commit_revoke_request", ("command",), "RevokeDisposition", "RevokeRequestV1"),
    ("finalize_revoke", ("command",), "RevokeDisposition", "RevokeFinalizeV1"),
    ("seal_epoch", ("command",), "StoreDisposition", "SealEpochV1"),
    ("run_maintenance", ("command",), "PurgeDisposition", "MaintenanceV1"),
    ("purge_full_store", ("command",), "PurgeDisposition", "FullPurgeV1"),
    ("drain_and_close", ("command",), "DrainDisposition", "DrainAndStopV1"),
    ("diagnostics", (), "EvidenceDiagnosticsV1", None),
)


def test_transport_protocol_declares_the_exact_v1_method_set() -> None:
    from hermes_realtime.evidence import transport

    protocol = transport.EvidenceWriterTransportV1
    declared = {
        name
        for name, value in vars(protocol).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    assert declared == {name for name, _, _, _ in EXPECTED_TRANSPORT_SIGNATURES}
    assert get_type_hints(protocol) == {"protocol_version": Literal[1]}


def test_transport_protocol_methods_have_exact_parameters_and_results() -> None:
    from hermes_realtime.evidence import models, transport

    protocol = transport.EvidenceWriterTransportV1
    for name, parameters, result, argument in EXPECTED_TRANSPORT_SIGNATURES:
        method = getattr(protocol, name)
        assert tuple(inspect.signature(method).parameters) == ("self", *parameters), name
        hints = get_type_hints(method)
        expected = {"return": getattr(models, result)}
        if argument is not None:
            expected[parameters[0]] = getattr(models, argument)
        assert hints == expected, name


def test_sqlite_spool_conforms_to_the_transport_protocol() -> None:
    from hermes_realtime.evidence import sqlite_spool, transport

    spool = sqlite_spool.SQLiteEvidenceSpool
    assert spool.protocol_version == 1
    for name, parameters, _result, argument in EXPECTED_TRANSPORT_SIGNATURES:
        concrete = getattr(spool, name)
        assert tuple(inspect.signature(concrete).parameters) == ("self", *parameters), name
        protocol_hints = get_type_hints(getattr(transport.EvidenceWriterTransportV1, name))
        assert get_type_hints(concrete) == protocol_hints, name
        assert argument is None or parameters == (parameters[0],)


def test_sqlite_writer_daemon_close_is_idempotently_successful_after_shutdown() -> None:
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceWriterDaemonV1

    closed: list[str] = []

    class Spool:
        owner_generation = 1

        def close_owner_marker(self) -> bool:
            return True

        def close(self) -> None:
            closed.append("spool")

    daemon = SQLiteEvidenceWriterDaemonV1(lambda: Spool())  # type: ignore[arg-type]

    assert daemon.close() is True
    assert daemon.close() is True
    assert closed == ["spool"]


def test_artifact_manifest_declares_exactly_the_ten_relative_names() -> None:

    from hermes_realtime.evidence import storage_security

    manifest = storage_security.MANIFEST_V1
    assert type(manifest) is storage_security.EvidenceArtifactManifestV1
    assert manifest.version == 1
    fields = {field.name: getattr(manifest, field.name) for field in dataclasses.fields(manifest)}
    assert fields == {"version": 1, **EXPECTED_MANIFEST_NAMES}
    assert manifest.all_names == tuple(EXPECTED_MANIFEST_NAMES.values())
    assert len(set(manifest.all_names)) == 10


def test_artifact_manifest_separates_deletable_retained_and_temp_names() -> None:

    from hermes_realtime.evidence import storage_security

    manifest = storage_security.MANIFEST_V1
    assert manifest.deletable_names == (
        "capture-v1.sqlite3",
        "capture-v1.sqlite3-journal",
        "capture-v1.sqlite3-wal",
        "capture-v1.sqlite3-shm",
        "capture-v1.sqlite3-vacuum",
        "capture-v1.sqlite3-tmp",
    )
    assert manifest.retained_names == (
        ".hermes-realtime-evidence-root-v1",
        "capture-v1.owner",
    )
    assert manifest.activation_temp_names == (
        ".hermes-realtime-evidence-root-v1.init",
        "capture-v1.owner.init",
    )
    assert set(manifest.deletable_names).isdisjoint(manifest.retained_names)
    assert set(manifest.deletable_names).isdisjoint(manifest.activation_temp_names)


def test_artifact_manifest_is_frozen_and_rejects_unlisted_children(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security

    manifest = storage_security.MANIFEST_V1
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.database = "other.sqlite3"  # type: ignore[misc]
    assert manifest.child(tmp_path, "capture-v1.sqlite3") == tmp_path / "capture-v1.sqlite3"
    for rejected in ("capture-v1.sqlite3-x", "CAPTURE-V1.SQLITE3", "..", "capture-v1.sqlite3 "):
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            manifest.child(tmp_path, rejected)
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


ROOT_ID = "3f2504e0-4f89-41d3-9a0c-0305e82c3301"
ROOT_MARKER_BYTES = (
    b'{"manifestVersion":1,"owner":"hermes-realtime-evidence",'
    b'"rootId":"3f2504e0-4f89-41d3-9a0c-0305e82c3301"}\n'
)


def test_root_marker_encodes_the_exact_canonical_record() -> None:

    from hermes_realtime.evidence import storage_security

    assert storage_security.ROOT_MARKER_OWNER == "hermes-realtime-evidence"
    assert storage_security.encode_root_marker(ROOT_ID) == ROOT_MARKER_BYTES
    assert storage_security.parse_root_marker(ROOT_MARKER_BYTES) == ROOT_ID


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(ROOT_MARKER_BYTES[:-1], id="missing_terminal_lf"),
        pytest.param(ROOT_MARKER_BYTES + b"\n", id="two_terminal_lf"),
        pytest.param(ROOT_MARKER_BYTES[:-1] + b"\r\n", id="crlf"),
        pytest.param(b"\xef\xbb\xbf" + ROOT_MARKER_BYTES, id="utf8_bom"),
        pytest.param(ROOT_MARKER_BYTES[:-1] + b"\x00\n", id="embedded_nul"),
        pytest.param(
            b'{\n  "manifestVersion": 1,\n  "owner": "hermes-realtime-evidence",\n'
            b'  "rootId": "3f2504e0-4f89-41d3-9a0c-0305e82c3301"\n}\n',
            id="pretty_printed",
        ),
        pytest.param(
            b'{"owner":"hermes-realtime-evidence","manifestVersion":1,'
            b'"rootId":"3f2504e0-4f89-41d3-9a0c-0305e82c3301"}\n',
            id="reordered_keys",
        ),
        pytest.param(
            ROOT_MARKER_BYTES.replace(b'"}\n', b'","extra":1}\n'),
            id="unknown_key",
        ),
        pytest.param(
            b'{"manifestVersion":1,"owner":"hermes-realtime-evidence"}\n',
            id="missing_root_id",
        ),
        pytest.param(
            ROOT_MARKER_BYTES.replace(b"hermes-realtime-evidence", b"hermes-realtime-profile"),
            id="wrong_owner",
        ),
        pytest.param(
            ROOT_MARKER_BYTES.replace(b'"manifestVersion":1', b'"manifestVersion":2'),
            id="wrong_manifest_version",
        ),
        pytest.param(
            ROOT_MARKER_BYTES.replace(b'"manifestVersion":1', b'"manifestVersion":true'),
            id="bool_manifest_version",
        ),
        pytest.param(
            ROOT_MARKER_BYTES.replace(b'"manifestVersion":1', b'"manifestVersion":"1"'),
            id="string_manifest_version",
        ),
        pytest.param(ROOT_MARKER_BYTES.replace(b"3f2504e0", b"3F2504E0"), id="uppercase_root_id"),
        pytest.param(ROOT_MARKER_BYTES.replace(b"41d3", b"11d3"), id="non_v4_root_id"),
        pytest.param(
            ROOT_MARKER_BYTES.replace(
                b'"3f2504e0-4f89-41d3-9a0c-0305e82c3301"',
                b'"3f2504e04f8941d39a0c0305e82c3301"',
            ),
            id="unhyphenated_root_id",
        ),
        pytest.param(
            b'{"manifestVersion":1,"manifestVersion":1,"owner":"hermes-realtime-evidence",'
            b'"rootId":"3f2504e0-4f89-41d3-9a0c-0305e82c3301"}\n',
            id="duplicate_key",
        ),
        pytest.param(b"", id="empty"),
        pytest.param(b"\n", id="lf_only"),
        pytest.param(b"\xff\xfe\n", id="not_utf8"),
    ],
)
def test_root_marker_parsing_is_fail_closed(raw: bytes) -> None:

    from hermes_realtime.evidence import storage_security

    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.parse_root_marker(raw)
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


def test_root_marker_encoding_rejects_a_non_canonical_root_id() -> None:

    from hermes_realtime.evidence import storage_security

    for rejected in ("3F2504E0-4F89-41D3-9A0C-0305E82C3301", "not-a-uuid", ROOT_ID.strip("1")):
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            storage_security.encode_root_marker(rejected)
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


STATE_ID = "6d2f8b71-9c0e-4a55-b3d1-71c58f0a2e94"
_ZERO_UUID = b"\x00" * 16


def raw_slot(
    generation: int,
    state_byte: int,
    uuid_bytes: bytes,
    *,
    reserved_a: bytes = b"\x00" * 7,
    reserved_b: bytes = b"\x00" * 184,
    digest: bytes | None = None,
) -> bytes:
    body = (
        struct.pack(">Q", generation)
        + bytes([state_byte])
        + reserved_a
        + uuid_bytes
        + reserved_b
    )
    assert len(body) == 216
    prefix = b"HRELOCK1" + struct.pack(">I", 1)
    return body + (hashlib.sha256(prefix + body).digest() if digest is None else digest)


def raw_image(
    slot_a: bytes,
    slot_b: bytes,
    *,
    magic: bytes = b"HRELOCK1",
    version: int = 1,
    reserved: bytes = b"\x00" * 4,
) -> bytes:
    return magic + struct.pack(">I", version) + reserved + slot_a + slot_b


CLEAR_SLOT_ZERO = raw_slot(0, 0, _ZERO_UUID)


def test_sentinel_constants_and_initial_image_are_the_exact_512_byte_format() -> None:

    from hermes_realtime.evidence import storage_security

    assert storage_security.SENTINEL_MAGIC == b"HRELOCK1"
    assert storage_security.SENTINEL_SIZE == 512
    assert storage_security.SENTINEL_SLOT_SIZE == 248
    assert storage_security.SENTINEL_VERSION == 1

    image = storage_security.initial_sentinel_image(STATE_ID)
    assert len(image) == 512
    assert image == raw_image(raw_slot(1, 3, UUID(STATE_ID).bytes), CLEAR_SLOT_ZERO)
    decoded = storage_security.decode_sentinel_image(image)
    assert decoded.active_index == 0
    assert decoded.active == storage_security.SentinelSlotV1(
        generation=1,
        state=storage_security.SentinelState.FIRST_CREATE_PENDING,
        state_generation_id=STATE_ID,
    )


def test_sentinel_state_byte_encoding_is_the_exact_closed_mapping() -> None:

    from hermes_realtime.evidence import storage_security

    expected = {
        storage_security.SentinelState.CLEAR: 0,
        storage_security.SentinelState.FULL_PURGE_PENDING: 1,
        storage_security.SentinelState.CLOCK_ROLLBACK_PURGE_PENDING: 2,
        storage_security.SentinelState.FIRST_CREATE_PENDING: 3,
    }
    assert set(expected) == set(storage_security.SentinelState)
    for state, byte in expected.items():
        identifier = None if state is storage_security.SentinelState.CLEAR else STATE_ID
        slot = storage_security.SentinelSlotV1(
            generation=7,
            state=state,
            state_generation_id=identifier,
        )
        encoded = storage_security.encode_sentinel_slot(slot)
        assert len(encoded) == 248
        assert encoded[8] == byte
        assert encoded == raw_slot(
            7, byte, _ZERO_UUID if identifier is None else UUID(STATE_ID).bytes
        )


def test_sentinel_selects_the_highest_valid_generation_and_ignores_a_corrupt_slot() -> None:

    from hermes_realtime.evidence import storage_security

    newer = raw_slot(9, 0, _ZERO_UUID)
    older = raw_slot(8, 2, UUID(STATE_ID).bytes)
    for image, index, generation in (
        (raw_image(newer, older), 0, 9),
        (raw_image(older, newer), 1, 9),
    ):
        decoded = storage_security.decode_sentinel_image(image)
        assert (decoded.active_index, decoded.active.generation) == (index, generation)

    corrupt = raw_slot(11, 0, _ZERO_UUID, digest=b"\x00" * 32)
    decoded = storage_security.decode_sentinel_image(raw_image(corrupt, older))
    assert decoded.active_index == 1
    assert decoded.slot_a is None
    assert decoded.active.generation == 8
    assert decoded.active.state is storage_security.SentinelState.CLOCK_ROLLBACK_PURGE_PENDING


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(raw_image(raw_slot(1, 0, _ZERO_UUID), CLEAR_SLOT_ZERO)[:-1], id="short"),
        pytest.param(
            raw_image(raw_slot(1, 0, _ZERO_UUID), CLEAR_SLOT_ZERO) + b"\x00", id="long"
        ),
        pytest.param(
            raw_image(raw_slot(1, 0, _ZERO_UUID), CLEAR_SLOT_ZERO, magic=b"HRELOCK2"),
            id="wrong_magic",
        ),
        pytest.param(
            raw_image(raw_slot(1, 0, _ZERO_UUID), CLEAR_SLOT_ZERO, version=2),
            id="wrong_version",
        ),
        pytest.param(
            raw_image(raw_slot(1, 0, _ZERO_UUID), CLEAR_SLOT_ZERO, reserved=b"\x00\x00\x00\x01"),
            id="reserved_header_not_zero",
        ),
        pytest.param(
            raw_image(
                raw_slot(1, 0, _ZERO_UUID, digest=b"\x11" * 32),
                raw_slot(2, 0, _ZERO_UUID, digest=b"\x22" * 32),
            ),
            id="both_slot_hashes_invalid",
        ),
        pytest.param(
            raw_image(
                raw_slot(1, 0, _ZERO_UUID, reserved_a=b"\x00" * 6 + b"\x01"),
                raw_slot(2, 0, _ZERO_UUID, reserved_a=b"\x01" + b"\x00" * 6),
            ),
            id="both_slot_reserved_a_not_zero",
        ),
        pytest.param(
            raw_image(
                raw_slot(1, 0, _ZERO_UUID, reserved_b=b"\x01" + b"\x00" * 183),
                raw_slot(2, 0, _ZERO_UUID, reserved_b=b"\x00" * 183 + b"\x01"),
            ),
            id="both_slot_reserved_b_not_zero",
        ),
        pytest.param(
            raw_image(raw_slot(1, 4, UUID(STATE_ID).bytes), raw_slot(2, 9, _ZERO_UUID)),
            id="unknown_state_byte",
        ),
        pytest.param(
            raw_image(
                raw_slot(1, 0, UUID(STATE_ID).bytes),
                raw_slot(2, 0, UUID(STATE_ID).bytes),
            ),
            id="clear_state_with_a_state_uuid",
        ),
        pytest.param(
            raw_image(raw_slot(1, 3, _ZERO_UUID), raw_slot(2, 1, _ZERO_UUID)),
            id="pending_state_without_a_state_uuid",
        ),
        pytest.param(
            raw_image(raw_slot(0, 3, UUID(STATE_ID).bytes), raw_slot(0, 1, UUID(STATE_ID).bytes)),
            id="generation_zero_is_not_clear",
        ),
        pytest.param(
            raw_image(raw_slot(5, 0, _ZERO_UUID), raw_slot(5, 0, _ZERO_UUID)),
            id="equal_nonzero_generations",
        ),
    ],
)
def test_sentinel_image_parsing_is_fail_closed(image: bytes) -> None:

    from hermes_realtime.evidence import storage_security

    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.decode_sentinel_image(image)
    assert caught.value.fault is storage_security.WriterFault.STORE_CORRUPT


def test_sentinel_update_writes_only_the_inactive_slot_and_advances_one_generation() -> None:

    from hermes_realtime.evidence import storage_security

    first = storage_security.initial_sentinel_image(STATE_ID)
    second = storage_security.next_sentinel_image(first, storage_security.SentinelState.CLEAR)
    assert second[16:264] == first[16:264]
    assert second[264:512] == raw_slot(2, 0, _ZERO_UUID)
    decoded = storage_security.decode_sentinel_image(second)
    assert (decoded.active_index, decoded.active.generation) == (1, 2)
    assert decoded.active.state is storage_security.SentinelState.CLEAR

    third = storage_security.next_sentinel_image(
        second,
        storage_security.SentinelState.FULL_PURGE_PENDING,
        state_generation_id=STATE_ID,
    )
    assert third[264:512] == second[264:512]
    assert third[16:264] == raw_slot(3, 1, UUID(STATE_ID).bytes)


def test_sentinel_update_is_fail_closed_on_bad_state_pairings_and_wraparound() -> None:

    from hermes_realtime.evidence import storage_security

    clear_image = storage_security.next_sentinel_image(
        storage_security.initial_sentinel_image(STATE_ID),
        storage_security.SentinelState.CLEAR,
    )
    with pytest.raises(storage_security.EvidenceStorageError):
        storage_security.next_sentinel_image(
            clear_image,
            storage_security.SentinelState.CLEAR,
            state_generation_id=STATE_ID,
        )
    with pytest.raises(storage_security.EvidenceStorageError):
        storage_security.next_sentinel_image(
            clear_image,
            storage_security.SentinelState.FULL_PURGE_PENDING,
        )

    saturated = raw_image(raw_slot(2**64 - 1, 0, _ZERO_UUID), CLEAR_SLOT_ZERO)
    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.next_sentinel_image(
            saturated,
            storage_security.SentinelState.FULL_PURGE_PENDING,
            state_generation_id=STATE_ID,
        )
    assert caught.value.fault is storage_security.WriterFault.STORE_CORRUPT


SUPPORTED_DATABASE = "C:\\ProgramData\\HermesRealtime\\evidence\\capture-v1.sqlite3"


class InjectedStorageProbe:
    """A seam standing in for Windows attributes physical tests cannot create."""

    def __init__(
        self,
        *,
        supported: bool = True,
        fixed: bool = True,
        reparse: frozenset[str] = frozenset(),
        streams: frozenset[str] = frozenset(),
        owner_only: frozenset[str] | None = None,
        allocations: dict[str, int] | None = None,
        free: int = 1 << 40,
    ) -> None:
        self.supported = supported
        self.fixed = fixed
        self.reparse = reparse
        self.streams = streams
        self.owner_only = owner_only
        self.allocations = {} if allocations is None else allocations
        self.free = free
        self.reparse_queries: list[str] = []

    def platform_is_supported(self) -> bool:
        return self.supported

    def volume_is_fixed_local(self, root: Path) -> bool:
        return self.fixed

    def path_has_reparse_point(self, path: Path) -> bool:
        self.reparse_queries.append(str(path))
        return str(path) in self.reparse

    def path_has_alternate_data_streams(self, path: Path) -> bool:
        return str(path) in self.streams

    def path_grants_only_owner(self, path: Path) -> bool:
        return self.owner_only is None or str(path) in self.owner_only

    def allocated_bytes(self, path: Path) -> int:
        return self.allocations.get(path.name, 0)

    def volume_free_bytes(self, root: Path) -> int:
        return self.free


class RecordingRootHandle:
    """A deterministic stand-in for the retained Windows parent authority."""

    instances: list[RecordingRootHandle] = []
    fail_fresh = False
    fail_revalidate = False

    def __init__(self, root: Path) -> None:
        from hermes_realtime.evidence import storage_security

        self.root = root
        self.closed = 0
        self.moves: list[tuple[Path, Path]] = []
        self._identity = storage_security.RootIdentityV1(
            volume_serial_number=1,
            file_index=2,
            attributes=storage_security.FILE_ATTRIBUTE_DIRECTORY,
            final_path=str(root),
        )
        RecordingRootHandle.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.fail_fresh = False
        cls.fail_revalidate = False

    @property
    def identity(self):  # type: ignore[no-untyped-def]
        return self._identity

    def revalidate(self):  # type: ignore[no-untyped-def]
        from hermes_realtime.evidence import storage_security

        if RecordingRootHandle.fail_revalidate:
            raise storage_security.EvidenceStorageError(
                storage_security.WriterFault.PATH_INVALID,
                "the retained evidence root authority no longer names the same object",
            )
        return self._identity

    def compare_fresh(self) -> None:
        from hermes_realtime.evidence import storage_security

        if RecordingRootHandle.fail_fresh:
            raise storage_security.EvidenceStorageError(
                storage_security.WriterFault.PATH_INVALID,
                "the configured evidence root no longer names the retained object",
            )

    def move_no_replace_write_through(self, source: Path, destination: Path) -> None:
        self.moves.append((source, destination))
        os.rename(source, destination)

    def activate_exact_temporary(self, temporary, final, retained, expected) -> None:  # type: ignore[no-untyped-def]
        from hermes_realtime.evidence import storage_security

        descriptor = retained.require_descriptor()
        os.lseek(descriptor, 0, os.SEEK_SET)
        if storage_security.identify_activation_temporary(descriptor) != retained.identity:
            raise storage_security.EvidenceStorageError(
                storage_security.WriterFault.PATH_INVALID,
                "the activation temporary changed identity before cleanup",
            )
        if os.read(descriptor, len(expected) + 1) != expected:
            raise storage_security.EvidenceStorageError(
                storage_security.WriterFault.PATH_INVALID,
                "the activation temporary changed before cleanup",
            )
        retained.close()
        os.rename(temporary, final)

    def close(self) -> None:
        self.closed += 1


def test_validated_root_accepts_an_exact_supported_windows_database_path() -> None:

    from hermes_realtime.evidence import storage_security

    probe = InjectedStorageProbe()
    validated = storage_security.validate_evidence_database_path(
        Path(SUPPORTED_DATABASE),
        probe=probe,
    )
    assert type(validated) is storage_security.ValidatedEvidenceRootV1
    assert validated.database == Path(SUPPORTED_DATABASE)
    assert validated.root == Path("C:\\ProgramData\\HermesRealtime\\evidence")
    assert validated.manifest is storage_security.MANIFEST_V1
    assert probe.reparse_queries == [
        "C:\\ProgramData\\HermesRealtime\\evidence",
        "C:\\ProgramData\\HermesRealtime",
        "C:\\ProgramData",
        "C:\\",
        # The final database leaf is probed last, after its whole ancestor chain.
        SUPPORTED_DATABASE,
    ]


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param("capture-v1.sqlite3", id="relative"),
        pytest.param("evidence\\capture-v1.sqlite3", id="relative_with_parent"),
        pytest.param("\\evidence\\capture-v1.sqlite3", id="driveless_rooted"),
        pytest.param("C:\\evidence\\other.sqlite3", id="wrong_basename"),
        pytest.param("C:\\evidence\\capture-v1.sqlite3.bak", id="suffixed_basename"),
        pytest.param("C:\\evidence\\Capture-V1.sqlite3", id="recased_basename"),
        pytest.param("C:\\evidence\\capture-v1.sqlite3-wal", id="artifact_basename"),
        pytest.param("C:\\evidence\\..\\evidence\\capture-v1.sqlite3", id="dotdot_component"),
        pytest.param("C:\\evi:l\\capture-v1.sqlite3", id="alternate_data_stream_syntax"),
        pytest.param("C:\\evidence\\capture-v1.sqlite3:evil", id="ads_on_database"),
        pytest.param("\\\\server\\share\\evidence\\capture-v1.sqlite3", id="unc"),
        pytest.param("\\\\.\\C:\\evidence\\capture-v1.sqlite3", id="device_namespace"),
        pytest.param("\\\\?\\C:\\evidence\\capture-v1.sqlite3", id="extended_namespace"),
        pytest.param("C:\\NUL\\capture-v1.sqlite3", id="reserved_nul"),
        pytest.param("C:\\evidence\\CON\\capture-v1.sqlite3", id="reserved_con"),
        pytest.param("C:\\evidence\\LPT1\\capture-v1.sqlite3", id="reserved_lpt1"),
        pytest.param("C:\\evidence \\capture-v1.sqlite3", id="trailing_space_component"),
        pytest.param("C:\\evidence.\\capture-v1.sqlite3", id="trailing_dot_component"),
        pytest.param("C:\\ev*\\capture-v1.sqlite3", id="wildcard_star"),
        pytest.param("C:\\ev?\\capture-v1.sqlite3", id="wildcard_question"),
        pytest.param("C:\\capture-v1.sqlite3", id="volume_root_parent"),
    ],
)
def test_validated_root_rejects_unsupported_path_shapes(candidate: str) -> None:

    from hermes_realtime.evidence import storage_security

    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(
            Path(candidate),
            probe=InjectedStorageProbe(),
        )
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


def test_validated_root_reports_unsupported_platform_before_any_path_work() -> None:

    from hermes_realtime.evidence import storage_security

    probe = InjectedStorageProbe(supported=False)
    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(Path("relative.sqlite3"), probe=probe)
    assert caught.value.fault is storage_security.WriterFault.UNSUPPORTED_PLATFORM
    assert probe.reparse_queries == []


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(InjectedStorageProbe(fixed=False), id="removable_or_network_volume"),
        pytest.param(
            InjectedStorageProbe(reparse=frozenset({"C:\\ProgramData\\HermesRealtime"})),
            id="reparse_point_on_an_ancestor",
        ),
        pytest.param(
            InjectedStorageProbe(
                reparse=frozenset({"C:\\ProgramData\\HermesRealtime\\evidence"})
            ),
            id="reparse_point_on_the_root",
        ),
        pytest.param(
            InjectedStorageProbe(
                streams=frozenset({"C:\\ProgramData\\HermesRealtime\\evidence"})
            ),
            id="alternate_data_stream_on_the_root",
        ),
        pytest.param(InjectedStorageProbe(owner_only=frozenset()), id="dacl_grants_others"),
    ],
)
def test_validated_root_rejects_hostile_physical_attributes(
    probe: InjectedStorageProbe,
) -> None:

    from hermes_realtime.evidence import storage_security

    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(Path(SUPPORTED_DATABASE), probe=probe)
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


THIS_CHECKOUT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "relative",
    [
        pytest.param("capture-v1.sqlite3", id="checkout_root"),
        pytest.param("evidence/capture-v1.sqlite3", id="checkout_child"),
        pytest.param("src/hermes_realtime/evidence/capture-v1.sqlite3", id="package_directory"),
        pytest.param(".hermes/plans/capture-v1.sqlite3", id="plans_directory"),
    ],
)
def test_repository_contained_roots_are_rejected(relative: str) -> None:

    from hermes_realtime.evidence import storage_security

    candidate = THIS_CHECKOUT.joinpath(*relative.split("/"))
    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(
            candidate,
            probe=InjectedStorageProbe(),
        )
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


def test_injected_repository_roots_reject_at_or_below_case_insensitively() -> None:

    from hermes_realtime.evidence import storage_security

    boundary = storage_security.FixedRepositoryBoundaryV1([Path("C:\\Work\\Repo")])
    for rejected in (
        "C:\\Work\\Repo\\capture-v1.sqlite3",
        "C:\\work\\repo\\sub\\capture-v1.sqlite3",
        "C:\\WORK\\REPO\\a\\b\\capture-v1.sqlite3",
    ):
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            storage_security.validate_evidence_database_path(
                Path(rejected),
                probe=InjectedStorageProbe(),
                boundary=boundary,
            )
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID

    # Containment is component-wise, never a raw string prefix.
    for accepted in (
        "C:\\Work\\RepoOther\\capture-v1.sqlite3",
        "C:\\Work\\Other\\capture-v1.sqlite3",
        "C:\\WorkRepo\\capture-v1.sqlite3",
    ):
        validated = storage_security.validate_evidence_database_path(
            Path(accepted),
            probe=InjectedStorageProbe(),
            boundary=boundary,
        )
        assert validated.database == Path(accepted)


def test_resolved_repository_containment_is_detected_and_fails_closed() -> None:

    from hermes_realtime.evidence import storage_security

    boundary = storage_security.FixedRepositoryBoundaryV1([Path("C:\\Work\\Repo")])
    clean = Path("C:\\Elsewhere\\evidence\\capture-v1.sqlite3")

    assert storage_security.validate_evidence_database_path(
        clean,
        probe=InjectedStorageProbe(),
        boundary=boundary,
        resolver=lambda path: path,
    ).root == Path("C:\\Elsewhere\\evidence")

    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(
            clean,
            probe=InjectedStorageProbe(),
            boundary=boundary,
            resolver=lambda path: Path("C:\\Work\\Repo\\linked"),
        )
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID

    def broken(path: Path) -> Path:
        raise OSError("cannot resolve")

    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(
            clean,
            probe=InjectedStorageProbe(),
            boundary=boundary,
            resolver=broken,
        )
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


def test_marker_repository_boundary_finds_checkouts_but_not_ordinary_roots(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security

    boundary = storage_security.MarkerRepositoryBoundaryV1()
    assert boundary.repository_root_for(THIS_CHECKOUT) == THIS_CHECKOUT
    assert boundary.repository_root_for(THIS_CHECKOUT / "src" / "hermes_realtime") == (
        THIS_CHECKOUT
    )
    assert boundary.repository_root_for(tmp_path) is None

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    assert boundary.repository_root_for(checkout / "deep" / "leaf") == checkout


def test_create_epoch_refuses_a_repository_contained_root_without_artifacts(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    root = checkout / "evidence"
    root.mkdir()

    owned = make_spool(checkout)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.PATH_INVALID
    finally:
        owned.close()

    assert list(root.iterdir()) == []


def test_ordinary_temporary_roots_remain_valid_under_the_default_boundary(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security

    candidate = tmp_path / "evidence" / "capture-v1.sqlite3"
    validated = storage_security.validate_evidence_database_path(
        candidate,
        probe=InjectedStorageProbe(),
    )
    assert validated.root == tmp_path / "evidence"


@pytest.mark.parametrize(
    "attribute",
    [
        pytest.param("reparse", id="database_leaf_is_a_reparse_point"),
        pytest.param("streams", id="database_leaf_carries_an_alternate_stream"),
        pytest.param("owner_only", id="database_leaf_grants_others"),
    ],
)
def test_final_database_leaf_physical_attributes_are_validated(attribute: str) -> None:

    from hermes_realtime.evidence import storage_security

    database = Path(SUPPORTED_DATABASE)
    root = database.parent
    if attribute == "owner_only":
        probe = InjectedStorageProbe(owner_only=frozenset({str(root)}))
    else:
        probe = InjectedStorageProbe(**{attribute: frozenset({str(database)})})

    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(database, probe=probe)
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


def test_final_database_leaf_is_probed_for_reparse_after_its_ancestors() -> None:

    from hermes_realtime.evidence import storage_security

    probe = InjectedStorageProbe()
    storage_security.validate_evidence_database_path(Path(SUPPORTED_DATABASE), probe=probe)
    assert probe.reparse_queries[-1] == SUPPORTED_DATABASE


@pytest.mark.skipif(sys.platform != "win32", reason="the real probe is Windows-only")
def test_real_symlinked_database_leaf_is_rejected_without_opening_it(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security

    elsewhere = tmp_path / "elsewhere"
    real = _seed_store(elsewhere)
    root = tmp_path / "evidence"
    root.mkdir()
    link = root / "capture-v1.sqlite3"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):  # pragma: no cover - needs Developer Mode
        pytest.skip("this account cannot create symbolic links")

    probe = storage_security.WindowsStorageProbeV1()
    assert probe.path_has_reparse_point(link) is True
    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(link, probe=probe)
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID

    before = real.read_bytes()
    owned = sqlite_spool.SQLiteEvidenceSpool(
        link,
        clock=StubClock(),
        uuid_factory=StubUuids(),
        probe=probe,
    )
    try:
        from hermes_realtime.evidence.models import StoreDisposition, WriterFault

        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.PATH_INVALID
    finally:
        owned.close()
    assert real.read_bytes() == before
    assert sorted(entry.name for entry in root.iterdir()) == ["capture-v1.sqlite3"]


def test_resolved_manifest_paths_are_exact_and_partitioned(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security

    resolved = storage_security.MANIFEST_V1.resolve(tmp_path)
    assert type(resolved) is storage_security.ResolvedEvidenceManifestV1
    assert resolved.version == 1
    assert {
        "root_marker": resolved.root_marker,
        "root_marker_init": resolved.root_marker_init,
        "sentinel": resolved.sentinel,
        "sentinel_init": resolved.sentinel_init,
        "database": resolved.database,
        "database_journal": resolved.database_journal,
        "database_wal": resolved.database_wal,
        "database_shm": resolved.database_shm,
        "database_vacuum": resolved.database_vacuum,
        "database_tmp": resolved.database_tmp,
    } == {name: tmp_path / value for name, value in EXPECTED_MANIFEST_NAMES.items()}
    assert resolved.deletable == tuple(
        tmp_path / name for name in storage_security.MANIFEST_V1.deletable_names
    )
    assert resolved.retained == (
        tmp_path / ".hermes-realtime-evidence-root-v1",
        tmp_path / "capture-v1.owner",
    )
    assert resolved.activation_temps == (
        tmp_path / ".hermes-realtime-evidence-root-v1.init",
        tmp_path / "capture-v1.owner.init",
    )
    assert len(set(resolved.all_paths)) == 10


def test_root_handle_flags_exclude_delete_sharing_and_follow_no_reparse() -> None:

    from hermes_realtime.evidence import storage_security

    assert storage_security.FILE_SHARE_DELETE == 0x00000004
    assert storage_security.ROOT_HANDLE_SHARE_MODE & storage_security.FILE_SHARE_DELETE == 0
    assert storage_security.ROOT_HANDLE_SHARE_MODE == (
        storage_security.FILE_SHARE_READ | storage_security.FILE_SHARE_WRITE
    )
    assert storage_security.ROOT_HANDLE_FLAGS & storage_security.FILE_FLAG_BACKUP_SEMANTICS
    assert storage_security.ROOT_HANDLE_FLAGS & storage_security.FILE_FLAG_OPEN_REPARSE_POINT


def test_activation_move_flags_are_write_through_without_replace() -> None:

    from hermes_realtime.evidence import storage_security

    assert storage_security.MOVEFILE_WRITE_THROUGH == 0x00000008
    assert storage_security.MOVEFILE_REPLACE_EXISTING == 0x00000001
    assert storage_security.ACTIVATION_MOVE_FLAGS & storage_security.MOVEFILE_WRITE_THROUGH
    assert storage_security.ACTIVATION_MOVE_FLAGS & storage_security.MOVEFILE_REPLACE_EXISTING == 0


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
@pytest.mark.parametrize(
    ("native_error", "expected_fault"),
    [
        pytest.param(80, "ownership_unavailable", id="file_exists"),
        pytest.param(183, "ownership_unavailable", id="already_exists"),
        pytest.param(5, "path_invalid", id="access_denied"),
    ],
)
def test_activation_handle_classifies_only_native_name_collisions_as_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_error: int,
    expected_fault: str,
) -> None:
    import ctypes

    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    final = root / "capture-v1.owner"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    kernel32 = storage_security._kernel32()

    class FailingRenameKernel32:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(kernel32, name)

        def SetFileInformationByHandle(
            self,
            handle: int,
            information_class: int,
            information: object,
            information_size: int,
        ) -> int:
            if information_class == 3:
                ctypes.set_last_error(native_error)
                return 0
            return kernel32.SetFileInformationByHandle(
                handle,
                information_class,
                information,
                information_size,
            )

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        monkeypatch.setattr(storage_security, "_kernel32", lambda: FailingRenameKernel32())
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            authority.activate_exact_temporary(temporary, final, retained, expected)
        assert caught.value.fault.value == expected_fault
        assert not final.exists()
        assert temporary.exists() is (expected_fault == "path_invalid")
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_real_root_authority_blocks_replacement_while_retained(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        identity = authority.identity
        assert type(identity) is storage_security.RootIdentityV1
        assert identity.volume_serial_number > 0
        assert identity.file_index > 0
        assert identity.attributes & storage_security.FILE_ATTRIBUTE_DIRECTORY
        assert identity.attributes & storage_security.FILE_ATTRIBUTE_REPARSE_POINT == 0
        assert identity.final_path.casefold().endswith("evidence")

        # No FILE_SHARE_DELETE means the root cannot be renamed or removed away.
        with pytest.raises(OSError):
            os.rename(root, tmp_path / "swapped")
        with pytest.raises(OSError):
            root.rmdir()

        assert authority.revalidate() == identity
        assert authority.compare_fresh() is None
    finally:
        authority.close()
    os.rename(root, tmp_path / "swapped")
    assert (tmp_path / "swapped").is_dir()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_real_handle_activation_is_no_replace_and_disposes_exact_loser(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    final = root / "capture-v1.owner"
    expected = b"new"
    retained = sqlite_spool._write_new_file(temporary, expected)
    final.write_bytes(b"original")

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            authority.activate_exact_temporary(temporary, final, retained, expected)
        assert caught.value.fault is storage_security.WriterFault.OWNERSHIP_UNAVAILABLE
        assert final.read_bytes() == b"original"
        assert not temporary.exists()

        final.unlink()
        retained = sqlite_spool._write_new_file(temporary, expected)
        assert authority.activate_exact_temporary(temporary, final, retained, expected) is None
        assert final.read_bytes() == expected
        assert not temporary.exists()
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_activation_creation_handle_blocks_a_pre_activation_writer(tmp_path: Path) -> None:
    import ctypes

    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    final = root / "capture-v1.owner"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    kernel32 = storage_security._kernel32()
    ctypes.set_last_error(0)
    attacker = kernel32.CreateFileW(
        str(temporary),
        storage_security.GENERIC_WRITE | storage_security._DELETE_ACCESS,
        (
            storage_security.FILE_SHARE_READ
            | storage_security.FILE_SHARE_WRITE
            | storage_security.FILE_SHARE_DELETE
        ),
        None,
        storage_security.OPEN_EXISTING,
        storage_security.FILE_ATTRIBUTE_NORMAL,
        None,
    )
    assert attacker in storage_security._INVALID_HANDLES
    assert ctypes.get_last_error() == 32

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        authority.activate_exact_temporary(temporary, final, retained, expected)
        assert not temporary.exists()
        assert final.read_bytes() == expected
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_activation_creation_handle_blocks_substitution_before_activation(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    displaced = root / "displaced"
    final = root / "capture-v1.owner"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    with pytest.raises(PermissionError) as caught:
        temporary.rename(displaced)
    assert caught.value.winerror == 32

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        authority.activate_exact_temporary(temporary, final, retained, expected)
        assert not temporary.exists()
        assert not displaced.exists()
        assert final.read_bytes() == expected
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_activation_cleanup_refuses_a_temporary_with_an_added_hard_link(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    extra_link = root / "extra-link"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    os.link(temporary, extra_link)

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            authority.activate_exact_temporary(
                temporary,
                root / "capture-v1.owner",
                retained,
                expected,
            )
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID
        assert temporary.read_bytes() == expected
        assert extra_link.read_bytes() == expected
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_activation_holds_the_verified_leaf_through_successful_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    final = root / "capture-v1.owner"
    replacement = root / "replacement"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    replacement.write_bytes(b"third-party")
    kernel32 = storage_security._kernel32()
    replacement_errors: list[int | None] = []

    class ReplacementAttemptKernel32:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(kernel32, name)

        def SetFileInformationByHandle(
            self,
            handle: int,
            information_class: int,
            information: object,
            information_size: int,
        ) -> int:
            try:
                os.replace(replacement, temporary)
            except OSError as exc:
                replacement_errors.append(exc.winerror)
            return kernel32.SetFileInformationByHandle(
                handle,
                information_class,
                information,
                information_size,
            )

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        monkeypatch.setattr(
            storage_security,
            "_kernel32",
            lambda: ReplacementAttemptKernel32(),
        )
        authority.activate_exact_temporary(temporary, final, retained, expected)
        assert replacement_errors == [5]
        assert not temporary.exists()
        assert final.read_bytes() == expected
        assert replacement.read_bytes() == b"third-party"
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_activation_creation_handle_blocks_alternate_data_stream_injection(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    final = root / "capture-v1.owner"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    with pytest.raises(PermissionError):
        Path(f"{temporary}:third-party").write_bytes(b"alternate")

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        authority.activate_exact_temporary(temporary, final, retained, expected)
        assert not temporary.exists()
        assert final.read_bytes() == expected
        assert not Path(f"{final}:third-party").exists()
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_activation_rejects_a_hard_link_injected_after_successful_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    final = root / "capture-v1.owner"
    extra_link = root / "third-party-link"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    kernel32 = storage_security._kernel32()

    class InjectingKernel32:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(kernel32, name)

        def SetFileInformationByHandle(
            self,
            handle: int,
            information_class: int,
            information: object,
            information_size: int,
        ) -> int:
            result = kernel32.SetFileInformationByHandle(
                handle,
                information_class,
                information,
                information_size,
            )
            error = ctypes.get_last_error()
            if information_class == 3 and result:
                os.link(final, extra_link)
            ctypes.set_last_error(error)
            return result

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        monkeypatch.setattr(storage_security, "_kernel32", lambda: InjectingKernel32())
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            authority.activate_exact_temporary(temporary, final, retained, expected)
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID
        assert not temporary.exists()
        assert not final.exists()
        assert extra_link.read_bytes() == expected
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_activation_rejects_attributes_changed_after_successful_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    final = root / "capture-v1.owner"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    kernel32 = storage_security._kernel32()
    set_attributes = ctypes.WinDLL("kernel32", use_last_error=True).SetFileAttributesW
    set_attributes.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
    set_attributes.restype = ctypes.c_int

    class InjectingKernel32:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(kernel32, name)

        def SetFileInformationByHandle(
            self,
            handle: int,
            information_class: int,
            information: object,
            information_size: int,
        ) -> int:
            result = kernel32.SetFileInformationByHandle(
                handle,
                information_class,
                information,
                information_size,
            )
            error = ctypes.get_last_error()
            if information_class == 3 and result:
                assert set_attributes(str(final), retained.attributes | 0x2)
            ctypes.set_last_error(error)
            return result

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        monkeypatch.setattr(storage_security, "_kernel32", lambda: InjectingKernel32())
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            authority.activate_exact_temporary(temporary, final, retained, expected)
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID
        assert not temporary.exists()
        assert not final.exists()
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_activation_collision_preserves_a_hard_link_injected_before_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    from hermes_realtime.evidence import sqlite_spool, storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    temporary = root / "capture-v1.owner.init"
    final = root / "capture-v1.owner"
    extra_link = root / "third-party-link"
    expected = b"owned-temporary"
    retained = sqlite_spool._write_new_file(temporary, expected)
    final.write_bytes(b"winner")
    kernel32 = storage_security._kernel32()

    class InjectingKernel32:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(kernel32, name)

        def SetFileInformationByHandle(
            self,
            handle: int,
            information_class: int,
            information: object,
            information_size: int,
        ) -> int:
            result = kernel32.SetFileInformationByHandle(
                handle,
                information_class,
                information,
                information_size,
            )
            error = ctypes.get_last_error()
            if information_class == 3 and not result:
                os.link(temporary, extra_link)
            ctypes.set_last_error(error)
            return result

    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        monkeypatch.setattr(storage_security, "_kernel32", lambda: InjectingKernel32())
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            authority.activate_exact_temporary(temporary, final, retained, expected)
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID
        assert final.read_bytes() == b"winner"
        assert temporary.read_bytes() == expected
        assert extra_link.read_bytes() == expected
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_real_two_contender_activation_forces_native_collision_and_cleans_loser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    from hermes_realtime.evidence import sqlite_spool, storage_security
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    root = tmp_path / "evidence"
    root.mkdir()
    (root / storage_security.MANIFEST_V1.root_marker).write_bytes(
        storage_security.encode_root_marker(INSTALL_ID)
    )
    original_write = sqlite_spool._write_new_file
    kernel32 = storage_security._kernel32()
    native_move_errors: list[int] = []
    both_absent = threading.Barrier(2)
    contender = threading.local()

    class RecordingMoveKernel32:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(kernel32, name)

        def SetFileInformationByHandle(
            self,
            handle: int,
            information_class: int,
            information: object,
            information_size: int,
        ) -> int:
            result = kernel32.SetFileInformationByHandle(
                handle,
                information_class,
                information,
                information_size,
            )
            error = ctypes.get_last_error()
            if information_class == 3 and not result:
                native_move_errors.append(error)
            ctypes.set_last_error(error)
            return result

    def synchronized_write(path: Path, data: bytes):  # type: ignore[no-untyped-def]
        if path.name == storage_security.MANIFEST_V1.sentinel_init:
            both_absent.wait(timeout=10.0)
            if contender.identifier == 1:
                final = path.with_name(storage_security.MANIFEST_V1.sentinel)
                for _ in range(10_000):
                    if final.is_file():
                        break
                    threading.Event().wait(0.001)
                else:  # pragma: no cover - bounded fail-closed guard
                    raise RuntimeError("the winning activation did not reach its final name")
        return original_write(path, data)

    monkeypatch.setattr(sqlite_spool, "_write_new_file", synchronized_write)
    monkeypatch.setattr(storage_security, "_kernel32", lambda: RecordingMoveKernel32())
    results: dict[int, tuple[StoreDisposition, WriterFault | None]] = {}
    failures: list[BaseException] = []

    def run(identifier: int) -> None:
        contender.identifier = identifier
        owned = make_spool(tmp_path)
        try:
            disposition = owned.create_epoch(make_create_epoch())
            results[identifier] = (disposition, owned.diagnostics().sticky_fault)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            owned.close()

    threads = [threading.Thread(target=run, args=(identifier,)) for identifier in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30.0)
    assert not any(thread.is_alive() for thread in threads)
    assert failures == []
    assert len(native_move_errors) == 1
    assert native_move_errors[0] in (80, 183)
    assert sorted(results.values(), key=lambda item: item[0].value) == [
        (StoreDisposition.COMMITTED, None),
        (StoreDisposition.FAULTED, WriterFault.OWNERSHIP_UNAVAILABLE),
    ]
    assert not (root / storage_security.MANIFEST_V1.sentinel_init).exists()
    assert {path.name for path in root.iterdir()} == {
        storage_security.MANIFEST_V1.root_marker,
        storage_security.MANIFEST_V1.sentinel,
        storage_security.MANIFEST_V1.database,
    }


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_real_root_authority_close_is_idempotent(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    authority.close()
    authority.close()
    with pytest.raises(storage_security.EvidenceStorageError):
        authority.revalidate()
    os.rename(root, tmp_path / "moved")


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_real_root_authority_rechecks_the_dacl_on_every_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from hermes_realtime.evidence import storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    authority = storage_security.WindowsEvidenceRootHandleV1(root)
    try:
        assert authority.revalidate() == authority.identity
        assert authority.compare_fresh() is None

        # Broaden the DACL only after the handle was opened and validated.
        monkeypatch.setattr(storage_security, "_path_grants_only_owner", lambda path: False)
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            authority.revalidate()
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID
        with pytest.raises(storage_security.EvidenceStorageError) as caught:
            authority.compare_fresh()
        assert caught.value.fault is storage_security.WriterFault.PATH_INVALID
    finally:
        authority.close()


@pytest.mark.skipif(sys.platform != "win32", reason="the real authority is Windows-only")
def test_dacl_broadened_before_activation_refuses_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    root = tmp_path / "evidence"
    root.mkdir()
    remaining = [1]

    def broadening(path: Path) -> bool:
        # The opening constructor check succeeds; every later re-check fails.
        if remaining[0] > 0:
            remaining[0] -= 1
            return True
        return False

    monkeypatch.setattr(storage_security, "_path_grants_only_owner", broadening)
    owned = sqlite_spool.SQLiteEvidenceSpool(
        root / "capture-v1.sqlite3",
        clock=StubClock(),
        uuid_factory=StubUuids(),
        probe=InjectedStorageProbe(),
    )
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.PATH_INVALID
    finally:
        owned.close()

    # The marker temporary exists but was never activated, and nothing outside the
    # original root was touched.
    assert sorted(entry.name for entry in root.iterdir()) == [
        ".hermes-realtime-evidence-root-v1.init"
    ]
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["evidence"]


def test_validated_root_retains_the_parent_authority_and_resolved_manifest(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security

    root = tmp_path / "evidence"
    root.mkdir()
    opened: list[Path] = []

    def opener(target: Path) -> storage_security.EvidenceRootHandleV1:
        opened.append(target)
        return RecordingRootHandle(target)

    validated = storage_security.validate_evidence_database_path(
        root / "capture-v1.sqlite3",
        probe=InjectedStorageProbe(),
        root_handle_opener=opener,
    )
    assert opened == [root]
    authority, resolved = validated.require_authority()
    assert type(resolved) is storage_security.ResolvedEvidenceManifestV1
    assert resolved.database == root / "capture-v1.sqlite3"
    assert authority.identity.final_path == str(root)

    without = storage_security.validate_evidence_database_path(
        root / "capture-v1.sqlite3",
        probe=InjectedStorageProbe(),
    )
    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        without.require_authority()
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


def test_physical_maintenance_ceiling_uses_the_exact_free_space_formula() -> None:

    from hermes_realtime.evidence import storage_security

    ceiling = storage_security.EVIDENCE_PHYSICAL_MAINTENANCE_CEILING_BYTES
    assert ceiling == 320 * 1024 * 1024
    assert storage_security.required_free_bytes(0) == ceiling
    assert storage_security.required_free_bytes(96 * 1024 * 1024) == 224 * 1024 * 1024
    assert storage_security.required_free_bytes(ceiling) == 0
    assert storage_security.required_free_bytes(ceiling + 1) == 0


def test_owned_allocation_counts_only_the_six_deletable_names_once() -> None:

    from hermes_realtime.evidence import storage_security

    probe = InjectedStorageProbe(
        allocations={
            "capture-v1.sqlite3": 4096,
            "capture-v1.sqlite3-journal": 8192,
            "capture-v1.sqlite3-wal": 16384,
            "capture-v1.owner": 512,
            ".hermes-realtime-evidence-root-v1": 128,
            "decoy.sqlite3": 1 << 30,
        }
    )
    root = Path("C:\\ProgramData\\HermesRealtime\\evidence")
    owned = storage_security.owned_allocated_bytes(root, probe=probe)
    assert owned == 4096 + 8192 + 16384
    assert storage_security.required_free_bytes(owned) == (
        storage_security.EVIDENCE_PHYSICAL_MAINTENANCE_CEILING_BYTES - owned
    )


def test_maintenance_headroom_preflight_compares_volume_free_bytes() -> None:

    from hermes_realtime.evidence import storage_security

    root = Path("C:\\ProgramData\\HermesRealtime\\evidence")
    allocations = {"capture-v1.sqlite3": 96 * 1024 * 1024}
    exact = InjectedStorageProbe(allocations=allocations, free=224 * 1024 * 1024)
    assert storage_security.check_maintenance_headroom(root, probe=exact) is None
    short = InjectedStorageProbe(allocations=allocations, free=224 * 1024 * 1024 - 1)
    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.check_maintenance_headroom(root, probe=short)
    assert caught.value.fault is storage_security.WriterFault.QUOTA_UNAVAILABLE


@pytest.mark.skipif(sys.platform != "win32", reason="the real probe is Windows-only")
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(PermissionError("denied"), id="permission_error"),
        pytest.param(OSError("unexpected"), id="generic_os_error"),
        pytest.param(ValueError("no attributes"), id="value_error"),
        pytest.param(AttributeError("no st_file_attributes"), id="attribute_error"),
    ],
)
def test_reparse_probe_fails_closed_on_an_unexpected_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:

    from hermes_realtime.evidence import storage_security

    probe = storage_security.WindowsStorageProbeV1()
    assert probe.path_has_reparse_point(tmp_path) is False
    assert probe.path_has_reparse_point(tmp_path / "absent") is False

    def refuse(target: object, **kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(storage_security.os, "lstat", refuse)
    assert probe.path_has_reparse_point(tmp_path) is True

    database = tmp_path / "evidence" / "capture-v1.sqlite3"
    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.validate_evidence_database_path(database, probe=probe)
    assert caught.value.fault is storage_security.WriterFault.PATH_INVALID


def test_storage_fault_messages_are_static_and_content_free() -> None:
    package = Path(__file__).resolve().parents[2] / "src" / "hermes_realtime" / "evidence"
    raisers = {"EvidenceStorageError", "EvidenceFrameError", "_store_corrupt", "_path_invalid"}
    # These two only relay a caller-supplied literal; every caller is checked below.
    forwarders = {"_store_corrupt", "_path_invalid"}
    offenders: list[str] = []
    for name in ("storage_security.py", "sqlite_spool.py"):
        module = package / name
        tree = ast.parse(module.read_text(encoding="utf-8"))
        relayed = {
            node
            for definition in ast.walk(tree)
            if isinstance(definition, ast.FunctionDef) and definition.name in forwarders
            for node in ast.walk(definition)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or node in relayed:
                continue
            called = node.func
            label = getattr(called, "id", None) or getattr(called, "attr", None)
            if label not in raisers or not node.args:
                continue
            message = node.args[-1]
            if not isinstance(message, ast.Constant) or type(message.value) is not str:
                offenders.append(f"{name}:{node.lineno}")
    assert offenders == []


def test_raised_fault_messages_leak_no_path_or_identifier(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security

    messages: list[str] = []

    def capture(action: object) -> None:
        assert callable(action)
        try:
            action()
        except storage_security.EvidenceStorageError as exc:
            messages.append(str(exc))
        except ValueError as exc:
            messages.append(str(exc))
        else:  # pragma: no cover - every action below must raise
            pytest.fail("the action did not fail closed")

    secret_root = tmp_path / "SecretFolder"
    secret_root.mkdir()
    (secret_root / "unowned.txt").write_bytes(b"x")
    capture(lambda: storage_security.audit_root_occupancy(secret_root))
    capture(lambda: storage_security.parse_root_marker(b"{}\n"))
    capture(lambda: storage_security.decode_sentinel_image(b"\x00" * 512))
    capture(
        lambda: storage_security.validate_evidence_database_path(
            Path("C:\\SecretFolder\\other.sqlite3"),
            probe=InjectedStorageProbe(),
        )
    )
    capture(lambda: sqlite_spool.validate_database_header(secret_root / "capture-v1.sqlite3"))
    capture(lambda: sqlite_spool.preflight_store_database(secret_root / "capture-v1.sqlite3"))
    capture(lambda: sqlite_spool.check_canonical_payload_size(b"x" * 32769))

    assert len(messages) == 7
    for message in messages:
        assert "\\" not in message
        assert "/" not in message
        assert "SecretFolder" not in message
        assert not re.search(r"[A-Za-z]:", message)
        assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}", message)


@pytest.mark.skipif(sys.platform != "win32", reason="the real probe is Windows-only")
def test_windows_storage_probe_reports_real_local_attributes(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security

    probe = storage_security.WindowsStorageProbeV1()
    assert probe.platform_is_supported() is True
    assert probe.volume_is_fixed_local(tmp_path) is True
    assert probe.path_has_reparse_point(tmp_path) is False
    assert probe.path_has_alternate_data_streams(tmp_path) is False
    assert type(probe.path_grants_only_owner(tmp_path)) is bool
    sample = tmp_path / "capture-v1.sqlite3"
    sample.write_bytes(b"x" * 8192)
    assert probe.path_has_alternate_data_streams(sample) is False
    assert probe.allocated_bytes(sample) >= 8192
    assert probe.allocated_bytes(tmp_path / "capture-v1.sqlite3-wal") == 0
    assert probe.volume_free_bytes(tmp_path) > 0

    Path(f"{sample}:smuggled").write_bytes(b"secret")
    assert probe.path_has_alternate_data_streams(sample) is True


@pytest.mark.skipif(sys.platform != "win32", reason="the real probe is Windows-only")
def test_windows_storage_probe_fails_closed_on_stream_enumeration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    from hermes_realtime.evidence import storage_security

    sample = tmp_path / "capture-v1.sqlite3"
    sample.write_bytes(b"canonical")

    class FailingStreamEnumerationKernel32:
        @staticmethod
        def FindFirstStreamW(path, level, data, flags):  # type: ignore[no-untyped-def]
            del path, level, flags
            stream = ctypes.cast(
                data,
                ctypes.POINTER(storage_security._StreamData),
            ).contents
            stream.cStreamName = "::$DATA"
            return 1

        @staticmethod
        def FindNextStreamW(handle, data):  # type: ignore[no-untyped-def]
            del handle, data
            ctypes.set_last_error(5)
            return 0

        @staticmethod
        def FindClose(handle):  # type: ignore[no-untyped-def]
            del handle
            return 1

    monkeypatch.setattr(
        storage_security,
        "_kernel32",
        lambda: FailingStreamEnumerationKernel32(),
    )
    assert storage_security.WindowsStorageProbeV1().path_has_alternate_data_streams(sample)


NULL_TAG_VECTOR: dict[str, object] = {
    "installation_id": "00000000-0000-4000-8000-000000000001",
    "producer_instance_id": "00000000-0000-4000-8000-000000000002",
    "event_id": "00000000-0000-4000-8000-000000000003",
    "logical_session_id": "00000000-0000-4000-8000-000000000004",
    "event_sequence": 1,
    "event_kind": "session_opened",
    "recorded_at_utc": "2026-08-08T00:00:00.000000Z",
    "payload_hash": "00" * 32,
    "previous_hash": None,
}
PREVIOUS_HASH_VECTOR: dict[str, object] = {
    **NULL_TAG_VECTOR,
    "event_sequence": 3,
    "event_kind": "user_final_accepted",
    "previous_hash": "11" * 32,
}
NULL_TAG_DIGEST = "5fd1e46b71e2fd3bbd673b01da1bc32b1293be6870d364c1ac0ffa2e2d5a6172"
PREVIOUS_HASH_DIGEST = "af58a6ea26369267612d6185f91cd2aab884cfdd01d49cfbbf0c595a27784c71"


def reference_frame(**vector: object) -> bytes:
    """Rebuild the §5.3 frame independently of production code."""

    kind = str(vector["event_kind"]).encode("utf-8")
    moment = str(vector["recorded_at_utc"]).encode("utf-8")
    previous = vector["previous_hash"]
    raw = b"HRE1" + struct.pack(">I", 1)
    for key in ("installation_id", "producer_instance_id", "event_id", "logical_session_id"):
        raw += UUID(str(vector[key])).bytes
    raw += struct.pack(">Q", int(str(vector["event_sequence"])))
    raw += struct.pack(">I", len(kind)) + kind
    raw += struct.pack(">I", len(moment)) + moment
    raw += bytes.fromhex(str(vector["payload_hash"]))
    if previous is None:
        return raw + b"\x00"
    return raw + b"\x01" + bytes.fromhex(str(previous))


def test_canonical_json_is_sorted_compact_utf8_and_finite() -> None:
    from hermes_realtime.evidence import sqlite_spool

    encoded = sqlite_spool.canonical_json_bytes(
        {"text": "café — naïve", "source": "typed", "count": 2}
    )
    assert encoded == b'{"count":2,"source":"typed","text":"caf\xc3\xa9 \xe2\x80\x94 na\xc3\xafve"}'
    assert hashlib.sha256(encoded).hexdigest() == (
        "836c6a13d089dd654011fc1693eda8a286209752aad5249e04deb2ef9af9da8e"
    )
    for rejected in ({"value": float("nan")}, {"value": float("inf")}, {"value": 1.5}):
        with pytest.raises(ValueError):
            sqlite_spool.canonical_json_bytes(rejected)


def test_canonical_utc_formats_and_validates_the_exact_spelling() -> None:
    from hermes_realtime.evidence import sqlite_spool

    moment = datetime(2026, 8, 8, 1, 2, 3, 456789, tzinfo=UTC)
    assert sqlite_spool.format_canonical_utc(moment) == "2026-08-08T01:02:03.456789Z"
    assert (
        sqlite_spool.format_canonical_utc(datetime(2026, 1, 1, tzinfo=UTC))
        == "2026-01-01T00:00:00.000000Z"
    )
    for rejected in (
        datetime(2026, 8, 8, 1, 2, 3, 456789),
        datetime(2026, 8, 8, tzinfo=timezone(timedelta(hours=2))),
    ):
        with pytest.raises(ValueError):
            sqlite_spool.format_canonical_utc(rejected)


def test_hre1_frame_matches_the_fixed_null_tag_and_previous_hash_vectors() -> None:
    from hermes_realtime.evidence import sqlite_spool

    assert sqlite_spool.HRE1_MAGIC == b"HRE1"
    assert sqlite_spool.HRE1_SCHEMA_VERSION == 1

    null_tag = sqlite_spool.hre1_record_frame(**NULL_TAG_VECTOR)  # type: ignore[arg-type]
    assert null_tag == reference_frame(**NULL_TAG_VECTOR)
    assert len(null_tag) == 162
    assert null_tag[-1:] == b"\x00"
    assert sqlite_spool.hre1_record_hash(**NULL_TAG_VECTOR) == NULL_TAG_DIGEST  # type: ignore[arg-type]

    chained = sqlite_spool.hre1_record_frame(**PREVIOUS_HASH_VECTOR)  # type: ignore[arg-type]
    assert chained == reference_frame(**PREVIOUS_HASH_VECTOR)
    assert len(chained) == 199
    assert chained[-33:] == b"\x01" + bytes.fromhex("11" * 32)
    assert (
        sqlite_spool.hre1_record_hash(**PREVIOUS_HASH_VECTOR)  # type: ignore[arg-type]
        == PREVIOUS_HASH_DIGEST
    )


def test_hre1_length_prefixes_defeat_concatenation_ambiguity() -> None:
    from hermes_realtime.evidence import sqlite_spool

    # The format itself must be unambiguous: two splits of the same concatenated
    # bytes across the kind/time boundary produce different frames.
    left = reference_frame(**{**NULL_TAG_VECTOR, "event_kind": "turn", "recorded_at_utc": "X" * 27})
    right = reference_frame(
        **{**NULL_TAG_VECTOR, "event_kind": "turnX", "recorded_at_utc": "X" * 26}
    )
    assert len(left) == len(right)
    assert left != right

    # Production emits exactly those prefixes, so it inherits that property, and it
    # additionally refuses to encode either invented field in the first place.
    frame = sqlite_spool.hre1_record_frame(**NULL_TAG_VECTOR)  # type: ignore[arg-type]
    kind = str(NULL_TAG_VECTOR["event_kind"]).encode()
    moment = str(NULL_TAG_VECTOR["recorded_at_utc"]).encode()
    kind_at = 4 + 4 + 64 + 8
    assert frame[kind_at : kind_at + 4] == struct.pack(">I", len(kind))
    assert frame[kind_at + 4 : kind_at + 4 + len(kind)] == kind
    moment_at = kind_at + 4 + len(kind)
    assert frame[moment_at : moment_at + 4] == struct.pack(">I", len(moment))
    assert frame[moment_at + 4 : moment_at + 4 + len(moment)] == moment
    for invented in ({"event_kind": "turn"}, {"recorded_at_utc": "X" * 27}):
        with pytest.raises(ValueError):
            sqlite_spool.hre1_record_frame(**{**NULL_TAG_VECTOR, **invented})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"recorded_at_utc": "2026-08-08T00:00:00.000001Z"}, id="one_microsecond"),
        pytest.param({"installation_id": "00000000-0000-4000-8000-00000000000a"}, id="install"),
        pytest.param({"producer_instance_id": "00000000-0000-4000-8000-00000000000b"}, id="prod"),
        pytest.param({"logical_session_id": "00000000-0000-4000-8000-00000000000c"}, id="session"),
        pytest.param({"event_id": "00000000-0000-4000-8000-00000000000d"}, id="event"),
        pytest.param({"event_sequence": 2}, id="sequence"),
        pytest.param({"event_kind": "binding_opened"}, id="kind"),
        pytest.param({"payload_hash": "0" * 63 + "1"}, id="payload"),
        pytest.param({"previous_hash": "00" * 32}, id="null_tag_versus_zero_hash"),
    ],
)
def test_hre1_hash_rejects_transplants_and_mutations(mutation: dict[str, object]) -> None:
    from hermes_realtime.evidence import sqlite_spool

    mutated = {**NULL_TAG_VECTOR, **mutation}
    assert sqlite_spool.hre1_record_hash(**mutated) != NULL_TAG_DIGEST  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"event_sequence": 0}, id="sequence_below_range"),
        pytest.param({"event_sequence": 9217}, id="sequence_above_range"),
        pytest.param({"event_sequence": True}, id="bool_sequence"),
        pytest.param({"payload_hash": "AB" * 32}, id="uppercase_payload_hash"),
        pytest.param({"payload_hash": "ab" * 31}, id="short_payload_hash"),
        pytest.param({"previous_hash": "AB" * 32}, id="uppercase_previous_hash"),
        pytest.param({"installation_id": "00000000000040008000000000000001"}, id="unhyphenated"),
        pytest.param({"event_id": "00000000-0000-1000-8000-000000000003"}, id="non_v4"),
        pytest.param({"recorded_at_utc": "2026-08-08T00:00:00Z"}, id="utc_without_micros"),
        pytest.param({"recorded_at_utc": "2026-08-08T00:00:00.000000+00:00"}, id="utc_offset"),
        pytest.param({"event_kind": "invented_kind"}, id="unknown_event_kind"),
    ],
)
def test_hre1_frame_is_fail_closed_on_noncanonical_fields(mutation: dict[str, object]) -> None:
    from hermes_realtime.evidence import sqlite_spool

    with pytest.raises(ValueError):
        sqlite_spool.hre1_record_frame(**{**NULL_TAG_VECTOR, **mutation})  # type: ignore[arg-type]


DENY_FILTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "hermes_realtime"
    / "evidence"
    / "deny_filter_v1.json"
)
DENY_FILTER_SHA256 = "476aaea16985b0609aea233b7c5bd883b10601a77e22ffae5b0c141cd564f0e2"
DENY_FILTER_DOCUMENT: dict[str, object] = {
    "engine": "python-re-search-v1",
    "flags": ["ASCII"],
    "patterns": [
        {
            "expression": "(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9])",
            "id": "openai_style_secret",
        },
        {
            "expression": "(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{36,255}(?![A-Za-z0-9])",
            "id": "github_token",
        },
        {
            "expression": (
                "(?i:(?<![A-Za-z0-9])bearer[ \\t]+[A-Za-z0-9._~+/=-]{20,}(?![A-Za-z0-9]))"
            ),
            "id": "bearer_credential",
        },
        {
            "expression": (
                "(?i:https?://[^\\s?#]{1,2048}\\?[^\\s#]{0,2048}"
                "(?:token|key|auth|capability)=[^\\s&#]{8,2048})"
            ),
            "id": "capability_query_url",
        },
        {
            "expression": "(?<![A-Za-z0-9])[A-Za-z]:\\\\[^\\r\\n\\u0000]{1,260}",
            "id": "windows_absolute_path",
        },
        {"expression": "\\\\\\\\[^\\r\\n\\u0000]{3,260}", "id": "windows_unc_path"},
        {
            "expression": "(?<![A-Za-z0-9])task_[A-Za-z0-9_-]{16,}(?![A-Za-z0-9])",
            "id": "private_task_handle",
        },
    ],
    "version": 1,
}
DENY_FILTER_IDS = (
    "openai_style_secret",
    "github_token",
    "bearer_credential",
    "capability_query_url",
    "windows_absolute_path",
    "windows_unc_path",
    "private_task_handle",
)


def test_deny_filter_resource_is_the_exact_pinned_canonical_bytes() -> None:
    from hermes_realtime.evidence import sqlite_spool

    raw = DENY_FILTER_PATH.read_bytes()
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert b"\r" not in raw and b"\x00" not in raw
    assert raw.decode("utf-8") == raw.decode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == DENY_FILTER_SHA256
    assert sqlite_spool.DENY_FILTER_RESOURCE_SHA256 == DENY_FILTER_SHA256
    assert raw == sqlite_spool.DENY_FILTER_CANONICAL_BYTES
    assert raw == sqlite_spool.canonical_json_bytes(DENY_FILTER_DOCUMENT) + b"\n"


def test_deny_filter_resource_is_lf_pinned_by_repository_policy() -> None:
    attributes = (DENY_FILTER_PATH.parents[3] / ".gitattributes").read_text(encoding="utf-8")
    expected = "/src/hermes_realtime/evidence/deny_filter_v1.json text eol=lf"
    assert expected in attributes.splitlines()


def test_deny_filter_loads_the_exact_typed_document_in_listed_order() -> None:
    from hermes_realtime.evidence import sqlite_spool

    deny = sqlite_spool.DenyFilterV1.load()
    assert deny.engine == "python-re-search-v1"
    assert deny.version == 1
    assert tuple(identifier for identifier, _ in deny.patterns) == DENY_FILTER_IDS
    assert all(pattern.flags & re.ASCII for _, pattern in deny.patterns)


@pytest.mark.parametrize(
    ("text", "identifier"),
    [
        pytest.param("token sk-" "abcdefghijklmnopqrst end", "openai_style_secret", id="openai"),
        pytest.param("sk-" "abcdefghijklmnopqrst", "openai_style_secret", id="openai_whole"),
        pytest.param("x sk-" "abcdefghijklmnopqrst", "openai_style_secret", id="openai_at_end"),
        pytest.param(f"ghp_{'a' * 36} tail", "github_token", id="github"),
        pytest.param(f"gho_{'Z' * 40}", "github_token", id="github_whole"),
        pytest.param(f"Bearer {'a' * 20}", "bearer_credential", id="bearer"),
        pytest.param(f"authorization: bearer {'b' * 24}", "bearer_credential", id="bearer_lower"),
        pytest.param(f"BEARER {'c' * 20}", "bearer_credential", id="bearer_upper"),
        pytest.param(
            "see https://example.test/path?token=abcdefgh now",
            "capability_query_url",
            id="capability_url",
        ),
        pytest.param(
            "HTTP://example.test/p?capability=abcdefghij",
            "capability_query_url",
            id="capability_url_upper",
        ),
        pytest.param("open C:\\Users\\me\\notes", "windows_absolute_path", id="windows_path"),
        pytest.param("C:\\a", "windows_absolute_path", id="windows_path_whole"),
        pytest.param("copy \\\\server\\share here", "windows_unc_path", id="unc"),
        pytest.param("\\\\abc", "windows_unc_path", id="unc_whole"),
        pytest.param(f"task_{'a' * 16} done", "private_task_handle", id="task"),
        pytest.param(f"task_{'a' * 20}", "private_task_handle", id="task_whole"),
    ],
)
def test_deny_filter_positive_vectors_cover_every_identifier(text: str, identifier: str) -> None:
    from hermes_realtime.evidence import sqlite_spool

    assert sqlite_spool.DenyFilterV1.load().first_match(text) == identifier


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("sk-short", id="sk_short"),
        pytest.param("github.com", id="github_com"),
        pytest.param("Bearer short", id="bearer_short"),
        pytest.param("https://example.test/?token=x", id="short_token_query"),
        pytest.param("C:relative", id="c_relative"),
        pytest.param("task_short", id="task_short"),
        pytest.param("the bearer of this message asked politely", id="ordinary_bearer_sentence"),
        pytest.param("", id="empty"),
        pytest.param("SK-abcdefghijklmnopqrst", id="uppercase_sk_is_not_case_folded"),
        pytest.param("xsk-" "abcdefghijklmnopqrst", id="left_boundary_blocked"),
        pytest.param(f"ghp_{'a' * 35}", id="github_too_short"),
        pytest.param("https://example.test/path#token=abcdefgh", id="fragment_not_query"),
    ],
)
def test_deny_filter_negative_vectors_do_not_match(text: str) -> None:
    from hermes_realtime.evidence import sqlite_spool

    assert sqlite_spool.DenyFilterV1.load().first_match(text) is None


def test_deny_filter_returns_the_first_match_in_listed_order() -> None:
    from hermes_realtime.evidence import sqlite_spool

    deny = sqlite_spool.DenyFilterV1.load()
    assert deny.first_match(f"C:\\tmp task_{'a' * 16}") == "windows_absolute_path"
    assert deny.first_match(f"task_{'a' * 16} then sk-" "abcdefghijklmnopqrst") == (
        "openai_style_secret"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param({"engine": "python-re-search-v2"}, id="wrong_engine"),
        pytest.param({"flags": ["UNICODE"]}, id="wrong_flags"),
        pytest.param({"flags": []}, id="empty_flags"),
        pytest.param({"version": 2}, id="wrong_version"),
        pytest.param({"extra": 1}, id="unknown_key"),
        pytest.param({"patterns": {}}, id="patterns_not_a_list"),
        pytest.param({"patterns": [{"id": "x"}]}, id="pattern_missing_expression"),
        pytest.param(
            {"patterns": [{"id": "x", "expression": "a", "note": "b"}]},
            id="pattern_unknown_key",
        ),
        pytest.param({"patterns": [{"id": "x", "expression": "("}]}, id="uncompilable"),
    ],
)
def test_deny_filter_parsing_is_fail_closed(mutation: dict[str, object]) -> None:
    from hermes_realtime.evidence import sqlite_spool

    document = {**DENY_FILTER_DOCUMENT, **mutation}
    with pytest.raises(ValueError):
        sqlite_spool.DenyFilterV1.from_document(document)


PINNED_SCHEMA_DDL_SHA256 = "658f62fb1866d3188685fe2bcfe885c23137882f6e05c86d73f84a57cdc3e276"
PINNED_APPLIED_SCHEMA_SHA256 = "d1f94c993131e1c9ca0a3578d5b981dd0ebaf29614f10a50fbc55cab2c7f3cfa"

UTC_TEXT = "2026-08-08T00:00:00.000000Z"
EPOCH_ID = "10000000-0000-4000-8000-000000000001"
SESSION_ID = "10000000-0000-4000-8000-000000000002"
INSTALL_ID = "10000000-0000-4000-8000-000000000003"
PRODUCER_ID = "10000000-0000-4000-8000-000000000004"
EVENT_ID = "10000000-0000-4000-8000-000000000005"
HEX64 = "ab" * 32

SEED_ROWS = (
    (
        "INSERT INTO producer_installation VALUES(1,?,?,?,0,NULL,NULL)",
        (INSTALL_ID, UTC_TEXT, UTC_TEXT),
    ),
    ("INSERT INTO consent_epochs VALUES(?,?,'active',?,NULL)", (EPOCH_ID, PRODUCER_ID, UTC_TEXT)),
    (
        "INSERT INTO evidence_sessions VALUES"
        "(?,?,?,?,?,'realtime-evidence-consent-v1','open',NULL,NULL,0,0,NULL)",
        (SESSION_ID, EPOCH_ID, PRODUCER_ID, UTC_TEXT, UTC_TEXT),
    ),
)


@pytest.fixture
def schema_connection(tmp_path: Path):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import sqlite_spool

    connection = sqlite_spool.create_store_database(tmp_path / "capture-v1.sqlite3")
    try:
        for statement, values in SEED_ROWS:
            connection.execute(statement, values)
        connection.commit()
        yield connection
    finally:
        connection.close()


def test_store_database_declares_the_exact_identity_and_pragmas(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool

    assert sqlite_spool.APPLICATION_ID == 0x48524531
    assert sqlite_spool.USER_VERSION == 1
    connection = sqlite_spool.create_store_database(tmp_path / "capture-v1.sqlite3")
    try:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == 0x48524531
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert sqlite_spool.read_store_pragmas(connection) == dict(
            sqlite_spool.REQUIRED_PRAGMAS
        )
        assert dict(sqlite_spool.REQUIRED_PRAGMAS) == {
            "journal_mode": "delete",
            "synchronous": 3,
            "encoding": "UTF-8",
            "page_size": 4096,
            "max_page_count": 24576,
            "foreign_keys": 1,
            "busy_timeout": 0,
            "trusted_schema": 0,
            "secure_delete": 1,
            "temp_store": 2,
        }
    finally:
        connection.close()


def test_normalized_ddl_digest_is_pinned_and_matches_the_applied_schema(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool

    normalized = sqlite_spool.normalize_schema_ddl(sqlite_spool.SCHEMA_DDL_V1)
    assert normalized == " ".join(normalized.split())
    assert normalized == sqlite_spool.NORMALIZED_SCHEMA_DDL_V1
    assert hashlib.sha256(normalized.encode()).hexdigest() == sqlite_spool.SCHEMA_DDL_SHA256
    assert sqlite_spool.SCHEMA_DDL_SHA256 == PINNED_SCHEMA_DDL_SHA256

    connection = sqlite_spool.create_store_database(tmp_path / "capture-v1.sqlite3")
    try:
        assert sqlite_spool.applied_schema_digest(connection) == PINNED_APPLIED_SCHEMA_SHA256
        sqlite_spool.validate_store_schema(connection)
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert names == {
            "producer_installation",
            "consent_epochs",
            "evidence_sessions",
            "evidence_events",
            "evidence_conflicts",
            "erasure_requests",
            "erasure_tombstones",
        }
    finally:
        connection.close()


def test_schema_declares_no_profile_audio_or_identity_surface(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool

    forbidden = (
        "profile",
        "audio",
        "pcm",
        "waveform",
        "sample",
        "voice",
        "speaker",
        "participant",
        "identity",
        "hermes_home",
        "path",
        "url",
        "token",
        "model",
        "provider",
        "tool",
        "task",
        "learning",
        "approval",
        "telemetry",
        "latency",
    )
    lowered = sqlite_spool.NORMALIZED_SCHEMA_DDL_V1.lower()
    assert [word for word in forbidden if word in lowered] == []

    connection = sqlite_spool.create_store_database(tmp_path / "capture-v1.sqlite3")
    try:
        columns: list[str] = []
        for (table,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall():
            columns.extend(
                str(row[1]).lower()
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            )
        assert [name for name in columns if any(word in name for word in forbidden)] == []
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("statement", "values"),
    [
        pytest.param(
            "INSERT INTO producer_installation VALUES(2,?,?,?,0,NULL,NULL)",
            ("20000000-0000-4000-8000-000000000001", UTC_TEXT, UTC_TEXT),
            id="installation_not_singleton",
        ),
        pytest.param(
            "INSERT INTO producer_installation VALUES(1,?,?,?,1,NULL,NULL)",
            ("20000000-0000-4000-8000-000000000002", UTC_TEXT, UTC_TEXT),
            id="purge_required_without_reason",
        ),
        pytest.param(
            "INSERT INTO producer_installation VALUES(1,?,?,?,1,'ttl','store')",
            ("20000000-0000-4000-8000-000000000003", UTC_TEXT, UTC_TEXT),
            id="purge_reason_is_not_clock_rollback",
        ),
        pytest.param(
            "INSERT INTO producer_installation VALUES(1,?,?,?,0,NULL,NULL)",
            ("10000000-0000-4000-8000-00000000000A", UTC_TEXT, UTC_TEXT),
            id="installation_uuid_is_uppercase",
        ),
        pytest.param(
            "INSERT INTO producer_installation VALUES(1,?,'2026-08-08T00:00:00Z',?,0,NULL,NULL)",
            ("20000000-0000-4000-8000-000000000004", UTC_TEXT),
            id="installation_time_lacks_microseconds",
        ),
        pytest.param(
            "INSERT INTO consent_epochs VALUES(?,?,'invented',?,NULL)",
            ("20000000-0000-4000-8000-000000000005", PRODUCER_ID, UTC_TEXT),
            id="epoch_state_is_invented",
        ),
        pytest.param(
            "INSERT INTO consent_epochs VALUES(?,?,'active',?,?)",
            ("20000000-0000-4000-8000-000000000006", PRODUCER_ID, UTC_TEXT, UTC_TEXT),
            id="active_epoch_has_a_close_time",
        ),
        pytest.param(
            "INSERT INTO consent_epochs VALUES(?,?,'closed',?,NULL)",
            ("20000000-0000-4000-8000-000000000007", PRODUCER_ID, UTC_TEXT),
            id="closed_epoch_lacks_a_close_time",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v1','open',9,NULL,0,0,NULL)",
            ("20000000-0000-4000-8000-000000000008", EPOCH_ID, PRODUCER_ID, UTC_TEXT, UTC_TEXT),
            id="open_session_has_a_final_sequence",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v1','sealed',9,NULL,0,9,NULL)",
            ("20000000-0000-4000-8000-000000000009", EPOCH_ID, PRODUCER_ID, UTC_TEXT, UTC_TEXT),
            id="sealed_session_lacks_a_head_hash",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v1','tainted',NULL,NULL,0,0,'invented')",
            ("20000000-0000-4000-8000-00000000000b", EPOCH_ID, PRODUCER_ID, UTC_TEXT, UTC_TEXT),
            id="session_taint_code_is_invented",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v2','open',NULL,NULL,0,0,NULL)",
            ("20000000-0000-4000-8000-00000000000c", EPOCH_ID, PRODUCER_ID, UTC_TEXT, UTC_TEXT),
            id="session_consent_version_is_wrong",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v1','open',NULL,NULL,0,9217,NULL)",
            ("20000000-0000-4000-8000-00000000000d", EPOCH_ID, PRODUCER_ID, UTC_TEXT, UTC_TEXT),
            id="session_event_count_exceeds_its_cap",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v1','open',NULL,NULL,-1,0,NULL)",
            ("20000000-0000-4000-8000-00000000000e", EPOCH_ID, PRODUCER_ID, UTC_TEXT, UTC_TEXT),
            id="session_canonical_bytes_is_negative",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v1','open',NULL,NULL,0,0,NULL)",
            (
                "20000000-0000-4000-8000-00000000000f",
                "30000000-0000-4000-8000-000000000001",
                PRODUCER_ID,
                UTC_TEXT,
                UTC_TEXT,
            ),
            id="session_references_a_missing_epoch",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v1','sealed',2,?,0,1,NULL)",
            (
                "20000000-0000-4000-8000-00000000002a",
                EPOCH_ID,
                PRODUCER_ID,
                UTC_TEXT,
                UTC_TEXT,
                HEX64,
            ),
            id="sealed_final_sequence_disagrees_with_event_count",
        ),
        pytest.param(
            "INSERT INTO evidence_sessions VALUES"
            "(?,?,?,?,?,'realtime-evidence-consent-v1','tainted',2,?,0,1,'writer_fault')",
            (
                "20000000-0000-4000-8000-00000000002b",
                EPOCH_ID,
                PRODUCER_ID,
                UTC_TEXT,
                UTC_TEXT,
                HEX64,
            ),
            id="quarantined_final_sequence_disagrees_with_event_count",
        ),
        pytest.param(
            "INSERT INTO evidence_events VALUES(?,?,1,'invented_kind',?,'{}',?,NULL,?,2)",
            ("20000000-0000-4000-8000-000000000010", SESSION_ID, UTC_TEXT, HEX64, HEX64),
            id="event_kind_is_invented",
        ),
        pytest.param(
            "INSERT INTO evidence_events VALUES(?,?,0,'turn_opened',?,'{}',?,NULL,?,2)",
            ("20000000-0000-4000-8000-000000000011", SESSION_ID, UTC_TEXT, HEX64, HEX64),
            id="event_sequence_below_range",
        ),
        pytest.param(
            "INSERT INTO evidence_events VALUES(?,?,9217,'turn_opened',?,'{}',?,NULL,?,2)",
            ("20000000-0000-4000-8000-000000000012", SESSION_ID, UTC_TEXT, HEX64, HEX64),
            id="event_sequence_above_range",
        ),
        pytest.param(
            "INSERT INTO evidence_events VALUES(?,?,1,'turn_opened',?,'{}',?,NULL,?,7)",
            ("20000000-0000-4000-8000-000000000013", SESSION_ID, UTC_TEXT, HEX64, HEX64),
            id="canonical_bytes_disagree_with_the_payload",
        ),
        pytest.param(
            "INSERT INTO evidence_events VALUES(?,?,1,'turn_opened',?,'{}','AB',NULL,?,2)",
            ("20000000-0000-4000-8000-000000000014", SESSION_ID, UTC_TEXT, HEX64),
            id="payload_hash_is_not_64_lowercase_hex",
        ),
        pytest.param(
            "INSERT INTO evidence_events VALUES(?,?,1,'turn_opened',?,'{}',?,'zz',?,2)",
            ("20000000-0000-4000-8000-000000000015", SESSION_ID, UTC_TEXT, HEX64, HEX64),
            id="previous_hash_is_not_64_lowercase_hex",
        ),
        pytest.param(
            "INSERT INTO evidence_events VALUES(?,?,1,'turn_opened',?,'{}',?,NULL,?,2)",
            (
                "20000000-0000-4000-8000-000000000016",
                "30000000-0000-4000-8000-000000000002",
                UTC_TEXT,
                HEX64,
                HEX64,
            ),
            id="event_references_a_missing_session",
        ),
        pytest.param(
            "INSERT INTO evidence_conflicts VALUES(?,?,?,'invented',?)",
            (
                "20000000-0000-4000-8000-000000000017",
                SESSION_ID,
                EVENT_ID,
                UTC_TEXT,
            ),
            id="conflict_reason_is_invented",
        ),
        pytest.param(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, resume_state, erased_session_count,"
            " erased_event_count) VALUES(?,'session',?,?,'ttl','pending',NULL,0,0)",
            ("20000000-0000-4000-8000-000000000018", SESSION_ID, UTC_TEXT),
            id="pending_request_carries_counts",
        ),
        pytest.param(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, resume_state, erased_session_count,"
            " erased_event_count) VALUES"
            "(?,'session',?,?,'ttl','logical_deleted',NULL,NULL,NULL)",
            ("20000000-0000-4000-8000-000000000019", SESSION_ID, UTC_TEXT),
            id="deleted_request_lacks_counts",
        ),
        pytest.param(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, resume_state, erased_session_count,"
            " erased_event_count) VALUES"
            "(?,'session',?,?,'ttl','purge_failed',NULL,NULL,NULL)",
            ("20000000-0000-4000-8000-00000000001a", SESSION_ID, UTC_TEXT),
            id="failed_request_lacks_resume_state",
        ),
        pytest.param(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, resume_state, erased_session_count,"
            " erased_event_count) VALUES"
            "(?,'consent_epoch',?,?,'ttl','pending',NULL,NULL,NULL)",
            ("20000000-0000-4000-8000-00000000001b", EPOCH_ID, UTC_TEXT),
            id="ttl_request_uses_an_epoch_scope",
        ),
        pytest.param(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, resume_state, erased_session_count,"
            " erased_event_count) VALUES"
            "(?,'store','not-store',?,'clock_rollback','pending',NULL,NULL,NULL)",
            ("20000000-0000-4000-8000-00000000001c", UTC_TEXT),
            id="clock_rollback_scope_key_is_not_store",
        ),
        pytest.param(
            "INSERT INTO erasure_tombstones (erasure_request_id, scope_kind, scope_key,"
            " reason_code, last_admission_ordinal, erased_at_utc, erased_session_count,"
            " erased_event_count) VALUES(?,'session',?,'ttl',1,?,-1,0)",
            ("20000000-0000-4000-8000-00000000001d", SESSION_ID, UTC_TEXT),
            id="tombstone_count_is_negative",
        ),
        pytest.param(
            "INSERT INTO erasure_tombstones (erasure_request_id, scope_kind, scope_key,"
            " reason_code, last_admission_ordinal, erased_at_utc, erased_session_count,"
            " erased_event_count) VALUES(?,'session',?,'invented',1,?,0,0)",
            ("20000000-0000-4000-8000-00000000001e", SESSION_ID, UTC_TEXT),
            id="tombstone_reason_is_invented",
        ),
    ],
)
def test_direct_relational_inserts_are_rejected(
    schema_connection, statement: str, values: tuple[object, ...]  # type: ignore[no-untyped-def]
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(statement, values)


def test_direct_relational_uniqueness_and_append_only_are_enforced(
    schema_connection,  # type: ignore[no-untyped-def]
) -> None:
    event = "INSERT INTO evidence_events VALUES(?,?,?,'turn_opened',?,'{}',?,NULL,?,2)"
    schema_connection.execute(event, (EVENT_ID, SESSION_ID, 1, UTC_TEXT, HEX64, HEX64))
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(
            event,
            ("20000000-0000-4000-8000-000000000020", SESSION_ID, 1, UTC_TEXT, HEX64, HEX64),
        )
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(event, (EVENT_ID, SESSION_ID, 2, UTC_TEXT, HEX64, HEX64))
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(
            "UPDATE evidence_events SET canonical_payload='{\"a\":1}' WHERE event_id=?",
            (EVENT_ID,),
        )

    conflict = "INSERT INTO evidence_conflicts VALUES(?,?,?,'cross_session_event_id',?)"
    schema_connection.execute(
        conflict,
        ("20000000-0000-4000-8000-000000000021", SESSION_ID, EVENT_ID, UTC_TEXT),
    )
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(
            conflict,
            ("20000000-0000-4000-8000-000000000022", SESSION_ID, EVENT_ID, UTC_TEXT),
        )

    request = (
        "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
        " requested_at_utc, reason_code, state, control_fingerprint_hash,"
        " ttl_consent_epoch_id, ttl_expires_at_utc,"
        " last_admission_ordinal, resume_state,"
        " erased_session_count, erased_event_count)"
        " VALUES(?,'session',?,?,'ttl','pending',?,?,?,1,NULL,NULL,NULL)"
    )
    schema_connection.execute(
        request,
        (
            "20000000-0000-4000-8000-000000000023",
            SESSION_ID,
            UTC_TEXT,
            HEX64,
            EPOCH_ID,
            UTC_TEXT,
        ),
    )
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(
            request,
            (
                "20000000-0000-4000-8000-000000000024",
                SESSION_ID,
                UTC_TEXT,
                HEX64,
                EPOCH_ID,
                UTC_TEXT,
            ),
        )


def test_erasure_tombstone_retains_exact_revoke_authority(schema_connection) -> None:  # type: ignore[no-untyped-def]
    request_id = "20000000-0000-4000-8000-000000000025"
    schema_connection.execute(
        "INSERT INTO erasure_tombstones (erasure_request_id, scope_kind, scope_key,"
        " reason_code, control_sequence, control_fingerprint_hash, last_admission_ordinal,"
        " final_admission_ordinal, erased_at_utc, erased_session_count, erased_event_count)"
        " VALUES (?, 'consent_epoch', ?, 'revoked', 7, ?, 11, 13, ?, 1, 2)",
        (request_id, EPOCH_ID, HEX64, UTC_TEXT),
    )
    assert schema_connection.execute(
        "SELECT erasure_request_id, scope_kind, scope_key, reason_code, control_sequence,"
        " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
        " erased_session_count, erased_event_count FROM erasure_tombstones"
    ).fetchall() == [
        (request_id, "consent_epoch", EPOCH_ID, "revoked", 7, HEX64, 11, 13, 1, 2)
    ]
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(
            "UPDATE erasure_tombstones SET erased_event_count=3 WHERE erasure_request_id=?",
            (request_id,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(
            "DELETE FROM erasure_tombstones WHERE erasure_request_id=?",
            (request_id,),
        )


def test_erasure_request_identity_is_immutable_and_final_watermark_sets_once(
    schema_connection,  # type: ignore[no-untyped-def]
) -> None:
    request_id = "20000000-0000-4000-8000-000000000027"
    schema_connection.execute(
        "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
        " requested_at_utc, reason_code, state, control_sequence, control_fingerprint_hash,"
        " last_admission_ordinal) VALUES (?, 'consent_epoch', ?, ?, 'revoked', 'pending',"
        " 7, ?, 11)",
        (request_id, EPOCH_ID, UTC_TEXT, HEX64),
    )
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(
            "UPDATE erasure_requests SET control_sequence=8 WHERE erasure_request_id=?",
            (request_id,),
        )
    schema_connection.execute(
        "UPDATE erasure_requests SET final_admission_ordinal=13 WHERE erasure_request_id=?",
        (request_id,),
    )
    with pytest.raises(sqlite3.IntegrityError):
        schema_connection.execute(
            "UPDATE erasure_requests SET final_admission_ordinal=14 WHERE erasure_request_id=?",
            (request_id,),
        )
    schema_connection.execute(
        "UPDATE erasure_requests SET state='logical_deleted', erased_session_count=0,"
        " erased_event_count=0 WHERE erasure_request_id=?",
        (request_id,),
    )


def test_erasure_request_retains_a_generic_admission_watermark(schema_connection) -> None:  # type: ignore[no-untyped-def]
    request_id = "20000000-0000-4000-8000-000000000026"
    schema_connection.execute(
        "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
        " requested_at_utc, reason_code, state, control_fingerprint_hash,"
        " ttl_consent_epoch_id, ttl_expires_at_utc,"
        " last_admission_ordinal)"
        " VALUES (?, 'session', ?, ?, 'ttl', 'pending', ?, ?, ?, 17)",
        (request_id, SESSION_ID, UTC_TEXT, HEX64, EPOCH_ID, UTC_TEXT),
    )
    assert schema_connection.execute(
        "SELECT erasure_request_id, reason_code, last_admission_ordinal"
        " FROM erasure_requests WHERE erasure_request_id=?",
        (request_id,),
    ).fetchone() == (request_id, "ttl", 17)


def test_sealed_sessions_must_agree_with_their_event_count(
    schema_connection,  # type: ignore[no-untyped-def]
) -> None:
    insert = (
        "INSERT INTO evidence_sessions VALUES"
        "(?,?,?,?,?,'realtime-evidence-consent-v1',?,?,?,0,?,?)"
    )

    def session(identifier: str, state: str, final: int | None, head: str | None,
                count: int, taint: str | None) -> tuple[object, ...]:
        return (identifier, EPOCH_ID, PRODUCER_ID, UTC_TEXT, UTC_TEXT, state, final, head,
                count, taint)

    accepted = (
        session("30000000-0000-4000-8000-000000000010", "sealed", 4, HEX64, 4, None),
        session("30000000-0000-4000-8000-000000000011", "open", None, None, 3, None),
        session("30000000-0000-4000-8000-000000000012", "tainted", None, None, 5, "oversize"),
        session("30000000-0000-4000-8000-000000000013", "tainted", 7, HEX64, 7, "purge_failed"),
    )
    for values in accepted:
        schema_connection.execute(insert, values)
    schema_connection.commit()
    assert schema_connection.execute(
        "SELECT COUNT(*) FROM evidence_sessions"
    ).fetchone()[0] == len(accepted) + 1

    for values in (
        session("30000000-0000-4000-8000-000000000020", "sealed", 3, HEX64, 4, None),
        session("30000000-0000-4000-8000-000000000021", "sealed", 5, HEX64, 4, None),
    ):
        with pytest.raises(sqlite3.IntegrityError):
            schema_connection.execute(insert, values)


def test_store_preflight_rejects_a_foreign_or_corrupt_database(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.storage_security import EvidenceStorageError, WriterFault

    absent = tmp_path / "absent" / "capture-v1.sqlite3"
    absent.parent.mkdir()
    with pytest.raises(EvidenceStorageError) as caught:
        sqlite_spool.preflight_store_database(absent)
    assert caught.value.fault is WriterFault.STORE_CORRUPT

    foreign = tmp_path / "foreign" / "capture-v1.sqlite3"
    foreign.parent.mkdir()
    other = sqlite3.connect(foreign)
    other.execute("CREATE TABLE unrelated(x INTEGER)")
    other.commit()
    other.close()
    with pytest.raises(EvidenceStorageError) as caught:
        sqlite_spool.preflight_store_database(foreign)
    assert caught.value.fault is WriterFault.STORE_CORRUPT

    garbage = tmp_path / "garbage" / "capture-v1.sqlite3"
    garbage.parent.mkdir()
    garbage.write_bytes(b"not a database at all")
    with pytest.raises(EvidenceStorageError) as caught:
        sqlite_spool.preflight_store_database(garbage)
    assert caught.value.fault is WriterFault.STORE_CORRUPT


def test_store_preflight_accepts_the_owned_store_and_its_hot_journal(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool

    database = tmp_path / "capture-v1.sqlite3"
    sqlite_spool.create_store_database(database).close()
    assert sqlite_spool.preflight_store_database(database) is None

    Path(f"{database}-journal").write_bytes(b"\xd9\xd5\x05\xf9\x20\xa1\x63\xd7" + b"\x00" * 64)
    assert sqlite_spool.preflight_store_database(database) is None


HOT_JOURNAL_HEADER = b"\xd9\xd5\x05\xf9\x20\xa1\x63\xd7" + b"\x00" * 64


def _seed_store(root: Path) -> Path:
    from hermes_realtime.evidence import sqlite_spool

    root.mkdir(parents=True, exist_ok=True)
    database = root / "capture-v1.sqlite3"
    sqlite_spool.create_store_database(database).close()
    return database


def test_preflight_targets_the_literal_percent_path_not_its_uri_alias(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.storage_security import EvidenceStorageError, WriterFault

    # A valid store at the percent-decoded alias must never rescue a corrupt literal.
    _seed_store(tmp_path / "a b")
    literal_root = tmp_path / "a%20b"
    literal_root.mkdir()
    literal = literal_root / "capture-v1.sqlite3"
    literal.write_bytes(b"this is not a database at all")

    with pytest.raises(EvidenceStorageError) as caught:
        sqlite_spool.preflight_store_database(literal)
    assert caught.value.fault is WriterFault.STORE_CORRUPT

    # And the reverse: a literal percent root holding a real store must validate.
    percent = _seed_store(tmp_path / "%41")
    assert not (tmp_path / "A").exists()
    assert sqlite_spool.preflight_store_database(percent) is None


def test_preflight_reports_a_percent_alias_store_as_absent(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.storage_security import EvidenceStorageError, WriterFault

    _seed_store(tmp_path / "b c")
    missing = tmp_path / "b%20c" / "capture-v1.sqlite3"
    missing.parent.mkdir()
    with pytest.raises(EvidenceStorageError) as caught:
        sqlite_spool.preflight_store_database(missing)
    assert caught.value.fault is WriterFault.STORE_CORRUPT


def test_hot_journal_lookup_uses_the_same_literal_target(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool

    root = tmp_path / "%41"
    database = _seed_store(root)
    alias = tmp_path / "A"
    alias.mkdir()
    (alias / "capture-v1.sqlite3-journal").write_bytes(HOT_JOURNAL_HEADER)

    assert sqlite_spool.hot_journal_path(database) == root / "capture-v1.sqlite3-journal"
    assert sqlite_spool.hot_journal_present(database) is False

    (root / "capture-v1.sqlite3-journal").write_bytes(HOT_JOURNAL_HEADER)
    assert sqlite_spool.hot_journal_present(database) is True
    assert sqlite_spool.preflight_store_database(database) is None


def test_database_header_identity_is_read_from_the_literal_file(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.storage_security import EvidenceStorageError, WriterFault

    database = _seed_store(tmp_path / "%41")
    assert sqlite_spool.validate_database_header(database) is None

    foreign = tmp_path / "foreign" / "capture-v1.sqlite3"
    foreign.parent.mkdir()
    other = sqlite3.connect(foreign)
    other.execute("CREATE TABLE unrelated(x INTEGER)")
    other.commit()
    other.close()
    with pytest.raises(EvidenceStorageError) as caught:
        sqlite_spool.validate_database_header(foreign)
    assert caught.value.fault is WriterFault.STORE_CORRUPT

    truncated = tmp_path / "short" / "capture-v1.sqlite3"
    truncated.parent.mkdir()
    truncated.write_bytes(b"SQLite format 3\x00")
    with pytest.raises(EvidenceStorageError):
        sqlite_spool.validate_database_header(truncated)


# --------------------------------------------------------------------------------------
# The owned single-writer transport.
# --------------------------------------------------------------------------------------

BINDING_ID = "10000000-0000-4000-8000-000000000006"
TURN_ID = "10000000-0000-4000-8000-000000000007"
UTTERANCE_ID = "10000000-0000-4000-8000-000000000008"
SUCCESSOR_ID = "10000000-0000-4000-8000-000000000009"
DISCLOSURE = "cd" * 32
START = datetime(2026, 8, 8, tzinfo=UTC)


def event_uuid(index: int) -> str:
    return f"40000000-0000-4000-8000-{index:012d}"


class StubClock:
    """A counting clock so idempotency can prove it never allocates a timestamp."""

    def __init__(self, start: datetime = START, step: timedelta = timedelta(seconds=1)) -> None:
        self._next = start
        self._step = step
        self.calls = 0
        self.override: list[datetime] = []

    def __call__(self) -> datetime:
        self.calls += 1
        if self.override:
            return self.override.pop(0)
        current = self._next
        self._next = current + self._step
        return current


class StubUuids:
    def __init__(self) -> None:
        self.issued = 0

    def __call__(self) -> str:
        self.issued += 1
        return f"50000000-0000-4000-8000-{self.issued:012d}"


def make_snapshot(  # type: ignore[no-untyped-def]
    kind,
    sequence: int,
    payload,
    *,
    event_id: str,
    session: str = SESSION_ID,
    installation: str = INSTALL_ID,
    producer: str = PRODUCER_ID,
):
    from hermes_realtime.evidence.models import EvidenceSnapshotV1

    return EvidenceSnapshotV1(
        schema_version=1,
        installation_id=installation,
        producer_instance_id=producer,
        logical_session_id=session,
        event_id=event_id,
        event_sequence=sequence,
        event_kind=kind,
        payload=payload,
    )


def opening_snapshots(*, session: str = SESSION_ID, predecessor: str | None = None, first: int = 1):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingOpenedPayloadV1,
        EventKind,
        SessionOpenedPayloadV1,
    )

    opened = make_snapshot(
        EventKind.SESSION_OPENED,
        1,
        SessionOpenedPayloadV1(
            consent_epoch_id=EPOCH_ID,
            binding_id=BINDING_ID,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=DISCLOSURE,
            retention_hours=24,
            microphone_accepted=False,
            typed_accepted=True,
            predecessor_session_id=predecessor,
        ),
        event_id=event_uuid(first),
        session=session,
    )
    binding = make_snapshot(
        EventKind.BINDING_OPENED,
        2,
        BindingOpenedPayloadV1(
            binding_id=BINDING_ID,
            binding_generation=1,
            microphone_available=False,
            typed_available=True,
        ),
        event_id=event_uuid(first + 1),
        session=session,
    )
    return opened, binding


def make_create_epoch():  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import CreateEpochV1

    opened, binding = opening_snapshots()
    return CreateEpochV1(
        protocol_version=1,
        installation_id=INSTALL_ID,
        producer_instance_id=PRODUCER_ID,
        consent_epoch_id=EPOCH_ID,
        logical_session_id=SESSION_ID,
        binding_id=BINDING_ID,
        binding_generation=1,
        consent_version="realtime-evidence-consent-v1",
        disclosure_digest=DISCLOSURE,
        retention_hours=24,
        microphone_accepted=False,
        typed_accepted=True,
        control_sequence=1,
        control_fingerprint_hash=HEX64,
        session_opened=opened,
        binding_opened=binding,
    )


def turn_opened_snapshot(sequence: int, *, event_index: int, session: str = SESSION_ID):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import EventKind, TurnKind, TurnOpenedPayloadV1

    return make_snapshot(
        EventKind.TURN_OPENED,
        sequence,
        TurnOpenedPayloadV1(
            evidence_turn_id=TURN_ID,
            turn_kind=TurnKind.USER_RESPONSE,
            utterance_id=UTTERANCE_ID,
            replay_of_evidence_turn_id=None,
        ),
        event_id=event_uuid(event_index),
        session=session,
    )


def user_final_snapshot(sequence: int, *, event_index: int, text: str = "hello there"):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import EventKind, InputSource, UserFinalAcceptedPayloadV1

    return make_snapshot(
        EventKind.USER_FINAL_ACCEPTED,
        sequence,
        UserFinalAcceptedPayloadV1(
            utterance_id=UTTERANCE_ID,
            evidence_turn_id=TURN_ID,
            source=InputSource.TYPED,
            routing_disposition="response",
            text=text,
        ),
        event_id=event_uuid(event_index),
    )


def ordinary_record(snapshot, ordinal: int):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import QueuedEvidenceRecordV1, QueueReservationClass

    return QueuedEvidenceRecordV1(
        protocol_version=1,
        snapshot=snapshot,
        admission_ordinal=ordinal,
        reservation_class=QueueReservationClass.ORDINARY,
        lease_open_ordinal=None,
    )


def make_spool(
    tmp_path: Path,
    *,
    clock: StubClock | None = None,
    probe: InjectedStorageProbe | None = None,
    **kwargs: object,
):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import sqlite_spool

    root = tmp_path / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    return sqlite_spool.SQLiteEvidenceSpool(
        root / "capture-v1.sqlite3",
        clock=clock or StubClock(),
        uuid_factory=StubUuids(),
        probe=probe or InjectedStorageProbe(),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def spool(tmp_path: Path):  # type: ignore[no-untyped-def]
    owned = make_spool(tmp_path)
    try:
        yield owned
    finally:
        owned.close()


def rows(spool, statement: str, values: tuple[object, ...] = ()):  # type: ignore[no-untyped-def]
    return spool.connection.execute(statement, values).fetchall()


def test_create_epoch_creates_the_owned_store_and_persists_two_opening_rows(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import sqlite_spool, storage_security
    from hermes_realtime.evidence.models import StoreDisposition

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        root = tmp_path / "evidence"
        marker = (root / ".hermes-realtime-evidence-root-v1").read_bytes()
        assert storage_security.parse_root_marker(marker)
        assert (root / "capture-v1.sqlite3").is_file()

        installation = rows(owned, "SELECT * FROM producer_installation")
        assert installation == [(1, INSTALL_ID, UTC_TEXT, UTC_TEXT, 0, None, None)]
        assert rows(owned, "SELECT consent_epoch_id, state FROM consent_epochs") == [
            (EPOCH_ID, "active")
        ]
        session = rows(
            owned,
            "SELECT logical_session_id, state, opened_at_utc, expires_at_utc, event_count,"
            " head_hash, canonical_bytes FROM evidence_sessions",
        )[0]
        assert session[:3] == (SESSION_ID, "open", UTC_TEXT)
        assert session[3] == "2026-08-09T00:00:00.000000Z"
        assert session[4] == 2

        events = rows(
            owned,
            "SELECT event_sequence, event_kind, previous_hash, record_hash, canonical_bytes"
            " FROM evidence_events ORDER BY event_sequence",
        )
        assert [row[0] for row in events] == [1, 2]
        assert [row[1] for row in events] == ["session_opened", "binding_opened"]
        assert events[0][2] is None
        assert events[1][2] == events[0][3]
        # §5.2 forbids a head hash while the session is open; the chain lives in the rows.
        assert session[5] is None
        assert session[6] == events[0][4] + events[1][4]
        assert sqlite_spool.preflight_store_database(root / "capture-v1.sqlite3") is None
        # The retained lease denies every outside read while this owner holds it.
        with pytest.raises(PermissionError):
            (root / "capture-v1.owner").read_bytes()
    finally:
        owned.close()
    sentinel = storage_security.decode_sentinel_image((root / "capture-v1.owner").read_bytes())
    assert sentinel.active.state is storage_security.SentinelState.CLEAR


def test_create_epoch_is_idempotent_and_refuses_a_second_epoch(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import StoreDisposition

    command = make_create_epoch()
    assert spool.create_epoch(command) is StoreDisposition.COMMITTED
    assert spool.create_epoch(command) is StoreDisposition.IDEMPOTENT
    assert rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0] == 2

    from hermes_realtime.evidence.models import CreateEpochV1

    other = CreateEpochV1(
        **{
            **{
                field: getattr(command, field)
                for field in (
                    "protocol_version",
                    "installation_id",
                    "producer_instance_id",
                    "binding_id",
                    "binding_generation",
                    "consent_version",
                    "disclosure_digest",
                    "retention_hours",
                    "microphone_accepted",
                    "typed_accepted",
                    "control_sequence",
                    "control_fingerprint_hash",
                )
            },
            "consent_epoch_id": EPOCH_ID,
            "logical_session_id": SESSION_ID,
            "session_opened": command.session_opened,
            "binding_opened": command.binding_opened,
        }
    )
    assert spool.create_epoch(other) is StoreDisposition.IDEMPOTENT


def test_append_record_persists_an_exact_chained_row(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import StoreDisposition, evidence_snapshot_to_primitive

    spool.create_epoch(make_create_epoch())
    snapshot = turn_opened_snapshot(3, event_index=10)
    assert spool.append_record(ordinary_record(snapshot, 3)) is StoreDisposition.COMMITTED

    row = rows(
        spool,
        "SELECT event_id, event_sequence, event_kind, canonical_payload, payload_hash,"
        " previous_hash, record_hash, canonical_bytes, recorded_at_utc"
        " FROM evidence_events WHERE event_sequence=3",
    )[0]
    payload = sqlite_spool.canonical_json_bytes(
        evidence_snapshot_to_primitive(snapshot)["payload"]
    )
    assert row[3].encode() == payload
    assert row[4] == hashlib.sha256(payload).hexdigest()
    assert row[7] == len(payload)
    assert row[6] == sqlite_spool.hre1_record_hash(
        installation_id=INSTALL_ID,
        producer_instance_id=PRODUCER_ID,
        event_id=event_uuid(10),
        logical_session_id=SESSION_ID,
        event_sequence=3,
        event_kind="turn_opened",
        recorded_at_utc=row[8],
        payload_hash=row[4],
        previous_hash=row[5],
    )
    previous = rows(
        spool, "SELECT record_hash FROM evidence_events WHERE event_sequence=2"
    )[0][0]
    assert row[5] == previous
    session = rows(
        spool, "SELECT event_count, head_hash, canonical_bytes FROM evidence_sessions"
    )[0]
    assert session[0] == 3
    assert session[1] is None
    high_water = rows(spool, "SELECT clock_high_water_utc FROM producer_installation")[0][0]
    assert high_water == row[8]


def test_append_record_idempotency_precedes_timestamp_allocation(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        owned.create_epoch(make_create_epoch())
        record = ordinary_record(turn_opened_snapshot(3, event_index=10), 3)
        assert owned.append_record(record) is StoreDisposition.COMMITTED
        after_first = clock.calls
        assert owned.append_record(record) is StoreDisposition.IDEMPOTENT
        assert clock.calls == after_first
        assert rows(owned, "SELECT event_count FROM evidence_sessions")[0][0] == 3
        assert rows(owned, "SELECT COUNT(*) FROM evidence_conflicts")[0][0] == 0
    finally:
        owned.close()


def test_append_binding_close_persists_the_enclosed_snapshot(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingClosedPayloadV1,
        BindingCloseReason,
        BindingCloseV1,
        EventKind,
        StoreDisposition,
    )

    spool.create_epoch(make_create_epoch())
    snapshot = make_snapshot(
        EventKind.BINDING_CLOSED,
        3,
        BindingClosedPayloadV1(
            binding_id=BINDING_ID,
            close_reason=BindingCloseReason.CLIENT_CLOSED,
        ),
        event_id=event_uuid(20),
    )
    command = BindingCloseV1(
        protocol_version=1,
        binding_id=BINDING_ID,
        consent_epoch_id=EPOCH_ID,
        logical_session_id=SESSION_ID,
        admission_ordinal=3,
        snapshot=snapshot,
    )
    assert spool.append_binding_close(command) is StoreDisposition.COMMITTED
    assert spool.append_binding_close(command) is StoreDisposition.IDEMPOTENT
    assert rows(spool, "SELECT event_kind FROM evidence_events WHERE event_sequence=3") == [
        ("binding_closed",)
    ]


def test_append_record_rejects_foreign_objects_without_raising(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import StoreDisposition

    spool.create_epoch(make_create_epoch())
    assert spool.append_record(object()) is StoreDisposition.REJECTED_STATE
    assert spool.append_binding_close(object()) is StoreDisposition.REJECTED_STATE
    assert spool.create_epoch(object()) is StoreDisposition.REJECTED_STATE
    assert spool.diagnostics().sticky_fault is None


def test_event_id_envelope_mismatch_conflicts_and_taints_without_storing_text(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import StoreDisposition

    spool.create_epoch(make_create_epoch())
    spool.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
    spool.append_record(ordinary_record(user_final_snapshot(4, event_index=11), 4))
    original = rows(spool, "SELECT canonical_payload FROM evidence_events WHERE event_sequence=4")

    changed = user_final_snapshot(4, event_index=11, text="a completely different secret")
    assert spool.append_record(ordinary_record(changed, 5)) is StoreDisposition.CONFLICT_TAINTED

    conflict = rows(
        spool,
        "SELECT logical_session_id, claimed_event_id, reason_code FROM evidence_conflicts",
    )
    assert conflict == [(SESSION_ID, event_uuid(11), "event_id_envelope_mismatch")]
    assert rows(spool, "SELECT state, taint_code FROM evidence_sessions") == [
        ("tainted", "event_id_conflict")
    ]
    assert rows(
        spool, "SELECT canonical_payload FROM evidence_events WHERE event_sequence=4"
    ) == original
    assert b"a completely different secret" not in (
        tmp_database_bytes := (spool.database).read_bytes()
    )
    assert tmp_database_bytes


def test_sequence_claim_and_gap_conflict_with_their_exact_taints(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    claim = make_spool(tmp_path / "claim")
    try:
        claim.create_epoch(make_create_epoch())
        claim.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
        duplicate = turn_opened_snapshot(3, event_index=12)
        assert claim.append_record(ordinary_record(duplicate, 4)) is (
            StoreDisposition.CONFLICT_TAINTED
        )
        assert rows(claim, "SELECT reason_code FROM evidence_conflicts") == [
            ("session_sequence_claimed",)
        ]
        assert rows(claim, "SELECT state, taint_code FROM evidence_sessions") == [
            ("tainted", "sequence_conflict")
        ]
    finally:
        claim.close()

    gap = make_spool(tmp_path / "gap")
    try:
        gap.create_epoch(make_create_epoch())
        assert gap.append_record(ordinary_record(turn_opened_snapshot(5, event_index=13), 3)) is (
            StoreDisposition.CONFLICT_TAINTED
        )
        assert rows(gap, "SELECT state, taint_code FROM evidence_sessions") == [
            ("tainted", "admission_gap")
        ]
        assert rows(gap, "SELECT COUNT(*) FROM evidence_events")[0][0] == 2
    finally:
        gap.close()


def test_seal_then_reuse_quarantines_the_sealed_session_without_incoming_text(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingClosedPayloadV1,
        BindingCloseReason,
        BindingCloseV1,
        EventKind,
        SealEpochV1,
        SessionSealRequestedPayloadV1,
        StoreDisposition,
    )

    spool.create_epoch(make_create_epoch())
    close = make_snapshot(
        EventKind.BINDING_CLOSED,
        3,
        BindingClosedPayloadV1(
            binding_id=BINDING_ID,
            close_reason=BindingCloseReason.HOST_SHUTDOWN,
        ),
        event_id=event_uuid(20),
    )
    spool.append_binding_close(
        BindingCloseV1(
            protocol_version=1,
            binding_id=BINDING_ID,
            consent_epoch_id=EPOCH_ID,
            logical_session_id=SESSION_ID,
            admission_ordinal=3,
            snapshot=close,
        )
    )
    seal_snapshot = make_snapshot(
        EventKind.SESSION_SEAL_REQUESTED,
        4,
        SessionSealRequestedPayloadV1(
            final_event_sequence=4,
            consent_epoch_id=EPOCH_ID,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=DISCLOSURE,
        ),
        event_id=event_uuid(21),
    )
    seal = SealEpochV1(
        protocol_version=1,
        consent_epoch_id=EPOCH_ID,
        logical_session_id=SESSION_ID,
        binding_id=BINDING_ID,
        final_event_sequence=4,
        close_reason=BindingCloseReason.HOST_SHUTDOWN,
        close_epoch=True,
        admission_ordinal=4,
        snapshot=seal_snapshot,
    )
    assert spool.seal_epoch(seal) is StoreDisposition.COMMITTED
    assert spool.seal_epoch(seal) is StoreDisposition.IDEMPOTENT
    sealed = rows(
        spool,
        "SELECT state, final_event_sequence, head_hash FROM evidence_sessions",
    )[0]
    assert sealed[0] == "sealed"
    assert sealed[1] == 4
    assert rows(spool, "SELECT state FROM consent_epochs") == [("closed",)]

    intruder = user_final_snapshot(5, event_index=22, text="late smuggled text")
    assert spool.append_record(ordinary_record(intruder, 9)) is StoreDisposition.CONFLICT_TAINTED
    assert rows(spool, "SELECT reason_code FROM evidence_conflicts") == [("sealed_session_reuse",)]
    quarantined = rows(
        spool, "SELECT state, taint_code, final_event_sequence, head_hash FROM evidence_sessions"
    )[0]
    assert quarantined[0] == "tainted"
    assert quarantined[1] == "event_id_conflict"
    assert quarantined[2:] == sealed[1:]
    assert b"late smuggled text" not in spool.database.read_bytes()


def test_rollover_opens_a_successor_and_cross_session_reuse_taints_the_incoming(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingClosedPayloadV1,
        BindingCloseReason,
        EventKind,
        RolloverSessionV1,
        SessionSealRequestedPayloadV1,
        StoreDisposition,
    )

    spool.create_epoch(make_create_epoch())
    close = make_snapshot(
        EventKind.BINDING_CLOSED,
        3,
        BindingClosedPayloadV1(
            binding_id=BINDING_ID,
            close_reason=BindingCloseReason.CAPACITY_ROLLOVER,
        ),
        event_id=event_uuid(30),
    )
    seal = make_snapshot(
        EventKind.SESSION_SEAL_REQUESTED,
        4,
        SessionSealRequestedPayloadV1(
            final_event_sequence=4,
            consent_epoch_id=EPOCH_ID,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=DISCLOSURE,
        ),
        event_id=event_uuid(31),
    )
    opened, binding = opening_snapshots(
        session=SUCCESSOR_ID,
        predecessor=SESSION_ID,
        first=32,
    )
    command = RolloverSessionV1(
        protocol_version=1,
        consent_epoch_id=EPOCH_ID,
        binding_id=BINDING_ID,
        binding_generation=1,
        predecessor_logical_session_id=SESSION_ID,
        successor_logical_session_id=SUCCESSOR_ID,
        predecessor_final_event_sequence=4,
        successor_expires_at_utc="2026-08-09T00:00:00.000000Z",
        admission_ordinal=5,
        snapshots=(close, seal, opened, binding),
    )
    assert spool.rollover_session(command) is StoreDisposition.COMMITTED
    assert spool.rollover_session(command) is StoreDisposition.IDEMPOTENT
    states = dict(rows(spool, "SELECT logical_session_id, state FROM evidence_sessions"))
    assert states == {SESSION_ID: "sealed", SUCCESSOR_ID: "open"}
    assert rows(spool, "SELECT state FROM consent_epochs") == [("active",)]

    transplanted = make_snapshot(
        EventKind.TURN_OPENED,
        3,
        turn_opened_snapshot(3, event_index=10).payload,
        event_id=event_uuid(1),
        session=SUCCESSOR_ID,
    )
    assert spool.append_record(ordinary_record(transplanted, 9)) is (
        StoreDisposition.CONFLICT_TAINTED
    )
    assert rows(spool, "SELECT logical_session_id, reason_code FROM evidence_conflicts") == [
        (SUCCESSOR_ID, "cross_session_event_id")
    ]
    assert dict(rows(spool, "SELECT logical_session_id, state FROM evidence_sessions")) == {
        SESSION_ID: "sealed",
        SUCCESSOR_ID: "tainted",
    }


def test_unknown_or_noncurrent_session_faults_capture(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    spool.create_epoch(make_create_epoch())
    stranger = turn_opened_snapshot(3, event_index=40, session=SUCCESSOR_ID)
    assert spool.append_record(ordinary_record(stranger, 3)) is StoreDisposition.FAULTED
    assert spool.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
    assert spool.append_record(ordinary_record(turn_opened_snapshot(3, event_index=41), 4)) is (
        StoreDisposition.FAULTED
    )
    assert rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0] == 2


def test_denied_text_is_rejected_and_never_reaches_the_store(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import StoreDisposition

    spool.create_epoch(make_create_epoch())
    spool.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
    secret = user_final_snapshot(4, event_index=11, text="my key is sk-" "abcdefghijklmnopqrst")
    assert spool.append_record(ordinary_record(secret, 4)) is StoreDisposition.REJECTED_STATE
    assert rows(spool, "SELECT state, taint_code FROM evidence_sessions") == [
        ("tainted", "deny_filter")
    ]
    assert rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0] == 3
    assert b"sk-" b"abcdefghijklmnopqrst" not in spool.database.read_bytes()


def test_canonical_payload_size_guard_is_exact() -> None:
    from hermes_realtime.evidence import sqlite_spool

    assert sqlite_spool.MAX_CANONICAL_PAYLOAD_BYTES == 32 * 1024
    assert sqlite_spool.check_canonical_payload_size(b"x" * 32768) is None
    with pytest.raises(ValueError):
        sqlite_spool.check_canonical_payload_size(b"x" * 32769)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("event_count", 9216, id="session_event_cap"),
        pytest.param("canonical_bytes", 40 * 1024 * 1024, id="session_byte_cap"),
    ],
)
def test_session_quota_preflight_rejects_without_a_partial_row(
    spool,  # type: ignore[no-untyped-def]
    column: str,
    value: int,
) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    spool.create_epoch(make_create_epoch())
    spool.connection.execute(f"UPDATE evidence_sessions SET {column}=?", (value,))
    spool.connection.commit()
    snapshot = turn_opened_snapshot(3, event_index=10)
    assert spool.append_record(ordinary_record(snapshot, 3)) is StoreDisposition.REJECTED_STATE
    assert rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0] == 2
    assert rows(spool, "SELECT taint_code FROM evidence_sessions") == [("quota_exceeded",)]


def test_page_quota_preflight_rejects_when_the_main_database_is_full(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import StoreDisposition

    spool.create_epoch(make_create_epoch())
    current = spool.connection.execute("PRAGMA page_count").fetchone()[0]
    spool.connection.execute(f"PRAGMA max_page_count={current}")
    assert spool.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3)) is (
        StoreDisposition.REJECTED_STATE
    )
    assert rows(spool, "SELECT taint_code FROM evidence_sessions") == [("quota_exceeded",)]


def test_global_live_quota_spans_every_session(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import sqlite_spool

    assert sqlite_spool.MAX_GLOBAL_LIVE_CANONICAL_BYTES == 80 * 1024 * 1024
    spool.create_epoch(make_create_epoch())
    assert spool.live_canonical_bytes() == (
        rows(spool, "SELECT canonical_bytes FROM evidence_sessions")[0][0]
    )


def test_sqlite_fault_rolls_back_and_latches_a_content_free_sticky_fault(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        CaptureState,
        OwnerState,
        StoreDisposition,
        WriterFault,
    )

    class FailingConnection(sqlite3.Connection):
        fail_on: str | None = None

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
            if FailingConnection.fail_on is not None and FailingConnection.fail_on in sql:
                raise sqlite3.OperationalError("injected disk I/O error")
            return super().execute(sql, *args)

    owned = make_spool(tmp_path, connection_factory=FailingConnection)
    try:
        owned.create_epoch(make_create_epoch())
        FailingConnection.fail_on = "INSERT INTO evidence_events"
        record = ordinary_record(turn_opened_snapshot(3, event_index=10), 3)
        assert owned.append_record(record) is StoreDisposition.FAULTED
        FailingConnection.fail_on = None

        assert rows(owned, "SELECT COUNT(*) FROM evidence_events")[0][0] == 2
        assert rows(owned, "SELECT event_count FROM evidence_sessions")[0][0] == 2
        assert owned.connection.in_transaction is False

        diagnostics = owned.diagnostics()
        assert diagnostics.sticky_fault is WriterFault.SQLITE_FAULT
        assert diagnostics.owner_state is OwnerState.FAULTED
        assert diagnostics.capture_state is CaptureState.FAULTED
        assert owned.append_record(record) is StoreDisposition.FAULTED
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
    finally:
        FailingConnection.fail_on = None
        owned.close()


def make_rollover_command():  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingClosedPayloadV1,
        BindingCloseReason,
        EventKind,
        RolloverSessionV1,
        SessionSealRequestedPayloadV1,
    )

    close = make_snapshot(
        EventKind.BINDING_CLOSED,
        3,
        BindingClosedPayloadV1(
            binding_id=BINDING_ID,
            close_reason=BindingCloseReason.CAPACITY_ROLLOVER,
        ),
        event_id=event_uuid(30),
    )
    seal = make_snapshot(
        EventKind.SESSION_SEAL_REQUESTED,
        4,
        SessionSealRequestedPayloadV1(
            final_event_sequence=4,
            consent_epoch_id=EPOCH_ID,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=DISCLOSURE,
        ),
        event_id=event_uuid(31),
    )
    opened, binding = opening_snapshots(session=SUCCESSOR_ID, predecessor=SESSION_ID, first=32)
    return RolloverSessionV1(
        protocol_version=1,
        consent_epoch_id=EPOCH_ID,
        binding_id=BINDING_ID,
        binding_generation=1,
        predecessor_logical_session_id=SESSION_ID,
        successor_logical_session_id=SUCCESSOR_ID,
        predecessor_final_event_sequence=4,
        successor_expires_at_utc="2026-08-09T00:00:00.000000Z",
        admission_ordinal=5,
        snapshots=(close, seal, opened, binding),
    )


def make_seal_command():  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingCloseReason,
        EventKind,
        SealEpochV1,
        SessionSealRequestedPayloadV1,
    )

    snapshot = make_snapshot(
        EventKind.SESSION_SEAL_REQUESTED,
        4,
        SessionSealRequestedPayloadV1(
            final_event_sequence=4,
            consent_epoch_id=EPOCH_ID,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=DISCLOSURE,
        ),
        event_id=event_uuid(21),
    )
    return SealEpochV1(
        protocol_version=1,
        consent_epoch_id=EPOCH_ID,
        logical_session_id=SESSION_ID,
        binding_id=BINDING_ID,
        final_event_sequence=4,
        close_reason=BindingCloseReason.HOST_SHUTDOWN,
        close_epoch=True,
        admission_ordinal=4,
        snapshot=snapshot,
    )


def make_binding_close_command():  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingClosedPayloadV1,
        BindingCloseReason,
        BindingCloseV1,
        EventKind,
    )

    snapshot = make_snapshot(
        EventKind.BINDING_CLOSED,
        3,
        BindingClosedPayloadV1(
            binding_id=BINDING_ID,
            close_reason=BindingCloseReason.CLIENT_CLOSED,
        ),
        event_id=event_uuid(20),
    )
    return BindingCloseV1(
        protocol_version=1,
        binding_id=BINDING_ID,
        consent_epoch_id=EPOCH_ID,
        logical_session_id=SESSION_ID,
        admission_ordinal=3,
        snapshot=snapshot,
    )


def build_create_epoch(**overrides: object):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingOpenedPayloadV1,
        CreateEpochV1,
        EventKind,
        SessionOpenedPayloadV1,
    )

    values: dict[str, object] = {
        "installation_id": INSTALL_ID,
        "producer_instance_id": PRODUCER_ID,
        "consent_epoch_id": EPOCH_ID,
        "logical_session_id": SESSION_ID,
        "binding_id": BINDING_ID,
        "binding_generation": 1,
        "disclosure_digest": DISCLOSURE,
        "retention_hours": 24,
        "microphone_accepted": False,
        "typed_accepted": True,
        "control_sequence": 1,
        "control_fingerprint_hash": HEX64,
        "opened_event_id": event_uuid(1),
        "binding_event_id": event_uuid(2),
    }
    values.update(overrides)

    def snapshot(kind, sequence: int, payload, event_id: str):  # type: ignore[no-untyped-def]
        return make_snapshot(
            kind,
            sequence,
            payload,
            event_id=event_id,
            session=cast(str, values["logical_session_id"]),
            installation=cast(str, values["installation_id"]),
            producer=cast(str, values["producer_instance_id"]),
        )

    opened = snapshot(
        EventKind.SESSION_OPENED,
        1,
        SessionOpenedPayloadV1(
            consent_epoch_id=cast(str, values["consent_epoch_id"]),
            binding_id=cast(str, values["binding_id"]),
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=cast(str, values["disclosure_digest"]),
            retention_hours=cast(int, values["retention_hours"]),
            microphone_accepted=cast(bool, values["microphone_accepted"]),
            typed_accepted=cast(bool, values["typed_accepted"]),
            predecessor_session_id=None,
        ),
        cast(str, values["opened_event_id"]),
    )
    binding = snapshot(
        EventKind.BINDING_OPENED,
        2,
        BindingOpenedPayloadV1(
            binding_id=cast(str, values["binding_id"]),
            binding_generation=cast(int, values["binding_generation"]),
            microphone_available=cast(bool, values["microphone_accepted"]),
            typed_available=cast(bool, values["typed_accepted"]),
        ),
        cast(str, values["binding_event_id"]),
    )
    return CreateEpochV1(
        protocol_version=1,
        installation_id=cast(str, values["installation_id"]),
        producer_instance_id=cast(str, values["producer_instance_id"]),
        consent_epoch_id=cast(str, values["consent_epoch_id"]),
        logical_session_id=cast(str, values["logical_session_id"]),
        binding_id=cast(str, values["binding_id"]),
        binding_generation=cast(int, values["binding_generation"]),
        consent_version="realtime-evidence-consent-v1",
        disclosure_digest=cast(str, values["disclosure_digest"]),
        retention_hours=cast(int, values["retention_hours"]),
        microphone_accepted=cast(bool, values["microphone_accepted"]),
        typed_accepted=cast(bool, values["typed_accepted"]),
        control_sequence=cast(int, values["control_sequence"]),
        control_fingerprint_hash=cast(str, values["control_fingerprint_hash"]),
        session_opened=opened,
        binding_opened=binding,
    )


def build_seal_command(**overrides: object):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingCloseReason,
        EventKind,
        SealEpochV1,
        SessionSealRequestedPayloadV1,
    )

    values: dict[str, object] = {
        "consent_epoch_id": EPOCH_ID,
        "logical_session_id": SESSION_ID,
        "binding_id": BINDING_ID,
        "final_event_sequence": 4,
        "close_reason": BindingCloseReason.HOST_SHUTDOWN,
        "admission_ordinal": 4,
        "disclosure_digest": DISCLOSURE,
        "event_id": event_uuid(21),
    }
    values.update(overrides)
    sequence = cast(int, values["final_event_sequence"])
    snapshot = make_snapshot(
        EventKind.SESSION_SEAL_REQUESTED,
        sequence,
        SessionSealRequestedPayloadV1(
            final_event_sequence=sequence,
            consent_epoch_id=cast(str, values["consent_epoch_id"]),
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=cast(str, values["disclosure_digest"]),
        ),
        event_id=cast(str, values["event_id"]),
        session=cast(str, values["logical_session_id"]),
    )
    return SealEpochV1(
        protocol_version=1,
        consent_epoch_id=cast(str, values["consent_epoch_id"]),
        logical_session_id=cast(str, values["logical_session_id"]),
        binding_id=cast(str, values["binding_id"]),
        final_event_sequence=sequence,
        close_reason=cast(BindingCloseReason, values["close_reason"]),
        close_epoch=True,
        admission_ordinal=cast(int, values["admission_ordinal"]),
        snapshot=snapshot,
    )


def build_rollover_command(**overrides: object):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        BindingClosedPayloadV1,
        BindingCloseReason,
        BindingOpenedPayloadV1,
        EventKind,
        RolloverSessionV1,
        SessionOpenedPayloadV1,
        SessionSealRequestedPayloadV1,
    )

    values: dict[str, object] = {
        "consent_epoch_id": EPOCH_ID,
        "binding_id": BINDING_ID,
        "binding_generation": 1,
        "predecessor_logical_session_id": SESSION_ID,
        "successor_logical_session_id": SUCCESSOR_ID,
        "predecessor_final_event_sequence": 4,
        "successor_expires_at_utc": "2026-08-09T00:00:00.000000Z",
        "admission_ordinal": 5,
        "close_reason": BindingCloseReason.CAPACITY_ROLLOVER,
        "disclosure_digest": DISCLOSURE,
        "close_event_id": event_uuid(30),
        "seal_event_id": event_uuid(31),
        "opened_event_id": event_uuid(32),
        "binding_event_id": event_uuid(33),
    }
    values.update(overrides)
    final = cast(int, values["predecessor_final_event_sequence"])
    predecessor = cast(str, values["predecessor_logical_session_id"])
    successor = cast(str, values["successor_logical_session_id"])
    epoch = cast(str, values["consent_epoch_id"])
    binding_id = cast(str, values["binding_id"])
    close = make_snapshot(
        EventKind.BINDING_CLOSED,
        final - 1,
        BindingClosedPayloadV1(
            binding_id=binding_id,
            close_reason=cast(BindingCloseReason, values["close_reason"]),
        ),
        event_id=cast(str, values["close_event_id"]),
        session=predecessor,
    )
    seal = make_snapshot(
        EventKind.SESSION_SEAL_REQUESTED,
        final,
        SessionSealRequestedPayloadV1(
            final_event_sequence=final,
            consent_epoch_id=epoch,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=cast(str, values["disclosure_digest"]),
        ),
        event_id=cast(str, values["seal_event_id"]),
        session=predecessor,
    )
    opened = make_snapshot(
        EventKind.SESSION_OPENED,
        1,
        SessionOpenedPayloadV1(
            consent_epoch_id=epoch,
            binding_id=binding_id,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=cast(str, values["disclosure_digest"]),
            retention_hours=24,
            microphone_accepted=False,
            typed_accepted=True,
            predecessor_session_id=predecessor,
        ),
        event_id=cast(str, values["opened_event_id"]),
        session=successor,
    )
    binding = make_snapshot(
        EventKind.BINDING_OPENED,
        2,
        BindingOpenedPayloadV1(
            binding_id=binding_id,
            binding_generation=cast(int, values["binding_generation"]),
            microphone_available=False,
            typed_available=True,
        ),
        event_id=cast(str, values["binding_event_id"]),
        session=successor,
    )
    return RolloverSessionV1(
        protocol_version=1,
        consent_epoch_id=epoch,
        binding_id=binding_id,
        binding_generation=cast(int, values["binding_generation"]),
        predecessor_logical_session_id=predecessor,
        successor_logical_session_id=successor,
        predecessor_final_event_sequence=final,
        successor_expires_at_utc=cast(str, values["successor_expires_at_utc"]),
        admission_ordinal=cast(int, values["admission_ordinal"]),
        snapshots=(close, seal, opened, binding),
    )


def test_command_builders_reproduce_the_canonical_commands() -> None:
    assert build_create_epoch() == make_create_epoch()
    assert build_seal_command() == make_seal_command()
    assert build_rollover_command() == make_rollover_command()


ALTERED_CREATE_EPOCH_FIELDS = (
    ("installation_id", {"installation_id": "70000000-0000-4000-8000-000000000001"}),
    ("producer_instance_id", {"producer_instance_id": "70000000-0000-4000-8000-000000000002"}),
    ("consent_epoch_id", {"consent_epoch_id": "70000000-0000-4000-8000-000000000003"}),
    ("logical_session_id", {"logical_session_id": "70000000-0000-4000-8000-000000000004"}),
    ("binding_id", {"binding_id": "70000000-0000-4000-8000-000000000005"}),
    ("binding_generation", {"binding_generation": 2}),
    ("disclosure_digest", {"disclosure_digest": "ef" * 32}),
    ("retention_hours", {"retention_hours": 48}),
    ("microphone_accepted", {"microphone_accepted": True}),
    ("typed_accepted", {"typed_accepted": False, "microphone_accepted": True}),
    ("control_sequence", {"control_sequence": 2}),
    ("control_fingerprint_hash", {"control_fingerprint_hash": "ba" * 32}),
    ("session_opened", {"opened_event_id": event_uuid(101)}),
    ("binding_opened", {"binding_event_id": event_uuid(102)}),
)


@pytest.mark.parametrize(
    ("field", "override"),
    [pytest.param(field, override, id=field) for field, override in ALTERED_CREATE_EPOCH_FIELDS],
)
def test_create_epoch_rejects_every_altered_command_field(
    spool,  # type: ignore[no-untyped-def]
    field: str,
    override: dict[str, object],
) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    assert spool.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    before = rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0]
    assert spool.create_epoch(build_create_epoch(**override)) is StoreDisposition.REJECTED_STATE
    assert rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0] == before
    assert rows(spool, "SELECT COUNT(*) FROM consent_epochs")[0][0] == 1
    assert spool.diagnostics().sticky_fault is None
    assert spool.create_epoch(make_create_epoch()) is StoreDisposition.IDEMPOTENT


ALTERED_SEAL_FIELDS = (
    ("consent_epoch_id", {"consent_epoch_id": "70000000-0000-4000-8000-000000000010"}),
    ("logical_session_id", {"logical_session_id": "70000000-0000-4000-8000-000000000011"}),
    ("binding_id", {"binding_id": "70000000-0000-4000-8000-000000000012"}),
    ("final_event_sequence", {"final_event_sequence": 5}),
    ("close_reason", {"close_reason": "client_closed"}),
    ("admission_ordinal", {"admission_ordinal": 9}),
    ("disclosure_digest", {"disclosure_digest": "ef" * 32}),
    ("snapshot", {"event_id": event_uuid(110)}),
)


@pytest.mark.parametrize(
    ("field", "override"),
    [pytest.param(field, override, id=field) for field, override in ALTERED_SEAL_FIELDS],
)
def test_seal_epoch_rejects_every_altered_command_field(
    spool,  # type: ignore[no-untyped-def]
    field: str,
    override: dict[str, object],
) -> None:
    from hermes_realtime.evidence.models import BindingCloseReason, StoreDisposition

    spool.create_epoch(make_create_epoch())
    spool.append_binding_close(make_binding_close_command())
    assert spool.seal_epoch(make_seal_command()) is StoreDisposition.COMMITTED
    before = rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0]

    if field == "close_reason":
        override = {"close_reason": BindingCloseReason.CLIENT_CLOSED}
    assert spool.seal_epoch(build_seal_command(**override)) is StoreDisposition.REJECTED_STATE
    assert rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0] == before
    assert rows(spool, "SELECT state, final_event_sequence FROM evidence_sessions") == [
        ("sealed", 4)
    ]
    assert spool.diagnostics().sticky_fault is None
    assert spool.seal_epoch(make_seal_command()) is StoreDisposition.IDEMPOTENT


ALTERED_ROLLOVER_FIELDS = (
    ("consent_epoch_id", {"consent_epoch_id": "70000000-0000-4000-8000-000000000020"}),
    ("binding_id", {"binding_id": "70000000-0000-4000-8000-000000000021"}),
    ("binding_generation", {"binding_generation": 3}),
    (
        "predecessor_logical_session_id",
        {"predecessor_logical_session_id": "70000000-0000-4000-8000-000000000022"},
    ),
    (
        "successor_logical_session_id",
        {"successor_logical_session_id": "70000000-0000-4000-8000-000000000023"},
    ),
    ("predecessor_final_event_sequence", {"predecessor_final_event_sequence": 5}),
    ("successor_expires_at_utc", {"successor_expires_at_utc": "2026-08-10T00:00:00.000000Z"}),
    ("admission_ordinal", {"admission_ordinal": 9}),
    ("disclosure_digest", {"disclosure_digest": "ef" * 32}),
    ("snapshots_close", {"close_event_id": event_uuid(120)}),
    ("snapshots_seal", {"seal_event_id": event_uuid(121)}),
    ("snapshots_opened", {"opened_event_id": event_uuid(122)}),
    ("snapshots_binding", {"binding_event_id": event_uuid(123)}),
)


@pytest.mark.parametrize(
    ("field", "override"),
    [pytest.param(field, override, id=field) for field, override in ALTERED_ROLLOVER_FIELDS],
)
def test_rollover_rejects_every_altered_command_field(
    spool,  # type: ignore[no-untyped-def]
    field: str,
    override: dict[str, object],
) -> None:
    from hermes_realtime.evidence.models import BindingCloseReason, StoreDisposition

    spool.create_epoch(make_create_epoch())
    assert spool.rollover_session(make_rollover_command()) is StoreDisposition.COMMITTED
    before = rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0]
    sessions = rows(spool, "SELECT logical_session_id, state FROM evidence_sessions")

    if field == "close_reason":  # pragma: no cover - retained for symmetry
        override = {"close_reason": BindingCloseReason.RETENTION_ROLLOVER}
    assert spool.rollover_session(build_rollover_command(**override)) is (
        StoreDisposition.REJECTED_STATE
    )
    assert rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0] == before
    assert rows(spool, "SELECT logical_session_id, state FROM evidence_sessions") == sessions
    assert spool.diagnostics().sticky_fault is None
    assert spool.rollover_session(make_rollover_command()) is StoreDisposition.IDEMPOTENT


def test_exact_repeats_allocate_no_timestamp_or_row(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        owned.create_epoch(make_create_epoch())
        owned.append_binding_close(make_binding_close_command())
        owned.seal_epoch(make_seal_command())
        settled = clock.calls
        counts = rows(owned, "SELECT COUNT(*) FROM evidence_events")[0][0]

        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.IDEMPOTENT
        assert owned.seal_epoch(make_seal_command()) is StoreDisposition.IDEMPOTENT
        assert clock.calls == settled
        assert rows(owned, "SELECT COUNT(*) FROM evidence_events")[0][0] == counts
        assert owned.diagnostics().sticky_fault is None
    finally:
        owned.close()


def test_retries_after_recovery_fail_closed_instead_of_claiming_idempotency(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path)
    try:
        first.create_epoch(make_create_epoch())
    finally:
        first.close()

    reopened = make_spool(tmp_path)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        # The accepted control identity is process-local and is never persisted, so a
        # post-restart repeat cannot be proven exact and must not claim idempotency.
        assert reopened.create_epoch(make_create_epoch()) is StoreDisposition.REJECTED_STATE
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_events")[0][0] == 0
        assert rows(
            reopened,
            "SELECT scope_kind, scope_key, reason_code FROM erasure_tombstones",
        ) == [("consent_epoch", EPOCH_ID, "unclean_epoch")]
        assert reopened.diagnostics().sticky_fault is None
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("method", "failing_select"),
    [
        pytest.param("append_record", "FROM evidence_events WHERE event_id=", id="append_record"),
        pytest.param(
            "append_binding_close",
            "FROM evidence_events WHERE event_id=",
            id="append_binding_close",
        ),
        pytest.param(
            "seal_epoch",
            "SELECT state, final_event_sequence",
            id="seal_epoch",
        ),
        pytest.param(
            "rollover_session",
            "SELECT state FROM evidence_sessions",
            id="rollover_session",
        ),
    ],
)
def test_read_path_sqlite_faults_never_escape(
    tmp_path: Path,
    method: str,
    failing_select: str,
) -> None:
    from hermes_realtime.evidence.models import (
        CaptureState,
        OwnerState,
        StoreDisposition,
        WriterFault,
    )

    class FailingConnection(sqlite3.Connection):
        fail_on: str | None = None

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
            if FailingConnection.fail_on is not None and FailingConnection.fail_on in sql:
                raise sqlite3.OperationalError("injected disk I/O error")
            return super().execute(sql, *args)

    commands = {
        "append_record": lambda: ordinary_record(turn_opened_snapshot(3, event_index=10), 3),
        "append_binding_close": make_binding_close_command,
        "seal_epoch": make_seal_command,
        "rollover_session": make_rollover_command,
    }
    owned = make_spool(tmp_path, connection_factory=FailingConnection)
    try:
        owned.create_epoch(make_create_epoch())
        command = commands[method]()
        FailingConnection.fail_on = failing_select
        result = getattr(owned, method)(command)
        FailingConnection.fail_on = None

        assert result is StoreDisposition.FAULTED
        assert owned.connection.in_transaction is False
        assert rows(owned, "SELECT COUNT(*) FROM evidence_events")[0][0] == 2

        diagnostics = owned.diagnostics()
        assert diagnostics.sticky_fault is WriterFault.SQLITE_FAULT
        assert diagnostics.owner_state is OwnerState.FAULTED
        assert diagnostics.capture_state is CaptureState.FAULTED
        assert getattr(owned, method)(command) is StoreDisposition.FAULTED
    finally:
        FailingConnection.fail_on = None
        owned.close()


def test_rollover_storage_errors_never_escape_its_planning_boundary(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    class RefusingSentinelHandle(storage_security.SentinelHandleV1):
        refuse = False

        def write_slot(self, offset: int, slot: bytes) -> None:
            if RefusingSentinelHandle.refuse:
                raise storage_security.EvidenceStorageError(
                    storage_security.WriterFault.STORE_CORRUPT,
                    "the sentinel lease is not held",
                )
            super().write_slot(offset, slot)

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock, sentinel_opener=RefusingSentinelHandle)
    try:
        owned.create_epoch(make_create_epoch())
        RefusingSentinelHandle.refuse = True
        clock.override.append(START - timedelta(hours=1))
        assert owned.rollover_session(make_rollover_command()) is StoreDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions")[0][0] == 1
    finally:
        RefusingSentinelHandle.refuse = False
        owned.close()


def test_diagnostics_is_exact_and_content_free(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        CaptureState,
        EvidenceDiagnosticsV1,
        OwnerState,
    )

    idle = spool.diagnostics()
    assert type(idle) is EvidenceDiagnosticsV1
    assert idle.owner_state is OwnerState.ABSENT
    assert idle.capture_state is CaptureState.UNAVAILABLE

    spool.create_epoch(make_create_epoch())
    active = spool.diagnostics()
    assert dataclasses.asdict(active) == {
        "protocol_version": 1,
        "owner_state": OwnerState.RUNNING,
        "capture_state": CaptureState.ACTIVE,
        "sticky_fault": None,
        "queue_record_count": 0,
        "queue_canonical_bytes": 0,
        "active_lease_count": 0,
        "pending_revoke": False,
        "purge_required": False,
    }
    rendered = repr(active)
    for secret in (str(spool.database), SESSION_ID, EPOCH_ID, INSTALL_ID, "hello there"):
        assert secret not in rendered


def test_clock_rollback_latches_the_sentinel_before_any_sqlite_work(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        owned.create_epoch(make_create_epoch())
        clock.override.append(START - timedelta(hours=1))
        record = ordinary_record(turn_opened_snapshot(3, event_index=10), 3)
        assert owned.append_record(record) is StoreDisposition.FAULTED

        assert owned.diagnostics().sticky_fault is WriterFault.PURGE_REQUIRED
        assert owned.diagnostics().purge_required is True
        assert rows(
            owned,
            "SELECT purge_required, purge_reason, purge_scope FROM producer_installation",
        ) == [(1, "clock_rollback", "store")]
        assert rows(owned, "SELECT COUNT(*) FROM evidence_events")[0][0] == 2
    finally:
        owned.close()
    sentinel = storage_security.decode_sentinel_image(
        (tmp_path / "evidence" / "capture-v1.owner").read_bytes()
    )
    assert sentinel.active.state is storage_security.SentinelState.CLOCK_ROLLBACK_PURGE_PENDING


def test_clock_rollback_latches_the_sentinel_even_when_sqlite_cannot_record_it(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import StoreDisposition

    class BrokenConnection(sqlite3.Connection):
        broken = False

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
            if BrokenConnection.broken and "purge_required" in sql:
                raise sqlite3.OperationalError("injected disk I/O error")
            return super().execute(sql, *args)

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock, connection_factory=BrokenConnection)
    try:
        owned.create_epoch(make_create_epoch())
        BrokenConnection.broken = True
        clock.override.append(START - timedelta(hours=1))
        assert (
            owned.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
            is StoreDisposition.FAULTED
        )
        assert rows(owned, "SELECT COUNT(*) FROM evidence_events")[0][0] == 2
    finally:
        BrokenConnection.broken = False
        owned.close()
    # The sentinel latch is durable even though SQLite never recorded the rollback.
    sentinel = storage_security.decode_sentinel_image(
        (tmp_path / "evidence" / "capture-v1.owner").read_bytes()
    )
    assert sentinel.active.state is storage_security.SentinelState.CLOCK_ROLLBACK_PURGE_PENDING


def test_recover_existing_reports_absent_recovered_and_fail_closed_states(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    empty = make_spool(tmp_path / "empty")
    try:
        assert empty.recover_existing() is RecoveryDisposition.ABSENT
    finally:
        empty.close()

    created = make_spool(tmp_path / "live")
    try:
        created.create_epoch(make_create_epoch())
    finally:
        created.close()
    reopened = make_spool(tmp_path / "live")
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_events")[0][0] == 0
        assert rows(
            reopened,
            "SELECT scope_kind, scope_key, reason_code FROM erasure_tombstones",
        ) == [("consent_epoch", EPOCH_ID, "unclean_epoch")]
    finally:
        reopened.close()

    root = tmp_path / "live" / "evidence"
    sentinel_path = root / "capture-v1.owner"
    sentinel_path.write_bytes(
        storage_security.next_sentinel_image(
            sentinel_path.read_bytes(),
            storage_security.SentinelState.FULL_PURGE_PENDING,
            state_generation_id=STATE_ID,
        )
    )
    pending = make_spool(tmp_path / "live")
    try:
        assert pending.recover_existing() is RecoveryDisposition.PURGE_COMPLETED
        assert pending.diagnostics().sticky_fault is None
    finally:
        pending.close()

    unsupported = make_spool(tmp_path / "linux")
    unsupported.probe.supported = False
    try:
        assert unsupported.recover_existing() is RecoveryDisposition.UNSUPPORTED_PLATFORM
    finally:
        unsupported.close()


def test_ownership_is_exclusive_between_two_spools(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        contender = make_spool(tmp_path)
        try:
            assert contender.recover_existing() is RecoveryDisposition.OWNERSHIP_UNAVAILABLE
            assert contender.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
        finally:
            contender.close()
    finally:
        first.close()


def test_drain_and_close_stops_the_owner_exactly_once(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import DrainAndStopV1, DrainDisposition, OwnerState

    spool.create_epoch(make_create_epoch())
    command = DrainAndStopV1(protocol_version=1, owner_generation=1, final_admission_ordinal=2)
    foreign = DrainAndStopV1(protocol_version=1, owner_generation=2, final_admission_ordinal=2)
    assert spool.drain_and_close(foreign) is DrainDisposition.ALREADY_QUEUED
    assert spool.drain_and_close(command) is DrainDisposition.STOPPED
    assert spool.drain_and_close(command) is DrainDisposition.STOPPED
    assert spool.diagnostics().owner_state is OwnerState.STOPPED


def test_writer_daemon_observes_exact_persisted_session_expiry_on_owner_thread(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import StoreDisposition

    database = tmp_path / "capture-v1.sqlite3"
    daemon = sqlite_spool.SQLiteEvidenceWriterDaemonV1(
        lambda: sqlite_spool.SQLiteEvidenceSpool(
            database,
            clock=StubClock(),
            uuid_factory=StubUuids(),
        )
    )
    command = make_create_epoch()
    assert daemon.create_epoch(command) is StoreDisposition.COMMITTED

    observed = daemon.active_session_expiry(command.logical_session_id)

    with sqlite3.connect(database) as connection:
        persisted = connection.execute(
            "SELECT expires_at_utc FROM evidence_sessions WHERE logical_session_id=?",
            (command.logical_session_id,),
        ).fetchone()
    assert persisted is not None
    assert observed == persisted[0]
    assert daemon.close() is True


def test_task_five_methods_are_narrow_and_never_raise(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        ErasureReason,
        ErasureScope,
        ExpireSessionV1,
        ExpiryMode,
        FullPurgeV1,
        MaintenanceV1,
        PurgeDisposition,
        RevokeDisposition,
        RevokeFinalizeV1,
        RevokeRequestV1,
        SentinelState,
    )

    spool.create_epoch(make_create_epoch())
    resolved = spool.resolved_manifest
    assert spool.expire_session(
        ExpireSessionV1(
            protocol_version=1,
            owner_generation=1,
            logical_session_id=SESSION_ID,
            consent_epoch_id=EPOCH_ID,
            expires_at_utc="2026-08-10T00:00:00.000000Z",
            last_admission_ordinal=2,
            erasure_request_id=event_uuid(90),
            mode=ExpiryMode.ERASE_STUCK,
        )
    ) is PurgeDisposition.PURGE_FAILED
    assert spool.commit_revoke_request(
        RevokeRequestV1(
            protocol_version=1,
            erasure_request_id=event_uuid(91),
            consent_epoch_id=EPOCH_ID,
            control_sequence=1,
            control_fingerprint_hash=HEX64,
            last_admission_ordinal=2,
        )
    ) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
    assert spool.finalize_revoke(
        RevokeFinalizeV1(
            protocol_version=1,
            erasure_request_id=event_uuid(91),
            consent_epoch_id=EPOCH_ID,
            control_sequence=1,
            control_fingerprint_hash=HEX64,
            final_admission_ordinal=2,
        )
    ) is RevokeDisposition.PURGE_COMPLETED
    assert spool.run_maintenance(
        MaintenanceV1(
            protocol_version=1,
            erasure_reason=ErasureReason.TTL,
            erasure_scope=ErasureScope.SESSION,
            erasure_request_id=event_uuid(92),
            scope_id=SESSION_ID,
            deadline_admission_ordinal=2,
            artifact_manifest_version=1,
        )
    ) is PurgeDisposition.PURGE_FAILED
    assert spool.purge_full_store(
        FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=event_uuid(93),
            sentinel_state=SentinelState.FULL_PURGE_PENDING,
            artifact_manifest_version=1,
        )
    ) is PurgeDisposition.PURGE_COMPLETED
    assert [path.name for path in resolved.deletable if path.exists()] == []


def test_task_5c_expire_session_persists_exact_ttl_authority_and_erases_only_target(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        ExpireSessionV1,
        ExpiryMode,
        PurgeDisposition,
        StoreDisposition,
    )

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        expires_at = rows(
            owned,
            "SELECT expires_at_utc FROM evidence_sessions WHERE logical_session_id=?",
            (SESSION_ID,),
        )[0][0]
        command = ExpireSessionV1(
            protocol_version=1,
            owner_generation=owned.owner_generation,
            logical_session_id=SESSION_ID,
            consent_epoch_id=EPOCH_ID,
            expires_at_utc=str(expires_at),
            last_admission_ordinal=2,
            erasure_request_id=event_uuid(990),
            mode=ExpiryMode.ERASE_STUCK,
        )

        clock.override.extend(
            (START + timedelta(hours=25), START + timedelta(hours=25, seconds=1))
        )
        assert owned.expire_session(command) is PurgeDisposition.PURGE_COMPLETED
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM evidence_events") == [(0,)]
        assert rows(
            owned,
            "SELECT scope_kind, scope_key, reason_code, erased_session_count,"
            " erased_event_count FROM erasure_tombstones",
        ) == [("session", SESSION_ID, "ttl", 1, 2)]
        assert owned.expire_session(command) is PurgeDisposition.ALREADY_ABSENT
    finally:
        owned.close()


def test_task_5c_maintenance_expires_only_elapsed_ttl_session(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import (
        ErasureReason,
        ErasureScope,
        MaintenanceV1,
        PurgeDisposition,
        StoreDisposition,
    )

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        # Maintenance samples once to schedule and again to tombstone; both must be
        # later than the persisted high-water mark.
        clock.override.extend((START + timedelta(hours=25), START + timedelta(hours=25, seconds=1)))
        command = MaintenanceV1(
            protocol_version=1,
            erasure_reason=ErasureReason.TTL,
            erasure_scope=ErasureScope.SESSION,
            erasure_request_id=event_uuid(991),
            scope_id=SESSION_ID,
            deadline_admission_ordinal=2,
            artifact_manifest_version=1,
        )

        assert owned.run_maintenance(command) is PurgeDisposition.PURGE_COMPLETED
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(
            owned,
            "SELECT scope_kind, scope_key, reason_code FROM erasure_tombstones",
        ) == [("session", SESSION_ID, "ttl")]
    finally:
        owned.close()


def test_task_5c_recovery_erases_an_elapsed_open_session_before_availability(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    clock = StubClock()
    first = make_spool(tmp_path, clock=clock)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()

    recovery_clock = StubClock(START + timedelta(hours=25))
    reopened = make_spool(tmp_path, clock=recovery_clock)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(
            reopened,
            "SELECT scope_kind, scope_key, reason_code FROM erasure_tombstones",
        ) == [("consent_epoch", EPOCH_ID, "unclean_epoch")]
    finally:
        reopened.close()


def test_task_5c_drain_refuses_a_watermark_before_the_last_durable_record(
    spool,
) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import (
        DrainAndStopV1,
        DrainDisposition,
        OwnerState,
        StoreDisposition,
    )

    assert spool.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    assert spool.append_record(ordinary_record(turn_opened_snapshot(3, event_index=992), 3)) is (
        StoreDisposition.COMMITTED
    )
    stale = DrainAndStopV1(protocol_version=1, owner_generation=1, final_admission_ordinal=2)
    exact = DrainAndStopV1(protocol_version=1, owner_generation=1, final_admission_ordinal=3)

    assert spool.drain_and_close(stale) is DrainDisposition.ALREADY_QUEUED
    assert spool.diagnostics().owner_state is OwnerState.RUNNING
    assert spool.drain_and_close(exact) is DrainDisposition.STOPPED


def test_task_5c_expiry_refuses_before_anchor_and_binds_terminal_replay_watermark(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        ExpireSessionV1,
        ExpiryMode,
        PurgeDisposition,
        StoreDisposition,
    )

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        expires_at = str(
            rows(
                owned,
                "SELECT expires_at_utc FROM evidence_sessions WHERE logical_session_id=?",
                (SESSION_ID,),
            )[0][0]
        )
        command = ExpireSessionV1(
            protocol_version=1,
            owner_generation=owned.owner_generation,
            logical_session_id=SESSION_ID,
            consent_epoch_id=EPOCH_ID,
            expires_at_utc=expires_at,
            last_admission_ordinal=2,
            erasure_request_id=event_uuid(993),
            mode=ExpiryMode.ERASE_STUCK,
        )

        assert owned.expire_session(command) is PurgeDisposition.PURGE_FAILED
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
        assert rows(owned, "SELECT COUNT(*) FROM erasure_requests") == [(0,)]

        clock.override.extend((START + timedelta(hours=25), START + timedelta(hours=25, seconds=1)))
        assert owned.expire_session(command) is PurgeDisposition.PURGE_COMPLETED
        altered = dataclasses.replace(command, last_admission_ordinal=3)
        assert owned.expire_session(altered) is PurgeDisposition.PURGE_FAILED
        assert (
            owned.expire_session(dataclasses.replace(command, consent_epoch_id=event_uuid(1003)))
            is PurgeDisposition.PURGE_FAILED
        )
        assert (
            owned.expire_session(
                dataclasses.replace(
                    command,
                    expires_at_utc="2025-01-02T01:00:00.000000Z",
                )
            )
            is PurgeDisposition.PURGE_FAILED
        )
    finally:
        owned.close()


def test_task_5c_maintenance_refuses_early_and_binds_terminal_replay_watermark(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        ErasureReason,
        ErasureScope,
        MaintenanceV1,
        PurgeDisposition,
        StoreDisposition,
    )

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        command = MaintenanceV1(
            protocol_version=1,
            erasure_reason=ErasureReason.TTL,
            erasure_scope=ErasureScope.SESSION,
            erasure_request_id=event_uuid(994),
            scope_id=SESSION_ID,
            deadline_admission_ordinal=2,
            artifact_manifest_version=1,
        )
        assert owned.run_maintenance(command) is PurgeDisposition.PURGE_FAILED
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
        assert rows(owned, "SELECT COUNT(*) FROM erasure_requests") == [(0,)]

        clock.override.extend((START + timedelta(hours=25), START + timedelta(hours=25, seconds=1)))
        assert owned.run_maintenance(command) is PurgeDisposition.PURGE_COMPLETED
        altered = dataclasses.replace(command, deadline_admission_ordinal=3)
        assert owned.run_maintenance(altered) is PurgeDisposition.PURGE_FAILED
    finally:
        owned.close()


def test_task_5c_recovery_privacy_purges_an_unclosed_epoch_before_ttl(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path, clock=StubClock())
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=StubClock(START + timedelta(hours=1)))
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(reopened, "SELECT COUNT(*) FROM consent_epochs") == [(0,)]
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(
            reopened,
            "SELECT scope_kind, scope_key, reason_code FROM erasure_tombstones",
        ) == [("consent_epoch", EPOCH_ID, "unclean_epoch")]
    finally:
        reopened.close()


def test_task_5c_recovery_detects_clock_rollback_and_finishes_full_store_purge(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path, clock=StubClock())
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            first.append_record(ordinary_record(turn_opened_snapshot(3, event_index=995), 3))
            is StoreDisposition.COMMITTED
        )
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=StubClock(START - timedelta(hours=1)))
    try:
        assert reopened.recover_existing() is RecoveryDisposition.PURGE_COMPLETED
        assert not (tmp_path / "evidence" / "capture-v1.sqlite3").exists()
        active = storage_security.decode_sentinel_image(
            (tmp_path / "evidence" / "capture-v1.owner").read_bytes()
        ).active
        assert active.state is storage_security.SentinelState.CLEAR
    finally:
        reopened.close()


def test_task_5c_drain_refuses_until_revoke_is_physically_finalized(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import (
        DrainAndStopV1,
        DrainDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
    )

    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        request = RevokeRequestV1(
            protocol_version=1,
            erasure_request_id=event_uuid(996),
            consent_epoch_id=EPOCH_ID,
            control_sequence=1,
            control_fingerprint_hash=HEX64,
            last_admission_ordinal=2,
        )
        assert (
            owned.commit_revoke_request(request)
            is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        )
        drain = DrainAndStopV1(
            protocol_version=1,
            owner_generation=owned.owner_generation,
            final_admission_ordinal=2,
        )
        assert owned.drain_and_close(drain) is DrainDisposition.ALREADY_QUEUED
        assert owned.connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        owned.close()


def test_task_5c_seal_rejects_conflict_or_elapsed_ttl_without_partial_close(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    clock = StubClock()
    conflicted = make_spool(tmp_path / "conflict", clock=clock)
    try:
        assert conflicted.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            conflicted.append_binding_close(make_binding_close_command())
            is StoreDisposition.COMMITTED
        )
        conflicted.connection.execute(
            "INSERT INTO evidence_conflicts (conflict_id, logical_session_id, claimed_event_id,"
            " reason_code, recorded_at_utc) VALUES (?,?,?,?,?)",
            (event_uuid(997), SESSION_ID, event_uuid(998), "session_sequence_claimed", UTC_TEXT),
        )
        assert conflicted.seal_epoch(make_seal_command()) is StoreDisposition.REJECTED_STATE
        assert rows(
            conflicted,
            "SELECT state, final_event_sequence FROM evidence_sessions",
        ) == [("open", None)]
        assert rows(conflicted, "SELECT state FROM consent_epochs") == [("active",)]
    finally:
        conflicted.close()

    elapsed_clock = StubClock()
    elapsed = make_spool(tmp_path / "elapsed", clock=elapsed_clock)
    try:
        assert elapsed.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            elapsed.append_binding_close(make_binding_close_command())
            is StoreDisposition.COMMITTED
        )
        elapsed_clock.override.append(START + timedelta(hours=25))
        assert elapsed.seal_epoch(make_seal_command()) is StoreDisposition.REJECTED_STATE
        assert rows(elapsed, "SELECT state FROM evidence_sessions") == [("open",)]
        assert rows(elapsed, "SELECT state FROM consent_epochs") == [("active",)]
    finally:
        elapsed.close()


def test_task_5c_physical_allocation_above_ceiling_fails_closed(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import WriterFault

    ceiling = storage_security.EVIDENCE_PHYSICAL_MAINTENANCE_CEILING_BYTES
    probe = InjectedStorageProbe(
        allocations={"capture-v1.sqlite3": ceiling + 1},
        free=ceiling,
    )
    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        storage_security.check_maintenance_headroom(tmp_path, probe=probe)
    assert caught.value.fault is WriterFault.QUOTA_UNAVAILABLE


def test_task_5c_seal_revalidates_the_stored_hash_chain_in_its_transaction(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            owned.append_binding_close(make_binding_close_command())
            is StoreDisposition.COMMITTED
        )
        trigger_sql = str(
            rows(
                owned,
                "SELECT sql FROM sqlite_master WHERE type='trigger'"
                " AND name='evidence_events_are_append_only'",
            )[0][0]
        )
        owned.connection.execute("DROP TRIGGER evidence_events_are_append_only")
        owned.connection.execute(
            "UPDATE evidence_events SET record_hash=? WHERE logical_session_id=?"
            " AND event_sequence=2",
            ("ab" * 32, SESSION_ID),
        )
        owned.connection.execute(trigger_sql)
        owned.connection.commit()

        assert owned.seal_epoch(make_seal_command()) is StoreDisposition.REJECTED_STATE
        assert rows(owned, "SELECT state, event_count FROM evidence_sessions") == [("open", 3)]
        assert owned.diagnostics().sticky_fault is None
    finally:
        owned.close()


def test_task_5c_already_open_recovery_privacy_purges_active_epoch(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(owned, "SELECT COUNT(*) FROM consent_epochs") == [(0,)]
        assert rows(
            owned,
            "SELECT scope_kind, scope_key, reason_code FROM erasure_tombstones",
        ) == [("consent_epoch", EPOCH_ID, "unclean_epoch")]
    finally:
        owned.close()


def test_task_5c_recovery_does_not_let_unrelated_revoke_hide_active_epoch(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
    )

    second_epoch = event_uuid(1001)
    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            first.commit_revoke_request(
                RevokeRequestV1(
                    protocol_version=1,
                    erasure_request_id=event_uuid(1002),
                    consent_epoch_id=EPOCH_ID,
                    control_sequence=1,
                    control_fingerprint_hash=HEX64,
                    last_admission_ordinal=2,
                )
            )
            is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        )
        first.connection.execute(
            "INSERT INTO consent_epochs (consent_epoch_id, producer_instance_id, state,"
            " opened_at_utc, closed_at_utc) VALUES (?, ?, 'active', ?, NULL)",
            (second_epoch, PRODUCER_ID, UTC_TEXT),
        )
        first.connection.commit()
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=StubClock(START + timedelta(seconds=10)))
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert rows(
            reopened,
            "SELECT consent_epoch_id FROM consent_epochs ORDER BY consent_epoch_id",
        ) == [(EPOCH_ID,)]
        assert rows(
            reopened,
            "SELECT scope_kind, scope_key, reason_code FROM erasure_tombstones",
        ) == [("consent_epoch", second_epoch, "unclean_epoch")]
    finally:
        reopened.close()


def test_task_5c_drain_requires_confirmed_sentinel_release(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import (
        DrainAndStopV1,
        DrainDisposition,
        StoreDisposition,
    )

    class CloseFailsOnce(storage_security.SentinelHandleV1):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.fail_close = True

        def close(self) -> None:
            if self.fail_close:
                self.fail_close = False
                raise OSError("injected close failure")
            super().close()

    owned = make_spool(tmp_path, sentinel_opener=CloseFailsOnce)
    drain = DrainAndStopV1(
        protocol_version=1,
        owner_generation=owned.owner_generation,
        final_admission_ordinal=2,
    )
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.drain_and_close(drain) is DrainDisposition.ALREADY_QUEUED
        assert owned._sentinel_handle is not None
        assert owned.drain_and_close(drain) is DrainDisposition.STOPPED
        assert owned._sentinel_handle is None
    finally:
        owned.close()


def test_task_5c_seal_rechecks_quota_inside_transaction(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import StoreDisposition

    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            owned.append_binding_close(make_binding_close_command())
            is StoreDisposition.COMMITTED
        )
        original = owned._check_quotas
        calls = 0

        def quota_changes_after_plan(**quota: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise sqlite_spool._Refused(StoreDisposition.REJECTED_STATE)
            original(**quota)

        monkeypatch.setattr(owned, "_check_quotas", quota_changes_after_plan)
        assert owned.seal_epoch(make_seal_command()) is StoreDisposition.REJECTED_STATE
        assert calls == 2
        assert rows(owned, "SELECT state, event_count FROM evidence_sessions") == [("open", 3)]
    finally:
        owned.close()


def test_task_5c_recovery_rejects_predeadline_pending_ttl_authority(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        expires_at = str(
            rows(
                first,
                "SELECT expires_at_utc FROM evidence_sessions WHERE logical_session_id=?",
                (SESSION_ID,),
            )[0][0]
        )
        first.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc,"
            " last_admission_ordinal, final_admission_ordinal, resume_state,"
            " erased_session_count, erased_event_count)"
            " VALUES (?, 'session', ?, '2025-01-01T00:00:01.000000Z', 'ttl', 'pending',"
            " NULL, ?, ?, ?, 2, NULL, NULL, NULL, NULL)",
            (event_uuid(1004), SESSION_ID, HEX64, EPOCH_ID, expires_at),
        )
        first.connection.commit()
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=StubClock(START + timedelta(seconds=2)))
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
    finally:
        reopened.close()


def test_task_5c_recovery_rejects_wrong_epoch_pending_ttl_authority(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        expires_at = str(
            rows(
                first,
                "SELECT expires_at_utc FROM evidence_sessions WHERE logical_session_id=?",
                (SESSION_ID,),
            )[0][0]
        )
        first.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc,"
            " last_admission_ordinal, final_admission_ordinal, resume_state,"
            " erased_session_count, erased_event_count)"
            " VALUES (?, 'session', ?, '2025-01-02T01:00:00.000000Z', 'ttl', 'pending',"
            " NULL, ?, ?, ?, 2, NULL, NULL, NULL, NULL)",
            (event_uuid(1005), SESSION_ID, HEX64, event_uuid(1006), expires_at),
        )
        first.connection.commit()
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=StubClock(START + timedelta(hours=25, seconds=1)))
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
    finally:
        reopened.close()


def test_task_5c_maintenance_persists_epoch_and_expiry_authority(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import (
        ErasureReason,
        ErasureScope,
        MaintenanceV1,
        PurgeDisposition,
        StoreDisposition,
    )

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        expires_at = str(
            rows(
                owned,
                "SELECT expires_at_utc FROM evidence_sessions WHERE logical_session_id=?",
                (SESSION_ID,),
            )[0][0]
        )
        clock.override.extend(
            (START + timedelta(hours=25), START + timedelta(hours=25, seconds=1))
        )
        assert (
            owned.run_maintenance(
                MaintenanceV1(
                    protocol_version=1,
                    erasure_scope=ErasureScope.SESSION,
                    scope_id=SESSION_ID,
                    erasure_reason=ErasureReason.TTL,
                    deadline_admission_ordinal=2,
                    erasure_request_id=event_uuid(1007),
                    artifact_manifest_version=1,
                )
            )
            is PurgeDisposition.PURGE_COMPLETED
        )
        assert rows(
            owned,
            "SELECT ttl_consent_epoch_id, ttl_expires_at_utc FROM erasure_tombstones",
        ) == [(EPOCH_ID, expires_at)]
    finally:
        owned.close()


@pytest.mark.parametrize("cold_start", [False, True])
def test_task_5c_recovery_rejects_preexpiry_terminal_ttl_authority(
    tmp_path: Path,
    cold_start: bool,
) -> None:
    from hermes_realtime.evidence.models import (
        ExpireSessionV1,
        ExpiryMode,
        PurgeDisposition,
        RecoveryDisposition,
        StoreDisposition,
        WriterFault,
    )
    from hermes_realtime.evidence.sqlite_spool import format_canonical_utc

    request_id = event_uuid(1008)
    expires_at = format_canonical_utc(START + timedelta(hours=24))
    erased_at = format_canonical_utc(START + timedelta(hours=23))
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        owned.connection.execute("PRAGMA ignore_check_constraints=ON")
        owned.connection.execute("DELETE FROM evidence_sessions")
        owned.connection.execute("DELETE FROM consent_epochs")
        owned.connection.execute(
            "INSERT INTO erasure_tombstones (erasure_request_id, scope_kind, scope_key,"
            " reason_code, control_sequence, control_fingerprint_hash,"
            " ttl_consent_epoch_id, ttl_expires_at_utc, last_admission_ordinal,"
            " final_admission_ordinal, erased_at_utc, erased_session_count,"
            " erased_event_count) VALUES (?, 'session', ?, 'ttl', NULL, ?, ?, ?, 2,"
            " NULL, ?, 1, 2)",
            (request_id, SESSION_ID, HEX64, EPOCH_ID, expires_at, erased_at),
        )
        owned.connection.execute("PRAGMA ignore_check_constraints=OFF")
        owned.connection.commit()
        if cold_start:
            owned.close()
            owned = make_spool(tmp_path)

        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(
            owned,
            "SELECT erased_at_utc, ttl_expires_at_utc FROM erasure_tombstones",
        ) == [(erased_at, expires_at)]
        assert rows(owned, "SELECT COUNT(*) FROM erasure_requests") == [(0,)]
        assert (
            owned.expire_session(
                ExpireSessionV1(
                    protocol_version=1,
                    owner_generation=1,
                    logical_session_id=SESSION_ID,
                    consent_epoch_id=EPOCH_ID,
                    expires_at_utc=expires_at,
                    last_admission_ordinal=2,
                    erasure_request_id=request_id,
                    mode=ExpiryMode.ERASE_STUCK,
                )
            )
            is PurgeDisposition.PURGE_FAILED
        )
    finally:
        owned.close()


def test_task_5c_pending_revoke_fails_closed_on_already_open_recovery(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
    )

    request_id = event_uuid(1009)
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            owned.commit_revoke_request(
                RevokeRequestV1(
                    protocol_version=1,
                    erasure_request_id=request_id,
                    consent_epoch_id=EPOCH_ID,
                    control_sequence=1,
                    control_fingerprint_hash=HEX64,
                    last_admission_ordinal=2,
                )
            )
            is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        )

        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
        assert rows(owned, "SELECT COUNT(*) FROM consent_epochs") == [(1,)]
        assert rows(owned, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
        assert rows(
            owned,
            "SELECT scope_kind, scope_key, reason_code, state FROM erasure_requests",
        ) == [("consent_epoch", EPOCH_ID, "revoked", "pending")]
    finally:
        owned.close()


@pytest.mark.parametrize("cold_start", [False, True])
def test_task_5c_recovery_revalidates_sealed_event_hash_chains(
    tmp_path: Path,
    cold_start: bool,
) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        StoreDisposition,
        WriterFault,
    )

    forged_hash = "ab" * 32
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            owned.append_binding_close(make_binding_close_command())
            is StoreDisposition.COMMITTED
        )
        assert owned.seal_epoch(make_seal_command()) is StoreDisposition.COMMITTED
        trigger_sql = str(
            rows(
                owned,
                "SELECT sql FROM sqlite_master WHERE type='trigger'"
                " AND name='evidence_events_are_append_only'",
            )[0][0]
        )
        owned.connection.execute("DROP TRIGGER evidence_events_are_append_only")
        owned.connection.execute(
            "UPDATE evidence_events SET record_hash=? WHERE logical_session_id=?"
            " AND event_sequence=2",
            (forged_hash, SESSION_ID),
        )
        owned.connection.execute(trigger_sql)
        owned.connection.commit()
        if cold_start:
            owned.close()
            owned = make_spool(tmp_path, clock=StubClock(START + timedelta(seconds=10)))

        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(
            owned,
            "SELECT record_hash FROM evidence_events WHERE logical_session_id=?"
            " AND event_sequence=2",
            (SESSION_ID,),
        ) == [(forged_hash,)]
        assert rows(owned, "SELECT state, event_count FROM evidence_sessions") == [
            ("sealed", 4)
        ]
    finally:
        owned.close()


@pytest.mark.parametrize("cold_start", [False, True])
@pytest.mark.parametrize("corruption", ["orphan_event", "orphan_conflict", "session_state"])
def test_task_5c_recovery_rejects_incomplete_durable_relations(
    tmp_path: Path,
    cold_start: bool,
    corruption: str,
) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        StoreDisposition,
        WriterFault,
    )

    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            owned.append_binding_close(make_binding_close_command())
            is StoreDisposition.COMMITTED
        )
        assert owned.seal_epoch(make_seal_command()) is StoreDisposition.COMMITTED
        if corruption == "orphan_event":
            owned.connection.commit()
            owned.connection.execute("PRAGMA foreign_keys=OFF")
            owned.connection.execute(
                "DELETE FROM evidence_sessions WHERE logical_session_id=?",
                (SESSION_ID,),
            )
            owned.connection.commit()
            owned.connection.execute("PRAGMA foreign_keys=ON")
        elif corruption == "orphan_conflict":
            owned.connection.commit()
            owned.connection.execute("PRAGMA foreign_keys=OFF")
            owned.connection.execute(
                "INSERT INTO evidence_conflicts (conflict_id, logical_session_id,"
                " claimed_event_id, reason_code, recorded_at_utc)"
                " VALUES (?, ?, ?, 'cross_session_event_id', ?)",
                (event_uuid(1010), event_uuid(1011), event_uuid(1012), UTC_TEXT),
            )
            owned.connection.commit()
            owned.connection.execute("PRAGMA foreign_keys=ON")
        else:
            owned.connection.execute("PRAGMA ignore_check_constraints=ON")
            owned.connection.execute(
                "UPDATE evidence_sessions SET state='unsupported'"
                " WHERE logical_session_id=?",
                (SESSION_ID,),
            )
            owned.connection.execute("PRAGMA ignore_check_constraints=OFF")
            owned.connection.commit()
        if cold_start:
            owned.close()
            owned = make_spool(tmp_path, clock=StubClock(START + timedelta(seconds=10)))

        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        if corruption == "orphan_event":
            assert rows(owned, "SELECT COUNT(*) FROM evidence_events") == [(4,)]
            assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        elif corruption == "orphan_conflict":
            assert rows(owned, "SELECT COUNT(*) FROM evidence_conflicts") == [(1,)]
        else:
            assert rows(owned, "SELECT state FROM evidence_sessions") == [("unsupported",)]
    finally:
        owned.close()


@pytest.mark.parametrize("cold_start", [False, True])
def test_task_5c_recovery_accepts_rollover_predecessor_before_unclean_purge(
    tmp_path: Path,
    cold_start: bool,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.rollover_session(make_rollover_command()) is StoreDisposition.COMMITTED
        assert rows(
            owned,
            "SELECT logical_session_id, state FROM evidence_sessions"
            " ORDER BY logical_session_id",
        ) == [(SESSION_ID, "sealed"), (SUCCESSOR_ID, "open")]
        if cold_start:
            owned.close()
            owned = make_spool(tmp_path, clock=StubClock(START + timedelta(seconds=10)))

        assert owned.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(owned, "SELECT COUNT(*) FROM consent_epochs") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM evidence_events") == [(0,)]
        assert rows(
            owned,
            "SELECT scope_kind, scope_key, reason_code FROM erasure_tombstones",
        ) == [("consent_epoch", EPOCH_ID, "unclean_epoch")]
    finally:
        owned.close()


def test_no_transport_method_raises_not_implemented(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import sqlite_spool

    source = inspect.getsource(sqlite_spool.SQLiteEvidenceSpool)
    assert "NotImplementedError" not in source
    assert "_deferred(" not in source
    spool.create_epoch(make_create_epoch())
    for name, _parameters, _result, _argument in EXPECTED_TRANSPORT_SIGNATURES:
        method = getattr(spool, name)
        try:
            method() if name in {"recover_existing", "diagnostics"} else method(object())
        except NotImplementedError as exc:  # pragma: no cover - the assertion is the contract
            pytest.fail(f"{name} raised NotImplementedError: {exc}")
        except TypeError:  # pragma: no cover - argument arity is covered elsewhere
            pytest.fail(f"{name} did not accept its exact V1 argument")


def test_sentinel_transitions_write_only_the_inactive_slot(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import StoreDisposition

    log: list[dict[str, object]] = []

    class RecordingSentinelHandle(storage_security.SentinelHandleV1):
        def write_slot(self, offset: int, slot: bytes) -> None:
            entry: dict[str, object] = {
                "offset": offset,
                "length": len(slot),
                "slot": slot,
                "before": self.read_image(),
            }
            super().write_slot(offset, slot)
            entry["after"] = self.read_image()
            log.append(entry)

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock, sentinel_opener=RecordingSentinelHandle)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        clock.override.append(START - timedelta(hours=1))
        owned.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
    finally:
        owned.close()

    # first_create_pending (slot A) -> clear (slot B) -> rollback latch (slot A).
    assert [(entry["offset"], entry["length"]) for entry in log] == [(264, 248), (16, 248)]

    for entry in log:
        offset = int(cast(int, entry["offset"]))
        before = cast(bytes, entry["before"])
        after = cast(bytes, entry["after"])
        assert len(before) == len(after) == 512
        assert after[:16] == before[:16], "the sentinel header must never be rewritten"
        assert after[offset : offset + 248] == entry["slot"]
        predecessor = 16 if offset == 264 else 264
        assert after[predecessor : predecessor + 248] == before[predecessor : predecessor + 248]
        assert storage_security.decode_sentinel_image(after).active_index == (
            0 if offset == 16 else 1
        )

    generations = [
        storage_security.decode_sentinel_image(cast(bytes, entry["after"])).active.generation
        for entry in log
    ]
    assert generations == [2, 3]


def test_next_sentinel_slot_targets_the_inactive_offset_and_fails_closed() -> None:

    from hermes_realtime.evidence import storage_security

    first = storage_security.initial_sentinel_image(STATE_ID)
    offset, slot = storage_security.next_sentinel_slot(first, storage_security.SentinelState.CLEAR)
    assert (offset, len(slot)) == (264, 248)
    assert slot == raw_slot(2, 0, _ZERO_UUID)

    second = first[:264] + slot
    offset, slot = storage_security.next_sentinel_slot(
        second,
        storage_security.SentinelState.FULL_PURGE_PENDING,
        state_generation_id=STATE_ID,
    )
    assert (offset, len(slot)) == (16, 248)
    assert slot == raw_slot(3, 1, UUID(STATE_ID).bytes)

    with pytest.raises(storage_security.EvidenceStorageError):
        storage_security.next_sentinel_slot(
            second,
            storage_security.SentinelState.CLEAR,
            state_generation_id=STATE_ID,
        )
    saturated = raw_image(raw_slot(2**64 - 1, 0, _ZERO_UUID), CLEAR_SLOT_ZERO)
    with pytest.raises(storage_security.EvidenceStorageError):
        storage_security.next_sentinel_slot(
            saturated,
            storage_security.SentinelState.FULL_PURGE_PENDING,
            state_generation_id=STATE_ID,
        )
    equal = raw_image(raw_slot(5, 0, _ZERO_UUID), raw_slot(5, 0, _ZERO_UUID))
    with pytest.raises(storage_security.EvidenceStorageError):
        storage_security.next_sentinel_slot(equal, storage_security.SentinelState.CLEAR)


def test_transport_resolves_the_manifest_once_and_uses_only_retained_paths(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import StoreDisposition

    class CountingManifest(storage_security.EvidenceArtifactManifestV1):
        resolutions = 0
        children = 0

        def resolve(self, root: Path) -> storage_security.ResolvedEvidenceManifestV1:
            CountingManifest.resolutions += 1
            return super().resolve(root)

        def child(self, root: Path, name: str) -> Path:
            CountingManifest.children += 1
            return super().child(root, name)

    manifest = CountingManifest()
    owned = make_spool(tmp_path, manifest=manifest)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert CountingManifest.resolutions == 1
        after_open = CountingManifest.children
        owned.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
        owned.append_binding_close(make_binding_close_command())
        owned.seal_epoch(make_seal_command())
        assert CountingManifest.resolutions == 1
        assert CountingManifest.children == after_open
        assert owned.resolved_manifest is owned.resolved_manifest
        assert owned.resolved_manifest.database == tmp_path / "evidence" / "capture-v1.sqlite3"
    finally:
        owned.close()


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("fresh", id="root_swapped_between_validation_and_activation"),
        pytest.param("drift", id="retained_handle_identity_drifted"),
    ],
)
def test_root_replacement_before_activation_is_refused(tmp_path: Path, failure: str) -> None:
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    RecordingRootHandle.reset()
    setattr(RecordingRootHandle, "fail_fresh" if failure == "fresh" else "fail_revalidate", True)
    owned = make_spool(tmp_path, root_handle_opener=RecordingRootHandle)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.PATH_INVALID
    finally:
        owned.close()
        RecordingRootHandle.reset()

    root = tmp_path / "evidence"
    assert [entry.name for entry in root.iterdir()] == [
        ".hermes-realtime-evidence-root-v1.init"
    ]
    assert not (tmp_path / "swapped").exists()


def test_valid_competing_sentinel_activation_is_an_ownership_refusal(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    class CompetingSentinelRootHandle(RecordingRootHandle):
        def activate_exact_temporary(  # type: ignore[no-untyped-def]
            self,
            temporary,
            final,
            retained,
            expected,
        ) -> None:
            if final.name == storage_security.MANIFEST_V1.sentinel:
                descriptor = retained.require_descriptor()
                os.lseek(descriptor, 0, os.SEEK_SET)
                image = os.read(descriptor, len(expected) + 1)
                retained.close()
                final.write_bytes(image)
                temporary.unlink()
                raise storage_security.EvidenceStorageError(
                    WriterFault.OWNERSHIP_UNAVAILABLE,
                    "another contender already activated this evidence artifact",
                )
            super().activate_exact_temporary(temporary, final, retained, expected)

    owned = make_spool(tmp_path, root_handle_opener=CompetingSentinelRootHandle)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.OWNERSHIP_UNAVAILABLE
    finally:
        owned.close()

    resolved = storage_security.MANIFEST_V1.resolve(tmp_path / "evidence")
    assert not resolved.sentinel_init.exists()
    assert storage_security.decode_sentinel_image(resolved.sentinel.read_bytes())
    assert storage_security.parse_root_marker(resolved.root_marker.read_bytes())
    assert not resolved.database.exists()


def test_root_authority_is_opened_once_and_closed_exactly_once(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import DrainAndStopV1, StoreDisposition

    RecordingRootHandle.reset()
    owned = make_spool(tmp_path, root_handle_opener=RecordingRootHandle)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.IDEMPOTENT
        owned.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
        assert len(RecordingRootHandle.instances) == 1
        assert RecordingRootHandle.instances[0].closed == 0
        owned.drain_and_close(
            DrainAndStopV1(protocol_version=1, owner_generation=1, final_admission_ordinal=3)
        )
        assert RecordingRootHandle.instances[0].closed == 1
    finally:
        owned.close()
    assert len(RecordingRootHandle.instances) == 1
    assert RecordingRootHandle.instances[0].closed == 1
    RecordingRootHandle.reset()


def test_unknown_root_occupant_refuses_ownership_before_any_artifact(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    root = tmp_path / "evidence"
    root.mkdir(parents=True)
    (root / "unowned.txt").write_bytes(b"someone else lives here")

    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.PATH_INVALID
    finally:
        owned.close()

    assert sorted(entry.name for entry in root.iterdir()) == ["unowned.txt"]


def test_marker_absent_permits_only_its_own_activation_temporary(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    empty = tmp_path / "empty" / "evidence"
    empty.mkdir(parents=True)
    with_temp = tmp_path / "with_temp" / "evidence"
    with_temp.mkdir(parents=True)
    (with_temp / ".hermes-realtime-evidence-root-v1.init").write_bytes(b"partial")
    with_sentinel = tmp_path / "with_sentinel" / "evidence"
    with_sentinel.mkdir(parents=True)
    (with_sentinel / "capture-v1.owner").write_bytes(b"\x00" * 512)

    for leaf, expected in (
        (empty, RecoveryDisposition.ABSENT),
        (with_temp, RecoveryDisposition.FAULTED),
        (with_sentinel, RecoveryDisposition.FAULTED),
    ):
        owned = make_spool(leaf.parent)
        try:
            assert owned.recover_existing() is expected, leaf.name
        finally:
            owned.close()

    created = make_spool(tmp_path / "empty")
    try:
        assert created.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        created.close()


def test_established_ownership_ignores_adjacent_decoys(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()

    root = tmp_path / "evidence"
    decoys = {"runner-decoy.bin": b"decoy", "notes.txt": b"kept"}
    for name, payload in decoys.items():
        (root / name).write_bytes(payload)

    reopened = make_spool(tmp_path)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_events")[0][0] == 0
        assert (
            reopened.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
            is StoreDisposition.REJECTED_STATE
        )
    finally:
        reopened.close()

    for name, payload in decoys.items():
        assert (root / name).read_bytes() == payload


# --------------------------------------------------------------------------------------
# Task 5A: physical-store recovery and exact full purge.
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Task 5B0/5B1: durable revoke authority schema and request commit.
# --------------------------------------------------------------------------------------


def test_revoke_commit_persists_exact_pending_request_and_is_restart_idempotent(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        CaptureState,
        OwnerState,
        RecoveryDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(940),
        consent_epoch_id=EPOCH_ID,
        control_sequence=1,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        content_counts = (
            rows(first, "SELECT COUNT(*) FROM evidence_conflicts")[0][0],
            rows(first, "SELECT COUNT(*) FROM evidence_sessions")[0][0],
            rows(first, "SELECT COUNT(*) FROM evidence_events")[0][0],
        )

        assert first.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        assert first.diagnostics().capture_state is CaptureState.REVOKED_PURGING
        assert first.diagnostics().pending_revoke is True
        blocked_record = ordinary_record(turn_opened_snapshot(3, event_index=941), 3)
        assert first.append_record(blocked_record) is StoreDisposition.REJECTED_STATE
        assert first.rollover_session(make_rollover_command()) is StoreDisposition.REJECTED_STATE
        assert rows(first, "SELECT consent_epoch_id, state FROM consent_epochs") == [
            (EPOCH_ID, "revoked")
        ]
        scheduled = rows(
            first,
            "SELECT erasure_request_id, scope_kind, scope_key, reason_code, state, "
            "control_sequence, control_fingerprint_hash, last_admission_ordinal, "
            "final_admission_ordinal, erased_session_count, erased_event_count, resume_state "
            "FROM erasure_requests",
        )
        assert scheduled == [
            (
                request.erasure_request_id,
                "consent_epoch",
                EPOCH_ID,
                "revoked",
                "pending",
                request.control_sequence,
                request.control_fingerprint_hash,
                request.last_admission_ordinal,
                None,
                None,
                None,
                None,
            )
        ]
        assert (
            rows(first, "SELECT COUNT(*) FROM evidence_conflicts")[0][0],
            rows(first, "SELECT COUNT(*) FROM evidence_sessions")[0][0],
            rows(first, "SELECT COUNT(*) FROM evidence_events")[0][0],
        ) == content_counts
        assert rows(first, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=StubClock(START + timedelta(seconds=10)))
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert reopened.diagnostics().capture_state is CaptureState.REVOKED_PURGING
        assert reopened.diagnostics().owner_state is OwnerState.RECOVERY_ONLY
        assert reopened.diagnostics().pending_revoke is True
        assert reopened.append_record(blocked_record) is StoreDisposition.REJECTED_STATE
        assert reopened.rollover_session(make_rollover_command()) is StoreDisposition.REJECTED_STATE
        assert reopened.commit_revoke_request(request) is RevokeDisposition.ALREADY_SCHEDULED
        assert rows(
            reopened,
            "SELECT erasure_request_id, scope_kind, scope_key, reason_code, state, "
            "control_sequence, control_fingerprint_hash, last_admission_ordinal, "
            "final_admission_ordinal, erased_session_count, erased_event_count, resume_state "
            "FROM erasure_requests",
        ) == scheduled
    finally:
        reopened.close()


def test_revoke_commit_fences_seal_mutation_before_and_after_restart(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(942),
        consent_epoch_id=EPOCH_ID,
        control_sequence=1,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=3,
    )
    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            first.append_binding_close(make_binding_close_command())
            is StoreDisposition.COMMITTED
        )
        assert first.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        assert first.seal_epoch(make_seal_command()) is StoreDisposition.REJECTED_STATE
        assert rows(first, "SELECT state, event_count FROM evidence_sessions") == [("open", 3)]
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=StubClock(START + timedelta(seconds=10)))
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert reopened.seal_epoch(make_seal_command()) is StoreDisposition.REJECTED_STATE
        assert rows(reopened, "SELECT state, event_count FROM evidence_sessions") == [("open", 3)]
    finally:
        reopened.close()


def test_revoke_commit_conflicts_remain_durably_nonmutating_after_restart(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(943),
        consent_epoch_id=EPOCH_ID,
        control_sequence=1,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert first.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=StubClock(START + timedelta(seconds=10)))
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        before = (
            rows(reopened, "SELECT * FROM erasure_requests"),
            rows(reopened, "SELECT * FROM consent_epochs"),
            rows(reopened, "SELECT clock_high_water_utc FROM producer_installation"),
        )
        conflicts = (
            dataclasses.replace(request, control_sequence=2),
            dataclasses.replace(request, control_fingerprint_hash="cd" * 32),
            dataclasses.replace(request, last_admission_ordinal=3),
            dataclasses.replace(request, erasure_request_id=event_uuid(944)),
        )
        for conflict in conflicts:
            assert reopened.commit_revoke_request(conflict) is RevokeDisposition.WRITER_FAULT
            assert reopened.diagnostics().sticky_fault is None
            assert (
                rows(reopened, "SELECT * FROM erasure_requests"),
                rows(reopened, "SELECT * FROM consent_epochs"),
                rows(reopened, "SELECT clock_high_water_utc FROM producer_installation"),
            ) == before
    finally:
        reopened.close()


def test_task_5b2_private_erasure_progression_deletes_scope_and_persists_tombstone(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import PurgeDisposition, StoreDisposition

    request_id = event_uuid(945)
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        owned.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
            " resume_state, erased_session_count, erased_event_count)"
            " VALUES (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, 2,"
            " NULL, NULL, NULL, NULL)",
            (request_id, EPOCH_ID, UTC_TEXT),
        )

        assert owned._progress_erasure(request_id) is PurgeDisposition.PURGE_COMPLETED
        assert rows(owned, "SELECT COUNT(*) FROM consent_epochs") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM evidence_events") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM erasure_requests") == [(0,)]
        assert rows(
            owned,
            "SELECT erasure_request_id, scope_kind, scope_key, reason_code,"
            " control_sequence, control_fingerprint_hash, last_admission_ordinal,"
            " final_admission_ordinal, erased_session_count, erased_event_count"
            " FROM erasure_tombstones",
        ) == [
            (
                request_id,
                "consent_epoch",
                EPOCH_ID,
                "unclean_epoch",
                None,
                None,
                2,
                None,
                1,
                2,
            )
        ]
    finally:
        owned.close()


def test_task_5b2_vacuum_failure_is_durable_and_restart_resumes_without_recount(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        PurgeDisposition,
        RecoveryDisposition,
        StoreDisposition,
    )

    class FailingVacuumConnection(sqlite3.Connection):
        fail_vacuum = True

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
            if FailingVacuumConnection.fail_vacuum and sql == "VACUUM":
                raise sqlite3.OperationalError("injected maintenance failure")
            return super().execute(sql, *args)

    request_id = event_uuid(946)
    first = make_spool(tmp_path, connection_factory=FailingVacuumConnection)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        first.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
            " resume_state, erased_session_count, erased_event_count)"
            " VALUES (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, 2,"
            " NULL, NULL, NULL, NULL)",
            (request_id, EPOCH_ID, UTC_TEXT),
        )

        assert first._progress_erasure(request_id) is PurgeDisposition.PURGE_FAILED
        assert rows(first, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(first, "SELECT COUNT(*) FROM evidence_events") == [(0,)]
        assert rows(
            first,
            "SELECT state, resume_state, erased_session_count, erased_event_count"
            " FROM erasure_requests WHERE erasure_request_id=?",
            (request_id,),
        ) == [("purge_failed", "logical_deleted", 1, 2)]
    finally:
        first.close()

    FailingVacuumConnection.fail_vacuum = False
    reopened = make_spool(tmp_path, connection_factory=FailingVacuumConnection)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(
            reopened,
            "SELECT erased_session_count, erased_event_count FROM erasure_tombstones"
            " WHERE erasure_request_id=?",
            (request_id,),
        ) == [(1, 2)]
        assert reopened._progress_erasure(request_id) is PurgeDisposition.ALREADY_ABSENT
    finally:
        reopened.close()


def test_task_5b2_revoke_cannot_progress_without_final_admission_watermark(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        PurgeDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(947),
        consent_epoch_id=EPOCH_ID,
        control_sequence=1,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        before = (
            rows(owned, "SELECT * FROM consent_epochs"),
            rows(owned, "SELECT * FROM evidence_sessions"),
            rows(owned, "SELECT * FROM evidence_events"),
        )

        assert owned._progress_erasure(request.erasure_request_id) is PurgeDisposition.PURGE_FAILED
        assert (
            rows(owned, "SELECT * FROM consent_epochs"),
            rows(owned, "SELECT * FROM evidence_sessions"),
            rows(owned, "SELECT * FROM evidence_events"),
        ) == before
        assert rows(
            owned,
            "SELECT state, resume_state, final_admission_ordinal,"
            " erased_session_count, erased_event_count FROM erasure_requests",
        ) == [("pending", None, None, None, None)]
        assert rows(owned, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        owned.close()


def test_task_5b2_session_scope_preserves_epoch_successor_and_exact_counts(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import PurgeDisposition, StoreDisposition
    from hermes_realtime.evidence.sqlite_spool import format_canonical_utc

    request_id = event_uuid(948)
    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.rollover_session(make_rollover_command()) is StoreDisposition.COMMITTED
        assert rows(
            owned,
            "SELECT logical_session_id, event_count FROM evidence_sessions"
            " ORDER BY logical_session_id",
        ) == [(SESSION_ID, 4), (SUCCESSOR_ID, 2)]
        requested_at = format_canonical_utc(START + timedelta(hours=25))
        expires_at = str(
            rows(
                owned,
                "SELECT expires_at_utc FROM evidence_sessions WHERE logical_session_id=?",
                (SESSION_ID,),
            )[0][0]
        )
        owned.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc,"
            " last_admission_ordinal, final_admission_ordinal, resume_state,"
            " erased_session_count, erased_event_count)"
            " VALUES (?, 'session', ?, ?, 'ttl', 'pending', NULL, ?, ?, ?, 6, NULL,"
            " NULL, NULL, NULL)",
            (
                request_id,
                SESSION_ID,
                requested_at,
                HEX64,
                EPOCH_ID,
                expires_at,
            ),
        )
        clock.override.append(START + timedelta(hours=25))

        assert owned._progress_erasure(request_id) is PurgeDisposition.PURGE_COMPLETED
        assert rows(owned, "SELECT consent_epoch_id FROM consent_epochs") == [(EPOCH_ID,)]
        assert rows(
            owned,
            "SELECT logical_session_id, event_count FROM evidence_sessions",
        ) == [(SUCCESSOR_ID, 2)]
        assert rows(
            owned,
            "SELECT DISTINCT logical_session_id FROM evidence_events",
        ) == [(SUCCESSOR_ID,)]
        assert rows(
            owned,
            "SELECT scope_kind, scope_key, reason_code, erased_session_count,"
            " erased_event_count FROM erasure_tombstones",
        ) == [("session", SESSION_ID, "ttl", 1, 4)]
    finally:
        owned.close()


def test_task_5b2_rejects_live_request_tombstone_overlap_before_deletion(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import PurgeDisposition, StoreDisposition

    request_id = event_uuid(949)
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        owned.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
            " resume_state, erased_session_count, erased_event_count)"
            " VALUES (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, 2,"
            " NULL, NULL, NULL, NULL)",
            (request_id, EPOCH_ID, UTC_TEXT),
        )
        owned.connection.execute(
            "INSERT INTO erasure_tombstones (erasure_request_id, scope_kind, scope_key,"
            " reason_code, control_sequence, control_fingerprint_hash,"
            " last_admission_ordinal, final_admission_ordinal, erased_at_utc,"
            " erased_session_count, erased_event_count)"
            " VALUES (?, 'consent_epoch', ?, 'unclean_epoch', NULL, NULL, 2, NULL, ?, 1, 2)",
            (request_id, EPOCH_ID, UTC_TEXT),
        )
        before = (
            rows(owned, "SELECT * FROM consent_epochs"),
            rows(owned, "SELECT * FROM evidence_sessions"),
            rows(owned, "SELECT * FROM evidence_events"),
        )

        assert owned._progress_erasure(request_id) is PurgeDisposition.PURGE_FAILED
        assert (
            rows(owned, "SELECT * FROM consent_epochs"),
            rows(owned, "SELECT * FROM evidence_sessions"),
            rows(owned, "SELECT * FROM evidence_events"),
        ) == before
        assert rows(
            owned,
            "SELECT state, resume_state, erased_session_count, erased_event_count"
            " FROM erasure_requests",
        ) == [("pending", None, None, None)]
    finally:
        owned.close()


def test_task_5b2_vacuum_removes_target_payload_bytes_from_every_owned_artifact(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import PurgeDisposition, StoreDisposition

    request_id = event_uuid(950)
    needle = "task5b2-erasure-needle-950"
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            owned.append_record(ordinary_record(turn_opened_snapshot(3, event_index=951), 3))
            is StoreDisposition.COMMITTED
        )
        assert (
            owned.append_record(
                ordinary_record(user_final_snapshot(4, event_index=952, text=needle), 4)
            )
            is StoreDisposition.COMMITTED
        )
        owned.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
            " resume_state, erased_session_count, erased_event_count)"
            " VALUES (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, 4,"
            " NULL, NULL, NULL, NULL)",
            (request_id, EPOCH_ID, UTC_TEXT),
        )

        assert owned._progress_erasure(request_id) is PurgeDisposition.PURGE_COMPLETED
        assert rows(
            owned,
            "SELECT erased_session_count, erased_event_count FROM erasure_tombstones",
        ) == [(1, 4)]
        encoded = needle.encode("utf-8")
        for artifact in owned.resolved_manifest.deletable:
            if artifact.is_file():
                assert encoded not in artifact.read_bytes(), artifact.name
    finally:
        owned.close()


@pytest.mark.parametrize("corruption", ["malformed_authority", "schema_drift"])
def test_task_5b2_corrupt_authority_fails_before_any_scope_mutation(
    tmp_path: Path,
    corruption: str,
) -> None:
    from hermes_realtime.evidence.models import (
        PurgeDisposition,
        StoreDisposition,
        WriterFault,
    )

    request_id = event_uuid(953)
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        if corruption == "malformed_authority":
            owned.connection.execute("PRAGMA ignore_check_constraints=ON")
            owned.connection.execute(
                "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
                " requested_at_utc, reason_code, state, control_sequence,"
                " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
                " resume_state, erased_session_count, erased_event_count)"
                " VALUES (?, 'consent_epoch', ?, ?, 'ttl', 'pending', NULL, NULL, 2,"
                " NULL, NULL, NULL, NULL)",
                (request_id, EPOCH_ID, UTC_TEXT),
            )
            owned.connection.execute("PRAGMA ignore_check_constraints=OFF")
        else:
            owned.connection.execute(
                "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
                " requested_at_utc, reason_code, state, control_sequence,"
                " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
                " resume_state, erased_session_count, erased_event_count)"
                " VALUES (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, 2,"
                " NULL, NULL, NULL, NULL)",
                (request_id, EPOCH_ID, UTC_TEXT),
            )
            owned.connection.execute("DROP TRIGGER erasure_request_authority_is_immutable")
        before = (
            rows(owned, "SELECT * FROM consent_epochs"),
            rows(owned, "SELECT * FROM evidence_sessions"),
            rows(owned, "SELECT * FROM evidence_events"),
        )

        assert owned._progress_erasure(request_id) is PurgeDisposition.PURGE_FAILED
        assert owned.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert (
            rows(owned, "SELECT * FROM consent_epochs"),
            rows(owned, "SELECT * FROM evidence_sessions"),
            rows(owned, "SELECT * FROM evidence_events"),
        ) == before
        assert rows(owned, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        owned.close()


def test_task_5b2_maintenance_headroom_failure_is_durable_after_logical_delete(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        PurgeDisposition,
        RecoveryDisposition,
        StoreDisposition,
    )

    probe = InjectedStorageProbe()
    request_id = event_uuid(954)
    owned = make_spool(tmp_path, probe=probe)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        owned.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
            " resume_state, erased_session_count, erased_event_count)"
            " VALUES (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, 2,"
            " NULL, NULL, NULL, NULL)",
            (request_id, EPOCH_ID, UTC_TEXT),
        )
        probe.free = 0

        assert owned._progress_erasure(request_id) is PurgeDisposition.PURGE_FAILED
        assert rows(
            owned,
            "SELECT state, resume_state, erased_session_count, erased_event_count"
            " FROM erasure_requests",
        ) == [("purge_failed", "logical_deleted", 1, 2)]
        assert rows(owned, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        probe.free = 1 << 40
        assert owned.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(
            owned,
            "SELECT erased_session_count, erased_event_count FROM erasure_tombstones",
        ) == [(1, 2)]
    finally:
        owned.close()


def test_task_5b2_predelete_failure_resumes_pending_checkpoint_on_recovery(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        PurgeDisposition,
        RecoveryDisposition,
        StoreDisposition,
    )

    class FailingDeleteConnection(sqlite3.Connection):
        fail_delete = True

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
            if FailingDeleteConnection.fail_delete and sql.startswith(
                "DELETE FROM evidence_sessions"
            ):
                raise sqlite3.OperationalError("injected predelete failure")
            return super().execute(sql, *args)

    request_id = event_uuid(955)
    first = make_spool(tmp_path, connection_factory=FailingDeleteConnection)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        first.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
            " resume_state, erased_session_count, erased_event_count)"
            " VALUES (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, 2,"
            " NULL, NULL, NULL, NULL)",
            (request_id, EPOCH_ID, UTC_TEXT),
        )

        assert first._progress_erasure(request_id) is PurgeDisposition.PURGE_FAILED
        assert rows(first, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
        assert rows(first, "SELECT COUNT(*) FROM evidence_events") == [(2,)]
        assert rows(
            first,
            "SELECT state, resume_state, erased_session_count, erased_event_count"
            " FROM erasure_requests",
        ) == [("purge_failed", "pending", None, None)]
    finally:
        first.close()

    FailingDeleteConnection.fail_delete = False
    reopened = make_spool(tmp_path, connection_factory=FailingDeleteConnection)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(
            reopened,
            "SELECT erased_session_count, erased_event_count FROM erasure_tombstones",
        ) == [(1, 2)]
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_events") == [(0,)]
    finally:
        reopened.close()


def test_task_5b2_clock_rollback_during_tombstone_time_remains_retryable(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import PurgeDisposition, StoreDisposition, WriterFault

    clock = StubClock()
    request_id = event_uuid(956)
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        owned.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
            " resume_state, erased_session_count, erased_event_count)"
            " VALUES (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, 2,"
            " NULL, NULL, NULL, NULL)",
            (request_id, EPOCH_ID, UTC_TEXT),
        )
        clock.override.append(START - timedelta(seconds=1))

        assert owned._progress_erasure(request_id) is PurgeDisposition.PURGE_FAILED
        assert owned.diagnostics().sticky_fault is WriterFault.PURGE_REQUIRED
        assert rows(
            owned,
            "SELECT state, resume_state, erased_session_count, erased_event_count"
            " FROM erasure_requests",
        ) == [("purge_failed", "logical_deleted", 1, 2)]
        assert rows(owned, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        owned.close()


def test_task_5b2_pending_missing_scope_is_never_inferred_as_success(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        StoreDisposition,
        WriterFault,
    )

    request_id = event_uuid(957)
    missing_session = event_uuid(958)
    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        first.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, state, control_sequence,"
            " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc,"
            " last_admission_ordinal, final_admission_ordinal, resume_state,"
            " erased_session_count, erased_event_count)"
            " VALUES (?, 'session', ?, ?, 'ttl', 'pending', NULL, ?, ?, ?, 2, NULL,"
            " NULL, NULL, NULL)",
            (request_id, missing_session, UTC_TEXT, HEX64, EPOCH_ID, UTC_TEXT),
        )
    finally:
        first.close()

    reopened = make_spool(tmp_path)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert reopened.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_events") == [(2,)]
        assert rows(
            reopened,
            "SELECT state, resume_state, erased_session_count, erased_event_count"
            " FROM erasure_requests",
        ) == [("pending", None, None, None)]
        assert rows(reopened, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        reopened.close()


def test_task_5b2_terminal_tombstone_cannot_coexist_with_its_live_scope(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition, WriterFault

    tombstone_id = event_uuid(959)
    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        first.connection.execute(
            "INSERT INTO erasure_tombstones (erasure_request_id, scope_kind, scope_key,"
            " reason_code, control_sequence, control_fingerprint_hash,"
            " ttl_consent_epoch_id, ttl_expires_at_utc, last_admission_ordinal,"
            " final_admission_ordinal, erased_at_utc, erased_session_count,"
            " erased_event_count)"
            " VALUES (?, 'session', ?, 'ttl', NULL, ?, ?, ?, 2, NULL, ?, 1, 2)",
            (tombstone_id, SESSION_ID, HEX64, EPOCH_ID, UTC_TEXT, UTC_TEXT),
        )
    finally:
        first.close()

    reopened = make_spool(tmp_path)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert reopened.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_events") == [(2,)]
        assert rows(reopened, "SELECT COUNT(*) FROM erasure_tombstones") == [(1,)]
    finally:
        reopened.close()


def test_task_5b2_deferred_revoke_authority_is_validated_before_recovery_success(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition, WriterFault

    request_id = event_uuid(960)
    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        first.connection.execute(
            "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
            " requested_at_utc, reason_code, control_sequence,"
            " control_fingerprint_hash, last_admission_ordinal, state)"
            " VALUES (?, 'consent_epoch', ?, ?, 'revoked', 1, ?, 2, 'pending')",
            (
                request_id,
                EPOCH_ID,
                "2026-99-99T99:99:99.000000Z",
                "cd" * 32,
            ),
        )
    finally:
        first.close()

    reopened = make_spool(tmp_path)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert reopened.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(reopened, "SELECT state FROM erasure_requests") == [("pending",)]
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
        assert rows(reopened, "SELECT COUNT(*) FROM evidence_events") == [(2,)]
    finally:
        reopened.close()


# --------------------------------------------------------------------------------------
# Task 5B3: revoke finalization and restart.
# --------------------------------------------------------------------------------------


def test_task_5b3_finalize_persists_watermark_and_erases_exact_revoke_scope(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RevokeDisposition,
        RevokeFinalizeV1,
        RevokeRequestV1,
        StoreDisposition,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(961),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    finalize = RevokeFinalizeV1(
        protocol_version=1,
        erasure_request_id=request.erasure_request_id,
        consent_epoch_id=request.consent_epoch_id,
        control_sequence=request.control_sequence,
        control_fingerprint_hash=request.control_fingerprint_hash,
        final_admission_ordinal=2,
    )
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED

        assert owned.finalize_revoke(finalize) is RevokeDisposition.PURGE_COMPLETED
        assert rows(owned, "SELECT COUNT(*) FROM erasure_requests") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM consent_epochs") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
        assert rows(owned, "SELECT COUNT(*) FROM evidence_events") == [(0,)]
        assert rows(
            owned,
            "SELECT erasure_request_id, scope_kind, scope_key, reason_code,"
            " control_sequence, control_fingerprint_hash, last_admission_ordinal,"
            " final_admission_ordinal, erased_session_count, erased_event_count"
            " FROM erasure_tombstones",
        ) == [
            (
                request.erasure_request_id,
                "consent_epoch",
                EPOCH_ID,
                "revoked",
                7,
                HEX64,
                2,
                2,
                1,
                2,
            )
        ]
    finally:
        owned.close()


def test_task_5b3_conflicting_or_missing_finalize_is_nonmutating(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import (
        RevokeDisposition,
        RevokeFinalizeV1,
        RevokeRequestV1,
        StoreDisposition,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(962),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    finalize = RevokeFinalizeV1(
        protocol_version=1,
        erasure_request_id=request.erasure_request_id,
        consent_epoch_id=request.consent_epoch_id,
        control_sequence=request.control_sequence,
        control_fingerprint_hash=request.control_fingerprint_hash,
        final_admission_ordinal=2,
    )
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        before = (
            rows(owned, "SELECT * FROM erasure_requests"),
            rows(owned, "SELECT * FROM consent_epochs"),
            rows(owned, "SELECT * FROM evidence_sessions"),
            rows(owned, "SELECT * FROM evidence_events"),
            rows(owned, "SELECT * FROM erasure_tombstones"),
        )

        conflicts = (
            dataclasses.replace(finalize, erasure_request_id=event_uuid(963)),
            dataclasses.replace(finalize, consent_epoch_id=event_uuid(966)),
            dataclasses.replace(finalize, control_sequence=8),
            dataclasses.replace(finalize, control_fingerprint_hash="cd" * 32),
            dataclasses.replace(finalize, final_admission_ordinal=1),
        )
        for conflict in conflicts:
            assert owned.finalize_revoke(conflict) is RevokeDisposition.WRITER_FAULT
            assert (
                rows(owned, "SELECT * FROM erasure_requests"),
                rows(owned, "SELECT * FROM consent_epochs"),
                rows(owned, "SELECT * FROM evidence_sessions"),
                rows(owned, "SELECT * FROM evidence_events"),
                rows(owned, "SELECT * FROM erasure_tombstones"),
            ) == before
    finally:
        owned.close()


def test_task_5b3_purge_failure_retains_final_authority_and_restart_completes(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        RevokeDisposition,
        RevokeFinalizeV1,
        RevokeRequestV1,
        StoreDisposition,
    )

    class FailingVacuumConnection(sqlite3.Connection):
        fail_vacuum = True

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
            if FailingVacuumConnection.fail_vacuum and sql == "VACUUM":
                raise sqlite3.OperationalError("injected maintenance failure")
            return super().execute(sql, *args)

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(964),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    finalize = RevokeFinalizeV1(
        protocol_version=1,
        erasure_request_id=request.erasure_request_id,
        consent_epoch_id=request.consent_epoch_id,
        control_sequence=request.control_sequence,
        control_fingerprint_hash=request.control_fingerprint_hash,
        final_admission_ordinal=3,
    )
    clock = StubClock()
    first = make_spool(
        tmp_path,
        clock=clock,
        connection_factory=FailingVacuumConnection,
    )
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert first.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        assert first.finalize_revoke(finalize) is RevokeDisposition.PURGE_FAILED
        assert rows(
            first,
            "SELECT state, resume_state, final_admission_ordinal,"
            " erased_session_count, erased_event_count FROM erasure_requests",
        ) == [("purge_failed", "logical_deleted", 3, 1, 2)]
        assert rows(first, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        first.close()

    FailingVacuumConnection.fail_vacuum = False
    reopened = make_spool(
        tmp_path,
        clock=clock,
        connection_factory=FailingVacuumConnection,
    )
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        tombstone = rows(reopened, "SELECT * FROM erasure_tombstones")
        assert len(tombstone) == 1
        assert reopened.finalize_revoke(finalize) is RevokeDisposition.PURGE_COMPLETED
        assert rows(reopened, "SELECT * FROM erasure_tombstones") == tombstone
    finally:
        reopened.close()


def test_task_5b3_restart_completes_after_final_authority_commit_before_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence.models import (
        PurgeDisposition,
        RecoveryDisposition,
        RevokeDisposition,
        RevokeFinalizeV1,
        RevokeRequestV1,
        StoreDisposition,
    )
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(968),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    finalize = RevokeFinalizeV1(
        protocol_version=1,
        erasure_request_id=request.erasure_request_id,
        consent_epoch_id=request.consent_epoch_id,
        control_sequence=request.control_sequence,
        control_fingerprint_hash=request.control_fingerprint_hash,
        final_admission_ordinal=3,
    )
    clock = StubClock()
    first = make_spool(tmp_path, clock=clock)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert first.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        with monkeypatch.context() as context:
            context.setattr(
                SQLiteEvidenceSpool,
                "_progress_erasure",
                lambda self, request_id: PurgeDisposition.PURGE_FAILED,
            )
            assert first.finalize_revoke(finalize) is RevokeDisposition.PURGE_FAILED
        assert rows(
            first,
            "SELECT state, final_admission_ordinal FROM erasure_requests",
        ) == [("pending", 3)]
        assert rows(first, "SELECT COUNT(*) FROM evidence_sessions") == [(1,)]
        assert rows(first, "SELECT COUNT(*) FROM evidence_events") == [(2,)]
    finally:
        first.close()

    reopened = make_spool(tmp_path, clock=clock)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.RECOVERED
        assert rows(reopened, "SELECT COUNT(*) FROM erasure_requests") == [(0,)]
        assert rows(reopened, "SELECT COUNT(*) FROM erasure_tombstones") == [(1,)]
        assert reopened.finalize_revoke(finalize) is RevokeDisposition.PURGE_COMPLETED
    finally:
        reopened.close()


def test_task_5b3_terminal_replay_requires_the_exact_complete_tombstone(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RevokeDisposition,
        RevokeFinalizeV1,
        RevokeRequestV1,
        StoreDisposition,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(965),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    finalize = RevokeFinalizeV1(
        protocol_version=1,
        erasure_request_id=request.erasure_request_id,
        consent_epoch_id=request.consent_epoch_id,
        control_sequence=request.control_sequence,
        control_fingerprint_hash=request.control_fingerprint_hash,
        final_admission_ordinal=2,
    )
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        assert owned.finalize_revoke(finalize) is RevokeDisposition.PURGE_COMPLETED
        tombstone = rows(owned, "SELECT * FROM erasure_tombstones")

        assert owned.finalize_revoke(finalize) is RevokeDisposition.PURGE_COMPLETED
        assert owned.finalize_revoke(
            dataclasses.replace(finalize, control_fingerprint_hash="cd" * 32)
        ) is RevokeDisposition.WRITER_FAULT
        assert owned.finalize_revoke(
            dataclasses.replace(finalize, final_admission_ordinal=3)
        ) is RevokeDisposition.WRITER_FAULT
        assert rows(owned, "SELECT * FROM erasure_tombstones") == tombstone
    finally:
        owned.close()


def test_task_5b3_terminal_replay_latches_exact_store_corruption(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import (
        RevokeDisposition,
        RevokeFinalizeV1,
        RevokeRequestV1,
        StoreDisposition,
        WriterFault,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(967),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    finalize = RevokeFinalizeV1(
        protocol_version=1,
        erasure_request_id=request.erasure_request_id,
        consent_epoch_id=request.consent_epoch_id,
        control_sequence=request.control_sequence,
        control_fingerprint_hash=request.control_fingerprint_hash,
        final_admission_ordinal=2,
    )
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        assert owned.finalize_revoke(finalize) is RevokeDisposition.PURGE_COMPLETED
        owned.connection.execute("DROP TRIGGER erasure_tombstones_are_append_only")
        owned.connection.commit()
        tombstone = rows(owned, "SELECT * FROM erasure_tombstones")

        assert owned.finalize_revoke(finalize) is RevokeDisposition.WRITER_FAULT
        assert owned.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(owned, "SELECT * FROM erasure_tombstones") == tombstone
    finally:
        owned.close()


def test_task_5b3_unfinalized_revoke_cannot_recover_as_logically_deleted(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
        WriterFault,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(969),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert first.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        first.connection.execute("PRAGMA ignore_check_constraints=ON")
        first.connection.execute("BEGIN IMMEDIATE")
        first.connection.execute(
            "DELETE FROM evidence_sessions WHERE consent_epoch_id=?",
            (EPOCH_ID,),
        )
        first.connection.execute(
            "DELETE FROM consent_epochs WHERE consent_epoch_id=?",
            (EPOCH_ID,),
        )
        first.connection.execute(
            "UPDATE erasure_requests SET state='logical_deleted',"
            " erased_session_count=1, erased_event_count=2"
            " WHERE erasure_request_id=?",
            (request.erasure_request_id,),
        )
        first.connection.commit()
    finally:
        first.close()

    reopened = make_spool(tmp_path)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert reopened.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(
            reopened,
            "SELECT state, final_admission_ordinal FROM erasure_requests",
        ) == [("logical_deleted", None)]
        assert rows(reopened, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        reopened.close()


def test_task_5b3_finalize_cannot_legitimize_an_unfinalized_deleted_state(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RevokeDisposition,
        RevokeFinalizeV1,
        RevokeRequestV1,
        StoreDisposition,
        WriterFault,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(970),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    finalize = RevokeFinalizeV1(
        protocol_version=1,
        erasure_request_id=request.erasure_request_id,
        consent_epoch_id=request.consent_epoch_id,
        control_sequence=request.control_sequence,
        control_fingerprint_hash=request.control_fingerprint_hash,
        final_admission_ordinal=2,
    )
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        owned.connection.execute("PRAGMA ignore_check_constraints=ON")
        owned.connection.execute("BEGIN IMMEDIATE")
        owned.connection.execute(
            "DELETE FROM evidence_sessions WHERE consent_epoch_id=?",
            (EPOCH_ID,),
        )
        owned.connection.execute(
            "DELETE FROM consent_epochs WHERE consent_epoch_id=?",
            (EPOCH_ID,),
        )
        owned.connection.execute(
            "UPDATE erasure_requests SET state='logical_deleted',"
            " erased_session_count=1, erased_event_count=2"
            " WHERE erasure_request_id=?",
            (request.erasure_request_id,),
        )
        owned.connection.commit()
        durable_before = rows(owned, "SELECT * FROM erasure_requests")

        assert owned.finalize_revoke(finalize) is RevokeDisposition.WRITER_FAULT
        assert owned.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(owned, "SELECT * FROM erasure_requests") == durable_before
        assert rows(owned, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        owned.close()


def test_task_5b3_already_open_recovery_revalidates_unfinalized_durable_state(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import (
        RecoveryDisposition,
        RevokeDisposition,
        RevokeRequestV1,
        StoreDisposition,
        WriterFault,
    )

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=event_uuid(971),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=HEX64,
        last_admission_ordinal=2,
    )
    owned = make_spool(tmp_path)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert owned.commit_revoke_request(request) is RevokeDisposition.REVOKE_DURABLY_SCHEDULED
        owned.connection.execute("PRAGMA ignore_check_constraints=ON")
        owned.connection.execute("BEGIN IMMEDIATE")
        owned.connection.execute(
            "DELETE FROM evidence_sessions WHERE consent_epoch_id=?",
            (EPOCH_ID,),
        )
        owned.connection.execute(
            "DELETE FROM consent_epochs WHERE consent_epoch_id=?",
            (EPOCH_ID,),
        )
        owned.connection.execute(
            "UPDATE erasure_requests SET state='logical_deleted',"
            " erased_session_count=1, erased_event_count=2"
            " WHERE erasure_request_id=?",
            (request.erasure_request_id,),
        )
        owned.connection.commit()
        durable_before = rows(owned, "SELECT * FROM erasure_requests")

        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        assert owned.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
        assert rows(owned, "SELECT * FROM erasure_requests") == durable_before
        assert rows(owned, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
    finally:
        owned.close()


def test_pre_5b_v1_schema_is_refused_without_mutation_or_migration(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition, WriterFault

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        database = first.resolved_manifest.database
    finally:
        first.close()

    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER erasure_request_authority_is_immutable")
        connection.commit()
    finally:
        connection.close()

    reopened = make_spool(tmp_path)
    try:
        assert reopened.recover_existing() is RecoveryDisposition.FAULTED
        assert reopened.diagnostics().sticky_fault is WriterFault.STORE_CORRUPT
    finally:
        reopened.close()


def test_full_purge_authorizes_the_exact_pending_generation_and_removes_only_six_artifacts(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        SentinelState,
        StoreDisposition,
    )

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()

    root = tmp_path / "evidence"
    resolved = storage_security.MANIFEST_V1.resolve(root)
    for artifact in resolved.deletable[1:]:
        artifact.write_bytes(b"owned-sidecar")
    decoys = {"capture-v1.sqlite3.backup": b"decoy", "keep-me.txt": b"adjacent"}
    for name, payload in decoys.items():
        (root / name).write_bytes(payload)
    generation = event_uuid(930)
    resolved.sentinel.write_bytes(
        storage_security.next_sentinel_image(
            resolved.sentinel.read_bytes(),
            SentinelState.FULL_PURGE_PENDING,
            state_generation_id=generation,
        )
    )

    owned = make_spool(tmp_path)
    try:
        unauthorized = FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=event_uuid(931),
            sentinel_state=SentinelState.FULL_PURGE_PENDING,
            artifact_manifest_version=1,
        )
        assert owned.purge_full_store(unauthorized) is PurgeDisposition.PURGE_FAILED
        assert all(path.exists() for path in resolved.deletable)

        command = FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=generation,
            sentinel_state=SentinelState.FULL_PURGE_PENDING,
            artifact_manifest_version=1,
        )
        assert owned.purge_full_store(command) is PurgeDisposition.PURGE_COMPLETED
        assert [path.name for path in resolved.deletable if path.exists()] == []
        assert storage_security.parse_root_marker(resolved.root_marker.read_bytes())
        assert (
            storage_security.decode_sentinel_image(resolved.sentinel.read_bytes()).active.state
            is SentinelState.CLEAR
        )
        for name, payload in decoys.items():
            assert (root / name).read_bytes() == payload
        assert owned.purge_full_store(command) is PurgeDisposition.ALREADY_ABSENT
    finally:
        owned.close()


def test_full_purge_keeps_its_pending_sentinel_until_the_exact_absence_check_succeeds(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        SentinelState,
        StoreDisposition,
    )

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()

    resolved = storage_security.MANIFEST_V1.resolve(tmp_path / "evidence")
    for artifact in resolved.deletable[1:-1]:
        artifact.write_bytes(b"owned-sidecar")
    # A nonempty manifest-owned child makes verified absence impossible without
    # manufacturing an unlisted path; the pending sentinel must survive failure.
    resolved.deletable[-1].mkdir()
    blocker = resolved.deletable[-1] / "not-deletable-yet"
    blocker.write_bytes(b"block")
    generation = event_uuid(932)
    resolved.sentinel.write_bytes(
        storage_security.next_sentinel_image(
            resolved.sentinel.read_bytes(),
            SentinelState.FULL_PURGE_PENDING,
            state_generation_id=generation,
        )
    )
    command = FullPurgeV1(
        protocol_version=1,
        full_purge_generation_id=generation,
        sentinel_state=SentinelState.FULL_PURGE_PENDING,
        artifact_manifest_version=1,
    )

    owned = make_spool(tmp_path)
    try:
        assert owned.purge_full_store(command) is PurgeDisposition.PURGE_FAILED
        assert (
            storage_security.decode_sentinel_image(resolved.sentinel.read_bytes()).active.state
            is SentinelState.FULL_PURGE_PENDING
        )
        blocker.unlink()
        resolved.deletable[-1].rmdir()
        assert owned.purge_full_store(command) is PurgeDisposition.PURGE_COMPLETED
        assert (
            storage_security.decode_sentinel_image(resolved.sentinel.read_bytes()).active.state
            is SentinelState.CLEAR
        )
    finally:
        owned.close()


def test_recovery_finishes_a_pending_full_purge_before_availability_and_linux_touches_nothing(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        RecoveryDisposition,
        SentinelState,
        StoreDisposition,
    )

    first = make_spool(tmp_path / "pending")
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()

    resolved = storage_security.MANIFEST_V1.resolve(tmp_path / "pending" / "evidence")
    for artifact in resolved.deletable[1:]:
        artifact.write_bytes(b"owned-sidecar")
    generation = event_uuid(933)
    resolved.sentinel.write_bytes(
        storage_security.next_sentinel_image(
            resolved.sentinel.read_bytes(),
            SentinelState.FULL_PURGE_PENDING,
            state_generation_id=generation,
        )
    )

    resumed = make_spool(tmp_path / "pending")
    try:
        assert resumed.recover_existing() is RecoveryDisposition.PURGE_COMPLETED
        assert [path.name for path in resumed.resolved_manifest.deletable if path.exists()] == []
        assert (
            storage_security.decode_sentinel_image(
                resumed.resolved_manifest.sentinel.read_bytes()
            ).active.state
            is SentinelState.CLEAR
        )
        assert resumed.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        resumed.close()

    linux = make_spool(tmp_path / "linux")
    linux.probe.supported = False
    try:
        assert (
            linux.purge_full_store(
                FullPurgeV1(
                    protocol_version=1,
                    full_purge_generation_id=event_uuid(934),
                    sentinel_state=SentinelState.FULL_PURGE_PENDING,
                    artifact_manifest_version=1,
                )
            )
            is PurgeDisposition.UNSUPPORTED_PLATFORM
        )
    finally:
        linux.close()
    assert list((tmp_path / "linux" / "evidence").iterdir()) == []


@pytest.mark.parametrize("artifact_count", [1, 6], ids=["one_artifact", "all_six_artifacts"])
def test_recovery_purges_first_create_pending_artifacts_before_clearing_the_sentinel(
    tmp_path: Path, artifact_count: int
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, SentinelState

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    resolved.root_marker.write_bytes(ROOT_MARKER_BYTES)
    resolved.sentinel.write_bytes(storage_security.initial_sentinel_image(STATE_ID))
    for artifact in resolved.deletable[:artifact_count]:
        artifact.write_bytes(b"abandoned-first-create")

    recovered = make_spool(tmp_path)
    try:
        assert recovered.recover_existing() is RecoveryDisposition.PURGE_COMPLETED
        assert [path.name for path in resolved.deletable if path.exists()] == []
        assert (
            storage_security.decode_sentinel_image(resolved.sentinel.read_bytes()).active.state
            is SentinelState.CLEAR
        )
    finally:
        recovered.close()


def test_recovery_keeps_first_create_pending_when_absence_cannot_be_verified(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, SentinelState

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    resolved.root_marker.write_bytes(ROOT_MARKER_BYTES)
    resolved.sentinel.write_bytes(storage_security.initial_sentinel_image(STATE_ID))
    resolved.database_tmp.mkdir()
    blocked = resolved.database_tmp / "retained-by-interruption"
    blocked.write_bytes(b"must-not-be-cleared-early")

    recovered = make_spool(tmp_path)
    try:
        outcome = recovered.recover_existing()
    finally:
        recovered.close()
    assert outcome is RecoveryDisposition.FAULTED
    assert blocked.read_bytes() == b"must-not-be-cleared-early"
    assert (
        storage_security.decode_sentinel_image(resolved.sentinel.read_bytes()).active.state
        is SentinelState.FIRST_CREATE_PENDING
    )


def test_recovery_resumes_clock_rollback_purge_before_availability_and_preserves_decoys(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, SentinelState, StoreDisposition

    clock = StubClock()
    first = make_spool(tmp_path, clock=clock)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        clock.override.append(START - timedelta(hours=1))
        assert (
            first.append_record(ordinary_record(turn_opened_snapshot(3, event_index=930), 3))
            is StoreDisposition.FAULTED
        )
    finally:
        first.close()

    resolved = storage_security.MANIFEST_V1.resolve(tmp_path / "evidence")
    for artifact in resolved.deletable[1:]:
        artifact.write_bytes(b"owned-rollback-sidecar")
    decoys = {"rollback-decoy.bin": b"preserve", "notes.txt": b"also-preserve"}
    for name, contents in decoys.items():
        (tmp_path / "evidence" / name).write_bytes(contents)

    resumed = make_spool(tmp_path)
    try:
        assert resumed.recover_existing() is RecoveryDisposition.PURGE_COMPLETED
        assert [path.name for path in resolved.deletable if path.exists()] == []
        assert (
            storage_security.decode_sentinel_image(resolved.sentinel.read_bytes()).active.state
            is SentinelState.CLEAR
        )
    finally:
        resumed.close()
    for name, contents in decoys.items():
        assert (tmp_path / "evidence" / name).read_bytes() == contents


def test_marker_only_recovery_is_absent_then_exactly_one_contender_creates_the_store(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    resolved.root_marker.write_bytes(ROOT_MARKER_BYTES)
    absent = make_spool(tmp_path)
    try:
        assert absent.recover_existing() is RecoveryDisposition.ABSENT
        assert resolved.root_marker.read_bytes() == ROOT_MARKER_BYTES
    finally:
        absent.close()

    winner, loser = make_spool(tmp_path), make_spool(tmp_path)
    try:
        outcomes = (
            winner.create_epoch(make_create_epoch()),
            loser.create_epoch(make_create_epoch()),
        )
        assert outcomes.count(StoreDisposition.COMMITTED) == 1
        assert outcomes.count(StoreDisposition.FAULTED) == 1
    finally:
        loser.close()
        winner.close()


@pytest.mark.parametrize(
    "partial",
    [
        pytest.param("database", id="marker_and_database_without_final_sentinel"),
        pytest.param("unknown", id="marker_and_unknown_without_final_sentinel"),
        pytest.param("sentinel_init_unknown", id="sentinel_init_with_unknown_occupant"),
    ],
)
def test_invalid_partial_root_without_a_final_sentinel_fails_closed_without_mutation(
    tmp_path: Path, partial: str
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    resolved.root_marker.write_bytes(ROOT_MARKER_BYTES)
    protected: dict[Path, bytes] = {resolved.root_marker: ROOT_MARKER_BYTES}
    if partial == "database":
        resolved.database.write_bytes(b"partial-database-bytes")
        protected[resolved.database] = b"partial-database-bytes"
    elif partial == "unknown":
        unknown = root / "unowned.bin"
        unknown.write_bytes(b"unowned-evidence")
        protected[unknown] = b"unowned-evidence"
    else:
        resolved.sentinel_init.write_bytes(b"contended-sentinel-init")
        unknown = root / "unowned.bin"
        unknown.write_bytes(b"unknown-occupant")
        protected.update(
            {resolved.sentinel_init: b"contended-sentinel-init", unknown: b"unknown-occupant"}
        )

    recovered = make_spool(tmp_path)
    try:
        assert recovered.recover_existing() is RecoveryDisposition.FAULTED
    finally:
        recovered.close()
    assert not resolved.sentinel.exists()
    for path, contents in protected.items():
        assert path.read_bytes() == contents


# --------------------------------------------------------------------------------------
# Task 5A final: recovery boundaries and same-owner full-purge authority.
# --------------------------------------------------------------------------------------


def test_task_5a_final_root_marker_init_only_is_cleaned_to_absent(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    # This is the exact valid root-marker image interrupted before activation; it is
    # the only occupant, so recovery owns the cleanup and must prove absence.
    resolved.root_marker_init.write_bytes(ROOT_MARKER_BYTES)

    owned = make_spool(tmp_path)
    try:
        assert owned.recover_existing() is RecoveryDisposition.ABSENT
    finally:
        owned.close()
    assert list(root.iterdir()) == []


@pytest.mark.parametrize(
    "partial_name",
    [
        pytest.param("sentinel", id="final_sentinel"),
        pytest.param("database", id="database"),
        pytest.param("database_journal", id="journal"),
        pytest.param("database_tmp", id="temporary_database_artifact"),
    ],
)
def test_task_5a_final_no_marker_partial_artifact_faults_without_mutation(
    tmp_path: Path, partial_name: str
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    partial = getattr(resolved, partial_name)
    payload = b"unowned-interrupted-artifact"
    partial.write_bytes(payload)

    owned = make_spool(tmp_path)
    try:
        assert owned.recover_existing() is RecoveryDisposition.FAULTED
    finally:
        owned.close()
    assert partial.read_bytes() == payload
    assert not resolved.root_marker.exists()
    assert not resolved.root_marker_init.exists()


@pytest.mark.parametrize("artifact_count", [0, 1, 6])
def test_task_5a_final_valid_sentinel_init_and_exact_debris_recover_to_absent(
    tmp_path: Path, artifact_count: int
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    resolved.root_marker.write_bytes(ROOT_MARKER_BYTES)
    resolved.sentinel_init.write_bytes(storage_security.initial_sentinel_image(STATE_ID))
    for artifact in resolved.deletable[:artifact_count]:
        artifact.write_bytes(b"exact-owned-debris")

    owned = make_spool(tmp_path)
    try:
        assert owned.recover_existing() is RecoveryDisposition.ABSENT
    finally:
        owned.close()
    assert resolved.root_marker.read_bytes() == ROOT_MARKER_BYTES
    assert [
        path.name for path in (*resolved.activation_temps, *resolved.deletable) if path.exists()
    ] == []


@pytest.mark.parametrize("blocker", ["unknown", "invalid_temp", "nonexclusive_temp"])
def test_task_5a_final_unownable_sentinel_initialization_remains_untouched_and_faults(
    tmp_path: Path, blocker: str
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    resolved.root_marker.write_bytes(ROOT_MARKER_BYTES)
    protected: dict[Path, bytes] = {resolved.root_marker: ROOT_MARKER_BYTES}
    if blocker == "unknown":
        unknown = root / "unowned.bin"
        unknown.write_bytes(b"unowned")
        protected[unknown] = b"unowned"
    elif blocker == "invalid_temp":
        resolved.sentinel_init.write_bytes(b"not-a-512-byte-sentinel")
        protected[resolved.sentinel_init] = b"not-a-512-byte-sentinel"
    else:
        # A directory at the temporary leaf cannot be exclusively opened as the
        # exact temporary file and must never be removed as a cleanup convenience.
        resolved.sentinel_init.mkdir()

    owned = make_spool(tmp_path)
    try:
        assert owned.recover_existing() is RecoveryDisposition.FAULTED
    finally:
        owned.close()
    for path, payload in protected.items():
        assert path.read_bytes() == payload
    if blocker == "nonexclusive_temp":
        assert resolved.sentinel_init.is_dir()


def test_task_5a_final_clear_live_store_purge_resets_same_owner_for_fresh_epoch(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        SentinelState,
        StoreDisposition,
    )

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        resolved = owned.resolved_manifest
        decoys = {"decoy.bin": b"keep", "capture-v1.sqlite3.bak": b"also-keep"}
        for name, payload in decoys.items():
            (tmp_path / "evidence" / name).write_bytes(payload)

        command = FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=event_uuid(940),
            sentinel_state=SentinelState.FULL_PURGE_PENDING,
            artifact_manifest_version=1,
        )
        assert owned.purge_full_store(command) is PurgeDisposition.PURGE_COMPLETED
        assert [path.name for path in resolved.deletable if path.exists()] == []
        assert storage_security.parse_root_marker(resolved.root_marker.read_bytes())
        assert (
            storage_security.decode_sentinel_image(resolved.sentinel.read_bytes()).active.state
            is SentinelState.CLEAR
        )
        assert owned.diagnostics().queue_record_count == 0
        assert owned.diagnostics().capture_state.value == "unavailable"
        for name, payload in decoys.items():
            assert (tmp_path / "evidence" / name).read_bytes() == payload

        fresh = build_create_epoch(
            installation_id=event_uuid(941),
            producer_instance_id=event_uuid(942),
            consent_epoch_id=event_uuid(943),
            logical_session_id=event_uuid(944),
            binding_id=event_uuid(945),
            opened_event_id=event_uuid(946),
            binding_event_id=event_uuid(947),
        )
        assert owned.create_epoch(fresh) is StoreDisposition.COMMITTED
        assert rows(
            owned, "SELECT installation_id, clock_high_water_utc FROM producer_installation"
        ) == [(event_uuid(941), "2026-08-08T00:00:01.000000Z")]
        assert rows(owned, "SELECT logical_session_id FROM evidence_sessions") == [
            (event_uuid(944),)
        ]
    finally:
        owned.close()


def test_task_5a_final_rollback_pending_same_owner_purge_clears_sticky_authority(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        SentinelState,
        StoreDisposition,
    )

    clock = StubClock()
    owned = make_spool(tmp_path, clock=clock)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        clock.override.append(START - timedelta(hours=1))
        assert (
            owned.append_record(ordinary_record(turn_opened_snapshot(3, event_index=950), 3))
            is StoreDisposition.FAULTED
        )
        # The retained real lease denies outside file reads while this owner is live;
        # inspect the same real leased image rather than manufacturing a test handle.
        active = storage_security.decode_sentinel_image(owned._read_sentinel()).active
        assert active.state is SentinelState.CLOCK_ROLLBACK_PURGE_PENDING
        assert active.state_generation_id is not None
        command = FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=active.state_generation_id,
            sentinel_state=SentinelState.CLOCK_ROLLBACK_PURGE_PENDING,
            artifact_manifest_version=1,
        )
        assert owned.purge_full_store(command) is PurgeDisposition.PURGE_COMPLETED
        assert [path.name for path in owned.resolved_manifest.deletable if path.exists()] == []
        assert owned.diagnostics().sticky_fault is None
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        owned.close()


def test_task_5a_final_purge_lease_and_bad_commands_leave_store_bytes_unchanged(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        SentinelState,
        StoreDisposition,
    )

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        resolved = first.resolved_manifest
        before = {
            path: path.read_bytes()
            for path in (resolved.root_marker, *resolved.deletable)
            if path.exists()
        }
        sentinel_before = first._read_sentinel()
        refused_command = FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=event_uuid(959),
            sentinel_state=SentinelState.CLOCK_ROLLBACK_PURGE_PENDING,
            artifact_manifest_version=1,
        )
        assert first.purge_full_store(refused_command) is PurgeDisposition.PURGE_FAILED
        contender = make_spool(tmp_path)
        try:
            held_command = FullPurgeV1(
                protocol_version=1,
                full_purge_generation_id=event_uuid(960),
                sentinel_state=SentinelState.FULL_PURGE_PENDING,
                artifact_manifest_version=1,
            )
            assert (
                contender.purge_full_store(held_command) is PurgeDisposition.OWNERSHIP_UNAVAILABLE
            )
        finally:
            contender.close()
        assert {path: path.read_bytes() for path in before} == before
        assert first._read_sentinel() == sentinel_before
    finally:
        first.close()

    # With no lease holder, make an exact pending image and prove malformed, wrong
    # state, and wrong generation requests cannot touch it or the database bytes.
    resolved = storage_security.MANIFEST_V1.resolve(tmp_path / "evidence")
    generation = event_uuid(961)
    resolved.sentinel.write_bytes(
        storage_security.next_sentinel_image(
            resolved.sentinel.read_bytes(),
            SentinelState.FULL_PURGE_PENDING,
            state_generation_id=generation,
        )
    )
    before = {
        path: path.read_bytes()
        for path in (*resolved.retained, *resolved.deletable)
        if path.exists()
    }
    owned = make_spool(tmp_path)
    try:
        wrong_state = FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=generation,
            sentinel_state=SentinelState.CLOCK_ROLLBACK_PURGE_PENDING,
            artifact_manifest_version=1,
        )
        wrong_generation = FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=event_uuid(962),
            sentinel_state=SentinelState.FULL_PURGE_PENDING,
            artifact_manifest_version=1,
        )
        for command in (object(), wrong_state, wrong_generation):
            assert owned.purge_full_store(command) is PurgeDisposition.PURGE_FAILED
            assert {path: path.read_bytes() for path in before} == before
    finally:
        owned.close()


def test_task_5a_mixed_activation_partial_faults_without_touching_any_owned_bytes(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    protected = {
        resolved.root_marker: ROOT_MARKER_BYTES,
        resolved.root_marker_init: ROOT_MARKER_BYTES,
        resolved.sentinel_init: storage_security.initial_sentinel_image(STATE_ID),
        resolved.database: b"partial-database",
        resolved.database_wal: b"partial-wal",
        resolved.database_tmp: b"partial-temporary-database",
    }
    for path, contents in protected.items():
        path.write_bytes(contents)

    owned = make_spool(tmp_path)
    try:
        assert owned.recover_existing() is RecoveryDisposition.FAULTED
    finally:
        owned.close()
    assert {path: path.read_bytes() for path in protected} == protected


def test_task_5a_pending_recovery_close_failure_is_typed_and_retains_the_lease(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, SentinelState, StoreDisposition

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()
    resolved = storage_security.MANIFEST_V1.resolve(tmp_path / "evidence")
    generation = event_uuid(970)
    resolved.sentinel.write_bytes(
        storage_security.next_sentinel_image(
            resolved.sentinel.read_bytes(),
            SentinelState.FULL_PURGE_PENDING,
            state_generation_id=generation,
        )
    )

    class CloseFailsOnceSentinelHandle(storage_security.SentinelHandleV1):
        instances: list[CloseFailsOnceSentinelHandle] = []

        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.fail_close = True
            self.instances.append(self)

        def close(self) -> None:
            if self.fail_close:
                self.fail_close = False
                raise OSError("injected close failure")
            super().close()

    resumed = make_spool(tmp_path, sentinel_opener=CloseFailsOnceSentinelHandle)
    try:
        assert resumed.recover_existing() is RecoveryDisposition.FAULTED
        assert resumed._sentinel_handle is CloseFailsOnceSentinelHandle.instances[-1]
    finally:
        if resumed._sentinel_handle is None:
            CloseFailsOnceSentinelHandle.instances[-1].close()
        else:
            resumed.close()


def test_task_5a_same_owner_purge_close_failure_is_typed_and_retryable(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        SentinelState,
        StoreDisposition,
    )

    class CloseFailsOnceSentinelHandle(storage_security.SentinelHandleV1):
        instances: list[CloseFailsOnceSentinelHandle] = []

        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.fail_close = True
            self.instances.append(self)

        def close(self) -> None:
            if self.fail_close:
                self.fail_close = False
                raise OSError("injected close failure")
            super().close()

    owned = make_spool(tmp_path, sentinel_opener=CloseFailsOnceSentinelHandle)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert (
            owned.purge_full_store(
                FullPurgeV1(
                    protocol_version=1,
                    full_purge_generation_id=event_uuid(971),
                    sentinel_state=SentinelState.FULL_PURGE_PENDING,
                    artifact_manifest_version=1,
                )
            )
            is PurgeDisposition.PURGE_FAILED
        )
        assert owned._sentinel_handle is CloseFailsOnceSentinelHandle.instances[-1]
    finally:
        if owned._sentinel_handle is None:
            CloseFailsOnceSentinelHandle.instances[-1].close()
        else:
            owned.close()


def test_task_5a_live_purge_closes_sqlite_before_each_exact_manifest_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        SentinelState,
        StoreDisposition,
    )

    events: list[str] = []

    class RecordingConnection(sqlite3.Connection):
        def close(self) -> None:
            events.append("sqlite-close")
            super().close()

    owned = make_spool(tmp_path, connection_factory=RecordingConnection)
    try:
        assert owned.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        resolved = owned.resolved_manifest
        unlink = Path.unlink

        def record_unlink(path: Path, *args: object, **kwargs: object) -> None:
            if path in resolved.deletable:
                events.append(path.name)
            unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", record_unlink)
        assert (
            owned.purge_full_store(
                FullPurgeV1(
                    protocol_version=1,
                    full_purge_generation_id=event_uuid(972),
                    sentinel_state=SentinelState.FULL_PURGE_PENDING,
                    artifact_manifest_version=1,
                )
            )
            is PurgeDisposition.PURGE_COMPLETED
        )
    finally:
        owned.close()
    assert events == ["sqlite-close", *(path.name for path in resolved.deletable)]


def test_task_5a_pending_recovery_never_opens_mutable_sqlite_before_cleanup(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, SentinelState, StoreDisposition

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()
    resolved = storage_security.MANIFEST_V1.resolve(tmp_path / "evidence")
    generation = event_uuid(973)
    resolved.sentinel.write_bytes(
        storage_security.next_sentinel_image(
            resolved.sentinel.read_bytes(),
            SentinelState.FULL_PURGE_PENDING,
            state_generation_id=generation,
        )
    )

    def mutable_sqlite_factory(*args: object, **kwargs: object) -> sqlite3.Connection:
        raise AssertionError("pending recovery must not open SQLite")

    resumed = make_spool(tmp_path, connection_factory=mutable_sqlite_factory)
    try:
        assert resumed.recover_existing() is RecoveryDisposition.PURGE_COMPLETED
    finally:
        resumed.close()


def test_task_5a_root_marker_temporary_close_failure_is_retryable(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    class CloseFailsOnce(storage_security.SentinelHandleV1):
        failures = 1
        instances: list[CloseFailsOnce] = []

        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.instances.append(self)

        def close(self) -> None:
            if type(self).failures:
                type(self).failures -= 1
                raise OSError("injected close failure")
            super().close()

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    temporary = resolved.root_marker_init
    temporary.write_bytes(ROOT_MARKER_BYTES)
    owned = make_spool(tmp_path, sentinel_opener=CloseFailsOnce)
    try:
        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        assert owned._sentinel_handle is CloseFailsOnce.instances[-1]
        assert owned._sentinel_handle.read_image() == ROOT_MARKER_BYTES
        assert owned.recover_existing() is RecoveryDisposition.ABSENT
        assert not temporary.exists()
    finally:
        if owned._sentinel_handle is None:
            for handle in CloseFailsOnce.instances:
                handle.close()
        else:
            owned.close()


def test_task_5a_sentinel_temporary_close_failure_preserves_exact_debris_for_retry(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    class CloseFailsOnce(storage_security.SentinelHandleV1):
        failures = 1
        instances: list[CloseFailsOnce] = []

        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.instances.append(self)

        def close(self) -> None:
            if type(self).failures:
                type(self).failures -= 1
                raise OSError("injected close failure")
            super().close()

    root = tmp_path / "evidence"
    root.mkdir()
    resolved = storage_security.MANIFEST_V1.resolve(root)
    resolved.root_marker.write_bytes(ROOT_MARKER_BYTES)
    temporary = resolved.sentinel_init
    temporary.write_bytes(storage_security.initial_sentinel_image(STATE_ID))
    debris = {path: b"exact-owned-debris" for path in resolved.deletable[::5]}
    for path, payload in debris.items():
        path.write_bytes(payload)
    owned = make_spool(tmp_path, sentinel_opener=CloseFailsOnce)
    try:
        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        assert owned._sentinel_handle is CloseFailsOnce.instances[-1]
        temporary_image = storage_security.initial_sentinel_image(STATE_ID)
        assert owned._sentinel_handle.read_image() == temporary_image
        assert {path: path.read_bytes() for path in debris} == debris
        assert owned.recover_existing() is RecoveryDisposition.ABSENT
        assert not temporary.exists()
        assert not any(path.exists() for path in debris)
    finally:
        if owned._sentinel_handle is None:
            for handle in CloseFailsOnce.instances:
                handle.close()
        else:
            owned.close()


def test_task_5a_final_root_marker_read_oserror_is_contained_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()
    resolved = storage_security.MANIFEST_V1.resolve(tmp_path / "evidence")
    before = {
        path: path.read_bytes()
        for path in (*resolved.retained, *resolved.deletable)
        if path.exists()
    }
    read_bytes = Path.read_bytes

    def fail_final_marker(path: Path) -> bytes:
        if path == resolved.root_marker:
            raise OSError("injected root marker read failure")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_final_marker)
    owned = make_spool(tmp_path)
    try:
        assert owned.recover_existing() is RecoveryDisposition.FAULTED
    finally:
        owned.close()
    assert {path: read_bytes(path) for path in before} == before


def test_task_5a_final_sentinel_read_oserror_is_contained_and_retryable(tmp_path: Path) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    first = make_spool(tmp_path)
    try:
        assert first.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
    finally:
        first.close()

    class ReadFailsOnce(storage_security.SentinelHandleV1):
        failures = 1
        instances: list[ReadFailsOnce] = []

        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.instances.append(self)

        def read_image(self) -> bytes:
            if type(self).failures:
                type(self).failures -= 1
                raise OSError("injected sentinel read failure")
            return super().read_image()

    owned = make_spool(tmp_path, sentinel_opener=ReadFailsOnce)
    try:
        assert owned.recover_existing() is RecoveryDisposition.FAULTED
        assert owned._sentinel_handle is ReadFailsOnce.instances[-1]
        assert owned.recover_existing() is RecoveryDisposition.RECOVERED
    finally:
        owned.close()


# --------------------------------------------------------------------------------------
# The one daemon writer that owns the synchronous spool.
# --------------------------------------------------------------------------------------


class RecordingSpool:
    """A real spool subclass that records which thread executed each operation."""

    instances: list[RecordingSpool] = []
    gate: threading.Event | None = None
    crash_on: str | None = None

    def __new__(cls, *args: object, **kwargs: object) -> RecordingSpool:
        from hermes_realtime.evidence import sqlite_spool

        if cls is RecordingSpool:
            cls = type("_RecordingSpool", (RecordingSpool, sqlite_spool.SQLiteEvidenceSpool), {})
        return cast("RecordingSpool", super().__new__(cls))

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[call-arg]
        self.calls: list[tuple[str, int]] = []
        self.closes = 0
        self._live = 0
        self.peak = 0
        self._lock = threading.Lock()
        RecordingSpool.instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.gate = None
        cls.crash_on = None

    def _enter(self, name: str) -> None:
        with self._lock:
            self.calls.append((name, threading.get_ident()))
            self._live += 1
            self.peak = max(self.peak, self._live)
        if RecordingSpool.gate is not None:
            RecordingSpool.gate.wait(10)
        if RecordingSpool.crash_on == name:
            with self._lock:
                self._live -= 1
            raise RuntimeError("injected worker crash")

    def _exit(self) -> None:
        with self._lock:
            self._live -= 1

    def create_epoch(self, command):  # type: ignore[no-untyped-def]
        self._enter("create_epoch")
        try:
            return super().create_epoch(command)  # type: ignore[misc]
        finally:
            self._exit()

    def append_record(self, item):  # type: ignore[no-untyped-def]
        self._enter("append_record")
        try:
            return super().append_record(item)  # type: ignore[misc]
        finally:
            self._exit()

    def diagnostics(self):  # type: ignore[no-untyped-def]
        self._enter("diagnostics")
        try:
            return super().diagnostics()  # type: ignore[misc]
        finally:
            self._exit()

    def drain_and_close(self, command):  # type: ignore[no-untyped-def]
        self._enter("drain_and_close")
        try:
            return super().drain_and_close(command)  # type: ignore[misc]
        finally:
            self._exit()

    def close(self) -> None:
        self.closes += 1
        super().close()  # type: ignore[misc]


def make_daemon(tmp_path: Path, *, factory=None):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import sqlite_spool

    root = tmp_path / "evidence"
    root.mkdir(parents=True, exist_ok=True)

    def default():  # type: ignore[no-untyped-def]
        return RecordingSpool(
            root / "capture-v1.sqlite3",
            clock=StubClock(),
            uuid_factory=StubUuids(),
            probe=InjectedStorageProbe(),
        )

    return sqlite_spool.SQLiteEvidenceWriterDaemonV1(factory or default)


def stop_command():  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import DrainAndStopV1

    return DrainAndStopV1(protocol_version=1, owner_generation=1, final_admission_ordinal=3)


def test_writer_daemon_conforms_to_the_transport_protocol() -> None:
    from hermes_realtime.evidence import sqlite_spool, transport

    daemon = sqlite_spool.SQLiteEvidenceWriterDaemonV1
    assert daemon.protocol_version == 1
    for name, parameters, _result, _argument in EXPECTED_TRANSPORT_SIGNATURES:
        concrete = getattr(daemon, name)
        assert tuple(inspect.signature(concrete).parameters) == ("self", *parameters), name
        expected = get_type_hints(getattr(transport.EvidenceWriterTransportV1, name))
        assert get_type_hints(concrete) == expected, name


def test_writer_daemon_one_shot_recovery_closes_on_owner_thread_before_return() -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import RecoveryDisposition

    calls: list[tuple[str, int]] = []

    class RecoverySpool:
        owner_generation = 7

        def recover_existing(self) -> RecoveryDisposition:
            calls.append(("recover_existing", threading.get_ident()))
            return RecoveryDisposition.ABSENT

        def close(self) -> None:
            calls.append(("close", threading.get_ident()))

    daemon = sqlite_spool.SQLiteEvidenceWriterDaemonV1(RecoverySpool)  # type: ignore[arg-type]

    assert daemon.recover_existing_and_close() is RecoveryDisposition.ABSENT
    assert daemon.join(1) is True
    assert daemon.is_running is False
    assert [name for name, _ident in calls] == ["recover_existing", "close"]
    assert {ident for _name, ident in calls} == {daemon.owner_thread_id}
    assert daemon.owner_thread_id != threading.get_ident()
    assert daemon.recover_existing() is RecoveryDisposition.FAULTED


def test_writer_daemon_one_shot_purge_closes_on_owner_thread_before_return() -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import (
        FullPurgeV1,
        PurgeDisposition,
        SentinelState,
    )

    calls: list[tuple[str, int]] = []
    command = FullPurgeV1(
        protocol_version=1,
        full_purge_generation_id="00000000-0000-4000-8000-000000000902",
        sentinel_state=SentinelState.FULL_PURGE_PENDING,
        artifact_manifest_version=1,
    )

    class PurgeSpool:
        owner_generation = 8

        def purge_full_store(self, supplied: FullPurgeV1) -> PurgeDisposition:
            assert supplied is command
            calls.append(("purge_full_store", threading.get_ident()))
            return PurgeDisposition.ALREADY_ABSENT

        def close(self) -> None:
            calls.append(("close", threading.get_ident()))

    daemon = sqlite_spool.SQLiteEvidenceWriterDaemonV1(PurgeSpool)  # type: ignore[arg-type]

    assert daemon.purge_full_store_and_close(command) is PurgeDisposition.ALREADY_ABSENT
    assert daemon.join(1) is True
    assert daemon.is_running is False
    assert [name for name, _ident in calls] == ["purge_full_store", "close"]
    assert {ident for _name, ident in calls} == {daemon.owner_thread_id}
    assert daemon.owner_thread_id != threading.get_ident()
    assert daemon.purge_full_store(command) is PurgeDisposition.PURGE_FAILED


def test_writer_daemon_runs_every_operation_on_one_owner_thread(tmp_path: Path) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import DrainDisposition, StoreDisposition

    RecordingSpool.reset()
    before = {thread.name for thread in threading.enumerate()}
    daemon = make_daemon(tmp_path)
    try:
        assert daemon.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        assert daemon.append_record(
            ordinary_record(turn_opened_snapshot(3, event_index=10), 3)
        ) is StoreDisposition.COMMITTED
        assert daemon.diagnostics().protocol_version == 1

        writers = [
            thread
            for thread in threading.enumerate()
            if thread.name == sqlite_spool.SQLiteEvidenceWriterDaemonV1.THREAD_NAME
        ]
        assert len(writers) == 1
        assert writers[0].daemon is True
        assert writers[0].name not in before
        assert writers[0].ident == daemon.owner_thread_id
        assert daemon.owner_thread_id != threading.get_ident()

        assert daemon.drain_and_close(stop_command()) is DrainDisposition.STOPPED
    finally:
        RecordingSpool.reset()

    spool = daemon.spool_for_test
    assert {ident for _name, ident in spool.calls} == {daemon.owner_thread_id}
    assert [name for name, _ident in spool.calls] == [
        "create_epoch",
        "append_record",
        "diagnostics",
        "drain_and_close",
    ]


def test_writer_daemon_owns_exactly_one_spool_and_connection(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    try:
        assert daemon.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        daemon.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
        daemon.diagnostics()
        assert len(RecordingSpool.instances) == 1
        spool = RecordingSpool.instances[0]
        assert daemon.spool_for_test is spool
        assert id(spool.connection) == id(spool.connection)
    finally:
        daemon.drain_and_close(stop_command())
        RecordingSpool.reset()


def test_writer_daemon_preserves_fifo_and_never_executes_concurrently(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import StoreDisposition

    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    entered = threading.Event()
    results: dict[str, object] = {}
    try:
        assert daemon.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        spool = daemon.spool_for_test
        gate = threading.Event()
        RecordingSpool.gate = gate

        def first() -> None:
            results["first"] = daemon.append_record(
                ordinary_record(turn_opened_snapshot(3, event_index=10), 3)
            )

        def second() -> None:
            entered.wait(10)
            results["second"] = daemon.diagnostics()

        one = threading.Thread(target=first)
        two = threading.Thread(target=second)
        one.start()
        while len(spool.calls) < 2:
            entered.wait(0.01)
        entered.set()
        two.start()
        # The blocked operation still holds the only worker, so nothing else ran.
        assert len(spool.calls) == 2
        gate.set()
        one.join(10)
        two.join(10)
        assert results["first"] is StoreDisposition.COMMITTED
        assert spool.peak == 1
        assert [name for name, _ident in spool.calls] == [
            "create_epoch",
            "append_record",
            "diagnostics",
        ]
    finally:
        RecordingSpool.gate = None
        daemon.drain_and_close(stop_command())
        RecordingSpool.reset()


def test_writer_daemon_drain_finishes_prior_work_then_exits_cleanly(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import (
        DrainDisposition,
        OwnerState,
        StoreDisposition,
    )

    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    try:
        daemon.create_epoch(make_create_epoch())
        daemon.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
        spool = daemon.spool_for_test
        assert daemon.drain_and_close(stop_command()) is DrainDisposition.STOPPED
        assert daemon.join(10) is True
        assert daemon.is_running is False
        assert spool.closes >= 1
        assert [name for name, _ident in spool.calls] == [
            "create_epoch",
            "append_record",
            "drain_and_close",
        ]

        # Repeated drain and every later call are idempotent, fail closed, and never block.
        assert daemon.drain_and_close(stop_command()) is DrainDisposition.STOPPED
        assert daemon.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
        assert daemon.append_record(
            ordinary_record(turn_opened_snapshot(4, event_index=11), 4)
        ) is StoreDisposition.FAULTED
        assert daemon.diagnostics().owner_state is OwnerState.STOPPED
    finally:
        RecordingSpool.reset()


@pytest.mark.parametrize(
    "foreign",
    [
        pytest.param("generation", id="wrong_owner_generation"),
        pytest.param("type", id="wrong_command_type"),
    ],
)
def test_writer_daemon_foreign_drain_does_not_stop_the_owner(
    tmp_path: Path,
    foreign: str,
) -> None:
    from hermes_realtime.evidence.models import (
        DrainAndStopV1,
        DrainDisposition,
        OwnerState,
        StoreDisposition,
    )

    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    try:
        assert daemon.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        spool = daemon.spool_for_test
        command: object = (
            DrainAndStopV1(protocol_version=1, owner_generation=7, final_admission_ordinal=3)
            if foreign == "generation"
            else object()
        )
        assert daemon.drain_and_close(command) is DrainDisposition.ALREADY_QUEUED

        # The owner keeps every capability it held before the foreign request.
        assert daemon.is_running is True
        assert daemon.spool_for_test is spool
        assert spool.closes == 0
        assert daemon.diagnostics().owner_state is OwnerState.RUNNING
        assert daemon.append_record(
            ordinary_record(turn_opened_snapshot(3, event_index=10), 3)
        ) is StoreDisposition.COMMITTED
        assert daemon.pending_ordinary_count == 0
        assert daemon.drain_is_pending is False

        # Only the owner's exact command actually stops it.
        assert daemon.drain_and_close(stop_command()) is DrainDisposition.STOPPED
        assert daemon.join(10) is True
        assert daemon.is_running is False
        assert spool.closes >= 1
    finally:
        RecordingSpool.reset()


def test_writer_daemon_reserves_a_drain_lane_under_ordinary_saturation(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import DrainDisposition, StoreDisposition

    capacity = sqlite_spool.SQLiteEvidenceWriterDaemonV1.ORDINARY_CAPACITY
    assert capacity == 64

    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    gate = threading.Event()
    ordinary_results: list[object] = []
    drain_observed: list[tuple[object, bool]] = []
    lock = threading.Lock()
    try:
        assert daemon.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        spool = daemon.spool_for_test
        RecordingSpool.gate = gate

        def submit(index: int) -> None:
            result = daemon.append_record(
                ordinary_record(turn_opened_snapshot(3, event_index=200 + index), 3)
            )
            with lock:
                ordinary_results.append(result)

        workers = [threading.Thread(target=submit, args=(index,)) for index in range(capacity)]
        for worker in workers:
            worker.start()
        while daemon.pending_ordinary_count < capacity:
            gate.wait(0.01)

        # Every ordinary slot is taken, so a further ordinary job is refused outright.
        assert daemon.pending_ordinary_count == capacity
        assert daemon.append_record(
            ordinary_record(turn_opened_snapshot(3, event_index=999), 3)
        ) is StoreDisposition.FAULTED
        assert daemon.pending_ordinary_count == capacity

        # The reserved lane still admits one valid drain behind all accepted work.
        def drain() -> None:
            result = daemon.drain_and_close(stop_command())
            drain_observed.append((result, daemon.is_running))

        closer = threading.Thread(target=drain)
        closer.start()
        while not daemon.drain_is_pending:
            gate.wait(0.01)
        # A concurrent foreign drain cannot cancel or replace the accepted one.
        assert daemon.drain_and_close(object()) is DrainDisposition.ALREADY_QUEUED
        # Ordinary work is refused without enqueuing once the drain is accepted.
        assert daemon.append_record(
            ordinary_record(turn_opened_snapshot(3, event_index=998), 3)
        ) is StoreDisposition.FAULTED

        gate.set()
        for worker in workers:
            worker.join(30)
        closer.join(30)

        assert len(ordinary_results) == capacity
        executed = [name for name, _ident in spool.calls]
        assert executed.count("append_record") == capacity
        assert executed[-1] == "drain_and_close"
        assert executed.index("drain_and_close") == len(executed) - 1
        assert drain_observed == [(DrainDisposition.STOPPED, False)]
        assert daemon.join(10) is True
        assert spool.peak == 1
    finally:
        RecordingSpool.gate = None
        gate.set()
        RecordingSpool.reset()


def test_writer_daemon_rejects_a_foreign_drain_without_seizing_the_lane(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import (
        DrainAndStopV1,
        DrainDisposition,
        StoreDisposition,
    )

    capacity = sqlite_spool.SQLiteEvidenceWriterDaemonV1.ORDINARY_CAPACITY
    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    gate = threading.Event()
    foreign_results: list[object] = []
    accepted: list[object] = []
    lock = threading.Lock()
    try:
        assert daemon.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        spool = daemon.spool_for_test
        assert daemon.owner_generation == spool.owner_generation
        RecordingSpool.gate = gate

        def submit(index: int) -> None:
            result = daemon.append_record(
                ordinary_record(turn_opened_snapshot(3, event_index=400 + index), 3)
            )
            with lock:
                accepted.append(result)

        workers = [
            threading.Thread(target=submit, args=(index,)) for index in range(capacity - 1)
        ]
        for worker in workers:
            worker.start()
        while daemon.pending_ordinary_count < capacity - 1:
            gate.wait(0.01)

        # Foreign and wrong-type drains are answered without any owner-thread work.
        def foreign(command: object) -> None:
            foreign_results.append(daemon.drain_and_close(command))

        for command in (
            DrainAndStopV1(protocol_version=1, owner_generation=9, final_admission_ordinal=3),
            object(),
        ):
            caller = threading.Thread(target=foreign, args=(command,))
            caller.start()
            caller.join(10)
            assert caller.is_alive() is False, "a foreign drain waited on the blocked worker"

        assert foreign_results == [DrainDisposition.ALREADY_QUEUED] * 2
        assert daemon.drain_is_pending is False
        assert daemon.pending_ordinary_count == capacity - 1
        executed = [name for name, _ident in spool.calls]
        assert "drain_and_close" not in executed

        # Ordinary work is still accepted because no valid drain has reserved yet.
        last = threading.Thread(target=submit, args=(capacity - 1,))
        last.start()
        while daemon.pending_ordinary_count < capacity:
            gate.wait(0.01)
        assert daemon.pending_ordinary_count == capacity

        # The owner's exact command can still reserve the lane behind all of it.
        def close() -> None:
            foreign_results.append(daemon.drain_and_close(stop_command()))

        closer = threading.Thread(target=close)
        closer.start()
        while not daemon.drain_is_pending:
            gate.wait(0.01)
        assert daemon.drain_is_pending is True

        gate.set()
        for worker in (*workers, last, closer):
            worker.join(30)
        assert foreign_results[-1] is DrainDisposition.STOPPED
        assert len(accepted) == capacity
        assert daemon.join(10) is True
    finally:
        RecordingSpool.gate = None
        gate.set()
        RecordingSpool.reset()


def test_writer_daemon_first_diagnostics_after_factory_failure_is_faulted(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import CaptureState, OwnerState, WriterFault

    def refuse():  # type: ignore[no-untyped-def]
        raise OSError("the store cannot be opened")

    daemon = make_daemon(tmp_path, factory=refuse)
    try:
        first = daemon.diagnostics()
        assert first.owner_state is OwnerState.FAULTED
        assert first.capture_state is CaptureState.FAULTED
        assert first.sticky_fault is WriterFault.STORE_CORRUPT
        assert daemon.diagnostics() == first
    finally:
        daemon.drain_and_close(stop_command())
        assert daemon.join(10) is True


def test_seal_shape_mismatch_rejects_without_mutation_or_sticky_fault(spool) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence.models import StoreDisposition

    spool.create_epoch(make_create_epoch())
    spool.append_binding_close(make_binding_close_command())

    # Corrupt the authoritative count so the caller's exact typed command no longer
    # matches event_count + 1 while the preceding binding close still sits at final-1.
    spool.connection.execute(
        "INSERT INTO evidence_events (event_id, logical_session_id, event_sequence, event_kind,"
        " recorded_at_utc, canonical_payload, payload_hash, previous_hash, record_hash,"
        " canonical_bytes) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            event_uuid(500),
            SESSION_ID,
            4,
            "turn_opened",
            UTC_TEXT,
            "{}",
            HEX64,
            HEX64,
            HEX64,
            2,
        ),
    )
    spool.connection.execute("UPDATE evidence_sessions SET event_count=4")
    spool.connection.commit()

    before = rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0]
    session_before = rows(spool, "SELECT state, taint_code, event_count FROM evidence_sessions")

    assert spool.seal_epoch(make_seal_command()) is StoreDisposition.REJECTED_STATE

    assert rows(spool, "SELECT COUNT(*) FROM evidence_events")[0][0] == before
    assert rows(spool, "SELECT state, taint_code, event_count FROM evidence_sessions") == (
        session_before
    )
    assert rows(spool, "SELECT COUNT(*) FROM evidence_conflicts")[0][0] == 0
    assert rows(spool, "SELECT state FROM consent_epochs") == [("active",)]
    assert spool.diagnostics().sticky_fault is None


def test_writer_daemon_join_before_first_submit_reports_not_running(tmp_path: Path) -> None:
    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    try:
        assert daemon.join(0) is True
        assert daemon.join() is True
        assert daemon.is_running is False
        assert daemon.owner_thread_id is None
        assert daemon.pending_ordinary_count == 0
        assert daemon.drain_is_pending is False
    finally:
        RecordingSpool.reset()


def test_writer_daemon_signals_every_waiter_when_a_job_crashes(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import DrainDisposition, StoreDisposition

    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    gate = threading.Event()
    results: list[object] = []
    lock = threading.Lock()
    try:
        daemon.create_epoch(make_create_epoch())
        RecordingSpool.gate = gate
        RecordingSpool.crash_on = "append_record"

        def submit(index: int) -> None:
            result = daemon.append_record(
                ordinary_record(turn_opened_snapshot(3, event_index=300 + index), 3)
            )
            with lock:
                results.append(result)

        waiters = [threading.Thread(target=submit, args=(index,)) for index in range(6)]
        for waiter in waiters:
            waiter.start()
        while daemon.pending_ordinary_count < 1:
            gate.wait(0.01)
        gate.set()
        for waiter in waiters:
            waiter.join(30)

        # Every waiter is signaled with a typed refusal; none deadlocks.
        assert len(results) == 6
        assert set(results) == {StoreDisposition.FAULTED}
        RecordingSpool.crash_on = None
        assert daemon.drain_and_close(stop_command()) is DrainDisposition.STOPPED
        assert daemon.join(10) is True
    finally:
        RecordingSpool.gate = None
        RecordingSpool.crash_on = None
        gate.set()
        RecordingSpool.reset()


def test_writer_daemon_isolates_unexpected_worker_exceptions(tmp_path: Path) -> None:
    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    RecordingSpool.reset()
    daemon = make_daemon(tmp_path)
    try:
        daemon.create_epoch(make_create_epoch())
        RecordingSpool.crash_on = "append_record"
        result = daemon.append_record(ordinary_record(turn_opened_snapshot(3, event_index=10), 3))
        assert result is StoreDisposition.FAULTED
        assert daemon.is_running is True

        diagnostics = daemon.diagnostics()
        assert diagnostics.sticky_fault is WriterFault.STORE_CORRUPT
        for value in dataclasses.asdict(diagnostics).values():
            assert "crash" not in str(value)
        RecordingSpool.crash_on = None
        assert daemon.append_record(
            ordinary_record(turn_opened_snapshot(3, event_index=10), 3)
        ) is StoreDisposition.FAULTED
    finally:
        daemon.drain_and_close(stop_command())
        RecordingSpool.reset()


def test_writer_daemon_leaks_no_thread_when_the_spool_cannot_be_created(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.models import (
        DrainDisposition,
        OwnerState,
        RecoveryDisposition,
        StoreDisposition,
        WriterFault,
    )

    def refuse():  # type: ignore[no-untyped-def]
        raise OSError("the store cannot be opened")

    daemon = make_daemon(tmp_path, factory=refuse)
    assert daemon.create_epoch(make_create_epoch()) is StoreDisposition.FAULTED
    assert daemon.recover_existing() is RecoveryDisposition.FAULTED
    diagnostics = daemon.diagnostics()
    assert diagnostics.owner_state is OwnerState.FAULTED
    assert diagnostics.sticky_fault is WriterFault.STORE_CORRUPT
    assert daemon.drain_and_close(stop_command()) is DrainDisposition.STOPPED
    assert daemon.join(10) is True
    assert [
        thread
        for thread in threading.enumerate()
        if thread.name == sqlite_spool.SQLiteEvidenceWriterDaemonV1.THREAD_NAME
    ] == []


TASK_5D_OLDER_EPOCH_ID = "10000000-0000-4000-8000-000000000003"
TASK_5D_OLDER_SESSION_ID = "10000000-0000-4000-8000-000000000004"


def test_task_5d_crash_worker_exposes_the_exact_closed_failpoint_matrix() -> None:
    import importlib.util

    worker_path = Path(__file__).with_name("spool_crash_worker.py")
    spec = importlib.util.spec_from_file_location("spool_crash_worker", worker_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    expected = {
        "before_begin",
        "after_event_insert_before_commit",
        "after_event_commit",
        "before_seal_commit",
        "after_seal_commit_before_ack",
        "after_revoke_request_commit",
        "after_logical_purge_before_vacuum",
        "after_vacuum_before_ack",
        "after_drain_ack_before_exit",
        "after_root_marker_init_create",
        "after_root_marker_init_partial_write",
        "after_root_marker_init_full_write",
        "after_root_marker_init_flush",
        "after_root_marker_activation",
        "after_sentinel_init_create",
        "after_sentinel_init_partial_write",
        "after_sentinel_init_full_write",
        "after_sentinel_init_flush",
        "after_sentinel_activation",
        "after_first_db_create",
        "after_first_schema_commit",
        "after_first_epoch_commit",
        "after_first_marker_clear",
        "after_recreate_pending_fsync",
        "after_recreate_db_create",
        "after_recreate_schema_commit",
        "after_recreate_epoch_commit",
        "after_recreate_marker_clear",
        "before_rollback_sentinel_write",
        "after_rollback_sentinel_fsync",
        "before_rollback_db_latch",
        "after_rollback_db_latch",
        "after_full_purge_marker_fsync",
        "after_full_purge_db_delete",
        "after_full_purge_journal_delete",
        "after_full_purge_wal_delete",
        "after_full_purge_shm_delete",
        "after_full_purge_vacuum_delete",
        "after_full_purge_tmp_delete",
        "after_full_purge_absence_verify",
        "after_full_purge_marker_clear",
    }
    assert set(worker.FAILPOINTS) == expected
    assert len(worker.FAILPOINTS) == len(expected) == 41
    assert worker.EXIT_MODES == (197, 198)
    assert worker.case_ids() == tuple(
        f"{failpoint}@exit{exit_mode}"
        for failpoint in worker.FAILPOINTS
        for exit_mode in worker.EXIT_MODES
    )
    assert len(worker.case_ids()) == 82


def task_5d_worker_launch_environment() -> tuple[Path, dict[str, str]]:
    """Return the exact isolated CPython 3.11 launch boundary for crash children."""

    assert sys.version_info[:2] == (3, 11)
    executable = Path(sys.executable).resolve()
    assert executable.is_file()
    allowed = ("COMSPEC", "PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "WINDIR")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    for forbidden in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        assert forbidden not in environment
    return executable, environment


def _task_5d_file_allows_delete_access(path: Path) -> bool:
    """Return whether no live handle denies delete sharing for one worker artifact."""

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        str(path),
        0x00010000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x00000080,
        None,
    )
    if handle in (None, wintypes.HANDLE(-1).value):
        error = ctypes.get_last_error()
        if error == 32:
            return False
        raise ctypes.WinError(error)
    if not close_handle(handle):
        raise ctypes.WinError(ctypes.get_last_error())
    return True


def _await_task_5d_child_handle_release(case_root: Path) -> None:
    """Bound Windows' short post-termination delete-share release interval."""

    import time

    deadline = time.monotonic() + 1.0
    while True:
        if all(
            _task_5d_file_allows_delete_access(path)
            for path in (case_root / "evidence").iterdir()
            if path.is_file()
        ):
            return
        if time.monotonic() >= deadline:
            pytest.fail(
                "Task 5D child file handles still denied delete sharing after process termination",
                pytrace=False,
            )
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 delete-share contract")
def test_task_5d_child_release_probe_requires_delete_share(tmp_path: Path) -> None:
    import ctypes
    import threading
    import time
    from ctypes import wintypes

    case_root = tmp_path / "case"
    evidence_root = case_root / "evidence"
    evidence_root.mkdir(parents=True)
    artifact = evidence_root / "marker.init"
    artifact.write_bytes(b"marker")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    blocker = create_file(
        str(artifact),
        0x80000000,
        0x00000001 | 0x00000002,
        None,
        3,
        0x00000080,
        None,
    )
    assert blocker not in (None, wintypes.HANDLE(-1).value)
    close_results: list[bool] = []

    def release_blocker() -> None:
        time.sleep(0.2)
        close_results.append(bool(close_handle(blocker)))

    release = threading.Thread(target=release_blocker, name="task-5d-delete-share-release")
    release.start()
    started = time.monotonic()
    try:
        _await_task_5d_child_handle_release(case_root)
        elapsed = time.monotonic() - started
    finally:
        release.join(timeout=2.0)
    assert close_results == [True]
    assert elapsed >= 0.15


_TASK_5D_READY_TIMEOUT_SECONDS = 30.0


def _await_task_5d_ready_line(ready_lines: queue.Queue[str]) -> str:
    """Wait for crash-worker setup without borrowing protocol-phase deadlines."""

    return ready_lines.get(timeout=_TASK_5D_READY_TIMEOUT_SECONDS)


def test_task_5d_ready_wait_has_a_separate_bounded_startup_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_lines: queue.Queue[str] = queue.Queue(maxsize=1)
    observed_timeouts: list[float | None] = []

    def simulate_loaded_worker(*, timeout: float | None = None) -> str:
        observed_timeouts.append(timeout)
        if timeout is None or timeout <= 10.0:
            raise queue.Empty
        return '{"state":"READY"}\n'

    monkeypatch.setattr(ready_lines, "get", simulate_loaded_worker)

    assert _await_task_5d_ready_line(ready_lines) == '{"state":"READY"}\n'
    assert observed_timeouts == [30.0]


def run_task_5d_worker(case_root: Path, failpoint: str, exit_mode: int) -> dict[str, object]:
    import json
    import subprocess
    import threading

    if sys.platform != "win32":
        pytest.skip("Task 5D requires Windows TerminateProcess semantics")
    worker = Path(__file__).with_name("spool_crash_worker.py")
    executable, environment = task_5d_worker_launch_environment()
    process = subprocess.Popen(
        [
            str(executable),
            "-I",
            str(worker),
            "--case-root",
            str(case_root),
            "--failpoint",
            failpoint,
            "--exit-mode",
            str(exit_mode),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    try:
        assert process.stdout is not None
        ready_lines: queue.Queue[str] = queue.Queue(maxsize=1)
        threading.Thread(
            target=lambda: ready_lines.put(process.stdout.readline()),
            name=f"task-5d-ready-{process.pid}",
            daemon=True,
        ).start()
        try:
            ready_line = _await_task_5d_ready_line(ready_lines)
        except queue.Empty:
            process.kill()
            process.wait(timeout=10)
            pytest.fail(
                f"worker did not reach {failpoint!r} before its bounded deadline",
                pytrace=False,
            )
        if not ready_line:
            pytest.fail(
                f"worker exited before the {failpoint!r} protocol frame",
                pytrace=False,
            )
        ready = json.loads(ready_line)
        assert type(ready.get("pid")) is int and ready["pid"] > 0
        del ready["pid"]
        expected_ready: dict[str, object] = {
            "caseId": f"{failpoint}@exit{exit_mode}",
            "importProvenance": "candidate-source",
            "protocolVersion": 1,
            "state": "READY",
        }
        if failpoint == "after_vacuum_before_ack":
            # The worker can emit READY only after the real sqlite3 VACUUM call
            # returned, never merely after it was selected for execution.
            expected_ready["vacuumReturned"] = True
        assert ready == expected_ready
        if exit_mode == 198:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            terminate_process = kernel32.TerminateProcess
            terminate_process.argtypes = (wintypes.HANDLE, wintypes.UINT)
            terminate_process.restype = wintypes.BOOL
            child_handle = wintypes.HANDLE(process._handle)  # type: ignore[attr-defined]
            if not terminate_process(child_handle, 198):
                raise ctypes.WinError(ctypes.get_last_error())
        assert process.wait(timeout=10) == exit_mode
        _await_task_5d_child_handle_release(case_root)
        return cast(dict[str, object], ready)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)


def test_task_5d_worker_reaches_before_begin_with_both_process_death_modes(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition

    for exit_mode in (197, 198):
        case_root = tmp_path / f"exit-{exit_mode}"
        run_task_5d_worker(case_root, "before_begin", exit_mode)

        recovered = make_spool(case_root, clock=StubClock(START + timedelta(hours=1)))
        try:
            assert recovered.recover_existing() is RecoveryDisposition.RECOVERED
            assert rows(recovered, "SELECT COUNT(*) FROM consent_epochs") == [(0,)]
            assert rows(recovered, "SELECT COUNT(*) FROM evidence_sessions") == [(0,)]
            assert rows(
                recovered,
                "SELECT reason_code, erased_session_count, erased_event_count"
                " FROM erasure_tombstones",
            ) == [("unclean_epoch", 1, 2)]
        finally:
            recovered.close()


@pytest.mark.parametrize("exit_mode", (197, 198))
@pytest.mark.parametrize(
    ("failpoint", "raw_event_count", "raw_session_state", "raw_epoch_state"),
    [
        pytest.param(
            "after_event_insert_before_commit", 2, "open", "active", id="event_insert"
        ),
        pytest.param("after_event_commit", 3, "open", "active", id="event_commit"),
        pytest.param("before_seal_commit", 3, "open", "active", id="seal_precommit"),
        pytest.param(
            "after_seal_commit_before_ack", 4, "sealed", "closed", id="seal_postcommit"
        ),
        pytest.param(
            "after_revoke_request_commit", 2, "open", "revoked", id="revoke_scheduled"
        ),
        pytest.param(
            "after_logical_purge_before_vacuum", 0, None, None, id="logical_purge"
        ),
        pytest.param("after_vacuum_before_ack", 0, None, None, id="vacuum_complete"),
        pytest.param(
            "after_drain_ack_before_exit", 4, "sealed", "closed", id="drain_acknowledged"
        ),
    ],
)
def test_task_5d_ordinary_crash_matrix_recovers_exact_transaction_boundaries(
    tmp_path: Path,
    failpoint: str,
    raw_event_count: int,
    raw_session_state: str | None,
    raw_epoch_state: str | None,
    exit_mode: int,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition

    case_root = tmp_path / f"{failpoint}-exit{exit_mode}"
    run_task_5d_worker(case_root, failpoint, exit_mode)
    database = case_root / "evidence" / "capture-v1.sqlite3"
    with sqlite3.connect(database) as inspection:
        assert inspection.execute("SELECT COUNT(*) FROM evidence_events").fetchone() == (
            raw_event_count,
        )
        session = inspection.execute("SELECT state FROM evidence_sessions").fetchone()
        epoch = inspection.execute("SELECT state FROM consent_epochs").fetchone()
        assert session == (raw_session_state,) if raw_session_state is not None else session is None
        assert epoch == (raw_epoch_state,) if raw_epoch_state is not None else epoch is None

    recovered = make_spool(case_root, clock=StubClock(START + timedelta(hours=1)))
    try:
        disposition = recovered.recover_existing()
        if failpoint == "after_revoke_request_commit":
            assert disposition is RecoveryDisposition.FAULTED
            assert rows(recovered, "SELECT state FROM erasure_requests") == [("pending",)]
            return
        assert disposition is RecoveryDisposition.RECOVERED
        if failpoint in {
            "after_seal_commit_before_ack",
            "after_drain_ack_before_exit",
        }:
            assert rows(recovered, "SELECT state FROM evidence_sessions") == [("sealed",)]
            assert rows(recovered, "SELECT state FROM consent_epochs") == [("closed",)]
            assert rows(recovered, "SELECT COUNT(*) FROM erasure_tombstones") == [(0,)]
        elif failpoint in {
            "after_logical_purge_before_vacuum",
            "after_vacuum_before_ack",
        }:
            assert rows(
                recovered,
                "SELECT reason_code, erased_session_count, erased_event_count"
                " FROM erasure_tombstones",
            ) == [("revoked", 1, 2)]
        else:
            assert rows(
                recovered,
                "SELECT reason_code, erased_session_count, erased_event_count"
                " FROM erasure_tombstones",
            ) == [("unclean_epoch", 1, raw_event_count)]
    finally:
        recovered.close()


def test_task_5d_marker_only_contenders_launch_two_isolated_children() -> None:
    """The marker-only ownership race must cross two independent CPython processes."""

    source = inspect.getsource(assert_task_5d_single_winner_consent_resume)

    assert "subprocess.Popen(" in source
    assert "range(2)" in source
    assert "SetEvent" in source
    assert 'write("GO\\n")' not in source


def test_task_5d_parent_does_not_reemit_private_child_stderr() -> None:
    """Qualification failures use fixed diagnostics, never child-controlled stderr."""

    crash_parent = inspect.getsource(run_task_5d_worker)
    contender_parent = inspect.getsource(assert_task_5d_single_winner_consent_resume)

    assert "stderr.read()" not in crash_parent
    assert "stderr.read()" not in contender_parent


def assert_task_5d_single_winner_consent_resume(
    case_root: Path,
    original_marker: bytes,
) -> None:
    """Race two isolated writers after marker-only recovery through the real create path."""

    import ctypes
    import json
    import queue
    import subprocess
    import uuid
    from ctypes import wintypes

    from hermes_realtime.evidence import storage_security

    root = case_root / "evidence"
    marker = root / EXPECTED_MANIFEST_NAMES["root_marker"]
    assert marker.read_bytes() == original_marker
    original_root_id = storage_security.parse_root_marker(original_marker)
    worker = Path(__file__).with_name("spool_crash_worker.py")
    executable, environment = task_5d_worker_launch_environment()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_event = kernel32.CreateEventW
    create_event.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
    create_event.restype = wintypes.HANDLE
    set_event = kernel32.SetEvent
    set_event.argtypes = (wintypes.HANDLE,)
    set_event.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    start_event_name = f"Local\\HermesRealtimeTask5D-{uuid.uuid4()}"
    start_event = create_event(None, True, False, start_event_name)
    if not start_event:
        raise ctypes.WinError(ctypes.get_last_error())
    contenders = [
        subprocess.Popen(
            [
                str(executable),
                "-I",
                str(worker),
                "--case-root",
                str(case_root),
                "--marker-only-contender",
                str(contender_id),
                "--start-event",
                start_event_name,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=environment,
        )
        for contender_id in range(2)
    ]

    def bounded_line(process: subprocess.Popen[str], label: str) -> str:
        stdout = process.stdout
        assert stdout is not None
        lines: queue.Queue[str] = queue.Queue(maxsize=1)
        threading.Thread(
            target=lambda: lines.put(stdout.readline()),
            name=f"task-5d-{label}-{process.pid}",
            daemon=True,
        ).start()
        try:
            line = lines.get(timeout=10)
        except queue.Empty:
            pytest.fail(
                f"Task 5D contender {label} did not emit a bounded protocol frame",
                pytrace=False,
            )
        if not line:
            pytest.fail(
                f"Task 5D contender {label} exited before its protocol frame",
                pytrace=False,
            )
        return line

    def frame(process: subprocess.Popen[str], label: str) -> dict[str, object]:
        try:
            decoded = json.loads(bounded_line(process, label))
        except json.JSONDecodeError:
            pytest.fail(
                f"Task 5D contender {label} emitted an invalid protocol frame",
                pytrace=False,
            )
        if type(decoded) is not dict:
            pytest.fail(
                f"Task 5D contender {label} emitted a non-object protocol frame",
                pytrace=False,
            )
        return cast(dict[str, object], decoded)

    try:
        ready = [frame(process, f"ready-{index}") for index, process in enumerate(contenders)]
        assert ready == [
            {"contenderId": index, "protocolVersion": 1, "state": "ARMED"}
            for index in range(2)
        ]
        if not set_event(start_event):
            raise ctypes.WinError(ctypes.get_last_error())
        terminal = [
            frame(process, f"terminal-{index}") for index, process in enumerate(contenders)
        ]
        outcomes: list[tuple[str, bool]] = []
        for item in terminal:
            assert type(item["contenderId"]) is int
            assert type(item["disposition"]) is str
            assert type(item["ownershipRefusal"]) is bool
            assert item["protocolVersion"] == 1
            assert item["state"] == "TERMINAL"
            outcomes.append(
                (cast(str, item["disposition"]), cast(bool, item["ownershipRefusal"]))
            )
        assert sorted(item["contenderId"] for item in terminal) == [0, 1]
        assert sorted(outcomes) == [("committed", False), ("faulted", True)]
    finally:
        # Idempotently release any child that reached the shared boundary before
        # an earlier protocol assertion failed, so cleanup never strands it.
        set_event(start_event)
        for process in contenders:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write("CLOSE\n")
                process.stdin.flush()
        for process in contenders:
            try:
                assert process.wait(timeout=10) == 0
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
                pytest.fail(
                    "Task 5D contender did not reap within its bounded close",
                    pytrace=False,
                )
        if not close_handle(start_event):
            raise ctypes.WinError(ctypes.get_last_error())

    assert marker.read_bytes() == original_marker
    assert storage_security.parse_root_marker(marker.read_bytes()) == original_root_id
    resolved = storage_security.MANIFEST_V1.resolve(root)
    assert not resolved.root_marker_init.exists()
    assert not resolved.sentinel_init.exists()
    assert {path.name for path in root.iterdir()} == {
        resolved.root_marker.name,
        resolved.sentinel.name,
        resolved.database.name,
    }
    sentinel = storage_security.decode_sentinel_image(resolved.sentinel.read_bytes())
    assert sentinel.active.state.value == "clear"
    with sqlite3.connect(resolved.database) as inspection:
        assert inspection.execute(
            "SELECT installation_id FROM producer_installation"
        ).fetchall() == [
            ("10000000-0000-4000-8000-000000000001",)
        ]
        assert inspection.execute(
            "SELECT consent_epoch_id, producer_instance_id, state FROM consent_epochs"
        ).fetchall() == [
            (
                "20000000-0000-4000-8000-000000000003",
                "10000000-0000-4000-8000-000000000002",
                "active",
            )
        ]
        assert inspection.execute(
            "SELECT logical_session_id, consent_epoch_id, state FROM evidence_sessions"
        ).fetchall() == [
            ("20000000-0000-4000-8000-000000000004", "20000000-0000-4000-8000-000000000003", "open")
        ]
        assert inspection.execute(
            "SELECT event_kind, COUNT(*) FROM evidence_events "
            "GROUP BY event_kind ORDER BY event_kind"
        ).fetchall() == [("binding_opened", 1), ("session_opened", 1)]


@pytest.mark.parametrize("exit_mode", (197, 198))
@pytest.mark.parametrize(
    ("failpoint", "expected_recovery"),
    [
        pytest.param("after_root_marker_init_create", "faulted", id="root-empty"),
        pytest.param("after_root_marker_init_partial_write", "faulted", id="root-partial"),
        pytest.param("after_root_marker_init_full_write", "absent", id="root-full"),
        pytest.param("after_root_marker_init_flush", "absent", id="root-flushed"),
        pytest.param("after_root_marker_activation", "absent", id="root-active"),
        pytest.param("after_sentinel_init_create", "faulted", id="sentinel-empty"),
        pytest.param(
            "after_sentinel_init_partial_write", "faulted", id="sentinel-partial"
        ),
        pytest.param(
            "after_sentinel_init_full_write", "absent", id="sentinel-full"
        ),
        pytest.param(
            "after_sentinel_init_flush", "absent", id="sentinel-flushed"
        ),
        pytest.param(
            "after_sentinel_activation", "purge_completed", id="sentinel-active"
        ),
        pytest.param("after_first_db_create", "purge_completed", id="database-created"),
        pytest.param(
            "after_first_schema_commit", "purge_completed", id="schema-committed"
        ),
        pytest.param("after_first_epoch_commit", "purge_completed", id="epoch-committed"),
        pytest.param("after_first_marker_clear", "recovered", id="marker-cleared"),
    ],
)
def test_task_5d_first_store_creation_crash_matrix_is_exact_and_recoverable(
    tmp_path: Path,
    failpoint: str,
    expected_recovery: str,
    exit_mode: int,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    case_root = tmp_path / f"{failpoint}-exit{exit_mode}"
    run_task_5d_worker(case_root, failpoint, exit_mode)
    marker_only_original: bytes | None = None
    if failpoint == "after_root_marker_activation":
        root = case_root / "evidence"
        assert {path.name for path in root.iterdir()} == {
            EXPECTED_MANIFEST_NAMES["root_marker"]
        }
        marker_only_original = (root / EXPECTED_MANIFEST_NAMES["root_marker"]).read_bytes()
        storage_security.parse_root_marker(marker_only_original)

    recovered = make_spool(case_root, clock=StubClock(START + timedelta(hours=1)))
    try:
        assert recovered.recover_existing() is RecoveryDisposition(expected_recovery)
    finally:
        recovered.close()

    if failpoint == "after_root_marker_activation":
        assert marker_only_original is not None
        marker = case_root / "evidence" / EXPECTED_MANIFEST_NAMES["root_marker"]
        assert marker.read_bytes() == marker_only_original
        storage_security.parse_root_marker(marker.read_bytes())
        assert_task_5d_single_winner_consent_resume(case_root, marker_only_original)
    elif expected_recovery in {"absent", "purge_completed"}:
        contender = make_spool(case_root, clock=StubClock(START + timedelta(hours=2)))
        try:
            assert contender.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        finally:
            contender.close()


@pytest.mark.parametrize("exit_mode", (197, 198))
@pytest.mark.parametrize(
    "failpoint",
    [
        "after_recreate_pending_fsync",
        "after_recreate_db_create",
        "after_recreate_schema_commit",
        "after_recreate_epoch_commit",
        "after_recreate_marker_clear",
    ],
)
def test_task_5d_post_purge_recreation_crash_matrix_resumes_without_decoy_loss(
    tmp_path: Path,
    failpoint: str,
    exit_mode: int,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition, StoreDisposition

    case_root = tmp_path / f"{failpoint}-exit{exit_mode}"
    run_task_5d_worker(case_root, failpoint, exit_mode)
    decoy = case_root / "evidence" / "recreation-decoy.bin"
    assert decoy.read_bytes() == b"not-owned"

    recovered = make_spool(case_root, clock=StubClock(START + timedelta(hours=2)))
    try:
        expected = (
            RecoveryDisposition.RECOVERED
            if failpoint == "after_recreate_marker_clear"
            else RecoveryDisposition.PURGE_COMPLETED
        )
        assert recovered.recover_existing() is expected
    finally:
        recovered.close()
    assert decoy.read_bytes() == b"not-owned"

    if failpoint != "after_recreate_marker_clear":
        contender = make_spool(case_root, clock=StubClock(START + timedelta(hours=3)))
        try:
            assert contender.create_epoch(make_create_epoch()) is StoreDisposition.COMMITTED
        finally:
            contender.close()


@pytest.mark.parametrize("exit_mode", (197, 198))
@pytest.mark.parametrize(
    ("failpoint", "sentinel_state", "purge_required"),
    [
        pytest.param("before_rollback_sentinel_write", "clear", 0, id="before-sentinel"),
        pytest.param(
            "after_rollback_sentinel_fsync",
            "clock_rollback_purge_pending",
            0,
            id="after-sentinel",
        ),
        pytest.param(
            "before_rollback_db_latch",
            "clock_rollback_purge_pending",
            0,
            id="before-database",
        ),
        pytest.param(
            "after_rollback_db_latch",
            "clock_rollback_purge_pending",
            1,
            id="after-database",
        ),
    ],
)
def test_task_5d_clock_rollback_crash_matrix_recovers_at_both_restart_clocks(
    tmp_path: Path,
    failpoint: str,
    sentinel_state: str,
    purge_required: int,
    exit_mode: int,
) -> None:
    import json
    import shutil

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition, SentinelState

    case_root = tmp_path / f"{failpoint}-exit{exit_mode}"
    run_task_5d_worker(case_root, failpoint, exit_mode)
    root = case_root / "evidence"
    sentinel_bytes = (root / "capture-v1.owner").read_bytes()
    active = storage_security.decode_sentinel_image(sentinel_bytes).active
    assert active.state.value == sentinel_state
    with sqlite3.connect(root / "capture-v1.sqlite3") as inspection:
        assert inspection.execute(
            "SELECT purge_required FROM producer_installation WHERE singleton=1"
        ).fetchone() == (purge_required,)
        database_sha256 = hashlib.sha256(
            "\n".join(inspection.iterdump()).encode("utf-8")
        ).hexdigest()
    if failpoint == "before_rollback_sentinel_write":
        assert json.loads((case_root / "rollback-baseline.json").read_text("utf-8")) == {
            "databaseSha256": database_sha256,
            "sentinelSha256": hashlib.sha256(sentinel_bytes).hexdigest(),
        }

    for restart_name, restart_time in (
        ("caught-up", START + timedelta(hours=2)),
        ("regressed", START - timedelta(hours=2)),
    ):
        restart_root = tmp_path / f"{failpoint}-exit{exit_mode}-{restart_name}"
        shutil.copytree(case_root, restart_root)
        recovered = make_spool(restart_root, clock=StubClock(restart_time))
        try:
            expected = (
                RecoveryDisposition.RECOVERED
                if failpoint == "before_rollback_sentinel_write" and restart_name == "caught-up"
                else RecoveryDisposition.PURGE_COMPLETED
            )
            assert recovered.recover_existing() is expected
            if expected is RecoveryDisposition.RECOVERED:
                assert rows(
                    recovered,
                    "SELECT consent_epoch_id, state FROM consent_epochs",
                ) == [(TASK_5D_OLDER_EPOCH_ID, "closed")]
                assert rows(
                    recovered,
                    "SELECT logical_session_id, state FROM evidence_sessions",
                ) == [(TASK_5D_OLDER_SESSION_ID, "sealed")]
                assert rows(
                    recovered,
                    "SELECT COUNT(*) FROM evidence_events WHERE logical_session_id=?",
                    (TASK_5D_OLDER_SESSION_ID,),
                ) == [(4,)]
                assert rows(
                    recovered,
                    "SELECT scope_kind, reason_code FROM erasure_requests"
                    " WHERE scope_kind='store' OR reason_code='clock_rollback'",
                ) == []
                assert rows(
                    recovered,
                    "SELECT scope_kind, reason_code FROM erasure_tombstones",
                ) == [("consent_epoch", "unclean_epoch")]
        finally:
            recovered.close()
        resolved = storage_security.MANIFEST_V1.resolve(restart_root / "evidence")
        if expected is RecoveryDisposition.PURGE_COMPLETED:
            assert all(not path.exists() for path in resolved.deletable)
            assert resolved.root_marker.is_file()
            assert resolved.sentinel.is_file()
            assert storage_security.decode_sentinel_image(
                resolved.sentinel.read_bytes()
            ).active.state is SentinelState.CLEAR
        assert (
            restart_root / "evidence" / "rollback-decoy.bin"
        ).read_bytes() == b"not-owned"


@pytest.mark.parametrize("exit_mode", (197, 198))
@pytest.mark.parametrize(
    ("failpoint", "deleted_prefix"),
    [
        pytest.param("after_full_purge_marker_fsync", 0, id="marker-pending"),
        pytest.param("after_full_purge_db_delete", 1, id="database"),
        pytest.param("after_full_purge_journal_delete", 2, id="journal"),
        pytest.param("after_full_purge_wal_delete", 3, id="wal"),
        pytest.param("after_full_purge_shm_delete", 4, id="shm"),
        pytest.param("after_full_purge_vacuum_delete", 5, id="vacuum"),
        pytest.param("after_full_purge_tmp_delete", 6, id="temporary"),
        pytest.param("after_full_purge_absence_verify", 6, id="absence-verified"),
        pytest.param("after_full_purge_marker_clear", 6, id="marker-cleared"),
    ],
)
def test_task_5d_full_purge_crash_matrix_deletes_exact_manifest_and_preserves_decoys(
    tmp_path: Path,
    failpoint: str,
    deleted_prefix: int,
    exit_mode: int,
) -> None:

    from hermes_realtime.evidence import storage_security
    from hermes_realtime.evidence.models import RecoveryDisposition

    case_root = tmp_path / f"{failpoint}-exit{exit_mode}"
    run_task_5d_worker(case_root, failpoint, exit_mode)
    root = case_root / "evidence"
    resolved = storage_security.MANIFEST_V1.resolve(root)
    original_marker = resolved.root_marker.read_bytes()
    original_root_id = storage_security.parse_root_marker(original_marker)
    assert [not path.exists() for path in resolved.deletable] == [
        index < deleted_prefix for index in range(6)
    ]
    expected_state = (
        "clear"
        if failpoint == "after_full_purge_marker_clear"
        else "full_purge_pending"
    )
    assert storage_security.decode_sentinel_image(
        resolved.sentinel.read_bytes()
    ).active.state.value == expected_state
    assert (root / "purge-decoy.bin").read_bytes() == b"not-owned"

    recovered = make_spool(case_root, clock=StubClock(START + timedelta(hours=2)))
    try:
        expected = (
            RecoveryDisposition.ABSENT
            if failpoint == "after_full_purge_marker_clear"
            else RecoveryDisposition.PURGE_COMPLETED
        )
        assert recovered.recover_existing() is expected
    finally:
        recovered.close()
    assert all(not path.exists() for path in resolved.deletable)
    assert resolved.root_marker.read_bytes() == original_marker
    assert storage_security.parse_root_marker(resolved.root_marker.read_bytes()) == original_root_id
    assert resolved.sentinel.is_file()
    assert storage_security.decode_sentinel_image(
        resolved.sentinel.read_bytes()
    ).active.state.value == "clear"
    assert not resolved.root_marker_init.exists()
    assert not resolved.sentinel_init.exists()
    assert {path.name for path in root.iterdir()} == {
        resolved.root_marker.name,
        resolved.sentinel.name,
        "purge-decoy.bin",
    }
    assert (root / "purge-decoy.bin").read_bytes() == b"not-owned"


def test_only_the_spool_module_imports_sqlite3() -> None:
    package = Path(__file__).resolve().parents[2] / "src" / "hermes_realtime"

    def imports_sqlite3(module: Path) -> bool:
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name.split(".")[0] == "sqlite3" for alias in node.names
            ):
                return True
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "sqlite3":
                return True
        return False

    modules = sorted(package.rglob("*.py"))
    assert [module for module in modules if module.name == "sqlite_spool.py"]
    offenders = sorted(
        module.relative_to(package).as_posix()
        for module in modules
        if module.name != "sqlite_spool.py" and imports_sqlite3(module)
    )
    assert offenders == []
    assert imports_sqlite3(package / "evidence" / "sqlite_spool.py") is True


def test_object_ace_sid_offset_accounts_for_its_conditional_guids() -> None:
    # Regression: the SID was read at a fixed offset of 8, correct only for
    # ACCESS_ALLOWED_ACE. ACCESS_ALLOWED_OBJECT_ACE inserts Flags and up to two
    # conditional GUIDs first, so an owner-only DACL carrying an object ACE was
    # rejected with a wrong diagnostic.
    import ctypes

    from hermes_realtime.evidence import storage_security

    assert (
        storage_security._ace_sid_offset(
            ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(64))),
            storage_security._ACCESS_ALLOWED_ACE_TYPE,
        )
        == 8
    )

    for flags, expected in ((0x0, 12), (0x1, 28), (0x2, 28), (0x3, 44)):
        buffer = ctypes.create_string_buffer(64)
        ctypes.memmove(
            ctypes.addressof(buffer) + 8,
            ctypes.byref(ctypes.c_uint32(flags)),
            ctypes.sizeof(ctypes.c_uint32),
        )
        assert (
            storage_security._ace_sid_offset(
                ctypes.c_void_p(ctypes.addressof(buffer)),
                storage_security._ACCESS_ALLOWED_OBJECT_ACE_TYPE,
            )
            == expected
        )


def test_unknown_ace_type_has_no_readable_sid_offset() -> None:
    import ctypes

    from hermes_realtime.evidence import storage_security

    assert (
        storage_security._ace_sid_offset(
            ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(64))),
            0x7F,
        )
        is None
    )


@pytest.mark.skipif(sys.platform != "win32", reason="the real probe is Windows-only")
def test_allocated_bytes_fails_closed_when_the_allocation_cannot_be_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: an unmeasurable allocation reported 0, which under-counted
    # owned bytes and let the maintenance headroom gate pass on a volume that
    # could not hold another maintenance instant.
    import ctypes

    from hermes_realtime.evidence import storage_security

    sample = tmp_path / "capture-v1.sqlite3"
    sample.write_bytes(b"x" * 4096)
    kernel32 = storage_security._kernel32()

    class UnmeasurableKernel32:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(kernel32, name)

        def GetCompressedFileSizeW(self, path: object, high: object) -> int:
            ctypes.set_last_error(5)
            return storage_security._INVALID_FILE_SIZE

    monkeypatch.setattr(storage_security, "_kernel32", lambda: UnmeasurableKernel32())
    probe = storage_security.WindowsStorageProbeV1()

    with pytest.raises(storage_security.EvidenceStorageError) as caught:
        probe.allocated_bytes(sample)
    assert caught.value.fault.value == "quota_unavailable"


@pytest.mark.skipif(sys.platform != "win32", reason="the real probe is Windows-only")
def test_allocated_bytes_still_reports_zero_for_an_absent_artifact(
    tmp_path: Path,
) -> None:

    from hermes_realtime.evidence import storage_security

    probe = storage_security.WindowsStorageProbeV1()
    assert probe.allocated_bytes(tmp_path / "capture-v1.sqlite3-wal") == 0
