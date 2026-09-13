"""Durability and state-machine tests for the private #62 run journal."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import qualification_run_journal as journal


def _binding() -> journal.RunJournalBindingV1:
    return journal.RunJournalBindingV1(
        run_id="run_0123456789abcdef",
        candidate_head="1" * 40,
        candidate_tree="2" * 40,
        qualification_input_sha256="3" * 64,
    )


def _location() -> journal.RecoveryLocationFactsV1:
    return journal.RecoveryLocationFactsV1(
        journal_path="C:/private-recovery/run.journal",
        parent_path="C:/private-recovery",
        volume_serial=7,
        journal_file_id=11,
        parent_file_id=12,
        marker_path="C:/private-recovery/.qualification-recovery-v1",
        marker_file_id=13,
        marker_sha256="4" * 64,
    )


def _filesystem() -> journal.FilesystemIntentV1:
    return journal.FilesystemIntentV1(
        role="host_temporary_root",
        path="C:/owned/run/host-temp",
        directory=True,
        cleanup_scope="owned_subtree",
        allowed_artifacts=(),
    )


def _filesystem_identity() -> journal.FilesystemIdentityV1:
    return journal.FilesystemIdentityV1(
        path="C:/owned/run/host-temp",
        parent_path="C:/owned/run",
        volume_serial=7,
        file_id=21,
        parent_file_id=22,
        marker_path="C:/owned/run/.qualification-owned-v1",
        marker_file_id=23,
        marker_sha256="5" * 64,
        directory=True,
    )


@pytest.mark.parametrize(
    "marker_path",
    [
        "C:/owned/run/host-temp",
        "c:/OWNED/RUN/HOST-TEMP",
    ],
)
def test_filesystem_identity_refuses_resource_marker_path_aliases(marker_path: str) -> None:
    with pytest.raises(ValueError, match="filesystem identity differs"):
        replace(_filesystem_identity(), marker_path=marker_path)


def test_filesystem_identity_refuses_resource_marker_hardlink_identity() -> None:
    identity = _filesystem_identity()

    with pytest.raises(ValueError, match="filesystem identity differs"):
        replace(identity, marker_file_id=identity.file_id)


@pytest.mark.parametrize(
    "marker_path",
    [
        "C:/private-recovery/run.journal",
        "c:/PRIVATE-RECOVERY/RUN.JOURNAL",
    ],
)
def test_recovery_location_refuses_journal_marker_path_aliases(marker_path: str) -> None:
    with pytest.raises(ValueError, match="journal recovery location differs"):
        replace(_location(), marker_path=marker_path)


def test_recovery_location_refuses_journal_marker_hardlink_identity() -> None:
    location = _location()

    with pytest.raises(ValueError, match="journal recovery location differs"):
        replace(location, marker_file_id=location.journal_file_id)


def _process() -> journal.ProcessIntentV1:
    return journal.ProcessIntentV1(
        scenario_id="normal_conversation",
        role="host_root",
        image_basename="python.exe",
        image_sha256="6" * 64,
    )


def _process_identity() -> journal.ProcessIdentityFactsV1:
    return journal.ProcessIdentityFactsV1(
        scenario_id="normal_conversation",
        role="host_root",
        pid=101,
        parent_pid=51,
        parent_creation_filetime=500,
        creation_filetime=1000,
        image_basename="python.exe",
        image_sha256="6" * 64,
    )


class _FakeIo:
    def __init__(self, raw: bytes = b"") -> None:
        self.raw = bytearray(raw)
        self.events: list[tuple[object, ...]] = []
        self.partial: int | None = None
        self.fail_flush = False
        self.flush_error: BaseException | None = None
        self.fail_write_offset: int | None = None
        self.grow_on_read = False
        self.read_completed = False
        self.fact_after_read: dict[str, object] = {}
        self.path_after_read: str | None = None
        self.fact = journal._JournalHandleFactV1(7, 11, len(raw), False, 1)

    def handle_fact(self, handle: int) -> journal._JournalHandleFactV1:
        assert handle == 41
        change = self.fact_after_read if self.read_completed else {}
        return replace(self.fact, size=len(self.raw), **change)

    def final_path(self, handle: int) -> str:
        assert handle == 41
        if self.read_completed and self.path_after_read is not None:
            return self.path_after_read
        return "C:/private-recovery/run.journal"

    def read(self, handle: int, maximum: int) -> bytes:
        assert handle == 41
        self.events.append(("read", maximum))
        raw = bytes(self.raw[: maximum + 1])
        if self.grow_on_read:
            self.raw.extend(b"x")
        self.read_completed = True
        return raw

    def write(self, handle: int, offset: int, raw: bytes) -> int:
        assert handle == 41 and 0 <= offset <= len(self.raw)
        if self.fail_write_offset is not None and offset == self.fail_write_offset:
            self.events.append(("write", offset, len(raw), 0))
            raise OSError("injected write detail")
        count = len(raw) if self.partial is None else self.partial
        self.events.append(("write", offset, len(raw), count))
        self.raw[offset : offset + count] = raw[:count]
        return count

    def flush(self, handle: int) -> None:
        assert handle == 41
        self.events.append(("flush",))
        if self.flush_error is not None:
            raise self.flush_error
        if self.fail_flush:
            raise OSError("injected flush detail")


def _new(monkeypatch: pytest.MonkeyPatch, io: _FakeIo) -> journal._RunJournalWriterV1:
    monkeypatch.setattr(journal, "_journal_io", lambda: io)
    return journal._create_run_journal(41, _binding(), _location())


def _complete(writer: journal._RunJournalWriterV1) -> None:
    filesystem = writer.intend_filesystem(_filesystem())
    writer.bind_filesystem(filesystem, _filesystem_identity())
    process = writer.intend_process(_process())
    writer.bind_suspended_process(process, _process_identity())
    writer.intend_resume(process)
    writer.record_resumed(process)
    writer.record_process_exited(process, _process_identity())
    writer.record_filesystem_removed(filesystem, _filesystem_identity())
    writer.record_sequence_complete()


def test_journal_records_each_transition_only_after_write_and_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    _complete(writer)

    assert [event[0] for event in io.events] == ["write", "flush"] * 21
    assert [event[1] for event in io.events if event[0] == "write"][2::2] == (
        [journal._HEADER_OFFSET] * 10
    )
    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert facts == journal.RunJournalFactsV1(
        frame_count=10,
        unconfirmed_frames=0,
        byte_count=len(io.raw),
        integrity_complete=True,
        partial_tail=False,
        recorded_complete=True,
        execution_uncertain=False,
        pending_filesystems=(),
        pending_processes=(),
        reason=None,
    )


def test_process_launch_and_resume_order_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _new(monkeypatch, _FakeIo())
    process = writer.intend_process(_process())

    with pytest.raises(ValueError, match="process transition differs"):
        writer.intend_resume(process)
    writer.bind_suspended_process(process, _process_identity())
    with pytest.raises(ValueError, match="process transition differs"):
        writer.record_resumed(process)
    writer.intend_resume(process)
    writer.record_resumed(process)
    with pytest.raises(ValueError, match="process transition differs"):
        writer.record_resumed(process)


def test_completion_refuses_pending_or_uncertain_obligations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _new(monkeypatch, _FakeIo())
    filesystem = writer.intend_filesystem(_filesystem())
    with pytest.raises(ValueError, match="sequence has unresolved obligations"):
        writer.record_sequence_complete()
    writer.record_filesystem_absent(filesystem)
    process = writer.intend_process(_process())
    writer.bind_suspended_process(process, _process_identity())
    writer.intend_resume(process)
    writer.record_process_exited(process, _process_identity())
    with pytest.raises(ValueError, match="sequence has unresolved execution"):
        writer.record_sequence_complete()


def test_absence_observations_close_only_unbound_intents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _new(monkeypatch, _FakeIo())
    filesystem = writer.intend_filesystem(_filesystem())
    process = writer.intend_process(_process())
    writer.record_filesystem_absent(filesystem)
    writer.record_process_absent(process)
    writer.record_sequence_complete()
    with pytest.raises(ValueError, match="journal sequence is already complete"):
        writer.intend_filesystem(_filesystem())


@pytest.mark.parametrize("fault", ["partial", "flush"])
def test_failed_durable_append_permanently_poisons_the_live_writer(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    if fault == "partial":
        io.partial = 3
    else:
        io.fail_flush = True

    with pytest.raises(journal.RunJournalDurabilityError, match="journal append is not durable"):
        writer.intend_filesystem(_filesystem())
    with pytest.raises(journal.RunJournalDurabilityError, match="journal writer is poisoned"):
        writer.intend_process(_process())
    if fault == "partial":
        assert io.events[-1][0] == "write"
    else:
        assert io.events[-1] == ("flush",)


def test_control_interruption_preserves_identity_and_poisons_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    interruption = KeyboardInterrupt("private interruption detail")
    io.flush_error = interruption

    with pytest.raises(KeyboardInterrupt) as raised:
        writer.intend_filesystem(_filesystem())
    assert raised.value is interruption
    with pytest.raises(journal.RunJournalDurabilityError, match="journal writer is poisoned"):
        writer.intend_process(_process())


def test_create_never_overwrites_an_existing_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo(b"retained")
    monkeypatch.setattr(journal, "_journal_io", lambda: io)

    with pytest.raises(ValueError, match="journal handle identity differs"):
        journal._create_run_journal(41, _binding(), _location())
    assert bytes(io.raw) == b"retained"
    assert io.events == []


@pytest.mark.parametrize(
    "change",
    [
        {"reparse": True},
        {"links": 2},
        {"directory": True},
        {"inheritable": True},
    ],
)
def test_create_requires_an_exclusive_regular_noninherited_handle(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, object]
) -> None:
    io = _FakeIo()
    io.fact = replace(io.fact, **change)
    monkeypatch.setattr(journal, "_journal_io", lambda: io)

    with pytest.raises(ValueError, match="journal handle identity differs"):
        journal._create_run_journal(41, _binding(), _location())
    assert io.events == []


def test_append_extent_drift_poisons_without_overwriting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    before = bytes(io.raw)
    events = tuple(io.events)
    io.raw.extend(b"foreign")

    with pytest.raises(journal.RunJournalDurabilityError, match="journal append is not durable"):
        writer.intend_filesystem(_filesystem())
    assert bytes(io.raw) == before + b"foreign"
    assert tuple(io.events) == events


def test_intent_and_resume_permissions_return_only_after_durable_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    process = writer.intend_process(_process())
    assert [event[0] for event in io.events[-2:]] == ["write", "flush"]
    writer.bind_suspended_process(process, _process_identity())
    writer.intend_resume(process)
    assert [event[0] for event in io.events[-2:]] == ["write", "flush"]


@pytest.mark.parametrize("mutation", ["tail", "digest", "growth"])
def test_torn_corrupt_or_growing_input_is_never_recorded_complete(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    _complete(writer)
    if mutation == "tail":
        del io.raw[-1]
    elif mutation == "digest":
        io.raw[-1] ^= 1
    else:
        io.grow_on_read = True

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert not facts.integrity_complete
    assert not facts.recorded_complete
    assert facts.reason in {"partial-tail", "digest", "changed-while-read"}
    with pytest.raises(ValueError, match="journal cannot be resumed"):
        journal._resume_run_journal(41, _binding(), _location())


@pytest.mark.parametrize("mutation", ["path", "links"])
def test_handle_policy_drift_during_read_is_never_recorded_complete(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    _complete(writer)
    if mutation == "path":
        io.path_after_read = "C:/private-recovery/renamed.journal"
    else:
        io.fact_after_read = {"links": 2}

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert not facts.integrity_complete
    assert not facts.recorded_complete
    assert facts.reason == "changed-while-read"


def test_binding_and_handle_identity_drift_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    io = _FakeIo()
    _new(monkeypatch, io)

    drifted = journal._inspect_run_journal(
        41, replace(_binding(), candidate_tree="9" * 40), _location()
    )
    assert not drifted.integrity_complete
    assert drifted.reason == "head-chain"
    io.fact = replace(io.fact, file_id=99)
    with pytest.raises(ValueError, match="journal handle identity differs"):
        journal._inspect_run_journal(41, _binding(), _location())


def test_resume_appends_to_an_exact_valid_incomplete_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    filesystem = writer.intend_filesystem(_filesystem())
    writer.bind_filesystem(filesystem, _filesystem_identity())

    resumed = journal._resume_run_journal(41, _binding(), _location())
    resumed.record_filesystem_removed(filesystem, _filesystem_identity())
    resumed.record_sequence_complete()
    assert journal._inspect_run_journal(41, _binding(), _location()).recorded_complete


def test_identity_and_transition_inputs_are_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _new(monkeypatch, _FakeIo())
    filesystem = writer.intend_filesystem(_filesystem())
    with pytest.raises(ValueError, match="filesystem identity differs"):
        writer.bind_filesystem(filesystem, replace(_filesystem_identity(), path="C:/foreign/path"))
    process = writer.intend_process(_process())
    with pytest.raises(ValueError, match="process identity differs"):
        writer.bind_suspended_process(process, replace(_process_identity(), image_sha256="7" * 64))
    with pytest.raises(TypeError, match="ordinal type differs"):
        writer.record_process_absent(True)  # type: ignore[arg-type]


def test_filesystem_removal_must_match_the_bound_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _new(monkeypatch, _FakeIo())
    filesystem = writer.intend_filesystem(_filesystem())
    writer.bind_filesystem(filesystem, _filesystem_identity())

    with pytest.raises(ValueError, match="filesystem identity differs"):
        writer.record_filesystem_removed(
            filesystem, replace(_filesystem_identity(), marker_file_id=99)
        )


def test_boolean_schema_version_is_not_an_integer_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    _new(monkeypatch, io)
    size = struct.unpack(">I", io.raw[journal._FRAMES_OFFSET : journal._FRAMES_OFFSET + 4])[0]
    start = journal._FRAMES_OFFSET + 4
    record = json.loads(io.raw[start : start + size])
    record["schemaVersion"] = True
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    length = struct.pack(">I", len(payload))
    digest = hashlib.sha256(journal._genesis(_binding(), _location()) + length + payload).digest()
    io.raw = bytearray(
        journal._MAGIC + journal._header_bytes(1, digest) + length + payload + digest
    )

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert not facts.integrity_complete
    assert facts.reason == "state"


def test_frame_and_journal_bounds_refuse_before_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    io = _FakeIo(b"x" * (journal._MAX_JOURNAL_BYTES + 1))
    monkeypatch.setattr(journal, "_journal_io", lambda: io)
    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert facts.reason == "size-bound"
    assert not facts.integrity_complete


@pytest.mark.skipif(os.name != "nt", reason="real WriteFile and FlushFileBuffers journal")
def test_real_windows_borrowed_handle_flushes_and_reopens(tmp_path: Path) -> None:
    import msvcrt

    path = tmp_path / "run.journal"
    with path.open("x+b", buffering=0) as stream:
        handle = msvcrt.get_osfhandle(stream.fileno())
        fact = journal._journal_io().handle_fact(handle)
        location = replace(
            _location(),
            journal_path=path.resolve().as_posix(),
            parent_path=path.resolve().parent.as_posix(),
            volume_serial=fact.volume_serial,
            journal_file_id=fact.file_id,
            parent_file_id=1,
            marker_path=(path.resolve().parent / ".qualification-recovery-v1").as_posix(),
        )
        writer = journal._create_run_journal(handle, _binding(), location)
        filesystem = writer.intend_filesystem(_filesystem())
        writer.record_filesystem_absent(filesystem)
        writer.record_sequence_complete()
        stream.flush()
    with path.open("r+b", buffering=0) as reopened:
        handle = msvcrt.get_osfhandle(reopened.fileno())
        facts = journal._inspect_run_journal(handle, _binding(), location)
    assert facts.integrity_complete and facts.recorded_complete


def _head(io: _FakeIo) -> tuple[int, bytes]:
    """The committed head exactly as any later reader of the file sees it."""
    parsed = journal._parse_header(bytes(io.raw[journal._HEADER_OFFSET : journal._FRAMES_OFFSET]))
    assert parsed is not None
    return parsed


def test_unflushed_final_record_is_never_confirmed(monkeypatch: pytest.MonkeyPatch) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    filesystem = writer.intend_filesystem(_filesystem())
    writer.bind_filesystem(filesystem, _filesystem_identity())
    writer.record_filesystem_removed(filesystem, _filesystem_identity())
    io.fail_flush = True
    with pytest.raises(journal.RunJournalDurabilityError, match="journal append is not durable"):
        writer.record_sequence_complete()
    io.fail_flush = False

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert not facts.recorded_complete
    assert facts.frame_count == 4
    assert facts.unconfirmed_frames == 1


def test_unflushed_removal_leaves_the_obligation_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    filesystem = writer.intend_filesystem(_filesystem())
    writer.bind_filesystem(filesystem, _filesystem_identity())
    io.fail_flush = True
    with pytest.raises(journal.RunJournalDurabilityError, match="journal append is not durable"):
        writer.record_filesystem_removed(filesystem, _filesystem_identity())
    io.fail_flush = False

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert facts.pending_filesystems == (filesystem,)
    assert facts.unconfirmed_frames == 1


def test_truncation_to_a_frame_boundary_below_the_head_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    filesystem = writer.intend_filesystem(_filesystem())
    writer.bind_filesystem(filesystem, _filesystem_identity())
    boundary = len(io.raw)
    writer.record_filesystem_removed(filesystem, _filesystem_identity())
    writer.record_sequence_complete()
    del io.raw[boundary:]

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert not facts.integrity_complete
    assert not facts.recorded_complete
    assert facts.reason == "head-chain"
    with pytest.raises(ValueError, match="journal cannot be resumed"):
        journal._resume_run_journal(41, _binding(), _location())


def test_forged_frame_beyond_the_head_is_unconfirmed_and_unapplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    filesystem = writer.intend_filesystem(_filesystem())
    confirmed = journal._inspect_run_journal(41, _binding(), _location())
    count, digest = _head(io)
    io.raw.extend(journal._record(count, digest, "filesystem_absent", {"ordinal": filesystem}))

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert facts.unconfirmed_frames == 1
    assert facts.frame_count == confirmed.frame_count == count
    assert facts.pending_filesystems == confirmed.pending_filesystems == (filesystem,)
    with pytest.raises(ValueError, match="journal cannot be resumed"):
        journal._resume_run_journal(41, _binding(), _location())


def test_a_journal_replayed_into_another_location_fails_the_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    _complete(writer)
    foreign = bytes(io.raw)

    other = replace(_location(), journal_file_id=31, parent_file_id=32, marker_file_id=33)
    copied = _FakeIo(foreign)
    copied.fact = replace(copied.fact, file_id=other.journal_file_id)
    monkeypatch.setattr(journal, "_journal_io", lambda: copied)

    facts = journal._inspect_run_journal(41, _binding(), other)
    assert not facts.integrity_complete
    assert not facts.recorded_complete
    assert facts.reason == "head-chain"
    assert facts.frame_count == 0
    with pytest.raises(ValueError, match="journal cannot be resumed"):
        journal._resume_run_journal(41, _binding(), other)


@pytest.mark.parametrize("offset", [0, 4, 36, 68])
def test_any_flipped_head_byte_refuses_the_journal(
    monkeypatch: pytest.MonkeyPatch, offset: int
) -> None:
    io = _FakeIo()
    _new(monkeypatch, io)
    io.raw[journal._HEADER_OFFSET + offset] ^= 1

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert not facts.integrity_complete
    assert not facts.recorded_complete
    assert facts.reason == "head"


def test_failed_head_update_after_a_flushed_frame_leaves_it_unconfirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    io = _FakeIo()
    writer = _new(monkeypatch, io)
    confirmed = journal._inspect_run_journal(41, _binding(), _location())
    io.fail_write_offset = journal._HEADER_OFFSET

    with pytest.raises(journal.RunJournalDurabilityError, match="journal append is not durable"):
        writer.intend_filesystem(_filesystem())
    with pytest.raises(journal.RunJournalDurabilityError, match="journal writer is poisoned"):
        writer.intend_process(_process())
    io.fail_write_offset = None

    facts = journal._inspect_run_journal(41, _binding(), _location())
    assert facts.frame_count == confirmed.frame_count == 1
    assert facts.unconfirmed_frames == 1
    assert not facts.recorded_complete
    with pytest.raises(ValueError, match="journal cannot be resumed"):
        journal._resume_run_journal(41, _binding(), _location())


@pytest.mark.parametrize(
    "value",
    [
        "C:/abs",
        "logs/out.txt:hidden-stream",
        "CON",
        "NUL",
        "COM1",
        "trailing.",
        "trailing ",
    ],
)
def test_relative_artifact_names_refuse_windows_escapes(value: str) -> None:
    with pytest.raises(ValueError, match="journal artifact fact differs"):
        journal._relative(value)


def test_relative_artifact_names_still_accept_an_ordinary_subpath() -> None:
    assert journal._relative("sub/dir/file") == "sub/dir/file"


@pytest.mark.parametrize(
    "value",
    [
        "//server/share/x",
        "//?/C:/x",
        "//./C:/x",
        "C:/private/run.journal.",
        "C:/private/run.journal ",
        "C:/private/run.journal:marker",
    ],
)
def test_absolute_paths_refuse_unc_device_and_stream_spellings(value: str) -> None:
    with pytest.raises(ValueError, match="journal path fact differs"):
        journal._path(value)


def test_absolute_paths_still_accept_the_owned_journal_spelling() -> None:
    assert journal._path("C:/private/run.journal") == "C:/private/run.journal"


@pytest.mark.parametrize("field", ["marker_file_id", "journal_file_id"])
def test_recovery_location_refuses_parent_hardlink_identity(field: str) -> None:
    location = _location()

    with pytest.raises(ValueError, match="journal recovery location differs"):
        replace(location, **{field: location.parent_file_id})


@pytest.mark.parametrize("field", ["marker_file_id", "file_id"])
def test_filesystem_identity_refuses_parent_hardlink_identity(field: str) -> None:
    identity = _filesystem_identity()

    with pytest.raises(ValueError, match="filesystem identity differs"):
        replace(identity, **{field: identity.parent_file_id})


@pytest.mark.parametrize("raw", [b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}'])
def test_strict_object_refuses_nonfinite_json_constants(raw: bytes) -> None:
    with pytest.raises(ValueError, match="journal record constant differs"):
        journal._strict_object(raw)


def test_supplied_location_facts_never_gain_cleanup_or_acceptance_methods() -> None:
    location = _location()
    facts_type = journal.RunJournalFactsV1
    assert not hasattr(location, "cleanup")
    assert not hasattr(location, "recover")
    assert not hasattr(facts_type, "accepted")
    assert not hasattr(facts_type, "qualified")
