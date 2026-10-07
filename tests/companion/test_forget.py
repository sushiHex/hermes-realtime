from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_realtime.companion.forget import DeleteTarget, VoiceForgetReconciler
from hermes_realtime.companion.integrity import EXPECTED_HEADER, ArchiveRefusal, genesis
from hermes_realtime.companion.store import (
    MAX_BOUND_CONVERSATIONS,
    CompanionStore,
    Progress,
)


class FakeDeletePort:
    def __init__(self) -> None:
        self.sessions: dict[str, str | None] = {}
        self.deleted: list[str] = []
        self.fail_after: int | None = None
        # Sessions Hermes's /branch copied from a chain session, which its delete keeps.
        self.branches: dict[str, str] = {}

    def capture_delete_targets(
        self, voice_session_id: str | None, allow_missing_voice: bool,
    ) -> tuple[DeleteTarget, ...]:
        roots = (voice_session_id,) if voice_session_id is not None else ()
        targets: list[DeleteTarget] = []
        for root in roots:
            if root not in self.sessions:
                if root == voice_session_id and allow_missing_voice:
                    continue
                raise ArchiveRefusal("missing")
            chain = [root]
            while True:
                children = [sid for sid, parent in self.sessions.items() if parent == chain[-1]]
                if not children:
                    break
                assert len(children) == 1
                chain.append(children[0])
            targets.extend(DeleteTarget(sid) for sid in reversed(chain))
        return tuple(targets)

    def delete_target(self, target: DeleteTarget) -> bool:
        if self.fail_after is not None and len(self.deleted) >= self.fail_after:
            raise RuntimeError("injected native failure")
        if target.session_id not in self.sessions:
            return False
        if any(
            parent == target.session_id and sid.startswith("delegate_")
            for sid, parent in self.sessions.items()
        ):
            raise ArchiveRefusal("lineage")
        self.deleted.append(target.session_id)
        del self.sessions[target.session_id]
        return True

    def absent(self, session_ids: tuple[str, ...]) -> bool:
        return all(sid not in self.sessions for sid in session_ids)

    def branch_copies_absent(self, session_ids: tuple[str, ...]) -> bool:
        return not any(source in session_ids for source in self.branches.values())


@pytest.fixture
def store(tmp_path: Path):
    opened = CompanionStore(tmp_path / "companion.db")
    try:
        yield opened
    finally:
        opened.close()


def test_unbound_tombstone_is_durable_and_fences_binding(store: CompanionStore) -> None:
    assert store.tombstone("conv", 3) is True
    assert store.tombstone("conv", 3) is False
    with pytest.raises(ArchiveRefusal, match="stale"):
        store.tombstone("conv", 4)
    with pytest.raises(ArchiveRefusal, match="tombstoned"):
        store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))


def test_tombstone_rejects_wrong_bound_generation(
    store: CompanionStore,
) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    store.promote("conv", Progress(genesis(EXPECTED_HEADER), None))
    advanced = Progress(genesis(EXPECTED_HEADER), None)
    # A nonempty committed cursor is the authority on the generation.
    from hermes_realtime.companion.integrity import Fingerprint, Identity

    store.begin_pending("conv", advanced, Progress(Fingerprint(1, "1" * 64), Identity(2, 0)))
    store.promote("conv", Progress(Fingerprint(1, "1" * 64), Identity(2, 0)))
    with pytest.raises(ArchiveRefusal, match="stale"):
        store.tombstone("conv", 1)
    assert store.tombstone("conv", 2) is True
    assert store.tombstone("conv", 2) is False
    with pytest.raises(ArchiveRefusal, match="stale"):
        store.tombstone("conv", 3)


def test_delete_manifest_is_immutable_and_bounded(store: CompanionStore) -> None:
    store.tombstone("conv", 0)
    with pytest.raises(TypeError, match="exact DeleteTarget"):
        store.set_delete_manifest("conv", ("voice_1",))  # type: ignore[arg-type]
    first = (DeleteTarget("voice_1"),)
    store.set_delete_manifest("conv", first)
    store.set_delete_manifest("conv", first)
    with pytest.raises(ArchiveRefusal, match="stale"):
        store.set_delete_manifest("conv", (DeleteTarget("voice_2"),))
    with pytest.raises(ValueError, match="delete ids must be a bounded tuple"):
        store.set_delete_manifest(
            "conv", tuple(
                DeleteTarget(f"voice_{number}")
                for number in range(MAX_BOUND_CONVERSATIONS + 1)
            ),
        )
    assert store.deletion("conv") is not None
    assert store.deletion("conv").targets == first  # type: ignore[union-attr]


def test_tombstone_schema_rejects_invalid_generation(tmp_path: Path) -> None:
    path = tmp_path / "companion.db"
    CompanionStore(path).close()
    with sqlite3.connect(path) as raw, pytest.raises(sqlite3.IntegrityError):
        raw.execute(
            "INSERT INTO voice_deletion(conversation_id,generation) VALUES ('conv',-1)"
        )


@pytest.mark.parametrize(
    "document", ['{"voice_1": 0}', '"voice_1"', " " * 600_000 + "[]"],
    ids=["object", "string", "oversized"],
)
def test_persisted_delete_manifest_requires_bounded_list(
    tmp_path: Path, document: str,
) -> None:
    path = tmp_path / "companion.db"
    store = CompanionStore(path)
    try:
        with sqlite3.connect(path) as raw:
            raw.execute(
                "INSERT INTO voice_deletion(conversation_id,generation,manifest) "
                "VALUES ('conv',0,?)", (document,),
            )
        # An unreadable deletion row quarantines its own conversation, and only it.
        with pytest.raises(ArchiveRefusal, match="quarantined"):
            store.deletion("conv")
        with pytest.raises(ArchiveRefusal, match="quarantined"):
            store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
        store.bind("other", "voice_2", Progress(genesis(EXPECTED_HEADER), None))
    finally:
        store.close()


@pytest.mark.asyncio
async def test_a_corrupt_deletion_row_never_stops_the_others_reconciling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "companion.db"
    store = CompanionStore(path)
    try:
        store.bind("good", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
        store.tombstone("good", 0)
        with sqlite3.connect(path) as raw:
            raw.execute(
                "INSERT INTO voice_deletion(conversation_id,generation,manifest) "
                "VALUES ('bad',0,'{}')"
            )
        port = FakeDeletePort()
        port.sessions = {"voice_1": None}
        reconciler = VoiceForgetReconciler(store, port, lambda _conversation: False)

        await reconciler.reconcile_all()

        assert port.deleted == ["voice_1"]
        assert '"refusal": "quarantined"' in capsys.readouterr().out
        with pytest.raises(ArchiveRefusal, match="quarantined"):
            await reconciler.forget("bad", 0)
    finally:
        store.close()


def test_a_completed_delete_frees_its_capacity_so_the_4097th_still_archives_and_deletes(
    store: CompanionStore,
) -> None:
    # Durability pragmas only slow this bookkeeping test; the transactions are unchanged.
    store._connection.execute("PRAGMA synchronous = OFF")
    store._connection.execute("PRAGMA journal_mode = MEMORY")
    creation = Progress(genesis(EXPECTED_HEADER), None)
    for number in range(MAX_BOUND_CONVERSATIONS + 1):
        conversation = f"conv_{number}"
        store.bind(conversation, f"voice_{number}", creation)
        store.tombstone(conversation, 0)
        store.set_delete_manifest(conversation, (DeleteTarget(f"voice_{number}"),))
        store.mark_delete_complete(conversation)
    # Every completed generation stays fenced for this owner's lifetime.
    with pytest.raises(ArchiveRefusal, match="tombstoned"):
        store.bind("conv_0", "voice_x", creation)


def test_an_owned_start_prunes_completed_fences(store: CompanionStore) -> None:
    creation = Progress(genesis(EXPECTED_HEADER), None)
    store.bind("done", "voice_1", creation)
    store.tombstone("done", 0)
    store.set_delete_manifest("done", (DeleteTarget("voice_1"),))
    store.mark_delete_complete("done")
    store.tombstone("open", 0)

    store.prune_completed()

    assert store.deletion("done") is None and store.read("done") is None
    assert store.pending_deletions() == ("open",)


@pytest.mark.asyncio
async def test_delete_manifest_survives_partial_native_delete(store: CompanionStore) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    port = FakeDeletePort()
    port.sessions = {"voice_1": None, "voice_2": "voice_1", "run_1": None}
    reconciler = VoiceForgetReconciler(store, port, lambda _conversation: False)
    port.fail_after = 1
    assert await reconciler.forget("conv", 0) == "pending"
    manifest = store.deletion("conv")
    assert manifest is not None and manifest.targets
    assert port.deleted == ["voice_2"]
    port.fail_after = None
    assert await reconciler.reconcile("conv") == "complete"
    assert port.sessions == {"run_1": None}
    assert await reconciler.forget("conv", 0) == "complete"
    assert port.deleted == ["voice_2", "voice_1"]


@pytest.mark.asyncio
async def test_admitted_review_defers_without_cancellation(
    store: CompanionStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    port = FakeDeletePort()
    port.sessions = {"voice_1": None}
    admitted = True
    reconciler = VoiceForgetReconciler(store, port, lambda _conversation: admitted)
    assert await reconciler.forget("conv", 0) == "pending"
    assert port.deleted == []
    assert (
        '[voice-forget] {"category": "review", "count": 1, "kind": "admitted", '
        '"stage": "defer", "version": 1}'
    ) in capsys.readouterr().out
    admitted = False
    assert await reconciler.reconcile("conv") == "complete"
    assert port.deleted == ["voice_1"]


@pytest.mark.asyncio
async def test_concurrent_forget_and_reconciliation_serialize(store: CompanionStore) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    port = FakeDeletePort()
    port.sessions = {"voice_1": None}
    reconciler = VoiceForgetReconciler(store, port, lambda _conversation: False)
    assert await asyncio.gather(
        reconciler.forget("conv", 0), reconciler.forget("conv", 0)
    ) == ["complete", "complete"]
    assert port.deleted == ["voice_1"]


@pytest.mark.asyncio
async def test_delegate_child_refuses_without_deleting_it_or_voice(
    store: CompanionStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    port = FakeDeletePort()
    port.sessions = {"voice_1": None, "delegate_1": "voice_1"}
    port.capture_delete_targets = lambda _voice, _missing: (DeleteTarget("voice_1"),)  # type: ignore[method-assign]
    reconciler = VoiceForgetReconciler(store, port, lambda _conversation: False)
    assert await reconciler.forget("conv", 0) == "pending"
    assert port.sessions == {"voice_1": None, "delegate_1": "voice_1"}
    assert port.deleted == []
    assert '[voice-forget] {"refusal": "lineage", "stage": "delete", "version": 1}' in (
        capsys.readouterr().out
    )
    del port.sessions["delegate_1"]
    assert await reconciler.reconcile("conv") == "complete"
    port.sessions["voice_1"] = None
    port.fail_after = len(port.deleted)
    assert await reconciler.forget("conv", 0) == "pending"


@pytest.mark.asyncio
async def test_cancelled_caller_keeps_native_capture_owned_until_it_returns(
    store: CompanionStore,
) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    port = FakeDeletePort()
    port.sessions = {"voice_1": None}
    entered, release = threading.Event(), threading.Event()
    original = port.capture_delete_targets

    def blocked(voice: str | None, allow_missing: bool):
        entered.set()
        assert release.wait(5)
        return original(voice, allow_missing)

    port.capture_delete_targets = blocked  # type: ignore[method-assign]
    reconciler = VoiceForgetReconciler(store, port, lambda _conversation: False)
    task = asyncio.create_task(reconciler.forget("conv", 0))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await reconciler.reconcile("conv") == "complete"


def _forget_markers(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("[voice-forget] ")]


@pytest.mark.asyncio
async def test_a_session_still_present_after_delete_is_pending_with_one_marker(
    store: CompanionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    port = FakeDeletePort()
    port.sessions = {"voice_1": None}
    # Native delete reports success, yet the walk still finds the session.
    port.delete_target = lambda target: True  # type: ignore[method-assign]
    reconciler = VoiceForgetReconciler(store, port, lambda _conversation: False)

    assert await reconciler.forget("conv", 0) == "pending"

    assert _forget_markers(capsys.readouterr().out) == [
        '[voice-forget] {"category": "present", "stage": "verify", "version": 1}'
    ]
    deletion = store.deletion("conv")
    assert deletion is not None and not deletion.complete


@pytest.mark.asyncio
async def test_branch_copies_keep_a_delete_pending_until_they_are_gone(
    store: CompanionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    port = FakeDeletePort()
    port.sessions = {"voice_1": None, "copy_1": None}
    port.branches = {"copy_1": "voice_1"}
    reconciler = VoiceForgetReconciler(store, port, lambda _conversation: False)

    assert await reconciler.forget("conv", 0) == "pending"

    # The voice chain is gone; the copy Hermes keeps is not ours to delete.
    assert port.sessions == {"copy_1": None}
    assert _forget_markers(capsys.readouterr().out) == [
        '[voice-forget] {"category": "branch_copies", "stage": "verify", "version": 1}'
    ]
    deletion = store.deletion("conv")
    assert deletion is not None and not deletion.complete
    del port.sessions["copy_1"], port.branches["copy_1"]
    assert await reconciler.reconcile("conv") == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["defer", "capture", "delete", "verify"])
async def test_every_pending_decision_leaves_exactly_one_marker(
    stage: str, store: CompanionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.bind("conv", "voice_1", Progress(genesis(EXPECTED_HEADER), None))
    port = FakeDeletePort()
    port.sessions = {"voice_1": None}

    def broken(*_arguments: object) -> object:
        raise RuntimeError("injected")

    if stage == "capture":
        port.capture_delete_targets = broken  # type: ignore[method-assign]
    elif stage == "delete":
        port.delete_target = broken  # type: ignore[method-assign]
    elif stage == "verify":
        port.absent = broken  # type: ignore[method-assign]
    reconciler = VoiceForgetReconciler(store, port, lambda _conversation: stage == "defer")

    assert await reconciler.forget("conv", 0) == "pending"

    markers = _forget_markers(capsys.readouterr().out)
    assert len(markers) == 1 and f'"stage": "{stage}"' in markers[0]
