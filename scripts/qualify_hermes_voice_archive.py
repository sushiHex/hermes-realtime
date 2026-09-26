"""Qualify the voice companion's storage core against the pinned Hermes (M0 spike).

One unattended command:

    uv run python scripts/qualify_hermes_voice_archive.py

It provisions the baseline Hermes and runs each step in a worker process inside Hermes's own
environment, with this repository's ``src`` on ``sys.path`` and a throwaway home in place of
the user's. Workers use Hermes's real ``SessionDB`` and functions; no model is involved.
Criteria, from the consensus design:

- 2 (deduplication and ownership): sequential, concurrent and overlapping retries leave one
  record per identity; a conflicting payload is refused without mutation; a leased foreign
  turn adds nothing; and each foreign mutation (manual compaction under a compression lock,
  a content-changing and a reordering ``replace_messages``, a display-metadata overwrite of
  the interruption flag, an unleased append, and a deleted session) is detected at the next
  verification with no companion mutation, quarantined durably, and still quarantined after
  a restart;
- 3 (crash and lease safety): a process killed at each commit boundary recovers exactly,
  with no duplicate and no false advance; a restart never matches the stale holder; and a
  refresh that returns False or raises fences work;
- 5 (foreign compaction): detected through the hash chain, both while the companion runs
  and at restart readiness; zero physical foreign compactions is not claimed;
- lineage: a branch or import child naming the archive as its parent, which leaves the
  chain untouched, is refused as ``lineage`` live and at restart, and quarantined;
- count: a foreign UPDATE of ``sessions.message_count`` alone, which Hermes keeps equal to
  the active rows, is refused as ``count`` live and at restart, and quarantined;
- 12 (compatibility): every surface name and signature matches the pin, and the private
  operation stores byte-identical rows to Hermes's own ``append_messages_batch``, with an
  identical fingerprint.

The output is one content-free evidence line: counts and categories only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from real_gate_support import PINNED_HERMES, installed_hermes_identity, provision_pinned_hermes

_PREFIX = "[hermes-voice-archive] "
_RESULT_PREFIX = "[hermes-voice-archive-step] "
_MARKERS = {
    "[voice-archive] ": "archive",
    "[voice-archive-open] ": "open",
    "[voice-archive-lease] ": "lease",
}
_SRC = Path(__file__).resolve().parents[1] / "src"
_CRASHED = 75
_STEP_TIMEOUT_SECONDS = 240
_CONVERSATION = "qualify"
_FOREIGN_KINDS = (
    "compaction",
    "replace_content",
    "replace_reorder",
    "display_flag",
    "unleased_append",
    "delete",
    "rotation",
    "replace_archived",
    "lineage",
    "count",
)
_FOREIGN_CATEGORY = {
    "delete": "missing",
    "rotation": "rotated",
    "lineage": "lineage",
    "count": "count",
}
_CRASH_POINTS = ("before_pending", "after_pending", "in_state", "after_state", "after_promote")
_ENVIRONMENT = (
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "OS",
)
_HOMES = ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "HERMES_HOME", "CODEX_HOME")


def _step(result: dict[str, object], markers: list[str], code: int = 0) -> dict[str, object]:
    return {"exit": code, "markers": markers, "result": result}


def _reopened(quarantine: str) -> dict[str, object]:
    """A restart finds the quarantine it left, refuses, and changes nothing."""
    return _step(
        {"archive": "not_ready", "mutations": 0, "open": "quarantined", "quarantine": quarantine},
        ["archive:not_ready", "open:quarantined"],
    )


def _foreign(category: str) -> list[dict[str, object]]:
    return [
        _step(
            {"companion_mutations": 0, "detected": category, "quarantine": category},
            [f"archive:{category}"],
        ),
        _reopened(category),
    ]


def _crash(point: str, pending: int, recovery: str, inserted: int) -> list[dict[str, object]]:
    return [
        _step({}, [], _CRASHED),
        _step(
            {
                "false_advances": 0,
                "max_per_identity": 1,
                "pending_after": 0,
                "pending_before": pending,
                "recovery": recovery,
                "resend_inserted": inserted,
                "rows": 8,
                "stale_write": "refused",
            },
            [],
        ),
    ]


_EXPECTED: dict[str, list[dict[str, object]]] = {
    "surface": [
        _step(
            {
                "differing_rows": 0,
                "drift_mutations": 0,
                "drift_refusal": "drift",
                "equivalent_rows": 8,
                "fingerprint_matches": 1,
                "message_count_matches": 1,
                "partition_mutations": 0,
                "partition_refusal": "partition",
                "surface_failures": 0,
                "unlisted_refused": 1,
            },
            ["archive:drift"],
        )
    ],
    "dedup": [
        _step(
            {
                "after_foreign_inserted": 2,
                "concurrent_acked": 4,
                "concurrent_inserted": 4,
                "conflict_mutations": 0,
                "conflict_refusal": "conflict",
                "foreign_acquire": "refused",
                "foreign_flush": "refused",
                "foreign_rows_added": 0,
                "identities": 12,
                "max_per_identity": 1,
                "overlap_inserted": 2,
                # Hermes's own rows and the committed fingerprint, before and after three
                # sequential retries: nothing added, nothing changed.
                "sequential_rows_added": 0,
                "sequential_unchanged": 1,
            },
            ["archive:conflict"],
        )
    ],
    **{
        f"foreign_{kind}": _foreign(_FOREIGN_CATEGORY.get(kind, "mismatch"))
        for kind in _FOREIGN_KINDS
    },
    "compaction_at_restart": [
        _step({"inserted": 4}, []),
        _step({"mutated": 1}, []),
        _step(
            {"archive": "not_ready", "mutations": 0, "open": "mismatch", "quarantine": "mismatch"},
            ["archive:not_ready", "open:mismatch"],
        ),
        _reopened("mismatch"),
    ],
    # A child appears while the companion is down: restart readiness refuses it.
    "lineage_at_restart": [
        _step({"inserted": 4}, []),
        _step({"mutated": 1}, []),
        _step(
            {"archive": "not_ready", "mutations": 0, "open": "lineage", "quarantine": "lineage"},
            ["archive:not_ready", "open:lineage"],
        ),
        _reopened("lineage"),
    ],
    # The counter goes stale while the companion is down: restart readiness refuses it.
    "count_at_restart": [
        _step({"inserted": 4}, []),
        _step({"mutated": 1}, []),
        _step(
            {"archive": "not_ready", "mutations": 0, "open": "count", "quarantine": "count"},
            ["archive:not_ready", "open:count"],
        ),
        _reopened("count"),
    ],
    "crash_before_pending": _crash("before_pending", 0, "none", 4),
    "crash_after_pending": _crash("after_pending", 1, "cleared", 4),
    "crash_in_state": _crash("in_state", 1, "cleared", 4),
    "crash_after_state": _crash("after_state", 1, "promoted", 0),
    "crash_after_promote": _crash("after_promote", 0, "none", 0),
    "crash_ambiguous": [
        _step({}, [], _CRASHED),
        _step({"mutated": 1}, []),
        _step(
            {"archive": "not_ready", "mutations": 0, "open": "recovery", "quarantine": "recovery"},
            ["archive:not_ready", "open:recovery"],
        ),
    ],
    # A foreign session already holds the id a new conversation was bound to: never adopted.
    "creation_occupied": [
        _step({"open": "recovery", "quarantine": "recovery"}, ["open:recovery"]),
    ],
    "lease_false": [
        _step(
            {"fenced": 1, "foreign_acquired": 1, "mutations": 0, "refusal": "not_ready"},
            ["archive:not_ready", "lease:lost"],
        )
    ],
    "lease_raise": [
        _step(
            {"fenced": 1, "foreign_acquired": 0, "mutations": 0, "refusal": "not_ready"},
            ["archive:not_ready", "lease:raised"],
        )
    ],
    # Taken before any refresh notices: the lease guard inside the transaction refuses.
    "lease_stolen": [
        _step(
            {"fenced": 1, "foreign_acquired": 1, "mutations": 0, "refusal": "lease_lost"},
            ["archive:lease_lost"],
        )
    ],
    # Negative control: a replace_messages round trip that preserves every value is NOT a
    # foreign mutation, live or at restart. Detecting it would be a canonicalization bug.
    "noop_replace": [
        _step({"detected": "accepted", "inserted": 2, "quarantine": None}, []),
        _step({"inserted": 2, "open": "none", "quarantine": None}, []),
    ],
    # The same process opens again: the fresh holder differs only in its boot nonce, the
    # previous one is released first, and neither a write nor a takeover under it passes.
    "holder_same_process": [
        _step(
            {
                "holders_differ": 1,
                "inserted": 2,
                "lease_owner": "fresh",
                "same_process": 1,
                "stale_acquire": "refused",
                "stale_write": "refused",
            },
            [],
        )
    ],
    # Hermes configured below FULL synchronous is never made ready, and takes no lease.
    "durability_normal": [
        _step({"lease_taken": 0, "level": 1, "open": "durability"}, ["open:durability"]),
    ],
    "cap": [
        _step(
            {
                "capacity_mutations": 0,
                "capacity_refusal": "capacity",
                "over_cap_refusal": "over_cap",
                "quarantine": "over_cap",
                "rows": 4096,
            },
            ["archive:capacity", "open:over_cap"],
        )
    ],
}

# Each scenario's worker steps, run in order, each in a fresh process.
_PLAN: dict[str, list[tuple[str, ...]]] = {
    "surface": [("surface",)],
    "dedup": [("dedup",)],
    **{f"foreign_{kind}": [("foreign", kind), ("reopen",)] for kind in _FOREIGN_KINDS},
    "compaction_at_restart": [("seed",), ("mutate", "compaction"), ("reopen",), ("reopen",)],
    "lineage_at_restart": [("seed",), ("mutate", "lineage"), ("reopen",), ("reopen",)],
    "count_at_restart": [("seed",), ("mutate", "count"), ("reopen",), ("reopen",)],
    **{f"crash_{point}": [("crash", point), ("recover",)] for point in _CRASH_POINTS},
    "crash_ambiguous": [("crash", "after_pending"), ("mutate", "unleased_append"), ("reopen",)],
    "creation_occupied": [("occupied",)],
    "lease_false": [("lease", "false")],
    "lease_raise": [("lease", "raise")],
    "lease_stolen": [("lease", "stolen")],
    "noop_replace": [("noop",), ("verify",)],
    "holder_same_process": [("holders",)],
    "durability_normal": [("durability",)],
    "cap": [("cap",)],
}


def _passed(observed: dict[str, list[dict[str, object]]], hermes: dict[str, object]) -> bool:
    """Only the qualified baseline can pass: evidence from another Hermes qualifies nothing."""
    return hermes.get("baseline") is True and observed == _EXPECTED


def _markers(stdout: str) -> list[str]:
    """The companion's refusal markers in a step's output, as ``kind:category`` only."""

    found: list[str] = []
    for line in stdout.splitlines():
        for prefix, kind in _MARKERS.items():
            if line.startswith(prefix):
                evidence = json.loads(line.removeprefix(prefix))
                found.append(f"{kind}:{evidence.get('refusal') or evidence.get('fence')}")
    return sorted(found)


def _run_step(
    python: Path, home: Path, arguments: tuple[str, ...]
) -> tuple[dict[str, object], dict[str, object]]:
    """Run one worker step in a fresh process; return its observation and measurements."""

    environment = {key: value for key, value in os.environ.items() if key.upper() in _ENVIRONMENT}
    temporary = home / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    environment |= dict.fromkeys(_HOMES, str(home)) | {
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "PYTHONIOENCODING": "utf-8",
    }
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # type: ignore[attr-defined]
    with (home / "step-stderr.log").open("ab") as log:
        completed = subprocess.run(
            (str(python), __file__, "--step", *arguments, "--home", str(home)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=log,
            env=environment,
            timeout=_STEP_TIMEOUT_SECONDS,
            creationflags=flags,
            check=False,
        )
    stdout = completed.stdout.decode("utf-8", "replace")
    results = [
        json.loads(line.removeprefix(_RESULT_PREFIX))
        for line in stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    if len(results) > 1:
        raise RuntimeError("a qualification step reported more than once")
    payload = results[0] if results else {}
    measurements = payload.pop("measurements", {})
    return _step(payload, _markers(stdout), completed.returncode), measurements


def _qualify(python: Path) -> None:
    observed: dict[str, list[dict[str, object]]] = {}
    measurements: dict[str, object] = {}
    evidence: dict[str, object] = {"version": 1}
    with tempfile.TemporaryDirectory(prefix="hermes-voice-archive-") as temporary:
        root = Path(temporary)
        try:
            for scenario, steps in _PLAN.items():
                home = root / scenario
                home.mkdir()
                observed[scenario] = []
                for arguments in steps:
                    step, measured = _run_step(python, home, arguments)
                    observed[scenario].append(step)
                    measurements |= measured
            version = measurements.pop("hermes_version", "")
            identity = installed_hermes_identity(str(version), PINNED_HERMES / "source")
            evidence["hermes"] = identity
            evidence["passed"] = _passed(observed, identity)
        except BaseException as error:
            evidence["failure"] = type(error).__name__
            raise
        finally:
            evidence["measurements"] = measurements
            evidence["scenarios"] = observed
            print(_PREFIX + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)
    if evidence["passed"] is not True:
        raise SystemExit(1)


# --- worker side: runs inside the pinned Hermes environment --------------------------------


def _fixture(
    start: int, stop: int, *, variant: str = "", gap: tuple[int, int] | None = None
) -> tuple[Any, ...]:
    """Synthetic rows only: never user content. ``gap`` goes on the first row."""

    from hermes_realtime.companion.integrity import Identity, VoiceRow

    texts = ("qualification row {}", "synthetic reply {} été", "row {} \U0001d11e ok")
    return tuple(
        VoiceRow(
            identity=Identity(0, seq),
            role="user" if seq % 2 == 0 else "assistant",
            text=texts[seq % 3].format(seq) + variant,
            interrupted=seq % 4 == 3,
            timestamp=1_700_000_000.0 + seq * 0.5,
            gap_before=gap if seq == start else None,
        )
        for seq in range(start, stop)
    )


def _batch(start: int, stop: int, *, variant: str = "") -> Any:
    """A gapless batch covering seq ``start`` to ``stop - 1``."""

    from hermes_realtime.companion.integrity import VoiceBatch

    return VoiceBatch(0, start, stop - 1, _fixture(start, stop, variant=variant))


def _gapped() -> Any:
    """Rows 0-3, user row 6 carrying the overflow gap 4-5, then 7-9: covers 0-9 in 8 rows."""

    from hermes_realtime.companion.integrity import VoiceBatch

    return VoiceBatch(0, 0, 9, (*_fixture(0, 4), *_fixture(6, 10, gap=(4, 5))))


class _Worker:
    def __init__(self, home: Path) -> None:
        sys.path.insert(0, str(_SRC))
        import hermes_state  # type: ignore[import-not-found]

        from hermes_realtime.companion.hermes_compat import HermesArchivePort
        from hermes_realtime.companion.store import CompanionStore

        self.home = home
        self.hermes_state = hermes_state
        self.db = hermes_state.SessionDB(db_path=home / "state.db")
        self.store_path = home / "companion.db"
        self.store = CompanionStore(self.store_path)
        self.port = HermesArchivePort(self.db)

    def archive(self, **options: Any) -> Any:
        from hermes_realtime.companion.archive import VoiceArchive

        return VoiceArchive(self.store, self.port, **options)

    def session_id(self) -> str:
        record = self.store.read(_CONVERSATION)
        assert record is not None
        return record.session_id

    def snapshot(self, session_id: str) -> list[tuple[object, ...]]:
        """Every column of every row, and the session row: the census the checks compare."""

        with contextlib.closing(
            sqlite3.connect(f"{(self.home / 'state.db').as_uri()}?mode=ro", uri=True)
        ) as raw:
            rows = raw.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
            session = raw.execute(
                "SELECT parent_session_id, source, ended_at, end_reason, message_count "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchall()
        return [tuple(row) for row in rows] + [("session", *row) for row in session]

    def identities(self, session_id: str) -> Counter[object]:
        return Counter(
            row[0]
            for row in self.db._execute_write(
                lambda conn: conn.execute(
                    "SELECT platform_message_id FROM messages WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
            )
        )

    def fresh_record(self) -> Any:
        from hermes_realtime.companion.store import CompanionStore

        store = CompanionStore(self.store_path)
        try:
            return store.read(_CONVERSATION)
        finally:
            store.close()

    def message(self, row: Any, conversation_id: str = _CONVERSATION) -> dict[str, object]:
        from hermes_realtime.companion.integrity import platform_message_id, voice_metadata

        return {
            "role": row.role,
            "content": row.text,
            "timestamp": row.timestamp,
            "platform_message_id": platform_message_id(conversation_id, row.identity),
            "display_metadata": voice_metadata(row),
        }


async def _refusal(awaitable: Any) -> str:
    from hermes_realtime.companion.integrity import ArchiveRefusal

    try:
        await awaitable
    except ArchiveRefusal as refusal:
        return refusal.category
    return "accepted"


def _mutate(worker: _Worker, kind: str) -> None:
    """One foreign mutation through Hermes's own write paths."""

    db, session_id = worker.db, worker.session_id()
    if kind == "compaction":
        holder = f"pid={os.getpid()}:compress=foreign"
        if not db.try_acquire_compression_lock(session_id, holder):
            raise RuntimeError("the compression lock was not acquired")
        db.archive_and_compact(
            session_id,
            [{"role": "user", "content": "compacted summary", "timestamp": 1_700_000_100.0}],
            lock_holder=holder,
        )
        db.release_compression_lock(session_id, holder)
    elif kind in ("replace_content", "replace_reorder"):
        messages = db.get_messages(session_id)
        if kind == "replace_content":
            messages[0]["content"] = messages[0]["content"] + " edited"
        else:
            messages[0], messages[2] = messages[2], messages[0]
        db.replace_messages(session_id, messages)
    elif kind == "display_flag":
        interrupted = _fixture(3, 4)[0]
        changed = db.set_latest_matching_message_display_kind(
            session_id,
            role="assistant",
            content=interrupted.text,
            display_kind="voice",
            display_metadata={"voice": {"gen": 0, "interrupted": False, "seq": 3}},
        )
        if changed is not True:
            raise RuntimeError("the display metadata was not overwritten")
    elif kind == "unleased_append":
        db.append_message(session_id, "user", "an unleased foreign row")
    elif kind == "delete":
        if db.delete_session(session_id) is not True:
            raise RuntimeError("the session was not deleted")
    elif kind == "replace_archived":
        # Soft-archives every row and re-inserts identical copies: the live rows alone are
        # unchanged, so only a projection that includes inactive rows can see it.
        db.replace_messages(session_id, db.get_messages(session_id), archive_dropped=True)
    elif kind == "count":
        # A foreign UPDATE of the session counter alone; every row stays as archived.
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                (session_id,),
            )
        )
    elif kind == "lineage":
        # A branch or import child naming the archive as its parent. Hermes changes nothing
        # in the archive itself, so only the lineage check can see it.
        db.create_session("foreign_branch", source="cli", parent_session_id=session_id)
    elif kind == "rotation":
        # What a hygiene rotation stamps on the parent it rotates away from.
        db.end_session(session_id, "compression")
    else:
        raise ValueError("unknown mutation")


async def _surface(worker: _Worker) -> dict[str, object]:
    import hermes_cli  # type: ignore[import-not-found]

    from hermes_realtime.companion.hermes_compat import (
        SURFACE,
        CompatError,
        archive_voice_rows,
        check_shapes,
        check_surface,
        read_projection,
        resolve,
    )
    from hermes_realtime.companion.integrity import (
        EXPECTED_HEADER,
        MAX_ARCHIVE_ROWS,
        VOICE_SOURCE,
        ArchiveRefusal,
        VoiceBatch,
        expected_after,
        genesis,
    )

    failures = check_surface() + check_shapes(worker.db)
    archive = worker.archive()
    await archive.open(_CONVERSATION)
    # Eight rows covering seq 0-9: an overflow gap travels on user row 6.
    rows = _gapped().rows
    await archive.archive(_CONVERSATION, _gapped())
    session_id = worker.session_id()
    # The same fixture through Hermes's own public append, into a session with the same header.
    worker.db.create_session("reference", source=VOICE_SOURCE)
    worker.db.append_messages_batch("reference", [worker.message(row) for row in rows])
    ours_snapshot, reference_snapshot = worker.snapshot(session_id), worker.snapshot("reference")
    ours = [row[2:] for row in ours_snapshot if row[0] != "session"]
    reference = [row[2:] for row in reference_snapshot if row[0] != "session"]
    differing = sum(a != b for a, b in zip(ours, reference, strict=False))
    differing += abs(len(ours) - len(reference))
    # The session's counter, updated as Hermes's own append updates it.
    counts = {row[-1] for row in (ours_snapshot[-1], reference_snapshot[-1])}
    projected = read_projection(worker.db, "reference", MAX_ARCHIVE_ROWS)
    record = worker.store.read(_CONVERSATION)
    precomputed = expected_after(genesis(EXPECTED_HEADER), _CONVERSATION, rows)
    matches = (
        projected is not None
        and record is not None
        and record.committed is not None
        and projected.fingerprint() == precomputed == record.committed.fingerprint
    )
    assert record is not None and record.committed is not None
    # The private operation itself refuses, whole, a batch that leaves a hole in its range.
    holed = VoiceBatch(0, 10, 12, (*_fixture(10, 11), *_fixture(12, 13)))
    before = worker.snapshot(session_id)
    try:
        archive_voice_rows(
            worker.db,
            session_id,
            archive.holder(_CONVERSATION),
            holed,
            record.committed.fingerprint,
            record.committed.fingerprint,
            conversation_id=_CONVERSATION,
            cap=MAX_ARCHIVE_ROWS,
            lease_ttl_seconds=300.0,
        )
        partition = "accepted"
    except ArchiveRefusal as refusal:
        partition = refusal.category
    partition_mutations = int(worker.snapshot(session_id) != before)
    # A Hermes that stores anything but the prediction is refused, and nothing lands.
    original = worker.db._insert_message_rows

    def drifting(conn: Any, target: str, messages: list[dict[str, object]]) -> Any:
        result = original(conn, target, messages)
        conn.execute(
            "UPDATE messages SET display_kind = 'drift' WHERE id = "
            "(SELECT MAX(id) FROM messages WHERE session_id = ?)",
            (target,),
        )
        return result

    before = worker.snapshot(session_id)
    worker.db._insert_message_rows = drifting
    drift = await _refusal(archive.archive(_CONVERSATION, _batch(10, 12)))
    del worker.db._insert_message_rows
    after = worker.snapshot(session_id)
    try:
        resolve("SessionDB.append_messages_batch")
        unlisted = 0
    except CompatError:
        unlisted = 1
    await archive.close()
    return {
        "differing_rows": differing,
        "drift_mutations": int(before != after),
        "drift_refusal": drift,
        "equivalent_rows": len(ours),
        "fingerprint_matches": int(matches),
        "message_count_matches": int(counts == {len(rows)}),
        "partition_mutations": partition_mutations,
        "partition_refusal": partition,
        "surface_failures": len(failures),
        "unlisted_refused": unlisted,
        "measurements": {"hermes_version": hermes_cli.__version__, "surface_names": len(SURFACE)},
    }


async def _dedup(worker: _Worker) -> dict[str, object]:
    archive = worker.archive()
    await archive.open(_CONVERSATION)
    first = _batch(0, 4)
    await archive.archive(_CONVERSATION, first)
    census = worker.snapshot(worker.session_id())
    committed = worker.fresh_record().committed
    for _ in range(3):
        await archive.archive(_CONVERSATION, first)
    sequential_added = len(worker.snapshot(worker.session_id())) - len(census)
    sequential_unchanged = int(
        worker.snapshot(worker.session_id()) == census
        and worker.fresh_record().committed == committed
    )
    concurrent = await asyncio.gather(
        *(archive.archive(_CONVERSATION, _batch(4, 8)) for _ in range(4))
    )
    overlap = await archive.archive(_CONVERSATION, _batch(6, 10))
    session_id = worker.session_id()
    before = worker.snapshot(session_id)
    conflict = await _refusal(archive.archive(_CONVERSATION, _batch(8, 10, variant="!")))
    conflict_mutations = int(worker.snapshot(session_id) != before)
    # A leased foreign agent turn: its flush and its acquisition both meet the companion's lease.
    foreign = f"pid={os.getpid()}:foreign=turn"
    before = worker.snapshot(session_id)
    try:
        worker.db.append_messages_batch(
            session_id,
            [{"role": "user", "content": "a leased foreign turn"}],
            turn_lease_holder=foreign,
        )
        flush = "accepted"
    except worker.hermes_state.SessionTurnLeaseLostError:
        flush = "refused"
    acquired = worker.db.try_acquire_session_turn_lease(session_id, foreign)
    added = len(worker.snapshot(session_id)) - len(before)
    after_foreign = await archive.archive(_CONVERSATION, _batch(10, 12))
    identities = worker.identities(session_id)
    await archive.close()
    return {
        "after_foreign_inserted": after_foreign.inserted,
        "concurrent_acked": len(concurrent),
        "concurrent_inserted": sum(ack.inserted for ack in concurrent),
        "conflict_mutations": conflict_mutations,
        "conflict_refusal": conflict,
        "foreign_acquire": "acquired" if acquired else "refused",
        "foreign_flush": flush,
        "foreign_rows_added": added,
        "identities": len(identities),
        "max_per_identity": max(identities.values()),
        "overlap_inserted": overlap.inserted,
        "sequential_rows_added": sequential_added,
        "sequential_unchanged": sequential_unchanged,
    }


async def _foreign_step(worker: _Worker, kind: str) -> dict[str, object]:
    archive = worker.archive()
    await archive.open(_CONVERSATION)
    await archive.archive(_CONVERSATION, _batch(0, 4))
    _mutate(worker, kind)
    session_id = worker.session_id()
    before = worker.snapshot(session_id)
    detected = await _refusal(archive.archive(_CONVERSATION, _batch(4, 6)))
    after = worker.snapshot(session_id)
    await archive.close()
    record = worker.fresh_record()
    return {
        "companion_mutations": int(before != after),
        "detected": detected,
        "quarantine": record.quarantine,
    }


async def _reopen(worker: _Worker) -> dict[str, object]:
    session_id = worker.session_id()
    before = worker.snapshot(session_id)
    archive = worker.archive()
    opened = await _refusal(archive.open(_CONVERSATION))
    archived = await _refusal(archive.archive(_CONVERSATION, _batch(4, 6)))
    after = worker.snapshot(session_id)
    await archive.close()
    return {
        "archive": archived,
        "mutations": int(before != after),
        "open": opened,
        "quarantine": worker.fresh_record().quarantine,
    }


async def _occupied(worker: _Worker) -> dict[str, object]:
    from hermes_realtime.companion.integrity import EXPECTED_HEADER, genesis
    from hermes_realtime.companion.store import Progress

    worker.store.bind(_CONVERSATION, "voice_occupied", Progress(genesis(EXPECTED_HEADER), None))
    worker.db.create_session("voice_occupied", source="cli")
    opened = await _refusal(worker.archive().open(_CONVERSATION))
    return {"open": opened, "quarantine": worker.fresh_record().quarantine}


async def _seed(worker: _Worker) -> dict[str, object]:
    archive = worker.archive()
    await archive.open(_CONVERSATION)
    ack = await archive.archive(_CONVERSATION, _batch(0, 4))
    await archive.close()
    return {"inserted": ack.inserted}


def _crash_at(worker: _Worker, point: str) -> None:
    def crash() -> None:
        sys.stdout.flush()
        os._exit(_CRASHED)

    store, db = worker.store, worker.db
    if point in ("before_pending", "after_pending"):
        begin = store.begin_pending

        def begin_pending(*arguments: Any) -> None:
            if point == "before_pending":
                crash()
            begin(*arguments)
            crash()

        store.begin_pending = begin_pending  # type: ignore[method-assign]
    elif point == "in_state":
        insert = db._insert_message_rows

        def insert_then_crash(*arguments: Any) -> Any:
            insert(*arguments)
            crash()  # Inside the write transaction: nothing has committed.

        db._insert_message_rows = insert_then_crash
    elif point in ("after_state", "after_promote"):
        promote = store.promote

        def promote_then_crash(*arguments: Any) -> None:
            if point == "after_state":
                crash()
            promote(*arguments)
            crash()

        store.promote = promote_then_crash  # type: ignore[method-assign]
    else:
        raise ValueError("unknown crash point")


async def _crash(worker: _Worker, point: str) -> dict[str, object]:
    archive = worker.archive()
    await archive.open(_CONVERSATION)
    await archive.archive(_CONVERSATION, _batch(0, 4))
    _crash_at(worker, point)
    await archive.archive(_CONVERSATION, _batch(4, 8))
    raise RuntimeError("the crash point was never reached")


async def _recover(worker: _Worker) -> dict[str, object]:
    from hermes_realtime.companion.hermes_compat import read_projection
    from hermes_realtime.companion.integrity import MAX_ARCHIVE_ROWS

    session_id = worker.session_id()
    before = worker.store.read(_CONVERSATION)
    assert before is not None and before.holder is not None
    archive = worker.archive()
    report = await archive.open(_CONVERSATION)

    def false_advance() -> int:
        record = worker.store.read(_CONVERSATION)
        projection = read_projection(worker.db, session_id, MAX_ARCHIVE_ROWS)
        assert record is not None and record.committed is not None and projection is not None
        identities = worker.identities(session_id)
        last = max(
            (tuple(int(part) for part in str(key).split(":")[2:]) for key in identities),
            default=None,
        )
        cursor = record.committed.cursor
        return int(
            record.committed.fingerprint != projection.fingerprint()
            or (cursor.generation, cursor.seq) != last
        )

    advances = false_advance()
    try:
        worker.db.append_messages_batch(
            session_id,
            [{"role": "user", "content": "a write under the stale holder"}],
            turn_lease_holder=before.holder,
        )
        stale_write = "accepted"
    except worker.hermes_state.SessionTurnLeaseLostError:
        stale_write = "refused"
    ack = await archive.archive(_CONVERSATION, _batch(4, 8))
    advances += false_advance()
    identities = worker.identities(session_id)
    final = worker.store.read(_CONVERSATION)
    assert final is not None
    result = {
        "false_advances": advances,
        "max_per_identity": max(identities.values()),
        "pending_after": int(final.pending is not None),
        "pending_before": int(before.pending is not None),
        "recovery": report.recovery,
        "resend_inserted": ack.inserted,
        "rows": sum(identities.values()),
        "stale_write": stale_write,
    }
    await archive.close()
    return result


async def _lease(worker: _Worker, mode: str) -> dict[str, object]:
    # "stolen" keeps the default TTL, so no refresh runs before the next archive.
    archive = worker.archive(lease_ttl_seconds=300.0 if mode == "stolen" else 1.5)
    await archive.open(_CONVERSATION)
    await archive.archive(_CONVERSATION, _batch(0, 4))
    session_id = worker.session_id()
    acquired = False
    await asyncio.sleep(0.2)  # Let the refresh made right after open finish first.
    if mode in ("false", "stolen"):
        # An operator frees the lease and a foreign turn takes it.
        worker.db.release_session_turn_lease(session_id, archive.holder(_CONVERSATION))
        acquired = worker.db.try_acquire_session_turn_lease(
            session_id, f"pid={os.getpid()}:foreign=turn"
        )
    else:

        def failing(*arguments: Any, **options: Any) -> bool:
            raise sqlite3.OperationalError("disk I/O error")

        worker.db.refresh_session_turn_lease = failing
    if mode != "stolen":
        async with asyncio.timeout(15):
            while archive.ready(_CONVERSATION):
                await asyncio.sleep(0.05)
    before = worker.snapshot(session_id)
    refusal = await _refusal(archive.archive(_CONVERSATION, _batch(4, 6)))
    after = worker.snapshot(session_id)
    fenced = int(not archive.ready(_CONVERSATION))
    with contextlib.suppress(Exception):
        await archive.close()
    return {
        "fenced": fenced,
        "foreign_acquired": int(acquired),
        "mutations": int(before != after),
        "refusal": refusal,
    }


async def _noop(worker: _Worker) -> dict[str, object]:
    """Negative control: a value-preserving replace_messages round trip."""

    archive = worker.archive()
    await archive.open(_CONVERSATION)
    await archive.archive(_CONVERSATION, _batch(0, 4))
    session_id = worker.session_id()
    worker.db.replace_messages(session_id, worker.db.get_messages(session_id))
    try:
        result = await archive.archive(_CONVERSATION, _batch(4, 6))
        detected, inserted = "accepted", result.inserted
    except Exception as error:
        detected, inserted = getattr(error, "category", type(error).__name__), 0
    await archive.close()
    return {"detected": detected, "inserted": inserted,
            "quarantine": worker.fresh_record().quarantine}


async def _verify(worker: _Worker) -> dict[str, object]:
    """Restart readiness after the negative control: clean, and still archiving."""

    archive = worker.archive()
    report = await archive.open(_CONVERSATION)
    result = await archive.archive(_CONVERSATION, _batch(6, 8))
    await archive.close()
    return {"inserted": result.inserted, "open": report.recovery,
            "quarantine": worker.fresh_record().quarantine}


async def _holders(worker: _Worker) -> dict[str, object]:
    """Reopen in the same process after a close that left its lease behind."""

    from hermes_realtime.companion.archive import VoiceArchive
    from hermes_realtime.companion.hermes_compat import HermesArchivePort

    leaky = HermesArchivePort(worker.db)
    leaky.release_lease = lambda session_id, holder: None  # type: ignore[method-assign]
    first = VoiceArchive(worker.store, leaky)
    await first.open(_CONVERSATION)
    await first.archive(_CONVERSATION, _batch(0, 4))
    stale = first.holder(_CONVERSATION)
    await first.close()  # The same-process lease stays: Hermes never reclaims its own PID.
    second = worker.archive()
    await second.open(_CONVERSATION)
    fresh = second.holder(_CONVERSATION)
    session_id = worker.session_id()
    owner = worker.db._execute_write(
        lambda conn: conn.execute(
            "SELECT holder FROM session_turn_leases WHERE conversation_id = ?", (session_id,)
        ).fetchone()
    )
    try:
        worker.db.append_messages_batch(
            session_id,
            [{"role": "user", "content": "a write under the stale holder"}],
            turn_lease_holder=stale,
        )
        stale_write = "accepted"
    except worker.hermes_state.SessionTurnLeaseLostError:
        stale_write = "refused"
    stale_acquire = worker.db.try_acquire_session_turn_lease(session_id, stale)
    result = await second.archive(_CONVERSATION, _batch(4, 6))
    await second.close()
    prefix = f"pid={os.getpid()}:voice={_CONVERSATION}:boot="
    return {
        "holders_differ": int(fresh != stale),
        "inserted": result.inserted,
        "lease_owner": "fresh" if owner is not None and owner[0] == fresh else "other",
        "same_process": int(fresh.startswith(prefix) and stale.startswith(prefix)),
        "stale_acquire": "acquired" if stale_acquire else "refused",
        "stale_write": stale_write,
    }


async def _durability(worker: _Worker) -> dict[str, object]:
    """Hermes configured with database.synchronous NORMAL (written before it opened)."""

    from hermes_realtime.companion.hermes_compat import durability_level

    archive = worker.archive()
    opened = await _refusal(archive.open(_CONVERSATION))
    record = worker.fresh_record()
    session_id = None if record is None else record.session_id
    leases = worker.db._execute_write(
        lambda conn: conn.execute(
            "SELECT COUNT(*) FROM session_turn_leases WHERE conversation_id = ?", (session_id,)
        ).fetchone()[0]
    )
    return {"lease_taken": leases, "level": durability_level(worker.db), "open": opened}


async def _cap(worker: _Worker) -> dict[str, object]:
    from hermes_realtime.companion.hermes_compat import read_projection
    from hermes_realtime.companion.integrity import MAX_ARCHIVE_ROWS, MAX_BATCH_ROWS

    archive = worker.archive()
    await archive.open(_CONVERSATION)
    for start in range(0, MAX_ARCHIVE_ROWS, MAX_BATCH_ROWS):
        await archive.archive(_CONVERSATION, _batch(start, start + MAX_BATCH_ROWS))
    session_id = worker.session_id()
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        read_projection(worker.db, session_id, MAX_ARCHIVE_ROWS)
        timings.append((time.perf_counter() - started) * 1000)
    before = worker.snapshot(session_id)
    capacity = await _refusal(
        archive.archive(_CONVERSATION, _batch(MAX_ARCHIVE_ROWS, MAX_ARCHIVE_ROWS + 1))
    )
    capacity_mutations = int(worker.snapshot(session_id) != before)
    await archive.close()
    worker.db.append_message(session_id, "user", "an unleased row past the cap")
    over_cap = await _refusal(worker.archive().open(_CONVERSATION))
    return {
        "capacity_mutations": capacity_mutations,
        "capacity_refusal": capacity,
        "over_cap_refusal": over_cap,
        "quarantine": worker.fresh_record().quarantine,
        "rows": len(before) - 1,
        "measurements": {"verify_ms_at_cap": round(statistics.median(timings), 1)},
    }


async def _run_worker(arguments: list[str], home: Path) -> None:
    step, *options = arguments
    if step == "durability":
        # Hermes reads its database settings when it opens state.db.
        (home / "config.yaml").write_text("database:\n  synchronous: NORMAL\n", encoding="utf-8")
    worker = _Worker(home)
    if step == "surface":
        result = await _surface(worker)
    elif step == "dedup":
        result = await _dedup(worker)
    elif step == "foreign":
        result = await _foreign_step(worker, options[0])
    elif step == "reopen":
        result = await _reopen(worker)
    elif step == "seed":
        result = await _seed(worker)
    elif step == "occupied":
        result = await _occupied(worker)
    elif step == "mutate":
        _mutate(worker, options[0])
        result = {"mutated": 1}
    elif step == "crash":
        result = await _crash(worker, options[0])
    elif step == "recover":
        result = await _recover(worker)
    elif step == "lease":
        result = await _lease(worker, options[0])
    elif step == "cap":
        result = await _cap(worker)
    elif step == "noop":
        result = await _noop(worker)
    elif step == "verify":
        result = await _verify(worker)
    elif step == "holders":
        result = await _holders(worker)
    elif step == "durability":
        result = await _durability(worker)
    else:
        raise ValueError("unknown step")
    print(_RESULT_PREFIX + json.dumps(result, separators=(",", ":"), sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--step", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--home", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.step is not None:
        asyncio.run(_run_worker(args.step, args.home))
    else:
        _qualify(provision_pinned_hermes())


if __name__ == "__main__":
    main()
