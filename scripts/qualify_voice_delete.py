#!/usr/bin/env python3
"""Qualify M3 delete against real pinned Hermes with a declared model stand-in.

    uv run python scripts/qualify_voice_delete.py

Only bounded counts and categories leave the throwaway qualification home.
The succession actors are labelled CLI and gateway processes running the real
VoiceCompanionHost with pinned Hermes; this does not launch the full CLI or API server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import qualify_voice_memory as m4
import qualify_voice_review as m2
from real_gate_support import (
    HERMES_BASELINE,
    PINNED_HERMES,
    installed_hermes_identity,
    provision_pinned_hermes,
)

_PREFIX = "[hermes-voice-delete] "
_STEP_PREFIX = "[hermes-voice-delete-step] "
_SRC = Path(__file__).resolve().parents[1] / "src"
_STEP_TIMEOUT = 240
_EXPECTED = {
    "absence": {
        "tail_phrase": 0,
        "chain_phrase": 0,
        "next_history_phrase": 0,
        "foreground_cleared": 1,
        "branch_copy_pending": 1,
        "state_db_scanned": 1,
        "state_db_phrase": 0,
        "chain_walked": 1,
        "foreign_child_pending": 1,
        "complete_after_child_removed": 1,
    },
    "fences": {
        "late_archive_refused": 1,
        "late_review_refused": 1,
        "stale_ack_ignored": 1,
        "stale_forget_ack_ignored": 1,
        "unsupported_incomplete": 1,
    },
    "running_review": {
        "pending_while_alive": 1,
        "not_cancelled": 1,
        "review_finished": 1,
        "complete_after_finish": 1,
    },
    "recovery": {
        "killed_mid_delete": 1,
        "restart_complete": 1,
        "successor_complete": 1,
        "no_resurrection": 1,
    },
    "succession": {
        "cli_first": 1,
        "gateway_waited": 1,
        "gateway_after_exit": 1,
        "one_owner": 1,
    },
    "learned_limit": {
        "learned_before_delete": 1,
        "open_readback_before_archive": 1,
        "readback_after_delete": 1,
        "model_requests": 2,
    },
    "scope": {
        "delegated_relation": 1,
        "delegated_objective_retained": 1,
        "fk_on": 1,
        "post_delete_child_refused": 1,
        "root_recreation_detected": 1,
    },
    "unattributed": 0,
}


def _passed(observed: object, hermes: object) -> bool:
    """Require the exact pinned identity and a complete, typed witness."""
    if hermes != HERMES_BASELINE | {"baseline": True}:
        return False
    if type(observed) is not dict or set(observed) != set(_EXPECTED):
        return False
    for section, expected in _EXPECTED.items():
        actual = observed[section]
        if type(expected) is dict:
            if type(actual) is not dict or set(actual) != set(expected):
                return False
            if any(
                type(actual[key]) is not int or actual[key] != value
                for key, value in expected.items()
            ):
                return False
        elif type(actual) is not int or actual != expected:
            return False
    return True


def _emit(evidence: dict[str, Any]) -> None:
    print(_PREFIX + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)


async def _worker_step(
    python: Path, home: Path, url: str, token: str, phrase: str, scenario: str
) -> dict[str, object]:
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in m2._ENVIRONMENT
    }
    environment |= dict.fromkeys(m2._HOMES, str(home)) | {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.pathsep.join((str(_SRC), str(PINNED_HERMES / "source"))),
        "M2_MODEL_URL": url,
        "M4_BRIDGE_TOKEN": token,
        "M3_UNIQUE_PHRASE": phrase,
        "TEMP": str(home), "TMP": str(home),
    }
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # type: ignore[attr-defined]
    with (home / "worker-stderr.log").open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            str(python), __file__, "--worker", scenario, "--home", str(home),
            stdin=subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=log,
            env=environment, creationflags=flags,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), _STEP_TIMEOUT)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
    lines = [
        line.removeprefix(_STEP_PREFIX)
        for line in stdout.decode("utf-8", "replace").splitlines()
        if line.startswith(_STEP_PREFIX)
    ]
    if scenario == "crash":
        marker = home / "crash.json"
        if process.returncode != 77 or lines or not marker.is_file():
            raise RuntimeError("delete crash worker did not die after one native deletion")
        result = json.loads(marker.read_text(encoding="utf-8"))
        if type(result) is not dict or set(result) != {"deleted_before_kill", "ids"}:
            raise RuntimeError("delete crash marker is malformed")
        return result
    if process.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"delete worker {scenario} failed without one result")
    result = json.loads(lines[0])
    if type(result) is not dict:
        raise RuntimeError("delete worker result is malformed")
    return result


def _all_messages(db: Any, session_ids: tuple[str, ...]) -> list[dict[str, object]]:
    return [message for session_id in session_ids for message in db.get_messages(session_id)]


def _has_phrase(messages: Any, phrase: str) -> bool:
    return phrase in json.dumps(messages, ensure_ascii=False)


def _scan_state(path: Path, phrase: str) -> dict[str, int]:
    """Count the phrase in every table of a Hermes state.db, FTS shadow tables included.

    A cell matches when it holds the phrase or its random hex, the one token an FTS
    tokenizer keeps whole. FTS segment blobs prefix-compress their terms, so an FTS
    ``MATCH`` for that token is the index's logical check. Raw file bytes (the
    database and its WAL) are counted too: SQLite leaves deleted content in free
    pages and WAL frames until they are reused, so that count is residue, not rows.
    """
    import sqlite3

    token = phrase.rsplit("-", 1)[1]
    needles = (phrase, token)
    bytes_needles = tuple(needle.encode("utf-8") for needle in needles)
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        schema = connection.execute(
            "SELECT name, COALESCE(sql, '') FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        cells = unreadable = matches = fts = 0
        for name, sql in schema:
            quoted = '"' + name.replace('"', '""') + '"'
            try:
                for row in connection.execute(f"SELECT * FROM {quoted}"):
                    for value in row:
                        if type(value) is str:
                            cells += any(needle in value for needle in needles)
                        elif type(value) is bytes:
                            cells += any(needle in value for needle in bytes_needles)
            except sqlite3.Error:
                unreadable += 1
                continue
            if sql.upper().startswith("CREATE VIRTUAL TABLE") and "FTS" in sql.upper():
                fts += 1
                matches += connection.execute(
                    f"SELECT count(*) FROM {quoted} WHERE {quoted} MATCH ?", (f'"{token}"',)
                ).fetchone()[0]
    finally:
        connection.close()
    residue = sum(
        file.read_bytes().count(needle)
        for file in (path, path.with_name(path.name + "-wal"))
        if file.is_file()
        for needle in bytes_needles
    )
    return {
        "tables": len(schema), "fts_tables": fts, "unreadable": unreadable,
        "cells": cells, "fts_matches": matches, "residue": residue,
    }


async def _qualify(python: Path) -> None:
    evidence: dict[str, object] = {"version": 1}
    model = m4._MemoryStandInModel()
    with tempfile.TemporaryDirectory(prefix="hermes-voice-delete-") as temporary:
        root = Path(temporary)
        try:
            url = await model.start()
            token = secrets.token_urlsafe(32)
            phrase = "synthetic-delete-" + secrets.token_hex(16)
            home = root / "core"
            home.mkdir()
            core = await _worker_step(python, home, url, token, phrase, "core")
            running_home = root / "running"
            running_home.mkdir()
            running = await _worker_step(
                python, running_home, url, token, phrase, "running"
            )
            foreign_home = root / "foreign"
            foreign_home.mkdir()
            foreign = await _worker_step(
                python, foreign_home, url, token, phrase, "foreign"
            )
            crash_home = root / "crash"
            crash_home.mkdir()
            crashed = await _worker_step(python, crash_home, url, token, phrase, "crash")
            restarted = await _worker_step(
                python, crash_home, url, token, phrase, "restart"
            )
            again = await _worker_step(
                python, crash_home, url, token, phrase, "restart"
            )
            succession_home = root / "succession"
            succession_home.mkdir()
            succession = await _succession(python, succession_home, url, token)
            absence = core["absence"]
            assert type(absence) is dict
            foreign_result = foreign["foreign"]
            assert type(foreign_result) is dict
            learned = core["learned_limit"]
            assert type(learned) is dict
            learned["model_requests"] = sum(
                count for (case, _model), count in model.calls.items()
                if case == "m2case-correction_a"
            )
            restarted_result = restarted["restart"]
            again_result = again["restart"]
            assert type(restarted_result) is dict and type(again_result) is dict
            observed = {
                "absence": absence | {
                    "foreign_child_pending": foreign_result["pending"],
                    "complete_after_child_removed": foreign_result["complete_after_removal"],
                },
                "fences": core["fences"],
                "running_review": running["running_review"],
                "recovery": {
                    "killed_mid_delete": int(
                        crashed["deleted_before_kill"] == 1
                        and restarted_result["residual_before"] == 1
                    ),
                    "restart_complete": restarted_result["complete"],
                    "successor_complete": succession["gateway_after_exit"],
                    "no_resurrection": again_result["no_resurrection"],
                },
                "succession": succession,
                "learned_limit": learned,
                "scope": core["scope"],
                "unattributed": model.unattributed,
            }
            identity = installed_hermes_identity("0.21.0", PINNED_HERMES / "source")
            evidence["hermes"] = identity
            evidence["observed"] = observed
            evidence["residue"] = core["residue"]
            evidence["passed"] = _passed(observed, identity)
        except BaseException as error:
            evidence["failure"] = type(error).__name__
            evidence["passed"] = False
            raise
        finally:
            _emit(evidence)
            await model.close()
    if evidence["passed"] is not True:
        raise SystemExit(1)


async def _run_worker(scenario: str, home: Path) -> None:
    if scenario == "core":
        result = await _core(home)
    elif scenario == "running":
        result = await _running(home)
    elif scenario == "crash":
        await _crash(home)
        raise RuntimeError("crash worker returned")
    elif scenario == "restart":
        result = await _restart(home)
    elif scenario == "foreign":
        result = await _foreign(home)
    elif scenario in {"host_cli", "host_gateway"}:
        result = _host_actor(home, scenario)
    else:
        raise NotImplementedError("delete worker scenario is not implemented")
    print(_STEP_PREFIX + json.dumps(result, separators=(",", ":"), sort_keys=True), flush=True)


async def _core(home: Path) -> dict[str, object]:
    import sqlite3

    from hermes_realtime.companion.host import VoiceCompanionService
    from hermes_realtime.conversation import ConversationContextStore
    from hermes_realtime.integration import LocalHermesBridgeClient
    from hermes_realtime.integration.voice_archive import (
        VoiceArchiveSender,
        archive_event,
        bridge_connector,
    )
    from hermes_realtime.integration.voice_forget import VoiceForgetSender, forget_connector
    from hermes_realtime.integration.voice_memory import VoiceMemoryReceiver
    from hermes_realtime.integration.voice_tail import VoiceTailWriter
    from hermes_realtime.protocol import (
        VOICE_MEMORY_CAPABILITY,
        VOICE_REVIEW_CAPABILITY,
        VoiceArchiveRefusedEvent,
        VoiceReviewEvent,
        VoiceReviewRefusedEvent,
    )
    from hermes_realtime.speech import Transcript

    m2._config(home, os.environ["M2_MODEL_URL"])
    worker = m2._Worker(home)
    service = VoiceCompanionService(worker.archive, worker.store, worker.port, worker.review)
    ids = iter(("m2case-correction_a", "m3next"))
    writer = VoiceTailWriter(home / "tail.json", conversation_ids=lambda: next(ids))
    context = ConversationContextStore(on_change=writer.update)
    phrase = os.environ["M3_UNIQUE_PHRASE"]
    token = os.environ["M4_BRIDGE_TOKEN"]
    archive_sender: VoiceArchiveSender | None = None
    forget_sender: VoiceForgetSender | None = None
    receiver: VoiceMemoryReceiver | None = None

    async def until(predicate: Any, timeout: float = 30.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("delete qualification did not settle")
            await asyncio.sleep(0.02)

    try:
        await writer.open(context)
        await service.start()
        async with m4._bridge(service) as server:
            archive_sender = VoiceArchiveSender(
                writer,
                bridge_connector(host=server.host, port=server.port, token=token),
            )
            context.record_user_transcript(Transcript(
                text="m2case-correction_a synthetic input " + phrase, final=True,
            ))
            original = archive_event(await writer.next_batch())
            archive_sender.start()
            await until(lambda: writer._cursor == 0)
            old_binding = writer.binding
            record = worker.store.read(old_binding[0])
            if record is None:
                raise RuntimeError("real Hermes archive did not bind")
            parent = record.session_id
            review = VoiceReviewEvent(
                protocol_version="0.3", type="voice_review",
                conversation_id=old_binding[0], generation=old_binding[1],
                seq_from=0, seq_through=0, memory=True, skills=True, closing=True,
            )
            client = await LocalHermesBridgeClient.connect(
                host=server.host, port=server.port, token=token,
                participant_id="voice-delete-review-qualification",
                capabilities=(VOICE_REVIEW_CAPABILITY,),
            )
            try:
                admission = await client.review(review)
            finally:
                await client.close()
            joined = await worker.review.join(old_binding[0], 90.0)
            learned_before = worker.port.read_builtin_memory()
            outcome = worker.review.outcome(old_binding[0], admission.review_id)
            # Two native compression continuations, each with the phrase. Capture
            # their identities before deletion so no orphan can evade the witness.
            child, grandchild = "m3_compression_child", "m3_compression_grandchild"
            worker.db.end_session(parent, "compression")
            worker.db.create_session(child, source="cli", parent_session_id=parent)
            worker.db.append_message(child, "user", phrase)
            worker.db.end_session(child, "compression")
            worker.db.create_session(grandchild, source="cli", parent_session_id=child)
            worker.db.append_message(grandchild, "assistant", phrase)
            chain_ids = (parent, child, grandchild)
            before = _all_messages(worker.db, chain_ids)
            # A Hermes /branch copy of the live segment: an independent conversation
            # Hermes keeps, so the delete stays pending until the user removes it.
            branch = "m3_branch_copy"
            worker.db.create_session(
                branch, source="cli", parent_session_id=grandchild,
                model_config={"_branched_from": grandchild},
            )
            worker.db.append_message(branch, "user", phrase)
            # An unrelated native delegated task belongs to Hermes, outside
            # the voice archive's deletion set.
            task_parent, task_child = "m3_task_parent", "m3_task_delegate"
            task_objective = "Synthetic delegated objective: use river."
            worker.db.create_session(task_parent, source="cli")
            worker.db.create_session(
                task_child, source="delegate", parent_session_id=task_parent,
                model_config={"_delegate_from": task_parent},
            )
            worker.db.append_message(task_child, "user", task_objective)
            task_before = worker.db.get_messages(task_child)
            delegated_relation = int(
                set(worker.db.get_session_delete_targets(task_parent))
                == {task_parent, task_child}
                and task_child not in worker.db.get_session_delete_targets(parent)
            )
            scan_before = _scan_state(home / "state.db", phrase)
            retired = await writer.request_forget(context)
            foreground_cleared = int(
                context.snapshot().messages == ()
                and context.snapshot().memory is None
            )
            tail_after_clear = (home / "tail.json").read_text(encoding="utf-8")
            stale_ack = not writer.acknowledge(
                original.conversation_id, original.generation,
                original.seq_from, original.seq_through,
            )
            await archive_sender.close()
            archive_sender = None
            forget_sender = VoiceForgetSender(
                writer,
                forget_connector(host=server.host, port=server.port, token=token),
                initial_backoff_seconds=0.05, max_backoff_seconds=0.1,
            )
            forget_sender.start()

            def chain_deleted() -> bool:
                deletion = worker.store.deletion(retired[0])
                return (
                    deletion is not None and deletion.targets is not None
                    and worker.port.absent(chain_ids)
                )

            await until(chain_deleted)
            # Several resend rounds pass; the surviving copy keeps the delete pending.
            await asyncio.sleep(0.5)
            branch_pending = int(
                retired in writer.pending_deletes
                and not worker.port.absent((branch,))
                and worker.store.deletion(retired[0]).complete is False
            )
            worker.db.delete_session(branch)
            await until(lambda: not writer.pending_deletes)
            stale_forget_ack = not writer.acknowledge_forget(retired)
            tail_after_complete = (home / "tail.json").read_text(encoding="utf-8")
            scan_after = _scan_state(home / "state.db", phrase)
            late_archive = await service.archive(original)
            late_review = await service.review(review)
            chain_after = _all_messages(worker.db, chain_ids)
            deletion = worker.store.deletion(retired[0])
            task_after = worker.db.get_messages(task_child)
            fk_on = int(worker.db._execute_write(
                lambda conn: conn.execute("PRAGMA foreign_keys").fetchone()[0]
            ) == 1)
            orphan = "m3_post_delete_orphan"
            try:
                worker.db.create_session(orphan, source="cli", parent_session_id=parent)
            except sqlite3.IntegrityError:
                orphan_refused = True
            else:
                orphan_refused = False
            post_delete_child_refused = int(
                fk_on == 1 and orphan_refused and worker.port.absent((orphan,))
            )
            worker.db.create_session(parent, source="cli")
            root_recreation_detected = int(not worker.port.absent((parent,)))
            from hermes_realtime.protocol import VoiceForgetEvent

            replay = await service.forget(VoiceForgetEvent(
                protocol_version="0.3", type="voice_forget",
                conversation_id=retired[0], generation=retired[1],
            ))
            root_recreation_detected *= int(
                replay is not None and replay.type == "voice_forget_ack"
                and replay.state == "complete" and worker.port.absent((parent,))
            )
            async def memory_connect() -> Any:
                return await LocalHermesBridgeClient.connect(
                    host=server.host, port=server.port, token=token,
                    participant_id="voice-delete-memory-qualification",
                    capabilities=(VOICE_MEMORY_CAPABILITY,),
                )

            receiver = VoiceMemoryReceiver(
                context, memory_connect, binding=lambda: writer.binding,
            )
            receiver.start()
            opened_memory = int(await m4._wait_memory(
                context, worker.port.read_builtin_memory(), timeout=10.0,
            ) and "Synthetic correction A: use cobalt." in context.snapshot().memory.memory)
            # The new conversation is archived through the same real bridge.
            # Readback above precedes its first archive; history below follows it.
            context.record_user_transcript(Transcript(text="next synthetic hello", final=True))
            archive_sender = VoiceArchiveSender(
                writer,
                bridge_connector(host=server.host, port=server.port, token=token),
            )
            archive_sender.start()
            await until(lambda: writer._cursor == 0)
            next_record = worker.store.read(writer.binding[0])
            if next_record is None:
                raise RuntimeError("next real Hermes archive did not bind")
            next_history = worker.db.get_messages(next_record.session_id)
            learned_after = int(
                opened_memory and "Synthetic correction A: use cobalt."
                in context.snapshot().memory.memory
            )
            unsupported_writer = VoiceTailWriter(
                home / "unsupported-tail.json",
                conversation_ids=iter(("m3_unsupported", "m3_unsupported_next")).__next__,
            )
            unsupported_context = ConversationContextStore(on_change=unsupported_writer.update)
            await unsupported_writer.open(unsupported_context)
            try:
                unsupported_old = await unsupported_writer.request_forget(unsupported_context)
                async with m4._bridge(None) as unsupported_bridge:
                    unsupported_sender = VoiceForgetSender(
                        unsupported_writer,
                        forget_connector(
                            host=unsupported_bridge.host,
                            port=unsupported_bridge.port,
                            token=token,
                        ),
                        initial_backoff_seconds=0.02,
                        max_backoff_seconds=0.02,
                    )
                    unsupported_sender.start()
                    try:
                        await asyncio.sleep(0.15)
                        unsupported_incomplete = int(
                            unsupported_writer.pending_deletes == (unsupported_old,)
                            and not unsupported_sender.negotiated
                            and unsupported_old[0] in (
                                home / "unsupported-tail.deletes.json"
                            ).read_text(encoding="utf-8")
                        )
                    finally:
                        await unsupported_sender.close()
            finally:
                await unsupported_writer.close()
            return {
                "absence": {
                    "tail_phrase": int(phrase in tail_after_clear or phrase in tail_after_complete),
                    "chain_phrase": int(_has_phrase(chain_after, phrase)),
                    "next_history_phrase": int(_has_phrase(next_history, phrase)),
                    "foreground_cleared": foreground_cleared,
                    "branch_copy_pending": branch_pending,
                    # The scanner sees the phrase before the delete (so it can), in
                    # every readable table, and in no table or FTS index after it.
                    "state_db_scanned": int(
                        scan_before["cells"] > 0 and scan_before["fts_tables"] > 0
                        and scan_before["fts_matches"] > 0
                        and scan_before["unreadable"] == 0 and scan_after["unreadable"] == 0
                        and scan_after["tables"] == scan_before["tables"]
                    ),
                    "state_db_phrase": scan_after["cells"] + scan_after["fts_matches"],
                    "chain_walked": int(
                        _has_phrase(before, phrase) and len(chain_ids) == 3
                        and deletion is not None and deletion.complete
                        and deletion.targets is not None
                        and {target.session_id for target in deletion.targets} == set(chain_ids)
                        and worker.port.absent(chain_ids)
                    ),
                },
                "fences": {
                    "late_archive_refused": int(
                        type(late_archive) is VoiceArchiveRefusedEvent
                        and late_archive.category == "tombstoned"
                    ),
                    "late_review_refused": int(
                        type(late_review) is VoiceReviewRefusedEvent
                        and late_review.category == "tombstoned"
                    ),
                    "stale_ack_ignored": int(stale_ack),
                    "stale_forget_ack_ignored": int(stale_forget_ack),
                    "unsupported_incomplete": unsupported_incomplete,
                },
                "learned_limit": {
                    "learned_before_delete": int(
                        joined and outcome == "finished"
                        and "Synthetic correction A: use cobalt." in learned_before.memory
                    ),
                    "open_readback_before_archive": opened_memory,
                    "readback_after_delete": learned_after,
                    "model_requests": 0,
                },
                "scope": {
                    "delegated_relation": delegated_relation,
                    "delegated_objective_retained": int(
                        _has_phrase(task_before, task_objective)
                        and task_after == task_before
                        and worker.port.absent(chain_ids)
                    ),
                    "fk_on": fk_on,
                    "post_delete_child_refused": post_delete_child_refused,
                    "root_recreation_detected": root_recreation_detected,
                },
                # Reported, not gated: free pages and WAL frames keep bytes until reuse.
                "residue": {"before": scan_before["residue"], "after": scan_after["residue"]},
            }
    finally:
        if receiver is not None:
            await receiver.close()
        if forget_sender is not None:
            await forget_sender.close()
        if archive_sender is not None:
            await archive_sender.close()
        await writer.close()
        await worker.close()


async def _running(home: Path) -> dict[str, object]:
    from hermes_realtime.companion.host import VoiceCompanionService
    from hermes_realtime.protocol import VoiceForgetEvent, VoiceReviewEvent

    m2._config(home, os.environ["M2_MODEL_URL"])
    worker = m2._Worker(home)
    service = VoiceCompanionService(worker.archive, worker.store, worker.port, worker.review)
    cancelled = 0
    native_cancel = worker.port.cancel

    def cancellation_witness(*args: Any, **kwargs: Any) -> None:
        nonlocal cancelled
        cancelled += 1
        native_cancel(*args, **kwargs)

    worker.port.cancel = cancellation_witness
    case = "m2case-busy"
    try:
        await service.start()
        await worker.archive_rows(case, 0, 2)
        review = VoiceReviewEvent(
            protocol_version="0.3", type="voice_review", conversation_id=case,
            generation=0, seq_from=0, seq_through=1, memory=True, skills=True,
            closing=True,
        )
        accepted = await service.review(review)
        active = worker.review._active.get(case)
        alive = active is not None and active.thread.is_alive()
        forget = VoiceForgetEvent(
            protocol_version="0.3", type="voice_forget",
            conversation_id=case, generation=0,
        )
        pending = await service.forget(forget)
        pending_still_live = int(
            accepted is not None and accepted.type == "voice_review_ack"
            and alive and pending is not None and pending.type == "voice_forget_ack"
            and pending.state == "pending"
            and worker.review.admitted(case)
        )
        joined = await worker.review.join(case, 90.0)
        outcome = None if accepted is None else worker.review.outcome(case, accepted.review_id)
        complete = await service.forget(forget)
        return {"running_review": {
            "pending_while_alive": pending_still_live,
            "not_cancelled": int(cancelled == 0),
            "review_finished": int(joined and outcome == "finished"),
            "complete_after_finish": int(
                complete is not None and complete.type == "voice_forget_ack"
                and complete.state == "complete"
            ),
        }}
    finally:
        await worker.close()


async def _crash(home: Path) -> None:
    """Kill the process after native Hermes deletes one frozen compression target."""
    from hermes_realtime.companion.host import VoiceCompanionService
    from hermes_realtime.protocol import VoiceForgetEvent

    m2._config(home, os.environ["M2_MODEL_URL"])
    worker = m2._Worker(home)
    service = VoiceCompanionService(worker.archive, worker.store, worker.port, worker.review)
    case = "m3crash"
    await service.start()
    await worker.archive_rows(case, 0, 2)
    record = worker.store.read(case)
    if record is None:
        raise RuntimeError("crash setup did not bind Hermes")
    parent, child = record.session_id, "m3_crash_compression"
    worker.db.end_session(parent, "compression")
    worker.db.create_session(child, source="cli", parent_session_id=parent)
    worker.db.append_message(child, "user", os.environ["M3_UNIQUE_PHRASE"])
    original = worker.port.delete_target
    deleted = 0

    def die_between_targets(target: Any) -> bool:
        nonlocal deleted
        if deleted:
            (home / "crash.json").write_text(
                json.dumps({"deleted_before_kill": deleted, "ids": [parent, child]}),
                encoding="utf-8",
            )
            os._exit(77)
        result = original(target)
        deleted += int(result)
        return result

    worker.port.delete_target = die_between_targets
    await service.forget(VoiceForgetEvent(
        protocol_version="0.3", type="voice_forget",
        conversation_id=case, generation=0,
    ))
    raise RuntimeError("the injected kill was not reached")


async def _restart(home: Path) -> dict[str, object]:
    from hermes_realtime.companion.host import VoiceCompanionService
    from hermes_realtime.protocol import VoiceForgetEvent

    m2._config(home, os.environ["M2_MODEL_URL"])
    worker = m2._Worker(home)
    service = VoiceCompanionService(worker.archive, worker.store, worker.port, worker.review)
    try:
        marker = json.loads((home / "crash.json").read_text(encoding="utf-8"))
        ids = tuple(marker["ids"])
        residual_before = int(
            not worker.port.absent((ids[0],)) and worker.port.absent((ids[1],))
        )
        await service.start()  # Owner start reconciles the frozen delete manifest.
        deletion = worker.store.deletion("m3crash")
        native_absent = worker.port.absent(ids)
        messages_absent = not _all_messages(worker.db, ids)
        repeat = await service.forget(VoiceForgetEvent(
            protocol_version="0.3", type="voice_forget",
            conversation_id="m3crash", generation=0,
        ))
        return {"restart": {
            "residual_before": residual_before,
            "complete": int(deletion is not None and deletion.complete and native_absent),
            "no_resurrection": int(
                messages_absent and repeat is not None
                and repeat.type == "voice_forget_ack" and repeat.state == "complete"
            ),
        }}
    finally:
        await worker.close()


async def _foreign(home: Path) -> dict[str, object]:
    """Insert a real Hermes child after capture; deletion must refuse the changed set."""
    from hermes_realtime.companion.host import VoiceCompanionService
    from hermes_realtime.protocol import VoiceForgetEvent

    m2._config(home, os.environ["M2_MODEL_URL"])
    worker = m2._Worker(home)
    service = VoiceCompanionService(worker.archive, worker.store, worker.port, worker.review)
    case = "m3foreign"
    child = "m3_foreign_child"
    continuation = "m3_foreign_continuation"
    try:
        await service.start()
        await worker.archive_rows(case, 0, 2)
        record = worker.store.read(case)
        if record is None:
            raise RuntimeError("foreign setup did not bind Hermes")
        worker.db.end_session(record.session_id, "compression")
        worker.db.create_session(
            continuation, source="cli", parent_session_id=record.session_id,
        )
        original = worker.port.capture_delete_targets
        captured = 0

        def mutate_after_capture(
            voice_session_id: str | None, allow_missing_voice: bool,
        ) -> Any:
            nonlocal captured
            targets = original(voice_session_id, allow_missing_voice)
            if captured == 0:
                worker.db.create_session(
                    child, source="cli", parent_session_id=record.session_id,
                )
                worker.db.append_message(child, "user", os.environ["M3_UNIQUE_PHRASE"])
            captured += 1
            return targets

        worker.port.capture_delete_targets = mutate_after_capture
        event = VoiceForgetEvent(
            protocol_version="0.3", type="voice_forget",
            conversation_id=case, generation=0,
        )
        pending = await service.forget(event)
        retained = worker.db.get_messages(child)
        pending_witness = int(
            captured == 1 and pending is not None
            and pending.type == "voice_forget_ack" and pending.state == "pending"
            and _has_phrase(retained, os.environ["M3_UNIQUE_PHRASE"])
        )
        worker.db.delete_session(child)
        complete = await service.forget(event)
        return {"foreign": {
            "pending": pending_witness,
            "complete_after_removal": int(
                complete is not None and complete.type == "voice_forget_ack"
                and complete.state == "complete"
                and worker.port.absent((record.session_id, child, continuation))
            ),
        }}
    finally:
        await worker.close()


def _host_actor(home: Path, role: str) -> dict[str, object]:
    """Use the real companion host in two labelled processes with one profile lock."""
    import hermes_state  # type: ignore[import-not-found]

    from hermes_realtime.companion.hermes_compat import HermesArchivePort
    from hermes_realtime.companion.host import VoiceCompanionHost
    from hermes_realtime.protocol import (
        VoiceArchiveEvent,
        VoiceArchiveRow,
        VoiceForgetEvent,
    )

    m2._config(home, os.environ["M2_MODEL_URL"])

    def open_port() -> HermesArchivePort:
        return HermesArchivePort(hermes_state.SessionDB(db_path=home / "state.db"))

    host = VoiceCompanionHost(
        store_path=home / "companion.db",
        open_port=open_port,
        bridge_factory=m4._bridge,
    )
    if not host.start():
        raise RuntimeError("owner did not start waiting")
    if role == "host_gateway":
        (home / "gateway-waiting.json").write_text('{"waiting":1}', encoding="utf-8")
    try:
        if not host.wait_ready(75.0):
            raise TimeoutError("host did not become ready")
        service = host.service
        if role == "host_cli":
            case = "m3_cli_handoff"
            row = VoiceArchiveRow(
                seq=0, role="user", text="synthetic owner handoff",
                interrupted=False, ts=1_700_000_000.0, gap_before=None,
            )
            archive = host.submit(service.archive(VoiceArchiveEvent(
                protocol_version="0.3", type="voice_archive",
                conversation_id=case, generation=0,
                seq_from=0, seq_through=0, rows=[row],
            )))
            if archive is None or archive.type != "voice_archive_ack":
                raise RuntimeError("CLI owner did not archive")

            def fail_delete(_target: Any) -> bool:
                raise RuntimeError("injected owner exit before native deletion")

            service._port.delete_target = fail_delete
            pending = host.submit(service.forget(VoiceForgetEvent(
                protocol_version="0.3", type="voice_forget",
                conversation_id=case, generation=0,
            )))
            result = {"pending": int(
                pending is not None and pending.type == "voice_forget_ack"
                and pending.state == "pending"
            )}
            (home / "cli-ready.json").write_text(json.dumps(result), encoding="utf-8")
            deadline = time.monotonic() + 75
            while not (home / "cli-stop").is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("CLI owner did not receive exit signal")
                time.sleep(0.05)
            return result
        case = "m3_cli_handoff"
        async def inspect() -> dict[str, int]:
            deletion = service._store.deletion(case)
            record = service._store.read(case)
            return {"complete": int(
                deletion is not None and deletion.complete
                and record is not None and service._port.absent((record.session_id,))
            )}

        result = host.submit(inspect())
        (home / "gateway-ready.json").write_text(json.dumps(result), encoding="utf-8")
        return result
    finally:
        host.close()


async def _succession(
    python: Path, home: Path, url: str, token: str
) -> dict[str, int]:
    environment = {
        key: value for key, value in os.environ.items() if key.upper() in m2._ENVIRONMENT
    }
    environment |= dict.fromkeys(m2._HOMES, str(home)) | {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.pathsep.join((str(_SRC), str(PINNED_HERMES / "source"))),
        "M2_MODEL_URL": url, "M4_BRIDGE_TOKEN": token,
        "TEMP": str(home), "TMP": str(home),
    }
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # type: ignore[attr-defined]

    async def until(
        path: Path, actor: asyncio.subprocess.Process, seconds: float = 75.0
    ) -> None:
        deadline = asyncio.get_running_loop().time() + seconds
        while not path.is_file():
            if actor.returncode is not None:
                raise RuntimeError("owner actor exited before its readiness marker")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("owner handoff did not settle")
            await asyncio.sleep(0.05)

    async def launch(role: str, log: Any) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            str(python), __file__, "--worker", role, "--home", str(home),
            stdin=subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=log,
            env=environment, creationflags=flags,
        )

    cli: asyncio.subprocess.Process | None = None
    gateway: asyncio.subprocess.Process | None = None
    with (home / "cli-stderr.log").open("ab") as cli_log, (
        home / "gateway-stderr.log"
    ).open("ab") as gateway_log:
        try:
            cli = await launch("host_cli", cli_log)
            await until(home / "cli-ready.json", cli)
            cli_ready = json.loads((home / "cli-ready.json").read_text(encoding="utf-8"))
            gateway = await launch("host_gateway", gateway_log)
            await until(home / "gateway-waiting.json", gateway)
            await asyncio.sleep(0.3)
            waited = int(not (home / "gateway-ready.json").exists() and cli.returncode is None)
            (home / "cli-stop").write_text("exit", encoding="utf-8")
            cli_stdout, _ = await asyncio.wait_for(cli.communicate(), 75)
            await until(home / "gateway-ready.json", gateway)
            gateway_stdout, _ = await asyncio.wait_for(gateway.communicate(), 75)
            gateway_ready = json.loads(
                (home / "gateway-ready.json").read_text(encoding="utf-8")
            )
            cli_lines = [line for line in cli_stdout.decode("utf-8", "replace").splitlines()
                         if line.startswith(_STEP_PREFIX)]
            gateway_lines = [
                line for line in gateway_stdout.decode("utf-8", "replace").splitlines()
                if line.startswith(_STEP_PREFIX)
            ]
            exact_exits = int(
                cli.returncode == gateway.returncode == 0
                and len(cli_lines) == len(gateway_lines) == 1
            )
            return {
                "cli_first": int(cli_ready == {"pending": 1}),
                "gateway_waited": waited,
                "gateway_after_exit": int(gateway_ready == {"complete": 1}),
                "one_owner": exact_exits,
            }
        finally:
            for actor in (cli, gateway):
                if actor is not None and actor.returncode is None:
                    actor.kill()
                    await actor.communicate()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--worker", choices=(
            "core", "running", "crash", "restart", "foreign", "host_cli", "host_gateway"
        ),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--home", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reuse-pinned", action="store_true")
    args = parser.parse_args()
    if args.worker is not None:
        if args.home is None:
            parser.error("--worker requires --home")
        asyncio.run(_run_worker(args.worker, args.home))
        return
    python = (
        PINNED_HERMES / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if args.reuse_pinned else provision_pinned_hermes()
    )
    if not python.is_file():
        raise SystemExit("pinned Hermes interpreter is unavailable")
    asyncio.run(_qualify(python))


if __name__ == "__main__":
    main()
