"""Qualify the voice archive path end to end against the pinned Hermes (M1).

One unattended command:

    uv run python scripts/qualify_voice_archive_path.py

It provisions the baseline Hermes, then runs two kinds of real process against one
throwaway Hermes home, with this repository's ``src`` on ``sys.path``:

- the **companion**: Hermes's own interpreter calls this plugin's real ``register`` with a
  stand-in plugin context; registration builds the companion, whose owned start binds the
  profile's real ``state.db`` and serves the bridge on a loopback port;
- **realtime**: this repository's interpreter drives a real context store, voice tail and
  archive sender through synthetic speech, in steps that stop, crash and restart it.

The steps: realtime fills its outbox past its bound while the companion is down, freezes a
batch and crashes before sending it; the companion starts; realtime restarts, resends the
frozen batch, loses one acknowledgment on purpose, drains, and overflows again while
connected; the companion unloads; realtime leaves a trailing gap; the companion starts again
and refuses a batch whose rows do not partition its range.

Criterion 1 (archive fidelity): the rows in ``state.db`` equal the frozen rows realtime sent,
in identity, role, text, interruption and timestamp; every closed row is archived or lies in
exactly one gap (a ``gap_before`` of an archived user row, or the tail's trailing gap); and
the malformed batch is refused with no mutation.

Criterion 4 (no negative reads): realtime opens no connection but the companion's bridge
(so 0 reads of the messages route, and no HTTP at all), and every resend is the unchanged
frozen batch, sent only after an unknown outcome. The connection witness is positive
controlled: it must observe realtime's bridge connections.

The output is one content-free evidence line: counts and categories only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import secrets
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from real_gate_support import (
    PINNED_HERMES,
    available_port,
    installed_hermes_identity,
    provision_pinned_hermes,
)

_PREFIX = "[voice-archive-path] "
_RESULT_PREFIX = "[voice-archive-path-step] "
_SRC = Path(__file__).resolve().parents[1] / "src"
_CONVERSATION = "qualify"
_CRASHED = 75
_STEP_TIMEOUT_SECONDS = 240
_READY_TIMEOUT_SECONDS = 120
_OUTBOX_ROWS = 6
_BATCH_ROWS = 3
_PORT_VARIABLE = "HERMES_REALTIME_COMPANION_PORT"
_TOKEN_VARIABLE = "HERMES_REALTIME_COMPANION_TOKEN"
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
_FLAGS = (
    subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    if os.name == "nt"
    else 0
)
# Synthetic speech only: never user content. (role, interrupted) per closed row, in order.
_FILL = (
    ("user", False), ("assistant", False), ("user", False), ("assistant", False),
    ("assistant", True), ("user", False), ("assistant", False), ("user", False),
    ("assistant", False), ("user", False), ("assistant", True), ("assistant", False),
    ("user", False), ("assistant", False),
)
_BURST = (
    ("user", False), ("assistant", False), ("assistant", True), ("user", False),
    ("assistant", False), ("assistant", False), ("user", False), ("assistant", False),
    ("user", False), ("assistant", True),
)
_TRAILING = (("user", False),) + (("assistant", False),) * 7


# --- the verdict (pure; unit-tested) ---------------------------------------------------------


def _sends(sent: list[dict[str, Any]]) -> dict[str, int]:
    """Criterion 4's resend census: a resend must repeat a frozen batch after an unknown."""

    resends = changed = keyed_otherwise = 0
    outcomes: dict[tuple[int, int], tuple[str, str]] = {}
    for record in sent:
        event, outcome = record["event"], record["outcome"]
        key = (event["seq_from"], event["seq_through"])
        body = json.dumps(event, sort_keys=True)
        previous = outcomes.get(key)
        if previous is not None:
            resends += 1
            changed += previous[0] != body
            keyed_otherwise += previous[1] not in ("lost", "unknown")
        outcomes[key] = (body, outcome)
    return {"resends": resends, "resends_changed": changed, "resends_not_after_unknown":
            keyed_otherwise}


def _fidelity(
    truth: list[list[Any]],
    sent: list[dict[str, Any]],
    archived: list[dict[str, Any]],
    tail: dict[str, Any],
) -> dict[str, int]:
    """Criterion 1: archived rows equal the frozen rows; every gap appears exactly once."""

    frozen: dict[int, dict[str, Any]] = {}
    for record in sent:
        for row in record["event"]["rows"]:
            frozen[row["seq"]] = row
    expected = [frozen[seq] for seq in sorted(frozen)]
    covered: dict[int, int] = {}
    gaps = 0
    for row in archived:
        if row["gap_before"] is not None:
            gaps += 1
            first, last = row["gap_before"]
            for seq in range(first, last + 1):
                covered[seq] = covered.get(seq, 0) + 1
    trailing = 0
    if tail["gap"] is not None:
        first, last = tail["gap"]
        trailing = last - first + 1
        for seq in range(first, last + 1):
            covered[seq] = covered.get(seq, 0) + 1
    by_seq = {row["seq"]: row for row in archived}
    lost = mismatched = overlapping = 0
    for seq, (role, text, interrupted) in enumerate(truth):
        row = by_seq.get(seq)
        if row is not None:
            overlapping += seq in covered
            mismatched += (row["role"], row["text"], row["interrupted"]) != (
                role, text, interrupted
            )
        elif covered.get(seq, 0) != 1:
            lost += 1
    return {
        "archived": len(archived),
        "archived_differs_from_frozen": int(archived != expected),
        "discarded": sum(covered.values()),
        "double_gapped": sum(count > 1 for count in covered.values()),
        "gaps": gaps,
        "lost": lost,
        "mismatched": mismatched,
        "outbox_left": len(tail["outbox"]),
        "overlapping": overlapping,
        "phantom": sum(seq >= len(truth) for seq in by_seq) + sum(
            seq >= len(truth) for seq in covered
        ),
        "trailing": trailing,
    }


def _passed(evidence: dict[str, Any], hermes: dict[str, object]) -> bool:
    """Only the qualified baseline can pass, and only with every guard exercised."""

    fidelity, sends, reads = evidence["fidelity"], evidence["sends"], evidence["reads"]
    return (
        hermes.get("baseline") is True
        and fidelity["archived"] > 0
        and fidelity["gaps"] >= 2
        and fidelity["trailing"] > 0
        and all(
            fidelity[key] == 0
            for key in (
                "archived_differs_from_frozen", "double_gapped", "lost", "mismatched",
                "outbox_left", "overlapping", "phantom",
            )
        )
        and sends["resends"] >= 1
        and sends["resends_changed"] == 0
        and sends["resends_not_after_unknown"] == 0
        and evidence["frozen_resent_unchanged"] is True
        and evidence["partition"] == {"category": "partition", "mutations": 0}
        and reads["bridge_connections"] > 0
        and reads["foreign_connections"] == 0
        and reads["http_requests"] == 0
        and reads["messages_route_reads"] == 0
        and evidence["steps"] == _EXPECTED_STEPS
    )


_EXPECTED_STEPS = {
    "fill": _CRASHED,
    "companion_ready": 1,
    "drain": 0,
    "companion_unloaded": 0,
    "trailing": 0,
    "companion_ready_again": 1,
    "settle": 0,
    "companion_unloaded_again": 0,
}


# --- the qualifier ---------------------------------------------------------------------------


def _environment(home: Path, port: int, token: str) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key.upper() in _ENVIRONMENT}
    temporary = home / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    return environment | dict.fromkeys(_HOMES, str(home)) | {
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "PYTHONIOENCODING": "utf-8",
        _PORT_VARIABLE: str(port),
        _TOKEN_VARIABLE: token,
    }


def _run(python: Path, home: Path, port: int, token: str, *arguments: str) -> int:
    with (home / "step-output.log").open("ab") as log:
        completed = subprocess.run(
            (str(python), __file__, "--step", *arguments, "--home", str(home)),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=_environment(home, port, token),
            timeout=_STEP_TIMEOUT_SECONDS,
            creationflags=_FLAGS,
            check=False,
        )
    return completed.returncode


class _Companion:
    """The companion process: detached, its output logged, stopped through a file."""

    def __init__(self, python: Path, home: Path, port: int, token: str) -> None:
        self.home = home
        (home / "ready").unlink(missing_ok=True)
        (home / "stop").unlink(missing_ok=True)
        log = (home / "companion-output.log").open("ab")
        self.process = subprocess.Popen(
            (str(python), __file__, "--step", "companion", "--home", str(home)),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=_environment(home, port, token),
            creationflags=_FLAGS,
        )
        log.close()

    def ready(self) -> int:
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if (self.home / "ready").exists():
                return 1
            if self.process.poll() is not None:
                return 0
            time.sleep(0.1)
        return 0

    def stop(self) -> int:
        (self.home / "stop").write_text("stop", encoding="utf-8")
        try:
            return self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=30)
            return -1


def _archived(hermes_home: Path) -> list[dict[str, Any]]:
    uri = (hermes_home / "plugin-data" / "voice-companion.db").as_uri()
    with contextlib.closing(sqlite3.connect(f"{uri}?mode=ro", uri=True)) as store:
        (session_id,) = store.execute(
            "SELECT session_id FROM voice_archive WHERE conversation_id = ?", (_CONVERSATION,)
        ).fetchone()
    with contextlib.closing(
        sqlite3.connect(f"{(hermes_home / 'state.db').as_uri()}?mode=ro", uri=True)
    ) as state:
        rows = state.execute(
            "SELECT platform_message_id, role, content, timestamp, display_metadata "
            "FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    archived = []
    for identity, role, content, timestamp, metadata in rows:
        voice = json.loads(metadata)["voice"]
        if identity != f"voice:{_CONVERSATION}:{voice['gen']}:{voice['seq']}":
            raise RuntimeError("an archived identity does not match its metadata")
        archived.append(
            {
                "seq": voice["seq"],
                "role": role,
                "text": content,
                "interrupted": voice["interrupted"],
                "ts": timestamp,
                "gap_before": voice["gap_before"],
            }
        )
    return archived


def _qualify(hermes_python: Path) -> None:
    evidence: dict[str, Any] = {"version": 1}
    steps: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="voice-archive-path-") as temporary:
        root = Path(temporary)
        hermes_home, realtime = root / "hermes", root / "realtime"
        hermes_home.mkdir()
        realtime.mkdir()
        port, token = available_port(), secrets.token_urlsafe(32)
        python = Path(sys.executable)
        companion: _Companion | None = None
        try:
            steps["fill"] = _run(python, realtime, port, token, "fill", str(port))
            companion = _Companion(hermes_python, hermes_home, port, token)
            steps["companion_ready"] = companion.ready()
            steps["drain"] = _run(python, realtime, port, token, "drain", str(port))
            steps["companion_unloaded"] = companion.stop()
            steps["trailing"] = _run(python, realtime, port, token, "trailing", str(port))
            companion = _Companion(hermes_python, hermes_home, port, token)
            steps["companion_ready_again"] = companion.ready()
            before = len(_archived(hermes_home))
            steps["settle"] = _run(python, realtime, port, token, "settle", str(port))
            steps["companion_unloaded_again"] = companion.stop()
            companion = None
            archived = _archived(hermes_home)
            truth = json.loads((realtime / "truth.json").read_text(encoding="utf-8"))
            sent = [
                json.loads(line)
                for line in (realtime / "sent.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            tail = json.loads((realtime / "tail.json").read_text(encoding="utf-8"))["archive"]
            frozen = json.loads((realtime / "frozen.json").read_text(encoding="utf-8"))
            partition = json.loads((realtime / "partition.json").read_text(encoding="utf-8"))
            evidence["fidelity"] = _fidelity(truth, sent, archived, tail)
            evidence["sends"] = _sends(sent)
            evidence["frozen_resent_unchanged"] = bool(sent) and sent[0]["event"] == frozen
            evidence["partition"] = {
                "category": partition["category"],
                "mutations": len(archived) - before,
            }
            evidence["reads"] = _reads(realtime)
            evidence["steps"] = steps
            version = (hermes_home / "hermes-version.txt").read_text(encoding="utf-8").strip()
            identity = installed_hermes_identity(version, PINNED_HERMES / "source")
            evidence["hermes"] = identity
            evidence["passed"] = _passed(evidence, identity)
        except BaseException as error:
            evidence["failure"] = type(error).__name__
            evidence["steps"] = steps
            evidence["passed"] = False
            raise
        finally:
            if companion is not None:
                companion.stop()
            print(_PREFIX + json.dumps(evidence, separators=(",", ":"), sort_keys=True), flush=True)
    if evidence["passed"] is not True:
        raise SystemExit(1)


def _reads(realtime: Path) -> dict[str, int]:
    totals = {
        "bridge_connections": 0,
        "foreign_connections": 0,
        "http_requests": 0,
        "messages_route_reads": 0,
    }
    for line in (realtime / "audit.jsonl").read_text(encoding="utf-8").splitlines():
        for key, value in json.loads(line).items():
            totals[key] += value
    return totals


# --- the companion process (Hermes's interpreter) --------------------------------------------


class _State:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir


class _Context:
    """The part of Hermes's PluginContext the plugin uses; everything else is Hermes's own."""

    def __init__(self, data_dir: Path) -> None:
        self.state = _State(data_dir)
        self.unload: list[Any] = []

    @property
    def subagent_lifecycle(self) -> object:
        return object()

    def on_unload(self, callback: Any) -> object:
        self.unload.append(callback)
        return object()


def _companion(home: Path) -> None:
    sys.path.insert(0, str(_SRC))

    import hermes_cli  # type: ignore[import-not-found]

    from hermes_realtime import hermes_plugin

    (home / "hermes-version.txt").write_text(hermes_cli.__version__, encoding="utf-8")
    context = _Context(home / "plugin-data")
    hermes_plugin.register(context)
    companion = hermes_plugin._companion
    if companion is None or not companion.wait_ready(_READY_TIMEOUT_SECONDS):
        raise SystemExit(2)
    (home / "ready").write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 900
    while not (home / "stop").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    for callback in context.unload:
        callback()
    (home / "ready").unlink(missing_ok=True)


# --- the realtime process (this repository's interpreter) ------------------------------------


class _Audit:
    """Counts every TCP connection and HTTP request this process makes (a selector loop,
    so asyncio's connections pass through ``socket.connect``)."""

    def __init__(self, bridge_port: int) -> None:
        self.bridge_port = bridge_port
        self.counts = {
            "bridge_connections": 0,
            "foreign_connections": 0,
            "http_requests": 0,
            "messages_route_reads": 0,
        }

    def __call__(self, event: str, arguments: tuple[Any, ...]) -> None:
        if event == "socket.connect":
            address = arguments[1]
            port = address[1] if isinstance(address, tuple) and len(address) >= 2 else None
            key = "bridge_connections" if port == self.bridge_port else "foreign_connections"
            self.counts[key] += 1
        elif event in ("http.client.connect", "urllib.Request"):
            self.counts["http_requests"] += 1
        elif event == "http.client.send":
            data = arguments[1] if len(arguments) > 1 else b""
            if isinstance(data, bytes) and b"/messages" in data:
                self.counts["messages_route_reads"] += 1


class _RecordingLink:
    """Wrap one bridge connection: record every send and its outcome; optionally lose the
    first acknowledgment (the companion committed it; realtime never sees it)."""

    def __init__(self, inner: Any, log: list[dict[str, Any]], lose_next: list[bool]) -> None:
        self._inner = inner
        self._log = log
        self._lose_next = lose_next

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset(self._inner.capabilities)

    async def archive(self, event: Any) -> Any:
        record: dict[str, Any] = {
            "event": json.loads(event.model_dump_json()), "outcome": "unknown"
        }
        self._log.append(record)
        reply = await self._inner.archive(event)
        if self._lose_next and self._lose_next.pop():
            record["outcome"] = "lost"
            raise ConnectionResetError("the acknowledgment was lost on purpose")
        record["outcome"] = reply.type
        return reply

    async def close(self) -> None:
        await self._inner.close()


def _drive(store: Any, role: str, interrupted: bool, text: str) -> None:
    from hermes_realtime.conversation import AssistantSegmentKey
    from hermes_realtime.speech import AudioFrame, DeliveredSpeechLedger, SpeechChunk, Transcript

    if role == "user":
        store.record_user_transcript(Transcript(text=text, final=True))
        return
    segment = AssistantSegmentKey()
    ledger = DeliveredSpeechLedger()
    chunk = SpeechChunk(
        turn_id="turn_q",
        chunk_id="chunk_q",
        text=text,
        audio=AudioFrame(pcm=b"\x01\x00", sample_rate_hz=16_000, channels=1),
    )
    admission = store.prepare_assistant_text(text, segment=segment, heard_text=text)
    ledger.queue(chunk, admission=admission)
    confirmation = ledger.mark_delivered_confirmed(ledger.mark_started("turn_q", "chunk_q"))
    store.record_assistant_delivery(admission=admission, ledger=ledger, confirmation=confirmation)
    if interrupted:
        store.mark_assistant_segment_interrupted(segment)
    else:
        store.close_assistant_segment(segment)


async def _realtime(home: Path, step: str, port: int, audit: _Audit) -> None:
    # Installed once the loop exists: its own self-pipe connection is not realtime's.
    sys.addaudithook(audit)
    from hermes_realtime.conversation import ConversationContextStore
    from hermes_realtime.integration.bridge import LocalHermesBridgeClient
    from hermes_realtime.integration.voice_archive import VoiceArchiveSender, bridge_connector
    from hermes_realtime.integration.voice_tail import VoiceTailWriter, parse_voice_tail
    from hermes_realtime.protocol import VoiceArchiveEvent, VoiceArchiveRow

    token = os.environ[_TOKEN_VARIABLE]
    tail_path = home / "tail.json"
    truth_path = home / "truth.json"
    truth: list[list[Any]] = (
        json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else []
    )
    sent_path = home / "sent.jsonl"
    writer = VoiceTailWriter(
        tail_path,
        max_outbox_rows=_OUTBOX_ROWS,
        max_batch_rows=_BATCH_ROWS,
        conversation_ids=lambda: _CONVERSATION,
    )
    store = ConversationContextStore(max_messages=16, max_item_chars=256, on_change=writer.update)
    await writer.open(store)

    def speak(pattern: tuple[tuple[str, bool], ...]) -> None:
        for role, interrupted in pattern:
            seq = len(truth)
            text = ("synthetic question {}" if role == "user" else "synthetic reply {} é").format(
                seq
            )
            _drive(store, role, interrupted, text)
            truth.append([role, text, interrupted])

    def archive_state() -> Any:
        raw = tail_path.read_bytes() if tail_path.exists() else b""
        tail = parse_voice_tail(
            raw, max_messages=16, max_item_chars=256, max_outbox_rows=_OUTBOX_ROWS
        )
        return None if tail is None else tail.archive

    def written(state: Any) -> bool:
        """The file holds every row closed so far (not an earlier write)."""
        return state is not None and state.next_seq == len(truth)

    def drained() -> bool:
        state = archive_state()
        return written(state) and not state.rows

    async def until(condition: Any, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise TimeoutError("the realtime step did not settle")
            await asyncio.sleep(0.02)

    log: list[dict[str, Any]] = []
    lose_next = [True] if step == "drain" else []
    connect = bridge_connector(host="127.0.0.1", port=port, token=token)

    async def recording_connect() -> Any:
        return _RecordingLink(await connect(), log, lose_next)

    async def fast(delay: float) -> None:
        await asyncio.sleep(min(delay, 0.2))

    sender = VoiceArchiveSender(writer, recording_connect, reply_timeout=30.0, sleep=fast)
    try:
        if step == "fill":
            speak(_FILL)
            sender.start()  # The companion is down: the batch freezes, and nothing arrives.
            await until(lambda: (state := archive_state()) is not None and state.frozen > 0)
            state = archive_state()
            batch = {
                "conversation_id": state.conversation_id,
                "generation": state.generation,
                "seq_from": state.rows[0].gap_before[0]
                if state.rows[0].gap_before
                else state.rows[0].seq,
                "seq_through": state.rows[state.frozen - 1].seq,
                "rows": [
                    {
                        "seq": row.seq, "role": row.role, "text": row.text,
                        "interrupted": row.interrupted, "ts": row.ts,
                        "gap_before": None if row.gap_before is None else list(row.gap_before),
                    }
                    for row in state.rows[: state.frozen]
                ],
            }
            frozen = VoiceArchiveEvent(type="voice_archive", **batch)
            (home / "frozen.json").write_text(frozen.model_dump_json(), encoding="utf-8")
            truth_path.write_text(json.dumps(truth), encoding="utf-8")
            os._exit(_CRASHED)  # A crash: no close, no final write.
        sender.start()
        if step == "drain":
            await until(drained)
            speak(_BURST)  # In one synchronous burst: overflow while connected.
            await until(drained)
        elif step == "trailing":
            await sender.close()  # The companion is down; the outbox only fills.
            speak(_TRAILING)
            await until(lambda: written(state := archive_state()) and state.gap is not None)
        elif step == "settle":
            await until(drained)
            client = await LocalHermesBridgeClient.connect(
                host="127.0.0.1", port=port, token=token, participant_id="qualify-partition",
                capabilities=("voice_archive",),
            )
            try:
                hole = VoiceArchiveEvent(
                    type="voice_archive",
                    conversation_id=_CONVERSATION,
                    generation=0,
                    seq_from=len(truth),
                    seq_through=len(truth) + 2,
                    rows=[
                        VoiceArchiveRow(seq=len(truth), role="user", text="synthetic hole",
                                        interrupted=False, ts=1.0, gap_before=None),
                        VoiceArchiveRow(seq=len(truth) + 2, role="assistant",
                                        text="synthetic hole", interrupted=False, ts=2.0,
                                        gap_before=None),
                    ],
                )
                reply = await client.archive(hole)
            finally:
                await client.close()
            category = getattr(reply, "category", reply.type)
            (home / "partition.json").write_text(json.dumps({"category": category}), "utf-8")
    finally:
        await sender.close()
        await writer.close()
        truth_path.write_text(json.dumps(truth), encoding="utf-8")
        with sent_path.open("a", encoding="utf-8") as sent:
            for record in log:
                sent.write(json.dumps(record, sort_keys=True) + "\n")


def _realtime_step(home: Path, step: str, port: int) -> None:
    sys.path.insert(0, str(_SRC))
    audit = _Audit(port)
    try:
        if os.name == "nt":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        asyncio.run(_realtime(home, step, port, audit))
    finally:
        with (home / "audit.jsonl").open("a", encoding="utf-8") as record:
            record.write(json.dumps(audit.counts) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--step", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--home", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.step is None:
        _qualify(provision_pinned_hermes())
    elif args.step[0] == "companion":
        _companion(args.home)
    else:
        step, port = args.step
        _realtime_step(args.home, step, int(port))


if __name__ == "__main__":
    main()
