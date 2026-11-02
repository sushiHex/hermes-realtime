"""Opt-in typed client for an already running isolated native preview.

Run as ``python -m scripts.native_typed_journey``. Without --execute this only
validates local inputs; it never bootstraps a session or contacts a provider.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hermes_realtime.client.projection import _EVENT_KINDS, BrowserPublicEvent
from scripts.native_run_witness import RUN_ID, NativeRunWitness, require, validate_origin

LABEL = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
TASK = re.compile(r"task_[A-Za-z0-9][A-Za-z0-9_.:-]{0,122}\Z")
TERMINAL = {"completed", "failed", "interrupted"}


def read_json(path: Path, cap: int = 65_536) -> Any:
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, "journey_file_bound")
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        raise ValueError("journey_json_invalid") from None


def load_scenario(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    require(
        type(value) is dict
        and set(value) == {"version", "steps"}
        and type(value["version"]) is int
        and value["version"] == 1
        and type(value["steps"]) is list
        and 1 <= len(value["steps"]) <= 12,
        "journey_scenario_shape",
    )
    labels, tasks = set(), set()
    for step in value["steps"]:
        require(
            type(step) is dict
            and {"label", "kind"} <= set(step)
            and not set(step) - {"label", "kind", "text", "task", "wait", "tool"},
            "journey_step_shape",
        )
        label, kind = step["label"], step["kind"]
        require(
            type(label) is str
            and LABEL.fullmatch(label) is not None
            and label not in labels
            and kind in {"dispatch", "status", "recall", "cancel"},
            "journey_step_identity",
        )
        labels.add(label)
        if kind == "cancel":
            require(set(step) == {"label", "kind", "task"}, "journey_cancel_shape")
        else:
            require(
                type(step.get("text")) is str and 1 <= len(step["text"].strip()) <= 4096,
                "journey_text_bound",
            )
        if kind == "dispatch":
            require(
                "task" not in step and step.get("wait", "completed") in {"accepted", "completed"},
                "journey_dispatch_shape",
            )
            if "tool" in step:
                require(
                    step["tool"] in {"web_search", "web_extract", "mcp__vram__vram_status"},
                    "journey_expected_tool",
                )
            tasks.add(label)
        else:
            require(
                step.get("task") in tasks and "wait" not in step and "tool" not in step,
                "journey_task_reference",
            )
    return value["steps"]


def read_journal(path: Path) -> tuple[set[str], int]:
    try:
        value = read_json(path)
    except FileNotFoundError:
        # A fresh native adapter has not created its first run record yet.
        # This never supplies an admitted ID or qualifies a dispatch.
        return set(), 0
    require(
        type(value) is dict
        and set(value) == {"version", "pending", "admitted"}
        and type(value["version"]) is int
        and value["version"] in {1, 2}
        and type(value["pending"]) is list
        and type(value["admitted"]) is list
        and len(value["pending"]) + len(value["admitted"]) <= 64,
        "journey_journal_shape",
    )
    ids = value["admitted"]
    require(
        all(type(item) is str and RUN_ID.fullmatch(item) for item in ids)
        and len(set(ids)) == len(ids),
        "journey_journal_identity",
    )
    return set(ids), len(value["pending"])


class EventTracker:
    """Keep identity only in memory; project receipt fields explicitly."""

    def __init__(self) -> None:
        self.cursor = 0
        self.current: dict[str, Any] | None = None
        self.turns: list[dict[str, Any]] = []
        self.pair: tuple[str, int] | None = None
        self.seen: set[tuple[str, int]] = set()
        self.markers: dict[tuple[str, int], dict[str, float]] = {}
        self.tasks: dict[str, str] = {}
        self.text = ""
        self.user_seen = False
        self.admitted = False
        self.started = 0.0

    def begin(self, label: str, sequence: int, text: str) -> None:
        require(len(self.turns) < 12, "journey_turn_bound")
        self.current = {
            "label": label,
            "inputSequence": sequence,
            "outcome": "pending",
            "assistantSegments": 0,
            "firstTokenMs": None,
            "firstPlayableAudioMs": None,
            "tokenToPlayableMs": None,
            "deliveredSegments": 0,
        }
        self.turns.append(self.current)
        self.pair, self.text, self.user_seen, self.admitted = None, text, False, False
        self.started = time.monotonic() * 1000

    def consume(self, batch: object) -> None:
        require(type(batch) is list and len(batch) <= 256, "journey_event_bound")
        for raw in batch:
            require(
                type(raw) is dict and set(raw) == {"sequence", "kind", "monotonicMs", "data"},
                "journey_event_shape",
            )
            seq, kind, clock, data = (raw[k] for k in ("sequence", "kind", "monotonicMs", "data"))
            require(
                type(seq) is int
                and seq == self.cursor + 1
                and kind in _EVENT_KINDS
                and type(clock) in {int, float}
                and math.isfinite(clock)
                and clock >= 0
                and type(data) is dict
                and len(json.dumps(data)) <= 32_768,
                "journey_event_identity",
            )
            event = BrowserPublicEvent(seq, kind, float(clock), data)
            self.cursor = seq
            pair = (data.get("turnId"), data.get("turnGeneration"))
            valid_pair = type(pair[0]) is str and type(pair[1]) is int and pair[1] > 0
            if kind in {"task_state", "task_result"}:
                identity, status = data.get("taskId"), data.get("status")
                require(
                    type(identity) is str
                    and TASK.fullmatch(identity) is not None
                    and status in {"active", "cancelling", *TERMINAL},
                    "journey_task_identity",
                )
                require(identity in self.tasks or len(self.tasks) < 12, "journey_task_bound")
                self.tasks[identity] = status
            if self.current is None:
                if valid_pair:
                    self.seen.add(pair)
                continue
            if kind == "transcript_final" and data.get("role") == "user":
                require(data.get("text") == self.text, "journey_input_source")
                self.user_seen = True
            if kind == "typed_input_admitted":
                require(
                    data.get("inputSequence") == self.current["inputSequence"],
                    "journey_input_sequence",
                )
                self.admitted = True
            if kind in {"first_foreground_token", "first_playable_audio"} and valid_pair:
                require(pair in self.markers or len(self.markers) < 128, "journey_marker_bound")
                self.markers.setdefault(pair, {}).setdefault(kind, event.monotonic_ms)
            if (
                kind == "first_foreground_token"
                and self.pair is None
                and self.user_seen
                and self.admitted
                and valid_pair
                and pair not in self.seen
            ):
                self.pair = pair
            if valid_pair and kind in {
                "assistant_text_generated",
                "assistant_turn_completed",
                "assistant_turn_interrupted",
            }:
                self.seen.add(pair)
                require(len(self.seen) <= 128, "journey_identity_bound")
            if not valid_pair or pair != self.pair:
                continue
            if kind == "assistant_text_generated":
                self.current["assistantSegments"] += 1
            elif kind == "transcript_final" and data.get("role") == "assistant":
                self.current["deliveredSegments"] += 1
            elif kind in {"assistant_turn_completed", "assistant_turn_interrupted"}:
                self.current["outcome"] = kind.removeprefix("assistant_turn_")
            for marker, key in (
                ("first_foreground_token", "firstTokenMs"),
                ("first_playable_audio", "firstPlayableAudioMs"),
            ):
                stamp = self.markers.get(pair, {}).get(marker)
                if stamp is not None and stamp >= self.started:
                    self.current[key] = stamp - self.started
            marks = self.markers.get(pair, {})
            token, playable = marks.get("first_foreground_token"), marks.get("first_playable_audio")
            if token is not None and playable is not None and playable >= token:
                self.current["tokenToPlayableMs"] = playable - token

    def receipt(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.turns]


def request_json(
    origin: str, route: str, *, token: str | None = None, payload: dict | None = None
) -> tuple[int, dict]:
    require(route in {"stable-bootstrap", "refresh", "events", "input", "stop"}, "journey_route")
    body = b"" if payload is None else json.dumps(payload).encode()
    headers = {"Origin": validate_origin(origin), "Content-Length": str(len(body))}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    if body:
        headers["Content-Type"] = "application/json"
    if route == "stable-bootstrap":
        headers["Sec-Fetch-Site"] = "same-origin"

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    req = urllib.request.Request(
        origin.rstrip("/") + "/api/v1/" + route, data=body, headers=headers, method="POST"
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        response = opener.open(req, timeout=8)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read(1_048_577)
        require(len(raw) <= 1_048_576, "journey_http_bound")
        value = json.loads(raw) if raw else {}
        require(type(value) is dict, "journey_http_shape")
        return response.status, value


class Journey:
    def __init__(self, options: argparse.Namespace, credential: dict, bearer: str) -> None:
        self.options, self.credential, self.bearer = options, credential, bearer
        self.tracker = EventTracker()
        self.journal = options.preview_dir / "state/hermes-runs-v1.json"
        self.expiry = time.monotonic() + credential["expiresInSeconds"] / 2
        self.admissions: set[str] = set()
        self.bindings: dict[str, tuple[str, str]] = {}
        self.results: dict[str, dict] = {}
        self.sequence = 0

    async def request(self, route: str, payload: dict | None = None) -> dict:
        code, value = await asyncio.to_thread(
            request_json,
            self.options.browser_origin,
            route,
            token=self.credential["token"],
            payload=payload,
        )
        require(code == (202 if route == "input" else 200), "journey_http_refused")
        return value

    async def poll(self) -> None:
        ids, _ = read_journal(self.journal)
        self.admissions.update(ids)
        require(len(self.admissions) <= 12, "journey_admission_bound")
        if time.monotonic() >= self.expiry:
            value = await self.request("refresh")
            require(
                all(
                    value.get(k) == self.credential[k]
                    for k in ("participantIdentity", "roomName", "workerIdentity")
                ),
                "journey_refresh_binding",
            )
            require(
                type(value.get("token")) is str
                and type(value.get("expiresInSeconds")) is int
                and 1 <= value["expiresInSeconds"] <= 3600,
                "journey_refresh_shape",
            )
            self.credential = value
            self.expiry = time.monotonic() + value["expiresInSeconds"] / 2
        value = await self.request("events", {"after": self.tracker.cursor})
        ids, _ = read_journal(self.journal)
        self.admissions.update(ids)
        require(len(self.admissions) <= 12, "journey_admission_bound")
        require(value.get("version") == 1, "journey_event_version")
        self.tracker.consume(value.get("events"))

    async def wait(self, predicate, seconds: int = 60) -> None:
        async with asyncio.timeout(seconds):
            while not predicate():
                await self.poll()
                await asyncio.sleep(0.05)

    async def collect(self, label: str) -> None:
        task, run = self.bindings[label]
        await self.wait(lambda: self.tracker.tasks.get(task) in TERMINAL, 150)
        value = await asyncio.to_thread(
            NativeRunWitness(
                self.bearer,
                run,
                profile=self.options.native_profile,
                api_origin=self.options.api_origin,
            ).read
        )
        expected = {"completed": "completed", "failed": "failed", "interrupted": "cancelled"}
        require(value["terminal"] == expected[self.tracker.tasks[task]], "journey_terminal_binding")
        self.results[label] = value

    async def step(self, step: dict) -> None:
        kind, label = step["kind"], step["label"]
        if kind == "recall":
            await self.collect(step["task"])
        if kind == "dispatch":
            require(
                all(self.tracker.tasks.get(task) in TERMINAL for task, _ in self.bindings.values()),
                "journey_previous_work_active",
            )
        before_tasks, before_runs = set(self.tracker.tasks), set(self.admissions)
        text = step.get("text")
        if kind == "cancel":
            task, _ = self.bindings[step["task"]]
            require(self.tracker.tasks.get(task) == "active", "journey_cancel_not_active")
            text = "cancel task " + task
        self.sequence += 1
        self.tracker.begin(label, self.sequence, text)
        await self.request("input", {"sequence": self.sequence, "text": text})
        if kind == "dispatch":
            await self.wait(lambda: bool(set(self.tracker.tasks) - before_tasks))
            new_tasks, new_runs = (
                set(self.tracker.tasks) - before_tasks,
                self.admissions - before_runs,
            )
            require(len(new_tasks) == len(new_runs) == 1, "journey_admission_binding")
            self.bindings[label] = (next(iter(new_tasks)), next(iter(new_runs)))
            if step.get("wait", "completed") == "completed":
                await self.collect(label)
                require(self.results[label]["terminal"] == "completed", "journey_task_failed")
            if "tool" in step:
                await self.collect(label)
                require(
                    any(
                        row["tool"] == step["tool"] and row["successfulResults"] > 0
                        for row in self.results[label]["toolCalls"]
                    ),
                    "journey_tool_unproven",
                )
        elif kind == "cancel":
            await self.collect(step["task"])
            require(
                self.results[step["task"]]["terminal"] == "cancelled", "journey_cancel_unproven"
            )
        else:
            await self.wait(lambda: self.tracker.current["outcome"] != "pending")
            require(self.tracker.current["outcome"] == "completed", "journey_turn_incomplete")
            require(
                set(self.tracker.tasks) == before_tasks and self.admissions == before_runs,
                "journey_followup_dispatched",
            )

    async def settle_owned(self) -> bool:
        # Cleanup commands are not another measured foreground turn.
        self.tracker.current = None
        self.tracker.text = ""
        for label, (task, _) in self.bindings.items():
            if self.tracker.tasks.get(task) not in TERMINAL:
                self.sequence += 1
                await self.request(
                    "input", {"sequence": self.sequence, "text": "cancel task " + task}
                )
                await self.wait(lambda task=task: self.tracker.tasks.get(task) in TERMINAL, 45)
                await self.collect(label)
        return read_journal(self.journal) == (set(), 0)


async def execute(options, steps, receipt) -> None:
    from hermes_realtime.host_launcher import _load_api_bearer

    bearer = _load_api_bearer(options.native_profile / ".env")
    from livekit import rtc

    require(
        read_journal(options.preview_dir / "state/hermes-runs-v1.json") == (set(), 0),
        "journey_initial_work",
    )
    credential, journey, room, bootstrap_code = None, None, None, None
    try:
        bootstrap_code, credential = await asyncio.to_thread(
            request_json, options.browser_origin, "stable-bootstrap"
        )
        suffix = hashlib.sha256(
            str(options.preview_dir / "active.private.json").encode()
        ).hexdigest()[:16]
        require(
            bootstrap_code == 200
            and credential.get("workerIdentity") == "worker_preview_" + suffix
            and credential.get("roomName") == "hermes-preview-" + suffix
            and credential.get("url") == options.livekit_url
            and type(credential.get("token")) is str
            and type(credential.get("participantIdentity")) is str
            and type(credential.get("expiresInSeconds")) is int
            and 1 <= credential["expiresInSeconds"] <= 3600,
            "journey_bootstrap_binding",
        )
        journey = Journey(options, credential, bearer)
        room = rtc.Room()
        await asyncio.wait_for(
            room.connect(
                credential["url"],
                credential["token"],
                options=rtc.RoomOptions(auto_subscribe=False),
            ),
            15,
        )
        await journey.poll()
        require(not journey.tracker.tasks and not journey.admissions, "journey_existing_task")
        for step in steps:
            await journey.step(step)
        for label in journey.bindings:
            if label not in journey.results:
                await journey.collect(label)
        require(read_journal(journey.journal) == (set(), 0), "journey_unsettled")
        receipt["outcome"] = "completed"
    finally:
        receipt["cleanup"] = {
            "ownedTasksSettled": journey is None,
            "sessionStopped": False,
            "roomDisconnected": room is None,
        }
        if journey is not None:
            with contextlib.suppress(Exception):
                receipt["cleanup"]["ownedTasksSettled"] = await journey.settle_owned()
            credential = journey.credential
        # Even a mismatched bootstrap may have provisioned a lease. Its returned
        # bearer is authority only to stop that lease at the same browser origin.
        token = credential.get("token") if type(credential) is dict else None
        if bootstrap_code == 200 and type(token) is str and 0 < len(token) <= 16_384:
            with contextlib.suppress(Exception):
                code, _ = await asyncio.to_thread(
                    request_json, options.browser_origin, "stop", token=token
                )
                receipt["cleanup"]["sessionStopped"] = code in {200, 202}
        if room is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(room.disconnect(), 10)
                receipt["cleanup"]["roomDisconnected"] = True
        if journey is not None:
            receipt.update(
                turns=journey.tracker.receipt(),
                tasks=journey.results,
                acceptedDispatches=len(journey.bindings),
            )
        if receipt["outcome"] == "completed":
            require(all(receipt["cleanup"].values()), "journey_cleanup_unsettled")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preview-dir", "native-profile", "scenario", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("browser-origin", "api-origin", "livekit-url"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args(argv)
    receipt: dict[str, Any] = {"version": 1, "outcome": "refused", "executed": False}
    reserved = None
    try:
        for name in ("preview_dir", "native_profile"):
            setattr(options, name, getattr(options, name).resolve(strict=True))
        validate_origin(options.browser_origin)
        validate_origin(options.api_origin)
        require(options.livekit_url.startswith("ws://"), "journey_livekit_scheme")
        validate_origin("http://" + options.livekit_url[5:])
        steps = load_scenario(options.scenario)
        reserved = options.receipt.open("x", encoding="utf-8")
        receipt["stepCount"] = len(steps)
        if options.execute:
            receipt["executed"] = True
            asyncio.run(execute(options, steps, receipt))
        else:
            receipt["outcome"] = "preflight_only"
    except Exception as error:
        category = (
            str(error)
            if isinstance(error, ValueError)
            and re.fullmatch(r"(?:journey|witness)_[a-z_]+", str(error))
            else type(error).__name__
        )
        receipt.update(outcome="refused", failure=category)
    finally:
        raw = json.dumps(receipt, sort_keys=True)
        if reserved is not None:
            with reserved:
                require(len(raw) <= 65_536, "journey_receipt_bound")
                reserved.write(raw + "\n")
        print("[native-typed-journey] " + raw)
    return int(receipt["outcome"] == "refused")


if __name__ == "__main__":
    raise SystemExit(main())
