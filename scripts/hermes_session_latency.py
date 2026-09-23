"""Measure how quickly Hermes session chat starts speaking, beside today's voice path.

One unattended command decides whether a full Hermes agent turn can be the voice foreground
(ADR 0003, #77):

    uv run python scripts/hermes_session_latency.py

It provisions the pinned Hermes release into the ignored ``.hermes/bench`` cache, runs it
in-process with a throwaway home and API key, and times streaming session chat with the voice
model and effort, with no tools and with the default toolset, on an empty and on a long history.
The same utterances then run through the realtime Codex provider as the baseline.

Credentials: the throwaway Hermes receives only the current Codex access token, as a
non-refreshing credential. It never holds a refresh token, so it cannot rotate the Codex or
Hermes logins, and ``CODEX_HOME`` points at nothing so Hermes cannot import one. Output is
timings and token counts only; no utterance, reply, or identifier is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_COMMIT = "29112bef099274229cadff79cdff7bf7b99c4b77"
_UPSTREAM = "https://github.com/NousResearch/hermes-agent.git"
_CACHE = Path(__file__).resolve().parents[1] / ".hermes" / "bench" / f"hermes-{_COMMIT[:12]}"
_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
_DELTA = "assistant.delta"
_TURN_TIMEOUT_SECONDS = 180
# CodexAppServerStreamingInference's default: text this long is spoken without a sentence end.
_MAX_SEGMENT_CHARS = 1024

# Ordinary conversation that no knowledge route selects, so both paths answer from the model.
_UTTERANCES = (
    "I think I'll take a slow walk after lunch today.",
    "That reminds me, I still haven't watered the tomatoes.",
    "Honestly the weekend went by far too quickly.",
    "Could you keep me company while I tidy the kitchen?",
    "I'm trying to decide between tea and a short nap.",
    "My neighbour's dog keeps visiting our garden.",
    "Let's talk about something calm for a minute.",
    "I finally finished that long book last night.",
    "The light in the evenings is lovely this time of year.",
    "Remind me why rainy mornings feel so slow.",
)


def _history(turns: int) -> list[dict[str, str]]:
    """Return a deterministic synthetic voice conversation of ``turns`` exchanges."""

    messages: list[dict[str, str]] = []
    for index in range(turns):
        messages.append(
            {
                "role": "user",
                "content": f"Earlier, point {index}: we chatted about the garden, the weather, "
                "and which chapter of the book I had reached.",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Noted, point {index}. The garden sounds well tended, the weather "
                "was mild, and you were enjoying the middle chapters.",
            }
        )
    return messages


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _codex_access_token() -> str:
    """Read only the current Codex access token and refuse one that expires soon."""

    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    tokens = json.loads((home / "auth.json").read_text(encoding="utf-8")).get("tokens") or {}
    token = tokens.get("access_token")
    if type(token) is not str or token.count(".") != 2:
        raise RuntimeError("no Codex access token; sign in with the Codex CLI first")
    claims = token.split(".")[1]
    expires = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))["exp"]
    if expires - time.time() < 3600:
        raise RuntimeError("the Codex access token expires within an hour; run codex once")
    return token


def _run(*command: str, cwd: Path, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _provision() -> Path:
    """Install the pinned Hermes the way its installer does, reusing the cache when current."""

    source = _CACHE / "source"
    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").exists():
        _run("git", "init", "-q", cwd=source)
        _run("git", "remote", "add", "origin", _UPSTREAM, cwd=source)
        # The documentation site is not runtime code and exceeds Windows path limits.
        _run("git", "sparse-checkout", "set", "--no-cone", "/*", "!/website/", cwd=source)
    if (
        subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=source, capture_output=True, text=True
        ).stdout.strip()
        != _COMMIT
    ):
        _run("git", "fetch", "-q", "--depth", "1", "origin", _COMMIT, cwd=source)
        _run("git", "checkout", "-q", "--detach", "FETCH_HEAD", cwd=source)
    if _run("git", "rev-parse", "HEAD", cwd=source) != _COMMIT:
        raise RuntimeError("the Hermes cache is not at the pinned commit")
    venv = _CACHE / "venv"
    _run(
        "uv",
        "sync",
        "--extra",
        "all",
        "--locked",
        "--python",
        "3.11",
        "--quiet",
        cwd=source,
        env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(venv)},
    )
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _config(provider: str, model: str, effort: str, tools: str) -> str:
    base_url = _CODEX_BASE_URL if provider == "openai-codex" else _OLLAMA_BASE_URL
    config: dict[str, Any] = {
        "model": {
            "provider": "openai-codex" if provider == "openai-codex" else "custom",
            "base_url": base_url,
            "default": model,
        },
        "agent": {"reasoning_effort": effort},
    }
    if tools == "none":
        config["platform_toolsets"] = {"api_server": []}
    return json.dumps(config)  # JSON is YAML.


def _hermes_arm(
    python: Path, args: argparse.Namespace, tools: str, token: str
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="hermes-latency-") as home:
        Path(home, "config.yaml").write_text(
            _config(args.provider, args.model, args.effort, tools), encoding="utf-8"
        )
        worker = subprocess.run(
            (
                str(python),
                __file__,
                "--worker",
                "--samples",
                str(args.samples),
                "--history-turns",
                str(args.history_turns),
            ),
            env=_worker_environment(home, token),
            capture_output=True,
            text=True,
        )
    if worker.returncode:
        # The utterances are synthetic, so the worker's diagnostics hold no user content.
        raise RuntimeError(f"the Hermes worker failed:\n{worker.stderr[-4000:]}")
    # Hermes may log to stdout; the worker's result is always its last line.
    return [
        {
            "path": f"hermes-tools-{tools}",
            "history_turns": sample["history_turns"],
            "cold": sample["cold"],
            "first_speech_ms": _first_speech_ms(sample["deltas"], sample["completed_ms"]),
            "completed_ms": sample["completed_ms"],
            "input_tokens": sample["input_tokens"],
        }
        for sample in json.loads(worker.stdout.strip().splitlines()[-1])
    ]


def _worker_environment(home: str, token: str) -> dict[str, str]:
    """OS runtime discovery, the isolated homes, and the one promised token; nothing else."""

    from hermes_realtime.providers.codex_app_server import _subscription_environment

    discovery: dict[str, str] = _subscription_environment(os.environ)
    return discovery | {
        "HERMES_HOME": home,
        "CODEX_HOME": str(Path(home, "no-codex-login")),
        "HERMES_LATENCY_ACCESS_TOKEN": token,
    }


@contextlib.contextmanager
def _refusal_evidence(path: str, samples: list[dict[str, Any]], planned: int) -> Iterator[None]:
    """Record why a measurement stopped early: counts and the failure's kind, never its text."""

    failure: str | None = None
    try:
        yield
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        if failure is not None:
            evidence = {"path": path, "completed": len(samples), "planned": planned}
            print(
                "[hermes-latency] " + json.dumps(evidence | {"failure": failure}),
                file=sys.stderr,
                flush=True,
            )


async def _codex_arm(args: argparse.Namespace) -> list[dict[str, Any]]:
    from hermes_realtime.conversation.context import (
        ConversationContextSnapshot,
        ConversationMessage,
    )
    from hermes_realtime.providers.codex_app_server import CodexAppServerStreamingInference

    history = tuple(
        ConversationMessage(role=message["role"], text=message["content"])
        for message in _history(args.history_turns)
    )
    inference = CodexAppServerStreamingInference(model=args.model, effort=args.effort)
    schedule = _schedule(args.samples, args.history_turns)
    samples: list[dict[str, Any]] = []
    try:
        with _refusal_evidence("realtime-codex", samples, len(schedule)):
            for index, (utterance, prior) in enumerate(schedule):
                prompt = (history if prior else ()) + (ConversationMessage("user", utterance),)
                snapshot = ConversationContextSnapshot(
                    revision=index, messages=prompt, active_tasks=()
                )
                started = time.perf_counter()
                first: float | None = None
                async for _ in inference.stream(snapshot, turn_id=f"latency_{index}"):
                    first = first or time.perf_counter()
                if first is None:
                    raise RuntimeError("the Codex baseline produced no speech")
                samples.append(
                    {
                        "path": "realtime-codex",
                        "history_turns": prior,
                        "cold": index == 0,
                        "first_speech_ms": (first - started) * 1000,
                        "completed_ms": (time.perf_counter() - started) * 1000,
                    }
                )
    finally:
        await inference.close()
    return samples


def _schedule(samples: int, history_turns: int) -> list[tuple[str, int]]:
    """One cold turn, then empty and long history interleaved so drift hits both alike."""

    turns = [(_UTTERANCES[0], 0)]
    for index in range(samples):
        utterance = _UTTERANCES[index % len(_UTTERANCES)]
        turns += [(utterance, 0), (utterance, history_turns)]
    return turns


def _worker(args: argparse.Namespace) -> None:
    """Run inside the pinned Hermes environment, with ``HERMES_HOME`` already isolated."""

    token = os.environ.pop("HERMES_LATENCY_ACCESS_TOKEN", "")
    if token:
        from agent.credential_pool import (  # type: ignore[import-not-found]
            AUTH_TYPE_API_KEY,
            SOURCE_MANUAL,
            PooledCredential,
            load_pool,
        )

        # An api_key-typed entry is never refreshed, so no refresh token is ever needed.
        load_pool("openai-codex").add_entry(
            PooledCredential(
                provider="openai-codex",
                id="latency",
                label="latency",
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=SOURCE_MANUAL,
                access_token=token,
                base_url=_CODEX_BASE_URL,
            )
        )
    print(json.dumps(asyncio.run(_measure(args))))


async def _measure(args: argparse.Namespace) -> list[dict[str, Any]]:
    import aiohttp
    from gateway.config import PlatformConfig  # type: ignore[import-not-found]
    from gateway.platforms.api_server import APIServerAdapter  # type: ignore[import-not-found]

    key = secrets.token_urlsafe(32)
    port = _free_port()
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": port, "key": key, "cors_origins": []},
        )
    )
    if not await adapter.connect():
        raise RuntimeError("the pinned Hermes API server did not start")
    base = f"http://127.0.0.1:{port}/api/sessions"
    schedule = _schedule(args.samples, args.history_turns)
    samples: list[dict[str, Any]] = []
    try:
        async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {key}"}) as http:
            with _refusal_evidence("hermes", samples, len(schedule)):
                await _measure_turns(adapter, http, base, schedule, samples)
    finally:
        await adapter.disconnect()
    return samples


async def _measure_turns(
    adapter: Any,
    http: Any,
    base: str,
    schedule: list[tuple[str, int]],
    samples: list[dict[str, Any]],
) -> None:
    for index, (utterance, prior) in enumerate(schedule):
        session_id = f"latency_{index}"
        async with http.post(base, json={"id": session_id}) as created:
            created.raise_for_status()
        if prior:
            db = await adapter._ensure_session_db_async()
            await asyncio.to_thread(db.replace_messages, session_id, _history(prior))
        sample = await asyncio.wait_for(
            _turn(http, f"{base}/{session_id}/chat/stream", utterance),
            _TURN_TIMEOUT_SECONDS,
        )
        samples.append(sample | {"history_turns": prior, "cold": index == 0})


async def _turn(http: Any, url: str, utterance: str) -> dict[str, Any]:
    started = time.perf_counter()
    frames: list[tuple[float, str, str]] = []
    event = ""
    async with http.post(url, json={"message": utterance}) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                elapsed = (time.perf_counter() - started) * 1000
                frames.append((elapsed, event, line.removeprefix("data: ")))
                if event in ("error", "done"):
                    break
    return _turn_sample(frames)


def _turn_sample(frames: list[tuple[float, str, str]]) -> dict[str, Any]:
    """Admit one session-chat stream, given its ``(elapsed ms, event, data)`` frames."""

    if any(event == "error" for _, event, _ in frames):
        raise RuntimeError("Hermes reported an error event")  # Its text may hold content.
    deltas = [(ms, json.loads(data)["delta"]) for ms, event, data in frames if event == _DELTA]
    completions = [(ms, json.loads(data)) for ms, event, data in frames if event == "run.completed"]
    if not deltas or not completions:
        raise RuntimeError("the Hermes stream ended without speech and completion")
    completed_ms, completion = completions[0]
    return {
        "deltas": deltas,
        "completed_ms": completed_ms,
        "input_tokens": (completion.get("usage") or {}).get("input_tokens"),
    }


def _first_speech_ms(deltas: list[tuple[float, str]], completed_ms: float) -> float:
    """When today's voice path would hand the accumulated text its first speakable segment."""

    from hermes_realtime.providers._text_segmentation import first_speakable_sentence_end

    text = ""
    for elapsed, delta in deltas:
        text += delta
        end = first_speakable_sentence_end(text)
        if (end is not None and end <= _MAX_SEGMENT_CHARS) or len(text) >= _MAX_SEGMENT_CHARS:
            return elapsed
    return completed_ms  # The unfinished remainder is spoken when the reply completes.


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return round(ordered[math.ceil(fraction * len(ordered)) - 1])


def _report(samples: list[dict[str, Any]]) -> None:
    """Print one path's summary as soon as it exists, so a later failure loses nothing."""

    cold = next(s for s in samples if s["cold"])
    for history in sorted({s["history_turns"] for s in samples}):
        warm = [s for s in samples if s["history_turns"] == history and not s["cold"]]
        tokens = sorted(t for s in warm if (t := s.get("input_tokens")) is not None)
        row = {
            "path": cold["path"],
            "history_turns": history,
            "n": len(warm),
            "first_speech_ms_p50": _percentile([s["first_speech_ms"] for s in warm], 0.5),
            "first_speech_ms_p90": _percentile([s["first_speech_ms"] for s in warm], 0.9),
            "completed_ms_p50": _percentile([s["completed_ms"] for s in warm], 0.5),
            "input_tokens_p50": tokens[len(tokens) // 2] if tokens else None,
            "cold_first_speech_ms": round(cold["first_speech_ms"]),
        }
        print(json.dumps(row), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=("openai-codex", "ollama"), default="openai-codex")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--history-turns", type=int, default=60)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return
    token = _codex_access_token() if args.provider == "openai-codex" else ""
    python = _provision()
    print(
        json.dumps(
            {
                "hermes_commit": _COMMIT,
                "provider": args.provider,
                "model": args.model,
                "effort": args.effort,
            }
        ),
        flush=True,
    )
    for tools in ("none", "full"):
        _report(_hermes_arm(python, args, tools, token))
    if args.provider == "openai-codex":
        _report(asyncio.run(_codex_arm(args)))


if __name__ == "__main__":
    main()
