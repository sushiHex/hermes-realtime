"""Shared credential, identity and loopback helpers for installed-boundary gates."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from collections import defaultdict
from pathlib import Path

# The owner-selected qualification baseline (#67, #159): a reference, not a version ceiling.
HERMES_BASELINE = {"version": "0.21.0", "commit": "29112bef099274229cadff79cdff7bf7b99c4b77"}
_UPSTREAM = "https://github.com/NousResearch/hermes-agent.git"
PINNED_HERMES = (
    Path(__file__).resolve().parents[1]
    / ".hermes"
    / "bench"
    / f"hermes-{HERMES_BASELINE['commit'][:12]}"
)


def provision_pinned_hermes() -> Path:
    """Install the baseline Hermes the way its installer does, reusing the cache when current.

    Returns the interpreter of its environment; the checkout is ``PINNED_HERMES / "source"``.
    """

    source = PINNED_HERMES / "source"
    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").exists():
        _git(source, "init", "-q")
        _git(source, "remote", "add", "origin", _UPSTREAM)
        # The documentation site is not runtime code and exceeds Windows path limits.
        _git(source, "sparse-checkout", "set", "--no-cone", "/*", "!/website/")
    head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=source, capture_output=True, text=True)
    if head.stdout.strip() != HERMES_BASELINE["commit"]:
        _git(source, "fetch", "-q", "--depth", "1", "origin", HERMES_BASELINE["commit"])
        _git(source, "checkout", "-q", "--detach", "FETCH_HEAD")
    if _git(source, "rev-parse", "HEAD").strip() != HERMES_BASELINE["commit"]:
        raise RuntimeError("the Hermes cache is not at the pinned commit")
    venv = PINNED_HERMES / "venv"
    subprocess.run(
        ("uv", "sync", "--extra", "all", "--locked", "--python", "3.11", "--quiet"),
        cwd=source,
        env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(venv)},
        check=True,
        capture_output=True,
    )
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def load_api_key(env_file: Path) -> str:
    """Load exactly one strong API server bearer without logging it."""

    if not isinstance(env_file, Path):
        raise TypeError("Hermes env file must be a Path")
    matches: list[str] = []
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("API_SERVER_KEY="):
            matches.append(line.split("=", 1)[1].strip().strip('"').strip("'"))
    if len(matches) != 1 or len(matches[0]) < 32:
        raise RuntimeError("Hermes env file must contain one explicit strong API_SERVER_KEY")
    return matches[0]


def installed_hermes_identity(version: str, checkout: Path) -> dict[str, object]:
    """Name the exact Hermes a gate ran against, and whether it is the qualified baseline.

    Evidence binds to a commit, so a checkout whose tracked files differ from every commit is
    refused. Untracked files, such as the installer's virtual environment, do not change it.
    """

    refusal: str | None = "unidentifiable"
    try:
        commit = _git(checkout, "rev-parse", "HEAD").strip()
        refusal = "modified"
        status = _git(checkout, "status", "--porcelain", "-z", "--untracked-files=no")
        changed = {entry[3:] for entry in status.split("\0") if entry}
        if changed - _unrepresentable_paths(checkout):
            raise RuntimeError("installed Hermes has local changes that no commit describes")
        refusal = None
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "installed Hermes is not a git checkout, so no commit describes it"
        ) from None
    finally:
        if refusal is not None:
            evidence = json.dumps({"refusal": refusal, "version": 1}, separators=(",", ":"))
            print(f"[hermes-identity] {evidence}", flush=True)
    identity = {"version": version, "commit": commit}
    return identity | {"baseline": identity == HERMES_BASELINE}


def _unrepresentable_paths(checkout: Path) -> set[str]:
    """Committed paths a case-insensitive checkout cannot hold apart, left exactly as committed.

    Such spellings share one file on disk, so git reports all but one as changed. They are
    unchanged when that one file is byte-identical to one of their committed versions.
    """

    ignorecase = _git(checkout, "config", "--type=bool", "--default=false", "core.ignorecase")
    if ignorecase.strip() != "true":
        return set()
    spellings: dict[str, dict[str, str]] = defaultdict(dict)
    for entry in _git(checkout, "ls-tree", "-r", "-z", "--full-tree", "HEAD").split("\0"):
        if entry:
            header, path = entry.split("\t", 1)
            spellings[path.casefold()][path] = header.split()[2]
    exempt: set[str] = set()
    for committed in spellings.values():
        if len(committed) > 1:
            on_disk = _git(checkout, "hash-object", "--", next(iter(committed))).strip()
            if on_disk in committed.values():
                exempt.update(committed)
    return exempt


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=checkout, check=True, capture_output=True, text=True
    ).stdout


def available_port() -> int:
    """Reserve and release one currently available loopback TCP port."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
