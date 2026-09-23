"""Shared credential, identity and loopback helpers for installed-boundary gates."""

from __future__ import annotations

import json
import socket
import subprocess
from collections import Counter
from pathlib import Path

# The owner-selected qualification baseline (#67, #159): a reference, not a version ceiling.
HERMES_BASELINE = {"version": "0.21.0", "commit": "29112bef099274229cadff79cdff7bf7b99c4b77"}


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
    """Tracked paths a case-insensitive checkout cannot hold apart, so git sees them changed."""

    ignorecase = _git(checkout, "config", "--type=bool", "--default=false", "core.ignorecase")
    if ignorecase.strip() != "true":
        return set()
    tracked = [path for path in _git(checkout, "ls-files", "-z").split("\0") if path]
    spellings = Counter(path.casefold() for path in tracked)
    return {path for path in tracked if spellings[path.casefold()] > 1}


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=checkout, check=True, capture_output=True, text=True
    ).stdout


def available_port() -> int:
    """Reserve and release one currently available loopback TCP port."""

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])
